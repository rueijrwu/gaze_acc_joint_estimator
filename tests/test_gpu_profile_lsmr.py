"""Focused parity checks for device profile and Krylov paths."""
import numpy as np
import pytest
from scipy.optimize._lsq.common import regularized_lsq_operator, right_multiplied_operator
from scipy.optimize._lsq.trf import lsmr as scipy_lsmr
from scipy.sparse.linalg import LinearOperator

from full_position.accommodation import CANDIDATES, PowerResponseModel
from full_position.calibrate import ProfiledProblem, fit
from full_position.device_lsmr import (lsmr as device_lsmr, regularized as device_regularized,
                                       right as device_right)
from full_position.gpu_profile import HostView, ORIGINAL_UPDATE, update as gpu_update
from full_position.model import STATE_SCALE


def cupy_or_skip():
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CuPy installed without a CUDA device")
    return cp


def tiny_problem(exponent, seed=1287, n_groups=8, rows_per_group=4):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    anchors = np.column_stack((np.linspace(-5, 5, n_groups),
                               np.linspace(0.8, 3.8, n_groups)))
    states = anchors[groups] + rng.normal(size=(len(groups), 2)) * [0.3, 0.15]
    angles = np.arange(3) * (2*np.pi/3)
    r0 = np.column_stack((np.cos(angles), np.sin(angles))) * 0.4
    r = np.broadcast_to(r0, (len(groups), 3, 2)).copy()
    beta = rng.normal(scale=0.03, size=27)
    model = PowerResponseModel(exponent, beta)
    y = model.predict(states, r) + rng.normal(scale=0.004, size=(len(groups), 6))
    cov = np.broadcast_to(np.eye(6)*0.02, (len(groups), 6, 6)).copy()
    return model, y, r, cov, groups, anchors, states


def matrix(problem, name):
    return np.asarray(getattr(problem, name))


def test_host_verification_cache_rejects_mutation():
    cp = cupy_or_skip()
    source = cp.arange(6, dtype=cp.float64).reshape(2, 3)
    view = HostView(source, cp)
    cached = np.asarray(view)
    assert not cached.flags.writeable
    with pytest.raises(ValueError):
        cached[0, 0] = -1
    sliced = view[:, :]
    assert not sliced.flags.writeable
    with pytest.raises(ValueError):
        sliced[0, 0] = -1
    np.testing.assert_array_equal(cp.asnumpy(source), np.arange(6).reshape(2, 3))


@pytest.mark.parametrize("exponent", CANDIDATES.values())
def test_gpu_profile_matches_cpu_and_captured_operator(exponent):
    cp = cupy_or_skip()
    model, y, r, cov, groups, anchors, states = tiny_problem(exponent)
    cpu = ProfiledProblem(model, y, r, cov, groups, anchors)
    gpu = ProfiledProblem(model, y, r, cov, groups, anchors)
    z = (states / STATE_SCALE).ravel()
    ORIGINAL_UPDATE(cpu, z)
    gpu_update(gpu, z)

    for name in ("beta", "optical_residual", "S"):
        np.testing.assert_allclose(matrix(gpu, name), matrix(cpu, name), rtol=3e-10, atol=3e-11)
    for name in ("B", "C"):
        np.testing.assert_allclose(matrix(gpu, name), matrix(cpu, name), rtol=3e-10, atol=3e-11)
    # QR signs are arbitrary; compare the represented factorization and |diag|.
    np.testing.assert_allclose(np.abs(np.diag(gpu.R)), np.abs(np.diag(cpu.R)), rtol=3e-10, atol=3e-11)
    assert gpu.Q is None  # GPU path retains Q only on device, not as a host array.
    signs = np.sign(np.diag(gpu.R) * np.diag(cpu.R))
    np.testing.assert_allclose(gpu.R * signs[:, None], cpu.R, rtol=3e-10, atol=3e-11)
    np.testing.assert_allclose(gpu.residual, cpu.residual, rtol=3e-10, atol=3e-11)

    old = cpu.jac(z)
    from full_position.linear_acceleration import contracted_jac
    new = contracted_jac(gpu, z, "cupy")
    rng = np.random.default_rng(903)
    v, w = rng.normal(size=old.shape[1]), rng.normal(size=old.shape[0])
    old_jv, old_jtw = old.matvec(v), old.rmatvec(w)
    np.testing.assert_allclose(new.matvec(v), old_jv, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(new.rmatvec(w), old_jtw, rtol=2e-8, atol=2e-9)
    # A rejected trial must not mutate either captured linearization.
    ORIGINAL_UPDATE(cpu, z + rng.normal(scale=2e-4, size=z.shape))
    new_jv = new.matvec(v)
    gpu_update(gpu, z + rng.normal(scale=2e-4, size=z.shape))
    np.testing.assert_array_equal(old.matvec(v), old_jv)
    np.testing.assert_array_equal(new.matvec(v), new_jv)


def test_device_lsmr_matches_scipy_on_scaled_regularized_operators():
    cp = cupy_or_skip()
    rng = np.random.default_rng(701)
    dense = rng.normal(size=(23, 9))
    dense[:, -1] *= 1e-3
    base = LinearOperator(dense.shape, matvec=lambda x: dense @ x,
                          rmatvec=lambda y: dense.T @ y, dtype=float)
    device = LinearOperator(dense.shape,
        matvec=lambda x: cp.asnumpy(cp.asarray(dense) @ cp.asarray(x)),
        rmatvec=lambda y: cp.asnumpy(cp.asarray(dense).T @ cp.asarray(y)), dtype=float)
    device.device_matvec = lambda x: cp.asarray(dense) @ x
    device.device_rmatvec = lambda y: cp.asarray(dense).T @ y
    device.array_backend = cp
    b = np.r_[rng.normal(size=23), np.zeros(9)]
    d = np.geomspace(0.2, 2.0, 9)
    x0 = rng.normal(size=9)
    for damp, initial in ((0., None), (0.3, None), (0., x0), (0.3, x0)):
        A = regularized_lsq_operator(right_multiplied_operator(base, d), np.full(9, 0.15))
        G = device_regularized(device_right(device, d), cp.full(9, 0.15))
        # With stopping tolerances disabled, both implementations execute the
        # same fixed Krylov budget, avoiding threshold-sensitive near-breakdown.
        expected = scipy_lsmr(A, b, damp=damp, atol=0., btol=0., conlim=0.,
                              maxiter=5, x0=initial)
        actual = device_lsmr(G, b, damp=damp, atol=0., btol=0., conlim=0.,
                             maxiter=5, x0=initial)
        np.testing.assert_allclose(actual[0], expected[0], rtol=2e-8, atol=2e-9)
        assert actual[1:3] == expected[1:3]
        np.testing.assert_allclose(actual[3:], expected[3:], rtol=2e-8, atol=2e-9)

    # In a converged solve, compare the answer and true residual/gradient;
    # recurrence condition/norm estimates can vary after near-breakdown.
    A = regularized_lsq_operator(right_multiplied_operator(base, d), np.full(9, 0.15))
    G = device_regularized(device_right(device, d), cp.full(9, 0.15))
    expected = scipy_lsmr(A, b, damp=0.3, atol=1e-10, btol=1e-10, maxiter=40, x0=x0)
    actual = device_lsmr(G, b, damp=0.3, atol=1e-10, btol=1e-10, maxiter=40, x0=x0)
    np.testing.assert_allclose(actual[0], expected[0], rtol=2e-7, atol=2e-8)
    assert actual[1] in (1, 2) and expected[1] in (1, 2)
    residual_cpu = A.matvec(expected[0]) - b
    residual_gpu = G.matvec(actual[0]) - b
    np.testing.assert_allclose(np.linalg.norm(residual_gpu), np.linalg.norm(residual_cpu),
                               rtol=2e-8, atol=2e-9)
    gradient_cpu = A.rmatvec(residual_cpu) + 0.3**2 * (expected[0] - x0)
    gradient_gpu = G.rmatvec(residual_gpu) + 0.3**2 * (actual[0] - x0)
    assert np.linalg.norm(gradient_cpu) < 1e-7
    assert np.linalg.norm(gradient_gpu) < 1e-7

    zero = LinearOperator((5, 3), matvec=lambda x: np.zeros(5),
                          rmatvec=lambda y: np.zeros(3), dtype=float)
    zero.device_matvec = lambda x: cp.zeros(5)
    zero.device_rmatvec = lambda y: cp.zeros(3)
    zero.array_backend = cp
    result = device_lsmr(zero, np.zeros(5), maxiter=12)
    np.testing.assert_array_equal(result[0], np.zeros(3))
    assert result[1:3] == (0, 0)


@pytest.mark.parametrize("exponent", CANDIDATES.values())
def test_full_fit_gpu_converges_and_respects_low_budget(exponent):
    cupy_or_skip()
    from full_position import linear_acceleration
    model, y, r, cov, groups, anchors, _ = tiny_problem(exponent, seed=320+int(exponent*10),
                                                        n_groups=8, rows_per_group=4)
    kwargs = dict(prior_strength=.01, anchor_scales=(.1, .25), starts=1,
                  seed=12, continuation_stages=1)
    try:
        linear_acceleration.enable("cupy", 0)
        _, _, good = fit(PowerResponseModel(exponent), y, r, cov, groups, anchors,
                         max_nfev=100, **kwargs)
        assert good["converged"] is True
        assert good["alternatives"][0]["projected_stationarity_physical"] < 1e-3
        assert good["alternatives"][0]["inner_stationarity_scaled"] < 1e-7

        _, _, short = fit(PowerResponseModel(exponent), y, r, cov, groups, anchors,
                          max_nfev=1, **kwargs)
        assert short["converged"] is False
        assert short["alternatives"][0]["nfev"] <= 1
        assert np.isfinite(short["alternatives"][0]["projected_stationarity_physical"])
    finally:
        linear_acceleration.disable()
