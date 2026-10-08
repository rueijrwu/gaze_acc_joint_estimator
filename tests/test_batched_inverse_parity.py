"""Focused parity checks for the all-start FP64 inverse backend."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from full_position.accommodation import CANDIDATES, PowerResponseModel
from full_position.batched_inverse import Objective, solve_batch
from full_position.geometry import context
from full_position.invert import STARTS, invert, objective_derivatives
from full_position.model import LOWER, UPPER, STATE_SCALE
from full_position.noise import whitening


P1 = np.array([[100., 60.], [180., 100.], [250., 50.]])
R = context(P1).r


def identified_model(exponent=.5):
    beta = np.random.default_rng(41).normal(scale=.08, size=27)
    beta[1], beta[9], beta[13], beta[25] = .03, .12, .2, -.17
    return PowerResponseModel(exponent, beta)


def covariance6():
    a = np.arange(36, dtype=float).reshape(6, 6)
    return (a @ a.T)*1e-8 + np.eye(6)*2e-4


@pytest.mark.parametrize("exponent", CANDIDATES.values())
@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("state", [[2.3, 2.1], [-20., 0.], [20., 6.]])
def test_objective_value_jacobian_hessian_matches_scalar(exponent, backend, state):
    model = identified_model(exponent)
    full_cov = covariance6()
    indices = np.array([0, 1, 2, 3])
    cov = full_cov[np.ix_(indices, indices)]
    W = whitening(cov)
    target = model.predict(np.array([3.1, 2.5]), R)[indices] + np.array([.002, -.003, .001, .004])
    xp = np if backend == "numpy" else pytest.importorskip("cupy")
    obj = Objective(model, xp.asarray(R[None]), xp.asarray(target[None]),
                    xp.asarray(W[None]), indices, xp)
    z = np.asarray(state, float)[None, :] / STATE_SCALE
    to_cpu = np.asarray if backend == "numpy" else xp.asnumpy
    cost, residual, jac, grad, hess = [to_cpu(x)[0] for x in obj.evaluate(xp.asarray(z), second=True)]
    expected_cost, expected_grad, expected_hess = objective_derivatives(model, R, target, W, indices, z[0])
    pred, pred_jac = model.predict(np.asarray(state), R, True)
    expected_residual = W @ (pred[indices]-target)
    expected_jac = W @ pred_jac[indices] @ np.diag(STATE_SCALE)
    np.testing.assert_allclose(cost, expected_cost, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(residual, expected_residual, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(jac, expected_jac, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(grad, expected_grad, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(hess, expected_hess, rtol=2e-10, atol=2e-10)


def _assert_inverse_parity(batch, scalar, *, atol=2e-5, rtol=2e-6):
    assert batch["available"] == scalar["available"]
    assert batch["ambiguous"] == scalar["ambiguous"] if scalar["available"] else True
    assert batch["start_count"] == scalar["start_count"] == len(STARTS)
    for a, b in zip(batch["candidates"], scalar["candidates"]):
        assert a["start"] == b["start"]
        assert a["accepted"] == b["accepted"]
        np.testing.assert_allclose(a["state"], b["state"], rtol=rtol, atol=atol)
        np.testing.assert_allclose(a["cost"], b["cost"], rtol=rtol, atol=atol)
    if scalar["available"]:
        np.testing.assert_allclose(batch["state"], scalar["state"], rtol=rtol, atol=atol)
        np.testing.assert_allclose(batch["cost"], scalar["cost"], rtol=rtol, atol=atol)
        assert batch["rank"] == scalar["rank"]
        assert batch["at_bound"] == scalar["at_bound"]


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_all_49_starts_identified_synthetic_and_bound_case(backend):
    if backend == "cupy":
        pytest.importorskip("cupy")
    model = identified_model(.5)
    cov = covariance6()
    for truth in (np.array([3.4, 2.6]), np.array([22., 7.])):
        y = model.predict(truth, R)
        batched = solve_batch(model, R[None], y[None], cov[None], backend=backend)[0]
        scalar = invert(model, R, y, cov, starts=STARTS)
        _assert_inverse_parity(batched, scalar)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_weak_ambiguous_case_matches_all_scalar_starts(backend):
    if backend == "cupy":
        pytest.importorskip("cupy")
    beta = np.zeros(27)
    beta[4] = .4       # even theta dependence: +/- branches
    beta[9] = .15      # accommodation signal in the second image axis
    model = PowerResponseModel(.5, beta)
    cov = covariance6()
    y = model.predict([5., 2.], R)
    batched = solve_batch(model, R[None], y[None], cov[None], backend=backend)[0]
    scalar = invert(model, R, y, cov, starts=STARTS)
    _assert_inverse_parity(batched, scalar)
    assert scalar["available"] and scalar["ambiguous"]


@pytest.mark.parametrize("name,exponent", [("ar27_log", 0.), ("ar27_sqrt", .5), ("ar27_quadratic", 2.)])
@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_saved_real_model_retained_input_matches_all_scalar_starts(name, exponent, backend):
    if backend == "cupy":
        pytest.importorskip("cupy")
    root = Path("experiments/full_position/accommodation_full_v2_gpu")
    artifact = json.loads((root/"fits/full_calibration"/name/"model.json").read_text())
    model = PowerResponseModel(exponent, artifact["coefficients"])
    with np.load(root/"splits/full_calibration/training_inputs.npz") as data:
        # Deterministic retained training rows provide real contexts, y, and
        # covariance without consulting any held-out coordinate or label.
        row = {"ar27_log": 101, "ar27_sqrt": 1001, "ar27_quadratic": 10001}[name]
        r, y, cov = data["r"][row], data["v"][row], data["covariance"][row]
    batched = solve_batch(model, r[None], y[None], cov[None], backend=backend)[0]
    scalar = invert(model, r, y, cov, starts=STARTS)
    _assert_inverse_parity(batched, scalar, atol=8e-5, rtol=8e-6)


@pytest.mark.parametrize("bad", ["shape", "nonfinite", "bounds", "empty_starts"])
def test_solver_rejects_invalid_shapes_nonfinite_and_bounds(bad):
    model = identified_model()
    r, y, cov = R[None], model.predict([2., 2.], R)[None], covariance6()[None]
    starts = STARTS
    if bad == "shape":
        r = np.zeros((1, 2, 2))
    elif bad == "nonfinite":
        y = y.copy(); y[0, 0] = np.nan
    elif bad == "bounds":
        starts = np.array([[20.1, 2.]])
    else:
        starts = np.empty((0, 2))
    with pytest.raises(ValueError):
        solve_batch(model, r, y, cov, starts=starts, backend="numpy")
