"""Contracts for optional NumPy/CuPy products of the unchanged fit Jacobian."""
import numpy as np
import pytest

from full_position.accommodation import CANDIDATES, PowerResponseModel
from full_position.calibrate import ProfiledProblem
from full_position.linear_acceleration import ORIGINAL_JAC, contracted_jac
from full_position.model import STATE_SCALE


def build_problem(exponent):
    rng = np.random.default_rng(20261007)
    groups = np.repeat(np.arange(20), 6)
    # Each group has six time-varying samples around a declared nominal state.
    anchors = np.column_stack((np.linspace(-10, 10, 20),
                               np.linspace(0.5, 4.5, 20)))
    states = anchors[groups] + rng.normal(size=(len(groups), 2)) * [0.15, 0.08]
    angles = np.arange(3) * (2*np.pi/3)
    r0 = np.column_stack((np.cos(angles), np.sin(angles))) * 0.5
    r = np.broadcast_to(r0, (len(groups), 3, 2)).copy()
    model = PowerResponseModel(exponent, rng.normal(scale=0.05, size=27))
    y = model.predict(states, r)
    cov = np.broadcast_to(np.eye(6)*0.03, (len(groups), 6, 6)).copy()
    problem = ProfiledProblem(model, y, r, cov, groups, anchors)
    z = states/STATE_SCALE
    problem.update(z.ravel())
    return problem, z.ravel(), rng


def relative_error(actual, expected):
    return np.linalg.norm(actual-expected) / max(np.linalg.norm(expected), 1e-30)


@pytest.mark.parametrize("exponent", CANDIDATES.values())
@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_accelerated_products_match_original_and_preserve_captured_jacobian(exponent, backend):
    if backend == "cupy":
        cp = pytest.importorskip("cupy")
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CuPy installed without a CUDA device")
    problem, z, rng = build_problem(exponent)
    old = ORIGINAL_JAC(problem, z)
    new = contracted_jac(problem, z, backend)
    vector = rng.normal(size=old.shape[1])
    cotangent = rng.normal(size=old.shape[0])
    old_jvp, old_vjp = old.matvec(vector), old.rmatvec(cotangent)
    new_jvp, new_vjp = new.matvec(vector), new.rmatvec(cotangent)
    assert relative_error(new_jvp, old_jvp) <= 2e-9
    assert relative_error(new_vjp, old_vjp) <= 2e-9
    assert abs(float(new_jvp@cotangent-vector@new_vjp)) <= 2e-9*(
        1+abs(float(new_jvp@cotangent)))

    # The original operator must retain its captured linearization after the
    # problem cache moves to a different state.
    problem.update(z + rng.normal(size=z.shape)*1e-3)
    np.testing.assert_array_equal(old.matvec(vector), old_jvp)
    np.testing.assert_array_equal(new.matvec(vector), new_jvp)
