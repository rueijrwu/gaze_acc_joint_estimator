"""PyTorch FP64 parity for the explicit batched inverse backend."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from full_position.accommodation import CANDIDATES, PowerResponseModel
from full_position.batched_holdout import predict_batch
from full_position.batched_inverse import solve_batch
from full_position.geometry import context
from full_position.information import predict_raw_holdout
from full_position.invert import STARTS, invert
from full_position.model import PositionModel
from full_position.torch_arrays import TorchArrays


P1 = np.array([[100., 60.], [180., 100.], [250., 50.]])
CTX = context(P1)
SIGMA = np.eye(12)*4e-4 + np.ones((12, 12))*1e-6
PILOT_BETA = np.random.default_rng(512).normal(scale=.05, size=27)
PILOT_BETA[[1, 9, 13, 25]] = [.02, .1, .16, -.11]
PILOT = PositionModel(27, PILOT_BETA)
REFERENCE = np.array([1.5, 2.])


def model_for(exponent):
    beta = np.random.default_rng(41).normal(scale=.08, size=27)
    beta[1], beta[9], beta[13], beta[25] = .03, .12, .2, -.17
    return PowerResponseModel(exponent, beta)


def fixture(model, n=3):
    states = np.array([[2.7+.1*i, 2.1+.05*i] for i in range(n)])
    p = np.broadcast_to(P1, (n, 3, 2)).copy()
    r = np.broadcast_to(CTX.r, (n, 3, 2)).copy()
    values = np.stack([model.predict(states[i], r[i]) for i in range(n)]).reshape(n, 3, 2)
    q = CTX.c+CTX.ell*values
    valid = np.ones((n, 3), dtype=bool)
    return p, q, valid, r


def compare(actual, expected):
    assert actual["available"] == expected["available"]
    assert actual["reason"] == expected["reason"]
    if not actual["available"]:
        return
    assert actual["ambiguous"] == expected["ambiguous"]
    assert actual["rank"] == expected["rank"]
    assert actual["at_bound"] == expected["at_bound"]
    assert len(actual["candidates"]) == len(expected["candidates"]) == 49
    for a, b in zip(actual["candidates"], expected["candidates"]):
        assert a["accepted"] == b["accepted"]
        np.testing.assert_allclose(a["state"], b["state"], rtol=3e-6, atol=4e-5)
        np.testing.assert_allclose(a["cost"], b["cost"], rtol=5e-6, atol=5e-7)
        assert a["polish"]["certified"] == b["polish"]["certified"]
    np.testing.assert_allclose(actual["state"], expected["state"], rtol=3e-6, atol=4e-5)
    np.testing.assert_allclose(actual["cost"], expected["cost"], rtol=5e-6, atol=5e-7)


def retained_cases():
    root = Path("experiments/full_position/accommodation_full_v2_gpu")
    with np.load(root/"splits/full_calibration/training_inputs.npz") as data:
        for name, exponent, row in (("ar27_log", 0., 201),
                                    ("ar27_sqrt", .5, 1201),
                                    ("ar27_quadratic", 2., 12001)):
            artifact = json.loads((root/"fits/full_calibration"/name/"model.json").read_text())
            model = PowerResponseModel(exponent, artifact["coefficients"])
            yield model, data["r"][row].copy(), data["v"][row].copy(), data["covariance"][row].copy()


def test_torch_all49_synthetic_and_saved_model_final_certificates():
    assert torch.cuda.is_available()
    assert TorchArrays(0).asarray([1.]).dtype == torch.float64
    cases = [(model_for(.5), CTX.r, None, None)]
    cases.extend(retained_cases())
    for model, r, y, cov in cases:
        if y is None:
            cov = np.eye(6)*2e-4 + np.ones((6, 6))*1e-6
            y = model.predict([3.4, 2.6], r)
        torch_result = solve_batch(model, r[None], y[None], cov[None],
                                   backend="torch", device=0, starts=STARTS)[0]
        scalar = invert(model, r, y, cov, starts=STARTS)
        compare(torch_result, scalar)


@pytest.mark.parametrize("exponent", CANDIDATES.values())
def test_torch_raw_mixedheld_and_excluded_mutation(exponent):
    model = model_for(exponent)
    p, q, valid, _ = fixture(model)
    held = np.array([2, 0, 1])
    batch = predict_batch(model, p, q, valid, held, PILOT, REFERENCE, SIGMA,
                          backend="torch", device=0, starts=STARTS)
    scalar = [predict_raw_holdout(model, p[i], q[i], valid[i], int(held[i]), PILOT,
                                  REFERENCE, SIGMA, "xy", starts=STARTS)
              for i in range(3)]
    for a, b in zip(batch, scalar):
        compare(a, b)
    changed_q, changed_valid = q.copy(), valid.copy()
    for i, point in enumerate(held):
        changed_q[i, point] = [np.nan, 1e10]
        changed_valid[i, point] = False
    changed = predict_batch(model, p, changed_q, changed_valid, held, PILOT, REFERENCE,
                            SIGMA, backend="torch", device=0, starts=STARTS)
    for a, b in zip(changed, scalar):
        compare(a, b)
