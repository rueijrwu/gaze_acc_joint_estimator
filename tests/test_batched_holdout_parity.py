"""Parity, exclusion, validity, and chunk-order contracts for raw batching."""
from __future__ import annotations

import numpy as np
import pytest

from full_position.accommodation import CANDIDATES, PowerResponseModel
from full_position.batched_holdout import predict_batch
from full_position.batched_inverse import solve_devices
from full_position.geometry import context
from full_position.information import predict_raw_holdout
from full_position.invert import STARTS
from full_position.model import PositionModel


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


def raw_fixture(model, n=3):
    states = np.array([[2.7+.1*i, 2.1+.05*i] for i in range(n)])
    rs = np.broadcast_to(CTX.r, (n, 3, 2)).copy()
    ps = np.broadcast_to(P1, (n, 3, 2)).copy()
    normalized = np.stack([model.predict(states[i], rs[i]) for i in range(n)]).reshape(n, 3, 2)
    qs = CTX.c+CTX.ell*normalized
    valid = np.ones((n, 3), dtype=bool)
    return ps, qs, valid, rs


def assert_same_prediction(actual, expected):
    assert actual["available"] == expected["available"]
    assert actual["reason"] == expected["reason"]
    if not actual["available"]:
        return
    assert actual["ambiguous"] == expected["ambiguous"]
    assert actual["rank"] == expected["rank"]
    assert actual["at_bound"] == expected["at_bound"]
    assert actual["start_count"] == expected["start_count"] == 49
    assert len(actual["candidates"]) == len(expected["candidates"]) == 49
    for a, b in zip(actual["candidates"], expected["candidates"]):
        assert a["start"] == b["start"]
        assert a["accepted"] == b["accepted"]
        np.testing.assert_allclose(a["state"], b["state"], rtol=2e-6, atol=3e-5)
        np.testing.assert_allclose(a["cost"], b["cost"], rtol=3e-6, atol=3e-7)
    np.testing.assert_allclose(actual["state"], expected["state"], rtol=2e-6, atol=3e-5)
    np.testing.assert_allclose(actual["cost"], expected["cost"], rtol=3e-6, atol=3e-7)


@pytest.mark.parametrize("exponent", CANDIDATES.values())
@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_mixed_held_batch_matches_raw_scalar_and_ignores_excluded_inputs(exponent, backend):
    if backend == "cupy":
        pytest.importorskip("cupy")
    model = model_for(exponent)
    p, q, valid, _ = raw_fixture(model)
    held = np.array([0, 1, 2])
    batched = predict_batch(model, p, q, valid, held, PILOT, REFERENCE, SIGMA,
                            backend=backend, starts=STARTS)
    scalar = [predict_raw_holdout(model, p[i], q[i], valid[i], int(held[i]), PILOT,
                                  REFERENCE, SIGMA, "xy", starts=STARTS)
              for i in range(len(held))]
    for a, b in zip(batched, scalar):
        assert_same_prediction(a, b)

    # Each scheduled held point's coordinates and validity are excluded from
    # the inverse. Replace them with nonfinite/extreme values and flip validity.
    changed_q, changed_valid = q.copy(), valid.copy()
    for i, point in enumerate(held):
        changed_q[i, point] = [np.nan, 1e12]
        changed_valid[i, point] = False
    changed = predict_batch(model, p, changed_q, changed_valid, held, PILOT,
                            REFERENCE, SIGMA, backend=backend, starts=STARTS)
    for a, b in zip(changed, scalar):
        assert_same_prediction(a, b)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_mixed_held_chunks_and_permutations_preserve_order(backend):
    if backend == "cupy":
        pytest.importorskip("cupy")
    model = model_for(.5)
    p, q, valid, _ = raw_fixture(model, 6)
    held = np.array([2, 0, 1, 2, 1, 0])
    expected = predict_batch(model, p, q, valid, held, PILOT, REFERENCE, SIGMA,
                             backend=backend, starts=STARTS)
    chunks = []
    for lo, hi in ((0, 1), (1, 4), (4, 6)):
        chunks.extend(predict_batch(model, p[lo:hi], q[lo:hi], valid[lo:hi], held[lo:hi],
                                    PILOT, REFERENCE, SIGMA, backend=backend, starts=STARTS))
    permutation = np.array([4, 1, 5, 0, 3, 2])
    permuted = predict_batch(model, p[permutation], q[permutation], valid[permutation],
                             held[permutation], PILOT, REFERENCE, SIGMA,
                             backend=backend, starts=STARTS)
    for i in range(len(expected)):
        assert_same_prediction(chunks[i], expected[i])
    for j, original in enumerate(permutation):
        assert_same_prediction(permuted[j], expected[original])


@pytest.mark.parametrize("backend", ["cupy", "torch"])
def test_two_device_uneven_and_single_device_fallback_parity(backend):
    if backend == "cupy":
        pytest.importorskip("cupy")
    else:
        pytest.importorskip("torch")
    model = model_for(.5)
    p, q, valid, _ = raw_fixture(model, 5)  # weighted split is intentionally uneven
    held = np.array([2, 0, 1, 2, 0])
    two = predict_batch(model, p, q, valid, held, PILOT, REFERENCE, SIGMA,
                        backend=backend, devices=[0, 1], starts=STARTS)
    scalar = [predict_raw_holdout(model, p[i], q[i], valid[i], int(held[i]), PILOT,
                                  REFERENCE, SIGMA, "xy", starts=STARTS)
              for i in range(5)]
    for actual, expected in zip(two, scalar):
        assert_same_prediction(actual, expected)

    # A one-row batch must take the explicit first-device fallback and preserve
    # that input's position in the output.
    one = predict_batch(model, p[:1], q[:1], valid[:1], held[:1], PILOT,
                        REFERENCE, SIGMA, backend=backend, devices=[0, 1], starts=STARTS)
    assert_same_prediction(one[0], scalar[0])

    # Held values and the held validity bit stay sealed after the split too.
    changed_q, changed_valid = q.copy(), valid.copy()
    for i, point in enumerate(held):
        changed_q[i, point] = [np.nan, -1e12]
        changed_valid[i, point] = False
    changed = predict_batch(model, p, changed_q, changed_valid, held, PILOT,
                            REFERENCE, SIGMA, backend=backend, devices=[0, 1], starts=STARTS)
    for actual, expected in zip(changed, scalar):
        assert_same_prediction(actual, expected)

    # All invalid device selectors reject before launching any device work.
    ctx = np.broadcast_to(CTX.r, (2, 3, 2))
    observations = np.stack([model.predict([2., 2.], CTX.r)]*2)
    covariance = np.broadcast_to(np.eye(6)*2e-4, (2, 6, 6)).copy()
    for devices in ([], [0, 0], [99]):
        with pytest.raises(ValueError):
            solve_devices(model, ctx, observations, covariance, np.arange(6),
                          devices=devices, backend=backend, starts=STARTS)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_invalid_geometry_retained_validity_and_nonfinite_retained_coordinates(backend):
    if backend == "cupy":
        pytest.importorskip("cupy")
    model = model_for(.5)
    p, q, valid, _ = raw_fixture(model)
    held = np.array([0, 1, 2])

    degenerate = p.copy()
    degenerate[0, 2] = degenerate[0, 0]+degenerate[0, 1]-degenerate[0, 0]
    out = predict_batch(model, degenerate, q, valid, held, PILOT, REFERENCE, SIGMA,
                        backend=backend)
    assert out[0]["reason"] == "invalid_P1_geometry"

    missing = valid.copy()
    missing[1, (held[1]+1) % 3] = False
    out = predict_batch(model, p, q, missing, held, PILOT, REFERENCE, SIGMA,
                        backend=backend)
    assert out[1]["reason"] == "insufficient_retained_P4"

    nonfinite = q.copy()
    nonfinite[2, (held[2]+1) % 3, 0] = np.nan
    out = predict_batch(model, p, nonfinite, valid, held, PILOT, REFERENCE, SIGMA,
                        backend=backend)
    assert out[2]["reason"] == "invalid_subset_input"


def test_raw_batch_rejects_bad_shapes_and_held_indices():
    model = model_for(.5)
    p, q, valid, _ = raw_fixture(model)
    with pytest.raises(ValueError, match="Raw batches"):
        predict_batch(model, p[:, :2], q, valid, np.array([0, 1, 2]),
                      PILOT, REFERENCE, SIGMA, backend="numpy")
    with pytest.raises(ValueError, match="Raw batches"):
        predict_batch(model, p, q, valid, np.array([0, 1, 3]),
                      PILOT, REFERENCE, SIGMA, backend="numpy")
