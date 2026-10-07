"""Numerical and solver contracts for the fixed accommodation response laws."""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.optimize._numdiff import approx_derivative
from copy import deepcopy

from full_position.accommodation import (
    CANDIDATES, PowerResponseModel, bases, linear_accommodation, rescale, response,
)
from full_position.accommodation_schema import artifact as power_artifact, from_object, load_model
from full_position.accommodation_selection import declared_policy, choose as choose_power, nested_grouped
from full_position.accelerated import forward as accelerated_forward, solve_batch
from full_position.calibrate import ProfiledProblem
from full_position.crosscheck import _triple_metrics
from full_position.geometry import context
from full_position.invert import invert, predict_holdout
from full_position.model import PositionModel, STATE_SCALE
from full_position.noise import marginal
from full_position.population import FIELDS, manifest
from full_position.selection import transfer_splits
from full_position.schema import artifact as legacy_artifact


P1 = np.array([[100., 60.], [180., 100.], [250., 50.]])
CTX = context(P1)


def coefficients(seed=41):
    rng = np.random.default_rng(seed)
    beta = rng.normal(scale=.08, size=27)
    # Give the two state directions clear optical signal without making the
    # synthetic inverse poorly scaled.
    beta[1] = .03
    beta[9] = .12
    beta[13] = .2
    beta[25] = -.17
    return beta


@pytest.mark.parametrize("exponent", CANDIDATES.values())
def test_response_values_derivatives_and_hessians(exponent):
    # Include the physical boundary and an interior point; all derivatives are
    # expressed in degrees and diopters.
    x = np.array([[0., 0.], [-13., .7], [20., 6.]])
    d, dd, s, ds, dh, sh = bases(x, exponent)
    phi, pa, paa = response(x[:, 1], exponent)
    if exponent == 0.:
        expected = np.log1p(x[:, 1])
    else:
        expected = np.expm1(exponent*np.log1p(x[:, 1]))/exponent
    np.testing.assert_allclose(phi, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(pa, (1+x[:, 1])**(exponent-1), rtol=1e-13)
    np.testing.assert_allclose(paa, (exponent-1)*(1+x[:, 1])**(exponent-2), rtol=1e-13, atol=1e-14)
    assert np.isfinite(np.concatenate([d.ravel(), dd.ravel(), s.ravel(), ds.ravel(), dh.ravel(), sh.ravel()])).all()

    # Full basis Jacobians and Hessians must agree with independent numerical
    # differentiation, including the zero-gaze / zero-accommodation corner.
    # The derivative at A=0 is checked against the closed form above; centered
    # numerical differences there would leave the declared physical domain.
    for row in x[1:]:
        values = lambda z: np.r_[bases(z, exponent)[0], bases(z, exponent)[2]]
        numerical = approx_derivative(values, row, method="3-point")
        analytic = np.concatenate((dd[np.where((x == row).all(axis=1))[0][0]],
                                   ds[np.where((x == row).all(axis=1))[0][0]]), axis=0)
        np.testing.assert_allclose(analytic, numerical, atol=2e-8, rtol=2e-7)
        for j in range(7):
            hfun = lambda z: bases(z, exponent)[0][j]
            numeric_h = approx_derivative(lambda z: approx_derivative(hfun, z, method="3-point"), row,
                                          method="3-point")
            np.testing.assert_allclose(dh[np.where((x == row).all(axis=1))[0][0], j],
                                       numeric_h, atol=2e-5, rtol=2e-4)
        for j in range(4):
            hfun = lambda z: bases(z, exponent)[2][j]
            numeric_h = approx_derivative(lambda z: approx_derivative(hfun, z, method="3-point"), row,
                                          method="3-point")
            np.testing.assert_allclose(sh[np.where((x == row).all(axis=1))[0][0], j],
                                       numeric_h, atol=2e-5, rtol=2e-4)


def test_logarithmic_limit_invalid_domain_and_no_clamping():
    A = np.array([0., 1e-7, .7, 6.])
    phi, pa, paa = response(A, 0.)
    np.testing.assert_allclose(phi, np.log1p(A), atol=0., rtol=0.)
    for exponent in (1e-4, 1e-5):
        np.testing.assert_allclose(response(A, exponent)[0],
                                   np.log1p(A)+.5*exponent*np.log1p(A)**2,
                                   atol=2e-8, rtol=2e-7)
    with pytest.raises(ValueError, match="finite A"):
        response(np.array([-1., 0.]), .5)
    with pytest.raises(ValueError, match="finite A"):
        response(np.array([np.nan]), 1.)
    with pytest.raises(ValueError, match="finite real"):
        response(1., np.nan)
    with pytest.raises(ValueError, match=r"finite \(\.\.\.,2\)"):
        bases([0., np.inf], .5)


@pytest.mark.parametrize("exponent", CANDIDATES.values())
def test_model_prediction_jacobian_hessian_and_27_coefficient_contract(exponent):
    model = PowerResponseModel(exponent, coefficients())
    assert model.size == model.capacity == 27
    assert model.channels == 6
    assert model.width == 4
    np.testing.assert_array_equal(model.intercepts, [0, 7, 11, 15, 19, 23])
    assert model.response_definition.exponent == exponent
    assert model.response_definition.capacity == 27
    x = np.array([3.2, 2.4])
    value, jac = model.predict(x, CTX.r, True)
    numeric_jac = approx_derivative(lambda z: model.predict(z, CTX.r), x, method="3-point")
    np.testing.assert_allclose(jac, numeric_jac, atol=2e-8, rtol=2e-7)
    hessian = model.state_hessian(x, CTX.r)
    for channel in range(6):
        numeric_h = approx_derivative(
            lambda z: model.predict(z, CTX.r, True)[1][channel], x, method="3-point")
        np.testing.assert_allclose(hessian[channel], numeric_h, atol=3e-8, rtol=3e-7)


def test_power_response_is_explicitly_unsupported_by_legacy_accelerators():
    power = PowerResponseModel(.5, coefficients(16))
    state = np.array([2., 1.5])
    y = power.predict(state, CTX.r)
    with pytest.raises(ValueError, match="verified only for legacy log"):
        solve_batch(power, CTX.r[None, ...], y[None, :], np.eye(6)[None, ...],
                    starts=np.array([state]))
    with pytest.raises(ValueError, match="has not passed parity validation"):
        accelerated_forward(power.beta, 27, state, CTX.r, response_family="conditional_power_response_v1")

    # Existing default and explicit legacy dispatch stay numerically identical.
    legacy = PositionModel(27, coefficients(16))
    expected, expected_jac = legacy.predict(state, CTX.r, True)
    value, jac = accelerated_forward(legacy.beta, 27, state, CTX.r)
    value_explicit, jac_explicit = accelerated_forward(
        legacy.beta, 27, state, CTX.r, response_family="legacy_log")
    np.testing.assert_allclose(value, expected, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(jac, expected_jac, atol=2e-14, rtol=2e-14)
    np.testing.assert_array_equal(value, value_explicit)
    np.testing.assert_array_equal(jac, jac_explicit)


def test_log_model_matches_conditional27_values_and_derivatives():
    beta = coefficients(4)
    old = PositionModel(27, beta.copy())
    log = PowerResponseModel(0., beta.copy())
    states = np.array([[0., 0.], [-20., 0.], [20., 6.], [4.3, 2.1]])
    for x in states:
        old_y, old_j = old.predict(x, CTX.r, True)
        new_y, new_j = log.predict(x, CTX.r, True)
        np.testing.assert_allclose(new_y, old_y, atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(new_j, old_j, atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(log.state_hessian(x, CTX.r), old.state_hessian(x, CTX.r),
                                   atol=2e-14, rtol=2e-14)


def test_log_power_inverse_matches_legacy_conditional27_with_fixed_covariance():
    beta = coefficients(4)
    legacy = PositionModel(27, beta.copy())
    power_log = PowerResponseModel(0., beta.copy())
    state = np.array([3.4, 2.6])
    observed = legacy.predict(state, CTX.r)
    covariance = np.eye(6)*1e-4+np.ones((6, 6))*1e-5
    starts = np.array([state, [-8., .8], [8., 4.5]])
    old_result = invert(legacy, CTX.r, observed, covariance, starts=starts, max_nfev=80)
    new_result = invert(power_log, CTX.r, observed, covariance, starts=starts, max_nfev=80)
    assert old_result["available"] and new_result["available"]
    np.testing.assert_allclose(new_result["state"], old_result["state"], atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(new_result["cost"], old_result["cost"], atol=2e-14, rtol=2e-14)
    assert new_result["rank"] == old_result["rank"]
    assert new_result["ambiguous"] == old_result["ambiguous"]
    np.testing.assert_allclose([b["state"] for b in new_result["branches"]],
                               [b["state"] for b in old_result["branches"]], atol=2e-12)


@pytest.mark.parametrize("exponent", CANDIDATES.values())
def test_profile_jvp_vjp_with_nonzero_optical_residual_and_anchor_terms(exponent):
    rng = np.random.default_rng(66)
    model = PowerResponseModel(exponent, coefficients(23))
    groups = np.repeat(np.arange(4), 3)
    anchors = np.array([[-7., .6], [-2., 2.], [3., 3.2], [8., 4.5]])
    states = anchors[groups]+rng.normal(scale=[.15, .12], size=(len(groups), 2))
    r = np.broadcast_to(CTX.r, (len(groups), 3, 2))
    y = model.predict(states, r)+rng.normal(scale=2e-3, size=(len(groups), 6))
    cov = np.broadcast_to(np.eye(6)*2e-4, (len(groups), 6, 6))
    problem = ProfiledProblem(model, y, r, cov, groups, anchors, prior_strength=.002,
                              anchor_scales=(.2, .4))
    z = (states/STATE_SCALE).ravel()
    residual = problem.fun(z)
    assert np.linalg.norm(residual[:len(groups)*6]) > 0
    assert np.linalg.norm(residual[len(groups)*6:]) > 0
    direction = rng.normal(size=z.size)
    analytic = problem.jvp(direction)
    numeric = (problem.fun(z+1e-6*direction)-problem.fun(z-1e-6*direction))/(2e-6)
    np.testing.assert_allclose(analytic, numeric, atol=3e-5, rtol=4e-4)
    problem.update(z)
    cotangent = rng.normal(size=len(problem.residual))
    np.testing.assert_allclose(problem.jvp(direction)@cotangent,
                               direction@problem.vjp(cotangent), atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("exponent", CANDIDATES.values())
def test_fixed_covariance_inverse_and_holdout_noninterference(exponent):
    model = PowerResponseModel(exponent, coefficients(17))
    state = np.array([2.3, 2.1])
    v = model.predict(state, CTX.r).reshape(3, 2)
    full_cov = np.eye(6)*1e-4 + np.ones((6, 6))*1e-5
    # An exact state start keeps this a small API/numerical contract check.
    starts = np.array([state, [-8., .8], [8., 4.5]])
    result = invert(model, CTX.r, v.ravel(), full_cov, starts=starts, max_nfev=80)
    assert result["available"]
    np.testing.assert_allclose(result["state"], state, atol=2e-5)
    for held in range(3):
        retained = np.array([i for i in range(6) if i//2 != held])
        first = predict_holdout(model, CTX, held, v.ravel()[retained], full_cov,
                                starts=starts)
        changed = v.copy()
        changed[held] += [1e6, -8e5]
        second = predict_holdout(model, CTX, held, changed.ravel()[retained], full_cov,
                                 starts=starts)
        assert first["available"] and second["available"]
        assert first["state"] == second["state"]
        assert first["branches"] == second["branches"]
        assert first["predictions"] == second["predictions"]
        kept = np.array([i for i in range(6) if i//2 != held])
        direct = invert(model, CTX.r, v.ravel()[retained], marginal(full_cov, kept), kept,
                        starts=starts, max_nfev=80)
        np.testing.assert_allclose(first["state"], direct["state"], atol=1e-12)


@pytest.mark.parametrize("exponent", CANDIDATES.values())
def test_shifted_power_state_gauge_preserves_predictions_and_transforms_jacobian(exponent):
    model = PowerResponseModel(exponent, coefficients(51))
    states = np.array([[-7., .2], [-2., 1.2], [3., 3.], [9., 5.]])
    g, h = 1.13, .91
    changed, transformed = rescale(model, states, g, h)
    for old_x, new_x in zip(states, transformed):
        before, before_j = model.predict(old_x, CTX.r, True)
        after, after_j = changed.predict(new_x, CTX.r, True)
        np.testing.assert_allclose(after, before, atol=3e-13, rtol=3e-13)
        before_j = before_j.copy()
        before_j[:, 0] /= g
        before_j[:, 1] /= h
        np.testing.assert_allclose(after_j, before_j, atol=3e-12, rtol=3e-12)


def test_linear_conditional_accommodation_against_bounded_numeric_profile():
    model = PowerResponseModel(1., coefficients(27))
    theta = 4.2
    full = model.predict([theta, 0.], CTX.r)
    slope = model.predict([theta, 1.], CTX.r)-full
    indices = np.array([0, 1, 4, 5])
    covariance = np.array([[3., .2, .1, 0.], [.2, 2., 0., .1],
                           [.1, 0., 2.5, .3], [0., .1, .3, 1.8]])*1e-4
    for true_A in (0., 2.7, 6.):
        observed = (full+true_A*slope)[indices]
        analytic = linear_accommodation(model, CTX.r, observed, covariance, indices, theta)
        numeric = minimize_scalar(
            lambda A: (observed-model.predict([theta, A], CTX.r)[indices]) @
                      np.linalg.solve(covariance,
                        observed-model.predict([theta, A], CTX.r)[indices]),
            bounds=(0., 6.), method="bounded", options={"xatol": 1e-12})
        assert analytic["observable"]
        assert analytic["A_D"] == pytest.approx(np.clip(numeric.x, 0., 6.), abs=2e-5)
        assert analytic["A_D"] == pytest.approx(true_A, abs=1e-10)

    no_accommodation = model.beta.copy()
    no_accommodation[[1, 3, 5, 9]] = 0.
    for k in range(5):
        no_accommodation[7+4*k+2] = 0.
        no_accommodation[7+4*k+3] = 0.
    unobservable = PowerResponseModel(1., no_accommodation)
    out = linear_accommodation(unobservable, CTX.r, full[indices], covariance, indices, theta)
    assert not out["observable"]
    assert out["A_D"] is None


def test_moving_within_fixation_states_remain_framewise_and_anchor_is_mean_only():
    model = PowerResponseModel(.5, coefficients(72))
    groups = np.repeat(np.arange(2), 4)
    anchors = np.array([[-5., 1.5], [5., 3.]])
    zero_mean_delta = np.array([[-.3, -.2], [.1, .15], [.25, -.1], [-.05, .15],
                                [.2, .1], [-.1, -.25], [-.25, .05], [.15, .1]])
    states = anchors[groups]+zero_mean_delta
    np.testing.assert_allclose(zero_mean_delta.reshape(2, 4, 2).mean(axis=1), 0., atol=1e-15)
    r = np.broadcast_to(CTX.r, (len(groups), 3, 2))
    y = model.predict(states, r)
    cov = np.broadcast_to(np.eye(6)*1e-3, (len(groups), 6, 6))
    problem = ProfiledProblem(model, y, r, cov, groups, anchors, prior_strength=0.,
                              anchor_scales=(.3, .5))
    z = (states/STATE_SCALE).ravel()
    residual = problem.fun(z)
    anchor_start = len(groups)*6
    np.testing.assert_allclose(residual[anchor_start:], 0., atol=1e-14)
    # A within-fixation zero-mean trajectory perturbation has no anchor cost.
    changed = states.copy()
    changed[0, 0] += .2
    changed[1, 0] -= .2
    changed[4, 1] += .3
    changed[5, 1] -= .3
    changed_residual = problem.fun((changed/STATE_SCALE).ravel())
    np.testing.assert_allclose(changed_residual[anchor_start:], residual[anchor_start:], atol=1e-14)
    assert np.linalg.norm(changed_residual[:anchor_start]) > np.linalg.norm(residual[:anchor_start])
    # A shifted group mean is a finite soft penalty, not a validity failure.
    shifted = states.copy()
    shifted[groups == 0, 1] += .2
    shifted_residual = problem.fun((shifted/STATE_SCALE).ravel())
    assert np.linalg.norm(shifted_residual[anchor_start:]) > 0.
    assert np.isfinite(shifted_residual).all()


def test_same_frame_crosscheck_agreement_ignores_temporal_spread_and_detects_subset_mismatch():
    errors = [[1., -2.], [-2., 1.], [1., 1.]]
    base_state = [2.5, 2.]
    slots = [dict(held_point=j, scored=True, error_px=errors[j],
                  error_normalized=errors[j], state=base_state.copy(),
                  all_three_state=None, interior=True, bound=False)
             for j in range(3)]
    equal = _triple_metrics(slots)
    assert equal["G_theta_deg"] == 0.
    assert equal["G_A_D"] == 0.
    # These states may sit at one frame in a genuinely moving fixation; the
    # cross-check disagreement is simultaneous and is computed within-frame.
    temporal_states = np.array([[1.5, 1.], [2.5, 2.], [3.5, 3.]])
    temporal_spread = np.sqrt(np.mean((temporal_states-temporal_states.mean(0))**2, axis=0))
    assert np.all(temporal_spread > 0.)
    slots[2]["state"] = [base_state[0]+.6, base_state[1]-.4]
    changed = _triple_metrics(slots)
    assert changed["G_theta_deg"] > 0.
    assert changed["G_A_D"] > 0.
    assert changed["E_px"] == equal["E_px"]  # scoring errors stay independently fixed


def _power_artifact(model=None):
    model = PowerResponseModel(.5, coefficients(98)) if model is None else model
    pilot = PositionModel(27, coefficients(99))
    return power_artifact(model, pilot, np.array([0., 2.]), np.eye(12),
        {"training_groups": [1, 2], "training_rows": [10, 11]},
        {"converged": True}, {"minimum_scored_fraction": .8})


def test_power_schema_roundtrip_and_strict_family_exponent_scale_order_checks(tmp_path):
    obj = _power_artifact()
    model, loaded = from_object(deepcopy(obj))
    assert model.exponent == .5
    np.testing.assert_array_equal(model.beta, obj["coefficients"])
    path = tmp_path/"power.json"
    from full_position.schema import write_json
    write_json(path, obj)
    file_model, metadata = load_model(path)
    assert file_model.name == "ar27_sqrt"
    assert metadata["schema"] == obj["schema"]

    cases = []
    legacy = legacy_artifact(PositionModel(27, coefficients()), PositionModel(27, coefficients()),
        [0., 2.], np.eye(12), {"training_groups": [1]}, {"converged": True})
    cases.append((legacy, "explicit schema"))
    missing_exponent = deepcopy(obj)
    del missing_exponent["accommodation_response"]["exponent"]
    cases.append((missing_exponent, None))
    family_mismatch = deepcopy(obj)
    family_mismatch["accommodation_response"]["family"] = "conditional27"
    cases.append((family_mismatch, "family, exponent, capacity or coefficient order"))
    exponent_mismatch = deepcopy(obj)
    exponent_mismatch["accommodation_response"]["exponent"] = 1.
    cases.append((exponent_mismatch, "family, exponent, capacity or coefficient order"))
    scale_mismatch = deepcopy(obj)
    scale_mismatch["accommodation_response"]["accommodation_scale_D"] = 2.
    cases.append((scale_mismatch, "family, exponent, capacity or coefficient order"))
    order_mismatch = deepcopy(obj)
    order_mismatch["coefficient_order"] = "legacy or reordered coefficients"
    cases.append((order_mismatch, "coefficient_order"))
    missing_coefficients = deepcopy(obj)
    del missing_coefficients["coefficients"]
    cases.append((missing_coefficients, "coefficient"))
    for bad, message in cases:
        with pytest.raises(ValueError) as exc:
            from_object(bad)
        if message is not None:
            assert message.lower() in str(exc.value).lower()


def _selection_slot(key, candidate, held, error, state=(0., 2.)):
    fold, capture, fixation, row, source = key
    return dict(split_family="power", fold=fold, model=candidate, capture=capture,
        fixation=fixation, row=row, source_frame_index=source, held_point=held,
        nominal_theta=1000. if candidate == "ar27_log" else -1000., demand=800. if candidate == "ar27_log" else -800.,
        scored=True, input_valid=True, available=True, testable=True, certified=True,
        identifiable=True, unambiguous=True, error_px=list(error), error_normalized=list(error),
        state=list(state), bound=False, interior=True, failure_reason=None,
        support={"theta_anchor": True, "A_anchor": True, "theta_empirical": True,
            "A_empirical": True, "P1_context_valid": True,
            "context_outside_training_extrema": False, "p1_parity_seen_in_training": True})


def _selection_frame(key, candidate, error_scale, *, complete=True, state=(0., 2.)):
    fold, capture, fixation, row, source = key
    errors = [[error_scale, -error_scale], [error_scale, -error_scale], [error_scale, -error_scale]]
    slots = [_selection_slot(key, candidate, j, errors[j], state) for j in range(3)]
    return dict(split_family="power", fold=fold, model=candidate, capture=capture,
        fixation=fixation, row=row, source_frame_index=source,
        nominal_theta=9e8 if candidate == "ar27_log" else -9e8,
        demand=9e7 if candidate == "ar27_log" else -9e7,
        slots=slots, complete_triple=complete, complete_interior=complete,
        E_px=float(np.sqrt(2)*error_scale) if complete else None,
        E_normalized=float(np.sqrt(2)*error_scale) if complete else None,
        G_theta_deg=0. if complete else None, G_A_D=0. if complete else None,
        worst_point_px=float(np.sqrt(2)*error_scale) if complete else None,
        shared_accommodation_clipping=False)


def test_selector_ranks_large_errors_without_accuracy_gates_and_ignores_reporting_labels():
    keys = [("inner", "capA", 1, i, i) for i in (10, 11)]
    population = manifest([dict(zip(FIELDS, key)) for key in keys])
    records = {
        "ar27_log": [_selection_frame(k, "ar27_log", 1e6) for k in keys],
        "ar27_linear": [_selection_frame(k, "ar27_linear", 5e5) for k in keys],
    }
    policy = declared_policy()
    decision = choose_power(records, policy, "ar27_log", population)
    assert decision["status"] == "selected_for_outer_evaluation"
    assert decision["selected"] == "ar27_linear"
    assert decision["promoted_for_deployment"] is False
    assert decision["paired_inner_L_cross"]["ar27_linear"] > 1e11
    assert policy["absolute_accuracy_thresholds"] is None
    assert policy["nominal_error_is_acceptance_gate"] is False

    # Evaluation labels are retained as descriptive record fields. Replacing
    # them cannot alter state, point predictions, or the paired ranking.
    changed = deepcopy(records)
    for candidate_frames in changed.values():
        for frame in candidate_frames:
            frame["nominal_theta"] *= -1e5
            frame["demand"] *= -1e4
            for slot in frame["slots"]:
                slot["nominal_theta"] *= -1e5
                slot["demand"] *= -1e4
    after = choose_power(changed, policy, "ar27_log", population)
    assert after["selected"] == decision["selected"]
    assert after["paired_inner_L_cross"] == decision["paired_inner_L_cross"]
    assert [s["state"] for s in changed["ar27_linear"][0]["slots"]] == \
           [s["state"] for s in records["ar27_linear"][0]["slots"]]


def test_selector_preserves_missing_scheduled_frames_and_lost_exposures():
    keys = [("inner", cap, 1, i, i) for cap, i in (("capA", 10), ("capB", 20))]
    population = manifest([dict(zip(FIELDS, key)) for key in keys])
    full = [_selection_frame(key, "ar27_log", 10.) for key in keys]
    missing = [_selection_frame(keys[0], "ar27_linear", 1.)]
    with pytest.raises(ValueError, match="Scheduled population mismatch"):
        choose_power({"ar27_log": full, "ar27_linear": missing}, declared_policy(),
                     "ar27_log", population)

    # A scheduled exposure with explicitly unavailable slots remains in the
    # denominator and makes the candidate ineligible; it cannot disappear.
    unavailable = _selection_frame(keys[1], "ar27_linear", 1., complete=False)
    for slot in unavailable["slots"]:
        slot.update(scored=False, available=False, testable=False, certified=False,
                    identifiable=False, unambiguous=False, error_px=None, error_normalized=None,
                    state=None, failure_reason="missing exposure data")
    unavailable["E_px"] = unavailable["E_normalized"] = None
    decision = choose_power({"ar27_log": full,
        "ar27_linear": [full[0] | {"model": "ar27_linear", "slots": [dict(s, model="ar27_linear") for s in full[0]["slots"]]}, unavailable]},
        declared_policy(), "ar27_log", population)
    assert decision["status"] == "selected_for_outer_evaluation"
    assert decision["selected"] == "ar27_log"
    assert "ar27_linear" in decision["rejected"]


def test_nested_capture_grouping_keeps_outer_group_sealed_until_selection():
    group_data = {i: {"capture": f"cap{i//2}", "target_theta_deg": float(i),
                      "nominal_reporting": float(i), "raw_rows": [i]} for i in range(6)}
    metadata = {i: {"capture": value["capture"], "target_theta_deg": value["target_theta_deg"]}
                for i, value in group_data.items()}
    fit_events, eval_events = [], []

    def schedule(validation, split_id):
        return manifest([dict(fold=str(split_id), capture=payload["capture"], fixation=group,
            row=group*100+0, source_frame_index=group*100+0)
            for group, payload in validation.items()])

    def fit(train, candidate):
        fit_events.append((set(train), candidate))
        return candidate

    def evaluate(model, validation, population):
        eval_events.append((model, set(validation), population.frame_ids))
        err = 2e6 if model == "ar27_log" else 1e6
        output = []
        for group in validation:
            for key in population.frame_ids:
                if key[2] == group:
                    output.append(_selection_frame(key, model, err))
        return output

    result = nested_grouped(group_data, [("sealed_outer", [5])],
        ["ar27_log", "ar27_linear"], fit, evaluate, "ar27_log",
        policy=declared_policy(), schedule_validation=schedule,
        inner_splits=lambda ids: transfer_splits(ids, metadata, "capture"))
    fold = result["sealed_outer"]
    assert fold["inner_grouping"] == "explicit_predeclared_partition"
    assert fold["sealed_outer_group_ids"] == [5]
    assert fold["decision"]["selected"] == "ar27_linear"
    assert fold["decision"]["status"] == "selected_for_outer_evaluation"
    # Within each split, equal capture labels stay together; all fitting before
    # inner choice excludes the sealed group, and outer evaluation is last.
    inner_eval = eval_events[:-1]
    assert inner_eval
    assert all(not (ids & {5}) for ids in (train for train, _ in fit_events[:-1]))
    assert eval_events[-1][0] == "ar27_linear"
    assert eval_events[-1][1] == {5}
    assert max(i for i, event in enumerate(eval_events) if event[1] == {5}) == len(eval_events)-1
    split_groups = [set(item["group_ids"]) for item in fold["inner_split_group_ids"]]
    assert all((0 in g) == (1 in g) and (2 in g) == (3 in g) for g in split_groups)
