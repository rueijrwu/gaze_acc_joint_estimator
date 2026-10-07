"""Joint-refit objective, start archival, and matched-population regressions."""
import copy

import numpy as np
import pytest

from full_position.calibrate import ProfiledProblem, fit
from full_position.geometry import context
from full_position.model import STATE_SCALE
from full_position.sensitivity import paired, settings
from test_full_position import synthetic_model
from test_crosscheck_contract import _all_three


def calibration_data():
    rng = np.random.default_rng(19)
    model = synthetic_model(27)
    groups = np.repeat(np.arange(8), 4)
    anchors = np.array([[t, a] for t in [-8., -2., 3., 8.] for a in [.5, 3.]])
    states = anchors[groups]+rng.normal(size=(len(groups), 2))*.05
    ctx = context(np.array([[100., 60.], [180., 100.], [250., 50.]]))
    r = np.broadcast_to(ctx.r, (len(groups), 3, 2))
    y = model.predict(states, r)+rng.normal(size=(len(states), 6))*.0001
    cov = np.broadcast_to(np.eye(6)*1e-4, (len(states), 6, 6))
    return model, groups, anchors, states, r, y, cov


def test_changed_anchor_scales_preserve_exact_profile_derivative_and_cost_weighting():
    model, groups, anchors, states, r, y, cov = calibration_data()
    weak = ProfiledProblem(model, y, r, cov, groups, anchors, anchor_scales=(.2, .5))
    strong = ProfiledProblem(model, y, r, cov, groups, anchors, anchor_scales=(.1, .25))
    z = (states/STATE_SCALE).ravel()
    a, b = weak.fun(z), strong.fun(z)
    np.testing.assert_allclose(a[:len(weak.b)], b[:len(strong.b)])
    np.testing.assert_allclose(b[len(strong.b):], 2*a[len(weak.b):])
    rng = np.random.default_rng(23)
    v = rng.normal(size=len(z))
    jv = weak.jac(z)@v
    h = 1e-6
    numeric = (weak.fun(z+h*v)-weak.fun(z-h*v))/(2*h)
    np.testing.assert_allclose(jv, numeric, atol=2e-5, rtol=2e-4)
    weak.update(z)
    w = rng.normal(size=len(weak.residual))
    assert jv@w == pytest.approx(v@weak.vjp(w), abs=1e-7)


@pytest.mark.parametrize("scales", [(0., .25), (-.1, .25), (np.nan, .25), (.1,), (.1, np.inf)])
def test_invalid_anchor_precision_is_rejected(scales):
    model, groups, anchors, _, r, y, cov = calibration_data()
    with pytest.raises(ValueError, match="Anchor scales"):
        ProfiledProblem(model, y, r, cov, groups, anchors, anchor_scales=scales)


def test_joint_fit_archives_failed_starts_and_returns_selected_coefficients_and_states():
    model, groups, anchors, states, r, y, cov = calibration_data()
    archives = []
    fitted, selected_states, diag = fit(model, y, r, cov, groups, anchors, starts=1,
        max_nfev=2, additional_initial_states=[states], anchor_scales=(.2, .5),
        checkpoint=lambda record, beta, x: archives.append((record, beta, x)))
    assert len(archives) == len(diag["alternatives"]) == 2
    assert diag["anchor_scales"] == [.2, .5]
    assert all(a[0]["start"] == i for i, a in enumerate(archives))
    record, beta, x = archives[diag["selected_start"]]
    np.testing.assert_array_equal(fitted.beta, beta)
    np.testing.assert_array_equal(selected_states, x)
    assert sum(record["objective_components"].values()) == pytest.approx(record["cost"], rel=1e-12)
    assert not diag["converged"]  # a two-evaluation checkpoint is never accepted as a fit


def test_variant_settings_use_training_anchor_quantiles_and_physical_precisions():
    anchors = np.array([[-10., .5], [-5., 2.], [0., 3.], [10., 4.]])
    meta = {"calibration": {"anchor_scales": [.1, .25], "prior_strength": .001},
            "reference_state": [-2.5, 2.5]}
    weak = settings(meta, anchors, "anchor_weak")
    assert weak["anchor_scales"] == [.2, .5]
    assert weak["anchor_precision_factor"] == .25
    assert settings(meta, anchors, "covariance_4x")["covariance_factor"] == 4
    np.testing.assert_array_equal(settings(meta, anchors, "reference_q25")["reference_state"],
                                  np.quantile(anchors, .25, axis=0))
    assert settings(meta, anchors, "prior_strong")["prior_strength"] == .01


def test_uniform_covariance_inflation_matches_relative_anchor_precision_control():
    model, groups, anchors, states, r, y, cov = calibration_data()
    inflated = ProfiledProblem(model, y, r, cov*4, groups, anchors, anchor_scales=(.1, .25))
    stronger = ProfiledProblem(model, y, r, cov, groups, anchors, anchor_scales=(.05, .125))
    z = (states/STATE_SCALE).ravel()
    a, b = inflated.fun(z), stronger.fun(z)
    np.testing.assert_allclose(inflated.beta, stronger.beta, atol=1e-10)
    assert b@b == pytest.approx(4*(a@a), rel=1e-10)
    np.testing.assert_allclose(stronger.vjp(b), 4*inflated.vjp(a), atol=1e-9)


def test_variant_comparisons_do_not_shrink_scheduled_denominators_or_mix_interior_masks():
    old = [_all_three(row=1), _all_three(row=2)]
    new = copy.deepcopy(old)
    new[0]["slots"][0].update(bound=True, interior=False)
    new[0]["complete_interior"] = False
    new[1]["slots"][1]["scored"] = False
    new[1]["complete_triple"] = False
    result = paired(old, new)
    assert result["scheduled_frame_count"] == 2
    assert len(result["shared_scored_point_ids"]) == 5
    assert len(result["shared_complete_frame_ids"]) == 1
    assert len(result["shared_interior_point_ids"]) == 4
    assert result["shared_complete_interior_frame_ids"] == []
    assert result["all_testable"]["candidate"]["E_px"]["pooled"]["n"] == 1
    assert result["interior"]["candidate"]["E_px"]["pooled"]["n"] == 0
    missing = new[:1]
    with pytest.raises(ValueError, match="identical scheduled"):
        paired(old, missing)
