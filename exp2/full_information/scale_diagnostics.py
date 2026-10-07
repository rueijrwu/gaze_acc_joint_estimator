"""Descriptive scale associations and auditable Gaussian information controls.

This module does not fit a physiological scale model. Its numerical controls use
synthetic residuals and explicitly hypothetical scale constraints. The free-scale
control uses precisely the same marginal relative covariance and radial loss.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


def raw_to_independent(raw):
    """[m,S1,S4] -> [d,rho4,log(S1)] on the positive S1 branch."""
    raw = np.asarray(raw, dtype=float)
    if raw.shape[-1] != 3 or not np.all(np.isfinite(raw)) or np.any(raw[..., 1] <= 0):
        raise ValueError("Raw coordinates must be finite [m,S1,S4] with S1 > 0")
    return np.stack((raw[..., 0] / raw[..., 1], raw[..., 2] / raw[..., 1],
                     np.log(raw[..., 1])), axis=-1)


def independent_to_raw(independent):
    """Inverse of raw_to_independent; no extra image-position channel."""
    independent = np.asarray(independent, dtype=float)
    if independent.shape[-1] != 3 or not np.all(np.isfinite(independent)):
        raise ValueError("Independent coordinates must be finite [d,rho4,log(S1)]")
    scale = np.exp(independent[..., 2])
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("Scale is outside the representable positive branch")
    return np.stack((independent[..., 0] * scale, scale,
                     independent[..., 1] * scale), axis=-1)


def independent_jacobian(raw_anchor):
    """Jacobian in raw [m,S1,S4] order; retains induced noise correlations."""
    m, s1, s4 = np.asarray(raw_anchor, dtype=float)
    raw_to_independent(raw_anchor)
    return np.array([[1 / s1, -m / s1**2, 0],
                     [0, -s4 / s1**2, 1 / s1], [0, 1 / s1, 0]])


def _radial_losses(q):
    q = np.asarray(q, dtype=float)
    return {"linear": q, "soft_l1": 2 * (np.sqrt(1 + q) - 1),
            "huber": np.where(q <= 1, q, 2 * np.sqrt(q) - 1),
            "cauchy": np.log1p(q), "arctan": np.arctan(q)}


def scale_information_controls(raw_covariance=None, raw_anchor=None):
    """Audit known/constrained/free scale with a correlated synthetic Gaussian.

    C=[[A,b],[b.T,c]] and h=c-b.T A^-1 b. For residual [r,t],
    q=r.T A^-1 r+(t-b.T A^-1 r)^2/h. Profiling an unrestricted
    per-frame t therefore exactly recovers the two-channel marginal objective.
    A Gaussian prior of variance tau² on t gives the diagnostic quadratic
    q_profile=q_relative+(t0-b.T A^-1 r)^2/(h+tau²). This constrained
    formula does NOT apply to a robust data term plus a separate Gaussian prior.
    """
    anchor = np.asarray(raw_anchor if raw_anchor is not None else [18., 446., 55.], dtype=float)
    synthetic = raw_covariance is None
    covariance = np.asarray(raw_covariance if raw_covariance is not None else
                            [[.09, .035, -.018], [.035, .25, .04], [-.018, .04, .16]], dtype=float)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ValueError("raw_covariance must be a finite 3x3 matrix")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-14):
        raise ValueError("raw_covariance must be symmetric")
    np.linalg.cholesky(covariance)
    jacobian = independent_jacobian(anchor)
    transformed = jacobian @ covariance @ jacobian.T
    a, b, c = transformed[:2, :2], transformed[:2, 2], transformed[2, 2]
    a_inv_b = np.linalg.solve(a, b)
    schur = float(c - b @ a_inv_b)
    if schur <= 0:
        raise ValueError("Conditional scale variance must be positive")
    rng = np.random.default_rng(1729)
    residuals = rng.normal(size=(256, 3)) @ np.linalg.cholesky(transformed).T
    r = residuals[:, :2]
    q_relative = np.einsum("ni,in->n", r, np.linalg.solve(a, r.T))
    profiled_t = r @ a_inv_b
    free_residuals = np.column_stack((r, profiled_t))
    q_free = np.einsum("ni,in->n", free_residuals, np.linalg.solve(transformed, free_residuals.T))
    q_full = np.einsum("ni,in->n", residuals, np.linalg.solve(transformed, residuals.T))
    q_decomposed = q_relative + (residuals[:, 2] - profiled_t)**2 / schur
    raw_residuals = np.linalg.solve(jacobian, residuals.T).T
    q_raw = np.einsum("ni,in->n", raw_residuals, np.linalg.solve(covariance, raw_residuals.T))
    # Independent synthetic parameter derivatives; unrestricted scale nuisance
    # has derivative e3. This Schur complement checks information, not just cost.
    h_model = rng.normal(size=(3, 4))
    full_model = np.column_stack((h_model, np.array([0., 0., 1.])))
    fisher = full_model.T @ np.linalg.solve(transformed, full_model)
    fisher_free = fisher[:-1, :-1] - np.outer(fisher[:-1, -1], fisher[-1, :-1]) / fisher[-1, -1]
    fisher_relative = h_model[:2].T @ np.linalg.solve(a, h_model[:2])
    relative_norm = max(float(np.linalg.norm(fisher_relative)), 1.)
    constrained = []
    for multiplier in (0., .1, 1., 10., 100.):
        tau2 = multiplier * schur
        t0 = residuals[:, 2]
        optimum = (schur * t0 + tau2 * profiled_t) / (schur + tau2)
        candidate = np.column_stack((r, optimum))
        q_candidate = np.einsum("ni,in->n", candidate, np.linalg.solve(transformed, candidate.T))
        if tau2 > 0:
            q_candidate += (optimum - t0)**2 / tau2
        closed = q_relative + (t0 - profiled_t)**2 / (schur + tau2)
        constrained_covariance = transformed.copy()
        constrained_covariance[2, 2] += tau2
        constrained_information = h_model.T @ np.linalg.solve(constrained_covariance, h_model)
        extra_information = constrained_information - fisher_relative
        # The rank-one scale contribution decreases to zero as its uncertainty
        # increases. These derivatives are synthetic, not estimated physiology.
        extra_eigenvalues = np.linalg.eigvalsh((extra_information + extra_information.T) / 2)
        constrained.append({"hypothesis": "known" if tau2 == 0 else "Gaussian-constrained-varying",
                            "prior_variance_over_conditional_variance": multiplier,
                            "mean_extra_quadratic_cost": float(np.mean(closed - q_relative)),
                            "max_closed_form_error": float(np.max(np.abs(closed - q_candidate))),
                            "synthetic_extra_information_eigenvalues": extra_eigenvalues.tolist()})
    losses_relative, losses_free = _radial_losses(q_relative), _radial_losses(q_free)
    roundtrip_error = float(np.max(np.abs(independent_to_raw(raw_to_independent(anchor)) - anchor)))
    return {
        "scope": "Synthetic algebra control; scale priors are hypotheses, not validated physiology.",
        "covariance_source": "synthetic correlated positive-definite raw covariance" if synthetic else
            "supplied descriptive raw covariance; may contain eye motion and detector variation",
        "raw_coordinate_order": ["m", "S1", "S4"],
        "independent_coordinate_order": ["d", "rho4", "logS1"],
        "raw_anchor": anchor.tolist(), "raw_covariance": covariance.tolist(),
        "independent_covariance": transformed.tolist(),
        "conditional_scale_variance": schur,
        "identity": "q_full=q_relative+(t-b.T@inv(A)@r)^2/(c-b.T@inv(A)@b)",
        "free_scale_optimum": "t=b.T@inv(A)@r, one unrestricted nuisance per frame",
        "free_scale_shape_prior": "none",
        "constrained_formula_scope": "Gaussian quadratic data term with Gaussian scale prior only",
        "constraints": constrained,
        "checks": {"coordinate_roundtrip_max_abs": roundtrip_error,
                   "linearized_raw_vs_independent_quadratic_max_abs": float(np.max(np.abs(q_raw - q_full))),
                   "quadratic_decomposition_max_abs": float(np.max(np.abs(q_full - q_decomposed))),
                   "free_profile_max_abs": float(np.max(np.abs(q_free - q_relative))),
                   "free_information_relative_error": float(np.linalg.norm(fisher_free - fisher_relative) / relative_norm),
                   "free_radial_robust_max_abs": {name: float(np.max(np.abs(losses_relative[name] - losses_free[name])))
                                                 for name in losses_relative}},
        "interpretation": "Additional information requires a constraint on scale; free per-frame scale exactly recovers the matched two-channel marginal objective, including common monotone radial robust losses. Separate per-channel losses are not this control.",
    }


def _number(row, key):
    try:
        value = float(row.get(key, np.nan))
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _association(x, y):
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[keep], np.asarray(y)[keep]
    if len(x) < 3 or np.ptp(x) == 0:
        return {"n": int(len(x)), "slope": None, "correlation": None}
    slope = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)[0][1]
    corr = np.corrcoef(x, y)[0, 1] if np.ptp(y) else np.nan
    return {"n": int(len(x)), "slope": float(slope),
            "correlation": float(corr) if np.isfinite(corr) else None}


def describe_scale_structure(rows: Iterable[Mapping]):
    """Consume frame-level mappings from the shared frozen measurement gate.

    Required: capture, fixation_index, frame_index, m, S1, S4.
    Optional: nominal_gaze_deg, demand_D and numeric detector_* QA fields.
    Absolute image positions are deliberately absent from the predictor set.
    """
    rows = [dict(row) for row in rows]
    rows = [row for row in rows if np.isfinite(_number(row, "S1")) and _number(row, "S1") > 0]
    if not rows:
        raise ValueError("No positive finite S1 observations")
    captures = sorted({str(row.get("capture", "unknown")) for row in rows})
    qa_keys = sorted({key for row in rows for key in row if key.startswith("detector_")})
    predictors = ["nominal_gaze_deg", "within_recording_order"] + qa_keys
    summaries, fixation_summaries, centered_rows = [], [], []
    for capture in captures:
        block = sorted((row for row in rows if str(row.get("capture", "unknown")) == capture),
                       key=lambda row: (_number(row, "frame_index"), _number(row, "fixation_index")))
        scales = np.array([_number(row, "S1") for row in block])
        frames = np.array([_number(row, "frame_index") for row in block])
        if np.all(np.isfinite(frames)) and np.ptp(frames):
            order = (frames - np.min(frames)) / np.ptp(frames)
        else:
            order = np.linspace(0, 1, len(block))
        for row, position in zip(block, order):
            row["within_recording_order"] = float(position)
        med = float(np.median(scales))
        demands = sorted({_number(row, "demand_D") for row in block if np.isfinite(_number(row, "demand_D"))})
        summary = {"capture": capture, "n": len(block), "S1_median_px": med,
                   "S1_p05_px": float(np.percentile(scales, 5)), "S1_p95_px": float(np.percentile(scales, 95)),
                   "S1_robust_cv": float(1.4826 * np.median(np.abs(scales - med)) / med),
                   "demand_D_values": demands, "associations": {}}
        for key in predictors:
            values = np.array([_number(row, key) for row in block])
            summary["associations"][key] = _association(values, scales)
        summaries.append(summary)
        for fixation in sorted({str(row.get("fixation_index", "unknown")) for row in block}):
            part = [row for row in block if str(row.get("fixation_index", "unknown")) == fixation]
            values = np.array([_number(row, "S1") for row in part])
            fixation_summaries.append({"capture": capture, "fixation_index": fixation,
                                      "n": len(part), "S1_mean_px": float(np.mean(values)),
                                      "S1_median_px": float(np.median(values)),
                                      "S1_std_px": float(np.std(values))})
        # Within-recording centering removes recording intercepts, including the
        # constant recording-specific demand. It cannot identify demand effects.
        matrix = np.array([[_number(row, key) for key in predictors] for row in block])
        log_scale = np.log(scales)
        centered_rows.append((matrix, log_scale))
    active = [i for i in range(len(predictors)) if any(
        np.sum(np.isfinite(matrix[:, i])) >= 3 and np.nanstd(matrix[:, i]) > 0
        for matrix, _ in centered_rows)]
    x_parts, y_parts = [], []
    for matrix, log_scale in centered_rows:
        selected = matrix[:, active]
        keep = np.all(np.isfinite(selected), axis=1) & np.isfinite(log_scale)
        if np.sum(keep) >= 2:
            x_parts.append(selected[keep] - np.mean(selected[keep], axis=0))
            y_parts.append(log_scale[keep] - np.mean(log_scale[keep]))
    conditional = {"scope": "Descriptive logS1 regression after recording-intercept removal; frame/order and available detector QA covariates. No causal interpretation or independent gaze reference."}
    if x_parts and active:
        x, y = np.vstack(x_parts), np.concatenate(y_parts)
        norms = np.linalg.norm(x, axis=0)
        nonzero = norms > 0
        x_standard = x[:, nonzero] / norms[nonzero]
        coefficients, _, rank, singular = np.linalg.lstsq(x_standard, y, rcond=None)
        full_rank = rank == x_standard.shape[1]
        fitted = x_standard @ coefficients
        total = float(y @ y)
        conditional.update({"n": len(y), "predictors": [predictors[active[i]] for i in np.flatnonzero(nonzero)],
                            "rank": int(rank), "full_rank": bool(full_rank),
                            "condition_number": float(singular[0] / singular[-1]) if full_rank else None,
                            "within_recording_R2": float(1 - np.sum((y - fitted)**2) / total) if total else None,
                            "coefficients_logS1_per_unit": {predictors[active[i]]: float(value / norms[i])
                                for i, value in zip(np.flatnonzero(nonzero), coefficients)} if full_rank else None,
                            "coefficient_warning": "Nominal gaze, order and QA may be collinear; associations depend on this model and cannot establish accommodation."})
    else:
        conditional.update({"n": 0, "predictors": [], "full_rank": False,
                            "coefficients_logS1_per_unit": None})
    return {"scope": "Descriptive retained-frame scale structure, conditional on the shared frozen gate.",
            "limitations": ["Demand and recording are completely confounded: capture1..4 correspond to approximately 0.36036, 4, 3, 2 D, each with five fixations.",
                            "Capture1 wall-clock timestamps are unreliable; all time comparisons use within-recording frame/order.",
                            "No independent scale, head-distance, gaze or accommodation reference is available.",
                            "Capture5 lacks fixation/protocol annotations and cannot establish an independent repeat.",
                            "Nominal-gaze, time and detector associations are observational; consecutive differences can contain actual eye motion."],
            "demand_effect_identifiable_with_recording_effects": False,
            "recordings": summaries, "fixations": fixation_summaries,
            "conditional_structure": conditional}


def run_scale_diagnostics(rows, output_dir, provenance=None):
    """Write new Stage2 artifacts; fail rather than replace existing artifacts."""
    output_dir = Path(output_dir)
    targets = [output_dir / name for name in ("scale_structure.json", "scale_controls.json", "scale_fixations.csv")]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite scale artifacts: " + ", ".join(existing))
    structure = describe_scale_structure(rows)
    structure["provenance"] = provenance
    controls = scale_information_controls()
    # Numerical tolerances apply to the synthetic algebra audit, not physiology.
    checks = controls["checks"]
    if checks["free_information_relative_error"] > 1e-10 or checks["free_profile_max_abs"] > 1e-10:
        raise AssertionError("Matched free-scale information control failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, payload in zip(targets[:2], (structure, controls)):
        with path.open("x") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
    with targets[2].open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["capture", "fixation_index", "n", "S1_mean_px", "S1_median_px", "S1_std_px"])
        writer.writeheader()
        writer.writerows(structure["fixations"])
    return {"structure": structure, "controls": controls,
            "artifacts": [str(path) for path in targets]}
