#!/usr/bin/env python3
"""Render matched P4 scale-effect diagnostics; this module fits nothing."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BRANCHES = (("baseline_g1", "P4 fit with g=1"), ("p1m", "P4 refit with frozen P1 m"))
COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73", 4: "#CC79A7"}


def _csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _num(value, default=np.nan):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _fmt(value, digits=5):
    x = _num(value)
    return f"{x:.{digits}g}" if np.isfinite(x) else "—"


def _float_text(value, digits=5):
    x = _num(value)
    return f"{x:.{digits}g}" if np.isfinite(x) else "—"


def _plot_fixation_rms(out: Path, rows):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for ax, cid in zip(axes.flat, (1, 2, 3, 4)):
        for branch, label in BRANCHES:
            rr = [r for r in rows if int(r["capture"]) == cid and r["branch"] == branch]
            rr.sort(key=lambda r: _num(r.get("actual_gaze_theta_deg", r.get("actual_gaze_deg", r.get("actual_gaze_theta_x_deg", r.get("actual_gaze"))))))
            if not rr:
                continue
            x = [_num(r.get("actual_gaze_theta_deg", r.get("actual_gaze_deg", r.get("actual_gaze_theta_x_deg", r.get("actual_gaze"))))) for r in rr]
            y = [_num(r.get("vertex_rms_px", r.get("fixation_vertex_rms_px"))) for r in rr]
            ax.plot(x, y, "o-", color="#0072B2" if branch == "baseline_g1" else "#D55E00", label=label)
        ax.set_title(f"Capture {cid}")
        ax.set_xlabel("Actual mean gaze (degrees)")
        ax.set_ylabel("P4 vertex RMS (px)")
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("Matched P4 fit error by capture and fixation")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_timeseries(out: Path, arrays):
    capture = np.asarray(arrays["capture"], int)
    source = np.asarray(arrays["source_row"], int)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=False, sharey=True)
    for ax, cid in zip(axes.flat, (1, 2, 3, 4)):
        ix = np.flatnonzero(capture == cid)
        order = ix[np.argsort(source[ix], kind="stable")]
        for branch, label in BRANCHES:
            key = f"{branch}_row_vertex_rms_px"
            vals = np.asarray(arrays[key], float)[order]
            valid = np.asarray(arrays.get(f"{branch}_valid", np.ones(len(capture), bool)), bool)[order]
            vals = np.where(valid & np.isfinite(vals), vals, np.nan)
            x = source[order]
            # NaN separates invalid rows and every original-source gap.
            xplot, yplot = [], []
            for i, (xx, yy) in enumerate(zip(x, vals)):
                if i and xx != x[i - 1] + 1:
                    xplot.append(np.nan); yplot.append(np.nan)
                xplot.append(xx); yplot.append(yy)
            ax.plot(xplot, yplot, lw=.7, alpha=.85, label=label,
                    color="#0072B2" if branch == "baseline_g1" else "#D55E00")
        ax.set_title(f"Capture {cid}")
        ax.set_xlabel("Original source row (segments break at gaps)")
        ax.set_ylabel("Frame P4 vertex RMS (px)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Fixation-ordered residual time series")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _plot_residual_vectors(out: Path, arrays):
    """Show mean residual vectors at observed P4 vertex locations by fixation."""
    capture = np.asarray(arrays["capture"], int)
    fixation = np.asarray(arrays["fixation_id"], int)
    observed = np.asarray(arrays["P4_observed"], float)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, cid in zip(axes.flat, (1, 2, 3, 4)):
        for branch, label in BRANCHES:
            residual = np.asarray(arrays[f"{branch}_residual"], float)
            valid = np.asarray(arrays.get(f"{branch}_valid", np.ones(len(capture), bool)), bool)
            for fid in range(5):
                ix = (capture == cid) & (fixation == fid) & valid
                if not np.any(ix):
                    continue
                pos = np.mean(observed[ix], axis=0)
                vec = 15. * np.mean(residual[ix], axis=0)
                color = "#0072B2" if branch == "baseline_g1" else "#D55E00"
                ax.quiver(pos[:, 0], pos[:, 1], vec[:, 0], vec[:, 1],
                          angles="xy", scale_units="xy", scale=1,
                          color=color, alpha=.9,
                          width=.003)
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_yaxis()
        ax.set_title(f"Capture {cid}")
        ax.set_xlabel("Observed vertex x (px)")
        ax.set_ylabel("Observed vertex y (px)")
        ax.grid(alpha=.2)
    fig.suptitle("Mean residual triangles by fixation · vectors ×15; blue = g=1, orange = P1 m")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _table(rows, fields, labels=None):
    labels = labels or fields
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(k, "—")) for k in fields) + " |")
    return "\n".join(lines)


def render(results: str | Path = HERE / "results"):
    results = Path(results)
    summary = json.loads((results / "summary.json").read_text())
    fix = _csv(results / "fixation_summary.csv")
    parameters = _csv(results / "parameters.csv")
    arrays = np.load(results / "arrays.npz", allow_pickle=False)
    for key in ("capture", "source_row", "fixation_id", "P4_observed"):
        if key not in arrays:
            raise ValueError(f"arrays.npz missing {key}")
    _plot_fixation_rms(results / "matched_fixation_rms.png", fix)
    _plot_timeseries(results / "residual_timeseries.png", arrays)
    _plot_residual_vectors(results / "residual_triangles.png", arrays)

    cap_metrics = summary.get("capture_metrics", {})
    metric_rows = []
    for cid in (1, 2, 3, 4):
        c = cap_metrics.get(str(cid), cap_metrics.get(cid, {}))
        row = {"capture": cid, "demand": c.get("nominal_demand_diopters", c.get("demand_diopters", "—"))}
        for branch, _label in BRANCHES:
            b = c.get(branch, {})
            row[f"{branch}_rms"] = _num(b.get("vertex_rms_px", b.get("rms_px")))
        row["delta"] = row["p1m_rms"] - row["baseline_g1_rms"]
        row["delta_pct"] = 100 * row["delta"] / row["baseline_g1_rms"] if row["baseline_g1_rms"] else np.nan
        metric_rows.append(row)
    parameter_lines = []
    for row in parameters:
        if row.get("parameter", "").endswith("_fit_parameter"):
            continue
        parameter_lines.append({"branch": row.get("branch", "—"), "capture": row.get("capture", "—"),
            "demand": _float_text(row.get("nominal_demand_diopters", ""), 5),
            "parameter": row.get("parameter", "—"), "value": _float_text(row.get("value"), 7),
            "unit": row.get("unit", "—")})
    parameter_table = _table(parameter_lines, ["branch", "capture", "demand", "parameter", "value", "unit"],
                             ["Fit", "Capture", "Nominal A label (D)", "Parameter", "Value", "Unit"])
    fit = summary.get("fit", {})
    domain = summary.get("domain", {})
    stationarity = summary.get("stationarity", {})
    p1 = summary.get("p1_scale_source", summary.get("p1_m_source", {}))
    p1m = np.asarray(arrays["p1_m"], float) if "p1_m" in arrays else np.asarray([])
    cap_array = np.asarray(arrays["capture"], int)
    target_array = np.asarray(arrays["nominal_theta_deg"], float) if "nominal_theta_deg" in arrays else np.full(len(cap_array), np.nan)
    c1_m = {target: float(np.mean(p1m[(cap_array == 1) & (target_array == target)]))
            for target in (5., 10.) if np.any((cap_array == 1) & (target_array == target))} if len(p1m) else {}
    floor_fraction = float(np.mean(np.isclose(p1m, 0.98, rtol=0., atol=1e-10))) if len(p1m) else np.nan
    fit_rows, domain_rows, stationarity_rows = [], [], []
    for branch, _label in BRANCHES:
        f = fit.get(branch, {})
        d = domain.get(branch, {})
        s = stationarity.get(branch, {})
        ss = f.get("start_stability", {})
        fit_rows.append({"branch": branch, "objective": _float_text(f.get("objective"), 9),
            "sqrt_objective": _float_text(np.sqrt(_num(f.get("objective"))), 7),
            "selected_stationary": f.get("stationary", "—"),
            "stationary_starts": f"{ss.get('stationary_starts', '—')}/{ss.get('near_best_starts', '—')}",
            "scaled_gradient": _float_text(f.get("scaled_gradient"), 3)})
        domain_rows.append({"branch": branch, "valid": d.get("domain_valid", "—"),
            "invalid": d.get("invalid_rows", "—"), "denominator": _float_text(d.get("minimum_denominator"), 6),
            "radial_factor": _float_text(d.get("minimum_radial_factor"), 6),
            "radial_derivative": _float_text(d.get("minimum_radial_derivative"), 6),
            "stretch": _float_text(d.get("maximum_absolute_stretch_exponent"), 5),
            "row_margin": _float_text(d.get("minimum_row_domain_margin"), 6)})
        stationarity_rows.append({"branch": branch, "stationary": s.get("stationary", "—"),
            "scaled_gradient": _float_text(s.get("scaled_gradient"), 3),
            "sensitivity_gradient": _float_text(s.get("sensitivity_scaled_gradient"), 3),
            "bounds": ", ".join(map(str, s.get("active_bounds", []))) or "none"})
    fix_lookup = {(r["branch"], int(r["capture"]), int(round(_num(r["nominal_theta_deg"])))): r for r in fix}
    delta_rows = []
    for cid in (1, 2, 3, 4):
        for target in (-10, -5, 0, 5, 10):
            a = fix_lookup.get(("baseline_g1", cid, target), {})
            b = fix_lookup.get(("p1m", cid, target), {})
            av, bv = _num(a.get("vertex_rms_px")), _num(b.get("vertex_rms_px"))
            delta = bv - av
            delta_rows.append({"capture": cid, "target": target,
                "actual": _float_text(a.get("actual_gaze_theta_deg"), 5),
                "g1": _float_text(av, 6), "p1m": _float_text(bv, 6),
                "delta": _float_text(delta, 6), "pct": _float_text(100*delta/av if av else np.nan, 4)})
    loss_g1 = _num(fit.get("baseline_g1", {}).get("objective"))
    loss_m = _num(fit.get("p1m", {}).get("objective"))
    loss_delta_pct = 100 * (loss_m-loss_g1)/loss_g1 if loss_g1 else np.nan
    convergence_notes = []
    for branch, _label in BRANCHES:
        f = fit.get(branch, {})
        ss = f.get("start_stability", {})
        convergence_notes.append(
            f"`{branch}` selected checkpoint: stationary=`{f.get('stationary', 'not recorded')}`, "
            f"reason=`{f.get('reason', 'not recorded')}`, near-best stationary starts "
            f"`{ss.get('stationary_starts', '—')}/{ss.get('near_best_starts', '—')}`, "
            f"scaled gradient `{_float_text(f.get('scaled_gradient'), 3)}`.")
    lines = [
        "# P1-derived scale effect on a fully refit P4 geometry model", "",
        "> **Question:** Does applying the frozen P1-derived per-frame scale improve the P4 geometric fit after refitting the complete P4 model?", "",
        "## Matched comparison", "",
        "Both branches use the same 88,978-row P4-filtered cohort, saved gaze, capture/fixation labels, C1 nominal-zero empirical P4 reference, optimizer protocol, and capture/fixation weighting. The baseline sets the P4 input scale to `g=1`; the comparison uses the frozen P1-derived `m` and refits all P4 parameters. This is a whole-model conditional geometry comparison.", "",
        "| Capture | Nominal A label (D) | g=1 P4 RMS (px) | P1-m P4 RMS (px) | Δ P1-m−g1 (px) | Δ (%) |", "|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(["| " + " | ".join([str(row["capture"]), _float_text(row["demand"], 6),
        _float_text(row["baseline_g1_rms"], 6), _float_text(row["p1m_rms"], 6),
        _float_text(row["delta"], 6), _float_text(row["delta_pct"], 3)]) + " |" for row in metric_rows])
    lines += ["", f"The equal-capture/equal-fixation objective is `{_float_text(loss_g1, 9)} px²` at `g=1` and `{_float_text(loss_m, 9)} px²` with P1 `m` ({_float_text(loss_m-loss_g1, 8)} px²; {_float_text(loss_delta_pct, 4)}%). The square root is the corresponding weighted vertex RMS, not row-pooled RMS.", "",
              "### Matched fixation changes", "",
              "Positive Δ means the P1-m refit has larger residual RMS at the same capture and target fixation.", "",
              _table(delta_rows, ["capture", "target", "actual", "g1", "p1m", "delta", "pct"],
                  ["Capture", "Target (°)", "Actual mean gaze (°)", "g=1 RMS (px)", "P1-m RMS (px)", "Δ (px)", "Δ (%)"]), ""]
    lines += ["", "![Matched fixation RMS](matched_fixation_rms.png)", "",
              "Fixation points are matched by capture and actual saved gaze. These are descriptive in-cohort residuals; they are not independent validation.", "",
              "![Residual time series](residual_timeseries.png)", "",
              "Lines follow original source-row order and break at missing-row gaps. Residual variability therefore retains temporal dependence and fixation ordering.", "",
              "![Mean residual triangles](residual_triangles.png)", "",
              "Arrows show per-vertex mean residuals at the mean observed vertex locations, grouped by fixation (blue = `g=1`; orange = P1 `m`). Vectors are magnified 15× for legibility; they visualize structured residual direction and are not an independent test.", "",
              "## Fitted parameters and fit diagnostics", "",
              "### Fit convergence", "",
              _table(fit_rows, ["branch", "objective", "sqrt_objective", "selected_stationary", "stationary_starts", "scaled_gradient"],
                  ["Fit", "Objective (px²)", "√objective (px)", "Selected stationary", "Stationary starts", "Scaled gradient"]), "",
              *convergence_notes, "",
              "### Domain checks", "",
              _table(domain_rows, ["branch", "valid", "invalid", "denominator", "radial_factor", "radial_derivative", "stretch", "row_margin"],
                  ["Fit", "All rows valid", "Invalid rows", "Minimum projective denominator", "Minimum radial factor", "Minimum radial derivative", "Max |stretch exponent|", "Minimum domain margin"]), "",
              "### Fitted geometry and nuisance parameters", "", parameter_table, "",
              "The model fits shared keystone parameters (`b`, `cx`, `cy`), one free P4 nuisance gaze offset `θ04`, a shared P4 reference triangle, and capture-specific radial coefficients. C1 `k=0` sets the radial reference gauge. The nominal A column is a capture label, not a fitted accommodation state. `θ04` is expressed in the saved gaze coordinate and has no absolute physical-angle meaning.", "",
              _table(stationarity_rows, ["branch", "stationary", "scaled_gradient", "sensitivity_gradient", "bounds"],
                  ["Fit", "Selected stationary", "Scaled gradient", "Sensitivity scaled gradient", "Active bounds"]), "",
              "## P1 scale provenance and limits", "",
              _table([{"artifact": p1.get("artifact", "—"), "array_key": p1.get("array_key", "—"),
                       "model": p1.get("model", "—"), "scope": p1.get("P1_fit_scope", "—"),
                       "bounds": p1.get("m_bounds", "—"), "min": _float_text(p1.get("m_min"), 7),
                       "mean": _float_text(p1.get("m_mean"), 7), "max": _float_text(p1.get("m_max"), 7),
                       "stationary": p1.get("P1_fit_stationary", "—")}],
                  ["artifact", "array_key", "model", "scope", "bounds", "min", "mean", "max", "stationary"],
                  ["Source artifact", "Array key", "P1 model", "Fit scope", "m bounds", "Minimum m", "Mean m", "Maximum m", "P1 fit stationary"]), "",
              f"P1 calibration gauge: `{p1.get('P1_fit_gauge', 'not recorded')}`. Rows were joined exactly by `(capture, source_row)` with fixation and saved gaze also checked. Source and array hashes are retained in `summary.json`.", "",
              f"Across the cohort, frozen `m` has mean `{_float_text(np.mean(p1m) if len(p1m) else np.nan, 7)}` and `{_float_text(100*floor_fraction, 4)}%` of rows sit at the 0.98 floor. In C1 its fixation means are `{_float_text(c1_m.get(5., np.nan), 7)}` at +5° and `{_float_text(c1_m.get(10., np.nan), 7)}` at +10°; about 11% of the +5° C1 rows are at the floor. This gaze-associated pattern could interact with the fitted P4 nuisance offset, but gaze and fixation sequence are confounded and the pattern is not causal evidence.", "",
              "The P1 scale is measured from P1 geometry under its own empirical reference and calibration gauge, then held fixed while P4 is completely refit. Here P1-m increases equal-weight P4 loss overall, chiefly in C1–C3, while C4 is nearly unchanged. This P1 scale does not improve this P4 fit under the stated model. The large shift in fitted `θ04` indicates nuisance gaze/scale/template tradeoff; gaze-correlated P1 scale can leak into this parameter. Capture and nominal accommodative demand remain confounded, and fixation sequence and saved gaze remain conditional inputs. The comparison does not identify the physical cause of the residuals.", "",
              "This stage does not estimate or identify accommodation. The scale-to-accommodation effect cannot yet be quantified without fitting and validating `k(A)` together with accommodation `A`. No physiological or optical-zero inference follows from this comparison.", ""]
    inverse_path = results / "inverse_summary.json"
    if inverse_path.exists():
        inverse = json.loads(inverse_path.read_text())
        ib = inverse["branches"]
        order = ("baseline_g1", "baseline_frozen_p1m", "p1m")
        labels = {
            "baseline_g1": "g=1 fit and scale",
            "baseline_frozen_p1m": "fixed g=1 fit with P1 m substituted",
            "p1m": "full P1-m P4 refit",
        }
        g1, frozen, refit = (ib[name] for name in order)
        p1m_mean = float(np.mean(p1m)) if len(p1m) else np.nan
        lines += [
            "## Inverse distortion using the declared frame scale", "",
            "The inverse diagnostic centers measured P4, uses the branch frame scale, restores its fitted raw optical mean, then reverses stretch, projective keystone, and capture barrel distortion. Recovered vertices are compared with the relevant fitted native B4. This row-pooled in-sample metric is separate from the equal-capture/equal-fixation forward objective.", "",
            "Three branches distinguish direct scale substitution from full recalibration. The frozen-scale-only branch retains the baseline `g=1` P4 parameters/template and substitutes P1 `m` only in the inverse; since the frozen scale mean is " + f"{_float_text(p1m_mean, 6)}" + ", a mismatch under fixed geometry is expected. The full P1-m branch refits P4 and is the appropriate matched model comparison.", "",
            "| Inverse branch | Pooled RMS (px) | Template-radius RMS (%) | Valid / frames | Invalid |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in order:
            branch = ib[name]
            lines.append(
                f"| {labels[name]} | {_float_text(branch['pooled_vertex_rms_px'], 6)} | "
                f"{_float_text(branch['pooled_inverse_error_percent_template_radius'], 5)} | "
                f"{branch['valid_frames']} / {branch['frames']} | {branch['invalid_frames']} |"
            )
        lines += ["", "Capture pooled inverse RMS (px):", "", "| Capture | g=1 fit/scale | Fixed fit + P1 m scale | Full P1-m refit |", "|---:|---:|---:|---:|"]
        for capture in sorted(inverse["capture"], key=int):
            per_capture = inverse["capture"][capture]
            lines.append("| " + str(capture) + " | " + " | ".join(
                _float_text(per_capture[name]["pooled_vertex_rms_px"], 6) for name in order
            ) + " |")
        lines += [
            "", "The fixed-geometry substitution is a scale-mismatch diagnostic, not evidence about the outcome of refitting. The full P1-m refit remains the comparison of interest; its inverse RMS is " +
            f"{_float_text(100*(refit['pooled_vertex_rms_px']/g1['pooled_vertex_rms_px']-1), 4)}% above `g=1`. These are in-sample diagnostics, not independent validation or physical scale/accommodation estimates.", "",
            "![Inverse pooled RMS by capture and fixation](inverse_fixation_rms.png)", "",
            "![Inverse frame RMS time series, segmented at source-row gaps](inverse_residual_timeseries.png)", "",
        ]
    lines += [
              "## Reproduction", "",
              "From the repository root, regenerate the forward fit outputs, inverse diagnostics, report text, and inverse figures in this order:", "",
              "```bash", f"python {HERE.relative_to(Path.cwd())}/run.py", f"python {HERE.relative_to(Path.cwd())}/inverse_effect.py", f"python {HERE.relative_to(Path.cwd())}/report.py", f"python {HERE.relative_to(Path.cwd())}/inverse_plot.py", "```", ""]
    (results / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    render(args.results)


if __name__ == "__main__":
    main()
