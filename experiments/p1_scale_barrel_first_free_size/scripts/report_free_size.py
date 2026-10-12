"""Post-run reporting for the controlled shared P4 template-size comparison."""
from __future__ import annotations

import csv
import json
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
OUT = FAMILY / "report"
PAIRS = [
    ("sigma050_optical2", "fixed_size_sigma050_optical2", "free_size_sigma050_optical2", "Primary (.50 D, optical 2×)"),
    ("sigma025_optical1", "fixed_size_sigma025_optical1", "free_size_sigma025_optical1", "Robustness (.25 D, optical 1×)"),
]


def jread(path):
    return json.loads(Path(path).read_text())


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def metric(summary, name):
    obj = summary.get(name, {})
    return obj.get("weighted_vertex_rms_px"), obj.get("pooled_vertex_rms_px")


def fmt(value, digits=6):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def csvrows(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def dump_csv(path, rows):
    if rows:
        with Path(path).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def geometry_metrics(residual, group, valid):
    residual = np.asarray(residual)
    group = np.asarray(group)
    error = np.mean(np.sum(residual * residual, axis=-1), axis=-1)
    ok = np.asarray(valid, bool) & np.isfinite(error)
    means = [float(np.mean(error[(group == g) & ok])) for g in np.unique(group) if np.any((group == g) & ok)]
    return {
        "weighted_vertex_rms_px": float(np.sqrt(np.mean(means))) if means else None,
        "pooled_vertex_rms_px": float(np.sqrt(np.mean(error[ok]))) if np.any(ok) else None,
        "scored_rows": int(ok.sum()),
        "invalid_rows": int((~ok).sum()),
    }


def make_series(source_frame, source_row, group, values):
    order = np.lexsort((source_row, source_frame))
    x = np.asarray(source_frame)[order]
    g = np.asarray(group)[order]
    y = np.asarray(values)[order]
    xx, yy = [], []
    for i in range(len(x)):
        if i and (g[i] != g[i - 1] or x[i] != x[i - 1] + 1):
            xx.append(np.nan)
            yy.append(np.nan)
        xx.append(float(x[i]))
        yy.append(float(y[i]))
    return np.asarray(xx), np.asarray(yy)


def load_case(branch, held=False):
    if held:
        if branch.startswith("fixed_size_"):
            base = FAMILY / "inputs" / "fixed_controls" / "held" / branch
        else:
            base = FAMILY / "validation" / "temporal" / branch
    elif branch.startswith("fixed_size_"):
        base = FAMILY / "inputs" / "fixed_controls" / "production" / branch
    else:
        base = FAMILY / "results" / "production" / branch
    # Per-bundle fixation_metrics.csv files are archived as duplicate derived
    # outputs. The canonical merged table remains under report/ and supplies
    # the cell-level rows used for presentation. Arrays and summaries continue
    # to be loaded from each immutable result bundle.
    if held:
        cells = []
    else:
        pair = next((label for label, fixed, free, _ in PAIRS if branch in (fixed, free)), None)
        if pair is None:
            raise ValueError(f"Unknown report branch: {branch}")
        model = "fixed" if branch.startswith("fixed_size_") else "free"
        cells = [
            row for row in csvrows(FAMILY / "report" / "fixation_metrics.csv")
            if row.get("pair") == pair and row.get("model") == model
        ]
        if len(cells) != 20:
            raise ValueError(f"Expected 20 canonical fixation rows for {pair}/{model}, found {len(cells)}")
    return jread(base / "summary.json"), arrays(base / "arrays.npz"), cells


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ids = arrays(FAMILY / "inputs" / "production" / "stage_01_gaze_initialization" / "arrays.npz")
    reviewed = np.asarray(ids.get("fit_mask", np.ones_like(ids["source_row"], bool)), bool)
    rows_out, cases = [], {}
    for label, fixed_name, free_name, title in PAIRS:
        fixed, free = load_case(fixed_name), load_case(free_name)
        cases[label] = (fixed, free, title)
        for model_name, branch_name, (summary, _, _) in (("fixed", fixed_name, fixed), ("free", free_name, free)):
            fw4, poolfw4 = metric(summary, "all_forward4")
            inv4, poolinv4 = metric(summary, "all_inverse4")
            tg = summary.get("template_gauge", {})
            cert = summary.get("selected_certificate", {})
            rows_out.append({
                "pair": label, "model": model_name, "branch": branch_name,
                "template_size_ratio": tg.get("template_size_ratio", 1.0),
                "learned_centered_energy_px2": tg.get("learned_centered_energy_px2"),
                "p4_forward_weighted_rms_px": fw4, "p4_forward_pooled_rms_px": poolfw4,
                "p4_inverse_weighted_rms_px": inv4, "p4_inverse_pooled_rms_px": poolinv4,
                "invalid_forward_rows": summary.get("all_forward_invalid_rows"),
                "invalid_inverse_rows": summary.get("all_inverse4_invalid_rows"),
                "certified_starts": summary.get("certified_starts"),
                "global_stationary": cert.get("global_stationary"),
                "state_stationary": cert.get("state_stationary"),
                "size_bound_contact": tg.get("size_bound_contact", False),
                "A_lower_contacts": cert.get("A_lower_contacts"),
                "A_upper_contacts": cert.get("A_upper_contacts"),
                "data_only_rank": (
                    f"{summary['data_only_rank'].get('data_only_rank_at_relative_1e_minus6')}/"
                    f"{summary['data_only_rank'].get('dimension')}"
                    if isinstance(summary.get("data_only_rank"), dict)
                    else summary.get("data_only_rank")
                ),
                "global_projected_gradient_scaled_inf": cert.get("global_projected_gradient_scaled_inf"),
                "state_projected_gradient_characteristic_inf": cert.get("state_projected_gradient_characteristic_inf"),
                "slope": summary.get("settings", {}).get("slope"),
                "offset_deg": summary.get("settings", {}).get("offset_deg"),
            })
    # Add per-capture accommodation span and the mean-offset versus residual-shape
    # decomposition of the paired fixation-mean A differences.
    contrast_rows = []
    for label, fixed_name, free_name, title in PAIRS:
        _, _, fixed_cells = load_case(fixed_name)
        _, _, free_cells = load_case(free_name)
        fixed_map = {(int(r["capture"]), int(r["fixation"])): float(r["A_mean"]) for r in fixed_cells}
        free_map = {(int(r["capture"]), int(r["fixation"])): float(r["A_mean"]) for r in free_cells}
        for cap in range(1, 5):
            keys = sorted(k for k in fixed_map if k[0] == cap and k in free_map)
            diffs = np.array([free_map[k] - fixed_map[k] for k in keys])
            offset = float(np.mean(diffs)) if len(diffs) else float("nan")
            shape = diffs - offset
            for i, key in enumerate(keys):
                contrast_rows.append({
                    "pair": label, "capture": cap, "fixation": key[1],
                    "fixed_A_mean_D": fixed_map[key], "free_A_mean_D": free_map[key],
                    "free_minus_fixed_A_mean_D": float(diffs[i]),
                    "capture_mean_offset_D": offset,
                    "residual_shape_after_offset_D": float(shape[i]),
                })
            for model, cells in (("fixed", fixed_cells), ("free", free_cells)):
                vals = np.array([float(r["A_mean"]) for r in cells if int(r["capture"]) == cap])
                row = next(r for r in rows_out if r["pair"] == label and r["model"] == model)
                row[f"C{cap}_A_span_D"] = float(vals.max() - vals.min()) if len(vals) else None
                row[f"C{cap}_A_mean_D"] = float(vals.mean()) if len(vals) else None
    # Training-only fitted metrics are recomputed from arrays and the frozen
    # training mask; do not label summary all-row diagnostics as training metrics.
    temporal_ids = arrays(FAMILY / "inputs" / "training" / "stage_01_gaze_initialization" / "arrays.npz")
    train_rows = np.asarray(temporal_ids.get("train_mask", temporal_ids.get("fit_mask")), bool)
    train_metric_rows = []
    for label, fixed_name, free_name, title in PAIRS:
        for model, branch in (("fixed", fixed_name), ("free", free_name)):
            train_dir = (FAMILY / "inputs" / "fixed_controls" / "training" / branch) if model == "fixed" else (FAMILY / "results" / "training" / branch)
            with np.load(train_dir / "arrays.npz", allow_pickle=False) as z:
                a = {k: z[k].copy() for k in z.files}
            residual = a["p4_prediction"] - temporal_ids["X4"]
            gm = geometry_metrics(residual[train_rows], temporal_ids["group"][train_rows], a["valid4"][train_rows])
            train_metric_rows.append({"pair": label, "model": model, **gm})
    # Case-level metric tables were redundant with the immutable summaries and
    # canonical merged report table; they remain in the cleanup archive.

    # Reviewed source frames only; break traces at source-frame gaps and fixation boundaries.
    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, sharey=True, constrained_layout=True)
    colors = {"fixed": "#555555", "free": "#d55e00"}
    for ax, cap in zip(axes, range(1, 5)):
        _, fixed_a, _ = load_case(PAIRS[0][1])
        _, free_a, _ = load_case(PAIRS[0][2])
        for model_name, a in (("fixed", fixed_a), ("free", free_a)):
            m = reviewed & (ids["capture"] == cap)
            x, y = make_series(ids["source_frame"][m], ids["source_row"][m], ids["group"][m], a["A"][m])
            ax.plot(x, y, lw=.65, color=colors[model_name], alpha=.9, label=model_name)
        ax.set_title(f"C{cap} · primary pair")
        ax.set_ylabel("A (D)")
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Original source frame")
    axes[0].legend(loc="upper right")
    fig.suptitle("A over reviewed source frames; traces break at gaps and fixation changes")
    fig.savefig(OUT / "A_over_source_frame_primary.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=True, sharey=True, constrained_layout=True)
    fixed_a, free_a = load_case(PAIRS[1][1])[1], load_case(PAIRS[1][2])[1]
    for ax, cap in zip(axes, range(1, 5)):
        for model_name, a in (("fixed", fixed_a), ("free", free_a)):
            m = reviewed & (ids["capture"] == cap)
            x, y = make_series(ids["source_frame"][m], ids["source_row"][m], ids["group"][m], a["A"][m])
            ax.plot(x, y, lw=.65, color=colors[model_name], alpha=.9, label=model_name)
        ax.set_title(f"C{cap} · robustness pair")
        ax.set_ylabel("A (D)")
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Original source frame")
    axes[0].legend(loc="upper right")
    fig.suptitle("Robustness pair A over reviewed source frames; breaks at gaps and fixation changes")
    fig.savefig(OUT / "A_over_source_frame_robustness.png", dpi=160)
    plt.close(fig)

    # A versus each model's own fitted fixation-mean horizontal gaze. Capture
    # color is stable across both panels; marker/line style identifies size model.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True, sharex=True, sharey=True)
    capture_colors = plt.get_cmap("tab10").colors[:4]
    for ax, (label, fixed_name, free_name, title) in zip(axes, PAIRS):
        for cap, color in zip(range(1, 5), capture_colors):
            for name, branch, marker, linestyle in (
                ("fixed", fixed_name, "o", "-"), ("free", free_name, "s", "--")
            ):
                _, _, fr = load_case(branch)
                cell = sorted((r for r in fr if int(r["capture"]) == cap), key=lambda r: int(r["fixation"]))
                theta = np.array([float(r["theta_mean"]) for r in cell])
                av = np.array([float(r["A_mean"]) for r in cell])
                ax.plot(theta, av, color=color, alpha=.9, lw=1.0, linestyle=linestyle)
                ax.scatter(theta, av, color=color, marker=marker, s=25, label=f"C{cap} {name}")
        ax.set_title(title)
        ax.set_xlabel("Fitted fixation-mean horizontal gaze θ (deg)")
        ax.set_ylabel("Fitted fixation-mean A (D)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=2, title="Capture color; model style")
    fig.suptitle("Gaze-stratified fixation means; fitted θ, not nominal target")
    fig.savefig(OUT / "A_vs_fitted_gaze.png", dpi=160)
    plt.close(fig)

    cell_rows = []
    for label, fixed_name, free_name, title in PAIRS:
        for model, branch in (("fixed", fixed_name), ("free", free_name)):
            _, _, fr = load_case(branch)
            for r in fr:
                cell_rows.append({"pair": label, "model": model, **r})
    dump_csv(OUT / "fixation_metrics.csv", cell_rows)

    # Compare the two jointly changed anchor/optical settings; this is not an
    # isolated anchor-width effect because both settings change together.
    stability_rows = []
    by_model = {}
    for model in ("fixed", "free"):
        branches = [PAIRS[0][1 if model == "fixed" else 2], PAIRS[1][1 if model == "fixed" else 2]]
        by_model[model] = []
        for branch in branches:
            _, _, fr = load_case(branch)
            by_model[model].append({(int(r["capture"]), int(r["fixation"])): float(r["A_mean"]) for r in fr})
        low_setting, high_setting = by_model[model]
        for cap in range(1, 5):
            keys = sorted(k for k in low_setting if k[0] == cap and k in high_setting)
            d = np.array([high_setting[k] - low_setting[k] for k in keys])
            center = float(np.mean(d)) if len(d) else float("nan")
            centered = d - center
            stability_rows.append({
                "model": model, "capture": cap,
                "settings_compared": "sigma050_optical2 minus sigma025_optical1",
                "mean_offset_D": center,
                "max_abs_raw_Amean_change_D": float(np.max(np.abs(d))) if len(d) else None,
                "rms_raw_Amean_change_D": float(np.sqrt(np.mean(d*d))) if len(d) else None,
                "max_abs_after_capture_offset_D": float(np.max(np.abs(centered))) if len(d) else None,
                "rms_after_capture_offset_D": float(np.sqrt(np.mean(centered*centered))) if len(d) else None,
            })
    dump_csv(OUT / "anchor_setting_stability.csv", stability_rows)
    setting_changes = {}
    for model in ("fixed", "free"):
        lo, hi = by_model[model]
        d = np.array([hi[k] - lo[k] for k in sorted(lo) if k in hi])
        centered = []
        for cap in range(1, 5):
            capd = np.array([hi[k] - lo[k] for k in sorted(lo) if k[0] == cap and k in hi])
            centered.extend((capd - np.mean(capd)).tolist())
        setting_changes[model] = {
            "rms": float(np.sqrt(np.mean(d*d))),
            "max_abs": float(np.max(np.abs(d))),
            "offset_removed_rms": float(np.sqrt(np.mean(np.square(centered)))),
            "offset_removed_max_abs": float(np.max(np.abs(centered))),
        }

    contrast_summary = []
    for label, fixed_name, free_name, title in PAIRS:
        for cap in range(1, 5):
            cells = [r for r in contrast_rows if r["pair"] == label and r["capture"] == cap]
            if not cells:
                continue
            values = np.array([r["free_minus_fixed_A_mean_D"] for r in cells])
            offset = float(np.mean(values))
            centered = values - offset
            contrast_summary.append({
                "pair": label, "capture": cap, "mean_offset_D": offset,
                "shape_rms_after_offset_D": float(np.sqrt(np.mean(centered*centered))),
                "shape_max_abs_after_offset_D": float(np.max(np.abs(centered))),
            })

    profile_dir = FAMILY / "validation" / "template_size_profile" / "free_size_sigma050_optical2"
    profile_path = profile_dir / "profile.csv"
    profile_md = "Profile artifact is not present."
    if profile_path.is_file():
        prof = csvrows(profile_path)
        dump_csv(OUT / "primary_size_profile.csv", prof)
        xs = np.array([float(r["size_ratio"]) for r in prof])
        ys = np.array([float(r["loss"]) for r in prof])
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        ax.plot(xs, ys, "o-", color="#0072b2")
        ax.axvline(1, color="#555", ls="--", lw=1, label="fixed-size λ=1")
        ax.set_xlabel("Shared centered-template size ratio λ")
        ax.set_ylabel("Refitted profile loss")
        ax.grid(alpha=.25)
        ax.legend()
        fig.savefig(OUT / "primary_size_profile.png", dpi=160)
        plt.close(fig)
        profile_rows = []
        for r in prof:
            profile_rows.append(
                f"| {r['point']} | {fmt(r['size_ratio'])} | {fmt(r['loss'])} | "
                f"{fmt(r['loss_data'])} | {fmt(r['loss_gaze'])} | {fmt(r['loss_Amean'])} | "
                f"{fmt(r['loss_optical'])} | {fmt(r['loss_barrel_mean_anchor'])} | "
                f"{fmt(r['unconstrained_scaled_size_gradient'])} | "
                f"{r['state_stationary']}/{r['nuisance_global_stationary']} | {r['invalid_rows']} |"
            )
        profile_md = "\n".join([
            "The free-size optimum, the two refitted ±0.5% points, and the exact fixed-size λ=1 objective are shown below. The λ=1 point reproduces the immutable fixed-template objective.",
            "",
            "| Profile point | λ | Total loss | Data | Gaze | A mean | Optical | Barrel mean | Free-size gradient* | State/global stationary | Invalid |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
            *profile_rows,
            "",
            "*At λ=1 and the fixed profile points λ is fixed, so its unconstrained size gradient is diagnostic and is not required to be zero.",
            "",
            "See [profile CSV](primary_size_profile.csv) and [profile plot](primary_size_profile.png).",
        ])

    held_rows = []
    held_ok = True
    for label, fixed_name, free_name, title in PAIRS:
        try:
            for model, branch in (("fixed", fixed_name), ("free", free_name)):
                s, _, _ = load_case(branch, held=True)
                # The Stage 5 joint state prediction is the paired endpoint.
                # Its nested "fixed" metric is the Stage 4 fixed-optics state
                # control, not the matched fixed-size Stage 5 branch.
                v = s["metrics"]["joint"]
                held_rows.append({"pair": label, "model": model, "weighted_rms_px": v.get("weighted_vertex_rms_px"), "pooled_rms_px": v.get("pooled_vertex_rms_px"), "held_rows": v.get("scored_rows"), "invalid_rows": v.get("invalid_rows"), "source_metric": "metrics.joint"})
        except FileNotFoundError:
            held_ok = False
    held_lines = []
    if held_ok:
        for label, _, _, title in PAIRS:
            f = next(r for r in held_rows if r["pair"] == label and r["model"] == "fixed")
            u = next(r for r in held_rows if r["pair"] == label and r["model"] == "free")
            held_lines.append(f"| {title} | {fmt(f['weighted_rms_px'])} | {fmt(u['weighted_rms_px'])} | {fmt(f['pooled_rms_px'])} | {fmt(u['pooled_rms_px'])} |")
    else:
        held_lines.append("| Held results pending | — | — | — | — |")

    audit_json = FAMILY / "audits" / "free_size_output_audit.json"
    held_cell_rows = []
    if audit_json.is_file():
        audit = jread(audit_json)
        for hc in audit.get("held", []):
            pair_label = "sigma050_optical2" if "sigma050" in hc["case"] else "sigma025_optical1"
            for cell in hc.get("capture_fixation_changes", []):
                held_cell_rows.append({
                    "pair": pair_label, "capture": cell["capture"], "fixation": cell["fixation"],
                    "fixed_cell_rms_px": cell["fixed_rms"], "free_cell_rms_px": cell["free_rms"],
                    "free_minus_fixed_px": cell["difference"],
                })
        dump_csv(OUT / "held_cell_changes.csv", held_cell_rows)

    index = [
        "# Free-size comparison artifact index", "",
        "The scientific findings, paired results, gaze-associated A pattern, cell regressions, profile interpretation and limitations are consolidated in the [canonical experiment record](../../RECORD.md) and [current status](../../../CURRENT_STATUS.md). This file indexes the supporting report artifacts; it intentionally does not duplicate the findings tables.", "",
        "## Figures", "",
        "All plots use reviewed fixation rows and preserve source-frame gaps; there is no post-fit residual filtering.", "",
        "![Primary pair: accommodation by original source frame](A_over_source_frame_primary.png)", "",
        "![Robustness pair: accommodation by original source frame](A_over_source_frame_robustness.png)", "",
        "![Fitted accommodation by actual fitted gaze](A_vs_fitted_gaze.png)", "",
        "![Conditional shared-size profile](primary_size_profile.png)", "",
        "## Derived tables", "",
        "- [Capture/fixation held-cell changes](held_cell_changes.csv)",
        "- [Fixation metrics](fixation_metrics.csv)",
        "- [Anchor-setting stability](anchor_setting_stability.csv)",
        "- [Primary profile points](primary_size_profile.csv)", "",
        "## Fit artifacts and audit", "",
        "- [Production free-size fits](../results/production/)",
        "- [Training fits](../results/training/)",
        "- [Fixed-size paired controls](../inputs/fixed_controls/)",
        "- [Held validation](../validation/temporal/)",
        "- [Profile arrays and summary](../validation/template_size_profile/free_size_sigma050_optical2/)",
        "- [Independent audit narrative](../audits/AUDIT.md)",
        "- [Machine-readable independent audit](../audits/free_size_output_audit.json)",
        "- [Audit manifest](../audits/MANIFEST.json)",
        "- [Frozen run plan and imports](../provenance/)", "",
        "To rebuild plots from the saved arrays and refresh retained derived tables, use [`scripts/report_free_size.py`](../scripts/report_free_size.py). The merged fixation table, profile points and held-cell audit are retained as canonical report inputs. Exact fit-time sources are preserved under [frozen sources](../provenance/frozen_sources/); duplicate snapshots and intermediate tables are archived.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(index) + "\n")

if __name__ == "__main__":
    main()
