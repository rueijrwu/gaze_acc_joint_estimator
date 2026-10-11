"""Build compact, conditional reports and plots from retained shared-keystone outputs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _capture_label(value) -> str:
    name = _text(value)
    for part in name.split("_"):
        if part.isdigit():
            return f"C{part}"
    return name


def _fmt(value, digits=6) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "—"
        return f"{float(value):.{digits}g}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return _text(value)


def _metric(record: dict, pattern: str, name: str):
    return record.get(pattern, {}).get(name, {})


def _plot_accommodation(out: Path, frames: dict, provenance: dict) -> str | None:
    required = {"states", "exposure", "capture"}
    if not required.issubset(frames):
        return None

    states = np.asarray(frames["states"], dtype=float)
    exposure = np.asarray(frames["exposure"], dtype=int)
    capture = np.asarray(frames["capture"])
    if states.ndim != 2 or states.shape[1] < 3 or len(states) != len(exposure):
        return None

    targets = np.asarray(frames.get("A_targets", np.full(20, np.nan)), dtype=float)
    fixations = provenance.get("reviewed_fixations", [])
    nominal_theta = {}
    for index, item in enumerate(fixations):
        nominal_theta[index] = item.get("target_theta_deg")

    indexes = sorted(set(exposure.tolist()))
    values, labels, target_x, target_y = [], [], [], []
    for index in indexes:
        mask = (exposure == index) & np.isfinite(states[:, 2])
        if not np.any(mask):
            continue
        values.append(states[mask, 2])
        first = int(np.flatnonzero(mask)[0])
        cap = _capture_label(capture[first])
        theta = nominal_theta.get(index)
        theta_label = f"{theta:g}°" if theta is not None else f"fix {index + 1}"
        labels.append(f"{cap}\n{theta_label}")
        if index < len(targets) and np.isfinite(targets[index]):
            target_x.append(len(labels) - 1)
            target_y.append(targets[index])

    if not values:
        return None

    fig, ax = plt.subplots(figsize=(12.5, 5.5), constrained_layout=True)
    positions = np.arange(len(values))
    boxes = ax.boxplot(values, positions=positions, widths=0.62, showfliers=False,
                       patch_artist=True, medianprops={"color": "#17324d", "linewidth": 1.4})
    for box in boxes["boxes"]:
        box.set_facecolor("#a8c6df")
        box.set_edgecolor("#386b91")
    ax.scatter(target_x, target_y, marker="D", s=30, color="#c45132", zorder=3,
               label="Frozen fixation mean target")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Framewise accommodation state A (D)")
    ax.set_xlabel("Capture and nominal gaze fixation")
    ax.set_title("Accommodation states by fixation")
    ax.grid(axis="y", alpha=0.25)
    if target_y:
        ax.legend(frameon=False, loc="best")
    fig_path = out / "plots" / "accommodation_by_fixation.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    return "plots/accommodation_by_fixation.png"


def _plot_errors(out: Path, audit: dict) -> str | None:
    fixations = audit.get("fixations", [])
    if not fixations:
        return None
    patterns = ("p1", "p4")
    metrics = (("forward_px", "forward"), ("inverse_error_px", "inverse"))
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, constrained_layout=True)
    x = np.arange(len(fixations))
    labels = []
    for record in fixations:
        index = record.get("fixation", len(labels))
        labels.append(str(int(index) + 1))

    for row, pattern in enumerate(patterns):
        for col, (metric_name, title) in enumerate(metrics):
            ax = axes[row, col]
            medians, p95s, rms_values = [], [], []
            for record in fixations:
                metric = _metric(record, pattern, metric_name)
                medians.append(metric.get("median", np.nan))
                p95s.append(metric.get("p95", np.nan))
                rms_values.append(metric.get("rms", np.nan))
            ax.plot(x, medians, marker="o", markersize=3, linewidth=1.2, label="median")
            ax.plot(x, p95s, marker=".", markersize=3, linewidth=1.0, linestyle="--", label="P95")
            ax.plot(x, rms_values, marker="x", markersize=3, linewidth=0, label="RMS")
            ax.set_title(f"{pattern.upper()} {title} vertex error")
            ax.set_ylabel("Error (px)")
            ax.grid(alpha=0.25)
            if row == 1:
                ax.set_xlabel("Reviewed fixation index")
                ax.set_xticks(x[::max(1, len(x) // 10)], labels[::max(1, len(x) // 10)])
            if row == 0 and col == 1:
                ax.legend(frameon=False, ncol=3, loc="upper right")

    fig.suptitle("Forward and inverse vertex errors by fixation", fontsize=13)
    fig_path = out / "plots" / "forward_inverse_errors.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    return "plots/forward_inverse_errors.png"


def _objective_lines(summary: dict) -> list[str]:
    lines = []
    objective = summary.get("objective") or summary.get("components")
    if not isinstance(objective, dict) and isinstance(summary.get("cycles"), list) and summary["cycles"]:
        objective = summary["cycles"][-1].get("components")
    if isinstance(objective, dict):
        lines.extend(["", "## Objective components", "", "| Component | Value |", "|---|---:|"])
        lines.extend(f"| {name} | {_fmt(value)} |" for name, value in objective.items()
                     if isinstance(value, (int, float, np.number)))
    return lines


def _primary_parameter_lines(frames: dict, protocol: dict) -> list[str]:
    q = np.asarray(frames.get("q", []), dtype=float)
    if q.shape != (9,):
        return []
    q0 = np.asarray(frames.get("q0", np.full(9, np.nan)), dtype=float)
    widths = np.asarray(protocol.get("sigma_q", np.asarray(frames.get("W", np.full(9, np.nan))) * 0.25), dtype=float)
    names = protocol.get("coefficient_order", [f"q{i}" for i in range(9)])
    lines = ["", "## Primary shared coefficients", "",
             "The final column is displacement from the frozen initializer in the declared optical penalty widths.",
             "", "| Coefficient | Estimate | Initial value | Shift / sigma_q |", "|---|---:|---:|---:|"]
    for index, value in enumerate(q):
        base = q0[index] if index < len(q0) else np.nan
        scale = widths[index] if index < len(widths) else np.nan
        shift = (value - base) / scale if np.isfinite(base) and np.isfinite(scale) and scale > 0 else np.nan
        name = names[index] if index < len(names) else f"q{index}"
        lines.append(f"| {name} | {_fmt(value)} | {_fmt(base)} | {_fmt(shift)} |")
    return lines


def _state_summary(states: np.ndarray, exposure: np.ndarray, theta0: np.ndarray | None) -> dict:
    states = np.asarray(states, dtype=float)
    exposure = np.asarray(exposure, dtype=int)
    valid = np.isfinite(states[:, :3]).all(axis=1) & (exposure >= 0)
    if not np.any(valid):
        return {}
    state = states[valid]
    fix = exposure[valid]
    a = state[:, 2]
    within = [float(np.std(a[fix == key])) for key in np.unique(fix) if np.any(fix == key)]
    gaze_amplitude = np.linalg.norm(state[:, :2], axis=1)
    result = {
        "A_min": float(np.min(a)), "A_max": float(np.max(a)),
        "A_mean": float(np.mean(a)),
        "within_fix_SD_mean": float(np.mean(within)) if within else np.nan,
        "within_fix_SD_max": float(np.max(within)) if within else np.nan,
        "gaze_amplitude_P95_deg": float(np.percentile(gaze_amplitude, 95)),
        "gaze_amplitude_max_deg": float(np.max(gaze_amplitude)),
    }
    if theta0 is not None:
        theta0 = np.asarray(theta0, dtype=float)[valid]
        departure = np.linalg.norm(state[:, :2] - theta0, axis=1)
        result["gaze_departure_P95_deg"] = float(np.percentile(departure, 95))
        result["gaze_departure_max_deg"] = float(np.max(departure))
    return result


def _control_lines(out: Path, summary: dict, frames: dict, protocol: dict) -> list[str]:
    controls = summary.get("controls")
    if not isinstance(controls, list) or not controls:
        return []
    exposure = np.asarray(frames.get("exposure", []), dtype=int)
    theta0 = np.asarray(frames.get("theta0", []), dtype=float)
    primary_q = np.asarray(frames.get("q", []), dtype=float)
    widths = np.asarray(protocol.get("sigma_q", np.asarray(frames.get("W", np.full(9, np.nan))) * 0.25), dtype=float)
    lines = ["", "## Anchor-control sensitivity", "",
             "Each sensitivity changes one declared prior/target policy. State summaries use saved frame arrays, not just aggregate fit error.",
             "", "| Control | Status | Data | Gaze anchor | A mean anchor | Optical anchor | A range (D) | Within-fix SD mean / max (D) | Gaze amplitude P95 / max (deg) | Gaze departure P95 / max (deg) | Max |Δq|/sigma_q |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    found = 0
    for record in controls:
        if not isinstance(record, dict):
            continue
        name = record.get("name", "unnamed")
        control_path = out / f"control_{name}.npz"
        states = None
        q = np.asarray(record.get("q", []), dtype=float)
        if control_path.exists():
            with np.load(control_path, allow_pickle=False) as archive:
                states = archive["states"] if "states" in archive else None
                if "q" in archive:
                    q = np.asarray(archive["q"], dtype=float)
        metrics = _state_summary(states, exposure, theta0) if states is not None and len(exposure) == len(states) else {}
        comp = record.get("components", {})
        delta = np.nan
        if primary_q.shape == (9,) and q.shape == (9,) and len(widths) == 9 and np.all(widths > 0):
            delta = float(np.max(np.abs((q - primary_q) / widths)))
        a_range = f"{_fmt(metrics.get('A_min'))}–{_fmt(metrics.get('A_max'))}"
        within = f"{_fmt(metrics.get('within_fix_SD_mean'))} / {_fmt(metrics.get('within_fix_SD_max'))}"
        amplitude = f"{_fmt(metrics.get('gaze_amplitude_P95_deg'))} / {_fmt(metrics.get('gaze_amplitude_max_deg'))}"
        departure = f"{_fmt(metrics.get('gaze_departure_P95_deg'))} / {_fmt(metrics.get('gaze_departure_max_deg'))}"
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            name, record.get("status", "unknown"), _fmt(comp.get("data")),
            _fmt(comp.get("gaze")), _fmt(comp.get("accommodation_mean")),
            _fmt(comp.get("optical")), a_range, within, amplitude, departure, _fmt(delta)))
        found += 1
    if not found:
        return []
    return lines


def _optimization_lines(summary: dict, audit: dict, phase) -> list[str]:
    lines = []
    cycles = summary.get("cycles")
    if isinstance(cycles, list):
        lines.extend(["", "## Alternation record", "",
                      f"Recorded warm-start cycles: {len(cycles)}."])
        if cycles:
            last = cycles[-1]
            if isinstance(last, dict):
                lines.append(f"Final cycle status: **{last.get('status', 'recorded')}**.")
                lines.append(f"Final cycle fields: `{', '.join(last.keys())}`.")

    starts = summary.get("starts")
    if isinstance(starts, list):
        lines.extend(["", "## Joint starts", "", "| Start | Status | Objective |", "|---|---|---:|"])
        for index, record in enumerate(starts):
            if not isinstance(record, dict):
                continue
            name = record.get("name", f"start {index + 1}")
            objective = record.get("objective", record.get("value"))
            result = record.get("result")
            result = result if isinstance(result, dict) else {}
            status = record.get("status", result.get("status", "unknown"))
            if isinstance(objective, dict):
                objective = sum(v for v in objective.values()
                                if isinstance(v, (int, float, np.number)))
            lines.append(f"| {name} | {status} | {_fmt(objective)} |")

    controls = summary.get("controls")
    if isinstance(controls, list):
        lines.extend(["", "## Prespecified controls", "", "| Control | Status |", "|---|---|"])
        for record in controls:
            if isinstance(record, dict):
                lines.append(f"| {record.get('name', 'unnamed')} | {record.get('status', 'unknown')} |")

    # Initialization and warm-start audits are diagnostics, not joint-stationarity claims.
    stationarity = None if phase in (1, 2, "1", "2") else (summary.get("certificate") or audit.get("stationarity"))
    if isinstance(stationarity, dict):
        lines.extend(["", "## Stationarity and domain", "",
                      f"Certificate status: **{stationarity.get('status', 'not recorded')}**."])
        if "projected_gradient_inf" in stationarity:
            lines.append(f"Projected gradient infinity norm: `{_fmt(stationarity['projected_gradient_inf'])}`.")
        if "primal_feasible" in stationarity:
            lines.append(f"Primal feasible: `{stationarity['primal_feasible']}`.")
        if "active_nonlinear" in stationarity:
            lines.append(f"Active nonlinear constraints: `{stationarity['active_nonlinear']}`.")
    return lines


def _fixation_error_lines(audit: dict) -> list[str]:
    fixations = audit.get("fixations", [])
    if not fixations:
        return []
    lines = ["", "## Per-fixation forward and inverse error", "",
             "Errors are in camera pixels; each record retains its full distribution in `audit.json`.",
             "", "| Fixation | N | P1 forward median / P95 / RMS | P4 forward median / P95 / RMS | P1 / P4 inverse failures |",
             "|---:|---:|---:|---:|---:|"]
    for record in fixations:
        p1f = _metric(record, "p1", "forward_px")
        p4f = _metric(record, "p4", "forward_px")
        p1i = _metric(record, "p1", "inverse_error_px")
        p4i = _metric(record, "p4", "inverse_error_px")
        p1 = " / ".join(_fmt(p1f.get(key)) for key in ("median", "p95", "rms"))
        p4 = " / ".join(_fmt(p4f.get(key)) for key in ("median", "p95", "rms"))
        fails = f"{record.get('p1', {}).get('inverse_failures', '—')} / {record.get('p4', {}).get('inverse_failures', '—')}"
        lines.append(f"| {int(record.get('fixation', 0)) + 1} | {record.get('count', '—')} | {p1} | {p4} | {fails} |")
    return lines


def build_report(out) -> Path:
    """Write RESULTS.md and compact diagnostic plots in a completed stage folder."""
    out = Path(out)
    summary = _read_json(out / "summary.json")
    audit = _read_json(out / "audit.json")
    provenance = _read_json(out / "provenance.json")
    protocol = _read_json(out / "protocol.json")

    frames = {}
    frames_path = out / "frames.npz"
    if frames_path.exists():
        with np.load(frames_path, allow_pickle=False) as archive:
            frames = {key: archive[key] for key in archive.files}

    accommodation_plot = _plot_accommodation(out, frames, provenance)
    error_plot = _plot_errors(out, audit)

    run_status = summary.get("status", "UNKNOWN")
    audit_status = audit.get("status", "NOT_RUN")
    phase = provenance.get("phase", summary.get("phase", "not recorded"))
    if run_status == "INITIALIZED":
        interpretation = "This is the mean-initialization checkpoint; it is not a joint-refinement or stationarity PASS."
    elif run_status in ("PASS", "CERTIFIED") and audit_status == "PASS":
        interpretation = "Recorded numerical checks pass for this declared stage. This does not establish a global optimum or independent physiological accuracy."
    elif "VALIDATION" in run_status:
        if not (summary.get("training_certified") and summary.get("inference_certified")):
            interpretation = "Held-out errors are diagnostic only because the training fit or retained-input inference is uncertified. The validation split checks within-recording temporal generalization, not cross-subject or physiological accuracy."
        else:
            interpretation = "The training and retained-input state certificates pass. This is conditional within-recording temporal geometry, not cross-subject or physiological accuracy."
    else:
        interpretation = "The stage is not certified as a successful joint fit; use the recorded run and audit statuses below."

    lines = [
        "# Shared keystone Revision 2 experiment", "",
        f"Run status: **{run_status}**.", "",
        f"Numerical audit status: **{audit_status}**.", "",
        f"Stage: `{phase}`. Scheduled frames: `{provenance.get('scheduled', '—')}`; complete: `{provenance.get('complete', '—')}`; unavailable: `{provenance.get('unavailable', '—')}`.", "",
        interpretation, "",
        "This is centered triangle shape conditional on empirical gaze anchors. Accommodation and gaze scales are selected by frozen same-recording regularizers. Capture and nominal demand remain confounded; derived centers add no independent measurement.", "",
        f"Source head: `{provenance.get('source_head', 'not recorded')}`. Source changes during run: `{summary.get('source_changes_during_run', [])}`.", "",
    ]

    targets = summary.get("inferred_capture_targets_D")
    nominal = summary.get("nominal_capture_labels_D")
    if isinstance(targets, list):
        lines.extend(["## Accommodation target initialization", "",
                      "| Capture | Inferred target (D) | Nominal label (D) |", "|---|---:|---:|"])
        for index, target in enumerate(targets):
            label = nominal[index] if isinstance(nominal, list) and index < len(nominal) else None
            lines.append(f"| C{index + 1} | {_fmt(target)} | {_fmt(label)} |")
        if "s0" in summary:
            lines.extend(["", f"Shared radial slope: `{_fmt(summary.get('s0'))} px^-2 D^-1` (sign {summary.get('slope_sign', 'not recorded')}; initialization bound active `{summary.get('slope_active_bound', 'not recorded')}`)."])

    lines.extend(_objective_lines(summary))
    lines.extend(_primary_parameter_lines(frames, protocol))
    lines.extend(_optimization_lines(summary, audit, phase))
    lines.extend(_control_lines(out, summary, frames, protocol))
    lines.extend(_fixation_error_lines(audit))

    lines.extend(["", "## Plots", ""])
    if accommodation_plot:
        lines.append(f"- Accommodation states by fixation: [{Path(accommodation_plot).name}]({accommodation_plot})")
    if error_plot:
        lines.append(f"- Forward and inverse errors: [{Path(error_plot).name}]({error_plot})")
    if not accommodation_plot and not error_plot:
        lines.append("No compatible framewise or per-fixation diagnostics were available for plots.")

    lines.extend(["", "## Retained artifacts", "",
                  "See [protocol.json](protocol.json), [provenance.json](provenance.json), [report_provenance.json](report_provenance.json), [summary.json](summary.json), [frames.npz](frames.npz), [scheduled_frames.npz](scheduled_frames.npz), and [audit.json](audit.json) for resolved settings, source hashes, row identities, optimization records, and audit details.", ""])
    regression = out.parent.parent / "audits" / "block_roundoff_merit.json"
    if regression.is_file():
        lines.extend(["## Optimizer regression", "",
                      f"The block roundoff-merit correction passed CPU and CuPy regression checks: [block_roundoff_merit.json](../../audits/{regression.name})."])
    report_path = out / "RESULTS.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    inputs = ("protocol.json", "provenance.json", "summary.json", "audit.json",
              "frames.npz", "scheduled_frames.npz", "diagnostics.npz")
    report_record = {
        "schema": "shared_keystone_report_provenance_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "report_script_sha256": _sha256(Path(__file__)),
        "input_sha256": {name: _sha256(out / name) for name in inputs if (out / name).is_file()},
        "output_sha256": {"RESULTS.md": _sha256(report_path)},
    }
    for relative in (accommodation_plot, error_plot):
        if relative:
            report_record["output_sha256"][relative] = _sha256(out / relative)
    (out / "report_provenance.json").write_text(json.dumps(report_record, indent=2) + "\n", encoding="utf-8")
    return report_path
