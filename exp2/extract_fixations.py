"""Extract and plot the five ordered target fixations in captures 1–4 only.

Intervals are approximate, at 100-row block resolution. All valid measurements
inside each full interval are retained; target demands are labels only. No
calibration fitting or capture-5 estimation is performed.
"""

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from scipy.ndimage import median_filter

from observations import measurements


TARGETS = [-15., -7.5, 0., 7.5, 15.]
DEMANDS = [1000 / 2775, 4., 3., 2.]
COLORS = ["#4477aa", "#66ccee", "#228833", "#ccbb44", "#ee6677"]


def extraction_validity(arrays, measurement_valid, require_valid_pupil=False):
    """Apply the stored pupil flag; this does not rerun native detection."""
    valid = np.asarray(measurement_valid, dtype=bool).copy()
    if require_valid_pupil:
        pupil = np.asarray(arrays["pupil_valid"])
        if pupil.shape != valid.shape or pupil.dtype != np.bool_:
            raise ValueError("Stored pupil_valid must be a boolean array matching measurements")
        valid &= pupil
    return valid


def block_medians(y, valid, block_size=100):
    """Return medians of valid detections; sparse blocks remain NaN for review."""
    starts = np.arange(0, len(y), block_size)
    medians = np.full((len(starts), 2), np.nan)
    for j, start in enumerate(starts):
        end = min(start + block_size, len(y))
        good = valid[start:end]
        if good.sum() >= (end - start) / 5:
            medians[j] = np.median(y[start:end][good], axis=0)
    return starts, medians


def find_fixations(y, valid, block_size=100):
    """Resolve longest stable runs in the known descending e_n target sequence.

    Freeview can occur anywhere outside the selected runs. Five median clusters
    assign plateau candidates; a five-block median filter suppresses short
    excursions. This assumes distinct plateaus and the supplied once-through
    protocol, and therefore still requires visual boundary review.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")
    _, medians = block_medians(y, valid, block_size)
    trace = medians[:, 0].copy()
    finite = np.isfinite(trace)
    if finite.sum() < 25:
        raise ValueError("Too few valid blocks to find five fixations")
    trace[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), trace[finite])
    trace = median_filter(trace, size=5)
    centers = np.quantile(trace, [.1, .3, .5, .7, .9])
    for _ in range(100):
        labels = np.argmin(np.abs(trace[:, None] - centers), axis=1)
        if any(not np.any(labels == j) for j in range(5)):
            raise ValueError("Five distinct gaze plateaus could not be resolved")
        update = np.array([np.median(trace[labels == j]) for j in range(5)])
        if np.allclose(update, centers, rtol=0, atol=1e-10):
            break
        centers = update
    centers = np.sort(centers)[::-1]
    labels = median_filter(np.argmin(np.abs(trace[:, None] - centers), axis=1), size=5)
    edges = np.r_[0, np.flatnonzero(np.diff(labels)) + 1, len(labels)]
    runs = []
    for j in range(5):
        choices = [(b-a, a, b) for a, b in zip(edges[:-1], edges[1:]) if labels[a] == j]
        if not choices:
            raise ValueError("A target plateau is missing")
        _, start, end = max(choices)
        run = (int(start * block_size), min(int(end * block_size), len(y)))
        if run[1] - run[0] < 2000:
            raise ValueError("Resolved fixation is too short for the stated protocol")
        runs.append(run)
    if any(runs[j][1] > runs[j+1][0] for j in range(4)):
        raise ValueError("Fixations are not once in ascending target order")
    return runs, centers


def summarize(arrays, y, valid, runs, capture, demand, core_margin=0):
    """Full interval counts/statistics plus a separately identified optional core."""
    frame = np.asarray(arrays["frame_index"])
    time = np.asarray(arrays["timestamp_ms"], dtype=float)
    if len(time) != len(y) or not np.isfinite(time).all():
        raise ValueError("Timestamps must be finite and match the rows")
    if len(frame) != len(y) or np.any(np.diff(frame) <= 0):
        raise ValueError("Frame indices must increase and match the rows")
    if core_margin < 0:
        raise ValueError("Core margin must be nonnegative")
    rows = []
    for theta, (start, end) in zip(TARGETS, runs):
        good = valid[start:end]
        median = np.median(y[start:end][good], axis=0) if good.any() else [np.nan, np.nan]
        row = {
            "capture": capture, "target_theta_deg": theta, "demand_diopters_label": demand,
            "start_row": start, "end_row_exclusive": end, "row_count": end-start,
            "first_frame": int(frame[start]), "last_frame_inclusive": int(frame[end-1]),
            "first_timestamp_ms": float(time[start]), "last_timestamp_ms": float(time[end-1]),
            "duration_ms_first_to_last": float(time[end-1]-time[start]),
            "timestamp_backward_jumps_in_interval": int(np.sum(np.diff(time[start:end]) < 0)),
            "timestamp_duration_reliable": bool(np.all(np.diff(time[start:end]) >= 0)),
            "frame_span_inclusive": int(frame[end-1]-frame[start]+1),
            "full_valid_count": int(good.sum()), "full_invalid_count": int((~good).sum()),
            "full_median_e_n": float(median[0]), "full_median_rho4": float(median[1]),
            "ends_at_recording_eof": end == len(y),
        }
        if core_margin:
            # No incoming/outgoing transition is observed beyond a recording edge.
            core_start = min(end, start + core_margin) if start > 0 else start
            core_end = max(core_start, end-core_margin) if end < len(y) else end
            row.update(core_start_row=core_start, core_end_row_exclusive=core_end,
                       core_valid_count=int(valid[core_start:core_end].sum()))
        rows.append(row)
    return rows


def frame_edge(frame, row):
    """Map an exclusive row boundary to the next frame or an extrapolated EOF."""
    return frame[row] if row < len(frame) else frame[-1] + np.median(np.diff(frame))


def plot_capture(axes, arrays, y, valid, summaries, block_size, compact=False):
    frame = np.asarray(arrays["frame_index"])
    starts, medians = block_medians(y, valid, block_size)
    midpoints = np.array([(frame[start] + frame[min(start+block_size, len(frame))-1])/2
                          for start in starts])
    for j, axis in enumerate(axes):
        axis.set_facecolor("#eeeeee")
        raw = np.where(valid, y[:, j], np.nan)
        axis.plot(frame, raw, color="#555555", alpha=.32, linewidth=.45, label="Raw valid measurements")
        axis.plot(midpoints, medians[:, j], color="#111111", linewidth=1, label=f"{block_size}-row median")
        for color, row in zip(COLORS, summaries):
            first, last = frame_edge(frame, row["start_row"]), frame_edge(frame, row["end_row_exclusive"])
            axis.axvspan(first, last, color=color, alpha=.22)
            axis.axvline(first, color=color, linewidth=1)
            axis.axvline(last, color=color, linewidth=1)
            label = f"{row['target_theta_deg']:g}°"
            if not compact:
                label += f"\n[{row['start_row']}, {row['end_row_exclusive']})\nvalid {row['full_valid_count']:,}"
            axis.text((first+last)/2, .97, label, transform=axis.get_xaxis_transform(),
                      ha="center", va="top", fontsize=8, backgroundcolor="white")
        axis.set_ylabel(["e_n = (mean P1x − mean P4x) / |ΔP1x|", "rho4 = |ΔP4x| / |ΔP1x|"][j]
                        if not compact else ["e_n", "rho4"][j])
        axis.set_xlim(frame[0], frame_edge(frame, len(frame)))
        axis.grid(alpha=.2)
    axes[-1].set_xlabel("Original frame index; shaded boundaries inferred at block resolution")


def main():
    experiment = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=experiment)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new review directory; existing frozen intervals are preserved")
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--core-margin", type=int, default=0,
                        help="optional separate core counts; never changes full intervals/counts")
    args = parser.parse_args()
    if args.block_size < 1 or args.core_margin < 0:
        parser.error("Block size must be positive and core margin nonnegative")
    if (args.output_dir / "fixation_intervals.json").exists():
        parser.error("Output contains frozen intervals; choose a new review directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries, recordings, sources = [], [], []
    for capture, demand in enumerate(DEMANDS, 1):
        path = args.experiment_dir / f"capture_{capture}_detections.pkl"
        payload = path.read_bytes()
        arrays = pickle.loads(payload)["arrays"]
        y, measurement_valid = measurements(arrays, "absolute_x")
        valid = extraction_validity(arrays, measurement_valid, require_valid_pupil=True)
        sources.append({"capture": path.name, "sha256": hashlib.sha256(payload).hexdigest(),
                        "row_count": len(valid), "measurement_valid_count": int(measurement_valid.sum()),
                        "extraction_valid_count": int(valid.sum()),
                        "excluded_by_pupil_gate_count": int((measurement_valid & ~valid).sum())})
        runs, _ = find_fixations(y, valid, args.block_size)
        summaries = summarize(arrays, y, valid, runs, path.name, demand, args.core_margin)
        all_summaries.extend(summaries)
        recordings.append((arrays, y, valid, summaries))
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, layout="constrained")
        plot_capture(axes, arrays, y, valid, summaries, args.block_size)
        fig.suptitle(f"Capture {capture}: five ordered fixations; demand label {demand:.4g} D\n"
                     "Stored pupil gate applied; "
                     "Full accepted data retained; gray = outside selected sequence")
        axes[0].legend(loc="lower left", fontsize=8)
        fig.savefig(args.output_dir / f"capture_{capture}_fixation_review.png", dpi=160)
        plt.close(fig)
        for row in summaries:
            print(f"Capture {capture} {row['target_theta_deg']:5g}°: rows [{row['start_row']}, "
                  f"{row['end_row_exclusive']}); frames {row['first_frame']}–{row['last_frame_inclusive']}; "
                  f"valid {row['full_valid_count']}/{row['row_count']}; "
                  f"{row['duration_ms_first_to_last']/1000:.3f} s first-to-last")
    fig, axes = plt.subplots(4, 2, figsize=(17, 13), layout="constrained")
    for i, (arrays, y, valid, summaries) in enumerate(recordings):
        plot_capture(axes[i], arrays, y, valid, summaries, args.block_size, compact=True)
        for axis in axes[i]:
            axis.set_title(f"Capture {i+1}; demand label {DEMANDS[i]:.4g} D", fontsize=10)
    fig.suptitle("Fixation extraction review, captures 1–4: no fixed edge trimming\n"
                 + "Pupil gate applied to stored detections; native detection not rerun")
    fig.legend(handles=[Patch(facecolor=color, alpha=.4, label=f"{theta:g}°")
                        for color, theta in zip(COLORS, TARGETS)], loc="outside lower center", ncol=5)
    fig.savefig(args.output_dir / "fixation_review_overview.png", dpi=160)
    plt.close(fig)
    with (args.output_dir / "fixation_intervals.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(all_summaries[0]))
        writer.writeheader()
        writer.writerows(all_summaries)
    report = {"schema_version": 1, "purpose": "extraction review only",
              "boundary_resolution_rows": args.block_size,
              "boundary_note": "Approximate block boundaries, not exact transition frames; visual review required.",
              "fixed_edge_trimming": False, "outlier_rejection": False,
              "core_margin_rows": args.core_margin, "capture5_used": False,
              "duration_note": "Duration is last reported timestamp minus first, excluding one sample interval. Capture1 has backward timestamp jumps: inspect timestamp_duration_reliable before using elapsed times; row/frame ranges are unaffected.",
              "method": "Five median clusters of full-trace e_n; longest ordered stable run per target.",
              "validity_policy": "measurement_valid AND stored pupil_valid",
              "native_detection_rerun": False, "require_valid_pupil": True,
              "sources": sources,
              "fixations": all_summaries}
    (args.output_dir / "fixation_intervals.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Review artifacts saved in {args.output_dir}")


if __name__ == "__main__":
    main()
