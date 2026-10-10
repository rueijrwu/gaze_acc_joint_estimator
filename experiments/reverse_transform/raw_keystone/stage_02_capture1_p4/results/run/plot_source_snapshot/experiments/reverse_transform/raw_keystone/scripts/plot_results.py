"""Plot saved raw-keystone chain results; performs no fitting."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "docs/Theory.md").is_file())
CHAIN = ROOT / "experiments/reverse_transform/raw_keystone"
PATTERNS = (1, 4)
GAZES = (-10, -5, 0, 5, 10)
COLORS = {"raw": "#4c78a8", "raw_p4": "#59a14f", "scale": "#e15759",
          "historical": "#f28e2b"}
VERTEX_COLORS = ("#e15759", "#4e79a7", "#59a14f")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_results(folder: Path, stage: int):
    summary = json.loads((folder / "summary.json").read_text())
    history = json.loads((folder / "historical_comparison.json").read_text())
    with np.load(folder / "frames.npz", allow_pickle=False) as z:
        frames = {key: z[key] for key in z.files}
    if summary.get("stage") != stage:
        raise ValueError(f"Saved stage {summary.get('stage')} does not match requested stage {stage}")
    if len(frames["row"]) != summary["complete"]:
        raise ValueError("Frame arrays do not match summary complete-frame count")
    return summary, history, frames


def stats(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"count": 0, "median": np.nan, "p95": np.nan, "rms": np.nan,
                "mean": np.nan, "std": np.nan}
    return {"count": int(x.size), "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)), "rms": float(np.sqrt(np.mean(x*x))),
            "mean": float(x.mean()), "std": float(x.std())}


def exposure_labels(frames):
    exposures = np.unique(frames["exposure"])
    return exposures, [f"C{int(e)//5+1}\n{GAZES[int(e)%5]:+d}°" for e in exposures]


def fixation_metrics(frames, pattern, exposure):
    sel = frames["exposure"] == exposure
    result = {}
    forward_key = f"forward_residual_p{pattern}"
    recovered_key = f"recovered_p{pattern}"
    if forward_key in frames:
        result["forward"] = stats(np.linalg.norm(frames[forward_key][sel], axis=2))
    if recovered_key in frames:
        valid_key = f"p{pattern}_inverse_valid"
        valid = sel & frames[valid_key] if valid_key in frames else sel
        distance = np.linalg.norm(frames[recovered_key][valid] - frames[f"reference_p{pattern}"], axis=2)
        result["inverse"] = stats(distance)
    return result


def plot_point_errors(frames, stage, results):
    exposures, labels = exposure_labels(frames)
    patterns = [n for n in PATTERNS if f"forward_residual_p{n}" in frames]
    fig, axes = plt.subplots(len(patterns), 2, figsize=(13, 4.5*len(patterns)),
                             squeeze=False, sharex=True, constrained_layout=True)
    metrics = ("median", "p95", "rms")
    for row, pattern in enumerate(patterns):
        for col, kind in enumerate(("forward", "inverse")):
            ax = axes[row, col]
            for metric, marker in zip(metrics, ("o", "s", "^") ):
                values = [fixation_metrics(frames, pattern, e).get(kind, {}).get(metric, np.nan)
                          for e in exposures]
                ax.plot(np.arange(len(exposures)), values, marker=marker, lw=1.3,
                        ms=3.5, label=metric.upper() if metric == "p95" else metric)
            ax.set_title(f"P{pattern} {kind} corresponding-vertex distance")
            ax.set_ylabel("Distance (px)")
            ax.set_xticks(np.arange(len(exposures)), labels)
            ax.grid(alpha=.25)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("Capture and nominal horizontal fixation")
    fig.suptitle(f"Raw-keystone Stage {stage}: per-fixation P1/P4 forward and inverse errors\n"
                 "Median, P95, and RMS of corresponding-vertex distances")
    path = results / "point_errors_by_fixation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def plot_size_factors(frames, stage, results):
    exposures, labels = exposure_labels(frames)
    series = []
    for pattern in PATTERNS:
        key = f"p{pattern}_raw_size_factor"
        if key in frames:
            series.append((f"P{pattern} raw keystone size factor", key,
                           "raw" if pattern == 1 else "raw_p4"))
    if "p1_magnification" in frames:
        series.append(("Common P1 scale M", "p1_magnification", "scale"))
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    offsets = np.linspace(-.18, .18, len(series)) if series else []
    for (label, key, group), offset in zip(series, offsets):
        means, sds = [], []
        for e in exposures:
            values = frames[key][frames["exposure"] == e]
            means.append(float(np.mean(values)))
            sds.append(float(np.std(values)))
        ax.errorbar(np.arange(len(exposures))+offset, means, yerr=sds, marker="o",
                    capsize=2.5, lw=1.25, ms=4, color=COLORS[group], label=label)
    ax.axhline(1., color="black", lw=.8, ls="--", label="Unit scale")
    ax.set_xticks(np.arange(len(exposures)), labels)
    ax.set_xlabel("Capture and nominal horizontal fixation")
    ax.set_ylabel("Scale factor (mean ± within-fixation SD)")
    ax.set_title(f"Raw-keystone Stage {stage}: profiled common P1 scale and unnormalized keystone size\n"
                 "Keystone factors are diagnostics; no post-keystone RMS normalization is applied")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    path = results / "scale_and_raw_size_by_fixation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def plot_p4_radius(frames, history, stage, results):
    if "recovered_p4" not in frames:
        return None
    records = {int(r["exposure"]): r for r in history["fixations"]}
    exposures, labels = exposure_labels(frames)
    raw_means, raw_sds, old_means, old_sds = [], [], [], []
    has_old = False
    for e in exposures:
        record = records[int(e)]
        raw_stats = record.get("raw_p4_inverse_common", {}).get("radius_ratio")
        if raw_stats is None:
            raw_stats = record.get("raw_p4_inverse", {}).get("radius_ratio")
        old_stats = record.get("historical_p4_inverse_common", {}).get("radius_ratio")
        if old_stats is None:
            old_stats = record.get("historical_p4_inverse", {}).get("radius_ratio")
        raw_means.append(np.nan if raw_stats is None else raw_stats["mean"])
        raw_sds.append(np.nan if raw_stats is None else raw_stats["std"])
        old_means.append(np.nan if old_stats is None else old_stats["mean"])
        old_sds.append(np.nan if old_stats is None else old_stats["std"])
        has_old |= old_stats is not None
    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    x = np.arange(len(exposures))
    ax.errorbar(x-.08, raw_means, yerr=raw_sds, marker="o", ms=4, capsize=2.5,
                color=COLORS["raw"], label="Raw-keystone inverse")
    if has_old:
        ax.errorbar(x+.08, old_means, yerr=old_sds, marker="s", ms=4, capsize=2.5,
                    color=COLORS["historical"], label="Historical inverse")
    ax.axhline(1., color="black", lw=.8, ls="--", label="Empirical reference radius")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Capture and nominal horizontal fixation")
    ax.set_ylabel("Recovered P4 / reference RMS radius")
    ax.set_title(f"Raw-keystone Stage {stage}: inverse P4 radius ratios\n"
                 "Historical and raw means use their shared valid frames when available")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    path = results / "p4_inverse_radius_ratio_by_fixation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def historical_arrays(history):
    path = ROOT / history["path"]
    if not path.is_file():
        raise FileNotFoundError(f"Historical inverse arrays unavailable: {path}")
    with np.load(path, allow_pickle=False) as z:
        old = {key: z[key] for key in z.files}
    result = {}
    for pattern in PATTERNS:
        key = f"recovered_p{pattern}"
        if key not in old and pattern == 1 and "recovered" in old:
            key = "recovered"
        if key in old:
            result[pattern] = old[key]
    return old, result


def plot_triangles(frames, history, stage, results):
    old, old_points = historical_arrays(history)
    if not np.array_equal(old["row"], frames["row"]) or not np.array_equal(old["population_index"], frames["population_index"]):
        raise ValueError("Historical triangle arrays do not match the current saved population order")
    exposures = np.unique(frames["exposure"])
    captures = np.unique(frames["capture_index"])
    outputs = []
    for pattern in PATTERNS:
        raw_key = f"recovered_p{pattern}"
        if raw_key not in frames or pattern not in old_points:
            continue
        ref = frames[f"reference_p{pattern}"]
        figure, axes = plt.subplots(5, len(captures), figsize=(4.0*len(captures), 18),
                                    squeeze=False, constrained_layout=True)
        raw = frames[raw_key]
        hist = old_points[pattern]
        valid_key = f"p{pattern}_inverse_valid"
        for local_fixation in range(5):
            for col, capture in enumerate(captures):
                exposure = int(capture)*5+local_fixation
                mask = ((frames["exposure"] == exposure) & np.isfinite(raw).all(axis=(1, 2)) &
                        np.isfinite(hist).all(axis=(1, 2)))
                if valid_key in frames:
                    mask &= frames[valid_key]
                indices = np.flatnonzero(mask)
                seed = 5705 + (pattern-1)*100 + exposure
                if len(indices) > 100:
                    indices = np.sort(np.random.default_rng(seed).choice(indices, 100, replace=False))
                ax = axes[local_fixation, col]
                for vertex, color in enumerate(VERTEX_COLORS):
                    if len(indices):
                        ax.scatter(raw[indices, vertex, 0], raw[indices, vertex, 1], s=5,
                                   alpha=.13, color=color, rasterized=True)
                        ax.scatter(hist[indices, vertex, 0], hist[indices, vertex, 1], s=9,
                                   alpha=.24, marker="x", color=color, rasterized=True)
                if mask.any():
                    raw_mean = raw[mask].mean(axis=0)
                    hist_mean = hist[mask].mean(axis=0)
                    for triangle, style in ((raw_mean, "b-"), (hist_mean, "#f28e2b")):
                        closed = np.vstack((triangle, triangle[0]))
                        ax.plot(closed[:, 0], closed[:, 1], style, marker="o", ms=2.4, lw=1.0)
                ref_closed = np.vstack((ref, ref[0]))
                ax.plot(ref_closed[:, 0], ref_closed[:, 1], "k--", lw=1.0)
                margin = max(8., .05*float(np.sqrt(np.mean(np.sum(ref*ref, axis=1)))))
                ax.set_xlim(float(ref[:, 0].min()-margin), float(ref[:, 0].max()+margin))
                ax.set_ylim(float(ref[:, 1].min()-margin), float(ref[:, 1].max()+margin))
                ax.set_aspect("equal", adjustable="box")
                ax.grid(alpha=.18)
                if local_fixation == 0:
                    ax.set_title(f"Capture {int(capture)+1}")
                if col == 0:
                    ax.set_ylabel(f"{GAZES[local_fixation]:+d}°\nCamera y (px)")
                if local_fixation == 4:
                    ax.set_xlabel("Camera x (px)")
        figure.suptitle(f"Stage {stage} P{pattern}: raw and historical recovered triangles\n"
                        "Same fixed valid frames and deterministic cloud sample per fixation; native coordinates, no alignment or radius normalization")
        name = f"p{pattern}_triangles_raw_vs_historical.png"
        figure.savefig(results/name, dpi=150)
        plt.close(figure)
        outputs.append(name)
    return outputs


def plot_accommodation(frames, history, stage, results):
    if stage != 4 or "A_D" not in frames:
        return []
    records = {int(r["exposure"]): r for r in history["fixations"]}
    exposures, labels = exposure_labels(frames)
    raw_mean, raw_sd, old_mean, old_sd, expected = [], [], [], [], []
    for exposure in exposures:
        sel = frames["exposure"] == exposure
        value = frames["A_D"][sel]
        raw_mean.append(float(value.mean()))
        raw_sd.append(float(value.std()))
        historic = records[int(exposure)].get("historical_A_D", {})
        old_mean.append(historic.get("mean", np.nan))
        old_sd.append(historic.get("std", np.nan))
        expected.append(float(frames["expected_A_D"][sel][0]))
    x = np.arange(len(exposures))
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    ax.errorbar(x-.08, raw_mean, yerr=raw_sd, marker="o", capsize=2.5,
                color=COLORS["raw"], label="Raw-keystone framewise A")
    ax.errorbar(x+.08, old_mean, yerr=old_sd, marker="s", capsize=2.5,
                color=COLORS["historical"], label="Historical framewise A")
    ax.plot(x, expected, "k--", marker="x", label="Nominal demand label")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Capture and nominal horizontal fixation")
    ax.set_ylabel("Accommodation-like state A (D), mean ± within-fixation SD")
    ax.set_title("Stage 4 forward A fit vs historical inverse A state\n"
                 "Error bars are within-fixation SD, not uncertainty of the mean")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    state_path = results / "accommodation_by_fixation.png"
    fig.savefig(state_path, dpi=160)
    plt.close(fig)
    row, state = frames["row"], frames["A_D"]
    fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
    for fixation in exposures:
        mask = frames["exposure"] == fixation
        ax.scatter(row[mask], state[mask], s=3, alpha=.32,
                   label=f"C{int(fixation)//5+1} {GAZES[int(fixation)%5]:+d}°")
    ax.set_xlabel("Source row (points only; no connecting gaps)")
    ax.set_ylabel("Raw-keystone fitted A (D)")
    ax.set_title("Stage 4 framewise accommodation-like state by source row\n"
                 "Source order is not a timestamp and is not interpolated")
    ax.grid(alpha=.2)
    ax.legend(ncol=5, fontsize=7)
    row_path = results / "accommodation_by_source_row.png"
    fig.savefig(row_path, dpi=160)
    plt.close(fig)
    return [state_path.name, row_path.name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=range(1, 5), required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    results = args.results.resolve()
    summary, history, frames = load_results(results, args.stage)
    outputs = [plot_point_errors(frames, args.stage, results),
               plot_size_factors(frames, args.stage, results),
               plot_p4_radius(frames, history, args.stage, results)]
    outputs = [name for name in outputs if name]
    outputs.extend(plot_triangles(frames, history, args.stage, results))
    outputs.extend(plot_accommodation(frames, history, args.stage, results))
    source = Path(__file__).resolve()
    rel = Path("experiments/reverse_transform/raw_keystone/scripts/plot_results.py")
    snapshot = results / "plot_source_snapshot" / rel
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, snapshot)
    input_names = ["summary.json", "frames.npz", "historical_comparison.json", "protocol.json", "provenance.json", "audit.json"]
    input_hashes = {name: sha256(results/name) for name in input_names if (results/name).is_file()}
    metadata = dict(stage=args.stage, input_sha256=input_hashes, plot_script_sha256=sha256(source),
                    plot_script_snapshot=str(Path("plot_source_snapshot")/rel),
                    outputs={name: sha256(results/name) for name in outputs},
                    plot_scope="Saved artifacts only; no fitting or refitting.",
                    triangle_sampling="Per pattern and fixation, same valid indices for raw and historical arrays; deterministic seed.",
                    triangle_coordinates="Native camera coordinates; no rotation alignment or radius normalization.",
                    size_factors="Raw model-size factors are diagnostics only; no post-keystone RMS normalization is applied.")
    (results/"plot_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
