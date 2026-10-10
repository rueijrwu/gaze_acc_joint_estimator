"""Compare raw and matched normalized-forward control A means; saved data only."""
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
DEFAULT_CONTROL = ROOT / "experiments/reverse_transform/raw_keystone/stage_04_framewise_accommodation/results/normalized_forward_control"
GAZES = (-10, -5, 0, 5, 10)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="Raw Stage 4 results directory")
    parser.add_argument("--control", type=Path, default=DEFAULT_CONTROL,
                        help="Matched normalized-forward control results directory")
    args = parser.parse_args()
    raw_dir, control_dir = args.results.resolve(), args.control.resolve()
    with np.load(raw_dir/"frames.npz", allow_pickle=False) as z:
        raw = {key: z[key] for key in z.files}
    with np.load(control_dir/"frames.npz", allow_pickle=False) as z:
        control = {key: z[key] for key in z.files}
    for key in ("row", "population_index", "exposure"):
        if not np.array_equal(raw[key], control[key]):
            raise ValueError(f"Raw and normalized control population mismatch: {key}")
    control_summary = json.loads((control_dir/"summary.json").read_text())
    if control_summary.get("status") != "COMPLETE":
        raise ValueError("Normalized forward control is not COMPLETE")
    exposures = np.unique(raw["exposure"])
    label = [f"C{int(e)//5+1}\n{GAZES[int(e)%5]:+d}°" for e in exposures]
    raw_mean, raw_sd, norm_mean, norm_sd, demand = [], [], [], [], []
    for exposure in exposures:
        sr = raw["exposure"] == exposure
        sc = control["exposure"] == exposure
        raw_values = raw["A_D"][sr]
        normalized_values = control["A_D"][sc]
        raw_mean.append(float(raw_values.mean()))
        raw_sd.append(float(raw_values.std()))
        norm_mean.append(float(normalized_values.mean()))
        norm_sd.append(float(normalized_values.std()))
        demand.append(float(raw["expected_A_D"][sr][0]))
    x = np.arange(len(exposures))
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    ax.errorbar(x-.09, raw_mean, yerr=raw_sd, marker="o", capsize=2.5, lw=1.4,
                color="#4c78a8", label="Raw-keystone Stage 4 A")
    ax.errorbar(x+.09, norm_mean, yerr=norm_sd, marker="s", capsize=2.5, lw=1.4,
                color="#f28e2b", label="Matched normalized-forward control A")
    ax.plot(x, demand, "k--", marker="x", lw=1., label="Nominal demand label")
    ax.set_xticks(x, label)
    ax.set_xlabel("Capture and nominal horizontal fixation")
    ax.set_ylabel("Accommodation-like state A (D), mean ± within-fixation SD")
    ax.set_title("Raw and normalized forward fits on the same 89,175 complete frames\n"
                 "Matched objective and fixation-mean anchors; error bars are within-fixation SD")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    output = raw_dir/"accommodation_raw_vs_normalized_forward_control.png"
    fig.savefig(output, dpi=165)
    plt.close(fig)
    source = Path(__file__).resolve()
    rel = Path("experiments/reverse_transform/raw_keystone/scripts/plot_normalized_control_comparison.py")
    snapshot = raw_dir/"plot_source_snapshot"/rel
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, snapshot)
    input_files = [(raw_dir, n) for n in ("summary.json", "frames.npz", "audit.json", "provenance.json")]
    input_files += [(control_dir, n) for n in ("summary.json", "frames.npz", "audit.json", "provenance.json")]
    metadata = dict(raw_control_population="Matched row, population_index, and exposure arrays.",
                    raw_input_sha256={n: sha256(raw_dir/n) for folder, n in input_files if folder == raw_dir and (folder/n).is_file()},
                    normalized_control_input_sha256={n: sha256(control_dir/n) for folder, n in input_files if folder == control_dir and (folder/n).is_file()},
                    plot_script_sha256=sha256(source), plot_script_snapshot=str(rel), output_sha256=sha256(output),
                    error_bars="Within-fixation standard deviation; not uncertainty of the mean.")
    (raw_dir/"normalized_control_plot_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
