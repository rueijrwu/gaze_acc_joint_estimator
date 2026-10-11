#!/usr/bin/env python3
"""Render saved inverse distortion plots; performs no fitting or inversion."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BRANCHES = (
    ("baseline_g1", "g=1 fit + g=1 scale", "#2878b5"),
    ("baseline_frozen_p1m", "g=1 fit + frozen P1 m", "#3b9b62"),
    ("p1m", "P1 m fit + P1 m", "#e07a28"),
)


def render(results: Path) -> None:
    with (results / "inverse_fixation_summary.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    z = np.load(results / "inverse_arrays.npz", allow_pickle=False)
    needed = {"capture", "source_row", "fixation_id"}
    for branch, _, _ in BRANCHES:
        needed |= {f"{branch}_valid", f"{branch}_frame_vertex_rms_px"}
    missing = needed.difference(z.files)
    if missing:
        raise ValueError(f"inverse_arrays.npz is missing keys: {sorted(missing)}")
    captures = sorted({int(r["capture"]) for r in rows})
    fixations = sorted({int(r["fixation"]) for r in rows})
    keys = {(r["branch"], int(r["capture"]), int(r["fixation"])) for r in rows}
    expected = {(b, c, f) for b, _, _ in BRANCHES for c in captures for f in fixations}
    if keys != expected or len(rows) != len(expected):
        raise ValueError("inverse_fixation_summary.csv lacks complete branch/capture/fixation coverage")
    for branch, _, _ in BRANCHES:
        if sum(int(r["frames"]) for r in rows if r["branch"] == branch) != len(z["capture"]):
            raise ValueError(f"CSV frame counts do not match inverse arrays for {branch}")
    fig, axes = plt.subplots(1, len(captures), figsize=(14, 3.8), sharey=True)
    for ax, capture in zip(axes, captures):
        for branch, label, color in BRANCHES:
            values = []
            for fixation in fixations:
                match = [r for r in rows if r["branch"] == branch and int(r["capture"]) == capture and int(r["fixation"]) == fixation]
                if len(match) != 1:
                    raise ValueError(f"expected one summary row for {branch}, capture {capture}, fixation {fixation}")
                values.append(float(match[0]["pooled_vertex_rms_px"]))
            ax.plot(fixations, values, "o-", color=color, label=label, linewidth=1.7, markersize=3.8)
        ax.set_title(f"Capture {capture}")
        ax.set_xticks(fixations)
        ax.set_xlabel("Fixation")
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Inverse pooled vertex RMS (px)")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Inverse residual by capture and fixation")
    fig.tight_layout()
    fig.savefig(results / "inverse_fixation_rms.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(BRANCHES), 1, figsize=(12, 7), sharex=True)
    capture_array = z["capture"]
    source_row = z["source_row"]
    for ax, (branch, label, color) in zip(axes, BRANCHES):
        valid = z[f"{branch}_valid"].astype(bool)
        error = z[f"{branch}_frame_vertex_rms_px"]
        for capture in np.unique(capture_array):
            ix = np.flatnonzero((capture_array == capture) & valid)
            ix = ix[np.argsort(source_row[ix])]
            cuts = np.r_[0, np.flatnonzero(np.diff(source_row[ix]) != 1) + 1, len(ix)]
            for start, stop in zip(cuts[:-1], cuts[1:]):
                if stop > start:
                    ax.plot(source_row[ix[start:stop]], error[ix[start:stop]], color=color, alpha=.55, lw=.55)
        ax.set_ylabel(f"{label}\nRMS (px)")
        ax.grid(alpha=.2)
    axes[-1].set_xlabel("Source row (lines break at source-row gaps)")
    fig.suptitle("Inverse frame RMS in source-row order")
    fig.tight_layout()
    fig.savefig(results / "inverse_residual_timeseries.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    render(args.results)


if __name__ == "__main__":
    main()
