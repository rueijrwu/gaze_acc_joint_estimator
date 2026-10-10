"""Plot per-fixation mean raw-forward vertex residual vectors in native pixels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "docs/Theory.md").is_file())
VERTEX_COLORS = ("#e15759", "#4e79a7", "#59a14f")
GAZES = (-10, -5, 0, 5, 10)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, choices=range(1, 5), required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    results = args.results.resolve()
    with np.load(results / "frames.npz", allow_pickle=False) as archive:
        frames = {key: archive[key] for key in archive.files}
    summary = json.loads((results / "summary.json").read_text())
    if summary.get("stage") != args.stage:
        raise ValueError("Requested stage does not match saved summary")
    patterns = [pattern for pattern in (1, 4) if f"forward_residual_p{pattern}" in frames]
    captures = np.unique(frames["capture_index"])
    fig, axes = plt.subplots(len(patterns), len(captures),
                             figsize=(4.1*len(captures), 4.0*len(patterns)),
                             squeeze=False)
    fig.subplots_adjust(top=.86, right=.88, bottom=.10, hspace=.28, wspace=.22)
    maxima = []
    for pattern in patterns:
        residual = frames[f"forward_residual_p{pattern}"]
        for capture in captures:
            for fixation in range(5):
                exposure = int(capture)*5+fixation
                mask = frames["exposure"] == exposure
                if mask.any():
                    maxima.append(float(np.max(np.abs(residual[mask].mean(axis=0)))))
    limit = max(2., 1.2*max(maxima, default=1.))
    for ri, pattern in enumerate(patterns):
        residual = frames[f"forward_residual_p{pattern}"]
        for ci, capture in enumerate(captures):
            ax = axes[ri, ci]
            for local_fixation, gaze in enumerate(GAZES):
                exposure = int(capture)*5+local_fixation
                mask = frames["exposure"] == exposure
                if not mask.any():
                    continue
                vectors = residual[mask].mean(axis=0)
                for vertex, color in enumerate(VERTEX_COLORS):
                    dx, dy = vectors[vertex]
                    ax.quiver(0., 0., dx, dy, angles="xy", scale_units="xy", scale=1,
                              color=color, alpha=.75, width=.004)
                    ax.scatter([dx], [dy], color=color, s=14, zorder=3)
                    # Keep fixation labels near their endpoints with a slight deterministic offset.
                    ax.annotate(f"{gaze:+d}°", (dx, dy), xytext=(2, 2+vertex*2),
                                textcoords="offset points", fontsize=6, color=color)
            ax.axhline(0., color="black", lw=.7)
            ax.axvline(0., color="black", lw=.7)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=.22)
            ax.set_title(f"Capture {int(capture)+1}")
            ax.set_xlabel("Mean residual x (px)")
            if ci == 0:
                ax.set_ylabel(f"P{pattern} mean residual y (px)")
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], color=color, marker="o", lw=1.5,
                     label=f"Vertex {i+1}") for i, color in enumerate(VERTEX_COLORS)]
    legend.append(Line2D([0], [0], color="black", marker=">", lw=1,
                         label="Arrow vector; 1 data unit = 1 px"))
    fig.legend(handles=legend, loc="center left", bbox_to_anchor=(.89, .5),
               ncol=1, fontsize=8)
    fig.suptitle(f"Raw-keystone Stage {args.stage}: mean forward vertex residual vectors\n"
                 "Arrows start at zero; labels show nominal fixation. This residual-plane view is in native pixels.",
                 y=.98)
    output = results / "vertex_residual_vectors_native_px.png"
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    source = Path(__file__).resolve()
    metadata = dict(stage=args.stage,
                    input_sha256={name: sha256(results/name) for name in ("summary.json", "frames.npz", "audit.json") if (results/name).is_file()},
                    plot_script_sha256=sha256(source), output_sha256=sha256(output),
                    source_snapshot=str(source.relative_to(ROOT)),
                    interpretation="Mean signed forward residual vectors by vertex and fixation; no fitting or alignment.")
    (results / "vertex_residual_plot_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
