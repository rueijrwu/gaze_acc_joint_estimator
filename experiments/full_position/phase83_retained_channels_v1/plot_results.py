#!/usr/bin/env python3
"""Make the standalone Phase 8.3 retained-x versus retained-xy metric plot."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
summary = json.loads((OUT / "summary.json").read_text())["responses"]
cells = [(response, family) for response in ("baseline27", "strong_anchor37")
         for family in ("gaze", "capture")]
labels = [f"{response.replace('strong_anchor', 'anchor ')}\n{family}"
          for response, family in cells]
metrics = [
    ("complete_frame_E_px", "E (px)"),
    ("complete_frame_G_theta_deg", "Gθ (degrees)"),
    ("complete_frame_G_A_D", "G_A (diopters)"),
    ("complete_frame_worst_point_px", "Worst point (px)"),
]
colors = {"x": "#276FBF", "xy": "#F28E2B"}
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
xx = np.arange(len(cells))
width = 0.34
for ax, (field, title) in zip(axes, metrics):
    for offset, method in ((-width / 2, "x"), (width / 2, "xy")):
        values = [summary[response][family]["methods"][method]["all_testable"][field]["equal_fixation"]["rms"]
                  for response, family in cells]
        ax.bar(xx + offset, values, width, label=f"retained {method}", color=colors[method])
    ax.set_title(title)
    ax.set_xticks(xx, labels)
    ax.grid(axis="y", alpha=.25)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("RMS, equally weighted across 20 exposure groups")
axes[-1].legend(frameon=False, loc="upper right")
fig.suptitle("Phase 8.3: retained-channel comparison", fontsize=13)
fig.savefig(OUT / "phase83_metrics.png", dpi=180)
