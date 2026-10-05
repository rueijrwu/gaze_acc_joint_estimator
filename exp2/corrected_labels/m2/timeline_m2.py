#!/usr/bin/env python3
"""Per-fixation timeline of gaze and accommodation, Holdout-3 vs Full model, same axes (lines only).

Both models are run through the same anchor-free inverse on held-out fixations 10-14 (3 D).
Usage: python3 timeline_m2.py CONFIG [CONFIG ...]   (e.g. M2a01 M0); run from this directory.
"""
import csv
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TARGETS = {10: -15.0, 11: -7.5, 12: 0.0, 13: 7.5, 14: 15.0}
DEMAND = 3.0
TITLES = {'M0': 'current model (piecewise), 1 deg gaze anchor',
          'M2a1': 'M2, 1 deg gaze anchor',
          'M2a01': 'M2, 0.1 deg gaze anchor'}


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in ('frame_index', 'fixation_index', 'theta_deg', 'A_diopters')}


def plot(config):
    base = f'{config}/predictions/holdout3/'
    h3, full = load(base + 'trained_predictions.csv'), load(base + 'fresh_matched_full_fixed_inverse.csv')
    assert np.array_equal(h3['frame_index'], full['frame_index']), 'models must cover identical frames'
    fig, axes = plt.subplots(5, 2, figsize=(14, 15), sharex=False)
    for row, fix in enumerate(TARGETS):
        m = h3['fixation_index'] == fix
        x = h3['frame_index'][m] - h3['frame_index'][m].min()
        for col, (key, target, label) in enumerate([('theta_deg', TARGETS[fix], 'Gaze (deg)'),
                                                    ('A_diopters', DEMAND, 'Accommodation (D)')]):
            ax = axes[row, col]
            ax.plot(x, h3[key][m], color='#1f77b4', lw=0.8, label='Holdout-3')
            ax.plot(x, full[key][m], color='#ff7f0e', lw=0.8, label='Full')
            ax.axhline(target, color='0.3', ls='--', lw=1.2, label=f'Target {target:g}')
            ax.set_ylabel(label)
            ax.set_title(f'Fixation {fix}: mean Holdout-3 {h3[key][m].mean():.3f}, Full {full[key][m].mean():.3f}, target {target:g}',
                         fontsize=9)
            ax.grid(alpha=0.3)
            if row == 0:
                ax.legend(fontsize=8, loc='upper right')
            if row == 4:
                ax.set_xlabel('Frames since fixation start')
    fig.suptitle(f'{TITLES.get(config, config)}: Holdout-3 vs Full, anchor-free inverse on held-out 3 D fixations', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = f'timeline_{config}.png'
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    for c in sys.argv[1:] or ['M2a01']:
        plot(c)
