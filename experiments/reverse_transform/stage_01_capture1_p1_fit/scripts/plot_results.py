#!/usr/bin/env python3
"""Plot saved capture1 P1-fit results without re-fitting or loading other data."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, required=True)
    args = ap.parse_args()
    root = args.results
    out = root
    summary = json.loads((root / 'summary.json').read_text())
    with np.load(root / 'frames.npz', allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    exposures = sorted(np.unique(a['exposure']).tolist())
    labels = [f"{summary['intervals'][i]['nominal_gaze_deg']:+g}°" for i in exposures]
    colors = plt.cm.viridis(np.linspace(.08, .92, len(exposures)))

    # Per-fixation magnification distributions and point error quantiles.
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    mags = [a['magnification'][a['exposure'] == e] for e in exposures]
    ax[0].boxplot(mags, tick_labels=labels, showfliers=False, patch_artist=True,
                  boxprops={'facecolor': '#9ecae1'}, medianprops={'color': '#08519c'})
    ax[0].axhline(1., color='black', lw=.8, ls='--')
    ax[0].set(title='Profiled magnification by fixation', xlabel='Nominal horizontal fixation', ylabel='M')
    kinds = [('Raw', 'raw_error'), ('Scale only', 'scale_only_error'), ('Full inverse', 'inverse_error')]
    med, p95 = [], []
    for _, key in kinds:
        d = np.linalg.norm(a[key], axis=-1)
        med.append([np.median(d[a['exposure'] == e]) for e in exposures])
        p95.append([np.percentile(d[a['exposure'] == e], 95) for e in exposures])
    x = np.arange(len(exposures)); width = .24
    method_colors = ['#777777', '#2878b5', '#238b45']
    for j, (name, _) in enumerate(kinds):
        pos = x + (j-1)*width
        ax[1].bar(pos, med[j], width, color=method_colors[j], alpha=.9, label=name)
        ax[1].scatter(pos, p95[j], marker='_', s=110, color=method_colors[j], linewidths=2, zorder=3)
    ax[1].set_xticks(x, labels)
    ax[1].set(title='Point error by fixation', xlabel='Nominal horizontal fixation', ylabel='Distance from empirical reference (px)')
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].text(.02, .98, 'Bars: median; same-color ticks: P95', transform=ax[1].transAxes, va='top', fontsize=8)
    fig.tight_layout(); fig.savefig(out / 'fit_distributions.png', dpi=180); plt.close(fig)

    # Full triangle overview plus one detail panel for each corresponding vertex.
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=(1.25, 1.0))
    overview = fig.add_subplot(gs[0, :])
    ref = a['reference']
    def triangle(ax, points, color, label, marker='o', ls='-'):
        q = np.vstack((points, points[:1]))
        ax.plot(q[:, 0], q[:, 1], marker=marker, ls=ls, color=color, label=label, lw=1.5, ms=4)
    triangle(overview, ref, 'black', 'Empirical reference', 's', '--')
    for j, (e, label, color) in enumerate(zip(exposures, labels, colors)):
        m = a['exposure'] == e
        triangle(overview, a['observed_centered_p1'][m].mean(axis=0), color,
                 'Observed' if j == 0 else '_nolegend_', 'o', ':')
        triangle(overview, (a['observed_centered_p1'][m] / a['magnification'][m, None, None]).mean(axis=0), color,
                 'Scale only' if j == 0 else '_nolegend_', '^', '--')
        triangle(overview, a['recovered_p1'][m].mean(axis=0), color,
                 'Full inverse' if j == 0 else '_nolegend_', 'x', '-')
    overview.set_aspect('equal', adjustable='datalim')
    overview.set(title='Mean P1 triangles: full view and per-vertex detail', xlabel='Centered x (px)', ylabel='Centered y (px)')
    overview.legend(ncol=4, fontsize=8, frameon=False, loc='best')
    for vi in range(3):
        axv = fig.add_subplot(gs[1, vi])
        bx, by = ref[vi]
        axv.scatter([bx], [by], marker='*', s=100, color='black', label='Reference', zorder=5)
        for j, (e, label, color) in enumerate(zip(exposures, labels, colors)):
            m = a['exposure'] == e
            observed = a['observed_centered_p1'][m, vi].mean(axis=0)
            recovered = a['recovered_p1'][m, vi].mean(axis=0)
            axv.scatter(*observed, marker='o', s=25, color=color,
                        label=f'{label} observed' if vi == 0 else '_nolegend_')
            axv.scatter(*recovered, marker='x', s=35, color=color,
                        label=f'{label} recovered' if vi == 0 else '_nolegend_')
        axv.set_xlim(bx-5, bx+5); axv.set_ylim(by-5, by+5)
        axv.set_aspect('equal', adjustable='box')
        axv.grid(alpha=.22)
        axv.set_title(('Left', 'Middle', 'Right')[vi] + ' vertex, ±5 px')
        axv.set_xlabel('Centered x (px)')
        if vi == 0: axv.set_ylabel('Centered y (px)')
    handles, labels_legend = fig.axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_legend, loc='lower center', ncol=6, fontsize=7, frameon=False,
               bbox_to_anchor=(.5, .005))
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(out / 'mean_triangles.png', dpi=180); plt.close(fig)

    # M and calibrated gaze against capture1 source row; gaze components have separate axes.
    order = np.argsort(a['row'])
    row = a['row'][order]
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].scatter(row, a['magnification'][order], s=3, alpha=.45, color='#2166ac')
    ax[0].axhline(1., color='black', lw=.8, ls='--')
    ax[0].set_ylabel('M')
    ax[0].set_title('Capture1 profiled magnification and centroid-calibrated gaze')
    ax[1].scatter(row, a['gaze_xy_deg'][order, 0], s=3, alpha=.4, color='#b2182b')
    ax[1].set_ylabel(r'$\theta_x$ (deg)')
    ax[2].scatter(row, a['gaze_xy_deg'][order, 1], s=3, alpha=.4, color='#4d9221')
    ax[2].set_ylabel(r'$\theta_y$ (deg)'); ax[2].set_xlabel('Capture1 source row')
    for q in ax: q.grid(alpha=.18)
    fig.tight_layout(); fig.savefig(out / 'magnification_gaze_by_row.png', dpi=180); plt.close(fig)


if __name__ == '__main__':
    main()
