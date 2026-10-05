#!/usr/bin/env python3
"""Inverse-free test: within each fixation, how does the P4-P1 displacement m move when S4 moves?

For all 20 fixations (4 demands x 5 gaze targets), using the raw measurements (m, S1, S4) only.
The shared S1 scale is removed by working with dm and dS4 (pixels); the slow band (> 0.5 s) is used.

Hypotheses (coefficients: Full model of CONFIG, Jacobian at the nominal state):
  fixed gaze, real accommodation change -> dm/dS4 = k_A = (dd/dA)/(drho/dA)
  co-movement at the geometric slope (S4 artifact, miscalibrated cross term, or exact motor coupling) -> dm/dS4 = 0
The observed ratio also gives the gaze/accommodation slope that the inverse would report (implied slope);
a motor coupling with its own physiology would give an implied slope that does not track the optics.

Usage (from this directory): python3 observation_coupling.py [CONFIG]   -> <CONFIG>/observation_coupling/
"""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import m2_model  # noqa: E402
from observations import relative_measurements  # noqa: E402
from temporal_correlation import glitch_mask, uniform_series, split_bands, pearson, slope  # noqa: E402

DEMAND = {1: 1000/2775, 2: 4.0, 3: 3.0, 4: 2.0}
TARGETS = {1: [-7.21983367, -3.62430389, 0.0, 3.62430389, 7.21983367]}
TARGETS.update({c: [-15.0, -7.5, 0.0, 7.5, 15.0] for c in (2, 3, 4)})
N_SEG = 4


def fixation_rows(config, capture):
    with open(HERE/config/'training'/'full'/'robust'/f'capture_{capture}_states.csv') as f:
        rows = [r for r in csv.DictReader(f) if r['status'] == 'calibration_core' and r['d']]
    return rows


def analyse(config):
    coef = np.array(json.loads((HERE/config/'training/full/robust/model.json').read_text())['coefficients'])
    out = []
    for capture in (1, 2, 3, 4):
        arrays = pickle.load(open(HERE.parents[1]/f'capture_{capture}_detections.pkl', 'rb'))['arrays']
        z = relative_measurements(arrays)['z']
        p1x = np.asarray(arrays['p1_points'], float)[:, :, 0]
        p4x = np.asarray(arrays['p4_points'], float)[:, :, 0]
        row = {int(f): i for i, f in enumerate(np.asarray(arrays['frame_index']))}
        rows = fixation_rows(config, capture)
        fix = np.array([int(r['fixation_index']) for r in rows])
        frames = np.array([int(r['frame_index']) for r in rows])
        d = np.array([float(r['d']) for r in rows])
        rho = np.array([float(r['rho4']) for r in rows])
        for k, fx in enumerate(sorted(set(fix))):
            m = fix == fx
            idx = np.array([row[int(f)] for f in frames[m]])
            excl = glitch_mask(d[m], rho[m])
            mid1 = p1x[idx].mean(axis=1)
            grid, s, real = uniform_series(frames[m], excl, m_px=z[idx, 0], S1=z[idx, 1], S4=z[idx, 2],
                                           P4L=p4x[idx, 0]-mid1, P4R=p4x[idx, 1]-mid1)
            bm, b4 = split_bands(s['m_px'])['slow'], split_bands(s['S4'])['slow']
            S1m = s['S1'][real].mean()
            bL, bR = split_bands(s['P4L'])['slow'][real], split_bands(s['P4R'])['slow'][real]
            share_R = float(np.cov(bR, bR-bL)[0, 1]/np.var(bR-bL, ddof=1))   # share of slow dS4 carried by the right P4 glint
            k_obs = slope(bm[real], b4[real])
            # segment slopes for an uncertainty estimate (slow band is strongly autocorrelated)
            seg = np.array_split(np.flatnonzero(real), N_SEG)
            seg_k = [slope(bm[i], b4[i]) for i in seg if np.std(b4[i]) > 0]
            theta0, A0 = TARGETS[capture][k], DEMAND[capture]
            _, J = m2_model.forward_jac_m2(np.array([theta0]), np.array([A0]), coef)
            J = J[0]
            k_A = J[0, 1]/J[1, 1]
            Jinv = np.linalg.inv(J)
            v = Jinv@np.array([k_obs, 1.0])
            implied = v[0]/v[1]
            out.append(dict(fixation=int(fx), capture=capture, demand=A0, target=theta0,
                            frames=int(m.sum()), glitch=int(excl.sum()),
                            sd_S4_px=float(np.std(b4[real])), sd_m_px=float(np.std(bm[real])), S1_px=float(S1m),
                            sd_P4L_px=float(np.std(bL)), sd_P4R_px=float(np.std(bR)), dS4_share_P4R=share_R,
                            r_m_S4=pearson(bm[real], b4[real]), k_obs=k_obs, k_obs_seg_sd=float(np.std(seg_k)),
                            k_fixed_gaze=float(k_A), geometric_slope=float(-J[0, 1]/J[0, 0]), implied_slope=float(implied)))
    return out


def write(config, res):
    out_dir = HERE/config/'observation_coupling'
    out_dir.mkdir(exist_ok=True)
    with open(out_dir/'observation_coupling.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
    g = np.array([r['geometric_slope'] for r in res])
    i = np.array([r['implied_slope'] for r in res])
    ko = np.array([r['k_obs'] for r in res])
    ka = np.array([r['k_fixed_gaze'] for r in res])
    fit = np.polyfit(g, i, 1)
    r2 = 1-np.var(i-np.polyval(fit, g))/np.var(i)
    md = [f'# Inverse-free coupling test: {config}', '',
          'Slow band (> 0.5 s), raw pixels, shared S1 scale removed (dm vs dS4). Coefficients: Full model; Jacobian at the nominal state.',
          'k_obs = dm/dS4 measured; k_fixed_gaze = what the calibration predicts if gaze is held and accommodation really changes.',
          'implied slope = gaze/accommodation slope the inverse reports for k_obs; geometric slope = the value for k_obs = 0.',
          f'k_obs segment SD = SD of the slope over {N_SEG} ~1 s segments (rough uncertainty).',
          'P4R share = share of the slow dS4 carried by the right P4 glint (0.5 = symmetric spacing change about a fixed centre; 1 = only P4R moves; >1 = P4L moves the same way).', '',
          '| fix | demand (D) | target (deg) | SD dS4 (px) | SD dm (px) | r(dm, dS4) | k_obs (seg SD) | k_fixed_gaze | implied slope (deg/D) | geometric slope (deg/D) | P4R share of dS4 |',
          '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in res:
        md.append(f"| {r['fixation']} | {r['demand']:.2f} | {r['target']:+.2f} | {r['sd_S4_px']:.2f} | {r['sd_m_px']:.2f} | {r['r_m_S4']:+.2f} | "
                  f"{r['k_obs']:+.2f} ({r['k_obs_seg_sd']:.2f}) | {r['k_fixed_gaze']:+.2f} | {r['implied_slope']:+.2f} | {r['geometric_slope']:+.2f} | {r['dS4_share_P4R']:+.2f} |")
    md += ['', f'Across 20 fixations: implied slope = {fit[0]:.2f} x geometric slope {fit[1]:+.2f} deg/D, R2 = {r2:.2f}.',
           f'Median |k_obs| = {np.median(np.abs(ko)):.2f}; median |k_fixed_gaze| = {np.median(np.abs(ka)):.2f}.', '']
    (out_dir/'observation_coupling.md').write_text('\n'.join(md))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    colors = {1: '#9467bd', 4: '#2ca02c', 3: '#1f77b4', 2: '#d62728'}
    for c in (1, 4, 3, 2):
        sel = [r for r in res if r['capture'] == c]
        t = [r['target'] for r in sel]
        ax[0].plot(t, [r['k_obs'] for r in sel], color=colors[c], lw=2, label=f"measured, {DEMAND[c]:.2f} D")
        ax[0].plot(t, [r['k_fixed_gaze'] for r in sel], color=colors[c], lw=1.2, ls='--', label=f"if gaze held, {DEMAND[c]:.2f} D")
        ax[1].plot(t, [r['implied_slope'] for r in sel], color=colors[c], lw=2, label=f"implied, {DEMAND[c]:.2f} D")
        ax[1].plot(t, [r['geometric_slope'] for r in sel], color=colors[c], lw=1.2, ls='--', label=f"geometric, {DEMAND[c]:.2f} D")
    ax[0].axhline(0, color='k', lw=0.8)
    ax[0].set_xlabel('Target gaze (deg)')
    ax[0].set_ylabel('dm / dS4 (px per px), slow band')
    ax[0].set_title('P4-P1 displacement change per S4 change')
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].grid(alpha=0.3)
    ax[1].axhline(0, color='k', lw=0.8)
    ax[1].set_xlabel('Target gaze (deg)')
    ax[1].set_ylabel('Gaze change per accommodation change (deg / D)')
    ax[1].set_title(f'Implied vs geometric slope (R2 = {r2:.2f})')
    ax[1].legend(fontsize=7, ncol=2)
    ax[1].grid(alpha=0.3)
    fig.suptitle(f'{config}: inverse-free test of gaze/accommodation co-movement, all 20 fixations', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir/'observation_coupling.png', dpi=120)
    plt.close(fig)
    print('\n'.join(md))


if __name__ == '__main__':
    config = sys.argv[1] if len(sys.argv) > 1 else 'M2a01'
    write(config, analyse(config))
