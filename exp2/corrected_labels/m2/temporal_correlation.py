#!/usr/bin/env python3
"""Temporal correlation of estimated gaze and accommodation, per held-out fixation (10-14, 3 D).

Reads one config of run_m2.sh (default M2a01): the Holdout-3 model and the Full model, both run through
the same anchor-free inverse on fixations 10-14 (predictions/holdout3/{trained_predictions,
fresh_matched_full_fixed_inverse}.csv), plus the raw measurements (m, S1, S4) of capture 3. Frames are 1 ms
apart. Gaps and glitch frames (see glitch_mask) are filled by linear interpolation for filtering/spectra, but
every statistic uses real, non-glitch samples only.

Bands (boxcar filters on the 1 kHz series):  slow = 501 ms moving average;
mid = (51 ms average) - (501 ms average);  fast = x - (51 ms average).

Outputs in <config>/temporal/: temporal_traces.png, temporal_coupling.png, temporal_slopes.png,
temporal_summary.md, temporal_summary.csv.   Usage (from this directory): python3 temporal_correlation.py [CONFIG] [--keep-glitches]
(--keep-glitches skips the glitch mask and writes to <config>/temporal_all_frames/.)
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
from scipy.ndimage import median_filter, uniform_filter1d, binary_dilation
from scipy.signal import coherence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))   # exp2/
import m2_model  # noqa: E402
from observations import relative_measurements  # noqa: E402

FIXATIONS = [10, 11, 12, 13, 14]
TARGETS = {10: -15.0, 11: -7.5, 12: 0.0, 13: 7.5, 14: 15.0}
FS = 1000.0
SLOW_W, MID_W = 501, 51
BANDS = ['slow', 'mid', 'fast']
MODELS = {'H3': 'Holdout-3', 'Full': 'Full'}
COLORS = {'H3': '#1f77b4', 'Full': '#ff7f0e'}
MAX_LAG = 200
GLITCH_MEDIAN_W, GLITCH_MAD_K, GLITCH_PAD = 301, 6.0, 50
CAPTURE_PKL = HERE.parents[1]/'capture_3_detections.pkl'


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    keys = ('frame_index', 'fixation_index', 'theta_deg', 'A_diopters', 'd', 'rho4')
    return {k: np.array([float(r[k]) for r in rows]) for k in keys}


def glitch_mask(theta, A):
    """True for frames far from a 301 ms running median in gaze or accommodation (robust 6 MAD), padded +-50 ms.

    Catches blink-like spikes and short plateaus (e.g. fixation 14, frames ~3600-3850); slow drift is kept.
    """
    flag = np.zeros(len(theta), bool)
    for x in (theta, A):
        dev = x-median_filter(x, GLITCH_MEDIAN_W, mode='nearest')
        mad = np.median(np.abs(dev-np.median(dev)))
        flag |= np.abs(dev) > GLITCH_MAD_K*1.4826*mad
    return binary_dilation(flag, iterations=GLITCH_PAD)


def uniform_series(frames, exclude, **series):
    """Interpolate onto the 1 ms frame grid across gaps and excluded frames; return grid, series, real-sample mask."""
    grid = np.arange(int(frames.min()), int(frames.max())+1)
    keep = ~exclude
    real = np.zeros(len(grid), bool)
    real[(frames[keep]-frames.min()).astype(int)] = True
    return grid, {k: np.interp(grid, frames[keep], v[keep]) for k, v in series.items()}, real


def split_bands(x):
    slow = uniform_filter1d(x, SLOW_W, mode='nearest')
    mid_avg = uniform_filter1d(x, MID_W, mode='nearest')
    return dict(slow=slow-slow.mean(), mid=mid_avg-slow, fast=x-mid_avg)


def pearson(a, b):
    a, b = a-a.mean(), b-b.mean()
    return float(a@b/np.sqrt((a@a)*(b@b)))


def slope(y, x):
    x = x-x.mean()
    return float(x@(y-y.mean())/(x@x))


def xcorr(a, b, max_lag):
    """Correlation of a[t] with b[t+lag] for lag in [-max_lag, max_lag] (frames)."""
    a, b = (a-a.mean())/a.std(), (b-b.mean())/b.std()
    lags = np.arange(-max_lag, max_lag+1)
    n = len(a)
    return lags, np.array([np.mean(a[max(0, -l):n-max(0, l)]*b[max(0, l):n-max(0, -l)]) for l in lags])


def analyse_model(tag, data, coef, exclude, raw):
    out = {}
    for fix in FIXATIONS:
        m = data['fixation_index'] == fix
        frames = data['frame_index'][m]
        grid, s, real = uniform_series(frames, exclude[fix], theta=data['theta_deg'][m], A=data['A_diopters'][m],
                                       d=data['d'][m], rho4=data['rho4'][m], m_px=raw[fix]['m'], S1=raw[fix]['S1'], S4=raw[fix]['S4'])
        bands = {k: split_bands(v) for k, v in s.items()}
        res = dict(grid=grid, real=real, series=s, bands=bands, n=int(m.sum()), filled=int((~real).sum()),
                   glitch=int(exclude[fix].sum()))
        # variance share per band
        res['var_share'] = {k: {b: float(np.var(bands[k][b][real])/sum(np.var(bands[k][c][real]) for c in BANDS)) for b in BANDS}
                            for k in ('theta', 'A')}
        res['sd'] = {k: float(np.std(s[k][real])) for k in s}
        res['corr_raw'] = pearson(s['theta'][real], s['A'][real])
        res['corr'] = {b: pearson(bands['theta'][b][real], bands['A'][b][real]) for b in BANDS}
        res['slope'] = {b: slope(bands['theta'][b][real], bands['A'][b][real]) for b in BANDS}
        res['corr_obs'] = {b: pearson(bands['d'][b][real], bands['rho4'][b][real]) for b in BANDS}
        # model-predicted coupling slopes (deg/D) from the Jacobian at the estimated states
        _, J = m2_model.forward_jac_m2(data['theta_deg'][m], data['A_diopters'][m], coef)
        a, b, c, e = J[:, 0, 0], J[:, 0, 1], J[:, 1, 0], J[:, 1, 1]
        res['pred_slope_rho'] = float(np.median(-b/a))   # error enters through rho4 only
        res['pred_slope_d'] = float(np.median(-e/c))     # error enters through d only
        # linearised attribution of theta/A variation to the d and rho4 observables
        Jm = np.array([[np.median(a), np.median(b)], [np.median(c), np.median(e)]])
        Jinv = np.linalg.inv(Jm)
        attr = {}
        for b_name in BANDS:
            dd, dr = bands['d'][b_name][real], bands['rho4'][b_name][real]
            th_d, th_r = Jinv[0, 0]*dd, Jinv[0, 1]*dr
            A_d, A_r = Jinv[1, 0]*dd, Jinv[1, 1]*dr
            th, A = bands['theta'][b_name][real], bands['A'][b_name][real]
            attr[b_name] = dict(
                theta_share_rho4=float(np.cov(th_r, th)[0, 1]/np.var(th, ddof=1)),
                A_share_rho4=float(np.cov(A_r, A)[0, 1]/np.var(A, ddof=1)),
                linearisation_r2_theta=float(1-np.var(th-(th_d+th_r))/np.var(th)),
                linearisation_r2_A=float(1-np.var(A-(A_d+A_r))/np.var(A)))
        res['attribution'] = attr
        # attribution to the raw measurements: rho4 = S4/S1 and d = m/S1, linearised with relative changes
        raw_attr = {}
        S1m, S4m, dm = s['S1'][real].mean(), s['S4'][real].mean(), s['d'][real].mean()
        rho_m = S4m/S1m
        for b_name in BANDS:
            s1r, s4r = bands['S1'][b_name][real]/S1m, bands['S4'][b_name][real]/S4m
            drho = rho_m*(s4r-s1r)
            dd = bands['m_px'][b_name][real]/S1m-dm*s1r
            Ab, thb = bands['A'][b_name][real], bands['theta'][b_name][real]
            raw_attr[b_name] = dict(
                sd_S1_pct=float(100*s1r.std()), sd_S4_pct=float(100*s4r.std()),
                rho4_share_S4=float(np.cov(rho_m*s4r, drho)[0, 1]/np.var(drho, ddof=1)),
                d_share_S1=float(np.cov(-dm*s1r, dd)[0, 1]/np.var(dd, ddof=1)),
                corr_A_S4=pearson(Ab, s4r), corr_A_S1=pearson(Ab, s1r), corr_theta_S1=pearson(thb, s1r))
        res['raw_attribution'] = raw_attr
        # cross-correlation (fast+mid band, and slow band separately) and coherence
        res['xcorr'] = {}
        for name, comb in [('slow', lambda B: B['slow']), ('mid+fast', lambda B: B['mid']+B['fast'])]:
            lags, cc = xcorr(comb(bands['theta']), comb(bands['A']), MAX_LAG)
            k = int(np.argmax(np.abs(cc)))
            res['xcorr'][name] = dict(lags=lags, cc=cc, peak_lag_ms=int(lags[k]), peak=float(cc[k]))
        f, cth = coherence(s['theta'], s['A'], fs=FS, nperseg=1000, detrend='linear')
        _, cobs = coherence(s['d'], s['rho4'], fs=FS, nperseg=1000, detrend='linear')
        res['coh'] = dict(f=f, theta_A=cth, d_rho4=cobs)
        out[fix] = res
    return out


def agreement(res_h3, res_full):
    out = {}
    for fix in FIXATIONS:
        a, b = res_h3[fix], res_full[fix]
        real = a['real']
        d_th = a['series']['theta']-b['series']['theta']
        d_A = a['series']['A']-b['series']['A']
        bt, bA = split_bands(d_th), split_bands(d_A)
        out[fix] = dict(
            corr_A_slow=pearson(a['bands']['A']['slow'][real], b['bands']['A']['slow'][real]),
            corr_theta_slow=pearson(a['bands']['theta']['slow'][real], b['bands']['theta']['slow'][real]),
            diff_theta_sd_slow=float(np.std(bt['slow'][real])), diff_theta_sd_fast=float(np.std((bt['mid']+bt['fast'])[real])),
            diff_A_sd_slow=float(np.std(bA['slow'][real])), diff_A_sd_fast=float(np.std((bA['mid']+bA['fast'])[real])),
            diff_theta_mean=float(d_th[real].mean()), diff_A_mean=float(d_A[real].mean()),
            corr_difference=pearson(d_th[real], d_A[real]))
    return out


def fig_traces(res, config, path):
    fig, axes = plt.subplots(5, 2, figsize=(15, 15))
    for row, fix in enumerate(FIXATIONS):
        r = res['H3'][fix]
        t = (r['grid']-r['grid'][0])/FS
        ax = axes[row, 0]
        ax.plot(t, r['bands']['theta']['slow'], color='#1f77b4', lw=1.6, label='Gaze, slow band (deg)')
        ax.set_ylabel('Gaze change (deg)', color='#1f77b4')
        ax2 = ax.twinx()
        ax2.plot(t, r['bands']['A']['slow'], color='#2ca02c', lw=1.6, label='Accommodation, slow band (D)')
        ax2.set_ylabel('Accommodation change (D)', color='#2ca02c')
        ax.set_title(f"Fixation {fix} (target {TARGETS[fix]:g} deg): slow band (>0.5 s), r = {r['corr']['slow']:+.2f}", fontsize=10)
        ax.grid(alpha=0.3)
        if row == 0:
            lines = ax.get_lines()+ax2.get_lines()
            ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc='upper right')
        ax = axes[row, 1]
        mid = len(t)//2
        sel = slice(mid-250, mid+250)
        fz = lambda x: np.where(r['real'], x/np.std(x[r['real']]), np.nan)[sel]   # NaN breaks the line over gaps/glitches
        both = lambda k: r['bands'][k]['mid']+r['bands'][k]['fast']
        ax.plot(t[sel], fz(both('theta')), color='#1f77b4', lw=0.9, label='Gaze (z)')
        ax.plot(t[sel], fz(both('A')), color='#2ca02c', lw=0.9, label='Accommodation (z)')
        ax.set_title(f"Fixation {fix}: mid+fast band (<0.5 s), 0.5 s excerpt, r = {pearson(both('theta')[r['real']], both('A')[r['real']]):+.2f}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_ylabel('z-score')
        if row == 0:
            ax.legend(fontsize=8, loc='upper right')
        if row == 4:
            axes[row, 0].set_xlabel('Time in fixation (s)')
            ax.set_xlabel('Time in fixation (s)')
    fig.suptitle(f'{config}: gaze and accommodation over time, Holdout-3 model, anchor-free inverse', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_coupling(res, config, path):
    fig, axes = plt.subplots(5, 2, figsize=(14, 15))
    for row, fix in enumerate(FIXATIONS):
        ax = axes[row, 0]
        for tag, ls in (('H3', '-'), ('Full', '--')):
            for name, col in (('slow', '#2ca02c'), ('mid+fast', '#d62728')):
                x = res[tag][fix]['xcorr'][name]
                ax.plot(x['lags'], x['cc'], color=col, ls=ls, lw=1.3,
                        label=f"{MODELS[tag]}, {name}" if row == 0 else None)
        ax.axvline(0, color='0.5', lw=0.8)
        ax.axhline(0, color='0.5', lw=0.8)
        ax.set_title(f"Fixation {fix}: cross-correlation, gaze vs accommodation", fontsize=10)
        ax.set_ylabel('Correlation')
        ax.grid(alpha=0.3)
        if row == 0:
            ax.legend(fontsize=7, loc='lower left', ncol=2)
        if row == 4:
            ax.set_xlabel('Lag of accommodation relative to gaze (ms)')
        ax = axes[row, 1]
        for tag, ls in (('H3', '-'), ('Full', '--')):
            c = res[tag][fix]['coh']
            sel = (c['f'] >= 1) & (c['f'] <= 200)
            ax.semilogx(c['f'][sel], c['theta_A'][sel], color=COLORS[tag], ls=ls, lw=1.3,
                        label=f'Estimates: {MODELS[tag]}' if row == 0 else None)
        c = res['H3'][fix]['coh']
        sel = (c['f'] >= 1) & (c['f'] <= 200)
        ax.semilogx(c['f'][sel], c['d_rho4'][sel], color='0.4', lw=1.3, label='Observations d vs rho4' if row == 0 else None)
        ax.set_ylim(0, 1)
        ax.set_title(f"Fixation {fix}: coherence", fontsize=10)
        ax.set_ylabel('Coherence')
        ax.grid(alpha=0.3, which='both')
        if row == 0:
            ax.legend(fontsize=7, loc='upper right')
        if row == 4:
            ax.set_xlabel('Frequency (Hz)')
    fig.suptitle(f'{config}: timing and frequency coupling of gaze and accommodation', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_slopes(res, config, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    x = np.arange(len(FIXATIONS))
    for ax, band in zip(axes, BANDS):
        w = 0.2
        ax.bar(x-w, [res['H3'][f]['slope'][band] for f in FIXATIONS], w, color='#1f77b4', label='Estimated slope, Holdout-3')
        ax.bar(x, [res['Full'][f]['slope'][band] for f in FIXATIONS], w, color='#ff7f0e', label='Estimated slope, Full')
        ax.bar(x+w, [res['H3'][f]['pred_slope_rho'] for f in FIXATIONS], w, color='#9ecae1', label='Predicted by model geometry (error via rho4)')
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{f}\n{TARGETS[f]:g} deg' for f in FIXATIONS])
        ax.set_title(f'{band} band')
        ax.grid(axis='y', alpha=0.3)
    axes[0].set_ylabel('Gaze change per accommodation change (deg / D)')
    axes[0].legend(fontsize=7, loc='upper left')
    fig.suptitle(f'{config}: measured gaze-accommodation slope vs slope predicted by the model geometry (the "via d" prediction, up to +-27 deg/D, is in the table only)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_summary(res, agree, config, out_dir):
    md = [f'# Temporal correlation of gaze and accommodation: {config}', '',
          'Held-out fixations 10-14 (3 D). Anchor-free inverse; 1 ms frames. Bands: slow > 0.5 s, mid 50-500 ms, fast < 50 ms.',
          'Statistics use real (non-interpolated) samples only.', '']
    rows = []
    for tag in ('H3', 'Full'):
        md += [f'## {MODELS[tag]} model', '',
               '| fixation | frames (filled; glitch-flagged) | SD gaze (deg) | SD A (D) | r raw | r slow | r mid | r fast | slope slow | slope mid | slope fast | predicted slope via rho4 | predicted slope via d |',
               '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for f in FIXATIONS:
            r = res[tag][f]
            md.append(f"| {f} | {r['n']} ({r['filled']}; {r['glitch']}) | {r['sd']['theta']:.3f} | {r['sd']['A']:.3f} | {r['corr_raw']:+.2f} | "
                      f"{r['corr']['slow']:+.2f} | {r['corr']['mid']:+.2f} | {r['corr']['fast']:+.2f} | "
                      f"{r['slope']['slow']:+.2f} | {r['slope']['mid']:+.2f} | {r['slope']['fast']:+.2f} | {r['pred_slope_rho']:+.2f} | {r['pred_slope_d']:+.2f} |")
            for b in BANDS:
                rows.append(dict(model=tag, fixation=f, band=b, corr=r['corr'][b], slope_deg_per_D=r['slope'][b],
                                 corr_observations_d_rho4=r['corr_obs'][b],
                                 var_share_theta=r['var_share']['theta'][b], var_share_A=r['var_share']['A'][b],
                                 theta_share_from_rho4=r['attribution'][b]['theta_share_rho4'],
                                 A_share_from_rho4=r['attribution'][b]['A_share_rho4'],
                                 pred_slope_via_rho4=r['pred_slope_rho'], pred_slope_via_d=r['pred_slope_d']))
        md += ['', 'Slopes are gaze change per accommodation change (deg / D). Predicted slopes are -(dd/dA)/(dd/dtheta) (error enters via rho4) and '
               '-(drho/dA)/(drho/dtheta) (error enters via d), medians over the fixation.', '',
               '| fixation | share of gaze variance slow / mid / fast | share of A variance slow / mid / fast | peak xcorr slow (lag ms) | peak xcorr mid+fast (lag ms) | r(d, rho4) slow / mid / fast |',
               '|---|---|---|---|---|---|']
        for f in FIXATIONS:
            r = res[tag][f]
            vt, va = r['var_share']['theta'], r['var_share']['A']
            md.append(f"| {f} | {vt['slow']:.2f} / {vt['mid']:.2f} / {vt['fast']:.2f} | {va['slow']:.2f} / {va['mid']:.2f} / {va['fast']:.2f} | "
                      f"{r['xcorr']['slow']['peak']:+.2f} ({r['xcorr']['slow']['peak_lag_ms']}) | {r['xcorr']['mid+fast']['peak']:+.2f} ({r['xcorr']['mid+fast']['peak_lag_ms']}) | "
                      f"{r['corr_obs']['slow']:+.2f} / {r['corr_obs']['mid']:+.2f} / {r['corr_obs']['fast']:+.2f} |")
        md += ['', 'Raw-measurement attribution (relative changes; rho4 = S4/S1, d = m/S1). Share of the rho4 band variance from the S4 term, '
               'share of the d band variance from the S1 term, SD of the relative S1 and S4 changes, and the correlation of the accommodation band with them.', '',
               '| fixation | band | SD dS1/S1 (%) | SD dS4/S4 (%) | rho4 share from S4 | d share from S1 | r(A, dS4/S4) | r(A, dS1/S1) | r(gaze, dS1/S1) |', '|---|---|---|---|---|---|---|---|---|']
        for f in FIXATIONS:
            for b in BANDS:
                a = res[tag][f]['raw_attribution'][b]
                md.append(f"| {f} | {b} | {a['sd_S1_pct']:.3f} | {a['sd_S4_pct']:.3f} | {a['rho4_share_S4']:.2f} | {a['d_share_S1']:.2f} | "
                          f"{a['corr_A_S4']:+.2f} | {a['corr_A_S1']:+.2f} | {a['corr_theta_S1']:+.2f} |")
        md += ['', 'Linearised attribution: share of the gaze (theta) and accommodation (A) band variance that comes from the rho4 observable '
               '(the rest comes from d). Shares sum to 1 with the d share. R2 = how well the linearisation reproduces the band.', '',
               '| fixation | band | theta share from rho4 | A share from rho4 | linearisation R2 theta | R2 A |', '|---|---|---|---|---|---|']
        for f in FIXATIONS:
            for b in BANDS:
                a = res[tag][f]['attribution'][b]
                md.append(f"| {f} | {b} | {a['theta_share_rho4']:.2f} | {a['A_share_rho4']:.2f} | {a['linearisation_r2_theta']:.3f} | {a['linearisation_r2_A']:.3f} |")
        md.append('')
    md += ['## Holdout-3 vs Full: temporal agreement', '',
           '| fixation | r(A) slow band | r(gaze) slow band | SD of gaze difference slow / mid+fast (deg) | SD of A difference slow / mid+fast (D) | mean gaze diff (deg) | mean A diff (D) | r(gaze diff, A diff) |',
           '|---|---|---|---|---|---|---|---|']
    for f in FIXATIONS:
        a = agree[f]
        md.append(f"| {f} | {a['corr_A_slow']:+.3f} | {a['corr_theta_slow']:+.3f} | {a['diff_theta_sd_slow']:.3f} / {a['diff_theta_sd_fast']:.3f} | "
                  f"{a['diff_A_sd_slow']:.3f} / {a['diff_A_sd_fast']:.3f} | {a['diff_theta_mean']:+.3f} | {a['diff_A_mean']:+.3f} | {a['corr_difference']:+.2f} |")
        for k, v in a.items():
            rows.append(dict(model='H3-Full', fixation=f, band=k, corr=v))
    md.append('')
    (out_dir/'temporal_summary.md').write_text('\n'.join(md))
    keys = sorted({k for r in rows for k in r}, key=lambda k: ['model', 'fixation', 'band'].index(k) if k in ('model', 'fixation', 'band') else 9)
    with open(out_dir/'temporal_summary.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    keep = '--keep-glitches' in sys.argv
    argv = [a for a in sys.argv[1:] if not a.startswith('--')]
    config = argv[0] if argv else 'M2a01'
    base = HERE/config
    pred = base/'predictions'/'holdout3'
    out_dir = base/('temporal_all_frames' if keep else 'temporal')
    out_dir.mkdir(exist_ok=True)
    data = {'H3': load(pred/'trained_predictions.csv'), 'Full': load(pred/'fresh_matched_full_fixed_inverse.csv')}
    assert np.array_equal(data['H3']['frame_index'], data['Full']['frame_index'])
    coefs = {'H3': np.array(json.loads((base/'training/holdout3/robust/model.json').read_text())['coefficients']),
             'Full': np.array(json.loads((base/'training/full/robust/model.json').read_text())['coefficients'])}
    arrays = pickle.load(open(CAPTURE_PKL, 'rb'))['arrays']
    z = relative_measurements(arrays)['z']
    row = {int(f): i for i, f in enumerate(np.asarray(arrays['frame_index']))}
    exclude, raw = {}, {}
    for fix in FIXATIONS:
        m = data['H3']['fixation_index'] == fix
        exclude[fix] = (np.zeros(int(m.sum()), bool) if keep else
                        glitch_mask(data['H3']['theta_deg'][m], data['H3']['A_diopters'][m]))
        idx = np.array([row[int(f)] for f in data['H3']['frame_index'][m]])
        raw[fix] = dict(m=z[idx, 0], S1=z[idx, 1], S4=z[idx, 2])
    res = {tag: analyse_model(tag, data[tag], coefs[tag], exclude, raw) for tag in data}
    agree = agreement(res['H3'], res['Full'])
    fig_traces(res, config, out_dir/'temporal_traces.png')
    fig_coupling(res, config, out_dir/'temporal_coupling.png')
    fig_slopes(res, config, out_dir/'temporal_slopes.png')
    write_summary(res, agree, config, out_dir)
    print('wrote', out_dir)


if __name__ == '__main__':
    main()
