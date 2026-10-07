#!/usr/bin/env python3
"""Plot the saved capture 5/6 gaze and accommodation estimates; no fitting."""
from __future__ import annotations
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

HERE = Path(__file__).resolve().parent


def read_estimates(values):
    if not values or 'elapsed_s' not in values:
        raise ValueError('No fitted estimate arrays in pickle payload')
    return dict(time=np.asarray(values['elapsed_s'],float), gaze=np.asarray(values['gaze_deg'],float),
                A=np.asarray(values['accommodation_D'],float), linear=np.asarray(values['linear_gaze_deg'],float),
                error=np.asarray(values['error_arcmin'],float))


def filter_plot_spikes(series, out):
    filtered, counts = {}, {}
    for c, original in series.items():
        s = {k: np.array(v, copy=True) for k, v in original.items()}
        t = s['time']; dt = np.median(np.diff(t)[np.diff(t) > 0])
        window = max(3, int(round(.3/dt)) | 1); bad = np.zeros(len(t), bool)
        for values, threshold in ((s['gaze'],1.), (s['A'],.5),
                                  (s['linear'],1.), (s['error'],30.)):
            valid = np.isfinite(values)
            if valid.any():
                local = median_filter(np.interp(t,t[valid],values[valid]),size=window,mode='nearest')
                bad |= valid & (np.abs(values-local)>threshold)
        for name in ('gaze','A','linear','error'): s[name][bad]=np.nan
        filtered[c]=s; counts[c]=int(bad.sum())
        with (out/f'capture_{c}_plot_mask.pkl').open('wb') as f:
            pickle.dump({'row':np.arange(len(t)), 'elapsed_s':t, 'excluded_spike':bad},f,protocol=pickle.HIGHEST_PROTOCOL)
    return filtered, dict(method='centered 0.3-second local median; common mask across panels',
        thresholds=dict(gaze_deg=1.0,accommodation_D=.5,difference_arcmin=30.0),excluded_rows=counts,
        removed_samples='NaN gaps in plotted lines; saved estimates remain unchanged')


def make_figures(series,out):
    series,filtering=filter_plot_spikes(series,out)
    plt.rcParams.update({'font.size':18,'axes.titlesize':20,'axes.labelsize':18,'xtick.labelsize':18,'ytick.labelsize':18})
    trials={5:'Trial 1',6:'Trial 2'}
    fig,ax=plt.subplots(2,2,figsize=(20,9),constrained_layout=True,sharex='col')
    for col,c in enumerate((5,6)):
        s=series[c]; ax[0,col].plot(s['time'],s['gaze'],'-',lw=1); ax[1,col].plot(s['time'],s['A'],'-',lw=1,color='tab:orange')
        ax[0,col].set(title=f"{trials[c]} - Gaze",ylabel='Horizontal gaze (deg)',ylim=(-12.5,7.5))
        ax[1,col].set(title=f"{trials[c]} - Accommodation",ylabel='Accommodation (D)',ylim=(0,4.5))
    for a in ax.flat:a.set_xlabel('Time (s)');a.grid(alpha=.25)
    fig_joint=fig
    fig,ax=plt.subplots(2,2,figsize=(20,9),constrained_layout=True,sharex='row')
    for row,c in enumerate((5,6)):
        s=series[c]; ax[row,0].plot(s['time'],s['linear'],'-',lw=1,color='tab:green'); ax[row,1].plot(s['time'],s['error'],'-',lw=1,color='tab:purple')
        ax[row,0].set(title=f"{trials[c]} - Gaze",ylabel='Horizontal gaze (deg)',ylim=(-12.5,7.5))
        ax[row,1].set(title=f"{trials[c]} - Error",ylabel='Gaze difference (arcmin)',ylim=(-50,50));ax[row,1].axhline(0,color='black',lw=.8)
    for a in ax.flat:a.set_xlabel('Time (s)');a.grid(alpha=.25)
    fig_comparison=fig
    (out/'plot_filter.json').write_text(json.dumps(filtering,indent=2)+'\n')
    plt.show(block=True)
    return filtering


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=HERE/'results/fitted_estimates.pkl',help='pickle containing saved capture 5/6 fitted estimates')
    p.add_argument('--output-dir',type=Path,default=HERE/'results')
    args=p.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    with args.data.open('rb') as f: saved=pickle.load(f)
    series={c:read_estimates(saved[c]) for c in (5,6)}
    filtering=make_figures(series,args.output_dir)
    print(json.dumps(dict(source=str(args.data.resolve()),processing='plotting saved estimates only; no model fitting or state estimation',plotting=filtering),indent=2))

if __name__=='__main__': main()
