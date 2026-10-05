#!/usr/bin/env python3
"""Summarize the m2 / piecewise corrected-label runs into summary.md and summary.csv (no pandas).

Reads <root>/<config>/{training,predictions,logs} as produced by run_m2.sh. Targets for fixations 10-14
come from the frozen fixation intervals plus target_overrides_v1.json (not from any model output); the
held-out demand is the interval label (3 D).
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
sys.path.insert(0, str(HERE))
import check_m2

CONFIGS = [('M0', 'piecewise, theta anchor 1.0 deg'), ('M2a1', 'm2, theta anchor 1.0 deg'), ('M2a01', 'm2, theta anchor 0.1 deg')]
HELDOUT = [10, 11, 12, 13, 14]
PIECEWISE_NAMES = ['b@0.36', 'b@2', 'b@3', 'b@4', 's@0.36', 's@2', 's@3', 's@4', 'r0', 'r1', 'r2', 'r3(a)', 'r4(a t)', 'r5(a t2)']


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def targets_and_demands(intervals, overrides):
    raw = Path(intervals).read_bytes()
    report = json.loads(raw)
    spec = json.loads(Path(overrides).read_bytes())
    if hashlib.sha256(raw).hexdigest() != spec['base_intervals_sha256']:
        raise ValueError('override base_intervals_sha256 mismatch')
    rows = report['fixations']
    nominal = sorted({r['target_theta_deg'] for r in rows})
    out = {}
    for j, r in enumerate(rows):
        t = r['target_theta_deg']
        if r['capture'] in spec['targets']:
            t = spec['targets'][r['capture']][nominal.index(t)]
        out[j] = (float(t), float(r['demand_diopters_label']))
    return out


def stats(v):
    v = np.asarray(v, float)
    return dict(n=len(v), bias=float(v.mean()), rmse=float(np.sqrt(np.mean(v*v))))


def vs_targets(rows, truth):
    out = {}
    for j in HELDOUT+['all']:
        local = [r for r in rows if j == 'all' or int(r['fixation_index']) == j]
        tg = np.array([truth[int(r['fixation_index'])][0] for r in local])
        dm = np.array([truth[int(r['fixation_index'])][1] for r in local])
        th = np.array([float(r['theta_deg']) for r in local])
        A = np.array([float(r['A_diopters']) for r in local])
        out[j] = dict(n=len(local), target_deg=float(tg.mean()) if j != 'all' else None,
                      gaze_bias=float((th-tg).mean()), gaze_rmse=float(np.sqrt(np.mean((th-tg)**2))),
                      A_bias=float((A-dm).mean()), A_rmse=float(np.sqrt(np.mean((A-dm)**2))))
    return out


def agreement(h3_rows, full_rows):
    key = lambda r: (int(r['fixation_index']), int(r['frame_index']))
    full = {key(r): r for r in full_rows}
    if set(full) != {key(r) for r in h3_rows}:
        raise ValueError('holdout3 / full inverse frame support mismatch')
    out = {}
    for j in HELDOUT+['all']:
        pairs = [(r, full[key(r)]) for r in h3_rows if j == 'all' or int(r['fixation_index']) == j]
        dth = np.array([float(a['theta_deg'])-float(b['theta_deg']) for a, b in pairs])
        dA = np.array([float(a['A_diopters'])-float(b['A_diopters']) for a, b in pairs])
        out[j] = dict(n=len(pairs), mean_gaze_diff=float(dth.mean()), gaze_rmse=float(np.sqrt(np.mean(dth**2))),
                      mean_A_diff=float(dA.mean()), A_rmse=float(np.sqrt(np.mean(dA**2))))
    return out


def training_info(cfg_dir, fold):
    info = {}
    wall = {}
    wall_file = cfg_dir/'logs'/'wall_seconds.tsv'
    if wall_file.exists():
        wall = {k: float(v) for k, v in (line.split('\t') for line in wall_file.read_text().splitlines())}
    info['driver_wall_seconds'] = wall.get(f'train_{fold}')
    for stage in ['quadratic', 'robust']:
        model = json.loads((cfg_dir/'training'/fold/stage/'model.json').read_text())
        st = model['diagnostics']['continuation_status']
        info[stage] = dict(status=st['status'], accepted_iterations=st['accepted_iterations'], nfev=st['nfev'],
                           elapsed_seconds=st['elapsed_seconds'], termination=st['termination'])
        if stage == 'robust':
            info['model'] = model
    return info


def coefficient_table(model):
    coef = model['coefficients']
    names = model.get('coefficient_names') or PIECEWISE_NAMES
    rows = list(zip(names, coef))
    if model.get('p') is not None:
        rows.append(('p (frozen exponent)', model['p']))
    return rows


def fmt(x, digits=4):
    return 'n/a' if x is None else f'{x:.{digits}f}'


def summarize_config(root, cfg, truth):
    cfg_dir = root/cfg
    res = dict(config=cfg)
    res['training'] = {fold: training_info(cfg_dir, fold) for fold in ['holdout3', 'full']}
    pred = cfg_dir/'predictions'/'holdout3'
    h3 = read_csv(pred/'trained_predictions.csv')
    full = read_csv(pred/'fresh_matched_full_fixed_inverse.csv')
    res['holdout3_inverse_vs_targets'] = vs_targets(h3, truth)
    res['full_inverse_vs_targets'] = vs_targets(full, truth)
    res['agreement'] = agreement(h3, full)
    res['monotonicity'] = {fold: list(check_m2.check_b([cfg_dir/'training'/fold/'robust'/'model.json']).values())[0]
                           for fold in ['holdout3', 'full']}
    res['inverse_diagnostics'] = {}
    for label, rows in [('holdout3_inverse', h3), ('full_inverse', full)]:
        res['inverse_diagnostics'][label] = dict(
            frames=len(rows), ambiguous=sum(int(r['equivalent_minima_count']) > 1 for r in rows),
            stationarity_unverified=sum(r['stationarity_verified'] != 'True' for r in rows),
            theta_bound=sum(r['theta_bound'] == 'True' for r in rows), A_bound=sum(r['A_bound'] == 'True' for r in rows),
            max_weighted_cost=max(float(r['weighted_cost']) for r in rows))
    return res


def write_outputs(root, results, descriptions):
    csv_rows, md = [], ['# m2 vs piecewise: corrected-label runs', '',
                        'Targets for fixations 10-14 come from `fixation_intervals.json` + `target_overrides_v1.json` '
                        '(capture 3 is not overridden); held-out demand is 3 D. Errors are model state minus nominal target '
                        '(retrospective model-state deviations, not measured physiological accuracy). Gaze in degrees, A in diopters.', '']
    def add(cfg, section, item, metric, value):
        csv_rows.append(dict(config=cfg, section=section, item=item, metric=metric, value=value))
    for res in results:
        cfg = res['config']
        md += [f'## {cfg}: {descriptions[cfg]}', '', '### Training convergence', '',
               '| fold | stage | status | accepted iter | nfev | solver s | driver wall s |', '|---|---|---|---|---|---|---|']
        for fold, info in res['training'].items():
            for stage in ['quadratic', 'robust']:
                st = info[stage]
                md.append(f"| {fold} | {stage} | {st['status']} | {st['accepted_iterations']} | {st['nfev']} | {st['elapsed_seconds']:.1f} | "
                          f"{fmt(info['driver_wall_seconds'], 1) if stage == 'robust' else ''} |")
                for k in ['status', 'accepted_iterations', 'nfev', 'elapsed_seconds']:
                    add(cfg, 'training', f'{fold}/{stage}', k, st[k])
            add(cfg, 'training', fold, 'driver_wall_seconds', info['driver_wall_seconds'])
        md += ['', '### Coefficients (robust stage)', '']
        tables = {fold: coefficient_table(info['model']) for fold, info in res['training'].items()}
        names = [n for n, _ in tables['holdout3']]
        md += ['| coefficient | holdout3 | full |', '|---|---|---|']
        for (name, a), (_, b) in zip(tables['holdout3'], tables['full']):
            md.append(f'| {name} | {a:.6g} | {b:.6g} |')
            add(cfg, 'coefficient', 'holdout3', name, a); add(cfg, 'coefficient', 'full', name, b)
        for title, key in [('Held-out fixations 10-14, holdout3-model anchor-free inverse vs targets', 'holdout3_inverse_vs_targets'),
                           ('Held-out fixations 10-14, full-model (same anchor-free inverse) vs targets', 'full_inverse_vs_targets')]:
            md += ['', f'### {title}', '', '| fixation | target deg | n | gaze bias | gaze RMSE | A bias | A RMSE |', '|---|---|---|---|---|---|---|']
            for j, m in res[key].items():
                md.append(f"| {j} | {fmt(m['target_deg'], 3)} | {m['n']} | {m['gaze_bias']:.3f} | {m['gaze_rmse']:.3f} | {m['A_bias']:.3f} | {m['A_rmse']:.3f} |")
                for k in ['n', 'gaze_bias', 'gaze_rmse', 'A_bias', 'A_rmse']:
                    add(cfg, key, str(j), k, m[k])
        md += ['', '### Agreement: holdout3 inverse vs full inverse (holdout3 minus full)', '',
               '| fixation | n | mean gaze diff | gaze RMSE | mean A diff | A RMSE |', '|---|---|---|---|---|---|']
        for j, m in res['agreement'].items():
            md.append(f"| {j} | {m['n']} | {m['mean_gaze_diff']:.3f} | {m['gaze_rmse']:.3f} | {m['mean_A_diff']:.3f} | {m['A_rmse']:.3f} |")
            for k in ['n', 'mean_gaze_diff', 'gaze_rmse', 'mean_A_diff', 'A_rmse']:
                add(cfg, 'agreement', str(j), k, m[k])
        md += ['', '### Monotonicity of d in theta, theta in [-20,20], A in [0,6]', '', '| fold | min dd/dtheta | at theta | at A | strictly monotonic |', '|---|---|---|---|---|']
        for fold, m in res['monotonicity'].items():
            md.append(f"| {fold} | {m['min_dd_dtheta']:.5f} | {m['at_theta_deg']:.1f} | {m['at_A_D']:.2f} | {m['strictly_monotonic']} |")
            add(cfg, 'monotonicity', fold, 'min_dd_dtheta', m['min_dd_dtheta']); add(cfg, 'monotonicity', fold, 'strictly_monotonic', m['strictly_monotonic'])
        md += ['', '### Inverse diagnostics (frames)', '', '| inverse | frames | ambiguous | stationarity unverified | theta bound | A bound | max cost |', '|---|---|---|---|---|---|---|']
        for label, d in res['inverse_diagnostics'].items():
            md.append(f"| {label} | {d['frames']} | {d['ambiguous']} | {d['stationarity_unverified']} | {d['theta_bound']} | {d['A_bound']} | {d['max_weighted_cost']:.2e} |")
            for k, v in d.items():
                add(cfg, 'inverse_diagnostics', label, k, v)
        md.append('')
    (root/'summary.md').write_text('\n'.join(md)+'\n')
    with (root/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['config', 'section', 'item', 'metric', 'value'])
        writer.writeheader(); writer.writerows(csv_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE)
    parser.add_argument('--configs', nargs='*', default=[c for c, _ in CONFIGS])
    parser.add_argument('--intervals', type=Path, default=EXP/'fixations/fixation_intervals.json')
    parser.add_argument('--target-overrides', type=Path, default=EXP/'corrected_labels/target_overrides_v1.json')
    args = parser.parse_args()
    truth = targets_and_demands(args.intervals, args.target_overrides)
    descriptions = dict(CONFIGS)
    results = []
    for cfg in args.configs:
        if not (args.root/cfg/'predictions'/'holdout3'/'fresh_matched_full_fixed_inverse.csv').exists():
            print(f'skipping {cfg}: run_m2.sh has not completed for it', file=sys.stderr)
            continue
        results.append(summarize_config(args.root, cfg, truth))
    if not results:
        sys.exit('no completed configs')
    write_outputs(args.root, results, descriptions)
    print('wrote', args.root/'summary.md', args.root/'summary.csv')


if __name__ == '__main__':
    main()
