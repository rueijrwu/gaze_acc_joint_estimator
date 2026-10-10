"""Raw-keystone Capture-1 A0 / gaze-only / joint-gaze-A ablation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
sys.path.insert(0, str(ROOT))
from distortion_model.joint_state import fit_states
from distortion_model.raw_keystone import RawJointState, RawKeystone, RawFrameAccommodation

STAGE3 = ROOT/'experiments/reverse_transform/raw_keystone/stage_03_independent_captures/results/run'
STAGE4 = ROOT/'experiments/reverse_transform/raw_keystone/stage_04_framewise_accommodation/results/run'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_inputs():
    with np.load(STAGE3/'frames.npz', allow_pickle=False) as z:
        p = dict(z)
    with np.load(STAGE4/'frames.npz', allow_pickle=False) as z:
        a = dict(z)
    s = json.loads((STAGE3/'summary.json').read_text())
    post = json.loads((STAGE3/'postfit.json').read_text())
    select = p['capture_index'] == 0
    select4 = a['capture_index'] == 0
    for key in ('population_index', 'row', 'source_frame', 'gaze_xy_deg', 'p1_magnification'):
        if not np.array_equal(p[key][select], a[key][select4]):
            raise ValueError(f'Baseline population mismatch: {key}')
    result = {name: p[name][select] for name in ('population_index', 'row', 'source_frame', 'exposure',
              'observed_centered_p1', 'observed_centered_p4', 'gaze_xy_deg', 'gaze_units_deg', 'p1_magnification')}
    result.update(reference_p1=p['reference_p1'], reference_p4=p['reference_p4'],
                  k1=np.array(s['captures'][0]['p1']['coefficients_scaled']),
                  k4=np.array(s['captures'][0]['p4']['coefficients_scaled'][:4]),
                  baseline_A=a['A_D'][select4], baseline_recovered_p4=a['recovered_p4'][select4],
                  slope=post['anchored_linear_slope_per_D_px2'],
                  reference_A=s['reference_demand_diopters_label'])
    return result


def make_model(p, joint=True, xp=np, gaze_width=.5):
    return RawJointState(p['observed_centered_p1'], p['observed_centered_p4'],
                      p['reference_p1'], p['reference_p4'], p['gaze_xy_deg'],
                      p['gaze_units_deg'], p['k1'], p['k4'], p['exposure'],
                      p['slope'], p['reference_A'], gaze_width=gaze_width, joint=joint, xp=xp)


def statistics(values):
    values = np.asarray(values).ravel()
    finite = values[np.isfinite(values)]
    if not len(finite):
        return dict(count=len(values), finite=0, mean=None, std=None, median=None, p95=None, rms=None)
    return dict(count=len(values), finite=len(finite), mean=float(finite.mean()), std=float(finite.std()),
                median=float(np.median(finite)), p95=float(np.percentile(finite, 95)),
                rms=float(np.sqrt(np.mean(finite**2))))


def correlation(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def radius(x):
    c = x-x.mean(axis=1, keepdims=True)
    return np.sqrt(np.mean(np.sum(c*c, axis=2), axis=1))


def inverse(p, states, m):
    gaze = p['gaze_xy_deg'].copy()
    gaze[:, 0] = states[:, 0]
    model1 = RawKeystone(gaze, p['reference_p1'], p['observed_centered_p1'],
                          np.ones(len(m)), p['gaze_units_deg'], m)
    back1, valid1 = model1.inverse(p['k1'], m)
    model4 = RawFrameAccommodation(p['observed_centered_p4'], p['reference_p4'], gaze,
                               p['gaze_units_deg'], np.tile(p['k4'], (len(m), 1)), m,
                               p['exposure'], np.full(5, p['reference_A']), p['slope'], p['reference_A'])
    back4, valid4 = model4.recover(states[:, 1])
    return back1, valid1, back4, valid4


def conditioning(jac):
    """Jacobian units px/degree and px/D. Correlation/angle are unit invariant."""
    j = np.asarray(jac).reshape(len(jac), -1, 2)
    singular = np.linalg.svd(j, compute_uv=False)
    norm = np.linalg.norm(j, axis=1)
    corr = np.sum(j[..., 0]*j[..., 1], axis=1)/np.maximum(norm[:, 0]*norm[:, 1], 1e-300)
    corr = np.clip(corr, -1, 1)
    condition = np.divide(singular[:, 0], singular[:, 1], out=np.full(len(j), np.inf), where=singular[:, 1] > 0)
    return dict(singular_max=singular[:, 0], singular_min=singular[:, 1], condition=condition,
                column_correlation=corr, angle_deg=np.rad2deg(np.arccos(corr)),
                acute_angle_deg=np.rad2deg(np.arccos(abs(corr))))


def collect(p, solutions, model, out):
    saved = {key: value for key, value in p.items() if key not in ('slope', 'reference_A')}
    arrays = {}
    for name, states in solutions.items():
        pred, jac, m, valid = model.forward(states)
        pred, jac, m, valid = [model.host(x) for x in pred], [model.host(x) for x in jac], model.host(m), model.host(valid)
        if not valid.all():
            raise ValueError(f'Invalid final forward geometry: {name}')
        if name == 'A0':
            if np.max(abs(m-p['p1_magnification'])) > 1e-10:
                raise ValueError('A0 profiled magnification changed')
            m = p['p1_magnification'].copy()
        b1, v1, b4, v4 = inverse(p, states, m)
        if name == 'A0' and np.max(abs(b4-p['baseline_recovered_p4'])) > 1e-8:
            raise ValueError('A0 inverse differs from preserved Stage04 baseline')
        arrays[name] = dict(states=states, magnification=m, prediction_p1=pred[0], prediction_p4=pred[1],
                            residual_p1=p['observed_centered_p1']-pred[0],
                            residual_p4=p['observed_centered_p4']-pred[1],
                            recovered_p1=b1, recovered_p4=b4, inverse_valid_p1=v1, inverse_valid_p4=v4)
        for label, j in (('p4', jac[1]), ('joint', np.concatenate(jac, axis=1))):
            for key, value in conditioning(j).items():
                arrays[name][f'{label}_{key}'] = value
    common = np.logical_and.reduce([v['inverse_valid_p4'] for v in arrays.values()])
    common1 = np.logical_and.reduce([v['inverse_valid_p1'] for v in arrays.values()])
    records = {}
    for name, data in arrays.items():
        def metrics(select):
            v4 = select & common
            v1 = select & common1
            states = data['states']
            result = dict(frames=int(select.sum()), inverse_common_frames=int(v4.sum()),
                theta_x_deg=statistics(states[select, 0]),
                delta_theta_deg=statistics(states[select, 0]-p['gaze_xy_deg'][select, 0]),
                A_D=statistics(states[select, 1]),
                delta_gaze_delta_A_correlation=correlation(states[select, 0]-p['gaze_xy_deg'][select, 0],
                                                          states[select, 1]-p['baseline_A'][select]),
                theta_lower_bound=int(np.sum(states[select, 0] <= -20+1e-7)),
                theta_upper_bound=int(np.sum(states[select, 0] >= 20-1e-7)),
                A_lower_bound=int(np.sum(states[select, 1] <= 1e-7)),
                A_upper_bound=int(np.sum(states[select, 1] >= 6-1e-7)),
                inverse_failures=int(np.sum(select & ~data['inverse_valid_p4'])),
                p1_forward_px=statistics(np.linalg.norm(data['residual_p1'][select], axis=2)),
                p4_forward_px=statistics(np.linalg.norm(data['residual_p4'][select], axis=2)),
                signed_p4_residual_mean_xy_px=data['residual_p4'][select].mean(axis=0).tolist(),
                p1_inverse_common_px=statistics(np.linalg.norm(data['recovered_p1'][v1]-p['reference_p1'], axis=2)),
                p4_inverse_common_px=statistics(np.linalg.norm(data['recovered_p4'][v4]-p['reference_p4'], axis=2)),
                p4_inverse_all_valid_px=statistics(np.linalg.norm(data['recovered_p4'][select & data['inverse_valid_p4']]-p['reference_p4'], axis=2)),
                observed_p4_P1scaled_radius_ratio=statistics(radius(p['observed_centered_p4'][select])/data['magnification'][select]/radius(p['reference_p4'][None])[0]),
                recovered_p4_radius_ratio=statistics(radius(data['recovered_p4'][v4])/radius(p['reference_p4'][None])[0]))
            result['conditioning'] = {label: {key: statistics(data[f'{label}_{key}'][select]) for key in
                ('singular_max', 'singular_min', 'condition', 'column_correlation', 'angle_deg', 'acute_angle_deg')}
                for label in ('p4', 'joint')}
            return result
        records[name] = dict(all=metrics(np.ones(len(common), dtype=bool)),
            fixations=[dict(exposure=f, nominal_gaze_deg=[-10, -5, 0, 5, 10][f], **metrics(p['exposure'] == f)) for f in range(5)])
        for key, value in data.items():
            saved[f'{name}_{key}'] = value
    saved['common_inverse_valid_p4'] = common
    saved['common_inverse_valid_p1'] = common1
    np.savez_compressed(out/'frames.npz', **saved)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import cupy as cp
    started = time.perf_counter()
    p = load_inputs()
    parents = {str(file.relative_to(ROOT)): digest(file) for folder in (STAGE3, STAGE4)
               for file in folder.rglob('*') if file.is_file()}
    protocol = dict(capture=1, complete=len(p['row']), keystone='Raw centered projective output, no RMS/area rescaling.', gaze_mean_anchor_width_deg=.5,
                    A_mean_anchor_width_D=.25, weights=dict(P1=1., P4=1.),
                    data_metric='Equal fixation mean of P1 mean-three-vertex SSE + P4 mean-three-vertex SSE in centered camera coordinates.',
                    anchors='Only arithmetic fixation-mean anchors; gaze nominal labels and A_ref. Residual scale 1 px.',
                    bounds=dict(theta_x_deg=[-20., 20.], A_D=[0., 6.]),
                    bounds_interpretation='Operational ranges, not framewise priors or calibration truth. Forward fitting does not restrict measured inverse-domain validity; failures are reported.',
                    frozen='References/origin, P1/P4 keystone, beta/A_ref, vertical gaze, units, correspondence, population.',
                    magnification='Positive P1 profiled magnification recomputed at every trial gaze; shared by P4.',
                    A0='Exact saved RAW Stage04 forward-fit A states/gaze/M, not refitted in this ablation.',
                    kappa_law=dict(slope=p['slope'], reference_A=p['reference_A']),
                    conditioning_units='Raw singular values/condition use degrees and diopters; normalized column correlations/angles are unit invariant.',
                    device=dict(backend='cupy', version=cp.__version__, dtype='float64',
                                gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode()))
    write(out/'protocol.json', protocol)
    sources = [Path(__file__), ROOT/'distortion_model/joint_state.py', ROOT/'distortion_model/raw_keystone.py',
               ROOT/'distortion_model/capture_shape.py', ROOT/'distortion_model/p1_shape.py', ROOT/'docs/REVERSE_TRANSFORM_NEXT_STEP.md']
    sources += [f for f in Path(__file__).parent.glob('*.py') if f not in sources]
    for source in sources:
        target = out/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    solutions = dict(A0=np.column_stack((p['gaze_xy_deg'][:, 0], p['baseline_A'])))
    fits = {}
    for name, joint in (('G', False), ('GA', True)):
        model = make_model(p, joint=joint, xp=cp)
        lower = np.tile([-20., 0.][:model.d], (model.n, 1))
        upper = np.tile([20., 6.][:model.d], (model.n, 1))
        records, candidates = [], []
        for offset in (0., -1., 1.):
            start = np.column_stack((p['gaze_xy_deg'][:, 0]+offset,
                                     np.clip(p['baseline_A']+offset*.25, 0, 6)))[:, :model.d]
            states, record = fit_states(model, np.clip(start, lower, upper), lower, upper)
            candidates.append(states)
            records.append(dict(initial_gaze_offset_deg=offset, **record))
            print(json.dumps(dict(model=name, **records[-1])), flush=True)
        best = int(np.argmin([r['objective_equal_fixation_px2'] for r in records]))
        fits[name] = dict(selected_start=best, starts=records,
                         objective_range=float(np.ptp([r['objective_equal_fixation_px2'] for r in records])))
        state = candidates[best]
        solutions[name] = state if joint else np.column_stack((state[:, 0], np.full(model.n, p['reference_A'])))
    model = make_model(p, xp=np)
    records = collect(p, solutions, model, out)
    passed = all(f['starts'][f['selected_start']]['certificate']['stationary_positive_curvature'] for f in fits.values())
    write(out/'summary.json', dict(status='COMPLETE' if passed else 'COMPLETE_NUMERICAL_LIMIT',
                                  protocol=protocol, fits=fits, comparisons=records,
                                  runtime_seconds=time.perf_counter()-started,
                                  limitations='Same recording, additional states; empirical relative radial law; no framewise ground truth. A0 is the new raw forward-P4/mean-anchor fit; G/GA fit joint forward P1/P4.'))
    write(out/'provenance.json', dict(started_UTC=datetime.now(timezone.utc).isoformat(), parent_sha256=parents,
                                     source_sha256={str(f.relative_to(ROOT)): digest(f) for f in sources}))
    if not all(digest(ROOT/key) == value for key, value in parents.items()):
        raise ValueError('Parent artifacts changed')
    print(json.dumps(dict(status='COMPLETE' if passed else 'COMPLETE_NUMERICAL_LIMIT',
                          runtime_seconds=time.perf_counter()-started)), flush=True)


if __name__ == '__main__':
    main()
