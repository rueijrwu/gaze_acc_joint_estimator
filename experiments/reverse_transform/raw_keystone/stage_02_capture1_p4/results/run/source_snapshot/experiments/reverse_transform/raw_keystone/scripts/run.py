"""Sequential raw-keystone chain: fresh P1, P4, all captures, forward A."""
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
from distortion_model.data import load_reviewed, make_population, array_hash
from distortion_model.raw_keystone import RawKeystone, RawFrameAccommodation, center
from helpers import calibrate_gaze, statistics, limits, fitting, radius, fit_accommodation

CHAIN = ROOT/'experiments/reverse_transform/raw_keystone'
NAMES = {1: 'stage_01_capture1_p1', 2: 'stage_02_capture1_p4',
         3: 'stage_03_independent_captures', 4: 'stage_04_framewise_accommodation'}
HIST = ROOT/'experiments/reverse_transform'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parent(stage):
    return CHAIN/NAMES[stage]/'results/run'


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def errors(points, reference):
    valid = np.isfinite(points).all(axis=(1, 2))
    return dict(valid_frames=int(valid.sum()), invalid_frames=int((~valid).sum()),
                point_distance_px=statistics(np.linalg.norm(points[valid]-reference, axis=2)),
                radius_ratio=statistics(radius(points[valid])/radius(reference)),
                signed_vertex_mean_xy_px=np.mean(points[valid]-reference, axis=0).tolist() if valid.any() else None)


def metrics(p):
    records = []
    for f in np.unique(p['exposure']):
        sel = p['exposure'] == f
        record = dict(exposure=int(f), capture=int(f//5+1), nominal_gaze_deg=[-10,-5,0,5,10][f%5],
                      complete=int(sel.sum()), gaze_x_deg=statistics(p['gaze_xy_deg'][sel, 0]),
                      common_P1_scale=statistics(p['p1_magnification'][sel]))
        for n in (1, 4):
            if f'forward_residual_p{n}' not in p:
                continue
            record[f'p{n}_forward_point_distance_px'] = statistics(np.linalg.norm(p[f'forward_residual_p{n}'][sel], axis=2))
            record[f'p{n}_inverse'] = errors(p[f'recovered_p{n}'][sel], p[f'reference_p{n}'])
            record[f'p{n}_raw_keystone_size_factor'] = statistics(p[f'p{n}_raw_size_factor'][sel])
            record[f'p{n}_signed_forward_vertex_mean_xy_px'] = p[f'forward_residual_p{n}'][sel].mean(axis=0).tolist()
        if 'A_D' in p:
            record['A_D'] = statistics(p['A_D'][sel])
            record['expected_A_D'] = float(p['expected_A_D'][sel][0])
            record['mean_anchor_shift_D'] = float(p['A_D'][sel].mean()-p['expected_A_D'][sel][0])
        if 'observed_centered_p4' in p:
            record['observed_P4_over_P1scaled_reference_radius'] = statistics(radius(p['observed_centered_p4'][sel])/p['p1_magnification'][sel]/radius(p['reference_p4']))
        records.append(record)
    return records


def install_p1(p, sel, k, model):
    pred, m = model.prediction(k)
    back, valid = model.inverse(k, m)
    h = model.host(model.shape(k)[0])
    p['p1_magnification'][sel] = m
    p['p1_keystone_scaled'][sel] = k
    p['prediction_p1'][sel] = pred
    p['forward_residual_p1'][sel] = p['observed_centered_p1'][sel]-pred
    p['recovered_p1'][sel] = back
    p['p1_inverse_valid'][sel] = valid
    p['p1_raw_size_factor'][sel] = radius(h)/radius(p['reference_p1'])


def install_p4(p, sel, k, model):
    pred, _ = model.prediction(k)
    back, valid = model.inverse(k)
    h = model.host(model.shape(k)[0])
    p['p4_keystone_scaled'][sel] = k[:4]
    p['kappa_per_px2'][sel] = (k[4] if len(k) == 5 else 0.)/radius(p['reference_p4'])**2
    p['prediction_p4'][sel] = pred
    p['forward_residual_p4'][sel] = p['observed_centered_p4'][sel]-pred
    p['recovered_p4'][sel] = back
    p['p4_inverse_valid'][sel] = valid
    p['p4_raw_size_factor'][sel] = radius(h)/radius(p['reference_p4'])


def allocate(p, n, patterns):
    for pattern in patterns:
        p[f'p{pattern}_keystone_scaled'] = np.empty((n, 4))
        for field in ('prediction', 'forward_residual', 'recovered'):
            p[f'{field}_p{pattern}'] = np.empty((n, 3, 2))
        p[f'p{pattern}_inverse_valid'] = np.zeros(n, dtype=bool)
        p[f'p{pattern}_raw_size_factor'] = np.empty(n)
    if 1 in patterns:
        p['p1_magnification'] = np.empty(n)
    if 4 in patterns:
        p['kappa_per_px2'] = np.empty(n)


def population(stage):
    captures, intervals, _, roster_hash = load_reviewed(ROOT, workers=4)
    if stage == 1:
        intervals = [i for i in intervals if i['capture'] == 'capture_1_detections.pkl']
    pop = make_population(captures, intervals)
    idx = np.flatnonzero(pop['complete_valid'])
    cap = np.array([int(name.split('_')[1])-1 for name in pop['capture'][idx]])
    x1, x4 = center(pop['p1'][idx]), center(pop['p4'][idx])
    exp = pop['exposure'][idx]
    zero = (cap == 0) & (exp % 5 == 2)
    p = dict(population_index=idx, capture_index=cap, exposure=exp, row=pop['row'][idx],
             source_frame=pop['source_frame'][idx], observed_centered_p1=x1, observed_centered_p4=x4,
             reference_p1=x1[zero].mean(axis=0), reference_p4=x4[zero].mean(axis=0),
             gaze_xy_deg=np.empty((len(idx), 2)), gaze_units_deg=np.empty((len(idx), 2)))
    return pop, p, dict(roster_sha256=roster_hash,
                       input_sha256={str((ROOT/'data/detections'/name).relative_to(ROOT)): cap.digest for name, cap in captures.items()},
                       population_semantic_hash=array_hash(pop))


def historical_comparison(stage, p):
    paths = {1: HIST/'stage_01_capture1_p1_fit/results/attempt_01/frames.npz',
             2: HIST/'stage_02_capture1_p4_fit/results/run/frames.npz',
             3: HIST/'stage_03_independent_captures/results/run/frames.npz',
             4: HIST/'stage_04_framewise_accommodation/results/run/frames.npz'}
    old = load_npz(paths[stage])
    assert np.array_equal(old['row'], p['row']) and np.array_equal(old['population_index'], p['population_index'])
    records = []
    for f in np.unique(p['exposure']):
        sel = p['exposure'] == f
        report = dict(exposure=int(f), nominal_gaze_deg=[-10,-5,0,5,10][f%5])
        for n in (1, 4):
            if f'forward_residual_p{n}' not in p:
                continue
            oldkey = 'forward_residual' if stage == 1 or n == 4 else None
            if oldkey is not None and oldkey in old:
                report[f'historical_p{n}_forward_px'] = statistics(np.linalg.norm(old[oldkey][sel], axis=2))
            oldback = f'recovered_p{n}'
            if oldback in old:
                report[f'historical_p{n}_inverse'] = errors(old[oldback][sel], p[f'reference_p{n}'])
                both = sel & np.isfinite(old[oldback]).all(axis=(1, 2)) & np.isfinite(p[oldback]).all(axis=(1, 2))
                report[f'common_valid_p{n}'] = int(both.sum())
                report[f'historical_p{n}_inverse_common'] = errors(old[oldback][both], p[f'reference_p{n}'])
                report[f'raw_p{n}_inverse_common'] = errors(p[oldback][both], p[f'reference_p{n}'])
            # Conditional inverse-keystone geometry before radial correction.
            if n == 4 and 'keystone_recovered_p4' in old and stage in (2, 3):
                report['historical_post_keystone_radius_ratio'] = statistics(radius(old['keystone_recovered_p4'][sel])/radius(p['reference_p4']))
        if stage == 4:
            report['historical_A_D'] = statistics(old['A_D'][sel])
        records.append(report)
    return dict(control='Historical normalized-keystone result; read only after new fitting. No old fitted coefficient, scale, radial law or state used for calibration.',
                path=str(paths[stage].relative_to(ROOT)), sha256=digest(paths[stage]), fixations=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=range(1, 5), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import cupy as cp
    start = time.perf_counter()
    stage = args.stage
    prior = stage-1 if stage < 4 else 3
    parents = {} if stage == 1 else {str(f.relative_to(ROOT)): digest(f) for f in parent(prior).rglob('*') if f.is_file()}
    protocol = dict(stage=stage, keystone='Raw centered projective output. NO post-keystone RMS/area rescaling.',
                    reference='Capture1 zero-gaze centered arithmetic means, demand0.36036036036036034D.',
                    scale='One positive P1-profiled scalar per frame, shared with P4; no P4 scale.',
                    metric='Original centered camera mean-three-vertex SSE, equal fixation weights; inverse secondary.',
                    old_results='Historical read-only comparisons only; no old fitted optics/states used.',
                    P1_gauge='Four-coefficient exp(a), exp(-a) and generalized projective denominator; identity at zero gaze; no arbitrary isotropic gaze function.',
                    device=dict(backend='cupy', version=cp.__version__, dtype='float64', gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode()))
    if stage == 4:
        protocol.update(anchor_width_D=.25, residual_scale_px=1., strength_px2_per_D2=16.,
                        anchors='Arithmetic fixation means only; no per-frame prior/smoothing.', A_bounds_D=[0., 6.],
                        domain='Forward model branch bounds only. Measured inverse failures reported, not trimmed.',
                        frozen='New raw Stage03 gaze, P1 scale, per-capture keystone and newly fitted radial law.')
    write(out/'protocol.json', protocol)
    sources = [Path(__file__).resolve(), Path(__file__).with_name('helpers.py'), ROOT/'distortion_model/raw_keystone.py',
               ROOT/'distortion_model/data.py', ROOT/'distortion_model/capture_shape.py', ROOT/'distortion_model/p1_shape.py',
               ROOT/'docs/REVERSE_TRANSFORM_NEXT_STEP.md']
    for source in sources:
        target = out/'source_snapshot'/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    inputs = {}
    if stage in (1, 3):
        pop, p, inputs = population(stage)
        allocate(p, len(p['row']), (1,) if stage == 1 else (1, 4))
        records = []
        reused = load_npz(parent(2)/'frames.npz') if stage == 3 else None
        previous = json.loads((parent(2)/'summary.json').read_text()) if stage == 3 else None
        for c in np.unique(p['capture_index']):
            sel = p['capture_index'] == c
            exp = p['exposure'][sel] % 5
            count = np.bincount(exp, minlength=5)
            w = 1/(15*count[exp])
            theta, calibration = calibrate_gaze(pop['p4'][p['population_index'][sel]].mean(axis=1)-pop['p1'][p['population_index'][sel]].mean(axis=1), exp, np.array([-10.,-5.,0.,5.,10.]))
            units = np.array([10., max(np.max(abs(theta[:, 1])), .1)])
            p['gaze_xy_deg'][sel], p['gaze_units_deg'][sel] = theta, units
            m1 = RawKeystone(theta, p['reference_p1'], p['observed_centered_p1'][sel], w, units, xp=cp)
            if c == 0 and stage == 3:
                for key in ('row', 'source_frame', 'gaze_xy_deg', 'gaze_units_deg'):
                    assert np.array_equal(p[key][sel], reused[key])
                rec = previous['captures'][0].copy()
                k1 = np.array(rec['p1']['coefficients_scaled'])
            else:
                k1, fit1 = fitting(m1, limits(theta, p['reference_p1'], units, False))
                rec = dict(capture=int(c+1), gaze_calibration=calibration, p1=fit1)
            install_p1(p, sel, k1, m1)
            if stage == 3:
                m4 = RawKeystone(theta, p['reference_p4'], p['observed_centered_p4'][sel], w, units, p['p1_magnification'][sel], cp)
                if c == 0:
                    k4 = np.array(rec['p4']['coefficients_scaled'])
                    assert np.max(abs(p['p1_magnification'][sel]-reused['p1_magnification'])) < 1e-10
                else:
                    k4, rec['p4'] = fitting(m4, limits(theta, p['reference_p4'], units, True))
                install_p4(p, sel, k4, m4)
                rec['relative_barrel_per_px2'] = float((k4[4] if len(k4) == 5 else 0.)/radius(p['reference_p4'])**2)
            rec['demand_diopters_label'] = float(pop['demand_diopters'][p['population_index'][sel]][0])
            records.append(rec)
            print(json.dumps(dict(capture=int(c+1), p1=rec['p1']['checks'], p4=rec.get('p4', {}).get('checks'))), flush=True)
    else:
        pop = load_npz(parent(prior)/'population.npz')
        p = load_npz(parent(prior)/'frames.npz')
        previous = json.loads((parent(prior)/'summary.json').read_text())
        records = previous['captures']
        if stage == 2:
            allocate(p, len(p['row']), (4,))
            exp = p['exposure']
            w = 1/(15*np.bincount(exp, minlength=5)[exp])
            model = RawKeystone(p['gaze_xy_deg'], p['reference_p4'], p['observed_centered_p4'], w, p['gaze_units_deg'], p['p1_magnification'], cp)
            k4, records[0]['p4'] = fitting(model, limits(p['gaze_xy_deg'], p['reference_p4'], p['gaze_units_deg'][0], False))
            records[0]['relative_barrel_per_px2'] = 0.
            install_p4(p, np.ones(len(exp), dtype=bool), k4, model)
        else:
            post = json.loads((parent(3)/'postfit.json').read_text())
            slope = post['anchored_linear_slope_per_D_px2']
            aref = records[0]['demand_diopters_label']
            index, exp = p['population_index'], p['exposure']
            expected_rows = pop['demand_diopters'][index]
            expected = np.array([expected_rows[exp == f][0] for f in range(20)])
            model = RawFrameAccommodation(p['observed_centered_p4'], p['reference_p4'], p['gaze_xy_deg'], p['gaze_units_deg'],
                    p['p4_keystone_scaled'], p['p1_magnification'], exp, expected, slope, aref, xp=cp)
            lo, hi = model.feasible_bounds()
            starts, candidates = [], []
            for offset in (0., -.5, .5):
                A, record = fit_accommodation(model, lo, hi, expected_rows+offset)
                starts.append(dict(initial_offset_D=offset, **record))
                candidates.append(A)
                print(json.dumps(starts[-1]), flush=True)
            best = int(np.argmin([r['objective_scaled_sum'] for r in starts]))
            A = candidates[best]
            pred, _, _, kap, valid = model.geometry(cp.asarray(A))
            back, inverse_valid = model.recover(cp.asarray(A))
            p.update(A_D=A, expected_A_D=expected_rows, lower_A_D=lo, upper_A_D=hi,
                     kappa_per_px2=model.host(kap), prediction_p4=model.host(pred),
                     forward_residual_p4=p['observed_centered_p4']-model.host(pred),
                     recovered_p4=back, p4_inverse_valid=inverse_valid,
                     p4_raw_size_factor=radius(model.host(pred)/p['p1_magnification'][:, None, None])/radius(p['reference_p4']))
            protocol.update(kappa_law=dict(slope=slope, reference_A=aref))
            write(out/'frame_fit.json', dict(selected_start=best, starts=starts,
                       objective_multistart_range=float(np.ptp([s['objective_equal_fixation_mean_px2'] for s in starts]))))
            if not model.host(valid).all():
                raise ValueError('Invalid final raw forward A domain')
    if stage == 3:
        demand = np.array([r['demand_diopters_label'] for r in records])
        kap = np.array([r['relative_barrel_per_px2'] for r in records])
        delta = demand-demand[0]
        slope = float(delta@kap/(delta@delta))
        write(out/'postfit.json', dict(anchored_linear_slope_per_D_px2=slope, reference_A_D=float(demand[0]),
              capture_coefficients=[dict(capture=r['capture'], demand_D=r['demand_diopters_label'], kappa_per_px2=r['relative_barrel_per_px2']) for r in records],
              coefficient_residuals_per_px2=(kap-slope*delta).tolist(),
              P4_keystone_scaled_by_capture=[r['p4']['coefficients_scaled'][:4] for r in records],
              accommodation_keystone_policy='Independent capture coefficients retained and inspected. A/capture confounded; trends alone do not support promoting shared K(theta,A). Any coupling needs a declared coordinate/identifiability comparison.',
              limitations='Post-fit demand association only; independent fits receive no demand inputs. Relative coefficients, not physical absolute barrel.'))
    passed = all(r['p1']['checks']['stationary'] and (stage == 1 or r['p4']['checks']['stationary']) for r in records)
    if stage == 4:
        passed &= starts[best]['certificate']['stationary']
    comparison = historical_comparison(stage, p)
    write(out/'historical_comparison.json', comparison)
    np.savez_compressed(out/'population.npz', **pop)
    np.savez_compressed(out/'frames.npz', **p)
    summary = dict(status='COMPLETE' if passed else 'COMPLETE_NUMERICAL_LIMIT', stage=stage, protocol=protocol,
                   scheduled=len(pop['row']), complete=len(p['row']), unavailable=len(pop['row'])-len(p['row']),
                   reference_demand_diopters_label=records[0]['demand_diopters_label'], captures=records,
                   fixations=metrics(p), runtime_seconds=time.perf_counter()-start,
                   limitations='In-sample empirical-reference fitting. Vertical slope assumed. Demand/capture confounded; no independent gaze/accommodation truth.')
    for n in (1, 4):
        if f'forward_residual_p{n}' in p:
            summary[f'p{n}_forward_point_distance_px'] = statistics(np.linalg.norm(p[f'forward_residual_p{n}'], axis=2))
            summary[f'p{n}_inverse'] = errors(p[f'recovered_p{n}'], p[f'reference_p{n}'])
    write(out/'summary.json', summary)
    write(out/'protocol.json', protocol)
    write(out/'provenance.json', dict(started_UTC=datetime.now(timezone.utc).isoformat(), parent_sha256=parents,
              source_sha256={str(s.relative_to(ROOT)): digest(s) for s in sources}, historical_control_sha256={comparison['path']: comparison['sha256']}, **inputs))
    assert all(digest(ROOT/key) == val for key, val in parents.items())
    print(json.dumps(dict(status=summary['status'], stage=stage, complete=summary['complete'], runtime_seconds=summary['runtime_seconds'])), flush=True)


if __name__ == '__main__':
    main()
