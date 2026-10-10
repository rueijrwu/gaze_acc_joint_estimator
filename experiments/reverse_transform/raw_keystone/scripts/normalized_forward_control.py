"""Historical normalized optics refitted ONLY as a matched-forward-loss control.

This comparison never supplies calibration/state inputs to the raw chain.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
sys.path.insert(0, str(ROOT))
from distortion_model.raw_keystone import RawFrameAccommodation, center
from distortion_model.frame_accommodation import FrameAccommodation
from helpers import fit_accommodation, statistics
import run


class NormalizedForwardControl(RawFrameAccommodation):
    def geometry(self, A):
        xp = self.xp
        raw, draw, mu, kap, valid = super().geometry(A)
        pre = self.b[None]*(1+kap[:, None]*self.r2)[..., None]
        dp = self.b[None]*(self.slope*self.r2)[None, :, None]
        pc, dpc = center(pre, xp), center(dp, xp)
        size = xp.sqrt(xp.sum(raw*raw, axis=(1, 2))/self.m**2/xp.sum(pc*pc, axis=(1, 2)))
        dlog = (xp.sum(raw*draw, axis=(1, 2))/xp.sum(raw*raw, axis=(1, 2))-
                xp.sum(pc*dpc, axis=(1, 2))/xp.sum(pc*pc, axis=(1, 2)))
        prediction = raw/size[:, None, None]
        derivative = draw/size[:, None, None]-prediction*dlog[:, None, None]
        return prediction, derivative, mu, kap, valid

    def recover(self, A, observed=None):
        legacy = FrameAccommodation(self.host(self.x), self.host(self.b), self.host(self.gaze), self.host(self.units),
              self.host(self.k), self.host(self.m), self.host(self.e), self.host(self.expected), self.slope, self.reference_A)
        return legacy.recover(self.host(A), observed=observed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    import cupy as cp
    started = time.perf_counter()
    old = ROOT/'experiments/reverse_transform/stage_03_independent_captures/results/run'
    p = run.load_npz(old/'frames.npz')
    pop = run.load_npz(old/'population.npz')
    summary = json.loads((old/'summary.json').read_text())
    post = json.loads((old/'postfit.json').read_text())
    raw = run.load_npz(run.parent(4)/'frames.npz')
    for key in ('population_index', 'row', 'source_frame', 'gaze_xy_deg', 'reference_p4', 'observed_centered_p4'):
        assert np.array_equal(raw[key], p[key]), key
    expected_rows = pop['demand_diopters'][p['population_index']]
    exp = p['exposure']
    expected = np.array([expected_rows[exp == f][0] for f in range(20)])
    k = np.array([summary['captures'][c]['p4']['coefficients_scaled'][:4] for c in p['capture_index']])
    slope, aref = post['anchored_linear_slope_per_D_px2'], summary['reference_demand_diopters_label']
    model = NormalizedForwardControl(p['observed_centered_p4'], p['reference_p4'], p['gaze_xy_deg'], p['gaze_units_deg'],
                                     k, p['p1_magnification'], exp, expected, slope, aref, xp=cp)
    lo, hi = model.feasible_bounds()
    starts, solutions = [], []
    for offset in (0., -.5, .5):
        A, fit = fit_accommodation(model, lo, hi, expected_rows+offset)
        starts.append(fit)
        solutions.append(A)
        print(json.dumps(dict(offset_D=offset, **fit)), flush=True)
    best = int(np.argmin([s['objective_scaled_sum'] for s in starts]))
    A = solutions[best]
    pred = model.host(model.geometry(cp.asarray(A))[0])
    back, valid = model.recover(A)
    cpu = NormalizedForwardControl(p['observed_centered_p4'], p['reference_p4'], p['gaze_xy_deg'], p['gaze_units_deg'],
                                   k, p['p1_magnification'], exp, expected, slope, aref)
    cert, _ = cpu.certificate(A, lo, hi)
    assert cert['stationary'], cert
    # Independent normalized forward helper is historical, with CPU arithmetic.
    legacy = FrameAccommodation(p['observed_centered_p4'], p['reference_p4'], p['gaze_xy_deg'], p['gaze_units_deg'],
                                 k, p['p1_magnification'], exp, expected, slope, aref)
    replay = legacy.geometry(A)[4]
    assert np.max(abs(replay-pred)) < 1e-8
    h = 1e-4
    plus = legacy.geometry(A+h)[4]
    minus = legacy.geometry(A-h)[4]
    cp1 = np.sum((plus-p['observed_centered_p4'])**2, axis=(1, 2))/3
    cm1 = np.sum((minus-p['observed_centered_p4'])**2, axis=(1, 2))/3
    _, analytic, _ = cpu.frame_cost(A)
    fd_error = float(np.max(abs((cp1-cm1)/(2*h)-analytic)/(1+abs(analytic))))
    assert fd_error < 1e-5
    clean, clean_valid = legacy.recover(A, observed=replay)
    assert clean_valid.all() and np.max(abs(clean-p['reference_p4'])) < 1e-8
    np.savez_compressed(out/'frames.npz', A_D=A, exposure=exp, row=p['row'], population_index=p['population_index'],
        recovered_p4=back, inverse_valid=valid, prediction_p4=pred, forward_residual=p['observed_centered_p4']-pred,
        lower_A_D=lo, upper_A_D=hi)
    records = []
    for f in range(20):
        sel = exp == f
        common = sel & valid & raw['p4_inverse_valid']
        records.append(dict(exposure=f, nominal_gaze_deg=[-10,-5,0,5,10][f%5],
            A_D=statistics(A[sel]), raw_A_D=statistics(raw['A_D'][sel]),
            forward_point_distance_px=statistics(np.linalg.norm(p['observed_centered_p4'][sel]-pred[sel], axis=2)),
            raw_forward_point_distance_px=statistics(np.linalg.norm(raw['forward_residual_p4'][sel], axis=2)),
            common_inverse_frames=int(common.sum()),
            inverse_common_point_distance_px=statistics(np.linalg.norm(back[common]-p['reference_p4'], axis=2)),
            raw_inverse_common_point_distance_px=statistics(np.linalg.norm(raw['recovered_p4'][common]-raw['reference_p4'], axis=2))))
    sources = [Path(__file__).resolve(), Path(run.__file__).resolve(),
               ROOT/'distortion_model/raw_keystone.py', ROOT/'distortion_model/frame_accommodation.py',
               ROOT/'distortion_model/capture_shape.py', ROOT/'distortion_model/joint_state.py',
               Path(__file__).with_name('helpers.py')]
    for src in sources:
        dest = out/'source_snapshot'/src.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
    run.write(out/'summary.json', dict(status='COMPLETE', role='Historical normalized optics with matched forward objective. Comparison only; never a parent calibration for raw results.',
        selected_start=best, starts=starts, fixations=records, runtime_seconds=time.perf_counter()-started,
        forward_point_distance_px=statistics(np.linalg.norm(p['observed_centered_p4']-pred, axis=2)),
        inverse_failures=int((~valid).sum()), all_frames=len(A)))
    run.write(out/'audit.json', dict(status='PASS', CPU=True, certificate=cert,
        independent_normalized_forward_difference_px=float(np.max(abs(replay-pred))),
        all_frame_gradient_relative_error_max=fd_error, synthetic_closure_px=float(np.max(abs(clean-p['reference_p4'])))))
    run.write(out/'provenance.json', dict(source_sha256={str(s.relative_to(ROOT)): run.digest(s) for s in sources},
        historical_parent_sha256={str(f.relative_to(ROOT)): run.digest(f) for f in old.rglob('*') if f.is_file()},
        raw_comparison_sha256={str((run.parent(4)/'frames.npz').relative_to(ROOT)): run.digest(run.parent(4)/'frames.npz')}))
    print(json.dumps(dict(status='COMPLETE', audit='PASS', runtime_seconds=time.perf_counter()-started)), flush=True)


if __name__ == '__main__':
    main()
