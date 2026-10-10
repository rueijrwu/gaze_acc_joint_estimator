"""Independent CPU replay of Capture-1 state-ablation saved results."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT))
import run
from distortion_model.raw_keystone import RawKeystone, RawFrameAccommodation
from distortion_model.joint_state import projected_gradient


def oracle(p, states):
    """Raw P1 profiler + independent vertexwise radial/keystone P4 replay."""
    gaze = p['gaze_xy_deg'].copy()
    gaze[:, 0] = states[:, 0]
    p1 = RawKeystone(gaze, p['reference_p1'], p['observed_centered_p1'],
                      np.ones(len(states)), p['gaze_units_deg'])
    pred1, m = p1.prediction(p['k1'])
    b = p['reference_p4']
    r = np.sqrt(np.mean(np.sum(b*b, axis=1)))
    k = p['k4']
    t = gaze/p['gaze_units_deg']
    kap = p['slope']*(states[:, 1]-p['reference_A'])
    transformed = np.empty((len(states), 3, 2))
    pre = np.empty_like(transformed)
    exponent = k[0]*t[:, 0]**2-k[1]*t[:, 1]**2
    for j in range(3):
        pre[:, j] = b[j]*(1+kap*np.dot(b[j], b[j]))[:, None]
        den = 1+(k[2]*t[:, 0]*pre[:, j, 1]+k[3]*t[:, 1]*pre[:, j, 0])/r
        transformed[:, j, 0] = pre[:, j, 0]*np.exp(exponent)/den
        transformed[:, j, 1] = pre[:, j, 1]*np.exp(-exponent)/den
        if np.min(den) <= 1e-8 or np.min(1+3*kap*np.dot(b[j], b[j])) <= 1e-8:
            raise ValueError('Independent forward domain failure')
    c = transformed-transformed.mean(axis=1, keepdims=True)
    return [pred1, m[:, None, None]*c], m


def oracle_cost(p, states):
    pred, _ = oracle(p, states)
    return sum(np.sum((x-y)**2, axis=(1, 2))/3 for x, y in
               zip(pred, (p['observed_centered_p1'], p['observed_centered_p4'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    out = args.results.resolve()
    provenance = json.loads((out/'provenance.json').read_text())
    for group in ('parent_sha256', 'source_sha256'):
        if not all(run.digest(ROOT/key) == value for key, value in provenance[group].items()):
            raise ValueError(f'{group} hash mismatch')
    p = run.load_inputs()
    with np.load(out/'frames.npz', allow_pickle=False) as z:
        saved = dict(z)
    summary = json.loads((out/'summary.json').read_text())
    protocol = summary['protocol']
    assert protocol['gaze_mean_anchor_width_deg'] == .5
    assert protocol['A_mean_anchor_width_D'] == .25
    assert protocol['weights'] == dict(P1=1., P4=1.)
    for key, value in p.items():
        if key in saved and not np.array_equal(value, saved[key]):
            raise ValueError(f'Frozen input changed: {key}')
    assert np.array_equal(saved['A0_states'][:, 0], p['gaze_xy_deg'][:, 0])
    assert np.array_equal(saved['A0_states'][:, 1], p['baseline_A'])
    assert np.array_equal(saved['A0_magnification'], p['p1_magnification'])
    checks = {}
    for name in ('A0', 'G', 'GA'):
        states = saved[f'{name}_states']
        pred, m = oracle(p, states)
        prediction_difference = max(float(np.max(abs(pred[j]-saved[f'{name}_prediction_p{1 if j == 0 else 4}']))) for j in range(2))
        assert prediction_difference < 1e-8
        assert np.max(abs(m-saved[f'{name}_magnification'])) < 1e-10
        for j, observed in enumerate((p['observed_centered_p1'], p['observed_centered_p4'])):
            field = f'{name}_residual_p{1 if j == 0 else 4}'
            assert np.max(abs(observed-pred[j]-saved[field])) < 1e-8
        model = run.make_model(p, joint=name != 'G')
        vector = states[:, :model.d]
        _, jac, _, valid = model.forward(vector)
        assert valid.all()
        jac_error = 0.
        independent_jac = [np.empty((len(states), 3, 2, 2)) for _ in range(2)]
        for j in range(2):
            plus, minus = states.copy(), states.copy()
            plus[:, j] += 1e-4
            minus[:, j] -= 1e-4
            pp, _ = oracle(p, plus)
            pm, _ = oracle(p, minus)
            for v in range(2):
                fd = (pp[v]-pm[v])/2e-4
                independent_jac[v][..., j] = fd
                if j < model.d:
                    jac_error = max(jac_error, float(np.max(abs(fd-jac[v][..., j])/(1+abs(fd)))))
        assert jac_error < 1e-6
        # Synthetic closure tests trial-dependent P1 scale and raw P4 geometry.
        gaze = p['gaze_xy_deg'].copy()
        gaze[:, 0] = states[:, 0]
        c1 = RawKeystone(gaze, p['reference_p1'], pred[0], np.ones(len(m)), p['gaze_units_deg'], m)
        back1, v1 = c1.inverse(p['k1'], m, observed=pred[0])
        c4 = RawFrameAccommodation(pred[1], p['reference_p4'], gaze, p['gaze_units_deg'],
                               np.tile(p['k4'], (len(m), 1)), m, p['exposure'],
                               np.full(5, p['reference_A']), p['slope'], p['reference_A'])
        back4, v4 = c4.recover(states[:, 1])
        assert v1.all() and v4.all()
        closure = max(float(np.max(abs(back1-p['reference_p1']))), float(np.max(abs(back4-p['reference_p4']))))
        assert closure < 1e-8
        check = dict(independent_forward_difference_px=prediction_difference,
                     all_frame_Jacobian_relative_error_max=jac_error, synthetic_closure_px=closure)
        conditioning_replay = {}
        near = ((states[:, 0] < -19.5) | (states[:, 0] > 19.5) |
                (states[:, 1] < .05) | (states[:, 1] > 5.95))
        for label, j in (('p4', independent_jac[1]), ('joint', np.concatenate(independent_jac, axis=1))):
            values = run.conditioning(j)
            for field in ('singular_max', 'singular_min', 'condition', 'column_correlation', 'acute_angle_deg'):
                old = saved[f'{name}_{label}_{field}']
                relative = np.max(abs(values[field]-old)/(1+abs(old)))
                assert relative < 1e-4
            conditioning_replay[label] = dict(near_bound_frames=int(near.sum()),
                near_bound_condition=run.statistics(values['condition'][near]),
                near_bound_acute_angle_deg=run.statistics(values['acute_angle_deg'][near]),
                inverse_failure_condition=run.statistics(values['condition'][~saved[f'{name}_inverse_valid_p4']]))
        check['conditioning_near_bounds_and_inverse_failures'] = conditioning_replay
        # Recompute inverse arrays and common masks, including honest failures.
        r1, iv1, r4, iv4 = run.inverse(p, states, m)
        assert np.array_equal(iv1, saved[f'{name}_inverse_valid_p1'])
        assert np.array_equal(iv4, saved[f'{name}_inverse_valid_p4'])
        assert np.max(abs(r1[iv1]-saved[f'{name}_recovered_p1'][iv1])) < 1e-8
        assert np.max(abs(r4[iv4]-saved[f'{name}_recovered_p4'][iv4])) < 1e-8
        if name == 'G':
            assert np.all(states[:, 1] == p['reference_A'])
        if name != 'A0':
            lo = np.tile([-20., 0.][:model.d], (model.n, 1))
            hi = np.tile([20., 6.][:model.d], (model.n, 1))
            assert np.all(vector >= lo) and np.all(vector <= hi)
            cost, gradient = model.evaluate(vector)
            gradient = gradient.reshape(vector.shape)
            pg, _ = projected_gradient(vector, gradient, lo, hi)
            minimum = summary['fits'][name]['starts'][summary['fits'][name]['selected_start']]['certificate']['minimum_free_frame_curvature']
            assert np.max(abs(pg)) < 1e-5
            basecost = oracle_cost(p, states)
            fd_gradient = np.empty_like(vector)
            hessian = np.empty((model.n, model.d, model.d))
            step = 1e-3
            for j in range(model.d):
                plus, minus = states.copy(), states.copy()
                plus[:, j] += step
                minus[:, j] -= step
                cp, cm = oracle_cost(p, plus), oracle_cost(p, minus)
                fd_gradient[:, j] = (cp-cm)/(2*step)
                hessian[:, j, j] = (cp-2*basecost+cm)/step**2
            if model.d == 2:
                values = []
                for signs in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    values.append(oracle_cost(p, states+step*np.array(signs)))
                hessian[:, 0, 1] = hessian[:, 1, 0] = (values[0]-values[1]-values[2]+values[3])/(4*step**2)
            exposure = p['exposure']
            counts = np.bincount(exposure, minlength=5)
            means = np.array([vector[exposure == f].mean(axis=0) for f in range(5)])
            expected = np.column_stack(([-10, -5, 0, 5, 10], np.full(5, p['reference_A'])))[:, :model.d]
            strength = np.array([4., 16.])[:model.d]
            q = model.n/(5*counts[exposure])
            independent_gradient = q[:, None]*(fd_gradient+2*strength*(means-expected)[exposure])
            relative = float(np.max(abs(independent_gradient-gradient)/(1+abs(gradient))))
            assert relative < 1e-4
            independent_cost = np.dot(q, basecost)/model.n + np.mean(np.sum(strength*(means-expected)**2, axis=1))
            assert abs(independent_cost-cost/model.n) < 1e-9
            analytic_hessian = model.curvature(vector)
            hessian_error = float(np.max(abs(hessian-analytic_hessian)/(1+abs(analytic_hessian))))
            assert hessian_error < 1e-3
            free = ~((vector <= lo+1e-7) | (vector >= hi-1e-7))
            masked = hessian*free[:, :, None]*free[:, None, :]
            masked += np.eye(model.d)[None]*(~free)[:, :, None]*1e30
            eig = np.linalg.eigvalsh(masked)
            relevant = eig[eig < 1e29]
            assert relevant.min() > 0 and minimum > 0
            check.update(projected_gradient_inf=float(np.max(abs(pg))),
                         independent_gradient_relative_error_max=relative,
                         independent_Hessian_relative_error_max=hessian_error,
                         independent_minimum_free_frame_curvature=float(relevant.min()),
                         objective_equal_fixation_px2=independent_cost)
        checks[name] = check
    for pnum in (1, 4):
        common = np.logical_and.reduce([saved[f'{name}_inverse_valid_p{pnum}'] for name in ('A0', 'G', 'GA')])
        assert np.array_equal(common, saved[f'common_inverse_valid_p{pnum}'])
    run.write(out/'audit.json', dict(status='PASS', independent_CPU=True, checks=checks,
                                   scope='All frames: immutable input/baseline replay, independent forward oracle, Jacobian/gradient/Hessian finite differences, projected stationarity, sufficient free curvature, synthetic inverse closure. No physiological or global-optimum certification.'))
    print(json.dumps(dict(status='PASS', checks=checks), indent=2), flush=True)


if __name__ == '__main__':
    main()
