"""Independent CPU audit of the raw-keystone chain, all fitted frames."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
sys.path.insert(0, str(ROOT))
from distortion_model.raw_keystone import RawKeystone, RawFrameAccommodation
import run


def oracle(gaze, units, b, k, kappa=0.):
    """Direct vertexwise projective equation, without RMS rescaling."""
    t = gaze/units
    r = np.sqrt(np.mean(np.sum(b*b, axis=1)))
    k = np.broadcast_to(k, (len(gaze), 4))
    kap = np.broadcast_to(kappa, len(gaze))
    a = k[:, 0]*t[:, 0]**2-k[:, 1]*t[:, 1]**2
    transformed = np.empty((len(gaze), 3, 2))
    for j in range(3):
        pre = b[j][None]*(1+kap*np.dot(b[j], b[j]))[:, None]
        den = 1+(k[:, 2]*t[:, 0]*pre[:, 1]+k[:, 3]*t[:, 1]*pre[:, 0])/r
        transformed[:, j, 0] = pre[:, 0]*np.exp(a)/den
        transformed[:, j, 1] = pre[:, 1]*np.exp(-a)/den
        if np.min(den) <= 1e-8 or np.min(1+3*kap*np.dot(b[j], b[j])) <= 1e-8:
            raise ValueError('Independent optical domain failure')
    return transformed-transformed.mean(axis=1, keepdims=True), transformed.mean(axis=1)


def profile(x, h):
    return np.sum(x*h, axis=(1, 2))/np.sum(h*h, axis=(1, 2))


def verify_global(p, sel, record, pattern):
    gaze, units = p['gaze_xy_deg'][sel], p['gaze_units_deg'][sel]
    b, x = p[f'reference_p{pattern}'], p[f'observed_centered_p{pattern}'][sel]
    exp = p['exposure'][sel] % 5
    weights = 1/(15*np.bincount(exp, minlength=5)[exp])
    k = np.array(record['coefficients_scaled'])
    fixed = None if pattern == 1 else p['p1_magnification'][sel]
    model = RawKeystone(gaze, b, x, weights, units, fixed)
    def cost(params):
        kap = (params[4] if len(params) == 5 else 0.)/run.radius(b)**2
        h, _ = oracle(gaze, units, b, params[:4], kap)
        m = profile(x, h) if fixed is None else fixed
        return float(np.sum(weights*np.sum((x-m[:, None, None]*h)**2, axis=(1, 2))))
    value, grad = model.evaluate(k)
    numeric = []
    for j in range(len(k)):
        d = np.eye(len(k))[j]*1e-6
        numeric.append((cost(k+d)-cost(k-d))/2e-6)
    relative = float(np.max(abs(grad-numeric)/(1+abs(grad))))
    assert relative < 1e-4, relative
    bounds = np.array(record.get('bounds_scaled'))
    pg = grad.copy()
    pg[((k <= bounds[:, 0]+1e-7) & (grad > 0)) | ((k >= bounds[:, 1]-1e-7) & (grad < 0))] = 0
    assert np.max(abs(pg)) < 1e-6, pg
    hessian = np.column_stack([(model.evaluate(k+np.eye(len(k))[j]*1e-5)[1]-
                               model.evaluate(k-np.eye(len(k))[j]*1e-5)[1])/2e-5 for j in range(len(k))])
    active = (abs(k-bounds[:, 0]) < 1e-7) | (abs(k-bounds[:, 1]) < 1e-7)
    free = np.flatnonzero(~active)
    eigen = np.linalg.eigvalsh(((hessian+hessian.T)/2)[np.ix_(free, free)])
    assert len(eigen) and eigen.min() > 0
    assert model.domain(k)['valid']
    return dict(independent_cost_difference=abs(cost(k)-value), analytic_gradient_relative_error=relative,
                projected_gradient_inf=float(np.max(abs(pg))), active_bounds=active.tolist(),
                free_hessian_eigenvalues=eigen.tolist(), domain=model.domain(k))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=range(1, 5), required=True)
    parser.add_argument('--results', type=Path, required=True)
    args = parser.parse_args()
    out = args.results.resolve()
    provenance = json.loads((out/'provenance.json').read_text())
    for group in ('source_sha256', 'parent_sha256', 'historical_control_sha256', 'input_sha256'):
        for key, value in provenance.get(group, {}).items():
            assert run.digest(ROOT/key) == value, key
            if group == 'source_sha256':
                assert run.digest(out/'source_snapshot'/key) == value, key
    p = run.load_npz(out/'frames.npz')
    pop = run.load_npz(out/'population.npz')
    summary = json.loads((out/'summary.json').read_text())
    assert summary['stage'] == args.stage
    assert summary['complete']+summary['unavailable'] == summary['scheduled'] == len(pop['row'])
    idx = p['population_index']
    assert np.array_equal(idx, np.flatnonzero(pop['complete_valid']))
    for key in ('row', 'source_frame', 'exposure'):
        assert np.array_equal(p[key], pop[key][idx])
    for n in (1, 4):
        observed = pop[f'p{n}'][idx]
        observed -= observed.mean(axis=1, keepdims=True)
        assert np.array_equal(observed, p[f'observed_centered_p{n}'])
        zero = (p['capture_index'] == 0) & (p['exposure'] % 5 == 2)
        assert np.max(abs(p[f'reference_p{n}']-observed[zero].mean(axis=0))) < 1e-10
    checks = dict(global_fits=[], inverse=[], all_complete_frames=len(idx))
    for rec in summary['captures']:
        sel = p['capture_index'] == rec['capture']-1
        exp = p['exposure'][sel] % 5
        delta = pop['p4'][idx[sel]].mean(axis=1)-pop['p1'][idx[sel]].mean(axis=1)
        gaze, _ = run.calibrate_gaze(delta, exp, np.array([-10., -5., 0., 5., 10.]))
        assert np.array_equal(gaze, p['gaze_xy_deg'][sel])
        assert np.max(abs(gaze[exp == 2].mean(axis=0))) < 1e-10
        for n in (1, 4):
            if f'prediction_p{n}' not in p:
                continue
            if args.stage < 4 or n == 1:
                checks['global_fits'].append(dict(capture=rec['capture'], pattern=n,
                    **verify_global(p, sel, rec[f'p{n}'], n)))
    m = p['p1_magnification']
    assert np.all(np.isfinite(m) & (m > 0))
    h1, _ = oracle(p['gaze_xy_deg'], p['gaze_units_deg'], p['reference_p1'], p['p1_keystone_scaled'])
    assert np.max(abs(profile(p['observed_centered_p1'], h1)-m)) < 1e-10
    # P1-only normalization is an exact profiled-scale reparameterization.
    ratio = run.radius(h1)/run.radius(p['reference_p1'])
    normalized = h1/ratio[:, None, None]
    normalized_m = profile(p['observed_centered_p1'], normalized)
    cancellation = float(np.max(abs(normalized_m[:, None, None]*normalized-m[:, None, None]*h1)))
    assert cancellation < 1e-9
    checks['P1_profile_normalization_cancellation_px'] = cancellation
    for n in (1, 4):
        if f'prediction_p{n}' not in p:
            continue
        kap = 0. if n == 1 else p['kappa_per_px2']
        h, _ = oracle(p['gaze_xy_deg'], p['gaze_units_deg'], p[f'reference_p{n}'], p[f'p{n}_keystone_scaled'], kap)
        predicted = m[:, None, None]*h
        forward_difference = float(np.max(abs(predicted-p[f'prediction_p{n}'])))
        assert forward_difference < 1e-8
        assert np.max(abs(p[f'forward_residual_p{n}']-(p[f'observed_centered_p{n}']-predicted))) < 1e-8
        assert np.max(abs(run.radius(h)/run.radius(p[f'reference_p{n}'])-p[f'p{n}_raw_size_factor'])) < 1e-10
        closure = 0.
        inverse_difference = 0.
        for c in np.unique(p['capture_index']):
            sel = p['capture_index'] == c
            if n == 4 and args.stage == 4:
                protocol = summary['protocol']
                exp = p['exposure']
                expected = np.array([p['expected_A_D'][exp == f][0] for f in range(20)])
                law = protocol['kappa_law']
                model = RawFrameAccommodation(p['observed_centered_p4'], p['reference_p4'], p['gaze_xy_deg'], p['gaze_units_deg'],
                    p['p4_keystone_scaled'], m, exp, expected, law['slope'], law['reference_A'])
                back, valid = model.recover(p['A_D'])
                clean, clean_valid = model.recover(p['A_D'], observed=predicted)
                break
            k = np.array(summary['captures'][c][f'p{n}']['coefficients_scaled'])
            model = RawKeystone(p['gaze_xy_deg'][sel], p[f'reference_p{n}'], p[f'observed_centered_p{n}'][sel],
                                np.ones(sel.sum()), p['gaze_units_deg'][sel], m[sel])
            back, valid = model.inverse(k, m[sel])
            clean, clean_valid = model.inverse(k, m[sel], observed=predicted[sel])
            assert np.array_equal(valid, p[f'p{n}_inverse_valid'][sel])
            inverse_difference = max(inverse_difference, float(np.max(abs(back[valid]-p[f'recovered_p{n}'][sel][valid]))))
            assert clean_valid.all()
            closure = max(closure, float(np.max(abs(clean-p[f'reference_p{n}']))))
        if n == 4 and args.stage == 4:
            assert np.array_equal(valid, p['p4_inverse_valid'])
            inverse_difference = float(np.max(abs(back[valid]-p['recovered_p4'][valid])))
            assert clean_valid.all()
            closure = float(np.max(abs(clean-p['reference_p4'])))
        assert closure < 1e-8 and inverse_difference < 1e-8
        checks['inverse'].append(dict(pattern=n, independent_forward_difference_px=forward_difference,
                                      measured_inverse_replay_difference_px=inverse_difference, synthetic_closure_px=closure))
    # A deliberately nontrivial K must retain its raw size change.
    probe_gaze = np.tile([10., 0.], (3, 1))*np.array([[-1., 1.], [0., 1.], [1., 1.]])
    probe = RawKeystone(probe_gaze, p['reference_p1'], np.zeros((3, 3, 2)), np.ones(3), np.array([10., 1.]))
    probe_h = probe.shape(np.array([.15, .07, .09, -.04]))[0]
    expected_probe, _ = oracle(probe_gaze, np.array([10., 1.]), p['reference_p1'], [.15, .07, .09, -.04])
    assert np.max(abs(probe_h-expected_probe)) < 1e-10
    size_span = float(np.ptp(run.radius(probe_h)))
    assert size_span > .01
    checks['nontrivial_raw_size_span_px'] = size_span
    if args.stage == 4:
        A, lo, hi = p['A_D'], p['lower_A_D'], p['upper_A_D']
        assert np.all(A >= lo) and np.all(A <= hi)
        cert, curvature = model.certificate(A, lo, hi)
        assert cert['stationary'], cert
        _, gradient = model.evaluate(A)
        def independent_frame_cost(a):
            hh, _ = oracle(p['gaze_xy_deg'], p['gaze_units_deg'], p['reference_p4'], p['p4_keystone_scaled'], law['slope']*(a-law['reference_A']))
            return np.sum((p['observed_centered_p4']-m[:, None, None]*hh)**2, axis=(1, 2))/3
        h = 1e-4
        fd = (independent_frame_cost(A+h)-independent_frame_cost(A-h))/(2*h)
        counts = np.bincount(p['exposure'], minlength=20)
        means = np.array([A[p['exposure'] == f].mean() for f in range(20)])
        independent_grad = model.host(model.q)*(fd+32*(means-expected)[p['exposure']])
        relative = float(np.max(abs(independent_grad-gradient)/(1+abs(gradient))))
        assert relative < 1e-5
        checks['frame_A'] = dict(certificate=cert, all_frame_gradient_relative_error=relative,
                                fixation_means_D=means.tolist(), forward_domain_limited_frames=int(np.sum(hi < 6-1e-7)))
    run.write(out/'audit.json', dict(status='PASS', independent_CPU=True, checks=checks,
              scope='Fresh population/calibration/reference and hashes; independent raw forward equation, global gradients/free curvature, positive common scale, exact inverse closure, retained raw size response, framewise forward gradients/stationarity. No physical/physiological certification.'))
    print(json.dumps(dict(status='PASS', checks=checks), indent=2), flush=True)


if __name__ == '__main__':
    main()
