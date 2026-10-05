#!/usr/bin/env python3
"""Correctness checks for the m2 model and for the unchanged piecewise path.

  (a) finite-difference check of the m2 forward model Jacobian wrt theta, A and all 13
      coefficients (plus the design-tensor derivatives and the ProfiledProblem Jacobian);
  (b) for each --model-json: monotonicity of d in theta over [-20,20] for A in [0,6],
      reporting the minimum dd/dtheta (deg^-1 in d units);
  (c) piecewise forward predictions / penalties / profile residuals equal the git ref (default HEAD).

Usage: python3 -B check_m2.py [--model-json PATH ...] [--ref HEAD] [--json-out PATH] [--skip-b] [--skip-c]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
sys.path.insert(0, str(EXP))
import m2_model as m2
import calibrate_profiled as cp

TOL = 1e-6


def rel(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-12))


def check_a(rng):
    results = {}
    coef = rng.normal(size=13)*np.array([.1, .02, .05, .02, .01, .01, .005, .7, .01, .005, .02, .01, .005])
    theta, A = rng.uniform(-20, 20, 40), rng.uniform(0, 6, 40)
    h = 1e-6
    pred, jac = m2.forward_jac_m2(theta, A, coef)
    fd_t = (m2.forward_jac_m2(theta+h, A, coef)[0]-m2.forward_jac_m2(theta-h, A, coef)[0])/(2*h)
    fd_A = (m2.forward_jac_m2(theta, A+h, coef)[0]-m2.forward_jac_m2(theta, A-h, coef)[0])/(2*h)
    results['forward_jac_dtheta'] = rel(jac[..., 0], fd_t)
    results['forward_jac_dA'] = rel(jac[..., 1], fd_A)
    H, dt, dA = m2.basis_m2(theta, A)
    results['basis_H_pred_vs_forward'] = rel(H@coef, pred)
    results['basis_dt_vs_forward_jac'] = rel(dt@coef, jac[..., 0])
    results['basis_dA_vs_forward_jac'] = rel(dA@coef, jac[..., 1])
    fd_coef = np.empty_like(H)
    for k in range(13):
        e = np.zeros(13); e[k] = h
        fd_coef[:, :, k] = (m2.forward_jac_m2(theta, A, coef+e)[0]-m2.forward_jac_m2(theta, A, coef-e)[0])/(2*h)
    results['jacobian_wrt_coefficients(all 13)'] = rel(H, fd_coef)
    results['per_coefficient_max'] = {name: rel(H[:, :, k], fd_coef[:, :, k]) for k, name in enumerate(m2.COEFFICIENT_NAMES)}
    fd_Ht = (m2.basis_m2(theta+h, A, False)[0]-m2.basis_m2(theta-h, A, False)[0])/(2*h)
    fd_HA = (m2.basis_m2(theta, A+h, False)[0]-m2.basis_m2(theta, A-h, False)[0])/(2*h)
    results['basis_dt_tensor'] = rel(dt, fd_Ht)
    results['basis_dA_tensor'] = rel(dA, fd_HA)
    results['_ok'] = all(v < TOL for k, v in results.items() if not k.startswith('_') and not isinstance(v, dict)) and \
        all(v < TOL for v in results['per_coefficient_max'].values())
    return results


def check_problem_jacobian(rng):
    """ProfiledProblem(model='m2') variable-projection Jacobian vs finite differences of the residual."""
    J, n_per = 4, 30
    groups = np.repeat(np.arange(J), n_per)
    targets, demands = np.array([-10., 0., 10., 5.]), np.array([.36, 2., 4., 3.])
    truth = np.array([.12, .02, .5, .1, .05, .02, .01, .6, .03, .01, .04, .02, .01])
    th0, A0 = targets[groups]+rng.normal(0, .5, len(groups)), demands[groups]+rng.normal(0, .1, len(groups))
    A0 = np.clip(A0, 0, 6)
    y = m2.basis_m2(th0, A0, False)[0]@truth+rng.normal(0, 1e-3, (len(groups), 2))
    frames = np.arange(len(groups))
    W = np.diag([1e4, 1e4])
    problem = cp.ProfiledProblem(y, frames, groups, targets, demands, None, truth, W, np.diag(1/np.ptp(y, axis=0)**2),
                                 model='m2', theta_anchor_scale_deg=.5)
    x = problem.encode(th0+.1, A0+.05)
    omega = np.ones(problem.n)
    op = problem.jacobian(x, omega)
    v = rng.normal(size=x.shape)
    eps = 1e-6
    fd = (problem.residual(x+eps*v, omega)-problem.residual(x-eps*v, omega))/(2*eps)
    err_mv = rel(op.matvec(v), fd)
    w = rng.normal(size=op.shape[0])
    adjoint = abs(w@op.matvec(v)-op.rmatvec(w)@v)/max(abs(w@op.matvec(v)), 1e-12)
    return dict(problem_jacobian_matvec_vs_fd=err_mv, problem_jacobian_adjoint_mismatch=float(adjoint),
                _ok=bool(err_mv < 1e-5 and adjoint < 1e-9))


def min_slope(model_json, theta, A):
    """dd/dtheta over a (theta, A) grid for an m2 or a piecewise model.json (d units per degree)."""
    coef = np.asarray(model_json['coefficients'])
    if model_json.get('model_type', 'piecewise') == 'm2':
        return m2.forward_jac_m2(theta, A, coef)[1][..., 0, 0]
    w = cp.interpolation(A.ravel(), cp.KNOTS)[0]
    return (w@coef[4:8]).reshape(A.shape)+0*theta  # piecewise d = b(A)+s(A)*theta: slope is s(A)


def check_b(paths):
    out = {}
    theta, A = np.meshgrid(np.linspace(-20, 20, 801), np.linspace(0, 6, 241), indexing='ij')
    for path in paths:
        model = json.loads(Path(path).read_text())
        slope = min_slope(model, theta, A)
        i, j = np.unravel_index(np.argmin(slope), slope.shape)
        out[str(path)] = dict(model_type=model.get('model_type', 'piecewise'), min_dd_dtheta=float(slope.min()),
                              at_theta_deg=float(theta[i, j]), at_A_D=float(A[i, j]),
                              max_dd_dtheta=float(slope.max()), strictly_monotonic=bool(slope.min() > 0),
                              grid='theta -20..20 (801) x A 0..6 (241)', _ok=bool(slope.min() > 0))
    return out


def load_ref_module(ref):
    src = subprocess.run(['git', 'show', f'{ref}:exp2/calibrate_profiled.py'], cwd=EXP, check=True,
                         capture_output=True, text=True).stdout
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'calibrate_profiled_ref.py'
        path.write_text(src)
        spec = importlib.util.spec_from_file_location('calibrate_profiled_ref', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def check_c(rng, ref):
    old = load_ref_module(ref)
    out = {}
    p = 0.6088611641569155
    theta, A = rng.uniform(-20, 20, 50), rng.uniform(0, 6, 50)
    coef = rng.normal(size=14)*.1
    Ho, dto, dao = old.basis(theta, A**p, p)
    Hn, dtn, dan = cp.basis(theta, A**p, p)
    out['basis_H_max_abs_diff'] = float(np.max(abs(Ho-Hn)))
    out['basis_dtheta_max_abs_diff'] = float(np.max(abs(dto-dtn)))
    out['basis_da_max_abs_diff'] = float(np.max(abs(dao-dan)))
    out['forward_prediction_max_abs_diff'] = float(np.max(abs(Ho@coef-Hn@coef)))
    # Whole-problem comparison on a tiny synthetic piecewise problem (default anchor scale).
    J, n_per = 4, 25
    groups = np.repeat(np.arange(J), n_per)
    targets, demands = np.array([-8., 0., 8., 4.]), old.DEMANDS[[0, 3, 1, 2]]
    y = rng.normal(0, .02, (len(groups), 2))+np.array([.1, .7])
    frames = np.arange(len(groups))
    init = np.zeros(14); init[:4] = [.1, .1, .1, .1]; init[4:8] = .04; init[8] = .7
    args = (y, frames, groups, targets, demands, p, init, np.diag([1e4, 1e4]), np.diag(1/np.ptp(y, axis=0)**2))
    po, pn = old.ProfiledProblem(*args), cp.ProfiledProblem(*args)
    x = po.encode(targets[groups]+rng.normal(0, .3, len(groups)), np.clip(demands[groups]+rng.normal(0, .1, len(groups)), 0, 6))
    omega = np.ones(po.n)
    th_s, A_s = x.reshape(-1, 2)[:, 0]*15, po.decode(x)[2]
    out['encode_decode_bitwise_equal'] = bool(np.array_equal(po.encode(th_s, A_s), pn.encode(th_s, A_s))
                                              and all(np.array_equal(a, b) for a, b in zip(po.decode(x), pn.decode(x))))
    out['penalties_bitwise_equal'] = bool(np.array_equal(po.penalties(x), pn.penalties(x)))
    out['residual_bitwise_equal'] = bool(np.array_equal(po.residual(x, omega), pn.residual(x, omega)))
    v = rng.normal(size=x.shape)
    out['jacobian_matvec_bitwise_equal'] = bool(np.array_equal(po.jacobian(x, omega).matvec(v), pn.jacobian(x, omega).matvec(v)))
    w = rng.normal(size=po.jacobian(x, omega).shape[0])
    out['jacobian_rmatvec_bitwise_equal'] = bool(np.array_equal(po.jacobian(x, omega).rmatvec(w), pn.jacobian(x, omega).rmatvec(w)))
    out['bounds_equal'] = bool(all(np.array_equal(a, b) for a, b in zip(po.bounds(), pn.bounds())))
    out['_ok'] = bool(out['basis_H_max_abs_diff'] == 0 and out['basis_dtheta_max_abs_diff'] == 0
                      and out['basis_da_max_abs_diff'] == 0 and out['forward_prediction_max_abs_diff'] == 0
                      and all(v is True for k, v in out.items() if isinstance(v, bool)))
    out['ref'] = ref
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-json', type=Path, nargs='*', default=[])
    parser.add_argument('--ref', default='HEAD')
    parser.add_argument('--json-out', type=Path)
    parser.add_argument('--skip-b', action='store_true')
    parser.add_argument('--skip-c', action='store_true')
    args = parser.parse_args()
    rng = np.random.default_rng(20261005)
    report = {'a_m2_finite_differences': check_a(rng), 'a2_problem_jacobian': check_problem_jacobian(rng)}
    if not args.skip_b and args.model_json:
        report['b_monotonicity'] = check_b(args.model_json)
    if not args.skip_c:
        report['c_piecewise_unchanged_vs_'+args.ref] = check_c(rng, args.ref)
    def ok(node):
        return all(node['_ok'] for node in ([node] if '_ok' in node else node.values()))
    print(json.dumps(report, indent=2))
    status = {k: ok(v) for k, v in report.items()}
    print('SUMMARY', json.dumps(status))
    if args.json_out:
        args.json_out.write_text(json.dumps(report, indent=2)+'\n')
    sys.exit(0 if all(status.values()) else 1)


if __name__ == '__main__':
    main()
