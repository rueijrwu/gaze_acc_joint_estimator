"""The "m2" forward model and its anchor-free inverse (13 coefficients, no knots, no exponent).

With t = theta/15 and L = log(1+A):

    d    = (b0 + b1*A) + (s0 + s1*L)*t + (c20 + c21*L)*t**2 + c3*t**3
    rho4 = (r0 + r1*t + r2*t**2) + L*(r3 + r4*t + r5*t**2)

Coefficient order: [b0, b1, s0, s1, c20, c21, c3, r0, r1, r2, r3, r4, r5].

State parameterization: the state is (theta, A) itself, encoded as (theta/15, A/4)
with the box [-20,20] x [0,6].  dL/dA = 1/(1+A) is bounded on A >= 0, so (unlike an
A**p transform) the map has no singular endpoint and all A-denominated penalties
(anchor scale 0.25 D, smoothing 0.25 D) stay linear in the state.

This module is standalone (no import of calibrate_profiled) so both can import it.
"""
import numpy as np

SCHEMA = 'profiled_m2_displacement_log_ratio_v1'
MODEL_TYPE = 'm2'
COEFFICIENT_ORDER = 'b0,b1,s0,s1,c20,c21,c3,r0,r1,r2,r3,r4,r5 (t=theta/15, L=log(1+A))'
COEFFICIENT_NAMES = ['b0', 'b1', 's0', 's1', 'c20', 'c21', 'c3', 'r0', 'r1', 'r2', 'r3', 'r4', 'r5']
BASIS_DESCRIPTION = ('d=(b0+b1*A)+(s0+s1*L)*t+(c20+c21*L)*t^2+c3*t^3; '
                     'rho=(r0+r1*t+r2*t^2)+L*(r3+r4*t+r5*t^2); t=theta/15, L=log(1+A); '
                     'state=(theta,A) encoded (theta/15,A/4); no knots, no exponent')
N_COEF = 13
A_SCALE = 4.0
CAL_A_RANGE = (1000/2775, 4.0)  # protocol demand support, for the extrapolation flag/audits only
THETA_BOUNDS, A_BOUNDS = (-20., 20.), (0., 6.)
# Multistart grid for the inverse: segment endpoints/midpoints of the protocol edges
# (0, .36, 2, 3, 4, 6) -- the same coverage the piecewise inverse uses per segment.
_EDGES = np.array([0., CAL_A_RANGE[0], 2., 3., 4., 6.])
A_STARTS = np.unique(np.r_[_EDGES, (_EDGES[:-1]+_EDGES[1:])/2])
THETA_STARTS = np.array([-20., -10., 0., 10., 20.])


def basis_m2(theta, A, derivatives=True):
    """Design tensor H (n,2,13) with pred = H@coef; dt, dA = dH/dtheta_deg, dH/dA."""
    theta, A = np.asarray(theta, float), np.asarray(A, float)
    n = len(theta)
    t, L = theta/15, np.log1p(A)
    H = np.zeros((n, 2, N_COEF))
    H[:, 0, :7] = np.column_stack([np.ones(n), A, t, L*t, t*t, L*t*t, t**3])
    H[:, 1, 7:] = np.column_stack([np.ones(n), t, t*t, L, L*t, L*t*t])
    if not derivatives:
        return H, None, None
    z, o, iL = np.zeros(n), np.ones(n), 1/(1+A)
    dt, dA = np.zeros_like(H), np.zeros_like(H)
    dt[:, 0, :7] = np.column_stack([z, z, o, L, 2*t, 2*L*t, 3*t*t])/15
    dt[:, 1, 7:] = np.column_stack([z, o, 2*t, z, L, 2*L*t])/15
    dA[:, 0, :7] = np.column_stack([z, o, z, t*iL, z, t*t*iL, z])
    dA[:, 1, 7:] = np.column_stack([z, z, z, iL, t*iL, t*t*iL])
    return H, dt, dA


def forward_jac_m2(theta, A, coef):
    """Analytic prediction (...,2) and Jacobian (...,2,2) wrt (theta_deg, A); any array shape."""
    theta, A = np.broadcast_arrays(np.asarray(theta, float), np.asarray(A, float))
    c = np.asarray(coef, float)
    t, L, iL = theta/15, np.log1p(A), 1/(1+A)
    s, q = c[2]+c[3]*L, c[4]+c[5]*L
    r, g = c[7]+c[8]*t+c[9]*t*t, c[10]+c[11]*t+c[12]*t*t
    pred = np.stack([c[0]+c[1]*A+s*t+q*t*t+c[6]*t**3, r+L*g], axis=-1)
    jac = np.empty(theta.shape+(2, 2))
    jac[..., 0, 0] = (s+2*q*t+3*c[6]*t*t)/15
    jac[..., 0, 1] = c[1]+(c[3]*t+c[5]*t*t)*iL
    jac[..., 1, 0] = (c[8]+2*c[9]*t+L*(c[11]+2*c[12]*t))/15
    jac[..., 1, 1] = g*iL
    return pred, jac


def initial_coefficients(y, groups, targets, demands, selected_count):
    """Linear LS of the training-frame observations at nominal (theta, A), equal weight per fixation.

    `groups` are 0..J-1 training indices; `targets`/`demands` per group.  The nominal state is
    constant within a fixation, so this equals the LS fit to per-fixation mean observations.
    d and rho4 use disjoint coefficients, hence two independent linear fits.
    """
    counts = np.bincount(groups, minlength=selected_count)
    w = np.sqrt(1/(selected_count*counts[groups]))
    H = basis_m2(np.asarray(targets, float)[groups], np.asarray(demands, float)[groups], derivatives=False)[0]
    coef = np.zeros(N_COEF)
    for row, cols in [(0, slice(0, 7)), (1, slice(7, 13))]:
        X = H[:, row, cols]*w[:, None]
        coef[cols], _, rank, sv = np.linalg.lstsq(X, y[:, row]*w, rcond=None)
        if rank < X.shape[1]:
            raise ValueError('m2 initial fit is rank deficient for observable %d' % row)
    return coef


def initial_theta(coef, d_obs, A, iterations=20):
    """Per-frame initial gaze: invert the fitted d(theta; A) with clipped Newton steps."""
    L = np.log1p(A)
    b, s, q, c3 = coef[0]+coef[1]*A, coef[2]+coef[3]*L, coef[4]+coef[5]*L, coef[6]
    t = np.clip((d_obs-b)/np.where(abs(s) > 1e-8, s, 1e-8), -20/15, 20/15)
    for _ in range(iterations):
        f = b+s*t+q*t*t+c3*t**3-d_obs
        df = s+2*q*t+3*c3*t*t
        step = np.where(df > 1e-8, f/np.where(df > 1e-8, df, 1.), 0.)
        t = np.clip(t-step, -20/15, 20/15)
    return 15*t


def invert_batch_m2(y, coef, W, maxiter=60, tolerance=1e-7):
    """Bounded batched multistart Levenberg-damped GN over the full box; no anchors/temporal terms."""
    L = np.linalg.cholesky(W).T
    starts = np.array([(th, a) for a in A_STARTS for th in THETA_STARTS])
    z = np.broadcast_to(starts[None], (len(y), len(starts), 2)).copy()
    lower, upper = np.array([THETA_BOUNDS[0], A_BOUNDS[0]]), np.array([THETA_BOUNDS[1], A_BOUNDS[1]])
    damping = np.full(z.shape[:2], 1e-4)
    for _ in range(maxiter):
        pred, J = forward_jac_m2(z[..., 0], z[..., 1], coef)
        error = (pred-y[:, None]) @ L.T
        JW = np.einsum('ab,nkbc->nkac', L, J)
        g = np.einsum('nkac,nka->nkc', JW, error)
        h00 = np.sum(JW[..., 0]**2, axis=-1)
        h11 = np.sum(JW[..., 1]**2, axis=-1)
        h01 = np.sum(JW[..., 0]*JW[..., 1], axis=-1)
        d0, d1 = damping*np.maximum(h00, 1.), damping*np.maximum(h11, 1.)
        determinant = np.maximum((h00+d0)*(h11+d1)-h01*h01, 1e-30)
        step = np.stack([((h11+d1)*g[..., 0]-h01*g[..., 1])/determinant,
                         ((h00+d0)*g[..., 1]-h01*g[..., 0])/determinant], axis=-1)
        trial = np.clip(z-step, lower, upper)
        before = np.sum(error**2, axis=-1)
        predicted = forward_jac_m2(trial[..., 0], trial[..., 1], coef)[0]
        after = np.sum(((predicted-y[:, None]) @ L.T)**2, axis=-1)
        accepted = after <= before
        z[accepted] = trial[accepted]
        damping = np.where(accepted, np.maximum(damping/3, 1e-12), np.minimum(damping*10, 1e12))
    pred, J = forward_jac_m2(z[..., 0], z[..., 1], coef)
    error = (pred-y[:, None]) @ L.T
    JW = np.einsum('ab,nkbc->nkac', L, J)
    g = np.einsum('nkac,nka->nkc', JW, error)
    stationarity = np.max(np.abs(z-np.clip(z-g, lower, upper)), axis=-1)
    candidates = z.copy()
    costs = np.sum(error**2, axis=-1)
    best = costs.min(axis=1)
    near = costs <= best[:, None]+tolerance
    n = len(y)
    chosen, count = np.empty(n, dtype=int), np.empty(n, dtype=int)
    distance = np.full(n, np.nan)
    second_theta_delta, second_A_delta = np.full(n, np.nan), np.full(n, np.nan)
    theta_range, A_range = np.empty(n), np.empty(n)
    for i in range(n):
        indices = np.flatnonzero(near[i])
        indices = indices[np.lexsort((candidates[i, indices, 0], candidates[i, indices, 1]))]
        unique = []
        for j in indices:
            if not any(np.linalg.norm((candidates[i, j]-candidates[i, k])/np.array([1., .25])) < 1e-4 for k in unique):
                unique.append(j)
        chosen[i], count[i] = unique[0], len(unique)
        theta_range[i] = np.ptp(candidates[i, unique, 0])
        A_range[i] = np.ptp(candidates[i, unique, 1])
        if len(unique) > 1:
            delta = candidates[i, unique[1]]-candidates[i, unique[0]]
            second_theta_delta[i], second_A_delta[i] = delta
            distance[i] = np.linalg.norm(delta/np.array([1., .25]))
    row = np.arange(n)
    return candidates[row, chosen], dict(weighted_cost=costs[row, chosen], minimum_candidate_cost=best,
        equivalent_minima_count=count, second_branch_distance=distance,
        second_branch_theta_delta_deg=second_theta_delta, second_branch_A_delta_D=second_A_delta,
        equivalent_theta_range_deg=theta_range, equivalent_A_range_D=A_range,
        segment_projected_stationarity=stationarity[row, chosen],
        candidate_cost_max=costs.max(axis=1))


def physical_optimality_m2(state, observation, coef, W):
    """Full-bound physical KKT residual, scaled by the reference units (1 deg, .25 D)."""
    theta, A = state.T
    pred, J = forward_jac_m2(theta, A, coef)
    force = (pred-observation) @ W
    gt = force[:, 0]*J[:, 0, 0]+force[:, 1]*J[:, 1, 0]
    ga = force[:, 0]*J[:, 0, 1]+force[:, 1]*J[:, 1, 1]
    pt, pa = gt.copy(), ga.copy()
    pt[((theta <= -20+1e-8) & (gt >= 0)) | ((theta >= 20-1e-8) & (gt <= 0))] = 0
    pa[((A <= 1e-10) & (ga >= 0)) | ((A >= 6-1e-8) & (ga <= 0))] = 0
    return np.maximum(abs(pt), .25*abs(pa))


def state_diagnostics_m2(states, observation, coef, W):
    """Same fields as experiment.state_diagnostics (no knots: those flags are all False)."""
    theta, A = states.T
    pred, jac = forward_jac_m2(theta, A, coef)
    stationarity = physical_optimality_m2(states, observation, coef, W)
    sv = np.linalg.svd(np.einsum('ab,nbc->nac', np.linalg.cholesky(W).T, jac)*np.array([1., .25]), compute_uv=False)
    no = np.zeros(len(A), dtype=bool)
    return pred, dict(physical_projected_stationarity=stationarity, stationarity_verified=stationarity <= 1e-5,
        condition=sv[:, 0]/np.maximum(sv[:, 1], 1e-14), minimum_scaled_singular_value=sv[:, 1],
        physical_derivative_defined=~no, interior_protocol_knot=no.copy(), interior_knot_nondifferentiable=no.copy(),
        theta_bound=abs(theta) >= 20-1e-5, A_bound=(A <= 1e-5) | (A >= 6-1e-5),
        extrapolation=(A < CAL_A_RANGE[0]) | (A > CAL_A_RANGE[1]))


def free_inverse_m2(coef, W, observation):
    """Single-observation anchor-free inverse: bounded multistart LSQ, smallest-A-then-theta tie rule."""
    from scipy.optimize import least_squares
    L = np.linalg.cholesky(W).T
    def fun(z): return L@(forward_jac_m2(z[0], z[1], coef)[0]-observation)
    def jac(z): return L@forward_jac_m2(z[0], z[1], coef)[1]
    candidates = []
    for a in A_STARTS:
        for th in THETA_STARTS:
            opt = least_squares(fun, [th, a], jac=jac, bounds=([-20, 0], [20, 6]), max_nfev=100)
            candidates.append([float(np.dot(opt.fun, opt.fun)), float(opt.x[0]), float(opt.x[1])])
    best = min(candidates, key=lambda q: (q[0], q[2], q[1]))
    roots = [q for q in candidates if q[0] <= best[0]+1e-7]
    unique = []
    for q in sorted(roots, key=lambda q: (q[2], q[1])):
        if not any(np.linalg.norm(np.array(q[1:])-r[1:]) < 1e-4 for r in unique):
            unique.append(q)
    chosen = unique[0]
    sv = np.linalg.svd(L@jac(np.array(chosen[1:]))@np.diag([1., .25]), compute_uv=False)
    return dict(weighted_cost=chosen[0], theta_deg=chosen[1], A_diopters=chosen[2], equivalent_minima=unique,
                exact_root_cost_tolerance=1e-10, is_near_exact_root=chosen[0] < 1e-10,
                extrapolation=bool(chosen[2] < CAL_A_RANGE[0] or chosen[2] > CAL_A_RANGE[1]),
                branch_rule='Smallest A then theta among equivalent minima; no labels',
                conditioning=dict(physical_derivative_defined=True, scaled_singular_values=sv.tolist(),
                                  condition=float(sv[0]/max(sv[1], 1e-14))),
                theta_bound=bool(abs(chosen[1]) >= 20-1e-5), A_bound=bool(chosen[2] <= 1e-5 or chosen[2] >= 6-1e-5))


def grid_audit_m2(coef, W):
    """Sampled optical geometry/branches (no global uniqueness certificate); no knot scopes."""
    L = np.linalg.cholesky(W).T
    rows = []
    for scope, th_values, A_values in [('calibrated_grid', np.linspace(-15, 15, 17), np.linspace(CAL_A_RANGE[0], 4., 25)),
                                     ('full_bounds_grid', np.linspace(-20, 20, 17), np.linspace(0., 6., 25))]:
        for theta in th_values:
            for A in A_values:
                J = forward_jac_m2(np.array([theta]), np.array([A]), coef)[1][0]
                sv = np.linalg.svd(L@J@np.diag([1., .25]), compute_uv=False)
                slope = J[1, 1]-J[1, 0]*J[0, 1]/J[0, 0] if abs(J[0, 0]) > 1e-14 else np.nan
                rows.append(dict(scope=scope, theta_deg=float(theta), A_diopters=float(A), side='ordinary', derivative_defined=True,
                    determinant=float(np.linalg.det(J)), fixed_displacement_ratio_A_slope=float(slope),
                    condition=float(sv[0]/max(sv[1], 1e-14)), minimum_scaled_singular_value=float(sv[1])))
    theta, A = np.meshgrid(np.linspace(-15, 15, 9), np.unique(np.r_[np.linspace(CAL_A_RANGE[0], 4., 13), [2., 3., 4.]]))
    true_states = np.column_stack([theta.ravel(), A.ravel()])
    obs = forward_jac_m2(true_states[:, 0], true_states[:, 1], coef)[0]
    states, diagnostics = invert_batch_m2(obs, coef, W, maxiter=100)
    stationarity = physical_optimality_m2(states, obs, coef, W)
    roundtrip_error = np.linalg.norm((states-true_states)/np.array([1., .25]), axis=1)
    return dict(scope=f'finite sampled Jacobian grids and {len(states)} calibrated-support roundtrip probes; inverse searches full bounds; not global coverage or uniqueness proof',
        model_type=MODEL_TYPE,
        sample_counts_by_scope={scope: sum(row['scope'] == scope for row in rows) for scope in {row['scope'] for row in rows}},
        branch_probe_count=len(states), roundtrip_selected_state_mismatch_count=int(np.sum(roundtrip_error > 1e-4)),
        branch_probe_missing_root_count=int(np.sum(diagnostics['minimum_candidate_cost'] > 1e-10)),
        branch_probe_ambiguous_count=int(np.sum(diagnostics['equivalent_minima_count'] > 1)),
        branch_probe_stationarity_unverified_count=int(np.sum(stationarity > 1e-5)),
        physical_jacobian_samples=rows,
        synthetic_branch_probes=[dict(input_theta_deg=float(t), input_A_diopters=float(a), inferred_theta_deg=float(s[0]),
            inferred_A_diopters=float(s[1]), equivalent_minima_count=int(diagnostics['equivalent_minima_count'][i]),
            weighted_cost=float(diagnostics['weighted_cost'][i]), physical_projected_stationarity=float(stationarity[i]),
            reference_scaled_roundtrip_distance=float(roundtrip_error[i])) for i, ((t, a), s) in enumerate(zip(true_states, states))])
