"""Conditional P1 raw keystone calibration with analytical frame scales.

The sole learned-template scale gauge fixes sum(B**2) to the initializer's
native-pixel energy. The template centroid is fixed at zero as a coordinate
origin convention. No physical distortion center or optical zero is inferred.
There are no priors, output normalizations, or free isotropic K multipliers.

Global parameters: b, cx, cy, theta0_deg, then (learned only) three dimensionless
stereographic template coordinates. Search boxes are numerical envelopes only.
CPU NumPy float64 is used for equations, derivatives, and the small global solve.
Optional explicitly declared positive m_bounds replace the unconstrained scalar
profile with its exact box-constrained minimum. Bound activation is reported.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize

PARAM_NAMES = ("b", "cx", "cy", "theta0", "template_q1", "template_q2", "template_q3")
GAUGE = "sum(B**2)=sum(initial_reference**2); centroid(B)=0 is the fixed coordinate-origin convention; no mean-log(m) constraint"
# Characteristic units used ONLY for optimization coordinates and SVD reporting.
PARAM_SCALES = np.asarray([.05, .05, .05, 10., .1, .1, .1], dtype=np.float64)
DEFAULT_BOUNDS = np.asarray([[-.25, .25], [-.2, .2], [-.2, .2], [-30., 30.], [-.75, .75], [-.75, .75], [-.75, .75]])


def _reference_geometry(reference):
    reference = np.asarray(reference, dtype=np.float64)
    if reference.shape != (3, 2) or not np.isfinite(reference).all():
        raise ValueError("reference must be a finite native-pixel (3,2) triangle")
    if np.max(np.abs(reference.mean(axis=0))) > 1e-10 * max(1., np.linalg.norm(reference)):
        raise ValueError("reference must be centered; its origin is not fitted")
    # Helmert columns span precisely the three-vertex centered subspace.
    helmert = np.asarray([[1/np.sqrt(2), 1/np.sqrt(6)], [-1/np.sqrt(2), 1/np.sqrt(6)], [0., -2/np.sqrt(6)]])
    basis = np.zeros((6, 4), dtype=np.float64)
    basis[0::2, 0:2] = helmert
    basis[1::2, 2:4] = helmert
    energy = float(np.sum(reference**2))
    if energy <= 0:
        raise ValueError("reference energy must be positive")
    singular = np.linalg.svd(reference, compute_uv=False)
    if singular[-1] <= 1e-10 * singular[0]:
        raise ValueError("reference triangle must be noncollinear")
    length = np.sqrt(energy)
    unit = basis.T @ reference.reshape(-1) / length
    tangent = null_space(unit[None, :])
    return reference, basis, unit, tangent, length


def template(reference, q=None, derivatives=False):
    """Exact fixed-energy centered template, via S^3 stereographic coordinates.

    v(q)=[(1-|q|²)u+2Tq]/(1+|q|²). The chart omits only -u (at infinity).
    Finite search bounds additionally restrict its numerical search region.
    """
    ref, basis, unit, tangent, length = _reference_geometry(reference)
    if q is None:
        return (ref.copy(), np.empty((3, 2, 0))) if derivatives else ref.copy()
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (3,) or not np.isfinite(q).all():
        raise ValueError("learned template requires three finite chart coordinates")
    squared = float(q @ q)
    denominator = 1. + squared
    numerator = (1. - squared) * unit + 2. * (tangent @ q)
    vector = numerator / denominator
    B = (length * (basis @ vector)).reshape(3, 2)
    if not derivatives:
        return B
    dvector = (2. * tangent - 2. * unit[:, None] * q[None, :] - 2. * vector[:, None] * q[None, :]) / denominator
    return B, (length * basis @ dvector).reshape(3, 2, 3)


def model(theta, B, alpha, theta0, R0, derivatives=False):
    """Raw projective K and its centered output; retains K's native size."""
    theta = np.asarray(theta, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    if theta.ndim != 1 or B.shape != (3, 2) or alpha.shape != (3,):
        raise ValueError("theta must be (n,), B (3,2), alpha (b,cx,cy)")
    if not np.isfinite(theta).all() or not np.isfinite(B).all() or not np.isfinite(alpha).all() or not np.isfinite(theta0) or not np.isfinite(R0) or R0 <= 0:
        raise ValueError("model inputs must be finite, with R0>0")
    phi = (theta - float(theta0)) / 10.
    slope = (alpha[1] * B[:, 0] + alpha[2] * B[:, 1]) / R0
    denominator = 1. + phi[:, None] * slope[None, :]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        factors = np.exp(alpha[0] * phi[:, None]**2 * np.asarray([1., -1.])[None, :])
        Z = B[None, :, :] * factors[:, None, :] / denominator[..., None]
        H = Z - Z.mean(axis=1, keepdims=True)
    result = dict(Z=Z, H=H, denominator=denominator, phi=phi, factors=factors,
                  valid=np.isfinite(H).all(axis=(1, 2)) & np.all(denominator > 0., axis=1))
    if derivatives:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            dZ_dphi = Z * (2 * alpha[0] * phi[:, None, None] * np.asarray([1., -1.])[None, None, :] - slope[None, :, None] / denominator[..., None])
        result["dH_dtheta"] = (dZ_dphi - dZ_dphi.mean(axis=1, keepdims=True)) / 10.
    return result


class _Problem:
    def __init__(self, X, theta, fixation, reference, learned, R0, m_bounds=None):
        self.X = np.asarray(X, dtype=np.float64)
        self.theta = np.asarray(theta, dtype=np.float64)
        self.fixation = np.asarray(fixation)
        self.reference, self.basis, self.unit, self.tangent, self.length = _reference_geometry(reference)
        self.learned = bool(learned)
        self.R0 = float(self.length / np.sqrt(3.) if R0 is None else R0)
        self.m_bounds = None if m_bounds is None else np.asarray(m_bounds, dtype=np.float64)
        if self.m_bounds is not None and (self.m_bounds.shape != (2,) or not np.isfinite(self.m_bounds).all() or not (0. < self.m_bounds[0] < self.m_bounds[1])):
            raise ValueError("m_bounds must be an explicit finite positive (lower,upper) interval with lower<upper")
        n = len(self.theta)
        if n == 0 or self.X.shape != (n, 3, 2) or self.fixation.shape != (n,):
            raise ValueError("nonempty row-aligned X (n,3,2), theta (n,), fixation (n,) required")
        if not np.isfinite(self.X).all() or not np.isfinite(self.theta).all() or not np.isfinite(self.R0) or self.R0 <= 0:
            raise ValueError("finite observations/gaze and positive frozen R0 required")
        if np.max(np.abs(self.X.mean(axis=1))) > 1e-10 * max(1., np.linalg.norm(self.X, axis=(1, 2)).max()):
            raise ValueError("X must be centered observations; no per-frame center nuisance is fitted")
        self.groups, inverse, counts = np.unique(self.fixation, return_inverse=True, return_counts=True)
        # Each fixation has weight 1/F; each vertex squared Euclidean distance 1/3.
        self.weights = 1. / (len(counts) * counts[inverse] * 3.)
        self.p = 7 if self.learned else 4
        self.names = PARAM_NAMES[:self.p]
        self.rejected_domain_evaluations = 0
        self.max_trial_invalid_rows = 0
        self.min_trial_denominator = float("inf")

    def unpack(self, params):
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.p,) or not np.isfinite(params).all():
            raise ValueError(f"expected {self.p} finite parameters ordered {self.names}")
        if not self.learned:
            return params, self.reference.copy(), np.empty((3, 2, 0))
        q = params[4:]
        squared = q @ q
        denominator = 1 + squared
        vector = ((1-squared)*self.unit + 2*self.tangent @ q) / denominator
        B = (self.length * self.basis @ vector).reshape(3, 2)
        dvector = (2*self.tangent - 2*self.unit[:, None]*q - 2*vector[:, None]*q) / denominator
        dB = (self.length*self.basis @ dvector).reshape(3, 2, 3)
        return params, B, dB

    def evaluate(self, params, derivatives=False, jacobian=False):
        params, B, dB = self.unpack(params)
        out = model(self.theta, B, params[:3], params[3], self.R0, derivatives=True)
        H, Z, den, phi = (out[key] for key in ("H", "Z", "denominator", "phi"))
        D = np.sum(H*H, axis=(1, 2))
        weak = (~np.isfinite(D)) | (D <= self.length**2 * np.finfo(float).eps)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            m_unconstrained = np.sum(H*self.X, axis=(1, 2))/D
            lower_active = np.zeros(len(m_unconstrained), dtype=bool)
            upper_active = np.zeros(len(m_unconstrained), dtype=bool)
            bound_tie = np.zeros(len(m_unconstrained), dtype=bool)
            if self.m_bounds is None:
                m = m_unconstrained.copy()
            else:
                lower, upper = self.m_bounds
                lower_active = m_unconstrained < lower
                upper_active = m_unconstrained > upper
                bound_tie = (m_unconstrained == lower) | (m_unconstrained == upper)
                # Exact minimizer of ||X-mH||² over the explicitly declared box.
                m = np.minimum(upper, np.maximum(lower, m_unconstrained))
            strict_active = lower_active | upper_active
            bound_active = strict_active | bound_tie
            prediction = m[:, None, None]*H
            residual = self.X-prediction
        valid = out["valid"] & ~weak & np.isfinite(m_unconstrained) & np.isfinite(m) & (m > 0.) & np.isfinite(residual).all(axis=(1, 2))
        result = dict(params=params.copy(), param_names=list(self.names), B=B, alpha=params[:3].copy(),
                      theta0=float(params[3]), R0=self.R0, m=m, m_unconstrained=m_unconstrained,
                      m_bounds=None if self.m_bounds is None else self.m_bounds.copy(),
                      m_lower_active=lower_active, m_upper_active=upper_active, m_bound_tie=bound_tie,
                      m_bound_active=bound_active, prediction=prediction,
                      residual=residual, valid=valid, denominator=den, weights=self.weights.copy(),
                      H=H, raw_K_size_ratio=np.sqrt(D)/self.length,
                      raw_K_area_scale=np.where(np.all(den > 0, axis=1), 1/np.sqrt(np.abs(np.prod(den, axis=1))), np.nan), weak_shape=weak,
                      loss=float(np.sum(self.weights*np.sum(residual**2, axis=(1, 2)))) if valid.all() else float("inf"),
                      gauge=GAUGE if self.learned else "B fixed to initial centered reference; no additional scale gauge",
                      template_energy=float(np.sum(B**2)), reference_energy=float(self.length**2))
        result["magnification_constraint"] = dict(
            bounds=None if self.m_bounds is None else self.m_bounds.copy(),
            method="unconstrained positive scalar profile" if self.m_bounds is None else "exact scalar least-squares minimum over explicit positive interval",
            unconstrained_m_min=float(np.min(m_unconstrained)), unconstrained_m_max=float(np.max(m_unconstrained)),
            unconstrained_nonpositive_rows=int(np.sum(m_unconstrained <= 0.)),
            lower_active_rows=int(lower_active.sum()), upper_active_rows=int(upper_active.sum()),
            lower_active_fraction=float(lower_active.mean()), upper_active_fraction=float(upper_active.mean()),
            bound_tie_rows=int(bound_tie.sum()), bound_tie_fraction=float(bound_tie.mean()),
            bound_active_rows=int(bound_active.sum()), bound_active_fraction=float(bound_active.mean()),
            derivative_convention="dm=0 on strictly active rows; exact ties use unconstrained interior derivative branch and are nonsmooth")
        if not derivatives:
            return result
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            dZ = [Z*phi[:, None, None]**2*np.asarray([1., -1.])[None, None, :],
                  -Z*(phi[:, None]*B[None, :, 0]/self.R0/den)[..., None],
                  -Z*(phi[:, None]*B[None, :, 1]/self.R0/den)[..., None]]
            dH = [v-v.mean(axis=1, keepdims=True) for v in dZ]
            dH.append(-out["dH_dtheta"])
            for k in range(dB.shape[-1]):
                dden = phi[:, None]*(params[1]*dB[:, 0, k]+params[2]*dB[:, 1, k])[None, :]/self.R0
                dz = dB[None, :, :, k]*out["factors"][:, None, :]/den[..., None]-Z*(dden/den)[..., None]
                dH.append(dz-dz.mean(axis=1, keepdims=True))
            dH = np.stack(dH, axis=-1)
            gradient = -2*np.einsum("n,n,njcp,njc->p", self.weights, m, dH, residual, optimize=True)
            dm_unconstrained = (np.einsum("njcp,njc->np", dH, self.X, optimize=True)-2*m_unconstrained[:, None]*np.einsum("njcp,njc->np", dH, H, optimize=True))/D[:, None]
            dH_theta = out["dH_dtheta"]
            dm_theta_unconstrained = (np.sum(dH_theta*self.X, axis=(1, 2))-2*m_unconstrained*np.sum(dH_theta*H, axis=(1, 2)))/D
            dm_X_unconstrained = H/D[:, None, None]
            noise_gain_unconstrained = 1/np.sqrt(D)
        dm = dm_unconstrained.copy()
        dm_theta = dm_theta_unconstrained.copy()
        dm_X = dm_X_unconstrained.copy()
        noise_gain = noise_gain_unconstrained.copy()
        dm[strict_active] = 0.
        dm_theta[strict_active] = 0.
        dm_X[strict_active] = 0.
        noise_gain[strict_active] = 0.
        result.update(gradient=gradient, dm_dparams=dm, dm_dtheta=dm_theta,
                      dm_dparams_unconstrained=dm_unconstrained, dm_dtheta_unconstrained=dm_theta_unconstrained,
                      dm_dX_unconstrained=dm_X_unconstrained, scale_coordinate_noise_gain_unconstrained=noise_gain_unconstrained,
                      dH_dtheta=dH_theta, dm_dX=dm_X, scale_coordinate_noise_gain=noise_gain)
        if jacobian:
            # Residual r=X-mH, including the exact analytical profiler derivative.
            J = -m[:, None, None, None]*dH-H[..., None]*dm[:, None, None, :]
            result["residual_jacobian"] = (np.sqrt(self.weights)[:, None, None, None]*J).reshape(-1, self.p)
        return result

    def objective(self, params):
        result = self.evaluate(params, derivatives=True)
        self.min_trial_denominator = min(self.min_trial_denominator, float(np.min(result["denominator"])))
        self.max_trial_invalid_rows = max(self.max_trial_invalid_rows, int(np.sum(~result["valid"])))
        if not result["valid"].all() or not np.isfinite(result["gradient"]).all():
            # A rejected-domain sentinel, never a reported loss or a prior.
            self.rejected_domain_evaluations += 1
            return float("inf"), np.zeros(self.p)
        return result["loss"], result["gradient"]


def evaluate(params, X, theta, fixation, reference, learned=True, R0=None, derivatives=False, m_bounds=None):
    """Profile every frame scalar using the explicitly selected scalar domain.

    derivatives=True returns exact global residual Jacobian, dm/dglobals,
    dm/dtheta at fixed observation, and dm/dX at fixed global calibration.
    m_bounds=None preserves the positivity-only unconstrained profile. Explicit
    m_bounds=(lower,upper) uses the exact constrained scalar optimum and reports
    bound activation. Its scalar map is nonsmooth at exact raw-profile ties;
    returned derivatives at ties follow the interior unconstrained branch.
    """
    return _Problem(X, theta, fixation, reference, learned, R0, m_bounds).evaluate(params, derivatives, derivatives)


def _bounds(bounds, p):
    value = DEFAULT_BOUNDS[:p].copy() if bounds is None else np.asarray(bounds, dtype=np.float64)
    if value.shape != (p, 2) or not np.isfinite(value).all() or np.any(value[:, 0] >= value[:, 1]):
        raise ValueError(f"bounds must be finite ({p},2) lower/upper numerical envelopes")
    return value


def _stationarity(params, gradient, bounds, tolerance=1e-5):
    scales = PARAM_SCALES[:len(params)]
    distances = np.minimum(params-bounds[:, 0], bounds[:, 1]-params)
    active = distances <= 1e-6*np.maximum(bounds[:, 1]-bounds[:, 0], 1.)
    projected = gradient*scales
    projected[(params <= bounds[:, 0]+1e-7*np.maximum(1., bounds[:, 1]-bounds[:, 0])) & (gradient > 0)] = 0
    projected[(params >= bounds[:, 1]-1e-7*np.maximum(1., bounds[:, 1]-bounds[:, 0])) & (gradient < 0)] = 0
    return dict(gradient_norm=float(np.linalg.norm(gradient)), scaled_gradient_inf=float(np.max(np.abs(gradient*scales))),
                projected_gradient_norm=float(np.max(np.abs(projected))), stationary=bool(np.max(np.abs(projected)) <= tolerance),
                active_bounds=[PARAM_NAMES[j] for j in np.flatnonzero(active)])


def _solve(problem, initial, bounds, maxiter, fixed=None):
    p = problem.p
    problem.rejected_domain_evaluations = 0
    problem.max_trial_invalid_rows = 0
    problem.min_trial_denominator = float("inf")
    free = np.asarray([j for j in range(p) if fixed is None or j not in fixed], dtype=int)
    base = np.asarray(initial, dtype=np.float64).copy()
    if fixed:
        for j, v in fixed.items():
            base[j] = v
    scales = PARAM_SCALES[:p]
    solve_scales = scales.copy()
    def unpack(y):
        x = base.copy()
        x[free] = y*solve_scales[free]
        return x
    def objective(y):
        value, gradient = problem.objective(unpack(y))
        return value, gradient[free]*solve_scales[free]
    attempts = []
    best = None
    # Domain-rejected first trial steps can stop L-BFGS with no useful move.
    # Retry smaller numerical coordinates when the true projected gradient
    # disproves convergence; this changes neither data objective nor bounds.
    for scale_factor in (1., .1, .01):
        solve_scales = scales*scale_factor
        solution = minimize(objective, base[free]/solve_scales[free], jac=True, method="L-BFGS-B",
                            bounds=bounds[free]/solve_scales[free, None],
                            options=dict(maxiter=int(maxiter), ftol=1e-14, gtol=1e-8*scale_factor, maxls=40, maxcor=20))
        candidate = unpack(solution.x)
        ev = problem.evaluate(candidate, derivatives=True)
        grad = ev["gradient"].copy()
        if fixed:
            grad[list(fixed)] = 0.
        stationary = _stationarity(candidate, grad, bounds)["stationary"]
        attempts.append(dict(coordinate_scale_factor=scale_factor, loss=ev["loss"], stationary=stationary, status=int(solution.status), iterations=int(solution.nit), evaluations=int(solution.nfev)))
        if best is None or ev["loss"] < best[0]:
            best = (ev["loss"], candidate.copy(), ev, solution)
        if stationary and ev["valid"].all():
            break
        if ev["valid"].all():
            base = candidate.copy()
    _, params, final, solution = best
    stationarity = _stationarity(params, final["gradient"], bounds)
    if fixed:
        # Stationarity on only the reoptimized nuisance subspace.
        nuisance_bounds = bounds.copy()
        nuisance_gradient = final["gradient"].copy()
        nuisance_gradient[list(fixed)] = 0.
        projected = _stationarity(params, nuisance_gradient, nuisance_bounds)
        stationarity["stationary"] = projected["stationary"]
        stationarity["projected_gradient_norm"] = projected["projected_gradient_norm"]
        stationarity["nuisance_scaled_gradient_inf"] = float(np.max(np.abs(final["gradient"][free]*scales[free])))
    return dict(params=params, loss=final["loss"], success=bool(solution.success), status=int(solution.status),
                message=str(solution.message), iterations=int(solution.nit), evaluations=int(solution.nfev),
                valid=bool(final["valid"].all()), optimizer_attempts=attempts,
                rejected_domain_evaluations=problem.rejected_domain_evaluations,
                max_trial_invalid_rows=problem.max_trial_invalid_rows, min_trial_denominator=problem.min_trial_denominator,
                magnification_constraint=final["magnification_constraint"], **stationarity)


def fit(X, theta, fixation, reference, learned=True, R0=None, n_starts=8, seed=1729, bounds=None, initial=None, maxiter=1000, m_bounds=None):
    """Multistart data-only profiled fit; preserve all starts and domain failures."""
    problem = _Problem(X, theta, fixation, reference, learned, R0, m_bounds)
    bounds = _bounds(bounds, problem.p)
    if n_starts < 1:
        raise ValueError("n_starts must be at least one")
    initializer = np.zeros(problem.p) if initial is None else np.asarray(initial, dtype=np.float64)
    if initializer.shape != (problem.p,) or np.any(initializer < bounds[:, 0]) or np.any(initializer > bounds[:, 1]):
        raise ValueError("initial parameters must be inside the search envelope")
    rng = np.random.default_rng(seed)
    initials = [initializer.copy()]
    theta_starts = (-20., -10., 10., 20., -5., 5., 0.)
    for index in range(1, int(n_starts)):
        candidate = initializer.copy()
        candidate[:3] += rng.normal(size=3)*np.asarray([.02, .02, .02])
        candidate[3] = theta_starts[(index-1) % len(theta_starts)]
        if learned:
            candidate[4:] += rng.normal(size=3)*.015
        candidate = np.clip(candidate, bounds[:, 0]+1e-8, bounds[:, 1]-1e-8)
        # Initialization rejection is recorded; no observation is excluded.
        if not problem.evaluate(candidate)["valid"].all():
            candidate = initializer.copy()
        initials.append(candidate)
    starts = []
    for index, start in enumerate(initials):
        result = _solve(problem, start, bounds, maxiter)
        result.update(start=index, initial_params=start)
        starts.append(result)
    valid_starts = [s for s in starts if s["valid"] and np.isfinite(s["loss"])]
    if not valid_starts:
        raise ValueError("all starts failed positive-scale/projective/shape domain")
    best = min(valid_starts, key=lambda item: item["loss"])
    final = problem.evaluate(best["params"], derivatives=True)
    final.update(starts=starts, best_start=best["start"], bounds=bounds, param_scales=PARAM_SCALES[:problem.p].copy(),
                 numerical_search_envelope="finite optimizer boxes only; not physical limits; repeat with widened boxes to audit sensitivity",
                 **{key: best[key] for key in ("stationary", "projected_gradient_norm", "gradient_norm", "active_bounds", "success", "message")})
    # Stability at identical observations; retain all valid starts, not only successful solver flags.
    losses = np.asarray([s["loss"] for s in valid_starts])
    final["start_stability"] = dict(valid_starts=len(valid_starts), loss_min=float(losses.min()), loss_max=float(losses.max()),
                                    parameter_min=np.min([s["params"] for s in valid_starts], axis=0), parameter_max=np.max([s["params"] for s in valid_starts], axis=0))
    scale_rms, prediction_rms, template_rms = [], [], []
    for start in valid_starts:
        ev = problem.evaluate(start["params"])
        scale_rms.append(float(np.sqrt(np.mean((ev["m"]-final["m"])**2))))
        prediction_rms.append(float(np.sqrt(np.mean((ev["prediction"]-final["prediction"])**2))))
        template_rms.append(float(np.sqrt(np.mean((ev["B"]-final["B"])**2))))
    final["start_stability"].update(scale_difference_rms=scale_rms, prediction_difference_coordinate_rms=prediction_rms, template_difference_coordinate_rms=template_rms)
    return final


def diagnostics(params, X, theta, fixation, reference, learned=True, R0=None, bounds=None, rank_rtol=1e-8, m_bounds=None):
    """Data-only profiled residual rank, stationarity, domain, and noise gains."""
    problem = _Problem(X, theta, fixation, reference, learned, R0, m_bounds)
    bounds = _bounds(bounds, problem.p)
    result = problem.evaluate(params, derivatives=True, jacobian=True)
    if not result["valid"].all():
        raise ValueError("diagnostics cannot assert a rank for invalid fitted rows")
    J = result["residual_jacobian"]
    def spectrum(jac):
        _, singular, right = np.linalg.svd(jac, full_matrices=False)
        tolerance = max(float(singular[0])*rank_rtol, np.finfo(float).eps*max(jac.shape)*float(singular[0]))
        rank = int(np.sum(singular > tolerance))
        return dict(singular_values=singular, rank=rank, rank_tolerance=tolerance, right_singular_vectors=right,
                    condition_number=float(singular[0]/singular[-1]) if singular[-1] > 0 else float("inf"))
    raw = spectrum(J)
    scaled = spectrum(J*PARAM_SCALES[:problem.p][None, :])
    orthogonal = np.sum(result["residual"]*result["H"], axis=(1, 2))
    lower_active = result["m_lower_active"]
    upper_active = result["m_upper_active"]
    interior = ~(lower_active | upper_active | result["m_bound_tie"])
    scalar_gradient = -2*orthogonal
    scalar_kkt_violation = np.abs(scalar_gradient)
    scalar_kkt_violation[lower_active] = np.maximum(-scalar_gradient[lower_active], 0.)
    scalar_kkt_violation[upper_active] = np.maximum(scalar_gradient[upper_active], 0.)
    def maximum_absolute_on(values, mask):
        return float(np.max(np.abs(values[mask]))) if np.any(mask) else None
    def range_on(values, mask):
        return [float(np.min(values[mask])), float(np.max(values[mask]))] if np.any(mask) else None
    def area(v):
        a, b = v[..., 1, :]-v[..., 0, :], v[..., 2, :]-v[..., 0, :]
        return .5*np.abs(a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0])
    direct_area_ratio = area(result["H"])/area(result["B"])
    area_identity_error = float(np.max(np.abs(direct_area_ratio-result["raw_K_area_scale"]**2)))
    m = result["m"]
    return dict(**raw, unit_scaled_spectrum=scaled, parameter_units=["dimensionless", "dimensionless", "dimensionless", "degree"]+(["dimensionless"]*3 if learned else []),
                parameter_scales=PARAM_SCALES[:problem.p].copy(), param_names=list(problem.names), rank_rtol=rank_rtol,
                rank_definition="weighted data-only residual Jacobian after exact selected scalar profiling; no prior/constraint residuals appended; bounded case conditional on reported active set and interior derivative branch at nonsmooth ties",
                magnification_constraint=result["magnification_constraint"],
                **_stationarity(np.asarray(params), result["gradient"], bounds),
                domain=dict(rows=len(m), invalid_rows=int((~result["valid"]).sum()), nonpositive_m=int(np.sum(m <= 0)), weak_shape_rows=int(result["weak_shape"].sum()),
                            nonpositive_projective_rows=int(np.any(result["denominator"] <= 0, axis=1).sum()), denominator_min=float(result["denominator"].min()), m_min=float(m.min()), m_max=float(m.max())),
                shape_scalar_orthogonality_max=float(np.max(np.abs(orthogonal))),
                shape_scalar_orthogonality_interpretation="all-row residual dot H is descriptive; scalar orthogonality applies only to unconstrained/interior rows, while active scalar bounds require the reported KKT signs",
                interior_shape_scalar_orthogonality_max=maximum_absolute_on(orthogonal, interior),
                scalar_bound_kkt=dict(gradient_definition="d(sum_vertex_coordinate_squared_error)/dm=-2*dot(residual,H), native px² per unit m",
                                      max_violation=float(np.max(scalar_kkt_violation)),
                                      lower_active_gradient_range=range_on(scalar_gradient, lower_active),
                                      upper_active_gradient_range=range_on(scalar_gradient, upper_active),
                                      interior_gradient_max_absolute=maximum_absolute_on(scalar_gradient, interior),
                                      expected_signs="lower-active gradient>=0; upper-active gradient<=0; interior gradient=0; exact bound ties are scalar-map nonsmooth points"),
                raw_K_area_identity_max_absolute_error=area_identity_error,
                scale_coordinate_noise_gain_min=float(result["scale_coordinate_noise_gain"].min()), scale_coordinate_noise_gain_max=float(result["scale_coordinate_noise_gain"].max()),
                scale_coordinate_noise_gain_unconstrained_min=float(result["scale_coordinate_noise_gain_unconstrained"].min()),
                scale_coordinate_noise_gain_unconstrained_max=float(result["scale_coordinate_noise_gain_unconstrained"].max()),
                template_centroid=result["B"].mean(axis=0), template_energy=result["template_energy"], reference_energy=result["reference_energy"],
                gauge=result["gauge"], bounds=bounds, objective=result["loss"])


def profile(params, X, theta, fixation, reference, learned=True, R0=None, indices=None, grids=None, bounds=None, maxiter=400, n_starts=2, seed=31415, m_bounds=None):
    """Data-only parameter profiles with nuisance reoptimization and scale drift.

    Defaults profile theta0,cx,cy and every learned-template chart coordinate.
    Grids can map parameter names (or indices) to explicit native-unit values.
    Profile deltas are objective differences, not posterior confidence bounds.
    """
    problem = _Problem(X, theta, fixation, reference, learned, R0, m_bounds)
    bounds = _bounds(bounds, problem.p)
    params = np.asarray(params, dtype=np.float64)
    baseline = problem.evaluate(params)
    if not baseline["valid"].all():
        raise ValueError("profile baseline must satisfy the model domain")
    if indices is None:
        indices = [3, 1, 2]+(list(range(4, 7)) if learned else [])
    indices = [problem.names.index(j) if isinstance(j, str) else int(j) for j in indices]
    grids = {} if grids is None else grids
    rng = np.random.default_rng(seed)
    output = {}
    for j in indices:
        name = problem.names[j]
        grid = grids.get(name, grids.get(j, None)) if isinstance(grids, dict) else grids[j]
        if grid is None:
            grid = np.unique(np.r_[np.linspace(bounds[j, 0], bounds[j, 1], 9), params[j]])
        grid = np.asarray(grid, dtype=np.float64)
        if grid.ndim != 1 or not np.isfinite(grid).all() or np.any(grid < bounds[j, 0]) or np.any(grid > bounds[j, 1]):
            raise ValueError(f"profile grid for {name} must lie within declared numerical search boxes")
        points = []
        previous = params.copy()
        # Start near baseline; then move to further locations for stable continuation.
        for value in sorted(grid.tolist(), key=lambda v: abs(v-params[j])):
            candidates = []
            for k in range(max(1, int(n_starts))):
                initial = params.copy() if k == 0 else previous.copy()
                if k > 1:
                    initial += rng.normal(size=problem.p)*PARAM_SCALES[:problem.p]*.1
                    initial = np.clip(initial, bounds[:, 0], bounds[:, 1])
                initial[j] = value
                if problem.evaluate(initial)["valid"].all():
                    candidates.append(_solve(problem, initial, bounds, maxiter, {j: value}))
            valid = [c for c in candidates if c["valid"] and np.isfinite(c["loss"])]
            if not valid:
                points.append(dict(value=float(value), loss=float("inf"), delta_loss=float("inf"), valid=False, starts=candidates))
                continue
            best = min(valid, key=lambda c: c["loss"])
            previous = best["params"].copy()
            ev = problem.evaluate(best["params"])
            points.append(dict(value=float(value), loss=best["loss"], delta_loss=best["loss"]-baseline["loss"], valid=True,
                               params=best["params"], stationary=best["stationary"], projected_gradient_norm=best["projected_gradient_norm"], active_bounds=best["active_bounds"],
                               m_mean=float(ev["m"].mean()), m_min=float(ev["m"].min()), m_max=float(ev["m"].max()),
                               magnification_constraint=ev["magnification_constraint"],
                               scale_difference_rms=float(np.sqrt(np.mean((ev["m"]-baseline["m"])**2))),
                               prediction_difference_coordinate_rms=float(np.sqrt(np.mean((ev["prediction"]-baseline["prediction"])**2))),
                               template_difference_coordinate_rms=float(np.sqrt(np.mean((ev["B"]-baseline["B"])**2))),
                               denominator_min=float(ev["denominator"].min()), starts=candidates))
        output[name] = sorted(points, key=lambda point: point["value"])
    return dict(profiles=output, baseline_loss=baseline["loss"], baseline_params=params.copy(), bounds=bounds,
                m_bounds=baseline["m_bounds"], baseline_magnification_constraint=baseline["magnification_constraint"],
                gauge=baseline["gauge"], interpretation="data-only conditional loss profiles; numerical box endpoints are not physical bounds; no confidence cutoff imposed")
