"""Bounded variable projection with exact matrix-free profile derivatives."""
from __future__ import annotations
import time
import numpy as np
from scipy.linalg import qr, solve_triangular
from scipy.optimize import least_squares
from scipy.sparse.linalg import LinearOperator
from .model import STATE_SCALE, LOWER, UPPER
from .noise import whitening


class ProfiledProblem:
    def __init__(self, model, y, r, cov, groups, anchors, prior_strength=.001,
                 anchor_scales=(.1, .25)):
        self.model, self.y, self.r = model, np.asarray(y), np.asarray(r)
        _, self.groups = np.unique(groups, return_inverse=True)
        self.anchors, self.anchor_scales = np.asarray(anchors), np.asarray(anchor_scales)
        if (self.anchor_scales.shape != (2,) or not np.isfinite(self.anchor_scales).all()
                or np.any(self.anchor_scales <= 0)):
            raise ValueError("Anchor scales must be two finite positive physical-unit values")
        if not np.isfinite(prior_strength) or prior_strength < 0:
            raise ValueError("Prior strength must be finite and nonnegative")
        self.n, self.c = y.shape
        self.k = len(anchors)
        self.counts = np.bincount(self.groups, minlength=self.k)
        if np.any(self.counts == 0):
            raise ValueError("Every calibration group needs observations")
        self.weights = whitening(cov)/np.sqrt(self.k*self.counts[self.groups])[:, None, None]
        self.bdata = np.einsum("nij,nj->ni", self.weights, y).ravel()
        x0 = self.anchors[self.groups]
        A = self.weighted_design(x0)
        self.column_scale = np.maximum(np.linalg.norm(A, axis=0), 1e-12)
        normalized = A/self.column_scale
        coef, _, self.design_rank, singular = np.linalg.lstsq(normalized, self.bdata, rcond=1e-10)
        self.beta0 = coef/self.column_scale
        self.design_singular = singular
        self.penalty = np.sqrt(prior_strength)*self.column_scale
        self.penalty[model.intercepts] = 0
        self.b = np.r_[self.bdata, self.penalty*self.beta0]
        self.last_z = None
        self.prior_strength = prior_strength

    def weighted_design(self, x):
        return np.einsum("nij,njp->nip", self.weights,
                         self.model.design(x, self.r)).reshape(self.n*self.c, self.model.size)

    def mean_states(self, x):
        means = np.zeros((self.k, 2))
        np.add.at(means, self.groups, x)
        return means/self.counts[:, None]

    def update(self, z):
        z = np.asarray(z).ravel()
        if self.last_z is not None and np.array_equal(z, self.last_z):
            return
        self.last_z = z.copy()
        self.x = z.reshape(-1, 2)*STATE_SCALE
        A, dA = self.model.design(self.x, self.r, True)
        self.C = np.einsum("nij,njpz->nipz", self.weights, dA)*STATE_SCALE
        Bdata = np.einsum("nij,njp->nip", self.weights, A).reshape(-1, self.model.size)
        self.B = np.vstack((Bdata, np.diag(self.penalty)))
        # Normalizing columns improves factorization without changing the prior.
        self.Q, self.R = qr(self.B/self.column_scale, mode="economic")
        if np.min(np.abs(np.diag(self.R))) < 1e-12:
            raise ValueError("Unidentifiable coefficient design under declared prior")
        self.beta = solve_triangular(self.R, self.Q.T@self.b)/self.column_scale
        self.optical_residual = self.B@self.beta-self.b
        anchor = (self.mean_states(self.x)-self.anchors)/self.anchor_scales/np.sqrt(self.k)
        self.residual = np.r_[self.optical_residual, anchor.ravel()]
        self.S = np.einsum("ncpz,p->ncz", self.C, self.beta)

    def fun(self, z):
        self.update(z)
        return self.residual.copy()

    def jvp(self, v):
        return self.jac(self.last_z).matvec(v)

    def vjp(self, w):
        return self.jac(self.last_z).rmatvec(w)

    def jac(self, z):
        self.update(z)
        # SciPy evaluates trial residuals before reusing the current Jacobian.
        # Capture its arrays: a live view of this mutable cache changes the
        # trust-region linearization when a trial is rejected.
        B, C, S, R = self.B, self.C, self.S, self.R
        residual = self.optical_residual
        column_scale = self.column_scale
        n, c, p = self.n, self.c, self.model.size
        def hsolve(t):
            q = solve_triangular(R.T, t/column_scale, lower=True)
            return solve_triangular(R, q)/column_scale
        def jvp(v):
            v = np.asarray(v).reshape(n, 2)
            a = np.einsum("ncz,nz->nc", S, v).ravel()
            dB = np.einsum("ncpz,nz->ncp", C, v).reshape(-1, p)
            db = -hsolve(dB.T@residual[:n*c]+B[:n*c].T@a)
            anchor = self.mean_states(v*STATE_SCALE)/self.anchor_scales/np.sqrt(self.k)
            return np.r_[np.r_[a, np.zeros(p)]+B@db, anchor.ravel()]
        def vjp(w):
            w = np.asarray(w).ravel()
            optical, anchor = w[:len(self.b)], w[len(self.b):].reshape(self.k, 2)
            u = hsolve(B.T@optical)
            projected = (optical-B@u)[:n*c].reshape(n, c)
            answer = np.einsum("ncz,nc->nz", S, projected)
            answer -= np.einsum("ncpz,p,nc->nz", C, u, residual[:n*c].reshape(n, c))
            answer += anchor[self.groups]*STATE_SCALE/(self.counts[self.groups, None]*
                                                       self.anchor_scales*np.sqrt(self.k))
            return answer.ravel()
        return LinearOperator((len(self.residual), 2*n), matvec=jvp, rmatvec=vjp, dtype=float)


def projected_gradient(x, g, lower, upper):
    """Unit-step projected gradient mapping in the caller's declared units.

    Unlike clipping signs only at exact bounds, this remains continuous for
    interior trust-region iterates that approach an active bound.
    """
    x, g = np.asarray(x), np.asarray(g)
    return float(np.max(np.abs(x-np.clip(x-g, lower, upper))))


def fit(model, y, r, cov, groups, anchors, prior_strength=.001, starts=2,
        max_nfev=300, seed=17, progress=None, additional_initial_states=None,
        anchor_scales=(.1, .25), checkpoint=None):
    problem = ProfiledProblem(model, y, r, cov, groups, anchors, prior_strength, anchor_scales)
    if starts < 1 or max_nfev < 1:
        raise ValueError("Starts and evaluation budget must be positive")
    lower = np.tile(LOWER/STATE_SCALE, problem.n)
    upper = np.tile(UPPER/STATE_SCALE, problem.n)
    rng = np.random.default_rng(seed)
    alternatives, solutions = [], []
    extra = [] if additional_initial_states is None else additional_initial_states
    for start in range(starts+len(extra)):
        z = problem.anchors[problem.groups]/STATE_SCALE
        if start >= starts:
            initial = np.asarray(extra[start-starts], float)
            if initial.shape != (problem.n, 2) or not np.isfinite(initial).all():
                raise ValueError("Additional initial states must match finite training trajectories")
            z = initial/STATE_SCALE
            z = np.clip(z, lower.reshape(-1, 2)+1e-9, upper.reshape(-1, 2)-1e-9)
        elif start:
            # Zero-mean within-fixation perturbations; no baseline state penalty.
            delta = rng.normal(size=z.shape)*np.array([.04, .08])
            delta -= problem.mean_states(delta)[problem.groups]
            z = np.clip(z+delta, lower.reshape(-1, 2)+1e-9, upper.reshape(-1, 2)-1e-9)
        begun = time.monotonic()
        certified = False
        previous = None
        def check_step(intermediate_result):
            nonlocal certified, previous
            point = intermediate_result.x
            problem.update(point)
            gradient = problem.vjp(problem.residual)/np.tile(STATE_SCALE, problem.n)
            pg = projected_gradient(point* np.tile(STATE_SCALE, problem.n), gradient,
                                    lower*np.tile(STATE_SCALE, problem.n), upper*np.tile(STATE_SCALE, problem.n))
            cost = float(intermediate_result.cost)
            if previous is not None:
                stable_cost = abs(cost-previous[0]) <= 1e-9*(1+cost)
                stable_step = np.max(np.abs((point-previous[1])*np.tile(STATE_SCALE, problem.n))) < 1e-4
                if pg < 1e-3 and stable_cost and stable_step:
                    certified = True
                    raise StopIteration
            previous = (cost, point.copy())
        result = least_squares(problem.fun, z.ravel(), jac=problem.jac, bounds=(lower, upper),
                               method="trf", tr_solver="lsmr", x_scale=1.,
                               ftol=None, xtol=1e-12, gtol=1e-4, max_nfev=max_nfev,
                               tr_options={"atol": 1e-9, "btol": 1e-9, "maxiter": 200},
                               callback=check_step)
        problem.update(result.x)
        grad = problem.vjp(problem.residual)
        stationarity = projected_gradient(result.x, grad, lower, upper)
        physical_stationarity = projected_gradient(problem.x.ravel(), grad/np.tile(STATE_SCALE, problem.n),
                                                    np.tile(LOWER, problem.n), np.tile(UPPER, problem.n))
        inner = float(np.max(np.abs(problem.B.T@problem.optical_residual)/problem.column_scale))
        # Strict independent checks; hitting a budget is never convergence.
        converged = bool((certified or result.success) and physical_stationarity < 1e-3 and inner < 1e-7)
        record = dict(start=start, cost=float(result.cost), nfev=result.nfev,
                      initialization="additional_training_states" if start >= starts else "nominal" if start == 0 else "perturbed_nominal",
                      status=result.status, message=result.message, converged=converged,
                      projected_stationarity_encoded=stationarity,
                      projected_stationarity_physical=physical_stationarity,
                      stable_step_cost_certificate=certified,
                      acceptance_reason="stable_step_cost_and_stationarity" if certified and converged else
                          "solver_stop_and_stationarity" if converged else "not_certified",
                      objective_components=dict(optical=float(np.sum(problem.optical_residual[:problem.n*problem.c]**2)/2),
                          prior=float(np.sum(problem.optical_residual[problem.n*problem.c:]**2)/2),
                          anchor=float(np.sum(problem.residual[len(problem.b):]**2)/2)),
                      inner_stationarity_scaled=inner, seconds=time.monotonic()-begun)
        alternatives.append(record)
        solutions.append((converged, result.cost, problem.beta.copy(), problem.x.copy()))
        if checkpoint:
            checkpoint(record, problem.beta.copy(), problem.x.copy())
        if progress:
            progress(record)
    # Failed fits are checkpoint candidates only, never accepted models.
    viable = [i for i, s in enumerate(solutions) if s[0]]
    best = min(viable or range(len(solutions)), key=lambda i: solutions[i][1])
    model.beta = solutions[best][2]
    return model, solutions[best][3], dict(converged=solutions[best][0], selected_start=best,
        alternatives=alternatives, coefficient_design_rank_nominal=int(problem.design_rank),
        coefficient_count=model.size, nominal_design_singular_values=problem.design_singular.tolist(),
        column_scale=problem.column_scale.tolist(), prior_strength=prior_strength,
        prior_center=problem.beta0.tolist(), temporal_strength=0., anchor_scales=problem.anchor_scales.tolist(),
        weighting="equal_fixation_1/(K*n_k)", projected_stationarity_tolerance_physical=1e-3,
        inner_stationarity_tolerance_scaled=1e-7, objective="fixed_covariance_quadratic_variable_projection")
