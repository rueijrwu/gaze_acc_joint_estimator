"""Conditional fixed-scale three-channel model and exact free-scale control.

The fixed log-scale offset is a hypothetical constraint. Coefficient profiling
uses the same frozen-weight quadratic surrogate as the maintained solver.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator

from calibrate_profiled import KNOTS, ProfiledProblem, basis, normal_solve, interpolation


def joint_basis(theta, a, p, derivatives=True):
    """Legacy normalized geometry plus log-scale shape [1,theta/15,A**p]."""
    legacy, legacy_dt, legacy_da = basis(theta, a, p, derivatives=derivatives)
    n = len(theta)
    H = np.zeros((n, 3, 17))
    H[:, :2, :14] = legacy
    H[:, 2, 14:] = np.column_stack((np.ones(n), np.asarray(theta) / 15, a))
    if not derivatives:
        return H, None, None
    dt, da = np.zeros_like(H), np.zeros_like(H)
    dt[:, :2, :14], da[:, :2, :14] = legacy_dt, legacy_da
    dt[:, 2, 15], da[:, 2, 16] = 1 / 15, 1.
    return H, dt, da


class KnownScaleProblem(ProfiledProblem):
    """Three independent channels conditional on fixed per-frame log(lambda)=0.

    Shape coefficients, exponent, precision and the function-grid reference
    must be frozen from common training support by the caller. No physical
    claim that scale is known follows from this hypothesis.
    """

    def __init__(self, y, frames, groups, targets, demands, p, initial_coef, W,
                 prior_W=None, coefficient_map=None, coefficient_solver="irls"):
        if coefficient_solver not in ("irls", "newton"):
            raise ValueError("Coefficient solver must be irls or newton")
        self.coefficient_solver = coefficient_solver
        y = np.asarray(y, dtype=float)
        initial_coef = np.asarray(initial_coef, dtype=float)
        W = np.asarray(W, dtype=float)
        if y.ndim != 2 or y.shape[1] != 3 or not np.isfinite(y).all():
            raise ValueError("Known-scale observations must be finite [d,rho4,logS1]")
        if initial_coef.shape != (17,) or not np.isfinite(initial_coef).all():
            raise ValueError("Known-scale coefficients must have 17 finite slots")
        if W.shape != (3, 3) or not np.allclose(W, W.T):
            raise ValueError("Known-scale precision must be symmetric 3x3 SPD")
        np.linalg.cholesky(W)
        covariance = np.linalg.inv(W)
        prior_W = W.copy() if prior_W is None else np.asarray(prior_W, dtype=float)
        if prior_W.shape != (3, 3) or not np.allclose(prior_W, prior_W.T):
            raise ValueError("Known-scale function-prior precision must be symmetric 3x3 SPD")
        np.linalg.cholesky(prior_W)
        E = np.eye(17) if coefficient_map is None else np.asarray(coefficient_map, dtype=float)
        if (E.ndim != 2 or E.shape[0] != 17 or not 1 <= E.shape[1] <= 17
                or not np.isfinite(E).all() or np.linalg.matrix_rank(E) != E.shape[1]):
            raise ValueError("Joint coefficient map must be full-column-rank with 17 rows")
        gamma = np.linalg.lstsq(E, initial_coef, rcond=None)[0]
        if not np.allclose(E @ gamma, initial_coef, rtol=1e-12, atol=1e-12):
            raise ValueError("Initial joint coefficients violate coefficient map")
        # Reuse only dimension-independent state/anchor/temporal bookkeeping.
        # The parent's temporary prior is replaced by the three-channel prior.
        super().__init__(y[:, :2], frames, groups, targets, demands, p,
                         initial_coef[:14], np.linalg.inv(covariance[:2, :2]),
                         prior_W=np.linalg.inv(np.linalg.inv(prior_W)[:2, :2]))
        self.y, self.W = y.copy(), W.copy()
        self.L = np.linalg.cholesky(self.W).T
        self.prior_W = prior_W.copy()
        self.prior_L = np.linalg.cholesky(self.prior_W).T
        self.function_prior_channel_split_available = bool(np.allclose(self.prior_W, np.diag(np.diag(self.prior_W))))
        self.initial_coef = initial_coef.copy()
        self.coefficient_count, self.free_coefficient_count = 17, E.shape[1]
        self.coefficient_map = E.copy()
        self.coefficient_map.setflags(write=False)
        th, aa = np.meshgrid(np.linspace(-15, 15, 9), np.linspace(KNOTS[0], KNOTS[-1], 9))
        H = joint_basis(th.ravel(), aa.ravel()**p, p, derivatives=False)[0]
        self.prior_M = np.einsum("ab,nbk->nak", self.prior_L, H).reshape(-1, 17) * np.sqrt(.1 / len(H))
        self.prior_v = self.prior_M @ initial_coef
        self.function_prior_rows = len(self.prior_v)
        self.cache, self._geometry = None, None
        self.scale_hypothesis = "fixed log(lambda)=0; conditional hypothesis, not established physical fact"
        self.scale_covariance = covariance.copy()
        self.scale_covariance.setflags(write=False)
        self.scale_coordinate_order = ("d", "rho4", "logS1")

    def forward_matrix(self, x):
        if self._geometry is not None and np.array_equal(x, self._geometry["x"]):
            return self._geometry["H"]
        theta, a, _, _ = self.decode(x)
        return joint_basis(theta, a, self.p, derivatives=False)[0]

    def profile(self, x, omega):
        x, omega = np.asarray(x), np.asarray(omega)
        if self.cache is not None and np.array_equal(x, self.cache["x"]) and np.array_equal(omega, self.cache["omega"]):
            return self.cache
        if self._geometry is None or not np.array_equal(x, self._geometry["x"]):
            theta, a, _, dA = self.decode(x)
            H, dt, da = joint_basis(theta, a, self.p)
            self._geometry = dict(x=x.copy(), H=H, dt=dt, da=da, dA=dA)
        geometry = self._geometry
        H, dt, da, dA = (geometry[key] for key in ("H", "dt", "da", "dA"))
        scale = np.sqrt(self.alpha * omega)
        def whiten(B):
            return np.einsum("ab,nbk->nak", self.L, B) * scale[:, None, None]
        Mopt = whiten(H).reshape(-1, 17)
        M = np.vstack((Mopt, self.prior_M)) @ self.coefficient_map
        v = np.r_[(self.y @ self.L.T * scale[:, None]).ravel(), self.prior_v]
        Q, R = np.linalg.qr(M, mode="reduced")
        singular = np.linalg.svd(R, compute_uv=False)
        if singular[-1] <= singular[0] * 1e-12:
            raise ValueError("Rank deficient joint coefficient profile")
        gamma = np.linalg.solve(R, Q.T @ v)
        e = M @ gamma - v
        Dt = (whiten(dt) * 15) @ self.coefficient_map
        Da = (whiten(da) * self.a_scale) @ self.coefficient_map
        local = np.stack((Dt @ gamma, Da @ gamma), axis=2)
        eo = e[:3 * self.n].reshape(-1, 3)
        C = np.stack((np.einsum("nbk,nb->nk", Dt, eo),
                      np.einsum("nbk,nb->nk", Da, eo)), axis=1)
        self.cache = dict(x=x.copy(), omega=omega.copy(), coef=self.coefficient_map @ gamma,
                          M=M, e=e, local=local, C=C, dA=dA, factor=R,
                          condition=float(singular[0] / singular[-1]))
        return self.cache

    def jacobian(self, x, omega):
        d = self.profile(x, omega)
        M, e, B, C = (d[key] for key in ("M", "e", "local", "C"))
        m, sqrtN = len(e), np.sqrt(self.n)
        def mv(v):
            v = np.asarray(v).ravel()
            vv = v.reshape(-1, 2)
            q = np.r_[np.einsum("nbs,ns->nb", B, vv).ravel(), np.zeros(len(self.prior_v))]
            correction = normal_solve(d["factor"], M.T @ q + np.einsum("nsk,ns->k", C, vv))
            return np.r_[q - M @ correction, self.penalty_jv(v, d["dA"])] * sqrtN
        def rmv(w):
            w = np.asarray(w).ravel()
            z = w[:m]
            small = normal_solve(d["factor"], M.T @ z)
            projected = (z - M @ small)[:3 * self.n].reshape(-1, 3)
            grad = np.einsum("nbs,nb->ns", B, projected) - np.einsum("nsk,k->ns", C, small)
            return (grad.ravel() + self.penalty_jtv(w[m:], d["dA"])) * sqrtN
        return LinearOperator((m + len(self.penalties(x)), 2 * self.n),
                              matvec=mv, rmatvec=rmv, dtype=float)

    def robust_coefficients(self, x, kappa=None, maxiter=100, tol=1e-8):
        if self.coefficient_solver == "newton":
            return self.robust_coefficients_newton(x, kappa, maxiter, tol)
        return self.robust_coefficients_irls(x, kappa, maxiter, tol)

    def robust_coefficients_irls(self, x, kappa=None, maxiter=100, tol=1e-8):
        """IRLS surrogate profiling with final weights/coefficient consistency."""
        coef, omega, report = super().robust_coefficients(x, kappa, maxiter, tol)
        # The base nonconverged path can return coefficients from the preceding
        # weights; always make the returned pair a matched quadratic profile.
        coef = self.profile(x, omega)["coef"].copy()
        r = (self.forward_matrix(x) @ coef - self.y) @ self.L.T
        updated = np.ones(self.n) if kappa is None else 1 / np.sqrt(1 + np.sum(r * r, axis=1) / kappa**2)
        report = dict(report)
        report["final_weight_consistency"] = float(np.max(np.abs(updated - omega)))
        report["objective"] = self.true_objective(x, coef, kappa)[0]
        return coef, omega, report

    def robust_coefficients_newton(self, x, kappa=None, maxiter=100, tol=1e-8):
        """Explicit opt-in acceleration; the default IRLS method is unchanged."""
        if __package__:
            from .coefficient_newton import solve_coefficients_newton
        else:
            from coefficient_newton import solve_coefficients_newton
        return solve_coefficients_newton(self, x, kappa, maxiter, tol)

    def physical_envelope_gradients(self, x, coef, omega):
        """Original weighted-objective physical derivatives and knot sides.

        At A=0 with p<1, singular physical ratio/log derivatives are excluded
        from the finite physical summary and separately audited in a-space.
        Envelope use requires converged coefficient IRLS weights.
        """
        theta, a, A, chain = self.decode(x)
        H, dt, da = joint_basis(theta, a, self.p)
        force = ((H @ coef - self.y) @ self.W) * (self.alpha * omega)[:, None]
        grad = np.zeros((self.n, 2))
        grad[:, 0] = np.sum(force * (dt @ coef), axis=1)
        _, dw = interpolation(A, KNOTS)
        displacement_A = dw @ coef[:4] + theta * (dw @ coef[4:8])
        t = theta / 15
        ratio_a = coef[11] + coef[12] * t + coef[13] * t*t
        self.physical_derivative_undefined_mask = ((A == 0) & (self.p < 1) &
                                                   ((ratio_a != 0) | (coef[16] != 0)))
        self.physical_derivative_undefined_count = int(np.sum(self.physical_derivative_undefined_mask))
        multiplier = np.zeros(self.n)
        positive = A > 0
        multiplier[positive] = self.p * A[positive]**(self.p - 1)
        if self.p == 1:
            multiplier[~positive] = 1.
        grad[:, 1] = (force[:, 0] * displacement_A +
                      multiplier * (force[:, 1] * ratio_a + force[:, 2] * coef[16]))
        mt = np.bincount(self.groups, weights=theta) / self.counts
        ma = np.bincount(self.groups, weights=A) / self.counts
        grad[:, 0] += (mt - self.targets)[self.groups] / (self.J * self.counts[self.groups])
        grad[:, 1] += (ma - self.demands)[self.groups] / (.25**2 * self.J * self.counts[self.groups])
        temporal = np.column_stack((theta[self.right] - theta[self.left],
                                    (A[self.right] - A[self.left]) / .25**2)) * self.link_scale[:, None]**2
        np.add.at(grad, self.left, -temporal)
        np.add.at(grad, self.right, temporal)
        left_index = np.clip(np.searchsorted(KNOTS, A, side="left") - 1, 0, len(KNOTS) - 2)
        right_index = np.clip(np.searchsorted(KNOTS, A, side="right") - 1, 0, len(KNOTS) - 2)
        knot_values = coef[:4][None, :] + theta[:, None] * coef[4:8][None, :]
        def slope(index):
            return ((knot_values[np.arange(self.n), index + 1] - knot_values[np.arange(self.n), index]) /
                    (KNOTS[index + 1] - KNOTS[index]))
        minusA = grad[:, 1] + force[:, 0] * (slope(left_index) - displacement_A)
        plusA = grad[:, 1] + force[:, 0] * (slope(right_index) - displacement_A)
        distance = np.min(np.abs(A[:, None] - np.asarray(KNOTS)[None, 1:-1]), axis=1)
        exact = distance <= 16 * np.finfo(float).eps * np.maximum(1., np.abs(A))
        near = (distance <= 1e-6) & ~exact
        encodedzero = grad[:, 1] * chain
        if self.p < 1:
            zero = A == 0
            encodedzero[zero] = self.a_scale * (force[zero, 1] * ratio_a[zero] + force[zero, 2] * coef[16])
        return grad, minusA, plusA, exact, near, encodedzero

    def state_curvature_scale(self, x, omega):
        """Dimension-general diagonal frozen-surrogate preconditioner."""
        profiled = self.profile(x, omega)
        _, _, _, chain = self.decode(x)
        curvature = np.sum(profiled["local"]**2, axis=1) * self.n
        physical_chain = np.column_stack((np.full(self.n, 15.), chain / .25))
        curvature += physical_chain**2 * (self.n / (self.J * self.counts[self.groups]**2))[:, None]
        links = np.zeros(self.n)
        np.add.at(links, self.left, self.link_scale**2)
        np.add.at(links, self.right, self.link_scale**2)
        curvature += physical_chain**2 * (self.n * links)[:, None]
        return np.clip(1 / np.sqrt(np.maximum(curvature, 1e-16)), 1e-6, 1e6).ravel()

    def true_objective(self, x, coef, kappa=None):
        value, parts = super().true_objective(x, coef, kappa)
        prior = (self.prior_M @ coef - self.prior_v).reshape(-1, 3)
        if self.function_prior_channel_split_available:
            parts["normalized_function_prior"] = float(.5 * np.sum(prior[:, :2]**2))
            parts["log_shape_function_prior"] = float(.5 * np.sum(prior[:, 2]**2))
        return value, parts


def make_scale_problem(scale_hypothesis, y, frames, groups, targets, demands, p,
                       initial_coef, covariance, prior_W=None, coefficient_map=None,
                       scale_variance=0., coefficient_solver="irls"):
    """Construct fixed-scale model or mathematically identical normalized model.

    Finite uncertain scale inflates covariance[2,2] by an explicitly assumed
    log-scale variance. This declares a different elliptical radial robust
    measurement model; it is NOT profiling robust data loss plus a separate
    Gaussian scale prior. Its function prior must remain explicitly frozen.
    Free scale omits every third-shape coefficient and prior contribution. The
    precision is the inverse marginal covariance, never the upper-left block
    of the full precision matrix.
    """
    y = np.asarray(y, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (3, 3) or not np.allclose(covariance, covariance.T):
        raise ValueError("Scale comparison requires symmetric 3x3 covariance")
    np.linalg.cholesky(covariance)
    if scale_hypothesis in ("uncertain", "constrained", "uncertain_scale"):
        scale_variance = float(scale_variance)
        if np.isnan(scale_variance) or scale_variance < 0:
            raise ValueError("Assumed log-scale variance must be nonnegative")
        if np.isinf(scale_variance):
            scale_hypothesis = "free"
        else:
            if prior_W is None:
                raise ValueError("Uncertain-scale sensitivity requires explicit frozen prior_W")
            inflated = covariance.copy()
            inflated[2, 2] += scale_variance
            problem = KnownScaleProblem(y, frames, groups, targets, demands, p,
                                        initial_coef, np.linalg.inv(inflated), prior_W=prior_W,
                                        coefficient_map=coefficient_map, coefficient_solver=coefficient_solver)
            problem.scale_hypothesis = "assumed log-scale uncertainty in elliptical radial robust measurement covariance"
            problem.assumed_log_scale_variance = scale_variance
            problem.base_scale_covariance = covariance.copy()
            problem.base_scale_covariance.setflags(write=False)
            problem.scale_uncertainty_interpretation = (
                "Only covariance[logS1,logS1] is inflated; normalized marginal and function prior remain frozen. "
                "Not Gaussian-prior nuisance profiling and not an independently measured scale uncertainty.")
            return problem
    if scale_hypothesis in ("known", "fixed", "fixed_scale"):
        if float(scale_variance) != 0:
            raise ValueError("Fixed-scale hypothesis requires zero scale_variance; use uncertain for covariance inflation")
        return KnownScaleProblem(y, frames, groups, targets, demands, p, initial_coef,
                                 np.linalg.inv(covariance), prior_W=prior_W,
                                 coefficient_map=coefficient_map, coefficient_solver=coefficient_solver)
    if scale_hypothesis not in ("free", "free_scale"):
        raise ValueError("Scale hypothesis must be fixed/known, uncertain/constrained, or free")
    if coefficient_solver != "irls":
        raise ValueError("Exact free-scale dispatch retains the unchanged normalized IRLS solver")
    E = None
    if coefficient_map is not None:
        full_E = np.asarray(coefficient_map, dtype=float)
        if full_E.ndim != 2 or full_E.shape[0] != 17 or not np.isfinite(full_E).all():
            raise ValueError("Free-scale dispatch requires a finite 17-row joint coefficient map")
        normalized_columns = np.any(np.abs(full_E[:14]) > 0, axis=0)
        if np.any(np.abs(full_E[14:, normalized_columns]) > 0) or np.any(np.abs(full_E[:14, ~normalized_columns]) > 0):
            raise ValueError("Free-scale dispatch requires a block-diagonal normalized/log-shape map")
        E = full_E[:14, normalized_columns]
    prior_relative = None if prior_W is None else np.linalg.inv(np.linalg.inv(np.asarray(prior_W))[:2, :2])
    problem = ProfiledProblem(y[:, :2], frames, groups, targets, demands, p,
                              np.asarray(initial_coef)[:14], np.linalg.inv(covariance[:2, :2]),
                              prior_W=prior_relative, coefficient_map=E)
    problem.scale_hypothesis = "free per-frame scale: exact normalized marginal model; no log-shape prior"
    problem.scale_covariance = covariance.copy()
    problem.scale_covariance.setflags(write=False)
    problem.scale_coordinate_order = ("d", "rho4", "logS1")
    problem.scale_conditional_variance = float(covariance[2, 2] -
        covariance[2, :2] @ np.linalg.solve(covariance[:2, :2], covariance[:2, 2]))
    return problem
