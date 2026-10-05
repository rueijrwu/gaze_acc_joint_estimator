"""Optional exact pseudo-Huber coefficient Newton solve at frozen states.

The production IRLS default is unchanged. This path uses the same objective,
scales coefficient coordinates, uses a positive Hessian factorization and
certifies the existing successive-weight stopping contract at requested tol.
"""

from __future__ import annotations

import numpy as np

from calibrate_profiled import normal_solve


def pseudo_huber_value_gradient_hessian(z, design, target, prior_matrix,
                                       prior_target, alpha, kappa,
                                       return_factor=False):
    """Exact derivatives in supplied coefficient coordinates.

    design is (frames,channels,coefficients), already precision-whitened.
    The data term is sum(alpha*kappa²*(sqrt(1+||r||²/kappa²)-1)).
    The quadratic function prior has the usual factor 1/2.
    """
    z = np.asarray(z, dtype=float)
    residual = np.einsum("nbi,i->nb", design, z) - target
    squared = np.sum(residual*residual, axis=1)
    prior_residual = prior_matrix @ z - prior_target
    if kappa is None:
        weights = np.ones(len(alpha))
        value = .5*np.dot(alpha, squared) + .5*np.dot(prior_residual, prior_residual)
        factor = np.vstack((design.reshape(-1, len(z))*np.repeat(np.sqrt(alpha), design.shape[1])[:, None], prior_matrix))
    else:
        root = np.sqrt(1+squared/kappa**2)
        weights = 1/root
        # Algebraically identical loss, avoiding cancellation for small r.
        value = np.sum(alpha*squared/(root+1)) + .5*np.dot(prior_residual, prior_residual)
        norm = np.sqrt(squared)
        direction = np.divide(residual, norm[:, None], out=np.zeros_like(residual), where=norm[:, None] > 0)
        parallel = np.einsum("nbi,nb->ni", design, direction)
        orthogonal = design - direction[:, :, None]*parallel[:, None, :]
        # Radial Hessian eigenvalues are omega (orthogonal) and omega³
        # (parallel). Build a positive factor rather than subtract Gram terms.
        orthogonal_factor = (orthogonal*np.sqrt(alpha*weights)[:, None, None]).reshape(-1, len(z))
        parallel_factor = parallel*np.sqrt(alpha*weights**3)[:, None]
        factor = np.vstack((orthogonal_factor, parallel_factor, prior_matrix))
    gradient = np.einsum("nbi,nb,n->i", design, residual, alpha*weights) + prior_matrix.T @ prior_residual
    hessian = factor.T @ factor
    if return_factor:
        return float(value), gradient, hessian, factor
    return float(value), gradient, hessian


def _weights(problem, x, coef, kappa):
    residual = (problem.forward_matrix(x) @ coef - problem.y) @ problem.L.T
    return 1/np.sqrt(1+np.sum(residual*residual, axis=1)/kappa**2)


def solve_coefficients_newton(problem, x, kappa=None, maxiter=100, tol=1e-8):
    """Opt-in small Newton solve, with unchanged IRLS fallback.

    Successful returns match existing IRLS: successive weight update < tol,
    followed by final QR at the updated weights. Final reprofile consistency
    is independently reported, not imposed as a stronger stopping criterion.
    Fallback preserves the unchanged existing IRLS convergence semantics.
    """
    if not np.isfinite(tol) or tol <= 0 or maxiter < 1:
        raise ValueError("Invalid optional coefficient solver settings")
    if kappa is None:
        coef, omega, report = problem.robust_coefficients_irls(x, None, maxiter, tol)
        report = dict(report, coefficient_solver="existing_gaussian_qr", strict_certificate_passed=True)
        return coef, omega, report
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("Pseudo-Huber kappa must be positive")
    evaluations, qr_certifications, newton_iterations = 0, 0, 0
    fallback_reason = None
    def fallback(reason):
        coef, omega, report = problem.robust_coefficients_irls(x, kappa, maxiter, tol)
        consistency = float(np.max(np.abs(_weights(problem, x, coef, kappa)-omega)))
        report = dict(report)
        report.update(coefficient_solver="newton_with_existing_irls_fallback",
                      fallback_reason=reason, newton_iterations=newton_iterations,
                      newton_objective_evaluations=evaluations, qr_certifications=qr_certifications,
                      final_weight_consistency=consistency,
                      strict_certificate_passed=bool(report["converged"] and report["weight_change"] < tol),
                      certificate_contract="successive weight update < tol; final QR consistency reported separately",
                      irls_reported_converged=bool(report["converged"]))
        return coef, omega, report
    try:
        E = np.asarray(problem.coefficient_map, dtype=float)
        H = problem.forward_matrix(x)
        design = np.einsum("ab,nbk->nak", problem.L, H) @ E
        prior = problem.prior_M @ E
        # Center at the frozen initial function to avoid repeatedly subtracting
        # large log-scale intercepts in the inner numerical model.
        reference_gamma = np.linalg.lstsq(E, problem.initial_coef, rcond=None)[0]
        reference_coef = E @ reference_gamma
        target = (problem.y - H @ reference_coef) @ problem.L.T
        prior_target = problem.prior_v - problem.prior_M @ reference_coef
        column_norm = np.sqrt(np.einsum("nbi,nbi,n->i", design, design, problem.alpha) + np.sum(prior*prior, axis=0))
        if np.any(~np.isfinite(column_norm)) or np.any(column_norm <= 0):
            return fallback("Invalid coefficient column scaling")
        design = design/column_norm
        prior = prior/column_norm
        optical = (design*np.sqrt(problem.alpha)[:, None, None]).reshape(-1, E.shape[1])
        M = np.vstack((optical, prior))
        rhs = np.r_[(target*np.sqrt(problem.alpha)[:, None]).ravel(), prior_target]
        Q, R = np.linalg.qr(M, mode="reduced")
        singular = np.linalg.svd(R, compute_uv=False)
        if singular[-1] <= singular[0]*1e-12:
            return fallback("Rank deficient scaled Gaussian initialization")
        z = np.linalg.solve(R, Q.T @ rhs)
        def value(candidate):
            nonlocal evaluations
            evaluations += 1
            residual = np.einsum("nbi,i->nb", design, candidate)-target
            squared = np.sum(residual*residual, axis=1)
            prior_residual = prior @ candidate-prior_target
            return float(np.sum(problem.alpha*squared/(np.sqrt(1+squared/kappa**2)+1)) + .5*np.dot(prior_residual, prior_residual))
        current_value = value(z)
        for previous in (getattr(problem, "_previous_newton_coef", None),
                         None if problem.cache is None else problem.cache.get("coef")):
            if previous is not None and np.shape(previous) == np.shape(reference_coef):
                gamma = np.linalg.lstsq(E, np.asarray(previous)-reference_coef, rcond=None)[0]
                candidate = gamma*column_norm
                candidate_value = value(candidate)
                if candidate_value < current_value:
                    z, current_value = candidate, candidate_value
        small_steps = 0
        for iteration in range(maxiter):
            newton_iterations = iteration+1
            current_value, gradient, hessian, factor = pseudo_huber_value_gradient_hessian(
                z, design, target, prior, prior_target, problem.alpha, kappa, return_factor=True)
            evaluations += 1
            # Certification uses original forward arithmetic and existing QR,
            # independently of the centered/scaled inner representation.
            coef_candidate = reference_coef + E @ (z/column_norm)
            omega = _weights(problem, x, coef_candidate, kappa)
            if np.linalg.norm(gradient, ord=np.inf) <= 1e-7*max(1., current_value) or iteration == maxiter-1:
                qr_certifications += 1
                coefficient_profile = problem.profile(x, omega)["coef"].copy()
                updated = _weights(problem, x, coefficient_profile, kappa)
                consistency = float(np.max(np.abs(updated-omega)))
                if consistency < tol:
                    # Mirror the base IRLS final reprofile exactly: coefficients
                    # at the newly updated weights, then separate consistency.
                    coefficient_profile = problem.profile(x, updated)["coef"].copy()
                    final_consistency = float(np.max(np.abs(_weights(problem, x, coefficient_profile, kappa)-updated)))
                    problem._previous_newton_coef = coefficient_profile.copy()
                    objective = problem.true_objective(x, coefficient_profile, kappa)[0]
                    return coefficient_profile, updated, dict(
                        converged=True, iterations=newton_iterations,
                        weight_change=consistency, final_weight_consistency=final_consistency,
                        objective=objective, coefficient_solver="scaled_exact_newton",
                        strict_certificate_passed=True, qr_certifications=qr_certifications,
                        certificate_contract="successive weight update < tol; final QR consistency reported separately",
                        newton_objective_evaluations=evaluations,
                        scaled_coefficient_gradient_inf=float(np.linalg.norm(gradient, ord=np.inf)))
            # Scaled SPD solve; ill-conditioned normal Hessians use their
            # explicit positive QR factor instead, without changing objective.
            if np.linalg.cond(hessian) > 1e10:
                R_hessian = np.linalg.qr(factor, mode="r")
                step = -normal_solve(R_hessian, gradient)
            else:
                chol = np.linalg.cholesky(hessian)
                step = -np.linalg.solve(chol.T, np.linalg.solve(chol, gradient))
            directional = float(gradient @ step)
            if not np.all(np.isfinite(step)) or directional >= 0:
                fallback_reason = "Non-descent or nonfinite Newton direction"
                break
            length = 1.
            accepted = False
            for _ in range(40):
                trial = z+length*step
                trial_value = value(trial)
                if np.isfinite(trial_value) and trial_value <= current_value+1e-4*length*directional:
                    accepted = True
                    break
                length *= .5
            if not accepted:
                fallback_reason = "Armijo line search did not decrease the exact coefficient objective"
                break
            delta = length*step
            small_steps = small_steps+1 if np.linalg.norm(delta, ord=np.inf) <= 1e-13*max(1., np.linalg.norm(z, ord=np.inf)) else 0
            z, current_value = trial, trial_value
            if small_steps >= 3:
                fallback_reason = "Numerical stagnation before strict QR/weight certification"
                break
        return fallback(fallback_reason or "Newton iteration limit before strict certification")
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return fallback("Guarded Newton failure: " + str(exc))
