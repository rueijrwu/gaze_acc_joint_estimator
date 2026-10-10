"""Constrained P1 initialization with complete analytic profile derivatives.

The empirical template/origin/length/axes and omega1 stay fixed. The initial
isotropic convention is alpha1+beta1=0: only trace-free anisotropy and keystone
are released. This is a declared reduced initialization, not physical recovery
of both independent optical scale coefficients. Revisit at G7 if supported.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares
from .optics import p1_reference
from .geometry import p1_scale


@dataclass(frozen=True)
class P1Model:
    b1: np.ndarray
    omega1: float
    k1: tuple = (0., 0., 0.)
    theta_bounds: tuple = (-20., 20.)
    denominator_min: float = 1e-8

    def __post_init__(self):
        b1 = np.array(self.b1, dtype=np.float64, copy=True)
        k1, bounds = tuple(float(x) for x in self.k1), tuple(float(x) for x in self.theta_bounds)
        if b1.shape != (3, 2) or not np.isfinite(b1).all():
            raise ValueError('finite (3,2) P1 reference required')
        if len(k1) != 3 or not np.isfinite(k1).all() or not np.isfinite(self.omega1):
            raise ValueError('finite P1 coefficients/offset required')
        if len(bounds) != 2 or bounds[0] >= bounds[1] or not np.isfinite(bounds).all():
            raise ValueError('invalid visual domain')
        if not np.isfinite(self.denominator_min) or self.denominator_min <= 0:
            raise ValueError('positive denominator margin required')
        b1.setflags(write=False)
        object.__setattr__(self, 'b1', b1)
        object.__setattr__(self, 'k1', k1)
        object.__setattr__(self, 'theta_bounds', bounds)


def reference_derivatives(theta_visual, model):
    """Return F, mu, edges, valid, plus theta/omega1/alpha/beta/gamma derivatives."""
    theta = np.asarray(theta_visual, dtype=np.float64)
    points, mean, edges, valid = p1_reference(theta, model)
    xi = theta-model.omega1
    alpha, beta, gamma = model.k1
    denominator = 1+gamma*xi[..., None]*model.b1[:, 1]
    with np.errstate(divide='ignore', invalid='ignore'):
        dtheta = (2*xi[..., None, None]*model.b1*np.array([alpha, beta]) / denominator[..., None]
                  - points*(gamma*model.b1[:, 1]/denominator)[..., None])
        dglobal = np.zeros(points.shape+(3,))
        dglobal[..., 0, 0] = xi[..., None]**2*model.b1[:, 0]/denominator
        dglobal[..., 1, 1] = xi[..., None]**2*model.b1[:, 1]/denominator
        dglobal[..., :, 2] = -points*(xi[..., None]*model.b1[:, 1]/denominator)[..., None]
    dtheta = np.where(valid[..., None, None], dtheta, np.nan)
    dglobal = np.where(valid[..., None, None, None], dglobal, np.nan)
    edge_theta = (dtheta[..., 1:, :]-dtheta[..., :1, :]).reshape(theta.shape+(4,))
    edge_global = (dglobal[..., 1:, :, :]-dglobal[..., :1, :, :]).reshape(theta.shape+(4, 3))
    derivatives = {'points_theta': dtheta, 'mean_theta': dtheta.mean(axis=-2), 'edges_theta': edge_theta,
                   'points_omega1': -dtheta, 'mean_omega1': -dtheta.mean(axis=-2), 'edges_omega1': -edge_theta,
                   'points_global': dglobal, 'edges_global': edge_global}
    return points, mean, edges, valid, derivatives


def profile_derivatives(reference_edges, observed_edges, edge_covariance, edge_derivatives):
    """g and derivatives through the P1 profile; final derivative axis = parameters."""
    a, e = np.broadcast_arrays(reference_edges, observed_edges)
    da = np.asarray(edge_derivatives)
    g, valid = p1_scale(a, e, edge_covariance)
    wa = np.linalg.solve(edge_covariance, a[..., None])[..., 0]
    we = np.linalg.solve(edge_covariance, e[..., None])[..., 0]
    energy = np.sum(a*wa, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        dg = np.einsum('...ip,...i->...p', da, we-2*g[..., None]*wa)/energy[..., None]
        prediction_derivative = a[..., :, None]*dg[..., None, :]+g[..., None, None]*da
    dg = np.where(valid[..., None], dg, np.nan)
    prediction_derivative = np.where(valid[..., None, None], prediction_derivative, np.nan)
    return g, valid, dg, prediction_derivative


def domain_margins(model):
    """Exact endpoint/extremum checks over visual-angle interval and fixed template."""
    lo, hi = np.asarray(model.theta_bounds)-model.omega1
    xi = np.array([lo, hi, np.clip(0., lo, hi)])
    alpha, beta, gamma = model.k1
    sx, sy = 1+alpha*xi**2, 1+beta*xi**2
    denominator = 1+gamma*xi[:, None]*model.b1[:, 1]
    return {'local_theta_bounds_deg': [float(lo), float(hi)], 'min_sx': float(sx.min()),
            'min_sy': float(sy.min()), 'min_denominator': float(denominator.min()),
            'valid_domain': bool(np.isfinite(sx).all() and np.isfinite(sy).all()
                                 and np.isfinite(denominator).all() and sx.min()>0 and sy.min()>0
                                 and denominator.min()>model.denominator_min)}


def model_from_scaled(template, omega1, scaled_parameters, theta_bounds=(-20., 20.)):
    radius = np.sqrt(np.mean(np.sum(np.asarray(template)**2, axis=1)))
    delta, keystone = scaled_parameters
    return P1Model(template, omega1, (delta/100, -delta/100, keystone/(10*radius)), theta_bounds)


def scaled_bounds(template, omega1, theta_bounds=(-20., 20.)):
    radius = float(np.sqrt(np.mean(np.sum(template**2, axis=1))))
    xmax = float(np.max(np.abs(np.asarray(theta_bounds)-omega1))/10)
    ymax = float(np.max(np.abs(template[:, 1]))/radius)
    upper = np.array([.5/max(xmax*xmax, 1e-12), .5/max(xmax*ymax, 1e-12)])
    return -upper, upper


class ProfileObjective:
    """Same P1 edge metric and equal-exposure weights for identity and K1."""
    def __init__(self, theta, observed_edges, exposure, covariance, template, omega1, theta_bounds=(-20., 20.)):
        self.theta, self.observed, self.exposure = np.asarray(theta), np.asarray(observed_edges), np.asarray(exposure)
        self.covariance, self.template, self.omega1, self.theta_bounds = covariance, template, omega1, theta_bounds
        if self.observed.shape != (len(self.theta), 4) or len(self.exposure) != len(self.theta):
            raise ValueError('invalid conditional P1 objective population')
        if not np.isfinite(self.theta).all() or not np.isfinite(self.observed).all():
            raise ValueError('nonfinite fit inputs')
        self.groups, inverse, counts = np.unique(self.exposure, return_inverse=True, return_counts=True)
        self.weights = 1/np.sqrt(len(self.groups)*counts[inverse])
        self.whitener = np.linalg.solve(np.linalg.cholesky(covariance), np.eye(4))
        self.radius = float(np.sqrt(np.mean(np.sum(template**2, axis=1))))
        self.weighted_observed = np.linalg.solve(covariance, self.observed.T).T
        self.cache = None

    def evaluate(self, scaled):
        if self.cache is not None and np.array_equal(scaled, self.cache[0]):
            return self.cache[1:]
        model = model_from_scaled(self.template, self.omega1, scaled, self.theta_bounds)
        _, _, edges, optical_valid, derivatives = reference_derivatives(self.theta, model)
        da = np.stack(((derivatives['edges_global'][..., 0]-derivatives['edges_global'][..., 1])/100,
                       derivatives['edges_global'][..., 2]/(10*self.radius)), axis=-1)
        # Fixed covariance solves are batched; no point/frame loop and no P4 input.
        wa = edges @ np.linalg.solve(self.covariance, np.eye(4)).T
        energy = np.sum(edges*wa, axis=1)
        g = np.sum(edges*self.weighted_observed, axis=1)/energy
        if not optical_valid.all() or not np.all(np.isfinite(g)&(g>0)):
            raise ValueError('inadmissible P1 profile proposal')
        dg = np.einsum('nip,ni->np', da, self.weighted_observed-2*g[:, None]*wa)/energy[:, None]
        raw_residual = self.observed-g[:, None]*edges
        raw_jacobian = -edges[:, :, None]*dg[:, None, :]-g[:, None, None]*da
        residual = (raw_residual @ self.whitener.T)*self.weights[:, None]
        jacobian = np.einsum('ij,njp->nip', self.whitener, raw_jacobian)*self.weights[:, None, None]
        self.cache = (np.array(scaled, copy=True), residual.ravel(), jacobian.reshape(-1, 2))
        return self.cache[1:]

    def fun(self, scaled):
        return self.evaluate(scaled)[0]

    def jac(self, scaled):
        return self.evaluate(scaled)[1]


def fit_p1(objective, starts, max_evaluations=100):
    lower, upper = scaled_bounds(objective.template, objective.omega1, objective.theta_bounds)
    outcomes = []
    for start in starts:
        start = np.asarray(start, dtype=float)
        if np.any(start<=lower) or np.any(start>=upper):
            raise ValueError('start outside declared P1 parameter bounds')
        result = least_squares(objective.fun, start, jac=objective.jac, bounds=(lower, upper),
                               method='trf', ftol=1e-10, xtol=1e-10, gtol=1e-8,
                               max_nfev=max_evaluations)
        residual, jacobian = objective.evaluate(result.x)
        gradient = jacobian.T@residual
        # Gradient in parameter-range units, projected only for outward bound motion.
        scaled_gradient = gradient*(upper-lower)
        near_lo = result.x-lower<=1e-8*(upper-lower)
        near_hi = upper-result.x<=1e-8*(upper-lower)
        projected = np.where((near_lo&(gradient>0))|(near_hi&(gradient<0)), 0., scaled_gradient)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        rank = int(np.linalg.matrix_rank(jacobian, tol=singular[0]*1e-10))
        margins = domain_margins(model_from_scaled(objective.template, objective.omega1, result.x, objective.theta_bounds))
        certified = bool(result.success and rank==2 and np.linalg.norm(projected, np.inf)<=1e-6 and margins['valid_domain'])
        outcomes.append({'start': start.tolist(), 'scaled_parameters': result.x.tolist(), 'cost': float(result.cost),
                         'optimizer_success': bool(result.success), 'status': int(result.status), 'message': str(result.message),
                         'nfev': result.nfev, 'njev': result.njev, 'active_mask': result.active_mask.tolist(),
                         'scaled_gradient': scaled_gradient.tolist(), 'scaled_projected_gradient_inf': float(np.linalg.norm(projected, np.inf)),
                         'jacobian_singular_values': singular.tolist(), 'conditional_rank': rank,
                         'gauss_newton_eigenvalues': np.linalg.eigvalsh(jacobian.T@jacobian).tolist(),
                         'domain': margins, 'conditional_fit_certified': certified})
    eligible = [outcome for outcome in outcomes if outcome['conditional_fit_certified']]
    best = min(eligible or outcomes, key=lambda outcome: outcome['cost'])
    return model_from_scaled(objective.template, objective.omega1, best['scaled_parameters'], objective.theta_bounds), best, outcomes


def evaluate_p1(theta, observed_edges, covariance, model):
    theta = np.asarray(theta, dtype=np.float64)
    points, mean, edges, optical_valid, derivatives = reference_derivatives(theta, model)
    g, scale_valid, dg, d_prediction = profile_derivatives(edges, observed_edges, covariance,
                                                         derivatives['edges_theta'][..., None])
    within = np.isfinite(theta)&(theta>=model.theta_bounds[0])&(theta<=model.theta_bounds[1])
    valid = optical_valid&scale_valid&within
    prediction = g[..., None]*edges
    residual = np.asarray(observed_edges)-prediction
    return {'g': np.where(valid, g, np.nan), 'valid': valid, 'reference_edges': edges,
            'edge_prediction': np.where(valid[..., None], prediction, np.nan),
            'edge_residual': np.where(valid[..., None], residual, np.nan), 'F1': points, 'mu1': mean,
            'dg_dtheta': np.where(valid, dg[..., 0], np.nan), 'dg_domega1': np.where(valid, -dg[..., 0], np.nan),
            'd_edge_prediction_dtheta': np.where(valid[..., None], d_prediction[..., 0], np.nan),
            'd_mu1_dtheta': derivatives['mean_theta']}
