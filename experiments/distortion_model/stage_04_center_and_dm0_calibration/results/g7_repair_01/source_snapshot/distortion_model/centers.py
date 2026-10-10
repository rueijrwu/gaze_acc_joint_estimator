"""Forward visual-gaze center law and its exact full-relative linear block.

Optical means occur once; corrected centers are derived diagnostics. The
constant covariance is the frozen raw metric, including all cross-terms.
"""
from dataclasses import dataclass
from math import comb
import numpy as np


@dataclass(frozen=True)
class CenterPolynomial:
    coefficients: np.ndarray
    aref: float
    degree: int = 2

    def __post_init__(self):
        c = np.array(self.coefficients, dtype=float, copy=True)
        if self.degree not in (1, 2, 3) or c.shape != (self.degree + 3, 2):
            raise ValueError('basis order is [1,a,t,a*t,t^2,t^3], with declared degree')
        if not np.isfinite(c).all() or not np.isfinite(self.aref):
            raise ValueError('finite center coefficients and A reference required')
        c.setflags(write=False)
        object.__setattr__(self, 'coefficients', c)


def center_basis(theta, accommodation, aref, degree=2):
    if degree not in (1, 2, 3):
        raise ValueError('declared degree must be 1, 2 or 3')
    theta, accommodation = np.broadcast_arrays(theta, accommodation)
    t, a = theta / 10., accommodation - aref
    return np.stack([np.ones_like(t), a, t, a*t] + [t**j for j in range(2, degree+1)], axis=-1)


def center_prediction(theta, accommodation, model):
    return center_basis(theta, accommodation, model.aref, model.degree) @ model.coefficients


def center_derivatives(theta, accommodation, model):
    theta, accommodation = np.broadcast_arrays(theta, accommodation)
    t, a = theta / 10., accommodation - model.aref
    c = model.coefficients
    dtheta = np.broadcast_to((c[2] + a[..., None]*c[3])/10., theta.shape+(2,)).copy()
    for j in range(2, model.degree+1):
        dtheta += j*t[..., None]**(j-1)*c[j+2]/10.
    return dtheta, c[1] + t[..., None]*c[3]


def local_coefficients(model, omega):
    """Exact visual t=t_local+omega/10 conversion, retaining A terms."""
    w = omega/10.
    result = np.zeros_like(model.coefficients)
    visual = [model.coefficients[0], model.coefficients[2]] + list(model.coefficients[4:])
    shifted = [sum(comb(n, m)*visual[n]*w**(n-m) for n in range(m, model.degree+1))
               for m in range(model.degree+1)]
    result[0], result[2] = shifted[:2]
    result[1] = model.coefficients[1] + w*model.coefficients[3]
    result[3] = model.coefficients[3]
    result[4:] = shifted[2:]
    return result


def relative_prediction(g, f1, f4, separation):
    f1, f4, g, separation = map(np.asarray, (f1, f4, g, separation))
    mu1 = f1.mean(axis=-2)
    e1 = (f1[..., 1:, :] - f1[..., :1, :]).reshape(f1.shape[:-2]+(4,))
    q = separation[..., None, :] + f4 - mu1[..., None, :]
    return g[..., None]*np.concatenate((e1, q.reshape(q.shape[:-2]+(6,))), axis=-1)


def corrected_centers(p1, p4, g, f1, f4):
    c1, c4 = np.mean(p1, axis=-2), np.mean(p4, axis=-2)
    mu1, mu4 = np.mean(f1, axis=-2), np.mean(f4, axis=-2)
    return {'raw_centroid_separation_px': c4-c1, 'mu4_minus_mu1_reference_px': mu4-mu1,
            'C1_hat_px': c1-g[..., None]*mu1, 'C4_hat_px': c4-g[..., None]*mu4,
            'D_obs_reference_px': (c4-c1)/g[..., None]-mu4+mu1}


class CenterBlock:
    """Ten shared coefficients, no dense per-frame global Jacobian.

The Hessian accumulates a basis Gram matrix and a 2x2 spatial metric.
Full point residuals determine the right hand side, including P1/P4
correlations. Chunks retain globally computed exposure weights.
"""
    def __init__(self, theta, accommodation, g, f1, f4, observed, exposure,
                 covariance, aref, degree=2, curvature_scale_px=190., chunk_size=16384):
        self.theta = np.asarray(theta, dtype=float)
        self.accommodation = np.asarray(accommodation, dtype=float)
        self.g = np.asarray(g, dtype=float)
        self.f1, self.f4 = np.asarray(f1), np.asarray(f4)
        self.observed, self.exposure = np.asarray(observed), np.asarray(exposure)
        n = len(self.theta)
        if (self.theta.shape != (n,) or self.accommodation.shape != (n,) or self.g.shape != (n,)
                or self.f1.shape != (n,3,2) or self.f4.shape != (n,3,2)
                or self.observed.shape != (n,10) or self.exposure.shape != (n,) or n == 0):
            raise ValueError('invalid complete relative population')
        if not all(np.isfinite(x).all() for x in (self.theta,self.accommodation,self.g,self.f1,self.f4,self.observed)) or np.any(self.g<=0):
            raise ValueError('finite states/optics and positive P1 scale required')
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (10,10) or not np.allclose(covariance,covariance.T,rtol=1e-12,atol=1e-12):
            raise ValueError('symmetric full ten-coordinate covariance required')
        np.linalg.cholesky(covariance)
        self.precision = np.linalg.solve(covariance, np.eye(10))
        self.aref, self.degree = float(aref), degree
        self.basis = center_basis(self.theta, self.accommodation, aref, degree)
        _, self.groups, counts = np.unique(self.exposure, return_inverse=True, return_counts=True)
        self.weights = 1/(len(counts)*counts[self.groups])
        self.insertion = np.vstack((np.zeros((4,2)), np.tile(np.eye(2),(3,1))))
        self.spatial_metric = self.insertion.T @ self.precision @ self.insertion
        self.base = relative_prediction(self.g,self.f1,self.f4,np.zeros((n,2)))
        self.scale = float(curvature_scale_px)
        self.chunk_size = int(chunk_size)
        if not np.isfinite(self.scale) or self.scale<=0 or self.chunk_size<1:
            raise ValueError('positive parameter scale and chunk size required')
        k = self.basis.shape[1]
        gram, rhs = np.zeros((k,k)), np.zeros((k,2))
        for start in range(0,n,self.chunk_size):
            sl = slice(start,start+self.chunk_size)
            phi, w, g = self.basis[sl], self.weights[sl], self.g[sl]
            gram += phi.T @ ((w*g*g)[:,None]*phi)
            residual = self.observed[sl]-self.base[sl]
            rhs += phi.T @ ((w*g)[:,None]*(residual @ self.precision @ self.insertion))
        self.gram = (gram+gram.T)/2
        self.data_hessian = np.kron(self.gram, self.spatial_metric)
        self.prior = np.zeros(k*2)
        self.prior[8:] = 1/self.scale**2
        self.hessian = self.data_hessian + np.diag(self.prior)
        self.rhs = rhs.ravel()

    def model(self, coefficients):
        return CenterPolynomial(np.asarray(coefficients).reshape(-1,2), self.aref, self.degree)

    def terms(self, coefficients, nominal, demand, theta_scale=.10, a_scale=.25):
        model = self.model(coefficients)
        d = center_prediction(self.theta,self.accommodation,model)
        prediction = relative_prediction(self.g,self.f1,self.f4,d)
        residual = self.observed-prediction
        point = .5*float(np.sum(self.weights*np.einsum('ni,ij,nj->n',residual,self.precision,residual)))
        count = np.bincount(self.groups)
        means = lambda x: np.bincount(self.groups,weights=x)/count
        theta_deviation = means(self.theta)-means(np.asarray(nominal))
        a_deviation = means(self.accommodation)-means(np.asarray(demand))
        theta_anchor = .5*float(np.mean((theta_deviation/theta_scale)**2))
        a_anchor = .5*float(np.mean((a_deviation/a_scale)**2))
        c = model.coefficients.ravel()
        regularization = .5*float(np.sum(self.prior*c*c))
        return {'total': point+theta_anchor+a_anchor+regularization, 'point': point,
                'theta_mean_anchor': theta_anchor, 'A_mean_anchor': a_anchor,
                'regularization': regularization, 'temporal': 0.,
                'theta_mean_deviation_deg': theta_deviation.tolist(), 'A_mean_deviation_D': a_deviation.tolist()}, prediction, residual

    def solve(self):
        """Equilibrated full-rank SPD solve; unsupported data terms fail closed."""
        n = len(self.rhs)
        scale = 1/np.sqrt(np.diag(self.data_hessian))
        normalized = self.data_hessian*scale[:,None]*scale[None,:]
        eigen = np.linalg.eigvalsh(normalized)
        rank = int(np.sum(eigen>eigen.max()*1e-10))
        if rank != n:
            raise ValueError('unsupported center terms: declare a reduced/shrunk policy before handoff')
        h = self.hessian*scale[:,None]*scale[None,:]
        c = scale*np.linalg.solve(h,scale*self.rhs)
        gradient = self.hessian@c-self.rhs
        norm = float(np.linalg.norm(gradient*self.scale,np.inf))
        constant = np.zeros(n)
        constant[:2] = np.linalg.solve(self.hessian[:2,:2],self.rhs[:2])
        return self.model(c), self.model(constant), {
            'data_rank': rank, 'parameter_count': n,
            'normalized_data_curvature_eigenvalues': eigen.tolist(),
            'normalized_data_condition_number': float(eigen.max()/eigen.min()),
            'normalized_penalized_curvature_eigenvalues': np.linalg.eigvalsh(h).tolist(),
            'scaled_gradient_inf': norm, 'gradient_scale_reference_px': self.scale,
            'certificate_gradient_inf_threshold': 1e-6,
            'linear_block_certified': bool(norm<=1e-6),
            'before_initializer': 'optimal constant D in the same complete point criterion; other degree-2 coefficients zero',
            'coefficient_order': ['b0','bA','s0','sA']+[f'c{j}' for j in range(2,self.degree+1)],
            'coefficients_reference_px': c.reshape(-1,2).tolist()}
