"""P4 shared raw keystone with a free global offset and capture barrel terms.

The all-row native-pixel model is g_i * center(K(B*(1+k_c*|B|^2),
(theta_i-theta04)/10)). g is supplied, fixed, and never profiled. There are no
accommodation states, priors, output normalization, or additional scales.

B is a learned centered triangle with four free coordinates, including size.
Its centroid is a coordinate-origin convention, not an estimated optical center.
k_capture1=0 anchors the radial reference; R0 is frozen from the initializer.
Known g and the anchor capture determine B's native size: fixing B energy would
artificially constrain this comparison. Search envelopes are numerical only.
"""
from __future__ import annotations

import time
import numpy as np
from .shared_keystone import inverse_radial

DELTA = 1e-8
GAUGE = ("centroid(B4)=0 fixes the coordinate origin; k_capture1=0 fixes the radial "
         "reference; R0 is fixed from the initializer; fixed supplied g anchors "
         "native size, with no B energy or mean-log-scale constraint")


def _basis():
    h = np.asarray([[1/np.sqrt(2), 1/np.sqrt(6)],
                    [-1/np.sqrt(2), 1/np.sqrt(6)], [0., -2/np.sqrt(6)]])
    basis = np.zeros((3, 2, 4), dtype=np.float64)
    basis[:, 0, :2] = h
    basis[:, 1, 2:] = h
    return basis


class ScaleComparisonProblem:
    """Identical model/objective for the two branches; only fixed g differs.

    X must contain every eligible centered labelled triangle, theta the frozen
    horizontal gaze, and capture/fixation row identities. Each capture receives
    equal weight, then each represented fixation within that capture receives
    equal weight. Three vertex squared Euclidean errors are averaged.

    Parameter order: b,cx,cy,theta04, four dimensionless centered-template
    coordinates, then beta_c=k_c*R0**2 for captures after capture 1. Template
    changes in pixels are R0*basis*q. A checkpoint callback receives a compact
    fit dictionary after each completed start. No frame arrays are checkpointed.
    """

    def __init__(self, X, theta, capture, fixation, reference, g=None, R0=None,
                 bounds=None, chunk_size=8192):
        self.X = np.asarray(X, dtype=np.float64)
        self.theta = np.asarray(theta, dtype=np.float64)
        self.capture = np.asarray(capture)
        self.fixation = np.asarray(fixation)
        self.reference = np.asarray(reference, dtype=np.float64).copy()
        n = len(self.theta)
        if n == 0 or self.theta.shape != (n,) or self.X.shape != (n, 3, 2):
            raise ValueError('Expected nonempty row-aligned X (N,3,2), theta (N,)')
        if self.capture.shape != (n,) or self.fixation.shape != (n,):
            raise ValueError('Capture and fixation identities must be row-aligned (N,)')
        if self.reference.shape != (3, 2) or not np.isfinite(self.reference).all():
            raise ValueError('Reference must be a finite (3,2) native-pixel triangle')
        if not np.isfinite(self.X).all() or not np.isfinite(self.theta).all():
            raise ValueError('All eligible observations and saved gaze must be finite')
        if not np.issubdtype(self.capture.dtype, np.number) or not np.isfinite(self.capture).all():
            raise ValueError('Finite integer capture identities are required')
        if not np.all(self.capture == np.round(self.capture)):
            raise ValueError('Integer capture identities are required')
        self.capture = self.capture.astype(np.int64)
        self.captures, self.capture_index = np.unique(self.capture, return_inverse=True)
        if self.captures[0] != 1:
            raise ValueError('Capture 1 must be present to anchor k1=0')
        for name, arr in (('X', self.X), ('reference', self.reference[None])):
            tolerance = 1e-10*max(1., float(np.max(np.linalg.norm(arr, axis=(1, 2)))))
            if np.max(np.abs(arr.mean(axis=1))) > tolerance:
                raise ValueError(f'{name} must be centered; no frame center nuisance is fitted')
        singular = np.linalg.svd(self.reference, compute_uv=False)
        if singular[-1] <= 1e-10*singular[0]:
            raise ValueError('The reference triangle must be noncollinear')
        self.R0 = float(np.sqrt(np.mean(np.sum(self.reference**2, axis=1))) if R0 is None else R0)
        if not np.isfinite(self.R0) or self.R0 <= 0:
            raise ValueError('Frozen R0 must be positive and finite')
        self.g = np.ones(n) if g is None else np.asarray(g, dtype=np.float64).copy()
        if self.g.shape != (n,) or not np.all(np.isfinite(self.g) & (self.g > 0)):
            raise ValueError('Fixed g must be positive, finite, and row-aligned')
        self.basis = _basis()
        self.n, self.nc = n, len(self.captures)
        self.np = 8+self.nc-1
        self.names = ('b', 'cx', 'cy', 'theta04')+tuple(f'template_q{i+1}' for i in range(4))+tuple(f'beta_capture{c}' for c in self.captures[1:])
        self.scales = np.r_[.05, .05, .05, 10., np.full(4, .1), np.full(self.nc-1, .1)]
        default = np.asarray([[-.5, .5], [-1., 1.], [-1., 1.], [-60., 60.]]+
                             [[-3., 3.]]*4+[[-4., 4.]]*(self.nc-1), dtype=np.float64)
        self.bounds = default if bounds is None else np.asarray(bounds, dtype=np.float64).copy()
        if self.bounds.shape != (self.np, 2) or not np.isfinite(self.bounds).all() or np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError(f'Bounds must be finite increasing ({self.np},2) numerical envelopes')
        self.weights = np.empty(n, dtype=np.float64)
        self.weight_groups = []
        for capture_id in self.captures:
            rows = np.flatnonzero(self.capture == capture_id)
            labels, inv, count = np.unique(self.fixation[rows], return_inverse=True, return_counts=True)
            self.weights[rows] = 1./(self.nc*len(labels)*count[inv]*3.)
            for label, size in zip(labels, count):
                self.weight_groups.append(dict(capture=int(capture_id), fixation=label.item() if hasattr(label, 'item') else label, rows=int(size), group_weight=float(1./(self.nc*len(labels)))))
        self.chunk_size = max(1, int(chunk_size))
        self.rejected_domain_evaluations = 0

    def unpack(self, params):
        params = np.asarray(params, dtype=np.float64)
        if params.shape != (self.np,) or not np.isfinite(params).all():
            raise ValueError(f'Expected {self.np} finite parameters ordered {self.names}')
        B = self.reference+self.R0*np.einsum('vdp,p->vd', self.basis, params[4:8])
        k = np.r_[0., params[8:]/self.R0**2]
        return B, params[:3].copy(), float(params[3]), k

    def _terms(self, params, rows, derivatives=False):
        B, alpha, theta0, k_all = self.unpack(params)
        theta = self.theta[rows]
        cid = self.capture_index[rows]
        g = self.g[rows]
        phi = (theta-theta0)/10.
        k = k_all[cid]
        r2 = np.sum(B*B, axis=1)
        radial = 1.+k[:, None]*r2[None]
        radial_derivative = 1.+3*k[:, None]*r2[None]
        Q = radial[..., None]*B[None]
        slope = np.einsum('nvd,d->nv', Q, alpha[1:])/self.R0
        den = 1.+phi[:, None]*slope
        exponent = alpha[0]*phi**2
        sign = np.asarray([1., -1.])
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            stretch = np.exp(exponent[:, None]*sign[None])
            F = stretch[:, None, :]*Q/den[..., None]
            mu = F.mean(axis=1)
            H = F-mu[:, None, :]
            prediction = g[:, None, None]*H
        # Stretch and triangle rank envelopes protect floating-point arithmetic;
        # their values are numerical choices, with contacts reported explicitly.
        sB = np.linalg.svd(B, compute_uv=False)
        template_valid = sB[-1] > 1e-8*self.R0
        valid = (np.isfinite(prediction).all(axis=(1, 2)) &
                 np.all(den > DELTA, axis=1) & np.all(radial > DELTA, axis=1) &
                 np.all(radial_derivative > DELTA, axis=1) &
                 (np.abs(exponent) < 20.) & template_valid)
        result = dict(prediction=prediction, H=H, Z=F, mu=mu, denominator=den,
                      phi=phi, radial_factor=radial, radial_derivative=radial_derivative,
                      exponent=exponent, valid=valid, B=B, alpha=alpha, theta04=theta0,
                      kappa=k_all, template_min_singular=float(sB[-1]))
        if not derivatives:
            return result
        n = len(theta)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            # dF/dQ includes the Q dependence of the raw projective denominator.
            D = stretch[:, None, :, None]*np.eye(2)[None, None]/den[:, :, None, None]
            D -= F[..., None]*(phi[:, None, None, None]*alpha[None, None, None, 1:]/self.R0)/den[:, :, None, None]
            J = np.zeros((n, 3, 2, self.np), dtype=np.float64)
            J[..., 0] = F*phi[:, None, None]**2*sign[None, None]
            J[..., 1:3] = -F[..., None]*(phi[:, None, None, None]*Q[:, :, None, :]/(self.R0*den[:, :, None, None]))
            J[..., 3] = -.1*F*(2*alpha[0]*phi[:, None, None]*sign[None, None]-slope[..., None]/den[..., None])
            dQdB = radial[:, :, None, None]*np.eye(2)[None, None]+2*k[:, None, None, None]*B[None, :, :, None]*B[None, :, None, :]
            dQ = self.R0*np.einsum('nvab,vbp->nvap', dQdB, self.basis)
            J[..., 4:8] = np.einsum('nvab,nvbp->nvap', D, dQ)
            dQdbeta = r2[:, None]*B/self.R0**2
            barrel_jac = np.einsum('nvab,vb->nva', D, dQdbeta)
            for c in range(1, self.nc):
                J[..., 8+c-1] = barrel_jac*(cid == c)[:, None, None]
            J -= J.mean(axis=1, keepdims=True)
            J *= g[:, None, None, None]
        result['jacobian'] = J
        return result

    def evaluate(self, params, derivatives=False, jacobian=False):
        """Evaluate all rows, rejecting the whole objective on any domain failure.

        Derivatives return the exact gradient and Gauss Newton matrix. The
        optional residual_jacobian is sqrt(weight)*d(prediction)/d(params).
        Returned residual is prediction-X in native pixels, before weighting.
        """
        params = np.asarray(params, dtype=np.float64)
        self.unpack(params)
        box_valid = bool(np.all(params >= self.bounds[:, 0]) and np.all(params <= self.bounds[:, 1]))
        arrays = {key: [] for key in ('prediction', 'H', 'Z', 'mu', 'denominator', 'phi', 'radial_factor', 'radial_derivative', 'exponent', 'valid')}
        gradient = np.zeros(self.np)
        normal = np.zeros((self.np, self.np))
        jblocks = []
        value = 0.
        for lo in range(0, self.n, self.chunk_size):
            rows = slice(lo, min(lo+self.chunk_size, self.n))
            term = self._terms(params, rows, derivatives or jacobian)
            for key in arrays:
                arrays[key].append(term[key])
            residual = term['prediction']-self.X[rows]
            w = self.weights[rows]
            value += float(np.sum(w[:, None, None]*residual**2))
            if derivatives or jacobian:
                J = term['jacobian']
                weighted_J = (J*np.sqrt(w)[:, None, None, None]).reshape(-1, self.np)
                r = (residual*np.sqrt(w)[:, None, None]).reshape(-1)
                gradient += 2*weighted_J.T@r
                normal += 2*weighted_J.T@weighted_J
                if jacobian:
                    jblocks.append(weighted_J)
        out = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
        valid = bool(box_valid and out['valid'].all() and np.isfinite(value))
        if not valid:
            self.rejected_domain_evaluations += 1
        out.update(objective=float(value) if valid else float('inf'), loss=float(value) if valid else float('inf'),
                   residual=out['prediction']-self.X, domain_valid=valid, box_valid=box_valid,
                   B=term['B'], alpha=term['alpha'], theta04=term['theta04'], kappa=term['kappa'],
                   template_min_singular=term['template_min_singular'], params=params.copy(),
                   domain=dict(minimum_denominator=float(out['denominator'].min()),
                               minimum_radial_factor=float(out['radial_factor'].min()),
                               minimum_radial_derivative=float(out['radial_derivative'].min()),
                               maximum_absolute_stretch_exponent=float(np.abs(out['exponent']).max()),
                               invalid_rows=int((~out['valid']).sum()), margin=DELTA))
        if derivatives or jacobian:
            out.update(gradient=gradient, normal_matrix=normal)
        if jacobian:
            out['residual_jacobian'] = np.vstack(jblocks)
        return out

    def _objective(self, params, derivatives=False):
        """Compact chunked fitting evaluation; does not retain per-frame arrays."""
        if not np.isfinite(params).all() or np.any(params < self.bounds[:, 0]) or np.any(params > self.bounds[:, 1]):
            return (float('inf'), None, None) if derivatives else float('inf')
        value = 0.
        gradient = np.zeros(self.np)
        normal = np.zeros((self.np, self.np))
        for lo in range(0, self.n, self.chunk_size):
            rows = slice(lo, min(lo+self.chunk_size, self.n))
            term = self._terms(params, rows, derivatives)
            if not term['valid'].all():
                self.rejected_domain_evaluations += 1
                return (float('inf'), None, None) if derivatives else float('inf')
            w = self.weights[rows]
            residual = term['prediction']-self.X[rows]
            value += float(np.sum(w[:, None, None]*residual**2))
            if derivatives:
                J = term['jacobian'].reshape(-1, self.np)
                rw = np.repeat(np.sqrt(2*w), 6)
                J = J*rw[:, None]
                r = residual.reshape(-1)*rw
                gradient += J.T@r
                normal += J.T@J
        if not np.isfinite(value):
            return (float('inf'), None, None) if derivatives else float('inf')
        return (value, gradient, normal) if derivatives else value

    def _stationarity(self, params, gradient, normal):
        scaled = gradient*self.scales
        width = self.bounds[:, 1]-self.bounds[:, 0]
        lower = params <= self.bounds[:, 0]+1e-7*width
        upper = params >= self.bounds[:, 1]-1e-7*width
        projected = scaled.copy()
        projected[lower & (gradient > 0)] = 0.
        projected[upper & (gradient < 0)] = 0.
        # Noise units offer an additional interpretable gradient convention.
        diagonal = np.maximum(np.diag(normal), 1e-20)
        return dict(gradient_norm=float(np.linalg.norm(gradient)),
                    scaled_gradient=float(np.max(np.abs(scaled))),
                    projected_gradient_norm=float(np.max(np.abs(projected))),
                    sensitivity_scaled_gradient=float(np.max(np.abs(gradient)/np.sqrt(diagonal))),
                    stationary=bool(np.max(np.abs(projected)) < 1e-6),
                    active_bounds=[self.names[i] for i in np.flatnonzero(lower | upper)])

    def _solve(self, start, max_iter):
        params = np.asarray(start, dtype=np.float64).copy()
        tic = time.monotonic()
        damping = 1e-3
        history = []
        reason = 'iteration_limit'
        if not np.isfinite(self._objective(params)):
            raise ValueError('Infeasible initial parameters')
        for iteration in range(int(max_iter)):
            value, gradient, normal = self._objective(params, True)
            status = self._stationarity(params, gradient, normal)
            history.append(dict(iteration=iteration, objective=float(value),
                                projected_gradient_norm=status['projected_gradient_norm'], damping=float(damping)))
            if status['stationary']:
                reason = 'projected_gradient'
                break
            # Numerical coordinate scaling only; no prior or diagonal penalty
            # is added to the reported objective or identifiability calculation.
            H = normal*self.scales[:, None]*self.scales[None]
            grad = gradient*self.scales
            diagonal = np.maximum(np.diag(H), 1e-12)
            lower = params <= self.bounds[:, 0]+1e-9
            upper = params >= self.bounds[:, 1]-1e-9
            free = np.flatnonzero(~((lower & (gradient > 0)) | (upper & (gradient < 0))))
            accepted = False
            for attempt in range(18):
                step = np.zeros(self.np)
                try:
                    step[free] = np.linalg.solve(H[np.ix_(free, free)]+damping*np.diag(diagonal[free]), -grad[free])
                except np.linalg.LinAlgError:
                    damping *= 10
                    continue
                if np.max(np.abs(step)) > 25:
                    step *= 25/np.max(np.abs(step))
                step *= self.scales
                # Backtracking enforces every row's nonlinear domain. Box
                # projection permits genuine numerical-envelope optima.
                factor = 1.
                for backtrack in range(24):
                    candidate = np.clip(params+factor*step, self.bounds[:, 0], self.bounds[:, 1])
                    delta = candidate-params
                    directional = float(gradient@delta)
                    if directional < 0:
                        trial = self._objective(candidate)
                        if np.isfinite(trial) and trial <= value+1e-4*directional:
                            params = candidate
                            damping = max(damping/3., 1e-12)
                            accepted = True
                            break
                    factor *= .5
                if accepted:
                    break
                damping *= 10.
            if not accepted:
                reason = 'no_feasible_descent'
                break
        value, gradient, normal = self._objective(params, True)
        status = self._stationarity(params, gradient, normal)
        return dict(params=params, objective=float(value), loss=float(value),
                    success=bool(status['stationary']), reason=reason, iterations=len(history),
                    seconds=time.monotonic()-tic, history=history, **status)

    def fit(self, start=None, max_iter=250, n_starts=8, seed=1729, checkpoint=None, starts=None):
        """Feasible damped Gauss Newton multistart with exact analytical Jacobian.

        Explicit starts support checkpoint/resume and sensitivity reruns. The
        selected start minimizes the identical all-row objective, with optimizer
        stationarity and near-tied parameter drift reported separately.
        """
        if int(n_starts) < 1:
            raise ValueError('n_starts must be at least one')
        base = np.zeros(self.np) if start is None else np.asarray(start, dtype=np.float64).copy()
        self.unpack(base)
        if not np.isfinite(self._objective(base)):
            raise ValueError('Base initializer is outside the model/search domain')
        rng = np.random.default_rng(seed)
        initials = [base]
        if starts is not None:
            initials.extend(np.asarray(s, dtype=np.float64).copy() for s in starts)
        else:
            theta_starts = (-30., -15., 15., 30., -5., 5., 0.)
            for j in range(1, int(n_starts)):
                candidate = base.copy()
                candidate[:3] += rng.normal(size=3)*np.asarray([.02, .03, .03])
                candidate[3] = theta_starts[(j-1) % len(theta_starts)]
                candidate[4:8] += rng.normal(size=4)*.02
                candidate[8:] += rng.normal(size=self.nc-1)*.02
                candidate = np.clip(candidate, self.bounds[:, 0], self.bounds[:, 1])
                # Shrink initialization only; all data remain in every fit.
                for shrink in range(30):
                    if np.isfinite(self._objective(candidate)):
                        break
                    candidate = (candidate+base)/2.
                initials.append(candidate)
        results = []
        for index, initial in enumerate(initials):
            result = self._solve(initial, max_iter)
            result.update(start=index, initial_params=initial.copy())
            results.append(result)
            if checkpoint is not None:
                checkpoint(dict(completed_start=index, result=result, best=min(results, key=lambda r: r['objective'])))
        best = min(results, key=lambda r: r['objective'])
        output = best.copy()
        output.update(multistart=results, starts=results, best_start=best['start'], bounds=self.bounds.copy(),
                      param_scales=self.scales.copy(), param_names=self.names, gauge=GAUGE,
                      R0=self.R0, weights_definition='equal capture, equal fixation within capture, equal row within fixation, mean 3 vertex squared Euclidean errors',
                      rejected_domain_evaluations=self.rejected_domain_evaluations)
        losses = np.asarray([r['objective'] for r in results])
        near = [r for r in results if r['objective'] <= best['objective']+max(1e-8, 1e-7*abs(best['objective']))]
        nearparams = np.asarray([r['params'] for r in near])
        output['start_stability'] = dict(objective_min=float(losses.min()), objective_max=float(losses.max()),
                                         stationary_starts=int(sum(r['stationary'] for r in results)),
                                         near_best_starts=len(near), near_best_parameter_min=nearparams.min(axis=0),
                                         near_best_parameter_max=nearparams.max(axis=0))
        return output

    def predictions(self, params):
        return self.evaluate(params)['prediction']

    def inverse(self, params, X=None):
        """Invert using the fitted raw optical mean; return native reference space.

        With X omitted invert observed centered triangles; supplying model
        predictions is useful for a forward/inverse numerical identity check.
        Centering loses translation, so inverse uses the model's raw mean. It
        does not fit a translation or rescale the corrected output.
        """
        ev = self.evaluate(params)
        X = self.X if X is None else np.asarray(X, dtype=np.float64)
        if X.shape != self.X.shape:
            raise ValueError('Inverse X must be row-aligned (N,3,2)')
        stretch = np.exp(ev['exponent'][:, None]*np.asarray([1., -1.])[None])
        v = (X/self.g[:, None, None]+ev['mu'][:, None])/stretch[:, None]
        inverse_den = 1.-ev['phi'][:, None]*np.einsum('nvd,d->nv', v, ev['alpha'][1:])/self.R0
        with np.errstate(invalid='ignore', divide='ignore'):
            q = v/inverse_den[..., None]
        back, radial_valid, info = inverse_radial(q, ev['kappa'][self.capture_index], delta=DELTA)
        valid = radial_valid & ev['valid'] & np.all(inverse_den > DELTA, axis=1)
        info.update(frame_failures=int((~valid).sum()), minimum_inverse_denominator=float(inverse_den.min()),
                    minimum_forward_denominator=ev['domain']['minimum_denominator'])
        return np.where(valid[:, None, None], back, np.nan), valid, info

    def diagnostics(self, params, rank_rtol=1e-8):
        """Weighted data Jacobian rank and gauges; no prior or constraint rows."""
        ev = self.evaluate(params, derivatives=True)
        if not ev['domain_valid']:
            raise ValueError('Cannot assess identifiability for an invalid fit')
        # QR each chunk before the small SVD, avoiding normal-matrix squared
        # conditioning and retaining only O(number_of_chunks * parameters²).
        blocks = []
        for lo in range(0, self.n, self.chunk_size):
            rows = slice(lo, min(lo+self.chunk_size, self.n))
            J = self._terms(params, rows, True)['jacobian']
            weighted = (J*np.sqrt(self.weights[rows])[:, None, None, None]).reshape(-1, self.np)
            blocks.append(np.linalg.qr(weighted*self.scales[None], mode='r'))
        _, singular, right = np.linalg.svd(np.vstack(blocks), full_matrices=False)
        tolerance = max(float(singular[0])*rank_rtol, np.finfo(float).eps*max(self.n*6, self.np)*float(singular[0]))
        rank = int(np.sum(singular > tolerance))
        status = self._stationarity(np.asarray(params), ev['gradient'], ev['normal_matrix'])
        contacts = list(status['active_bounds'])
        for name in ('minimum_denominator', 'minimum_radial_factor', 'minimum_radial_derivative'):
            if ev['domain'][name] <= 100*DELTA:
                contacts.append(name)
        if ev['domain']['maximum_absolute_stretch_exponent'] >= 20.-1e-5:
            contacts.append('stretch_exponent_envelope')
        if ev['template_min_singular'] <= 1e-6*self.R0:
            contacts.append('template_rank_envelope')
        per_capture = {}
        for capture_id in self.captures:
            rows = self.capture == capture_id
            per_capture[int(capture_id)] = dict(rows=int(rows.sum()),
                weighted_objective=float(np.sum(self.weights[rows, None, None]*ev['residual'][rows]**2)),
                unweighted_vertex_rms_px=float(np.sqrt(np.mean(np.sum(ev['residual'][rows]**2, axis=2)))),
                minimum_denominator=float(ev['denominator'][rows].min()))
        return dict(objective=ev['objective'], rank=rank, parameters=self.np,
                    singular_values=singular, right_singular_vectors_scaled=right,
                    rank_rtol=rank_rtol, rank_tolerance=tolerance,
                    condition_number=float(singular[0]/singular[-1]) if singular[-1] > 0 else float('inf'),
                    spectrum_definition='weighted all-row native-pixel data residual Jacobian in reported parameter scales; no prior or constraint rows',
                    B_px=ev['B'], B_centroid_px=ev['B'].mean(axis=0), B_energy_px2=float(np.sum(ev['B']**2)),
                    alpha=ev['alpha'], theta04_deg=ev['theta04'], kappa_px_inv2=ev['kappa'],
                    beta=ev['kappa']*self.R0**2, capture_ids=self.captures.copy(), R0_px=self.R0,
                    g_range=[float(self.g.min()), float(self.g.max())], supplied_g_fixed=True,
                    gauge=GAUGE, scale_identifiability='known fixed g and k1=0 anchor template size; no free global magnification',
                    domain=ev['domain'], bound_contacts=contacts, bounds=self.bounds.copy(),
                    numerical_search_envelope='finite boxes, |stretch exponent|<20 and noncollinear B are numerical envelopes, not physical limits; widen boxes to audit sensitivity',
                    parameter_names=self.names, parameter_scales=self.scales.copy(),
                    weight_groups=self.weight_groups, per_capture=per_capture, **status)


def fit(X, theta, capture, fixation, reference, g=None, R0=None, bounds=None, **kwargs):
    """Convenience wrapper around ScaleComparisonProblem.fit."""
    return ScaleComparisonProblem(X, theta, capture, fixation, reference, g, R0, bounds).fit(**kwargs)


def evaluate(params, X, theta, capture, fixation, reference, g=None, R0=None, bounds=None, **kwargs):
    """Convenience all-row model evaluation."""
    return ScaleComparisonProblem(X, theta, capture, fixation, reference, g, R0, bounds).evaluate(params, **kwargs)


def diagnostics(params, X, theta, capture, fixation, reference, g=None, R0=None, bounds=None, **kwargs):
    """Convenience data-only identifiability/domain diagnostics."""
    return ScaleComparisonProblem(X, theta, capture, fixation, reference, g, R0, bounds).diagnostics(params, **kwargs)
