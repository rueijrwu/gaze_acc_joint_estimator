"""Capture-1 horizontal-gaze/A ablation with profiled P1 scale.

All optical constants are frozen. Arithmetic and analytic derivatives can run
on NumPy or CuPy; only the optimizer vector crosses the device boundary.
"""
import numpy as np


class JointState:
    def __init__(self, x1, x4, b1, b4, gaze, units, k1, k4, exposure,
                 slope, reference_A, gaze_width=.5, A_width=.25, joint=True, xp=np):
        self.xp = xp
        for name, value in dict(x1=x1, x4=x4, b1=b1, b4=b4, gaze=gaze,
                                units=units, k1=k1, k4=k4).items():
            setattr(self, name, xp.asarray(value, dtype=xp.float64))
        self.e = xp.asarray(exposure, dtype=xp.int64)
        self.count = xp.bincount(self.e, minlength=5)
        if len(self.count) != 5 or bool(xp.any(self.count == 0)):
            raise ValueError('Exactly five nonempty Capture-1 fixations required')
        self.n = len(x1)
        self.q = self.n / (5 * self.count[self.e])
        self.nominal = xp.asarray([-10., -5., 0., 5., 10.])
        self.slope, self.reference_A = float(slope), float(reference_A)
        self.strength = xp.asarray([1 / gaze_width**2, 1 / A_width**2])
        self.joint = joint
        self.d = 2 if joint else 1

    def host(self, value):
        return np.asarray(value) if self.xp is np else self.xp.asnumpy(value)

    def shape(self, theta, A, b, k, radial):
        """Size-normalized keystone and derivatives in degree / diopter units."""
        xp = self.xp
        tx = theta / self.units[:, 0]
        ty = self.gaze[:, 1] / self.units[:, 1]
        r = xp.sqrt(xp.mean(xp.sum(b*b, axis=1)))
        r2 = xp.sum(b*b, axis=1)
        kap = self.slope*(A-self.reference_A) if radial else xp.zeros_like(A)
        pre = b[None] * (1 + kap[:, None]*r2)[..., None]
        dp = xp.zeros((self.n, 3, 2, 2), dtype=xp.float64)
        if radial:
            dp[..., 1] = b[None]*(self.slope*r2)[None, :, None]
        stretch = xp.stack((xp.exp(k[0]*tx*tx-k[1]*ty*ty),
                            xp.exp(-k[0]*tx*tx+k[1]*ty*ty)), axis=1)
        den = 1 + k[2]*tx[:, None]*pre[..., 1]/r + k[3]*ty[:, None]*pre[..., 0]/r
        dd = (k[2]*tx[:, None, None]*dp[:, :, 1, :] +
              k[3]*ty[:, None, None]*dp[:, :, 0, :])/r
        dd[..., 0] += k[2]*pre[..., 1]/(r*self.units[:, 0, None])
        f = pre*stretch[:, None]/den[..., None]
        df = dp*stretch[:, None, :, None]/den[..., None, None] - f[..., None]*(dd/den[..., None])[:, :, None, :]
        df[..., 0] += f*xp.stack((2*k[0]*tx/self.units[:, 0],
                                -2*k[0]*tx/self.units[:, 0]), axis=1)[:, None]
        c = f-f.mean(axis=1, keepdims=True)
        dc = df-df.mean(axis=1, keepdims=True)
        pc = pre-pre.mean(axis=1, keepdims=True)
        dpc = dp-dp.mean(axis=1, keepdims=True)
        cc = xp.sum(c*c, axis=(1, 2))
        pp = xp.sum(pc*pc, axis=(1, 2))
        size = xp.sqrt(cc/pp)
        dl = (xp.sum(c[..., None]*dc, axis=(1, 2))/cc[:, None] -
              xp.sum(pc[..., None]*dpc, axis=(1, 2))/pp[:, None])
        h = c/size[:, None, None]
        dh = dc/size[:, None, None, None]-h[..., None]*dl[:, None, None]
        valid = (xp.all(den > 1e-8, axis=1) & xp.all(1+3*kap[:, None]*r2 > 1e-8, axis=1))
        return h, dh, valid

    def forward(self, states):
        xp = self.xp
        states = xp.asarray(states, dtype=xp.float64).reshape(self.n, self.d)
        theta = states[:, 0]
        A = states[:, 1] if self.joint else xp.full(self.n, self.reference_A)
        h1, dh1, v1 = self.shape(theta, A, self.b1, self.k1, False)
        h4, dh4, v4 = self.shape(theta, A, self.b4, self.k4, True)
        numerator = xp.sum(self.x1*h1, axis=(1, 2))
        denominator = xp.sum(h1*h1, axis=(1, 2))
        m = numerator/denominator
        dm = (xp.sum(self.x1[..., None]*dh1, axis=(1, 2)) -
              2*m[:, None]*xp.sum(h1[..., None]*dh1, axis=(1, 2)))/denominator[:, None]
        predictions = []
        derivatives = []
        for h, dh in ((h1, dh1), (h4, dh4)):
            predictions.append(m[:, None, None]*h)
            derivatives.append((dm[:, None, None]*h[..., None] + m[:, None, None, None]*dh)[..., :self.d])
        return predictions, derivatives, m, v1 & v4 & (m > 0)

    def frame_terms(self, states):
        xp = self.xp
        predictions, derivatives, m, valid = self.forward(states)
        cost = xp.zeros(self.n)
        grad = xp.zeros((self.n, self.d))
        for x, pred, jac in zip((self.x1, self.x4), predictions, derivatives):
            error = pred-x
            cost += xp.sum(error*error, axis=(1, 2))/3
            grad += 2*xp.sum(error[..., None]*jac, axis=(1, 2))/3
        return cost, grad, valid

    def means(self, states):
        xp = self.xp
        return xp.stack([xp.bincount(self.e, weights=states[:, j], minlength=5)/self.count
                         for j in range(self.d)], axis=1)

    def evaluate(self, vector):
        xp = self.xp
        states = xp.asarray(vector).reshape(self.n, self.d)
        cost, grad, valid = self.frame_terms(states)
        if not bool(xp.all(valid)):
            raise ValueError('Trial outside frozen forward optical domain')
        targets = xp.stack((self.nominal, xp.full(5, self.reference_A)), axis=1)[:, :self.d]
        delta = self.means(states)-targets
        value = xp.sum(self.q*cost) + self.n/5*xp.sum(self.strength[:self.d]*delta*delta)
        gradient = self.q[:, None]*(grad+2*self.strength[:self.d]*delta[self.e])
        packed = self.host(xp.concatenate((value[None], gradient.ravel())))
        if not np.isfinite(packed).all():
            raise ValueError('Nonfinite joint objective')
        return float(packed[0]), packed[1:]

    def curvature(self, states):
        """Observed frame Hessians, excluding positive mean-anchor blocks."""
        xp = self.xp
        states = xp.asarray(states).reshape(self.n, self.d)
        columns = []
        for j in range(self.d):
            step = xp.zeros_like(states)
            step[:, j] = 1e-4
            columns.append((self.frame_terms(states+step)[1]-self.frame_terms(states-step)[1])/2e-4)
        h = xp.stack(columns, axis=2)
        return self.host((h+h.transpose(0, 2, 1))/2)


def projected_gradient(states, gradient, lower, upper):
    result = gradient.copy()
    active = ((states <= lower+1e-7) & (gradient > 0)) | ((states >= upper-1e-7) & (gradient < 0))
    result[active] = 0
    return result, active


def fit_states(model, start, lower, upper):
    """L-BFGS-B plus block-diagonal / fixation-rank Newton polishing."""
    from scipy.optimize import minimize
    solution = minimize(model.evaluate, start.ravel(), jac=True, method='L-BFGS-B',
                        bounds=list(zip(lower.ravel(), upper.ravel())),
                        options=dict(maxiter=150, maxls=40, maxcor=15, ftol=1e-15, gtol=1e-7))
    states = solution.x.reshape(model.n, model.d)
    value, gradient = model.evaluate(states)
    q, exposure, counts, strength = map(model.host, (model.q, model.e, model.count, model.strength))
    polish = []
    for iteration in range(160):
        gradient = gradient.reshape(states.shape)
        pg, active = projected_gradient(states, gradient, lower, upper)
        if np.max(abs(pg)) < 1e-7:
            break
        h = model.curvature(states)
        free = ~active
        masked = h*free[:, :, None]*free[:, None, :]
        masked += np.eye(model.d)[None]*active[:, :, None]
        # Weak or negative local curvature occurs around shallow gaze branches.
        # Damping guides descent; final certification uses undamped curvature.
        eigen = np.linalg.eigvalsh(masked)[:, 0]
        damping = np.maximum(0., 1e-4-eigen)
        masked += np.eye(model.d)[None]*damping[:, None, None]*free[:, :, None]
        inverse = np.linalg.inv(masked)*free[:, :, None]*free[:, None, :]
        local = gradient/q[:, None]
        initial = -np.einsum('nij,nj->ni', inverse, local)
        step = initial.copy()
        for f in range(5):
            select = exposure == f
            rank = np.diag(2*strength[:model.d]/counts[f])
            correction = np.linalg.solve(np.eye(model.d)+rank@inverse[select].sum(axis=0),
                                         rank@initial[select].sum(axis=0))
            step[select] -= np.einsum('nij,j->ni', inverse[select], correction)
        # Limit a full Newton step without adding a state prior to the objective.
        trust = np.array([2., .5])[:model.d]
        factor = np.maximum(1., np.max(abs(step)/trust, axis=1))
        step /= factor[:, None]
        accepted = False
        for power in range(25):
            candidate = np.clip(states+step*2.**(-power), lower, upper)
            nv, ng = model.evaluate(candidate)
            new_pg, _ = projected_gradient(candidate, ng.reshape(states.shape), lower, upper)
            directional = float(np.sum(gradient*(candidate-states)))
            if (nv <= value+1e-4*min(directional, 0.) or
                    (nv <= value+1e-7 and np.max(abs(new_pg)) < np.max(abs(pg)))):
                polish.append(dict(iteration=iteration, before=value, after=nv))
                states, value, gradient = candidate, nv, ng
                accepted = True
                break
        if not accepted:
            break
    _, gradient = model.evaluate(states)
    pg, _ = projected_gradient(states, gradient.reshape(states.shape), lower, upper)
    h = model.curvature(states)
    at_bound = (states <= lower+1e-7) | (states >= upper-1e-7)
    # Sufficient curvature certificate on the full free subspace; mean terms PSD.
    free = ~at_bound
    masked = h*free[:, :, None]*free[:, None, :]
    masked += np.eye(model.d)[None]*at_bound[:, :, None]*1e30
    eigen = np.linalg.eigvalsh(masked)
    relevant = eigen[eigen < 1e29]
    minimum = float(relevant.min()) if len(relevant) else None
    certificate = dict(projected_gradient_inf=float(np.max(abs(pg))),
                       minimum_free_frame_curvature=minimum,
                       lower_bound_counts=(states <= lower+1e-7).sum(axis=0).tolist(),
                       upper_bound_counts=(states >= upper-1e-7).sum(axis=0).tolist())
    certificate['stationary_positive_curvature'] = bool(np.max(abs(pg)) < 1e-5 and
                                                       (minimum is None or minimum > 0))
    return states, dict(objective_equal_fixation_px2=value/model.n,
                        optimizer_success=bool(solution.success), iterations=int(solution.nit),
                        optimizer_message=str(solution.message), certificate=certificate, polish=polish)
