"""Raw projective keystone: center output, never normalize its RMS size.

The four-coefficient definition fixes the identity at zero gaze and excludes
an arbitrary isotropic gaze-scale function. One common scale is profiled from
P1 and applied to P4. Radial coefficients are relative to an empirical origin.
"""
import numpy as np
from .capture_shape import inverse_radial
from .joint_state import JointState


def center(points, xp=np):
    return points-points.mean(axis=-2, keepdims=True)


class RawKeystone:
    def __init__(self, gaze, reference, observed, weights, units, magnification=None, xp=np):
        self.xp = xp
        self.gaze = xp.asarray(gaze, dtype=xp.float64)
        self.b = xp.asarray(reference, dtype=xp.float64)
        self.x = xp.asarray(observed, dtype=xp.float64)
        self.w = xp.asarray(weights, dtype=xp.float64)
        self.units = xp.asarray(units, dtype=xp.float64)
        self.t = self.gaze/self.units
        self.m = None if magnification is None else xp.asarray(magnification, dtype=xp.float64)
        self.r = xp.sqrt(xp.mean(xp.sum(self.b*self.b, axis=1)))
        self.rho2 = xp.sum(self.b*self.b, axis=1)/self.r**2

    def host(self, a):
        return np.asarray(a) if self.xp is np else self.xp.asnumpy(a)

    def shape(self, coefficients, derivatives=False):
        xp = self.xp
        k = xp.asarray(coefficients, dtype=xp.float64)
        tx, ty = self.t[:, 0], self.t[:, 1]
        lam = k[4] if len(k) == 5 else 0.
        pre = self.b*(1+lam*self.rho2)[:, None]
        stretch = xp.exp(k[0]*tx*tx-k[1]*ty*ty)
        factors = xp.stack((stretch, 1/stretch), axis=1)[:, None, :]
        vx = tx[:, None]*pre[:, 1]/self.r
        vy = ty[:, None]*pre[:, 0]/self.r
        denominator = 1+k[2]*vx+k[3]*vy
        f = pre*factors/denominator[..., None]
        h = center(f, xp)
        # Compatibility return slot is a diagnostic raw/input size ratio only.
        pc = center(pre, xp)
        ratio = xp.sqrt(xp.sum(h*h, axis=(1, 2))/xp.sum(pc*pc))
        if not derivatives:
            return h, f.mean(axis=1), ratio
        df = [f*xp.stack((tx*tx, -tx*tx), axis=1)[:, None, :],
              f*xp.stack((-ty*ty, ty*ty), axis=1)[:, None, :],
              -f*(vx/denominator)[..., None], -f*(vy/denominator)[..., None]]
        if len(k) == 5:
            dp = self.b*self.rho2[:, None]
            dd = (k[2]*tx[:, None]*dp[:, 1]+k[3]*ty[:, None]*dp[:, 0])/self.r
            df.append(dp*factors/denominator[..., None]-f*(dd/denominator)[..., None])
        df = xp.stack(df, axis=-1)
        dh = df-df.mean(axis=1, keepdims=True)
        return h, f.mean(axis=1), ratio, dh

    def prediction(self, coefficients):
        xp = self.xp
        h = self.shape(coefficients)[0]
        m = self.m
        if m is None:
            m = xp.sum(self.x*h, axis=(1, 2))/xp.sum(h*h, axis=(1, 2))
        if not bool(xp.all(xp.isfinite(m) & (m > 0))):
            raise ValueError('Nonpositive raw P1 profile scale')
        return self.host(m[:, None, None]*h), self.host(m)

    def evaluate(self, coefficients):
        xp = self.xp
        h, _, _, dh = self.shape(coefficients, True)
        m = self.m
        if m is None:
            m = xp.sum(self.x*h, axis=(1, 2))/xp.sum(h*h, axis=(1, 2))
        error = self.x-m[:, None, None]*h
        cost = xp.sum(self.w*xp.sum(error*error, axis=(1, 2)))
        grad = -2*xp.sum(self.w[:, None]*m[:, None]*xp.sum(error[..., None]*dh, axis=(1, 2)), axis=0)
        packed = self.host(xp.concatenate((cost[None], grad)))
        if not np.isfinite(packed).all() or not bool(xp.all(m > 0)):
            raise ValueError('Nonfinite raw-keystone objective/domain')
        return float(packed[0]), packed[1:]

    def inverse(self, coefficients, magnification=None, observed=None, remove_barrel=True):
        k = np.asarray(coefficients)
        _, mu, _ = self.shape(k)
        x = self.host(self.x) if observed is None else center(np.asarray(observed))
        m = self.prediction(k)[1] if magnification is None else np.asarray(magnification)
        # Restore model optical centroid. No RMS factor in either direction.
        z = center(x)/m[:, None, None]+self.host(mu)[:, None, :]
        t = self.host(self.t)
        r = float(self.host(self.r))
        a = k[0]*t[:, 0]**2-k[1]*t[:, 1]**2
        u = z/np.stack((np.exp(a), np.exp(-a)), axis=1)[:, None, :]
        den = 1-k[3]*t[:, 1, None]*u[..., 0]/r-k[2]*t[:, 0, None]*u[..., 1]/r
        pre = u/den[..., None]
        valid = np.all(den > 1e-8, axis=1) & (m > 0) & np.isfinite(pre).all(axis=(1, 2))
        if remove_barrel:
            back, radial_valid = inverse_radial(pre, np.full(len(x), (k[4] if len(k) == 5 else 0.)/r**2))
            valid &= radial_valid
        else:
            back = pre
        return np.where(valid[:, None, None], back, np.nan), valid

    def domain(self, coefficients):
        k = np.asarray(coefficients)
        b, t, r = self.host(self.b), self.host(self.t), float(self.host(self.r))
        lam = k[4] if len(k) == 5 else 0.
        rho = np.sum(b*b, axis=1)/r**2
        pre = b*(1+lam*rho)[:, None]
        den = 1+(k[2]*t[:, 0, None]*pre[:, 1]+k[3]*t[:, 1, None]*pre[:, 0])/r
        return dict(min_radial_factor=float(np.min(1+lam*rho)),
                    min_radial_derivative=float(np.min(1+3*lam*rho)),
                    min_keystone_denominator=float(den.min()),
                    valid=bool(np.min(1+lam*rho) > 0 and np.min(1+3*lam*rho) > 1e-8 and den.min() > 1e-8))


class RawFrameAccommodation:
    """Frozen raw keystone/gaze/P1 scale; forward camera-coordinate A fit."""
    def __init__(self, observed, reference, gaze, units, keystone, magnification,
                 exposure, expected, slope, reference_A, anchor_width=.25, xp=np):
        self.xp = xp
        for name, value in dict(x=observed, b=reference, gaze=gaze, units=units,
                                k=keystone, m=magnification, expected=expected).items():
            setattr(self, name, xp.asarray(value, dtype=xp.float64))
        self.t = self.gaze/self.units
        self.e = xp.asarray(exposure, dtype=xp.int64)
        self.count = xp.bincount(self.e, minlength=len(expected))
        if bool(xp.any(self.count == 0)):
            raise ValueError('Empty fixation')
        self.n = len(observed)
        self.nref = self.n/len(expected)
        self.q = self.nref/self.count[self.e]
        self.slope, self.reference_A = float(slope), float(reference_A)
        self.strength = 1/anchor_width**2
        self.r = xp.sqrt(xp.mean(xp.sum(self.b*self.b, axis=1)))
        self.r2 = xp.sum(self.b*self.b, axis=1)
        a = self.k[:, 0]*self.t[:, 0]**2-self.k[:, 1]*self.t[:, 1]**2
        self.stretch = xp.stack((xp.exp(a), xp.exp(-a)), axis=1)[:, None]
        self.qx = self.k[:, 3]*self.t[:, 1]/self.r
        self.qy = self.k[:, 2]*self.t[:, 0]/self.r

    def host(self, a):
        return np.asarray(a) if self.xp is np else self.xp.asnumpy(a)

    def geometry(self, A):
        xp = self.xp
        A = xp.asarray(A, dtype=xp.float64)
        kap = self.slope*(A-self.reference_A)
        pre = self.b[None]*(1+kap[:, None]*self.r2)[..., None]
        den = 1+self.qx[:, None]*pre[..., 0]+self.qy[:, None]*pre[..., 1]
        f = pre*self.stretch/den[..., None]
        dp = self.b[None]*(self.slope*self.r2)[None, :, None]
        dd = self.qx[:, None]*dp[..., 0]+self.qy[:, None]*dp[..., 1]
        df = dp*self.stretch/den[..., None]-f*(dd/den)[..., None]
        prediction = self.m[:, None, None]*center(f, xp)
        derivative = self.m[:, None, None]*center(df, xp)
        valid = (xp.all(den > 1e-8, axis=1) & xp.all(1+3*kap[:, None]*self.r2 > 1e-8, axis=1)
                 & xp.all(1+kap[:, None]*self.r2 > 0, axis=1) & (self.m > 0))
        return prediction, derivative, f.mean(axis=1), kap, valid

    def frame_cost(self, A):
        prediction, derivative, _, _, valid = self.geometry(A)
        error = prediction-self.x
        xp = self.xp
        return xp.sum(error*error, axis=(1, 2))/3, 2*xp.sum(error*derivative, axis=(1, 2))/3, valid

    def means(self, A):
        return self.xp.bincount(self.e, weights=A, minlength=len(self.expected))/self.count

    def evaluate(self, A):
        xp = self.xp
        A = xp.asarray(A, dtype=xp.float64)
        cost, grad, valid = self.frame_cost(A)
        if not bool(xp.all(valid)):
            raise ValueError('Invalid forward raw-keystone domain')
        delta = self.means(A)-self.expected
        value = xp.sum(self.q*cost)+self.nref*self.strength*xp.sum(delta*delta)
        gradient = self.q*(grad+2*self.strength*delta[self.e])
        packed = self.host(xp.concatenate((value[None], gradient)))
        if not np.isfinite(packed).all():
            raise ValueError('Nonfinite raw forward A objective')
        return float(packed[0]), packed[1:]

    def feasible_bounds(self, lower=0., upper=6.):
        xp = self.xp
        lo, hi = xp.full(self.n, lower), xp.full(self.n, upper)
        if not bool(xp.all(self.geometry(lo)[4])):
            raise ValueError('Invalid lower forward A domain')
        lost = xp.zeros(self.n, dtype=bool)
        for value in np.linspace(lower, upper, 25):
            valid = self.geometry(xp.full(self.n, value))[4]
            if bool(xp.any(lost & valid)):
                raise ValueError('Noncontiguous forward domain')
            lost |= ~valid
        upper_valid = self.geometry(hi)[4]
        left, right = lo.copy(), hi.copy()
        for _ in range(40):
            mid = (left+right)/2
            valid = self.geometry(mid)[4]
            left, right = xp.where(valid, mid, left), xp.where(valid, right, mid)
        hi = xp.where(upper_valid, hi, xp.maximum(lo, left-1e-6))
        return self.host(lo), self.host(hi)

    def certificate(self, A, lower, upper):
        A = np.asarray(A)
        _, gradient = self.evaluate(A)
        pg = gradient.copy()
        pg[((A <= lower+1e-7) & (gradient > 0)) | ((A >= upper-1e-7) & (gradient < 0))] = 0
        xp = self.xp
        a = xp.asarray(A)
        plus, minus = xp.minimum(a+1e-4, xp.asarray(upper)), xp.maximum(a-1e-4, xp.asarray(lower))
        curvature = self.host((self.frame_cost(plus)[1]-self.frame_cost(minus)[1])/(plus-minus))
        free = ~((A <= lower+1e-7) | (A >= upper-1e-7))
        minimum = float(curvature[free].min()) if free.any() else None
        return dict(projected_gradient_inf=float(np.max(abs(pg))), free_frame_curvature_min=minimum,
                    lower_bound_frames=int(np.sum(A <= lower+1e-7)), upper_bound_frames=int(np.sum(A >= upper-1e-7)),
                    forward_domain_limited_frames=int(np.sum(upper < 6-1e-7)),
                    stationary=bool(np.max(abs(pg)) < 1e-5 and (minimum is None or minimum > 0))), curvature

    def recover(self, A, observed=None):
        # CPU inversion is diagnostic only and cannot change the fitting metric.
        _, _, mu, kap, forward_valid = self.geometry(A)
        x = self.host(self.x) if observed is None else center(np.asarray(observed))
        z = x/self.host(self.m)[:, None, None]+self.host(mu)[:, None]
        u = z/self.host(self.stretch)
        den = 1-self.host(self.qx)[:, None]*u[..., 0]-self.host(self.qy)[:, None]*u[..., 1]
        pre = u/den[..., None]
        result, valid = inverse_radial(pre, self.host(kap))
        valid &= self.host(forward_valid) & np.all(den > 1e-8, axis=1)
        return np.where(valid[:, None, None], result, np.nan), valid


class RawJointState(JointState):
    """Joint horizontal gaze/A with raw K and trial-profiled common P1 scale."""
    def shape(self, theta, A, b, k, radial):
        xp = self.xp
        tx, ty = theta/self.units[:, 0], self.gaze[:, 1]/self.units[:, 1]
        r = xp.sqrt(xp.mean(xp.sum(b*b, axis=1)))
        r2 = xp.sum(b*b, axis=1)
        kap = self.slope*(A-self.reference_A) if radial else xp.zeros_like(A)
        pre = b[None]*(1+kap[:, None]*r2)[..., None]
        dp = xp.zeros((self.n, 3, 2, 2), dtype=xp.float64)
        if radial:
            dp[..., 1] = b[None]*(self.slope*r2)[None, :, None]
        a = k[0]*tx*tx-k[1]*ty*ty
        stretch = xp.stack((xp.exp(a), xp.exp(-a)), axis=1)
        den = 1+(k[2]*tx[:, None]*pre[..., 1]+k[3]*ty[:, None]*pre[..., 0])/r
        dd = (k[2]*tx[:, None, None]*dp[:, :, 1, :]+k[3]*ty[:, None, None]*dp[:, :, 0, :])/r
        dd[..., 0] += k[2]*pre[..., 1]/(r*self.units[:, 0, None])
        f = pre*stretch[:, None]/den[..., None]
        df = dp*stretch[:, None, :, None]/den[..., None, None]-f[..., None]*(dd/den[..., None])[:, :, None]
        df[..., 0] += f*xp.stack((2*k[0]*tx/self.units[:, 0], -2*k[0]*tx/self.units[:, 0]), axis=1)[:, None]
        h = center(f, xp)
        dh = df-df.mean(axis=1, keepdims=True)
        valid = xp.all(den > 1e-8, axis=1) & xp.all(1+3*kap[:, None]*r2 > 1e-8, axis=1)
        return h, dh, valid
