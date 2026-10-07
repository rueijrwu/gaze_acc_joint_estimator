"""Linear coefficient designs and physical-degree state derivatives."""
from __future__ import annotations
import numpy as np

THETA_SCALE = 10.0
STATE_SCALE = np.array([10., 4.])
LOWER = np.array([-20., 0.])
UPPER = np.array([20., 6.])


def bases(x):
    x = np.asarray(x, float)
    t, A = x[..., 0]/THETA_SCALE, x[..., 1]
    L, La = np.log1p(A), 1/(1+A)
    one, zero = np.ones_like(t), np.zeros_like(t)
    d = np.stack((one, A, t, t*L, t*t, t*t*L, t**3), -1)
    dt = np.stack((zero, zero, one, L, 2*t, 2*t*L, 3*t*t), -1)/THETA_SCALE
    da = np.stack((zero, one, zero, t*La, zero, t*t*La, zero), -1)
    s = np.stack((one, t, L, t*L, t*t, t*t*L), -1)
    st = np.stack((zero, one, zero, L, 2*t, 2*t*L), -1)/THETA_SCALE
    sa = np.stack((zero, zero, La, t*La, zero, t*t*La), -1)
    return d, np.stack((dt, da), -1), s, np.stack((st, sa), -1)


class PositionModel:
    channels = 6

    def __init__(self, capacity=27, beta=None):
        if capacity not in (27, 37):
            raise ValueError("Only conditional27/37 are implemented")
        self.capacity, self.size = capacity, capacity
        self.width = 4 if capacity == 27 else 6
        self.beta = None if beta is None else np.asarray(beta, float)
        self.name = f"conditional{capacity}"
        self.intercepts = [0] + [7+j*self.width for j in range(5)]

    def design(self, x, r, derivatives=False):
        x, r = np.asarray(x, float), np.asarray(r, float)
        d, dd, s, ds = bases(x)
        shape = np.broadcast_shapes(x.shape[:-1], r.shape[:-2])
        r = np.broadcast_to(r, shape+(3, 2))
        out = np.zeros(shape+(6, self.size))
        jac = np.zeros(shape+(6, self.size, 2)) if derivatives else None
        # D_x,D_y,T11,T12,T21,T22; pair-major coordinate order.
        for group, axis, raxis in [(0, 0, None), (1, 1, None), (2, 0, 0),
                                   (3, 0, 1), (4, 1, 0), (5, 1, 1)]:
            sl = slice(0, 7) if group == 0 else slice(7+(group-1)*self.width,
                                                     7+group*self.width)
            b, db = (d, dd) if group == 0 else (s[..., :self.width], ds[..., :self.width, :])
            factor = np.ones(shape+(3,)) if raxis is None else r[..., :, raxis]
            out[..., axis::2, sl] = factor[..., :, None]*b[..., None, :]
            if derivatives:
                jac[..., axis::2, sl, :] = factor[..., :, None, None]*db[..., None, :, :]
        return (out, jac) if derivatives else out

    def predict(self, x, r, derivatives=False):
        if derivatives:
            b, db = self.design(x, r, True)
            return b @ self.beta, np.einsum("...cpz,p->...cz", db, self.beta)
        return self.design(x, r) @ self.beta

    def components(self, x):
        d, _, s, _ = bases(x)
        vals = [d@self.beta[:7]] + [s[..., :self.width] @
            self.beta[7+j*self.width:7+(j+1)*self.width] for j in range(5)]
        return np.stack(vals[:2], -1), np.stack(vals[2:], -1).reshape(np.shape(vals[0])+(2, 2))


class SummaryModel:
    """Fresh fold-local 13-coefficient two-channel control; theta scale 10.

    Same functional family as the frozen baseline, without loading its coefficients.
    """
    channels, size, name, intercepts = 2, 13, "two_channel13", [0, 7]

    def __init__(self, beta=None):
        self.beta = None if beta is None else np.asarray(beta, float)

    def design(self, x, r=None, derivatives=False):
        d, dd, s, ds = bases(x)
        # Baseline area family [1,t,t²,L,tL,t²L].
        order = [0, 1, 4, 2, 3, 5]
        out = np.zeros(np.shape(x)[:-1]+(2, 13))
        out[..., 0, :7], out[..., 1, 7:] = d, s[..., order]
        if not derivatives:
            return out
        jac = np.zeros(np.shape(x)[:-1]+(2, 13, 2))
        jac[..., 0, :7, :], jac[..., 1, 7:, :] = dd, ds[..., order, :]
        return out, jac

    def predict(self, x, r=None, derivatives=False):
        if derivatives:
            b, db = self.design(x, r, True)
            return b@self.beta, np.einsum("...cpz,p->...cz", db, self.beta)
        return self.design(x, r)@self.beta
