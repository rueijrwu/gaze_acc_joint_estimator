"""Forward derivatives of selected outward-interval endpoint branches.

Rounding chooses the enclosing endpoint value; its smooth algebra supplies
derivatives (the floating-point rounding operation itself is not differentiated).
Ties with different derivatives are reported, never certified as smooth.
"""
import numpy as np


class Jet:
    dimension = 19

    def __init__(self, value, gradient=None, hessian=None, order=1, nonsmooth=False):
        self.v = np.asarray(value, dtype=float)
        self.order = order
        self.g = np.zeros(self.v.shape + (self.dimension,)) if gradient is None else gradient
        self.h = (np.zeros(self.v.shape + (self.dimension, self.dimension))
                  if hessian is None and order == 2 else hessian)
        self.nonsmooth = np.broadcast_to(nonsmooth, self.v.shape)

    @classmethod
    def variable(cls, value, index, order):
        gradient = np.zeros(cls.dimension); gradient[index] = 1.
        return cls(value, gradient, order=order)

    def wrap(self, value):
        return value if isinstance(value, Jet) else Jet(value, order=self.order)

    def rounded(self, direction):
        return Jet(np.nextafter(self.v, direction), self.g, self.h,
                   self.order, self.nonsmooth)

    def __add__(self, value):
        b = self.wrap(value)
        return Jet(self.v+b.v, self.g+b.g,
                   self.h+b.h if self.order == 2 else None,
                   self.order, self.nonsmooth | b.nonsmooth)
    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.v, -self.g, -self.h if self.order == 2 else None,
                   self.order, self.nonsmooth)

    def __sub__(self, value): return self + -self.wrap(value)
    def __rsub__(self, value): return self.wrap(value) + -self

    def __mul__(self, value):
        b = self.wrap(value)
        g = self.g*b.v[..., None]+b.g*self.v[..., None]
        h = None
        if self.order == 2:
            outer = self.g[..., :, None]*b.g[..., None, :]
            h = (self.h*b.v[..., None, None]+b.h*self.v[..., None, None]
                 + outer+np.swapaxes(outer, -1, -2))
        return Jet(self.v*b.v, g, h, self.order, self.nonsmooth | b.nonsmooth)
    __rmul__ = __mul__

    def power(self, exponent):
        v = self.v**exponent
        first = exponent*self.v**(exponent-1)
        second = exponent*(exponent-1)*self.v**(exponent-2)
        h = None if self.order == 1 else (
            first[..., None, None]*self.h
            + second[..., None, None]*self.g[..., :, None]*self.g[..., None, :])
        return Jet(v, first[..., None]*self.g, h, self.order, self.nonsmooth)

    def __truediv__(self, value): return self*self.wrap(value).power(-1)


def select(candidates, minimum):
    """Select the same endpoint as min/max, tracking derivative-relevant ties."""
    shape = np.broadcast_shapes(*(c.v.shape for c in candidates))
    v = np.stack([np.broadcast_to(c.v, shape) for c in candidates])
    g = np.stack([np.broadcast_to(c.g, shape+(Jet.dimension,)) for c in candidates])
    chosen = v.argmin(axis=0) if minimum else v.argmax(axis=0)
    take = lambda a: np.take_along_axis(a, chosen[None, ..., *([None]*(a.ndim-v.ndim))], axis=0)[0]
    selected_v, selected_g = take(v), take(g)
    flags = np.stack([np.broadcast_to(c.nonsmooth, shape) for c in candidates])
    nonsmooth = take(flags).copy()
    # Treat numerical endpoint ties conservatively. Identical branches (e.g.
    # point intervals) are smooth and must not trigger false tie failures.
    tied = np.abs(v-selected_v) <= 16*np.finfo(float).eps*np.maximum(1., np.abs(selected_v))
    different = np.max(np.abs(g-selected_g), axis=-1) > 1e-10
    order = candidates[0].order; selected_h = None
    if order == 2:
        h = np.stack([np.broadcast_to(c.h, shape+(Jet.dimension, Jet.dimension)) for c in candidates])
        selected_h = take(h)
        different |= np.max(np.abs(h-selected_h), axis=(-2, -1)) > 1e-9
    nonsmooth |= np.any(tied & different, axis=0)
    return Jet(selected_v, selected_g, selected_h, order, nonsmooth)


class BranchInterval:
    def __init__(self, lower, upper=None, order=1):
        self.lo = lower if isinstance(lower, Jet) else Jet(lower, order=order)
        self.hi = self.lo if upper is None else (upper if isinstance(upper, Jet) else Jet(upper, order=order))

    def wrap(self, value):
        return value if isinstance(value, BranchInterval) else BranchInterval(value, order=self.lo.order)

    @classmethod
    def outward(cls, lo, hi): return cls(lo.rounded(-np.inf), hi.rounded(np.inf))

    def __add__(self, value):
        b = self.wrap(value); return self.outward(self.lo+b.lo, self.hi+b.hi)
    __radd__ = __add__
    def __neg__(self): return BranchInterval(-self.hi, -self.lo)
    def __sub__(self, value): return self + -self.wrap(value)
    def __rsub__(self, value): return self.wrap(value) + -self
    def __mul__(self, value):
        b = self.wrap(value)
        choices = [self.lo*b.lo, self.lo*b.hi, self.hi*b.lo, self.hi*b.hi]
        return self.outward(select(choices, True), select(choices, False))
    __rmul__ = __mul__
    def __truediv__(self, value):
        b = self.wrap(value)
        if np.any((b.lo.v <= 0)&(b.hi.v >= 0)):
            raise ValueError('interval denominator crosses zero')
        return self*self.outward(b.hi.power(-1), b.lo.power(-1))
    def square(self):
        choices = [self.lo*self.lo, self.hi*self.hi]
        lower = select(choices, True)
        crossing = (self.lo.v <= 0)&(self.hi.v >= 0)
        if crossing.any():
            lower = Jet(np.where(crossing, 0., lower.v),
                        np.where(crossing[..., None], 0., lower.g),
                        np.where(crossing[..., None, None], 0., lower.h) if lower.order == 2 else None,
                        lower.order, np.where(crossing, False, lower.nonsmooth))
        return self.outward(lower, select(choices, False))
