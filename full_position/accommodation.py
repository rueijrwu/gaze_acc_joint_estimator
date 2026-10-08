"""Fixed shifted-power accommodation laws, in physical degrees and diopters.

This adapter leaves the legacy log models unchanged. No temporal penalty or
within-fixation constant-state constraint is introduced.
"""
from dataclasses import dataclass, asdict
import numpy as np
from .model import THETA_SCALE, STATE_SCALE, LOWER, UPPER

CANDIDATES = {"ar27_log": 0., "ar27_sqrt": .5, "ar27_linear": 1., "ar27_quadratic": 2.}
COEFFICIENT_ORDER = "Dx[1,a,t,tphi,t2,t2phi,t3]; Dy,T11,T12,T21,T22[1,t,phi,tphi]"


def response_arrays(A, exponent, xp=np, basis="shifted_boxcox"):
    """Array response algebra shared by host and FP64 device solvers."""
    if basis == "literal_power":
        L = xp.log(A)
        return (xp.exp(exponent*L), exponent*xp.exp((exponent-1)*L),
                exponent*(exponent-1)*xp.exp((exponent-2)*L))
    if basis != "shifted_boxcox":
        raise ValueError("Unknown accommodation response basis")
    L = xp.log1p(A)
    return (L if exponent == 0 else xp.expm1(exponent*L)/exponent,
            xp.exp((exponent-1)*L), (exponent-1)*xp.exp((exponent-2)*L))


def response(A, exponent, basis="shifted_boxcox"):
    """Return phi, dphi/dA and d²phi/dA²; A*=1 D.

    The mathematical domain is A>-1 D. The inverse separately enforces its
    physical computational bounds. Invalid input is rejected, never clipped.
    """
    A = np.asarray(A, float)
    if isinstance(exponent, (bool, np.bool_)):
        raise ValueError("A finite real response exponent is required")
    try:
        exponent = float(exponent)
    except (ValueError,TypeError) as exc:
        raise ValueError("A finite real response exponent is required") from exc
    if not np.isfinite(exponent):
        raise ValueError("A finite real response exponent is required")
    if not np.isfinite(A).all() or np.any(A <= (0. if basis == "literal_power" else -1.)):
        raise ValueError("Response requires finite A with 1+A/A* > 0")
    return response_arrays(A, exponent, np, basis)


def bases(x, exponent, hessians=True, basis="shifted_boxcox"):
    x = np.asarray(x, float)
    if x.ndim < 1 or x.shape[-1] != 2 or not np.isfinite(x).all():
        raise ValueError("States must be finite (...,2) physical theta,A values")
    t, a = x[..., 0]/THETA_SCALE, x[..., 1]
    phi, pa, paa = response(a, exponent, basis)
    o, z = np.ones_like(t), np.zeros_like(t)
    d = np.stack((o, a, t, t*phi, t*t, t*t*phi, t**3), -1)
    dt = np.stack((z,z,o,phi,2*t,2*t*phi,3*t*t), -1)/THETA_SCALE
    da = np.stack((z,o,z,t*pa,z,t*t*pa,z), -1)
    s = np.stack((o,t,phi,t*phi), -1)
    st = np.stack((z,o,z,phi), -1)/THETA_SCALE
    sa = np.stack((z,z,pa,t*pa), -1)
    if not hessians:
        return d, np.stack((dt,da),-1), s, np.stack((st,sa),-1)
    def hessian(tt, ta, aa):
        tt, ta, aa = np.stack(tt,-1)/THETA_SCALE**2, np.stack(ta,-1)/THETA_SCALE, np.stack(aa,-1)
        return np.stack((np.stack((tt,ta),-1), np.stack((ta,aa),-1)), -2)
    dh = hessian((z,z,z,z,2*o,2*phi,6*t), (z,z,z,pa,z,2*t*pa,z), (z,z,z,t*paa,z,t*t*paa,z))
    sh = hessian((z,z,z,z), (z,z,z,pa), (z,z,paa,t*paa))
    return d, np.stack((dt,da),-1), s, np.stack((st,sa),-1), dh, sh


@dataclass(frozen=True)
class ResponseDefinition:
    exponent: float
    family: str = "conditional_power_response_v1"
    basis: str = "shifted_boxcox"
    accommodation_unit: str = "diopter"
    accommodation_scale_D: float = 1.
    direct_linear_A_in_Dx: bool = True
    capacity: int = 27


class PowerResponseModel:
    capacity = size = 27
    width = 4
    channels = 6
    intercepts = np.array([0,7,11,15,19,23])
    def __init__(self, exponent, beta=None):
        if isinstance(exponent, (bool,np.bool_)) or exponent not in CANDIDATES.values():
            raise ValueError("Use one of the four declared response exponents")
        self._response = ResponseDefinition(float(exponent))
        self.name = next(k for k,v in CANDIDATES.items() if v == exponent)
        self.beta = np.zeros(self.size) if beta is None else np.asarray(beta,float).copy()
        if self.beta.shape != (self.size,) or not np.isfinite(self.beta).all():
            raise ValueError("Power response requires 27 finite coefficients")

    @property
    def exponent(self):
        return self._response.exponent

    @property
    def response_definition(self):
        return self._response

    def metadata(self):
        return asdict(self._response)

    def _bases(self, x, hessians=True):
        return bases(x, self.exponent, hessians, getattr(self,"response_basis","shifted_boxcox"))

    def design(self, x, r, derivatives=False):
        d, dd, s, ds = self._bases(x,hessians=False)
        r = np.asarray(r,float)
        if r.shape[-2:] != (3,2) or not np.isfinite(r).all():
            raise ValueError("P1 context must contain three finite normalized points")
        shape = np.broadcast_shapes(d.shape[:-1], r.shape[:-2])
        d,dd,s,ds,r = (np.broadcast_to(v,shape+tail) for v,tail in
            ((d,(7,)),(dd,(7,2)),(s,(4,)),(ds,(4,2)),(r,(3,2))))
        H = np.zeros(shape+(6,27))
        H[...,0::2,:7], H[...,1::2,7:11] = d[...,None,:], s[...,None,:]
        if derivatives:
            J = np.zeros(shape+(6,27,2))
            J[...,0::2,:7,:], J[...,1::2,7:11,:] = dd[...,None,:,:], ds[...,None,:,:]
        for k in range(4):
            axis, component, start = k//2, k%2, 11+4*k
            H[...,axis::2,start:start+4] = s[...,None,:]*r[...,component,None]
            if derivatives:
                J[...,axis::2,start:start+4,:] = ds[...,None,:,:]*r[...,component,None,None]
        return (H,J) if derivatives else H

    def predict(self,x,r,derivatives=False):
        if derivatives:
            H,J = self.design(x,r,True)
            return H@self.beta, np.einsum("...cbk,b->...ck",J,self.beta)
        return self.design(x,r)@self.beta

    def components(self,x):
        d,_,s,_ = self._bases(x,hessians=False)
        D = np.stack((d@self.beta[:7],s@self.beta[7:11]),-1)
        T = (s@self.beta[11:].reshape(4,4).T).reshape(s.shape[:-1]+(2,2))
        return D,T

    def state_hessian(self,x,r):
        _,_,_,_,dh,sh = self._bases(x)
        D = np.stack((np.einsum("...buv,b->...uv",dh,self.beta[:7]),
                      np.einsum("...buv,b->...uv",sh,self.beta[7:11])),-3)
        T = np.einsum("...buv,kb->...kuv",sh,self.beta[11:].reshape(4,4))
        T = T.reshape(sh.shape[:-3]+(2,2,2,2))
        value = D[...,None,:,:,:]+np.einsum("...ijuv,...pj->...piuv",T,np.asarray(r,float))
        return value.reshape(value.shape[:-4]+(6,2,2))


def rescale(model, states, gaze_gain, accommodation_gain):
    """Exact optical gauge identity. Training diagnosis only, never eval rescue."""
    g,h = float(gaze_gain),float(accommodation_gain)
    if not np.isfinite([g,h]).all() or min(g,h) <= 0:
        raise ValueError("Gauge gains must be finite and positive")
    # b(new state) = b(old state) @ M; recover beta' by solving M beta'=beta.
    x = np.asarray(states,float)
    out = x.copy(); out[...,0] *= g; out[...,1] = h*(1+x[...,1])-1
    # Use exact closure coefficients, avoiding a data-dependent least-squares fit.
    f = h**model.exponent
    c = np.log(h) if model.exponent == 0 else np.expm1(model.exponent*np.log(h))/model.exponent
    Md = np.zeros((7,7)); Ms = np.zeros((4,4))
    Md[0,0]=1; Md[0,1]=h-1; Md[1,1]=h; Md[2,2]=g
    Md[2,3]=g*c; Md[3,3]=g*f; Md[4,4]=g*g
    Md[4,5]=g*g*c; Md[5,5]=g*g*f; Md[6,6]=g**3
    Ms[0,0]=1; Ms[1,1]=g; Ms[0,2]=c; Ms[2,2]=f; Ms[1,3]=g*c; Ms[3,3]=g*f
    beta = np.concatenate((np.linalg.solve(Md,model.beta[:7]),
                           np.linalg.solve(Ms,model.beta[7:].reshape(5,4).T).T.ravel()))
    return PowerResponseModel(model.exponent,beta),out


def linear_accommodation(model, r, retained_v, retained_cov, indices, theta_deg):
    """Independent fixed-gaze check for lambda=1; no branch selection shortcut."""
    if model.exponent != 1.:
        raise ValueError("Analytic conditional accommodation requires lambda=1")
    from .noise import whitening
    indices = np.asarray(indices,int)
    c = model.predict([theta_deg,0.],r)[indices]
    b = model.predict([theta_deg,1.],r)[indices]-c
    W = whitening(np.asarray(retained_cov,float))
    wb,wy = W@b,W@(np.asarray(retained_v,float)-c)
    denominator = float(wb@wb)
    if not np.isfinite(denominator):
        raise ValueError("Invalid retained accommodation information")
    if denominator <= np.finfo(float).tiny:
        return dict(observable=False,A_D=None,denominator=denominator)
    A = float(np.clip((wb@wy)/denominator,LOWER[1],UPPER[1]))
    return dict(observable=True,A_D=A,denominator=denominator)
