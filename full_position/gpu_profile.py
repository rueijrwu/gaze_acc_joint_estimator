"""GPU variable projection; host matrices are copied only for verification."""
import numpy as np
from . import calibrate
from .accommodation import PowerResponseModel, response_arrays
from .model import STATE_SCALE

ORIGINAL_UPDATE=calibrate.ProfiledProblem.update


class HostView:
    """Lazy immutable NumPy view for the original independent fit checks."""
    def __init__(self,value,xp):
        self.value,self.xp,self.cached=value,xp,None
        self.shape=value.shape
    def __array__(self,dtype=None,copy=None):
        if self.cached is None:
            self.cached=self.xp.asnumpy(self.value)
            self.cached.setflags(write=False)
        out=np.asarray(self.cached,dtype=dtype)
        return out.copy() if copy else out
    @property
    def T(self): return np.asarray(self).T
    def __matmul__(self,other): return np.asarray(self)@other
    def __pow__(self,power): return np.asarray(self)**power
    def __getitem__(self,key): return np.asarray(self)[key]


def update(problem,z):
    if not isinstance(problem.model,PowerResponseModel):
        return ORIGINAL_UPDATE(problem,z)
    z=np.asarray(z).ravel()
    previous=getattr(problem,"_gpu_profile",None)
    if (problem.last_z is not None and np.array_equal(z,problem.last_z)
            and previous is not None and np.array_equal(z,previous["z"])): return
    import cupy as xp
    from cupyx.scipy.linalg import solve_triangular
    problem.last_z=z.copy(); problem.x=z.reshape(-1,2)*STATE_SCALE
    cache=getattr(problem,"_gpu_fixed",None)
    if cache is None:
        cache=dict(r=xp.asarray(problem.r),weights=xp.asarray(problem.weights),
                   scale=xp.asarray(problem.column_scale),b=xp.asarray(problem.b),
                   groups=xp.asarray(problem.groups),counts=xp.asarray(problem.counts),
                   anchors=xp.asarray(problem.anchors),anchor_scales=xp.asarray(problem.anchor_scales))
        problem._gpu_fixed=cache
    # Contract derivatives of the eleven shared basis terms with sparse
    # coefficient templates, rather than calling the NumPy model on each step.
    x=xp.asarray(problem.x); t=x[:,0]/10; a=x[:,1]
    phi,pa,_=response_arrays(a,problem.model.exponent,xp,getattr(problem.model,"response_basis","shifted_boxcox"))
    one,zero=xp.ones_like(t),xp.zeros_like(t)
    d=xp.stack((one,a,t,t*phi,t*t,t*t*phi,t**3),-1)
    dt=xp.stack((zero,zero,one,phi,2*t,2*t*phi,3*t*t),-1)/10
    da=xp.stack((zero,one,zero,t*pa,zero,t*t*pa,zero),-1)
    s=xp.stack((one,t,phi,t*phi),-1)
    st=xp.stack((zero,one,zero,phi),-1)/10
    sa=xp.stack((zero,zero,pa,t*pa),-1)
    dd,ds=xp.stack((dt,da),-1),xp.stack((st,sa),-1)
    A=xp.zeros((problem.n,6,27)); dA=xp.zeros((problem.n,6,27,2))
    A[:,0::2,:7]=d[:,None]; A[:,1::2,7:11]=s[:,None]
    dA[:,0::2,:7]=dd[:,None]; dA[:,1::2,7:11]=ds[:,None]
    r=cache["r"]
    for k in range(4):
        axis,component,start=k//2,k%2,11+4*k
        A[:,axis::2,start:start+4]=s[:,None,:]*r[:,:,component,None]
        dA[:,axis::2,start:start+4]=ds[:,None,:,:]*r[:,:,component,None,None]
    C=xp.einsum("nij,njpz->nipz",cache["weights"],dA)*xp.asarray(STATE_SCALE)
    Bdata=xp.einsum("nij,njp->nip",cache["weights"],A).reshape(-1,27)
    B=xp.concatenate((Bdata,xp.diag(xp.asarray(problem.penalty)),xp.asarray(problem.curvature_penalty)),axis=0)
    Q,R=xp.linalg.qr(B/cache["scale"],mode="reduced")
    if float(xp.min(xp.abs(xp.diag(R))))<1e-12:
        raise ValueError("Unidentifiable coefficient design under declared prior")
    beta=solve_triangular(R,Q.T@cache["b"])/cache["scale"]
    residual=B@beta-cache["b"]
    S=xp.einsum("ncpz,p->ncz",C,beta)
    D=xp.einsum("ncpz,nc->nzp",C,residual[:problem.n*problem.c].reshape(problem.n,problem.c)).reshape(2*problem.n,27)
    problem._gpu_profile=dict(B=B,S=S,D=D,R=R,scale=cache["scale"],z=z.copy())
    problem.B,problem.C,problem.S=HostView(B,xp),HostView(C,xp),HostView(S,xp)
    problem.Q=None; problem.R=xp.asnumpy(R); problem.beta=xp.asnumpy(beta)
    problem.optical_residual=xp.asnumpy(residual)
    anchor=(problem.mean_states(problem.x)-problem.anchors)/problem.anchor_scales/np.sqrt(problem.k)
    problem.residual=np.r_[problem.optical_residual,anchor.ravel()]


def enable(): calibrate.ProfiledProblem.update=update
def disable(): calibrate.ProfiledProblem.update=ORIGINAL_UPDATE
