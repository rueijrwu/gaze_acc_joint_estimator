"""Independent capture shape fitting with device-resident CuPy arithmetic.

The optional fifth parameter is one constant relative barrel coefficient
within a capture. This model accepts no accommodation or demand labels.
"""
import numpy as np
from .p1_shape import centered


def inverse_radial(points, kappa):
    """Invert r + kappa*r**3 on its positive monotone branch."""
    points=np.asarray(points,dtype=float);kappa=np.asarray(kappa,dtype=float)
    rd=np.linalg.norm(points,axis=-1);k=kappa[...,None]
    with np.errstate(divide='ignore',invalid='ignore'):
        turning=np.sqrt(np.where(k<0,-1/(3*k),np.inf))
    point_valid=np.isfinite(rd)&np.isfinite(k)&((k>=0)|(rd<(2/3)*turning))
    low=np.zeros_like(rd);high=np.where(k<0,turning,rd)
    for _ in range(64):
        middle=(low+high)/2
        lower=(middle+k*middle**3)<rd
        low=np.where(lower,middle,low);high=np.where(lower,high,middle)
    root=np.where(k==0,rd,(low+high)/2)
    ratio=np.divide(root,rd,out=np.ones_like(root),where=rd>0)
    result=points*ratio[...,None]
    point_valid &= (1+3*k*root**2)>1e-8
    valid=np.all(point_valid,axis=-1)
    return np.where(valid[...,None,None],result,np.nan),valid


class CaptureShape:
    def __init__(self, gaze, reference, observed, weights, units, magnification=None, xp=np):
        self.xp = xp
        self.gaze = xp.asarray(gaze, dtype=xp.float64)
        self.b = xp.asarray(reference, dtype=xp.float64)
        self.x = xp.asarray(observed, dtype=xp.float64)
        self.w = xp.asarray(weights, dtype=xp.float64)
        self.units = xp.asarray(units, dtype=xp.float64)
        self.m = None if magnification is None else xp.asarray(magnification, dtype=xp.float64)
        self.r = xp.sqrt(xp.mean(xp.sum(self.b*self.b, axis=1)))
        self.rho2 = xp.sum(self.b*self.b, axis=1)/self.r**2
        self.t = self.gaze/self.units

    def host(self, a):
        return np.asarray(a) if self.xp is np else self.xp.asnumpy(a)

    def shape(self, coefficients, derivatives=False):
        xp=self.xp; k=xp.asarray(coefficients, dtype=xp.float64)
        tx,ty=self.t[:,0],self.t[:,1]
        lam=k[4] if len(coefficients)==5 else 0.
        pre=self.b*(1+lam*self.rho2)[:,None]
        stretch=xp.exp(k[0]*tx*tx-k[1]*ty*ty)
        factors=xp.stack((stretch,1/stretch),axis=1)[:,None,:]
        vx=tx[:,None]*pre[:,1]/self.r
        vy=ty[:,None]*pre[:,0]/self.r
        den=1+k[2]*vx+k[3]*vy
        f=pre*factors/den[...,None]
        c=f-f.mean(axis=1,keepdims=True)
        pc=pre-pre.mean(axis=0,keepdims=True)
        pre_radius=xp.sqrt(xp.mean(xp.sum(pc*pc,axis=1)))
        size=xp.sqrt(xp.mean(xp.sum(c*c,axis=2),axis=1))/pre_radius
        h=c/size[:,None,None]
        if not derivatives: return h,f.mean(axis=1),size
        df=[f*xp.stack((tx*tx,-tx*tx),axis=1)[:,None,:],
            f*xp.stack((-ty*ty,ty*ty),axis=1)[:,None,:],
            -f*(vx/den)[...,None],-f*(vy/den)[...,None]]
        if len(coefficients)==5:
            dp=self.b*self.rho2[:,None]
            dd=(k[2]*tx[:,None]*dp[:,1]+k[3]*ty[:,None]*dp[:,0])/self.r
            df.append(dp*factors/den[...,None]-f*(dd/den)[...,None])
        df=xp.stack(df,axis=-1); dc=df-df.mean(axis=1,keepdims=True)
        dlog=xp.sum(c[...,None]*dc,axis=(1,2))/xp.sum(c*c,axis=(1,2))[:,None]
        if len(coefficients)==5:
            dpc=dp-dp.mean(axis=0,keepdims=True)
            dlog[:,4]-=xp.sum(pc*dpc)/xp.sum(pc*pc)
        dh=dc/size[:,None,None,None]-h[...,None]*dlog[:,None,None,:]
        return h,f.mean(axis=1),size,dh

    def prediction(self, coefficients):
        xp=self.xp; h,_,_=self.shape(coefficients)
        m=self.m
        if m is None: m=xp.sum(self.x*h,axis=(1,2))/xp.sum(h*h,axis=(1,2))
        return self.host(m[:,None,None]*h),self.host(m)

    def evaluate(self, coefficients):
        xp=self.xp; h,_,_,dh=self.shape(coefficients,True)
        m=self.m
        if m is None: m=xp.sum(self.x*h,axis=(1,2))/xp.sum(h*h,axis=(1,2))
        error=self.x-m[:,None,None]*h
        cost=xp.sum(self.w*xp.sum(error*error,axis=(1,2)))
        grad=-2*xp.sum(self.w[:,None]*m[:,None]*xp.sum(error[...,None]*dh,axis=(1,2)),axis=0)
        packed=self.host(xp.concatenate((cost[None],grad)))
        if not np.isfinite(packed).all(): raise ValueError('Nonfinite capture objective')
        return float(packed[0]),packed[1:]

    def inverse(self, coefficients, magnification=None, observed=None, remove_barrel=True):
        _,mu,size=self.shape(coefficients); mu,size=self.host(mu),self.host(size)
        x=self.host(self.x) if observed is None else centered(observed)
        m=self.prediction(coefficients)[1] if magnification is None else np.asarray(magnification)
        t=self.host(self.t); k=np.asarray(coefficients); r=float(self.host(self.r))
        z=centered(x)*size[:,None,None]/m[:,None,None]+mu[:,None,:]
        a=k[0]*t[:,0]**2-k[1]*t[:,1]**2
        u=z/np.stack((np.exp(a),np.exp(-a)),axis=1)[:,None,:]
        den=1-k[3]*t[:,1,None]*u[...,0]/r-k[2]*t[:,0,None]*u[...,1]/r
        pre=u/den[...,None]
        valid=np.all(den>1e-8,axis=1)&(m>0)&np.isfinite(pre).all(axis=(1,2))
        if remove_barrel:
            kappa=np.full(len(x),(k[4] if len(k)==5 else 0.)/r**2)
            result,radial_valid=inverse_radial(pre,kappa); valid &= radial_valid
        else: result=pre
        return np.where(valid[:,None,None],result,np.nan),valid

    def domain(self, coefficients):
        k=np.asarray(coefficients); b=self.host(self.b); r=float(self.host(self.r)); t=self.host(self.t)
        lam=k[4] if len(k)==5 else 0.
        factor=1+lam*np.sum(b*b,axis=1)/r**2
        derivative=1+3*lam*np.sum(b*b,axis=1)/r**2
        pre=b*factor[:,None]
        den=1+k[2]*t[:,0,None]*pre[:,1]/r+k[3]*t[:,1,None]*pre[:,0]/r
        return {'min_radial_factor':float(factor.min()),'min_radial_derivative':float(derivative.min()),
                'min_keystone_denominator':float(den.min()),
                'valid':bool(factor.min()>0 and derivative.min()>0 and den.min()>0)}
