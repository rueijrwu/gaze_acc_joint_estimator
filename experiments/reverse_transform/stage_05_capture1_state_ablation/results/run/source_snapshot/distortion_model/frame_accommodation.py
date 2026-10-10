"""Frozen gaze/keystone/P1 scale, variable accommodation, inverse vertex cost."""
import numpy as np


class FrameAccommodation:
    def __init__(self, observed, reference, gaze, units, keystone, magnification,
                 exposure, expected, slope, reference_A, anchor_width=.25,
                 residual_scale=1., xp=np):
        self.xp=xp
        self.x=xp.asarray(observed,dtype=xp.float64)
        self.b=xp.asarray(reference,dtype=xp.float64)
        self.t=xp.asarray(gaze,dtype=xp.float64)/xp.asarray(units,dtype=xp.float64)
        self.k=xp.asarray(keystone,dtype=xp.float64)
        self.m=xp.asarray(magnification,dtype=xp.float64)
        self.e=xp.asarray(exposure,dtype=xp.int64)
        self.expected=xp.asarray(expected,dtype=xp.float64)
        self.slope=float(slope);self.reference_A=float(reference_A)
        self.anchor_width=float(anchor_width);self.residual_scale=float(residual_scale)
        self.strength=(residual_scale/anchor_width)**2
        self.r=xp.sqrt(xp.mean(xp.sum(self.b*self.b,axis=1)))
        self.r2=xp.sum(self.b*self.b,axis=1)
        self.count=xp.bincount(self.e,minlength=len(expected))
        if bool(xp.any(self.count==0)):raise ValueError('Empty fixation')
        self.n=len(observed);self.nref=self.n/len(expected)
        self.q=self.nref/self.count[self.e]
        a=self.k[:,0]*self.t[:,0]**2-self.k[:,1]*self.t[:,1]**2
        self.stretch=xp.stack((xp.exp(a),xp.exp(-a)),axis=1)[:,None,:]
        self.qx=self.k[:,3]*self.t[:,1]/self.r
        self.qy=self.k[:,2]*self.t[:,0]/self.r

    def host(self,a):return np.asarray(a) if self.xp is np else self.xp.asnumpy(a)

    def kappa(self,A):return self.slope*(A-self.reference_A)

    def geometry(self,A,derivative=False,observed=None):
        xp=self.xp;A=xp.asarray(A,dtype=xp.float64);kap=self.kappa(A)
        pre=self.b[None,:,:]*(1+kap[:,None]*self.r2[None,:])[:,:,None]
        denominator=1+self.qx[:,None]*pre[:,:,0]+self.qy[:,None]*pre[:,:,1]
        f=pre*self.stretch/denominator[:,:,None]
        c=f-f.mean(axis=1,keepdims=True);pc=pre-pre.mean(axis=1,keepdims=True)
        size=xp.sqrt(xp.sum(c*c,axis=(1,2))/xp.sum(pc*pc,axis=(1,2)))
        mu=f.mean(axis=1)
        x=self.x if observed is None else xp.asarray(observed,dtype=xp.float64)
        z=x*size[:,None,None]/self.m[:,None,None]+mu[:,None,:]
        u=z/self.stretch
        inverse_den=1-self.qx[:,None]*u[:,:,0]-self.qy[:,None]*u[:,:,1]
        v=u/inverse_den[:,:,None];rd=xp.sqrt(xp.sum(v*v,axis=2))
        # For kappa<0 the cubic inverse exists only before its turning point.
        turning=1/xp.sqrt(xp.maximum(-3*kap,1e-300))
        valid=(xp.all(denominator>1e-8,axis=1)&xp.all(inverse_den>1e-8,axis=1)&
               xp.all(1+3*kap[:,None]*self.r2>1e-8,axis=1)&
               xp.all(1+kap[:,None]*self.r2>0,axis=1)&
               xp.all((kap[:,None]>=0)|(rd<(2/3)*turning[:,None]),axis=1)&
               xp.isfinite(v).all(axis=(1,2))&(self.m>0))
        if not derivative:return v,rd,kap,valid,c/size[:,None,None]*self.m[:,None,None]
        dp=self.b[None,:,:]*(self.slope*self.r2)[None,:,None]
        dd=self.qx[:,None]*dp[:,:,0]+self.qy[:,None]*dp[:,:,1]
        df=dp*self.stretch/denominator[:,:,None]-f*(dd/denominator)[:,:,None]
        dc=df-df.mean(axis=1,keepdims=True);dpc=dp-dp.mean(axis=1,keepdims=True)
        ds=size*(xp.sum(c*dc,axis=(1,2))/xp.sum(c*c,axis=(1,2))-
                 xp.sum(pc*dpc,axis=(1,2))/xp.sum(pc*pc,axis=(1,2)))
        dz=x*ds[:,None,None]/self.m[:,None,None]+df.mean(axis=1)[:,None,:]
        du=dz/self.stretch
        did=-self.qx[:,None]*du[:,:,0]-self.qy[:,None]*du[:,:,1]
        dv=du/inverse_den[:,:,None]-v*(did/inverse_den)[:,:,None]
        drd=xp.sum(v*dv,axis=2)/xp.maximum(rd,1e-300)
        return v,rd,kap,valid,dv,drd

    def recover(self,A,derivative=False,observed=None):
        xp=self.xp;geo=self.geometry(A,derivative,observed)
        v,rd,kap,valid=geo[:4];k=kap[:,None]
        turning=1/xp.sqrt(xp.maximum(-3*k,1e-300))
        low=xp.zeros_like(rd);high=xp.where(k<0,turning,rd)
        # Invalid observations never enter a fit: caller supplies feasible A.
        # Mask them here for honest baseline reporting and domain construction.
        for _ in range(55):
            middle=(low+high)/2;lower=middle+k*middle**3<rd
            low=xp.where(lower,middle,low);high=xp.where(lower,high,middle)
        r=xp.where(k==0,rd,(low+high)/2)
        factor=xp.where(rd>0,r/xp.maximum(rd,1e-300),1.)
        result=v*factor[:,:,None]
        valid &= xp.all(1+3*k*r*r>1e-8,axis=1)
        result=xp.where(valid[:,None,None],result,xp.nan)
        if not derivative:return result,valid
        dv,drd=geo[4:]
        dr=(drd-self.slope*r**3)/(1+3*k*r*r)
        dresult=dv*factor[:,:,None]+v*((dr-factor*drd)/xp.maximum(rd,1e-300))[:,:,None]
        return result,valid,dresult

    def frame_cost(self,A):
        xp=self.xp;back,valid,derivative=self.recover(A,True)
        error=back-self.b[None,:,:]
        cost=xp.sum(error*error,axis=(1,2))/3
        grad=2*xp.sum(error*derivative,axis=(1,2))/3
        return cost,grad,valid

    def means(self,A):
        return self.xp.bincount(self.e,weights=A,minlength=len(self.expected))/self.count

    def evaluate(self,A):
        xp=self.xp;A=xp.asarray(A,dtype=xp.float64)
        cost,gradient,valid=self.frame_cost(A)
        if not bool(xp.all(valid)):raise ValueError('Optimizer proposed noninvertible accommodation')
        delta=self.means(A)-self.expected
        total=xp.sum(self.q*cost)+self.nref*self.strength*xp.sum(delta*delta)
        grad=self.q*(gradient+2*self.strength*delta[self.e])
        packed=self.host(xp.concatenate((total[None],grad)))
        if not np.isfinite(packed).all():raise ValueError('Nonfinite objective/gradient')
        return float(packed[0]),packed[1:]

    def feasible_bounds(self,lower=0.,upper=6.):
        xp=self.xp;lo=xp.full(self.n,lower,dtype=xp.float64);hi=xp.full(self.n,upper,dtype=xp.float64)
        if not bool(xp.all(self.geometry(lo)[3])):raise ValueError('Some complete frames have no valid inverse at lower A bound')
        # Check the assumed contiguous valid branch on a fixed grid first.
        lost=xp.zeros(self.n,dtype=bool)
        for value in np.linspace(lower,upper,25):
            valid=self.geometry(xp.full(self.n,value))[3]
            if bool(xp.any(lost&valid)):raise ValueError('Noncontiguous inverse domain; cannot use interval bounds')
            lost |= ~valid
        upper_valid=self.geometry(hi)[3];a=lo.copy();b=hi.copy()
        for _ in range(40):
            mid=(a+b)/2;valid=self.geometry(mid)[3]
            a=xp.where(valid,mid,a);b=xp.where(valid,b,mid)
        hi=xp.where(upper_valid,hi,xp.maximum(lo,a-1e-6))
        if not bool(xp.all(self.geometry(hi)[3])):raise ValueError('Invalid computed inverse bounds')
        return self.host(lo),self.host(hi)

    def certificate(self,A,lower,upper):
        A=np.asarray(A);_,gradient=self.evaluate(A)
        projected=gradient.copy()
        constrained=((A<=lower+1e-7)&(gradient>0))|((A>=upper-1e-7)&(gradient<0))
        projected[constrained]=0
        xp=self.xp;device=xp.asarray(A)
        # Differentiate each independent frame term, holding means out of it.
        h=1e-4
        plus=xp.minimum(device+h,xp.asarray(upper));minus=xp.maximum(device-h,xp.asarray(lower))
        curvature=self.host((self.frame_cost(plus)[1]-self.frame_cost(minus)[1])/(plus-minus))
        free=~((A<=lower+1e-7)|(A>=upper-1e-7))
        minimum=float(curvature[free].min()) if free.any() else None
        return {'projected_gradient_inf':float(np.max(abs(projected))),
                'free_frame_curvature_min':minimum,
                'lower_bound_frames':int(np.sum(A<=lower+1e-7)),
                'upper_bound_frames':int(np.sum(A>=upper-1e-7)),
                'inverse_domain_limited_frames':int(np.sum(upper<6.-1e-7)),
                'stationary':bool(np.max(abs(projected))<1e-5 and (minimum is None or minimum>0))},curvature
