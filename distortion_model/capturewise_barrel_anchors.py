"""Independent same-gaze radial calibration and keystone-first joint states."""
import numpy as np
from scipy.optimize import minimize
from .capturewise_reverse import forward as p1_forward,host,equal_group_weights
from .capturewise_keystone_first import forward,inverse
from .capturewise_states import StateProblem,A_REF
from .p1_joint import template


def solve_small(evaluate,p0,scales,bounds,starts=3,maxiter=500,seed=20261024):
    """Strict certified multistart small solve with domain-step retries."""
    rng=np.random.default_rng(seed);fits=[]
    for start in range(starts):
        p=p0.copy()
        if start:p=np.clip(p+rng.normal(size=len(p))*scales*.15,bounds[:,0],bounds[:,1])
        if not np.isfinite(evaluate(p)[0]):p=p0.copy()
        initial=p.copy();history=[]
        for factor in (1.,.1,.01,.001):
            unit=scales*factor
            opt=minimize(lambda u:(lambda z:(z[0],z[1]*unit))(evaluate(u*unit)),p/unit,jac=True,method='L-BFGS-B',bounds=bounds/unit[:,None],options=dict(maxiter=maxiter,ftol=1e-14,gtol=1e-7*factor,maxls=40))
            candidate=opt.x*unit;lc,gc=evaluate(candidate);lp,gp=evaluate(p)
            if np.isfinite(lc) and (lc<lp or (lc<=lp+1e-12 and np.max(abs(gc*scales))<np.max(abs(gp*scales)))):p=candidate
            loss,gradient=evaluate(p);pg=gradient*scales;tol=1e-7*scales
            pg[((p<=bounds[:,0]+tol)&(pg>0))|((p>=bounds[:,1]-tol)&(pg<0))]=0
            norm=float(np.max(abs(pg)));history.append(dict(coordinate_factor=factor,loss=loss,projected_gradient_scaled_inf=norm,optimizer_success=bool(opt.success),message=str(opt.message)))
            if norm<1e-5:break
        fits.append(dict(params=p,initial_params=initial,loss=loss,stationary=bool(np.isfinite(loss) and norm<1e-5),projected_gradient_scaled_inf=norm,gradient=gradient,history=history,bound_contacts=np.isclose(p,bounds[:,0],atol=tol,rtol=0)|np.isclose(p,bounds[:,1],atol=tol,rtol=0)))
    certified=[f for f in fits if f['stationary']]
    if not certified:raise RuntimeError('No wholly domain-valid stationary calibration start')
    return min(certified,key=lambda f:f['loss']),fits


class GazeBarrelCalibration:
    """Five independent centered latent triangles with finite native priors.

    Mean sufficient statistics are exactly equivalent to equally weighted
    scale-corrected frame losses; within-cell variance is retained explicitly.
    """
    def __init__(self,data,g,xp=np,sigma_reference_px=1.):
        self.xp=xp;self.sigma=float(sigma_reference_px)
        if not np.isfinite(self.sigma) or self.sigma<=0:raise ValueError('Positive finite native reference width required')
        X=np.asarray(data['X4'])/np.asarray(g)[:,None,None]
        self.means=np.empty((4,5,3,2));self.counts=np.zeros((4,5),int);variance=[]
        for c in range(4):
            for f in range(5):
                m=(data['capture']==c+1)&(data['fixation']==f)
                if not m.any():raise ValueError('All20 training cells required for per-gaze anchors')
                self.counts[c,f]=m.sum();self.means[c,f]=X[m].mean(axis=0);variance.append(np.mean(np.sum((X[m]-self.means[c,f])**2,axis=2)))
        self.variance=float(np.mean(variance));self.references=self.means[0].copy();self.references-=self.references.mean(axis=1,keepdims=True)
        self.radii=np.sqrt(np.mean(np.sum(self.references**2,axis=2),axis=1))
        self.p0=np.zeros(35);self.scales=np.r_[np.full(15,.05),np.tile([.01,.1,.1,.1],5)]
        self.bounds=np.array([[-.25,.25]]*15+([[-.2,.2]]+[[-.75,.75]]*3)*5)

    def unpack(self,p):
        k=np.zeros((4,5));k[1:]=p[:15].reshape(3,5)/self.radii[None,:]**2
        Bs=[];Ds=[]
        for f in range(5):
            q=p[15+4*f:19+4*f];B,D=template(self.references[f],q[1:],True);B=np.exp(q[0])*B
            Bs.append(B);Ds.append(np.concatenate((B[...,None],np.exp(q[0])*D),axis=-1))
        return k,np.array(Bs),np.array(Ds)

    def evaluate(self,p,details=False):
        xp=self.xp;k,B,dB=self.unpack(p);BB=xp.asarray(np.tile(B,(4,1,1)));kk=xp.asarray(k.ravel());obs=xp.asarray(self.means.reshape(20,3,2))
        ev=forward(BB,xp.zeros(20),xp.zeros(3),0.,1.,1.,kk,xp,True)
        singular=np.linalg.svd(B,compute_uv=False);noncollinear=bool(np.all(singular[:,1]>1e-10*singular[:,0]))
        if not bool(xp.all(ev['valid'])) or not noncollinear:return (float('inf'),np.zeros(35)) if not details else dict(loss=float('inf'),valid=host(ev['valid'],xp))
        residual=ev['prediction']-obs;prior=B-self.references
        data=float(host(xp.sum(residual**2)/60,xp))+self.variance;reference=float(np.sum(prior**2)/(15*self.sigma**2));gradient=np.zeros(35)
        kg=host(xp.sum(residual*ev['dkappa'],axis=(1,2))/30,xp).reshape(4,5)
        gradient[:15]=(kg[1:]/self.radii[None,:]**2).ravel()
        raw=xp.einsum('njab,njbp->njap',ev['point_jacobian'],xp.asarray(np.tile(dB,(4,1,1,1))))
        centered=raw-raw.mean(axis=1,keepdims=True)
        bg=host(xp.einsum('nja,njap->np',residual,centered)/30,xp).reshape(4,5,4).sum(axis=0)
        bg+=2*np.einsum('fja,fjap->fp',prior,dB)/(15*self.sigma**2)
        gradient[15:]=bg.ravel()
        if details:return dict(loss=data+reference,data=data,reference_prior=reference,within_cell_variance=self.variance,kappa=k,latent_templates=B,latent_derivatives=dB,predicted_means=host(ev['prediction'],xp).reshape(4,5,3,2),valid=host(ev['valid'],xp),radial_derivative_min=float(host(ev['radial_derivative'].min(),xp)))
        return data+reference,gradient

    def data_rank(self,p):
        """Per-gaze native mean data rank; excludes the reference prior."""
        xp=self.xp;k,B,dB=self.unpack(p);reports=[]
        for f in range(5):
            ev=forward(B[f],xp.zeros(4),xp.zeros(3),0.,1.,1.,k[:,f],xp,True)
            J=xp.zeros((4,3,2,7))
            for c in range(1,4):J[c,:,:,c-1]=ev['dkappa'][c]/self.radii[f]**2
            raw=xp.einsum('njab,jbp->njap',ev['point_jacobian'],xp.asarray(dB[f]));J[:,:,:,3:]=raw-raw.mean(axis=1,keepdims=True)
            units=np.r_[np.full(3,.05),.01,np.full(3,.1)];J*=xp.asarray(units)
            gram=host(xp.einsum('njap,njaq->pq',J,J)/12,xp);eigen,modes=np.linalg.eigh(gram);sv=np.sqrt(np.maximum(eigen,0))[::-1]
            reports.append(dict(fixation=f,dimension=7,data_only_scaled_singular_values=sv,rank_at_relative_1e_minus6=int(np.sum(sv>sv[0]*1e-6)),weakest_scaled_mode=modes[:,0],method='Four capture mean triangles at this gaze; centered latent size+shape and3 relative k, no native reference prior'))
        return reports

    def fit(self,starts=3,maxiter=500,seed=20261024):
        best,allstarts=solve_small(self.evaluate,self.p0,self.scales,self.bounds,starts,maxiter,seed)
        return dict(best=best,starts=allstarts,**self.evaluate(best['params'],True))


def fit_new_keystone(data,g,kappa_targets,B,R,p3,xp=np,starts=3,maxiter=500):
    """Forward new-order K initializer only; no change to joint optical priors."""
    c=np.asarray(data['capture'],int)-1;f=np.asarray(data['fixation'],int);theta=xp.asarray(data['theta0']);g=xp.asarray(g);X=xp.asarray(data['X4']);w=xp.asarray(equal_group_weights(data['group'],data['capture']));cc=xp.asarray(c);rr=xp.asarray(R)[cc];kk=xp.asarray(kappa_targets[c,f])
    p0=np.r_[np.asarray(p3['alpha']).ravel(),float(p3['offset'])];scales=np.r_[np.full(12,.05),10.];bounds=np.array([[-.25,.25],[-.2,.2],[-.2,.2]]*4+[[-120.,120.]])
    def evaluate(p):
        ev=forward(B,theta,xp.asarray(p[:12].reshape(4,3))[cc],p[12],rr,g,kk,xp,True)
        if not bool(xp.all(ev['valid'])):return float('inf'),np.zeros(13)
        r=ev['prediction']-X;row=2*xp.einsum('n,nja,njap->np',w,r,ev['dalpha']);grad=xp.concatenate([row[cc==j].sum(axis=0) for j in range(4)]+[xp.asarray([-2*xp.einsum('n,nja,nja->',w,r,ev['dtheta'])])])
        return float(host(xp.einsum('n,nja,nja->',w,r,r),xp)),host(grad,xp)
    best,allstarts=solve_small(evaluate,p0,scales,bounds,starts,maxiter)
    return dict(alpha=best['params'][:12].reshape(4,3),offset=best['params'][12],best=best,starts=allstarts,interpretation='Initialization only; joint optical anchors remain original Stage03/04 raw coefficient targets')


class KeystoneFirstStateProblem(StateProblem):
    def __init__(self,*args,kappa_targets=None,sigma_kappa=None,barrel_anchor=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.barrel_anchor=bool(barrel_anchor);self.kappa_targets_all=np.asarray(kappa_targets) if kappa_targets is not None else np.zeros((4,5))
        self.sigma_kappa=float(abs(self.s0)*.25 if sigma_kappa is None else sigma_kappa)
        if self.kappa_targets_all.shape!=(4,5) or not np.isfinite(self.kappa_targets_all).all() or not np.isfinite(self.sigma_kappa) or self.sigma_kappa<=0:raise ValueError('Finite4x5 barrel targets and positive width required')
        self.kappa_targets=self.xp.asarray([self.kappa_targets_all[self.capture[np.flatnonzero(self.group_np==i)[0]],int(self.data['fixation'][np.flatnonzero(self.group_np==i)[0]])] for i in range(self.G)])

    def predictions(self,theta,A,q,derivatives=False):
        xp=self.xp;a,offset,s=self.unpack(q)
        f1=p1_forward(self.B1,theta,self.p1[:3],self.p1[3],self.R1,self.g,xp=xp,derivatives=derivatives)
        B,dB,valid=self.template_geometry(q,True);f4=forward(B,theta,a,offset,self.R4,self.g,s*(A-A_REF),xp,derivatives)
        f4['valid'] &= valid;f4.update(template=B,template_valid=valid)
        Z=f4['Z'][:,self.vertices4];f4['selected_prediction']=self.g[:,None,None]*(Z-Z.mean(axis=1,keepdims=True))
        if derivatives:
            for key in ('theta','kappa','alpha'):
                d=f4['raw_d'+key][:,self.vertices4];f4['selected_d'+key]=self.g.reshape((-1,)+(1,)*(d.ndim-1))*(d-d.mean(axis=1,keepdims=True))
            if self.learned_template:
                raw=xp.einsum('njab,jbp->njap',f4['point_jacobian'],dB);f4['dtemplate']=self.g[:,None,None,None]*(raw-raw.mean(axis=1,keepdims=True))
                selected=raw[:,self.vertices4];f4['selected_dtemplate']=self.g[:,None,None,None]*(selected-selected.mean(axis=1,keepdims=True))
        return f1,f4,s

    def evaluate(self,theta,A,q=None,derivatives=False,optical=False):
        q=self.q0 if q is None else q;out=super().evaluate(theta,A,q,derivatives,optical);xp=self.xp
        penalty=0.
        if self.barrel_anchor:
            s=float(np.asarray(q)[self.slope_index]);mean=out['means'];diff=s*(mean-A_REF)-self.kappa_targets
            penalty=float(host(xp.mean((diff/self.sigma_kappa)**2),xp));out['loss']+=penalty
            if derivatives:
                out['gradient'][:,1]+=2*s*diff[self.group]/(self.G*self.sigma_kappa**2*self.count[self.group])
                out['global_gradient'][self.slope_index]+=float(host(2*xp.mean(diff*(mean-A_REF))/self.sigma_kappa**2,xp))
        out['components']['barrel_mean_anchor']=penalty
        return out

    def mean_anchor_beta(self,q):
        beta=super().mean_anchor_beta(q)
        if self.barrel_anchor:beta+=2*float(q[self.slope_index])**2/(self.G*self.sigma_kappa**2*self.count**2)
        return beta
