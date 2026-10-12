"""Shared P1 raw keystone/template calibration with declared fixed g=1.

Matches the bounded-scale primary's native template energy, coordinate origin,
initializer, frozen radius, frame roster and capture/fixation weighting.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .p1_joint import template,PARAM_SCALES,DEFAULT_BOUNDS
from .capturewise_reverse import forward,host,equal_group_weights


def fit(X,theta,capture,group,reference,R,*,xp=np,n_starts=6,maxiter=1000,seed=2705,initial=None,m_bounds=None):
    X=xp.asarray(X,dtype=xp.float64);theta=xp.asarray(theta,dtype=xp.float64)
    if X.shape!=(len(theta),3,2):raise ValueError('Native row-aligned P1 triangles required')
    reference=np.asarray(reference,dtype=float);template(reference,np.zeros(3))
    weights=xp.asarray(equal_group_weights(group,capture))
    if m_bounds is not None:
        m_bounds=np.asarray(m_bounds,dtype=float)
        if m_bounds.shape!=(2,) or not np.isfinite(m_bounds).all() or not 0<m_bounds[0]<m_bounds[1]:raise ValueError('Declared positive finite scale interval required')
    scales=PARAM_SCALES.copy();bounds=DEFAULT_BOUNDS.copy();bounds[3]=[-120.,120.]
    counters=dict(evaluations=0,rejected_domain_evaluations=0,max_invalid_rows=0)
    def evaluate(p,derivatives=False):
        B,dB=template(reference,p[4:],True)
        f=forward(B,theta,p[:3],p[3],R,1.,xp=xp,derivatives=derivatives)
        H=f['H'];D=xp.sum(H*H,axis=(1,2))
        unconstrained=xp.sum(H*X,axis=(1,2))/D
        m=xp.ones(len(theta),dtype=xp.float64) if m_bounds is None else xp.minimum(m_bounds[1],xp.maximum(m_bounds[0],unconstrained))
        prediction=m[:,None,None]*H
        valid=f['valid']&xp.isfinite(m)&(m>0)&(D>float(R)**2*xp.finfo(xp.float64).eps)&xp.isfinite(prediction-X).all(axis=(1,2))
        residual=prediction-X;invalid=int(host(xp.sum(~valid),xp));counters['evaluations']+=1
        if invalid:
            counters['rejected_domain_evaluations']+=1;counters['max_invalid_rows']=max(counters['max_invalid_rows'],invalid)
            return dict(loss=float('inf'),gradient=np.zeros(7),valid=False)
        loss=float(host(xp.sum(weights*xp.sum(residual*residual,axis=(1,2))),xp))
        result=dict(loss=loss,B=B,prediction=prediction,valid=valid,denominator=f['denominator'],g=m,g_unconstrained=unconstrained)
        if derivatives:
            dq=xp.einsum('njab,jbp->njap',f['point_jacobian'],xp.asarray(dB));dq-=dq.mean(axis=1,keepdims=True)
            dH=xp.concatenate((f['dalpha'],-f['dtheta'][...,None],dq),axis=-1)
            J=m[:,None,None,None]*dH
            gradient=host(2*xp.einsum('n,nja,njap->p',weights,residual,J),xp)
            if m_bounds is not None:
                dm=(xp.einsum('njap,nja->np',dH,X)-2*unconstrained[:,None]*xp.einsum('njap,nja->np',dH,H))/D[:,None]
                active=(unconstrained<m_bounds[0])|(unconstrained>m_bounds[1]);dm=xp.where(active[:,None],0.,dm)
                J=J+H[...,None]*dm[:,None,None,:]
            result.update(gradient=gradient,J=J)
        return result
    def certificate(p,gradient):
        projected=gradient*scales;contact=np.isclose(p,bounds[:,0],atol=1e-7*scales,rtol=0)|np.isclose(p,bounds[:,1],atol=1e-7*scales,rtol=0)
        projected[((p<=bounds[:,0]+1e-7*scales)&(projected>0))|((p>=bounds[:,1]-1e-7*scales)&(projected<0))]=0
        return dict(projected_gradient_scaled_inf=float(np.max(abs(projected))),gradient_native=gradient,stationary=bool(np.max(abs(projected))<1e-5),bound_contact=contact)
    rng=np.random.default_rng(seed);starts=[];warm=np.zeros(7) if initial is None else np.asarray(initial,dtype=float).copy()
    for index in range(n_starts):
        p=warm.copy() if index!=1 else np.zeros(7)
        if index>1:
            p+=rng.normal(size=7)*scales*.12;p[3]=np.clip(warm[3]+[-15.,15.,-30.,30.][(index-2)%4],*bounds[3])
        p=np.minimum(bounds[:,1],np.maximum(bounds[:,0],p));initial_p=p.copy()
        initial_eval=evaluate(p,True)
        if not np.isfinite(initial_eval['loss']):
            p=np.zeros(7);initial_eval=evaluate(p,True);fallback='Perturbed initial domain invalid; zero initializer used'
        else:fallback=None
        effective_initial=p.copy();initial_gradient=initial_eval['gradient'].copy();best_loss=initial_eval['loss'];trials=[]
        for factor in (1.,.1,.01,.001):
            coordinates=scales*factor
            def objective(u):
                ev=evaluate(u*coordinates,True)
                return ev['loss'],ev['gradient']*coordinates
            opt=minimize(objective,p/coordinates,jac=True,method='L-BFGS-B',bounds=bounds/coordinates[:,None],options=dict(maxiter=maxiter,ftol=1e-14,gtol=1e-7*factor,maxls=40))
            candidate=opt.x*coordinates;ev=evaluate(candidate,True)
            accepted=bool(np.isfinite(ev['loss']) and ev['loss']<=best_loss+1e-12*max(1.,abs(best_loss)))
            if accepted:p=candidate;best_loss=ev['loss']
            cert=certificate(p,evaluate(p,True)['gradient'])
            trials.append(dict(coordinate_factor=factor,success=bool(opt.success),message=str(opt.message),iterations=int(opt.nit),accepted=accepted,loss=best_loss,certificate=cert))
            if cert['projected_gradient_scaled_inf']<1e-6:break
        ev=evaluate(p,True);cert=certificate(p,ev['gradient'])
        starts.append(dict(start=index,params=p,initial_params=initial_p,effective_initial_params=effective_initial,initial_gradient_native=initial_gradient,fallback_reason=fallback,loss=ev['loss'],valid=np.isfinite(ev['loss']),trials=trials,**cert))
        print(f'P1 {"fixed g" if m_bounds is None else "bounded profile"} start {index}: loss={ev["loss"]:.9g}, gradient={cert["projected_gradient_scaled_inf"]:.3g}',flush=True)
    stationary=[s for s in starts if s['valid'] and s['stationary']]
    if not stationary:raise RuntimeError('No valid stationary fixed-scale P1 start; do not promote calibration')
    best=min(stationary,key=lambda s:s['loss']);ev=evaluate(best['params'],True)
    J=ev['J']*xp.asarray(scales)[None,None,None,:]
    gram=host(xp.einsum('n,njap,njaq->pq',weights,J,J),xp);eigenvectors=np.linalg.eigh(gram);singular=np.sqrt(np.maximum(eigenvectors[0],0))[::-1]
    return dict(params=best['params'],B=ev['B'],R=float(R),loss=ev['loss'],starts=starts,best_start=best['start'],certificate=certificate(best['params'],ev['gradient']),bounds=bounds,characteristic_scales=scales,counters=counters,data_only_scaled_singular_values=singular,data_only_rank=int(np.sum(singular>singular[0]*1e-8)),weakest_scaled_mode=eigenvectors[1][:,0],gauge='native centered template energy equals frozen initializer; no additional mean-g constraint',scale_policy='fixed g=1 throughout P1 calibration and downstream branch' if m_bounds is None else 'exact bounded scalar P1-only least-squares profile',m_bounds=m_bounds)
