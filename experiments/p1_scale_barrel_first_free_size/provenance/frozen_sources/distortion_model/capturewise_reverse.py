"""Native horizontal raw keystone/radial equations and capturewise fitting.

All public geometry returns complete row-aligned arrays. Invalid rows are marked,
never deleted. The numerical core supports NumPy or CuPy float64.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .p1_joint import template


def _inputs(B, theta, alpha, offset, R, g, kappa, xp):
    theta = xp.atleast_1d(xp.asarray(theta, dtype=xp.float64))
    B=xp.asarray(B,dtype=xp.float64);alpha=xp.asarray(alpha,dtype=xp.float64)
    lengths=[len(theta)]
    if B.ndim==3:lengths.append(B.shape[0])
    if alpha.ndim==2:lengths.append(alpha.shape[0])
    for value in (offset,R,g,kappa):
        value=xp.asarray(value)
        if value.ndim:lengths.append(len(value))
    n=max(lengths);theta=xp.broadcast_to(theta,(n,))
    def rows(value, tail=()):
        return xp.broadcast_to(xp.asarray(value, dtype=xp.float64), (n,)+tail)
    return rows(B, (3, 2)), theta, rows(alpha, (3,)), rows(offset), rows(R), rows(g), rows(kappa)


def _raw(B, theta, alpha, offset, R, kappa, xp, derivatives):
    phi = (theta-offset)/10
    r2 = xp.sum(B*B, axis=2)
    radial_factor = 1+kappa[:, None]*r2
    radial_derivative = 1+3*kappa[:, None]*r2
    Q = radial_factor[..., None]*B
    signs = xp.asarray([1., -1.])
    factors = xp.exp(alpha[:, 0, None]*phi[:, None]**2*signs)
    c = alpha[:, 1:]/R[:, None]
    slope = xp.sum(c[:, None]*Q, axis=2)
    denominator = 1+phi[:, None]*slope
    Z = factors[:, None]*Q/denominator[..., None]
    valid = (xp.isfinite(Z).all(axis=(1, 2)) & (R > 0) & xp.isfinite(R) & xp.isfinite(alpha).all(axis=1)
             & xp.all(denominator > 0, axis=1) & xp.all(radial_factor > 0, axis=1)
             & xp.all(radial_derivative > 0, axis=1))
    result = dict(Z=Z, denominator=denominator, radial_factor=radial_factor,
                  radial_derivative=radial_derivative, valid=valid, factors=factors)
    if derivatives:
        dt = Z*(2*alpha[:, 0, None, None]*phi[:, None, None]*signs-slope[..., None]/denominator[..., None])/10
        da = xp.stack((Z*phi[:, None, None]**2*signs,
                       -Z*(phi[:, None]*Q[..., 0]/R[:, None]/denominator)[..., None],
                       -Z*(phi[:, None]*Q[..., 1]/R[:, None]/denominator)[..., None]), axis=-1)
        eye = xp.eye(2, dtype=xp.float64)
        Jk = factors[:, None, :, None]/denominator[..., None, None]*eye - Z[..., :, None]*(phi[:, None, None,None]*c[:,None,None,:]/denominator[...,None,None])
        Jr = radial_factor[..., None,None]*eye + 2*kappa[:,None,None,None]*B[..., :,None]*B[...,None,:]
        J = xp.einsum('njab,njbc->njac', Jk, Jr)
        dk = xp.einsum('njab,njb->nja', Jk, B*r2[...,None])
        result.update(raw_dtheta=dt, raw_dalpha=da, raw_dkappa=dk, point_jacobian=J)
    return result


def forward(B, theta, alpha, offset, R, g, kappa=0, xp=np, derivatives=False):
    B,theta,alpha,offset,R,g,kappa = _inputs(B,theta,alpha,offset,R,g,kappa,xp)
    out = _raw(B,theta,alpha,offset,R,kappa,xp,derivatives)
    Z = out['Z']; mu = Z.mean(axis=1); H = Z-mu[:,None]
    out.update(mu=mu, H=H, prediction=g[:,None,None]*H)
    out['valid'] &= (g > 0) & xp.isfinite(g) & xp.isfinite(out['prediction']).all(axis=(1,2))
    if derivatives:
        for key in ('theta','kappa','alpha'):
            d = out['raw_d'+key]
            out['d'+key] = g.reshape((-1,)+(1,)*(d.ndim-1))*(d-d.mean(axis=1,keepdims=True))
    return out


def _radial_inverse(Q, kappa, xp):
    """Safeguarded Newton on the strictly increasing radial branch (24 steps).

    Negative kappa ends at rcrit=sqrt(-1/(3*kappa)); observations beyond its
    maximum image radius are explicitly invalid. No clipping changes the data.
    """
    s = xp.sqrt(xp.sum(Q*Q, axis=2)); k = kappa[:,None]
    if bool(xp.all(kappa == 0)):
        return Q.copy(), xp.isfinite(s), xp.zeros_like(s), xp.full_like(s,xp.inf)
    negative = k < 0
    safe_k = xp.where(negative, k, -1.)
    critical = xp.sqrt(-1/(3*safe_k))
    max_image = 2*critical/3
    feasible = xp.isfinite(s) & (~negative | (s < max_image))
    lo = xp.zeros_like(s)
    hi = xp.where(negative, critical, s)
    r = xp.where(negative, xp.minimum(s, .5*hi), s/(1+xp.maximum(k,0)*s*s))
    for _ in range(24):
        value = r*(1+k*r*r)-s
        lo = xp.where(value < 0, r, lo); hi = xp.where(value >= 0,r,hi)
        candidate = r-value/(1+3*k*r*r)
        candidate = xp.where((candidate > lo)&(candidate < hi)&xp.isfinite(candidate),candidate,(lo+hi)/2)
        # At an exact root keep it: otherwise bisection can degrade convergence.
        r = xp.where(xp.abs(value) <= 8*xp.finfo(xp.float64).eps*xp.maximum(s,1),r,candidate)
    residual = xp.abs(r*(1+k*r*r)-s)
    valid = feasible & (1+k*r*r > 0) & (1+3*k*r*r > 0) & (residual <= 2e-11*xp.maximum(s,1))
    factor = xp.where(s > 0,r/xp.where(s > 0,s,1),1.)
    return Q*factor[...,None], valid, residual, xp.where(negative,max_image-s,xp.inf)


def inverse(B, X, theta, alpha, offset, R, g, kappa=0, xp=np, derivatives=False):
    B,theta,alpha,offset,R,g,kappa = _inputs(B,theta,alpha,offset,R,g,kappa,xp)
    X = xp.asarray(X,dtype=xp.float64)
    model = forward(B,theta,alpha,offset,R,g,kappa,xp,derivatives)
    restored = X/g[:,None,None]+model['mu'][:,None]
    U = restored/model['factors'][:,None]
    phi = (theta-offset)/10
    inv_den = 1-phi[:,None]*xp.sum(U*alpha[:,None,1:],axis=2)/R[:,None]
    Q = U/inv_den[...,None]
    recovered, radial_valid, radial_error, radial_margin = _radial_inverse(Q,kappa,xp)
    valid = model['valid'] & xp.all(inv_den > 0,axis=1) & radial_valid.all(axis=1) & xp.isfinite(recovered).all(axis=(1,2))
    # Preserve candidates and masks separately so invalid observations cannot be
    # accidentally hidden from fitting by nanmean or finite-row filtering.
    out = dict(recovered=recovered,valid=valid,mu=model['mu'],restored=restored,
               denominator=model['denominator'],inverse_denominator=inv_den,
               radial_factor=model['radial_factor'],radial_derivative=model['radial_derivative'],
               radial_image_margin=radial_margin,radial_equation_error=radial_error,
               recovered_radial_factor=1+kappa[:,None]*xp.sum(recovered*recovered,axis=2),
               recovered_radial_derivative=1+3*kappa[:,None]*xp.sum(recovered*recovered,axis=2))
    if derivatives:
        raw = _raw(recovered,theta,alpha,offset,R,kappa,xp,True)
        J = raw['point_jacobian']
        determinant = J[...,0,0]*J[...,1,1]-J[...,0,1]*J[...,1,0]
        Ji = xp.stack((J[...,1,1],-J[...,0,1],-J[...,1,0],J[...,0,0]),axis=-1).reshape(J.shape)/determinant[...,None,None]
        for key in ('theta','kappa','alpha'):
            rhs = model['raw_d'+key].mean(axis=1,keepdims=True)-raw['raw_d'+key]
            out['d'+key] = xp.einsum('njab,njb->nja',Ji,rhs) if rhs.ndim == 3 else xp.einsum('njab,njbp->njap',Ji,rhs)
        out['inverse_point_jacobian'] = Ji
        out['template_point_jacobian'] = model['point_jacobian']
    return out


def host(value, xp=np):
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def equal_group_weights(group, capture=None):
    group=np.asarray(group)
    if capture is None:
        _, inv, count = np.unique(group,return_inverse=True,return_counts=True)
        return 1/(len(count)*count[inv]*3.)
    capture=np.asarray(capture); weights=np.zeros(len(group));captures=np.unique(capture)
    for c in captures:
        m=capture==c;weights[m]=equal_group_weights(group[m])/len(captures)
    return weights


def fit_keystone(X, theta, capture, group, g, references, radii, xp=np,
                  learned=False, objective='inverse', n_starts=4,maxiter=450,seed=20261011,initial=None):
    """Four K blocks, one shared offset; exact equal-energy shape chart control."""
    cap = np.asarray(capture,dtype=int)-1
    w = xp.asarray(equal_group_weights(group,capture)); X=xp.asarray(X); theta=xp.asarray(theta); g=xp.asarray(g)
    c=xp.asarray(cap); R=xp.asarray(radii)[c]
    nparam = 25 if learned else 13
    scales=np.r_[np.tile([.05,.05,.05],4),10.,np.full(12,.05)][:nparam]
    bounds=np.array([[-.25,.25],[-.2,.2],[-.2,.2]]*4+[[-30.,30.]]+[[-.75,.75]]*12)[:nparam]
    counters=dict(evaluations=0,invalid_evaluations=0,max_invalid_rows=0)
    def unpack(p):
        Bs=[]; Ds=[]
        for j in range(4):
            B,D=template(references[j],p[13+3*j:16+3*j] if learned else None,True)
            Bs.append(B);Ds.append(D)
        return xp.asarray(np.array(Bs))[c],xp.asarray(np.array(Ds))[c]
    def evaluate(p,gradient=False):
        B,dB=unpack(p); alpha=xp.asarray(p[:12].reshape(4,3))[c]
        fn=inverse if objective=='inverse' else forward
        ev=fn(B,X,theta,alpha,p[12],R,g,xp=xp,derivatives=gradient) if objective=='inverse' else fn(B,theta,alpha,p[12],R,g,xp=xp,derivatives=gradient)
        residual=ev['recovered']-B if objective=='inverse' else ev['prediction']-X
        invalid=int(host(xp.sum(~ev['valid']),xp));counters['evaluations']+=1
        if invalid:
            counters['invalid_evaluations']+=1;counters['max_invalid_rows']=max(counters['max_invalid_rows'],invalid)
            return (float('inf'),np.zeros(nparam)) if gradient else (float('inf'),ev,B)
        loss=float(host(xp.sum(w*xp.sum(residual*residual,axis=(1,2))),xp))
        if not gradient:return loss,ev,B
        grad=xp.zeros(nparam)
        rowgrad=2*w[:,None]*xp.sum(residual[...,None]*ev['dalpha'],axis=(1,2))
        for j in range(4):grad[3*j:3*j+3]=xp.sum(rowgrad*(c==j)[:,None],axis=0)
        grad[12]=-2*xp.sum(w*xp.sum(residual*ev['dtheta'],axis=(1,2)))
        if learned:
            rawd=xp.einsum('njab,njbp->njap',ev['template_point_jacobian'] if objective=='inverse' else ev['point_jacobian'],dB)
            if objective=='inverse':
                d=xp.einsum('njab,njbp->njap',ev['inverse_point_jacobian'],xp.broadcast_to(rawd.mean(axis=1,keepdims=True),rawd.shape))-dB
            else:d=g[:,None,None,None]*(rawd-rawd.mean(axis=1,keepdims=True))
            rowgrad=2*w[:,None]*xp.sum(residual[...,None]*d,axis=(1,2))
            for j in range(4):grad[13+3*j:16+3*j]=xp.sum(rowgrad*(c==j)[:,None],axis=0)
        return loss,host(grad,xp)
    rng=np.random.default_rng(seed); starts=[];p0=np.zeros(nparam) if initial is None else np.r_[initial,np.zeros(max(0,nparam-len(initial)))][:nparam]
    for start in range(n_starts):
        p=p0.copy()
        if start:p[:12]+=rng.normal(size=12)*.015;p[12]=[-10.,10.,20.,-20.][(start-1)%4]
        result=minimize(lambda u:(lambda z:(z[0],z[1]*scales))(evaluate(u*scales,True)),p/scales,jac=True,method='L-BFGS-B',bounds=bounds/scales[:,None],options=dict(maxiter=maxiter,ftol=1e-13,gtol=1e-7,maxls=40))
        p=result.x*scales; loss,grad=evaluate(p,True);pg=grad*scales
        pg[((p<=bounds[:,0]+1e-8)&(pg>0))|((p>=bounds[:,1]-1e-8)&(pg<0))]=0
        starts.append(dict(params=p,loss=loss,success=bool(result.success),message=str(result.message),iterations=int(result.nit),gradient_native=grad,projected_gradient_scaled_inf=float(np.max(abs(pg))),stationary=bool(np.isfinite(loss) and np.max(abs(pg))<1e-5),bound_contact=(np.isclose(p,bounds[:,0],atol=1e-7)|np.isclose(p,bounds[:,1],atol=1e-7))))
    finite=[s for s in starts if np.isfinite(s['loss'])]
    if not finite:raise RuntimeError('No wholly domain-valid keystone solution')
    stationary=[s for s in finite if s['stationary']]
    best=min(stationary or finite,key=lambda s:s['loss']);p=best['params'];loss,ev,B=evaluate(p)
    Bs=np.array([template(references[j],p[13+3*j:16+3*j] if learned else None) for j in range(4)])
    B,dB=unpack(p);alpha=xp.asarray(p[:12].reshape(4,3))[c]
    ev=inverse(B,X,theta,alpha,p[12],R,g,xp=xp,derivatives=True) if objective=='inverse' else forward(B,theta,alpha,p[12],R,g,xp=xp,derivatives=True)
    J=xp.zeros((len(cap),3,2,nparam))
    for j in range(4):J[...,3*j:3*j+3]=ev['dalpha']*(c==j)[:,None,None,None]
    J[...,12]=-ev['dtheta']
    if learned:
        rawd=xp.einsum('njab,njbp->njap',ev['template_point_jacobian'] if objective=='inverse' else ev['point_jacobian'],dB)
        d=xp.einsum('njab,njbp->njap',ev['inverse_point_jacobian'],xp.broadcast_to(rawd.mean(axis=1,keepdims=True),rawd.shape))-dB if objective=='inverse' else g[:,None,None,None]*(rawd-rawd.mean(axis=1,keepdims=True))
        for j in range(4):J[...,13+3*j:16+3*j]=d*(c==j)[:,None,None,None]
    J*=xp.asarray(scales)[None,None,None,:]
    gram=host(xp.einsum('n,njap,njaq->pq',w,J,J),xp)
    eig,vectors=np.linalg.eigh(gram);singular=np.sqrt(np.maximum(eig,0))[::-1]
    rank=int(np.sum(singular>singular[0]*1e-8)) if singular[0]>0 else 0
    audit=dict(data_only_scaled_singular_values=singular,data_only_rank=rank,dimension=nparam,weakest_scaled_mode=vectors[:,0],interpretation='All fitted training rows, parameter characteristic units, no priors; small eigenvalues of Gram may lose relative precision')
    if objective=='inverse':
        noise=xp.sqrt(xp.sum(ev['inverse_point_jacobian']**2,axis=(2,3)))/g[:,None]
        audit.update(inverse_noise_gain_frobenius_min=float(host(noise.min(),xp)),inverse_noise_gain_frobenius_max=float(host(noise.max(),xp)))
    return dict(alpha=p[:12].reshape(4,3),offset=float(p[12]),templates=Bs,params=p,loss=loss,starts=starts,best=best,bounds=bounds,scales=scales,counters=counters,certified=best['stationary'],objective=objective,identifiability=audit)
