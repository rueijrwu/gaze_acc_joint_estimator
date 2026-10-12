"""Native P4 order D(K(B)): radial inverse precedes keystone inverse.

This is a separate geometry family. P1 and legacy P4 geometry are unchanged.
"""
import numpy as np
from .capturewise_reverse import _inputs, _radial_inverse, forward as old_forward


def _raw(B,theta,alpha,offset,R,kappa,xp,derivatives):
    keystone=old_forward(B,theta,alpha,offset,R,1.,xp=xp,derivatives=derivatives)
    U=keystone['Z'];r2=xp.sum(U*U,axis=2)
    factor=1+kappa[:,None]*r2;radial_derivative=1+3*kappa[:,None]*r2
    Z=factor[...,None]*U
    out=dict(Z=Z,keystone_points=U,denominator=keystone['denominator'],
             factors=keystone['factors'],radial_factor=factor,radial_derivative=radial_derivative,
             valid=keystone['valid']&xp.all(factor>0,axis=1)&xp.all(radial_derivative>0,axis=1)&xp.isfinite(Z).all(axis=(1,2)))
    if derivatives:
        Jrad=factor[...,None,None]*xp.eye(2)+2*kappa[:,None,None,None]*U[..., :,None]*U[...,None,:]
        out['point_jacobian']=xp.einsum('njab,njbc->njac',Jrad,keystone['point_jacobian'])
        out['raw_dtheta']=xp.einsum('njab,njb->nja',Jrad,keystone['raw_dtheta'])
        out['raw_dalpha']=xp.einsum('njab,njbp->njap',Jrad,keystone['raw_dalpha'])
        out['raw_dkappa']=r2[...,None]*U
    return out


def forward(B,theta,alpha,offset,R,g,kappa=0,xp=np,derivatives=False):
    B,theta,alpha,offset,R,g,kappa=_inputs(B,theta,alpha,offset,R,g,kappa,xp)
    out=_raw(B,theta,alpha,offset,R,kappa,xp,derivatives)
    Z=out['Z'];mu=Z.mean(axis=1);H=Z-mu[:,None]
    out.update(mu=mu,H=H,prediction=g[:,None,None]*H)
    out['valid'] &= (g>0)&xp.isfinite(g)&xp.isfinite(out['prediction']).all(axis=(1,2))
    if derivatives:
        for key in ('theta','alpha','kappa'):
            d=out['raw_d'+key];out['d'+key]=g.reshape((-1,)+(1,)*(d.ndim-1))*(d-d.mean(axis=1,keepdims=True))
    return out


def inverse(B,X,theta,alpha,offset,R,g,kappa=0,xp=np,derivatives=False):
    B,theta,alpha,offset,R,g,kappa=_inputs(B,theta,alpha,offset,R,g,kappa,xp)
    model=forward(B,theta,alpha,offset,R,g,kappa,xp,derivatives)
    restored=xp.asarray(X)/g[:,None,None]+model['mu'][:,None]
    U,rv,re,rm=_radial_inverse(restored,kappa,xp)
    V=U/model['factors'][:,None];phi=(theta-offset)/10
    den=1-phi[:,None]*xp.sum(V*alpha[:,None,1:],axis=2)/R[:,None]
    recovered=V/den[...,None]
    valid=model['valid']&rv.all(axis=1)&(den>0).all(axis=1)&xp.isfinite(recovered).all(axis=(1,2))
    out=dict(recovered=recovered,valid=valid,mu=model['mu'],restored=restored,
             inverse_denominator=den,radial_image_margin=rm,radial_equation_error=re,
             denominator=model['denominator'],radial_factor=model['radial_factor'],radial_derivative=model['radial_derivative'])
    recovered_r2=xp.sum(U*U,axis=2)
    out.update(recovered_radial_factor=1+kappa[:,None]*recovered_r2,recovered_radial_derivative=1+3*kappa[:,None]*recovered_r2)
    if derivatives:
        raw=_raw(recovered,theta,alpha,offset,R,kappa,xp,True);J=raw['point_jacobian']
        determinant=J[...,0,0]*J[...,1,1]-J[...,0,1]*J[...,1,0]
        Ji=xp.stack((J[...,1,1],-J[...,0,1],-J[...,1,0],J[...,0,0]),axis=-1).reshape(J.shape)/determinant[...,None,None]
        for key in ('theta','alpha','kappa'):
            rhs=model['raw_d'+key].mean(axis=1,keepdims=True)-raw['raw_d'+key]
            out['d'+key]=xp.einsum('njab,njb->nja',Ji,rhs) if rhs.ndim==3 else xp.einsum('njab,njbp->njap',Ji,rhs)
        out.update(inverse_point_jacobian=Ji,template_point_jacobian=model['point_jacobian'])
    return out
