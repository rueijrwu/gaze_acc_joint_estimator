"""Shared size sensitivity; immutable barrel-first calibration and shared g."""
import numpy as np
from distortion_model.barrel_first_pipeline import BarrelFirstStateProblem
from distortion_model.capturewise_reverse import host
from distortion_model.capturewise_states import A_REF
from distortion_model.p1_joint import template

CASES={
 'free_size_sigma050_optical2':dict(sigma_A=.5,optical_width=2.),
 'free_size_sigma025_optical1':dict(sigma_A=.25,optical_width=1.),
}
FIXED_CASES={name:name.replace('free_size','fixed_size') for name in CASES}

class FreeSizeProblem(BarrelFirstStateProblem):
    def __init__(self,*args,optical_width=1.,free_size=True,size_bounds=(.5,2.),**kwargs):
        if not np.isfinite(optical_width) or optical_width<=0:raise ValueError('Positive optical width required')
        self.optical_width=float(optical_width);self.free_size=bool(free_size);self.size_bounds=np.asarray(size_bounds,float)
        if self.size_bounds.shape!=(2,) or not (0<self.size_bounds[0]<1<self.size_bounds[1]):raise ValueError('Numerical size bounds must contain one')
        super().__init__(*args,**kwargs)
        if self.free_size:
            self.q0=np.r_[self.q0,0.];self.scales=np.r_[self.scales,.01];self.bounds=np.vstack((self.bounds,np.log(self.size_bounds)));self.template_slice=slice(14,18)
    def optical_prior(self,q):
        loss,grad=super().optical_prior(q)
        if self.free_size:grad=np.r_[grad,0.]
        return loss/self.optical_width**2,grad/self.optical_width**2
    def template_geometry(self,q,derivatives=False):
        if not self.free_size:return super().template_geometry(q,derivatives)
        centered,dshape=template(self.B4_centered,np.asarray(q)[14:17],True);size=np.exp(np.asarray(q)[17]);B=self.B4_centroid+size*centered
        singular=np.linalg.svd(B-B.mean(axis=0),compute_uv=False);valid=bool(np.isfinite(B).all() and singular[-1]>1e-10*max(singular[0],1.))
        if derivatives:return self.xp.asarray(B),self.xp.asarray(np.concatenate((size*dshape,(size*centered)[...,None]),axis=-1)),valid
        return self.xp.asarray(B)
    def capture_means(self,A):
        return self.xp.stack([self.means(A)[self.labels//5==c].mean() for c in range(4)])

    def data_only_rank(self,theta,A,q,chunk_size=4096):
        """Optical/template data Jacobian with frame theta/A projected out.

        Includes native P1 and retained P4 data, excludes every prior and bound.
        The slope/A scale-null direction must remain in this audit.
        """
        xp=self.xp;theta=xp.asarray(theta);A=xp.asarray(A)
        f1,f4,s=self.predictions(theta,A,q,True);V=len(self.vertices4);P=len(self.scales)
        factor=np.sqrt(3./V)
        jt=xp.concatenate((f1['dtheta'].reshape(self.n,6),f4['selected_dtheta'].reshape(self.n,2*V)*factor),axis=1)
        ja=xp.concatenate((xp.zeros((self.n,6)),(f4['selected_dkappa']*s).reshape(self.n,2*V)*factor),axis=1)
        states=xp.stack((jt*.5,ja*.25),axis=-1)
        gram=xp.zeros((P,P),dtype=xp.float64);slope_residual=0.
        for lo in range(0,self.n,chunk_size):
            hi=min(self.n,lo+chunk_size);n=hi-lo;glob=xp.zeros((n,6+2*V,P),dtype=xp.float64)
            da=f4['selected_dalpha'][lo:hi].reshape(n,2*V,3)*factor
            if self.shared:
                glob[:,6:,:3]=da;glob[:,6:,1:3]*=(self.R4[lo:hi]/self.Rref)[:,None,None]
            else:
                for c in range(4):glob[:,6:,3*c:3*c+3]=da*(self.c[lo:hi]==c)[:,None,None]
            glob[:,6:,self.offset_index]=-f4['selected_dtheta'][lo:hi].reshape(n,2*V)*factor
            glob[:,6:,self.slope_index]=f4['selected_dkappa'][lo:hi].reshape(n,2*V)*(A[lo:hi]-A_REF)[:,None]*factor
            if self.learned_template:glob[:,6:,self.template_slice]=f4['selected_dtemplate'][lo:hi].reshape(n,2*V,self.template_slice.stop-self.template_slice.start)*factor
            glob*=xp.asarray(self.scales);st=states[lo:hi]
            h=xp.einsum('nip,niq->npq',st,st);cross=xp.einsum('nip,niq->npq',st,glob)
            inv=xp.linalg.pinv(h,rcond=1e-12)
            projected=glob-xp.einsum('nip,npq->niq',st,xp.einsum('npq,nqk->npk',inv,cross))
            gram+=xp.einsum('n,nip,niq->pq',self.w[lo:hi]/6,projected,projected)
            slope_residual=max(slope_residual,float(host(xp.max(xp.abs(projected[:,:,self.slope_index])),xp)))
        eigen,vectors=np.linalg.eigh(host(gram,xp));singular=np.sqrt(np.maximum(eigen,0))[::-1]
        names=(['b_shared','cx_shared_Rref','cy_shared_Rref'] if self.shared else [f'capture{c+1}_{name}' for c in range(4) for name in ('b','cx','cy')])+['offset_deg','slope_pxminus2_perD']
        if self.learned_template:names+=['template_q1','template_q2','template_q3']
        if self.free_size:names+=['log_template_size']
        return dict(dimension=P,parameter_names=names,data_only_profiled_optical_singular_values=singular,relative_singular_threshold=1e-6,data_only_rank_at_relative_1e_minus6=int(np.sum(singular>singular[0]*1e-6)) if singular[0]>0 else 0,weakest_scaled_modes=vectors[:,:min(4,P)],slope_column_after_state_projection_max=slope_residual,method='framewise data-only theta/A projection includes P1 and P4; no priors/bounds; native equal-pattern weights, declared parameter characteristic units',interpretation='Gram spectrum can lose precision in weakest modes. Anchors select D units; data alone leaves slope/A scale null. Learned template may add weak or ambiguous modes.',chunk_size=chunk_size)
