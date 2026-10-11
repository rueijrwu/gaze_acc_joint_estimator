"""Frame-batched P4 joint geometry, analytic derivatives, and damped Gauss Newton.

Coordinates are native pixels. The radial origin, gaze, and each capture's
keystone radius/origin remain fixed; no output scale or recentering is fitted.
"""
from __future__ import annotations

import time
import numpy as np
from .shared_keystone import inverse_radial

DELTA = .2


def host(x):
    return x.get() if hasattr(x, 'get') else np.asarray(x)


class JointProblem:
    def __init__(self, inputs, dataset, branch='learned_radial', anchor_factor=1., backend='numpy'):
        if backend == 'cupy':
            import cupy as xp
        else:
            xp = np
        self.xp, self.branch = xp, branch
        self.learned = branch != 'fixed_radial'
        self.uniform = branch == 'learned_uniform'
        self.B0 = np.asarray(inputs['B_ref'], float)
        self.K0 = np.asarray(inputs['K_anchor'], float)
        self.k0 = np.asarray(inputs['kappa_anchor'], float).copy()
        self.k0[0] = 0.
        self.origins = np.asarray(inputs['origins'], float)
        self.Rc = np.asarray(inputs['Rc'], float)
        self.M = float(np.max(np.sum(self.B0**2, axis=1)))
        self.offset = 6 if self.learned else 0
        self.np = self.offset + 15
        self.s0 = np.ones(4)
        self.s0[1:] = 1. + self.k0[1:] * float(np.mean(np.sum(self.B0**2, axis=1)))
        self.scale = np.r_[np.full(self.offset, 5.), np.tile([.01, .02, .02], 4), np.full(3, .05)]
        self.batches = []
        for c in dataset['captures']:
            if not len(c['X']):
                continue
            cid = int(c['capture']) - 1
            self.batches.append(dict(c=cid, X=xp.asarray(c['X'], dtype=xp.float64),
                phi=xp.asarray((np.asarray(c['theta'])-self.origins[cid])/10.),
                weight=xp.asarray(c['weights'], dtype=xp.float64)))
        self.domain_phi = [(np.asarray(c['theta'])-self.origins[int(c['capture'])-1])/10.
                           for c in inputs['captures']]
        # One operational prior width per parameter: its first-order prediction
        # movement equals this capture's existing model discrepancy. These are
        # model-sensitivity widths, not independent statistical uncertainties.
        self.width = np.ones(self.np)
        _, _, H, _, capture_loss = self.evaluate(np.zeros(self.np), True, priors=False)
        self.anchor_noise = float(inputs.get('anchor_noise_px', max(float(np.sqrt(sum(capture_loss.values()))), .5)))
        for c in range(4):
            for idx in range(self.offset+3*c, self.offset+3*c+3):
                self.width[idx] = self.anchor_noise / max(np.sqrt(max(H[idx,idx], 0.)/2.), 1e-8)
        for c in range(1,4):
            idx=self.offset+12+c-1
            self.width[idx] = self.anchor_noise / max(np.sqrt(max(H[idx,idx], 0.)/2.), 1e-8)
        self.width *= anchor_factor
        self.prior_coefficient = self.anchor_noise**2/15.

    def unpack(self, p):
        p=np.asarray(p, float)
        B = self.B0 + p[:6].reshape(3,2)*5. if self.learned else self.B0.copy()
        K = self.K0 + (p[self.offset:self.offset+12]*self.scale[self.offset:self.offset+12]).reshape(4,3)
        k=self.k0.copy(); s=np.ones(4)
        if self.uniform:
            k[:]=0.; s=self.s0.copy(); s[1:] += p[-3:]*.05
        else:
            k[1:] += p[-3:]*.05/self.M
        return B,K,k,s

    def feasible(self, p):
        B,K,k,s=self.unpack(p)
        if not all(np.isfinite(x).all() for x in (B,K,k,s)):
            return False
        if np.any(np.abs(K[:,0])>.3) or np.any(np.abs(K[:,1:])>.2):
            return False
        if self.uniform and (np.any(s<.5) or np.any(s>1.5)):
            return False
        if np.any(k*self.M>1.):
            return False
        r2=np.sum(B*B,axis=1)
        for c in range(4):
            if np.min(1+k[c]*r2)<=DELTA or np.min(1+3*k[c]*r2)<=DELTA:
                return False
            q=s[c]*(1+k[c]*r2[:,None])*B
            d=1+self.domain_phi[c][:,None]*(q@K[c,1:])[None,:]/self.Rc[c]
            if not np.isfinite(d).all() or np.min(d)<=DELTA:
                return False
        return True

    def _terms(self, B, K, k, s, phi, c, jac=False):
        xp=self.xp
        B=xp.asarray(B); phi=xp.asarray(phi)
        r2=xp.sum(B*B,axis=1)
        q=s*(1+k*r2[:,None])*B
        a=K[0]*phi*phi
        stretch=xp.stack((xp.exp(a),xp.exp(-a)),axis=1)[:,None,:]
        h=xp.asarray(K[1:])/self.Rc[c]
        den=1+phi[:,None]*(q@h)[None,:]
        F=stretch*q[None,:,:]/den[:,:,None]
        mu=xp.mean(F,axis=1); pred=F-mu[:,None,:]
        if not jac:
            return pred,mu,den
        J=xp.zeros((*F.shape,self.np),dtype=xp.float64)
        # Derivative of projective/stretch mapping with respect to q_j.
        eye=xp.eye(2)
        D=stretch[:,:,:,None]*eye[None,None,:,:]/den[:,:,None,None]
        D -= F[:,:,:,None]*(phi[:,None,None,None]*h[None,None,None,:])/den[:,:,None,None]
        if self.learned:
            Q=s*((1+k*r2)[:,None,None]*eye[None,:,:]+2*k*B[:,:,None]*B[:,None,:])
            for j in range(3):
                J[:,j,:,2*j:2*j+2]=(D[:,j]@Q[j])*5.
        j0=self.offset+3*c
        signs=xp.asarray([1.,-1.])
        J[:,:,:,j0]=F*(phi*phi)[:,None,None]*signs[None,None,:]*.01
        for t in range(2):
            J[:,:,:,j0+1+t]=-F*(phi[:,None]*q[None,:,t]/(self.Rc[c]*den))[:,:,None]*.02
        if c:
            dq=q/s*.05 if self.uniform else (r2[:,None]*B)*(.05/self.M)
            J[:,:,:,self.offset+12+c-1]=xp.einsum('njab,jb->nja',D,dq)
        J-=xp.mean(J,axis=1,keepdims=True)
        return pred,mu,den,J

    def evaluate(self, p, derivatives=False, priors=True):
        xp=self.xp
        B,K,k,s=self.unpack(p)
        if not self.feasible(p):
            return (float('inf'),None,None,None,{}) if derivatives else float('inf')
        loss=xp.asarray(0.,dtype=xp.float64)
        g=xp.zeros(self.np); H=xp.zeros((self.np,self.np)); parts={}
        for batch in self.batches:
            c=batch['c']; result=self._terms(B,K[c],k[c],s[c],batch['phi'],c,derivatives)
            pred=result[0]; e=pred-batch['X']; w=batch['weight']
            lc=xp.sum(e*e*w[:,None,None]); loss+=lc; parts[c]=float(host(lc))
            if derivatives:
                J=result[3].reshape(-1,self.np)
                rw=xp.sqrt(2*w)[:,None,None]*xp.ones_like(e)
                J=J*rw.reshape(-1,1); r=(e*rw).reshape(-1)
                g+=J.T@r; H+=J.T@J
        data_loss=float(host(loss)); penalty=0.
        if priors:
            active=np.arange(self.offset,self.np)
            v=np.asarray(p)[active]/self.width[active]
            penalty=self.prior_coefficient*float(v@v)
            if derivatives:
                g[active]+=xp.asarray(2*self.prior_coefficient*np.asarray(p)[active]/self.width[active]**2)
                H[active,active]+=xp.asarray(2*self.prior_coefficient/self.width[active]**2)
        if derivatives:
            return data_loss+penalty,host(g),host(H),dict(data_loss=data_loss,prior_loss=penalty),parts
        return data_loss+penalty

    def fit(self, start=None, max_iter=250, fixed=None):
        p=np.zeros(self.np) if start is None else np.asarray(start,float).copy()
        fixed={} if fixed is None else fixed
        for index,value in fixed.items():
            p[int(index)]=float(value)
        free=np.array([i for i in range(self.np) if i not in fixed],dtype=int)
        if not self.feasible(p):
            raise ValueError('Infeasible start')
        tic=time.monotonic(); damping=1e-3; history=[]; reason='iteration_limit'; success=False
        for it in range(max_iter):
            loss,g,H,parts,_=self.evaluate(p,True)
            diagonal=np.maximum(np.diag(H),1e-8)
            grad=float(np.max(np.abs(g[free])/np.sqrt(diagonal[free])))
            history.append(dict(iteration=it,objective=float(loss),**parts,scaled_gradient=grad,damping=float(damping)))
            if grad<1e-7:
                reason='scaled_gradient'; success=True; break
            accepted=False
            for attempt in range(18):
                step=np.zeros(self.np)
                step[free]=np.linalg.solve(H[np.ix_(free,free)]+damping*np.diag(diagonal[free]),-g[free])
                if np.max(np.abs(step))>20:
                    step*=20/np.max(np.abs(step))
                trial=p+step; trial_loss=self.evaluate(trial)
                if np.isfinite(trial_loss) and trial_loss<loss:
                    p=trial; damping=max(damping/3,1e-12); accepted=True; break
                damping*=10
            if not accepted:
                reason='no_descent'; success=grad<1e-6; break
        loss,g,H,parts,_=self.evaluate(p,True)
        grad=float(np.max(np.abs(g[free])/np.sqrt(np.maximum(np.diag(H)[free],1e-8))))
        if grad<1e-6: success=True
        return dict(p=p,objective=float(loss),**parts,scaled_gradient=grad,success=bool(success),reason=reason,
                    iterations=len(history),seconds=time.monotonic()-tic,history=history,fixed={str(i):float(v) for i,v in fixed.items()})

    def predictions(self, p, cap):
        c=int(cap['capture'])-1; B,K,k,s=self.unpack(p)
        phi=(np.asarray(cap['theta'])-self.origins[c])/10.
        return tuple(host(v) for v in self._terms(B,K[c],k[c],s[c],phi,c))

    def inverse(self,p,cap):
        c=int(cap['capture'])-1; B,K,k,s=self.unpack(p)
        pred,mu,den=self.predictions(p,cap)
        phi=(np.asarray(cap['theta'])-self.origins[c])/10.
        a=K[c,0]*phi*phi; stretch=np.stack((np.exp(a),np.exp(-a)),axis=1)[:,None,:]
        v=(np.asarray(cap['X'])+mu[:,None,:])/stretch
        invden=1-phi[:,None]*(v@K[c,1:])/self.Rc[c]
        q=v/invden[:,:,None]/s[c]
        back,radial_valid,diag=inverse_radial(q,np.full(len(q),k[c]),delta=DELTA)
        valid=radial_valid & np.all(invden>DELTA,axis=1) & np.all(den>DELTA,axis=1)
        diag.update(frame_failures=int((~valid).sum()),minimum_inverse_denominator=float(np.min(invden)),
                    minimum_forward_denominator=float(np.min(den)))
        return np.where(valid[:,None,None],back,np.nan),valid,diag

    def diagnostics(self,p):
        # QR of each complete frame Jacobian followed by a small SVD avoids
        # squaring its condition number in a normal-matrix rank estimate.
        B,K,k,s=self.unpack(p); blocks=[]
        for batch in self.batches:
            c=batch['c']; J=self._terms(B,K[c],k[c],s[c],batch['phi'],c,True)[3]
            J=host(J)*np.sqrt(2*host(batch['weight']))[:,None,None,None]
            blocks.append(np.linalg.qr(J.reshape(-1,self.np),mode='r'))
        sing=np.linalg.svd(np.vstack(blocks),compute_uv=False)
        tol=sing[0]*1e-8
        rank=int((sing>tol).sum())
        r2=np.sum(B*B,axis=1)
        radial_scale=1+k[:,None]*r2[None,:]
        radial_derivative=1+3*k[:,None]*r2[None,:]
        min_den=[]
        for c in range(4):
            q=s[c]*radial_scale[c,:,None]*B
            den=1+self.domain_phi[c][:,None]*(q@K[c,1:])[None,:]/self.Rc[c]
            min_den.append(float(den.min()))
        bound_contacts=[]
        for c in range(4):
            if .3-abs(K[c,0])<1e-6 or .2-np.max(np.abs(K[c,1:]))<1e-6:
                bound_contacts.append(f'capture_{c+1}_keystone_box')
            if k[c]*self.M>1-1e-6:
                bound_contacts.append(f'capture_{c+1}_beta_upper')
            if self.uniform and min(s[c]-.5,1.5-s[c])<1e-6:
                bound_contacts.append(f'capture_{c+1}_uniform_scale_box')
            if min(radial_scale[c].min(),radial_derivative[c].min(),min_den[c])<DELTA+1e-6:
                bound_contacts.append(f'capture_{c+1}_nonlinear_domain')
        return dict(data_only_singular_values=sing.tolist(),rank=rank,parameters=self.np,
            rank_relative_tolerance=1e-8,condition=float(sing[0]/sing[-1]) if sing[-1]>0 else None,
            B_px=B.tolist(),B_centroid_px=B.mean(axis=0).tolist(),K=K.tolist(),kappa_px_inv2=k.tolist(),
            intrinsic_scales=s.tolist(),frozen_Rc_px=self.Rc.tolist(),frozen_origins_deg=self.origins.tolist(),
            fixed_M_px2=self.M,prior_width_scaled=self.width[self.offset:].tolist(),
            prior_parameter_scales=self.scale[self.offset:].tolist(),prior_noise_px=self.anchor_noise,
            prior_coefficient=self.prior_coefficient,radial_scale_factors=radial_scale.tolist(),
            radial_derivative_factors=radial_derivative.tolist(),minimum_forward_denominators=min_den,
            bound_contacts=bound_contacts,domain_margin=DELTA,
            operational_boxes=dict(absolute_b0_max=.3,absolute_projective_coefficient_max=.2,
                                   beta_upper=1.,uniform_scale=[.5,1.5]))
