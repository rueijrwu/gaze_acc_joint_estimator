"""Revision 2 shared raw keystone with P1-only scale and frozen anchors.

All geometry and batch state updates use the supplied NumPy/CuPy float64 backend.
The nine optical coefficients are global; every state is (theta_x,theta_y,A).
No RMS normalization, fitted center, per-frame optical coefficient, or smoothing
is introduced. Gradients include both optical means and the P1 profile scale.
"""
from __future__ import annotations

import numpy as np

A_REF = 0.36036036036036034
DELTA = 1e-8
STATE_SCALE = np.array([.5, .5, .25])


def bounds_from_templates(B1, B4, theta0, A_ref=A_REF):
    """Fixed common-basis optical half-widths, computed before optimization."""
    B1, B4, theta0 = map(lambda a: np.asarray(a, dtype=np.float64), (B1, B4, theta0))
    T = np.maximum(np.max(np.abs(theta0 / 10.), axis=0), .01)
    widths = []
    for B, envelope in ((B1, 1.), (B4, 4 / 3)):
        R = np.sqrt(np.mean(np.sum(B * B, axis=1)))
        lengths = envelope * np.max(np.abs(B), axis=0)
        denominators = [T[0] ** 2, T[1] ** 2,
                        T[0] * lengths[1], T[1] * lengths[0]]
        if not np.all(np.isfinite(denominators)) or min(denominators) <= 0 or R <= 0:
            raise ValueError('UNIDENTIFIABLE_TEMPLATE_BASIS')
        widths.extend([.5 / denominators[0], .5 / denominators[1],
                       .4 * R / denominators[2], .4 * R / denominators[3]])
    M = np.max(np.sum(B4 * B4, axis=1))
    Da = max(A_ref, 6 - A_ref)
    if not np.isfinite(M) or M <= 0:
        raise ValueError('UNIDENTIFIABLE_TEMPLATE_BASIS')
    widths.append((1 - DELTA) / (3 * M * Da))
    return np.array(widths)


def _raw_pattern(B, theta, A, b, slope, A_ref, radial, xp, derivatives):
    """Full uncentered three-vertex map and physical-coordinate derivatives.

    Derivative slots: theta_x,theta_y,A,b0,b1,b2,b3,s.
    """
    n = len(theta)
    tx, ty = theta[:, 0] / 10., theta[:, 1] / 10.
    R = xp.sqrt(xp.mean(xp.sum(B * B, axis=1)))
    r2 = xp.sum(B * B, axis=1)
    kappa = slope * (A - A_ref) if radial else xp.zeros(n, dtype=xp.float64)
    factor = 1 + kappa[:, None] * r2
    radial_derivative = 1 + 3 * kappa[:, None] * r2
    P = B[None] * factor[..., None]
    a = b[0] * tx * tx - b[1] * ty * ty
    stretch = xp.stack((xp.exp(a), xp.exp(-a)), axis=1)[:, None, :]
    den = 1 + (b[2] * tx[:, None] * P[..., 1] + b[3] * ty[:, None] * P[..., 0]) / R
    F = P * stretch / den[..., None]
    result = dict(F=F,mu=F.mean(axis=1),den=den,a=a,radial_factor=factor,
                  radial_derivative=radial_derivative,kappa=kappa)
    if not derivatives:
        return result
    dP = xp.zeros((n, 3, 2, 8), dtype=xp.float64)
    if radial:
        dP[..., 2] = B[None] * (slope * r2)[None, :, None]
        dP[..., 7] = B[None] * ((A - A_ref)[:, None] * r2)[..., None]
    da = xp.zeros((n, 8), dtype=xp.float64)
    da[:, 0] = 2 * b[0] * tx / 10.
    da[:, 1] = -2 * b[1] * ty / 10.
    da[:, 3] = tx * tx
    da[:, 4] = -ty * ty
    dd = (b[2] * tx[:, None, None] * dP[:, :, 1, :] +
          b[3] * ty[:, None, None] * dP[:, :, 0, :]) / R
    dd[..., 0] += b[2] * P[..., 1] / (10 * R)
    dd[..., 1] += b[3] * P[..., 0] / (10 * R)
    dd[..., 5] += tx[:, None] * P[..., 1] / R
    dd[..., 6] += ty[:, None] * P[..., 0] / R
    sign = xp.asarray([1., -1.])
    dF = (dP * stretch[..., None] / den[..., None, None]
          + F[..., None] * (sign[None, None, :, None] * da[:, None, None, :]
                            - dd[:, :, None, :] / den[..., None, None]))
    result['dF'] = dF
    return result


def forward_geometry(B1, B4, observed1, observed4, q, states, A_ref=A_REF,
                     retained_p4=None, xp=np, derivatives=False):
    """Exact model; P4 centering uses only retained slots when requested."""
    B1, B4, observed1, observed4, q, states = [xp.asarray(v, dtype=xp.float64)
        for v in (B1, B4, observed1, observed4, q, states)]
    slots = np.arange(3) if retained_p4 is None else np.asarray(retained_p4, dtype=int)
    if slots.ndim != 1 or len(slots) < 1 or len(set(slots.tolist())) != len(slots) or np.any((slots < 0) | (slots > 2)):
        raise ValueError('Invalid retained P4 slots')
    p1 = _raw_pattern(B1, states[:, :2], states[:, 2], q[:4], q[8], A_ref, False, xp, derivatives)
    p4 = _raw_pattern(B4, states[:, :2], states[:, 2], q[4:8], q[8], A_ref, True, xp, derivatives)
    X1 = observed1 - observed1.mean(axis=1, keepdims=True)
    Y4 = observed4[:, slots]
    X4 = Y4 - Y4.mean(axis=1, keepdims=True)
    H1 = p1['F'] - p1['F'].mean(axis=1, keepdims=True)
    F4I = p4['F'][:, slots]
    H4 = F4I - F4I.mean(axis=1, keepdims=True)
    energy = xp.sum(H1 * H1, axis=(1, 2))
    g = xp.sum(X1 * H1, axis=(1, 2)) / energy
    pred1, pred4 = g[:, None, None] * H1, g[:, None, None] * H4
    out = dict(p1=p1,p4=p4,X1=X1,X4=X4,H1=H1,H4=H4,g=g,energy=energy,
               prediction1=pred1,prediction4=pred4,error1=pred1-X1,error4=pred4-X4)
    if not derivatives:
        return out
    dH1 = p1['dF'] - p1['dF'].mean(axis=1, keepdims=True)
    dF4I = p4['dF'][:, slots]
    dH4 = dF4I - dF4I.mean(axis=1, keepdims=True)
    dg1 = (xp.sum(X1[..., None] * dH1, axis=(1, 2)) -
           2 * g[:, None] * xp.sum(H1[..., None] * dH1, axis=(1, 2))) / energy[:, None]
    # dP1 and dP4 derivative slots become common [state3,global9].
    jac1 = xp.zeros((*H1.shape, 12), dtype=xp.float64)
    jac4 = xp.zeros((*H4.shape, 12), dtype=xp.float64)
    map1 = [0,1,2,3,4,5,6,11]
    map4 = [0,1,2,7,8,9,10,11]
    for source, dest in enumerate(map1):
        jac1[..., dest] += g[:, None, None] * dH1[..., source] + H1 * dg1[:, source, None, None]
        jac4[..., dest] += H4 * dg1[:, source, None, None]
    for source, dest in enumerate(map4):
        jac4[..., dest] += g[:, None, None] * dH4[..., source]
    out.update(jacobian1=jac1,jacobian4=jac4,dg=dg1)
    return out


class SharedKeystone:
    """One all-frame objective with equal fixation weights and frozen priors."""
    def __init__(self, B1, B4, observed1, observed4, fixation, theta0, A_targets,
                 q0=None, xp=np, retained_p4=None, sigma_theta=.5,
                 optical_multiplier=1., sigma_A=.25, A_ref=A_REF, widths=None, bounds_W=None):
        self.xp, self.A_ref = xp, float(A_ref)
        self.retained_p4 = retained_p4
        self.B1, self.B4, self.observed1, self.observed4, self.theta0 = [
            xp.asarray(v, dtype=xp.float64) for v in (B1,B4,observed1,observed4,theta0)]
        self.fixation = xp.asarray(fixation,dtype=xp.int64)
        self.A_targets = xp.asarray(A_targets,dtype=xp.float64)
        self.n, self.nfix = len(observed1), len(A_targets)
        if self.n == 0 or self.theta0.shape != (self.n,2):
            raise ValueError('Invalid states/empty frame population')
        if self.observed1.shape != (self.n,3,2) or self.observed4.shape != (self.n,3,2):
            raise ValueError('Observed arrays must be N,3,2')
        if not bool(xp.all((self.fixation >= 0) & (self.fixation < self.nfix))):
            raise ValueError('Invalid fixation identities')
        self.count = xp.bincount(self.fixation,minlength=self.nfix)
        if bool(xp.any(self.count == 0)):
            raise ValueError('EMPTY_FIXATION')
        self.weights = 1. / (self.nfix * self.count[self.fixation])
        widths = bounds_W if bounds_W is not None else widths
        self.widths = bounds_from_templates(self.host(self.B1),self.host(self.B4),self.host(self.theta0),A_ref) if widths is None else np.asarray(widths,dtype=float)
        if self.widths.shape != (9,) or not np.all(np.isfinite(self.widths) & (self.widths > 0)):
            raise ValueError('Invalid fixed optical widths')
        self.W = self.widths
        self.sigma_q = .25 * self.widths * (optical_multiplier if optical_multiplier > 0 else 1.)
        self.optical_prior = optical_multiplier > 0
        self.sigma_theta = np.broadcast_to(np.asarray(sigma_theta,dtype=float),(2,)).copy()
        self.sigma_A = float(sigma_A)
        self.state_scale = xp.asarray(STATE_SCALE)
        self.q_scale = xp.asarray(self.sigma_q)
        self.state_lower = xp.concatenate((self.theta0-5.,xp.zeros((self.n,1))),axis=1)
        self.state_upper = xp.concatenate((self.theta0+5.,xp.full((self.n,1),6.)),axis=1)
        self.q_lower, self.q_upper = xp.asarray(-self.widths),xp.asarray(self.widths)
        self.reference_energy = float(np.sum(self.host(self.B1)**2))
        self.configure(np.zeros(9) if q0 is None else q0)

    def host(self, x):
        return np.asarray(x) if self.xp is np else self.xp.asnumpy(x)

    def configure(self,q0):
        self.q0 = self.xp.asarray(q0,dtype=self.xp.float64).copy()
        if self.q0.shape != (9,):
            raise ValueError('Expected nine global optics')
        return self

    def means(self, A):
        return self.xp.bincount(self.fixation,weights=A,minlength=self.nfix)/self.count

    def _domain(self,q,states,geom):
        xp = self.xp
        p1,p4 = geom['p1'],geom['p4']
        constraints = dict(p1_denominator=p1['den']-DELTA,p4_denominator=p4['den']-DELTA,
            radial_factor=p4['radial_factor']-DELTA,radial_derivative=p4['radial_derivative']-DELTA,
            p1_stretch_upper=2-p1['a'],p1_stretch_lower=2+p1['a'],
            p4_stretch_upper=2-p4['a'],p4_stretch_lower=2+p4['a'],
            positive_scale=geom['g'],p1_energy=geom['energy']/self.reference_energy-1e-12)
        # Nonlinear constraints have the unit scales required by Revision 2.
        mins=xp.stack([xp.min(v) for v in constraints.values()])
        finite=xp.all(xp.isfinite(p1['F'])) & xp.all(xp.isfinite(p4['F'])) & xp.all(xp.isfinite(mins))
        box=xp.all((states>=self.state_lower)&(states<=self.state_upper)) & xp.all((q>=self.q_lower)&(q<=self.q_upper))
        # Positive g and energy are strict; other domain margins allow equality.
        nonlinear=xp.all(mins>=0) & xp.all(geom['g']>0) & xp.all(geom['energy']>1e-12*self.reference_energy)
        valid=bool(finite & box & nonlinear)
        margins=dict(zip(constraints.keys(),self.host(mins).tolist()))
        return valid,margins,constraints

    def evaluate(self,q,states,derivatives=True,mask=None,hessians=True):
        xp=self.xp
        q,states=xp.asarray(q,dtype=xp.float64),xp.asarray(states,dtype=xp.float64)
        geom=forward_geometry(self.B1,self.B4,self.observed1,self.observed4,q,states,
            self.A_ref,self.retained_p4,xp,derivatives)
        valid,margins,constraints=self._domain(q,states,geom)
        weight=self.weights if mask is None else self.weights*xp.asarray(mask,dtype=bool)
        e1,e4=geom['error1'],geom['error4']
        m=e4.shape[1]
        frame_cost=xp.sum(e1*e1,axis=(1,2))/3+xp.sum(e4*e4,axis=(1,2))/m
        gazediff=(states[:,:2]-self.theta0)/xp.asarray(self.sigma_theta)
        mean_delta=(self.means(states[:,2])-self.A_targets)/self.sigma_A
        dq=(q-self.q0)/self.q_scale
        component_arrays=[xp.sum(weight*frame_cost),xp.sum(weight*xp.sum(gazediff*gazediff,axis=1)),
                          xp.mean(mean_delta*mean_delta),xp.sum(dq*dq) if self.optical_prior else xp.asarray(0.)]
        packed=self.host(xp.stack(component_arrays))
        components=dict(zip(('data','gaze','accommodation_mean','optical'),packed.tolist()))
        value=float(packed.sum())
        valid=bool(valid and np.isfinite(value))
        out=dict(value=value,components=components,valid=valid,margins=margins,
                 geometry=geom,frame_data_cost=frame_cost,constraints=constraints)
        if not derivatives:
            return out
        j1,j4=geom['jacobian1'],geom['jacobian4']
        frame_gradient=2*(xp.sum(e1[...,None]*j1,axis=(1,2))/3+xp.sum(e4[...,None]*j4,axis=(1,2))/m)
        state_gradient=weight[:,None]*frame_gradient[:,:3]
        state_gradient[:,:2]+=2*weight[:,None]*(states[:,:2]-self.theta0)/xp.asarray(self.sigma_theta)**2
        state_gradient[:,2]+=2*(mean_delta/self.sigma_A)[self.fixation]*self.weights
        q_gradient=xp.sum(weight[:,None]*frame_gradient[:,3:],axis=0)
        if self.optical_prior:
            q_gradient+=2*(q-self.q0)/self.q_scale**2
        out.update(state_gradient=state_gradient,q_gradient=q_gradient)
        if hessians:
            # Optional block selection avoids building unused N x 3 x 9 arrays.
            blocks='joint' if hessians is True else hessians
            if blocks in ('joint','state'):
                hs=2*weight[:,None,None]*(xp.einsum('njcp,njcq->npq',j1[...,:3],j1[...,:3])/3+
                       xp.einsum('njcp,njcq->npq',j4[...,:3],j4[...,:3])/m)
                hs[:,0,0]+=2*weight/self.sigma_theta[0]**2
                hs[:,1,1]+=2*weight/self.sigma_theta[1]**2
                out['state_hessian']=hs
            if blocks in ('joint','global'):
                hq=2*(xp.einsum('n,njcp,njcq->pq',weight,j1[...,3:],j1[...,3:])/3+
                        xp.einsum('n,njcp,njcq->pq',weight,j4[...,3:],j4[...,3:])/m)
                if self.optical_prior:hq+=xp.diag(2/self.q_scale**2)
                out['q_hessian']=hq
            if blocks=='joint':
                out['state_q_hessian']=2*weight[:,None,None]*(xp.einsum('njcp,njcq->npq',j1[...,:3],j1[...,3:])/3+
                       xp.einsum('njcp,njcq->npq',j4[...,:3],j4[...,3:])/m)
        return out

    def normalized_projected_gradient(self,q,states,ev=None):
        xp=self.xp
        ev=self.evaluate(q,states,hessians=False) if ev is None else ev
        gs=ev['state_gradient']*self.n*self.state_scale
        gq=ev['q_gradient']*self.n*self.q_scale
        tol=1e-9
        gs=xp.where(((states<=self.state_lower+tol)&(gs>0))|((states>=self.state_upper-tol)&(gs<0)),0.,gs)
        gq=xp.where(((q<=self.q_lower+tol*self.q_scale)&(gq>0))|((q>=self.q_upper-tol*self.q_scale)&(gq<0)),0.,gq)
        return gs,gq

    def _search(self,q,states,dq,ds,current,state_merit_mask=None,global_merit_indices=None):
        """Declared Armijo fractions against the full objective and population."""
        xp=self.xp
        records=[]
        state_merit=xp.ones((self.n,3),dtype=bool) if state_merit_mask is None else xp.broadcast_to(xp.asarray(state_merit_mask,dtype=bool),(self.n,3))
        global_merit=xp.ones(9,dtype=bool)
        if global_merit_indices is not None:
            global_merit[:]=False
            global_merit[list(global_merit_indices)]=True
        gradient_scope=dict(state_coordinate_count=int(self.host(xp.sum(state_merit))),
            global_indices=np.flatnonzero(self.host(global_merit)).tolist(),objective='full L')
        def merit(state_gradient,q_gradient):
            return max(float(self.host(xp.max(xp.abs(xp.where(state_merit,state_gradient,0.))))),
                       float(self.host(xp.max(xp.abs(xp.where(global_merit,q_gradient,0.))))))
        for power in range(25):
            alpha=2.**(-power)
            qt=xp.clip(q+alpha*dq,self.q_lower,self.q_upper)
            st=xp.clip(states+alpha*ds,self.state_lower,self.state_upper)
            directional=float(self.host(xp.dot(current['q_gradient'],qt-q)+xp.sum(current['state_gradient']*(st-states))))
            if not directional<0:
                records.append(dict(fraction=alpha,valid=None,reason='NON_DESCENT',directional=directional))
                continue
            trial=self.evaluate(qt,st,derivatives=False)
            allowance=1e-12*max(1.,abs(current['value']))
            armijo=current['value']+1e-4*directional
            # Roundoff is recorded, never relabeled as a scientific decrease.
            improvement=current['value']-trial['value']
            scientific=improvement>allowance
            roundoff_only=not scientific
            accepted=trial['valid'] and trial['value']<=armijo+allowance
            gradient_before=gradient_after=None
            if accepted and roundoff_only:
                # A numerically equal proposal may improve stationarity. Audit
                # the optimized coordinates of its full J gradient; never
                # present it as a scientific decrease. Frozen gradients cannot
                # prevent a block from reaching its declared stationarity goal.
                before_s,before_q=self.normalized_projected_gradient(q,states,current)
                gradient_before=merit(before_s,before_q)
                trial_gradient=self.evaluate(qt,st,hessians=False)
                after_s,after_q=self.normalized_projected_gradient(qt,st,trial_gradient)
                gradient_after=merit(after_s,after_q)
                accepted=gradient_after<gradient_before
            records.append(dict(fraction=alpha,valid=trial['valid'],value=trial['value'],
                directional=directional,armijo_limit=armijo,roundoff_allowance=allowance,
                scientific_decrease=improvement,scientific_improvement=scientific,
                roundoff_only=roundoff_only,gradient_before=gradient_before,
                gradient_after=gradient_after,gradient_scope=gradient_scope,accepted=accepted))
            if accepted:
                return qt,st,trial,records
        return q,states,current,records

    def optimize_states(self,q,states,mask=None,components=(0,1,2),maxiter=500,tolerance=1e-5):
        """Vectorized normalized GN updates with exact dynamic mean coupling."""
        xp=self.xp
        q=xp.asarray(q,dtype=xp.float64).copy()
        states=xp.asarray(states,dtype=xp.float64).copy()
        selected=xp.ones((self.n,3),dtype=bool)
        if mask is not None:
            selected &= xp.asarray(mask,dtype=bool)[:,None]
        comp=np.zeros(3,dtype=bool);comp[list(components)]=True
        selected &= xp.asarray(comp)[None,:]
        trace=[];status='ITERATION_CAP'
        for iteration in range(maxiter):
            current=self.evaluate(q,states,hessians='state')
            if not current['valid']:
                status='INFEASIBLE_START';break
            projected,_=self.normalized_projected_gradient(q,states,current)
            pg=float(self.host(xp.max(xp.abs(xp.where(selected,projected,0.)))))
            if pg<=tolerance:
                status='BLOCK_STATIONARY';break
            g=current['state_gradient']*self.n*self.state_scale
            H=current['state_hessian']*self.n*self.state_scale[None,:,None]*self.state_scale[None,None,:]
            # Freeze selected variables and outward box directions. The latter
            # can be reconsidered at the next nonlinear iteration.
            free=selected & ~(((states<=self.state_lower+1e-10)&(g>0))|((states>=self.state_upper-1e-10)&(g<0)))
            H*=free[:,:,None]*free[:,None,:]
            for j in range(3):
                H[:,j,j]+=xp.where(free[:,j],1e-8,1.)
            rhs=-xp.where(free,g,0.)
            solution=xp.linalg.solve(H,rhs[...,None])[...,0]
            # Sherman Morrison per fixation for 2N/(F n_f² sigma_A²)
            # outer products, including the frozen/full proposed fixation mean.
            v=xp.zeros_like(solution);v[:,2]=free[:,2]*STATE_SCALE[2]
            hv=xp.linalg.solve(H,v[...,None])[...,0]
            scalar=xp.sum(v*solution,axis=1)
            denom=xp.sum(v*hv,axis=1)
            coeff=2*self.n/(self.nfix*self.count**2*self.sigma_A**2)
            vs=xp.bincount(self.fixation,weights=scalar,minlength=self.nfix)
            vh=xp.bincount(self.fixation,weights=denom,minlength=self.nfix)
            correction=coeff*vs/(1+coeff*vh)
            solution-=hv*correction[self.fixation,None]
            ds=solution*self.state_scale
            if not bool(xp.all(xp.isfinite(ds))):
                status='NONFINITE_DIRECTION';break
            # Feasible clipping belongs to trial construction; unlike an anchor,
            # every frame state is explicitly box constrained by the protocol.
            qt,st,trial,search=self._search(q,states,xp.zeros(9),ds,current,
                state_merit_mask=selected,global_merit_indices=())
            accepted=bool(search and search[-1].get('accepted',False))
            trace.append(dict(iteration=iteration,value=current['value'],projected_gradient=pg,
                              accepted=accepted,line_search=search))
            if not accepted:
                status='LINE_SEARCH_REJECTED';break
            states=st
        else:
            iteration=maxiter
        return dict(q=q,states=states,status=status,trace=trace,iterations=len(trace))

    def optimize_globals(self,q,states,mask=None,indices=None,maxiter=500,tolerance=1e-5):
        """Global normalized BFGS, with restricted proposals and full acceptance."""
        xp=self.xp
        q=xp.asarray(q,dtype=xp.float64).copy()
        states=xp.asarray(states,dtype=xp.float64).copy()
        indices=np.arange(9) if indices is None else np.asarray(indices,dtype=int)
        trace=[];status='ITERATION_CAP';inverse=None;previous=None
        for iteration in range(maxiter):
            current=self.evaluate(q,states,hessians='global')
            if not current['valid']:
                status='INFEASIBLE_START';break
            proposal=current if mask is None else self.evaluate(q,states,mask=mask,hessians='global')
            grad=self.host(proposal['q_gradient']*self.n*self.q_scale)[indices]
            _,pgall=self.normalized_projected_gradient(q,states,proposal)
            pg=float(np.max(np.abs(self.host(pgall)[indices])))
            if pg<=tolerance:
                status='BLOCK_STATIONARY';break
            h=self.host(proposal['q_hessian']*self.n*self.q_scale[:,None]*self.q_scale[None,:])[np.ix_(indices,indices)]
            if inverse is None:
                inverse=np.linalg.pinv(h+np.eye(len(indices))*1e-8)
            if previous is not None:
                step=self.host(q/self.q_scale)[indices]-previous[0]
                y=grad-previous[1];ys=float(y@step)
                if ys>1e-12*np.linalg.norm(y)*np.linalg.norm(step):
                    rho=1/ys;I=np.eye(len(indices));V=I-rho*np.outer(step,y)
                    inverse=V@inverse@V.T+rho*np.outer(step,step)
                else:
                    inverse=np.linalg.pinv(h+np.eye(len(indices))*1e-8)
            direction=-inverse@grad
            dq=xp.zeros(9,dtype=xp.float64)
            dq[indices]=xp.asarray(direction)*self.q_scale[indices]
            qt,st,trial,search=self._search(q,states,dq,xp.zeros_like(states),current,
                state_merit_mask=False,global_merit_indices=indices)
            accepted=bool(search and search[-1].get('accepted',False))
            trace.append(dict(iteration=iteration,value=current['value'],proposal_value=proposal['value'],
                projected_gradient=pg,accepted=accepted,line_search=search))
            if not accepted:
                status='LINE_SEARCH_REJECTED';break
            previous=(self.host(q/self.q_scale)[indices],grad)
            q=qt
        return dict(q=q,states=states,status=status,trace=trace,iterations=len(trace))

    def _mean_coupled_solve(self,H,rhs,free):
        """Invert frame blocks plus the exact fixation A-mean rank-one terms."""
        xp=self.xp
        vector=rhs.ndim==2
        if vector:rhs=rhs[...,None]
        solved=xp.linalg.solve(H,rhs)
        v=xp.zeros((self.n,3),dtype=xp.float64)
        v[:,2]=free[:,2]*STATE_SCALE[2]
        hv=xp.linalg.solve(H,v[...,None])[...,0]
        coeff=2*self.n/(self.nfix*self.count**2*self.sigma_A**2)
        vh=xp.bincount(self.fixation,weights=xp.sum(v*hv,axis=1),minlength=self.nfix)
        for j in range(solved.shape[-1]):
            vs=xp.bincount(self.fixation,weights=xp.sum(v*solved[...,j],axis=1),minlength=self.nfix)
            solved[...,j]-=hv*(coeff*vs/(1+coeff*vh))[self.fixation,None]
        return solved[...,0] if vector else solved

    def optimize_joint(self,q,states,maxiter=500,tolerance=1e-5):
        """Coupled GN Schur updates resolve the anchored data scale gauges."""
        xp=self.xp;q=xp.asarray(q,dtype=xp.float64).copy();states=xp.asarray(states,dtype=xp.float64).copy()
        trace=[];status='ITERATION_CAP'
        for iteration in range(maxiter):
            current=self.evaluate(q,states)
            if not current['valid']:status='INFEASIBLE_START';break
            pgs,pgq=self.normalized_projected_gradient(q,states,current)
            pg=max(float(self.host(xp.max(xp.abs(pgs)))),float(self.host(xp.max(xp.abs(pgq)))))
            if pg<=tolerance:status='JOINT_STATIONARY';break
            gs=current['state_gradient']*self.n*self.state_scale
            gq=current['q_gradient']*self.n*self.q_scale
            sf=~(((states<=self.state_lower+1e-10)&(gs>0))|((states>=self.state_upper-1e-10)&(gs<0)))
            qf=~(((q<=self.q_lower+1e-10*self.q_scale)&(gq>0))|((q>=self.q_upper-1e-10*self.q_scale)&(gq<0)))
            H=current['state_hessian']*self.n*self.state_scale[None,:,None]*self.state_scale[None,None,:]
            C=current['state_q_hessian']*self.n*self.state_scale[None,:,None]*self.q_scale[None,None,:]
            Q=current['q_hessian']*self.n*self.q_scale[:,None]*self.q_scale[None,:]
            H*=sf[:,:,None]*sf[:,None,:];C*=sf[:,:,None]*qf[None,None,:];Q*=qf[:,None]*qf[None,:]
            for j in range(3):H[:,j,j]+=xp.where(sf[:,j],1e-8,1.)
            Q+=xp.diag(xp.where(qf,1e-8,1.))
            gs=xp.where(sf,gs,0.);gq=xp.where(qf,gq,0.)
            rhs=xp.concatenate((gs[...,None],C),axis=2)
            solved=self._mean_coupled_solve(H,rhs,sf)
            ig,iC=solved[...,0],solved[...,1:]
            schur=Q-xp.einsum('nsi,nsj->ij',C,iC)
            right=gq-xp.einsum('nsi,ns->i',C,ig)
            # The theoretical GN Schur is PSD; this tiny normalized damping
            # protects the solve from cancellation along data-only gauge modes.
            schur=(schur+schur.T)/2
            dqnorm=-xp.linalg.solve(schur+xp.eye(9)*1e-8,right)
            dsnorm=-ig-xp.einsum('nsq,q->ns',iC,dqnorm)
            dq=dqnorm*self.q_scale;ds=dsnorm*self.state_scale
            qt,st,trial,search=self._search(q,states,dq,ds,current)
            accepted=bool(search and search[-1].get('accepted',False))
            trace.append(dict(iteration=iteration,value=current['value'],projected_gradient=pg,
                accepted=accepted,line_search=search))
            if not accepted:status='LINE_SEARCH_REJECTED';break
            q,states=qt,st
        return dict(q=q,states=states,status=status,trace=trace,iterations=len(trace))

    def joint_fit(self,q,states,max_cycles=100,block_iterations=500):
        """Alternating all-frame blocks; only the full certificate can pass."""
        xp=self.xp;q=xp.asarray(q,dtype=xp.float64).copy();states=xp.asarray(states,dtype=xp.float64).copy()
        trace=[]
        for cycle in range(max_cycles):
            before=self.evaluate(q,states,derivatives=False)
            sr=self.optimize_states(q,states,maxiter=block_iterations)
            states=sr['states']
            qr=self.optimize_globals(q,states,maxiter=block_iterations)
            q=qr['q']
            jr=self.optimize_joint(q,states,maxiter=block_iterations)
            q,states=jr['q'],jr['states']
            ev=self.evaluate(q,states)
            gs,gq=self.normalized_projected_gradient(q,states,ev)
            pg=max(float(self.host(xp.max(xp.abs(gs)))),float(self.host(xp.max(xp.abs(gq)))))
            trace.append(dict(cycle=cycle,value=ev['value'],components=ev['components'],
                projected_gradient=pg,state_status=sr['status'],global_status=qr['status'],
                state_trace=sr['trace'],global_trace=qr['trace'],joint_status=jr['status'],joint_trace=jr['trace']))
            if ev['valid'] and pg<=1e-5:
                break
            if not any(r['accepted'] for r in sr['trace']+qr['trace']+jr['trace']):
                break
        certificate=self.certificate(q,states,curvature=True)
        return dict(q=q,states=states,trace=trace,certificate=certificate,
                    status=certificate['status'],iterations=len(trace))

    def deterministic_starts(self,q,states):
        xp=self.xp;q=xp.asarray(q);states=xp.asarray(states)
        candidates=[('phase_II',q.copy(),states.copy())]
        for sign in (1,-1):
            st=states.copy();st[:,2]+=sign*.25;candidates.append((f'A_{sign:+d}',q.copy(),st))
        for index_set,label in (([0,4],'horizontal_quadratic'),([8],'slope')):
            for sign in (1,-1):
                qt=q.copy();qt[index_set]+=sign*.1*self.q_scale[index_set]
                candidates.append((f'{label}_{sign:+d}',qt,states.copy()))
        out=[]
        for label,qt,st in candidates:
            qt=xp.clip(qt,self.q_lower,self.q_upper);st=xp.clip(st,self.state_lower,self.state_upper)
            for power in range(25):
                fraction=2.**(-power);qf=q+fraction*(qt-q);sf=states+fraction*(st-states)
                ev=self.evaluate(qf,sf,derivatives=False)
                if ev['valid']:
                    out.append(dict(label=label,q=qf,states=sf,feasible_fraction=fraction));break
            else:
                out.append(dict(label=label,q=q.copy(),states=states.copy(),feasible_fraction=0.,status='START_REJECTED'))
        return out

    def certificate(self,q,states,curvature=False):
        """Normalized J=N L certificate; nonlinear boundary cases stay honest.

        No nonlinear KKT implementation is claimed: such cases are explicitly
        DOMAIN_LIMITED. The structured finite-difference Hessian inertia audit
        includes state/optical coupling; unresolved curvature gives UNCERTIFIED.
        """
        xp=self.xp;q=xp.asarray(q);states=xp.asarray(states)
        ev=self.evaluate(q,states,hessians=False)
        gs,gq=self.normalized_projected_gradient(q,states,ev)
        pg=max(float(self.host(xp.max(xp.abs(gs)))),float(self.host(xp.max(xp.abs(gq)))))
        active={name:int(self.host(xp.sum(arr<=1e-7))) for name,arr in ev['constraints'].items()}
        bound_counts=dict(state_lower=self.host(xp.sum(states<=self.state_lower+1e-7,axis=0)).tolist(),
            state_upper=self.host(xp.sum(states>=self.state_upper-1e-7,axis=0)).tolist(),
            q_lower=np.flatnonzero(self.host(q<=self.q_lower+1e-7*self.q_scale)).tolist(),
            q_upper=np.flatnonzero(self.host(q>=self.q_upper-1e-7*self.q_scale)).tolist())
        result=dict(J=self.n*ev['value'],normalization='J=N*L, theta/.5,A/.25,q/sigma_q',
            projected_gradient_inf=pg,primal_feasible=ev['valid'],margins=ev['margins'],
            nonlinear_active_counts=active,constraint_scales={name:1. for name in active},
            bounds=bound_counts,nonlinear_KKT_certified=False,curvature_certified=False)
        if not ev['valid']:
            result['status']='INFEASIBLE'
        elif any(active.values()):
            result['status']='DOMAIN_LIMITED'
        elif pg>1e-5:
            result['status']='NOT_CONVERGED'
        else:
            result['status']='UNCERTIFIED'
        if curvature and ev['valid'] and not any(active.values()) and pg<=1e-5:
            result['curvature']=self.curvature_diagnostics(q,states)
            # Frame-block inertia and the global Schur complement account for
            # every free joint direction, including the fixation-mean coupling.
            if ev['valid'] and not any(active.values()) and pg<=1e-5:
                if result['curvature'].get('certified_nonnegative',False):
                    result['status']='PASS'
                    result['curvature_certified']=True
                elif result['curvature'].get('negative_feasible_curvature',False):
                    result['status']='NEGATIVE_CURVATURE'
        result['stationary']=ev['valid'] and not any(active.values()) and pg<=1e-5
        return result

    def curvature_diagnostics(self,q,states):
        """Full structured finite-difference Hessian inertia, including cross terms.

        The exact Hessian is frame-block diagonal plus mean-A rank one terms,
        coupled to nine globals. Positive state blocks and a nonnegative Schur
        complement certify its PSD without a high-dimensional random probe.
        Weak active box multipliers cannot exclude feasible curvature directions.
        """
        xp=self.xp;q=xp.asarray(q);states=xp.asarray(states)
        base=self.evaluate(q,states,hessians=False)
        gs=base['state_gradient']*self.n*self.state_scale
        gq=base['q_gradient']*self.n*self.q_scale
        active_s=(states<=self.state_lower+1e-7)|(states>=self.state_upper-1e-7)
        active_q=(q<=self.q_lower+1e-7*self.q_scale)|(q>=self.q_upper-1e-7*self.q_scale)
        if bool(xp.any(active_s&(xp.abs(gs)<=1e-5))) or bool(xp.any(active_q&(xp.abs(gq)<=1e-5))):
            return dict(certified_nonnegative=False,status='UNRESOLVED_WEAK_BOX_MULTIPLIERS')
        sf=~active_s;qf=~active_q
        def build(epsilon):
            H=xp.zeros((self.n,3,3));C=xp.zeros((self.n,3,9));Q=xp.zeros((9,9))
            for k in range(3):
                gradients=[]
                for sign in (1,-1):
                    st=states.copy();st[:,k]+=sign*epsilon*self.state_scale[k]*sf[:,k]
                    ev=self.evaluate(q,st,hessians=False)
                    if not ev['valid']:raise ValueError('STATE_CURVATURE_DOMAIN')
                    gradients.append(ev['state_gradient']*self.n*self.state_scale)
                H[:,:,k]=(gradients[0]-gradients[1])/(2*epsilon)
            # Subtract the effect of simultaneously moving all free A states.
            # This leaves the per-frame data/gaze Hessian before the known rank1.
            freecount=xp.bincount(self.fixation,weights=sf[:,2].astype(xp.float64),minlength=self.nfix)
            coefficient=2*self.n*STATE_SCALE[2]**2/(self.nfix*self.count**2*self.sigma_A**2)
            H[:,2,2]-=coefficient[self.fixation]*freecount[self.fixation]
            for k in range(9):
                if not bool(qf[k]):continue
                sg=[];qg=[]
                for sign in (1,-1):
                    qt=q.copy();qt[k]+=sign*epsilon*self.q_scale[k]
                    ev=self.evaluate(qt,states,hessians=False)
                    if not ev['valid']:raise ValueError('OPTICAL_CURVATURE_DOMAIN')
                    sg.append(ev['state_gradient']*self.n*self.state_scale)
                    qg.append(ev['q_gradient']*self.n*self.q_scale)
                C[:,:,k]=(sg[0]-sg[1])/(2*epsilon)
                Q[:,k]=(qg[0]-qg[1])/(2*epsilon)
            H*=sf[:,:,None]*sf[:,None,:];C*=sf[:,:,None]*qf[None,None,:];Q*=qf[:,None]*qf[None,:]
            for k in range(3):H[:,k,k]+=~sf[:,k]
            Q+=xp.diag((~qf).astype(xp.float64))
            return (H+H.transpose(0,2,1))/2,C,(Q+Q.T)/2
        try:
            H,C,Q=build(1e-4)
            Hfine,Cfine,Qfine=build(5e-5)
            errors=[float(self.host(xp.max(xp.abs(a-b)))) for a,b in ((H,Hfine),(C,Cfine),(Q,Qfine))]
            heigen=xp.linalg.eigvalsh(Hfine)
            hmin=float(self.host(xp.min(heigen)))
            if hmin<=max(errors[0],1e-8):
                # Indefinite frame blocks alone do not imply feasible negative
                # curvature when the positive mean penalty couples them.
                return dict(certified_nonnegative=False,status='UNRESOLVED_STATE_BLOCK_INERTIA',
                    minimum_frame_eigenvalue=hmin,finite_difference_errors=errors)
            inverseC=self._mean_coupled_solve(Hfine,Cfine,sf)
            schur=Qfine-xp.einsum('nsi,nsj->ij',Cfine,inverseC)
            schur=(schur+schur.T)/2
            mineigen=float(self.host(xp.min(xp.linalg.eigvalsh(schur))))
            inverseC0=self._mean_coupled_solve(H,C,sf)
            schur0=Q-xp.einsum('nsi,nsj->ij',C,inverseC0)
            schur_error=float(self.host(xp.linalg.norm(schur-schur0,ord=2)))
            nonnegative=mineigen>=max(1e-8,schur_error) and hmin>max(1e-8,errors[0])
            return dict(certified_nonnegative=bool(nonnegative),
                negative_feasible_curvature=mineigen<-max(1e-5,schur_error),
                minimum_frame_eigenvalue=hmin,minimum_schur_eigenvalue=mineigen,
                schur_step_difference_norm=schur_error,finite_difference_errors=errors,
                finite_difference_steps=[1e-4,5e-5],
                method='full frame3-plus-meanA-rank1 Hessian and global9 Schur inertia',
                strict_box_complementarity_required=True)
        except (ValueError,np.linalg.LinAlgError) as exc:
            return dict(certified_nonnegative=False,status='UNRESOLVED',reason=str(exc))


def inverse_radial(points, kappa, delta=DELTA):
    """CPU radial inverse with the nonsingular branch continuous from r=0."""
    points=np.asarray(points,dtype=np.float64)
    kappa=np.broadcast_to(np.asarray(kappa,dtype=np.float64),(len(points),))
    rd=np.linalg.norm(points,axis=-1)
    k=kappa[:,None]
    finite=np.isfinite(rd)&np.isfinite(k)
    negative=k<0
    turning=np.sqrt(np.divide(-1.,3*k,out=np.full_like(k,np.inf),where=negative))
    reachable=np.where(negative,(2/3)*turning,np.inf)
    valid=finite & (rd<reachable)
    lo=np.zeros_like(rd)
    hi=np.where(negative,np.broadcast_to(turning,rd.shape),rd)
    hi=np.where(valid,hi,0.)
    zero=(k==0)
    # Bisection is safeguarded on the declared monotonic branch. Relative radius
    # tolerance is tightened below 1e-10 even for finite precision camera inputs.
    iterations=0
    for iterations in range(100):
        mid=(lo+hi)/2
        mapped=mid+k*mid**3
        lo=np.where(mapped<rd,mid,lo)
        hi=np.where(mapped>=rd,mid,hi)
        converged=(hi-lo)<=1e-12*np.maximum(hi,1e-12)
        if np.all(converged|~valid|zero):
            break
    r=np.where(zero,rd,(lo+hi)/2)
    derivative=1+3*k*r*r
    valid &= derivative>=delta
    ratio=np.divide(r,rd,out=np.ones_like(rd),where=rd>0)
    back=points*ratio[...,None]
    valid &= np.isfinite(back).all(axis=-1)
    frame_valid=valid.all(axis=1)
    back=np.where(frame_valid[:,None,None],back,np.nan)
    return back,frame_valid,dict(vertex_failures=int(np.sum(~valid)),
        beyond_branch_vertices=int(np.sum(finite & (rd>reachable))),
        turning_point_vertices=int(np.sum(finite & negative & (rd==reachable))),
        singular_vertices=int(np.sum(derivative<delta)),nonfinite_vertices=int(np.sum(~finite)),
        iterations=iterations+1,min_recovered_radial_derivative=float(np.min(derivative[valid])) if valid.any() else None)


def inverse_geometry(B,X,g,theta,b,kappa=None,mu=None,delta=DELTA):
    """Independent CPU inverse using trial-consistent restored optical means.

    X is centered using the same retained set as the forward objective. For
    partial P4, pass its retained modeled mean and corresponding centered points.
    """
    B,X,g,theta,b=[np.asarray(x,dtype=float) for x in (B,X,g,theta,b)]
    n=len(X)
    kappa=np.zeros(n) if kappa is None else np.broadcast_to(np.asarray(kappa,dtype=float),(n,))
    R=np.sqrt(np.mean(np.sum(B*B,axis=1)))
    tx,ty=theta[:,0]/10,theta[:,1]/10
    a=b[0]*tx**2-b[1]*ty**2
    stretch=np.stack((np.exp(a),np.exp(-a)),axis=1)[:,None]
    if mu is None:
        pre=B[None]*(1+kappa[:,None]*np.sum(B*B,axis=1))[...,None]
        den=1+(b[2]*tx[:,None]*pre[...,1]+b[3]*ty[:,None]*pre[...,0])/R
        mu=(pre*stretch/den[...,None]).mean(axis=1)
    mu=np.asarray(mu)
    z=X/g[:,None,None]+mu[:,None]
    unstreched=z/stretch
    inverse_den=1-(b[3]*ty[:,None]*unstreched[...,0]+b[2]*tx[:,None]*unstreched[...,1])/R
    pre=unstreched/inverse_den[...,None]
    back,radial_valid,diagnostics=inverse_radial(pre,kappa,delta)
    valid=radial_valid & np.all(inverse_den>=delta,axis=1) & (g>0) & np.isfinite(pre).all(axis=(1,2))
    diagnostics.update(inverse_denominator_failures=int(np.sum(np.any(inverse_den<delta,axis=1))),
        nonpositive_scale_frames=int(np.sum(g<=0)),frame_failures=int(np.sum(~valid)),
        min_inverse_denominator=float(np.min(inverse_den)))
    return np.where(valid[:,None,None],back,np.nan),valid,diagnostics
