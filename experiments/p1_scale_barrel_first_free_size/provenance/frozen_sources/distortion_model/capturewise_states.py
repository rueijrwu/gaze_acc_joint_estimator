"""Native joint horizontal gaze/accommodation with frozen P1 scale.

Frame state GN solves use 2x2 blocks and exact rank-one fixation mean anchors.
Small optical solves alternate with state solves; both gradients are certified.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from .capturewise_reverse import forward,inverse,host,equal_group_weights
from .p1_joint import template

A_REF=.36036036036036034
DEMAND=np.array([A_REF,4.,3.,2.])


class StateProblem:
    def __init__(self,data,p1,p3,p4,*,sigma_A=.25,targets=None,nominal_targets=False,shared=False,xp=np,theta_anchor_override=None,g_override=None,vertices4=None,learned_template=False):
        self.xp=xp;self.data=data;self.shared=shared;self.learned_template=bool(learned_template);self.sigma_A=float(sigma_A);self.sigma_theta=.5
        self.capture=np.asarray(data['capture'],int)-1;self.c=xp.asarray(self.capture);self.n=len(self.capture)
        labels,self.group_np=np.unique(data['group'],return_inverse=True);self.labels=labels;self.group=xp.asarray(self.group_np);self.G=len(labels)
        self.count=xp.bincount(self.group,minlength=self.G).astype(xp.float64)
        self.w=xp.asarray(equal_group_weights(data['group'],data['capture'])*3)
        self.theta0=xp.asarray(data['theta0'] if theta_anchor_override is None else theta_anchor_override,dtype=xp.float64)
        self.g=xp.asarray(p1.get('primary_g',p1.get('g')) if g_override is None else g_override,dtype=xp.float64)
        if self.g.shape!=(self.n,) or bool(xp.any(~xp.isfinite(self.g)|(self.g<=0))):raise ValueError('Finite positive row-aligned frozen g required')
        self.B1=xp.asarray(p1.get('primary_B1',p1.get('B1')));self.p1=np.asarray(p1.get('primary_params',p1.get('params')));self.R1=float(p1.get('frozen_R01_px',p1.get('R01')))
        self.B4=xp.asarray(p4['B4_ref']);
        self.B4_initial=np.asarray(p4['B4_ref'],dtype=float).copy();self.B4_centroid=self.B4_initial.mean(axis=0)
        self.B4_centered=self.B4_initial-self.B4_centroid;self.B4_energy=float(np.sum(self.B4_centered**2))
        if self.learned_template:template(self.B4_centered,np.zeros(3))
        self.R4=xp.asarray(p3['R4'])[self.c];self.Rref=float(np.sqrt(np.mean(np.sum(np.asarray(p4['B4_ref'])**2,axis=1))))
        self.alpha0=np.asarray(p3['alpha']);self.offset0=float(p3['offset'])
        kappa=np.asarray(p4['kappa']);self.s0=float(np.dot(DEMAND-A_REF,kappa)/np.dot(DEMAND-A_REF,DEMAND-A_REF))
        if not np.isfinite(self.s0) or abs(self.s0)*self.Rref**2<1e-7:raise ValueError('Uninformative anchored radial slope; cannot initialize accommodation')
        self.targets_capture=DEMAND.copy() if nominal_targets else A_REF+kappa/self.s0
        if targets is not None:self.targets_capture=np.asarray(targets,dtype=float)
        if self.targets_capture.shape!=(4,) or not np.isfinite(self.targets_capture).all() or np.any(self.targets_capture<0)|np.any(self.targets_capture>6):raise ValueError('Invalid frozen Amean target; no clipping/substitution')
        self.targets=xp.asarray([self.targets_capture[self.capture[np.flatnonzero(self.group_np==i)[0]]] for i in range(self.G)])
        self.X1=xp.asarray(data['X1'],dtype=xp.float64)
        self.vertices4=np.arange(3) if vertices4 is None else np.asarray(vertices4,dtype=int)
        if len(self.vertices4)<2 or len(np.unique(self.vertices4))!=len(self.vertices4) or np.any(self.vertices4<0)|np.any(self.vertices4>2):raise ValueError('Two or three unique P4 slots required')
        raw=data.get('P4_raw',data.get('P4'))
        if raw is None:raise ValueError('Original raw P4 required for retained-vertex centering')
        selected=xp.asarray(raw,dtype=xp.float64)[:,self.vertices4]
        self.X4=selected-selected.mean(axis=1,keepdims=True)
        q0=np.r_[self.alpha0.ravel(),self.offset0,self.s0]
        if shared:
            factor=self.Rref/np.asarray(p3['R4'])
            common=self.alpha0.copy();common[:,1:]*=factor[:,None]
            q0=np.r_[common.mean(axis=0),self.offset0,self.s0]
        self.optical_dimension=len(q0);self.offset_index=self.optical_dimension-2;self.slope_index=self.optical_dimension-1
        self.template_slice=slice(self.optical_dimension,self.optical_dimension+3)
        self.q0=np.r_[q0,np.zeros(3)] if self.learned_template else q0
        self.scales=np.r_[np.tile([.05,.05,.05],1 if shared else 4),10.,abs(self.s0)]
        self.bounds=np.array([[-.25,.25],[-.2,.2],[-.2,.2]]*(1 if shared else 4)+[[-120.,120.],sorted([.05*self.s0,3*self.s0])])
        if self.learned_template:
            self.scales=np.r_[self.scales,np.full(3,.1)];self.bounds=np.vstack((self.bounds,np.tile([-.75,.75],(3,1))))
        self.rejections=0

    def unpack(self,q):
        xp=self.xp;q=xp.asarray(q)
        if self.shared:
            alpha=xp.broadcast_to(q[:3],(self.n,3)).copy();alpha[:,1:]*=(self.R4/self.Rref)[:,None]
        else:alpha=q[:12].reshape(4,3)[self.c]
        return alpha,q[self.offset_index],q[self.slope_index]

    def optical_prior(self,q):
        q=np.asarray(q);a,offset,s=self.unpack(q);a=host(a,self.xp)
        ac=np.array([a[np.flatnonzero(self.capture==c)[0]] if np.any(self.capture==c) else self.alpha0[c] for c in range(4)])
        differences=np.r_[(ac-self.alpha0).ravel()/.05,(q[self.offset_index]-self.offset0)/10,(q[self.slope_index]-self.s0)/abs(self.s0)]
        loss=float(np.sum(differences**2)/14)
        ga=2*(ac-self.alpha0)/(.05**2*14)
        if self.shared:
            ratios=np.array([host(self.R4[self.c==c][0],self.xp)/self.Rref if np.any(self.capture==c) else 0 for c in range(4)])
            grad=np.r_[ga[:,0].sum(),np.sum(ga[:,1]*ratios),np.sum(ga[:,2]*ratios),2*(q[self.offset_index]-self.offset0)/(100*14),2*(q[self.slope_index]-self.s0)/(self.s0**2*14)]
        else:grad=np.r_[ga.ravel(),2*(q[self.offset_index]-self.offset0)/(100*14),2*(q[self.slope_index]-self.s0)/(self.s0**2*14)]
        if self.learned_template:grad=np.r_[grad,np.zeros(3)]
        return loss,grad

    def template_geometry(self,q,derivatives=False):
        if not self.learned_template:
            return (self.B4,None,True) if derivatives else self.B4
        q=np.asarray(host(q,self.xp),dtype=float)
        centered,dB=template(self.B4_centered,q[self.template_slice],True)
        singular=np.linalg.svd(centered,compute_uv=False)
        valid=bool(np.isfinite(centered).all() and singular[-1]>1e-10*singular[0])
        B=self.xp.asarray(centered+self.B4_centroid)
        return (B,self.xp.asarray(dB),valid) if derivatives else B

    def predictions(self,theta,A,q,derivatives=False):
        xp=self.xp;a,offset,s=self.unpack(q)
        f1=forward(self.B1,theta,self.p1[:3],self.p1[3],self.R1,self.g,xp=xp,derivatives=derivatives)
        B,dB,template_valid=self.template_geometry(q,True)
        f4=forward(B,theta,a,offset,self.R4,self.g,s*(A-A_REF),xp,derivatives)
        f4['valid'] &= template_valid
        f4.update(template=B,template_valid=template_valid)
        # Raw centering on retained vertices is essential when one is withheld.
        Z=f4['Z'][:,self.vertices4];f4['selected_prediction']=self.g[:,None,None]*(Z-Z.mean(axis=1,keepdims=True))
        if derivatives:
            for key in ('theta','kappa','alpha'):
                d=f4['raw_d'+key][:,self.vertices4]
                f4['selected_d'+key]=self.g.reshape((-1,)+(1,)*(d.ndim-1))*(d-d.mean(axis=1,keepdims=True))
        if derivatives and self.learned_template:
            raw=xp.einsum('njab,jbp->njap',f4['point_jacobian'],dB)
            f4['dtemplate']=self.g[:,None,None,None]*(raw-raw.mean(axis=1,keepdims=True))
            selected=raw[:,self.vertices4]
            f4['selected_dtemplate']=self.g[:,None,None,None]*(selected-selected.mean(axis=1,keepdims=True))
        return f1,f4,s

    def means(self,A):return self.xp.bincount(self.group,weights=A,minlength=self.G)/self.count

    def mean_anchor_beta(self,q):
        return 2/(self.G*self.sigma_A**2*self.count**2)

    def evaluate(self,theta,A,q=None,derivatives=False,optical=False):
        xp=self.xp;q=self.q0 if q is None else q;theta=xp.asarray(theta);A=xp.asarray(A)
        f1,f4,s=self.predictions(theta,A,q,derivatives);valid=f1['valid']&f4['valid']&xp.isfinite(A)&(A>=0)&(A<=6)
        r1=f1['prediction']-self.X1;r4=f4['selected_prediction']-self.X4
        V=len(self.vertices4)
        data=xp.sum(self.w*(xp.sum(r1*r1,axis=(1,2))/6+xp.sum(r4*r4,axis=(1,2))/(2*V)))
        gaze=xp.sum(self.w*((theta-self.theta0)/self.sigma_theta)**2)
        diff=self.means(A)-self.targets;amean=xp.mean((diff/self.sigma_A)**2)
        prior=self.optical_prior(q)[0] if optical else 0.
        total=float(host(data+gaze+amean,xp))+prior
        if not bool(xp.all(valid)):total=float('inf');self.rejections+=1
        out=dict(loss=total,components=dict(data=float(host(data,xp)),gaze=float(host(gaze,xp)),Amean=float(host(amean,xp)),optical=prior),valid=valid,p1=f1,p4=f4,theta=theta,A=A,means=self.means(A))
        if derivatives:
            j1=f1['dtheta'];jt=f4['selected_dtheta'];ja=f4['selected_dkappa']*s
            gt=self.w*(xp.sum(r1*j1,axis=(1,2))/3+xp.sum(r4*jt,axis=(1,2))/V+2*(theta-self.theta0)/self.sigma_theta**2)
            ga=self.w*xp.sum(r4*ja,axis=(1,2))/V+2*diff[self.group]/(self.G*self.sigma_A**2*self.count[self.group])
            h00=self.w*(xp.sum(j1*j1,axis=(1,2))/3+xp.sum(jt*jt,axis=(1,2))/V+2/self.sigma_theta**2)
            h01=self.w*xp.sum(jt*ja,axis=(1,2))/V;h11=self.w*xp.sum(ja*ja,axis=(1,2))/V
            out.update(gradient=xp.stack((gt,ga),axis=1),h00=h00,h01=h01,h11=h11)
            row=xp.sum(r4[...,None]*f4['selected_dalpha'],axis=(1,2))*self.w[:,None]/V
            if self.shared:
                glob=xp.stack((xp.sum(row[:,0]),xp.sum(row[:,1]*self.R4/self.Rref),xp.sum(row[:,2]*self.R4/self.Rref)))
            else:
                glob=xp.concatenate([xp.sum(row*(self.c==c)[:,None],axis=0) for c in range(4)])
            offset=-xp.sum(self.w*xp.sum(r4*jt,axis=(1,2))/V)
            slope=xp.sum(self.w*xp.sum(r4*f4['selected_dkappa'],axis=(1,2))*(A-A_REF)/V)
            global_gradient=xp.concatenate((glob,xp.asarray([offset,slope])))
            if self.learned_template:
                shape=xp.einsum('n,nja,njap->p',self.w/V,r4,f4['selected_dtemplate'])
                global_gradient=xp.concatenate((global_gradient,shape))
            out['global_gradient']=host(global_gradient,xp)+(self.optical_prior(q)[1] if optical else 0)
        return out

    def certificate(self,theta,A,q,optical=False):
        xp=self.xp;ev=self.evaluate(theta,A,q,True,optical);gradient=ev['gradient'];states=xp.stack((theta,A),axis=1)
        lower=xp.stack((self.theta0-5,xp.zeros(self.n)),axis=1);upper=xp.stack((self.theta0+5,xp.full(self.n,6.)),axis=1)
        pg=xp.where(((states<=lower+1e-7)&(gradient>0))|((states>=upper-1e-7)&(gradient<0)),0,gradient)
        # Remove cohort weighting for a certificate in local px²/state units.
        state_inf=float(host(xp.max(xp.abs(pg/self.w[:,None])*xp.asarray([.5,.25])),xp))
        grad=ev['global_gradient']*self.scales;q=np.asarray(q)
        qtol=1e-7*self.scales
        grad[((q<=self.bounds[:,0]+qtol)&(grad>0))|((q>=self.bounds[:,1]-qtol)&(grad<0))]=0
        return dict(state_projected_gradient_characteristic_inf=state_inf,global_projected_gradient_scaled_inf=float(np.max(abs(grad))),state_stationary=state_inf<2e-5,global_stationary=bool(np.max(abs(grad))<2e-5),invalid_rows=int(host(xp.sum(~ev['valid']),xp)),theta_lower_contacts=int(host(xp.sum(theta<=self.theta0-5+1e-7),xp)),theta_upper_contacts=int(host(xp.sum(theta>=self.theta0+5-1e-7),xp)),A_lower_contacts=int(host(xp.sum(A<=1e-7),xp)),A_upper_contacts=int(host(xp.sum(A>=6-1e-7),xp)),global_native_gradient=ev['global_gradient'],global_bound_contacts=np.isclose(q,self.bounds[:,0],atol=qtol,rtol=0)|np.isclose(q,self.bounds[:,1],atol=qtol,rtol=0))

    def fit_states(self,theta=None,A=None,q=None,maxiter=150,tolerance=2e-5,log=False):
        xp=self.xp;q=self.q0.copy() if q is None else np.asarray(q).copy()
        theta=self.theta0.copy() if theta is None else xp.asarray(theta).copy()
        A=self.targets[self.group].copy() if A is None else xp.asarray(A).copy()
        history=[];damping=1e-8
        for iteration in range(maxiter):
            ev=self.evaluate(theta,A,q,True)
            if not np.isfinite(ev['loss']):raise ValueError('Invalid initial joint state; no clipping optical domain')
            cert=self.certificate(theta,A,q)
            if cert['state_projected_gradient_characteristic_inf']<tolerance:break
            grad=ev['gradient']
            free0=~(((theta<=self.theta0-5+1e-8)&(grad[:,0]>0))|((theta>=self.theta0+5-1e-8)&(grad[:,0]<0)))
            free1=~(((A<=1e-8)&(grad[:,1]>0))|((A>=6-1e-8)&(grad[:,1]<0)))
            h00=xp.where(free0,ev['h00']+damping*self.w,1.)
            h11=xp.where(free1,ev['h11']+damping*self.w,1.)
            h01=xp.where(free0&free1,ev['h01'],0.);det=h00*h11-h01*h01
            i00=xp.where(free0,h11/det,0.);i11=xp.where(free1,h00/det,0.);i01=xp.where(free0&free1,-h01/det,0.)
            step0=-(i00*grad[:,0]+i01*grad[:,1]);step1=-(i01*grad[:,0]+i11*grad[:,1])
            beta=self.mean_anchor_beta(q)
            summed=xp.bincount(self.group,weights=step1,minlength=self.G)
            denom=1+beta*xp.bincount(self.group,weights=i11,minlength=self.G)
            factor=beta*summed/denom
            step0-=i01*factor[self.group];step1-=i11*factor[self.group]
            accepted=False
            for power in range(18):
                fraction=.5**power
                candidate_t=xp.minimum(self.theta0+5,xp.maximum(self.theta0-5,theta+fraction*step0))
                candidate_A=xp.minimum(6,xp.maximum(0,A+fraction*step1))
                trial=self.evaluate(candidate_t,candidate_A,q)
                if np.isfinite(trial['loss']) and trial['loss']<ev['loss']:
                    theta,A=candidate_t,candidate_A;accepted=True;damping=max(damping*.3,1e-12);break
            history.append(dict(iteration=iteration,loss=ev['loss'],state_gradient=cert['state_projected_gradient_characteristic_inf'],step_fraction=fraction,accepted=accepted))
            if log and iteration%20==0:print(f'state iteration {iteration}: L={ev["loss"]:.9g}, gradient={cert["state_projected_gradient_characteristic_inf"]:.3g}',flush=True)
            if not accepted:
                damping*=10
                if damping>1e8:break
        return dict(theta=host(theta,xp),A=host(A,xp),q=q,certificate=self.certificate(theta,A,q),loss=self.evaluate(theta,A,q)['loss'],history=history)

    def fit_joint(self,initial=None,maxiter=30,state_maxiter=100,log=False):
        xp=self.xp
        if initial is None:initial=self.fit_states(maxiter=state_maxiter,log=log)
        theta=xp.asarray(initial['theta']);A=xp.asarray(initial['A']);q=np.asarray(initial['q']).copy();history=[]
        for iteration in range(maxiter):
            def projected_norm(params,gradient):
                projected=np.asarray(gradient)*self.scales
                tolerance=1e-7*self.scales
                projected[((params<=self.bounds[:,0]+tolerance)&(projected>0))|((params>=self.bounds[:,1]-tolerance)&(projected<0))]=0
                return float(np.max(abs(projected)))
            current=self.evaluate(theta,A,q,True,True)
            if not np.isfinite(current['loss']):raise ValueError('Invalid current optics; cannot restart from rejected domain')
            best_q=q.copy();best_loss=current['loss'];best_gradient=projected_norm(best_q,current['global_gradient']);trials=[]
            # Changing solver coordinates reduces its first trial step without
            # altering characteristic units in the physical certificate.
            for coordinate_factor in (1.,.1,.01,.001):
                if best_gradient<1e-6:break
                coordinates=self.scales*coordinate_factor
                rejected_before=self.rejections
                def objective(u):
                    evaluation=self.evaluate(theta,A,u*coordinates,True,True)
                    return evaluation['loss'],evaluation['global_gradient']*coordinates if np.isfinite(evaluation['loss']) else np.zeros(len(q))
                opt=minimize(objective,best_q/coordinates,jac=True,method='L-BFGS-B',bounds=self.bounds/coordinates[:,None],options=dict(maxiter=150,ftol=1e-14,gtol=1e-7*coordinate_factor,maxls=40))
                candidate=opt.x*coordinates;evaluation=self.evaluate(theta,A,candidate,True,True)
                candidate_gradient=projected_norm(candidate,evaluation['global_gradient'])
                roundoff=1e-12*max(1.,abs(best_loss))
                accepted=bool(np.isfinite(evaluation['loss']) and (evaluation['loss']<best_loss or (evaluation['loss']<=best_loss+roundoff and candidate_gradient<best_gradient)))
                trials.append(dict(coordinate_factor=coordinate_factor,loss=evaluation['loss'],physical_projected_gradient_scaled_inf=candidate_gradient,accepted=accepted,success=bool(opt.success),message=str(opt.message),iterations=int(opt.nit),rejected_domain_evaluations=self.rejections-rejected_before))
                if accepted:best_q=candidate.copy();best_loss=evaluation['loss'];best_gradient=candidate_gradient
                if log:print(f'optical coordinate factor {coordinate_factor:g}: L={evaluation["loss"]:.9g}, physical gradient={candidate_gradient:.3g}, accepted={accepted}',flush=True)
            q=best_q;state=self.fit_states(theta,A,q,state_maxiter,log=log);theta=xp.asarray(state['theta']);A=xp.asarray(state['A'])
            cert=self.certificate(theta,A,q,True);ev=self.evaluate(theta,A,q,optical=True)
            history.append(dict(iteration=iteration,loss=ev['loss'],certificate=cert,optical_trials=trials,optical_success=any(t['success'] and t['accepted'] for t in trials),optical_message='Physical gradient certificate controls completion; numerical coordinate retries are recorded'))
            if log:print(f'joint iteration {iteration}: L={ev["loss"]:.9g}, state={cert["state_projected_gradient_characteristic_inf"]:.3g}, optical={cert["global_projected_gradient_scaled_inf"]:.3g}',flush=True)
            if cert['state_stationary'] and cert['global_stationary']:break
        return dict(theta=host(theta,xp),A=host(A,xp),q=q,loss=self.evaluate(theta,A,q,optical=True)['loss'],certificate=self.certificate(theta,A,q,True),history=history)

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
            if self.learned_template:glob[:,6:,self.template_slice]=f4['selected_dtemplate'][lo:hi].reshape(n,2*V,3)*factor
            glob*=xp.asarray(self.scales);st=states[lo:hi]
            h=xp.einsum('nip,niq->npq',st,st);cross=xp.einsum('nip,niq->npq',st,glob)
            inv=xp.linalg.pinv(h,rcond=1e-12)
            projected=glob-xp.einsum('nip,npq->niq',st,xp.einsum('npq,nqk->npk',inv,cross))
            gram+=xp.einsum('n,nip,niq->pq',self.w[lo:hi]/6,projected,projected)
            slope_residual=max(slope_residual,float(host(xp.max(xp.abs(projected[:,:,self.slope_index])),xp)))
        eigen,vectors=np.linalg.eigh(host(gram,xp));singular=np.sqrt(np.maximum(eigen,0))[::-1]
        names=(['b_shared','cx_shared_Rref','cy_shared_Rref'] if self.shared else [f'capture{c+1}_{name}' for c in range(4) for name in ('b','cx','cy')])+['offset_deg','slope_pxminus2_perD']
        if self.learned_template:names+=['template_q1','template_q2','template_q3']
        return dict(dimension=P,parameter_names=names,data_only_profiled_optical_singular_values=singular,relative_singular_threshold=1e-6,data_only_rank_at_relative_1e_minus6=int(np.sum(singular>singular[0]*1e-6)) if singular[0]>0 else 0,weakest_scaled_modes=vectors[:,:min(4,P)],slope_column_after_state_projection_max=slope_residual,method='framewise data-only theta/A projection includes P1 and P4; no priors/bounds; native equal-pattern weights, declared parameter characteristic units',interpretation='Gram spectrum can lose precision in weakest modes. Anchors select D units; data alone leaves slope/A scale null. Learned template may add weak or ambiguous modes.',chunk_size=chunk_size)

    def predict_held(self,vertex,theta,A,q=None):
        q=self.q0 if q is None else q;f1,f4,s=self.predictions(self.xp.asarray(theta),self.xp.asarray(A),q)
        return host(self.g[:,None]*(f4['Z'][:,vertex]-f4['Z'][:,self.vertices4].mean(axis=1)),self.xp)

    def sensitivities(self,theta,A,q=None):
        xp=self.xp;q=self.q0 if q is None else q;f1,f4,s=self.predictions(xp.asarray(theta),xp.asarray(A),q,True)
        JA=f4['dkappa']*s;Jt=f4['dtheta'];Jg=f4['H'];r=xp.asarray(self.data['X4'])-f4['prediction']
        support=xp.sum(JA*JA,axis=(1,2));safe=xp.where(support>1e-12,support,xp.nan)
        result={name:host(value,xp) for name,value in dict(JA_support=support,deltaA_equivalent=xp.sum(JA*r,axis=(1,2))/safe,dA_dtheta_error=-xp.sum(JA*Jt,axis=(1,2))/safe,dA_dg_error=-xp.sum(JA*Jg,axis=(1,2))/safe,weak_A_support=support<=1e-12).items()}
        result.update(deltaA_equiv=result['deltaA_equivalent'],dA_dtheta=result['dA_dtheta_error'],dA_dg=result['dA_dg_error'])
        return result
