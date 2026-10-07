"""Declared anchor sensitivity and achieved-fixation-mean profile probes."""
import numpy as np
from calibrate_profiled import ProfiledProblem
from calibrate_continuation import envelope_gradients,physical_diagnostics


class SensitivityProblem(ProfiledProblem):
    def __init__(self,*args,theta_anchor_multiplier=1.,A_anchor_multiplier=1.,
                 theta_temporal_multiplier=1.,A_temporal_multiplier=1.,
                 probe_group=None,probe_target_deg=None,probe_strength=0.,probe_scale_deg=1.,**kwargs):
        super().__init__(*args,**kwargs)
        values=[theta_anchor_multiplier,A_anchor_multiplier,theta_temporal_multiplier,A_temporal_multiplier,probe_strength]
        if not np.isfinite(values).all() or min(values)<0 or not np.isfinite(probe_scale_deg) or probe_scale_deg<=0:
            raise ValueError('Sensitivity strengths must be finite/nonnegative and scales positive')
        self.theta_anchor_multiplier=float(theta_anchor_multiplier);self.A_anchor_multiplier=float(A_anchor_multiplier)
        self.theta_temporal_multiplier=float(theta_temporal_multiplier);self.A_temporal_multiplier=float(A_temporal_multiplier)
        self.probe_group=probe_group;self.probe_target_deg=probe_target_deg
        self.probe_strength=float(probe_strength);self.probe_scale_deg=float(probe_scale_deg)
        if probe_strength>0 and (probe_group is None or not 0<=probe_group<self.J or not np.isfinite(probe_target_deg)):
            raise ValueError('Probe requires an explicitly supported training fixation and finite mean target')
        self.legacy_penalty_count=2*self.J+2*len(self.left)+(self.J if self.previous_mean_strength>0 else 0)

    def _scale_penalties(self,penalty):
        result=penalty.copy()
        result[:2*self.J].reshape(-1,2)[:]*=np.sqrt([self.theta_anchor_multiplier,self.A_anchor_multiplier])
        result[2*self.J:2*self.J+2*len(self.left)].reshape(-1,2)[:]*=np.sqrt([self.theta_temporal_multiplier,self.A_temporal_multiplier])
        return result

    def penalties(self,x):
        penalty=self._scale_penalties(super().penalties(x))
        if self.probe_strength==0: return penalty
        theta=self.decode(x)[0]
        achieved=theta[self.groups==self.probe_group].mean()
        return np.r_[penalty,np.sqrt(self.probe_strength)*(achieved-self.probe_target_deg)/self.probe_scale_deg]

    def penalty_jv(self,v,dA):
        result=self._scale_penalties(super().penalty_jv(v,dA))
        if self.probe_strength==0: return result
        shift=15*v.reshape(-1,2)[self.groups==self.probe_group,0].mean()
        return np.r_[result,np.sqrt(self.probe_strength)*shift/self.probe_scale_deg]

    def penalty_jtv(self,w,dA):
        base=self._scale_penalties(np.asarray(w[:self.legacy_penalty_count]))
        result=super().penalty_jtv(base,dA)
        if self.probe_strength>0:
            result.reshape(-1,2)[self.groups==self.probe_group,0]+=15*w[-1]*np.sqrt(self.probe_strength)/(self.probe_scale_deg*self.counts[self.probe_group])
        return result

    def true_objective(self,x,coef,kappa=None):
        value,parts=super().true_objective(x,coef,kappa)
        penalty=self.penalties(x)
        if self.previous_mean_strength>0:
            previous=penalty[2*self.J+2*len(self.left):self.legacy_penalty_count]
            parts['previous_mean_prior']=float(.5*previous@previous)
        if self.probe_strength>0:
            parts['mean_probe']=float(.5*penalty[-1]**2)
        return value,parts

    def physical_envelope_gradients(self,x,coef,omega,include_probe=True):
        grad,minus,plus,exact,near,zero=envelope_gradients(self,x,coef,omega,use_problem_hook=False)
        theta,_,A,chain=self.decode(x)
        mt=np.bincount(self.groups,weights=theta)/self.counts
        ma=np.bincount(self.groups,weights=A)/self.counts
        delta=np.column_stack([(self.theta_anchor_multiplier-1)*(mt-self.targets)[self.groups]/(self.J*self.counts[self.groups]),
            (self.A_anchor_multiplier-1)*(ma-self.demands)[self.groups]/(.25**2*self.J*self.counts[self.groups])])
        differences=np.column_stack([(self.theta_temporal_multiplier-1)*(theta[self.right]-theta[self.left]),
            (self.A_temporal_multiplier-1)*(A[self.right]-A[self.left])/.25**2])*self.link_scale[:,None]**2
        np.add.at(delta,self.left,-differences);np.add.at(delta,self.right,differences)
        if include_probe and self.probe_strength>0:
            mask=self.groups==self.probe_group
            delta[mask,0]+=self.probe_strength*(theta[mask].mean()-self.probe_target_deg)/(self.probe_scale_deg**2*self.counts[self.probe_group])
        grad=grad+delta;minus=minus+delta[:,1];plus=plus+delta[:,1]
        if self.p==1: zero=zero+delta[:,1]*chain
        return grad,minus,plus,exact,near,zero

    def state_curvature_scale(self,x,omega):
        data=self.profile(x,omega);curvature=np.sum(data['local']**2,axis=1)*self.n
        physical=np.column_stack([np.full(self.n,15.),data['dA']/.25])
        curvature+=physical**2*np.array([self.theta_anchor_multiplier,self.A_anchor_multiplier])/(self.J*self.counts[self.groups,None]**2)*self.n
        links=np.zeros(self.n)
        np.add.at(links,self.left,self.link_scale**2);np.add.at(links,self.right,self.link_scale**2)
        curvature+=physical**2*links[:,None]*np.array([self.theta_temporal_multiplier,self.A_temporal_multiplier])*self.n
        if self.previous_mean_strength>0:
            curvature[:,0]+=15**2*self.previous_mean_strength/(self.J*self.previous_mean_scale_deg**2*self.counts[self.groups]**2)*self.n
        if self.probe_strength>0:
            curvature[self.groups==self.probe_group,0]+=15**2*self.probe_strength/(self.probe_scale_deg**2*self.counts[self.probe_group]**2)*self.n
        return np.clip(1/np.sqrt(np.maximum(curvature.ravel(),1e-16)),1e-6,1e6)

    def constrained_mean_stationarity(self,x,coef,omega):
        """KKT of the unprobed objective at the achieved mean, including bounds."""
        if self.probe_group is None: raise ValueError('No supported mean constraint was declared')
        theta=self.decode(x)[0];mask=self.groups==self.probe_group
        original=self.physical_envelope_gradients(x,coef,omega,include_probe=False)[0]
        lower=theta[mask]<=-20+1e-12;upper=theta[mask]>=20-1e-12
        interior=~(lower|upper);gradient=original[mask,0];count=self.counts[self.probe_group]
        if interior.any(): multiplier=-count*float(gradient[interior].mean())
        else:
            low=float(np.max(-count*gradient[lower])) if lower.any() else -np.inf
            high=float(np.min(-count*gradient[upper])) if upper.any() else np.inf
            multiplier=float(np.clip(0.,low,high)) if low<=high else float((low+high)/2)
        shifted=original.copy();shifted[mask,0]+=multiplier/count
        theta_projected=shifted[:,0].copy()
        theta_projected[((theta<=-20+1e-12)&(theta_projected>0))|((theta>=20-1e-12)&(theta_projected<0))]=0
        augmented=physical_diagnostics(self,x,coef,omega)
        # Probe changes only gaze: existing A one-sided/bound/zero tests are exact
        # for both the augmented and original constrained objectives.
        saved=self.probe_strength
        try:
            self.probe_strength=0.
            original_diag=physical_diagnostics(self,x,coef,omega)
        finally: self.probe_strength=saved
        A_scale=.25
        _,minus,plus,exact,_,encodedzero=self.physical_envelope_gradients(x,coef,omega,include_probe=False)
        A=self.decode(x)[2];Aprojected=shifted[:,1].copy()
        Aprojected[((A<=1e-12)&(Aprojected>0))|((A>=6-1e-12)&(Aprojected<0))]=0
        Aprojected[exact]=np.maximum(np.maximum(minus[exact],-plus[exact]),0.)
        if self.p<1: Aprojected[A==0]=0
        optimality=max(float(abs(theta_projected).max()),float(A_scale*abs(Aprojected).max()))
        achieved=float(theta[mask].mean())
        probe_multiplier=self.probe_strength*(achieved-self.probe_target_deg)/self.probe_scale_deg**2 if self.probe_strength>0 else 0.
        return dict(achieved_mean_deg=achieved,estimated_mean_constraint_multiplier=multiplier,
            probe_multiplier=probe_multiplier,multiplier_difference=multiplier-probe_multiplier,
            constrained_physical_projected_optimality=optimality,
            selected_fixation_interior_frame_count=int(interior.sum()),
            zero_A_feasible_descent_count=original_diag['zero_A_feasible_descent_count'],
            zero_A_stationarity_unverified_count=original_diag['zero_A_stationarity_unverified_count'],
            augmented_physical_projected_optimality=augmented['physical_projected_optimality'],
            interpretation='local constrained first-order test at achieved mean; target need not be attained; no global profile certificate or confidence interval')
