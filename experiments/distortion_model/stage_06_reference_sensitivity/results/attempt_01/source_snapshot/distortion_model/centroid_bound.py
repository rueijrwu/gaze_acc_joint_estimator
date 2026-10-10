"""A declared one-degree accommodation bound on the complete centroid law.

The data objective is unchanged. Outward-rounded interval inequalities cover
the whole visual-theta/A rectangle, including optical mean corrections.
Small constrained shared-parameter QPs accompany the chunked GPU state solve.
"""
import numpy as np
from types import SimpleNamespace
from scipy.linalg import null_space
from scipy.optimize import minimize, LinearConstraint, Bounds, nnls, lsq_linear
from .joint import JointDM0
from .interval_ad import Jet, BranchInterval


class Interval:
    def __init__(self, lower, upper=None):
        self.lo=np.asarray(lower,dtype=float)
        self.hi=self.lo if upper is None else np.asarray(upper,dtype=float)

    @staticmethod
    def wrap(value):return value if isinstance(value,Interval) else Interval(value)

    @staticmethod
    def outward(lo,hi):return Interval(np.nextafter(lo,-np.inf),np.nextafter(hi,np.inf))

    def __add__(self,value):
        b=self.wrap(value);return self.outward(self.lo+b.lo,self.hi+b.hi)
    __radd__=__add__
    def __neg__(self):return Interval(-self.hi,-self.lo)
    def __sub__(self,value):return self+-self.wrap(value)
    def __rsub__(self,value):return self.wrap(value)+-self
    def __mul__(self,value):
        b=self.wrap(value);v=np.stack(np.broadcast_arrays(self.lo*b.lo,self.lo*b.hi,self.hi*b.lo,self.hi*b.hi))
        return self.outward(v.min(axis=0),v.max(axis=0))
    __rmul__=__mul__
    def __truediv__(self,value):
        b=self.wrap(value)
        if np.any((b.lo<=0)&(b.hi>=0)):raise ValueError('interval denominator crosses zero')
        return self*self.outward(1/b.hi,1/b.lo)
    def square(self):
        low=np.where((self.lo<=0)&(self.hi>=0),0.,np.minimum(self.lo**2,self.hi**2))
        return self.outward(low,np.maximum(self.lo**2,self.hi**2))


class CentroidBound:
    """Sufficient continuous-domain constraint, not a finite-grid assertion.

Require H_x,theta >= s_min and |H_x,A| <= s_min*1degree/4D.
The second inequality bounds every A0-to-A4 centroid shift by s_min pixels;
the first converts it to at most one degree on a valid monotone inverse.
The derivative bound is conservatively extended to the existing A0..6 domain.
"""
    def __init__(self,spec,initial_parameters,bound_deg=1.,theta_step=.5,a_step=.25,*,slope_floor=None):
        self.spec=spec;self.bound_deg=float(bound_deg)
        ts=np.linspace(*spec.theta_bounds,int(round(np.ptp(spec.theta_bounds)/theta_step))+1)
        aa=np.linspace(*spec.a_bounds,int(round(np.ptp(spec.a_bounds)/a_step))+1)
        tl,al=np.meshgrid(ts[:-1],aa[:-1],indexing='ij')
        th,ah=np.meshgrid(ts[1:],aa[1:],indexing='ij')
        self.theta=Interval(tl.ravel(),th.ravel());self.a=Interval(al.ravel(),ah.ravel())
        slope,_=self.derivatives(initial_parameters)
        self.reference_min_slope=float(slope.lo.min())
        if slope_floor is None:
            if self.reference_min_slope<=0:raise ValueError('positive G6 reference gaze sensitivity required')
            self.slope_floor=.5*self.reference_min_slope
        else:
            if not np.isfinite(slope_floor) or slope_floor<=0:raise ValueError('positive shared gaze slope floor required')
            self.slope_floor=float(slope_floor)
        self.a_slope_cap=self.slope_floor*self.bound_deg/4.
        self.theta_step,self.a_step=theta_step,a_step

    def derivatives(self,p):
        return self._derivatives(self.spec.parameters(np.asarray(p)),self.theta,self.a,Interval)

    def _derivatives(self,params,theta,a,interval):
        s=self.spec
        ar=a-s.aref;t=theta/10.
        c=params.center
        slope=(interval(c[2,0])+ar*c[3,0]+t*(2*c[4,0]))/10.
        accommodation=interval(c[1,0])+t*c[3,0]
        def mean_derivatives(template,omega,k,M,m):
            xi=theta-omega;sx=1+xi.square()*k[0]
            endpoint=theta.lo.v if isinstance(theta.lo,Jet) else theta.lo
            dt=interval(np.zeros_like(endpoint));da=interval(np.zeros_like(endpoint))
            for bx,by in template:
                denominator=1+xi*M*(k[2]*by)
                dt=dt+M*bx*(xi*(2*k[0])/denominator-sx*M*(k[2]*by)/denominator.square())/3.
                da=da+sx*(m*bx)/denominator.square()/3.
            return dt,da
        dt1,_=mean_derivatives(params.b1,params.omega1,params.k1,interval(1.),0.)
        dt4,da4=mean_derivatives(params.b4,params.omega4,params.k4,1+ar*params.m1,params.m1)
        return slope+dt4-dt1,accommodation+da4

    def values(self,p):
        slope,accommodation=self.derivatives(p)
        return np.r_[slope.lo/self.slope_floor-1.,
                     1.-accommodation.hi/self.a_slope_cap,
                     1.+accommodation.lo/self.a_slope_cap]

    def branch_derivatives(self,p,order=1,cells=None):
        """Differentiate selected endpoint algebra; flag genuine active ties.

        The ordinary outward interval values remain the feasibility authority.
        No finite-difference stencil crosses a parameter sign switch here.
        Hessians can evaluate just the active cells to bound memory use.
        """
        if order not in (1,2):raise ValueError('derivative order must be 1 or 2')
        s=self.spec;p=np.asarray(p,dtype=float)
        variables=[Jet.variable(value,j,order) for j,value in enumerate(p)]
        norm=(1+sum(v*v for v in variables[3:6])).power(.5)
        basis=s.template_basis
        template=[[ (s.b4[i,j]+np.sqrt(3)*s.r4*sum(basis[i,j,k]*variables[3+k] for k in range(3)))/norm
                    for j in range(2)] for i in range(3)]
        center=np.array([variables[9+j]*s.r4 for j in range(10)],dtype=object).reshape(5,2)
        params=SimpleNamespace(b1=s.b1,b4=template,omega1=s.omega1,omega4=s.omega4,
            k1=(variables[0]/100,-variables[0]/100,variables[1]/(10*s.r1)),
            k4=(variables[6]/100,variables[7]/100,variables[8]/(10*s.r4)),
            m1=variables[2]/10,center=center)
        indices=np.arange(len(self.theta.lo)) if cells is None else np.asarray(cells,dtype=int)
        interval=lambda lo,hi=None:BranchInterval(lo,hi,order=order)
        theta=interval(self.theta.lo[indices],self.theta.hi[indices])
        a=interval(self.a.lo[indices],self.a.hi[indices])
        slope,accommodation=self._derivatives(params,theta,a,interval)
        outputs=[slope.lo/self.slope_floor-1.,1.-accommodation.hi/self.a_slope_cap,
                 1.+accommodation.lo/self.a_slope_cap]
        result={'values':np.concatenate([v.v for v in outputs]),
                'jacobian':np.concatenate([v.g for v in outputs]),
                'nonsmooth':np.concatenate([v.nonsmooth for v in outputs]),
                'cells':indices}
        if order==2:result['hessian']=np.concatenate([v.h for v in outputs])
        return result

    def jacobian(self,p):return self.branch_derivatives(p)['jacobian']

    def weighted_hessian(self,p,weights):
        weights=np.asarray(weights,dtype=float);n=len(self.theta.lo)
        rows=np.flatnonzero(weights)
        if not len(rows):return np.zeros((19,19)),{'active_branch_status':'SMOOTH','nonsmooth_rows':[]}
        cells=np.unique(rows%n);derivatives=self.branch_derivatives(p,2,cells)
        local=np.array([(row//n)*len(cells)+np.searchsorted(cells,row%n) for row in rows])
        ties=rows[derivatives['nonsmooth'][local]]
        metadata={'active_branch_status':'NONSMOOTH' if len(ties) else 'SMOOTH',
                  'nonsmooth_rows':ties.tolist(),'method':'selected-endpoint forward automatic differentiation, order2'}
        hessian=np.einsum('n,nij->ij',weights[rows],derivatives['hessian'][local])
        return (hessian+hessian.T)/2,metadata

    def record(self,p):
        slope,accommodation=self.derivatives(p)
        maximum=max(float(np.max(np.abs(accommodation.lo))),float(np.max(np.abs(accommodation.hi))))
        minimum=float(slope.lo.min())
        return {'kind':'user-authorized continuous-domain sufficient 1degree bound',
                'bound_deg':self.bound_deg,'accommodation_interval_D':[0.,4.],
                'constraint_derivative_extension_D':list(self.spec.a_bounds),
                'theta_domain_deg':list(self.spec.theta_bounds),'interval_cells':len(self.theta.lo),
                'theta_cell_width_deg':self.theta_step,'A_cell_width_D':self.a_step,
                'reference_interval_min_gaze_slope_px_per_deg':self.reference_min_slope,
                'required_min_gaze_slope_px_per_deg':self.slope_floor,
                'allowed_max_accommodation_slope_px_per_D':self.a_slope_cap,
                'certified_min_gaze_slope_px_per_deg':minimum,
                'certified_max_abs_accommodation_slope_px_per_D':maximum,
                'certified_shift_equivalent_upper_bound_deg':4*maximum/minimum if minimum>0 else None,
                'minimum_normalized_slack':float(self.values(p).min()),
                'feasible':bool(self.values(p).min()>=-1e-10),
                'meaning':'H=D+mu4-mu1 at g=1; bound on horizontal centroid response. Inverse-angle guarantee applies where the monotone inverse lies in the declared theta domain. No bound on individual theta or fixation-mean deviation.'}


class BoundedJointDM0(JointDM0):
    def __init__(self,*args,physical_bound,**kwargs):
        super().__init__(*args,**kwargs);self.physical_bound=physical_bound
        self._cache_p=None;self._cache_c=None;self._cache_j=None;self._cache_nonsmooth=None;self._derivative_mode=False

    def constraints(self,p,jacobian=False):
        p=self.host(p)
        if self._cache_p is None or not np.array_equal(p,self._cache_p):
            self._cache_p=p.copy();self._cache_c=self.physical_bound.values(p);self._cache_j=None;self._cache_nonsmooth=None
        if jacobian and self._cache_j is None:
            derivative=self.physical_bound.branch_derivatives(p)
            self._cache_j=derivative['jacobian'];self._cache_nonsmooth=derivative['nonsmooth']
        return self._cache_c,self._cache_j

    def evaluate(self,x,p,**kwargs):
        if not self._derivative_mode and self.constraints(p)[0].min() < -1e-10:
            raise ValueError('one-degree centroid constraint violated')
        return super().evaluate(x,p,**kwargs)

    def observed_hessian(self,x,p):
        self._derivative_mode=True
        try:return super().observed_hessian(x,p)
        finally:self._derivative_mode=False

    def kkt(self,p,gradient):
        p=self.host(p);gradient=self.host(gradient);scales=self.host(self.global_scales)
        c,j=self.constraints(p,True);indices=np.flatnonzero(c<=1e-7)
        rows=[v for v in j[indices]];slacks=list(c[indices]);physical_count=len(rows)
        lo,hi=self.host(self.lower),self.host(self.upper)
        for i in range(19):
            if np.isfinite(lo[i]) and (p[i]-lo[i])/scales[i]<=1e-8:
                row=np.zeros(19);row[i]=1/scales[i];rows.append(row);slacks.append((p[i]-lo[i])/scales[i])
            if np.isfinite(hi[i]) and (hi[i]-p[i])/scales[i]<=1e-8:
                row=np.zeros(19);row[i]=-1/scales[i];rows.append(row);slacks.append((hi[i]-p[i])/scales[i])
        matrix=np.asarray(rows).reshape(-1,19)
        if len(rows):
            design=matrix.T*scales[:,None];target=gradient*scales
            try:multiplier,_=nnls(design,target,maxiter=max(1000,30*len(rows)))
            except RuntimeError:
                multiplier=lsq_linear(design,target,bounds=(0.,np.inf),method='bvls',tol=1e-12,max_iter=1000).x
            residual=target-design@multiplier
        else:multiplier=np.zeros(0);residual=gradient*scales
        return {'norm':float(np.max(np.abs(residual))),'residual':residual,'multipliers':multiplier,
                'rows':matrix,'physical_indices':indices,'physical_count':physical_count,
                'nonsmooth_active_rows':indices[self._cache_nonsmooth[indices]].tolist(),
                'complementarity_inf':float(np.max(np.abs(multiplier*np.asarray(slacks)),initial=0.))}

    def stationarity(self,x,p,outcome):
        norms=super().stationarity(x,p,outcome)
        norms['global_projected_gradient_inf']=self.kkt(p,outcome['gp'])['norm']
        return norms

    def quadratic_step(self,p,h,rhs,indices):
        indices=np.asarray(indices,dtype=int);p=self.host(p);h=self.host(h);rhs=self.host(rhs)
        c,j=self.constraints(p,True);h=(h+h.T)/2
        if len(indices)==0:return self.xp.zeros(19)
        diagonal=np.maximum(np.abs(np.diag(h)),1e-12);scale=1/np.sqrt(diagonal)
        normalized=h*scale[:,None]*scale[None,:];linear=-rhs*scale
        if np.linalg.eigvalsh(normalized).min()<=0:raise ValueError('nonpositive shared proposal curvature')
        lo=(self.host(self.lower)[indices]-p[indices])/scale
        hi=(self.host(self.upper)[indices]-p[indices])/scale
        constraint=LinearConstraint(j[:,indices]*scale[None,:],-c,np.inf)
        result=minimize(lambda v:.5*v@normalized@v+linear@v,np.zeros(len(indices)),
                        jac=lambda v:normalized@v+linear,method='SLSQP',bounds=Bounds(lo,hi),
                        constraints=[constraint],options={'ftol':1e-12,'maxiter':120})
        if not result.success:raise ValueError('constrained proposal QP: '+result.message)
        step=np.zeros(19);step[indices]=scale*result.x
        if np.min(c+j@step)<-1e-7:raise ValueError('infeasible linearized proposal QP')
        return self.xp.asarray(step)

    def direction(self,x,p,o,kind='joint',regularization=1e-6):
        xp=self.xp
        if kind in ('A','theta'):return super().direction(x,p,o,kind,regularization)
        if kind=='joint':
            active=self.active(x,o['gx'],self.state_lower,self.state_upper)
            apply=self.coupled_inverse(o['hx'],active,regularization);ig=apply(o['gx']);z=apply(o['cross'])
            h=o['hg']-xp.einsum('nsp,nsq->pq',o['cross'],z);h=(h+h.T)/2
            h+=xp.diag(regularization*xp.maximum(xp.abs(xp.diag(o['hg'])),1e-12))
            rhs=-o['gp']+xp.einsum('nsp,ns->p',o['cross'],ig)
            dp=self.quadratic_step(p,h,rhs,np.arange(19))
            return -ig-xp.einsum('nsp,p->ns',z,dp),dp
        blocks={'P1':range(2),'baseline':range(2,6),'K4':range(6,9),'D':range(9,19)}
        indices=np.asarray(list(blocks[kind]));ii=xp.asarray(indices)
        h=o['hg'][xp.ix_(ii,ii)].copy()
        h+=xp.diag(regularization*xp.maximum(xp.abs(xp.diag(h)),1e-12))
        dp=self.quadratic_step(p,h,-o['gp'][ii],indices)
        return xp.zeros_like(x),dp

    def update(self,x,p,kind='joint',observed=False):
        if observed and kind=='joint':
            from .polishing import update
            return update(self,x,p)
        xp=self.xp;o=self.evaluate(x,p,hessian=True);before=self.stationarity(x,p,o)
        if observed:o.update(self.observed_hessian(x,p))
        reasons={};accepted=False;used=None
        for attempt in range(7):
            lm=(0. if observed and attempt==0 else 10.**(attempt-6))
            try:dx,dp=self.direction(x,p,o,kind,lm)
            except (ValueError,np.linalg.LinAlgError) as error:
                key=str(error);reasons[key]=reasons.get(key,0)+1;continue
            cap=xp.maximum(1.,xp.maximum(xp.abs(dx[:,0])/2.,xp.abs(dx[:,1])))
            dx/=cap[:,None]  # Per-frame caps do not shrink the shared global step.
            for damping in range(18):
                step=.5**damping;qx=xp.clip(x+step*dx,self.state_lower,self.state_upper)
                qp=xp.clip(p+step*dp,self.lower,self.upper)
                try:candidate=self.evaluate(qx,qp)
                except ValueError as error:
                    key=str(error);reasons[key]=reasons.get(key,0)+1;continue
                norms=self.stationarity(qx,qp,candidate)
                allowance=64*np.finfo(float).eps*max(1.,o['cost'])
                if candidate['cost']<o['cost'] or (candidate['cost']<=o['cost']+allowance and max(norms.values())<max(before.values())):
                    x,p=qx,qp;accepted=True
                    used={'lm':lm,'damping':damping,'max_frame_trust_cap':self.scalar(cap.max()),'cost_after':candidate['cost'],**norms};break
                reasons['same_objective_line_search']=reasons.get('same_objective_line_search',0)+1
            if accepted:break
        return x,p,{'block':kind,'observed_curvature':observed,'accepted':accepted,'cost_before':o['cost'],
                    'outcome':used,'gradient_before':before,'rejection_counts':reasons}

    def certificate(self,x,p):
        xp=self.xp;o=self.evaluate(x,p,hessian=True);norms=self.stationarity(x,p,o);kkt=self.kkt(p,o['gp'])
        record={**norms,'stationarity_threshold':1e-6,'physical_centroid_constraint':self.physical_bound.record(self.host(p)),
                'states_at_bounds':self.host(xp.sum((x<=self.state_lower+1e-9)|(x>=self.state_upper-1e-9),axis=0)).tolist(),
                'globals_at_bounds':self.host(xp.flatnonzero((p<=self.lower+1e-9)|(p>=self.upper-1e-9))).tolist(),
                'active_physical_inequality_count':kkt['physical_count'],'KKT_multipliers':kkt['multipliers'].tolist(),
                'KKT_complementarity_inf':kkt['complementarity_inf'],'KKT_complementarity_threshold':1e-6,
                'constraint_derivative_method':'selected-endpoint forward automatic differentiation',
                'nonsmooth_active_constraint_rows':kkt['nonsmooth_active_rows'],
                'constraint_derivative_status':'NONSMOOTH_ACTIVE_BRANCH' if kkt['nonsmooth_active_rows'] else 'SMOOTH_ACTIVE_BRANCH'}
        local=xp.linalg.eigvalsh(o['hx']/self.weights[:,None,None]);rank=local[:,0]>local[:,1]*1e-10
        record['state_data_rank2_count']=int(self.scalar(rank.sum()))
        exact=self.observed_hessian(x,p)
        active=self.active(x,o['gx'],self.state_lower,self.state_upper)
        # Record local state curvature independently of later profile failures.
        host_h=self.host(exact['hx']/self.weights[:,None,None]);free=~self.host(active)
        both=free.all(axis=1);one=free.sum(axis=1)==1
        eigenvalues=np.r_[np.linalg.eigvalsh(host_h[both]).ravel(),
                         host_h[one,np.argmax(free[one],axis=1),np.argmax(free[one],axis=1)]]
        state_positive=bool(np.all(eigenvalues>0))
        record.update(free_state_observed_curvature_positive=state_positive,
                      free_state_observed_curvature_status='PASSED' if state_positive else 'FAILED',
                      free_state_eigenvalue_count=len(eigenvalues),
                      free_state_curvature_normalization='observed per-frame Hessian divided by positive objective row weight; sign-only comparison with weighted historical census',
                      negative_free_state_eigenvalue_count=int(np.sum(eigenvalues<0)),
                      minimum_free_state_eigenvalue=float(eigenvalues.min()) if len(eigenvalues) else None,
                      observed_profile_rank=None,observed_profile_positive=None,
                      profile_curvature_evaluated=False,profile_curvature_status='NOT_EVALUATED')
        try:
            if kkt['nonsmooth_active_rows']:raise ValueError('active interval endpoint tie: smooth certificate unavailable')
            apply=self.coupled_inverse(exact['hx'],active);z=apply(exact['cross'])
            schur=self.host(exact['hg']-xp.einsum('nsp,nsq->pq',exact['cross'],z));schur=(schur+schur.T)/2
            # Curvature of the Lagrangian includes the nonlinear inequalities.
            weights=np.zeros(len(self.constraints(p)[0]));count=kkt['physical_count']
            weights[kkt['physical_indices']]=kkt['multipliers'][:count]
            constraint_hessian,branch=self.physical_bound.weighted_hessian(self.host(p),weights)
            record['constraint_curvature']=branch
            if branch['nonsmooth_rows']:raise ValueError('multiplier-weighted interval branch is nonsmooth')
            schur-=(constraint_hessian+constraint_hessian.T)/2
            positive=kkt['multipliers']>1e-8
            basis=null_space(kkt['rows'][positive]) if positive.any() else np.eye(19)
            projected=basis.T@schur@basis
            diagonal=np.diag(projected)
            scaling=np.maximum(np.abs(diagonal),1e-12)
            normal=projected/np.sqrt(scaling[:,None]*scaling[None,:]) if len(diagonal) else projected
            eigen=np.linalg.eigvalsh(normal);condition_positive=not len(eigen) or eigen.min()>max(abs(eigen))*1e-10
            record.update(observed_profile_curvature_eigenvalues=eigen.tolist(),
                          observed_profile_rank=int(np.sum(eigen>max(abs(eigen),default=0.)*1e-10)),
                          observed_profile_positive=bool(condition_positive),
                          constrained_tangent_dimension=basis.shape[1],profile_curvature_evaluated=True,
                          profile_curvature_status='PASSED' if condition_positive else 'FAILED')
        except (ValueError,np.linalg.LinAlgError) as error:
            record['curvature_error']=str(error)
        from .p1 import P1Model,domain_margins as p1_domain
        from .accommodation import DM0Shape,domain_margins as p4_domain
        params=self.spec.parameters(self.host(p))
        record['domain']={'P1':p1_domain(P1Model(params.b1,params.omega1,params.k1,params.theta_bounds)),
                          'P4':p4_domain(DM0Shape(params.b4,params.omega4,params.aref,params.m1,params.k4,params.a_bounds,params.theta_bounds))}
        record['fit_certified']=bool(max(norms.values())<=1e-6 and record['physical_centroid_constraint']['feasible']
            and kkt['complementarity_inf']<=1e-6
            and record['state_data_rank2_count']==self.n and record['observed_profile_positive']
            and record['free_state_observed_curvature_positive'] and all(v['valid_domain'] for v in record['domain'].values()))
        return record
