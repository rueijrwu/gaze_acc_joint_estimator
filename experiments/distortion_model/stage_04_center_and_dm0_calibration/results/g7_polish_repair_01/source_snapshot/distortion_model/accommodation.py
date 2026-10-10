"""Conditional DM0 shape initialization, bounded frame A and soft full means.

This is a P4 edge marginal initialization, not the final ten-coordinate J.
Only P1 g multiplies the prediction; M(A) is retained accommodation signal.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from .optics import keystone

STATE_TARGET=1e-11
STATE_CERTIFICATE=1e-8
GLOBAL_CERTIFICATE=1e-6


@dataclass(frozen=True)
class DM0Shape:
    b4: np.ndarray
    omega4: float
    aref: float
    m1: float
    k4: tuple=(0.,0.,0.)
    a_bounds: tuple=(0.,6.)
    theta_bounds: tuple=(-20.,20.)

    def __post_init__(self):
        b=np.array(self.b4,dtype=float,copy=True);k=tuple(float(v) for v in self.k4)
        if b.shape!=(3,2) or not np.isfinite(b).all() or np.linalg.norm(b)<=1e-12:
            raise ValueError('finite noncollapsed empirical P4 reference required')
        if len(k)!=3 or not np.isfinite([self.omega4,self.aref,self.m1,*k]).all():
            raise ValueError('finite optical parameters required')
        for name in ['a_bounds','theta_bounds']:
            bounds=tuple(float(v) for v in getattr(self,name))
            if len(bounds)!=2 or not np.isfinite(bounds).all() or bounds[0]>=bounds[1]:raise ValueError('invalid domain')
            object.__setattr__(self,name,bounds)
        if not self.a_bounds[0]<=self.aref<=self.a_bounds[1]:raise ValueError('reference A outside domain')
        b.setflags(write=False);object.__setattr__(self,'b4',b);object.__setattr__(self,'k4',k)


def shape_covariance(relative_covariance):
    """Marginal covariance of four P4 edges from the frozen relative metric."""
    r=np.asarray(relative_covariance,dtype=float)
    if r.shape!=(10,10):raise ValueError('full relative covariance required')
    transform=np.zeros((4,10));transform[:2,4:6]=-np.eye(2);transform[:2,6:8]=np.eye(2)
    transform[2:,4:6]=-np.eye(2);transform[2:,8:10]=np.eye(2)
    covariance=transform@r@transform.T
    np.linalg.cholesky(covariance)
    return covariance


def edges(points):
    return (points[...,1:,:]-points[...,:1,:]).reshape(points.shape[:-2]+(4,))


def scalar_projection(centered_shape,template):
    """Descriptive Euclidean projection; never used to normalize observed P4."""
    template=np.asarray(template,dtype=float);energy=np.sum(template**2)
    if template.shape!=(3,2) or not np.isfinite(template).all() or energy<=1e-12:raise ValueError('invalid diagnostic template')
    rho=np.sum(np.asarray(centered_shape)*template,axis=(-2,-1))/energy
    return rho,np.asarray(centered_shape)-rho[...,None,None]*template


def shape_derivatives(theta,accommodation,model):
    """Composed F4, dA, dAA and scaled-global derivatives (m,alpha,beta,gamma).

    Numerical coordinates: [10*m1,100*alpha4,100*beta4,10*radius*gamma4].
    Derivatives retain M inside the projective denominator and model mean.
    """
    theta,a=np.broadcast_arrays(np.asarray(theta,dtype=float),np.asarray(accommodation,dtype=float))
    xi=theta-model.omega4;m=1+model.m1*(a-model.aref)
    alpha,beta,gamma=model.k4;radius=np.sqrt(np.mean(np.sum(model.b4**2,axis=1)))
    sx,sy=1+alpha*xi**2,1+beta*xi**2
    denominator=1+gamma*xi[...,None]*m[...,None]*model.b4[:,1]
    base=model.b4*np.stack((sx,sy),axis=-1)[...,None,:]
    f=m[...,None,None]*base/denominator[...,None]
    dM=base/denominator[...,None]**2
    da=model.m1*dM
    daa=-2*(gamma*xi[...,None]*model.b4[:,1]*model.m1**2)[...,None]*base/denominator[...,None]**3
    dg=np.zeros(f.shape+(4,));dg[...,0]=(a-model.aref)[...,None,None]*dM/10
    dg[...,0,1]=xi[...,None]**2*m[...,None]*model.b4[:,0]/denominator/100
    dg[...,1,2]=xi[...,None]**2*m[...,None]*model.b4[:,1]/denominator/100
    dg[...,3]=-f*(xi[...,None]*m[...,None]*model.b4[:,1]/denominator)[...,None]/(10*radius)
    dt=(2*xi[...,None,None]*m[...,None,None]*model.b4*np.array([alpha,beta])/denominator[...,None]
        -f*(gamma*m[...,None]*model.b4[:,1]/denominator)[...,None])
    valid=(np.isfinite(f).all(axis=(-2,-1))&(m>0)&(sx>0)&(sy>0)&(denominator>1e-8).all(axis=-1)
           &(theta>=model.theta_bounds[0])&(theta<=model.theta_bounds[1])&(a>=model.a_bounds[0])&(a<=model.a_bounds[1]))
    return {'F4':f,'mu4':f.mean(axis=-2),'edges':edges(f),'dA':edges(da),'dAA':edges(daa),
            'dglobal':(dg[...,1:,:,:]-dg[...,:1,:,:]).reshape(theta.shape+(4,4)),
            'dtheta':edges(dt),'dmu_dA':da.mean(axis=-2),'dmu_dtheta':dt.mean(axis=-2),'valid':valid}


def from_scaled(template,omega4,aref,parameters,deformation=False):
    p=np.asarray(parameters);radius=np.sqrt(np.mean(np.sum(template**2,axis=1)))
    k=(p[1]/100,p[2]/100,p[3]/(10*radius)) if deformation else (0.,0.,0.)
    return DM0Shape(template,omega4,aref,p[0]/10,k)


def parameter_bounds(template,omega4,aref,deformation=False):
    maxa=max(abs(np.array([0.,6.])-aref));m=.5/maxa*10
    upper=[m]
    if deformation:
        xmax=max(abs(np.array([-20.,20.])-omega4))/10
        radius=np.sqrt(np.mean(np.sum(template**2,axis=1)));ymax=max(abs(template[:,1]))/radius
        upper.extend([.5/xmax**2,.5/xmax**2,.5/(xmax*ymax*1.5)])
    return -np.array(upper),np.array(upper)


def domain_margins(model):
    xi=np.array([model.theta_bounds[0]-model.omega4,model.theta_bounds[1]-model.omega4,0.])
    m=1+model.m1*(np.asarray(model.a_bounds)-model.aref);alpha,beta,gamma=model.k4
    den=1+gamma*xi[:,None,None]*m[None,:,None]*model.b4[None,None,:,1]
    return {'min_M':float(m.min()),'max_M':float(m.max()),'min_sx':float((1+alpha*xi**2).min()),
            'min_sy':float((1+beta*xi**2).min()),'min_denominator':float(den.min()),
            'valid_domain':bool(m.min()>0 and (1+alpha*xi**2).min()>0 and (1+beta*xi**2).min()>0 and den.min()>1e-8)}


class ShapeProfile:
    """Profile bounded individual A with coupled full-exposure mean anchors.

    G4 data_indices are the declared near window; outside states stay provisional
    and enter full fixation means. G5 data_indices cover every complete frame.
    No temporal penalty. All residuals use native pixels and one fixed marginal.
    """
    def __init__(self,theta,observed_edges,g,exposure,demand,initial_a,template,omega4,aref,covariance,
                 data_indices=None,data_groups=None,deformation=False,s_a=.25):
        self.theta,self.observed,self.g=np.asarray(theta),np.asarray(observed_edges),np.asarray(g)
        self.exposure,self.demand,self.initial_a=np.asarray(exposure),np.asarray(demand),np.asarray(initial_a).copy()
        self.template,self.omega4,self.aref=template,omega4,aref;self.deformation=deformation;self.s_a=s_a
        if not np.isfinite(s_a) or s_a<=0:raise ValueError('positive soft-anchor scale required')
        n=len(self.theta)
        if self.observed.shape!=(n,4) or any(len(x)!=n for x in [self.g,self.exposure,self.demand,self.initial_a]):raise ValueError('invalid full population')
        if not all(np.isfinite(x).all() for x in [self.theta,self.observed,self.g,self.demand,self.initial_a]) or not np.all(self.g>0):raise ValueError('finite positive P1 scales and states required')
        if not np.array_equal(np.unique(self.exposure),np.arange(max(self.exposure)+1)):raise ValueError('consecutive exposure labels required')
        if np.any(self.initial_a<0) or np.any(self.initial_a>6):raise ValueError('initial A outside bounds')
        self.indices=np.arange(n) if data_indices is None else np.asarray(data_indices,dtype=int)
        if len(self.indices)==0 or len(np.unique(self.indices))!=len(self.indices) or self.indices.min()<0 or self.indices.max()>=n:raise ValueError('invalid conditional data window')
        self.e=self.exposure[self.indices];self.full_counts=np.bincount(self.exposure)
        self.full_demand=np.bincount(self.exposure,weights=self.demand)/self.full_counts
        if not np.allclose(self.demand,self.full_demand[self.exposure],rtol=0,atol=1e-12):raise ValueError('demand varies within exposure')
        fixed=np.ones(n,dtype=bool);fixed[self.indices]=False
        self.fixed_sum=np.bincount(self.exposure[fixed],weights=self.initial_a[fixed],minlength=len(self.full_counts))
        groups=self.exposure if data_groups is None else np.asarray(data_groups)
        unique,inverse,counts=np.unique(groups[self.indices],return_inverse=True,return_counts=True)
        self.weights=1/(len(unique)*counts[inverse]);self.precision=np.linalg.solve(covariance,np.eye(4))
        self.anchor_c=1/(len(self.full_counts)*s_a*s_a)
        self.mean_h=self.anchor_c/self.full_counts**2
        self.last_a=self.initial_a[self.indices].copy();self.cache=None;self.evaluations=0

    def terms(self,parameters,a):
        model=from_scaled(self.template,self.omega4,self.aref,parameters,self.deformation)
        optical=shape_derivatives(self.theta[self.indices],a,model)
        if not optical['valid'].all():raise ValueError('invalid optical proposal')
        prediction=self.g[self.indices,None]*optical['edges'];residual=self.observed[self.indices]-prediction
        wr=residual@self.precision.T
        mean=(np.bincount(self.e,weights=a,minlength=len(self.full_counts))+self.fixed_sum)/self.full_counts
        anchor=mean-self.full_demand
        cost=.5*np.sum(self.weights*np.sum(residual*wr,axis=1))+.5*self.anchor_c*np.sum(anchor**2)
        da=self.g[self.indices,None]*optical['dA'];daa=self.g[self.indices,None]*optical['dAA']
        gradient=-self.weights*np.sum(da*wr,axis=1)+self.anchor_c*anchor[self.e]/self.full_counts[self.e]
        gn=self.weights*np.sum(da*(da@self.precision.T),axis=1)
        curvature=gn-self.weights*np.sum(daa*wr,axis=1)
        return cost,gradient,curvature,gn,optical,residual,wr,anchor

    @staticmethod
    def projected(a,gradient):
        return np.where(((a<=1e-10)&(gradient>0))|((a>=6-1e-10)&(gradient<0)),0.,gradient*6)

    def solve_states(self,parameters):
        if parameters[0]==0:
            # No A information when M is identically one; choose zero anchor loss.
            a=self.full_demand[self.e].copy()
            return a,0,True
        a=self.last_a.copy();success=False
        for iteration in range(80):
            cost,gradient,curvature,gn,*_=self.terms(parameters,a)
            projected=self.projected(a,gradient);norm=np.linalg.norm(projected,np.inf)
            if norm<=STATE_TARGET:success=True;break
            active=((a<=1e-10)&(gradient>0))|((a>=6-1e-10)&(gradient<0))
            diagonal=np.where(curvature>np.maximum(gn*1e-6,1e-18),curvature,np.maximum(gn,1e-18))
            inverse=np.where(active,0.,1/diagonal);rhs=-gradient
            sum_inverse=np.bincount(self.e,weights=inverse,minlength=len(self.full_counts))
            sum_rhs=np.bincount(self.e,weights=inverse*rhs,minlength=len(self.full_counts))
            correction=self.mean_h*sum_rhs/(1+self.mean_h*sum_inverse)
            direction=inverse*(rhs-correction[self.e])
            accepted=False
            for damping in range(12):
                candidate=np.clip(a+direction*(.5**damping),0.,6.)
                candidate_cost,candidate_gradient,*_=self.terms(parameters,candidate)
                allowance=64*np.finfo(float).eps*max(1.,cost)
                candidate_norm=np.linalg.norm(self.projected(candidate,candidate_gradient),np.inf)
                if candidate_cost<cost or (candidate_cost<=cost+allowance and candidate_norm<norm):
                    a=candidate;accepted=True;break
            if not accepted:break
        final_gradient=self.terms(parameters,a)[1]
        success=bool(np.linalg.norm(self.projected(a,final_gradient),np.inf)<=STATE_CERTIFICATE)
        return a,iteration+1,success

    def evaluate(self,parameters):
        p=np.asarray(parameters,dtype=float)
        if self.cache is not None and np.array_equal(p,self.cache[0]):return self.cache[1]
        a,iterations,inner_certified=self.solve_states(p)
        cost,state_gradient,curvature,gn,optical,residual,wr,anchor=self.terms(p,a)
        derivative=self.g[self.indices,None,None]*optical['dglobal'][...,:len(p)]
        gradient=-np.einsum('n,nip,ni->p',self.weights,derivative,wr)
        self.last_a=a.copy();self.evaluations+=1
        outcome={'cost':float(cost),'gradient':gradient,'a':a,'inner_iterations':iterations,'inner_certified':inner_certified,
                 'state_projected_gradient_inf':float(np.linalg.norm(self.projected(a,state_gradient),np.inf)),
                 'anchor_deviation_D':anchor,'data_cost':float(.5*np.sum(self.weights*np.sum(residual*wr,axis=1))),
                 'anchor_cost':float(.5*self.anchor_c*np.sum(anchor**2))}
        self.cache=(p.copy(),outcome);return outcome

    def fun(self,p):
        o=self.evaluate(p);return o['cost'],o['gradient']

    def full_states(self,p):
        result=self.initial_a.copy();result[self.indices]=self.evaluate(p)['a'];return result


def fit_shape(profile_factory,starts):
    outcomes=[];solutions=[]
    for start in starts:
        profile=profile_factory();lower,upper=parameter_bounds(profile.template,profile.omega4,profile.aref,profile.deformation)
        result=minimize(profile.fun,np.asarray(start,dtype=float),jac=True,method='L-BFGS-B',bounds=list(zip(lower,upper)),
                        options={'ftol':1e-14,'gtol':1e-8,'maxiter':100,'maxls':30})
        p=result.x.copy();polish=[]
        for iteration in range(5):
            o=profile.evaluate(p);gradient=o['gradient'];scaled=gradient*(upper-lower)
            projected=np.where(((p-lower<1e-9)&(gradient>0))|((upper-p<1e-9)&(gradient<0)),0.,scaled)
            if np.linalg.norm(projected,np.inf)<=GLOBAL_CERTIFICATE:break
            h=np.empty((len(p),len(p)))
            for j in range(len(p)):
                step=np.eye(len(p))[j]*1e-6
                plus=np.minimum(p+step,upper);minus=np.maximum(p-step,lower)
                h[:,j]=(profile.evaluate(plus)['gradient']-profile.evaluate(minus)['gradient'])/(plus[j]-minus[j])
            h=(h+h.T)/2
            if np.linalg.eigvalsh(h).min()<=0:break
            direction=np.linalg.solve(h,-gradient);accepted=False
            for damping in range(12):
                q=np.clip(p+direction*.5**damping,lower,upper);candidate=profile.evaluate(q)
                gq=candidate['gradient'];pq=np.where(((q-lower<1e-9)&(gq>0))|((upper-q<1e-9)&(gq<0)),0.,gq*(upper-lower))
                allowance=64*np.finfo(float).eps*max(1.,o['cost'])
                if candidate['cost']<=o['cost']+allowance and np.linalg.norm(pq,np.inf)<np.linalg.norm(projected,np.inf):
                    polish.append({'iteration':iteration,'cost_before':o['cost'],'cost_after':candidate['cost'],
                                   'rounding_allowance':allowance,'gradient_before':float(np.linalg.norm(projected,np.inf)),
                                   'gradient_after':float(np.linalg.norm(pq,np.inf))});p=q;accepted=True;break
            if not accepted:break
        o=profile.evaluate(p);g=o['gradient'];projected=np.where(((p-lower<1e-9)&(g>0))|((upper-p<1e-9)&(g<0)),0.,g*(upper-lower))
        # Conditional profiled curvature; finite differences of analytic envelope gradients.
        h=np.empty((len(p),len(p)))
        for j in range(len(p)):
            step=np.eye(len(p))[j]*1e-6;plus=np.minimum(p+step,upper);minus=np.maximum(p-step,lower)
            h[:,j]=(profile.evaluate(plus)['gradient']-profile.evaluate(minus)['gradient'])/(plus[j]-minus[j])
        eigenvalues=np.linalg.eigvalsh((h+h.T)/2);rank=int(np.sum(eigenvalues>max(abs(eigenvalues))*1e-10))
        # Refresh same optimum after curvature probes; snapshots must agree.
        o=profile.evaluate(p);g=o['gradient'];projected=np.where(((p-lower<1e-9)&(g>0))|((upper-p<1e-9)&(g<0)),0.,g*(upper-lower));model=from_scaled(profile.template,profile.omega4,profile.aref,p,profile.deformation);domain=domain_margins(model)
        certified=bool(rank==len(p) and domain['valid_domain'] and o['inner_certified'] and np.linalg.norm(projected,np.inf)<=GLOBAL_CERTIFICATE)
        record={'start':list(start),'scaled_parameters':p.tolist(),'cost':o['cost'],'data_cost':o['data_cost'],'anchor_cost':o['anchor_cost'],
                'optimizer_success':bool(result.success),'optimizer_message':str(result.message),'nfev':int(result.nfev),'nit':int(result.nit),
                'evaluations_with_polish_and_curvature':profile.evaluations,'stationarity_polish':polish,'conditional_rank':rank,
                'profile_curvature_eigenvalues':eigenvalues.tolist(),'scaled_projected_gradient_inf':float(np.linalg.norm(projected,np.inf)),
                'state_projected_gradient_inf':o['state_projected_gradient_inf'],'conditional_fit_certified':certified,
                'domain':domain,'anchor_deviation_D':o['anchor_deviation_D'].tolist()}
        outcomes.append(record);solutions.append((model,profile.full_states(p),profile))
    candidates=[i for i,o in enumerate(outcomes) if o['conditional_fit_certified']]
    best=min(candidates or range(len(outcomes)),key=lambda i:outcomes[i]['cost'])
    return solutions[best][0],solutions[best][1],outcomes[best],outcomes
