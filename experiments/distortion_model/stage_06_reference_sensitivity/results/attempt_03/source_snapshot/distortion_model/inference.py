"""Frozen DM0 retained-P4 state inference; no held observation or labels input."""
import numpy as np
from .joint import JointDM0
from .geometry import retained_indices


def infer_retained_legacy(spec,parameters,observed_retained,covariance_retained,held_point,xp=np,max_iterations=60):
    """49 fixed grid starts plus one retained-only center proposal.

Inputs contain exactly four P1 edge and four retained P4 coordinates. The
omitted measurement, all-three state, all-three centroid and target labels
cannot enter this API. Select the covariance marginal before inversion.
"""
    y=np.asarray(observed_retained,dtype=float);r=np.asarray(covariance_retained,dtype=float)
    if y.ndim!=2 or y.shape[1]!=8 or not np.isfinite(y).all() or r.shape!=(8,8):
        raise ValueError('finite eight-coordinate retained input and covariance required')
    np.linalg.cholesky(r)
    n=len(y);selected=retained_indices(held_point)
    grid=np.array([(t,a) for t in np.linspace(*spec.theta_bounds,7) for a in np.linspace(*spec.a_bounds,7)])
    starts=50;expanded=np.repeat(y,starts,axis=0)
    padded=np.zeros((n*starts,10));padded[:,selected]=expanded
    # A benign complete SPD matrix is used only to install the P1 marginal.
    # Retained losses use the exact provided eight-coordinate covariance.
    full_cov=np.eye(10);full_cov[:4,:4]=r[:4,:4]
    objective=JointDM0(spec,padded,np.zeros(n*starts,dtype=int),full_cov,np.zeros((1,2)),xp=xp)
    p=xp.asarray(parameters);precision=xp.asarray(np.linalg.solve(r,np.eye(8)))
    observation=xp.asarray(expanded);idx=xp.asarray(selected)
    x=xp.asarray(np.tile(np.vstack((grid,[0.,spec.aref])),(n,1)))
    # Retained-only proposal from one trial theta/A and predicted optical means.
    _,_,_,_,aux=objective.batch(x,p,slice(0,len(x)),False)
    keep=[j for j in range(3) if j!=held_point]
    means=observation.reshape(n,starts,8)[:,49,4:].reshape(n,2,2).mean(axis=1)
    f4mean=aux['F4'].reshape(n,starts,3,2)[:,49,keep,:].mean(axis=1)
    mu1=aux['mu1'].reshape(n,starts,2)[:,49];g=aux['g'].reshape(n,starts)[:,49]
    d_obs=means/g[:,None]-f4mean+mu1
    center=p[9:].reshape(5,2)*spec.r4
    proposal=10*(d_obs[:,0]-center[0,0])/center[2,0]
    x.reshape(n,starts,2)[:,49,0]=xp.clip(proposal,*spec.theta_bounds)

    def evaluate(states):
        prediction,jacobian,_,valid,_=objective.batch(states,p,slice(0,len(states)),False)
        residual=observation-prediction[:,idx];js=jacobian[:,idx,:]
        wr=residual@precision
        cost=.5*xp.sum(residual*wr,axis=1)
        gradient=-xp.einsum('nis,ni->ns',js,wr)
        hessian=xp.einsum('nis,ij,njq->nsq',js,precision,js)
        cost=xp.where(valid,cost,xp.inf)
        return cost,gradient,hessian,valid

    def projected(states,gradient):
        active=objective.active(states,gradient,objective.state_lower,objective.state_upper)
        return xp.where(active,0.,gradient*objective.state_scales)

    for iteration in range(max_iterations):
        cost,gradient,hessian,valid=evaluate(x)
        pg=projected(x,gradient);norm=xp.max(xp.abs(pg),axis=1)
        if bool(xp.all(norm<=1e-6)):break
        active=objective.active(x,gradient,objective.state_lower,objective.state_upper)
        h=hessian.copy();h[:,0,1]=xp.where(active.any(axis=1),0.,h[:,0,1]);h[:,1,0]=h[:,0,1]
        h[:,0,0]=xp.where(active[:,0],1.,h[:,0,0]);h[:,1,1]=xp.where(active[:,1],1.,h[:,1,1])
        h[:,0,0]+=1e-8*xp.maximum(h[:,0,0],1e-12);h[:,1,1]+=1e-8*xp.maximum(h[:,1,1],1e-12)
        rhs=xp.where(active,0.,gradient)
        determinant=h[:,0,0]*h[:,1,1]-h[:,0,1]**2
        dx=xp.stack(((-h[:,1,1]*rhs[:,0]+h[:,0,1]*rhs[:,1])/determinant,
                     (h[:,0,1]*rhs[:,0]-h[:,0,0]*rhs[:,1])/determinant),axis=1)
        cap=xp.maximum(1.,xp.maximum(xp.abs(dx[:,0])/4.,xp.abs(dx[:,1])/2.));dx/=cap[:,None]
        accepted=(norm<=1e-6)|(~valid);updated=x.copy()
        for damping in range(14):
            trial=xp.clip(x+dx*.5**damping,objective.state_lower,objective.state_upper)
            c,g,h,v=evaluate(trial);newnorm=xp.max(xp.abs(projected(trial,g)),axis=1)
            allowance=64*np.finfo(float).eps*xp.maximum(1.,cost)
            good=(~accepted)&v&((c<cost)|((c<=cost+allowance)&(newnorm<norm)))
            updated=xp.where(good[:,None],trial,updated);accepted|=good
            if bool(xp.all(accepted)):break
        x=updated
        if not bool(xp.any((norm>1e-6)&accepted)):break
    cost,gradient,hessian,valid=evaluate(x)
    eigen=xp.linalg.eigvalsh(hessian);rank=(eigen[:,0]>eigen[:,1]*1e-10)
    norm=xp.max(xp.abs(projected(x,gradient)),axis=1)
    eligible=valid&rank&(norm<=1e-6)
    costs=cost.reshape(n,starts);elig=eligible.reshape(n,starts)
    winner=xp.argmin(xp.where(elig,costs,xp.inf),axis=1)
    rows=xp.arange(n);best=x.reshape(n,starts,2)[rows,winner]
    best_cost=costs[rows,winner];best_norm=norm.reshape(n,starts)[rows,winner]
    has=xp.any(elig,axis=1)
    competing=elig&(xp.abs(costs-best_cost[:,None])<=1e-8*xp.maximum(1.,best_cost[:,None]))
    distance=xp.abs(x.reshape(n,starts,2)-best[:,None,:])
    ambiguity=xp.any(competing&(xp.max(distance,axis=2)>1e-4),axis=1)
    bounds=xp.any((best<=objective.state_lower+1e-9)|(best>=objective.state_upper-1e-9),axis=1)
    # Conditional observed curvature at each selected branch. Repeat padding
    # keeps all model inputs retained-only; no calibration state is used.
    base=x.copy();base.reshape(n,starts,2)[:]=best[:,None,:]
    exact=xp.empty((n,2,2))
    for j,step in enumerate((1e-4,1e-5)):
        plus=base.copy();minus=base.copy()
        plus[:,j]=xp.minimum(plus[:,j]+step,objective.state_upper[j]);minus[:,j]=xp.maximum(minus[:,j]-step,objective.state_lower[j])
        gp=evaluate(plus)[1].reshape(n,starts,2)[:,0];gm=evaluate(minus)[1].reshape(n,starts,2)[:,0]
        denominator=(plus[:,j]-minus[:,j]).reshape(n,starts)[:,0]
        exact[:,:,j]=(gp-gm)/denominator[:,None]
    exact=(exact+exact.transpose(0,2,1))/2
    active=objective.active(best,gradient.reshape(n,starts,2)[rows,winner],objective.state_lower,objective.state_upper)
    free=~active;h=exact.copy()
    h[:,0,1]=xp.where(free.all(axis=1),h[:,0,1],0.);h[:,1,0]=h[:,0,1]
    h[:,0,0]=xp.where(free[:,0],h[:,0,0],1.);h[:,1,1]=xp.where(free[:,1],h[:,1,1],1.)
    positive=xp.linalg.eigvalsh(h)[:,0]>0
    certified=has&positive
    return {key:objective.host(value) for key,value in {
        'states':best,'cost':best_cost,'projected_gradient_inf':best_norm,'certified':certified,
        'ambiguous':ambiguity,'at_bounds':bounds,'rank2':rank.reshape(n,starts)[rows,winner],
        'candidate_certified_count':xp.sum(elig,axis=1),'observed_curvature_eigenvalues':xp.linalg.eigvalsh(exact),
        'observed_free_curvature_positive':positive}.items()} | {
        'iterations':iteration+1,'starts':starts}


def infer_retained(spec,parameters,observed_retained,covariance_retained,held_point,xp=np,max_iterations=60):
    """Infer from four P1 edge and four retained P4 coordinates only."""
    if held_point not in (0,1,2):raise ValueError('held_point must be 0, 1 or 2')
    return _infer_frozen(spec,parameters,observed_retained,covariance_retained,held_point,xp,max_iterations)


def infer_all(spec,parameters,observed,covariance,xp=np,max_iterations=60):
    """Label-free all-three inference with the identical multistart contract."""
    return _infer_frozen(spec,parameters,observed,covariance,None,xp,max_iterations)


def _infer_frozen(spec,parameters,observed,covariance,held_point,xp,max_iterations):
    y=np.asarray(observed,dtype=float);r=np.asarray(covariance,dtype=float)
    selected=retained_indices(held_point);width=len(selected)
    if y.ndim!=2 or y.shape[1]!=width or len(y)==0 or not np.isfinite(y).all() or r.shape!=(width,width):
        raise ValueError('finite retained coordinates and matching covariance marginal required')
    if not np.allclose(r,r.T,rtol=1e-12,atol=1e-12):raise ValueError('symmetric covariance required')
    np.linalg.cholesky(r)
    if max_iterations<1:raise ValueError('positive iteration budget required')
    n=len(y);starts=50
    grid=np.array([(t,a) for t in np.linspace(*spec.theta_bounds,7) for a in np.linspace(*spec.a_bounds,7)])
    expanded=np.repeat(y,starts,axis=0);padded=np.zeros((n*starts,10));padded[:,selected]=expanded
    full_cov=np.eye(10);full_cov[:4,:4]=r[:4,:4]
    # JointDM0 supplies the shared forward/Jacobian only. Its evaluate() mean
    # anchors and priors are never called by this application inverse.
    objective=JointDM0(spec,padded,np.zeros(n*starts,dtype=int),full_cov,np.zeros((1,2)),xp=xp)
    p=xp.asarray(parameters);precision=xp.asarray(np.linalg.solve(r,np.eye(width)))
    observation=xp.asarray(expanded);idx=xp.asarray(selected)
    x=xp.asarray(np.tile(np.vstack((grid,[0.,spec.aref])),(n,1)))
    proposal_positions=xp.arange(n)*starts+49
    _,_,_,_,aux=objective.batch(x[proposal_positions],p,proposal_positions,False)
    keep=[j for j in range(3) if j!=held_point]
    means=observation[proposal_positions,4:].reshape(n,len(keep),2).mean(axis=1)
    f4mean=aux['F4'][:,keep,:].mean(axis=1)
    d_obs=means/aux['g'][:,None]-f4mean+aux['mu1']
    center=p[9:].reshape(5,2)*spec.r4
    if abs(float(center[2,0]))>1e-12:
        proposal=10*(d_obs[:,0]-center[0,0])/center[2,0]
        x[proposal_positions,0]=xp.clip(proposal,*spec.theta_bounds)

    def evaluate(states,positions=None):
        sl=slice(0,len(states)) if positions is None else positions
        prediction,jacobian,_,valid,_=objective.batch(states,p,sl,False)
        yy=observation if positions is None else observation[positions]
        residual=yy-prediction[:,idx];js=jacobian[:,idx,:];wr=residual@precision
        cost=.5*xp.sum(residual*wr,axis=1)
        gradient=-xp.einsum('nis,ni->ns',js,wr)
        hessian=xp.einsum('nis,ij,njq->nsq',js,precision,js)
        valid=valid&xp.isfinite(cost)&xp.isfinite(gradient).all(axis=1)&xp.isfinite(hessian).all(axis=(1,2))
        return xp.where(valid,cost,xp.inf),gradient,hessian,valid

    def projected(states,gradient):
        active=objective.active(states,gradient,objective.state_lower,objective.state_upper)
        return xp.where(active,0.,gradient*objective.state_scales)

    # Candidates are independent with frozen globals. Cache accepted evaluations
    # and batch only pending candidates. A candidate exhausting every damping
    # would repeat the identical state on all subsequent iterations.
    cost,gradient,hessian,valid=evaluate(x)
    stalled=~valid;steps_taken=xp.zeros(n*starts,dtype=xp.int16)
    for iteration in range(max_iterations):
        norm=xp.max(xp.abs(projected(x,gradient)),axis=1)
        positions=xp.flatnonzero((norm>1e-6)&valid&(~stalled))
        if len(positions)==0:break
        states=x[positions];gg=gradient[positions]
        active=objective.active(states,gg,objective.state_lower,objective.state_upper)
        h=hessian[positions].copy()
        h[:,0,1]=xp.where(active.any(axis=1),0.,h[:,0,1]);h[:,1,0]=h[:,0,1]
        h[:,0,0]=xp.where(active[:,0],1.,h[:,0,0]);h[:,1,1]=xp.where(active[:,1],1.,h[:,1,1])
        h[:,0,0]+=1e-8*xp.maximum(h[:,0,0],1e-12);h[:,1,1]+=1e-8*xp.maximum(h[:,1,1],1e-12)
        rhs=xp.where(active,0.,gg);determinant=h[:,0,0]*h[:,1,1]-h[:,0,1]**2
        dx=xp.stack(((-h[:,1,1]*rhs[:,0]+h[:,0,1]*rhs[:,1])/determinant,
                     (h[:,0,1]*rhs[:,0]-h[:,0,0]*rhs[:,1])/determinant),axis=1)
        cap=xp.maximum(1.,xp.maximum(xp.abs(dx[:,0])/4.,xp.abs(dx[:,1])/2.));dx/=cap[:,None]
        good_direction=(determinant>0)&xp.isfinite(dx).all(axis=1)
        stalled[positions[~good_direction]]=True
        pending=xp.flatnonzero(good_direction)
        for damping in range(14):
            if len(pending)==0:break
            pp=positions[pending]
            trial=xp.clip(states[pending]+dx[pending]*.5**damping,objective.state_lower,objective.state_upper)
            c,g,h,v=evaluate(trial,pp);newnorm=xp.max(xp.abs(projected(trial,g)),axis=1)
            allowance=64*np.finfo(float).eps*xp.maximum(1.,cost[pp])
            good=v&((c<cost[pp])|((c<=cost[pp]+allowance)&(newnorm<norm[pp])))
            accepted=pp[good]
            x[accepted]=trial[good];cost[accepted]=c[good];gradient[accepted]=g[good]
            hessian[accepted]=h[good];valid[accepted]=v[good];steps_taken[accepted]+=1
            pending=pending[~good]
        stalled[positions[pending]]=True
    cost,gradient,hessian,valid=evaluate(x)
    eigen=xp.linalg.eigvalsh(hessian);rank=eigen[:,0]>eigen[:,1]*1e-10
    norm=xp.max(xp.abs(projected(x,gradient)),axis=1)
    eligible=valid&rank&(norm<=1e-6);costs=cost.reshape(n,starts);elig=eligible.reshape(n,starts)
    winner=xp.argmin(xp.where(elig,costs,xp.inf),axis=1);rows=xp.arange(n);positions=rows*starts+winner
    best=x[positions];has=xp.any(elig,axis=1);best_cost=cost[positions];best_norm=norm[positions]
    competing=elig&(xp.abs(costs-best_cost[:,None])<=1e-8*xp.maximum(1.,best_cost[:,None]))
    distance=xp.abs(x.reshape(n,starts,2)-best[:,None,:])
    ambiguity=has&xp.any(competing&(xp.max(distance,axis=2)>1e-4),axis=1)
    bounds=has&xp.any((best<=objective.state_lower+1e-9)|(best>=objective.state_upper-1e-9),axis=1)
    exact=xp.empty((n,2,2))
    for j,step in enumerate((1e-4,1e-5)):
        plus=best.copy();minus=best.copy()
        plus[:,j]=xp.minimum(plus[:,j]+step,objective.state_upper[j]);minus[:,j]=xp.maximum(minus[:,j]-step,objective.state_lower[j])
        gp=evaluate(plus,positions)[1];gm=evaluate(minus,positions)[1]
        exact[:,:,j]=(gp-gm)/(plus[:,j]-minus[:,j])[:,None]
    exact=(exact+exact.transpose(0,2,1))/2
    active=objective.active(best,gradient[positions],objective.state_lower,objective.state_upper)
    free=~active;h=exact.copy();h[:,0,1]=xp.where(free.all(axis=1),h[:,0,1],0.);h[:,1,0]=h[:,0,1]
    h[:,0,0]=xp.where(free[:,0],h[:,0,0],1.);h[:,1,1]=xp.where(free[:,1],h[:,1,1],1.)
    positive=xp.isfinite(h).all(axis=(1,2))&(xp.linalg.eigvalsh(h)[:,0]>0)
    certified=has&positive;information=hessian[positions]
    scaled_information=information*objective.state_scales[None,:,None]*objective.state_scales[None,None,:]
    scaled_eigen=xp.linalg.eigvalsh(scaled_information)
    cosine=information[:,0,1]/xp.sqrt(information[:,0,0]*information[:,1,1])
    conditional_A=information[:,1,1]-information[:,0,1]**2/information[:,0,0]
    return {key:objective.host(value) for key,value in {
        'states':xp.where(has[:,None],best,xp.nan),'cost':xp.where(has,best_cost,xp.inf),
        'projected_gradient_inf':xp.where(has,best_norm,xp.inf),'certified':certified,'has_candidate':has,
        'ambiguous':ambiguity,'at_bounds':bounds,'rank2':has&rank[positions],
        'candidate_certified_count':xp.sum(elig,axis=1),'candidate_stalled_count':xp.sum(stalled.reshape(n,starts),axis=1),
        'selected_start':winner,'selected_accepted_steps':steps_taken[positions],
        'observed_curvature_eigenvalues':xp.linalg.eigvalsh(exact),
        'observed_free_curvature_positive':has&positive,'active_state_bounds':active,
        'gn_information':xp.where(has[:,None,None],information,xp.nan),
        'scaled_information_eigenvalues':xp.where(has[:,None],scaled_eigen,xp.nan),
        'jacobian_column_cosine':xp.where(has,cosine,xp.nan),
        'conditional_accommodation_information':xp.where(has,conditional_A,xp.nan),
        'scaled_information_condition':xp.where(has,scaled_eigen[:,1]/scaled_eigen[:,0],xp.nan)
    }.items()} | {'iterations':iteration+1,'starts':starts}
