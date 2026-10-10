"""Local constrained Newton polishing and same-objective difference ledger.

Broad-fit positive-definite QP safeguards are unchanged. This local alternative
requires a smooth, strictly complementary, independent equality working set,
positive tangent Lagrangian curvature, and checks all other inequalities.
"""
import numpy as np
from scipy.linalg import null_space


def local_direction(objective,x,p,o):
    xp=objective.xp;host=objective.host;ph=host(p)
    kkt=objective.kkt(p,o['gp'])
    if kkt['nonsmooth_active_rows']:raise ValueError('local polish: nonsmooth active endpoint')
    if np.any(kkt['multipliers']<=1e-8):
        raise ValueError('local polish: weak active row requires a different working set')
    c,j=objective.constraints(p,True)
    rows=kkt['rows'];count=kkt['physical_count']
    slacks=list(c[kkt['physical_indices']])
    lo,hi=host(objective.lower),host(objective.upper);scale=host(objective.global_scales)
    for i in range(19):
        if np.isfinite(lo[i]) and (ph[i]-lo[i])/scale[i]<=1e-8:slacks.append((ph[i]-lo[i])/scale[i])
        if np.isfinite(hi[i]) and (hi[i]-ph[i])/scale[i]<=1e-8:slacks.append((hi[i]-ph[i])/scale[i])
    slacks=np.asarray(slacks)
    active=objective.active(x,o['gx'],objective.state_lower,objective.state_upper)
    apply=objective.coupled_inverse(o['hx'],active)
    ig=apply(o['gx']);z=apply(o['cross'])
    schur=host(o['hg']-xp.einsum('nsp,nsq->pq',o['cross'],z));schur=(schur+schur.T)/2
    reduced_gradient=host(o['gp']-xp.einsum('nsp,ns->p',o['cross'],ig))
    weights=np.zeros(len(c));weights[kkt['physical_indices']]=kkt['multipliers'][:count]
    correction,branch=objective.physical_bound.weighted_hessian(ph,weights)
    if branch['nonsmooth_rows']:raise ValueError('local polish: ambiguous constraint Hessian')
    lagrangian=schur-correction
    # Work in declared dimensionless parameter units, then equilibrate the
    # small system; all reported KKT residuals return to those declared units.
    scaled=lagrangian*scale[:,None]*scale[None,:]
    equilibrator=1/np.sqrt(np.maximum(np.abs(np.diag(scaled)),1e-12))
    mapping=scale*equilibrator
    h=lagrangian*mapping[:,None]*mapping[None,:]
    gradient=reduced_gradient*mapping;b=rows*mapping[None,:]
    norms=np.linalg.norm(b,axis=1)
    if np.any(norms==0):raise ValueError('local polish: zero active normal')
    bn=b/norms[:,None];target=-slacks/norms
    rank=int(np.linalg.matrix_rank(bn,tol=1e-12))
    if rank!=len(rows):raise ValueError('local polish: dependent active rows')
    particular=np.linalg.lstsq(bn,target,rcond=1e-12)[0] if len(rows) else np.zeros(19)
    basis=null_space(bn,rcond=1e-12) if len(rows) else np.eye(19)
    tangent=basis.T@h@basis;tangent=(tangent+tangent.T)/2
    diagonal=np.sqrt(np.maximum(np.abs(np.diag(tangent)),1e-12))
    normalized=tangent/(diagonal[:,None]*diagonal[None,:])
    eigen=np.linalg.eigvalsh(normalized)
    if len(eigen) and eigen.min()<=max(abs(eigen))*1e-10:
        raise ValueError('local polish: nonpositive tangent Lagrangian curvature')
    tangent_rhs=-basis.T@(gradient+h@particular)
    tangent_step=np.linalg.solve(normalized,tangent_rhs/diagonal)/diagonal if len(eigen) else np.zeros(0)
    v=particular+basis@tangent_step
    nu=np.linalg.lstsq(bn.T,h@v+gradient,rcond=1e-12)[0] if len(rows) else np.zeros(0)
    multipliers=nu/norms
    if np.any(multipliers < -1e-10):raise ValueError('local polish: negative working-set multiplier')
    # Independent bordered solve with balanced constraint rows.
    bordered=np.block([[h,-bn.T],[bn,np.zeros((len(rows),len(rows)))]] )
    alternate=np.linalg.solve(bordered,np.r_[-gradient,target])
    dp=mapping*v;alternate_dp=mapping*alternate[:19]
    ledger_gradient=(lagrangian@dp+reduced_gradient-rows.T@multipliers)*scale
    feasibility=rows@dp+slacks
    route_difference=float(np.max(np.abs((dp-alternate_dp)/scale)))
    stationarity=float(np.max(np.abs(ledger_gradient)))
    linear_slack=c+j@dp
    box_slack=np.r_[(ph+dp-lo)[np.isfinite(lo)]/scale[np.isfinite(lo)],
                    (hi-ph-dp)[np.isfinite(hi)]/scale[np.isfinite(hi)]]
    if min(linear_slack.min(),box_slack.min(initial=np.inf)) < -1e-10:
        raise ValueError('local polish: new blocking inequality; working-set update required')
    if stationarity>1e-8 or route_difference>1e-10 or np.max(np.abs(feasibility),initial=0)>1e-10:
        raise ValueError('local polish: small-system accuracy check failed')
    dx=-ig-xp.einsum('nsp,p->ns',z,xp.asarray(dp))
    # Local proposals may not jump to a different state-bound working set.
    qx=x+dx
    if bool(xp.any(qx<objective.state_lower-1e-10)|xp.any(qx>objective.state_upper+1e-10)):
        raise ValueError('local polish: new blocking state bound')
    record={'method':'equilibrated null-space Lagrangian Newton; independent bordered KKT check',
            'active_row_rank':rank,'active_rows':len(rows),'tangent_dimension':basis.shape[1],
            'physical_active_indices':kkt['physical_indices'].tolist(),
            'tangent_normalized_eigenvalues':eigen.tolist(),
            'objective_full_space_min_eigenvalue':float(np.linalg.eigvalsh(schur*scale[:,None]*scale[None,:]).min()),
            'lagrangian_full_space_min_eigenvalue':float(np.linalg.eigvalsh(scaled).min()),
            'subproblem_scaled_stationarity_inf':stationarity,
            'subproblem_active_feasibility_inf':float(np.max(np.abs(feasibility),initial=0)),
            'subproblem_complementarity_inf':float(np.max(np.abs(multipliers*feasibility),initial=0)),
            'minimum_multiplier':float(multipliers.min()) if len(multipliers) else None,
            'multipliers':multipliers.tolist(),'independent_route_scaled_step_difference_inf':route_difference,
            'minimum_linearized_physical_slack':float(linear_slack.min()),
            'minimum_linearized_box_slack':float(box_slack.min()) if len(box_slack) else None,
            'shared_step':dp.tolist(),'shared_step_scaled':(dp/scale).tolist(),
            'predicted_reduced_quadratic_change':float(reduced_gradient@dp+.5*dp@lagrangian@dp),
            'full_objective_directional_change':float(xp.sum(o['gx']*dx)+xp.sum(o['gp']*xp.asarray(dp))),
            'state_step_max_deg_D':host(xp.max(xp.abs(dx),axis=0)).tolist(),
            'original_state_active_count':int(host(active).sum()),
            'weak_active_policy':'reject rather than freeze weak active rows',
            'tie_policy':'reject nonsmooth active branches; retain outward feasibility checks'}
    return dx,xp.asarray(dp),record


def objective_change(objective,x,p,qx,qp):
    """Evaluate the same J difference without subtracting accumulated costs."""
    xp=objective.xp;point=xp.asarray(0.);max_prediction=xp.asarray(0.)
    for start in range(0,objective.n,objective.chunk_size):
        sl=slice(start,min(start+objective.chunk_size,objective.n))
        f,_,_,valid,_=objective.batch(x[sl],p,sl,False)
        qf,_,_,qvalid,_=objective.batch(qx[sl],qp,sl,False)
        if not bool(xp.all(valid)&xp.all(qvalid)):raise ValueError('inadmissible direct-difference optical proposal')
        r=objective.y[sl]-f;df=qf-f
        point+=xp.sum(objective.weights[sl]*(-xp.einsum('ni,ij,nj->n',r,objective.precision,df)
                      +.5*xp.einsum('ni,ij,nj->n',df,objective.precision,df)))
        max_prediction=xp.maximum(max_prediction,xp.max(xp.abs(df)))
    dev=objective.group_sum(x)/objective.counts[:,None]-objective.target
    change=objective.group_sum(qx-x)/objective.counts[:,None]
    anchors=xp.sum((dev*change+.5*change**2)/objective.mean_scales**2,axis=0)/objective.k
    dp=qp-p;prior=xp.sum(objective.prior*(p*dp+.5*dp**2))
    parts={'point':float(point),'theta_anchor':float(anchors[0]),'A_anchor':float(anchors[1]),
           'regularization':float(prior),'temporal':0.}
    return {'components':parts,'total':sum(parts.values()),'maximum_prediction_change_px':float(max_prediction)}


def trial(objective,x,p,o,dx,dp,damping):
    xp=objective.xp;step=.5**damping
    qx=xp.clip(x+step*dx,objective.state_lower,objective.state_upper)
    qp=p+step*dp;linear=objective.physical_bound.values(objective.host(p))
    jac=objective.physical_bound.jacobian(objective.host(p));predicted=linear+jac@objective.host(step*dp)
    actual=objective.physical_bound.values(objective.host(qp))
    record={'damping':damping,'step_fraction':step,
            'minimum_predicted_constraint_slack':float(predicted.min()),
            'minimum_actual_constraint_slack':float(actual.min()),
            'maximum_constraint_linearization_error':float(np.max(np.abs(actual-predicted)))}
    hostp=objective.host(qp);lo,hi=objective.host(objective.lower),objective.host(objective.upper)
    scale=objective.host(objective.global_scales)
    margin=np.r_[(hostp-lo)[np.isfinite(lo)]/scale[np.isfinite(lo)],
                 (hi-hostp)[np.isfinite(hi)]/scale[np.isfinite(hi)]]
    record.update(minimum_actual_box_slack=float(margin.min()) if len(margin) else None,
                  active_physical_rows_before=np.flatnonzero(linear<=1e-7).tolist(),
                  active_physical_rows_after=np.flatnonzero(actual<=1e-7).tolist(),
                  center_sA_x_sign_changed=bool(np.sign(objective.host(p)[15])!=np.sign(hostp[15])))
    if len(margin) and margin.min() < -1e-10:
        record.update(accepted=False,reason='parameter_box');return x,p,record
    if actual.min() < -1e-10:
        record.update(accepted=False,reason='nonlinear_physical_constraint');return x,p,record
    derivative=objective.physical_bound.branch_derivatives(objective.host(qp))
    active_rows=np.flatnonzero(actual<=1e-7)
    if derivative['nonsmooth'][active_rows].any():
        record.update(accepted=False,reason='nonsmooth_candidate_active_branch');return x,p,record
    candidate=objective.evaluate(qx,qp)
    norms=objective.stationarity(qx,qp,candidate);before=objective.stationarity(x,p,o)
    direct=objective_change(objective,x,p,qx,qp)
    allowance=64*np.finfo(float).eps*max(1.,o['cost'])
    naive=candidate['cost']-o['cost']
    accepted=(direct['total']<0 and naive<=allowance) or (
        direct['total']<=allowance and naive<=allowance and max(norms.values())<max(before.values()))
    record.update(accepted=bool(accepted),reason='accepted' if accepted else 'same_objective_acceptance',
                  stable_objective_change=direct,accumulated_cost_difference=naive,
                  cancellation_difference=naive-direct['total'],numerical_allowance=allowance,
                  gradient_after=norms,
                  component_accumulated_differences={key:candidate[key]-o[key] for key in direct['components']})
    return (qx,qp,record) if accepted else (x,p,record)


def update(objective,x,p):
    outcome=objective.evaluate(x,p,hessian=True)
    outcome.update(objective.observed_hessian(x,p))
    event={'block':'joint','observed_curvature':True,'local_constrained_polish':True,
           'cost_before':outcome['cost'],'gradient_before':objective.stationarity(x,p,outcome),'trials':[]}
    try:dx,dp,ledger=local_direction(objective,x,p,outcome)
    except (ValueError,np.linalg.LinAlgError) as error:
        event.update(accepted=False,reason=str(error));return x,p,event
    event['subproblem']=ledger
    for damping in range(12):
        qx,qp,record=trial(objective,x,p,outcome,dx,dp,damping);event['trials'].append(record)
        if record['accepted']:
            event.update(accepted=True,outcome=record);return qx,qp,event
    event.update(accepted=False,reason='local same-objective line search exhausted');return x,p,event
