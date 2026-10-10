"""Matched Newton continuation with explicit state-bound working-set updates.

Preserve the original campaign and objective. Existing local polishing keeps
fixed working sets and rejects new blocking inequalities. This wrapper solves
the same quadratic with explicit updates and independent accuracy guards.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import null_space
from scipy.optimize import minimize
sys.path.insert(0,str(Path(__file__).parent))
from run import ROOT,context,write,digest,load_npz
from distortion_model.centroid_bound import BoundedJointDM0
from distortion_model.geometry import relative_coordinates
from distortion_model.polishing import local_direction,trial


def corrected_trial(objective,x,p,o,dx,dp,damping):
    """Second-order feasibility restoration followed by original J acceptance."""
    step=.5**damping;host=objective.host;ph=host(p);raw=ph+step*host(dp)
    bound=objective.physical_bound;slack=bound.values(raw);record=None;proposal=dp
    if slack.min()<2e-12:
        scale=host(objective.global_scales);lo,hi=host(objective.lower),host(objective.upper)
        result=minimize(lambda v:(.5*v@v,v),np.zeros(19),jac=True,method='SLSQP',
            bounds=list(zip((lo-raw)/scale,(hi-raw)/scale)),
            constraints=[{'type':'ineq','fun':lambda v:bound.values(raw+scale*v)-2e-12,
                'jac':lambda v:bound.jacobian(raw+scale*v)*scale[None,:]}],
            options={'ftol':1e-15,'maxiter':50})
        correction=scale*result.x;restored=raw+correction;after=bound.values(restored)
        record={'method':'nearest declared-unit global correction; original nonlinear inequalities',
            'maximum_iterations':50,'solver_success':bool(result.success),'solver_message':str(result.message),
            'iterations':int(result.nit),'minimum_raw_slack':float(slack.min()),'minimum_restored_slack':float(after.min()),
            'correction_scaled_inf':float(np.max(np.abs(result.x))),'correction_native':correction.tolist()}
        if not np.isfinite(restored).all() or after.min()< -1e-10:
            return x,p,{'accepted':False,'reason':'nonlinear feasibility restoration failed','damping':damping,'feasibility_correction':record}
        proposal=dp+objective.xp.asarray(correction/step)
    qx,qp,event=trial(objective,x,p,o,dx,proposal,damping)
    event['feasibility_correction']=record
    return qx,qp,event


def global_working_direction(objective,x,p,o):
    """Feasible primal active-set solve of the same observed SQP quadratic.

    New inequalities enter the quadratic working set with zero contribution
    to the *current* Lagrangian Hessian. Final nonlinear certification is
    still performed by the original objective.
    """
    xp=objective.xp;host=objective.host;ph=host(p);kkt=objective.kkt(p,o['gp'])
    if kkt['nonsmooth_active_rows']:
        raise ValueError('continuation requires smooth current global rows')
    apply=objective.coupled_inverse(o['hx'],objective.active(x,o['gx'],objective.state_lower,objective.state_upper))
    ig=apply(o['gx']);z=apply(o['cross'])
    schur=host(o['hg']-xp.einsum('nsp,nsq->pq',o['cross'],z));schur=(schur+schur.T)/2
    gradient=host(o['gp']-xp.einsum('nsp,ns->p',o['cross'],ig))
    c,j=objective.constraints(p,True);weights=np.zeros(len(c))
    weights[kkt['physical_indices']]=kkt['multipliers'][:kkt['physical_count']]
    correction,branch=objective.physical_bound.weighted_hessian(ph,weights)
    if branch['nonsmooth_rows']:raise ValueError('ambiguous current constraint Hessian')
    lagrangian=schur-correction;scale=host(objective.global_scales)
    mapping=scale/np.sqrt(np.maximum(np.abs(np.diag(lagrangian)*scale**2),1e-12))
    h=lagrangian*mapping[:,None]*mapping[None,:];g=gradient*mapping
    lo,hi=host(objective.lower),host(objective.upper)
    lower=np.flatnonzero(np.isfinite(lo));upper=np.flatnonzero(np.isfinite(hi))
    normals=np.vstack((j,np.eye(19)[lower]/scale[lower,None],-np.eye(19)[upper]/scale[upper,None]))
    slack=np.r_[c,(ph[lower]-lo[lower])/scale[lower],(hi[upper]-ph[upper])/scale[upper]]
    lengths=np.linalg.norm(normals*mapping[None,:],axis=1)
    if np.any((lengths==0)&(slack< -1e-10)):raise ValueError('infeasible constant inequality')
    usable=lengths>0;original_indices=np.flatnonzero(usable)
    a=normals[usable]*mapping[None,:]/lengths[usable,None];s=slack[usable]/lengths[usable]
    index_map={int(v):i for i,v in enumerate(original_indices)}
    active=list(kkt['physical_indices'])
    active += [len(c)+i for i,q in enumerate(lower) if (ph[q]-lo[q])/scale[q]<=1e-8]
    active += [len(c)+len(lower)+i for i,q in enumerate(upper) if (hi[q]-ph[q])/scale[q]<=1e-8]
    # Weak current rows remain inequalities. Do not freeze them as equalities.
    working=[index_map[i] for i,mu in zip(active,kkt['multipliers']) if mu>1e-8]
    v=np.zeros(19);events=[]
    for iteration in range(96):
        b=a[working];target=-s[working]
        if np.linalg.matrix_rank(b,tol=1e-12)!=len(working):raise ValueError('dependent global working rows')
        particular=np.linalg.lstsq(b,target,rcond=1e-12)[0] if working else np.zeros(19)
        basis=null_space(b,rcond=1e-12) if working else np.eye(19)
        tangent=basis.T@h@basis;tangent=(tangent+tangent.T)/2
        diagonal=np.sqrt(np.maximum(np.abs(np.diag(tangent)),1e-12))
        normalized=tangent/(diagonal[:,None]*diagonal[None,:]);eigen=np.linalg.eigvalsh(normalized)
        if len(eigen) and eigen.min()<=max(abs(eigen))*1e-10:
            raise ValueError('nonpositive global working-set tangent curvature')
        rhs=-basis.T@(g+h@particular)
        q=particular+basis@(np.linalg.solve(normalized,rhs/diagonal)/diagonal) if len(eigen) else particular
        multipliers=np.linalg.lstsq(b.T,h@q+g,rcond=1e-12)[0] if working else np.zeros(0)
        actual=slack+normals@(mapping*q)
        violated=(actual[usable]< -1e-10)
        if violated.any():
            delta=a@(q-v);current=s+a@v
            possible=violated&(delta<0)
            if working:possible[working]=False
            if not possible.any():raise ValueError('global working-set has no feasible blocker')
            ratio=np.full(len(s),np.inf);ratio[possible]=np.maximum(current[possible],0)/(-delta[possible])
            blocker=int(np.argmin(ratio));fraction=float(np.clip(ratio[blocker],0,1))
            v=v+fraction*(q-v);working.append(blocker)
            events.append({'iteration':iteration,'action':'add','original_constraint':int(original_indices[blocker]),'fraction':fraction})
            continue
        if len(multipliers) and multipliers.min()< -1e-10:
            release=int(np.argmin(multipliers));events.append({'iteration':iteration,'action':'release','original_constraint':int(original_indices[working[release]])})
            v=q;working.pop(release);continue
        dp=mapping*q;mu=multipliers/lengths[usable][working]
        rows=normals[usable][working];residual=(lagrangian@dp+gradient-rows.T@mu)*scale
        bordered=np.block([[h,-b.T],[b,np.zeros((len(working),len(working)))]] )
        alternative=np.linalg.solve(bordered,np.r_[-g,target])[:19]*mapping
        accuracy=float(np.max(np.abs(residual)));difference=float(np.max(np.abs((dp-alternative)/scale)))
        feasibility=float(np.max(np.abs(slack[usable][working]+rows@dp),initial=0))
        if accuracy>1e-8 or difference>1e-10 or feasibility>1e-10 or actual.min()< -1e-10:
            raise ValueError('global working-set independent accuracy check failed')
        dx=-ig-xp.einsum('nsp,p->ns',z,xp.asarray(dp))
        return dx,xp.asarray(dp),{'method':'observed SQP with global primal working-set updates; independent bordered solve',
            'global_working_set':events,'global_final_constraint_indices':original_indices[working].tolist(),
            'subproblem_scaled_stationarity_inf':accuracy,'independent_route_scaled_step_difference_inf':difference,
            'subproblem_active_feasibility_inf':feasibility,'minimum_linearized_constraint_slack':float(actual.min()),
            'minimum_multiplier':float(mu.min()) if len(mu) else None,'tangent_normalized_eigenvalues':eigen.tolist(),
            'weak_current_rows_kept_as_inequalities':int(np.sum(kkt['multipliers']<=1e-8))}
    raise ValueError('global working-set iteration budget exhausted')


def state_hessian_apply(objective,o,dx):
    xp=objective.xp
    grouped=objective.group_sum(dx)/(objective.k*objective.counts[:,None]**2*objective.mean_scales**2)
    return xp.einsum('nij,nj->ni',o['hx'],dx)+grouped[objective.exposure]


class WorkingSet:
    """Supply shifted quadratic gradients and the frozen SQP multiplier set."""
    def __init__(self,objective,active,kkt):self.objective=objective;self.mask=active;self.saved_kkt=kkt
    def __getattr__(self,name):return getattr(self.objective,name)
    def active(self,*args):return self.mask
    def kkt(self,*args):return self.saved_kkt
    @property
    def state_lower(self):return self.xp.full(2,-self.xp.inf)
    @property
    def state_upper(self):return self.xp.full(2,self.xp.inf)


def direction(objective,x,p,o):
    xp=objective.xp;active=objective.active(x,o['gx'],objective.state_lower,objective.state_upper)
    lower=active&(x<=objective.state_lower+1e-9);upper=active&(~lower)
    kkt=objective.kkt(p,o['gp']);records=[]
    for iteration in range(32):
        mask=lower|upper
        fixed=xp.where(lower,objective.state_lower-x,xp.where(upper,objective.state_upper-x,0.))
        adjusted=dict(o);adjusted['gx']=o['gx']+state_hessian_apply(objective,o,fixed)
        adjusted['gp']=o['gp']+xp.einsum('nsp,ns->p',o['cross'],fixed)
        proxy=WorkingSet(objective,mask,kkt)
        try:free_step,dp,ledger=local_direction(proxy,x+fixed,p,adjusted)
        except ValueError as error:
            if not any(message in str(error) for message in ['new blocking inequality','negative working-set multiplier','weak active row']):raise
            free_step,dp,ledger=global_working_direction(proxy,x+fixed,p,adjusted)
        dx=fixed+free_step;qx=x+dx
        below=qx<objective.state_lower-1e-10;above=qx>objective.state_upper+1e-10
        gradient=o['gx']+state_hessian_apply(objective,o,dx)+xp.einsum('nsp,p->ns',o['cross'],dp)
        release_lower=lower&(gradient*objective.state_scales < -1e-8)
        release_upper=upper&(gradient*objective.state_scales > 1e-8)
        records.append({'iteration':iteration,'working_bounds':int(mask.sum()),
            'new_lower':int(below.sum()),'new_upper':int(above.sum()),
            'release_lower':int(release_lower.sum()),'release_upper':int(release_upper.sum())})
        if not bool(xp.any(below|above|release_lower|release_upper)):
            residual=xp.where(mask,0.,gradient*objective.state_scales)
            norm=float(xp.max(xp.abs(residual)))
            if norm>1e-8:raise ValueError('state working-set quadratic stationarity failed')
            ledger.update(state_working_set=records,state_quadratic_scaled_stationarity_inf=norm,
                final_state_bound_count=int(mask.sum()),original_state_bound_count=int(active.sum()),
                state_step_max_deg_D=objective.host(xp.max(xp.abs(dx),axis=0)).tolist())
            return dx,dp,ledger
        lower=(lower&~release_lower)|below;upper=(upper&~release_upper)|above
    raise ValueError('state working-set iteration budget exhausted')


def main():
    import cupy as cp
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--candidate',choices=['control','adjacent'],required=True);parser.add_argument('--start',choices=['common','perturbed'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--steps',type=int,default=1)
    args=parser.parse_args();campaign=args.campaign.resolve();output=args.output.resolve()
    if not 1<=args.steps<=12:raise ValueError('declared maximum twelve continuation steps')
    candidate,pop,cov,valid,policy,spec,initial,bound=context(campaign,args.candidate)
    source=candidate/args.start/'solution.npz';saved=load_npz(source);x,p=cp.asarray(saved['states']),cp.asarray(saved['scaled_globals'])
    y=relative_coordinates(pop['p1'][valid],pop['p4'][valid]);e=pop['exposure'][valid]
    targets=np.array([[pop['target_theta_deg'][pop['exposure']==k][0],pop['demand_diopters'][pop['exposure']==k][0]] for k in range(20)])
    objective=BoundedJointDM0(spec,y,e,cov['relative'],targets,xp=cp,physical_bound=bound)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'config.json',{'kind':'state-bound working-set Newton continuation','candidate':args.candidate,'start':args.start,
        'campaign':str(campaign.relative_to(ROOT)),'maximum_steps':args.steps,'state_working_set_max_iterations':32,
        'global_working_set_max_iterations':96,'nonlinear_feasibility_correction_max_iterations':50,
        'stationarity_threshold':1e-6,'same_objective':True,'calibration_checkpoint_written':False,
        'source_solution_sha256':digest(source),'repair_script_sha256':digest(Path(__file__))})
    before=digest(source);history=[]
    for step in range(args.steps):
        o=objective.evaluate(x,p,hessian=True);norms=objective.stationarity(x,p,o)
        if max(norms.values())<=1e-6:break
        o.update(objective.observed_hessian(x,p));event={'step':step,'before':norms,'cost_before':o['cost'],'trials':[]}
        try:dx,dp,ledger=direction(objective,x,p,o)
        except (ValueError,np.linalg.LinAlgError) as error:
            event.update(accepted=False,reason=str(error));history.append(event);break
        event['subproblem']=ledger
        for damping in range(12):
            qx,qp,record=corrected_trial(objective,x,p,o,dx,dp,damping);event['trials'].append(record)
            if record['accepted']:x,p=qx,qp;event['accepted']=True;break
        else:event.update(accepted=False,reason='same-objective line search exhausted')
        history.append(event);write(output/'history.json',history)
        print(args.candidate,args.start,step,event.get('accepted'),event['trials'][-1] if event['trials'] else event.get('reason'),flush=True)
        if not event['accepted']:break
    final=objective.evaluate(x,p);certificate=objective.certificate(x,p)
    if digest(source)!=before:raise ValueError('original optimization point modified')
    np.savez_compressed(output/'solution.npz',states=objective.host(x),scaled_globals=objective.host(p))
    write(output/'history.json',history)
    write(output/'summary.json',{'components':{k:float(final[k]) for k in ['cost','point','theta_anchor','A_anchor','regularization','temporal']},
        'certificate':certificate,'original_source_unchanged':True,'new_optical_mechanism':False,'calibration_checkpoint_written':False})
    print(json.dumps(certificate,indent=2),flush=True)


if __name__=='__main__':main()
