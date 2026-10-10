"""Saved-point comparison of legacy QPs and constraint-consistent polishing.

Evaluates proposed candidates; never writes a calibration checkpoint or changes
the input attempt. A fresh experiment directory contains all numerical ledgers.
"""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from scipy.optimize import nnls

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
STAGE=ROOT/'experiments/distortion_model/stage_04_center_and_dm0_calibration'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from repair_g7 import load_context,write,digest
from distortion_model.centroid_bound import BoundedJointDM0
from distortion_model.geometry import relative_coordinates
from distortion_model.polishing import local_direction,trial


def legacy_ledger(objective,x,p,o,lm,dx,dp):
    xp=objective.xp;host=objective.host
    active=objective.active(x,o['gx'],objective.state_lower,objective.state_upper)
    apply=objective.coupled_inverse(o['hx'],active,lm);ig=apply(o['gx']);z=apply(o['cross'])
    h=host(o['hg']-xp.einsum('nsp,nsq->pq',o['cross'],z));h=(h+h.T)/2
    h+=np.diag(lm*np.maximum(np.abs(np.diag(host(o['hg']))),1e-12))
    gradient=host(o['gp']-xp.einsum('nsp,ns->p',o['cross'],ig))
    step=host(dp);ph=host(p);scales=host(objective.global_scales)
    c,j=objective.constraints(p,True);linear=c+j@step;indices=np.flatnonzero(linear<=1e-7)
    rows=list(j[indices]);slacks=list(linear[indices])
    lo,hi=host(objective.lower),host(objective.upper)
    for i in range(19):
        if np.isfinite(lo[i]) and (ph[i]+step[i]-lo[i])/scales[i]<=1e-8:
            row=np.zeros(19);row[i]=1/scales[i];rows.append(row);slacks.append((ph[i]+step[i]-lo[i])/scales[i])
        if np.isfinite(hi[i]) and (hi[i]-ph[i]-step[i])/scales[i]<=1e-8:
            row=np.zeros(19);row[i]=-1/scales[i];rows.append(row);slacks.append((hi[i]-ph[i]-step[i])/scales[i])
    normals=np.asarray(rows).reshape(-1,19);qp_gradient=(h@step+gradient)*scales
    multipliers=nnls(normals.T*scales[:,None],qp_gradient,maxiter=10000)[0] if len(rows) else np.zeros(0)
    residual=qp_gradient-(normals.T*scales[:,None])@multipliers
    return {'LM':lm,'active_row_rank':int(np.linalg.matrix_rank(normals)) if len(rows) else 0,
            'subproblem_scaled_stationarity_inf':float(np.max(np.abs(residual))),
            'minimum_linearized_physical_slack':float(linear.min()),
            'multipliers':multipliers.tolist(),
            'subproblem_complementarity_inf':float(np.max(np.abs(multipliers*np.asarray(slacks)),initial=0)),
            'shared_step':step.tolist(),'shared_step_scaled':(step/scales).tolist(),
            'predicted_reduced_quadratic_change':float(gradient@step+.5*step@h@step)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',type=Path,default=STAGE/'results/g7_attempt_03')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();attempt=args.attempt.resolve();output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    paths=list((ROOT/'distortion_model').glob('*.py'))+list((STAGE/'scripts').glob('*.py'))
    paths += [ROOT/'docs/Theory.md',ROOT/'docs/STAGE_GATES.md',ROOT/'docs/audits/G7_REPAIR_RESULTS_AUDIT.md',ROOT/'requirements-stage7.txt']
    sources={}
    for path in paths:
        relative=path.relative_to(ROOT);target=output/'source_snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);sources[str(relative)]=digest(path)
    inputs_sha={str(path.relative_to(attempt)):digest(path) for path in attempt.rglob('*') if path.is_file()}
    write(output/'provenance.json',{'kind':'saved-point proposal diagnosis; no calibration campaign',
          'input_attempt':str(attempt.relative_to(ROOT)),'input_sha256':inputs_sha,'source_sha256':sources,
          'source_hash':sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
          'started_UTC':datetime.now(timezone.utc).isoformat()})
    _,inputs,spec,bound,summary,x,p=load_context(attempt)
    import cupy as cp
    pop=inputs['population'];e=inputs['exposure'];valid=inputs['valid']
    y=relative_coordinates(pop['p1'][valid],pop['p4'][valid])
    targets=np.array([[pop['target_theta_deg'][np.flatnonzero(pop['exposure']==k)[0]],
                       pop['demand_diopters'][np.flatnonzero(pop['exposure']==k)[0]]] for k in range(20)])
    objective=BoundedJointDM0(spec,y,e,inputs['covariance']['relative'],targets,xp=cp,physical_bound=bound)
    x,p=cp.asarray(x),cp.asarray(p);o=objective.evaluate(x,p,hessian=True)
    assert abs(o['cost']-summary['components']['cost'])<1e-8
    o.update(objective.observed_hessian(x,p))
    old=[]
    for lm in (0.,1e-5,1e-4,1e-3,1e-2,.1,1.):
        try:
            dx,dp=objective.direction(x,p,o,'joint',lm)
            ledger=legacy_ledger(objective,x,p,o,lm,dx,dp)
            _,_,candidate=trial(objective,x,p,o,dx,dp,0)
            old.append({'LM':lm,'status':'PROPOSED','ledger':ledger,'full_step':candidate})
        except (ValueError,np.linalg.LinAlgError) as error:old.append({'LM':lm,'status':'REJECTED','reason':str(error)})
    write(output/'legacy_proposals.json',old)
    dx,dp,ledger=local_direction(objective,x,p,o)
    candidates=[];accepted=None;qx,qp=x,p
    for damping in range(12):
        qx,qp,record=trial(objective,x,p,o,dx,dp,damping);candidates.append(record)
        if record['accepted']:accepted=record;break
    candidate_certificate=objective.certificate(qx,qp) if accepted else None
    if accepted:
        np.savez_compressed(output/'proposed_candidate.npz',states=objective.host(qx),scaled_globals=objective.host(qp))
    np.savez_compressed(output/'proposed_step.npz',state_step=objective.host(dx),shared_step=objective.host(dp))
    after={str(path.relative_to(attempt)):digest(path) for path in attempt.rglob('*') if path.is_file()}
    assert inputs_sha==after,'saved input was modified'
    write(output/'summary.json',{'stage':'S7/G7 saved-point constrained polishing diagnosis','status':'PROBE_PENDING_REVIEW',
          'input_attempt_unchanged':True,'calibration_checkpoint_written':False,
          'model_objective_bound_population_unchanged':True,'full_population':objective.n,
          'old_global_stationarity':summary['certificate']['global_projected_gradient_inf'],
          'legacy_proposal_statuses':old,'local_subproblem':ledger,'trials':candidates,
          'accepted_candidate':accepted,'candidate_certificate':candidate_certificate,
          'elapsed_seconds':time.perf_counter()-started})
    print(json.dumps({'local_subproblem':ledger,'accepted_candidate':accepted,'candidate_certificate':candidate_certificate},indent=2),flush=True)


if __name__=='__main__':main()
