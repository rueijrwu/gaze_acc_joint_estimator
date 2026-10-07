"""Anchor-free held-fixation inference with rebuilt training-policy guards."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent)); sys.path.insert(0,str(HERE))
from calibrate_profiled import load_data,basis
from calibrate_continuation import atomic_json, read_checkpoint
from experiment import new_output, write_csv
from joint_inverse import infer_joint
from estimate_profiled import physical_optimality
from matched_experiment import observations,training_sets,common_policy,construct_problem,experiment_manifest


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def conditional_geometry(state,coef,p,covariance,hypothesis,variance,at_knot):
    theta,A=state.T
    _,dt,da=basis(theta,A**p,p)
    normalized=coef[:14]
    theta_J=np.einsum('nik,k->ni',dt,normalized)
    A_J=np.einsum('nik,k->ni',da,normalized)
    dA=np.zeros(len(A));positive=A>0
    dA[positive]=p*A[positive]**(p-1)
    A_J*=dA[:,None]
    covariance=np.asarray(covariance).copy()
    if hypothesis=='free': covariance=covariance[:2,:2]
    else:
        theta_J=np.column_stack((theta_J,np.full(len(A),coef[15]/15)))
        A_J=np.column_stack((A_J,coef[16]*dA))
        covariance[2,2]+=variance
    J=np.stack((theta_J,A_J),axis=2)
    L=np.linalg.cholesky(np.linalg.inv(covariance)).T
    reference=np.asarray([1.,.25])
    whitened=np.einsum('ab,nbk->nak',L,J)*reference
    singular=np.linalg.svd(whitened,compute_uv=False)
    rank=np.sum(singular>np.maximum(singular[:,:1],1e-300)*1e-12,axis=1)
    valid=(rank==2)&(A>1e-5)&(~np.asarray(at_knot,dtype=bool))&(np.abs(theta)<20-1e-5)&(A<6-1e-5)
    noise_map=np.linalg.pinv(whitened,rcond=1e-12)*reference[None,:,None]
    local_cov=np.einsum('nik,njk->nij',noise_map,noise_map)
    sd=np.sqrt(np.maximum(np.diagonal(local_cov,axis1=1,axis2=2),0))
    correlation=local_cov[:,0,1]/np.maximum(sd[:,0]*sd[:,1],1e-300)
    condition=np.divide(singular[:,0],singular[:,1],out=np.full(len(A),np.inf),where=singular[:,1]>0)
    sd[~valid]=np.nan;correlation[~valid]=np.nan
    return dict(local_theta_noise_sd_deg=sd[:,0],local_A_noise_sd_D=sd[:,1],
        local_noise_correlation=correlation,local_noise_valid=valid,
        conditional_jacobian_rank=rank,reference_scaled_jacobian_condition=condition)


def predict(args):
    source=args.model_dir
    hashes={name:sha(source/name) for name in ('model.json','checkpoint.npz')}
    model=json.loads((source/'model.json').read_text())
    if model.get('schema')!='full_information_matched_model_v1': raise ValueError('Unsupported matched model schema')
    if not model['converged'] and not args.allow_incomplete: raise ValueError('Model did not converge; explicit --allow-incomplete required')
    stored=model['manifest']
    y,frames,groups,meta,records,provenance=load_data(HERE.parent,args.interval_path,None)
    raw,y3,ordering=observations(meta,records,y,groups)
    three,four,held=training_sets(stored['heldout_fixations'])
    policy=common_policy(y3,raw,frames,groups,meta,three)
    selected=three if stored['training_demand_count']==3 else four
    hypothesis=stored['scale_hypothesis']; sd=stored['assumed_log_scale_sd'] or 0.
    problem,initial,global_groups=construct_problem(y3,frames,groups,meta,selected,policy,hypothesis,sd,model.get('coefficient_solver','irls'))
    expected=experiment_manifest(problem,provenance,global_groups,policy,selected,held,ordering,stored['training_demand_count'],hypothesis,sd,stored['kappa'])
    if json.dumps(stored,sort_keys=True)!=json.dumps(expected,sort_keys=True):
        raise ValueError('Model identity differs from rebuilt exact training policy/source identity')
    x,checkpoint_coef,_,_=read_checkpoint(source,expected,problem,global_groups)
    coef=np.asarray(model['coefficients'])
    if not np.allclose(coef,checkpoint_coef,rtol=1e-7,atol=1e-6): raise ValueError('Model/checkpoint coefficients disagree')
    profile_coef,_,_=problem.robust_coefficients(x,stored['kappa'],tol=1e-10,maxiter=500)
    if not np.allclose(coef,profile_coef,rtol=1e-7,atol=1e-6): raise ValueError('Exported model does not reconstruct source profile')
    mask=np.isin(groups,held); indices=np.flatnonzero(mask)
    out=new_output(args.output_dir)
    rows=[]
    for start in range(0,len(indices),args.batch_size):
        batch=indices[start:start+args.batch_size]
        state,diag=infer_joint(y3[batch],coef,problem.p,np.asarray(policy['covariance']),
            scale_hypothesis=hypothesis,scale_variance=sd**2,maxiter=args.maxiter)
        if hypothesis=='free':
            optimality,at_knot=physical_optimality(state,y3[batch,:2],coef[:14],problem.p,
                np.linalg.inv(np.asarray(policy['covariance'])[:2,:2]))
            diag=dict(diag,physical_optimality=optimality,at_interior_knot=at_knot)
        geometry=conditional_geometry(state,coef,problem.p,np.asarray(policy['covariance']),hypothesis,sd**2,
            diag.get('at_interior_knot',np.zeros(len(batch),dtype=bool)))
        diag=dict(diag,**geometry)
        for local,index in enumerate(batch):
            theta,A=map(float,state[local]); j=int(groups[index])
            row=dict(fixation_index=j,frame_index=int(frames[index]),theta_deg=theta,accommodation_D=A,
                nominal_theta_deg=float(meta[j]['target_theta_deg']),nominal_A_D=float(meta[j]['demand_diopters_label']),
                training_support=False,failed=not(np.isfinite(theta) and np.isfinite(A)),
                theta_bound=bool(np.isclose(abs(theta),20,atol=1e-5)),A_bound=bool(A<1e-5 or A>6-1e-5),
                nominal_knot_extrapolation=bool(A<min(problem.demands) or A>max(problem.demands)))
            for key,value in diag.items():
                array=np.asarray(value)
                if array.shape==(len(batch),):
                    item=array[local]; row[key]=item.item() if hasattr(item,'item') else item
            row['ambiguous']=bool(row.get('equivalent_minima_count',1)>1)
            row['stationarity_unverified']=bool(row.get('physical_optimality',row.get('segment_projected_stationarity',np.inf))>1e-4 or row.get('physical_optimality_unverified',False))
            rows.append(row)
    write_csv(out/'heldout_predictions.csv',rows)
    if hashes!={name:sha(source/name) for name in hashes}: raise RuntimeError('Frozen source changed during inference')
    summary=[]
    for j in held:
        selected_rows=[r for r in rows if r['fixation_index']==j]
        valid=[r for r in selected_rows if not r['failed']]
        theta=np.asarray([r['theta_deg'] for r in valid]); A=np.asarray([r['accommodation_D'] for r in valid])
        nominal_theta=float(meta[j]['target_theta_deg']); nominal_A=float(meta[j]['demand_diopters_label'])
        summary.append(dict(fixation_index=j,training_support=False,frame_count=len(selected_rows),coverage=len(valid)/max(1,len(selected_rows)),
            theta_nominal_mean_offset_deg=float(np.mean(theta)-nominal_theta) if len(theta) else None,
            A_nominal_mean_offset_D=float(np.mean(A)-nominal_A) if len(A) else None,
            theta_within_fixation_sd_deg=float(np.std(theta)) if len(theta) else None,A_within_fixation_sd_D=float(np.std(A)) if len(A) else None,
            **{flag+'_count':sum(bool(r[flag]) for r in selected_rows) for flag in ('failed','ambiguous','stationarity_unverified','theta_bound','A_bound','nominal_knot_extrapolation')}))
    write_csv(out/'heldout_fixations.csv',summary)
    atomic_json(out/'prediction_manifest.json',dict(schema='full_information_heldout_predictions_v1',
        source_model_sha256=hashes['model.json'],source_checkpoint_sha256=hashes['checkpoint.npz'],
        source_model_dir=str(source),common_policy_sha256=expected['common_policy_sha256'],heldout_fixations=held,
        inverse_state_prior='none; no nominal anchors',scale_hypothesis=hypothesis,conditional_scale=True,
        source_converged=model['converged'],prediction_csv_sha256=sha(out/'heldout_predictions.csv'),fixations=summary,
        stationarity_threshold=1e-4,stationarity_policy='existing physical inverse optimality threshold; encoded one-sided check at singular zero accommodation',
        conditional_geometry_scope='fixed coefficient local propagation; 1deg/0.25D reference-scaled Jacobian; excludes knots, physical bounds and rank deficiency; not physiological uncertainty or confidence intervals',
        independent_accuracy_status='not_evaluable_without_independent_references'))
    print(json.dumps(dict(output_dir=str(out),frame_count=len(rows),fixations=summary)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--interval-path',type=Path,default=HERE.parent/'fixations'/'fixation_intervals.json')
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--maxiter',type=int,default=100)
    p.add_argument('--allow-incomplete',action='store_true')
    predict(p.parse_args())
