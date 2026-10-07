"""Training-only matched demand/channel experiments; conditional scale hypotheses."""
import os
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize_scalar

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
sys.path.insert(0,str(EXP))
sys.path.insert(0,str(HERE))
from calibrate_profiled import KNOTS, basis, design, load_data, noise_covariance
from calibrate_continuation import atomic_json, array_hash, continue_fit, make_manifest, read_checkpoint, physical_diagnostics
from experiment import measurement_rows, new_output, frozen, write_csv
from joint_problem import make_scale_problem
from scale_diagnostics import independent_jacobian


def training_sets(heldout):
    held=sorted(set(int(j) for j in heldout))
    if not held or any(j<0 or j>=20 for j in held):
        raise ValueError('Declare at least one complete held-out fixation in 0..19')
    four=[j for j in range(20) if j not in held]
    three=[j for j in four if j not in range(10,15)]
    for selected in (three,four):
        for start in (0,5,15):
            if sum(j in selected for j in range(start,start+5))<2:
                raise ValueError('Each common training demand needs at least two fixations')
    return three,four,held


def observations(meta,records,y,groups):
    rows,audit=measurement_rows(meta,records)
    raw=np.asarray([[r['m'],r['S1'],r['S4']] for r in rows],dtype=float)
    observed=np.column_stack((raw[:,0]/raw[:,1],raw[:,2]/raw[:,1],np.log(raw[:,1])))
    row_groups=np.asarray([r['fixation_index'] for r in rows],dtype=int)
    if observed.shape!=(len(y),3) or not np.array_equal(row_groups,groups):
        raise ValueError('Relative observations do not preserve exact loader support/order')
    if not np.allclose(observed[:,:2],y,rtol=0,atol=2e-14):
        raise ValueError('Relative observations and normalized loader disagree')
    return raw,observed,audit


def common_policy(y3,raw,frames,groups,meta,selected):
    mask=np.isin(groups,selected)
    means=np.asarray([y3[groups==j].mean(axis=0) for j in selected])
    theta=np.asarray([float(meta[j]['target_theta_deg']) for j in selected])
    demand=np.asarray([float(meta[j]['demand_diopters_label']) for j in selected])
    def exponent_loss(p):
        X=design(theta,demand**p)
        coef=np.linalg.lstsq(X,means[:,1],rcond=None)[0]
        return float(np.sum((X@coef-means[:,1])**2))
    p=float(minimize_scalar(exponent_loss,bounds=(.15,1.),method='bounded').x)
    E=frozen.coefficient_map('holdout3')
    H=basis(theta,demand**p,p)[0]
    reduced=np.linalg.lstsq(H.reshape(-1,14)@E,means[:,:2].reshape(-1),rcond=None)[0]
    normalized=E@reduced
    log_X=np.column_stack((np.ones(len(theta)),theta/15,demand**p))
    log_coef=np.linalg.lstsq(log_X,means[:,2],rcond=None)[0]
    raw_cov=noise_covariance(raw[mask],frames[mask],groups[mask])
    anchor=raw[mask].mean(axis=0)
    T=independent_jacobian(anchor)
    covariance=T@raw_cov@T.T
    np.linalg.cholesky(covariance)
    ranges=np.ptp(y3[mask],axis=0)
    if np.any(ranges<=0):
        raise ValueError('Training-only function-prior ranges must be positive')
    prior_W=np.diag(1/ranges**2)
    return dict(schema='full_information_common_policy_v1',training_fixations=selected,p=p,
        coefficient_map=E.tolist(),normalized_coefficients=normalized.tolist(),
        log_shape_coefficients=log_coef.tolist(),covariance=covariance.tolist(),
        raw_covariance=raw_cov.tolist(),raw_anchor=anchor.tolist(),delta_method_jacobian=T.tolist(),
        prior_precision=prior_W.tolist(),prior_ranges=ranges.tolist(),
        training_observation_hash=array_hash(y3[mask]),training_raw_hash=array_hash(raw[mask]),
        training_frame_hash=array_hash(frames[mask]),training_group_hash=array_hash(groups[mask]),
        initialization_policy='equal-fixation ordinary means; constrained normalized least squares; low-order log-shape least squares; fresh per-frame displacement inverse at nominal demand',
        noise_policy='robust adjacent-frame raw differences; includes motion and detector variation; transformed Gaussian delta-method approximation',
        prior_policy='common training-only function center; diagonal inverse squared common frame range; existing strength .1; no previous-mean prior; no curvature',
        scale_interpretation='fixed/finite scale are conditional hypotheses; free scale is exact normalized control',
        cost_comparability='No cross-scale likelihood ranking: no covariance logdet and different prior dimensions')


def construct_problem(y3,frames,groups,meta,selected,policy,hypothesis,log_sd,coefficient_solver='irls'):
    if hypothesis=='free' and coefficient_solver!='irls':
        raise ValueError('Exact free-scale control always uses the legacy coefficient IRLS')
    mask=np.isin(groups,selected)
    global_groups=groups[mask]
    local=np.searchsorted(selected,global_groups)
    targets=np.asarray([float(meta[j]['target_theta_deg']) for j in selected])
    demands=np.asarray([float(meta[j]['demand_diopters_label']) for j in selected])
    p=float(policy['p'])
    E=np.asarray(policy['coefficient_map'])
    coef=np.r_[policy['normalized_coefficients'],policy['log_shape_coefficients']]
    E3=np.zeros((17,E.shape[1]+3)); E3[:14,:E.shape[1]]=E; E3[14:,E.shape[1]:]=np.eye(3)
    covariance=np.asarray(policy['covariance'])
    problem=make_scale_problem(hypothesis,y3[mask],frames[mask],local,targets,demands,p,
        coef,covariance,np.asarray(policy['prior_precision']),coefficient_map=E3,
        scale_variance=float(log_sd)**2,coefficient_solver=coefficient_solver)
    A=demands[local]
    H0=basis(np.zeros(len(A)),A**p,p)[0]
    H1=basis(np.ones(len(A)),A**p,p)[0]
    normalized=np.asarray(policy['normalized_coefficients'])
    b=H0[:,0]@normalized; gain=(H1[:,0]-H0[:,0])@normalized
    if np.any(np.abs(gain)<1e-8):
        raise ValueError('Insufficient common-prior displacement gain')
    theta=np.clip((y3[mask,0]-b)/gain,-20,20)
    return problem,problem.encode(theta,A),global_groups


def experiment_manifest(problem,provenance,global_groups,policy,selected,held,ordering,demands,hypothesis,log_sd,kappa):
    three=policy['training_fixations']
    provenance=dict(provenance,training_fixations=selected,heldout_fixations=held,
        heldout_fixation=held[0] if len(held)==1 else None,
        experiment='full_information_matched',common_policy_training_fixations=three,
        common_policy_hash=hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        all_observation_ordering=ordering)
    identity=make_manifest(problem,provenance,global_groups,kappa)
    identity.update(schema='full_information_matched_v1',experiment='full_information_matched',
        common_policy=policy,common_policy_sha256=provenance['common_policy_hash'],
        scale_hypothesis=hypothesis,assumed_log_scale_sd=log_sd if hypothesis=='uncertain' else None,
        training_fixations=selected,heldout_fixations=held,training_demand_count=demands,
        observation_basis=['d','rho'] if hypothesis=='free' else ['d','rho','logS1'],
        conditional_scale=True,extra_log_shape_coefficients=0 if hypothesis=='free' else 3,
        independent_accuracy_status='not_evaluable_without_independent_references')
    return identity


def train(args):
    y,frames,groups,meta,records,provenance=load_data(EXP,args.interval_path,None)
    raw,y3,ordering=observations(meta,records,y,groups)
    three,four,held=training_sets(args.heldout_fixation)
    policy=common_policy(y3,raw,frames,groups,meta,three)
    selected=three if args.demands==3 else four
    hypothesis=args.scale_hypothesis
    if hypothesis=='uncertain' and (not np.isfinite(args.log_scale_sd) or args.log_scale_sd<=0):
        raise ValueError('Finite scale hypothesis requires an explicit positive log-scale SD')
    coefficient_solver=getattr(args,'coefficient_solver','irls')
    problem,x,global_groups=construct_problem(y3,frames,groups,meta,selected,policy,hypothesis,args.log_scale_sd,coefficient_solver)
    out=new_output(args.output_dir)
    identity=experiment_manifest(problem,provenance,global_groups,policy,selected,held,ordering,args.demands,hypothesis,args.log_scale_sd,args.kappa)
    resume_initial=None;prior_phases=[];resume_source=None
    if args.resume_dir is not None:
        x,verified_coef,verified_weights,resume_info=read_checkpoint(args.resume_dir,identity,problem,global_groups)
        previous_model=json.loads((args.resume_dir/'model.json').read_text())
        if previous_model['manifest']!=identity: raise ValueError('Resume model differs from exact rebuilt objective identity')
        with np.load(args.resume_dir/'checkpoint.npz',allow_pickle=False) as saved:
            saved_coef=saved['coefficients'].copy();saved_weights=saved['weights'].copy()
        if not np.allclose(saved_coef,verified_coef,rtol=1e-7,atol=1e-6) or not np.allclose(saved_weights,verified_weights,rtol=1e-7,atol=1e-6):
            raise ValueError('Resume checkpoint coefficients/weights differ from reconstruction')
        if not np.allclose(saved_coef,previous_model['coefficients'],rtol=1e-7,atol=1e-6):
            raise ValueError('Resume model and checkpoint coefficients differ')
        resume_initial=dict(coefficients=saved_coef,weights=saved_weights,objective=previous_model['objective'])
        resume_source=dict(directory=str(args.resume_dir.resolve()),model_sha256=file_sha(args.resume_dir/'model.json'),
            checkpoint_sha256=file_sha(args.resume_dir/'checkpoint.npz'))
        prior_phases=previous_model.get('continuation_phases',[])
        if not prior_phases:
            prior_phases=[dict(directory=str(args.resume_dir.resolve()),status=json.loads((args.resume_dir/'status.json').read_text()),
                coefficient_solver=previous_model.get('coefficient_solver','irls'),
                checkpoint_sha256=file_sha(args.resume_dir/'checkpoint.npz'),history_sha256=file_sha(args.resume_dir/'history.json'),
                lsmr_history_sha256=file_sha(args.resume_dir/'lsmr_history.json'))]
    atomic_json(out/'common_policy.json',policy)
    x,coef,history,converged,status=continue_fit(problem,x,out,identity,kappa=args.kappa,
        max_nfev=args.max_nfev,wall_seconds=args.wall_seconds,lsmr_maxiter=args.lsmr_maxiter,
        lsmr_atol=args.lsmr_atol,lsmr_btol=args.lsmr_btol,
        robust_outer=args.robust_outer,robust_max_nfev=args.robust_max_nfev,resume_initial=resume_initial)
    value,parts=problem.true_objective(x,coef,args.kappa)
    rebuilt_coef,weights,certificate=problem.robust_coefficients(x,args.kappa,maxiter=500,tol=1e-10)
    if not certificate['converged'] or not np.allclose(rebuilt_coef,coef,rtol=1e-7,atol=1e-6):
        raise RuntimeError('Final coefficient/profile certificate failed')
    physical=physical_diagnostics(problem,x,coef,weights)
    theta,_,A,_=problem.decode(x)
    phases=prior_phases+[dict(directory=str(out.resolve()),status=status,coefficient_solver=coefficient_solver,checkpoint_sha256=file_sha(out/'checkpoint.npz'),
        history_sha256=file_sha(out/'history.json'),lsmr_history_sha256=file_sha(out/'lsmr_history.json'))]
    totals={key:sum(float(phase['status'].get(key,0)) for phase in phases) for key in ('elapsed_seconds','nfev','accepted_iterations')}
    model=dict(schema='full_information_matched_model_v1',manifest=identity,p=problem.p,
        continuation_phases=phases,cumulative_continuation_totals=totals,resume_source=resume_source,
        resume_initial_objective_verified=resume_initial is not None,
        coefficient_solver=coefficient_solver,coefficient_profile_certificate=certificate,
        final_coefficient_profile_max_difference=float(np.max(np.abs(rebuilt_coef-coef))),
        coefficients=coef.tolist(),objective=value,objective_parts=parts,converged=bool(converged),status=status,
        coefficient_count=len(coef),free_coefficient_count=problem.free_coefficient_count,
        covariance=np.linalg.inv(problem.W).tolist(),scale_hypothesis=hypothesis,
        physical_diagnostics=physical,
        finite_physical_gradient_summary_complete=not bool(getattr(problem,'physical_derivative_undefined_count',physical.get('zero_A_physical_derivative_undefined_count',0))),
        physical_gradient_scope='At zero accommodation with p<1, singular physical derivatives are undefined; encoded one-sided bound checks determine feasible descent.',
        assumed_log_scale_sd=args.log_scale_sd if hypothesis=='uncertain' else None)
    atomic_json(out/'model.json',model)
    write_csv(out/'training_states.csv',[dict(fixation_index=int(j),frame_index=int(f),theta_deg=float(t),accommodation_D=float(a),training_support=True)
        for j,f,t,a in zip(global_groups,problem.frames,theta,A)])
    atomic_json(out/'result.json',dict(converged=bool(converged),status=status,objective=value,parts=parts,
        continuation_phases=phases,cumulative_continuation_totals=totals,
        validation_status='not_evaluable_without_independent_references',training_fixations=selected,heldout_fixations=held))
    print(json.dumps(dict(output_dir=str(out),converged=bool(converged),status=status,objective=value)))


def file_sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--interval-path',type=Path,default=EXP/'fixations'/'fixation_intervals.json')
    p.add_argument('--demands',type=int,choices=(3,4),required=True)
    p.add_argument('--scale-hypothesis',choices=('free','known','uncertain'),required=True)
    p.add_argument('--coefficient-solver',choices=('irls','newton'),default='irls',help='Numerical solver provenance only; free control requires irls')
    p.add_argument('--log-scale-sd',type=float,default=0.)
    p.add_argument('--heldout-fixation',type=int,action='append',default=None)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--resume-dir',type=Path)
    p.add_argument('--kappa',type=float,default=2.)
    p.add_argument('--max-nfev',type=int,default=600)
    p.add_argument('--wall-seconds',type=float,default=900.)
    p.add_argument('--lsmr-maxiter',type=int,default=300)
    p.add_argument('--lsmr-atol',type=float,default=1e-7)
    p.add_argument('--lsmr-btol',type=float,default=1e-7)
    p.add_argument('--robust-outer',type=int,default=24)
    p.add_argument('--robust-max-nfev',type=int,default=25)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.heldout_fixation is None: args.heldout_fixation=[17]
    train(args)
