"""Staged diagnosis of optical information and conditional calibration assumptions.

Existing calibration artifacts are immutable inputs. Nominal target agreement,
conditional noise propagation and model disagreement are not state accuracy.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OMP_THREAD_LIMIT', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP))
from calibrate_profiled import ProfiledProblem, KNOTS, basis, load_data
from calibrate_continuation import (atomic_json, array_hash, envelope_gradients,make_manifest,
                                    continue_fit,read_checkpoint,physical_diagnostics)
sys.path.insert(0,str(HERE))
from profile_problem import SensitivityProblem

_spec = importlib.util.spec_from_file_location('frozen_relative_protocol', EXP/'relative_calibration/experiment.py')
frozen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(frozen)

SCHEMA = 'full_information_diagnostic_v1'
REFERENCE_UNITS = np.array([1., .25])
DEFAULT_MODELS = dict(H0=EXP/'reduced_calibration/training/holdout3/robust',
    FC0=EXP/'relative_calibration/training/FC0/robust', FF0=EXP/'full_calibration/matched/robust',
    Hq=EXP/'relative_calibration/training/Hq/robust', HqP=EXP/'relative_calibration/training/HqP/robust',
    FCq=EXP/'relative_calibration/training/FCq/robust', FFq=EXP/'relative_calibration/training/FFq_continued/robust')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows: raise ValueError('Cannot write an empty result table')
    with Path(path).open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def finite_quantiles(values):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    return np.quantile(values,[.05,.5,.95,1.]).tolist() if len(values) else []


def new_output(path):
    path=Path(path)
    if path.exists(): raise ValueError('Use a new artifact directory; inputs and completed outputs are immutable')
    path.mkdir(parents=True)
    return path


def original_problem(directory, y, frames, groups, meta, provenance):
    """Rebuild and verify the original objective, not a refitted replacement."""
    directory=Path(directory)
    with np.load(directory/'checkpoint.npz',allow_pickle=False) as checkpoint:
        identity=json.loads(str(checkpoint['manifest_json']))
        x=checkpoint['states'].copy();coef=checkpoint['coefficients'].copy()
    selected=identity['training_fixations']
    if (len(selected)!=len(set(selected)) or selected!=sorted(selected) or
            not selected or not set(selected)<=set(range(20))):
        raise ValueError('Invalid source training fixation contract')
    mask=np.isin(groups,selected)
    global_groups=groups[mask].copy();local_groups=np.searchsorted(selected,global_groups)
    previous=identity.get('previous_mean_prior') or {}
    problem=ProfiledProblem(y[mask].copy(),frames[mask].copy(),local_groups,
        np.array([meta[j]['target_theta_deg'] for j in selected]),
        np.array([meta[j]['demand_diopters_label'] for j in selected]), identity['p'],
        np.asarray(identity['initial_coefficients']),np.asarray(identity['precision']),np.asarray(identity['prior_precision']),
        coefficient_map=identity.get('coefficient_map'),curvature=len(coef)==15,
        curvature_strength=identity.get('curvature_strength',0.),
        curvature_scale_output=identity.get('curvature_scale_output',1.),
        previous_means=previous.get('means_deg'),previous_mean_strength=previous.get('strength',0.),
        previous_mean_scale_deg=previous.get('scale_deg',1.),previous_mean_provenance=previous.get('source'))
    # This checks model/checkpoint equality, hashes/support, bounds, initial prior,
    # rebuilt manifest, and same-objective coefficient/robust-weight reconstruction.
    frozen.previous_source(directory,problem,selected,global_groups,provenance)
    rebuilt,omega,info=problem.robust_coefficients(x,identity['kappa'],maxiter=500,tol=1e-10)
    if not info['converged'] or not np.allclose(rebuilt,coef,atol=1e-7,rtol=1e-6):
        raise ValueError('Original objective coefficient reconstruction failed')
    model=json.loads((directory/'model.json').read_text())
    source=dict(directory=str(directory.resolve()),model_sha256=sha(directory/'model.json'),
        checkpoint_sha256=sha(directory/'checkpoint.npz'),training_fixations=selected,
        frame_hash=array_hash(problem.frames),global_group_hash=array_hash(global_groups),
        observation_hash=array_hash(problem.y),source_sha256=provenance['source_sha256'],
        interval_sha256=provenance['selected_interval_sha256'],schema=model['schema'])
    return problem,x,coef,omega,identity,model,selected,source


def conditional_geometry(problem,x,coef):
    """Local fixed-coefficient noise propagation, not joint-fit uncertainty."""
    theta,a,A,_=problem.decode(x)
    H,dt,da=basis(theta,a,problem.p,curvature=problem.curvature)
    defined=(A>0)|(problem.p==1)
    J=np.full((problem.n,2,2),np.nan)
    J[:,:,0]=dt@coef
    J[defined,:,1]=(da@coef)[defined]*(problem.p*A[defined]**(problem.p-1))[:,None]
    condition=np.full(problem.n,np.nan);minimum=np.full(problem.n,np.nan)
    theta_noise=np.full(problem.n,np.nan);A_noise=np.full(problem.n,np.nan)
    noise_correlation=np.full(problem.n,np.nan);raw_inverse=np.full_like(J,np.nan)
    rank=np.zeros(problem.n,dtype=int)
    if defined.any():
        whitened=np.einsum('ab,nbc->nac',problem.L,J[defined])*REFERENCE_UNITS
        singular=np.linalg.svd(whitened,compute_uv=False)
        rank[defined]=np.sum(singular>singular[:,[0]]*1e-12,axis=1)
        condition[defined]=singular[:,0]/np.maximum(singular[:,1],1e-30)
        minimum[defined]=singular[:,1]
        noise_map=REFERENCE_UNITS[None,:,None]*np.linalg.pinv(whitened,rcond=1e-12)
        covariance=noise_map@np.swapaxes(noise_map,1,2)
        theta_noise[defined]=np.sqrt(covariance[:,0,0]);A_noise[defined]=np.sqrt(covariance[:,1,1])
        noise_correlation[defined]=covariance[:,0,1]/np.maximum(np.sqrt(covariance[:,0,0]*covariance[:,1,1]),1e-30)
        raw_inverse[defined]=noise_map@problem.L
    knot=np.zeros(problem.n,bool)
    for value in KNOTS[1:-1]: knot |= abs(A-value)<=16*np.finfo(float).eps*max(1.,value)
    return H@coef,dict(defined=defined,rank=rank,condition=condition,minimum=minimum,
        theta_noise_deg=theta_noise,A_noise_D=A_noise,noise_correlation=noise_correlation,
        theta_per_d=raw_inverse[:,0,0],theta_per_rho=raw_inverse[:,0,1],
        A_per_d=raw_inverse[:,1,0],A_per_rho=raw_inverse[:,1,1],interior_knot=knot,
        theta_bound=abs(theta)>=20-1e-5,A_bound=(A<=1e-5)|(A>=6-1e-5),
        nominal_knot_extrapolation=(A<KNOTS[0])|(A>KNOTS[-1]))


def diagnose(args):
    y,frames,groups,meta,records,provenance=load_data(args.experiment_dir,args.intervals)
    models=DEFAULT_MODELS.copy()
    for assignment in args.model:
        label,separator,path=assignment.partition('=')
        if not separator or not label: raise ValueError('Use LABEL=model directory')
        models[label]=Path(path)
    output=new_output(args.output_dir)
    reports,fixation_rows,matrix=[],[],[]
    for label,directory in models.items():
        problem,x,coef,omega,identity,model,selected,source=original_problem(directory,y,frames,groups,meta,provenance)
        theta,_,A,_=problem.decode(x)
        value,parts=problem.true_objective(x,coef,identity['kappa'])
        penalty=problem.penalties(x);anchors=penalty[:2*problem.J].reshape(-1,2)
        temporal_end=2*problem.J+2*len(problem.left)
        temporal=penalty[2*problem.J:temporal_end].reshape(-1,2)
        split=dict(parts,theta_anchor=float(.5*np.sum(anchors[:,0]**2)),
            A_anchor=float(.5*np.sum(anchors[:,1]**2)),theta_temporal=float(.5*np.sum(temporal[:,0]**2)),
            A_temporal=float(.5*np.sum(temporal[:,1]**2)))
        prediction,geometry=conditional_geometry(problem,x,coef)
        error=prediction-problem.y;whitened=error@problem.L.T
        squared=np.sum(whitened**2,axis=1)
        kappa=identity['kappa']
        block_loss=squared if kappa is None else 2*kappa*kappa*(np.sqrt(1+squared/kappa**2)-1)
        total_gradient=envelope_gradients(problem,x,coef,omega)[0]
        H,dt,da=basis(theta,A**problem.p,problem.p,curvature=problem.curvature)
        force=(error@problem.W)*(problem.alpha*omega)[:,None]
        optical_gradient=np.full((problem.n,2),np.nan)
        optical_gradient[:,0]=np.sum(force*(dt@coef),axis=1)
        defined=geometry['defined']
        optical_gradient[defined,1]=np.sum(force[defined]*(da@coef)[defined],axis=1)*problem.p*A[defined]**(problem.p-1)
        local=[]
        for j,global_j in enumerate(selected):
            mask=problem.groups==j
            row=dict(model=label,fixation_index=global_j,capture=meta[global_j]['capture'],count=int(mask.sum()),
                nominal_theta_deg=float(problem.targets[j]),nominal_demand_D=float(problem.demands[j]),
                fitted_theta_mean_deg=float(theta[mask].mean()),fitted_A_mean_D=float(A[mask].mean()),
                nominal_mean_theta_offset_deg=float(theta[mask].mean()-problem.targets[j]),
                nominal_mean_A_offset_D=float(A[mask].mean()-problem.demands[j]),
                within_fixation_theta_SD_deg=float(theta[mask].std()),within_fixation_A_SD_D=float(A[mask].std()),
                optical_objective=float(.5*np.sum(problem.alpha[mask]*block_loss[mask])),
                d_RMSE=float(np.sqrt(np.mean(error[mask,0]**2))),rho_RMSE=float(np.sqrt(np.mean(error[mask,1]**2))),
                theta_anchor_objective=float(.5*anchors[j,0]**2),A_anchor_objective=float(.5*anchors[j,1]**2),
                theta_optical_gradient_RMS=float(np.sqrt(np.mean(optical_gradient[mask,0]**2))),
                theta_total_gradient_RMS=float(np.sqrt(np.mean(total_gradient[mask,0]**2))),
                A_optical_gradient_RMS=float(np.sqrt(np.nanmean(optical_gradient[mask,1]**2))),
                A_total_gradient_RMS=float(np.sqrt(np.mean(total_gradient[mask,1]**2))),
                physical_derivative_undefined_count=int((~defined[mask]).sum()),
                optical_rank_deficient_count=int((geometry['rank'][mask]<2).sum()),
                interior_knot_count=int(geometry['interior_knot'][mask].sum()),
                theta_bound_count=int(geometry['theta_bound'][mask].sum()),A_bound_count=int(geometry['A_bound'][mask].sum()),
                nominal_knot_extrapolation_count=int(geometry['nominal_knot_extrapolation'][mask].sum()))
            detail=dict(row,condition_quantiles=finite_quantiles(geometry['condition'][mask]),
                minimum_scaled_singular_quantiles=finite_quantiles(geometry['minimum'][mask]),
                conditional_theta_noise_SD_deg_quantiles=finite_quantiles(geometry['theta_noise_deg'][mask]),
                conditional_A_noise_SD_D_quantiles=finite_quantiles(geometry['A_noise_D'][mask]),
                conditional_theta_A_noise_correlation_quantiles=finite_quantiles(geometry['noise_correlation'][mask]),
                theta_per_d_quantiles=finite_quantiles(geometry['theta_per_d'][mask]),
                theta_per_rho_quantiles=finite_quantiles(geometry['theta_per_rho'][mask]),
                A_per_d_quantiles=finite_quantiles(geometry['A_per_d'][mask]),
                A_per_rho_quantiles=finite_quantiles(geometry['A_per_rho'][mask]))
            local.append(detail);fixation_rows.append(row)
        report=dict(label=label,source=source,training_frame_count=problem.n,training_fixation_count=problem.J,
            original_objective=value,objective_components=split,loss=identity['loss'],kappa=kappa,
            source_converged=model['diagnostics']['converged'],source_termination=model['diagnostics'].get('termination'),
            nominal_consistency=dict(theta_mean_offset_RMSE_deg=float(np.sqrt(np.mean([r['nominal_mean_theta_offset_deg']**2 for r in local]))),
                A_mean_offset_RMSE_D=float(np.sqrt(np.mean([r['nominal_mean_A_offset_D']**2 for r in local])))),
            conditional_geometry=dict(reference_units=REFERENCE_UNITS.tolist(),
                condition_quantiles=finite_quantiles(geometry['condition']),
                theta_noise_SD_deg_quantiles=finite_quantiles(geometry['theta_noise_deg']),
                A_noise_SD_D_quantiles=finite_quantiles(geometry['A_noise_D']),
                interpretation='local first-order propagation of estimated optical noise at fixed fitted coefficients; ignores coefficient uncertainty, correlated reuse, model bias and global inverse branches; not an accuracy estimate or confidence interval'),
            fixation_diagnostics=local,
            interpretation='soft nominal-mean consistency and fitted-state variability are distinct from independent accuracy; original optical/prior contributions are reported separately, not ranked across objectives')
        reports.append(report)
        matrix.append(dict(model=label,frames=problem.n,fixations=problem.J,objective=value,
            optical=parts['observations'],theta_anchor=split['theta_anchor'],A_anchor=split['A_anchor'],
            model_prior=parts['model_prior'],curvature_prior=parts.get('curvature_prior',0.),
            previous_mean_prior=parts.get('previous_mean_prior',0.),theta_temporal=split['theta_temporal'],A_temporal=split['A_temporal'],
            nominal_theta_mean_offset_RMSE_deg=report['nominal_consistency']['theta_mean_offset_RMSE_deg'],
            nominal_A_mean_offset_RMSE_D=report['nominal_consistency']['A_mean_offset_RMSE_D']))
        if sha(directory/'model.json')!=source['model_sha256'] or sha(directory/'checkpoint.npz')!=source['checkpoint_sha256']:
            raise ValueError('Input model/checkpoint changed during diagnosis')
        print(json.dumps(dict(stage='diagnose',model=label,objective=value,parts=split)),flush=True)
    write_csv(output/'objective_and_nominal_consistency.csv',matrix)
    write_csv(output/'fixation_diagnostics.csv',fixation_rows)
    atomic_json(output/'current_solution_diagnostic.json',dict(schema=SCHEMA,provenance=provenance,models=reports,
        independent_accuracy_evaluated=False,independent_reference_available=False,
        next_gate='reoptimize fixation-mean probe directions and anchor/initialization sensitivities before choosing a relative model',
        interpretation='No model-derived trajectory is treated as ground truth; conditional noise diagnostics do not establish statistical or physiological identification'))


def measurement_rows(meta,records,selected=None):
    """Original support and relative coordinates for descriptive scale analysis."""
    selected=set(range(20) if selected is None else selected)
    rows=[];ordering=[]
    from observations import relative_measurements
    for record in records:
        arrays=record['arrays'];relative=relative_measurements(arrays)
        eligible=record['core']&record['valid']&np.isin(record['groups'],list(selected))
        good=eligible&relative['valid']
        ordering.append(dict(capture=record['capture'],eligible_count=int(eligible.sum()),
            ordering_failure_count=int((eligible&relative['ordering_failure']).sum())))
        for index in np.flatnonzero(good):
            fixation=int(record['groups'][index]);m,s1,s4=relative['z'][index]
            row=dict(capture=record['capture'],fixation_index=fixation,frame_index=int(arrays['frame_index'][index]),
                m=float(m),S1=float(s1),S4=float(s4),nominal_gaze_deg=float(meta[fixation]['target_theta_deg']),
                demand_D=float(meta[fixation]['demand_diopters_label']),timestamp_ms=float(arrays['timestamp_ms'][index]),
                timestamp_reliable=record['capture']!=1)
            for field in ['pupil_angle_deg','p1_angle_deg','p4_angle_deg','p1_count','p4_count',
                          'p4_inside_count','scale_ratio','scale_difference_percent']:
                if field in arrays:
                    value=np.asarray(arrays[field])[index]
                    if np.ndim(value)==0: row['detector_'+field]=float(value)
            if 'pupil_axes' in arrays:
                for component,value in enumerate(np.asarray(arrays['pupil_axes'])[index]):
                    row['detector_pupil_axis_'+str(component)]=float(value)
            rows.append(row)
    return rows,ordering


def scale(args):
    from scale_diagnostics import run_scale_diagnostics
    y,frames,groups,meta,records,provenance=load_data(args.experiment_dir,args.intervals)
    rows,ordering=measurement_rows(meta,records)
    provenance=dict(provenance,ordering_audit=ordering,
        time_policy='original frame order; capture1 timestamps unreliable; no gaze/demand randomization')
    run_scale_diagnostics(rows,args.output_dir,provenance=provenance)


def source_clone(problem,**settings):
    if problem.curvature or problem.previous_mean_strength!=0:
        raise ValueError('This declared sensitivity/profile family requires q OFF and previous-mean prior OFF')
    return SensitivityProblem(problem.y.copy(),problem.frames.copy(),problem.groups.copy(),
        problem.targets.copy(),problem.demands.copy(),problem.p,problem.initial_coef.copy(),problem.W.copy(),
        problem.prior_W.copy(),coefficient_map=problem.coefficient_map,**settings)


def mean_profile(args):
    y,frames,groups,meta,records,provenance=load_data(args.experiment_dir,args.intervals)
    original,x,coef,omega,identity,model,selected,source=original_problem(args.model_dir,y,frames,groups,meta,provenance)
    if args.fixation not in selected: raise ValueError('Requested fixation is absent from source training support')
    local=selected.index(args.fixation)
    baseline_mean=float(original.decode(x)[0][original.groups==local].mean())
    target=baseline_mean+args.mean_offset_deg
    problem=source_clone(original,probe_group=local,probe_target_deg=target,
        probe_strength=args.probe_strength,probe_scale_deg=args.probe_scale_deg)
    output=new_output(args.output_dir)
    run_provenance=dict(provenance,training_fixations=selected,heldout_fixation=None,
        heldout_fixations=[j for j in range(20) if j not in selected],source_model=source,
        experiment='full_information_mean_probe_v1',
        nominal_anchor_policy='unchanged original objective; separate probe penalty',
        original_training_support_unchanged=True)
    global_groups=groups[np.isin(groups,selected)]
    manifest=make_manifest(problem,run_provenance,global_groups,identity['kappa'])
    probe=dict(global_fixation=args.fixation,local_group=local,baseline_fitted_mean_deg=baseline_mean,
        requested_offset_deg=args.mean_offset_deg,requested_mean_deg=target,strength=args.probe_strength,scale_deg=args.probe_scale_deg)
    manifest.update(schema='full_information_mean_probe_v1',experiment='full_information_mean_probe_v1',
        source=source,mean_probe=probe)
    manifest['settings']=dict(manifest['settings'],mean_probe=probe)
    resume_initial=None;prior_phases=[]
    if args.resume_dir:
        x,verified_coef,verified_weights,resume_info=read_checkpoint(args.resume_dir,manifest,problem,global_groups,False)
        prior_result=json.loads((args.resume_dir/'mean_profile.json').read_text())
        if prior_result['probe']!=probe or prior_result['source']!=source:
            raise ValueError('Resume probe/source differs from the original guarded objective')
        with np.load(args.resume_dir/'checkpoint.npz',allow_pickle=False) as saved:
            saved_coef=saved['coefficients'].copy();saved_weights=saved['weights'].copy()
        if not np.allclose(saved_coef,verified_coef,rtol=1e-7,atol=1e-6) or not np.allclose(saved_weights,verified_weights,rtol=1e-7,atol=1e-6):
            raise ValueError('Resume checkpoint coefficients/weights differ from reconstruction')
        resume_initial=dict(coefficients=saved_coef,weights=saved_weights,objective=prior_result['augmented_objective'])
        prior_phases=prior_result.get('continuation_phases',[])
        if not prior_phases:
            previous_status=json.loads((args.resume_dir/'status.json').read_text())
            prior_phases=[dict(directory=str(args.resume_dir.resolve()),status=previous_status,
                checkpoint_sha256=sha(args.resume_dir/'checkpoint.npz'),history_sha256=sha(args.resume_dir/'history.json'),
                lsmr_history_sha256=sha(args.resume_dir/'lsmr_history.json'))]
    with np.load(args.model_dir/'checkpoint.npz',allow_pickle=False) as baseline_checkpoint:
        baseline_states=baseline_checkpoint['states'].copy()
    baseline_value,baseline_parts=original.true_objective(baseline_states,coef,identity['kappa'])
    atomic_json(output/'run_provenance.json',run_provenance)
    x,coef,history,converged,status=continue_fit(problem,x,output,manifest,kappa=identity['kappa'],
        max_nfev=args.max_nfev,wall_seconds=args.wall_seconds,lsmr_maxiter=args.lsmr_maxiter,
        lsmr_atol=args.lsmr_atol,lsmr_btol=args.lsmr_btol,robust_outer=args.robust_outer,
        robust_max_nfev=args.robust_max_nfev,physical_gtol=1e-5,physical_step_tol=1e-5,
        relative_cost_tol=1e-10,weight_tol=1e-7,resume_initial=resume_initial)
    coef,omega,inner=problem.robust_coefficients(x,identity['kappa'],maxiter=500,tol=1e-10)
    value,parts=problem.true_objective(x,coef,identity['kappa'])
    kkt=problem.constrained_mean_stationarity(x,coef,omega)
    theta,_,A,_=problem.decode(x)
    source_value=value-parts['mean_probe']
    phases=prior_phases+[dict(directory=str(output.resolve()),status=status,checkpoint_sha256=sha(output/'checkpoint.npz'),
        history_sha256=sha(output/'history.json'),lsmr_history_sha256=sha(output/'lsmr_history.json'))]
    totals={key:sum(float(phase['status'].get(key,0)) for phase in phases) for key in ('elapsed_seconds','nfev','accepted_iterations')}
    results=dict(schema='achieved_mean_local_profile_v1',source=source,probe=probe,
        continuation_phases=phases,cumulative_continuation_totals=totals,
        resume_initial_objective_verified=resume_initial is not None,
        achieved_offset_deg=kkt['achieved_mean_deg']-baseline_mean,
        augmented_objective=value,original_objective=source_value,original_objective_components={k:v for k,v in parts.items() if k!='mean_probe'},
        probe_objective=parts['mean_probe'],baseline_original_objective=baseline_value,baseline_original_parts=baseline_parts,
        original_objective_change=source_value-baseline_value,calibration_converged=converged,
        constrained_stationarity=kkt,constrained_stationarity_verified=(kkt['constrained_physical_projected_optimality']<=1e-5 and
            kkt['zero_A_feasible_descent_count']==0 and kkt['zero_A_stationarity_unverified_count']==0),
        continuation_status=status,coefficient_irls=inner,
        interpretation='Other training states and coefficients were reoptimized. This is a local constrained-stationary point at achieved mean only if KKT passes; no exact requested-mean or global profile/CI claim.')
    atomic_json(output/'mean_profile.json',results)
    write_csv(output/'reoptimized_training_states.csv',[dict(fixation_index=selected[int(problem.groups[i])],
        frame_index=int(problem.frames[i]),theta_deg=float(theta[i]),A_diopters=float(A[i])) for i in range(problem.n)])
    print(json.dumps(results),flush=True)


def penalty_breakdown(problem,x):
    penalties=problem.penalties(x)
    anchors=penalties[:2*problem.J].reshape(-1,2)
    temporal=penalties[2*problem.J:2*problem.J+2*len(problem.left)].reshape(-1,2)
    return dict(theta_anchor_objective=float(.5*np.sum(anchors[:,0]**2)),
        A_anchor_objective=float(.5*np.sum(anchors[:,1]**2)),
        theta_temporal_objective=float(.5*np.sum(temporal[:,0]**2)),
        A_temporal_objective=float(.5*np.sum(temporal[:,1]**2)))


def profile_summary(args):
    output=new_output(args.output_dir);rows=[];sources=[];source_cache={};baseline_added=set()
    y,frames,groups,meta,records,provenance=load_data(EXP,EXP/'fixations'/'fixation_intervals.json',None)
    all_results={str(directory.resolve()):json.loads((directory/'mean_profile.json').read_text()) for directory in args.profile_dir}
    continuation_ancestors={str(Path(phase['directory']).resolve()) for directory,result in all_results.items()
        for phase in result.get('continuation_phases',[]) if str(Path(phase['directory']).resolve())!=directory}
    for directory in args.profile_dir:
        path=directory/'mean_profile.json';result=json.loads(path.read_text());probe=result['probe'];parts=result['original_objective_components']
        sources.append(dict(path=str(path.resolve()),sha256=sha(path),continuation_ancestor=str(directory.resolve()) in continuation_ancestors))
        if str(directory.resolve()) in continuation_ancestors: continue
        source_label=Path(result['source']['directory']).parent.name
        if Path(result['source']['directory']).resolve()==DEFAULT_MODELS['H0'].resolve(): source_label='H0'
        source_dir=Path(result['source']['directory'])
        if str(source_dir) not in source_cache:
            source_cache[str(source_dir)]=original_problem(source_dir,y,frames,groups,meta,provenance)
        original,baseline_x,_,_,_,_,_,_=source_cache[str(source_dir)]
        with np.load(directory/'checkpoint.npz',allow_pickle=False) as checkpoint:
            profile_x=checkpoint['states'].copy()
        if profile_x.shape!=baseline_x.shape: raise ValueError('Profile and source training support differ')
        split=penalty_breakdown(original,profile_x)
        if not np.isclose(split['theta_anchor_objective']+split['A_anchor_objective'],parts['anchors'],rtol=1e-12,atol=1e-12):
            raise ValueError('Saved profile anchor decomposition differs from checkpoint state')
        rows.append(dict(source=source_label,profile_directory=str(directory.resolve()),fixation=probe['global_fixation'],
            baseline_mean_deg=probe['baseline_fitted_mean_deg'],requested_mean_deg=probe['requested_mean_deg'],
            achieved_mean_deg=result['constrained_stationarity']['achieved_mean_deg'],achieved_offset_deg=result['achieved_offset_deg'],
            original_objective=result['original_objective'],optical_objective=parts['observations'],
            anchor_objective=parts['anchors'],temporal_objective=parts['temporal'],model_prior_objective=parts['model_prior'],
            probe_objective=result['probe_objective'],calibration_converged=result['calibration_converged'],
            constrained_stationarity_verified=result['constrained_stationarity_verified'],
            constrained_physical_projected_optimality=result['constrained_stationarity']['constrained_physical_projected_optimality'],
            cumulative_elapsed_seconds=result.get('cumulative_continuation_totals',result['continuation_status']).get('elapsed_seconds'),
            cumulative_nfev=result.get('cumulative_continuation_totals',result['continuation_status']).get('nfev'),
            cumulative_accepted_iterations=result.get('cumulative_continuation_totals',result['continuation_status']).get('accepted_iterations'),
            sample_kind='reoptimized_mean_probe',**split))
        if str(source_dir) not in baseline_added:
            baseline_added.add(str(source_dir));base=result['baseline_original_parts']
            rows.append(dict(source=source_label,profile_directory=str(source_dir.resolve()),fixation=probe['global_fixation'],
                baseline_mean_deg=probe['baseline_fitted_mean_deg'],requested_mean_deg=probe['baseline_fitted_mean_deg'],
                achieved_mean_deg=probe['baseline_fitted_mean_deg'],achieved_offset_deg=0.,
                original_objective=result['baseline_original_objective'],optical_objective=base['observations'],
                anchor_objective=base['anchors'],temporal_objective=base['temporal'],model_prior_objective=base['model_prior'],
                probe_objective=0.,calibration_converged=True,constrained_stationarity_verified=None,
                constrained_physical_projected_optimality=None,sample_kind='frozen_source_baseline',**penalty_breakdown(original,baseline_x)))
    baselines={r['source']:r for r in rows if r['sample_kind']=='frozen_source_baseline'}
    for row in rows:
        baseline=baselines[row['source']]
        for field in ('original_objective','optical_objective','anchor_objective','temporal_objective','model_prior_objective',
            'theta_anchor_objective','A_anchor_objective','theta_temporal_objective','A_temporal_objective'):
            row['change_'+field]=row[field]-baseline[field]
    write_csv(output/'achieved_mean_profiles.csv',rows)
    atomic_json(output/'achieved_mean_profiles.json',dict(rows=rows,sources=sources,
        interpretation='Local achieved-mean profile samples, with incomplete/nonstationary points explicitly retained; no formal confidence interval'))
    lines=['# Achieved-mean local profiles','',
        'Offsets and cost changes are relative to each frozen source solution. Other states and coefficients were reoptimized. Only points passing the original constrained KKT gate are verified; these are local samples, not global profiles or confidence intervals.','',
        '| Source | Achieved offset (deg) | Optical cost change | Gaze-anchor change | Accommodation-anchor change | KKT | Solver converged |',
        '|---|---:|---:|---:|---:|---:|---|']
    for row in rows:
        kkt='' if row['constrained_physical_projected_optimality'] is None else f"{row['constrained_physical_projected_optimality']:.4g}"
        lines.append(f"| {row['source']} | {row['achieved_offset_deg']:.6g} | {row['change_optical_objective']:.6g} | {row['change_theta_anchor_objective']:.6g} | {row['change_A_anchor_objective']:.6g} | {kkt} | {row['calibration_converged']} |")
    (output/'achieved_mean_profiles.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for label in sorted(set(r['source'] for r in rows)):
        local=sorted([r for r in rows if r['source']==label],key=lambda r:r['achieved_mean_deg'])
        accepted=[r for r in local if r['sample_kind']=='frozen_source_baseline' or (r['calibration_converged'] and r['constrained_stationarity_verified'])]
        incomplete=[r for r in local if r not in accepted]
        for axis,field in zip(axes,['change_optical_objective','change_anchor_objective','change_model_prior_objective']):
            line=axis.plot([r['achieved_mean_deg'] for r in accepted],[r[field] for r in accepted],marker='o',label=label+' verified/baseline')[0]
            if incomplete:
                axis.plot([r['achieved_mean_deg'] for r in incomplete],[r[field] for r in incomplete],marker='o',
                    markerfacecolor='none',linestyle='none',color=line.get_color(),label=label+' incomplete')
                for row in incomplete: axis.annotate('not verified',(row['achieved_mean_deg'],row[field]),fontsize=7,xytext=(3,5),textcoords='offset points')
            axis.set(xlabel='Achieved fixation-mean gaze (degrees)',ylabel=field)
            axis.legend()
    fig.savefig(output/'achieved_mean_optical_and_penalties.png',dpi=140);plt.close(fig)


def sensitivity(args):
    y,frames,groups,meta,records,provenance=load_data(args.experiment_dir,args.intervals,None)
    original,x,coef,omega,identity,model,selected,source=original_problem(args.model_dir,y,frames,groups,meta,provenance)
    settings=dict(theta_anchor_multiplier=args.anchor_multiplier,A_anchor_multiplier=args.anchor_multiplier)
    problem=source_clone(original,**settings)
    theta,_,A,_=problem.decode(x)
    baseline_theta,baseline_A=theta.copy(),A.copy()
    theta_clipped=A_clipped=0
    if args.perturb_initial_states:
        rng=np.random.default_rng(args.seed)
        theta_raw=theta+rng.normal(0,.2,len(theta))
        A_raw=A+rng.normal(0,.05,len(A))
        theta_clipped=int(np.count_nonzero((theta_raw< -20)|(theta_raw>20)))
        A_clipped=int(np.count_nonzero((A_raw<0)|(A_raw>6)))
        theta=np.clip(theta_raw,-20,20)
        A=np.clip(A_raw,0,6)
        x=problem.encode(theta,A)
    output=new_output(args.output_dir)
    held=[j for j in range(20) if j not in selected]
    run_provenance=dict(provenance,training_fixations=selected,heldout_fixations=held,
        heldout_fixation=held[0] if len(held)==1 else None,
        source=source,experiment='full_information_sensitivity')
    manifest=make_manifest(problem,run_provenance,groups[np.isin(groups,selected)],identity['kappa'])
    sensitivity_settings=dict(anchor_multiplier=args.anchor_multiplier,both_nominal_anchors=True,
        perturb_initial_states=args.perturb_initial_states,seed=args.seed,
        theta_initial_sd_deg=.2 if args.perturb_initial_states else 0.,A_initial_sd_D=.05 if args.perturb_initial_states else 0.,
        theta_initial_clipped_count=theta_clipped,A_initial_clipped_count=A_clipped,
        unchanged_initial_coefficients=original.initial_coef.tolist(),
        unchanged_prior_matrix_hash=array_hash(original.prior_M),unchanged_prior_value_hash=array_hash(original.prior_v),
        coefficient_prior_unchanged=True,initialization='verified fitted source states; deterministic independent Gaussian state perturbation clipped to physical bounds' if args.perturb_initial_states else 'verified fitted source states')
    manifest.update(schema='full_information_sensitivity_v1',experiment='full_information_sensitivity',sensitivity=sensitivity_settings,source=source)
    manifest['settings']=dict(manifest.get('settings',{}),sensitivity=sensitivity_settings)
    if args.resume_dir is not None: x,_,_,_=read_checkpoint(args.resume_dir,manifest,problem,groups[np.isin(groups,selected)])
    x,coef,history,converged,status=continue_fit(problem,x,output,manifest,kappa=identity['kappa'],
        max_nfev=args.max_nfev,wall_seconds=args.wall_seconds,lsmr_maxiter=args.lsmr_maxiter,
        lsmr_atol=args.lsmr_atol,lsmr_btol=args.lsmr_btol,robust_outer=args.robust_outer,robust_max_nfev=args.robust_max_nfev)
    value,parts=problem.true_objective(x,coef,identity['kappa'])
    original_value,original_parts=original.true_objective(x,coef,identity['kappa'])
    theta,_,A,_=problem.decode(x)
    rows=[]
    for local,j in enumerate(selected):
        mask=problem.groups==local
        rows.append(dict(fixation_index=j,training_support=True,nominal_theta_deg=float(problem.targets[local]),nominal_A_D=float(problem.demands[local]),
            theta_mean_deg=float(theta[mask].mean()),A_mean_D=float(A[mask].mean()),theta_sd_deg=float(theta[mask].std()),A_sd_D=float(A[mask].std()),
            theta_mean_change_from_source_deg=float((theta[mask]-baseline_theta[mask]).mean()),A_mean_change_from_source_D=float((A[mask]-baseline_A[mask]).mean())))
    write_csv(output/'fixation_sensitivity.csv',rows)
    atomic_json(output/'sensitivity.json',dict(schema='full_information_sensitivity_result_v1',source=source,
        settings=sensitivity_settings,converged=bool(converged),status=status,objective=value,parts=parts,
        separate_penalty_components=penalty_breakdown(problem,x),
        original_objective_at_result=original_value,original_parts_at_result=original_parts,fixations=rows,
        interpretation='local objective/initialization sensitivity; nominal targets and source states are not independent truth'))
    print(json.dumps(dict(output_dir=str(output),converged=bool(converged),objective=value)))


def sensitivity_summary(args):
    output=new_output(args.output_dir);cases=[];fixations=[];sources=[]
    for directory in args.sensitivity_dir:
        path=directory/'sensitivity.json';result=json.loads(path.read_text());parts=result['parts']
        local=result['fixations'];settings=result['settings'];label=directory.name
        cases.append(dict(case=label,converged=result['converged'],anchor_multiplier=settings['anchor_multiplier'],
            perturb_initial_states=settings['perturb_initial_states'],seed=settings['seed'],objective=result['objective'],
            optical_objective=parts['observations'],anchor_objective=parts['anchors'],temporal_objective=parts['temporal'],
            model_prior_objective=parts['model_prior'],elapsed_seconds=result['status'].get('elapsed_seconds'),
            equal_fixation_theta_mean_change_RMS_deg=float(np.sqrt(np.mean([r['theta_mean_change_from_source_deg']**2 for r in local]))),
            equal_fixation_A_mean_change_RMS_D=float(np.sqrt(np.mean([r['A_mean_change_from_source_D']**2 for r in local]))),
            **result.get('separate_penalty_components',{})))
        fixations.extend(dict(case=label,**row) for row in local)
        sources.append(dict(path=str(path.resolve()),sha256=sha(path)))
    write_csv(output/'sensitivity_cases.csv',cases);write_csv(output/'sensitivity_fixations.csv',fixations)
    atomic_json(output/'sensitivity_summary.json',dict(cases=cases,fixations=fixations,sources=sources,
        interpretation='Same source cohort/prior; anchor objectives differ by declared strength. State changes measure local sensitivity, not accuracy or model ranking.'))
    lines=['# Anchor and initialization sensitivity','',
        'Both nominal anchors change together; the source observations and function prior remain fixed. State changes are local sensitivity, not accuracy errors.','',
        '| Case | Converged | Optical cost | Anchor cost | Gaze mean change RMS (deg) | Accommodation mean change RMS (D) |',
        '|---|---|---:|---:|---:|---:|']
    for row in cases:
        lines.append(f"| {row['case']} | {row['converged']} | {row['optical_objective']:.6g} | {row['anchor_objective']:.6g} | {row['equal_fixation_theta_mean_change_RMS_deg']:.6g} | {row['equal_fixation_A_mean_change_RMS_D']:.6g} |")
    (output/'sensitivity_summary.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
    for label in [r['case'] for r in cases]:
        local=sorted([r for r in fixations if r['case']==label],key=lambda r:r['fixation_index'])
        axes[0].plot([r['fixation_index'] for r in local],[r['theta_mean_deg']-r['nominal_theta_deg'] for r in local],marker='o',label=label)
        axes[1].plot([r['fixation_index'] for r in local],[r['A_mean_D']-r['nominal_A_D'] for r in local],marker='o',label=label)
    axes[0].set_ylabel('Gaze mean minus nominal target (deg)');axes[1].set_ylabel('Accommodation mean minus nominal demand (D)')
    axes[1].set_xlabel('Original fixation index');axes[0].legend();figure.suptitle('In-sample nominal consistency; no independent state reference')
    figure.savefig(output/'sensitivity_nominal_consistency.png',dpi=150);plt.close(figure)


def validation_contract(args):
    from validation import write_validation_contract
    print(json.dumps(write_validation_contract(args.output_dir)))


def validation_evaluate(args):
    from validation import evaluate_validation
    print(json.dumps(evaluate_validation(args.prediction_csv,args.reference_csv,args.protocol_json,args.output_dir)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='stage',required=True)
    command=sub.add_parser('diagnose')
    command.add_argument('--experiment-dir',type=Path,default=EXP)
    command.add_argument('--intervals',type=Path,default=EXP/'fixations/fixation_intervals.json')
    command.add_argument('--model',action='append',default=[],help='LABEL=immutable model directory; overrides/adds a source')
    command.add_argument('--output-dir',type=Path,required=True)
    command=sub.add_parser('validation-contract');command.add_argument('--output-dir',type=Path,required=True)
    command=sub.add_parser('validation-evaluate')
    command.add_argument('--prediction-csv',type=Path)
    command.add_argument('--reference-csv',type=Path)
    command.add_argument('--protocol-json',type=Path)
    command.add_argument('--output-dir',type=Path,required=True)
    command=sub.add_parser('sensitivity-summary')
    command.add_argument('--sensitivity-dir',type=Path,action='append',required=True)
    command.add_argument('--output-dir',type=Path,required=True)
    command=sub.add_parser('sensitivity')
    command.add_argument('--experiment-dir',type=Path,default=EXP)
    command.add_argument('--intervals',type=Path,default=EXP/'fixations'/'fixation_intervals.json')
    command.add_argument('--model-dir',type=Path,required=True)
    command.add_argument('--resume-dir',type=Path)
    command.add_argument('--anchor-multiplier',type=float,default=1.)
    command.add_argument('--perturb-initial-states',action='store_true')
    command.add_argument('--seed',type=int,default=20261004)
    command.add_argument('--output-dir',type=Path,required=True)
    for name,default,kind in [('max-nfev',600,int),('wall-seconds',900.,float),('lsmr-maxiter',300,int),('lsmr-atol',1e-7,float),('lsmr-btol',1e-7,float),('robust-outer',24,int),('robust-max-nfev',25,int)]:
        command.add_argument('--'+name,type=kind,default=default)
    command=sub.add_parser('scale')
    command.add_argument('--experiment-dir',type=Path,default=EXP)
    command.add_argument('--intervals',type=Path,default=EXP/'fixations/fixation_intervals.json')
    command.add_argument('--output-dir',type=Path,required=True)
    command=sub.add_parser('mean-profile')
    command.add_argument('--experiment-dir',type=Path,default=EXP)
    command.add_argument('--intervals',type=Path,default=EXP/'fixations/fixation_intervals.json')
    command.add_argument('--model-dir',type=Path,required=True);command.add_argument('--resume-dir',type=Path)
    command.add_argument('--fixation',type=int,required=True);command.add_argument('--mean-offset-deg',type=float,required=True)
    command.add_argument('--probe-strength',type=float,default=1.);command.add_argument('--probe-scale-deg',type=float,default=1.)
    command.add_argument('--output-dir',type=Path,required=True)
    for name,default,kind in [('max-nfev',600,int),('wall-seconds',900.,float),('lsmr-maxiter',300,int),
        ('lsmr-atol',1e-7,float),('lsmr-btol',1e-7,float),('robust-outer',24,int),('robust-max-nfev',25,int)]:
        command.add_argument('--'+name,type=kind,default=default)
    command=sub.add_parser('profile-summary');command.add_argument('--profile-dir',type=Path,action='append',required=True)
    command.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    globals()[args.stage.replace('-','_')](args)


if __name__=='__main__': main()
