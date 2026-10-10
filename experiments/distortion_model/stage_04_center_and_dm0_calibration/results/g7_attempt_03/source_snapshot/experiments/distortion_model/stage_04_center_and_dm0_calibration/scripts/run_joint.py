"""S7/G7 fresh full DM0 fit; all-row CuPy, two starts and compact masking."""
import argparse
from hashlib import sha256
from importlib.util import spec_from_file_location,module_from_spec
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'experiments/distortion_model/stage_03_p4_baseline_and_deformation/scripts'))
from g45_common import load_inputs,load_npz,start_output,write_json,describe
_g6_spec=spec_from_file_location('g6_parent_io',Path(__file__).with_name('run.py'))
_g6_io=module_from_spec(_g6_spec);_g6_spec.loader.exec_module(_g6_io)
verify_chain=_g6_io.verify_chain
from distortion_model.data import array_hash
from distortion_model.geometry import relative_coordinates,retained_indices
from distortion_model.joint import JointSpec,JointDM0,fit_joint
from distortion_model.inference import infer_retained
from distortion_model.optics import predict_relative

CONFIG={'stage':'S7/G7','candidate':'DM0-M1, degree2 forward center law',
        'population':'all complete frames of all twenty full reviewed intervals; captures1-4 only',
        'states':'two dynamic free states/frame: visual theta [-20,20] degrees and A [0,6] D',
        'scale':'positive P1-only GLS, reevaluated and differentiated at every trial',
        'metric':'unchanged full ten-coordinate raw covariance and full-exposure equal weights',
        'anchors':{'theta_mean_scale_deg':.10,'A_mean_scale_D':.25,'temporal':0.},
        'free_globals':['trace-free P1 alpha1/beta1 pair','P1 gamma1','m1',
                        'three P4 template tangent coordinates at fixed centroid/RMS radius',
                        'alpha4','beta4','gamma4','ten degree2 forward D coefficients'],
        'fixed_globals':['empirical omega1/omega4 separately fixed','P1 empirical template/origin/length',
                         'P4 template origin/RMS length','source identities/camera axes','Aref','degree2','no radial term'],
        'regularization':'0.5*sum((template_tangent/.10)^2)+0.5*sum((c2/P4_reference_radius)^2); other priors zero',
        'global_coordinates':'[100alpha1,10r1gamma1,10m1,u0,u1,u2,100alpha4,100beta4,10r4gamma4,C_D/r4]',
        'block_order':['A','theta with fresh g','P1','P4 baseline m1/template','K4','D','joint coupled refinement'],
        'chunk_size':32768,'max_outer':12,'joint_steps_per_outer':4,'observed_Newton_polish_steps':6,
        'same_objective_line_search':'up to14 halvings;64epsilon rounding allowance only with smaller projected gradient',
        'proposal_controls':'at most2degree/1D joint step before damping;LM1e-6 through1 for positive reduced proposal curvature',
        'stationarity_threshold':1e-6,'state_gradient_units':'full-J derivatives times computational state ranges',
        'global_gradient_units':'full-J derivatives times declared bounded parameter ranges, center coordinates C/r4',
        'curvature':'observed Hessian by FD of analytic gradients, full-mean Woodbury coupling and projected Schur rank',
        'curvature_fd_state_steps':[1e-4,1e-5],'curvature_fd_global_step':1e-5,'rank_rtol':1e-10,
        'starts':['reviewed common G6 initialization','PCG64 seed20261010 reproducible perturbation'],
        'backend':'CuPy float64 with same NumPy equations; one GPU process',
        'empirical_backend_audit':'first2 complete frames/exposure, benchmark only; both full means and chunked derivatives',
        'backend_agreement_rtol':3e-10,'backend_agreement_atol':1e-8,
        'gradient_audit_rtol':3e-6,'gradient_audit_atol':1e-5,
        'compact_schedule':'five equally spaced original scheduled rows/exposure, all3held-P4 slots, selected before fitting',
        'compact_completed_outer_checkpoints':[4,8,12],'compact_final_check':True,
        'compact_inference':'49fixed7x7 grid starts plus retained-only center proposal;max60iterations;no labels/anchors/global refit',
        'compact_status':'progress only; not G8; never selects a calibration checkpoint by cross-error',
        'automated_tests':'not added or run; benchmark/gradient/curvature evidence uses actual recorded arrays'}


def check_parent(parent):
    cp=json.loads((parent/'checkpoint.json').read_text());summary=json.loads((parent/'summary.json').read_text())
    if summary.get('G6_decision') not in ['GO','GO_WITH_LIMIT'] or not cp.get('center_block_certified'):
        raise ValueError('compatible reviewed G6 required')
    ancestor=ROOT/cp['parent_checkpoint_path']
    if sha256(ancestor.read_bytes()).hexdigest()!=cp['parent_checkpoint_sha256'] or sha256((ancestor.parent/'summary.json').read_bytes()).hexdigest()!=cp['parent_summary_sha256']:
        raise ValueError('changed reviewed G5 parent')
    inputs=load_inputs(ancestor.parent,None,'G5',0);verify_chain(inputs)
    arrays=load_npz(parent/cp['frame_states_file'],cp['frame_states_hash'])
    if sha256((parent/cp['model_file']).read_bytes()).hexdigest()!=cp['model_sha256']:raise ValueError('changed G6 model')
    record=json.loads((parent/cp['model_file']).read_text())
    if record.get('visual_degree')!=2 or record.get('scale_policy')!='p1_profile_v1' or record.get('schema')!='distortion_dual_alignment_v1':
        raise ValueError('incompatible model family/schema')
    if not np.array_equal(arrays['input_valid'],inputs['valid']):raise ValueError('changed population')
    inputs['parent']=parent;inputs['g6_checkpoint']=cp
    return inputs,arrays,record


def empirical_audit(spec,p,x,y,e,covariance,targets,backend):
    import cupy as cp
    indices=np.concatenate([np.flatnonzero(e==k)[:2] for k in range(20)])
    cpu=JointDM0(spec,y[indices],e[indices],covariance,targets,xp=np,chunk_size=13)
    gpu=JointDM0(spec,y[indices],e[indices],covariance,targets,xp=cp,chunk_size=7)
    started=time.perf_counter();a=cpu.evaluate(x[indices],p,hessian=True);cpu_time=time.perf_counter()-started
    gx,gp=cp.asarray(x[indices]),cp.asarray(p);gpu.evaluate(gx,gp,hessian=True);cp.cuda.Stream.null.synchronize()
    started=time.perf_counter();b=gpu.evaluate(gx,gp,hessian=True);cp.cuda.Stream.null.synchronize();gpu_time=time.perf_counter()-started
    differences={}
    for key in ['cost','point','theta_anchor','A_anchor','regularization','gx','gp','hx','cross','hg','means']:
        aa=np.asarray(a[key]);bb=gpu.host(b[key])
        if not np.allclose(aa,bb,rtol=CONFIG['backend_agreement_rtol'],atol=CONFIG['backend_agreement_atol']):
            raise ValueError('CPU/GPU empirical disagreement: '+key)
        differences[key]=float(np.max(np.abs(aa-bb)))
    # Actual recorded rows, before launching the expensive all-row fit.
    global_records=[];h=1e-5
    for j in range(19):
        plus=p.copy();minus=p.copy();plus[j]+=h;minus[j]-=h
        fd=(cpu.evaluate(x[indices],plus)['cost']-cpu.evaluate(x[indices],minus)['cost'])/(2*h)
        actual=float(a['gp'][j]);global_records.append({'coordinate':j,'analytic':actual,'finite_difference':fd,'absolute_difference':abs(actual-fd)})
        if not np.isclose(actual,fd,rtol=CONFIG['gradient_audit_rtol'],atol=CONFIG['gradient_audit_atol']):raise ValueError('empirical global gradient mismatch: '+str(j))
    rng=np.random.default_rng(20261010);direction=rng.normal(size=(len(indices),2))
    direction[(x[indices]<=np.array([-20.,0.])+1e-4)|(x[indices]>=np.array([20.,6.])-1e-4)]=0.
    fd=(cpu.evaluate(x[indices]+h*direction,p)['cost']-cpu.evaluate(x[indices]-h*direction,p)['cost'])/(2*h)
    analytic=float(np.sum(a['gx']*direction))
    if not np.isclose(analytic,fd,rtol=CONFIG['gradient_audit_rtol'],atol=CONFIG['gradient_audit_atol']):raise ValueError('empirical state/full-mean gradient mismatch')
    return {'population':'first2completeframes/exposure; not calibration subsampling','indices_in_complete_population':indices.tolist(),
            'rows':len(indices),'CPU_chunk_size':13,'GPU_chunk_size':7,'CPU_seconds':cpu_time,'GPU_warmed_seconds':gpu_time,
            'backend':backend,'maximum_absolute_differences':differences,'global_gradient_records':global_records,
            'state_directional_gradient':{'analytic':analytic,'finite_difference':fd,'absolute_difference':abs(analytic-fd)},
            'passed':True,'automated_test_suite_run':False}


def model_record(parent,spec,p,provenance):
    params=spec.parameters(p);record=dict(parent)
    record.update(stage='S7/G7',model_config_revision='S7_G7_DM0_joint_v1',source_hash=provenance['source_hash'],config_sha256=provenance['config_hash'],
                  b4_reference_px=params.b4.tolist(),P4_reference_template_before_joint_px=spec.b4.tolist(),
                  template_tangent_coordinates=np.asarray(p[3:6]).tolist(),template_tangent_basis=spec.template_basis.tolist(),
                  k1_native=list(params.k1),k4_native=list(params.k4),m1_per_D=params.m1,
                  center_coefficients_reference_px=params.center[:5].tolist(),
                  fixed_free_roster={'fixed':CONFIG['fixed_globals'],'free':CONFIG['free_globals']},
                  scope='fresh all-row joint fit; certificate in candidate fit record; physical accuracy remains unproven')
    return record


def compact_check(output,name,inputs,schedule,spec,p,calibration_certified=False):
    import cupy as cp
    population=inputs['population'];r=inputs['covariance']['relative'];records=[]
    params=spec.parameters(p)
    for held in range(3):
        keep=[j for j in range(3) if j!=held]
        retained_valid=population['p1_available'][schedule].all(axis=1)&population['p4_available'][schedule][:,keep].all(axis=1)
        rows=schedule[retained_valid]
        selected=retained_indices(held)
        y=relative_coordinates(population['p1'][rows],population['p4'][rows],held_point=held)
        solved=infer_retained(spec,p,y,r[np.ix_(selected,selected)],held,xp=cp) if len(rows) else None
        positions={int(row):j for j,row in enumerate(rows)}
        full_prediction=None;g=None
        if len(rows):
            x=solved['states'];full_prediction,g,pred_valid=predict_relative(x[:,0],x[:,1],y[:,:4],r[:4,:4],params)
        for row in schedule:
            record={'capture':str(population['capture'][row]),'exposure':int(population['exposure'][row]),
                    'row':int(population['row'][row]),'source_frame':int(population['source_frame'][row]),'held_point':held,
                    'calibration_status':'certified' if calibration_certified else 'progress_uncertified_snapshot',
                    'raw_input_valid':bool(population['complete_valid'][row]),'retained_input_valid':bool(retained_valid[np.flatnonzero(schedule==row)[0]]),
                    'held_input_valid':bool(population['p4_available'][row,held])}
            if row not in positions:record.update(inference_status='input_unavailable',reason='retained_input_unavailable')
            else:
                j=positions[int(row)];certificate=bool(solved['certified'][j])
                record.update(inference_status=('certified_bound' if solved['at_bounds'][j] else 'certified') if certificate else 'unresolved',
                              reason='ambiguous' if solved['ambiguous'][j] else ('none' if certificate else 'numerical_certificate_failed'),
                              state_visual_theta_deg=float(solved['states'][j,0]),state_A_D=float(solved['states'][j,1]),g=float(g[j]),
                              cost=float(solved['cost'][j]) if np.isfinite(solved['cost'][j]) else None,
                              projected_gradient_inf=float(solved['projected_gradient_inf'][j]),rank2=bool(solved['rank2'][j]),
                              ambiguous=bool(solved['ambiguous'][j]),at_bounds=bool(solved['at_bounds'][j]),
                              certified=certificate,starts=solved['starts'],candidate_certified_count=int(solved['candidate_certified_count'][j]),
                              observed_curvature_eigenvalues=solved['observed_curvature_eigenvalues'][j].tolist())
                # Held observation is read only after all state/branch decisions.
                if record['held_input_valid'] and pred_valid[j]:
                    observed=population['p4'][row,held]-population['p1'][row].mean(axis=0)
                    error=observed-full_prediction[j,4+2*held:6+2*held]
                    record['held_native_error_px']=error.tolist()
            records.append(record)
    file=output/'compact'/f'{name}.json';file.parent.mkdir(exist_ok=True)
    usable=[v for v in records if v.get('certified') and not v.get('ambiguous') and 'held_native_error_px' in v]
    summary={'slots':len(records),'retained_input_valid':sum(v['retained_input_valid'] for v in records),
             'certified':sum(v.get('certified',False) for v in records),'ambiguous':sum(v.get('ambiguous',False) for v in records),
             'at_bounds':sum(v.get('at_bounds',False) for v in records),'usable_scored':len(usable),
             'usable_native_coordinate_rms_px':float(np.sqrt(np.mean(np.asarray([v['held_native_error_px'] for v in usable])**2))) if usable else None}
    write_json(file,{'kind':'predeclared compact progress diagnostic; not G8','model_hash':array_hash({'scaled_globals':np.asarray(p)}),
                     'summary':summary,'slots':records})
    return summary


def main():
    import cupy as cp
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=ROOT/'experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g6_attempt_02')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    inputs,initial,record=check_parent(args.parent.resolve())
    population=inputs['population'];valid=inputs['valid'];e=inputs['exposure']
    spec=JointSpec(np.array(record['b1_reference_px']),np.array(record['b4_reference_px']),record['omega1_visual_deg'],record['omega4_visual_deg'],record['Aref_D'])
    p0=spec.pack(record);x0=np.column_stack((initial['theta_visual_deg'][valid],initial['accommodation_D'][valid]))
    y=relative_coordinates(population['p1'][valid],population['p4'][valid]);cov=inputs['covariance']['relative']
    targets=np.array([[population['target_theta_deg'][np.flatnonzero(population['exposure']==k)[0]],population['demand_diopters'][np.flatnonzero(population['exposure']==k)[0]]] for k in range(20)])
    schedule=np.concatenate([rows[np.linspace(0,len(rows)-1,5,dtype=int)] for rows in [np.flatnonzero(population['exposure']==k) for k in range(20)]])
    config=dict(CONFIG,P1_reference_radius_px=spec.r1,P4_reference_radius_px=spec.r4,
                template_tangent_basis=spec.template_basis.tolist(),global_lower=spec.bounds()[0].tolist(),global_upper=spec.bounds()[1].tolist(),
                compact_scheduled_indices=schedule.tolist())
    # JSON has no portable infinity values; center bounds are explicitly free.
    config['global_lower']=[v if np.isfinite(v) else None for v in config['global_lower']]
    config['global_upper']=[v if np.isfinite(v) else None for v in config['global_upper']]
    output=args.output.resolve();provenance=start_output(output,inputs,config,'G7',extra_sources=Path(__file__).parent.glob('*.py'))
    runtime=cp.cuda.runtime.getDeviceProperties(0);name=runtime['name'];name=name.decode() if isinstance(name,bytes) else str(name)
    provenance['runtime'].update(backend='CuPy float64 GPU; NumPy same-equation reference',gpu_name=name,
                                gpu_memory_setup_bytes=list(cp.cuda.runtime.memGetInfo()))
    write_json(output/'provenance.json',provenance)
    write_json(output/'summary.json',{'stage':'S7/G7','status':'RUNNING','G7_decision':'none','source_hash':provenance['source_hash']})
    np.savez_compressed(output/'compact_schedule.npz',scheduled_population_index=schedule,
                        exposure=population['exposure'][schedule],row=population['row'][schedule],source_frame=population['source_frame'][schedule])
    print('Auditing same-equation CPU/GPU derivatives on40 predeclared actual frames',flush=True)
    audit=empirical_audit(spec,p0,x0,y,e,cov,targets,name);write_json(output/'backend_audit.json',audit)
    objective=JointDM0(spec,y,e,cov,targets,xp=cp,chunk_size=CONFIG['chunk_size'])
    rng=np.random.default_rng(20261010)
    perturb_p=np.r_[rng.normal(0,.0002,2),rng.normal(0,.005),rng.normal(0,.005,3),rng.normal(0,.001,3),rng.normal(0,.001,10)]
    lower,upper=spec.bounds();perturbed_p=np.clip(p0+perturb_p,lower,upper)
    perturbed_x=np.clip(x0+rng.normal(size=x0.shape)*[.20,.10],[-20.,0.],[20.,6.])
    starts=[('common',x0,p0),('perturbed',perturbed_x,perturbed_p)]
    if CONFIG.get('compatible_warm_start_attempt'):
        warm=ROOT/CONFIG['compatible_warm_start_attempt']
        warm_checkpoint=json.loads((warm/'checkpoint.json').read_text())
        if warm_checkpoint['parent_checkpoint_sha256']!=provenance['parent_checkpoint_sha256']:
            raise ValueError('warm-start G6 reference/prior origin changed')
        warm_arrays=load_npz(warm/'fitted.npz')
        if not np.array_equal(warm_arrays['input_valid'],valid):raise ValueError('warm-start population changed')
        warm_schedule=load_npz(warm/'compact_schedule.npz')['scheduled_population_index']
        if not np.array_equal(warm_schedule,schedule):raise ValueError('warm-start compact schedule changed')
        warm_solution_path=warm/CONFIG['compatible_warm_start_selected']/'solution.npz'
        if sha256(warm_solution_path.read_bytes()).hexdigest()!=CONFIG['compatible_warm_start_solution_sha256']:
            raise ValueError('warm-start solution provenance changed')
        warm_solution=load_npz(warm_solution_path);wx,wp=warm_solution['states'],warm_solution['scaled_globals']
        if wx.shape!=x0.shape or wp.shape!=p0.shape or not np.isfinite(wx).all() or not np.isfinite(wp).all():
            raise ValueError('invalid compatible warm-start arrays')
        starts=[('continued',wx,wp)]
    outcomes=[]
    for label,sx,sp in starts:
        candidate=output/label;candidate.mkdir();begin=time.perf_counter()
        np.savez_compressed(candidate/'initial.npz',states=sx,scaled_globals=sp)
        def checkpoint(outer,x,p,current,norms):
            hostx,hostp=objective.host(x),objective.host(p)
            np.savez_compressed(candidate/f'outer_{outer+1:02d}.npz',states=hostx,scaled_globals=hostp)
            components={key:current[key] for key in ['cost','point','theta_anchor','A_anchor','regularization','temporal']}
            write_json(candidate/'latest.json',{'outer':outer+1,**components,**norms})
            print(label,'outer',outer+1,components,norms,flush=True)
            if outer+1 in CONFIG['compact_completed_outer_checkpoints']:
                compact=compact_check(output,f'{label}_outer_{outer+1:02d}',inputs,schedule,spec,hostp)
                print('compact progress',label,outer+1,compact,flush=True)
        x,p,final,certificate,history=fit_joint(objective,sx,sp,max_outer=CONFIG['max_outer'],joint_steps=CONFIG['joint_steps_per_outer'],
                                               polish_steps=CONFIG['observed_Newton_polish_steps'],checkpoint=checkpoint)
        hostx,hostp=objective.host(x),objective.host(p)
        write_json(candidate/'history.json',history)
        components={key:final[key] for key in ['cost','point','theta_anchor','A_anchor','regularization','temporal']}
        fit={'label':label,'components':components,'certificate':certificate,'elapsed_seconds':time.perf_counter()-begin,
             'state_change':{'theta_deg':describe(hostx[:,0]-x0[:,0]),'A_D':describe(hostx[:,1]-x0[:,1])},
             'scaled_globals':hostp.tolist(),'scaled_global_change':(hostp-p0).tolist(),
             'means':objective.host(final['means']).tolist(),'mean_deviations':objective.host(final['deviation']).tolist()}
        np.savez_compressed(candidate/'solution.npz',states=hostx,scaled_globals=hostp)
        write_json(candidate/'fit.json',fit);write_json(candidate/'model.json',model_record(record,spec,hostp,provenance))
        compact=compact_check(output,f'{label}_final',inputs,schedule,spec,hostp,certificate['fit_certified']);fit['compact_final']=compact
        write_json(candidate/'fit.json',fit);outcomes.append(fit)
        print(label,'final',components,certificate,flush=True)
    eligible=[i for i,v in enumerate(outcomes) if v['certificate']['fit_certified']]
    best=min(eligible or range(len(outcomes)),key=lambda i:outcomes[i]['components']['cost']);chosen=outcomes[best];label=chosen['label']
    solution=load_npz(output/label/'solution.npz');x,p=solution['states'],solution['scaled_globals'];snapshot=objective.snapshot(cp.asarray(x),cp.asarray(p))
    n=len(valid)
    def extend(value):
        a=np.full((n,)+value.shape[1:],np.nan);a[valid]=value;return a
    arrays={'input_valid':valid,'joint_state_evaluated':valid.copy(),'theta_visual_deg':extend(x[:,0]),'accommodation_D':extend(x[:,1]),
            'status':np.where(valid,'joint_fit_certified' if eligible else 'joint_fit_uncertified','input_unavailable')}
    arrays.update({key:extend(value) for key,value in snapshot.items()})
    arrays['relative_residual_px']=extend(y-snapshot['prediction']);np.savez_compressed(output/'fitted.npz',**arrays)
    write_json(output/'model.json',model_record(record,spec,p,provenance))
    conditions=[]
    for k in range(20):
        select=e==k;rows=np.flatnonzero(population['exposure']==k);fullrows=rows[valid[rows]];r=y[select]-snapshot['prediction'][select]
        conditions.append({'exposure':k,'capture':str(population['capture'][rows[0]]),'nominal_deg':float(targets[k,0]),'demand_D':float(targets[k,1]),
             'scheduled':len(rows),'valid':int(select.sum()),'unavailable':int((~valid[rows]).sum()),
             'theta_deg':describe(x[select,0]),'A_D':describe(x[select,1]),'g':describe(snapshot['g'][select]),
             'relative_coordinate_rms_px':float(np.sqrt(np.mean(r*r))),'signed_relative_means_px':r.mean(axis=0).tolist(),
             'state_bounds':np.sum((x[select]<=np.array([-20.,0.])+1e-9)|(x[select]>=np.array([20.,6.])-1e-9),axis=0).tolist(),
             'blocks':[{'block':j,'scheduled':len(part),'valid':int(valid[part].sum()),'unavailable':int((~valid[part]).sum()),
                        'relative_coordinate_rms_px':float(np.sqrt(np.mean(arrays['relative_residual_px'][part[valid[part]]]**2))) if valid[part].any() else None}
                       for j,part in enumerate(np.array_split(rows,5))]})
    write_json(output/'conditions.json',conditions)
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for cap in range(4):
        records=conditions[cap*5:(cap+1)*5];nom=[v['nominal_deg'] for v in records]
        axes[0].plot(nom,[v['theta_deg']['mean'] for v in records],'o-',label=f'capture{cap+1}')
        axes[1].plot(nom,[v['A_D']['mean'] for v in records],'o-')
        axes[2].plot(nom,[v['relative_coordinate_rms_px'] for v in records],'o-')
    axes[0].set(xlabel='Nominal gaze (degrees)',ylabel='Mean refined visual theta (degrees)');axes[0].legend()
    axes[1].set(xlabel='Nominal gaze (degrees)',ylabel='Mean refined A (D)')
    axes[2].set(xlabel='Nominal gaze (degrees)',ylabel='Native relative-coordinate RMS (px)')
    fig.tight_layout();fig.savefig(output/'states_and_residuals.png',dpi=150);plt.close(fig)
    params=spec.parameters(p);fig,axes=plt.subplots(1,2,figsize=(12,5))
    aa=np.linspace(0,6,101);axes[0].plot(aa,1+params.m1*(aa-spec.aref));axes[0].set(xlabel='A (D)',ylabel='M(A)')
    from distortion_model.optics import p4_reference
    for ac in [0.,2.,4.,6.]:
        tt=np.linspace(-20,20,201);f4,mu4,ok=p4_reference(tt,np.full_like(tt,ac),params)
        shape=f4-mu4[:,None,:];axes[1].plot(tt,np.sqrt(np.mean(shape**2,axis=(1,2))),label=f'A={ac:g} D')
    axes[1].set(xlabel='Common visual theta (degrees)',ylabel='Centered P4 coordinate RMS (reference px)');axes[1].legend()
    fig.tight_layout();fig.savefig(output/'response_curves.png',dpi=150);plt.close(fig)
    cpout={'schema':'g7_joint_checkpoint_v1','source_commit':provenance['source_commit'],'source_hash':provenance['source_hash'],
           'parent_checkpoint_path':str((inputs['parent']/'checkpoint.json').relative_to(ROOT)),
           'parent_checkpoint_sha256':provenance['parent_checkpoint_sha256'],'parent_summary_sha256':provenance['parent_summary_sha256'],
           'base_g3_checkpoint_path':str(inputs['basepath'].relative_to(ROOT)),
           'base_g3_checkpoint_sha256':sha256(inputs['basepath'].read_bytes()).hexdigest(),
           'model_file':'model.json','model_sha256':sha256((output/'model.json').read_bytes()).hexdigest(),
           'frame_states_file':'fitted.npz','frame_states_hash':array_hash(arrays),'selected_start':label,
           'fit_complete':bool(eligible),'fit_certified':bool(eligible),'crosscheck_complete':False,'comparison_complete':False}
    if CONFIG.get('compatible_warm_start_attempt'):
        cpout.update(optimization_parent_attempt=CONFIG['compatible_warm_start_attempt'],
                     optimization_parent_solution_sha256=CONFIG['compatible_warm_start_solution_sha256'],
                     same_state_reassessment_summary_sha256=CONFIG['same_state_reassessment_summary_sha256'])
    write_json(output/'checkpoint.json',cpout)
    summary={'stage':'S7/G7','status':'COMPLETE_PENDING_REVIEW','G7_decision':'none','source_hash':provenance['source_hash'],
             'scheduled':n,'complete_valid':int(valid.sum()),'unavailable':int((~valid).sum()),'joint_state_evaluated':len(x),
             'starts_completed':len(outcomes),'certified_start_count':len(eligible),'selected_start':label,
             'components':chosen['components'],'certificate':chosen['certificate'],'all_start_outcomes':outcomes,
             'equal_exposure_native_relative_rms_px':float(np.sqrt(np.mean([v['relative_coordinate_rms_px']**2 for v in conditions]))),
             'M_slope_per_D':params.m1,'theta_visual_deg':describe(x[:,0]),'A_D':describe(x[:,1]),'g':describe(snapshot['g']),
             'GPU_memory_final_bytes':list(cp.cuda.runtime.memGetInfo()),'GPU_pool_used_bytes':cp.get_default_memory_pool().used_bytes(),
             'CPU_max_RSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'elapsed_seconds':time.perf_counter()-started,
             'objective_evaluations':objective.evaluations,'fit_complete':bool(eligible),'fit_certified':bool(eligible),
             'crosscheck_complete':False,'comparison_complete':False,'automated_tests_run':False}
    write_json(output/'summary.json',summary);(output/'STAGE_REPORT.md').write_text('# G7 pending root review\n')
    (output/'PROGRESS.md').write_text('One next action: review G7 certificate and empirical limitations.\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
