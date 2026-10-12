#!/usr/bin/env python3
"""Controlled common-template-size fit; immutable paired calibration and priors."""
import sys,argparse,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import FAMILY,load_stage,save_stage,result_dir,geometry_metrics,hashfile,read_json,write_json,save_csv,safe,verify_relocated_hashes
from model import CASES,FIXED_CASES,FreeSizeProblem
from distortion_model.capturewise_keystone_first import forward,inverse
from distortion_model.capturewise_reverse import host,inverse as p1_inverse
from distortion_model.capturewise_states import A_REF

def subset(d,mask):return {k:v[mask] if isinstance(v,np.ndarray) and v.ndim and len(v)==len(mask) else v for k,v in d.items()}
def compact(f):return {k:v for k,v in f.items() if k not in ('theta','A')}
def inputs(family,stages):return {str(result_dir(i,family=family)/'arrays.npz'):hashfile(result_dir(i,family=family)/'arrays.npz') for i in stages}
def training_only(d):return np.asarray(d['fit_mask'],bool)
def checked_stage(stage,args,d,p1):
    path=result_dir(stage,family=args.family);summary=read_json(path/'summary.json');arrays=load_stage(stage,family=args.family)
    # Exact copied artifacts retain original provenance paths. Compare stage SHA values.
    expected=inputs(args.family,[1,2] if stage==3 else [1,2,3])
    verify_relocated_hashes(summary['input_artifact_hashes'],expected,f'Stage{stage} frozen inputs')
    if summary['arrays_sha256']!=hashfile(path/'arrays.npz'):raise ValueError(f'Stage{stage} output hash differs')
    if not np.array_equal(arrays['fit_mask'],training_only(d)) or not np.array_equal(arrays['g'],p1['primary_g']):raise ValueError('Frozen calibration mask/sharedg differs')
    return arrays

def initializers(args,d,p1,p4,case):
    branch=FIXED_CASES[case];baseline=load_stage(5,branch=branch,family=args.family);path=result_dir(5,branch=branch,family=args.family)
    summary=read_json(path/'summary.json')
    if summary['arrays_sha256']!=hashfile(path/'arrays.npz'):raise ValueError('Immutable control output hash differs')
    verify_relocated_hashes(summary['input_artifact_hashes'],inputs(args.family,[1,2,3,4]),'Immutable fixed-size control calibration')
    if not np.array_equal(baseline['g'],p1['primary_g']) or not np.array_equal(baseline['fit_mask'],d['fit_mask']):raise ValueError('Immutable control cohort/sharedg mismatch')
    if not np.array_equal(baseline['initial_B4'],p4['B4_ref']) or not np.array_equal(baseline['R4'],p4['R4']):raise ValueError('Immutable control gauge/radius differs')
    return baseline,{str(path/'arrays.npz'):hashfile(path/'arrays.npz')}

def make_proposals(problem,baseline,mask,starts,seed):
    xp=problem.xp;theta=np.asarray(baseline['theta'])[mask];A=np.asarray(baseline['A'])[mask];q=np.r_[baseline['q'],0.]
    loss=problem.evaluate(xp.asarray(theta),xp.asarray(A),q,optical=True)['loss']
    if not np.isfinite(loss):raise ValueError('Exact fixedsize control initializer invalid')
    proposals=[dict(q=q,theta=theta,A=A,metadata=dict(label='Exact paired fixed-size optimum at logsize zero',starting_objective=loss))];rng=np.random.default_rng(seed)
    while len(proposals)<starts:
        for attempt in range(50):
            proposal=q.copy();proposal[:12]+=rng.normal(size=12)*.005;proposal[12]+=rng.choice([-5.,5.]);proposal[13]*=rng.uniform(.9,1.1);proposal[14:17]+=rng.normal(size=3)*.02;proposal[17]=np.log(1.02);proposal=np.clip(proposal,problem.bounds[:,0],problem.bounds[:,1]);value=problem.evaluate(xp.asarray(theta),xp.asarray(A),proposal,optical=True)['loss']
            if np.isfinite(value):break
        else:raise RuntimeError('No feasible distinct size/optical start')
        proposals.append(dict(q=proposal,theta=theta.copy(),A=A.copy(),metadata=dict(label='Independent size/optical perturbation',starting_objective=value,attempt=attempt)))
    return proposals

def stage5(args,d,p1,barrel,p4,xp,case):
    out=result_dir(5,branch=case,family=args.family)
    if (out/'summary.json').exists():raise FileExistsError(out)
    started=time.monotonic();mask=training_only(d);g=np.asarray(p1['primary_g']);training=subset(d,mask);settings=CASES[case];problem=FreeSizeProblem(training,p1,p4,barrel,xp=xp,g_override=g[mask],**settings);fixed=problem.fit_states(maxiter=args.state_maxiter,log=True)
    baseline,initializer_hashes=initializers(args,d,p1,p4,case);proposals=make_proposals(problem,baseline,mask,args.starts,args.seed+20)
    starts=[]
    for index,proposal in enumerate(proposals):
        q=proposal['q'];print(f'Sensitivity {case} start{index} rows{mask.sum()} initializer={proposal["metadata"]["label"]}',flush=True)
        seed=problem.fit_states(proposal['theta'],proposal['A'],q,args.state_maxiter,log=True)
        fitted=problem.fit_joint(seed,args.maxiter,args.state_maxiter,log=True)
        fitted.update(start=index,initial_q=q,initial_capture_Amean_D=host(problem.capture_means(xp.asarray(proposal['A'])),xp),initializer=proposal['metadata']);starts.append(fitted)
    certified=[z for z in starts if np.isfinite(z['loss']) and z['certificate']['state_stationary'] and z['certificate']['global_stationary'] and not z['certificate']['invalid_rows']]
    if not certified:
        write_json(out/'CHECKPOINT.json',dict(stage_gate='uncertified freshjoint checkpoint',starts=[compact(z) for z in starts]));raise RuntimeError('No certified freshjoint start; no completed summary promoted')
    best=min(certified,key=lambda z:z['loss']);theta=np.empty(len(g));A=np.empty(len(g));theta[mask]=best['theta'];A[mask]=best['A'];complement=None
    if (~mask).any():
        held=FreeSizeProblem(subset(d,~mask),p1,p4,barrel,xp=xp,g_override=g[~mask],**settings);hh=held.fit_states(q=best['q'],maxiter=args.state_maxiter,log=True);theta[~mask]=hh['theta'];A[~mask]=hh['A'];complement=hh['certificate']
    full=FreeSizeProblem(d,p1,p4,barrel,xp=xp,g_override=g,**settings);ev=full.evaluate(xp.asarray(theta),xp.asarray(A),best['q'],True,True);tev=problem.evaluate(xp.asarray(best['theta']),xp.asarray(best['A']),best['q'],True,True);B=host(full.template_geometry(best['q']),xp);alpha,offset,s=full.unpack(best['q']);alpha=host(alpha,xp);offset=float(host(offset,xp));s=float(host(s,xp));k=s*(A-A_REF);c=d['capture'].astype(int)-1
    if not host(ev['valid'],xp).all():raise RuntimeError('Final freshjoint forward invalid; no completed promotion')
    iv=inverse(B,d['X4'],theta,alpha,offset,p4['R4'][c],g,k);i1=p1_inverse(p1['primary_B1'],d['X1'],theta,p1['primary_params'][:3],p1['primary_params'][3],float(p1['frozen_R01_px']),g)
    if not iv['valid'].all() or not i1['valid'].all():raise RuntimeError('Final freshjoint inverse invalid; no completed promotion')
    pred1=host(ev['p1']['prediction'],xp);pred4=host(ev['p4']['prediction'],xp);sens=full.sensitivities(theta,A,best['q']);qstack=np.stack([z['q'] for z in certified]);shapes=np.stack([host(problem.template_geometry(z['q']),xp) for z in certified]);capturemeans=np.array([host(problem.capture_means(xp.asarray(z['A'])),xp) for z in certified]);rank=problem.data_only_rank(best['theta'],best['A'],best['q'])
    hashes=inputs(args.family,[1,2,3,4]);gauge=dict(initial_centered_energy_px2=problem.B4_energy,learned_centered_energy_px2=float(np.sum((B-B.mean(axis=0))**2)),initial_centroid_px=problem.B4_centroid,learned_centroid_px=B.mean(axis=0),centroid_max_abs_error_px=float(np.max(abs(B.mean(axis=0)-problem.B4_centroid))),chart_coordinates=best['q'][14:17],chart_bound_contacts=best['certificate']['global_bound_contacts'][14:17],frozen_capture_radii_px=p4['R4'],template_prior='none; shared centered size free, centroid fixed',template_size_ratio=float(np.exp(best['q'][17])),log_template_size=float(best['q'][17]),numerical_size_ratio_bounds=[.5,2.],size_bound_contact=best['certificate']['global_bound_contacts'][17])
    arrays=dict(template_size_ratio=np.asarray(np.exp(best['q'][17])),log_template_size=np.asarray(best['q'][17]),size_ratio_bounds=np.array([.5,2.]),initial_B4=p4['B4_ref'],learned_B4=B,B4_ref=B,R4=p4['R4'],q=best['q'],fixed_q=problem.q0,targets=problem.fixation_targets,targets4=problem.targets_capture,Amean_targets=problem.fixation_targets,theta=theta,A=A,g=g,fit_mask=mask,p1_prediction=pred1,p4_prediction=pred4,predicted1=pred1,predicted4=pred4,recovered1=i1['recovered'],recovered4=iv['recovered'],valid1=i1['valid'],valid4=iv['valid'],kappa=k,alpha=alpha,offset=np.asarray(offset),slope=np.asarray(s),s0=np.asarray(problem.s0),kappa_targets=barrel['kappa_targets'],sigma_A=np.asarray(settings['sigma_A']),optical_width=np.asarray(settings['optical_width']),anchor_mode=np.asarray('fixation'),capture_Amean_targets=problem.fixation_targets.mean(axis=1),**sens)
    summary=dict(stage_gate='stationary barrel-first anchor sensitivity conditional estimator',case=case,settings=settings,initializer_artifact_hashes=initializer_hashes,baseline_comparison=read_json(result_dir(5,branch=FIXED_CASES[case],family=args.family)/'summary.json')['all_forward4'],backend=args.backend,rows=len(g),training_rows=int(mask.sum()),runtime_seconds=time.monotonic()-started,selected_start=best['start'],selected_certificate=best['certificate'],certified_starts=len(certified),selection_policy='Lowest loss among valid starts meeting both unchanged state/global stationarity certificates; uncertified lower finite starts not promoted',lowest_all_start_loss=min(z['loss'] for z in starts),starts=[compact(z) for z in starts],fixed_state_certificate=fixed['certificate'],complement_state_certificate=complement,training_loss_components=tev['components'],all_row_loss_components=ev['components'],all_forward1=geometry_metrics(pred1-d['X1'],d['group']),all_forward4=geometry_metrics(pred4-d['X4'],d['group']),all_inverse1=geometry_metrics(i1['recovered']-p1['primary_B1'],d['group'],i1['valid']),all_inverse4=geometry_metrics(iv['recovered']-B,d['group'],iv['valid']),all_forward_invalid_rows=int((~host(ev['valid'],xp)).sum()),all_inverse4_invalid_rows=int((~iv['valid']).sum()),template_gauge=gauge,data_only_rank=rank,multistart_spread=dict(template_size_ratio_min=float(np.exp(qstack[:,17]).min()),template_size_ratio_max=float(np.exp(qstack[:,17]).max()),certified_loss_min=min(z['loss'] for z in certified),certified_loss_max=max(z['loss'] for z in certified),native_q_min=qstack.min(axis=0),native_q_max=qstack.max(axis=0),template_vertex_max_abs_spread_px=float(np.max(shapes.max(axis=0)-shapes.min(axis=0))),capture_Amean_min_D=capturemeans.min(axis=0),capture_Amean_max_D=capturemeans.max(axis=0)),fresh_Amean_targets=problem.fixation_targets,capture_Amean_targets=problem.fixation_targets.mean(axis=1),A_anchor_policy='Equal20 fixationmean standardized-square average',fresh_s0=problem.s0,input_artifact_hashes=hashes,order='forward D(K(B)); inverse barrel FIRST then K',optical_anchor_policy=f'FreshStage04 alpha/offset andStage03s0 only; mean14 standardized squares with widths {settings["optical_width"]} times [.05,10deg,abs(s0)]; template shape and size no prior. Numerical size ratio bounds [.5,2] do not assert physical size constraints. Individual Gaussian interpretation includes sqrt14 averaging factor. Raw coefficients same model/gauge; Original17 characteristic gradient certificate units unchanged; appended logsize characteristic unit .01.',interpretation='One frozen sharedphysical P1/P4 g. All calibration/prior targets exclusively active barrel-first Stages1..4. Paired immutable fixed-size control used only as initial point. Exactly one shared template size DOF added, no size prior or second g. Data-only slope/A ambiguity and capture/demand confounding remain.')
    save_stage(5,arrays,summary,dict(settings=dict(CASES[case],cli=vars(args)),inputs='Immutable active Stage1..4 exact copies; paired fixed-size Stage05 fit initialization only',state_anchors=summary['A_anchor_policy']+'; gaze .5deg unchanged; no mean-k extra penalty',optical_anchors=summary['optical_anchor_policy'],training='Onlyfit_mask optimizesP4globals/states; complement separately inferred withfrozenq',input_artifact_hashes=hashes),branch=case,family=args.family)
    rows=[]
    for cc in range(4):
        for ff in range(5):
            m=(c==cc)&(d['fixation']==ff);tm=m&mask;unit=B/np.linalg.norm(B,axis=1)[:,None];res=iv['recovered'][m]-B;rad=np.sum(res*unit,axis=2);tan=np.sum(res*np.stack((-unit[:,1],unit[:,0]),axis=1),axis=2)
            rows.append(dict(capture=cc+1,fixation=ff,rows=int(m.sum()),training_rows=int(tm.sum()),theta_mean=float(theta[m].mean()),A_mean=float(A[m].mean()),training_A_mean=float(A[tm].mean()),A_std=float(A[m].std()),Amean_target=float(problem.fixation_targets[cc,ff]),Amean_target_active_penalty=True,capture_Amean_target=float(problem.fixation_targets.mean(axis=1)[cc]),kappa_mean=float(k[m].mean()),kappa_target=float(barrel['kappa_targets'][cc,ff]),P1_forward_vertex_rms_px=float(np.sqrt(np.mean(np.sum((pred1-d['X1'])[m]**2,axis=2)))),P4_forward_vertex_rms_px=float(np.sqrt(np.mean(np.sum((pred4-d['X4'])[m]**2,axis=2)))),P4_inverse_vertex_rms_px=float(np.sqrt(np.mean(np.sum(res**2,axis=2)))),P4_inverse_radial_rms_px=float(np.sqrt(np.mean(rad**2))),P4_inverse_tangential_rms_px=float(np.sqrt(np.mean(tan**2))),deltaA_equivalent_rms_D=float(np.sqrt(np.nanmean(sens['deltaA_equivalent'][m]**2)))))
    summary['training_fixation_A_spans_D']=[float(np.ptp([r['training_A_mean'] for r in rows if r['capture']==cc])) for cc in range(1,5)]
    # Add descriptive cell means after saving, without altering fitted arrays.
    saved_summary=read_json(out/'summary.json');saved_summary['training_fixation_A_spans_D']=summary['training_fixation_A_spans_D'];write_json(out/'summary.json',saved_summary)
    save_csv(out/'fixation_metrics.csv',rows);print(safe(dict(stage=5,output=out,loss=best['loss'],certificate=best['certificate'])),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--family',type=Path,default=FAMILY);ap.add_argument('--backend',choices=['numpy','cupy'],default='cupy');ap.add_argument('--case',choices=['all',*CASES],default='all');ap.add_argument('--starts',type=int,default=2);ap.add_argument('--maxiter',type=int,default=150);ap.add_argument('--state-maxiter',type=int,default=400);ap.add_argument('--seed',type=int,default=20261027);args=ap.parse_args()
    if args.starts<2:raise ValueError('At least two distinct initial points required')
    xp=np
    if args.backend=='cupy':import cupy as xp
    d=load_stage(1,family=args.family);p1=load_stage(2,family=args.family);barrel=checked_stage(3,args,d,p1);p4=checked_stage(4,args,d,p1)
    for case in CASES if args.case=='all' else [args.case]:stage5(args,d,p1,barrel,p4,xp,case)
if __name__=='__main__':main()
