#!/usr/bin/env python3
"""Conditional size profile: refit all nuisance globals/states at fixed size."""
import sys,argparse,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import FAMILY,load_stage,result_dir,write_json,save_csv,read_json,hashfile,source_manifest,verify_relocated_hashes
from model import FreeSizeProblem,CASES,FIXED_CASES
from run import subset,checked_stage,inputs,compact
from distortion_model.capturewise_reverse import host

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--family',type=Path,default=FAMILY);ap.add_argument('--backend',choices=['numpy','cupy'],default='cupy');ap.add_argument('--case',choices=list(CASES),default='free_size_sigma050_optical2');ap.add_argument('--maxiter',type=int,default=150);ap.add_argument('--state-maxiter',type=int,default=400);args=ap.parse_args();xp=np
    if args.backend=='cupy':import cupy as xp
    d=load_stage(1,family=args.family);p1=load_stage(2,family=args.family);barrel=checked_stage(3,args,d,p1);p4=checked_stage(4,args,d,p1);z=load_stage(5,branch=args.case,family=args.family);mask=np.asarray(d['fit_mask'],bool);path=result_dir(5,branch=args.case,family=args.family);summ=read_json(path/'summary.json')
    verify_relocated_hashes(summ['input_artifact_hashes'],inputs(args.family,[1,2,3,4]),'Profile frozen inputs')
    if summ['arrays_sha256']!=hashfile(path/'arrays.npz') or not np.array_equal(z['g'],p1['primary_g']) or not np.array_equal(z['fit_mask'],mask):raise ValueError('Profile frozenfit output hash/cohort mismatch')
    control=load_stage(5,branch=FIXED_CASES[args.case],family=args.family);controlpath=result_dir(5,branch=FIXED_CASES[args.case],family=args.family);controlsummary=read_json(controlpath/'summary.json');controlfit=next(t for t in controlsummary['starts'] if t['start']==controlsummary['selected_start'])
    if controlsummary['arrays_sha256']!=hashfile(controlpath/'arrays.npz') or not np.array_equal(control['g'],z['g']) or not np.array_equal(control['fit_mask'],mask):raise ValueError('Profile immutable fixedcontrol hash/cohort mismatch')
    out=args.family/'validation'/'template_size_profile'/args.case
    if (out/'summary.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);opt=float(np.exp(z['q'][17]));points=[('lower_0p5percent',opt*.995),('optimum',opt),('upper_0p5percent',opt*1.005),('initial_size_one',1.)];rows=[];fits=[];arrays={};start=time.monotonic()
    for label,size in points:
        if not .5<=size<=2:raise ValueError('Requested local size profile beyond declared numerical bounds; no silent clipping')
        p=FreeSizeProblem(subset(d,mask),p1,p4,barrel,xp=xp,g_override=z['g'][mask],**CASES[args.case]);p.bounds[17]=np.log(size);q=z['q'].copy();q[17]=np.log(size)
        initial_theta=z['theta'][mask];initial_A=z['A'][mask]
        if label=='initial_size_one':q=np.r_[control['q'],0.];initial_theta=control['theta'][mask];initial_A=control['A'][mask]
        seed=p.fit_states(initial_theta,initial_A,q,args.state_maxiter,log=True);fit=p.fit_joint(seed,args.maxiter,args.state_maxiter,log=True);ev=p.evaluate(xp.asarray(fit['theta']),xp.asarray(fit['A']),fit['q'],True,True)
        cert=fit['certificate'];native_size_gradient=float(ev['global_gradient'][17]);scaled_size_gradient=native_size_gradient*p.scales[17]
        means=host(p.means(xp.asarray(fit['A'])),xp);spans=np.array([np.ptp(means[p.labels//5==c]) for c in range(4)])
        row=dict(immutable_fixed_control_loss=float(controlfit['loss']),loss_minus_immutable_fixed_control=fit['loss']-controlfit['loss'] if label=='initial_size_one' else None,point=label,size_ratio=size,log_size=np.log(size),loss=fit['loss'],unconstrained_size_gradient=native_size_gradient,unconstrained_scaled_size_gradient=scaled_size_gradient,state_stationary=cert['state_stationary'],nuisance_global_stationary=cert['global_stationary'],invalid_rows=cert['invalid_rows'],slope=fit['q'][13],offset_deg=fit['q'][12],**{f'loss_{k}':v for k,v in ev['components'].items()},**{f'C{c+1}_Amean_span_D':spans[c] for c in range(4)})
        if label=='initial_size_one':
            row['immutable_fixed_control_loss']=float(controlfit['loss']);row['loss_minus_immutable_fixed_control']=fit['loss']-controlfit['loss']
            if row['loss_minus_immutable_fixed_control']>1e-8:raise RuntimeError('Sizeone nuisance profile worse than exact immutable fixedcontrol initializer')
        rows.append(row);fits.append(dict(point=label,fit=compact(fit),conditional_size_gradient=native_size_gradient,loss_components=ev['components'],capture_Amean_D=host(p.capture_means(xp.asarray(fit['A'])),xp),fixation_Amean_D=means,template_centered_energy_px2=float(np.sum((host(p.template_geometry(fit['q']),xp)-p.B4_centroid)**2))))
        arrays[label+'_q']=fit['q'];arrays[label+'_theta']=fit['theta'];arrays[label+'_A']=fit['A'];print(row,flush=True)
    np.savez_compressed(out/'arrays.npz',**arrays);gate=all(r['state_stationary'] and r['nuisance_global_stationary'] and r['invalid_rows']==0 for r in rows)
    write_json(out/'summary.json',dict(stage_gate='Conditional nuisance-refit size profile',profile_gate=gate,case=args.case,optimum_size_ratio=opt,points=rows,fits=fits,runtime_seconds=time.monotonic()-start,input_artifact_hashes={**inputs(args.family,[1,2,3,4]),str(path/'arrays.npz'):hashfile(path/'arrays.npz'),str(controlpath/'arrays.npz'):hashfile(controlpath/'arrays.npz')},source_hashes=source_manifest(),arrays_sha256=hashfile(out/'arrays.npz'),certificate_scope='Size coordinate held fixed by equal lower/upper bounds; only free nuisance global and state gradients certify. Unconstrained size gradient separately reported, not required zero except free optimum.',interpretation='Conditional anchored objective profile, not data-only likelihood or confidence interval. Components separate geometry fit from A/gaze/optical penalties. Shared g, centroid, Aref and Stage1..4 priors unchanged.'))
    save_csv(out/'profile.csv',rows)
    if not gate:raise RuntimeError('Uncertified profile points saved with gate false; no interpretation as minimized profile')
if __name__=='__main__':main()
