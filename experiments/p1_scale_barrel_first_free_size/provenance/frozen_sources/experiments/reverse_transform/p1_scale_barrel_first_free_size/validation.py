#!/usr/bin/env python3
"""Train-only sensitivity calibration and retained-two held-vertex scoring."""
import sys,argparse,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import FAMILY,load_stage,result_dir,geometry_metrics,read_json,write_json,save_csv,hashfile,source_manifest
from run import subset,checked_stage,inputs
from model import CASES,FreeSizeProblem
from distortion_model.capturewise_keystone_first import forward
from distortion_model.capturewise_reverse import forward as p1_forward,host


def validate(args,case):
    xp=np
    if args.backend=='cupy':import cupy as xp
    started=time.monotonic();trainfamily=args.family/'validation_temporal';d=load_stage(1,family=trainfamily);p1=load_stage(2,family=trainfamily)
    from types import SimpleNamespace
    checkargs=SimpleNamespace(family=trainfamily,reference_width_px=None)
    barrel=checked_stage(3,checkargs,d,p1);p4=checked_stage(4,checkargs,d,p1);p5=load_stage(5,branch=case,family=trainfamily)
    p5path=result_dir(5,branch=case,family=trainfamily);p5summary=read_json(p5path/'summary.json')
    if p5summary['input_artifact_hashes']!=inputs(trainfamily,[1,2,3,4]) or p5summary['arrays_sha256']!=hashfile(p5path/'arrays.npz') or not np.array_equal(p5['g'],p1['primary_g']):raise ValueError('FreshStage05 training dependency/output hash or sharedg mismatch')
    train=np.asarray(d['fit_mask'],bool);test=np.asarray(d['test_mask'],bool);dt=subset(d,test);anchor=np.empty(int(test.sum()))
    if not np.array_equal(barrel['fit_mask'],train) or not np.array_equal(p4['fit_mask'],train) or not np.array_equal(p5['fit_mask'],train):raise ValueError('Exact training masks required across freshStages3..5')
    for group in np.unique(dt['group']):
        m=train&(d['group']==group)
        if not m.any():raise ValueError('Matching training fixation required for independent testthetaanchor')
        anchor[dt['group']==group]=d['theta0'][m].mean()
    pars=p1['primary_params'];fe=p1_forward(p1['primary_B1'],anchor,pars[:3],pars[3],float(p1['frozen_R01_px']),1.,xp=xp)
    if not host(fe['valid'],xp).all():raise ValueError('P1 heldscale profiling outside domain')
    H=host(fe['H'],xp);g=np.clip(np.sum(H*dt['X1'],axis=(1,2))/np.sum(H*H,axis=(1,2)),.98,1.02)
    out=result_dir(5,family=args.family)/'validation'/'temporal'/case
    if (out/'summary.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True);saved=dict(capture=dt['capture'],group=dt['group'],source_frame=dt['source_frame'],source_row=dt['source_row'],theta_anchor=anchor,g=g,Amean_targets=p5['Amean_targets'],targets=p5['Amean_targets']);allerrors=dict(initial=[],fixed=[],joint=[]);audits=[];rows=[]
    c=dt['capture'].astype(int)-1;fix=dt['fixation'].astype(int)
    for held in range(3):
        retained=[v for v in range(3) if v!=held];problem=FreeSizeProblem(dt,p1,p4,barrel,xp=xp,g_override=g,theta_anchor_override=anchor,vertices4=retained,**CASES[case])
        observed=dt['P4_raw'][:,held]-dt['P4_raw'][:,retained].mean(axis=1)
        initial=forward(p4['B4_ref'],anchor,p4['alpha'][c],float(p4['offset']),p4['R4'][c],g,barrel['kappa_targets'][c,fix],xp=xp);Z=host(initial['Z'],xp);initprediction=g[:,None]*(Z[:,held]-Z[:,retained].mean(axis=1));initialerror=observed-initprediction;allerrors['initial'].append(initialerror);saved[f'initial_residual_vertex{held}']=initialerror
        for label,q in [('fixed',p5['fixed_q']),('joint',p5['q'])]:
            fit=problem.fit_states(q=q,maxiter=args.maxiter);prediction=problem.predict_held(held,fit['theta'],fit['A'],q);error=observed-prediction;allerrors[label].append(error);saved[f'{label}_theta_vertex{held}']=fit['theta'];saved[f'{label}_A_vertex{held}']=fit['A'];saved[f'{label}_residual_vertex{held}']=error;audits.append(dict(held_vertex=held,model=label,loss=fit['loss'],certificate=fit['certificate']))
            print(f'Freshheld vertex{held} {label}: pooledRMS={np.sqrt(np.mean(np.sum(error**2,axis=1))):.8g}',flush=True)
        for label in allerrors:
            error=allerrors[label][-1]
            for group in np.unique(dt['group']):
                m=dt['group']==group;rows.append(dict(model=label,held_vertex=held,capture=int(dt['capture'][m][0]),fixation=int(dt['fixation'][m][0]),rows=int(m.sum()),relative_vertex_rms_px=float(np.sqrt(np.mean(np.sum(error[m]**2,axis=1))))))
    metrics={label:geometry_metrics(np.stack(e,axis=1),dt['group']) for label,e in allerrors.items()};np.savez_compressed(out/'arrays.npz',**saved)
    hashes={str(result_dir(i,branch=case if i==5 else 'primary',family=trainfamily)/'arrays.npz'):hashfile(result_dir(i,branch=case if i==5 else 'primary',family=trainfamily)/'arrays.npz') for i in range(1,6)}
    summary=dict(case=case,settings=CASES[case],stage_gate='retained-two held-vertex conditional validation',state_inference_gate=all(z['certificate']['state_stationary'] and not z['certificate']['invalid_rows'] for z in audits),training_rows=int(train.sum()),gap_rows=int(d['gap_mask'].sum()),test_rows=int(test.sum()),held_vertex_scores=int(3*test.sum()),metrics=metrics,state_audits=audits,training_artifact_hashes=hashes,source_hashes=source_manifest(),implementation_sha256=hashfile(Path(__file__)),arrays_sha256=hashfile(out/'arrays.npz'),runtime_seconds=time.monotonic()-started,learned_template_frozen_at_inference=True,fixed_control='Fresh Stage04/s0 fixed optical q with this scenario A-mean state anchor; changing optical width alone leaves fixed inference unchanged',protocol=dict(calibration='Immutable train-only Stage03barrel andStage04keystone; sensitivity Stage05 fit trainonly with fresh calibrated priors. Paired fixed-size train-fit initialization only, never prior.',theta_test_anchor='Trainingmean initialgaze inmatchingcapture/fixation, excludesalltestP4',scale='Bounded[.98,1.02] P1-onlyprofile attraining-derivedthetaanchor, samephysicalg forP1/P4, frozenforstateinference',P4_state_input='Onlyretainedtwo rawP4vertices centered ontheircentroid',scoring='Omittedrawvertex minusretainedrawcentroid vsrawmodelomitted minusmodelretainedcentroid',composition='D(K(B)); barrelinverseFIRST',targets=p5summary['A_anchor_policy']+' with targets frozen only from training Stage03; widths '+str(CASES[case]),limitations='Same-recording conditionalheldgeometry; reviewedeligibility/capture/fixationknown; noindependentphysicalgaze/Atruth'))
    write_json(out/'summary.json',summary);save_csv(out/'fixation_metrics.csv',rows)
    if not summary['state_inference_gate']:raise RuntimeError('Heldstateinferencefailedcertificate; reportedcomplete-rosterscoresnotpromoted')
    print(dict(output=str(out),metrics=metrics,state_inference_gate=True),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--family',type=Path,default=FAMILY);ap.add_argument('--backend',choices=['numpy','cupy'],default='cupy');ap.add_argument('--maxiter',type=int,default=400);ap.add_argument('--case',choices=['all',*CASES],default='all');args=ap.parse_args()
    for case in CASES if args.case=='all' else [args.case]:validate(args,case)

if __name__=='__main__':main()
