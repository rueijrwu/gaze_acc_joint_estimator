"""Actual-record G8 inference preflight and independent saved-result audit."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).parent))
from run import ROOT,STAGE,load_frozen,digest,write
from distortion_model.crosscheck import infer_omission,frame_metrics,equal_exposure
from distortion_model.geometry import relative_coordinates,retained_indices
from distortion_model.inference import infer_retained,infer_retained_legacy,infer_all
from distortion_model.optics import predict_relative
from distortion_model.data import array_hash


def preflight(parent,output):
    output.mkdir(parents=True,exist_ok=False)
    inputs,spec,p,calibration,_=load_frozen(parent);pop=inputs['population'];r=inputs['covariance']['relative']
    rows=np.concatenate([np.flatnonzero((pop['exposure']==k)&pop['complete_valid'])[:2] for k in range(20)])
    import cupy as cp
    records=[]
    for held in range(3):
        idx=retained_indices(held);y=relative_coordinates(pop['p1'][rows],pop['p4'][rows],held_point=held);marginal=r[np.ix_(idx,idx)]
        started=time.perf_counter();legacy=infer_retained_legacy(spec,p,y,marginal,held,xp=cp);legacy_seconds=time.perf_counter()-started
        started=time.perf_counter();new=infer_retained(spec,p,y,marginal,held,xp=cp);new_seconds=time.perf_counter()-started
        oldvalid=legacy['certified']&(~legacy['ambiguous']);newvalid=new['certified']&(~new['ambiguous']);common=oldvalid&newvalid
        state_difference=np.max(np.abs(legacy['states'][common]-new['states'][common]),axis=0)
        cost_difference=float(np.max(np.abs(legacy['cost'][common]-new['cost'][common])))
        assert np.array_equal(oldvalid,newvalid),'legacy/new numerical coverage mismatch requires review'
        assert np.max(state_difference)<2e-5 and cost_difference<1e-7,'legacy/new selected branch mismatch'
        cpu=infer_retained(spec,p,y[::2],marginal,held,xp=np)
        cpuvalid=cpu['certified']&(~cpu['ambiguous']);gpuvalid=newvalid[::2];both=cpuvalid&gpuvalid
        assert np.array_equal(cpuvalid,gpuvalid),'CPU/GPU numerical coverage mismatch'
        backend_difference=np.max(np.abs(cpu['states'][both]-new['states'][::2][both]),axis=0)
        assert np.max(backend_difference)<2e-5
        baseline=infer_omission(spec,p,pop['p1'][rows],pop['p4'][rows],pop['p1_available'][rows],pop['p4_available'][rows],r,held,xp=cp)
        masking=[]
        for mode in ('huge_finite','unavailable_nan'):
            changed=pop['p4'][rows].copy();flags=pop['p4_available'][rows].copy()
            if mode=='huge_finite':changed[:,held,:]=changed[:,held,:]+np.array([1e9,-2e9])
            else:changed[:,held,:]=np.nan;flags[:,held]=False
            perturbed=infer_omission(spec,p,pop['p1'][rows],changed,pop['p1_available'][rows],flags,r,held,xp=cp)
            assert np.array_equal(baseline['retained_input_valid'],perturbed['retained_input_valid'])
            checks={key:bool(np.array_equal(baseline['solution'][key],perturbed['solution'][key],equal_nan=True))
                    for key in ('states','cost','projected_gradient_inf','certified','ambiguous','rank2','at_bounds',
                                'candidate_certified_count','selected_start','gn_information')}
            assert all(checks.values()),'held measurement leaked into inverse'
            assert np.array_equal(baseline['predicted_held'],perturbed['predicted_held'],equal_nan=True)
            masking.append({'mode':mode,'bitwise_inverse_and_predicted_held_invariant':True,'fields':checks})
        records.append({'held_point':held,'rows':len(rows),'legacy_seconds':legacy_seconds,'compacted_seconds':new_seconds,
                        'legacy_compacted_max_state_difference_deg_D':state_difference.tolist(),'cost_max_difference':cost_difference,
                        'CPU_GPU_max_state_difference_deg_D':backend_difference.tolist(),'masking':masking})
    y=relative_coordinates(pop['p1'][rows],pop['p4'][rows]);all3=infer_all(spec,p,y,r,xp=cp)
    cpuall=infer_all(spec,p,y[::2],r,xp=np);valid=cpuall['certified']&all3['certified'][::2]
    assert np.max(np.abs(cpuall['states'][valid]-all3['states'][::2][valid]))<2e-5
    params=spec.parameters(p);good=all3['certified']&(~all3['ambiguous'])
    prediction,scale,ok=predict_relative(all3['states'][good,0],all3['states'][good,1],y[good,:4],r[:4,:4],params)
    residual=y[good]-prediction;precision=np.linalg.solve(r,np.eye(10));cost=.5*np.einsum('ni,ij,nj->n',residual,precision,residual)
    assert ok.all() and np.max(np.abs(cost-all3['cost'][good]))<1e-7
    benchrows=np.flatnonzero(pop['complete_valid'])[:1024]
    started=time.perf_counter();benchmark=infer_omission(spec,p,pop['p1'][benchrows],pop['p4'][benchrows],pop['p1_available'][benchrows],pop['p4_available'][benchrows],r,0,xp=cp)
    elapsed=time.perf_counter()-started
    source_paths=[ROOT/'distortion_model/inference.py',ROOT/'distortion_model/crosscheck.py',Path(__file__)]
    result={'kind':'actual-record numerical verification, legacy replay, CPU/GPU and whole-loop omitted-input perturbation',
            'passed':True,'predeclared_population_indices':rows.tolist(),'records':records,'all3_scored_rows':int(good.sum()),
            'all3_public_optics_cost_max_difference':float(np.max(np.abs(cost-all3['cost'][good]))),
            'benchmark':{'rows':len(benchrows),'seconds':elapsed,'scored':int(benchmark['scored'].sum()),
                         'rough_full_omission_seconds_at_this_rate':elapsed*300270/len(benchrows)},
            'source_sha256':{str(path.resolve().relative_to(ROOT)):digest(path) for path in source_paths},
            'parent_checkpoint_sha256':digest(parent/'checkpoint.json'),'automated_repository_suite_run':False}
    write(output/'summary.json',result);print(json.dumps(result,indent=2),flush=True)


def saved_audit(parent,attempt):
    inputs,spec,p,calibration,checkpoint=load_frozen(parent);pop=inputs['population'];r=inputs['covariance']['relative']
    with np.load(attempt/'crosscheck.npz',allow_pickle=False) as z:out=dict(z)
    cpout=json.loads((attempt/'checkpoint.json').read_text());prov=json.loads((attempt/'provenance.json').read_text())
    assert array_hash(out)==cpout['crosscheck_hash']
    assert digest(parent/'checkpoint.json')==cpout['parent_checkpoint_sha256']
    for path,hash_ in prov['parent_files_sha256'].items():assert digest(parent/path)==hash_
    for path,hash_ in prov['source_sha256'].items():assert digest(attempt/'source_snapshot'/path)==hash_
    assert out['states'].shape==(100090,3,2) and out['held_point'].shape==(100090,3)
    assert np.array_equal(out['held_point'],np.tile(np.arange(3),(100090,1)))
    for key in ('capture','row','source_frame','exposure'):assert np.array_equal(out[key],pop[key])
    records=[];params=spec.parameters(p)
    for held in range(3):
        keep=[j for j in range(3) if j!=held]
        retained=pop['p1_available'].all(axis=1)&pop['p4_available'][:,keep].all(axis=1)
        assert np.array_equal(retained,out['retained_input_valid'][:,held])
        assert np.array_equal(retained,out['inference_attempted'][:,held])
        scored=out['scored'][:,held];rows=np.flatnonzero(scored);states=out['states'][rows,held]
        y=relative_coordinates(pop['p1'][rows],pop['p4'][rows],held_point=held);idx=retained_indices(held)
        prediction,g,valid=predict_relative(states[:,0],states[:,1],y[:,:4],r[:4,:4],params)
        error=pop['p4'][rows,held]-pop['p1'][rows].mean(axis=1)-prediction[:,4+2*held:6+2*held]
        residual=y-prediction[:,idx];prec=np.linalg.solve(r[np.ix_(idx,idx)],np.eye(8))
        cost=.5*np.einsum('ni,ij,nj->n',residual,prec,residual)
        differences={'error_max_px':float(np.max(np.abs(error-out['error'][rows,held]))),
                     'g_max':float(np.max(np.abs(g-out['g'][rows,held]))),
                     'cost_max':float(np.max(np.abs(cost-out['cost'][rows,held])))}
        assert valid.all() and differences['error_max_px']<1e-8 and differences['g_max']<1e-10 and differences['cost_max']<1e-6
        assert np.all(out['projected_gradient_inf'][scored,held]<=1e-6)
        records.append({'held_point':held,'retained_valid':int(retained.sum()),'scored':len(rows),**differences})
    # Reconstruct primary metrics directly, without aggregation module helpers.
    complete=out['scored'].all(axis=1);ee=np.mean(np.sum(out['error']**2,axis=2),axis=1)
    ss=out['states'];d=np.stack((ss[:,0]-ss[:,1],ss[:,0]-ss[:,2],ss[:,1]-ss[:,2]),axis=1)
    gg=np.mean(d**2,axis=1);expected={}
    for label,values in (('E_px',ee),('Gtheta_deg',gg[:,0]),('GA_D',gg[:,1])):
        means=[np.mean(values[complete&(pop['exposure']==k)]) for k in range(20)]
        expected[label]=float(np.sqrt(np.mean(means)))
    metrics=json.loads((attempt/'metrics.json').read_text())['primary_complete_triples']
    for key,value in expected.items():assert abs(metrics[key]-value)<1e-10
    result={'passed':True,'scheduled_rows':100090,'scheduled_slots':300270,'parent_unchanged':True,
            'archived_sources_match':True,'slot_identity_and_availability_match':True,'records':records,
            'independent_equal_exposure_metrics':expected,'complete_eligible_triples':int(complete.sum()),
            'audit_script_sha256':digest(Path(__file__)),'new_inference_run':False}
    write(attempt/'independent_audit.json',result);print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=ROOT/'experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04')
    parser.add_argument('--preflight-output',type=Path);parser.add_argument('--attempt',type=Path)
    args=parser.parse_args()
    if args.preflight_output:preflight(args.parent.resolve(),args.preflight_output.resolve())
    elif args.attempt:saved_audit(args.parent.resolve(),args.attempt.resolve())
    else:parser.error('provide --preflight-output or --attempt')


if __name__=='__main__':main()
