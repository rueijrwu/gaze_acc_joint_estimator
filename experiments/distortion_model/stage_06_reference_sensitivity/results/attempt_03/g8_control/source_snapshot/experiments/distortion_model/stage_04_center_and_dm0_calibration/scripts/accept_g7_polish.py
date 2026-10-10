"""Record one reviewed saved-point correction; no fit loop or subset inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
STAGE=ROOT/'experiments/distortion_model/stage_04_center_and_dm0_calibration'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
from repair_g7 import load_context,write,digest
from distortion_model.centroid_bound import BoundedJointDM0
from distortion_model.geometry import relative_coordinates
from distortion_model.polishing import trial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',type=Path,default=STAGE/'results/g7_polish_repair_01')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();probe=args.probe.resolve();output=args.output.resolve()
    provenance=json.loads((probe/'provenance.json').read_text())
    review=json.loads((probe/'summary.json').read_text())
    attempt=ROOT/provenance['input_attempt']
    before={str(p.relative_to(attempt)):digest(p) for p in attempt.rglob('*') if p.is_file()}
    assert before==provenance['input_sha256'],'frozen optimization parent changed'
    reporting_change={}
    reporter=str((STAGE/'scripts/summarize_g7_compact.py').relative_to(ROOT))
    for path,hash_ in provenance['source_sha256'].items():
        current=digest(ROOT/path)
        if path==reporter and current!=hash_:
            # Root reviewed the post-probe E1 aggregation addition. This script
            # only reads existing compact records; it is not imported here or
            # by any objective, proposal, prediction or certificate function.
            reporting_change[path]={'probe_sha256':hash_,'current_sha256':current,
                                    'scope':'existing-record empirical reporting only; root reviewed'}
        else:assert current==hash_,f'probe source changed: {path}'
    assert review['accepted_candidate']['accepted'] and review['candidate_certificate']['fit_certified']
    assert review['input_attempt_unchanged'] and not review['calibration_checkpoint_written']
    runner,inputs,spec,bound,parent_summary,oldx,oldp=load_context(attempt)
    with np.load(probe/'proposed_candidate.npz',allow_pickle=False) as z:
        x,p=z['states'].copy(),z['scaled_globals'].copy()
    with np.load(probe/'proposed_step.npz',allow_pickle=False) as z:
        dx,dp=z['state_step'].copy(),z['shared_step'].copy()
    import cupy as cp
    pop=inputs['population'];valid=inputs['valid'];e=inputs['exposure']
    y=relative_coordinates(pop['p1'][valid],pop['p4'][valid])
    targets=np.array([[pop['target_theta_deg'][np.flatnonzero(pop['exposure']==k)[0]],
                       pop['demand_diopters'][np.flatnonzero(pop['exposure']==k)[0]]] for k in range(20)])
    objective=BoundedJointDM0(spec,y,e,inputs['covariance']['relative'],targets,xp=cp,physical_bound=bound)
    ox,op=cp.asarray(oldx),cp.asarray(oldp);initial=objective.evaluate(ox,op)
    qx,qp,ledger=trial(objective,ox,op,initial,cp.asarray(dx),cp.asarray(dp),review['accepted_candidate']['damping'])
    assert ledger['accepted'] and np.array_equal(objective.host(qx),x) and np.array_equal(objective.host(qp),p)
    certificate=objective.certificate(qx,qp)
    assert certificate['fit_certified'],'full unchanged certificate failed'
    final=objective.evaluate(qx,qp);snapshot=objective.snapshot(qx,qp)
    # Independent NumPy evaluation uses the same complete population, covariance,
    # priors and full mean anchors. This is verification, not new inference.
    cpu=BoundedJointDM0(spec,y,e,inputs['covariance']['relative'],targets,xp=np,physical_bound=bound)
    independent=cpu.evaluate(x,p);cpu_snapshot=cpu.snapshot(x,p)
    components={key:float(final[key]) for key in ('cost','point','theta_anchor','A_anchor','regularization','temporal')}
    differences={key:float(independent[key])-components[key] for key in components}
    assert max(map(abs,differences.values()))<1e-8
    prediction_error=float(np.max(np.abs(cpu_snapshot['prediction']-snapshot['prediction'])))
    assert prediction_error<1e-8
    after={str(v.relative_to(attempt)):digest(v) for v in attempt.rglob('*') if v.is_file()}
    assert before==after
    output.mkdir(parents=True,exist_ok=False)
    sources={path:digest(ROOT/path) for path in provenance['source_sha256']}
    sources[str(Path(__file__).resolve().relative_to(ROOT))]=digest(Path(__file__))
    for path in sources:
        target=output/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/path,target)
    from hashlib import sha256
    source_hash=sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    config={'kind':'one reviewed saved-point constrained Newton correction','optimization_parent':str(attempt.relative_to(ROOT)),
            'probe':str(probe.relative_to(ROOT)),'fit_loop_executed':False,'new_subset_inference':False,
            'stationarity_threshold':1e-6,'root_review':'accepted after source and saved-point KKT/feasibility/cost review; user requested audit repair'}
    write(output/'config.json',config)
    metadata={'source_hash':source_hash,'config_hash':digest(output/'config.json')}
    g6=json.loads((inputs['parent']/'model.json').read_text())
    model=runner.model_record(g6,spec,p,metadata)
    model.update(model_config_revision='S7_G7_DM0_joint_1degree_Lagrangian_polish_v3',
                 physical_centroid_constraint=bound.record(p),
                 physical_constraint_authority='user accepted one degree over0to4D',
                 scope='one constrained local correction; G6 reference/prior origin retained; physical accuracy unproven')
    write(output/'model.json',model)
    selected=output/'polished';selected.mkdir()
    np.savez_compressed(selected/'initial.npz',states=oldx,scaled_globals=oldp)
    np.savez_compressed(selected/'solution.npz',states=x,scaled_globals=p)
    fit={'label':'polished','components':components,'certificate':certificate,'local_correction':ledger,
         'scaled_globals':p.tolist(),'means':objective.host(final['means']).tolist(),
         'mean_deviations':objective.host(final['deviation']).tolist()}
    write(selected/'fit.json',fit);write(selected/'history.json',[ledger])
    def extend(value):
        out=np.full((len(valid),)+value.shape[1:],np.nan);out[valid]=value;return out
    arrays={'input_valid':valid,'joint_state_evaluated':valid.copy(),'theta_visual_deg':extend(x[:,0]),
            'accommodation_D':extend(x[:,1]),'status':np.where(valid,'joint_fit_certified','input_unavailable')}
    arrays.update({key:extend(value) for key,value in snapshot.items()})
    arrays['relative_residual_px']=extend(y-snapshot['prediction'])
    np.savez_compressed(output/'fitted.npz',**arrays)
    shutil.copyfile(attempt/'compact_schedule.npz',output/'compact_schedule.npz')
    parent_checkpoint=json.loads((attempt/'checkpoint.json').read_text())
    checkpoint=dict(parent_checkpoint)
    checkpoint.update(source_hash=source_hash,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                      selected_start='polished',model_sha256=digest(output/'model.json'),frame_states_hash=runner.array_hash(arrays),
                      fit_complete=True,fit_certified=True,crosscheck_complete=False,comparison_complete=False,
                      optimization_parent_attempt=str(attempt.relative_to(ROOT)),
                      optimization_parent_solution_sha256=digest(attempt/parent_summary['selected_start']/'solution.npz'),
                      saved_point_probe_summary_sha256=digest(probe/'summary.json'))
    write(output/'checkpoint.json',checkpoint)
    write(output/'provenance.json',dict(config,source_sha256=sources,source_hash=source_hash,
          optimization_parent_files_sha256=before,probe_candidate_sha256=digest(probe/'proposed_candidate.npz'),
          probe_provenance_sha256=digest(probe/'provenance.json'),reviewed_reporting_source_changes=reporting_change,
          created_UTC=datetime.now(timezone.utc).isoformat()))
    verification={'original_attempt_unchanged':before==after,'candidate_bitwise_matches_reviewed_probe':True,
                  'independent_numpy_component_differences':differences,'independent_prediction_max_error_px':prediction_error,
                  'full_certificate_recomputed':True,'full_population':int(valid.sum()),'automated_test_suite_run':False}
    write(output/'verification.json',verification)
    rms=float(np.sqrt(np.mean([np.mean((y[e==k]-snapshot['prediction'][e==k])**2) for k in range(20)])))
    summary={'stage':'S7/G7','source_hash':source_hash,'status':'COMPLETE_CERTIFIED_WITH_LIMIT','G7_decision':'GO_WITH_LIMIT',
             'scheduled':len(valid),'complete_valid':int(valid.sum()),'unavailable':int((~valid).sum()),
             'joint_state_evaluated':len(x),'selected_start':'polished','starts_completed':1,'certified_start_count':1,
             'components':components,'certificate':certificate,'all_start_outcomes':[fit],
             'equal_exposure_native_relative_rms_px':rms,'local_correction':ledger,'verification':verification,
             'fit_complete':True,'fit_certified':True,'crosscheck_complete':False,'comparison_complete':False,
             'historical_compact_diagnostics':str((probe/'empirical_diagnostics').relative_to(ROOT)),
             'empirical_scope':'historical attempt03 compact results; no inference repeated for corrected model',
             'next_action':'separately authorize empirical diagnostic or G8; numerical closure does not establish optical adequacy'}
    write(output/'summary.json',summary)
    print(json.dumps({'certificate':certificate,'verification':verification,'RMS_px':rms},indent=2),flush=True)


if __name__=='__main__':main()
