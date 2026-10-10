"""Declare a matched numerical continuation and publish its unchanged-objective fits."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from run import ROOT,context,write,digest,load_npz,joint_runner,array_hash
from distortion_model.centroid_bound import BoundedJointDM0
from distortion_model.geometry import relative_coordinates


def declare(source,output):
    # Verify both original campaigns before freezing a separate continuation.
    for label in ['control','adjacent']:context(source,label)
    output.mkdir(parents=True,exist_ok=False)
    policy=json.loads((source/'protocol.json').read_text())
    policy['numerical_continuation']={'source_campaign':str(source.relative_to(ROOT)),
        'reason':'fixed local working sets rejected new blocking frame or global inequalities',
        'method':'same observed Lagrangian Newton with state/global inequality working-set updates',
        'maximum_steps_per_original_start':12,'state_working_set_max_iterations':32,'global_working_set_max_iterations':96,
        'nonlinear_feasibility_correction_max_iterations':50,'correction_policy':'nearest declared-unit global feasibility correction; re-evaluate same original J and actual inequalities',
        'starts':['common','perturbed'],'matched_candidates':['control','adjacent'],
        'same_objective':True,'same_thresholds':True,'original_budget_outcomes_preserved':True,
        'global_working_set_policy':'current smooth multipliers define the Hessian; only strong current rows initialize equalities; weak rows remain inequalities; new proposal rows enter with zero current multiplier; positive tangent curvature, independent bordered solve and nonlinear feasibility guards retained'}
    write(output/'protocol.json',policy)
    for name in ['population.npz','covariance.npz']:shutil.copyfile(source/name,output/name)
    provenance=json.loads((source/'provenance.json').read_text())
    shutil.copytree(source/'source_snapshot',output/'source_snapshot')
    for path in [Path(__file__).resolve(),Path(__file__).with_name('repair.py').resolve()]:
        relative=str(path.relative_to(ROOT));provenance['source_sha256'][relative]=digest(path)
        target=output/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    provenance['parent_sha256'].update({str(path.relative_to(ROOT)):digest(path) for path in source.rglob('*') if path.is_file()})
    provenance.update(protocol_sha256=digest(output/'protocol.json'),original_campaign=str(source.relative_to(ROOT)))
    write(output/'provenance.json',provenance)
    for label in ['control','adjacent']:
        candidate=output/label;candidate.mkdir()
        for name in ['joint_spec.json','initialization.npz','initialization.json','compact_schedule.npz']:
            shutil.copyfile(source/label/name,candidate/name)
        shutil.copytree(source/label/'g6',candidate/'g6')
    write(output/'summary.json',{'status':'CONTINUATION_DECLARED','comparison_complete':False})


def publish(output,label):
    import cupy as cp
    candidate,pop,cov,valid,policy,spec,initial,bound=context(output,label)
    outcomes=[]
    for name in ['common','perturbed']:
        folder=candidate/name;summary=json.loads((folder/'summary.json').read_text())
        config=json.loads((folder/'config.json').read_text())
        if config['maximum_steps']!=policy['numerical_continuation']['maximum_steps_per_original_start']:
            raise ValueError('continuation budget differs from declaration')
        if config['repair_script_sha256']!=digest(Path(__file__).with_name('repair.py')):
            raise ValueError('continuation source changed')
        original=ROOT/config['campaign']/label/name/'solution.npz'
        if digest(original)!=config['source_solution_sha256'] or not summary['original_source_unchanged']:
            raise ValueError('original optimization point modified')
        record={'label':name,'components':summary['components'],'certificate':summary['certificate']}
        write(folder/'fit.json',record);outcomes.append(record)
    certified=[r for r in outcomes if r['certificate']['fit_certified']]
    selected=min(certified or outcomes,key=lambda r:r['components']['cost']);name=selected['label']
    solution=load_npz(candidate/name/'solution.npz');x,p=solution['states'],solution['scaled_globals']
    y=relative_coordinates(pop['p1'][valid],pop['p4'][valid]);e=pop['exposure'][valid]
    targets=np.array([[pop['target_theta_deg'][pop['exposure']==k][0],pop['demand_diopters'][pop['exposure']==k][0]] for k in range(20)])
    objective=BoundedJointDM0(spec,y,e,cov['relative'],targets,xp=cp,physical_bound=bound)
    snap=objective.snapshot(cp.asarray(x),cp.asarray(p));n=len(valid)
    def extend(value):
        a=np.full((n,)+value.shape[1:],np.nan);a[valid]=value;return a
    arrays={'input_valid':valid,'theta_visual_deg':extend(x[:,0]),'accommodation_D':extend(x[:,1]),
        'status':np.where(valid,'joint_fit_certified' if certified else 'joint_fit_uncertified','input_unavailable')}
    arrays.update({k:extend(v) for k,v in snap.items()});arrays['relative_residual_px']=extend(y-snap['prediction'])
    np.savez_compressed(candidate/'fitted.npz',**arrays)
    model=joint_runner.model_record(json.loads((candidate/'g6/model.json').read_text()),spec,p,
        {'source_hash':digest(output/'provenance.json'),'config_hash':digest(output/'protocol.json')})
    model.update(model_config_revision='GX_reference_DM0_1degree_v1',physical_centroid_constraint=bound.record(p),
        scope='matched reference calibration with separately declared state-bound numerical continuation')
    write(candidate/'model.json',model)
    write(candidate/'checkpoint.json',{'schema':'reference_sensitivity_calibration_v1','model_file':'model.json','model_sha256':digest(candidate/'model.json'),
        'frame_states_file':'fitted.npz','frame_states_hash':array_hash(arrays),'selected_start':name,'solution_sha256':digest(candidate/name/'solution.npz'),
        'joint_spec_file':'joint_spec.json','joint_spec_sha256':digest(candidate/'joint_spec.json'),
        'initialization_file':'initialization.npz','initialization_sha256':digest(candidate/'initialization.npz'),
        'campaign_path':str(output.relative_to(ROOT)),'campaign_provenance_sha256':digest(output/'provenance.json'),
        'population_sha256':digest(output/'population.npz'),'covariance_sha256':digest(output/'covariance.npz'),
        'fit_complete':True,'fit_certified':bool(certified),'crosscheck_complete':False,'comparison_complete':False})
    write(candidate/'summary.json',{'status':'COMPLETE_PENDING_REVIEW' if certified else 'COMPLETE_UNCERTIFIED','G7_decision':'none','selected_start':name,
        'components':selected['components'],'certificate':selected['certificate'],'starts_completed':2,'certified_start_count':len(certified),'outcomes':outcomes,
        'fit_complete':True,'fit_certified':bool(certified),'crosscheck_complete':False,'comparison_complete':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['declare','publish'])
    parser.add_argument('--source',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidate',choices=['control','adjacent']);args=parser.parse_args()
    if args.phase=='declare':
        if args.source is None:parser.error('--source required')
        declare(args.source.resolve(),args.output.resolve())
    else:
        if args.candidate is None:parser.error('--candidate required')
        publish(args.output.resolve(),args.candidate)


if __name__=='__main__':main()
