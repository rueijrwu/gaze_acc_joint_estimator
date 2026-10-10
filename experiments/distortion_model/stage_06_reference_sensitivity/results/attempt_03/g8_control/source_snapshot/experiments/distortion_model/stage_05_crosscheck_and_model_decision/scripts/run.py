"""Full G8 frozen attempt04 omissions, with a predeclared all-three diagnostic."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
STAGE=ROOT/'experiments/distortion_model/stage_05_crosscheck_and_model_decision'
S4=ROOT/'experiments/distortion_model/stage_04_center_and_dm0_calibration'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(S4/'scripts'))
from repair_g7 import load_context
from distortion_model.data import array_hash
from distortion_model.geometry import relative_coordinates
from distortion_model.crosscheck import infer_omission,summarize
from distortion_model.inference import infer_all


def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load_frozen(parent):
    checkpoint=json.loads((parent/'checkpoint.json').read_text());summary=json.loads((parent/'summary.json').read_text())
    if not checkpoint['fit_certified'] or summary['G7_decision'] not in ('GO','GO_WITH_LIMIT'):
        raise ValueError('reviewed certified G7 checkpoint required')
    if checkpoint.get('schema')=='reference_sensitivity_calibration_v1':
        return load_reference_frozen(parent,checkpoint,summary)
    if digest(parent/checkpoint['model_file'])!=checkpoint['model_sha256']:raise ValueError('changed frozen model')
    for key in ('parent_checkpoint','parent_summary','base_g3_checkpoint'):
        path=ROOT/checkpoint[key+'_path'] if key!='parent_summary' else (ROOT/checkpoint['parent_checkpoint_path']).parent/'summary.json'
        if digest(path)!=checkpoint[key+'_sha256']:raise ValueError('changed calibration ancestry: '+key)
    runner,inputs,spec,bound,summary,x,p=load_context(parent)
    with np.load(parent/checkpoint['frame_states_file'],allow_pickle=False) as z:calibration=dict(z)
    if array_hash(calibration)!=checkpoint['frame_states_hash']:raise ValueError('changed frozen frame states')
    if not np.array_equal(calibration['input_valid'],inputs['valid']):raise ValueError('changed population')
    if not np.array_equal(x[:,0],calibration['theta_visual_deg'][inputs['valid']]) or not np.array_equal(x[:,1],calibration['accommodation_D'][inputs['valid']]):
        raise ValueError('solution and checkpoint states differ')
    model=json.loads((parent/'model.json').read_text());params=spec.parameters(p)
    if not np.allclose(params.b4,model['b4_reference_px'],rtol=0,atol=1e-12):raise ValueError('global solution/template mismatch')
    if not np.allclose(params.center[:5],model['center_coefficients_reference_px'],rtol=0,atol=1e-12):raise ValueError('global center mismatch')
    return inputs,spec,p,calibration,checkpoint


def load_reference_frozen(parent,checkpoint,summary):
    """Load a fresh reference candidate using its own immutable template gauge."""
    from distortion_model.joint import JointSpec
    def checked(path,hash_):
        if digest(path)!=hash_:raise ValueError('changed reference candidate input: '+str(path))
    campaign=ROOT/checkpoint['campaign_path']
    checked(campaign/'provenance.json',checkpoint['campaign_provenance_sha256'])
    provenance=json.loads((campaign/'provenance.json').read_text())
    checked(campaign/'protocol.json',provenance['protocol_sha256'])
    checked(campaign/'population.npz',checkpoint['population_sha256'])
    checked(campaign/'covariance.npz',checkpoint['covariance_sha256'])
    for name in ('model','joint_spec','initialization'):
        checked(parent/checkpoint[name+'_file'],checkpoint[name+'_sha256'])
    solution_file=parent/checkpoint['selected_start']/'solution.npz';checked(solution_file,checkpoint['solution_sha256'])
    if summary['selected_start']!=checkpoint['selected_start']:raise ValueError('selected start mismatch')
    with np.load(campaign/'population.npz',allow_pickle=False) as z:pop=dict(z)
    with np.load(campaign/'covariance.npz',allow_pickle=False) as z:cov=dict(z)
    with np.load(parent/checkpoint['frame_states_file'],allow_pickle=False) as z:calibration=dict(z)
    if array_hash(calibration)!=checkpoint['frame_states_hash']:raise ValueError('changed candidate calibration arrays')
    with np.load(solution_file,allow_pickle=False) as z:x,p=z['states'].copy(),z['scaled_globals'].copy()
    spec_record=json.loads((parent/checkpoint['joint_spec_file']).read_text());spec_record.pop('slope_floor')
    spec=JointSpec(**spec_record);valid=pop['complete_valid']
    if not np.array_equal(valid,calibration['input_valid']) or not np.array_equal(x[:,0],calibration['theta_visual_deg'][valid]) or not np.array_equal(x[:,1],calibration['accommodation_D'][valid]):
        raise ValueError('reference candidate state/population mismatch')
    model=json.loads((parent/checkpoint['model_file']).read_text());params=spec.parameters(p)
    for actual,stored in ((params.b4,model['b4_reference_px']),(params.center[:5],model['center_coefficients_reference_px']),
                          (params.k1,model['k1_native']),(params.k4,model['k4_native'])):
        if not np.allclose(actual,stored,rtol=0,atol=1e-12):raise ValueError('reference globals/model mismatch')
    if params.omega4!=model['omega4_visual_deg'] or params.m1!=model['m1_per_D']:raise ValueError('reference model convention mismatch')
    if not (parent/'independent_public_optics_audit.json').exists() or not json.loads((parent/'independent_public_optics_audit.json').read_text())['passed']:
        raise ValueError('reviewed public-optics audit required')
    inputs={'population':pop,'covariance':cov,'valid':valid,'exposure':pop['exposure'][valid],
            'interval_hash':provenance.get('interval_sha256',digest(ROOT/'data/fixations/fixation_intervals.json'))}
    return inputs,spec,p,calibration,checkpoint


def allocate(pop):
    n=len(pop['row']);out={key:pop[key].copy() for key in ('capture','exposure','row','source_frame','timestamp_ms')}
    out['held_point']=np.tile(np.arange(3,dtype=np.int8),(n,1))
    out['raw_input_valid']=np.repeat(pop['complete_valid'][:,None],3,axis=1)
    for key,shape in {'states':(2,),'error':(2,),'predicted_held':(2,),'gn_information':(2,2),
                      'observed_curvature_eigenvalues':(2,),'scaled_information_eigenvalues':(2,)}.items():
        out[key]=np.full((n,3)+shape,np.nan)
    for key in ('g','cost','projected_gradient_inf','jacobian_column_cosine','conditional_accommodation_information','scaled_information_condition'):
        out[key]=np.full((n,3),np.nan)
    for key in ('retained_input_valid','held_input_valid','certified','ambiguous','at_bounds','rank2','scored','inference_attempted','has_candidate'):
        out[key]=np.zeros((n,3),dtype=bool)
    out['active_state_bounds']=np.zeros((n,3,2),dtype=bool)
    for key in ('candidate_certified_count','candidate_stalled_count','selected_start','selected_accepted_steps','iterations'):
        out[key]=np.full((n,3),-1,dtype=np.int16)
    out['status']=np.full((n,3),'input_unavailable',dtype='U32')
    out['reason']=np.full((n,3),'retained_input_unavailable',dtype='U40')
    out['score_status']=np.full((n,3),'not_inferred',dtype='U32')
    fraction=np.zeros(n)
    for k in range(20):
        rows=np.flatnonzero(pop['exposure']==k);fraction[rows]=np.arange(len(rows))/max(len(rows)-1,1)
    out['interval_fraction']=fraction;out['endpoint']=(fraction==0)|(fraction==1)
    out['boundary_band']=(fraction<=.05)|(fraction>=.95)
    out['interval_block']=np.minimum((fraction*5).astype(np.int8),4)
    return out


def record_chunk(out,indices,held,record):
    out['retained_input_valid'][indices,held]=record['retained_input_valid']
    out['held_input_valid'][indices,held]=record['held_input_valid']
    if 'solution' not in record:return
    rows=indices[record['rows']];solved=record['solution'];out['inference_attempted'][rows,held]=True
    for key in ('states','cost','projected_gradient_inf','certified','has_candidate','ambiguous','at_bounds','rank2',
                'candidate_certified_count','candidate_stalled_count','selected_start','selected_accepted_steps',
                'observed_curvature_eigenvalues','active_state_bounds','gn_information','scaled_information_eigenvalues',
                'jacobian_column_cosine','conditional_accommodation_information','scaled_information_condition'):
        out[key][rows,held]=solved[key]
    out['iterations'][rows,held]=solved['iterations'];out['g'][rows,held]=record['scale']
    out['predicted_held'][rows,held]=record['predicted_held'];out['error'][rows,held]=record['error'];out['scored'][rows,held]=record['scored']
    status=np.where(~solved['certified'],'unresolved',np.where(solved['ambiguous'],'ambiguous',np.where(solved['at_bounds'],'certified_bound','certified')))
    reason=np.where(~solved['has_candidate'],'no_stationary_rank2_candidate',np.where(~solved['certified'],'observed_curvature_failed',np.where(solved['ambiguous'],'competing_state_branches','none')))
    out['status'][rows,held]=status;out['reason'][rows,held]=reason
    out['score_status'][rows,held]=np.where(record['scored'],'scored',np.where(~record['held_input_valid'][record['rows']],'held_input_unavailable','inverse_unresolved'))


def diagnostic_schedule(parent,pop):
    with np.load(parent/'compact_schedule.npz',allow_pickle=False) as z:compact=z['scheduled_population_index'].copy()
    neighbors=np.flatnonzero((pop['capture']=='capture_2_detections.pkl')&(pop['row']>=6794)&(pop['row']<=6805))
    rows=np.unique(np.r_[compact,neighbors])
    return {'population_index':rows,'legacy_compact_member':np.isin(rows,compact),'neighbor_case_member':np.isin(rows,neighbors),
            'capture':pop['capture'][rows],'exposure':pop['exposure'][rows],'row':pop['row'][rows],'source_frame':pop['source_frame'][rows]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=S4/'results/g7_attempt_04')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--chunk-size',type=int,default=1024)
    args=parser.parse_args();parent=args.parent.resolve();output=args.output.resolve();started=time.perf_counter()
    if args.chunk_size<1:raise ValueError('positive chunk size required')
    inputs,spec,p,calibration,parent_checkpoint=load_frozen(parent);pop=inputs['population'];r=inputs['covariance']['relative']
    parent_hashes={str(path.relative_to(parent)):digest(path) for path in parent.rglob('*') if path.is_file()}
    schedule=diagnostic_schedule(parent,pop)
    output.mkdir(parents=True,exist_ok=False)
    config={'stage':'S8/G8','authority':'user requested frozen G8 or its matched reference-comparison reuse',
            'population':'all100090 reviewed scheduled rows;3P4omissions/row;no trimming',
            'model':'frozen certified G7 parent; DM0, separate fixed operational zeros, P1-only GLS g, existing covariance',
            'parent_path':str(parent.relative_to(ROOT)),'omega1_visual_deg':spec.omega1,'omega4_visual_deg':spec.omega4,
            'inference':'49fixed7x7grid starts plus one retained-only center proposal;max60iterations;no labels or anchors',
            'numerical_policy':'projected gradient1e-6;rank rtol1e-10;observed free curvature positive;ambiguous inverses excluded from primary metrics',
            'acceleration':'CuPy float64, active-candidate line search compaction, chunked rows, one GPU process',
            'chunk_size':args.chunk_size,'max_iterations':60,'metric':'complete eligible triples;within-exposure squared means then20equal weights',
            'diagnostic_schedule':'original100compact rows union capture2source rows6794..6805 selected from previous audit before new inference',
            'diagnostic_rows':len(schedule['population_index']),'interval_strata':'exactendpoints,first/last5percent,central90percent,5contiguousblocks',
            'conditioning':'J.T R^-1 J and cosine/conditionalAinformation in declared weighting;not independently calibrated physiological precision',
            'later_optical_extension':'choose one from resulting mechanism evidence; no extension in this frozen-model experiment'}
    write(output/'config.json',config);np.savez_compressed(output/'diagnostic_schedule.npz',**schedule)
    files=list((ROOT/'distortion_model').glob('*.py'))+list((STAGE/'scripts').glob('*.py'))
    files+=list((S4/'scripts').glob('*.py'))+[ROOT/'docs/Theory.md',ROOT/'docs/ESTIMATOR_PLAN.md',ROOT/'docs/STAGE_GATES.md',
             ROOT/'docs/stages/05_CROSSCHECK_AND_MODEL_DECISION.md',ROOT/'docs/audits/G7_CERTIFIED_RESULTS_AUDIT.md',ROOT/'requirements-stage8.txt']
    sources={str(path.relative_to(ROOT)):digest(path) for path in files}
    for path in files:
        target=output/'source_snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    import cupy as cp
    device=cp.cuda.runtime.getDeviceProperties(0);gpu_name=device['name'];gpu_name=gpu_name.decode() if isinstance(gpu_name,bytes) else str(gpu_name)
    provenance={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                'source_hash':sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),'source_sha256':sources,
                'parent_path':str(parent.relative_to(ROOT)),'parent_files_sha256':parent_hashes,
                'interval_sha256':inputs['interval_hash'],'population_semantic_hash':array_hash({key:pop[key] for key in ('capture','exposure','row','source_frame','p1','p4','p1_available','p4_available')}),
                'covariance_semantic_hash':array_hash(inputs['covariance']),'runtime':{'python':sys.version,'numpy':np.__version__,'cupy':cp.__version__,'gpu':gpu_name},
                'started_UTC':datetime.now(timezone.utc).isoformat()}
    write(output/'provenance.json',provenance);np.savez_compressed(output/'covariance.npz',**inputs['covariance'])
    np.savez_compressed(output/'population.npz',**{key:pop[key] for key in ('capture','exposure','row','source_frame','timestamp_ms','target_theta_deg','demand_diopters','p1','p4','p1_available','p4_available','complete_valid')})
    out=allocate(pop)
    write(output/'summary.json',{'stage':'S8/G8','status':'RUNNING','fit_certified':True,'crosscheck_complete':False,'comparison_complete':False})
    for held in range(3):
        for k in range(20):
            indices=np.flatnonzero(pop['exposure']==k)
            for start in range(0,len(indices),args.chunk_size):
                rows=indices[start:start+args.chunk_size]
                record=infer_omission(spec,p,pop['p1'][rows],pop['p4'][rows],pop['p1_available'][rows],pop['p4_available'][rows],r,held,xp=cp)
                record_chunk(out,rows,held,record)
            progress={'held_point':held,'exposure_completed':k,'elapsed_seconds':time.perf_counter()-started,
                      'inferred_slots':int(out['inference_attempted'].sum()),'scored_slots':int(out['scored'].sum()),
                      'unresolved':int((out['status']=='unresolved').sum()),'GPU_pool_bytes':cp.get_default_memory_pool().used_bytes()}
            write(output/'progress.json',progress);print(json.dumps(progress),flush=True)
        np.savez_compressed(output/f'held_{held}.npz',**{key:value[:,held] for key,value in out.items() if value.ndim>=2 and value.shape[1]==3})
    indices=schedule['population_index'];complete=pop['complete_valid'][indices];rows=indices[complete]
    y=relative_coordinates(pop['p1'][rows],pop['p4'][rows]);diag={key:value.copy() for key,value in schedule.items()}
    diag['input_valid']=complete;diag['calibration_states']=np.column_stack((calibration['theta_visual_deg'][indices],calibration['accommodation_D'][indices]))
    diag['omission_states']=out['states'][indices];diag['omission_scored']=out['scored'][indices]
    solved=infer_all(spec,p,y,r,xp=cp)
    for key,value in solved.items():
        if isinstance(value,np.ndarray):
            full=np.full((len(indices),)+value.shape[1:],np.nan) if value.dtype.kind=='f' else np.zeros((len(indices),)+value.shape[1:],dtype=value.dtype)
            full[complete]=value;diag['all3_'+key]=full
    np.savez_compressed(output/'diagnostic_states.npz',**diag)
    np.savez_compressed(output/'crosscheck.npz',**out)
    result,metrics=summarize(pop,out);np.savez_compressed(output/'frame_metrics.npz',**metrics)
    write(output/'metrics.json',result)
    after={str(path.relative_to(parent)):digest(path) for path in parent.rglob('*') if path.is_file()}
    if parent_hashes!=after:raise ValueError('frozen calibration was modified')
    for path,hash_ in sources.items():
        if digest(ROOT/path)!=hash_:raise ValueError('source changed during G8: '+path)
    checkpoint={'schema':'g8_frozen_crosscheck_v1','parent_checkpoint_path':str((parent/'checkpoint.json').relative_to(ROOT)),
                'parent_checkpoint_sha256':digest(parent/'checkpoint.json'),'model_sha256':parent_checkpoint['model_sha256'],
                'source_hash':provenance['source_hash'],'fit_complete':True,'fit_certified':True,
                'crosscheck_complete':True,'comparison_complete':False,'scheduled_rows':len(pop['row']),'scheduled_slots':3*len(pop['row']),
                'crosscheck_file':'crosscheck.npz','crosscheck_hash':array_hash(out),'diagnostic_states_hash':array_hash(diag),
                'scientific_disposition':'PENDING_ROOT_REVIEW'}
    write(output/'checkpoint.json',checkpoint)
    summary={'stage':'S8/G8','status':'COMPLETE_PENDING_AUDIT','G8_decision':'none','fit_complete':True,'fit_certified':True,
             'crosscheck_complete':True,'comparison_complete':False,'source_hash':provenance['source_hash'],
             'primary':result['primary_complete_triples'],'coverage':result['all_slots'],
             'calibration_unchanged':True,'elapsed_seconds':time.perf_counter()-started,'next_action':'independent reconstruction, masking audit, and scientific interpretation'}
    write(output/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
