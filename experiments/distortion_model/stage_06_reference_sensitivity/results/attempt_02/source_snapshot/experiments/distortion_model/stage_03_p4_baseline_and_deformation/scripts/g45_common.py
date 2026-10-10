"""Shared evidence I/O for conditional G4/G5 experiments."""
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path
import json
import shutil
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').exists() and (p/'data/fixations/fixation_intervals.json').exists())
sys.path.insert(0,str(ROOT))
from distortion_model.data import array_hash,load_reviewed,make_population,validate_slots
from distortion_model.accommodation import shape_covariance,shape_derivatives,edges,DM0Shape


def write_json(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def load_npz(path,expected=None):
    arrays=dict(np.load(path,allow_pickle=False))
    if expected is not None and array_hash(arrays)!=expected:raise ValueError('semantic hash mismatch '+str(path))
    return arrays


def describe(x):
    x=np.asarray(x);x=x[np.isfinite(x)]
    if not len(x):return {'count':0,'mean':None,'std':None,'min':None,'max':None,'p05':None,'p95':None}
    return {'count':len(x),'mean':float(x.mean()),'std':float(x.std()),'min':float(x.min()),'max':float(x.max()),'p05':float(np.quantile(x,.05)),'p95':float(np.quantile(x,.95))}


def load_inputs(parent,tests_path,gate,min_tests):
    summary=json.loads((parent/'summary.json').read_text());checkpoint=json.loads((parent/'checkpoint.json').read_text())
    if summary.get(gate+'_decision') not in ['GO','GO_WITH_LIMIT']:raise ValueError('reviewed '+gate+' required')
    tests=None
    if tests_path is not None:
        tests=json.loads(tests_path.read_text());sources=sorted(list((ROOT/'distortion_model').glob('*.py'))+list((ROOT/'tests').glob('*.py')))
        current={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources}
        if tests['passed']<min_tests or tests['failed'] or tests['errors'] or tests['skipped'] or tests['source_hashes']!=current:raise ValueError('current passing contracts required')
    elif gate!='G5':
        raise ValueError('G4/G5 fitting requires a current passing test report')
    if gate=='G3':base=checkpoint;basepath=parent/'checkpoint.json'
    else:
        basepath=ROOT/checkpoint['base_g3_checkpoint_path']
        if sha256(basepath.read_bytes()).hexdigest()!=checkpoint['base_g3_checkpoint_sha256']:raise ValueError('changed G3 checkpoint')
        base=json.loads(basepath.read_text())
        for prefix in ['model','frame_states']:
            file=parent/checkpoint[prefix+'_file']
            if prefix=='model':
                if sha256(file.read_bytes()).hexdigest()!=checkpoint[prefix+'_sha256']:raise ValueError('changed G4 model')
            else:load_npz(file,checkpoint[prefix+'_hash'])
    manifest=load_npz(ROOT/base['population_path'],base['population_hash']);slots=load_npz(ROOT/base['slots_path'],base['slots_hash']);validate_slots(manifest,slots)
    covariance=load_npz(ROOT/base['covariance_path'],base['covariance_hash']);states=load_npz(ROOT/base['frame_states_path'],base['frame_states_hash'])
    p1=load_npz(ROOT/base['p1_export_path'],base['p1_export_hash'])
    source_parent=Path(ROOT/base['parent_checkpoint_path']).parent
    if sha256((source_parent/'checkpoint.json').read_bytes()).hexdigest()!=base['parent_checkpoint_sha256']:raise ValueError('changed G2 checkpoint')
    if sha256((source_parent/'summary.json').read_bytes()).hexdigest()!=base['parent_summary_sha256']:raise ValueError('changed G2 review')
    reference=json.loads((basepath.parent/base['p4_reference_file']).read_text())
    if sha256((basepath.parent/base['p4_reference_file']).read_bytes()).hexdigest()!=base['p4_reference_sha256']:raise ValueError('changed P4 reference')
    diagnostics=load_npz(basepath.parent/base['diagnostics_file'],base['diagnostics_hash'])
    captures,intervals,reviewed,interval_hash=load_reviewed(ROOT,workers=4);population=make_population(captures,intervals)
    if array_hash({key:population[key] for key in manifest})!=base['population_hash']:raise ValueError('changed reviewed schedule')
    measurements=load_npz((ROOT/base['population_path']).parent/'measurements.npz')
    for key,stored in [('p1','p1'),('p4','p4_corresponding')]:
        if not np.array_equal(population[key],measurements[stored],equal_nan=True):raise ValueError('changed native measurements')
    valid=population['complete_valid'];indices=np.flatnonzero(valid)
    if not np.array_equal(valid,p1['valid']) or not np.array_equal(valid,states['initialization_valid']):raise ValueError('changed valid starts/scales')
    return {'parent':parent,'checkpoint':checkpoint,'base':base,'basepath':basepath,'reference':reference,'population':population,'intervals':intervals,
            'covariance':covariance,'shape_covariance':shape_covariance(covariance['relative']),'states':states,'p1':p1,'diagnostics':diagnostics,
            'valid':valid,'indices':indices,'theta':states['theta_visual_deg'][valid],'g':p1['g'][valid],
            'observed_edges':edges(population['p4'][valid]),'exposure':population['exposure'][valid],
            'demand':population['demand_diopters'][valid],'initial_a':states['accommodation_start_D'][valid],
            'tests':tests,'interval_hash':interval_hash}


def model_json(model):
    return {'schema':'conditional_dm0_shape_v1','b4_reference_px':model.b4.tolist(),'omega4_visual_deg':model.omega4,'Aref_D':model.aref,
            'm1_per_D':model.m1,'k4_native':list(model.k4),'a_bounds_D':list(model.a_bounds),'theta_bounds_deg':list(model.theta_bounds),
            'baseline':'empirical incremental DM0-M1, no radial term','origin':'fixed empirical centroid convention, physical origin unknown',
            'scale_policy':'unchanged p1_profile_v1, g outside composed optics','scope':'conditional P4 shape initialization; no center polynomial/full fit'}


def read_model(record):return DM0Shape(record['b4_reference_px'],record['omega4_visual_deg'],record['Aref_D'],record['m1_per_D'],record['k4_native'],record['a_bounds_D'],record['theta_bounds_deg'])


def start_output(output,inputs,config,gate,extra_sources=()):
    if output.exists():raise ValueError('choose a fresh attempt path')
    output.mkdir(parents=True);write_json(output/'config.json',config)
    if inputs['tests'] is not None:write_json(output/'tests.json',inputs['tests'])
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote=subprocess.run(['git','ls-remote','origin','refs/heads/exp5_distortion_model'],cwd=ROOT,text=True,capture_output=True,timeout=20)
    if remote.returncode or not remote.stdout.strip().startswith(head):raise ValueError('remote branch compatibility requires review')
    files=sorted(set(list((ROOT/'distortion_model').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+list((ROOT/'docs').glob('*.md'))+
                     list((ROOT/'docs/stages').glob('*.md'))+list(Path(__file__).parent.glob('*.py'))+[ROOT/f'requirements-stage{gate[-1]}.txt']+list(extra_sources)))
    hashes={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in files}
    for file in files:
        archive=output/'source_snapshot'/file.relative_to(ROOT);archive.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,archive)
    runtime={'python':sys.version,'numpy':np.__version__,'remote':remote.stdout.strip(),'backend':'NumPy float64 CPU, BLAS/OMP threads 1'}
    try:
        import cupy
        runtime['cupy']={'version':cupy.__version__,'device_count':cupy.cuda.runtime.getDeviceCount()}
    except Exception as exc:runtime['cupy']={'unavailable':type(exc).__name__}
    provenance={'created_utc':datetime.now(timezone.utc).isoformat(),'source_commit':head,'source_hash':sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
                'source_hashes':hashes,'config_hash':sha256((output/'config.json').read_bytes()).hexdigest(),'runtime':runtime,
                'parent_path':str(inputs['parent'].relative_to(ROOT)),'parent_checkpoint_sha256':sha256((inputs['parent']/'checkpoint.json').read_bytes()).hexdigest(),
                'parent_summary_sha256':sha256((inputs['parent']/'summary.json').read_bytes()).hexdigest(),'interval_sha256':inputs['interval_hash']}
    write_json(output/'provenance.json',provenance);return provenance


def save_snapshot(output,name,inputs,model,a,fitmask):
    p=inputs['population'];valid=inputs['valid'];n=len(valid);opt=shape_derivatives(inputs['theta'],a,model)
    pred=inputs['g'][:,None]*opt['edges'];raw=inputs['observed_edges']-pred
    centered=p['p4'][valid]-p['p4'][valid].mean(axis=1,keepdims=True)
    point_residual=centered-inputs['g'][:,None,None]*(opt['F4']-opt['mu4'][:,None,:])
    def extend(x):
        result=np.full((n,)+x.shape[1:],np.nan);result[valid]=x;return result
    arrays={'accommodation_D':extend(a),'M':extend(1+model.m1*(a-model.aref)),'F4':extend(opt['F4']),'mu4':extend(opt['mu4']),
            'edge_prediction_px':extend(pred),'edge_residual_px':extend(raw),'centered_point_residual_px':extend(point_residual),
            'input_valid':valid,'conditional_state_solved':np.where(valid,False,False),
            'theta_visual_deg':inputs['states']['theta_visual_deg'],'g':inputs['p1']['g']}
    arrays['conditional_state_solved'][valid]=fitmask
    arrays['status']=np.where(valid,np.where(arrays['conditional_state_solved'],'conditional_shape_A','provisional_demand_A'),p['invalid_reason'])
    np.savez_compressed(output/f'{name}.npz',**arrays)
    records=[]
    for e,interval in enumerate(inputs['intervals']):
        select=(inputs['exposure']==e);native=(p['exposure']==e);original=inputs['indices'][select];blocks=[]
        ids=np.minimum(4,5*(p['row'][original]-interval['start_row'])//interval['row_count'])
        for b in range(5):
            sub=select.copy();sub[select]=ids==b
            blocks.append({'block':b,'count':int(sub.sum()),'edge_rms_px':float(np.sqrt(np.mean(raw[sub]**2))) if sub.any() else None,
                           'mean_A_D':describe(a[sub]),'signed_edge_mean_px':raw[sub].mean(axis=0).tolist() if sub.any() else None})
        records.append({'exposure':e,'capture':interval['capture'],'nominal_target_deg':interval['target_theta_deg'],'demand_D':interval['demand_diopters_label'],
                        'scheduled':int(native.sum()),'valid':int(select.sum()),'unavailable':int((native&~valid).sum()),'conditional_A_solved':int(fitmask[select].sum()),
                        'edge_rms_px':float(np.sqrt(np.mean(raw[select]**2))),'centered_point_rms_px':float(np.sqrt(np.mean(point_residual[select]**2))),
                        'signed_edge_mean_px':raw[select].mean(axis=0).tolist(),'signed_point_mean_px':point_residual[select].mean(axis=0).tolist(),
                        'A_D':describe(a[select]),'M':describe((1+model.m1*(a-model.aref))[select]),
                        'A_at_lower_bound':int((a[select]<=1e-9).sum()),'A_at_upper_bound':int((a[select]>=6-1e-9).sum()),'blocks':blocks})
    write_json(output/f'{name}_residuals.json',records)
    return arrays,records


def save_checkpoint(output,inputs,model,states,provenance,gate):
    write_json(output/'model.json',model_json(model))
    cp={'schema':f'conditional_{gate.lower()}_checkpoint_v1','source_hash':provenance['source_hash'],'source_commit':provenance['source_commit'],
        'parent_checkpoint_path':str((inputs['parent']/'checkpoint.json').relative_to(ROOT)),
        'parent_checkpoint_sha256':provenance['parent_checkpoint_sha256'],'parent_summary_sha256':provenance['parent_summary_sha256'],
        'base_g3_checkpoint_path':str(inputs['basepath'].relative_to(ROOT)),'base_g3_checkpoint_sha256':sha256(inputs['basepath'].read_bytes()).hexdigest(),
        'model_file':'model.json','model_sha256':sha256((output/'model.json').read_bytes()).hexdigest(),
        'frame_states_file':'fitted.npz','frame_states_hash':array_hash(states),'fit_complete':False,'fit_certified':False,'crosscheck_complete':False,
        'scope':'P4 shape initialization only, no optical center polynomial; conditional certificate recorded separately'}
    write_json(output/'checkpoint.json',cp)


def residual_plot(output,identity,fitted,title):
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for cap in sorted({r['capture'] for r in fitted}):
        a=[r for r in identity if r['capture']==cap];b=[r for r in fitted if r['capture']==cap]
        target=[r['nominal_target_deg'] for r in b]
        axes[0].plot(target,[r['edge_rms_px'] for r in a],'--o',alpha=.6)
        axes[0].plot(target,[r['edge_rms_px'] for r in b],'-o',label=cap)
        axes[1].errorbar(target,[r['A_D']['mean'] for r in b],yerr=[r['A_D']['std'] for r in b],fmt='o-',label=cap)
    axes[0].set(xlabel='Nominal visual gaze (degrees)',ylabel='Native P4 edge RMS (px)',title='Dashed identity / solid fitted')
    axes[1].set(xlabel='Nominal visual gaze (degrees)',ylabel='Conditional frame A mean ± SD (D)',title='Spread is descriptive, not uncertainty')
    for ax in axes:ax.legend(fontsize=7)
    fig.suptitle(title);fig.tight_layout();fig.savefig(output/'residuals_and_states.png',dpi=150);plt.close(fig)
