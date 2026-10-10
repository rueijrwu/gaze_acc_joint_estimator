"""S3/G3 independent P4 operational reference and dual-zero evidence only."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file() and (p/'data/fixations/fixation_intervals.json').is_file())
sys.path.insert(0,str(ROOT))
from distortion_model.data import load_reviewed, make_population, array_hash, validate_slots
from distortion_model.p1 import P1Model, evaluate_p1
from distortion_model.p4 import DualZeros, corrected_p4_pattern, p4_balance, select_shared_reference, near_reference
from distortion_model.optics import p1_reference, keystone

CONFIG={
    'stage':'S3/G3', 'scope':'independent angular P4 reference only; no P4 optical fit',
    'population':'unchanged reviewed schedule and complete-valid initialization population',
    'reference_selection':'smallest equal-capture mean fixed-axis balance across all five nominal gaze bins; stable nominal-order tie break',
    'omega4_policy':'one shared offset fixed at mean exported visual theta in low-demand selected exposure',
    'reference_alternatives':'retain all five sampled low-demand offsets; adjacent nominal reference(s) bounded alternatives',
    'symmetry_metric':'norm([q0x+q2x-2*q1x,q2y-q0y])/norm(q2-q0); fixed camera axes, source-corresponding slots, no sorting',
    'geometry_status':'physical illuminator symmetry and camera-optical alignment unknown; empirical pattern diagnostic',
    'scale_policy':'frozen G2 p1_profile_v1; centered P4 divided only by exported P1 g, no P4 size/area normalization',
    'p1_policy':'native reference/template/omega1/K1 unchanged; xi1=xi4+Delta14',
    'states':'G2 visual theta and provisional A unchanged; no fixation flattened to zero',
    'block_policy':'five contiguous original-row partitions in every full interval; diagnostic only',
    'near_reference_half_width_deg':2.5,
    'near_reference_policy':'half nominal target spacing; diagnostic support, not trimming or calibration validity',
    'theta_bounds_deg':[-20.,20.], 'workers':4, 'plot_stride':10,
    'backend':'vectorized NumPy float64 CPU; parallel exposure summaries',
    'identity_tolerances':{'rtol':1e-10,'atol':1e-9},
    'free_parameters':[], 'fixed_parameters':['G2 P1 model/scale','G2 visual/A starts','source correspondence','camera axes','one selected omega4'],
    'derived_parameters':['Delta14','framewise xi1/xi4','P4 corrected diagnostic shape'],
}


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def describe(values):
    v=np.asarray(values);v=v[np.isfinite(v)]
    if not len(v):return {'count':0,'mean':None,'std':None,'min':None,'max':None,'p05':None,'p95':None}
    return {'count':len(v),'mean':float(v.mean()),'std':float(v.std()),'min':float(v.min()),'max':float(v.max()),
            'p05':float(np.quantile(v,.05)),'p95':float(np.quantile(v,.95))}


def check_array(path,expected):
    arrays=dict(np.load(path,allow_pickle=False))
    if array_hash(arrays)!=expected:raise ValueError(f'checkpoint semantic hash mismatch: {path}')
    return arrays


def runtime():
    observed={'python':sys.version,'numpy':np.__version__,'matplotlib':matplotlib.__version__}
    try:
        import cupy
        observed['cupy']={'version':cupy.__version__,'device_count':cupy.cuda.runtime.getDeviceCount()}
    except Exception as exc:observed['cupy']={'unavailable':type(exc).__name__}
    remote=subprocess.run(['git','ls-remote','origin','refs/heads/exp5_distortion_model'],cwd=ROOT,capture_output=True,text=True,timeout=20)
    observed['remote']={'returncode':remote.returncode,'stdout':remote.stdout.strip(),'stderr':remote.stderr.strip()}
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if remote.returncode or remote.stdout.split()[0]!=head:raise ValueError('remote branch/head compatibility requires review')
    return observed,head


def plots(output,records,profiles,population,arrays,zeros):
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for capture in sorted({r['capture'] for r in records}):
        selected=[r for r in records if r['capture']==capture]
        target=[r['nominal_target_deg'] for r in selected]; label=f"capture {capture.split('_')[1]} / {selected[0]['demand_D']:g} D"
        for ax,key in zip(axes,['score','horizontal','vertical']):
            ax.plot(target,[r[key]['mean'] for r in selected],'o-',label=label)
            for r in selected:
                ax.scatter([r['nominal_target_deg']]*5,[b[key]['mean'] for b in r['blocks']],s=14,alpha=.3)
    for ax in axes:
        ax.axvline(profiles['chosen']['nominal_target_deg'],color='k',ls='--');ax.set_xlabel('Nominal visual target (degrees)');ax.legend(fontsize=7)
    for ax,title in zip(axes,['P4 balance (dimensionless diagnostic)','Horizontal imbalance','Vertical imbalance']):ax.set_title(title)
    fig.suptitle('Independent P4 reference: physical source symmetry and optical zero unresolved')
    fig.tight_layout();fig.savefig(output/'p4_balance.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for capture in sorted({r['capture'] for r in records}):
        select=(population['capture']==capture)&arrays['valid'];rows=population['row'][select][::10]
        axes[0].plot(rows,arrays['xi4_deg'][select][::10],lw=.6,label=capture)
        axes[1].plot(rows,arrays['xi1_deg'][select][::10],lw=.6,label=capture)
    for ax in axes:
        ax.axhline(0.,color='k',ls='--');ax.set_xlabel('Original source row');ax.legend(fontsize=7)
    axes[0].axhspan(-2.5,2.5,color='gray',alpha=.15);axes[0].set_ylabel('xi4 (degrees)')
    axes[1].set_ylabel('xi1 (degrees)');fig.suptitle(f'One visual gaze, two native angles; Delta14={zeros.delta14:.6f} degrees')
    fig.tight_layout();fig.savefig(output/'native_angle_trajectories.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for capture in sorted({r['capture'] for r in records}):
        selected=[r for r in records if r['capture']==capture]
        axes[0].plot([r['nominal_target_deg'] for r in selected],[r['corrected_radius_px']['mean'] for r in selected],'o-',label=capture)
        axes[1].plot([r['nominal_target_deg'] for r in selected],[r['near_reference_count'] for r in selected],'o-',label=capture)
    axes[0].set(ylabel='Centered P4 RMS radius / P1 g (reference px)',title='P4 size retained, no self-normalization')
    axes[1].set(ylabel='Frames within |xi4| <= 2.5 degrees',title='Measured near-reference support')
    for ax in axes:ax.set_xlabel('Nominal visual target (degrees)');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'p4_size_and_support.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=ROOT/'experiments/distortion_model/stage_02_p1_reference_and_scale/results/attempt_02')
    parser.add_argument('--test-report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter();parent=args.parent.resolve();output=args.output.resolve()
    if output.exists():raise SystemExit('Choose a fresh attempt directory; prior results are preserved')
    ps=json.loads((parent/'summary.json').read_text());cp=json.loads((parent/'checkpoint.json').read_text())
    if ps.get('G2_decision') not in ['GO','GO_WITH_LIMIT'] or not ps['conditional_fit_certified']:raise ValueError('reviewed usable G2 required')
    tests=json.loads(args.test_report.read_text())
    sources=sorted(list((ROOT/'distortion_model').glob('*.py'))+list((ROOT/'tests').glob('*.py')))
    test_hashes={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources}
    if tests['failed'] or tests['errors'] or tests['skipped'] or tests['passed']<40 or tests['source_hashes']!=test_hashes:raise ValueError('forty passing current-source contracts required')
    observed,head=runtime()
    states=check_array(parent/cp['frame_states_file'],cp['frame_states_hash']);p1_saved=check_array(parent/cp['consistent_exported_p1_file'],cp['consistent_exported_p1_hash'])
    for field in ['p1_reference','bootstrap']:
        if sha256((parent/cp[field+'_file']).read_bytes()).hexdigest()!=cp[field+'_sha256']:raise ValueError('changed parent '+field)
    manifest=check_array(ROOT/cp['parent_population_path'],cp['parent_population_hash'])
    covariance=check_array(ROOT/cp['parent_covariance_path'],cp['parent_covariance_hash'])
    slots=check_array(ROOT/cp['parent_slot_path'],cp['parent_slot_hash']);validate_slots(manifest,slots)
    captures,intervals,reviewed,interval_sha=load_reviewed(ROOT,workers=4);population=make_population(captures,intervals)
    if array_hash({key:population[key] for key in manifest})!=cp['parent_population_hash']:raise ValueError('changed full schedule/source measurements')
    upstream=Path(ROOT/cp['parent_population_path']).parent
    measurements=dict(np.load(upstream/'measurements.npz',allow_pickle=False))
    for key,parentkey in [('p1','p1'),('p4','p4_corresponding')]:
        if not np.array_equal(population[key],measurements[parentkey],equal_nan=True):raise ValueError('changed original measurements')
    model_record=json.loads((parent/cp['p1_reference_file']).read_text())
    model=P1Model(model_record['b1_reference_px'],model_record['omega1_visual_deg'],
                  (model_record['alpha1_deg_minus2'],model_record['beta1_deg_minus2'],model_record['gamma1_px_minus1_deg_minus1']),model_record['theta_bounds_deg'])
    theta=states['theta_visual_deg'];g=p1_saved['g'];edges=(population['p1'][:,1:]-population['p1'][:,:1]).reshape(-1,4)
    rebuilt=evaluate_p1(theta,edges,covariance['R11'],model)
    for key in rebuilt:
        if not np.allclose(rebuilt[key],p1_saved[key],rtol=1e-10,atol=1e-9,equal_nan=True):raise ValueError('inconsistent G2 P1 export '+key)
    corrected,shape_valid=corrected_p4_pattern(population['p4'],g)
    score,components,score_valid=p4_balance(population['p4'],g)
    valid=population['complete_valid'] & states['initialization_valid'] & p1_saved['valid'] & shape_valid & score_valid
    if not np.array_equal(valid,population['complete_valid']):raise ValueError('P4 diagnostic missing required valid rows')
    radius=np.sqrt(np.mean(np.sum(corrected**2,axis=2),axis=1));block_ids=np.zeros(len(theta),dtype=np.int8)
    for e,interval in enumerate(intervals):
        select=population['exposure']==e
        block_ids[select]=np.minimum(4,5*(population['row'][select]-interval['start_row'])//interval['row_count'])
    def exposure_record(e):
        interval=intervals[e];scheduled=population['exposure']==e;select=scheduled&valid
        blocks=[]
        for b in range(5):
            subset=select&(block_ids==b)
            blocks.append({'block':b,'valid':int(subset.sum()),'score':describe(score[subset]),
                           'horizontal':describe(components[subset,0]),'vertical':describe(components[subset,1]),
                           'visual_theta_deg':describe(theta[subset])})
        return {'exposure':e,'capture':interval['capture'],'nominal_target_deg':interval['target_theta_deg'],
                'demand_D':interval['demand_diopters_label'],'scheduled':int(scheduled.sum()),'valid':int(select.sum()),
                'unavailable':int((scheduled&~valid).sum()),'score':describe(score[select]),
                'horizontal':describe(components[select,0]),'vertical':describe(components[select,1]),
                'visual_theta_deg':describe(theta[select]),'corrected_radius_px':describe(radius[select]),'blocks':blocks}
    with ThreadPoolExecutor(max_workers=4) as executor:records=list(executor.map(exposure_record,range(20)))
    chosen,profiles=select_shared_reference(records)
    aref=min(r['demand_D'] for r in records);low=[r for r in records if r['demand_D']==aref]
    if len(low)!=5 or len({r['capture'] for r in low})!=1:raise ValueError('unique five-condition low-demand reference required')
    alternatives=[]
    for r in low:
        alternatives.append({'nominal_target_deg':r['nominal_target_deg'],'exposure':r['exposure'],
                             'omega4_visual_deg':r['visual_theta_deg']['mean'],'score':r['score']['mean'],
                             'equal_capture_mean_score':next(a['equal_capture_mean_score'] for a in profiles if a['nominal_target_deg']==r['nominal_target_deg'])})
    reference=next(r for r in alternatives if r['nominal_target_deg']==chosen['nominal_target_deg'])
    zeros=DualZeros(model.omega1,reference['omega4_visual_deg']);xi1,xi4=zeros.angles(theta)
    near=near_reference(theta,zeros,CONFIG['near_reference_half_width_deg'])&valid
    for r in records:
        select=(population['exposure']==r['exposure'])&valid
        r.update(xi1_deg=describe(xi1[select]),xi4_deg=describe(xi4[select]),near_reference_count=int((near&select).sum()))
        for b in r['blocks']:
            sub=select&(block_ids==b['block']);b['xi4_deg']=describe(xi4[sub]);b['near_reference_count']=int((sub&near).sum())
    support=[];capture_minima=[]
    for capture in sorted({r['capture'] for r in records}):
        rs=[r for r in records if r['capture']==capture];select=(population['capture']==capture)&valid
        support.append({'capture':capture,'demand_D':rs[0]['demand_D'],'valid':int(select.sum()),
                        'near_reference_count':int((near&select).sum()),'near_reference_xi4_deg':describe(xi4[near&select]),'xi4_deg':describe(xi4[select])})
        best=min(rs,key=lambda r:r['score']['mean'])
        capture_minima.append({'capture':capture,'demand_D':best['demand_D'],'nominal_minimum_deg':best['nominal_target_deg'],
                               'visual_mean_at_minimum_deg':best['visual_theta_deg']['mean'],'mean_score':best['score']['mean']})
    maximum=lambda x:float(np.nanmax(np.abs(x)))
    identity={'xi1_equals_xi4_plus_delta_max_abs_deg':maximum(xi1[valid]-xi4[valid]-zeros.delta14),
              'visual_from_xi1_max_abs_deg':maximum(xi1[valid]+zeros.omega1-theta[valid]),
              'visual_from_xi4_max_abs_deg':maximum(zeros.visual_from_p4(xi4[valid])-theta[valid]),
              'p1_native_identity_max_abs_px':maximum(p1_reference(zeros.omega1,model)[0]-model.b1),
              'p1_at_p4_zero_conversion_max_abs_px':maximum(p1_reference(zeros.omega4,model)[0]-keystone(zeros.delta14,model.b1,model.k1)[0]),
              'p1_at_p4_zero_change_from_native_max_abs_px':maximum(p1_reference(zeros.omega4,model)[0]-model.b1),
              'P4_native_identity_kind':'synthetic contract; empirical K4/M not fitted at G3'}
    if any(v>1e-9 for k,v in identity.items() if k.endswith(('max_abs_deg','identity_max_abs_px','conversion_max_abs_px'))):raise ValueError('dual-zero/native identity failed')
    output.mkdir(parents=True);write_json(output/'config.json',CONFIG);write_json(output/'tests.json',tests)
    sources=sorted(set(sources+list((ROOT/'docs').glob('*.md'))+list((ROOT/'docs/stages').glob('*.md'))+[Path(__file__).resolve(),ROOT/'requirements-stage3.txt']))
    source_hashes={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in sources}
    for file in sources:
        destination=output/'source_snapshot'/file.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,destination)
    source_hash=sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()
    provenance={'created_utc':datetime.now(timezone.utc).isoformat(),'source_commit':head,'source_hash':source_hash,'source_hashes':source_hashes,
                'config_hash':sha256((output/'config.json').read_bytes()).hexdigest(),'runtime':observed,'parent_path':str(parent.relative_to(ROOT)),
                'parent_summary_sha256':sha256((parent/'summary.json').read_bytes()).hexdigest(),'parent_checkpoint_sha256':sha256((parent/'checkpoint.json').read_bytes()).hexdigest(),
                'interval_sha256':interval_sha}
    write_json(output/'provenance.json',provenance)
    adjacent=[r for r in alternatives if abs(r['nominal_target_deg']-reference['nominal_target_deg'])==5]
    reference_record={'schema':'p4_angular_reference_v1','omega1_visual_deg':zeros.omega1,'omega4_visual_deg':zeros.omega4,'Delta14_deg':zeros.delta14,
                      'selected_nominal_target_deg':reference['nominal_target_deg'],'reference_capture':low[0]['capture'],'reference_exposure':reference['exposure'],
                      'Aref_demand_D':aref,'selection':'independent equal-capture P4 balance; visual offset assigned in low-demand capture',
                      'physical_zero_identified':False,'physical_spatial_origin':'unknown; angular reference is distinct',
                      'all_sampled_alternatives':alternatives,'bounded_adjacent_alternatives':adjacent,
                      'parameter_roster':{'fitted':[],'fixed':CONFIG['fixed_parameters'],'derived':CONFIG['derived_parameters']},
                      'scope':'angular convention only; P4 accommodation baseline/template/keystone not fitted',
                      'source_hash':source_hash,'parent_p1_reference_sha256':cp['p1_reference_sha256']}
    write_json(output/'p4_reference.json',reference_record)
    write_json(output/'p4_balance.json',{'chosen':chosen,'equal_capture_profiles':profiles,'capture_minima':capture_minima,'exposures':records})
    write_json(output/'native_support.json',{'half_width_deg':2.5,'captures':support,'identities':identity,
                                          'visual_theta':describe(theta[valid]),'xi1_deg':describe(xi1[valid]),'xi4_deg':describe(xi4[valid]),
                                          'full_visual_domain_xi1_deg':(np.asarray([-20.,20.])-zeros.omega1).tolist(),
                                          'full_visual_domain_xi4_deg':(np.asarray([-20.,20.])-zeros.omega4).tolist()})
    arrays={'xi1_deg':xi1,'xi4_deg':xi4,'valid':valid,'score':score,'components':components,'block_ids':block_ids,
            'near_reference_diagnostic':near,'corrected_centered_p4_px':corrected,
            'corrected_centroid_displacement_px':measurements['centroid_displacement']/g[:,None],
            'status':np.where(valid,'diagnostic_available',population['invalid_reason'])}
    np.savez_compressed(output/'p4_diagnostics.npz',**arrays)
    checkpoint={'schema':'stage_03_independent_p4_reference_v1','source_hash':source_hash,'source_commit':head,
                'parent_checkpoint_path':str((parent/'checkpoint.json').relative_to(ROOT)),
                'parent_checkpoint_sha256':provenance['parent_checkpoint_sha256'],'parent_summary_sha256':provenance['parent_summary_sha256'],
                'p4_reference_file':'p4_reference.json','p4_reference_sha256':sha256((output/'p4_reference.json').read_bytes()).hexdigest(),
                'diagnostics_file':'p4_diagnostics.npz','diagnostics_hash':array_hash(arrays),
                'frame_states_path':str((parent/cp['frame_states_file']).relative_to(ROOT)),'frame_states_hash':cp['frame_states_hash'],
                'p1_export_path':str((parent/cp['consistent_exported_p1_file']).relative_to(ROOT)),'p1_export_hash':cp['consistent_exported_p1_hash'],
                'population_path':cp['parent_population_path'],'population_hash':cp['parent_population_hash'],
                'slots_path':cp['parent_slot_path'],'slots_hash':cp['parent_slot_hash'],
                'covariance_path':cp['parent_covariance_path'],'covariance_hash':cp['parent_covariance_hash'],
                'fit_complete':False,'fit_certified':False,'crosscheck_complete':False,'next_gate_requires_review':True}
    write_json(output/'checkpoint.json',checkpoint)
    plots(output,records,{'chosen':chosen},population,arrays,zeros)
    summary={'status':'COMPLETE_PENDING_REVIEW','G3_decision':'none','stage':'S3/G3','source_hash':source_hash,
             'scheduled':len(theta),'valid_diagnostics':int(valid.sum()),'unavailable':int((~valid).sum()),'unresolved_valid':0,
             'expected_slots_unchanged':len(slots['held_point']),'chosen_nominal_target_deg':reference['nominal_target_deg'],
             'chosen_reference_is_endpoint':reference['nominal_target_deg'] in [-10.,10.],
             'omega1_visual_deg':zeros.omega1,'omega4_visual_deg':zeros.omega4,'Delta14_deg':zeros.delta14,
             'physical_zero_identified':False,'near_reference_counts':support,'identities':identity,
             'all_captures_have_near_reference_frames':all(r['near_reference_count']>0 for r in support),
             'tests':{k:tests[k] for k in ['passed','failed','errors','skipped']},'elapsed_seconds':time.perf_counter()-started,
             'fit_complete':False,'fit_certified':False,'crosscheck_complete':False,'comparison_complete':False}
    write_json(output/'summary.json',summary)
    (output/'STAGE_REPORT.md').write_text('# G3 evidence pending execution-owner review\n\nStatus: COMPLETE_PENDING_REVIEW. Decision: none.\n')
    (output/'PROGRESS.md').write_text('Current gate: G3 review. One next action: execution-owner audit.\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
