"""Matched fresh G4-G7 calibration for one operational P4 reference alternative."""
import argparse
from datetime import datetime,timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from scipy.optimize import minimize

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
STAGE=ROOT/'experiments/distortion_model/stage_06_reference_sensitivity'
S3=ROOT/'experiments/distortion_model/stage_03_p4_baseline_and_deformation'
S4=ROOT/'experiments/distortion_model/stage_04_center_and_dm0_calibration'
S5=ROOT/'experiments/distortion_model/stage_05_crosscheck_and_model_decision'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(S3/'scripts'))
from g45_common import load_inputs,model_json,save_snapshot,load_npz
from distortion_model.accommodation import ShapeProfile,fit_shape,scalar_projection,shape_derivatives
from distortion_model.centers import CenterBlock
from distortion_model.centroid_bound import CentroidBound,BoundedJointDM0
from distortion_model.data import array_hash
from distortion_model.geometry import relative_coordinates
from distortion_model.joint import JointSpec,fit_joint
from distortion_model.optics import predict_relative
from distortion_model.p4 import near_reference,DualZeros


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def digest(path):return sha256(Path(path).read_bytes()).hexdigest()
def load_module(name,path):
    loader=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module);return module
joint_runner=load_module('reference_joint_runner',S4/'scripts/run_joint.py')


def protocol(output):
    inputs,_,_=joint_runner.check_parent(S4/'results/g6_attempt_02')
    reference=inputs['reference'];alternative=reference['bounded_adjacent_alternatives'][0]
    bound=json.loads((S4/'results/g7_attempt_04/model.json').read_text())['physical_centroid_constraint']
    output.mkdir(parents=True,exist_ok=False)
    policy={'stage':'GX/reference sensitivity','authority':'user requested next stage after completed G8',
        'candidates':{'control':{'omega4_deg':reference['omega4_visual_deg'],'reference_exposure':reference['reference_exposure']},
                      'adjacent':{'omega4_deg':alternative['omega4_visual_deg'],'reference_exposure':alternative['exposure']}},
        'population':'all100090 scheduled frames;89175complete;20exposures;no trimming',
        'intervention':'one P4 reference convention; empirical baseline and affected G4-G8 stages refit consistently',
        'fixed':'omega1/source identities/camera axes, covariance, domains, anchors .10deg/.25D, zero temporal penalty, DM0-M1 degree2 family',
        'prior_policy':'template tangent sigma.10 and normalized D c2 prior; native radius and template gauge recorded per reference',
        'initialization':'fresh G2 theta/g and provisional demand A; near-window G4 then all-row G5 then exact G6 D solve',
        'conditional_starts':'G4 measured projection slope and1.5times;G5 fixed-G4 keystone seed and zero keystone;matched identity baseline also recorded',
        'joint_starts':['fresh own G6 common','PCG64seed20261010 same scaled/global and state perturbations'],
        'max_outer':48,'joint_steps':4,'polish_steps':12,'chunk_size':32768,
        'stationarity_threshold':1e-6,'budget_policy':'two fresh starts per reference; fixed48outer and12repairedpolish; preserve uncertified outcomes without ranking',
        'bound_deg':1.,'shared_slope_floor':bound['required_min_gaze_slope_px_per_deg'],
        'bound_policy':'same numeric certified G7 slope floor across references; do not recompute a different physical constraint from each G6',
        'G8_policy':'identical50starts,8192rowchunks,full300270slots;only certified and reviewed globals enter inference',
        'comparison_policy':'exactcommonframe/point identities;all-scheduled coverage, paired20exposure E/Gtheta/GA, point1signedaxes/timeblocks, bounds and conditional A information',
        'diagnostic_schedule':'same original100compact union capture2rows6794..6805 as G8',
        'selection':'lowest J among numerically certified starts; cross-error never selects a fit',
        'scientific_scope':'operational-reference/model-family sensitivity; not physical zero or physiological ground truth'}
    write(output/'protocol.json',policy)
    pop=inputs['population'];np.savez_compressed(output/'population.npz',**{key:pop[key] for key in ('capture','exposure','row','source_frame','timestamp_ms','target_theta_deg','demand_diopters','p1','p4','p1_available','p4_available','complete_valid')})
    np.savez_compressed(output/'covariance.npz',**inputs['covariance'])
    files=list((ROOT/'distortion_model').glob('*.py'))+list((STAGE/'scripts').glob('*.py'))+list((S3/'scripts').glob('*.py'))+list((S4/'scripts').glob('*.py'))+list((S5/'scripts').glob('*.py'))
    files+=[ROOT/'docs/Theory.md',ROOT/'docs/ESTIMATOR_PLAN.md',ROOT/'docs/STAGE_GATES.md',STAGE/'PLAN.md',STAGE/'requirements.txt']
    sources={str(path.relative_to(ROOT)):digest(path) for path in files}
    for path in files:
        target=output/'source_snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    parents={}
    for parent in [S3/'results/attempt_01',S3/'results/g4_attempt_01',S3/'results/g5_attempt_01',S4/'results/g6_attempt_02',S4/'results/g7_attempt_04',S5/'results/attempt_01']:
        parents.update({str(path.relative_to(ROOT)):digest(path) for path in parent.rglob('*') if path.is_file()})
    write(output/'provenance.json',{'created_UTC':datetime.now(timezone.utc).isoformat(),
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':sources,'parent_sha256':parents,'protocol_sha256':digest(output/'protocol.json'),
        'population_sha256':digest(output/'population.npz'),'covariance_sha256':digest(output/'covariance.npz'),
        'interval_sha256':inputs['interval_hash'],'runtime':{'python':sys.version,'numpy':np.__version__}})
    write(output/'summary.json',{'status':'DECLARED','comparison_complete':False})
    print(json.dumps(policy,indent=2),flush=True)


def initialize(output,label):
    policy=json.loads((output/'protocol.json').read_text());candidate=output/label;candidate.mkdir(exist_ok=False)
    inputs,_,g6_control=joint_runner.check_parent(S4/'results/g6_attempt_02')
    choice=policy['candidates'][label];omega=choice['omega4_deg'];aref=inputs['reference']['Aref_demand_D']
    theta=inputs['theta'];near=near_reference(theta,DualZeros(inputs['reference']['omega1_visual_deg'],omega))
    groups=inputs['population']['capture'][inputs['valid']]
    refmask=near&(inputs['exposure']==choice['reference_exposure'])
    if refmask.sum()<2 or len(np.unique(groups[near]))!=4:raise ValueError('insufficient near-reference support')
    corrected=inputs['diagnostics']['corrected_centered_p4_px'][inputs['valid']]
    template=corrected[refmask].mean(axis=0)
    rho,_=scalar_projection(corrected,template)
    demand=np.array([inputs['demand'][(groups==cap)&near][0] for cap in sorted(set(groups))])
    response=np.array([rho[(groups==cap)&near].mean() for cap in sorted(set(groups))])
    slope=float(np.dot(demand-aref,response-1)/np.sum((demand-aref)**2))
    def factory(a,mask=None,deformation=False):
        return ShapeProfile(theta,inputs['observed_edges'],inputs['g'],inputs['exposure'],inputs['demand'],a,
            template,omega,aref,inputs['shape_covariance'],data_indices=None if mask is None else np.flatnonzero(mask),
            data_groups=None if mask is None else groups,deformation=deformation)
    g4=candidate/'g4';g4.mkdir();begin=time.perf_counter()
    m4,a4,best4,records4=fit_shape(lambda:factory(inputs['initial_a'],near),[[10*slope],[15*slope]])
    write(g4/'fit.json',{'best':best4,'outcomes':records4,'template_frame_count':int(refmask.sum()),'near_frames':int(near.sum()),'near_by_exposure':np.bincount(inputs['exposure'][near],minlength=20).tolist(),'elapsed_seconds':time.perf_counter()-begin})
    write(g4/'model.json',model_json(m4));save_snapshot(g4,'fitted',inputs,m4,a4,near)
    if not best4['conditional_fit_certified']:raise ValueError('G4 conditional initialization uncertified; outcome preserved')
    g5=candidate/'g5';g5.mkdir();begin=time.perf_counter()
    identity,ia,ib,ir=fit_shape(lambda:factory(a4),[[10*m4.m1],[15*m4.m1]])
    deformation=load_module('reference_deformation',S3/'scripts/run_deformation.py')
    seed,seed_record=deformation.seed_keystone(inputs,m4,a4)
    m5,a5,best5,records5=fit_shape(lambda:factory(a4,deformation=True),[seed.tolist(),[10*m4.m1,0.,0.,0.]])
    write(g5/'fit.json',{'best':best5,'outcomes':records5,'identity_best':ib,'identity_outcomes':ir,'keystone_seed':seed_record,'elapsed_seconds':time.perf_counter()-begin})
    write(g5/'model.json',model_json(m5));save_snapshot(g5,'fitted',inputs,m5,a5,np.ones(len(a5),bool))
    save_snapshot(g5,'identity',inputs,identity,ia,np.ones(len(ia),bool))
    if not best5['conditional_fit_certified'] or not ib['conditional_fit_certified']:raise ValueError('G5 initializer uncertified; outcome preserved')
    valid=inputs['valid'];pop=inputs['population'];y=relative_coordinates(pop['p1'][valid],pop['p4'][valid])
    f1=inputs['p1']['F1'][valid];f4=shape_derivatives(theta,a5,m5)['F4']
    radius=float(np.sqrt(np.mean(np.sum(template**2,axis=1))))
    block=CenterBlock(theta,a5,inputs['g'],f1,f4,y,inputs['exposure'],inputs['covariance']['relative'],aref,degree=2,curvature_scale_px=radius)
    center,constant,certificate=block.solve()
    g6=candidate/'g6';g6.mkdir();record=dict(g6_control)
    record.update(omega4_visual_deg=omega,Delta14_deg=omega-record['omega1_visual_deg'],b4_reference_px=template.tolist(),
        m1_per_D=m5.m1,k4_native=list(m5.k4),center_coefficients_reference_px=center.coefficients.tolist(),
        scope='fresh reference-specific G4-G6 initialization; original population/covariance/P1 reference retained')
    write(g6/'model.json',record);write(g6/'fit.json',certificate)
    spec=JointSpec(np.array(record['b1_reference_px']),template,record['omega1_visual_deg'],omega,aref)
    p0=spec.pack(record);x0=np.column_stack((theta,a5));bound=CentroidBound(spec,p0,1.,slope_floor=policy['shared_slope_floor'])
    original=p0.copy();projection={'required':bool(bound.values(p0).min() < -1e-10)}
    if projection['required']:
        def value(d):q=original.copy();q[9:]=d;return bound.values(q)
        def jac(d):q=original.copy();q[9:]=d;return bound.jacobian(q)[:,9:]
        result=minimize(lambda d:(.5*np.sum((d-original[9:])**2),d-original[9:]),original[9:],jac=True,
            method='SLSQP',constraints=[{'type':'ineq','fun':value,'jac':jac}],options={'ftol':1e-12,'maxiter':300})
        p0[9:]=result.x;projection.update(success=bool(result.success),message=str(result.message),D_scaled_change=(p0[9:]-original[9:]).tolist())
    projection['minimum_physical_slack']=float(bound.values(p0).min());write(g6/'feasible_initialization.json',projection)
    if projection['minimum_physical_slack'] < -1e-10:raise ValueError('feasible shared-bound initialization failed')
    if not certificate['linear_block_certified']:raise ValueError('G6 exact center initializer uncertified')
    write(candidate/'joint_spec.json',{'b1':spec.b1.tolist(),'b4':spec.b4.tolist(),'omega1':spec.omega1,'omega4':spec.omega4,'aref':spec.aref,'slope_floor':bound.slope_floor,'theta_bounds':list(spec.theta_bounds),'a_bounds':list(spec.a_bounds),'template_sigma':spec.template_sigma})
    np.savez_compressed(candidate/'initialization.npz',states=x0,scaled_globals=p0,input_valid=valid)
    schedule=np.concatenate([rows[np.linspace(0,len(rows)-1,5,dtype=int)] for rows in [np.flatnonzero(pop['exposure']==k) for k in range(20)]])
    np.savez_compressed(candidate/'compact_schedule.npz',scheduled_population_index=schedule)
    write(candidate/'initialization.json',{'label':label,'reference':choice,'template_radius_px':radius,'G4_certified':True,'G5_certified':True,'G6_certified':True,
        'g4_fit_sha256':digest(g4/'fit.json'),'g5_fit_sha256':digest(g5/'fit.json'),'g6_model_sha256':digest(g6/'model.json'),'joint_spec_sha256':digest(candidate/'joint_spec.json'),'initialization_sha256':digest(candidate/'initialization.npz')})
    print(label,'G4-G6 initialization complete',flush=True)


def context(output,label):
    provenance=json.loads((output/'provenance.json').read_text())
    for filename,key in [('protocol.json','protocol_sha256'),('population.npz','population_sha256'),('covariance.npz','covariance_sha256')]:
        if digest(output/filename)!=provenance[key]:raise ValueError('campaign input changed: '+filename)
    for path,hash_ in provenance['source_sha256'].items():
        if digest(ROOT/path)!=hash_:raise ValueError('declared implementation changed: '+path)
    pop=load_npz(output/'population.npz');cov=load_npz(output/'covariance.npz');valid=pop['complete_valid'];policy=json.loads((output/'protocol.json').read_text())
    candidate=output/label;record=json.loads((candidate/'joint_spec.json').read_text());floor=record.pop('slope_floor')
    spec=JointSpec(**record);initial=load_npz(candidate/'initialization.npz');bound=CentroidBound(spec,initial['scaled_globals'],1.,slope_floor=floor)
    return candidate,pop,cov,valid,policy,spec,initial,bound


def fit(output,label):
    import cupy as cp
    candidate,pop,cov,valid,policy,spec,initial,bound=context(output,label)
    device=cp.cuda.runtime.getDeviceProperties(0);name=device['name'];name=name.decode() if isinstance(name,bytes) else str(name)
    write(candidate/'runtime.json',{'python':sys.version,'numpy':np.__version__,'cupy':cp.__version__,'device':name,
        'memory_before_bytes':list(cp.cuda.runtime.memGetInfo()),'backend':'CuPy float64, one GPU process',
        'BLAS_OMP_threads':1})
    e=pop['exposure'][valid];y=relative_coordinates(pop['p1'][valid],pop['p4'][valid])
    targets=np.array([[pop['target_theta_deg'][pop['exposure']==k][0],pop['demand_diopters'][pop['exposure']==k][0]] for k in range(20)])
    x0,p0=initial['states'],initial['scaled_globals'];lower,upper=spec.bounds();rng=np.random.default_rng(20261010)
    backend=joint_runner.empirical_audit(spec,p0,x0,y,e,cov['relative'],targets,name)
    write(candidate/'backend_audit.json',backend)
    dp=np.r_[rng.normal(0,.0002,2),rng.normal(0,.005),rng.normal(0,.005,3),rng.normal(0,.001,3),rng.normal(0,.001,10)]
    px=np.clip(x0+rng.normal(size=x0.shape)*[.20,.10],[-20.,0.],[20.,6.]);pp=np.clip(p0+dp,lower,upper)
    # Restore physical feasibility along the declared perturbation ray.
    for damping in range(30):
        pp=p0+(np.clip(p0+dp,lower,upper)-p0)*.5**damping
        if bound.values(pp).min()>=-1e-10:break
    else:pp=p0.copy()
    write(candidate/'perturbation.json',{'seed':20261010,'global_ray_halvings':damping,'minimum_physical_slack':float(bound.values(pp).min())})
    objective=BoundedJointDM0(spec,y,e,cov['relative'],targets,xp=cp,physical_bound=bound,chunk_size=policy['chunk_size'])
    outcomes=[]
    for name,sx,sp in [('common',x0,p0),('perturbed',px,pp)]:
        folder=candidate/name;folder.mkdir(exist_ok=False);begin=time.perf_counter()
        np.savez_compressed(folder/'initial.npz',states=sx,scaled_globals=sp)
        write(candidate/'summary.json',{'status':'RUNNING','selected_start':None,'fit_certified':False,'G7_decision':'none','current_start':name})
        def save(outer,x,p,current,norms):
            np.savez_compressed(folder/'latest.npz',states=objective.host(x),scaled_globals=objective.host(p))
            write(folder/'progress.json',{'outer':outer+1,'cost':current['cost'],**norms,'elapsed_seconds':time.perf_counter()-begin})
            print(label,name,'outer',outer+1,current['cost'],norms,flush=True)
        x,p,final,certificate,history=fit_joint(objective,sx,sp,max_outer=policy['max_outer'],joint_steps=policy['joint_steps'],polish_steps=policy['polish_steps'],checkpoint=save)
        hx,hp=objective.host(x),objective.host(p);np.savez_compressed(folder/'solution.npz',states=hx,scaled_globals=hp)
        record={'label':name,'components':{k:float(final[k]) for k in ('cost','point','theta_anchor','A_anchor','regularization','temporal')},'certificate':certificate,
            'means':objective.host(final['means']).tolist(),'elapsed_seconds':time.perf_counter()-begin,'scaled_globals':hp.tolist()}
        write(folder/'fit.json',record);write(folder/'history.json',history);outcomes.append(record)
        print(label,name,'fit complete',record['components'],certificate['fit_certified'],flush=True)
    certified=[r for r in outcomes if r['certificate']['fit_certified']];selected=min(certified or outcomes,key=lambda r:r['components']['cost'])
    name=selected['label'];solution=load_npz(candidate/name/'solution.npz');x,p=solution['states'],solution['scaled_globals']
    snap=objective.snapshot(cp.asarray(x),cp.asarray(p));n=len(valid)
    def extend(value):a=np.full((n,)+value.shape[1:],np.nan);a[valid]=value;return a
    arrays={'input_valid':valid,'theta_visual_deg':extend(x[:,0]),'accommodation_D':extend(x[:,1]),'status':np.where(valid,'joint_fit_certified' if certified else 'joint_fit_uncertified','input_unavailable')}
    arrays.update({k:extend(v) for k,v in snap.items()});arrays['relative_residual_px']=extend(y-snap['prediction']);np.savez_compressed(candidate/'fitted.npz',**arrays)
    initrecord=json.loads((candidate/'g6/model.json').read_text());metadata={'source_hash':digest(output/'provenance.json'),'config_hash':digest(output/'protocol.json')}
    model=joint_runner.model_record(initrecord,spec,p,metadata);model.update(model_config_revision='GX_reference_DM0_1degree_v1',physical_centroid_constraint=bound.record(p),scope='fresh full matched reference calibration; numerical certificate required')
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


def audit(output,label):
    candidate,pop,cov,valid,policy,spec,initial,bound=context(output,label)
    summary=json.loads((candidate/'summary.json').read_text());solution=load_npz(candidate/summary['selected_start']/'solution.npz');x,p=solution['states'],solution['scaled_globals'];saved=load_npz(candidate/'fitted.npz')
    y=relative_coordinates(pop['p1'][valid],pop['p4'][valid]);e=pop['exposure'][valid]
    prediction,g,ok=predict_relative(x[:,0],x[:,1],y[:,:4],cov['relative'][:4,:4],spec.parameters(p));r=y-prediction
    counts=np.bincount(e,minlength=20);w=1/(20*counts[e]);point=.5*np.sum(w*np.einsum('ni,ij,nj->n',r,np.linalg.solve(cov['relative'],np.eye(10)),r))
    means=np.stack([x[e==k].mean(axis=0) for k in range(20)]);targets=np.array([[pop['target_theta_deg'][pop['exposure']==k][0],pop['demand_diopters'][pop['exposure']==k][0]] for k in range(20)])
    delta=means-targets;components={'point':float(point),'theta_anchor':float(.5*np.mean(delta[:,0]**2/.10**2)),
        'A_anchor':float(.5*np.mean(delta[:,1]**2/.25**2)),'regularization':float(.5*np.sum((p[3:6]/.10)**2)+.5*np.sum(p[17:19]**2)),'temporal':0.}
    components['cost']=sum(components.values());differences={k:components[k]-summary['components'][k] for k in components}
    pred_error=float(np.max(np.abs(prediction-saved['prediction'][valid])));g_error=float(np.max(np.abs(g-saved['g'][valid])))
    if not ok.all() or max(map(abs,differences.values()))>1e-8 or pred_error>1e-8 or g_error>1e-8:raise ValueError('public optical/objective reconstruction failed')
    if not np.array_equal(x[:,0],saved['theta_visual_deg'][valid]) or not np.array_equal(x[:,1],saved['accommodation_D'][valid]):raise ValueError('solution/frame ledger mismatch')
    prov=json.loads((output/'provenance.json').read_text())
    if any(digest(ROOT/path)!=hash_ for path,hash_ in prov['parent_sha256'].items()):raise ValueError('historical parent modified')
    if any(digest(output/'source_snapshot'/path)!=hash_ for path,hash_ in prov['source_sha256'].items()):raise ValueError('archived source modified')
    write(candidate/'independent_public_optics_audit.json',{'passed':True,'kind':'full saved public-optics/objective reconstruction; no inference or tests',
        'population_rows':int(valid.sum()),'component_differences':differences,'prediction_max_abs_difference_px':pred_error,'g_max_difference':g_error,
        'all_public_optics_valid':bool(ok.all()),'parents_unchanged':True,'archived_sources_match':True,
        'equal_exposure_native_relative_rms_px':float(np.sqrt(np.mean([np.mean(r[e==k]**2) for k in range(20)])))})
    print(label,'independent audit passed',differences,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['declare','initialize','fit','audit']);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--candidate',choices=['control','adjacent'])
    args=parser.parse_args();output=args.output.resolve()
    if args.phase=='declare':protocol(output)
    else:
        if args.candidate is None:parser.error('--candidate required')
        {'initialize':initialize,'fit':fit,'audit':audit}[args.phase](output,args.candidate)


if __name__=='__main__':main()
