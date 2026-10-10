"""Fit framewise A with frozen linear barrel law and soft fixation means."""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from scipy.optimize import minimize

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
sys.path.insert(0,str(ROOT))
from distortion_model.frame_accommodation import FrameAccommodation
from distortion_model.data import array_hash
PARENT=ROOT/'experiments/reverse_transform/stage_03_independent_captures/results/run'
spec=importlib.util.spec_from_file_location('stage3',ROOT/'experiments/reverse_transform/stage_03_independent_captures/scripts/run.py')
stage3=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage3)


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(model,lower,upper,start):
    sol=minimize(model.evaluate,np.clip(start,lower,upper),jac=True,method='L-BFGS-B',
                 bounds=list(zip(lower,upper)),options={'maxiter':600,'ftol':1e-15,'gtol':1e-7,'maxls':40,'maxcor':10})
    a=sol.x.copy();value,gradient=model.evaluate(a);polish=[]
    # Exact diagonal-plus-one-mean-per-fixation Newton system. No dense Hessian.
    exposure=model.host(model.e);counts=model.host(model.count);q=model.host(model.q)
    for iteration in range(8):
        cert,curvature=model.certificate(a,lower,upper)
        if cert['projected_gradient_inf']<1e-7:break
        constrained=((a<=lower+1e-7)&(gradient>0))|((a>=upper-1e-7)&(gradient<0))
        free=~constrained
        if np.any(curvature[free]<=0) or not np.isfinite(curvature[free]).all():break
        inverse=np.where(free,1/np.maximum(curvature,1e-12),0.)
        local_gradient=gradient/q
        sums=np.bincount(exposure,weights=inverse,minlength=len(counts))
        terms=np.bincount(exposure,weights=local_gradient*inverse,minlength=len(counts))
        rank=2*model.strength/counts
        correction=rank*terms/(1+rank*sums)
        step=(-local_gradient+correction[exposure])*inverse
        accepted=False
        for power in range(25):
            proposal=np.clip(a+step*.5**power,lower,upper)
            new_value,new_grad=model.evaluate(proposal)
            projected=new_grad.copy()
            projected[((proposal<=lower+1e-7)&(new_grad>0))|((proposal>=upper-1e-7)&(new_grad<0))]=0
            if new_value<=value+1e-7 and np.max(abs(projected))<cert['projected_gradient_inf']:
                polish.append({'iteration':iteration,'objective_before':value,'objective_after':new_value})
                a,value,gradient=proposal,new_value,new_grad;accepted=True;break
        if not accepted:break
    cert,_=model.certificate(a,lower,upper)
    return a,{'objective_scaled_sum':value,'objective_equal_fixation_mean_px2':value/model.n,
              'optimizer_success':bool(sol.success),'optimizer_message':str(sol.message),'iterations':int(sol.nit),
              'certificate':cert,'numerical_polish':polish}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--anchor-width',type=float,default=.25)
    args=parser.parse_args()
    if not np.isfinite(args.anchor_width) or args.anchor_width<=0:raise ValueError('Positive anchor width required')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    import cupy as cp
    started=time.perf_counter()
    parent_summary=json.loads((PARENT/'summary.json').read_text());post=json.loads((PARENT/'postfit.json').read_text())
    with np.load(PARENT/'frames.npz',allow_pickle=False) as z:p=dict(z)
    with np.load(PARENT/'population.npz',allow_pickle=False) as z:population=dict(z)
    parent_hash={str(file.relative_to(ROOT)):digest(file) for file in PARENT.rglob('*') if file.is_file()}
    slope=post['anchored_linear_slope_per_D_px2'];aref=parent_summary['reference_demand_diopters_label']
    index=p['population_index'];exposure=p['exposure'];targets=population['demand_diopters'][index]
    expected=np.array([targets[exposure==f][0] for f in range(20)])
    keystone=np.array([parent_summary['captures'][c]['p4']['coefficients_scaled'][:4] for c in p['capture_index']])
    model=FrameAccommodation(p['observed_centered_p4'],p['reference_p4'],p['gaze_xy_deg'],p['gaze_units_deg'],
          keystone,p['p1_magnification'],exposure,expected,slope,aref,args.anchor_width,xp=cp)
    props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    device={'backend':'cupy','version':cp.__version__,'gpu':props['name'].decode(),'dtype':'float64'}
    protocol={'frozen':'Stage03 gaze, per-capture keystone, framewise P1 magnification, common reference and postfit slope.',
              'kappa_law':{'slope_per_D_px2':slope,'reference_A_D':aref},
              'frame_parameter':'A only; no P4 magnification, rotation, smoothing or radius normalization in fitting.',
              'objective':'Equal-fixation mean of mean-three-vertex squared inverse reference distances + equal-fixation mean of ((mean A - expected A)/anchor_width)^2 * residual_scale^2.',
              'anchor_width_D':args.anchor_width,'residual_scale_px':1.,'penalty_strength_px2_per_D2':model.strength,
              'anchor_mean':'Arithmetic mean over complete frames in each of the 20 fixation periods.',
              'A_bounds_D':[0.,6.],'inverse_domain':'Per-frame upper bound reduced only to keep the monotone inverse branch; no complete frame discarded.',
              'expected_A':'Nominal capture demand, not framewise accommodation ground truth.',
              'comparison':'Stage03 capture-constant κ and frozen-law nominal A, each scored on common valid frames.',
              'device':device}
    write(out/'protocol.json',protocol)
    sources=[Path(__file__).resolve(),ROOT/'distortion_model/frame_accommodation.py',ROOT/'distortion_model/capture_shape.py',
             ROOT/'distortion_model/p1_shape.py',ROOT/'distortion_model/data.py',
             ROOT/'experiments/reverse_transform/stage_03_independent_captures/scripts/run.py',
             ROOT/'experiments/reverse_transform/stage_01_capture1_p1_fit/scripts/run.py']
    for source in sources:
        dest=out/'source_snapshot'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    lower,upper=model.feasible_bounds()
    print(json.dumps({'complete_frames':len(index),'inverse_domain_limited_frames':int(np.sum(upper<6.-1e-7)),
                      'anchor_width_D':args.anchor_width}),flush=True)
    starts=[];solutions=[]
    for offset in (0.,-.5,.5):
        a,record=fit(model,lower,upper,targets+offset)
        starts.append({'initial_offset_D':offset,**record});solutions.append(a)
        print(json.dumps(starts[-1]),flush=True)
    best=int(np.argmin([r['objective_scaled_sum'] for r in starts]));a=solutions[best]
    recovered,valid=model.recover(cp.asarray(a));recovered=model.host(recovered);valid=model.host(valid)
    if not valid.all():raise ValueError('Final fit has invalid inverse')
    nominal,nominal_valid=model.recover(cp.asarray(targets));nominal=model.host(nominal);nominal_valid=model.host(nominal_valid)
    parent_kappa=np.array([parent_summary['captures'][c]['relative_barrel_per_px2'] for c in p['capture_index']])
    equivalent_A=aref+parent_kappa/slope
    parent_reproduced,parent_valid=model.recover(cp.asarray(equivalent_A))
    parent_reproduced=model.host(parent_reproduced);parent_valid=model.host(parent_valid)
    assert np.array_equal(parent_valid,p['inverse_valid'])
    difference=float(np.max(abs(parent_reproduced[parent_valid]-p['recovered_p4'][parent_valid])))
    assert difference<1e-8
    common=valid&nominal_valid&parent_valid
    comparison={'capture_constant':p['recovered_p4'],'nominal_A':nominal,'framewise_A':recovered}
    prediction=model.host(model.geometry(cp.asarray(a))[4])
    records=[]
    for c in range(4):
        sel=p['capture_index']==c;keep=sel&common
        records.append({'capture':c+1,'complete':int(sel.sum()),'common_valid':int(keep.sum()),
             'A_D':stage3.helpers.statistics(a[sel]),'framewise_errors_all_complete':stage3.errors(recovered[sel],p['reference_p4']),
             'forward_point_distance_all_complete_px':stage3.helpers.statistics(np.linalg.norm(
                 p['observed_centered_p4'][sel]-prediction[sel],axis=2)),
             'errors':{name:stage3.errors(points[keep],p['reference_p4']) for name,points in comparison.items()}})
    intervals=[]
    for f in range(20):
        sel=exposure==f;keep=sel&common
        intervals.append({'exposure':f,'capture':f//5+1,'nominal_gaze_deg':[-10,-5,0,5,10][f%5],
            'expected_A_D':float(expected[f]),'complete':int(sel.sum()),'common_valid':int(keep.sum()),
            'A_D':stage3.helpers.statistics(a[sel]),'mean_anchor_shift_D':float(a[sel].mean()-expected[f]),
            'errors':{name:stage3.errors(points[keep],p['reference_p4']) for name,points in comparison.items()}})
    frame_cost,_,_=model.frame_cost(cp.asarray(a));means=model.host(model.means(cp.asarray(a)))
    q=model.host(model.q)
    data_cost=float(q@model.host(frame_cost)/model.n)
    anchor_cost=float(model.strength*np.mean((means-expected)**2))
    cpu=FrameAccommodation(p['observed_centered_p4'],p['reference_p4'],p['gaze_xy_deg'],p['gaze_units_deg'],keystone,
          p['p1_magnification'],exposure,expected,slope,aref,args.anchor_width,xp=np)
    cv,cg=cpu.evaluate(a);gv,gg=model.evaluate(a)
    summary={'status':'COMPLETE' if starts[best]['certificate']['stationary'] else 'COMPLETE_NUMERICAL_LIMIT',
        'scheduled':parent_summary['scheduled'],'complete':len(index),'unavailable':parent_summary['unavailable'],
        'common_valid':int(common.sum()),'fitted_inverse_failures':int((~valid).sum()),
        'parent_inverse_failures':int((~parent_valid).sum()),'nominal_A_inverse_failures':int((~nominal_valid).sum()),
        'protocol':protocol,'selected_start':best,'starts':starts,
        'multistart_objective_mean_range':float(np.ptp([r['objective_equal_fixation_mean_px2'] for r in starts])),
        'data_inverse_vertex_MSE_px2':data_cost,'mean_anchor_penalty_px2':anchor_cost,
        'objective_equal_fixation_mean_px2':data_cost+anchor_cost,'captures':records,'fixations':intervals,
        'parent_inverse_reproduction_max_px':difference,'cpu_gpu_agreement':{'objective_difference':abs(cv-gv),'gradient_max_difference':float(np.max(abs(cg-gg)))},
        'runtime_seconds':time.perf_counter()-started,
        'limitations':'In-sample inverse-space fit adds one A per frame; error reduction is not physiological validation. Frozen κ law is capture/demand-confounded; capture4 frozen keystone has an active bound. No temporal smoothing.'}
    write(out/'summary.json',summary)
    np.savez_compressed(out/'frames.npz',population_index=index,capture_index=p['capture_index'],exposure=exposure,row=p['row'],
        source_frame=p['source_frame'],gaze_xy_deg=p['gaze_xy_deg'],gaze_units_deg=p['gaze_units_deg'],
        p1_magnification=p['p1_magnification'],reference_p4=p['reference_p4'],observed_centered_p4=p['observed_centered_p4'],
        keystone_scaled=keystone,A_D=a,kappa_per_px2=slope*(a-aref),expected_A_D=targets,
        lower_A_D=lower,upper_A_D=upper,recovered_p4=recovered,nominal_A_recovered_p4=nominal,
        parent_recovered_p4=p['recovered_p4'],inverse_valid=valid,nominal_A_inverse_valid=nominal_valid,
        parent_inverse_valid=parent_valid,common_inverse_valid=common,forward_residual=p['observed_centered_p4']-prediction)
    write(out/'provenance.json',{'started_UTC':datetime.now(timezone.utc).isoformat(),'parent_results':str(PARENT.relative_to(ROOT)),
          'parent_sha256':parent_hash,'source_sha256':{str(s.relative_to(ROOT)):digest(s) for s in sources},
          'population_semantic_hash':array_hash(population)})
    assert all(digest(ROOT/name)==hash_value for name,hash_value in parent_hash.items())
    print(json.dumps({key:summary[key] for key in ('status','common_valid','fitted_inverse_failures','runtime_seconds')},indent=2))


if __name__=='__main__':main()
