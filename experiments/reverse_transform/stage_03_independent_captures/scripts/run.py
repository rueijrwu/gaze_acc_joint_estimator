"""Independent P1/P4 capture fits, then descriptive demand association."""
import argparse
from datetime import datetime, timezone
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
from distortion_model.data import load_reviewed,make_population,array_hash
from distortion_model.p1_shape import centered
from distortion_model.capture_shape import CaptureShape
P1=ROOT/'experiments/reverse_transform/stage_01_capture1_p1_fit'
spec=importlib.util.spec_from_file_location('gaze_calibration',P1/'scripts/run.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)


def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def radius(points): return np.sqrt(np.mean(np.sum(centered(points)**2,axis=-1),axis=-1))


def limits(gaze,b,units,barrel):
    extent=np.max(abs(gaze/units),axis=0);r=radius(b)
    widths=list(np.r_[.5/np.maximum(extent**2,1e-12),
                     .4/np.maximum(1.25*extent*np.max(abs(b[:,[1,0]]),axis=0)/r,1e-12)])
    if barrel: widths.append(.25/np.max(np.sum(b*b,axis=1)/r**2))
    return np.asarray(widths)


def fitting(model,widths):
    n=len(widths);fractions=np.zeros((6,n))
    fractions[1,:4]=[.1,.1,0,0];fractions[2,:4]=[-.1,-.1,0,0]
    fractions[3,:4]=[0,0,.1,.1];fractions[4,:4]=[0,0,-.1,-.1]
    fractions[5,:4]=[.1,-.1,.1,-.1]
    if n==5: fractions[1:,4]=[.1,-.1,.1,-.1,0]
    starts=[]
    for start in fractions*widths:
        sol=minimize(model.evaluate,start,jac=True,method='L-BFGS-B',bounds=list(zip(-widths,widths)),
                     options={'maxiter':400,'ftol':1e-14,'gtol':1e-9,'maxls':50})
        starts.append({'coefficients':sol.x.tolist(),'objective_point_px2':float(sol.fun),
                       'success':bool(sol.success),'message':str(sol.message),'iterations':int(sol.nit)})
    best=min(starts,key=lambda s:s['objective_point_px2']);k=np.array(best['coefficients'])
    value,gradient=model.evaluate(k);polish=[]
    for _ in range(3):
        constrained=((k<=-widths+1e-7)&(gradient>0))|((k>=widths-1e-7)&(gradient<0))
        projected=gradient.copy();projected[constrained]=0;free=np.flatnonzero(~constrained)
        if max(abs(projected))<1e-7 or not len(free):break
        hessian=np.column_stack([(model.evaluate(k+np.eye(n)[j]*1e-5)[1]-
                                  model.evaluate(k-np.eye(n)[j]*1e-5)[1])/(2e-5) for j in range(n)])
        hessian=(hessian+hessian.T)/2;hf=hessian[np.ix_(free,free)]
        if np.linalg.eigvalsh(hf).min()<=0:break
        step=np.zeros(n);step[free]=np.linalg.solve(hf,-gradient[free]);accepted=False
        for power in range(20):
            proposal=k+step*.5**power
            if np.any(abs(proposal)>widths):continue
            pv,pg=model.evaluate(proposal);pp=pg.copy()
            pp[((proposal<=-widths+1e-7)&(pg>0))|((proposal>=widths-1e-7)&(pg<0))]=0
            if pv<=value+1e-10 and max(abs(pp))<max(abs(projected)):
                polish.append({'objective_before':value,'objective_after':pv})
                k,value,gradient=proposal,pv,pg;accepted=True;break
        if not accepted:break
    hessian=np.column_stack([(model.evaluate(k+np.eye(n)[j]*1e-5)[1]-
                              model.evaluate(k-np.eye(n)[j]*1e-5)[1])/(2e-5) for j in range(n)])
    active=abs(abs(k)-widths)<1e-7;free=np.flatnonzero(~active)
    eigen=np.linalg.eigvalsh(((hessian+hessian.T)/2)[np.ix_(free,free)])
    projected=gradient.copy()
    projected[((k<=-widths+1e-7)&(gradient>0))|((k>=widths-1e-7)&(gradient<0))]=0
    checks={'projected_gradient_inf':float(max(abs(projected))), 'active_bounds':active.tolist(),
            'free_hessian_eigenvalues':eigen.tolist(), 'domain':model.domain(k),
            'multistart_objective_range':float(max(s['objective_point_px2'] for s in starts)-min(s['objective_point_px2'] for s in starts)),
            'stationary':bool(max(abs(projected))<1e-6 and eigen.min()>0)}
    return k,{'coefficients_scaled':k.tolist(),'bounds_scaled':list(zip(-widths,widths)),
              'objective_point_px2':value,'checks':checks,'starts':starts,'numerical_polish':polish}


def errors(points,b):
    centered_points=centered(points)
    ratio=radius(centered_points)/radius(b)
    shape=centered_points/ratio[:,None,None]
    return {'point_distance_px':helpers.statistics(np.linalg.norm(points-b,axis=2)),
            'shape_point_distance_px':helpers.statistics(np.linalg.norm(shape-b,axis=2)),
            'radius_ratio':helpers.statistics(ratio)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--capture1-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    import cupy as cp
    started=time.perf_counter()
    props=cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    device={'backend':'cupy','cupy_version':cp.__version__,'gpu':props['name'].decode(),'dtype':'float64',
            'optimizer':'SciPy L-BFGS-B controller, CuPy device-resident cost and analytic gradient'}
    selected=[1] if args.capture1_only else [1,2,3,4]
    write(out/'protocol.json',{'captures':selected,'reference':'Capture1 zero-gaze mean at demand0.3603603604 D.',
        'gaze':'Per-capture centroid quadratic x calibration and same first-order x/y slope; zero-fixation mean gaze zero.',
        'P1':'Independent capture keystone fits, one frame magnification; capture1 audited gaze/M exactly retained.',
        'P4':'Independent keystone and one constant barrel increment within each capture; fixed framewise P1 magnification.',
        'barrel_reference':'Capture1 increment fixed zero; empirical reference already contains baseline barrel.',
        'demand':'Not supplied to image-model fitting. Descriptive straight-line association only after independent fitting.',
        'order':'Radial -> keystone -> model-only keystone size normalization -> P1 magnification.',
        'metric':'Original centered P4 point SSE; five equal fixation weights within each capture; no trimming.',
        'diagnostic':'After inverse, per-frame radius normalization only for shape performance. No rotation alignment.',
        'origin':'Fixed empirical P4 reference centroid; physical optical origin not independently measured.', 'device':device})
    sources=[Path(__file__).resolve(),ROOT/'distortion_model/capture_shape.py',ROOT/'distortion_model/p1_shape.py',
             ROOT/'distortion_model/data.py',P1/'scripts/run.py']
    for source in sources:
        dest=out/'source_snapshot'/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
    parents={str(p.relative_to(ROOT)):digest(p) for p in (P1/'results/attempt_01').rglob('*') if p.is_file()}
    captures,all_intervals,_,roster_hash=load_reviewed(ROOT,workers=4)
    intervals=[i for i in all_intervals if int(i['capture'].split('_')[1]) in selected]
    pop=make_population(captures,intervals);index=np.flatnonzero(pop['complete_valid'])
    with np.load(P1/'results/attempt_01/frames.npz',allow_pickle=False) as z:parent=dict(z)
    capidx=np.array([int(n.split('_')[1])-1 for n in pop['capture'][index]])
    exposure=pop['exposure'][index];local=exposure%5
    p1,p4=pop['p1'][index],pop['p4'][index]
    zero=(capidx==0)&(local==2);b1=centered(p1[zero]).mean(axis=0);b4=centered(p4[zero]).mean(axis=0)
    assert np.max(abs(b1-parent['reference']))<1e-10
    gaze=np.empty((len(index),2));magnification=np.empty(len(index));units_rows=np.empty_like(gaze)
    recovered1=np.empty_like(p1);before=centered(p4).copy();keystone=np.empty_like(p4);recovered4=np.empty_like(p4)
    inverse_valid=np.zeros(len(index),dtype=bool);keystone_valid=np.zeros(len(index),dtype=bool)
    residual=np.empty_like(p4);records=[];cpu_gpu=[]
    for capture in selected:
        c=capture-1;sel=capidx==c;fix=local[sel];count=np.bincount(fix,minlength=5)
        weights=1/(15*count[fix]);x1,x4=centered(p1[sel]),centered(p4[sel])
        theta,calibration=helpers.calibrate_gaze(p4[sel].mean(axis=1)-p1[sel].mean(axis=1),fix,np.array([-10.,-5.,0.,5.,10.]))
        units=parent['gaze_units_deg'].copy() if c==0 else np.array([10.,max(np.max(abs(theta[:,1])),.1)])
        gaze[sel]=theta;units_rows[sel]=units
        model1=CaptureShape(theta,b1,x1,weights,units,xp=cp)
        if c==0:
            assert np.array_equal(theta,parent['gaze_xy_deg'])
            k1=parent['coefficients'];_,mcheck=model1.prediction(k1)
            assert np.max(abs(mcheck-parent['magnification']))<1e-10
            m=parent['magnification'].copy();v,g1=model1.evaluate(k1)
            rec1={'coefficients_scaled':k1.tolist(),'objective_point_px2':v,
                  'checks':{'stationary':True,'domain':model1.domain(k1)},'source':'Audited capture1 P1 fit; gaze/M unchanged'}
        else:
            k1,rec1=fitting(model1,limits(theta,b1,units,False));_,m=model1.prediction(k1)
        back1,valid1=model1.inverse(k1,m)
        if not valid1.all() or not (m>0).all():raise ValueError('Invalid P1 inversion/magnification')
        magnification[sel]=m;recovered1[sel]=back1;before[sel]=x4/m[:,None,None]
        rec1['errors']=errors(back1,b1);rec1['magnification']=helpers.statistics(m)
        model4=CaptureShape(theta,b4,x4,weights,units,m,cp)
        k4,rec4=fitting(model4,limits(theta,b4,units,c!=0))
        prediction,_=model4.prediction(k4)
        back4,valid4=model4.inverse(k4,m)
        backk,validk=model4.inverse(k4,m,remove_barrel=False)
        recovered4[sel]=back4;keystone[sel]=backk;inverse_valid[sel]=valid4;keystone_valid[sel]=validk
        residual[sel]=x4-prediction
        # Audit CPU/GPU equality for the actual fitted model, not a benchmark.
        cpu=CaptureShape(theta,b4,x4,weights,units,m,np);cv,cg=cpu.evaluate(k4);gv,gg=model4.evaluate(k4)
        cpu_gpu.append({'capture':capture,'cost_difference':float(abs(cv-gv)),'gradient_max_difference':float(max(abs(cg-gg)))})
        common=valid4&validk
        rec4['inverse_failures']=int((~valid4).sum());rec4['common_inverse_valid']=int(common.sum())
        rec4['baseline_objective_point_px2']=float(np.sum(weights*np.sum((x4-m[:,None,None]*b4)**2,axis=(1,2))))
        rec4['errors']={name:errors(points[common],b4) for name,points in [('before',before[sel]),('keystone',backk),('recovered',back4)]}
        rec4['intervals']=[]
        for e,nominal in enumerate([-10,-5,0,5,10]):
            rows=fix==e;validrows=rows&common;interval=next(i for i in intervals if i['capture']==f'capture_{capture}_detections.pkl' and i['target_theta_deg']==nominal)
            rec4['intervals'].append({'nominal_gaze_deg':nominal,'scheduled':interval['row_count'],'complete':int(rows.sum()),
                'common_inverse_valid':int(validrows.sum()),'inverse_failures':int((rows&~valid4).sum()),
                'errors':{name:errors(points[validrows],b4) for name,points in [('before',before[sel]),('keystone',backk),('recovered',back4)]}})
        # Coefficients fitted without accommodation inputs; labels attached only here.
        kappa=float((k4[4] if len(k4)==5 else 0.)/radius(b4)**2)
        record={'capture':capture,'demand_diopters_label':float(pop['demand_diopters'][index][sel][0]),
                'scheduled':int(np.sum(pop['capture']==f'capture_{capture}_detections.pkl')),'complete':int(sel.sum()),
                'gaze_units_deg':units.tolist(),'gaze_calibration':calibration,'p1':rec1,'p4':rec4,
                'relative_barrel_per_px2':kappa,
                'keystone_native':{'k_ax_per_deg2':float(k4[0]/units[0]**2),'k_ay_per_deg2':float(k4[1]/units[1]**2),
                    'k_px_per_deg_px':float(k4[2]/(units[0]*radius(b4))),'k_py_per_deg_px':float(k4[3]/(units[1]*radius(b4)))}}
        records.append(record)
        folder=out/f'capture_{capture}';folder.mkdir();write(folder/'fit.json',record)
        print(json.dumps({'capture':capture,'p1_cost':rec1['objective_point_px2'],'p4_cost':rec4['objective_point_px2'],
                          'barrel_per_px2':kappa,'stationary':rec4['checks']['stationary'],'inverse_failures':rec4['inverse_failures']}),flush=True)
    common=inverse_valid&keystone_valid
    np.savez_compressed(out/'population.npz',**pop)
    np.savez_compressed(out/'frames.npz',population_index=index,capture_index=capidx,exposure=exposure,
        row=pop['row'][index],source_frame=pop['source_frame'][index],gaze_xy_deg=gaze,gaze_units_deg=units_rows,
        p1_magnification=magnification,reference_p1=b1,reference_p4=b4,observed_centered_p1=centered(p1),
        observed_centered_p4=centered(p4),recovered_p1=recovered1,before_p4=before,
        keystone_recovered_p4=keystone,recovered_p4=recovered4,inverse_valid=inverse_valid,
        keystone_inverse_valid=keystone_valid,common_inverse_valid=common,forward_residual=residual)
    write(out/'provenance.json',{'started_UTC':datetime.now(timezone.utc).isoformat(),'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in sources},
        'parent_sha256':parents,'input_sha256':{str((ROOT/'data/detections'/n).relative_to(ROOT)):cap.digest for n,cap in captures.items()},
        'roster_sha256':roster_hash,'population_semantic_hash':array_hash(pop)})
    assert all(digest(ROOT/p)==v for p,v in parents.items())
    complete=all(r['p1']['checks']['stationary'] and r['p4']['checks']['stationary'] for r in records)
    summary={'status':'COMPLETE' if complete else 'COMPLETE_NUMERICAL_LIMIT','device':device,'scheduled':len(pop['row']),
        'complete':len(index),'unavailable':len(pop['row'])-len(index),'common_inverse_valid':int(common.sum()),
        'inverse_failures':int((~inverse_valid).sum()),'reference_capture':1,'reference_gaze_deg':0,
        'reference_demand_diopters_label':records[0]['demand_diopters_label'],'reference_p1':b1.tolist(),'reference_p4':b4.tolist(),
        'captures':records,'cpu_gpu_agreement':cpu_gpu,'capture1_p1_gaze_M_exact':True,
        'runtime_seconds':time.perf_counter()-started,
        'limitations':'Empirical radial origin and relative coefficients; one capture per demand; no framewise accommodation estimate. Radius normalization is diagnostic only.'}
    write(out/'summary.json',summary)
    # Sole accommodation analysis: a descriptive line through independent offsets.
    if len(records)>1:
        d=np.array([r['demand_diopters_label'] for r in records]);kap=np.array([r['relative_barrel_per_px2'] for r in records])
        delta=d-d[0];slope=float(delta@kap/(delta@delta))
        write(out/'postfit.json',{'definition':'Demand labels used only AFTER independent image fitting.',
            'capture_coefficients':[{'capture':r['capture'],'demand_diopters_label':r['demand_diopters_label'],'relative_barrel_per_px2':r['relative_barrel_per_px2']} for r in records],
            'anchored_linear_slope_per_D_px2':slope,'coefficient_residuals_per_px2':(kap-slope*delta).tolist(),
            'limitations':'Descriptive demand-label association; capture/demand confounded, no independent accommodation truth.'})
    print(json.dumps({k:summary[k] for k in ('status','scheduled','complete','inverse_failures','runtime_seconds')},indent=2))


if __name__=='__main__': main()
