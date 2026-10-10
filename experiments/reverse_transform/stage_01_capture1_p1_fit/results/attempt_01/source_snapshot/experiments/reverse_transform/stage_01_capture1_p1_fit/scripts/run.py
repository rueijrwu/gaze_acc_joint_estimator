"""Fresh capture1 centroid-gaze calibration and one-magnification P1 fit."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from scipy.optimize import minimize

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
sys.path.insert(0,str(ROOT))
from distortion_model.data import load_capture,make_population,array_hash
from distortion_model.p1_shape import centered,shape_model,profile_magnification,profile_objective,recover_p1


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def statistics(values):
    a=np.asarray(values);a=a[np.isfinite(a)]
    if not len(a):return {'count':0}
    return {'count':int(a.size),'mean':float(a.mean()),'std':float(a.std()),
            'min':float(a.min()),'p05':float(np.percentile(a,5)),'median':float(np.median(a)),
            'p95':float(np.percentile(a,95)),'p99':float(np.percentile(a,99)),
            'max':float(a.max()),'rms':float(np.sqrt(np.mean(a*a)))}


def calibrate_gaze(delta,exposure,targets):
    """Quadratic conversion with equal fixation weights and exact zero mean.

    Fit frame-mean polynomial moments to the five nominal targets, rather
    than pretending polynomial(mean x) equals mean(polynomial(x)).
    """
    zero=np.flatnonzero(np.asarray(targets)==0).item()
    delta_x=delta[:,0]
    means=np.array([delta_x[exposure==k].mean() for k in range(5)])
    origin=means[zero]; scale=(means.max()-means.min())/2
    if scale<=0:raise ValueError('centroid gaze signal is degenerate')
    u=(delta_x-origin)/scale
    moments=np.array([[u[exposure==k].mean(),(u[exposure==k]**2).mean()] for k in range(5)])
    design=moments-moments[zero]
    ab=np.linalg.lstsq(design,targets,rcond=None)[0]
    c=-moments[zero]@ab
    gaze=c+ab[0]*u+ab[1]*u*u
    coefficient_raw=[float(c-ab[0]*origin/scale+ab[1]*origin**2/scale**2),
                     float(ab[0]/scale-2*ab[1]*origin/scale**2),float(ab[1]/scale**2)]
    fitted_means=np.array([gaze[exposure==k].mean() for k in range(5)])
    derivative=(ab[0]+2*ab[1]*u)/scale
    if np.min(derivative)*np.max(derivative)<=0:raise ValueError('nonmonotone gaze polynomial over observed frames')
    # User assumption: equal x/y first-order slope at the zero-gaze reference.
    slope=ab[0]/scale
    origin_y=delta[exposure==zero,1].mean()
    gaze_y=slope*(delta[:,1]-origin_y)
    return np.stack((gaze,gaze_y),axis=1),{'degree':2,'input':'mean(P4)-mean(P1), native px',
        'delta_x_origin_px':float(origin),'delta_x_unit_px':float(scale),
        'coefficients_normalized_ascending':[float(c),float(ab[0]),float(ab[1])],
        'coefficients_native_ascending_deg':[float(v) for v in coefficient_raw],
        'fixation_mean_delta_x_px':means.tolist(),'nominal_targets_deg':list(map(float,targets)),
        'fixation_mean_estimated_gaze_deg':fitted_means.tolist(),
        'fixation_mean_calibration_error_deg':(fitted_means-targets).tolist(),
        'derivative_deg_per_px_range':[float(derivative.min()),float(derivative.max())],
        'shared_first_order_slope_deg_per_px':float(slope),'delta_y_origin_px':float(origin_y),
        'vertical_mapping':'theta_y=shared_reference_slope*(delta_y-zero_fixation_mean_delta_y)',
        'fixation_mean_estimated_vertical_gaze_deg':[float(gaze_y[exposure==k].mean()) for k in range(5)],
        'zero_reference_exposure':int(zero),'scope':'fixation-label calibration, not framewise ground truth'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    protocol={'population':'capture1 only, all five reviewed intervals; complete P1/P4 needed for initial gaze',
        'previous_model':'none; no prior calibration parameters, states, templates or covariance loaded',
        'gaze':'horizontal quadratic centroid map; vertical linear centroid map with same first-order reference slope; zero fixation means x=y=0',
        'reference':'arithmetic mean of centered P1 triangles in nominal zero fixation, fixed throughout fit',
        'keystone':'sx=exp(eta_x*t_x^2-eta_y*t_y^2), sy=1/sx, denominator=1+beta_x*t_x*B_y/R_B+beta_y*t_y*B_x/R_B',
        'one_magnification':'normalize predicted centered keystone triangle RMS radius to R_B; one profiled positive M per frame contains all total scale',
        'objective':'original centered camera px SSE, equal five fixation weights; no normalization of measured data and no separate camera scale',
        'fit':'four shared 2D shape coefficients; six deterministic bounded starts, analytic profile gradient, L-BFGS-B',
        'bounds':'each anisotropic log-stretch contribution <=0.5; each perspective denominator contribution <=0.4 over observed gazes',
        'statistics':'per-fixation and five contiguous scheduled-source-row blocks; point median/p95/signed error; RMS secondary',
        'scope':'in-sample initial P1 deformation fit; no accommodation estimate, no physiological accuracy claim'}
    write(out/'protocol.json',protocol)
    sources=[Path(__file__).resolve(),ROOT/'distortion_model/p1_shape.py',ROOT/'distortion_model/data.py']
    provenance={'started_UTC':datetime.now(timezone.utc).isoformat(),'python':sys.version,'numpy':np.__version__,
        'source_sha256':{str(path.relative_to(ROOT)):digest(path) for path in sources}}
    for path in sources:
        target=out/'source_snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    interval_path=ROOT/'data/fixations/fixation_intervals.json'
    roster=json.loads(interval_path.read_text());name='capture_1_detections.pkl'
    source=next(s for s in roster['sources'] if s['capture']==name)
    data_path=ROOT/'data/detections'/name
    cap=load_capture(data_path,source['sha256'])
    intervals=[i for i in roster['fixations'] if i['capture']==name]
    if len(intervals)!=5:raise ValueError('five capture1 fixations required')
    pop=make_population({name:cap},intervals)
    for interval in intervals:
        a,b=interval['start_row'],interval['end_row_exclusive']
        assert cap.arrays['frame_index'][a]==interval['first_frame']
        assert cap.arrays['frame_index'][b-1]==interval['last_frame_inclusive']
    provenance['input_sha256']={str(path.relative_to(ROOT)):digest(path) for path in (data_path,interval_path)}
    provenance['population_semantic_hash']=array_hash(pop);write(out/'provenance.json',provenance)
    np.savez_compressed(out/'population.npz',**pop)
    index=np.flatnonzero(pop['complete_valid']);exp=pop['exposure'][index]
    p1,p4=pop['p1'][index],pop['p4'][index]
    delta=p4.mean(axis=1)-p1.mean(axis=1)
    targets=np.array([i['target_theta_deg'] for i in intervals])
    theta,polynomial=calibrate_gaze(delta,exp,targets)
    zero=polynomial['zero_reference_exposure'];x=centered(p1);b=x[exp==zero].mean(axis=0)
    radius=np.sqrt(np.mean(np.sum(b*b,axis=1)))
    # One scalar per frame is analytically profiled, never a second scale.
    counts=np.bincount(exp,minlength=5);weights=1/(15*counts[exp])
    units=np.array([10.,max(np.max(abs(theta[:,1])),.1)])
    t=theta/units
    extents=np.max(abs(t),axis=0)
    eta_bound=.5/np.maximum(extents**2,1e-12)
    beta_bound=.4/np.maximum(extents*np.max(abs(b[:,[1,0]]),axis=0)/radius,1e-12)
    bound_size=np.r_[eta_bound,beta_bound]
    bounds=[(-float(v),float(v)) for v in bound_size]
    starts=(np.array([[0,0,0,0],[.1,.1,0,0],[-.1,-.1,0,0],
                     [0,0,.1,.1],[0,0,-.1,-.1],[.1,-.1,.1,-.1]])*bound_size).tolist()
    fits=[]
    for start in starts:
        sol=minimize(profile_objective,start,args=(theta,b,x,weights,units),jac=True,method='L-BFGS-B',bounds=bounds,
            options={'maxiter':200,'ftol':1e-14,'gtol':1e-9,'maxls':40})
        fits.append({'start':start,'coefficients':sol.x.tolist(),'objective_point_px2':float(sol.fun),
            'gradient':sol.jac.tolist(),'success':bool(sol.success),'message':str(sol.message),'iterations':int(sol.nit)})
        print(json.dumps(fits[-1]),flush=True)
    best=min(fits,key=lambda f:f['objective_point_px2']);k=np.array(best['coefficients'])
    h,mu,size=shape_model(theta,b,k,gaze_units=units);M=profile_magnification(x,h)
    predicted=M[:,None,None]*h;residual=x-predicted
    recovered,inv_valid=recover_p1(p1,theta,b,k,M,gaze_units=units)
    scale_only=profile_magnification(x,np.broadcast_to(b,x.shape))
    raw_error=x-b;scale_error=x/scale_only[:,None,None]-b;inverse_error=recovered-b
    forward_scale_only=x-scale_only[:,None,None]*b
    # Algebra, profile derivatives and scale bookkeeping checks.
    test_theta=np.stack((np.linspace(theta[:,0].min(),theta[:,0].max(),101),
                         np.linspace(theta[:,1].max(),theta[:,1].min(),101)),axis=1)
    test_h,_,_=shape_model(test_theta,b,k,gaze_units=units);test_M=np.linspace(.8,1.2,101)
    clean=test_M[:,None,None]*test_h
    clean_recovered,clean_valid=recover_p1(clean,test_theta,b,k,test_M,gaze_units=units)
    derivative_h=1e-6;numeric=[]
    for j in range(4):
        kp,km=k.copy(),k.copy();kp[j]+=derivative_h;km[j]-=derivative_h
        numeric.append((profile_objective(kp,theta,b,x,weights,units)[0]-profile_objective(km,theta,b,x,weights,units)[0])/(2*derivative_h))
    grad=profile_objective(k,theta,b,x,weights,units)[1]
    hessian=np.column_stack([(profile_objective(k+np.eye(4)[j]*1e-5,theta,b,x,weights,units)[1]-
                            profile_objective(k-np.eye(4)[j]*1e-5,theta,b,x,weights,units)[1])/(2e-5) for j in range(4)])
    eig=np.linalg.eigvalsh((hessian+hessian.T)/2)
    checks={'synthetic_inverse_closure_max_px':float(np.max(abs(clean_recovered-b))),
        'shape_rms_radius_max_difference_px':float(np.max(abs(np.sqrt(np.mean(np.sum(h*h,axis=2),axis=1))-radius))),
        'profile_gradient_fd_max_error':float(np.max(abs(grad-numeric))),
        'zero_fixation_mean_gaze_xy_deg':theta[exp==zero].mean(axis=0).tolist(),
        'positive_magnifications':bool((M>0).all()),'synthetic_valid':bool(clean_valid.all()),
        'hessian_eigenvalues':eig.tolist(),'best_gradient_inf':float(np.max(abs(grad))),
        'multistart_objective_range':float(max(f['objective_point_px2'] for f in fits)-min(f['objective_point_px2'] for f in fits)),
        'coefficient_at_bound':bool(any(abs(k[j]-a)<1e-6 or abs(k[j]-c)<1e-6 for j,(a,c) in enumerate(bounds)))}
    checks['implementation_passed']=bool(checks['synthetic_inverse_closure_max_px']<1e-8 and
        checks['shape_rms_radius_max_difference_px']<1e-8 and checks['profile_gradient_fd_max_error']<1e-4 and
        max(abs(np.asarray(checks['zero_fixation_mean_gaze_xy_deg'])))<1e-10 and checks['positive_magnifications'] and checks['synthetic_valid'])
    checks['fit_stationary']=bool(checks['best_gradient_inf']<1e-6 and min(eig)>0 and not checks['coefficient_at_bound'])
    assert checks['implementation_passed'],checks
    np.savez_compressed(out/'frames.npz',population_index=index,exposure=exp,row=pop['row'][index],source_frame=pop['source_frame'][index],
        delta_p4_p1_px=delta,gaze_xy_deg=theta,gaze_units_deg=units,magnification=M,reference=b,coefficients=k,observed_centered_p1=x,
        predicted_centered_p1=predicted,forward_residual=residual,recovered_p1=recovered,inverse_valid=inv_valid,
        raw_error=raw_error,scale_only_error=scale_error,inverse_error=inverse_error)
    reports=[]
    for exposure,target in enumerate(targets):
        keep=exp==exposure;complete=index[keep];scheduled=np.flatnonzero(pop['exposure']==exposure)
        blocks=[]
        edges=np.linspace(pop['row'][scheduled].min(),pop['row'][scheduled].max()+1,6)
        for block in range(5):
            m=keep & (pop['row'][index]>=edges[block]) & (pop['row'][index]<edges[block+1])
            blocks.append({'block':block,'complete':int(m.sum()),'magnification':statistics(M[m]),
                'inverse_point_distance_px':statistics(np.linalg.norm(inverse_error[m],axis=-1)),
                'forward_mean_point_error_xy':residual[m].mean(axis=0).tolist() if m.any() else None})
        reports.append({'exposure':exposure,'nominal_gaze_deg':float(target),'scheduled':len(scheduled),'complete':int(keep.sum()),
            'gaze_x_deg':statistics(theta[keep,0]),'gaze_y_deg':statistics(theta[keep,1]),'magnification':statistics(M[keep]),
            'raw_point_distance_px':statistics(np.linalg.norm(raw_error[keep],axis=-1)),
            'scale_only_point_distance_px':statistics(np.linalg.norm(scale_error[keep],axis=-1)),
            'inverse_point_distance_px':statistics(np.linalg.norm(inverse_error[keep],axis=-1)),
            'forward_point_distance_px':statistics(np.linalg.norm(residual[keep],axis=-1)),
            'inverse_mean_point_error_xy':np.nanmean(inverse_error[keep],axis=0).tolist(),
            'inverse_domain_failures':int((~inv_valid[keep]).sum()),'blocks':blocks})
    summary={'status':'COMPLETE' if checks['fit_stationary'] else 'COMPLETE_FIT_UNCERTIFIED','capture':name,
        'previous_model_used':False,'scheduled':len(pop['row']),'complete':len(index),'unavailable':len(pop['row'])-len(index),
        'inverse_domain_failures':int((~inv_valid).sum()),'polynomial':polynomial,
        'reference_triangle_px':b.tolist(),'reference_rms_radius_px':float(radius),
        'gaze_units_deg':units.tolist(),
        'coefficients':{'scaled_eta_x':float(k[0]),'scaled_eta_y':float(k[1]),'scaled_beta_x':float(k[2]),'scaled_beta_y':float(k[3]),
                        'k_a_x_per_deg_squared':float(k[0]/units[0]**2),'k_a_y_per_deg_squared':float(k[1]/units[1]**2),
                        'k_p_x_per_deg_px':float(k[2]/(units[0]*radius)),'k_p_y_per_deg_px':float(k[3]/(units[1]*radius))},
        'bounds_scaled':bounds,'magnification':statistics(M),
        'raw_point_distance_px':statistics(np.linalg.norm(raw_error,axis=-1)),
        'scale_only_point_distance_px':statistics(np.linalg.norm(scale_error,axis=-1)),
        'inverse_point_distance_px':statistics(np.linalg.norm(inverse_error,axis=-1)),
        'forward_point_distance_px':statistics(np.linalg.norm(residual,axis=-1)),
        'equal_fixation_forward_objective_point_px2':best['objective_point_px2'],
        'equal_fixation_scale_only_objective_point_px2':float(np.sum(weights*np.sum(forward_scale_only**2,axis=(1,2)))),
        'checks':checks,'starts':fits,'intervals':reports,
        'limitations':'in-sample fixed-gaze estimate/reference; no framewise gaze truth, accommodation inference, radial term or absolute optical origin',
        'runtime_seconds':time.perf_counter()-started}
    write(out/'summary.json',summary);write(out/'verification.json',checks)
    print(json.dumps({key:value for key,value in summary.items() if key not in ('intervals','starts')},indent=2))


if __name__=='__main__':main()
