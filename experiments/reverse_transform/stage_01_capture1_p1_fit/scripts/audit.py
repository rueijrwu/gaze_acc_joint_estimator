"""Artifact, invariance and gradient audit for the fresh 2D P1 experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
sys.path.insert(0,str(ROOT))
from distortion_model.p1_shape import centered,shape_model,profile_magnification,profile_objective,recover_p1


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args();out=args.results.resolve()
    provenance=json.loads((out/'provenance.json').read_text())
    count=0
    for group in ('source_sha256','input_sha256'):
        for name,expected in provenance[group].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
            count+=1
    with np.load(out/'population.npz',allow_pickle=False) as z:pop=dict(z)
    with np.load(out/'frames.npz',allow_pickle=False) as z:rec=dict(z)
    assert np.all(pop['capture']=='capture_1_detections.pkl')
    assert np.array_equal(rec['population_index'],np.flatnonzero(pop['complete_valid']))
    assert len(np.unique(pop['source_frame']))==len(pop['source_frame'])
    exp=rec['exposure'];theta=rec['gaze_xy_deg'];reference=rec['reference'];k=rec['coefficients'];units=rec['gaze_units_deg']
    zero=np.flatnonzero([i['target_theta_deg']==0 for i in json.loads((ROOT/'data/fixations/fixation_intervals.json').read_text())['fixations'] if i['capture']=='capture_1_detections.pkl']).item()
    assert np.max(abs(reference-rec['observed_centered_p1'][exp==zero].mean(axis=0)))<1e-12
    summary=json.loads((out/'summary.json').read_text())
    radius=np.sqrt(np.mean(np.sum(reference**2,axis=1)))
    derivative=np.asarray(summary['polynomial']['coefficients_native_ascending_deg'])
    dx=rec['delta_p4_p1_px'][:,0]
    mapped=derivative[0]+derivative[1]*dx+derivative[2]*dx**2
    polynomial_error=float(np.max(abs(mapped-theta[:,0])))
    slope=summary['polynomial']['shared_first_order_slope_deg_per_px']
    mapped_y=slope*(rec['delta_p4_p1_px'][:,1]-summary['polynomial']['delta_y_origin_px'])
    assert np.max(abs(mapped_y-theta[:,1]))<1e-12
    # Check analytic shape derivatives throughout data, not just objective gradient
    # near a stationary point where errors can cancel.
    take=np.linspace(0,len(theta)-1,257,dtype=int)
    h,_,_,dh=shape_model(theta[take],reference,k,derivatives=True,gaze_units=units)
    jac_errors=[]
    for j in range(4):
        plus,minus=k.copy(),k.copy();plus[j]+=1e-6;minus[j]-=1e-6
        fd=(shape_model(theta[take],reference,plus,gaze_units=units)[0]-shape_model(theta[take],reference,minus,gaze_units=units)[0])/(2e-6)
        jac_errors.append(float(np.max(abs(fd-dh[...,j]))))
    x=rec['observed_centered_p1'];m=rec['magnification']
    transformed=centered(2*x+np.array([513.,-234.]))
    ht,_,_=shape_model(theta,reference,k,gaze_units=units)
    mt=profile_magnification(transformed,ht)
    rt,vt=recover_p1(transformed,theta,reference,k,mt,gaze_units=units)
    scale_error=float(np.max(abs(mt-2*m)))
    recovered_error=float(np.nanmax(abs(rt-rec['recovered_p1'])))
    # Profile residual Jacobian includes dM/dk, since M is derived from P1.
    _,_,_,dht=shape_model(theta,reference,k,derivatives=True,gaze_units=units)
    dm=np.sum(x[...,None]*dht,axis=(1,2))/np.sum(ht*ht,axis=(1,2))[:,None]
    jr=-m[:,None,None,None]*dht-ht[...,None]*dm[:,None,None,:]
    weights=1/(15*np.bincount(exp,minlength=5)[exp])
    gn=2*np.einsum('n,nijk,nijl->kl',weights,jr,jr)
    eig=np.linalg.eigvalsh(gn)
    cosine=float(np.max(abs(np.sum(x*ht,axis=(1,2))-m*np.sum(ht*ht,axis=(1,2)))))
    # Independently recalculate fit metric and all key reference distances.
    cost=float(np.sum(weights*np.sum((x-m[:,None,None]*ht)**2,axis=(1,2))))
    assert abs(cost-summary['equal_fixation_forward_objective_point_px2'])<1e-10
    distances=np.linalg.norm(rec['inverse_error'],axis=2)
    assert abs(np.nanmedian(distances)-summary['inverse_point_distance_px']['median'])<1e-12
    report={'passed':bool(polynomial_error<1e-10 and max(jac_errors)<1e-6 and scale_error<1e-10 and recovered_error<1e-8 and vt.all()),
        'sha256_files_checked':count,'capture1_only':True,'previous_model_used':False,
        'polynomial_native_vs_normalized_max_deg':polynomial_error,
        'shape_derivative_fd_errors':jac_errors,'combined_scale_and_translation_magnification_error':scale_error,
        'combined_scale_and_translation_recovery_max_px':recovered_error,
        'profile_scale_normal_equation_max_error':cosine,
        'profile_gauss_newton_eigenvalues':eig.tolist(),'profile_gauss_newton_rank':int((eig>eig.max()*1e-10).sum()),
        'profile_gauss_newton_condition':float(eig.max()/eig.min()) if eig.min()>0 else None,
        'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'in-sample P1 fit, fixed polynomial gaze; neither coefficient precision nor gaze truth is certified'}
    assert report['passed'],report
    (out/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
