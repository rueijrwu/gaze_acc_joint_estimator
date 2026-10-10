"""Reproduce reference recovery and verify the independent-capture contract."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
sys.path.insert(0,str(ROOT))
from distortion_model.capture_shape import CaptureShape
from distortion_model.p1_shape import centered
spec=importlib.util.spec_from_file_location('runner',Path(__file__).with_name('run.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args();folder=args.results
    summary=json.loads((folder/'summary.json').read_text())
    provenance=json.loads((folder/'provenance.json').read_text())
    for group in ('source_sha256','parent_sha256','input_sha256'):
        for name,digest in provenance[group].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
            if group=='source_sha256':
                assert hashlib.sha256((folder/'source_snapshot'/name).read_bytes()).hexdigest()==digest,name
    with np.load(folder/'frames.npz',allow_pickle=False) as z: f=dict(z)
    with np.load(folder/'population.npz',allow_pickle=False) as z: pop=dict(z)
    assert runner.array_hash(pop)==provenance['population_semantic_hash']
    assert np.array_equal(f['population_index'],np.flatnonzero(pop['complete_valid']))
    assert len(pop['row'])==summary['scheduled']
    assert len(f['population_index'])==summary['complete']
    assert int((~f['inverse_valid']).sum())==summary['inverse_failures']
    assert int(f['common_inverse_valid'].sum())==summary['common_inverse_valid']
    for kind in ('p1','p4'):
        assert np.array_equal(centered(pop[kind][f['population_index']]),f['observed_centered_'+kind])
    zero=(f['capture_index']==0)&(f['exposure']%5==2)
    for kind in ('p1','p4'):
        assert np.max(abs(f['observed_centered_'+kind][zero].mean(axis=0)-f['reference_'+kind]))<1e-10
    checks=[]
    for record in summary['captures']:
        capture=record['capture'];sel=f['capture_index']==capture-1
        fix=f['exposure'][sel]%5;counts=np.bincount(fix,minlength=5)
        weights=1/(15*counts[fix]);units=record['gaze_units_deg']
        theta=f['gaze_xy_deg'][sel];m=f['p1_magnification'][sel]
        raw=f['population_index'][sel]
        delta=pop['p4'][raw].mean(axis=1)-pop['p1'][raw].mean(axis=1)
        recalibrated,_=runner.helpers.calibrate_gaze(delta,fix,np.array([-10.,-5.,0.,5.,10.]))
        assert np.array_equal(recalibrated,theta)
        assert np.max(abs(theta[fix==2].mean(axis=0)))<1e-10
        assert np.all(m>0)
        maximum_gradient_error=0.;maximum_closure_error=0.
        certificates={}
        for kind in ('p1','p4'):
            coefficients=np.array(record[kind]['coefficients_scaled'])
            assert len(coefficients)==(5 if kind=='p4' and capture!=1 else 4)
            model=CaptureShape(theta,f['reference_'+kind],f['observed_centered_'+kind][sel],weights,units,
                               m if kind=='p4' else None)
            cost,gradient=model.evaluate(coefficients)
            assert abs(cost-record[kind]['objective_point_px2'])<1e-8
            projected=gradient.copy();active=np.zeros(len(coefficients),dtype=bool)
            if 'bounds_scaled' in record[kind]:
                bounds=np.array(record[kind]['bounds_scaled'])
                lower,upper=bounds[:,0],bounds[:,1]
                assert np.all(coefficients>=lower-1e-10) and np.all(coefficients<=upper+1e-10)
                active=(abs(coefficients-lower)<1e-7)|(abs(coefficients-upper)<1e-7)
                projected[((coefficients<=lower+1e-7)&(gradient>0))|
                          ((coefficients>=upper-1e-7)&(gradient<0))]=0
                assert active.tolist()==record[kind]['checks']['active_bounds']
            hessian=np.column_stack([(model.evaluate(coefficients+np.eye(len(coefficients))[j]*1e-5)[1]-
                                      model.evaluate(coefficients-np.eye(len(coefficients))[j]*1e-5)[1])/2e-5
                                     for j in range(len(coefficients))])
            free=np.flatnonzero(~active)
            curvature=float(np.linalg.eigvalsh(((hessian+hessian.T)/2)[np.ix_(free,free)]).min())
            assert np.max(abs(projected))<1e-6 and curvature>0,(capture,kind,projected,curvature)
            certificates[kind]={'projected_gradient_inf':float(np.max(abs(projected))),
                                'free_hessian_min_eigenvalue':curvature,'active_bounds':active.tolist()}
            numerical=np.array([(model.evaluate(coefficients+np.eye(len(coefficients))[j]*1e-6)[0]-
                                 model.evaluate(coefficients-np.eye(len(coefficients))[j]*1e-6)[0])/2e-6
                                for j in range(len(coefficients))])
            gradient_error=float(np.max(abs(numerical-gradient)))
            maximum_gradient_error=max(maximum_gradient_error,gradient_error)
            assert gradient_error<1e-4,(capture,kind,gradient_error)
            pred,mcheck=model.prediction(coefficients)
            if kind=='p1':assert np.max(abs(mcheck-m))<1e-10
            recovered,valid=model.inverse(coefficients,m)
            assert np.array_equal(valid,f['inverse_valid'][sel]) if kind=='p4' else valid.all()
            assert np.allclose(recovered,f['recovered_'+kind][sel],rtol=0,atol=1e-9,equal_nan=True)
            synthetic,sv=model.inverse(coefficients,m,observed=pred)
            closure=float(np.max(abs(synthetic-f['reference_'+kind])))
            maximum_closure_error=max(maximum_closure_error,closure)
            assert sv.all() and closure<1e-8,(capture,kind,closure)
            assert model.domain(coefficients)['valid']
            if kind=='p4':
                k,kv=model.inverse(coefficients,m,remove_barrel=False)
                assert np.array_equal(kv,f['keystone_inverse_valid'][sel])
                assert np.allclose(k,f['keystone_recovered_p4'][sel],rtol=0,atol=1e-9,equal_nan=True)
                common=valid&kv
                assert np.array_equal(common,f['common_inverse_valid'][sel])
                for name,points in [('before',f['before_p4'][sel]),('keystone',k),('recovered',recovered)]:
                    metrics=runner.errors(points[common],f['reference_p4'])
                    for metric in ('point_distance_px','shape_point_distance_px','radius_ratio'):
                        for quantile in ('median','p95'):
                            assert abs(metrics[metric][quantile]-record['p4']['errors'][name][metric][quantile])<1e-10
        checks.append({'capture':capture,'finite_difference_gradient_max_error':maximum_gradient_error,
                       'synthetic_forward_inverse_max_px':maximum_closure_error,
                       'stationary':record['p4']['checks']['stationary'],'independent_certificates':certificates})
    if len(summary['captures'])>1:
        post=json.loads((folder/'postfit.json').read_text())
        d=np.array([r['demand_diopters_label'] for r in summary['captures']]);d-=d[0]
        k=np.array([r['relative_barrel_per_px2'] for r in summary['captures']])
        assert abs(post['anchored_linear_slope_per_D_px2']-d@k/(d@d))<1e-15
    with np.load(ROOT/'experiments/reverse_transform/stage_01_capture1_p1_fit/results/attempt_01/frames.npz') as z:
        sel=f['capture_index']==0
        assert np.array_equal(z['magnification'],f['p1_magnification'][sel])
        assert np.array_equal(z['gaze_xy_deg'],f['gaze_xy_deg'][sel])
    result={'status':'PASS','checks':checks,'contract':'Independent fits, one P4 barrel per capture, fixed P1 frame scales, capture1-only reference; post-fit demand association.',
            'numerical_limits':[r['capture'] for r in summary['captures'] if not r['p4']['checks']['stationary']],
            'constrained_bound_captures':[{'capture':r['capture'],'p1':r['p1']['checks'].get('active_bounds',[]),
                                          'p4':r['p4']['checks']['active_bounds']} for r in summary['captures']
                                         if any(r['p1']['checks'].get('active_bounds',[])) or any(r['p4']['checks']['active_bounds'])]}
    (folder/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
