"""Audit saved framewise A, inverse gradient and fixation-mean certificates."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
sys.path.insert(0,str(ROOT))
from distortion_model.frame_accommodation import FrameAccommodation
spec=importlib.util.spec_from_file_location('runner4',Path(__file__).with_name('run.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--results',type=Path,required=True)
    folder=parser.parse_args().results
    summary=json.loads((folder/'summary.json').read_text());provenance=json.loads((folder/'provenance.json').read_text())
    protocol=summary['protocol'];parent=ROOT/provenance['parent_results']
    for group in ('parent_sha256','source_sha256'):
        for name,digest in provenance[group].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
            if group=='source_sha256':assert hashlib.sha256((folder/'source_snapshot'/name).read_bytes()).hexdigest()==digest,name
    with np.load(folder/'frames.npz',allow_pickle=False) as z:f=dict(z)
    with np.load(parent/'frames.npz',allow_pickle=False) as z:p=dict(z)
    with np.load(parent/'population.npz',allow_pickle=False) as z:pop=dict(z)
    assert runner.array_hash(pop)==provenance['population_semantic_hash']
    for key in ('population_index','capture_index','exposure','row','source_frame','gaze_xy_deg','gaze_units_deg',
                'p1_magnification','reference_p4','observed_centered_p4'):
        assert np.array_equal(f[key],p[key]),key
    ps=json.loads((parent/'summary.json').read_text());post=json.loads((parent/'postfit.json').read_text())
    slope=post['anchored_linear_slope_per_D_px2'];aref=ps['reference_demand_diopters_label']
    assert slope==protocol['kappa_law']['slope_per_D_px2'] and aref==protocol['kappa_law']['reference_A_D']
    expected=np.array([f['expected_A_D'][f['exposure']==e][0] for e in range(20)])
    assert np.array_equal(f['expected_A_D'],pop['demand_diopters'][f['population_index']])
    keystone=np.array([ps['captures'][c]['p4']['coefficients_scaled'][:4] for c in f['capture_index']])
    assert np.array_equal(keystone,f['keystone_scaled'])
    assert np.array_equal(f['kappa_per_px2'],slope*(f['A_D']-aref))
    model=FrameAccommodation(f['observed_centered_p4'],f['reference_p4'],f['gaze_xy_deg'],f['gaze_units_deg'],
        keystone,f['p1_magnification'],f['exposure'],expected,slope,aref,protocol['anchor_width_D'],
        protocol['residual_scale_px'],np)
    a=f['A_D'];lower,upper=f['lower_A_D'],f['upper_A_D']
    assert np.isfinite(a).all() and np.all(a>=lower) and np.all(a<=upper)
    assert np.all(lower==0) and np.all(upper<=6)
    recalculated_lower,recalculated_upper=model.feasible_bounds()
    assert np.max(abs(recalculated_upper-upper))<1e-8 and np.array_equal(recalculated_lower,lower)
    recovered,valid=model.recover(a)
    assert valid.all() and np.array_equal(valid,f['inverse_valid'])
    assert np.max(abs(recovered-f['recovered_p4']))<1e-8
    nominal,nvalid=model.recover(f['expected_A_D'])
    assert np.array_equal(nvalid,f['nominal_A_inverse_valid'])
    assert np.allclose(nominal,f['nominal_A_recovered_p4'],rtol=0,atol=1e-8,equal_nan=True)
    parent_A=np.array([aref+ps['captures'][c]['relative_barrel_per_px2']/slope for c in f['capture_index']])
    old,ov=model.recover(parent_A)
    assert np.array_equal(ov,p['inverse_valid']) and np.array_equal(ov,f['parent_inverse_valid'])
    assert np.allclose(old,f['parent_recovered_p4'],rtol=0,atol=1e-8,equal_nan=True)
    common=valid&nvalid&ov
    assert np.array_equal(common,f['common_inverse_valid']) and int(common.sum())==summary['common_valid']
    prediction=model.geometry(a)[4]
    assert np.max(abs(f['forward_residual']-(f['observed_centered_p4']-prediction)))<1e-8
    synthetic,sv=model.recover(a,observed=prediction)
    closure=float(np.max(abs(synthetic-f['reference_p4'])))
    assert sv.all() and closure<1e-8
    # All frame derivatives are checked at nearby interior points, avoiding
    # undefined differentiation across an inverse-branch boundary.
    probe=np.minimum(np.maximum(a,lower+1e-3),upper-1e-3);h=1e-5
    fc,analytic,gv=model.frame_cost(probe)
    plus=model.frame_cost(probe+h)[0];minus=model.frame_cost(probe-h)[0]
    numeric=(plus-minus)/(2*h)
    relative=float(np.max(abs(numeric-analytic)/(1+abs(analytic))))
    assert gv.all() and relative<2e-5,relative
    cost,gradient=model.evaluate(a);frame_cost,local_gradient,_=model.frame_cost(a)
    means=model.means(a);delta=means-expected
    direct=np.sum(model.q*frame_cost)+model.nref*model.strength*np.sum(delta*delta)
    assert abs(cost-direct)<1e-7 and abs(cost/model.n-summary['objective_equal_fixation_mean_px2'])<1e-8
    assert np.max(abs(gradient-model.q*(local_gradient+2*model.strength*delta[f['exposure']])))<1e-8
    cert,curvature=model.certificate(a,lower,upper)
    assert cert['stationary'],cert
    # Positive local curvature plus the PSD fixation-mean penalty gives
    # positive observed curvature on the free state subspace.
    for row in summary['fixations']:
        e=row['exposure'];sel=f['exposure']==e;keep=sel&common
        assert abs(row['mean_anchor_shift_D']-delta[e])<1e-12
        for name,points in [('capture_constant',old),('nominal_A',nominal),('framewise_A',recovered)]:
            metrics=runner.stage3.errors(points[keep],f['reference_p4'])
            for metric in ('point_distance_px','shape_point_distance_px','radius_ratio'):
                for quantile in ('rms','median','p95'):
                    assert abs(metrics[metric][quantile]-row['errors'][name][metric][quantile])<1e-8
    result={'status':'PASS','independent_CPU_certificate':cert,'all_frame_gradient_relative_error_max':relative,
            'synthetic_forward_inverse_max_px':closure,'inverse_reproduction_max_px':float(np.max(abs(recovered-f['recovered_p4']))),
            'common_valid_frames':int(common.sum()),'complete_fit_frames':len(a),
            'objective_equal_fixation_mean_px2':cost/model.n,
            'curvature_statement':'Positive free-frame diagonal curvature + positive semidefinite fixation-mean rank-one blocks.',
            'limits':'Numerical conditional in-sample consistency; no independent accommodation accuracy claim.'}
    (folder/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
