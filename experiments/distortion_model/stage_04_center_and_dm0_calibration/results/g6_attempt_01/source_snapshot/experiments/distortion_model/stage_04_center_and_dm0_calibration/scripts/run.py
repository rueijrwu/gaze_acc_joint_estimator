"""S6/G6: exact forward center block at the immutable G5 optical/state snapshot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'docs/Theory.md').is_file())
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'experiments/distortion_model/stage_03_p4_baseline_and_deformation/scripts'))
from g45_common import load_inputs, load_npz, read_model, start_output, write_json, describe
from distortion_model.centers import (CenterBlock, CenterPolynomial, center_prediction,
                                     local_coefficients, corrected_centers)
from distortion_model.geometry import relative_coordinates
from distortion_model.p1 import P1Model, evaluate_p1, domain_margins as p1_domain
from distortion_model.accommodation import shape_derivatives, domain_margins as p4_domain
from distortion_model.optics import Parameters, predict_relative
from distortion_model.data import array_hash

CONFIG = {
    'stage': 'S6/G6', 'candidate': 'DM0-M1 conditional forward D initialization',
    'visual_degree': 2, 'basis': '[1,a,t,a*t,t^2]',
    't': 'theta_visual_deg/10 degrees', 'a': '(A_D-Aref_D)/1 D',
    'sign': 'P4 minus P1', 'coefficient_axes': ['image x','image y; not vertical gaze'],
    'free': ['b0_x/y','bA_x/y','s0_x/y','sA_x/y','c2_x/y'],
    'fixed': ['G5 individual theta/A','G2 positive P1 g','both optical references and zeros',
              'G5 DM0-M1 and K4','template length/origin and camera axes'],
    'metric': 'complete ten-coordinate relative residual, frozen full raw R including cross-terms',
    'weights': '1/(20*full valid exposure count), global counts retained across chunks',
    'soft_mean_theta_scale_deg': .10, 'soft_mean_A_scale_D': .25,
    'regularization': '0.5*sum((c2/reference_P4_rms_radius_px)^2); all other D priors zero',
    'parameter_gradient_scale': 'one fixed empirical P4 RMS radius in reference pixels',
    'stationarity_threshold': 1e-6, 'normalized_data_rank_rtol': 1e-10,
    'identity_rtol': 1e-10, 'identity_atol': 1e-9,
    'chunk_size': 16384, 'workers': 4, 'plot_stride': 20,
    'temporal_penalty': 0., 'backend': 'NumPy float64; chunked moments, 10x10 linear solve',
    'before': 'optimal constant D within same declared degree-2 family; all nonconstant coefficients initially zero',
    'no_additional_center_loss': True,
    'automated_tests': 'not added or run; historical upstream test evidence is not a current G6 test result',
}


def same(actual, expected, label):
    if not np.allclose(actual, expected, rtol=CONFIG['identity_rtol'], atol=CONFIG['identity_atol'], equal_nan=True):
        raise ValueError('incompatible snapshot or bookkeeping: '+label)
    delta = np.abs(np.asarray(actual)-np.asarray(expected))
    return float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.


def verify_chain(inputs):
    parent = inputs['parent']
    while True:
        cp = json.loads((parent/'checkpoint.json').read_text())
        if 'parent_checkpoint_path' not in cp or 'parent_checkpoint_sha256' not in cp:
            break
        path = ROOT/cp['parent_checkpoint_path']
        if sha256(path.read_bytes()).hexdigest()!=cp['parent_checkpoint_sha256']:
            raise ValueError('changed checkpoint ancestor: '+str(path))
        if 'parent_summary_sha256' in cp and sha256((path.parent/'summary.json').read_bytes()).hexdigest()!=cp['parent_summary_sha256']:
            raise ValueError('changed ancestor review: '+str(path.parent))
        parent = path.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=ROOT/'experiments/distortion_model/stage_03_p4_baseline_and_deformation/results/g5_attempt_01')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    inputs = load_inputs(args.parent.resolve(), None, 'G5', 0)
    verify_chain(inputs)
    cp = inputs['checkpoint']
    prior = load_npz(inputs['parent']/cp['frame_states_file'], cp['frame_states_hash'])
    model4 = read_model(json.loads((inputs['parent']/cp['model_file']).read_text()))
    valid, p = inputs['valid'], inputs['population']
    theta, g = inputs['theta'], inputs['g']
    a = prior['accommodation_D'][valid]
    audit = {'frozen_theta_max_difference': same(theta,prior['theta_visual_deg'][valid],'theta'),
             'frozen_g_max_difference': same(g,prior['g'][valid],'g')}
    g2 = ROOT/inputs['base']['parent_checkpoint_path']
    g2cp = json.loads(g2.read_text())
    p1path = g2.parent/g2cp['p1_reference_file']
    if sha256(p1path.read_bytes()).hexdigest()!=g2cp['p1_reference_sha256']:
        raise ValueError('changed P1 reference')
    r1 = json.loads(p1path.read_text())
    model1 = P1Model(r1['b1_reference_px'],r1['omega1_visual_deg'],
                    (r1['alpha1_deg_minus2'],r1['beta1_deg_minus2'],r1['gamma1_px_minus1_deg_minus1']))
    y = relative_coordinates(p['p1'][valid],p['p4'][valid])
    opt1 = evaluate_p1(theta,y[:,:4],inputs['covariance']['R11'],model1)
    opt4 = shape_derivatives(theta,a,model4)
    if not opt1['valid'].all() or not opt4['valid'].all():
        raise ValueError('invalid frozen optical predictions')
    for name in ['g','F1','mu1']:
        audit['recomputed_'+name+'_max_difference'] = same(opt1[name], inputs['p1'][name][valid],name)
    for name in ['F4','mu4']:
        audit['recomputed_'+name+'_max_difference'] = same(opt4[name], prior[name][valid],name)
    radius = float(np.sqrt(np.mean(np.sum(model4.b4**2,axis=1))))
    config = dict(CONFIG, curvature_scale_reference_px=radius, Aref_D=model4.aref,
                  omega1_visual_deg=model1.omega1, omega4_visual_deg=model4.omega4)
    output = args.output.resolve()
    provenance = start_output(output,inputs,config,'G6',extra_sources=Path(__file__).parent.glob('*.py'))
    block = CenterBlock(theta,a,g,opt1['F1'],opt4['F4'],y,inputs['exposure'],inputs['covariance']['relative'],
                        model4.aref,degree=2,curvature_scale_px=radius,chunk_size=CONFIG['chunk_size'])
    fitted, initial, certificate = block.solve()
    nominal, demand = p['target_theta_deg'][valid], inputs['demand']
    before, initial_prediction, initial_residual = block.terms(initial.coefficients,nominal,demand)
    after, prediction, residual = block.terms(fitted.coefficients,nominal,demand)
    if after['total']>before['total']+64*np.finfo(float).eps*max(1.,before['total']):
        raise ValueError('same-objective coefficient solve increased J')
    ledger = corrected_centers(p['p1'][valid],p['p4'][valid],g,opt1['F1'],opt4['F4'])
    d = center_prediction(theta,a,fitted)
    h = d+ledger['mu4_minus_mu1_reference_px']
    audit['predicted_centroid_identity_max_difference_px'] = same(prediction[:,4:].reshape(-1,3,2).mean(axis=1),g[:,None]*h,'centroid identity')
    audit['corrected_center_sign_max_difference_reference_px'] = same((ledger['C4_hat_px']-ledger['C1_hat_px'])/g[:,None],ledger['D_obs_reference_px'],'center sign/units')
    # Re-expression of this same empirical polynomial at the frozen P4 zero.
    local = CenterPolynomial(local_coefficients(fitted,model4.omega4),model4.aref,degree=2)
    audit['polynomial_origin_conversion_max_difference_reference_px'] = same(center_prediction(theta-model4.omega4,a,local),d,'polynomial origin conversion')
    unified = Parameters(model1.b1,model4.b4,model1.omega1,model4.omega4,model1.k1,model4.k4,model4.aref,model4.m1,
                         np.vstack((fitted.coefficients,np.zeros((1,2)))))
    generic, generic_g, generic_valid = predict_relative(theta,a,y[:,:4],inputs['covariance']['R11'],unified)
    if not generic_valid.all():raise ValueError('generic forward model invalid')
    audit['unified_forward_max_difference_px'] = same(prediction,generic,'generic forward adapter')
    audit['unified_P1_g_max_difference'] = same(g,generic_g,'generic P1 profile')
    write_json(output/'fit.json',dict(certificate=certificate,before=before,after=after,
                                     initial_coefficients_reference_px=initial.coefficients.tolist(),
                                     gradient=(block.hessian@fitted.coefficients.ravel()-block.rhs).tolist()))
    write_json(output/'bookkeeping.json',dict(audit=audit,p1_domain=p1_domain(model1),p4_domain=p4_domain(model4),
              historical_tests='G5 tests.json has 55 historical passes; not rerun for G6',
              current_evidence='all-frame empirical forward/centroid/sign/origin identities and exact coefficient stationarity'))
    record = {'schema':'distortion_dual_alignment_v1','stage':'S6/G6','mechanism':'DM0-M1',
              'baseline':'empirical_incremental','gaze_polynomial_target':'center_separation','visual_degree':2,
              'center_basis':'[1,(A-Aref)/1D,theta/10deg,(A-Aref)/1D*theta/10deg,(theta/10deg)^2]',
              'center_coefficients_reference_px':fitted.coefficients.tolist(),'coefficient_order':certificate['coefficient_order'],
              'center_c3_reference_px':[0.,0.],'b1_reference_px':model1.b1.tolist(),'b4_reference_px':model4.b4.tolist(),
              'omega1_visual_deg':model1.omega1,'omega4_visual_deg':model4.omega4,'Delta14_deg':model4.omega4-model1.omega1,
              'k1_native':list(model1.k1),'k4_native':list(model4.k4),'Aref_D':model4.aref,'m1_per_D':model4.m1,
              'theta_bounds_deg':list(model4.theta_bounds),'a_bounds_D':list(model4.a_bounds),
              'fixed_free_roster':{'fixed':CONFIG['fixed'],'free':CONFIG['free']},'scale_policy':'p1_profile_v1',
              'sign':'P4-P1','gauges':'fixed empirical template centroid origins/lengths/camera axes; physical origins unknown',
              'covariance_policy':'unchanged full raw relative covariance, not plug-in residual covariance',
              'source_hash':provenance['source_hash'],'population_hash':inputs['base']['population_hash'],
              'scope':'center block initializer at frozen states/optics; full joint fit and physiological accuracy unresolved'}
    write_json(output/'model.json',record)
    n = len(valid)
    def extend(x):
        result = np.full((n,)+x.shape[1:],np.nan);result[valid]=x;return result
    arrays = {'input_valid':valid,'center_block_evaluated':valid.copy(),'theta_visual_deg':prior['theta_visual_deg'],
              'accommodation_D':prior['accommodation_D'],'g':prior['g'],
              'status':np.where(valid,'evaluated_fixed_state','input_unavailable'),
              'D_prediction_reference_px':extend(d),'centroid_prediction_px':extend(g[:,None]*h),
              'centroid_residual_px':extend(ledger['raw_centroid_separation_px']-g[:,None]*h),
              'relative_observed_px':extend(y),'relative_prediction_px':extend(prediction),'relative_residual_px':extend(residual),
              'initial_relative_prediction_px':extend(initial_prediction),'initial_relative_residual_px':extend(initial_residual),
              'F1':extend(opt1['F1']),'F4':prior['F4'],'mu1':extend(opt1['mu1']),'mu4':prior['mu4']}
    arrays.update({key:extend(value) for key,value in ledger.items()})
    np.savez_compressed(output/'fitted.npz',**arrays)
    p1_residual = p['p1'][valid]-p['p1'][valid].mean(axis=1,keepdims=True)-g[:,None,None]*(opt1['F1']-opt1['mu1'][:,None,:])
    def metric(rows):
        idx = rows[valid[rows]]
        if not len(idx):return {'scheduled':len(rows),'valid':0,'unavailable':len(rows)}
        r = arrays['relative_residual_px'][idx]
        return {'scheduled':len(rows),'valid':len(idx),'evaluated':len(idx),'unavailable':len(rows)-len(idx),
                'point_cost_contribution':float(.5*np.sum((r@block.precision)*r)/(20*np.sum(inputs['exposure']==p['exposure'][rows[0]]))),
                'relative_coordinate_rms_px':float(np.sqrt(np.mean(r*r))),
                'initial_relative_coordinate_rms_px':float(np.sqrt(np.mean(arrays['initial_relative_residual_px'][idx]**2))),
                'P4_relative_point_rms_px':float(np.sqrt(np.mean(r[:,4:]**2))),
                'signed_relative_means_px':np.mean(r,axis=0).tolist(),
                'centroid_rms_px':float(np.sqrt(np.mean(arrays['centroid_residual_px'][idx]**2))),
                'mean_theta_deg':float(np.mean(arrays['theta_visual_deg'][idx])),
                'mean_A_D':float(np.mean(arrays['accommodation_D'][idx]))}
    def exposure_record(e):
        rows=np.flatnonzero(p['exposure']==e)
        result={'exposure':e,'capture':str(p['capture'][rows[0]]),'target_theta_deg':float(p['target_theta_deg'][rows[0]]),
                'demand_D':float(p['demand_diopters'][rows[0]]),**metric(rows)}
        result['blocks']=[{'block':j,**metric(part)} for j,part in enumerate(np.array_split(rows,5))]
        result['mean_correction_ledger']={key:np.mean(arrays[key][rows[valid[rows]]],axis=0).tolist()
                                         for key in list(ledger)+['D_prediction_reference_px']}
        result['representative_rows']=[]
        for row in rows[np.linspace(0,len(rows)-1,5,dtype=int)]:
            representative={'row':int(p['row'][row]),'source_frame':int(p['source_frame'][row]),'status':str(arrays['status'][row])}
            if valid[row]:
                representative.update({'g':float(arrays['g'][row]),**{key:arrays[key][row].tolist() for key in ledger},
                                       'D_prediction_reference_px':arrays['D_prediction_reference_px'][row].tolist()})
            result['representative_rows'].append(representative)
        return result
    with ThreadPoolExecutor(max_workers=CONFIG['workers']) as pool:
        conditions=list(pool.map(exposure_record,range(20)))
    write_json(output/'conditions.json',conditions)
    for kind in ['components','residuals']:
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for e,condition in enumerate(conditions):
            select=inputs['exposure']==e
            for axis in range(2):
                x=theta[select][::CONFIG['plot_stride']]
                values=ledger['D_obs_reference_px'][select,axis] if kind=='components' else residual[select,4:].reshape(-1,3,2).mean(axis=1)[:,axis]
                axes[axis].scatter(x,values[::CONFIG['plot_stride']],s=2,alpha=.25,label=f"c{1+e//5}, demand {condition['demand_D']:.2f} D" if e%5==0 else None)
                if kind=='components':
                    order=np.argsort(theta[select]);axes[axis].plot(theta[select][order][::20],d[select,axis][order][::20],lw=.8)
                axes[axis].set(xlabel='Frozen individual visual theta (degrees)',ylabel=('Corrected D (reference px)' if kind=='components' else 'Native centroid residual (px)'),title=f'{kind}: image '+['x','y'][axis])
        axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(output/f'{kind}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for axis in range(2):
        for cap in range(4):
            subset=conditions[cap*5:(cap+1)*5]
            axes[axis].plot([r['target_theta_deg'] for r in subset],
                           [r['mean_correction_ledger']['mu4_minus_mu1_reference_px'][axis] for r in subset],
                           'o-',label=f"capture {cap+1}, demand {subset[0]['demand_D']:.2f} D")
        axes[axis].set(xlabel='Nominal visual target (degrees)',ylabel='Mean mu4-mu1 (reference px)',title='Optical mean correction: image '+['x','y'][axis]);axes[axis].legend(fontsize=7)
    fig.tight_layout();fig.savefig(output/'optical_mean_corrections.png',dpi=150);plt.close(fig)
    checkpoint={'schema':'g6_center_checkpoint_v1','source_commit':provenance['source_commit'],'source_hash':provenance['source_hash'],
                'parent_checkpoint_path':str((inputs['parent']/'checkpoint.json').relative_to(ROOT)),
                'parent_checkpoint_sha256':provenance['parent_checkpoint_sha256'],'parent_summary_sha256':provenance['parent_summary_sha256'],
                'base_g3_checkpoint_path':str(inputs['basepath'].relative_to(ROOT)),
                'base_g3_checkpoint_sha256':sha256(inputs['basepath'].read_bytes()).hexdigest(),
                'model_file':'model.json','model_sha256':sha256((output/'model.json').read_bytes()).hexdigest(),
                'frame_states_file':'fitted.npz','frame_states_hash':array_hash(arrays),
                'center_block_certified':certificate['linear_block_certified'],
                'fit_complete':False,'fit_certified':False,'crosscheck_complete':False,'comparison_complete':False}
    write_json(output/'checkpoint.json',checkpoint)
    aggregate=lambda name:float(np.sqrt(np.mean([r[name]**2 for r in conditions])))
    summary={'stage':'S6/G6','status':'COMPLETE_PENDING_REVIEW','G6_decision':'none','source_hash':provenance['source_hash'],
             'scheduled':n,'complete_valid':int(valid.sum()),'unavailable':int((~valid).sum()),'evaluated_fixed_states':int(valid.sum()),
             'center_degree':2,'center_coefficients_reference_px':fitted.coefficients.tolist(),'before':before,'after':after,
             'before_equal_exposure_relative_rms_px':aggregate('initial_relative_coordinate_rms_px'),
             'after_equal_exposure_relative_rms_px':aggregate('relative_coordinate_rms_px'),
             'after_equal_exposure_centroid_rms_px':aggregate('centroid_rms_px'),
             'P1_centered_point_rms_px':float(np.sqrt(np.mean(p1_residual**2))),
             'center_block_certified':certificate['linear_block_certified'],'data_rank':certificate['data_rank'],
             'scaled_gradient_inf':certificate['scaled_gradient_inf'],'audit':audit,'elapsed_seconds':time.perf_counter()-started,
             'automated_tests_run':False,'fit_complete':False,'fit_certified':False,'crosscheck_complete':False,'comparison_complete':False}
    write_json(output/'summary.json',summary)
    (output/'STAGE_REPORT.md').write_text('# G6 pending execution-owner review\n')
    (output/'PROGRESS.md').write_text('One next action: review G6 center-block evidence.\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
