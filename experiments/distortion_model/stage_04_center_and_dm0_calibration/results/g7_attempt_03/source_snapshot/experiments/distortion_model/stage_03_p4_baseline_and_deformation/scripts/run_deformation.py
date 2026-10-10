"""S5/G5 conditional composed P4 keystone and DM0-M1 refinement."""
import argparse
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from g45_common import *
from distortion_model.accommodation import ShapeProfile,fit_shape,from_scaled,parameter_bounds,domain_margins

CONFIG={'stage':'S5/G5','candidate':'DM0-M1 with A-independent K4 coefficients',
        'population':'all complete frames in all twenty full reviewed exposures; schedule/slots unchanged',
        'metric':'native four P4 edges, frozen T R T-transpose, equal exposure weighting; full fixation means anchored equally',
        'soft_mean_A_scale_D':.25,'A_bounds_D':[0.,6.],'theta_policy':'frozen G2 framewise visual theta; P1 g unchanged',
        'free':['m1','alpha4','beta4','gamma4','individual A for all complete frames'],
        'fixed':['empirical G4 template length/origin','G3 zeros and axes','G2 P1 reference/g','DM0-M1 family, no radial term'],
        'order':'seed K4 at frozen G4 m/A; then conditional joint m/K4/A refinement; revisit original G4 window using each xi4',
        'comparison':'refit identity K4 and full K4 on same frames, covariance, weights, A anchors and bounds',
        'state_target_gradient_inf':1e-11,'state_certificate_gradient_inf':1e-8,'global_certificate_gradient_inf':1e-6,
        'optimizer':'profiled L-BFGS-B ftol1e-14 gtol1e-8 maxiter100; at most5 bounded same-objective stationarity-polish steps',
        'domain':'positive M and x/y scales; denominator bounded on full shifted visual/A rectangle',
        'starts':'fixed-G4-state keystone seed and zero keystone; each uses G4 m and starting A',
        'temporal_penalty':0.,'backend':'vectorized NumPy float64 CPU'}


def seed_keystone(inputs,model,initial_a):
    e=inputs['exposure'];counts=np.bincount(e);weights=1/(len(counts)*counts[e]);precision=np.linalg.solve(inputs['shape_covariance'],np.eye(4))
    lower,upper=parameter_bounds(model.b4,model.omega4,model.aref,True);m=10*model.m1
    def objective(k):
        trial=from_scaled(model.b4,model.omega4,model.aref,np.r_[m,k],True)
        opt=shape_derivatives(inputs['theta'],initial_a,trial);r=inputs['observed_edges']-inputs['g'][:,None]*opt['edges'];wr=r@precision.T
        jac=inputs['g'][:,None,None]*opt['dglobal'][...,1:]
        return float(.5*np.sum(weights*np.sum(r*wr,axis=1))),-np.einsum('n,nip,ni->p',weights,jac,wr)
    result=minimize(objective,np.zeros(3),jac=True,method='L-BFGS-B',bounds=list(zip(lower[1:],upper[1:])),options={'ftol':1e-14,'gtol':1e-8,'maxiter':100,'maxls':30})
    cost,gradient=objective(result.x)
    return np.r_[m,result.x],{'kind':'initialization seed only, frozen G4 A/m','scaled_parameters':np.r_[m,result.x].tolist(),
                             'cost':cost,'gradient':gradient.tolist(),'optimizer_success':bool(result.success),'message':str(result.message),'nfev':int(result.nfev)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',type=Path,default=ROOT/'experiments/distortion_model/stage_03_p4_baseline_and_deformation/results/g4_attempt_01')
    parser.add_argument('--test-report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();started=time.perf_counter()
    inputs=load_inputs(args.parent.resolve(),args.test_report,'G4',55);output=args.output.resolve();provenance=start_output(output,inputs,CONFIG,'G5')
    prior=read_model(json.loads((inputs['parent']/'model.json').read_text()));prior_arrays=load_npz(inputs['parent']/inputs['checkpoint']['frame_states_file'],inputs['checkpoint']['frame_states_hash'])
    initial_a=prior_arrays['accommodation_D'][inputs['valid']]
    def factory(deformation=False):
        return ShapeProfile(inputs['theta'],inputs['observed_edges'],inputs['g'],inputs['exposure'],inputs['demand'],initial_a,
                            prior.b4,prior.omega4,prior.aref,inputs['shape_covariance'],deformation=deformation)
    print('Fitting matched full-population identity-K4 baseline',flush=True)
    identity,identity_a,identity_best,identity_outcomes=fit_shape(lambda:factory(False),[[10*prior.m1],[15*prior.m1]])
    write_json(output/'identity_fit.json',{'best':identity_best,'outcomes':identity_outcomes})
    if not identity_best['conditional_fit_certified']:
        write_json(output/'summary.json',{'status':'COMPLETE_UNCERTIFIED','G5_decision':'none','failed_component':'identity baseline','best':identity_best,'source_hash':provenance['source_hash']});print('Identity baseline uncertified; no handoff',flush=True);return
    print('Seeding native-angle K4 at fixed G4 states, then refining all A and m/K4',flush=True)
    seed,seed_record=seed_keystone(inputs,prior,initial_a);write_json(output/'keystone_seed.json',seed_record)
    model,a,best,outcomes=fit_shape(lambda:factory(True),[seed.tolist(),[10*prior.m1,0.,0.,0.]])
    write_json(output/'fit.json',{'best':best,'outcomes':outcomes})
    if not best['conditional_fit_certified']:
        write_json(output/'summary.json',{'status':'COMPLETE_UNCERTIFIED','G5_decision':'none','failed_component':'deformation','best':best,'source_hash':provenance['source_hash']});print('Composed shape fit uncertified; no handoff',flush=True);return
    mask=np.ones(len(a),dtype=bool)
    identity_arrays,identity_records=save_snapshot(output,'identity',inputs,identity,identity_a,mask)
    arrays,records=save_snapshot(output,'fitted',inputs,model,a,mask);save_checkpoint(output,inputs,model,arrays,provenance,'G5')
    residual_plot(output,identity_records,records,'G5 matched conditional P4 shape fit; visual theta and P1 scale frozen')
    near=inputs['diagnostics']['near_reference_diagnostic'][inputs['valid']];revisit=[]
    for cap in sorted(set(inputs['population']['capture'][inputs['valid']])):
        select=near&(inputs['population']['capture'][inputs['valid']]==cap)
        revisit.append({'capture':cap,'count':int(select.sum()),'xi4_deg':describe(inputs['theta'][select]-model.omega4),
                        'G4_A_D':describe(initial_a[select]),'G5_A_D':describe(a[select]),
                        'G4_native_edge_rms_px':float(np.sqrt(np.mean(prior_arrays['edge_residual_px'][inputs['valid']][select]**2))),
                        'G5_native_edge_rms_px':float(np.sqrt(np.mean(arrays['edge_residual_px'][inputs['valid']][select]**2)))})
    write_json(output/'near_reference_revisit.json',{'original_G3_window_unchanged':True,'captures':revisit,'G4_m1_per_D':prior.m1,'G5_m1_per_D':model.m1,
                                                   'unknown_still':['physical P4 zero/source symmetry/spatial origin','capture-demand confound','P1 reference sensitivity','approximate visual theta'],
                                                   'resolved_for_initialization':['use individual xi4 for every near-reference frame','include M inside K4 denominator','update one DM0 M and all frame A under full mean anchors']})
    n=len(inputs['valid']);m=1+model.m1*(a-model.aref);opt=shape_derivatives(inputs['theta'],a,model)
    baseline=m[:,None,None]*model.b4;rotation=opt['F4']-baseline
    def extend(x):
        result=np.full((n,)+x.shape[1:],np.nan);result[inputs['valid']]=x;return result
    contributions={'magnification_point_change_reference_px':extend(baseline-model.b4),'keystone_point_change_reference_px':extend(rotation),
                   'mu4_reference_px':arrays['mu4'],'mu1_reference_px':inputs['p1']['mu1'],
                   'D_obs_reference_px':inputs['diagnostics']['corrected_centroid_displacement_px']-arrays['mu4']+inputs['p1']['mu1'],
                   'input_valid':inputs['valid']}
    np.savez_compressed(output/'contributions_and_centers.npz',**contributions)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for e,r in enumerate(records):
        select=inputs['exposure']==e;native=inputs['indices'][select];rows=inputs['population']['row'][native]
        xi=inputs['theta'][select]-model.omega4
        axes[0].scatter(xi[::10],np.sqrt(np.mean(rotation[select][::10]**2,axis=(1,2))),s=2,alpha=.3)
        axes[1].scatter(inputs['g'][select][::10],np.sqrt(np.mean(arrays['edge_residual_px'][native][::10]**2,axis=1)),s=2,alpha=.3)
    axes[0].set(xlabel='Individual xi4 (degrees)',ylabel='Keystone point change RMS (reference px)',title='Composition contribution, no inverse keystone')
    axes[1].set(xlabel='Frozen P1 g',ylabel='Native edge residual RMS (px)',title='Residual dependence on shared scale remains for G8')
    fig.tight_layout();fig.savefig(output/'composition_and_scale.png',dpi=150);plt.close(fig)
    def aggregate(rs):return float(np.sqrt(np.mean([r['edge_rms_px']**2 for r in rs])))
    summary={'stage':'S5/G5','status':'COMPLETE_PENDING_REVIEW','G5_decision':'none','source_hash':provenance['source_hash'],
             'scheduled':len(inputs['valid']),'complete_valid':int(inputs['valid'].sum()),'unavailable':int((~inputs['valid']).sum()),'conditional_A_solved':len(a),
             'identity_m1_per_D':identity.m1,'m1_per_D':model.m1,'k4_native':list(model.k4),
             'identity_cost':identity_best['cost'],'composed_cost':best['cost'],'identity_equal_exposure_native_edge_rms_px':aggregate(identity_records),
             'composed_equal_exposure_native_edge_rms_px':aggregate(records),'near_native_edge_rms_px':float(np.sqrt(np.mean(arrays['edge_residual_px'][inputs['valid']][near]**2))),
             'conditional_fit_certified':best['conditional_fit_certified'],'conditional_rank':best['conditional_rank'],
             'scaled_projected_gradient_inf':best['scaled_projected_gradient_inf'],'state_projected_gradient_inf':best['state_projected_gradient_inf'],
             'A_at_bounds':int(((a<=1e-9)|(a>=6-1e-9)).sum()),'A_D':describe(a),'domain':best['domain'],
             'elapsed_seconds':time.perf_counter()-started,'tests':{k:inputs['tests'][k] for k in ['passed','failed','errors','skipped']},
             'fit_complete':False,'fit_certified':False,'crosscheck_complete':False,'comparison_complete':False}
    write_json(output/'summary.json',summary);(output/'STAGE_REPORT.md').write_text('# G5 pending execution-owner review\n');(output/'PROGRESS.md').write_text('One next action: G5 evidence review.\n');print(summary,flush=True)


if __name__=='__main__':main()
