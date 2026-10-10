"""S4/G4 conditional DM0-M1 near-reference accommodation initialization."""
import argparse
from pathlib import Path
import time
import numpy as np
from g45_common import *
from distortion_model.accommodation import ShapeProfile,fit_shape,scalar_projection,DM0Shape
from distortion_model.p4 import DualZeros,near_reference

CONFIG={'stage':'S4/G4','candidate':'DM0-M1','data_population':'fixed G3 |xi4|<=2.5 degrees; full schedule preserved',
        'metric':'native four P4 edge marginal, equal capture weights in near window; full fixation means anchored equally across 20 exposures',
        'soft_mean_A_scale_D':.25,'A_bounds_D':[0.,6.],'theta_policy':'frozen G2 per-frame visual theta',
        'free':['one m1','individual near-window A'], 'fixed':['empirical centered reference from low-demand selected exposure near frames','K4=identity','G3 zeros','G2 P1 g','outside-window provisional demand A'],
        'reference_policy':'native empirical shape; no radial coefficient or P4 nuisance scale',
        'state_target_gradient_inf':1e-11,'state_certificate_gradient_inf':1e-8,'global_certificate_gradient_inf':1e-6,
        'optimizer':'profiled L-BFGS-B, ftol1e-14 gtol1e-8 maxiter100; at most5 bounded stationarity-polish steps',
        'sensitivity':['one doubled soft-mean anchor scale .50 D','one G3 adjacent operational reference'],
        'temporal_penalty':0.,'backend':'vectorized NumPy CPU'}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent',type=Path,default=ROOT/'experiments/distortion_model/stage_03_p4_baseline_and_deformation/results/attempt_01')
    parser.add_argument('--test-report',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();started=time.perf_counter()
    inputs=load_inputs(args.parent.resolve(),args.test_report,'G3',50);output=args.output.resolve();provenance=start_output(output,inputs,CONFIG,'G4')
    p=inputs['population'];reference=inputs['reference'];near=inputs['diagnostics']['near_reference_diagnostic'][inputs['valid']]
    template_mask=near&(inputs['exposure']==reference['reference_exposure']);template=inputs['diagnostics']['corrected_centered_p4_px'][inputs['valid']][template_mask].mean(axis=0)
    if template_mask.sum()<2:raise ValueError('no empirical reference frames')
    aref=reference['Aref_demand_D'];omega=reference['omega4_visual_deg'];groups=p['capture'][inputs['valid']]
    def factory(b=template,o=omega,mask=near,anchor=.25):
        return ShapeProfile(inputs['theta'],inputs['observed_edges'],inputs['g'],inputs['exposure'],inputs['demand'],inputs['initial_a'],b,o,aref,inputs['shape_covariance'],
                            data_indices=np.flatnonzero(mask),data_groups=groups,s_a=anchor)
    corrected=inputs['diagnostics']['corrected_centered_p4_px'][inputs['valid']];rho,direction=scalar_projection(corrected,template)
    measured=[]
    for cap in sorted(set(groups)):
        select=(groups==cap)&near
        measured.append({'capture':cap,'demand_D':float(inputs['demand'][select][0]),'count':int(select.sum()),'rho':describe(rho[select]),
                         'directional_rms_reference_px':float(np.sqrt(np.mean(direction[select]**2))),
                         'signed_directional_mean_reference_px':direction[select].mean(axis=0).tolist(),
                         'theta_deg':describe(inputs['theta'][select]),'xi4_deg':describe(inputs['theta'][select]-omega)})
    slope=np.dot(np.array([r['demand_D']-aref for r in measured]),np.array([r['rho']['mean']-1 for r in measured]))/sum((r['demand_D']-aref)**2 for r in measured)
    write_json(output/'measured_response.json',{'measured_before_fitted_A':True,'template_frame_count':int(template_mask.sum()),'template_px':template.tolist(),'captures':measured,
                                              'descriptive_demand_projection_slope_per_D':float(slope),'projection_kind':'unweighted shape diagnostic, not an inference nuisance scale'})
    print('Fitting conditional G4 slope and near-window states',flush=True)
    start=10*slope
    model,a,best,outcomes=fit_shape(factory,[[start],[start*1.5]])
    write_json(output/'fit.json',{'best':best,'outcomes':outcomes})
    if not best['conditional_fit_certified']:
        write_json(output/'summary.json',{'status':'COMPLETE_UNCERTIFIED','G4_decision':'none','best':best,'source_hash':provenance['source_hash']});print('Conditional fit uncertified; preserved without handoff',flush=True);return
    identity=DM0Shape(template,omega,aref,0.);identity_arrays,identity_records=save_snapshot(output,'identity',inputs,identity,inputs['initial_a'],np.zeros(len(a),bool))
    arrays,records=save_snapshot(output,'fitted',inputs,model,a,near);save_checkpoint(output,inputs,model,arrays,provenance,'G4')
    sensitivity=[]
    for label,b,o,mask,anchor in [('anchor_0.50_D',template,omega,near,.5)]:
        print('Running '+label,flush=True)
        sm,sa,sbest,sout=fit_shape(lambda:factory(b,o,mask,anchor),[[10*model.m1],[start]])
        sensitivity.append({'label':label,'s_A_D':anchor,'model':model_json(sm),'best':sbest,'outcomes':sout,'A_change_D':describe(sa-a)})
    for alt in reference['bounded_adjacent_alternatives'][:1]:
        o=alt['omega4_visual_deg'];mask=near_reference(inputs['theta'],DualZeros(reference['omega1_visual_deg'],o))
        refmask=mask&(inputs['exposure']==alt['exposure']);b=corrected[refmask].mean(axis=0)
        if refmask.sum()<2:raise ValueError('no measured adjacent-reference template')
        print('Running bounded reference alternative',flush=True)
        sm,sa,sbest,sout=fit_shape(lambda:factory(b,o,mask,.25),[[start],[start*1.5]])
        sensitivity.append({'label':'adjacent_G3_reference','omega4_visual_deg':o,'template_count':int(refmask.sum()),'near_count':int(mask.sum()),
                            'model':model_json(sm),'best':sbest,'outcomes':sout,'A_change_D':describe(sa-a),
                            'comparison':'different initialization window/template; descriptive sensitivity, not matched score comparison'})
    write_json(output/'sensitivity.json',sensitivity)
    residual_plot(output,identity_records,records,'G4 near-window initialization; outside-window A remains provisional')
    fig,axes=plt.subplots(1,2,figsize=(12,5));demand=[r['demand_D'] for r in measured]
    axes[0].errorbar(demand,[r['rho']['mean'] for r in measured],yerr=[r['rho']['std'] for r in measured],fmt='o',label='Measured projection mean ± SD')
    grid=np.linspace(0,6,100);axes[0].plot(grid,1+model.m1*(grid-aref),label='Provisional M at A=demand (display only)')
    axes[0].set(xlabel='Recorded demand label (D)',ylabel='Measured projection / provisional M');axes[0].legend(fontsize=8)
    axes[1].bar(demand,[r['directional_rms_reference_px'] for r in measured],width=.15);axes[1].set(xlabel='Recorded demand label (D)',ylabel='Remaining directional RMS (reference px)')
    fig.tight_layout();fig.savefig(output/'measured_response.png',dpi=150);plt.close(fig)
    raw_before=identity_arrays['edge_residual_px'][inputs['valid']][near];raw_after=arrays['edge_residual_px'][inputs['valid']][near]
    summary={'stage':'S4/G4','status':'COMPLETE_PENDING_REVIEW','G4_decision':'none','source_hash':provenance['source_hash'],
             'scheduled':len(inputs['valid']),'complete_valid':int(inputs['valid'].sum()),'unavailable':int((~inputs['valid']).sum()),'near_A_solved':int(near.sum()),
             'outside_A_provisional':int((~near).sum()),'m1_per_D':model.m1,'conditional_fit_certified':best['conditional_fit_certified'],
             'conditional_rank':best['conditional_rank'],'scaled_projected_gradient_inf':best['scaled_projected_gradient_inf'],'state_projected_gradient_inf':best['state_projected_gradient_inf'],
             'near_native_edge_rms_before_px':float(np.sqrt(np.mean(raw_before**2))),'near_native_edge_rms_after_px':float(np.sqrt(np.mean(raw_after**2))),
             'near_A_D':describe(a[near]),'near_A_at_bounds':int(((a[near]<=1e-9)|(a[near]>=6-1e-9)).sum()),
             'domain':best['domain'],'elapsed_seconds':time.perf_counter()-started,'tests':{k:inputs['tests'][k] for k in ['passed','failed','errors','skipped']},
             'fit_complete':False,'fit_certified':False,'crosscheck_complete':False,'comparison_complete':False}
    write_json(output/'summary.json',summary);(output/'STAGE_REPORT.md').write_text('# G4 pending execution-owner review\n');(output/'PROGRESS.md').write_text('One next action: G4 evidence review.\n')
    print(summary,flush=True)


if __name__=='__main__':main()
