"""Interpret saved G8 inversions, measurement geometry and conditioning."""
from hashlib import sha256
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).parent))
from run import ROOT,load_frozen,write,digest
from distortion_model.crosscheck import distribution,errors_summary
from distortion_model.geometry import relative_coordinates,retained_indices
from distortion_model.joint import JointDM0
from distortion_model.optics import predict_relative


def load(path):
    with np.load(path,allow_pickle=False) as z:return dict(z)


def paired_state_summary(delta,exposure,mask):
    by=[]
    for k in range(20):
        selected=mask&(exposure==k)
        by.append({'exposure':k,'frames':int(selected.sum()),
                   'theta_squared_mean':float(np.mean(delta[selected,0]**2)) if selected.any() else None,
                   'A_squared_mean':float(np.mean(delta[selected,1]**2)) if selected.any() else None,
                   'signed_mean_deg_D':delta[selected].mean(axis=0).tolist() if selected.any() else [None,None]})
    complete=all(row['frames'] for row in by)
    return {'matched_frames':int(mask.sum()),'all20exposures_present':complete,
            'equal_exposure_theta_RMS_deg':float(np.sqrt(np.mean([row['theta_squared_mean'] for row in by]))) if complete else None,
            'equal_exposure_A_RMS_D':float(np.sqrt(np.mean([row['A_squared_mean'] for row in by]))) if complete else None,
            'abs_theta_difference_deg':distribution(np.abs(delta[mask,0])),
            'abs_A_difference_D':distribution(np.abs(delta[mask,1])),
            'by_exposure':by}


def conditional(h):return h[:,1,1]-h[:,0,1]**2/h[:,0,0]
def signed_area(points):
    u=points[:,1]-points[:,0];v=points[:,2]-points[:,0]
    return .5*(u[:,0]*v[:,1]-u[:,1]*v[:,0])
def correlation(x,y):
    good=np.isfinite(x)&np.isfinite(y);x=x[good];y=y[good]
    if len(x)<3 or np.std(x)==0 or np.std(y)==0:return None
    return float(np.corrcoef(x,y)[0,1])
def fmt(value):return 'unavailable' if value is None else f'{value:.6g}'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--attempt',type=Path,required=True)
    parser.add_argument('--parent',type=Path,default=ROOT/'experiments/distortion_model/stage_04_center_and_dm0_calibration/results/g7_attempt_04')
    args=parser.parse_args();attempt=args.attempt.resolve();inputs,spec,p,cal,checkpoint=load_frozen(args.parent.resolve())
    if not json.loads((attempt/'independent_audit.json').read_text())['passed']:raise ValueError('independent audit required')
    pop=inputs['population'];r=inputs['covariance']['relative'];out=load(attempt/'crosscheck.npz');diag=load(attempt/'diagnostic_states.npz')
    metrics=json.loads((attempt/'metrics.json').read_text());indices=diag['population_index'];e=diag['exposure']
    frame_scores=load(attempt/'frame_metrics.npz');eligible=frame_scores['complete_eligible_triple']
    frame_distributions={key:distribution(np.sqrt(frame_scores[key][eligible])) for key in ('E2','Gtheta2','GA2')}
    write(attempt/'native_frame_distributions.json',{
        'population':'all complete eligible triples; no outcome-dependent exclusion',
        'weighting':'Pooled frame quantiles of sqrt(per-frame squared mean). Primary RMS instead uses twenty equal exposure weights.',
        'per_frame_E_px':frame_distributions['E2'],'per_frame_Gtheta_deg':frame_distributions['Gtheta2'],
        'per_frame_GA_D':frame_distributions['GA2']})
    np.savez_compressed(attempt/'slot_context.npz',
        population_index=np.arange(len(pop['row'])),held_point=out['held_point'],
        calibration_status=np.full(out['held_point'].shape,'certified_frozen_global_model'),
        calibration_frame_available=np.repeat(cal['input_valid'][:,None],3,axis=1),
        calibration_checkpoint_sha256=np.array(digest(args.parent.resolve()/'checkpoint.json')))
    valid=diag['input_valid']&diag['all3_certified']&(~diag['all3_ambiguous'])
    comparison={'schedule_rows':len(indices),'all3_available':int(diag['input_valid'].sum()),
                'all3_certified':int(diag['all3_certified'].sum()),'all3_ambiguous':int(diag['all3_ambiguous'].sum()),
                'all3_bounds':int(diag['all3_at_bounds'].sum()),'mask':'same scheduled timestamps, no outcome-driven trimming',
                'all3_minus_calibration':paired_state_summary(diag['all3_states']-diag['calibration_states'],e,valid),
                'each_omission_minus_all3':[],'same_state_retained_information':[]}
    for held in range(3):
        mask=valid&diag['omission_scored'][:,held]
        comparison['each_omission_minus_all3'].append({'held_point':held,
            **paired_state_summary(diag['omission_states'][:,held]-diag['all3_states'],e,mask)})
    schedule=load(attempt/'diagnostic_schedule.npz')
    comparison['schedule_partitions']={}
    for name,member in (('original_compact',schedule['legacy_compact_member']),
                        ('predeclared_neighbors',schedule['neighbor_case_member']),
                        ('outside_neighbor_case',~schedule['neighbor_case_member'])):
        mask=valid&member
        comparison['schedule_partitions'][name]={
            'scheduled_rows':int(member.sum()),
            'all3_minus_calibration':paired_state_summary(diag['all3_states']-diag['calibration_states'],e,mask),
            'each_omission_minus_all3':[{'held_point':held,**paired_state_summary(
                diag['omission_states'][:,held]-diag['all3_states'],e,mask&diag['omission_scored'][:,held])}
                for held in range(3)]}
    rows=indices[valid];states=diag['all3_states'][valid];y=relative_coordinates(pop['p1'][rows],pop['p4'][rows])
    objective=JointDM0(spec,y,np.zeros(len(y),dtype=int),r,np.zeros((1,2)),xp=np)
    prediction,jac,_,ok,_=objective.batch(states,p,slice(0,len(y)),False)
    assert ok.all()
    full_information=np.einsum('nis,ij,njq->nsq',jac,np.linalg.solve(r,np.eye(10)),jac)
    full_conditional=conditional(full_information)
    information_arrays={'population_index':rows,'all3_information':full_information,'all3_conditional_A_information':full_conditional}
    for held in range(3):
        idx=retained_indices(held);js=jac[:,idx];rr=r[np.ix_(idx,idx)]
        info=np.einsum('nis,ij,njq->nsq',js,np.linalg.solve(rr,np.eye(8)),js)
        ci=conditional(info);ratio=ci/full_conditional
        if not (np.isfinite(ratio).all() and np.all(ratio>=0) and np.all(ratio<=1+1e-10)):
            raise ValueError('Marginal information fraction must lie in [0,1]')
        information_arrays[f'held_{held}_information_at_all3_state']=info
        information_arrays[f'held_{held}_conditional_A_fraction']=ratio
        comparison['same_state_retained_information'].append({'held_point':held,'frames':len(rows),
            'conditional_A_information':distribution(ci),'fraction_of_all3_conditional_A_information':distribution(ratio),
            'abs_column_cosine':distribution(np.abs(info[:,0,1]/np.sqrt(info[:,0,0]*info[:,1,1])))})
    information_arrays['information_interpretation']=np.array('declared R metric; not independently calibrated localization covariance')
    np.savez_compressed(attempt/'diagnostic_information.npz',**information_arrays)
    anchored=diag['calibration_states'][valid]
    cpred,_,cok=predict_relative(anchored[:,0],anchored[:,1],y[:,:4],r[:4,:4],spec.parameters(p))
    assert cok.all();precision=np.linalg.solve(r,np.eye(10))
    cr=y-cpred;ur=y-prediction
    anchored_cost=.5*np.einsum('ni,ij,nj->n',cr,precision,cr);unanchored_cost=.5*np.einsum('ni,ij,nj->n',ur,precision,ur)
    comparison['all3_point_cost_minus_anchored_point_cost']=distribution(unanchored_cost-anchored_cost)
    comparison['all3_point_cost_higher_count']=int(np.sum(unanchored_cost>anchored_cost+1e-7))
    comparison['note']='Calibration has full-exposure mean anchors. All application inversions use identical frozen globals and only measured coordinates, with no labels or state warm starts.'
    write(attempt/'state_comparison.json',comparison)

    # Native geometry, correction ledger and descriptive scale associations.
    c1=pop['p1'].mean(axis=1);c4=pop['p4'].mean(axis=1);raw_center=c4-c1
    r1=np.sqrt(np.mean(np.sum((pop['p1']-c1[:,None])**2,axis=2),axis=1))
    r4=np.sqrt(np.mean(np.sum((pop['p4']-c4[:,None])**2,axis=2),axis=1))
    a4=signed_area(pop['p4']);predarea=cal['g']**2*signed_area(cal['F4'])
    area_conflict=pop['complete_valid']&(a4*predarea<0)
    raw={'P1_centroid_px':c1,'P4_centroid_px':c4,'centroid_separation_px':raw_center,
         'raw_P1_radius_px':r1,'raw_P4_radius_px':r4,'raw_P4_signed_area_px2':a4,'calibration_predicted_P4_signed_area_px2':predarea,
         'P4_area_orientation_conflict':area_conflict,'calibration_scale':cal['g'],
         'calibration_model_origin1_px':c1-cal['g'][:,None]*cal['mu1'],
         'calibration_model_origin4_px':c4-cal['g'][:,None]*cal['mu4'],
         'calibration_origin_separation_prediction_px':cal['g'][:,None]*cal['D'],
         'calibration_optical_mean_correction_px':cal['g'][:,None]*(cal['mu4']-cal['mu1']),
         'calibration_centroid_residual_px':raw_center-cal['g'][:,None]*(cal['D']+cal['mu4']-cal['mu1'])}
    np.savez_compressed(attempt/'raw_geometry.npz',**raw)
    geometry={'complete_measurements':int(pop['complete_valid'].sum()),'P4_orientation_conflicts':int(area_conflict.sum()),
              'P4_measured_signed_area':distribution(a4[pop['complete_valid']]),
              'orientation_interpretation':'Positive g and M and positive keystone domain margins preserve template triangle orientation. A measured sign conflict can reflect noise, detector behavior or correspondence; it is not by itself proof of a source swap.',
              'by_exposure':[],'scale_error_associations':[],'gaze_bins':[]}
    for k in range(20):
        mask=(pop['exposure']==k)&pop['complete_valid'];ids=np.flatnonzero(mask)
        geometry['by_exposure'].append({'exposure':k,'capture':str(pop['capture'][ids[0]]),
             'nominal_theta_deg':float(pop['target_theta_deg'][ids[0]]),'demand_D':float(pop['demand_diopters'][ids[0]]),
             'complete_rows':len(ids),'P4_orientation_conflicts':int(area_conflict[mask].sum()),
             'centroid_separation_mean_px':raw_center[mask].mean(axis=0).tolist(),
             'mean_optical_correction_px':raw['calibration_optical_mean_correction_px'][mask].mean(axis=0).tolist(),
             'mean_gD_px':raw['calibration_origin_separation_prediction_px'][mask].mean(axis=0).tolist(),
             'centroid_residual_px':errors_summary(raw['calibration_centroid_residual_px'][mask]),
             'raw_P4_radius_px':distribution(r4[mask]),'P4_radius_divided_by_P1_g':distribution(r4[mask]/cal['g'][mask]),
             'P1_g':distribution(cal['g'][mask])})
        for held in range(3):
            scored=(pop['exposure']==k)&out['scored'][:,held];xx=out['states'][scored,held];g=out['g'][scored,held]
            error=out['error'][scored,held];design=np.column_stack((np.ones(len(xx)),xx))
            gres=g-design@np.linalg.lstsq(design,g,rcond=None)[0]
            eres=error-design@np.linalg.lstsq(design,error,rcond=None)[0]
            geometry['scale_error_associations'].append({'exposure':k,'held_point':held,'scored':int(scored.sum()),
                'raw_g_error_xy_correlation':[correlation(g,error[:,j]) for j in range(2)],
                'g_error_xy_after_linear_theta_A_adjustment':[correlation(gres,eres[:,j]) for j in range(2)]})
        bins=np.floor(cal['theta_visual_deg'][ids]/2).astype(int)
        for bin_ in np.unique(bins):
            selected=ids[bins==bin_]
            geometry['gaze_bins'].append({'exposure':k,'calibration_theta_bin_deg':[int(2*bin_),int(2*bin_+2)],
                'rows':len(selected),'g_mean':float(np.mean(cal['g'][selected])),
                'raw_P4_radius_mean_px':float(np.mean(r4[selected])),
                'centroid_separation_mean_px':raw_center[selected].mean(axis=0).tolist()})
    geometry['association_note']='Descriptive within-exposure correlations; derived g, theta and A share measurements. Adjustment and model-based gaze bins do not establish a causal scale-transfer effect.'
    write(attempt/'measurement_diagnostics.json',geometry)
    # Joint point/exposure/block summaries from saved inversions only. Preserve
    # the original full-run metrics and its archived source unchanged.
    point_groups=[]
    for k in range(20):
        for block in [None,*range(5)]:
            group=pop['exposure']==k
            if block is not None:group&=out['interval_block']==block
            for held in range(3):
                scored=group&out['scored'][:,held]
                point_groups.append({'exposure':k,'block':block,'held_point':held,
                    'scheduled':int(group.sum()),
                    'retained_input_valid':int((group&out['retained_input_valid'][:,held]).sum()),
                    'certified':int((group&out['certified'][:,held]).sum()),
                    'scored':int(scored.sum()),
                    'bound_active_scored':int((scored&out['at_bounds'][:,held]).sum()),
                    'status_counts':{str(key):int(value) for key,value in zip(*np.unique(out['status'][group,held],return_counts=True))},
                    'errors':errors_summary(out['error'][scored,held]),
                    'conditional_A_information':distribution(out['conditional_accommodation_information'][scored,held])})
    write(attempt/'point_exposure_diagnostics.json',{'weighting':'Each group is pooled within its explicitly declared exposure/block and omitted point. Primary equal-exposure metrics unchanged.',
          'block_definition':'five contiguous equal fractions of each original scheduled exposure, including endpoints',
          'groups':point_groups})
    neighbors=np.flatnonzero((pop['capture']=='capture_2_detections.pkl')&(pop['row']>=6794)&(pop['row']<=6805))
    case=[]
    for row in neighbors:
        record={'population_index':int(row),'row':int(pop['row'][row]),'source_frame':int(pop['source_frame'][row]),
                'exposure':int(pop['exposure'][row]),'p1_available':pop['p1_available'][row].tolist(),
                'p4_available':pop['p4_available'][row].tolist(),'p1_xy_px':pop['p1'][row].tolist(),
                'p4_relative_to_P1_centroid_px':(pop['p4'][row]-c1[row]).tolist(),
                'P4_signed_area_px2':float(a4[row]),'calibration_predicted_signed_area_px2':float(predarea[row]),
                'orientation_conflict':bool(area_conflict[row]),'calibration_state_deg_D':[float(cal['theta_visual_deg'][row]),float(cal['accommodation_D'][row])],
                'omission_states_deg_D':out['states'][row].tolist(),'held_errors_xy_px':out['error'][row].tolist(),
                'omission_status':out['status'][row].tolist()}
        case.append(record)
    # Every selected neighborhood row is complete in this reviewed dataset.
    write(attempt/'neighbor_case.json',{'selection':'capture2rows6794..6805;predeclaredfrompreviousaudit','rows':case})

    primary=metrics['primary_complete_triples'];fig,axes=plt.subplots(1,3,figsize=(16,4))
    for axis,key,label in zip(axes,('E2','Gtheta2','GA2'),('Held-P4 vector RMS (px)','Gaze disagreement (deg)','Accommodation disagreement (D)')):
        axis.plot(range(20),[np.sqrt(row[key]) if row[key] is not None else np.nan for row in primary['by_exposure']],'o-')
        axis.set(xlabel='Exposure index',ylabel=label);axis.grid(alpha=.25)
    fig.tight_layout();fig.savefig(attempt/'exposure_scorecard.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,7));nn=pop['row'][neighbors]
    for held in range(3):
        relative=pop['p4'][neighbors,held]-c1[neighbors]
        axes[0,0].plot(nn,relative[:,0],'o-',label=f'P4 {held}');axes[0,1].plot(nn,relative[:,1],'o-')
        axes[1,1].plot(nn,out['states'][neighbors,held,1],'o-',label=f'omit {held}')
    axes[1,0].plot(nn,a4[neighbors],'o-',label='measured');axes[1,0].plot(nn,predarea[neighbors],'o-',label='calibration predicted')
    for axis,label in zip(axes.flat,('P4 minus P1 centroid: x (px)','P4 minus P1 centroid: y (px)','P4 signed area (px²)','Subset accommodation (D)')):
        axis.set(xlabel='Capture2 source row',ylabel=label);axis.grid(alpha=.25)
    axes[0,0].legend();axes[1,0].legend();axes[1,1].legend();fig.tight_layout();fig.savefig(attempt/'neighbor_geometry.png',dpi=150);plt.close(fig)
    review=json.loads((attempt/'scientific_review.json').read_text()) if (attempt/'scientific_review.json').exists() else None
    status=f"**Status: {review['status']} / {review['G8_decision']}.** See [scientific review](SCIENTIFIC_REVIEW.md)." if review else '**Status: computed and independently reconstructed; pending root scientific review.**'
    lines=['# G8 full frozen-model evaluation','', status,'',
           f"Primary complete triples: {primary['eligible_frames']} across {len(primary['exposures_present'])}/20 exposures. All {metrics['scheduled_slots']} scheduled slots are retained.",'',
           '| Metric | Equal-exposure result |','|---|---:|',
           f"| E, held-P4 vector RMS | {fmt(primary['E_px'])} px |",f"| Gtheta, same-frame state disagreement | {fmt(primary['Gtheta_deg'])} deg |",f"| GA, same-frame state disagreement | {fmt(primary['GA_D'])} D |",'',
           'Pooled per-frame medians and tails are saved separately in [native_frame_distributions.json](native_frame_distributions.json); they use pooled frame weighting, whereas the primary RMS uses equal exposure weighting.','',
           '## Coverage and native errors by omitted point','', '| Omitted P4 | Scheduled | Inferred | Scored | Bound scored | Vector median / p95 / p99 (px) | x / y bias (px) | x / y RMS (px) |',
           '|---:|---:|---:|---:|---:|---|---|---|']
    for row in metrics['by_held_point']:
        error=row['errors'];dist=error['vector_distribution'];lines.append(f"| {row['held_point']} | {row['scheduled']} | {row['retained_input_valid']} | {row['scored']} | {row['bound_active_scored']} | {fmt(dist['median'])} / {fmt(dist['p95'])} / {fmt(dist['p99'])} | {' / '.join(fmt(v) for v in error['signed_bias_px'])} | {' / '.join(fmt(v) for v in error['axis_rms_px'])} |")
    lines+=['','## Same-timestamp application comparison','',f"The predeclared schedule has {len(indices)} rows; {int(valid.sum())} all-three inverses are certified and unambiguous. Calibration states have full-exposure anchors; application inversions are label-free.",'',
            '| State comparison | Matched rows | Equal-exposure theta RMS difference (deg) | Equal-exposure A RMS difference (D) |','|---|---:|---:|---:|']
    row=comparison['all3_minus_calibration'];lines.append(f"| All-three minus calibration | {row['matched_frames']} | {fmt(row['equal_exposure_theta_RMS_deg'])} | {fmt(row['equal_exposure_A_RMS_D'])} |")
    for row in comparison['each_omission_minus_all3']:lines.append(f"| Omit {row['held_point']} minus all-three | {row['matched_frames']} | {fmt(row['equal_exposure_theta_RMS_deg'])} | {fmt(row['equal_exposure_A_RMS_D'])} |")
    lines+=['','## Retained-pair conditioning at the same all-three state','',
            'Ratios compare conditional accommodation information after allowing gaze, using the declared covariance marginals. They describe the weighting metric and do not establish physical precision.','',
            '| Omitted point | Conditional A information fraction: median / p05 | Absolute column cosine median / p95 |','|---:|---|---|']
    for row in comparison['same_state_retained_information']:
        ratio=row['fraction_of_all3_conditional_A_information'];cosine=row['abs_column_cosine']
        lines.append(f"| {row['held_point']} | {fmt(ratio['median'])} / {fmt(ratio['p05'])} | {fmt(cosine['median'])} / {fmt(cosine['p95'])} |")
    lines+=['','## Measurement diagnostics','',f"{geometry['P4_orientation_conflicts']} of {geometry['complete_measurements']} complete measured P4 triangles have a signed-area orientation different from the frozen positive-domain model. Inspect measurement noise, detection and correspondence before attributing this to optical coefficients.",'',
            'The predeclared capture2 neighborhood is saved in [neighbor_case.json](neighbor_case.json) and plotted below. All original endpoints remain in the primary evaluation.','',
            '![Neighbor geometry](neighbor_geometry.png)','', '![Exposure scorecard](exposure_scorecard.png)','',
            'Joint point/axis/exposure/block counts, signed errors and tails are in [point_exposure_diagnostics.json](point_exposure_diagnostics.json). Boundary/bound strata, independent E²/Gtheta²/GA² contribution ranks and conditioning distributions are in [metrics.json](metrics.json). Raw centroids, scale, model origins and distortion corrections are in [raw_geometry.npz](raw_geometry.npz); condition and scale associations are in [measurement_diagnostics.json](measurement_diagnostics.json).','',
            '## Interpretation limits','',
            'This is internal measurement agreement on recordings used for calibration. Physical optical zeros, absolute accommodation calibration, capture/demand confounding and P1/P4 scale transfer remain unresolved. Rank two and positive local curvature provide numerical eligibility; they do not validate physiological accuracy. No fixed RMS ceiling is used.','',
            'The all-three comparison is a diagnostic on a predeclared subset that deliberately includes the known capture2 neighborhood. It is not a representative full-period sample. Schedule partitions, including rows outside that neighborhood, are reported separately in [state_comparison.json](state_comparison.json). The whole-period primary metrics use all eligible full-schedule triples. The earlier compact attempt03 score is a different population and is not an improvement baseline.']
    (attempt/'REPORT.md').write_text('\n'.join(lines)+'\n')
    post={'kind':'saved-result interpretation; no optimization or new inference','report_script_sha256':digest(Path(__file__)),
          'core_inference_source_sha256':digest(ROOT/'distortion_model/inference.py'),'input_crosscheck_sha256':digest(attempt/'crosscheck.npz'),
          'diagnostic_states_sha256':digest(attempt/'diagnostic_states.npz')}
    target=attempt/'postprocessing_source/report.py';target.parent.mkdir(exist_ok=True);shutil.copyfile(Path(__file__),target)
    write(attempt/'postprocessing_provenance.json',post)
    print(json.dumps({'primary':{key:primary[key] for key in ('eligible_frames','E_px','Gtheta_deg','GA_D')},
                      'state_comparison':comparison,'P4_orientation_conflicts':geometry['P4_orientation_conflicts']},indent=2),flush=True)


if __name__=='__main__':main()
