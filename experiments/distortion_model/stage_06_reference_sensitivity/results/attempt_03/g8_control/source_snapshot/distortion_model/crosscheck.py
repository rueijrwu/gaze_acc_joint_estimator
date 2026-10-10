"""Frozen-model omission inference and native-unit agreement summaries."""
import numpy as np
from .geometry import relative_coordinates,retained_indices
from .inference import infer_retained,infer_all
from .optics import predict_relative


def infer_omission(spec,p,p1,p4,p1_available,p4_available,covariance,held,xp=np,max_iterations=60):
    """Validity, initialization and inverse receive retained measurements only.

    The held measurement enters solely after the inverse has returned, to score
    a native-pixel prediction. A missing held measurement does not block inverse.
    """
    keep=[j for j in range(3) if j!=held]
    retained=p1_available.all(axis=1)&p4_available[:,keep].all(axis=1)
    rows=np.flatnonzero(retained);idx=retained_indices(held)
    result={'retained_input_valid':retained,'held_input_valid':p4_available[:,held].copy(),'rows':rows}
    if not len(rows):return result
    y=relative_coordinates(p1[rows],p4[rows],held_point=held)
    solved=infer_retained(spec,p,y,covariance[np.ix_(idx,idx)],held,xp=xp,max_iterations=max_iterations)
    params=spec.parameters(p);states=solved['states'];pred=np.full((len(rows),10),np.nan);g=np.full(len(rows),np.nan)
    has=np.isfinite(states).all(axis=1);optics_valid=np.zeros(len(rows),dtype=bool)
    if has.any():
        pred[has],g[has],optics_valid[has]=predict_relative(states[has,0],states[has,1],y[has,:4],covariance[:4,:4],params)
    errors=np.full((len(rows),2),np.nan)
    scoreable=p4_available[rows,held]&optics_valid
    # This is the first read of omitted coordinates anywhere in the pipeline.
    errors[scoreable]=(p4[rows[scoreable],held]-p1[rows[scoreable]].mean(axis=1)
                       -pred[scoreable,4+2*held:6+2*held])
    result.update(solution=solved,scale=g,predicted_held=pred[:,4+2*held:6+2*held],
                  error=errors,optics_valid=optics_valid,
                  scored=solved['certified']&(~solved['ambiguous'])&scoreable)
    return result


def distribution(value):
    value=np.asarray(value);value=value[np.isfinite(value)]
    if not len(value):return {'count':0,'mean':None,'median':None,'p05':None,'p95':None,'p99':None,'max':None}
    return {'count':len(value),'mean':float(value.mean()),'median':float(np.median(value)),
            'p05':float(np.quantile(value,.05)),'p95':float(np.quantile(value,.95)),
            'p99':float(np.quantile(value,.99)),'max':float(value.max())}


def errors_summary(error):
    error=np.asarray(error).reshape(-1,2);error=error[np.isfinite(error).all(axis=1)]
    if not len(error):return {'count':0,'vector_rms_px':None,'coordinate_rms_px':None,'signed_bias_px':[None,None],
                              'axis_rms_px':[None,None],'vector_distribution':distribution([])}
    return {'count':len(error),'vector_rms_px':float(np.sqrt(np.mean(np.sum(error**2,axis=1)))),
            'coordinate_rms_px':float(np.sqrt(np.mean(error**2))),'signed_bias_px':error.mean(axis=0).tolist(),
            'axis_rms_px':np.sqrt(np.mean(error**2,axis=0)).tolist(),
            'vector_distribution':distribution(np.linalg.norm(error,axis=1))}


def frame_metrics(arrays):
    states=arrays['states'];error=arrays['error'];eligible=arrays['scored'].all(axis=1)
    out={'complete_eligible_triple':eligible}
    out['E2']=np.mean(np.sum(error**2,axis=2),axis=1)
    differences=np.stack((states[:,0]-states[:,1],states[:,0]-states[:,2],states[:,1]-states[:,2]),axis=1)
    out['Gtheta2']=np.mean(differences[:,:,0]**2,axis=1);out['GA2']=np.mean(differences[:,:,1]**2,axis=1)
    for key in ('E2','Gtheta2','GA2'):out[key]=np.where(eligible,out[key],np.nan)
    return out


def equal_exposure(metrics,exposure,select):
    rows=[]
    for k in range(20):
        mask=select&(exposure==k)
        rows.append({'exposure':k,'frames':int(mask.sum()),
                     **{key:float(np.mean(metrics[key][mask])) if mask.any() else None for key in ('E2','Gtheta2','GA2')}})
    present=[row['exposure'] for row in rows if row['frames']]
    means={key:float(np.mean([row[key] for row in rows])) if len(present)==20 else None for key in ('E2','Gtheta2','GA2')}
    return {'eligible_frames':int(select.sum()),'exposures_present':present,'all_20_exposures_present':len(present)==20,
            'squared':means,'E_px':np.sqrt(means['E2']) if means['E2'] is not None else None,
            'Gtheta_deg':np.sqrt(means['Gtheta2']) if means['Gtheta2'] is not None else None,
            'GA_D':np.sqrt(means['GA2']) if means['GA2'] is not None else None,'by_exposure':rows}


def summarize(pop,arrays):
    n=len(pop['row']);metrics=frame_metrics(arrays);eligible=metrics['complete_eligible_triple'];e=pop['exposure']
    def slots(mask):
        mask=np.broadcast_to(mask,arrays['scored'].shape);score=mask&arrays['scored']
        return {'scheduled':int(mask.sum()),'retained_input_valid':int((mask&arrays['retained_input_valid']).sum()),
                'held_input_valid':int((mask&arrays['held_input_valid']).sum()),
                'certified':int((mask&arrays['certified']).sum()),'ambiguous':int((mask&arrays['ambiguous']).sum()),
                'scored':int(score.sum()),'bound_active_scored':int((score&arrays['at_bounds']).sum()),
                'status_counts':{str(k):int(v) for k,v in zip(*np.unique(arrays['status'][mask],return_counts=True))},
                'errors':errors_summary(arrays['error'][score]),
                'conditional_A_information':distribution(arrays['conditional_accommodation_information'][score]),
                'abs_column_cosine':distribution(np.abs(arrays['jacobian_column_cosine'][score])),
                'scaled_information_condition':distribution(arrays['scaled_information_condition'][score])}
    primary=equal_exposure(metrics,e,eligible)
    result={'scheduled_rows':n,'scheduled_slots':3*n,'all_slots':slots(np.ones((n,3),dtype=bool)),
            'primary_complete_triples':primary,'by_held_point':[], 'by_exposure':[], 'by_block':[], 'frame_strata':{}}
    for held in range(3):
        mask=np.zeros((n,3),dtype=bool);mask[:,held]=True
        result['by_held_point'].append({'held_point':held,**slots(mask)})
    for k in range(20):
        mask=e==k;ident=np.flatnonzero(mask)[0]
        record={'exposure':k,'capture':str(pop['capture'][ident]),'nominal_theta_deg':float(pop['target_theta_deg'][ident]),
                'demand_D':float(pop['demand_diopters'][ident]),**slots(mask[:,None])}
        result['by_exposure'].append(record)
        for block in range(5):
            bm=mask&(arrays['interval_block']==block)
            result['by_block'].append({'exposure':k,'block':block,**slots(bm[:,None])})
    anybound=arrays['at_bounds'].any(axis=1)
    strata={'exact_endpoint':arrays['endpoint'],'interior_no_exact_endpoint':~arrays['endpoint'],
            'first_or_last_5percent':arrays['boundary_band'],'central_90percent':~arrays['boundary_band'],
            'any_subset_at_bound':anybound,'no_subset_at_bound':~anybound}
    for name,mask in strata.items():
        selected=eligible&mask
        result['frame_strata'][name]={'primary_denominator_unchanged':True,'slots':slots(mask[:,None]),
             'frames':int(selected.sum()),'equal_exposure':equal_exposure(metrics,e,selected),
             **{key:distribution(np.sqrt(metrics[key][selected])) for key in ('E2','Gtheta2','GA2')},
             'pooled_frame_RMS':{key:float(np.sqrt(np.mean(metrics[key][selected]))) if selected.any() else None for key in ('E2','Gtheta2','GA2')}}
    # Exact additive attribution to the equal-exposure primary squared metrics.
    counts=np.bincount(e[eligible],minlength=20)
    result['top_contributors']={}
    if primary['all_20_exposures_present']:
        rows=np.flatnonzero(eligible)
        for key in ('E2','Gtheta2','GA2'):
            contribution=metrics[key][rows]/(20*counts[e[rows]])
            ordered=np.argsort(-contribution,kind='stable')[:20];total=primary['squared'][key];cumulative=0.;records=[]
            for rank,index in enumerate(ordered,1):
                row=rows[index];share=contribution[index]/total if total else 0.;cumulative+=share
                records.append({'rank':rank,'population_index':int(row),'capture':str(pop['capture'][row]),'exposure':int(e[row]),
                                'row':int(pop['row'][row]),'source_frame':int(pop['source_frame'][row]),
                                'frame_squared_value':float(metrics[key][row]),'contribution':float(contribution[index]),
                                'share':float(share),'cumulative_share':float(cumulative),'endpoint':bool(arrays['endpoint'][row]),
                                'boundary_band':bool(arrays['boundary_band'][row]),'bound_slots':int(arrays['at_bounds'][row].sum()),
                                'states':arrays['states'][row].tolist(),'error_xy_px':arrays['error'][row].tolist()})
            result['top_contributors'][key]=records
    result['conditioning_note']='J.T R^-1 J is information in the declared metric, not independently calibrated physical uncertainty; scaled condition uses theta range40deg and A range6D.'
    return result,metrics
