"""Compare saved certified reference candidates on exact common native identities."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).parent))
from run import ROOT,write,digest,load_npz
from distortion_model.crosscheck import errors_summary,distribution


def scores(a):
    d=np.stack([a['states'][:,i]-a['states'][:,j] for i,j in [(0,1),(0,2),(1,2)]],axis=1)
    return np.stack([np.mean(np.sum(a['error']**2,axis=2),axis=1),np.mean(d[:,:,0]**2,axis=1),np.mean(d[:,:,1]**2,axis=1)],axis=1)


def metric(s,e,mask):
    rows=[{'exposure':k,'frames':int(np.sum(mask&(e==k))),
           'squared':s[mask&(e==k)].mean(axis=0).tolist() if np.any(mask&(e==k)) else [None]*3} for k in range(20)]
    complete=all(r['frames'] for r in rows)
    squared=np.mean([r['squared'] for r in rows],axis=0) if complete else None
    return {'frames':int(mask.sum()),'all20exposures_present':complete,'by_exposure':rows,
        'squared':squared.tolist() if complete else None,'E_Gtheta_GA':np.sqrt(squared).tolist() if complete else None,
        'pooled_frame_magnitudes':[distribution(np.sqrt(s[mask,j])) for j in range(3)]}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--attempt',type=Path,required=True);args=parser.parse_args();output=args.attempt.resolve()
    pop=load_npz(output/'population.npz');e=pop['exposure'];candidates={}
    summaries={label:json.loads((output/label/'summary.json').read_text()) for label in ['control','adjacent']}
    if not all(record['fit_certified'] for record in summaries.values()):
        write(output/'comparison.json',{'status':'INCOMPLETE_NUMERICAL_CERTIFICATION','comparison_complete':False,
            'candidates':{name:{key:record[key] for key in ['fit_complete','fit_certified','selected_start','certified_start_count']} for name,record in summaries.items()},
            'script_sha256':digest(Path(__file__)),'protocol_sha256':digest(output/'protocol.json'),
            'candidate_checkpoint_sha256':{name:digest(output/name/'checkpoint.json') for name in summaries},
            'scheduled_rows':len(e),'scheduled_slots':3*len(e),'paired_metrics':None,'no_optical_ranking':True,
            'reason':'Uncertified model cannot enter held-point inference or optical ranking.'})
        return
    for label in ['control','adjacent']:
        folder=output/('g8_'+label)
        if not json.loads((folder/'independent_audit.json').read_text())['passed']:raise ValueError('independent G8 audit required')
        a=load_npz(folder/'crosscheck.npz')
        for key in ['capture','exposure','row','source_frame']:
            if not np.array_equal(a[key],pop[key]):raise ValueError('mismatched identity: '+key)
        if not np.array_equal(a['held_point'],np.tile(np.arange(3),(len(e),1))):raise ValueError('held-point identity mismatch')
        candidates[label]=a
    a,b=candidates['control'],candidates['adjacent'];sa,sb=scores(a),scores(b)
    eligible_a=a['scored'].all(axis=1);eligible_b=b['scored'].all(axis=1);common=eligible_a&eligible_b
    result={'kind':'one matched operational-reference experiment; no independent physiological accuracy',
        'scheduled_rows':len(e),'scheduled_slots':3*len(e),'comparison_complete':bool(np.all(np.bincount(e[common],minlength=20)>0)),
        'common_frames':int(common.sum()),'common_fraction_of_scheduled':float(common.mean()),
        'control_eligible_only':int((eligible_a&~eligible_b).sum()),'adjacent_eligible_only':int((eligible_b&~eligible_a).sum()),
        'all_scheduled_coverage':{label:json.loads((output/('g8_'+label)/'metrics.json').read_text())['all_slots'] for label in candidates},
        'candidate_specific':{'control':metric(sa,e,eligible_a),'adjacent':metric(sb,e,eligible_b)},
        'paired':{'control':metric(sa,e,common),'adjacent':metric(sb,e,common)},'paired_exposure_changes':[],
        'paired_point_exposure_blocks':[],'conditioning':{},'strata':{},'top_squared_changes':{}}
    for k in range(20):
        mask=common&(e==k);ds=(sb[mask]-sa[mask]).mean(axis=0) if mask.any() else np.full(3,np.nan)
        result['paired_exposure_changes'].append({'exposure':k,'frames':int(mask.sum()),
            'adjacent_minus_control_squared':ds.tolist() if mask.any() else [None]*3,
            'control_bound_slots':int(a['at_bounds'][mask].sum()),'adjacent_bound_slots':int(b['at_bounds'][mask].sum())})
        for block in [None,*range(5)]:
            group=mask.copy()
            if block is not None:group&=a['interval_block']==block
            for held in range(3):
                result['paired_point_exposure_blocks'].append({'exposure':k,'block':block,'held_point':held,'common_frames':int(group.sum()),
                    'control':errors_summary(a['error'][group,held]),'adjacent':errors_summary(b['error'][group,held])})
    for held in range(3):
        result['conditioning'][str(held)]={label:{'conditional_A_information':distribution(arr['conditional_accommodation_information'][common,held]),
            'abs_column_cosine':distribution(np.abs(arr['jacobian_column_cosine'][common,held])),
            'scaled_condition':distribution(arr['scaled_information_condition'][common,held]),'bound_slots':int(arr['at_bounds'][common,held].sum())}
            for label,arr in candidates.items()}
    neighbor=(pop['capture']=='capture_2_detections.pkl')&(pop['row']>=6794)&(pop['row']<=6805)
    for name,mask in [('outside_predeclared_neighbor_case',~neighbor),('central90percent',~a['boundary_band']),
                      ('both_no_subset_bound',~a['at_bounds'].any(axis=1)&~b['at_bounds'].any(axis=1))]:
        paired=common&mask;result['strata'][name]={'primary_denominator_unchanged':True,'control':metric(sa,e,paired),'adjacent':metric(sb,e,paired)}
    rows=np.flatnonzero(common);counts=np.bincount(e[common],minlength=20)
    if result['comparison_complete']:
        delta=(sb[rows]-sa[rows])/(20*counts[e[rows],None])
        for j,key in enumerate(['E2','Gtheta2','GA2']):
            summed=float(delta[:,j].sum());expected=result['paired']['adjacent']['squared'][j]-result['paired']['control']['squared'][j]
            if not np.isclose(summed,expected,rtol=1e-12,atol=1e-10):raise ValueError('squared change attribution mismatch')
            result['top_squared_changes'][key]={'total_adjacent_minus_control':summed,'largest_improvements':[],'largest_worsenings':[]}
            for key_order,order in [('largest_improvements',np.argsort(delta[:,j],kind='stable')[:20]),('largest_worsenings',np.argsort(-delta[:,j],kind='stable')[:20])]:
                for ix in order:
                    row=rows[ix];result['top_squared_changes'][key][key_order].append({'population_index':int(row),'capture':str(pop['capture'][row]),'row':int(pop['row'][row]),'source_frame':int(pop['source_frame'][row]),'exposure':int(e[row]),'signed_squared_contribution':float(delta[ix,j]),'in_neighbor_case':bool(neighbor[row])})
    np.savez_compressed(output/'paired_scores.npz',population_index=np.arange(len(e)),common_eligible=common,control_squared=sa,adjacent_squared=sb)
    write(output/'comparison.json',result)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for label,values in result['paired'].items():
        for j,axis in enumerate(axes):axis.plot(range(20),[np.sqrt(r['squared'][j]) if r['frames'] else np.nan for r in values['by_exposure']],'o-',label=label)
    for axis,title in zip(axes,['Held-point E (px)','Gaze disagreement (deg)','Accommodation disagreement (D)']):axis.set(xlabel='Exposure index',ylabel=title);axis.grid(alpha=.25);axis.legend()
    fig.tight_layout();fig.savefig(output/'paired_exposure_scores.png',dpi=150);plt.close(fig)
    write(output/'comparison_provenance.json',{'script_sha256':digest(Path(__file__)),'protocol_sha256':digest(output/'protocol.json'),
        'inputs':{label:{'crosscheck_sha256':digest(output/('g8_'+label)/'crosscheck.npz'),'calibration_model_sha256':digest(output/label/'model.json'),'audit_sha256':digest(output/('g8_'+label)/'independent_audit.json')} for label in candidates}})
    print(json.dumps({key:result[key] for key in ['comparison_complete','common_frames','paired']},indent=2),flush=True)


if __name__=='__main__':main()
