"""Recorded branch checks and training-only discrepancy diagnostics."""
from __future__ import annotations
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .ablation import same_response_summaries, retained_x_prediction
from .calibrate import ProfiledProblem
from .data import load_reviewed, training_data
from .geometry import context
from .invert import invert, predict_holdout, score_holdout
from .model import PositionModel
from .noise import reference_covariance, marginal
from .profile import profile_inverse
from .schema import load_model, write_json, source_hashes
from .validate import stats


def targeted_checks(source, output, captures):
    path=source/'gaze_-10/conditional37/model.json'
    model,meta=load_model(path)
    pilot=PositionModel(27,meta['pilot_coefficients'])
    cap=captures['capture_1_detections.pkl']
    records, ablations=[],[]
    with gzip.open(output/'targeted_candidates.jsonl.gz','wt') as stream:
        for row in [1380,1928]:
            ctx=context(cap.p[row])
            R=reference_covariance(cap.p[row:row+1],pilot,np.array(meta['reference_state']),
                                   np.array(meta['coordinate_covariance']))[0]
            v=cap.v[row].ravel()
            for held in [None,0,1,2]:
                ix=np.array([k for k in range(6) if held is None or k//2!=held])
                scalar=invert(model,ctx.r,v[ix],marginal(R,ix),ix)
                profile=profile_inverse(model,ctx.r,v[ix],marginal(R,ix),ix)
                for name,result in [('scalar',scalar),('profile',profile)]:
                    stream.write(json.dumps(dict(row=row,held_point=held,method=name,
                        candidates=result.pop('candidates',[])))+'\n')
                record=dict(row=row,held_point=held,scalar=scalar,profile=profile)
                if held is not None:
                    prediction=score_holdout(predict_holdout(model,ctx,held,v[ix],R),cap.q[row,held],ctx.ell)
                    record['scalar_withheld_error_px']=float(np.linalg.norm(prediction['error_px'])) if prediction.get('score_available') else None
                    xonly=score_holdout(retained_x_prediction(model,ctx,held,v,R),cap.q[row,held],ctx.ell)
                    xonly.pop('candidates',None)
                    ablations.append(dict(row=row,held_point=held,
                        xy_state=prediction.get('state'),xy_rank=prediction.get('rank'),
                        xy_error_px=record['scalar_withheld_error_px'],
                        retained_x=xonly,policy='both inverses exclude the tested P4 point; no all-three area'))
                else:
                    reference_v=pilot.predict(np.array(meta['reference_state']),ctx.r)
                    reduced=same_response_summaries(model,ctx.r,v,reference_v,R)
                    reduced.pop('candidates',None)
                    ablations.append(dict(row=row,coordinate_state=scalar.get('state'),
                        derived_summaries=reduced,policy='same frozen response; all-three diagnostic, not withheld validation'))
                records.append(record)
    write_json(output/'targeted_branch_checks.json',dict(records=records,
        cases=8,scope='two targeted original rows; finite profile grid is not completeness proof'))
    write_json(output/'measurement_ablations.json',dict(records=ablations,
        scope='two-row diagnostic only; no new population or model selection'))


def training_diagnostics(source, output, captures, groups):
    components, residual_groups, sensitivities=[],[],[]
    for path in sorted(source.glob('*/*/model.json')):
        model,meta=load_model(path)
        fold=path.parent.parent.name
        ids=meta['provenance']['training_group_ids']
        data,anchors=training_data(captures,groups,ids,48)
        trained=json.loads((path.parent/'training_states.json').read_text())
        states=np.asarray(trained['states'])
        assert np.array_equal(data['rows'],trained['rows'])
        pilot=PositionModel(27,meta['pilot_coefficients'])
        cov=reference_covariance(data['p'],pilot,np.array(meta['reference_state']),
                                np.array(meta['coordinate_covariance']),model.channels==2)
        y=data['v'] if model.channels==6 else data['y2']
        problem=ProfiledProblem(model,y,data['r'],cov,data['groups'],anchors,
                               meta['calibration']['prior_strength'])
        residual=model.predict(states,data['r'])-y
        optical=np.einsum('nij,nj->ni',problem.weights,residual)
        anchor=(problem.mean_states(states)-anchors)/problem.anchor_scales/np.sqrt(problem.k)
        prior=problem.penalty*(model.beta-problem.beta0)
        record=dict(fold=fold,model=model.name,optical_cost=float(np.sum(optical**2)/2),
            anchor_cost=float(np.sum(anchor**2)/2),prior_cost=float(prior@prior/2),
            selected_saved_cost=meta['calibration']['alternatives'][meta['calibration']['selected_start']]['cost'])
        record['reconstructed_cost']=sum(record[k] for k in ['optical_cost','anchor_cost','prior_cost'])
        record['difference']=record['reconstructed_cost']-record['selected_saved_cost']
        components.append(record)
        if model.channels!=6:
            continue
        # Conditional coefficient refits isolate optical/prior/reference effects
        # at the saved training states; they do not refit latent states/anchors.
        variants=[('prior_0.1x',meta['calibration']['prior_strength']*.1,np.array(meta['reference_state']),1.),
                  ('prior_10x',meta['calibration']['prior_strength']*10,np.array(meta['reference_state']),1.),
                  ('covariance_4x',meta['calibration']['prior_strength'],np.array(meta['reference_state']),4.),
                  ('reference_q25',meta['calibration']['prior_strength'],np.quantile(anchors,.25,axis=0),1.),
                  ('reference_q75',meta['calibration']['prior_strength'],np.quantile(anchors,.75,axis=0),1.)]
        for label,strength,reference,factor in variants:
            changed_cov=reference_covariance(data['p'],pilot,reference,np.array(meta['coordinate_covariance']))*factor
            conditional=ProfiledProblem(model,y,data['r'],changed_cov,data['groups'],anchors,strength)
            conditional.update(states/np.array([10.,4.]))
            prediction_change=np.einsum('ncp,p->nc',model.design(states,data['r']),conditional.beta-model.beta)
            change_px=prediction_change.reshape(-1,3,2)*context(data['p']).ell[:,None,None]
            sensitivities.append(dict(fold=fold,model=model.name,variant=label,
                prior_strength=strength,reference=reference.tolist(),covariance_factor=factor,
                prediction_change_rms_px=float(np.sqrt(np.mean(change_px**2))),
                normalized_coefficient_change=float(np.linalg.norm((conditional.beta-model.beta)*conditional.column_scale)),
                latent_states_refitted=False))
        error_px=-residual.reshape(-1,3,2)*context(data['p']).ell[:,None,None]
        # Group using training labels only; a demand/capture split is confounded.
        for local,gi in enumerate(ids):
            mask=data['groups']==local
            for point in range(3):
                for axis in range(2):
                    values=error_px[mask,point,axis]
                    residual_groups.append(dict(fold=fold,model=model.name,fixation=gi,
                        capture=groups[gi]['capture'],nominal_theta=anchors[local,0],demand=anchors[local,1],
                        point=point,axis='xy'[axis],count=len(values),signed_bias_px=float(values.mean()),
                        rms_px=float(np.sqrt(np.mean(values**2))),
                        r_mean=data['r'][mask].mean(0).tolist()))
    write_json(output/'training_objective_components.json',dict(records=components,
        convention='half weighted sum of squares; original coefficients and saved states; no refit'))
    write_json(output/'training_residual_groups.json',dict(records=residual_groups,
        scope='training-only in-sample residuals; descriptive discrepancy diagnostics, not validation',
        capture_demand_confounded=True))
    write_json(output/'conditional_coefficient_sensitivity.json',dict(records=sensitivities,
        scope='training-only QR coefficient refits at fixed saved latent states; no physiological accuracy or coefficient covariance claim',
        anchor_sensitivity_not_tested=True,full_latent_state_refits_required=True))
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for point in range(3):
        for axis_index,axis in enumerate('xy'):
            ax=axes[axis_index,point]
            for name,color in [('conditional27','#2674a8'),('conditional37','#db7a29')]:
                data=[r for r in residual_groups if r['point']==point and r['axis']==axis and r['model']==name]
                ax.scatter([r['nominal_theta'] for r in data],[r['signed_bias_px'] for r in data],s=9,alpha=.4,color=color,label=name)
            ax.axhline(0,color='gray',lw=1);ax.set(title=f'P4 {point+1}, {axis}',xlabel='Training nominal gaze (deg)',ylabel='Signed mean residual (px)')
            ax.legend(fontsize=7)
    fig.savefig(output/'training_signed_residuals.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for name,color in [('conditional27','#2674a8'),('conditional37','#db7a29')]:
        data=[r for r in residual_groups if r['model']==name]
        axes[0].scatter([r['demand'] for r in data],[r['signed_bias_px'] for r in data],s=6,alpha=.25,color=color,label=name)
        axes[1].scatter([r['r_mean'][0][0] for r in data],[r['signed_bias_px'] for r in data],s=6,alpha=.25,color=color,label=name)
    for ax in axes:
        ax.axhline(0,color='gray',lw=1);ax.set_ylabel('Signed training fixation/point/axis bias (px)');ax.legend(fontsize=8)
    axes[0].set_xlabel('Training demand (D), confounded with capture')
    axes[1].set_xlabel('Mean training P1 context r1x')
    fig.savefig(output/'training_demand_context_residuals.png',dpi=160);plt.close(fig)
    write_json(output/'diagnosis_summary.json',dict(
        max_objective_reconstruction_error=max(abs(r['difference']) for r in components),
        training_signed_bias_px=stats([abs(r['signed_bias_px']) for r in residual_groups]),
        distinction='Effective short-term coordinate noise is unchanged; coefficient uncertainty and response discrepancy remain excluded.',
        next_experiment='Grouped training-only coefficient refits and anchor/prior/reference-covariance sensitivity before basis changes.'))


def check_transitions(source,revised,output,captures,groups):
    """Independent profile checks on every newly selected lower-cost subset."""
    comparison=json.loads((revised/'comparison.json').read_text())
    records=[]
    with gzip.open(output/'transition_profile_candidates.jsonl.gz','wt') as stream:
        for key,entry in comparison.items():
            if 'holdout_transitions' not in entry:
                continue
            name=key.split('/')[1]
            for transition in entry['holdout_transitions']:
                if not transition['lower_cost']:
                    continue
                fold,gi,row,held=transition['key']
                model,meta=load_model(source/fold/name/'model.json')
                pilot=PositionModel(27,meta['pilot_coefficients'])
                cap=captures[groups[gi]['capture']]
                ctx=context(cap.p[row])
                R=reference_covariance(cap.p[row:row+1],pilot,np.array(meta['reference_state']),
                    np.array(meta['coordinate_covariance']))[0]
                ix=np.array([k for k in range(6) if k//2!=held])
                profile=profile_inverse(model,ctx.r,cap.v[row].ravel()[ix],marginal(R,ix),ix)
                stream.write(json.dumps(dict(key=transition['key'],candidates=profile.pop('candidates',[])))+'\n')
                records.append(dict(key=transition['key'],revised_scalar_cost=transition['cost_after'],
                    profile=profile,profile_cost_difference=profile.get('cost',np.nan)-transition['cost_after']))
    write_json(output/'transition_profile_checks.json',dict(records=records,
        scope='all newly lower-cost scored subsets; independent finite-grid check, not completeness proof'))


def run(source,output,revised=None):
    source,output=Path(source),Path(output)
    output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    captures,groups,_=load_reviewed(root)
    targeted_checks(source,output,captures)
    training_diagnostics(source,output,captures,groups)
    if revised is not None:
        check_transitions(source,Path(revised),output,captures,groups)
    write_json(output/'completion.json',dict(complete=True,retrained=False,untouched_captures=[5,6],
        implementation_hashes=source_hashes(root)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('experiments/full_position/grouped_v2'))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--revised',type=Path,help='Also profile lower-cost subset transitions from this reevaluation')
    args=parser.parse_args()
    run(args.source,args.output,args.revised)
