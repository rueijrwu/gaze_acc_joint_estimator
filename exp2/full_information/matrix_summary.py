"""Summarize the predeclared matched experiment without ranking scale costs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from calibrate_continuation import atomic_json


def read_csv(path):
    with path.open(newline='') as handle: return list(csv.DictReader(handle))


def summarize(matrix_dir,output_dir,case_dirs=None):
    output_dir=Path(output_dir)
    if output_dir.exists(): raise FileExistsError(output_dir)
    rows=[]; cases={}; policies=[]; withheld_keys=[]
    for name in ('N3','N4','R3','R4','U3_tau001','U3_tau01'):
        folder=Path((case_dirs or {}).get(name,Path(matrix_dir)/name))
        if not (folder/'model.json').exists():
            rows.append(dict(case=name,run_status='not_available'))
            continue
        model=json.loads((folder/'model.json').read_text())
        manifest=model['manifest']; parts=model['objective_parts']
        policy=manifest['common_policy']
        policy_hash=hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if policy_hash!=manifest['common_policy_sha256']:
            raise ValueError(f'{name}: common policy hash mismatch')
        if 17 in manifest['training_fixations'] or 17 in policy['training_fixations']:
            raise ValueError(f'{name}: withheld fixation entered training')
        policies.append(policy_hash)
        row=dict(case=name,run_status='converged' if model['converged'] else 'incomplete',
            training_fixation_count=len(manifest['training_fixations']),heldout_fixations=','.join(map(str,manifest['heldout_fixations'])),
            coefficient_count=model['coefficient_count'],free_coefficient_count=model['free_coefficient_count'],
            scale_hypothesis=model['scale_hypothesis'],conditional_scale=True,
            assumed_log_scale_sd=model.get('assumed_log_scale_sd'),
            model_directory=str(folder.resolve()),
            coefficient_solver=model.get('coefficient_solver','irls'),
            cumulative_elapsed_seconds=model.get('cumulative_continuation_totals',model['status']).get('elapsed_seconds'),
            cumulative_nfev=model.get('cumulative_continuation_totals',model['status']).get('nfev'),
            cumulative_accepted_iterations=model.get('cumulative_continuation_totals',model['status']).get('accepted_iterations'),
            common_policy_sha256=policy_hash,objective=model['objective'],
            optical=parts.get('observations'),anchors=parts.get('anchors'),temporal=parts.get('temporal'),
            function_prior=parts.get('model_prior'),normalized_function_prior=parts.get('normalized_function_prior'),
            log_shape_function_prior=parts.get('log_shape_function_prior'))
        prediction_path=folder/'predictions'/'heldout_predictions.csv'
        if prediction_path.exists():
            predictions=read_csv(prediction_path)
            if any(int(r['fixation_index']) not in manifest['heldout_fixations'] for r in predictions):
                raise ValueError(f'{name}: prediction cohort is not declared held-out support')
            withheld_keys.append(sorted((int(r['fixation_index']),int(r['frame_index'])) for r in predictions))
            valid=[r for r in predictions if str(r.get('failed','false')).lower() not in ('1','true')]
            row.update(heldout_frame_count=len(predictions),prediction_coverage=len(valid)/max(1,len(predictions)))
            for coordinate,label in (('theta_deg','theta'),('accommodation_D','A')):
                values=np.asarray([float(r[coordinate]) for r in valid])
                if len(values): row[f'heldout_{label}_mean']=float(np.mean(values)); row[f'heldout_{label}_sd']=float(np.std(values))
            fixations=read_csv(folder/'predictions'/'heldout_fixations.csv')
            for coordinate,label in (('theta_nominal_mean_offset_deg','theta'),('A_nominal_mean_offset_D','A')):
                offsets=np.asarray([float(r[coordinate]) for r in fixations if r[coordinate]])
                if len(offsets): row[f'heldout_{label}_equal_fixation_nominal_mean_RMS']=float(np.sqrt(np.mean(offsets**2)))
            for flag in ('failed','ambiguous','stationarity_unverified','theta_bound','A_bound','nominal_knot_extrapolation'):
                row['heldout_'+flag+'_count']=sum(str(r.get(flag,'false')).lower() in ('true','1') for r in predictions)
            geometry=[r for r in predictions if str(r.get('local_noise_valid','false')).lower() in ('true','1')]
            row['heldout_local_geometry_valid_count']=len(geometry)
            for key in ('local_theta_noise_sd_deg','local_A_noise_sd_D','local_noise_correlation','reference_scaled_jacobian_condition'):
                values=np.asarray([float(r[key]) for r in geometry if r.get(key)])
                values=values[np.isfinite(values)]
                if len(values): row['heldout_'+key+'_median']=float(np.median(values));row['heldout_'+key+'_p95']=float(np.quantile(values,.95))
        rows.append(row); cases[name]=model
    if policies and len(set(policies))!=1: raise ValueError('Matched cells have different common training policies')
    if withheld_keys and any(keys!=withheld_keys[0] for keys in withheld_keys[1:]):
        raise ValueError('Matched predictions do not use identical withheld frame identities')
    output_dir.mkdir(parents=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (output_dir/'matrix_summary.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    report=dict(schema='full_information_matrix_summary_v1',cases=rows,
        shared_policy_verified=len(set(policies))<=1,available_case_count=len(cases),
        same_withheld_frame_support_verified=bool(withheld_keys),
        comparison='whole-fixation withheld nominal consistency and stability; costs across scale hypotheses are not comparable likelihoods',
        external_accuracy_status='not_evaluable_without_independent_references')
    atomic_json(output_dir/'matrix_summary.json',report)
    lines=['# Matched internal experiment','',
        'All cells withhold fixation 17. Fixed scale is a conditional hypothesis. Costs are decompositions of each fitted objective; they do not select scale assumptions or establish accuracy.','',
        '| Case | Status | Training fixations | Coefficients (free) | Optical cost | Anchor cost | Log-shape prior |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        if row['run_status']=='not_available': lines.append(f"| {row['case']} | not available | | | | | |"); continue
        fmt=lambda key: '' if row.get(key) is None else f"{row[key]:.6g}"
        lines.append(f"| {row['case']} | {row['run_status']} | {row['training_fixation_count']} | {row['coefficient_count']} ({row['free_coefficient_count']}) | {fmt('optical')} | {fmt('anchors')} | {fmt('log_shape_function_prior')} |")
    lines+=['','Independent accuracy: **not evaluable without independent references**.']
    predicted=[r for r in rows if 'heldout_theta_sd' in r]
    if predicted:
        lines+=['','The following are nominal-label consistency and within-fixation variability, not accuracy errors.','',
            '| Case | Coverage | Mean gaze target offset RMS (deg) | Mean accommodation target offset RMS (D) | Gaze SD (deg) | Accommodation SD (D) |',
            '|---|---:|---:|---:|---:|---:|']
        for row in predicted:
            lines.append('| '+row['case']+' | '+' | '.join(f"{row.get(key,float('nan')):.6g}" for key in ('prediction_coverage','heldout_theta_equal_fixation_nominal_mean_RMS','heldout_A_equal_fixation_nominal_mean_RMS','heldout_theta_sd','heldout_A_sd'))+' |')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure,axes=plt.subplots(2,2,figsize=(10,6),constrained_layout=True)
        labels=[r['case'] for r in predicted]
        for axis,key,title,unit in zip(axes.flat,
            ('heldout_theta_equal_fixation_nominal_mean_RMS','heldout_A_equal_fixation_nominal_mean_RMS','heldout_theta_sd','heldout_A_sd'),
            ('Mean gaze nominal consistency','Mean accommodation nominal consistency','Within-fixation gaze variability','Within-fixation accommodation variability'),
            ('deg','D','deg','D')):
            axis.bar(labels,[r.get(key,np.nan) for r in predicted]); axis.set_title(title); axis.set_ylabel(unit)
        figure.suptitle('Whole-fixation internal holdout; conditional fixed scale; no independent accuracy reference')
        figure.savefig(output_dir/'heldout_consistency_and_spread.png',dpi=160); plt.close(figure)
    (output_dir/'matrix_summary.md').write_text('\n'.join(lines)+'\n')
    return report


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--case',action='append',default=[],help='Explicit NAME=MODEL_DIRECTORY override for a preserved continuation')
    args=parser.parse_args()
    case_dirs={}
    for item in args.case:
        name,path=item.split('=',1)
        if name not in ('N3','N4','R3','R4','U3_tau001','U3_tau01') or name in case_dirs: raise ValueError('Unknown or repeated case override')
        case_dirs[name]=Path(path)
    print(json.dumps(summarize(args.matrix_dir,args.output_dir,case_dirs)))
