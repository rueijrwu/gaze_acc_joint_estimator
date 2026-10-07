"""Reevaluate frozen fold models after an inverse change, without retraining."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from .data import load_reviewed
from .model import PositionModel
from .schema import load_model, write_json, source_hashes
from .validate import evaluate, stats, summarize
from .report import generate


def evaluate_fold(source, output, fold):
    root = Path(__file__).resolve().parents[1]
    captures, groups, _ = load_reviewed(root)
    results = {}
    for model_path in sorted((source/fold).glob('*/model.json')):
        model, meta = load_model(model_path)
        model.training_range = meta['provenance']['anchor_range']
        pilot = PositionModel(27, meta['pilot_coefficients'])
        dest = output/fold/model.name
        dest.mkdir(parents=True)
        for filename in ['model.json', 'training_states.json', 'determinant_audit.json']:
            path = model_path.parent/filename
            if path.exists():
                shutil.copy2(path, dest/filename)
        summary = evaluate(model, captures, groups, meta['provenance']['evaluation_group_ids'],
            pilot, np.array(meta['reference_state']), np.array(meta['coordinate_covariance']),
            dest, count=8, progress=lambda *args, **kwargs: None)
        summary.update(calibration_converged=True, calibration=meta['calibration'])
        results[f'{fold}/{model.name}'] = summary
        print(f'{fold}/{model.name}: {summary["estimated_rows"]}/{summary["evaluation_rows"]} frames; '
              f'{summary["testable_holdouts"]} point scores', flush=True)
    return results


def read_holdouts(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def compare(source, output):
    comparison = {}
    for family in ['gaze', 'capture']:
        common_state_rows = {}
        for name in ['conditional27', 'conditional37', 'two_channel13']:
            before, after, old_h, new_h = {}, {}, {}, {}
            for path in sorted((output).glob(f'{family}_*/{name}/frames.json')):
                fold = path.parent.parent.name
                for dest, base in [(before, source), (after, output)]:
                    for row in json.loads((base/fold/name/'frames.json').read_text()):
                        dest[(fold, row['fixation'], row['row'])] = row
                for dest, base in [(old_h, source), (new_h, output)]:
                    for row in read_holdouts(base/fold/name/'holdouts.jsonl'):
                        dest[(fold, row['fixation'], row['row'], row['held_point'])] = row
            common = [key for key in before if before[key].get('estimated') and after[key].get('estimated')]
            point_keys = [key for key in old_h if old_h[key].get('score_available') and new_h[key].get('score_available')]
            errors = lambda rows, keys: stats([np.linalg.norm(rows[key]['error_px']) for key in keys])
            score_keys = lambda rows: [key for key in rows if rows[key].get('score_available')]
            transitions = []
            for key, old in old_h.items():
                new = new_h[key]
                lower_cost = old.get('available') and new.get('available') and new['cost'] < old['cost']-1e-6*(1+old['cost'])
                if lower_cost or old.get('score_available') != new.get('score_available'):
                    transitions.append(dict(key=list(key), lower_cost=bool(lower_cost),
                        state_before=old.get('state'), state_after=new.get('state'),
                        cost_before=old.get('cost'), cost_after=new.get('cost'),
                        scored_before=old.get('score_available',False),scored_after=new.get('score_available',False),
                        error_before=float(np.linalg.norm(old['error_px'])) if old.get('score_available') else None,
                        error_after=float(np.linalg.norm(new['error_px'])) if new.get('score_available') else None))
            comparison[f'{family}/{name}'] = dict(
                frames_before=sum(r.get('estimated', False) for r in before.values()),
                frames_after=sum(r.get('estimated', False) for r in after.values()),
                recovered_frames=[list(k) for k in before if not before[k].get('estimated') and after[k].get('estimated')],
                lost_frames=[list(k) for k in before if before[k].get('estimated') and not after[k].get('estimated')],
                changed_states=sum(np.max(np.abs(np.array([before[k]['theta'],before[k]['A']])-
                    [after[k]['theta'],after[k]['A']])) > .01 for k in common),
                lower_cost_frames=sum(after[k]['cost'] < before[k]['cost']-1e-6*(1+before[k]['cost']) for k in common),
                old_point_count=len(score_keys(old_h)), new_point_count=len(score_keys(new_h)),
                point_error_before=errors(old_h,score_keys(old_h)), point_error_after=errors(new_h,score_keys(new_h)),
                paired_point_count=len(point_keys), paired_error_before=errors(old_h,point_keys), paired_error_after=errors(new_h,point_keys),
                recovered_point_scores=sum(not old_h[k].get('score_available') and new_h[k].get('score_available') for k in old_h),
                lost_point_scores=sum(old_h[k].get('score_available') and not new_h[k].get('score_available') for k in old_h),
                lower_cost_holdouts=sum(old_h[k].get('available') and new_h[k].get('available') and
                    new_h[k]['cost'] < old_h[k]['cost']-1e-6*(1+old_h[k]['cost']) for k in old_h),
                holdout_transitions=transitions)
            common_state_rows[name] = after
        keys = set.intersection(*[{k for k,r in rows.items() if r.get('estimated')}
                                 for rows in common_state_rows.values()])
        matched_means = {}
        for name, rows in common_state_rows.items():
            summary = summarize([rows[k] for k in sorted(keys)], [])
            matched_means[name] = {k: summary[k] for k in ['fixation_means',
                'fixation_mean_gaze_anchor_discrepancy_deg','fixation_mean_accommodation_demand_discrepancy_D','fixation_weighting']}
        comparison[f'{family}/common_state_support'] = dict(count=len(keys), models=matched_means,
            rows=[list(k) for k in sorted(keys)], definition='same estimated rows in all three revised models; equal fixation weights')
    write_json(output/'comparison.json', comparison)
    return comparison


def run(source, output, workers=6):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob('*') if p.is_file()}
    write_json(output/'config.json', dict(source=str(source), source_hashes=hashes,
        implementation_hashes=source_hashes(Path(__file__).resolve().parents[1]), workers=workers,
        retrained=False, original_population_sampling=8,
        inverse_policy='49 scalar starts; bounded exact-Hessian polish; original stationarity threshold'))
    folds = sorted({p.parent.parent.name for p in source.glob('*/*/model.json')})
    summaries = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate_fold, source, output, fold) for fold in folds]
        for future in as_completed(futures):
            summaries.update(future.result())
            write_json(output/'summary.json', summaries)
    generate(output)
    compare(source, output)
    from .diagnostics import enrich
    enrich(output)
    for name, digest in hashes.items():
        assert hashlib.sha256((source/name).read_bytes()).hexdigest() == digest, name
    write_json(output/'completion.json', dict(complete=True, seconds=time.monotonic()-begun,
        original_results_unchanged=True, model_count=len(summaries), retrained=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('experiments/full_position/grouped_v2'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    run(args.source, args.output, args.workers)
