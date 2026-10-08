"""Bounded exponent search, confirmation, and full GPU agreement evaluation.

Every stage freezes original-row schedules before reading validity. Calibration
uses independent frame states and training-only contiguous noise blocks. The
sampled screen is development evidence; the full stage is internal agreement.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits
from . import linear_acceleration
from .audit83 import reviewed
from .batched_holdout import predict_batch
from .calibrate import fit
from .crosscheck import join_records
from .data import fixed_sample, window_rows
from .geometry import context
from .invert import score_holdout
from .literal_power import LiteralPowerModel
from .model import PositionModel
from .noise import coordinate_covariance, reference_covariance
from .population import frame_id, manifest
from .schema import clean_json, write_json
from .scorecard import build
from .validate import pilot_fit

GRID=(.25,.5,.75,1.,1.25,1.5,2.)
METRICS=('E_cross_px','G_theta_cross_deg','G_A_cross_D','worst_point_px')


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''): h.update(block)
    return h.hexdigest()


def case(exponent, floor=.01):
    label='log' if exponent is None else f'n{exponent:.8g}'
    return dict(id=f'{label}_amin{floor:.8g}',exponent=exponent,accommodation_min=floor)


def training_blocks(rows, count):
    """Sixteen contiguous blocks distributed across the full fixation period."""
    if count<=0 or count>=len(rows): return rows.copy()
    lengths=[count//16+(i<count%16) for i in range(16)]
    centers=np.linspace(0,len(rows)-1,16).round().astype(int)
    chosen=set()
    for center,length in zip(centers,lengths):
        begin=max(0,min(int(center)-length//2,len(rows)-length))
        chosen.update(map(int,rows[begin:begin+length]))
    # A deterministic fill is needed only for very short overlapping intervals.
    for row in fixed_sample(rows,count):
        if len(chosen)>=count: break
        chosen.add(int(row))
    return np.array(sorted(chosen),int)


def prepare(root, stage, train_count, evaluation_count):
    stage.mkdir(parents=True,exist_ok=False)
    captures,groups,interval_hash=reviewed(str(root))
    parts,anchors,population,identities,noise=[],[],[],[],[]
    frozen=[]
    for gi,g in enumerate(groups):
        cap=captures[g['capture']]
        period=window_rows(g,'fixation_period')
        train=training_blocks(period,train_count)
        pool=period if evaluation_count==0 else np.setdiff1d(period,train,assume_unique=True)
        evaluation=fixed_sample(pool,evaluation_count)
        if not len(evaluation): raise ValueError('No disjoint evaluation rows')
        frozen.append(dict(group=gi,capture=cap.name,training_original_rows=train.tolist(),evaluation_original_rows=evaluation.tolist()))
        if evaluation_count and np.intersect1d(train,evaluation).size:
            raise ValueError('Sampled calibration/evaluation overlap')
        valid=train[cap.baseline_valid[train]]
        if not len(valid): raise ValueError(f'Empty valid training group {gi}')
        parts.append(dict(p=cap.p[valid],q=cap.q[valid],r=cap.ctx.r[valid],v=cap.v[valid].reshape(-1,6),
                          groups=np.full(len(valid),gi,int),original_group=np.full(len(valid),gi,int),rows=valid))
        anchors.append([g['target_theta_deg'],g['demand_diopters_label']])
        gaps=np.flatnonzero((np.diff(valid)!=1)|(np.diff(cap.frame[valid])!=1))+1
        for run in np.split(valid,gaps):
            if len(run)>=3: noise.append(np.concatenate((cap.p[run].reshape(-1,6),cap.q[run].reshape(-1,6)),-1))
        for row in evaluation:
            i=int(row)
            population.append(dict(capture=cap.name,fixation=gi,row=i,frame=int(cap.frame[i]),
                selected_for_evaluation=True,p1_valid_geometry=bool(cap.ctx.valid[i]),
                baseline_valid=bool(cap.baseline_valid[i]),
                **{f'p4_{j+1}_valid':bool(cap.point_valid[i,j]) for j in range(3)}))
            identities.append(dict(fold=stage.name,capture=cap.name,fixation=gi,row=i,source_frame_index=int(cap.frame[i])))
    write_json(stage/'frozen_rows.json',frozen)
    write_json(stage/'population.json',population)
    write_json(stage/'manifest.json',dict(frames=identities,sampling='full_period_contiguous_training_blocks_and_disjoint_evenly_spaced_evaluation',
        train_per_fixation=train_count,evaluation_per_fixation=evaluation_count,
        evaluation_type='internal_full_population' if evaluation_count==0 else 'sampled_development_disjoint_frames'))
    data={key:np.concatenate([p[key] for p in parts]) for key in parts[0]}
    anchors=np.asarray(anchors,float)
    pilot=pilot_fit(data,anchors)
    sigma,noise_meta=coordinate_covariance(noise)
    reference=np.median(anchors,axis=0)
    covariance=reference_covariance(data['p'],pilot,reference,sigma)
    np.savez_compressed(stage/'training_inputs.npz',**data,anchors=anchors,covariance=covariance)
    info=dict(pilot_coefficients=pilot.beta,reference_state=reference,coordinate_covariance=sigma,
              noise=noise_meta,groups=groups,interval_sha256=interval_hash,
              capture_sha256={name:getattr(cap,'sha256',None) for name,cap in captures.items()},
              calibration_rows=len(data['rows']),scheduled_frames=len(population),
              contiguous_noise_training_only=True,schedule_sha256=digest(stage/'manifest.json'),
              data_sha256=digest(stage/'training_inputs.npz'))
    write_json(stage/'training.json',info)
    return info


def task(job):
    root,stage,spec,device,settings=job
    root,stage=Path(root),Path(stage)
    dest=stage/'fits'/spec['id']; dest.mkdir(parents=True,exist_ok=False)
    import cupy as cp
    cp.cuda.Device(device).use()
    try:
        with threadpool_limits(limits=settings['cpu_threads']):
            return _task(root,stage,spec,device,settings,dest,cp)
    except Exception as exc:
        (dest/'exception.txt').write_text(traceback.format_exc())
        count=len(read(stage/'manifest.json')['frames'])
        result=dict(id=spec['id'],exponent=spec['exponent'],accommodation_min=spec['accommodation_min'],
            certified=False,device=device,exception=repr(exc),coverage=dict(scheduled_frames=count,
                scheduled_slots=3*count,scored=0,complete_triples=0))
        write_json(dest/'completion.json',dict(complete=False,**result))
        return result


def _task(root,stage,spec,device,settings,dest,cp):
    info=read(stage/'training.json')
    with np.load(stage/'training_inputs.npz') as saved:
        data={k:saved[k].copy() for k in saved.files}
    model=LiteralPowerModel(0 if spec['exponent'] is None else spec['exponent'],
        accommodation_min=spec['accommodation_min'],log_control=spec['exponent'] is None)
    begun=time.monotonic()
    linear_acceleration.enable('cupy',device)
    try:
        with gzip.open(dest/'calibration_candidates.jsonl.gz','wt') as stream:
            def checkpoint(record,beta,states):
                stream.write(json.dumps(clean_json(dict(record=record,coefficients=beta)),allow_nan=False)+'\n'); stream.flush()
                np.savez_compressed(dest/'latest_calibration_checkpoint.npz',coefficients=beta,states=states)
            model,states,diagnostics=fit(model,data['v'],data['r'],data['covariance'],data['groups'],data['anchors'],
                starts=2,max_nfev=settings['max_nfev'],seed=17,checkpoint=checkpoint,
                continuation_stages=2,prior_strength=.001,anchor_scales=(.1,.25),curvature_strength=0.)
    finally:
        linear_acceleration.disable()
    accepted=diagnostics['converged'] is True
    artifact=dict(schema='literal_power_model_v1',model=model.name,coefficients=model.beta,
        accommodation_response=model.metadata(),calibration=diagnostics,
        pilot_coefficients=info['pilot_coefficients'],reference_state=info['reference_state'],
        coordinate_covariance=info['coordinate_covariance'],provenance=info,
        state_policy='independent_two_variable_state_per_frame',temporal_strength=0.)
    write_json(dest/('model.json' if accepted else 'failed_checkpoint.json'),artifact)
    np.savez_compressed(dest/'training_states.npz',states=states,groups=data['groups'],rows=data['rows'])
    if accepted: model=LiteralPowerModel.from_object(read(dest/'model.json'))
    empirical_min,empirical_max=states.min(0),states.max(0)
    anchor_min,anchor_max=data['anchors'].min(0),data['anchors'].max(0)
    rmin,rmax=data['r'].min(0),data['r'].max(0)
    parity=set(np.sign(context(data['p']).signed_area).tolist())
    del data; gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    captures,groups,_=reviewed(str(root))
    population=read(stage/'population.json')
    pilot=PositionModel(27,info['pilot_coefficients'])
    reference,sigma=np.asarray(info['reference_state']),np.asarray(info['coordinate_covariance'])
    holds=[]
    batch_size=settings['batch_size']
    with gzip.open(dest/'inverse_candidates.jsonl.gz','wt') as archive:
        # Each block has at most batch_size held-point inverses, each with 49 starts.
        schedule=[(row,j) for row in population for j in range(3)]
        for begin in range(0,len(schedule),batch_size):
            batch=schedule[begin:begin+batch_size]
            ps=np.array([captures[row['capture']].p[row['row']] for row,j in batch])
            qs=np.array([captures[row['capture']].q[row['row']] for row,j in batch])
            valid=np.array([captures[row['capture']].point_valid[row['row']] for row,j in batch])
            held=np.array([j for row,j in batch])
            predictions=(predict_batch(model,ps,qs,valid,held,pilot,reference,sigma,device=device)
                         if accepted else [dict(available=False,held_point=j,reason='calibration_failed',state=None) for j in held])
            ctx=context(ps)
            for local,((row,j),prediction) in enumerate(zip(batch,predictions)):
                candidates=prediction.pop('candidates',[])
                archive.write(json.dumps(clean_json(dict(capture=row['capture'],fixation=row['fixation'],row=row['row'],held_point=j,candidates=candidates)),allow_nan=False)+'\n')
                scored=score_holdout(prediction,qs[local,j] if valid[local,j] else np.full(2,np.nan),ctx.ell[local])
                scored.update(capture=row['capture'],fixation=row['fixation'],row=row['row'],held_point=j,retained_image_channels='xy')
                holds.append(scored)
            archive.flush()
            write_json(dest/'progress.json',dict(calibration_certified=accepted,agreement_slots=min(begin+len(batch),len(schedule)),scheduled_slots=len(schedule),device=device))
    frames=[dict(capture=row['capture'],fixation=row['fixation'],row=row['row'],frame=row['frame'],estimated=False,
                 nominal_theta=groups[row['fixation']]['target_theta_deg'],demand=groups[row['fixation']]['demand_diopters_label'],
                 reason='subset_crosscheck_only' if accepted else 'calibration_failed') for row in population]
    joined=join_records('literal_power',stage.name,spec['id'],population,frames,holds)
    del holds
    for frame in joined:
        row=frame['row']; cap=captures[frame['capture']]
        rc=cap.ctx.r[row]
        for slot in frame['slots']:
            x=np.asarray(slot['state']) if slot.get('state') is not None else None
            slot['support']=dict(P1_context_valid=bool(cap.ctx.valid[row]),
                context_outside_training_extrema=bool(np.any(rc<rmin)|np.any(rc>rmax)),
                p1_parity_seen_in_training=bool(np.sign(cap.ctx.signed_area[row]) in parity),
                theta_empirical=None if x is None else bool(empirical_min[0]<=x[0]<=empirical_max[0]),
                A_empirical=None if x is None else bool(empirical_min[1]<=x[1]<=empirical_max[1]),
                theta_anchor=None if x is None else bool(anchor_min[0]<=x[0]<=anchor_max[0]),
                A_anchor=None if x is None else bool(anchor_min[1]<=x[1]<=anchor_max[1]))
    with gzip.open(dest/'frames.jsonl.gz','wt') as stream:
        for frame in joined: stream.write(json.dumps(clean_json(frame),allow_nan=False)+'\n')
    pop=manifest(read(stage/'manifest.json')['frames'])
    card=build(joined,population=pop)
    with gzip.open(dest/'scorecard.json.gz','wt') as stream: json.dump(clean_json(card),stream,allow_nan=False)
    result=dict(id=spec['id'],exponent=spec['exponent'],accommodation_min=spec['accommodation_min'],
                certified=accepted,device=device,seconds=time.monotonic()-begun,coverage=card['coverage'])
    write_json(dest/'completion.json',dict(complete=True,**result))
    return result


def execute(root,stage,specs,settings):
    devices=settings['devices']
    jobs=[(str(root),str(stage),spec,devices[i%len(devices)],settings) for i,spec in enumerate(specs)
          if not (stage/'fits'/spec['id']).exists()]
    results=[]
    with ProcessPoolExecutor(max_workers=settings['workers'],mp_context=mp.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(task,job) for job in jobs]):
            result=future.result();results.append(result)
            print(json.dumps(dict(stage=stage.name,**result)),flush=True)
    return results


def comparison(stage,ids):
    pop=manifest(read(stage/'manifest.json')['frames'])
    rows={}; results={}
    for name in ids:
        folder=stage/'fits'/name
        completion=read(folder/'completion.json');results[name]=completion
        if not completion['certified']: continue
        with gzip.open(folder/'frames.jsonl.gz','rt') as stream:
            rows[name]=[json.loads(line) for line in stream]
        pop.validate(rows[name])
    tables={name:{frame_id(f):f for f in frames if f['complete_triple']} for name,frames in rows.items()}
    shared=sorted(set.intersection(*(set(v) for v in tables.values()))) if tables else []
    missing=sorted(set(pop.exposures)-{key[:3] for key in shared})
    cards={name:build([table[key] for key in shared],expected_exposures=pop.exposures,
                     purpose='literal_power_development_comparison') for name,table in tables.items()}
    values={name:{metric:card['outcomes'][metric]['equal_exposure_rms'] for metric in METRICS} for name,card in cards.items()}
    finite=all(value is not None and np.isfinite(value) for scores in values.values() for value in scores.values())
    order={metric:sorted(values,key=lambda name:(values[name][metric],name)) for metric in METRICS} if shared and finite else {}
    valid=bool(shared and finite and not missing and any(name.startswith('log_') for name in rows))
    result=dict(status='comparable' if valid else 'incomplete',scheduled_frames=len(pop.frame_ids),
        shared_complete_frames=len(shared),expected_exposures=pop.exposures,missing_exposures=missing,
        rms=values,metric_order=order,individual=results,
        tails={name:{key:cards[name]['outcomes'][key]['distribution'] for key in METRICS} for name in cards},
        bounds={name:cards[name]['coverage'] for name in cards},
        scope='sampled_development' if stage.name!='full' else 'internal_full_population',
        independent_physiological_accuracy=False,selected_for_deployment=None)
    write_json(stage/'comparison.json',result)
    with gzip.open(stage/'shared_membership.jsonl.gz','wt') as stream:
        for key in shared: stream.write(json.dumps(key)+'\n')
    lines=[f'# Literal-power {stage.name} comparison','',f'Shared complete frames: {len(shared):,} / {len(pop.frame_ids):,}.',
           f'Missing exposures: {len(missing)}.','',
           '| Candidate | P4 error (px) | Gaze disagreement (deg) | Accommodation disagreement (D) | Worst-point error (px) |',
           '|---|---:|---:|---:|---:|']
    for name in order.get('E_cross_px',[]):
        lines.append('| '+name+' | '+' | '.join(f'{values[name][key]:.6g}' for key in METRICS)+' |')
    lines+=['','Uncertified candidates: '+', '.join(name for name,record in results.items() if not record['certified']),
            '', 'All scores measure internal/development agreement. They do not establish independent physiological accuracy.','']
    (stage/'RESULTS.md').write_text('\n'.join(lines))
    return result


def shortlist(report):
    if report['status']!='comparable': return []
    return [name for name in report['metric_order']['E_cross_px'] if name.startswith('n')][:2]


def run(root,output,settings):
    root,output=Path(root).resolve(),Path(output).resolve()
    validation_path=Path(settings['validation_report'])
    if not validation_path.is_absolute(): validation_path=root/validation_path
    validation=read(validation_path)
    snapshots={str(p.relative_to(root)):digest(p) for p in (root/'full_position').glob('*.py')}
    if (validation.get('passed') is not True or validation.get('source_sha256')!=snapshots
            or validation.get('family')!='literal_power_response_v1'
            or validation.get('verified_backend')!='cupy_float64'):
        raise ValueError('A current passing literal-power CPU/GPU validation report is required')
    import cupy as cp
    if (len(settings['devices'])!=2 or len(set(settings['devices']))!=2
            or any(i<0 or i>=cp.cuda.runtime.getDeviceCount() for i in settings['devices'])):
        raise ValueError('Select both available, distinct CUDA devices')
    output.mkdir(parents=True,exist_ok=False)
    protected={str(p.relative_to(root)):digest(p) for directory in ('data','models')
               for p in (root/directory).rglob('*') if p.is_file()}
    for rel in snapshots:
        dst=output/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/rel,dst)
    write_json(output/'config.json',dict(settings=settings,coarse_exponents=GRID,
        response='literal (A / 1 D)^n',default_accommodation_min_D=.01,
        calibration_starts=2,continuation_stages=2,seed=17,
        refinement='midpoints around best sampled exponent; one boundary extension',
        bound_sensitivity_D=[.001,.05],near_best_loss_fraction=.05,
        frames_free_per_fixation=True,source_sha256=snapshots,protected_input_sha256=protected,
        validation_report_sha256=digest(validation_path),
        selection_is_development_only=True,independent_validation=False))
    def progress(status,**extra):
        write_json(output/'progress.json',dict(utc=datetime.now(timezone.utc).isoformat(),status=status,**extra))
    progress('coarse_preparation')
    coarse=output/'coarse';prepare(root,coarse,256,64)
    specs=[case(None)]+[case(n) for n in GRID]
    progress('coarse_evaluation',candidates=[s['id'] for s in specs]);execute(root,coarse,specs,settings)
    report=comparison(coarse,[s['id'] for s in specs]);chosen=shortlist(report)
    if not chosen:
        progress('stopped_no_certified_comparable_power');return
    best=next(s['exponent'] for s in specs if s['id']==chosen[0])
    i=list(GRID).index(best);fine=[]
    if i: fine.append((GRID[i-1]+best)/2)
    if i<len(GRID)-1: fine.append((best+GRID[i+1])/2)
    if i==0: fine.append(.1)
    if i==len(GRID)-1: fine.append(3.)
    extra=[case(n) for n in fine]
    progress('refinement_evaluation',candidates=[s['id'] for s in extra]);execute(root,coarse,extra,settings)
    specs+=extra;report=comparison(coarse,[s['id'] for s in specs]);chosen=shortlist(report)
    if not chosen:
        progress('stopped_refinement_incomplete_common_exposure_support');return
    chosen_specs=[next(s for s in specs if s['id']==name) for name in chosen]
    best_loss=report['rms'][chosen[0]]['E_cross_px']**2
    promising=[s['exponent'] for s in specs if s['exponent'] is not None and s['id'] in report['rms']
               and report['rms'][s['id']]['E_cross_px']**2<=1.05*best_loss]
    write_json(output/'screen_selection.json',dict(shortlist=chosen_specs,near_best_exponents=sorted(promising),
        near_best_definition='within 5% of best squared P4 loss; exploratory range, not a confidence interval'))
    confirm=output/'confirm';prepare(root,confirm,512,128)
    confirm_specs=[case(None)]+chosen_specs
    progress('confirmation_evaluation',candidates=[s['id'] for s in confirm_specs]);execute(root,confirm,confirm_specs,settings)
    confirmation=comparison(confirm,[s['id'] for s in confirm_specs]);chosen=shortlist(confirmation)
    if not chosen:
        progress('stopped_confirmation_uncertified_or_incomplete');return
    finalists=[next(s for s in confirm_specs if s['id']==name) for name in chosen]
    best=finalists[0]['exponent']
    sensitivity=[]
    for floor in (.001,.05):
        probes=[case(None,floor),case(best,floor)]
        progress('bound_sensitivity',floor=floor);execute(root,confirm,probes,settings)
        sensitivity.append(dict(floor=floor,comparison=comparison(confirm,[s['id'] for s in probes])))
        write_json(confirm/f'bound_sensitivity_{floor:.8g}.json',sensitivity[-1])
    # Restore the primary default-bound report after auxiliary comparisons.
    confirmation=comparison(confirm,[s['id'] for s in confirm_specs])
    write_json(output/'full_selection.json',dict(finalists=finalists,control=case(None),bound_sensitivity=sensitivity,
        frozen_before_full=True,selection_scope='sampled_development',no_automatic_promotion=True))
    full=output/'full';info=prepare(root,full,0,0)
    if info['calibration_rows']!=89175 or info['scheduled_frames']!=100090:
        raise ValueError('Full capture population changed; inspect the new frozen schedule')
    full_specs=[case(None)]+finalists
    progress('full_evaluation',candidates=[s['id'] for s in full_specs]);execute(root,full,full_specs,settings)
    final=comparison(full,[s['id'] for s in full_specs])
    changed=[rel for rel,h in {**snapshots,**protected}.items() if digest(root/rel)!=h]
    if changed: raise RuntimeError(f'Source changed during run: {changed}')
    write_json(output/'completion.json',dict(complete=True,full_status=final['status'],
        fully_certified=all(record['certified'] for record in final['individual'].values()),
        scheduled_frames_per_model=100090,calibration_rows_per_model=89175))
    progress('complete',full_status=final['status'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--devices',type=int,nargs=2,default=[0,1])
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--batch-size',type=int,default=8192)
    parser.add_argument('--max-nfev',type=int,default=300)
    parser.add_argument('--validation-report',default='experiments/full_position/literal_power_validation/validation.json')
    args=vars(parser.parse_args());root,output=args.pop('root'),args.pop('output')
    if args['workers'] not in (2,4) or args['cpu_threads']<1 or args['workers']*args['cpu_threads']>mp.cpu_count():
        parser.error('Use two or four workers and a CPU thread budget that fits this host')
    if args['batch_size']<1 or args['max_nfev']<1: parser.error('Positive batch and solver budgets required')
    try:
        run(root,output,args)
    except Exception:
        if output.exists():
            (output/'exception.txt').write_text(traceback.format_exc())
            write_json(output/'progress.json',dict(utc=datetime.now(timezone.utc).isoformat(),status='failed',exception=traceback.format_exc()))
        raise


if __name__=='__main__': main()
