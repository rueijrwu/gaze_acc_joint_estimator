"""Fresh 36-fit sampled development screen for fixed accommodation laws.

The screen cannot select a deployment model. A denser nested study needs its
own frozen schedule, inner/outer split manifests and coverage policy.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import csv
import gzip
import hashlib
import json
import os
import platform
import shutil
import time
import traceback
import numpy as np
import scipy
from .accommodation import CANDIDATES, PowerResponseModel
from .accommodation_schema import artifact, load_model
from .accommodation_selection import declared_policy
from .audit83 import reviewed, supported
from .calibrate import fit
from .crosscheck import join_records
from .data import core_rows, fixed_sample, training_data, noise_blocks
from .geometry import context
from .information import predict_raw_holdout
from .invert import score_holdout
from .latest_audit import paired_summary, COHORTS
from .model import PositionModel
from .noise import coordinate_covariance, reference_covariance
from .population import manifest
from .schema import clean_json, source_hashes, write_json
from .scorecard import build
from .validate import fold_specs, pilot_fit


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def schedule(captures,groups,held,fold,count):
    rows,identities = [],[]
    for gi in held:
        group = groups[gi]; cap = captures[group["capture"]]
        core = core_rows(group); selected = set(map(int,fixed_sample(core,count)))
        for i in core:
            i = int(i)
            row = dict(fixation=gi,capture=cap.name,row=i,frame=int(cap.frame[i]),
                timestamp_ms=float(cap.timestamp[i]),selected_for_evaluation=i in selected,
                p1_valid_geometry=bool(cap.ctx.valid[i]),baseline_valid=bool(cap.baseline_valid[i]))
            row.update({f"p4_{j+1}_valid":bool(cap.point_valid[i,j]) for j in range(3)})
            rows.append(row)
            if i in selected:
                identities.append(dict(fold=fold,capture=cap.name,fixation=gi,row=i,
                    source_frame_index=int(cap.frame[i])))
    return rows,manifest(identities)


def trajectory(states,data,anchors):
    records=[]
    for local,gi in enumerate(sorted(set(map(int,data["original_group"])))):
        # local group numbering follows declared train IDs (sorted by runner).
        values=states[data["groups"]==local]; mean=values.mean(0)
        offset=mean-anchors[local]; spread=np.mean((values-mean)**2,0)
        mse=np.mean((values-anchors[local])**2,0)
        records.append(dict(group_id=gi,rows=len(values),mean=mean,nominal_mean_offset=offset,
            temporal_std=np.sqrt(spread),nominal_RMS=np.sqrt(mse),rms_identity_error=mse-offset**2-spread))
    return dict(groups=records,temporal_strength=0,nominal_errors_are_descriptive=True,
                finite_mean_anchor_scales=[.1,.25])


def _fit_task(job):
    root,output,fold,name,settings = job
    root,output=Path(root),Path(output)
    split=output/"splits"/fold; dest=output/"fits"/fold/name
    dest.mkdir(parents=True,exist_ok=False)
    info=read(split/"training.json")
    with np.load(split/"training_inputs.npz") as saved:
        data={k:saved[k].copy() for k in saved.files}
    anchors=data.pop("anchors"); cov=data.pop("covariance")
    pilot=PositionModel(27,info["pilot_coefficients"])
    sigma=np.asarray(info["coordinate_covariance"]); reference=np.asarray(info["reference_state"])
    model=PowerResponseModel(CANDIDATES[name])
    states=None
    with gzip.open(dest/"calibration_candidates.jsonl.gz","wt") as stream:
        def checkpoint(record,beta,physical_states):
            stream.write(json.dumps(clean_json(dict(record=record,coefficients=beta,states=physical_states)),allow_nan=False)+"\n")
            stream.flush()
        try:
            model,states,diagnostics=fit(model,data["v"],data["r"],cov,data["groups"],anchors,
                prior_strength=.001,anchor_scales=(.1,.25),curvature_strength=0.,
                starts=2,max_nfev=settings["max_nfev"],seed=settings["seed"],checkpoint=checkpoint,
                continuation_stages=2)
        except Exception as exc:
            diagnostics=dict(converged=False,exception=repr(exc),starts=[])
            (dest/"exception.txt").write_text(traceback.format_exc())
    meta=artifact(model,pilot,reference,sigma,info["provenance"],diagnostics,settings["coverage_policy"])
    accepted=diagnostics.get("converged") is True
    write_json(dest/("model.json" if accepted else "failed_checkpoint.json"),meta)
    if states is not None:
        write_json(dest/"training_states.json",dict(states=states,groups=data["groups"],
            original_group=data["original_group"],rows=data["rows"]))
        write_json(dest/"trajectory.json",trajectory(states,data,anchors))
    if accepted:
        # Strict schema round trip before the new family is used for prediction.
        model,_=load_model(dest/"model.json")
    captures,groups,_=reviewed(str(root))
    population=read(split/"population.json")
    schedule_rows=read(split/"manifest.json")["frames"]
    frames,holds=[],[]
    with gzip.open(dest/"inverse_candidates.jsonl.gz","wt") as archive:
        for row in population:
            if not row["selected_for_evaluation"]: continue
            gi,i=int(row["fixation"]),int(row["row"])
            cap=captures[row["capture"]]; ctx=context(cap.p[i])
            frames.append(dict(capture=cap.name,fixation=gi,row=i,frame=int(cap.frame[i]),
                estimated=False,nominal_theta=groups[gi]["target_theta_deg"],
                demand=groups[gi]["demand_diopters_label"],reason="subset_crosscheck_only"))
            for j in range(3):
                prediction=(predict_raw_holdout(model,cap.p[i],cap.q[i],cap.point_valid[i],j,
                    pilot,reference,sigma,"xy") if accepted else
                    dict(available=False,held_point=j,reason="calibration_failed",state=None,candidates=[]))
                candidates=prediction.pop("candidates",[])
                archive.write(json.dumps(clean_json(dict(fixation=gi,row=i,held_point=j,
                    candidates=candidates)),allow_nan=False)+"\n")
                # No held coordinate or label is read by the subset solver.
                scored=score_holdout(prediction,cap.q[i,j] if cap.point_valid[i,j] else np.full(2,np.nan),ctx.ell)
                scored.update(capture=cap.name,fixation=gi,row=i,retained_image_channels="xy")
                holds.append(scored)
    with (dest/"holdouts.jsonl").open("w") as stream:
        for record in holds: stream.write(json.dumps(clean_json(record),allow_nan=False)+"\n")
    joined=join_records(fold.split("_")[0],fold,name,population,frames,holds)
    supported(root,joined,dest,"xy")
    pop=manifest(schedule_rows)
    write_json(dest/"frames.json",joined)
    write_json(dest/"scorecard.json",build(joined,population=pop))
    write_json(dest/"completion.json",dict(complete=True,calibration_converged=accepted,
        scheduled_frames=len(pop.frame_ids),scheduled_slots=3*len(pop.frame_ids)))
    return dict(fold=fold,candidate=name,converged=accepted,exception=diagnostics.get("exception"))


def summarize(output):
    output=Path(output); config=read(output/"config.json")
    families={}
    with gzip.open(output/"paired_memberships.jsonl.gz","wt") as archive:
        for family in ("gaze","capture"):
            by_name={name:[] for name in CANDIDATES}; identities=[]
            for fold in config["folds"]:
                if not fold.startswith(family+"_"): continue
                identities.extend(read(output/"splits"/fold/"manifest.json")["frames"])
                for name in CANDIDATES:
                    by_name[name].extend(read(output/"fits"/fold/name/"frames.json"))
            pop=manifest(identities); comparisons={}
            for name in CANDIDATES:
                if name=="ar27_log": continue
                cohorts={}
                for cohort in COHORTS:
                    report,membership=paired_summary(by_name["ar27_log"],by_name[name],pop,cohort)
                    report["direction"]=name+"_minus_fresh_log"
                    report["scientific_scope"]="sampled_development_law_comparison_not_nested_selection"
                    cohorts[cohort]=report
                    archive.write(json.dumps(clean_json(dict(family=family,candidate=name,**membership)),allow_nan=False)+"\n")
                comparisons[name]=cohorts
            families[family]=dict(scorecards={name:build(rows,population=pop) for name,rows in by_name.items()},
                comparisons=comparisons,scheduled_exposures=pop.exposures)
    gains=[]
    for fold in config["folds"]:
        log_path=output/"fits"/fold/"ar27_log"/"training_states.json"
        if not log_path.exists(): continue
        log=read(log_path); ref=np.asarray(log["states"])
        for name in CANDIDATES:
            path=output/"fits"/fold/name/"training_states.json"
            if not path.exists(): continue
            obj=read(path)
            if obj["rows"]!=log["rows"] or obj["original_group"]!=log["original_group"]:
                raise ValueError("Training trajectories must match original IDs")
            state=np.asarray(obj["states"])
            base=np.column_stack((ref[:,0],1+ref[:,1])); new=np.column_stack((state[:,0],1+state[:,1]))
            gain=(base*new).sum(0)/(base*base).sum(0)
            gains.append(dict(fold=fold,candidate=name,positive_gains=gain.tolist() if np.all(gain>0) else None,
                residual_RMS=np.sqrt(np.mean((new-base*gain)**2,0)),
                diagnostic_only=True,evaluation_states_rescaled=False))
    write_json(output/"summary.json",dict(families=families,training_only_scale_diagnostics=gains,
        selected_for_outer_evaluation=None,promoted_for_deployment=False,
        scope="sampled_screen_only; do not rank a global winner from these outer development scores"))


def run(root,output,source_commit,workers=12,max_nfev=300,seed=17):
    root,output=Path(root).resolve(),Path(output).resolve()
    if not source_commit or len(source_commit)!=40:
        raise ValueError("Record the source commit plus the working implementation hashes")
    if workers<1 or workers>min(12,os.cpu_count() or 1):
        raise ValueError("Declare 1..12 CPU workers, capped by the available CPUs")
    if any(os.environ.get(k)!="1" for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS")):
        raise ValueError("Use one BLAS/OMP thread per independent worker")
    output.mkdir(parents=True,exist_ok=False)
    captures,groups,interval_hash=reviewed(str(root))
    implementation=source_hashes(root)
    for name in ("ACCOMMODATION_RESPONSE_THEORY.md","ACCOMMODATION_RESPONSE_PLAN.md"):
        implementation[f"docs/{name}"]=digest(root/"docs"/name)
    historic={str(p.relative_to(root)):digest(p) for p in (root/"experiments/full_position").rglob("*")
              if p.is_file() and output not in p.parents and "__pycache__" not in p.parts}
    for rel in implementation:
        dest=output/("implementation_snapshot" if rel.startswith("full_position/") else "design_snapshot")/Path(rel).name
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(root/rel,dest)
    folds=fold_specs(groups,("gaze","capture")); policy=declared_policy()
    config=dict(schema="accommodation_response_screen_v1",source_commit=source_commit,
        implementation_hashes=implementation,historical_source_hashes=historic,
        candidates=CANDIDATES,folds=[f for f,_ in folds],workers=workers,seed=seed,max_nfev=max_nfev,
        train_per_fixation=48,eval_per_fixation=8,sampling="central80_evenly_spaced_original_rows_before_validity",
        coverage_policy=policy,purpose="sampled_development_screen_not_nested_selection",
        pilot="fresh_fold_local_log27_shared_all_candidates",covariance="fixed_shared_training_only",
        anchor_scales=[.1,.25],prior_strength=.001,curvature_strength=0,temporal_strength=0,
        calibration_starts=2,continuation_stages=2,inverse_starts="common_49_start_certified_scalar",
        selected_for_outer_evaluation=None,promoted_for_deployment=False,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            system=platform.platform(),logical_cpus=os.cpu_count(),process_start_method="fork",
            OPENBLAS_NUM_THREADS=os.environ.get("OPENBLAS_NUM_THREADS"),OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS")))
    write_json(output/"config.json",config)
    split_manifest=[]
    started=time.monotonic()
    for fold,held in folds:
        train=sorted(set(range(len(groups)))-set(held)); split=output/"splits"/fold
        split.mkdir(parents=True)
        population,pop=schedule(captures,groups,held,fold,8)
        write_json(split/"population.json",population)
        write_json(split/"manifest.json",dict(frames=[dict(zip(("fold","capture","fixation","row","source_frame_index"),key))
            for key in pop.frame_ids],expected_exposures=pop.exposures,slots_per_frame=3,
            created_before_candidate_outputs=True))
        with gzip.open(split/"population.csv.gz","wt",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(population[0])); writer.writeheader(); writer.writerows(population)
        data,anchors=training_data(captures,groups,train,48)
        sigma,noise_meta=coordinate_covariance(noise_blocks(captures,groups,train))
        pilot=pilot_fit(data,anchors); reference=np.median(anchors,axis=0)
        cov=reference_covariance(data["p"],pilot,reference,sigma)
        np.savez_compressed(split/"training_inputs.npz",**data,anchors=anchors,covariance=cov)
        provenance=dict(interval_sha256=interval_hash,source_sha256=implementation,
            capture_sha256={n:c.sha256 for n,c in captures.items()},training_group_ids=train,evaluation_group_ids=held,
            training_groups=[groups[i] for i in train],sampled_training_rows=data["rows"].tolist(),
            sampled_training_group=data["original_group"].tolist(),sampling_policy=config["sampling"],noise=noise_meta,
            correspondence={n:c.metadata["pair_index"] for n,c in captures.items()},
            detector_configuration={n:c.metadata.get("config") for n,c in captures.items()},
            detector_location_definition="stored selected native coordinates; fitted-center implementation unavailable",
            detector_implementation_version=None,source_grid_geometry=None,seed=seed,runtime=config["runtime"],
            anchor_range=[anchors.min(0).tolist(),anchors.max(0).tolist()],
            conditional_context_support=dict(r_min=data["r"].min(0).tolist(),r_max=data["r"].max(0).tolist(),
                signed_P1_area_branches=np.unique(np.sign(context(data["p"]).signed_area)).tolist()))
        write_json(split/"training.json",dict(provenance=provenance,pilot_coefficients=pilot.beta,
            reference_state=reference,coordinate_covariance=sigma))
        split_manifest.append(dict(fold=fold,training_group_ids=train,evaluation_group_ids=held,
            family=fold.split("_")[0],training_input_sha256=digest(split/"training_inputs.npz"),
            schedule_sha256=digest(split/"manifest.json"),nested_selection=False))
    write_json(output/"split_manifest.json",split_manifest)
    settings={k:config[k] for k in ("max_nfev","seed","coverage_policy")}
    jobs=[(root,output,fold,name,settings) for fold,_ in folds for name in CANDIDATES]
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context("fork")) as pool:
        outcomes=list(pool.map(_fit_task,jobs))
    write_json(output/"outcomes.json",outcomes)
    summarize(output)
    if any(digest(root/rel)!=h for rel,h in {**historic,**implementation}.items()):
        raise RuntimeError("Frozen implementation or historical source changed during the screen")
    write_json(output/"selection.json",dict(status="development_screen_only",selected_for_outer_evaluation=None,
        promoted_for_deployment=False,policy=policy,nested_study="not_executed; freeze denser schedules first"))
    write_json(output/"completion.json",dict(complete=True,seconds=time.monotonic()-started,
        fit_tasks=len(jobs),certified=sum(o["converged"] for o in outcomes),historical_files_unchanged=len(historic)))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--source-commit",required=True)
    parser.add_argument("--workers",type=int,default=12)
    parser.add_argument("--max-nfev",type=int,default=300)
    parser.add_argument("--seed",type=int,default=17)
    args=parser.parse_args()
    run(args.root,args.output,args.source_commit,args.workers,args.max_nfev,args.seed)
