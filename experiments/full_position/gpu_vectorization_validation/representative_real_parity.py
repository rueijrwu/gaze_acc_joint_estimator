"""180 representative raw inverse parity checks from all fixation groups."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import cupy as cp

from full_position.accommodation import PowerResponseModel
from full_position.batched_holdout import predict_batch
from full_position.information import predict_raw_holdout
from full_position.invert import STARTS
from full_position.model import PositionModel


ROOT = Path("experiments/full_position/accommodation_full_v2_gpu")
OUT = Path("experiments/full_position/gpu_vectorization_validation/representative_real_parity.json")
LAWS = (("ar27_log", 0.), ("ar27_sqrt", .5), ("ar27_quadratic", 2.))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scalar(task):
    model, p, q, valid, held, pilot, reference, sigma = task
    return predict_raw_holdout(model, p, q, valid, held, pilot, reference, sigma,
                               "xy", starts=STARTS)


def check(gpu, cpu):
    states, costs, status_pairs, nfev_diffs = [], [], Counter(), []
    acceptance_diff = branch_diff = 0
    for row, (a, b) in enumerate(zip(gpu, cpu)):
        if a["available"] != b["available"]:
            raise AssertionError(f"availability mismatch row={row}: {a['available']} {b['available']}")
        if not a["available"]:
            continue
        meta = ("reason", "rank", "ambiguous", "at_bound")
        if any(a[key] != b[key] for key in meta):
            raise AssertionError(f"branch metadata mismatch row={row}")
        if (len(a["branches"]), len(a["plausible_branches"])) != (
                len(b["branches"]), len(b["plausible_branches"])):
            branch_diff += 1
            raise AssertionError(f"branch count mismatch row={row}")
        np.testing.assert_allclose(a["state"], b["state"], rtol=5e-6, atol=6e-5)
        np.testing.assert_allclose(a["cost"], b["cost"], rtol=7e-6, atol=7e-7)
        states.extend(np.abs(np.asarray(a["state"])-np.asarray(b["state"])).tolist())
        costs.append(abs(a["cost"]-b["cost"]))
        for ca, cb in zip(a["candidates"], b["candidates"]):
            acceptance_diff += int(ca["accepted"] != cb["accepted"])
            status_pairs[(ca["initial"]["status"], cb["initial"]["status"])] += 1
            nfev_diffs.append(abs(ca["initial"]["nfev"]-cb["initial"]["nfev"]))
            if ca["accepted"] != cb["accepted"]:
                raise AssertionError(f"certificate mismatch row={row} start={ca['start']}")
            np.testing.assert_allclose(ca["state"], cb["state"], rtol=5e-6, atol=6e-5)
            np.testing.assert_allclose(ca["cost"], cb["cost"], rtol=7e-6, atol=7e-7)
    return dict(max_abs_final_state_error=max(states, default=0.),
        max_abs_final_cost_error=max(costs, default=0.), candidate_certificate_mismatches=acceptance_diff,
        branch_count_mismatches=branch_diff,
        initial_status_pair_counts={f"gpu{g}_cpu{c}": n for (g,c),n in status_pairs.items()},
        initial_status_mismatches=sum(n for (g,c),n in status_pairs.items() if g!=c),
        initial_nfev_mismatch_count=sum(x!=0 for x in nfev_diffs),
        initial_nfev_max_abs_difference=max(nfev_diffs,default=0.))


def main():
    info=json.loads((ROOT/"splits/full_calibration/training.json").read_text())
    pilot=PositionModel(27,info["pilot_coefficients"])
    reference,sigma=np.asarray(info["reference_state"]),np.asarray(info["coordinate_covariance"])
    with np.load(ROOT/"splits/full_calibration/training_inputs.npz") as saved:
        groups=saved["original_group"].copy()
        # A deterministic central row from each of the twenty fixation groups.
        selected=np.array([np.flatnonzero(groups==g)[np.flatnonzero(groups==g).size//2]
                           for g in range(20)])
        p,q=(saved[k][selected].copy() for k in ("p","q"))
    valid=np.ones((20,3),bool)
    held=np.tile(np.arange(3),20)
    raw_p=np.repeat(p,3,axis=0); raw_q=np.repeat(q,3,axis=0); raw_valid=np.repeat(valid,3,axis=0)
    tasks_by_law={}; models={}
    for name,exponent in LAWS:
        artifact=json.loads((ROOT/f"fits/full_calibration/{name}/model.json").read_text())
        model=PowerResponseModel(exponent,artifact["coefficients"])
        models[name]=model
        tasks_by_law[name]=[(model,raw_p[i],raw_q[i],raw_valid[i],int(held[i]),pilot,reference,sigma)
                            for i in range(60)]

    # Start the GPU context on both devices with an uneven two-row split.
    model=models["ar27_sqrt"]
    warm_start=time.perf_counter()
    predict_batch(model,raw_p[:2],raw_q[:2],raw_valid[:2],held[:2],pilot,reference,sigma,
                  backend="cupy",devices=[0,1],starts=STARTS)
    cp.cuda.runtime.deviceSynchronize()
    warm_seconds=time.perf_counter()-warm_start

    cpu_start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context("spawn")) as pool:
        cpu_by_law={name:list(pool.map(_scalar,tasks,chunksize=1))
                    for name,tasks in tasks_by_law.items()}
    cpu_seconds=time.perf_counter()-cpu_start

    gpu_by_law={}; timings={}
    for name,_ in LAWS:
        model=models[name]
        start=time.perf_counter()
        gpu_by_law[name]=predict_batch(model,raw_p,raw_q,raw_valid,held,pilot,reference,sigma,
                                        backend="cupy",devices=[0,1],starts=STARTS)
        cp.cuda.runtime.deviceSynchronize()
        timings[name]=time.perf_counter()-start
    per_law={name:check(gpu_by_law[name],cpu_by_law[name]) for name,_ in LAWS}
    report=dict(schema="representative_real_parity_v1",model_laws=[n for n,_ in LAWS],
        input_rows=len(selected),groups=20,slots_per_group=3,total_checks=180,
        selected_input_npz_indices=selected.tolist(),held_order=held.tolist(),start_count=len(STARTS),
        gpu_backend="cupy",gpu_devices=[0,1],cpu_backend="original_scalar_invert",cpu_workers=8,
        device_names=[cp.cuda.runtime.getDeviceProperties(i)["name"].decode() for i in (0,1)],
        gpu_seconds_by_law=timings,gpu_total_seconds=sum(timings.values()),
        gpu_total_rows_per_second=180/sum(timings.values()),cpu_8worker_seconds=cpu_seconds,
        cpu_8worker_rows_per_second=180/cpu_seconds,gpu_warmup_seconds=warm_seconds,
        per_law=per_law,source_sha256={f:sha(f) for f in ("full_position/batched_inverse.py",
            "full_position/batched_holdout.py","full_position/torch_arrays.py")},
        training_inputs_sha256=sha(ROOT/"splits/full_calibration/training_inputs.npz"),
        fit_model_sha256={name:sha(ROOT/f"fits/full_calibration/{name}/model.json") for name,_ in LAWS})
    report["passed"]=all(value["candidate_certificate_mismatches"]==0 and
                           value["branch_count_mismatches"]==0 for value in per_law.values())
    OUT.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
