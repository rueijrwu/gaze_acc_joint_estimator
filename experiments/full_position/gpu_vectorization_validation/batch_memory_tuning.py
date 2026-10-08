"""Size GPU batches under the four-law concurrent memory workload."""
from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import time

OUT=Path(__file__).with_name("batch_memory_tuning.json")
ROOT=Path("experiments/full_position/accommodation_full_v2_gpu")
RESERVE_GIB=3.
POOL_LIMIT_GIB=6.75


def worker(batch,device,slot):
    import cupy as cp
    import numpy as np
    from threadpoolctl import threadpool_limits
    from full_position.accommodation import PowerResponseModel
    from full_position.batched_holdout import predict_batch
    from full_position.model import PositionModel
    cp.cuda.Device(device).use()
    pool=cp.get_default_memory_pool()
    pool.set_limit(size=int(POOL_LIMIT_GIB*2**30))
    result=dict(batch_size=batch,device=device,slot=slot,reserve_gib=RESERVE_GIB,pool_limit_gib=POOL_LIMIT_GIB)
    try:
        reserved=cp.empty(int(RESERVE_GIB*2**30),dtype=cp.uint8)
        reserved[:1]=0
        meta=json.loads((ROOT/"fits/full_calibration/ar27_sqrt/model.json").read_text())
        model=PowerResponseModel(.5,meta["coefficients"])
        info=json.loads((ROOT/"splits/full_calibration/training.json").read_text())
        pilot=PositionModel(27,info["pilot_coefficients"])
        ref,sigma=np.asarray(info["reference_state"]),np.asarray(info["coordinate_covariance"])
        with np.load(ROOT/"splits/full_calibration/training_inputs.npz") as data:
            groups=data["original_group"]
            chosen=[]
            for g in range(20):
                rows=np.flatnonzero(groups==g)
                count=batch//20+(g<batch%20)
                chosen.append(rows[np.linspace(0,len(rows)-1,count).round().astype(int)])
            ix=np.array([chosen[g][j] for j in range(max(map(len,chosen))) for g in range(20) if j<len(chosen[g])])
            p,q=data["p"][ix].copy(),data["q"][ix].copy()
        valid=np.ones((batch,3),bool); held=np.arange(batch)%3
        with threadpool_limits(limits=4):
            warm=predict_batch(model,p[:16],q[:16],valid[:16],held[:16],pilot,ref,sigma,device=device)
            del warm; gc.collect(); cp.cuda.runtime.deviceSynchronize()
            started=time.perf_counter()
            rows=predict_batch(model,p,q,valid,held,pilot,ref,sigma,device=device)
            cp.cuda.runtime.deviceSynchronize()
            result.update(passed=True,seconds=time.perf_counter()-started,
                          candidate_count=sum(len(row.get("candidates",[])) for row in rows),
                          certified_candidates=sum(c["accepted"] for row in rows for c in row.get("candidates",[])),
                          available_rows=sum(row["available"] for row in rows),
                          pool_reserved_gib=pool.total_bytes()/2**30,
                          representative=[dict(state=row.get("state"),cost=row.get("cost"),available=row["available"]) for row in rows[:3]])
            result["checks_per_second"]=batch/result["seconds"]
    except cp.cuda.memory.OutOfMemoryError as exc:
        result.update(passed=False,reason="allocation_cap_exceeded",detail=str(exc))
    return result


def memory_sample():
    value=subprocess.check_output(["nvidia-smi","--query-gpu=index,name,memory.used,memory.free","--format=csv,noheader,nounits"],text=True)
    return [dict(index=int(parts[0]),name=parts[1],used_mib=int(parts[2]),free_mib=int(parts[3]))
            for line in value.strip().splitlines() if (parts:=[p.strip() for p in line.split(",")])]


def main():
    report=dict(reserve_gib_per_worker=RESERVE_GIB,pool_limit_gib_per_worker=POOL_LIMIT_GIB,
                workers_per_gpu=2,devices=[0,1],start_count=49,groups=20,trials=[],
                source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                               ("full_position/batched_inverse.py","full_position/batched_holdout.py")})
    for batch in (8192,16384,32768):
        peaks={}
        start=time.perf_counter()
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context("spawn")) as pool:
            tasks=[pool.submit(worker,batch,device,slot) for device in (0,1) for slot in (0,1)]
            while not all(f.done() for f in tasks):
                for sample in memory_sample():
                    index=sample["index"]
                    if index not in peaks or sample["used_mib"]>peaks[index]["used_mib"]:
                        peaks[index]=sample
                time.sleep(.3)
            results=[task.result() for task in tasks]
        trial=dict(batch_size=batch,seconds=time.perf_counter()-start,workers=results,peak_memory=list(peaks.values()))
        trial["passed"]=all(r["passed"] and r.get("candidate_count")==49*batch for r in results)
        trial["memory_headroom_passed"]=all(p["free_mib"]>=1536 for p in peaks.values())
        report["trials"].append(trial)
        OUT.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(trial),flush=True)
        if not trial["passed"] or not trial["memory_headroom_passed"]: break
    passed=[t for t in report["trials"] if t["passed"] and t["memory_headroom_passed"]]
    report["largest_memory_safe_batch"]=passed[-1]["batch_size"] if passed else None
    # Select speed among memory-safe sizes, based on the actual concurrent
    # workload makespan, rather than summed isolated worker throughput.
    best=max(passed,key=lambda t:4*t["batch_size"]/t["seconds"]) if passed else None
    report["fastest_measured_batch_size"]=best["batch_size"] if best else None
    # Prefer the largest tested safe batch within 5% of best throughput.
    # The stratified subsets differ, so small timing differences are approximate.
    rate=4*best["batch_size"]/best["seconds"] if best else 0
    eligible=[t for t in passed if 4*t["batch_size"]/t["seconds"]>=.95*rate]
    report["selection_policy"]="largest safe tested batch within 5% of best measured throughput"
    report["selected_batch_size"]=max(t["batch_size"] for t in eligible) if eligible else None
    OUT.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"selected_batch_size":report["selected_batch_size"],"largest_memory_safe_batch":report["largest_memory_safe_batch"]}),flush=True)


if __name__=="__main__": main()
