"""Compare one- and two-GPU raw inverse throughput on 20-group rows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import cupy as cp
import numpy as np

from full_position.accommodation import PowerResponseModel
from full_position.batched_holdout import predict_batch
from full_position.invert import STARTS
from full_position.model import PositionModel


ROOT = Path("experiments/full_position/accommodation_full_v2_gpu")
OUT = Path("experiments/full_position/gpu_vectorization_validation/groupstratified_benchmark.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    model_data=json.loads((ROOT/"fits/full_calibration/ar27_sqrt/model.json").read_text())
    model=PowerResponseModel(.5,model_data["coefficients"])
    info=json.loads((ROOT/"splits/full_calibration/training.json").read_text())
    pilot=PositionModel(27,info["pilot_coefficients"])
    reference,sigma=np.asarray(info["reference_state"]),np.asarray(info["coordinate_covariance"])
    with np.load(ROOT/"splits/full_calibration/training_inputs.npz") as data:
        groups=data["original_group"].copy()
        chosen=[]; counts=[]
        for g in range(20):
            available=np.flatnonzero(groups==g)
            count=4096//20+(g<4096%20)
            counts.append(int(count))
            chosen.append(available[np.linspace(0,len(available)-1,count).round().astype(int)])
        indices=np.array([chosen[g][j] for j in range(max(counts))
                          for g in range(20) if j<len(chosen[g])],dtype=int)
        p,q=data["p"][indices].copy(),data["q"][indices].copy()
    valid,held=np.ones((len(indices),3),bool),np.arange(len(indices))%3

    benchmarks={}
    for label,kwargs in (("gpu0_only",dict(backend="cupy",device=0)),
                         ("both_gpus",dict(backend="cupy",devices=[0,1]))):
        warm=predict_batch(model,p[:8],q[:8],valid[:8],held[:8],pilot,reference,sigma,
                           starts=STARTS,**kwargs)
        cp.cuda.runtime.deviceSynchronize(); del warm
        start=time.perf_counter()
        results=predict_batch(model,p,q,valid,held,pilot,reference,sigma,starts=STARTS,**kwargs)
        cp.cuda.runtime.deviceSynchronize(); elapsed=time.perf_counter()-start
        candidates=[c for row in results for c in row.get("candidates",[])]
        benchmarks[label]=dict(seconds=elapsed,rows_per_second=len(p)/elapsed,
            available_rows=sum(bool(row["available"]) for row in results),
            candidate_count=len(candidates),
            certified_candidates=sum(bool(c["accepted"]) for c in candidates),
            initial_status_counts={str(s):sum(c["initial"]["status"]==s for c in candidates)
                                   for s in sorted({c["initial"]["status"] for c in candidates})})
        del results,candidates

    report=dict(schema="group_stratified_gpu_comparison_v1",model="ar27_sqrt",dtype="float64",
        start_count=len(STARTS),rows=len(indices),groups=20,rows_per_group=counts,
        group_row_indices=indices.tolist(),held_order_counts={str(i):int(np.sum(held==i)) for i in range(3)},
        warmup_batch_size=8,benchmark=benchmarks,
        device_names=[cp.cuda.runtime.getDeviceProperties(i)["name"].decode() for i in (0,1)],
        source_sha256={f:sha(f) for f in ("full_position/batched_inverse.py",
            "full_position/batched_holdout.py","full_position/torch_arrays.py")})
    OUT.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="group_row_indices"},indent=2))


if __name__=="__main__": main()
