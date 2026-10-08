"""Same actual raw rows split over both CUDA devices with 49 starts."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np

from full_position.accommodation import PowerResponseModel
from full_position.batched_holdout import predict_batch
from full_position.invert import STARTS
from full_position.model import PositionModel


ROOT = Path("experiments/full_position/accommodation_full_v2_gpu")
OUT = Path("experiments/full_position/gpu_vectorization_validation/multi_gpu_benchmark.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    artifacts = json.loads((ROOT/"fits/full_calibration/ar27_sqrt/model.json").read_text())
    model = PowerResponseModel(.5, artifacts["coefficients"])
    info = json.loads((ROOT/"splits/full_calibration/training.json").read_text())
    pilot = PositionModel(27, info["pilot_coefficients"])
    reference, sigma = np.asarray(info["reference_state"]), np.asarray(info["coordinate_covariance"])
    with np.load(ROOT/"splits/full_calibration/training_inputs.npz") as data:
        p, q = data["p"][:4096].copy(), data["q"][:4096].copy()
    valid, held = np.ones((4096, 3), bool), np.arange(4096) % 3
    warm = predict_batch(model, p[:8], q[:8], valid[:8], held[:8], pilot, reference, sigma,
                         backend="cupy", devices=[0, 1], starts=STARTS)
    cp.cuda.runtime.deviceSynchronize(); del warm
    free0, total0 = cp.cuda.runtime.memGetInfo()
    start = time.perf_counter()
    results = predict_batch(model, p, q, valid, held, pilot, reference, sigma,
                            backend="cupy", devices=[0, 1], starts=STARTS)
    cp.cuda.runtime.deviceSynchronize()
    seconds = time.perf_counter()-start
    free1, total1 = cp.cuda.runtime.memGetInfo()
    candidates = [c for row in results for c in row.get("candidates", [])]
    report = dict(schema="two_gpu_raw_benchmark_v1", backend="cupy", dtype="float64",
        start_count=len(STARTS), rows=4096, seconds=seconds, rows_per_second=4096/seconds,
        model="ar27_sqrt", training_input_rows=[0,4095], device_runtime_indices=[0,1],
        device_names=[cp.cuda.runtime.getDeviceProperties(i)["name"].decode() for i in (0,1)],
        memory_device0_before_bytes=total0-free0, memory_device0_after_bytes=total1-free1,
        available_rows=sum(bool(x["available"]) for x in results), candidate_count=len(candidates),
        certified_candidates=sum(bool(c["accepted"]) for c in candidates),
        source_sha256={f: sha(f) for f in ("full_position/batched_inverse.py",
            "full_position/batched_holdout.py", "full_position/torch_arrays.py")})
    OUT.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
