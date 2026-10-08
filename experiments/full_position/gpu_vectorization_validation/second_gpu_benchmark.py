"""Same-input full raw batch probe on the second CUDA device."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np

from full_position.accommodation import PowerResponseModel
from full_position.batched_holdout import predict_batch
from full_position.model import PositionModel
from full_position.invert import STARTS


ROOT = Path("experiments/full_position/accommodation_full_v2_gpu")
OUT = Path("experiments/full_position/gpu_vectorization_validation/second_gpu_benchmark.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    device = 1  # runtime order: 0=RTX 5070 Ti, 1=RTX PRO 2000 Blackwell
    cp.cuda.Device(device).use()
    props = cp.cuda.runtime.getDeviceProperties(device)
    name = props["name"].decode()
    artifact = json.loads((ROOT/"fits/full_calibration/ar27_sqrt/model.json").read_text())
    model = PowerResponseModel(.5, artifact["coefficients"])
    info = json.loads((ROOT/"splits/full_calibration/training.json").read_text())
    pilot = PositionModel(27, info["pilot_coefficients"])
    reference, sigma = np.asarray(info["reference_state"]), np.asarray(info["coordinate_covariance"])
    with np.load(ROOT/"splits/full_calibration/training_inputs.npz") as data:
        p, q = data["p"][:4096].copy(), data["q"][:4096].copy()
    valid, held = np.ones((4096, 3), bool), np.arange(4096) % 3
    warm = predict_batch(model, p[:8], q[:8], valid[:8], held[:8], pilot,
                         reference, sigma, backend="cupy", device=device, starts=STARTS)
    cp.cuda.get_current_stream().synchronize()
    del warm
    before_free, before_total = cp.cuda.runtime.memGetInfo()
    before_used = before_total-before_free
    start = time.perf_counter()
    results = predict_batch(model, p, q, valid, held, pilot, reference, sigma,
                            backend="cupy", device=device, starts=STARTS)
    cp.cuda.get_current_stream().synchronize()
    seconds = time.perf_counter()-start
    after_free, after_total = cp.cuda.runtime.memGetInfo()
    candidates = [c for row in results for c in row.get("candidates", [])]
    report = dict(schema="second_cuda_device_raw_benchmark_v1", backend="cupy", dtype="float64",
        start_count=len(STARTS), rows=4096, rows_per_second=4096/seconds, seconds=seconds,
        model="ar27_sqrt", training_input_rows=[0,4095], device_runtime_index=device,
        device_name=name, memory_before_bytes=before_used,
        memory_after_bytes=after_total-after_free,
        available_rows=sum(bool(x["available"]) for x in results),
        candidate_count=len(candidates), certified_candidates=sum(bool(c["accepted"]) for c in candidates),
        source_sha256={f: digest(f) for f in ("full_position/batched_inverse.py",
            "full_position/batched_holdout.py", "full_position/torch_arrays.py")})
    OUT.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
