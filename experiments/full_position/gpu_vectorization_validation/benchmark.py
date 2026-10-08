"""Initial warmed CuPy throughput/parity probe for batched inverse."""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import cupy as cp

from full_position.accommodation import PowerResponseModel
from full_position.batched_inverse import solve_batch
from full_position.invert import STARTS, invert


ROOT = Path("experiments/full_position/accommodation_full_v2_gpu")
OUT = Path("experiments/full_position/gpu_vectorization_validation")
NAME = "ar27_sqrt"


def timed(function):
    start = time.perf_counter()
    result = function()
    cp.cuda.get_current_stream().synchronize()
    return result, time.perf_counter()-start


def main():
    artifact = json.loads((ROOT/"fits/full_calibration"/NAME/"model.json").read_text())
    model = PowerResponseModel(.5, artifact["coefficients"])
    with np.load(ROOT/"splits/full_calibration/training_inputs.npz") as saved:
        ix = np.arange(500, 756)
        r, y, cov = saved["r"][ix].copy(), saved["v"][ix].copy(), saved["covariance"][ix].copy()

    # Initialize the selected device/context and warm the complete 49-start
    # path before timing representative batch sizes.
    _, warm_s = timed(lambda: solve_batch(model, r[:2], y[:2], cov[:2],
                                          backend="cupy", device=0, starts=STARTS))
    gpu = {}
    for batch_size in (16, 64, 256):
        result, seconds = timed(lambda n=batch_size: solve_batch(
            model, r[:n], y[:n], cov[:n], backend="cupy", device=0, starts=STARTS))
        gpu[str(batch_size)] = {"seconds": seconds, "rows_per_second": batch_size/seconds,
                                "results": result}

    cpu, cpu_seconds = None, None
    cpu = []
    start = time.perf_counter()
    for i in range(16):
        cpu.append(invert(model, r[i], y[i], cov[i], starts=STARTS))
    cpu_seconds = time.perf_counter()-start

    batched = gpu["16"]["results"]
    state_errors, cost_errors, accepted_differences, status_differences = [], [], 0, 0
    status_pairs, nfev_differences, first_status_differences = Counter(), [], []
    for frame, (a, b) in enumerate(zip(batched, cpu)):
        for ca, cb in zip(a["candidates"], b["candidates"]):
            state_errors.extend(np.abs(np.asarray(ca["state"])-np.asarray(cb["state"])).tolist())
            cost_errors.append(abs(float(ca["cost"])-float(cb["cost"])))
            accepted_differences += int(ca["accepted"] != cb["accepted"])
            gs, cs = ca["initial"]["status"], cb["initial"]["status"]
            status_differences += int(gs != cs)
            status_pairs[f"gpu{gs}_cpu{cs}"] += 1
            nd = abs(ca["initial"]["nfev"]-cb["initial"]["nfev"])
            nfev_differences.append(nd)
            if gs != cs and len(first_status_differences) < 20:
                first_status_differences.append({"frame": frame, "start": ca["start"],
                    "gpu_status": gs, "cpu_status": cs,
                    "gpu_nfev": ca["initial"]["nfev"], "cpu_nfev": cb["initial"]["nfev"]})
    report = {
        "schema": "gpu_vectorization_initial_benchmark_v1",
        "model": NAME,
        "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        "start_count": len(STARTS),
        "row_indices": {"warmed_gpu": list(map(int, ix[:256])), "cpu_comparison": list(map(int, ix[:16]))},
        "warmup_batch2_seconds": warm_s,
        "cupy_batches": {k: {"seconds": v["seconds"], "rows_per_second": v["rows_per_second"]}
                         for k, v in gpu.items()},
        "cpu_scalar_batch16_seconds": cpu_seconds,
        "cpu_rows_per_second": 16/cpu_seconds,
        "batch16_cpu_vs_gpu": {
            "max_abs_physical_state_error": float(max(state_errors, default=0.)),
            "max_abs_cost_error": float(max(cost_errors, default=0.)),
            "candidate_acceptance_mismatches": accepted_differences,
            "initial_status_mismatches": status_differences,
            "initial_status_pair_counts": dict(status_pairs),
            "initial_nfev_mismatch_count": sum(value != 0 for value in nfev_differences),
            "initial_nfev_max_abs_difference": max(nfev_differences, default=0),
            "first_initial_status_mismatches": first_status_differences,
            "gpu_available": [bool(x["available"]) for x in batched],
            "cpu_available": [bool(x["available"]) for x in cpu],
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"initial_benchmark.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cupy_batches"}, indent=2))
    print("cupy_batches", json.dumps(report["cupy_batches"], indent=2))


if __name__ == "__main__":
    main()
