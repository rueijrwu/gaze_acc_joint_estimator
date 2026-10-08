"""Raw end-to-end CuPy/Torch throughput and scalar worker comparison."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import cupy as cp
import torch

from full_position.accommodation import PowerResponseModel
from full_position.batched_holdout import predict_batch
from full_position.information import predict_raw_holdout
from full_position.invert import STARTS
from full_position.model import PositionModel


ROOT = Path("experiments/full_position/accommodation_full_v2_gpu")
OUT = Path("experiments/full_position/gpu_vectorization_validation")
OUT.mkdir(parents=True, exist_ok=True)
SNAP = OUT/"source_snapshot"
SNAP.mkdir(parents=True, exist_ok=True)
SOURCE_FILES = ("full_position/batched_inverse.py", "full_position/batched_holdout.py",
                "full_position/torch_arrays.py")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync(backend):
    if backend == "cupy":
        cp.cuda.get_current_stream().synchronize()
    else:
        torch.cuda.synchronize(0)


def _cpu_predict(args):
    return predict_raw_holdout(*args)


def compare(gpu, cpu):
    state_errors, cost_errors, status_pairs, nfev_diffs = [], [], {}, []
    accept_diff = 0
    for row, (a, b) in enumerate(zip(gpu, cpu)):
        if a["available"] != b["available"]:
            raise AssertionError(f"availability mismatch at row {row}")
        if not a["available"]:
            continue
        if (a["ambiguous"], a["rank"], a["at_bound"]) != (b["ambiguous"], b["rank"], b["at_bound"]):
            raise AssertionError(f"branch metadata mismatch at row {row}")
        if (len(a.get("branches", [])), len(a.get("plausible_branches", []))) != (
                len(b.get("branches", [])), len(b.get("plausible_branches", []))):
            raise AssertionError(f"branch count mismatch at row {row}")
        state_errors.extend(np.abs(np.asarray(a["state"])-np.asarray(b["state"])).tolist())
        cost_errors.append(abs(float(a["cost"])-float(b["cost"])))
        for ca, cb in zip(a["candidates"], b["candidates"]):
            accept_diff += int(ca["accepted"] != cb["accepted"])
            key = f"gpu{ca['initial']['status']}_cpu{cb['initial']['status']}"
            status_pairs[key] = status_pairs.get(key, 0)+1
            nfev_diffs.append(abs(ca["initial"]["nfev"]-cb["initial"]["nfev"]))
            if ca["accepted"] != cb["accepted"]:
                raise AssertionError(f"candidate certificate mismatch at row {row}, start {ca['start']}")
            np.testing.assert_allclose(ca["state"], cb["state"], rtol=4e-6, atol=5e-5)
            np.testing.assert_allclose(ca["cost"], cb["cost"], rtol=6e-6, atol=6e-7)
        np.testing.assert_allclose(a["state"], b["state"], rtol=4e-6, atol=5e-5)
        np.testing.assert_allclose(a["cost"], b["cost"], rtol=6e-6, atol=6e-7)
    return dict(max_abs_state_error=max(state_errors, default=0.),
        max_abs_cost_error=max(cost_errors, default=0.), candidate_certificate_mismatches=accept_diff,
        initial_status_pair_counts=status_pairs,
        initial_status_mismatches=sum(k.split("_")[0][3:] != k.split("_")[1][3:] for k in status_pairs for _ in range(status_pairs[k])),
        initial_nfev_mismatch_count=sum(x != 0 for x in nfev_diffs),
        initial_nfev_max_abs_difference=max(nfev_diffs, default=0))


def main():
    for name in SOURCE_FILES:
        destination = SNAP/Path(name).name
        destination.write_bytes(Path(name).read_bytes())
    source_hashes = {name: _sha(name) for name in SOURCE_FILES}

    artifact = json.loads((ROOT/"fits/full_calibration/ar27_sqrt/model.json").read_text())
    model = PowerResponseModel(.5, artifact["coefficients"])
    info = json.loads((ROOT/"splits/full_calibration/training.json").read_text())
    pilot = PositionModel(27, info["pilot_coefficients"])
    reference, sigma = np.asarray(info["reference_state"]), np.asarray(info["coordinate_covariance"])
    with np.load(ROOT/"splits/full_calibration/training_inputs.npz") as data:
        p, q = data["p"][:4096].copy(), data["q"][:4096].copy()
    valid = np.ones((len(p), 3), dtype=bool)
    held = np.arange(len(p), dtype=int) % 3

    output = dict(schema="raw_gpu_torch_benchmark_v1", model="ar27_sqrt", dtype="float64",
        autograd_used=False, start_count=len(STARTS), row_indices=[0, 4095],
        source_sha256=source_hashes,
        source_snapshot={"directory": str(SNAP), "captured_before_benchmark": True},
        device="NVIDIA GeForce RTX 5070 Ti", batch_results={})
    # Warm each backend using the complete raw geometry/noise/solver/assembly path.
    for backend in ("cupy", "torch"):
        warm = predict_batch(model, p[:8], q[:8], valid[:8], held[:8], pilot,
                             reference, sigma, backend=backend, device=0, starts=STARTS)
        sync(backend)
        del warm
        backend_results = {}
        for size in (256, 1024, 4096):
            start = time.perf_counter()
            results = predict_batch(model, p[:size], q[:size], valid[:size], held[:size],
                                    pilot, reference, sigma, backend=backend,
                                    device=0, starts=STARTS)
            sync(backend)
            seconds = time.perf_counter()-start
            candidates = [c for result in results for c in result.get("candidates", [])]
            backend_results[str(size)] = dict(seconds=seconds, rows_per_second=size/seconds,
                available_rows=sum(bool(x["available"]) for x in results),
                certified_candidates=sum(bool(c["accepted"]) for c in candidates),
                candidate_count=len(candidates),
                status_counts={str(status): sum(c["initial"]["status"] == status for c in candidates)
                               for status in sorted({c["initial"]["status"] for c in candidates})})
            del candidates, results
            output["batch_results"][backend] = backend_results
            (OUT/"raw_backend_benchmark_progress.json").write_text(json.dumps(output, indent=2)+"\n")
        output["batch_results"][backend] = backend_results
    # A fixed 24-row true-raw subset supports identical CPU 8/24 worker runs.
    args = [(model, p[i], q[i], valid[i], int(held[i]), pilot, reference, sigma, "xy", STARTS)
            for i in range(24)]
    cpu_results = {}
    for workers in (8, 24):
        start = time.perf_counter()
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                mp_context=mp.get_context("spawn")) as pool:
            results = list(pool.map(_cpu_predict, args))
        seconds = time.perf_counter()-start
        cpu_results[str(workers)] = dict(seconds=seconds, rows_per_second=24/seconds,
                                         available_rows=sum(bool(x["available"]) for x in results),
                                         results=results)
    # Re-evaluate the fixed subset on each device backend for a direct raw parity
    # comparison and to avoid retaining/serializing thousands of benchmark rows.
    gpu24 = predict_batch(model, p[:24], q[:24], valid[:24], held[:24], pilot,
                          reference, sigma, backend="cupy", starts=STARTS)
    torch24 = predict_batch(model, p[:24], q[:24], valid[:24], held[:24], pilot,
                            reference, sigma, backend="torch", starts=STARTS)
    cupy_status_semantics = compare(gpu24, cpu_results["24"]["results"])
    torch_status_semantics = compare(torch24, cpu_results["24"]["results"])
    output["cpu_worker_batches_24_rows"] = {
        k: {key: value for key, value in result.items() if key != "results"}
        for k, result in cpu_results.items()}
    output["cupy_raw_vs_cpu24_raw"] = cupy_status_semantics
    output["torch_raw_vs_cpu24_raw"] = torch_status_semantics
    output["cpu_8_vs_24"] = {"same_inputs": True, "workers": [8, 24], "rows": 24}
    output["peak_host_memory_note"] = "No host RSS sampler was active; GPU device memory snapshots are recorded separately."
    (OUT/"raw_backend_benchmark.json").write_text(json.dumps(output, indent=2)+"\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
