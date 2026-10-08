import gc
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

from full_position.accommodation import PowerResponseModel
from full_position.calibrate import ProfiledProblem
from full_position import calibrate, device_lsmr, gpu_profile, linear_acceleration
from full_position.model import STATE_SCALE


ROOT = Path.cwd()
NPZ = ROOT / "experiments/full_position/accommodation_full_v2_gpu/splits/full_calibration/training_inputs.npz"


def rss_mib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


with np.load(NPZ) as saved:
    y, r = saved["v"], saved["r"]
    cov, groups, anchors = saved["covariance"], saved["groups"], saved["anchors"]
    n = len(y)
    z = (anchors[groups] / STATE_SCALE).ravel()

model = PowerResponseModel(1.)
report = {"rows": n, "coefficient_count": model.size, "npz": str(NPZ),
          "threads_init": {}, "threads_tiny_products": {}, "full_cpu_gpu": {}}

# Use the saved run's real per-row covariance and all 89,175 reviewed training rows.
for limit in (1, 2, 4, 8):
    with threadpool_limits(limits=limit):
        before = rss_mib()
        problem, elapsed = timed(lambda: ProfiledProblem(model, y, r, cov, groups, anchors))
        report["threads_init"][str(limit)] = {"seconds": elapsed,
            "rss_before_mib": before, "rss_after_mib": rss_mib(),
            "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "threadpools": threadpool_info()}
        del problem
        gc.collect()

# Balanced 50 real samples per each of 20 groups, preserving that row's covariance.
indices = np.concatenate([np.flatnonzero(groups == group)[:50] for group in range(len(anchors))])
small_y, small_r, small_cov, small_g = y[indices], r[indices], cov[indices], groups[indices]
small_z = (anchors[small_g] / STATE_SCALE).ravel()
for limit in (1, 2, 4, 8):
    with threadpool_limits(limits=limit):
        small = ProfiledProblem(model, small_y, small_r, small_cov, small_g, anchors)
        small.update(small_z)
        op = small.jac(small_z)
        rng = np.random.default_rng(989)
        v = rng.normal(size=op.shape[1])
        w = rng.normal(size=op.shape[0])
        # Warm once, then report median of five products separately.
        op.matvec(v); op.rmatvec(w)
        jvp = []; vjp = []
        for _ in range(5):
            _, elapsed = timed(lambda: op.matvec(v)); jvp.append(elapsed)
            _, elapsed = timed(lambda: op.rmatvec(w)); vjp.append(elapsed)
        report["threads_tiny_products"][str(limit)] = {
            "rows": len(indices), "jvp_median_seconds": float(np.median(jvp)),
            "vjp_median_seconds": float(np.median(vjp))}
        del small, op
        gc.collect()

# Full actual-data update and captured JVP/VJP, using four CPU BLAS threads.
with threadpool_limits(limits=4):
    before = rss_mib()
    cpu, cpu_init_time = timed(lambda: ProfiledProblem(model, y, r, cov, groups, anchors))
    _, cpu_update_time = timed(lambda: calibrate.ProfiledProblem.update(cpu, z))
    cpu_op = linear_acceleration.ORIGINAL_JAC(cpu, z)
    rng = np.random.default_rng(1201)
    vec = rng.normal(size=2*n)
    cot = rng.normal(size=cpu_op.shape[0])
    cpu_op.matvec(vec); cpu_op.rmatvec(cot)
    _, cpu_jvp_time = timed(lambda: cpu_op.matvec(vec))
    _, cpu_vjp_time = timed(lambda: cpu_op.rmatvec(cot))
    residual = cpu.residual.copy()
    cpu_report = {"init_seconds": cpu_init_time, "update_seconds": cpu_update_time,
                  "jvp_seconds": cpu_jvp_time, "vjp_seconds": cpu_vjp_time,
                  "rss_before_mib": before, "rss_after_mib": rss_mib()}

    import cupy as cp
    cp.cuda.Device(0).use()
    # GPU path uses device QR/update and keeps the product vectors resident.
    gpu = ProfiledProblem(model, y, r, cov, groups, anchors)
    _, gpu_update_time = timed(lambda: gpu_profile.update(gpu, z))
    cp.cuda.Stream.null.synchronize()
    gpu_op = linear_acceleration.contracted_jac(gpu, z, "cupy")
    gv = cp.asarray(vec); gw = cp.asarray(cot)
    gpu_op.device_matvec(gv); gpu_op.device_rmatvec(gw)
    cp.cuda.Stream.null.synchronize()
    _, gpu_jvp_time = timed(lambda: gpu_op.device_matvec(gv)); cp.cuda.Stream.null.synchronize()
    _, gpu_vjp_time = timed(lambda: gpu_op.device_rmatvec(gw)); cp.cuda.Stream.null.synchronize()
    gpu_report = {"update_seconds": gpu_update_time, "jvp_seconds": gpu_jvp_time,
                  "vjp_seconds": gpu_vjp_time, "rss_after_mib": rss_mib(),
                  "device_pool_used_mib": cp.get_default_memory_pool().used_bytes()/2**20,
                  "device_pool_total_mib": cp.get_default_memory_pool().total_bytes()/2**20}

    # Same captured operators, same RHS/scale/regularization, fixed 200 LSMR budget.
    from scipy.optimize._lsq.common import regularized_lsq_operator, right_multiplied_operator
    import scipy.optimize._lsq.trf as trf
    scale = np.ones(2*n)
    diag = np.full(2*n, 1e-3)
    cpu_aug = regularized_lsq_operator(right_multiplied_operator(cpu_op, scale), diag)
    gpu_aug = device_lsmr.regularized(device_lsmr.right(gpu_op, cp.asarray(scale)), cp.asarray(diag))
    rhs = np.r_[residual, np.zeros(2*n)]
    _, cpu_lsmr_time = timed(lambda: trf.lsmr(cpu_aug, rhs, damp=0., atol=0., btol=0.,
                                               conlim=0., maxiter=200))
    cp.cuda.Stream.null.synchronize()
    _, gpu_lsmr_time = timed(lambda: device_lsmr.lsmr(gpu_aug, rhs, damp=0., atol=0., btol=0.,
                                                        conlim=0., maxiter=200))
    cp.cuda.Stream.null.synchronize()
    cpu_report["lsmr_200_seconds"] = cpu_lsmr_time
    gpu_report["lsmr_200_seconds"] = gpu_lsmr_time
    cpu_report["lsmr_budget"] = 200
    gpu_report["lsmr_budget"] = 200
    gpu_report["lsmr_200_resident_vectors"] = True
    report["full_cpu_gpu"] = {"cpu": cpu_report, "gpu": gpu_report}

for name in ("full_position/linear_acceleration.py", "full_position/gpu_profile.py", "full_position/device_lsmr.py"):
    report.setdefault("source_sha256", {})[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()

Path("/tmp/profile_gpu_benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps(report, indent=2, allow_nan=False))
