"""Full calibration with GPU profiling, device LSMR and batched inverses."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from . import accommodation_full as base
from . import accommodation_study as study
from .audit83 import reviewed
from .information import predict_raw_holdout as scalar_predict
from . import linear_acceleration

ORIGINAL_TASK = study._fit_task
_INVERSE_CONTEXT = None


def inverse_init(model, pilot, reference, sigma):
    global _INVERSE_CONTEXT
    from threadpoolctl import threadpool_limits
    for name in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
        os.environ[name]="1"
    # Spawn imports may already have initialized BLAS; also set runtime limits.
    threadpool_limits(limits=1)
    _INVERSE_CONTEXT = (model, pilot, reference, sigma)


def inverse_batch(tasks):
    model, pilot, reference, sigma = _INVERSE_CONTEXT
    return [scalar_predict(model, p, q, valid, held, pilot, reference, sigma, "xy")
            for p, q, valid, held in tasks]


class ParallelPredictions:
    """Bounded, ordered prefetch for the original worker's sequential loop."""
    def __init__(self, root, output, fold, workers, batch_size=16):
        self.root, self.output, self.fold = root, output, fold
        self.workers, self.batch_size = workers, batch_size
        self.pool = None
        self.pending = deque()
        self.ready = deque()
        self.expected = deque()
        self.tasks = None
        self.exhausted = False

    def task_stream(self):
        captures, _, _ = reviewed(str(self.root))
        path = Path(self.output)/"splits"/self.fold/"population.json"
        for row in json.loads(path.read_text()):
            if not row["selected_for_evaluation"]:
                continue
            cap, i = captures[row["capture"]], int(row["row"])
            for held in range(3):
                yield (cap.p[i], cap.q[i], cap.point_valid[i], held)

    def fill(self):
        while not self.exhausted and len(self.pending) < 2*self.workers:
            batch = []
            for _ in range(self.batch_size):
                try:
                    batch.append(next(self.tasks))
                except StopIteration:
                    self.exhausted = True
                    break
            if batch:
                self.pending.append((batch, self.pool.submit(inverse_batch, batch)))

    def __call__(self, model, p, q, valid, held, pilot, reference, sigma, mask="xy"):
        if mask != "xy":
            raise ValueError("Parallel full-calibration checks require xy")
        if self.pool is None:
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=mp.get_context("spawn"),
                initializer=inverse_init, initargs=(model, pilot, reference, sigma))
            self.tasks = iter(self.task_stream())
            self.fill()
        if not self.ready:
            if not self.pending:
                raise RuntimeError("Inverse worker requested a slot outside the frozen schedule")
            batch, future = self.pending.popleft()
            self.ready.extend(future.result())
            self.expected.extend(batch)
            self.fill()
        expected_p, expected_q, expected_valid, expected_held = self.expected.popleft()
        # Check only the inputs allowed to influence the inverse. The held
        # coordinate is sealed by scalar_predict and is scored by _fit_task.
        retained = np.arange(3) != held
        if (expected_held != held
                or not np.array_equal(p, expected_p, equal_nan=True)
                or not np.array_equal(q[retained], expected_q[retained], equal_nan=True)
                or not np.array_equal(valid[retained], expected_valid[retained])):
            raise RuntimeError("Inverse task order differs from the frozen population")
        return self.ready.popleft()

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)


def accelerated_task(job):
    """Set explicit per-law OpenMP/BLAS limits without nested inverse pools."""
    from threadpoolctl import threadpool_limits
    threads=job[-1].get("cpu_threads",1)
    names=("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")
    previous={name:os.environ.get(name) for name in names}
    for name in names: os.environ[name]=str(threads)
    try:
        with threadpool_limits(limits=threads):
            return _accelerated_task(job)
    finally:
        for name,value in previous.items():
            if value is None: os.environ.pop(name,None)
            else: os.environ[name]=value


def _accelerated_task(job):
    root, output, fold, name, settings = job
    backend = settings["linear_backend"]
    if backend == "cupy" or settings.get("inverse_backend","scalar")!="scalar":
        import cupy as cp
        count=cp.cuda.runtime.getDeviceCount()
        if not count: raise RuntimeError("GPU execution requested but no CUDA devices are available")
        device = base.NAMES.index(name) % count
    else:
        device = 0
    original_predict = study.predict_raw_holdout
    predictions=None
    try:
        linear_acceleration.enable(backend, device)
        inverse_backend=settings.get("inverse_backend","scalar")
        if inverse_backend=="scalar":
            predictions=ParallelPredictions(root,output,fold,settings["inverse_workers"])
        else:
            from .batched_holdout import BatchedPredictions
            predictions=BatchedPredictions(root,output,fold,settings["inverse_batch_size"],inverse_backend,device,settings["inverse_devices"])
        study.predict_raw_holdout = predictions
        outcome = ORIGINAL_TASK(job)
        return dict(outcome, linear_backend=backend,
                    gpu_device=device if backend == "cupy" else None,
                    inverse_workers=settings["inverse_workers"] if inverse_backend=="scalar" else 0,
                    inverse_backend=inverse_backend)
    finally:
        study.predict_raw_holdout = original_predict
        try:
            if predictions is not None: predictions.close()
        finally:
            linear_acceleration.disable()


def run(root, output, *, workers=4, inverse_workers=8, linear_backend="cupy",
        parity_report=None, max_nfev=300, seed=17, source_commit=None,
        training_window="fixation_period", agreement_window="fixation_period",
        inverse_backend="cupy",inverse_batch_size=8192,inverse_parity_report=None,
        cpu_threads=4,inverse_devices=None):
    if workers<1 or cpu_threads<1 or workers*cpu_threads>(mp.cpu_count() or 1):
        raise ValueError("CPU thread budget must fit the available CPUs")
    if inverse_backend not in ("scalar","cupy","torch") or inverse_batch_size<1:
        raise ValueError("Choose scalar/cupy/torch inverses and a positive batch size")
    if inverse_backend=="scalar" and (inverse_workers < 1 or inverse_workers*workers > (mp.cpu_count() or 1)):
        raise ValueError("Inverse workers across laws must fit the available CPU count")
    if linear_backend not in ("numpy", "cupy"):
        raise ValueError("Unknown linear backend")
    if parity_report is None:
        raise ValueError("A passing backend parity report is required")
    parity_report = Path(parity_report)
    parity = json.loads(parity_report.read_text())
    source_hash = hashlib.sha256(Path(linear_acceleration.__file__).read_bytes()).hexdigest()
    if (parity.get("passed") is not True
            or parity.get("linear_acceleration_sha256") != source_hash
            or linear_backend not in parity.get("verified_backends", [])):
        raise ValueError("Parity report does not verify this backend source")
    if linear_backend=="cupy":
        from . import gpu_profile,device_lsmr
        fitting_hashes={Path(module.__file__).name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                        for module in (gpu_profile,device_lsmr)}
        if parity.get("gpu_fitting_sha256")!=fitting_hashes:
            raise ValueError("Parity report does not verify GPU profiling and inner LSMR")
    inverse_parity=None
    if inverse_backend!="scalar":
        import cupy as cp
        inverse_devices=None if inverse_devices is None else list(inverse_devices)
        if inverse_devices is not None and (not inverse_devices or len(set(inverse_devices))!=len(inverse_devices)
                or any(d<0 or d>=cp.cuda.runtime.getDeviceCount() for d in inverse_devices)):
            raise ValueError("Select at least one distinct available inverse GPU")
        from . import batched_inverse,batched_holdout,torch_arrays
        hashes={Path(module.__file__).name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for module in (batched_inverse,batched_holdout,torch_arrays)}
        if inverse_parity_report is None:
            raise ValueError("A passing batched inverse parity report is required")
        inverse_parity=json.loads(Path(inverse_parity_report).read_text())
        if (inverse_parity.get("passed") is not True or inverse_parity.get("source_sha256")!=hashes
                or inverse_backend not in inverse_parity.get("verified_backends",[])):
            raise ValueError("Inverse parity report does not verify this solver source/backend")
    execution = dict(linear_backend=linear_backend, numeric_dtype="float64",
                     inverse_backend="unchanged_scalar_cpu_49_start" if inverse_backend=="scalar" else f"batched_{inverse_backend}_trf_49_start",
                     inverse_workers_per_law=inverse_workers if inverse_backend=="scalar" else 0,
                     inverse_cpu_threads_per_worker=1 if inverse_backend=="scalar" else 0,
                     inverse_batch_size=16 if inverse_backend=="scalar" else inverse_batch_size,
                     inverse_prefetch_batches_per_worker=2 if inverse_backend=="scalar" else 0,
                     cpu_threads_per_law=cpu_threads,gpu_profiled_qr=linear_backend=="cupy",
                     gpu_resident_lsmr=linear_backend=="cupy",
                     inverse_devices=inverse_devices if inverse_backend!="scalar" else [],
                     inverse_device_scheduling="one assigned GPU per law" if inverse_devices is None else "concurrent frame chunks weighted by GPU multiprocessor count",
                     parity_report_sha256=hashlib.sha256(parity_report.read_bytes()).hexdigest(),
                     fitting_worker_unchanged=True,outer_optimizer="scipy_trf",
                     inner_optimizer="gpu_lsmr" if linear_backend=="cupy" else "scipy_lsmr")
    original_task, original_write = base._fit_task, base.write_json

    def configured_task(job):
        r, o, f, n, settings = job
        return accelerated_task((r, o, f, n, dict(settings, linear_backend=linear_backend,
                                                inverse_workers=inverse_workers,inverse_backend=inverse_backend,
                                                inverse_batch_size=inverse_batch_size,cpu_threads=cpu_threads,
                                                inverse_devices=inverse_devices)))

    # Executor.map must receive a module-level function under forkserver/spawn.
    class ConfiguredExecutor(ProcessPoolExecutor):
        def map(self, function, jobs, *args, **kwargs):
            prepared = [(r, o, f, n, dict(s, linear_backend=linear_backend,
                                        inverse_workers=inverse_workers,inverse_backend=inverse_backend,
                                        inverse_batch_size=inverse_batch_size,cpu_threads=cpu_threads,
                                        inverse_devices=inverse_devices))
                        for r, o, f, n, s in jobs]
            return super().map(accelerated_task, prepared, *args, **kwargs)

    original_executor = base.ProcessPoolExecutor

    def write_with_execution(path, value):
        path = Path(path)
        if path.name == "config.json":
            value = dict(value, execution=execution, parity_report=parity,inverse_parity_report=inverse_parity)
            value["runtime"] = dict(value["runtime"], fitting_linear_backend=linear_backend,
                                    worker_cpu_threads=cpu_threads,
                                    inverse_backend=execution["inverse_backend"],
                                    inverse_workers_per_law=execution["inverse_workers_per_law"],
                                    gpu_linear_operator_parity_verified=linear_backend == "cupy",
                                    gpu_power_response_inverse_parity_verified=inverse_backend!="scalar")
            value["runtime"].pop("gpu_power_response_parity_verified", None)
        elif path.name == "training.json":
            value = dict(value, provenance=dict(value["provenance"], execution=execution))
        return original_write(path, value)

    base._fit_task = configured_task
    base.ProcessPoolExecutor = ConfiguredExecutor
    base.write_json = write_with_execution
    try:
        return base.run(root, output, workers, max_nfev, seed, 0, source_commit,
                        agreement_window, training_window)
    finally:
        base._fit_task, base.ProcessPoolExecutor, base.write_json = original_task, original_executor, original_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--inverse-workers", type=int, default=8)
    parser.add_argument("--linear-backend", choices=("numpy", "cupy"), default="cupy")
    parser.add_argument("--inverse-backend",choices=("scalar","cupy","torch"),default="cupy")
    parser.add_argument("--inverse-batch-size",type=int,default=8192)
    parser.add_argument("--inverse-parity-report",type=Path)
    parser.add_argument("--cpu-threads",type=int,default=4)
    parser.add_argument("--inverse-devices",type=int,nargs="+",help="Optional CUDA IDs per batch; default: one assigned GPU per law")
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--max-nfev", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--source-commit")
    parser.add_argument("--training-window", choices=("core", "fixation_period"), default="fixation_period")
    parser.add_argument("--agreement-window", choices=("core", "fixation_period"), default="fixation_period")
    args = vars(parser.parse_args())
    run(args.pop("root"), args.pop("output"), **args)


if __name__ == "__main__":
    main()
