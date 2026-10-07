"""Run independent grouped folds in isolated Python processes, then collect outputs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .schema import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--train-per-fixation", type=int, default=48)
    parser.add_argument("--eval-per-fixation", type=int, default=8)
    parser.add_argument("--max-nfev", type=int, default=300)
    args = parser.parse_args()
    if args.workers < 1 or args.blas_threads < 1:
        raise ValueError("Worker/thread counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = args.output/".workers"
    jobs.mkdir(exist_ok=False)
    specs = [(f"gaze_{g:+g}", "gaze") for g in [-10., -5., 0., 5., 10.]] + [(f"capture_{c}", "capture") for c in range(1, 5)]
    def run_job(spec):
        name, family = spec
        output = jobs/name
        log = jobs/f"{name}.log"
        cmd = [sys.executable, "-m", "full_position", "--output", str(output), "--folds", family,
            "--only-fold", name, "--train-per-fixation", str(args.train_per_fixation),
            "--eval-per-fixation", str(args.eval_per_fixation), "--max-nfev", str(args.max_nfev)]
        env = dict(os.environ, OPENBLAS_NUM_THREADS=str(args.blas_threads), OMP_NUM_THREADS=str(args.blas_threads))
        print(f"Starting fold {name}", flush=True)
        with log.open("w") as f:
            result = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Fold {name} failed; inspect {log}")
        summary = json.loads((output/"summary.json").read_text())
        fold_path = args.output/name
        if fold_path.exists():
            raise RuntimeError(f"Refusing to overwrite existing fold {name}")
        shutil.move(str(output/name), str(fold_path))
        write_json(fold_path/"execution.json", dict(
            config=json.loads((output/"config.json").read_text()),
            completion=json.loads((output/"completion.json").read_text())))
        shutil.rmtree(output)
        log.unlink()
        print(f"Finished fold {name}", flush=True)
        return name, summary
    completed, errors = {}, []
    begun = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, spec): spec[0] for spec in specs}
        for future in as_completed(futures):
            try:
                name, summary = future.result()
                completed.update(summary)
                write_json(args.output/"summary.json", completed)
            except Exception as exc:
                errors.append(dict(fold=futures[future], error=str(exc)))
                write_json(args.output/"parallel_failures.json", errors)
    write_json(args.output/"summary.json", completed)
    write_json(args.output/"parallel_config.json", dict(workers=args.workers, blas_threads=args.blas_threads,
        independent_processes=True, fold_execution_paths=[f"{name}/execution.json"
        for name, _ in specs if (args.output/name/"execution.json").exists()]))
    write_json(args.output/"completion.json", dict(complete=not errors, seconds=time.monotonic()-begun, errors=errors))
    if errors:
        raise SystemExit(1)
    jobs.rmdir()


if __name__ == "__main__":
    main()
