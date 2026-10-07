"""Run independent grouped folds in isolated Python processes, then collect outputs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
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
    parser.add_argument("--existing-fold", help="Wait for this already-running fold and reuse it")
    parser.add_argument("--stop-existing-pid", type=int,
                        help="PID of our sequential experiment; stop it AFTER the existing fold finishes")
    args = parser.parse_args()
    if args.workers < 1 or args.blas_threads < 1:
        raise ValueError("Worker/thread counts must be positive")
    args.output.mkdir(parents=True, exist_ok=bool(args.existing_fold))
    jobs = args.output/"worker_runs"
    jobs.mkdir(exist_ok=False)
    specs = [(f"gaze_{g:+g}", "gaze") for g in [-10., -5., 0., 5., 10.]] + [(f"capture_{c}", "capture") for c in range(1, 5)]
    existing_done = threading.Event()
    existing_summary = {}
    def wait_existing():
        if not args.existing_fold:
            existing_done.set(); return
        while True:
            path = args.output/"summary.json"
            if path.exists():
                summary = json.loads(path.read_text())
                keys = [f"{args.existing_fold}/{name}" for name in ["conditional27", "conditional37", "two_channel13"]]
                if all(key in summary for key in keys):
                    if args.stop_existing_pid:
                        try:
                            os.kill(args.stop_existing_pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    existing_summary.update({key: summary[key] for key in keys})
                    existing_done.set()
                    print(f"Collected existing fold {args.existing_fold}", flush=True)
                    return
            time.sleep(.2)
    watcher = threading.Thread(target=wait_existing, daemon=True)
    watcher.start()
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
        print(f"Finished fold {name}", flush=True)
        return name, json.loads((output/"summary.json").read_text())
    completed, errors = {}, []
    begun = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_job, spec): spec[0] for spec in specs if spec[0] != args.existing_fold}
        for future in as_completed(futures):
            try:
                name, summary = future.result()
                # Do not change the active sequential process's summary file.
                completed.update(summary)
                write_json(args.output/"parallel_summary.json", completed)
            except Exception as exc:
                errors.append(dict(fold=futures[future], error=str(exc)))
                write_json(args.output/"parallel_failures.json", errors)
    existing_done.wait()
    for name, _ in specs:
        if name == args.existing_fold or not (jobs/name/name).exists():
            continue
        if (args.output/name).exists():
            raise RuntimeError(f"Refusing to overwrite existing fold {name}")
        shutil.move(str(jobs/name/name), str(args.output/name))
    completed.update(existing_summary)
    write_json(args.output/"summary.json", completed)
    write_json(args.output/"parallel_config.json", dict(workers=args.workers, blas_threads=args.blas_threads,
        independent_processes=True, reused_fold=args.existing_fold, worker_config_paths=[str(jobs/name/"config.json")
        for name, _ in specs if name != args.existing_fold]))
    write_json(args.output/"completion.json", dict(complete=not errors, seconds=time.monotonic()-begun, errors=errors))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
