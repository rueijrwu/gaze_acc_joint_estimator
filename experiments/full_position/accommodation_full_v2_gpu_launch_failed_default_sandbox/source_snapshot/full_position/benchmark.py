"""Benchmark BLAS thread counts on the actual calibration kernel, in fresh processes."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(model_path, threads):
    import numpy as np
    from .data import load_reviewed, training_data
    from .model import PositionModel, STATE_SCALE
    from .noise import reference_covariance
    from .calibrate import ProfiledProblem
    path = Path(model_path)
    obj = json.loads(path.read_text())
    root = Path(__file__).resolve().parents[1]
    captures, groups, _ = load_reviewed(root)
    data, anchors = training_data(captures, groups, obj["provenance"]["training_group_ids"], 48)
    pilot = PositionModel(27, obj["pilot_coefficients"])
    cov = reference_covariance(data["p"], pilot, np.asarray(obj["reference_state"]), np.array(obj["coordinate_covariance"]))
    x = np.array(json.loads((path.parent/"training_states.json").read_text())["states"])
    if len(x) != len(data["p"]):
        raise ValueError("Benchmark currently expects a 48-row-per-fixation experiment")
    z = (x/STATE_SCALE).ravel()
    rng = np.random.default_rng(17)
    v = rng.normal(size=z.shape)
    timings = {}
    for capacity in [27, 37]:
        problem = ProfiledProblem(PositionModel(capacity), data["v"], data["r"], cov, data["groups"], anchors)
        measured = []
        for repeat in range(3):
            begun = time.perf_counter()
            for i in range(40):
                point = z+v*1e-7*(i+1)
                problem.update(point)
                operator = problem.jac(point)
                operator.rmatvec(operator@v)
            measured.append(time.perf_counter()-begun)
        timings[f"conditional{capacity}"] = dict(seconds_median=float(np.median(measured)), repeats=measured)
    print(json.dumps(dict(threads=threads, calibration_rows=len(data["p"]), timings=timings)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.model, args.threads[0])
        return
    results = []
    for threads in args.threads:
        env = dict(os.environ, OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads))
        result = subprocess.run([sys.executable, "-m", "full_position.benchmark", str(args.model),
            "--threads", str(threads), "--worker"], env=env, text=True, capture_output=True, check=True)
        measured = json.loads(result.stdout)
        results.append(measured)
        print(json.dumps(measured), flush=True)
    if args.output:
        from .schema import write_json
        write_json(args.output, results)


if __name__ == "__main__":
    main()
