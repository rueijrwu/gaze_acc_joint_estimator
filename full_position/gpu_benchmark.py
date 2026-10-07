"""Measure batch-size crossover for CuPy using representative real inputs."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .accelerated import solve_batch
from .data import load_reviewed, core_rows, fixed_sample
from .geometry import context
from .model import PositionModel
from .noise import reference_covariance
from .schema import load_model, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    import cupy as cp
    cp.cuda.Device(args.device).use()
    model, meta = load_model(args.model)
    captures, groups, _ = load_reviewed(Path(__file__).resolve().parents[1])
    inputs = [(captures[groups[gi]["capture"]], row) for gi in meta["provenance"]["evaluation_group_ids"]
        for row in fixed_sample(core_rows(groups[gi]), 16) if captures[groups[gi]["capture"]].baseline_valid[row]]
    p = np.array([c.p[row] for c, row in inputs]); q = np.array([c.q[row] for c, row in inputs])
    ctx = context(p)
    y = ((q-ctx.c[:, None, :])/ctx.ell[:, None, None]).reshape(-1, 6)
    pilot = PositionModel(27, meta["pilot_coefficients"])
    cov = reference_covariance(p, pilot, np.array(meta["reference_state"]), np.array(meta["coordinate_covariance"]))
    results = []
    for count in args.sizes:
        ix = np.arange(count)%len(p)
        for backend in ["numpy", "cupy"]:
            solve_batch(model, ctx.r[ix[:2]], y[ix[:2]], cov[ix[:2]], backend=backend, device=args.device, maxiter=2)
            if backend == "cupy":
                cp.cuda.Stream.null.synchronize()
            begun = time.perf_counter()
            solved = solve_batch(model, ctx.r[ix], y[ix], cov[ix], backend=backend, device=args.device)
            if backend == "cupy":
                cp.cuda.Stream.null.synchronize()
            seconds = time.perf_counter()-begun
            record = dict(batch_rows=count, backend=backend, seconds=seconds,
                rows_per_second=count/seconds, converged_starts=int(solved["converged"].sum()),
                iterations=solved["iterations"], gpu_device=args.device,
                input_policy="repeated_representative_heldout_rows_for_timing_not_new_statistical_evidence")
            results.append(record); print(json.dumps(record), flush=True)
            write_json(args.output, results)


if __name__ == "__main__":
    main()
