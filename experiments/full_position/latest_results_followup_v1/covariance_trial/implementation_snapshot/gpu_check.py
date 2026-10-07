"""Check optional CuPy acceleration against NumPy and the scalar reference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import numpy as np
from .accelerated import forward, solve_batch, assemble
from .data import load_reviewed, fixed_sample, core_rows
from .geometry import context
from .model import PositionModel
from .noise import reference_covariance, marginal
from .schema import load_model, write_json
from .invert import invert


def run(model_path, output, device=0, sample=12):
    import cupy as cp
    cp.cuda.Device(device).use()
    model, meta = load_model(model_path)
    root = Path(__file__).resolve().parents[1]
    captures, groups, _ = load_reviewed(root)
    test = meta["provenance"]["evaluation_group_ids"]
    if not test:
        raise ValueError("Use a held-out fold artifact")
    points = [(gi, row) for gi in test for row in fixed_sample(core_rows(groups[gi]), sample)
              if captures[groups[gi]["capture"]].baseline_valid[row]]
    if len(points) > sample:
        points = [points[i] for i in np.linspace(0, len(points)-1, sample).round().astype(int)]
    p = np.array([captures[groups[gi]["capture"]].p[row] for gi, row in points])
    q = np.array([captures[groups[gi]["capture"]].q[row] for gi, row in points])
    ctx = context(p); v = ((q-ctx.c[:, None, :])/ctx.ell[:, None, None]).reshape(-1, 6)
    pilot = PositionModel(27, meta["pilot_coefficients"])
    cov = reference_covariance(p, pilot, np.array(meta["reference_state"]), np.array(meta["coordinate_covariance"]))
    rng = np.random.default_rng(17)
    x = np.column_stack((rng.uniform(-20, 20, len(p)), rng.uniform(0, 6, len(p))))
    expected, J = model.predict(x, ctx.r, True)
    actual, K = forward(cp.asarray(model.beta), model.capacity, cp.asarray(x), cp.asarray(ctx.r), cp)
    forward_error = float(np.max(abs(cp.asnumpy(actual)-expected)))
    derivative_error = float(np.max(abs(cp.asnumpy(K)-J)))
    comparison, timings = [], []
    for held in [None, 0, 1, 2]:
        indices = np.array([i for i in range(6) if held is None or i//2 != held])
        y, R = v[:, indices], marginal(cov, indices)
        sets = {}
        for backend in ["numpy", "cupy"]:
            # Warm-up includes CUDA kernel/library initialization.
            solve_batch(model, ctx.r[:1], y[:1], R[:1], indices, backend, device, maxiter=2)
            if backend == "cupy":
                cp.cuda.Stream.null.synchronize()
            begun = time.perf_counter()
            solutions = solve_batch(model, ctx.r, y, R, indices, backend, device)
            if backend == "cupy":
                cp.cuda.Stream.null.synchronize()
            elapsed = time.perf_counter()-begun
            sets[backend] = assemble(model, ctx.r, R, indices, solutions)
            timings.append(dict(held_point=held, backend=backend, seconds=elapsed,
                rows=len(p), converged_starts=int(solutions["converged"].sum()), iterations=solutions["iterations"]))
        begun = time.perf_counter()
        for i in range(len(p)):
            scalar = invert(model, ctx.r[i], y[i], R[i], indices)
            records = dict(held_point=held, fixation=points[i][0], row=int(points[i][1]),
                           scalar=scalar, numpy=sets["numpy"][i], cupy=sets["cupy"][i])
            if scalar["available"] and sets["cupy"][i]["available"]:
                records["gpu_state_difference"] = (np.array(sets["cupy"][i]["state"])-scalar["state"]).tolist()
                records["gpu_cost_difference"] = sets["cupy"][i]["cost"]-scalar["cost"]
                records["plausible_branch_count_match"] = len(scalar["plausible_branches"]) == len(sets["cupy"][i]["plausible_branches"])
            comparison.append(records)
        timings.append(dict(held_point=held, backend="scalar_scipy", rows=len(p), seconds=time.perf_counter()-begun))
    tested = [r for r in comparison if "gpu_state_difference" in r]
    agreement = bool(len(tested) == len(comparison) and all(
        np.all(np.abs(r["gpu_state_difference"]) < [.01, .01]) and
        abs(r["gpu_cost_difference"]) < 1e-3*(1+r["scalar"]["cost"]) and
        r["plausible_branch_count_match"] for r in tested))
    summary = dict(device=device, gpu_name=cp.cuda.runtime.getDeviceProperties(device)["name"].decode(),
        cupy_version=cp.__version__, model=str(model_path), physical_forward_max_error=forward_error,
        physical_derivative_max_error=derivative_error, reference_cases=len(comparison), comparable_cases=len(tested),
        scalar_agreement=agreement, timings=timings,
        policy="Acceleration is optional; scalar reference retained, especially for nonagreement cases")
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write_json(output/"summary.json", summary)
    write_json(output/"comparisons.json", comparison)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sample", type=int, default=12)
    args = parser.parse_args()
    run(args.model, args.output, args.device, args.sample)
