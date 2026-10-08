"""Optional FP64 products for the existing profiled calibration Jacobian.

The optimizer, residual, coefficient QR, starts, and certificates are unchanged.
Only LinearOperator products run on the selected array backend. The derivative
times optical residual is contracted once per Jacobian rather than per product.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import LinearOperator

from . import calibrate
from .model import STATE_SCALE

ORIGINAL_JAC = calibrate.ProfiledProblem.jac


def contracted_jac(problem, z, backend="numpy"):
    problem.update(z)
    if backend == "cupy":
        import cupy as xp
        to_host = xp.asnumpy
    elif backend == "numpy":
        xp = np
        to_host = np.asarray
    else:
        raise ValueError(f"Unknown linear backend: {backend}")
    n, c, p = problem.n, problem.c, problem.model.size
    residual = problem.optical_residual
    R, scale = problem.R, problem.column_scale
    # dB.T @ residual = D.T @ v; the same D is used by the adjoint.
    cache = getattr(problem, "_gpu_profile", None) if backend == "cupy" else None
    if cache is not None:
        B, S, D = cache["B"], cache["S"], cache["D"]
    else:
        D = np.einsum("ncpz,nc->nzp", problem.C,
                      residual[:n*c].reshape(n, c)).reshape(2*n, p)
        B = xp.asarray(problem.B, dtype=xp.float64)
        S = xp.asarray(problem.S, dtype=xp.float64)
        D = xp.asarray(D, dtype=xp.float64)
    optical_size, data_size = len(residual), n*c
    groups, counts = problem.groups, problem.counts
    anchor_scales, k = problem.anchor_scales, problem.k

    if backend == "cupy":
        from cupyx.scipy.linalg import solve_triangular as device_triangular
        from .device_lsmr import operator
        if cache is not None:
            B, S, D = cache["B"], cache["S"], cache["D"]
            device_R, device_scale = cache["R"], cache["scale"]
        else:
            device_R, device_scale = xp.asarray(R), xp.asarray(scale)
        device_groups, device_counts = xp.asarray(groups), xp.asarray(counts)
        physical, anchors = xp.asarray(STATE_SCALE), xp.asarray(anchor_scales)

        def device_hsolve(t):
            q = device_triangular(device_R.T, t/device_scale, lower=True)
            return device_triangular(device_R, q)/device_scale

        def device_jvp(vector):
            v = vector.reshape(n, 2)
            a = xp.einsum("ncz,nz->nc", S, v).ravel()
            db = -device_hsolve(D.T@vector+B[:data_size].T@a)
            optical = B@db
            optical[:data_size] += a
            means = xp.stack([xp.bincount(device_groups, weights=v[:,j], minlength=k)
                              for j in range(2)], axis=-1)/device_counts[:,None]
            anchor = means*physical/anchors/np.sqrt(k)
            return xp.concatenate((optical, anchor.ravel()))

        def device_vjp(vector):
            optical, anchor = vector[:optical_size], vector[optical_size:].reshape(k,2)
            u = device_hsolve(B.T@optical)
            projected = (optical-B@u)[:data_size].reshape(n,c)
            answer = xp.einsum("ncz,nc->nz",S,projected).ravel()-D@u
            return (answer.reshape(n,2)+anchor[device_groups]*physical/
                    (device_counts[device_groups,None]*anchors*np.sqrt(k))).ravel()

        return operator((len(problem.residual),2*n),device_jvp,device_vjp,xp)

    def hsolve(t):
        q = solve_triangular(R.T, to_host(t)/scale, lower=True)
        return solve_triangular(R, q)/scale

    def jvp(vector):
        host = np.asarray(vector).reshape(n, 2)
        v = xp.asarray(host, dtype=xp.float64)
        a = xp.einsum("ncz,nz->nc", S, v).ravel()
        db = -hsolve(D.T @ v.ravel() + B[:data_size].T @ a)
        optical = B @ xp.asarray(db, dtype=xp.float64)
        optical[:data_size] += a
        anchor = problem.mean_states(host*STATE_SCALE)/anchor_scales/np.sqrt(k)
        return np.r_[to_host(optical), anchor.ravel()]

    def vjp(vector):
        host = np.asarray(vector).ravel()
        optical = xp.asarray(host[:optical_size], dtype=xp.float64)
        anchor = host[optical_size:].reshape(k, 2)
        u = xp.asarray(hsolve(B.T @ optical), dtype=xp.float64)
        projected = (optical-B @ u)[:data_size].reshape(n, c)
        answer = xp.einsum("ncz,nc->nz", S, projected).ravel() - D @ u
        result = to_host(answer).reshape(n, 2)
        result += anchor[groups]*STATE_SCALE/(counts[groups, None]*anchor_scales*np.sqrt(k))
        return result.ravel()

    return LinearOperator((len(problem.residual), 2*n), matvec=jvp,
                          rmatvec=vjp, dtype=float)


def enable(backend="cupy", device=0):
    """Select a process-local product backend without replacing fit()."""
    disable()
    if backend == "cupy":
        import cupy as cp
        cp.cuda.Device(device).use()
        from . import device_lsmr, gpu_profile
        device_lsmr.enable()
        gpu_profile.enable()
    elif backend != "numpy":
        raise ValueError(f"Unknown linear backend: {backend}")
    calibrate.ProfiledProblem.jac = lambda self, z: contracted_jac(self, z, backend)


def disable():
    calibrate.ProfiledProblem.jac = ORIGINAL_JAC
    from . import device_lsmr, gpu_profile
    device_lsmr.disable()
    gpu_profile.disable()
