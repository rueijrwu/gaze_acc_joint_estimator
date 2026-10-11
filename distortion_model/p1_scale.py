"""Vectorized P1 scalar measurements with retained affine shape information.

No optical inverse or shape correction is applied to the input triangles.
Noise sensitivities assume independent unit pixel noise, a fixed reference,
and local perturbations; they are not measured uncertainty estimates.
"""
from __future__ import annotations

import numpy as np


def analyze(p1, reference):
    """Measure scales of labelled triangles against one native-pixel reference.

    Inputs are (N,3,2) and (3,2). Translation is removed for calculations only.
    ``affine`` maps centered reference to centered observations. ``det1_shape``
    retains rotation, anisotropy and shear; it is never applied to the data.
    """
    points = np.asarray(p1, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if points.ndim != 3 or points.shape[1:] != (3, 2) or ref.shape != (3, 2):
        raise ValueError('Expected triangles (N,3,2) and reference (3,2)')
    if not np.isfinite(ref).all():
        raise ValueError('Nonfinite reference')
    B = ref - ref.mean(axis=0)
    X = points - points.mean(axis=1, keepdims=True)
    gram = B.T @ B
    eig = np.linalg.eigvalsh(gram)
    if eig[0] <= 0 or eig[-1] / eig[0] > 1e12:
        raise ValueError('Degenerate or numerically ill-conditioned reference')
    energy = float(np.sum(B * B))
    cross = np.einsum('nvi,vj->nij', X, B)
    M = cross @ np.linalg.inv(gram)
    determinant = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]

    def area(T):
        u, v = T[..., 1, :] - T[..., 0, :], T[..., 2, :] - T[..., 0, :]
        return .5 * (u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0])

    ref_area = float(area(B))
    signed_area = area(X)
    ratio = signed_area / ref_area
    finite = np.isfinite(points).all(axis=(1, 2))
    valid = finite & np.isfinite(determinant) & (determinant > 0) & (ratio > 0)
    g_area = np.full(len(X), np.nan)
    g_area[valid] = np.sqrt(determinant[valid])
    D = np.full_like(M, np.nan)
    D[valid] = M[valid] / g_area[valid, None, None]
    dot = np.sum(X * B[None], axis=(1, 2))
    turn = np.sum(B[None, :, 0] * X[:, :, 1] - B[None, :, 1] * X[:, :, 0], axis=1)
    g_fixed = dot / energy
    g_rot = np.hypot(dot, turn) / energy
    angle = np.arctan2(turn, dot)
    R = np.stack((np.cos(angle), -np.sin(angle), np.sin(angle), np.cos(angle)), axis=1).reshape(-1, 2, 2)
    RB = np.einsum('nij,vj->nvi', R, B)
    g_rms = np.sqrt(np.sum(X * X, axis=(1, 2)) / energy)
    recovered = np.einsum('nij,vj->nvi', M, B)
    reconstruction = np.sqrt(np.mean(np.sum((recovered - X)**2, axis=2), axis=1))
    col0 = np.sum(M[:, :, 0]**2, axis=1)
    col1 = np.sum(M[:, :, 1]**2, axis=1)
    prod = np.sum(M[:, :, 0] * M[:, :, 1], axis=1)
    largest2 = .5 * (col0 + col1 + np.sqrt((col0-col1)**2 + 4*prod**2))
    anisotropy = np.full(len(X), np.nan)
    anisotropy[valid] = largest2[valid] / determinant[valid]
    polar_angle = np.arctan2(M[:, 1, 0] - M[:, 0, 1], M[:, 0, 0] + M[:, 1, 1])
    shape = np.einsum('nij,vj->nvi', D, B)

    def residual(model):
        return np.sqrt(np.mean(np.sum((X-model)**2, axis=2), axis=1))

    def scaled_residual(model, g):
        value = residual(model)
        return np.divide(value, g, out=np.full_like(value, np.nan), where=finite & (g > 0))

    # Derivatives of signed triangle area with respect to each vertex.
    grad_area = np.empty_like(X)
    for j in range(3):
        nxt, prev = (j+1) % 3, (j+2) % 3
        grad_area[:, j, 0] = .5 * (X[:, nxt, 1] - X[:, prev, 1])
        grad_area[:, j, 1] = .5 * (X[:, prev, 0] - X[:, nxt, 0])
    rel_noise = np.full(len(X), np.nan)
    rel_noise[valid] = np.sqrt(np.sum(grad_area[valid]**2, axis=(1, 2))) / (2*np.abs(signed_area[valid]))
    return dict(
        area_scale=g_area, similarity_fixed_scale=g_fixed,
        similarity_rotation_scale=g_rot, rms_scale=g_rms,
        affine=M, det1_shape=D, determinant=determinant,
        signed_area_px2=signed_area, area_ratio=ratio,
        anisotropy=anisotropy, rotation_deg=np.rad2deg(polar_angle),
        similarity_rotation_deg=np.rad2deg(angle), valid=valid,
        finite=finite, similarity_fixed_valid=finite & (g_fixed > 0),
        similarity_rotation_valid=finite & (g_rot > 0),
        rms_valid=finite & (g_rms > 0),
        reference_condition=float(eig[-1]/eig[0]),
        reference_coordinate_condition=float(np.sqrt(eig[-1]/eig[0])),
        reference_energy_px2=energy, reference_signed_area_px2=ref_area,
        affine_reconstruction_rms_px=reconstruction,
        area_det_identity_error=np.abs(determinant-ratio),
        det1_identity_error=np.abs(D[:, 0, 0]*D[:, 1, 1]-D[:, 0, 1]*D[:, 1, 0]-1),
        retained_shape_rms_px=np.sqrt(np.mean(np.sum((shape-B[None])**2, axis=2), axis=1)),
        normalized_similarity_residual_px=scaled_residual(g_fixed[:, None, None]*B[None], g_fixed),
        normalized_rot_similarity_residual_px=scaled_residual(g_rot[:, None, None]*RB, g_rot),
        area_relative_noise_per_px=rel_noise,
        area_absolute_noise_per_px=g_area*rel_noise,
        similarity_absolute_noise_per_px=1/np.sqrt(energy),
    )
