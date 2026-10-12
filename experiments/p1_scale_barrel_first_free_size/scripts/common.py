"""Artifact and weighting helpers for the barrel-first free-size P4 family.

This module resolves Stage 01–05 outputs inside this sensitivity family. The run
script declares its two Stage 05 initializer artifacts explicitly; no archived
fit is used as a prior or as a source of calibration targets.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
STAGES = {
    1: "stage_01_gaze_initialization",
    2: "stage_02_shared_p1_scale",
    3: "stage_03_per_gaze_barrel",
    4: "stage_04_capturewise_keystone",
    5: "stage_05_joint_states",
}


def safe(value):
    """Convert NumPy/path values to strict JSON-safe builtins."""
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def hashfile(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def save_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _context(family):
    """Resolve the legacy validation_temporal argument to the training context."""
    path = Path(family).resolve()
    if path.name == "validation_temporal":
        return path.parent, "training"
    if path.name == "training" and path.parent.name == "inputs":
        return path.parent.parent, "training"
    return path, "production"


def stage_folder(stage, family=FAMILY):
    try:
        name = STAGES[int(stage)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unknown pipeline stage {stage!r}") from exc
    family_root, context = _context(family)
    if int(stage) <= 4:
        return family_root / "inputs" / context / name
    return family_root / "results" / context


def result_dir(stage, branch="primary", family=FAMILY):
    stage = int(stage)
    family_root, context = _context(family)
    if stage <= 4:
        return stage_folder(stage, family)
    if str(branch).startswith("fixed_size_"):
        return family_root / "inputs" / "fixed_controls" / context / str(branch)
    return stage_folder(stage, family) / str(branch)


def load_stage(stage, branch="primary", family=FAMILY):
    """Load a saved array bundle from this family only; never search elsewhere."""
    path = result_dir(stage, branch, family) / "arrays.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Required stage artifact is absent: {path}")
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key].copy() for key in arrays.files}


def verify_relocated_hashes(saved_hashes, resolved_hashes, label="input artifacts"):
    """Match frozen pre-move manifests to family-local files by exact SHA-256.

    The JSON manifests keep their original absolute path keys. After relocation,
    the explicit path-alias map resolves each old stage path to its current local
    bundle; both path identity and SHA-256 must match without rewriting JSON.
    """
    relocation_path = FAMILY / "provenance" / "relocation_manifest.json"
    aliases = read_json(relocation_path).get("path_aliases", {}) if relocation_path.is_file() else {}

    def relative(path):
        path = Path(path)
        if path.is_absolute():
            try:
                path = path.resolve().relative_to(ROOT)
            except ValueError as exc:
                raise ValueError(f"{label}: path is outside repository: {path}") from exc
        return path.as_posix()

    expected = {}
    for old_path, digest in saved_hashes.items():
        current_path = aliases.get(relative(old_path), relative(old_path))
        if current_path in expected:
            raise ValueError(f"{label}: duplicate resolved input path {current_path}")
        expected[current_path] = str(digest)
    actual = {relative(path): str(digest) for path, digest in resolved_hashes.items()}
    if expected != actual:
        raise ValueError(f"{label} path mapping or SHA-256 differs from frozen inputs")
    return resolved_hashes


def source_manifest():
    """Hash free-size scientific code and explicitly reused implementations."""
    paths = [
        ROOT / "CURRENT_PLAN.md",
        FAMILY / "scripts/common.py",
        FAMILY / "scripts/model.py",
        ROOT / "distortion_model/p1_joint.py",
        ROOT / "distortion_model/p1_fixed_scale.py",
        ROOT / "distortion_model/capturewise_reverse.py",
        ROOT / "distortion_model/capturewise_keystone_first.py",
        ROOT / "distortion_model/capturewise_barrel_anchors.py",
        ROOT / "distortion_model/capturewise_states.py",
        ROOT / "distortion_model/barrel_first_pipeline.py",
        FAMILY / "scripts/audit_free_size.py",
        FAMILY / "scripts/run.py",
        FAMILY / "scripts/validation.py",
        FAMILY / "scripts/profile.py",
    ]
    manifest = {}
    for path in paths:
        if path.is_file():
            try:
                name = str(path.resolve().relative_to(ROOT))
            except ValueError:
                name = str(path.resolve())
            manifest[name] = hashfile(path)
    return manifest


def save_stage(stage, arrays, summary, protocol=None, branch="primary", family=FAMILY):
    path = result_dir(stage, branch, family)
    path.mkdir(parents=True, exist_ok=True)
    if (path / "summary.json").exists():
        raise FileExistsError(f"Completed output exists: {path}; select a new family or branch")
    np.savez_compressed(path / "arrays.npz", **arrays)
    summary = dict(
        summary,
        stage=int(stage),
        branch=str(branch),
        source_hashes=source_manifest(),
        arrays_sha256=hashfile(path / "arrays.npz"),
    )
    write_json(path / "summary.json", summary)
    write_json(path / "protocol.json", protocol or {})
    snapshot = path / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name in summary["source_hashes"]:
        source = ROOT / name
        if source.suffix == ".py" and source.is_file():
            destination = snapshot / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return path


def row_weights(group):
    """Equal total weight per observed group, then equal row weight in group."""
    _, ids, counts = np.unique(np.asarray(group), return_inverse=True, return_counts=True)
    return 1.0 / (len(counts) * counts[ids])


def geometry_metrics(residual, group, valid=None):
    residual = np.asarray(residual)
    group = np.asarray(group)
    error = np.mean(np.sum(residual * residual, axis=-1), axis=-1)
    valid_rows = np.isfinite(error) if valid is None else np.asarray(valid, bool) & np.isfinite(error)
    group_values = [
        float(np.mean(error[(group == value) & valid_rows]))
        for value in np.unique(group)
        if np.any((group == value) & valid_rows)
    ]
    return {
        "weighted_vertex_rms_px": float(np.sqrt(np.mean(group_values))) if group_values else None,
        "pooled_vertex_rms_px": float(np.sqrt(np.mean(error[valid_rows]))) if valid_rows.any() else None,
        "scored_rows": int(valid_rows.sum()),
        "invalid_rows": int((~valid_rows).sum()),
        "complete_denominator": int(len(valid_rows)),
        "conditional_on_valid": bool((~valid_rows).any()),
    }


def backend(name="auto"):
    """Return a NumPy/CuPy array namespace and the selected backend label."""
    if name not in ("auto", "numpy", "cupy"):
        raise ValueError("backend must be auto, numpy, or cupy")
    if name == "numpy":
        return np, "numpy"
    try:
        import cupy as cp
        cp.cuda.runtime.getDevice()
        return cp, "cupy"
    except Exception:
        if name == "cupy":
            raise RuntimeError("CuPy backend requested but no usable CUDA device is available")
        return np, "numpy"
