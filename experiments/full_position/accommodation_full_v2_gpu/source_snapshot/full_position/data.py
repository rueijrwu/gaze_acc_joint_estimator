"""Frozen exp5 payload loading; fixed row sampling and grouped data splits."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from .geometry import context, reorder, observations, summaries


@dataclass
class Capture:
    name: str
    p: np.ndarray
    q: np.ndarray
    found: np.ndarray
    frame: np.ndarray
    timestamp: np.ndarray
    ctx: object
    v: np.ndarray
    point_valid: np.ndarray
    baseline_valid: np.ndarray
    sha256: str
    metadata: dict


def load_capture(path, expected_hash=None):
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_hash and digest != expected_hash:
        raise ValueError(f"Frozen detection hash mismatch: {path}")
    # Only the repository's trusted local detection pickles are loaded.
    payload = pickle.loads(raw)
    a, meta = payload["arrays"], payload["meta"]
    p = np.asarray(a["p1_xy"], float)
    q, found = reorder(a["p4_xy"], a["p4_found"], meta.get("pair_index"))
    ctx = context(p)
    v, valid = observations(ctx, q, found)
    return Capture(path.name, p, q, found, np.asarray(a["frame_index"]),
                   np.asarray(a["timestamp_ms"]), ctx, v, valid, valid.all(-1), digest, meta)


def load_reviewed(root):
    root = Path(root)
    path = root/"data/fixations/fixation_intervals.json"
    report = json.loads(path.read_text())
    hashes = {s["capture"]: s["sha256"] for s in report["sources"]}
    captures = {name: load_capture(root/"data/detections"/name, digest)
                for name, digest in hashes.items()}
    groups = report["fixations"]
    if len(groups) != 20:
        raise ValueError("Expected 20 reviewed fixations")
    for name in captures:
        targets = sorted(g["target_theta_deg"] for g in groups if g["capture"] == name)
        if targets != [-10., -5., 0., 5., 10.]:
            raise ValueError("Legacy or invalid target labels; expected [-10,-5,0,5,10]")
    return captures, groups, hashlib.sha256(path.read_bytes()).hexdigest()


def core_rows(group):
    start, end = group["start_row"], group["end_row_exclusive"]
    cut = int(np.floor(.1*(end-start)))
    return np.arange(start+cut, end-cut)


def fixed_sample(rows, count):
    """Sample original rows BEFORE observing detection quality or model residuals."""
    if count <= 0 or count >= len(rows):
        return rows
    return rows[np.linspace(0, len(rows)-1, count).round().astype(int)]


def window_rows(group, window="core"):
    """Select reviewed interval rows; validity is applied by the caller."""
    if window == "core":
        return core_rows(group)
    if window == "fixation_period":
        return np.arange(int(group["start_row"]), int(group["end_row_exclusive"]))
    raise ValueError(f"Unknown fixation window: {window}")


def training_data(captures, groups, group_ids, count=48, window="core"):
    parts, anchors = [], []
    for local, gi in enumerate(group_ids):
        g, cap = groups[gi], captures[groups[gi]["capture"]]
        rows = fixed_sample(window_rows(g, window), count)
        rows = rows[cap.baseline_valid[rows]]
        if not len(rows):
            raise ValueError(f"Empty sampled training fixation {gi}")
        parts.append(dict(p=cap.p[rows], q=cap.q[rows], r=cap.ctx.r[rows],
                          v=cap.v[rows].reshape(-1, 6),
                          y2=summaries(context(cap.p[rows]), cap.q[rows])[:, [0, 2]],
                          groups=np.full(len(rows), local, int), original_group=np.full(len(rows), gi),
                          rows=rows))
        anchors.append([g["target_theta_deg"], g["demand_diopters_label"]])
    data = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    return data, np.asarray(anchors)


def noise_blocks(captures, groups, group_ids, window="core"):
    blocks = []
    for gi in group_ids:
        cap = captures[groups[gi]["capture"]]
        rows = window_rows(groups[gi], window)
        rows = rows[cap.baseline_valid[rows]]
        # Split on original frame gaps AND row gaps, never bridge missing points.
        split = np.flatnonzero((np.diff(rows) != 1) | (np.diff(cap.frame[rows]) != 1))+1
        for run in np.split(rows, split):
            if len(run) >= 3:
                blocks.append(np.concatenate((cap.p[run].reshape(-1, 6), cap.q[run].reshape(-1, 6)), -1))
    return blocks
