"""Trusted reviewed payloads, full intervals and candidate-independent identities."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import pickle
import numpy as np


@dataclass
class Capture:
    name: str
    digest: str
    meta: dict
    arrays: dict
    p1: np.ndarray
    p4: np.ndarray
    p1_available: np.ndarray
    p4_available: np.ndarray


def load_capture(path, expected_hash):
    """Only repository-trusted pickle bytes pinned by the reviewed source hash."""
    path = Path(path)
    content = path.read_bytes()
    digest = sha256(content).hexdigest()
    if digest != expected_hash:
        raise ValueError(f"source hash mismatch: {path.name}")
    payload = pickle.loads(content)
    meta, arrays = payload["meta"], payload["arrays"]
    permutation = np.asarray(meta["pair_index"], dtype=int)
    if permutation.shape != (3,) or sorted(permutation.tolist()) != [0, 1, 2]:
        raise ValueError(f"invalid source correspondence: {path.name}")
    if permutation.tolist() != [2, 1, 0]:
        raise ValueError("unexpected correspondence; review required before changing convention")
    if not meta["array_layout"]["coordinates"].startswith("px;"):
        raise ValueError("native camera pixel units not established")
    p1 = np.asarray(arrays["p1_xy"], dtype=np.float64)
    p4 = np.asarray(arrays["p4_xy"], dtype=np.float64)[:, permutation]
    n = len(arrays["frame_index"])
    if p1.shape != (n, 3, 2) or p4.shape != (n, 3, 2):
        raise ValueError("invalid native point shape")
    if len(np.unique(arrays["frame_index"])) != n:
        raise ValueError(f"duplicated source frame identity: {path.name}")
    p1_available = np.isfinite(p1).all(axis=-1)
    if "p1_valid" in arrays:
        if arrays["p1_valid"].shape != (n, 3):
            raise ValueError("invalid P1 flag shape")
        p1_available &= arrays["p1_valid"]
    p4_available = np.isfinite(p4).all(axis=-1)
    if "p4_found" in arrays:
        if arrays["p4_found"].shape != (n, 3):
            raise ValueError("invalid P4 flag shape")
        p4_available &= arrays["p4_found"][:, permutation]
    return Capture(path.name, digest, meta, arrays, p1, p4, p1_available, p4_available)


def load_reviewed(root, workers=4):
    root = Path(root)
    interval_path = root / "data/fixations/fixation_intervals.json"
    interval_bytes = interval_path.read_bytes()
    reviewed = json.loads(interval_bytes)
    sources = reviewed["sources"]
    expected_names = {f"capture_{i}_detections.pkl" for i in range(1, 5)}
    if {s["capture"] for s in sources} != expected_names or len(sources) != 4:
        raise ValueError("expected exactly captures 1–4")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        captures = list(executor.map(lambda s: load_capture(root / "data/detections" / s["capture"],
                                                          s["sha256"]), sources))
    captures = {capture.name: capture for capture in captures}
    intervals = reviewed["fixations"]
    if len(intervals) != 20:
        raise ValueError("expected exactly twenty exposures")
    for name, capture in captures.items():
        source = next(s for s in sources if s["capture"] == name)
        if source["row_count"] != len(capture.p1):
            raise ValueError(f"source row count mismatch: {name}")
        selected = [i for i in intervals if i["capture"] == name]
        if len(selected) != 5 or sorted(i["target_theta_deg"] for i in selected) != [-10, -5, 0, 5, 10]:
            raise ValueError(f"five visual targets missing or duplicated: {name}")
        if len({i["demand_diopters_label"] for i in selected}) != 1:
            raise ValueError("inconsistent capture demand")
        occupancy = np.zeros(len(capture.p1), dtype=bool)
        for interval in selected:
            start, end = interval["start_row"], interval["end_row_exclusive"]
            if not 0 <= start < end <= len(occupancy) or occupancy[start:end].any():
                raise ValueError("overlapping or out-of-range reviewed intervals")
            if interval["row_count"] != end - start:
                raise ValueError("reviewed interval length mismatch")
            frames = capture.arrays["frame_index"]
            if frames[start] != interval["first_frame"] or frames[end-1] != interval["last_frame_inclusive"]:
                raise ValueError("reviewed/source identity mismatch")
            occupancy[start:end] = True
    return captures, intervals, reviewed, sha256(interval_bytes).hexdigest()


def make_population(captures, intervals):
    parts = []
    for exposure, interval in enumerate(intervals):
        capture = captures[interval["capture"]]
        rows = np.arange(interval["start_row"], interval["end_row_exclusive"])
        n = len(rows)
        parts.append({"capture": np.full(n, interval["capture"], dtype="U32"),
                      "exposure": np.full(n, exposure, dtype=np.int16),
                      "row": rows, "source_frame": capture.arrays["frame_index"][rows],
                      "timestamp_ms": capture.arrays["timestamp_ms"][rows],
                      "target_theta_deg": np.full(n, interval["target_theta_deg"]),
                      "demand_diopters": np.full(n, interval["demand_diopters_label"]),
                      "p1": capture.p1[rows], "p4": capture.p4[rows],
                      "p1_available": capture.p1_available[rows],
                      "p4_available": capture.p4_available[rows],
                      "detector_status": capture.arrays["status"][rows]})
    result = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    result["complete_valid"] = result["p1_available"].all(axis=1) & result["p4_available"].all(axis=1)
    result["invalid_reason"] = np.select(
        [~result["p1_available"].all(axis=1), ~result["p4_available"].all(axis=1)],
        ["p1_unavailable", "p4_unavailable"], default="available")
    keys = list(zip(result["capture"], result["source_frame"]))
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate scheduled frame identity")
    return result


def make_slot_manifest(population, calibration_status="not_run_stage1"):
    """One explicit outcome for every frame and held P4 source, including failures."""
    n = len(population["row"])
    result = {key: np.repeat(population[key], 3, axis=0)
              for key in ("capture", "exposure", "row", "source_frame", "timestamp_ms")}
    result["held_point"] = np.tile(np.arange(3, dtype=np.int8), n)
    available = population["p4_available"]
    retained_valid = np.column_stack([population["p1_available"].all(axis=1)
                                     & available[:, [k for k in range(3) if k != j]].all(axis=1)
                                     for j in range(3)])
    result["retained_input_valid"] = retained_valid.ravel()
    result["held_input_valid"] = available.ravel()
    result["raw_input_valid"] = np.repeat(population["complete_valid"], 3)
    result["calibration_status"] = np.full(3*n, calibration_status)
    reason = "not_run_uncertified_model" if calibration_status == "uncertified" else "not_run_stage1"
    result["inference_status"] = np.full(3*n, reason)
    result["reason"] = np.where(result["retained_input_valid"], reason, "retained_input_unavailable")
    return result


def array_hash(arrays):
    digest = sha256()
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def validate_slots(population, slots):
    n = len(population["row"])
    required = ("held_point", "capture", "exposure", "row", "source_frame", "timestamp_ms",
                "raw_input_valid", "retained_input_valid", "held_input_valid",
                "calibration_status", "inference_status", "reason")
    if any(key not in slots or len(slots[key]) != 3*n for key in required):
        raise ValueError("missing field or incomplete slot population")
    if not np.array_equal(slots["held_point"], np.tile(np.arange(3), n)):
        raise ValueError("missing or duplicated held-point identity")
    for key in ("capture", "exposure", "row", "source_frame", "timestamp_ms"):
        if not np.array_equal(slots[key], np.repeat(population[key], 3), equal_nan=key == "timestamp_ms"):
            raise ValueError("slot/frame identity mismatch")
    return True
