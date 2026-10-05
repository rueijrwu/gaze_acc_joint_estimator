"""Summarize missing pairs and render representative diagnostic frames."""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np


def runs(mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    cuts = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(indices) - 1]
    return [(int(indices[start]), int(indices[end])) for start, end in zip(starts, ends)]


def candidate_slice(arrays, prefix, frame_index):
    offsets = arrays[f"{prefix}_candidate_offsets"]
    start, end = offsets[frame_index:frame_index + 2]
    return (arrays[f"{prefix}_candidate_xy"][start:end],
            arrays[f"{prefix}_candidate_valid"][start:end])


def annotate(frame, arrays, frame_index, cause, run):
    pupil_valid = bool(arrays["pupil_valid"][frame_index])
    if pupil_valid:
        center = tuple(float(value) for value in arrays["pupil_center"][frame_index])
        axes = arrays["pupil_axes"][frame_index]
        angle = float(arrays["pupil_angle_deg"][frame_index])
        cv2.ellipse(frame, (center, (2.0 * float(axes[0]), 2.0 * float(axes[1])), angle),
                    (0, 255, 0), 3, cv2.LINE_AA)

    for prefix, color, radius in (("p1", (255, 255, 0), 18),
                                  ("p4", (255, 0, 255), 11)):
        points, valid = candidate_slice(arrays, prefix, frame_index)
        for candidate_index, (point, is_valid) in enumerate(zip(points, valid)):
            if not is_valid:
                continue
            position = tuple(int(round(value)) for value in point)
            cv2.circle(frame, position, radius, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"{prefix.upper()}{candidate_index}",
                        (position[0] + radius, position[1] - radius),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    if arrays["pair_valid"][frame_index]:
        p4_points, _ = candidate_slice(arrays, "p4", frame_index)
        for candidate_index in arrays["p4_selected_indices"][frame_index]:
            position = tuple(int(round(value)) for value in p4_points[candidate_index])
            cv2.circle(frame, position, 20, (0, 255, 255), 3, cv2.LINE_AA)

    start, end = run
    lines = [
        f"frame {frame_index}  run {start}-{end} ({end - start + 1} frames)",
        cause,
        (f"pupil={pupil_valid}  P1 valid={arrays['p1_valid_count'][frame_index]}  "
         f"P4 valid={arrays['p4_valid_count'][frame_index]}  "
         f"P4 inside={arrays['p4_inside_count'][frame_index]}"),
    ]
    for row, text in enumerate(lines):
        cv2.putText(frame, text, (25, 40 + 32 * row), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def contact_sheet(frames, output_path, columns=2, tile_width=900):
    tiles = []
    for frame in frames:
        scale = float(tile_width) / frame.shape[1]
        tiles.append(cv2.resize(frame, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA))
    rows = []
    blank = np.zeros_like(tiles[0])
    for start in range(0, len(tiles), columns):
        row = tiles[start:start + columns]
        while len(row) < columns:
            row.append(blank)
        rows.append(np.hstack(row))
    cv2.imwrite(str(output_path), np.vstack(rows))


def classify_pair_rejections(arrays, seed_scale, angle_gate, scale_gate):
    reasons = {}
    for frame_index in range(len(arrays["frame_index"])):
        if arrays["pair_valid"][frame_index]:
            continue
        p1_points, p1_valid = candidate_slice(arrays, "p1", frame_index)
        p4_points, p4_valid = candidate_slice(arrays, "p4", frame_index)
        p1_points = p1_points[p1_valid]
        p4_points = p4_points[p4_valid]

        if len(p1_points) != 2:
            reasons["P1 count"] = reasons.get("P1 count", 0) + 1
            continue
        if len(p4_points) < 2:
            reasons["P4 count"] = reasons.get("P4 count", 0) + 1
            continue

        p1_points = p1_points[np.argsort(p1_points[:, 0])]
        p1_left, p1_right = p1_points
        p1_vector = p1_right - p1_left
        p1_length = np.linalg.norm(p1_vector)
        p1_angle = np.degrees(np.arctan2(p1_vector[1], p1_vector[0]))
        stages = {"angle": 0, "scale": 0}
        for first in range(len(p4_points)):
            for second in range(first + 1, len(p4_points)):
                left, right = sorted((p4_points[first], p4_points[second]),
                                     key=lambda point: point[0])
                p4_vector = right - left
                p4_angle = np.degrees(np.arctan2(p4_vector[1], p4_vector[0]))
                angle_error = abs((p4_angle - p1_angle + 90.0) % 180.0 - 90.0)
                if angle_error > angle_gate:
                    continue
                stages["angle"] += 1
                scale_error = abs(np.linalg.norm(p4_vector) / p1_length / seed_scale - 1.0)
                if scale_error > scale_gate:
                    continue
                stages["scale"] += 1

        if not stages["angle"]:
            reason = "3-degree angle gate"
        elif not stages["scale"]:
            reason = "10-percent scale gate"
        else:
            reason = "unclassified"
        reasons[reason] = reasons.get(reason, 0) + 1
    return reasons


parser = argparse.ArgumentParser()
parser.add_argument("pickle", type=Path)
parser.add_argument("video", type=Path)
parser.add_argument("--output-prefix", type=Path, required=True)
args = parser.parse_args()

with args.pickle.open("rb") as file:
    payload = pickle.load(file)
arrays = payload["arrays"]
pupil_invalid = ~arrays["pupil_valid"]
pair_missing = ~arrays["pair_valid"]
pupil_invalid_pair_selected = pupil_invalid & arrays["pair_valid"]
valid_pupil_pair_missing = arrays["pupil_valid"] & pair_missing
rejection_reasons = classify_pair_rejections(
    arrays, payload["seed_scale_ratio"], payload["angle_gate_deg"],
    payload["scale_gate_fraction"])

pupil_runs = sorted(runs(pupil_invalid), key=lambda run: run[1] - run[0], reverse=True)
pair_runs = sorted(runs(pair_missing), key=lambda run: run[1] - run[0], reverse=True)
recovered_runs = sorted(runs(pupil_invalid_pair_selected),
                        key=lambda run: run[1] - run[0], reverse=True)
groups = {
    "pupil_invalid": pupil_runs,
    "pupil_invalid_pair_selected": recovered_runs[:12],
    "pair_missing": pair_runs[:12],
}
targets = {}
for cause, selected_runs in groups.items():
    for run in selected_runs:
        targets[(run[0] + run[1]) // 2] = (cause, run)

capture = cv2.VideoCapture(str(args.video))
rendered = {cause: [] for cause in groups}
frame_index = 0
while targets:
    decoded, frame = capture.read()
    if not decoded:
        break
    target = targets.pop(frame_index, None)
    if target is not None:
        cause, run = target
        rendered[cause].append((frame_index, annotate(frame, arrays, frame_index, cause, run)))
    frame_index += 1
capture.release()

for cause, selected in rendered.items():
    selected.sort()
    if selected:
        contact_sheet([frame for _, frame in selected],
                      args.output_prefix.with_name(f"{args.output_prefix.name}_{cause}.png"),
                      columns=4 if cause == "pupil_invalid" else 2,
                      tile_width=560 if cause == "pupil_invalid" else 900)

summary_path = args.output_prefix.with_name(f"{args.output_prefix.name}_summary.txt")
with summary_path.open("w", encoding="utf-8") as file:
    print(f"frames={len(arrays['frame_index'])}", file=file)
    print(f"pair_valid={int(arrays['pair_valid'].sum())}", file=file)
    print(f"pupil_invalid={int(pupil_invalid.sum())} runs={len(pupil_runs)}", file=file)
    print(f"pupil_invalid_pair_selected={int(pupil_invalid_pair_selected.sum())} "
          f"runs={len(recovered_runs)}", file=file)
    print(f"valid_pupil_pair_missing={int(valid_pupil_pair_missing.sum())}", file=file)
    print(f"pair_missing={int(pair_missing.sum())} runs={len(pair_runs)}", file=file)
    print(f"pair_rejection_reasons={rejection_reasons}", file=file)
    print("longest pupil-invalid runs:", file=file)
    for start, end in pupil_runs[:15]:
        print(f"  {start}-{end}: {end - start + 1}", file=file)
    print("longest pair-missing runs:", file=file)
    for start, end in pair_runs[:15]:
        print(f"  {start}-{end}: {end - start + 1}", file=file)

print(f"Saved {summary_path} and {len(rendered)} contact sheets")
