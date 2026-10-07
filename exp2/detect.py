"""Track the full video, save NumPy results, and overlay only its last frame."""

import argparse
import itertools
import json
import math
import pickle
from pathlib import Path

import cv2
import numpy as np
import pkj_image_process_py as ip


experiment_dir = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Track P1 and P4 candidates in a video.")
parser.add_argument("input_video", type=Path,
                    help="video file to process (for example exp2/capture_1.mkv)")
parser.add_argument("-o", "--output-pickle", "--output", dest="output_pickle", type=Path,
                    help="output pickle path (default: <video>_detections.pkl beside input)")
parser.add_argument("--overlay", type=Path,
                    help="last-frame overlay path (default: <video>_overlay.png beside input)")
parser.add_argument("--config", type=Path, default=experiment_dir.parent / "config.json",
                    help="pipeline config JSON (default: repository config.json)")
parser.add_argument("--max-frames", type=int,
                    help="maximum number of frames to process (default: entire video)")
parser.add_argument("--log-file", type=Path,
                    help="optional per-frame text log path (no frame log by default)")
args = parser.parse_args()
if args.max_frames is not None and args.max_frames < 1:
    parser.error("--max-frames must be a positive integer")

video_path = args.input_video.expanduser().resolve()
config_path = args.config.expanduser().resolve()
results_path = (args.output_pickle.expanduser().resolve() if args.output_pickle else
                video_path.with_name(f"{video_path.stem}_detections.pkl"))
output_path = (args.overlay.expanduser().resolve() if args.overlay else
               video_path.with_name(f"{video_path.stem}_overlay.png"))
log_path = args.log_file.expanduser().resolve() if args.log_file else None
if not video_path.is_file():
    parser.error(f"input video does not exist: {video_path}")
if not config_path.is_file():
    parser.error(f"config file does not exist: {config_path}")
if results_path == video_path or output_path == video_path:
    parser.error("output paths must not overwrite the input video")
if results_path == output_path:
    parser.error("pickle and overlay output paths must be different")
if log_path is not None and log_path in (video_path, results_path, output_path):
    parser.error("log path must be different from input and output paths")
results_path.parent.mkdir(parents=True, exist_ok=True)
output_path.parent.mkdir(parents=True, exist_ok=True)
if log_path is not None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
max_pair_angle_error_deg = 3.0
max_pair_scale_error = 0.10

config = json.loads(config_path.read_text())
image_config = config["pipeline"]["image_process"]
static = ip.PipelineStaticConfig()
for key, value in image_config["static"].items():
    if hasattr(static, key):
        setattr(static, key, value)
static.enable_debug_outputs = bool(config["system"]["enable_debug_outputs"])
static.enable_telemetry = bool(config["system"]["enable_telemetry"])
params = ip.PipelineParams()
for key, value in image_config["params"].items():
    if hasattr(params, key):
        setattr(params, key, value)

native = ip.ImageProcess()
native.initialize(static)
capture = cv2.VideoCapture(str(video_path))
log_file = log_path.open("w", encoding="utf-8") if log_path is not None else None
if log_file is not None:
    print(f"input_video={video_path} max_frames={args.max_frames}", file=log_file)

# The P4 pair is seeded on the first valid-pupil frame with exactly one
# geometrically eligible pupil-contained pair. Every frame requires both pupil
# validity flags before accepting/exporting any P1 or P4 candidates. An invalid
# frame preserves the established scale for a later valid-pupil frame. Candidate
# slots are not persistent IDs; pair angle and scale are checked against P1.
p4_scale = None
last_frame = None
last_result = None
last_selection = [None, None]
last_pupil_valid = False
frames_with_extra_p4 = 0
frames_with_outside_p4 = 0
frames_with_full_pair = 0
frames_with_valid_pupil = 0
max_selected_angle_error_deg = 0.0
max_selected_scale_error = 0.0
frames_processed = 0
last_timestamp_ms = None
records = {key: [] for key in (
    "frame_index", "timestamp_ms", "pupil_valid", "pupil_center", "pupil_axes",
    "pupil_angle_deg", "p1_count", "p4_count", "p1_valid_count",
    "p4_valid_count", "p4_inside_count", "p4_selected_indices",
    "pair_valid", "p1_points", "p4_points", "p1_vector", "p4_vector",
    "p1_length", "p4_length", "p1_angle_deg", "p4_angle_deg",
    "angle_difference_deg", "angle_error_deg", "scale_ratio",
    "scale_difference_percent",
)}
p1_candidate_xy = []
p1_candidate_valid = []
p1_candidate_offsets = [0]
p4_candidate_xy = []
p4_candidate_valid = []
p4_candidate_offsets = [0]

try:
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    frame_index = 0
    while args.max_frames is None or frame_index < args.max_frames:
        decoded, frame = capture.read()
        if not decoded:
            break

        timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        if (last_timestamp_ms is not None
                and timestamp_ms < last_timestamp_ms - 1000):
            break
        last_timestamp_ms = timestamp_ms

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape != (static.frame_height, static.frame_width):
            raise ValueError(f"Expected {(static.frame_height, static.frame_width)}, got {gray.shape}")

        native.upload(gray)
        native.process(params)
        native.wait_for_completion()
        result = native.get_result()

        pupil = result.pupil
        pupil_valid = bool(result.pupil_valid and pupil.valid)
        p1_candidates = result.p1[:result.p1_count] if pupil_valid else []
        p4_candidates = result.p4[:result.p4_count] if pupil_valid else []
        for loc in p1_candidates:
            p1_candidate_xy.append((loc.cx, loc.cy))
            p1_candidate_valid.append(bool(loc.valid))
        p1_candidate_offsets.append(len(p1_candidate_xy))
        for loc in p4_candidates:
            p4_candidate_xy.append((loc.cx, loc.cy))
            p4_candidate_valid.append(bool(loc.valid))
        p4_candidate_offsets.append(len(p4_candidate_xy))

        selection = [None, None]
        selected_p1_angle = None
        selected_p4_angle = None
        selected_angle_error = None
        selected_scale_error = None
        selected_scale_ratio = None
        valid_p1_locs = [loc for loc in p1_candidates if loc.valid]
        valid_p1 = len(valid_p1_locs)
        valid_p4 = [(index, loc) for index, loc in enumerate(p4_candidates)
                    if loc.valid]
        if len(valid_p4) > 2:
            frames_with_extra_p4 += 1

        inside_pupil = []
        if pupil_valid:
            frames_with_valid_pupil += 1
            angle_rad = math.radians(pupil.angle)
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)
            for candidate_index, loc in valid_p4:
                dx = loc.cx - pupil.cx
                dy = loc.cy - pupil.cy
                ellipse_x = cos_angle * dx + sin_angle * dy
                ellipse_y = -sin_angle * dx + cos_angle * dy
                normalized_radius_sq = ((ellipse_x / pupil.major) ** 2 +
                                        (ellipse_y / pupil.minor) ** 2)
                if normalized_radius_sq <= 1.0:
                    inside_pupil.append((candidate_index, loc))
            if len(inside_pupil) < len(valid_p4):
                frames_with_outside_p4 += 1

        if len(valid_p1_locs) == 2:
            p1_left, p1_right = sorted(valid_p1_locs, key=lambda loc: loc.cx)
            p1_dx = p1_right.cx - p1_left.cx
            p1_dy = p1_right.cy - p1_left.cy
            p1_length = math.hypot(p1_dx, p1_dy)
            p1_angle = math.degrees(math.atan2(p1_dy, p1_dx))
            best_cost = float("inf")
            best_p4_angle = None
            best_angle_error = None
            best_scale_error = None
            best_scale_ratio = None
            eligible_pair_count = 0

            # P4 containment remains authoritative; invalid-pupil frames have
            # no accepted candidates and cannot initialize or use the seed.
            pair_candidates = inside_pupil

            if p1_length > 0:
                for (first_index, first_loc), (second_index, second_loc) in itertools.combinations(
                        pair_candidates, 2):
                    if first_loc.cx <= second_loc.cx:
                        left_index, left_loc = first_index, first_loc
                        right_index, right_loc = second_index, second_loc
                    else:
                        left_index, left_loc = second_index, second_loc
                        right_index, right_loc = first_index, first_loc
                    p4_dx = right_loc.cx - left_loc.cx
                    p4_dy = right_loc.cy - left_loc.cy
                    p4_length = math.hypot(p4_dx, p4_dy)
                    if p4_length == 0:
                        continue
                    p4_angle = math.degrees(math.atan2(p4_dy, p4_dx))
                    angle_error = abs((p4_angle - p1_angle + 90.0) % 180.0 - 90.0)
                    if angle_error > max_pair_angle_error_deg:
                        continue
                    scale = p4_length / p1_length
                    scale_error = abs(scale / p4_scale - 1.0) if p4_scale is not None else 0.0
                    if scale_error > max_pair_scale_error:
                        continue
                    eligible_pair_count += 1

                    cost = 2.0 * angle_error + 100.0 * scale_error
                    if cost < best_cost:
                        best_cost = cost
                        selection = [left_index, right_index]
                        best_p4_angle = p4_angle
                        best_angle_error = angle_error
                        best_scale_error = scale_error
                        best_scale_ratio = scale

            if p4_scale is None and eligible_pair_count != 1:
                selection = [None, None]

            if all(index is not None for index in selection):
                selected_p1_angle = p1_angle
                selected_p4_angle = best_p4_angle
                selected_angle_error = best_angle_error
                selected_scale_error = best_scale_error
                selected_scale_ratio = best_scale_ratio
                if p4_scale is None:
                    p4_scale = math.hypot(
                        result.p4[selection[1]].cx - result.p4[selection[0]].cx,
                        result.p4[selection[1]].cy - result.p4[selection[0]].cy,
                    ) / p1_length
                max_selected_angle_error_deg = max(max_selected_angle_error_deg,
                                                   best_angle_error)
                max_selected_scale_error = max(max_selected_scale_error,
                                               best_scale_error)

        if all(index is not None for index in selection):
            frames_with_full_pair += 1
        pair_valid = all(index is not None for index in selection)
        p1_points = ((math.nan, math.nan), (math.nan, math.nan))
        p4_points = ((math.nan, math.nan), (math.nan, math.nan))
        p1_vector = (math.nan, math.nan)
        p4_vector = (math.nan, math.nan)
        p1_length_out = math.nan
        p4_length_out = math.nan
        angle_difference = math.nan
        scale_difference_percent = math.nan
        if pair_valid:
            p1_left, p1_right = sorted(valid_p1_locs, key=lambda loc: loc.cx)
            p4_left, p4_right = (result.p4[index] for index in selection)
            p1_points = ((p1_left.cx, p1_left.cy), (p1_right.cx, p1_right.cy))
            p4_points = ((p4_left.cx, p4_left.cy), (p4_right.cx, p4_right.cy))
            p1_vector = (p1_right.cx - p1_left.cx, p1_right.cy - p1_left.cy)
            p4_vector = (p4_right.cx - p4_left.cx, p4_right.cy - p4_left.cy)
            p1_length_out = math.hypot(*p1_vector)
            p4_length_out = math.hypot(*p4_vector)
            angle_difference = ((selected_p4_angle - selected_p1_angle + 90.0)
                                % 180.0 - 90.0)
            scale_difference_percent = 100.0 * (selected_scale_ratio / p4_scale - 1.0)

        records["frame_index"].append(frame_index)
        records["timestamp_ms"].append(timestamp_ms)
        records["pupil_valid"].append(pupil_valid)
        records["pupil_center"].append((pupil.cx, pupil.cy) if pupil_valid
                                       else (math.nan, math.nan))
        records["pupil_axes"].append((pupil.major, pupil.minor) if pupil_valid
                                     else (math.nan, math.nan))
        records["pupil_angle_deg"].append(pupil.angle if pupil_valid else math.nan)
        records["p1_count"].append(len(p1_candidates))
        records["p4_count"].append(len(p4_candidates))
        records["p1_valid_count"].append(valid_p1)
        records["p4_valid_count"].append(len(valid_p4))
        records["p4_inside_count"].append(len(inside_pupil) if pupil_valid else 0)
        records["p4_selected_indices"].append(tuple(selection) if pair_valid else (-1, -1))
        records["pair_valid"].append(pair_valid)
        records["p1_points"].append(p1_points)
        records["p4_points"].append(p4_points)
        records["p1_vector"].append(p1_vector)
        records["p4_vector"].append(p4_vector)
        records["p1_length"].append(p1_length_out)
        records["p4_length"].append(p4_length_out)
        records["p1_angle_deg"].append(selected_p1_angle if pair_valid else math.nan)
        records["p4_angle_deg"].append(selected_p4_angle if pair_valid else math.nan)
        records["angle_difference_deg"].append(angle_difference)
        records["angle_error_deg"].append(selected_angle_error if pair_valid else math.nan)
        records["scale_ratio"].append(selected_scale_ratio if pair_valid else math.nan)
        records["scale_difference_percent"].append(scale_difference_percent)

        if log_file is not None:
            p1_left_text = (f"{p1_points[0][0]:.2f},{p1_points[0][1]:.2f}"
                            if pair_valid else "nan,nan")
            p1_right_text = (f"{p1_points[1][0]:.2f},{p1_points[1][1]:.2f}"
                             if pair_valid else "nan,nan")
            p4_left_text = (f"{p4_points[0][0]:.2f},{p4_points[0][1]:.2f}"
                            if pair_valid else "nan,nan")
            p4_right_text = (f"{p4_points[1][0]:.2f},{p4_points[1][1]:.2f}"
                             if pair_valid else "nan,nan")
            print(
                f"frame={frame_index} timestamp_ms={timestamp_ms:.3f} "
                f"pupil_valid={pupil_valid} p1_valid={valid_p1} "
                f"p4_valid={len(valid_p4)} "
                f"p4_inside={len(inside_pupil) if pupil_valid else 0} "
                f"pair_valid={pair_valid} p1_left={p1_left_text} "
                f"p1_right={p1_right_text} p4_left={p4_left_text} "
                f"p4_right={p4_right_text} "
                f"p1_angle_deg={selected_p1_angle if pair_valid else math.nan:.4f} "
                f"p4_angle_deg={selected_p4_angle if pair_valid else math.nan:.4f} "
                f"angle_error_deg={selected_angle_error if pair_valid else math.nan:.4f} "
                f"scale_ratio={selected_scale_ratio if pair_valid else math.nan:.6f} "
                f"scale_difference_percent={scale_difference_percent:.4f}",
                file=log_file,
            )

        last_frame = frame
        last_result = result
        last_selection = selection
        last_pupil_valid = pupil_valid
        frame_index += 1
        frames_processed = frame_index
        if log_file is not None and frames_processed % 100 == 0:
            log_file.flush()
        if log_file is not None and frames_processed % 1000 == 0:
            print(
                f"Tracking progress: {frames_processed} frames; "
                f"valid pupil={frames_with_valid_pupil}; "
                f"selected P4 pair={frames_with_full_pair}",
                flush=True,
            )
finally:
    capture.release()
    if log_file is not None:
        log_file.close()

if last_frame is None:
    raise RuntimeError("No frames were processed")

if last_pupil_valid:
    pupil = last_result.pupil
    cv2.ellipse(last_frame,
                ((float(pupil.cx), float(pupil.cy)),
                 (2.0 * pupil.major, 2.0 * pupil.minor), float(pupil.angle)),
                (0, 255, 0), 2, cv2.LINE_AA)
    cv2.drawMarker(last_frame, (round(pupil.cx), round(pupil.cy)),
                   (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

    last_p1 = [(index, loc) for index, loc in enumerate(last_result.p1[:last_result.p1_count])
               if loc.valid]
    last_p1.sort(key=lambda item: item[1].cx)
    for endpoint, (candidate_index, loc) in zip(("left", "right"), last_p1):
        position = (round(loc.cx), round(loc.cy))
        cv2.circle(last_frame, position, 20, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(last_frame, f"P1 {endpoint} ({candidate_index})",
                    (position[0] + 25, position[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)

    for candidate_index, loc in enumerate(last_result.p4[:last_result.p4_count]):
        if loc.valid:
            position = (round(loc.cx), round(loc.cy))
            cv2.circle(last_frame, position, 9, (255, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(last_frame, f"P4 candidate {candidate_index}",
                        (position[0] + 15, position[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    for label, candidate_index in zip(("P4 left", "P4 right"), last_selection):
        if candidate_index is not None:
            loc = last_result.p4[candidate_index]
            position = (round(loc.cx), round(loc.cy))
            cv2.circle(last_frame, position, 18, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(last_frame, position, (0, 255, 255), cv2.MARKER_CROSS, 12, 1)
            cv2.putText(last_frame, label, (position[0] + 20, position[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

cv2.putText(last_frame,
            f"Frame {frames_processed - 1}: P4 candidates {records['p4_count'][-1]}, "
            f"selected {sum(index is not None for index in last_selection)}/2",
            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
if not cv2.imwrite(str(output_path), last_frame):
    raise RuntimeError(f"Could not save {output_path}")

integer_keys = {"frame_index", "p1_count", "p4_count", "p1_valid_count",
                "p4_valid_count", "p4_inside_count", "p4_selected_indices"}
boolean_keys = {"pupil_valid", "pair_valid"}
arrays = {}
for key, values in records.items():
    dtype = np.int32 if key in integer_keys else np.bool_ if key in boolean_keys else np.float32
    arrays[key] = np.asarray(values, dtype=dtype)
arrays["p1_candidate_xy"] = np.asarray(p1_candidate_xy, dtype=np.float32).reshape(-1, 2)
arrays["p1_candidate_valid"] = np.asarray(p1_candidate_valid, dtype=np.bool_)
arrays["p1_candidate_offsets"] = np.asarray(p1_candidate_offsets, dtype=np.int64)
arrays["p4_candidate_xy"] = np.asarray(p4_candidate_xy, dtype=np.float32).reshape(-1, 2)
arrays["p4_candidate_valid"] = np.asarray(p4_candidate_valid, dtype=np.bool_)
arrays["p4_candidate_offsets"] = np.asarray(p4_candidate_offsets, dtype=np.int64)

payload = {
    "schema_version": 1,
    "requires_valid_pupil": True,
    "source_video": str(video_path),
    "frame_width": static.frame_width,
    "frame_height": static.frame_height,
    "seed_scale_ratio": p4_scale,
    "angle_gate_deg": max_pair_angle_error_deg,
    "scale_gate_fraction": max_pair_scale_error,
    "max_frames_requested": args.max_frames,
    "arrays": arrays,
}
with results_path.open("wb") as file:
    pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)

seed_scale_text = f"{p4_scale:.3f}" if p4_scale is not None else "none"
print(
    f"Processed {frames_processed} frames; valid pupil in {frames_with_valid_pupil}; "
    f">2 valid P4 candidates in {frames_with_extra_p4}; "
    f"outside-pupil P4 in {frames_with_outside_p4}; "
    f"full P4 pair selected in {frames_with_full_pair}; "
    f"seed scale={seed_scale_text}; max pair angle error={max_selected_angle_error_deg:.2f} deg; "
    f"max pair scale error={max_selected_scale_error:.3f}; "
    f"Saved {results_path} and {output_path}",
    flush=True,
)
if log_path is not None:
    print(f"Per-frame log saved to {log_path}", flush=True)
