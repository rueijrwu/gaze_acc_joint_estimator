#!/usr/bin/env python3
"""Aggregate saved G7 compact-check JSON; never runs inference or changes a fit."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_vec(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        result = [float(value[0]), float(value[1])]
    except (TypeError, ValueError):
        return None
    return result if all(math.isfinite(v) for v in result) else None


def stats(errors):
    """Metrics over 2D errors; coordinate RMS uses denominator 2*n."""
    if not errors:
        return {"n": 0, "coordinate_rms_px": None, "vector_norm": _empty_stats(),
                "signed_bias_px": {"x": None, "y": None},
                "axis_rms_px": {"x": None, "y": None}}
    a = np.asarray(errors, dtype=np.float64).reshape(-1, 2)
    norms = np.linalg.norm(a, axis=1)
    return {
        "n": int(len(a)),
        "coordinate_rms_px": float(np.sqrt(np.mean(a * a))),
        "vector_norm": _distribution(norms),
        "signed_bias_px": {"x": float(a[:, 0].mean()), "y": float(a[:, 1].mean())},
        "axis_rms_px": {"x": float(np.sqrt(np.mean(a[:, 0] ** 2))),
                        "y": float(np.sqrt(np.mean(a[:, 1] ** 2)))},
    }


def _empty_stats():
    return {"median": None, "p90": None, "p95": None, "max": None}


def _distribution(values):
    a = np.asarray(values, dtype=np.float64)
    return {"median": float(np.quantile(a, .5)), "p90": float(np.quantile(a, .9)),
            "p95": float(np.quantile(a, .95)), "max": float(a.max())}


def eligible(slot):
    return bool(slot.get("certified")) and not bool(slot.get("ambiguous")) and slot.get("error_xy_px") is not None


def _group_summary(slots):
    scored = [s for s in slots if eligible(s)]
    errors = [s["error_xy_px"] for s in scored]
    return {
        "scheduled_slots": len(slots),
        "raw_input_valid": sum(bool(s.get("raw_input_valid")) for s in slots),
        "retained_input_valid": sum(bool(s.get("retained_input_valid")) for s in slots),
        "held_input_valid": sum(bool(s.get("held_input_valid")) for s in slots),
        "certified": sum(bool(s.get("certified")) for s in slots),
        "ambiguous": sum(bool(s.get("ambiguous")) for s in slots),
        "unresolved": sum(s.get("inference_status") == "unresolved" for s in slots),
        "input_unavailable": sum(s.get("inference_status") == "input_unavailable" for s in slots),
        "at_bounds": sum(bool(s.get("at_bounds")) for s in slots),
        "scored": len(scored),
        "scores_at_bounds": sum(bool(s.get("at_bounds")) for s in scored),
        "metrics": stats(errors),
    }


def _key(slot):
    return (str(slot.get("capture")), int(slot["exposure"]), int(slot["row"]),
            int(slot["source_frame"]), int(slot["held_point"]))


def load_compact(path: Path):
    raw = path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document.get("slots"), list):
        raise ValueError(f"{path}: missing slots list")
    records = []
    for index, original in enumerate(document["slots"]):
        slot = dict(original)
        error = finite_vec(slot.get("held_native_error_px"))
        slot["error_xy_px"] = error
        slot["vector_error_norm_px"] = float(np.linalg.norm(error)) if error is not None else None
        slot["schedule_index_in_file"] = index
        records.append(slot)

    # Recover the original five selected positions in each exposure by their
    # encounter order in the persisted schedule. Preserve every position,
    # including both endpoints and unavailable slots.
    position_by_key = {}
    for held in sorted({int(s["held_point"]) for s in records}):
        seen = defaultdict(int)
        for s in records:
            if int(s["held_point"]) != held:
                continue
            exp = int(s["exposure"])
            seen[exp] += 1
            position_by_key[_key(s)] = seen[exp]
    counts = defaultdict(int)
    for s in records:
        counts[(int(s["held_point"]), int(s["exposure"]))] += 1
    for s in records:
        count = counts[(int(s["held_point"]), int(s["exposure"]))]
        position = position_by_key[_key(s)]
        s["schedule_position"] = position
        s["schedule_position_count"] = count
        s["schedule_endpoint"] = position in (1, count)
    return document, records, hashlib.sha256(raw).hexdigest()


def summarize_file(path: Path):
    document, slots, digest = load_compact(path)
    by_held = defaultdict(list)
    by_exposure = defaultdict(list)
    by_held_exposure = defaultdict(list)
    by_position = defaultdict(list)
    by_bound = defaultdict(list)
    for slot in slots:
        h, e = int(slot["held_point"]), int(slot["exposure"])
        by_held[h].append(slot)
        by_exposure[e].append(slot)
        by_held_exposure[(h, e)].append(slot)
        by_position[(h, int(slot["schedule_position"]))].append(slot)
        by_bound[(bool(slot.get("at_bounds")), h)].append(slot)

    held_summary = {str(h): _group_summary(v) for h, v in sorted(by_held.items())}
    exposure_summary = {str(e): _group_summary(v) for e, v in sorted(by_exposure.items())}
    held_exposure_summary = {f"{h}:{e}": _group_summary(v)
                             for (h, e), v in sorted(by_held_exposure.items())}
    position_summary = {}
    for (held, pos), group in sorted(by_position.items()):
        # A position may have only a single exposure's slots. Include exposure
        # and row identities so endpoints remain auditable.
        position_summary[f"held{held}:position{pos}"] = {
            "slots": [{"exposure": s["exposure"], "row": s["row"],
                       "source_frame": s["source_frame"], "endpoint": s["schedule_endpoint"],
                       "inference_status": s.get("inference_status"),
                       "at_bounds": bool(s.get("at_bounds")), "scored": eligible(s),
                       "error_xy_px": s["error_xy_px"]} for s in group],
            **_group_summary(group),
        }
    bound_summary = {f"at_bounds={int(bound)}:held={held}": _group_summary(group)
                     for (bound, held), group in sorted(by_bound.items())}
    eligible_slots = [s for s in slots if eligible(s)]
    summary = {
        "input_file": path.name,
        "input_sha256": digest,
        "kind": document.get("kind"),
        "model_hash": document.get("model_hash"),
        "original_summary": document.get("summary"),
        "all_slots": _group_summary(slots),
        "by_held_point": held_summary,
        "by_exposure": exposure_summary,
        "by_held_point_exposure": held_exposure_summary,
        "by_held_point_schedule_position": position_summary,
        "by_bound_and_held_point": bound_summary,
        "slot_count": len(slots),
        "slots": slots,
    }
    return summary


def frame_summary(records):
    """Build complete held-point triples, retaining incomplete group records."""
    grouped = defaultdict(list)
    for slot in records:
        key = (str(slot["capture"]), int(slot["exposure"]), int(slot["row"]), int(slot["source_frame"]))
        grouped[key].append(slot)
    frames = []
    for key, slots in sorted(grouped.items()):
        by_held = defaultdict(list)
        for slot in slots:
            by_held[int(slot["held_point"])].append(slot)
        triples = all(len(by_held[h]) == 1 for h in (0, 1, 2))
        triple = [by_held[h][0] for h in (0, 1, 2)] if triples else []
        complete = triples and all(eligible(s) for s in triple)
        record = {"capture": key[0], "exposure": key[1], "row": key[2], "source_frame": key[3],
                  "slot_count": len(slots), "held_point_ids": sorted(by_held),
                  "complete_eligible_triple": bool(complete),
                  "certified_count": sum(bool(s.get("certified")) for s in slots),
                  "ambiguous_count": sum(bool(s.get("ambiguous")) for s in slots),
                  "at_bounds_count": sum(bool(s.get("at_bounds")) for s in slots),
                  "slots": [{"held_point": int(s["held_point"]),
                             "certified": bool(s.get("certified")),
                             "ambiguous": bool(s.get("ambiguous")),
                             "at_bounds": bool(s.get("at_bounds")),
                             "inference_status": s.get("inference_status"),
                             "error_xy_px": s.get("error_xy_px"),
                             "state_visual_theta_deg": s.get("state_visual_theta_deg"),
                             "state_A_D": s.get("state_A_D"),
                             "schedule_position": s.get("schedule_position"),
                             "schedule_endpoint": s.get("schedule_endpoint")} for s in sorted(slots, key=lambda z: int(z["held_point"]))]}
        if complete:
            errors = np.asarray([s["error_xy_px"] for s in triple], dtype=np.float64)
            theta = np.asarray([float(s["state_visual_theta_deg"]) for s in triple])
            accommodation = np.asarray([float(s["state_A_D"]) for s in triple])
            theta_pair_sq = [(theta[j] - theta[k]) ** 2 for j in range(3) for k in range(j + 1, 3)]
            a_pair_sq = [(accommodation[j] - accommodation[k]) ** 2 for j in range(3) for k in range(j + 1, 3)]
            record.update({"E_i_squared_px2": float(np.mean(np.sum(errors * errors, axis=1))),
                           "E_i_px": float(np.sqrt(np.mean(np.sum(errors * errors, axis=1)))),
                           "Gtheta_i_squared_deg2": float(np.mean(theta_pair_sq)),
                           "Gtheta_i_deg": float(np.sqrt(np.mean(theta_pair_sq))),
                           "GA_i_squared_D2": float(np.mean(a_pair_sq)),
                           "GA_i_D": float(np.sqrt(np.mean(a_pair_sq)))})
        else:
            record.update({"E_i_squared_px2": None, "E_i_px": None,
                           "Gtheta_i_squared_deg2": None, "Gtheta_i_deg": None,
                           "GA_i_squared_D2": None, "GA_i_D": None})
        frames.append(record)
    return frames


def summarize_frames(frames):
    eligible_frames = [f for f in frames if f["complete_eligible_triple"]]
    by_exp = defaultdict(list)
    for frame in eligible_frames:
        by_exp[int(frame["exposure"])].append(frame)
    all_exposures = sorted({int(f["exposure"]) for f in frames})
    exposure_records = []
    for exposure in all_exposures:
        fs = by_exp.get(exposure, [])
        if fs:
            vals = {name: float(np.mean([f[key] for f in fs]))
                    for name, key in (("E_i_squared_px2", "E_i_squared_px2"),
                                      ("Gtheta_i_squared_deg2", "Gtheta_i_squared_deg2"),
                                      ("GA_i_squared_D2", "GA_i_squared_D2"))}
        else:
            vals = {"E_i_squared_px2": None, "Gtheta_i_squared_deg2": None, "GA_i_squared_D2": None}
        exposure_records.append({"exposure": exposure, "eligible_frames": len(fs), **vals})
    complete = len(by_exp) == 20 and all(by_exp.get(k) for k in range(20))
    # A frame contributes its exposure's mean squared metric divided by 20.
    # This is its exact additive contribution to the equal-exposure scorecard.
    # Keep the unweighted frame values as well; the weighted contribution is
    # a descriptive attribution, not a new score or eligibility rule.
    for frame in eligible_frames:
        n_exp = len(by_exp[int(frame["exposure"])])
        frame["equal_exposure_contribution"] = {
            "E_i_squared_px2": frame["E_i_squared_px2"] / (20 * n_exp),
            "Gtheta_i_squared_deg2": frame["Gtheta_i_squared_deg2"] / (20 * n_exp),
            "GA_i_squared_D2": frame["GA_i_squared_D2"] / (20 * n_exp),
            "exposure_frame_count": n_exp,
            "exposure_weight": 1 / 20,
        }
    for frame in frames:
        if not frame["complete_eligible_triple"]:
            frame["equal_exposure_contribution"] = None
    equal_exposure = {}
    for key, root in (("E_i_squared_px2", "E_px"),
                      ("Gtheta_i_squared_deg2", "Gtheta_deg"),
                      ("GA_i_squared_D2", "GA_D")):
        if complete:
            mean_sq = float(np.mean([r[key] for r in exposure_records]))
            equal_exposure[key] = mean_sq
            equal_exposure[root] = float(np.sqrt(mean_sq))
        else:
            equal_exposure[key] = None
            equal_exposure[root] = None
    metrics = (("E_i_squared_px2", "E_px", "E_i_px"),
               ("Gtheta_i_squared_deg2", "Gtheta_deg", "Gtheta_i_deg"),
               ("GA_i_squared_D2", "GA_D", "GA_i_D"))
    ranked = {}
    if complete:
        for source_key, _, _ in metrics:
            ordered = sorted(eligible_frames,
                             key=lambda f: (-f["equal_exposure_contribution"][source_key],
                                            f["capture"], f["exposure"], f["row"], f["source_frame"]))
            total = float(np.mean([r[source_key] for r in exposure_records]))
            cumulative = 0.0
            entries = []
            for rank, frame in enumerate(ordered, 1):
                contribution = frame["equal_exposure_contribution"][source_key]
                cumulative += contribution
                entries.append({"rank": rank,
                                "capture": frame["capture"], "exposure": frame["exposure"],
                                "row": frame["row"], "source_frame": frame["source_frame"],
                                "frame_value": frame[source_key],
                                "weighted_contribution": contribution,
                                "share_of_equal_exposure_total": contribution / total if total else None,
                                "cumulative_share_top_k": cumulative / total if total else None,
                                "at_bounds_count": frame["at_bounds_count"],
                                "endpoint_count": sum(bool(s.get("schedule_endpoint")) for s in frame["slots"]),
                                "slots": frame["slots"]})
            ranked[source_key] = {"total": total, "top_contributors": entries}
    return {"eligible_complete_frames": len(eligible_frames), "all_frame_groups": len(frames),
            "present_exposures": sorted(by_exp), "missing_exposures": sorted(set(range(20)) - set(by_exp)),
            "all_20_exposures_present": complete,
            "equal_exposure_metrics": equal_exposure,
            "by_exposure": exposure_records,
            "bound_strata": {
                "any_slot_at_bounds": _frame_stratum(eligible_frames, lambda f: f["at_bounds_count"] > 0),
                "no_slot_at_bounds": _frame_stratum(eligible_frames, lambda f: f["at_bounds_count"] == 0),
            },
            "endpoint_strata": {
                "any_slot_endpoint": _frame_stratum(eligible_frames, lambda f: any(bool(s.get("schedule_endpoint")) for s in f["slots"])),
                "no_slot_endpoint": _frame_stratum(eligible_frames, lambda f: not any(bool(s.get("schedule_endpoint")) for s in f["slots"])),
            },
            "ranked_equal_exposure_contributions": ranked,
            "frames": frames}


def _frame_stratum(frames, predicate):
    selected = [f for f in frames if predicate(f)]
    result = {"frames": len(selected),
            "E_i_px": float(np.sqrt(np.mean([f["E_i_squared_px2"] for f in selected]))) if selected else None,
            "Gtheta_i_deg": float(np.sqrt(np.mean([f["Gtheta_i_squared_deg2"] for f in selected]))) if selected else None,
            "GA_i_D": float(np.sqrt(np.mean([f["GA_i_squared_D2"] for f in selected]))) if selected else None}
    for key, metric in (("E_i_px", "E_i_px"), ("Gtheta_i_deg", "Gtheta_i_deg"), ("GA_i_D", "GA_i_D")):
        values = [f[metric] for f in selected]
        result[key + "_distribution"] = ({"median": float(np.median(values)), "p95": float(np.quantile(values, .95))}
                                          if values else {"median": None, "p95": None})
    return result


def compare_to_final(file_summaries):
    """Pair each outer checkpoint to its same-start final on common IDs."""
    by_start = defaultdict(list)
    for item in file_summaries:
        name = item["input_file"]
        stem = name[:-5] if name.endswith(".json") else name
        start = next((candidate for candidate in ("continued", "perturbed", "common")
                      if stem.startswith(candidate)), None)
        if start:
            by_start[start].append(item)
    comparisons = []
    for start, items in sorted(by_start.items()):
        finals = [x for x in items if x["input_file"] == f"{start}_final.json"]
        if len(finals) != 1:
            continue
        final = finals[0]
        final_slots = {_key(s): s for s in final["slots"]}
        final_eligible = {k: s for k, s in final_slots.items() if eligible(s)}
        for checkpoint in sorted(items, key=lambda x: x["input_file"]):
            if checkpoint is final:
                continue
            old_slots = {_key(s): s for s in checkpoint["slots"]}
            old_eligible = {k: s for k, s in old_slots.items() if eligible(s)}
            common = sorted(set(old_eligible) & set(final_eligible))
            old_only, final_only = set(old_eligible) - set(final_eligible), set(final_eligible) - set(old_eligible)
            old_err = [old_eligible[k]["error_xy_px"] for k in common]
            new_err = [final_eligible[k]["error_xy_px"] for k in common]
            delta = (np.asarray(new_err) - np.asarray(old_err)) if common else np.empty((0, 2))
            old_frames = {tuple((f["capture"], f["exposure"], f["row"], f["source_frame"])): f
                          for f in frame_summary(checkpoint["slots"]) if f["complete_eligible_triple"]}
            new_frames = {tuple((f["capture"], f["exposure"], f["row"], f["source_frame"])): f
                          for f in frame_summary(final["slots"]) if f["complete_eligible_triple"]}
            common_frames = sorted(set(old_frames) & set(new_frames))
            comparisons.append({
                "start": start, "checkpoint": checkpoint["input_file"], "final": final["input_file"],
                "checkpoint_sha256": checkpoint["input_sha256"], "final_sha256": final["input_sha256"],
                "checkpoint_eligible_slots": len(old_eligible), "final_eligible_slots": len(final_eligible),
                "common_eligible_slots": len(common), "lost_eligible_slots": len(old_only),
                "gained_eligible_slots": len(final_only),
                "coverage_change_slots": len(final_eligible) - len(old_eligible),
                "common_slot_ids": [list(k) for k in common],
                "common_slot_checkpoint_metrics": stats(old_err),
                "common_slot_final_metrics": stats(new_err),
                "common_slot_signed_mean_change_px": ({"x": float(delta[:, 0].mean()), "y": float(delta[:, 1].mean())}
                                                        if len(delta) else {"x": None, "y": None}),
                "checkpoint_eligible_frames": len(old_frames), "final_eligible_frames": len(new_frames),
                "common_eligible_frames": len(common_frames),
                "common_frame_ids": [list(k) for k in common_frames],
                "common_frame_metric_change": {
                    "mean_E_i_squared_change_px2": float(np.mean([new_frames[k]["E_i_squared_px2"] - old_frames[k]["E_i_squared_px2"] for k in common_frames])) if common_frames else None,
                    "mean_Gtheta_i_squared_change_deg2": float(np.mean([new_frames[k]["Gtheta_i_squared_deg2"] - old_frames[k]["Gtheta_i_squared_deg2"] for k in common_frames])) if common_frames else None,
                    "mean_GA_i_squared_change_D2": float(np.mean([new_frames[k]["GA_i_squared_D2"] - old_frames[k]["GA_i_squared_D2"] for k in common_frames])) if common_frames else None,
                },
            })
    return comparisons


def markdown_report(doc):
    lines = ["# G7 saved compact-check aggregation", "",
             "This report aggregates already-saved compact JSON records only. It does not run inference, refit a model, or turn the compact diagnostic into G8. All scheduled slots, missing outcomes, and endpoints are retained.", "",
             f"Attempt: `{doc['attempt']}`", "",
             "## Input files", "", "| File | SHA-256 | Slots | Scored | Unavailable | Unresolved |", "|---|---|---:|---:|---:|---:|"]
    for f in doc["files"]:
        a = f["all_slots"]
        lines.append(f"| `{f['input_file']}` | `{f['input_sha256']}` | {a['scheduled_slots']} | {a['scored']} | {a['input_unavailable']} | {a['unresolved']} |")
    selected = next((f for f in doc["files"] if f["input_file"] == "perturbed_final.json"), None)
    if selected is not None:
        lines += ["", "## Selected perturbed final: pooled descriptive summaries", "",
                  "These are pooled slot-level descriptions, not equal-exposure scorecards. They retain the saved five schedule positions, including both endpoints.", "",
                  "### By held point", "", "| Held P4 | Scored | Vector median (px) | p95 | max | x bias (px) | y bias (px) | x RMS (px) | y RMS (px) |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for held, row in sorted(selected["by_held_point"].items(), key=lambda kv: int(kv[0])):
            m = row["metrics"]
            lines.append(f"| {held} | {m['n']} | {_fmt(m['vector_norm']['median'])} | {_fmt(m['vector_norm']['p95'])} | {_fmt(m['vector_norm']['max'])} | {_fmt(m['signed_bias_px']['x'])} | {_fmt(m['signed_bias_px']['y'])} | {_fmt(m['axis_rms_px']['x'])} | {_fmt(m['axis_rms_px']['y'])} |")
        lines += ["", "### By saved schedule position", "", "Position 1 and position 5 are the original per-exposure endpoints; no endpoint slots are dropped.", "",
                  "| Held P4 | Position | Endpoint | Scored | Vector median (px) | p95 | max | x bias (px) | y bias (px) | x RMS (px) | y RMS (px) |", "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for key, row in sorted(selected["by_held_point_schedule_position"].items()):
            head, pos = key.split(":position")
            m = row["metrics"]
            endpoint = "yes" if int(pos) in (1, 5) else "no"
            lines.append(f"| {head.removeprefix('held')} | {pos} | {endpoint} | {m['n']} | {_fmt(m['vector_norm']['median'])} | {_fmt(m['vector_norm']['p95'])} | {_fmt(m['vector_norm']['max'])} | {_fmt(m['signed_bias_px']['x'])} | {_fmt(m['signed_bias_px']['y'])} | {_fmt(m['axis_rms_px']['x'])} | {_fmt(m['axis_rms_px']['y'])} |")
        bound_groups = {True: [], False: []}
        for slot in selected["slots"]:
            if eligible(slot):
                bound_groups[bool(slot.get("at_bounds"))].append(slot["error_xy_px"])
        lines += ["", "### Bound versus non-bound slots", "", "Pooled descriptive strata; these are not equal-exposure scorecards.", "",
                  "| State at bound | Scored | Vector median (px) | p95 | max | x bias (px) | y bias (px) | x RMS (px) | y RMS (px) |", "|:---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for bound in (True, False):
            m = stats(bound_groups[bound])
            lines.append(f"| {'yes' if bound else 'no'} | {m['n']} | {_fmt(m['vector_norm']['median'])} | {_fmt(m['vector_norm']['p95'])} | {_fmt(m['vector_norm']['max'])} | {_fmt(m['signed_bias_px']['x'])} | {_fmt(m['signed_bias_px']['y'])} | {_fmt(m['axis_rms_px']['x'])} | {_fmt(m['axis_rms_px']['y'])} |")
    lines += ["", "## Complete held-point triples", "",
              "For each complete eligible frame, `E_i²` is the mean of the three squared 2D native-pixel errors. `Gtheta_i²` and `GA_i²` are the mean of the three pairwise squared state differences. Exposure means are computed on squared quantities and then equally weighted; an absent exposure makes the all-exposure summary incomplete.", "",
              "| Compact file | Eligible triples | Exposures present | Missing exposures | Equal-exposure E (px) | Gtheta (deg) | GA (D) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for item in doc["files"]:
        t = item["triple_summary"]
        m = t["equal_exposure_metrics"]
        v = lambda x: "incomplete" if x is None else f"{x:.6g}"
        lines.append(f"| `{item['input_file']}` | {t['eligible_complete_frames']} | {len(t['present_exposures'])}/20 | {','.join(map(str,t['missing_exposures'])) or 'none'} | {v(m['E_px'])} | {v(m['Gtheta_deg'])} | {v(m['GA_D'])} |")
    if selected is not None:
        triple = selected["triple_summary"]
        ranked = triple["ranked_equal_exposure_contributions"]
        e_key, t_key, a_key = "E_i_squared_px2", "Gtheta_i_squared_deg2", "GA_i_squared_D2"
        e_top = ranked[e_key]["top_contributors"]
        t_top = ranked[t_key]["top_contributors"]
        a_top = ranked[a_key]["top_contributors"]
        e_map = {(x["capture"], x["exposure"], x["row"], x["source_frame"]): x for x in e_top}
        t_map = {(x["capture"], x["exposure"], x["row"], x["source_frame"]): x for x in t_top}
        a_map = {(x["capture"], x["exposure"], x["row"], x["source_frame"]): x for x in a_top}
        lines += ["", "### Largest frame contributions to the equal-exposure scorecard", "",
                  "Each contribution is the frame's squared metric divided by 20 times the number of eligible frames in that exposure. The table is ranked by E²; `Gtheta²` and `GA²` are that same frame's contributions, not their independent rank order. `Share` is fraction of the corresponding equal-exposure total. These are descriptive attributions and do not establish why a frame differs.", "",
                  "| Rank | Capture / exposure / row / source frame | Endpoint slots | Bound slots | E² (px²) | E² contribution | E² share | Gtheta² (deg²) | Gtheta² contribution | Gtheta² share | GA² (D²) | GA² contribution | GA² share |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for item in e_top[:10]:
            identity = (item["capture"], item["exposure"], item["row"], item["source_frame"])
            t = t_map[identity]; a = a_map[identity]
            slots = item["slots"]
            lines.append(f"| {item['rank']} | {identity[0]} / {identity[1]} / {identity[2]} / {identity[3]} | {item['endpoint_count']} | {item['at_bounds_count']} | {_fmt(item['frame_value'])} | {_fmt(item['weighted_contribution'])} | {_fmt(100*item['share_of_equal_exposure_total'])}% | {_fmt(t['frame_value'])} | {_fmt(t['weighted_contribution'])} | {_fmt(100*t['share_of_equal_exposure_total'])}% | {_fmt(a['frame_value'])} | {_fmt(a['weighted_contribution'])} | {_fmt(100*a['share_of_equal_exposure_total'])}% |")
        e2 = e_top[1]
        t2 = t_top[1]
        lines += ["", f"The two largest E² rows contribute **{100*(e_top[0]['share_of_equal_exposure_total'] + e2['share_of_equal_exposure_total']):.4f}%** of equal-exposure E². The two largest Gtheta² rows contribute **{100*(t_top[0]['share_of_equal_exposure_total'] + t2['share_of_equal_exposure_total']):.4f}%** of equal-exposure Gtheta².", "",
                  "#### Independent top contributors by metric", "",
                  "| Metric | Rank | Capture / exposure / row / source frame | Frame squared metric | Weighted contribution | Share | Cumulative top-k share |", "|---|---:|---|---:|---:|---:|---:|"]
        for label, key in (("E² (px²)", e_key), ("Gtheta² (deg²)", t_key), ("GA² (D²)", a_key)):
            for entry in ranked[key]["top_contributors"][:5]:
                ident = f"{entry['capture']} / {entry['exposure']} / {entry['row']} / {entry['source_frame']}"
                lines.append(f"| {label} | {entry['rank']} | {ident} | {_fmt(entry['frame_value'])} | {_fmt(entry['weighted_contribution'])} | {_fmt(100*entry['share_of_equal_exposure_total'])}% | {_fmt(100*entry['cumulative_share_top_k'])}% |")
        lines += ["", "#### Matched triple partitions", "",
                  "Complete triples are partitioned descriptively by whether any of the three scheduled held slots is an original endpoint (position 1 or 5), and whether any state is at a bound. RMS columns summarize squared frame metrics; median and p95 show the frame-level tails. These partitions are not causal comparisons.", "",
                  "| Partition | Frames | E RMS (px) | E median / p95 (px) | Gtheta RMS (deg) | Gtheta median / p95 (deg) | GA RMS (D) | GA median / p95 (D) |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for group_name, strata in (("Endpoint", triple["endpoint_strata"]), ("Bound", triple["bound_strata"])):
            labels = (("any_slot_endpoint", "any endpoint"), ("no_slot_endpoint", "interior only")) if group_name == "Endpoint" else (("any_slot_at_bounds", "any state at bound"), ("no_slot_at_bounds", "no state at bound"))
            for stratum_key, label in labels:
                row = strata[stratum_key]
                e_dist = row["E_i_px_distribution"]; t_dist = row["Gtheta_i_deg_distribution"]; a_dist = row["GA_i_D_distribution"]
                lines.append(f"| {group_name}: {label} | {row['frames']} | {_fmt(row['E_i_px'])} | {_fmt(e_dist['median'])} / {_fmt(e_dist['p95'])} | {_fmt(row['Gtheta_i_deg'])} | {_fmt(t_dist['median'])} / {_fmt(t_dist['p95'])} | {_fmt(row['GA_i_D'])} | {_fmt(a_dist['median'])} / {_fmt(a_dist['p95'])} |")
    lines += ["", "## Matched checkpoint coverage", "",
              "Each checkpoint is compared with its same-start final snapshot on common eligible slot/frame IDs. Coverage gains/losses are reported separately from matched residual changes.", "",
              "| Start | Checkpoint | Common slots | Lost | Gained | Common frames | Mean ΔE² (px²) |", "|---|---|---:|---:|---:|---:|---:|"]
    for c in doc["matched_to_final"]:
        lines.append(f"| {c['start']} | `{c['checkpoint']}` | {c['common_eligible_slots']} | {c['lost_eligible_slots']} | {c['gained_eligible_slots']} | {c['common_eligible_frames']} | {c['common_frame_metric_change']['mean_E_i_squared_change_px2']} |")
    lines += ["", "## Interpretation limits", "",
              "The subset state spreads `Gtheta` and `GA` measure internal agreement among overlapping retained-point inversions, not accuracy. Errors use native relative pixels and are not divided by inferred scale. This compact schedule is a progress diagnostic, not the full G8 evaluation. The slot and frame records in `summary.json` preserve the identities and outcomes behind these tables.", ""]
    return "\n".join(lines)


def _fmt(value):
    return "—" if value is None else f"{value:.6g}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True, type=Path, help="Existing G7 attempt directory")
    parser.add_argument("--output", required=True, type=Path, help="Fresh output directory for aggregation artifacts")
    args = parser.parse_args(argv)
    attempt = args.attempt.resolve()
    output = args.output.resolve()
    compact = attempt / "compact"
    if not compact.is_dir():
        parser.error(f"attempt has no compact directory: {compact}")
    files = sorted(compact.glob("*.json"))
    if not files:
        parser.error(f"no compact JSON files in {compact}")
    if output.exists() and any(output.iterdir()):
        parser.error(f"output directory must be fresh or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    summaries = []
    for path in files:
        summary = summarize_file(path)
        summary["triple_summary"] = summarize_frames(frame_summary(summary["slots"]))
        summaries.append(summary)
    document = {
        "schema": "g7_compact_aggregation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": str(attempt),
        "method": {
            "input_scope": "compact/*.json only; no inference or fitting",
            "slot_eligibility": "certified and nonambiguous with finite held_native_error_px",
            "coordinate_rms": "sqrt(mean(error_x^2,error_y^2)) across scored slots; denominator 2*S",
            "frame_eligibility": "exactly one eligible record for each held_point 0,1,2 sharing capture/exposure/row/source_frame",
            "frame_metrics": "E_i^2=mean_j ||e_ij||^2; Gtheta_i^2 and GA_i^2 are means of the three pairwise squared differences",
            "exposure_weighting": "mean frame squared metrics within exposure, then equal mean across all 20 exposures; incomplete when any exposure has no eligible triple",
            "position_order": "one-based encounter order within exposure in the saved schedule; first and last remain marked endpoints",
        },
        "files": summaries,
        "matched_to_final": compare_to_final(summaries),
    }
    json_path = output / "summary.json"
    json_path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "REPORT.md").write_text(markdown_report(document))
    print(f"Aggregated {len(files)} compact JSON files, {sum(f['slot_count'] for f in summaries)} slot records")
    print(f"Wrote {json_path} and {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
