"""Supplemental retained-only profile of earlier ambiguous/weak-rank x solves.

Diagnostic only: never replaces saved branch policy, scores, or model selection.
"""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import numpy as np
from full_position.audit83 import source_folder
from full_position.geometry import context
from full_position.information import TransformedResponse, retained_transform
from full_position.model import PositionModel
from full_position.noise import marginal, reference_covariance
from full_position.profile import profile_inverse
from full_position.schema import clean_json, load_model, write_json
from full_position.sensitivity import digest, reviewed

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[2]


def task(case):
    response, fold, frame, slot = case
    captures, _, _ = reviewed(str(ROOT))
    frozen = source_folder(ROOT, response, fold)
    model, meta = load_model(frozen/"model.json")
    cap = captures[frame["capture"]]
    row, held = frame["row"], slot["held_point"]
    ctx = context(cap.p[row])
    kept, H = retained_transform(held, "x")
    # Select only retained x measurements; no held coordinates or image-y data.
    ix = kept[::2]
    observed_x = (cap.q[row].ravel()[ix]-ctx.c[ix % 2])/ctx.ell
    pilot = PositionModel(27, meta["pilot_coefficients"])
    R = reference_covariance(cap.p[row:row+1], pilot, np.asarray(meta["reference_state"]),
                             np.asarray(meta["coordinate_covariance"]))[0]
    response_model = TransformedResponse(model, kept, H)
    result = profile_inverse(response_model, ctx.r, observed_x, H@marginal(R, kept)@H.T, grid=65)
    old_states = [np.asarray(b["state"]) for b in slot.get("branches", [])]
    new_branches = [b for b in result.get("branches", []) if not any(
        np.all(np.abs(np.asarray(b["state"])-s) < [.01, .01]) for s in old_states)]
    return dict(response=response, fold=fold, capture=frame["capture"], fixation=frame["fixation"],
        row=row, held_point=held, frozen_model_sha256=digest(frozen/"model.json"),
        primary_cost=slot["retained_subset_cost"], primary_state=slot["state"],
        primary_rank=slot.get("rank"), primary_unambiguous=slot["unambiguous"],
        primary_branch_count=len(old_states), profile=result, newly_discovered_branches=new_branches,
        diagnostic_cost_minus_primary=(result["cost"]-slot["retained_subset_cost"] if result["available"] else None),
        score_replaced=False)


def main():
    cases = []
    for path in sorted((ROOT/"experiments/full_position/phase83_retained_channels_v1").glob("*/*/x_frames.json")):
        response, fold = path.parent.parent.name, path.parent.name
        for frame in json.loads(path.read_text()):
            for slot in frame["slots"]:
                if slot["available"] and (not slot["identifiable"] or not slot["unambiguous"]):
                    cases.append((response, fold, frame, slot))
    with ProcessPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(task, cases))
    with gzip.open(OUTPUT/"existing_x_profile_audits.jsonl.gz", "wt") as stream:
        for result in results:
            stream.write(json.dumps(clean_json(result), allow_nan=False)+"\n")
    write_json(OUTPUT/"existing_x_profile_summary.json", dict(cases=len(cases),
        rank_weak=sum(r["primary_rank"] < 2 for r in results),
        ambiguous=sum(not r["primary_unambiguous"] for r in results),
        profiles_available=sum(r["profile"]["available"] for r in results),
        newly_discovered_branch_cases=sum(bool(r["newly_discovered_branches"]) for r in results),
        lower_retained_cost_cases=sum(r["diagnostic_cost_minus_primary"] is not None and
            r["diagnostic_cost_minus_primary"] < -1e-8*(1+abs(r["primary_cost"])) for r in results),
        scope="independent retained-only 65-grid polynomial profile; finite grid is not global completeness",
        primary_scores_changed=False, script_sha256=digest(Path(__file__)),
        cost_gap_policy="saved gap=2 is heuristic, not a calibrated probability or significance test"))


if __name__ == "__main__":
    main()
