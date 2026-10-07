"""Saved-result boundary transitions and training-only state gauge diagnostics."""
from __future__ import annotations
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
from .schema import clean_json, write_json
from .sensitivity import digest, identity, joined, read_json, stats


def training_gains(reference, candidate):
    for field in ("rows", "groups", "nominal_anchors"):
        if reference[field] != candidate[field]:
            raise ValueError("Gauge diagnostics require identical ordered training rows/groups")
    x, y = (np.asarray(r["states"], float) for r in (reference, candidate))
    groups = np.asarray(reference["groups"])
    _, counts = np.unique(groups, return_counts=True)
    labels, inverse = np.unique(groups, return_inverse=True)
    weights = 1/(len(labels)*counts[inverse])
    a, b = x.copy(), y.copy()
    a[:, 1] += 1
    b[:, 1] += 1
    raw = np.sum(weights[:, None]*a*b, axis=0)/np.sum(weights[:, None]*a*a, axis=0)
    gains = np.maximum(raw, np.finfo(float).eps)
    residual = b-a*gains
    return dict(gaze_gain=float(gains[0]), one_plus_accommodation_gain=float(gains[1]),
        unconstrained_gains=raw.tolist(), positive_constraint_active=(raw <= 0).tolist(),
        residual_rms_physical=np.sqrt(np.sum(weights[:, None]*residual**2, axis=0)).tolist(),
        unadjusted_shift_rms_physical=np.sqrt(np.sum(weights[:, None]*(y-x)**2, axis=0)).tolist(),
        count=len(x), weighting="equal fixation", usage="descriptive training-only; evaluation states unchanged")


def transitions(reference, candidate):
    """Exact paired squared changes, including the four bound/interior transitions."""
    a, b = ({identity(p, True): p for f in frames for p in f["slots"]}
            for frames in (reference, candidate))
    if a.keys() != b.keys():
        raise ValueError("Comparison requires identical scheduled point identities")
    records = []
    for key in sorted(a):
        old, new = a[key], b[key]
        shared = old["scored"] and new["scored"]
        category = (("interior" if old["interior"] else "bound")+"_to_"+
                    ("interior" if new["interior"] else "bound")) if shared else "unavailable_pair"
        record = dict(identity=key, transition=category, shared_scored=shared,
            capture=old["capture"], fixation=old["fixation"], signed_gaze=old["nominal_theta"],
            reference_scored=old["scored"], candidate_scored=new["scored"])
        if shared:
            eo, en = (np.asarray(p["error_px"]) for p in (old, new))
            record.update(delta_squared_vector_error_px2=float(en@en-eo@eo),
                          delta_squared_axis_error_px2=(en**2-eo**2).tolist())
        records.append(record)
    # Complete-frame E/G comparisons are separately paired on exact frame IDs.
    fa, fb = ({identity(f): f for f in frames} for frames in (reference, candidate))
    frame_records = []
    for key in sorted(fa):
        old, new = fa[key], fb[key]
        if not (old["complete_triple"] and new["complete_triple"]):
            continue
        frame_records.append(dict(identity=key, capture=old["capture"], fixation=old["fixation"],
            signed_gaze=old["nominal_theta"], transition=("interior" if old["complete_interior"] else "bound")+
                "_to_"+("interior" if new["complete_interior"] else "bound"),
            **{f"delta_squared_{field}": float(new[field]**2-old[field]**2)
               for field in ("E_px", "G_theta_deg", "G_A_D", "worst_point_px")}))
    def summarize(rows, field):
        grouped = defaultdict(list)
        for row in rows:
            if field in row:
                grouped[(row["identity"][1], row["capture"], row["fixation"])].append(row[field])
        return dict(pooled=stats([r[field] for r in rows if field in r]),
                    equal_fixation_mean=float(np.mean([np.mean(v) for v in grouped.values()])) if grouped else None)
    strata = {}
    for label, getter in (("transition", lambda r: r["transition"]),
                          ("fixation", lambda r: str((r["identity"][1], r["capture"], r["fixation"]))),
                          ("signed_gaze", lambda r: str(r["signed_gaze"])),
                          ("capture", lambda r: r["capture"])):
        strata[label] = {}
        for value in sorted(set(map(getter, records))):
            rows = [r for r in records if getter(r) == value]
            fr = [r for r in frame_records if getter(r) == value]
            strata[label][value] = dict(scheduled_points=len(rows), shared_scored_points=sum(r["shared_scored"] for r in rows),
                point_squared_error_change=summarize(rows, "delta_squared_vector_error_px2"),
                paired_complete_frames=len(fr), frame_squared_changes={field: summarize(fr, f"delta_squared_{field}")
                    for field in ("E_px", "G_theta_deg", "G_A_D", "worst_point_px")})
    return dict(points=records, frames=frame_records, strata=strata)


def run(root, source, output, candidate_source=None):
    root, source, output = map(Path, (root, source, output))
    candidate_source = source if candidate_source is None else Path(candidate_source)
    output.mkdir(parents=True, exist_ok=False)
    folds = read_json(source/"config.json")["folds"]
    variants = read_json(candidate_source/"config.json")["variants"]
    results, hashes = {}, {}
    source_hashes = {str(p): digest(p) for base in {source, candidate_source}
                     for p in (base/"variants").rglob("*") if p.is_file()}
    with gzip.open(output/"paired_transition_membership.jsonl.gz", "wt") as stream:
        for variant in variants:
            for name in ("conditional27", "conditional37"):
                gains = {}
                for fold in folds:
                    base = source/"variants"/"baseline"/fold/name/"training_states.json"
                    candidate = candidate_source/"variants"/variant/fold/name/"training_states.json"
                    if not candidate.exists():
                        continue
                    hashes[str(base)] = digest(base)
                    hashes[str(candidate)] = digest(candidate)
                    gains[fold] = training_gains(read_json(base), read_json(candidate))
                for family in ("gaze", "capture"):
                    selected = [f for f in folds if f.startswith(family+"_")]
                    ref = [r for f in selected for r in joined(root, source/"variants"/"baseline", f, name, "baseline")]
                    new = [r for f in selected for r in joined(root, candidate_source/"variants"/variant, f, name, variant)]
                    comparison = transitions(ref, new)
                    key = f"{family}/{variant}/{name}"
                    stream.write(json.dumps(clean_json(dict(comparison=key, points=comparison.pop("points"),
                        frames=comparison.pop("frames"))), allow_nan=False)+"\n")
                    results[key] = dict(**comparison, training_gains={f: gains[f] for f in selected if f in gains})
    write_json(output/"summary.json", dict(schema="phase82_audit_transition_v1", comparisons=results,
        training_source_hashes=hashes, source_hashes=source_hashes,
        source_unchanged=all(digest(p) == h for p, h in source_hashes.items()),
        meaning="candidate minus baseline squared error; negative is improvement; descriptive development comparison"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source", type=Path, default=Path("experiments/full_position/joint_sensitivity_v1"))
    parser.add_argument("--candidate-source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.root, args.source, args.output, args.candidate_source)
