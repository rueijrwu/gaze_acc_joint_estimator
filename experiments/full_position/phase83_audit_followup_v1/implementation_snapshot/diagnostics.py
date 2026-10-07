"""Add covariance and context-support diagnostics to existing experiment outputs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import load_reviewed
from .geometry import context, area_gradient
from .model import PositionModel
from .noise import reference_covariance, marginal, whitening
from .schema import load_model, write_json


COVARIANCE_DEFINITION = "local fixed-calibration conditional covariance; excludes coefficient uncertainty and model discrepancy"


def subset_support(ctx, meta, state, empirical_range=None):
    """Subset's own state support, never support copied from the all-three fit."""
    result = dict(theta_anchor=None, A_anchor=None, theta_empirical=None, A_empirical=None,
        P1_context_valid=bool(ctx.valid), context_outside_training_extrema=None,
        p1_parity_seen_in_training=None, unavailable_reasons={},
        definition="componentwise training extrema diagnostics; not a validated multidimensional envelope",
        training_group_ids=meta.get("provenance", {}).get("training_group_ids"))
    x = np.asarray(state, float) if state is not None else None
    for label, bounds in (("anchor", meta.get("provenance", {}).get("anchor_range")),
                          ("empirical", empirical_range)):
        for axis, name in enumerate(("theta", "A")):
            key = name+"_"+label
            if x is None or x.shape != (2,) or not np.isfinite(x).all():
                result["unavailable_reasons"][key] = "subset_state_unavailable"
            elif bounds is None:
                result["unavailable_reasons"][key] = "training_"+label+"_range_unavailable"
            else:
                result[key] = bool(bounds[0][axis] <= x[axis] <= bounds[1][axis])
    support = meta.get("provenance", {}).get("conditional_context_support", {})
    for key in ("context_outside_training_extrema", "p1_parity_seen_in_training"):
        if not ctx.valid:
            result["unavailable_reasons"][key] = "invalid_P1_geometry"
        elif key == "context_outside_training_extrema" and "r_min" in support and "r_max" in support:
            result[key] = bool(np.any((ctx.r < support["r_min"]) | (ctx.r > support["r_max"])))
        elif key == "p1_parity_seen_in_training" and "signed_P1_area_branches" in support:
            result[key] = bool(np.sign(ctx.signed_area) in support["signed_P1_area_branches"])
        else:
            result["unavailable_reasons"][key] = "training_context_metadata_unavailable"
    return result


def enrich_subset_records(frames, captures, meta, training_states=None, model_hash=None, channels="xy"):
    from .schema import clean_json, protocol_metadata
    noise = {k: meta.get(k) for k in ("pilot_coefficients", "reference_state", "coordinate_covariance", "normalization")}
    noise_hash = hashlib.sha256(json.dumps(clean_json(noise), sort_keys=True).encode()).hexdigest()
    empirical = None
    if training_states is not None and len(training_states):
        x = np.asarray(training_states, float)
        empirical = [x.min(0), x.max(0)]
    for frame in frames:
        ctx = context(captures[frame["capture"]].p[frame["row"]])
        frame.update(protocol=protocol_metadata(), E_cross=frame.get("E_px"),
                     G_theta_cross=frame.get("G_theta_deg"), G_A_cross=frame.get("G_A_D"))
        for slot in frame["slots"]:
            slot["support"] = subset_support(ctx, meta, slot.get("state"), empirical)
            slot.update(retained_image_channels=slot.get("retained_image_channels") or channels,
                normalization=meta.get("normalization"), frozen_model_sha256=model_hash,
                frozen_noise_policy_sha256=noise_hash, protocol=protocol_metadata())
    return frames


def frame_diagnostics(model, meta, p, point_valid, inverse, empirical_range=None, R=None):
    """Same support/parity/covariance contract for evaluation and new captures."""
    ctx = context(p)
    record = dict(p1_edge_condition=float(ctx.edge_condition), p1_signed_area=float(ctx.signed_area),
        residual_covariance=None, state_covariance_physical=None,
        context_outside_training_extrema=None, state_outside_empirical_training_range=None,
        state_outside_anchor_range=None, p1_parity_seen_in_training=None,
        covariance_definition=COVARIANCE_DEFINITION)
    if not ctx.valid:
        return record
    sigma = np.asarray(meta['coordinate_covariance'])
    support = meta.get('provenance', {}).get('conditional_context_support', {})
    if 'r_min' in support and 'r_max' in support:
        record['context_outside_training_extrema'] = bool(np.any((ctx.r < support['r_min']) | (ctx.r > support['r_max'])))
    if 'signed_P1_area_branches' in support:
        record['p1_parity_seen_in_training'] = bool(np.sign(ctx.signed_area) in support['signed_P1_area_branches'])
    grad = area_gradient(ctx.p).ravel()
    record['p1_area_signal_to_noise'] = float(abs(ctx.signed_area)/np.sqrt(grad@sigma[:6,:6]@grad))
    if R is None:
        pilot = PositionModel(27, meta['pilot_coefficients'])
        R = reference_covariance(np.asarray(p)[None], pilot, np.asarray(meta['reference_state']), sigma, model.channels == 2)[0]
    record['residual_covariance'] = R.tolist()
    if inverse.get('available'):
        x = np.asarray(inverse['state'])
        anchor_range = meta.get('provenance', {}).get('anchor_range')
        if anchor_range is not None:
            record['state_outside_anchor_range'] = bool(np.any((x < anchor_range[0]) | (x > anchor_range[1])))
        if empirical_range is not None:
            record['state_outside_empirical_training_range'] = bool(np.any((x < empirical_range[0]) | (x > empirical_range[1])))
        ix = np.repeat(point_valid, 2).nonzero()[0] if model.channels == 6 else np.arange(2)
        _, J = model.predict(x, ctx.r, True)
        WJ = whitening(marginal(R, ix))@J[ix]
        if inverse['rank'] == 2 and not inverse['at_bound'] and not inverse['ambiguous']:
            record['state_covariance_physical'] = np.linalg.inv(WJ.T@WJ).tolist()
    return record


def enrich(run):
    run = Path(run)
    captures, _, _ = load_reviewed(Path(__file__).resolve().parents[1])
    for path in sorted(run.glob("*/*/model.json")):
        frames_path = path.parent/"frames.json"
        if not frames_path.exists():
            continue
        model, meta = load_model(path)
        trained = json.loads((path.parent/"training_states.json").read_text())
        states = np.array(trained["states"])
        low, high = states.min(0), states.max(0)
        records = []
        for row in json.loads(frames_path.read_text()):
            cap = captures[row["capture"]]
            inverse = dict(available=row['estimated'])
            if row['estimated']:
                inverse.update(state=[row['theta'], row['A']], rank=row['rank'],
                               at_bound=row['at_bound'], ambiguous=row['ambiguous'])
            record = frame_diagnostics(model, meta, cap.p[row['row']], cap.point_valid[row['row']], inverse, [low, high])
            record.update(row=row['row'], capture=row['capture'], fixation=row['fixation'])
            records.append(record)
        write_json(path.parent/"frame_uncertainty.json", dict(records=records,
            empirical_training_state_range=[low.tolist(), high.tolist()],
            context_support_definition="componentwise training extrema diagnostic; not a validated support envelope",
            covariance_definition=COVARIANCE_DEFINITION))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    enrich(args.run)
