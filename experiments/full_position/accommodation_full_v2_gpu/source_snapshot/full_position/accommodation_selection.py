"""Inner paired ranking with coverage gates and no accuracy cutoffs.

Selection permits an outer evaluation. It never promotes a deployment model.
Nominal errors, state spread, support and clipping remain diagnostics.
"""
import numpy as np
from .population import PopulationManifest, frame_id
from .scorecard import build

VERSION = "paired_crosscheck_no_absolute_accuracy_gates_v1"
FRACTIONS = ("minimum_scored_fraction","minimum_complete_fraction",
            "minimum_shared_frame_fraction","minimum_shared_frame_fraction_per_exposure")


def declared_policy():
    return dict(version=VERSION,provenance="AR0 sampled development: 48 train / 8 evaluation core rows",
        minimum_scored_fraction=.8,minimum_complete_fraction=.8,
        minimum_shared_frame_fraction=.8,minimum_shared_frame_fraction_per_exposure=.5,
        absolute_accuracy_thresholds=None,nominal_error_is_acceptance_gate=False,
        tie_rule="exact loss tie favors the log reference",deployment_promotion=False,
        uncertainty="paired exposure summaries; overlapping folds are correlated")


def choose(inner_records,guards=None,reference=None,population=None):
    if not isinstance(population,PopulationManifest):
        raise ValueError("Declare an independent immutable scheduled population")
    for frames in inner_records.values():
        population.validate(frames)
    base = dict(selected=None,reference=reference,promoted_for_deployment=False,policy_version=VERSION)
    if not guards or guards.get("version") != VERSION or not guards.get("provenance") or any(k not in guards for k in FRACTIONS):
        return dict(base,status="inconclusive",reason="a complete predeclared coverage policy is required")
    if guards.get("absolute_accuracy_thresholds","missing") is not None or guards.get("nominal_error_is_acceptance_gate") is not False:
        raise ValueError("This policy has no absolute accuracy or nominal-target gates")
    if any(isinstance(guards[k],bool) or not np.isfinite(guards[k]) or not 0<=guards[k]<=1 for k in FRACTIONS):
        raise ValueError("Coverage fractions must be finite values between zero and one")
    cards = {name:build(rows,purpose="inner_validation",population=population) for name,rows in inner_records.items()}
    eligible,rejected = [],{}
    for name,card in cards.items():
        n,nf = card["coverage"]["scheduled_slots"],card["coverage"]["scheduled_frames"]
        checks = dict(scored_coverage=card["coverage"]["scored"]/n>=guards[FRACTIONS[0]],
            complete_coverage=card["coverage"]["complete_triples"]/nf>=guards[FRACTIONS[1]],
            exposure_integrity=not card["outcomes"]["E_cross_px"]["squared_error_aggregation"]["absent_exposure_ids"])
        scored = [s for f in inner_records[name] for s in f["slots"] if s.get("scored")]
        checks["finite_certified_scores"] = bool(scored) and all(
            all(s.get(k) is True for k in ("available","certified","identifiable","unambiguous"))
            and np.asarray(s.get("state"),float).shape == (2,)
            and np.isfinite(np.asarray(s.get("state"),float)).all()
            and np.asarray(s.get("error_px"),float).shape == (2,)
            and np.isfinite(np.asarray(s.get("error_px"),float)).all() for s in scored)
        if all(checks.values()): eligible.append(name)
        else: rejected[name] = [k for k,v in checks.items() if not v]
    if reference not in eligible:
        return dict(base,status="inconclusive",reason="no valid fresh log-reference comparison",rejected=rejected)
    tables = {name:{frame_id(f):f for f in inner_records[name] if f["complete_triple"]} for name in eligible}
    shared = set.intersection(*(set(t) for t in tables.values()))
    coverage = [dict(exposure_id=k,scheduled=sum(i[:3]==k for i in population.frame_ids),
        shared=sum(i[:3]==k for i in shared)) for k in population.exposures]
    for item in coverage: item["fraction"] = item["shared"]/item["scheduled"]
    shared_coverage = dict(scheduled_frames=len(population.frame_ids),shared_frames=len(shared),
        overall_fraction=len(shared)/len(population.frame_ids),per_exposure=coverage,
        absent_exposure_ids=[v["exposure_id"] for v in coverage if not v["shared"]])
    if (shared_coverage["overall_fraction"]<guards[FRACTIONS[2]] or shared_coverage["absent_exposure_ids"]
        or any(v["fraction"]<guards[FRACTIONS[3]] for v in coverage)):
        return dict(base,status="inconclusive",reason="shared cohort fails declared coverage",
            shared_coverage=shared_coverage,rejected=rejected)
    paired = {name:build([table[k] for k in sorted(shared)],expected_exposures=population.exposures,
        purpose="inner_validation") for name,table in tables.items()}
    losses = {name:card["outcomes"]["E_cross_px"]["squared_error_aggregation"]["mean"] for name,card in paired.items()}
    if any(value is None or not np.isfinite(value) for value in losses.values()):
        raise ValueError("Paired ranking requires finite squared losses")
    selected = min(eligible,key=lambda n:(losses[n],n!=reference,n))
    tradeoffs = {}
    for name,card in paired.items():
        flags=[]
        for metric in ("G_theta_cross_deg","G_A_cross_D","worst_point_px"):
            a,b=card["outcomes"][metric]["equal_exposure_rms"],paired[reference]["outcomes"][metric]["equal_exposure_rms"]
            if a is not None and b is not None and a>b: flags.append(metric)
        for axis in ("x","y"):
            if card["axis_errors"][axis]["equal_exposure_rms"]>paired[reference]["axis_errors"][axis]["equal_exposure_rms"]:
                flags.append(axis+"_axis")
        for metric in ("E_cross_px","worst_point_px"):
            if card["outcomes"][metric]["distribution"]["p95"]>paired[reference]["outcomes"][metric]["distribution"]["p95"]:
                flags.append(metric+"_p95")
        tradeoffs[name]=flags
    return dict(base,status="selected_for_outer_evaluation",selected=selected,
        paired_inner_L_cross=losses,shared_frame_ids=sorted(shared),shared_coverage=shared_coverage,
        rejected=rejected,tradeoff_flags=tradeoffs,guards=dict(guards),
        interpretation="predictive_gain_with_tradeoff" if tradeoffs[selected] else "paired_inner_ranking_only")


def nested_grouped(*args,policy,**kwargs):
    from .selection import nested_grouped as coordinator
    if "guards" in kwargs or "selector" in kwargs:
        raise ValueError("Use the explicit non-threshold policy")
    return coordinator(*args,guards=policy,selector=choose,**kwargs)
