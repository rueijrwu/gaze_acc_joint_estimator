"""Export one auditable calibration/training-diagnostic row per task.

Run from the repository root with:
    rtk proxy python experiments/full_position/joint_sensitivity_v1/export_diagnostics.py

The exporter reads only the saved summary and preserves failed/unaccepted rows.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
FIELDS = (
    "variant", "fold", "model", "calibration_converged", "evaluation_complete",
    "accepted", "selected_start", "initialization", "solver_status", "solver_message",
    "acceptance_reason", "selected_nfev", "selected_cost", "projected_stationarity_physical",
    "inner_stationarity_scaled", "optical_cost", "anchor_cost", "coefficient_prior_cost",
    "training_equal_fixation_point_vector_rms_px", "training_gaze_anchor_discrepancy_rms_deg",
    "training_accommodation_demand_discrepancy_rms_D", "training_state_shift_rms_theta_deg",
    "training_state_shift_rms_A_D",
)


def main() -> None:
    summary = json.loads((HERE / "summary.json").read_text())
    rows = []
    for task, completion in sorted(summary["calibrations"].items()):
        variant, fold, model = task.split("/", 2)
        calibration = completion.get("calibration") or {}
        selected_start = calibration.get("selected_start")
        alternatives = calibration.get("alternatives") or []
        selected = next((item for item in alternatives
                         if item.get("start") == selected_start), {})
        components = selected.get("objective_components") or {}
        training = completion.get("training_diagnostics") or {}
        row = {
            "variant": variant,
            "fold": fold,
            "model": model,
            "calibration_converged": completion.get("calibration_converged"),
            "evaluation_complete": completion.get("evaluation_complete"),
            "accepted": bool(completion.get("calibration_converged")
                              and completion.get("evaluation_complete")),
            "selected_start": selected_start,
            "initialization": selected.get("initialization"),
            "solver_status": selected.get("status"),
            "solver_message": selected.get("message"),
            "acceptance_reason": selected.get("acceptance_reason"),
            "selected_nfev": selected.get("nfev"),
            "selected_cost": selected.get("cost"),
            "projected_stationarity_physical": selected.get("projected_stationarity_physical"),
            "inner_stationarity_scaled": selected.get("inner_stationarity_scaled"),
            "optical_cost": components.get("optical"),
            "anchor_cost": components.get("anchor"),
            "coefficient_prior_cost": components.get("prior"),
            "training_equal_fixation_point_vector_rms_px": training.get("training_equal_fixation_point_vector_rms_px"),
            "training_gaze_anchor_discrepancy_rms_deg": (training.get("training_fixation_mean_gaze_anchor_discrepancy_deg") or {}).get("rms"),
            "training_accommodation_demand_discrepancy_rms_D": (training.get("training_fixation_mean_accommodation_demand_discrepancy_D") or {}).get("rms"),
            "training_state_shift_rms_theta_deg": (training.get("training_state_shift_theta_deg") or {}).get("rms"),
            "training_state_shift_rms_A_D": (training.get("training_state_shift_A_D") or {}).get("rms"),
        }
        rows.append(row)

    with (HERE / "calibration_diagnostics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
