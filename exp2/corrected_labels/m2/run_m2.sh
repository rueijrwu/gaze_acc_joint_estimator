#!/usr/bin/env bash
# Corrected-label model comparison: piecewise (M0) vs m2 with theta anchor scale 1.0 / 0.1 deg.
#   usage: exp2/corrected_labels/m2/run_m2.sh [CONFIG ...]     (default: M0 M2a1 M2a01)
#   env:   WALL_SECONDS (per-stage training wall budget, default 1800), PARALLEL=1 (run configs concurrently)
# Per config <C> everything lands in exp2/corrected_labels/m2/<C>/:
#   training/{holdout3,full}/{quadratic,robust}   predictions/holdout3/   logs/
# Training/estimate/compare options replicate the existing corrected-label runs
# (max-nfev 400, lsmr 300 / 1e-9, robust kappa 2, estimate batch 512 / 100 inverse iterations).
# The full-model robust coefficients are pushed through the SAME anchor-free inverse on fixations 10-14
# by `compare --matched-dir training/full/robust` -> predictions/holdout3/fresh_matched_full_fixed_inverse.csv
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # repository root
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_THREAD_LIMIT=1
BASE=exp2/corrected_labels/m2
OVERRIDES=exp2/corrected_labels/target_overrides_v1.json
EXPERIMENT=exp2/reduced_calibration/experiment.py
WALL_SECONDS="${WALL_SECONDS:-1800}"

# config -> "model theta_anchor_scale_deg"
declare -A CONFIG=( [M0]="piecewise 1.0" [M2a1]="m2 1.0" [M2a01]="m2 0.1" )

step() {  # step <config> <name> <command...>: logs stdout+stderr and wall time; skips completed steps
  local cfg="$1" name="$2"; shift 2
  local logs="$BASE/$cfg/logs"; mkdir -p "$logs"
  if [[ -e "$logs/$name.done" ]]; then echo "[$cfg] $name: already done"; return; fi
  echo "[$cfg] $name: $*" | tee "$logs/$name.cmd"
  local t0=$SECONDS
  "$@" > "$logs/$name.log" 2>&1
  printf '%s\t%s\n' "$name" "$((SECONDS-t0))" >> "$logs/wall_seconds.tsv"
  echo "$((SECONDS-t0))" > "$logs/$name.done"
  echo "[$cfg] $name: $((SECONDS-t0)) s"
}

run_config() {
  local cfg="$1"; read -r model anchor <<< "${CONFIG[$cfg]}"
  local out="$BASE/$cfg"
  local train=(python3 -B "$EXPERIMENT" train --model "$model" --theta-anchor-scale-deg "$anchor"
               --target-overrides "$OVERRIDES" --max-nfev 400 --wall-seconds "$WALL_SECONDS"
               --lsmr-maxiter 300 --lsmr-atol 1e-9 --lsmr-btol 1e-9)
  step "$cfg" train_holdout3 "${train[@]}" --fold holdout3 --output-dir "$out/training/holdout3"
  step "$cfg" train_full     "${train[@]}" --fold full     --output-dir "$out/training/full"
  step "$cfg" estimate_holdout3 python3 -B "$EXPERIMENT" estimate --fold holdout3 --model "$model" \
       --target-overrides "$OVERRIDES" --model-dir "$out/training/holdout3/robust" --output-dir "$out/predictions/holdout3"
  step "$cfg" compare_holdout3 python3 -B "$EXPERIMENT" compare --fold holdout3 --model "$model" \
       --prediction-dir "$out/predictions/holdout3" --matched-dir "$out/training/full/robust" --no-legacy
  python3 -B "$BASE/check_m2.py" --skip-c --model-json "$out/training/holdout3/robust/model.json" \
       "$out/training/full/robust/model.json" --json-out "$out/logs/check_monotonicity.json" \
       > "$out/logs/check_monotonicity.log" 2>&1 || echo "[$cfg] WARNING: monotonicity check failed (see logs)"
}

configs=("$@"); [[ ${#configs[@]} -gt 0 ]] || configs=(M0 M2a1 M2a01)
for cfg in "${configs[@]}"; do [[ -n "${CONFIG[$cfg]:-}" ]] || { echo "unknown config $cfg" >&2; exit 2; }; done
if [[ "${PARALLEL:-0}" == 1 ]]; then
  pids=(); for cfg in "${configs[@]}"; do run_config "$cfg" & pids+=($!); done
  rc=0; for pid in "${pids[@]}"; do wait "$pid" || rc=1; done; exit $rc
else
  for cfg in "${configs[@]}"; do run_config "$cfg"; done
fi
