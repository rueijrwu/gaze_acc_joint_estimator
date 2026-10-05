#!/usr/bin/env bash
# Anchor-scale x curvature grid for the corrected-label robust calibration.
# Usage: exp2/corrected_labels/grid/run_grid.sh [config ...]   (default: all four)
# Config names: a<anchor_deg>_c<0|1>.  a1.0_c0 reproduces the existing baseline settings.
set -euo pipefail
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
EXP="$REPO/exp2/reduced_calibration/experiment.py"
OVERRIDES="$REPO/exp2/corrected_labels/target_overrides_v1.json"
CONFIGS=("$@"); [ ${#CONFIGS[@]} -gt 0 ] || CONFIGS=(a1.0_c0 a1.0_c1 a0.1_c0 a0.1_c1)

# Same solver settings as the existing corrected-label runs (reduced_calibration/README.md).
SOLVER=(--max-nfev 400 --wall-seconds 600 --lsmr-maxiter 300 --lsmr-atol 1e-9 --lsmr-btol 1e-9
        --robust-outer 16 --robust-max-nfev 25)

for cfg in "${CONFIGS[@]}"; do
  anchor="${cfg#a}"; anchor="${anchor%%_c*}"; curv="${cfg##*_c}"
  KNOBS=(--theta-anchor-scale-deg "$anchor"); [ "$curv" = 1 ] && KNOBS+=(--curvature)
  OUT="$HERE/$cfg"; mkdir -p "$OUT/logs"
  for fold in holdout3 full; do
    [ -d "$OUT/training/$fold" ] || python3 -B "$EXP" train --fold "$fold" --target-overrides "$OVERRIDES" \
        "${SOLVER[@]}" "${KNOBS[@]}" --output-dir "$OUT/training/$fold" > "$OUT/logs/${fold}_train.log" 2>&1
  done
  # holdout3 model: anchor-free inverse on fixations 10-14
  [ -d "$OUT/predictions/holdout3" ] || python3 -B "$EXP" estimate --fold holdout3 --target-overrides "$OVERRIDES" \
      --model-dir "$OUT/training/holdout3/robust" --output-dir "$OUT/predictions/holdout3" > "$OUT/logs/estimate.log" 2>&1
  # full model through the SAME inverse on 10-14 (writes fresh_matched_full_fixed_inverse.csv)
  [ -f "$OUT/predictions/holdout3/comparison_summary.json" ] || python3 -B "$EXP" compare --fold holdout3 \
      --prediction-dir "$OUT/predictions/holdout3" --matched-dir "$OUT/training/full/robust" --no-legacy > "$OUT/logs/compare.log" 2>&1
done
python3 -B "$HERE/summarize_grid.py"
