# Current status

## Scientific position

The estimator is an exploratory geometric model. No accommodation exponent has been confirmed, selected for deployment, or validated against independent physiological measurements. Keep the frozen quadratic estimator as the baseline.

## Latest literal-power study

The response is `(A / 1 D)^n`. All 10 coarse calibrations passed certification. On 1,134 common complete frames across 20 exposures, `n=3` had the lowest P4 RMS: 3.0034 px, 18.46% below the matched log control at 3.6836 px. Gaze disagreement increased from 0.07587 to 0.12834 degrees; accommodation disagreement increased from 0.02301 to 0.20055 D. `n=3` was the upper tested boundary, so the study did not establish an optimum. The user stopped the run during confirmation. Lower-bound sensitivity and full evaluation did not run.

See the [coarse comparison](../experiments/full_position/literal_power_search_v1/RESULTS.md), [approved plan](LITERAL_POWER_PLAN.md), and [validation record](../experiments/full_position/literal_power_validation/validation.json). The coarse frame records, frozen schedules, and fitted models remain available. Solver debug archives and the incomplete confirmation checkpoint were removed.

## Latest retained full calibration comparison

Three certified laws were compared on 89,172 common complete frames from all 20 exposures. The log response ranked first on all four reported metrics. The comparison measures internal agreement, not independent accuracy.

| Law | P4 RMS (px) | Gaze disagreement (deg) | Accommodation disagreement (D) | Worst-point RMS (px) |
|---|---:|---:|---:|---:|
| Log | 4.7929 | 0.2519 | 0.03583 | 6.2983 |
| Square root | 7.3190 | 0.8895 | 0.10538 | 9.9985 |
| Quadratic | 7.6118 | 0.4833 | 0.22071 | 10.5096 |

See the [three-law report](../experiments/full_position/accommodation_full_v4_gpu/THREE_MODEL_COMPARISON.md). The interrupted v4 run's bulky frame, holdout, inverse, and linear-failure archives were removed; certified model artifacts, completion records, and the frozen schedule metadata remain.

## Other retained results

- Superseded shifted-response outputs and old parity-run folders were removed. Only the fitted model inputs needed by contract tests remain in `tests/fixtures/response_inverse/`.
- Older grouped-development and audit-phase result trees were removed as intermediate research output. Core theory, code, test contracts, baseline data, and current response-law results remain.
- The frozen source data, fixation intervals, and quadratic model remain in `data/` and `models/`.

## Reproduction starting point

Read [Theory.md](Theory.md), [ESTIMATOR_PLAN.md](ESTIMATOR_PLAN.md), and [LITERAL_POWER_PLAN.md](LITERAL_POWER_PLAN.md). Code is in `full_position/`; contract and numerical checks are in `tests/`. GPU and CPU parity tests use compact saved model and training-input fixtures under `tests/fixtures/gpu_inverse_parity/`.
