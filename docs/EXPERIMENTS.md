# Retained experiment evidence

Only the latest full accommodation calibration comparison and the literal-power search/validation are kept as accommodation experiment results. Older accommodation result trees were removed.

| Study | Retained evidence |
|---|---|
| [`literal_power_search_v1`](../experiments/full_position/literal_power_search_v1/RESULTS.md) | Ten certified coarse `A^n` fits, common-frame scores, frame records, accepted model artifacts and frozen schedules. Confirmation stopped before sensitivity/full evaluation. |
| [`literal_power_validation`](../experiments/full_position/literal_power_validation/validation.json) | Solver implementation parity and validation evidence for the literal-power study. |
| [`accommodation_full_v4_gpu`](../experiments/full_position/accommodation_full_v4_gpu/THREE_MODEL_COMPARISON.md) | Latest retained full calibration comparison: log, square-root, and quadratic laws on a common cohort, with certified models and schedule metadata. |
| [`gpu_vectorization_validation`](../experiments/full_position/gpu_vectorization_validation/) | Solver timing and implementation parity evidence. |

The small model and training-array files in `tests/fixtures/` support deterministic inverse-parity contracts. They are test fixtures, not retained experiment outputs. All comparisons are development or internal-agreement evidence; they do not establish independent physiological accuracy or select a deployment model.
