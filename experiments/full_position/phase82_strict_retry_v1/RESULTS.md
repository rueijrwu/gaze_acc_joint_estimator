# Strict continuation retry

The previously uncertified `prior_weak/capture_1/conditional27` training start was continued from its saved training trajectory. Its original selected checkpoint had physical projected stationarity **0.0031837353924304423** and objective **1468.772727271089**. After **12 preconditioned continuation evaluations** on the preserved start, physical projected stationarity fell to **0.0009979299425033616** and objective to **1468.7727227437979**. This meets the unchanged strict gates: physical projected stationarity `<0.001` and inner coefficient stationarity `<1e-7` (observed **2.3956e-12**). The saved solver success alone was not used as acceptance; the existing stationarity and stable-step certificate passed.

The retry used the 630 rows in the capture_1 training split and did not read evaluation rows. `verify_results.py` independently rebuilt the covariance and profiled objective from the saved training states. Coefficients, total cost, objective components, physical stationarity, and inner stationarity matched at saved precision. All 29 implementation/design source hashes matched the retry manifest.

The original failed fit remains unchanged at `experiments/full_position/joint_sensitivity_v1/variants/prior_weak/capture_1/conditional27`; its state, completion, and candidate archive hashes are recorded in `verification.json`. Four start/stage checkpoint rows remain in `calibration_candidates.jsonl.gz`.

The paired strong-anchor transition report is maintained separately in `../phase82_audit_transitions_v1/RESULTS.md`.
