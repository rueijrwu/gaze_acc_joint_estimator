# G0 / attempt 01 / bootstrap_only

Status: INTERRUPTED. Decision: REPAIR.
Question: Can reviewed identities be loaded before inference?
Evidence kind: synthetic numerical checks plus implementation failure; no empirical calibration.
The 16 contracts in [tests.json](tests.json) passed, but the experiment failed with
`KeyError: intervals`. The actual reviewed schema uses `fixations`.
No input population, starts, or calibration was produced. Source/config hashes are in
[provenance.json](provenance.json). No usable checkpoint exists.
One next action: repair that loader key, add a full reviewed-schema contract, and rerun G0/G1
in a new attempt. Reviewer: Codex implementation audit, 2026-10-09.
