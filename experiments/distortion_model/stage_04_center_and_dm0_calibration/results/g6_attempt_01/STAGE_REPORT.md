# G6 / attempt 01 — retained initial execution

Status: **COMPLETE**. Decision: **REPAIR**, reporting only. Reviewer: root implementation/audit agent, 2026-10-10 02:28:53 UTC.

The empirical center block solved correctly: 89,175 complete frames evaluated, 10,915 unavailable rows retained; coefficient rank10, scaled gradient6.49e−11. Fixed-state full objective54,759.434869→2,134.091666; relative-coordinate RMS72.070822→4.560659 native px. [Independent reconstruction](reproduction_checks.json) verifies the actual weighted solve, objective and bookkeeping. Full calibration and cross-check flags remain false.

Root visual audit found exposure colors cycled while legend labels described capture/demand, making the grouping ambiguous. The export needed explicit model revision/authority/reference provenance fields. One reporting repair produced [attempt02](../g6_attempt_02/STAGE_REPORT.md), with unchanged optics, data, numerical method, metric, prior and states. Every fitted array is bitwise identical, and fit/condition records are exactly equal.

This attempt remains preserved with original source/config/numerical bytes, [provenance](provenance.json), arrays, plots and console log. It is superseded by attempt02 for reporting and handoff.

One next action: use the reviewed [attempt02 checkpoint](../g6_attempt_02/checkpoint.json) to initialize G7. Retry scope: reporting repair complete. See its audit for limitations and analytical cases. No automated tests were added or run.
