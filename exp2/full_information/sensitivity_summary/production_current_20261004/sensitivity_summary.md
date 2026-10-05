# Anchor and initialization sensitivity

Both nominal anchors change together; the source observations and function prior remain fixed. State changes are local sensitivity, not accuracy errors.

| Case | Converged | Optical cost | Anchor cost | Gaze mean change RMS (deg) | Accommodation mean change RMS (D) |
|---|---|---:|---:|---:|---:|
| H0_anchor0p5 | True | 0.000596805 | 0.330884 | 0.000906184 | 0.00050685 |
| H0_anchor2 | False | 0.00951716 | 1.30926 | 0.00177407 | 0.00101181 |
| H0_perturbed_seed20261004 | True | 0.00238064 | 0.659389 | 0.000304104 | 0.00012127 |
