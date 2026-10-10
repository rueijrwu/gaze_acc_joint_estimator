# Subplan 2 — Establish the P1 reference and shared nuisance scale

**Covers:** S2/G2. **Status:** NOT_RUN.  
**Parent:** [Gate policy](../STAGE_GATES.md). **Requires:** usable G0/G1 evidence.  
**Equations:** [Theory](../Theory.md), P1-reference scale and separate optical zeros.

## Question

Can one accommodation-independent, gaze-conditioned P1 reference describe the observed P1 geometry well enough to supply the nuisance scale used by the next stage? This gate does not prove that P4 has exactly the same axial response.

## Smallest experiment

First specify a symmetry diagnostic from the actual source correspondence and fixed image axes. Inspect its distribution and contiguous-block summaries at all five nominal gazes. Initialize omega1 from supported P1 evidence, not from nominal zero or an invented equilateral triangle. Keep the scores and alternative references when the minimum is broad.

Build one empirical P1 reference at its native optical zero from many frames. Preserve its existing radial distortion. Fix its length, origin, image-axis and isotropic gaze-scale conventions. Compare the fixed-reference initialization with the minimal K1 distortion map using the same input rows and a declared provisional gaze snapshot. Fit only the supported P1 block; no A coefficient or unrestricted affine map is introduced.

For each frame and trial visual gaze, compute the reference edges a1(theta-omega1) and the existing P1-only positive GLS scale:

`g = (a1^T W1 e1) / (a1^T W1 a1)`.

The same g will be applied to both reflections. Save the edge residual `e1 - g*a1`; this residual, not the returned g alone, shows how well a scalar matches the pattern. Propagate shared edge noise in the declared metric. A nonpositive g is an explicit unusable outcome, not a value to clip silently.

## Evidence to inspect

Use three small views: independent P1 symmetry scores versus visual gaze; g and P1 edge residuals by gaze/capture and time; measured versus predicted relative P1 patterns at representative conditions. Quantify scale-parallel and remaining directional discrepancy when useful, without counting them as extra measurements.

Compare within-capture time blocks and matched nominal gaze conditions across captures. An apparent demand dependence is also a capture/pose/bootstrap dependence in these data: do not call it a proven P1 accommodation effect. In particular, transferred centroid gaze is still approximate. Record which pattern comparisons use nominal gaze bins and which use provisional inferred gaze.

Check recovery of known synthetic scale, invariance under common translation and a changed edge origin with correctly transformed covariance, and analytic/autodiff scale derivatives. A P4 perturbation or A change at fixed theta/P1 must not directly change g. Weak P1 deformation is not an instruction to recover gaze from P1 alone.

## G2 decision

**GO:** a declared P1 reference and scale convention yield valid positive scales and interpretable relative-shape residuals over the initialization population, without an unresolved bookkeeping contradiction. The same calibrated P1 rule is used across captures, and its derivatives pass the numerical checks.

**GO_WITH_LIMIT:** omega1 or part of its distortion is weakly identified, but a supported fixed operational reference gives usable scale. Keep that parameter fixed or retain a bounded reference alternative. Mark scale as effective reference magnification, not measured Z, and assign the uncertainty to G5/G7/G8 for reassessment after better gaze is available. Do not require a sharp symmetry minimum.

**REPAIR:** double subtraction of omega1, wrong template units, incorrect scale/projective order, covariance errors, or a fitted per-frame affine/P4-dependent normalization. Repair this before proceeding.

**PAUSE:** incompatible P1 shapes cannot be represented under the declared fixed conventions and current gaze uncertainty, or no supported reference/positive scale is available for required conditions. First test one diagnosed reference or bootstrap issue. Do not add free capture scales or a P1 accommodation term to make the error disappear.

The magnitude of P1 residuals is reported in original units; no 1-micrometer or pixel-RMS cutoff is imposed. If K1 does not improve an already adequate initialization detectably, freeze unsupported coefficients rather than expand them; preserve the K1 interface and the prescribed optical-zero distinction.

## Saved result and next action

Save the chosen P1 template/omega1, fixed/free roster, scale-policy/covariance hashes, per-frame g/status, and enough edge residuals to reproduce the report. Append G2's decision and deferred checks to the stage ledger.

**Next:** G3 independently determines P4's reference using this scale. A useful P1 g does not yet certify common P1/P4 scaling, physical eye distance, or the final accommodation response. No full GPU calibration or second backend is needed to inspect this stage.
