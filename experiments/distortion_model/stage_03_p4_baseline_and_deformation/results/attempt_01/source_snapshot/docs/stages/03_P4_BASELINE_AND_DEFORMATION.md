# Subplan 3 — Separate P4 reference, accommodation response, and gaze deformation

**Covers:** S3/G3, S4/G4, S5/G5. **Status:** NOT_RUN.  
**Requires:** a usable G2 scale/reference, with all provisional assumptions carried forward.  
**Parent:** [Gate policy](../STAGE_GATES.md). Use [Theory](../Theory.md), not new equations or free per-frame optical parameters.

## S3 / G3 — Is there a supported P4 optical reference?

**Freeze:** P1 reference/scale policy, source geometry, demand labels and provisional visual gazes. Inspect scale-corrected P4 symmetry at all five gazes across captures. Use the real source pairing; a scale-invariant symmetry diagnostic is allowed for reference selection but is not P4 normalization in inference.

**Output:** per-demand symmetry curves with block-level variability, selected omega4 and alternatives, Delta14=omega4-omega1, and native-angle support. Test xi1=xi4+Delta14 and the native identities. The angular zero does not establish a measured spatial radial center.

**GO:** P4 has its own supported reference and the conversion to P1/visual angles is correct. It may differ substantially from both other zeros.

**GO_WITH_LIMIT:** the minimum is broad or an endpoint wins. Use a declared supported operational reference and a bounded alternative; label the physical symmetry zero unresolved and assign recheck to G5/G7. Do not claim a zero outside the observed range, and do not set every frame in a selected fixation to xi4=0.

**REPAIR:** inherited omega1, accidental visual-zero substitution, double offset, or symmetry manufactured by free affine alignment.

**PAUSE:** no defensible reference can be specified for the chosen symmetry-constrained family. A coherent shift by demand/capture must be recorded; first inspect scale and gaze-bootstrap uncertainty, not add separate capture zeros. With no near-reference frames, do not invent interpolated observations; record the limitation and decide on one explicit alternate initialization.

**Next:** initialize DM0's accommodation baseline, not fit all P4 parameters at once.

## S4 / G4 — Does corrected P4 geometry contain a usable accommodation-related response?

**Freeze:** the G3 reference convention and provisional near-reference gaze; do not free radial centers, K4, or unrelated per-frame k values. Start with `DM0-M1`, using the real template b4ref, `M(Aref)=1`, and `M=1+m1*(A-Aref)/(1 D)`.

Inspect a translation-free P4 shape at near-reference gaze after dividing by P1 g. A descriptive effective scale can be fitted against the centered template using a scalar projection; this is an empirical diagnostic, not a second nuisance scale used to normalize P4. Preserve the remaining directional residual. Full-P4 centroids are allowed in this calibration diagnostic, not in a subsequent omitted-P4 solve.

Initialize M and framewise A under soft fixation-mean demand anchors. Report the directly measured scale/shape summaries against the recorded demand labels before showing curves against fitted A. Fitting A and then drawing a perfect M(A) curve is not independent evidence. Each demand is tied to a capture, so demand-linked differences are only consistent with accommodation; capture effects remain an alternative.

**Output:** measured corrected shape/effective-scale distributions by demand and gaze window; the provisional M(A) curve; coefficient and frame-state status; residuals before/after M; effects of the declared reference and A anchors. Do not compare simulated absolute kappa values with these empirical increments without matching geometry.

**GO:** an identifiable effective response with a coherent sign/shape supplies usable accommodation starts, and it is not merely a reference-unit change or all-bound solution. This is initialization evidence, not a full linear-law validation.

**GO_WITH_LIMIT:** response is weak or contaminated by unmodeled off-axis deformation, but starts can be represented without inventing data or overconstraining A. Carry the ambiguity to G5/G7. A near-zero M slope makes accommodation from shape unresolved; it must remain labeled unresolved, not certified by the demand anchors.

**REPAIR:** double radial correction, P4 self-normalization, arbitrary per-frame M/k/A combination, or fixing all A values to demand.

**PAUSE:** no resolvable accommodation-related shape information remains after the specific reference/scale checks, and the proposed initialization cannot support subsequent state inference. Do not search many exponents to manufacture a signal. A limited null-model diagnostic can still be reported with no accommodation claim.

**Next:** G5 fits rotation response with M initialized. Radial identifiability is diagnostic here; it is not a prerequisite for DM0 and does not authorize DM1 yet.

## S5 / G5 — Does the proposed rotation map explain the remaining gaze-dependent shape?

**Free block:** minimal K4 coefficients and only permitted global-reference refinements. **Freeze:** initial M family, reference length/origin, shared P1-scale policy and candidate population. K4 uses xi4, not xi1, and preserves identity at its own optical zero for every A.

Fit the composed P4 pattern across all five gazes and all demand conditions. Start from A-independent keystone coefficients, while retaining A inside the baseline and the projective denominator. A centered-shape diagnostic must use the centered *model prediction*; do not inverse-keystone a measured centroid as if it were the optical center.

Revisit G4's frames using individual xi4, then update M under the same model. Compare K4-fixed-identity and minimal K4 on the same rows as a controlled component diagnostic; both remain hypotheses until full calibration. Examine residual directions by source, axis, gaze and capture, not only the fitted parameter curves.

**Output:** complete predicted patterns, scale/keystone contributions, denominator/domain checks, state/global derivative checks, and a list of G3/G4 provisional findings resolved or still open.

**GO:** the composed model is well-defined over the required reference/domain and supports the next joint calculation without an unexplained reference/sign error. Remaining residuals can be tested in the full DM0 fit; they need not vanish.

**GO_WITH_LIMIT:** a distortion coefficient is weakly determined or systematic mismatch remains. Fix unsupported freedom and name the residual signature to test at G7/G8. Freeze, do not inflate, uncertainty merely to hide it.

**REPAIR:** wrong transformation order, incorrect native angle, missed baseline or mean derivative, invalid denominator, or different hidden scale in P4.

**PAUSE:** a required pattern change is structurally outside the specified family and prevents usable joint inference after one targeted check. Report the failing geometry. Raising the gaze degree cannot repair a spatial deformation the map cannot represent. Any new spatial/coupling term requires the later controlled-change decision, not silent expansion here.

**Next:** G6 corrected-center polynomial. This gate does not prove common axial P1/P4 behavior; residual dependence on P1 g is carried into the final empirical compatibility review.
