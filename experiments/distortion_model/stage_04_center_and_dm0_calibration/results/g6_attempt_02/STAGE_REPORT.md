# G6 / attempt 02 — corrected centers and forward D polynomial

Status: **COMPLETE**. Decision: **GO_WITH_LIMIT**. Scientific disposition: `protocol_ready; full_model_support_unresolved`. Evidence kind: empirical conditional center initialization, analytical cases and independent reconstruction. Reviewer: root implementation/audit agent, 2026-10-10 02:28:53 UTC.

## Provenance and retry scope

Source commit `efd88bd3b326653aedc3b383143ef10856d22092`; remote matched at execution. Dirty-source hash `acb145815227c40fe85ad881e824d9826661e30d668260f260f658943445fa21`. [provenance.json](provenance.json) saves theory/plan/config hashes, runtime, parent bytes and archived source hashes. [source_snapshot](source_snapshot/) preserves source bytes; [config.json](config.json) declares policy before fitting.

Parent: reviewed [G5 GO_WITH_LIMIT](../../../stage_03_p4_baseline_and_deformation/results/g5_attempt_01/STAGE_REPORT.md). Its chain pins G5 states/model, G3 independent zeros, G2 P1 reference/scale and G0 measurements/population/covariance/slots. Raw sources are freshly hash-checked and matched to the schedule. Schema `distortion_dual_alignment_v1` records basis, units, references, revision, fixed/free roster and separate bootstrap inverse.

[Attempt01](../g6_attempt_01/STAGE_REPORT.md) is retained with a reporting REPAIR: exposure colors cycled while legend entries described capture/demand, and the model needed explicit revision/reference provenance fields. Attempt02 fixes that reporting issue. All fitted arrays are bitwise identical; `fit.json` and `conditions.json` are exactly equal. Optics, states, population, metric, prior and coefficients are unchanged. No scientific alternative was tried.

## Question and parameter roster

Are optical mean shifts and relative center separation accounted for once, and can the degree-2 forward D block be solved against the complete point criterion?

Fix all individual G5 theta/A states, G2 P1 g/reference, G3 optical zeros, empirical template length/origin gauges and axes, and G5 DM0-M1/K4. Fit ten global D coefficients: image x/y components of b0, bA, s0, sA and c2.

```text
t = theta_visual_deg/10; a = (A-Aref)/1D
D = b0 + bA*a + (s0+sA*a)*t + c2*t^2
Aref = 0.36036036036 D
omega1 = -10.02090411159 degrees
omega4 = -10.01697413285 degrees
Delta14 = omega4-omega1 = +0.00392997874 degrees
```

No zero intercept is imposed. c3=0; higher powers have no A coupling. Dy is an image response to the one gaze state. No independent frame centers, h_cent polynomial, nuisance P4 scale or radial term are fitted.

```text
Dobs = (c4-c1)/g - mu4 + mu1
C1_hat = c1-g*mu1; C4_hat = c4-g*mu4
centroid_prediction_native_px = g*(D+mu4-mu1)
```

The sign remains P4-minus-P1. Dobs is a derived diagnostic in reference pixels. The inferred origins are model-informed; they are not independently measured physical positions or additional observations.

## Complete criterion and coefficient solve

Use four P1 edge components and three P4-minus-P1-centroid point pairs with the original full raw covariance R. Keep P1/P4 correlations and P1-only substituted g. This criterion does not invert the singular plug-in residual covariance or have calibrated chi-square meaning.

E inserts one image pair into all three P4 coordinate pairs, with zeros in the four P1 rows. Let phi=`[1,a,t,a*t,t^2]`, base=`g*[P1_edges,F4-mu1]`, r0=y-base and w_i=1/(20*N_exposure_i). Then

```text
B_i = g_i*(phi_i^T tensor E)
H_data = sum_i w_i*g_i^2*(phi_i*phi_i^T tensor E^T*R^-1*E)
rhs = sum_i w_i*g_i*vec(phi_i*(E^T*R^-1*r0_i)^T)
(H_data + prior_diagonal)*coefficient_vector = rhs
```

Chunks use full exposure counts. The RHS contains all point residual components and covariance cross-terms. There is no second Dobs loss. Weighted moments produce a 10×10 coefficient Hessian, avoiding a dense empirical per-frame global Jacobian.

Predeclared prior: `.5*sum((c2/190.3026851583 reference_px)^2)` using the fixed empirical P4 RMS radius; other center priors are zero. This radius also defines coefficient-gradient units. It is not an accuracy threshold. Full-fixation mean scales .10 degree/.25 D are finite penalties and remain constant during this solve. Temporal-flatness penalty is zero.

## Measured coefficients and objective

| Coefficient | Image x (reference px) | Image y (reference px) |
|---|---:|---:|
| b0 | 49.01847188 | 51.13192592 |
| bA | .43620768 | .72080895 |
| s0 | 205.47366454 | 8.95048355 |
| sA | .28332891 | −.64960427 |
| c2 | 1.20232509 | .13273655 |

Coefficients multiply the dimensionless basis above. At Aref the dominant visual slopes are 20.5474 reference px/degree in x and .89505 in y. This conditional response remains dependent on the empirical gauge and transferred visual gaze.

| Full fixed-state J component | Constant-D initializer | Degree-2 D |
|---|---:|---:|
| Complete point term | 54,636.352525 | 2,011.009301 |
| Full-mean theta anchor | 122.644527 | 122.644527 |
| Full-mean A anchor | .437818 | .437818 |
| Center regularization | 0 | .000020202 |
| Temporal term | 0 | 0 |
| Total | 54,759.434869 | 2,134.091666 |
| Equal-exposure relative-coordinate RMS (native px) | 72.070822 | 4.560659 |

The before initializer solves the constant pair in the same degree-2 family and complete criterion, with other coefficients zero. Its large error makes the decrease expected after adding the declared gaze response. Acceptance rests on consistency, rank, stationarity and accounting. The decrease does not validate optical accuracy; these values are not comparable to G5's four-edge marginal cost.

After-fit centroid-coordinate RMS is **4.894303 native px**; descriptive P1 centered point RMS .989170 px is unchanged. [Corrected centers](components.png), [mean corrections](optical_mean_corrections.png) and [native centroid residuals](residuals.png) use display stride20 only; fitting/reporting use all rows.

## Population and limitations

100,090 scheduled rows; **89,175 complete and evaluated**; 10,915 unavailable original inputs; zero unresolved valid evaluations. All twenty full intervals and 300,270 expected omission slots remain. Every theta/A/g value is unchanged from its upstream snapshot.

[fitted.npz](fitted.npz) saves full-row status, states/g, optics/means, raw centroid separation, inferred origins, Dobs/D prediction and native relative prediction/residual arrays. [conditions.json](conditions.json) saves all twenty exposure records and one hundred contiguous original-row blocks, including unavailable counts and signed means. Five equally spaced original rows per exposure form the representative ledger, retaining unavailable representatives explicitly.

| Exposure | Capture | Target θ (°) | Scheduled | Valid | Unavailable | Relative RMS (px) | Centroid RMS (px) | Mean θ (°) | Mean A (D) | Signed centroid x/y means (px) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | capture_1 | -10.0 | 4800 | 4433 | 367 | 1.802559 | 2.278536 | -10.016974 | 0.362428 | +0.291926, -1.972238 |
| 1 | capture_1 | -5.0 | 4800 | 4800 | 0 | 2.102355 | 2.330630 | -5.014899 | 0.577500 | +2.048481, -2.459163 |
| 2 | capture_1 | 0.0 | 4900 | 4900 | 0 | 3.505706 | 3.904796 | -0.002070 | 0.336148 | +3.333216, -3.981434 |
| 3 | capture_1 | 5.0 | 5000 | 4993 | 7 | 4.489594 | 4.002944 | 5.121306 | 0.230636 | +3.970921, -3.684894 |
| 4 | capture_1 | 10.0 | 5200 | 5154 | 46 | 4.762243 | 3.857070 | 9.912637 | 0.774177 | +3.677786, -3.865107 |
| 5 | capture_2 | -10.0 | 4600 | 3912 | 688 | 12.181809 | 14.756648 | -9.866688 | 3.483936 | -0.129272, -0.117119 |
| 6 | capture_2 | -5.0 | 5000 | 4345 | 655 | 7.412486 | 8.909547 | -5.992736 | 3.901710 | +0.867842, -2.493265 |
| 7 | capture_2 | 0.0 | 5000 | 4553 | 447 | 2.742647 | 3.053102 | -2.068178 | 3.856487 | +1.587686, -3.828364 |
| 8 | capture_2 | 5.0 | 5000 | 4781 | 219 | 3.370999 | 2.603208 | 1.943447 | 4.101407 | +1.815407, -3.069326 |
| 9 | capture_2 | 10.0 | 5290 | 4435 | 855 | 3.757675 | 3.112626 | 6.115163 | 3.721870 | +1.957093, -3.690591 |
| 10 | capture_3 | -10.0 | 4700 | 4505 | 195 | 1.384530 | 1.437004 | -9.777954 | 2.891977 | -0.007456, +1.605318 |
| 11 | capture_3 | -5.0 | 4900 | 3902 | 998 | 1.895908 | 1.239292 | -5.585879 | 3.579236 | +1.053654, -0.690623 |
| 12 | capture_3 | 0.0 | 5000 | 4811 | 189 | 3.059854 | 2.251983 | -1.518658 | 3.202161 | +1.912481, -2.374111 |
| 13 | capture_3 | 5.0 | 5000 | 4251 | 749 | 3.516845 | 3.361019 | 2.912341 | 3.133146 | +2.326839, -4.007061 |
| 14 | capture_3 | 10.0 | 5500 | 2326 | 3174 | 3.657305 | 2.966810 | 7.124119 | 2.855078 | +2.432438, -3.107582 |
| 15 | capture_4 | -10.0 | 4800 | 3914 | 886 | 1.668938 | 1.676304 | -9.539150 | 1.844865 | +0.247072, +0.545651 |
| 16 | capture_4 | -5.0 | 4900 | 4899 | 1 | 1.568761 | 1.355681 | -5.086136 | 2.114046 | +1.582054, +0.247167 |
| 17 | capture_4 | 0.0 | 5000 | 4747 | 253 | 3.297969 | 3.473268 | -0.638230 | 2.054669 | +2.499575, -3.560505 |
| 18 | capture_4 | 5.0 | 4900 | 4738 | 162 | 4.026005 | 3.753713 | 3.967671 | 2.081460 | +2.918809, -4.258177 |
| 19 | capture_4 | 10.0 | 5800 | 4776 | 1024 | 6.625749 | 6.309890 | 8.483731 | 1.967852 | +2.932617, -5.479725 |

Optical mean correction is mainly in y, reaching −4.020835 reference px at capture1 +10°. Structured residuals remain: capture4 +10° centroid mean biases are approximately [+2.933,−5.480] native px. Capture2 −10° whole-interval centroid RMS is **14.756648 px**; extreme transition tails remain. Small mean bias does not remove that tail.

Theta-mean deviations range from −3.884837° at capture2 +10° to +.460850° at capture4 −10°. The finite theta anchor contributes 122.644527 at this provisional snapshot. G7 must jointly refine gaze with fresh g and optics while retaining dynamic states. These means are not required to lie within the penalty scale at initialization.

G5's 961 lower-bound A states and mean-demand discrepancies remain unchanged. Endpoint zero ambiguity, unknown physical spatial origins, reference-sensitive M slope and capture-demand confounding remain open. D's A coefficients do not establish a physical accommodation displacement law.

## Numerical evidence

Data-only normalized coefficient rank **10/10**; eigenvalues .0656401–3.7586917; condition number **57.2622**. Curvature prior is not needed to manufacture rank. Positive full-domain P1/P4 optical margins remain verified. Degree-2 D is finite over the declared rectangle.

Pixel-scaled gradient infinity norm **6.49047e−11**, below fixed1e−6. This certifies the conditional linear block. Full state/global stationarity and joint curvature have not been established.

Maximum empirical absolute discrepancies: frozen/recomputed theta/g/F1/mu1/F4/mu4 **0**; predicted native centroid identity **8.5265e−14 px**; corrected sign/units **2.8422e−13 reference px**; visual/local polynomial conversion **1.7053e−13 reference px**; complete forward adapter **1.1369e−13 native px**, with identical P1 g.

[Independent reconstruction](reproduction_checks.json) rebuilds full-metric moments, coefficients, objective/anchors/prior, rank/stationarity, all 20/100 summaries, frozen states, hashes and bookkeeping from saved empirical arrays. Reordered linear algebra agrees within the predeclared1e−9 absolute identity tolerance. Attempt02/01 fitted arrays are separately bitwise identical. Actual experiment runtime **7.513 s**, vectorized NumPy float64 with parallel input loading/exposure summaries. GPU optimization was unnecessary.

## Analytical synthetic cases reviewed without automated test execution

**Sign, units, correction count.** Choose C1=(100,50) native px, g=2, mu1=(1,−2), mu4=(3,4) reference px and D=(10,−5) reference px. Then C4=(120,40), c1=(102,46), c4=(126,48). Measured centroid difference is (24,2) native px; division by g and subtraction of mu4−mu1 gives D=(10,−5). Derived h_cent=(12,1), so g*h_cent=(24,2). Inferred origins recover C1/C4 without additional measurements.

**Covariance changes the weighted solve.** Let g=2, precision Q=I10 with Q[0,4]=Q[4,0]=.25, residual r0[0]=3 and the three P4 x residuals (10,20,30); all remaining residuals zero. Q is positive definite. A zero first predicted P1 edge component makes the P1 profile compatible with this residual. For a constant x term, E^T Q E=3 and E^T Q r0=60.75. Full-metric D_x=60.75/(2*3)=**10.125**, versus independent unweighted-center D_x=10. Its full gradient is 2*(60.75−6*10.125)=0. The implemented normal equations retain this cross-term.

**Origin conversion.** Degree2 requires b0'=b0+s0*w+c2*w², bA'=bA+sA*w, s0'=s0+2*c2*w, sA'=sA and c2'=c2, with w=omega4/10. For x coefficients (2,3,4,5,6), y (−1,.5,2,−3,−2), w=−1, a=.2 and visual t=0, D=(2.6,−.9). Converted coefficients x=(4,−2,−8,5,6), y=(−5,3.5,6,−3,−2), at local t4=1 give the same D. Shifting the angle without conversion would change predictions. Production uses this binomial formula.

No automated tests were added or run. These are analytical audit cases and empirical reconstruction evidence. Historical G5's 55 contracts, including synthetic centroid/sign checks in the unchanged complete forward adapter, remain historical upstream evidence; they are not a fresh G6 test result.

## Decision and handoff

**GO_WITH_LIMIT**: coherent centroid/center predictions, immutable snapshot, full-covariance solve, rank/stationarity and complete scheduled accounting are available. The polynomial depends on provisional visual gaze/A and empirical references. Its conditional rank does not establish joint physical identifiability or accuracy.

One next action: **S7/G7 full DM0 joint optimization**, with dynamic theta/A, fresh P1 g, allowed globals, the same J, common and reproducibly perturbed starts, numerical certification and predeclared compact P4 progress checks. G7 owns residual/state/reference review; G8 owns full scheduled whole-loop agreement. This initializer does not justify new degree, stronger anchors, per-frame centers or expanded accommodation family.

Last usable checkpoint: [checkpoint.json](checkpoint.json). Retry scope: reporting repair complete. Full-fit/certificate/cross-check/comparison flags remain **false**.
