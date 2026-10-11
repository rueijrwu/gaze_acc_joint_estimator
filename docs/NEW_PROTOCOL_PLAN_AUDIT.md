# Audit of the new iterative and joint calibration protocol

Reviewed 2026-10-10. Basis: working-copy [NEW_PROTOCOL_PLAN.md](NEW_PROTOCOL_PLAN.md), repository HEAD e8a1d3cb89ea40cea40275225c0ce93588b9695e. Reviewed plan SHA256: 419d1dc4f29f4aab60cbc6cc4acca8bdbadf55cf2dff2c4a06321846628d7feb.

Status: DOCUMENT AND ANALYTIC AUDIT; no new estimator, numerical experiment, or physiological validation was performed.

Initial-review decision: the sequence was coherent but needed the corrections below. Revision 2 specification disposition is recorded at the end; numerical verification remains pending.

## Findings by priority

| ID | Priority | Finding | Required resolution |
|---|---|---|---|
| NP-01 | High | Exact gaze/keystone and accommodation/barrel scale ambiguities remain in the data model | Declare anchor-defined scales and audit data information separately from regularized information |
| NP-02 | High | Fixation accommodation targets and optical/gaze penalty strengths are not fully specified | Freeze an explicit target construction and numerical penalty policy before fitting |
| NP-03 | High | The alternating blocks do not update accommodation at Capture 1's nonzero-gaze fixations | Add an A update covering those frames before the next Capture 1 keystone update |
| NP-04 | High | Keystone inversion before barrel initialization depends on the unknown barrel-induced optical mean | Use a declared provisional radial seed and iterate, or fit the scalar radial parameter in the forward model |
| NP-05 | Medium | The reduced centered-triangle objective omits the measured relative-center channel | Describe the joint fit as conditional on empirical gaze anchors; reserve a full D model for a separate comparison |
| NP-06 | Medium | Omission checks lack a complete retained-input inference recipe | Rebuild all frame-dependent quantities and specify retained-point translation reconstruction |
| NP-07 | Medium | Exact operator, angle convention, bounds, and convergence rules are deferred | Pin a complete machine-readable protocol before implementation/run |

## NP-01 — exact scale ambiguities

Source: plan sections 1, 3.I3–I4, 5, and 6.

The image model depends on accommodation through kappa_i = s*(A_i-A_ref). For any positive c that preserves the declared domain,

```text
A_i' = A_ref + c*(A_i-A_ref)
s'   = s/c
```

leaves every kappa_i and image prediction unchanged in the initial model. If the optional keystone slope is enabled, also set b4,0,A' = b4,0,A/c and its predictions remain unchanged. Reference kappa(A_ref)=0 fixes an origin, not the diopter scale. Bounds can restrict this family, but that does not turn a bound into physical calibration.

There are corresponding gaze-scale symmetries in the inherited raw keystone. In a fixed angle basis, write a = b0*tx^2 - b1*ty^2 and denominator = 1 + b2*tx*B_y/R + b3*ty*B_x/R. Replacing tx by cx*tx and ty by cy*ty while replacing b0 by b0/cx^2, b1 by b1/cy^2, b2 by b2/cx, and b3 by b3/cy leaves the transform unchanged. Apply this to both P1 and P4. Profiled P1 magnification also remains unchanged because its modeled triangle is unchanged.

The polynomial/optical anchors establish the angular scale, and nominal-label initialization plus the accommodation/optical anchors establish the diopter scale. That can define a useful regularized estimator, but absolute state scales are not independently identified by triangle predictions.

Required corrections:

- State explicitly which fixed quantities/anchors select each scale.
- Examine image-data Jacobians and anchor-augmented Jacobians separately; report the known scale directions rather than demanding full data-only rank.
- Profile keystone/barrel tradeoffs after accounting for frame states and these gauges. An augmented positive Hessian can be supplied by the priors.
- Report how state amplitudes and coefficient estimates change under weaker/stronger anchors, not only how vertex errors change.

This is an analytic model symmetry, not a measured numerical condition number or a claim that optimization must fail.

## NP-02 — anchor targets and widths are not yet executable

Source: plan sections 3.I3–I4 and 5.

The plan first permits either nominal-demand-based or inferred-P4 accommodation targets and then says to freeze Abar0_f. These choices produce different estimators. If each nonzero-gaze fixation supplies its own P4-derived target, residual gaze deformation in the initialization can be preserved by the accommodation prior. The optical anchors may preserve that same initial allocation of deformation a second way.

Recommended primary construction, matching the user's zero-gaze initialization:

1. Fit one initial relative radial coefficient kappa0_c from each capture's corrected zero-gaze data, with kappa0_1=0 by reference convention.
2. Fit s0 to kappa0_c versus the nominal demand difference from A_ref, with a declared weighting rule.
3. Define Abar0_c = A_ref + kappa0_c/s0 and use that capture's zero-gaze-derived target for its five fixation-mean anchors. This assumes the nominal demand remains constant across its five fixations. It permits framewise fluctuations and finite differences between fixation means under the soft penalty.
4. Record these inferred targets alongside nominal demand. Treat nominal-demand mean anchors as a declared sensitivity control. Reject an invalid/near-zero slope or targets outside the operational domain; do not silently clip them.

This is a recommendation to resolve the open choice, not a completed user-approved calibration result. If separate P4-derived targets per fixation are desired instead, document that their gaze-dependent bias can be inherited and compare against the zero-gaze-derived policy.

Keep the agreed 0.25 D fixation-mean width. Specify numerical sigma_theta, lambda_gaze, and Lambda_optical, coefficient units, and the construction of optical widths before a run. Only the ratio lambda_gaze/sigma_theta^2 determines each gaze penalty strength; avoid treating those as independent tunable evidence. Fix a base setting and prespecified sensitivity settings. Same-recording anchor construction is regularization, not independent measurements or calibrated uncertainty.

## NP-03 — the warm-start feedback loop is incomplete

Source: plan section 4.

The Capture 1 block updates keystone and gaze but holds A fixed. The next block updates A only in nominal zero-gaze fixations. Therefore Capture 1 A at -10, -5, +5, and +10 degrees remains at initialization during every restricted cycle. Accommodation variation at those fixations cannot feed back into the Capture 1 keystone calibration as described. The final joint fit eventually releases A, but the claimed iterative warm start has not done so.

Correction: after updating the shared radial law, update A for at least all Capture 1 frames, preferably all complete frames, with current shared optics and fixed anchors. Then repeat the Capture 1 keystone/gaze step. A practical cycle is:

```text
Capture 1 framewise keystone/gaze update
-> all-capture zero-gaze radial-law update
-> all-frame accommodation update under current shared optics
-> re-evaluate full objective and domains
-> repeat
```

Restricted optical updates are warm-start proposals. For a monotone version, accept/damp them against the one full objective; otherwise set a declared iteration cap, save objective increases, and require the final all-frame solve to supply convergence. Do not describe restricted-subset convergence as joint stationarity. Do not double-count Capture 1 zero-gaze frames when constructing a common data objective.

## NP-04 — correcting centered P4 before fitting barrel has a dependency

Source: plan sections 2 and 3.I3.

Measured centered P4 coordinates satisfy X4_i/g_i = F4_i - mu4_i. To invert keystone, first restore mu4_i = mean_vertices(K4(theta_i, radial(B4,kappa_i))). That mean depends on the unknown kappa_i and generally is nonzero for a non-equilateral empirical triangle. Applying inverse keystone directly to the measured centered triangle uses the wrong origin. The plan correctly requires optical-mean restoration, but its initialization order does not specify the radial value used to calculate that mean.

Resolve the dependency explicitly. A mean-based approximate start can use kappa=0 and label the result provisional. Then fit/update kappa and recompute the optical mean and inverse consistently. A cleaner refinement fits the capture scalar kappa using the forward centered predictions at the actual framewise gaze and P1 scales; corrected means are then reported from that fitted model.

At exact model gaze zero the keystone operator is identity, which simplifies a mean initializer. Actual frames in the zero-gaze fixation still have nonzero estimated gaze; do not set their gaze to zero to evade the dependency. Means are initialization summaries, while the final fit uses individual vertices and the full forward transform.

## NP-05 — scope of the centered-triangle joint objective

Source: plan sections 2 and 5; compare [Theory.md](Theory.md), sections 6–8.

Centering P1 and P4 independently removes their relative centroid displacement from the vertex objective. The strong P4–P1 gaze signal enters through the empirical polynomial gaze anchor. The derived optical centers are outputs of the same model/measurements and add no independent information.

This is a valid conditional shape-plus-gaze-anchor experiment. It does not implement the full relative-coordinate D(theta,A) model described in Theory, and same-recording gaze anchors are not independent validation of the inferred states. Releasing gaze or global keystone can improve shape errors partly by moving against that prior.

Keep the reduced experiment for the requested first comparison, name its scope explicitly, and report gaze-anchor cost, gaze departures, and shape-only gaze sensitivity. Do not silently add an independent centroid penalty on top of the same data. A full relative-coordinate D model would require a separately declared ablation and observation/weighting contract.

## NP-06 — omission validation needs an explicit inference recipe

Source: plan section 7.

The full-three-point measured P4 centroid, polynomial gaze anchor, inferred A anchors, initialization and cached states can all contain a point claimed to be omitted. Merely removing that point's final residual is insufficient.

For a frozen centered-shape model and retained P4 set I, a possible retained-input translation reconstruction is:

```text
C4_retained = mean_I(p4_measured_j - g*F4_j)
p4_predicted_omitted = C4_retained + g*F4_omitted
```

Infer states from retained inputs under a declared retained-point observation metric. Recompute the frame gaze anchor from a retained-input mapping or use an explicitly independent initializer/prior; the existing three-point polynomial anchor cannot be reused unchanged. P4-derived fixation targets must also exclude held measurements or originate from a separate frozen training split. Check rank when using only two P4 vertices.

Distinguish an internal omission check with globally frozen calibration from true held-out calibration/inference. For predictive claims, training data used to form references, gaze/optical/mean-A anchors, and global coefficients must respect the split. A within-split perturbation of the omitted measurement must leave inference and its prediction unchanged.

## NP-07 — pin the remaining numerical conventions

Source: plan sections 2, 3.I4, 4, 5, and 7.

Before implementation/run, specify:

- The exact inherited raw exponential/projective operator, coefficient ordering and common angle units. The general Theory operator is not identical to the current raw empirical parameterization.
- Operational gaze zero versus physical optical zero. The current empirical model fixes identity at its reference gaze; that does not establish the separate physical P1/P4 optical zeros in Theory. Preserve this declared operational convention initially.
- Gaze limits, optical/radial coefficient bounds, positive denominator/branch margins, and rejection policy. Check the full proposed frame population after every shared update, including warm starts.
- The signed radial branch and inverse solver, including how invalid measured inversions remain reported while their frames stay in the forward objective.
- Gaze/optical prior strengths, multistart construction, step acceptance/damping, iteration caps, and joint projected-gradient/active-constraint tolerances. Report when a frame or coefficient lies on a bound.
- Complete derivatives through P1-only g. P4 residual derivatives include dg/dtheta and dg/db1 even though P4 never enters the scale's profiling formula. This is a P1-constrained objective, not a joint least-squares profile of g over P1 and P4.
- No temporal smoothing unless separately declared, and equal-fixation scaling of every mean/state penalty.

These are reproducibility requirements; this audit does not invent numeric physical bounds without an explicit parameter-domain justification.

## What the plan already gets right

- Fixed empirical references and relative radial gauge.
- One shared optical law estimated from individual frames.
- One positive P1-derived magnification shared with P4.
- Raw keystone without an extra RMS/area factor.
- Frozen initial anchor targets and finite fixation-mean A penalties.
- A final common all-frame objective, multistart checks, and joint stationarity.
- A minimal A-dependent keystone extension only after the shared baseline.
- Preserved historical outputs and stage-organized reproducible artifacts.

## Recommended revision order

1. Define anchor target construction, explicit operator/units, and numeric regularization/domain policy.
2. Repair initialization's optical-mean dependency and the missing all-frame A feedback step.
3. Record known scale gauges and the reduced centered-model scope.
4. Implement the shared baseline and audit all derivatives/constraints under one common objective.
5. Evaluate data-versus-prior sensitivity and an explicitly retained-input validation protocol before promoting A-dependent keystone.

No implementation, new run, commit, or push was performed for this audit.

## Revision 2 specification disposition

The user requested incorporation of this audit. [NEW_PROTOCOL_PLAN.md](NEW_PROTOCOL_PLAN.md), Revision 2, SHA256 a833ded4b067b95cf028bde3a21f73ca13538aa26b351341640770fe3a44a96d, now addresses the findings in the specification:

| Finding | Revision 2 resolution | Verification status |
|---|---|---|
| NP-01 | Section 7 explicitly states gaze and accommodation scale symmetries, anchor-defined units, and separate data/regularized information diagnostics | Specified; numerical sensitivity pending |
| NP-02 | Sections 3.I4 and 4 define zero-gaze-derived capture targets reused across five fixations, fixed 0.25 D mean width, gaze/optical penalty defaults and prespecified controls | Specified; initialization and sensitivity pending |
| NP-03 | Section 5 adds all-frame A updates and full-objective acceptance before the next Capture 1 keystone update | Specified; convergence audit pending |
| NP-04 | Section 3.I3 uses a trial-consistent forward radial fit and labels any zero-radial inverse initializer provisional | Specified; centroid/inverse closure audit pending |
| NP-05 | Sections 2 and 7 explicitly name the reduced centered-observation scope and empirical gaze prior | Scope recorded; full D remains separate |
| NP-06 | Section 9 specifies training-only references/anchors, retained-input gaze mappings, translation reconstruction and omitted-coordinate noninterference | Specified; leakage/rank checks pending |
| NP-07 | Sections 2, 4 and 6 fix the exact operator/units, deterministic bounds/widths, numerical branches, starts, acceptance and stationarity rules | Specified; implementation/certification pending |

The shared-keystone companion plan has been aligned with this sequence. These changes resolve the document-level omissions; they are not a numerical audit PASS. The proposed operational defaults and prior scales remain assumptions to evaluate through the declared controls. Existing Stage 06 numerical results do not certify this new shared model. No new fit has been run.
