# New protocol: mean initialization, framewise alternation, and joint calibration

Revision 2, 2026-10-10. Status: REVISED SPECIFICATION / NOT IMPLEMENTED.

This revision incorporates [NEW_PROTOCOL_PLAN_AUDIT.md](NEW_PROTOCOL_PLAN_AUDIT.md). It is the current execution plan and supersedes the earlier sequence in [SHARED_KEYSTONE_CALIBRATION_PLAN.md](SHARED_KEYSTONE_CALIBRATION_PLAN.md). The [shared-keystone audit](SHARED_KEYSTONE_CALIBRATION_AUDIT.md) remains relevant. No revised estimator has been implemented or run.

The sequence is: initialize from fixation means; freeze initial anchor targets; alternate Capture 1 keystone/gaze calibration, zero-gaze barrel calibration, and an all-frame accommodation update; then jointly refine shared optics and framewise states using all observations and one objective.

## 1. Shared parameters and framewise states

The shared baseline S0 fits one P1 keystone vector b1, one P4 keystone vector b4, and one barrel/accommodation slope s across every capture. Each frame has theta_x, theta_y and A. One positive magnification g is profiled from that frame's P1 observations and applied unchanged to P4.

Every frame contributes residuals to the same shared optical coefficients. The effective transforms vary frame by frame through gaze and kappa_i = s*(A_i-A_ref). There are no independent final keystone or barrel coefficients per frame/capture. Temporary capture-constant radial fits are permitted only for initialization. The optional S1 model adds one global accommodation-dependent P4 keystone slope after S0 is completed.

## 2. Data, references and exact transform

Use the reviewed vertex correspondence and identities: 100,090 scheduled frames, 89,175 complete frames, 10,915 unavailable; four captures and 20 fixations. The P4 permutation is already applied by the loader. Preserve this full-data population; declared validation splits have their own explicitly reported train/test populations.

Fix B1 and B4 to the centered arithmetic-mean triangles from Capture 1's nominal zero-gaze fixation. A_ref = 0.36036036036036034 D and kappa(A_ref)=0 define the empirical radial origin. These references can contain baseline physical distortion.

Use theta in degrees and t_x=theta_x/10, t_y=theta_y/10 in every capture. For each pattern r, coefficient order is b_r=(b_r0,b_r1,b_r2,b_r3). R_r=sqrt(mean_j(||B_rj||^2)) is a fixed reference length used in the denominator, not an output normalization.

```text
P1_ij = B1_j
P4_ij = (1 + s*(A_i-A_ref)*||B4_j||^2) * B4_j
a_r_i = b_r0*t_x_i^2 - b_r1*t_y_i^2
d_r_ij = 1 + (b_r2*t_x_i*P_r_ij,y + b_r3*t_y_i*P_r_ij,x)/R_r
F_r_ij = (exp(a_r_i)*P_r_ij,x, exp(-a_r_i)*P_r_ij,y) / d_r_ij
mu_r_i = mean_j(F_r_ij)
H_r_ij = F_r_ij - mu_r_i
X_r_ij = measured_r_ij - mean_j(measured_r_ij)
g_i = sum_j(dot(X1_ij,H1_ij)) / sum_j(||H1_ij||^2)
predicted_X1_ij = g_i*H1_ij
predicted_X4_ij = g_i*H4_ij
```

Keystone b coefficients are dimensionless in this basis; native quadratic coefficients are b0/100 and b1/100 deg^-2. Native projective coefficients before P/R are b2/10 and b3/10 deg^-1. The slope s has px^-2 D^-1 units. Convert historical coefficients through their recorded angle units before comparison; do not transplant them as calibration constants.

Identity at theta=(0,0) is an operational empirical convention. This does not identify separate physical P1/P4 optical zeros from [Theory.md](Theory.md). No optical-zero offset, image-axis rotation, template adjustment, extra P4 scale, RMS/area factor or temporal smoothing is released in S0.

Reprofile g whenever theta or b1 changes. At fixed theta and b1, A and P4 coordinates do not enter g directly. Derivatives of P4 predictions must nevertheless include dg/dtheta and dg/db1. This is a P1-constrained criterion, not a joint least-squares profile of g using both patterns.

Derived centers are C_r=measured_centroid_r-g*mu_r. To invert a centered observation use z_r=X_r/g+mu_r at the same trial states/coefficients, undo the exponential stretch and projective denominator, then undo the radial map for P4. No free centers are fitted.

Historical controls now retain compact reports/metadata under [experiment retention policy](../experiments/README.md). Archived frame arrays and snapshots were removed from the working tree; full-state controls require restoration from the recorded cleanup commit or replay from source detections. New calibration can reconstruct its references from detections; the previous raw reference is also retained in REFERENCE.json.

## 3. Phase I: initialization and fixed anchors

### I1. Polynomial gaze from five arithmetic fixation means

For each capture compute framewise delta=mean(P4)-mean(P1) and its arithmetic mean in each fixation. Let x0,y0 be the mean displacement at the nominal zero-gaze fixation, h=(max_f(mean_delta_x)-min_f(mean_delta_x))/2, and u_f=(mean_delta_x_f-x0)/h. Fit a1,a2 by equal-weight least squares of a1*u_f+a2*u_f^2 to [-10,-5,0,5,10] degrees; the intercept at u=0 is zero.

For individual frames set theta0_x=a1*u+a2*u^2 and theta0_y=(a1/h)*(delta_y-y0), using u=(delta_x-x0)/h. Reject a degenerate h or a polynomial whose derivative changes sign over the observed frame inputs. This uses polynomial values at mean inputs; report separately the means of framewise polynomial outputs. Separate per-capture polynomials are the declared provisional gaze policy. Their targets remain fixed throughout this experiment.

### I2. Capture 1 keystone initialization

Use Capture 1's mean patterns and fixation gaze summaries for initial b1/b4 estimates. Profile scale during P1 fitting; divide individual frames by their P1-derived scale before producing corrected mean summaries. Then refine the initial coefficients against Capture 1's individual measured vertices with theta=theta0, A=A_ref and relative kappa=0. Use the common bounds below and enforce the resulting shared geometry's domain on all complete frames.

This gives a provisional shared b1_0 and b4_0. Compute P1-derived g for every capture. The approximation A=A_ref during this initialization may allocate accommodation variation to keystone; Phase II explicitly revisits it.

### I3. Zero-gaze radial initialization without circular inversion

Use all captures' nominal zero-gaze fixation frames at their actual estimated gaze. A provisional kappa=0 may be used for a first inverse/mean display only. Every inverse must restore mu4 from that declared provisional model.

For the calibration result, fit one scalar kappa0_c per capture through the centered forward P4 model at the actual framewise theta0 and P1 scale, with shared b4_0 fixed. Fix kappa0_1=0 as the reference convention. Use equal frame weights within each zero-gaze fixation and the valid radial branch. At every trial kappa recompute the radial vertices, keystone, mu4 and predictions together. Correct each frame using the fitted trial-consistent model before averaging its recovered P4 pattern. Do not invert measured centered coordinates about a fixed measured centroid.

These temporary capture-constant kappa fits initialize the shared law; they are not final framewise/capture-specific parameters.

### I4. Accommodation scale and targets

The nominal labels in capture order are D_c=[0.36036036036036034,4,3,2] D. Set x_c=D_c-A_ref. Fit s0 by equal-capture least squares of kappa0_c=s0*x_c, through the anchored origin and under the declared slope bound. An unconstrained interior solution is sum_c(x_c*kappa0_c)/sum_c(x_c^2).

Define one target per capture:

```text
Abar0_c = A_ref + kappa0_c/s0
Abar0_f = Abar0_c for each of that capture's five fixations
```

This uses the zero-gaze P4 estimate as the anchor target and assumes constant nominal demand across a capture's five fixation periods. Each fixation's inferred mean remains free under the 0.25 D soft penalty; individual A values remain free within operational/forward domains. Record both inferred targets and nominal labels.

Require finite s0 and |s0|*max_j(||B4_j||^2)*max_c(|D_c-A_ref|) > 1e-8. Require all Abar0_c in [0,6] D. If initialization violates these checks, report INITIALIZATION_INVALID with the offending values; do not clip targets, substitute nominal targets, or claim calibration completion. A slope sign and any active initialization bound must be reported. A nominal-demand target control is a separately labeled sensitivity run, not a silent fallback.

Initialize every frame A_i=Abar0_c, then obtain initial framewise A by a forward fit with optics and theta0 fixed and the same mean anchors. Freeze q0=(b1_0,b4_0,s0), theta0_i, Abar0_f and all penalty settings before Phase II. Do not move the targets to follow subsequent fitted solutions. P4-derived targets are same-recording regularizers, not independent accommodation measurements.

## 4. Fixed numerical policy for the first implementation

These are proposed computational defaults for the experiment, not physical limits or accuracy claims. Save resolved arrays/constants in protocol.json before optimizing. If a required domain is infeasible, fail/report it and make any changed policy a separately versioned comparison.

### State and optical domains

- A_i is in [0,6] D. Each gaze component is within theta0_component +/-5 degrees; these are broad operational boxes around its initial measurement-derived estimate. Record all active bounds and their sensitivity.
- Let delta=1e-8, M=max_j(||B4_j||^2), and Da=max(A_ref,6-A_ref). Bound |s| by W_s=(1-delta)/(3*M*Da). This conservative bound keeps the reference-vertex radial derivative at least delta for every A in [0,6], for either sign of s. For initialization use |kappa0_c| <= (1-delta)/(3*M). Check radial factor >=delta and radial derivative >=delta numerically as well.
- Define T_l=max(max_i|theta0_i,l/10|,0.01) for each axis, once from the fixed initial gaze. Let L1_l=max_j|B1_j,l| and L4_l=(4/3)*max_j|B4_j,l|; the latter envelopes reference vertices under the slope/radial bounds above.
- For each pattern define fixed coefficient half-widths W_r=(0.5/T_x^2, 0.5/T_y^2, 0.4*R_r/(T_x*Lr_y), 0.4*R_r/(T_y*Lr_x)). Require finite nonzero denominators in these width definitions; otherwise report an unidentifiable template/basis. Bound each b_rj in [-W_rj,W_rj]. These scale the existing raw-chain 0.5 stretch/0.4 projective search conventions to one common basis; the per-trial checks below remain mandatory as gaze varies.
- At every proposal, on all complete frames require finite coordinates, |a_r_i|<=2, forward denominator >=delta, positive P1 template energy and g_i>0. P1 template energy must exceed 1e-12 times its reference energy. Reject an invalid proposal; never remove its observations from the forward fit. Global changes must check all frames, including during restricted warm starts.
- Numerical inverse validity is diagnostic. Undo the projective map only when its inverse denominator is >=delta. For radial inversion of r_d=r+kappa*r^3 use the nonnegative branch continuous from r=0. For kappa<0 bracket r in [0,sqrt(-1/(3*kappa))]; for kappa>=0 bracket [0,r_d]. Handle kappa=0 directly. Reject radii beyond the branch's reachable r_d, use a safeguarded root solve with relative tolerance 1e-10, and report inverse failures without changing fitting masks/bounds. At the recovered radius require radial derivative >=delta; a singular turning-point inverse is reported invalid.

### Fixed soft penalties

Use a 1 px residual scale for anchor penalties. Set sigma_theta_x=sigma_theta_y=0.5 degrees, with no second free gaze multiplier. Define W by concatenating the eight W_r coefficients and W_s, and sigma_q=0.25*W. Use the diagonal optical penalty (1 px)^2*sum_j(((q_j-q0_j)/sigma_q_j)^2).

These widths are declared regularization scales based on the frozen search coordinates, not estimated standard errors. The 0.25 D accommodation mean width remains fixed. Prespecified sensitivity runs change only one policy at a time: gaze width 0.25 or 1 degree; optical width multiplier 0.5 or 2; or nominal-demand A mean targets instead of Abar0. Keep the original targets otherwise fixed. Do not select a setting merely to flatten A curves or reduce training error. Contiguous-block uncertainty analysis can be reported separately without replacing the frozen primary penalties.

## 5. Phase II: framewise alternating warm start

After mean initialization, fit individual measured vertices. A cycle is:

1. Propose a Capture 1 b1/b4 and gaze update using all its fixations, its current A, fixed anchors and P1-profiled scale.
2. Propose a shared s update from every capture's zero-gaze fixation, using current framewise A, gaze and shared keystone. A is not assumed nominal in this update.
3. Update A on **all complete frames** with the current optics/gaze and full fixation-mean penalties. Capture 1 accommodation at its nonzero-gaze fixations now feeds into the next keystone update.
4. Recompute predictions, optical means, P1 scales, centers, domain margins and the full objective.

The restricted optical blocks propose steps. Accept/damp every proposed step against the one full objective in section 6, including its anchors and all affected frames. If no feasible decreasing step exists, record the rejected proposal and move to the next block. Capture 1 zero-gaze frames occur once in the full data objective.

Use at most 20 warm-start cycles. Stop earlier if two consecutive cycles have relative full-objective change <=1e-7 and max state/optical change in normalized coordinates <=1e-4; also stop if every block is rejected in a cycle. This is an initialization checkpoint, not a joint-stationarity PASS. Proceed to Phase III with its recorded status and states.

## 6. Phase III: one joint objective on all frames

Optimize shared q=(b1,b4,s) and all framewise theta_x,theta_y,A, profiling g at every relevant proposal. For each fixation f, average its frame data costs; then average over 20 fixations. With e_rij=X_rij-g_i*H_rij, use:

```text
L_data = mean_f(mean_i_in_f((sum_j||e1_ij||^2 + sum_j||e4_ij||^2)/3))
L_gaze = (1 px)^2 * mean_f(mean_i_in_f(sum_xy(((theta_i-theta0_i)/0.5 deg)^2)))
L_A = (1 px)^2 * mean_f(((mean_i_in_f(A_i)-Abar0_f)/0.25 D)^2)
L_optical = (1 px)^2 * sum_j(((q_j-q0_j)/sigma_q_j)^2)
L = L_data + L_gaze + L_A + L_optical
```

This gives equal P1/P4 data weights and equal fixation weights. The gaze sum is the squared norm of the componentwise normalized differences. Every component is reported separately. Anchors use dynamic fixation means of the current proposed A; a state block must include its effect on those full means. There is no per-frame A prior or temporal smoothing.

Use vectorized CuPy float64 state updates and shared optical updates under the same L. Recompute all dependencies within every trial; do not cache g or mu across a changed state/optical vector. Use normalized variables theta/0.5 degrees, A/0.25 D and q/sigma_q for numerical steps and certificates.

For line searches try fractions 1,1/2,...,2^-24, requiring full-domain validity and an Armijo decrease with constant 1e-4 for a descent direction. Reject a restricted proposal that is not a descent direction for the full objective. Numerical roundoff allowance is 1e-12*max(1,|L|), recorded separately from an accepted scientific improvement.

Use up to 100 joint block cycles and 500 iterations per numerical block. Exhausting either cap without the certificate below gives NOT_CONVERGED. Run seven deterministic starts: the Phase II checkpoint; A shifted by +/-0.25 D; and q shifted by +/-0.1*sigma_q in either the two shared horizontal-quadratic entries together or the radial-slope entry alone. Project starting points only onto declared box bounds and backtrack toward the feasible base for nonlinear domain validity; record the actual start. This never clips observations or anchor targets.

For certification use J=N*L/(1 px)^2, with N the complete-frame count, and normalized coordinates. Require a maximum absolute projected gradient <=1e-5 on free/box-constrained directions, plus primal feasibility. If nonlinear forward constraints are active, require their KKT stationarity, multiplier sign and complementarity residuals <=1e-5 after declared constraint scaling; a box-only projected gradient is insufficient. If a feasible KKT certificate cannot be obtained, label DOMAIN_LIMITED or UNCERTIFIED. Save the constraint scales and margins. Use dimensionless constraints d-delta, radial_factor-delta, radial_derivative-delta, 2-a, 2+a, g, and P1_energy/reference_energy-1e-12, each with unit scale; classify a nonlinear constraint as active within 1e-7 of its boundary while still enforcing primal feasibility. Record any alternate scaling as a protocol change. Check for negative feasible curvature with analytic Hessian-vector products or finite-difference gradients; unresolved negative curvature prevents a local-minimum claim. A stationary fit is not a global optimum.

## 7. Scale gauges and the scope of inference

For any admissible positive c, A_i'=A_ref+c*(A_i-A_ref) and s'=s/c leave every kappa and S0 image prediction unchanged. The reference fixes a radial origin, while labels/anchors select the diopter scale. For gaze, theta_x'=cx*theta_x and theta_y'=cy*theta_y can be compensated in both keystone vectors by b0'=b0/cx^2, b1'=b1/cy^2, b2'=b2/cx, b3'=b3/cy. The image predictions and profiled g remain unchanged.

Report these known data-only null directions explicitly. Examine data Jacobians separately from anchor-augmented Jacobians, and profile global tradeoffs after accounting for state nuisance parameters and the known gauges. Prior-induced curvature is not independent physiological information. Bounds may restrict a gauge without establishing its physical scale. Report state amplitudes, coefficients and data/anchor costs under the prespecified anchor controls.

P1 and P4 are centered separately in L_data. The strong relative-centroid gaze signal therefore enters through theta0 and its soft penalty. This is a conditional shape-plus-gaze-anchor calibration; it does not implement Theory's full D(theta,A) observation model. Derived centers add no independent measurement. Report gaze departures, anchor cost and shape-only gaze sensitivity. A full relative-displacement model is a separate later comparison.

## 8. Optional S1 accommodation-dependent keystone

After completing and auditing S0, test only:

```text
b4,0(A_i) = b4,0,ref + b4,0,A*(A_i-A_ref)
```

Use the same initialization targets, references, data and primary penalties as S0. Anchor b4,0,A at zero with width sigma_b4,0/Da and bound its absolute value by W_b4,0/Da; check actual |a|/projective/radial domains on every trial. Jointly refit the existing globals/states as well. Include both radial and explicit keystone terms in d(prediction)/dA, and update mu and inverse at the same A.

Under the accommodation scale symmetry, b4,0,A also transforms to b4,0,A/c. Zero-gaze observations alone cannot identify its (A-A_ref)*theta_x^2 response. Use all captures/gaze positions, profiles against s, multiple starts, fixed anchor controls, and the validation below. Additional keystone A terms require a separate declared deficiency/comparison. Compare corresponding vertex predictions and state sensitivity rather than adding the new prior cost to an otherwise incomparable score.

## 9. CPU audit and retained-input validation

Independently replay forward/inverse equations on CPU. Check derivatives of every global/state parameter through g and mu, synthetic closure, common-translation invariance, branch bounds, complete-frame identities, equal-fixation weights, dynamic mean-anchor gradients and final stationarity. Report forward median/P95/RMS and signed vertex errors, inverse tails/coverage, A mean and within-fixation SD, gaze changes, scale statistics, active constraints and all objective components. Preserve all failure counts. Capture and nominal demand remain confounded.

Prespecify a separate validation run: within each fixation, use the final 20% of scheduled source rows as test and the earlier rows as training, with the final 50 scheduled training rows reserved as a gap. Apply the existing completeness rule within each split and report scheduled/complete/gap counts. Require usable data in every needed training/test fixation. Rebuild empirical references, global coefficients, gaze mappings and A/optical anchors from training data only. Paired S0/S1 candidates use the same split-specific references and policies. Temporal drift makes this a temporal generalization check within the recorded conditions; it does not establish cross-subject or physiological accuracy.

For each test-frame P4 slot j, omit that measured vertex and infer theta/A/g using all P1 vertices and the two retained P4 vertices I. Fit each retained-slot gaze polynomial on training data using the same retained P4 centroid definition. Initialize test states from that retained-input polynomial and training-only A targets, never from cached full-P4 states. Use the same declared soft widths; test fixation-mean targets come from training. Check the reduced observation Jacobian rank and report prior dependence.

For the reduced P4 shape objective use centered retained observations and centered retained modeled vertices, divided by the number retained. Recover translation with C4_I=mean_l_in_I(p4_measured_l-g*F4_l). Predict the omitted vertex as C4_I+g*F4_j and report its error in original camera pixels. P1 scale remains derived from all retained P1 points. No full-three-point measured P4 centroid enters this inference. Altering the omitted test coordinate must leave inferred states, g, branch decisions and prediction unchanged.

An internal omission run using a globally frozen full-data calibration may also be reported, but must be labeled internal consistency and cannot replace the training/test procedure. Confidence claims require an appropriate analysis of temporal dependence and reuse of measurements.

## 10. Artifacts, decisions and implementation order

Create new outputs under `experiments/reverse_transform/shared_keystone/`:

1. `stage_01_mean_initialization`: references, mean summaries, polynomial fits, provisional/refined initial optics, target/width/bound manifests and hashes.
2. `stage_02_framewise_alternation`: all cycle proposals, accept/reject decisions, all-frame A feedback, objective decomposition and domain checks.
3. `stage_03_joint_refinement`: shared S0 fit, multistart/stationarity records, frame outputs, diagnostics and prespecified anchor controls.
4. `stage_04_a_dependent_keystone`: optional matched S1 comparison with the full A derivative.
5. `stage_05_validation`: split-specific calibration provenance and retained-input prediction records for completed candidates.

Every saved run includes protocol.json, provenance.json, immutable source snapshot, summary.json, frames.npz, audit.json, RESULTS.md and plots as appropriate; use a scientific review for the paired comparison. Hash initial and accepted anchor/parameter manifests and retain all scheduled row identities. Document-only audit closure is not numerical audit PASS.

A successful numerical fit retains the declared population and passes its equation/derivative/domain/stationarity audit. Scientific promotion additionally requires stable response parameters under the declared controls and defensible held-out geometry. A flatter A curve, lower training SSE, or bounds that suppress gauge motion are insufficient alone. Preserve canonical raw Stages 01–06. No implementation, fit, commit or push is part of this document update.
