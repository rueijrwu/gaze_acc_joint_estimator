# Audit: capture-specific keystone calibration can absorb accommodation response

**Repository:** `rueijrwu/gaze_acc_joint_estimator`  
**Branch:** `exp5_distortion_model`  
**Code/results reviewed at:** `980d6d3640f06bf5823e8e908f31d422c94006a7` (2026-10-10)  
**Status:** DOCUMENT-ONLY AUDIT; no new fit or numerical experiment was run  
**Companion:** [Shared-keystone corrective experiment plan](SHARED_KEYSTONE_CALIBRATION_PLAN.md)

## Executive decision

**Finding SK-01 — confirmed implementation/design mismatch; high priority.** Raw Stage 03 fits **independent P4 keystone coefficients for each capture**. These same capture-specific coefficients are then frozen in Stages 04, 05, and 06, while framewise accommodation is primarily represented by a **shared radial law**. This is inconsistent with the intended final model of a *single calibrated optical system*, whose keystone should be one shared function of gaze and, only if supported, accommodation:

\[
\boxed{K_4(\theta;\boldsymbol\beta)\quad\text{or}\quad
K_4(\theta,A;\boldsymbol\beta_0,\boldsymbol\beta_A).}
\]

The latter's coefficients are globally fitted once and evaluated at the current accommodation. It is **not** four unrelated capture-specific keystone operators.

**Finding SK-02 — scientifically plausible but unproven mechanism.** Accommodation-dependent P4 shape/keystone deformation can be allocated to independently fitted \(K_{4,c}\), to fitted \(\kappa_c\), or partly to both. Later inversion then removes the effect assigned to \(K_{4,c}\) as though it were a fixed capture property, so the framewise \(A\) fit cannot recover that part through \(\kappa(A)\). Neither the observed coefficient trends nor numerical audit PASS establish the magnitude of this absorption. The one-demand-per-capture design also confounds true accommodation with capture/setup differences.

**Recommendation:** Keep all canonical Stages 01–06 as read-only historical controls. Next implement and audit a **global raw-keystone calibration** from the same four captures, comparing shared \(K_4(\theta)\) first and a minimal shared \(K_4(\theta,A)\) only as a declared extension. Refit the radial law and downstream states consistently; do not transplant old coefficients.

## 1. Verified code path

Reviewed sources:

- [Stage 01–04 shared runner](../experiments/reverse_transform/raw_keystone/scripts/run.py), specifically the `stage in (1,3)` capture loop and Stage 02/04 cases.
- [Raw projective operator and inverse](../distortion_model/raw_keystone.py), classes `RawKeystone` and `RawFrameAccommodation`.
- [Fit and gaze-calibration helpers](../experiments/reverse_transform/raw_keystone/scripts/helpers.py).
- [Stage 06 framewise center/gaze cycle](../experiments/reverse_transform/raw_keystone/stage_06_framewise_centers/scripts/run.py).
- [Raw Stage 03 results](../experiments/reverse_transform/raw_keystone/stage_03_independent_captures/RESULTS.md) and [Stage 06 results](../experiments/reverse_transform/raw_keystone/stage_06_framewise_centers/RESULTS.md).
- [Theory](Theory.md), especially the proposed shared transformations and full relative-coordinate model.

**Stage 01.** Capture-1 P1 keystone is calibrated against a fixed empirical, non-equilateral P1 triangle, with frame gaze initialized from the P4–P1 centroid signal. One positive frame scale \(g_i\) is profiled from P1.

**Stage 02.** Capture-1 P4 keystone \(K_{4,1}\) is fit at fixed gaze and P1-derived \(g_i\); its *relative* radial increment is fixed to the reference gauge \(\kappa_1=0\).

**Stage 03.** Capture 1 reuses Stages 01–02. For each of Captures 2–4, the runner independently recalibrates gaze, fits P1 keystone \(K_{1,c}\), then jointly fits four P4 keystone coefficients \(K_{4,c}\) and one capture-constant relative radial \(\kappa_c\), using the P1 frame scale. Nominal demand labels are used *afterward* to regress \(\kappa_c\) against demand. The saved per-capture \(K_{4,c}\) are not constrained by a shared \(K_4(\theta,A)\) law.

**Stage 04.** The per-capture gaze, P1 scale, \(K_{4,c}\), and post-fit \(\kappa(A)\) relation are frozen. Only \(A_i\) changes in the P4 centered forward objective.

**Stage 05.** Horizontal gaze and/or \(A_i\) are varied in a Capture-1 ablation. The globally missing accommodation dependence of \(K_4\) is not identified or fit.

**Stage 06.** Derived per-frame centers are used to refit a **separate gaze polynomial per capture**, then P1 scale is reprofiled and \(A_i\) refit. Stage 03 capture-specific P4 keystone coefficients remain frozen. Stage 06 reaches a consistent fixed point, but that does not imply the fixed keystone coefficients are globally/physically correct.

All numerical audits of these conditional fits may pass while the optical parameter allocation is scientifically incomplete.

## 2. Exact current keystone and inverse

Let \(B_j\) denote a fixed reference P4 vertex, \(R\) the reference P4 RMS radius, \(\mathbf t_i=(\theta_{xi}/u_x,\theta_{yi}/u_{y,c})\), and \(c\) the capture index.

The Stage-03 forward model is:

\[
P_{ij}=(1+\kappa_c\|B_j\|^2)B_j,\qquad
a_i=k_{0,c}t_{xi}^2-k_{1,c}t_{yi}^2,
\]

\[
d_{ij}=1+
k_{2,c}t_{xi}P_{ij,y}/R+
k_{3,c}t_{yi}P_{ij,x}/R,
\]

\[
F_{ij}=\frac{\operatorname{diag}(e^{a_i},e^{-a_i})P_{ij}}{d_{ij}},
\qquad
\widehat X_{4ij}=g_i\big(F_{ij}-\overline F_i\big).
\]

The four \(k_{\cdot,c}\) (and for nonreference captures \(\kappa_c\)) are estimated by minimizing corresponding-vertex error in original centered camera pixels with equal fixation weighting.

**There is no separately fitted inverse operator.** The saved forward coefficients are used in the deterministic inverse:

\[
z_{ij}=X_{4ij}/g_i+\overline F_i,\qquad
u_{ij}=\operatorname{diag}(e^{-a_i},e^{a_i})z_{ij},
\]

\[
P_{ij}=
\frac{u_{ij}}{1-(k_{3,c}t_{yi}/R)u_{ij,x}-(k_{2,c}t_{xi}/R)u_{ij,y}},
\]

followed by inversion of \(r_d=r+\kappa r^3\) on the valid radial branch. The forward/inverse **raw** keystone no longer contains any RMS-size or area renormalization. Do not reintroduce it.

The risk arises because the *forward-calibrated* \(K_{4,c}\) coefficients were allowed to absorb capture- or accommodation-dependent optical changes; the inverse simply applies that calibration exactly.

## 3. Quantitative evidence, and what it does not establish

The fitted **horizontal quadratic P4 keystone** coefficient is comparable across captures because the horizontal scaled unit is consistently 10 degrees. In native degree units:

| Capture | Nominal demand | Native P4 horizontal quadratic coefficient (deg^-2) |
|---|---:|---:|
| 1 | 0.36036 D | -1.432657e-5 |
| 4 | 2 D | +1.076269e-5 |
| 3 | 3 D | +2.760664e-5 |
| 2 | 4 D | +6.252566e-5 |

That is a strong *capture/demand association*, worth testing as a minimal shared accommodation-dependent \(k_{0}(A)\). It is **not** proof that the crystalline lens alone caused the trend.

Important implementation details:

- Other scaled coefficients, especially those involving vertical gaze, are **not directly comparable** across captures because \(u_{y,c}\) varies. Convert to a single native-angle basis before any global regression, plotting, fitting, or bound comparison.
- Capture 2's P4 vertical projective coefficient is at its lower bound. Treat it as a model-fit warning, not a precise physical estimate.
- The fitted capture-specific radial coefficients were approximately \([0,-1.85421,-1.47467,-0.847917]\times10^{-6}\,\mathrm{px}^{-2}\) in capture order 1/2/3/4. The post-fit law is \(\kappa(A)=-5.253231191989352\times10^{-7}(A-A_{ref})\,\mathrm{px}^{-2}\), with \(A_{ref}=0.36036036036036034\,D\). Its coefficients must be **refitted** when changing \(K_4\); the old law is not an independent ground truth.
- Capture 1 Stage-06 fixation-mean \(A\) stays near \(0.567,0.587,0.362,0.214,0.244\,D\) for \(-10,-5,0,+5,+10\) degrees while nominal demand remains \(0.36036\,D\); the gaze association remains large (~0.373 D peak-to-trough).
- Raw Stage-06 P4 forward RMS, before-to-after, was 1.162->1.175 (Capture 1), 3.620->3.594 (Capture 2), 1.836->1.753 (Capture 3), and 2.984->2.985 px (Capture 4). Inverse tails worsen in Capture 2 (5.696->5.965 px).
- Stage 06 independently recalibrates gaze per capture, a separate potential compensation mechanism. Hold that policy fixed during the **first** shared-keystone ablation so any observed change can be attributed to optical-parameter sharing rather than a simultaneously changed gaze fit.

## 4. Identifiability and alternative explanations

**The absorption claim is a model-structure risk, not a measured numerical decomposition.** Independent fits \(K_{4,c}\) and \(\kappa_c\) have overlapping ways of explaining some pattern changes. We have not yet performed a conditional/profiled rank test establishing how much residual information remains for \(A\) once shared keystone is enforced.

**Demand equals capture identity in this data.** One nominal accommodation level occurs in each of four recordings. A term \(k_A(A-A_{ref})\), arbitrary per-capture keystone offsets, capture drift, changes in gaze scale, and changes in optical geometry can be confounded. The present recordings can support a constrained shared-law comparison, not a definitive causal physiological calibration.

**P1 scale is a nuisance gauge.** P1 provides one positive \(g_i\), and that exact \(g_i\) must be shared with P4. Do not add P4-only scale or radius normalization to hide mismatch. Reprofiling P1 scale at every trial gaze is mandatory when gaze is adjusted.

**Keystone size change is physically allowed.** Use \(C(K_r)\) directly. No RMS or area normalization, no equilateralization, and no arbitrary unconstrained isotropic gaze function.

**Inverse optimization is not independent evidence.** Report original-camera-coordinate residuals first, then inverse-reference errors. A biased forward model can invert itself to machine precision and still estimate the wrong physiological state.

**Gaze/displacement effects remain a separate hypothesis.** P4–P1 displacement and the full relative-coordinate theory are important, but this audit's first controlled test is **keystone coefficient sharing**. Do not simultaneously add a new \(D(\theta,A)\) law, rework gaze calibration, and change keystone, because the specific absorption hypothesis would become untestable.

## 5. Required corrective decision

1. Use one common P1 keystone calibration where the measurement setup warrants a shared optical system; evaluate any P1 capture variation independently rather than embedding it silently.
2. Replace \(K_{4,c}\) in the *authoritative* model with one shared \(K_4(\theta)\). Fit across all captures with a consistent native gaze-angle basis and the same fixed empirical reference.
3. Compare against one minimal **global** accommodation-dependent coefficient, e.g. \(k_0(A)=k_{0,ref}+k_{0,A}(A-A_{ref})\), while retaining the same radial and observation-model conventions.
4. Refit the shared radial response jointly or through a clearly declared consistent staged procedure under each keystone model. Do not freeze old \(\kappa_c\) or \(\kappa(A)\).
5. Re-estimate framewise \(A\) under the accepted global model, including the full \(\partial_A K_4\) contribution when present. Keep the same fitting-frame population, mean anchors, and model convention for an interpretable paired comparison.
6. Evaluate native forward residuals, gaze-conditioned \(A\) trends, held-out predictive checks, global/state Jacobian rank and parameter correlations, active bounds, and inverse validity/tails. Require an explicit result before declaring that keystone had absorbed accommodation.

Implementation order, model controls, acceptance/reporting criteria, and reproducibility contracts are in [SHARED_KEYSTONE_CALIBRATION_PLAN.md](SHARED_KEYSTONE_CALIBRATION_PLAN.md).

## 6. Audit disposition

| Statement | Disposition |
|---|---|
| Per-capture P4 keystone was fitted independently in Stage 03 | **Confirmed** |
| Stage 04/05/06 freeze capture-specific P4 keystone | **Confirmed** |
| Raw keystone uses no extra RMS normalization | **Confirmed** |
| The current \(A\)-only derivative does not include explicit \(\partial_A K_4\) | **Confirmed** |
| Across-capture coefficient differences reflect real accommodation effects | **Not established** |
| Independent keystone has removed a quantifiable fraction of accommodation information | **Not yet measured** |
| A shared \(K_4(\theta,A)\) will recover physiological framewise accommodation accurately | **Not yet demonstrated** |

**Next action:** implement the planned **global shared-keystone ablation** on the frozen raw Stage-06 baseline, without modifying historical canonical outputs.