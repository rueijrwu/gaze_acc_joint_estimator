# Reverse-transform next-step audit and experiment sequence

**Branch:** `exp5_distortion_model`  
**Basis:** Audit and recommendations already completed for reverse-transform stages 01–04.  
**Purpose:** Preserve the agreed findings and define the next controlled experiment. This document does not add a new audit or claim new results.

## 1. Current conclusion

The reverse-transform implementation is numerically sound enough to proceed. Stage 04 demonstrates that allowing one framewise accommodation-like state reduces both inverse-reference and original-camera residuals, but the fitted A retains a systematic gaze dependence.

The next task is therefore **not** to make the accommodation law more flexible. It is to determine whether the current A state is compensating residual gaze/keystone error.

The next experiment should use **Capture 1 only** and jointly investigate gaze and accommodation while keeping the optical model fixed.

## 2. Key implementation finding

The current keystone operator is normalized to preserve the RMS size of the pre-keystone triangle.

Schematically,

[
F=K(	heta;B),qquad
H(	heta)=rac{C(F)}{R[C(F)]/R[B]},
]

so

[
R[H(	heta)]=R[B].
]

For P4 with accommodation/radial pre-transform (P(A)),

[
R[H_4(	heta,A)]=R[P(A)].
]

After applying the P1 frame scale,

[
widehat X_{4i}=M_{1i}H_4(	heta_i,A_i),
]

and therefore

[
R[widehat X_{4i}]=M_{1i}R[P(A_i)].
]

Under this convention, gaze/keystone can change P4 **shape** but cannot change its total RMS size relative to the P1 scale.

The Capture-1 data retain a gaze-dependent P4/reference radius after P1 scale and keystone correction:

| Nominal gaze | P4/reference RMS radius |
|---:|---:|
| -10 deg | 0.99137 |
| -5 deg | 0.99040 |
| 0 deg | 1.00003 |
| +5 deg | 1.00713 |
| +10 deg | 1.00644 |

Stage 04 simultaneously estimates Capture-1 fixation-mean A:

| Nominal gaze | fitted A |
|---:|---:|
| -10 deg | 0.566 D |
| -5 deg | 0.592 D |
| 0 deg | 0.359 D |
| +5 deg | 0.196 D |
| +10 deg | 0.207 D |

This does not prove that gaze causes the fitted A trend, but it gives a specific mechanism to test: the A/radial state can absorb relative P4 size changes that the current size-normalized keystone cannot represent as gaze.

## 3. Interpretation of the current radial coefficient

The empirical P4 reference is non-equilateral. Its vertex radii about the present empirical center are approximately

[
190.96,quad198.59,quad183.93 {m px}.
]

Thus radial deformation is not exactly identical to uniform scale.

However, the three radii are fairly similar. A local geometry check found the centered uniform-scale and first radial-deformation directions to be highly correlated (about 0.999, roughly 2.5 degrees apart).

Therefore the present kappa should be interpreted conservatively as

[
oxed{	ext{effective relative P4 scale + radial-shape deformation}}
]

rather than a separately established physical barrel coefficient.

The immediate goal is to determine whether this effective deformation remains necessary after gaze is jointly corrected.

## 4. Next experiment: Capture-1 state ablation

Use the same Capture-1 complete-frame population and the same fixed empirical reference.

Compare three models.

### A0 — existing A-only baseline

Preserve the current Stage-04 Capture-1 result exactly.

Gaze remains the Stage-01 estimate and A varies per frame through the frozen radial law.

Do not rerun this baseline under a different objective and call it the same result.

### G — gaze-only control

Fix

[
A_i=A_{ref},
qquad
kappa_i=0
]

for Capture 1.

Estimate only horizontal gaze:

[
	heta_{x,i}=	heta^{(0)}_{x,i}+Delta	heta_{x,i}.
]

Keep vertical gaze frozen.

This tests whether correcting gaze alone can explain the residual previously absorbed by A.

### GA — joint gaze + accommodation

Freeze the existing radial response law

[
oxed{
kappa(A)=
eta(A-A_{ref})
}
]

with

[
eta=-5.25521030136462	imes10^{-7} {m px^{-2}/D},
]

[
A_{ref}=0.36036036036036034 {m D}.
]

Estimate per frame

[
oxed{x_i=(	heta_{x,i},A_i).}
]

Do not fit a free framewise kappa and do not refit beta in this experiment.

The important comparison is **G versus GA**.

If "known k for Capture 1" means fixing kappa=0 for every frame, that is the G control; A then has no optical effect and is not estimated.

## 5. Recompute P1 magnification whenever gaze changes

This is required.

The P1 frame scale is conditional on gaze:

[
oxed{
M_i(	heta)=
rac{
langle X_{1i},H_1(	heta)angle
}{
langle H_1(	heta),H_1(	heta)angle
}.
}
]

When a trial gaze changes, the P1 predicted shape changes. Therefore (M_i) must be recomputed at that trial gaze.

Do **not** free gaze while keeping the saved Stage-03/04 P1 magnification fixed.

The correct framewise sequence is:

[
oxed{
	heta_i
ightarrow
H_1(	heta_i)
ightarrow
M_i(	heta_i)
ightarrow
H_4(	heta_i,A_i)
ightarrow
	ext{P1/P4 residual}.
}
]

This allows P1 to constrain gaze while still supplying the common nuisance magnification.

## 6. Freeze all global optical parameters

For the first G/GA experiment, freeze:

- P1 reference triangle;
- P4 reference triangle;
- reference center/origin convention;
- P1 keystone coefficients;
- P4 keystone coefficients;
- gaze units;
- radial origin convention;
- (A_{ref});
- beta;
- point correspondence;
- fixation/population definitions.

Do not refit keystone at the same time as framewise gaze.

Do not introduce a P4-specific free magnification.

This keeps the experiment interpretable: only the state allocation between gaze and accommodation changes.

## 7. Estimate horizontal gaze only

The calibration contains five horizontal gaze targets but no independent vertical gaze targets.

Current vertical gaze is based on the shared first-order x/y centroid-slope assumption. Freeing vertical gaze at the same time would introduce another poorly constrained compensation direction.

Therefore the first joint state should be

[
oxed{x_i=(	heta_{x,i},A_i)}
]

with (	heta_y) frozen.

Vertical gaze can be revisited only after horizontal gaze/A separation is understood.

## 8. Fitting metric

Use original centered camera coordinates as the primary fitting/comparison space.

For each frame,

[
X_{1i}=C(P1_i),
qquad
X_{4i}=C(P4_i).
]

At a trial state,

[
widehat X_{1i}=M_i(	heta_i)H_1(	heta_i),
]

[
widehat X_{4i}=M_i(	heta_i)H_4(	heta_i,A_i).
]

Use an equal-fixation forward residual objective, with P1 and P4 components retained separately in reporting.

A suitable structure is

[
J_{data}
=
rac15sum_frac1{N_f}sum_{iin f}
left[
w_1|X_{1i}-widehat X_{1i}|^2
+
w_4|X_{4i}-widehat X_{4i}|^2
ight].
]

Use one predeclared weighting policy. Do not tune P1/P4 weights after looking at the desired state result.

Continue to evaluate the existing inverse-reference metric as a major secondary diagnostic.

A credible result should improve or preserve both:

1. original-camera forward prediction;
2. inverse recovery to the empirical reference.

## 9. Soft calibration anchors

Nominal fixation labels are calibration constraints, not framewise truth.

Use only fixation-mean anchors:

[
J_	heta
=
lambda_	heta
rac15sum_f
(ar	heta_f-	heta_f^{nom})^2,
]

[
J_A
=
lambda_A
rac15sum_f
(ar A_f-A_{ref})^2.
]

For the first GA comparison, retain the existing 0.25-D A mean-anchor width unless anchor sensitivity is explicitly being tested.

Choose the gaze mean-anchor width before running the comparison.

Do not impose framewise gaze or accommodation smoothing/prior in the primary diagnostic.

## 10. Gaze/accommodation identifiability diagnostic

For each frame, or a declared representative sample, evaluate

[
J_i=
egin{bmatrix}
partialmathbf e_i/partial	heta_x&
partialmathbf e_i/partial A
end{bmatrix}.
]

Report:

- singular values;
- condition number;
- normalized column correlation;
- angle between gaze and A response directions;
- distributions by nominal gaze;
- behavior near state/inverse-domain bounds.

Evaluate this for:

1. P4-only residuals;
2. joint P1+P4 residuals.

The Stage-04 positive A curvature proves the one-state problem is locally well behaved. It does not prove that theta and A are separable when both are free.

## 11. Required A0 / G / GA comparison

Use the exact same Capture-1 frame population.

For each fixation report:

- mean and SD of fitted horizontal gaze;
- mean delta-gaze from Stage-01;
- mean and SD of A for A0 and GA;
- state-bound counts;
- correlation between delta-gaze and delta-A;
- P1 forward RMS/median/P95;
- P4 forward RMS/median/P95;
- signed P4 x/y residuals by vertex;
- P4/reference radius ratio;
- inverse-reference RMS/median/P95;
- local theta/A conditioning.

The key qualitative result is whether the strong fixation-dependent A trend decreases when gaze is free.

## 12. Decision sequence

### Case 1 — G explains most of the Stage-04 gain

Interpret much of Stage-04 A variation as compensation for gaze/model error.

Next work should improve gaze calibration or gaze-dependent P4 transformation, not accommodation-law complexity.

### Case 2 — GA clearly outperforms G and theta/A remain identifiable

This is the strongest evidence that an accommodation-like deformation remains after gaze correction.

Next run the held-P4/cross-agreement test using the complete joint state estimator.

### Case 3 — GA lowers residual but theta/A are nearly collinear

Classify accommodation identifiability as unresolved.

Do not add another response exponent or more framewise parameters. Add independent information or improve the optical constraint.

### Case 4 — A retains the same gaze trend after gaze is free

Then ordinary framewise gaze correction is insufficient.

The next controlled candidate should be a **P4/P1 differential gaze-scale term**, because the current normalized keystone mathematically cannot express gaze-dependent total relative P4 size.

For example, only as a later ablation,

[
s_{41}(	heta_x)=1+c_1	heta_x+c_2	heta_x^2.
]

Do not introduce this term in the first G/GA experiment.

## 13. Cross-agreement comes after state separation

Do not start by adding more optical terms.

First establish the A0/G/GA result on Capture 1.

If GA survives the gaze-only control and is reasonably identifiable, freeze the global model and run the three-way omitted-P4 procedure:

1. omit one P4;
2. rerun the full joint theta/A inference using the retained measurements;
3. predict the omitted P4 in original relative camera coordinates;
4. rotate through all three omissions;
5. compare same-frame theta and A estimates across subsets.

Cross-agreement should then become the main internal validation of the joint estimator.

## 14. Recommended immediate sequence

[
oxed{
egin{array}{l}
	extbf{Step 1:} 	ext{ Preserve Stage-04 A-only baseline.}\
	extbf{Step 2:} 	ext{ Implement Capture-1 gaze-only G control.}\
	extbf{Step 3:} 	ext{ Implement Capture-1 joint GA with frozen }kappa(A).\
	extbf{Step 4:} 	ext{ Recompute P1 M at every trial gaze.}\
	extbf{Step 5:} 	ext{ Compare forward + inverse metrics on identical frames.}\
	extbf{Step 6:} 	ext{ Audit local theta/A identifiability.}\
	extbf{Step 7:} 	ext{ Decide whether residual is gaze, joint A, or missing gaze-scale physics.}\
	extbf{Step 8:} 	ext{ Only if GA survives, proceed to held-P4 cross-agreement.}
end{array}
}
]

The next stage should remain a **diagnostic separation experiment**, not a full model expansion.
