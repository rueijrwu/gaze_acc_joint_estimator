# Final P4 calibration model and protocol

This calibration describes the current measured triangles. Remaining irregular accommodation offsets at each capture–gaze fixation are accepted as calibration artifacts. The objective is minimum equal-weight Cartesian vertex RMS with shared optical parameters. Offset magnitude, smoothness, monotonicity and capture transfer are diagnostic outputs, not completion gates.

## Fixed data and reference

Use the existing 20 fixation means: four captures × five nominal gazes (−10°, −5°, 0°, +5°, +10°). Each frame's P4 triangle is centered and divided by its saved P1 magnification **before** averaging. Reuse the saved per-capture horizontal P4−P1 quadratic gaze estimator, calibrated on fixation-mean displacements; use the mean of its frame estimates for optical calibration. Freeze the existing common refined reference Q, anchored to C1 at nominal zero. Its measured baseline distortion remains part of this empirical reference.

For the assumed target plane normal β=0°, use A_dem,cg=A_c0 cos(θ_target,g), with trigonometric arguments converted to radians. A target label supplies the geometry; polynomial gaze supplies the optical angle. Central demands are 0.36036036036, 4, 3 and 2 D. A_ref=0.36036036036 D. Accommodation and gaze remain fixed during optical fitting. This dataset calibrates horizontal gaze dependence; vertical targets are all zero.

## Final candidate: normalized asymmetric keystone with independent size terms

Use normalized coordinates Q/R and origins O/R, with the existing fixed R; define a=(A−A_ref)/(4−A_ref) and t=θ/(10°). All equations below use these dimensionless coordinates.

$$B_j=Q_j+k a\|Q_j-O_B\|^2(Q_j-O_B).$$

$$z_j=R_\phi^T(B_j-O_K),\qquad H_j=O_K+R_\phi\frac{\operatorname{diag}(e^d,e^{-d})z_j}{1+qz_{j,y}}.$$

$$d=(d_{10}+d_{1A}a)t+(d_{20}+d_{2A}a)t^2,\qquad q=(q_0+q_Aa)t.$$

$$\boxed{\log M=(u_0+u_Aa)t+(v_0+v_Aa)t^2.}$$

$$\boxed{F(Q,A,\theta)=M\frac{r(B)}{r(H)}\mathcal C[H],\qquad \widehat X_{px}=R F.}$$

C subtracts the triangle centroid. r is centered RMS radius. The normalized shape map preserves the predicted barrel radius, and M supplies the entire total keystone radius gain. Normalization uses only predicted coordinates. M(A,0)=1; F(Q,A_ref,0)=Q. Forward order is barrel then keystone; coordinate inversion restores the predicted pre-centering centroid, analytically inverts the effective keystone, then numerically inverts the nonfolding radial barrel branch.

There are 16 shared parameters: barrel slope and origin (3), anisotropy amplitudes (4), projective amplitudes (2), keystone origin and orientation (3), and size coefficients (4). O_B and O_K are separately free but each shared across all captures and gazes. Origins are effective calibration parameters. No fitted optical symmetry angle is required. Positive stretch, projective denominator and radial Jacobian are required over the evaluated state domain.

## Existing benchmark

Also replay the existing 16-parameter oriented asymmetric keystone with symmetric barrel on exactly the same means, reference, demands and gaze:

$$B=Q+k a\|Q-O_B\|^2(Q-O_B),\quad z=R_\phi^T(B-O_K),\quad F=\mathcal C\left[O_K+R_\phi\frac{\operatorname{diag}(s_x,s_y)z}{1+qz_y}\right].$$

$$s_x=1+(\ell_{x0}+\ell_{xA}a)t+(\alpha_{x0}+\alpha_{xA}a)t^2,$$

$$s_y=1+(\ell_{y0}+\ell_{yA}a)t+(\alpha_{y0}+\alpha_{yA}a)t^2,\qquad q=(q_0+q_Aa)t.$$

Its saved valid converged vertex RMS is 0.793512 px. Replay the 15-parameter shifted-quadratic normalized-size model as well (1.086778 px). These are different empirical families; normalization is not assumed to make their parameterizations equivalent.

## Frozen final protocol

1. Copy the established mean and frame datasets with hashes; verify the frame means, gaze means and counts reproduce the 20 calibration inputs. Keep frame P1 magnification, gaze, masks and vertex correspondence fixed.
2. Freeze Q and the reference gauge for every candidate. Use corrected demands during fitting.
3. Minimize E=sqrt(sum_cgj ||R F_cgj−X_cgj||²/60). Each fixation has equal weight, irrespective of its number of frames. Retain measured size; do not align, rotate or radius-match residual triangles.
4. Run one predetermined multistart set using CuPy float64 model evaluations and batched complex-step derivatives. Initialize from existing origin/orientation branches. The shifted law is nested exactly: u0=−2s c0, uA=−2s cA, v0=c0, vA=cA, s=θ_s/10°. Save every start, its convergence and feasibility. Use a fixed iteration limit and select the lowest-RMS converged valid final-candidate solution.
5. Independently verify pixel replay, CPU/GPU agreement, derivatives, reference identity, shape-radius preservation, total gain M, inverse roundtrip, and positive branch over 0–5 D and the calibration gaze interval. Check validity at every evaluated frame state and flag extrapolation from that interval.
6. Replay existing benchmark predictions and choose the lowest-RMS verified candidate. Freeze the winner. No accommodation-offset penalty or free fixation offset enters this choice.
7. For each mean and frame, numerically infer A by minimizing the full frozen forward vertex error over 0–5 D at its fixed polynomial gaze. Scan the domain, refine all grid-resolved minima and compare endpoints. Recompute all accommodation-dependent terms and the predicted normalization at each trial A. Save boundaries and validity flags. State inversion changes only A, not the optical calibration.
8. Save a Markdown report with model, parameter tables, global/per-fixation RMS, and the 20 accepted mean accommodation offsets A_est−A_dem. Plot 1×5 triangle overlays across all captures with scale retained, inverse-corrected triangles against Q, and frame accommodation versus time colored by fixation. Export model and frame estimates as reusable JSON/NPZ/pickle with provenance.

Completion: choose the lowest-RMS converged valid solution from this fixed comparison, pass numerical checks and freeze its parameters. Declare this calibration complete. Remaining gaze offsets are accepted artifacts and do not trigger further model extensions. Report any unresolved start honestly; a finite multistart search establishes the best accepted solution found, not a proof of a global minimum.

[Final experiment](../experiments/archive/p4_final_linear_quadratic_size/README.md)
