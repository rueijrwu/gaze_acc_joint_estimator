# Capture 1 P4 reflection: fixed-scale 2D keystone

This is a fresh capture1-only P4 reflection stage. It reuses the audited capture1 P1 gaze and framewise magnification exactly; there is no P4 magnification or accommodation state. The fixed P4 reference is the mean centered triangle in capture1's nominal zero-gaze interval, with RMS radius `R_ref = 191.254 px`. The capture1 relative radial coefficient is fixed at zero as the reference gauge. No demand or accommodation labels enter fitting.

The fitted model has four shared 2D keystone coefficients. With centered reference vertices `B_i`, RMS radius `R_B`, normalized gaze `t=theta/U`, and `a=k_ax theta_x^2-k_ay theta_y^2`, it computes

```
F_i = (exp(a) B_ix, exp(-a) B_iy)
      / (1 + k_px theta_x B_iy + k_py theta_y B_ix)
K_i = C(F)_i * R_B / sqrt(mean_j ||C(F)_j||^2)
P4_predicted_centered,i = M_P1(frame) * K_i
```

where `C` subtracts the three-vertex centroid. `K` is a model-only shape normalization to the fixed reference RMS radius; the measured triangles are not normalized for the forward fit. The one P1 magnification is frozen for P4. There is no P4-specific radial term in this capture1 reference fit.

The primary score is original centered P4 point SSE with equal total weight for each of the five fixation intervals. The report also gives a post-inverse shape diagnostic: after inversion, each recovered triangle is centered and scaled to the fixed reference RMS radius, then corresponding points are compared. For each inverse triangle `Q_i`, center it and compute `R_i = sqrt(mean_j ||C(Q_i)_j||^2)`, then use `s_i = R_ref/R_i` and compare `s_i C(Q_i)` with the fixed reference. This removes uniform size for the shape diagnostic only; it is not a fitted or applied P4 magnification and does not change the forward model.

Run from the repository root:

```bash
python experiments/reverse_transform/stage_02_capture1_p4_fit/scripts/run.py \
  --output experiments/reverse_transform/stage_02_capture1_p4_fit/results/run
python experiments/reverse_transform/stage_02_capture1_p4_fit/scripts/audit.py \
  --results experiments/reverse_transform/stage_02_capture1_p4_fit/results/run
python experiments/reverse_transform/stage_02_capture1_p4_fit/scripts/plot_results.py \
  --results experiments/reverse_transform/stage_02_capture1_p4_fit/results/run
```

The run and audit completed. See [results](results/run/RESULTS.md), the saved [audit](results/run/audit.json), and [scientific review](SCIENTIFIC_REVIEW.md). Plots are [point error by fixation](results/run/point_error_by_fixation.png), [shape error by fixation](results/run/shape_error_by_fixation.png), [triangle overlays](results/run/triangle_overlays.png), [normalized vertex clouds](results/run/normalized_vertex_clouds.png), and [radius distributions](results/run/radius_distributions.png).

This is conditional same-recording evidence. The empirical reference, initial gaze, and P1 magnification derive from the same capture; centroid-derived gaze is not framewise ground truth. The +5 degree interval shows reduced shape median but its P95 is lower only modestly, and the zero-gaze interval is worse after keystone inversion than before. No physical barrel coefficient, accommodation estimate, absolute optical origin, or physiological accuracy is established. Capture1 has only one nominal demand label, so no demand response is identifiable here.
