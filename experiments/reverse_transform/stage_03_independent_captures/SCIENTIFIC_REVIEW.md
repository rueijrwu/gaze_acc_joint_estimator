# Scientific review: independent capture fits

## Implementation audit

The image model receives estimated gaze, the common reference triangle, observed centered points, fixation weights, and P1 magnification. It receives no accommodation labels. Each capture has its own keystone parameters and one constant P4 radial increment; no coefficient is shared between captures. Capture 1's increment is fixed at zero because its measured reference already contains baseline distortion.

The common P1/P4 templates are the mean centered triangles from capture 1's nominal zero-gaze fixation at 0.3603603604 D. Capture 1's previously audited gaze and framewise P1 magnification are preserved exactly. Captures 2–4 receive independent P1 fits. P4 uses the corresponding P1 magnification without an additional fitted frame scale.

Keystone's overall size is included in the total P1 magnification convention. The P4 model preserves the constant radial deformation's size change. Inversion restores the forward model centroid before undoing the projective map, then solves the radial cubic on its monotone branch. It does not invert directly about the observed triangle centroid.

## Numerical evidence

The saved-results audit passed for all four captures: analytic gradients agree with finite differences within 6.23e-9, and synthetic forward/inverse closure is within 2.85e-13 px. CuPy and NumPy cost/gradient evaluations agree within 4e-15 and 1.3e-12 respectively. Six bounded starts are used to select each new fit, followed by a stationarity and curvature check. Capture 4's vertical quadratic keystone parameter is at its bound: its certificate is constrained stationarity, with positive curvature in the free parameter subspace.

All 89,175 complete frames enter the original centered-camera-coordinate forward objective, with equal weights for the five fixation intervals within each capture. No outlier is trimmed. There are 43 invalid radial inverses, all in capture 2. Inverse comparisons use the same valid frames for each diagnostic and explicitly report unavailable frames.

## Interpretation

After radius normalization, keystone correction improves median shape error for captures 1–3. Capture 4 worsens: median/p95 point distance rises from 1.785/3.064 px before correction to 1.927/4.067 px after keystone correction. Adding the fitted constant radial inverse gives 1.896/4.317 px, still worse than its baseline. Therefore the current keystone parameterization is not demonstrated to improve shape across all captures.

The independent relative radial coefficients are monotonic against demand labels: 0 at 0.36036 D, −0.8477e-6 at 2 D, −1.4728e-6 at 3 D, and −1.8569e-6 px⁻² at 4 D. The sole accommodation analysis is a descriptive straight line fitted to these coefficients afterward; its anchored slope is −5.2552e-7 px⁻²/D. This line is never used to fit image coordinates or compute their inverses.

These coefficients describe deformation relative to the empirical reference under fixed P1 scale and a fixed empirical radial origin. They can absorb P4 size changes and other capture differences; they do not identify physical barrel distortion or framewise accommodation. One recording per demand confounds demand with capture. Radius normalization is a diagnostic that removes uniform size, retains rotation and anisotropic shape, and does not remove general barrel deformation.

The corrected stage is complete as an empirical analysis. The capture 4 bound and shape deterioration remain unresolved before promoting this model to accommodation inference. No automated test suite was added or run; verification consists of the requested experiment and saved-results audit.
