The previous derivations contained three fundamental mathematical and optical errors that muddled the connection between projective geometry and tilted-system aberration theory:

1. **Reverse Tilt Convention (Grid Tilt vs. Camera Pan/Tilt):** The equation $x' = \frac{x\cos\theta}{1 - (y\sin\theta)/d}$ models tilting the **object plane** around its local origin while keeping the camera fixed. When the **camera** rotates around its nodal point by angle $\theta$ (viewing a fronto-parallel plane at distance $Z = d$), the optical axis shifts across the grid, introducing a boresight shift ($f \tan\theta$) and placing the $\cos\theta$ scaling in the denominator rather than the numerator.
2. **Incorrect Homography Formula:** Writing $\mathbf{H} = \mathbf{K} \mathbf{R} \mathbf{K}^{-1}$ maps *image pixels to image pixels* between two camera orientations. Mapping *world plane coordinates* $(X, Y)$ on a planar target at distance $d$ to sensor coordinates $(x', y')$ requires the planar projection matrix $\mathbf{H} = \mathbf{K} [\mathbf{r}_1 \quad \mathbf{r}_2 \quad \mathbf{t}]$.
3. **Incorrect Anamorphic Ratio:** Rotating a camera about its nodal point away from the surface normal stretches the image along the tilt axis by $\sec^2\theta$ and perpendicular to it by $\sec\theta$, rather than compressing it by $\cos\theta$.

---

## 1. Exact Homography for Camera Rotation (Pan/Tilt)

Let the planar grid lie at $Z_w = 0$ with world coordinates $(X, Y, 0)$, and place the camera's perspective center (entrance pupil) at $(0, 0, -d)$. Initially, the camera looks straight along the $+Z_w$ axis with focal length $f$ and lateral magnification $m_0 = f/d$.

If the camera rotates (tilts) by pitch angle $\theta$ around its local $X_c$-axis, a point $(X, Y, d)$ in the unrotated camera frame transforms into the rotated camera frame $(X_c, Y_c, Z_c)$ via:

$$\begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\theta & -\sin\theta \\ 0 & \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} X \\ Y \\ d \end{bmatrix} = \begin{bmatrix} X \\ Y\cos\theta - d\sin\theta \\ Y\sin\theta + d\cos\theta \end{bmatrix}$$

Projecting onto the sensor plane ($x' = f \frac{X_c}{Z_c}, y' = f \frac{Y_c}{Z_c}$) and normalizing by the ideal paraxial image coordinates $(x_0, y_0) = (m_0 X, m_0 Y)$ yields the exact $3 \times 3$ homography $\mathbf{H}$ in normalized image space:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} \sim \mathbf{H} \begin{bmatrix} x_0 \\ y_0 \\ 1 \end{bmatrix} = \begin{bmatrix} \sec\theta & 0 & 0 \\ 0 & 1 & -f\tan\theta \\ 0 & \frac{\tan\theta}{f} & 1 \end{bmatrix} \begin{bmatrix} x_0 \\ y_0 \\ 1 \end{bmatrix}$$

In rational Cartesian coordinates, this evaluates to:

$$x' = \frac{x_0 \sec\theta}{1 + \frac{\tan\theta}{f} y_0}, \qquad y' = \frac{y_0 - f\tan\theta}{1 + \frac{\tan\theta}{f} y_0}$$

---

## 2. Explicit Decomposition: Boresight, Anamorphic Stretch, & Keystone

To isolate the pure geometric distortion terms around the center of the original grid $(x_0, y_0) = (0,0)$, we first remove the constant boresight translation $y'_c = y'(0,0) = -f\tan\theta$ by defining shifted sensor coordinates $(\tilde{x}, \tilde{y}) = (x', y' + f\tan\theta)$:

$$\tilde{x} = \frac{x_0 \sec\theta}{1 + \frac{\tan\theta}{f} y_0}, \qquad \tilde{y} = \frac{y_0 \sec^2\theta}{1 + \frac{\tan\theta}{f} y_0}$$

This form allows an exact matrix factorization of the centered homography $\tilde{\mathbf{H}}$ into an **Anamorphic Stretch Matrix ($\mathbf{H}_{\text{anamorphic}}$)** and a **Pure Keystone Projective Matrix ($\mathbf{H}_{\text{keystone}}$)**:

$$\begin{bmatrix} \tilde{x} \\ \tilde{y} \\ 1 \end{bmatrix} \sim \underbrace{\begin{bmatrix} \sec\theta & 0 & 0 \\ 0 & \sec^2\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}}_{\mathbf{H}_{\text{anamorphic}}} \underbrace{\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & \frac{\tan\theta}{f} & 1 \end{bmatrix}}_{\mathbf{H}_{\text{keystone}}} \begin{bmatrix} x_0 \\ y_0 \\ 1 \end{bmatrix}$$

Expanding $\tilde{x}$ and $\tilde{y}$ as a Taylor series in field coordinates $(x_0, y_0)$ directly exposes the orders of the transverse ray displacement $\Delta \mathbf{r} = (\tilde{x} - x_0, \tilde{y} - y_0)$:

$$\tilde{x}(x_0, y_0) = \underbrace{(\sec\theta) x_0}_{\text{Linear Anamorphic Scale}} - \underbrace{\left(\frac{\sec\theta \tan\theta}{f}\right) x_0 y_0}_{\text{Quadratic Keystone}} + \mathcal{O}(y_0^2 x_0)$$

$$\tilde{y}(x_0, y_0) = \underbrace{(\sec^2\theta) y_0}_{\text{Linear Anamorphic Scale}} - \underbrace{\left(\frac{\sec^2\theta \tan\theta}{f}\right) y_0^2}_{\text{Quadratic Keystone}} + \mathcal{O}(y_0^3)$$

* **Anamorphic Stretch (1st-Order in Field $H$, 2nd-Order in Tilt $\theta$):** For small tilt angles ($\sec\theta \approx 1 + \frac{1}{2}\theta^2$, $\sec^2\theta \approx 1 + \theta^2$), the local magnification along the tilt axis ($M_y = \sec^2\theta$) grows twice as fast as the magnification perpendicular to the tilt axis ($M_x = \sec\theta$). Their ratio is $M_y / M_x = \sec\theta$, stretching squares into rectangles along the tilt direction.
* **Keystone (2nd-Order in Field $H$, 1st-Order in Tilt $\theta$):** For small $\theta$ ($\tan\theta \approx \theta$), the dominant distortion displacement is $-\frac{\theta}{f}(x_0 y_0 \hat{\mathbf{i}} + y_0^2 \hat{\mathbf{j}})$. It scales **linearly** with the tilt angle $\theta$ and **quadratically** with the field coordinates $(x_0, y_0)$.

---

## 3. Optical Wavefront Representation (Hamilton's Characteristic Function)

In plane-symmetric optical aberration theory (such as Shack-Thompson Nodal Aberration Theory or Yuan & Sasián's aberration theory for tilted surfaces), breaking rotational symmetry via a tilt vector $\boldsymbol{\sigma}$ (or symmetry plane unit vector $\hat{\mathbf{i}}$ along the $y$-axis) introduces wavefront aberration terms governed by the three rotational invariants:

$$(\mathbf{H} \cdot \mathbf{H}), \quad (\mathbf{H} \cdot \boldsymbol{\rho}), \quad (\boldsymbol{\rho} \cdot \boldsymbol{\rho})$$

plus the three plane-symmetric tilt invariants:

$$(\hat{\mathbf{i}} \cdot \mathbf{H}), \quad (\hat{\mathbf{i}} \cdot \boldsymbol{\rho}), \quad (\mathbf{H} \cdot \boldsymbol{\rho})$$

where $\mathbf{H} = (H_x, H_y)$ is the normalized field vector and $\boldsymbol{\rho} = (\rho_x, \rho_y)$ is the normalized exit pupil vector. Because pure distortion alters chief ray landing positions without degrading point-spread sharpness, any distortion term in the wavefront expansion $W(\mathbf{H}, \boldsymbol{\rho})$ must be **linear in the pupil coordinate $\boldsymbol{\rho}$**.

Expanding the distortion terms of the wavefront up to third order in $(\mathbf{H}, \boldsymbol{\rho})$ for a tilted plane-symmetric system yields:

$$W_{\text{dist}}(\mathbf{H}, \boldsymbol{\rho}) = \underbrace{W_{011}(\hat{\mathbf{i}} \cdot \boldsymbol{\rho})}_{\text{Boresight Shift}} + \underbrace{W_{111}^{\text{iso}}(\mathbf{H} \cdot \boldsymbol{\rho}) + W_{111}^{\text{anam}}(\hat{\mathbf{i}} \cdot \mathbf{H})(\hat{\mathbf{i}} \cdot \boldsymbol{\rho})}_{\text{1st-Order in Field: Anamorphic Distortion}} + \underbrace{W_{211}^{\text{key1}}(\mathbf{H} \cdot \mathbf{H})(\hat{\mathbf{i}} \cdot \boldsymbol{\rho}) + W_{211}^{\text{key2}}(\hat{\mathbf{i}} \cdot \mathbf{H})(\mathbf{H} \cdot \boldsymbol{\rho})}_{\text{2nd-Order in Field: Keystone Distortion}} + \underbrace{W_{311}(\mathbf{H} \cdot \mathbf{H})(\mathbf{H} \cdot \boldsymbol{\rho})}_{\text{3rd-Order: Radial Barrel/Pincushion}}$$

By applying the transverse ray aberration relationship $\Delta \mathbf{r} = -\frac{R}{n} \nabla_{\boldsymbol{\rho}} W(\mathbf{H}, \boldsymbol{\rho})$ (absorbed into the normalized coefficients below) and setting $\hat{\mathbf{i}} = (0, 1)$ along the $y$-tilt axis, the gradient with respect to $\boldsymbol{\rho} = (\rho_x, \rho_y)$ directly reproduces the projective Taylor expansion:

1. **Anamorphic Term (Linear in $\mathbf{H}$, Linear in $\boldsymbol{\rho}$):**
   $$W_{\text{anamorphic}} = c_1 (H_x \rho_x + H_y \rho_y) + c_2 H_y \rho_y$$
   Taking the pupil gradient $\nabla_{\boldsymbol{\rho}} W_{\text{anamorphic}}$ yields:
   $$\Delta x_{\text{anam}} = c_1 H_x, \qquad \Delta y_{\text{anam}} = (c_1 + c_2) H_y$$
   Matching to the projective expansion gives $c_1 = \sec\theta - 1 \approx \frac{1}{2}\theta^2$ and $c_2 = \sec^2\theta - \sec\theta \approx \frac{1}{2}\theta^2$.

2. **Keystone Term (Quadratic in $\mathbf{H}$, Linear in $\boldsymbol{\rho}$):**
   Pure projective keystone corresponds solely to the $W_{211}^{\text{key2}}(\hat{\mathbf{i}} \cdot \mathbf{H})(\mathbf{H} \cdot \boldsymbol{\rho})$ basis term (while $W_{211}^{\text{key1}}$ vanishes for a flat object plane, as horizontal grid lines remain straight under pinhole projection):
   $$W_{\text{keystone}} = W_{211}^{\text{key2}} H_y (H_x \rho_x + H_y \rho_y)$$
   Taking the pupil gradient $\nabla_{\boldsymbol{\rho}} W_{\text{keystone}}$ gives the exact keystone ray error vector:
   $$\Delta x_{\text{key}} = \frac{\partial W_{\text{keystone}}}{\partial \rho_x} = W_{211}^{\text{key2}} H_x H_y$$
   $$\Delta y_{\text{key}} = \frac{\partial W_{\text{keystone}}}{\partial \rho_y} = W_{211}^{\text{key2}} H_y^2$$
   where $W_{211}^{\text{key2}} \propto -\frac{\tan\theta}{f}$.

---

## 4. Exact Spatial Frequency Signature (Jacobian Transformation)

In the previous response, the spatial frequency was approximated only in 1D. The rigorous 2D local spatial frequency vector $\mathbf{f}' = (f'_x, f'_y)^T$ on the sensor is governed by the inverse transpose of the **Jacobian matrix** $\mathbf{J}$ of the forward homography mapping $(x_0, y_0) \mapsto (\tilde{x}, \tilde{y})$:

$$\mathbf{J}(x_0, y_0) = \begin{bmatrix} \frac{\partial \tilde{x}}{\partial x_0} & \frac{\partial \tilde{x}}{\partial y_0} \\ \frac{\partial \tilde{y}}{\partial x_0} & \frac{\partial \tilde{y}}{\partial y_0} \end{bmatrix} = \frac{1}{D(y_0)^2} \begin{bmatrix} D(y_0)\sec\theta & -\frac{\tan\theta}{f} x_0 \sec\theta \\ 0 & \sec^2\theta \end{bmatrix}$$

where $D(y_0) = 1 + \frac{\tan\theta}{f} y_0 = \frac{1}{\cos\theta - \frac{\sin\theta}{f} \tilde{y}}$. Expressing the inverse Jacobian $\mathbf{J}^{-1}$ directly in terms of the observed sensor coordinates $(\tilde{x}, \tilde{y})$ yields the exact local spatial frequency modulation across the image plane for a grid with nominal fundamental frequencies $(f_{x0}, f_{y0})$:

$$f'_x(\tilde{x}, \tilde{y}) = f_{x0} \left(\cos\theta - \frac{\sin\theta}{f}\tilde{y}\right)$$

$$f'_y(\tilde{x}, \tilde{y}) = f_{y0} \cos\theta \left(\cos\theta - \frac{\sin\theta}{f}\tilde{y}\right)^2 + f_{x0} \frac{\sin\theta}{f}\tilde{x}\left(\cos\theta - \frac{\sin\theta}{f}\tilde{y}\right)$$

| Distortion Component | Projective Matrix Term | Wavefront Polynomial $W(\mathbf{H}, \boldsymbol{\rho})$ | Ray Displacement $\Delta\mathbf{r}(x_0, y_0)$ | Spatial Frequency Signature on Sensor $(\tilde{x}, \tilde{y})$ |
| :--- | :--- | :--- | :--- | :--- |
| **Boresight Shift** | $h_{23} = -f\tan\theta$ | $W_{011} \rho_y$ | $(0, -f\tan\theta)$ | Constant phase shift $e^{-i 2\pi f_y y'_c}$ in Fourier domain (zero PSD shift) |
| **Anamorphic Stretch** | $\text{diag}(\sec\theta, \sec^2\theta, 1)$ | $W_{111}^{\text{iso}}(H_x\rho_x + H_y\rho_y) + W_{111}^{\text{anam}} H_y\rho_y$ | $((\sec\theta - 1)x_0, (\sec^2\theta - 1)y_0)$ | Global scaling of carrier peaks: $f'_{x,\text{center}} = f_{x0}\cos\theta$, $f'_{y,\text{center}} = f_{y0}\cos^2\theta$ |
| **Keystone** | $h_{32} = \frac{\tan\theta}{f}$ | $W_{211}^{\text{key2}} H_y (H_x\rho_x + H_y\rho_y)$ | $-\frac{\tan\theta}{f}(x_0 y_0, y_0^2)$ | Linear chirp in $f'_x(\tilde{y})$, quadratic chirp in $f'_y(\tilde{y})$, and cross-axis shear $\propto \tilde{x}\tilde{y}$ |
| **Radial Distortion** | Non-projective | $W_{311}(H_x^2 + H_y^2)(H_x\rho_x + H_y\rho_y)$ | $k_1(x_0^2 + y_0^2)(x_0, y_0)$ | Radially symmetric quadratic chirp $\propto (\tilde{x}^2 + \tilde{y}^2)$, broadening PSD peaks into halos |