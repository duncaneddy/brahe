# Attitude Representations

Brahe supports multiple mathematical representations for the attitude, or orientation, of 3D objects such as spacecraft. Each representation has its own advantages and disadvantages depending on the application. These representations are implemented based on the comprehensive treatment found in [Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors](https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf) by James Diebel.

## Overview

Attitude representation is fundamental to spacecraft dynamics and control. Brahe provides four different representations, each with their own advantages:

- **[Quaternions](quaternions.md)**: Singularity-free, compact representation (4 parameters)
- **[Rotation Matrices](rotation_matrices.md)**: Direct transformation matrices (9 parameters)
- **[Euler Angles](euler_angles.md)**: Intuitive angular representation (3 parameters, but with singularities)
- **[Euler Axis](euler_axis.md)**: Axis-angle representation (4 parameters)

## Conversions

Brahe provides functions to convert between all attitude representations. You can initialize an attitude in one representation and convert it to any other one as needed.
<div class="plotly-embed">
    <img class="only-light" src="../../assets/attitude_representations_light.svg" alt="Attitude representations diagram" loading="lazy">
    <img class="only-dark" src="../../assets/attitude_representations_dark.svg" alt="Attitude representations diagram" loading="lazy">
</div>

## Choosing a Representation

**Use Quaternions when:**

- Numerical stability is critical
- Interpolating between attitudes
- Propagating attitude dynamics

**Use Rotation Matrices when:**

- Transforming vectors between frames

**Use Euler Angles when:**

- Human readability is important

**Use Euler Axis when:**

- Representing single rotations about an axis

## Attitude Frames

An attitude relates two frames: it is the passive rotation taking vector
components in a source frame A to components in a target frame B. Both
endpoints are `ReferenceFrame` values. A `Celestial` endpoint is any frame the
frame transformation system evaluates from an epoch alone (GCRF, ITRF,
EME2000, and the other members of `CelestialFrame`); it composes directly with
`rotation_frame_to_frame`. An `OrbitRelative` endpoint is a local orbital frame
such as RTN or LVLH, defined only given an orbit state. A `Body` endpoint is an
object-local frame — a spacecraft body, sensor, or actuator frame — whose
orientation the attitude data itself supplies.

CCSDS attitude messages (APM, AEM) declare their frame pair with these
semantics, and brahe's CCSDS module converts between the CCSDS frame vocabulary
and `ReferenceFrame` where a native equivalent exists. A CCSDS frame keyword
names the frame but not the object it belongs to, so orbit-relative and body
keywords convert to unbound endpoints; binding them to an object is the
caller's job.

## Attitude Kinematics

Four functions relate an attitude representation's time derivative to angular velocity: `quaternion_derivative`, `angular_velocity_from_quaternion_derivative`, `euler_rates_to_angular_velocity`, and `angular_velocity_to_euler_rates`. Given an attitude (quaternion or Euler angle set) and either its time derivative or the corresponding angular velocity, these functions compute the other quantity. Angular velocity is always $\omega$, the angular velocity of frame B relative to frame A, expressed in frame B, in rad/s, where A and B are the source and target frames of the attitude — the same passive rotation the attitude itself represents.

All four functions are grounded in Diebel (2006), *Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors*, the same reference the other attitude representations are built from. Diebel's quaternion product and his Euler-angle sequence labels do not match brahe's conventions directly. Both differences are fixed exchanges, derived once here; every cited Diebel equation number in the kinematics API documentation refers to Diebel's printed equation with these two exchanges already applied.

### Product convention: Hamilton product vs. Diebel's product

Brahe's quaternion product (`Quaternion`'s multiplication operator, and the corresponding operation on raw, non-unit components used internally by the kinematics functions) is the Hamilton product. For $q = (s, \mathbf{u})$ and $p = (t, \mathbf{v})$, with $s, t$ the scalar parts and $\mathbf{u}, \mathbf{v}$ the vector parts:

$$q * p = \left(st - \mathbf{u} \cdot \mathbf{v},\ \ s\mathbf{v} + t\mathbf{u} + \mathbf{u} \times \mathbf{v}\right)$$

Diebel's product (eq. 102), printed as $q \cdot p$, carries the same scalar part but the opposite sign on the cross-product term:

$$q \cdot p = \left(st - \mathbf{u} \cdot \mathbf{v},\ \ s\mathbf{v} + t\mathbf{u} - \mathbf{u} \times \mathbf{v}\right)$$

The two products differ only in the sign of $\mathbf{u} \times \mathbf{v}$, which is exactly the effect of exchanging the operands: $q * p = p \cdot q$. Every Diebel equation cited below is restated under this exchange — Diebel's $q \cdot p$ becomes brahe's $p * q$ — before being implemented.

### Sequence labeling: application order vs. Diebel's matrix order

Brahe labels an Euler-angle sequence `ABC` by application order: the rotation about axis A is applied first, then B, then C. Diebel's eq. 34 labels a sequence $R_{ijk}(\phi, \theta, \psi) := R_i(\phi) R_j(\theta) R_k(\psi)$ by left-to-right matrix order, with the angle vector $u = [\phi, \theta, \psi]$ listed in that same order. Because a passive rotation matrix product applies its rightmost factor first, the rotation order for $R_{ijk}$ is $k \to j \to i$ — the reverse of the label.

For brahe order `ABC` with angles $(\phi, \theta, \psi)$, the equivalent Diebel sequence is therefore $(i, j, k) = (C, B, A)$ with Diebel-order angles $(\phi_D, \theta_D, \psi_D) = (\psi, \theta, \phi)$. This is the same relabeling `EulerAngleOrder::reversed()` performs for the existing quaternion and rotation-matrix conversions (see [Euler Angles](euler_angles.md)); the kinematics functions apply it to Diebel's Euler-angle rate matrices in the same way.

### Kinematic equations in brahe form

With both exchanges applied, `quaternion_derivative` computes the attitude quaternion's time derivative from the body-frame angular velocity as

$$\dot{q} = \frac{1}{2}\, q * \bar{\omega}$$

where $\bar{\omega} = (0, \omega)$. This is Diebel's eq. 157, $\dot q = \tfrac12\,[0;\omega']\cdot q$, restated under the product exchange: the body-frame rate $\omega$, which left-multiplies under Diebel's product, right-multiplies under brahe's.

`angular_velocity_from_quaternion_derivative` inverts this relation:

$$\bar{\omega} = 2\, \bar{q} * \dot{q}$$

from Diebel's eq. 147, $[0;\omega'] = 2\,\dot q \cdot \bar q$, restated the same way, with $\bar q$ the quaternion conjugate.

`euler_rates_to_angular_velocity` and its inverse `angular_velocity_to_euler_rates` compute the body-frame angular velocity from Euler-angle rates, and back, as

$$\omega = E'\, \dot{u}$$

from Diebel's eqs. 38 and 40, $\omega' = E'_{ijk}(u)\,\dot u$ with $E'_{ijk}(u) = \left[\hat e_i,\ R_i(\phi_D)\hat e_j,\ R_i(\phi_D) R_j(\theta_D)\hat e_k\right]$. $E'$ is built and applied entirely in the relabeled Diebel sequence $(i,j,k)$ and angle order $u = (\phi_D, \theta_D, \psi_D)$; the rates $\dot u$ and the resulting angle-rate vector are reordered between brahe's $(\dot\phi, \dot\theta, \dot\psi)$ and Diebel's $(\dot\phi_D, \dot\theta_D, \dot\psi_D) = (\dot\psi, \dot\theta, \dot\phi)$ at the boundary of each function.

### Gimbal lock

`angular_velocity_to_euler_rates` inverts $E'$, which is singular at the sequence's gimbal-lock condition. The singularity location depends on the sequence family: Tait-Bryan sequences (three distinct axes, e.g. `XYZ`, `ZYX`) have $\det E' = \pm\cos\theta$ and are singular at $\theta = \pm 90°$; symmetric sequences (repeated first and third axis, e.g. `ZXZ`, `XYX`) have $\det E' = \pm\sin\theta$ and are singular at $\theta = 0°$ or $180°$. Both cases match Diebel §5's per-sequence singularity statements. Near a singularity, `angular_velocity_to_euler_rates` returns an error rather than an ill-conditioned result.

---

## See Also

- [API Reference - Attitude](../../library_api/attitude/index.md)
- [Frame Graph](../frames/frame_graph.md)
- [Kinematics](../../library_api/attitude/kinematics.md)
