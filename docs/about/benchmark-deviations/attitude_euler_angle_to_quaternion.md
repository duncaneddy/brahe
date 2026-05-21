# attitude.euler_angle_to_quaternion

_Last updated from accuracy run: 2026-05-21_

## Observed residual

- p99 max-abs error: 2.000e+00 (vs python)
- Worst-sample max-abs error: 2.000e+00
- Samples in sweep: 100
- Module threshold (p99): 1.000e-06

## Cause

Euler-angle convention mismatch identified.

**brahe** interprets `EulerAngleOrder::ZYX` with input `(phi, theta, psi)` as **extrinsic ZYX**: `R = Rx(psi) * Ry(theta) * Rz(phi)`. That is, phi rotates about the world-frame Z axis first, then Y, then X.

**Orekit / Basilisk / GMAT** in this benchmark interpret the same triplet as **intrinsic ZYX**: `R = Rz(phi) * Ry(theta) * Rx(psi)`. That is, phi rotates about Z, then theta about the new (body-frame) Y', then psi about the new X''.

These two conventions are *inverses* of each other — applying brahe's R then Orekit's R to the same vector yields different rotated vectors. That is why all backends agree with each other on this task (they all use intrinsic ZYX) but disagree with brahe (extrinsic ZYX).

Why the inverse task (`quaternion_to_euler_angle`) still agrees to machine epsilon: each library is internally consistent — brahe round-trips against its own extrinsic convention; the others round-trip against intrinsic. The benchmark feeds a random quaternion into each library and reads back the Euler triplet *in that library's own convention*. The output triplets differ between libraries, but they all map back to the same input quaternion when run through their own Euler→Quat, so the *quaternion-space* residual (which is what the benchmark measures) is zero.

Residuals are reported as the Frobenius norm of $R_a - R_b$; a value near 2 corresponds to roughly a 90° rotation gap, near 2√2 ≈ 2.83 corresponds to a 180° flip.

**Action**: brahe's `EulerAngle::from_euler_angle` for `ZYX` should be reviewed against the conventions used in the wider astrodynamics literature. Most aerospace texts (Wertz, Schaub/Junkins) document ZYX as the intrinsic order Rz·Ry·Rx; if brahe intends to follow that convention, the per-order quaternion formulas in `src/attitude/quaternion.rs` need to be rewritten. Out of scope for the benchmark redesign — filed as a brahe core issue.
