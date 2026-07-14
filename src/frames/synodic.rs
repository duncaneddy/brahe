/*!
 * Synodic (two-body rotating) reference frames: the Earth-Moon Rotating
 * (EMR), Sun-Earth Rotating (SER), and Geocentric Solar Ecliptic (GSE)
 * frames defined in NASA TP-20220014814 §2.5, with the exact
 * (GTDS/STK-convention) rotation-matrix time derivative of §4.6.1 —
 * including the dẑ/dt term, evaluated from native SPK acceleration —
 * rather than the GMAT dẑ/dt ≈ 0 approximation.
 */

use nalgebra::Vector3;

use crate::math::{SMatrix3, SVector6};

/// Computes the inertial→synodic rotation matrix `S` and its exact time
/// derivative `Ṡ` from the relative state of the two primaries
/// (NASA TP-20220014814 Eq. 66/69).
///
/// Axes: x̂ = r₁₂/‖r₁₂‖, ẑ = (r₁₂×v₁₂)/‖r₁₂×v₁₂‖, ŷ = ẑ×x̂; `S` has rows
/// x̂ᵀ, ŷᵀ, ẑᵀ. The derivative rows use dẑ/dt evaluated from `a12` (exact
/// GTDS/STK convention).
///
/// # Arguments
/// - `r12`: Position of the secondary relative to the primary, inertial axes. Units: [m]
/// - `v12`: Velocity of the secondary relative to the primary. Units: [m/s]
/// - `a12`: Acceleration of the secondary relative to the primary. Units: [m/s²]
///
/// # Returns
/// - `(S, Ṡ)`: Rotation matrix from inertial axes to synodic axes, and its
///   time derivative. Units: [-], [1/s]
///
/// # Examples
/// ```ignore
/// // Crate-internal: circular orbit in the xy-plane => S = I and Ṡ is the
/// // instantaneous rotation rate about ẑ.
/// let (s, s_dot) = synodic_axes(r12, v12, a12);
/// ```
#[allow(dead_code)] // Tasks 5-7 consume this helper
pub(crate) fn synodic_axes(
    r12: Vector3<f64>,
    v12: Vector3<f64>,
    a12: Vector3<f64>,
) -> (SMatrix3, SMatrix3) {
    let r_norm = r12.norm();
    let x_hat = r12 / r_norm;

    let h = r12.cross(&v12);
    let h_norm = h.norm();
    let z_hat = h / h_norm;

    let y_hat = z_hat.cross(&x_hat);

    // TP Eq. 69: dx̂/dt = (v₁₂ - x̂(x̂·v₁₂))/‖r₁₂‖ — the component of the
    // relative velocity perpendicular to x̂, divided by the separation.
    let x_hat_dot = (v12 - x_hat * x_hat.dot(&v12)) / r_norm;

    // dh/dt = d/dt(r₁₂×v₁₂) = v₁₂×v₁₂ + r₁₂×a₁₂ = r₁₂×a₁₂; dẑ/dt is its
    // component perpendicular to ẑ, divided by ‖h‖ (exact GTDS/STK form).
    let h_dot = r12.cross(&a12);
    let z_hat_dot = (h_dot - z_hat * z_hat.dot(&h_dot)) / h_norm;

    let y_hat_dot = z_hat_dot.cross(&x_hat) + z_hat.cross(&x_hat_dot);

    let s = SMatrix3::from_rows(&[x_hat.transpose(), y_hat.transpose(), z_hat.transpose()]);
    let s_dot = SMatrix3::from_rows(&[
        x_hat_dot.transpose(),
        y_hat_dot.transpose(),
        z_hat_dot.transpose(),
    ]);
    (s, s_dot)
}

/// Transforms an inertial-axis state (already translated to the synodic
/// frame's origin) into synodic axes: `r_s = S r`, `v_s = S v + Ṡ r`
/// (NASA TP-20220014814 Eq. 67/70, translation handled by the caller).
///
/// # Arguments
/// - `s`: Inertial→synodic rotation matrix from [`synodic_axes`]
/// - `s_dot`: Its time derivative. Units: [1/s]
/// - `x`: Cartesian state (position, velocity), inertial axes. Units: [m; m/s]
///
/// # Returns
/// - Cartesian state in synodic axes. Units: [m; m/s]
#[allow(dead_code)] // Tasks 5-7 consume this helper
pub(crate) fn state_inertial_to_synodic(s: &SMatrix3, s_dot: &SMatrix3, x: SVector6) -> SVector6 {
    let r = x.fixed_rows::<3>(0).into_owned();
    let v = x.fixed_rows::<3>(3).into_owned();
    let r_s: Vector3<f64> = s * r;
    let v_s: Vector3<f64> = s * v + s_dot * r;
    SVector6::new(r_s[0], r_s[1], r_s[2], v_s[0], v_s[1], v_s[2])
}

/// Inverse of [`state_inertial_to_synodic`]: `r = Sᵀ r_s`,
/// `v = Sᵀ v_s + Ṡᵀ r_s` (using Ṡᵀ = −SᵀṠSᵀ from d/dt(SᵀS) = 0, which
/// reduces TP Eq. 68/71 to this form).
///
/// # Arguments
/// - `s`: Inertial→synodic rotation matrix from [`synodic_axes`]
/// - `s_dot`: Its time derivative. Units: [1/s]
/// - `x`: Cartesian state (position, velocity), synodic axes. Units: [m; m/s]
///
/// # Returns
/// - Cartesian state in inertial axes. Units: [m; m/s]
#[allow(dead_code)] // Tasks 5-7 consume this helper
pub(crate) fn state_synodic_to_inertial(s: &SMatrix3, s_dot: &SMatrix3, x: SVector6) -> SVector6 {
    let r_s = x.fixed_rows::<3>(0).into_owned();
    let v_s = x.fixed_rows::<3>(3).into_owned();
    let r: Vector3<f64> = s.transpose() * r_s;
    let v: Vector3<f64> = s.transpose() * v_s + s_dot.transpose() * r_s;
    SVector6::new(r[0], r[1], r[2], v[0], v[1], v[2])
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    use super::*;

    /// Analytic circular-orbit relative state in the xy-plane at phase
    /// `theta`: r = R(cosθ, sinθ, 0), v = RΩ(−sinθ, cosθ, 0), a = −Ω²r.
    fn circular_rel_state(
        radius: f64,
        omega: f64,
        theta: f64,
    ) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let r = Vector3::new(radius * theta.cos(), radius * theta.sin(), 0.0);
        let v = Vector3::new(
            -radius * omega * theta.sin(),
            radius * omega * theta.cos(),
            0.0,
        );
        let a = -r * omega * omega;
        (r, v, a)
    }

    #[test]
    #[parallel]
    fn test_synodic_axes_circular_orbit() {
        let radius = 3.844e8;
        let omega = 2.66e-6;
        let (r, v, a) = circular_rel_state(radius, omega, 0.0);
        let (s, s_dot) = synodic_axes(r, v, a);

        // At theta = 0 the synodic axes coincide with the inertial axes.
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(s[(i, j)], if i == j { 1.0 } else { 0.0 }, epsilon = 1e-14);
            }
        }
        // Ṡ is the instantaneous rotation about ẑ at rate omega.
        assert_abs_diff_eq!(s_dot[(0, 1)], omega, epsilon = 1e-18);
        assert_abs_diff_eq!(s_dot[(1, 0)], -omega, epsilon = 1e-18);
        assert_abs_diff_eq!(s_dot[(2, 0)], 0.0, epsilon = 1e-18);
        assert_abs_diff_eq!(s_dot[(2, 1)], 0.0, epsilon = 1e-18);
    }

    #[test]
    #[parallel]
    fn test_synodic_axes_orthonormal_and_skew() {
        // Generic inclined, non-circular input: orthonormality and the
        // rigid-rotation identity ṠSᵀ + SṠᵀ = 0 must hold regardless.
        let r = Vector3::new(2.5e8, -1.2e8, 0.9e8);
        let v = Vector3::new(300.0, 800.0, -150.0);
        let a = Vector3::new(-1.9e-3, 0.9e-3, -0.7e-3);
        let (s, s_dot) = synodic_axes(r, v, a);

        let identity = s * s.transpose();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    identity[(i, j)],
                    if i == j { 1.0 } else { 0.0 },
                    epsilon = 1e-14
                );
            }
        }
        assert_abs_diff_eq!(s.determinant(), 1.0, epsilon = 1e-14);

        let skew = s_dot * s.transpose() + s * s_dot.transpose();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(skew[(i, j)], 0.0, epsilon = 1e-18);
            }
        }
    }

    #[test]
    #[parallel]
    fn test_synodic_axes_derivative_matches_finite_difference() {
        let radius = 3.844e8;
        let omega = 2.66e-6;
        let theta = 0.7;
        let dt = 1.0;

        let (r, v, a) = circular_rel_state(radius, omega, theta);
        let (_, s_dot) = synodic_axes(r, v, a);

        let (rp, vp, ap) = circular_rel_state(radius, omega, theta + omega * dt);
        let (sp, _) = synodic_axes(rp, vp, ap);
        let (rm, vm, am) = circular_rel_state(radius, omega, theta - omega * dt);
        let (sm, _) = synodic_axes(rm, vm, am);

        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    s_dot[(i, j)],
                    (sp[(i, j)] - sm[(i, j)]) / (2.0 * dt),
                    epsilon = 1e-12
                );
            }
        }
    }

    #[test]
    #[parallel]
    fn test_state_transform_roundtrip() {
        let r = Vector3::new(2.5e8, -1.2e8, 0.9e8);
        let v = Vector3::new(300.0, 800.0, -150.0);
        let a = Vector3::new(-1.9e-3, 0.9e-3, -0.7e-3);
        let (s, s_dot) = synodic_axes(r, v, a);

        let x = SVector6::new(1.0e8, -2.0e8, 5.0e7, 1.0e3, -2.0e3, 0.5e3);
        let x_syn = state_inertial_to_synodic(&s, &s_dot, x);
        let x_back = state_synodic_to_inertial(&s, &s_dot, x_syn);
        for i in 0..3 {
            assert_abs_diff_eq!(x_back[i], x[i], epsilon = 1e-6);
        }
        for i in 3..6 {
            assert_abs_diff_eq!(x_back[i], x[i], epsilon = 1e-10);
        }
    }
}
