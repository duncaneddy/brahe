/*!
 * Cartesian state kinematics shared by every reference frame transformation.
 *
 * A frame pair either differs by a rotation alone, in which case position and
 * velocity rotate identically, or the target axes turn relative to the source
 * axes, in which case the velocity picks up a transport term. This module
 * holds one implementation of each case, together with the extraction of a
 * frame's angular velocity from a rotation matrix and its time derivative.
 *
 * Throughout, `R` is the rotation matrix taking a vector expressed in the
 * inertial (non-rotating) axes into the rotating axes, and `omega_b` is the
 * angular velocity of the rotating axes with respect to the inertial axes,
 * itself expressed in the rotating axes. The two are related by
 * `Ṙ = -[omega_b]× R`, where `[a]×` is the skew-symmetric matrix satisfying
 * `[a]× b = a × b`.
 */

use nalgebra::Vector3;

use crate::math::{SMatrix3, SVector6};

/// Rotates a Cartesian state between two axis sets that have no relative
/// angular velocity: `p' = R p` and `v' = R v`.
///
/// # Arguments
/// - `r`: Rotation matrix from the source axes to the target axes (dimensionless)
/// - `x`: Cartesian state (position, velocity) in the source axes. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian state (position, velocity) in the target axes. Units: (*m*; *m/s*)
pub(crate) fn rotate_state(r: &SMatrix3, x: &SVector6) -> SVector6 {
    let p: Vector3<f64> = r * x.fixed_rows::<3>(0);
    let v: Vector3<f64> = r * x.fixed_rows::<3>(3);
    SVector6::new(p[0], p[1], p[2], v[0], v[1], v[2])
}

/// Maps a Cartesian state from inertial axes into rotating axes:
/// `p' = R p` and `v' = R v - omega_b × p'`.
///
/// The transport term `omega_b × p'` removes the velocity that a point fixed
/// in the rotating axes acquires purely from the rotation of those axes.
///
/// # Arguments
/// - `r`: Rotation matrix from the inertial axes to the rotating axes (dimensionless)
/// - `omega_b`: Angular velocity of the rotating axes, expressed in the rotating axes. Units: (*rad/s*)
/// - `x`: Cartesian state (position, velocity) in the inertial axes. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian state (position, velocity) in the rotating axes. Units: (*m*; *m/s*)
pub(crate) fn state_inertial_to_rotating(
    r: &SMatrix3,
    omega_b: &Vector3<f64>,
    x: &SVector6,
) -> SVector6 {
    let p: Vector3<f64> = r * x.fixed_rows::<3>(0);
    let v: Vector3<f64> = r * x.fixed_rows::<3>(3) - omega_b.cross(&p);
    SVector6::new(p[0], p[1], p[2], v[0], v[1], v[2])
}

/// Inverse of [`state_inertial_to_rotating`]: `p = Rᵀ p'` and
/// `v = Rᵀ (v' + omega_b × p')`.
///
/// # Arguments
/// - `r`: Rotation matrix from the inertial axes to the rotating axes (dimensionless)
/// - `omega_b`: Angular velocity of the rotating axes, expressed in the rotating axes. Units: (*rad/s*)
/// - `x`: Cartesian state (position, velocity) in the rotating axes. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian state (position, velocity) in the inertial axes. Units: (*m*; *m/s*)
pub(crate) fn state_rotating_to_inertial(
    r: &SMatrix3,
    omega_b: &Vector3<f64>,
    x: &SVector6,
) -> SVector6 {
    let p_rot: Vector3<f64> = x.fixed_rows::<3>(0).into_owned();
    let v_rot: Vector3<f64> = x.fixed_rows::<3>(3).into_owned();
    let p: Vector3<f64> = r.transpose() * p_rot;
    let v: Vector3<f64> = r.transpose() * (v_rot + omega_b.cross(&p_rot));
    SVector6::new(p[0], p[1], p[2], v[0], v[1], v[2])
}

/// Extracts the angular velocity of a rotating axis set from its rotation
/// matrix and that matrix's time derivative, through `[omega_b]× = -Ṙ Rᵀ`.
///
/// The product is skew-symmetric for an exactly orthonormal `R`; the
/// skew-symmetric part is taken so that a matrix orthonormal only to within
/// floating-point round-off still yields the angular velocity that best fits
/// it.
///
/// # Arguments
/// - `r`: Rotation matrix from the inertial axes to the rotating axes (dimensionless)
/// - `r_dot`: Time derivative of `r`. Units: (*1/s*)
///
/// # Returns
/// - Angular velocity of the rotating axes, expressed in the rotating axes. Units: (*rad/s*)
pub(crate) fn angular_velocity_from_rotation_rate(r: &SMatrix3, r_dot: &SMatrix3) -> Vector3<f64> {
    let m = -(r_dot * r.transpose());
    Vector3::new(
        0.5 * (m[(2, 1)] - m[(1, 2)]),
        0.5 * (m[(0, 2)] - m[(2, 0)]),
        0.5 * (m[(1, 0)] - m[(0, 1)]),
    )
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    use super::*;
    use crate::attitude::Rz;
    use crate::constants::AngleFormat;

    /// `R(t) = Rz(rate t)` and its analytic derivative, for a frame turning
    /// about the inertial z-axis at `rate` rad/s.
    fn spinning_axes(rate: f64, t: f64) -> (SMatrix3, SMatrix3) {
        let theta = rate * t;
        let (s, c) = theta.sin_cos();
        let r = Rz(theta, AngleFormat::Radians);
        let r_dot = rate * SMatrix3::new(-s, c, 0.0, -c, -s, 0.0, 0.0, 0.0, 0.0);
        (r, r_dot)
    }

    #[test]
    #[parallel]
    fn test_rotate_state() {
        // Quarter turn about z: x̂ -> -ŷ and ŷ -> x̂ in the rotated axes.
        let r = Rz(std::f64::consts::FRAC_PI_2, AngleFormat::Radians);
        let x = SVector6::new(7.0e6, 0.0, 1.0e5, 0.0, 7.5e3, -1.0);
        let out = rotate_state(&r, &x);

        assert_abs_diff_eq!(out[0], 0.0, epsilon = 1e-8);
        assert_abs_diff_eq!(out[1], -7.0e6, epsilon = 1e-8);
        assert_abs_diff_eq!(out[2], 1.0e5, epsilon = 1e-12);
        assert_abs_diff_eq!(out[3], 7.5e3, epsilon = 1e-9);
        assert_abs_diff_eq!(out[4], 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!(out[5], -1.0, epsilon = 1e-12);
    }

    #[test]
    #[parallel]
    fn test_rotating_state_round_trip() {
        let (r, _) = spinning_axes(7.2921150e-5, 3.6e3);
        let omega_b = Vector3::new(1.0e-6, -2.0e-6, 7.2921150e-5);
        let x = SVector6::new(7.0e6, -1.2e6, 3.4e5, -1.1e3, 7.4e3, 2.0e2);

        let x_rot = state_inertial_to_rotating(&r, &omega_b, &x);
        let x_back = state_rotating_to_inertial(&r, &omega_b, &x_rot);

        // Relative tolerance: the round trip is exact to a few units in the
        // last place of each component.
        for i in 0..6 {
            assert_abs_diff_eq!(x_back[i], x[i], epsilon = 1e-12 * x[i].abs());
        }
    }

    #[test]
    #[parallel]
    fn test_state_inertial_to_rotating_removes_transport_velocity() {
        // A point fixed in the rotating axes has zero velocity there.
        let rate = 7.2921150e-5;
        let (r, _) = spinning_axes(rate, 1.234e4);
        let omega_b = Vector3::new(0.0, 0.0, rate);
        let omega_inertial = Vector3::new(0.0, 0.0, rate);

        let p = Vector3::new(6.378e6, 1.0e6, -2.0e6);
        let v = omega_inertial.cross(&p);
        let x = SVector6::new(p[0], p[1], p[2], v[0], v[1], v[2]);

        let x_rot = state_inertial_to_rotating(&r, &omega_b, &x);
        for i in 3..6 {
            assert_abs_diff_eq!(x_rot[i], 0.0, epsilon = 1e-9);
        }
    }

    #[test]
    #[parallel]
    fn test_angular_velocity_from_rotation_rate() {
        let rate = 2.5e-4;
        for t in [0.0, 1.0e3, 5.5e3] {
            let (r, r_dot) = spinning_axes(rate, t);
            let omega_b = angular_velocity_from_rotation_rate(&r, &r_dot);
            assert_abs_diff_eq!(omega_b[0], 0.0, epsilon = 1e-18);
            assert_abs_diff_eq!(omega_b[1], 0.0, epsilon = 1e-18);
            assert_abs_diff_eq!(omega_b[2], rate, epsilon = 1e-18);
        }
    }

    #[test]
    #[parallel]
    fn test_transport_term_matches_rotation_rate_form() {
        // `-omega_b × (R p)` and `Ṙ p` are the same transport term.
        let rate = 2.5e-4;
        let (r, r_dot) = spinning_axes(rate, 7.7e3);
        let omega_b = angular_velocity_from_rotation_rate(&r, &r_dot);
        let x = SVector6::new(2.5e8, -1.2e8, 0.9e8, 3.0e2, 8.0e2, -1.5e2);

        let out = state_inertial_to_rotating(&r, &omega_b, &x);
        let p: Vector3<f64> = x.fixed_rows::<3>(0).into_owned();
        let v: Vector3<f64> = x.fixed_rows::<3>(3).into_owned();
        let v_expected: Vector3<f64> = r * v + r_dot * p;

        for i in 0..3 {
            assert_abs_diff_eq!(out[i + 3], v_expected[i], epsilon = 1e-9);
        }
    }
}
