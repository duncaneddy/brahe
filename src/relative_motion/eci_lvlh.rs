/*!
 * Earth-Centered Inertial (ECI) to Local-Vertical Local-Horizontal (LVLH) Frame Transformations
 */

use nalgebra::Vector3;

use crate::frames::{OrbitRelativeFrameVariant, rotate_covariance_6};
use crate::math::{SMatrix3, SMatrix6, SVector6};
use crate::relative_motion::common::{
    jacobian_from_inertial, jacobian_to_inertial, relative_state_from_frame,
    relative_state_to_frame,
};
use crate::relative_motion::omega_rtn;

/// Computes the rotation matrix transforming a vector in the Local-Vertical Local-Horizontal
/// (LVLH) frame to the Earth-Centered Inertial (ECI) frame.
///
/// The ECI frame can be any inertial frame centered on the orbited body, such as GCRF or EME2000.
///
/// The LVLH frame follows the CCSDS and SANA definition:
/// - Z: Unit vector collinear with and opposite to the position vector (nadir).
/// - Y: Unit vector collinear with and opposite to the orbital angular momentum `r × v`.
/// - X: `Y × Z`, completing the right-handed set (along-track for a circular orbit).
///
/// This is a signed permutation of the RTN axes: `X = T`, `Y = −N`, `Z = −R`. Vallado and
/// STK use the name LVLH for the RTN axes themselves; brahe follows the CCSDS convention.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from LVLH to ECI frame
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `LVLH_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - CCSDS 500.0-G-4, *Navigation Data—Definitions and Conventions*, Section 4.3.7.2, p. 4-7, November 2019
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::R_EARTH;
/// use brahe::orbits::*;
/// use brahe::relative_motion::*;
///
/// let sma = R_EARTH + 700e3;
/// let x_eci = SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0);
///
/// let rotation_matrix = rotation_lvlh_to_eci(x_eci);
/// ```
pub fn rotation_lvlh_to_eci(x_eci: SVector6) -> SMatrix3 {
    let r = x_eci.fixed_rows::<3>(0);
    let v = x_eci.fixed_rows::<3>(3);

    let r_hat = r / r.norm();
    let h = r.cross(&v);
    let h_hat = h / h.norm();

    let z_hat = -r_hat;
    let y_hat = -h_hat;
    let x_hat = y_hat.cross(&z_hat);

    SMatrix3::from_columns(&[x_hat, y_hat, z_hat])
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Inertial (ECI)
/// frame to the Local-Vertical Local-Horizontal (LVLH) frame.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from ECI to LVLH frame
///
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `LVLH_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - CCSDS 500.0-G-4, *Navigation Data—Definitions and Conventions*, Section 4.3.7.2, p. 4-7, November 2019
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::R_EARTH;
/// use brahe::orbits::*;
/// use brahe::relative_motion::*;
///
/// let sma = R_EARTH + 700e3;
/// let x_eci = SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0);
///
/// let rotation_matrix = rotation_eci_to_lvlh(x_eci);
/// ```
pub fn rotation_eci_to_lvlh(x_eci: SVector6) -> SMatrix3 {
    rotation_lvlh_to_eci(x_eci).transpose()
}

/// Computes the angular velocity of the Local-Vertical Local-Horizontal (LVLH) frame with
/// respect to the Earth-Centered Inertial (ECI) frame, expressed in LVLH axes.
///
/// The LVLH frame rotates about the orbit normal at the true-anomaly rate `ḟ = |r × v| / r²`
/// (Alfriend et al. equation 2.16). The orbit normal is the negative LVLH Y axis, so the
/// angular velocity is `[0, −ḟ, 0]`. The rate is exact under two-body motion and is the rate
/// of the osculating frame otherwise.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `omega`: Angular velocity of the LVLH frame relative to ECI, expressed in LVLH axes (rad/s)
///
/// # References:
/// - K. T. Alfriend, S. R. Vadali, P. Gurfil, J. P. How, L. S. Breger, *Spacecraft Formation Flying*, Elsevier, 2010, eq. 2.16
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::R_EARTH;
/// use brahe::orbits::*;
/// use brahe::relative_motion::*;
///
/// let sma = R_EARTH + 700e3;
/// let x_eci = SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0);
///
/// let omega = omega_lvlh(x_eci);
/// ```
pub fn omega_lvlh(x_eci: SVector6) -> Vector3<f64> {
    let f_dot = omega_rtn(x_eci)[2];
    Vector3::new(0.0, -f_dot, 0.0)
}

/// 6x6 Jacobian taking an LVLH state covariance into ECI axes.
///
/// With `R` the LVLH-to-ECI rotation and `ω` the LVLH angular velocity, the Jacobian is
/// `[[R, 0], [R [ω]×, R]]` for the rotating variant and `[[R, 0], [0, R]]` for the inertial
/// snapshot.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `variant`: Whether the LVLH axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_eci = J P_lvlh Jᵀ`
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::R_EARTH;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::orbits::*;
/// use brahe::relative_motion::*;
///
/// let sma = R_EARTH + 700e3;
/// let x_eci = SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0);
///
/// let j = jacobian_lvlh_to_eci(x_eci, OrbitRelativeFrameVariant::Rotating);
/// ```
pub fn jacobian_lvlh_to_eci(x_eci: SVector6, variant: OrbitRelativeFrameVariant) -> SMatrix6 {
    jacobian_to_inertial(&rotation_lvlh_to_eci(x_eci), &omega_lvlh(x_eci), variant)
}

/// 6x6 Jacobian taking an ECI state covariance into LVLH axes. Exact inverse of
/// [`jacobian_lvlh_to_eci`].
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `variant`: Whether the LVLH axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `j`: 6x6 Jacobian such that `P_lvlh = J P_eci Jᵀ`
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::R_EARTH;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::orbits::*;
/// use brahe::relative_motion::*;
///
/// let sma = R_EARTH + 700e3;
/// let x_eci = SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0);
///
/// let j = jacobian_eci_to_lvlh(x_eci, OrbitRelativeFrameVariant::Rotating);
/// ```
pub fn jacobian_eci_to_lvlh(x_eci: SVector6, variant: OrbitRelativeFrameVariant) -> SMatrix6 {
    jacobian_from_inertial(&rotation_eci_to_lvlh(x_eci), &omega_lvlh(x_eci), variant)
}

/// Transforms a 6x6 state covariance from LVLH axes into ECI axes.
///
/// Applies the congruence `P_eci = J P_lvlh Jᵀ` with `J` from [`jacobian_lvlh_to_eci`], and
/// symmetrizes the result.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in LVLH axes (m², m²/s, m²/s²)
/// - `variant`: Whether the LVLH axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `p_eci`: 6x6 state covariance in ECI axes (m², m²/s, m²/s²)
///
/// # Examples:
/// ```
/// use brahe::{SVector6, SMatrix6};
/// use brahe::R_EARTH;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::orbits::*;
/// use brahe::relative_motion::*;
///
/// let sma = R_EARTH + 700e3;
/// let x_eci = SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0);
/// let p_lvlh = SMatrix6::identity();
///
/// let p_eci = covariance_lvlh_to_eci(x_eci, &p_lvlh, OrbitRelativeFrameVariant::Rotating);
/// ```
pub fn covariance_lvlh_to_eci(
    x_eci: SVector6,
    covariance: &SMatrix6,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    rotate_covariance_6(covariance, &jacobian_lvlh_to_eci(x_eci, variant))
}

/// Transforms a 6x6 state covariance from ECI axes into LVLH axes.
///
/// Applies the congruence `P_lvlh = J P_eci Jᵀ` with `J` from [`jacobian_eci_to_lvlh`], and
/// symmetrizes the result.
///
/// # Arguments:
/// - `x_eci`: 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `covariance`: 6x6 state covariance in ECI axes (m², m²/s, m²/s²)
/// - `variant`: Whether the LVLH axes rotate with the orbit or are frozen at the epoch
///
/// # Returns:
/// - `p_lvlh`: 6x6 state covariance in LVLH axes (m², m²/s, m²/s²)
///
/// # Examples:
/// ```
/// use brahe::{SVector6, SMatrix6};
/// use brahe::R_EARTH;
/// use brahe::frames::OrbitRelativeFrameVariant;
/// use brahe::orbits::*;
/// use brahe::relative_motion::*;
///
/// let sma = R_EARTH + 700e3;
/// let x_eci = SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0);
/// let p_eci = SMatrix6::identity();
///
/// let p_lvlh = covariance_eci_to_lvlh(x_eci, &p_eci, OrbitRelativeFrameVariant::Rotating);
/// ```
pub fn covariance_eci_to_lvlh(
    x_eci: SVector6,
    covariance: &SMatrix6,
    variant: OrbitRelativeFrameVariant,
) -> SMatrix6 {
    rotate_covariance_6(covariance, &jacobian_eci_to_lvlh(x_eci, variant))
}

/// Transforms the absolute states of a chief and deputy satellite from the Earth-Centered
/// Inertial (ECI) frame to the relative state of the deputy with respect to the chief in the
/// rotating Local-Vertical Local-Horizontal (LVLH) frame.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_deputy`: 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `x_rel_lvlh`: 6D relative state of the deputy with respect to the chief in the LVLH frame [ρ_X, ρ_Y, ρ_Z, ρ̇_X, ρ̇_Y, ρ̇_Z] (m, m/s)
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::*;
///
/// let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
/// let oe_deputy = SVector6::new(R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05);
///
/// let x_chief = state_koe_to_eci(oe_chief, AngleFormat::Degrees);
/// let x_deputy = state_koe_to_eci(oe_deputy, AngleFormat::Degrees);
///
/// let x_rel_lvlh = state_eci_to_lvlh(x_chief, x_deputy);
/// ```
pub fn state_eci_to_lvlh(x_chief: SVector6, x_deputy: SVector6) -> SVector6 {
    relative_state_to_frame(
        &rotation_eci_to_lvlh(x_chief),
        &omega_lvlh(x_chief),
        x_chief,
        x_deputy,
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite
/// from the rotating Local-Vertical Local-Horizontal (LVLH) frame to the absolute state of
/// the deputy in the Earth-Centered Inertial (ECI) frame.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_rel_lvlh`: 6D relative state of the deputy with respect to the chief in the LVLH frame [ρ_X, ρ_Y, ρ_Z, ρ̇_X, ρ̇_Y, ρ̇_Z] (m, m/s)
///
/// # Returns:
/// - `x_deputy`: 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::*;
///
/// let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
/// let x_chief = state_koe_to_eci(oe_chief, AngleFormat::Degrees);
/// let x_rel_lvlh = SVector6::new(1000.0, 500.0, -300.0, 0.0, 0.0, 0.0);
///
/// let x_deputy = state_lvlh_to_eci(x_chief, x_rel_lvlh);
/// ```
pub fn state_lvlh_to_eci(x_chief: SVector6, x_rel_lvlh: SVector6) -> SVector6 {
    relative_state_from_frame(
        &rotation_eci_to_lvlh(x_chief),
        &omega_lvlh(x_chief),
        x_chief,
        x_rel_lvlh,
    )
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::AngleFormat;
    use crate::R_EARTH;
    use crate::coordinates::state_koe_to_eci;
    use crate::frames::angular_velocity_from_rotation_rate;
    use crate::math::{block_diagonal, skew_symmetric, vector6_from_array};
    use crate::orbits::mean_motion;
    use crate::relative_motion::{rotation_rtn_to_eci, state_eci_to_rtn};
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    fn inclined_test_state() -> SVector6 {
        state_koe_to_eci(
            SVector6::new(R_EARTH + 700e3, 0.1, 97.8, 15.0, 30.0, 45.0),
            AngleFormat::Degrees,
        )
    }

    /// The same orbit advanced by `dt` seconds of two-body motion.
    fn advanced_state(dt: f64) -> SVector6 {
        let n = mean_motion(R_EARTH + 700e3, AngleFormat::Degrees);
        state_koe_to_eci(
            SVector6::new(R_EARTH + 700e3, 0.1, 97.8, 15.0, 30.0, 45.0 + n * dt),
            AngleFormat::Degrees,
        )
    }

    #[test]
    #[parallel]
    fn test_rotation_lvlh_to_eci_axes_match_definition() {
        let x = inclined_test_state();
        let r = x.fixed_rows::<3>(0).into_owned();
        let v = x.fixed_rows::<3>(3).into_owned();
        let r_hat = r / r.norm();
        let h_hat = r.cross(&v) / r.cross(&v).norm();

        let m = rotation_lvlh_to_eci(x);
        let x_axis: Vector3<f64> = m.column(0).into();
        let y_axis: Vector3<f64> = m.column(1).into();
        let z_axis: Vector3<f64> = m.column(2).into();

        assert_abs_diff_eq!(z_axis, -r_hat, epsilon = 1e-15);
        assert_abs_diff_eq!(y_axis, -h_hat, epsilon = 1e-15);
        assert_abs_diff_eq!(x_axis, y_axis.cross(&z_axis), epsilon = 1e-15);
        assert_abs_diff_eq!(m.determinant(), 1.0, epsilon = 1e-14);
        assert_abs_diff_eq!(m * m.transpose(), SMatrix3::identity(), epsilon = 1e-14);
    }

    #[test]
    #[parallel]
    fn test_rotation_lvlh_is_signed_permutation_of_rtn() {
        let x = inclined_test_state();
        let rtn = rotation_rtn_to_eci(x);
        let lvlh = rotation_lvlh_to_eci(x);
        let lvlh_x: Vector3<f64> = lvlh.column(0).into();
        let lvlh_y: Vector3<f64> = lvlh.column(1).into();
        let lvlh_z: Vector3<f64> = lvlh.column(2).into();
        let rtn_r: Vector3<f64> = rtn.column(0).into();
        let rtn_t: Vector3<f64> = rtn.column(1).into();
        let rtn_n: Vector3<f64> = rtn.column(2).into();
        // X_lvlh = T, Y_lvlh = -N, Z_lvlh = -R
        assert_abs_diff_eq!(lvlh_x, rtn_t, epsilon = 1e-15);
        assert_abs_diff_eq!(lvlh_y, -rtn_n, epsilon = 1e-15);
        assert_abs_diff_eq!(lvlh_z, -rtn_r, epsilon = 1e-15);
    }

    #[test]
    #[parallel]
    fn test_rotation_eci_to_lvlh_is_transpose() {
        let x = inclined_test_state();
        let forward = rotation_lvlh_to_eci(x);
        let inverse = rotation_eci_to_lvlh(x);
        assert_eq!(inverse, forward.transpose());
        assert_abs_diff_eq!(forward * inverse, SMatrix3::identity(), epsilon = 1e-14);
    }

    #[test]
    #[parallel]
    fn test_omega_lvlh_matches_rtn_rate_about_minus_y() {
        let x = inclined_test_state();
        let omega = omega_lvlh(x);
        let f_dot = omega_rtn(x)[2];
        assert_abs_diff_eq!(omega, Vector3::new(0.0, -f_dot, 0.0), epsilon = 1e-18);
    }

    #[test]
    #[parallel]
    fn test_omega_lvlh_matches_finite_difference() {
        let dt = 0.05;
        let x0 = inclined_test_state();
        let r_minus = rotation_eci_to_lvlh(advanced_state(-dt));
        let r_plus = rotation_eci_to_lvlh(advanced_state(dt));
        let r_dot = (r_plus - r_minus) / (2.0 * dt);
        let omega_fd = angular_velocity_from_rotation_rate(&rotation_eci_to_lvlh(x0), &r_dot);
        assert_abs_diff_eq!(omega_fd, omega_lvlh(x0), epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_jacobian_lvlh_to_eci_inertial_is_block_diagonal() {
        let x = inclined_test_state();
        let j = jacobian_lvlh_to_eci(x, OrbitRelativeFrameVariant::Inertial);
        let r = rotation_lvlh_to_eci(x);
        assert_abs_diff_eq!((j - block_diagonal(&r, &r)).norm(), 0.0, epsilon = 1e-15);
    }

    #[test]
    #[parallel]
    fn test_jacobian_lvlh_to_eci_rotating_coupling() {
        let x = inclined_test_state();
        let j = jacobian_lvlh_to_eci(x, OrbitRelativeFrameVariant::Rotating);
        let expected = rotation_lvlh_to_eci(x) * skew_symmetric(&omega_lvlh(x));
        let coupling: SMatrix3 = j.fixed_view::<3, 3>(3, 0).into();
        assert_abs_diff_eq!((coupling - expected).norm(), 0.0, epsilon = 1e-18);
        assert!(coupling.norm() > 0.0);
    }

    #[test]
    #[parallel]
    fn test_jacobian_lvlh_eci_inverse_identity() {
        let x = inclined_test_state();
        for variant in [
            OrbitRelativeFrameVariant::Inertial,
            OrbitRelativeFrameVariant::Rotating,
        ] {
            let forward = jacobian_lvlh_to_eci(x, variant);
            let inverse = jacobian_eci_to_lvlh(x, variant);
            assert_abs_diff_eq!(
                (inverse * forward - SMatrix6::identity()).norm(),
                0.0,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    #[parallel]
    fn test_covariance_lvlh_eci_round_trip() {
        let x = inclined_test_state();
        let mut p = SMatrix6::zeros();
        for i in 0..3 {
            p[(i, i)] = 100.0;
            p[(3 + i, 3 + i)] = 0.01;
        }
        p[(0, 1)] = 25.0;
        p[(1, 0)] = 25.0;
        for variant in [
            OrbitRelativeFrameVariant::Inertial,
            OrbitRelativeFrameVariant::Rotating,
        ] {
            let p_eci = covariance_lvlh_to_eci(x, &p, variant);
            let p_back = covariance_eci_to_lvlh(x, &p_eci, variant);
            assert_abs_diff_eq!((p_back - p).norm() / p.norm(), 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!((p_eci - p_eci.transpose()).norm(), 0.0, epsilon = 1e-18);
        }
    }

    #[test]
    #[parallel]
    fn test_state_eci_to_lvlh_matches_permuted_rtn() {
        let x_chief = inclined_test_state();
        let x_deputy = state_koe_to_eci(
            SVector6::new(R_EARTH + 701e3, 0.1015, 97.85, 15.05, 30.05, 45.05),
            AngleFormat::Degrees,
        );
        let rtn = state_eci_to_rtn(x_chief, x_deputy);
        let lvlh = state_eci_to_lvlh(x_chief, x_deputy);
        // [X, Y, Z] = [T, -N, -R] for both position and velocity components
        let expected = vector6_from_array([rtn[1], -rtn[2], -rtn[0], rtn[4], -rtn[5], -rtn[3]]);
        assert_abs_diff_eq!(lvlh, expected, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_state_lvlh_to_eci_round_trip() {
        let x_chief = inclined_test_state();
        let x_rel = SVector6::new(1000.0, 500.0, -300.0, 0.1, -0.05, 0.02);
        let x_deputy = state_lvlh_to_eci(x_chief, x_rel);
        let recovered = state_eci_to_lvlh(x_chief, x_deputy);
        assert_abs_diff_eq!(recovered, x_rel, epsilon = 1e-8);
    }
}
