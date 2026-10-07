/*!
 * Earth-Centered Inertial (ECI) to Local-Vertical Local-Horizontal (LVLH) Frame Transformations
 */

use nalgebra::Vector3;

use crate::math::{SMatrix3, SVector6};
use crate::relative_motion::common::{
    absolute_states_to_relative_state, relative_state_to_absolute_state, true_anomaly_rate,
};
use crate::relative_motion::rotation_rtn_to_eci;
use crate::utils::BraheError;
use crate::utils::batch::{batch_map, batch_zip};

/// Computes the rotation matrix transforming a vector in the Local-Vertical Local-Horizontal
/// (LVLH) frame to the Earth-Centered Inertial (ECI) frame.
///
/// The ECI frame can be any inertial frame centered on the orbited body, such as GCRF or EME2000.
///
/// The LVLH frame follows the CCSDS and SANA definition:
/// - Z: Unit vector collinear with and opposite to the position vector (nadir).
/// - Y: Unit vector collinear with and opposite to the orbital angular momentum `r × v`.
/// - X: `Y × Z`, completing the right-handed set (the RTN along-track axis, along the velocity
///   for a circular orbit).
///
/// This is a signed permutation of the RTN axes: `X = T`, `Y = −N`, `Z = −R`. Vallado and
/// STK use the name LVLH for the RTN axes themselves; brahe follows the CCSDS convention.
/// The matrix is assembled from the RTN axes as `[T, −N, −R]`, which equals the definition
/// exactly.
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
    let rtn = rotation_rtn_to_eci(x_eci);
    let r_hat: Vector3<f64> = rtn.column(0).into();
    let t_hat: Vector3<f64> = rtn.column(1).into();
    let n_hat: Vector3<f64> = rtn.column(2).into();

    SMatrix3::from_columns(&[t_hat, -n_hat, -r_hat])
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
    Vector3::new(0.0, -true_anomaly_rate(x_eci), 0.0)
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
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `LVLH_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - CCSDS 500.0-G-4, *Navigation Data—Definitions and Conventions*, Section 4.3.7.2, p. 4-7, November 2019
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
    absolute_states_to_relative_state(
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
/// # References:
/// - SANA Orbit-Relative Reference Frames registry, `LVLH_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - CCSDS 500.0-G-4, *Navigation Data—Definitions and Conventions*, Section 4.3.7.2, p. 4-7, November 2019
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
    relative_state_to_absolute_state(
        &rotation_eci_to_lvlh(x_chief),
        &omega_lvlh(x_chief),
        x_chief,
        x_rel_lvlh,
    )
}

/// Computes the LVLH-to-ECI rotation matrix for each state in `x_eci`.
///
/// Batch form of [`rotation_lvlh_to_eci`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Rotation matrices transforming LVLH -> ECI, one per state, in input order
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `LVLH_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - CCSDS 500.0-G-4, *Navigation Data—Definitions and Conventions*, Section 4.3.7.2, p. 4-7, November 2019
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::rotations_lvlh_to_eci;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let r = rotations_lvlh_to_eci(&[x, x]);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_lvlh_to_eci(x_eci: &[SVector6]) -> Vec<SMatrix3> {
    batch_map(|x| rotation_lvlh_to_eci(*x), x_eci)
}

/// Computes the ECI-to-LVLH rotation matrix for each state in `x_eci`.
///
/// Batch form of [`rotation_eci_to_lvlh`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Rotation matrices transforming ECI -> LVLH, one per state, in input order
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `LVLH_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - CCSDS 500.0-G-4, *Navigation Data—Definitions and Conventions*, Section 4.3.7.2, p. 4-7, November 2019
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::rotations_eci_to_lvlh;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let r = rotations_eci_to_lvlh(&[x, x]);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_eci_to_lvlh(x_eci: &[SVector6]) -> Vec<SMatrix3> {
    batch_map(|x| rotation_eci_to_lvlh(*x), x_eci)
}

/// Computes the LVLH frame angular velocity for each state in `x_eci`.
///
/// Batch form of [`omega_lvlh`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Angular velocities of the LVLH frame relative to ECI, expressed in LVLH axes, one per
///   state, in input order. Units: (*rad/s*)
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `LVLH_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - CCSDS 500.0-G-4, *Navigation Data—Definitions and Conventions*, Section 4.3.7.2, p. 4-7, November 2019
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::omegas_lvlh;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let omega = omegas_lvlh(&[x, x]);
/// assert_eq!(omega.len(), 2);
/// ```
pub fn omegas_lvlh(x_eci: &[SVector6]) -> Vec<Vector3<f64>> {
    batch_map(|x| omega_lvlh(*x), x_eci)
}

/// Computes the LVLH relative state of each deputy with respect to its chief.
///
/// Batch form of [`state_eci_to_lvlh`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The chief and deputy arguments follow the broadcast rule: each has length 1
/// or the common batch length, so one chief may be paired with many deputies
/// and vice versa.
///
/// # Arguments
/// - `x_chief`: Chief Cartesian ECI states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_deputy`: Deputy Cartesian ECI states, length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Deputy relative states in the chief LVLH frame, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `LVLH_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - CCSDS 500.0-G-4, *Navigation Data—Definitions and Conventions*, Section 4.3.7.2, p. 4-7, November 2019
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::states_eci_to_lvlh;
/// use brahe::vector6_from_array;
///
/// let chief = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let deputies = vec![
///     state_koe_to_eci(vector6_from_array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05]), AngleFormat::Degrees),
///     state_koe_to_eci(vector6_from_array([R_EARTH + 702e3, 0.0012, 97.82, 15.02, 30.02, 45.02]), AngleFormat::Degrees),
/// ];
/// let rel = states_eci_to_lvlh(&[chief], &deputies).unwrap();
/// assert_eq!(rel.len(), 2);
/// ```
pub fn states_eci_to_lvlh(
    x_chief: &[SVector6],
    x_deputy: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(|c, d| state_eci_to_lvlh(*c, *d), x_chief, x_deputy)
}

/// Computes the ECI state of each deputy from its LVLH relative state and chief.
///
/// Batch form of [`state_lvlh_to_eci`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The chief and deputy arguments follow the broadcast rule: each has length 1
/// or the common batch length, so one chief may be paired with many deputies
/// and vice versa.
///
/// # Arguments
/// - `x_chief`: Chief Cartesian ECI states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_rel_lvlh`: Deputy relative states in the chief LVLH frame, length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Deputy Cartesian ECI states, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # References
/// - SANA Orbit-Relative Reference Frames registry, `LVLH_ROTATING`, <https://sanaregistry.org/r/orbit_relative_reference_frames>
/// - CCSDS 500.0-G-4, *Navigation Data—Definitions and Conventions*, Section 4.3.7.2, p. 4-7, November 2019
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::states_lvlh_to_eci;
/// use brahe::vector6_from_array;
///
/// let chief = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let rel = vec![vector6_from_array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0]); 2];
/// let deputies = states_lvlh_to_eci(&[chief], &rel).unwrap();
/// assert_eq!(deputies.len(), 2);
/// ```
pub fn states_lvlh_to_eci(
    x_chief: &[SVector6],
    x_rel_lvlh: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(|c, r| state_lvlh_to_eci(*c, *r), x_chief, x_rel_lvlh)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::AngleFormat;
    use crate::R_EARTH;
    use crate::coordinates::state_koe_to_eci;
    use crate::frames::angular_velocity_from_rotation_rate;
    use crate::math::vector6_from_array;
    use crate::orbits::mean_motion;
    use crate::relative_motion::{omega_rtn, rotation_rtn_to_eci, state_eci_to_rtn};
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
        assert_abs_diff_eq!(x_axis, h_hat.cross(&r_hat), epsilon = 1e-15);
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
        assert_eq!(lvlh_x, rtn_t);
        assert_eq!(lvlh_y, -rtn_n);
        assert_eq!(lvlh_z, -rtn_r);
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

    #[test]
    #[parallel]
    fn test_batch_lvlh_match_scalar() {
        let chiefs: Vec<SVector6> = (0..3)
            .map(|i| {
                state_koe_to_eci(
                    vector6_from_array([
                        R_EARTH + 700e3 + 1e3 * i as f64,
                        0.01,
                        97.8,
                        15.0,
                        30.0,
                        45.0 + i as f64,
                    ]),
                    AngleFormat::Degrees,
                )
            })
            .collect();
        let deputies: Vec<SVector6> = (0..3)
            .map(|i| {
                state_koe_to_eci(
                    vector6_from_array([
                        R_EARTH + 701e3 + 1e3 * i as f64,
                        0.0115,
                        97.85,
                        15.05,
                        30.05,
                        45.05 + i as f64,
                    ]),
                    AngleFormat::Degrees,
                )
            })
            .collect();

        let rot = rotations_lvlh_to_eci(&chiefs);
        let rot_inv = rotations_eci_to_lvlh(&chiefs);
        let omegas = omegas_lvlh(&chiefs);
        let rel = states_eci_to_lvlh(&chiefs, &deputies).unwrap();
        let rel_one_chief = states_eci_to_lvlh(&chiefs[..1], &deputies).unwrap();
        let back = states_lvlh_to_eci(&chiefs, &rel).unwrap();
        for i in 0..3 {
            assert_eq!(rot[i], rotation_lvlh_to_eci(chiefs[i]));
            assert_eq!(rot_inv[i], rotation_eci_to_lvlh(chiefs[i]));
            assert_eq!(omegas[i], omega_lvlh(chiefs[i]));
            assert_eq!(rel[i], state_eci_to_lvlh(chiefs[i], deputies[i]));
            assert_eq!(rel_one_chief[i], state_eci_to_lvlh(chiefs[0], deputies[i]));
            assert_eq!(back[i], state_lvlh_to_eci(chiefs[i], rel[i]));
        }
        assert!(states_eci_to_lvlh(&chiefs[..2], &deputies).is_err());
        assert!(rotations_lvlh_to_eci(&[]).is_empty());
    }
}
