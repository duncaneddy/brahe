/*!
 * Earth-Centered Inertial (ECI) to Radial, Along-Track, Cross-Track (RTN) Frame Transformations
 */

use crate::math::{SMatrix3, SVector6};
use nalgebra::Vector3;

use crate::utils::BraheError;
use crate::utils::batch::{batch_map, batch_zip};

/// Computes the rotation matrix transforming a vector in the radial, along-track, cross-track (RTN)
/// frame to the Earth-Centered Inertial (ECI) frame at a given epoch.
///
/// The ECI frame can be any inertial frame centered at the Earth's center, such as GCRF or EME2000.
///
/// The RTN frame is defined as follows:
/// - R (Radial): Points from the Earth's center to the satellite's position.
/// - N (Cross-Track): Perpendicular to the orbital plane, defined by the angular momentum vector (cross product of position and velocity).
/// - T (Along-Track): Completes the right-handed coordinate system, lying in the orbital plane and perpendicular to R and N.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from RTN to ECI frame
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::R_EARTH;
/// use brahe::frames::*;
/// use brahe::orbits::*;
/// use brahe::relative_motion::*;
///
/// // Define satellite position
/// let sma = R_EARTH + 700e3; // Semi-major axis in meters
/// let x_eci = SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0);
///
/// let rotation_matrix = rotation_rtn_to_eci(x_eci);
/// ```
pub fn rotation_rtn_to_eci(x_eci: SVector6) -> SMatrix3 {
    // Extract position and velocity
    let r = x_eci.fixed_rows::<3>(0);
    let v = x_eci.fixed_rows::<3>(3);

    // Compute RTN frame unit vectors
    let r_norm = r.norm();

    let h = r.cross(&v); // Angular momentum vector
    let h_norm = h.norm();

    // RTN frame:
    // R (Radial): Along position vector (away from Earth)
    // T (Along-track): Completes right-handed system (C × R)
    // N (Normal): Along angular momentum (perpendicular to orbital plane)
    let r_hat = r / r_norm;
    let n_hat = h / h_norm;
    let t_hat = n_hat.cross(&r_hat);

    // Construct rotation matrix from RTN to ECI
    SMatrix3::from_columns(&[r_hat, t_hat, n_hat])
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Inertial (ECI)
/// frame to the radial, along-track, cross-track (RTN) frame at a given epoch.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming from ECI to RTN frame
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::R_EARTH;
/// use brahe::frames::*;
/// use brahe::orbits::*;
/// use brahe::relative_motion::*;
///
/// // Define satellite position
/// let sma = R_EARTH + 700e3; // Semi-major axis in meters
/// let x_eci = SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0);
///
/// let rotation_matrix = rotation_eci_to_rtn(x_eci);
/// ```
pub fn rotation_eci_to_rtn(x_eci: SVector6) -> SMatrix3 {
    rotation_rtn_to_eci(x_eci).transpose()
}

/// Computes the angular velocity of the radial, along-track, cross-track (RTN) frame with
/// respect to the Earth-Centered Inertial (ECI) frame, expressed in RTN axes.
///
/// The RTN frame rotates about its cross-track axis at the orbital true-anomaly rate
/// `ḟ = |r × v| / r²` (Alfriend equation 2.16), so the angular velocity is `[0, 0, ḟ]`.
///
/// The returned vector's components are those of the RTN frame, not the ECI frame: the
/// third component is the rate about the cross-track axis. Multiply by
/// [`rotation_rtn_to_eci`] to express the same angular velocity in ECI axes.
///
/// # Arguments:
/// - `x_eci`: 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `omega`: Angular velocity of the RTN frame relative to ECI, expressed in RTN axes (rad/s)
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::R_EARTH;
/// use brahe::orbits::*;
/// use brahe::relative_motion::*;
///
/// // Define satellite position
/// let sma = R_EARTH + 700e3; // Semi-major axis in meters
/// let x_eci = SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0);
///
/// let omega = omega_rtn(x_eci);
/// ```
pub fn omega_rtn(x_eci: SVector6) -> Vector3<f64> {
    // Extract position and velocity
    let rc = x_eci.fixed_rows::<3>(0);
    let vc = x_eci.fixed_rows::<3>(3);

    // Get angular velocity of RTN frame with respect to ECI frame (Alfriend equation 2.16)
    let f_dot = (rc.cross(&vc)).norm() / (rc.norm().powi(2));
    Vector3::new(0.0, 0.0, f_dot)
}

/// Transforms the absolute states of a chief and deputy satellite from the Earth-Centered Inertial (ECI)
/// frame to the relative state of the deputy with respect to the chief in the rotating
/// Radial, Along-Track, Cross-Track (RTN) frame.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_deputy`: 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
///
/// # Returns:
/// - `x_rel_rtn`: 6D relative state vector of the deputy with respect to the chief in the RTN frame [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N] (m, m/s)
///
/// # Examples:
/// ```
/// use brahe::SVector6;
/// use brahe::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::*;
///
/// // Define chief and deputy satellite positions
/// let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
/// let oe_deputy = SVector6::new(R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05);
///
/// let x_chief = state_koe_to_eci(oe_chief, AngleFormat::Degrees);
/// let x_deputy = state_koe_to_eci(oe_deputy, AngleFormat::Degrees);
///
/// let x_rel_rtn = state_eci_to_rtn(x_chief, x_deputy);
/// ```
pub fn state_eci_to_rtn(x_chief: SVector6, x_deputy: SVector6) -> SVector6 {
    // NOTE: This could potentially be more accurately revised based on equations in section 4.7.1 of Alfriend

    // Get RTN rotation matrix
    let r_eci_to_rtn = rotation_eci_to_rtn(x_chief);

    // Relative position and velocity in ECI frame
    let rho_eci = x_deputy.fixed_rows::<3>(0) - x_chief.fixed_rows::<3>(0);
    let rho_dot_eci = x_deputy.fixed_rows::<3>(3) - x_chief.fixed_rows::<3>(3);

    // Get angular velocity of RTN frame with respect to ECI frame
    let omega = omega_rtn(x_chief);

    // Transform relative position and velocity to RTN frame
    let rho_rtn = r_eci_to_rtn * rho_eci;
    let rho_dot_rtn = r_eci_to_rtn * rho_dot_eci - omega.cross(&rho_rtn);

    SVector6::new(
        rho_rtn[0],
        rho_rtn[1],
        rho_rtn[2],
        rho_dot_rtn[0],
        rho_dot_rtn[1],
        rho_dot_rtn[2],
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite
/// from the rotating Radial, Along-Track, Cross-Track (RTN) frame to the absolute states
/// of the chief and deputy in the Earth-Centered Inertial (ECI) frame.
///
/// # Arguments:
/// - `x_chief`: 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s)
/// - `x_rel_rtn`: 6D relative state vector of the deputy with respect to the chief in the RTN frame [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N] (m, m/s)
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
/// // Define chief and deputy satellite positions
/// let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
/// let oe_deputy = SVector6::new(R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05);
///
/// let x_chief = state_koe_to_eci(oe_chief, AngleFormat::Degrees);
/// let x_deputy = state_koe_to_eci(oe_deputy, AngleFormat::Degrees);
///
/// let x_rel_rtn = state_eci_to_rtn(x_chief, x_deputy);
/// let x_deputy_reconstructed = state_rtn_to_eci(x_chief, x_rel_rtn);
/// ```
pub fn state_rtn_to_eci(x_chief: SVector6, x_rel_rtn: SVector6) -> SVector6 {
    // Extract chief position and velocity
    let rc = x_chief.fixed_rows::<3>(0);
    let vc = x_chief.fixed_rows::<3>(3);

    // Get RTN rotation matrix
    let r_rtn_to_eci = rotation_rtn_to_eci(x_chief);

    // Extract relative position and velocity in RTN frame
    let rho_rtn = x_rel_rtn.fixed_rows::<3>(0);
    let rho_dot_rtn = x_rel_rtn.fixed_rows::<3>(3);

    // Get angular velocity of RTN frame with respect to ECI frame
    let omega = omega_rtn(x_chief);

    // Compute deputy absolute state in ECI frame
    let r_deputy = rc + r_rtn_to_eci * rho_rtn;
    let v_deputy = r_rtn_to_eci * (rho_dot_rtn + omega.cross(&rho_rtn)) + vc;

    SVector6::new(
        r_deputy[0],
        r_deputy[1],
        r_deputy[2],
        v_deputy[0],
        v_deputy[1],
        v_deputy[2],
    )
}

/// Computes the RTN-to-ECI rotation matrix for each state in `x_eci`.
///
/// Batch form of [`rotation_rtn_to_eci`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Rotation matrices transforming RTN -> ECI, one per state, in input order
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::rotations_rtn_to_eci;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let r = rotations_rtn_to_eci(&[x, x]);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_rtn_to_eci(x_eci: &[SVector6]) -> Vec<SMatrix3> {
    batch_map(|x| rotation_rtn_to_eci(*x), x_eci)
}

/// Computes the ECI-to-RTN rotation matrix for each state in `x_eci`.
///
/// Batch form of [`rotation_eci_to_rtn`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_eci`: Cartesian ECI states (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - Rotation matrices transforming ECI -> RTN, one per state, in input order
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::rotations_eci_to_rtn;
/// use brahe::vector6_from_array;
///
/// let x = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let r = rotations_eci_to_rtn(&[x, x]);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_eci_to_rtn(x_eci: &[SVector6]) -> Vec<SMatrix3> {
    batch_map(|x| rotation_eci_to_rtn(*x), x_eci)
}

/// Computes the RTN relative state of each deputy with respect to its chief.
///
/// Batch form of [`state_eci_to_rtn`]. Evaluation runs on the global thread pool for
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
/// - Deputy relative states in the chief RTN frame, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::states_eci_to_rtn;
/// use brahe::vector6_from_array;
///
/// let chief = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let deputies = vec![
///     state_koe_to_eci(vector6_from_array([R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05]), AngleFormat::Degrees),
///     state_koe_to_eci(vector6_from_array([R_EARTH + 702e3, 0.0012, 97.82, 15.02, 30.02, 45.02]), AngleFormat::Degrees),
/// ];
/// let rel = states_eci_to_rtn(&[chief], &deputies).unwrap();
/// assert_eq!(rel.len(), 2);
/// ```
pub fn states_eci_to_rtn(
    x_chief: &[SVector6],
    x_deputy: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(|c, d| state_eci_to_rtn(*c, *d), x_chief, x_deputy)
}

/// Computes the ECI state of each deputy from its RTN relative state and chief.
///
/// Batch form of [`state_rtn_to_eci`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// The chief and deputy arguments follow the broadcast rule: each has length 1
/// or the common batch length, so one chief may be paired with many deputies
/// and vice versa.
///
/// # Arguments
/// - `x_chief`: Chief Cartesian ECI states, length 1 or the batch length. Units: (*m*; *m/s*)
/// - `x_rel_rtn`: Deputy relative states in the chief RTN frame, length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Deputy Cartesian ECI states, in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, AngleFormat};
/// use brahe::coordinates::state_koe_to_eci;
/// use brahe::relative_motion::states_rtn_to_eci;
/// use brahe::vector6_from_array;
///
/// let chief = state_koe_to_eci(vector6_from_array([R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0]), AngleFormat::Degrees);
/// let rel = vec![vector6_from_array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0]); 2];
/// let deputies = states_rtn_to_eci(&[chief], &rel).unwrap();
/// assert_eq!(deputies.len(), 2);
/// ```
pub fn states_rtn_to_eci(
    x_chief: &[SVector6],
    x_rel_rtn: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_zip(|c, r| state_rtn_to_eci(*c, *r), x_chief, x_rel_rtn)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::AngleFormat;
    use crate::R_EARTH;
    use crate::coordinates::state_koe_to_eci;
    use crate::math::vector6_from_array;
    use crate::orbits::{mean_motion, perigee_velocity};
    use crate::utils::testing::setup_global_test_eop;
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    fn get_test_state() -> SVector6 {
        let sma = R_EARTH + 700e3; // Semi-major axis in meters
        SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0)
    }

    #[test]
    #[parallel]
    fn test_rotation_rtn_to_eci() {
        let x_eci = get_test_state();
        let p_eci = x_eci.fixed_rows::<3>(0);

        // Confirm that multiplying by the position vector yields the correct transformation
        let r_rtn = rotation_rtn_to_eci(x_eci);
        let r_eci = r_rtn * Vector3::new(1.0, 0.0, 0.0) * p_eci.norm();

        // Confirm that the transformed vector matches the original position vector
        assert!((r_eci - p_eci).norm() < 1e-6);
    }

    #[test]
    #[parallel]
    fn test_rotation_eci_to_rtn_inverse() {
        let x_eci = get_test_state();

        let r_rtn_to_eci = rotation_rtn_to_eci(x_eci);
        let r_eci_to_rtn = rotation_eci_to_rtn(x_eci);

        // Confirm that the product of the two rotation matrices is the identity matrix
        let identity = r_rtn_to_eci * r_eci_to_rtn;
        assert!((identity - SMatrix3::identity()).norm() < 1e-10);
    }

    #[test]
    #[parallel]
    fn test_omega_rtn_circular_orbit_matches_mean_motion() {
        let sma = R_EARTH + 700e3;
        let x_eci = get_test_state();

        // For a circular orbit the true-anomaly rate equals the mean motion
        let omega = omega_rtn(x_eci);
        assert_abs_diff_eq!(omega[0], 0.0, epsilon = 1e-18);
        assert_abs_diff_eq!(omega[1], 0.0, epsilon = 1e-18);
        assert_abs_diff_eq!(
            omega[2],
            mean_motion(sma, AngleFormat::Radians),
            epsilon = 1e-12
        );
    }

    #[test]
    #[parallel]
    fn test_state_eci_to_rtn_and_back() {
        let x_chief = get_test_state();
        let x_deputy = get_test_state() + SVector6::new(100.0, 200.0, 300.0, 0.1, 0.2, 0.3);
        let x_rel_rtn = state_eci_to_rtn(x_chief, x_deputy);
        let x_deputy_reconstructed = state_rtn_to_eci(x_chief, x_rel_rtn);
        assert!((x_deputy - x_deputy_reconstructed).norm() < 1e-6);
    }

    #[test]
    #[serial_test::parallel]
    fn test_state_eci_to_rtn_and_back_non_aligned() {
        setup_global_test_eop();

        // Use an inclined orbit where RTN != ECI axes
        let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
        let oe_deputy = SVector6::new(R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05);

        let x_chief = state_koe_to_eci(oe_chief, AngleFormat::Degrees);
        let x_deputy = state_koe_to_eci(oe_deputy, AngleFormat::Degrees);

        // Round-trip: ECI -> RTN -> ECI
        let x_rel_rtn = state_eci_to_rtn(x_chief, x_deputy);
        let x_deputy_reconstructed = state_rtn_to_eci(x_chief, x_rel_rtn);

        let pos_err =
            (x_deputy.fixed_rows::<3>(0) - x_deputy_reconstructed.fixed_rows::<3>(0)).norm();
        let vel_err =
            (x_deputy.fixed_rows::<3>(3) - x_deputy_reconstructed.fixed_rows::<3>(3)).norm();

        assert!(pos_err < 1e-8, "Position round-trip error: {pos_err} m");
        assert!(vel_err < 1e-8, "Velocity round-trip error: {vel_err} m/s");
    }

    #[test]
    #[serial_test::parallel]
    fn test_state_rtn_to_eci_and_back_non_aligned() {
        setup_global_test_eop();

        // Use an inclined orbit where RTN != ECI axes
        let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
        let x_chief = state_koe_to_eci(oe_chief, AngleFormat::Degrees);

        // Known RTN offset
        let x_rel_rtn = SVector6::new(1000.0, 500.0, -300.0, 0.1, -0.05, 0.02);

        // Round-trip: RTN -> ECI -> RTN
        let x_deputy = state_rtn_to_eci(x_chief, x_rel_rtn);
        let x_rel_rtn_recovered = state_eci_to_rtn(x_chief, x_deputy);

        let pos_err =
            (x_rel_rtn.fixed_rows::<3>(0) - x_rel_rtn_recovered.fixed_rows::<3>(0)).norm();
        let vel_err =
            (x_rel_rtn.fixed_rows::<3>(3) - x_rel_rtn_recovered.fixed_rows::<3>(3)).norm();

        assert!(pos_err < 1e-8, "Position round-trip error: {pos_err} m");
        assert!(vel_err < 1e-8, "Velocity round-trip error: {vel_err} m/s");
    }

    #[test]
    #[parallel]
    fn test_batch_rtn_match_scalar() {
        let chiefs: Vec<SVector6> = (0..3)
            .map(|i| {
                state_koe_to_eci(
                    vector6_from_array([
                        R_EARTH + 700e3 + 1e3 * i as f64,
                        0.001,
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
                        0.0015,
                        97.85,
                        15.05,
                        30.05,
                        45.05 + i as f64,
                    ]),
                    AngleFormat::Degrees,
                )
            })
            .collect();

        let rot = rotations_rtn_to_eci(&chiefs);
        let rot_inv = rotations_eci_to_rtn(&chiefs);
        let rel = states_eci_to_rtn(&chiefs, &deputies).unwrap();
        let rel_one_chief = states_eci_to_rtn(&chiefs[..1], &deputies).unwrap();
        let rel_one_deputy = states_eci_to_rtn(&chiefs, &deputies[..1]).unwrap();
        let back = states_rtn_to_eci(&chiefs, &rel).unwrap();
        let back_one = states_rtn_to_eci(&chiefs[..1], &rel_one_chief).unwrap();
        for i in 0..3 {
            assert_eq!(rot[i], rotation_rtn_to_eci(chiefs[i]));
            assert_eq!(rot_inv[i], rotation_eci_to_rtn(chiefs[i]));
            assert_eq!(rel[i], state_eci_to_rtn(chiefs[i], deputies[i]));
            assert_eq!(rel_one_chief[i], state_eci_to_rtn(chiefs[0], deputies[i]));
            assert_eq!(rel_one_deputy[i], state_eci_to_rtn(chiefs[i], deputies[0]));
            assert_eq!(back[i], state_rtn_to_eci(chiefs[i], rel[i]));
            assert_eq!(back_one[i], state_rtn_to_eci(chiefs[0], rel_one_chief[i]));
            for k in 0..3 {
                assert!((back[i][k] - deputies[i][k]).abs() < 1e-6);
            }
        }
        assert!(states_eci_to_rtn(&chiefs[..2], &deputies).is_err());
        assert!(rotations_rtn_to_eci(&[]).is_empty());
    }
}
