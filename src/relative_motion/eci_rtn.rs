/*!
 * Earth-Centered Inertial (ECI) to Radial, Along-Track, Cross-Track (RTN) Frame Transformations
 */

use crate::math::{SMatrix3, SVector6};
use nalgebra::Vector3;

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
/// use brahe::coordinates::state_osculating_to_cartesian;
/// use brahe::relative_motion::*;
///
/// // Define chief and deputy satellite positions
/// let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
/// let oe_deputy = SVector6::new(R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05);
///
/// let x_chief = state_osculating_to_cartesian(oe_chief, AngleFormat::Degrees);
/// let x_deputy = state_osculating_to_cartesian(oe_deputy, AngleFormat::Degrees);
///
/// let x_rel_rtn = state_eci_to_rtn(x_chief, x_deputy);
/// ```
pub fn state_eci_to_rtn(x_chief: SVector6, x_deputy: SVector6) -> SVector6 {
    // NOTE: This could potentially be more accurately revised based on equations in section 4.7.1 of Alfriend

    // Extract chief position and velocity
    let rc = x_chief.fixed_rows::<3>(0);
    let vc = x_chief.fixed_rows::<3>(3);

    // Get RTN rotation matrix
    let r_eci_to_rtn = rotation_eci_to_rtn(x_chief);

    // Relative position and velocity in ECI frame
    let rho_eci = x_deputy.fixed_rows::<3>(0) - x_chief.fixed_rows::<3>(0);
    let rho_dot_eci = x_deputy.fixed_rows::<3>(3) - x_chief.fixed_rows::<3>(3);

    // Get angular velocity of RTN frame with respect to ECI frame (Alfriend equation 2.16)
    let f_dot = (rc.cross(&vc)).norm() / (rc.norm().powi(2));
    let omega = Vector3::new(0.0, 0.0, f_dot);

    // Transform relative position and velocity to RTN frame
    let rho_rtn = r_eci_to_rtn * rho_eci;
    let rho_dot_rtn = r_eci_to_rtn * rho_dot_eci - omega.cross(&rho_eci);

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
/// use brahe::coordinates::state_osculating_to_cartesian;
/// use brahe::relative_motion::*;
///
/// // Define chief and deputy satellite positions
/// let oe_chief = SVector6::new(R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0);
/// let oe_deputy = SVector6::new(R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05);
///
/// let x_chief = state_osculating_to_cartesian(oe_chief, AngleFormat::Degrees);
/// let x_deputy = state_osculating_to_cartesian(oe_deputy, AngleFormat::Degrees);
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

    // Get angular velocity of RTN frame with respect to ECI frame (Alfriend equation 2.16)
    let f_dot = (rc.cross(&vc)).norm() / (rc.norm().powi(2));
    let omega = Vector3::new(0.0, 0.0, f_dot);

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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::R_EARTH;
    use crate::orbits::perigee_velocity;

    fn get_test_state() -> SVector6 {
        let sma = R_EARTH + 700e3; // Semi-major axis in meters
        SVector6::new(sma, 0.0, 0.0, 0.0, perigee_velocity(sma, 0.0), 0.0)
    }

    #[test]
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
    fn test_rotation_eci_to_rtn_inverse() {
        let x_eci = get_test_state();

        let r_rtn_to_eci = rotation_rtn_to_eci(x_eci);
        let r_eci_to_rtn = rotation_eci_to_rtn(x_eci);

        // Confirm that the product of the two rotation matrices is the identity matrix
        let identity = r_rtn_to_eci * r_eci_to_rtn;
        assert!((identity - SMatrix3::identity()).norm() < 1e-10);
    }

    #[test]
    fn test_state_eci_to_rtn_and_back() {
        let x_chief = get_test_state();
        let x_deputy = get_test_state() + SVector6::new(100.0, 200.0, 300.0, 0.1, 0.2, 0.3);
        let x_rel_rtn = state_eci_to_rtn(x_chief, x_deputy);
        let x_deputy_reconstructed = state_rtn_to_eci(x_chief, x_rel_rtn);
        assert!((x_deputy - x_deputy_reconstructed).norm() < 1e-6);
    }
}
