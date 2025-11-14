/*!
 * Earth-Centered Inertial (ECI) to Radial, Along-Track, Cross-Track (RTN) Frame Transformations
 */

use crate::utils::{SMatrix3, SVector6};
#[allow(unused_imports)]
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
/// use brahe::utils::SVector6;
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
/// use brahe::utils::SVector6;
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

#[cfg(test)]
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
}
