/*!
 Implement central-body gravity force models.
 */

use nalgebra::Vector3;

/// Compute the acceleration due to point-mass gravity.
///
/// # Arguments
///
/// - `r_object`: Position vector of the object.
/// - `r_central_body`: Position vector of the central body. If the central body is at the origin, this is the zero vector.
/// - `gm`: Product of the gravitational parameter and the mass of the central body.
///
/// # Returns
///
/// - `a_grav` : Acceleration due to gravity of the central body.
///
/// # Examples
///
/// ```
/// use brahe::constants::{R_EARTH, GM_EARTH};
/// use brahe::orbit_dynamics::acceleration_point_mass_gravity;
/// use nalgebra::Vector3;
///
/// let r_object = Vector3::new(R_EARTH, 0.0, 0.0);
/// let r_central_body = Vector3::new(0.0, 0.0, 0.0);
///
/// let a_grav = acceleration_point_mass_gravity(&r_object, &r_central_body, GM_EARTH);
///
/// // Acceleration should be in the negative x-direction and magnitude should be GM_EARTH / R_EARTH^2
/// // Roughly -9.81 m/s^2
/// assert!((a_grav - Vector3::new(-GM_EARTH / R_EARTH.powi(2), 0.0, 0.0)).norm() < 1e-12);
/// ```
///
/// # References
///
/// - TODO: Add references
pub fn acceleration_point_mass_gravity(
    r_object: &Vector3<f64>,
    r_central_body: &Vector3<f64>,
    gm: f64,
) -> Vector3<f64> {
    let d = r_object - r_central_body;

    let d_norm = d.norm();
    let r_central_body_norm = r_central_body.norm();

    if r_central_body_norm != 0.0 {
        -gm * ( d / d_norm.powi(3) - r_central_body / r_central_body_norm.powi(3) )
    } else {
        -gm * d / d_norm.powi(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{R_EARTH, GM_EARTH};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_acceleration_point_mass_gravity() {
        let r_object = Vector3::new(R_EARTH, 0.0, 0.0);
        let r_central_body = Vector3::new(0.0, 0.0, 0.0);

        let a_grav = acceleration_point_mass_gravity(&r_object, &r_central_body, GM_EARTH);

        // Acceleration should be in the negative x-direction and magnitude should be GM_EARTH / R_EARTH^2
        // Roughly -9.8 m/s^2
        assert_abs_diff_eq!(a_grav[0], -GM_EARTH / R_EARTH.powi(2), epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[2], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav.norm(), 9.798, epsilon = 1e-3);

        let r_object = Vector3::new(0.0, R_EARTH, 0.0);
        let a_grav = acceleration_point_mass_gravity(&r_object, &r_central_body, GM_EARTH);

        // Acceleration should be in the negative y-direction and magnitude should be GM_EARTH / R_EARTH^2
        // Roughly -9.8 m/s^2
        assert_abs_diff_eq!(a_grav[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[1], -GM_EARTH / R_EARTH.powi(2), epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[2], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav.norm(), 9.798, epsilon = 1e-3);

        let r_object = Vector3::new(0.0, 0.0, R_EARTH);
        let a_grav = acceleration_point_mass_gravity(&r_object, &r_central_body, GM_EARTH);

        // Acceleration should be in the negative z-direction and magnitude should be GM_EARTH / R_EARTH^2
        // Roughly -9.8 m/s^2
        assert_abs_diff_eq!(a_grav[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[2], -GM_EARTH / R_EARTH.powi(2), epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav.norm(), 9.798, epsilon = 1e-3);
    }
}