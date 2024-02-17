/*!
 Models of solar radiation pressure.
*/

use crate::constants::{AU, R_EARTH, R_SUN};
use nalgebra::Vector3;

/// Calculate the acceleration due to solar radiation pressure.
///
/// # Arguments
///
/// - `r_object`: Position vector of the object.
/// - `r_sun`: Position vector of the sun. If the sun is at the origin, this is the zero vector.
/// - `mass`: Mass of the object.
/// - `cr`: Coefficient of reflectivity.
/// - `area`: Cross-sectional area of the object.
/// - `p0`: Solar radiation pressure at 1 AU.
///
/// # Returns
///
/// - `a_srp` : Acceleration due to solar radiation pressure.
///
/// # Examples
///
/// ```
/// use brahe::orbit_dynamics::acceleration_solar_radiation_pressure;
/// use nalgebra::Vector3;
/// use brahe::constants::AU;
///
/// let r_object = Vector3::new(AU, 0.0, 0.0);
/// let r_sun = Vector3::new(0.0, 0.0, 0.0);
///
/// let a_srp = acceleration_solar_radiation_pressure(&r_object, &r_sun, 1.0, 1.0, 1.0, 4.5e-6);
///
/// // Acceleration should be in the negative x-direction and magnitude should be 4.5e-6 AU^2
/// assert_eq!(a_srp, Vector3::new(4.5e-6, 0.0, 0.0));
/// ```
pub fn acceleration_solar_radiation_pressure(
    r_object: &Vector3<f64>,
    r_sun: &Vector3<f64>,
    mass: f64,
    cr: f64,
    area: f64,
    p0: f64,
) -> Vector3<f64> {
    let d = r_object - r_sun;

    d * cr*(area/mass)*p0*AU.powi(2) / d.norm().powi(3)
}

/// Calculate the fraction of the object that is illuminated by the sun using a conical model
/// for Earth shadowing.
///
/// # Arguments
///
/// - `r_object`: Position vector of the object in the ECI frame.
/// - `r_sun`: Position vector of the sun. If the sun is at the origin, this is the zero vector.
///
/// # Returns
///
/// - `nu`: Illumination fraction of the object (0.0 = fully shadowed, 1.0 = fully illuminated).
///
/// # Examples
///
/// ```
/// use brahe::orbit_dynamics::eclipse_conical;
/// use nalgebra::Vector3;
/// use brahe::constants::R_EARTH;
///
/// let r_object = Vector3::new(R_EARTH, 0.0, 0.0);
/// let r_sun = Vector3::new(0.0, 0.0, 0.0);
///
/// let nu = eclipse_conical(&r_object, &r_sun);
///
/// // The object is shadowed, so the illumination fraction should be 0.0
/// assert_eq!(nu, 0.0);
/// ```
#[allow(non_snake_case)] // To better comply with the literature
pub fn eclipse_conical(r_object: &Vector3<f64>, r_sun: &Vector3<f64>) -> f64 {

    // Occultation Geometry
    let a = (R_SUN/(r_sun - r_object).norm()).asin();
    let b = (R_EARTH/r_object.norm()).asin();
    let c = (r_object.dot(&(r_sun-r_object))/(r_object.norm()*(r_sun-r_object).norm())).acos();

    // Test Occulation Conditions and return illumination fraction
    if (a - b).abs() < c && c < (a + b) {
        // Partial occultation

        let xx = (c.powi(2) + a.powi(2) - b.powi(2)) / (2.0 * c);
        let yy = (a.powi(2) - xx.powi(2)).sqrt();
        let A = a.powi(2) * (xx / a).acos() + b.powi(2) * ((c - xx) / b).acos() - c * yy;

        1.0 - A / (std::f64::consts::PI * a.powi(2))
    } else if (a + b) <= c {
        // No occultation
        1.0
    } else {
        // Full occultation
        0.0
    }
}

/// Calculate the fraction of the object that is illuminated by the sun using a cylindrical model
/// for Earth shadowing.
///
/// # Arguments
///
/// - `r_object`: Position vector of the object in the ECI frame.
/// - `r_sun`: Position vector of the sun. If the sun is at the origin, this is the zero vector..
///
/// # Returns
///
/// - `nu`: Illumination fraction of the object.
///
/// # Examples
///
/// ```
/// use brahe::orbit_dynamics::eclipse_cylindrical;
/// use nalgebra::Vector3;
/// use brahe::constants::R_EARTH;
///
/// let r_object = Vector3::new(R_EARTH, 0.0, 0.0);
/// let r_sun = Vector3::new(0.0, 0.0, 0.0);
///
/// let nu = eclipse_cylindrical(&r_object, &r_sun);
///
/// // The object is shadowed, so the illumination fraction should be 0.0
/// assert_eq!(nu, 0.0);
/// ```
pub fn eclipse_cylindrical(r_object: &Vector3<f64>, r_sun: &Vector3<f64>) -> f64 {
    // Unit vector in the direction of the sun
    let e_sun = r_sun / r_sun.norm();

    // Projection of spacecraft position vector onto the sun vector
    let r_proj = r_object.dot(&e_sun);

    // Compute illumination fraction
    let mut nu = 0.0;
    if r_proj >= 1.0 || (r_object - r_proj*e_sun).norm() > R_EARTH {
        nu = 1.0;
    }

    nu
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_acceleration_solar_radiation_pressure() {
        let r_object = Vector3::new(AU, 0.0, 0.0);
        let r_sun = Vector3::new(0.0, 0.0, 0.0);

        let a_srp = acceleration_solar_radiation_pressure(&r_object, &r_sun, 1.0, 1.0, 1.0, 4.5e-6);

        // Acceleration should be in the negative x-direction and magnitude should be 4.5e-6 AU^2
        assert_abs_diff_eq!(a_srp[0], 4.5e-6, epsilon = 1e-12);
        assert_abs_diff_eq!(a_srp[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_srp[2], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_srp.norm(), 4.5e-6, epsilon = 1e-12);
    }

    #[test]
    fn test_eclipse_conical() {
        // TODO: Add tests
    }

    #[test]
    fn test_eclipse_cylindrical() {
        // TODO: Add tests
    }
}