/*!
 * Provides the geocentric coordinate transformations.
 */

use std::f64::consts::PI;

use nalgebra::Vector3;

use crate::constants;
use crate::constants::AngleFormat;

/// Convert geocentric position to equivalent Earth-fixed position.
///
/// The valid input range for each component is:
/// - lon: [-inf, +inf]. Larger values will be wrapped appropriately
/// - lat: [-90, +90], Out-of-bounds values will result in an `Error`
/// - alt: [-inf, +inf]. All values are valid, but may give unintended results
///
/// # Arguments:
/// - `x_geoc`: Geocentric coordinates (lon, lat, altitude). Units: (*rad* or *deg* and *m*)
/// - `angle_format`: Format for angular elements (Radians or Degrees)
///
/// # Returns
/// - `x_ecef`: Earth-fixed coordinates. Units (*m*)
///
/// # Examples
/// ```
/// use brahe::constants::DEGREES;
/// use brahe::utils::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let geoc = vector3_from_array([0.0, 0.0, 0.0]);
/// let ecef = position_geocentric_to_ecef(geoc, DEGREES).unwrap();
/// // Returns state [R_EARTH, 0.0, 0.0]
/// ```
pub fn position_geocentric_to_ecef(
    x_geoc: Vector3<f64>,
    angle_format: AngleFormat,
) -> Result<Vector3<f64>, String> {
    // Convert angles to radians
    let (lon, lat) = match angle_format {
        AngleFormat::Degrees => (
            x_geoc[0] * constants::DEG2RAD,
            x_geoc[1] * constants::DEG2RAD,
        ),
        AngleFormat::Radians => (x_geoc[0], x_geoc[1]),
    };
    let alt = x_geoc[2];

    // Check validity of inputs
    if !(-PI / 2.0..=PI / 2.0).contains(&lat) {
        return Err(format!(
            "Input latitude out of range. Input must be between -90 and 90 degrees. Input: {}",
            lat
        ));
    }

    // Compute Earth-fixed position
    let r = constants::WGS84_A + alt;
    let x = r * lat.cos() * lon.cos();
    let y = r * lat.cos() * lon.sin();
    let z = r * lat.sin();

    Ok(Vector3::new(x, y, z))
}

/// Convert Earth-fixed position into equivalent of geocentric position.
///
/// # Arguments:
/// - `x_ecef`: Earth-fixed coordinates. Units (*m*)
/// - `angle_format`: Format for angular elements in output (Radians or Degrees)
///
/// # Returns
/// - `x_geoc`: Geocentric coordinates (lon, lat, altitude). Units: (*rad* or *deg* and *m*)
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, DEGREES};
/// use brahe::utils::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let ecef = vector3_from_array([R_EARTH, 0.0, 0.0]);
/// let geoc = position_ecef_to_geocentric(ecef, DEGREES);
/// // Returns state [0.0, 0.0, 0.0]
/// ```
pub fn position_ecef_to_geocentric(
    x_ecef: Vector3<f64>,
    angle_format: AngleFormat,
) -> Vector3<f64> {
    let x = x_ecef[0];
    let y = x_ecef[1];
    let z = x_ecef[2];

    // Compute geocentric coordinates
    let lon = y.atan2(x);
    let lat = z.atan2((x * x + y * y).sqrt());
    let alt = (x * x + y * y + z * z).sqrt() - constants::WGS84_A;

    // Convert angles to requested format
    match angle_format {
        AngleFormat::Degrees => {
            Vector3::new(lon * constants::RAD2DEG, lat * constants::RAD2DEG, alt)
        }
        AngleFormat::Radians => Vector3::new(lon, lat, alt),
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::constants::{DEGREES, RADIANS, WGS84_A};

    use super::*;

    #[test]
    fn test_position_geocentric() {
        let tol = 1.0e-7;

        // Test known position conversions
        let geoc1 = Vector3::new(0.0, 0.0, 0.0);
        let ecef1 = position_geocentric_to_ecef(geoc1, RADIANS).unwrap();

        assert_abs_diff_eq!(ecef1[0], WGS84_A, epsilon = tol);
        assert_abs_diff_eq!(ecef1[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(ecef1[2], 0.0, epsilon = tol);

        let geoc2 = Vector3::new(90.0, 0.0, 0.0);
        let ecef2 = position_geocentric_to_ecef(geoc2, DEGREES).unwrap();

        assert_abs_diff_eq!(ecef2[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(ecef2[1], WGS84_A, epsilon = tol);
        assert_abs_diff_eq!(ecef2[2], 0.0, epsilon = tol);

        let geoc3 = Vector3::new(0.0, 90.0, 0.0);
        let ecef3 = position_geocentric_to_ecef(geoc3, DEGREES).unwrap();

        assert_abs_diff_eq!(ecef3[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(ecef3[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(ecef3[2], WGS84_A, epsilon = tol);

        // Test two-input format
        let geoc = Vector3::new(0.0, 0.0, 0.0);
        let ecef = position_geocentric_to_ecef(geoc, RADIANS).unwrap();

        assert_abs_diff_eq!(ecef[0], WGS84_A, epsilon = tol);
        assert_abs_diff_eq!(ecef[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(ecef[2], 0.0, epsilon = tol);

        let geoc = Vector3::new(90.0, 0.0, 0.0);
        let ecef = position_geocentric_to_ecef(geoc, DEGREES).unwrap();

        assert_abs_diff_eq!(ecef[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(ecef[1], WGS84_A, epsilon = tol);
        assert_abs_diff_eq!(ecef[2], 0.0, epsilon = tol);

        let geoc = Vector3::new(0.0, 90.0, 0.0);
        let ecef = position_geocentric_to_ecef(geoc, DEGREES).unwrap();

        assert_abs_diff_eq!(ecef[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(ecef[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(ecef[2], WGS84_A, epsilon = tol);

        // Test circularity
        let geoc4 = position_ecef_to_geocentric(ecef1, DEGREES);
        let geoc5 = position_ecef_to_geocentric(ecef2, DEGREES);
        let geoc6 = position_ecef_to_geocentric(ecef3, DEGREES);

        assert_abs_diff_eq!(geoc4[0], geoc1[0], epsilon = tol);
        assert_abs_diff_eq!(geoc4[1], geoc1[1], epsilon = tol);
        assert_abs_diff_eq!(geoc4[2], geoc1[2], epsilon = tol);

        assert_abs_diff_eq!(geoc5[0], geoc2[0], epsilon = tol);
        assert_abs_diff_eq!(geoc5[1], geoc2[1], epsilon = tol);
        assert_abs_diff_eq!(geoc5[2], geoc2[2], epsilon = tol);

        assert_abs_diff_eq!(geoc6[0], geoc3[0], epsilon = tol);
        assert_abs_diff_eq!(geoc6[1], geoc3[1], epsilon = tol);
        assert_abs_diff_eq!(geoc6[2], geoc3[2], epsilon = tol);

        // Random point circularity
        let geoc = Vector3::new(77.875000, 20.975200, 0.000000);
        let ecef = position_geocentric_to_ecef(geoc, DEGREES).unwrap();
        let geocc = position_ecef_to_geocentric(ecef, DEGREES);
        assert_abs_diff_eq!(geoc[0], geocc[0], epsilon = tol);
        assert_abs_diff_eq!(geoc[1], geocc[1], epsilon = tol);
        assert_abs_diff_eq!(geoc[2], geocc[2], epsilon = tol);

        assert!(position_geocentric_to_ecef(Vector3::new(0.0, 90.1, 0.0), DEGREES).is_err());

        assert!(position_geocentric_to_ecef(Vector3::new(0.0, -90.1, 0.0), DEGREES).is_err());
    }
}
