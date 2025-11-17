/*!
 * Provides a geodetic coordinate transformations.
 */

use std::f64::consts::PI;

use nalgebra::Vector3;

use crate::constants;
use crate::constants::AngleFormat;

const ECC2: f64 = constants::WGS84_F * (2.0 - constants::WGS84_F);

/// Convert geodetic position to equivalent Earth-fixed position.
///
/// # Arguments:
/// - `x_geod`: Geodetic coordinates (lon, lat, altitude). Units: (*rad* or *deg* and *m*)
/// - `angle_format`: Format for angular coordinates (Radians or Degrees)
///
/// # Returns
/// - `x_ecef`: Earth-fixed coordinates. Units (*m*)
///
/// # Examples
/// ```
/// use brahe::constants::DEGREES;
/// use brahe::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let geod = vector3_from_array([0.0, 0.0, 0.0]);
/// let ecef = position_geodetic_to_ecef(geod, DEGREES).unwrap();
/// // Returns state [R_EARTH, 0.0, 0.0]
/// ```
#[allow(non_snake_case)]
pub fn position_geodetic_to_ecef(
    x_geod: Vector3<f64>,
    angle_format: AngleFormat,
) -> Result<Vector3<f64>, String> {
    let lon = match angle_format {
        AngleFormat::Degrees => x_geod[0] * constants::DEG2RAD,
        AngleFormat::Radians => x_geod[0],
    };
    let lat = match angle_format {
        AngleFormat::Degrees => x_geod[1] * constants::DEG2RAD,
        AngleFormat::Radians => x_geod[1],
    };
    let alt = x_geod[2];

    // Check validity of inputs
    if !(-PI / 2.0..=PI / 2.0).contains(&lat) {
        return Err(format!(
            "Input latitude out of range. Input must be between -90 and 90 degrees. Input: {}",
            lat
        ));
    }

    // Compute Earth-fixed position
    let N = constants::WGS84_A / (1.0 - ECC2 * lat.sin().powi(2)).sqrt();
    let x = (N + alt) * lat.cos() * lon.cos();
    let y = (N + alt) * lat.cos() * lon.sin();
    let z = ((1.0 - ECC2) * N + alt) * lat.sin();

    Ok(Vector3::new(x, y, z))
}

/// Convert Earth-fixed position into equivalent of geodetic position.
///
/// # Arguments:
/// - `x_ecef`: Earth-fixed coordinates. Units (*m*)
/// - `angle_format`: Format for angular coordinates in output (Radians or Degrees)
///
/// # Returns
/// - `x_geod`: Geodetic coordinates (lon, lat, altitude). Units: (*rad* or *deg* and *m*)
///
/// # Examples
/// ```
/// use brahe::constants::{R_EARTH, DEGREES};
/// use brahe::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let ecef = vector3_from_array([R_EARTH, 0.0, 0.0]);
/// let geoc = position_ecef_to_geodetic(ecef, DEGREES);
/// // Returns state [0.0, 0.0, 0.0]
/// ```
#[allow(non_snake_case)]
pub fn position_ecef_to_geodetic(x_ecef: Vector3<f64>, angle_format: AngleFormat) -> Vector3<f64> {
    let x = x_ecef[0];
    let y = x_ecef[1];
    let z = x_ecef[2];

    // Compute intermediate quantities
    let eps = f64::EPSILON * 1.0e3;
    let rho2 = x * x + y * y;
    let mut dz = ECC2 * z;
    let mut N = 0.0;

    // Iterative refine coordinate estimate
    let mut iter = 0;
    while iter < 10 {
        let zdz = z + dz;
        let Nh = (rho2 + zdz * zdz).sqrt();
        let sinphi = zdz / Nh;
        N = constants::WGS84_A / (1.0 - ECC2 * sinphi * sinphi).sqrt();
        let dz_new = N * ECC2 * sinphi;

        // Check convergence requirement
        if (dz - dz_new).abs() < eps {
            break;
        }

        dz = dz_new;
        iter += 1;
    }

    if iter == 20 {
        panic!("Reached maximum number of iterations.");
    }

    // Extract geodetic coordiantes
    let zdz = z + dz;
    let lon = y.atan2(x);
    let lat = zdz.atan2(rho2.sqrt());
    let alt = (rho2 + zdz * zdz).sqrt() - N;

    match angle_format {
        AngleFormat::Degrees => {
            Vector3::new(lon * constants::RAD2DEG, lat * constants::RAD2DEG, alt)
        }
        AngleFormat::Radians => Vector3::new(lon, lat, alt),
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::constants::{DEGREES, RADIANS, WGS84_A, WGS84_F};

    use super::*;

    #[test]
    fn test_position_geodetic() {
        let tol = 1.0e-7;

        // Test known position conversions
        let geod1 = Vector3::new(0.0, 0.0, 0.0);
        let ecef1 = position_geodetic_to_ecef(geod1, RADIANS).unwrap();

        assert_abs_diff_eq!(ecef1[0], WGS84_A, epsilon = tol);
        assert_abs_diff_eq!(ecef1[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(ecef1[2], 0.0, epsilon = tol);

        let geod2 = Vector3::new(90.0, 0.0, 0.0);
        let ecef2 = position_geodetic_to_ecef(geod2, DEGREES).unwrap();

        assert_abs_diff_eq!(ecef2[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(ecef2[1], WGS84_A, epsilon = tol);
        assert_abs_diff_eq!(ecef2[2], 0.0, epsilon = tol);

        let geod3 = Vector3::new(0.0, 90.0, 0.0);
        let ecef3 = position_geodetic_to_ecef(geod3, DEGREES).unwrap();

        assert_abs_diff_eq!(ecef3[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(ecef3[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(ecef3[2], WGS84_A * (1.0 - WGS84_F), epsilon = tol);

        // Test two-input format
        let geod = Vector3::new(0.0, 0.0, 0.0);
        let ecef = position_geodetic_to_ecef(geod, RADIANS).unwrap();

        assert_abs_diff_eq!(ecef[0], WGS84_A, epsilon = tol);
        assert_abs_diff_eq!(ecef[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(ecef[2], 0.0, epsilon = tol);

        let geod = Vector3::new(90.0, 0.0, 0.0);
        let ecef = position_geodetic_to_ecef(geod, DEGREES).unwrap();

        assert_abs_diff_eq!(ecef[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(ecef[1], WGS84_A, epsilon = tol);
        assert_abs_diff_eq!(ecef[2], 0.0, epsilon = tol);

        let geod = Vector3::new(0.0, 90.0, 0.0);
        let ecef = position_geodetic_to_ecef(geod, DEGREES).unwrap();

        assert_abs_diff_eq!(ecef[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(ecef[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(ecef[2], WGS84_A * (1.0 - WGS84_F), epsilon = tol);

        // Test circularity
        let geod4 = position_ecef_to_geodetic(ecef1, DEGREES);
        let geod5 = position_ecef_to_geodetic(ecef2, DEGREES);
        let geod6 = position_ecef_to_geodetic(ecef3, DEGREES);

        assert_abs_diff_eq!(geod4[0], geod1[0], epsilon = tol);
        assert_abs_diff_eq!(geod4[1], geod1[1], epsilon = tol);
        assert_abs_diff_eq!(geod4[2], geod1[2], epsilon = tol);

        assert_abs_diff_eq!(geod5[0], geod2[0], epsilon = tol);
        assert_abs_diff_eq!(geod5[1], geod2[1], epsilon = tol);
        assert_abs_diff_eq!(geod5[2], geod2[2], epsilon = tol);

        assert_abs_diff_eq!(geod6[0], geod3[0], epsilon = tol);
        assert_abs_diff_eq!(geod6[1], geod3[1], epsilon = tol);
        assert_abs_diff_eq!(geod6[2], geod3[2], epsilon = tol);

        // Random point circularity
        let geod = Vector3::new(77.875000, 20.975200, 0.000000);
        let ecef = position_geodetic_to_ecef(geod, DEGREES).unwrap();
        let geodd = position_ecef_to_geodetic(ecef, DEGREES);
        assert_abs_diff_eq!(geod[0], geodd[0], epsilon = tol);
        assert_abs_diff_eq!(geod[1], geodd[1], epsilon = tol);
        assert_abs_diff_eq!(geod[2], geodd[2], epsilon = tol);

        assert!(position_geodetic_to_ecef(Vector3::new(0.0, 90.1, 0.0), DEGREES).is_err());

        assert!(position_geodetic_to_ecef(Vector3::new(0.0, -90.1, 0.0), DEGREES).is_err());
    }
}
