/*!
 * Provides topocentric coordiante transformations.
 */

use std::f64::consts::PI;

use nalgebra::{Matrix3, Vector3};

use crate::coordinates::coordinate_types::EllipsoidalConversionType;
use crate::coordinates::geocentric::position_ecef_to_geocentric;
use crate::coordinates::geodetic::position_ecef_to_geodetic;
use crate::utils::math::{from_degrees, to_degrees};

/// Compute the rotation matrix from body-fixed to East-North-Zenith (ENZ)
/// Cartesian coordinates for a given set of coordinates on an ellipsoidal body.
/// The ellipsoidal coordinates can either be geodetic or geocentric.
///
/// # Args:
/// - `x_ellipsoid`: Ellipsoidal coordinates.  Expected format (lon, lat, alt)
/// - `use_degrees`: Interprets input as (deg) if `true` or (rad) if `false`
///
/// # Returns:
/// - `E`: Earth-fixed to Topocentric rotation matrix
///
/// # Examples:
/// ```
/// use brahe::utils::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_geo = vector3_from_array([30.0, 60.0, 0.0]);
/// let rot = rotation_ellipsoid_to_enz(&x_geo, true);
/// ```
pub fn rotation_ellipsoid_to_enz(x_ellipsoid: &Vector3<f64>, as_degrees: bool) -> Matrix3<f64> {
    let lon = from_degrees(x_ellipsoid[0], as_degrees);
    let lat = from_degrees(x_ellipsoid[1], as_degrees);

    // Construct Rotation matrix
    Matrix3::new(
        -lon.sin(),
        lon.cos(),
        0.0, // E-base vector
        -lat.sin() * lon.cos(),
        -lat.sin() * lon.sin(),
        lat.cos(), // N-base vector
        lat.cos() * lon.cos(),
        lat.cos() * lon.sin(),
        lat.sin(), // Z-base vector
    )
}

/// Compute the rotation matrix from East-North-Zenith (ENZ) to body-fixed
/// Cartesian coordinates for a given set of coordinates on an ellipsoidal body.
/// The ellipsoidal coordinates can either be geodetic or geocentric.
///
/// # Args:
/// - `x_ellipsoid`: Ellipsoidal coordinates.  Expected format (lon, lat, alt)
/// - `use_degrees`: Interprets input as (deg) if `true` or (rad) if `false`
///
/// # Returns:
/// - `E`: Topocentric to Earth-fixed rotation matrix
///
/// # Examples:
/// ```
/// use brahe::utils::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_geo = vector3_from_array([30.0, 60.0, 0.0]);
/// let rot = rotation_enz_to_ellipsoid(&x_geo, true);
/// ```
pub fn rotation_enz_to_ellipsoid(x_ellipsoid: &Vector3<f64>, as_degrees: bool) -> Matrix3<f64> {
    rotation_ellipsoid_to_enz(x_ellipsoid, as_degrees).transpose()
}

/// Computes the relative state in East-North-Zenith (ENZ) coordinates for a target
/// object in the ECEF frame with respect to a fixed location (station) also in
/// the ECEF frame.
///
/// # Args:
/// - `location_ecef`: Cartesian position of the observing station in the ECEF frame.
/// - `x_ecef`: Cartesian position of the observed object in the ECEF frame
/// - `conversion_type`: Type of conversion to apply for computing the topocentric frame based on station coordinates.
///
/// # Returns:
/// - `r_rel`: Relative position of object in ENZ coordinates based on the station location.
///
/// # Examples:
/// ```
/// use brahe::constants::R_EARTH;
/// use brahe::utils::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_station = vector3_from_array([R_EARTH, 0.0, 0.0]);
/// let x_sat = vector3_from_array([R_EARTH + 500.0e3, 0.0, 0.0]);
///
/// let r_enz = relative_position_ecef_to_enz(
///     &x_station, &x_sat, EllipsoidalConversionType::Geocentric
/// );
/// ```
#[allow(non_snake_case)]
pub fn relative_position_ecef_to_enz(
    location_ecef: &Vector3<f64>,
    r_ecef: &Vector3<f64>,
    conversion_type: EllipsoidalConversionType,
) -> Vector3<f64> {
    // Create ENZ rotation matrix
    let E = match conversion_type {
        EllipsoidalConversionType::Geocentric => {
            rotation_ellipsoid_to_enz(&position_ecef_to_geocentric(&location_ecef, false), false)
        }
        EllipsoidalConversionType::Geodetic => {
            rotation_ellipsoid_to_enz(&position_ecef_to_geodetic(&location_ecef, false), false)
        }
    };

    // Compute range transformation
    let r = r_ecef - location_ecef;
    E * r
}

/// Computes the absolute Earth-fixed coordinates for an object given its relative
/// position in East-North-Zenith (ENZ) coordinates and the Cartesian body-fixed
/// coordinates of the observing location/station.
///
/// # Args:
/// - `location_ecef`: Cartesian position of the observing station in the ECEF frame.
/// - `r_rel`: Relative position of object in ENZ coordinates based on the station location.
/// - `conversion_type`: Type of conversion to apply for computing the topocentric frame based on station coordinates.
///
/// # Returns:
/// - `r_ecef`: Cartesian position of the observed object in the ECEF frame
///
/// # Examples:
/// ```
/// use brahe::constants::R_EARTH;
/// use brahe::utils::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_station = vector3_from_array([R_EARTH, 0.0, 0.0]);
/// let r_enz = vector3_from_array([0.0, 0.0, 500.0e3]);
///
/// let r_ecef = relative_position_enz_to_ecef(
///     &x_station, &r_enz, EllipsoidalConversionType::Geocentric
/// );
/// ```
#[allow(non_snake_case)]
pub fn relative_position_enz_to_ecef(
    location_ecef: &Vector3<f64>,
    r_enz: &Vector3<f64>,
    conversion_type: EllipsoidalConversionType,
) -> Vector3<f64> {
    // Create ENZ rotation matrix
    let Et = match conversion_type {
        EllipsoidalConversionType::Geocentric => {
            rotation_enz_to_ellipsoid(&position_ecef_to_geocentric(&location_ecef, false), false)
        }
        EllipsoidalConversionType::Geodetic => {
            rotation_enz_to_ellipsoid(&position_ecef_to_geodetic(&location_ecef, false), false)
        }
    };

    // Compute range transformation
    let r = r_enz;
    location_ecef + Et * r
}

/// Compute the rotation matrix from body-fixed to South-East-Zenith (SEZ)
/// Cartesian coordinates for a given set of coordinates on an ellipsoidal body.
/// The ellipsoidal coordinates can either be geodetic or geocentric.
///
/// # Args:
/// - `x_ellipsoid`: Ellipsoidal coordinates.  Expected format (lon, lat, alt)
/// - `use_degrees`: Interprets input as (deg) if `true` or (rad) if `false`
///
/// # Returns:
/// - `E`: Earth-fixed to Topocentric rotation matrix
///
/// # Examples:
/// ```
/// use brahe::utils::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_geo = vector3_from_array([30.0, 60.0, 0.0]);
/// let rot = rotation_sez_to_ellipsoid(&x_geo, true);
/// ```
pub fn rotation_ellipsoid_to_sez(x_ellipsoid: &Vector3<f64>, as_degrees: bool) -> Matrix3<f64> {
    let lon = from_degrees(x_ellipsoid[0], as_degrees);
    let lat = from_degrees(x_ellipsoid[1], as_degrees);

    // Construct Rotation matrix
    Matrix3::new(
        lat.sin() * lon.cos(),
        lat.sin() * lon.sin(),
        -lat.cos(), // S-base vector
        -lon.sin(),
        lon.cos(),
        0.0, // E-base vector
        lat.cos() * lon.cos(),
        lat.cos() * lon.sin(),
        lat.sin(), // Z-base vector
    )
}

/// Compute the rotation matrix from South-East-Zenith (SEZ) to body-fixed
/// Cartesian coordinates for a given set of coordinates on an ellipsoidal body.
/// The ellipsoidal coordinates can either be geodetic or geocentric.
///
/// # Args:
/// - `x_ellipsoid`: Ellipsoidal coordinates. Expected format (lon, lat, alt)
/// - `use_degrees`: Interprets input as (deg) if `true` or (rad) if `false`
///
/// # Returns:
/// - `E`: Topocentric to Earth-fixed rotation matrix
///
/// # Examples:
/// ```
/// use brahe::utils::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_geo = vector3_from_array([30.0, 60.0, 0.0]);
/// let rot = rotation_sez_to_ellipsoid(&x_geo, true);
/// ```
pub fn rotation_sez_to_ellipsoid(x_ellipsoid: &Vector3<f64>, as_degrees: bool) -> Matrix3<f64> {
    rotation_ellipsoid_to_sez(&x_ellipsoid, as_degrees).transpose()
}

/// Computes the relative state in South-East-Zenith (SEZ) coordinates for a target
/// object in the ECEF frame with respect to a fixed location (station) also in
/// the ECEF frame.
///
/// # Args:
/// - `location_ecef`: Cartesian position of the observing station in the ECEF frame.
/// - `r_ecef`: Cartesian position of the observed object in the ECEF frame
/// - `conversion_type`: Type of conversion to apply for computing the topocentric frame based on station coordinates.
///
/// # Returns:
/// - `r_rel`: Relative position of object in ENZ coordinates based on the station location.
///
/// # Examples:
/// ```
/// use brahe::constants::R_EARTH;
/// use brahe::utils::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_station = vector3_from_array([R_EARTH, 0.0, 0.0]);
/// let x_sat = vector3_from_array([R_EARTH + 500.0e3, 0.0, 0.0]);
///
/// let r_enz = relative_position_ecef_to_enz(
///     &x_station, &x_sat, EllipsoidalConversionType::Geocentric
/// );
/// ```
#[allow(non_snake_case)]
pub fn relative_position_ecef_to_sez(
    location_ecef: &Vector3<f64>,
    r_ecef: &Vector3<f64>,
    conversion_type: EllipsoidalConversionType,
) -> Vector3<f64> {
    // Create ENZ rotation matrix
    let E = match conversion_type {
        EllipsoidalConversionType::Geocentric => {
            rotation_ellipsoid_to_sez(&position_ecef_to_geocentric(&location_ecef, false), false)
        }
        EllipsoidalConversionType::Geodetic => {
            rotation_ellipsoid_to_sez(&position_ecef_to_geodetic(&location_ecef, false), false)
        }
    };

    // Compute range transformation
    let r = r_ecef - location_ecef;
    E * r
}

/// Computes the absolute Earth-fixed coordinates for an object given its relative
/// position in East-North-Zenith (ENZ) coordinates and the Cartesian body-fixed
/// coordinates of the observing location/station.
///
/// # Args:
/// - `location_ecef`: Cartesian position of the observing station in the ECEF frame.
/// - `r_rel`: Relative position of object in ENZ coordinates based on the station location.
/// - `conversion_type`: Type of conversion to apply for computing the topocentric frame based on station coordinates.
///
/// # Returns:
/// - `r_ecef`: Cartesian position of the observed object in the ECEF frame
///
/// # Examples:
/// ```
/// use brahe::constants::R_EARTH;
/// use brahe::utils::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_station = vector3_from_array([R_EARTH, 0.0, 0.0]);
/// let r_sez = vector3_from_array([0.0, 0.0, 500.0e3]);
///
/// let r_ecef = relative_position_sez_to_ecef(
///     &x_station, &r_sez, EllipsoidalConversionType::Geocentric
/// );
/// ```
#[allow(non_snake_case)]
pub fn relative_position_sez_to_ecef(
    location_ecef: &Vector3<f64>,
    x_sez: &Vector3<f64>,
    conversion_type: EllipsoidalConversionType,
) -> Vector3<f64> {
    // Create SEZ rotation matrix
    let Et = match conversion_type {
        EllipsoidalConversionType::Geocentric => {
            rotation_sez_to_ellipsoid(&position_ecef_to_geocentric(&location_ecef, false), false)
        }
        EllipsoidalConversionType::Geodetic => {
            rotation_sez_to_ellipsoid(&position_ecef_to_geodetic(&location_ecef, false), false)
        }
    };

    // Compute range transformation
    let r = x_sez;
    location_ecef + Et * r
}

/// Converts East-North-Zenith topocentric coordinates of an location
/// into azimuth, elevation, and range from that same location. Azimuth is measured
/// clockwise from North.
///
/// # Args:
/// - `x_enz`: Relative Cartesian position of object to location East-North-Up coordinates. Units: (*m*)
/// - `use_degrees`: Returns output as (*deg*) if `true` or (*rad*) if `false`
///
/// # Returns:
/// - `x_azel`: Azimuth, elevation and range. Units: (*angle*, *angle*, *m*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_EARTH;
/// use brahe::utils::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_enz = vector3_from_array([100.0, 0.0, 0.0]);
///
/// let x_azel = position_enz_to_azel(&x_enz, true);
/// // x_azel = [90.0, 0.0, 100.0]
/// ```
pub fn position_enz_to_azel(x_enz: &Vector3<f64>, as_degrees: bool) -> Vector3<f64> {
    // Range
    let rho = x_enz.norm();

    // Elevation
    let el = x_enz[2].atan2((x_enz[0].powi(2) + x_enz[1].powi(2)).sqrt());

    // Azimuth
    let az = if el != PI / 2.0 {
        let azt = x_enz[0].atan2(x_enz[1]);

        if azt >= 0.0 {
            azt
        } else {
            azt + 2.0 * PI
        }
    } else {
        // If at peak elevation azimuth is ambiguous so define as 0.0
        0.0
    };

    Vector3::new(to_degrees(az, as_degrees), to_degrees(el, as_degrees), rho)
}

/// Converts South-East-Zenith topocentric coordinates of an location
/// into azimuth, elevation, and range from that same location. Azimuth is measured
/// clockwise from North.
///
/// # Args:
/// - `x_sez`: Relative Cartesian position of object to location South-East-Zenith coordinates. Units: (*m*)
/// - `use_degrees`: Returns output as (*deg*) if `true` or (*rad*) if `false`
///
/// # Returns:
/// - `x_azel`: Azimuth, elevation and range. Units: (*angle*, *angle*, *m*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_EARTH;
/// use brahe::utils::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_enz = vector3_from_array([0.0, 100.0, 0.0]);
///
/// let x_azel = position_sez_to_azel(&x_enz, true);
/// // x_azel = [90.0, 0.0, 100.0]
/// ```
pub fn position_sez_to_azel(x_sez: &Vector3<f64>, as_degrees: bool) -> Vector3<f64> {
    // Range
    let rho = x_sez.norm();

    // Elevation
    let el = x_sez[2].atan2((x_sez[0].powi(2) + x_sez[1].powi(2)).sqrt());

    // Azimuth
    let az = if el != PI / 2.0 {
        let azt = (x_sez[1]).atan2(-x_sez[0]);

        if azt >= 0.0 {
            azt
        } else {
            azt + 2.0 * PI
        }
    } else {
        // If at peak elevation azimuth is ambiguous so define as 0.0
        0.0
    };

    Vector3::new(to_degrees(az, as_degrees), to_degrees(el, as_degrees), rho)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::{position_geocentric_to_ecef, position_geodetic_to_ecef, R_EARTH};

    use super::*;

    #[test]
    fn test_rotation_ellipsoid_to_enz() {
        // Epsilon Tolerance
        let tol = f64::EPSILON;

        // Test aligned coordinates
        let x_sta = Vector3::new(0.0, 0.0, 0.0);
        let rot1 = rotation_ellipsoid_to_enz(&x_sta, true);

        // ECEF input X - [1, 0, 0] - Expected output is ENZ Z-dir
        assert_abs_diff_eq!(rot1[(0, 0)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 0)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 0)], 1.0, epsilon = tol);

        // ECEF input Y - [0, 1, 0] - Expected output is ENZ E-dir
        assert_abs_diff_eq!(rot1[(0, 1)], 1.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 1)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 1)], 0.0, epsilon = tol);

        // ECEF input Z - [0, 0, 1] - Expected output is ENZ N-dir
        assert_abs_diff_eq!(rot1[(0, 2)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 2)], 1.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 2)], 0.0, epsilon = tol);

        assert_abs_diff_eq!(rot1.determinant(), 1.0, epsilon = tol);

        // Test 90 degree longitude
        let x_sta = Vector3::new(90.0, 0.0, 0.0);
        let rot1 = rotation_ellipsoid_to_enz(&x_sta, true);

        // ECEF input X - [1, 0, 0] - Expected output is ENZ -E-dir
        assert_abs_diff_eq!(rot1[(0, 0)], -1.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 0)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 0)], 0.0, epsilon = tol);

        // ECEF input Y - [0, 1, 0] - Expected output is ENZ Z-dir
        assert_abs_diff_eq!(rot1[(0, 1)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 1)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 1)], 1.0, epsilon = tol);

        // ECEF input Z - [0, 0, 1] - Expected output is ENZ N-dir
        assert_abs_diff_eq!(rot1[(0, 2)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 2)], 1.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 2)], 0.0, epsilon = tol);

        assert_abs_diff_eq!(rot1.determinant(), 1.0, epsilon = tol);

        // Test 90 degree latitude
        let x_sta = Vector3::new(00.0, 90.0, 0.0);
        let rot1 = rotation_ellipsoid_to_enz(&x_sta, true);

        // ECEF input X - [1, 0, 0] - Expected output is ENZ -N-dir
        assert_abs_diff_eq!(rot1[(0, 0)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 0)], -1.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 0)], 0.0, epsilon = tol);

        // ECEF input Y - [0, 1, 0] - Expected output is ENZ E-dir
        assert_abs_diff_eq!(rot1[(0, 1)], 1.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 1)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 1)], 0.0, epsilon = tol);

        // ECEF input Z - [0, 0, 1] - Expected output is ENZ Z-dir
        assert_abs_diff_eq!(rot1[(0, 2)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 2)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 2)], 1.0, epsilon = tol);

        assert_abs_diff_eq!(rot1.determinant(), 1.0, epsilon = tol);
    }

    #[test]
    fn test_rotation_enz_to_ellipsoid() {
        let tol = f64::EPSILON;

        let x_sta = Vector3::new(42.1, 53.9, 100.0);
        let rot = rotation_ellipsoid_to_enz(&x_sta, true);
        let rot_t = rotation_enz_to_ellipsoid(&x_sta, true);

        let r = rot * rot_t;

        // Confirm identity
        assert_abs_diff_eq!(r[(0, 0)], 1.0, epsilon = tol);
        assert_abs_diff_eq!(r[(0, 1)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r[(0, 2)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r[(1, 0)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r[(1, 1)], 1.0, epsilon = tol);
        assert_abs_diff_eq!(r[(1, 2)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r[(2, 0)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r[(2, 1)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r[(2, 2)], 1.0, epsilon = tol);
    }

    #[test]
    fn test_relative_position_ecef_to_enz() {
        let tol = f64::EPSILON;

        // 100m Overhead
        let x_sta = Vector3::new(R_EARTH, 0.0, 0.0);
        let r_ecef = Vector3::new(R_EARTH + 100.0, 0.0, 0.0);

        let r_enz =
            relative_position_ecef_to_enz(&x_sta, &r_ecef, EllipsoidalConversionType::Geocentric);

        assert_abs_diff_eq!(r_enz[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r_enz[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r_enz[2], 100.0, epsilon = tol);

        // 100m North
        let x_sta = Vector3::new(R_EARTH, 0.0, 0.0);
        let r_ecef = Vector3::new(R_EARTH, 0.0, 100.0);

        let r_enz =
            relative_position_ecef_to_enz(&x_sta, &r_ecef, EllipsoidalConversionType::Geocentric);

        assert_abs_diff_eq!(r_enz[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r_enz[1], 100.0, epsilon = tol);
        assert_abs_diff_eq!(r_enz[2], 0.0, epsilon = tol);

        // 100m East
        let x_sta = Vector3::new(R_EARTH, 0.0, 0.0);
        let r_ecef = Vector3::new(R_EARTH, 100.0, 0.0);

        let r_enz =
            relative_position_ecef_to_enz(&x_sta, &r_ecef, EllipsoidalConversionType::Geocentric);

        assert_abs_diff_eq!(r_enz[0], 100.0, epsilon = tol);
        assert_abs_diff_eq!(r_enz[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r_enz[2], 0.0, epsilon = tol);

        // Confirm higher latitude and longitude is (+E, +N, -Z)
        let x_sta = Vector3::new(R_EARTH, 0.0, 0.0);
        let x_geoc = Vector3::new(0.5, 0.5, 0.0);
        let r_ecef = position_geocentric_to_ecef(&x_geoc, true).unwrap();

        let r_enz_geoc =
            relative_position_ecef_to_enz(&x_sta, &r_ecef, EllipsoidalConversionType::Geocentric);

        assert!(r_enz_geoc[0] > 0.0);
        assert!(r_enz_geoc[1] > 0.0);
        assert!(r_enz_geoc[2] < 0.0);

        // Confirm difference in geocentric and geodetic conversions
        let x_sta = Vector3::new(R_EARTH, 0.0, 0.0);
        let x_geod = Vector3::new(0.5, 0.5, 0.0);
        let r_ecef = position_geodetic_to_ecef(&x_geod, true).unwrap();

        let r_enz_geod =
            relative_position_ecef_to_enz(&x_sta, &r_ecef, EllipsoidalConversionType::Geodetic);

        assert!(r_enz_geod[0] > 0.0);
        assert!(r_enz_geod[1] > 0.0);
        assert!(r_enz_geod[2] < 0.0);

        for i in 0..3 {
            assert_ne!(r_enz_geoc[i], r_enz_geod[i]);
        }
    }

    #[test]
    fn test_relative_position_enz_to_ecef() {
        let tol = f64::EPSILON;

        let x_sta = Vector3::new(R_EARTH, 0.0, 0.0);
        let r_enz = Vector3::new(0.0, 0.0, 100.0);

        let r_ecef =
            relative_position_enz_to_ecef(&x_sta, &r_enz, EllipsoidalConversionType::Geodetic);

        assert_abs_diff_eq!(r_ecef[0], R_EARTH + 100.0, epsilon = tol);
        assert_abs_diff_eq!(r_ecef[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r_ecef[2], 0.0, epsilon = tol);
    }

    #[test]
    fn test_rotation_ellipsoid_to_sez() {
        // Epsilon Tolerance
        let tol = f64::EPSILON;

        // Test aligned coordinates
        let x_sta = Vector3::new(0.0, 0.0, 0.0);
        let rot1 = rotation_ellipsoid_to_sez(&x_sta, true);

        // ECEF input X - [1, 0, 0] - Expected output is SEZ Z-dir
        assert_abs_diff_eq!(rot1[(0, 0)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 0)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 0)], 1.0, epsilon = tol);

        // ECEF input Y - [0, 1, 0] - Expected output is SEZ E-dir
        assert_abs_diff_eq!(rot1[(0, 1)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 1)], 1.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 1)], 0.0, epsilon = tol);

        // ECEF input Z - [0, 0, 1] - Expected output is SEZ -S-dir
        assert_abs_diff_eq!(rot1[(0, 2)], -1.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 2)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 2)], 0.0, epsilon = tol);

        assert_abs_diff_eq!(rot1.determinant(), 1.0, epsilon = tol);

        // Test 90 degree longitude
        let x_sta = Vector3::new(90.0, 0.0, 0.0);
        let rot1 = rotation_ellipsoid_to_sez(&x_sta, true);

        // ECEF input X - [1, 0, 0] - Expected output is SEZ -E-dir
        assert_abs_diff_eq!(rot1[(0, 0)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 0)], -1.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 0)], 0.0, epsilon = tol);

        // ECEF input Y - [0, 1, 0] - Expected output is SEZ Z-dir
        assert_abs_diff_eq!(rot1[(0, 1)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 1)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 1)], 1.0, epsilon = tol);

        // ECEF input Z - [0, 0, 1] - Expected output is SEZ -S-dir
        assert_abs_diff_eq!(rot1[(0, 2)], -1.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 2)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 2)], 0.0, epsilon = tol);

        assert_abs_diff_eq!(rot1.determinant(), 1.0, epsilon = tol);

        // Test 90 degree latitude
        let x_sta = Vector3::new(00.0, 90.0, 0.0);
        let rot1 = rotation_ellipsoid_to_sez(&x_sta, true);

        // ECEF input X - [1, 0, 0] - Expected output is SEZ S-dir
        assert_abs_diff_eq!(rot1[(0, 0)], 1.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 0)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 0)], 0.0, epsilon = tol);

        // ECEF input Y - [0, 1, 0] - Expected output is SEZ E-dir
        assert_abs_diff_eq!(rot1[(0, 1)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 1)], 1.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 1)], 0.0, epsilon = tol);

        // ECEF input Z - [0, 0, 1] - Expected output is SEZ Z-dir
        assert_abs_diff_eq!(rot1[(0, 2)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(1, 2)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(rot1[(2, 2)], 1.0, epsilon = tol);

        assert_abs_diff_eq!(rot1.determinant(), 1.0, epsilon = tol);
    }

    #[test]
    fn test_rotation_sez_to_ellipsoid() {
        let tol = f64::EPSILON;

        let x_sta = Vector3::new(42.1, 53.9, 100.0);
        let rot = rotation_ellipsoid_to_sez(&x_sta, true);
        let rot_t = rotation_sez_to_ellipsoid(&x_sta, true);

        let r = rot * rot_t;

        // Confirm identity
        assert_abs_diff_eq!(r[(0, 0)], 1.0, epsilon = tol);
        assert_abs_diff_eq!(r[(0, 1)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r[(0, 2)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r[(1, 0)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r[(1, 1)], 1.0, epsilon = tol);
        assert_abs_diff_eq!(r[(1, 2)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r[(2, 0)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r[(2, 1)], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r[(2, 2)], 1.0, epsilon = tol);
    }

    #[test]
    fn test_relative_position_ecef_to_sez() {
        let tol = f64::EPSILON;

        // 100m Overhead
        let x_sta = Vector3::new(R_EARTH, 0.0, 0.0);
        let r_ecef = Vector3::new(R_EARTH + 100.0, 0.0, 0.0);

        let r_sez =
            relative_position_ecef_to_sez(x_sta, r_ecef, EllipsoidalConversionType::Geocentric);

        assert_abs_diff_eq!(r_sez[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r_sez[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r_sez[2], 100.0, epsilon = tol);

        // 100m North
        let x_sta = Vector3::new(R_EARTH, 0.0, 0.0);
        let r_ecef = Vector3::new(R_EARTH, 0.0, 100.0);

        let r_sez =
            relative_position_ecef_to_sez(x_sta, r_ecef, EllipsoidalConversionType::Geocentric);

        assert_abs_diff_eq!(r_sez[0], -100.0, epsilon = tol);
        assert_abs_diff_eq!(r_sez[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r_sez[2], 0.0, epsilon = tol);

        // 100m East
        let x_sta = Vector3::new(R_EARTH, 0.0, 0.0);
        let r_ecef = Vector3::new(R_EARTH, 100.0, 0.0);

        let r_sez =
            relative_position_ecef_to_sez(x_sta, r_ecef, EllipsoidalConversionType::Geocentric);

        assert_abs_diff_eq!(r_sez[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r_sez[1], 100.0, epsilon = tol);
        assert_abs_diff_eq!(r_sez[2], 0.0, epsilon = tol);

        // Confirm higher latitude and longitude is (+E, +N, -Z)
        let x_sta = Vector3::new(R_EARTH, 0.0, 0.0);
        let x_geoc = Vector3::new(0.5, 0.5, 0.0);
        let r_ecef = position_geocentric_to_ecef(&x_geoc, true).unwrap();

        let r_sez_geoc =
            relative_position_ecef_to_sez(x_sta, r_ecef, EllipsoidalConversionType::Geocentric);

        assert!(r_sez_geoc[0] < 0.0);
        assert!(r_sez_geoc[1] > 0.0);
        assert!(r_sez_geoc[2] < 0.0);

        // Confirm difference in geocentric and geodetic conversions
        let x_sta = Vector3::new(R_EARTH, 0.0, 0.0);
        let x_geod = Vector3::new(0.5, 0.5, 0.0);
        let r_ecef = position_geodetic_to_ecef(&x_geod, true).unwrap();

        let r_sez_geod =
            relative_position_ecef_to_sez(x_sta, r_ecef, EllipsoidalConversionType::Geodetic);

        assert!(r_sez_geod[0] < 0.0);
        assert!(r_sez_geod[1] > 0.0);
        assert!(r_sez_geod[2] < 0.0);

        for i in 0..3 {
            assert_ne!(r_sez_geoc[i], r_sez_geod[i]);
        }
    }

    #[test]
    fn test_relative_position_sez_to_ecef() {
        let tol = f64::EPSILON;

        let x_sta = Vector3::new(R_EARTH, 0.0, 0.0);
        let r_sez = Vector3::new(0.0, 0.0, 100.0);

        let r_ecef =
            relative_position_sez_to_ecef(&x_sta, &r_sez, EllipsoidalConversionType::Geodetic);

        assert_abs_diff_eq!(r_ecef[0], R_EARTH + 100.0, epsilon = tol);
        assert_abs_diff_eq!(r_ecef[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r_ecef[2], 0.0, epsilon = tol);
    }

    #[test]
    fn test_position_enz_to_azel() {
        let tol = f64::EPSILON;

        // Directly above
        let r_enz = Vector3::new(0.0, 0.0, 100.0);
        let x_azel = position_enz_to_azel(&r_enz, true);

        assert_abs_diff_eq!(x_azel[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[1], 90.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[2], 100.0, epsilon = tol);

        // North
        let r_enz = Vector3::new(0.0, 100.0, 0.0);
        let x_azel = position_enz_to_azel(&r_enz, true);

        assert_abs_diff_eq!(x_azel[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[2], 100.0, epsilon = tol);

        // East
        let r_enz = Vector3::new(100.0, 0.0, 0.0);
        let x_azel = position_enz_to_azel(&r_enz, true);

        assert_abs_diff_eq!(x_azel[0], 90.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[2], 100.0, epsilon = tol);

        // North-West
        let r_enz = Vector3::new(-100.0, 100.0, 0.0);
        let x_azel = position_enz_to_azel(&r_enz, true);

        assert_abs_diff_eq!(x_azel[0], 315.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[2], 100.0 * 2.0_f64.sqrt(), epsilon = tol);
    }

    #[test]
    fn test_position_sez_to_azel() {
        let tol = f64::EPSILON;

        // Directly above
        let r_sez = Vector3::new(0.0, 0.0, 100.0);
        let x_azel = position_sez_to_azel(&r_sez, true);

        assert_abs_diff_eq!(x_azel[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[1], 90.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[2], 100.0, epsilon = tol);

        // North
        let r_sez = Vector3::new(-100.0, 0.0, 0.0);
        let x_azel = position_sez_to_azel(&r_sez, true);

        assert_abs_diff_eq!(x_azel[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[2], 100.0, epsilon = tol);

        // East
        let r_sez = Vector3::new(0.0, 100.0, 0.0);
        let x_azel = position_sez_to_azel(&r_sez, true);

        assert_abs_diff_eq!(x_azel[0], 90.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[2], 100.0, epsilon = tol);

        // North-West
        let r_sez = Vector3::new(-100.0, -100.0, 0.0);
        let x_azel = position_sez_to_azel(&r_sez, true);

        assert_abs_diff_eq!(x_azel[0], 315.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[2], 100.0 * 2.0_f64.sqrt(), epsilon = tol);
    }
}
