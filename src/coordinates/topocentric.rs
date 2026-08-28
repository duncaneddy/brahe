/*!
 * Provides topocentric coordiante transformations.
 */

use std::f64::consts::PI;

use nalgebra::Vector3;

use crate::math::SMatrix3;

use crate::constants;
use crate::constants::AngleFormat;
use crate::coordinates::coordinate_types::EllipsoidalConversionType;
use crate::coordinates::geocentric::position_ecef_to_geocentric;
use crate::coordinates::geodetic::position_ecef_to_geodetic;
use crate::utils::BraheError;
use crate::utils::batch::{batch_map, batch_zip};

/// Compute the rotation matrix from body-fixed to East-North-Zenith (ENZ)
/// Cartesian coordinates for a given set of coordinates on an ellipsoidal body.
/// The ellipsoidal coordinates can either be geodetic or geocentric.
///
/// # Args:
/// - `x_ellipsoid`: Ellipsoidal coordinates.  Expected format (lon, lat, alt)
/// - `angle_format`: Format for angular coordinates (Radians or Degrees)
///
/// # Returns:
/// - `E`: Earth-fixed to Topocentric rotation matrix
///
/// # Examples:
/// ```
/// use brahe::constants::DEGREES;
/// use brahe::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_geo = vector3_from_array([30.0, 60.0, 0.0]);
/// let rot = rotation_ellipsoid_to_enz(x_geo, DEGREES);
/// ```
pub fn rotation_ellipsoid_to_enz(x_ellipsoid: Vector3<f64>, angle_format: AngleFormat) -> SMatrix3 {
    let lon = match angle_format {
        AngleFormat::Degrees => x_ellipsoid[0] * constants::DEG2RAD,
        AngleFormat::Radians => x_ellipsoid[0],
    };
    let lat = match angle_format {
        AngleFormat::Degrees => x_ellipsoid[1] * constants::DEG2RAD,
        AngleFormat::Radians => x_ellipsoid[1],
    };

    // Construct Rotation matrix
    SMatrix3::new(
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
/// - `angle_format`: Format for angular coordinates (Radians or Degrees)
///
/// # Returns:
/// - `E`: Topocentric to Earth-fixed rotation matrix
///
/// # Examples:
/// ```
/// use brahe::constants::DEGREES;
/// use brahe::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_geo = vector3_from_array([30.0, 60.0, 0.0]);
/// let rot = rotation_enz_to_ellipsoid(x_geo, DEGREES);
/// ```
pub fn rotation_enz_to_ellipsoid(x_ellipsoid: Vector3<f64>, angle_format: AngleFormat) -> SMatrix3 {
    rotation_ellipsoid_to_enz(x_ellipsoid, angle_format).transpose()
}

/// Rotation matrix from ECEF axes to the ENZ frame of `location_ecef`.
///
/// # Arguments
/// - `location_ecef`: Cartesian ECEF site position. Units: (*m*)
/// - `conversion_type`: Ellipsoidal conversion used to orient the local frame
///
/// # Returns
/// - Rotation matrix transforming ECEF -> ENZ at the site
///
/// # Examples
///
/// ```ignore
/// use brahe::constants::R_EARTH;
/// use brahe::coordinates::EllipsoidalConversionType;
/// use nalgebra::Vector3;
///
/// let e = enz_rotation_at(Vector3::new(R_EARTH, 0.0, 0.0), EllipsoidalConversionType::Geodetic);
/// ```
fn enz_rotation_at(
    location_ecef: Vector3<f64>,
    conversion_type: EllipsoidalConversionType,
) -> SMatrix3 {
    match conversion_type {
        EllipsoidalConversionType::Geocentric => rotation_ellipsoid_to_enz(
            position_ecef_to_geocentric(location_ecef, AngleFormat::Radians),
            AngleFormat::Radians,
        ),
        EllipsoidalConversionType::Geodetic => rotation_ellipsoid_to_enz(
            position_ecef_to_geodetic(location_ecef, AngleFormat::Radians),
            AngleFormat::Radians,
        ),
    }
}

/// Rotation matrix from the ENZ frame of `location_ecef` to ECEF axes.
///
/// # Arguments
/// - `location_ecef`: Cartesian ECEF site position. Units: (*m*)
/// - `conversion_type`: Ellipsoidal conversion used to orient the local frame
///
/// # Returns
/// - Rotation matrix transforming ENZ -> ECEF at the site
///
/// # Examples
///
/// ```ignore
/// use brahe::constants::R_EARTH;
/// use brahe::coordinates::EllipsoidalConversionType;
/// use nalgebra::Vector3;
///
/// let et = enz_inverse_rotation_at(Vector3::new(R_EARTH, 0.0, 0.0), EllipsoidalConversionType::Geodetic);
/// ```
fn enz_inverse_rotation_at(
    location_ecef: Vector3<f64>,
    conversion_type: EllipsoidalConversionType,
) -> SMatrix3 {
    match conversion_type {
        EllipsoidalConversionType::Geocentric => rotation_enz_to_ellipsoid(
            position_ecef_to_geocentric(location_ecef, AngleFormat::Radians),
            AngleFormat::Radians,
        ),
        EllipsoidalConversionType::Geodetic => rotation_enz_to_ellipsoid(
            position_ecef_to_geodetic(location_ecef, AngleFormat::Radians),
            AngleFormat::Radians,
        ),
    }
}

/// Rotation matrix from ECEF axes to the SEZ frame of `location_ecef`.
///
/// # Arguments
/// - `location_ecef`: Cartesian ECEF site position. Units: (*m*)
/// - `conversion_type`: Ellipsoidal conversion used to orient the local frame
///
/// # Returns
/// - Rotation matrix transforming ECEF -> SEZ at the site
///
/// # Examples
///
/// ```ignore
/// use brahe::constants::R_EARTH;
/// use brahe::coordinates::EllipsoidalConversionType;
/// use nalgebra::Vector3;
///
/// let e = sez_rotation_at(Vector3::new(R_EARTH, 0.0, 0.0), EllipsoidalConversionType::Geodetic);
/// ```
fn sez_rotation_at(
    location_ecef: Vector3<f64>,
    conversion_type: EllipsoidalConversionType,
) -> SMatrix3 {
    match conversion_type {
        EllipsoidalConversionType::Geocentric => rotation_ellipsoid_to_sez(
            position_ecef_to_geocentric(location_ecef, AngleFormat::Radians),
            AngleFormat::Radians,
        ),
        EllipsoidalConversionType::Geodetic => rotation_ellipsoid_to_sez(
            position_ecef_to_geodetic(location_ecef, AngleFormat::Radians),
            AngleFormat::Radians,
        ),
    }
}

/// Rotation matrix from the SEZ frame of `location_ecef` to ECEF axes.
///
/// # Arguments
/// - `location_ecef`: Cartesian ECEF site position. Units: (*m*)
/// - `conversion_type`: Ellipsoidal conversion used to orient the local frame
///
/// # Returns
/// - Rotation matrix transforming SEZ -> ECEF at the site
///
/// # Examples
///
/// ```ignore
/// use brahe::constants::R_EARTH;
/// use brahe::coordinates::EllipsoidalConversionType;
/// use nalgebra::Vector3;
///
/// let et = sez_inverse_rotation_at(Vector3::new(R_EARTH, 0.0, 0.0), EllipsoidalConversionType::Geodetic);
/// ```
fn sez_inverse_rotation_at(
    location_ecef: Vector3<f64>,
    conversion_type: EllipsoidalConversionType,
) -> SMatrix3 {
    match conversion_type {
        EllipsoidalConversionType::Geocentric => rotation_sez_to_ellipsoid(
            position_ecef_to_geocentric(location_ecef, AngleFormat::Radians),
            AngleFormat::Radians,
        ),
        EllipsoidalConversionType::Geodetic => rotation_sez_to_ellipsoid(
            position_ecef_to_geodetic(location_ecef, AngleFormat::Radians),
            AngleFormat::Radians,
        ),
    }
}

/// Express `r_ecef` relative to `location_ecef` in the local frame given by
/// `rot` (ECEF -> local).
///
/// # Arguments
/// - `rot`: Rotation matrix transforming ECEF -> local (ENZ or SEZ)
/// - `location_ecef`: Cartesian ECEF site position. Units: (*m*)
/// - `r_ecef`: Cartesian ECEF position to express relative to the site. Units: (*m*)
///
/// # Returns
/// - Relative position in the local frame. Units: (*m*)
///
/// # Examples
///
/// ```ignore
/// use brahe::constants::R_EARTH;
/// use brahe::coordinates::EllipsoidalConversionType;
/// use nalgebra::Vector3;
///
/// let site = Vector3::new(R_EARTH, 0.0, 0.0);
/// let rot = enz_rotation_at(site, EllipsoidalConversionType::Geodetic);
/// let enz = apply_relative_ecef_to_local(&rot, &site, &Vector3::new(R_EARTH + 500e3, 0.0, 0.0));
/// ```
fn apply_relative_ecef_to_local(
    rot: &SMatrix3,
    location_ecef: &Vector3<f64>,
    r_ecef: &Vector3<f64>,
) -> Vector3<f64> {
    let r = r_ecef - location_ecef;
    rot * r
}

/// Express a local-frame relative position as an ECEF position, given `rot`
/// (local -> ECEF).
///
/// # Arguments
/// - `rot`: Rotation matrix transforming local (ENZ or SEZ) -> ECEF
/// - `location_ecef`: Cartesian ECEF site position. Units: (*m*)
/// - `r_local`: Relative position in the local frame. Units: (*m*)
///
/// # Returns
/// - Cartesian ECEF position. Units: (*m*)
///
/// # Examples
///
/// ```ignore
/// use brahe::constants::R_EARTH;
/// use brahe::coordinates::EllipsoidalConversionType;
/// use nalgebra::Vector3;
///
/// let site = Vector3::new(R_EARTH, 0.0, 0.0);
/// let rot = enz_inverse_rotation_at(site, EllipsoidalConversionType::Geodetic);
/// let ecef = apply_relative_local_to_ecef(&rot, &site, &Vector3::new(0.0, 0.0, 500e3));
/// ```
fn apply_relative_local_to_ecef(
    rot: &SMatrix3,
    location_ecef: &Vector3<f64>,
    r_local: &Vector3<f64>,
) -> Vector3<f64> {
    let r = *r_local;
    location_ecef + rot * r
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
/// use brahe::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_station = vector3_from_array([R_EARTH, 0.0, 0.0]);
/// let x_sat = vector3_from_array([R_EARTH + 500.0e3, 0.0, 0.0]);
///
/// let r_enz = relative_position_ecef_to_enz(
///     x_station, x_sat, EllipsoidalConversionType::Geocentric
/// );
/// ```
pub fn relative_position_ecef_to_enz(
    location_ecef: Vector3<f64>,
    r_ecef: Vector3<f64>,
    conversion_type: EllipsoidalConversionType,
) -> Vector3<f64> {
    apply_relative_ecef_to_local(
        &enz_rotation_at(location_ecef, conversion_type),
        &location_ecef,
        &r_ecef,
    )
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
/// use brahe::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_station = vector3_from_array([R_EARTH, 0.0, 0.0]);
/// let r_enz = vector3_from_array([0.0, 0.0, 500.0e3]);
///
/// let r_ecef = relative_position_enz_to_ecef(
///     x_station, r_enz, EllipsoidalConversionType::Geocentric
/// );
/// ```
pub fn relative_position_enz_to_ecef(
    location_ecef: Vector3<f64>,
    r_enz: Vector3<f64>,
    conversion_type: EllipsoidalConversionType,
) -> Vector3<f64> {
    apply_relative_local_to_ecef(
        &enz_inverse_rotation_at(location_ecef, conversion_type),
        &location_ecef,
        &r_enz,
    )
}

/// Compute the rotation matrix from body-fixed to South-East-Zenith (SEZ)
/// Cartesian coordinates for a given set of coordinates on an ellipsoidal body.
/// The ellipsoidal coordinates can either be geodetic or geocentric.
///
/// # Args:
/// - `x_ellipsoid`: Ellipsoidal coordinates.  Expected format (lon, lat, alt)
/// - `angle_format`: Format for angular coordinates (Radians or Degrees)
///
/// # Returns:
/// - `E`: Earth-fixed to Topocentric rotation matrix
///
/// # Examples:
/// ```
/// use brahe::constants::DEGREES;
/// use brahe::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_geo = vector3_from_array([30.0, 60.0, 0.0]);
/// let rot = rotation_sez_to_ellipsoid(x_geo, DEGREES);
/// ```
pub fn rotation_ellipsoid_to_sez(x_ellipsoid: Vector3<f64>, angle_format: AngleFormat) -> SMatrix3 {
    let lon = match angle_format {
        AngleFormat::Degrees => x_ellipsoid[0] * constants::DEG2RAD,
        AngleFormat::Radians => x_ellipsoid[0],
    };
    let lat = match angle_format {
        AngleFormat::Degrees => x_ellipsoid[1] * constants::DEG2RAD,
        AngleFormat::Radians => x_ellipsoid[1],
    };

    // Construct Rotation matrix
    SMatrix3::new(
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
/// - `angle_format`: Format for angular coordinates (Radians or Degrees)
///
/// # Returns:
/// - `E`: Topocentric to Earth-fixed rotation matrix
///
/// # Examples:
/// ```
/// use brahe::constants::DEGREES;
/// use brahe::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_geo = vector3_from_array([30.0, 60.0, 0.0]);
/// let rot = rotation_sez_to_ellipsoid(x_geo, DEGREES);
/// ```
pub fn rotation_sez_to_ellipsoid(x_ellipsoid: Vector3<f64>, angle_format: AngleFormat) -> SMatrix3 {
    rotation_ellipsoid_to_sez(x_ellipsoid, angle_format).transpose()
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
/// use brahe::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_station = vector3_from_array([R_EARTH, 0.0, 0.0]);
/// let x_sat = vector3_from_array([R_EARTH + 500.0e3, 0.0, 0.0]);
///
/// let r_enz = relative_position_ecef_to_enz(
///     x_station, x_sat, EllipsoidalConversionType::Geocentric
/// );
/// ```
pub fn relative_position_ecef_to_sez(
    location_ecef: Vector3<f64>,
    r_ecef: Vector3<f64>,
    conversion_type: EllipsoidalConversionType,
) -> Vector3<f64> {
    apply_relative_ecef_to_local(
        &sez_rotation_at(location_ecef, conversion_type),
        &location_ecef,
        &r_ecef,
    )
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
/// use brahe::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_station = vector3_from_array([R_EARTH, 0.0, 0.0]);
/// let r_sez = vector3_from_array([0.0, 0.0, 500.0e3]);
///
/// let r_ecef = relative_position_sez_to_ecef(
///     x_station, r_sez, EllipsoidalConversionType::Geocentric
/// );
/// ```
pub fn relative_position_sez_to_ecef(
    location_ecef: Vector3<f64>,
    x_sez: Vector3<f64>,
    conversion_type: EllipsoidalConversionType,
) -> Vector3<f64> {
    apply_relative_local_to_ecef(
        &sez_inverse_rotation_at(location_ecef, conversion_type),
        &location_ecef,
        &x_sez,
    )
}

/// Converts East-North-Zenith topocentric coordinates of an location
/// into azimuth, elevation, and range from that same location. Azimuth is measured
/// clockwise from North.
///
/// # Args:
/// - `x_enz`: Relative Cartesian position of object to location East-North-Up coordinates. Units: (*m*)
/// - `angle_format`: Format for angular output (Radians or Degrees)
///
/// # Returns:
/// - `x_azel`: Azimuth, elevation and range. Units: (*angle*, *angle*, *m*)
///
/// # Examples:
/// ```
/// use brahe::constants::{R_EARTH, DEGREES};
/// use brahe::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_enz = vector3_from_array([100.0, 0.0, 0.0]);
///
/// let x_azel = position_enz_to_azel(x_enz, DEGREES);
/// // x_azel = [90.0, 0.0, 100.0]
/// ```
pub fn position_enz_to_azel(x_enz: Vector3<f64>, angle_format: AngleFormat) -> Vector3<f64> {
    // Range
    let rho = x_enz.norm();

    // Elevation
    let el = x_enz[2].atan2((x_enz[0].powi(2) + x_enz[1].powi(2)).sqrt());

    // Azimuth
    let az = if el != PI / 2.0 {
        let azt = x_enz[0].atan2(x_enz[1]);

        if azt >= 0.0 { azt } else { azt + 2.0 * PI }
    } else {
        // If at peak elevation azimuth is ambiguous so define as 0.0
        0.0
    };

    match angle_format {
        AngleFormat::Degrees => Vector3::new(az * constants::RAD2DEG, el * constants::RAD2DEG, rho),
        AngleFormat::Radians => Vector3::new(az, el, rho),
    }
}

/// Converts South-East-Zenith topocentric coordinates of an location
/// into azimuth, elevation, and range from that same location. Azimuth is measured
/// clockwise from North.
///
/// # Args:
/// - `x_sez`: Relative Cartesian position of object to location South-East-Zenith coordinates. Units: (*m*)
/// - `angle_format`: Format for angular output (Radians or Degrees)
///
/// # Returns:
/// - `x_azel`: Azimuth, elevation and range. Units: (*angle*, *angle*, *m*)
///
/// # Examples:
/// ```
/// use brahe::constants::{R_EARTH, DEGREES};
/// use brahe::vector3_from_array;
/// use brahe::coordinates::*;
///
/// let x_enz = vector3_from_array([0.0, 100.0, 0.0]);
///
/// let x_azel = position_sez_to_azel(x_enz, DEGREES);
/// // x_azel = [90.0, 0.0, 100.0]
/// ```
pub fn position_sez_to_azel(x_sez: Vector3<f64>, angle_format: AngleFormat) -> Vector3<f64> {
    // Range
    let rho = x_sez.norm();

    // Elevation
    let el = x_sez[2].atan2((x_sez[0].powi(2) + x_sez[1].powi(2)).sqrt());

    // Azimuth
    let az = if el != PI / 2.0 {
        let azt = (x_sez[1]).atan2(-x_sez[0]);

        if azt >= 0.0 { azt } else { azt + 2.0 * PI }
    } else {
        // If at peak elevation azimuth is ambiguous so define as 0.0
        0.0
    };

    match angle_format {
        AngleFormat::Degrees => Vector3::new(az * constants::RAD2DEG, el * constants::RAD2DEG, rho),
        AngleFormat::Radians => Vector3::new(az, el, rho),
    }
}

/// Computes the ellipsoidal-to-ENZ rotation matrix for each site in `x_ellipsoid`.
///
/// Batch form of [`rotation_ellipsoid_to_enz`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_ellipsoid`: Ellipsoidal (geodetic or geocentric) site positions `[lon, lat, alt]`. Units: (angles per `angle_format`, *m*)
/// - `angle_format`: Format of the angular coordinates
///
/// # Returns
/// - Rotation matrices transforming ellipsoidal -> ENZ, one per site, in input order
///
/// # Examples
/// ```
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::rotations_ellipsoid_to_enz;
/// use nalgebra::Vector3;
///
/// let sites = vec![Vector3::new(-122.4, 37.8, 0.0), Vector3::new(151.2, -33.9, 0.0)];
/// let r = rotations_ellipsoid_to_enz(&sites, AngleFormat::Degrees);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_ellipsoid_to_enz(
    x_ellipsoid: &[Vector3<f64>],
    angle_format: AngleFormat,
) -> Vec<SMatrix3> {
    batch_map(|x| rotation_ellipsoid_to_enz(*x, angle_format), x_ellipsoid)
}

/// Computes the ENZ-to-ellipsoidal rotation matrix for each site in `x_ellipsoid`.
///
/// Batch form of [`rotation_enz_to_ellipsoid`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_ellipsoid`: Ellipsoidal (geodetic or geocentric) site positions `[lon, lat, alt]`. Units: (angles per `angle_format`, *m*)
/// - `angle_format`: Format of the angular coordinates
///
/// # Returns
/// - Rotation matrices transforming ENZ -> ellipsoidal, one per site, in input order
///
/// # Examples
/// ```
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::rotations_enz_to_ellipsoid;
/// use nalgebra::Vector3;
///
/// let sites = vec![Vector3::new(-122.4, 37.8, 0.0), Vector3::new(151.2, -33.9, 0.0)];
/// let r = rotations_enz_to_ellipsoid(&sites, AngleFormat::Degrees);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_enz_to_ellipsoid(
    x_ellipsoid: &[Vector3<f64>],
    angle_format: AngleFormat,
) -> Vec<SMatrix3> {
    batch_map(|x| rotation_enz_to_ellipsoid(*x, angle_format), x_ellipsoid)
}

/// Computes the ellipsoidal-to-SEZ rotation matrix for each site in `x_ellipsoid`.
///
/// Batch form of [`rotation_ellipsoid_to_sez`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_ellipsoid`: Ellipsoidal (geodetic or geocentric) site positions `[lon, lat, alt]`. Units: (angles per `angle_format`, *m*)
/// - `angle_format`: Format of the angular coordinates
///
/// # Returns
/// - Rotation matrices transforming ellipsoidal -> SEZ, one per site, in input order
///
/// # Examples
/// ```
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::rotations_ellipsoid_to_sez;
/// use nalgebra::Vector3;
///
/// let sites = vec![Vector3::new(-122.4, 37.8, 0.0), Vector3::new(151.2, -33.9, 0.0)];
/// let r = rotations_ellipsoid_to_sez(&sites, AngleFormat::Degrees);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_ellipsoid_to_sez(
    x_ellipsoid: &[Vector3<f64>],
    angle_format: AngleFormat,
) -> Vec<SMatrix3> {
    batch_map(|x| rotation_ellipsoid_to_sez(*x, angle_format), x_ellipsoid)
}

/// Computes the SEZ-to-ellipsoidal rotation matrix for each site in `x_ellipsoid`.
///
/// Batch form of [`rotation_sez_to_ellipsoid`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_ellipsoid`: Ellipsoidal (geodetic or geocentric) site positions `[lon, lat, alt]`. Units: (angles per `angle_format`, *m*)
/// - `angle_format`: Format of the angular coordinates
///
/// # Returns
/// - Rotation matrices transforming SEZ -> ellipsoidal, one per site, in input order
///
/// # Examples
/// ```
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::rotations_sez_to_ellipsoid;
/// use nalgebra::Vector3;
///
/// let sites = vec![Vector3::new(-122.4, 37.8, 0.0), Vector3::new(151.2, -33.9, 0.0)];
/// let r = rotations_sez_to_ellipsoid(&sites, AngleFormat::Degrees);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_sez_to_ellipsoid(
    x_ellipsoid: &[Vector3<f64>],
    angle_format: AngleFormat,
) -> Vec<SMatrix3> {
    batch_map(|x| rotation_sez_to_ellipsoid(*x, angle_format), x_ellipsoid)
}

/// Transforms a batch of positions between ECEF and the local ENZ frame of one or more sites.
///
/// Batch form of [`relative_position_ecef_to_enz`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// `location_ecef` and the relative-position argument follow the broadcast
/// rule: each has length 1 or the common batch length. A single location
/// computes its rotation matrix once and applies it to every position.
///
/// # Arguments
/// - `location_ecef`: Cartesian ECEF site positions, length 1 or the batch length. Units: (*m*)
/// - `r_ecef`: Cartesian ECEF positions, length 1 or the batch length. Units: (*m*)
/// - `conversion_type`: Ellipsoidal conversion used to orient the local frame
///
/// # Returns
/// - Relative positions in the local ENZ frame in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples
/// ```
/// use brahe::constants::{AngleFormat, R_EARTH};
/// use brahe::coordinates::{EllipsoidalConversionType, position_geodetic_to_ecef, relative_positions_ecef_to_enz};
/// use nalgebra::Vector3;
///
/// let station = position_geodetic_to_ecef(Vector3::new(-122.4, 37.8, 0.0), AngleFormat::Degrees).unwrap();
/// let targets = vec![Vector3::new(R_EARTH + 500e3, 0.0, 0.0), Vector3::new(0.0, R_EARTH + 500e3, 0.0)];
/// let out = relative_positions_ecef_to_enz(&[station], &targets, EllipsoidalConversionType::Geodetic).unwrap();
/// assert_eq!(out.len(), 2);
/// ```
pub fn relative_positions_ecef_to_enz(
    location_ecef: &[Vector3<f64>],
    r_ecef: &[Vector3<f64>],
    conversion_type: EllipsoidalConversionType,
) -> Result<Vec<Vector3<f64>>, BraheError> {
    if location_ecef.len() == 1 {
        let rot = enz_rotation_at(location_ecef[0], conversion_type);
        return Ok(batch_map(
            |x| apply_relative_ecef_to_local(&rot, &location_ecef[0], x),
            r_ecef,
        ));
    }
    batch_zip(
        |loc, x| relative_position_ecef_to_enz(*loc, *x, conversion_type),
        location_ecef,
        r_ecef,
    )
}

/// Transforms a batch of positions between ECEF and the local ENZ frame of one or more sites.
///
/// Batch form of [`relative_position_enz_to_ecef`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// `location_ecef` and the relative-position argument follow the broadcast
/// rule: each has length 1 or the common batch length. A single location
/// computes its rotation matrix once and applies it to every position.
///
/// # Arguments
/// - `location_ecef`: Cartesian ECEF site positions, length 1 or the batch length. Units: (*m*)
/// - `r_enz`: Relative positions in the local ENZ frame, length 1 or the batch length. Units: (*m*)
/// - `conversion_type`: Ellipsoidal conversion used to orient the local frame
///
/// # Returns
/// - Cartesian ECEF positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples
/// ```
/// use brahe::constants::{AngleFormat, R_EARTH};
/// use brahe::coordinates::{EllipsoidalConversionType, position_geodetic_to_ecef, relative_positions_enz_to_ecef};
/// use nalgebra::Vector3;
///
/// let station = position_geodetic_to_ecef(Vector3::new(-122.4, 37.8, 0.0), AngleFormat::Degrees).unwrap();
/// let targets = vec![Vector3::new(R_EARTH + 500e3, 0.0, 0.0), Vector3::new(0.0, R_EARTH + 500e3, 0.0)];
/// let out = relative_positions_enz_to_ecef(&[station], &targets, EllipsoidalConversionType::Geodetic).unwrap();
/// assert_eq!(out.len(), 2);
/// ```
pub fn relative_positions_enz_to_ecef(
    location_ecef: &[Vector3<f64>],
    r_enz: &[Vector3<f64>],
    conversion_type: EllipsoidalConversionType,
) -> Result<Vec<Vector3<f64>>, BraheError> {
    if location_ecef.len() == 1 {
        let rot = enz_inverse_rotation_at(location_ecef[0], conversion_type);
        return Ok(batch_map(
            |x| apply_relative_local_to_ecef(&rot, &location_ecef[0], x),
            r_enz,
        ));
    }
    batch_zip(
        |loc, x| relative_position_enz_to_ecef(*loc, *x, conversion_type),
        location_ecef,
        r_enz,
    )
}

/// Transforms a batch of positions between ECEF and the local SEZ frame of one or more sites.
///
/// Batch form of [`relative_position_ecef_to_sez`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// `location_ecef` and the relative-position argument follow the broadcast
/// rule: each has length 1 or the common batch length. A single location
/// computes its rotation matrix once and applies it to every position.
///
/// # Arguments
/// - `location_ecef`: Cartesian ECEF site positions, length 1 or the batch length. Units: (*m*)
/// - `r_ecef`: Cartesian ECEF positions, length 1 or the batch length. Units: (*m*)
/// - `conversion_type`: Ellipsoidal conversion used to orient the local frame
///
/// # Returns
/// - Relative positions in the local SEZ frame in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples
/// ```
/// use brahe::constants::{AngleFormat, R_EARTH};
/// use brahe::coordinates::{EllipsoidalConversionType, position_geodetic_to_ecef, relative_positions_ecef_to_sez};
/// use nalgebra::Vector3;
///
/// let station = position_geodetic_to_ecef(Vector3::new(-122.4, 37.8, 0.0), AngleFormat::Degrees).unwrap();
/// let targets = vec![Vector3::new(R_EARTH + 500e3, 0.0, 0.0), Vector3::new(0.0, R_EARTH + 500e3, 0.0)];
/// let out = relative_positions_ecef_to_sez(&[station], &targets, EllipsoidalConversionType::Geodetic).unwrap();
/// assert_eq!(out.len(), 2);
/// ```
pub fn relative_positions_ecef_to_sez(
    location_ecef: &[Vector3<f64>],
    r_ecef: &[Vector3<f64>],
    conversion_type: EllipsoidalConversionType,
) -> Result<Vec<Vector3<f64>>, BraheError> {
    if location_ecef.len() == 1 {
        let rot = sez_rotation_at(location_ecef[0], conversion_type);
        return Ok(batch_map(
            |x| apply_relative_ecef_to_local(&rot, &location_ecef[0], x),
            r_ecef,
        ));
    }
    batch_zip(
        |loc, x| relative_position_ecef_to_sez(*loc, *x, conversion_type),
        location_ecef,
        r_ecef,
    )
}

/// Transforms a batch of positions between ECEF and the local SEZ frame of one or more sites.
///
/// Batch form of [`relative_position_sez_to_ecef`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// `location_ecef` and the relative-position argument follow the broadcast
/// rule: each has length 1 or the common batch length. A single location
/// computes its rotation matrix once and applies it to every position.
///
/// # Arguments
/// - `location_ecef`: Cartesian ECEF site positions, length 1 or the batch length. Units: (*m*)
/// - `x_sez`: Relative positions in the local SEZ frame, length 1 or the batch length. Units: (*m*)
/// - `conversion_type`: Ellipsoidal conversion used to orient the local frame
///
/// # Returns
/// - Cartesian ECEF positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples
/// ```
/// use brahe::constants::{AngleFormat, R_EARTH};
/// use brahe::coordinates::{EllipsoidalConversionType, position_geodetic_to_ecef, relative_positions_sez_to_ecef};
/// use nalgebra::Vector3;
///
/// let station = position_geodetic_to_ecef(Vector3::new(-122.4, 37.8, 0.0), AngleFormat::Degrees).unwrap();
/// let targets = vec![Vector3::new(R_EARTH + 500e3, 0.0, 0.0), Vector3::new(0.0, R_EARTH + 500e3, 0.0)];
/// let out = relative_positions_sez_to_ecef(&[station], &targets, EllipsoidalConversionType::Geodetic).unwrap();
/// assert_eq!(out.len(), 2);
/// ```
pub fn relative_positions_sez_to_ecef(
    location_ecef: &[Vector3<f64>],
    x_sez: &[Vector3<f64>],
    conversion_type: EllipsoidalConversionType,
) -> Result<Vec<Vector3<f64>>, BraheError> {
    if location_ecef.len() == 1 {
        let rot = sez_inverse_rotation_at(location_ecef[0], conversion_type);
        return Ok(batch_map(
            |x| apply_relative_local_to_ecef(&rot, &location_ecef[0], x),
            x_sez,
        ));
    }
    batch_zip(
        |loc, x| relative_position_sez_to_ecef(*loc, *x, conversion_type),
        location_ecef,
        x_sez,
    )
}

/// Converts a batch of ENZ relative positions to azimuth/elevation/range.
///
/// Batch form of [`position_enz_to_azel`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_enz`: Relative positions in the ENZ frame. Units: (*m*)
/// - `angle_format`: Format of the returned angles
///
/// # Returns
/// - `[az, el, range]` in input order. Units: (angles per `angle_format`, *m*)
///
/// # Examples
/// ```
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::positions_enz_to_azel;
/// use nalgebra::Vector3;
///
/// let rel = vec![Vector3::new(1.0e3, 2.0e3, 3.0e3), Vector3::new(-1.0e3, 0.5e3, 2.0e3)];
/// let azel = positions_enz_to_azel(&rel, AngleFormat::Degrees);
/// assert_eq!(azel.len(), 2);
/// ```
pub fn positions_enz_to_azel(
    x_enz: &[Vector3<f64>],
    angle_format: AngleFormat,
) -> Vec<Vector3<f64>> {
    batch_map(|x| position_enz_to_azel(*x, angle_format), x_enz)
}

/// Converts a batch of SEZ relative positions to azimuth/elevation/range.
///
/// Batch form of [`position_sez_to_azel`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `x_sez`: Relative positions in the SEZ frame. Units: (*m*)
/// - `angle_format`: Format of the returned angles
///
/// # Returns
/// - `[az, el, range]` in input order. Units: (angles per `angle_format`, *m*)
///
/// # Examples
/// ```
/// use brahe::constants::AngleFormat;
/// use brahe::coordinates::positions_sez_to_azel;
/// use nalgebra::Vector3;
///
/// let rel = vec![Vector3::new(1.0e3, 2.0e3, 3.0e3), Vector3::new(-1.0e3, 0.5e3, 2.0e3)];
/// let azel = positions_sez_to_azel(&rel, AngleFormat::Degrees);
/// assert_eq!(azel.len(), 2);
/// ```
pub fn positions_sez_to_azel(
    x_sez: &[Vector3<f64>],
    angle_format: AngleFormat,
) -> Vec<Vector3<f64>> {
    batch_map(|x| position_sez_to_azel(*x, angle_format), x_sez)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    use crate::constants::DEGREES;
    use crate::{R_EARTH, position_geocentric_to_ecef, position_geodetic_to_ecef};

    use super::*;

    #[test]
    fn test_rotation_ellipsoid_to_enz() {
        // Epsilon Tolerance
        let tol = f64::EPSILON;

        // Test aligned coordinates
        let x_sta = Vector3::new(0.0, 0.0, 0.0);
        let rot1 = rotation_ellipsoid_to_enz(x_sta, DEGREES);

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
        let rot1 = rotation_ellipsoid_to_enz(x_sta, DEGREES);

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
        let rot1 = rotation_ellipsoid_to_enz(x_sta, DEGREES);

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
        let rot = rotation_ellipsoid_to_enz(x_sta, DEGREES);
        let rot_t = rotation_enz_to_ellipsoid(x_sta, DEGREES);

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
            relative_position_ecef_to_enz(x_sta, r_ecef, EllipsoidalConversionType::Geocentric);

        assert_abs_diff_eq!(r_enz[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r_enz[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r_enz[2], 100.0, epsilon = tol);

        // 100m North
        let x_sta = Vector3::new(R_EARTH, 0.0, 0.0);
        let r_ecef = Vector3::new(R_EARTH, 0.0, 100.0);

        let r_enz =
            relative_position_ecef_to_enz(x_sta, r_ecef, EllipsoidalConversionType::Geocentric);

        assert_abs_diff_eq!(r_enz[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r_enz[1], 100.0, epsilon = tol);
        assert_abs_diff_eq!(r_enz[2], 0.0, epsilon = tol);

        // 100m East
        let x_sta = Vector3::new(R_EARTH, 0.0, 0.0);
        let r_ecef = Vector3::new(R_EARTH, 100.0, 0.0);

        let r_enz =
            relative_position_ecef_to_enz(x_sta, r_ecef, EllipsoidalConversionType::Geocentric);

        assert_abs_diff_eq!(r_enz[0], 100.0, epsilon = tol);
        assert_abs_diff_eq!(r_enz[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r_enz[2], 0.0, epsilon = tol);

        // Confirm higher latitude and longitude is (+E, +N, -Z)
        let x_sta = Vector3::new(R_EARTH, 0.0, 0.0);
        let x_geoc = Vector3::new(0.5, 0.5, 0.0);
        let r_ecef = position_geocentric_to_ecef(x_geoc, DEGREES).unwrap();

        let r_enz_geoc =
            relative_position_ecef_to_enz(x_sta, r_ecef, EllipsoidalConversionType::Geocentric);

        assert!(r_enz_geoc[0] > 0.0);
        assert!(r_enz_geoc[1] > 0.0);
        assert!(r_enz_geoc[2] < 0.0);

        // Confirm difference in geocentric and geodetic conversions
        let x_sta = Vector3::new(R_EARTH, 0.0, 0.0);
        let x_geod = Vector3::new(0.5, 0.5, 0.0);
        let r_ecef = position_geodetic_to_ecef(x_geod, DEGREES).unwrap();

        let r_enz_geod =
            relative_position_ecef_to_enz(x_sta, r_ecef, EllipsoidalConversionType::Geodetic);

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
            relative_position_enz_to_ecef(x_sta, r_enz, EllipsoidalConversionType::Geodetic);

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
        let rot1 = rotation_ellipsoid_to_sez(x_sta, DEGREES);

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
        let rot1 = rotation_ellipsoid_to_sez(x_sta, DEGREES);

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
        let rot1 = rotation_ellipsoid_to_sez(x_sta, DEGREES);

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
        let rot = rotation_ellipsoid_to_sez(x_sta, DEGREES);
        let rot_t = rotation_sez_to_ellipsoid(x_sta, DEGREES);

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
        let r_ecef = position_geocentric_to_ecef(x_geoc, DEGREES).unwrap();

        let r_sez_geoc =
            relative_position_ecef_to_sez(x_sta, r_ecef, EllipsoidalConversionType::Geocentric);

        assert!(r_sez_geoc[0] < 0.0);
        assert!(r_sez_geoc[1] > 0.0);
        assert!(r_sez_geoc[2] < 0.0);

        // Confirm difference in geocentric and geodetic conversions
        let x_sta = Vector3::new(R_EARTH, 0.0, 0.0);
        let x_geod = Vector3::new(0.5, 0.5, 0.0);
        let r_ecef = position_geodetic_to_ecef(x_geod, DEGREES).unwrap();

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
            relative_position_sez_to_ecef(x_sta, r_sez, EllipsoidalConversionType::Geodetic);

        assert_abs_diff_eq!(r_ecef[0], R_EARTH + 100.0, epsilon = tol);
        assert_abs_diff_eq!(r_ecef[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(r_ecef[2], 0.0, epsilon = tol);
    }

    #[test]
    fn test_position_enz_to_azel() {
        let tol = f64::EPSILON;

        // Directly above
        let r_enz = Vector3::new(0.0, 0.0, 100.0);
        let x_azel = position_enz_to_azel(r_enz, DEGREES);

        assert_abs_diff_eq!(x_azel[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[1], 90.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[2], 100.0, epsilon = tol);

        // North
        let r_enz = Vector3::new(0.0, 100.0, 0.0);
        let x_azel = position_enz_to_azel(r_enz, DEGREES);

        assert_abs_diff_eq!(x_azel[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[2], 100.0, epsilon = tol);

        // East
        let r_enz = Vector3::new(100.0, 0.0, 0.0);
        let x_azel = position_enz_to_azel(r_enz, DEGREES);

        assert_abs_diff_eq!(x_azel[0], 90.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[2], 100.0, epsilon = tol);

        // North-West
        let r_enz = Vector3::new(-100.0, 100.0, 0.0);
        let x_azel = position_enz_to_azel(r_enz, DEGREES);

        assert_abs_diff_eq!(x_azel[0], 315.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[2], 100.0 * 2.0_f64.sqrt(), epsilon = tol);
    }

    #[test]
    fn test_position_sez_to_azel() {
        let tol = f64::EPSILON;

        // Directly above
        let r_sez = Vector3::new(0.0, 0.0, 100.0);
        let x_azel = position_sez_to_azel(r_sez, DEGREES);

        assert_abs_diff_eq!(x_azel[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[1], 90.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[2], 100.0, epsilon = tol);

        // North
        let r_sez = Vector3::new(-100.0, 0.0, 0.0);
        let x_azel = position_sez_to_azel(r_sez, DEGREES);

        assert_abs_diff_eq!(x_azel[0], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[2], 100.0, epsilon = tol);

        // East
        let r_sez = Vector3::new(0.0, 100.0, 0.0);
        let x_azel = position_sez_to_azel(r_sez, DEGREES);

        assert_abs_diff_eq!(x_azel[0], 90.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[2], 100.0, epsilon = tol);

        // North-West
        let r_sez = Vector3::new(-100.0, -100.0, 0.0);
        let x_azel = position_sez_to_azel(r_sez, DEGREES);

        assert_abs_diff_eq!(x_azel[0], 315.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[1], 0.0, epsilon = tol);
        assert_abs_diff_eq!(x_azel[2], 100.0 * 2.0_f64.sqrt(), epsilon = tol);
    }

    #[test]
    #[parallel]
    fn test_batch_topocentric_match_scalar() {
        let sites = vec![
            Vector3::new(-122.4, 37.8, 0.0),
            Vector3::new(151.2, -33.9, 100.0),
            Vector3::new(0.0, 0.0, 10.0),
        ];
        let stations: Vec<Vector3<f64>> = sites
            .iter()
            .map(|s| position_geodetic_to_ecef(*s, DEGREES).unwrap())
            .collect();
        let targets: Vec<Vector3<f64>> = (0..3)
            .map(|i| Vector3::new(R_EARTH + 500e3 + 1e3 * i as f64, 1e5 * i as f64, 2e5))
            .collect();

        for i in 0..3 {
            assert_eq!(
                rotations_ellipsoid_to_enz(&sites, DEGREES)[i],
                rotation_ellipsoid_to_enz(sites[i], DEGREES)
            );
            assert_eq!(
                rotations_enz_to_ellipsoid(&sites, DEGREES)[i],
                rotation_enz_to_ellipsoid(sites[i], DEGREES)
            );
            assert_eq!(
                rotations_ellipsoid_to_sez(&sites, DEGREES)[i],
                rotation_ellipsoid_to_sez(sites[i], DEGREES)
            );
            assert_eq!(
                rotations_sez_to_ellipsoid(&sites, DEGREES)[i],
                rotation_sez_to_ellipsoid(sites[i], DEGREES)
            );
        }

        for ct in [
            EllipsoidalConversionType::Geodetic,
            EllipsoidalConversionType::Geocentric,
        ] {
            let enz = relative_positions_ecef_to_enz(&stations, &targets, ct).unwrap();
            let enz1 = relative_positions_ecef_to_enz(&stations[..1], &targets, ct).unwrap();
            let enz_one_target =
                relative_positions_ecef_to_enz(&stations, &targets[..1], ct).unwrap();
            let ecef = relative_positions_enz_to_ecef(&stations, &enz, ct).unwrap();
            let ecef1 = relative_positions_enz_to_ecef(&stations[..1], &enz1, ct).unwrap();
            let sez = relative_positions_ecef_to_sez(&stations, &targets, ct).unwrap();
            let sez1 = relative_positions_ecef_to_sez(&stations[..1], &targets, ct).unwrap();
            let ecef_s = relative_positions_sez_to_ecef(&stations, &sez, ct).unwrap();
            let ecef_s1 = relative_positions_sez_to_ecef(&stations[..1], &sez1, ct).unwrap();
            for i in 0..3 {
                assert_eq!(
                    enz[i],
                    relative_position_ecef_to_enz(stations[i], targets[i], ct)
                );
                assert_eq!(
                    enz1[i],
                    relative_position_ecef_to_enz(stations[0], targets[i], ct)
                );
                assert_eq!(
                    enz_one_target[i],
                    relative_position_ecef_to_enz(stations[i], targets[0], ct)
                );
                assert_eq!(
                    ecef[i],
                    relative_position_enz_to_ecef(stations[i], enz[i], ct)
                );
                assert_eq!(
                    ecef1[i],
                    relative_position_enz_to_ecef(stations[0], enz1[i], ct)
                );
                assert_eq!(
                    sez[i],
                    relative_position_ecef_to_sez(stations[i], targets[i], ct)
                );
                assert_eq!(
                    sez1[i],
                    relative_position_ecef_to_sez(stations[0], targets[i], ct)
                );
                assert_eq!(
                    ecef_s[i],
                    relative_position_sez_to_ecef(stations[i], sez[i], ct)
                );
                assert_eq!(
                    ecef_s1[i],
                    relative_position_sez_to_ecef(stations[0], sez1[i], ct)
                );
                for k in 0..3 {
                    assert_abs_diff_eq!(ecef[i][k], targets[i][k], epsilon = 1e-6);
                    assert_abs_diff_eq!(ecef_s[i][k], targets[i][k], epsilon = 1e-6);
                }
            }
            let azel = positions_enz_to_azel(&enz, DEGREES);
            let azel_s = positions_sez_to_azel(&sez, DEGREES);
            for i in 0..3 {
                assert_eq!(azel[i], position_enz_to_azel(enz[i], DEGREES));
                assert_eq!(azel_s[i], position_sez_to_azel(sez[i], DEGREES));
            }
            assert!(relative_positions_ecef_to_enz(&stations[..2], &targets, ct).is_err());
            assert!(
                relative_positions_ecef_to_enz(&stations[..1], &[], ct)
                    .unwrap()
                    .is_empty()
            );
        }
    }
}
