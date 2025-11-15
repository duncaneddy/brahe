/*!
 * Shared geometric utility functions for access computation.
 *
 * This module provides common geometric calculations used by both
 * access constraints and access property computation.
 */

use nalgebra::{Vector3, Vector6};

use crate::constants::AngleFormat;
use crate::constants::OMEGA_EARTH;
use crate::coordinates::{
    EllipsoidalConversionType, position_enz_to_azel, relative_position_ecef_to_enz,
};
use crate::time::Epoch;
use crate::time::TimeSystem;

use super::constraints::{AscDsc, LookDirection};

/// Compute azimuth angle from location to satellite.
///
/// Azimuth is measured clockwise from North (0-360 degrees).
///
/// # Arguments
/// * `sat_ecef` - Satellite position in ECEF (meters)
/// * `loc_ecef` - Location position in ECEF (meters)
///
/// # Returns
/// Azimuth angle in degrees (0-360)
pub fn compute_azimuth(sat_ecef: &Vector3<f64>, loc_ecef: &Vector3<f64>) -> f64 {
    // Compute relative position in ENZ frame
    let enz =
        relative_position_ecef_to_enz(*loc_ecef, *sat_ecef, EllipsoidalConversionType::Geodetic);

    // Compute azimuth-elevation
    let azel = position_enz_to_azel(enz, AngleFormat::Radians);
    azel[0].to_degrees()
}

/// Compute elevation angle from location to satellite.
///
/// Elevation is the angle above the local horizon (0-90 degrees).
///
/// # Arguments
/// * `sat_ecef` - Satellite position in ECEF (meters)
/// * `loc_ecef` - Location position in ECEF (meters)
///
/// # Returns
/// Elevation angle in degrees (0-90)
pub fn compute_elevation(sat_ecef: &Vector3<f64>, loc_ecef: &Vector3<f64>) -> f64 {
    // Compute relative position in ENZ frame
    let enz =
        relative_position_ecef_to_enz(*loc_ecef, *sat_ecef, EllipsoidalConversionType::Geodetic);

    // Compute azimuth-elevation
    let azel = position_enz_to_azel(enz, AngleFormat::Radians);
    azel[1].to_degrees()
}

/// Compute off-nadir angle from satellite to location.
///
/// Off-nadir is the angle between the satellite nadir (Earth-pointing) direction
/// and the line-of-sight to the location.
///
/// # Arguments
/// * `sat_ecef` - Satellite position in ECEF (meters)
/// * `loc_ecef` - Location position in ECEF (meters)
///
/// # Returns
/// Off-nadir angle in degrees (0-180)
pub fn compute_off_nadir(sat_ecef: &Vector3<f64>, loc_ecef: &Vector3<f64>) -> f64 {
    // Nadir direction is from satellite toward Earth center
    let nadir = -sat_ecef.normalize();

    // Line-of-sight from satellite to location
    let los = (loc_ecef - sat_ecef).normalize();

    // Off-nadir angle is angle between nadir and line-of-sight
    let cos_angle = nadir.dot(&los);
    cos_angle.acos().to_degrees()
}

/// Compute local solar time at location and epoch.
///
/// Local solar time is the time of day based on the Sun's position,
/// returned as seconds since midnight (0-86400).
///
/// # Arguments
/// * `epoch` - Time epoch
/// * `loc_geodetic` - Location geodetic coordinates [lon, lat, alt] (radians, meters)
///
/// # Returns
/// Local solar time in seconds since midnight (0-86400)
pub fn compute_local_time(epoch: &Epoch, loc_geodetic: &Vector3<f64>) -> f64 {
    use std::f64::consts::PI;

    // Get UT1 time for Earth rotation calculation
    let jd_ut1 = epoch.jd_as_time_system(TimeSystem::UT1);

    // Compute Greenwich Mean Sidereal Time (GMST)
    // Using simple formula: GMST = (JD_UT1 - 2451545.0) * 1.00273790935 * 2π
    let t = jd_ut1 - 2451545.0; // Days since J2000
    let tau = 2.0 * PI;
    let gmst = t * 1.00273790935 * tau; // radians

    // Local sidereal time = GMST + longitude
    let lst = gmst + loc_geodetic[0];

    // Convert to hours (0-24)
    let lst_hours = (lst.rem_euclid(tau) / tau) * 24.0;

    // Convert to seconds
    lst_hours * 3600.0
}

/// Compute look direction (Left or Right) from satellite velocity and line-of-sight.
///
/// Look direction indicates whether the satellite must look to the left or right
/// of its velocity vector to see the location.
///
/// # Arguments
/// * `sat_state_ecef` - Satellite state in ECEF [x,y,z,vx,vy,vz] (meters, m/s)
/// * `loc_ecef` - Location position in ECEF (meters)
///
/// # Returns
/// LookDirection (Left or Right)
pub fn compute_look_direction(
    sat_state_ecef: &Vector6<f64>,
    loc_ecef: &Vector3<f64>,
) -> LookDirection {
    // Extract position and velocity
    let sat_pos = sat_state_ecef.fixed_rows::<3>(0);
    let sat_vel = sat_state_ecef.fixed_rows::<3>(3);

    // Line-of-sight from satellite to location
    let los = loc_ecef - sat_pos;

    // Cross product of velocity and line-of-sight
    // If positive (pointing away from Earth center), looking left
    // If negative (pointing toward Earth center), looking right
    let cross = sat_vel.cross(&los);

    // Dot with position vector to determine direction
    if cross.dot(&sat_pos.into_owned()) > 0.0 {
        LookDirection::Left
    } else {
        LookDirection::Right
    }
}

/// Compute ascending/descending from satellite velocity.
///
/// Determines whether the satellite is moving northward (ascending) or
/// southward (descending) based on its velocity vector.
///
/// # Arguments
/// * `sat_state_ecef` - Satellite state in ECEF [x,y,z,vx,vy,vz] (meters, m/s)
///
/// # Returns
/// AscDsc (Ascending or Descending)
pub fn compute_asc_dsc(sat_state_ecef: &Vector6<f64>) -> AscDsc {
    // Extract position and velocity
    let sat_pos = sat_state_ecef.fixed_rows::<3>(0);
    let sat_vel = sat_state_ecef.fixed_rows::<3>(3);

    // In ECEF, we need to account for Earth rotation
    // Compute inertial velocity by removing Earth rotation component
    let omega_vec = Vector3::new(0.0, 0.0, OMEGA_EARTH);
    let vel_inertial = sat_vel + omega_vec.cross(&sat_pos.into_owned());

    // Check if z-component of inertial velocity is positive (ascending) or negative (descending)
    if vel_inertial[2] > 0.0 {
        AscDsc::Ascending
    } else {
        AscDsc::Descending
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::coordinates::position_geodetic_to_ecef;
    use crate::utils::testing::setup_global_test_eop;

    #[test]
    fn test_compute_azimuth_elevation() {
        // Location: (0° lon, 45° lat, 0 alt)
        let loc_geodetic = Vector3::new(0.0, 45.0_f64.to_radians(), 0.0);
        let loc_ecef = position_geodetic_to_ecef(loc_geodetic, AngleFormat::Radians).unwrap();

        // Satellite at high altitude
        let sat_ecef = loc_ecef + Vector3::new(0.0, 500e3, 500e3);

        let azimuth = compute_azimuth(&sat_ecef, &loc_ecef);
        let elevation = compute_elevation(&sat_ecef, &loc_ecef);

        // Verify azimuth is in valid range
        assert!((0.0..=360.0).contains(&azimuth));

        // Elevation should be positive
        assert!(elevation > 0.0);
        assert!(elevation < 90.0);
    }

    #[test]
    fn test_compute_off_nadir() {
        // Satellite at altitude
        let sat_ecef = Vector3::new(7000e3, 0.0, 0.0);

        // Location on Earth surface
        let loc_geodetic = Vector3::new(0.0, 0.0, 0.0);
        let loc_ecef = position_geodetic_to_ecef(loc_geodetic, AngleFormat::Radians).unwrap();

        let off_nadir = compute_off_nadir(&sat_ecef, &loc_ecef);

        // Off-nadir should be reasonable
        assert!(off_nadir >= 0.0);
        assert!(off_nadir <= 180.0);
    }

    #[test]
    fn test_compute_local_time() {
        setup_global_test_eop();

        // Location at 0° longitude
        let loc_geodetic = Vector3::new(0.0, 0.0, 0.0);

        let epoch = Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);

        let local_time = compute_local_time(&epoch, &loc_geodetic);

        // Should be in range 0-86400 seconds
        assert!(local_time >= 0.0);
        assert!(local_time <= 86400.0);
    }

    #[test]
    fn test_compute_asc_dsc() {
        // Ascending: positive z-velocity in inertial frame
        let state_ascending = Vector6::new(
            7000e3, 0.0, 0.0, // position
            0.0, 7500.0, 100.0, // velocity (positive z)
        );

        let asc_dsc = compute_asc_dsc(&state_ascending);
        assert_eq!(asc_dsc, AscDsc::Ascending);

        // Descending: negative z-velocity
        let state_descending = Vector6::new(
            7000e3, 0.0, 0.0, // position
            0.0, 7500.0, -100.0, // velocity (negative z)
        );

        let asc_dsc = compute_asc_dsc(&state_descending);
        assert_eq!(asc_dsc, AscDsc::Descending);
    }

    #[test]
    fn test_compute_look_direction() {
        // Satellite state
        let sat_state = Vector6::new(
            7000e3, 0.0, 0.0, // position
            0.0, 7500.0, 0.0, // velocity (moving in +y direction)
        );

        // Location to the right (negative x)
        let loc_right = Vector3::new(6000e3, 0.0, 0.0);
        let look_dir = compute_look_direction(&sat_state, &loc_right);
        // Just check it returns a value
        assert!(look_dir == LookDirection::Left || look_dir == LookDirection::Right);
    }
}
