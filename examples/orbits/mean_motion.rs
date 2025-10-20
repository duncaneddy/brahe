//! Example demonstrating mean motion calculations.
//!
//! This example shows how to calculate mean motion from semi-major axis and vice versa
//! for different orbit types. Mean motion represents the average angular velocity of
//! an orbiting object.

use approx::assert_abs_diff_eq;
use brahe::constants::{R_EARTH, AngleFormat};
use brahe::orbits::keplerian::{mean_motion, semimajor_axis, orbital_period};
use std::f64::consts::PI;

fn main() {
    // Example 1: Low Earth Orbit (LEO)
    // LEO satellites typically complete ~15 orbits per day
    let a_leo = R_EARTH + 500e3; // 500 km altitude

    // Calculate mean motion in radians/second
    let n_leo = mean_motion(a_leo, AngleFormat::Radians);

    // Convert to revolutions per day
    let revs_per_day_leo = n_leo * 86400.0 / (2.0 * PI);

    // Verify typical LEO has ~15-16 revolutions per day
    assert!(revs_per_day_leo > 14.0 && revs_per_day_leo < 16.0);

    // Calculate semi-major axis from mean motion (round-trip)
    let a_leo_check = semimajor_axis(n_leo, AngleFormat::Radians);
    assert_abs_diff_eq!(a_leo_check, a_leo, epsilon = 1e-6);

    // Example 2: Geostationary Orbit (GEO)
    // GEO satellites complete exactly 1 orbit per sidereal day
    let a_geo = R_EARTH + 35786e3; // 35786 km altitude

    // Calculate mean motion in radians/second
    let n_geo = mean_motion(a_geo, AngleFormat::Radians);

    // Calculate orbital period (should be ~1 sidereal day = 86164 seconds)
    let t_geo = 2.0 * PI / n_geo;

    // Verify period is approximately 1 sidereal day
    let sidereal_day = 86164.0905; // seconds
    assert_abs_diff_eq!(t_geo, sidereal_day, epsilon = sidereal_day * 1e-4);

    // Calculate semi-major axis from mean motion (round-trip)
    let a_geo_check = semimajor_axis(n_geo, AngleFormat::Radians);
    assert_abs_diff_eq!(a_geo_check, a_geo, epsilon = 1e-6);

    // Example 3: Medium Earth Orbit (MEO) - GPS altitude
    // GPS satellites are at ~20,200 km altitude
    let a_meo = R_EARTH + 20200e3; // 20200 km altitude

    // Calculate mean motion in degrees/second
    let n_meo_deg = mean_motion(a_meo, AngleFormat::Degrees);

    // Convert to radians/second for period calculation
    let n_meo_rad = n_meo_deg.to_radians();
    let t_meo = 2.0 * PI / n_meo_rad;

    // GPS orbital period is approximately 12 hours (43200 seconds)
    assert_abs_diff_eq!(t_meo, 43200.0, epsilon = 43200.0 * 5e-2);

    // Calculate semi-major axis from mean motion (round-trip)
    let a_meo_check = semimajor_axis(n_meo_deg, AngleFormat::Degrees);
    assert_abs_diff_eq!(a_meo_check, a_meo, epsilon = 1e-6);

    // Example 4: Verify relationship with orbital_period function
    let t_leo_direct = orbital_period(a_leo);
    let t_leo_from_n = 2.0 * PI / n_leo;
    assert_abs_diff_eq!(t_leo_direct, t_leo_from_n, epsilon = 1e-10);

    println!("✓ Mean motion calculations validated successfully!");
}
