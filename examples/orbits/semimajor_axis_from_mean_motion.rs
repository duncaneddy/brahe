//! Compute semi-major axis from mean motion.
//!
//! This example demonstrates computing the semi-major axis from mean motion,
//! useful when working with TLE data or orbit design specifications that
//! provide mean motion instead of semi-major axis.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Example 1: ISS-like orbit with ~15.5 revolutions per day
    let n_iss = 15.5 * 360.0 / 86400.0; // Convert revs/day to deg/s
    let a_iss = bh::orbits::semimajor_axis(n_iss, bh::constants::AngleFormat::Degrees);
    let altitude_iss = a_iss - bh::constants::R_EARTH;

    println!("ISS-like Orbit (15.5 revs/day):");
    println!("  Mean motion: {:.6} deg/s", n_iss);
    println!("  Semi-major axis: {:.3} m", a_iss);
    println!("  Altitude: {:.3} km", altitude_iss / 1e3);

    // Example 2: Geosynchronous orbit (1 revolution per day)
    let n_geo = 1.0 * 360.0 / 86400.0; // 1 rev/day in deg/s
    let a_geo = bh::orbits::semimajor_axis(n_geo, bh::constants::AngleFormat::Degrees);
    let altitude_geo = a_geo - bh::constants::R_EARTH;

    println!("\nGeosynchronous Orbit (1 rev/day):");
    println!("  Mean motion: {:.6} deg/s", n_geo);
    println!("  Semi-major axis: {:.3} m", a_geo);
    println!("  Altitude: {:.3} km", altitude_geo / 1e3);

    // Example 3: Using radians
    let n_leo_rad = 0.001; // rad/s
    let a_leo = bh::orbits::semimajor_axis(n_leo_rad, bh::constants::AngleFormat::Radians);

    println!("\nLEO from radians/s:");
    println!("  Mean motion: {:.6} rad/s", n_leo_rad);
    println!("  Semi-major axis: {:.3} m", a_leo);
    println!("  Altitude: {:.3} km", (a_leo - bh::constants::R_EARTH) / 1e3);

    // Verify round-trip conversion
    let n_verify = bh::orbits::mean_motion(a_iss, bh::constants::AngleFormat::Degrees);
    println!("\nRound-trip verification:");
    println!("  Original mean motion: {:.6} deg/s", n_iss);
    println!("  Computed mean motion: {:.6} deg/s", n_verify);
    println!("  Difference: {:.2e} deg/s", (n_iss - n_verify).abs());

    // Expected output:
    // ISS-like Orbit (15.5 revs/day):
    //   Mean motion: 0.064583 deg/s
    //   Semi-major axis: 6794863.068 m
    //   Altitude: 416.727 km

    // Geosynchronous Orbit (1 rev/day):
    //   Mean motion: 0.004167 deg/s
    //   Semi-major axis: 42241095.664 m
    //   Altitude: 35862.959 km

    // LEO from radians/s:
    //   Mean motion: 0.001000 rad/s
    //   Semi-major axis: 7359459.593 m
    //   Altitude: 981.323 km

    // Round-trip verification:
    //   Original mean motion: 0.064583 deg/s
    //   Computed mean motion: 0.064583 deg/s
    //   Difference: 9.71e-17 deg/s
}
