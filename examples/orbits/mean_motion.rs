//! Compute mean motion from semi-major axis.
//!
//! This example demonstrates computing the mean motion (average angular rate)
//! of a satellite from its semi-major axis.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Define orbit parameters
    let a_leo = bh::constants::R_EARTH + 500.0e3; // LEO satellite at 500 km altitude
    let a_geo = bh::constants::R_EARTH + 35786e3; // GEO satellite

    // Compute mean motion in radians/s (Earth-specific)
    let n_leo_rad = bh::orbits::mean_motion(a_leo, bh::constants::AngleFormat::Radians);
    let n_geo_rad = bh::orbits::mean_motion(a_geo, bh::constants::AngleFormat::Radians);

    println!("Mean Motion in radians/second:");
    println!("  LEO (500 km): {:.6} rad/s", n_leo_rad);
    println!("  GEO:          {:.6} rad/s", n_geo_rad);

    // Compute mean motion in degrees/s
    let n_leo_deg = bh::orbits::mean_motion(a_leo, bh::constants::AngleFormat::Degrees);
    let n_geo_deg = bh::orbits::mean_motion(a_geo, bh::constants::AngleFormat::Degrees);

    println!("\nMean Motion in degrees/second:");
    println!("  LEO (500 km): {:.6} deg/s", n_leo_deg);
    println!("  GEO:          {:.6} deg/s", n_geo_deg);

    // Convert to degrees/day (common unit for TLEs)
    println!("\nMean Motion in degrees/day:");
    println!("  LEO (500 km): {:.3} deg/day", n_leo_deg * 86400.0);
    println!("  GEO:          {:.3} deg/day", n_geo_deg * 86400.0);

    // Verify using general function
    let n_leo_general = bh::orbits::mean_motion_general(a_leo, bh::constants::GM_EARTH, bh::constants::AngleFormat::Radians);
    println!("\nVerification (general function): {:.6} rad/s", n_leo_general);
    println!("Difference: {:.2e} rad/s", (n_leo_rad - n_leo_general).abs());

    // Expected output:
    // Mean Motion in radians/second:
    //   LEO (500 km): 0.001107 rad/s
    //   GEO:          0.000073 rad/s

    // Mean Motion in degrees/second:
    //   LEO (500 km): 0.063414 deg/s
    //   GEO:          0.004178 deg/s

    // Mean Motion in degrees/day:
    //   LEO (500 km): 5478.972 deg/day
    //   GEO:          360.986 deg/day

    // Verification (general function): 0.001107 rad/s
    // Difference: 0.00e+00 rad/s
}
