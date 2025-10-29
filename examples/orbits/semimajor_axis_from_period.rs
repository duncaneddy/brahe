//! Compute semi-major axis from orbital period.
//!
//! This example demonstrates the inverse relationship between orbital period
//! and semi-major axis, useful for orbit design when you know the desired
//! orbital period.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Example 1: LEO satellite with 98-minute period
    let period_leo = 98.0 * 60.0; // 98 minutes in seconds
    let a_leo = bh::orbits::semimajor_axis_from_orbital_period(period_leo);
    let altitude_leo = a_leo - bh::constants::R_EARTH;

    println!("LEO Satellite (98 min period):");
    println!("  Semi-major axis: {:.3} m", a_leo);
    println!("  Altitude: {:.3} km", altitude_leo / 1e3);

    // Example 2: Geosynchronous orbit (24-hour period)
    let period_geo = 24.0 * 3600.0; // 24 hours in seconds
    let a_geo = bh::orbits::semimajor_axis_from_orbital_period(period_geo);
    let altitude_geo = a_geo - bh::constants::R_EARTH;

    println!("\nGeosynchronous Orbit (24 hour period):");
    println!("  Semi-major axis: {:.3} m", a_geo);
    println!("  Altitude: {:.3} km", altitude_geo / 1e3);

    // Example 3: Using general function for Moon orbit
    let period_moon = 27.3 * 24.0 * 3600.0; // 27.3 days in seconds
    let a_moon = bh::orbits::semimajor_axis_from_orbital_period_general(period_moon, bh::constants::GM_EARTH);

    println!("\nMoon's orbit (27.3 day period):");
    println!("  Semi-major axis: {:.3} km", a_moon / 1e3);

    // Verify round-trip conversion
    let period_verify = bh::orbits::orbital_period(a_leo);
    println!("\nRound-trip verification:");
    println!("  Original period: {:.3} s", period_leo);
    println!("  Computed period: {:.3} s", period_verify);
    println!("  Difference: {:.2e} s", (period_leo - period_verify).abs());

    // Expected output:
    // LEO Satellite (98 min period):
    //   Semi-major axis: 7041160.278 m
    //   Altitude: 663.024 km

    // Geosynchronous Orbit (24 hour period):
    //   Semi-major axis: 42241095.664 m
    //   Altitude: 35862.959 km

    // Moon's orbit (27.3 day period):
    //   Semi-major axis: 382980.745 km

    // Round-trip verification:
    //   Original period: 5880.000 s
    //   Computed period: 5880.000 s
    //   Difference: 8.19e-12 s
}
