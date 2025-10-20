//! Demonstrates calculation of orbital period using Kepler's Third Law.
//!
//! The orbital period depends only on the semi-major axis and central body
//! gravitational parameter, independent of eccentricity.

use approx::assert_relative_eq;
use brahe::constants::R_EARTH;
use brahe::orbits::keplerian::orbital_period;

fn main() {
    // Define orbital parameters
    let a = R_EARTH + 500e3; // Semi-major axis: 500 km altitude (meters)

    // Calculate orbital period (uses GM_EARTH by default)
    let t = orbital_period(a); // Returns period in seconds

    // Expected period for ~500 km LEO is approximately 94.6 minutes
    let expected_minutes = 94.6;
    assert_relative_eq!(t / 60.0, expected_minutes, max_relative = 0.01);

    println!("Orbital Period Calculation:");
    println!("  Altitude: 500 km");
    println!("  Semi-major axis: {:.1} km", a / 1000.0);
    println!("  Period: {:.2} seconds", t);
    println!("  Period: {:.2} minutes", t / 60.0);
    println!("  Period: {:.4} hours", t / 3600.0);

    // Show periods for different altitudes
    println!("\nPeriods for various altitudes:");
    for alt_km in &[200, 400, 600, 800, 1000] {
        let a_temp = R_EARTH + (*alt_km as f64) * 1000.0;
        let t_temp = orbital_period(a_temp);
        println!("  {:4} km: {:6.2} minutes", alt_km, t_temp / 60.0);
    }

    println!("\n✓ Orbital period calculations validated successfully!");
}
