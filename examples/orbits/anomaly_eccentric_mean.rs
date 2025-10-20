//! Example demonstrating conversion between eccentric and mean anomaly.
//!
//! This example shows how to convert between eccentric anomaly (E) and mean anomaly (M)
//! for orbits with different eccentricities. The conversion involves solving Kepler's equation.

use approx::assert_abs_diff_eq;
use brahe::constants::AngleFormat;
use brahe::orbits::keplerian::{anomaly_eccentric_to_mean, anomaly_mean_to_eccentric};

fn main() {
    // Test case 1: Low eccentricity (near-circular orbit)
    let e_start = 45.0; // Eccentric anomaly in degrees
    let e_low = 0.01;   // Low eccentricity

    // Convert eccentric to mean anomaly
    let m = anomaly_eccentric_to_mean(e_start, e_low, AngleFormat::Degrees);

    // Convert back to eccentric anomaly
    let e_end = anomaly_mean_to_eccentric(m, e_low, AngleFormat::Degrees).unwrap();

    // Verify round-trip conversion
    assert_abs_diff_eq!(e_end, e_start, epsilon = 1e-8);

    // Test case 2: Moderate eccentricity
    let e_start = 60.0; // Eccentric anomaly in degrees
    let e_mod = 0.3;    // Moderate eccentricity

    // Convert eccentric to mean anomaly
    let m = anomaly_eccentric_to_mean(e_start, e_mod, AngleFormat::Degrees);

    // Convert back to eccentric anomaly
    let e_end = anomaly_mean_to_eccentric(m, e_mod, AngleFormat::Degrees).unwrap();

    // Verify round-trip conversion
    assert_abs_diff_eq!(e_end, e_start, epsilon = 1e-8);

    // Test case 3: High eccentricity (elliptical orbit)
    let e_start = 120.0; // Eccentric anomaly in degrees
    let e_high = 0.7;    // High eccentricity

    // Convert eccentric to mean anomaly
    let m = anomaly_eccentric_to_mean(e_start, e_high, AngleFormat::Degrees);

    // Convert back to eccentric anomaly
    let e_end = anomaly_mean_to_eccentric(m, e_high, AngleFormat::Degrees).unwrap();

    // Verify round-trip conversion
    assert_abs_diff_eq!(e_end, e_start, epsilon = 1e-8);

    println!("✓ Eccentric-mean anomaly conversions validated successfully!");
}
