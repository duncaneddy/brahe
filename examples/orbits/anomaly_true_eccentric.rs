//! Example demonstrating conversion between true and eccentric anomaly.
//!
//! This example shows how to convert between true anomaly (nu) and eccentric anomaly (E)
//! for orbits with different eccentricities. The relationship depends on the orbital geometry.

use approx::assert_abs_diff_eq;
use brahe::constants::AngleFormat;
use brahe::orbits::keplerian::{anomaly_true_to_eccentric, anomaly_eccentric_to_true};

fn main() {
    // Test case 1: Low eccentricity (near-circular orbit)
    let nu_start = 45.0; // True anomaly in degrees
    let e_low = 0.01;    // Low eccentricity

    // Convert true to eccentric anomaly
    let e = anomaly_true_to_eccentric(nu_start, e_low, AngleFormat::Degrees);

    // Convert back to true anomaly
    let nu_end = anomaly_eccentric_to_true(e, e_low, AngleFormat::Degrees);

    // Verify round-trip conversion
    assert_abs_diff_eq!(nu_end, nu_start, epsilon = 1e-8);

    // Test case 2: Moderate eccentricity
    let nu_start = 90.0; // True anomaly in degrees
    let e_mod = 0.4;     // Moderate eccentricity

    // Convert true to eccentric anomaly
    let e = anomaly_true_to_eccentric(nu_start, e_mod, AngleFormat::Degrees);

    // Convert back to true anomaly
    let nu_end = anomaly_eccentric_to_true(e, e_mod, AngleFormat::Degrees);

    // Verify round-trip conversion
    assert_abs_diff_eq!(nu_end, nu_start, epsilon = 1e-8);

    // Test case 3: High eccentricity (elliptical orbit)
    let nu_start = 135.0; // True anomaly in degrees
    let e_high = 0.8;     // High eccentricity

    // Convert true to eccentric anomaly
    let e = anomaly_true_to_eccentric(nu_start, e_high, AngleFormat::Degrees);

    // Convert back to true anomaly
    let nu_end = anomaly_eccentric_to_true(e, e_high, AngleFormat::Degrees);

    // Verify round-trip conversion
    assert_abs_diff_eq!(nu_end, nu_start, epsilon = 1e-8);

    println!("✓ True-eccentric anomaly conversions validated successfully!");
}
