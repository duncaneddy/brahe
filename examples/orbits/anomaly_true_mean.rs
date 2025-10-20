//! Demonstrates conversion between true anomaly and mean anomaly.
//!
//! True anomaly describes the actual angular position of a body in its orbit,
//! while mean anomaly represents a fictitious uniform motion. This example
//! shows how to convert between them for different eccentricities.

use approx::assert_abs_diff_eq;
use brahe::constants::AngleFormat;
use brahe::orbits::keplerian::{anomaly_true_to_mean, anomaly_mean_to_true};

fn main() {
    // Define orbital parameters
    let nu_start = 45.0; // True anomaly in degrees
    let e = 0.3;         // Eccentricity

    // Convert true anomaly to mean anomaly
    let m = anomaly_true_to_mean(nu_start, e, AngleFormat::Degrees);

    // Convert back to true anomaly
    let nu_end = anomaly_mean_to_true(m, e, AngleFormat::Degrees).unwrap();

    // Verify round-trip conversion
    assert_abs_diff_eq!(nu_start, nu_end, epsilon = 1e-12);

    // Show progression for different eccentricities
    println!("True → Mean Anomaly Conversion:");
    for ecc in &[0.0, 0.1, 0.3, 0.5, 0.7] {
        let mean = anomaly_true_to_mean(nu_start, *ecc, AngleFormat::Degrees);
        println!("  e={:.1}: ν={:6.2}° → M={:6.2}°", ecc, nu_start, mean);
    }

    println!("\n✓ Anomaly conversions validated successfully!");
}
