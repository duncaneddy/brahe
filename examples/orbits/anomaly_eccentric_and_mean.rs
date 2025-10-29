//! Convert between eccentric anomaly and mean anomaly.
//!
//! This example demonstrates the conversion between eccentric and mean anomaly
//! for a given eccentricity, and validates that the round-trip conversion
//! returns to the original value.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    let ecc = 45.0; // Starting eccentric anomaly (degrees)
    let e = 0.01;   // Eccentricity

    // Convert to mean anomaly
    let mean_anomaly = bh::orbits::anomaly_eccentric_to_mean(ecc, e, bh::constants::AngleFormat::Degrees);
    println!("Eccentric anomaly: {:.3} deg", ecc);
    println!("Mean anomaly:      {:.3} deg", mean_anomaly);

    // Convert back from mean to eccentric anomaly
    let ecc_2 = bh::orbits::anomaly_mean_to_eccentric(mean_anomaly, e, bh::constants::AngleFormat::Degrees).unwrap();
    println!("Round-trip result: {:.3} deg", ecc_2);

    // Verify round-trip accuracy
    println!("Difference:        {:.2e} deg", (ecc - ecc_2).abs());

    // Expected output:
    // Eccentric anomaly: 45.000 deg
    // Mean anomaly:      44.595 deg
    // Round-trip result: 45.000 deg
    // Difference:        0.00e0 deg
}
