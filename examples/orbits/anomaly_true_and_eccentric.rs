//! Convert between true anomaly and eccentric anomaly.
//!
//! This example demonstrates the conversion between true and eccentric anomaly
//! for a given eccentricity, and validates that the round-trip conversion
//! returns to the original value.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    let nu = 45.0; // Starting true anomaly (degrees)
    let e = 0.01;  // Eccentricity

    // Convert to eccentric anomaly
    let ecc_anomaly = bh::orbits::anomaly_true_to_eccentric(nu, e, bh::constants::AngleFormat::Degrees);
    println!("True anomaly:      {:.3} deg", nu);
    println!("Eccentric anomaly: {:.3} deg", ecc_anomaly);

    // Convert back from eccentric to true anomaly
    let nu_2 = bh::orbits::anomaly_eccentric_to_true(ecc_anomaly, e, bh::constants::AngleFormat::Degrees);
    println!("Round-trip result: {:.3} deg", nu_2);

    // Verify round-trip accuracy
    println!("Difference:        {:.2e} deg", (nu - nu_2).abs());

    // Expected output:
    // True anomaly:      45.000 deg
    // Eccentric anomaly: 44.596 deg
    // Round-trip result: 45.000 deg
    // Difference:        0.00e0 deg
}
