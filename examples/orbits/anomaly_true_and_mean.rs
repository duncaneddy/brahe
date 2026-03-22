//! Convert between true anomaly and mean anomaly.
//!
//! This example demonstrates the conversion between true and mean anomaly
//! for a given eccentricity, and validates that the round-trip conversion
//! returns to the original value.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    let nu = 45.0; // Starting true anomaly (degrees)
    let e = 0.01;  // Eccentricity

    // Convert to mean anomaly
    let mean_anomaly = bh::orbits::anomaly_true_to_mean(nu, e, bh::constants::AngleFormat::Degrees);
    println!("True anomaly:      {:.3} deg", nu);
    println!("Mean anomaly:      {:.3} deg", mean_anomaly);

    // Convert back from mean to true anomaly
    let nu_2 = bh::orbits::anomaly_mean_to_true(mean_anomaly, e, bh::constants::AngleFormat::Degrees).unwrap();
    println!("Round-trip result: {:.3} deg", nu_2);

    // Verify round-trip accuracy
    println!("Difference:        {:.2e} deg", (nu - nu_2).abs());

}

