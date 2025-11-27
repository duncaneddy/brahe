//! Compute orbital period from a state vector.
//!
//! This example demonstrates computing the orbital period directly from a
//! Cartesian state vector in ECI coordinates, which is useful when you have
//! satellite state data but don't know the orbital elements.

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define orbital elements for a LEO satellite
    let a = bh::constants::R_EARTH + 500.0e3; // Semi-major axis (m)
    let e = 0.01; // Eccentricity
    let i = 97.8; // Inclination (degrees)
    let raan = 15.0; // Right ascension of ascending node (degrees)
    let argp = 30.0; // Argument of periapsis (degrees)
    let nu = 45.0; // True anomaly (degrees)

    // Convert to Cartesian state
    let oe = na::SVector::<f64, 6>::new(a, e, i, raan, argp, nu);
    let state_eci = bh::state_koe_to_eci(oe, bh::constants::AngleFormat::Degrees);

    println!("ECI State (position in km, velocity in km/s):");
    println!("  r = [{:.3}, {:.3}, {:.3}] km", state_eci[0]/1e3, state_eci[1]/1e3, state_eci[2]/1e3);
    println!("  v = [{:.3}, {:.3}, {:.3}] km/s", state_eci[3]/1e3, state_eci[4]/1e3, state_eci[5]/1e3);

    // Compute orbital period from state vector
    let period = bh::orbits::orbital_period_from_state(&state_eci, bh::constants::GM_EARTH);
    println!("\nOrbital period from state: {:.3} s", period);
    println!("Orbital period from state: {:.3} min", period / 60.0);

    // Verify against period computed from semi-major axis
    let period_from_sma = bh::orbits::orbital_period(a);
    println!("\nOrbital period from SMA: {:.3} s", period_from_sma);
    println!("Difference: {:.2e} s", (period - period_from_sma).abs());

    // Expected output:
    // ECI State (position in km, velocity in km/s):
    // r = [1848.964, -434.937, 6560.411] km
    // v = [-7.098, -2.173, 1.913] km/s

    // Orbital period from state: 5676.977 s
    // Orbital period from state: 94.616 min

    // Orbital period from SMA: 5676.977 s
    // Difference: 3.64e-12 s
}
