//! Transform state vector (position and velocity) from ECI to ECEF

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define orbital elements in degrees
    // LEO satellite: 500 km altitude, sun-synchronous orbit
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,  // Semi-major axis (m)
        0.01,                  // Eccentricity
        97.8,                  // Inclination (deg)
        15.0,                  // Right ascension of ascending node (deg)
        30.0,                  // Argument of periapsis (deg)
        45.0,                  // Mean anomaly (deg)
    );

    println!("Orbital elements (degrees):");
    println!("  a    = {:.3} m = {:.1} km altitude", oe[0], (oe[0] - bh::R_EARTH) / 1e3);
    println!("  e    = {:.4}", oe[1]);
    println!("  i    = {:.4}°", oe[2]);
    println!("  Ω    = {:.4}°", oe[3]);
    println!("  ω    = {:.4}°", oe[4]);
    println!("  M    = {:.4}°\n", oe[5]);

    let epc = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    println!("Epoch: {}", epc);

    // Convert to ECI Cartesian state
    let state_eci = bh::state_osculating_to_cartesian(oe, bh::AngleFormat::Degrees);

    println!("ECI state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_eci[0], state_eci[1], state_eci[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_eci[3], state_eci[4], state_eci[5]);

    // Transform to ECEF at specific epoch
    let state_ecef = bh::state_eci_to_ecef(epc, state_eci);

    println!("\nECEF state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_ecef[0], state_ecef[1], state_ecef[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s", state_ecef[3], state_ecef[4], state_ecef[5]);

    // Expected output:
    // Position: [3210319.128, 5246384.459, 2649959.679] m
    // Velocity: [-5539.021093, 3461.463903, 3791.888925] m/s
}
