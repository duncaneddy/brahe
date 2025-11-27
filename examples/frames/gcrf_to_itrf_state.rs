//! Transform state vector (position and velocity) from GCRF to ITRF

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
    // Orbital elements (degrees):
    //   a    = 6878136.300 m = 500.0 km altitude
    //   e    = 0.0100
    //   i    = 97.8000°
    //   Ω    = 15.0000°
    //   ω    = 30.0000°
    //   M    = 45.0000°

    let epc = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    println!("Epoch: {}", epc);
    // Epoch: 2024-01-01 12:00:00.000 UTC

    // Convert to GCRF Cartesian state
    let state_gcrf = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    println!("GCRF state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_gcrf[0], state_gcrf[1], state_gcrf[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_gcrf[3], state_gcrf[4], state_gcrf[5]);
    // GCRF state vector:
    //   Position: [1848964.106, -434937.468, 6560410.530] m
    //   Velocity: [-7098.379734, -2173.344867, 1913.333385] m/s

    // Transform to ITRF at specific epoch
    let state_itrf = bh::state_gcrf_to_itrf(epc, state_gcrf);

    println!("\nITRF state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_itrf[0], state_itrf[1], state_itrf[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s", state_itrf[3], state_itrf[4], state_itrf[5]);
    // ITRF state vector:
    //   Position: [757164.267, 1725863.563, 6564672.302] m
    //   Velocity: [989.350643, -7432.740021, 1896.768934] m/s
}