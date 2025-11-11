//! Transform state vector (position and velocity) from ITRF to GCRF

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

    // Convert to GCRF Cartesian state
    let state_gcrf = bh::state_osculating_to_cartesian(oe, bh::AngleFormat::Degrees);

    // Define epoch
    let epc = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    println!("Epoch: {}", epc);
    println!("GCRF state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_gcrf[0], state_gcrf[1], state_gcrf[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_gcrf[3], state_gcrf[4], state_gcrf[5]);
    // Position: [1848964.106, -434937.468, 6560410.530] m
    // Velocity: [-7098.379734, -2173.344867, 1913.333385] m/s

    // Transform to ITRF
    let state_itrf = bh::state_gcrf_to_itrf(epc, state_gcrf);

    println!("ITRF state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_itrf[0], state_itrf[1], state_itrf[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_itrf[3], state_itrf[4], state_itrf[5]);
    // Position: [757164.267, 1725863.563, 6564672.302] m
    // Velocity: [989.350643, -7432.740021, 1896.768934] m/s

    // Transform back to GCRF
    let state_gcrf_back = bh::state_itrf_to_gcrf(epc, state_itrf);

    println!("\nGCRF state vector (transformed from ITRF):");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_gcrf_back[0], state_gcrf_back[1], state_gcrf_back[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s", state_gcrf_back[3], state_gcrf_back[4], state_gcrf_back[5]);
    // Position: [1848964.106, -434937.468, 6560410.530] m
    // Velocity: [-7098.379734, -2173.344867, 1913.333385] m/s

    // Verify round-trip transformation
    let diff_pos = (na::Vector3::new(state_gcrf[0], state_gcrf[1], state_gcrf[2]) -
                    na::Vector3::new(state_gcrf_back[0], state_gcrf_back[1], state_gcrf_back[2])).norm();
    let diff_vel = (na::Vector3::new(state_gcrf[3], state_gcrf[4], state_gcrf[5]) -
                    na::Vector3::new(state_gcrf_back[3], state_gcrf_back[4], state_gcrf_back[5])).norm();
    println!("\nRound-trip error:");
    println!("  Position: {:.6e} m", diff_pos);
    println!("  Velocity: {:.6e} m/s", diff_vel);

    // Expected output:
    //   Position: 9.617484e-10 m
    //   Velocity: 9.094947e-13 m/s
}
