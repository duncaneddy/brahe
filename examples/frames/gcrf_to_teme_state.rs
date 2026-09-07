//! Transform a state vector from GCRF to the true equator and mean equinox of date (TEME)

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

    let epc = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    println!("Epoch: {}", epc);

    // Convert to GCRF Cartesian state
    let state_gcrf = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    println!("\nGCRF state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_gcrf[0], state_gcrf[1], state_gcrf[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_gcrf[3], state_gcrf[4], state_gcrf[5]);

    // Transform to TEME at the given epoch
    let state_teme = bh::state_gcrf_to_teme(epc, state_gcrf);

    println!("TEME state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_teme[0], state_teme[1], state_teme[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_teme[3], state_teme[4], state_teme[5]);

    let pos_gcrf = na::Vector3::new(state_gcrf[0], state_gcrf[1], state_gcrf[2]);
    let pos_teme = na::Vector3::new(state_teme[0], state_teme[1], state_teme[2]);
    let pos_diff = (pos_gcrf - pos_teme).norm();
    println!("Position difference norm: {:.3} m", pos_diff);
}
