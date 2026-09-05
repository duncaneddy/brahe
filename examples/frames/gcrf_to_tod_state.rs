//! Transform a state vector from GCRF to the true equator and equinox of date (TOD)

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

    // Transform to TOD at the given epoch
    let state_tod = bh::state_gcrf_to_tod(epc, state_gcrf);

    println!("TOD state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_tod[0], state_tod[1], state_tod[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_tod[3], state_tod[4], state_tod[5]);

    let pos_gcrf = na::Vector3::new(state_gcrf[0], state_gcrf[1], state_gcrf[2]);
    let pos_tod = na::Vector3::new(state_tod[0], state_tod[1], state_tod[2]);
    let pos_diff = (pos_gcrf - pos_tod).norm();
    println!("Position difference norm: {:.3} m", pos_diff);
}
