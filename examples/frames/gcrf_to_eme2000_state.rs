//! Transform state vector (position and velocity) from GCRF to EME2000

use brahe as bh;
use nalgebra as na;

fn main() {
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

    // (Starting in EME2000 to get GCRF representation)
    let state_eme2000_orig = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let state_gcrf = bh::state_eme2000_to_gcrf(state_eme2000_orig);

    println!("GCRF state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_gcrf[0], state_gcrf[1], state_gcrf[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_gcrf[3], state_gcrf[4], state_gcrf[5]);

    let state_eme2000 = bh::state_gcrf_to_eme2000(state_gcrf);

    println!("EME2000 state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_eme2000[0], state_eme2000[1], state_eme2000[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_eme2000[3], state_eme2000[4], state_eme2000[5]);

    // Transform back to GCRF to verify round-trip
    let state_gcrf_back = bh::state_eme2000_to_gcrf(state_eme2000);

    println!("GCRF state vector (transformed from EME2000):");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_gcrf_back[0], state_gcrf_back[1], state_gcrf_back[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_gcrf_back[3], state_gcrf_back[4], state_gcrf_back[5]);

    let diff_pos = (na::Vector3::new(state_gcrf[0], state_gcrf[1], state_gcrf[2]) -
                    na::Vector3::new(state_gcrf_back[0], state_gcrf_back[1], state_gcrf_back[2])).norm();
    let diff_vel = (na::Vector3::new(state_gcrf[3], state_gcrf[4], state_gcrf[5]) -
                    na::Vector3::new(state_gcrf_back[3], state_gcrf_back[4], state_gcrf_back[5])).norm();
    println!("Round-trip error:");
    println!("  Position: {:.6e} m", diff_pos);
    println!("  Velocity: {:.6e} m/s", diff_vel);
    println!("\nNote: Transformation is constant (time-independent, no epoch needed)");
}

