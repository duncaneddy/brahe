//! Transform state vector (position and velocity) from EME2000 to GCRF

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
    // Orbital elements (degrees):
    //   a    = 6878136.300 m = 500.0 km altitude
    //   e    = 0.0100
    //   i    = 97.8000°
    //   Ω    = 15.0000°
    //   ω    = 30.0000°
    //   M    = 45.0000°

    // Convert to EME2000 Cartesian state
    // Note: state_osculating_to_cartesian produces EME2000 states by default
    let state_eme2000 = bh::state_osculating_to_cartesian(oe, bh::AngleFormat::Degrees);

    println!("EME2000 state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_eme2000[0], state_eme2000[1], state_eme2000[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_eme2000[3], state_eme2000[4], state_eme2000[5]);
    // EME2000 state vector:
    //   Position: [1848964.106, -434937.468, 6560410.530] m
    //   Velocity: [-7098.379734, -2173.344867, 1913.333385] m/s

    // Transform to GCRF (constant transformation, no epoch needed)
    let state_gcrf = bh::state_eme2000_to_gcrf(state_eme2000);

    println!("GCRF state vector:");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_gcrf[0], state_gcrf[1], state_gcrf[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_gcrf[3], state_gcrf[4], state_gcrf[5]);
    // GCRF state vector:
    //   Position: [1848963.547, -434937.816, 6560410.665] m
    //   Velocity: [-7098.380042, -2173.344428, 1913.332741] m/s

    // Transform back to EME2000 to verify round-trip
    let state_eme2000_back = bh::state_gcrf_to_eme2000(state_gcrf);

    println!("EME2000 state vector (transformed from GCRF):");
    println!("  Position: [{:.3}, {:.3}, {:.3}] m", state_eme2000_back[0], state_eme2000_back[1], state_eme2000_back[2]);
    println!("  Velocity: [{:.6}, {:.6}, {:.6}] m/s\n", state_eme2000_back[3], state_eme2000_back[4], state_eme2000_back[5]);
    // EME2000 state vector (transformed from GCRF):
    //   Position: [1848964.106, -434937.468, 6560410.530] m
    //   Velocity: [-7098.379734, -2173.344867, 1913.333385] m/s

    // Verify round-trip transformation
    let diff_pos = (na::Vector3::new(state_eme2000[0], state_eme2000[1], state_eme2000[2]) -
                    na::Vector3::new(state_eme2000_back[0], state_eme2000_back[1], state_eme2000_back[2])).norm();
    let diff_vel = (na::Vector3::new(state_eme2000[3], state_eme2000[4], state_eme2000[5]) -
                    na::Vector3::new(state_eme2000_back[3], state_eme2000_back[4], state_eme2000_back[5])).norm();
    println!("Round-trip error:");
    println!("  Position: {:.6e} m", diff_pos);
    println!("  Velocity: {:.6e} m/s", diff_vel);
    println!("\nNote: Transformation is constant (time-independent, no epoch needed)");
// Round-trip error:
//   Position: 3.863884e-08 m
//   Velocity: 3.876304e-11 m/s
}
