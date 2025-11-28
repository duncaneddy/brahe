//! ```cargo
//! [dependencies]
//! brahe = { path = "../.." }
//! nalgebra = "0.33"
//! ```

//! Compute point-mass gravitational acceleration for an Earth satellite

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define satellite position in ECI frame (LEO satellite at 500 km altitude)
    // Using Keplerian elements and converting to Cartesian
    let a = bh::constants::R_EARTH + 500e3; // Semi-major axis (m)
    let e = 0.001;                          // Eccentricity
    let i = 97.8;                           // Inclination (deg)
    let raan = 0.0;                         // RAAN (deg)
    let argp = 0.0;                         // Argument of perigee (deg)
    let nu = 0.0;                           // True anomaly (deg)

    // Convert to Cartesian state
    let oe = na::SVector::<f64, 6>::new(a, e, i, raan, argp, nu);
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let r_sat = na::Vector3::new(state[0], state[1], state[2]); // Position vector (m)

    println!("Satellite position (ECI, m):");
    println!("  x = {:.3}", r_sat[0]);
    println!("  y = {:.3}", r_sat[1]);
    println!("  z = {:.3}", r_sat[2]);

    // Compute point-mass gravitational acceleration
    // For Earth-centered case, central body is at origin
    let r_earth = na::Vector3::<f64>::zeros();
    let accel = bh::orbit_dynamics::accel_point_mass_gravity(r_sat, r_earth, bh::constants::GM_EARTH);

    println!("\nPoint-mass gravity acceleration (m/s²):");
    println!("  ax = {:.6}", accel[0]);
    println!("  ay = {:.6}", accel[1]);
    println!("  az = {:.6}", accel[2]);

    // Compute magnitude
    let accel_mag = accel.norm();
    println!("\nAcceleration magnitude: {:.6} m/s²", accel_mag);

    // Compare to theoretical value: GM/r²
    let r_mag = r_sat.norm();
    let accel_theoretical = bh::constants::GM_EARTH / (r_mag * r_mag);
    println!("Theoretical magnitude: {:.6} m/s²", accel_theoretical);

    // Expected output:
    // Satellite position (ECI, m):
    //   x = 6871258.164
    //   y = 0.000
    //   z = 0.000

    // Point-mass gravity acceleration (m/s²):
    //   ax = -8.442387
    //   ay = -0.000000
    //   az = -0.000000

    // Acceleration magnitude: 8.442387 m/s²
    // Theoretical magnitude: 8.442387 m/s²
}
