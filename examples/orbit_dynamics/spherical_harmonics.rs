//! ```cargo
//! [dependencies]
//! brahe = { path = "../.." }
//! nalgebra = "0.33"
//! ```

//! Compute spherical harmonic gravitational acceleration for an Earth satellite

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create an epoch for frame transformations
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Define satellite position in ECI frame (LEO satellite at 500 km altitude)
    let a = bh::constants::R_EARTH + 500e3; // Semi-major axis (m)
    let e = 0.001;                          // Eccentricity
    let i = 97.8;                           // Inclination (deg)
    let raan = 45.0;                        // RAAN (deg)
    let argp = 30.0;                        // Argument of perigee (deg)
    let nu = 60.0;                          // True anomaly (deg)

    // Convert to Cartesian state
    let oe = na::SVector::<f64, 6>::new(a, e, i, raan, argp, nu);
    let state_eci = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let r_eci = na::Vector3::new(state_eci[0], state_eci[1], state_eci[2]); // Position vector (m)

    println!("Satellite position (ECI, m):");
    println!("  x = {:.3}", r_eci[0]);
    println!("  y = {:.3}", r_eci[1]);
    println!("  z = {:.3}", r_eci[2]);

    // Load gravity model (GGM05S - degree/order 180)
    let gravity_model = bh::orbit_dynamics::GravityModel::from_model_type(
        &bh::orbit_dynamics::GravityModelType::GGM05S
    ).unwrap();
    println!("\nGravity model: GGM05S (max degree {}, max order {})",
             gravity_model.n_max, gravity_model.m_max);

    // For spherical harmonics, we need the ECI to body-fixed rotation matrix
    // This rotates from ECI (inertial) to ECEF (Earth-fixed) frame
    let r_eci_ecef = bh::rotation_eci_to_ecef(epoch);

    // Compute spherical harmonic acceleration (degree 10, order 10)
    let n_max = 10;
    let m_max = 10;
    let accel_sh = bh::orbit_dynamics::accel_gravity_spherical_harmonics(
        r_eci, r_eci_ecef, &gravity_model, n_max, m_max
    );

    println!("\nSpherical harmonic acceleration (degree {}, order {}):", n_max, m_max);
    println!("  ax = {:.9} m/s²", accel_sh[0]);
    println!("  ay = {:.9} m/s²", accel_sh[1]);
    println!("  az = {:.9} m/s²", accel_sh[2]);

    // Compute point-mass for comparison
    let accel_pm = bh::orbit_dynamics::accel_point_mass_gravity(
        r_eci, na::Vector3::<f64>::zeros(), bh::constants::GM_EARTH
    );

    println!("\nPoint-mass acceleration:");
    println!("  ax = {:.9} m/s²", accel_pm[0]);
    println!("  ay = {:.9} m/s²", accel_pm[1]);
    println!("  az = {:.9} m/s²", accel_pm[2]);

    // Compute difference (perturbation due to non-spherical Earth)
    let accel_pert = accel_sh - accel_pm;

    println!("\nPerturbation (spherical harmonics - point mass):");
    println!("  Δax = {:.9} m/s²", accel_pert[0]);
    println!("  Δay = {:.9} m/s²", accel_pert[1]);
    println!("  Δaz = {:.9} m/s²", accel_pert[2]);
    println!("  Magnitude: {:.9} m/s²", accel_pert.norm());

    // Expected output:
    // Satellite position (ECI, m):
    //   x = 651307.572
    //   y = -668157.599
    //   z = 6811086.322

    // Gravity model: GGM05S (max degree 180, max order 180)

    // Spherical harmonic acceleration (degree 10, order 10):
    //   ax = -0.794811805 m/s²
    //   ay = 0.815141691 m/s²
    //   az = -8.333760910 m/s²

    // Point-mass acceleration:
    //   ax = -0.799028363 m/s²
    //   ay = 0.819700085 m/s²
    //   az = -8.355884974 m/s²

    // Perturbation (spherical harmonics - point mass):
    //   Δax = 0.004216558 m/s²
    //   Δay = -0.004558395 m/s²
    //   Δaz = 0.022124064 m/s²
    //   Magnitude: 0.022978958 m/s²
}
