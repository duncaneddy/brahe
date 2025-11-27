//! ```cargo
//! [dependencies]
//! brahe = { path = "../.." }
//! nalgebra = "0.33"
//! ```

//! Compute third-body gravitational perturbations from Sun and Moon

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create an epoch
    let epoch = bh::Epoch::from_datetime(2024, 6, 21, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Define satellite position (GPS-like MEO satellite at ~20,000 km altitude)
    let a = bh::constants::R_EARTH + 20180e3; // Semi-major axis (m)
    let e = 0.01;                             // Eccentricity
    let i = 55.0;                             // Inclination (deg)
    let raan = 120.0;                         // RAAN (deg)
    let argp = 45.0;                          // Argument of perigee (deg)
    let nu = 90.0;                            // True anomaly (deg)

    // Convert to Cartesian state
    let oe = na::SVector::<f64, 6>::new(a, e, i, raan, argp, nu);
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let r_sat = na::Vector3::new(state[0], state[1], state[2]); // Position vector (m)

    println!("Satellite position (ECI, m):");
    println!("  x = {:.1} km", r_sat[0] / 1e3);
    println!("  y = {:.1} km", r_sat[1] / 1e3);
    println!("  z = {:.1} km", r_sat[2] / 1e3);
    println!("  Altitude: {:.1} km", (r_sat.norm() - bh::constants::R_EARTH) / 1e3);

    // Compute Sun perturbation using analytical model
    let accel_sun = bh::orbit_dynamics::accel_third_body_sun(epoch, r_sat);

    println!("\nSun third-body acceleration (analytical):");
    println!("  ax = {:.12} m/s²", accel_sun[0]);
    println!("  ay = {:.12} m/s²", accel_sun[1]);
    println!("  az = {:.12} m/s²", accel_sun[2]);
    println!("  Magnitude: {:.12} m/s²", accel_sun.norm());

    // Compute Moon perturbation using analytical model
    let accel_moon = bh::orbit_dynamics::accel_third_body_moon(epoch, r_sat);

    println!("\nMoon third-body acceleration (analytical):");
    println!("  ax = {:.12} m/s²", accel_moon[0]);
    println!("  ay = {:.12} m/s²", accel_moon[1]);
    println!("  az = {:.12} m/s²", accel_moon[2]);
    println!("  Magnitude: {:.12} m/s²", accel_moon.norm());

    // Compute combined Sun + Moon acceleration
    let accel_combined = accel_sun + accel_moon;

    println!("\nCombined Sun + Moon acceleration:");
    println!("  ax = {:.12} m/s²", accel_combined[0]);
    println!("  ay = {:.12} m/s²", accel_combined[1]);
    println!("  az = {:.12} m/s²", accel_combined[2]);
    println!("  Magnitude: {:.12} m/s²", accel_combined.norm());

    // Compare Sun vs Moon relative magnitude
    let ratio = accel_sun.norm() / accel_moon.norm();
    println!("\nSun/Moon acceleration ratio: {:.3}", ratio);

    // Expected output:
    // Satellite position (ECI, m):
    //   x = 435.7 km
    //   y = -21864.6 km
    //   z = 15074.0 km
    //   Altitude: 20182.7 km

    // Sun third-body acceleration (analytical):
    //   ax = -0.000000011195 m/s²
    //   ay = -0.000000636482 m/s²
    //   az = -0.000001202965 m/s²
    //   Magnitude: 0.000001361014 m/s²

    // Moon third-body acceleration (analytical):
    //   ax = -0.000000403111 m/s²
    //   ay = -0.000000652316 m/s²
    //   az = -0.000002920304 m/s²
    //   Magnitude: 0.000003019303 m/s²

    // Combined Sun + Moon acceleration:
    //   ax = -0.000000414306 m/s²
    //   ay = -0.000001288798 m/s²
    //   az = -0.000004123269 m/s²
    //   Magnitude: 0.000004339815 m/s²

    // Sun/Moon acceleration ratio: 0.451
}
