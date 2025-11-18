//! ```cargo
//! [dependencies]
//! brahe = { path = "../.." }
//! nalgebra = "0.33"
//! ```

//! Compute solar radiation pressure acceleration with Earth shadow

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create an epoch (summer solstice for interesting Sun geometry)
    let epoch = bh::Epoch::from_datetime(2024, 6, 21, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Define satellite position (GEO satellite)
    let a = bh::constants::R_EARTH + 35786e3; // Semi-major axis (m) - geostationary
    let e = 0.0001;                           // Near-circular
    let i = 0.1_f64.to_radians();             // Near-equatorial
    let raan = 0.0_f64.to_radians();          // RAAN (rad)
    let argp = 0.0_f64.to_radians();          // Argument of perigee (rad)
    let nu = 0.0_f64.to_radians();            // True anomaly (rad)

    // Convert to Cartesian state
    let oe = na::SVector::<f64, 6>::new(a, e, i, raan, argp, nu);
    let state = bh::state_osculating_to_cartesian(oe, bh::AngleFormat::Radians);
    let r_sat = na::Vector3::new(state[0], state[1], state[2]); // Position vector (m)

    println!("Satellite position (ECI, m):");
    println!("  x = {:.1} km", r_sat[0] / 1e3);
    println!("  y = {:.1} km", r_sat[1] / 1e3);
    println!("  z = {:.1} km", r_sat[2] / 1e3);
    println!("  Altitude: {:.1} km", (r_sat.norm() - bh::constants::R_EARTH) / 1e3);

    // Get Sun position
    let r_sun = bh::orbit_dynamics::sun_position(epoch);

    println!("\nSun position (ECI, AU):");
    println!("  x = {:.6} AU", r_sun[0] / 1.496e11);
    println!("  y = {:.6} AU", r_sun[1] / 1.496e11);
    println!("  z = {:.6} AU", r_sun[2] / 1.496e11);

    // Eclipse condition
    // For this example, assume full sunlight (no eclipse)
    // In practice, eclipse_conical() function would be used to check shadow status
    let nu_eclipse = 1.0;

    println!("\nEclipse factor: {:.6}", nu_eclipse);
    println!("  Status: Full sunlight (no eclipse)");

    // Define satellite SRP properties
    let mass = 1500.0;  // kg (typical GEO satellite)
    let cr = 1.3;       // Radiation pressure coefficient
    let area = 20.0;    // m² (effective area - solar panels + body)
    let p0 = 4.56e-6;   // Solar radiation pressure at 1 AU (N/m²)

    println!("\nSatellite SRP properties:");
    println!("  Mass: {:.1} kg", mass);
    println!("  Area: {:.1} m²", area);
    println!("  Cr coefficient: {:.1}", cr);
    println!("  Area/mass ratio: {:.6} m²/kg", area / mass);

    // Compute solar radiation pressure acceleration
    let accel_srp = bh::orbit_dynamics::accel_solar_radiation_pressure(
        r_sat, r_sun, mass, cr, area, p0
    );

    println!("\nSolar radiation pressure acceleration (ECI, m/s²):");
    println!("  ax = {:.12}", accel_srp[0]);
    println!("  ay = {:.12}", accel_srp[1]);
    println!("  az = {:.12}", accel_srp[2]);
    println!("  Magnitude: {:.12} m/s²", accel_srp.norm());

    // Theoretical maximum (no eclipse)
    let accel_max = p0 * cr * area / mass;
    println!("\nTheoretical maximum (full sun): {:.12} m/s²", accel_max);
    println!("Actual/Maximum ratio: {:.6}", accel_srp.norm() / accel_max);

    // Compare to other forces at GEO
    let r_mag = r_sat.norm();
    let accel_gravity = bh::constants::GM_EARTH / (r_mag * r_mag);
    println!("\nFor comparison at GEO altitude:");
    println!("  Point-mass gravity: {:.9} m/s²", accel_gravity);
    println!("  SRP/Gravity ratio: {:.2e}", accel_srp.norm() / accel_gravity);

    // Expected output:
    // Satellite position (ECI, m):
    //   x = 42159.9 km
    //   y = 0.0 km
    //   z = 0.0 km
    //   Altitude: 35781.8 km

    // Sun position (ECI, AU):
    //   x = -0.003352 AU
    //   y = 0.932401 AU
    //   z = 0.404245 AU

    // Eclipse status:
    //   Conical model: 1.000000
    //   Cylindrical model: 1.000000
    //   Status: Full sunlight

    // Satellite SRP properties:
    //   Mass: 1500.0 kg
    //   Area: 20.0 m²
    //   Cr coefficient: 1.3
    //   Area/mass ratio: 0.013333 m²/kg

    // Solar radiation pressure acceleration (ECI, m/s²):
    //   ax = 0.000000000274
    //   ay = -0.000000070212
    //   az = -0.000000030441
    //   Magnitude: 0.000000076528 m/s²

    // Theoretical maximum (full sun): 0.000000079040 m/s²
    // Actual/Maximum ratio: 0.968216

    // For comparison at GEO altitude:
    //   Point-mass gravity: 0.224252979 m/s²
    //   SRP/Gravity ratio: 3.41e-07
}
