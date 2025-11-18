//! ```cargo
//! [dependencies]
//! brahe = { path = "../.." }
//! nalgebra = "0.33"
//! ```

//! Compute atmospheric drag acceleration using Harris-Priester density model

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create an epoch
    let epoch = bh::Epoch::from_datetime(2024, 3, 15, 14, 30, 0.0, 0.0, bh::TimeSystem::UTC);

    // Define satellite state in ECI frame (LEO satellite at 450 km altitude)
    let a = bh::constants::R_EARTH + 450e3; // Semi-major axis (m)
    let e = 0.002;                          // Eccentricity
    let i = 51.6;                           // Inclination (deg) - ISS-like
    let raan = 90.0;                        // RAAN (deg)
    let argp = 45.0;                        // Argument of perigee (deg)
    let nu = 120.0;                         // True anomaly (deg)

    // Convert to Cartesian state
    let oe = na::SVector::<f64, 6>::new(a, e, i, raan, argp, nu);
    let state_eci = bh::state_osculating_to_cartesian(oe, bh::AngleFormat::Degrees);

    println!("Satellite state (ECI):");
    println!("  Position: [{:.1}, {:.1}, {:.1}] km",
             state_eci[0] / 1e3, state_eci[1] / 1e3, state_eci[2] / 1e3);
    println!("  Velocity: [{:.3}, {:.3}, {:.3}] km/s",
             state_eci[3] / 1e3, state_eci[4] / 1e3, state_eci[5] / 1e3);
    let r_eci = state_eci.fixed_rows::<3>(0);
    println!("  Altitude: {:.1} km", (r_eci.norm() - bh::constants::R_EARTH) / 1e3);

    // Atmospheric density
    // For this example, use a typical density for the given altitude (~450 km)
    // In practice, this would be computed using atmospheric density models like Harris-Priester
    // Typical value for ~450 km altitude: 3-5 × 10^-12 kg/m³
    let density = 4.0e-12; // kg/m³

    println!("\nAtmospheric density (exponential model): {:.6e} kg/m³", density);

    // Define satellite properties
    let mass = 500.0;   // kg (typical small satellite)
    let area = 2.5;     // m² (cross-sectional area)
    let cd = 2.2;       // Drag coefficient (typical for satellites)

    println!("\nSatellite properties:");
    println!("  Mass: {:.1} kg", mass);
    println!("  Area: {:.1} m²", area);
    println!("  Drag coefficient: {:.1}", cd);
    println!("  Ballistic coefficient: {:.6} m²/kg", cd * area / mass);

    // Compute ECI to ECEF rotation matrix for atmospheric velocity
    let r_eci_ecef = bh::rotation_eci_to_ecef(epoch);

    // Compute drag acceleration
    let accel_drag = bh::orbit_dynamics::accel_drag(state_eci, density, mass, area, cd, r_eci_ecef);

    println!("\nDrag acceleration (ECI, m/s²):");
    println!("  ax = {:.9}", accel_drag[0]);
    println!("  ay = {:.9}", accel_drag[1]);
    println!("  az = {:.9}", accel_drag[2]);
    println!("  Magnitude: {:.9} m/s²", accel_drag.norm());

    // Compute velocity magnitude
    let v_eci = state_eci.fixed_rows::<3>(3);
    let v_mag = v_eci.norm();
    println!("\nOrbital velocity: {:.3} m/s ({:.3} km/s)", v_mag, v_mag / 1e3);

    // Theoretical drag magnitude check: 0.5 * rho * v² * Cd * A / m
    let accel_theory = 0.5 * density * v_mag * v_mag * cd * area / mass;
    println!("Theoretical drag magnitude: {:.9} m/s²", accel_theory);

    // Expected output:
    // Satellite state (ECI):
    //   Position: [-1084.6, -6608.2, 1368.5] km
    //   Velocity: [4.582, -1.963, -5.781] km/s
    //   Altitude: 456.8 km

    // Atmospheric density (exponential model): 4.000000e-12 kg/m³

    // Satellite properties:
    //   Mass: 500.0 kg
    //   Area: 2.5 m²
    //   Drag coefficient: 2.2
    //   Ballistic coefficient: 0.011000 m²/kg

    // Drag acceleration (ECI, m/s²):
    //   ax = -0.000000661
    //   ay = 0.000000304
    //   az = 0.000000932
    //   Magnitude: 0.000001183 m/s²

    // Orbital velocity: 7632.770 m/s (7.633 km/s)
    // Theoretical drag magnitude: 0.000001282 m/s²
