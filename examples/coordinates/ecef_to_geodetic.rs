//! Convert between ECEF Cartesian and geodetic (WGS84 ellipsoid) coordinates

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define a satellite state (convert orbital elements to ECEF state)
    let epc = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let state_oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,  // Semi-major axis (m)
        0.0,                  // Eccentricity
        97.8_f64,             // Inclination (deg)
        15.0_f64,             // Right ascension of ascending node (deg)
        30.0_f64,             // Argument of periapsis (deg)
        45.0_f64              // Mean anomaly (deg)
    );
    let state_eci = bh::state_koe_to_eci(state_oe, bh::AngleFormat::Degrees);
    let state_ecef = bh::state_eci_to_ecef(epc, state_eci);

    println!("ECEF Cartesian state [x, y, z, vx, vy, vz] (m, m/s):");
    println!("Position: [{:.3}, {:.3}, {:.3}]", state_ecef[0], state_ecef[1], state_ecef[2]);
    println!("Velocity: [{:.6}, {:.6}, {:.6}]\n", state_ecef[3], state_ecef[4], state_ecef[5]);
    // Expected output:
    // Position: [-735665.465, -1838913.314, 6586801.432]
    // Velocity: [-1060.370171, 7357.551468, 1935.662061]

    // Convert ECEF Cartesian to geodetic position
    let ecef_pos = na::Vector3::new(state_ecef[0], state_ecef[1], state_ecef[2]);
    let geodetic = bh::position_ecef_to_geodetic(ecef_pos, bh::AngleFormat::Degrees);

    println!("Geodetic coordinates (WGS84 ellipsoid model):");
    println!("Longitude: {:.4}° = {:.6} rad", geodetic[0], geodetic[0].to_radians());
    println!("Latitude:  {:.4}° = {:.6} rad", geodetic[1], geodetic[1].to_radians());
    println!("Altitude:  {:.1} m", geodetic[2]);
    // Expected output:
    // Longitude: -111.8041° = -1.951350 rad
    // Latitude:  73.3622° = 1.280412 rad
    // Altitude:  519618.1 m
}
