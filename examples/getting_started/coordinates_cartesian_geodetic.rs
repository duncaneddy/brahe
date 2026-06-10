use brahe as bh;
use nalgebra as na;

fn main() {
    // Initialize EOP
    bh::initialize_eop().unwrap();

    // Create an epoch
    let epc = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Initialize a Keplerian state
    // Define orbital elements [a, e, i, Ω, ω, M] in meters and degrees
    // LEO satellite: 500 km altitude, 97.8° inclination (approx sun-synchronous)
    let oe_deg = na::vector![
        bh::R_EARTH + 500e3,  // Semi-major axis (m)
        0.01,                  // Eccentricity
        97.8,                  // Inclination (deg)
        15.0,                  // Right ascension of ascending node (deg)
        30.0,                  // Argument of periapsis (deg)
        45.0,                  // Mean anomaly (deg)
    ];

    // Convert orbital elements to Cartesian state using degrees
    let x_deg = bh::state_koe_to_eci(oe_deg, bh::AngleFormat::Degrees);

    // Convert ECI cartesian state to ECEF cartesian state
    let x_ecef = bh::state_eci_to_ecef(epc, x_deg);

    // Convert ECEF cartesian state to geodetic coordinates
    let geodetic = bh::position_ecef_to_geodetic(x_ecef.fixed_rows::<3>(0).into(), bh::AngleFormat::Degrees);
    println!("Geodetic coordinates (lat, lon, alt): {:.6}°, {:.6}°, {:.3} km", geodetic[0], geodetic[1], geodetic[2] / 1e3);
}

