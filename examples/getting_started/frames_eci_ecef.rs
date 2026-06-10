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
    let x_eci_1 = bh::state_koe_to_eci(oe_deg, bh::AngleFormat::Degrees);

    // Convert ECI cartesian state to ECEF cartesian state
    let x_ecef = bh::state_eci_to_ecef(epc, x_eci_1);

    // Convert ECEF back to ECI to verify consistency
    let x_eci_2 = bh::state_ecef_to_eci(epc, x_ecef);
    let eci_roundtrip_diff = (x_eci_2 - x_eci_1).norm();
    println!("ECI -> ECEF -> ECI roundtrip difference: {:.3e}", eci_roundtrip_diff);

    // Perform same transformation with GCRF/ITRF naming
    let x_gcrf_1 = x_eci_1;
    let x_itrf = bh::state_gcrf_to_itrf(epc, x_gcrf_1);
    let x_gcrf_2 = bh::state_itrf_to_gcrf(epc, x_itrf);
    let gcrf_roundtrip_diff = (x_gcrf_2 - x_gcrf_1).norm();
    println!("GCRF -> ITRF -> GCRF roundtrip difference: {:.3e}", gcrf_roundtrip_diff);

    let ecef_itrf_diff = (x_ecef - x_itrf).norm();
    println!("ECEF <> ITRF difference: {:.3e}", ecef_itrf_diff);
}

