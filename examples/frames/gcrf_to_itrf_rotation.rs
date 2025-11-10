//! Get ECI to ECEF rotation matrix and use it to transform position vectors

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define epoch
    let epc = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let epc = epc + 12.0 * 3600.0;  // Add 12 hours

    // Get rotation matrix from ECI to ECEF
    let r_eci_to_ecef = bh::rotation_eci_to_ecef(epc);

    println!("Epoch: {}", epc); // Epoch: 2024-01-01 12:00:00 UTC
    println!("\nECI to ECEF rotation matrix:");
    println!("  [{:10.7}, {:10.7}, {:10.7}]", r_eci_to_ecef[(0, 0)], r_eci_to_ecef[(0, 1)], r_eci_to_ecef[(0, 2)]);
    println!("  [{:10.7}, {:10.7}, {:10.7}]", r_eci_to_ecef[(1, 0)], r_eci_to_ecef[(1, 1)], r_eci_to_ecef[(1, 2)]);
    println!("  [{:10.7}, {:10.7}, {:10.7}]\n", r_eci_to_ecef[(2, 0)], r_eci_to_ecef[(2, 1)], r_eci_to_ecef[(2, 2)]);
    // [ 0.1794538, -0.9837663, -0.0003836]
    // [ 0.9837637,  0.1794542, -0.0022908]
    // [ 0.0023225,  0.0000338,  0.9999973]

    // Define orbital elements in degrees for satellite position
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,  // Semi-major axis (m)
        0.01,                  // Eccentricity
        97.8,                  // Inclination (deg)
        15.0,                  // RAAN (deg)
        30.0,                  // Argument of periapsis (deg)
        45.0,                  // Mean anomaly (deg)
    );

    // Convert to ECI Cartesian state and extract position
    let state_eci = bh::state_osculating_to_cartesian(oe, bh::AngleFormat::Degrees);
    let pos_eci = na::Vector3::new(state_eci[0], state_eci[1], state_eci[2]);

    println!("Position in ECI:");
    println!("  [{:.3}, {:.3}, {:.3}] m\n", pos_eci[0], pos_eci[1], pos_eci[2]);
    // [1848964.106, -434937.468, 6560410.530] m

    // Transform position using rotation matrix
    let pos_ecef = r_eci_to_ecef * pos_eci;

    println!("Position in ECEF (using rotation matrix):");
    println!("  [{:.3}, {:.3}, {:.3}] m", pos_ecef[0], pos_ecef[1], pos_ecef[2]);
    // [757164.267, 1725863.563, 6564672.302] m

    // Verify using position transformation function
    let pos_ecef_direct = bh::position_eci_to_ecef(epc, pos_eci);
    println!("\nPosition in ECEF (using position_eci_to_ecef):");
    println!("  [{:.3}, {:.3}, {:.3}] m", pos_ecef_direct[0], pos_ecef_direct[1], pos_ecef_direct[2]);
    // [757164.267, 1725863.563, 6564672.302] m
}
