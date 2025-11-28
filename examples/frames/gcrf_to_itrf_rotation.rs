//! Get GCRF to ITRF rotation matrix and use it to transform position vectors

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define epoch
    let epc = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let epc = epc + 12.0 * 3600.0;  // Add 12 hours

    // Get rotation matrix from GCRF to ITRF
    let r_gcrf_to_itrf = bh::rotation_gcrf_to_itrf(epc);

    println!("Epoch: {}", epc); // Epoch: 2024-01-01 12:00:00 UTC
    println!("\nGCRF to ITRF rotation matrix:");
    println!("  [{:10.7}, {:10.7}, {:10.7}]", r_gcrf_to_itrf[(0, 0)], r_gcrf_to_itrf[(0, 1)], r_gcrf_to_itrf[(0, 2)]);
    println!("  [{:10.7}, {:10.7}, {:10.7}]", r_gcrf_to_itrf[(1, 0)], r_gcrf_to_itrf[(1, 1)], r_gcrf_to_itrf[(1, 2)]);
    println!("  [{:10.7}, {:10.7}, {:10.7}]\n", r_gcrf_to_itrf[(2, 0)], r_gcrf_to_itrf[(2, 1)], r_gcrf_to_itrf[(2, 2)]);
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

    // Convert to GCRF Cartesian state and extract position
    let state_gcrf = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let pos_gcrf = na::Vector3::new(state_gcrf[0], state_gcrf[1], state_gcrf[2]);

    println!("Position in GCRF:");
    println!("  [{:.3}, {:.3}, {:.3}] m\n", pos_gcrf[0], pos_gcrf[1], pos_gcrf[2]);
    // [1848964.106, -434937.468, 6560410.530] m

    // Transform position using rotation matrix
    let pos_itrf = r_gcrf_to_itrf * pos_gcrf;

    println!("Position in ITRF (using rotation matrix):");
    println!("  [{:.3}, {:.3}, {:.3}] m", pos_itrf[0], pos_itrf[1], pos_itrf[2]);
    // [757164.267, 1725863.563, 6564672.302] m

    // Verify using position transformation function
    let pos_itrf_direct = bh::position_gcrf_to_itrf(epc, pos_gcrf);
    println!("\nPosition in ITRF (using position_gcrf_to_itrf):");
    println!("  [{:.3}, {:.3}, {:.3}] m", pos_itrf_direct[0], pos_itrf_direct[1], pos_itrf_direct[2]);
    // [757164.267, 1725863.563, 6564672.302] m
}
