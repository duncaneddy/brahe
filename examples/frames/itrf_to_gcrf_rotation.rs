//! Get ITRF to GCRF rotation matrix and use it to transform position vectors

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define epoch
    let epc = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let epc = epc + 12.0 * 3600.0;  // Add 12 hours

    // Get rotation matrix from ITRF to GCRF
    let r_itrf_to_gcrf = bh::rotation_itrf_to_gcrf(epc);

    println!("Epoch: 2024-01-01 12:00:00 UTC");
    println!("\nITRF to GCRF rotation matrix:");
    println!("  [{:10.7}, {:10.7}, {:10.7}]", r_itrf_to_gcrf[(0, 0)], r_itrf_to_gcrf[(0, 1)], r_itrf_to_gcrf[(0, 2)]);
    println!("  [{:10.7}, {:10.7}, {:10.7}]", r_itrf_to_gcrf[(1, 0)], r_itrf_to_gcrf[(1, 1)], r_itrf_to_gcrf[(1, 2)]);
    println!("  [{:10.7}, {:10.7}, {:10.7}]\n", r_itrf_to_gcrf[(2, 0)], r_itrf_to_gcrf[(2, 1)], r_itrf_to_gcrf[(2, 2)]);
    // [ 0.1794538,  0.9837637,  0.0023225]
    // [-0.9837663,  0.1794542,  0.0000338]
    // [-0.0003836, -0.0022908,  0.9999973]

    // Verify it's the transpose of GCRF to ITRF rotation
    let r_gcrf_to_itrf = bh::rotation_gcrf_to_itrf(epc);
    let diff = (r_itrf_to_gcrf - r_gcrf_to_itrf.transpose()).abs();
    let max_diff = diff.max();
    println!("Verification: R_itrf_to_gcrf = R_gcrf_to_itrf^T");
    println!("  Max difference: {:.2e}\n", max_diff);
    // Max difference: 0.00e+00

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
    let pos_gcrf_orig = na::Vector3::new(state_gcrf[0], state_gcrf[1], state_gcrf[2]);

    // Transform to ITRF
    let pos_itrf = bh::position_gcrf_to_itrf(epc, pos_gcrf_orig);

    println!("Satellite position in ITRF:");
    println!("  [{:.3}, {:.3}, {:.3}] m\n", pos_itrf[0], pos_itrf[1], pos_itrf[2]);
    // [757164.267, 1725863.563, 6564672.302] m

    // Transform back to GCRF using rotation matrix
    let pos_gcrf = r_itrf_to_gcrf * pos_itrf;

    println!("Satellite position in GCRF (using rotation matrix):");
    println!("  [{:.3}, {:.3}, {:.3}] m", pos_gcrf[0], pos_gcrf[1], pos_gcrf[2]);
    // [1848964.106, -434937.468, 6560410.530] m

    // Verify using position transformation function
    let pos_gcrf_direct = bh::position_itrf_to_gcrf(epc, pos_itrf);
    println!("\nSatellite position in GCRF (using position_itrf_to_gcrf):");
    println!("  [{:.3}, {:.3}, {:.3}] m", pos_gcrf_direct[0], pos_gcrf_direct[1], pos_gcrf_direct[2]);
    // [1848964.106, -434937.468, 6560410.530] m
}
