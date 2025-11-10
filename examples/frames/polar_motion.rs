//! Get the Polar Motion matrix

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define epoch
    let epc = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let epc = epc + 12.0 * 3600.0;  // Add 12 hours

    // Get polar motion matrix (TIRS to ITRF transformation)
    let r_pm = bh::polar_motion(epc);

    println!("Epoch: 2024-01-01 12:00:00 UTC");
    println!("\nPolar Motion matrix:");
    println!("Transforms from TIRS to ITRF");
    println!("  [{:10.7}, {:10.7}, {:10.7}]", r_pm[(0, 0)], r_pm[(0, 1)], r_pm[(0, 2)]);
    println!("  [{:10.7}, {:10.7}, {:10.7}]", r_pm[(1, 0)], r_pm[(1, 1)], r_pm[(1, 2)]);
    println!("  [{:10.7}, {:10.7}, {:10.7}]\n", r_pm[(2, 0)], r_pm[(2, 1)], r_pm[(2, 2)]);
    // [ 1.0000000, -0.0000000,  0.0000007]
    // [ 0.0000000,  1.0000000, -0.0000010]
    // [-0.0000007,  0.0000010,  1.0000000]

    // Define orbital elements in degrees
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,  // Semi-major axis (m)
        0.01,                  // Eccentricity
        97.8,                  // Inclination (deg)
        15.0,                  // RAAN (deg)
        30.0,                  // Argument of periapsis (deg)
        45.0,                  // Mean anomaly (deg)
    );

    // Convert through the full chain: GCRF → CIRS → TIRS
    let state_gcrf = bh::state_osculating_to_cartesian(oe, bh::AngleFormat::Degrees);
    let pos_gcrf = na::Vector3::new(state_gcrf[0], state_gcrf[1], state_gcrf[2]);
    let r_bpn = bh::bias_precession_nutation(epc);
    let r_er = bh::earth_rotation(epc);
    let pos_tirs = r_er * r_bpn * pos_gcrf;

    println!("Satellite position in TIRS:");
    println!("  [{:.3}, {:.3}, {:.3}] m\n", pos_tirs[0], pos_tirs[1], pos_tirs[2]);
    // [757159.942, 1725870.003, 6564671.107] m

    // Apply polar motion to get ITRF
    let pos_itrf = r_pm * pos_tirs;

    println!("Satellite position in ITRF:");
    println!("  [{:.3}, {:.3}, {:.3}] m", pos_itrf[0], pos_itrf[1], pos_itrf[2]);
    // [757164.267, 1725863.563, 6564672.302] m

    // Calculate the magnitude of the change
    let diff = (pos_tirs - pos_itrf).norm();
    println!("\nPosition change magnitude: {:.3} m", diff);
    println!("Note: Polar motion effects are typically centimeters to meters");
    // Position change magnitude: 7.849 m

    // Verify against full transformation
    let pos_itrf_direct = bh::position_gcrf_to_itrf(epc, pos_gcrf);
    println!("\nVerification using position_gcrf_to_itrf:");
    println!("  [{:.3}, {:.3}, {:.3}] m", pos_itrf_direct[0], pos_itrf_direct[1], pos_itrf_direct[2]);
    let max_diff = (pos_itrf - pos_itrf_direct).abs().max();
    println!("  Max difference: {:.2e} m", max_diff);
    // [757164.267, 1725863.563, 6564672.302] m
    // Max difference: 1.16e-10 m
}
