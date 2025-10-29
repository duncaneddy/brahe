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

    // Apply polar motion to get ITRF (ECEF)
    let pos_itrf = r_pm * pos_tirs;

    println!("Satellite position in ITRF (ECEF):");
    println!("  [{:.3}, {:.3}, {:.3}] m", pos_itrf[0], pos_itrf[1], pos_itrf[2]);

    // Calculate the magnitude of the change
    let diff = (pos_tirs - pos_itrf).norm();
    println!("\nPosition change magnitude: {:.3} m", diff);
    println!("Note: Polar motion effects are typically centimeters to meters");

    // Verify against full transformation
    let pos_ecef_direct = bh::position_eci_to_ecef(epc, pos_gcrf);
    println!("\nVerification using position_eci_to_ecef:");
    println!("  [{:.3}, {:.3}, {:.3}] m", pos_ecef_direct[0], pos_ecef_direct[1], pos_ecef_direct[2]);
    let max_diff = (pos_itrf - pos_ecef_direct).abs().max();
    println!("  Max difference: {:.2e} m", max_diff);

    // Expected output:
    // Position in ITRF: [3210319.128, 5246384.459, 2649959.679] m
    // Position change magnitude: ~0.3 m
    // Max difference: ~1e-9 m (numerical precision)
}
