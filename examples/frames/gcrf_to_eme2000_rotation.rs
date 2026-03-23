//! Get GCRF to EME2000 rotation matrix and use it to transform position vectors

use brahe as bh;
use nalgebra as na;

fn main() {
    // Get constant rotation matrix from GCRF to EME2000
    let r_gcrf_to_eme2000 = bh::rotation_gcrf_to_eme2000();

    println!("GCRF to EME2000 rotation matrix:");
    println!("  [{:13.10}, {:13.10}, {:13.10}]", r_gcrf_to_eme2000[(0, 0)], r_gcrf_to_eme2000[(0, 1)], r_gcrf_to_eme2000[(0, 2)]);
    println!("  [{:13.10}, {:13.10}, {:13.10}]", r_gcrf_to_eme2000[(1, 0)], r_gcrf_to_eme2000[(1, 1)], r_gcrf_to_eme2000[(1, 2)]);
    println!("  [{:13.10}, {:13.10}, {:13.10}]\n", r_gcrf_to_eme2000[(2, 0)], r_gcrf_to_eme2000[(2, 1)], r_gcrf_to_eme2000[(2, 2)]);

    let r_eme2000_to_gcrf = bh::rotation_eme2000_to_gcrf();
    let diff = (r_gcrf_to_eme2000 - r_eme2000_to_gcrf.transpose()).abs();
    let max_diff = diff.max();
    println!("Verification: R_gcrf_to_eme2000 = R_eme2000_to_gcrf^T");
    println!("  Max difference: {:.2e}\n", max_diff);

    let identity = r_gcrf_to_eme2000 * r_gcrf_to_eme2000.transpose();
    let identity_ref = na::Matrix3::<f64>::identity();
    let max_dev = (identity - identity_ref).abs().max();
    println!("Verify orthonormality (R @ R^T should be identity):");
    println!("  Max deviation from identity: {:.2e}\n", max_dev);

    // Define orbital elements for testing transformation
    let oe = na::SVector::<f64, 6>::new(
        bh::R_EARTH + 500e3,  // Semi-major axis (m)
        0.01,                  // Eccentricity
        97.8,                  // Inclination (deg)
        15.0,                  // RAAN (deg)
        30.0,                  // Argument of periapsis (deg)
        45.0,                  // Mean anomaly (deg)
    );

    // Convert to EME2000, transform to GCRF, and extract position
    let state_eme2000 = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let state_gcrf = bh::state_eme2000_to_gcrf(state_eme2000);
    let pos_gcrf = na::Vector3::new(state_gcrf[0], state_gcrf[1], state_gcrf[2]);

    println!("Satellite position in GCRF:");
    println!("  [{:.3}, {:.3}, {:.3}] m\n", pos_gcrf[0], pos_gcrf[1], pos_gcrf[2]);

    // Transform using rotation matrix
    let pos_eme2000_matrix = r_gcrf_to_eme2000 * pos_gcrf;

    println!("Satellite position in EME2000 (using rotation matrix):");
    println!("  [{:.3}, {:.3}, {:.3}] m", pos_eme2000_matrix[0], pos_eme2000_matrix[1], pos_eme2000_matrix[2]);

    let pos_eme2000_direct = bh::position_gcrf_to_eme2000(pos_gcrf);
    println!("\nSatellite position in EME2000 (using position_gcrf_to_eme2000):");
    println!("  [{:.3}, {:.3}, {:.3}] m", pos_eme2000_direct[0], pos_eme2000_direct[1], pos_eme2000_direct[2]);

    let diff = (pos_eme2000_matrix - pos_eme2000_direct).norm();
    println!("\nDifference between methods: {:.6e} m", diff);
    println!("\nNote: Frame bias is constant (same at all epochs)");
}

