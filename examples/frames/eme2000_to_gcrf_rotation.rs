//! Get EME2000 to GCRF rotation matrix and use it to transform position vectors

use brahe as bh;
use nalgebra as na;

fn main() {
    // Get constant rotation matrix from EME2000 to GCRF
    let r_eme2000_to_gcrf = bh::rotation_eme2000_to_gcrf();

    println!("EME2000 to GCRF rotation matrix:");
    println!("  [{:13.10}, {:13.10}, {:13.10}]", r_eme2000_to_gcrf[(0, 0)], r_eme2000_to_gcrf[(0, 1)], r_eme2000_to_gcrf[(0, 2)]);
    println!("  [{:13.10}, {:13.10}, {:13.10}]", r_eme2000_to_gcrf[(1, 0)], r_eme2000_to_gcrf[(1, 1)], r_eme2000_to_gcrf[(1, 2)]);
    println!("  [{:13.10}, {:13.10}, {:13.10}]\n", r_eme2000_to_gcrf[(2, 0)], r_eme2000_to_gcrf[(2, 1)], r_eme2000_to_gcrf[(2, 2)]);

    let identity = r_eme2000_to_gcrf * r_eme2000_to_gcrf.transpose();
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

    // Convert to EME2000 Cartesian state and extract position
    let state_eme2000 = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);
    let pos_eme2000 = na::Vector3::new(state_eme2000[0], state_eme2000[1], state_eme2000[2]);

    println!("Satellite position in EME2000:");
    println!("  [{:.3}, {:.3}, {:.3}] m\n", pos_eme2000[0], pos_eme2000[1], pos_eme2000[2]);

    // Transform using rotation matrix
    let pos_gcrf_matrix = r_eme2000_to_gcrf * pos_eme2000;

    println!("Satellite position in GCRF (using rotation matrix):");
    println!("  [{:.3}, {:.3}, {:.3}] m", pos_gcrf_matrix[0], pos_gcrf_matrix[1], pos_gcrf_matrix[2]);

    let pos_gcrf_direct = bh::position_eme2000_to_gcrf(pos_eme2000);
    println!("\nSatellite position in GCRF (using position_eme2000_to_gcrf):");
    println!("  [{:.3}, {:.3}, {:.3}] m", pos_gcrf_direct[0], pos_gcrf_direct[1], pos_gcrf_direct[2]);

    let diff = (pos_gcrf_matrix - pos_gcrf_direct).norm();
    println!("\nDifference between methods: {:.6e} m", diff);
    println!("\nNote: Frame bias is constant (same at all epochs)");
}

