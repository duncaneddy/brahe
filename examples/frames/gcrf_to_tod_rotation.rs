//! Get GCRF to TOD rotation matrix and compare it with the CIO-based bias-precession-nutation matrix

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    let epc = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    println!("Epoch: {}", epc);

    let r_gcrf_to_tod = bh::rotation_gcrf_to_tod(epc);

    println!("\nGCRF to TOD rotation matrix:");
    println!("  [{:13.10}, {:13.10}, {:13.10}]", r_gcrf_to_tod[(0, 0)], r_gcrf_to_tod[(0, 1)], r_gcrf_to_tod[(0, 2)]);
    println!("  [{:13.10}, {:13.10}, {:13.10}]", r_gcrf_to_tod[(1, 0)], r_gcrf_to_tod[(1, 1)], r_gcrf_to_tod[(1, 2)]);
    println!("  [{:13.10}, {:13.10}, {:13.10}]\n", r_gcrf_to_tod[(2, 0)], r_gcrf_to_tod[(2, 1)], r_gcrf_to_tod[(2, 2)]);

    let identity = r_gcrf_to_tod * r_gcrf_to_tod.transpose();
    let identity_ref = na::Matrix3::<f64>::identity();
    let max_dev = (identity - identity_ref).abs().max();
    println!("Verify orthonormality (R @ R^T should be identity):");
    println!("  Max deviation from identity: {:.2e}\n", max_dev);

    let r_cio = bh::bias_precession_nutation(epc);
    let max_diff = (r_gcrf_to_tod - r_cio).abs().max();
    println!("Comparison with the CIO-based bias-precession-nutation matrix:");
    println!("  Max element difference: {:.6e}", max_diff);

    println!("\nNote: rotation_gcrf_to_tod and bias_precession_nutation share the same");
    println!("Celestial Intermediate Pole direction (third row) and differ only by the");
    println!("equation of the origins, a rotation within the equatorial plane.");
}
