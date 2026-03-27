//! Demonstrate the different ways to specify measurement noise covariance.

#[allow(unused_imports)]
use brahe as bh;
use nalgebra::DMatrix;

fn main() {
    // --- Scalar sigma: same noise on all axes ---
    let model = bh::estimation::EcefPositionMeasurementModel::new(5.0);
    let r = model.noise_covariance();
    println!("Scalar (5 m isotropic):");
    println!("  diag = [{:.1}, {:.1}, {:.1}]", r[(0,0)], r[(1,1)], r[(2,2)]);

    // --- Per-axis sigma: different noise per component ---
    let model = bh::estimation::EcefPositionMeasurementModel::new_per_axis(3.0, 3.0, 8.0);
    let r = model.noise_covariance();
    println!("\nPer-axis (3, 3, 8 m):");
    println!("  diag = [{:.1}, {:.1}, {:.1}]", r[(0,0)], r[(1,1)], r[(2,2)]);

    // --- Full covariance: captures cross-axis correlations ---
    let mut cov = DMatrix::zeros(3, 3);
    cov[(0,0)] = 9.0; cov[(0,1)] = 1.0;
    cov[(1,0)] = 1.0; cov[(1,1)] = 9.0;
    cov[(2,2)] = 64.0;
    let model = bh::estimation::EcefPositionMeasurementModel::from_covariance(cov).unwrap();
    let r = model.noise_covariance();
    println!("\nFull covariance (with correlations):");
    println!("  (0,0)={:.1} (0,1)={:.1} (2,2)={:.1}", r[(0,0)], r[(0,1)], r[(2,2)]);

    // --- Upper-triangular: compact packed form ---
    // Elements: [c00, c01, c02, c11, c12, c22]
    let model = bh::estimation::EcefPositionMeasurementModel::from_upper_triangular(
        &[9.0, 1.0, 0.0, 9.0, 0.0, 64.0]
    ).unwrap();
    let r = model.noise_covariance();
    println!("\nUpper-triangular packed:");
    println!("  (0,0)={:.1} (0,1)={:.1} (2,2)={:.1}", r[(0,0)], r[(0,1)], r[(2,2)]);

    // --- Standalone covariance helpers ---
    let r = bh::math::covariance::isotropic_covariance(3, 10.0);
    println!("\nisotropic_covariance(3, 10.0) diag: [{:.0}, {:.0}, {:.0}]",
        r[(0,0)], r[(1,1)], r[(2,2)]);

    let r = bh::math::covariance::diagonal_covariance(&[5.0, 10.0, 15.0]);
    println!("diagonal_covariance([5, 10, 15]) diag: [{:.0}, {:.0}, {:.0}]",
        r[(0,0)], r[(1,1)], r[(2,2)]);
}
