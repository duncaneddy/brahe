//! Compares different finite difference methods for numerical Jacobian computation.

#[allow(unused_imports)]
use brahe::math::jacobian::{DNumericalJacobian, DJacobianProvider};
use brahe::constants::GM_EARTH;
use brahe::R_EARTH;
use nalgebra::{DMatrix, DVector};

fn main() {
    // Define two-body gravity dynamics: state = [x, y, z, vx, vy, vz]
    let gravity_dynamics = |_t: f64, state: DVector<f64>| -> DVector<f64> {
        let r = state.fixed_rows::<3>(0);  // Position
        let v = state.fixed_rows::<3>(3);  // Velocity
        let r_norm = r.norm();

        // Acceleration from two-body gravity: a = -mu * r / |r|^3
        let a = -GM_EARTH * r / r_norm.powi(3);

        DVector::from_iterator(6, v.iter().chain(a.iter()).copied())
    };

    // Analytical Jacobian for two-body gravity
    let analytical_jacobian = |state: &DVector<f64>| -> DMatrix<f64> {
        let r = state.fixed_rows::<3>(0);
        let r_norm = r.norm();
        let r3 = r_norm.powi(3);
        let r5 = r_norm.powi(5);

        // Top-left: zeros (3x3)
        // Top-right: identity (3x3)
        // Bottom-left: gravity gradient (3x3)
        // Bottom-right: zeros (3x3)
        let mut jac = DMatrix::zeros(6, 6);

        // Velocity contribution to position derivative
        for i in 0..3 {
            jac[(i, 3 + i)] = 1.0;
        }

        // Gravity gradient term - Montenbruck Eqn 7.56
        let r_vec = r.into_owned();
        let gradient = -GM_EARTH * (DMatrix::identity(3, 3) / r3 - 3.0 * r_vec * r_vec.transpose() / r5);
        jac.view_mut((3, 0), (3, 3)).copy_from(&gradient);

        jac
    };

    // Create numerical Jacobians with different methods
    let jacobian_forward = DNumericalJacobian::forward(Box::new(gravity_dynamics));
    let jacobian_central = DNumericalJacobian::central(Box::new(gravity_dynamics));
    let jacobian_backward = DNumericalJacobian::backward(Box::new(gravity_dynamics));

    // Test state: Low Earth Orbit position and velocity
    let t = 0.0;
    let state = DVector::from_vec(vec![
        R_EARTH + 500e3,  // x position (m)
        0.0,               // y position
        0.0,               // z position
        0.0,               // x velocity
        7500.0,            // y velocity (m/s)
        0.0                // z velocity
    ]);

    // Compute analytical Jacobian
    let j_analytical = analytical_jacobian(&state);

    // Compute Jacobians with each method
    let j_forward = jacobian_forward.compute(t, state.clone());
    let j_central = jacobian_central.compute(t, state.clone());
    let j_backward = jacobian_backward.compute(t, state.clone());

    println!("Forward Difference Jacobian:");
    for i in 0..6 {
        print!("[");
        for j in 0..6 {
            print!("{:>10.2e}", j_forward[(i, j)]);
            if j < 5 { print!("  "); }
        }
        println!("]");
    }
    let error_forward = (j_forward.clone() - j_analytical.clone()).norm();
    println!("Error: {:.2e}\n", error_forward);

    println!("Central Difference Jacobian:");
    for i in 0..6 {
        print!("[");
        for j in 0..6 {
            print!("{:>10.2e}", j_central[(i, j)]);
            if j < 5 { print!("  "); }
        }
        println!("]");
    }
    let error_central = (j_central.clone() - j_analytical.clone()).norm();
    println!("Error: {:.2e}\n", error_central);

    println!("Backward Difference Jacobian:");
    for i in 0..6 {
        print!("[");
        for j in 0..6 {
            print!("{:>10.2e}", j_backward[(i, j)]);
            if j < 5 { print!("  "); }
        }
        println!("]");
    }
    let error_backward = (j_backward - j_analytical).norm();
    println!("Error: {:.2e}\n", error_backward);

    // Summary
    println!("Accuracy Comparison:");
    println!("  Forward:  {:.2e} (O(h))", error_forward);
    println!("  Central:  {:.2e} (O(h²))", error_central);
    println!("  Backward: {:.2e} (O(h))", error_backward);
    println!("\nCentral is {:.1}x more accurate than forward", error_forward / error_central);
    println!("Central is {:.1}x more accurate than backward", error_backward / error_central);
}

// Expected output:
// Forward Difference Jacobian:
// [    0.00e0      0.00e0      0.00e0      1.00e0      0.00e0      0.00e0]
// [    0.00e0      0.00e0      0.00e0      0.00e0      1.00e0      0.00e0]
// [    0.00e0      0.00e0      0.00e0      0.00e0      0.00e0      1.00e0]
// [   2.45e-6      0.00e0      0.00e0      0.00e0      0.00e0      0.00e0]
// [    0.00e0    -1.22e-6      0.00e0      0.00e0      0.00e0      0.00e0]
// [    0.00e0      0.00e0    -1.22e-6      0.00e0      0.00e0      0.00e0]
// Error: 5.54e-14

// Central Difference Jacobian:
// [    0.00e0      0.00e0      0.00e0      1.00e0      0.00e0      0.00e0]
// [    0.00e0      0.00e0      0.00e0      0.00e0      1.00e0      0.00e0]
// [    0.00e0      0.00e0      0.00e0      0.00e0      0.00e0      1.00e0]
// [   2.45e-6      0.00e0      0.00e0      0.00e0      0.00e0      0.00e0]
// [    0.00e0    -1.22e-6      0.00e0      0.00e0      0.00e0      0.00e0]
// [    0.00e0      0.00e0    -1.22e-6      0.00e0      0.00e0      0.00e0]
// Error: 3.40e-15

// Backward Difference Jacobian:
// [    0.00e0      0.00e0      0.00e0      1.00e0      0.00e0      0.00e0]
// [    0.00e0      0.00e0      0.00e0      0.00e0      1.00e0      0.00e0]
// [    0.00e0      0.00e0      0.00e0      0.00e0      0.00e0      1.00e0]
// [   2.45e-6      0.00e0      0.00e0      0.00e0      0.00e0      0.00e0]
// [    0.00e0    -1.22e-6      0.00e0      0.00e0      0.00e0      0.00e0]
// [    0.00e0      0.00e0    -1.22e-6      0.00e0      0.00e0      0.00e0]
// Error: 4.86e-14

// Accuracy Comparison:
//   Forward:  5.54e-14 (O(h))
//   Central:  3.40e-15 (O(h²))
//   Backward: 4.86e-14 (O(h))

// Central is 16.3x more accurate than forward
// Central is 14.3x more accurate than backward