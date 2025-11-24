//! Demonstrates using analytical Jacobian computation for a simple harmonic oscillator.

#[allow(unused_imports)]
use brahe::math::jacobian::{DAnalyticJacobian, DJacobianProvider};
use nalgebra::{DMatrix, DVector};

fn main() {
    // Define analytical Jacobian function
    // For simple harmonic oscillator: J = [[0, 1], [-1, 0]]
    let jacobian_fn = |_t: f64, _state: DVector<f64>, _params: Option<&DVector<f64>>| -> DMatrix<f64> {
        DMatrix::from_row_slice(2, 2, &[
            0.0,  1.0,
           -1.0,  0.0
        ])
    };

    // Create analytical Jacobian provider
    let jacobian = DAnalyticJacobian::new(Box::new(jacobian_fn));

    // Compute Jacobian at a specific state
    let t = 0.0;
    let state = DVector::from_vec(vec![1.0, 0.0]);
    let jac = jacobian.compute(t, state.clone(), None);

    println!("Analytical Jacobian:");
    println!("{}", jac);
    // Expected output:
    // [[ 0.  1.]
    //  [-1.  0.]]

    // Verify it's time-invariant for this system
    let t2 = 10.0;
    let state2 = DVector::from_vec(vec![0.5, 0.866]);
    let jac2 = jacobian.compute(t2, state2, None);

    println!("\nJacobian at different time and state:");
    println!("{}", jac2);

    // Check if matrices are equal
    let are_equal = (jac - jac2).norm() < 1e-10;
    println!("\nJacobians are equal: {}", are_equal);
    // Output: true
}
