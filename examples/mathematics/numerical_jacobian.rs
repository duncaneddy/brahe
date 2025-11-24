//! Demonstrates numerical Jacobian computation using finite differences.

#[allow(unused_imports)]
use brahe::math::jacobian::{DNumericalJacobian, DJacobianProvider};
use nalgebra::{DMatrix, DVector};

fn main() {
    // Define dynamics function
    let dynamics = |_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
        DVector::from_vec(vec![state[1], -state[0]])
    };

    // Create numerical Jacobian with default settings (central differences)
    let jacobian = DNumericalJacobian::central(Box::new(dynamics));

    // Compute Jacobian at a specific state
    let t = 0.0;
    let state = DVector::from_vec(vec![1.0, 0.0]);
    let jac_numerical = jacobian.compute(t, state.clone(), None);

    println!("Numerical Jacobian (central differences):");
    println!("{}", jac_numerical);
    // Expected output (should be very close to analytical):
    // [[ 0.  1.]
    //  [-1.  0.]]

    // Compare with analytical solution
    let jac_analytical = DMatrix::from_row_slice(2, 2, &[
        0.0,  1.0,
       -1.0,  0.0
    ]);

    let error = (jac_numerical.clone() - jac_analytical.clone()).norm();
    println!("\nError vs analytical: {:.2e}", error);
    // Output: ~1e-8 (machine precision)

    // Verify accuracy at different state
    let state2 = DVector::from_vec(vec![0.5, 0.866]);
    let jac_numerical2 = jacobian.compute(t, state2, None);

    println!("\nNumerical Jacobian at different state:");
    println!("{}", jac_numerical2);

    let error2 = (jac_numerical2 - jac_analytical).norm();
    println!("Error vs analytical: {:.2e}", error2);
    // Output: ~1e-8 (consistent accuracy)
}
