//! Compares different perturbation strategies for numerical Jacobian computation.

#[allow(unused_imports)]
use brahe::math::jacobian::{DNumericalJacobian, DJacobianProvider};
use nalgebra::{DMatrix, DVector};

fn main() {
    // Define dynamics with mixed-scale state
    let dynamics = |_t: f64, state: DVector<f64>| -> DVector<f64> {
        let x = state[0];
        let v = state[1];
        DVector::from_vec(vec![v, -x * 1e-6])
    };

    // Analytical Jacobian
    let analytical_jacobian = || -> DMatrix<f64> {
        DMatrix::from_row_slice(2, 2, &[
            0.0, 1.0,
            -1e-6, 0.0
        ])
    };

    // Test state with very different magnitudes
    let state = DVector::from_vec(vec![7000.0, 7.5]); // Position in km, velocity in km/s
    let t = 0.0;

    let j_analytical = analytical_jacobian();

    println!("Testing perturbation strategies on mixed-scale state:");
    println!("State: position={} km, velocity={} km/s\n", state[0], state[1]);

    // Strategy 1: Fixed perturbation
    println!("1. Fixed Perturbation (h = 1e-6)");
    let jacobian_fixed = DNumericalJacobian::central(Box::new(dynamics))
        .with_fixed_offset(1e-6);
    let j_fixed = jacobian_fixed.compute(t, state.clone());
    let error_fixed = (j_fixed - j_analytical.clone()).norm();
    println!("   Error: {:.2e}\n", error_fixed);

    // Strategy 2: Percentage perturbation
    println!("2. Percentage Perturbation (0.001%)");
    let jacobian_pct = DNumericalJacobian::central(Box::new(dynamics))
        .with_percentage(1e-5);
    let j_pct = jacobian_pct.compute(t, state.clone());
    let error_pct = (j_pct - j_analytical.clone()).norm();
    println!("   Error: {:.2e}\n", error_pct);

    // Strategy 3: Adaptive perturbation (recommended)
    println!("3. Adaptive Perturbation (scale=1.0, min=1.0)");
    let jacobian_adaptive = DNumericalJacobian::central(Box::new(dynamics))
        .with_adaptive(1.0, 1.0);
    let j_adaptive = jacobian_adaptive.compute(t, state.clone());
    let error_adaptive = (j_adaptive - j_analytical.clone()).norm();
    println!("   Error: {:.2e}\n", error_adaptive);

    // Summary
    println!("Strategy Comparison:");
    println!("  Fixed:      {:.2e}", error_fixed);
    println!("  Percentage: {:.2e}", error_pct);
    println!("  Adaptive:   {:.2e}", error_adaptive);
    println!("\nBest strategy: Adaptive (handles mixed scales robustly)");

    // Test with state component near zero
    println!("\n{}", "=".repeat(60));
    println!("Testing with component near zero:");
    let state_zero = DVector::from_vec(vec![7000.0, 1e-9]); // Very small velocity
    println!("State: position={} km, velocity={} km/s\n", state_zero[0], state_zero[1]);

    let j_analytical_zero = analytical_jacobian();

    // Percentage can be problematic when component is near zero
    let j_pct_zero = jacobian_pct.compute(t, state_zero.clone());
    let error_pct_zero = (j_pct_zero - j_analytical_zero.clone()).norm();
    println!("Percentage: Error = {:.2e}", error_pct_zero);

    // Adaptive handles it gracefully
    let j_adaptive_zero = jacobian_adaptive.compute(t, state_zero);
    let error_adaptive_zero = (j_adaptive_zero - j_analytical_zero).norm();
    println!("Adaptive:   Error = {:.2e}", error_adaptive_zero);

    println!("\nConclusion: Adaptive perturbation is most robust");
}
