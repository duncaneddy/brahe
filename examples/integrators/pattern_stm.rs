//! ```cargo
//! [dependencies]
//! brahe = { path = "../.." }
//! nalgebra = "0.34"
//! ```

//! State Transition Matrix propagation pattern example.
//!
//! Demonstrates the basic pattern for propagating a state transition matrix
//! alongside the state using variational equations.

use brahe::integrators::*;
use brahe::math::jacobian::*;
use nalgebra::{DMatrix, DVector};

fn main() {
    // Dynamics function: Exponential decay dx/dt = -k*x
    let k = 0.1;
    let dynamics = move |_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
        DVector::from_vec(vec![-k * state[0]])
    };
    // Clone for Jacobian computation
    let dynamics_for_jac = move |_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
        DVector::from_vec(vec![-k * state[0]])
    };

    // Create Jacobian for variational equations
    let jacobian = DNumericalJacobian::central(Box::new(dynamics_for_jac))
        .with_adaptive(1e-8, 1e-6);

    // Create integrator with Jacobian
    let config = IntegratorConfig::adaptive(1e-12, 1e-11);
    let integrator = DormandPrince54DIntegrator::with_config(
        1,
        Box::new(dynamics),
        Some(Box::new(jacobian)),
        None,
        None,
        config
    );

    // Propagate state and STM
    let t = 0.0;
    let state = DVector::from_vec(vec![1.0]);
    let phi = DMatrix::identity(1, 1); // 1x1 identity matrix
    let dt = 60.0;

    let result = integrator.step_with_varmat(t, state, phi, Some(dt));
    let new_state = result.state;
    let new_phi = result.phi.unwrap();
    let dt_used = result.dt_used;

    println!("Initial state: {:.6}", 1.0);
    println!("State after {:.2}s: {:.6}", dt_used, new_state[0]);
    println!("State transition matrix:");
    println!("  Φ = {:.6}", new_phi[(0, 0)]);
    println!("  (Analytical Φ = {:.6})", (-0.1 * dt_used).exp());
}
