//! ```cargo
//! [dependencies]
//! brahe = { path = "../.." }
//! nalgebra = "0.34"
//! ```
//!
//! Simple adaptive-step integration pattern example.
//!
//! Demonstrates the basic pattern for using an adaptive-step integrator
//! with exponential decay dynamics.

use brahe::integrators::*;
use nalgebra::DVector;

fn main() {
    // Dynamics function: Exponential decay dx/dt = -k*x
    let k = 0.1;
    let dynamics = move |_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
        DVector::from_vec(vec![-k * state[0]])
    };

    // Create adaptive integrator
    let config = IntegratorConfig::adaptive(1e-10, 1e-9);
    let integrator = DormandPrince54DIntegrator::with_config(
        1,
        Box::new(dynamics),
        None,
        None,
        None,
        config
    );

    // Integrate with automatic step control
    let t = 0.0;
    let initial_state = DVector::from_vec(vec![1.0]);
    let dt = 60.0; // Initial guess

    let result = integrator.step(t, initial_state.clone(), Some(dt));

    println!("Initial state: {:.6}", initial_state[0]);
    println!("State after step: {:.6}", result.state[0]);
    println!("Step used: {:.2}s", result.dt_used);
    println!("Recommended next step: {:.2}s", result.dt_next);
    println!("Error estimate: {:.2e}", result.error_estimate.unwrap());
}
