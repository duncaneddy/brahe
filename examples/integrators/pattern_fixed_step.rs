//! ```cargo
//! [dependencies]
//! brahe = { path = "../.." }
//! nalgebra = "0.34"
//! ```

//! Simple fixed-step integration pattern example.
//!
//! Demonstrates the basic pattern for using a fixed-step integrator
//! with exponential decay dynamics.

use brahe::integrators::*;
use nalgebra::DVector;

fn main() {
    // Dynamics function: Exponential decay dx/dt = -k*x
    let k = 0.1;
    let dynamics = move |_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
        DVector::from_vec(vec![-k * state[0]])
    };

    // --8<-- [start:snippet]
    // Create fixed-step integrator
    let config = IntegratorConfig::fixed_step(10.0);
    let integrator = RK4DIntegrator::with_config(
        1,
        Box::new(dynamics),
        None,
        None,
        None,
        config
    );

    // Integrate one step
    let t = 0.0;
    let initial_state = DVector::from_vec(vec![1.0]);
    let result = integrator.step(t, initial_state.clone(), None);
    let new_state = result.state;
    // --8<-- [end:snippet]

    println!("Initial state: {:.6}", initial_state[0]);
    println!("State after 10s: {:.6}", new_state[0]);
    println!("Analytical: {:.6}", initial_state[0] * (-0.1 * 10.0_f64).exp());
}
