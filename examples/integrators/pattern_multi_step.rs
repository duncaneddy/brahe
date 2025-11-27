//! ```cargo
//! [dependencies]
//! brahe = { path = "../.." }
//! nalgebra = "0.34"
//! ```

//! Multi-step propagation pattern example.
//!
//! Demonstrates the pattern for propagating over an extended time period
//! using an adaptive integrator, using the recommended step size from
//! each step for the next step.

use brahe::integrators::*;
use nalgebra::DVector;

fn main() {
    // Dynamics function: Exponential decay dx/dt = -k*x
    let k = 0.1;
    let dynamics = move |_t: f64, state: &DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
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

    // Propagate from t=0 to t_end
    let mut t = 0.0;
    let t_end = 86400.0; // One day
    let mut state = DVector::from_vec(vec![1.0]);
    let mut dt: f64 = 60.0;

    let mut step_count = 0;
    while t < t_end {
        let result = integrator.step(t, state, None, Some(dt.min(t_end - t)));
        t += result.dt_used;
        state = result.state;
        dt = result.dt_next;
        step_count += 1;
    }

    let analytical = (-0.1 * t_end).exp();
    println!("Propagated from 0 to {}s in {} steps", t_end, step_count);
    println!("Final state: {:.6e}", state[0]);
    println!("Analytical: {:.6e}", analytical);
    println!("Error: {:.2e}", (state[0] - analytical).abs());
}
