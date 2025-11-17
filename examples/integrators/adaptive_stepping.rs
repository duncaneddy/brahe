//! Demonstrates adaptive-step integration with automatic error control.

#[allow(unused_imports)]
use brahe::integrators::*;
use nalgebra::DVector;

fn main() {
    // Define dynamics: Van der Pol oscillator (stiff for large mu)
    let mu = 1.0;
    let dynamics = move |_t: f64, state: DVector<f64>| -> DVector<f64> {
        let x = state[0];
        let v = state[1];
        DVector::from_vec(vec![v, mu * (1.0 - x.powi(2)) * v - x])
    };

    // Initial conditions
    let t0 = 0.0;
    let state0 = DVector::from_vec(vec![2.0, 0.0]);
    let t_end = 10.0;

    // Create adaptive integrator
    let abs_tol = 1e-8;
    let rel_tol = 1e-7;
    let config = IntegratorConfig::adaptive(abs_tol, rel_tol);
    let integrator = DormandPrince54DIntegrator::with_config(2, Box::new(dynamics), None, config);

    println!("Adaptive integration of Van der Pol oscillator (μ={})", mu);
    println!("Tolerances: abs_tol={}, rel_tol={}", abs_tol, rel_tol);
    println!("Integration time: 0 to {} seconds\n", t_end);

    // Integrate with adaptive stepping
    let mut t = t0;
    let mut state = state0.clone();
    let mut dt = 0.1_f64;  // Initial guess
    let mut steps = 0;
    let mut min_dt = f64::INFINITY;
    let mut max_dt = 0.0_f64;

    println!("   Time    State              Step Size   Error Est");
    println!("{}", "-".repeat(65));

    while t < t_end {
        let result = integrator.step(t, state, dt.min(t_end - t));

        // Track step size statistics
        min_dt = min_dt.min(result.dt_used);
        max_dt = max_dt.max(result.dt_used);

        // Update state
        t += result.dt_used;
        state = result.state;
        dt = result.dt_next;
        steps += 1;

        // Print progress
        if steps % 10 == 1 {
            println!("{:7.3}    [{:6.3}, {:6.3}]    {:7.4}     {:.2e}",
                     t, state[0], state[1], result.dt_used, result.error_estimate);
        }
    }

    println!("\nIntegration complete!");
    println!("Total steps: {}", steps);
    println!("Min step size: {:.6} s", min_dt);
    println!("Max step size: {:.6} s", max_dt);
    println!("Average step: {:.6} s", t_end / steps as f64);
    println!("\nAdaptive stepping automatically adjusted step size");
    println!("by {:.1}x during integration", max_dt / min_dt);
}
