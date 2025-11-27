//! Demonstrates RK4 fixed-step integration.

#[allow(unused_imports)]
use brahe::integrators::*;
use nalgebra::DVector;

fn main() {
    // Define simple harmonic oscillator
    let omega: f64 = 1.0;
    let dynamics = move |_t: f64, state: &DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
        let x = state[0];
        let v = state[1];
        DVector::from_vec(vec![v, -omega.powi(2) * x])
    };

    // Analytical solution
    let analytical = |t: f64, x0: f64, v0: f64| -> (f64, f64) {
        let x = x0 * (omega * t).cos() + (v0 / omega) * (omega * t).sin();
        let v = -omega * x0 * (omega * t).sin() + v0 * (omega * t).cos();
        (x, v)
    };

    // Initial conditions
    let state0 = DVector::from_vec(vec![1.0, 0.0]);
    let t_end = 4.0 * std::f64::consts::PI;  // Two periods

    println!("RK4 Fixed-Step Integration Demonstration");
    println!("System: Simple Harmonic Oscillator (ω=1.0)");
    println!("Integration time: 0 to {:.2} (2 periods)\n", t_end);

    // Test different step sizes
    let step_sizes = vec![0.5, 0.2, 0.1, 0.05];

    for &dt in &step_sizes {
        let config = IntegratorConfig::fixed_step(dt);
        let integrator = RK4DIntegrator::with_config(2, Box::new(dynamics), None, None, None, config);

        let mut t = 0.0;
        let mut state = state0.clone();
        let mut steps = 0;

        // Integrate to end
        while t < t_end - 1e-10 {
            state = integrator.step(t, state, None, None).state;
            t += dt;
            steps += 1;
        }

        // Compare with analytical solution
        let (exact_x, exact_v) = analytical(t, 1.0, 0.0);
        let error = ((state[0] - exact_x).powi(2) + (state[1] - exact_v).powi(2)).sqrt();

        println!("Step size dt={:5.2}:", dt);
        println!("  Steps:      {}", steps);
        println!("  Final state: [{:.6}, {:.6}]", state[0], state[1]);
        println!("  Exact:       [{:.6}, {:.6}]", exact_x, exact_v);
        println!("  Error:       {:.2e}", error);
        println!();
    }
}

// Expected output:
// RK4 Fixed-Step Integration Demonstration
// System: Simple Harmonic Oscillator (ω=1.0)
// Integration time: 0 to 12.57 (2 periods)

// Step size dt= 0.50:
//   Steps:      26
//   Final state: [0.907541, -0.413422]
//   Exact:       [0.907447, -0.420167]
//   Error:       6.75e-03

// Step size dt= 0.20:
//   Steps:      63
//   Final state: [0.999412, -0.033457]
//   Exact:       [0.999435, -0.033623]
//   Error:       1.68e-04

// Step size dt= 0.10:
//   Steps:      126
//   Final state: [0.999434, -0.033613]
//   Exact:       [0.999435, -0.033623]
//   Error:       1.05e-05

// Step size dt= 0.05:
//   Steps:      252
//   Final state: [0.999435, -0.033622]
//   Exact:       [0.999435, -0.033623]
//   Error:       6.56e-07