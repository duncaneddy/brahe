//! Using NumericalPropagator for arbitrary dynamics.
//! Demonstrates propagating non-orbital systems (simple harmonic oscillator).

use brahe as bh;
use bh::traits::{DStatePropagator, DStateProvider};
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize EOP data (needed for epoch operations)
    bh::initialize_eop().unwrap();

    // Create initial epoch
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Simple Harmonic Oscillator (SHO)
    // State: [x, v] where x is position and v is velocity
    // Dynamics: dx/dt = v, dv/dt = -omega^2 * x
    let omega = 2.0 * PI; // 1 Hz oscillation frequency

    // Initial state: displaced from equilibrium
    let x0 = 1.0; // 1 meter displacement
    let v0 = 0.0; // Starting from rest
    let initial_state = na::DVector::from_vec(vec![x0, v0]);

    // SHO dynamics function
    let dynamics_fn: bh::DStateDynamics = Box::new(
        move |_t: f64, state: &na::DVector<f64>, params: Option<&na::DVector<f64>>| {
            let x = state[0];
            let v = state[1];
            let omega_sq = params.map(|p| p[0]).unwrap_or(omega * omega);
            na::DVector::from_vec(vec![v, -omega_sq * x])
        },
    );

    // Parameters (omega^2)
    let params = na::DVector::from_vec(vec![omega * omega]);

    // Create generic numerical propagator
    let mut prop = bh::DNumericalPropagator::new(
        epoch,
        initial_state,
        dynamics_fn,
        bh::NumericalPropagationConfig::default(),
        Some(params),
        None, // No control input
        None, // No initial covariance
    )
    .unwrap();

    // Propagate for several periods
    let period = 2.0 * PI / omega; // Period = 2*pi/omega = 1 second
    prop.propagate_to(epoch + 5.0 * period);

    // Sample trajectory
    println!("Simple Harmonic Oscillator Trajectory:");
    println!("  omega = 2*pi rad/s (1 Hz)");
    println!("  x0 = 1.0 m, v0 = 0.0 m/s");
    println!("\nTime (s)  Position (m)  Velocity (m/s)  Analytical x");
    println!("{}", "-".repeat(55));

    for i in 0..11 {
        let t = (i as f64) * period / 2.0; // Sample at half-period intervals
        let state = prop.state(epoch + t).unwrap();
        // Analytical solution: x(t) = x0*cos(omega*t), v(t) = -x0*omega*sin(omega*t)
        let x_analytical = x0 * (omega * t).cos();
        println!(
            "  {:.2}      {:+.6}      {:+.6}      {:+.6}",
            t, state[0], state[1], x_analytical
        );
    }

    // Validate - after full period should return to initial
    let final_state = prop.state(epoch + 5.0 * period).unwrap();
    let error_x = (final_state[0] - x0).abs();
    let error_v = (final_state[1] - v0).abs();

    println!("\nAfter 5 periods:");
    println!("  Position error: {:.2e} m", error_x);
    println!("  Velocity error: {:.2e} m/s", error_v);

    assert!(error_x < 0.01);  // Within 1 cm
    assert!(error_v < 0.1);   // Within 10 cm/s

    println!("\nExample validated successfully!");
}
