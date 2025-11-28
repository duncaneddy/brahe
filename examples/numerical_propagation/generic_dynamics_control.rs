//! Using NumericalPropagator with control_input.
//! Demonstrates adding damping control to a simple harmonic oscillator via control_input.

use brahe as bh;
use bh::traits::{DStatePropagator, DStateProvider};
use nalgebra as na;
use std::f64::consts::PI;

fn main() {
    // Initialize EOP data (needed for epoch operations)
    bh::initialize_eop().unwrap();

    // Create initial epoch
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Simple Harmonic Oscillator with damping control
    // State: [x, v] where x is position and v is velocity
    // Natural dynamics: dx/dt = v, dv/dt = -omega^2 * x
    // Control adds damping: u = -c * v
    let omega = 2.0 * PI; // 1 Hz natural frequency
    let damping_ratio = 0.1; // Damping ratio (zeta)

    // Initial state: displaced from equilibrium
    let x0 = 1.0; // 1 meter displacement
    let v0 = 0.0; // Starting from rest
    let initial_state = na::DVector::from_vec(vec![x0, v0]);

    // SHO dynamics function (undamped - natural dynamics only)
    // Control is added separately via control_input
    let sho_dynamics: bh::DStateDynamics = Box::new(
        move |_t: f64, state: &na::DVector<f64>, params: Option<&na::DVector<f64>>| {
            let x = state[0];
            let v = state[1];
            let omega_sq = params.map(|p| p[0]).unwrap_or(omega * omega);
            na::DVector::from_vec(vec![v, -omega_sq * x])
        },
    );

    // Another copy for the undamped propagator
    let sho_dynamics_undamped: bh::DStateDynamics = Box::new(
        move |_t: f64, state: &na::DVector<f64>, params: Option<&na::DVector<f64>>| {
            let x = state[0];
            let v = state[1];
            let omega_sq = params.map(|p| p[0]).unwrap_or(omega * omega);
            na::DVector::from_vec(vec![v, -omega_sq * x])
        },
    );

    // Damping control input: u = -c * v (opposes velocity)
    // The control_input function returns a state derivative contribution
    // that is added to the dynamics output at each integration step.
    let damping_coeff = 2.0 * damping_ratio * omega;
    let damping_control: bh::DControlInput = Some(Box::new(
        move |_t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| {
            let v = state[1];
            // Control adds acceleration that opposes velocity
            na::DVector::from_vec(vec![0.0, -damping_coeff * v])
        },
    ));

    // Parameters (omega^2)
    let params = na::DVector::from_vec(vec![omega * omega]);

    // Create propagator with dynamics AND control_input
    let mut prop_damped = bh::DNumericalPropagator::new(
        epoch,
        initial_state.clone(),
        sho_dynamics,
        bh::NumericalPropagationConfig::default(),
        Some(params.clone()),
        damping_control, // Separate control function
        None,            // No initial covariance
    )
    .unwrap();

    // Create undamped propagator for comparison (no control_input)
    let mut prop_undamped = bh::DNumericalPropagator::new(
        epoch,
        initial_state,
        sho_dynamics_undamped,
        bh::NumericalPropagationConfig::default(),
        Some(params),
        None, // No control input
        None,
    )
    .unwrap();

    // Propagate for several periods
    let period = 2.0 * PI / omega; // Period = 1 second
    prop_damped.propagate_to(epoch + 10.0 * period);
    prop_undamped.propagate_to(epoch + 10.0 * period);

    // Sample trajectory and compare
    println!("Damped vs Undamped Harmonic Oscillator:");
    println!("  Natural frequency: {:.1} Hz", omega / (2.0 * PI));
    println!("  Damping ratio: {}", damping_ratio);
    println!("  Damping coefficient: {:.3} /s", damping_coeff);
    println!("\nTime (s)  Damped x    Undamped x  Amplitude ratio");
    println!("{}", "-".repeat(55));

    for i in 0..11 {
        let t = (i as f64) * period; // Sample at period intervals
        let state_damped = prop_damped.state(epoch + t).unwrap();
        let state_undamped = prop_undamped.state(epoch + t).unwrap();
        let ratio = state_damped[0].abs() / state_undamped[0].abs().max(1e-10);
        println!(
            "  {:.1}       {:+.4}      {:+.4}       {:.3}",
            t, state_damped[0], state_undamped[0], ratio
        );
    }

    // Validate - damped oscillator should decay
    let final_damped = prop_damped.state(epoch + 10.0 * period).unwrap();
    let final_undamped = prop_undamped.state(epoch + 10.0 * period).unwrap();

    // Expected decay: amplitude ~ exp(-zeta*omega*t) = exp(-0.1 * 2*pi * 10) ~ 0.002
    let expected_ratio = (-damping_ratio * omega * 10.0 * period).exp();
    let actual_ratio = final_damped[0].abs() / x0;

    println!("\nAfter 10 periods:");
    println!("  Damped amplitude: {:.4} m", final_damped[0].abs());
    println!("  Undamped amplitude: {:.4} m", final_undamped[0].abs());
    println!("  Expected decay ratio: {:.4}", expected_ratio);
    println!("  Actual decay ratio: {:.4}", actual_ratio);

    assert!(final_damped[0].abs() < final_undamped[0].abs()); // Damped has smaller amplitude
    assert!(actual_ratio < 0.1); // Should decay significantly

    println!("\nExample validated successfully!");
}
