//! ```cargo
//! [dependencies]
//! brahe = { path = "../.." }
//! nalgebra = "0.34"
//! ```

//! Control input pattern example.
//!
//! Demonstrates adding a control input function that perturbs the dynamics,
//! useful for modeling thrust or other external forcing functions.

use brahe::constants::GM_EARTH;
use brahe::integrators::*;
use brahe::orbits::keplerian::orbital_period;
use nalgebra::DVector;

fn main() {
    // Orbital dynamics (gravity only)
    let dynamics =
        move |_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
            let r = state.fixed_rows::<3>(0);
            let v = state.fixed_rows::<3>(3);
            let r_norm = r.norm();
            let a_grav = -GM_EARTH / (r_norm * r_norm * r_norm) * r;

            let mut state_dot = DVector::<f64>::zeros(6);
            state_dot
                .fixed_rows_mut::<3>(0)
                .copy_from(&v.clone_owned());
            state_dot.fixed_rows_mut::<3>(3).copy_from(&a_grav);
            state_dot
        };

    // Control input: constant low thrust in velocity direction
    let control_input = move |_t: f64, state: DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
        let v = state.fixed_rows::<3>(3);
        let v_norm = v.norm();

        let mut control = DVector::<f64>::zeros(6);
        if v_norm > 0.0 {
            let thrust_magnitude = 0.001; // m/s^2
            let a_thrust = thrust_magnitude * v / v_norm;
            control.fixed_rows_mut::<3>(3).copy_from(&a_thrust);
        }
        control
    };

    // Create integrator with control input
    let config = IntegratorConfig::adaptive(1e-10, 1e-8);
    let integrator = DormandPrince54DIntegrator::with_config(
        6,
        Box::new(dynamics),
        None,                        // No variational matrix
        None,                        // No sensitivity provider
        Some(Box::new(control_input)), // Control input
        config,
    );

    // Initial LEO state (500 km altitude)
    let sma = 6.878e6; // R_EARTH + 500km
    let state = DVector::from_vec(vec![
        sma, 0.0, 0.0, // Position
        0.0, 7612.6, 0.0, // Velocity (circular orbit)
    ]);

    // Integrate for one orbit period
    let period = orbital_period(sma);
    let mut t = 0.0;
    let mut dt = 60.0;
    let mut current_state = state.clone();

    println!("Initial position magnitude: {:.3} km", sma / 1000.0);
    println!("Integrating with thrust for {:.2} hours...", period / 3600.0);

    while t < period {
        let result = integrator.step(t, current_state, Some(dt));
        current_state = result.state;
        t += result.dt_used;
        dt = result.dt_next;
    }

    let final_r = current_state.fixed_rows::<3>(0).norm();
    println!("Final position magnitude: {:.3} km", final_r / 1000.0);
    println!("Change: {:.3} km", (final_r - sma) / 1000.0);
}
