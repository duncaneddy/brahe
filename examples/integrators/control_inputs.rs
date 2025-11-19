//! Control input example.
//!
//! Demonstrates adding a control input function that adds external forcing
//! to the dynamics, such as thrust maneuvers.

use brahe::constants::{GM_EARTH, R_EARTH};
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
    let control_input = move |_t: f64, state: DVector<f64>| -> DVector<f64> {
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
        None,                          // No Jacobian
        None,                          // No sensitivity provider
        Some(Box::new(control_input)), // Control input
        config,
    );

    // Initial LEO state (500 km altitude)
    let sma = R_EARTH + 500e3;
    let state = DVector::from_vec(vec![
        sma, 0.0, 0.0, // Position
        0.0, 7612.6, 0.0, // Velocity (circular orbit)
    ]);

    // Integrate for one orbit period
    let period = orbital_period(sma);
    let mut t = 0.0;
    let mut dt = 60.0;
    let mut current_state = state.clone();

    println!("Initial semi-major axis: {:.3} km", sma / 1000.0);
    println!(
        "Integrating with continuous thrust for {:.2} hours...",
        period / 3600.0
    );

    while t < period {
        let result = integrator.step(t, current_state, dt);
        current_state = result.state;
        t += result.dt_used;
        dt = result.dt_next;
    }

    let final_r = current_state.fixed_rows::<3>(0).norm();
    let delta_r = final_r - sma;
    println!("Final radius: {:.3} km", final_r / 1000.0);
    println!("Orbit raised by: {:.3} km", delta_r / 1000.0);
}

// Initial radius: 6878.136 km
// Orbital period: 1.58 hours

// After one orbit:
//   With thrust: 6889.325 km (delta_r = 11.189 km)
//   Coast only:  6878.136 km (delta_r = 0.000 km)
