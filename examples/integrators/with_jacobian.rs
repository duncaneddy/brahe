//! Demonstrates integration with state transition matrix propagation.

#[allow(unused_imports)]
use brahe::*;
use nalgebra::{DMatrix, DVector};

fn main() {
    // Initialize EOP
    initialize_eop().unwrap();

    // Define two-body dynamics (for integrator - takes optional params)
    let dynamics = |_t: f64, state: &DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
        let r = nalgebra::Vector3::new(state[0], state[1], state[2]);
        let v = nalgebra::Vector3::new(state[3], state[4], state[5]);
        let r_norm = r.norm();
        let a = -GM_EARTH / r_norm.powi(3) * r;
        DVector::from_vec(vec![v[0], v[1], v[2], a[0], a[1], a[2]])
    };

    // Same dynamics for Jacobian (without params argument)
    let dynamics_for_jac = |_t: f64, state: &DVector<f64>, _params: Option<&DVector<f64>>| -> DVector<f64> {
        let r = nalgebra::Vector3::new(state[0], state[1], state[2]);
        let v = nalgebra::Vector3::new(state[3], state[4], state[5]);
        let r_norm = r.norm();
        let a = -GM_EARTH / r_norm.powi(3) * r;
        DVector::from_vec(vec![v[0], v[1], v[2], a[0], a[1], a[2]])
    };

    // Create numerical Jacobian for variational equations
    let jacobian = DNumericalJacobian::central(Box::new(dynamics_for_jac))
        .with_adaptive(1.0, 1e-6);

    // Initial orbit (LEO)
    let r0 = DVector::from_vec(vec![R_EARTH + 600e3, 0.0, 0.0]);
    let v0 = DVector::from_vec(vec![0.0, 7.5e3, 0.0]);
    let state0 = DVector::from_vec(vec![r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]]);

    // Initial state transition matrix (identity)
    let phi0 = DMatrix::identity(6, 6);

    println!("Integration with State Transition Matrix");
    println!("Initial orbit: {:.1} km altitude", r0[0] / 1e3);

    // Create integrator with Jacobian
    let config = IntegratorConfig::adaptive(1e-12, 1e-11);
    let integrator = DormandPrince54DIntegrator::with_config(6, Box::new(dynamics), Some(Box::new(jacobian)), None, None, config);

    // Propagate for one orbit period
    let mut t = 0.0;
    let mut state = state0.clone();
    let mut phi = phi0.clone();
    let mut dt: f64 = 60.0;

    // Approximate orbital period
    let period = orbital_period(r0.norm());

    println!("Time   Position STM[0,0]  Velocity STM[3,3]  Det(STM)");
    println!("{}", "-".repeat(60));

    let mut steps = 0;
    while t < period {
        // Propagate state and STM together
        let result = integrator.step_with_varmat(t, state, None, phi, Some(dt.min(period - t)));

        let new_state = result.state;
        let new_phi = result.phi.unwrap();
        let dt_used = result.dt_used;
        let dt_next = result.dt_next;

        t += dt_used;
        state = new_state;
        phi = new_phi;
        dt = dt_next;
        steps += 1;

        // Print progress
        if steps % 20 == 1 {
            let det_phi = phi.clone().lu().determinant();
            println!("{:6.0}s    {:8.4}      {:8.4}        {:8.4}",
                     t, phi[(0, 0)], phi[(3, 3)], det_phi);
        }
    }

    println!("\nPropagation complete! ({} steps)", steps);

    // Example: Map initial position uncertainty to final uncertainty
    println!("\nExample: Uncertainty Propagation");
    let dx = 100.0;
    println!("Initial position uncertainty: ±{} m in each direction", dx);
    let delta_r0 = DVector::from_vec(vec![dx, dx, dx, 0.0, 0.0, 0.0]);
    let delta_rf = &phi * &delta_r0;
    println!("Final position uncertainty: [{:.1}, {:.1}, {:.1}] m",
             delta_rf[0], delta_rf[1], delta_rf[2]);

    let r0_norm = (delta_r0[0].powi(2) + delta_r0[1].powi(2) + delta_r0[2].powi(2)).sqrt();
    let rf_norm = (delta_rf[0].powi(2) + delta_rf[1].powi(2) + delta_rf[2].powi(2)).sqrt();
    println!("Uncertainty growth: {:.1}x", rf_norm / r0_norm);
}
