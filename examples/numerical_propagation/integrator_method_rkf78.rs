//! Configuring the RKF78 adaptive integrator method.

use brahe as bh;

fn main() {
    // RKF78: Runge-Kutta-Fehlberg 7(8), useful for tight tolerances
    let config = bh::NumericalPropagationConfig {
        method: bh::IntegratorMethod::RKF78,
        integrator: bh::IntegratorConfig::adaptive(1e-10, 1e-8),
        ..bh::NumericalPropagationConfig::default()
    };

    println!("Method: {:?}", config.method);
    println!("abs_tol: {:e}", config.integrator.abs_tol);
    println!("rel_tol: {:e}", config.integrator.rel_tol);
}
