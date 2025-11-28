//! Configuring the DP54 adaptive integrator method (default).

use brahe as bh;

fn main() {
    // DP54: Dormand-Prince 5(4) - the default integrator
    let config = bh::NumericalPropagationConfig::default();

    // Customize tolerances
    let config_tight = bh::NumericalPropagationConfig {
        integrator: bh::IntegratorConfig::adaptive(1e-9, 1e-6),
        ..bh::NumericalPropagationConfig::default()
    };

    println!("Method: {:?}", config.method);
    println!("abs_tol: {:e}", config.integrator.abs_tol);
    println!("rel_tol: {:e}", config.integrator.rel_tol);

    // Silence unused warning
    let _ = config_tight;
    // Method: DP54
    // abs_tol: 1e-6
    // rel_tol: 1e-3
}
