//! Using struct update syntax to customize integrator configuration.

use brahe as bh;

fn main() {
    // Use struct update syntax (..) to customize from a preset
    let config = bh::NumericalPropagationConfig {
        integrator: bh::IntegratorConfig {
            abs_tol: 1e-9,
            rel_tol: 1e-6,
            initial_step: Some(60.0),
            max_step: Some(300.0),
            ..bh::IntegratorConfig::default()
        },
        ..bh::NumericalPropagationConfig::default()
    };

    println!("Method: {:?}", config.method);
    println!("abs_tol: {:e}", config.integrator.abs_tol);
    println!("rel_tol: {:e}", config.integrator.rel_tol);
    println!("max_step: {:?}", config.integrator.max_step);
    // Method: DP54
    // abs_tol: 1e-9
    // rel_tol: 1e-6
    // max_step: Some(300.0)
}
