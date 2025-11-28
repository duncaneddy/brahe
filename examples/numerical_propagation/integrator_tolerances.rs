//! Setting custom error tolerances for adaptive integrators.

use brahe as bh;

fn main() {
    // Different tolerance levels for various use cases
    let config_quick = bh::NumericalPropagationConfig {
        integrator: bh::IntegratorConfig::adaptive(1e-3, 1e-1),
        ..bh::NumericalPropagationConfig::default()
    };
    let config_standard = bh::NumericalPropagationConfig::default(); // abs=1e-6, rel=1e-3
    let config_precision = bh::NumericalPropagationConfig {
        integrator: bh::IntegratorConfig::adaptive(1e-9, 1e-6),
        ..bh::NumericalPropagationConfig::default()
    };
    let config_maximum = bh::NumericalPropagationConfig::high_precision(); // abs=1e-10, rel=1e-8

    println!("Quick:     abs={:e}, rel={:e}", config_quick.integrator.abs_tol, config_quick.integrator.rel_tol);
    println!("Standard:  abs={:e}, rel={:e}", config_standard.integrator.abs_tol, config_standard.integrator.rel_tol);
    println!("Precision: abs={:e}, rel={:e}", config_precision.integrator.abs_tol, config_precision.integrator.rel_tol);
    println!("Maximum:   abs={:e}, rel={:e}", config_maximum.integrator.abs_tol, config_maximum.integrator.rel_tol);
    // Quick:     abs=1e-3, rel=1e-1
    // Standard:  abs=1e-6, rel=1e-3
    // Precision: abs=1e-9, rel=1e-6
    // Maximum:   abs=1e-10, rel=1e-8
}
