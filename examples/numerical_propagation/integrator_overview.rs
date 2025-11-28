//! Complete overview of NumericalPropagationConfig showing all configuration fields.
//! This example demonstrates every configurable option for integrator configuration.

use brahe as bh;

fn main() {
    // Create a fully-configured integrator configuration
    let config = bh::NumericalPropagationConfig {
        // Integration method: Dormand-Prince 5(4)
        method: bh::IntegratorMethod::DP54,
        // Integrator settings: tolerances and step control
        integrator: bh::IntegratorConfig {
            abs_tol: 1e-9,
            rel_tol: 1e-6,
            initial_step: Some(60.0), // 60 second initial step
            min_step: Some(1e-6),     // Minimum step size
            max_step: Some(300.0),    // Maximum step size (5 minutes)
            step_safety_factor: Some(0.9),      // Safety margin
            min_step_scale_factor: Some(0.2),   // Minimum step reduction
            max_step_scale_factor: Some(10.0),  // Maximum step growth
            max_step_attempts: 10,              // Max attempts per step
            fixed_step_size: None,              // Not using fixed step
        },
        // Variational configuration: STM and sensitivity settings
        variational: bh::VariationalConfig {
            enable_stm: true,
            enable_sensitivity: false,
            store_stm_history: true,
            store_sensitivity_history: false,
            jacobian_method: bh::DifferenceMethod::Central,
            sensitivity_method: bh::DifferenceMethod::Central,
        },
    };

    println!("Method: {:?}", config.method);
    println!("Integrator: {:?}", config.integrator);
    println!("Variational: {:?}", config.variational);
    // Method: DP54
    // Integrator: IntegratorConfig { abs_tol: 1e-9, rel_tol: 1e-6, ... }
    // Variational: VariationalConfig { enable_stm: true, enable_sensitivity: false, ... }
}
