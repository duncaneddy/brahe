//! Configuring the RK4 fixed-step integrator method.

use brahe as bh;

fn main() {
    // RK4: Fixed-step 4th-order Runge-Kutta
    let config = bh::NumericalPropagationConfig {
        method: bh::IntegratorMethod::RK4,
        integrator: bh::IntegratorConfig::fixed_step(60.0), // 60 second fixed steps
        variational: bh::VariationalConfig::default(),
        store_accelerations: true,
        interpolation_method: bh::InterpolationMethod::Linear,
    };

    println!("Method: {:?}", config.method);
    println!("Fixed step: {:?} seconds", config.integrator.fixed_step_size);
    // Method: RK4
    // Fixed step: Some(60.0) seconds
}
