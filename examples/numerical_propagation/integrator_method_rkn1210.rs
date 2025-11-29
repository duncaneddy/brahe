//! Configuring the RKN1210 high-precision integrator method.

use brahe as bh;

fn main() {
    // RKN1210: High-order adaptive integrator for maximum precision
    let config = bh::NumericalPropagationConfig::high_precision();

    // Or manually configure with custom tolerances
    let config_custom = bh::NumericalPropagationConfig {
        method: bh::IntegratorMethod::RKN1210,
        integrator: bh::IntegratorConfig::adaptive(1e-12, 1e-10),
        variational: bh::VariationalConfig::default(),
        store_accelerations: true,
        interpolation_method: bh::InterpolationMethod::Linear,
    };

    println!("Method: {:?}", config.method);
    println!("abs_tol: {:e}", config.integrator.abs_tol);
    println!("rel_tol: {:e}", config.integrator.rel_tol);

    // Silence unused warning
    let _ = config_custom;
    // Method: RKN1210
    // abs_tol: 1e-10
    // rel_tol: 1e-8
}
