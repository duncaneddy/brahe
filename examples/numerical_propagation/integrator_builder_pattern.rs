//! Using struct initialization to customize integrator configuration.
//! Demonstrates creating configurations with custom settings.

use brahe as bh;

fn main() {
    // In Rust, configuration is done through struct initialization
    // rather than builder pattern (unlike the Python API)

    // Example 1: Customize from default with modified tolerances
    let config1 = bh::NumericalPropagationConfig {
        integrator: bh::IntegratorConfig {
            abs_tol: 1e-9,
            rel_tol: 1e-6,
            max_step: Some(300.0),
            ..bh::IntegratorConfig::default()
        },
        ..bh::NumericalPropagationConfig::default()
    };

    // Example 2: Start with specific method and customize
    let config2 = bh::NumericalPropagationConfig {
        method: bh::IntegratorMethod::RKF45,
        integrator: bh::IntegratorConfig {
            abs_tol: 1e-8,
            rel_tol: 1e-5,
            min_step: Some(1e-6),
            max_step: Some(600.0),
            ..bh::IntegratorConfig::default()
        },
        variational: bh::VariationalConfig::default(),
    };

    // Example 3: High precision with step constraints
    let config3 = bh::NumericalPropagationConfig {
        integrator: bh::IntegratorConfig {
            max_step: Some(120.0), // Limit max step for output resolution
            ..bh::IntegratorConfig::adaptive(1e-10, 1e-8)
        },
        ..bh::NumericalPropagationConfig::high_precision()
    };

    // Example 4: Fixed-step configuration
    let config4 = bh::NumericalPropagationConfig {
        method: bh::IntegratorMethod::RK4,
        integrator: bh::IntegratorConfig::fixed_step(30.0), // 30 second fixed steps
        variational: bh::VariationalConfig::default(),
    };

    // Silence unused warnings
    let _ = (config1, config2, config3, config4);

    println!("Configuration Customization in Rust");
    println!("{}", "=".repeat(60));

    println!("\nIntegratorConfig Fields:");
    println!("  abs_tol: f64            - Absolute error tolerance");
    println!("  rel_tol: f64            - Relative error tolerance");
    println!("  initial_step: Option    - Initial step size");
    println!("  min_step: Option        - Minimum step size");
    println!("  max_step: Option        - Maximum step size");
    println!("  fixed_step_size: Option - Fixed step (for RK4)");

    println!("\nFactory Methods:");
    println!("  IntegratorConfig::default()           - Standard settings");
    println!("  IntegratorConfig::fixed_step(dt)      - Fixed step size");
    println!("  IntegratorConfig::adaptive(abs, rel)  - Custom tolerances");

    println!("\nExample 1: Precision DP54");
    println!("  NumericalPropagationConfig {{");
    println!("      integrator: IntegratorConfig {{");
    println!("          abs_tol: 1e-9,");
    println!("          rel_tol: 1e-6,");
    println!("          max_step: Some(300.0),");
    println!("          ..IntegratorConfig::default()");
    println!("      }},");
    println!("      ..NumericalPropagationConfig::default()");
    println!("  }}");

    println!("\nExample 2: Custom RKF45");
    println!("  NumericalPropagationConfig {{");
    println!("      method: IntegratorMethod::RKF45,");
    println!("      integrator: IntegratorConfig {{");
    println!("          abs_tol: 1e-8,");
    println!("          rel_tol: 1e-5,");
    println!("          ..IntegratorConfig::default()");
    println!("      }},");
    println!("      variational: VariationalConfig::default(),");
    println!("  }}");

    println!("\nBenefits of Struct Initialization:");
    println!("  - Use struct update syntax (..) for defaults");
    println!("  - Only specify fields you want to change");
    println!("  - Type-safe configuration");
    println!("  - Compile-time validation");

    println!("\nExample validated successfully!");
}
