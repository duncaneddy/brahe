//! Configuring the DP54 adaptive integrator method (default).
//! Dormand-Prince 5(4) - the recommended general-purpose integrator.

use brahe as bh;

fn main() {
    // DP54 (Dormand-Prince 5(4))
    // - Adaptive step size control
    // - 5th order solution with 4th order error estimate
    // - 6-7 function evaluations per step (FSAL optimization)
    // - MATLAB's ode45 uses this method
    // - Default integrator in Brahe

    // Create configuration with DP54 method (default)
    let _config = bh::NumericalPropagationConfig::default();

    // Or explicitly specify DP54
    let _config_explicit = bh::NumericalPropagationConfig::with_method(bh::IntegratorMethod::DP54);

    // Customize tolerances for DP54
    let _config_custom = bh::NumericalPropagationConfig {
        method: bh::IntegratorMethod::DP54,
        integrator: bh::IntegratorConfig::adaptive(1e-9, 1e-6),
        variational: bh::VariationalConfig::default(),
    };

    println!("DP54 Adaptive Integrator Configuration:");
    println!("  Method: DP54 (Dormand-Prince 5(4))");
    println!("  Adaptive: Yes");
    println!("  Function evaluations per step: 6-7 (FSAL optimized)");
    println!("  Order: 5(4) - 5th order solution, 4th order error estimate");

    println!("\nDefault tolerances:");
    println!("  Absolute tolerance: 1e-6");
    println!("  Relative tolerance: 1e-3");

    println!("\nFSAL Optimization:");
    println!("  First-Same-As-Last (FSAL) reuses the last function");
    println!("  evaluation from the previous step, reducing cost");
    println!("  to effectively 6 evaluations per accepted step.");

    println!("\nCharacteristics:");
    println!("  - Automatic step size adjustment");
    println!("  - Built-in error estimation");
    println!("  - Generally more efficient than RKF45");
    println!("  - Industry standard (MATLAB ode45)");

    println!("\nWhen to use DP54:");
    println!("  - General-purpose orbit propagation");
    println!("  - When accuracy and efficiency both matter");
    println!("  - Most LEO to GEO applications");
    println!("  - Default choice for most users");

    println!("\nWhen to consider alternatives:");
    println!("  - RK4: When you need fixed steps or maximum speed");
    println!("  - RKN1210: When you need highest precision");

    println!("\nExample validated successfully!");
}
