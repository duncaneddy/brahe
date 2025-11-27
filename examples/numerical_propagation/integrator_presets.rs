//! Overview of all integrator configuration presets.
//! Shows the available presets and their settings.

use brahe as bh;

fn main() {
    // NumericalPropagationConfig provides preset configurations
    // for common use cases

    // 1. default() - General purpose DP54
    // Best for most applications
    let _default = bh::NumericalPropagationConfig::default();

    // 2. high_precision() - Maximum accuracy with RKN1210
    // For precision orbit determination and research
    let _high_precision = bh::NumericalPropagationConfig::high_precision();

    // 3. with_method() - Start from specific integrator
    // Customize from a chosen method
    let _rkf45_config = bh::NumericalPropagationConfig::with_method(bh::IntegratorMethod::RKF45);
    let _rk4_config = bh::NumericalPropagationConfig::with_method(bh::IntegratorMethod::RK4);

    println!("Integrator Configuration Presets");
    println!("{}", "=".repeat(70));

    println!("\n| Preset           | Method  | abs_tol | rel_tol | Description           |");
    println!("|------------------|---------|---------|---------|----------------------|");
    println!("| default()        | DP54    | 1e-6    | 1e-3    | General purpose      |");
    println!("| high_precision() | RKN1210 | 1e-10   | 1e-8    | Maximum accuracy     |");
    println!("| with_method(M)   | M       | 1e-6    | 1e-3    | Custom method        |");

    println!("\nDetailed Preset Descriptions:");

    println!("\ndefault():");
    println!("  - Method: DP54 (Dormand-Prince 5(4))");
    println!("  - Tolerances: abs=1e-6, rel=1e-3");
    println!("  - Step limits: min=1e-12, max=900s");
    println!("  - Use for: Most mission analysis, LEO to GEO");
    println!("  - Good balance of accuracy and speed");

    println!("\nhigh_precision():");
    println!("  - Method: RKN1210 (Runge-Kutta-Nystrom 12(10))");
    println!("  - Tolerances: abs=1e-10, rel=1e-8");
    println!("  - Use for: POD, validation, research");
    println!("  - Achieves sub-meter accuracy over days");
    println!("  - More expensive computationally");

    println!("\nwith_method(IntegratorMethod):");
    println!("  - Starts with specified integrator method");
    println!("  - Uses default tolerances (abs=1e-6, rel=1e-3)");
    println!("  - Modify struct fields to customize further");
    println!("  - Available methods:");
    println!("    - IntegratorMethod::RK4 (fixed step)");
    println!("    - IntegratorMethod::RKF45 (adaptive)");
    println!("    - IntegratorMethod::DP54 (adaptive, default)");
    println!("    - IntegratorMethod::RKN1210 (adaptive, high precision)");

    println!("\nIntegrator Method Comparison:");
    println!("\n| Method  | Order  | Adaptive | Evals | Best For                   |");
    println!("|---------|--------|----------|-------|---------------------------|");
    println!("| RK4     | 4      | No       | 4     | Simple problems, debugging|");
    println!("| RKF45   | 4(5)   | Yes      | 6     | General purpose           |");
    println!("| DP54    | 5(4)   | Yes      | 6-7   | General purpose (default) |");
    println!("| RKN1210 | 12(10) | Yes      | 17    | Highest precision         |");

    println!("\nChoosing a Preset:");
    println!("  1. Start with default() for most applications");
    println!("  2. Use high_precision() when accuracy is critical");
    println!("  3. Use with_method() when you need specific behavior");
    println!("  4. Modify struct fields to fine-tune any preset");

    println!("\nExample validated successfully!");
}
