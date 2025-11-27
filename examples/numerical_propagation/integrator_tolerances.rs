//! Setting custom error tolerances for adaptive integrators.
//! Shows how tolerances affect accuracy and performance.

use brahe as bh;

fn main() {
    // Error tolerances control the accuracy of adaptive integrators
    // The integrator adjusts step size to keep error within:
    //   error < abs_tol + rel_tol * |state|

    // Default configuration (general purpose)
    let _config_default = bh::NumericalPropagationConfig::default();

    // Quick analysis (looser tolerances, faster)
    let _config_quick = bh::NumericalPropagationConfig {
        method: bh::IntegratorMethod::DP54,
        integrator: bh::IntegratorConfig::adaptive(1e-3, 1e-1),
        variational: bh::VariationalConfig::default(),
    };

    // Standard precision
    let _config_standard = bh::NumericalPropagationConfig {
        method: bh::IntegratorMethod::DP54,
        integrator: bh::IntegratorConfig::adaptive(1e-6, 1e-3),
        variational: bh::VariationalConfig::default(),
    };

    // High precision
    let _config_precision = bh::NumericalPropagationConfig {
        method: bh::IntegratorMethod::DP54,
        integrator: bh::IntegratorConfig::adaptive(1e-9, 1e-6),
        variational: bh::VariationalConfig::default(),
    };

    // Maximum precision (use with RKN1210)
    let _config_maximum = bh::NumericalPropagationConfig {
        method: bh::IntegratorMethod::RKN1210,
        integrator: bh::IntegratorConfig::adaptive(1e-12, 1e-10),
        variational: bh::VariationalConfig::default(),
    };

    println!("Integrator Error Tolerances");
    println!("{}", "=".repeat(60));

    println!("\nError Control Formula:");
    println!("  error < abs_tol + rel_tol * |state|");
    println!("\n  - abs_tol: Bounds error when state is small");
    println!("  - rel_tol: Bounds error proportional to state magnitude");

    println!("\nTolerance Presets:");
    println!("\n| Level          | abs_tol | rel_tol | Use Case                   |");
    println!("|----------------|---------|---------|----------------------------|");
    println!("| Quick analysis | 1e-3    | 1e-1    | Rough estimates            |");
    println!("| Standard       | 1e-6    | 1e-3    | General mission analysis   |");
    println!("| High precision | 1e-9    | 1e-6    | Precision applications     |");
    println!("| Maximum        | 1e-12   | 1e-10   | POD, research (use RKN1210)|");

    println!("\nGuidelines for Choosing Tolerances:");

    println!("\nAbsolute Tolerance (abs_tol):");
    println!("  - Controls error when state components are small");
    println!("  - For position: set to desired position accuracy (meters)");
    println!("  - For velocity: set to desired velocity accuracy (m/s)");
    println!("  - Default: 1e-6 (~1 micrometer position)");

    println!("\nRelative Tolerance (rel_tol):");
    println!("  - Controls error as fraction of state magnitude");
    println!("  - 1e-3 = 0.1% relative error");
    println!("  - 1e-6 = 0.0001% relative error");
    println!("  - Default: 1e-3 (0.1% relative accuracy)");

    println!("\nTrade-offs:");
    println!("  Tighter tolerances -> Smaller steps -> More computation");
    println!("  Looser tolerances -> Larger steps -> Less computation");
    println!("  Find the minimum accuracy that meets your requirements");

    println!("\nExample validated successfully!");
}
