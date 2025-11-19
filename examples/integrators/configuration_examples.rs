//! Examples of different integrator configurations for various scenarios.

#[allow(unused_imports)]
use brahe::{constants::*, eop::*, integrators::*};

fn main() {
    println!("Integrator Configuration Examples\n");
    println!("{}", "=".repeat(70));

    // Example 1: Conservative (High Reliability)
    println!("\n1. CONSERVATIVE Configuration (Mission-Critical)");
    println!("{}", "-".repeat(70));
    // --8<-- [start:conservative]
    let conservative_config = IntegratorConfig {
        abs_tol: 1e-12,
        rel_tol: 1e-11,
        initial_step: Some(10.0),
        min_step: Some(0.01),
        max_step: Some(100.0),
        step_safety_factor: Some(0.85),  // More conservative
        min_step_scale_factor: Some(0.3),
        max_step_scale_factor: Some(5.0),  // Limit step growth
        max_step_attempts: 15,
        fixed_step_size: None,
    };
    // --8<-- [end:conservative]

    println!("  abs_tol: {:.0e}", conservative_config.abs_tol);
    println!("  rel_tol: {:.0e}", conservative_config.rel_tol);
    println!("  max_step: {:.0} s", conservative_config.max_step.unwrap());
    println!("  safety_factor: {}", conservative_config.step_safety_factor.unwrap());
    println!("  Use case: Critical operations, high-precision ephemeris");

    // Example 2: Balanced (Recommended Default)
    println!("\n2. BALANCED Configuration (Recommended)");
    println!("{}", "-".repeat(70));
    // --8<-- [start:balanced]
    let balanced_config = IntegratorConfig::adaptive(1e-10, 1e-9);
    // --8<-- [end:balanced]

    println!("  abs_tol: {:.0e}", balanced_config.abs_tol);
    println!("  rel_tol: {:.0e}", balanced_config.rel_tol);
    println!("  max_step: {:.0e} s", balanced_config.max_step.unwrap());
    println!("  safety_factor: {}", balanced_config.step_safety_factor.unwrap());
    println!("  Use case: Most applications, ~1-10m accuracy");

    // Example 3: Aggressive (High Performance)
    println!("\n3. AGGRESSIVE Configuration (Fast Computation)");
    println!("{}", "-".repeat(70));
    // --8<-- [start:aggressive]
    let aggressive_config = IntegratorConfig {
        abs_tol: 1e-8,
        rel_tol: 1e-7,
        initial_step: Some(60.0),
        min_step: Some(1.0),
        max_step: Some(1000.0),  // Large steps allowed
        step_safety_factor: Some(0.95),  // Less conservative
        min_step_scale_factor: Some(0.1),
        max_step_scale_factor: Some(15.0),  // Allow rapid growth
        max_step_attempts: 8,
        fixed_step_size: None,
    };
    // --8<-- [end:aggressive]

    println!("  abs_tol: {:.0e}", aggressive_config.abs_tol);
    println!("  rel_tol: {:.0e}", aggressive_config.rel_tol);
    println!("  max_step: {:.0} s", aggressive_config.max_step.unwrap());
    println!("  safety_factor: {}", aggressive_config.step_safety_factor.unwrap());
    println!("  Use case: Fast trajectory analysis, ~10-100m accuracy");

    // Example 4: High Precision (RKN1210)
    println!("\n4. HIGH PRECISION Configuration (Sub-meter)");
    println!("{}", "-".repeat(70));
    // --8<-- [start:high_precision]
    let high_precision_config = IntegratorConfig {
        abs_tol: 1e-14,
        rel_tol: 1e-13,
        initial_step: Some(10.0),
        min_step: Some(0.001),
        max_step: Some(200.0),
        step_safety_factor: Some(0.9),
        min_step_scale_factor: Some(0.2),
        max_step_scale_factor: Some(10.0),
        max_step_attempts: 12,
        fixed_step_size: None,
    };
    // --8<-- [end:high_precision]

    println!("  abs_tol: {:.0e}", high_precision_config.abs_tol);
    println!("  rel_tol: {:.0e}", high_precision_config.rel_tol);
    println!("  max_step: {:.0} s", high_precision_config.max_step.unwrap());
    println!("  safety_factor: {}", high_precision_config.step_safety_factor.unwrap());
    println!("  Use case: High-precision orbit determination, <1m accuracy");
    println!("  Requires: RKN1210 integrator");

    // Example 5: Fixed-Step Configuration
    println!("\n5. FIXED-STEP Configuration");
    println!("{}", "-".repeat(70));
    let _fixed_config = IntegratorConfig::fixed_step(60.0);

    println!("  step_size: {} s", 60.0);
    println!("  Note: Step size can be overridden with dt parameter in integrator.step()");
    println!("  Use case: Regular output intervals, predictable cost");

    println!("\n{}", "=".repeat(70));
    println!("\nRecommendations:");
    println!("• Start with BALANCED for most applications");
    println!("• Use CONSERVATIVE for mission-critical operations");
    println!("• Use AGGRESSIVE only when accuracy can be sacrificed for speed");
    println!("• Use HIGH PRECISION with RKN1210 for sub-meter accuracy");
    println!("• Use FIXED-STEP when regular output intervals are required");
}
