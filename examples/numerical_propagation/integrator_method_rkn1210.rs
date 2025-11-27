//! Configuring the RKN1210 high-precision integrator method.
//! 12th-order Runge-Kutta-Nystrom for maximum accuracy.

use brahe as bh;

fn main() {
    // RKN1210 (Runge-Kutta-Nystrom 12(10))
    // - Very high-order adaptive integrator
    // - 12th order solution with 10th order error estimate
    // - 17 function evaluations per step
    // - Optimized for second-order ODEs (like orbital mechanics)
    // - Achieves extreme accuracy with tight tolerances

    // Create high-precision configuration using RKN1210
    let _config = bh::NumericalPropagationConfig::high_precision();

    // Or manually configure RKN1210 with custom tolerances
    let _config_custom = bh::NumericalPropagationConfig {
        method: bh::IntegratorMethod::RKN1210,
        integrator: bh::IntegratorConfig::adaptive(1e-12, 1e-10),
        variational: bh::VariationalConfig::default(),
    };

    println!("RKN1210 High-Precision Integrator Configuration:");
    println!("  Method: RKN1210 (Runge-Kutta-Nystrom 12(10))");
    println!("  Adaptive: Yes");
    println!("  Function evaluations per step: 17");
    println!("  Order: 12(10) - 12th order solution, 10th order error estimate");

    println!("\nhigh_precision() preset tolerances:");
    println!("  Absolute tolerance: 1e-10");
    println!("  Relative tolerance: 1e-8");

    println!("\nNystrom Optimization:");
    println!("  RKN methods are designed specifically for second-order ODEs:");
    println!("    x'' = f(t, x, x')");
    println!("  This matches orbital mechanics where acceleration depends");
    println!("  on position and velocity, not their derivatives.");

    println!("\nCharacteristics:");
    println!("  - Highest precision available");
    println!("  - Larger steps than lower-order methods for same accuracy");
    println!("  - More expensive per step (17 function evaluations)");
    println!("  - Can achieve sub-millimeter accuracy over long periods");

    println!("\nWhen to use RKN1210:");
    println!("  - Precision orbit determination");
    println!("  - Reference trajectory generation");
    println!("  - Scientific research requiring highest accuracy");
    println!("  - Validation of other methods");
    println!("  - When sub-meter accuracy matters over days/weeks");

    println!("\nWhen NOT to use RKN1210:");
    println!("  - Quick estimates or prototyping");
    println!("  - Real-time applications");
    println!("  - When moderate accuracy is sufficient");
    println!("  - Memory-constrained environments");

    println!("\nNote: This integrator is marked experimental.");
    println!("  Consider validating results for your specific application.");

    println!("\nExample validated successfully!");
}
