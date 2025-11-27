//! Configuring the RK4 fixed-step integrator method.
//! Classic 4th-order Runge-Kutta for simple problems.

use brahe as bh;

fn main() {
    // RK4 (Classical Runge-Kutta 4th order)
    // - Fixed step size (no adaptive error control)
    // - 4 function evaluations per step
    // - Good for simple problems or when you need predictable timing
    // - Requires careful step size selection

    // Create configuration with RK4 method
    let _config = bh::NumericalPropagationConfig::with_method(bh::IntegratorMethod::RK4);

    // For fixed-step integrators, create config with fixed step
    let _config_with_step = bh::NumericalPropagationConfig {
        method: bh::IntegratorMethod::RK4,
        integrator: bh::IntegratorConfig::fixed_step(60.0), // 60 second steps
        variational: bh::VariationalConfig::default(),
    };

    println!("RK4 Fixed-Step Integrator Configuration:");
    println!("  Method: RK4 (Classical 4th-order Runge-Kutta)");
    println!("  Adaptive: No (fixed step size)");
    println!("  Function evaluations per step: 4");
    println!("  Order: 4");

    println!("\nCharacteristics:");
    println!("  - Predictable step timing");
    println!("  - No error estimation/control");
    println!("  - Fast per-step computation");
    println!("  - Requires manual step size tuning");

    println!("\nWhen to use RK4:");
    println!("  - Simple dynamical systems");
    println!("  - When consistent timing matters more than accuracy");
    println!("  - Initial prototyping and testing");
    println!("  - Educational/demonstration purposes");

    println!("\nWhen NOT to use RK4:");
    println!("  - High-precision requirements");
    println!("  - Stiff differential equations");
    println!("  - Long-duration propagations");
    println!("  - When dynamics vary significantly over orbit");

    println!("\nStep size guidelines for orbital mechanics:");
    println!("  - LEO: 30-60 seconds");
    println!("  - MEO: 60-120 seconds");
    println!("  - GEO: 120-300 seconds");
    println!("  - Highly eccentric: Smaller near perigee");

    println!("\nExample validated successfully!");
}
