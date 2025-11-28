//! Overview of integrator configuration presets.

use brahe as bh;

fn main() {
    // Preset configurations for common use cases
    let default = bh::NumericalPropagationConfig::default();
    let high_precision = bh::NumericalPropagationConfig::high_precision();
    let rkf45 = bh::NumericalPropagationConfig::with_method(bh::IntegratorMethod::RKF45);
    let rk4 = bh::NumericalPropagationConfig::with_method(bh::IntegratorMethod::RK4);

    println!("default():          {:?}", default.method);
    println!("high_precision():   {:?}", high_precision.method);
    println!("with_method(RKF45): {:?}", rkf45.method);
    println!("with_method(RK4):   {:?}", rk4.method);
    // default():          DP54
    // high_precision():   RKN1210
    // with_method(RKF45): RKF45
    // with_method(RK4):   RK4
}
