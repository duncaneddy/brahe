//! Propagate KeplerianPropagator to a specific target epoch with precision

#[allow(unused_imports)]
use brahe as bh;
use bh::traits::SStatePropagator;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let elements = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let mut prop = bh::KeplerianPropagator::from_keplerian(
        epoch, elements, bh::AngleFormat::Degrees, 60.0
    );

    // Propagate exactly 500 seconds (not evenly divisible by step size)
    let target = epoch + 500.0;
    prop.propagate_to(target);

    println!("Target epoch: {}", target);
    // Target epoch: 2024-01-01 00:08:20.000 UTC
    println!("Current epoch: {}", prop.current_epoch());
    // Current epoch: 2024-01-01 00:08:20.000 UTC
    println!("Difference: {:.10} seconds",
             (prop.current_epoch() - target).abs());
    // Difference: 0.0000000000 seconds
    // Output shows machine precision agreement
}
