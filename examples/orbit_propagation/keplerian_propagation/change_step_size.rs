//! Change KeplerianPropagator step size during propagation

#[allow(unused_imports)]
use brahe as bh;
use bh::traits::OrbitPropagator;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let elements = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let mut prop = bh::KeplerianPropagator::from_keplerian(
        epoch, elements, bh::AngleFormat::Degrees, 60.0
    );

    println!("Initial step size: {} seconds", prop.step_size());
    // Initial step size: 60 seconds

    // Change step size
    prop.set_step_size(120.0);
    println!("New step size: {} seconds", prop.step_size());
    // New step size: 120 seconds

    // Subsequent steps use new step size
    prop.step();  // Advances 120 seconds
    println!("After step: {:.1} seconds elapsed", prop.current_epoch() - epoch);
    // After step: 120.0 seconds elapsed
}
