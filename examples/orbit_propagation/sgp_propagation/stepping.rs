//! Single and multiple step propagation with SGPPropagator

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::{OrbitPropagator, Trajectory};

fn main() {
    bh::initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let mut prop = bh::SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    // Single step (60 seconds)
    prop.step();
    println!("After 1 step: {}", prop.current_epoch());

    // Multiple steps
    prop.propagate_steps(10);
    println!("After 11 total steps: {} states", prop.trajectory.len());

    // Step by custom duration
    prop.step_by(120.0);
    println!("After custom step: {}", prop.current_epoch());
}

// Output
// After 1 step: 2008-09-20 12:26:40.104 UTC
// After 11 total steps: 12 states
// After custom step: 2008-09-20 12:38:40.104 UTC