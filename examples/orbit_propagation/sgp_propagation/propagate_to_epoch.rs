//! Propagate to a specific target epoch

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::{SStatePropagator, Trajectory};

fn main() {
    bh::initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let mut prop = bh::SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    // Propagate to specific epoch
    let target = prop.epoch + 7200.0;  // 2 hours later
    prop.propagate_to(target);

    println!("Target epoch: {}", target);
    println!("Current epoch: {}", prop.current_epoch());
    println!("Trajectory contains {} states", prop.trajectory.len());

    // Expected output:
    // Target epoch: 2008-09-20 14:25:40.104 UTC
    // Current epoch: 2008-09-20 14:25:40.104 UTC
    // Trajectory contains 121 states
}
