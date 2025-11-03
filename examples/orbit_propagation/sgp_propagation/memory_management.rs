//! Control trajectory memory usage with eviction policies

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::{OrbitPropagator, Trajectory};

fn main() {
    bh::initialize_eop().unwrap();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    let mut prop = bh::SGPPropagator::from_tle(line1, line2, 60.0).unwrap();

    // Keep only 50 most recent states for memory efficiency
    prop.set_eviction_policy_max_size(50).unwrap();

    // Propagate many steps
    prop.propagate_steps(200);
    println!("Trajectory length: {}", prop.trajectory.len());  // Will be 50

    // Alternative: Keep states within 30 minutes of current
    prop.reset();
    prop.set_eviction_policy_max_age(1800.0).unwrap();  // 1800 seconds = 30 minutes
    prop.propagate_steps(200);
    println!("Trajectory length with age policy: {}", prop.trajectory.len());
    // Expected output:
    // Trajectory length: 50
    // Trajectory length with age policy: 31
}
