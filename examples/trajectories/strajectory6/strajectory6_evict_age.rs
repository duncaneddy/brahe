//! Use maximum age eviction policy to keep only recent states

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Keep only states within last 2 minutes (120 seconds)
    let mut traj = bh::STrajectory6::new()
        .with_eviction_policy_max_age(120.0);

    let epoch0 = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::TimeSystem::UTC);

    // Add states spanning 4 minutes
    for i in 0..5 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
        );
        traj.add(epoch, state);
    }

    // Only states within 120 seconds of the most recent are kept
    println!("Trajectory length: {}", traj.len());
    // Trajectory length: 3

    println!("Timespan: {:.1} seconds", traj.timespan().unwrap());
    // Timespan: 120.0 seconds
}
