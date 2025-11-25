//! Memory management with maximum age eviction policy

#[allow(unused_imports)]
use brahe as bh;
use bh::time::Epoch;
use bh::trajectories::DTrajectory;
use bh::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Keep only states within last 2 minutes (120 seconds)
    let mut traj = DTrajectory::new(6)
        .with_eviction_policy_max_age(120.0);

    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);

    // Add states spanning 4 minutes
    for i in 0..5 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::DVector::from_vec(vec![
            R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
        ]);
        traj.add(epoch, state);
    }

    // Only states within 120 seconds of the most recent are kept
    println!("Trajectory length: {}", traj.len());
    println!("Timespan: {:.1} seconds", traj.timespan().unwrap());
}

// Output:
// Trajectory length: 3
// Timespan: 120.0 seconds