//! Memory management with maximum size eviction policy

#[allow(unused_imports)]
use brahe as bh;
use bh::time::Epoch;
use bh::trajectories::DTrajectory;
use bh::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create trajectory with max size limit
    let mut traj = DTrajectory::new(6)
        .with_eviction_policy_max_size(3);

    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);

    // Add 5 states
    for i in 0..5 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::DVector::from_vec(vec![
            R_EARTH + 500e3 + (i as f64) * 1000.0, 0.0, 0.0, 0.0, 7600.0, 0.0
        ]);
        traj.add(epoch, state);
    }

    // Only the 3 most recent states are kept
    println!("Trajectory length: {}", traj.len());
    println!("Start epoch: {}", traj.start_epoch().unwrap());
    println!("Start altitude: {:.2} km",
        (traj.state_at_idx(0).unwrap()[0] - R_EARTH) / 1e3);
    // Output: ~502 km (states 0 and 1 were evicted)
}

// Output
// Trajectory length: 3
// Start epoch: 2024-01-01 00:02:00.000 UTC
// Start altitude: 502.00 km