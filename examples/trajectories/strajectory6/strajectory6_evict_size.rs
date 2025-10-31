//! Use maximum size eviction policy to limit trajectory size

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create trajectory with max size limit
    let mut traj = bh::STrajectory6::new()
        .with_eviction_policy_max_size(3);

    let epoch0 = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::TimeSystem::UTC);

    // Add 5 states
    for i in 0..5 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3 + (i as f64) * 1000.0, 0.0, 0.0, 0.0, 7600.0, 0.0
        );
        traj.add(epoch, state);
    }

    // Only the 3 most recent states are kept
    println!("Trajectory length: {}", traj.len());
    // Trajectory length: 3

    println!("Start altitude: {:.2} km",
        (traj.state_at_idx(0).unwrap()[0] - R_EARTH) / 1e3);
    // Start altitude: 502.00 km (states 0 and 1 were evicted)
}
