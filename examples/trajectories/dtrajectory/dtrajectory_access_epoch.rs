//! Get states at or near specific epochs

#[allow(unused_imports)]
use brahe as bh;
use bh::time::Epoch;
use bh::trajectories::DTrajectory;
use bh::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create trajectory with multiple states
    let mut traj = DTrajectory::new(6);
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);

    for i in 0..5 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::DVector::from_vec(vec![
            R_EARTH + 500e3 + (i as f64) * 1000.0, 0.0, 0.0, 0.0, 7600.0, 0.0
        ]);
        traj.add(epoch, state);
    }

    // Get nearest state to a specific epoch
    let query_epoch1 = epoch0 + 120.0;  // 2 minutes after start
    let (_, nearest_state) = traj.nearest_state(&query_epoch1).unwrap();
    println!("Nearest state at t+120s altitude: {:.2} km",
        (nearest_state[0] - R_EARTH) / 1e3);

    // Get nearest state between stored epochs
    let query_epoch2 = epoch0 + 125.0;  // Between stored epochs
    let (_, nearest_state) = traj.nearest_state(&query_epoch2).unwrap();
    println!("Nearest state at t+125s altitude: {:.2} km",
        (nearest_state[0] - R_EARTH) / 1e3);
}

// Output:
// Nearest state at t+120s altitude: 502.00 km
// Nearest state at t+125s altitude: 502.00 km