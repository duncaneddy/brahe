//! Iterate over all epoch-state pairs in a trajectory

#[allow(unused_imports)]
use brahe as bh;
use bh::time::Epoch;
use bh::trajectories::DTrajectory;
use bh::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create and populate trajectory
    let mut traj = DTrajectory::new(6);
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);

    for i in 0..3 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::DVector::from_vec(vec![
            R_EARTH + 500e3 + (i as f64) * 1000.0, 0.0, 0.0, 0.0, 7600.0, 0.0
        ]);
        traj.add(epoch, state);
    }

    // Iterate over all states
    for (epoch, state) in &traj {
        let altitude = (state[0] - R_EARTH) / 1e3;
        println!("Epoch: {}, Altitude: {:.2} km", epoch, altitude);
    }
    
    // Output:
    // Epoch: 2024-01-01 00:00:00.000 UTC, Altitude: 500.00 km
    // Epoch: 2024-01-01 00:01:00.000 UTC, Altitude: 501.00 km
    // Epoch: 2024-01-01 00:02:00.000 UTC, Altitude: 502.00 km
}
