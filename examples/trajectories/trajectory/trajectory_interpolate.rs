//! Linear interpolation to estimate states at arbitrary epochs

#[allow(unused_imports)]
use brahe as bh;
use bh::time::Epoch;
use bh::trajectories::DTrajectory;
use bh::traits::{Trajectory, InterpolatableTrajectory};
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create trajectory with sparse data
    let mut traj = DTrajectory::new(6);
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);

    // Add states every 60 seconds
    for i in 0..3 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::DVector::from_vec(vec![
            R_EARTH + 500e3 + (i as f64) * 10000.0, 0.0, 0.0, 0.0, 7600.0, 0.0
        ]);
        traj.add(epoch, state);
    }

    // Interpolate state at intermediate time
    let query_epoch = epoch0 + 30.0;  // Halfway between first two states
    let interpolated_state = traj.interpolate(&query_epoch).unwrap();

    println!("Interpolated altitude: {:.2} km",
        (interpolated_state[0] - R_EARTH) / 1e3);
    // Expected: approximately 505 km (halfway between 500 and 510 km)
}

// Output:
// Interpolated altitude: 505.00 km