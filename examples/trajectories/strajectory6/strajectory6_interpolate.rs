//! Interpolate trajectory states at arbitrary epochs

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::{Trajectory, InterpolatableTrajectory};
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create trajectory with sparse data
    let mut traj = bh::STrajectory6::new();
    let epoch0 = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::TimeSystem::UTC);

    // Add states every 60 seconds
    for i in 0..3 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3 + (i as f64) * 10000.0, 0.0, 0.0, 0.0, 7600.0, 0.0
        );
        traj.add(epoch, state);
    }

    // Interpolate state at intermediate time
    let query_epoch = epoch0 + 30.0;
    let interpolated_state = traj.interpolate(&query_epoch).unwrap();

    println!("Interpolated altitude: {:.2} km",
        (interpolated_state[0] - R_EARTH) / 1e3);
    // Interpolated altitude: 505.00 km
}
