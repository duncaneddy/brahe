//! Iterate over all epoch-state pairs in a trajectory

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create and populate trajectory
    let mut traj = bh::STrajectory6::new();
    let epoch0 = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::TimeSystem::UTC);

    for i in 0..3 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3 + (i as f64) * 1000.0, 0.0, 0.0, 0.0, 7600.0, 0.0
        );
        traj.add(epoch, state);
    }

    // Iterate over all states
    for (epoch, state) in &traj {
        let altitude = (state[0] - R_EARTH) / 1e3;
        let velocity = state.fixed_rows::<3>(3).norm();
        println!("Epoch: {}, Altitude: {:.2} km, Speed: {:.0} m/s",
            epoch, altitude, velocity);
    }
    // Epoch: 2024-01-01 00:00:00.000 UTC, Altitude: 500.00 km, Speed: 7600 m/s
    // Epoch: 2024-01-01 00:01:00.000 UTC, Altitude: 501.00 km, Speed: 7600 m/s
    // Epoch: 2024-01-01 00:02:00.000 UTC, Altitude: 502.00 km, Speed: 7600 m/s
}
