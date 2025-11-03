//! Access and iterate over the KeplerianPropagator's internal trajectory

#[allow(unused_imports)]
use brahe as bh;
use bh::traits::{OrbitPropagator, Trajectory};
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let elements = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let mut prop = bh::KeplerianPropagator::from_keplerian(
        epoch, elements, bh::AngleFormat::Degrees, 60.0
    );

    // Propagate for several steps
    prop.propagate_steps(5);

    // Access trajectory
    let traj = &prop.trajectory;
    println!("Trajectory contains {} states", traj.len());
    // Trajectory contains 6 states

    // Access by index
    for i in 0..traj.len() {
        let epoch = traj.epoch_at_idx(i).unwrap();
        let state = traj.state_at_idx(i).unwrap();
        println!("Epoch: {}, semi-major axis: {:.1} km", epoch, state[0] / 1e3);
    }
    // Epoch: 2024-01-01 00:00:00.000 UTC, semi-major axis: 6878.1 km
    // Epoch: 2024-01-01 00:01:00.000 UTC, semi-major axis: 6878.1 km
    // Epoch: 2024-01-01 00:02:00.000 UTC, semi-major axis: 6878.1 km
    // Epoch: 2024-01-01 00:03:00.000 UTC, semi-major axis: 6878.1 km
    // Epoch: 2024-01-01 00:04:00.000 UTC, semi-major axis: 6878.1 km
    // Epoch: 2024-01-01 00:05:00.000 UTC, semi-major axis: 6878.1 km
}
