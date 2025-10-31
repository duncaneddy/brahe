//! Convert trajectory data to matrix format for analysis

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create trajectory
    let mut traj = bh::STrajectory6::new();
    let epoch0 = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::TimeSystem::UTC);

    for i in 0..3 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0 + (i as f64) * 10.0, 0.0
        );
        traj.add(epoch, state);
    }

    // Convert to matrix (rows are states, columns are dimensions)
    let matrix = traj.to_matrix().unwrap();
    println!("Matrix shape: ({}, {})", matrix.nrows(), matrix.ncols());
    // Matrix shape: (3, 6)

    println!("First state velocity: {:.1} m/s", matrix[(0, 4)]);
    // First state velocity: 7600.0 m/s

    println!("Last state velocity: {:.1} m/s", matrix[(2, 4)]);
    // Last state velocity: 7620.0 m/s
}
