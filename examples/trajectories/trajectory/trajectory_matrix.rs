//! Convert trajectory data to matrix format for analysis

#[allow(unused_imports)]
use brahe as bh;
use bh::time::Epoch;
use bh::trajectories::DTrajectory;
use bh::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create trajectory
    let mut traj = DTrajectory::new(6);
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);

    for i in 0..3 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::DVector::from_vec(vec![
            R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0 + (i as f64) * 10.0, 0.0
        ]);
        traj.add(epoch, state);
    }

    // Convert to matrix (rows are states, columns are dimensions)
    let matrix = traj.to_matrix().unwrap();
    println!("Matrix shape: ({}, {})", matrix.nrows(), matrix.ncols());
    println!("First state velocity: {:.1} m/s", matrix[(0, 4)]);
}

// Output:
// Matrix shape: (3, 6)
// First state velocity: 7600.0 m/s