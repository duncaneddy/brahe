//! Create empty DTrajectory instances with different dimensions

#[allow(unused_imports)]
use brahe as bh;
use bh::trajectories::DTrajectory;

fn main() {
    bh::initialize_eop().unwrap();

    // Create 6D trajectory (default)
    let traj = DTrajectory::default();
    println!("Dimension: {}", traj.dimension());

    // Create 3D trajectory
    let traj_3d = DTrajectory::new(3);
    println!("Dimension: {}", traj_3d.dimension());

    // Create 12D trajectory
    let traj_12d = DTrajectory::new(12);
    println!("Dimension: {}", traj_12d.dimension());
}

