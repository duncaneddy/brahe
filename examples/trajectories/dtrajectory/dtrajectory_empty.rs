//! Create empty DTrajectory instances with different dimensions

#[allow(unused_imports)]
use brahe as bh;
use bh::trajectories::DTrajectory;
use bh::traits::Trajectory;

fn main() {
    bh::initialize_eop().unwrap();

    // Create 6D trajectory (default)
    let traj = DTrajectory::default();
    println!("Dimension: {}", traj.dimension());
    // Output: 6

    // Create 3D trajectory
    let traj_3d = DTrajectory::new(3);
    println!("Dimension: {}", traj_3d.dimension());
    // Output: 3

    // Create 12D trajectory
    let traj_12d = DTrajectory::new(12);
    println!("Dimension: {}", traj_12d.dimension());
    // Output: 12
}
