//! Create an empty 6D trajectory

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::Trajectory;

fn main() {
    bh::initialize_eop().unwrap();

    // Create empty 6D trajectory
    let traj = bh::STrajectory6::new();
    println!("Trajectory length: {}", traj.len());
    // Trajectory length: 0

    println!("Is empty: {}", traj.is_empty());
    // Is empty: true
}
