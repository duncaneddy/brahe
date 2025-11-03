//! Access trajectory states at specific epochs

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create trajectory with multiple states
    let mut traj = bh::STrajectory6::new();
    let epoch0 = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::TimeSystem::UTC);

    for i in 0..5 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3 + (i as f64) * 1000.0, 0.0, 0.0, 0.0, 7600.0, 0.0
        );
        traj.add(epoch, state);
    }

    // Get nearest state (exact match)
    let query_epoch = epoch0 + 120.0;
    let (_nearest_epoch, nearest_state) = traj.nearest_state(&query_epoch).unwrap();
    println!("Exact match found at altitude: {:.2} km",
        (nearest_state[0] - R_EARTH) / 1e3);

    // Get nearest state (between stored epochs)
    let query_epoch = epoch0 + 125.0;
    let (_nearest_epoch, nearest_state) = traj.nearest_state(&query_epoch).unwrap();
    println!("Nearest state altitude: {:.2} km",
        (nearest_state[0] - R_EARTH) / 1e3);
}

// Output:
// Exact match found at altitude: 502.00 km
// Nearest state altitude: 502.00 km