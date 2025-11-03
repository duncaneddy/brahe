//! Query trajectory properties like timespan, start, and end epochs

#[allow(unused_imports)]
use brahe as bh;
use brahe::traits::Trajectory;
use bh::constants::R_EARTH;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create trajectory spanning 5 minutes
    let mut traj = bh::STrajectory6::new();
    let epoch0 = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::TimeSystem::UTC);

    for i in 0..6 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let state = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0
        );
        traj.add(epoch, state);
    }

    // Query properties
    println!("Number of states: {}", traj.len());
    // Number of states: 6

    println!("Start epoch: {}", traj.start_epoch().unwrap());
    println!("End epoch: {}", traj.end_epoch().unwrap());
    println!("Timespan: {:.1} seconds", traj.timespan().unwrap());
    // Timespan: 300.0 seconds

    println!("Is empty: {}", traj.is_empty());
    // Is empty: false

    // Access first and last states
    let (first_epoch, _first_state) = traj.first().unwrap();
    let (last_epoch, _last_state) = traj.last().unwrap();
    println!("First epoch: {}", first_epoch);
    println!("Last epoch: {}", last_epoch);
}

// Output:
// Number of states: 6
// Start epoch: 2024-01-01 00:00:00.000 UTC
// End epoch: 2024-01-01 00:05:00.000 UTC
// Timespan: 300.0 seconds
// Is empty: false
// First epoch: 2024-01-01 00:00:00.000 UTC
// Last epoch: 2024-01-01 00:05:00.000 UTC