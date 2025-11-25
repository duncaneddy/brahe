//! Use standard trajectory operations (length, timespan, interpolation)

#[allow(unused_imports)]
use brahe as bh;
use bh::time::Epoch;
use bh::trajectories::SOrbitTrajectory;
use bh::trajectories::traits::{OrbitFrame, OrbitRepresentation};
use bh::traits::{Trajectory, InterpolatableTrajectory};
use bh::{state_osculating_to_cartesian, R_EARTH, AngleFormat};
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Create trajectory
    let mut traj = SOrbitTrajectory::new(
        OrbitFrame::ECI,
        OrbitRepresentation::Cartesian,
        None
    );

    // Add states
    let epoch0 = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0,
        bh::time::TimeSystem::UTC);
    for i in 0..10 {
        let epoch = epoch0 + (i as f64) * 60.0;
        let oe = na::SVector::<f64, 6>::new(
            R_EARTH + 500e3, 0.001, 0.9, 1.0, 0.5, (i as f64) * 0.1
        );
        let state = state_osculating_to_cartesian(oe, AngleFormat::Radians);
        traj.add(epoch, state);
    }

    // Query properties
    println!("Length: {}", traj.len());
    println!("Timespan: {:.1} seconds", traj.timespan().unwrap());
    println!("Start epoch: {}", traj.start_epoch().unwrap());
    println!("End epoch: {}", traj.end_epoch().unwrap());

    // Interpolate at intermediate time
    let interp_epoch = epoch0 + 45.0;
    let interp_state = traj.interpolate(&interp_epoch).unwrap();
    println!("\nInterpolated state at {}:", interp_epoch);
    println!("  Position (km): [{}, {}, {}] km",
        interp_state[0] / 1e3, interp_state[1] / 1e3, interp_state[2] / 1e3
    );
    println!("  Velocity (m/s): [{}, {}, {}] m/s",
        interp_state[3], interp_state[4], interp_state[5]
    );

    // Iterate over states
    for (i, (epoch, state)) in traj.into_iter().enumerate().take(2) {
        let pos_mag = state.fixed_rows::<3>(0).norm();
        println!("State {}: Epoch={}, Position magnitude={:.2} km",
            i, epoch, pos_mag / 1e3);
    }
}

// Output:
// Length: 10
// Timespan: 540.0 seconds
// Start epoch: 2024-01-01 00:00:00.000 UTC
// End epoch: 2024-01-01 00:09:00.000 UTC

// Interpolated state at 2024-01-01 00:00:45.000 UTC:
//   Position (km): [1159.0159730226278, 6101.297890257402, 2925.16369357997] km
//   Velocity (m/s): [-5578.867341523014, -1338.7748300095711, 5004.22925363932] m/s
// State 0: Epoch=2024-01-01 00:00:00.000 UTC, Position magnitude=6871.26 km
// State 1: Epoch=2024-01-01 00:01:00.000 UTC, Position magnitude=6871.29 km