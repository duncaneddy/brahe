//! Use standard trajectory operations (length, timespan, interpolation)

#[allow(unused_imports)]
use brahe as bh;
use bh::time::Epoch;
use bh::trajectories::SOrbitTrajectory;
use bh::trajectories::traits::{OrbitFrame, OrbitRepresentation};
use bh::traits::{Trajectory, InterpolatableTrajectory};
use bh::{state_koe_to_eci, R_EARTH, AngleFormat};
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
        let state = state_koe_to_eci(oe, AngleFormat::Radians);
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

