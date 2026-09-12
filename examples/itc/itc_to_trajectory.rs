//! Convert a Modified ITC ephemeris to a trajectory and use it.
//!
//! Builds an OrbitTrajectory in the file's EME2000 frame with the RTN
//! covariance rotated into that frame, interpolates a state between two
//! records, converts the trajectory to ITRF, and computes ground-station
//! access windows over the ephemeris span.

use brahe as bh;
use brahe::traits::*;

const PATH: &str = "test_assets/starlink/MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt";

fn main() {
    bh::initialize_eop().unwrap();

    let itc = bh::itc::ITC::from_file(PATH).unwrap();
    let trajectory = itc.to_trajectory().unwrap();
    let start = trajectory.start_epoch().unwrap();
    let end = trajectory.end_epoch().unwrap();
    println!("Trajectory: {} samples from {} to {}", trajectory.len(), start, end);

    let mid = start + 30.0;
    let state = trajectory.interpolate(&mid).unwrap();
    println!("State at {} [m, m/s]: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]", mid, state[0], state[1], state[2], state[3], state[4], state[5]);
    let covariance = trajectory.covariance_at(mid).unwrap().unwrap();
    println!("Position 1-sigma at {} [m]: [{:.3}, {:.3}, {:.3}]", mid, covariance[(0, 0)].sqrt(), covariance[(1, 1)].sqrt(), covariance[(2, 2)].sqrt());

    let itrf = trajectory.to_itrf().unwrap();
    let ecef = itrf.interpolate(&mid).unwrap();
    println!("ITRF position at {} [m]: [{:.3}, {:.3}, {:.3}]", mid, ecef[0], ecef[1], ecef[2]);

    let station = bh::PointLocation::new(-122.4194, 37.7749, 0.0).unwrap().with_name("San Francisco");
    let constraint = bh::ElevationConstraint::new(Some(10.0), None).unwrap();
    let windows = bh::location_accesses(&station, &trajectory, start, end, &constraint, None, None).unwrap();
    println!("Access windows above 10 deg: {}", windows.len());
    for window in windows.iter().take(3) {
        println!("  {} to {} ({:.1} s)", window.window_open, window.window_close, window.duration());
    }
}
