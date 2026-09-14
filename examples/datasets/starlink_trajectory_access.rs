//! Download a Starlink satellite's ephemeris, interpolate its trajectory,
//! read its covariance in another frame, and compute ground-station access
//! windows.
//!
//! Uses the manifest and ephemeris cached under the brahe cache directory
//! (seeded by `just seed-starlink-cache`).

use brahe as bh;
use brahe::frames::CelestialFrame;
use brahe::starlink::StarlinkClient;
use brahe::traits::*;

fn main() {
    bh::initialize_eop().unwrap();

    let client = StarlinkClient::with_cache_age(7.0 * 86400.0);
    let manifest = client.get_manifest().unwrap();
    let entry = manifest.find_by_object_name("STARLINK-38128").unwrap();

    let trajectory = client.get_trajectory(entry.norad_cat_id).unwrap();
    let start = trajectory.start_epoch().unwrap();
    let end = trajectory.end_epoch().unwrap();
    println!("Trajectory: {} samples from {} to {} in {}", trajectory.len(), start, end, trajectory.frame);

    let mid = start + 30.0;
    let state = trajectory.interpolate(&mid).unwrap();
    println!("State at {} [m, m/s]: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]", mid, state[0], state[1], state[2], state[3], state[4], state[5]);

    let covariance = trajectory.covariance_in_frame(CelestialFrame::ITRF, mid).unwrap();
    println!("ITRF position 1-sigma at {} [m]: [{:.3}, {:.3}, {:.3}]", mid, covariance[(0, 0)].sqrt(), covariance[(1, 1)].sqrt(), covariance[(2, 2)].sqrt());

    let station = bh::PointLocation::new(-122.4194, 37.7749, 0.0).unwrap().with_name("San Francisco");
    let constraint = bh::ElevationConstraint::new(Some(10.0), None).unwrap();
    let windows = bh::location_accesses(&station, &trajectory, start, end, &constraint, None, None).unwrap();
    println!("Access windows above 10 deg: {}", windows.len());
    for window in windows.iter().take(2) {
        println!("  {} to {} ({:.1} s)", window.window_open, window.window_close, window.duration());
    }
}
