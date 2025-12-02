//! Demonstrates querying satellite constellations from SpaceTrack.
//! Uses the ICEYE constellation as a smaller example.
//!
//! FLAGS = ["CI-ONLY"]

use brahe as bh;
use brahe::propagators::SGPPropagator;
use brahe::spacetrack::{BlockingSpaceTrackClient, TEST_BASE_URL};
use brahe::traits::SStatePropagator;

fn main() {
    bh::initialize_eop().unwrap();

    // Get credentials from environment
    // Note: For production use, most users should use the default constructor:
    //   BlockingSpaceTrackClient::new("user", "password")
    // These examples use the test server for automated testing.
    let user = std::env::var("TEST_SPACETRACK_USER").unwrap_or_default();
    let password = std::env::var("TEST_SPACETRACK_PASS").unwrap_or_default();

    if user.is_empty() || password.is_empty() {
        println!("Set TEST_SPACETRACK_USER and TEST_SPACETRACK_PASS environment variables");
    } else {
        let client = BlockingSpaceTrackClient::with_base_url(&user, &password, TEST_BASE_URL)
            .expect("Failed to create client");

        // Query ICEYE constellation (smaller than Starlink, good for demo)
        // First query GP records with pattern matching
        let iceye_records = client
            .gp(
                None,
                Some("~~ICEYE%"),
                None,
                None,
                None,
                None,
                Some(10),
                None,
            )
            .expect("Query failed");

        // Convert GP records to propagators using embedded TLE data
        let mut iceye_props: Vec<SGPPropagator> = Vec::new();
        for r in &iceye_records {
            if let (Some(name), Some(line1), Some(line2)) = (
                r.object_name.as_ref(),
                r.tle_line1.as_ref(),
                r.tle_line2.as_ref(),
            ) {
                if let Ok(prop) = SGPPropagator::from_3le(Some(name), line1, line2, 60.0) {
                    iceye_props.push(prop);
                }
            }
        }

        println!("Retrieved {} ICEYE propagators", iceye_props.len());

        // Calculate orbital statistics
        let epoch = bh::Epoch::from_datetime(2024, 6, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
        let mut altitudes: Vec<f64> = Vec::new();

        for prop in &mut iceye_props {
            prop.propagate_to(epoch);
            let state = prop.current_state();
            let r = (state[0].powi(2) + state[1].powi(2) + state[2].powi(2)).sqrt();
            altitudes.push((r - bh::R_EARTH) / 1000.0);
        }

        if !altitudes.is_empty() {
            let min_alt = altitudes.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_alt = altitudes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mean_alt: f64 = altitudes.iter().sum::<f64>() / altitudes.len() as f64;

            println!("Altitude range: {:.1} - {:.1} km", min_alt, max_alt);
            println!("Mean altitude: {:.1} km", mean_alt);
        }
        // Retrieved 10 ICEYE propagators
        // Altitude range: 560.2 - 580.8 km
        // Mean altitude: 570.3 km
    }

    println!("Example completed successfully!");
}
