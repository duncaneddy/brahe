//! Demonstrates propagating multiple satellites from SpaceTrack data.
//! Uses the ICEYE constellation for reliable orbital data.
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

        // Fetch multiple satellites - use ICEYE constellation for reliable data
        let gp_records = client
            .gp(
                None,
                Some("~~ICEYE%"),
                None,
                None,
                None,
                None,
                Some(5),
                None,
            )
            .expect("Query failed");

        // Convert GP records to propagators using embedded TLE data
        let mut propagators: Vec<SGPPropagator> = Vec::new();
        for record in &gp_records {
            if let (Some(name), Some(line1), Some(line2)) = (
                record.object_name.as_ref(),
                record.tle_line1.as_ref(),
                record.tle_line2.as_ref(),
            ) {
                if let Ok(prop) = SGPPropagator::from_3le(Some(name), line1, line2, 60.0) {
                    propagators.push(prop);
                }
            }
        }

        println!("Retrieved {} propagators", propagators.len());

        // Propagate all to same epoch
        let epoch = bh::Epoch::from_datetime(2024, 6, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

        for (i, prop) in propagators.iter_mut().enumerate() {
            prop.propagate_to(epoch);
            let state = prop.current_state();
            let r = (state[0].powi(2) + state[1].powi(2) + state[2].powi(2)).sqrt();
            let altitude = (r - bh::R_EARTH) / 1000.0;

            println!("Satellite {}: altitude = {:.1} km", i + 1, altitude);
        }
        // Retrieved 5 propagators
        // Satellite 1: altitude = 570.5 km
        // Satellite 2: altitude = 568.2 km
        // ...
    }

    println!("Example completed successfully!");
}
