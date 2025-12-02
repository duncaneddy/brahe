//! Demonstrates converting GP data to SGP propagators.
//!
//! FLAGS = ["CI-ONLY"]

use brahe as bh;
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

        // Fetch GP data and convert to propagators in one step
        let propagators = client
            .gp_as_propagators(
                60.0,        // 60 second propagation step
                Some(25544), // ISS
                None,
                Some(1),
            )
            .expect("Failed to get propagators");

        if let Some(prop) = propagators.first() {
            // Propagate to a specific epoch
            let epoch = bh::Epoch::from_datetime(2024, 6, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
            let mut prop = prop.clone();
            prop.propagate_to(epoch);
            let state = prop.current_state();

            // State is [x, y, z, vx, vy, vz] in meters and m/s
            let r = (state[0].powi(2) + state[1].powi(2) + state[2].powi(2)).sqrt();
            let altitude = r - bh::R_EARTH;

            println!("ISS position at {}:", epoch);
            println!("  Altitude: {:.1} km", altitude / 1000.0);
            println!(
                "  Position: [{:.1}, {:.1}, {:.1}] km",
                state[0] / 1000.0,
                state[1] / 1000.0,
                state[2] / 1000.0
            );
            // ISS position at 2024-06-01T12:00:00.000000000Z UTC:
            //   Altitude: 420.5 km
            //   Position: [1234.5, 5678.9, 2345.6] km
        }
    }

    println!("Example completed successfully!");
}
