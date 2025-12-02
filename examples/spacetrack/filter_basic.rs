//! Demonstrates basic filtering in SpaceTrack queries.
//!
//! FLAGS = ["CI-ONLY"]

use brahe::spacetrack::{BlockingSpaceTrackClient, TEST_BASE_URL};

fn main() {
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

        // Filter by NORAD catalog ID
        let records = client
            .gp(
                Some(25544), // Filter by ISS NORAD ID
                None,
                None,
                None,
                None,
                None,
                Some(1),
                None,
            )
            .expect("Query failed");

        println!("Found {} records for ISS", records.len());

        // Filter by object type (PAYLOAD, ROCKET BODY, DEBRIS)
        let payload_records = client
            .gp(
                None,
                None,
                None,
                None,
                Some("PAYLOAD"), // Only payloads
                None,
                Some(5),
                None,
            )
            .expect("Query failed");

        println!("Found {} payload records", payload_records.len());
        // Found 1 records for ISS
        // Found 5 payload records
    }

    println!("Example completed successfully!");
}
