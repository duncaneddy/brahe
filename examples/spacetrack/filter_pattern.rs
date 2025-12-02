//! Demonstrates pattern matching in SpaceTrack queries.
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

        // Pattern matching with LIKE operator (~~)
        // Use %% for wildcard matching
        let starlink_records = client
            .gp(
                None,
                Some("~~STARLINK%"), // Names starting with STARLINK
                None,
                None,
                None,
                None,
                Some(5),
                None,
            )
            .expect("Query failed");

        println!("Found {} STARLINK satellites", starlink_records.len());

        // Starts with operator (^)
        let iss_records = client
            .gp(
                None,
                Some("^ISS"), // Names starting with ISS
                None,
                None,
                None,
                None,
                Some(5),
                None,
            )
            .expect("Query failed");

        println!("Found {} ISS-related objects", iss_records.len());
        // Found 5 STARLINK satellites
        // Found 5 ISS-related objects
    }

    println!("Example completed successfully!");
}
