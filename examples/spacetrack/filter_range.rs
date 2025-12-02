//! Demonstrates range queries in SpaceTrack.
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

        // Date range query using "--" syntax
        let records = client
            .gp(
                None,
                None,
                None,
                Some("2024-01-01--2024-06-30"), // epoch between dates
                None,
                None,
                Some(10),
                None,
            )
            .expect("Query failed");

        println!(
            "Found {} records between 2024-01-01 and 2024-06-30",
            records.len()
        );
        // Found 10 records between 2024-01-01 and 2024-06-30
    }

    println!("Example completed successfully!");
}
