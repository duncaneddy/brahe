//! Demonstrates comparison operators in SpaceTrack queries.
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

        // Use comparison operator: greater than
        // The ">" prefix means "greater than"
        let recent_records = client
            .gp(
                None,
                None,
                None,
                Some(">2024-01-01"), // epoch > 2024-01-01
                None,
                None,
                Some(5),
                None,
            )
            .expect("Query failed");

        println!("Found {} records with epoch > 2024-01-01", recent_records.len());

        // Use comparison operator: less than
        // The "<" prefix means "less than"
        let old_records = client
            .gp(
                None,
                None,
                None,
                Some("<2020-01-01"), // epoch < 2020-01-01
                None,
                None,
                Some(5),
                None,
            )
            .expect("Query failed");

        println!("Found {} records with epoch < 2020-01-01", old_records.len());
        // Found 5 records with epoch > 2024-01-01
        // Found 5 records with epoch < 2020-01-01
    }

    println!("Example completed successfully!");
}
