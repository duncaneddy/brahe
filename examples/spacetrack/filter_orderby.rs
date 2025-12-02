//! Demonstrates ordering results in SpaceTrack queries.
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

        // Order by epoch descending (most recent first)
        let recent_iss = client
            .gp(
                Some(25544),
                None,
                None,
                None,
                None,
                None,
                Some(5),
                Some("EPOCH desc"),
            )
            .expect("Query failed");

        println!("ISS GP records (most recent first):");
        for record in &recent_iss {
            println!("  Epoch: {}", record.epoch.as_deref().unwrap_or("Unknown"));
        }

        // Order by launch date ascending (oldest first)
        let oldest_sats = client
            .satcat(
                None,
                None,
                None,
                None,
                None,
                None,
                Some("Y"), // current = Y (still in orbit)
                Some(5),
                Some("LAUNCH asc"),
            )
            .expect("Query failed");

        println!("\nOldest satellites still in orbit:");
        for sat in &oldest_sats {
            println!(
                "  {}: launched {}",
                sat.object_name.as_deref().unwrap_or("Unknown"),
                sat.launch.as_deref().unwrap_or("Unknown")
            );
        }
        // ISS GP records (most recent first):
        //   Epoch: 2024-01-15 12:00:00
        //   Epoch: 2024-01-14 12:00:00
        //   ...
        // Oldest satellites still in orbit:
        //   VANGUARD 1: launched 1958-03-17
        //   ...
    }

    println!("Example completed successfully!");
}
