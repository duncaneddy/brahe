//! Demonstrates making a generic SpaceTrack API request.
//!
//! FLAGS = ["CI-ONLY"]

use brahe::spacetrack::{BlockingSpaceTrackClient, TEST_BASE_URL, equals};

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

        // Make a raw API request with custom predicates
        let predicates = [
            ("NORAD_CAT_ID", equals("25544")),
            ("limit", equals("1")),
        ];

        let response = client
            .generic_request("basicspacedata", "gp", &predicates, None)
            .expect("Request failed");

        // Response is raw JSON string
        println!("Response length: {} bytes", response.len());
        println!("Contains ISS: {}", response.contains("ISS"));
        // Response length: 1234 bytes
        // Contains ISS: true
    }

    println!("Example completed successfully!");
}
