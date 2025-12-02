//! Demonstrates creating a SpaceTrack client with authentication.
//!
//! FLAGS = ["CI-ONLY"]

use brahe::spacetrack::{BlockingSpaceTrackClient, TEST_BASE_URL};

fn main() {
    // Get credentials from environment variables
    // Note: For production use, most users should use the default constructor:
    //   BlockingSpaceTrackClient::new("user", "password")
    // These examples use the test server for automated testing.
    let user = std::env::var("TEST_SPACETRACK_USER").unwrap_or_default();
    let password = std::env::var("TEST_SPACETRACK_PASS").unwrap_or_default();

    if user.is_empty() || password.is_empty() {
        println!("Set TEST_SPACETRACK_USER and TEST_SPACETRACK_PASS environment variables");
        println!("Register at: https://www.space-track.org/auth/createAccount");
    } else {
        // Create authenticated client using test server
        let client = BlockingSpaceTrackClient::with_base_url(&user, &password, TEST_BASE_URL)
            .expect("Failed to create client");

        // Check authentication status
        println!("Authenticated: {}", client.is_authenticated());
        println!("Base URL: {}", client.base_url());
        // Authenticated: true
        // Base URL: https://for-testing-only.space-track.org/
    }

    println!("Example completed successfully!");
}
