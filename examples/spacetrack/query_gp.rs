//! Demonstrates querying GP (General Perturbations) data from SpaceTrack.
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

        // Query GP records for ISS (NORAD ID 25544)
        let records = client
            .gp(
                Some(25544), // norad_cat_id
                None,        // object_name
                None,        // object_id
                None,        // epoch
                None,        // object_type
                None,        // country_code
                Some(1),     // limit
                None,        // orderby
            )
            .expect("Query failed");

        // Access record fields as attributes
        if let Some(record) = records.first() {
            println!("Object: {}", record.object_name.as_deref().unwrap_or("Unknown"));
            println!("NORAD ID: {:?}", record.norad_cat_id);
            println!("Epoch: {}", record.epoch.as_deref().unwrap_or("Unknown"));
            println!("Inclination: {:.2}°", record.inclination.unwrap_or(0.0));
            println!("Eccentricity: {:.7}", record.eccentricity.unwrap_or(0.0));
            // Object: ISS (ZARYA)
            // NORAD ID: Some(25544)
            // Epoch: 2024-01-15 12:00:00
            // Inclination: 51.64°
            // Eccentricity: 0.0006703
        }
    }

    println!("Example completed successfully!");
}
