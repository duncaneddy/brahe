//! Demonstrates querying satellite catalog (SATCAT) data from SpaceTrack.
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

        // Query SATCAT for ISS
        let records = client
            .satcat(
                Some(25544), // norad_cat_id
                None,        // satname
                None,        // intldes
                None,        // object_type
                None,        // country
                None,        // launch
                None,        // current
                Some(1),     // limit
                None,        // orderby
            )
            .expect("Query failed");

        // Records are returned as SATCATRecord structs with attribute access
        if let Some(record) = records.first() {
            println!("Name: {}", record.object_name.as_deref().unwrap_or("Unknown"));
            println!("NORAD ID: {:?}", record.norad_cat_id);
            println!("Intl Designator: {}", record.intldes.as_deref().unwrap_or("Unknown"));
            println!("Launch Date: {}", record.launch.as_deref().unwrap_or("Unknown"));
            println!("Country: {}", record.country.as_deref().unwrap_or("Unknown"));
            println!("Object Type: {}", record.object_type.as_deref().unwrap_or("Unknown"));
            // Name: ISS (ZARYA)
            // NORAD ID: Some(25544)
            // Intl Designator: 1998-067A
            // Launch Date: 1998-11-20
            // Country: ISS
            // Object Type: PAYLOAD
        }
    }

    println!("Example completed successfully!");
}
