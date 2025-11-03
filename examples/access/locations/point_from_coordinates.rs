//! Create a PointLocation from geodetic coordinates (longitude, latitude, altitude).
//! Demonstrates basic construction and naming.

#[allow(unused_imports)]
use brahe as bh;
use bh::utils::Identifiable;
use bh::AccessibleLocation;

fn main() {
    bh::initialize_eop().unwrap();

    // Create location (longitude, latitude, altitude in meters)
    // San Francisco, CA
    let sf = bh::PointLocation::new(
        -122.4194,  // longitude in degrees
        37.7749,    // latitude in degrees
        0.0         // altitude in meters
    ).with_name("San Francisco");

    let geodetic = sf.center_geodetic();
    println!("Location: {}", sf.get_name().unwrap_or_default());
    println!("Longitude: {:.4} deg", geodetic[0]);
    println!("Latitude: {:.4} deg", geodetic[1]);

    // Expected output:
    // Location: San Francisco
    // Longitude: -122.4194 deg
    // Latitude: 37.7749 deg
}
