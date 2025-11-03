//! Demonstrate various methods for accessing PointLocation coordinates.
//! Shows degree accessors, radian conversion, and ECEF conversion.

#[allow(unused_imports)]
use brahe as bh;
use bh::AccessibleLocation;

fn main() {
    bh::initialize_eop().unwrap();

    let location = bh::PointLocation::new(
        -122.4194,
        37.7749,
        0.0
    );

    // Access geodetic coordinates (in degrees)
    let geodetic = location.center_geodetic();
    println!("Longitude: {:.4} deg", geodetic[0]);
    println!("Latitude: {:.4} deg", geodetic[1]);
    println!("Altitude: {:.1} m", geodetic[2]);


    // Get ECEF Cartesian position [x, y, z] in meters
    let ecef = location.center_ecef();
    println!("ECEF: [{:.1}, {:.1}, {:.1}] m", ecef[0], ecef[1], ecef[2]);

    // Expected output:
    // Longitude: -122.4194 deg
    // Latitude: 37.7749 deg
    // Altitude: 0.0 m
    // ECEF: [-2706174.8, -4261059.5, 3885725.5] m
}
