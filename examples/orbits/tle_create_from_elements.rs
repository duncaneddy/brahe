//! Create a TLE from orbital elements.

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define orbital epoch
    let epoch = bh::Epoch::from_datetime(
        2025, 10, 29, 11, 44, 55.766182, 0.0, bh::TimeSystem::UTC
    );

    // Define ISS orbital elements
    // Order: [a, e, i, raan, argp, M]
    // Note: Angles must be in DEGREES for TLE creation (exception to library convention)
    let elements = na::SVector::<f64, 6>::new(
        6795445.0,      // Semi-major axis (m)
        0.0004808,      // Eccentricity
        51.6347,        // Inclination (deg)
        1.5519,         // Right Ascension of Ascending Node (deg)
        353.3325,       // Argument of Periapsis (deg)
        6.7599          // Mean Anomaly (deg)
    );

    // Create TLE lines with NORAD ID
    let norad_id = "25544";
    let (line1, line2) = bh::keplerian_elements_to_tle(&epoch, &elements, norad_id).unwrap();

    println!("Generated TLE:");
    println!("{}", line1);
    println!("{}", line2);

}

