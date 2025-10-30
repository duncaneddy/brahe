//! Verify a generated TLE by parsing it back.

#[allow(unused_imports)]
use brahe as bh;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    // Define orbital epoch
    let epoch = bh::Epoch::from_datetime(
        2025, 10, 29, 11, 44, 55.766182, 0.0, bh::TimeSystem::UTC
    );

    // Define ISS orbital elements (angles in degrees)
    let elements = na::SVector::<f64, 6>::new(
        6795445.0,      // Semi-major axis (m)
        0.0004808,      // Eccentricity
        51.6347,        // Inclination (deg)
        1.5519,         // RAAN (deg)
        353.3325,       // Argument of Periapsis (deg)
        6.7599          // Mean Anomaly (deg)
    );

    // Create TLE
    let norad_id = "25544";
    let (line1, line2) = bh::keplerian_elements_to_tle(&epoch, &elements, norad_id).unwrap();

    // Verify by parsing the generated TLE back
    let (parsed_epoch, parsed_elements) = bh::keplerian_elements_from_tle(&line1, &line2).unwrap();

    println!("Verification:");
    println!("Epoch matches: {}", (epoch.jd() - parsed_epoch.jd()).abs() < 1e-9);

    let elements_match = elements.iter()
        .zip(parsed_elements.iter())
        .all(|(a, b)| (a - b).abs() / a.abs().max(1e-10) < 1e-5);
    println!("Elements match: {}", elements_match);

    // Expected output:
    // Verification:
    // Epoch matches: true
    // Elements match: true
}
