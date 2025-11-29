//! Generate a basic Walker Delta constellation with Keplerian propagators.
//!
//! This example demonstrates creating a GPS-like 24:6:2 Walker Delta constellation
//! using the WalkerConstellationGenerator and generating Keplerian propagators.

#[allow(unused_imports)]
use brahe as bh;
use bh::utils::Identifiable;

fn main() {
    bh::initialize_eop().unwrap();

    // Create epoch for constellation
    let epoch = bh::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Create a GPS-like 24:6:2 Walker Delta constellation
    // T:P:F = 24:6:2 means:
    //   - T = 24 total satellites
    //   - P = 6 orbital planes
    //   - F = 2 phasing factor
    let walker = bh::orbits::WalkerConstellationGenerator::new(
        24,   // T = 24 total satellites
        6,    // P = 6 orbital planes
        2,    // F = 2 phasing factor
        bh::constants::R_EARTH + 20200e3, // GPS altitude
        0.0,  // eccentricity
        55.0, // inclination (degrees)
        0.0,  // argument of perigee
        0.0,  // reference RAAN
        0.0,  // reference mean anomaly
        epoch,
        bh::constants::AngleFormat::Degrees,
        bh::orbits::WalkerPattern::Delta,
    )
    .with_base_name("GPS");

    // Print constellation properties
    println!("Total satellites: {}", walker.total_satellites);
    println!("Number of planes: {}", walker.num_planes);
    println!("Satellites per plane: {}", walker.satellites_per_plane());
    println!("Phasing factor: {}", walker.phasing);
    println!("Pattern: {:?}", walker.pattern);

    // Get orbital elements for the first satellite in each plane
    println!("\nFirst satellite in each plane:");
    for plane in 0..walker.num_planes {
        let elements = walker.satellite_elements(plane, 0);
        let raan_deg = elements[3] * bh::constants::RAD2DEG;
        let ma_deg = elements[5] * bh::constants::RAD2DEG;
        println!("  Plane {}: RAAN = {:.1} deg, MA = {:.1} deg", plane, raan_deg, ma_deg);
    }

    // Generate Keplerian propagators for all satellites
    let propagators = walker.as_keplerian_propagators(60.0); // 60 second step size
    println!("\nGenerated {} Keplerian propagators", propagators.len());
    println!("First propagator name: {}", propagators[0].get_name().unwrap_or_default());
    println!("Last propagator name: {}", propagators.last().unwrap().get_name().unwrap_or_default());

    // Expected output:
    // Total satellites: 24
    // Number of planes: 6
    // Satellites per plane: 4
    // Phasing factor: 2
    // Pattern: Delta
    //
    // First satellite in each plane:
    //   Plane 0: RAAN = 0.0 deg, MA = 0.0 deg
    //   Plane 1: RAAN = 60.0 deg, MA = 30.0 deg
    //   Plane 2: RAAN = 120.0 deg, MA = 60.0 deg
    //   Plane 3: RAAN = 180.0 deg, MA = 90.0 deg
    //   Plane 4: RAAN = 240.0 deg, MA = 120.0 deg
    //   Plane 5: RAAN = 300.0 deg, MA = 150.0 deg
    //
    // Generated 24 Keplerian propagators
    // First propagator name: GPS-P0-S0
    // Last propagator name: GPS-P5-S3
}
