//! Generate a Walker Star constellation with Keplerian propagators.
//!
//! This example demonstrates creating an Iridium-like 66:6:2 Walker Star constellation.
//! Walker Star uses a 180 degree RAAN spread (vs 360 for Walker Delta), suitable for
//! polar coverage patterns.

#[allow(unused_imports)]
use brahe as bh;

fn main() {
    bh::initialize_eop().unwrap();

    // Create epoch for constellation
    let epoch = bh::time::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);

    // Create an Iridium-like 66:6:2 Walker Star constellation
    // Walker Star uses 180 degree RAAN spread (vs 360 for Delta)
    let walker = bh::orbits::WalkerConstellationGenerator::new(
        66,   // T = 66 total satellites
        6,    // P = 6 orbital planes
        2,    // F = 2 phasing factor
        bh::constants::R_EARTH + 780e3, // Iridium altitude (~780 km)
        0.0,  // eccentricity
        86.4, // near-polar inclination (degrees)
        0.0,  // argument of perigee
        0.0,  // reference RAAN
        0.0,  // reference mean anomaly
        epoch,
        bh::constants::AngleFormat::Degrees,
        bh::orbits::WalkerPattern::Star, // Star pattern uses 180 deg RAAN spread
    )
    .with_base_name("IRIDIUM");

    // Print constellation properties
    println!("Total satellites: {}", walker.total_satellites);
    println!("Number of planes: {}", walker.num_planes);
    println!("Satellites per plane: {}", walker.satellites_per_plane());
    println!("Phasing factor: {}", walker.phasing);
    println!("Pattern: {:?}", walker.pattern);

    // Show RAAN spacing difference from Walker Delta
    // Walker Star: 180 / P = 180 / 6 = 30 degree spacing
    // Walker Delta: 360 / P = 360 / 6 = 60 degree spacing
    println!("\nFirst satellite in each plane (Walker Star):");
    for plane in 0..walker.num_planes {
        let elements = walker.satellite_elements(plane, 0);
        let raan_deg = elements[3] * bh::constants::RAD2DEG;
        println!("  Plane {}: RAAN = {:.1} deg", plane, raan_deg);
    }

    // Compare with what Walker Delta would give
    println!("\nRemark: Walker Delta with same P=6 would have 60 deg RAAN spacing");
    println!("Walker Star spreads planes over 180 deg (0-150 deg)");
    println!("Walker Delta spreads planes over 360 deg (0-300 deg)");

    // Generate Keplerian propagators
    let propagators = walker.as_keplerian_propagators(60.0);
    println!("\nGenerated {} Keplerian propagators", propagators.len());

    // Expected output:
    // Total satellites: 66
    // Number of planes: 6
    // Satellites per plane: 11
    // Phasing factor: 2
    // Pattern: Star
    //
    // First satellite in each plane (Walker Star):
    //   Plane 0: RAAN = 0.0 deg
    //   Plane 1: RAAN = 30.0 deg
    //   Plane 2: RAAN = 60.0 deg
    //   Plane 3: RAAN = 90.0 deg
    //   Plane 4: RAAN = 120.0 deg
    //   Plane 5: RAAN = 150.0 deg
    //
    // Remark: Walker Delta with same P=6 would have 60 deg RAAN spacing
    // Walker Star spreads planes over 180 deg (0-150 deg)
    // Walker Delta spreads planes over 360 deg (0-300 deg)
    //
    // Generated 66 Keplerian propagators
}
