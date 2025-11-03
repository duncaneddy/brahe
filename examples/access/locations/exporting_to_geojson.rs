//! Convert locations to GeoJSON format.
//! Demonstrates roundtrip export with names and IDs.

#[allow(unused_imports)]
use brahe as bh;
use bh::utils::Identifiable;
use bh::AccessibleLocation;

fn main() {
    bh::initialize_eop().unwrap();

    let location = bh::PointLocation::new(-122.4194, 37.7749, 0.0)
        .with_name("San Francisco")
        .with_id(1);

    // Export to GeoJSON
    let geojson = location.to_geojson();
    println!("Exported GeoJSON:");
    println!("{}", geojson);

    // The output includes all properties and identifiers
    // Can be loaded back with from_geojson()
    let reloaded = bh::PointLocation::from_geojson(&geojson).unwrap();
    println!("\nReloaded: {} (ID: {})",
        reloaded.get_name().unwrap_or_default(),
        reloaded.get_id().unwrap_or(0));

    // Expected output:
    // Exported GeoJSON:
    // {"geometry":{"coordinates":[-122.4194,37.7749,0.0],"type":"Point"},"properties":{"id":1,"name":"San Francisco"},"type":"Feature"}
    //
    // Reloaded: San Francisco (ID: 1)
}
