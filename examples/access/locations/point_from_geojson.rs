//! Load a PointLocation from a GeoJSON string.
//! Demonstrates GeoJSON Feature format with properties.

#[allow(unused_imports)]
use brahe as bh;
use bh::utils::Identifiable;
use bh::AccessibleLocation;

fn main() {
    bh::initialize_eop().unwrap();

    // GeoJSON Point feature
    let geojson = r#"
    {
        "type": "Feature",
        "properties": {"name": "Svalbard Station"},
        "geometry": {
            "type": "Point",
            "coordinates": [15.4038, 78.2232, 458.0]
        }
    }
    "#;

    // Parse JSON string first
    let json: serde_json::Value = serde_json::from_str(geojson).unwrap();
    let location = bh::PointLocation::from_geojson(&json).unwrap();
    let geodetic = location.center_geodetic();
    println!("Loaded: {}", location.get_name().unwrap_or_default());
    println!("Longitude: {:.4} deg", geodetic[0]);
    println!("Latitude: {:.4} deg", geodetic[1]);
    println!("Altitude: {:.1} m", geodetic[2]);

    // Expected output:
    // Loaded: Svalbard Station
    // Longitude: 15.4038 deg
    // Latitude: 78.2232 deg
    // Altitude: 458.0 m
}
