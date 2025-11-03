//! Load a PolygonLocation from a GeoJSON string.
//! Demonstrates GeoJSON Polygon format with nested coordinate arrays.

#[allow(unused_imports)]
use brahe as bh;
use bh::utils::Identifiable;
use bh::AccessibleLocation;

fn main() {
    bh::initialize_eop().unwrap();

    let geojson = r#"
    {
        "type": "Feature",
        "properties": {"name": "Target Area"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-122.5, 37.7, 0],
                [-122.35, 37.7, 0],
                [-122.35, 37.8, 0],
                [-122.5, 37.8, 0],
                [-122.5, 37.7, 0]
            ]]
        }
    }
    "#;

    // Parse JSON string first
    let json: serde_json::Value = serde_json::from_str(geojson).unwrap();
    let polygon = bh::PolygonLocation::from_geojson(&json).unwrap();

    let center = polygon.center_geodetic();
    println!("Name: {}", polygon.get_name().unwrap_or_default());
    println!("Vertices: {}", polygon.num_vertices());
    println!("Center: ({:.4}, {:.4})", center[0], center[1]);

    // Expected output:
    // Name: Target Area
    // Vertices: 4
    // Center: (-122.4250, 37.7500)
}
