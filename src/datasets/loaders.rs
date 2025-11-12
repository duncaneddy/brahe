/*!
 * Generic GeoJSON data loading utilities
 *
 * Provides common functionality for loading location data from GeoJSON files.
 */

use crate::access::location::PointLocation;
use crate::utils::errors::BraheError;
use serde_json::Value as JsonValue;
use std::fs;
use std::path::Path;

/// Load PointLocations from a GeoJSON FeatureCollection file
///
/// Parses a GeoJSON file and extracts all Point features as PointLocation objects.
///
/// # Arguments
/// * `filepath` - Path to GeoJSON file
///
/// # Returns
/// * `Result<Vec<PointLocation>, BraheError>` - Vector of PointLocation objects
///
/// # Errors
/// * File not found or cannot be read
/// * Invalid JSON format
/// * Invalid GeoJSON structure (not a FeatureCollection)
/// * Features with non-Point geometry (skipped with warning)
///
/// # Example
/// ```no_run
/// use brahe::datasets::loaders::load_point_locations_from_geojson;
///
/// let locations = load_point_locations_from_geojson("stations.geojson").unwrap();
/// ```
pub fn load_point_locations_from_geojson(filepath: &str) -> Result<Vec<PointLocation>, BraheError> {
    // Read and parse file
    let contents = fs::read_to_string(Path::new(filepath))
        .map_err(|e| BraheError::IoError(format!("Failed to read file {}: {}", filepath, e)))?;

    let geojson: JsonValue = serde_json::from_str(&contents)
        .map_err(|e| BraheError::ParseError(format!("Invalid JSON: {}", e)))?;

    // Parse from GeoJSON
    parse_point_locations_from_geojson(&geojson)
}

/// Parse PointLocations from a GeoJSON Value
///
/// Internal function to parse GeoJSON data that's already loaded.
///
/// # Arguments
/// * `geojson` - GeoJSON Value (should be a FeatureCollection)
///
/// # Returns
/// * `Result<Vec<PointLocation>, BraheError>` - Vector of PointLocation objects
///
/// # Example
/// ```
/// use brahe::datasets::loaders::parse_point_locations_from_geojson;
/// use serde_json::json;
///
/// let geojson = json!({
///     "type": "FeatureCollection",
///     "features": [
///         {
///             "type": "Feature",
///             "geometry": {
///                 "type": "Point",
///                 "coordinates": [15.4, 78.2, 0.0]
///             },
///             "properties": {
///                 "name": "Svalbard"
///             }
///         }
///     ]
/// });
///
/// let locations = parse_point_locations_from_geojson(&geojson).unwrap();
/// assert_eq!(locations.len(), 1);
/// ```
pub fn parse_point_locations_from_geojson(
    geojson: &JsonValue,
) -> Result<Vec<PointLocation>, BraheError> {
    // Validate it's a FeatureCollection
    if geojson.get("type").and_then(|t| t.as_str()) != Some("FeatureCollection") {
        return Err(BraheError::ParseError(
            "GeoJSON must be a FeatureCollection".to_string(),
        ));
    }

    // Extract features
    let features = geojson
        .get("features")
        .and_then(|f| f.as_array())
        .ok_or_else(|| BraheError::ParseError("Missing or invalid 'features' array".to_string()))?;

    if features.is_empty() {
        return Err(BraheError::ParseError(
            "FeatureCollection contains no features".to_string(),
        ));
    }

    // Parse each Point feature
    let mut locations = Vec::new();
    for (idx, feature) in features.iter().enumerate() {
        // Check geometry type
        let geom_type = feature
            .get("geometry")
            .and_then(|g| g.get("type"))
            .and_then(|t| t.as_str());

        match geom_type {
            Some("Point") => match PointLocation::from_geojson(feature) {
                Ok(loc) => locations.push(loc),
                Err(e) => {
                    eprintln!("Warning: Skipping feature {} due to error: {}", idx, e);
                }
            },
            Some(other) => {
                eprintln!(
                    "Warning: Skipping feature {} with non-Point geometry: {}",
                    idx, other
                );
            }
            None => {
                eprintln!(
                    "Warning: Skipping feature {} with missing geometry type",
                    idx
                );
            }
        }
    }

    if locations.is_empty() {
        return Err(BraheError::ParseError(
            "No valid Point features found in FeatureCollection".to_string(),
        ));
    }

    Ok(locations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_single_point() {
        let geojson = json!({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [15.4, 78.2, 0.0]
                    },
                    "properties": {
                        "name": "Svalbard"
                    }
                }
            ]
        });

        let locations = parse_point_locations_from_geojson(&geojson).unwrap();
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0].lon(), 15.4);
        assert_eq!(locations[0].lat(), 78.2);
        assert_eq!(locations[0].alt(), 0.0);
    }

    #[test]
    fn test_parse_multiple_points() {
        let geojson = json!({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [15.4, 78.2]
                    },
                    "properties": {
                        "name": "Station 1"
                    }
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [-122.4, 37.8, 100.0]
                    },
                    "properties": {
                        "name": "Station 2"
                    }
                }
            ]
        });

        let locations = parse_point_locations_from_geojson(&geojson).unwrap();
        assert_eq!(locations.len(), 2);
    }

    #[test]
    fn test_parse_empty_feature_collection() {
        let geojson = json!({
            "type": "FeatureCollection",
            "features": []
        });

        let result = parse_point_locations_from_geojson(&geojson);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_type() {
        let geojson = json!({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [15.4, 78.2]
            }
        });

        let result = parse_point_locations_from_geojson(&geojson);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_mixed_geometries() {
        // Should skip non-Point geometries and still return valid Points
        let geojson = json!({
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [15.4, 78.2]
                    },
                    "properties": {
                        "name": "Valid Point"
                    }
                },
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]
                    },
                    "properties": {
                        "name": "Invalid Polygon"
                    }
                }
            ]
        });

        let locations = parse_point_locations_from_geojson(&geojson).unwrap();
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0].lon(), 15.4);
    }

    #[test]
    fn test_load_from_nonexistent_file() {
        // Test error when trying to load from a file that doesn't exist
        let result = load_point_locations_from_geojson("/nonexistent/path/to/file.geojson");
        assert!(result.is_err());

        // Verify it's an IoError
        match result {
            Err(BraheError::IoError(msg)) => {
                assert!(msg.contains("Failed to read file"));
            }
            _ => panic!("Expected IoError"),
        }
    }

    #[test]
    fn test_load_from_invalid_json_file() {
        // Test error when file contains invalid JSON
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a temporary file with invalid JSON
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{{ this is not valid JSON }}").unwrap();
        let temp_path = temp_file.path().to_str().unwrap();

        let result = load_point_locations_from_geojson(temp_path);
        assert!(result.is_err());

        // Verify it's a ParseError
        match result {
            Err(BraheError::ParseError(msg)) => {
                assert!(msg.contains("Invalid JSON"));
            }
            _ => panic!("Expected ParseError for invalid JSON"),
        }
    }

    #[test]
    fn test_load_from_valid_file() {
        // Test successfully loading from a valid GeoJSON file
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a temporary file with valid GeoJSON
        let geojson_content = r#"{
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [15.4, 78.2, 0.0]
                    },
                    "properties": {
                        "name": "Test Station"
                    }
                }
            ]
        }"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "{}", geojson_content).unwrap();
        temp_file.flush().unwrap();
        let temp_path = temp_file.path().to_str().unwrap();

        let result = load_point_locations_from_geojson(temp_path);
        assert!(result.is_ok());

        let locations = result.unwrap();
        assert_eq!(locations.len(), 1);
        assert_eq!(locations[0].lon(), 15.4);
        assert_eq!(locations[0].lat(), 78.2);
    }
}
