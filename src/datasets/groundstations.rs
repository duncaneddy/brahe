/*!
 * Groundstation dataset loading
 *
 * Provides access to curated groundstation location datasets from various providers.
 * Data is embedded in the binary for offline use.
 */

use crate::access::location::PointLocation;
use crate::datasets::loaders::{
    load_point_locations_from_geojson, parse_point_locations_from_geojson,
};
use crate::utils::errors::BraheError;

/// Available groundstation providers
pub const AVAILABLE_PROVIDERS: &[&str] = &[
    "atlas", "aws", "ksat", "leaf", "nasa dsn", "nasa nen", "ssc", "viasat",
];

/// Load embedded groundstation data from compiled binary
///
/// Returns the embedded JSON data for a specific provider.
///
/// # Arguments
/// * `provider` - Provider name (e.g., "atlas", "ksat")
///
/// # Returns
/// * `Result<&'static str, BraheError>` - Embedded JSON string
fn get_embedded_groundstation_data(provider: &str) -> Result<&'static str, BraheError> {
    match provider.to_lowercase().as_str() {
        "atlas" => Ok(include_str!("../../data/groundstations/atlas.json")),
        "aws" => Ok(include_str!("../../data/groundstations/aws.json")),
        "ksat" => Ok(include_str!("../../data/groundstations/ksat.json")),
        "leaf" => Ok(include_str!("../../data/groundstations/leaf.json")),
        "nasa dsn" => Ok(include_str!("../../data/groundstations/dsn.json")),
        "nasa nen" => Ok(include_str!("../../data/groundstations/nen.json")),
        "ssc" => Ok(include_str!("../../data/groundstations/ssc.json")),
        "viasat" => Ok(include_str!("../../data/groundstations/viasat.json")),
        _ => Err(BraheError::Error(format!(
            "Unknown groundstation provider '{}'. Available: {}",
            provider,
            AVAILABLE_PROVIDERS.join(", ")
        ))),
    }
}

/// Load groundstation locations for a specific provider
///
/// Loads groundstation locations from embedded data for the specified provider.
/// The data is compiled into the binary and does not require external files.
///
/// # Arguments
/// * `provider` - Provider name (case-insensitive). Available providers:
///   - "atlas": Atlas Space Operations
///   - "aws": Amazon Web Services Ground Station
///   - "ksat": Kongsberg Satellite Services
///   - "leaf": Leaf Space
///   - "ssc": Swedish Space Corporation
///   - "viasat": Viasat
///
/// # Returns
/// * `Result<Vec<PointLocation>, BraheError>` - Vector of PointLocation objects with properties:
///   - `name`: Groundstation name
///   - `provider`: Provider name
///   - `frequency_bands`: Array of supported frequency bands (e.g., ["S", "X"])
///
/// # Example
/// ```
/// use brahe::datasets::groundstations::load_groundstations;
/// use crate::brahe::utils::Identifiable;
///
/// // Load all KSAT groundstations
/// let ksat_stations = load_groundstations("ksat").unwrap();
/// for station in &ksat_stations {
///     println!("{}: ({:.2}, {:.2})",
///         station.get_name().unwrap_or("Unknown"),
///         station.lon(),
///         station.lat()
///     );
/// }
/// ```
pub fn load_groundstations(provider: &str) -> Result<Vec<PointLocation>, BraheError> {
    // Get embedded JSON data
    let json_str = get_embedded_groundstation_data(provider)?;

    // Parse JSON
    let geojson: serde_json::Value = serde_json::from_str(json_str)
        .map_err(|e| BraheError::ParseError(format!("Failed to parse embedded JSON: {}", e)))?;

    // Parse locations
    parse_point_locations_from_geojson(&geojson)
}

/// Load groundstations from a custom GeoJSON file
///
/// Loads groundstation locations from a user-provided GeoJSON file.
/// The file must be a FeatureCollection with Point geometries.
///
/// # Arguments
/// * `filepath` - Path to GeoJSON file
///
/// # Returns
/// * `Result<Vec<PointLocation>, BraheError>` - Vector of PointLocation objects
///
/// # Example
/// ```no_run
/// use brahe::datasets::groundstations::load_groundstations_from_file;
///
/// let custom_stations = load_groundstations_from_file("my_stations.geojson").unwrap();
/// ```
pub fn load_groundstations_from_file(filepath: &str) -> Result<Vec<PointLocation>, BraheError> {
    load_point_locations_from_geojson(filepath)
}

/// Load all groundstations from all providers
///
/// Convenience function to load groundstations from all available providers.
///
/// # Returns
/// * `Result<Vec<PointLocation>, BraheError>` - Combined vector of all groundstations
///
/// # Example
/// ```
/// use brahe::datasets::groundstations::load_all_groundstations;
///
/// let all_stations = load_all_groundstations().unwrap();
/// println!("Loaded {} total groundstations", all_stations.len());
/// ```
pub fn load_all_groundstations() -> Result<Vec<PointLocation>, BraheError> {
    let mut all_stations = Vec::new();

    for provider in AVAILABLE_PROVIDERS {
        match load_groundstations(provider) {
            Ok(mut stations) => all_stations.append(&mut stations),
            Err(e) => {
                eprintln!("Warning: Failed to load {} groundstations: {}", provider, e);
            }
        }
    }

    if all_stations.is_empty() {
        return Err(BraheError::Error(
            "Failed to load any groundstations".to_string(),
        ));
    }

    Ok(all_stations)
}

/// Get list of available groundstation providers
///
/// Returns a vector of provider names that can be used with `load_groundstations()`.
///
/// # Example
/// ```
/// use brahe::datasets::groundstations::list_providers;
///
/// for provider in list_providers() {
///     println!("Available provider: {}", provider);
/// }
/// ```
pub fn list_providers() -> Vec<String> {
    AVAILABLE_PROVIDERS.iter().map(|s| s.to_string()).collect()
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::access::location::AccessibleLocation;
    use crate::utils::identifiable::Identifiable;

    #[test]
    fn test_load_ksat_groundstations() {
        let stations = load_groundstations("ksat").unwrap();
        assert!(
            !stations.is_empty(),
            "KSAT should have at least one groundstation"
        );

        // Check all stations are valid
        for station in &stations {
            assert!(station.lon() >= -180.0 && station.lon() <= 180.0);
            assert!(station.lat() >= -90.0 && station.lat() <= 90.0);

            assert!(station.get_name().is_some(), "Station should have a name");
        }
    }

    #[test]
    fn test_load_atlas_groundstations() {
        let stations = load_groundstations("atlas").unwrap();
        assert!(
            !stations.is_empty(),
            "Atlas should have at least one groundstation"
        );

        // Check all stations are valid
        for station in &stations {
            assert!(station.lon() >= -180.0 && station.lon() <= 180.0);
            assert!(station.lat() >= -90.0 && station.lat() <= 90.0);

            assert!(station.get_name().is_some(), "Station should have a name");
        }
    }

    #[test]
    fn test_load_aws_groundstations() {
        let stations = load_groundstations("aws").unwrap();
        assert!(
            !stations.is_empty(),
            "AWS should have at least one groundstation"
        );

        // Check all stations are valid
        for station in &stations {
            assert!(station.lon() >= -180.0 && station.lon() <= 180.0);
            assert!(station.lat() >= -90.0 && station.lat() <= 90.0);

            assert!(station.get_name().is_some(), "Station should have a name");
        }
    }

    #[test]
    fn test_load_leaf_groundstations() {
        let stations = load_groundstations("leaf").unwrap();
        assert!(
            !stations.is_empty(),
            "Leaf should have at least one groundstation"
        );

        // Check all stations are valid
        for station in &stations {
            assert!(station.lon() >= -180.0 && station.lon() <= 180.0);
            assert!(station.lat() >= -90.0 && station.lat() <= 90.0);

            assert!(station.get_name().is_some(), "Station should have a name");
        }
    }

    #[test]
    fn test_load_ssc_groundstations() {
        let stations = load_groundstations("ssc").unwrap();
        assert!(
            !stations.is_empty(),
            "SSC should have at least one groundstation"
        );

        // Check all stations are valid
        for station in &stations {
            assert!(station.lon() >= -180.0 && station.lon() <= 180.0);
            assert!(station.lat() >= -90.0 && station.lat() <= 90.0);

            assert!(station.get_name().is_some(), "Station should have a name");
        }
    }

    #[test]
    fn test_load_viasat_groundstations() {
        let stations = load_groundstations("viasat").unwrap();
        assert!(
            !stations.is_empty(),
            "Viasat should have at least one groundstation"
        );

        // Check all stations are valid
        for station in &stations {
            assert!(station.lon() >= -180.0 && station.lon() <= 180.0);
            assert!(station.lat() >= -90.0 && station.lat() <= 90.0);

            assert!(station.get_name().is_some(), "Station should have a name");
        }
    }

    #[test]
    fn test_load_invalid_provider() {
        let result = load_groundstations("nonexistent");
        assert!(result.is_err(), "Should error on invalid provider");
    }

    #[test]
    fn test_case_insensitive_provider() {
        let stations1 = load_groundstations("KSAT").unwrap();
        let stations2 = load_groundstations("ksat").unwrap();
        let stations3 = load_groundstations("KsAt").unwrap();

        assert_eq!(stations1.len(), stations2.len());
        assert_eq!(stations1.len(), stations3.len());
    }

    #[test]
    fn test_load_all_groundstations() {
        let all_stations = load_all_groundstations().unwrap();
        assert!(
            all_stations.len() > 10,
            "Should have multiple groundstations across all providers"
        );

        // Verify total count matches sum of individual providers
        let mut total = 0;
        for provider in AVAILABLE_PROVIDERS {
            let stations = load_groundstations(provider).unwrap();
            total += stations.len();
        }

        assert_eq!(
            all_stations.len(),
            total,
            "Total should match sum of all providers"
        );
    }

    #[test]
    fn test_list_providers() {
        let providers = list_providers();
        assert_eq!(providers.len(), AVAILABLE_PROVIDERS.len());

        // Check that all expected providers are present
        for expected in AVAILABLE_PROVIDERS {
            assert!(
                providers.contains(&expected.to_string()),
                "Should contain provider: {}",
                expected
            );
        }
    }

    #[test]
    fn test_groundstation_properties() {
        for provider in AVAILABLE_PROVIDERS {
            let stations = load_groundstations(provider).unwrap();

            for station in &stations {
                // Should have name
                assert!(
                    station.get_name().is_some(),
                    "Groundstation should have a name"
                );

                // Should have valid coordinates
                assert!(
                    station.lon() >= -180.0 && station.lon() <= 180.0,
                    "Longitude should be valid"
                );
                assert!(
                    station.lat() >= -90.0 && station.lat() <= 90.0,
                    "Latitude should be valid"
                );

                // Should have properties
                let props = station.properties();
                assert!(
                    props.contains_key("provider"),
                    "Should have provider property"
                );
                assert!(
                    props.contains_key("frequency_bands"),
                    "Should have frequency_bands property"
                );

                // Frequency bands should be an array
                let freq_bands = &props["frequency_bands"];
                assert!(freq_bands.is_array(), "Frequency bands should be an array");
            }
        }
    }
}
