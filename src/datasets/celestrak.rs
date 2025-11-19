/*!
 * CelesTrak data source implementation for downloading satellite ephemeris.
 *
 * CelesTrak (https://celestrak.org) is maintained by T.S. Kelso and provides
 * authoritative TLE data for satellites organized by various groupings.
 */

use crate::datasets::parsers::parse_3le_text;
use crate::datasets::serializers::{
    serialize_3le_to_csv, serialize_3le_to_json, serialize_3le_to_txt,
};
use crate::propagators::SGPPropagator;
use crate::utils::BraheError;
use crate::utils::cache::get_celestrak_cache_dir;
use crate::utils::threading::get_thread_pool;
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Base URL for CelesTrak GP (General Perturbations) data API
const CELESTRAK_BASE_URL: &str = "https://celestrak.org/NORAD/elements/gp.php";

/// Default maximum age for cached files in seconds (6 hours)
const DEFAULT_MAX_CACHE_AGE_SECONDS: f64 = 6.0 * 3600.0;

// =============================================================================
// HTTP Client Trait for Dependency Injection
// =============================================================================

/// Trait for HTTP client operations, allowing for dependency injection and mocking.
#[cfg_attr(test, mockall::automock)]
pub trait HttpClient: Send + Sync {
    /// Perform an HTTP GET request and return the response body as a string.
    fn get_string(&self, url: &str) -> Result<String, BraheError>;
}

/// Default HTTP client implementation using ureq.
pub struct UreqHttpClient;

impl HttpClient for UreqHttpClient {
    fn get_string(&self, url: &str) -> Result<String, BraheError> {
        let mut response = ureq::get(url)
            .call()
            .map_err(|e| BraheError::Error(format!("HTTP request failed for {}: {}", url, e)))?;

        let body = response
            .body_mut()
            .read_to_string()
            .map_err(|e| BraheError::Error(format!("Failed to read response body: {}", e)))?;

        Ok(body)
    }
}

// =============================================================================
// CelesTrak Client with Dependency Injection
// =============================================================================

/// CelesTrak client with injectable HTTP client for testability.
pub struct CelestrakClient<C: HttpClient> {
    client: C,
    base_url: String,
    max_cache_age: f64,
}

impl<C: HttpClient> CelestrakClient<C> {
    /// Create a new CelestrakClient with a custom HTTP client.
    pub fn new(client: C) -> Self {
        Self {
            client,
            base_url: CELESTRAK_BASE_URL.to_string(),
            max_cache_age: DEFAULT_MAX_CACHE_AGE_SECONDS,
        }
    }

    /// Create a new CelestrakClient with custom settings (for testing).
    #[cfg(test)]
    pub fn with_config(client: C, base_url: String, max_cache_age: f64) -> Self {
        Self {
            client,
            base_url,
            max_cache_age,
        }
    }

    /// Download 3LE data from CelesTrak for a specific satellite group.
    fn fetch_3le_data(&self, group: &str) -> Result<String, BraheError> {
        let url = format!("{}?GROUP={}&FORMAT=3le", self.base_url, group);

        let body = self.client.get_string(&url)?;

        if body.trim().is_empty() {
            return Err(BraheError::Error(format!(
                "No data returned from CelesTrak for group '{}'",
                group
            )));
        }

        Ok(body)
    }

    /// Get satellite ephemeris data from CelesTrak.
    pub fn get_tles(&self, group: &str) -> Result<Vec<(String, String, String)>, BraheError> {
        let cache_dir = get_celestrak_cache_dir()?;
        let cache_path = PathBuf::from(&cache_dir).join(format!("{}_gp.txt", group));

        let text = if should_refresh_file(&cache_path, self.max_cache_age) {
            let data = self.fetch_3le_data(group)?;

            if let Err(e) = fs::write(&cache_path, &data) {
                eprintln!(
                    "Warning: Failed to cache ephemeris data to {}: {}",
                    cache_path.display(),
                    e
                );
            }

            data
        } else {
            fs::read_to_string(&cache_path).map_err(|e| {
                BraheError::Error(format!(
                    "Failed to read cached ephemeris from {}: {}",
                    cache_path.display(),
                    e
                ))
            })?
        };

        parse_3le_text(&text)
    }

    /// Get satellite ephemeris as SGP propagators.
    pub fn get_tles_as_propagators(
        &self,
        group: &str,
        step_size: f64,
    ) -> Result<Vec<SGPPropagator>, BraheError> {
        let ephemeris = self.get_tles(group)?;

        let propagators: Vec<SGPPropagator> = get_thread_pool().install(|| {
            ephemeris
                .par_iter()
                .filter_map(|(name, line1, line2)| {
                    match SGPPropagator::from_3le(Some(name), line1, line2, step_size) {
                        Ok(prop) => Some(prop),
                        Err(e) => {
                            eprintln!("Warning: Failed to create propagator for {}: {}", name, e);
                            None
                        }
                    }
                })
                .collect()
        });

        if propagators.is_empty() {
            return Err(BraheError::Error(format!(
                "No valid propagators could be created from group '{}'",
                group
            )));
        }

        Ok(propagators)
    }

    /// Get TLE data for a specific satellite by NORAD catalog number.
    pub fn get_tle_by_id(
        &self,
        norad_id: u32,
        group: Option<&str>,
    ) -> Result<(String, String, String), BraheError> {
        let cache_dir = get_celestrak_cache_dir()?;
        let cache_path = PathBuf::from(&cache_dir).join(format!("tle_{}.txt", norad_id));

        let text = if should_refresh_file(&cache_path, self.max_cache_age) {
            let url = format!("{}?CATNR={}&FORMAT=3le", self.base_url, norad_id);

            let result = self.client.get_string(&url);

            let data = match result {
                Ok(body) => {
                    if body.trim().is_empty() {
                        if let Some(grp) = group {
                            return self.get_tle_by_id_from_group(norad_id, grp);
                        } else {
                            return Err(BraheError::Error(format!(
                                "No TLE data found for NORAD ID {}. Try providing a group parameter.",
                                norad_id
                            )));
                        }
                    }
                    body
                }
                Err(e) => {
                    if let Some(grp) = group {
                        return self.get_tle_by_id_from_group(norad_id, grp);
                    } else {
                        return Err(BraheError::Error(format!(
                            "Failed to download TLE for NORAD ID {}: {}",
                            norad_id, e
                        )));
                    }
                }
            };

            if let Err(e) = fs::write(&cache_path, &data) {
                eprintln!(
                    "Warning: Failed to cache TLE data to {}: {}",
                    cache_path.display(),
                    e
                );
            }

            data
        } else {
            fs::read_to_string(&cache_path).map_err(|e| {
                BraheError::Error(format!(
                    "Failed to read cached TLE from {}: {}",
                    cache_path.display(),
                    e
                ))
            })?
        };

        let tles = parse_3le_text(&text)?;

        if tles.is_empty() {
            return Err(BraheError::Error(format!(
                "No TLE found in response for NORAD ID {}",
                norad_id
            )));
        }

        Ok(tles[0].clone())
    }

    /// Helper function to search for a TLE by NORAD ID within a group.
    fn get_tle_by_id_from_group(
        &self,
        norad_id: u32,
        group: &str,
    ) -> Result<(String, String, String), BraheError> {
        let ephemeris = self.get_tles(group)?;
        let norad_str = format!("{:5}", norad_id);

        for (name, line1, line2) in ephemeris {
            if line1.len() >= 7 {
                let catalog_str = &line1[2..7];
                if catalog_str.trim() == norad_str.trim() {
                    return Ok((name, line1, line2));
                }
            }
        }

        Err(BraheError::Error(format!(
            "NORAD ID {} not found in group '{}'",
            norad_id, group
        )))
    }

    /// Get TLE data for a specific satellite by name.
    pub fn get_tle_by_name(
        &self,
        name: &str,
        group: Option<&str>,
    ) -> Result<(String, String, String), BraheError> {
        let name_upper = name.to_uppercase();

        // Helper to search within a group
        let search_in_group = |grp: &str| -> Result<(String, String, String), BraheError> {
            let ephemeris = self.get_tles(grp)?;

            for (sat_name, line1, line2) in ephemeris {
                if sat_name.to_uppercase().contains(&name_upper) {
                    return Ok((sat_name, line1, line2));
                }
            }

            Err(BraheError::Error(format!(
                "Satellite '{}' not found in group '{}'",
                name, grp
            )))
        };

        // Strategy 1: Search in specified group if provided
        if let Some(grp) = group
            && let Ok(result) = search_in_group(grp)
        {
            return Ok(result);
        }

        // Strategy 2: Search in "active" group
        if let Ok(result) = search_in_group("active") {
            return Ok(result);
        }

        // Strategy 3: Use CelesTrak NAME API
        let cache_dir = get_celestrak_cache_dir()?;
        let cache_path =
            PathBuf::from(&cache_dir).join(format!("name_{}.txt", name.replace(' ', "_")));

        let text = if should_refresh_file(&cache_path, self.max_cache_age) {
            let url = format!(
                "{}?NAME={}&FORMAT=3le",
                self.base_url,
                name.replace(' ', "%20")
            );

            let result = self.client.get_string(&url);

            let data = match result {
                Ok(body) => {
                    if body.trim().is_empty() {
                        return Err(BraheError::Error(format!(
                            "No TLE data found for satellite name '{}'. Try specifying a group or check the name.",
                            name
                        )));
                    }
                    body
                }
                Err(e) => {
                    return Err(BraheError::Error(format!(
                        "Failed to download TLE for satellite name '{}': {}",
                        name, e
                    )));
                }
            };

            if let Err(e) = fs::write(&cache_path, &data) {
                eprintln!(
                    "Warning: Failed to cache TLE data to {}: {}",
                    cache_path.display(),
                    e
                );
            }

            data
        } else {
            fs::read_to_string(&cache_path).map_err(|e| {
                BraheError::Error(format!(
                    "Failed to read cached TLE from {}: {}",
                    cache_path.display(),
                    e
                ))
            })?
        };

        let tles = parse_3le_text(&text)?;

        if tles.is_empty() {
            return Err(BraheError::Error(format!(
                "No TLE found in response for satellite name '{}'",
                name
            )));
        }

        Ok(tles[0].clone())
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Check if a cached file should be refreshed based on its age.
fn should_refresh_file(filepath: &Path, max_age_seconds: f64) -> bool {
    if !filepath.exists() {
        return true;
    }

    let metadata = match fs::metadata(filepath) {
        Ok(m) => m,
        Err(_) => return true,
    };

    let modified = match metadata.modified() {
        Ok(m) => m,
        Err(_) => return true,
    };

    let now = SystemTime::now();
    let age_duration = match now.duration_since(modified) {
        Ok(d) => d,
        Err(_) => return true,
    };

    age_duration.as_secs_f64() > max_age_seconds
}

// =============================================================================
// Public API - Convenience Functions
// =============================================================================

/// Get satellite ephemeris data from CelesTrak
///
/// Downloads and parses 3LE data for the specified satellite group.
/// Uses cached data if available and less than 6 hours old.
///
/// # Arguments
/// * `group` - Satellite group name (e.g., "active", "stations", "gnss", "last-30-days")
///
/// # Returns
/// * `Result<Vec<(String, String, String)>, BraheError>` - Vector of (name, line1, line2) tuples
pub fn get_tles(group: &str) -> Result<Vec<(String, String, String)>, BraheError> {
    let client = CelestrakClient::new(UreqHttpClient);
    client.get_tles(group)
}

/// Get satellite ephemeris as SGP propagators from CelesTrak
pub fn get_tles_as_propagators(
    group: &str,
    step_size: f64,
) -> Result<Vec<SGPPropagator>, BraheError> {
    let client = CelestrakClient::new(UreqHttpClient);
    client.get_tles_as_propagators(group, step_size)
}

/// Download satellite ephemeris from CelesTrak and save to file
pub fn download_tles(
    group: &str,
    filepath: &str,
    content_format: &str,
    file_format: &str,
) -> Result<(), BraheError> {
    let include_names = match content_format.to_lowercase().as_str() {
        "tle" => false,
        "3le" => true,
        _ => {
            return Err(BraheError::Error(format!(
                "Invalid content format '{}'. Must be 'tle' or '3le'",
                content_format
            )));
        }
    };

    let file_format_lower = file_format.to_lowercase();
    if !matches!(file_format_lower.as_str(), "txt" | "csv" | "json") {
        return Err(BraheError::Error(format!(
            "Invalid file format '{}'. Must be 'txt', 'csv', or 'json'",
            file_format
        )));
    }

    let ephemeris = get_tles(group)?;

    let output = match file_format_lower.as_str() {
        "txt" => serialize_3le_to_txt(&ephemeris, include_names),
        "csv" => serialize_3le_to_csv(&ephemeris, include_names),
        "json" => serialize_3le_to_json(&ephemeris, include_names),
        _ => unreachable!(),
    };

    let filepath = Path::new(filepath);
    if let Some(parent_dir) = filepath.parent() {
        fs::create_dir_all(parent_dir).map_err(|e| {
            BraheError::Error(format!(
                "Failed to create directory {}: {}",
                parent_dir.display(),
                e
            ))
        })?;
    }

    fs::write(filepath, output).map_err(|e| {
        BraheError::Error(format!(
            "Failed to write file {}: {}",
            filepath.display(),
            e
        ))
    })?;

    Ok(())
}

/// Get TLE data for a specific satellite by NORAD catalog number
pub fn get_tle_by_id(
    norad_id: u32,
    group: Option<&str>,
) -> Result<(String, String, String), BraheError> {
    let client = CelestrakClient::new(UreqHttpClient);
    client.get_tle_by_id(norad_id, group)
}

/// Get TLE data for a specific satellite as an SGP propagator
pub fn get_tle_by_id_as_propagator(
    norad_id: u32,
    group: Option<&str>,
    step_size: f64,
) -> Result<SGPPropagator, BraheError> {
    let (name, line1, line2) = get_tle_by_id(norad_id, group)?;
    SGPPropagator::from_3le(Some(&name), &line1, &line2, step_size)
}

/// Get TLE data for a specific satellite by name
pub fn get_tle_by_name(
    name: &str,
    group: Option<&str>,
) -> Result<(String, String, String), BraheError> {
    let client = CelestrakClient::new(UreqHttpClient);
    client.get_tle_by_name(name, group)
}

/// Get TLE data for a specific satellite by name as an SGP propagator
pub fn get_tle_by_name_as_propagator(
    name: &str,
    group: Option<&str>,
    step_size: f64,
) -> Result<SGPPropagator, BraheError> {
    let (sat_name, line1, line2) = get_tle_by_name(name, group)?;
    SGPPropagator::from_3le(Some(&sat_name), &line1, &line2, step_size)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use serial_test::serial;
    use tempfile::tempdir;

    // =========================================================================
    // Test Data Helpers
    // =========================================================================

    fn get_test_3le_data() -> String {
        "ISS (ZARYA)\n\
         1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997\n\
         2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003\n\
         STARLINK-1007\n\
         1 44713U 19074A   21001.50000000  .00001764  00000-0  40967-4 0  9997\n\
         2 44713  53.0000 100.0000 0001000  90.0000  10.0000 15.00000000000003"
            .to_string()
    }

    fn get_test_asset_path(filename: &str) -> PathBuf {
        use std::env;
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        Path::new(&manifest_dir).join("test_assets").join(filename)
    }

    // =========================================================================
    // Mock Tests - CelestrakClient
    // =========================================================================

    #[test]
    #[serial]
    fn test_get_tles_success_mocked() {
        let mut mock_client = MockHttpClient::new();

        mock_client
            .expect_get_string()
            .withf(|url: &str| url.contains("GROUP=stations"))
            .times(1)
            .returning(|_| Ok(get_test_3le_data()));

        let client = CelestrakClient::new(mock_client);

        // Clear cache
        let cache_dir = get_celestrak_cache_dir().unwrap();
        let cache_path = PathBuf::from(&cache_dir).join("stations_gp.txt");
        let _ = fs::remove_file(&cache_path);

        let result = client.get_tles("stations");
        assert!(result.is_ok());

        let ephemeris = result.unwrap();
        assert_eq!(ephemeris.len(), 2);
        assert_eq!(ephemeris[0].0, "ISS (ZARYA)");
        assert!(ephemeris[0].1.starts_with("1 25544"));

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    #[serial]
    fn test_get_tles_empty_response_mocked() {
        let mut mock_client = MockHttpClient::new();

        mock_client
            .expect_get_string()
            .times(1)
            .returning(|_| Ok(String::new()));

        let client = CelestrakClient::new(mock_client);

        // Clear cache
        let cache_dir = get_celestrak_cache_dir().unwrap();
        let cache_path = PathBuf::from(&cache_dir).join("empty-group_gp.txt");
        let _ = fs::remove_file(&cache_path);

        let result = client.get_tles("empty-group");
        assert!(result.is_err());

        let err = result.unwrap_err();
        match err {
            BraheError::Error(msg) => {
                assert!(msg.contains("No data returned"));
            }
            _ => panic!("Expected BraheError::Error"),
        }
    }

    #[test]
    #[serial]
    fn test_get_tles_network_error_mocked() {
        let mut mock_client = MockHttpClient::new();

        mock_client
            .expect_get_string()
            .times(1)
            .returning(|_| Err(BraheError::Error("Connection refused".to_string())));

        let client = CelestrakClient::new(mock_client);

        // Clear cache
        let cache_dir = get_celestrak_cache_dir().unwrap();
        let cache_path = PathBuf::from(&cache_dir).join("error-group_gp.txt");
        let _ = fs::remove_file(&cache_path);

        let result = client.get_tles("error-group");
        assert!(result.is_err());
    }

    #[test]
    #[serial]
    fn test_get_tles_caching_mocked() {
        let mut mock_client = MockHttpClient::new();

        // Only expect one call - second should use cache
        mock_client
            .expect_get_string()
            .times(1)
            .returning(|_| Ok(get_test_3le_data()));

        let client = CelestrakClient::new(mock_client);

        // Clear cache
        let cache_dir = get_celestrak_cache_dir().unwrap();
        let cache_path = PathBuf::from(&cache_dir).join("cache-test_gp.txt");
        let _ = fs::remove_file(&cache_path);

        // First call downloads
        let result1 = client.get_tles("cache-test");
        assert!(result1.is_ok());

        // Second call uses cache
        let result2 = client.get_tles("cache-test");
        assert!(result2.is_ok());

        assert_eq!(result1.unwrap().len(), result2.unwrap().len());

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    #[serial]
    fn test_get_tle_by_id_success_mocked() {
        let mut mock_client = MockHttpClient::new();

        let single_tle = "ISS (ZARYA)\n\
            1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997\n\
            2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";

        mock_client
            .expect_get_string()
            .withf(|url: &str| url.contains("CATNR=25544"))
            .times(1)
            .returning(move |_| Ok(single_tle.to_string()));

        let client = CelestrakClient::new(mock_client);

        // Clear cache
        let cache_dir = get_celestrak_cache_dir().unwrap();
        let cache_path = PathBuf::from(&cache_dir).join("tle_25544.txt");
        let _ = fs::remove_file(&cache_path);

        let result = client.get_tle_by_id(25544, None);
        assert!(result.is_ok());

        let (name, line1, line2) = result.unwrap();
        assert!(name.contains("ISS"));
        assert!(line1.starts_with("1 25544"));
        assert!(line2.starts_with("2 25544"));

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    #[serial]
    fn test_get_tle_by_id_not_found_mocked() {
        let mut mock_client = MockHttpClient::new();

        mock_client
            .expect_get_string()
            .times(1)
            .returning(|_| Ok(String::new())); // Empty response = not found

        let client = CelestrakClient::new(mock_client);

        // Clear cache
        let cache_dir = get_celestrak_cache_dir().unwrap();
        let cache_path = PathBuf::from(&cache_dir).join("tle_99999.txt");
        let _ = fs::remove_file(&cache_path);

        let result = client.get_tle_by_id(99999, None);
        assert!(result.is_err());

        let err = result.unwrap_err();
        match err {
            BraheError::Error(msg) => {
                assert!(msg.contains("No TLE data found"));
            }
            _ => panic!("Expected BraheError::Error"),
        }
    }

    #[test]
    #[serial]
    fn test_get_tle_by_id_with_group_fallback_mocked() {
        let mut mock_client = MockHttpClient::new();

        // First call (CATNR) returns empty
        mock_client
            .expect_get_string()
            .withf(|url: &str| url.contains("CATNR=25544"))
            .times(1)
            .returning(|_| Ok(String::new()));

        // Second call (GROUP) returns data
        mock_client
            .expect_get_string()
            .withf(|url: &str| url.contains("GROUP=stations"))
            .times(1)
            .returning(|_| Ok(get_test_3le_data()));

        let client = CelestrakClient::new(mock_client);

        // Clear caches
        let cache_dir = get_celestrak_cache_dir().unwrap();
        let _ = fs::remove_file(PathBuf::from(&cache_dir).join("tle_25544.txt"));
        let _ = fs::remove_file(PathBuf::from(&cache_dir).join("stations_gp.txt"));

        let result = client.get_tle_by_id(25544, Some("stations"));
        assert!(result.is_ok());

        let (name, _, _) = result.unwrap();
        assert!(name.contains("ISS"));
    }

    #[test]
    #[serial]
    fn test_get_tle_by_name_success_mocked() {
        let mut mock_client = MockHttpClient::new();

        // Return data for "active" group search
        mock_client
            .expect_get_string()
            .withf(|url: &str| url.contains("GROUP=active"))
            .times(1)
            .returning(|_| Ok(get_test_3le_data()));

        let client = CelestrakClient::new(mock_client);

        // Clear cache
        let cache_dir = get_celestrak_cache_dir().unwrap();
        let cache_path = PathBuf::from(&cache_dir).join("active_gp.txt");
        let _ = fs::remove_file(&cache_path);

        let result = client.get_tle_by_name("ISS", None);
        assert!(result.is_ok());

        let (name, _, _) = result.unwrap();
        assert!(name.contains("ISS"));

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    // =========================================================================
    // Validation and Serialization Tests
    // =========================================================================

    #[test]
    fn test_should_refresh_file_nonexistent() {
        let nonexistent_path = Path::new("/tmp/nonexistent_file_brahe_test.txt");
        assert!(should_refresh_file(nonexistent_path, 3600.0));
    }

    #[test]
    fn test_should_refresh_file_fresh() {
        use std::io::Write;
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("fresh_file.txt");

        let mut file = fs::File::create(&filepath).unwrap();
        file.write_all(b"test data").unwrap();
        drop(file);

        assert!(!should_refresh_file(&filepath, 3600.0));
    }

    #[test]
    fn test_should_refresh_file_stale() {
        use std::io::Write;
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("stale_file.txt");

        let mut file = fs::File::create(&filepath).unwrap();
        file.write_all(b"test data").unwrap();
        drop(file);

        // Max age of 0 makes any file stale
        assert!(should_refresh_file(&filepath, 0.0));
    }

    #[test]
    fn test_download_tles_invalid_content_format() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("test.txt");

        let result = download_tles("test-group", filepath.to_str().unwrap(), "invalid", "txt");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid content format")
        );
    }

    #[test]
    fn test_download_tles_invalid_file_format() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("test.txt");

        let result = download_tles("test-group", filepath.to_str().unwrap(), "3le", "invalid");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid file format")
        );
    }

    #[test]
    fn test_parse_3le_from_test_asset() {
        let test_file = get_test_asset_path("celestrak_stations_3le.txt");
        let contents = fs::read_to_string(test_file).unwrap();
        let result = parse_3le_text(&contents);

        assert!(result.is_ok());
        let ephemeris = result.unwrap();
        assert_eq!(ephemeris.len(), 2);

        assert_eq!(ephemeris[0].0, "ISS (ZARYA)");
        assert!(ephemeris[0].1.starts_with("1 25544"));
    }

    #[test]
    fn test_parse_3le_empty_file() {
        let test_file = get_test_asset_path("celestrak_empty.txt");
        let contents = fs::read_to_string(test_file).unwrap();
        let result = parse_3le_text(&contents);

        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No valid 3LE entries")
        );
    }

    #[test]
    fn test_serialization_formats() {
        let test_data = vec![(
            "ISS (ZARYA)".to_string(),
            "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9991".to_string(),
            "2 25544  51.6400 247.4627 0001013  37.6900 322.4251 15.50030129000011".to_string(),
        )];

        // TXT with names
        let txt = serialize_3le_to_txt(&test_data, true);
        assert!(txt.contains("ISS (ZARYA)"));
        assert!(txt.contains("1 25544"));

        // TXT without names
        let txt = serialize_3le_to_txt(&test_data, false);
        assert!(!txt.contains("ISS (ZARYA)"));
        assert!(txt.contains("1 25544"));

        // CSV
        let csv = serialize_3le_to_csv(&test_data, true);
        assert!(csv.contains("name,line1,line2"));
        assert!(csv.contains("ISS (ZARYA)"));

        // JSON
        let json = serialize_3le_to_json(&test_data, true);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_array());
        assert_eq!(parsed[0]["name"], "ISS (ZARYA)");
    }

    // =========================================================================
    // Network Tests - Only run with manual feature
    // =========================================================================

    #[test]
    #[cfg_attr(not(feature = "manual"), ignore)]
    fn test_get_tles_network() {
        let result = get_tles("stations");
        assert!(result.is_ok());

        let ephemeris = result.unwrap();
        assert!(!ephemeris.is_empty());

        let (name, line1, line2) = &ephemeris[0];
        assert!(!name.is_empty());
        assert!(line1.starts_with("1 "));
        assert!(line2.starts_with("2 "));
    }

    #[test]
    #[cfg_attr(not(feature = "manual"), ignore)]
    #[serial]
    fn test_get_tles_as_propagators_network() {
        use crate::utils::testing::setup_global_test_eop;
        setup_global_test_eop();

        let result = get_tles_as_propagators("stations", 60.0);
        assert!(result.is_ok());

        let propagators = result.unwrap();
        assert!(!propagators.is_empty());
        assert!(propagators[0].satellite_name.is_some());
    }

    #[test]
    #[cfg_attr(not(feature = "manual"), ignore)]
    fn test_download_tles_network() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("stations.json");

        let result = download_tles("stations", filepath.to_str().unwrap(), "3le", "json");
        assert!(result.is_ok());
        assert!(filepath.exists());

        let contents = fs::read_to_string(&filepath).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert!(parsed.is_array());
        assert!(!parsed.as_array().unwrap().is_empty());
    }

    #[test]
    #[cfg_attr(not(feature = "manual"), ignore)]
    fn test_get_tle_by_id_network() {
        let result = get_tle_by_id(25544, None);
        assert!(result.is_ok());

        let (name, line1, line2) = result.unwrap();
        assert!(name.contains("ISS") || name.contains("ZARYA"));
        assert!(line1.starts_with("1 25544"));
        assert!(line2.starts_with("2 25544"));
    }

    #[test]
    #[cfg_attr(not(feature = "manual"), ignore)]
    #[serial]
    fn test_get_tle_by_id_as_propagator_network() {
        use crate::utils::testing::setup_global_test_eop;
        setup_global_test_eop();

        let result = get_tle_by_id_as_propagator(25544, None, 60.0);
        assert!(result.is_ok());

        let propagator = result.unwrap();
        assert!(propagator.satellite_name.is_some());
    }

    #[test]
    #[cfg_attr(not(feature = "manual"), ignore)]
    #[serial]
    fn test_caching_behavior_network() {
        use std::thread;
        use std::time::Duration;

        let cache_dir = get_celestrak_cache_dir().unwrap();
        let cache_path = PathBuf::from(&cache_dir).join("tle_25544.txt");
        let _ = fs::remove_file(&cache_path);

        let result1 = get_tle_by_id(25544, None);
        assert!(result1.is_ok());
        assert!(cache_path.exists());

        let metadata1 = fs::metadata(&cache_path).unwrap();
        let mtime1 = metadata1.modified().unwrap();

        thread::sleep(Duration::from_millis(200));

        let result2 = get_tle_by_id(25544, None);
        assert!(result2.is_ok());

        let metadata2 = fs::metadata(&cache_path).unwrap();
        let mtime2 = metadata2.modified().unwrap();

        // Cache should not have been refreshed
        assert_eq!(mtime1, mtime2);
        assert_eq!(result1.unwrap(), result2.unwrap());

        let _ = fs::remove_file(&cache_path);
    }
}
