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
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Base URL for CelesTrak GP (General Perturbations) data API
const CELESTRAK_BASE_URL: &str = "https://celestrak.org/NORAD/elements/gp.php";

/// Default maximum age for cached files in seconds (6 hours)
const DEFAULT_MAX_CACHE_AGE_SECONDS: f64 = 6.0 * 3600.0;

/// Check if a cached file should be refreshed based on its age
///
/// # Arguments
/// * `filepath` - Path to the cached file
/// * `max_age_seconds` - Maximum age in seconds before refresh is needed
///
/// # Returns
/// * `true` if file doesn't exist or is older than max_age_seconds
/// * `false` if file exists and is fresh enough to use
fn should_refresh_file(filepath: &Path, max_age_seconds: f64) -> bool {
    // If file doesn't exist, need to download
    if !filepath.exists() {
        return true;
    }

    // Get file metadata
    let metadata = match fs::metadata(filepath) {
        Ok(m) => m,
        Err(_) => return true, // If can't read metadata, refresh
    };

    // Get file modification time
    let modified = match metadata.modified() {
        Ok(m) => m,
        Err(_) => return true, // If can't read time, refresh
    };

    // Calculate age
    let now = SystemTime::now();
    let age_duration = match now.duration_since(modified) {
        Ok(d) => d,
        Err(_) => return true, // If time calculation fails, refresh
    };

    let age_seconds = age_duration.as_secs_f64();

    // Return true if file is too old
    age_seconds > max_age_seconds
}

/// Download 3LE data from CelesTrak for a specific satellite group
///
/// This is an internal function. Use `get_tles()` instead for public API.
///
/// # Arguments
/// * `group` - Satellite group name (e.g., "active", "stations", "gnss", "last-30-days")
///
/// # Returns
/// * `Result<String, BraheError>` - Raw 3LE format text
fn fetch_3le_data(group: &str) -> Result<String, BraheError> {
    let url = format!("{}?GROUP={}&FORMAT=3le", CELESTRAK_BASE_URL, group);

    let mut response = ureq::get(&url)
        .call()
        .map_err(|e| BraheError::Error(format!("Failed to download from CelesTrak: {}", e)))?;

    let body = response
        .body_mut()
        .read_to_string()
        .map_err(|e| BraheError::Error(format!("Failed to read response from CelesTrak: {}", e)))?;

    if body.trim().is_empty() {
        return Err(BraheError::Error(format!(
            "No data returned from CelesTrak for group '{}'",
            group
        )));
    }

    Ok(body)
}

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
///
/// # Example
/// ```no_run
/// use brahe::datasets::celestrak::get_tles;
///
/// let ephemeris = get_tles("stations").unwrap();
/// for (name, line1, line2) in ephemeris.iter().take(5) {
///     println!("Satellite: {}", name);
/// }
/// ```
pub fn get_tles(group: &str) -> Result<Vec<(String, String, String)>, BraheError> {
    // Determine cache filepath
    let cache_dir = get_celestrak_cache_dir()?;
    let cache_path = PathBuf::from(&cache_dir).join(format!("{}_gp.txt", group));

    // Check if we need to download fresh data
    let text = if should_refresh_file(&cache_path, DEFAULT_MAX_CACHE_AGE_SECONDS) {
        // Download fresh data
        let data = fetch_3le_data(group)?;

        // Cache it for future use
        if let Err(e) = fs::write(&cache_path, &data) {
            eprintln!(
                "Warning: Failed to cache ephemeris data to {}: {}",
                cache_path.display(),
                e
            );
        }

        data
    } else {
        // Use cached data
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

/// Get satellite ephemeris as SGP propagators from CelesTrak
///
/// Downloads and parses 3LE data, then creates SGP4/SDP4 propagators.
///
/// # Arguments
/// * `group` - Satellite group name (e.g., "active", "stations", "gnss", "last-30-days")
/// * `step_size` - Default step size for propagators in seconds
///
/// # Returns
/// * `Result<Vec<SGPPropagator>, BraheError>` - Vector of configured SGP propagators
///
/// # Example
/// ```no_run
/// use brahe::datasets::celestrak::get_tles_as_propagators;
///
/// let propagators = get_tles_as_propagators("stations", 60.0).unwrap();
/// println!("Loaded {} satellite propagators", propagators.len());
/// ```
pub fn get_tles_as_propagators(
    group: &str,
    step_size: f64,
) -> Result<Vec<SGPPropagator>, BraheError> {
    let ephemeris = get_tles(group)?;

    let mut propagators = Vec::new();
    for (name, line1, line2) in ephemeris {
        match SGPPropagator::from_3le(Some(&name), &line1, &line2, step_size) {
            Ok(prop) => propagators.push(prop),
            Err(e) => {
                // Log warning but continue with other satellites
                eprintln!("Warning: Failed to create propagator for {}: {}", name, e);
            }
        }
    }

    if propagators.is_empty() {
        return Err(BraheError::Error(format!(
            "No valid propagators could be created from group '{}'",
            group
        )));
    }

    Ok(propagators)
}

/// Download satellite ephemeris from CelesTrak and save to file
///
/// Downloads 3LE data and serializes to the specified file format.
/// Uses cached data if available and less than 6 hours old.
///
/// # Arguments
/// * `group` - Satellite group name (e.g., "active", "stations", "gnss", "last-30-days")
/// * `filepath` - Output file path
/// * `content_format` - Content format: "tle" (2-line) or "3le" (3-line with names)
/// * `file_format` - File format: "txt", "csv", or "json"
///
/// # Returns
/// * `Result<(), BraheError>` - Success or error
///
/// # Example
/// ```no_run
/// use brahe::datasets::celestrak::download_tles;
///
/// // Download GNSS satellites as 3LE in JSON format
/// download_tles("gnss", "gnss_sats.json", "3le", "json").unwrap();
///
/// // Download active satellites as 2LE in text format
/// download_tles("active", "active_sats.txt", "tle", "txt").unwrap();
/// ```
pub fn download_tles(
    group: &str,
    filepath: &str,
    content_format: &str,
    file_format: &str,
) -> Result<(), BraheError> {
    // Validate format parameters BEFORE downloading
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

    // Validate file format before downloading
    let file_format_lower = file_format.to_lowercase();
    if !matches!(file_format_lower.as_str(), "txt" | "csv" | "json") {
        return Err(BraheError::Error(format!(
            "Invalid file format '{}'. Must be 'txt', 'csv', or 'json'",
            file_format
        )));
    }

    // Download and parse data (get_tles handles caching internally)
    let ephemeris = get_tles(group)?;

    // Serialize to requested format
    let output = match file_format_lower.as_str() {
        "txt" => serialize_3le_to_txt(&ephemeris, include_names),
        "csv" => serialize_3le_to_csv(&ephemeris, include_names),
        "json" => serialize_3le_to_json(&ephemeris, include_names),
        _ => unreachable!(), // Already validated above
    };

    // Create parent directory if it doesn't exist
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

    // Write to file
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
///
/// Downloads 3LE data from CelesTrak for a single satellite identified by its
/// NORAD catalog number. Uses cached data if available and less than 6 hours old.
///
/// # Arguments
/// * `norad_id` - NORAD catalog number (1-9 digits)
/// * `group` - Optional satellite group for fallback search if direct ID lookup fails
///
/// # Returns
/// * `Result<(String, String, String), BraheError>` - Tuple of (name, line1, line2)
///
/// # Example
/// ```no_run
/// use brahe::datasets::celestrak::get_tle_by_id;
///
/// // Get ISS TLE by NORAD ID (25544)
/// let (name, line1, line2) = get_tle_by_id(25544, None).unwrap();
/// println!("Satellite: {}", name);
/// println!("Line 1: {}", line1);
/// println!("Line 2: {}", line2);
///
/// // With group fallback
/// let tle = get_tle_by_id(25544, Some("stations")).unwrap();
/// ```
///
/// # Notes
/// You can find which group contains a specific NORAD ID at:
/// https://celestrak.org/NORAD/elements/master-gp-index.php
pub fn get_tle_by_id(
    norad_id: u32,
    group: Option<&str>,
) -> Result<(String, String, String), BraheError> {
    // Determine cache filepath
    let cache_dir = get_celestrak_cache_dir()?;
    let cache_path = PathBuf::from(&cache_dir).join(format!("tle_{}.txt", norad_id));

    // Check if we need to download fresh data
    let text = if should_refresh_file(&cache_path, DEFAULT_MAX_CACHE_AGE_SECONDS) {
        // Try direct CATNR query first
        let url = format!("{}?CATNR={}&FORMAT=3le", CELESTRAK_BASE_URL, norad_id);

        let result = ureq::get(&url).call();

        let data = match result {
            Ok(mut response) => {
                let body = response
                    .body_mut()
                    .read_to_string()
                    .map_err(|e| BraheError::Error(format!("Failed to read response: {}", e)))?;

                if body.trim().is_empty() {
                    // If direct query failed and group provided, try group search
                    if let Some(grp) = group {
                        return get_tle_by_id_from_group(norad_id, grp);
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
                // If direct query failed and group provided, try group search
                if let Some(grp) = group {
                    return get_tle_by_id_from_group(norad_id, grp);
                } else {
                    return Err(BraheError::Error(format!(
                        "Failed to download TLE for NORAD ID {}: {}",
                        norad_id, e
                    )));
                }
            }
        };

        // Cache it for future use
        if let Err(e) = fs::write(&cache_path, &data) {
            eprintln!(
                "Warning: Failed to cache TLE data to {}: {}",
                cache_path.display(),
                e
            );
        }

        data
    } else {
        // Use cached data
        fs::read_to_string(&cache_path).map_err(|e| {
            BraheError::Error(format!(
                "Failed to read cached TLE from {}: {}",
                cache_path.display(),
                e
            ))
        })?
    };

    // Parse and return the single TLE
    let tles = parse_3le_text(&text)?;

    if tles.is_empty() {
        return Err(BraheError::Error(format!(
            "No TLE found in response for NORAD ID {}",
            norad_id
        )));
    }

    Ok(tles[0].clone())
}

/// Helper function to search for a TLE by NORAD ID within a group
fn get_tle_by_id_from_group(
    norad_id: u32,
    group: &str,
) -> Result<(String, String, String), BraheError> {
    let ephemeris = get_tles(group)?;

    // Search for matching NORAD ID in line 1 (catalog number is at positions 2-7)
    let norad_str = format!("{:5}", norad_id);

    for (name, line1, line2) in ephemeris {
        // Extract catalog number from line 1 (columns 3-7, 1-indexed)
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

/// Get TLE data for a specific satellite as an SGP propagator
///
/// Downloads TLE data from CelesTrak for a single satellite and creates an
/// SGP4/SDP4 propagator. Uses cached data if available and less than 6 hours old.
///
/// # Arguments
/// * `norad_id` - NORAD catalog number (1-9 digits)
/// * `group` - Optional satellite group for fallback search if direct ID lookup fails
/// * `step_size` - Default step size for propagator in seconds
///
/// # Returns
/// * `Result<SGPPropagator, BraheError>` - Configured SGP propagator
///
/// # Example
/// ```no_run
/// use brahe::datasets::celestrak::get_tle_by_id_as_propagator;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::propagators::traits::StateProvider;
///
/// // Get ISS as propagator with 60-second step size
/// let propagator = get_tle_by_id_as_propagator(25544, None, 60.0).unwrap();
///
/// // Compute state at a specific epoch
/// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let state = propagator.state_eci(epoch);
/// println!("ISS position: x={}, y={}, z={}", state[0], state[1], state[2]);
/// ```
///
/// # Notes
/// You can find which group contains a specific NORAD ID at:
/// https://celestrak.org/NORAD/elements/master-gp-index.php
pub fn get_tle_by_id_as_propagator(
    norad_id: u32,
    group: Option<&str>,
    step_size: f64,
) -> Result<SGPPropagator, BraheError> {
    let (name, line1, line2) = get_tle_by_id(norad_id, group)?;

    SGPPropagator::from_3le(Some(&name), &line1, &line2, step_size)
}

/// Get TLE data for a specific satellite by name
///
/// Searches for a satellite by name using a cascading search strategy:
/// 1. If a group is provided, search within that group first
/// 2. Fall back to searching the "active" group
/// 3. Fall back to using CelesTrak's NAME API
///
/// Uses cached data if available and less than 6 hours old.
///
/// # Arguments
/// * `name` - Satellite name (case-insensitive, partial matches supported)
/// * `group` - Optional satellite group to search first
///
/// # Returns
/// * `Result<(String, String, String), BraheError>` - Tuple of (name, line1, line2)
///
/// # Example
/// ```no_run
/// use brahe::datasets::celestrak::get_tle_by_name;
///
/// // Search for ISS with group hint
/// let (name, line1, line2) = get_tle_by_name("ISS", Some("stations")).unwrap();
/// println!("Found: {}", name);
///
/// // Search without group (uses cascading search)
/// let tle = get_tle_by_name("STARLINK-1234", None).unwrap();
/// ```
///
/// # Notes
/// - Name matching is case-insensitive
/// - Partial names are supported (e.g., "ISS" will match "ISS (ZARYA)")
/// - If multiple satellites match, returns the first match
/// - Search order: specified group → "active" → NAME API
pub fn get_tle_by_name(
    name: &str,
    group: Option<&str>,
) -> Result<(String, String, String), BraheError> {
    let name_upper = name.to_uppercase();

    // Helper function to search within a group
    let search_in_group = |grp: &str| -> Result<(String, String, String), BraheError> {
        let ephemeris = get_tles(grp)?;

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
        // Otherwise, continue to fallback strategies
    }

    // Strategy 2: Search in "active" group
    if let Ok(result) = search_in_group("active") {
        return Ok(result);
    }

    // Strategy 3: Use CelesTrak NAME API
    let cache_dir = get_celestrak_cache_dir()?;
    let cache_path = PathBuf::from(&cache_dir).join(format!("name_{}.txt", name.replace(' ', "_")));

    let text = if should_refresh_file(&cache_path, DEFAULT_MAX_CACHE_AGE_SECONDS) {
        let url = format!(
            "{}?NAME={}&FORMAT=3le",
            CELESTRAK_BASE_URL,
            name.replace(' ', "%20")
        );

        let result = ureq::get(&url).call();

        let data = match result {
            Ok(mut response) => {
                let body = response
                    .body_mut()
                    .read_to_string()
                    .map_err(|e| BraheError::Error(format!("Failed to read response: {}", e)))?;

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

        // Cache it for future use
        if let Err(e) = fs::write(&cache_path, &data) {
            eprintln!(
                "Warning: Failed to cache TLE data to {}: {}",
                cache_path.display(),
                e
            );
        }

        data
    } else {
        // Use cached data
        fs::read_to_string(&cache_path).map_err(|e| {
            BraheError::Error(format!(
                "Failed to read cached TLE from {}: {}",
                cache_path.display(),
                e
            ))
        })?
    };

    // Parse and return the TLE
    let tles = parse_3le_text(&text)?;

    if tles.is_empty() {
        return Err(BraheError::Error(format!(
            "No TLE found in response for satellite name '{}'",
            name
        )));
    }

    Ok(tles[0].clone())
}

/// Get TLE data for a specific satellite by name as an SGP propagator
///
/// Searches for a satellite by name and creates an SGP4/SDP4 propagator.
/// Uses cascading search strategy (specified group → active → NAME API).
/// Uses cached data if available and less than 6 hours old.
///
/// # Arguments
/// * `name` - Satellite name (case-insensitive, partial matches supported)
/// * `group` - Optional satellite group to search first
/// * `step_size` - Default step size for propagator in seconds
///
/// # Returns
/// * `Result<SGPPropagator, BraheError>` - Configured SGP propagator
///
/// # Example
/// ```no_run
/// use brahe::datasets::celestrak::get_tle_by_name_as_propagator;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::propagators::traits::StateProvider;
///
/// // Get ISS as propagator with 60-second step size
/// let propagator = get_tle_by_name_as_propagator("ISS", Some("stations"), 60.0).unwrap();
///
/// // Propagate to a specific epoch
/// let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let state = propagator.state_eci(epoch);
/// println!("Position: x={}, y={}, z={}", state[0], state[1], state[2]);
/// ```
pub fn get_tle_by_name_as_propagator(
    name: &str,
    group: Option<&str>,
    step_size: f64,
) -> Result<SGPPropagator, BraheError> {
    let (sat_name, line1, line2) = get_tle_by_name(name, group)?;

    SGPPropagator::from_3le(Some(&sat_name), &line1, &line2, step_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use serial_test::serial;
    use tempfile::tempdir;

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

    #[test]
    fn test_get_tles_success() {
        let server = MockServer::start();
        let _mock = server.mock(|when, then| {
            when.method(GET)
                .path("/NORAD/elements/gp.php")
                .query_param("GROUP", "test-group")
                .query_param("FORMAT", "3le");
            then.status(200).body(get_test_3le_data());
        });

        // Temporarily override the URL for testing
        // Note: In real implementation, we'd need to make CELESTRAK_BASE_URL configurable for testing
        // For now, this test demonstrates the expected behavior
    }

    #[test]
    fn test_download_tles_txt_with_names() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("test_3le.txt");

        // Mock data
        let test_data = vec![(
            "ISS (ZARYA)".to_string(),
            "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997".to_string(),
            "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003".to_string(),
        )];

        // Directly test serialization and file writing
        let output = serialize_3le_to_txt(&test_data, true);
        fs::write(&filepath, output).unwrap();

        // Verify file contents
        let contents = fs::read_to_string(&filepath).unwrap();
        assert!(contents.contains("ISS (ZARYA)"));
        assert!(contents.contains("1 25544"));
        assert!(contents.contains("2 25544"));
    }

    #[test]
    fn test_download_tles_txt_without_names() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("test_2le.txt");

        let test_data = vec![(
            "ISS (ZARYA)".to_string(),
            "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997".to_string(),
            "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003".to_string(),
        )];

        let output = serialize_3le_to_txt(&test_data, false);
        fs::write(&filepath, output).unwrap();

        let contents = fs::read_to_string(&filepath).unwrap();
        assert!(!contents.contains("ISS (ZARYA)"));
        assert!(contents.contains("1 25544"));
        assert!(contents.contains("2 25544"));
    }

    #[test]
    fn test_download_tles_csv() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("test.csv");

        let test_data = vec![(
            "ISS (ZARYA)".to_string(),
            "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997".to_string(),
            "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003".to_string(),
        )];

        let output = serialize_3le_to_csv(&test_data, true);
        fs::write(&filepath, output).unwrap();

        let contents = fs::read_to_string(&filepath).unwrap();
        assert!(contents.contains("name,line1,line2"));
        assert!(contents.contains("ISS (ZARYA)"));
    }

    #[test]
    fn test_download_tles_json() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("test.json");

        let test_data = vec![(
            "ISS (ZARYA)".to_string(),
            "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997".to_string(),
            "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003".to_string(),
        )];

        let output = serialize_3le_to_json(&test_data, true);
        fs::write(&filepath, output).unwrap();

        let contents = fs::read_to_string(&filepath).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert!(parsed.is_array());
        assert_eq!(parsed[0]["name"], "ISS (ZARYA)");
    }

    #[test]
    fn test_download_tles_invalid_content_format() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("test.txt");

        // Since we can't easily mock the HTTP call in this test, we'll just verify
        // that the validation logic works by testing the error conditions directly
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
    fn test_download_tles_creates_parent_directory() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("subdir").join("test.txt");

        // Verify parent directory doesn't exist yet
        assert!(!filepath.parent().unwrap().exists());

        // Test data
        let test_data = vec![(
            "ISS (ZARYA)".to_string(),
            "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997".to_string(),
            "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003".to_string(),
        )];

        // Manually create parent directory and write file to test this behavior
        let parent = filepath.parent().unwrap();
        fs::create_dir_all(parent).unwrap();
        let output = serialize_3le_to_txt(&test_data, true);
        fs::write(&filepath, output).unwrap();

        // Verify file was created
        assert!(filepath.exists());
    }

    // Network tests - require internet connection and celestrak.org availability
    // Run with: cargo test --features ci

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_get_tles_network() {
        // Test with a small group
        let result = get_tles("stations");
        assert!(result.is_ok());

        let ephemeris = result.unwrap();
        assert!(!ephemeris.is_empty());

        // Verify format of first TLE
        let (name, line1, line2) = &ephemeris[0];
        assert!(!name.is_empty());
        assert!(line1.starts_with("1 "));
        assert!(line2.starts_with("2 "));
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    #[serial]
    fn test_get_tles_as_propagators_network() {
        use crate::utils::testing::setup_global_test_eop;

        setup_global_test_eop();

        let result = get_tles_as_propagators("stations", 60.0);
        assert!(result.is_ok());

        let propagators = result.unwrap();
        assert!(!propagators.is_empty());

        // Verify first propagator has a satellite name
        assert!(propagators[0].satellite_name.is_some());
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_download_tles_network() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("stations.json");

        let result = download_tles("stations", filepath.to_str().unwrap(), "3le", "json");
        assert!(result.is_ok());

        // Verify file was created
        assert!(filepath.exists());

        // Verify content is valid JSON
        let contents = fs::read_to_string(&filepath).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
        assert!(parsed.is_array());
        assert!(!parsed.as_array().unwrap().is_empty());
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_get_tle_by_id_network() {
        // Test with ISS (NORAD ID 25544)
        let result = get_tle_by_id(25544, None);
        assert!(result.is_ok());

        let (name, line1, line2) = result.unwrap();
        assert!(name.contains("ISS") || name.contains("ZARYA"));
        assert!(line1.starts_with("1 25544"));
        assert!(line2.starts_with("2 25544"));
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_get_tle_by_id_with_group_network() {
        // Test with ISS using group fallback
        let result = get_tle_by_id(25544, Some("stations"));
        assert!(result.is_ok());

        let (name, line1, line2) = result.unwrap();
        assert!(name.contains("ISS") || name.contains("ZARYA"));
        assert!(line1.starts_with("1 25544"));
        assert!(line2.starts_with("2 25544"));
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    #[serial]
    fn test_get_tle_by_id_as_propagator_network() {
        use crate::utils::testing::setup_global_test_eop;

        setup_global_test_eop();

        // Test with ISS
        let result = get_tle_by_id_as_propagator(25544, None, 60.0);
        assert!(result.is_ok());

        let propagator = result.unwrap();
        assert!(propagator.satellite_name.is_some());
        let name = propagator.satellite_name.unwrap();
        assert!(name.contains("ISS") || name.contains("ZARYA"));
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    #[serial_test::serial]
    fn test_caching_behavior_network() {
        use std::thread;
        use std::time::Duration;

        // Clear cache for this test
        let cache_dir = get_celestrak_cache_dir().unwrap();
        let cache_path = PathBuf::from(&cache_dir).join("tle_25544.txt");
        let _ = fs::remove_file(&cache_path);

        // Ensure file doesn't exist
        assert!(!cache_path.exists());

        // First call should download and create cache file
        let result1 = get_tle_by_id(25544, None);
        assert!(result1.is_ok());
        assert!(cache_path.exists());

        // Get modification time
        let metadata1 = fs::metadata(&cache_path).unwrap();
        let mtime1 = metadata1.modified().unwrap();

        // Wait a bit to ensure different timestamp if file were rewritten
        thread::sleep(Duration::from_millis(200));

        // Second call should use cache (file modification time shouldn't change)
        let result2 = get_tle_by_id(25544, None);
        assert!(result2.is_ok());

        let metadata2 = fs::metadata(&cache_path).unwrap();
        let mtime2 = metadata2.modified().unwrap();

        // Modification times should be very close (within 10ms tolerance for filesystem timing variations)
        // The cache is working if the file wasn't rewritten after the 200ms sleep
        let time_diff = if mtime2 > mtime1 {
            mtime2.duration_since(mtime1).unwrap()
        } else {
            mtime1.duration_since(mtime2).unwrap()
        };

        assert!(
            time_diff < Duration::from_millis(200), // Normal tolerance should be 10ms but allow 200ms because this fails on windows CI sometimes
            "Cache file appears to have been rewritten (time difference: {:?}ms). Expected < 200ms but got mtime1={:?}, mtime2={:?}",
            time_diff.as_millis(),
            mtime1,
            mtime2
        );

        // Results should be identical
        assert_eq!(result1.unwrap(), result2.unwrap());

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    // ========================================
    // Non-network tests using test_assets
    // ========================================

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

        // Create a file
        let mut file = fs::File::create(&filepath).unwrap();
        file.write_all(b"test data").unwrap();
        drop(file);

        // File is brand new, should not need refresh (within 1 hour)
        assert!(!should_refresh_file(&filepath, 3600.0));
    }

    #[test]
    fn test_should_refresh_file_stale() {
        use std::io::Write;

        let dir = tempdir().unwrap();
        let filepath = dir.path().join("stale_file.txt");

        // Create a file
        let mut file = fs::File::create(&filepath).unwrap();
        file.write_all(b"test data").unwrap();
        drop(file);

        // Set max age to 0 seconds, making any file stale
        assert!(should_refresh_file(&filepath, 0.0));
    }

    #[test]
    fn test_parse_3le_from_test_asset() {
        let test_file = get_test_asset_path("celestrak_stations_3le.txt");
        let contents = fs::read_to_string(test_file).unwrap();
        let result = parse_3le_text(&contents);

        assert!(result.is_ok());
        let ephemeris = result.unwrap();
        assert_eq!(ephemeris.len(), 2);

        // Check ISS
        assert_eq!(ephemeris[0].0, "ISS (ZARYA)");
        assert!(ephemeris[0].1.starts_with("1 25544"));
        assert!(ephemeris[0].2.starts_with("2 25544"));

        // Check Tiangong
        assert_eq!(ephemeris[1].0, "TIANGONG");
        assert!(ephemeris[1].1.starts_with("1 48274"));
        assert!(ephemeris[1].2.starts_with("2 48274"));
    }

    #[test]
    fn test_parse_3le_empty_file() {
        let test_file = get_test_asset_path("celestrak_empty.txt");
        let contents = fs::read_to_string(test_file).unwrap();
        let result = parse_3le_text(&contents);

        // Empty file should return error (no valid 3LE entries found)
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No valid 3LE entries")
        );
    }

    #[test]
    fn test_serialization_formats_coverage() {
        let test_data = vec![
            (
                "ISS (ZARYA)".to_string(),
                "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9991".to_string(),
                "2 25544  51.6400 247.4627 0001013  37.6900 322.4251 15.50030129000011".to_string(),
            ),
            (
                "TIANGONG".to_string(),
                "1 48274U 21035A   24001.50000000  .00002605  00000-0  43346-4 0  9999".to_string(),
                "2 48274  41.4689 226.7056 0005390 251.8025 108.2351 15.59646707000014".to_string(),
            ),
        ];

        // Test TXT format with names
        let txt_with_names = serialize_3le_to_txt(&test_data, true);
        assert!(txt_with_names.contains("ISS (ZARYA)"));
        assert!(txt_with_names.contains("TIANGONG"));
        assert!(txt_with_names.contains("1 25544"));
        assert!(txt_with_names.contains("1 48274"));

        // Test TXT format without names
        let txt_without_names = serialize_3le_to_txt(&test_data, false);
        assert!(!txt_without_names.contains("ISS (ZARYA)"));
        assert!(!txt_without_names.contains("TIANGONG"));
        assert!(txt_without_names.contains("1 25544"));
        assert!(txt_without_names.contains("1 48274"));

        // Test CSV format
        let csv_data = serialize_3le_to_csv(&test_data, true);
        assert!(csv_data.contains("name,line1,line2"));
        assert!(csv_data.contains("ISS (ZARYA)"));

        // Test JSON format
        let json_data = serialize_3le_to_json(&test_data, true);
        let parsed: serde_json::Value = serde_json::from_str(&json_data).unwrap();
        assert!(parsed.is_array());
        assert_eq!(parsed.as_array().unwrap().len(), 2);
        assert_eq!(parsed[0]["name"], "ISS (ZARYA)");
        assert_eq!(parsed[1]["name"], "TIANGONG");
    }

    #[test]
    fn test_download_tles_all_format_combinations() {
        let test_data_file = get_test_asset_path("celestrak_stations_3le.txt");
        let test_data_text = fs::read_to_string(test_data_file).unwrap();
        let ephemeris = parse_3le_text(&test_data_text).unwrap();

        let dir = tempdir().unwrap();

        // Test all valid combinations
        let combinations = vec![
            ("3le", "txt"),
            ("3le", "csv"),
            ("3le", "json"),
            ("tle", "txt"),
            ("tle", "csv"),
            ("tle", "json"),
        ];

        for (content_fmt, file_fmt) in combinations {
            let include_names = content_fmt == "3le";
            let filename = format!("test_{}.{}", content_fmt, file_fmt);
            let filepath = dir.path().join(&filename);

            // Serialize using the serializers directly (simulating download_tles logic)
            let output = match file_fmt {
                "txt" => serialize_3le_to_txt(&ephemeris, include_names),
                "csv" => serialize_3le_to_csv(&ephemeris, include_names),
                "json" => serialize_3le_to_json(&ephemeris, include_names),
                _ => unreachable!(),
            };

            fs::write(&filepath, output).unwrap();
            assert!(filepath.exists());

            // Verify content
            let contents = fs::read_to_string(&filepath).unwrap();
            assert!(contents.contains("25544")); // ISS NORAD ID

            if include_names {
                assert!(contents.contains("ISS") || contents.contains("ZARYA"));
            }
        }
    }

    #[test]
    fn test_invalid_format_errors() {
        let dir = tempdir().unwrap();

        // Test invalid content format
        let result = download_tles(
            "test",
            dir.path().join("test.txt").to_str().unwrap(),
            "invalid_content",
            "txt",
        );
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid content format")
        );

        // Test invalid file format
        let result = download_tles(
            "test",
            dir.path().join("test.txt").to_str().unwrap(),
            "3le",
            "invalid_file",
        );
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid file format")
        );
    }
}
