/*!
 * Internal fetch and caching logic for star catalog data files.
 *
 * Implements file-based caching with configurable TTL, following the same
 * pattern as GCAT's fetch implementation. Star catalogs are fixed datasets,
 * so a cache TTL of `None` means the cached copy never goes stale.
 */

use std::fs;
use std::io::Read;
use std::path::Path;
use std::time::SystemTime;

use crate::utils::BraheError;
use crate::utils::atomic_write;
use crate::utils::cache::get_star_catalog_cache_dir;

/// Fetch a file from a URL with file-based caching.
///
/// Checks for a cached copy first. If the cache file exists and is not
/// stale, returns the cached content. Otherwise, downloads from the URL and
/// writes to cache before returning.
///
/// # Arguments
///
/// * `url` - Full URL to fetch
/// * `filename` - Local filename to use in the cache directory
/// * `cache_max_age` - Maximum cache age in seconds. `Some(max_age)` marks
///   the cached file stale once it is older than `max_age` seconds. `None`
///   means the cached file never goes stale, appropriate for fixed star
///   catalogs that do not change once published.
///
/// # Returns
///
/// * `Result<String, BraheError>` - File contents as a string
#[allow(dead_code)] // consumed by the FK5/Hipparcos/Tycho-2 catalog loaders (Tasks 5-7)
pub(crate) fn fetch_with_cache(
    url: &str,
    filename: &str,
    cache_max_age: Option<f64>,
) -> Result<String, BraheError> {
    let cache_dir = get_star_catalog_cache_dir()?;
    let cache_path = Path::new(&cache_dir).join(filename);

    // Check cache
    if cache_path.exists() {
        let stale = match cache_max_age {
            Some(max_age) => is_cache_stale(&cache_path, max_age)?,
            None => false, // fixed catalogs: cached copy never expires
        };
        if !stale {
            let contents = fs::read_to_string(&cache_path).map_err(|e| {
                BraheError::IoError(format!("Failed to read star catalog cache file: {}", e))
            })?;
            return Ok(contents);
        }
    }

    // Fetch from network
    let body = execute_get(url)?;

    // Write to cache
    atomic_write(&cache_path, body.as_bytes()).map_err(|e| {
        BraheError::IoError(format!("Failed to write star catalog cache file: {}", e))
    })?;

    Ok(body)
}

/// Check if a cache file is older than the maximum cache age.
fn is_cache_stale(path: &Path, cache_max_age: f64) -> Result<bool, BraheError> {
    let metadata = fs::metadata(path)
        .map_err(|e| BraheError::IoError(format!("Failed to read file metadata: {}", e)))?;

    let modified = metadata.modified().map_err(|e| {
        BraheError::IoError(format!("Failed to read file modification time: {}", e))
    })?;

    let age = SystemTime::now()
        .duration_since(modified)
        .unwrap_or_default();

    Ok(age.as_secs_f64() >= cache_max_age)
}

/// Execute an HTTP GET request and return the response body.
fn execute_get(url: &str) -> Result<String, BraheError> {
    let agent = ureq::Agent::new_with_defaults();
    let response = agent
        .get(url)
        .call()
        .map_err(|e| BraheError::IoError(format!("Star catalog request failed: {}", e)))?;

    // Read body manually to avoid ureq's default 10MB size limit.
    // Tycho-2 is ~526MB.
    let mut buffer = Vec::new();
    let mut reader = response.into_body().into_reader();
    reader
        .read_to_end(&mut buffer)
        .map_err(|e| BraheError::IoError(format!("Failed to read star catalog response: {}", e)))?;

    String::from_utf8(buffer).map_err(|e| {
        BraheError::IoError(format!("Star catalog response is not valid UTF-8: {}", e))
    })
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use serial_test::parallel;

    #[test]
    #[parallel]
    fn test_get_star_catalog_cache_dir() {
        let cache_dir = get_star_catalog_cache_dir().unwrap();
        assert!(!cache_dir.is_empty());
        assert!(cache_dir.contains("star_catalog"));
        assert!(std::path::Path::new(&cache_dir).exists());
    }

    #[test]
    #[parallel]
    fn test_fetch_with_cache_never_stale() {
        let cache_dir = get_star_catalog_cache_dir().unwrap();
        let cache_path = Path::new(&cache_dir).join("test_never_stale.dat");

        // Write a test file to cache
        let test_content = "fixed catalog data";
        fs::write(&cache_path, test_content).unwrap();

        // With cache_max_age = None, a dead URL must never be reached: the
        // cached copy is always considered fresh.
        let result = fetch_with_cache("http://127.0.0.1:1/x", "test_never_stale.dat", None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), test_content);

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    #[parallel]
    fn test_fetch_with_cache_force_refresh() {
        let server = httpmock::MockServer::start();

        let mock = server.mock(|when, then| {
            when.method("GET").path_includes("/catalog.dat");
            then.status(200).body("fresh data");
        });

        let cache_dir = get_star_catalog_cache_dir().unwrap();
        let test_filename = "test_force_refresh.dat";
        let cache_path = Path::new(&cache_dir).join(test_filename);

        // Seed a cache file that would otherwise be treated as fresh.
        fs::write(&cache_path, "stale data").unwrap();

        let url = format!("{}/catalog.dat", server.base_url());
        let result = fetch_with_cache(&url, test_filename, Some(0.0));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "fresh data");

        mock.assert_calls(1);

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    #[parallel]
    fn test_fetch_with_cache_network_error() {
        // Non-existent URL with no cached copy should produce an error
        let result = fetch_with_cache(
            "http://127.0.0.1:1/nonexistent",
            "test_nonexistent.dat",
            Some(86400.0),
        );
        assert!(result.is_err());
    }
}
