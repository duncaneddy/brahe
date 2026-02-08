/*!
 * Internal fetch and caching logic for GCAT data files.
 *
 * Implements file-based caching with configurable TTL, following the same
 * pattern as CelestrakClient's cache implementation.
 */

use std::fs;
use std::io::Read;
use std::path::Path;
use std::time::SystemTime;

use crate::utils::BraheError;
use crate::utils::cache::get_brahe_cache_dir_with_subdir;

/// Get the GCAT cache directory path.
///
/// Returns `~/.cache/brahe/gcat` (or `$BRAHE_CACHE/gcat`).
fn get_gcat_cache_dir() -> Result<String, BraheError> {
    get_brahe_cache_dir_with_subdir(Some("gcat"))
}

/// Fetch a file from a URL with file-based caching.
///
/// Checks for a cached copy first. If the cache file exists and is younger
/// than `cache_max_age` seconds, returns the cached content. Otherwise,
/// downloads from the URL and writes to cache before returning.
///
/// # Arguments
///
/// * `url` - Full URL to fetch
/// * `filename` - Local filename to use in the cache directory
/// * `cache_max_age` - Maximum cache age in seconds
///
/// # Returns
///
/// * `Result<String, BraheError>` - File contents as a string
pub fn fetch_with_cache(
    url: &str,
    filename: &str,
    cache_max_age: f64,
) -> Result<String, BraheError> {
    let cache_dir = get_gcat_cache_dir()?;
    let cache_path = Path::new(&cache_dir).join(filename);

    // Check cache
    if cache_path.exists() && !is_cache_stale(&cache_path, cache_max_age)? {
        let contents = fs::read_to_string(&cache_path)
            .map_err(|e| BraheError::IoError(format!("Failed to read GCAT cache file: {}", e)))?;
        return Ok(contents);
    }

    // Fetch from network
    let body = execute_get(url)?;

    // Write to cache
    fs::write(&cache_path, &body)
        .map_err(|e| BraheError::IoError(format!("Failed to write GCAT cache file: {}", e)))?;

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
        .map_err(|e| BraheError::IoError(format!("GCAT request failed: {}", e)))?;

    // Read body manually to avoid ureq's default 10MB size limit.
    // GCAT SATCAT is ~18MB.
    let mut buffer = Vec::new();
    let mut reader = response.into_body().into_reader();
    reader
        .read_to_end(&mut buffer)
        .map_err(|e| BraheError::IoError(format!("Failed to read GCAT response: {}", e)))?;

    String::from_utf8(buffer)
        .map_err(|e| BraheError::IoError(format!("GCAT response is not valid UTF-8: {}", e)))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_get_gcat_cache_dir() {
        let cache_dir = get_gcat_cache_dir().unwrap();
        assert!(!cache_dir.is_empty());
        assert!(cache_dir.contains("gcat"));
        assert!(std::path::Path::new(&cache_dir).exists());
    }

    #[test]
    fn test_fetch_with_cache_network_error() {
        // Non-existent URL should produce an error
        let result = fetch_with_cache(
            "http://127.0.0.1:1/nonexistent",
            "test_nonexistent.tsv",
            86400.0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_fetch_with_cache_reads_from_cache() {
        let cache_dir = get_gcat_cache_dir().unwrap();
        let cache_path = Path::new(&cache_dir).join("test_cached.tsv");

        // Write a test file to cache
        let test_content = "test data";
        fs::write(&cache_path, test_content).unwrap();

        // Should read from cache (large max_age so it won't be stale)
        let result = fetch_with_cache(
            "http://127.0.0.1:1/should_not_reach",
            "test_cached.tsv",
            86400.0,
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), test_content);

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    fn test_is_cache_stale() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        fs::write(&path, "test").unwrap();

        // Just-written file should not be stale with 24h TTL
        assert!(!is_cache_stale(&path, 86400.0).unwrap());

        // Just-written file should be stale with 0s TTL
        assert!(is_cache_stale(&path, 0.0).unwrap());
    }

    #[test]
    fn test_fetch_with_httpmock() {
        let server = httpmock::MockServer::start();

        let mock = server.mock(|when, then| {
            when.method("GET").path_includes("/satcat.tsv");
            then.status(200).body("#JCAT\tSatcat\nS001\tdata");
        });

        let cache_dir = get_gcat_cache_dir().unwrap();
        let test_filename = "test_httpmock_satcat.tsv";
        let cache_path = Path::new(&cache_dir).join(test_filename);

        // Clean any existing cache
        let _ = fs::remove_file(&cache_path);

        let url = format!("{}/satcat.tsv", server.base_url());
        let result = fetch_with_cache(&url, test_filename, 86400.0);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("S001"));

        mock.assert_calls(1);

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }
}
