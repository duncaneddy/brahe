/*!
 * Internal fetch and caching logic for star catalog data files.
 *
 * Implements file-based caching with configurable TTL, following the same
 * pattern as GCAT's fetch implementation. Star catalogs are fixed datasets,
 * so a cache TTL of `None` means the cached copy never goes stale.
 */

use std::fs;
use std::path::Path;
use std::time::SystemTime;

use crate::utils::BraheError;
use crate::utils::atomic_write;
use crate::utils::cache::get_star_catalogs_cache_dir;
use crate::utils::download::download_bytes_with_user_agent;

/// `User-Agent` header sent with star catalog downloads.
///
/// The mirror (`simplespacedata.org`) returns `403 Forbidden` to default HTTP
/// client user agents (e.g. ureq's or curl's default), so a browser-style
/// User-Agent is required.
const USER_AGENT: &str = "Mozilla/5.0 (compatible; brahe)";

/// Read a cached copy of a star catalog file, if present and not stale.
///
/// Does not touch the network. Callers are expected to parse the returned
/// content and, if parsing fails, fall back to [`download`] followed by
/// [`commit_cache`] once the freshly downloaded data has been validated by
/// parsing successfully. This "parse-before-cache" ordering keeps a bad
/// download (or a bad cache write from a previous run) from poisoning the
/// cache forever, since a corrupt cached file always triggers exactly one
/// forced re-download.
///
/// # Arguments
///
/// * `filename` - Local filename to use in the cache directory
/// * `cache_max_age` - Maximum cache age in seconds. `Some(max_age)` marks
///   the cached file stale once it is older than `max_age` seconds. `None`
///   means the cached file never goes stale, appropriate for fixed star
///   catalogs that do not change once published.
///
/// # Returns
///
/// * `Result<Option<String>, BraheError>` - `Some(contents)` on a cache hit,
///   `None` on a cache miss or stale cache
pub(crate) fn read_cache(
    filename: &str,
    cache_max_age: Option<f64>,
) -> Result<Option<String>, BraheError> {
    let cache_dir = get_star_catalogs_cache_dir()?;
    let cache_path = Path::new(&cache_dir).join(filename);

    if !cache_path.exists() {
        return Ok(None);
    }

    let stale = match cache_max_age {
        Some(max_age) => is_cache_stale(&cache_path, max_age)?,
        None => false, // fixed catalogs: cached copy never expires
    };
    if stale {
        return Ok(None);
    }

    let contents = fs::read_to_string(&cache_path).map_err(|e| {
        BraheError::IoError(format!("Failed to read star catalog cache file: {}", e))
    })?;
    Ok(Some(contents))
}

/// Download a file from a URL, without touching the cache.
///
/// Uses the shared [`crate::utils::download`] retry/backoff machinery (with a
/// browser-style `User-Agent`, see [`USER_AGENT`]) reading the body through a
/// streaming reader to avoid ureq's default in-memory size limit — the
/// Tycho-2 catalog is ~526 MB.
///
/// # Arguments
///
/// * `url` - Full URL to fetch
///
/// # Returns
///
/// * `Result<String, BraheError>` - File contents as a string
pub(crate) fn download(url: &str) -> Result<String, BraheError> {
    let bytes = download_bytes_with_user_agent(url, USER_AGENT)
        .map_err(|e| BraheError::IoError(format!("Star catalog request failed: {}", e)))?;

    String::from_utf8(bytes).map_err(|e| {
        BraheError::IoError(format!("Star catalog response is not valid UTF-8: {}", e))
    })
}

/// Write a successfully parsed star catalog file to the cache.
///
/// Callers must only call this after confirming `body` parses successfully,
/// so that a bad download never overwrites a good cache entry (or leaves a
/// newly poisoned one behind).
///
/// # Arguments
///
/// * `filename` - Local filename to use in the cache directory
/// * `body` - File contents to write
///
/// # Returns
///
/// * `Result<(), BraheError>` - `Ok(())` on success
pub(crate) fn commit_cache(filename: &str, body: &str) -> Result<(), BraheError> {
    let cache_dir = get_star_catalogs_cache_dir()?;
    let cache_path = Path::new(&cache_dir).join(filename);

    atomic_write(&cache_path, body.as_bytes())
        .map_err(|e| BraheError::IoError(format!("Failed to write star catalog cache file: {}", e)))
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use serial_test::parallel;

    #[test]
    #[parallel]
    fn test_get_star_catalogs_cache_dir() {
        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        assert!(!cache_dir.is_empty());
        assert!(cache_dir.contains("star_catalogs"));
        assert!(std::path::Path::new(&cache_dir).exists());
    }

    #[test]
    #[parallel]
    fn test_read_cache_never_stale() {
        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        let cache_path = Path::new(&cache_dir).join("test_never_stale.dat");

        // Write a test file to cache
        let test_content = "fixed catalog data";
        fs::write(&cache_path, test_content).unwrap();

        // With cache_max_age = None, the cached copy is always considered fresh.
        let result = read_cache("test_never_stale.dat", None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Some(test_content.to_string()));

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    #[parallel]
    fn test_read_cache_miss_when_absent() {
        // No cache file written for this filename -> cache miss (None), not an error.
        let result = read_cache("test_read_cache_miss_absent.dat", None);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), None);
    }

    #[test]
    #[parallel]
    fn test_read_cache_force_refresh_reports_stale_as_miss() {
        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        let test_filename = "test_force_refresh.dat";
        let cache_path = Path::new(&cache_dir).join(test_filename);

        // Seed a cache file that would otherwise be treated as fresh.
        fs::write(&cache_path, "stale data").unwrap();

        // cache_max_age = Some(0.0) forces the cached copy to be treated as
        // stale (a cache miss), so callers fall through to a fresh download.
        let result = read_cache(test_filename, Some(0.0));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), None);

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    #[parallel]
    fn test_download_and_commit_cache() {
        let server = httpmock::MockServer::start();

        let mock = server.mock(|when, then| {
            when.method("GET").path_includes("/catalog.dat");
            then.status(200).body("fresh data");
        });

        let url = format!("{}/catalog.dat", server.base_url());
        let body = download(&url).unwrap();
        assert_eq!(body, "fresh data");
        mock.assert_calls(1);

        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        let test_filename = "test_download_and_commit.dat";
        let cache_path = Path::new(&cache_dir).join(test_filename);

        commit_cache(test_filename, &body).unwrap();
        let contents = fs::read_to_string(&cache_path).unwrap();
        assert_eq!(contents, "fresh data");

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    #[parallel]
    fn test_download_network_error() {
        // Non-existent URL should produce an error
        let result = download("http://127.0.0.1:1/nonexistent");
        assert!(result.is_err());
    }

    #[test]
    #[parallel]
    fn test_download_errors_on_invalid_utf8_body() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("GET").path_includes("/binary.dat");
            then.status(200).body([0xFF, 0xFE, 0xFD]);
        });

        let url = format!("{}/binary.dat", server.base_url());
        let result = download(&url);
        assert!(result.is_err());
        mock.assert_calls(1);
    }

    #[test]
    #[parallel]
    fn test_read_cache_errors_when_cache_path_is_a_directory() {
        // A directory at the cache path passes the `exists()` check in
        // read_cache but cannot be opened as a file, exercising the
        // fs::read_to_string I/O error branch.
        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        let test_filename = "test_read_cache_is_a_directory.dat";
        let cache_path = Path::new(&cache_dir).join(test_filename);

        fs::create_dir_all(&cache_path).unwrap();

        let result = read_cache(test_filename, None);
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&cache_path);
    }

    #[test]
    #[parallel]
    fn test_commit_cache_errors_when_parent_path_is_blocked() {
        // A regular file sitting where commit_cache's target path needs a
        // directory component makes atomic_write's `create_dir_all` fail,
        // exercising commit_cache's I/O error branch.
        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        let blocker_name = "test_commit_cache_blocker.dat";
        let blocker_path = Path::new(&cache_dir).join(blocker_name);
        fs::write(&blocker_path, "not a directory").unwrap();

        let nested_filename = format!("{}/nested.dat", blocker_name);
        let result = commit_cache(&nested_filename, "data");
        assert!(result.is_err());

        let _ = fs::remove_file(&blocker_path);
    }
}
