/*!
 * SBDB Lookup HTTP client with on-disk caching.
 */

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use crate::datasets::sbdb::responses::SBDBObject;
use crate::utils::cache::{get_sbdb_cache_dir, short_hash};
use crate::utils::download::{download_string_no_redirect, urlencode};
use crate::utils::network::{CacheDecision, cache_policy};
use crate::utils::{BraheError, atomic_write};

/// Default base URL for the JPL SSD/SBDB API.
const DEFAULT_BASE_URL: &str = "https://ssd-api.jpl.nasa.gov";
/// Default cache max age: 30 days.
const DEFAULT_CACHE_MAX_AGE: u64 = 30 * 24 * 60 * 60;

/// Client for the JPL Small-Body Database (SBDB) Lookup API.
///
/// Resolves a search string to an [`SBDBObject`]. Responses are cached on disk
/// under the SBDB cache directory and reused until `cache_max_age` elapses.
/// The `BRAHE_NETWORK_MODE` environment variable controls whether a stale
/// cached response is refreshed or served; see [`crate::utils::network`].
///
/// # Examples
///
/// ```no_run
/// use brahe::datasets::sbdb::SBDBClient;
///
/// let client = SBDBClient::new();
/// let ceres = client.lookup("Ceres").unwrap();
/// assert_eq!(ceres.naif_id(), 20000001);
/// ```
pub struct SBDBClient {
    base_url: String,
    cache_max_age: u64,
}

impl Default for SBDBClient {
    fn default() -> Self {
        Self::new()
    }
}

impl SBDBClient {
    /// Create a client with the default base URL and 30-day cache age.
    pub fn new() -> Self {
        SBDBClient {
            base_url: DEFAULT_BASE_URL.to_string(),
            cache_max_age: DEFAULT_CACHE_MAX_AGE,
        }
    }

    /// Create a client pointed at a custom base URL (e.g. a mock server).
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL without a trailing slash.
    pub fn with_base_url(base_url: &str) -> Self {
        SBDBClient {
            base_url: base_url.trim_end_matches('/').to_string(),
            cache_max_age: DEFAULT_CACHE_MAX_AGE,
        }
    }

    /// Create a client with a custom cache max age in seconds (`0` = always refetch).
    ///
    /// # Arguments
    ///
    /// * `seconds` - Maximum cache age in seconds.
    pub fn with_cache_age(seconds: u64) -> Self {
        SBDBClient {
            base_url: DEFAULT_BASE_URL.to_string(),
            cache_max_age: seconds,
        }
    }

    /// Create a client with a custom base URL and cache max age.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL without a trailing slash.
    /// * `seconds` - Maximum cache age in seconds (0 = always refetch).
    pub fn with_base_url_and_cache_age(base_url: &str, seconds: u64) -> Self {
        SBDBClient {
            base_url: base_url.trim_end_matches('/').to_string(),
            cache_max_age: seconds,
        }
    }

    /// Resolve a search string (name or designation) to an [`SBDBObject`].
    ///
    /// # Arguments
    ///
    /// * `sstr` - Object search string, e.g. `"Ceres"` or `"20000001"`.
    ///
    /// # Returns
    ///
    /// * `Ok(SBDBObject)` - The resolved object.
    /// * `Err(BraheError)` - On ambiguous/no match, network, or parse errors,
    ///   or if `BRAHE_NETWORK_MODE` forbids the request needed to fill or
    ///   refresh the cache; see [`crate::utils::network`].
    pub fn lookup(&self, sstr: &str) -> Result<SBDBObject, BraheError> {
        // `full-prec=1` returns full-precision physical parameters; without it
        // SBDB rounds GM/radius for display, which would bake reduced-precision
        // constants into a propagation when these feed a custom central body.
        let url = format!(
            "{}/sbdb.api?sstr={}&phys-par=1&full-prec=1",
            self.base_url,
            urlencode(sstr)
        );

        let cache_path =
            PathBuf::from(get_sbdb_cache_dir()?).join(format!("{}.json", short_hash(sstr)));

        // Fall through to refetch on a stale/corrupt cache parse failure.
        if let Some(body) = self.read_fresh_cache(&cache_path)?
            && let Ok(obj) = SBDBObject::from_json(&body)
        {
            return Ok(obj);
        }

        let body = download_string_no_redirect(&url, "SBDB lookup")?;
        // Parse before caching so error responses are never cached.
        let obj = SBDBObject::from_json(&body)?;
        atomic_write(&cache_path, body.as_bytes()).map_err(|e| {
            BraheError::IoError(format!(
                "Failed to write SBDB cache {}: {}",
                cache_path.display(),
                e
            ))
        })?;
        Ok(obj)
    }

    /// Return the cached body if present and servable under the network mode.
    ///
    /// A file younger than `cache_max_age` is always served. An older file is
    /// served in `offline` mode, ignored in `online` mode, and an error in
    /// `offline-strict` mode.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the cached response file
    ///
    /// # Returns
    ///
    /// * `Ok(Some(String))` - Cached body, either fresh or served stale under
    ///   `offline`
    /// * `Ok(None)` - No cache file exists, or it is stale and `online` mode
    ///   allows a refresh
    /// * `Err(BraheError)` - The cache file is stale and `offline-strict`
    ///   forbids serving it, or its modification time cannot be read
    fn read_fresh_cache(&self, path: &Path) -> Result<Option<String>, BraheError> {
        let Ok(metadata) = std::fs::metadata(path) else {
            return Ok(None);
        };
        let modified = metadata.modified().map_err(|e| {
            BraheError::IoError(format!(
                "Failed to read modification time of {}: {}",
                path.display(),
                e
            ))
        })?;
        let age = SystemTime::now()
            .duration_since(modified)
            .unwrap_or(Duration::ZERO);
        let stale = age > Duration::from_secs(self.cache_max_age);
        let resource = format!("SBDB lookup cache {}", path.display());
        match cache_policy(&resource, stale)? {
            CacheDecision::Serve => Ok(std::fs::read_to_string(path).ok()),
            CacheDecision::Refresh => Ok(None),
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::utils::testing::{CacheRedirect, NetworkModeGuard};
    use httpmock::prelude::*;
    use serial_test::serial;
    use std::fs;
    use std::time::{Duration, SystemTime};

    const CERES_BODY: &str = r#"{"object":{"spkid":"2000001","fullname":"1 Ceres",
        "des":"1","shortname":"Ceres","neo":false,"kind":"an"},
        "phys_par":[{"name":"GM","value":"62.6284","units":"km^3/s^2"},
                    {"name":"diameter","value":"939.4","units":"km"}]}"#;

    #[test]
    #[serial]
    fn test_lookup_success() {
        let _redirect = CacheRedirect::new();
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET)
                .path("/sbdb.api")
                .query_param("sstr", "Ceres");
            then.status(200).body(CERES_BODY);
        });

        let client = SBDBClient::with_base_url(&server.base_url());
        let obj = client.lookup("Ceres").unwrap();
        assert_eq!(obj.naif_id(), 2000001);
        mock.assert();
    }

    #[test]
    #[serial]
    fn test_lookup_uses_cache_on_second_call() {
        let _redirect = CacheRedirect::new();
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/sbdb.api");
            then.status(200).body(CERES_BODY);
        });

        let client = SBDBClient::with_base_url(&server.base_url());
        let _ = client.lookup("Ceres").unwrap();
        let _ = client.lookup("Ceres").unwrap();
        // Second lookup is served from cache: exactly one HTTP call.
        mock.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_lookup_ambiguous_errors() {
        let _redirect = CacheRedirect::new();
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/sbdb.api");
            then.status(300)
                .body(r#"{"code":"300","list":[{"pdes":"1","name":"Ceres"},{"pdes":"2","name":"Pallas"}]}"#);
        });

        let client = SBDBClient::with_base_url(&server.base_url());
        let err = client.lookup("C").unwrap_err();
        assert!(err.to_string().contains("multiple"));
    }

    #[test]
    #[serial]
    fn test_with_base_url_and_cache_age_ctor() {
        let c = SBDBClient::with_base_url_and_cache_age("https://example.test/", 123);
        assert_eq!(c.base_url, "https://example.test");
        assert_eq!(c.cache_max_age, 123);
    }

    #[test]
    #[serial_test::parallel]
    fn test_new_and_default_ctors() {
        let c = SBDBClient::new();
        assert_eq!(c.base_url, DEFAULT_BASE_URL);
        assert_eq!(c.cache_max_age, DEFAULT_CACHE_MAX_AGE);

        let d = SBDBClient::default();
        assert_eq!(d.base_url, DEFAULT_BASE_URL);
        assert_eq!(d.cache_max_age, DEFAULT_CACHE_MAX_AGE);
    }

    #[test]
    #[serial_test::parallel]
    fn test_with_cache_age_ctor() {
        let c = SBDBClient::with_cache_age(3600);
        assert_eq!(c.base_url, DEFAULT_BASE_URL);
        assert_eq!(c.cache_max_age, 3600);
    }

    #[test]
    #[serial]
    fn test_lookup_write_cache_failure_errors() {
        let _redirect = CacheRedirect::new();
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/sbdb.api");
            then.status(200).body(CERES_BODY);
        });

        // Pre-create a directory at the exact cache-file path so the
        // write's final rename fails, exercising the atomic_write error arm.
        let cache_path = PathBuf::from(get_sbdb_cache_dir().unwrap())
            .join(format!("{}.json", short_hash("Ceres")));
        std::fs::create_dir_all(&cache_path).unwrap();

        let client = SBDBClient::with_base_url(&server.base_url());
        let err = client.lookup("Ceres").unwrap_err();
        assert!(matches!(err, BraheError::IoError(_)));
    }

    #[test]
    #[serial]
    fn test_lookup_zero_cache_age_refetches() {
        let _redirect = CacheRedirect::new();
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/sbdb.api");
            then.status(200).body(CERES_BODY);
        });

        // cache_max_age = 0 forces a refetch every call.
        let client = SBDBClient::with_base_url(&server.base_url());
        let client = SBDBClient {
            base_url: client.base_url,
            cache_max_age: 0,
        };
        let _ = client.lookup("Ceres").unwrap();
        let _ = client.lookup("Ceres").unwrap();
        mock.assert_calls(2);
    }

    #[test]
    #[serial]
    fn test_lookup_offline_serves_stale_cache() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("offline"));
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/sbdb.api");
            then.status(200).body(CERES_BODY);
        });
        let client = SBDBClient::with_base_url_and_cache_age(&server.base_url(), 60);
        let cache_path = PathBuf::from(get_sbdb_cache_dir().unwrap())
            .join(format!("{}.json", short_hash("Ceres")));
        fs::write(&cache_path, CERES_BODY).unwrap();
        let file = fs::File::options().write(true).open(&cache_path).unwrap();
        file.set_modified(SystemTime::now() - Duration::from_secs(86400))
            .unwrap();

        let obj = client.lookup("Ceres").unwrap();
        assert_eq!(obj.naif_id(), 2000001);
        assert_eq!(mock.calls(), 0);
    }

    #[test]
    #[serial]
    fn test_lookup_offline_strict_stale_cache_errors() {
        let _cache = CacheRedirect::new();
        let _mode = NetworkModeGuard::set(Some("offline-strict"));
        let client = SBDBClient::with_base_url_and_cache_age("http://127.0.0.1:9", 60);
        let cache_path = PathBuf::from(get_sbdb_cache_dir().unwrap())
            .join(format!("{}.json", short_hash("Ceres")));
        fs::write(&cache_path, CERES_BODY).unwrap();
        let file = fs::File::options().write(true).open(&cache_path).unwrap();
        file.set_modified(SystemTime::now() - Duration::from_secs(86400))
            .unwrap();

        let err = client.lookup("Ceres").unwrap_err().to_string();
        assert!(err.contains("SBDB lookup cache"), "{err}");
        assert!(err.contains("is older than its cache limit"), "{err}");
    }
}
