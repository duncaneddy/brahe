/*!
 * SBDB Lookup HTTP client with on-disk caching.
 */

use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use crate::datasets::sbdb::responses::SBDBObject;
use crate::utils::cache::{get_sbdb_cache_dir, short_hash};
use crate::utils::download::{download_string_no_redirect, urlencode};
use crate::utils::{BraheError, atomic_write};

/// Default base URL for the JPL SSD/SBDB API.
const DEFAULT_BASE_URL: &str = "https://ssd-api.jpl.nasa.gov";
/// Default cache max age: 30 days.
const DEFAULT_CACHE_MAX_AGE: u64 = 30 * 24 * 60 * 60;

/// Client for the JPL Small-Body Database (SBDB) Lookup API.
///
/// Resolves a search string to an [`SBDBObject`]. Responses are cached on disk
/// under the SBDB cache directory and reused until `cache_max_age` elapses.
///
/// # Examples
///
/// ```no_run
/// use brahe::datasets::sbdb::SBDBClient;
///
/// let client = SBDBClient::new();
/// let ceres = client.lookup("Ceres").unwrap();
/// assert_eq!(ceres.naif_id(), 2000001);
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

    /// Resolve a search string (name or designation) to an [`SBDBObject`].
    ///
    /// # Arguments
    ///
    /// * `sstr` - Object search string, e.g. `"Ceres"` or `"2000001"`.
    ///
    /// # Returns
    ///
    /// * `Ok(SBDBObject)` - The resolved object.
    /// * `Err(BraheError)` - On ambiguous/no match, network, or parse errors.
    pub fn lookup(&self, sstr: &str) -> Result<SBDBObject, BraheError> {
        let url = format!(
            "{}/sbdb.api?sstr={}&phys-par=1",
            self.base_url,
            urlencode(sstr)
        );

        let cache_path =
            PathBuf::from(get_sbdb_cache_dir()?).join(format!("{}.json", short_hash(sstr)));

        // Fall through to refetch on a stale/corrupt cache parse failure.
        if let Some(body) = self.read_fresh_cache(&cache_path)
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

    /// Return the cached body if present and younger than `cache_max_age`.
    fn read_fresh_cache(&self, path: &std::path::Path) -> Option<String> {
        let metadata = std::fs::metadata(path).ok()?;
        let modified = metadata.modified().ok()?;
        let age = SystemTime::now()
            .duration_since(modified)
            .unwrap_or(Duration::ZERO);
        if age <= Duration::from_secs(self.cache_max_age) {
            std::fs::read_to_string(path).ok()
        } else {
            None
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::utils::testing::CacheRedirect;
    use httpmock::prelude::*;
    use serial_test::serial;

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
}
