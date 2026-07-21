/*!
 * Horizons SPK HTTP client with on-disk caching.
 */

use std::path::PathBuf;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::Deserialize;

use crate::datasets::horizons::request::HorizonsSPKRequest;
use crate::datasets::horizons::response::HorizonsSPKResponse;
use crate::utils::cache::get_horizons_cache_dir;
use crate::utils::download::download_string;
use crate::utils::{BraheError, atomic_write};

/// Default base URL for the JPL Horizons API host.
const DEFAULT_BASE_URL: &str = "https://ssd.jpl.nasa.gov";

#[derive(Deserialize)]
struct HorizonsSPKPayload {
    spk: Option<String>,
    spk_file_id: Option<String>,
    error: Option<String>,
    result: Option<String>,
}

/// Client for the JPL Horizons SPK generation API.
///
/// Generates a targeted SPK for a small body over a time span, caches the
/// `.bsp` under the Horizons cache directory, and returns a
/// [`HorizonsSPKResponse`] handle. A cached kernel for the same request is
/// reused without a network call.
///
/// # Examples
///
/// ```no_run
/// use brahe::datasets::horizons::{HorizonsClient, HorizonsSPKRequest};
/// use brahe::time::{Epoch, TimeSystem};
///
/// let client = HorizonsClient::new();
/// let t0 = Epoch::from_datetime(2015, 12, 1, 0, 0, 0.0, 0.0, TimeSystem::TDB);
/// let t1 = Epoch::from_datetime(2016, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::TDB);
/// let resp = client.get_spk(&HorizonsSPKRequest::for_spkid(20000001, t0, t1)).unwrap();
/// resp.load().unwrap();
/// ```
pub struct HorizonsClient {
    base_url: String,
}

impl Default for HorizonsClient {
    fn default() -> Self {
        Self::new()
    }
}

impl HorizonsClient {
    /// Create a client with the default Horizons base URL.
    pub fn new() -> Self {
        HorizonsClient {
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    /// Create a client pointed at a custom base URL (e.g. a mock server).
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL without a trailing slash.
    pub fn with_base_url(base_url: &str) -> Self {
        HorizonsClient {
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Generate (or reuse a cached) SPK for `request`.
    ///
    /// # Arguments
    ///
    /// * `request` - The SPK generation request.
    ///
    /// # Returns
    ///
    /// * `Ok(HorizonsSPKResponse)` - Handle to the cached `.bsp`.
    /// * `Err(BraheError)` - On network, server, decode, or IO errors.
    pub fn get_spk(&self, request: &HorizonsSPKRequest) -> Result<HorizonsSPKResponse, BraheError> {
        let cache_path = PathBuf::from(get_horizons_cache_dir()?).join(request.cache_key());

        if cache_path.exists() {
            return Ok(HorizonsSPKResponse::new(cache_path, None));
        }

        let url = format!("{}/api/horizons.api{}", self.base_url, request.query());
        let body = download_string(&url, "Horizons SPK")?;

        let payload: HorizonsSPKPayload = serde_json::from_str(&body).map_err(|e| {
            BraheError::ParseError(format!("Failed to parse Horizons response: {}", e))
        })?;

        let spk_b64 = payload.spk.ok_or_else(|| {
            let detail = payload
                .error
                .or(payload.result)
                .unwrap_or_else(|| "no 'spk' field in response".to_string());
            BraheError::Error(format!("Horizons SPK generation failed: {}", detail))
        })?;

        // Horizons wraps the base64-encoded SPK at 76 columns, so strip all
        // embedded whitespace before decoding (the standard engine rejects it).
        let spk_b64: String = spk_b64.chars().filter(|c| !c.is_whitespace()).collect();
        let decoded = BASE64.decode(&spk_b64).map_err(|e| {
            BraheError::ParseError(format!("Failed to base64-decode Horizons SPK: {}", e))
        })?;
        if decoded.is_empty() {
            return Err(BraheError::Error(
                "Horizons returned an empty SPK".to_string(),
            ));
        }

        atomic_write(&cache_path, &decoded).map_err(|e| {
            BraheError::IoError(format!(
                "Failed to write SPK cache {}: {}",
                cache_path.display(),
                e
            ))
        })?;

        Ok(HorizonsSPKResponse::new(cache_path, payload.spk_file_id))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::testing::CacheRedirect;
    use base64::Engine;
    use httpmock::prelude::*;
    use serial_test::serial;

    fn req() -> HorizonsSPKRequest {
        let t0 = Epoch::from_datetime(2015, 12, 1, 0, 0, 0.0, 0.0, TimeSystem::TDB);
        let t1 = Epoch::from_datetime(2016, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::TDB);
        HorizonsSPKRequest::for_spkid(2000001, t0, t1)
    }

    #[test]
    #[serial]
    fn test_get_spk_success_writes_cache() {
        let _redirect = CacheRedirect::new();
        let payload_bytes = b"DAF/SPK fake kernel payload";
        let b64 = BASE64.encode(payload_bytes);
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/api/horizons.api");
            then.status(200)
                .body(format!(r#"{{"spk_file_id":"2000001","spk":"{}"}}"#, b64));
        });

        let client = HorizonsClient::with_base_url(&server.base_url());
        let resp = client.get_spk(&req()).unwrap();
        assert!(resp.path().exists());
        assert_eq!(resp.spk_file_id(), Some("2000001"));
        assert_eq!(resp.bytes().unwrap(), payload_bytes);
        mock.assert();
    }

    #[test]
    #[serial]
    fn test_get_spk_decodes_line_wrapped_base64() {
        // Horizons wraps the base64 SPK at 76 columns with newlines; the
        // decoder must strip embedded whitespace rather than choke on it.
        let _redirect = CacheRedirect::new();
        let payload_bytes = b"DAF/SPK a longer fake kernel payload spanning wrapped base64 lines";
        let b64 = BASE64.encode(payload_bytes);
        // Escaped `\n` in the JSON body decodes to real newlines in the
        // String, reproducing Horizons' 76-column line wrapping.
        let wrapped = b64
            .as_bytes()
            .chunks(76)
            .map(|c| std::str::from_utf8(c).unwrap())
            .collect::<Vec<_>>()
            .join("\\n");
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/api/horizons.api");
            then.status(200).body(format!(r#"{{"spk":"{}"}}"#, wrapped));
        });

        let client = HorizonsClient::with_base_url(&server.base_url());
        let resp = client.get_spk(&req()).unwrap();
        assert_eq!(resp.bytes().unwrap(), payload_bytes);
        mock.assert();
    }

    #[test]
    #[serial]
    fn test_get_spk_reuses_cache() {
        let _redirect = CacheRedirect::new();
        let b64 = BASE64.encode(b"DAF/SPK payload");
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path("/api/horizons.api");
            then.status(200).body(format!(r#"{{"spk":"{}"}}"#, b64));
        });

        let client = HorizonsClient::with_base_url(&server.base_url());
        let _ = client.get_spk(&req()).unwrap();
        let _ = client.get_spk(&req()).unwrap();
        // Second call served from cache: exactly one HTTP call.
        mock.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_get_spk_error_response() {
        let _redirect = CacheRedirect::new();
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/api/horizons.api");
            then.status(200)
                .body(r#"{"error":"Cannot interpret COMMAND"}"#);
        });

        let client = HorizonsClient::with_base_url(&server.base_url());
        let err = client.get_spk(&req()).unwrap_err();
        assert!(err.to_string().contains("Cannot interpret COMMAND"));
    }

    #[test]
    fn test_new_and_default_ctors() {
        let c = HorizonsClient::new();
        assert_eq!(c.base_url, DEFAULT_BASE_URL);

        let d = HorizonsClient::default();
        assert_eq!(d.base_url, DEFAULT_BASE_URL);
    }

    #[test]
    #[serial]
    fn test_get_spk_non_json_response_errors() {
        let _redirect = CacheRedirect::new();
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/api/horizons.api");
            then.status(200).body("not json");
        });

        let client = HorizonsClient::with_base_url(&server.base_url());
        let err = client.get_spk(&req()).unwrap_err();
        assert!(matches!(err, BraheError::ParseError(_)));
    }

    #[test]
    #[serial]
    fn test_get_spk_empty_spk_errors() {
        let _redirect = CacheRedirect::new();
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/api/horizons.api");
            then.status(200).body(r#"{"spk":""}"#);
        });

        let client = HorizonsClient::with_base_url(&server.base_url());
        let err = client.get_spk(&req()).unwrap_err();
        assert!(err.to_string().contains("empty SPK"));
    }

    #[test]
    #[serial]
    fn test_get_spk_invalid_base64_errors() {
        let _redirect = CacheRedirect::new();
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/api/horizons.api");
            then.status(200).body(r#"{"spk":"@@@not-base64@@@"}"#);
        });

        let client = HorizonsClient::with_base_url(&server.base_url());
        let err = client.get_spk(&req()).unwrap_err();
        assert!(matches!(err, BraheError::ParseError(_)));
    }

    #[test]
    #[serial]
    fn test_get_spk_write_cache_failure_errors() {
        let _redirect = CacheRedirect::new();
        let payload_bytes = b"DAF/SPK fake kernel payload";
        let b64 = BASE64.encode(payload_bytes);
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(GET).path("/api/horizons.api");
            then.status(200).body(format!(r#"{{"spk":"{}"}}"#, b64));
        });

        // The cache_path.exists() short-circuit means a pre-created
        // directory at cache_path would be reused as a hit, not hit the
        // write path. Instead, strip the owner write bit from the (already
        // created) horizons cache directory so atomic_write's file create
        // fails, exercising the same error arm.
        let cache_dir = get_horizons_cache_dir().unwrap();
        let original_perms = std::fs::metadata(&cache_dir).unwrap().permissions();
        let mut readonly_perms = original_perms.clone();
        readonly_perms.set_readonly(true);
        std::fs::set_permissions(&cache_dir, readonly_perms).unwrap();

        let client = HorizonsClient::with_base_url(&server.base_url());
        let result = client.get_spk(&req());

        std::fs::set_permissions(&cache_dir, original_perms).unwrap();

        let err = result.unwrap_err();
        assert!(matches!(err, BraheError::IoError(_)));
    }
}
