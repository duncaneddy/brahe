/*!
 * Star catalog module.
 *
 * Provides access to fixed-epoch star catalogs (FK5, Hipparcos, Tycho-2)
 * used for reference-frame realization and star-based attitude
 * determination. Unlike other datasets, star catalogs are static: once
 * published they do not change, so cached copies never go stale.
 *
 * Data is downloaded from `https://www.simplespacedata.org/star_catalog/cds`.
 */

pub(crate) mod fetch;
pub mod fk5;
pub mod hipparcos;
pub mod traits;
pub mod tycho2;

pub use fk5::{FK5Catalog, FK5Record};
pub use hipparcos::{HipparcosCatalog, HipparcosRecord};
pub use traits::StarRecord;
pub use tycho2::{Tycho2Catalog, Tycho2Record};

use crate::utils::BraheError;

/// Default base URL for star catalog data files.
pub const DEFAULT_BASE_URL: &str = "https://www.simplespacedata.org/star_catalog/cds";

/// Parse an optional `f64` value from a fixed-width text field.
///
/// Returns `None` for empty or whitespace-only fields, or if the trimmed
/// text cannot be parsed as a floating-point number. This is a deliberately
/// lenient policy: a malformed non-blank numeric field parses as `None`
/// rather than raising a per-field error, consistent with CDS-catalog
/// conventions and the `gcat` parsers this module follows. File-level
/// integrity is instead enforced by the getters (`get_fk5_catalog`,
/// `get_hipparcos_catalog`, `get_tycho2_catalog`), which validate that a
/// downloaded file parses to at least one record before it is written to
/// cache.
pub(crate) fn opt_f64(value: &str) -> Option<f64> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        trimmed.parse::<f64>().ok()
    }
}

/// Parse an optional `String` value from a fixed-width text field.
///
/// Returns `None` for empty or whitespace-only fields.
pub(crate) fn opt_string(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Download and parse the FK5 star catalog.
///
/// Fetches the fixed-width FK5 catalog text file with file-based caching.
/// FK5 is a fixed, published catalog, so the default cache never expires.
///
/// # Arguments
///
/// * `cache_max_age` - Maximum cache age in seconds. `None` means the
///   cached copy never goes stale (the default; appropriate since FK5 does
///   not change once published). Pass `Some(0.0)` to force a fresh download.
///
/// # Returns
///
/// * `Result<FK5Catalog, BraheError>` - Parsed FK5 catalog container
///
/// # Examples
/// ```no_run
/// use brahe::datasets::star_catalogs::get_fk5_catalog;
/// let catalog = get_fk5_catalog(None).unwrap();
/// println!("Loaded {} records", catalog.len());
/// ```
pub fn get_fk5_catalog(cache_max_age: Option<f64>) -> Result<FK5Catalog, BraheError> {
    get_fk5_catalog_from_url(DEFAULT_BASE_URL, cache_max_age)
}

/// As [`get_fk5_catalog`], but fetching from a caller-specified base URL.
///
/// Not part of the public API; the URL seam exists solely so tests can point
/// the getter at a mock server without exposing a public URL override.
///
/// Tries the cache first; on a cache miss, or if the cached content fails to
/// parse (including parsing to zero records), forces a fresh download. The
/// downloaded body is only written to cache after it has been confirmed to
/// parse successfully with at least one record, so a bad download never
/// poisons the cache, and a previously poisoned cache self-heals on the next
/// call.
pub(crate) fn get_fk5_catalog_from_url(
    base_url: &str,
    cache_max_age: Option<f64>,
) -> Result<FK5Catalog, BraheError> {
    let filename = "FK5_Catalog.txt";

    if let Some(cached) = fetch::read_cache(filename, cache_max_age)?
        && let Ok(records) = fk5::parse_fk5_text(&cached)
        && !records.is_empty()
    {
        return Ok(FK5Catalog::new(records));
    }

    let url = format!("{}/fk5/latest/FK5_Catalog.txt", base_url);
    let body = fetch::download(&url)?;
    let records = fk5::parse_fk5_text(&body)?;
    if records.is_empty() {
        return Err(BraheError::ParseError(
            "FK5 catalog download parsed to 0 records".to_string(),
        ));
    }
    fetch::commit_cache(filename, &body)?;
    Ok(FK5Catalog::new(records))
}

/// Download and parse the Hipparcos star catalog.
///
/// Fetches the pipe-delimited Hipparcos catalog text file with file-based
/// caching. Hipparcos is a fixed, published catalog, so the default cache
/// never expires.
///
/// # Arguments
///
/// * `cache_max_age` - Maximum cache age in seconds. `None` means the
///   cached copy never goes stale (the default; appropriate since Hipparcos
///   does not change once published). Pass `Some(0.0)` to force a fresh download.
///
/// # Returns
///
/// * `Result<HipparcosCatalog, BraheError>` - Parsed Hipparcos catalog container
///
/// # Examples
/// ```no_run
/// use brahe::datasets::star_catalogs::get_hipparcos_catalog;
/// let catalog = get_hipparcos_catalog(None).unwrap();
/// println!("Loaded {} records", catalog.len());
/// ```
pub fn get_hipparcos_catalog(cache_max_age: Option<f64>) -> Result<HipparcosCatalog, BraheError> {
    get_hipparcos_catalog_from_url(DEFAULT_BASE_URL, cache_max_age)
}

/// As [`get_hipparcos_catalog`], but fetching from a caller-specified base URL.
///
/// Not part of the public API; the URL seam exists solely so tests can point
/// the getter at a mock server without exposing a public URL override.
///
/// Tries the cache first; on a cache miss, or if the cached content fails to
/// parse (including parsing to zero records), forces a fresh download. The
/// downloaded body is only written to cache after it has been confirmed to
/// parse successfully with at least one record, so a bad download never
/// poisons the cache, and a previously poisoned cache self-heals on the next
/// call.
pub(crate) fn get_hipparcos_catalog_from_url(
    base_url: &str,
    cache_max_age: Option<f64>,
) -> Result<HipparcosCatalog, BraheError> {
    let filename = "Hipparcos_Catalog.txt";

    if let Some(cached) = fetch::read_cache(filename, cache_max_age)?
        && let Ok(records) = hipparcos::parse_hipparcos_text(&cached)
        && !records.is_empty()
    {
        return Ok(HipparcosCatalog::new(records));
    }

    let url = format!("{}/hipparcos/latest/Hipparcos_Catalog.txt", base_url);
    let body = fetch::download(&url)?;
    let records = hipparcos::parse_hipparcos_text(&body)?;
    if records.is_empty() {
        return Err(BraheError::ParseError(
            "Hipparcos catalog download parsed to 0 records".to_string(),
        ));
    }
    fetch::commit_cache(filename, &body)?;
    Ok(HipparcosCatalog::new(records))
}

/// Download and parse the Tycho-2 star catalog.
///
/// Fetches the pipe-delimited Tycho-2 catalog text file with file-based
/// caching. Tycho-2 is a fixed, published catalog, so the default cache
/// never expires. The source file is large (~526 MB, ~2.54 million
/// records), so the first call may take some time.
///
/// # Arguments
///
/// * `cache_max_age` - Maximum cache age in seconds. `None` means the
///   cached copy never goes stale (the default; appropriate since Tycho-2
///   does not change once published). Pass `Some(0.0)` to force a fresh download.
///
/// # Returns
///
/// * `Result<Tycho2Catalog, BraheError>` - Parsed Tycho-2 catalog container
///
/// # Examples
/// ```no_run
/// use brahe::datasets::star_catalogs::get_tycho2_catalog;
/// let catalog = get_tycho2_catalog(None).unwrap();
/// println!("Loaded {} records", catalog.len());
/// ```
pub fn get_tycho2_catalog(cache_max_age: Option<f64>) -> Result<Tycho2Catalog, BraheError> {
    get_tycho2_catalog_from_url(DEFAULT_BASE_URL, cache_max_age)
}

/// As [`get_tycho2_catalog`], but fetching from a caller-specified base URL.
///
/// Not part of the public API; the URL seam exists solely so tests can point
/// the getter at a mock server without exposing a public URL override.
///
/// Tries the cache first; on a cache miss, or if the cached content fails to
/// parse (including parsing to zero records), forces a fresh download. The
/// downloaded body is only written to cache after it has been confirmed to
/// parse successfully with at least one record, so a bad download never
/// poisons the cache, and a previously poisoned cache self-heals on the next
/// call.
pub(crate) fn get_tycho2_catalog_from_url(
    base_url: &str,
    cache_max_age: Option<f64>,
) -> Result<Tycho2Catalog, BraheError> {
    let filename = "Tycho2_Catalog.txt";

    if let Some(cached) = fetch::read_cache(filename, cache_max_age)?
        && let Ok(records) = tycho2::parse_tycho2_text(&cached)
        && !records.is_empty()
    {
        return Ok(Tycho2Catalog::new(records));
    }

    let url = format!("{}/tycho2/latest/Tycho2_Catalog.txt", base_url);
    let body = fetch::download(&url)?;
    let records = tycho2::parse_tycho2_text(&body)?;
    if records.is_empty() {
        return Err(BraheError::ParseError(
            "Tycho-2 catalog download parsed to 0 records".to_string(),
        ));
    }
    fetch::commit_cache(filename, &body)?;
    Ok(Tycho2Catalog::new(records))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::fs;
    use std::path::Path;

    use serial_test::serial;

    use crate::utils::cache::get_star_catalogs_cache_dir;

    use super::*;

    const SAMPLE: &str = include_str!("../../../test_assets/star_catalogs/FK5_Catalog_sample.txt");
    const HIPPARCOS_SAMPLE: &str =
        include_str!("../../../test_assets/star_catalogs/Hipparcos_Catalog_sample.txt");
    const TYCHO2_SAMPLE: &str =
        include_str!("../../../test_assets/star_catalogs/Tycho2_Catalog_sample.txt");

    // All tests below read/write a real star catalog cache file
    // (`FK5_Catalog.txt`/`Hipparcos_Catalog.txt`/`Tycho2_Catalog.txt`), so
    // tests sharing the same file must not run concurrently with each other
    // (tests for different catalogs use distinct files and may still run in
    // parallel with each other).
    #[test]
    #[serial(fk5_catalog_cache)]
    fn test_get_catalog_does_not_cache_bad_download() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("GET")
                .path_includes("/fk5/latest/FK5_Catalog.txt");
            then.status(200)
                .body("this is not a valid FK5 catalog file\ngarbage\n");
        });

        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        let cache_path = Path::new(&cache_dir).join("FK5_Catalog.txt");
        let _ = fs::remove_file(&cache_path); // start from a clean (no-cache) state

        let result = get_fk5_catalog_from_url(&server.base_url(), None);
        assert!(result.is_err());
        assert!(
            !cache_path.exists(),
            "a failed parse of freshly downloaded data must not be cached"
        );

        mock.assert_calls(1);
    }

    #[test]
    #[serial(fk5_catalog_cache)]
    fn test_get_catalog_heals_corrupt_cache() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("GET")
                .path_includes("/fk5/latest/FK5_Catalog.txt");
            then.status(200).body(SAMPLE);
        });

        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        let cache_path = Path::new(&cache_dir).join("FK5_Catalog.txt");
        fs::write(&cache_path, "corrupt cache data that will not parse\n").unwrap();

        let result = get_fk5_catalog_from_url(&server.base_url(), None);
        assert!(
            result.is_ok(),
            "corrupt cache must self-heal via re-download"
        );
        assert_eq!(result.unwrap().len(), 10);

        mock.assert_calls(1);

        let cached_contents = fs::read_to_string(&cache_path).unwrap();
        assert_eq!(
            cached_contents, SAMPLE,
            "cache must be replaced with the valid download"
        );

        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    #[serial(fk5_catalog_cache)]
    fn test_get_fk5_catalog_from_url_cache_hit_skips_download() {
        // A fresh, valid cache file present on disk is used directly: the
        // getter must not touch the network at all.
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("GET")
                .path_includes("/fk5/latest/FK5_Catalog.txt");
            then.status(200).body(SAMPLE);
        });

        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        let cache_path = Path::new(&cache_dir).join("FK5_Catalog.txt");
        fs::write(&cache_path, SAMPLE).unwrap();

        let result = get_fk5_catalog_from_url(&server.base_url(), None);
        assert_eq!(result.unwrap().len(), 10);
        mock.assert_calls(0);

        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    #[serial(fk5_catalog_cache)]
    fn test_get_fk5_catalog_from_url_empty_records_not_cached() {
        // A download that parses successfully but to zero records (e.g. a
        // body of only blank lines) is rejected rather than cached.
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("GET")
                .path_includes("/fk5/latest/FK5_Catalog.txt");
            then.status(200).body("\n\n");
        });

        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        let cache_path = Path::new(&cache_dir).join("FK5_Catalog.txt");
        let _ = fs::remove_file(&cache_path);

        let result = get_fk5_catalog_from_url(&server.base_url(), None);
        assert!(result.is_err());
        assert!(!cache_path.exists());

        mock.assert_calls(1);
    }

    #[test]
    #[serial(hipparcos_catalog_cache)]
    fn test_get_hipparcos_catalog_from_url_does_not_cache_bad_download() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("GET")
                .path_includes("/hipparcos/latest/Hipparcos_Catalog.txt");
            then.status(200).body("H|not-a-number|garbage\n");
        });

        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        let cache_path = Path::new(&cache_dir).join("Hipparcos_Catalog.txt");
        let _ = fs::remove_file(&cache_path);

        let result = get_hipparcos_catalog_from_url(&server.base_url(), None);
        assert!(result.is_err());
        assert!(!cache_path.exists());

        mock.assert_calls(1);
    }

    #[test]
    #[serial(hipparcos_catalog_cache)]
    fn test_get_hipparcos_catalog_from_url_heals_corrupt_cache() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("GET")
                .path_includes("/hipparcos/latest/Hipparcos_Catalog.txt");
            then.status(200).body(HIPPARCOS_SAMPLE);
        });

        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        let cache_path = Path::new(&cache_dir).join("Hipparcos_Catalog.txt");
        fs::write(&cache_path, "corrupt cache data that will not parse\n").unwrap();

        let result = get_hipparcos_catalog_from_url(&server.base_url(), None);
        assert_eq!(result.unwrap().len(), 20);
        mock.assert_calls(1);

        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    #[serial(hipparcos_catalog_cache)]
    fn test_get_hipparcos_catalog_from_url_cache_hit_skips_download() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("GET")
                .path_includes("/hipparcos/latest/Hipparcos_Catalog.txt");
            then.status(200).body(HIPPARCOS_SAMPLE);
        });

        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        let cache_path = Path::new(&cache_dir).join("Hipparcos_Catalog.txt");
        fs::write(&cache_path, HIPPARCOS_SAMPLE).unwrap();

        let result = get_hipparcos_catalog_from_url(&server.base_url(), None);
        assert_eq!(result.unwrap().len(), 20);
        mock.assert_calls(0);

        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    #[serial(hipparcos_catalog_cache)]
    fn test_get_hipparcos_catalog_from_url_empty_records_not_cached() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("GET")
                .path_includes("/hipparcos/latest/Hipparcos_Catalog.txt");
            then.status(200).body("\n\n");
        });

        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        let cache_path = Path::new(&cache_dir).join("Hipparcos_Catalog.txt");
        let _ = fs::remove_file(&cache_path);

        let result = get_hipparcos_catalog_from_url(&server.base_url(), None);
        assert!(result.is_err());
        assert!(!cache_path.exists());

        mock.assert_calls(1);
    }

    #[test]
    #[serial(tycho2_catalog_cache)]
    fn test_get_tycho2_catalog_from_url_does_not_cache_bad_download() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("GET")
                .path_includes("/tycho2/latest/Tycho2_Catalog.txt");
            then.status(200).body("not a valid tycho2 line\n");
        });

        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        let cache_path = Path::new(&cache_dir).join("Tycho2_Catalog.txt");
        let _ = fs::remove_file(&cache_path);

        let result = get_tycho2_catalog_from_url(&server.base_url(), None);
        assert!(result.is_err());
        assert!(!cache_path.exists());

        mock.assert_calls(1);
    }

    #[test]
    #[serial(tycho2_catalog_cache)]
    fn test_get_tycho2_catalog_from_url_heals_corrupt_cache() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("GET")
                .path_includes("/tycho2/latest/Tycho2_Catalog.txt");
            then.status(200).body(TYCHO2_SAMPLE);
        });

        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        let cache_path = Path::new(&cache_dir).join("Tycho2_Catalog.txt");
        fs::write(&cache_path, "corrupt cache data that will not parse\n").unwrap();

        let result = get_tycho2_catalog_from_url(&server.base_url(), None);
        assert_eq!(result.unwrap().len(), 20);
        mock.assert_calls(1);

        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    #[serial(tycho2_catalog_cache)]
    fn test_get_tycho2_catalog_from_url_cache_hit_skips_download() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("GET")
                .path_includes("/tycho2/latest/Tycho2_Catalog.txt");
            then.status(200).body(TYCHO2_SAMPLE);
        });

        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        let cache_path = Path::new(&cache_dir).join("Tycho2_Catalog.txt");
        fs::write(&cache_path, TYCHO2_SAMPLE).unwrap();

        let result = get_tycho2_catalog_from_url(&server.base_url(), None);
        assert_eq!(result.unwrap().len(), 20);
        mock.assert_calls(0);

        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    #[serial(tycho2_catalog_cache)]
    fn test_get_tycho2_catalog_from_url_empty_records_not_cached() {
        let server = httpmock::MockServer::start();
        let mock = server.mock(|when, then| {
            when.method("GET")
                .path_includes("/tycho2/latest/Tycho2_Catalog.txt");
            then.status(200).body("\n\n");
        });

        let cache_dir = get_star_catalogs_cache_dir().unwrap();
        let cache_path = Path::new(&cache_dir).join("Tycho2_Catalog.txt");
        let _ = fs::remove_file(&cache_path);

        let result = get_tycho2_catalog_from_url(&server.base_url(), None);
        assert!(result.is_err());
        assert!(!cache_path.exists());

        mock.assert_calls(1);
    }
}
