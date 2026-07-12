//! ICGEM index file: read/write, TTL, and list query.

use crate::datasets::icgem::body::ICGEMBody;
use crate::utils::BraheError;
use crate::utils::cache::get_icgem_cache_dir;
use crate::utils::fs::atomic_write;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// One ICGEM model row: a specific (body, name, degree) triple with its
/// opaque download path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexEntry {
    /// Celestial body this model is for.
    pub body: ICGEMBody,
    /// Model name as listed on ICGEM (e.g. "EGM2008", "GRGM1200B").
    pub name: String,
    /// Publication year, if parseable from the listing page.
    pub year: Option<u16>,
    /// Maximum spherical harmonic degree of this variant.
    pub degree: u32,
    /// Relative URL path including the opaque ICGEM hash, e.g.
    /// `/getmodel/gfc/<hash>/EGM2008.gfc`.
    pub download_path: String,
}

/// On-disk shape of an index cache file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct IndexFile {
    /// Unix timestamp (seconds since epoch) when this index was fetched.
    pub fetched_at: u64,
    /// All entries parsed from the listing page at fetch time.
    pub entries: Vec<IndexEntry>,
}

/// Default TTL before an index file is considered stale (30 days, in seconds).
pub(crate) const DEFAULT_INDEX_TTL_SECONDS: u64 = 30 * 24 * 60 * 60;

/// Filename for the Earth index file.
pub(crate) const EARTH_INDEX_FILE: &str = "index_earth.json";

/// Filename for the celestial (non-Earth) index file.
pub(crate) const CELESTIAL_INDEX_FILE: &str = "index_celestial.json";

/// Resolve the on-disk path for a body's index file.
pub(crate) fn index_path_for(body: &ICGEMBody) -> Result<PathBuf, BraheError> {
    let dir = get_icgem_cache_dir()?;
    let filename = if body.is_earth() {
        EARTH_INDEX_FILE
    } else {
        CELESTIAL_INDEX_FILE
    };
    Ok(PathBuf::from(dir).join(filename))
}

/// Current Unix time in seconds.
pub(crate) fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Read an index file from disk. Returns `Ok(None)` if the file does not exist.
pub(crate) fn read_index_file(path: &Path) -> Result<Option<IndexFile>, BraheError> {
    if !path.exists() {
        return Ok(None);
    }
    let data = std::fs::read_to_string(path).map_err(|e| {
        BraheError::Error(format!(
            "Failed to read ICGEM index file {}: {}",
            path.display(),
            e
        ))
    })?;
    let parsed: IndexFile = serde_json::from_str(&data).map_err(|e| {
        BraheError::Error(format!(
            "Failed to parse ICGEM index file {}: {}",
            path.display(),
            e
        ))
    })?;
    Ok(Some(parsed))
}

/// Write an index file to disk atomically.
pub(crate) fn write_index_file(path: &Path, file: &IndexFile) -> Result<(), BraheError> {
    let data = serde_json::to_string_pretty(file)
        .map_err(|e| BraheError::Error(format!("Failed to serialize ICGEM index: {}", e)))?;
    atomic_write(path, data.as_bytes()).map_err(|e| {
        BraheError::Error(format!(
            "Failed to write ICGEM index to {}: {}",
            path.display(),
            e
        ))
    })
}

/// Production ICGEM base URL. All fetch functions take an explicit base URL
/// (private `_with_url` variants) so tests can redirect to a local mock.
pub(crate) const ICGEM_BASE_URL: &str = "https://icgem.gfz.de";

/// Listing page path for Earth models.
pub(crate) const EARTH_PATH: &str = "/tom_longtime";

/// Listing page path for celestial models.
pub(crate) const CELESTIAL_PATH: &str = "/tom_celestial";

/// Fetch and parse a listing page from a given base URL.
pub(crate) fn fetch_index_with_url(
    body: &ICGEMBody,
    base_url: &str,
) -> Result<IndexFile, BraheError> {
    use std::io::Read;
    let path = if body.is_earth() {
        EARTH_PATH
    } else {
        CELESTIAL_PATH
    };
    let url = format!("{}{}", base_url, path);

    let response = ureq::get(&url).call().map_err(|e| {
        BraheError::Error(format!("Failed to fetch ICGEM index from {}: {}", url, e))
    })?;
    let mut buf = String::new();
    response
        .into_body()
        .into_reader()
        .read_to_string(&mut buf)
        .map_err(|e| {
            BraheError::Error(format!("Failed to read ICGEM index from {}: {}", url, e))
        })?;

    let entries = if body.is_earth() {
        crate::datasets::icgem::parser::parse_earth_catalog(&buf)?
    } else {
        crate::datasets::icgem::parser::parse_celestial_catalog(&buf)?
    };

    Ok(IndexFile {
        fetched_at: now_unix_seconds(),
        entries,
    })
}

/// List models for a body, refreshing the index transparently if stale or
/// missing. On a TTL miss with no network, falls back to the stale cache and
/// logs a warning.
pub fn list_icgem_models(body: ICGEMBody) -> Result<Vec<IndexEntry>, BraheError> {
    list_icgem_models_with_url(&body, ICGEM_BASE_URL)
}

/// Variant of `list_icgem_models` that targets a configurable base URL (for tests).
pub fn list_icgem_models_with_url(
    body: &ICGEMBody,
    base_url: &str,
) -> Result<Vec<IndexEntry>, BraheError> {
    let path = index_path_for(body)?;
    let existing = read_index_file(&path)?;
    let now = now_unix_seconds();

    let needs_refresh = match &existing {
        None => true,
        Some(f) => now.saturating_sub(f.fetched_at) > DEFAULT_INDEX_TTL_SECONDS,
    };

    // The celestial cache file (`index_celestial.json`) holds entries for ALL
    // non-Earth bodies. Always filter on read so a caller asking for Mars
    // doesn't get Moon/Venus/Ceres entries back. Earth's cache only contains
    // Earth entries, so the filter is a no-op there.
    let filter_for_body = |all: Vec<IndexEntry>| -> Vec<IndexEntry> {
        all.into_iter().filter(|e| &e.body == body).collect()
    };

    if !needs_refresh {
        return Ok(filter_for_body(existing.unwrap().entries));
    }

    match fetch_index_with_url(body, base_url) {
        Ok(fresh) => {
            write_index_file(&path, &fresh)?;
            Ok(filter_for_body(fresh.entries))
        }
        Err(fetch_err) => {
            if let Some(stale) = existing {
                eprintln!(
                    "Warning: ICGEM index refresh failed ({}); using stale cache from {}",
                    fetch_err,
                    path.display()
                );
                Ok(filter_for_body(stale.entries))
            } else {
                Err(fetch_err)
            }
        }
    }
}

/// Force-refresh the index for a single body, regardless of TTL.
pub fn refresh_icgem_index(body: ICGEMBody) -> Result<(), BraheError> {
    refresh_icgem_index_with_url(&body, ICGEM_BASE_URL)
}

pub(crate) fn refresh_icgem_index_with_url(
    body: &ICGEMBody,
    base_url: &str,
) -> Result<(), BraheError> {
    let fresh = fetch_index_with_url(body, base_url)?;
    let path = index_path_for(body)?;
    write_index_file(&path, &fresh)
}

/// Force-refresh both Earth and celestial index files.
pub fn refresh_all_icgem_indexes() -> Result<(), BraheError> {
    refresh_icgem_index(ICGEMBody::Earth)?;
    // Any non-Earth body triggers the celestial fetch — pick Moon arbitrarily.
    refresh_icgem_index(ICGEMBody::Moon)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_index_entry_round_trip_json() {
        let entry = IndexEntry {
            body: ICGEMBody::Earth,
            name: "EGM2008".into(),
            year: Some(2008),
            degree: 2190,
            download_path: "/getmodel/gfc/abc/EGM2008.gfc".into(),
        };
        let s = serde_json::to_string(&entry).unwrap();
        let back: IndexEntry = serde_json::from_str(&s).unwrap();
        assert_eq!(entry, back);
    }

    #[test]
    fn test_index_file_round_trip_json() {
        let file = IndexFile {
            fetched_at: 1_700_000_000,
            entries: vec![IndexEntry {
                body: ICGEMBody::Moon,
                name: "GRGM1200B".into(),
                year: Some(2016),
                degree: 1200,
                download_path: "/getmodel/gfc/xyz/GRGM1200B.gfc".into(),
            }],
        };
        let s = serde_json::to_string(&file).unwrap();
        let back: IndexFile = serde_json::from_str(&s).unwrap();
        assert_eq!(back.entries.len(), 1);
        assert_eq!(back.fetched_at, 1_700_000_000);
    }

    #[test]
    fn test_index_path_dispatches_by_body() {
        let earth = index_path_for(&ICGEMBody::Earth).unwrap();
        assert!(earth.to_string_lossy().ends_with("index_earth.json"));
        let moon = index_path_for(&ICGEMBody::Moon).unwrap();
        assert!(moon.to_string_lossy().ends_with("index_celestial.json"));
        let pluto = index_path_for(&ICGEMBody::Other("pluto".into())).unwrap();
        assert!(pluto.to_string_lossy().ends_with("index_celestial.json"));
    }

    #[test]
    fn test_read_index_file_missing_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nope.json");
        assert!(read_index_file(&path).unwrap().is_none());
    }

    #[test]
    fn test_write_then_read_index_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("x.json");
        let file = IndexFile {
            fetched_at: 42,
            entries: vec![],
        };
        write_index_file(&path, &file).unwrap();
        let back = read_index_file(&path).unwrap().unwrap();
        assert_eq!(back.fetched_at, 42);
    }

    #[test]
    fn test_fetch_index_http_404() {
        use httpmock::prelude::*;
        let server = MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(GET).path("/tom_longtime");
            then.status(404);
        });
        let result = fetch_index_with_url(&ICGEMBody::Earth, &server.base_url());
        assert!(result.is_err());
    }

    #[test]
    fn test_fetch_index_success_serves_fixture() {
        use httpmock::prelude::*;
        let fixture =
            std::fs::read_to_string("test_assets/icgem/tom_longtime_sample.html").unwrap();
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(GET).path_includes("/tom_longtime");
            then.status(200).body(&fixture);
        });
        let result = fetch_index_with_url(&ICGEMBody::Earth, &server.base_url());
        let file = result.unwrap();
        assert!(!file.entries.is_empty());
        mock.assert_calls(1);
    }

    #[test]
    #[serial_test::serial]
    fn test_list_models_uses_fresh_cache() {
        let dir = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("BRAHE_CACHE", dir.path());
        }

        let path = index_path_for(&ICGEMBody::Earth).unwrap();
        let file = IndexFile {
            fetched_at: now_unix_seconds(),
            entries: vec![IndexEntry {
                body: ICGEMBody::Earth,
                name: "JGM3".into(),
                year: Some(1996),
                degree: 70,
                download_path: "/getmodel/gfc/abc/JGM3.gfc".into(),
            }],
        };
        write_index_file(&path, &file).unwrap();

        // No HTTP mock — if the implementation tries to fetch, the test fails.
        let entries = list_icgem_models_with_url(&ICGEMBody::Earth, "http://127.0.0.1:1").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "JGM3");

        unsafe {
            std::env::remove_var("BRAHE_CACHE");
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_list_models_refreshes_stale_cache() {
        use httpmock::prelude::*;
        let dir = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("BRAHE_CACHE", dir.path());
        }

        let path = index_path_for(&ICGEMBody::Earth).unwrap();
        let stale = IndexFile {
            fetched_at: 0,
            entries: vec![],
        };
        write_index_file(&path, &stale).unwrap();

        let fixture =
            std::fs::read_to_string("test_assets/icgem/tom_longtime_sample.html").unwrap();
        let server = MockServer::start();
        let _m = server.mock(|when, then| {
            when.method(GET).path_includes("/tom_longtime");
            then.status(200).body(&fixture);
        });

        let entries = list_icgem_models_with_url(&ICGEMBody::Earth, &server.base_url()).unwrap();
        assert!(!entries.is_empty());

        unsafe {
            std::env::remove_var("BRAHE_CACHE");
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_list_models_stale_fallback_on_network_failure() {
        let dir = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("BRAHE_CACHE", dir.path());
        }

        let path = index_path_for(&ICGEMBody::Earth).unwrap();
        let stale = IndexFile {
            fetched_at: 0,
            entries: vec![IndexEntry {
                body: ICGEMBody::Earth,
                name: "STALE".into(),
                year: None,
                degree: 10,
                download_path: "/getmodel/gfc/x/STALE.gfc".into(),
            }],
        };
        write_index_file(&path, &stale).unwrap();

        // Point at an unreachable URL so refresh fails; stale cache should be returned.
        let entries = list_icgem_models_with_url(&ICGEMBody::Earth, "http://127.0.0.1:1").unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name, "STALE");

        unsafe {
            std::env::remove_var("BRAHE_CACHE");
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_list_models_filters_celestial_by_body() {
        // The celestial cache file holds entries for ALL non-Earth bodies.
        // list_icgem_models(Mars) must return only Mars entries — not Moon,
        // Venus, or Ceres ones that share the same file.
        let dir = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("BRAHE_CACHE", dir.path());
        }

        let path = index_path_for(&ICGEMBody::Moon).unwrap(); // celestial index file
        let file = IndexFile {
            fetched_at: now_unix_seconds(),
            entries: vec![
                IndexEntry {
                    body: ICGEMBody::Moon,
                    name: "GRGM1200B".into(),
                    year: Some(2016),
                    degree: 1200,
                    download_path: "/getmodel/gfc/m/GRGM1200B.gfc".into(),
                },
                IndexEntry {
                    body: ICGEMBody::Mars,
                    name: "MRO120F".into(),
                    year: Some(2016),
                    degree: 120,
                    download_path: "/getmodel/gfc/r/MRO120F.gfc".into(),
                },
                IndexEntry {
                    body: ICGEMBody::Venus,
                    name: "MGNP180U".into(),
                    year: Some(1999),
                    degree: 180,
                    download_path: "/getmodel/gfc/v/MGNP180U.gfc".into(),
                },
            ],
        };
        write_index_file(&path, &file).unwrap();

        let moon_entries =
            list_icgem_models_with_url(&ICGEMBody::Moon, "http://127.0.0.1:1").unwrap();
        assert_eq!(moon_entries.len(), 1);
        assert_eq!(moon_entries[0].body, ICGEMBody::Moon);
        assert_eq!(moon_entries[0].name, "GRGM1200B");

        let mars_entries =
            list_icgem_models_with_url(&ICGEMBody::Mars, "http://127.0.0.1:1").unwrap();
        assert_eq!(mars_entries.len(), 1);
        assert_eq!(mars_entries[0].body, ICGEMBody::Mars);
        assert_eq!(mars_entries[0].name, "MRO120F");

        let venus_entries =
            list_icgem_models_with_url(&ICGEMBody::Venus, "http://127.0.0.1:1").unwrap();
        assert_eq!(venus_entries.len(), 1);
        assert_eq!(venus_entries[0].body, ICGEMBody::Venus);

        // A body not present in the file should give an empty list, not error.
        let ceres_entries =
            list_icgem_models_with_url(&ICGEMBody::Ceres, "http://127.0.0.1:1").unwrap();
        assert!(ceres_entries.is_empty());

        unsafe {
            std::env::remove_var("BRAHE_CACHE");
        }
    }
}
