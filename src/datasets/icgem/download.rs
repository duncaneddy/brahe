//! ICGEM model name → entry resolution and download.

use crate::datasets::icgem::body::ICGEMBody;
use crate::datasets::icgem::index::{IndexEntry, index_path_for, read_index_file};
use crate::utils::BraheError;
use crate::utils::network::{NetworkMode, ensure_online, network_mode};

/// Resolve `name` (optionally with `-<degree>` suffix) against `entries` for
/// `body`. Returns the matched `IndexEntry`, or an error with a helpful hint.
///
/// Algorithm:
/// 1. Filter entries to `body`.
/// 2. Exact name match → return that variant (largest degree if multiple).
/// 3. Strip a trailing `-<digits>` suffix and retry exact-name match; if name
///    matches but the requested degree does not, error listing available degrees.
/// 4. No match → error listing the 3 nearest names by edit distance.
pub fn resolve_icgem_model<'a>(
    body: &ICGEMBody,
    name: &str,
    entries: &'a [IndexEntry],
) -> Result<&'a IndexEntry, BraheError> {
    let body_entries: Vec<&IndexEntry> = entries.iter().filter(|e| &e.body == body).collect();

    // Step 2: exact-name match.
    let exact: Vec<&IndexEntry> = body_entries
        .iter()
        .copied()
        .filter(|e| e.name == name)
        .collect();
    if !exact.is_empty() {
        let best = exact.iter().copied().max_by_key(|e| e.degree).unwrap();
        return Ok(best);
    }

    // Step 3: strip `-<digits>` suffix.
    if let Some((base, suffix)) = name.rsplit_once('-')
        && let Ok(req_degree) = suffix.parse::<u32>()
    {
        let base_matches: Vec<&IndexEntry> = body_entries
            .iter()
            .copied()
            .filter(|e| e.name == base)
            .collect();
        if !base_matches.is_empty() {
            if let Some(match_with_degree) = base_matches
                .iter()
                .copied()
                .find(|e| e.degree == req_degree)
            {
                return Ok(match_with_degree);
            }
            let degrees: Vec<u32> = base_matches.iter().map(|e| e.degree).collect();
            return Err(BraheError::Error(format!(
                "ICGEM model '{}' has no variant at degree {}. Available: {:?}",
                base, req_degree, degrees
            )));
        }
    }

    // Step 4: typo hint.
    let nearest = nearest_names(name, &body_entries, 3);
    Err(BraheError::Error(format!(
        "ICGEM model '{}' not found for body '{}'. Did you mean: {}?",
        name,
        body.as_name(),
        nearest.join(", ")
    )))
}

fn nearest_names(target: &str, entries: &[&IndexEntry], k: usize) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut unique: Vec<&IndexEntry> = Vec::new();
    for e in entries {
        if seen.insert(e.name.clone()) {
            unique.push(e);
        }
    }
    let mut scored: Vec<(usize, String)> = unique
        .iter()
        .map(|e| (levenshtein(target, &e.name), e.name.clone()))
        .collect();
    scored.sort_by_key(|(d, _)| *d);
    scored.into_iter().take(k).map(|(_, n)| n).collect()
}

fn levenshtein(a: &str, b: &str) -> usize {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    let (n, m) = (a.len(), b.len());
    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr = vec![0usize; m + 1];
    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (curr[j - 1] + 1).min(prev[j] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[m]
}

use crate::datasets::icgem::index::{ICGEM_BASE_URL, list_icgem_models_with_url};
use crate::utils::cache::get_icgem_cache_dir;
use crate::utils::fs::atomic_write;
use std::path::{Path, PathBuf};

/// Number of leading hex characters of the ICGEM download hash to embed in
/// the cache filename. Twelve characters gives 48 bits of entropy — collisions
/// are astronomically unlikely across a single model's variants, while keeping
/// filenames human-readable.
const ICGEM_CACHE_HASH_LEN: usize = 12;

/// Extract ICGEM's opaque hash segment from a download path of the form
/// `/getmodel/gfc/<hash>/<name>.gfc`. Returns `None` if the path doesn't
/// match the expected shape.
fn extract_icgem_hash(download_path: &str) -> Option<&str> {
    download_path
        .strip_prefix("/getmodel/gfc/")
        .and_then(|s| s.split('/').next())
        .filter(|h| !h.is_empty())
}

/// Build the local cache filename for an index entry.
///
/// Format: `<name>-<degree>-<hashprefix>.gfc`. Embedding the hash means that
/// if ICGEM republishes a model under the same name+degree but a new hashed
/// URL (e.g. a corrected coefficient set), the cache path changes and the new
/// file is fetched on next access — rather than serving the stale local copy
/// forever.
fn cache_filename_for_entry(entry: &IndexEntry) -> String {
    let hash = extract_icgem_hash(&entry.download_path).unwrap_or("nohash");
    let short = &hash[..hash.len().min(ICGEM_CACHE_HASH_LEN)];
    format!("{}-{}-{}.gfc", entry.name, entry.degree, short)
}

/// Compute the local cache file path for a resolved ICGEM index entry.
///
/// # Arguments
///
/// * `body` - Celestial body the model is for
/// * `entry` - Resolved index entry for the model
/// * `cache_root` - Root of the ICGEM cache directory
///
/// # Returns
///
/// * `PathBuf` - Path the model's `.gfc` file is (or would be) cached at
fn model_cache_path(body: &ICGEMBody, entry: &IndexEntry, cache_root: &Path) -> PathBuf {
    let body_subdir = match body {
        ICGEMBody::Earth => "earth".to_string(),
        ICGEMBody::Moon => "moon".to_string(),
        ICGEMBody::Mars => "mars".to_string(),
        ICGEMBody::Venus => "venus".to_string(),
        ICGEMBody::Ceres => "ceres".to_string(),
        ICGEMBody::Other(n) => format!("other/{}", n),
    };
    cache_root
        .join("models")
        .join(&body_subdir)
        .join(cache_filename_for_entry(entry))
}

/// Look up an already-downloaded model by resolving `name` against the cached
/// index file directly, ignoring the index's time-to-live.
///
/// Callers should only use this in `offline` or `offline-strict` mode: it
/// lets a model that has already been downloaded be served even when the
/// cached index itself is stale, since the model file has no time-to-live
/// of its own once present. In `online` mode a stale index should always be
/// refreshed instead, so a model republished under a new hash is re-fetched.
///
/// # Arguments
///
/// * `body` - Celestial body the model is for
/// * `name` - ICGEM model name, optionally with a `-<degree>` suffix
/// * `cache_root` - Root of the ICGEM cache directory
///
/// # Returns
///
/// * `Some(PathBuf)` - Path to the already-downloaded model file
/// * `None` - No cached index, `name` does not resolve against it, or the
///   resolved model's file is not on disk
fn cached_model_path(body: &ICGEMBody, name: &str, cache_root: &Path) -> Option<PathBuf> {
    let index_path = index_path_for(body).ok()?;
    let index = read_index_file(&index_path).ok().flatten()?;
    let entry = resolve_icgem_model(body, name, &index.entries).ok()?;
    let cache_file = model_cache_path(body, entry, cache_root);
    cache_file.exists().then_some(cache_file)
}

/// Copy `cache_file` to `output_path` if given, otherwise return it as-is.
///
/// # Arguments
///
/// * `cache_file` - Path to the cached model file
/// * `output_path` - If `Some`, also copy the model file to this path
///
/// # Returns
///
/// * `Ok(PathBuf)` - `output_path` if given, otherwise `cache_file`
/// * `Err(BraheError)` - On failure to create the output directory or copy the file
fn finalize_model_path(
    cache_file: &Path,
    output_path: Option<PathBuf>,
) -> Result<PathBuf, BraheError> {
    match output_path {
        Some(out) => {
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    BraheError::Error(format!(
                        "Failed to create output directory {}: {}",
                        parent.display(),
                        e
                    ))
                })?;
            }
            std::fs::copy(cache_file, &out).map_err(|e| {
                BraheError::Error(format!(
                    "Failed to copy ICGEM model from {} to {}: {}",
                    cache_file.display(),
                    out.display(),
                    e
                ))
            })?;
            Ok(out)
        }
        None => Ok(cache_file.to_path_buf()),
    }
}

/// Download (or load from cache) a `.gfc` file for the named ICGEM model.
///
/// If `output_path` is `Some`, also copies the cached file there and returns
/// that path. Otherwise returns the cache path.
///
/// Requests are refused when `BRAHE_NETWORK_MODE` is `offline` or
/// `offline-strict`; see [`crate::utils::network`]. In `offline` and
/// `offline-strict` mode, an already-downloaded model is served even when
/// the cached index is past its time-to-live; resolving a model name that
/// has not yet been downloaded still requires the index, so it is subject to
/// the index's time-to-live. In `online` mode a stale index is always
/// refreshed first, so a model republished under a new hash is still
/// re-fetched.
///
/// # Arguments
///
/// * `body` - Celestial body the model is for
/// * `name` - ICGEM model name, optionally with a `-<degree>` suffix
/// * `output_path` - If `Some`, also copy the model file to this path
///
/// # Returns
///
/// * `Ok(PathBuf)` - Path to the model file: `output_path` if given, otherwise
///   the cache path
/// * `Err(BraheError)` - On index/model resolution, I/O, or network errors
///
/// # Examples
///
/// ```no_run
/// use brahe::datasets::icgem::{ICGEMBody, download_icgem_model};
///
/// let path = download_icgem_model(ICGEMBody::Earth, "JGM3", None).unwrap();
/// assert!(path.exists());
/// ```
pub fn download_icgem_model(
    body: ICGEMBody,
    name: &str,
    output_path: Option<PathBuf>,
) -> Result<PathBuf, BraheError> {
    download_icgem_model_with_url(&body, name, output_path, ICGEM_BASE_URL)
}

/// Variant of [`download_icgem_model`] that targets a configurable base URL
/// (for tests).
///
/// # Arguments
///
/// * `body` - Celestial body the model is for
/// * `name` - ICGEM model name, optionally with a `-<degree>` suffix
/// * `output_path` - If `Some`, also copy the model file to this path
/// * `base_url` - Base URL to fetch from (production or a test mock)
///
/// # Returns
///
/// * `Ok(PathBuf)` - Path to the model file: `output_path` if given, otherwise
///   the cache path
/// * `Err(BraheError)` - On index/model resolution, I/O, or network errors
pub(crate) fn download_icgem_model_with_url(
    body: &ICGEMBody,
    name: &str,
    output_path: Option<PathBuf>,
    base_url: &str,
) -> Result<PathBuf, BraheError> {
    let cache_root = PathBuf::from(get_icgem_cache_dir()?);

    if network_mode()? != NetworkMode::Online
        && let Some(cache_file) = cached_model_path(body, name, &cache_root)
    {
        return finalize_model_path(&cache_file, output_path);
    }

    let entries = list_icgem_models_with_url(body, base_url)?;
    let entry = resolve_icgem_model(body, name, &entries)?.clone();
    let cache_file = model_cache_path(body, &entry, &cache_root);

    if !cache_file.exists() {
        let url = format!("{}{}", base_url, entry.download_path);
        ensure_online(&url, &format!("ICGEM model {}", entry.name))?;
        let response = ureq::get(&url).call().map_err(|e| {
            BraheError::Error(format!(
                "Failed to download ICGEM model '{}': {}",
                entry.name, e
            ))
        })?;
        use std::io::Read;
        let mut buf = Vec::new();
        response
            .into_body()
            .into_reader()
            .read_to_end(&mut buf)
            .map_err(|e| {
                BraheError::Error(format!(
                    "Failed to read ICGEM model '{}' body: {}",
                    entry.name, e
                ))
            })?;
        if buf.is_empty() {
            return Err(BraheError::Error(format!(
                "Empty response for ICGEM model '{}'",
                entry.name
            )));
        }
        atomic_write(&cache_file, &buf).map_err(|e| {
            BraheError::Error(format!(
                "Failed to write ICGEM model cache {}: {}",
                cache_file.display(),
                e
            ))
        })?;
    }

    finalize_model_path(&cache_file, output_path)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::datasets::icgem::index::{
        IndexFile, index_path_for, now_unix_seconds, write_index_file,
    };
    use crate::utils::testing::CacheRedirect;
    use crate::utils::testing::NetworkModeGuard;
    use serial_test::{parallel, serial};

    fn entry(body: ICGEMBody, name: &str, degree: u32) -> IndexEntry {
        IndexEntry {
            body,
            name: name.into(),
            year: None,
            degree,
            download_path: format!("/getmodel/gfc/h/{}.gfc", name),
        }
    }

    fn earth_fixture() -> Vec<IndexEntry> {
        vec![
            entry(ICGEMBody::Earth, "JGM3", 70),
            entry(ICGEMBody::Earth, "EGM2008", 2190),
            entry(ICGEMBody::Earth, "WHU-CASM-UGM2025_2159", 760),
            entry(ICGEMBody::Earth, "WHU-CASM-UGM2025_2159", 2190),
            entry(ICGEMBody::Earth, "WHU-CASM-UGM2025_2159", 11000),
            entry(ICGEMBody::Moon, "GRGM1200B", 1200),
        ]
    }

    #[test]
    #[parallel]
    fn test_resolve_exact_single_variant() {
        let entries = earth_fixture();
        let got = resolve_icgem_model(&ICGEMBody::Earth, "JGM3", &entries).unwrap();
        assert_eq!(got.name, "JGM3");
        assert_eq!(got.degree, 70);
    }

    #[test]
    #[parallel]
    fn test_resolve_largest_degree_when_ambiguous() {
        let entries = earth_fixture();
        let got =
            resolve_icgem_model(&ICGEMBody::Earth, "WHU-CASM-UGM2025_2159", &entries).unwrap();
        assert_eq!(got.degree, 11000);
    }

    #[test]
    #[parallel]
    fn test_resolve_with_explicit_degree_suffix() {
        let entries = earth_fixture();
        let got =
            resolve_icgem_model(&ICGEMBody::Earth, "WHU-CASM-UGM2025_2159-2190", &entries).unwrap();
        assert_eq!(got.degree, 2190);
    }

    #[test]
    #[parallel]
    fn test_resolve_missing_degree_errors_with_available_list() {
        let entries = earth_fixture();
        let err = resolve_icgem_model(&ICGEMBody::Earth, "WHU-CASM-UGM2025_2159-99", &entries)
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no variant at degree 99"));
        assert!(msg.contains("760") && msg.contains("2190") && msg.contains("11000"));
    }

    #[test]
    #[parallel]
    fn test_resolve_typo_returns_nearest_names() {
        let entries = earth_fixture();
        let err = resolve_icgem_model(&ICGEMBody::Earth, "EGM200", &entries).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("EGM2008"));
    }

    #[test]
    #[parallel]
    fn test_resolve_other_body_does_not_leak_earth_results() {
        let entries = earth_fixture();
        let err = resolve_icgem_model(&ICGEMBody::Mars, "EGM2008", &entries).unwrap_err();
        assert!(err.to_string().contains("not found for body 'Mars'"));
    }

    #[test]
    #[parallel]
    fn test_resolve_exact_match_takes_precedence_over_suffix_split() {
        let mut entries = earth_fixture();
        entries.push(entry(ICGEMBody::Earth, "MODEL-X-2020", 200));
        let got = resolve_icgem_model(&ICGEMBody::Earth, "MODEL-X-2020", &entries).unwrap();
        assert_eq!(got.name, "MODEL-X-2020");
        assert_eq!(got.degree, 200);
    }

    #[test]
    #[serial]
    fn test_download_end_to_end_with_mock_server() {
        use httpmock::prelude::*;

        let _cache = CacheRedirect::new();

        let html = std::fs::read_to_string("test_assets/icgem/tom_longtime_sample.html").unwrap();
        let gfc = std::fs::read_to_string("data/gravity_models/JGM3.gfc").unwrap();

        let server = MockServer::start();
        let _list = server.mock(|when, then| {
            when.method(GET).path_includes("/tom_longtime");
            then.status(200).body(&html);
        });
        let _file = server.mock(|when, then| {
            when.method(GET).path_includes("/getmodel/gfc/");
            then.status(200).body(&gfc);
        });

        // Discover a model name from the fixture dynamically.
        let entries = crate::datasets::icgem::parser::parse_earth_catalog(&html).unwrap();
        let target = entries
            .first()
            .expect("fixture has at least one entry")
            .name
            .clone();

        let path =
            download_icgem_model_with_url(&ICGEMBody::Earth, &target, None, &server.base_url())
                .unwrap();
        assert!(path.exists());
        assert!(path.to_string_lossy().contains("models"));
        assert!(path.to_string_lossy().contains("earth"));
    }

    #[test]
    #[serial]
    fn test_download_offline_errors_without_request() {
        use httpmock::prelude::*;

        let _cache = CacheRedirect::new();

        let html = std::fs::read_to_string("test_assets/icgem/tom_longtime_sample.html").unwrap();
        let gfc = std::fs::read_to_string("data/gravity_models/JGM3.gfc").unwrap();

        let server = MockServer::start();
        let _list = server.mock(|when, then| {
            when.method(GET).path_includes("/tom_longtime");
            then.status(200).body(&html);
        });
        let download_mock = server.mock(|when, then| {
            when.method(GET).path_includes("/getmodel/gfc/");
            then.status(200).body(&gfc);
        });

        let entries = crate::datasets::icgem::parser::parse_earth_catalog(&html).unwrap();
        let target = entries.first().unwrap().name.clone();

        // Warm the index online so the offline failure is the model download itself.
        list_icgem_models_with_url(&ICGEMBody::Earth, &server.base_url()).unwrap();

        let _mode = NetworkModeGuard::set(Some("offline"));
        let err = download_icgem_model_with_url(
            &ICGEMBody::Earth,
            &target,
            None,
            "https://icgem.invalid",
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.starts_with("BRAHE_NETWORK_MODE is offline; ICGEM model "),
            "{err}"
        );
        download_mock.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_download_uses_cache_on_second_call() {
        use httpmock::prelude::*;

        let _cache = CacheRedirect::new();

        let html = std::fs::read_to_string("test_assets/icgem/tom_longtime_sample.html").unwrap();
        let gfc = std::fs::read_to_string("data/gravity_models/JGM3.gfc").unwrap();

        let server = MockServer::start();
        let list_mock = server.mock(|when, then| {
            when.method(GET).path_includes("/tom_longtime");
            then.status(200).body(&html);
        });
        let download_mock = server.mock(|when, then| {
            when.method(GET).path_includes("/getmodel/gfc/");
            then.status(200).body(&gfc);
        });

        let entries = crate::datasets::icgem::parser::parse_earth_catalog(&html).unwrap();
        let target = entries.first().unwrap().name.clone();

        let _ = download_icgem_model_with_url(&ICGEMBody::Earth, &target, None, &server.base_url())
            .unwrap();
        let _ = download_icgem_model_with_url(&ICGEMBody::Earth, &target, None, &server.base_url())
            .unwrap();

        // Only one HTTP fetch for the file itself.
        download_mock.assert_calls(1);
        // The index listing is fetched only once: the first call writes it to
        // disk with fetched_at = now, so the second call finds a fresh cache
        // and skips the network entirely.
        list_mock.assert_calls(1);
    }

    #[test]
    #[serial]
    fn test_download_offline_strict_serves_cached_model_with_stale_index() {
        use httpmock::prelude::*;

        let _cache = CacheRedirect::new();

        let gfc = std::fs::read_to_string("data/gravity_models/JGM3.gfc").unwrap();
        let target_entry = entry(ICGEMBody::Earth, "JGM3", 70);

        let cache_root = PathBuf::from(get_icgem_cache_dir().unwrap());
        let cache_dir = cache_root.join("models").join("earth");
        std::fs::create_dir_all(&cache_dir).unwrap();
        let cache_file = cache_dir.join(cache_filename_for_entry(&target_entry));
        std::fs::write(&cache_file, &gfc).unwrap();

        let stale_seconds = 40 * 24 * 60 * 60;
        let index_path = index_path_for(&ICGEMBody::Earth).unwrap();
        write_index_file(
            &index_path,
            &IndexFile {
                fetched_at: now_unix_seconds().saturating_sub(stale_seconds),
                entries: vec![target_entry],
            },
        )
        .unwrap();

        let server = MockServer::start();
        let list_mock = server.mock(|when, then| {
            when.method(GET).path_includes("/tom_longtime");
            then.status(200).body("");
        });
        let download_mock = server.mock(|when, then| {
            when.method(GET).path_includes("/getmodel/gfc/");
            then.status(200).body("");
        });

        let _mode = NetworkModeGuard::set(Some("offline-strict"));
        let path =
            download_icgem_model_with_url(&ICGEMBody::Earth, "JGM3", None, &server.base_url())
                .unwrap();
        assert_eq!(path, cache_file);
        list_mock.assert_calls(0);
        download_mock.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_download_offline_strict_stale_index_missing_model_errors() {
        use httpmock::prelude::*;

        let _cache = CacheRedirect::new();

        let target_entry = entry(ICGEMBody::Earth, "JGM3", 70);
        let stale_seconds = 40 * 24 * 60 * 60;
        let index_path = index_path_for(&ICGEMBody::Earth).unwrap();
        write_index_file(
            &index_path,
            &IndexFile {
                fetched_at: now_unix_seconds().saturating_sub(stale_seconds),
                entries: vec![target_entry],
            },
        )
        .unwrap();

        let server = MockServer::start();
        let list_mock = server.mock(|when, then| {
            when.method(GET).path_includes("/tom_longtime");
            then.status(200).body("");
        });
        let download_mock = server.mock(|when, then| {
            when.method(GET).path_includes("/getmodel/gfc/");
            then.status(200).body("");
        });

        let _mode = NetworkModeGuard::set(Some("offline-strict"));
        let err =
            download_icgem_model_with_url(&ICGEMBody::Earth, "JGM3", None, &server.base_url())
                .unwrap_err()
                .to_string();
        assert!(err.contains("is older than its cache limit"), "{err}");
        list_mock.assert_calls(0);
        download_mock.assert_calls(0);
    }

    #[test]
    #[serial]
    fn test_download_online_refreshes_stale_index_even_with_cached_model() {
        use httpmock::prelude::*;

        let _cache = CacheRedirect::new();

        let html = std::fs::read_to_string("test_assets/icgem/tom_longtime_sample.html").unwrap();
        let gfc = std::fs::read_to_string("data/gravity_models/JGM3.gfc").unwrap();
        let parsed_entries = crate::datasets::icgem::parser::parse_earth_catalog(&html).unwrap();
        let target_entry = resolve_icgem_model(&ICGEMBody::Earth, "JGM3", &parsed_entries)
            .expect("fixture has a JGM3 entry")
            .clone();

        let cache_root = PathBuf::from(get_icgem_cache_dir().unwrap());
        let cache_dir = cache_root.join("models").join("earth");
        std::fs::create_dir_all(&cache_dir).unwrap();
        let cache_file = cache_dir.join(cache_filename_for_entry(&target_entry));
        std::fs::write(&cache_file, &gfc).unwrap();

        let stale_seconds = 40 * 24 * 60 * 60;
        let index_path = index_path_for(&ICGEMBody::Earth).unwrap();
        write_index_file(
            &index_path,
            &IndexFile {
                fetched_at: now_unix_seconds().saturating_sub(stale_seconds),
                entries: vec![target_entry.clone()],
            },
        )
        .unwrap();

        let server = MockServer::start();
        let list_mock = server.mock(|when, then| {
            when.method(GET).path_includes("/tom_longtime");
            then.status(200).body(&html);
        });
        let download_mock = server.mock(|when, then| {
            when.method(GET).path_includes("/getmodel/gfc/");
            then.status(200).body(&gfc);
        });

        let _mode = NetworkModeGuard::set(Some("online"));
        let path = download_icgem_model_with_url(
            &ICGEMBody::Earth,
            &target_entry.name,
            None,
            &server.base_url(),
        )
        .unwrap();
        assert_eq!(path, cache_file);
        list_mock.assert_calls(1);
        download_mock.assert_calls(0);
    }

    // TODO: This test is super flakey because it depends on the live ICGEM service
    // #[test]
    // #[cfg_attr(not(feature = "integration"), ignore)]
    // #[serial_test::serial]
    // fn test_download_live_jgm3_network() {
    //     // Smoke test against real ICGEM. Skipped unless `--features integration`.
    //     let dir = tempfile::tempdir().unwrap();
    //     unsafe { std::env::set_var("BRAHE_CACHE", dir.path()); }

    //     let path = download_icgem_model(ICGEMBody::Earth, "JGM3", None);
    //     // JGM3 is small and stable; if ICGEM is reachable, this must succeed.
    //     assert!(path.is_ok(), "live download failed: {:?}", path.err());

    //     unsafe { std::env::remove_var("BRAHE_CACHE"); }
    // }

    #[test]
    #[parallel]
    fn test_extract_icgem_hash_well_formed() {
        let h = extract_icgem_hash("/getmodel/gfc/abc123def456/EGM2008.gfc");
        assert_eq!(h, Some("abc123def456"));
    }

    #[test]
    #[parallel]
    fn test_extract_icgem_hash_malformed_returns_none() {
        assert_eq!(extract_icgem_hash(""), None);
        assert_eq!(extract_icgem_hash("/wrong/prefix/abc/x.gfc"), None);
        assert_eq!(extract_icgem_hash("/getmodel/gfc//x.gfc"), None);
    }

    #[test]
    #[parallel]
    fn test_cache_filename_includes_hash_so_republished_models_get_new_path() {
        // Two index entries for the same body/name/degree but with different
        // ICGEM download hashes (e.g. the model was republished) must produce
        // distinct cache filenames so the new file is fetched on the next
        // download rather than being shadowed by the stale cached copy.
        let old = IndexEntry {
            body: ICGEMBody::Earth,
            name: "EGM2008".into(),
            year: Some(2008),
            degree: 2190,
            download_path: "/getmodel/gfc/old_hash_aaaaaaaaaaa/EGM2008.gfc".into(),
        };
        let new = IndexEntry {
            body: ICGEMBody::Earth,
            name: "EGM2008".into(),
            year: Some(2008),
            degree: 2190,
            download_path: "/getmodel/gfc/new_hash_bbbbbbbbbbb/EGM2008.gfc".into(),
        };
        let old_name = cache_filename_for_entry(&old);
        let new_name = cache_filename_for_entry(&new);
        assert_ne!(
            old_name, new_name,
            "republished model under a new hash must not collide with the old cache file"
        );
        assert!(old_name.starts_with("EGM2008-2190-"));
        assert!(new_name.starts_with("EGM2008-2190-"));
        assert!(old_name.ends_with(".gfc"));
    }

    #[test]
    #[parallel]
    fn test_cache_filename_falls_back_when_hash_missing() {
        // Defensive: if download_path doesn't match the /getmodel/gfc/<hash>/...
        // pattern (shouldn't happen in practice), we still produce a stable
        // filename rather than panicking.
        let entry = IndexEntry {
            body: ICGEMBody::Earth,
            name: "X".into(),
            year: None,
            degree: 70,
            download_path: "unexpected".into(),
        };
        assert_eq!(cache_filename_for_entry(&entry), "X-70-nohash.gfc");
    }

    #[test]
    #[parallel]
    fn test_model_cache_path_dispatches_by_body() {
        let cache_root = PathBuf::from("/cache");
        for (body, subdir) in [
            (ICGEMBody::Earth, "earth"),
            (ICGEMBody::Moon, "moon"),
            (ICGEMBody::Mars, "mars"),
            (ICGEMBody::Venus, "venus"),
            (ICGEMBody::Ceres, "ceres"),
        ] {
            let e = entry(body.clone(), "MODEL", 10);
            let path = model_cache_path(&body, &e, &cache_root);
            assert!(
                path.starts_with(cache_root.join("models").join(subdir)),
                "body {body:?} -> {path:?}"
            );
        }

        let other = ICGEMBody::Other("pluto".into());
        let e = entry(other.clone(), "MODEL", 10);
        let path = model_cache_path(&other, &e, &cache_root);
        assert!(path.starts_with(cache_root.join("models").join("other/pluto")));
    }

    #[test]
    #[parallel]
    fn test_finalize_model_path_copies_to_output_path() {
        let dir = tempfile::tempdir().unwrap();
        let cache_file = dir.path().join("cached.gfc");
        std::fs::write(&cache_file, b"gfc bytes").unwrap();

        let output_path = dir.path().join("nested").join("out.gfc");
        let result = finalize_model_path(&cache_file, Some(output_path.clone())).unwrap();

        assert_eq!(result, output_path);
        assert_eq!(std::fs::read(&output_path).unwrap(), b"gfc bytes");
    }

    #[test]
    #[parallel]
    fn test_finalize_model_path_returns_cache_file_without_output_path() {
        let dir = tempfile::tempdir().unwrap();
        let cache_file = dir.path().join("cached.gfc");
        std::fs::write(&cache_file, b"gfc bytes").unwrap();

        let result = finalize_model_path(&cache_file, None).unwrap();
        assert_eq!(result, cache_file);
    }
}
