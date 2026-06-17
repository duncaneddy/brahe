//! ICGEM model name → entry resolution and download.

use crate::datasets::icgem::body::ICGEMBody;
use crate::datasets::icgem::index::IndexEntry;
use crate::utils::BraheError;

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
use std::path::PathBuf;

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

/// Download (or load from cache) a `.gfc` file for the named ICGEM model.
///
/// If `output_path` is `Some`, also copies the cached file there and returns
/// that path. Otherwise returns the cache path.
pub fn download_icgem_model(
    body: ICGEMBody,
    name: &str,
    output_path: Option<PathBuf>,
) -> Result<PathBuf, BraheError> {
    download_icgem_model_with_url(&body, name, output_path, ICGEM_BASE_URL)
}

pub(crate) fn download_icgem_model_with_url(
    body: &ICGEMBody,
    name: &str,
    output_path: Option<PathBuf>,
    base_url: &str,
) -> Result<PathBuf, BraheError> {
    let entries = list_icgem_models_with_url(body, base_url)?;
    let entry = resolve_icgem_model(body, name, &entries)?.clone();

    let cache_root = PathBuf::from(get_icgem_cache_dir()?);
    let body_subdir = match body {
        ICGEMBody::Earth => "earth".to_string(),
        ICGEMBody::Moon => "moon".to_string(),
        ICGEMBody::Mars => "mars".to_string(),
        ICGEMBody::Venus => "venus".to_string(),
        ICGEMBody::Ceres => "ceres".to_string(),
        ICGEMBody::Other(n) => format!("other/{}", n),
    };
    let cache_dir = cache_root.join("models").join(&body_subdir);
    let cache_file = cache_dir.join(cache_filename_for_entry(&entry));

    if !cache_file.exists() {
        let url = format!("{}{}", base_url, entry.download_path);
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

    if let Some(out) = output_path {
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                BraheError::Error(format!(
                    "Failed to create output directory {}: {}",
                    parent.display(),
                    e
                ))
            })?;
        }
        std::fs::copy(&cache_file, &out).map_err(|e| {
            BraheError::Error(format!(
                "Failed to copy ICGEM model from {} to {}: {}",
                cache_file.display(),
                out.display(),
                e
            ))
        })?;
        Ok(out)
    } else {
        Ok(cache_file)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

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
    fn test_resolve_exact_single_variant() {
        let entries = earth_fixture();
        let got = resolve_icgem_model(&ICGEMBody::Earth, "JGM3", &entries).unwrap();
        assert_eq!(got.name, "JGM3");
        assert_eq!(got.degree, 70);
    }

    #[test]
    fn test_resolve_largest_degree_when_ambiguous() {
        let entries = earth_fixture();
        let got =
            resolve_icgem_model(&ICGEMBody::Earth, "WHU-CASM-UGM2025_2159", &entries).unwrap();
        assert_eq!(got.degree, 11000);
    }

    #[test]
    fn test_resolve_with_explicit_degree_suffix() {
        let entries = earth_fixture();
        let got =
            resolve_icgem_model(&ICGEMBody::Earth, "WHU-CASM-UGM2025_2159-2190", &entries).unwrap();
        assert_eq!(got.degree, 2190);
    }

    #[test]
    fn test_resolve_missing_degree_errors_with_available_list() {
        let entries = earth_fixture();
        let err = resolve_icgem_model(&ICGEMBody::Earth, "WHU-CASM-UGM2025_2159-99", &entries)
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("no variant at degree 99"));
        assert!(msg.contains("760") && msg.contains("2190") && msg.contains("11000"));
    }

    #[test]
    fn test_resolve_typo_returns_nearest_names() {
        let entries = earth_fixture();
        let err = resolve_icgem_model(&ICGEMBody::Earth, "EGM200", &entries).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("EGM2008"));
    }

    #[test]
    fn test_resolve_other_body_does_not_leak_earth_results() {
        let entries = earth_fixture();
        let err = resolve_icgem_model(&ICGEMBody::Mars, "EGM2008", &entries).unwrap_err();
        assert!(err.to_string().contains("not found for body 'Mars'"));
    }

    #[test]
    fn test_resolve_exact_match_takes_precedence_over_suffix_split() {
        let mut entries = earth_fixture();
        entries.push(entry(ICGEMBody::Earth, "MODEL-X-2020", 200));
        let got = resolve_icgem_model(&ICGEMBody::Earth, "MODEL-X-2020", &entries).unwrap();
        assert_eq!(got.name, "MODEL-X-2020");
        assert_eq!(got.degree, 200);
    }

    #[test]
    #[serial_test::serial]
    fn test_download_end_to_end_with_mock_server() {
        use httpmock::prelude::*;

        let dir = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("BRAHE_CACHE", dir.path());
        }

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

        unsafe {
            std::env::remove_var("BRAHE_CACHE");
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_download_uses_cache_on_second_call() {
        use httpmock::prelude::*;

        let dir = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("BRAHE_CACHE", dir.path());
        }

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

        unsafe {
            std::env::remove_var("BRAHE_CACHE");
        }
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
    fn test_extract_icgem_hash_well_formed() {
        let h = extract_icgem_hash("/getmodel/gfc/abc123def456/EGM2008.gfc");
        assert_eq!(h, Some("abc123def456"));
    }

    #[test]
    fn test_extract_icgem_hash_malformed_returns_none() {
        assert_eq!(extract_icgem_hash(""), None);
        assert_eq!(extract_icgem_hash("/wrong/prefix/abc/x.gfc"), None);
        assert_eq!(extract_icgem_hash("/getmodel/gfc//x.gfc"), None);
    }

    #[test]
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
}
