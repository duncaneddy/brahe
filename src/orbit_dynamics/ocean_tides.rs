/*!
FES2004 ocean tide geopotential corrections (IERS Conventions 2010, TN36 §6.3).

Data source: FES2004 normalized Stokes-coefficient amplitude file, downloaded
once into `$BRAHE_CACHE/tides/` from
<https://iers-conventions.obspm.fr/content/chapter6/additional_info/tidemodels/fes2004_Cnm-Snm.dat>.
*/

use std::io::Read;
use std::path::PathBuf;

use crate::utils::BraheError;
use crate::utils::cache::get_tides_cache_dir;
use crate::utils::fs::atomic_write;

const FES2004_URL: &str = "https://iers-conventions.obspm.fr/content/chapter6/additional_info/tidemodels/fes2004_Cnm-Snm.dat";
const FES2004_FILENAME: &str = "fes2004_Cnm-Snm.dat";

/// Path to the cached FES2004 ocean tide coefficient file, downloading it
/// (one-time, ~3.7 MB) from IERS if not already cached.
///
/// # Returns
///
/// * `PathBuf` - Location of `fes2004_Cnm-Snm.dat` inside `$BRAHE_CACHE/tides/`.
///
/// # Errors
///
/// Returns `BraheError` if the tides cache directory cannot be created, or if
/// no cached copy exists and the download from IERS fails. The error message
/// names the URL and the target cache path so the file can be fetched
/// manually to proceed offline.
pub fn fes2004_coefficients_path() -> Result<PathBuf, BraheError> {
    let dir = PathBuf::from(get_tides_cache_dir()?);
    let path = dir.join(FES2004_FILENAME);
    if path.exists() {
        return Ok(path);
    }

    let response = ureq::get(FES2004_URL).call().map_err(|e| {
        BraheError::Error(format!(
            "FES2004 ocean tide coefficients are not cached and the download from {FES2004_URL} \
             failed: {e}. Download the file manually to {} to proceed offline.",
            path.display()
        ))
    })?;
    let mut buf = Vec::new();
    response
        .into_body()
        .into_reader()
        .read_to_end(&mut buf)
        .map_err(|e| BraheError::Error(format!("FES2004 download read failed: {e}")))?;
    if buf.is_empty() {
        return Err(BraheError::Error(format!(
            "Empty response downloading FES2004 ocean tide coefficients from {FES2004_URL}"
        )));
    }
    atomic_write(&path, &buf).map_err(|e| {
        BraheError::Error(format!(
            "Failed to write FES2004 cache {}: {}",
            path.display(),
            e
        ))
    })?;
    Ok(path)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn test_fes2004_path_uses_cache_without_network() {
        let _guard = crate::utils::testing::setup_test_fes2004_cache();
        let path = fes2004_coefficients_path().unwrap();
        assert!(path.exists());
        assert!(path.ends_with("tides/fes2004_Cnm-Snm.dat"));
    }

    #[test]
    #[serial_test::serial]
    #[cfg_attr(not(feature = "integration"), ignore)]
    fn test_fes2004_download() {
        // Network-gated: clean cache, real download, file is complete.
        let dir = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("BRAHE_CACHE", dir.path());
        }
        let path = fes2004_coefficients_path().unwrap();
        let len = std::fs::metadata(&path).unwrap().len();
        assert!(len > 3_500_000, "downloaded file too small: {len}");
        unsafe {
            std::env::remove_var("BRAHE_CACHE");
        }
    }
}
