/*!
 * Module that provides download functionality for different EOP file sources.
 */

use std::path::Path;

use crate::utils::{BraheError, atomic_write};

const STANDARD_FILE_SOURCE: &str =
    "https://datacenter.iers.org/data/latestVersion/finals.all.iau2000.txt";
const C04_FILE_SOURCE: &str =
    "https://datacenter.iers.org/data/latestVersion/EOP_20_C04_one_file_1962-now.txt";

/// Download latest C04 Earth orientation parameter file. Will attempt to download the latest
/// parameter file to the specified location. Creating any missing directories as required.
///
/// The download source is the [IERS Earth Orientation Data Products](https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html)
///
/// # Arguments
/// - `filepath`: Path of desired output file
pub fn download_c04_eop_file(filepath: &str) -> Result<(), BraheError> {
    let filepath = Path::new(filepath);

    let body = ureq::get(C04_FILE_SOURCE)
        .call()
        .map_err(|e| BraheError::IoError(format!("C04 EOP download request failed: {}", e)))?
        .body_mut()
        .read_to_string()
        .map_err(|e| {
            BraheError::IoError(format!("Failed to read C04 EOP download response: {}", e))
        })?;

    atomic_write(filepath, body.as_bytes())?;

    Ok(())
}

/// Download latest standard Earth orientation parameter file. Will attempt to download the latest
/// parameter file to the specified location. Creating any missing directories as required.
///
/// The download source is the [IERS Earth Orientation Data Products](https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html)
///
/// # Arguments
/// - `filepath`: Path of desired output file
pub fn download_standard_eop_file(filepath: &str) -> Result<(), BraheError> {
    let filepath = Path::new(filepath);

    let body = ureq::get(STANDARD_FILE_SOURCE)
        .call()
        .map_err(|e| BraheError::IoError(format!("Standard EOP download request failed: {}", e)))?
        .body_mut()
        .read_to_string()
        .map_err(|e| {
            BraheError::IoError(format!(
                "Failed to read standard EOP download response: {}",
                e
            ))
        })?;

    atomic_write(filepath, body.as_bytes())?;

    Ok(())
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::env;
    use std::fs;

    use httpmock::prelude::*;
    use tempfile::tempdir;

    use super::*;

    fn get_standard_file_contents() -> String {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = Path::new(&manifest_dir)
            .join("test_assets")
            .join("finals.all.iau2000.txt");

        fs::read_to_string(filepath).expect("Failed to read file")
    }

    fn get_c04_file_contents() -> String {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = Path::new(&manifest_dir)
            .join("test_assets")
            .join("EOP_20_C04_one_file_1962-now.txt");

        fs::read_to_string(filepath).expect("Failed to read file")
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)] // Only run in CI to avoid network calls
    fn test_download_standard_eop_file() {
        // Mock return of contents of HTTP call
        let server = MockServer::start();
        let _download_mock = server.mock(|when, then| {
            when.method(GET).path(STANDARD_FILE_SOURCE);
            then.status(200)
                .header("content-type", "text/html; charset=UTF-8")
                .body(get_standard_file_contents());
        });

        // Create Temporary File
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("temp-standard-file.txt");
        let filepath_str = filepath.to_str().unwrap();

        download_standard_eop_file(filepath_str).expect("Failed to download file");
        assert!(Path::new(filepath_str).exists());
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)] // Only run in CI to avoid network calls
    fn test_download_c04_eop_file() {
        // Mock return of contents of HTTP call
        let server = MockServer::start();
        let _download_mock = server.mock(|when, then| {
            when.method(GET).path(C04_FILE_SOURCE);
            then.status(200)
                .header("content-type", "text/html; charset=UTF-8")
                .body(get_c04_file_contents());
        });

        // Create Temporary File
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("temp-c04-file.txt");
        let filepath_str = filepath.to_str().unwrap();

        download_c04_eop_file(filepath_str).expect("Failed to download file");
        assert!(Path::new(filepath_str).exists());
    }
}
