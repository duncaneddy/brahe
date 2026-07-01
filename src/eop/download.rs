/*!
 * Module that provides download functionality for different EOP file sources.
 */

use std::path::Path;

use crate::utils::BraheError;
use crate::utils::download::download_to_file;

// Sourced from USNO and Paris Observatory (IERS) mirrors; the primary IERS
// datacenter (datacenter.iers.org) is frequently unavailable.
const STANDARD_FILE_SOURCE: &str = "https://maia.usno.navy.mil/ser7/finals2000A.all";
const C04_FILE_SOURCE: &str = "https://hpiers.obspm.fr/iers/eop/eopc04/eopc04.1962-now";

/// Download latest C04 Earth orientation parameter file. Will attempt to download the latest
/// parameter file to the specified location. Creating any missing directories as required.
///
/// The download source is the [Paris Observatory IERS EOP C04 series](https://hpiers.obspm.fr/iers/eop/eopc04/eopc04.1962-now).
///
/// # Arguments
/// - `filepath`: Path of desired output file
pub fn download_c04_eop_file(filepath: &str) -> Result<(), BraheError> {
    download_to_file(C04_FILE_SOURCE, "C04 EOP", Path::new(filepath))
}

/// Download latest standard Earth orientation parameter file. Will attempt to download the latest
/// parameter file to the specified location. Creating any missing directories as required.
///
/// The download source is the [USNO finals2000A.all Bulletin A product](https://maia.usno.navy.mil/ser7/finals2000A.all).
///
/// # Arguments
/// - `filepath`: Path of desired output file
pub fn download_standard_eop_file(filepath: &str) -> Result<(), BraheError> {
    download_to_file(STANDARD_FILE_SOURCE, "Standard EOP", Path::new(filepath))
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
            .join("EOP_C04_one_file_1962-now.txt");

        fs::read_to_string(filepath).expect("Failed to read file")
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)] // Only run in CI to avoid network calls
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
    #[cfg_attr(not(feature = "integration"), ignore)] // Only run in CI to avoid network calls
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
