/*!
 * NAIF data source implementation for downloading JPL ephemeris kernels.
 *
 * NAIF (Navigation and Ancillary Information Facility) is NASA's archive of
 * planetary ephemerides, maintained by JPL. This module provides functions
 * to download DE (Development Ephemeris) kernels in SPK (SPICE Kernel) format.
 */

use crate::utils::BraheError;
use crate::utils::cache::get_naif_cache_dir;
use std::fs;
use std::io::Read;
use std::path::PathBuf;

/// Base URL for NAIF generic kernels repository
const NAIF_BASE_URL: &str = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/";

/// Supported DE kernel names
const SUPPORTED_KERNELS: &[&str] = &[
    "de430", "de432s", "de435", "de438", "de440", "de440s", "de442", "de442s",
];

/// Validate that a kernel name is supported
///
/// # Arguments
/// * `name` - Kernel name to validate
///
/// # Returns
/// * `Result<(), BraheError>` - Ok if valid, error otherwise
fn validate_kernel_name(name: &str) -> Result<(), BraheError> {
    if SUPPORTED_KERNELS.contains(&name) {
        Ok(())
    } else {
        Err(BraheError::Error(format!(
            "Unsupported kernel name '{}'. Supported kernels: {}",
            name,
            SUPPORTED_KERNELS.join(", ")
        )))
    }
}

/// Download a DE kernel file from NAIF with configurable base URL
///
/// This is an internal function for testing. Use `fetch_de_kernel()` or `download_de_kernel()` for the public API.
///
/// # Arguments
/// * `name` - Kernel name (e.g., "de440", "de440s")
/// * `base_url` - Base URL for the NAIF repository
///
/// # Returns
/// * `Result<Vec<u8>, BraheError>` - Binary kernel data
fn fetch_de_kernel_with_url(name: &str, base_url: &str) -> Result<Vec<u8>, BraheError> {
    let filename = format!("{}.bsp", name);
    let url = format!("{}{}", base_url, filename);

    let response = ureq::get(&url).call().map_err(|e| {
        BraheError::Error(format!(
            "Failed to download kernel {} from NAIF: {}",
            name, e
        ))
    })?;

    // Read response body manually to avoid default size limits
    // DE kernels can be large (de440s ~17MB, de440 ~114MB)
    let mut buffer = Vec::new();
    let mut reader = response.into_body().into_reader();

    reader.read_to_end(&mut buffer).map_err(|e| {
        BraheError::Error(format!(
            "Failed to read kernel {} from NAIF response: {}",
            name, e
        ))
    })?;

    if buffer.is_empty() {
        return Err(BraheError::Error(format!(
            "No data returned from NAIF for kernel '{}'",
            name
        )));
    }

    Ok(buffer)
}

/// Download a DE kernel file from NAIF
///
/// This is an internal function. Use `download_de_kernel()` for the public API.
///
/// # Arguments
/// * `name` - Kernel name (e.g., "de440", "de440s")
///
/// # Returns
/// * `Result<Vec<u8>, BraheError>` - Binary kernel data
fn fetch_de_kernel(name: &str) -> Result<Vec<u8>, BraheError> {
    fetch_de_kernel_with_url(name, NAIF_BASE_URL)
}

/// Download a DE kernel from NAIF with caching support
///
/// Downloads the specified DE kernel file and caches it locally. If the kernel
/// is already cached, returns the cached path without re-downloading.
/// Optionally copies the kernel to a user-specified location.
///
/// # Arguments
/// * `name` - Kernel name (e.g., "de440", "de440s", "de430", "de432s", "de435", "de438", "de442", "de442s")
/// * `output_path` - Optional path to copy the kernel to after download/cache retrieval
///
/// # Returns
/// * `Result<PathBuf, BraheError>` - Path to the kernel file (cache location or output_path if specified)
///
/// # Examples
/// ```no_run
/// use brahe::datasets::naif::download_de_kernel;
/// use std::path::PathBuf;
///
/// // Download and cache de440s kernel
/// let kernel_path = download_de_kernel("de440s", None).unwrap();
/// println!("Kernel cached at: {}", kernel_path.display());
///
/// // Download and copy to specific location
/// let output = PathBuf::from("/path/to/my_kernel.bsp");
/// let kernel_path = download_de_kernel("de440s", Some(output.clone())).unwrap();
/// assert_eq!(kernel_path, output);
/// ```
pub fn download_de_kernel(name: &str, output_path: Option<PathBuf>) -> Result<PathBuf, BraheError> {
    // Validate kernel name
    validate_kernel_name(name)?;

    // Determine cache filepath
    let cache_dir = get_naif_cache_dir()?;
    let cache_path = PathBuf::from(&cache_dir).join(format!("{}.bsp", name));

    // Check if kernel is already cached
    if !cache_path.exists() {
        // Download kernel
        let data = fetch_de_kernel(name)?;

        // Cache it for future use
        fs::write(&cache_path, &data).map_err(|e| {
            BraheError::Error(format!(
                "Failed to cache kernel {} to {}: {}",
                name,
                cache_path.display(),
                e
            ))
        })?;
    }

    // If output path is specified, copy the cached file there
    if let Some(output) = output_path {
        // Create parent directory if needed
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).map_err(|e| {
                BraheError::Error(format!(
                    "Failed to create output directory {}: {}",
                    parent.display(),
                    e
                ))
            })?;
        }

        // Copy file to output location
        fs::copy(&cache_path, &output).map_err(|e| {
            BraheError::Error(format!(
                "Failed to copy kernel from {} to {}: {}",
                cache_path.display(),
                output.display(),
                e
            ))
        })?;

        Ok(output)
    } else {
        Ok(cache_path)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use serial_test::serial;
    use tempfile::tempdir;

    /// Setup helper: Copy de440s.bsp from test_assets to cache for CI tests.
    ///
    /// This avoids hitting NAIF servers during CI runs by pre-populating the cache
    /// with the test asset from test_assets/de440s.bsp.
    fn setup_test_kernel() {
        let test_asset_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_assets")
            .join("de440s.bsp");

        // Only copy if test asset exists (it might not in local dev)
        if !test_asset_path.exists() {
            return;
        }

        let cache_dir = get_naif_cache_dir().expect("Failed to get NAIF cache dir");
        let cache_path = PathBuf::from(&cache_dir).join("de440s.bsp");

        // Copy test asset to cache if not already there
        if !cache_path.exists() {
            fs::copy(&test_asset_path, &cache_path).expect("Failed to copy test asset to cache");
        }
    }

    #[test]
    fn test_validate_kernel_name() {
        // Valid kernels
        assert!(validate_kernel_name("de430").is_ok());
        assert!(validate_kernel_name("de432s").is_ok());
        assert!(validate_kernel_name("de435").is_ok());
        assert!(validate_kernel_name("de438").is_ok());
        assert!(validate_kernel_name("de440").is_ok());
        assert!(validate_kernel_name("de440s").is_ok());
        assert!(validate_kernel_name("de442").is_ok());
        assert!(validate_kernel_name("de442s").is_ok());

        // Invalid kernels
        assert!(validate_kernel_name("de999").is_err());
        assert!(validate_kernel_name("invalid").is_err());
        assert!(validate_kernel_name("").is_err());
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    #[serial]
    fn test_download_de440s_network() {
        setup_test_kernel();

        // Test downloading de440s kernel (smaller file)
        let result = download_de_kernel("de440s", None);
        assert!(result.is_ok());

        let kernel_path = result.unwrap();
        assert!(kernel_path.exists());
        assert!(kernel_path.to_string_lossy().contains("de440s.bsp"));

        // Verify file is not empty
        let metadata = fs::metadata(&kernel_path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    #[serial]
    fn test_download_with_output_path_network() {
        setup_test_kernel();

        // Create temporary output path
        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_de440s_output.bsp");

        // Clean up any existing file
        let _ = fs::remove_file(&output_path);

        // Download kernel to specific location
        let result = download_de_kernel("de440s", Some(output_path.clone()));
        assert!(result.is_ok());

        let returned_path = result.unwrap();
        assert_eq!(returned_path, output_path);
        assert!(output_path.exists());

        // Verify file is not empty
        let metadata = fs::metadata(&output_path).unwrap();
        assert!(metadata.len() > 0);

        // Clean up
        let _ = fs::remove_file(&output_path);
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    #[serial]
    fn test_caching_behavior_network() {
        setup_test_kernel();

        // First download
        let result1 = download_de_kernel("de440s", None);
        assert!(result1.is_ok());
        let kernel_path = result1.unwrap();

        // Get initial file metadata
        let metadata1 = fs::metadata(&kernel_path).unwrap();
        let modified1 = metadata1.modified().unwrap();

        // Small delay to ensure time difference if file is rewritten
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Second download - should use cache
        let result2 = download_de_kernel("de440s", None);
        assert!(result2.is_ok());

        //
        assert_eq!(kernel_path, result2.unwrap());

        // Verify file wasn't re-downloaded (modification time unchanged)
        let metadata2 = fs::metadata(&kernel_path).unwrap();
        let modified2 = metadata2.modified().unwrap();
        assert_eq!(modified1, modified2);
    }

    #[test]
    fn test_unsupported_kernel_name() {
        let result = download_de_kernel("de999", None);
        assert!(result.is_err());

        let err = result.unwrap_err();
        match err {
            BraheError::Error(msg) => {
                assert!(msg.contains("Unsupported kernel name"));
                assert!(msg.contains("de999"));
            }
            _ => panic!("Expected BraheError::Error"),
        }
    }

    #[test]
    fn test_supported_kernels_list() {
        // Verify all supported kernels are present
        assert_eq!(SUPPORTED_KERNELS.len(), 8);
        assert!(SUPPORTED_KERNELS.contains(&"de430"));
        assert!(SUPPORTED_KERNELS.contains(&"de432s"));
        assert!(SUPPORTED_KERNELS.contains(&"de435"));
        assert!(SUPPORTED_KERNELS.contains(&"de438"));
        assert!(SUPPORTED_KERNELS.contains(&"de440"));
        assert!(SUPPORTED_KERNELS.contains(&"de440s"));
        assert!(SUPPORTED_KERNELS.contains(&"de442"));
        assert!(SUPPORTED_KERNELS.contains(&"de442s"));
    }

    // ========== HTTP Error Tests ==========

    #[test]
    fn test_fetch_de_kernel_http_404() {
        // Setup mock server that returns 404 Not Found
        let server = MockServer::start();
        let _mock = server.mock(|when, then| {
            when.method(GET).path("/de440s.bsp");
            then.status(404);
        });

        // Attempt to fetch kernel from mock server
        let result = fetch_de_kernel_with_url("de440s", &server.url("/"));

        // Should fail with appropriate error
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            BraheError::Error(msg) => {
                assert!(msg.contains("Failed to download kernel"));
                assert!(msg.contains("de440s"));
            }
            _ => panic!("Expected BraheError::Error"),
        }
    }

    #[test]
    fn test_fetch_de_kernel_http_500() {
        // Setup mock server that returns 500 Server Error
        let server = MockServer::start();
        let _mock = server.mock(|when, then| {
            when.method(GET).path("/de440s.bsp");
            then.status(500);
        });

        // Attempt to fetch kernel from mock server
        let result = fetch_de_kernel_with_url("de440s", &server.url("/"));

        // Should fail with appropriate error
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            BraheError::Error(msg) => {
                assert!(msg.contains("Failed to download kernel"));
                assert!(msg.contains("de440s"));
            }
            _ => panic!("Expected BraheError::Error"),
        }
    }

    #[test]
    fn test_fetch_de_kernel_empty_response() {
        // Setup mock server that returns 200 OK with empty body
        let server = MockServer::start();
        let _mock = server.mock(|when, then| {
            when.method(GET).path("/de440s.bsp");
            then.status(200).body("");
        });

        // Attempt to fetch kernel from mock server
        let result = fetch_de_kernel_with_url("de440s", &server.url("/"));

        // Should fail with "No data returned" error
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            BraheError::Error(msg) => {
                assert!(msg.contains("No data returned"));
                assert!(msg.contains("de440s"));
            }
            _ => panic!("Expected BraheError::Error"),
        }
    }

    // ========== File I/O Error Tests ==========

    #[test]
    fn test_download_output_is_directory() {
        // Create a temporary directory
        let temp_dir = tempdir().unwrap();
        let output_path = temp_dir.path().join("some_directory");
        fs::create_dir_all(&output_path).unwrap();

        // Setup test kernel in cache first
        setup_test_kernel();

        // Attempt to download with output path pointing to a directory (not a file)
        let result = download_de_kernel("de440s", Some(output_path.clone()));

        // Should fail because output path is a directory
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            BraheError::Error(msg) => {
                assert!(msg.contains("Failed to copy kernel"));
            }
            _ => panic!("Expected BraheError::Error"),
        }
    }

    #[test]
    #[cfg(unix)]
    fn test_download_invalid_cache_permissions() {
        use std::os::unix::fs::PermissionsExt;

        // Create a temporary directory to use as cache
        let temp_dir = tempdir().unwrap();
        let cache_file = temp_dir.path().join("de440s.bsp");

        // Create the cache file and make it read-only
        fs::write(&cache_file, b"test").unwrap();
        let mut perms = fs::metadata(&cache_file).unwrap().permissions();
        perms.set_mode(0o444); // Read-only
        fs::set_permissions(&cache_file, perms).unwrap();

        // Make the parent directory read-only as well to prevent new file creation
        let mut dir_perms = fs::metadata(temp_dir.path()).unwrap().permissions();
        dir_perms.set_mode(0o555); // Read and execute only, no write
        fs::set_permissions(temp_dir.path(), dir_perms).unwrap();

        // Create output path pointing to the read-only directory
        let output_path = temp_dir.path().join("new_file.bsp");

        // Setup test kernel in actual cache
        setup_test_kernel();

        // Attempt to copy to the read-only directory
        let result = download_de_kernel("de440s", Some(output_path));

        // Restore permissions for cleanup
        let mut dir_perms = fs::metadata(temp_dir.path()).unwrap().permissions();
        dir_perms.set_mode(0o755);
        fs::set_permissions(temp_dir.path(), dir_perms).unwrap();

        // Should fail due to permission denied
        assert!(result.is_err());
    }

    #[test]
    fn test_download_output_copy_failure() {
        // Create a temporary file, then try to create a directory with the same name
        let temp_dir = tempdir().unwrap();
        let output_path = temp_dir.path().join("conflict");

        // Create a directory where the output file should be
        fs::create_dir_all(&output_path).unwrap();

        // Setup test kernel in cache
        setup_test_kernel();

        // Attempt to download with output path that exists as a directory
        let result = download_de_kernel("de440s", Some(output_path));

        // Should fail because we can't overwrite a directory with a file
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            BraheError::Error(msg) => {
                assert!(msg.contains("Failed to copy kernel"));
            }
            _ => panic!("Expected BraheError::Error"),
        }
    }

    // ========== Edge Case Tests ==========

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    #[serial]
    fn test_output_path_creates_directories() {
        // Setup test kernel in cache
        setup_test_kernel();

        // Create temporary directory with nested path
        let temp_dir = tempdir().unwrap();
        let nested_path = temp_dir
            .path()
            .join("level1")
            .join("level2")
            .join("level3")
            .join("de440s.bsp");

        // Parent directories don't exist yet
        assert!(!nested_path.parent().unwrap().exists());

        // Download should create all parent directories
        let result = download_de_kernel("de440s", Some(nested_path.clone()));
        assert!(result.is_ok());
        assert!(nested_path.exists());
        assert!(nested_path.parent().unwrap().exists());

        // Verify file is valid
        let metadata = fs::metadata(&nested_path).unwrap();
        assert!(metadata.len() > 0);
    }
}
