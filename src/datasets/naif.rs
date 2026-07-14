/*!
 * NAIF data source implementation for downloading JPL ephemeris and
 * orientation kernels.
 *
 * NAIF (Navigation and Ancillary Information Facility) is NASA's archive of
 * planetary ephemerides and orientation data, maintained by JPL. This module
 * downloads and caches the kernels enumerated by [`SPICEKernel`] (planetary DE
 * SPK kernels, satellite ephemeris kernels, and binary PCK kernels).
 */

use crate::spice::SPICEKernel;
use crate::utils::BraheError;
use crate::utils::atomic_write;
use crate::utils::cache::get_naif_cache_dir;
use std::fs;
use std::path::PathBuf;

/// Download a kernel's bytes from an explicit base URL.
///
/// Internal test seam: production code uses [`fetch_kernel`], which derives
/// the full URL from the kernel via [`SPICEKernel::url`]. This override lets
/// the mock-server tests point downloads at a local server while exercising
/// the same read/error handling.
///
/// # Arguments
/// * `kernel` - Kernel to download (its [`SPICEKernel::filename`] is appended
///   to `base_url`)
/// * `base_url` - Base URL to fetch from
///
/// # Returns
/// * `Result<Vec<u8>, BraheError>` - Binary kernel data
#[cfg(test)]
fn fetch_kernel_with_url(kernel: SPICEKernel, base_url: &str) -> Result<Vec<u8>, BraheError> {
    let url = format!("{}{}", base_url, kernel.filename());
    fetch_kernel_from_url(&url, kernel.name())
}

/// Download a kernel's bytes from NAIF using the kernel's own archive URL.
///
/// # Arguments
/// * `kernel` - Kernel to download
///
/// # Returns
/// * `Result<Vec<u8>, BraheError>` - Binary kernel data
fn fetch_kernel(kernel: SPICEKernel) -> Result<Vec<u8>, BraheError> {
    fetch_kernel_from_url(&kernel.url(), kernel.name())
}

/// Download the bytes at `url`, reading the whole body without the default
/// size limit (kernels can be large; e.g. de440 ~120MB).
///
/// # Arguments
/// * `url` - Full URL to fetch
/// * `label` - Short kernel name used in error messages
///
/// # Returns
/// * `Result<Vec<u8>, BraheError>` - Binary kernel data
fn fetch_kernel_from_url(url: &str, label: &str) -> Result<Vec<u8>, BraheError> {
    // Retries transient network/server failures with exponential backoff so a
    // single dropped connection to NAIF (e.g. "Connection refused") doesn't fail
    // an otherwise-recoverable download.
    let buffer = crate::utils::download::download_bytes(url).map_err(|e| {
        BraheError::Error(format!(
            "Failed to download kernel {} from NAIF: {}",
            label, e
        ))
    })?;

    if buffer.is_empty() {
        return Err(BraheError::Error(format!(
            "No data returned from NAIF for kernel '{}'",
            label
        )));
    }

    Ok(buffer)
}

/// Download a NAIF kernel with caching support.
///
/// Downloads the specified kernel file and caches it locally. If the kernel
/// is already cached, returns the cached path without re-downloading.
/// Optionally copies the kernel to a user-specified location.
///
/// # Arguments
/// * `kernel` - Which kernel to download (see [`SPICEKernel`])
/// * `output_path` - Optional path to copy the kernel to after download/cache retrieval
///
/// # Returns
/// * `Result<PathBuf, BraheError>` - Path to the kernel file (cache location or `output_path` if specified)
///
/// # Examples
/// ```no_run
/// use brahe::datasets::naif::download_spice_kernel;
/// use brahe::spice::SPICEKernel;
/// use std::path::PathBuf;
///
/// // Download and cache the de440s kernel
/// let kernel_path = download_spice_kernel(SPICEKernel::DE440s, None).unwrap();
/// println!("Kernel cached at: {}", kernel_path.display());
///
/// // Download and copy to a specific location
/// let output = PathBuf::from("/path/to/my_kernel.bsp");
/// let kernel_path = download_spice_kernel(SPICEKernel::DE440s, Some(output.clone())).unwrap();
/// assert_eq!(kernel_path, output);
/// ```
pub fn download_spice_kernel(
    kernel: SPICEKernel,
    output_path: Option<PathBuf>,
) -> Result<PathBuf, BraheError> {
    // Determine cache filepath
    let cache_dir = get_naif_cache_dir()?;
    let cache_path = PathBuf::from(&cache_dir).join(kernel.filename());

    // Check if kernel is already cached
    if !cache_path.exists() {
        // Download kernel
        let data = fetch_kernel(kernel)?;

        // Cache it for future use
        atomic_write(&cache_path, &data).map_err(|e| {
            BraheError::Error(format!(
                "Failed to cache kernel {} to {}: {}",
                kernel.name(),
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
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial]
    fn test_download_de_network() {
        setup_test_kernel();

        // Test downloading de440s kernel (smaller file)
        let result = download_spice_kernel(SPICEKernel::DE440s, None);
        assert!(result.is_ok());

        let kernel_path = result.unwrap();
        assert!(kernel_path.exists());
        assert!(kernel_path.to_string_lossy().contains("de440s.bsp"));

        // Verify file is not empty
        let metadata = fs::metadata(&kernel_path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial]
    fn test_download_with_output_path_network() {
        setup_test_kernel();

        // Create temporary output path
        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_de_output.bsp");

        // Clean up any existing file
        let _ = fs::remove_file(&output_path);

        // Download kernel to specific location
        let result = download_spice_kernel(SPICEKernel::DE440s, Some(output_path.clone()));
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
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial]
    fn test_caching_behavior_network() {
        setup_test_kernel();

        // First download
        let result1 = download_spice_kernel(SPICEKernel::DE440s, None);
        assert!(result1.is_ok());
        let kernel_path = result1.unwrap();

        // Get initial file metadata
        let metadata1 = fs::metadata(&kernel_path).unwrap();
        let modified1 = metadata1.modified().unwrap();

        // Small delay to ensure time difference if file is rewritten
        std::thread::sleep(std::time::Duration::from_millis(100));

        // Second download - should use cache
        let result2 = download_spice_kernel(SPICEKernel::DE440s, None);
        assert!(result2.is_ok());

        //
        assert_eq!(kernel_path, result2.unwrap());

        // Verify file wasn't re-downloaded (modification time unchanged)
        let metadata2 = fs::metadata(&kernel_path).unwrap();
        let modified2 = metadata2.modified().unwrap();
        assert_eq!(modified1, modified2);
    }

    // ========== HTTP Error Tests ==========

    #[test]
    fn test_fetch_kernel_http_404() {
        // Setup mock server that returns 404 Not Found
        let server = MockServer::start();
        let _mock = server.mock(|when, then| {
            when.method(GET).path("/de440s.bsp");
            then.status(404);
        });

        // Attempt to fetch kernel from mock server
        let result = fetch_kernel_with_url(SPICEKernel::DE440s, &server.url("/"));

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
    fn test_fetch_kernel_http_500() {
        // Setup mock server that returns 500 Server Error
        let server = MockServer::start();
        let _mock = server.mock(|when, then| {
            when.method(GET).path("/de440s.bsp");
            then.status(500);
        });

        // Attempt to fetch kernel from mock server
        let result = fetch_kernel_with_url(SPICEKernel::DE440s, &server.url("/"));

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
    fn test_fetch_kernel_empty_response() {
        // Setup mock server that returns 200 OK with empty body
        let server = MockServer::start();
        let _mock = server.mock(|when, then| {
            when.method(GET).path("/de440s.bsp");
            then.status(200).body("");
        });

        // Attempt to fetch kernel from mock server
        let result = fetch_kernel_with_url(SPICEKernel::DE440s, &server.url("/"));

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

    #[test]
    fn test_fetch_kernel_with_url_uses_filename_override() {
        // The Ura184 kernel's file name differs from its short name; the
        // with-URL seam must append the *filename*, not the name.
        let server = MockServer::start();
        let _mock = server.mock(|when, then| {
            when.method(GET).path("/ura184_part-3.bsp");
            then.status(200).body("data");
        });

        let result = fetch_kernel_with_url(SPICEKernel::Ura184, &server.url("/"));
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), b"data");
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
        let result = download_spice_kernel(SPICEKernel::DE440s, Some(output_path.clone()));

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
        let result = download_spice_kernel(SPICEKernel::DE440s, Some(output_path));

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
        let result = download_spice_kernel(SPICEKernel::DE440s, Some(output_path));

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
    #[cfg_attr(not(feature = "integration"), ignore)]
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
        let result = download_spice_kernel(SPICEKernel::DE440s, Some(nested_path.clone()));
        assert!(result.is_ok());
        assert!(nested_path.exists());
        assert!(nested_path.parent().unwrap().exists());

        // Verify file is valid
        let metadata = fs::metadata(&nested_path).unwrap();
        assert!(metadata.len() > 0);
    }

    // ========== PCK Kernel Tests ==========

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial]
    fn test_download_pck_network() {
        let result = download_spice_kernel(SPICEKernel::MoonPaDe440, None);
        assert!(result.is_ok());
        let path = result.unwrap();
        assert!(path.exists());
        assert!(path.to_string_lossy().contains("moon_pa_de440_200625.bpc"));
    }
}
