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

// =============================================================================
// HTTP Client Trait for Dependency Injection
// =============================================================================

/// Trait for HTTP client operations, allowing for dependency injection and mocking.
#[cfg_attr(test, mockall::automock)]
pub trait HttpClient: Send + Sync {
    /// Perform an HTTP GET request and return the response body as bytes.
    ///
    /// # Arguments
    /// * `url` - The URL to fetch
    ///
    /// # Returns
    /// * `Result<Vec<u8>, BraheError>` - Response body as bytes
    fn get(&self, url: &str) -> Result<Vec<u8>, BraheError>;
}

/// Default HTTP client implementation using ureq.
pub struct UreqHttpClient;

impl HttpClient for UreqHttpClient {
    fn get(&self, url: &str) -> Result<Vec<u8>, BraheError> {
        let response = ureq::get(url)
            .call()
            .map_err(|e| BraheError::Error(format!("HTTP request failed for {}: {}", url, e)))?;

        // Read response body manually to avoid default size limits
        // DE kernels can be large (de440s ~17MB, de440 ~114MB)
        let mut buffer = Vec::new();
        let mut reader = response.into_body().into_reader();

        reader.read_to_end(&mut buffer).map_err(|e| {
            BraheError::Error(format!("Failed to read response body from {}: {}", url, e))
        })?;

        Ok(buffer)
    }
}

// =============================================================================
// NAIF Downloader with Dependency Injection
// =============================================================================

/// NAIF kernel downloader with injectable HTTP client for testability.
pub struct NaifDownloader<C: HttpClient> {
    client: C,
    base_url: String,
}

impl<C: HttpClient> NaifDownloader<C> {
    /// Create a new NaifDownloader with a custom HTTP client.
    ///
    /// # Arguments
    /// * `client` - HTTP client implementation
    pub fn new(client: C) -> Self {
        Self {
            client,
            base_url: NAIF_BASE_URL.to_string(),
        }
    }

    /// Create a new NaifDownloader with a custom HTTP client and base URL.
    ///
    /// # Arguments
    /// * `client` - HTTP client implementation
    /// * `base_url` - Base URL for NAIF kernels
    #[cfg(test)]
    pub fn with_base_url(client: C, base_url: String) -> Self {
        Self { client, base_url }
    }

    /// Download a DE kernel from NAIF with caching support.
    ///
    /// Downloads the specified DE kernel file and caches it locally. If the kernel
    /// is already cached, returns the cached path without re-downloading.
    /// Optionally copies the kernel to a user-specified location.
    ///
    /// # Arguments
    /// * `name` - Kernel name (e.g., "de440", "de440s")
    /// * `output_path` - Optional path to copy the kernel to
    ///
    /// # Returns
    /// * `Result<PathBuf, BraheError>` - Path to the kernel file
    pub fn download_de_kernel(
        &self,
        name: &str,
        output_path: Option<PathBuf>,
    ) -> Result<PathBuf, BraheError> {
        // Validate kernel name
        validate_kernel_name(name)?;

        // Determine cache filepath
        let cache_dir = get_naif_cache_dir()?;
        let cache_path = PathBuf::from(&cache_dir).join(format!("{}.bsp", name));

        // Check if kernel is already cached
        if !cache_path.exists() {
            // Download kernel
            let data = self.fetch_de_kernel(name)?;

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

    /// Fetch kernel data from NAIF server.
    fn fetch_de_kernel(&self, name: &str) -> Result<Vec<u8>, BraheError> {
        let filename = format!("{}.bsp", name);
        let url = format!("{}{}", self.base_url, filename);

        let buffer = self.client.get(&url).map_err(|e| {
            BraheError::Error(format!(
                "Failed to download kernel {} from NAIF: {}",
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
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Validate that a kernel name is supported.
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

// =============================================================================
// Public API - Convenience Functions
// =============================================================================

/// Download a DE kernel from NAIF with caching support.
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
    let downloader = NaifDownloader::new(UreqHttpClient);
    downloader.download_de_kernel(name, output_path)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use serial_test::serial;
    use tempfile::TempDir;

    // =========================================================================
    // Helper Functions
    // =========================================================================

    /// Setup helper: Copy de440s.bsp from test_assets to cache for CI tests.
    fn setup_test_kernel() {
        let test_asset_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_assets")
            .join("de440s.bsp");

        if !test_asset_path.exists() {
            return;
        }

        let cache_dir = get_naif_cache_dir().expect("Failed to get NAIF cache dir");
        let cache_path = PathBuf::from(&cache_dir).join("de440s.bsp");

        if !cache_path.exists() {
            fs::copy(&test_asset_path, &cache_path).expect("Failed to copy test asset to cache");
        }
    }

    /// Create sample kernel data for mocking
    fn create_mock_kernel_data() -> Vec<u8> {
        // Simple mock data - just needs to be non-empty
        vec![0x44, 0x41, 0x46, 0x2f, 0x53, 0x50, 0x4b, 0x00] // "DAF/SPK\0" header
    }

    // =========================================================================
    // Validation Tests
    // =========================================================================

    #[test]
    fn test_validate_kernel_name_valid() {
        assert!(validate_kernel_name("de430").is_ok());
        assert!(validate_kernel_name("de432s").is_ok());
        assert!(validate_kernel_name("de435").is_ok());
        assert!(validate_kernel_name("de438").is_ok());
        assert!(validate_kernel_name("de440").is_ok());
        assert!(validate_kernel_name("de440s").is_ok());
        assert!(validate_kernel_name("de442").is_ok());
        assert!(validate_kernel_name("de442s").is_ok());
    }

    #[test]
    fn test_validate_kernel_name_invalid() {
        assert!(validate_kernel_name("de999").is_err());
        assert!(validate_kernel_name("invalid").is_err());
        assert!(validate_kernel_name("").is_err());
    }

    #[test]
    fn test_supported_kernels_list() {
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

    #[test]
    fn test_unsupported_kernel_name_error_message() {
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

    // =========================================================================
    // Mock Tests - HTTP Client
    // =========================================================================

    #[test]
    #[serial]
    fn test_download_de_kernel_success_mocked() {
        let mut mock_client = MockHttpClient::new();
        let mock_data = create_mock_kernel_data();

        mock_client
            .expect_get()
            .withf(|url: &str| url.contains("de440s.bsp"))
            .times(1)
            .returning(move |_| Ok(mock_data.clone()));

        let downloader = NaifDownloader::new(mock_client);

        // Clear any existing cached file
        let cache_dir = get_naif_cache_dir().unwrap();
        let cache_path = PathBuf::from(&cache_dir).join("de440s.bsp");
        let _ = fs::remove_file(&cache_path);

        let result = downloader.download_de_kernel("de440s", None);
        assert!(result.is_ok());

        let path = result.unwrap();
        assert!(path.exists());
        assert!(path.to_string_lossy().contains("de440s.bsp"));

        // Cleanup
        let _ = fs::remove_file(&path);
    }

    #[test]
    #[serial]
    fn test_download_de_kernel_network_error_mocked() {
        let mut mock_client = MockHttpClient::new();

        mock_client
            .expect_get()
            .times(1)
            .returning(|_| Err(BraheError::Error("Connection refused".to_string())));

        let downloader = NaifDownloader::new(mock_client);

        // Clear cache to force download
        let cache_dir = get_naif_cache_dir().unwrap();
        let cache_path = PathBuf::from(&cache_dir).join("de440s.bsp");
        let _ = fs::remove_file(&cache_path);

        let result = downloader.download_de_kernel("de440s", None);
        assert!(result.is_err());

        let err = result.unwrap_err();
        match err {
            BraheError::Error(msg) => {
                assert!(msg.contains("Failed to download kernel"));
            }
            _ => panic!("Expected BraheError::Error"),
        }
    }

    #[test]
    #[serial]
    fn test_download_de_kernel_empty_response_mocked() {
        let mut mock_client = MockHttpClient::new();

        mock_client
            .expect_get()
            .times(1)
            .returning(|_| Ok(Vec::new())); // Empty response

        let downloader = NaifDownloader::new(mock_client);

        // Clear cache
        let cache_dir = get_naif_cache_dir().unwrap();
        let cache_path = PathBuf::from(&cache_dir).join("de440s.bsp");
        let _ = fs::remove_file(&cache_path);

        let result = downloader.download_de_kernel("de440s", None);
        assert!(result.is_err());

        let err = result.unwrap_err();
        match err {
            BraheError::Error(msg) => {
                assert!(msg.contains("No data returned"));
            }
            _ => panic!("Expected BraheError::Error"),
        }
    }

    #[test]
    #[serial]
    fn test_download_de_kernel_caching_mocked() {
        let mut mock_client = MockHttpClient::new();
        let mock_data = create_mock_kernel_data();

        // Only expect one call - second should use cache
        mock_client
            .expect_get()
            .times(1)
            .returning(move |_| Ok(mock_data.clone()));

        let downloader = NaifDownloader::new(mock_client);

        // Clear cache
        let cache_dir = get_naif_cache_dir().unwrap();
        let cache_path = PathBuf::from(&cache_dir).join("de440s.bsp");
        let _ = fs::remove_file(&cache_path);

        // First download
        let result1 = downloader.download_de_kernel("de440s", None);
        assert!(result1.is_ok());

        // Second download - should use cache (no HTTP call)
        let result2 = downloader.download_de_kernel("de440s", None);
        assert!(result2.is_ok());
        assert_eq!(result1.unwrap(), result2.unwrap());

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    #[serial]
    fn test_download_de_kernel_with_output_path_mocked() {
        let mut mock_client = MockHttpClient::new();
        let mock_data = create_mock_kernel_data();

        mock_client
            .expect_get()
            .times(1)
            .returning(move |_| Ok(mock_data.clone()));

        let downloader = NaifDownloader::new(mock_client);

        // Clear cache
        let cache_dir = get_naif_cache_dir().unwrap();
        let cache_path = PathBuf::from(&cache_dir).join("de440s.bsp");
        let _ = fs::remove_file(&cache_path);

        // Create temp directory for output
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir.path().join("custom_kernel.bsp");

        let result = downloader.download_de_kernel("de440s", Some(output_path.clone()));
        assert!(result.is_ok());

        let returned_path = result.unwrap();
        assert_eq!(returned_path, output_path);
        assert!(output_path.exists());

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    #[serial]
    fn test_download_de_kernel_creates_output_directory_mocked() {
        let mut mock_client = MockHttpClient::new();
        let mock_data = create_mock_kernel_data();

        mock_client
            .expect_get()
            .times(1)
            .returning(move |_| Ok(mock_data.clone()));

        let downloader = NaifDownloader::new(mock_client);

        // Clear cache
        let cache_dir = get_naif_cache_dir().unwrap();
        let cache_path = PathBuf::from(&cache_dir).join("de440s.bsp");
        let _ = fs::remove_file(&cache_path);

        // Create temp directory with nested path
        let temp_dir = TempDir::new().unwrap();
        let output_path = temp_dir
            .path()
            .join("nested")
            .join("dir")
            .join("kernel.bsp");

        let result = downloader.download_de_kernel("de440s", Some(output_path.clone()));
        assert!(result.is_ok());
        assert!(output_path.exists());

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    #[test]
    fn test_download_multiple_kernel_types_mocked() {
        // Test that different kernel names generate correct URLs
        let mut mock_client = MockHttpClient::new();
        let mock_data = create_mock_kernel_data();

        mock_client
            .expect_get()
            .withf(|url: &str| url.contains("de430.bsp"))
            .times(1)
            .returning({
                let data = mock_data.clone();
                move |_| Ok(data.clone())
            });

        let downloader = NaifDownloader::new(mock_client);

        // Clear cache
        let cache_dir = get_naif_cache_dir().unwrap();
        let cache_path = PathBuf::from(&cache_dir).join("de430.bsp");
        let _ = fs::remove_file(&cache_path);

        let result = downloader.download_de_kernel("de430", None);
        assert!(result.is_ok());

        // Cleanup
        let _ = fs::remove_file(&cache_path);
    }

    // =========================================================================
    // Network Tests - Only run with manual feature
    // =========================================================================

    #[test]
    #[cfg_attr(not(feature = "manual"), ignore)]
    #[serial]
    fn test_download_de440s_network() {
        setup_test_kernel();

        let result = download_de_kernel("de440s", None);
        assert!(result.is_ok());

        let kernel_path = result.unwrap();
        assert!(kernel_path.exists());
        assert!(kernel_path.to_string_lossy().contains("de440s.bsp"));

        let metadata = fs::metadata(&kernel_path).unwrap();
        assert!(metadata.len() > 0);
    }

    #[test]
    #[cfg_attr(not(feature = "manual"), ignore)]
    #[serial]
    fn test_download_with_output_path_network() {
        setup_test_kernel();

        let temp_dir = std::env::temp_dir();
        let output_path = temp_dir.join("test_de440s_output.bsp");
        let _ = fs::remove_file(&output_path);

        let result = download_de_kernel("de440s", Some(output_path.clone()));
        assert!(result.is_ok());

        let returned_path = result.unwrap();
        assert_eq!(returned_path, output_path);
        assert!(output_path.exists());

        let metadata = fs::metadata(&output_path).unwrap();
        assert!(metadata.len() > 0);

        let _ = fs::remove_file(&output_path);
    }

    #[test]
    #[cfg_attr(not(feature = "manual"), ignore)]
    #[serial]
    fn test_caching_behavior_network() {
        setup_test_kernel();

        let result1 = download_de_kernel("de440s", None);
        assert!(result1.is_ok());
        let kernel_path = result1.unwrap();

        let metadata1 = fs::metadata(&kernel_path).unwrap();
        let modified1 = metadata1.modified().unwrap();

        std::thread::sleep(std::time::Duration::from_millis(100));

        let result2 = download_de_kernel("de440s", None);
        assert!(result2.is_ok());
        assert_eq!(kernel_path, result2.unwrap());

        let metadata2 = fs::metadata(&kernel_path).unwrap();
        let modified2 = metadata2.modified().unwrap();
        assert_eq!(modified1, modified2);
    }
}
