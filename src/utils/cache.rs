/*!
 * Module containing cache directory management utilities.
 */

use std::env;
use std::fs;
use std::path::PathBuf;

use crate::utils::BraheError;

/// Get the brahe cache directory path, optionally with a subdirectory.
///
/// The cache directory is determined by the `BRAHE_CACHE` environment variable.
/// If not set, defaults to `~/.cache/brahe`.
///
/// The directory is created if it doesn't exist.
///
/// # Arguments
///
/// * `subdirectory` - Optional subdirectory to append to the cache path
///
/// # Returns
///
/// * `Result<String, BraheError>` - The full path to the cache directory as a String
///
/// # Examples
/// ```
/// use brahe::utils::cache::get_brahe_cache_dir_with_subdir;
/// let cache_dir = get_brahe_cache_dir_with_subdir(None).unwrap();
/// println!("Cache directory: {}", cache_dir);
///
/// let eop_cache_dir = get_brahe_cache_dir_with_subdir(Some("eop")).unwrap();
/// println!("EOP cache directory: {}", eop_cache_dir);
/// ```
pub fn get_brahe_cache_dir_with_subdir(subdirectory: Option<&str>) -> Result<String, BraheError> {
    // Check for environment variable first
    let mut cache_path = if let Ok(cache_env) = env::var("BRAHE_CACHE") {
        PathBuf::from(cache_env)
    } else {
        // Default to ~/.cache/brahe
        let home_dir = dirs::home_dir()
            .ok_or_else(|| BraheError::IoError("Could not determine home directory".to_string()))?;
        home_dir.join(".cache").join("brahe")
    };

    // Append subdirectory if provided
    if let Some(subdir) = subdirectory {
        cache_path = cache_path.join(subdir);
    }

    // Create directory if it doesn't exist
    if !cache_path.exists() {
        fs::create_dir_all(&cache_path)
            .map_err(|e| BraheError::IoError(format!("Failed to create cache directory: {}", e)))?;
    }

    // Convert to string
    cache_path
        .to_str()
        .ok_or_else(|| {
            BraheError::IoError("Cache path contains invalid UTF-8 characters".to_string())
        })
        .map(|s| s.to_string())
}

/// Get the brahe cache directory path.
///
/// The cache directory is determined by the `BRAHE_CACHE` environment variable.
/// If not set, defaults to `~/.cache/brahe`.
///
/// The directory is created if it doesn't exist.
///
/// # Returns
///
/// * `Result<String, BraheError>` - The full path to the cache directory as a String
///
/// # Examples
/// ```
/// use brahe::utils::get_brahe_cache_dir;
/// let cache_dir = get_brahe_cache_dir().unwrap();
/// println!("Cache directory: {}", cache_dir);
/// ```
pub fn get_brahe_cache_dir() -> Result<String, BraheError> {
    get_brahe_cache_dir_with_subdir(None)
}

/// Get the EOP cache directory path.
///
/// Returns `~/.cache/brahe/eop` (or `$BRAHE_CACHE/eop` if environment variable is set).
/// The directory is created if it doesn't exist.
///
/// # Returns
///
/// * `Result<String, BraheError>` - The full path to the EOP cache directory as a String
///
/// # Examples
/// ```
/// use brahe::utils::cache::get_eop_cache_dir;
/// let eop_cache_dir = get_eop_cache_dir().unwrap();
/// println!("EOP cache directory: {}", eop_cache_dir);
/// ```
pub fn get_eop_cache_dir() -> Result<String, BraheError> {
    get_brahe_cache_dir_with_subdir(Some("eop"))
}

/// Get the CelesTrak cache directory path.
///
/// Returns `~/.cache/brahe/celestrak` (or `$BRAHE_CACHE/celestrak` if environment variable is set).
/// The directory is created if it doesn't exist.
///
/// # Returns
///
/// * `Result<String, BraheError>` - The full path to the CelesTrak cache directory as a String
///
/// # Examples
/// ```
/// use brahe::utils::cache::get_celestrak_cache_dir;
/// let celestrak_cache_dir = get_celestrak_cache_dir().unwrap();
/// println!("CelesTrak cache directory: {}", celestrak_cache_dir);
/// ```
pub fn get_celestrak_cache_dir() -> Result<String, BraheError> {
    get_brahe_cache_dir_with_subdir(Some("celestrak"))
}

/// Get the NAIF cache directory path.
///
/// Returns `~/.cache/brahe/naif` (or `$BRAHE_CACHE/naif` if environment variable is set).
/// The directory is created if it doesn't exist.
///
/// # Returns
///
/// * `Result<String, BraheError>` - The full path to the NAIF cache directory as a String
///
/// # Examples
/// ```
/// use brahe::utils::cache::get_naif_cache_dir;
/// let naif_cache_dir = get_naif_cache_dir().unwrap();
/// println!("NAIF cache directory: {}", naif_cache_dir);
/// ```
pub fn get_naif_cache_dir() -> Result<String, BraheError> {
    get_brahe_cache_dir_with_subdir(Some("naif"))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn test_get_brahe_cache_dir() {
        // Save current env var state
        let original_brahe_cache = env::var("BRAHE_CACHE").ok();

        // Clear BRAHE_CACHE to ensure we use default path
        unsafe {
            env::remove_var("BRAHE_CACHE");
        }

        let cache_dir = get_brahe_cache_dir().unwrap();
        assert!(!cache_dir.is_empty());
        assert!(std::path::Path::new(&cache_dir).exists());
        assert!(cache_dir.ends_with("brahe"));

        // Restore original state
        unsafe {
            if let Some(original) = original_brahe_cache {
                env::set_var("BRAHE_CACHE", original);
            }
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_get_eop_cache_dir() {
        // Save current env var state
        let original_brahe_cache = env::var("BRAHE_CACHE").ok();

        // Clear BRAHE_CACHE to ensure we use default path
        unsafe {
            env::remove_var("BRAHE_CACHE");
        }

        let eop_cache_dir = get_eop_cache_dir().unwrap();
        assert!(!eop_cache_dir.is_empty());
        assert!(std::path::Path::new(&eop_cache_dir).exists());
        // Check that path contains both brahe and eop directories
        assert!(eop_cache_dir.contains("brahe"));
        assert!(eop_cache_dir.contains("eop"));

        // Restore original state
        unsafe {
            if let Some(original) = original_brahe_cache {
                env::set_var("BRAHE_CACHE", original);
            }
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_get_celestrak_cache_dir() {
        // Save current env var state
        let original_brahe_cache = env::var("BRAHE_CACHE").ok();

        // Clear BRAHE_CACHE to ensure we use default path
        unsafe {
            env::remove_var("BRAHE_CACHE");
        }

        let celestrak_cache_dir = get_celestrak_cache_dir().unwrap();
        assert!(!celestrak_cache_dir.is_empty());
        assert!(std::path::Path::new(&celestrak_cache_dir).exists());
        // Check that path contains both brahe and celestrak directories
        assert!(celestrak_cache_dir.contains("brahe"));
        assert!(celestrak_cache_dir.contains("celestrak"));

        // Restore original state
        unsafe {
            if let Some(original) = original_brahe_cache {
                env::set_var("BRAHE_CACHE", original);
            }
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_get_naif_cache_dir() {
        // Save current env var state
        let original_brahe_cache = env::var("BRAHE_CACHE").ok();

        // Clear BRAHE_CACHE to ensure we use default path
        unsafe {
            env::remove_var("BRAHE_CACHE");
        }

        let naif_cache_dir = get_naif_cache_dir().unwrap();
        assert!(!naif_cache_dir.is_empty());
        assert!(std::path::Path::new(&naif_cache_dir).exists());
        // Check that path contains both brahe and naif directories
        assert!(naif_cache_dir.contains("brahe"));
        assert!(naif_cache_dir.contains("naif"));

        // Restore original state
        unsafe {
            if let Some(original) = original_brahe_cache {
                env::set_var("BRAHE_CACHE", original);
            }
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_cache_dir_from_env() {
        // Save current env var state
        let original_brahe_cache = env::var("BRAHE_CACHE").ok();

        let test_dir = std::env::temp_dir().join("brahe_test_cache_env");
        unsafe {
            env::set_var("BRAHE_CACHE", test_dir.to_str().unwrap());
        }

        let cache_dir = get_brahe_cache_dir().unwrap();
        assert_eq!(cache_dir, test_dir.to_str().unwrap());
        assert!(std::path::Path::new(&cache_dir).exists());

        // Test that subdirectories also use the custom cache path
        let eop_cache_dir = get_eop_cache_dir().unwrap();
        assert!(
            eop_cache_dir.starts_with(test_dir.to_str().unwrap()),
            "Expected eop_cache_dir '{}' to start with '{}'",
            eop_cache_dir,
            test_dir.to_str().unwrap()
        );
        assert!(eop_cache_dir.contains("eop"));

        let celestrak_cache_dir = get_celestrak_cache_dir().unwrap();
        assert!(
            celestrak_cache_dir.starts_with(test_dir.to_str().unwrap()),
            "Expected celestrak_cache_dir '{}' to start with '{}'",
            celestrak_cache_dir,
            test_dir.to_str().unwrap()
        );
        assert!(celestrak_cache_dir.contains("celestrak"));

        let naif_cache_dir = get_naif_cache_dir().unwrap();
        assert!(
            naif_cache_dir.starts_with(test_dir.to_str().unwrap()),
            "Expected naif_cache_dir '{}' to start with '{}'",
            naif_cache_dir,
            test_dir.to_str().unwrap()
        );
        assert!(naif_cache_dir.contains("naif"));

        // Cleanup - restore original state
        unsafe {
            if let Some(original) = original_brahe_cache {
                env::set_var("BRAHE_CACHE", original);
            } else {
                env::remove_var("BRAHE_CACHE");
            }
        }
        let _ = std::fs::remove_dir_all(&test_dir);
    }

    #[test]
    #[serial_test::serial]
    fn test_get_brahe_cache_dir_with_subdir() {
        // Save current env var state
        let original_brahe_cache = env::var("BRAHE_CACHE").ok();

        // Clear BRAHE_CACHE to ensure we use default path
        unsafe {
            env::remove_var("BRAHE_CACHE");
        }

        let cache_dir_no_subdir = get_brahe_cache_dir_with_subdir(None).unwrap();
        assert!(!cache_dir_no_subdir.is_empty());

        let cache_dir_with_subdir = get_brahe_cache_dir_with_subdir(Some("test_subdir")).unwrap();
        assert!(!cache_dir_with_subdir.is_empty());

        // Verify the subdirectory was created
        let path = PathBuf::from(&cache_dir_with_subdir);
        assert!(path.exists());

        // Verify the path includes the subdirectory
        assert!(cache_dir_with_subdir.contains("test_subdir"));

        // Verify final component is "test_subdir"
        assert_eq!(path.file_name().unwrap().to_string_lossy(), "test_subdir");

        // Verify parent is the brahe cache directory (string comparison)
        assert_eq!(
            path.parent().unwrap().to_str().unwrap(),
            cache_dir_no_subdir
        );

        // Cleanup test directory
        let _ = std::fs::remove_dir_all(&cache_dir_with_subdir);

        // Restore original state
        unsafe {
            if let Some(original) = original_brahe_cache {
                env::set_var("BRAHE_CACHE", original);
            }
        }
    }
}
