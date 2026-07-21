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

/// Get the space weather cache directory path.
///
/// Returns `~/.cache/brahe/space_weather` (or `$BRAHE_CACHE/space_weather` if environment variable is set).
/// The directory is created if it doesn't exist.
///
/// # Returns
///
/// * `Result<String, BraheError>` - The full path to the space weather cache directory as a String
///
/// # Examples
/// ```
/// use brahe::utils::cache::get_space_weather_cache_dir;
/// let sw_cache_dir = get_space_weather_cache_dir().unwrap();
/// println!("Space weather cache directory: {}", sw_cache_dir);
/// ```
pub fn get_space_weather_cache_dir() -> Result<String, BraheError> {
    get_brahe_cache_dir_with_subdir(Some("space_weather"))
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

/// Get the ICGEM cache directory path.
///
/// Returns `~/.cache/brahe/icgem` (or `$BRAHE_CACHE/icgem` if environment variable is set).
/// The directory is created if it doesn't exist.
///
/// # Returns
///
/// * `Result<String, BraheError>` - The full path to the ICGEM cache directory as a String
///
/// # Examples
/// ```
/// use brahe::utils::cache::get_icgem_cache_dir;
/// let icgem_cache_dir = get_icgem_cache_dir().unwrap();
/// println!("ICGEM cache directory: {}", icgem_cache_dir);
/// ```
pub fn get_icgem_cache_dir() -> Result<String, BraheError> {
    get_brahe_cache_dir_with_subdir(Some("icgem"))
}

/// Get the brahe tide-model cache directory (`$BRAHE_CACHE/tides`), creating
/// it if needed. Holds the one-time FES2004 ocean tide coefficient download.
///
/// # Returns
///
/// * `Result<String, BraheError>` - The full path to the tides cache directory as a String
///
/// # Examples
/// ```
/// use brahe::utils::cache::get_tides_cache_dir;
/// let tides_cache_dir = get_tides_cache_dir().unwrap();
/// println!("Tides cache directory: {}", tides_cache_dir);
/// ```
pub fn get_tides_cache_dir() -> Result<String, BraheError> {
    get_brahe_cache_dir_with_subdir(Some("tides"))
}

/// Get the star catalog cache directory path.
///
/// Returns `~/.cache/brahe/star_catalogs` (or `$BRAHE_CACHE/star_catalogs` if environment variable is set).
/// The directory is created if it doesn't exist.
///
/// # Returns
///
/// * `Result<String, BraheError>` - The full path to the star catalog cache directory as a String
///
/// # Examples
/// ```
/// use brahe::utils::cache::get_star_catalogs_cache_dir;
/// let star_catalogs_cache_dir = get_star_catalogs_cache_dir().unwrap();
/// println!("Star catalog cache directory: {}", star_catalogs_cache_dir);
/// ```
pub fn get_star_catalogs_cache_dir() -> Result<String, BraheError> {
    get_brahe_cache_dir_with_subdir(Some("star_catalogs"))
}

/// Get the SBDB cache directory path.
///
/// Returns `~/.cache/brahe/sbdb` (or `$BRAHE_CACHE/sbdb` if the environment
/// variable is set). The directory is created if it doesn't exist.
///
/// # Returns
///
/// * `Result<String, BraheError>` - The full path to the SBDB cache directory.
///
/// # Examples
/// ```
/// use brahe::utils::cache::get_sbdb_cache_dir;
/// let dir = get_sbdb_cache_dir().unwrap();
/// println!("SBDB cache directory: {}", dir);
/// ```
pub fn get_sbdb_cache_dir() -> Result<String, BraheError> {
    get_brahe_cache_dir_with_subdir(Some("sbdb"))
}

/// Get the Horizons cache directory path.
///
/// Returns `~/.cache/brahe/horizons` (or `$BRAHE_CACHE/horizons` if the
/// environment variable is set). The directory is created if it doesn't exist.
///
/// # Returns
///
/// * `Result<String, BraheError>` - The full path to the Horizons cache directory.
///
/// # Examples
/// ```
/// use brahe::utils::cache::get_horizons_cache_dir;
/// let dir = get_horizons_cache_dir().unwrap();
/// println!("Horizons cache directory: {}", dir);
/// ```
pub fn get_horizons_cache_dir() -> Result<String, BraheError> {
    get_brahe_cache_dir_with_subdir(Some("horizons"))
}

/// Compute a stable 16-hex-character hash of `input` for cache-key filenames.
///
/// Used to derive collision-resistant cache filenames from request parameters
/// (e.g. an SBDB search string, or a Horizons command + time span + center).
///
/// # Arguments
///
/// * `input` - The string to hash.
///
/// # Returns
///
/// * `String` - A 16-character lowercase hexadecimal digest.
pub fn short_hash(input: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    input.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    /// Restore `BRAHE_CACHE` to `original`: set it back if it was
    /// previously set, or leave it unset otherwise. Shared by the
    /// sbdb/horizons cache-dir tests below so both restore arms are
    /// exercised across their two scenarios (BRAHE_CACHE ambient-unset vs.
    /// already-set).
    fn restore_brahe_cache(original: Option<String>) {
        match original {
            Some(v) => unsafe { env::set_var("BRAHE_CACHE", v) },
            None => unsafe { env::remove_var("BRAHE_CACHE") },
        }
    }

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
    fn test_get_space_weather_cache_dir() {
        // Save current env var state
        let original_brahe_cache = env::var("BRAHE_CACHE").ok();

        // Clear BRAHE_CACHE to ensure we use default path
        unsafe {
            env::remove_var("BRAHE_CACHE");
        }

        let sw_cache_dir = get_space_weather_cache_dir().unwrap();
        assert!(!sw_cache_dir.is_empty());
        assert!(std::path::Path::new(&sw_cache_dir).exists());
        // Check that path contains both brahe and space_weather directories
        assert!(sw_cache_dir.contains("brahe"));
        assert!(sw_cache_dir.contains("space_weather"));

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

        let sw_cache_dir = get_space_weather_cache_dir().unwrap();
        assert!(
            sw_cache_dir.starts_with(test_dir.to_str().unwrap()),
            "Expected sw_cache_dir '{}' to start with '{}'",
            sw_cache_dir,
            test_dir.to_str().unwrap()
        );
        assert!(sw_cache_dir.contains("space_weather"));

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
    fn test_get_icgem_cache_dir() {
        let original_brahe_cache = env::var("BRAHE_CACHE").ok();
        unsafe {
            env::remove_var("BRAHE_CACHE");
        }

        let icgem_cache_dir = get_icgem_cache_dir().unwrap();
        assert!(!icgem_cache_dir.is_empty());
        assert!(std::path::Path::new(&icgem_cache_dir).exists());
        assert!(icgem_cache_dir.contains("brahe"));
        assert!(icgem_cache_dir.contains("icgem"));

        unsafe {
            if let Some(original) = original_brahe_cache {
                env::set_var("BRAHE_CACHE", original);
            }
        }
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

    #[test]
    #[serial_test::serial]
    fn test_get_sbdb_cache_dir() {
        let true_original = env::var("BRAHE_CACHE").ok();

        // BRAHE_CACHE unset: falls back to the ~/.cache/brahe default, and
        // restoring afterward takes the "leave unset" arm.
        unsafe { env::remove_var("BRAHE_CACHE") };
        let original = env::var("BRAHE_CACHE").ok();
        let dir = get_sbdb_cache_dir().unwrap();
        // Component-based match so it holds regardless of the OS path separator.
        assert!(std::path::Path::new(&dir).ends_with("brahe/sbdb"));
        restore_brahe_cache(original);

        // BRAHE_CACHE already set: the directory nests under it instead,
        // and restoring afterward takes the "put back the previous value"
        // arm.
        let sentinel = std::env::temp_dir().join("brahe_test_sbdb_cache_sentinel");
        unsafe { env::set_var("BRAHE_CACHE", sentinel.to_str().unwrap()) };
        let original = env::var("BRAHE_CACHE").ok();
        let test_dir = std::env::temp_dir().join("brahe_test_sbdb_cache_env");
        unsafe { env::set_var("BRAHE_CACHE", test_dir.to_str().unwrap()) };
        let dir = get_sbdb_cache_dir().unwrap();
        assert!(dir.ends_with("sbdb"));
        assert!(dir.starts_with(test_dir.to_str().unwrap()));
        let _ = std::fs::remove_dir_all(&test_dir);
        restore_brahe_cache(original);
        assert_eq!(env::var("BRAHE_CACHE").ok().as_deref(), sentinel.to_str());

        restore_brahe_cache(true_original);
    }

    #[test]
    #[serial_test::serial]
    fn test_get_horizons_cache_dir() {
        let true_original = env::var("BRAHE_CACHE").ok();

        // BRAHE_CACHE unset: falls back to the ~/.cache/brahe default, and
        // restoring afterward takes the "leave unset" arm.
        unsafe { env::remove_var("BRAHE_CACHE") };
        let original = env::var("BRAHE_CACHE").ok();
        let dir = get_horizons_cache_dir().unwrap();
        // Component-based match so it holds regardless of the OS path separator.
        assert!(std::path::Path::new(&dir).ends_with("brahe/horizons"));
        restore_brahe_cache(original);

        // BRAHE_CACHE already set: the directory nests under it instead,
        // and restoring afterward takes the "put back the previous value"
        // arm.
        let sentinel = std::env::temp_dir().join("brahe_test_horizons_cache_sentinel");
        unsafe { env::set_var("BRAHE_CACHE", sentinel.to_str().unwrap()) };
        let original = env::var("BRAHE_CACHE").ok();
        let test_dir = std::env::temp_dir().join("brahe_test_horizons_cache_env");
        unsafe { env::set_var("BRAHE_CACHE", test_dir.to_str().unwrap()) };
        let dir = get_horizons_cache_dir().unwrap();
        assert!(dir.ends_with("horizons"));
        assert!(dir.starts_with(test_dir.to_str().unwrap()));
        let _ = std::fs::remove_dir_all(&test_dir);
        restore_brahe_cache(original);
        assert_eq!(env::var("BRAHE_CACHE").ok().as_deref(), sentinel.to_str());

        restore_brahe_cache(true_original);
    }

    #[test]
    fn test_short_hash_is_stable_and_hex() {
        let a = short_hash("DES=2000001;|2015-12-01|2016-03-01|500@0");
        let b = short_hash("DES=2000001;|2015-12-01|2016-03-01|500@0");
        assert_eq!(a, b);
        assert_eq!(a.len(), 16);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, short_hash("different"));
    }
}
