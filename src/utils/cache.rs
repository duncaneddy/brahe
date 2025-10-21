/*!
 * Module containing cache directory management utilities.
 */

use std::env;
use std::fs;
use std::path::PathBuf;

use crate::utils::BraheError;

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
    // Check for environment variable first
    let cache_path = if let Ok(cache_env) = env::var("BRAHE_CACHE") {
        PathBuf::from(cache_env)
    } else {
        // Default to ~/.cache/brahe
        let home_dir = dirs::home_dir()
            .ok_or_else(|| BraheError::IoError("Could not determine home directory".to_string()))?;
        home_dir.join(".cache").join("brahe")
    };

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_brahe_cache_dir() {
        let cache_dir = get_brahe_cache_dir().unwrap();
        assert!(!cache_dir.is_empty());
        assert!(std::path::Path::new(&cache_dir).exists());
    }

    #[test]
    fn test_cache_dir_from_env() {
        let test_dir = std::env::temp_dir().join("brahe_test_cache");
        unsafe {
            env::set_var("BRAHE_CACHE", test_dir.to_str().unwrap());
        }

        let cache_dir = get_brahe_cache_dir().unwrap();
        assert_eq!(cache_dir, test_dir.to_str().unwrap());
        assert!(std::path::Path::new(&cache_dir).exists());

        // Cleanup
        unsafe {
            env::remove_var("BRAHE_CACHE");
        }
        let _ = std::fs::remove_dir_all(&test_dir);
    }
}
