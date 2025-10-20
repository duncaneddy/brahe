/*!
 * Defines the CachingEOPProvider that checks file age and downloads updates
 */

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use crate::eop::download::{download_c04_eop_file, download_standard_eop_file};
use crate::eop::eop_provider::EarthOrientationProvider;
use crate::eop::eop_types::{EOPExtrapolation, EOPType};
use crate::eop::file_provider::FileEOPProvider;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

/// Provides Earth Orientation Parameter (EOP) data with automatic cache refresh.
///
/// The `CachingEOPProvider` wraps a `FileEOPProvider` and automatically checks the age
/// of the EOP file. If the file is older than the configured maximum age, it downloads
/// an updated version before loading the data.
///
/// This is useful for applications that need to maintain current EOP data without manual
/// intervention, such as long-running services or applications that need accurate
/// reference frame transformations.
///
/// # Fields
///
/// - `filepath`: Path to the EOP file
/// - `eop_type`: Type of EOP file (C04 or StandardBulletinA)
/// - `max_age_seconds`: Maximum age of the file in seconds before triggering a download
/// - `auto_refresh`: If true, automatically check file age on each access and refresh if needed
/// - `interpolate`: Whether to interpolate between data points
/// - `extrapolate`: Behavior for out-of-bounds data access
/// - `provider`: Internal FileEOPProvider that actually loads and provides the data
/// - `file_loaded_at`: Timestamp when the file was last loaded
///
/// # Example
///
/// ```no_run
/// use std::path::Path;
/// use brahe::eop::{CachingEOPProvider, EOPType, EOPExtrapolation};
///
/// // Create a caching provider that refreshes files older than 7 days
/// let filepath = Path::new("/tmp/finals.all.iau2000.txt");
/// let max_age_days = 7;
/// let max_age_seconds = max_age_days * 86400;
///
/// let provider = CachingEOPProvider::new(
///     filepath,
///     EOPType::StandardBulletinA,
///     max_age_seconds,
///     false,
///     true,
///     EOPExtrapolation::Hold
/// ).unwrap();
/// ```
#[derive(Clone)]
pub struct CachingEOPProvider {
    filepath: PathBuf,
    eop_type: EOPType,
    max_age_seconds: u64,
    pub auto_refresh: bool,
    interpolate: bool,
    extrapolate: EOPExtrapolation,
    provider: Arc<Mutex<FileEOPProvider>>,
    file_loaded_at: Arc<Mutex<SystemTime>>,
}

impl CachingEOPProvider {
    /// Creates a new CachingEOPProvider that checks file age and downloads updates as needed.
    ///
    /// If the file doesn't exist, it will be downloaded. If the file exists but is older than
    /// `max_age_seconds`, it will be re-downloaded before loading. Otherwise, the existing file
    /// is loaded.
    ///
    /// # Arguments
    ///
    /// * `filepath` - Path to the EOP file
    /// * `eop_type` - Type of EOP file (C04 or StandardBulletinA)
    /// * `max_age_seconds` - Maximum age of the file in seconds before triggering a download
    /// * `auto_refresh` - If true, automatically check file age on each access and refresh if needed
    /// * `interpolate` - Whether to interpolate between data points
    /// * `extrapolate` - Behavior for out-of-bounds data access
    ///
    /// # Returns
    ///
    /// * `Result<CachingEOPProvider, BraheError>` - CachingEOPProvider with loaded data, or an error
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::path::Path;
    /// use brahe::eop::{CachingEOPProvider, EOPType, EOPExtrapolation};
    ///
    /// let filepath = Path::new("/tmp/finals.all.iau2000.txt");
    /// let provider = CachingEOPProvider::new(
    ///     filepath,
    ///     EOPType::StandardBulletinA,
    ///     7 * 86400, // 7 days
    ///     false,     // auto_refresh
    ///     true,
    ///     EOPExtrapolation::Hold
    /// ).unwrap();
    /// ```
    pub fn new(
        filepath: &Path,
        eop_type: EOPType,
        max_age_seconds: u64,
        auto_refresh: bool,
        interpolate: bool,
        extrapolate: EOPExtrapolation,
    ) -> Result<Self, BraheError> {
        let filepath = filepath.to_path_buf();

        // Check if file needs to be downloaded
        let needs_download = Self::check_file_age(&filepath, max_age_seconds)?;

        if needs_download {
            Self::download_file(&filepath, eop_type)?;
        }

        // Load the file into a FileEOPProvider
        let provider = FileEOPProvider::from_file(&filepath, interpolate, extrapolate)?;

        // Record when file was loaded
        let file_loaded_at = Arc::new(Mutex::new(SystemTime::now()));

        Ok(Self {
            filepath,
            eop_type,
            max_age_seconds,
            auto_refresh,
            interpolate,
            extrapolate,
            provider: Arc::new(Mutex::new(provider)),
            file_loaded_at,
        })
    }

    /// Checks if a file needs to be downloaded based on its age.
    ///
    /// Returns `true` if:
    /// - The file doesn't exist
    /// - The file's modification time cannot be determined
    /// - The file is older than `max_age_seconds`
    ///
    /// # Arguments
    ///
    /// * `filepath` - Path to check
    /// * `max_age_seconds` - Maximum acceptable age in seconds
    ///
    /// # Returns
    ///
    /// * `Ok(true)` - File needs to be downloaded
    /// * `Ok(false)` - File exists and is current
    /// * `Err(BraheError)` - Error checking file
    fn check_file_age(filepath: &Path, max_age_seconds: u64) -> Result<bool, BraheError> {
        // If file doesn't exist, we need to download it
        if !filepath.exists() {
            return Ok(true);
        }

        // Get file metadata
        let metadata = fs::metadata(filepath).map_err(|e| {
            BraheError::IoError(format!(
                "Failed to get metadata for {}: {}",
                filepath.display(),
                e
            ))
        })?;

        // Get file modification time
        let modified = metadata.modified().map_err(|e| {
            BraheError::IoError(format!(
                "Failed to get modification time for {}: {}",
                filepath.display(),
                e
            ))
        })?;

        // Get current time
        let now = SystemTime::now();

        // Calculate file age in seconds
        let age = now
            .duration_since(modified)
            .map_err(|e| {
                BraheError::IoError(format!(
                    "Failed to calculate file age for {}: {}",
                    filepath.display(),
                    e
                ))
            })?
            .as_secs();

        // Return true if file is older than max age
        Ok(age > max_age_seconds)
    }

    /// Downloads an EOP file to the specified path.
    ///
    /// # Arguments
    ///
    /// * `filepath` - Path where the file should be saved
    /// * `eop_type` - Type of EOP file to download (C04 or StandardBulletinA)
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Download succeeded
    /// * `Err(BraheError)` - Download failed
    fn download_file(filepath: &Path, eop_type: EOPType) -> Result<(), BraheError> {
        let filepath_str = filepath
            .to_str()
            .ok_or_else(|| BraheError::IoError("Invalid file path".to_string()))?;

        match eop_type {
            EOPType::C04 => download_c04_eop_file(filepath_str).map_err(|e| {
                BraheError::IoError(format!("Failed to download C04 EOP file: {}", e))
            }),
            EOPType::StandardBulletinA => download_standard_eop_file(filepath_str).map_err(|e| {
                BraheError::IoError(format!("Failed to download Standard EOP file: {}", e))
            }),
            _ => Err(BraheError::EOPError(format!(
                "Unsupported EOP type for download: {:?}",
                eop_type
            ))),
        }
    }

    /// Refreshes the cached EOP data by re-checking the file age and reloading if necessary.
    ///
    /// This method allows manual refresh of the cache without creating a new provider instance.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - Refresh succeeded (or wasn't needed)
    /// * `Err(BraheError)` - Refresh failed
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::path::Path;
    /// use brahe::eop::{CachingEOPProvider, EOPType, EOPExtrapolation};
    ///
    /// let filepath = Path::new("/tmp/finals.all.iau2000.txt");
    /// let mut provider = CachingEOPProvider::new(
    ///     filepath,
    ///     EOPType::StandardBulletinA,
    ///     7 * 86400,
    ///     false,
    ///     true,
    ///     EOPExtrapolation::Hold
    /// ).unwrap();
    ///
    /// // Later, force a refresh check
    /// provider.refresh().unwrap();
    /// ```
    pub fn refresh(&self) -> Result<(), BraheError> {
        let needs_download = Self::check_file_age(&self.filepath, self.max_age_seconds)?;

        if needs_download {
            Self::download_file(&self.filepath, self.eop_type)?;
            let new_provider =
                FileEOPProvider::from_file(&self.filepath, self.interpolate, self.extrapolate)?;
            *self.provider.lock().unwrap() = new_provider;
            *self.file_loaded_at.lock().unwrap() = SystemTime::now();
        }

        Ok(())
    }

    /// Returns the Epoch when the EOP file was last loaded into memory, in UTC.
    ///
    /// This represents the timestamp when the file was last loaded into memory.
    ///
    /// # Returns
    ///
    /// * `Epoch` - Epoch representing when the file was loaded, in UTC time system
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::path::Path;
    /// use brahe::eop::{CachingEOPProvider, EOPType, EOPExtrapolation};
    ///
    /// let filepath = Path::new("/tmp/finals.all.iau2000.txt");
    /// let provider = CachingEOPProvider::new(
    ///     filepath,
    ///     EOPType::StandardBulletinA,
    ///     7 * 86400,
    ///     false,
    ///     true,
    ///     EOPExtrapolation::Hold
    /// ).unwrap();
    ///
    /// let file_epoch = provider.file_epoch();
    /// println!("EOP file loaded at: {}", file_epoch);
    /// ```
    pub fn file_epoch(&self) -> Epoch {
        let system_time = *self.file_loaded_at.lock().unwrap();

        // Convert SystemTime to Epoch
        // SystemTime is based on UNIX epoch (1970-01-01 00:00:00 UTC)
        let duration_since_unix_epoch = system_time
            .duration_since(SystemTime::UNIX_EPOCH)
            .expect("System time is before UNIX epoch");

        let seconds_since_unix = duration_since_unix_epoch.as_secs_f64();

        // UNIX epoch in MJD: 1970-01-01 00:00:00 UTC = MJD 40587.0
        const UNIX_EPOCH_MJD: f64 = 40587.0;
        let mjd = UNIX_EPOCH_MJD + seconds_since_unix / 86400.0;

        Epoch::from_mjd(mjd, TimeSystem::UTC)
    }

    /// Returns the age of the currently loaded EOP file in seconds.
    ///
    /// Calculates how many seconds have elapsed since the file was loaded.
    ///
    /// # Returns
    ///
    /// * `f64` - Age of the loaded file in seconds
    ///
    /// # Example
    ///
    /// ```no_run
    /// use std::path::Path;
    /// use brahe::eop::{CachingEOPProvider, EOPType, EOPExtrapolation};
    ///
    /// let filepath = Path::new("/tmp/finals.all.iau2000.txt");
    /// let provider = CachingEOPProvider::new(
    ///     filepath,
    ///     EOPType::StandardBulletinA,
    ///     7 * 86400,
    ///     false,
    ///     true,
    ///     EOPExtrapolation::Hold
    /// ).unwrap();
    ///
    /// let age_seconds = provider.file_age();
    /// println!("EOP file age: {:.2} seconds", age_seconds);
    /// ```
    pub fn file_age(&self) -> f64 {
        let system_time = *self.file_loaded_at.lock().unwrap();
        let now = SystemTime::now();

        let duration = now
            .duration_since(system_time)
            .expect("System time went backwards");

        duration.as_secs_f64()
    }

    /// Checks if auto-refresh is needed and performs it if necessary.
    ///
    /// This is an internal method called by EarthOrientationProvider trait methods
    /// when auto_refresh is enabled.
    fn check_auto_refresh(&self) -> Result<(), BraheError> {
        if self.auto_refresh {
            self.refresh()?;
        }
        Ok(())
    }
}

impl EarthOrientationProvider for CachingEOPProvider {
    fn is_initialized(&self) -> bool {
        self.provider.lock().unwrap().is_initialized()
    }

    fn len(&self) -> usize {
        self.provider.lock().unwrap().len()
    }

    fn eop_type(&self) -> EOPType {
        self.provider.lock().unwrap().eop_type()
    }

    fn extrapolation(&self) -> EOPExtrapolation {
        self.provider.lock().unwrap().extrapolation()
    }

    fn interpolation(&self) -> bool {
        self.provider.lock().unwrap().interpolation()
    }

    fn mjd_min(&self) -> f64 {
        self.provider.lock().unwrap().mjd_min()
    }

    fn mjd_max(&self) -> f64 {
        self.provider.lock().unwrap().mjd_max()
    }

    fn mjd_last_lod(&self) -> f64 {
        self.provider.lock().unwrap().mjd_last_lod()
    }

    fn mjd_last_dxdy(&self) -> f64 {
        self.provider.lock().unwrap().mjd_last_dxdy()
    }

    fn get_ut1_utc(&self, mjd: f64) -> Result<f64, BraheError> {
        self.check_auto_refresh()?;
        self.provider.lock().unwrap().get_ut1_utc(mjd)
    }

    fn get_pm(&self, mjd: f64) -> Result<(f64, f64), BraheError> {
        self.check_auto_refresh()?;
        self.provider.lock().unwrap().get_pm(mjd)
    }

    fn get_dxdy(&self, mjd: f64) -> Result<(f64, f64), BraheError> {
        self.check_auto_refresh()?;
        self.provider.lock().unwrap().get_dxdy(mjd)
    }

    fn get_lod(&self, mjd: f64) -> Result<f64, BraheError> {
        self.check_auto_refresh()?;
        self.provider.lock().unwrap().get_lod(mjd)
    }

    fn get_eop(&self, mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), BraheError> {
        self.check_auto_refresh()?;
        self.provider.lock().unwrap().get_eop(mjd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs::File;
    use std::thread;
    use std::time::Duration;
    use tempfile::tempdir;

    #[test]
    fn test_check_file_age_nonexistent() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("nonexistent.txt");

        // Non-existent file should need download
        assert!(CachingEOPProvider::check_file_age(&filepath, 86400).unwrap());
    }

    #[test]
    fn test_check_file_age_current() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("current.txt");

        // Create a new file
        File::create(&filepath).unwrap();

        // File should be current (less than 1 day old)
        assert!(!CachingEOPProvider::check_file_age(&filepath, 86400).unwrap());
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_check_file_age_stale() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("stale.txt");

        // Create a file
        File::create(&filepath).unwrap();

        // Sleep briefly to ensure some time passes
        // Some file systems have 1-second resolution, so we need to sleep at least 1 second
        thread::sleep(Duration::from_secs(2));

        // Check with a very small max age (file should be stale)
        assert!(CachingEOPProvider::check_file_age(&filepath, 1).unwrap());
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_new_with_existing_file() {
        // Copy test EOP file to temporary location
        let dir = tempdir().unwrap();
        let src_path = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("test_assets")
            .join("finals.all.iau2000.txt");
        let dest_path = dir.path().join("test_eop.txt");

        fs::copy(&src_path, &dest_path).unwrap();

        // Create provider with large max age (file should be used as-is)
        let provider = CachingEOPProvider::new(
            &dest_path,
            EOPType::StandardBulletinA,
            365 * 86400, // 1 year
            false,       // auto_refresh
            true,
            EOPExtrapolation::Hold,
        )
        .unwrap();

        assert!(provider.is_initialized());
        assert_eq!(provider.eop_type(), EOPType::StandardBulletinA);
        assert!(provider.len() > 0);
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_new_creates_missing_file() {
        // This test requires network access and is marked to skip in normal test runs
        // Uncomment the line below to run it manually
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("downloaded_eop.txt");

        let provider = CachingEOPProvider::new(
            &filepath,
            EOPType::StandardBulletinA,
            7 * 86400,
            true,
            true,
            EOPExtrapolation::Hold,
        )
        .unwrap();

        assert!(filepath.exists());
        assert!(provider.is_initialized());
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_refresh() {
        // Copy test EOP file to temporary location
        let dir = tempdir().unwrap();
        let src_path = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("test_assets")
            .join("finals.all.iau2000.txt");
        let dest_path = dir.path().join("test_eop_refresh.txt");

        fs::copy(&src_path, &dest_path).unwrap();

        // Create provider
        let provider = CachingEOPProvider::new(
            &dest_path,
            EOPType::StandardBulletinA,
            365 * 86400,
            false,
            true,
            EOPExtrapolation::Hold,
        )
        .unwrap();

        let original_len = provider.len();

        // Refresh should succeed (no download needed)
        provider.refresh().unwrap();

        // Length should be unchanged
        assert_eq!(provider.len(), original_len);
    }

    #[test]
    #[cfg_attr(not(feature = "ci"), ignore)]
    fn test_eop_provider_delegation() {
        // Test that EarthOrientationProvider methods are properly delegated
        let dir = tempdir().unwrap();
        let src_path = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("test_assets")
            .join("finals.all.iau2000.txt");
        let dest_path = dir.path().join("test_eop_delegation.txt");

        fs::copy(&src_path, &dest_path).unwrap();

        let provider = CachingEOPProvider::new(
            &dest_path,
            EOPType::StandardBulletinA,
            365 * 86400,
            false,
            true,
            EOPExtrapolation::Hold,
        )
        .unwrap();

        // Test basic properties
        assert!(provider.is_initialized());
        assert_eq!(provider.eop_type(), EOPType::StandardBulletinA);
        assert_eq!(provider.extrapolation(), EOPExtrapolation::Hold);
        assert!(provider.interpolation());
        assert_eq!(provider.mjd_min(), 41684.0);
        // The max mjd may change as the packaged EOP file is updated
        assert!(provider.mjd_max() >= 60672.0);

        // Test data retrieval
        let ut1_utc = provider.get_ut1_utc(59569.0).unwrap();
        assert_eq!(ut1_utc, -0.1079939);

        let (pm_x, pm_y) = provider.get_pm(59569.0).unwrap();
        assert!(pm_x > 0.0);
        assert!(pm_y > 0.0);

        let (dx, dy) = provider.get_dxdy(59569.0).unwrap();
        assert!(dx != 0.0 || dy != 0.0);

        let lod = provider.get_lod(59569.0).unwrap();
        assert!(lod != 0.0);
    }
}
