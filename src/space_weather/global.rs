/*!
 * Defines crate-wide space weather loading functionality
 */

use once_cell::sync::Lazy;
use std::sync::{Arc, RwLock};

use crate::space_weather::provider::SpaceWeatherProvider;
use crate::space_weather::static_provider::StaticSpaceWeatherProvider;
use crate::space_weather::types::{SpaceWeatherExtrapolation, SpaceWeatherType};
use crate::time::Epoch;
use crate::utils::BraheError;

#[cfg(test)]
use serial_test::serial;

static GLOBAL_SW: Lazy<Arc<RwLock<Box<dyn SpaceWeatherProvider + Sync + Send>>>> =
    Lazy::new(|| Arc::new(RwLock::new(Box::new(StaticSpaceWeatherProvider::new()))));

/// Set the crate-wide static space weather data provider.
///
/// This function should be called before any other function in the crate which
/// accesses the global space weather data. If this function is not called, the
/// crate-wide provider will not be initialized.
///
/// # Arguments
///
/// - `provider`: Object which implements the `SpaceWeatherProvider` trait
///
/// # Examples
///
/// ```
/// use brahe::space_weather::*;
///
/// // Initialize from StaticSpaceWeatherProvider
/// let sw = StaticSpaceWeatherProvider::from_zero();
/// set_global_space_weather_provider(sw);
///
/// // Initialize from FileSpaceWeatherProvider
/// let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
/// set_global_space_weather_provider(sw);
/// ```
pub fn set_global_space_weather_provider<T: SpaceWeatherProvider + Sync + Send + 'static>(
    provider: T,
) {
    *GLOBAL_SW.write().unwrap() = Box::new(provider);
}

/// Get Kp index for the specified MJD from the global provider.
///
/// Uses the fractional MJD to determine which 3-hour interval to return.
///
/// # Arguments
/// - `mjd`: Modified Julian date
///
/// # Returns
/// - Kp index (0.0-9.0 scale)
pub fn get_global_kp(mjd: f64) -> Result<f64, BraheError> {
    GLOBAL_SW.read().unwrap().get_kp(mjd)
}

/// Get all eight 3-hourly Kp indices for the day containing the given MJD.
pub fn get_global_kp_all(mjd: f64) -> Result<[f64; 8], BraheError> {
    GLOBAL_SW.read().unwrap().get_kp_all(mjd)
}

/// Get the daily average Kp index for the given MJD.
pub fn get_global_kp_daily(mjd: f64) -> Result<f64, BraheError> {
    GLOBAL_SW.read().unwrap().get_kp_daily(mjd)
}

/// Get Ap index for the specified MJD from the global provider.
///
/// Uses the fractional MJD to determine which 3-hour interval to return.
///
/// # Arguments
/// - `mjd`: Modified Julian date
///
/// # Returns
/// - Ap index
pub fn get_global_ap(mjd: f64) -> Result<f64, BraheError> {
    GLOBAL_SW.read().unwrap().get_ap(mjd)
}

/// Get all eight 3-hourly Ap indices for the day containing the given MJD.
pub fn get_global_ap_all(mjd: f64) -> Result<[f64; 8], BraheError> {
    GLOBAL_SW.read().unwrap().get_ap_all(mjd)
}

/// Get the daily average Ap index for the given MJD.
pub fn get_global_ap_daily(mjd: f64) -> Result<f64, BraheError> {
    GLOBAL_SW.read().unwrap().get_ap_daily(mjd)
}

/// Get observed F10.7 solar flux for the specified MJD.
///
/// # Arguments
/// - `mjd`: Modified Julian date
///
/// # Returns
/// - F10.7 flux in solar flux units (sfu)
pub fn get_global_f107_observed(mjd: f64) -> Result<f64, BraheError> {
    GLOBAL_SW.read().unwrap().get_f107_observed(mjd)
}

/// Get adjusted F10.7 solar flux for the specified MJD.
pub fn get_global_f107_adjusted(mjd: f64) -> Result<f64, BraheError> {
    GLOBAL_SW.read().unwrap().get_f107_adjusted(mjd)
}

/// Get observed 81-day centered average F10.7 flux.
pub fn get_global_f107_obs_avg81(mjd: f64) -> Result<f64, BraheError> {
    GLOBAL_SW.read().unwrap().get_f107_obs_avg81(mjd)
}

/// Get adjusted 81-day centered average F10.7 flux.
pub fn get_global_f107_adj_avg81(mjd: f64) -> Result<f64, BraheError> {
    GLOBAL_SW.read().unwrap().get_f107_adj_avg81(mjd)
}

/// Get International Sunspot Number for the specified MJD.
pub fn get_global_sunspot_number(mjd: f64) -> Result<u32, BraheError> {
    GLOBAL_SW.read().unwrap().get_sunspot_number(mjd)
}

/// Returns initialization state of global space weather provider.
pub fn get_global_sw_initialization() -> bool {
    GLOBAL_SW.read().unwrap().is_initialized()
}

/// Return length of loaded global space weather data.
pub fn get_global_sw_len() -> usize {
    GLOBAL_SW.read().unwrap().len()
}

/// Returns the type of loaded space weather data.
pub fn get_global_sw_type() -> SpaceWeatherType {
    GLOBAL_SW.read().unwrap().sw_type()
}

/// Return extrapolation setting of loaded space weather provider.
pub fn get_global_sw_extrapolation() -> SpaceWeatherExtrapolation {
    GLOBAL_SW.read().unwrap().extrapolation()
}

/// Returns the earliest MJD available in the loaded space weather data.
pub fn get_global_sw_mjd_min() -> f64 {
    GLOBAL_SW.read().unwrap().mjd_min()
}

/// Returns the latest MJD available in the loaded space weather data.
pub fn get_global_sw_mjd_max() -> f64 {
    GLOBAL_SW.read().unwrap().mjd_max()
}

/// Returns the last MJD with observed data.
pub fn get_global_sw_mjd_last_observed() -> f64 {
    GLOBAL_SW.read().unwrap().mjd_last_observed()
}

/// Returns the last MJD with daily predicted data.
pub fn get_global_sw_mjd_last_daily_predicted() -> f64 {
    GLOBAL_SW.read().unwrap().mjd_last_daily_predicted()
}

/// Returns the last MJD with monthly predicted data.
pub fn get_global_sw_mjd_last_monthly_predicted() -> f64 {
    GLOBAL_SW.read().unwrap().mjd_last_monthly_predicted()
}

/// Get the last N 3-hourly Kp values from the global provider.
///
/// Returns a vector with the oldest value first and newest last.
///
/// # Arguments
/// - `mjd`: Modified Julian Date (end point)
/// - `n`: Number of 3-hourly values to return
///
/// # Returns
/// - Vector of Kp indices (0.0-9.0 scale), oldest first
pub fn get_global_last_kp(mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
    GLOBAL_SW.read().unwrap().get_last_kp(mjd, n)
}

/// Get the last N 3-hourly Ap values from the global provider.
///
/// Returns a vector with the oldest value first and newest last.
///
/// # Arguments
/// - `mjd`: Modified Julian Date (end point)
/// - `n`: Number of 3-hourly values to return
///
/// # Returns
/// - Vector of Ap indices, oldest first
pub fn get_global_last_ap(mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
    GLOBAL_SW.read().unwrap().get_last_ap(mjd, n)
}

/// Get the last N daily average Kp values from the global provider.
///
/// Returns a vector with the oldest value first and newest last.
///
/// # Arguments
/// - `mjd`: Modified Julian Date (end point)
/// - `n`: Number of daily values to return
///
/// # Returns
/// - Vector of daily average Kp indices, oldest first
pub fn get_global_last_daily_kp(mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
    GLOBAL_SW.read().unwrap().get_last_daily_kp(mjd, n)
}

/// Get the last N daily average Ap values from the global provider.
///
/// Returns a vector with the oldest value first and newest last.
///
/// # Arguments
/// - `mjd`: Modified Julian Date (end point)
/// - `n`: Number of daily values to return
///
/// # Returns
/// - Vector of daily average Ap indices, oldest first
pub fn get_global_last_daily_ap(mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
    GLOBAL_SW.read().unwrap().get_last_daily_ap(mjd, n)
}

/// Get the last N daily observed F10.7 values from the global provider.
///
/// Returns a vector with the oldest value first and newest last.
///
/// # Arguments
/// - `mjd`: Modified Julian Date (end point)
/// - `n`: Number of daily values to return
///
/// # Returns
/// - Vector of F10.7 values in sfu, oldest first
pub fn get_global_last_f107(mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
    GLOBAL_SW.read().unwrap().get_last_f107(mjd, n)
}

/// Get the epochs for the last N 3-hourly Kp/Ap intervals from the global provider.
///
/// Returns a vector with the oldest epoch first and newest last.
/// Each epoch is at the start of a 3-hour UT interval.
///
/// # Arguments
/// - `mjd`: Modified Julian Date (end point)
/// - `n`: Number of 3-hourly epochs to return
///
/// # Returns
/// - Vector of Epoch objects, oldest first
pub fn get_global_last_kpap_epochs(mjd: f64, n: usize) -> Result<Vec<Epoch>, BraheError> {
    GLOBAL_SW.read().unwrap().get_last_kpap_epochs(mjd, n)
}

/// Get the epochs for the last N daily values from the global provider.
///
/// Returns a vector with the oldest epoch first and newest last.
/// Each epoch is at 00:00 UT for the given day.
///
/// # Arguments
/// - `mjd`: Modified Julian Date (end point)
/// - `n`: Number of daily epochs to return
///
/// # Returns
/// - Vector of Epoch objects, oldest first
pub fn get_global_last_daily_epochs(mjd: f64, n: usize) -> Result<Vec<Epoch>, BraheError> {
    GLOBAL_SW.read().unwrap().get_last_daily_epochs(mjd, n)
}

/// Initialize the global space weather provider with recommended default settings.
///
/// This convenience function creates a `CachingSpaceWeatherProvider` with sensible
/// defaults and sets it as the global provider. The provider will:
/// - Automatically download/update space weather files when older than 7 days
/// - Use the default cache location (~/.cache/brahe/sw19571001.txt)
/// - Hold the last known value when extrapolating beyond available data
///
/// # Returns
///
/// - `Result<(), BraheError>`: Ok if initialization succeeded, Error if failed
///
/// # Examples
///
/// ```no_run
/// use brahe::space_weather::initialize_sw;
///
/// // Initialize with recommended defaults
/// initialize_sw().unwrap();
///
/// // Now you can access space weather data
/// ```
pub fn initialize_sw() -> Result<(), BraheError> {
    use crate::space_weather::caching_provider::CachingSpaceWeatherProvider;

    let provider = CachingSpaceWeatherProvider::new(
        None,      // Use default cache location
        7 * 86400, // 7 days in seconds
        false,     // auto_refresh disabled by default
        SpaceWeatherExtrapolation::Hold,
    )?;

    set_global_space_weather_provider(provider);
    Ok(())
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
#[serial]
mod tests {
    use super::*;
    use crate::space_weather::file_provider::FileSpaceWeatherProvider;

    fn setup_test_global_sw() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        set_global_space_weather_provider(sw);
    }

    fn clear_test_global_sw() {
        set_global_space_weather_provider(StaticSpaceWeatherProvider::new());
    }

    #[test]
    #[serial]
    fn test_set_global_sw_from_zero() {
        clear_test_global_sw();
        assert!(!get_global_sw_initialization());

        let sw = StaticSpaceWeatherProvider::from_zero();
        set_global_space_weather_provider(sw);

        assert!(get_global_sw_initialization());
        assert_eq!(get_global_sw_len(), 1);
        assert_eq!(get_global_sw_type(), SpaceWeatherType::Static);
    }

    #[test]
    #[serial]
    fn test_set_global_sw_from_file() {
        clear_test_global_sw();
        assert!(!get_global_sw_initialization());

        setup_test_global_sw();

        assert!(get_global_sw_initialization());
        assert!(get_global_sw_len() > 0);
        assert_eq!(get_global_sw_type(), SpaceWeatherType::CssiSpaceWeather);
    }

    #[test]
    #[serial]
    fn test_get_global_kp() {
        setup_test_global_sw();
        let kp = get_global_kp(36114.0).unwrap();
        assert!((0.0..=9.0).contains(&kp));
    }

    #[test]
    #[serial]
    fn test_get_global_ap() {
        setup_test_global_sw();
        let ap = get_global_ap_daily(36114.0).unwrap();
        assert!((0.0..=400.0).contains(&ap));
    }

    #[test]
    #[serial]
    fn test_get_global_f107() {
        setup_test_global_sw();
        let f107 = get_global_f107_observed(36114.0).unwrap();
        assert!(f107 > 0.0);
    }

    #[test]
    #[serial]
    fn test_get_global_sunspot() {
        setup_test_global_sw();
        let isn = get_global_sunspot_number(36114.0).unwrap();
        assert!(isn < 500);
    }
}
