/*!
 * Defines crate-wide space weather loading functionality
 */

use once_cell::sync::Lazy;
use std::cell::RefCell;
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

thread_local! {
    static THREAD_LOCAL_SW: RefCell<Option<Arc<dyn SpaceWeatherProvider + Sync + Send>>> = const { RefCell::new(None) };
}

/// Wrapper that implements `SpaceWeatherProvider` by delegating to an inner `Arc`.
///
/// This is used internally to bridge `Arc<dyn SpaceWeatherProvider>` into a
/// `Box<dyn SpaceWeatherProvider>` context (e.g., the global provider).
struct ArcSWWrapper(Arc<dyn SpaceWeatherProvider + Sync + Send>);

impl SpaceWeatherProvider for ArcSWWrapper {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn sw_type(&self) -> SpaceWeatherType {
        self.0.sw_type()
    }

    fn is_initialized(&self) -> bool {
        self.0.is_initialized()
    }

    fn extrapolation(&self) -> SpaceWeatherExtrapolation {
        self.0.extrapolation()
    }

    fn mjd_min(&self) -> f64 {
        self.0.mjd_min()
    }

    fn mjd_max(&self) -> f64 {
        self.0.mjd_max()
    }

    fn mjd_last_observed(&self) -> f64 {
        self.0.mjd_last_observed()
    }

    fn mjd_last_daily_predicted(&self) -> f64 {
        self.0.mjd_last_daily_predicted()
    }

    fn mjd_last_monthly_predicted(&self) -> f64 {
        self.0.mjd_last_monthly_predicted()
    }

    fn get_kp(&self, mjd: f64) -> Result<f64, BraheError> {
        self.0.get_kp(mjd)
    }

    fn get_kp_all(&self, mjd: f64) -> Result<[f64; 8], BraheError> {
        self.0.get_kp_all(mjd)
    }

    fn get_kp_daily(&self, mjd: f64) -> Result<f64, BraheError> {
        self.0.get_kp_daily(mjd)
    }

    fn get_ap(&self, mjd: f64) -> Result<f64, BraheError> {
        self.0.get_ap(mjd)
    }

    fn get_ap_all(&self, mjd: f64) -> Result<[f64; 8], BraheError> {
        self.0.get_ap_all(mjd)
    }

    fn get_ap_daily(&self, mjd: f64) -> Result<f64, BraheError> {
        self.0.get_ap_daily(mjd)
    }

    fn get_f107_observed(&self, mjd: f64) -> Result<f64, BraheError> {
        self.0.get_f107_observed(mjd)
    }

    fn get_f107_adjusted(&self, mjd: f64) -> Result<f64, BraheError> {
        self.0.get_f107_adjusted(mjd)
    }

    fn get_f107_obs_avg81(&self, mjd: f64) -> Result<f64, BraheError> {
        self.0.get_f107_obs_avg81(mjd)
    }

    fn get_f107_adj_avg81(&self, mjd: f64) -> Result<f64, BraheError> {
        self.0.get_f107_adj_avg81(mjd)
    }

    fn get_sunspot_number(&self, mjd: f64) -> Result<u32, BraheError> {
        self.0.get_sunspot_number(mjd)
    }

    fn get_last_kp(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.0.get_last_kp(mjd, n)
    }

    fn get_last_ap(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.0.get_last_ap(mjd, n)
    }

    fn get_last_daily_kp(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.0.get_last_daily_kp(mjd, n)
    }

    fn get_last_daily_ap(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.0.get_last_daily_ap(mjd, n)
    }

    fn get_last_f107(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        self.0.get_last_f107(mjd, n)
    }

    fn get_last_kpap_epochs(&self, mjd: f64, n: usize) -> Result<Vec<Epoch>, BraheError> {
        self.0.get_last_kpap_epochs(mjd, n)
    }

    fn get_last_daily_epochs(&self, mjd: f64, n: usize) -> Result<Vec<Epoch>, BraheError> {
        self.0.get_last_daily_epochs(mjd, n)
    }
}

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
    let arc: Arc<dyn SpaceWeatherProvider + Sync + Send> = Arc::new(provider);
    *GLOBAL_SW.write().unwrap() = Box::new(ArcSWWrapper(Arc::clone(&arc)));
    THREAD_LOCAL_SW.with(|tl| {
        *tl.borrow_mut() = Some(arc);
    });
}

/// Set a thread-local space weather provider override.
///
/// When set, all `get_global_*` accessor functions on this thread will use
/// this provider instead of the global one. This is useful for Monte Carlo
/// simulations where each thread needs independent space weather data.
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
/// let sw = StaticSpaceWeatherProvider::from_values(3.0, 15.0, 150.0, 148.0, 100);
/// set_thread_local_space_weather_provider(sw);
///
/// // This thread now uses the thread-local provider
/// assert_eq!(get_global_kp(60000.0).unwrap(), 3.0);
///
/// // Clean up
/// clear_thread_local_space_weather_provider();
/// ```
pub fn set_thread_local_space_weather_provider<T: SpaceWeatherProvider + Sync + Send + 'static>(
    provider: T,
) {
    THREAD_LOCAL_SW.with(|tl| {
        *tl.borrow_mut() = Some(Arc::new(provider));
    });
}

/// Clear the thread-local space weather provider override.
///
/// After calling this, `get_global_*` accessor functions on this thread will
/// fall back to the global provider.
pub fn clear_thread_local_space_weather_provider() {
    THREAD_LOCAL_SW.with(|tl| {
        *tl.borrow_mut() = None;
    });
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
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_kp(mjd);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_kp(mjd)
    })
}

/// Get all eight 3-hourly Kp indices for the day containing the given MJD.
pub fn get_global_kp_all(mjd: f64) -> Result<[f64; 8], BraheError> {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_kp_all(mjd);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_kp_all(mjd)
    })
}

/// Get the daily average Kp index for the given MJD.
pub fn get_global_kp_daily(mjd: f64) -> Result<f64, BraheError> {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_kp_daily(mjd);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_kp_daily(mjd)
    })
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
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_ap(mjd);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_ap(mjd)
    })
}

/// Get all eight 3-hourly Ap indices for the day containing the given MJD.
pub fn get_global_ap_all(mjd: f64) -> Result<[f64; 8], BraheError> {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_ap_all(mjd);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_ap_all(mjd)
    })
}

/// Get the daily average Ap index for the given MJD.
pub fn get_global_ap_daily(mjd: f64) -> Result<f64, BraheError> {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_ap_daily(mjd);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_ap_daily(mjd)
    })
}

/// Get observed F10.7 solar flux for the specified MJD.
///
/// # Arguments
/// - `mjd`: Modified Julian date
///
/// # Returns
/// - F10.7 flux in solar flux units (sfu)
pub fn get_global_f107_observed(mjd: f64) -> Result<f64, BraheError> {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_f107_observed(mjd);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_f107_observed(mjd)
    })
}

/// Get adjusted F10.7 solar flux for the specified MJD.
pub fn get_global_f107_adjusted(mjd: f64) -> Result<f64, BraheError> {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_f107_adjusted(mjd);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_f107_adjusted(mjd)
    })
}

/// Get observed 81-day centered average F10.7 flux.
pub fn get_global_f107_obs_avg81(mjd: f64) -> Result<f64, BraheError> {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_f107_obs_avg81(mjd);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_f107_obs_avg81(mjd)
    })
}

/// Get adjusted 81-day centered average F10.7 flux.
pub fn get_global_f107_adj_avg81(mjd: f64) -> Result<f64, BraheError> {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_f107_adj_avg81(mjd);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_f107_adj_avg81(mjd)
    })
}

/// Get International Sunspot Number for the specified MJD.
pub fn get_global_sunspot_number(mjd: f64) -> Result<u32, BraheError> {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_sunspot_number(mjd);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_sunspot_number(mjd)
    })
}

/// Returns initialization state of global space weather provider.
pub fn get_global_sw_initialization() -> bool {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.is_initialized();
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().is_initialized()
    })
}

/// Return length of loaded global space weather data.
pub fn get_global_sw_len() -> usize {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.len();
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().len()
    })
}

/// Returns the type of loaded space weather data.
pub fn get_global_sw_type() -> SpaceWeatherType {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.sw_type();
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().sw_type()
    })
}

/// Return extrapolation setting of loaded space weather provider.
pub fn get_global_sw_extrapolation() -> SpaceWeatherExtrapolation {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.extrapolation();
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().extrapolation()
    })
}

/// Returns the earliest MJD available in the loaded space weather data.
pub fn get_global_sw_mjd_min() -> f64 {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.mjd_min();
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().mjd_min()
    })
}

/// Returns the latest MJD available in the loaded space weather data.
pub fn get_global_sw_mjd_max() -> f64 {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.mjd_max();
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().mjd_max()
    })
}

/// Returns the last MJD with observed data.
pub fn get_global_sw_mjd_last_observed() -> f64 {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.mjd_last_observed();
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().mjd_last_observed()
    })
}

/// Returns the last MJD with daily predicted data.
pub fn get_global_sw_mjd_last_daily_predicted() -> f64 {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.mjd_last_daily_predicted();
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().mjd_last_daily_predicted()
    })
}

/// Returns the last MJD with monthly predicted data.
pub fn get_global_sw_mjd_last_monthly_predicted() -> f64 {
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.mjd_last_monthly_predicted();
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().mjd_last_monthly_predicted()
    })
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
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_last_kp(mjd, n);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_last_kp(mjd, n)
    })
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
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_last_ap(mjd, n);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_last_ap(mjd, n)
    })
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
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_last_daily_kp(mjd, n);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_last_daily_kp(mjd, n)
    })
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
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_last_daily_ap(mjd, n);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_last_daily_ap(mjd, n)
    })
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
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_last_f107(mjd, n);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_last_f107(mjd, n)
    })
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
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_last_kpap_epochs(mjd, n);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_last_kpap_epochs(mjd, n)
    })
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
    THREAD_LOCAL_SW.with(|tl| {
        let borrow = tl.borrow();
        if let Some(ref provider) = *borrow {
            return provider.get_last_daily_epochs(mjd, n);
        }
        drop(borrow);
        GLOBAL_SW.read().unwrap().get_last_daily_epochs(mjd, n)
    })
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
    use crate::utils::testing::setup_global_test_space_weather;

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

        setup_global_test_space_weather();

        assert!(get_global_sw_initialization());
        assert!(get_global_sw_len() > 0);
        assert_eq!(get_global_sw_type(), SpaceWeatherType::CssiSpaceWeather);
    }

    #[test]
    #[serial]
    fn test_get_global_kp() {
        setup_global_test_space_weather();
        let kp = get_global_kp(36114.0).unwrap();
        assert!((0.0..=9.0).contains(&kp));
    }

    #[test]
    #[serial]
    fn test_get_global_ap() {
        setup_global_test_space_weather();
        let ap = get_global_ap_daily(36114.0).unwrap();
        assert!((0.0..=400.0).contains(&ap));
    }

    #[test]
    #[serial]
    fn test_get_global_f107() {
        setup_global_test_space_weather();
        let f107 = get_global_f107_observed(36114.0).unwrap();
        assert!(f107 > 0.0);
    }

    #[test]
    #[serial]
    fn test_get_global_sunspot() {
        setup_global_test_space_weather();
        let isn = get_global_sunspot_number(36114.0).unwrap();
        assert!(isn < 500);
    }

    #[test]
    #[serial]
    fn test_get_global_sw_mjd_last_daily_predicted() {
        setup_global_test_space_weather();
        let mjd_last_daily = get_global_sw_mjd_last_daily_predicted();
        // Should be at least as far as last observed
        assert!(mjd_last_daily >= get_global_sw_mjd_last_observed());
        // Should be a reasonable value (after 2020)
        assert!(mjd_last_daily > 58849.0);
    }

    #[test]
    #[serial]
    fn test_get_global_sw_mjd_last_monthly_predicted() {
        setup_global_test_space_weather();
        let mjd_last_monthly = get_global_sw_mjd_last_monthly_predicted();
        // Should be at least as far as daily predicted
        assert!(mjd_last_monthly >= get_global_sw_mjd_last_daily_predicted());
        // Should be a reasonable value (after 2020)
        assert!(mjd_last_monthly > 58849.0);
    }

    #[test]
    #[serial]
    fn test_get_global_kp_all() {
        setup_global_test_space_weather();
        let kp_all = get_global_kp_all(36114.0).unwrap();
        assert_eq!(kp_all.len(), 8);
        for kp in kp_all.iter() {
            assert!((0.0..=9.0).contains(kp));
        }
    }

    #[test]
    #[serial]
    fn test_get_global_kp_daily() {
        setup_global_test_space_weather();
        let kp_daily = get_global_kp_daily(36114.0).unwrap();
        assert!((0.0..=9.0).contains(&kp_daily));
    }

    #[test]
    #[serial]
    fn test_get_global_ap_3hourly() {
        setup_global_test_space_weather();
        let ap = get_global_ap(36114.0).unwrap();
        assert!(ap >= 0.0);
    }

    #[test]
    #[serial]
    fn test_get_global_ap_all() {
        setup_global_test_space_weather();
        let ap_all = get_global_ap_all(36114.0).unwrap();
        assert_eq!(ap_all.len(), 8);
        for ap in ap_all.iter() {
            assert!(*ap >= 0.0);
        }
    }

    #[test]
    #[serial]
    fn test_get_global_f107_adjusted() {
        setup_global_test_space_weather();
        let f107_adj = get_global_f107_adjusted(36114.0).unwrap();
        assert!(f107_adj >= 0.0);
    }

    #[test]
    #[serial]
    fn test_get_global_f107_obs_avg81() {
        setup_global_test_space_weather();
        let f107_avg = get_global_f107_obs_avg81(60000.0).unwrap();
        assert!(f107_avg > 0.0);
    }

    #[test]
    #[serial]
    fn test_get_global_f107_adj_avg81() {
        setup_global_test_space_weather();
        let f107_avg = get_global_f107_adj_avg81(60000.0).unwrap();
        assert!(f107_avg > 0.0);
    }

    #[test]
    #[serial]
    fn test_get_global_sw_extrapolation() {
        setup_global_test_space_weather();
        let extrapolation = get_global_sw_extrapolation();
        assert_eq!(extrapolation, SpaceWeatherExtrapolation::Hold);
    }

    #[test]
    #[serial]
    fn test_get_global_sw_mjd_min() {
        setup_global_test_space_weather();
        let mjd_min = get_global_sw_mjd_min();
        // First data point is 1957-10-01 (MJD 36112)
        assert_eq!(mjd_min, 36112.0);
    }

    #[test]
    #[serial]
    fn test_get_global_sw_mjd_max() {
        setup_global_test_space_weather();
        let mjd_max = get_global_sw_mjd_max();
        // Should have data through recent dates
        assert!(mjd_max > 60000.0);
    }

    #[test]
    #[serial]
    fn test_get_global_sw_mjd_last_observed() {
        setup_global_test_space_weather();
        let mjd_last_obs = get_global_sw_mjd_last_observed();
        // Should have recent observed data
        assert!(mjd_last_obs > 60000.0);
    }

    #[test]
    #[serial]
    fn test_get_global_last_kp() {
        setup_global_test_space_weather();
        let kp_values = get_global_last_kp(60000.0, 5).unwrap();
        assert_eq!(kp_values.len(), 5);
        for kp in &kp_values {
            assert!((0.0..=9.0).contains(kp));
        }
    }

    #[test]
    #[serial]
    fn test_get_global_last_ap() {
        setup_global_test_space_weather();
        let ap_values = get_global_last_ap(60000.0, 5).unwrap();
        assert_eq!(ap_values.len(), 5);
        for ap in &ap_values {
            assert!(*ap >= 0.0);
        }
    }

    #[test]
    #[serial]
    fn test_get_global_last_daily_kp() {
        setup_global_test_space_weather();
        let daily_kp = get_global_last_daily_kp(60000.0, 3).unwrap();
        assert_eq!(daily_kp.len(), 3);
        for kp in &daily_kp {
            assert!((0.0..=9.0).contains(kp));
        }
    }

    #[test]
    #[serial]
    fn test_get_global_last_daily_ap() {
        setup_global_test_space_weather();
        let daily_ap = get_global_last_daily_ap(60000.0, 3).unwrap();
        assert_eq!(daily_ap.len(), 3);
        for ap in &daily_ap {
            assert!(*ap >= 0.0);
        }
    }

    #[test]
    #[serial]
    fn test_get_global_last_f107() {
        setup_global_test_space_weather();
        let f107_values = get_global_last_f107(60000.0, 3).unwrap();
        assert_eq!(f107_values.len(), 3);
        for f107 in &f107_values {
            assert!(*f107 > 0.0);
        }
    }

    #[test]
    #[serial]
    fn test_get_global_last_kpap_epochs() {
        setup_global_test_space_weather();
        let epochs = get_global_last_kpap_epochs(60000.0, 5).unwrap();
        assert_eq!(epochs.len(), 5);
        // Verify epochs are in ascending order
        for i in 0..epochs.len() - 1 {
            assert!(epochs[i].mjd() < epochs[i + 1].mjd());
        }
    }

    #[test]
    #[serial]
    fn test_get_global_last_daily_epochs() {
        setup_global_test_space_weather();
        let epochs = get_global_last_daily_epochs(60000.0, 3).unwrap();
        assert_eq!(epochs.len(), 3);
        // Verify epochs are in ascending order
        for i in 0..epochs.len() - 1 {
            assert!(epochs[i].mjd() < epochs[i + 1].mjd());
        }
    }

    #[test]
    #[serial]
    fn test_initialize_sw() {
        clear_test_global_sw();
        assert!(!get_global_sw_initialization());

        // Initialize with default settings
        initialize_sw().unwrap();

        assert!(get_global_sw_initialization());
        assert_eq!(get_global_sw_type(), SpaceWeatherType::CssiSpaceWeather);
        assert!(get_global_sw_len() > 0);
    }

    #[test]
    #[serial]
    fn test_thread_local_sw_override() {
        // Set global to file-based provider
        setup_global_test_space_weather();

        // Set thread-local to a static provider with known values
        let sw = StaticSpaceWeatherProvider::from_values(5.0, 25.0, 200.0, 198.0, 150);
        set_thread_local_space_weather_provider(sw);

        // Thread-local should override
        assert_eq!(get_global_kp(60000.0).unwrap(), 5.0);
        assert_eq!(get_global_ap(60000.0).unwrap(), 25.0);
        assert_eq!(get_global_ap_daily(60000.0).unwrap(), 25.0);
        assert_eq!(get_global_f107_observed(60000.0).unwrap(), 200.0);
        assert_eq!(get_global_f107_adjusted(60000.0).unwrap(), 198.0);
        assert_eq!(get_global_sunspot_number(60000.0).unwrap(), 150);
        assert_eq!(get_global_sw_type(), SpaceWeatherType::Static);
        assert_eq!(get_global_sw_len(), 1);

        // Clean up
        clear_thread_local_space_weather_provider();
    }

    #[test]
    #[serial]
    fn test_thread_local_sw_clear_falls_back_to_global() {
        // Set global to file-based provider
        setup_global_test_space_weather();

        // Set thread-local to a static provider
        let sw = StaticSpaceWeatherProvider::from_values(5.0, 25.0, 200.0, 198.0, 150);
        set_thread_local_space_weather_provider(sw);

        // Verify thread-local is active
        assert_eq!(get_global_kp(60000.0).unwrap(), 5.0);

        // Clear thread-local
        clear_thread_local_space_weather_provider();

        // Should fall back to global (file-based) provider
        assert_eq!(get_global_sw_type(), SpaceWeatherType::CssiSpaceWeather);
        // Kp from file should NOT be 5.0 (our static value)
        let kp = get_global_kp(36114.0).unwrap();
        assert!((0.0..=9.0).contains(&kp));
    }

    #[test]
    #[serial]
    fn test_thread_local_sw_kp_all_and_daily() {
        let sw = StaticSpaceWeatherProvider::from_values(4.0, 20.0, 170.0, 168.0, 120);
        set_thread_local_space_weather_provider(sw);

        let kp_all = get_global_kp_all(60000.0).unwrap();
        assert_eq!(kp_all, [4.0; 8]);

        let kp_daily = get_global_kp_daily(60000.0).unwrap();
        assert_eq!(kp_daily, 4.0);

        clear_thread_local_space_weather_provider();
    }

    #[test]
    #[serial]
    fn test_thread_local_sw_ap_all() {
        let sw = StaticSpaceWeatherProvider::from_values(4.0, 20.0, 170.0, 168.0, 120);
        set_thread_local_space_weather_provider(sw);

        let ap_all = get_global_ap_all(60000.0).unwrap();
        assert_eq!(ap_all, [20.0; 8]);

        clear_thread_local_space_weather_provider();
    }

    #[test]
    #[serial]
    fn test_thread_local_sw_f107_averages() {
        let sw = StaticSpaceWeatherProvider::from_values(4.0, 20.0, 170.0, 168.0, 120);
        set_thread_local_space_weather_provider(sw);

        let f107_obs_avg = get_global_f107_obs_avg81(60000.0).unwrap();
        assert_eq!(f107_obs_avg, 170.0);

        let f107_adj_avg = get_global_f107_adj_avg81(60000.0).unwrap();
        assert_eq!(f107_adj_avg, 168.0);

        clear_thread_local_space_weather_provider();
    }

    #[test]
    #[serial]
    fn test_thread_local_sw_metadata() {
        let sw = StaticSpaceWeatherProvider::from_values(4.0, 20.0, 170.0, 168.0, 120);
        set_thread_local_space_weather_provider(sw);

        assert!(get_global_sw_initialization());
        assert_eq!(
            get_global_sw_extrapolation(),
            SpaceWeatherExtrapolation::Hold
        );
        assert_eq!(get_global_sw_mjd_min(), 0.0);
        assert_eq!(get_global_sw_mjd_max(), f64::MAX);
        assert_eq!(get_global_sw_mjd_last_observed(), f64::MAX);
        assert_eq!(get_global_sw_mjd_last_daily_predicted(), f64::MAX);
        assert_eq!(get_global_sw_mjd_last_monthly_predicted(), f64::MAX);

        clear_thread_local_space_weather_provider();
    }

    #[test]
    #[serial]
    fn test_thread_local_sw_last_values() {
        let sw = StaticSpaceWeatherProvider::from_values(4.0, 20.0, 170.0, 168.0, 120);
        set_thread_local_space_weather_provider(sw);

        let last_kp = get_global_last_kp(60000.0, 3).unwrap();
        assert_eq!(last_kp.len(), 3);
        for kp in &last_kp {
            assert_eq!(*kp, 4.0);
        }

        let last_ap = get_global_last_ap(60000.0, 3).unwrap();
        assert_eq!(last_ap.len(), 3);
        for ap in &last_ap {
            assert_eq!(*ap, 20.0);
        }

        let last_daily_kp = get_global_last_daily_kp(60000.0, 3).unwrap();
        assert_eq!(last_daily_kp.len(), 3);

        let last_daily_ap = get_global_last_daily_ap(60000.0, 3).unwrap();
        assert_eq!(last_daily_ap.len(), 3);

        let last_f107 = get_global_last_f107(60000.0, 3).unwrap();
        assert_eq!(last_f107.len(), 3);

        clear_thread_local_space_weather_provider();
    }

    #[test]
    #[serial]
    fn test_thread_local_sw_last_epochs() {
        let sw = StaticSpaceWeatherProvider::from_values(4.0, 20.0, 170.0, 168.0, 120);
        set_thread_local_space_weather_provider(sw);

        let kpap_epochs = get_global_last_kpap_epochs(60000.0, 3).unwrap();
        assert_eq!(kpap_epochs.len(), 3);
        for i in 0..kpap_epochs.len() - 1 {
            assert!(kpap_epochs[i].mjd() < kpap_epochs[i + 1].mjd());
        }

        let daily_epochs = get_global_last_daily_epochs(60000.0, 3).unwrap();
        assert_eq!(daily_epochs.len(), 3);
        for i in 0..daily_epochs.len() - 1 {
            assert!(daily_epochs[i].mjd() < daily_epochs[i + 1].mjd());
        }

        clear_thread_local_space_weather_provider();
    }
}
