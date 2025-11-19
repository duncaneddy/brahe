/*!
Defines the SpaceWeatherProvider trait
*/

use crate::space_weather::types::{SpaceWeatherExtrapolation, SpaceWeatherType};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

/// `SpaceWeatherProvider` is a trait for objects that provide space weather data.
///
/// This trait defines a common interface for all space weather providers. A space weather provider
/// is an object that can provide geomagnetic indices (Kp, Ap) and solar activity data (F10.7 flux,
/// sunspot number).
///
/// Implementations of this trait are expected to provide specific methods for retrieving these parameters,
/// such as `get_kp`, `get_ap`, `get_f107_observed`, etc. These methods should return the requested
/// parameter for a given Modified Julian Date (MJD).
///
/// This trait can be extended to implement custom space weather providers.
///
/// # Example
///
/// ```ignore
/// use brahe::space_weather::SpaceWeatherProvider;
///
/// struct MySpaceWeatherProvider;
///
/// impl SpaceWeatherProvider for MySpaceWeatherProvider {
///     // Implement the methods here
/// }
/// ```
pub trait SpaceWeatherProvider: Send + Sync {
    /// Returns the number of space weather data entries loaded in the provider.
    fn len(&self) -> usize;

    /// Returns true if the provider contains no space weather data entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the type of space weather data loaded (CssiSpaceWeather or Static).
    fn sw_type(&self) -> SpaceWeatherType;

    /// Returns true if the provider has been initialized with valid space weather data.
    fn is_initialized(&self) -> bool;

    /// Returns the extrapolation behavior (Zero, Hold, or Error) for out-of-range requests.
    fn extrapolation(&self) -> SpaceWeatherExtrapolation;

    /// Returns the minimum Modified Julian Date covered by the loaded space weather data.
    fn mjd_min(&self) -> f64;

    /// Returns the maximum Modified Julian Date covered by the loaded space weather data.
    fn mjd_max(&self) -> f64;

    /// Returns the last MJD with observed (historical) space weather data.
    fn mjd_last_observed(&self) -> f64;

    /// Returns the last MJD with daily predicted space weather data.
    fn mjd_last_daily_predicted(&self) -> f64;

    /// Returns the last MJD with monthly predicted space weather data.
    fn mjd_last_monthly_predicted(&self) -> f64;

    /// Get the Kp index for a specific 3-hour interval at the given MJD.
    ///
    /// The MJD's fractional day determines which of the 8 daily intervals is returned:
    /// - Index 0: 00:00-03:00 UT
    /// - Index 1: 03:00-06:00 UT
    /// - Index 2: 06:00-09:00 UT
    /// - Index 3: 09:00-12:00 UT
    /// - Index 4: 12:00-15:00 UT
    /// - Index 5: 15:00-18:00 UT
    /// - Index 6: 18:00-21:00 UT
    /// - Index 7: 21:00-24:00 UT
    ///
    /// # Returns
    /// Kp index on the 0.0-9.0 scale
    fn get_kp(&self, mjd: f64) -> Result<f64, BraheError>;

    /// Get all eight 3-hourly Kp indices for the day containing the given MJD.
    ///
    /// # Returns
    /// Array of 8 Kp indices on the 0.0-9.0 scale
    fn get_kp_all(&self, mjd: f64) -> Result<[f64; 8], BraheError>;

    /// Get the daily average Kp index (sum of 8 intervals / 8) for the given MJD.
    ///
    /// # Returns
    /// Daily average Kp index
    fn get_kp_daily(&self, mjd: f64) -> Result<f64, BraheError>;

    /// Get the Ap index for a specific 3-hour interval at the given MJD.
    ///
    /// The MJD's fractional day determines which of the 8 daily intervals is returned.
    ///
    /// # Returns
    /// Ap index (integer value 0-400)
    fn get_ap(&self, mjd: f64) -> Result<f64, BraheError>;

    /// Get all eight 3-hourly Ap indices for the day containing the given MJD.
    ///
    /// # Returns
    /// Array of 8 Ap indices
    fn get_ap_all(&self, mjd: f64) -> Result<[f64; 8], BraheError>;

    /// Get the daily average Ap index for the given MJD.
    ///
    /// # Returns
    /// Daily average Ap index
    fn get_ap_daily(&self, mjd: f64) -> Result<f64, BraheError>;

    /// Get the observed 10.7 cm solar radio flux for the given MJD.
    ///
    /// # Returns
    /// F10.7 flux in solar flux units (sfu)
    fn get_f107_observed(&self, mjd: f64) -> Result<f64, BraheError>;

    /// Get the adjusted 10.7 cm solar radio flux for the given MJD.
    ///
    /// The adjusted value normalizes the observed flux to 1 AU.
    ///
    /// # Returns
    /// Adjusted F10.7 flux in solar flux units (sfu)
    fn get_f107_adjusted(&self, mjd: f64) -> Result<f64, BraheError>;

    /// Get the observed 81-day centered average F10.7 flux for the given MJD.
    ///
    /// # Returns
    /// 81-day centered average F10.7 in sfu
    fn get_f107_obs_avg81(&self, mjd: f64) -> Result<f64, BraheError>;

    /// Get the adjusted 81-day centered average F10.7 flux for the given MJD.
    ///
    /// # Returns
    /// 81-day centered average adjusted F10.7 in sfu
    fn get_f107_adj_avg81(&self, mjd: f64) -> Result<f64, BraheError>;

    /// Get the International Sunspot Number for the given MJD.
    ///
    /// # Returns
    /// International Sunspot Number
    fn get_sunspot_number(&self, mjd: f64) -> Result<u32, BraheError>;

    /// Get the last N 3-hourly Kp values ending at or before the given MJD.
    ///
    /// Returns a vector with the oldest value first and newest last.
    /// Each value corresponds to one 3-hour interval.
    ///
    /// # Arguments
    /// - `mjd`: Modified Julian Date (end point)
    /// - `n`: Number of 3-hourly values to return
    ///
    /// # Returns
    /// Vector of Kp indices (0.0-9.0 scale), oldest first
    fn get_last_kp(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError>;

    /// Get the last N 3-hourly Ap values ending at or before the given MJD.
    ///
    /// Returns a vector with the oldest value first and newest last.
    /// Each value corresponds to one 3-hour interval.
    ///
    /// # Arguments
    /// - `mjd`: Modified Julian Date (end point)
    /// - `n`: Number of 3-hourly values to return
    ///
    /// # Returns
    /// Vector of Ap indices, oldest first
    fn get_last_ap(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError>;

    /// Get the last N daily average Kp values ending at or before the given MJD.
    ///
    /// Returns a vector with the oldest value first and newest last.
    /// Each value is the daily average (kp_sum / 8).
    ///
    /// # Arguments
    /// - `mjd`: Modified Julian Date (end point)
    /// - `n`: Number of daily values to return
    ///
    /// # Returns
    /// Vector of daily average Kp indices, oldest first
    fn get_last_daily_kp(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError>;

    /// Get the last N daily average Ap values ending at or before the given MJD.
    ///
    /// Returns a vector with the oldest value first and newest last.
    /// Each value is the daily average Ap.
    ///
    /// # Arguments
    /// - `mjd`: Modified Julian Date (end point)
    /// - `n`: Number of daily values to return
    ///
    /// # Returns
    /// Vector of daily average Ap indices, oldest first
    fn get_last_daily_ap(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError>;

    /// Get the last N daily observed F10.7 values ending at or before the given MJD.
    ///
    /// Returns a vector with the oldest value first and newest last.
    ///
    /// # Arguments
    /// - `mjd`: Modified Julian Date (end point)
    /// - `n`: Number of daily values to return
    ///
    /// # Returns
    /// Vector of F10.7 values in sfu, oldest first
    fn get_last_f107(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError>;

    /// Get the epochs for the last N 3-hourly Kp/Ap intervals ending at or before the given MJD.
    ///
    /// Returns a vector with the oldest epoch first and newest last.
    /// Each epoch is at the start of a 3-hour UT interval (00:00, 03:00, ..., 21:00).
    ///
    /// # Arguments
    /// - `mjd`: Modified Julian Date (end point)
    /// - `n`: Number of 3-hourly epochs to return
    ///
    /// # Returns
    /// Vector of Epoch objects, oldest first
    fn get_last_kpap_epochs(&self, mjd: f64, n: usize) -> Result<Vec<Epoch>, BraheError>;

    /// Get the epochs for the last N daily values ending at or before the given MJD.
    ///
    /// Returns a vector with the oldest epoch first and newest last.
    /// Each epoch is at 00:00 UT for the given day.
    ///
    /// # Arguments
    /// - `mjd`: Modified Julian Date (end point)
    /// - `n`: Number of daily epochs to return
    ///
    /// # Returns
    /// Vector of Epoch objects, oldest first
    fn get_last_daily_epochs(&self, mjd: f64, n: usize) -> Result<Vec<Epoch>, BraheError>;
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    // Test implementation of SpaceWeatherProvider for testing the default is_empty() method
    struct MockSpaceWeatherProvider {
        len: usize,
    }

    impl SpaceWeatherProvider for MockSpaceWeatherProvider {
        fn len(&self) -> usize {
            self.len
        }

        fn sw_type(&self) -> SpaceWeatherType {
            SpaceWeatherType::Static
        }

        fn is_initialized(&self) -> bool {
            true
        }

        fn extrapolation(&self) -> SpaceWeatherExtrapolation {
            SpaceWeatherExtrapolation::Hold
        }

        fn mjd_min(&self) -> f64 {
            0.0
        }

        fn mjd_max(&self) -> f64 {
            0.0
        }

        fn mjd_last_observed(&self) -> f64 {
            0.0
        }

        fn mjd_last_daily_predicted(&self) -> f64 {
            0.0
        }

        fn mjd_last_monthly_predicted(&self) -> f64 {
            0.0
        }

        fn get_kp(&self, _mjd: f64) -> Result<f64, BraheError> {
            Ok(0.0)
        }

        fn get_kp_all(&self, _mjd: f64) -> Result<[f64; 8], BraheError> {
            Ok([0.0; 8])
        }

        fn get_kp_daily(&self, _mjd: f64) -> Result<f64, BraheError> {
            Ok(0.0)
        }

        fn get_ap(&self, _mjd: f64) -> Result<f64, BraheError> {
            Ok(0.0)
        }

        fn get_ap_all(&self, _mjd: f64) -> Result<[f64; 8], BraheError> {
            Ok([0.0; 8])
        }

        fn get_ap_daily(&self, _mjd: f64) -> Result<f64, BraheError> {
            Ok(0.0)
        }

        fn get_f107_observed(&self, _mjd: f64) -> Result<f64, BraheError> {
            Ok(0.0)
        }

        fn get_f107_adjusted(&self, _mjd: f64) -> Result<f64, BraheError> {
            Ok(0.0)
        }

        fn get_f107_obs_avg81(&self, _mjd: f64) -> Result<f64, BraheError> {
            Ok(0.0)
        }

        fn get_f107_adj_avg81(&self, _mjd: f64) -> Result<f64, BraheError> {
            Ok(0.0)
        }

        fn get_sunspot_number(&self, _mjd: f64) -> Result<u32, BraheError> {
            Ok(0)
        }

        fn get_last_kp(&self, _mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
            Ok(vec![0.0; n])
        }

        fn get_last_ap(&self, _mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
            Ok(vec![0.0; n])
        }

        fn get_last_daily_kp(&self, _mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
            Ok(vec![0.0; n])
        }

        fn get_last_daily_ap(&self, _mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
            Ok(vec![0.0; n])
        }

        fn get_last_f107(&self, _mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
            Ok(vec![0.0; n])
        }

        fn get_last_kpap_epochs(&self, mjd: f64, n: usize) -> Result<Vec<Epoch>, BraheError> {
            let mut epochs = Vec::with_capacity(n);
            for i in 0..n {
                epochs.push(Epoch::from_mjd(
                    mjd - (i as f64 * 0.125),
                    crate::time::TimeSystem::UTC,
                ));
            }
            epochs.reverse();
            Ok(epochs)
        }

        fn get_last_daily_epochs(&self, mjd: f64, n: usize) -> Result<Vec<Epoch>, BraheError> {
            let mut epochs = Vec::with_capacity(n);
            for i in 0..n {
                epochs.push(Epoch::from_mjd(
                    mjd.floor() - i as f64,
                    crate::time::TimeSystem::UTC,
                ));
            }
            epochs.reverse();
            Ok(epochs)
        }
    }

    #[test]
    fn test_space_weather_provider_is_empty_default() {
        // Test that default is_empty() returns true when len() == 0
        let empty_provider = MockSpaceWeatherProvider { len: 0 };
        assert!(empty_provider.is_empty());

        // Test that default is_empty() returns false when len() > 0
        let non_empty_provider = MockSpaceWeatherProvider { len: 10 };
        assert!(!non_empty_provider.is_empty());

        // Test boundary case with len() == 1
        let single_entry_provider = MockSpaceWeatherProvider { len: 1 };
        assert!(!single_entry_provider.is_empty());
    }

    #[test]
    fn test_mock_space_weather_provider_trait_methods() {
        // Test all SpaceWeatherProvider trait methods on MockSpaceWeatherProvider
        let provider = MockSpaceWeatherProvider { len: 5 };

        // Test metadata methods
        assert_eq!(provider.len(), 5);
        assert!(!provider.is_empty());
        assert_eq!(provider.sw_type(), SpaceWeatherType::Static);
        assert!(provider.is_initialized());
        assert_eq!(provider.extrapolation(), SpaceWeatherExtrapolation::Hold);

        // Test MJD range methods
        assert_eq!(provider.mjd_min(), 0.0);
        assert_eq!(provider.mjd_max(), 0.0);
        assert_eq!(provider.mjd_last_observed(), 0.0);
        assert_eq!(provider.mjd_last_daily_predicted(), 0.0);
        assert_eq!(provider.mjd_last_monthly_predicted(), 0.0);

        // Test space weather data retrieval methods
        let test_mjd = 60000.0;
        assert_eq!(provider.get_kp(test_mjd).unwrap(), 0.0);
        assert_eq!(provider.get_kp_all(test_mjd).unwrap(), [0.0; 8]);
        assert_eq!(provider.get_kp_daily(test_mjd).unwrap(), 0.0);
        assert_eq!(provider.get_ap(test_mjd).unwrap(), 0.0);
        assert_eq!(provider.get_ap_all(test_mjd).unwrap(), [0.0; 8]);
        assert_eq!(provider.get_ap_daily(test_mjd).unwrap(), 0.0);
        assert_eq!(provider.get_f107_observed(test_mjd).unwrap(), 0.0);
        assert_eq!(provider.get_f107_adjusted(test_mjd).unwrap(), 0.0);
        assert_eq!(provider.get_f107_obs_avg81(test_mjd).unwrap(), 0.0);
        assert_eq!(provider.get_f107_adj_avg81(test_mjd).unwrap(), 0.0);
        assert_eq!(provider.get_sunspot_number(test_mjd).unwrap(), 0);
    }
}
