/*!
The static provider module implements a space weather provider that returns
static values for all space weather parameters.
*/

use std::fmt;

use crate::space_weather::provider::SpaceWeatherProvider;
use crate::space_weather::types::{SpaceWeatherExtrapolation, SpaceWeatherType};
use crate::time::{Epoch, TimeSystem};
use crate::utils::errors::BraheError;

/// StaticSpaceWeatherProvider is a SpaceWeatherProvider that returns static
/// values for all space weather parameters at all times.
///
/// It can be initialized as zero-valued or with specific values. It will
/// never extrapolate or interpolate since the data is only for a single
/// time point.
///
/// # Example
///
/// ```
/// use brahe::space_weather::{StaticSpaceWeatherProvider, SpaceWeatherProvider};
///
/// let sw = StaticSpaceWeatherProvider::from_zero();
/// assert!(sw.is_initialized());
/// assert_eq!(sw.get_ap_daily(60000.0).unwrap(), 0.0);
/// ```
#[derive(Clone)]
pub struct StaticSpaceWeatherProvider {
    /// Internal variable to indicate whether the provider has been properly initialized
    initialized: bool,
    /// Kp index value (0.0-9.0)
    kp: f64,
    /// Ap index value
    ap: f64,
    /// F10.7 observed solar flux (sfu)
    f107_obs: f64,
    /// F10.7 adjusted solar flux (sfu)
    f107_adj: f64,
    /// International Sunspot Number
    isn: u32,
}

impl fmt::Display for StaticSpaceWeatherProvider {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "StaticSpaceWeatherProvider - type: {}, {} entries, Kp: {}, Ap: {}, F10.7: {}",
            self.sw_type(),
            self.len(),
            self.kp,
            self.ap,
            self.f107_obs
        )
    }
}

impl fmt::Debug for StaticSpaceWeatherProvider {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "StaticSpaceWeatherProvider<Type: {}, Length: {}, Kp: {}, Ap: {}, F10.7: {}>",
            self.sw_type(),
            self.len(),
            self.kp,
            self.ap,
            self.f107_obs
        )
    }
}

impl Default for StaticSpaceWeatherProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl StaticSpaceWeatherProvider {
    /// Creates a new, uninitialized `StaticSpaceWeatherProvider` with zero values.
    /// This is the default constructor. It is not recommended to use this constructor
    /// unless a placeholder `StaticSpaceWeatherProvider` allocation is needed.
    ///
    /// # Returns
    ///
    /// * `StaticSpaceWeatherProvider` - New `StaticSpaceWeatherProvider` that is not initialized.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::space_weather::{StaticSpaceWeatherProvider, SpaceWeatherProvider};
    ///
    /// let sw = StaticSpaceWeatherProvider::new();
    /// assert_eq!(sw.is_initialized(), false);
    /// ```
    pub fn new() -> Self {
        StaticSpaceWeatherProvider {
            initialized: false,
            kp: 0.0,
            ap: 0.0,
            f107_obs: 0.0,
            f107_adj: 0.0,
            isn: 0,
        }
    }

    /// Creates a new `StaticSpaceWeatherProvider` with zero values for all parameters.
    ///
    /// # Returns
    ///
    /// * `StaticSpaceWeatherProvider` - New initialized `StaticSpaceWeatherProvider` with zero values.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::space_weather::{StaticSpaceWeatherProvider, SpaceWeatherProvider};
    ///
    /// let sw = StaticSpaceWeatherProvider::from_zero();
    /// assert!(sw.is_initialized());
    /// assert_eq!(sw.get_kp_daily(60000.0).unwrap(), 0.0);
    /// ```
    pub fn from_zero() -> Self {
        StaticSpaceWeatherProvider {
            initialized: true,
            kp: 0.0,
            ap: 0.0,
            f107_obs: 0.0,
            f107_adj: 0.0,
            isn: 0,
        }
    }

    /// Creates a new `StaticSpaceWeatherProvider` with the given values.
    /// The static values will be returned for all times.
    ///
    /// # Arguments
    ///
    /// * `kp` - Kp index (0.0-9.0 scale)
    /// * `ap` - Ap index
    /// * `f107_obs` - Observed F10.7 solar flux. Units: sfu
    /// * `f107_adj` - Adjusted F10.7 solar flux. Units: sfu
    /// * `isn` - International Sunspot Number
    ///
    /// # Returns
    ///
    /// * `StaticSpaceWeatherProvider` - New initialized `StaticSpaceWeatherProvider` with given values.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::space_weather::{StaticSpaceWeatherProvider, SpaceWeatherProvider};
    ///
    /// let sw = StaticSpaceWeatherProvider::from_values(3.0, 15.0, 150.0, 148.0, 100);
    /// assert!(sw.is_initialized());
    /// assert_eq!(sw.get_kp_daily(60000.0).unwrap(), 3.0);
    /// assert_eq!(sw.get_ap_daily(60000.0).unwrap(), 15.0);
    /// assert_eq!(sw.get_f107_observed(60000.0).unwrap(), 150.0);
    /// ```
    pub fn from_values(kp: f64, ap: f64, f107_obs: f64, f107_adj: f64, isn: u32) -> Self {
        StaticSpaceWeatherProvider {
            initialized: true,
            kp,
            ap,
            f107_obs,
            f107_adj,
            isn,
        }
    }
}

impl SpaceWeatherProvider for StaticSpaceWeatherProvider {
    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn len(&self) -> usize {
        1
    }

    fn sw_type(&self) -> SpaceWeatherType {
        SpaceWeatherType::Static
    }

    fn extrapolation(&self) -> SpaceWeatherExtrapolation {
        SpaceWeatherExtrapolation::Hold
    }

    fn mjd_min(&self) -> f64 {
        0.0
    }

    fn mjd_max(&self) -> f64 {
        f64::MAX
    }

    fn mjd_last_observed(&self) -> f64 {
        f64::MAX
    }

    fn mjd_last_daily_predicted(&self) -> f64 {
        f64::MAX
    }

    fn mjd_last_monthly_predicted(&self) -> f64 {
        f64::MAX
    }

    fn get_kp(&self, _mjd: f64) -> Result<f64, BraheError> {
        if self.initialized {
            Ok(self.kp)
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }

    fn get_kp_all(&self, _mjd: f64) -> Result<[f64; 8], BraheError> {
        if self.initialized {
            Ok([self.kp; 8])
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }

    fn get_kp_daily(&self, _mjd: f64) -> Result<f64, BraheError> {
        if self.initialized {
            Ok(self.kp)
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }

    fn get_ap(&self, _mjd: f64) -> Result<f64, BraheError> {
        if self.initialized {
            Ok(self.ap)
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }

    fn get_ap_all(&self, _mjd: f64) -> Result<[f64; 8], BraheError> {
        if self.initialized {
            Ok([self.ap; 8])
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }

    fn get_ap_daily(&self, _mjd: f64) -> Result<f64, BraheError> {
        if self.initialized {
            Ok(self.ap)
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }

    fn get_f107_observed(&self, _mjd: f64) -> Result<f64, BraheError> {
        if self.initialized {
            Ok(self.f107_obs)
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }

    fn get_f107_adjusted(&self, _mjd: f64) -> Result<f64, BraheError> {
        if self.initialized {
            Ok(self.f107_adj)
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }

    fn get_f107_obs_avg81(&self, _mjd: f64) -> Result<f64, BraheError> {
        if self.initialized {
            Ok(self.f107_obs)
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }

    fn get_f107_adj_avg81(&self, _mjd: f64) -> Result<f64, BraheError> {
        if self.initialized {
            Ok(self.f107_adj)
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }

    fn get_sunspot_number(&self, _mjd: f64) -> Result<u32, BraheError> {
        if self.initialized {
            Ok(self.isn)
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }

    fn get_last_kp(&self, _mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        if self.initialized {
            Ok(vec![self.kp; n])
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }

    fn get_last_ap(&self, _mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        if self.initialized {
            Ok(vec![self.ap; n])
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }

    fn get_last_daily_kp(&self, _mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        if self.initialized {
            Ok(vec![self.kp; n])
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }

    fn get_last_daily_ap(&self, _mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        if self.initialized {
            Ok(vec![self.ap; n])
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }

    fn get_last_f107(&self, _mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        if self.initialized {
            Ok(vec![self.f107_obs; n])
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }

    fn get_last_kpap_epochs(&self, mjd: f64, n: usize) -> Result<Vec<Epoch>, BraheError> {
        if self.initialized {
            let mut epochs = Vec::with_capacity(n);
            let mjd_floor = mjd.floor();
            let fraction = mjd - mjd_floor;
            let hours = fraction * 24.0;
            let mut current_index = (hours / 3.0).floor() as usize;
            current_index = current_index.min(7);
            let mut current_mjd = mjd_floor;

            for _ in 0..n {
                let interval_mjd = current_mjd + (current_index as f64 * 3.0 / 24.0);
                epochs.push(Epoch::from_mjd(interval_mjd, TimeSystem::UTC));

                // Move to previous interval
                if current_index == 0 {
                    current_index = 7;
                    current_mjd -= 1.0;
                } else {
                    current_index -= 1;
                }
            }

            epochs.reverse();
            Ok(epochs)
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }

    fn get_last_daily_epochs(&self, mjd: f64, n: usize) -> Result<Vec<Epoch>, BraheError> {
        if self.initialized {
            let mut epochs = Vec::with_capacity(n);
            let mut current_mjd = mjd.floor();

            for _ in 0..n {
                epochs.push(Epoch::from_mjd(current_mjd, TimeSystem::UTC));
                current_mjd -= 1.0;
            }

            epochs.reverse();
            Ok(epochs)
        } else {
            Err(BraheError::SpaceWeatherError(String::from(
                "Space weather provider not initialized",
            )))
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_uninitialized_provider() {
        let provider = StaticSpaceWeatherProvider::new();
        assert!(!provider.is_initialized());
    }

    #[test]
    fn test_from_zero() {
        let sw = StaticSpaceWeatherProvider::from_zero();

        assert!(sw.is_initialized());
        assert_eq!(sw.len(), 1);
        assert_eq!(sw.mjd_min(), 0.0);
        assert_eq!(sw.mjd_max(), f64::MAX);
        assert_eq!(sw.sw_type(), SpaceWeatherType::Static);
        assert_eq!(sw.extrapolation(), SpaceWeatherExtrapolation::Hold);

        // Values
        assert_eq!(sw.get_kp(59950.0).unwrap(), 0.0);
        assert_eq!(sw.get_kp_all(59950.0).unwrap(), [0.0; 8]);
        assert_eq!(sw.get_kp_daily(59950.0).unwrap(), 0.0);
        assert_eq!(sw.get_ap(59950.0).unwrap(), 0.0);
        assert_eq!(sw.get_ap_all(59950.0).unwrap(), [0.0; 8]);
        assert_eq!(sw.get_ap_daily(59950.0).unwrap(), 0.0);
        assert_eq!(sw.get_f107_observed(59950.0).unwrap(), 0.0);
        assert_eq!(sw.get_f107_adjusted(59950.0).unwrap(), 0.0);
        assert_eq!(sw.get_sunspot_number(59950.0).unwrap(), 0);
    }

    #[test]
    fn test_from_values() {
        let sw = StaticSpaceWeatherProvider::from_values(3.5, 15.0, 150.0, 148.0, 100);

        assert!(sw.is_initialized());
        assert_eq!(sw.len(), 1);
        assert_eq!(sw.mjd_min(), 0.0);
        assert_eq!(sw.mjd_max(), f64::MAX);
        assert_eq!(sw.sw_type(), SpaceWeatherType::Static);
        assert_eq!(sw.extrapolation(), SpaceWeatherExtrapolation::Hold);

        // Values
        assert_eq!(sw.get_kp(59950.0).unwrap(), 3.5);
        assert_eq!(sw.get_kp_daily(59950.0).unwrap(), 3.5);
        assert_eq!(sw.get_ap(59950.0).unwrap(), 15.0);
        assert_eq!(sw.get_ap_daily(59950.0).unwrap(), 15.0);
        assert_eq!(sw.get_f107_observed(59950.0).unwrap(), 150.0);
        assert_eq!(sw.get_f107_adjusted(59950.0).unwrap(), 148.0);
        assert_eq!(sw.get_f107_obs_avg81(59950.0).unwrap(), 150.0);
        assert_eq!(sw.get_f107_adj_avg81(59950.0).unwrap(), 148.0);
        assert_eq!(sw.get_sunspot_number(59950.0).unwrap(), 100);
    }

    #[test]
    fn test_error_when_not_initialized() {
        let provider = StaticSpaceWeatherProvider::new();

        assert!(provider.get_kp(60000.0).is_err());
        assert!(provider.get_kp_all(60000.0).is_err());
        assert!(provider.get_kp_daily(60000.0).is_err());
        assert!(provider.get_ap(60000.0).is_err());
        assert!(provider.get_ap_all(60000.0).is_err());
        assert!(provider.get_ap_daily(60000.0).is_err());
        assert!(provider.get_f107_observed(60000.0).is_err());
        assert!(provider.get_f107_adjusted(60000.0).is_err());
        assert!(provider.get_f107_obs_avg81(60000.0).is_err());
        assert!(provider.get_f107_adj_avg81(60000.0).is_err());
        assert!(provider.get_sunspot_number(60000.0).is_err());
    }

    #[test]
    fn test_default_implementation() {
        let sw_default = StaticSpaceWeatherProvider::default();
        let sw_new = StaticSpaceWeatherProvider::new();

        assert_eq!(sw_default.is_initialized(), sw_new.is_initialized());
        assert!(!sw_default.is_initialized());
        assert_eq!(sw_default.sw_type(), sw_new.sw_type());
        assert_eq!(sw_default.len(), sw_new.len());
    }

    #[test]
    fn test_display() {
        let sw = StaticSpaceWeatherProvider::from_values(3.0, 15.0, 150.0, 148.0, 100);
        let display = format!("{}", sw);
        assert!(display.contains("StaticSpaceWeatherProvider"));
        assert!(display.contains("Kp: 3"));
        assert!(display.contains("Ap: 15"));
        assert!(display.contains("F10.7: 150"));
    }

    #[test]
    fn test_debug() {
        let sw = StaticSpaceWeatherProvider::from_values(3.0, 15.0, 150.0, 148.0, 100);
        let debug = format!("{:?}", sw);
        assert!(debug.contains("StaticSpaceWeatherProvider"));
    }

    #[test]
    fn test_clone() {
        let sw1 = StaticSpaceWeatherProvider::from_values(3.0, 15.0, 150.0, 148.0, 100);
        let sw2 = sw1.clone();

        assert_eq!(
            sw1.get_kp_daily(60000.0).unwrap(),
            sw2.get_kp_daily(60000.0).unwrap()
        );
        assert_eq!(
            sw1.get_ap_daily(60000.0).unwrap(),
            sw2.get_ap_daily(60000.0).unwrap()
        );
    }

    #[test]
    fn test_mjd_last_daily_predicted() {
        let sw = StaticSpaceWeatherProvider::from_values(3.0, 15.0, 150.0, 148.0, 100);
        // Static provider returns max float for all MJD boundaries
        assert_eq!(sw.mjd_last_daily_predicted(), f64::MAX);
    }

    #[test]
    fn test_mjd_last_monthly_predicted() {
        let sw = StaticSpaceWeatherProvider::from_values(3.0, 15.0, 150.0, 148.0, 100);
        // Static provider returns max float for all MJD boundaries
        assert_eq!(sw.mjd_last_monthly_predicted(), f64::MAX);
    }

    #[test]
    fn test_get_last_kp() {
        let sw = StaticSpaceWeatherProvider::from_values(3.0, 15.0, 150.0, 148.0, 100);
        let kp_values = sw.get_last_kp(60000.0, 5).unwrap();
        assert_eq!(kp_values.len(), 5);
        for kp in kp_values {
            assert_eq!(kp, 3.0);
        }
    }

    #[test]
    fn test_get_last_ap() {
        let sw = StaticSpaceWeatherProvider::from_values(3.0, 15.0, 150.0, 148.0, 100);
        let ap_values = sw.get_last_ap(60000.0, 5).unwrap();
        assert_eq!(ap_values.len(), 5);
        for ap in ap_values {
            assert_eq!(ap, 15.0);
        }
    }

    #[test]
    fn test_get_last_daily_kp() {
        let sw = StaticSpaceWeatherProvider::from_values(3.0, 15.0, 150.0, 148.0, 100);
        let daily_kp = sw.get_last_daily_kp(60000.0, 3).unwrap();
        assert_eq!(daily_kp.len(), 3);
        for kp in daily_kp {
            assert_eq!(kp, 3.0);
        }
    }

    #[test]
    fn test_get_last_daily_ap() {
        let sw = StaticSpaceWeatherProvider::from_values(3.0, 15.0, 150.0, 148.0, 100);
        let daily_ap = sw.get_last_daily_ap(60000.0, 3).unwrap();
        assert_eq!(daily_ap.len(), 3);
        for ap in daily_ap {
            assert_eq!(ap, 15.0);
        }
    }

    #[test]
    fn test_get_last_f107() {
        let sw = StaticSpaceWeatherProvider::from_values(3.0, 15.0, 150.0, 148.0, 100);
        let f107_values = sw.get_last_f107(60000.0, 3).unwrap();
        assert_eq!(f107_values.len(), 3);
        for f107 in f107_values {
            assert_eq!(f107, 150.0);
        }
    }

    #[test]
    fn test_get_last_kpap_epochs() {
        let sw = StaticSpaceWeatherProvider::from_values(3.0, 15.0, 150.0, 148.0, 100);
        let epochs = sw.get_last_kpap_epochs(60000.0, 5).unwrap();
        assert_eq!(epochs.len(), 5);

        // Verify epochs are in ascending order (oldest first)
        for i in 0..epochs.len() - 1 {
            assert!(epochs[i].mjd() < epochs[i + 1].mjd());
        }
    }

    #[test]
    fn test_get_last_daily_epochs() {
        let sw = StaticSpaceWeatherProvider::from_values(3.0, 15.0, 150.0, 148.0, 100);
        let epochs = sw.get_last_daily_epochs(60000.0, 3).unwrap();
        assert_eq!(epochs.len(), 3);

        // Verify epochs are in ascending order (oldest first)
        for i in 0..epochs.len() - 1 {
            assert!(epochs[i].mjd() < epochs[i + 1].mjd());
        }
    }
}
