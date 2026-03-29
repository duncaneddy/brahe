/*!
 * Defines the TableSpaceWeatherProvider struct for providing
 * space weather data from an in-memory table of entries.
 */

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

use crate::space_weather::provider::SpaceWeatherProvider;
use crate::space_weather::types::{
    SpaceWeatherData, SpaceWeatherExtrapolation, SpaceWeatherSection, SpaceWeatherType,
};
use crate::time::conversions::datetime_to_mjd;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

// Type alias for the data map
type SWDataMap = BTreeMap<SWKey, SpaceWeatherData>;

// Custom key type for the space weather data BTreeMap (same pattern as FileSpaceWeatherProvider)
#[derive(Clone, PartialEq)]
struct SWKey(f64);

impl PartialOrd for SWKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SWKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

impl Eq for SWKey {}

/// Provides space weather data from an in-memory table of entries.
///
/// The `TableSpaceWeatherProvider` stores space weather data in a `BTreeMap` keyed
/// by MJD. It is constructed from a `Vec<SpaceWeatherData>`, making it suitable
/// for programmatically-generated or sampled space weather scenarios (e.g., Monte
/// Carlo simulations).
///
/// # Example
///
/// ```
/// use brahe::space_weather::{
///     TableSpaceWeatherProvider, SpaceWeatherProvider, SpaceWeatherData,
///     SpaceWeatherExtrapolation,
/// };
///
/// let entry = SpaceWeatherData {
///     year: 2024, month: 1, day: 15,
///     kp: [3.0; 8], kp_sum: 24.0,
///     ap: [15.0; 8], ap_avg: 15.0,
///     f107_obs: 150.0,
///     f107_adj_ctr81: 148.0,
///     f107_obs_ctr81: 149.0,
///     ..SpaceWeatherData::default()
/// };
///
/// let provider = TableSpaceWeatherProvider::from_entries(
///     vec![entry],
///     SpaceWeatherExtrapolation::Hold,
/// );
///
/// assert!(provider.is_initialized());
/// assert_eq!(provider.get_kp(60324.0).unwrap(), 3.0);
/// ```
#[derive(Clone)]
pub struct TableSpaceWeatherProvider {
    data: SWDataMap,
    extrapolate: SpaceWeatherExtrapolation,
    mjd_min: f64,
    mjd_max: f64,
    mjd_last_obs: f64,
    mjd_last_daily: f64,
    mjd_last_monthly: f64,
}

impl fmt::Display for TableSpaceWeatherProvider {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "TableSpaceWeatherProvider - {} entries, mjd_min: {:.1}, mjd_max: {:.1}, \
            extrapolation: {}",
            self.len(),
            self.mjd_min,
            self.mjd_max,
            self.extrapolate,
        )
    }
}

impl fmt::Debug for TableSpaceWeatherProvider {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "TableSpaceWeatherProvider<Length: {}, mjd_min: {:.1}, mjd_max: {:.1}, \
            extrapolation: {}>",
            self.len(),
            self.mjd_min,
            self.mjd_max,
            self.extrapolate,
        )
    }
}

impl TableSpaceWeatherProvider {
    /// Creates a new `TableSpaceWeatherProvider` from a vector of space weather data entries.
    ///
    /// Each entry's date (year, month, day) is converted to an MJD key. The MJD range
    /// and section boundaries are computed from the entries' `section` fields.
    ///
    /// # Arguments
    ///
    /// - `entries`: Vector of `SpaceWeatherData` entries
    /// - `extrapolate`: Extrapolation behavior for out-of-range requests
    ///
    /// # Returns
    ///
    /// A new `TableSpaceWeatherProvider` containing the provided entries.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::space_weather::{
    ///     TableSpaceWeatherProvider, SpaceWeatherData, SpaceWeatherExtrapolation,
    /// };
    ///
    /// let entries = vec![SpaceWeatherData {
    ///     year: 2024, month: 6, day: 1,
    ///     kp: [2.0; 8], kp_sum: 16.0,
    ///     ap: [7.0; 8], ap_avg: 7.0,
    ///     f107_obs: 120.0,
    ///     ..SpaceWeatherData::default()
    /// }];
    ///
    /// let provider = TableSpaceWeatherProvider::from_entries(
    ///     entries,
    ///     SpaceWeatherExtrapolation::Hold,
    /// );
    /// ```
    pub fn from_entries(
        entries: Vec<SpaceWeatherData>,
        extrapolate: SpaceWeatherExtrapolation,
    ) -> Self {
        let mut data: SWDataMap = BTreeMap::new();
        let mut mjd_min = f64::MAX;
        let mut mjd_max = f64::MIN;
        let mut mjd_last_obs = 0.0_f64;
        let mut mjd_last_daily = 0.0_f64;
        let mut mjd_last_monthly = 0.0_f64;

        for entry in entries {
            let mjd = datetime_to_mjd(entry.year, entry.month, entry.day, 0, 0, 0.0, 0.0);

            if mjd < mjd_min {
                mjd_min = mjd;
            }
            if mjd > mjd_max {
                mjd_max = mjd;
            }

            match entry.section {
                SpaceWeatherSection::Observed => {
                    if mjd > mjd_last_obs {
                        mjd_last_obs = mjd;
                    }
                }
                SpaceWeatherSection::DailyPredicted => {
                    if mjd > mjd_last_daily {
                        mjd_last_daily = mjd;
                    }
                }
                SpaceWeatherSection::MonthlyPredicted => {
                    if mjd > mjd_last_monthly {
                        mjd_last_monthly = mjd;
                    }
                }
            }

            data.insert(SWKey(mjd), entry);
        }

        // If no daily predictions, set to observed
        if mjd_last_daily == 0.0 {
            mjd_last_daily = mjd_last_obs;
        }

        // If no monthly predictions, set to daily
        if mjd_last_monthly == 0.0 {
            mjd_last_monthly = mjd_last_daily;
        }

        // Handle empty input
        if data.is_empty() {
            mjd_min = 0.0;
            mjd_max = 0.0;
        }

        TableSpaceWeatherProvider {
            data,
            extrapolate,
            mjd_min,
            mjd_max,
            mjd_last_obs,
            mjd_last_daily,
            mjd_last_monthly,
        }
    }

    /// Get the space weather data entry for a given MJD.
    ///
    /// Uses the floor of the MJD to find the daily data entry.
    fn get_data(&self, mjd: f64) -> Result<&SpaceWeatherData, BraheError> {
        if self.data.is_empty() {
            return Err(BraheError::SpaceWeatherError(
                "Table space weather provider has no data".to_string(),
            ));
        }

        let mjd_floor = mjd.floor();

        // Check bounds and handle extrapolation
        if mjd_floor < self.mjd_min || mjd_floor > self.mjd_max {
            match self.extrapolate {
                SpaceWeatherExtrapolation::Error => {
                    return Err(BraheError::SpaceWeatherError(format!(
                        "MJD {} is outside data range [{}, {}]",
                        mjd, self.mjd_min, self.mjd_max
                    )));
                }
                SpaceWeatherExtrapolation::Hold => {
                    let key = if mjd_floor < self.mjd_min {
                        SWKey(self.mjd_min)
                    } else {
                        SWKey(self.mjd_max)
                    };
                    if let Some(data) = self.data.get(&key) {
                        return Ok(data);
                    }
                }
                SpaceWeatherExtrapolation::Zero => {
                    // Fall through to return an error
                }
            }
        }

        // Find the entry for this day
        let key = SWKey(mjd_floor);

        // First try exact match
        if let Some(data) = self.data.get(&key) {
            return Ok(data);
        }

        // Find the previous entry
        if let Some((_, data)) = self.data.range(..=key).next_back() {
            return Ok(data);
        }

        Err(BraheError::SpaceWeatherError(format!(
            "No space weather data found for MJD {}",
            mjd
        )))
    }

    /// Get the 3-hour interval index (0-7) from the fractional MJD.
    fn get_interval_index(mjd: f64) -> usize {
        let fraction = mjd - mjd.floor();
        let hours = fraction * 24.0;
        let index = (hours / 3.0).floor() as usize;
        index.min(7)
    }

    /// Get the Kp/Ap data for a given MJD, applying extrapolation for MONTHLY_PREDICTED section.
    fn get_kp_ap_data(&self, mjd: f64) -> Result<&SpaceWeatherData, BraheError> {
        if self.data.is_empty() {
            return Err(BraheError::SpaceWeatherError(
                "Table space weather provider has no data".to_string(),
            ));
        }

        let mjd_floor = mjd.floor();

        // Check if we're in the MONTHLY_PREDICTED section
        if mjd_floor > self.mjd_last_daily {
            match self.extrapolate {
                SpaceWeatherExtrapolation::Error => {
                    return Err(BraheError::SpaceWeatherError(format!(
                        "Kp/Ap data not available for MJD {} (beyond daily predicted range, last daily: {})",
                        mjd, self.mjd_last_daily
                    )));
                }
                SpaceWeatherExtrapolation::Hold | SpaceWeatherExtrapolation::Zero => {
                    let key = SWKey(self.mjd_last_daily);
                    if let Some(data) = self.data.get(&key) {
                        return Ok(data);
                    }
                    if let Some((_, data)) = self.data.range(..=key).next_back() {
                        return Ok(data);
                    }
                }
            }
        }

        self.get_data(mjd)
    }

    /// Check if a value should be zeroed due to extrapolation mode and section.
    fn should_zero_kp_ap(&self, mjd: f64) -> bool {
        mjd.floor() > self.mjd_last_daily && self.extrapolate == SpaceWeatherExtrapolation::Zero
    }
}

impl SpaceWeatherProvider for TableSpaceWeatherProvider {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn sw_type(&self) -> SpaceWeatherType {
        SpaceWeatherType::Static
    }

    fn is_initialized(&self) -> bool {
        !self.data.is_empty()
    }

    fn extrapolation(&self) -> SpaceWeatherExtrapolation {
        self.extrapolate
    }

    fn mjd_min(&self) -> f64 {
        self.mjd_min
    }

    fn mjd_max(&self) -> f64 {
        self.mjd_max
    }

    fn mjd_last_observed(&self) -> f64 {
        self.mjd_last_obs
    }

    fn mjd_last_daily_predicted(&self) -> f64 {
        self.mjd_last_daily
    }

    fn mjd_last_monthly_predicted(&self) -> f64 {
        self.mjd_last_monthly
    }

    fn get_kp(&self, mjd: f64) -> Result<f64, BraheError> {
        let data = self.get_kp_ap_data(mjd)?;
        let index = Self::get_interval_index(mjd);
        if self.should_zero_kp_ap(mjd) {
            Ok(0.0)
        } else {
            Ok(data.kp[index])
        }
    }

    fn get_kp_all(&self, mjd: f64) -> Result<[f64; 8], BraheError> {
        let data = self.get_kp_ap_data(mjd)?;
        if self.should_zero_kp_ap(mjd) {
            Ok([0.0; 8])
        } else {
            Ok(data.kp)
        }
    }

    fn get_kp_daily(&self, mjd: f64) -> Result<f64, BraheError> {
        let data = self.get_kp_ap_data(mjd)?;
        if self.should_zero_kp_ap(mjd) {
            Ok(0.0)
        } else {
            Ok(data.kp_sum / 8.0)
        }
    }

    fn get_ap(&self, mjd: f64) -> Result<f64, BraheError> {
        let data = self.get_kp_ap_data(mjd)?;
        let index = Self::get_interval_index(mjd);
        if self.should_zero_kp_ap(mjd) {
            Ok(0.0)
        } else {
            Ok(data.ap[index])
        }
    }

    fn get_ap_all(&self, mjd: f64) -> Result<[f64; 8], BraheError> {
        let data = self.get_kp_ap_data(mjd)?;
        if self.should_zero_kp_ap(mjd) {
            Ok([0.0; 8])
        } else {
            Ok(data.ap)
        }
    }

    fn get_ap_daily(&self, mjd: f64) -> Result<f64, BraheError> {
        let data = self.get_kp_ap_data(mjd)?;
        if self.should_zero_kp_ap(mjd) {
            Ok(0.0)
        } else {
            Ok(data.ap_avg)
        }
    }

    fn get_f107_observed(&self, mjd: f64) -> Result<f64, BraheError> {
        let data = self.get_data(mjd)?;
        Ok(data.f107_obs)
    }

    fn get_f107_adjusted(&self, mjd: f64) -> Result<f64, BraheError> {
        let data = self.get_data(mjd)?;
        Ok(data.f107_adj_ctr81)
    }

    fn get_f107_obs_avg81(&self, mjd: f64) -> Result<f64, BraheError> {
        let data = self.get_data(mjd)?;
        Ok(data.f107_obs_ctr81)
    }

    fn get_f107_adj_avg81(&self, mjd: f64) -> Result<f64, BraheError> {
        let data = self.get_data(mjd)?;
        Ok(data.f107_adj_ctr81)
    }

    fn get_sunspot_number(&self, mjd: f64) -> Result<u32, BraheError> {
        let data = self.get_data(mjd)?;
        Ok(data.isn)
    }

    fn get_last_kp(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        if self.data.is_empty() {
            return Err(BraheError::SpaceWeatherError(
                "Table space weather provider has no data".to_string(),
            ));
        }

        if n == 0 {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(n);
        let mut current_mjd = mjd;
        let mut current_index = Self::get_interval_index(mjd);

        while result.len() < n {
            let data = self.get_kp_ap_data(current_mjd)?;
            let value = if self.should_zero_kp_ap(current_mjd) {
                0.0
            } else {
                data.kp[current_index]
            };
            result.push(value);

            if current_index == 0 {
                current_index = 7;
                current_mjd -= 1.0;
            } else {
                current_index -= 1;
            }
        }

        result.reverse();
        Ok(result)
    }

    fn get_last_ap(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        if self.data.is_empty() {
            return Err(BraheError::SpaceWeatherError(
                "Table space weather provider has no data".to_string(),
            ));
        }

        if n == 0 {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(n);
        let mut current_mjd = mjd;
        let mut current_index = Self::get_interval_index(mjd);

        while result.len() < n {
            let data = self.get_kp_ap_data(current_mjd)?;
            let value = if self.should_zero_kp_ap(current_mjd) {
                0.0
            } else {
                data.ap[current_index]
            };
            result.push(value);

            if current_index == 0 {
                current_index = 7;
                current_mjd -= 1.0;
            } else {
                current_index -= 1;
            }
        }

        result.reverse();
        Ok(result)
    }

    fn get_last_daily_kp(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        if self.data.is_empty() {
            return Err(BraheError::SpaceWeatherError(
                "Table space weather provider has no data".to_string(),
            ));
        }

        if n == 0 {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(n);
        let mut current_mjd = mjd.floor();

        while result.len() < n {
            let data = self.get_kp_ap_data(current_mjd)?;
            let value = if self.should_zero_kp_ap(current_mjd) {
                0.0
            } else {
                data.kp_sum / 8.0
            };
            result.push(value);
            current_mjd -= 1.0;
        }

        result.reverse();
        Ok(result)
    }

    fn get_last_daily_ap(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        if self.data.is_empty() {
            return Err(BraheError::SpaceWeatherError(
                "Table space weather provider has no data".to_string(),
            ));
        }

        if n == 0 {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(n);
        let mut current_mjd = mjd.floor();

        while result.len() < n {
            let data = self.get_kp_ap_data(current_mjd)?;
            let value = if self.should_zero_kp_ap(current_mjd) {
                0.0
            } else {
                data.ap_avg
            };
            result.push(value);
            current_mjd -= 1.0;
        }

        result.reverse();
        Ok(result)
    }

    fn get_last_f107(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        if self.data.is_empty() {
            return Err(BraheError::SpaceWeatherError(
                "Table space weather provider has no data".to_string(),
            ));
        }

        if n == 0 {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(n);
        let mut current_mjd = mjd.floor();

        while result.len() < n {
            let data = self.get_data(current_mjd)?;
            result.push(data.f107_obs);
            current_mjd -= 1.0;
        }

        result.reverse();
        Ok(result)
    }

    fn get_last_kpap_epochs(&self, mjd: f64, n: usize) -> Result<Vec<Epoch>, BraheError> {
        if self.data.is_empty() {
            return Err(BraheError::SpaceWeatherError(
                "Table space weather provider has no data".to_string(),
            ));
        }

        if n == 0 {
            return Ok(Vec::new());
        }

        let mut epochs = Vec::with_capacity(n);
        let mut current_mjd = mjd.floor();
        let mut current_index = Self::get_interval_index(mjd);

        while epochs.len() < n {
            let interval_mjd = current_mjd + (current_index as f64 * 3.0 / 24.0);
            epochs.push(Epoch::from_mjd(interval_mjd, TimeSystem::UTC));

            if current_index == 0 {
                current_index = 7;
                current_mjd -= 1.0;
            } else {
                current_index -= 1;
            }
        }

        epochs.reverse();
        Ok(epochs)
    }

    fn get_last_daily_epochs(&self, mjd: f64, n: usize) -> Result<Vec<Epoch>, BraheError> {
        if self.data.is_empty() {
            return Err(BraheError::SpaceWeatherError(
                "Table space weather provider has no data".to_string(),
            ));
        }

        if n == 0 {
            return Ok(Vec::new());
        }

        let mut epochs = Vec::with_capacity(n);
        let mut current_mjd = mjd.floor();

        while epochs.len() < n {
            epochs.push(Epoch::from_mjd(current_mjd, TimeSystem::UTC));
            current_mjd -= 1.0;
        }

        epochs.reverse();
        Ok(epochs)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn make_entry(year: u32, month: u8, day: u8, kp: f64, ap: f64, f107: f64) -> SpaceWeatherData {
        SpaceWeatherData {
            year,
            month,
            day,
            kp: [kp; 8],
            kp_sum: kp * 8.0,
            ap: [ap; 8],
            ap_avg: ap,
            f107_obs: f107,
            f107_adj_ctr81: f107 - 2.0,
            f107_obs_ctr81: f107 - 1.0,
            isn: 100,
            section: SpaceWeatherSection::Observed,
            ..SpaceWeatherData::default()
        }
    }

    fn make_test_provider() -> TableSpaceWeatherProvider {
        let entries = vec![
            make_entry(2024, 1, 14, 2.0, 7.0, 140.0),
            make_entry(2024, 1, 15, 3.0, 15.0, 150.0),
            make_entry(2024, 1, 16, 4.0, 22.0, 160.0),
        ];
        TableSpaceWeatherProvider::from_entries(entries, SpaceWeatherExtrapolation::Hold)
    }

    #[test]
    fn test_from_entries_basic() {
        let provider = make_test_provider();
        assert!(provider.is_initialized());
        assert_eq!(provider.len(), 3);
        assert_eq!(provider.sw_type(), SpaceWeatherType::Static);
        assert_eq!(provider.extrapolation(), SpaceWeatherExtrapolation::Hold);
    }

    #[test]
    fn test_from_entries_empty() {
        let provider =
            TableSpaceWeatherProvider::from_entries(vec![], SpaceWeatherExtrapolation::Hold);
        assert!(!provider.is_initialized());
        assert_eq!(provider.len(), 0);
        assert!(provider.is_empty());
    }

    #[test]
    fn test_get_kp() {
        let provider = make_test_provider();
        // MJD for 2024-01-15 is approximately 60324
        let mjd = datetime_to_mjd(2024, 1, 15, 0, 0, 0.0, 0.0);
        let kp = provider.get_kp(mjd).unwrap();
        assert_eq!(kp, 3.0);
    }

    #[test]
    fn test_get_kp_all() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 15, 0, 0, 0.0, 0.0);
        let kp_all = provider.get_kp_all(mjd).unwrap();
        assert_eq!(kp_all, [3.0; 8]);
    }

    #[test]
    fn test_get_kp_daily() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 15, 0, 0, 0.0, 0.0);
        let kp_daily = provider.get_kp_daily(mjd).unwrap();
        assert_eq!(kp_daily, 3.0); // kp_sum = 24.0, / 8 = 3.0
    }

    #[test]
    fn test_get_ap() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 15, 0, 0, 0.0, 0.0);
        let ap = provider.get_ap(mjd).unwrap();
        assert_eq!(ap, 15.0);
    }

    #[test]
    fn test_get_ap_all() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 15, 0, 0, 0.0, 0.0);
        let ap_all = provider.get_ap_all(mjd).unwrap();
        assert_eq!(ap_all, [15.0; 8]);
    }

    #[test]
    fn test_get_ap_daily() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 15, 0, 0, 0.0, 0.0);
        let ap_daily = provider.get_ap_daily(mjd).unwrap();
        assert_eq!(ap_daily, 15.0);
    }

    #[test]
    fn test_get_f107_observed() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 15, 0, 0, 0.0, 0.0);
        let f107 = provider.get_f107_observed(mjd).unwrap();
        assert_eq!(f107, 150.0);
    }

    #[test]
    fn test_get_f107_adjusted() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 15, 0, 0, 0.0, 0.0);
        let f107_adj = provider.get_f107_adjusted(mjd).unwrap();
        assert_eq!(f107_adj, 148.0);
    }

    #[test]
    fn test_get_f107_obs_avg81() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 15, 0, 0, 0.0, 0.0);
        let f107_avg = provider.get_f107_obs_avg81(mjd).unwrap();
        assert_eq!(f107_avg, 149.0);
    }

    #[test]
    fn test_get_f107_adj_avg81() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 15, 0, 0, 0.0, 0.0);
        let f107_avg = provider.get_f107_adj_avg81(mjd).unwrap();
        assert_eq!(f107_avg, 148.0);
    }

    #[test]
    fn test_get_sunspot_number() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 15, 0, 0, 0.0, 0.0);
        let isn = provider.get_sunspot_number(mjd).unwrap();
        assert_eq!(isn, 100);
    }

    #[test]
    fn test_hold_extrapolation_below_range() {
        let provider = make_test_provider();
        // Request well before the data range
        let mjd = datetime_to_mjd(2020, 1, 1, 0, 0, 0.0, 0.0);
        let kp = provider.get_kp(mjd).unwrap();
        // Should hold first entry's value (kp = 2.0 for 2024-01-14)
        assert_eq!(kp, 2.0);
    }

    #[test]
    fn test_hold_extrapolation_above_range() {
        let provider = make_test_provider();
        // Request well after the data range
        let mjd = datetime_to_mjd(2030, 1, 1, 0, 0, 0.0, 0.0);
        let f107 = provider.get_f107_observed(mjd).unwrap();
        // Should hold last entry's value (f107 = 160.0 for 2024-01-16)
        assert_eq!(f107, 160.0);
    }

    #[test]
    fn test_error_extrapolation() {
        let entries = vec![make_entry(2024, 1, 15, 3.0, 15.0, 150.0)];
        let provider =
            TableSpaceWeatherProvider::from_entries(entries, SpaceWeatherExtrapolation::Error);

        let mjd = datetime_to_mjd(2020, 1, 1, 0, 0, 0.0, 0.0);
        assert!(provider.get_f107_observed(mjd).is_err());
    }

    #[test]
    fn test_empty_provider_errors() {
        let provider =
            TableSpaceWeatherProvider::from_entries(vec![], SpaceWeatherExtrapolation::Hold);

        assert!(provider.get_kp(60000.0).is_err());
        assert!(provider.get_ap(60000.0).is_err());
        assert!(provider.get_f107_observed(60000.0).is_err());
        assert!(provider.get_sunspot_number(60000.0).is_err());
        assert!(provider.get_last_kp(60000.0, 1).is_err());
        assert!(provider.get_last_ap(60000.0, 1).is_err());
        assert!(provider.get_last_daily_kp(60000.0, 1).is_err());
        assert!(provider.get_last_daily_ap(60000.0, 1).is_err());
        assert!(provider.get_last_f107(60000.0, 1).is_err());
        assert!(provider.get_last_kpap_epochs(60000.0, 1).is_err());
        assert!(provider.get_last_daily_epochs(60000.0, 1).is_err());
    }

    #[test]
    fn test_mjd_range() {
        let provider = make_test_provider();
        let mjd_14 = datetime_to_mjd(2024, 1, 14, 0, 0, 0.0, 0.0);
        let mjd_16 = datetime_to_mjd(2024, 1, 16, 0, 0, 0.0, 0.0);
        assert_eq!(provider.mjd_min(), mjd_14);
        assert_eq!(provider.mjd_max(), mjd_16);
    }

    #[test]
    fn test_display() {
        let provider = make_test_provider();
        let display = format!("{}", provider);
        assert!(display.contains("TableSpaceWeatherProvider"));
        assert!(display.contains("3 entries"));
    }

    #[test]
    fn test_debug() {
        let provider = make_test_provider();
        let debug = format!("{:?}", provider);
        assert!(debug.contains("TableSpaceWeatherProvider"));
    }

    #[test]
    fn test_clone() {
        let provider = make_test_provider();
        let cloned = provider.clone();
        assert_eq!(provider.len(), cloned.len());
        assert_eq!(provider.mjd_min(), cloned.mjd_min());
    }

    #[test]
    fn test_get_last_kp() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 16, 0, 0, 0.0, 0.0);
        let kp_values = provider.get_last_kp(mjd, 3).unwrap();
        assert_eq!(kp_values.len(), 3);
    }

    #[test]
    fn test_get_last_ap() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 16, 0, 0, 0.0, 0.0);
        let ap_values = provider.get_last_ap(mjd, 3).unwrap();
        assert_eq!(ap_values.len(), 3);
    }

    #[test]
    fn test_get_last_daily_kp() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 16, 0, 0, 0.0, 0.0);
        let daily_kp = provider.get_last_daily_kp(mjd, 3).unwrap();
        assert_eq!(daily_kp.len(), 3);
    }

    #[test]
    fn test_get_last_daily_ap() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 16, 0, 0, 0.0, 0.0);
        let daily_ap = provider.get_last_daily_ap(mjd, 3).unwrap();
        assert_eq!(daily_ap.len(), 3);
    }

    #[test]
    fn test_get_last_f107() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 16, 0, 0, 0.0, 0.0);
        let f107_values = provider.get_last_f107(mjd, 3).unwrap();
        assert_eq!(f107_values.len(), 3);
    }

    #[test]
    fn test_get_last_kpap_epochs() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 16, 0, 0, 0.0, 0.0);
        let epochs = provider.get_last_kpap_epochs(mjd, 3).unwrap();
        assert_eq!(epochs.len(), 3);
        // Verify ascending order
        for i in 0..epochs.len() - 1 {
            assert!(epochs[i].mjd() < epochs[i + 1].mjd());
        }
    }

    #[test]
    fn test_get_last_daily_epochs() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 16, 0, 0, 0.0, 0.0);
        let epochs = provider.get_last_daily_epochs(mjd, 3).unwrap();
        assert_eq!(epochs.len(), 3);
        // Verify ascending order
        for i in 0..epochs.len() - 1 {
            assert!(epochs[i].mjd() < epochs[i + 1].mjd());
        }
    }

    #[test]
    fn test_get_last_zero_count() {
        let provider = make_test_provider();
        let mjd = datetime_to_mjd(2024, 1, 16, 0, 0, 0.0, 0.0);
        assert_eq!(provider.get_last_kp(mjd, 0).unwrap().len(), 0);
        assert_eq!(provider.get_last_ap(mjd, 0).unwrap().len(), 0);
        assert_eq!(provider.get_last_daily_kp(mjd, 0).unwrap().len(), 0);
        assert_eq!(provider.get_last_daily_ap(mjd, 0).unwrap().len(), 0);
        assert_eq!(provider.get_last_f107(mjd, 0).unwrap().len(), 0);
        assert_eq!(provider.get_last_kpap_epochs(mjd, 0).unwrap().len(), 0);
        assert_eq!(provider.get_last_daily_epochs(mjd, 0).unwrap().len(), 0);
    }

    #[test]
    fn test_section_tracking() {
        let mut entries = vec![
            make_entry(2024, 1, 14, 2.0, 7.0, 140.0),
            make_entry(2024, 1, 15, 3.0, 15.0, 150.0),
        ];
        // Mark second entry as DailyPredicted
        entries[1].section = SpaceWeatherSection::DailyPredicted;

        let provider =
            TableSpaceWeatherProvider::from_entries(entries, SpaceWeatherExtrapolation::Hold);

        let mjd_14 = datetime_to_mjd(2024, 1, 14, 0, 0, 0.0, 0.0);
        let mjd_15 = datetime_to_mjd(2024, 1, 15, 0, 0, 0.0, 0.0);

        assert_eq!(provider.mjd_last_observed(), mjd_14);
        assert_eq!(provider.mjd_last_daily_predicted(), mjd_15);
    }
}
