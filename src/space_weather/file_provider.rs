/*!
 * Defines the FileSpaceWeatherProvider struct for loading
 * and accessing space weather data from files.
 */

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::io::BufReader;
use std::io::prelude::*;
use std::ops::Bound;
use std::path::Path;

use crate::space_weather::parser::{is_data_line, parse_cssi_line_with_section};
use crate::space_weather::provider::SpaceWeatherProvider;
use crate::space_weather::types::{
    SpaceWeatherData, SpaceWeatherExtrapolation, SpaceWeatherSection, SpaceWeatherType,
};
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

// Type alias for complex space weather data structures
type SWDataMap = BTreeMap<SWKey, SpaceWeatherData>;

// Package space weather data as part of crate
/// Packaged CSSI Space Weather Data File
static PACKAGED_SW_FILE: &[u8] = include_bytes!("../../data/space_weather/sw19571001.txt");

// Define a custom key type for the space weather data BTreeMap
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

/// Provides space weather data from a file.
///
/// The `FileSpaceWeatherProvider` struct represents a provider of space weather data
/// loaded from a CSSI-formatted file. It stores the loaded data and provides methods
/// to access geomagnetic indices (Kp, Ap) and solar flux (F10.7).
///
/// # Example
///
/// ```
/// use brahe::space_weather::{FileSpaceWeatherProvider, SpaceWeatherProvider};
///
/// let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
/// assert!(sw.is_initialized());
/// ```
#[derive(Clone)]
pub struct FileSpaceWeatherProvider {
    initialized: bool,
    /// Type of space weather data loaded
    pub sw_type: SpaceWeatherType,
    data: SWDataMap,
    /// Extrapolation behavior when requested time is outside data range
    pub extrapolate: SpaceWeatherExtrapolation,
    /// Minimum Modified Julian Date in loaded dataset
    pub mjd_min: f64,
    /// Maximum Modified Julian Date in loaded dataset
    pub mjd_max: f64,
    /// Last MJD with observed (historical) data
    pub mjd_last_obs: f64,
    /// Last MJD with daily predicted data
    pub mjd_last_daily: f64,
    /// Last MJD with monthly predicted data
    pub mjd_last_monthly: f64,
}

impl fmt::Display for FileSpaceWeatherProvider {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FileSpaceWeatherProvider - type: {}, {} entries, mjd_min: {:.1}, mjd_max: {:.1}, \
            mjd_last_observed: {:.1}, extrapolation: {}",
            self.sw_type(),
            self.len(),
            self.mjd_min(),
            self.mjd_max(),
            self.mjd_last_observed(),
            self.extrapolation()
        )
    }
}

impl fmt::Debug for FileSpaceWeatherProvider {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "FileSpaceWeatherProvider<Type: {}, Length: {}, mjd_min: {:.1}, mjd_max: {:.1}, \
            mjd_last_observed: {:.1}, extrapolation: {}>",
            self.sw_type(),
            self.len(),
            self.mjd_min(),
            self.mjd_max(),
            self.mjd_last_observed(),
            self.extrapolation()
        )
    }
}

impl Default for FileSpaceWeatherProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl FileSpaceWeatherProvider {
    /// Creates a new, uninitialized `FileSpaceWeatherProvider`.
    ///
    /// # Returns
    ///
    /// * `FileSpaceWeatherProvider` - New uninitialized provider
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::space_weather::FileSpaceWeatherProvider;
    ///
    /// let sw = FileSpaceWeatherProvider::new();
    /// ```
    pub fn new() -> Self {
        Self {
            initialized: false,
            sw_type: SpaceWeatherType::Unknown,
            data: BTreeMap::new(),
            extrapolate: SpaceWeatherExtrapolation::Hold,
            mjd_min: 0.0,
            mjd_max: 0.0,
            mjd_last_obs: 0.0,
            mjd_last_daily: 0.0,
            mjd_last_monthly: 0.0,
        }
    }

    /// Creates a new `FileSpaceWeatherProvider` from a file path.
    ///
    /// # Arguments
    ///
    /// * `filepath` - Path to the space weather file
    /// * `extrapolate` - Extrapolation behavior for out-of-range requests
    ///
    /// # Returns
    ///
    /// * `Result<FileSpaceWeatherProvider, BraheError>` - Provider or error
    ///
    /// # Example
    ///
    /// ```
    /// use std::env;
    /// use std::path::Path;
    /// use brahe::space_weather::{FileSpaceWeatherProvider, SpaceWeatherExtrapolation};
    ///
    /// let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    /// let filepath = Path::new(&manifest_dir)
    ///                 .join("data")
    ///                 .join("space_weather")
    ///                 .join("sw19571001.txt");
    ///
    /// let sw = FileSpaceWeatherProvider::from_file(&filepath, SpaceWeatherExtrapolation::Hold).unwrap();
    /// ```
    pub fn from_file(
        filepath: &Path,
        extrapolate: SpaceWeatherExtrapolation,
    ) -> Result<Self, BraheError> {
        let file = std::fs::File::open(filepath)?;
        let reader = BufReader::new(file);
        Self::from_bufreader(reader, extrapolate)
    }

    /// Creates a new `FileSpaceWeatherProvider` from the packaged default file.
    ///
    /// # Returns
    ///
    /// * `Result<FileSpaceWeatherProvider, BraheError>` - Provider or error
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::space_weather::{FileSpaceWeatherProvider, SpaceWeatherProvider};
    ///
    /// let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
    /// assert!(sw.is_initialized());
    /// ```
    pub fn from_default_file() -> Result<Self, BraheError> {
        let reader = BufReader::new(PACKAGED_SW_FILE);
        Self::from_bufreader(reader, SpaceWeatherExtrapolation::Hold)
    }

    /// Creates a new `FileSpaceWeatherProvider` from a BufReader.
    fn from_bufreader<R: BufRead>(
        reader: R,
        extrapolate: SpaceWeatherExtrapolation,
    ) -> Result<Self, BraheError> {
        let mut data: SWDataMap = BTreeMap::new();
        let mut mjd_min = f64::MAX;
        let mut mjd_max = f64::MIN;
        let mut mjd_last_obs = 0.0;
        let mut mjd_last_daily = 0.0;
        let mut mjd_last_monthly = 0.0;

        let mut current_section: Option<SpaceWeatherSection> = None;

        for line_result in reader.lines() {
            let line = line_result?;
            let trimmed = line.trim();

            // Skip empty lines
            if trimmed.is_empty() {
                continue;
            }

            // Check for section markers
            if trimmed.starts_with("BEGIN OBSERVED") {
                current_section = Some(SpaceWeatherSection::Observed);
                continue;
            } else if trimmed.starts_with("BEGIN DAILY_PREDICTED") {
                current_section = Some(SpaceWeatherSection::DailyPredicted);
                continue;
            } else if trimmed.starts_with("BEGIN MONTHLY_PREDICTED") {
                current_section = Some(SpaceWeatherSection::MonthlyPredicted);
                continue;
            } else if trimmed.starts_with("END ") {
                continue;
            }

            // Skip non-data lines or lines before any section
            if !is_data_line(trimmed) || current_section.is_none() {
                continue;
            }

            let section = current_section.unwrap();

            // Parse the data line with section information
            match parse_cssi_line_with_section(trimmed, section) {
                Ok((mjd, sw_data)) => {
                    // Update MJD range
                    if mjd < mjd_min {
                        mjd_min = mjd;
                    }
                    if mjd > mjd_max {
                        mjd_max = mjd;
                    }

                    // Track section boundaries
                    match section {
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

                    data.insert(SWKey(mjd), sw_data);
                }
                Err(e) => {
                    // Log error but continue - some lines may still fail to parse
                    eprintln!("Warning: Failed to parse space weather line: {}", e);
                    continue;
                }
            }
        }

        // If no daily predictions, set to observed
        if mjd_last_daily == 0.0 {
            mjd_last_daily = mjd_last_obs;
        }

        // If no monthly predictions, set to daily
        if mjd_last_monthly == 0.0 {
            mjd_last_monthly = mjd_last_daily;
        }

        Ok(Self {
            initialized: !data.is_empty(),
            sw_type: SpaceWeatherType::CssiSpaceWeather,
            data,
            extrapolate,
            mjd_min,
            mjd_max,
            mjd_last_obs,
            mjd_last_daily,
            mjd_last_monthly,
        })
    }

    /// Get the space weather data entry for a given MJD.
    ///
    /// Uses the floor of the MJD to find the daily data entry.
    fn get_data(&self, mjd: f64) -> Result<&SpaceWeatherData, BraheError> {
        if !self.initialized {
            return Err(BraheError::SpaceWeatherError(
                "Space weather provider not initialized".to_string(),
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
                    // Get the nearest boundary value
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
                    // This is a special case - we need to return zeros, but we don't have
                    // a default zero entry. We'll fall through and return an error.
                }
            }
        }

        // Find the entry for this day
        let key = SWKey(mjd_floor);

        // First try exact match
        if let Some(data) = self.data.get(&key) {
            return Ok(data);
        }

        // Use cursor to find the previous entry
        let mut cursor = self.data.upper_bound(Bound::Included(&key));
        if let Some((_, data)) = cursor.prev() {
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
        index.min(7) // Ensure we don't exceed index 7
    }

    /// Get the Kp/Ap data for a given MJD, applying extrapolation for MONTHLY_PREDICTED section.
    ///
    /// For dates beyond mjd_last_daily (i.e., in MONTHLY_PREDICTED), this returns the
    /// data from mjd_last_daily with extrapolation behavior applied.
    fn get_kp_ap_data(&self, mjd: f64) -> Result<&SpaceWeatherData, BraheError> {
        if !self.initialized {
            return Err(BraheError::SpaceWeatherError(
                "Space weather provider not initialized".to_string(),
            ));
        }

        let mjd_floor = mjd.floor();

        // Check if we're in the MONTHLY_PREDICTED section
        if mjd_floor > self.mjd_last_daily {
            // Apply extrapolation behavior from last daily predicted
            match self.extrapolate {
                SpaceWeatherExtrapolation::Error => {
                    return Err(BraheError::SpaceWeatherError(format!(
                        "Kp/Ap data not available for MJD {} (beyond daily predicted range, last daily: {})",
                        mjd, self.mjd_last_daily
                    )));
                }
                SpaceWeatherExtrapolation::Hold | SpaceWeatherExtrapolation::Zero => {
                    // Get the last daily predicted data
                    let key = SWKey(self.mjd_last_daily);
                    if let Some(data) = self.data.get(&key) {
                        return Ok(data);
                    }
                    // Fallback: find nearest data
                    let mut cursor = self.data.upper_bound(Bound::Included(&key));
                    if let Some((_, data)) = cursor.prev() {
                        return Ok(data);
                    }
                }
            }
        }

        // Normal data retrieval
        self.get_data(mjd)
    }

    /// Check if a value should be zeroed due to extrapolation mode and section.
    fn should_zero_kp_ap(&self, mjd: f64) -> bool {
        mjd.floor() > self.mjd_last_daily && self.extrapolate == SpaceWeatherExtrapolation::Zero
    }
}

impl SpaceWeatherProvider for FileSpaceWeatherProvider {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn sw_type(&self) -> SpaceWeatherType {
        self.sw_type
    }

    fn is_initialized(&self) -> bool {
        self.initialized
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
        // Use the adjusted 81-day centered average as "adjusted" F10.7
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
        if !self.initialized {
            return Err(BraheError::SpaceWeatherError(
                "Space weather provider not initialized".to_string(),
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

            // Move to previous interval
            if current_index == 0 {
                current_index = 7;
                current_mjd -= 1.0; // Previous day
            } else {
                current_index -= 1;
            }
        }

        // Reverse to get oldest first
        result.reverse();
        Ok(result)
    }

    fn get_last_ap(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        if !self.initialized {
            return Err(BraheError::SpaceWeatherError(
                "Space weather provider not initialized".to_string(),
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

            // Move to previous interval
            if current_index == 0 {
                current_index = 7;
                current_mjd -= 1.0; // Previous day
            } else {
                current_index -= 1;
            }
        }

        // Reverse to get oldest first
        result.reverse();
        Ok(result)
    }

    fn get_last_daily_kp(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        if !self.initialized {
            return Err(BraheError::SpaceWeatherError(
                "Space weather provider not initialized".to_string(),
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

        // Reverse to get oldest first
        result.reverse();
        Ok(result)
    }

    fn get_last_daily_ap(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        if !self.initialized {
            return Err(BraheError::SpaceWeatherError(
                "Space weather provider not initialized".to_string(),
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

        // Reverse to get oldest first
        result.reverse();
        Ok(result)
    }

    fn get_last_f107(&self, mjd: f64, n: usize) -> Result<Vec<f64>, BraheError> {
        if !self.initialized {
            return Err(BraheError::SpaceWeatherError(
                "Space weather provider not initialized".to_string(),
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

        // Reverse to get oldest first
        result.reverse();
        Ok(result)
    }

    fn get_last_kpap_epochs(&self, mjd: f64, n: usize) -> Result<Vec<Epoch>, BraheError> {
        if !self.initialized {
            return Err(BraheError::SpaceWeatherError(
                "Space weather provider not initialized".to_string(),
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

            // Move to previous interval
            if current_index == 0 {
                current_index = 7;
                current_mjd -= 1.0;
            } else {
                current_index -= 1;
            }
        }

        // Reverse to get oldest first
        epochs.reverse();
        Ok(epochs)
    }

    fn get_last_daily_epochs(&self, mjd: f64, n: usize) -> Result<Vec<Epoch>, BraheError> {
        if !self.initialized {
            return Err(BraheError::SpaceWeatherError(
                "Space weather provider not initialized".to_string(),
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

        // Reverse to get oldest first
        epochs.reverse();
        Ok(epochs)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_from_default_file() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        assert!(sw.is_initialized());
        assert!(sw.len() > 0);
        assert_eq!(sw.sw_type(), SpaceWeatherType::CssiSpaceWeather);
    }

    #[test]
    fn test_mjd_range() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        // Data should start from 1957
        assert!(sw.mjd_min() < 40000.0);
        // Data should extend into the future
        assert!(sw.mjd_max() > 60000.0);
    }

    #[test]
    fn test_get_kp() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        // Test data for 1957-10-01 (MJD ~36114)
        let mjd = 36114.0;
        let kp = sw.get_kp(mjd).unwrap();
        // Kp should be in valid range
        assert!((0.0..=9.0).contains(&kp));
    }

    #[test]
    fn test_get_kp_all() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        let mjd = 36114.0;
        let kp_all = sw.get_kp_all(mjd).unwrap();
        assert_eq!(kp_all.len(), 8);
        for kp in kp_all.iter() {
            assert!(*kp >= 0.0 && *kp <= 9.0);
        }
    }

    #[test]
    fn test_get_ap_daily() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        let mjd = 36114.0;
        let ap = sw.get_ap_daily(mjd).unwrap();
        // Ap should be in valid range
        assert!((0.0..=400.0).contains(&ap));
    }

    #[test]
    fn test_get_f107() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        let mjd = 36114.0;
        let f107 = sw.get_f107_observed(mjd).unwrap();
        // F10.7 should be positive
        assert!(f107 > 0.0);
    }

    #[test]
    fn test_get_sunspot_number() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        let mjd = 36114.0;
        let isn = sw.get_sunspot_number(mjd).unwrap();
        // Should be a reasonable value
        assert!(isn < 500);
    }

    #[test]
    fn test_interval_index() {
        // 00:00 -> index 0
        assert_eq!(FileSpaceWeatherProvider::get_interval_index(36114.0), 0);
        // 01:30 -> index 0
        assert_eq!(FileSpaceWeatherProvider::get_interval_index(36114.0625), 0);
        // 03:00 -> index 1
        assert_eq!(FileSpaceWeatherProvider::get_interval_index(36114.125), 1);
        // 12:00 -> index 4
        assert_eq!(FileSpaceWeatherProvider::get_interval_index(36114.5), 4);
        // 23:00 -> index 7
        assert_abs_diff_eq!(
            FileSpaceWeatherProvider::get_interval_index(36114.0 + 23.0 / 24.0) as f64,
            7.0,
            epsilon = 0.1
        );
    }

    #[test]
    fn test_extrapolation_hold() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        // Request data before the start
        let mjd = sw.mjd_min() - 1.0;
        let result = sw.get_ap_daily(mjd);
        // Should return data (hold behavior)
        assert!(result.is_ok());
    }

    #[test]
    fn test_not_initialized() {
        let sw = FileSpaceWeatherProvider::new();
        assert!(!sw.is_initialized());
        assert!(sw.get_kp(60000.0).is_err());
    }

    #[test]
    fn test_display() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        let display = format!("{}", sw);
        assert!(display.contains("FileSpaceWeatherProvider"));
        assert!(display.contains("CSSI Space Weather"));
    }

    #[test]
    fn test_debug() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        let debug = format!("{:?}", sw);
        assert!(debug.contains("FileSpaceWeatherProvider"));
    }

    #[test]
    fn test_clone() {
        let sw1 = FileSpaceWeatherProvider::from_default_file().unwrap();
        let sw2 = sw1.clone();
        assert_eq!(sw1.len(), sw2.len());
        assert_eq!(sw1.mjd_min(), sw2.mjd_min());
    }

    #[test]
    fn test_get_last_kp_boundary_hours() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        let base_mjd = 60000.0;

        // Get all 8 Kp values for the day to compare
        let kp_all = sw.get_kp_all(base_mjd).unwrap();

        // Test each 3-hour boundary
        // At 00:00 (MJD.0), get_last_kp(1) should return the first interval's Kp
        let kp_00 = sw.get_last_kp(base_mjd, 1).unwrap();
        assert_abs_diff_eq!(kp_00[0], kp_all[0], epsilon = 1e-10);

        // At 03:00 (MJD + 3h = MJD + 0.125), get_last_kp(1) should return the second interval's Kp
        let kp_03 = sw.get_last_kp(base_mjd + 0.125, 1).unwrap();
        assert_abs_diff_eq!(kp_03[0], kp_all[1], epsilon = 1e-10);

        // At 06:00 (MJD + 0.25), should return third interval
        let kp_06 = sw.get_last_kp(base_mjd + 0.25, 1).unwrap();
        assert_abs_diff_eq!(kp_06[0], kp_all[2], epsilon = 1e-10);

        // At 12:00 (MJD + 0.5), should return fifth interval
        let kp_12 = sw.get_last_kp(base_mjd + 0.5, 1).unwrap();
        assert_abs_diff_eq!(kp_12[0], kp_all[4], epsilon = 1e-10);

        // At 21:00 (MJD + 0.875), should return eighth interval
        let kp_21 = sw.get_last_kp(base_mjd + 0.875, 1).unwrap();
        assert_abs_diff_eq!(kp_21[0], kp_all[7], epsilon = 1e-10);
    }

    #[test]
    fn test_get_last_ap_boundary_hours() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        let base_mjd = 60000.0;

        // Get all 8 Ap values for the day to compare
        let ap_all = sw.get_ap_all(base_mjd).unwrap();

        // Test boundaries like we did for Kp
        let ap_00 = sw.get_last_ap(base_mjd, 1).unwrap();
        assert_abs_diff_eq!(ap_00[0], ap_all[0], epsilon = 1e-10);

        let ap_03 = sw.get_last_ap(base_mjd + 0.125, 1).unwrap();
        assert_abs_diff_eq!(ap_03[0], ap_all[1], epsilon = 1e-10);

        let ap_12 = sw.get_last_ap(base_mjd + 0.5, 1).unwrap();
        assert_abs_diff_eq!(ap_12[0], ap_all[4], epsilon = 1e-10);
    }

    #[test]
    fn test_get_last_kp_crosses_day_boundary() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        let base_mjd = 60000.0;

        // Get last 10 values starting from 03:00 (second interval)
        // This should get: intervals 0,1 from today and 8 from yesterday
        let values = sw.get_last_kp(base_mjd + 0.125, 10).unwrap();
        assert_eq!(values.len(), 10);

        // Verify the last two values match today's first two intervals
        let kp_all_today = sw.get_kp_all(base_mjd).unwrap();
        assert_abs_diff_eq!(values[8], kp_all_today[0], epsilon = 1e-10);
        assert_abs_diff_eq!(values[9], kp_all_today[1], epsilon = 1e-10);

        // Verify the first values match yesterday's intervals
        let kp_all_yesterday = sw.get_kp_all(base_mjd - 1.0).unwrap();
        assert_abs_diff_eq!(values[0], kp_all_yesterday[0], epsilon = 1e-10);
        assert_abs_diff_eq!(values[7], kp_all_yesterday[7], epsilon = 1e-10);
    }

    #[test]
    fn test_get_last_daily_kp() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        let base_mjd = 60000.0;

        // Get last 3 daily averages
        let daily_kp = sw.get_last_daily_kp(base_mjd, 3).unwrap();
        assert_eq!(daily_kp.len(), 3);

        // Verify each matches the get_kp_daily for that day
        assert_abs_diff_eq!(
            daily_kp[0],
            sw.get_kp_daily(base_mjd - 2.0).unwrap(),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            daily_kp[1],
            sw.get_kp_daily(base_mjd - 1.0).unwrap(),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            daily_kp[2],
            sw.get_kp_daily(base_mjd).unwrap(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_get_last_daily_ap() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        let base_mjd = 60000.0;

        // Get last 3 daily averages
        let daily_ap = sw.get_last_daily_ap(base_mjd, 3).unwrap();
        assert_eq!(daily_ap.len(), 3);

        // Verify each matches the get_ap_daily for that day
        assert_abs_diff_eq!(
            daily_ap[0],
            sw.get_ap_daily(base_mjd - 2.0).unwrap(),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            daily_ap[1],
            sw.get_ap_daily(base_mjd - 1.0).unwrap(),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            daily_ap[2],
            sw.get_ap_daily(base_mjd).unwrap(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_get_last_f107() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        let base_mjd = 60000.0;

        // Get last 3 daily F10.7 values
        let f107 = sw.get_last_f107(base_mjd, 3).unwrap();
        assert_eq!(f107.len(), 3);

        // Verify each matches the get_f107_observed for that day
        assert_abs_diff_eq!(
            f107[0],
            sw.get_f107_observed(base_mjd - 2.0).unwrap(),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            f107[1],
            sw.get_f107_observed(base_mjd - 1.0).unwrap(),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            f107[2],
            sw.get_f107_observed(base_mjd).unwrap(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_get_last_kp_mid_interval() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        let base_mjd = 60000.0;

        // At 01:30 (MJD + 1.5/24 = MJD + 0.0625), still in first interval
        let kp_all = sw.get_kp_all(base_mjd).unwrap();
        let kp_mid = sw.get_last_kp(base_mjd + 0.0625, 1).unwrap();
        assert_abs_diff_eq!(kp_mid[0], kp_all[0], epsilon = 1e-10);

        // At 04:30 (MJD + 4.5/24 = MJD + 0.1875), in second interval
        let kp_mid2 = sw.get_last_kp(base_mjd + 0.1875, 1).unwrap();
        assert_abs_diff_eq!(kp_mid2[0], kp_all[1], epsilon = 1e-10);
    }

    #[test]
    fn test_kp_quantization() {
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
        let base_mjd = 60000.0;

        // Get several Kp values and verify they are properly quantized to 1/3 increments
        let kp_all = sw.get_kp_all(base_mjd).unwrap();

        for kp in kp_all.iter() {
            // Each Kp should be a multiple of 1/3
            let kp_times_3 = kp * 3.0;
            let rounded = kp_times_3.round();
            assert_abs_diff_eq!(kp_times_3, rounded, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_from_test_file() {
        // Load from the test file with known values
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = std::path::Path::new(&manifest_dir)
            .join("test_assets")
            .join("sw19571001.txt");

        let sw = FileSpaceWeatherProvider::from_file(&filepath, SpaceWeatherExtrapolation::Hold)
            .unwrap();

        assert!(sw.is_initialized());
        assert_eq!(sw.sw_type(), SpaceWeatherType::CssiSpaceWeather);
        assert_eq!(sw.extrapolation(), SpaceWeatherExtrapolation::Hold);
        // First data point is 1957-10-01 (MJD 36112)
        assert_eq!(sw.mjd_min(), 36112.0);
    }

    #[test]
    fn test_from_test_file_known_kp_values() {
        // Load from the test file and verify known Kp values for 1957-10-01
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = std::path::Path::new(&manifest_dir)
            .join("test_assets")
            .join("sw19571001.txt");

        let sw = FileSpaceWeatherProvider::from_file(&filepath, SpaceWeatherExtrapolation::Hold)
            .unwrap();

        // MJD 36112.0 = 1957-10-01
        let mjd = 36112.0;

        // Known Kp values from first data line: 43 40 30 20 37 23 43 37
        // These are stored as (integer * 10 + fraction), where fraction 3=+1/3, 7=+2/3
        // So 43 -> 4 + 1/3, 40 -> 4.0, 37 -> 3 + 2/3, 23 -> 2 + 1/3
        let kp_all = sw.get_kp_all(mjd).unwrap();
        assert_eq!(kp_all.len(), 8);
        assert_abs_diff_eq!(kp_all[0], 4.0 + 1.0 / 3.0, epsilon = 1e-10); // 43 -> 4+1/3
        assert_abs_diff_eq!(kp_all[1], 4.0, epsilon = 1e-10); // 40 -> 4.0
        assert_abs_diff_eq!(kp_all[2], 3.0, epsilon = 1e-10); // 30 -> 3.0
        assert_abs_diff_eq!(kp_all[3], 2.0, epsilon = 1e-10); // 20 -> 2.0
        assert_abs_diff_eq!(kp_all[4], 3.0 + 2.0 / 3.0, epsilon = 1e-10); // 37 -> 3+2/3
        assert_abs_diff_eq!(kp_all[5], 2.0 + 1.0 / 3.0, epsilon = 1e-10); // 23 -> 2+1/3
        assert_abs_diff_eq!(kp_all[6], 4.0 + 1.0 / 3.0, epsilon = 1e-10); // 43 -> 4+1/3
        assert_abs_diff_eq!(kp_all[7], 3.0 + 2.0 / 3.0, epsilon = 1e-10); // 37 -> 3+2/3
    }

    #[test]
    fn test_from_test_file_known_ap_values() {
        // Load from the test file and verify known Ap values for 1957-10-01
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = std::path::Path::new(&manifest_dir)
            .join("test_assets")
            .join("sw19571001.txt");

        let sw = FileSpaceWeatherProvider::from_file(&filepath, SpaceWeatherExtrapolation::Hold)
            .unwrap();

        // MJD 36112.0 = 1957-10-01
        let mjd = 36112.0;

        // Known Ap values: 32 27 15 7 22 9 32 22
        let ap_all = sw.get_ap_all(mjd).unwrap();
        assert_eq!(ap_all.len(), 8);
        assert_abs_diff_eq!(ap_all[0], 32.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ap_all[1], 27.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ap_all[2], 15.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ap_all[3], 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ap_all[4], 22.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ap_all[5], 9.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ap_all[6], 32.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ap_all[7], 22.0, epsilon = 1e-10);

        // Known Ap daily average: 21
        assert_abs_diff_eq!(sw.get_ap_daily(mjd).unwrap(), 21.0, epsilon = 1e-10);
    }

    #[test]
    fn test_from_test_file_known_f107_and_sunspot() {
        // Load from the test file and verify known F10.7 and sunspot values
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = std::path::Path::new(&manifest_dir)
            .join("test_assets")
            .join("sw19571001.txt");

        let sw = FileSpaceWeatherProvider::from_file(&filepath, SpaceWeatherExtrapolation::Hold)
            .unwrap();

        // MJD 36112.0 = 1957-10-01
        let mjd = 36112.0;

        // Known F10.7 adjusted: 269.8 (from data line)
        // Note: get_f107_observed returns the adjusted value in CSSI format
        assert_abs_diff_eq!(sw.get_f107_observed(mjd).unwrap(), 269.8, epsilon = 1e-10);

        // Known International Sunspot Number: 334
        assert_eq!(sw.get_sunspot_number(mjd).unwrap(), 334);
    }

    #[test]
    fn test_from_test_file_kp_daily() {
        // Test daily Kp average calculation from known values
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = std::path::Path::new(&manifest_dir)
            .join("test_assets")
            .join("sw19571001.txt");

        let sw = FileSpaceWeatherProvider::from_file(&filepath, SpaceWeatherExtrapolation::Hold)
            .unwrap();

        // MJD 36112.0 = 1957-10-01
        // The daily average uses the Kp sum from the file (273)
        // 273 / 10 / 8 = 27.3 / 8 = 3.4125
        let kp_daily = sw.get_kp_daily(36112.0).unwrap();
        let expected = 273.0 / 10.0 / 8.0;
        assert_abs_diff_eq!(kp_daily, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_section_boundaries() {
        // Test that section boundary MJDs are correctly set
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();

        // All boundaries should be set and in order
        assert!(sw.mjd_last_observed() > 0.0);
        assert!(sw.mjd_last_daily_predicted() >= sw.mjd_last_observed());
        assert!(sw.mjd_last_monthly_predicted() >= sw.mjd_last_daily_predicted());

        // Monthly predicted should extend further into the future
        assert!(sw.mjd_last_monthly_predicted() > sw.mjd_last_daily_predicted());
    }

    #[test]
    fn test_f107_in_monthly_predicted() {
        // Test that F10.7 data is available in MONTHLY_PREDICTED section
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();

        // Get an MJD in the MONTHLY_PREDICTED section
        let monthly_mjd = sw.mjd_last_daily_predicted() + 10.0;

        // F10.7 should work (returns actual predicted values)
        let f107 = sw.get_f107_observed(monthly_mjd);
        assert!(
            f107.is_ok(),
            "F10.7 should be available in MONTHLY_PREDICTED"
        );
        assert!(f107.unwrap() > 0.0, "F10.7 should be a positive value");
    }

    #[test]
    fn test_kp_ap_extrapolation_hold() {
        // Test that Kp/Ap in MONTHLY_PREDICTED section extrapolates from last daily
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();

        // Get the last daily predicted values
        let last_daily_mjd = sw.mjd_last_daily_predicted();
        let last_daily_kp = sw.get_kp(last_daily_mjd).unwrap();
        let last_daily_ap = sw.get_ap_daily(last_daily_mjd).unwrap();

        // Get values from MONTHLY_PREDICTED section
        let monthly_mjd = last_daily_mjd + 10.0;
        let monthly_kp = sw.get_kp(monthly_mjd).unwrap();
        let monthly_ap = sw.get_ap_daily(monthly_mjd).unwrap();

        // With Hold extrapolation, should get the last daily values
        assert_abs_diff_eq!(monthly_kp, last_daily_kp, epsilon = 1e-10);
        assert_abs_diff_eq!(monthly_ap, last_daily_ap, epsilon = 1e-10);
    }

    #[test]
    fn test_kp_ap_extrapolation_zero() {
        // Test that Kp/Ap in MONTHLY_PREDICTED section returns zero with Zero extrapolation
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = std::path::Path::new(&manifest_dir)
            .join("test_assets")
            .join("sw19571001.txt");

        let sw = FileSpaceWeatherProvider::from_file(&filepath, SpaceWeatherExtrapolation::Zero)
            .unwrap();

        // Get an MJD in the MONTHLY_PREDICTED section
        let monthly_mjd = sw.mjd_last_daily_predicted() + 10.0;

        // With Zero extrapolation, should get zeros
        let kp = sw.get_kp(monthly_mjd).unwrap();
        let ap_daily = sw.get_ap_daily(monthly_mjd).unwrap();

        assert_abs_diff_eq!(kp, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(ap_daily, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_kp_ap_extrapolation_error() {
        // Test that Kp/Ap in MONTHLY_PREDICTED section errors with Error extrapolation
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = std::path::Path::new(&manifest_dir)
            .join("test_assets")
            .join("sw19571001.txt");

        let sw = FileSpaceWeatherProvider::from_file(&filepath, SpaceWeatherExtrapolation::Error)
            .unwrap();

        // Get an MJD in the MONTHLY_PREDICTED section
        let monthly_mjd = sw.mjd_last_daily_predicted() + 10.0;

        // With Error extrapolation, should return an error
        let kp_result = sw.get_kp(monthly_mjd);
        assert!(
            kp_result.is_err(),
            "Should error when accessing Kp in MONTHLY_PREDICTED with Error mode"
        );
    }

    #[test]
    fn test_monthly_predicted_f107_not_nan() {
        // Verify that F10.7 values in MONTHLY_PREDICTED are actual values, not NaN
        let sw = FileSpaceWeatherProvider::from_default_file().unwrap();

        // Check multiple dates in the MONTHLY_PREDICTED section
        for i in 1..=5 {
            let monthly_mjd = sw.mjd_last_daily_predicted() + (i as f64 * 30.0);
            if monthly_mjd <= sw.mjd_max() {
                let f107 = sw.get_f107_observed(monthly_mjd).unwrap();
                assert!(
                    !f107.is_nan(),
                    "F10.7 at MJD {} should not be NaN",
                    monthly_mjd
                );
                assert!(
                    f107 > 0.0,
                    "F10.7 at MJD {} should be positive",
                    monthly_mjd
                );
            }
        }
    }
}
