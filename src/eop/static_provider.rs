/*!
The static provider module implements an EOP provider that returns zero
or static values.
*/

use std::fmt;

use crate::eop::eop_provider::EarthOrientationProvider;
use crate::eop::eop_types::{EOPExtrapolation, EOPType};
use crate::utils::errors::BraheError;

/// StaticEOPProvider is an EarthOrientationProvider that returns static
/// values for all EOP parameters at all times.
///
/// It can be initialized as zero-valued or with specific values. It will
/// never extrapolate or interpolate since the data is only for a single
/// time.
#[derive(Clone, Copy)]
pub struct StaticEOPProvider {
    /// Internal variable to indicate whether the Earth Orietnation data Object
    /// has been properly initialized
    initialized: bool,
    /// Primary data structure storing loaded Earth orientation parameter data.
    ///
    /// Key:
    /// - `mjd`: Modified Julian date of the parameter values
    ///
    /// Values:
    /// - `pm_x`: x-component of polar motion correction. Units: (radians)
    /// - `pm_y`: y-component of polar motion correction. Units: (radians)
    /// - `ut1_utc`: Offset of UT1 time scale from UTC time scale. Units: (seconds)
    /// - `dX`: "X" component of Celestial Intermediate Pole (CIP) offset. Units: (radians)
    /// - `dY`: "Y" component of Celestial Intermediate Pole (CIP) offset. Units: (radians)
    /// - `lod`: Difference between astronomically determined length of day and 86400 second TAI.Units: (seconds)
    ///   day. Units: (seconds)
    data: (f64, f64, f64, f64, f64, f64),
}

impl fmt::Display for StaticEOPProvider {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "StaticEOPProvider - type: {}, {} entries, mjd_min: {}, mjd_max: {},  mjd_last_lod: \
        {}, mjd_last_dxdy: {}, extrapolate: {}, \
        interpolate: {}",
            self.eop_type(),
            self.len(),
            self.mjd_min(),
            self.mjd_max(),
            self.mjd_last_lod(),
            self.mjd_last_dxdy(),
            self.extrapolation(),
            self.interpolation()
        )
    }
}

impl fmt::Debug for StaticEOPProvider {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "StaticEOPProvider<Type: {}, Length: {}, mjd_min: {}, mjd_max: {},  mjd_last_lod: \
        {}, mjd_last_dxdy: {}, extrapolate: {}, interpolate: {}>",
            self.eop_type(),
            self.len(),
            self.mjd_min(),
            self.mjd_max(),
            self.mjd_last_lod(),
            self.mjd_last_dxdy(),
            self.extrapolation(),
            self.interpolation()
        )
    }
}

impl Default for StaticEOPProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl StaticEOPProvider {
    /// Creates a new, uninitialized `StaticEOPProvider` with zero values for all EOP parameters.
    /// This is the default constructor. It is not recommended to use this constructor unless a
    /// placeholder `StaticEOPProvider` allocation is needed.
    ///
    /// # Returns
    ///
    /// * `StaticEOPProvider` - New `StaticEOPProvider` that is not initialized.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{StaticEOPProvider, EarthOrientationProvider};
    ///
    /// let eop = StaticEOPProvider::new();
    /// assert_eq!(eop.is_initialized(), false);
    /// ```
    pub fn new() -> Self {
        StaticEOPProvider {
            initialized: false,
            data: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        }
    }

    /// Creates a new `StaticEOPProvider` with zero values for all EOP parameters.
    ///
    /// # Returns
    ///
    /// * `StaticEOPProvider` - New `StaticEOPProvider` that returns 0 for all EOP parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{StaticEOPProvider, EarthOrientationProvider};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    /// assert_eq!(eop.is_initialized(), true);
    /// ```
    pub fn from_zero() -> Self {
        StaticEOPProvider {
            initialized: true,
            data: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        }
    }

    /// Creates a new `StaticEOPProvider` with the given values for all EOP parameters. The
    /// static values will be returned for all times.
    ///
    /// # Arguments
    ///
    /// * `pm_x` - x-component of polar motion correction. Units: (radians)
    /// * `pm_y` - y-component of polar motion correction. Units: (radians)
    /// * `ut1_utc` - Offset of UT1 time scale from UTC time scale. Units: (seconds)
    /// * `dX` - "X" component of Celestial Intermediate Pole (CIP) offset. Units: (radians)
    /// * `dY` - "Y" component of Celestial Intermediate Pole (CIP) offset. Units: (radians)
    /// * `lod` - Difference between astronomically determined length of day and 86400 second TAI.Units: (seconds)
    ///   day. Units: (seconds)
    ///
    /// # Returns
    ///
    /// * `StaticEOPProvider` - New `StaticEOPProvider` with the given values for all EOP parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{StaticEOPProvider, EarthOrientationProvider};
    ///
    /// let eop = StaticEOPProvider::from_values((0.1, 0.2, 0.3, 0.4, 0.5, 0.6));
    /// assert_eq!(eop.is_initialized(), true);
    /// ```
    pub fn from_values(values: (f64, f64, f64, f64, f64, f64)) -> Self {
        StaticEOPProvider {
            initialized: true,
            data: values,
        }
    }
}

impl EarthOrientationProvider for StaticEOPProvider {
    /// Returns the initialization status of the EOP data structure. Value is `true` if the
    /// EOP data structure has been properly initialized and `false` otherwise.
    ///
    /// # Returns
    ///
    /// * `bool` - `true` if the EOP data structure has been properly initialized, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{StaticEOPProvider, EarthOrientationProvider};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    /// assert_eq!(eop.is_initialized(), true);
    /// ```
    fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Returns the number of entries in the EOP data structure. For static EOP data structures,
    /// this will always be 1.
    ///
    /// # Returns
    ///
    /// * `usize` - Number of entries in the EOP data structure.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{StaticEOPProvider, EarthOrientationProvider};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    /// assert_eq!(eop.len(), 1);
    /// ```
    fn len(&self) -> usize {
        1
    }

    /// Returns the type of EOP data stored in the data structure. See the `EOPType` enum for
    /// possible values.
    ///
    /// # Returns
    ///
    /// * `EOPType` - Type of EOP data stored in the data structure.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{StaticEOPProvider, EarthOrientationProvider, EOPType};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    /// assert_eq!(eop.eop_type(), EOPType::Static);
    /// ```
    fn eop_type(&self) -> EOPType {
        EOPType::Static
    }

    /// Returns the extrapolation method used by the EOP data structure. See the `EOPExtrapolation`
    /// enum for possible values.
    ///
    /// # Returns
    ///
    /// * `EOPExtrapolation` - Extrapolation method used by the EOP data structure.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{StaticEOPProvider, EarthOrientationProvider, EOPExtrapolation};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    ///
    /// assert_eq!(eop.extrapolation(), EOPExtrapolation::Hold);
    /// ```
    fn extrapolation(&self) -> EOPExtrapolation {
        EOPExtrapolation::Hold
    }

    /// Returns whether the EOP data structure supports interpolation. Returns `false` for static
    /// EOP data structures.
    ///
    /// # Returns
    ///
    /// * `bool` - `true` if the EOP data structure supports interpolation, `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{StaticEOPProvider, EarthOrientationProvider};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    ///
    /// assert_eq!(eop.interpolation(), false);
    /// ```
    fn interpolation(&self) -> bool {
        false
    }

    /// Returns the minimum Modified Julian Date (MJD) supported by the EOP data structure.
    /// For static EOP data structures, this will always be 0.0.
    ///
    /// # Returns
    ///
    /// * `f64` - Minimum Modified Julian Date (MJD) supported by the EOP data structure.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{StaticEOPProvider, EarthOrientationProvider};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    ///
    /// assert_eq!(eop.mjd_min(), 0.0);
    /// ```
    fn mjd_min(&self) -> f64 {
        0.0
    }

    /// Returns the maximum Modified Julian Date (MJD) supported by the EOP data structure.
    /// For static EOP data structures, this will always be `f64::MAX`.
    ///
    /// # Returns
    ///
    /// * `f64` - Maximum Modified Julian Date (MJD) supported by the EOP data structure.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{StaticEOPProvider, EarthOrientationProvider};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    ///
    /// assert_eq!(eop.mjd_max(), f64::MAX);
    /// ```
    fn mjd_max(&self) -> f64 {
        f64::MAX
    }

    /// Returns the last Modified Julian Date (MJD) supported by the EOP data structure
    /// for which the length of day (LOD) is known. For static EOP data structures, this
    /// will always be `f64::MAX`.
    ///
    /// # Returns
    ///
    /// * `f64` - Last Modified Julian Date (MJD) supported by the EOP data structure
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{StaticEOPProvider, EarthOrientationProvider};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    ///
    /// assert_eq!(eop.mjd_last_lod(), f64::MAX);
    /// ```
    fn mjd_last_lod(&self) -> f64 {
        f64::MAX
    }

    /// Returns the last Modified Julian Date (MJD) supported by the EOP data structure
    /// for which celestial pole offsets (dX, dY) are known. For static EOP data structures, this
    /// will always be `f64::MAX`.
    ///
    /// # Returns
    ///
    /// * `f64` - Last Modified Julian Date (MJD) supported by the EOP data structure
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{StaticEOPProvider, EarthOrientationProvider};
    ///
    /// let eop = StaticEOPProvider::from_zero();
    ///
    /// assert_eq!(eop.mjd_last_dxdy(), f64::MAX);
    /// ```
    fn mjd_last_dxdy(&self) -> f64 {
        f64::MAX
    }

    /// Returns the UT1-UTC offset for the given Modified Julian Date (MJD). For static EOP data
    /// structures, this will always be the same value.
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the UT1-UTC offset for.
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - UT1-UTC offset in seconds.
    /// * `Err(String)` - Error message if the UT1-UTC offset could not be retrieved.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{StaticEOPProvider, EarthOrientationProvider};
    ///
    /// let eop = StaticEOPProvider::from_values((0.1, 0.2, 0.3, 0.4, 0.5, 0.6));
    ///
    /// assert_eq!(eop.get_ut1_utc(59950.0).unwrap(), 0.3);
    ///
    /// // Can also use any MJD value since the EOP data is static
    /// assert_eq!(eop.get_ut1_utc(0.0).unwrap(), 0.3);
    /// ```
    fn get_ut1_utc(&self, _mjd: f64) -> Result<f64, BraheError> {
        if self.initialized {
            Ok(self.data.2)
        } else {
            Err(BraheError::EOPError(String::from(
                "EOP provider not initialized",
            )))
        }
    }

    /// Returns the polar motion (PM) values for the given Modified Julian Date (MJD). For static EOP data
    /// structures, this will always be the same value.
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the polar motion (PM) values for.
    ///
    /// # Returns
    ///
    /// * `Ok((f64, f64))` - Polar motion (PM) values in radians.
    /// * `Err(String)` - Error message if the polar motion (PM) values could not be retrieved.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{StaticEOPProvider, EarthOrientationProvider};
    ///
    /// let eop = StaticEOPProvider::from_values((0.1, 0.2, 0.3, 0.4, 0.5, 0.6));
    ///
    /// assert_eq!(eop.get_pm(59950.0).unwrap().0, 0.1);
    /// assert_eq!(eop.get_pm(59950.0).unwrap().1, 0.2);
    ///
    /// // Can also use any MJD value since the EOP data is static
    /// assert_eq!(eop.get_pm(0.0).unwrap().0, 0.1);
    /// assert_eq!(eop.get_pm(0.0).unwrap().1, 0.2);
    /// ```
    fn get_pm(&self, _mjd: f64) -> Result<(f64, f64), BraheError> {
        if self.initialized {
            Ok((self.data.0, self.data.1))
        } else {
            Err(BraheError::EOPError(String::from(
                "EOP provider not initialized",
            )))
        }
    }

    /// Returns the Celestial Intermediate Pole (CIP) offset values for the given Modified Julian Date (MJD).
    /// For static EOP data structures, this will always be the same value.
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the CIP offset values for.
    ///
    /// # Returns
    ///
    /// * `Ok((f64, f64))` - CIP offset values in radians.
    /// * `Err(String)` - Error message if the CIP offset values could not be retrieved.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{StaticEOPProvider, EarthOrientationProvider};
    ///
    /// let eop = StaticEOPProvider::from_values((0.1, 0.2, 0.3, 0.4, 0.5, 0.6));
    ///
    /// assert_eq!(eop.get_dxdy(59950.0).unwrap().0, 0.4);
    /// assert_eq!(eop.get_dxdy(59950.0).unwrap().1, 0.5);
    ///
    /// // Can also use any MJD value since the EOP data is static
    ///
    /// assert_eq!(eop.get_dxdy(0.0).unwrap().0, 0.4);
    /// assert_eq!(eop.get_dxdy(0.0).unwrap().1, 0.5);
    /// ```
    fn get_dxdy(&self, _mjd: f64) -> Result<(f64, f64), BraheError> {
        if self.initialized {
            Ok((self.data.3, self.data.4))
        } else {
            Err(BraheError::EOPError(String::from(
                "EOP provider not initialized",
            )))
        }
    }

    /// Returns the length of day (LOD) value for the given Modified Julian Date (MJD). For static EOP data
    /// structures, this will always be the same value.
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the LOD value for.
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - Length of day (LOD) value in seconds.
    /// * `Err(String)` - Error message if the LOD value could not be retrieved.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{StaticEOPProvider, EarthOrientationProvider};
    ///
    /// let eop = StaticEOPProvider::from_values((0.1, 0.2, 0.3, 0.4, 0.5, 0.6));
    ///
    /// assert_eq!(eop.get_lod(59950.0).unwrap(), 0.6);
    ///
    /// // Can also use any MJD value since the EOP data is static
    /// assert_eq!(eop.get_lod(0.0).unwrap(), 0.6);
    /// ```
    fn get_lod(&self, _mjd: f64) -> Result<f64, BraheError> {
        if self.initialized {
            Ok(self.data.5)
        } else {
            Err(BraheError::EOPError(String::from(
                "EOP provider not initialized",
            )))
        }
    }

    /// Returns the Earth orientation parameter (EOP) values for the given Modified Julian Date (MJD).
    /// For static EOP data structures, this will always be the same value.
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the EOP values for.
    ///
    /// # Returns
    ///
    /// * `Ok((f64, f64, f64, f64, f64, f64))` - EOP values.
    /// * `Err(String)` - Error message if the EOP values could not be retrieved.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{StaticEOPProvider, EarthOrientationProvider};
    ///
    /// let eop = StaticEOPProvider::from_values((0.1, 0.2, 0.3, 0.4, 0.5, 0.6));
    ///
    /// assert_eq!(eop.get_eop(59950.0).unwrap(), (0.1, 0.2, 0.3, 0.4, 0.5, 0.6));
    ///
    /// // Can also use any MJD value since the EOP data is static
    /// assert_eq!(eop.get_eop(0.0).unwrap(), (0.1, 0.2, 0.3, 0.4, 0.5, 0.6));
    /// ```
    fn get_eop(&self, _mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), BraheError> {
        if self.initialized {
            Ok(self.data)
        } else {
            Err(BraheError::EOPError(String::from(
                "EOP provider not initialized",
            )))
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use crate::eop::eop_types::*;

    use super::*;

    #[test]
    fn test_uninitialized_provider() {
        let provider = StaticEOPProvider::new();
        assert!(!provider.is_initialized());
    }

    #[test]
    fn test_from_zero() {
        let eop = StaticEOPProvider::from_zero();

        assert!(eop.is_initialized());
        assert_eq!(eop.len(), 1);
        assert_eq!(eop.mjd_min(), 0.0);
        assert_eq!(eop.mjd_max(), f64::MAX);
        assert_eq!(eop.eop_type(), EOPType::Static);
        assert_eq!(eop.extrapolation(), EOPExtrapolation::Hold);
        assert!(!eop.interpolation());

        // EOP Values
        assert_eq!(eop.get_ut1_utc(59950.0).unwrap(), 0.0);
        assert_eq!(eop.get_pm(59950.0).unwrap().0, 0.0);
        assert_eq!(eop.get_pm(59950.0).unwrap().1, 0.0);
        assert_eq!(eop.get_dxdy(59950.0).unwrap().0, 0.0);
        assert_eq!(eop.get_dxdy(59950.0).unwrap().1, 0.0);
        assert_eq!(eop.get_lod(59950.0).unwrap(), 0.0);
    }

    #[test]
    fn test_from_values() {
        let eop = StaticEOPProvider::from_values((0.001, 0.002, 0.003, 0.004, 0.005, 0.006));

        assert!(eop.is_initialized());
        assert_eq!(eop.len(), 1);
        assert_eq!(eop.mjd_min(), 0.0);
        assert_eq!(eop.mjd_max(), f64::MAX);
        assert_eq!(eop.eop_type(), EOPType::Static);
        assert_eq!(eop.extrapolation(), EOPExtrapolation::Hold);
        assert!(!eop.interpolation());

        // EOP Values
        assert_eq!(eop.get_pm(59950.0).unwrap().0, 0.001);
        assert_eq!(eop.get_pm(59950.0).unwrap().1, 0.002);
        assert_eq!(eop.get_ut1_utc(59950.0).unwrap(), 0.003);
        assert_eq!(eop.get_dxdy(59950.0).unwrap().0, 0.004);
        assert_eq!(eop.get_dxdy(59950.0).unwrap().1, 0.005);
        assert_eq!(eop.get_lod(59950.0).unwrap(), 0.006);
    }

    #[test]
    fn test_get_ut1_utc() {
        let provider = StaticEOPProvider::from_values((0.1, 0.2, 0.3, 0.4, 0.5, 0.6));
        let result = provider.get_ut1_utc(2459455.5);
        assert_eq!(result, Ok(0.3));
    }

    #[test]
    fn test_get_ut1_utc_error() {
        let provider = StaticEOPProvider::new();
        let result = provider.get_ut1_utc(2459455.5);
        assert_eq!(
            result,
            Err(BraheError::EOPError(String::from(
                "EOP provider not initialized"
            )))
        );
    }

    #[test]
    fn test_get_pm() {
        let provider = StaticEOPProvider::from_values((0.1, 0.2, 0.3, 0.4, 0.5, 0.6));
        let result = provider.get_pm(2459455.5);
        assert_eq!(result, Ok((0.1, 0.2)));
    }

    #[test]
    fn test_get_pm_error() {
        let provider = StaticEOPProvider::new();
        let result = provider.get_pm(2459455.5);
        assert_eq!(
            result,
            Err(BraheError::EOPError(String::from(
                "EOP provider not initialized"
            )))
        );
    }

    #[test]
    fn test_get_dxdy() {
        let provider = StaticEOPProvider::from_values((0.1, 0.2, 0.3, 0.4, 0.5, 0.6));
        let result = provider.get_dxdy(2459455.5);
        assert_eq!(result, Ok((0.4, 0.5)));
    }

    #[test]
    fn test_get_dxdy_error() {
        let provider = StaticEOPProvider::new();
        let result = provider.get_dxdy(2459455.5);
        assert_eq!(
            result,
            Err(BraheError::EOPError(String::from(
                "EOP provider not initialized"
            )))
        );
    }

    #[test]
    fn test_get_lod() {
        let provider = StaticEOPProvider::from_values((0.1, 0.2, 0.3, 0.4, 0.5, 0.6));
        let result = provider.get_lod(2459455.5);
        assert_eq!(result, Ok(0.6));
    }

    #[test]
    fn test_get_lod_error() {
        let provider = StaticEOPProvider::new();
        let result = provider.get_lod(2459455.5);
        assert_eq!(
            result,
            Err(BraheError::EOPError(String::from(
                "EOP provider not initialized"
            )))
        );
    }

    #[test]
    fn test_get_eop() {
        let provider = StaticEOPProvider::from_values((0.1, 0.2, 0.3, 0.4, 0.5, 0.6));
        let result = provider.get_eop(2459455.5);
        assert_eq!(result, Ok((0.1, 0.2, 0.3, 0.4, 0.5, 0.6)));
    }

    #[test]
    fn test_get_eop_error() {
        let provider = StaticEOPProvider::new();
        let result = provider.get_eop(2459455.5);
        assert_eq!(
            result,
            Err(BraheError::EOPError(String::from(
                "EOP provider not initialized"
            )))
        );
    }

    #[test]
    fn test_default_implementation() {
        // Test that Default::default() is equivalent to new()
        let eop_default = StaticEOPProvider::default();
        let eop_new = StaticEOPProvider::new();

        // Both should be uninitialized
        assert_eq!(eop_default.is_initialized(), eop_new.is_initialized());
        assert!(!eop_default.is_initialized());

        // Both should have the same type
        assert_eq!(eop_default.eop_type(), eop_new.eop_type());
        assert_eq!(eop_default.eop_type(), EOPType::Static);

        // Both should have the same length
        assert_eq!(eop_default.len(), eop_new.len());
        assert_eq!(eop_default.len(), 1);
    }

    #[test]
    fn test_display_format() {
        let eop = StaticEOPProvider::from_values((0.001, 0.002, 0.003, 0.004, 0.005, 0.006));
        let display_string = format!("{}", eop);

        // Verify key information is present in the display string
        assert!(display_string.contains("StaticEOPProvider"));
        assert!(display_string.contains("Static"));
        assert!(display_string.contains("1"));
        assert!(display_string.contains("entries"));
        assert!(display_string.contains("EOPExtrapolation::Hold"));
        assert!(display_string.contains("false"));
    }

    #[test]
    fn test_debug_format() {
        let eop = StaticEOPProvider::from_values((0.001, 0.002, 0.003, 0.004, 0.005, 0.006));
        let debug_string = format!("{:?}", eop);

        // Verify key information is present in the debug string
        assert!(debug_string.contains("StaticEOPProvider"));
        assert!(debug_string.contains("Static"));
        assert!(debug_string.contains("Length"));
        assert!(debug_string.contains("EOPExtrapolation::Hold"));
        assert!(debug_string.contains("false"));
    }

    #[test]
    fn test_mjd_last_lod() {
        // Test that static provider returns f64::MAX for mjd_last_lod
        let eop = StaticEOPProvider::from_zero();
        assert_eq!(eop.mjd_last_lod(), f64::MAX);

        // Also test with custom values
        let eop2 = StaticEOPProvider::from_values((0.001, 0.002, 0.003, 0.004, 0.005, 0.006));
        assert_eq!(eop2.mjd_last_lod(), f64::MAX);
    }

    #[test]
    fn test_mjd_last_dxdy() {
        // Test that static provider returns f64::MAX for mjd_last_dxdy
        let eop = StaticEOPProvider::from_zero();
        assert_eq!(eop.mjd_last_dxdy(), f64::MAX);

        // Also test with custom values
        let eop2 = StaticEOPProvider::from_values((0.001, 0.002, 0.003, 0.004, 0.005, 0.006));
        assert_eq!(eop2.mjd_last_dxdy(), f64::MAX);
    }
}
