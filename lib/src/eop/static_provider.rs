/*!
The static provider module implements an EOP provider that returns zero
or static values.
*/

use std::fmt;

use crate::eop::eop_provider::EarthOrientationProvider;
use crate::eop::types::{EOPType, EOPExtrapolation};

/// StaticEOPProvider is an EarthOrientationProvider that returns static
/// values for all EOP parameters at all times.
/// 
/// It can be initialized as zero-valued or with specific values. It will
/// never extrapolate or interpolate since the data is only for a single
/// time.
#[derive(Clone)]
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
            self.extrapolate(),
            self.interpolate()
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
            self.extrapolate(),
            self.interpolate()
        )
    }
}

impl StaticEOPProvider {
    /// Creates a new, uninitialized `StaticEOPProvider`
    /// 
    /// # Returns
    /// 
    /// * `StaticEOPProvider` - New `StaticEOPProvider` that is not initialized.
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
    pub fn from_zero() -> Self {
        StaticEOPProvider {
            initialized: true,
            data: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        }
    }

    /// Creates a new `StaticEOPProvider` with the given values for all EOP parameters.
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
    pub fn from_values(values: (f64, f64, f64, f64, f64, f64)) -> Self {
        StaticEOPProvider {
            initialized: true,
            data: values
        }
    }
}

impl EarthOrientationProvider for StaticEOPProvider {
    /// Returns the number of entries in the EOP data structure.
    fn len(&self) -> usize {
        1
    }

    /// Returns the type of EOP data stored in the data structure.
    fn eop_type(&self) -> EOPType {
        EOPType::Static
    }

    /// Returns the extrapolation method used by the EOP data structure.
    fn extrapolate(&self) -> EOPExtrapolation {
        EOPExtrapolation::Hold
    }

    /// Returns whether the EOP data structure supports interpolation.
    fn interpolate(&self) -> bool {
        false
    }

    /// Returns the minimum Modified Julian Date (MJD) supported by the EOP data structure.
    fn mjd_min(&self) -> u32 {
        0
    }

    /// Returns the maximum Modified Julian Date (MJD) supported by the EOP data structure.
    fn mjd_max(&self) -> u32 {
        u32::MAX
    }

    /// Returns the last Modified Julian Date (MJD) supported by the EOP data structure
    /// for which the length of day (LOD) is known.
    fn mjd_last_lod(&self) -> u32 {
        u32::MAX
    }

    /// Returns the last Modified Julian Date (MJD) supported by the EOP data structure
    /// for which celestial pole offsets (dX, dY) are known.
    fn mjd_last_dxdy(&self) -> u32 {
        u32::MAX
    }

    /// Returns the UT1-UTC offset for the given Modified Julian Date (MJD).
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the UT1-UTC offset for.
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - UT1-UTC offset in seconds.
    /// * `Err(String)` - Error message if the UT1-UTC offset could not be retrieved.
    fn get_ut1_utc(&self, _mjd: f64) -> Result<f64, String> {
        if self.initialized {
            Ok(self.data.2)
        } else {
            Err(String::from("EOP provider not initialized"))
        }
    }
    /// Returns the polar motion (PM) values for the given Modified Julian Date (MJD).
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the polar motion (PM) values for.
    ///
    /// # Returns
    ///
    /// * `Ok((f64, f64))` - Polar motion (PM) values in radians.
    /// * `Err(String)` - Error message if the polar motion (PM) values could not be retrieved.
    fn get_pm(&self, _mjd: f64) -> Result<(f64, f64), String> {
        if self.initialized {
            Ok((self.data.0, self.data.1))
        } else {
            Err(String::from("EOP provider not initialized"))
        }
    }

    /// Returns the Celestial Intermediate Pole (CIP) offset values for the given Modified Julian Date (MJD).
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the CIP offset values for.
    ///
    /// # Returns
    ///
    /// * `Ok((f64, f64))` - CIP offset values in radians.
    /// * `Err(String)` - Error message if the CIP offset values could not be retrieved.
    fn get_dxdy(&self, _mjd: f64) -> Result<(f64, f64), String> {
        if self.initialized {
            Ok((self.data.3, self.data.4))
        } else {
            Err(String::from("EOP provider not initialized"))
        }
    }

    /// Returns the length of day (LOD) value for the given Modified Julian Date (MJD).
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the LOD value for.
    ///
    /// # Returns
    ///
    /// * `Ok(f64)` - Length of day (LOD) value in seconds.
    /// * `Err(String)` - Error message if the LOD value could not be retrieved.
    fn get_lod(&self, _mjd: f64) -> Result<f64, String> {
        if self.initialized {
            Ok(self.data.5)
        } else {
            Err(String::from("EOP provider not initialized"))
        }
    }

    /// Returns the Earth orientation parameter (EOP) values for the given Modified Julian Date (MJD).
    ///
    /// # Arguments
    ///
    /// * `mjd` - Modified Julian Date (MJD) to retrieve the EOP values for.
    ///
    /// # Returns
    ///
    /// * `Ok((f64, f64, f64, f64, f64, f64))` - EOP values.
    /// * `Err(String)` - Error message if the EOP values could not be retrieved.
    fn get_eop(&self, _mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), String> {
        if self.initialized {
            Ok(self.data)
        } else {
            Err(String::from("EOP provider not initialized"))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::eop::types::*;

    use super::*;

    #[test]
    fn test_uninitialized_provider() {
        let provider = StaticEOPProvider::new();
        assert!(!provider.initialized);
    }

    #[test]
    fn test_from_zero() {
        let eop = StaticEOPProvider::from_zero();

        assert!(eop.initialized);
        assert_eq!(eop.len(), 1);
        assert_eq!(eop.mjd_min(), 0);
        assert_eq!(eop.mjd_max(), u32::MAX);
        assert_eq!(eop.eop_type(), EOPType::Static);
        assert_eq!(eop.extrapolate(), EOPExtrapolation::Hold);
        assert_eq!(eop.interpolate(), false);

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

        assert!(eop.initialized);
        assert_eq!(eop.len(), 1);
        assert_eq!(eop.mjd_min(), 0);
        assert_eq!(eop.mjd_max(), u32::MAX);
        assert_eq!(eop.eop_type(), EOPType::Static);
        assert_eq!(eop.extrapolate(), EOPExtrapolation::Hold);
        assert_eq!(eop.interpolate(), false);

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
        assert_eq!(result, Err(String::from("EOP provider not initialized")));
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
        assert_eq!(result, Err(String::from("EOP provider not initialized")));
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
        assert_eq!(result, Err(String::from("EOP provider not initialized")));
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
        assert_eq!(result, Err(String::from("EOP provider not initialized")));
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
        assert_eq!(result, Err(String::from("EOP provider not initialized")));
    }
}