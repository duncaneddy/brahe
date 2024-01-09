/*!
 * Defines crate-wide EOP loading functionality
 */

use once_cell::sync::Lazy;
use std::sync::{Arc, RwLock};

use crate::eop::types::{EOPExtrapolation, EOPType};
use crate::eop::eop_provider::EarthOrientationProvider;
use crate::eop::static_provider::StaticEOPProvider;
use crate::utils::BraheError;

static GLOBAL_EOP: Lazy<Arc<RwLock<Box<dyn EarthOrientationProvider + Sync + Send>>>> = Lazy::new(|| {
    Arc::new(RwLock::new(Box::new(StaticEOPProvider::new())))
});

/// Set the crate-wide static Earth orientation data provider. This function should be called
/// before any other function in the crate which accesses the global Earth orientation data.
/// If this function is not called, the crate-wide static Earth orientation data provider will
/// not be initialized and any function which accesses it will panic.
/// 
/// The global provider can be set to any object which implements the `EarthOrientationProvider`
/// trait. This includes the `StaticEOPProvider` and `FileEOPProvider` objects. The global
/// provider can also be set to a custom object which implements the `EarthOrientationProvider`
/// trait.
/// 
/// # Arguments
/// 
/// - `provider`: Object which implements the `EarthOrientationProvider` trait
/// 
/// # Examples
/// 
/// ```
/// use brahe::eop::*;
/// 
/// // Initialize Global EOP from StaticEOPProvider
/// 
/// let eop = StaticEOPProvider::from_zero();
/// set_global_eop_provider(eop);
/// 
/// // Initialize Global EOP from FileEOPProvider
/// 
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
/// ```
pub fn set_global_eop_provider<T: EarthOrientationProvider + Sync + Send + 'static>(provider: T) {
    *GLOBAL_EOP.write().unwrap() = Box::new(provider);
}

/// Get UT1-UTC offset set for specified date from the crate-wide static Earth orientation data
/// provider. The crate-wide provider must be initialized before this function is called or
/// the function will panic.
///
/// Function will return the UT1-UTC time scale for the given date.
/// Function is guaranteed to return a value. If the request value is beyond the end of the
/// loaded Earth orientation data set the behavior is specified by the `extrapolate` setting of
/// the underlying `EarthOrientationData` object. The possible behaviors for the returned
/// data are:
/// - `Zero`: Returned values will be `0.0` where data is not available
/// - `Hold`: Will return the last available returned value when data is not available
/// - `Error`: Function call will panic and terminate the program
///
/// If the date is in between data points, which typically are at integer day intervals, the
/// function will linearly interpolate between adjacent data points if `interpolate` was set
/// to `true` for the `EarthOrientationData` object or will return the value from the most
/// recent data point if `false`.
///
/// # Arguments
/// - `mjd`: Modified Julian date to get Earth orientation parameters for
///
/// # Returns
/// - `ut1_utc`: Offset of UT1 time scale from UTC time scale. Units: (seconds)
///
/// # Examples
/// 
/// ```
/// use brahe::eop::*;
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Access UT1-UTC offset value at specific date
/// let ut1_utc = get_global_ut1_utc(59422.0).unwrap();
/// ```
pub fn get_global_ut1_utc(mjd: f64) -> Result<f64, BraheError> {
    GLOBAL_EOP.read().unwrap().get_ut1_utc(mjd)
}

/// Get polar motion offset set for specified date from the crate-wide static Earth orientation data
/// provider. The crate-wide provider must be initialized before this function is called or
/// the function will panic.
///
/// Function will return the pm-x and pm-y for the given date.
/// Function is guaranteed to return a value. If the request value is beyond the end of the
/// loaded Earth orientation data set the behavior is specified by the `extrapolate` setting of
/// the underlying `EarthOrientationData` object. The possible behaviors for the returned
/// data are:
/// - `Zero`: Returned values will be `0.0` where data is not available
/// - `Hold`: Will return the last available returned value when data is not available
/// - `Error`: Function call will panic and terminate the program
///
/// If the date is in between data points, which typically are at integer day intervals, the
/// function will linearly interpolate between adjacent data points if `interpolate` was set
/// to `true` for the `EarthOrientationData` object or will return the value from the most
/// recent data point if `false`.
///
/// # Arguments
/// - `mjd`: Modified Julian date to get Earth orientation parameters for
///
/// # Returns
/// - `pm_x`: x-component of polar motion correction. Units: (radians)
/// - `pm_y`: y-component of polar motion correction. Units: (radians)
///
/// # Examples
/// 
/// ```
/// use brahe::eop::*;
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Get polar motion x and y values for 36 hours before the end of the table
/// let (pm_x, pm_y) = get_global_pm(59422.0).unwrap();
/// ```
pub fn get_global_pm(mjd: f64) -> Result<(f64, f64), BraheError> {
    GLOBAL_EOP.read().unwrap().get_pm(mjd)
}

/// Get precession-nutation for specified date from the crate-wide static Earth orientation data
/// provider. The crate-wide provider must be initialized before this function is called or
/// the function will panic.
///
/// Function will return the dX and dY for the given date.
/// Function is guaranteed to return a value. If the request value is beyond the end of the
/// loaded Earth orientation data set the behavior is specified by the `extrapolate` setting of
/// the underlying `EarthOrientationData` object. The possible behaviors for the returned
/// data are:
/// - `Zero`: Returned values will be `0.0` where data is not available
/// - `Hold`: Will return the last available returned value when data is not available
/// - `Error`: Function call will panic and terminate the program
///
/// If the date is in between data points, which typically are at integer day intervals, the
/// function will linearly interpolate between adjacent data points if `interpolate` was set
/// to `true` for the `EarthOrientationData` object or will return the value from the most
/// recent data point if `false`.
///
/// # Arguments
/// - `mjd`: Modified Julian date to get Earth orientation parameters for
///
/// # Returns
/// - `dX`: "X" component of Celestial Intermediate Pole (CIP) offset. Units: (radians)
/// - `dY`: "Y" component of Celestial Intermediate Pole (CIP) offset. Units: (radians)
///
/// # Examples
/// 
/// ```
/// use brahe::eop::*;
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Get dX and dY for 36 hours before the end of the table
/// let (dx, dy) = get_global_dxdy(59422.0).unwrap();
/// ```
pub fn get_global_dxdy(mjd: f64) -> Result<(f64, f64), BraheError> {
    GLOBAL_EOP.read().unwrap().get_dxdy(mjd)
}

/// Get length of day offset set for specified date from the crate-wide static Earth orientation data
/// provider. The crate-wide provider must be initialized before this function is called or
/// the function will panic.
///
/// Function will return the LOD offset for the given date.
/// Function is guaranteed to return a value. If the request value is beyond the end of the
/// loaded Earth orientation data set the behavior is specified by the `extrapolate` setting of
/// the underlying `EarthOrientationData` object. The possible behaviors for the returned
/// data are:
/// - `Zero`: Returned values will be `0.0` where data is not available
/// - `Hold`: Will return the last available returned value when data is not available
/// - `Error`: Function call will panic and terminate the program
///
/// If the date is in between data points, which typically are at integer day intervals, the
/// function will linearly interpolate between adjacent data points if `interpolate` was set
/// to `true` for the `EarthOrientationData` object or will return the value from the most
/// recent data point if `false`.
///
/// # Arguments
/// - `mjd`: Modified Julian date to get Earth orientation parameters for
///
/// # Returns
/// - `lod`: Difference between length of astronomically determined solar day and 86400 second
///     TAI day. Units: (seconds)
///
/// # Examples
/// 
/// ```
/// use brahe::eop::*;
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Get LOD for 36 hours before the end of the table
/// let lod = get_global_lod(59422.0).unwrap();
/// ```
pub fn get_global_lod(mjd: f64) -> Result<f64, BraheError> {
    GLOBAL_EOP.read().unwrap().get_lod(mjd)
}

/// Get Earth orientation parameter set for specified date from the crate-wide static Earth orientation data
/// provider. The crate-wide provider must be initialized before this function is called or
/// the function will panic.
///
/// Function will return the full set of Earth orientation parameters for the given date.
/// Function is guaranteed to provide the full set of Earth Orientation parameters according
/// to the behavior specified by the `extrapolate` setting of the underlying
/// `EarthOrientationData` object. The possible behaviors for the returned data are:
/// - `Zero`: Returned values will be `0.0` where data is not available
/// - `Hold`: Will return the last available returned value when data is not available
/// - `Error`: Function call will panic and terminate the program
///
/// Note, if the type is `Hold` for an StandardBulletinB file which does not contain LOD data
/// a value of `0.0` for LOD will be returned instead.
///
/// If the date is in between data points, which typically are at integer day intervals, the
/// function will linearly interpolate between adjacent data points if `interpolate` was set
/// to `true` for the `EarthOrientationData` object or will return the value from the most
/// recent data point if `false`.
///
/// # Arguments
/// - `mjd`: Modified Julian date to get Earth orientation parameters for
///
/// # Returns
/// - `pm_x`: x-component of polar motion correction. Units: (radians)
/// - `pm_y`: y-component of polar motion correction. Units: (radians)
/// - `ut1_utc`: Offset of UT1 time scale from UTC time scale. Units: (seconds)
/// - `dX`: "X" component of Celestial Intermediate Pole (CIP) offset. Units: (radians)
/// - `dY`: "Y" component of Celestial Intermediate Pole (CIP) offset. Units: (radians)
/// - `lod`: Difference between length of astronomically determined solar day and 86400 second
///    TAI day. Units: (seconds)
///
/// # Examples
/// 
/// ```
/// use brahe::eop::*;
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Get EOP for 36 hours before the end of the table
/// let eop_params = get_global_eop(59422.0).unwrap();
/// ```
#[allow(non_snake_case)]
pub fn get_global_eop(mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), BraheError> {
    GLOBAL_EOP.read().unwrap().get_eop(mjd)
}

/// Returns initialzation state of global Earth orientation data provider.
///
/// # Returns
/// 
/// - `intialized`: Boolean, which if `true` indicates that the global static variable has been properly initialized.
///
/// # Examples
/// 
/// ```
/// use brahe::eop::*;
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// assert_eq!(get_global_eop_initialization(), true);
/// ```
pub fn get_global_eop_initialization() -> bool {
    GLOBAL_EOP.read().unwrap().is_initialized()
}

/// Return length of loaded global Earth orientation data provider.
///
/// # Returns
/// 
/// - `len`: length of number of loaded EOP data points
///
/// # Examples
/// 
/// ```
/// use brahe::eop::*;
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Confirm initialization complete
/// assert!(get_global_eop_len() >= 10000);
/// ```
pub fn get_global_eop_len() -> usize {
    GLOBAL_EOP.read().unwrap().len()
}

/// Returns the type of loaded EarthOrientationData provider in the global Earth orientation data provider.
/// See the `EOPType` enum for possible values.
///
/// # Returns
/// 
/// - `eop_type`: Type of loaded Earth Orientation data
///
/// # Examples
/// 
/// ```
/// use brahe::eop::*;
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Confirm initialization complete
/// assert_eq!(get_global_eop_type(), EOPType::StandardBulletinA);
/// ```
pub fn get_global_eop_type() -> EOPType {
    GLOBAL_EOP.read().unwrap().eop_type()
}

/// Return extrapolation setting of loaded EarthOrientationData provider in the global Earth orientation data provider.
/// See the `EOPExtrapolation` enum for possible values.
///
/// # Returns
/// 
/// - `extrapolation`: Extrapolation setting of loaded Earth Orientation data
///
/// # Examples
/// 
/// ```
/// use brahe::eop::*;
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Confirm initialization complete
/// assert_eq!(get_global_eop_extrapolation(), EOPExtrapolation::Zero);
/// ```
pub fn get_global_eop_extrapolation() -> EOPExtrapolation {
    GLOBAL_EOP.read().unwrap().extrapolation()
}

/// Return interpolation status of the global Earth orientation data provider.
/// 
/// When `true`, the global Earth orientation data provider will linearly interpolate between
/// data points when the requested date is between two data points. When `false`, the
/// global Earth orientation data provider will return the value from the most recent
/// data point.
///
/// # Returns
/// 
/// - `interpolation`: Interpolation setting of loaded Earth Orientation data
///
/// # Examples
/// 
/// ```
/// use brahe::eop::*;
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Confirm initialization complete
/// assert_eq!(get_global_eop_interpolation(), true);
/// ```
pub fn get_global_eop_interpolation() -> bool {
    GLOBAL_EOP.read().unwrap().interpolation()
}

/// Returns the earliest Modified Julian Date (MJD) available in the loaded EarthOrientationData
/// provider. Attempting to access data before this date will result in an error.
///
/// # Returns
/// 
/// - `mjd_min`: Minimum MJD of loaded EOP data points
///
/// # Examples
/// 
/// ```
/// use brahe::eop::*;
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Confirm initialization complete
/// assert!(get_global_eop_mjd_min() > 0.0);
/// assert!(get_global_eop_mjd_min() < 99999.0);
/// ```
pub fn get_global_eop_mjd_min() -> f64 {
    GLOBAL_EOP.read().unwrap().mjd_min()
}

/// Returns the latest Modified Julian Date (MJD) available in the loaded EarthOrientationData
/// provider. Attempting to access data after this date will result in the behavior specified
/// by the `extrapolation` setting of the underlying `EarthOrientationData` object.
///
/// # Returns
/// 
/// - `mjd_max`: Maximum MJD of loaded EOP data points
///
/// # Examples
/// 
/// ```
/// use brahe::eop::*;
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Confirm initialization complete
/// assert!(get_global_eop_mjd_max() > 0.0);
/// assert!(get_global_eop_mjd_max() < 99999.0);
/// ```
pub fn get_global_eop_mjd_max() -> f64 {
    GLOBAL_EOP.read().unwrap().mjd_max()
}

/// Returns the Modified Julian Date (MJD) of the last data point with a valid length of day
/// (LOD) value. Attempting to access data after this date will result in the behavior specified
/// by the `extrapolation` setting of the underlying `EarthOrientationData` object.
///
/// # Returns
/// 
/// - `mjd_last_lod`: MJD of latest chronological EOP data points with a valid LOD value
///
/// # Examples
/// 
/// ```
/// use brahe::eop::*;
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Confirm initialization complete
/// assert!(get_global_eop_mjd_last_lod() > 0.0);
/// assert!(get_global_eop_mjd_last_lod() < 99999.0);
/// ```
pub fn get_global_eop_mjd_last_lod() -> f64 {
    GLOBAL_EOP.read().unwrap().mjd_last_lod()
}

/// Returns the Modified Julian Date (MJD) of the last data point with a valid celestial
/// intermediate pole (CIP) offset. Attempting to access data after this date will result in
/// the behavior specified by the `extrapolation` setting of the underlying `EarthOrientationData`.
///
/// # Returns
/// 
/// - `mjd_last_dxdy`: MJD of latest chronological EOP data points with valid dX, dY values
///
/// # Examples
/// 
/// ```
/// use brahe::eop::*;
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Confirm initialization complete
/// assert!(get_global_eop_mjd_last_dxdy() > 0.0);
/// assert!(get_global_eop_mjd_last_dxdy() < 99999.0);
/// ```
pub fn get_global_eop_mjd_last_dxdy() -> f64 {
    GLOBAL_EOP.read().unwrap().mjd_last_dxdy()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::path::Path;
    use crate::constants::AS2RAD;
    use crate::eop::static_provider::StaticEOPProvider;
    use crate::eop::file_provider::FileEOPProvider;

    use serial_test::serial;

    use approx::assert_abs_diff_eq;

    fn setup_test_global_eop(eop_interpolation: bool, eop_extrapolation: EOPExtrapolation) {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = Path::new(&manifest_dir)
            .join("test_assets")
            .join("finals.all.iau2000.txt");

        let eop_result = FileEOPProvider::from_file(&filepath, eop_interpolation, eop_extrapolation);
        assert_eq!(eop_result.is_err(), false);
        let eop = eop_result.unwrap();
        assert_eq!(eop.is_initialized(), true);

        set_global_eop_provider(eop);
    }

    fn clear_test_global_eop() {
        set_global_eop_provider(StaticEOPProvider::new());
    }

    #[test]
    #[serial]
    fn test_set_global_eop_from_zero() {
        clear_test_global_eop();

        assert_eq!(get_global_eop_initialization(), false);

        let eop = StaticEOPProvider::from_zero();

        set_global_eop_provider(eop);

        assert_eq!(get_global_eop_initialization(), true);
        assert_eq!(get_global_eop_len(), 1);
        assert_eq!(get_global_eop_mjd_min(), 0.0);
        assert_eq!(get_global_eop_mjd_max(), f64::MAX);
        assert_eq!(get_global_eop_type(), EOPType::Static);
        assert_eq!(get_global_eop_extrapolation(), EOPExtrapolation::Hold);
        assert_eq!(get_global_eop_interpolation(), false);

        // EOP Values
        assert_eq!(get_global_ut1_utc(59950.0).unwrap(), 0.0);
        assert_eq!(get_global_pm(59950.0).unwrap().0, 0.0);
        assert_eq!(get_global_pm(59950.0).unwrap().1, 0.0);
        assert_eq!(get_global_dxdy(59950.0).unwrap().0, 0.0);
        assert_eq!(get_global_dxdy(59950.0).unwrap().1, 0.0);
        assert_eq!(get_global_lod(59950.0).unwrap(), 0.0);
    }

    #[test]
    #[serial]
    fn test_set_global_eop_from_static_values() {
        clear_test_global_eop();
        
        assert_eq!(get_global_eop_initialization(), false);

        let eop = StaticEOPProvider::from_values((0.001, 0.002, 0.003, 0.004, 0.005, 0.006));

        set_global_eop_provider(eop);

        assert_eq!(get_global_eop_initialization(), true);
        assert_eq!(get_global_eop_len(), 1);
        assert_eq!(get_global_eop_mjd_min(), 0.0);
        assert_eq!(get_global_eop_mjd_max(), f64::MAX);
        assert_eq!(get_global_eop_type(), EOPType::Static);
        assert_eq!(get_global_eop_extrapolation(), EOPExtrapolation::Hold);
        assert_eq!(get_global_eop_interpolation(), false);

        // EOP Values
        assert_eq!(get_global_ut1_utc(59950.0).unwrap(), 0.003);
        assert_eq!(get_global_pm(59950.0).unwrap().0, 0.001);
        assert_eq!(get_global_pm(59950.0).unwrap().1, 0.002);
        assert_eq!(get_global_dxdy(59950.0).unwrap().0, 0.004);
        assert_eq!(get_global_dxdy(59950.0).unwrap().1, 0.005);
        assert_eq!(get_global_lod(59950.0).unwrap(), 0.006);
    }

    #[test]
    #[serial]
    fn test_set_global_eop_from_c04_file() {
        clear_test_global_eop();
        assert_eq!(get_global_eop_initialization(), false);

        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = Path::new(&manifest_dir)
            .join("test_assets")
            .join("EOP_20_C04_one_file_1962-now.txt");

        let eop = FileEOPProvider::from_file(&filepath, true, EOPExtrapolation::Hold).unwrap();

        set_global_eop_provider(eop);

        assert_eq!(get_global_eop_initialization(), true);
        assert_eq!(get_global_eop_len(), 22605);
        assert_eq!(get_global_eop_mjd_min(), 37665.0);
        assert_eq!(get_global_eop_mjd_max(), 60269.0);
        assert_eq!(get_global_eop_type(), EOPType::C04);
        assert_eq!(get_global_eop_extrapolation(), EOPExtrapolation::Hold);
        assert_eq!(get_global_eop_interpolation(), true);
    }

    #[test]
    #[serial]
    fn test_set_global_eop_from_default_c04() {
        clear_test_global_eop();
        assert_eq!(get_global_eop_initialization(), false);

        let eop = FileEOPProvider::from_default_file(EOPType::C04, false, EOPExtrapolation::Zero).unwrap();
        
        set_global_eop_provider(eop);

        assert_eq!(get_global_eop_initialization(), true);
        assert_eq!(get_global_eop_len(), 22619);
        assert_eq!(get_global_eop_mjd_min(), 37665.0);
        assert!(get_global_eop_mjd_max() >= 60269.0);
        assert_eq!(get_global_eop_type(), EOPType::C04);
        assert_eq!(get_global_eop_extrapolation(), EOPExtrapolation::Zero);
        assert_eq!(get_global_eop_interpolation(), false);
    }

    #[test]
    #[serial]
    fn test_set_global_eop_from_standard_file() {
        clear_test_global_eop();
        assert_eq!(get_global_eop_initialization(), false);

        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let filepath = Path::new(&manifest_dir)
            .join("test_assets")
            .join("finals.all.iau2000.txt");

        let eop = FileEOPProvider::from_file(&filepath, true, EOPExtrapolation::Hold).unwrap();

        set_global_eop_provider(eop);

        assert_eq!(get_global_eop_initialization(), true);
        assert_eq!(get_global_eop_len(), 18989);
        assert_eq!(get_global_eop_mjd_min(), 41684.0);
        assert_eq!(get_global_eop_mjd_max(), 60672.0);
        assert_eq!(get_global_eop_type(), EOPType::StandardBulletinA);
        assert_eq!(get_global_eop_extrapolation(), EOPExtrapolation::Hold);
        assert_eq!(get_global_eop_interpolation(), true);
    }

    #[test]
    #[serial]
    fn test_set_global_eop_from_default_standard() {
        clear_test_global_eop();
        assert_eq!(get_global_eop_initialization(), false);

        let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, false, EOPExtrapolation::Zero).unwrap();
        
        set_global_eop_provider(eop);

        assert_eq!(get_global_eop_initialization(), true);
        assert_eq!(get_global_eop_len(), 18996);
        assert_eq!(get_global_eop_mjd_min(), 41684.0);
        assert!(get_global_eop_mjd_max() >= 60672.0);
        assert_eq!(get_global_eop_type(), EOPType::StandardBulletinA);
        assert_eq!(get_global_eop_extrapolation(), EOPExtrapolation::Zero);
        assert_eq!(get_global_eop_interpolation(), false);
    }

    #[test]
    #[serial]
    fn test_get_global_ut1_utc() {
        clear_test_global_eop();
        setup_test_global_eop(true, EOPExtrapolation::Hold);
        assert_eq!(get_global_eop_initialization(), true);


        // Test getting exact point in table
        let ut1_utc = get_global_ut1_utc(59569.0).unwrap();
        assert_eq!(ut1_utc, -0.1079939);

        // Test interpolating within table
        let ut1_utc = get_global_ut1_utc(59569.5).unwrap();
        assert_eq!(ut1_utc, (-0.1079939 + -0.1075984) / 2.0);

        // Test extrapolation hold
        let ut1_utc = get_global_ut1_utc(99999.0).unwrap();
        assert_eq!(ut1_utc, 0.0420038);

        // Test extrapolation zero
        setup_test_global_eop(true, EOPExtrapolation::Zero);

        let ut1_utc = get_global_ut1_utc(99999.0).unwrap();
        assert_eq!(ut1_utc, 0.0);
    }

    #[test]
    #[serial]
    fn test_get_global_pm_xy() {
        clear_test_global_eop();
        setup_test_global_eop(true, EOPExtrapolation::Hold);
        assert_eq!(get_global_eop_initialization(), true);

        // Test getting exact point in table
        let (pm_x, pm_y) = get_global_pm(59569.0).unwrap();
        assert_eq!(pm_x, 0.075382 * AS2RAD);
        assert_eq!(pm_y, 0.263451 * AS2RAD);

        // Test interpolating within table
        let (pm_x, pm_y) = get_global_pm(59569.5).unwrap();
        assert_eq!(pm_x, (0.075382 * AS2RAD + 0.073157 * AS2RAD) / 2.0);
        assert_eq!(pm_y, (0.263451 * AS2RAD + 0.264273 * AS2RAD) / 2.0);

        // Test extrapolation hold
        let (pm_x, pm_y) = get_global_pm(99999.0).unwrap();
        assert_eq!(pm_x, 0.173369 * AS2RAD);
        assert_eq!(pm_y, 0.266914 * AS2RAD);

        // Test extrapolation zero
        setup_test_global_eop(true, EOPExtrapolation::Zero);

        let (pm_x, pm_y) = get_global_pm(99999.0).unwrap();
        assert_eq!(pm_x, 0.0);
        assert_eq!(pm_y, 0.0);
    }

    #[test]
    #[serial]
    #[allow(non_snake_case)]
    fn test_get_global_dxdy() {
        clear_test_global_eop();
        setup_test_global_eop(true, EOPExtrapolation::Hold);
        assert_eq!(get_global_eop_initialization(), true);

        // Test getting exact point in table
        let (dX, dY) = get_global_dxdy(59569.0).unwrap();
        assert_eq!(dX,  0.265 * 1.0e-3 * AS2RAD);
        assert_eq!(dY, -0.067 * 1.0e-3 * AS2RAD);

        // Test interpolating within table
        let (dX, dY) = get_global_dxdy(59569.5).unwrap();
        assert_eq!(dX, ( 0.265 * AS2RAD + 0.268 * AS2RAD) * 1.0e-3 / 2.0);
        assert_abs_diff_eq!(dY, (-0.067 * AS2RAD + -0.067 * AS2RAD) * 1.0e-3 / 2.0, epsilon = f64::EPSILON);

        // Test extrapolation hold
        let (dX, dY) = get_global_dxdy(99999.0).unwrap();
        assert_eq!(dX,  0.006 * 1.0e-3 * AS2RAD);
        assert_eq!(dY, -0.118 * 1.0e-3 * AS2RAD);

        // Test extrapolation zero
        setup_test_global_eop(true, EOPExtrapolation::Zero);

        let (dX, dY) = get_global_dxdy(99999.0).unwrap();
        assert_eq!(dX, 0.0);
        assert_eq!(dY, 0.0);
    }

    #[test]
    #[serial]
    fn test_get_global_lod() {
        clear_test_global_eop();
        setup_test_global_eop(true, EOPExtrapolation::Hold);
        assert_eq!(get_global_eop_initialization(), true);

        // Test getting exact point in table
        let lod = get_global_lod(59569.0).unwrap();
        assert_eq!(lod, -0.3999 * 1.0e-3);

        // Test interpolating within table
        let lod = get_global_lod(59569.5).unwrap();
        assert_eq!(lod, (-0.3999 + -0.3604) * 1.0e-3 / 2.0);

        // Test extrapolation hold
        let lod = get_global_lod(99999.0).unwrap();
        assert_eq!(lod, 0.7706 * 1.0e-3);

        // Test extrapolation zero
        setup_test_global_eop(true, EOPExtrapolation::Zero);

        let lod = get_global_lod(99999.0).unwrap();
        assert_eq!(lod, 0.0);
    }

    #[test]
    #[serial]
    fn test_get_global_eop_initialization() {
        clear_test_global_eop();
        setup_test_global_eop(true, EOPExtrapolation::Hold);
        assert_eq!(get_global_eop_initialization(), true);
    }

    #[test]
    #[serial]
    fn test_get_global_eop_len() {
        clear_test_global_eop();
        setup_test_global_eop(true, EOPExtrapolation::Hold);
        assert_eq!(get_global_eop_initialization(), true);

        assert_eq!(get_global_eop_len(), 18989);
    }

    #[test]
    #[serial]
    fn test_get_global_eop_type() {
        clear_test_global_eop();
        setup_test_global_eop(true, EOPExtrapolation::Hold);
        assert_eq!(get_global_eop_initialization(), true);

        assert_eq!(get_global_eop_type(), EOPType::StandardBulletinA);
    }

    #[test]
    #[serial]
    fn test_get_global_eop_extrapolation() {
        clear_test_global_eop();
        setup_test_global_eop(true, EOPExtrapolation::Hold);
        assert_eq!(get_global_eop_initialization(), true);

        assert_eq!(get_global_eop_extrapolation(), EOPExtrapolation::Hold);
    }

    #[test]
    #[serial]
    fn test_get_global_eop_interpolation() {
        clear_test_global_eop();
        setup_test_global_eop(true, EOPExtrapolation::Hold);
        assert_eq!(get_global_eop_initialization(), true);

        assert_eq!(get_global_eop_interpolation(), true);
    }

    #[test]
    #[serial]
    fn test_get_global_eop_mjd_min() {
        clear_test_global_eop();
        setup_test_global_eop(true, EOPExtrapolation::Hold);
        assert_eq!(get_global_eop_initialization(), true);

        assert_eq!(get_global_eop_mjd_min(), 41684.0);
    }

    #[test]
    #[serial]
    fn test_get_global_eop_mjd_max() {
        clear_test_global_eop();
        setup_test_global_eop(true, EOPExtrapolation::Hold);
        assert_eq!(get_global_eop_initialization(), true);

        assert_eq!(get_global_eop_mjd_max(), 60672.0);
    }

    #[test]
    #[serial]
    fn test_get_global_eop_mjd_last_lod() {
        clear_test_global_eop();
        setup_test_global_eop(true, EOPExtrapolation::Hold);
        assert_eq!(get_global_eop_initialization(), true);

        assert_eq!(get_global_eop_mjd_last_lod(), 60298.0);
    }

    #[test]
    #[serial]
    fn test_get_global_eop_mjd_last_dxdy() {
        clear_test_global_eop();
        setup_test_global_eop(true, EOPExtrapolation::Hold);
        assert_eq!(get_global_eop_initialization(), true);

        assert_eq!(get_global_eop_mjd_last_dxdy(), 60373.0);
    }
}