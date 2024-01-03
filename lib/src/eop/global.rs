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
    Arc::new(RwLock::new(Box::new(StaticEOPProvider::from_zero())))
});

pub fn set_global_eop_provider<T: EarthOrientationProvider + Sync + Send + 'static>(provider: T) {
    *GLOBAL_EOP.write().unwrap() = Box::new(provider);
}

/// Get UT1-UTC offset set for specified date from loaded static Earth orientation data.
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
/// ```rust
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

/// Get polar motion offset set for specified date from loaded static Earth orientation data.
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
/// ```rust
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

/// Get precession-nutation for specified date from loaded static Earth orientation data.
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
/// ```rust
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

/// Get length of day offset set for specified date from loaded static Earth orientation data.
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
/// ```rust
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

/// Get Earth orientation parameter set for specified date from loaded static Earth orientation data.
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
/// ```rust
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

/// Returns initialzation state of global Earth orientation data
///
/// # Returns
/// - `intialized`: Boolean, which if `true` indicates that the global static variable has been properly initialized.
///
/// # Examples
/// ```rust
/// use brahe::eop::*;
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// assert_eq!(get_global_eop_initialization(), true);
/// ```
pub fn get_global_eop_initialization() -> bool {
    GLOBAL_EOP.read().unwrap().initialized()
}

/// Return length of loaded EarthOrientationData
///
/// # Returns
/// - `len`: length of number of loaded EOP data points
///
/// # Examples
/// ```rust
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

/// Return eop_type value of loaded EarthOrientationData
///
/// # Returns
/// - `eop_type`: Type of loaded Earth Orientation data
///
/// # Examples
/// ```rust
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

/// Return extrapolation value of loaded EarthOrientationData
///
/// # Returns
/// - `extrapolation`: Extrapolation setting of loaded Earth Orientation data
///
/// # Examples
/// ```rust
/// use brahe::eop::*;
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Confirm initialization complete
/// assert_eq!(get_global_eop_extrapolate(), EOPExtrapolation::Zero);
/// ```
pub fn get_global_eop_extrapolate() -> EOPExtrapolation {
    GLOBAL_EOP.read().unwrap().extrapolate()
}

/// Return interpolation value of loaded EarthOrientationData
///
/// # Returns
/// - `interpolation`: Interpolation setting of loaded Earth Orientation data
///
/// # Examples
/// ```rust
/// use brahe::eop::*;
///
/// // Initialize Global EOP
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// // Confirm initialization complete
/// assert_eq!(get_global_eop_interpolate(), true);
/// ```
pub fn get_global_eop_interpolate() -> bool {
    GLOBAL_EOP.read().unwrap().interpolate()
}

/// Return mjd_min value of loaded EarthOrientationData
///
/// # Returns
/// - `mjd_min`: Minimum MJD of loaded EOP data points
///
/// # Examples
/// ```rust
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

/// Return mjd_max value of loaded EarthOrientationData
///
/// # Returns
/// - `mjd_max`: Maximum MJD of loaded EOP data points
///
/// # Examples
/// ```rust
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

/// Return mjd_last_lod value of loaded EarthOrientationData
///
/// # Returns
/// - `mjd_last_lod`: MJD of latest chronological EOP data points with a valid LOD value
///
/// # Examples
/// ```rust
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

/// Return mjd_last_dxdy value of loaded EarthOrientationData
///
/// # Returns
/// - `mjd_last_dxdy`: MJD of latest chronological EOP data points with valid dX, dY values
///
/// # Examples
/// ```rust
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