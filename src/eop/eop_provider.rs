/*!
Defines the EarthOrientationProvider trait
*/

use crate::eop::eop_types::{EOPExtrapolation, EOPType};
use crate::utils::errors::BraheError;

/// `EarthOrientationProvider` is a trait for objects that provide Earth orientation parameters.
///
/// This trait defines a common interface for all Earth orientation providers. An Earth orientation provider
/// is an object that can provide Earth orientation parameters like UT1-UTC, polar motion, and rate of change
/// of the length of day.
///
/// Implementations of this trait are expected to provide specific methods for retrieving these parameters,
/// such as `get_ut1_utc`, `get_pm`, `get_dxdy`, and `get_lod`. These methods should return the requested
/// parameter for a given Modified Julian Date (MJD).
///
/// This trait can be extended to implement custom Earth orientation providers. For example, a custom provider
/// could retrieve Earth orientation parameters from a file type that is not yet supported or a database.
///
/// # Example
///
/// ```ignore
/// use brahe::eop::EarthOrientationProvider;
///
/// struct MyEOPProvider;
///
/// impl EarthOrientationProvider for MyEOPProvider {
///     // Implement the methods here
/// }
/// ```
pub trait EarthOrientationProvider {
    fn len(&self) -> usize;
    fn eop_type(&self) -> EOPType;
    fn is_initialized(&self) -> bool;
    fn extrapolation(&self) -> EOPExtrapolation;
    fn interpolation(&self) -> bool;
    fn mjd_min(&self) -> f64;
    fn mjd_max(&self) -> f64;
    fn mjd_last_lod(&self) -> f64;
    fn mjd_last_dxdy(&self) -> f64;
    fn get_ut1_utc(&self, mjd: f64) -> Result<f64, BraheError>;
    fn get_pm(&self, mjd: f64) -> Result<(f64, f64), BraheError>;
    fn get_dxdy(&self, mjd: f64) -> Result<(f64, f64), BraheError>;
    fn get_lod(&self, mjd: f64) -> Result<f64, BraheError>;
    fn get_eop(&self, mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), BraheError>;
}
