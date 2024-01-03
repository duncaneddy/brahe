/*!
Defines the EarthOrientationProvider trait
*/

use crate::eop::types::{EOPExtrapolation, EOPType};
use crate::utils::errors::BraheError;

pub trait EarthOrientationProvider {
    fn len(&self) -> usize;
    fn eop_type(&self) -> EOPType;
    fn initialized(&self) -> bool;
    fn extrapolate(&self) -> EOPExtrapolation;
    fn interpolate(&self) -> bool;
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
