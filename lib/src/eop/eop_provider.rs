/*!
Defines the EarthOrientationProvider trait 
*/

use crate::eop::types::{EOPType, EOPExtrapolation};

pub trait EarthOrientationProvider {
    fn len(&self) -> usize;
    fn eop_type(&self) -> EOPType;
    fn extrapolate(&self) -> EOPExtrapolation;
    fn interpolate(&self) -> bool;
    fn mjd_min(&self) -> u32;
    fn mjd_max(&self) -> u32;
    fn mjd_last_lod(&self) -> u32;
    fn mjd_last_dxdy(&self) -> u32;
    fn get_ut1_utc(&self, mjd: f64) -> Result<f64, String>;
    fn get_pm(&self, mjd: f64) -> Result<(f64, f64), String>;
    fn get_dxdy(&self, mjd: f64) -> Result<(f64, f64), String>;
    fn get_lod(&self, mjd: f64) -> Result<f64, String>;
    fn get_eop(&self, mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), String>;
}