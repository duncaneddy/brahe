/*!
Module implementing orbit dynamics models.
 */

pub use drag::*;
pub use ephemerides::*;
pub use gravity::*;
pub use ocean_tides::fes2004_coefficients_path;
pub use relativity::*;
pub use solar_radiation_pressure::*;
pub use third_body::*;

// Re-export atmospheric density models from earth_models for backward compatibility
pub mod atmospheric_density_models {
    //! Backward-compatible re-export of atmospheric density models.
    //!
    //! These models have been moved to [`crate::earth_models`].
    //! This module is kept for backward compatibility.
    pub use crate::earth_models::*;
}

pub mod drag;
pub mod ephemerides;
pub mod gravity;
pub mod ocean_tides;
mod ocean_tides_admittance;
pub mod relativity;
pub mod solar_radiation_pressure;
pub mod third_body;
pub mod tides;
pub mod tides_step2_tables;
pub use tides::{
    PERM_C20_DIRECT, PERM_C20_INDIRECT, SolidTideConfig, accel_solid_earth_tides,
    solid_earth_tide_deltas, tide_system_c20_offset,
};
