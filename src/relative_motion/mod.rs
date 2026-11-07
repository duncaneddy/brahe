/*!
 * The relative_motion module contains types and functions for working with
 * satellite relative motion and orbital reference frames.
 *
 * This module provides transformations between inertial frames and orbital
 * reference frames such as RTN (Radial-Tangential-Normal), LVLH
 * (Local-Vertical Local-Horizontal), and NTW (Normal-Tangential-Cross-track).
 */

pub(crate) mod common;
pub mod eci_lvlh;
pub mod eci_ntw;
pub mod eci_roe;
pub mod eci_rtn;
pub mod oe_roe;

pub use eci_lvlh::*;
pub use eci_ntw::*;
pub use eci_roe::*;
pub use eci_rtn::*;
pub use oe_roe::*;
