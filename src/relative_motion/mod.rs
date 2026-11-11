/*!
 * The relative_motion module contains types and functions for working with
 * satellite relative motion and orbital reference frames.
 *
 * This module provides transformations between inertial frames and orbital
 * reference frames such as RTN (Radial-Tangential-Normal), LVLH
 * (Local-Vertical Local-Horizontal), NTW (Normal-Tangential-Cross-track),
 * TNW (Tangential-Normal-Cross-track), VNC (Velocity-Normal-Co-normal),
 * PQW (Perifocal), EQW (Equinoctial), NSW (Nadir-Sun-Normal), and SEZ
 * (South-East-Zenith, topocentric).
 */

pub(crate) mod common;
pub mod ecef_sez;
pub mod eci_eqw;
pub mod eci_lvlh;
pub mod eci_nsw;
pub mod eci_ntw;
pub mod eci_pqw;
pub mod eci_roe;
pub mod eci_rtn;
pub mod eci_tnw;
pub mod eci_vnc;
pub mod oe_roe;

pub use ecef_sez::*;
pub use eci_eqw::*;
pub use eci_lvlh::*;
pub use eci_nsw::*;
pub use eci_ntw::*;
pub use eci_pqw::*;
pub use eci_roe::*;
pub use eci_rtn::*;
pub use eci_tnw::*;
pub use eci_vnc::*;
pub use oe_roe::*;
