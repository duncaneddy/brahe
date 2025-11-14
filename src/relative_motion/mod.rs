/*!
 * The relative_motion module contains types and functions for working with
 * satellite relative motion and orbital reference frames.
 *
 * This module provides transformations between inertial frames and orbital
 * reference frames such as RTN (Radial-Tangential-Normal).
 */

pub mod eci_rtn;
pub mod oe_roe;

pub use eci_rtn::*;
pub use oe_roe::*;
