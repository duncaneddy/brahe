/*!
 * Reference frame transformations between Earth-Centered Inertial (ECI) and
 * Earth-Centered Earth-Fixed (ECEF) coordinate systems.
 *
 * This module provides transformations between various reference frames:
 * - GCRF/ITRF: Geocentric Celestial Reference Frame and International Terrestrial Reference Frame
 * - ECI/ECEF: Earth-Centered Inertial and Earth-Centered Earth-Fixed (aliases for GCRF/ITRF)
 * - EME2000: Earth Mean Equator and Equinox of J2000.0
 * - EMR/SER/GSE: Earth-Moon Rotating, Sun-Earth Rotating, and Geocentric Solar Ecliptic (synodic frames)
 */

pub mod custom;
pub mod eci_ecef;
pub mod emb;
pub mod eme_2000;
pub mod frame;
pub mod gcrf_itrf;
pub mod iau_rotation;
pub mod lunar;
pub mod mars;
pub mod object_registry;
pub mod orientation;
// Not `pub`: `crate::spice::registry` is already a public module of that
// name, and `pub mod registry;` here would make `pub use frames::*;` (in
// `lib.rs`) collide with `pub use spice::*;` on the module name itself. The
// glob re-export below still surfaces every public item.
mod registry;
pub mod synodic;
pub mod transform;

pub use custom::*;
pub use eci_ecef::*;
pub use emb::*;
pub use eme_2000::*;
pub use frame::*;
pub use gcrf_itrf::*;
pub use iau_rotation::*;
pub use lunar::*;
pub use mars::*;
pub use object_registry::*;
pub use orientation::*;
pub use registry::*;
pub use synodic::*;
pub use transform::*;
