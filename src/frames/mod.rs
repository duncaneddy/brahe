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
pub mod gcrf_itrf;
pub mod iau_rotation;
pub mod lunar;
pub mod mars;
pub mod synodic;
pub mod transform;

pub use custom::*;
pub use eci_ecef::*;
pub use emb::*;
pub use eme_2000::*;
pub use gcrf_itrf::*;
pub use iau_rotation::*;
pub use lunar::*;
pub use mars::*;
pub use synodic::*;
pub use transform::*;
