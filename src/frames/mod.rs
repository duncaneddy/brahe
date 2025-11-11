/*!
 * Reference frame transformations between Earth-Centered Inertial (ECI) and
 * Earth-Centered Earth-Fixed (ECEF) coordinate systems.
 *
 * This module provides transformations between various reference frames:
 * - GCRF/ITRF: Geocentric Celestial Reference Frame and International Terrestrial Reference Frame
 * - ECI/ECEF: Earth-Centered Inertial and Earth-Centered Earth-Fixed (aliases for GCRF/ITRF)
 * - EME2000: Earth Mean Equator and Equinox of J2000.0
 */

pub mod eci_ecef;
pub mod eme_2000;
pub mod gcrf_itrf;

pub use eci_ecef::*;
pub use eme_2000::*;
pub use gcrf_itrf::*;
