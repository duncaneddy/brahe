/*!
 * The orbits module contains the core types and functions for working with orbital element
 * representations and Two-Line Element (TLE) handling.
 */

/// Keplerian ↔ equinoctial element conversions (Vallado 2-99).
pub mod equinoctial;
pub mod keplerian;
/// Mean-Osculating Keplerian element conversions using Brouwer-Lyddane theory.
pub mod mean_elements;
/// Numerical windowed-averaging core for osculating-to-mean conversion (internal).
pub(crate) mod mean_elements_numerical;
/// Two-Line Element (TLE) format parsing and handling.
pub mod tle;
/// Walker Delta constellation generator.
pub mod walker;

pub use equinoctial::*;
pub use keplerian::*;
pub use mean_elements::*;
pub use tle::*;
pub use walker::*;
