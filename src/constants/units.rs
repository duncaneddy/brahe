/*!
 * Unit constants and enumerations for consistent unit handling across the library.
 *
 * This module provides type-safe representations of unit formats used throughout
 * the Brahe library, particularly for angle representations.
 */

use serde::{Deserialize, Serialize};

/// Enumeration of angle formats for consistent angle representation across the library.
///
/// This enum is used throughout the library to specify whether angular quantities
/// should be interpreted or returned in radians or degrees. It replaces the legacy
/// `as_degrees: bool` pattern with a more explicit and type-safe approach.
///
/// For trajectory and propagator metadata where "no angles" (Cartesian) needs to be
/// represented, use `Option<AngleFormat>` where `None` indicates Cartesian representation.
///
/// # Examples
///
/// ```
/// use brahe::constants::AngleFormat;
///
/// let format = AngleFormat::Degrees;
/// let angle = match format {
///     AngleFormat::Radians => 3.14159,
///     AngleFormat::Degrees => 180.0,
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AngleFormat {
    /// Angles represented in radians
    Radians,
    /// Angles represented in degrees
    Degrees,
}

/// Constant for specifying angles in degrees.
///
/// This constant provides a convenient shorthand for `AngleFormat::Degrees`
/// and should be used for better code readability.
///
/// # Examples
///
/// ```
/// use brahe::constants::DEGREES;
/// use brahe::orbits::mean_motion;
///
/// let n = mean_motion(7000e3, DEGREES);
/// ```
pub const DEGREES: AngleFormat = AngleFormat::Degrees;

/// Constant for specifying angles in radians.
///
/// This constant provides a convenient shorthand for `AngleFormat::Radians`
/// and should be used for better code readability.
///
/// # Examples
///
/// ```
/// use brahe::constants::RADIANS;
/// use brahe::orbits::mean_motion;
///
/// let n = mean_motion(7000e3, RADIANS);
/// ```
pub const RADIANS: AngleFormat = AngleFormat::Radians;
