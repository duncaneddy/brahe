/*!
 * Types for the Time module.
 */

use std::fmt;

use serde::{Deserialize, Serialize};

/// Enumeration of different time systems.
///
/// A time system is a recognized time standard for representing instants in time
/// along a consistent, continuous scale. Because all current time systems utilize
/// the same definition of a second, the spacing between instants in time is the
/// same across all time scales. This leaves the only difference between them being
/// offsets between them.
///
/// The currently supposed time systems are:
/// - GPS: Global Positioning System. GPS is a time scale used defined by the GPS navigation system control segment.
///   GPS time was aligned with UTC at system inception (January 6, 1980 0h), but
///   does not include leap seconds since it is an atomic time scale.
/// - TAI: Temps Atomique International. TAI is an atomic time scale, which represents
///   passage of time on Earth's geoid.
/// - TT: Terrestrial Time. TT is a theoretical time standard primarily used for astronomy.
///   TT is offset from TAI by a fixed number of seconds at TAI's inception. This number has not
///   been officially updated, however reprocessing of data from the ensemble of atomic clocks
///   that define TAI could lead to a difference. For exact applications that require precise corrections
///   updated yearly BIPM provides these offsets.
/// - UTC: Universal Coordinated Time. UTC is an atomic time scale steered to remain within
///   +/- 0.9 seconds of solar time. Since the rotation of the Earth is continuously changing,
///   UTC periodically incorporates leap seconds to ensure that the difference between
///   UTC and UT1 remains within the expeccted bounds.
/// - UT1: Universal Time 1. UT1 is a solar time that is conceptually the mean time at 0 degrees
///   longitude. UT1 is the same everywhere on Earth simultaneously and represents the rotation of the
///   Earth with respect to the ICRF inertial reference frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TimeSystem {
    GPS,
    TAI,
    TT,
    UTC,
    UT1,
}

impl fmt::Display for TimeSystem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TimeSystem::GPS => write!(f, "GPS"),
            TimeSystem::TAI => write!(f, "TAI"),
            TimeSystem::TT => write!(f, "TT"),
            TimeSystem::UTC => write!(f, "UTC"),
            TimeSystem::UT1 => write!(f, "UT1"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_system_display() {
        assert_eq!(format!("{}", TimeSystem::GPS), "GPS");
        assert_eq!(format!("{}", TimeSystem::TAI), "TAI");
        assert_eq!(format!("{}", TimeSystem::TT), "TT");
        assert_eq!(format!("{}", TimeSystem::UTC), "UTC");
        assert_eq!(format!("{}", TimeSystem::UT1), "UT1");
    }
}
