/*!
Types for the EOP module.
*/

use std::fmt;

/// Enumerated value that indicates the preferred behavior of the Earth Orientation Data provider
/// when the desired time point is not present.
///
/// # Values
/// - `Zero`: Return a value of zero for the missing data
/// - `Hold`: Return the last value prior to the requested date
/// - `Error`: Panics current execution thread, immediately terminating the program
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum EOPExtrapolation {
    Zero,
    Hold,
    Error,
}

impl fmt::Display for EOPExtrapolation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            EOPExtrapolation::Zero => write!(f, "EOPExtrapolation::Zero"),
            EOPExtrapolation::Hold => write!(f, "EOPExtrapolation::Hold"),
            EOPExtrapolation::Error => write!(f, "EOPExtrapolation::Error"),
        }
    }
}

/// Enumerates type of Earth Orientation data loaded. All models assumed to be
/// consistent with IAU2000 precession Nutation Model
///
/// # Values
/// - `C04`: IERS Long Term Data Product EOP 14 C04
/// - `StandardBulletinA`: IERS Standard Data Bulletin A from finals2000 file
/// - `StandardBulletinB`: IERS Standard Data Bulletin B from finals2000 file
/// - `Static`: Static EOP data
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum EOPType {
    Unknown,
    C04,
    StandardBulletinA,
    Static,
}

impl fmt::Display for EOPType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            EOPType::Unknown => write!(f, "Unknown"),
            EOPType::C04 => write!(f, "C04"),
            EOPType::StandardBulletinA => write!(f, "Bulletin A"),
            EOPType::Static => write!(f, "Static"),
        }
    }
}
