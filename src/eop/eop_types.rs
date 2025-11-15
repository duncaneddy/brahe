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
    /// Return zero for missing EOP data points. Use when missing data should be treated
    /// as negligible or when approximate calculations are acceptable.
    Zero,
    /// Return the last known value prior to the requested date. Use for near-term
    /// extrapolation when EOP values change slowly and continuity is important.
    Hold,
    /// Panic and terminate execution when data is missing. Use when accuracy is critical
    /// and operating with missing EOP data would produce unacceptable errors.
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
    /// Unknown or unspecified EOP data source. Default value before initialization.
    Unknown,
    /// IERS EOP 14 C04 long-term data product. High-quality historical data covering
    /// 1962-present, updated annually. Best for historical analysis and validation.
    C04,
    /// IERS Bulletin A (finals2000A.all file). Near real-time data with predictions
    /// extending ~1 year into the future. Standard choice for operational applications.
    StandardBulletinA,
    /// Static EOP values (typically zeros or constants). Use for testing or when
    /// EOP corrections are not needed for the application's accuracy requirements.
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_eop_extrapolation_display() {
        assert_eq!(
            format!("{}", EOPExtrapolation::Zero),
            "EOPExtrapolation::Zero"
        );
        assert_eq!(
            format!("{}", EOPExtrapolation::Hold),
            "EOPExtrapolation::Hold"
        );
        assert_eq!(
            format!("{}", EOPExtrapolation::Error),
            "EOPExtrapolation::Error"
        );
    }

    #[test]
    fn test_eop_type_display() {
        assert_eq!(format!("{}", EOPType::Unknown), "Unknown");
        assert_eq!(format!("{}", EOPType::C04), "C04");
        assert_eq!(format!("{}", EOPType::StandardBulletinA), "Bulletin A");
        assert_eq!(format!("{}", EOPType::Static), "Static");
    }
}
