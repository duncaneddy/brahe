//! SP3 enums and satellite identifiers.

use std::fmt;
use std::str::FromStr;

use crate::time::TimeSystem;
use crate::utils::BraheError;

fn sp3_error(detail: &str) -> BraheError {
    BraheError::ParseError(format!("SP3: {}", detail))
}

/// SP3 format version.
///
/// The version letter appears as the second character of the first header
/// line (`#a`, `#b`, `#c`, `#d`).
///
/// # Examples
///
/// ```
/// use brahe::datafiles::sp3::SP3Version;
///
/// assert_eq!(SP3Version::from_char('D').unwrap(), SP3Version::D);
/// assert_eq!(SP3Version::D.as_char(), 'd');
/// assert_eq!(SP3Version::C.to_string(), "c");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SP3Version {
    /// SP3-a, GPS satellites only
    A,
    /// SP3-b, adds multi-GNSS satellite identifiers
    B,
    /// SP3-c, adds clock event flags and correlation records
    C,
    /// SP3-d, adds extended satellite counts and long comments
    D,
}

impl SP3Version {
    /// Parse a version letter, ignoring case.
    ///
    /// # Arguments
    ///
    /// * `c` - Version letter, one of `a`, `b`, `c`, `d` in either case
    ///
    /// # Returns
    ///
    /// * `Result<SP3Version, BraheError>` - The version, or a parse error for any other character
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3Version;
    ///
    /// assert_eq!(SP3Version::from_char('c').unwrap(), SP3Version::C);
    /// assert!(SP3Version::from_char('e').is_err());
    /// ```
    pub fn from_char(c: char) -> Result<Self, BraheError> {
        match c.to_ascii_lowercase() {
            'a' => Ok(SP3Version::A),
            'b' => Ok(SP3Version::B),
            'c' => Ok(SP3Version::C),
            'd' => Ok(SP3Version::D),
            _ => Err(sp3_error(&format!("unsupported version '{}'", c))),
        }
    }

    /// Lowercase version letter as written in the first header line.
    ///
    /// # Returns
    ///
    /// * `char` - One of `a`, `b`, `c`, `d`
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3Version;
    ///
    /// assert_eq!(SP3Version::B.as_char(), 'b');
    /// ```
    pub fn as_char(&self) -> char {
        match self {
            SP3Version::A => 'a',
            SP3Version::B => 'b',
            SP3Version::C => 'c',
            SP3Version::D => 'd',
        }
    }
}

impl fmt::Display for SP3Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_char())
    }
}

/// Satellite navigation system of an SP3 satellite identifier.
///
/// Each system has a one-letter code used as the first character of the
/// satellite identifier (`G05`, `R12`, ...). Variants order as
/// GPS, GLONASS, Galileo, BeiDou, QZSS, IRNSS, SBAS, LEO.
///
/// # Examples
///
/// ```
/// use brahe::datafiles::sp3::GNSSSystem;
///
/// assert_eq!(GNSSSystem::from_code('E').unwrap(), GNSSSystem::Galileo);
/// assert_eq!(GNSSSystem::BeiDou.code(), 'C');
/// assert_eq!(GNSSSystem::GLONASS.to_string(), "GLONASS");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GNSSSystem {
    /// Global Positioning System (`G`)
    GPS,
    /// GLONASS (`R`)
    GLONASS,
    /// Galileo (`E`)
    Galileo,
    /// BeiDou (`C`)
    BeiDou,
    /// Quasi-Zenith Satellite System (`J`)
    QZSS,
    /// Indian Regional Navigation Satellite System (`I`)
    IRNSS,
    /// Satellite-based augmentation systems (`S`)
    SBAS,
    /// Low Earth orbiters (`L`)
    LEO,
}

impl GNSSSystem {
    /// Parse a system code letter.
    ///
    /// # Arguments
    ///
    /// * `c` - System code, one of `G`, `R`, `E`, `C`, `J`, `I`, `S`, `L`
    ///
    /// # Returns
    ///
    /// * `Result<GNSSSystem, BraheError>` - The system, or a parse error for any other character
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::GNSSSystem;
    ///
    /// assert_eq!(GNSSSystem::from_code('R').unwrap(), GNSSSystem::GLONASS);
    /// assert!(GNSSSystem::from_code('X').is_err());
    /// ```
    pub fn from_code(c: char) -> Result<Self, BraheError> {
        match c {
            'G' => Ok(GNSSSystem::GPS),
            'R' => Ok(GNSSSystem::GLONASS),
            'E' => Ok(GNSSSystem::Galileo),
            'C' => Ok(GNSSSystem::BeiDou),
            'J' => Ok(GNSSSystem::QZSS),
            'I' => Ok(GNSSSystem::IRNSS),
            'S' => Ok(GNSSSystem::SBAS),
            'L' => Ok(GNSSSystem::LEO),
            _ => Err(sp3_error(&format!("unknown satellite system code '{}'", c))),
        }
    }

    /// One-letter system code.
    ///
    /// # Returns
    ///
    /// * `char` - One of `G`, `R`, `E`, `C`, `J`, `I`, `S`, `L`
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::GNSSSystem;
    ///
    /// assert_eq!(GNSSSystem::QZSS.code(), 'J');
    /// ```
    pub fn code(&self) -> char {
        match self {
            GNSSSystem::GPS => 'G',
            GNSSSystem::GLONASS => 'R',
            GNSSSystem::Galileo => 'E',
            GNSSSystem::BeiDou => 'C',
            GNSSSystem::QZSS => 'J',
            GNSSSystem::IRNSS => 'I',
            GNSSSystem::SBAS => 'S',
            GNSSSystem::LEO => 'L',
        }
    }
}

impl fmt::Display for GNSSSystem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = match self {
            GNSSSystem::GPS => "GPS",
            GNSSSystem::GLONASS => "GLONASS",
            GNSSSystem::Galileo => "Galileo",
            GNSSSystem::BeiDou => "BeiDou",
            GNSSSystem::QZSS => "QZSS",
            GNSSSystem::IRNSS => "IRNSS",
            GNSSSystem::SBAS => "SBAS",
            GNSSSystem::LEO => "LEO",
        };
        write!(f, "{}", name)
    }
}

/// Satellite system contained in an SP3 file.
///
/// The file type is the first value of the `%c` header line. It names a
/// single system, or `Mixed` for files carrying more than one.
///
/// # Examples
///
/// ```
/// use brahe::datafiles::sp3::SP3FileType;
///
/// assert_eq!(SP3FileType::from_code('M').unwrap(), SP3FileType::Mixed);
/// assert_eq!(SP3FileType::GPS.code(), 'G');
/// assert_eq!(SP3FileType::Mixed.to_string(), "Mixed");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SP3FileType {
    /// GPS satellites only (`G`)
    GPS,
    /// GLONASS satellites only (`R`)
    GLONASS,
    /// Galileo satellites only (`E`)
    Galileo,
    /// BeiDou satellites only (`C`)
    BeiDou,
    /// QZSS satellites only (`J`)
    QZSS,
    /// IRNSS satellites only (`I`)
    IRNSS,
    /// SBAS satellites only (`S`)
    SBAS,
    /// Low Earth orbiters only (`L`)
    LEO,
    /// More than one system (`M`)
    Mixed,
}

impl SP3FileType {
    /// Parse a file type code letter.
    ///
    /// # Arguments
    ///
    /// * `c` - File type code, one of `G`, `R`, `E`, `C`, `J`, `I`, `S`, `L`, `M`
    ///
    /// # Returns
    ///
    /// * `Result<SP3FileType, BraheError>` - The file type, or a parse error for any other character
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3FileType;
    ///
    /// assert_eq!(SP3FileType::from_code('E').unwrap(), SP3FileType::Galileo);
    /// assert!(SP3FileType::from_code('X').is_err());
    /// ```
    pub fn from_code(c: char) -> Result<Self, BraheError> {
        match c {
            'G' => Ok(SP3FileType::GPS),
            'R' => Ok(SP3FileType::GLONASS),
            'E' => Ok(SP3FileType::Galileo),
            'C' => Ok(SP3FileType::BeiDou),
            'J' => Ok(SP3FileType::QZSS),
            'I' => Ok(SP3FileType::IRNSS),
            'S' => Ok(SP3FileType::SBAS),
            'L' => Ok(SP3FileType::LEO),
            'M' => Ok(SP3FileType::Mixed),
            _ => Err(sp3_error(&format!("unknown file type code '{}'", c))),
        }
    }

    /// One-letter file type code.
    ///
    /// # Returns
    ///
    /// * `char` - One of `G`, `R`, `E`, `C`, `J`, `I`, `S`, `L`, `M`
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3FileType;
    ///
    /// assert_eq!(SP3FileType::Mixed.code(), 'M');
    /// ```
    pub fn code(&self) -> char {
        match self {
            SP3FileType::GPS => 'G',
            SP3FileType::GLONASS => 'R',
            SP3FileType::Galileo => 'E',
            SP3FileType::BeiDou => 'C',
            SP3FileType::QZSS => 'J',
            SP3FileType::IRNSS => 'I',
            SP3FileType::SBAS => 'S',
            SP3FileType::LEO => 'L',
            SP3FileType::Mixed => 'M',
        }
    }
}

impl fmt::Display for SP3FileType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = match self {
            SP3FileType::GPS => "GPS",
            SP3FileType::GLONASS => "GLONASS",
            SP3FileType::Galileo => "Galileo",
            SP3FileType::BeiDou => "BeiDou",
            SP3FileType::QZSS => "QZSS",
            SP3FileType::IRNSS => "IRNSS",
            SP3FileType::SBAS => "SBAS",
            SP3FileType::LEO => "LEO",
            SP3FileType::Mixed => "Mixed",
        };
        write!(f, "{}", name)
    }
}

/// Time system label of an SP3 file.
///
/// The label is the `%c` header value (`GPS`, `GLO`, `GAL`, `BDT`, `TAI`,
/// `UTC`, `QZS`, `IRN`). It is kept as written; [`SP3TimeSystem::to_time_system`]
/// maps it to a Brahe [`TimeSystem`] only where the two are the same time
/// scale, and applies no offsets.
///
/// # Examples
///
/// ```
/// use brahe::datafiles::sp3::SP3TimeSystem;
/// use brahe::time::TimeSystem;
///
/// assert_eq!(SP3TimeSystem::parse("GAL").unwrap(), SP3TimeSystem::GAL);
/// assert_eq!(SP3TimeSystem::GAL.to_time_system(), Some(TimeSystem::GST));
/// assert_eq!(SP3TimeSystem::GLO.to_time_system(), None);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SP3TimeSystem {
    /// GPS time
    GPS,
    /// GLONASS time
    GLO,
    /// Galileo System Time
    GAL,
    /// BeiDou Time
    BDT,
    /// International Atomic Time
    TAI,
    /// Coordinated Universal Time
    UTC,
    /// QZSS time
    QZS,
    /// IRNSS time
    IRN,
}

impl SP3TimeSystem {
    /// Parse a time system label.
    ///
    /// Surrounding whitespace is ignored. The SP3-a placeholder `ccc` is read
    /// as GPS time.
    ///
    /// # Arguments
    ///
    /// * `s` - Time system label, for example `GPS` or `GAL`
    ///
    /// # Returns
    ///
    /// * `Result<SP3TimeSystem, BraheError>` - The time system, or a parse error naming the label
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3TimeSystem;
    ///
    /// assert_eq!(SP3TimeSystem::parse("UTC").unwrap(), SP3TimeSystem::UTC);
    /// assert_eq!(SP3TimeSystem::parse("ccc").unwrap(), SP3TimeSystem::GPS);
    /// assert!(SP3TimeSystem::parse("XYZ").is_err());
    /// ```
    pub fn parse(s: &str) -> Result<Self, BraheError> {
        match s.trim() {
            "GPS" | "ccc" => Ok(SP3TimeSystem::GPS),
            "GLO" => Ok(SP3TimeSystem::GLO),
            "GAL" => Ok(SP3TimeSystem::GAL),
            "BDT" => Ok(SP3TimeSystem::BDT),
            "TAI" => Ok(SP3TimeSystem::TAI),
            "UTC" => Ok(SP3TimeSystem::UTC),
            "QZS" => Ok(SP3TimeSystem::QZS),
            "IRN" => Ok(SP3TimeSystem::IRN),
            other => Err(sp3_error(&format!("unknown time system '{}'", other))),
        }
    }

    /// Convert to the Brahe time system representing the same time scale.
    ///
    /// # Returns
    ///
    /// * `Option<TimeSystem>` - `GPS`, `GST` (for `GAL`), `BDT`, `TAI` or `UTC`; `None` for
    ///   `GLO`, `QZS` and `IRN`
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3TimeSystem;
    /// use brahe::time::TimeSystem;
    ///
    /// assert_eq!(SP3TimeSystem::GPS.to_time_system(), Some(TimeSystem::GPS));
    /// assert_eq!(SP3TimeSystem::QZS.to_time_system(), None);
    /// ```
    pub fn to_time_system(&self) -> Option<TimeSystem> {
        match self {
            SP3TimeSystem::GPS => Some(TimeSystem::GPS),
            SP3TimeSystem::GAL => Some(TimeSystem::GST),
            SP3TimeSystem::BDT => Some(TimeSystem::BDT),
            SP3TimeSystem::TAI => Some(TimeSystem::TAI),
            SP3TimeSystem::UTC => Some(TimeSystem::UTC),
            SP3TimeSystem::GLO | SP3TimeSystem::QZS | SP3TimeSystem::IRN => None,
        }
    }

    /// Convert a Brahe time system to the SP3 label for the same time scale.
    ///
    /// # Arguments
    ///
    /// * `ts` - Brahe time system
    ///
    /// # Returns
    ///
    /// * `Option<SP3TimeSystem>` - The label for `GPS`, `GST`, `BDT`, `TAI` or `UTC`; `None`
    ///   for any other time system
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3TimeSystem;
    /// use brahe::time::TimeSystem;
    ///
    /// assert_eq!(SP3TimeSystem::from_time_system(TimeSystem::GST), Some(SP3TimeSystem::GAL));
    /// assert_eq!(SP3TimeSystem::from_time_system(TimeSystem::TT), None);
    /// ```
    pub fn from_time_system(ts: TimeSystem) -> Option<Self> {
        match ts {
            TimeSystem::GPS => Some(SP3TimeSystem::GPS),
            TimeSystem::GST => Some(SP3TimeSystem::GAL),
            TimeSystem::BDT => Some(SP3TimeSystem::BDT),
            TimeSystem::TAI => Some(SP3TimeSystem::TAI),
            TimeSystem::UTC => Some(SP3TimeSystem::UTC),
            _ => None,
        }
    }
}

impl fmt::Display for SP3TimeSystem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let name = match self {
            SP3TimeSystem::GPS => "GPS",
            SP3TimeSystem::GLO => "GLO",
            SP3TimeSystem::GAL => "GAL",
            SP3TimeSystem::BDT => "BDT",
            SP3TimeSystem::TAI => "TAI",
            SP3TimeSystem::UTC => "UTC",
            SP3TimeSystem::QZS => "QZS",
            SP3TimeSystem::IRN => "IRN",
        };
        write!(f, "{}", name)
    }
}

/// Orbit type label of an SP3 file.
///
/// The standard labels are `FIT` (fit), `EXT` (extrapolated or predicted),
/// `BCT` (broadcast) and `HLM` (fit after Helmert transformation). Any other
/// label is preserved in `Other`.
///
/// # Examples
///
/// ```
/// use brahe::datafiles::sp3::SP3OrbitType;
///
/// assert_eq!(SP3OrbitType::parse("FIT"), SP3OrbitType::FIT);
/// assert_eq!(SP3OrbitType::parse("BHN"), SP3OrbitType::Other("BHN".into()));
/// assert_eq!(SP3OrbitType::parse("BHN").to_string(), "BHN");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SP3OrbitType {
    /// Fitted orbit
    FIT,
    /// Extrapolated or predicted orbit
    EXT,
    /// Broadcast orbit
    BCT,
    /// Fit after Helmert transformation
    HLM,
    /// Any other label, preserved as written
    Other(String),
}

impl SP3OrbitType {
    /// Parse an orbit type label.
    ///
    /// Surrounding whitespace is ignored. Labels other than the four standard
    /// ones are returned as `Other`.
    ///
    /// # Arguments
    ///
    /// * `s` - Orbit type label, for example `FIT` or `EXT`
    ///
    /// # Returns
    ///
    /// * `SP3OrbitType` - The orbit type
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3OrbitType;
    ///
    /// assert_eq!(SP3OrbitType::parse("EXT"), SP3OrbitType::EXT);
    /// assert_eq!(SP3OrbitType::parse("XYZ"), SP3OrbitType::Other("XYZ".into()));
    /// ```
    pub fn parse(s: &str) -> Self {
        match s.trim() {
            "FIT" => SP3OrbitType::FIT,
            "EXT" => SP3OrbitType::EXT,
            "BCT" => SP3OrbitType::BCT,
            "HLM" => SP3OrbitType::HLM,
            other => SP3OrbitType::Other(other.to_string()),
        }
    }
}

impl fmt::Display for SP3OrbitType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SP3OrbitType::FIT => write!(f, "FIT"),
            SP3OrbitType::EXT => write!(f, "EXT"),
            SP3OrbitType::BCT => write!(f, "BCT"),
            SP3OrbitType::HLM => write!(f, "HLM"),
            SP3OrbitType::Other(s) => write!(f, "{}", s),
        }
    }
}

/// Satellite identifier of an SP3 record.
///
/// Consists of a system and a number from 1 to 99. The text form is the
/// system code followed by a zero-padded two-digit number (`G05`). Identifiers
/// order by system (GPS, GLONASS, Galileo, BeiDou, QZSS, IRNSS, SBAS, LEO),
/// then by number.
///
/// # Examples
///
/// ```
/// use brahe::datafiles::sp3::{GNSSSystem, SP3SatelliteId};
///
/// let id = SP3SatelliteId::new(GNSSSystem::GPS, 5).unwrap();
/// assert_eq!(id.to_string(), "G05");
/// assert_eq!(SP3SatelliteId::parse("G 5").unwrap(), id);
/// assert_eq!("  5".parse::<SP3SatelliteId>().unwrap(), id);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SP3SatelliteId {
    /// Satellite system
    pub system: GNSSSystem,
    /// Satellite number within the system, 1 to 99
    pub number: u16,
}

impl SP3SatelliteId {
    /// Create a satellite identifier.
    ///
    /// # Arguments
    ///
    /// * `system` - Satellite system
    /// * `number` - Satellite number within the system, 1 to 99 inclusive
    ///
    /// # Returns
    ///
    /// * `Result<SP3SatelliteId, BraheError>` - The identifier, or an error if `number` is outside 1 to 99
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{GNSSSystem, SP3SatelliteId};
    ///
    /// assert!(SP3SatelliteId::new(GNSSSystem::Galileo, 12).is_ok());
    /// assert!(SP3SatelliteId::new(GNSSSystem::Galileo, 0).is_err());
    /// assert!(SP3SatelliteId::new(GNSSSystem::Galileo, 100).is_err());
    /// ```
    pub fn new(system: GNSSSystem, number: u16) -> Result<Self, BraheError> {
        if !(1..=99).contains(&number) {
            return Err(sp3_error(&format!(
                "satellite number {} out of range 1..=99",
                number
            )));
        }
        Ok(SP3SatelliteId { system, number })
    }

    /// Parse a three-character satellite identifier field.
    ///
    /// The first character is the system code, or a blank for GPS (SP3-a).
    /// The remaining two characters are the number, with a blank allowed in
    /// place of a leading zero.
    ///
    /// # Arguments
    ///
    /// * `s` - Identifier field of exactly three characters, for example `G05`, `G 5`, `R12` or `  5`
    ///
    /// # Returns
    ///
    /// * `Result<SP3SatelliteId, BraheError>` - The identifier, or a parse error for a malformed field
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{GNSSSystem, SP3SatelliteId};
    ///
    /// let id = SP3SatelliteId::parse("R12").unwrap();
    /// assert_eq!(id.system, GNSSSystem::GLONASS);
    /// assert_eq!(id.number, 12);
    /// assert_eq!(SP3SatelliteId::parse(" 05").unwrap().system, GNSSSystem::GPS);
    /// assert!(SP3SatelliteId::parse("X01").is_err());
    /// ```
    pub fn parse(s: &str) -> Result<Self, BraheError> {
        let mut chars = s.chars();
        let (sys_char, d1, d2) = match (chars.next(), chars.next(), chars.next(), chars.next()) {
            (Some(a), Some(b), Some(c), None) => (a, b, c),
            _ => {
                return Err(sp3_error(&format!(
                    "satellite identifier '{}' is not three characters",
                    s
                )));
            }
        };

        let system = if sys_char == ' ' {
            GNSSSystem::GPS
        } else {
            GNSSSystem::from_code(sys_char)?
        };

        let tens = match d1 {
            ' ' => Some(0),
            c => c.to_digit(10),
        };
        let number = match (tens, d2.to_digit(10)) {
            (Some(t), Some(u)) => t * 10 + u,
            _ => {
                return Err(sp3_error(&format!("invalid satellite number in '{}'", s)));
            }
        };

        SP3SatelliteId::new(system, number as u16)
    }
}

impl fmt::Display for SP3SatelliteId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}{:02}", self.system.code(), self.number)
    }
}

impl FromStr for SP3SatelliteId {
    type Err = BraheError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        SP3SatelliteId::parse(s)
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use serial_test::parallel;

    #[test]
    #[parallel]
    fn test_SP3SatelliteId_parse() {
        assert_eq!(
            SP3SatelliteId::parse("G05").unwrap(),
            SP3SatelliteId::new(GNSSSystem::GPS, 5).unwrap()
        );
        assert_eq!(SP3SatelliteId::parse("G 5").unwrap().to_string(), "G05");
        assert_eq!(
            SP3SatelliteId::parse("  5").unwrap().system,
            GNSSSystem::GPS
        );
        assert_eq!(SP3SatelliteId::parse(" 05").unwrap().number, 5);
        assert_eq!(
            SP3SatelliteId::parse("R12").unwrap().system,
            GNSSSystem::GLONASS
        );
        assert_eq!(
            "L01".parse::<SP3SatelliteId>().unwrap().system,
            GNSSSystem::LEO
        );
        assert!(SP3SatelliteId::parse("X01").is_err());
        assert!(SP3SatelliteId::new(GNSSSystem::GPS, 0).is_err());
        assert!(SP3SatelliteId::new(GNSSSystem::GPS, 100).is_err());
    }

    #[test]
    #[parallel]
    fn test_SP3SatelliteId_parse_invalid() {
        assert!(SP3SatelliteId::parse("").is_err());
        assert!(SP3SatelliteId::parse("G5").is_err());
        assert!(SP3SatelliteId::parse("G005").is_err());
        assert!(SP3SatelliteId::parse("G0A").is_err());
        assert!(SP3SatelliteId::parse("G+5").is_err());
        assert!(SP3SatelliteId::parse("G00").is_err());
        assert!(SP3SatelliteId::parse("G  ").is_err());
    }

    #[test]
    #[parallel]
    fn test_SP3SatelliteId_ordering() {
        let mut v: Vec<SP3SatelliteId> = ["E01", "G10", "G02", "R01"]
            .iter()
            .map(|s| s.parse().unwrap())
            .collect();
        v.sort();
        assert_eq!(
            v.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            ["G02", "G10", "R01", "E01"]
        );
    }

    #[test]
    #[parallel]
    fn test_SP3SatelliteId_Display() {
        let id = SP3SatelliteId::new(GNSSSystem::BeiDou, 7).unwrap();
        assert_eq!(id.to_string(), "C07");
        assert_eq!(SP3SatelliteId::parse(&id.to_string()).unwrap(), id);
    }

    #[test]
    #[parallel]
    fn test_SP3TimeSystem_to_time_system() {
        assert_eq!(SP3TimeSystem::parse("ccc").unwrap(), SP3TimeSystem::GPS);
        assert_eq!(SP3TimeSystem::GAL.to_time_system(), Some(TimeSystem::GST));
        assert_eq!(SP3TimeSystem::BDT.to_time_system(), Some(TimeSystem::BDT));
        for t in [SP3TimeSystem::GLO, SP3TimeSystem::QZS, SP3TimeSystem::IRN] {
            assert_eq!(t.to_time_system(), None);
        }
        assert_eq!(
            SP3TimeSystem::from_time_system(TimeSystem::GST),
            Some(SP3TimeSystem::GAL)
        );
        assert_eq!(SP3TimeSystem::from_time_system(TimeSystem::TT), None);
    }

    #[test]
    #[parallel]
    fn test_SP3TimeSystem_roundtrip() {
        for t in [
            SP3TimeSystem::GPS,
            SP3TimeSystem::GLO,
            SP3TimeSystem::GAL,
            SP3TimeSystem::BDT,
            SP3TimeSystem::TAI,
            SP3TimeSystem::UTC,
            SP3TimeSystem::QZS,
            SP3TimeSystem::IRN,
        ] {
            assert_eq!(SP3TimeSystem::parse(&t.to_string()).unwrap(), t);
        }
        assert_eq!(SP3TimeSystem::parse(" GPS ").unwrap(), SP3TimeSystem::GPS);
        assert!(SP3TimeSystem::parse("XYZ").is_err());
        for ts in [
            TimeSystem::GPS,
            TimeSystem::TAI,
            TimeSystem::UTC,
            TimeSystem::BDT,
        ] {
            let sp3 = SP3TimeSystem::from_time_system(ts).unwrap();
            assert_eq!(sp3.to_time_system(), Some(ts));
        }
    }

    #[test]
    #[parallel]
    fn test_SP3OrbitType_parse() {
        assert_eq!(SP3OrbitType::parse("FIT"), SP3OrbitType::FIT);
        assert_eq!(
            SP3OrbitType::parse("BHN"),
            SP3OrbitType::Other("BHN".into())
        );
        assert_eq!(SP3OrbitType::parse("BHN").to_string(), "BHN");
        for t in ["FIT", "EXT", "BCT", "HLM"] {
            assert_eq!(SP3OrbitType::parse(t).to_string(), t);
        }
        assert_eq!(SP3OrbitType::parse(" EXT "), SP3OrbitType::EXT);
    }

    #[test]
    #[parallel]
    fn test_SP3Version_roundtrip() {
        for (c, v) in [
            ('a', SP3Version::A),
            ('b', SP3Version::B),
            ('c', SP3Version::C),
            ('d', SP3Version::D),
        ] {
            assert_eq!(SP3Version::from_char(c).unwrap(), v);
            assert_eq!(SP3Version::from_char(c.to_ascii_uppercase()).unwrap(), v);
            assert_eq!(v.as_char(), c);
            assert_eq!(v.to_string(), c.to_string());
        }
        assert_eq!(SP3Version::from_char('D').unwrap(), SP3Version::D);
        assert!(SP3Version::from_char('e').is_err());
    }

    #[test]
    #[parallel]
    fn test_GNSSSystem_roundtrip() {
        let all = [
            (GNSSSystem::GPS, 'G', "GPS"),
            (GNSSSystem::GLONASS, 'R', "GLONASS"),
            (GNSSSystem::Galileo, 'E', "Galileo"),
            (GNSSSystem::BeiDou, 'C', "BeiDou"),
            (GNSSSystem::QZSS, 'J', "QZSS"),
            (GNSSSystem::IRNSS, 'I', "IRNSS"),
            (GNSSSystem::SBAS, 'S', "SBAS"),
            (GNSSSystem::LEO, 'L', "LEO"),
        ];
        for (sys, c, name) in all {
            assert_eq!(GNSSSystem::from_code(c).unwrap(), sys);
            assert_eq!(sys.code(), c);
            assert_eq!(sys.to_string(), name);
        }
        assert!(GNSSSystem::from_code('X').is_err());
        assert!(GNSSSystem::from_code('M').is_err());
    }

    #[test]
    #[parallel]
    fn test_SP3FileType_roundtrip() {
        let all = [
            (SP3FileType::GPS, 'G', "GPS"),
            (SP3FileType::GLONASS, 'R', "GLONASS"),
            (SP3FileType::Galileo, 'E', "Galileo"),
            (SP3FileType::BeiDou, 'C', "BeiDou"),
            (SP3FileType::QZSS, 'J', "QZSS"),
            (SP3FileType::IRNSS, 'I', "IRNSS"),
            (SP3FileType::SBAS, 'S', "SBAS"),
            (SP3FileType::LEO, 'L', "LEO"),
            (SP3FileType::Mixed, 'M', "Mixed"),
        ];
        for (ft, c, name) in all {
            assert_eq!(SP3FileType::from_code(c).unwrap(), ft);
            assert_eq!(ft.code(), c);
            assert_eq!(ft.to_string(), name);
        }
        assert!(SP3FileType::from_code('X').is_err());
    }
}
