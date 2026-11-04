//! SP3 enums and satellite identifiers.

use std::collections::HashSet;
use std::fmt;
use std::str::FromStr;

use crate::constants::{GPS_ZERO, SECONDS_PER_DAY};
use crate::datafiles::sp3::parse::sp3_error;
use crate::time::{Epoch, TimeSystem, datetime_to_mjd};
use crate::utils::BraheError;

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

/// Correlation record accompanying an SP3 `EP` or `EV` line.
///
/// Holds the standard deviations and the six off-diagonal correlation
/// coefficients of the position-clock (`EP`) or velocity-clock-rate (`EV`)
/// covariance. Standard deviations are in SI units; the correlation
/// coefficients are dimensionless.
///
/// # Examples
///
/// ```
/// use brahe::datafiles::sp3::SP3Correlation;
///
/// let corr = SP3Correlation {
///     std: [Some(0.01), Some(0.01), Some(0.02)],
///     clock_std: Some(1e-11),
///     xy: 0.1, xz: 0.0, xc: 0.0,
///     yz: -0.1, yc: 0.0, zc: 0.0,
/// };
/// assert_eq!(corr.std[2], Some(0.02));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SP3Correlation {
    /// Standard deviation of x, y, z. Units: m (`EP`) or m/s (`EV`)
    pub std: [Option<f64>; 3],
    /// Standard deviation of the clock. Units: s (`EP`) or s/s (`EV`)
    pub clock_std: Option<f64>,
    /// Correlation between x and y
    pub xy: f64,
    /// Correlation between x and z
    pub xz: f64,
    /// Correlation between x and the clock
    pub xc: f64,
    /// Correlation between y and z
    pub yz: f64,
    /// Correlation between y and the clock
    pub yc: f64,
    /// Correlation between z and the clock
    pub zc: f64,
}

/// One satellite state at one epoch of an SP3 file.
///
/// All quantities are SI. Optional fields are `None` when the file leaves
/// them blank or does not carry them.
///
/// # Examples
///
/// ```
/// use brahe::datafiles::sp3::SP3Record;
///
/// let record = SP3Record::new([15_000_000.0, 20_000_000.0, 5_000_000.0])
///     .with_velocity([1000.0, -2000.0, 500.0])
///     .with_clock(2.5e-5)
///     .with_clock_rate(1e-12);
/// assert_eq!(record.clock, Some(2.5e-5));
/// assert_eq!(record.velocity, Some([1000.0, -2000.0, 500.0]));
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SP3Record {
    /// Position in the file's coordinate system. Units: m
    pub position: [f64; 3],
    /// Satellite clock correction. Units: s
    pub clock: Option<f64>,
    /// Standard deviation of the position. Units: m
    pub position_std: [Option<f64>; 3],
    /// Standard deviation of the clock correction. Units: s
    pub clock_std: Option<f64>,
    /// Velocity in the file's coordinate system. Units: m/s
    pub velocity: Option<[f64; 3]>,
    /// Rate of change of the clock correction. Units: s/s
    pub clock_rate: Option<f64>,
    /// Standard deviation of the velocity. Units: m/s
    pub velocity_std: [Option<f64>; 3],
    /// Standard deviation of the clock rate. Units: s/s
    pub clock_rate_std: Option<f64>,
    /// Clock event flag
    pub clock_event: bool,
    /// Clock prediction flag
    pub clock_predicted: bool,
    /// Maneuver flag
    pub maneuver: bool,
    /// Orbit prediction flag
    pub orbit_predicted: bool,
    /// Position-clock correlation record (`EP` line)
    pub position_correlation: Option<SP3Correlation>,
    /// Velocity-clock-rate correlation record (`EV` line)
    pub velocity_correlation: Option<SP3Correlation>,
}

impl SP3Record {
    /// Create a record holding only a position.
    ///
    /// # Arguments
    ///
    /// * `position` - Position `[x, y, z]`. Units: m
    ///
    /// # Returns
    ///
    /// * `SP3Record` - Record with all other fields unset
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3Record;
    ///
    /// let record = SP3Record::new([1.0, 2.0, 3.0]);
    /// assert_eq!(record.position, [1.0, 2.0, 3.0]);
    /// assert!(record.velocity.is_none() && record.clock.is_none());
    /// ```
    pub fn new(position: [f64; 3]) -> Self {
        SP3Record {
            position,
            ..Default::default()
        }
    }

    /// Set the velocity.
    ///
    /// # Arguments
    ///
    /// * `velocity` - Velocity `[vx, vy, vz]`. Units: m/s
    ///
    /// # Returns
    ///
    /// * `SP3Record` - The record with the velocity set
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3Record;
    ///
    /// let record = SP3Record::new([1.0, 2.0, 3.0]).with_velocity([4.0, 5.0, 6.0]);
    /// assert_eq!(record.velocity, Some([4.0, 5.0, 6.0]));
    /// ```
    pub fn with_velocity(mut self, velocity: [f64; 3]) -> Self {
        self.velocity = Some(velocity);
        self
    }

    /// Set the clock correction.
    ///
    /// # Arguments
    ///
    /// * `clock` - Satellite clock correction. Units: s
    ///
    /// # Returns
    ///
    /// * `SP3Record` - The record with the clock correction set
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3Record;
    ///
    /// let record = SP3Record::new([1.0, 2.0, 3.0]).with_clock(1e-6);
    /// assert_eq!(record.clock, Some(1e-6));
    /// ```
    pub fn with_clock(mut self, clock: f64) -> Self {
        self.clock = Some(clock);
        self
    }

    /// Set the clock rate.
    ///
    /// # Arguments
    ///
    /// * `clock_rate` - Rate of change of the clock correction. Units: s/s
    ///
    /// # Returns
    ///
    /// * `SP3Record` - The record with the clock rate set
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3Record;
    ///
    /// let record = SP3Record::new([1.0, 2.0, 3.0]).with_clock_rate(1e-12);
    /// assert_eq!(record.clock_rate, Some(1e-12));
    /// ```
    pub fn with_clock_rate(mut self, clock_rate: f64) -> Self {
        self.clock_rate = Some(clock_rate);
        self
    }
}

/// SP3 file header.
///
/// Quantities are SI. `start_epoch`, `number_of_epochs`, `gps_week`,
/// `seconds_of_week`, `mjd`, and `fractional_day` describe the first epoch of
/// the file and are stored as read or as set by [`SP3::push_epoch`].
/// `position_base` and `clock_base` are the bases of the standard deviation
/// exponents in millimeters (or 1e-4 mm/s) and picoseconds (or 1e-4 ps/s)
/// respectively, and are `None` when the file declares none. `accuracies`
/// holds one entry per satellite in header order, in meters, with
/// `f64::INFINITY` for the "too large" marker and `None` for an unknown
/// accuracy.
///
/// # Examples
///
/// ```
/// use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3SatelliteId, SP3Version, SP3WriteOptions};
///
/// let sat = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
/// let sp3 = SP3::new(vec![sat], 900.0, SP3WriteOptions::default()).unwrap();
/// assert_eq!(sp3.header().version, SP3Version::D);
/// assert_eq!(sp3.header().agency, "BRHE");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SP3Header {
    /// SP3 format version
    pub version: SP3Version,
    /// Whether the file carries velocity records
    pub has_velocity: bool,
    /// First epoch of the file
    pub start_epoch: Epoch,
    /// Number of epochs in the file
    pub number_of_epochs: usize,
    /// Data used descriptor
    pub data_used: String,
    /// Coordinate system label
    pub coordinate_system: String,
    /// Orbit type
    pub orbit_type: SP3OrbitType,
    /// Producing agency
    pub agency: String,
    /// GPS week of the first epoch
    pub gps_week: u32,
    /// Seconds of the GPS week of the first epoch. Units: s
    pub seconds_of_week: f64,
    /// Epoch interval. Units: s
    pub epoch_interval: f64,
    /// Modified Julian day number of the first epoch
    pub mjd: u32,
    /// Fractional day of the first epoch
    pub fractional_day: f64,
    /// File type
    pub file_type: SP3FileType,
    /// Time system of the epochs
    pub time_system: SP3TimeSystem,
    /// Base of the position standard deviation exponent. Units: mm
    pub position_base: Option<f64>,
    /// Base of the clock standard deviation exponent. Units: ps
    pub clock_base: Option<f64>,
    /// Satellite accuracies in header order. Units: m
    pub accuracies: Vec<Option<f64>>,
    /// Comment lines
    pub comments: Vec<String>,
}

/// Options controlling the header of an SP3 file built in memory.
///
/// # Examples
///
/// ```
/// use brahe::datafiles::sp3::{SP3OrbitType, SP3WriteOptions};
///
/// let options = SP3WriteOptions::new()
///     .with_agency("ABC")
///     .with_orbit_type(SP3OrbitType::FIT)
///     .with_velocity(false);
/// assert_eq!(options.agency, "ABC");
/// assert_eq!(options.coordinate_system, "IGS20");
/// assert!(!options.include_velocity);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SP3WriteOptions {
    /// Producing agency, at most 4 characters
    pub agency: String,
    /// Time system of the epochs
    pub time_system: SP3TimeSystem,
    /// Coordinate system label, at most 5 characters
    pub coordinate_system: String,
    /// Orbit type
    pub orbit_type: SP3OrbitType,
    /// Data used descriptor, at most 5 characters
    pub data_used: String,
    /// Whether velocity records are written
    pub include_velocity: bool,
    /// Comment lines
    pub comments: Vec<String>,
}

impl Default for SP3WriteOptions {
    fn default() -> Self {
        SP3WriteOptions {
            agency: "BRHE".to_string(),
            time_system: SP3TimeSystem::GPS,
            coordinate_system: "IGS20".to_string(),
            orbit_type: SP3OrbitType::EXT,
            data_used: "ORBIT".to_string(),
            include_velocity: true,
            comments: Vec::new(),
        }
    }
}

impl SP3WriteOptions {
    /// Create options with the default values: agency `BRHE`, GPS time,
    /// coordinate system `IGS20`, orbit type `EXT`, data used `ORBIT`,
    /// velocity records included, and no comments.
    ///
    /// # Returns
    ///
    /// * `SP3WriteOptions` - Default options
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3WriteOptions;
    ///
    /// let options = SP3WriteOptions::new();
    /// assert_eq!(options.agency, "BRHE");
    /// assert!(options.include_velocity);
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the producing agency.
    ///
    /// # Arguments
    ///
    /// * `agency` - Agency label, at most 4 characters
    ///
    /// # Returns
    ///
    /// * `SP3WriteOptions` - The options with the agency set
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3WriteOptions;
    ///
    /// assert_eq!(SP3WriteOptions::new().with_agency("ESA").agency, "ESA");
    /// ```
    pub fn with_agency(mut self, agency: &str) -> Self {
        self.agency = agency.to_string();
        self
    }

    /// Set the time system of the epochs.
    ///
    /// # Arguments
    ///
    /// * `time_system` - Time system label
    ///
    /// # Returns
    ///
    /// * `SP3WriteOptions` - The options with the time system set
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{SP3TimeSystem, SP3WriteOptions};
    ///
    /// let options = SP3WriteOptions::new().with_time_system(SP3TimeSystem::UTC);
    /// assert_eq!(options.time_system, SP3TimeSystem::UTC);
    /// ```
    pub fn with_time_system(mut self, time_system: SP3TimeSystem) -> Self {
        self.time_system = time_system;
        self
    }

    /// Set the coordinate system label.
    ///
    /// # Arguments
    ///
    /// * `label` - Coordinate system label, at most 5 characters
    ///
    /// # Returns
    ///
    /// * `SP3WriteOptions` - The options with the coordinate system set
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3WriteOptions;
    ///
    /// let options = SP3WriteOptions::new().with_coordinate_system("IGb14");
    /// assert_eq!(options.coordinate_system, "IGb14");
    /// ```
    pub fn with_coordinate_system(mut self, label: &str) -> Self {
        self.coordinate_system = label.to_string();
        self
    }

    /// Set the orbit type.
    ///
    /// # Arguments
    ///
    /// * `orbit_type` - Orbit type
    ///
    /// # Returns
    ///
    /// * `SP3WriteOptions` - The options with the orbit type set
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{SP3OrbitType, SP3WriteOptions};
    ///
    /// let options = SP3WriteOptions::new().with_orbit_type(SP3OrbitType::FIT);
    /// assert_eq!(options.orbit_type, SP3OrbitType::FIT);
    /// ```
    pub fn with_orbit_type(mut self, orbit_type: SP3OrbitType) -> Self {
        self.orbit_type = orbit_type;
        self
    }

    /// Set the data used descriptor.
    ///
    /// # Arguments
    ///
    /// * `data_used` - Data used descriptor, at most 5 characters
    ///
    /// # Returns
    ///
    /// * `SP3WriteOptions` - The options with the data used descriptor set
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3WriteOptions;
    ///
    /// let options = SP3WriteOptions::new().with_data_used("DOPPL");
    /// assert_eq!(options.data_used, "DOPPL");
    /// ```
    pub fn with_data_used(mut self, data_used: &str) -> Self {
        self.data_used = data_used.to_string();
        self
    }

    /// Set whether velocity records are included.
    ///
    /// # Arguments
    ///
    /// * `include_velocity` - `true` to include velocity records
    ///
    /// # Returns
    ///
    /// * `SP3WriteOptions` - The options with the velocity flag set
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3WriteOptions;
    ///
    /// assert!(!SP3WriteOptions::new().with_velocity(false).include_velocity);
    /// ```
    pub fn with_velocity(mut self, include_velocity: bool) -> Self {
        self.include_velocity = include_velocity;
        self
    }

    /// Set the comment lines.
    ///
    /// # Arguments
    ///
    /// * `comments` - Comment lines
    ///
    /// # Returns
    ///
    /// * `SP3WriteOptions` - The options with the comments set
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3WriteOptions;
    ///
    /// let options = SP3WriteOptions::new().with_comments(vec!["generated".to_string()]);
    /// assert_eq!(options.comments.len(), 1);
    /// ```
    pub fn with_comments(mut self, comments: Vec<String>) -> Self {
        self.comments = comments;
        self
    }
}

/// Maximum number of satellites an SP3 file can declare.
const MAX_SATELLITES: usize = 999;

/// Tolerance for an epoch to be considered on the epoch grid. Units: s
const GRID_TOLERANCE: f64 = 1.0e-3;

/// Calendar-derived header fields for an epoch expressed in a time system.
///
/// Returns the GPS week, seconds of week, modified Julian day number, and
/// fractional day of the epoch, all in `time_system`.
fn header_time_fields(epoch: &Epoch, time_system: TimeSystem) -> (u32, f64, u32, f64) {
    let (year, month, day, hour, minute, second, nanosecond) =
        epoch.to_datetime_as_time_system(time_system);
    let mjd = datetime_to_mjd(year, month, day, 0, 0, 0.0, 0.0).round() as i64;
    let seconds_of_day =
        f64::from(hour) * 3600.0 + f64::from(minute) * 60.0 + second + nanosecond * 1.0e-9;
    let days_since_gps_zero = mjd - GPS_ZERO as i64;
    let week = days_since_gps_zero.div_euclid(7);
    let day_of_week = days_since_gps_zero.rem_euclid(7);
    (
        week as u32,
        day_of_week as f64 * SECONDS_PER_DAY + seconds_of_day,
        mjd as u32,
        seconds_of_day / SECONDS_PER_DAY,
    )
}

/// File type implied by the systems of a satellite list.
pub(crate) fn file_type_from_satellites(satellites: &[SP3SatelliteId]) -> SP3FileType {
    let Some(first) = satellites.first() else {
        return SP3FileType::Mixed;
    };
    if satellites.iter().any(|s| s.system != first.system) {
        return SP3FileType::Mixed;
    }
    match first.system {
        GNSSSystem::GPS => SP3FileType::GPS,
        GNSSSystem::GLONASS => SP3FileType::GLONASS,
        GNSSSystem::Galileo => SP3FileType::Galileo,
        GNSSSystem::BeiDou => SP3FileType::BeiDou,
        GNSSSystem::QZSS => SP3FileType::QZSS,
        GNSSSystem::IRNSS => SP3FileType::IRNSS,
        GNSSSystem::SBAS => SP3FileType::SBAS,
        GNSSSystem::LEO => SP3FileType::LEO,
    }
}

/// An SP3 precise orbit file held in memory.
///
/// Stores the header, an ascending list of epochs, the satellites in header
/// order, and a dense table of optional [`SP3Record`] values indexed by epoch
/// then satellite. A `None` entry means the satellite is absent at that epoch
/// or its position is the format's bad-value marker. All quantities are SI.
///
/// # Examples
///
/// ```
/// use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
/// use brahe::time::{Epoch, TimeSystem};
///
/// let sat = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
/// let mut sp3 = SP3::new(vec![sat], 900.0, SP3WriteOptions::default()).unwrap();
///
/// let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
/// let k = sp3.push_epoch(t0).unwrap();
/// sp3.set_record(sat, k, SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
///
/// assert_eq!(sp3.num_epochs(), 1);
/// assert!(sp3.record(sat, 0).unwrap().is_some());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SP3 {
    header: SP3Header,
    epochs: Vec<Epoch>,
    satellites: Vec<SP3SatelliteId>,
    records: Vec<Vec<Option<SP3Record>>>,
}

impl SP3 {
    /// Create an empty SP3 file with a fixed epoch grid.
    ///
    /// The header is filled from `options`: version D, a file type derived
    /// from the satellite systems (the common system when all satellites
    /// share one, `Mixed` otherwise), velocity records according to
    /// `include_velocity`, no standard deviation bases, and unknown
    /// accuracies. The start epoch and GPS week fields are set by the first
    /// call to [`SP3::push_epoch`]; until then they describe GPS epoch zero.
    ///
    /// # Arguments
    ///
    /// * `satellites` - Satellites in header order; at most 999, no duplicates
    /// * `epoch_interval` - Spacing of the epoch grid. Units: s
    /// * `options` - Header options; the time system must have a Brahe equivalent
    ///   (GPS, GAL, BDT, TAI, or UTC)
    ///
    /// # Returns
    ///
    /// * `Result<SP3, BraheError>` - The empty file, or an error for duplicate satellites,
    ///   more than 999 satellites, a non-positive or non-finite interval, or a time system
    ///   without a Brahe equivalent
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3FileType, SP3SatelliteId, SP3WriteOptions};
    ///
    /// let g01 = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
    /// let e01 = SP3SatelliteId::new(GNSSSystem::Galileo, 1).unwrap();
    /// let sp3 = SP3::new(vec![g01, e01], 300.0, SP3WriteOptions::default()).unwrap();
    /// assert_eq!(sp3.header().file_type, SP3FileType::Mixed);
    /// assert_eq!(sp3.num_satellites(), 2);
    /// ```
    pub fn new(
        satellites: Vec<SP3SatelliteId>,
        epoch_interval: f64,
        options: SP3WriteOptions,
    ) -> Result<Self, BraheError> {
        if satellites.len() > MAX_SATELLITES {
            return Err(sp3_error(&format!(
                "{} satellites exceeds the maximum of {}",
                satellites.len(),
                MAX_SATELLITES
            )));
        }
        let mut seen = HashSet::with_capacity(satellites.len());
        for id in &satellites {
            if !seen.insert(*id) {
                return Err(sp3_error(&format!("duplicate satellite {}", id)));
            }
        }
        if !(epoch_interval.is_finite() && epoch_interval > 0.0) {
            return Err(sp3_error(&format!(
                "epoch interval must be positive and finite, got {}",
                epoch_interval
            )));
        }
        let time_system = options.time_system.to_time_system().ok_or_else(|| {
            sp3_error(&format!(
                "time system {} has no Brahe equivalent",
                options.time_system
            ))
        })?;

        let start_epoch = Epoch::from_gps_date(0, 0.0);
        let (gps_week, seconds_of_week, mjd, fractional_day) =
            header_time_fields(&start_epoch, time_system);

        let header = SP3Header {
            version: SP3Version::D,
            has_velocity: options.include_velocity,
            start_epoch,
            number_of_epochs: 0,
            data_used: options.data_used,
            coordinate_system: options.coordinate_system,
            orbit_type: options.orbit_type,
            agency: options.agency,
            gps_week,
            seconds_of_week,
            epoch_interval,
            mjd,
            fractional_day,
            file_type: file_type_from_satellites(&satellites),
            time_system: options.time_system,
            position_base: None,
            clock_base: None,
            accuracies: vec![None; satellites.len()],
            comments: options.comments,
        };

        Ok(SP3 {
            header,
            epochs: Vec::new(),
            satellites,
            records: Vec::new(),
        })
    }

    /// Assemble a file from already validated parts.
    ///
    /// `records` must hold one row per epoch and one entry per satellite in
    /// each row.
    pub(crate) fn from_parts(
        header: SP3Header,
        epochs: Vec<Epoch>,
        satellites: Vec<SP3SatelliteId>,
        records: Vec<Vec<Option<SP3Record>>>,
    ) -> SP3 {
        SP3 {
            header,
            epochs,
            satellites,
            records,
        }
    }

    /// Append an epoch to the file.
    ///
    /// The epoch must be later than the last epoch and lie on the grid of
    /// `epoch_interval` anchored at the first epoch, to within 1 ms. Grid
    /// points may be skipped. The first epoch sets the start epoch, GPS week,
    /// seconds of week, MJD, and fractional day of the header. Every satellite
    /// is absent at the new epoch until [`SP3::set_record`] is called.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Epoch to append
    ///
    /// # Returns
    ///
    /// * `Result<usize, BraheError>` - Index of the new epoch, or an error if the epoch
    ///   is not after the last epoch or is off the grid
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3SatelliteId, SP3WriteOptions};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let sat = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
    /// let mut sp3 = SP3::new(vec![sat], 900.0, SP3WriteOptions::default()).unwrap();
    /// let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    ///
    /// assert_eq!(sp3.push_epoch(t0).unwrap(), 0);
    /// assert_eq!(sp3.push_epoch(t0 + 2700.0).unwrap(), 1);
    /// assert!(sp3.push_epoch(t0 + 3000.0).is_err());
    /// ```
    pub fn push_epoch(&mut self, epoch: Epoch) -> Result<usize, BraheError> {
        if let (Some(first), Some(last)) = (self.epochs.first(), self.epochs.last()) {
            if epoch - *last <= 0.0 {
                return Err(sp3_error(&format!(
                    "epoch {} is not after the last epoch {}",
                    epoch, last
                )));
            }
            let interval = self.header.epoch_interval;
            let offset = epoch - *first;
            if ((offset / interval).round() * interval - offset).abs() > GRID_TOLERANCE {
                return Err(sp3_error(&format!(
                    "epoch {} is not on the {} s grid starting at {}",
                    epoch, interval, first
                )));
            }
        } else {
            self.set_start_epoch(epoch);
        }

        self.epochs.push(epoch);
        self.records.push(vec![None; self.satellites.len()]);
        self.header.number_of_epochs = self.epochs.len();
        Ok(self.epochs.len() - 1)
    }

    /// Set the header fields derived from the first epoch.
    fn set_start_epoch(&mut self, epoch: Epoch) {
        let time_system = self
            .header
            .time_system
            .to_time_system()
            .unwrap_or(TimeSystem::GPS);
        let (gps_week, seconds_of_week, mjd, fractional_day) =
            header_time_fields(&epoch, time_system);
        self.header.start_epoch = epoch;
        self.header.gps_week = gps_week;
        self.header.seconds_of_week = seconds_of_week;
        self.header.mjd = mjd;
        self.header.fractional_day = fractional_day;
    }

    /// Store the record of a satellite at an epoch.
    ///
    /// A record whose position is exactly `[0, 0, 0]` marks the satellite as
    /// absent and is stored as `None`.
    ///
    /// # Arguments
    ///
    /// * `id` - Satellite identifier
    /// * `epoch_index` - Index of the epoch, as returned by [`SP3::push_epoch`]
    /// * `record` - Record to store
    ///
    /// # Returns
    ///
    /// * `Result<(), BraheError>` - An error if the satellite is not in the file or the
    ///   epoch index is out of range
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let sat = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
    /// let mut sp3 = SP3::new(vec![sat], 900.0, SP3WriteOptions::default()).unwrap();
    /// let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// sp3.push_epoch(t0).unwrap();
    ///
    /// sp3.set_record(sat, 0, SP3Record::new([2.0e7, 1.0e7, 1.0e7]).with_clock(1e-5)).unwrap();
    /// assert_eq!(sp3.record(sat, 0).unwrap().unwrap().clock, Some(1e-5));
    /// assert!(sp3.set_record(sat, 1, SP3Record::default()).is_err());
    /// ```
    pub fn set_record(
        &mut self,
        id: SP3SatelliteId,
        epoch_index: usize,
        record: SP3Record,
    ) -> Result<(), BraheError> {
        let sat = self.satellite_index(id)?;
        self.check_epoch_index(epoch_index)?;
        self.records[epoch_index][sat] = if record.position == [0.0; 3] {
            None
        } else {
            Some(record)
        };
        Ok(())
    }

    /// Header of the file.
    ///
    /// # Returns
    ///
    /// * `&SP3Header` - The header
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3SatelliteId, SP3WriteOptions};
    ///
    /// let sat = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
    /// let sp3 = SP3::new(vec![sat], 900.0, SP3WriteOptions::default()).unwrap();
    /// assert_eq!(sp3.header().epoch_interval, 900.0);
    /// ```
    pub fn header(&self) -> &SP3Header {
        &self.header
    }

    /// Mutable header of the file.
    ///
    /// # Returns
    ///
    /// * `&mut SP3Header` - The header
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3SatelliteId, SP3WriteOptions};
    ///
    /// let sat = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
    /// let mut sp3 = SP3::new(vec![sat], 900.0, SP3WriteOptions::default()).unwrap();
    /// sp3.header_mut().agency = "XYZ".to_string();
    /// assert_eq!(sp3.header().agency, "XYZ");
    /// ```
    pub fn header_mut(&mut self) -> &mut SP3Header {
        &mut self.header
    }

    /// Epochs of the file in ascending order.
    ///
    /// # Returns
    ///
    /// * `&[Epoch]` - The epochs
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3SatelliteId, SP3WriteOptions};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let sat = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
    /// let mut sp3 = SP3::new(vec![sat], 900.0, SP3WriteOptions::default()).unwrap();
    /// let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// sp3.push_epoch(t0).unwrap();
    /// assert_eq!(sp3.epochs(), &[t0]);
    /// ```
    pub fn epochs(&self) -> &[Epoch] {
        &self.epochs
    }

    /// Satellites of the file in header order.
    ///
    /// # Returns
    ///
    /// * `&[SP3SatelliteId]` - The satellite identifiers
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3SatelliteId, SP3WriteOptions};
    ///
    /// let sat = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
    /// let sp3 = SP3::new(vec![sat], 900.0, SP3WriteOptions::default()).unwrap();
    /// assert_eq!(sp3.satellite_ids(), &[sat]);
    /// ```
    pub fn satellite_ids(&self) -> &[SP3SatelliteId] {
        &self.satellites
    }

    /// Number of epochs in the file.
    ///
    /// # Returns
    ///
    /// * `usize` - The epoch count
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3SatelliteId, SP3WriteOptions};
    ///
    /// let sat = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
    /// let sp3 = SP3::new(vec![sat], 900.0, SP3WriteOptions::default()).unwrap();
    /// assert_eq!(sp3.num_epochs(), 0);
    /// ```
    pub fn num_epochs(&self) -> usize {
        self.epochs.len()
    }

    /// Number of satellites in the file.
    ///
    /// # Returns
    ///
    /// * `usize` - The satellite count
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3SatelliteId, SP3WriteOptions};
    ///
    /// let sat = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
    /// let sp3 = SP3::new(vec![sat], 900.0, SP3WriteOptions::default()).unwrap();
    /// assert_eq!(sp3.num_satellites(), 1);
    /// ```
    pub fn num_satellites(&self) -> usize {
        self.satellites.len()
    }

    /// First epoch of the file.
    ///
    /// # Returns
    ///
    /// * `Epoch` - The first epoch, or the header start epoch when the file has no epochs
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3SatelliteId, SP3WriteOptions};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let sat = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
    /// let mut sp3 = SP3::new(vec![sat], 900.0, SP3WriteOptions::default()).unwrap();
    /// let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// sp3.push_epoch(t0).unwrap();
    /// assert_eq!(sp3.start_epoch(), t0);
    /// ```
    pub fn start_epoch(&self) -> Epoch {
        self.epochs
            .first()
            .copied()
            .unwrap_or(self.header.start_epoch)
    }

    /// Last epoch of the file.
    ///
    /// # Returns
    ///
    /// * `Epoch` - The last epoch, or the header start epoch when the file has no epochs
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3SatelliteId, SP3WriteOptions};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let sat = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
    /// let mut sp3 = SP3::new(vec![sat], 900.0, SP3WriteOptions::default()).unwrap();
    /// let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// sp3.push_epoch(t0).unwrap();
    /// sp3.push_epoch(t0 + 900.0).unwrap();
    /// assert_eq!(sp3.end_epoch(), t0 + 900.0);
    /// ```
    pub fn end_epoch(&self) -> Epoch {
        self.epochs
            .last()
            .copied()
            .unwrap_or(self.header.start_epoch)
    }

    /// Whether the file declares a satellite.
    ///
    /// # Arguments
    ///
    /// * `id` - Satellite identifier
    ///
    /// # Returns
    ///
    /// * `bool` - `true` if the satellite is in the header satellite list
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3SatelliteId, SP3WriteOptions};
    ///
    /// let g01 = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
    /// let g02 = SP3SatelliteId::new(GNSSSystem::GPS, 2).unwrap();
    /// let sp3 = SP3::new(vec![g01], 900.0, SP3WriteOptions::default()).unwrap();
    /// assert!(sp3.contains(g01));
    /// assert!(!sp3.contains(g02));
    /// ```
    pub fn contains(&self, id: SP3SatelliteId) -> bool {
        self.satellites.contains(&id)
    }

    /// Record of a satellite at an epoch of the file grid.
    ///
    /// # Arguments
    ///
    /// * `id` - Satellite identifier
    /// * `epoch_index` - Index into [`SP3::epochs`]
    ///
    /// # Returns
    ///
    /// * `Result<Option<&SP3Record>, BraheError>` - The record, `None` if the satellite is
    ///   absent at that epoch, or an error if the satellite is not in the file or the epoch
    ///   index is out of range
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let sat = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
    /// let mut sp3 = SP3::new(vec![sat], 900.0, SP3WriteOptions::default()).unwrap();
    /// let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// sp3.push_epoch(t0).unwrap();
    /// assert!(sp3.record(sat, 0).unwrap().is_none());
    ///
    /// sp3.set_record(sat, 0, SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// assert_eq!(sp3.record(sat, 0).unwrap().unwrap().position[0], 2.0e7);
    /// ```
    pub fn record(
        &self,
        id: SP3SatelliteId,
        epoch_index: usize,
    ) -> Result<Option<&SP3Record>, BraheError> {
        let sat = self.satellite_index(id)?;
        self.check_epoch_index(epoch_index)?;
        Ok(self.records[epoch_index][sat].as_ref())
    }

    /// Index of a satellite in the header satellite list.
    pub(crate) fn satellite_index(&self, id: SP3SatelliteId) -> Result<usize, BraheError> {
        self.satellites
            .iter()
            .position(|s| *s == id)
            .ok_or_else(|| sp3_error(&format!("satellite {} is not in the file", id)))
    }

    fn check_epoch_index(&self, epoch_index: usize) -> Result<(), BraheError> {
        if epoch_index < self.epochs.len() {
            Ok(())
        } else {
            Err(sp3_error(&format!(
                "epoch index {} out of range for {} epochs",
                epoch_index,
                self.epochs.len()
            )))
        }
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

    fn gps_t0() -> Epoch {
        Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS)
    }

    fn sid(s: &str) -> SP3SatelliteId {
        s.parse().unwrap()
    }

    #[test]
    #[parallel]
    fn test_SP3_new_and_push_epoch() {
        let sats = vec![sid("G01"), sid("G02")];
        let mut sp3 = SP3::new(sats, 900.0, SP3WriteOptions::default()).unwrap();
        let t0 = gps_t0();
        assert_eq!(sp3.push_epoch(t0).unwrap(), 0);
        assert_eq!(sp3.push_epoch(t0 + 900.0).unwrap(), 1);
        assert!(sp3.push_epoch(t0 + 1000.0).is_err());
        assert!(sp3.push_epoch(t0).is_err());
        assert_eq!(sp3.push_epoch(t0 + 2700.0).unwrap(), 2);
        let id = sid("G02");
        sp3.set_record(id, 1, SP3Record::new([1.0, 2.0, 3.0]).with_clock(1e-6))
            .unwrap();
        assert_eq!(sp3.record(id, 1).unwrap().unwrap().clock, Some(1e-6));
        assert!(sp3.record(id, 0).unwrap().is_none());
        assert!(sp3.record(sid("G03"), 0).is_err());
        assert!(sp3.set_record(id, 9, SP3Record::default()).is_err());
        assert!(sp3.set_record(sid("G03"), 0, SP3Record::default()).is_err());
        assert_eq!(sp3.header().agency, "BRHE");
        assert_eq!(sp3.header().time_system, SP3TimeSystem::GPS);
        assert!(sp3.header().has_velocity);
    }

    #[test]
    #[parallel]
    fn test_SP3_new_header_fields() {
        let sp3 = SP3::new(
            vec![sid("G01"), sid("G02"), sid("G03")],
            300.0,
            SP3WriteOptions::default(),
        )
        .unwrap();
        let h = sp3.header();
        assert_eq!(h.version, SP3Version::D);
        assert_eq!(h.file_type, SP3FileType::GPS);
        assert_eq!(h.epoch_interval, 300.0);
        assert_eq!(h.number_of_epochs, 0);
        assert_eq!(h.accuracies, vec![None, None, None]);
        assert_eq!(h.position_base, None);
        assert_eq!(h.clock_base, None);
        assert_eq!(h.orbit_type, SP3OrbitType::EXT);
        assert_eq!(h.coordinate_system, "IGS20");
        assert_eq!(h.data_used, "ORBIT");
        assert_eq!(sp3.num_satellites(), 3);
        assert_eq!(sp3.num_epochs(), 0);
    }

    #[test]
    #[parallel]
    fn test_SP3_empty_start_end_epoch() {
        let sp3 = SP3::new(vec![sid("G01")], 900.0, SP3WriteOptions::default()).unwrap();
        let zero = Epoch::from_gps_date(0, 0.0);
        assert_eq!(sp3.start_epoch(), zero);
        assert_eq!(sp3.end_epoch(), zero);
        assert_eq!(sp3.header().start_epoch, zero);
        assert_eq!(sp3.header().gps_week, 0);
        assert_eq!(sp3.header().seconds_of_week, 0.0);
        assert_eq!(sp3.header().mjd, 44244);
        assert_eq!(sp3.header().fractional_day, 0.0);
    }

    #[test]
    #[parallel]
    fn test_SP3_first_push_epoch_sets_header_fields() {
        let mut sp3 = SP3::new(vec![sid("G01")], 900.0, SP3WriteOptions::default()).unwrap();
        let t0 = Epoch::from_datetime(2024, 11, 10, 6, 0, 0.0, 0.0, TimeSystem::GPS);
        sp3.push_epoch(t0).unwrap();
        sp3.push_epoch(t0 + 900.0).unwrap();
        let h = sp3.header();
        assert_eq!(h.start_epoch, t0);
        assert_eq!(h.number_of_epochs, 2);
        assert_eq!(h.gps_week, 2340);
        assert_eq!(h.seconds_of_week, 6.0 * 3600.0);
        assert_eq!(h.mjd, 60624);
        assert_eq!(h.fractional_day, 0.25);
        assert_eq!(sp3.start_epoch(), t0);
        assert_eq!(sp3.end_epoch(), t0 + 900.0);
        assert_eq!(sp3.epochs(), &[t0, t0 + 900.0]);
    }

    #[test]
    #[parallel]
    fn test_SP3_set_record_zero_position_is_absent() {
        let mut sp3 = SP3::new(vec![sid("G01")], 900.0, SP3WriteOptions::default()).unwrap();
        sp3.push_epoch(gps_t0()).unwrap();
        sp3.set_record(sid("G01"), 0, SP3Record::new([1.0, 2.0, 3.0]))
            .unwrap();
        assert!(sp3.record(sid("G01"), 0).unwrap().is_some());
        sp3.set_record(sid("G01"), 0, SP3Record::new([0.0, 0.0, 0.0]))
            .unwrap();
        assert!(sp3.record(sid("G01"), 0).unwrap().is_none());
    }

    #[test]
    #[parallel]
    fn test_SP3_contains_and_satellite_ids() {
        let sp3 = SP3::new(
            vec![sid("G02"), sid("G01")],
            900.0,
            SP3WriteOptions::default(),
        )
        .unwrap();
        assert!(sp3.contains(sid("G01")));
        assert!(!sp3.contains(sid("G03")));
        assert_eq!(sp3.satellite_ids(), &[sid("G02"), sid("G01")]);
        assert_eq!(sp3.satellite_index(sid("G01")).unwrap(), 1);
        assert!(sp3.satellite_index(sid("E01")).is_err());
    }

    #[test]
    #[parallel]
    fn test_SP3_header_mut() {
        let mut sp3 = SP3::new(vec![sid("G01")], 900.0, SP3WriteOptions::default()).unwrap();
        sp3.header_mut().agency = "XYZ".to_string();
        assert_eq!(sp3.header().agency, "XYZ");
    }

    #[test]
    #[parallel]
    fn test_SP3_new_options_applied() {
        let opts = SP3WriteOptions::new()
            .with_agency("ABC")
            .with_time_system(SP3TimeSystem::UTC)
            .with_coordinate_system("ITRF2")
            .with_orbit_type(SP3OrbitType::FIT)
            .with_data_used("DOPPL")
            .with_velocity(false)
            .with_comments(vec!["note".to_string()]);
        let sp3 = SP3::new(vec![sid("G01")], 60.0, opts).unwrap();
        let h = sp3.header();
        assert_eq!(h.agency, "ABC");
        assert_eq!(h.time_system, SP3TimeSystem::UTC);
        assert_eq!(h.coordinate_system, "ITRF2");
        assert_eq!(h.orbit_type, SP3OrbitType::FIT);
        assert_eq!(h.data_used, "DOPPL");
        assert!(!h.has_velocity);
        assert_eq!(h.comments, vec!["note".to_string()]);
    }

    #[test]
    #[parallel]
    fn test_SP3_new_errors() {
        let opts = SP3WriteOptions::default;
        assert!(SP3::new(vec![sid("G01"), sid("G01")], 900.0, opts()).is_err());
        assert!(SP3::new(vec![sid("G01")], 0.0, opts()).is_err());
        assert!(SP3::new(vec![sid("G01")], -900.0, opts()).is_err());
        assert!(SP3::new(vec![sid("G01")], f64::NAN, opts()).is_err());
        assert!(
            SP3::new(
                vec![sid("G01")],
                900.0,
                opts().with_time_system(SP3TimeSystem::GLO)
            )
            .is_err()
        );

        let many: Vec<SP3SatelliteId> = [GNSSSystem::GPS, GNSSSystem::GLONASS, GNSSSystem::Galileo]
            .iter()
            .flat_map(|&s| (1..=99u16).map(move |n| SP3SatelliteId::new(s, n).unwrap()))
            .chain(
                [
                    GNSSSystem::BeiDou,
                    GNSSSystem::QZSS,
                    GNSSSystem::IRNSS,
                    GNSSSystem::SBAS,
                ]
                .iter()
                .flat_map(|&s| (1..=99u16).map(move |n| SP3SatelliteId::new(s, n).unwrap())),
            )
            .chain((1..=99u16).map(|n| SP3SatelliteId::new(GNSSSystem::LEO, n).unwrap()))
            .collect();
        assert_eq!(many.len(), 792);
        assert!(SP3::new(many, 900.0, opts()).is_ok());
    }

    #[test]
    #[parallel]
    fn test_SP3_new_rejects_more_than_999_satellites() {
        // Only 8 systems x 99 numbers = 792 distinct identifiers exist, so the
        // limit is checked on the input length before duplicates.
        let ids: Vec<SP3SatelliteId> = std::iter::repeat_n(sid("G01"), 1000).collect();
        let err = SP3::new(ids, 900.0, SP3WriteOptions::default()).unwrap_err();
        assert!(err.to_string().contains("999"));
    }

    #[test]
    #[parallel]
    fn test_SP3_new_file_type_from_systems() {
        let ft = |ids: &[&str]| {
            SP3::new(
                ids.iter().map(|s| sid(s)).collect(),
                900.0,
                SP3WriteOptions::default(),
            )
            .unwrap()
            .header()
            .file_type
        };
        assert_eq!(ft(&["G01", "G02"]), SP3FileType::GPS);
        assert_eq!(ft(&["R01"]), SP3FileType::GLONASS);
        assert_eq!(ft(&["E01"]), SP3FileType::Galileo);
        assert_eq!(ft(&["C01"]), SP3FileType::BeiDou);
        assert_eq!(ft(&["J01"]), SP3FileType::QZSS);
        assert_eq!(ft(&["I01"]), SP3FileType::IRNSS);
        assert_eq!(ft(&["S01"]), SP3FileType::SBAS);
        assert_eq!(ft(&["L01", "L02"]), SP3FileType::LEO);
        assert_eq!(ft(&["G01", "E01"]), SP3FileType::Mixed);
        assert_eq!(ft(&["G01", "L01"]), SP3FileType::Mixed);
    }

    #[test]
    #[parallel]
    fn test_SP3_push_epoch_non_gps_time_system() {
        let opts = SP3WriteOptions::default().with_time_system(SP3TimeSystem::UTC);
        let mut sp3 = SP3::new(vec![sid("G01")], 900.0, opts).unwrap();
        let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        sp3.push_epoch(t0).unwrap();
        assert_eq!(sp3.header().mjd, 60624);
        assert_eq!(sp3.header().fractional_day, 0.0);
    }

    #[test]
    #[parallel]
    fn test_SP3WriteOptions_defaults() {
        let o = SP3WriteOptions::default();
        assert_eq!(
            (
                o.agency.as_str(),
                o.coordinate_system.as_str(),
                o.data_used.as_str()
            ),
            ("BRHE", "IGS20", "ORBIT")
        );
        assert_eq!(o.orbit_type, SP3OrbitType::EXT);
        assert_eq!(o.time_system, SP3TimeSystem::GPS);
        assert!(o.include_velocity && o.comments.is_empty());
        assert_eq!(SP3WriteOptions::new(), o);
    }

    #[test]
    #[parallel]
    fn test_SP3Record_builders() {
        let r = SP3Record::new([1.0, 2.0, 3.0])
            .with_velocity([4.0, 5.0, 6.0])
            .with_clock(1e-6)
            .with_clock_rate(1e-12);
        assert_eq!(r.position, [1.0, 2.0, 3.0]);
        assert_eq!(r.velocity, Some([4.0, 5.0, 6.0]));
        assert_eq!(r.clock, Some(1e-6));
        assert_eq!(r.clock_rate, Some(1e-12));
        assert_eq!(r.position_std, [None, None, None]);
        assert!(!r.clock_event && !r.clock_predicted && !r.maneuver && !r.orbit_predicted);
        assert!(r.position_correlation.is_none() && r.velocity_correlation.is_none());
        assert_eq!(SP3Record::default().position, [0.0; 3]);
    }

    #[test]
    #[parallel]
    fn test_SP3_from_parts() {
        let header = SP3::new(vec![sid("G01")], 900.0, SP3WriteOptions::default())
            .unwrap()
            .header()
            .clone();
        let t0 = gps_t0();
        let records = vec![vec![Some(SP3Record::new([1.0, 2.0, 3.0]))], vec![None]];
        let sp3 = SP3::from_parts(header, vec![t0, t0 + 900.0], vec![sid("G01")], records);
        assert_eq!(sp3.num_epochs(), 2);
        assert_eq!(sp3.record(sid("G01"), 0).unwrap().unwrap().position[1], 2.0);
        assert!(sp3.record(sid("G01"), 1).unwrap().is_none());
        assert!(sp3.record(sid("G01"), 2).is_err());
    }
}
