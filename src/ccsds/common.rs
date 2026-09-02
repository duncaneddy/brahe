/*!
 * Common types shared across CCSDS message formats.
 *
 * This module defines the shared data structures used by OEM, OMM, and OPM
 * message types, including the ODM header, reference frame and time system
 * enumerations, covariance matrices, and spacecraft parameters.
 */

use std::collections::HashMap;
use std::fmt;

use nalgebra::SMatrix;
use serde::{Deserialize, Serialize};

use crate::time::Epoch;

/// Controls the casing of CCSDS data field keys in JSON output.
///
/// Container/structural keys (e.g., `"header"`, `"segments"`, `"metadata"`) are
/// always lowercase regardless of this setting. Only CCSDS data field keywords
/// (e.g., `OBJECT_NAME`, `CREATION_DATE`, `X`, `Y`) are affected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CCSDSJsonKeyCase {
    /// Lowercase keys (default): `"object_name"`, `"creation_date"`
    Lower,
    /// Uppercase CCSDS keywords: `"OBJECT_NAME"`, `"CREATION_DATE"`
    Upper,
}

/// CCSDS message encoding format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CCSDSFormat {
    /// Keyword=Value Notation (text-based)
    KVN,
    /// XML encoding
    XML,
    /// JSON encoding
    JSON,
}

impl fmt::Display for CCSDSFormat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CCSDSFormat::KVN => write!(f, "KVN"),
            CCSDSFormat::XML => write!(f, "XML"),
            CCSDSFormat::JSON => write!(f, "JSON"),
        }
    }
}

/// Auto-detect the encoding format of a CCSDS message string.
///
/// Detection logic:
/// - Starts with `<?xml` or `<`: XML
/// - Starts with `{` or `[`: JSON
/// - Otherwise: KVN (default)
///
/// # Arguments
///
/// * `content` - String content of the CCSDS message
///
/// # Returns
///
/// * `CCSDSFormat` - Detected format
pub(crate) fn detect_format(content: &str) -> CCSDSFormat {
    let trimmed = content.trim_start();
    if trimmed.starts_with("<?xml") || trimmed.starts_with('<') {
        CCSDSFormat::XML
    } else if trimmed.starts_with('{') || trimmed.starts_with('[') {
        CCSDSFormat::JSON
    } else {
        CCSDSFormat::KVN
    }
}

/// CCSDS time system identifier.
///
/// Maps CCSDS time system keywords to their standard definitions.
/// Only UTC, TAI, GPS, TT, and UT1 can be directly converted to brahe `TimeSystem`;
/// other values are preserved for round-trip fidelity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CCSDSTimeSystem {
    /// Coordinated Universal Time
    UTC,
    /// International Atomic Time
    TAI,
    /// Global Positioning System time
    GPS,
    /// Terrestrial Time
    TT,
    /// Universal Time 1
    UT1,
    /// Barycentric Dynamical Time
    TDB,
    /// Barycentric Coordinate Time
    TCB,
    /// Tracking Data Relay time
    TDR,
    /// Geocentric Coordinate Time
    TCG,
    /// BeiDou Navigation Satellite System Time
    BDT,
    /// Galileo System Time
    GST,
    /// Greenwich Mean Sidereal Time
    GMST,
    /// Mission Elapsed Time
    MET,
    /// Mission Relative Time
    MRT,
    /// Spacecraft Clock
    SCLK,
}

impl fmt::Display for CCSDSTimeSystem {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CCSDSTimeSystem::UTC => write!(f, "UTC"),
            CCSDSTimeSystem::TAI => write!(f, "TAI"),
            CCSDSTimeSystem::GPS => write!(f, "GPS"),
            CCSDSTimeSystem::TT => write!(f, "TT"),
            CCSDSTimeSystem::UT1 => write!(f, "UT1"),
            CCSDSTimeSystem::TDB => write!(f, "TDB"),
            CCSDSTimeSystem::TCB => write!(f, "TCB"),
            CCSDSTimeSystem::TDR => write!(f, "TDR"),
            CCSDSTimeSystem::TCG => write!(f, "TCG"),
            CCSDSTimeSystem::BDT => write!(f, "BDT"),
            CCSDSTimeSystem::GST => write!(f, "GST"),
            CCSDSTimeSystem::GMST => write!(f, "GMST"),
            CCSDSTimeSystem::MET => write!(f, "MET"),
            CCSDSTimeSystem::MRT => write!(f, "MRT"),
            CCSDSTimeSystem::SCLK => write!(f, "SCLK"),
        }
    }
}

impl CCSDSTimeSystem {
    /// Parse a CCSDS time system string.
    pub fn parse(s: &str) -> Result<Self, crate::utils::errors::BraheError> {
        match s.trim() {
            "UTC" => Ok(CCSDSTimeSystem::UTC),
            "TAI" => Ok(CCSDSTimeSystem::TAI),
            "GPS" => Ok(CCSDSTimeSystem::GPS),
            "TT" => Ok(CCSDSTimeSystem::TT),
            "UT1" => Ok(CCSDSTimeSystem::UT1),
            "TDB" => Ok(CCSDSTimeSystem::TDB),
            "TCB" => Ok(CCSDSTimeSystem::TCB),
            "TDR" => Ok(CCSDSTimeSystem::TDR),
            "TCG" => Ok(CCSDSTimeSystem::TCG),
            "BDT" => Ok(CCSDSTimeSystem::BDT),
            "GST" => Ok(CCSDSTimeSystem::GST),
            "GMST" => Ok(CCSDSTimeSystem::GMST),
            "MET" => Ok(CCSDSTimeSystem::MET),
            "MRT" => Ok(CCSDSTimeSystem::MRT),
            "SCLK" => Ok(CCSDSTimeSystem::SCLK),
            _ => Err(crate::ccsds::error::ccsds_parse_error(
                "common",
                &format!("unknown time system '{}'", s),
            )),
        }
    }

    /// Convert to brahe `TimeSystem` if the CCSDS time system has a direct mapping.
    pub fn to_time_system(&self) -> Option<crate::time::TimeSystem> {
        match self {
            CCSDSTimeSystem::UTC => Some(crate::time::TimeSystem::UTC),
            CCSDSTimeSystem::TAI => Some(crate::time::TimeSystem::TAI),
            CCSDSTimeSystem::GPS => Some(crate::time::TimeSystem::GPS),
            CCSDSTimeSystem::TT => Some(crate::time::TimeSystem::TT),
            CCSDSTimeSystem::UT1 => Some(crate::time::TimeSystem::UT1),
            CCSDSTimeSystem::TDB => Some(crate::time::TimeSystem::TDB),
            CCSDSTimeSystem::TCG => Some(crate::time::TimeSystem::TCG),
            CCSDSTimeSystem::TCB => Some(crate::time::TimeSystem::TCB),
            CCSDSTimeSystem::BDT => Some(crate::time::TimeSystem::BDT),
            CCSDSTimeSystem::GST => Some(crate::time::TimeSystem::GST),
            _ => None,
        }
    }
}

/// CCSDS reference frame identifier.
///
/// Covers the reference frames defined in CCSDS 502.0-B-3. Includes both
/// celestial body frames and orbit-relative frames used for covariance.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CCSDSRefFrame {
    /// Earth Mean Equator and Equinox of J2000.0
    EME2000,
    /// Geocentric Celestial Reference Frame
    GCRF,
    /// International Terrestrial Reference Frame (2000)
    ITRF2000,
    /// International Terrestrial Reference Frame (1993)
    ITRF93,
    /// International Terrestrial Reference Frame (1997)
    ITRF97,
    /// International Terrestrial Reference Frame (2005)
    ITRF2005,
    /// International Terrestrial Reference Frame (2008)
    ITRF2008,
    /// International Terrestrial Reference Frame (2014)
    ITRF2014,
    /// True Equator Mean Equinox (used by SGP4)
    TEME,
    /// True of Date
    TOD,
    /// J2000 (alias for EME2000)
    J2000,
    /// Tracking Data Relay frame
    TDR,
    /// Radial-Transverse-Normal (orbit-relative)
    RTN,
    /// Transverse-Normal-Along (orbit-relative)
    TNW,
    /// Radial-Along-Cross (orbit-relative)
    RSW,
    /// Other non-standard frame
    Other(String),
}

impl fmt::Display for CCSDSRefFrame {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            CCSDSRefFrame::EME2000 => write!(f, "EME2000"),
            CCSDSRefFrame::GCRF => write!(f, "GCRF"),
            CCSDSRefFrame::ITRF2000 => write!(f, "ITRF2000"),
            CCSDSRefFrame::ITRF93 => write!(f, "ITRF93"),
            CCSDSRefFrame::ITRF97 => write!(f, "ITRF97"),
            CCSDSRefFrame::ITRF2005 => write!(f, "ITRF2005"),
            CCSDSRefFrame::ITRF2008 => write!(f, "ITRF2008"),
            CCSDSRefFrame::ITRF2014 => write!(f, "ITRF2014"),
            CCSDSRefFrame::TEME => write!(f, "TEME"),
            CCSDSRefFrame::TOD => write!(f, "TOD"),
            CCSDSRefFrame::J2000 => write!(f, "J2000"),
            CCSDSRefFrame::TDR => write!(f, "TDR"),
            CCSDSRefFrame::RTN => write!(f, "RTN"),
            CCSDSRefFrame::TNW => write!(f, "TNW"),
            CCSDSRefFrame::RSW => write!(f, "RSW"),
            CCSDSRefFrame::Other(s) => write!(f, "{}", s),
        }
    }
}

impl CCSDSRefFrame {
    /// Parse a CCSDS reference frame string.
    ///
    /// Known frames are mapped to their enum variants. Unknown frames are
    /// stored as `Other(String)` to preserve round-trip fidelity.
    pub fn parse(s: &str) -> Self {
        match s.trim() {
            "EME2000" => CCSDSRefFrame::EME2000,
            "GCRF" => CCSDSRefFrame::GCRF,
            "ITRF2000" | "ITRF-2000" => CCSDSRefFrame::ITRF2000,
            "ITRF93" | "ITRF-93" => CCSDSRefFrame::ITRF93,
            "ITRF97" | "ITRF-97" | "ITRF1997" => CCSDSRefFrame::ITRF97,
            "ITRF2005" | "ITRF-2005" => CCSDSRefFrame::ITRF2005,
            "ITRF2008" | "ITRF-2008" => CCSDSRefFrame::ITRF2008,
            "ITRF2014" | "ITRF-2014" => CCSDSRefFrame::ITRF2014,
            "TEME" => CCSDSRefFrame::TEME,
            "TOD" => CCSDSRefFrame::TOD,
            "J2000" => CCSDSRefFrame::J2000,
            "TDR" => CCSDSRefFrame::TDR,
            "RTN" => CCSDSRefFrame::RTN,
            "TNW" => CCSDSRefFrame::TNW,
            "RSW" => CCSDSRefFrame::RSW,
            other => CCSDSRefFrame::Other(other.to_string()),
        }
    }
}

/// Common ODM header present in all CCSDS message types.
#[derive(Debug, Clone)]
pub struct ODMHeader {
    /// CCSDS format version number (e.g., 2.0 or 3.0)
    pub format_version: f64,
    /// Optional classification string (e.g., "public, test-data")
    pub classification: Option<String>,
    /// Creation date of the message
    pub creation_date: Epoch,
    /// Originator of the message
    pub originator: String,
    /// Optional unique message identifier
    pub message_id: Option<String>,
    /// Comments associated with the header
    pub comments: Vec<String>,
}

/// 6x6 covariance matrix with optional epoch and reference frame.
///
/// The matrix is stored in SI units:
/// - Position-position: m²
/// - Position-velocity: m²/s
/// - Velocity-velocity: m²/s²
///
/// CCSDS files store covariance in km²/km²s⁻¹/km²s⁻² which is converted
/// on parse and converted back on write.
#[derive(Debug, Clone)]
pub struct CCSDSCovariance {
    /// Optional epoch for the covariance (if different from state epoch)
    pub epoch: Option<Epoch>,
    /// Optional reference frame for the covariance
    pub cov_ref_frame: Option<CCSDSRefFrame>,
    /// 6x6 symmetric covariance matrix in SI units (m², m²/s, m²/s²)
    pub matrix: SMatrix<f64, 6, 6>,
    /// Comments associated with this covariance block
    pub comments: Vec<String>,
}

/// Spacecraft physical parameters.
#[derive(Debug, Clone)]
pub struct CCSDSSpacecraftParameters {
    /// Spacecraft mass. Units: kg
    pub mass: Option<f64>,
    /// Solar radiation pressure area. Units: m²
    pub solar_rad_area: Option<f64>,
    /// Solar radiation pressure coefficient (dimensionless)
    pub solar_rad_coeff: Option<f64>,
    /// Atmospheric drag area. Units: m²
    pub drag_area: Option<f64>,
    /// Atmospheric drag coefficient (dimensionless)
    pub drag_coeff: Option<f64>,
    /// Comments associated with spacecraft parameters
    pub comments: Vec<String>,
}

/// User-defined parameters.
#[derive(Debug, Clone)]
pub struct CCSDSUserDefined {
    /// Map of parameter names to string values.
    /// Keys are stored without the "USER_DEFINED_" prefix.
    pub parameters: HashMap<String, String>,
}

/// Parse a CCSDS datetime string into an Epoch.
///
/// Handles both calendar format (`YYYY-MM-DDThh:mm:ss.sss`) and
/// day-of-year format (`YYYY-DDDThh:mm:ss.sss`).
///
/// The time system parameter specifies which time system the epoch
/// should be created in (CCSDS dates don't carry time system info
/// in the string itself).
pub fn parse_ccsds_datetime(
    s: &str,
    time_system: &CCSDSTimeSystem,
) -> Result<Epoch, crate::utils::errors::BraheError> {
    let s = s.trim();
    let ts = time_system.to_time_system().ok_or_else(|| {
        crate::ccsds::error::ccsds_parse_error(
            "common",
            &format!(
                "time system '{}' is not supported for epoch conversion. Supported: UTC, TAI, GPS, TT, UT1, TDB, TCG, TCB, BDT, GST",
                time_system
            ),
        )
    })?;

    // Try day-of-year format: YYYY-DDDThh:mm:ss.sss
    if let Some(t_pos) = s.find('T') {
        let date_part = &s[..t_pos];
        let time_part = &s[t_pos + 1..];

        // Check if it's DOY format (YYYY-DDD where DDD is 3 digits)
        let parts: Vec<&str> = date_part.split('-').collect();
        if parts.len() == 2 && parts[1].len() == 3 {
            // Day-of-year format
            let year: u32 = parts[0].parse().map_err(|_| {
                crate::ccsds::error::ccsds_parse_error(
                    "datetime",
                    &format!("invalid year in '{}'", s),
                )
            })?;
            let doy: u32 = parts[1].parse().map_err(|_| {
                crate::ccsds::error::ccsds_parse_error(
                    "datetime",
                    &format!("invalid DOY in '{}'", s),
                )
            })?;

            // Parse time part
            let time_parts: Vec<&str> = time_part.split(':').collect();
            if time_parts.len() != 3 {
                return Err(crate::ccsds::error::ccsds_parse_error(
                    "datetime",
                    &format!("invalid time format in '{}'", s),
                ));
            }
            let hour: u8 = time_parts[0].parse().map_err(|_| {
                crate::ccsds::error::ccsds_parse_error(
                    "datetime",
                    &format!("invalid hour in '{}'", s),
                )
            })?;
            let minute: u8 = time_parts[1].parse().map_err(|_| {
                crate::ccsds::error::ccsds_parse_error(
                    "datetime",
                    &format!("invalid minute in '{}'", s),
                )
            })?;
            let sec_str = time_parts[2];
            let second: f64 = sec_str.parse().map_err(|_| {
                crate::ccsds::error::ccsds_parse_error(
                    "datetime",
                    &format!("invalid second in '{}'", s),
                )
            })?;

            // Convert DOY + time to fractional day of year
            let whole_second = second.floor();
            let frac_second = second - whole_second;
            let fractional_day = (doy as f64)
                + (hour as f64) / 24.0
                + (minute as f64) / 1440.0
                + whole_second / 86400.0
                + frac_second / 86400.0;

            return Epoch::from_day_of_year(year, fractional_day, ts);
        }
    }

    // Calendar format: YYYY-MM-DDThh:mm:ss.sss or YYYY-MM-DD hh:mm:ss.sss
    // Replace 'T' with space for the custom format parser
    let normalized = s.replace('T', " ");
    let parts: Vec<&str> = normalized.splitn(2, ' ').collect();
    if parts.len() != 2 {
        // Try date-only
        let date_parts: Vec<&str> = s.split('-').collect();
        if date_parts.len() == 3 {
            let year: u32 = date_parts[0].parse().map_err(|_| {
                crate::ccsds::error::ccsds_parse_error(
                    "datetime",
                    &format!("invalid year in '{}'", s),
                )
            })?;
            let month: u8 = date_parts[1].parse().map_err(|_| {
                crate::ccsds::error::ccsds_parse_error(
                    "datetime",
                    &format!("invalid month in '{}'", s),
                )
            })?;
            let day: u8 = date_parts[2].parse().map_err(|_| {
                crate::ccsds::error::ccsds_parse_error(
                    "datetime",
                    &format!("invalid day in '{}'", s),
                )
            })?;
            return Ok(Epoch::from_date(year, month, day, ts));
        }
        return Err(crate::ccsds::error::ccsds_parse_error(
            "datetime",
            &format!("unrecognized date format '{}'", s),
        ));
    }

    let date_part = parts[0];
    let time_part = parts[1];

    let date_parts: Vec<&str> = date_part.split('-').collect();
    if date_parts.len() != 3 {
        return Err(crate::ccsds::error::ccsds_parse_error(
            "datetime",
            &format!("invalid date format in '{}'", s),
        ));
    }

    let year: u32 = date_parts[0].parse().map_err(|_| {
        crate::ccsds::error::ccsds_parse_error("datetime", &format!("invalid year in '{}'", s))
    })?;
    let month: u8 = date_parts[1].parse().map_err(|_| {
        crate::ccsds::error::ccsds_parse_error("datetime", &format!("invalid month in '{}'", s))
    })?;
    let day: u8 = date_parts[2].parse().map_err(|_| {
        crate::ccsds::error::ccsds_parse_error("datetime", &format!("invalid day in '{}'", s))
    })?;

    let time_parts: Vec<&str> = time_part.split(':').collect();
    if time_parts.len() != 3 {
        return Err(crate::ccsds::error::ccsds_parse_error(
            "datetime",
            &format!("invalid time format in '{}'", s),
        ));
    }

    let hour: u8 = time_parts[0].parse().map_err(|_| {
        crate::ccsds::error::ccsds_parse_error("datetime", &format!("invalid hour in '{}'", s))
    })?;
    let minute: u8 = time_parts[1].parse().map_err(|_| {
        crate::ccsds::error::ccsds_parse_error("datetime", &format!("invalid minute in '{}'", s))
    })?;

    let sec_str = time_parts[2];
    let second: f64 = sec_str.parse().map_err(|_| {
        crate::ccsds::error::ccsds_parse_error("datetime", &format!("invalid second in '{}'", s))
    })?;

    let whole_second = second.floor();
    let frac_ns = (second - whole_second) * 1e9;

    Ok(Epoch::from_datetime(
        year,
        month,
        day,
        hour,
        minute,
        whole_second,
        frac_ns,
        ts,
    ))
}

/// Format an `Epoch` as a CCSDS datetime string in the epoch's own time system.
///
/// Output ranges from `YYYY-MM-DDThh:mm:ss.sss` to
/// `YYYY-MM-DDThh:mm:ss.sssssssss`; CCSDS 502.0-B-3 subsection 7.5.10 leaves
/// the number of fractional-second digits to the writer, and trailing zeros
/// are trimmed to milliseconds.
///
/// Ten fractional digits are emitted before trimming, which is finer than a
/// nanosecond and so absorbs the roughly 0.01 ns quantization that
/// nanoseconds-into-day arithmetic imposes on an `Epoch`. That keeps a written
/// value stable when it is read back and written again.
///
/// # Arguments
///
/// * `epoch` - The epoch to format, rendered in its own time system. Use
///   [`format_ccsds_datetime_in`] to render it in the system a message
///   declares.
///
/// # Returns
///
/// * `String` - The CCSDS time code for the epoch.
///
/// # Examples
///
/// ```
/// use brahe::ccsds::common::format_ccsds_datetime;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epoch = Epoch::from_datetime(1996, 11, 4, 17, 22, 31.0, 0.0, TimeSystem::UTC);
/// assert_eq!(format_ccsds_datetime(&epoch), "1996-11-04T17:22:31.000");
/// ```
pub fn format_ccsds_datetime(epoch: &Epoch) -> String {
    let (year, month, day, hour, minute, second, nanosecond) = epoch.to_datetime();
    let total_seconds = second + nanosecond / 1e9;
    if nanosecond == 0.0 {
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:06.3}",
            year, month, day, hour, minute, total_seconds
        )
    } else {
        // Use enough decimal places to represent the precision
        let formatted = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:013.10}",
            year, month, day, hour, minute, total_seconds
        );
        // Trim trailing zeros but keep at least one decimal place
        let trimmed = formatted.trim_end_matches('0');
        if trimmed.ends_with('.') {
            format!("{}0", trimmed)
        } else {
            trimmed.to_string()
        }
    }
}

/// Format an `Epoch` as a CCSDS datetime string in a given CCSDS time system.
///
/// CCSDS 502.0-B-3 subsection 7.5.11 ties every time or epoch keyword other
/// than `CREATION_DATE` to the message's `TIME_SYSTEM`, and CCSDS 508.0-B-1
/// subsection 6.2.3.4 fixes every CDM time tag to UTC. An `Epoch` carries its
/// own time system, which need not be the one the message declares, so writers
/// convert before formatting.
///
/// Five CCSDS time systems — `MET`, `MRT`, `SCLK`, `GMST`, and `TDR` — are
/// mission- or spacecraft-specific clocks with no fixed relationship to the
/// physical time systems `Epoch` represents. For those the epoch is formatted
/// as stored, unconverted.
///
/// # Arguments
///
/// * `epoch` - The epoch to format.
/// * `time_system` - The CCSDS time system the message declares.
///
/// # Returns
///
/// * `String` - The epoch rendered as `YYYY-MM-DDThh:mm:ss.sss`.
///
/// # Examples
///
/// ```
/// use brahe::ccsds::common::{CCSDSTimeSystem, format_ccsds_datetime_in};
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epoch = Epoch::from_datetime(2024, 3, 1, 12, 0, 0.0, 0.0, TimeSystem::TAI);
/// let written = format_ccsds_datetime_in(&epoch, &CCSDSTimeSystem::UTC);
/// assert!(written.starts_with("2024-03-01T11:59:"));
/// ```
pub fn format_ccsds_datetime_in(epoch: &Epoch, time_system: &CCSDSTimeSystem) -> String {
    match time_system.to_time_system() {
        Some(ts) => format_ccsds_datetime(&epoch.to_time_system(ts)),
        None => format_ccsds_datetime(epoch),
    }
}

/// Strip unit annotations from a CCSDS KVN value string.
///
/// CCSDS KVN values may contain optional unit annotations in square brackets
/// (e.g., "6655.9942 [km]"). This function removes the bracketed portion.
pub fn strip_units(value: &str) -> &str {
    if let Some(bracket_pos) = value.find('[') {
        value[..bracket_pos].trim()
    } else {
        value.trim()
    }
}

/// Parse a lower-triangular covariance matrix from 21 values.
///
/// CCSDS stores the lower-triangular elements of the 6x6 symmetric
/// covariance matrix row by row:
/// ```text
/// CX_X
/// CY_X CY_Y
/// CZ_X CZ_Y CZ_Z
/// CX_DOT_X CX_DOT_Y CX_DOT_Z CX_DOT_X_DOT
/// CY_DOT_X CY_DOT_Y CY_DOT_Z CY_DOT_X_DOT CY_DOT_Y_DOT
/// CZ_DOT_X CZ_DOT_Y CZ_DOT_Z CZ_DOT_X_DOT CZ_DOT_Y_DOT CZ_DOT_Z_DOT
/// ```
///
/// # Arguments
///
/// * `values` - 21 lower-triangular elements in row-major order
/// * `scale` - Scale factor to apply (e.g., 1e6 to convert km² to m²)
///
/// # Returns
///
/// * 6x6 symmetric matrix with scale applied
pub fn covariance_from_lower_triangular(values: &[f64; 21], scale: f64) -> SMatrix<f64, 6, 6> {
    let mut matrix = SMatrix::<f64, 6, 6>::zeros();
    let mut idx = 0;
    for row in 0..6 {
        for col in 0..=row {
            let val = values[idx] * scale;
            matrix[(row, col)] = val;
            matrix[(col, row)] = val;
            idx += 1;
        }
    }
    matrix
}

/// Indicates how many dimensions of a CDM extended covariance matrix are populated.
///
/// CDM covariance can extend beyond the standard 6×6 position/velocity block
/// to include drag (row 7), solar radiation pressure (row 8), and thrust (row 9)
/// uncertainty cross-correlations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CDMCovarianceDimension {
    /// 6×6 position/velocity only (21 lower-triangular elements)
    SixBySix,
    /// 7×7 with drag row/column (28 lower-triangular elements)
    SevenBySeven,
    /// 8×8 with drag + SRP rows/columns (36 lower-triangular elements)
    EightByEight,
    /// 9×9 with drag + SRP + thrust rows/columns (45 lower-triangular elements)
    NineByNine,
}

impl CDMCovarianceDimension {
    /// Return the matrix dimension (6, 7, 8, or 9).
    pub fn size(&self) -> usize {
        match self {
            CDMCovarianceDimension::SixBySix => 6,
            CDMCovarianceDimension::SevenBySeven => 7,
            CDMCovarianceDimension::EightByEight => 8,
            CDMCovarianceDimension::NineByNine => 9,
        }
    }

    /// Return the number of lower-triangular elements for this dimension.
    pub fn num_elements(&self) -> usize {
        let n = self.size();
        n * (n + 1) / 2
    }

    /// Determine the dimension from the number of lower-triangular elements.
    pub fn from_num_elements(n: usize) -> Result<Self, crate::utils::errors::BraheError> {
        match n {
            21 => Ok(CDMCovarianceDimension::SixBySix),
            28 => Ok(CDMCovarianceDimension::SevenBySeven),
            36 => Ok(CDMCovarianceDimension::EightByEight),
            45 => Ok(CDMCovarianceDimension::NineByNine),
            _ => Err(crate::ccsds::error::ccsds_parse_error(
                "CDM",
                &format!(
                    "invalid number of covariance elements: {} (expected 21, 28, 36, or 45)",
                    n
                ),
            )),
        }
    }
}

/// Parse lower-triangular values into a 9×9 symmetric covariance matrix.
///
/// CDM covariance values are already in SI units (m², m²/s, m²/s² for the
/// 6×6 core; m³/kg, m⁴/kg² for drag/SRP rows). No unit scaling is applied.
///
/// # Arguments
///
/// * `values` - Lower-triangular elements in row-major order (21, 28, 36, or 45 elements)
///
/// # Returns
///
/// * `(SMatrix<f64, 9, 9>, CDMCovarianceDimension)` - Symmetric matrix (zeroed beyond populated dimension) and dimension indicator
pub fn covariance9x9_from_lower_triangular(
    values: &[f64],
) -> Result<(SMatrix<f64, 9, 9>, CDMCovarianceDimension), crate::utils::errors::BraheError> {
    let dim = CDMCovarianceDimension::from_num_elements(values.len())?;
    let n = dim.size();
    let mut matrix = SMatrix::<f64, 9, 9>::zeros();
    let mut idx = 0;
    for row in 0..n {
        for col in 0..=row {
            let val = values[idx];
            matrix[(row, col)] = val;
            matrix[(col, row)] = val;
            idx += 1;
        }
    }
    Ok((matrix, dim))
}

/// Extract lower-triangular values from a 9×9 matrix up to the given dimension.
///
/// # Arguments
///
/// * `matrix` - 9×9 symmetric covariance matrix
/// * `dimension` - How many rows/columns to extract
///
/// # Returns
///
/// * `Vec<f64>` - Lower-triangular elements in row-major order
pub fn covariance9x9_to_lower_triangular(
    matrix: &SMatrix<f64, 9, 9>,
    dimension: CDMCovarianceDimension,
) -> Vec<f64> {
    let n = dimension.size();
    let mut values = Vec::with_capacity(dimension.num_elements());
    for row in 0..n {
        for col in 0..=row {
            values.push(matrix[(row, col)]);
        }
    }
    values
}

/// Significant decimal digits a CCSDS numeric value is written with.
///
/// An `f64` carries a little under 16 significant decimal digits, and the
/// unit conversions the ODM requires — metres to kilometres, m² to km² — are
/// not exactly invertible in binary floating point, so a value written at full
/// precision comes back one unit in the last place away from where it started
/// and the encoded text never settles. Fifteen digits absorb that difference
/// while staying inside what an `f64` can represent, and well inside the
/// sixteen digits CCSDS 502.0-B-3 subsection 7.5.7 permits.
const CCSDS_SIGNIFICANT_DIGITS: usize = 15;

/// Round a converted CCSDS value to the precision it is written with.
///
/// # Arguments
///
/// * `value` - The value after any unit conversion.
///
/// # Returns
///
/// * `f64` - The value rounded to [`CCSDS_SIGNIFICANT_DIGITS`], which is a
///   fixed point of the write-and-reread cycle.
///
/// # Examples
///
/// ```
/// use brahe::ccsds::common::round_ccsds_value;
///
/// // The metre/kilometre round trip is off by one unit in the last place.
/// let km = 3.3313494e-4;
/// assert_ne!(km * 1e6 * 1e-6, km);
/// assert_eq!(round_ccsds_value(km * 1e6 * 1e-6), km);
/// ```
pub fn round_ccsds_value(value: f64) -> f64 {
    if !value.is_finite() || value == 0.0 {
        return value;
    }
    format!("{:.*e}", CCSDS_SIGNIFICANT_DIGITS - 1, value)
        .parse()
        .unwrap_or(value)
}

/// Extract 21 lower-triangular values from a 6x6 symmetric matrix.
///
/// # Arguments
///
/// * `matrix` - 6x6 symmetric covariance matrix
/// * `scale` - Scale factor to apply (e.g., 1e-6 to convert m² to km²)
///
/// # Returns
///
/// * 21 lower-triangular elements in row-major order
pub fn covariance_to_lower_triangular(matrix: &SMatrix<f64, 6, 6>, scale: f64) -> [f64; 21] {
    let mut values = [0.0; 21];
    let mut idx = 0;
    for row in 0..6 {
        for col in 0..=row {
            values[idx] = matrix[(row, col)] * scale;
            idx += 1;
        }
    }
    values
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::ccsds::cdm::CDM;
    use crate::ccsds::oem::OEM;
    use crate::ccsds::omm::OMM;
    use crate::ccsds::opm::OPM;

    #[test]
    #[serial_test::parallel]
    fn test_ccsds_format_display() {
        assert_eq!(format!("{}", CCSDSFormat::KVN), "KVN");
        assert_eq!(format!("{}", CCSDSFormat::XML), "XML");
        assert_eq!(format!("{}", CCSDSFormat::JSON), "JSON");
    }

    #[test]
    #[serial_test::parallel]
    fn test_ccsds_time_system_parse() {
        assert_eq!(CCSDSTimeSystem::parse("UTC").unwrap(), CCSDSTimeSystem::UTC);
        assert_eq!(CCSDSTimeSystem::parse("TAI").unwrap(), CCSDSTimeSystem::TAI);
        assert_eq!(CCSDSTimeSystem::parse("GPS").unwrap(), CCSDSTimeSystem::GPS);
        assert_eq!(CCSDSTimeSystem::parse("TT").unwrap(), CCSDSTimeSystem::TT);
        assert_eq!(CCSDSTimeSystem::parse("UT1").unwrap(), CCSDSTimeSystem::UT1);
        assert_eq!(CCSDSTimeSystem::parse("TDB").unwrap(), CCSDSTimeSystem::TDB);
        assert_eq!(CCSDSTimeSystem::parse("TCB").unwrap(), CCSDSTimeSystem::TCB);
        assert_eq!(CCSDSTimeSystem::parse("TCG").unwrap(), CCSDSTimeSystem::TCG);
        assert_eq!(CCSDSTimeSystem::parse("BDT").unwrap(), CCSDSTimeSystem::BDT);
        assert_eq!(CCSDSTimeSystem::parse("GST").unwrap(), CCSDSTimeSystem::GST);
        assert_eq!(CCSDSTimeSystem::parse("MET").unwrap(), CCSDSTimeSystem::MET);
        assert_eq!(CCSDSTimeSystem::parse("MRT").unwrap(), CCSDSTimeSystem::MRT);
        assert!(CCSDSTimeSystem::parse("INVALID").is_err());
    }

    #[test]
    #[serial_test::parallel]
    fn test_ccsds_time_system_to_brahe() {
        assert!(CCSDSTimeSystem::UTC.to_time_system().is_some());
        assert!(CCSDSTimeSystem::TAI.to_time_system().is_some());
        assert!(CCSDSTimeSystem::GPS.to_time_system().is_some());
        assert!(CCSDSTimeSystem::TT.to_time_system().is_some());
        assert!(CCSDSTimeSystem::UT1.to_time_system().is_some());
        assert!(CCSDSTimeSystem::TDB.to_time_system().is_some());
        assert!(CCSDSTimeSystem::TCG.to_time_system().is_some());
        assert!(CCSDSTimeSystem::TCB.to_time_system().is_some());
        assert!(CCSDSTimeSystem::BDT.to_time_system().is_some());
        assert!(CCSDSTimeSystem::GST.to_time_system().is_some());
        assert!(CCSDSTimeSystem::MET.to_time_system().is_none());
    }

    #[test]
    #[serial_test::parallel]
    fn test_ccsds_ref_frame_parse() {
        assert_eq!(CCSDSRefFrame::parse("EME2000"), CCSDSRefFrame::EME2000);
        assert_eq!(CCSDSRefFrame::parse("GCRF"), CCSDSRefFrame::GCRF);
        assert_eq!(CCSDSRefFrame::parse("ITRF2000"), CCSDSRefFrame::ITRF2000);
        assert_eq!(CCSDSRefFrame::parse("ITRF-2000"), CCSDSRefFrame::ITRF2000);
        assert_eq!(CCSDSRefFrame::parse("ITRF1997"), CCSDSRefFrame::ITRF97);
        assert_eq!(CCSDSRefFrame::parse("TEME"), CCSDSRefFrame::TEME);
        assert_eq!(CCSDSRefFrame::parse("RTN"), CCSDSRefFrame::RTN);
        assert_eq!(
            CCSDSRefFrame::parse("CUSTOM_FRAME"),
            CCSDSRefFrame::Other("CUSTOM_FRAME".to_string())
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_ccsds_ref_frame_display() {
        assert_eq!(format!("{}", CCSDSRefFrame::EME2000), "EME2000");
        assert_eq!(format!("{}", CCSDSRefFrame::RTN), "RTN");
        assert_eq!(
            format!("{}", CCSDSRefFrame::Other("CUSTOM".to_string())),
            "CUSTOM"
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_strip_units() {
        assert_eq!(strip_units("6655.9942 [km]"), "6655.9942");
        assert_eq!(strip_units("3.11548208 [km/s]"), "3.11548208");
        assert_eq!(strip_units("0.020842611"), "0.020842611");
        assert_eq!(strip_units("  1913.000   [kg]  "), "1913.000");
    }

    #[test]
    #[serial_test::parallel]
    fn test_covariance_round_trip() {
        let values: [f64; 21] = [
            3.331e-04, 4.619e-04, 6.782e-04, -3.070e-04, -4.221e-04, 3.232e-04, -3.349e-07,
            -4.686e-07, 2.485e-07, 4.296e-10, -2.212e-07, -2.864e-07, 1.798e-07, 2.609e-10,
            1.768e-10, -3.041e-07, -4.989e-07, 3.540e-07, 1.869e-10, 1.009e-10, 6.224e-10,
        ];
        let matrix = covariance_from_lower_triangular(&values, 1.0);
        let recovered = covariance_to_lower_triangular(&matrix, 1.0);
        for i in 0..21 {
            assert!((values[i] - recovered[i]).abs() < 1e-15);
        }
        // Verify symmetry
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(matrix[(i, j)], matrix[(j, i)]);
            }
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_parse_ccsds_datetime_calendar() {
        let ts = CCSDSTimeSystem::UTC;
        let epoch = parse_ccsds_datetime("1996-12-18T12:00:00.331", &ts).unwrap();
        let (year, month, day, hour, minute, _second, _ns) = epoch.to_datetime();
        assert_eq!(year, 1996);
        assert_eq!(month, 12);
        assert_eq!(day, 18);
        assert_eq!(hour, 12);
        assert_eq!(minute, 0);
    }

    #[test]
    #[serial_test::parallel]
    fn test_parse_ccsds_datetime_doy() {
        let ts = CCSDSTimeSystem::UTC;
        // 1996-353 = 1996-12-18
        let epoch = parse_ccsds_datetime("1996-353T12:00:00.331", &ts).unwrap();
        let (year, month, day, hour, minute, _second, _ns) = epoch.to_datetime();
        assert_eq!(year, 1996);
        assert_eq!(month, 12);
        assert_eq!(day, 18);
        assert_eq!(hour, 12);
        assert_eq!(minute, 0);
    }

    #[test]
    #[serial_test::parallel]
    fn test_cdm_covariance_dimension() {
        assert_eq!(CDMCovarianceDimension::SixBySix.size(), 6);
        assert_eq!(CDMCovarianceDimension::SevenBySeven.size(), 7);
        assert_eq!(CDMCovarianceDimension::EightByEight.size(), 8);
        assert_eq!(CDMCovarianceDimension::NineByNine.size(), 9);
        assert_eq!(CDMCovarianceDimension::SixBySix.num_elements(), 21);
        assert_eq!(CDMCovarianceDimension::SevenBySeven.num_elements(), 28);
        assert_eq!(CDMCovarianceDimension::EightByEight.num_elements(), 36);
        assert_eq!(CDMCovarianceDimension::NineByNine.num_elements(), 45);
        assert_eq!(
            CDMCovarianceDimension::from_num_elements(21).unwrap(),
            CDMCovarianceDimension::SixBySix
        );
        assert_eq!(
            CDMCovarianceDimension::from_num_elements(45).unwrap(),
            CDMCovarianceDimension::NineByNine
        );
        assert!(CDMCovarianceDimension::from_num_elements(10).is_err());
    }

    #[test]
    #[serial_test::parallel]
    fn test_covariance9x9_round_trip_6x6() {
        // Standard 6x6 RTN covariance values from CDMExample1.txt Object1
        let values: Vec<f64> = vec![
            4.142e+01, -8.579e+00, 2.533e+03, -2.313e+01, 1.336e+01, 7.098e+01, 2.520e-03,
            -5.476e+00, 8.626e-04, 5.744e-03, -1.006e-02, 4.041e-03, -1.359e-03, -1.502e-05,
            1.049e-05, 1.053e-03, -3.412e-03, 1.213e-02, -3.004e-06, -1.091e-06, 5.529e-05,
        ];
        let (matrix, dim) = covariance9x9_from_lower_triangular(&values).unwrap();
        assert_eq!(dim, CDMCovarianceDimension::SixBySix);
        let recovered = covariance9x9_to_lower_triangular(&matrix, dim);
        for i in 0..21 {
            assert!((values[i] - recovered[i]).abs() < 1e-15);
        }
        // Verify symmetry in populated region
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(matrix[(i, j)], matrix[(j, i)]);
            }
        }
        // Verify unpopulated region is zero
        for i in 6..9 {
            for j in 0..9 {
                assert_eq!(matrix[(i, j)], 0.0);
                assert_eq!(matrix[(j, i)], 0.0);
            }
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_covariance9x9_round_trip_8x8() {
        // 8x8 = 36 elements (6x6 core + drag row + SRP row)
        let mut values = vec![0.0; 36];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i + 1) as f64 * 0.1;
        }
        let (matrix, dim) = covariance9x9_from_lower_triangular(&values).unwrap();
        assert_eq!(dim, CDMCovarianceDimension::EightByEight);
        let recovered = covariance9x9_to_lower_triangular(&matrix, dim);
        assert_eq!(values.len(), recovered.len());
        for i in 0..36 {
            assert!((values[i] - recovered[i]).abs() < 1e-15);
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_parse_ccsds_datetime_no_fractional() {
        let ts = CCSDSTimeSystem::UTC;
        let epoch = parse_ccsds_datetime("1998-11-06T09:23:57", &ts).unwrap();
        let (year, month, day, hour, minute, second, _ns) = epoch.to_datetime();
        assert_eq!(year, 1998);
        assert_eq!(month, 11);
        assert_eq!(day, 6);
        assert_eq!(hour, 9);
        assert_eq!(minute, 23);
        assert_eq!(second, 57.0);
    }

    #[test]
    #[serial_test::parallel]
    fn test_detect_format_kvn() {
        assert_eq!(detect_format("CCSDS_OEM_VERS = 3.0\n"), CCSDSFormat::KVN);
    }

    #[test]
    #[serial_test::parallel]
    fn test_detect_format_xml() {
        assert_eq!(
            detect_format("<?xml version=\"1.0\"?>\n<oem>"),
            CCSDSFormat::XML
        );
        assert_eq!(detect_format("<oem>"), CCSDSFormat::XML);
    }

    #[test]
    #[serial_test::parallel]
    fn test_detect_format_json() {
        assert_eq!(detect_format("{\"header\": {}}"), CCSDSFormat::JSON);
        assert_eq!(detect_format("[{\"header\": {}}]"), CCSDSFormat::JSON);
    }

    #[test]
    #[serial_test::parallel]
    fn test_detect_format_whitespace() {
        assert_eq!(
            detect_format("  \n  CCSDS_OEM_VERS = 3.0"),
            CCSDSFormat::KVN
        );
        assert_eq!(detect_format("  \n  <?xml"), CCSDSFormat::XML);
    }

    #[test]
    #[serial_test::parallel]
    fn test_parse_ccsds_datetime_unsupported_time_system() {
        let unsupported = [
            CCSDSTimeSystem::TDR,
            CCSDSTimeSystem::GMST,
            CCSDSTimeSystem::MET,
            CCSDSTimeSystem::MRT,
            CCSDSTimeSystem::SCLK,
        ];
        for ts in &unsupported {
            let result = parse_ccsds_datetime("2024-01-15T12:00:00.000", ts);
            assert!(result.is_err(), "Time system {} should return an error", ts);
            let err_msg = format!("{}", result.unwrap_err());
            assert!(
                err_msg.contains("not supported"),
                "Error for {} should mention 'not supported': {}",
                ts,
                err_msg
            );
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_parse_ccsds_datetime_supported_time_systems() {
        let supported = [
            CCSDSTimeSystem::UTC,
            CCSDSTimeSystem::TAI,
            CCSDSTimeSystem::GPS,
            CCSDSTimeSystem::TT,
        ];
        for ts in &supported {
            let result = parse_ccsds_datetime("2024-01-15T12:00:00.000", ts);
            assert!(
                result.is_ok(),
                "Time system {} should succeed: {}",
                ts,
                result.unwrap_err()
            );
        }
    }

    // --- 1. CCSDSTimeSystem Display for exotic variants ---

    #[test]
    #[serial_test::parallel]
    fn test_ccsds_time_system_display_exotic() {
        assert_eq!(format!("{}", CCSDSTimeSystem::TDB), "TDB");
        assert_eq!(format!("{}", CCSDSTimeSystem::TCB), "TCB");
        assert_eq!(format!("{}", CCSDSTimeSystem::TDR), "TDR");
        assert_eq!(format!("{}", CCSDSTimeSystem::TCG), "TCG");
        assert_eq!(format!("{}", CCSDSTimeSystem::GMST), "GMST");
        assert_eq!(format!("{}", CCSDSTimeSystem::MET), "MET");
        assert_eq!(format!("{}", CCSDSTimeSystem::MRT), "MRT");
        assert_eq!(format!("{}", CCSDSTimeSystem::SCLK), "SCLK");
    }

    // --- 2. CCSDSTimeSystem::parse() for exotic variants ---

    #[test]
    #[serial_test::parallel]
    fn test_ccsds_time_system_parse_exotic() {
        assert_eq!(CCSDSTimeSystem::parse("TCB").unwrap(), CCSDSTimeSystem::TCB);
        assert_eq!(CCSDSTimeSystem::parse("TDR").unwrap(), CCSDSTimeSystem::TDR);
        assert_eq!(CCSDSTimeSystem::parse("TCG").unwrap(), CCSDSTimeSystem::TCG);
        assert_eq!(
            CCSDSTimeSystem::parse("GMST").unwrap(),
            CCSDSTimeSystem::GMST
        );
        assert_eq!(
            CCSDSTimeSystem::parse("SCLK").unwrap(),
            CCSDSTimeSystem::SCLK
        );
    }

    // --- 3. CCSDSRefFrame Display for untested variants ---

    #[test]
    #[serial_test::parallel]
    fn test_ccsds_ref_frame_display_all_variants() {
        assert_eq!(format!("{}", CCSDSRefFrame::GCRF), "GCRF");
        assert_eq!(format!("{}", CCSDSRefFrame::ITRF2000), "ITRF2000");
        assert_eq!(format!("{}", CCSDSRefFrame::ITRF93), "ITRF93");
        assert_eq!(format!("{}", CCSDSRefFrame::ITRF97), "ITRF97");
        assert_eq!(format!("{}", CCSDSRefFrame::ITRF2005), "ITRF2005");
        assert_eq!(format!("{}", CCSDSRefFrame::ITRF2008), "ITRF2008");
        assert_eq!(format!("{}", CCSDSRefFrame::ITRF2014), "ITRF2014");
        assert_eq!(format!("{}", CCSDSRefFrame::TEME), "TEME");
        assert_eq!(format!("{}", CCSDSRefFrame::TDR), "TDR");
        assert_eq!(format!("{}", CCSDSRefFrame::TNW), "TNW");
        assert_eq!(format!("{}", CCSDSRefFrame::RSW), "RSW");
        assert_eq!(format!("{}", CCSDSRefFrame::TOD), "TOD");
        assert_eq!(format!("{}", CCSDSRefFrame::J2000), "J2000");
    }

    // --- 4. CCSDSRefFrame::parse() alternative formats ---

    #[test]
    #[serial_test::parallel]
    fn test_ccsds_ref_frame_parse_alternative_formats() {
        assert_eq!(CCSDSRefFrame::parse("ITRF-2005"), CCSDSRefFrame::ITRF2005);
        assert_eq!(CCSDSRefFrame::parse("ITRF-2008"), CCSDSRefFrame::ITRF2008);
        assert_eq!(CCSDSRefFrame::parse("ITRF-2014"), CCSDSRefFrame::ITRF2014);
        assert_eq!(CCSDSRefFrame::parse("ITRF-97"), CCSDSRefFrame::ITRF97);
        assert_eq!(CCSDSRefFrame::parse("ITRF-93"), CCSDSRefFrame::ITRF93);
        assert_eq!(CCSDSRefFrame::parse("TDR"), CCSDSRefFrame::TDR);
        assert_eq!(CCSDSRefFrame::parse("TOD"), CCSDSRefFrame::TOD);
        assert_eq!(CCSDSRefFrame::parse("J2000"), CCSDSRefFrame::J2000);
        assert_eq!(CCSDSRefFrame::parse("TNW"), CCSDSRefFrame::TNW);
        assert_eq!(CCSDSRefFrame::parse("RSW"), CCSDSRefFrame::RSW);
    }

    // --- 5. format_ccsds_datetime edge cases ---

    #[test]
    #[serial_test::parallel]
    fn test_format_ccsds_datetime_zero_nanoseconds() {
        // Zero nanoseconds should produce the simpler 3-decimal format (via the nanosecond==0.0 branch)
        // Use from_date which guarantees zero nanosecond component
        let epoch = Epoch::from_date(2024, 1, 15, crate::time::TimeSystem::UTC);
        let formatted = format_ccsds_datetime(&epoch);
        assert_eq!(formatted, "2024-01-15T00:00:00.000");
    }

    #[test]
    #[serial_test::parallel]
    fn test_format_ccsds_datetime_trailing_zeros_trimmed() {
        // Non-zero nanoseconds that would leave trailing zeros after trimming
        let epoch = Epoch::from_datetime(
            2024,
            1,
            15,
            12,
            0,
            0.0,
            500_000_000.0,
            crate::time::TimeSystem::UTC,
        );
        let formatted = format_ccsds_datetime(&epoch);
        assert!(formatted.contains("T12:00:00.5"));
        assert!(!formatted.ends_with('0') || formatted.ends_with(".0"));
    }

    #[test]
    #[serial_test::parallel]
    fn test_format_ccsds_datetime_integer_second_with_nanoseconds() {
        // Nanoseconds that result in a trailing-dot scenario after trim (whole number of seconds from ns)
        // e.g. 1_000_000_000 ns = 1.0 extra second. This tests the ".0" branch.
        let epoch =
            Epoch::from_datetime(2024, 6, 1, 0, 0, 10.0, 100.0, crate::time::TimeSystem::UTC);
        let formatted = format_ccsds_datetime(&epoch);
        // Should contain decimal portion and not end with bare '.'
        assert!(!formatted.ends_with('.'));
    }

    // --- 6. parse_ccsds_datetime edge cases ---

    #[test]
    #[serial_test::parallel]
    fn test_parse_ccsds_datetime_date_only() {
        let ts = CCSDSTimeSystem::UTC;
        let epoch = parse_ccsds_datetime("2024-01-15", &ts).unwrap();
        let (year, month, day, hour, minute, second, _ns) = epoch.to_datetime();
        assert_eq!(year, 2024);
        assert_eq!(month, 1);
        assert_eq!(day, 15);
        assert_eq!(hour, 0);
        assert_eq!(minute, 0);
        assert_eq!(second, 0.0);
    }

    #[test]
    #[serial_test::parallel]
    fn test_parse_ccsds_datetime_doy_high_precision() {
        let ts = CCSDSTimeSystem::UTC;
        // DOY format with high-precision fractional seconds
        let epoch = parse_ccsds_datetime("2024-032T06:30:15.123456789", &ts).unwrap();
        let (year, month, day, hour, minute, _second, _ns) = epoch.to_datetime();
        assert_eq!(year, 2024);
        assert_eq!(month, 2); // Day 32 of 2024 = Feb 1
        assert_eq!(day, 1);
        assert_eq!(hour, 6);
        assert_eq!(minute, 30);
    }

    #[test]
    #[serial_test::parallel]
    fn test_parse_ccsds_datetime_ut1() {
        // UT1 is supported but requires EOP initialization
        crate::utils::testing::setup_global_test_eop();
        let ts = CCSDSTimeSystem::UT1;
        let epoch = parse_ccsds_datetime("2020-06-15T00:00:00.000", &ts);
        assert!(epoch.is_ok());
    }

    // --- 7. Covariance scale factor tests ---

    #[test]
    #[serial_test::parallel]
    fn test_covariance_from_lower_triangular_with_scale() {
        let values: [f64; 21] = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0,
        ];
        // Scale by 1e6 (km^2 -> m^2)
        let matrix = covariance_from_lower_triangular(&values, 1e6);
        assert_eq!(matrix[(0, 0)], 1.0e6);
        assert_eq!(matrix[(1, 0)], 2.0e6);
        assert_eq!(matrix[(0, 1)], 2.0e6); // symmetry
        assert_eq!(matrix[(5, 5)], 21.0e6);
    }

    #[test]
    #[serial_test::parallel]
    fn test_covariance_to_lower_triangular_with_scale() {
        let values: [f64; 21] = [
            1.0e6, 2.0e6, 3.0e6, 4.0e6, 5.0e6, 6.0e6, 7.0e6, 8.0e6, 9.0e6, 10.0e6, 11.0e6, 12.0e6,
            13.0e6, 14.0e6, 15.0e6, 16.0e6, 17.0e6, 18.0e6, 19.0e6, 20.0e6, 21.0e6,
        ];
        let matrix = covariance_from_lower_triangular(&values, 1.0);
        // Scale by 1e-6 (m^2 -> km^2)
        let recovered = covariance_to_lower_triangular(&matrix, 1e-6);
        for (i, val) in recovered.iter().enumerate() {
            assert!((val - (i + 1) as f64).abs() < 1e-9);
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_covariance_round_trip_with_scale() {
        let values: [f64; 21] = [
            3.331e-04, 4.619e-04, 6.782e-04, -3.070e-04, -4.221e-04, 3.232e-04, -3.349e-07,
            -4.686e-07, 2.485e-07, 4.296e-10, -2.212e-07, -2.864e-07, 1.798e-07, 2.609e-10,
            1.768e-10, -3.041e-07, -4.989e-07, 3.540e-07, 1.869e-10, 1.009e-10, 6.224e-10,
        ];
        // Convert km^2 -> m^2 and back
        let matrix = covariance_from_lower_triangular(&values, 1e6);
        let recovered = covariance_to_lower_triangular(&matrix, 1e-6);
        for i in 0..21 {
            assert!(
                (values[i] - recovered[i]).abs() < 1e-15,
                "Mismatch at index {}: {} vs {}",
                i,
                values[i],
                recovered[i]
            );
        }
    }

    // --- 8. covariance9x9 7x7 and 9x9 round-trips ---

    #[test]
    #[serial_test::parallel]
    fn test_covariance9x9_round_trip_7x7() {
        let mut values = vec![0.0; 28];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i + 1) as f64 * 0.01;
        }
        let (matrix, dim) = covariance9x9_from_lower_triangular(&values).unwrap();
        assert_eq!(dim, CDMCovarianceDimension::SevenBySeven);
        let recovered = covariance9x9_to_lower_triangular(&matrix, dim);
        assert_eq!(values.len(), recovered.len());
        for i in 0..28 {
            assert!(
                (values[i] - recovered[i]).abs() < 1e-15,
                "Mismatch at index {}: {} vs {}",
                i,
                values[i],
                recovered[i]
            );
        }
        // Verify symmetry in populated 7x7 region
        for i in 0..7 {
            for j in 0..7 {
                assert_eq!(matrix[(i, j)], matrix[(j, i)]);
            }
        }
        // Verify unpopulated rows 7-8 are zero
        for i in 7..9 {
            for j in 0..9 {
                assert_eq!(matrix[(i, j)], 0.0);
            }
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_covariance9x9_round_trip_9x9() {
        let mut values = vec![0.0; 45];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i + 1) as f64 * 0.001;
        }
        let (matrix, dim) = covariance9x9_from_lower_triangular(&values).unwrap();
        assert_eq!(dim, CDMCovarianceDimension::NineByNine);
        let recovered = covariance9x9_to_lower_triangular(&matrix, dim);
        assert_eq!(values.len(), recovered.len());
        for i in 0..45 {
            assert!(
                (values[i] - recovered[i]).abs() < 1e-15,
                "Mismatch at index {}: {} vs {}",
                i,
                values[i],
                recovered[i]
            );
        }
        // Verify full 9x9 symmetry
        for i in 0..9 {
            for j in 0..9 {
                assert_eq!(matrix[(i, j)], matrix[(j, i)]);
            }
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_covariance9x9_invalid_element_count() {
        let values = vec![0.0; 10]; // Invalid count
        let result = covariance9x9_from_lower_triangular(&values);
        assert!(result.is_err());
    }

    // --- Additional edge cases for CCSDSTimeSystem ---

    #[test]
    #[serial_test::parallel]
    fn test_ccsds_time_system_display_all_standard() {
        // Cover the standard variants that Display test above already covers
        // but ensure all are tested including UTC/TAI/GPS/TT/UT1
        assert_eq!(format!("{}", CCSDSTimeSystem::UTC), "UTC");
        assert_eq!(format!("{}", CCSDSTimeSystem::TAI), "TAI");
        assert_eq!(format!("{}", CCSDSTimeSystem::GPS), "GPS");
        assert_eq!(format!("{}", CCSDSTimeSystem::TT), "TT");
        assert_eq!(format!("{}", CCSDSTimeSystem::UT1), "UT1");
    }

    #[test]
    #[serial_test::parallel]
    fn test_ccsds_time_system_to_brahe_exotic_none() {
        // Verify exotic/unsupported variants return None
        assert!(CCSDSTimeSystem::TDR.to_time_system().is_none());
        assert!(CCSDSTimeSystem::GMST.to_time_system().is_none());
        assert!(CCSDSTimeSystem::MRT.to_time_system().is_none());
        assert!(CCSDSTimeSystem::SCLK.to_time_system().is_none());
    }

    /// Retag an epoch into another time system without moving the instant, the
    /// way a user-constructed message can end up holding epochs in a system
    /// other than the one its metadata declares.
    fn retag(e: &Epoch) -> Epoch {
        e.to_time_system(crate::time::TimeSystem::TAI)
    }

    #[test]
    #[serial_test::parallel]
    fn test_format_ccsds_datetime_in_converts_to_declared_system() {
        let utc = Epoch::from_datetime(2024, 3, 1, 12, 0, 0.0, 0.0, crate::time::TimeSystem::UTC);

        // Same instant, formatted in each declared system.
        assert_eq!(
            format_ccsds_datetime_in(&utc, &CCSDSTimeSystem::UTC),
            "2024-03-01T12:00:00.000"
        );
        assert_eq!(
            format_ccsds_datetime_in(&utc, &CCSDSTimeSystem::TAI),
            "2024-03-01T12:00:37.000"
        );

        // The epoch's own time system does not influence the result.
        assert_eq!(
            format_ccsds_datetime_in(&retag(&utc), &CCSDSTimeSystem::UTC),
            format_ccsds_datetime_in(&utc, &CCSDSTimeSystem::UTC)
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_format_ccsds_datetime_in_leaves_mission_clocks_unconverted() {
        let tai = Epoch::from_datetime(2024, 3, 1, 12, 0, 0.0, 0.0, crate::time::TimeSystem::TAI);

        // MET, MRT, SCLK, GMST, and TDR have no `TimeSystem` counterpart, so
        // the epoch is written exactly as stored.
        for ts in [
            CCSDSTimeSystem::MET,
            CCSDSTimeSystem::MRT,
            CCSDSTimeSystem::SCLK,
            CCSDSTimeSystem::GMST,
            CCSDSTimeSystem::TDR,
        ] {
            assert_eq!(
                format_ccsds_datetime_in(&tai, &ts),
                format_ccsds_datetime(&tai)
            );
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_writers_use_declared_time_system_not_epoch_time_system() {
        let formats = [CCSDSFormat::KVN, CCSDSFormat::XML, CCSDSFormat::JSON];

        // OEM
        let src = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = OEM::from_str(&src).unwrap();
        let mut retagged = oem.clone();
        retagged.header.creation_date = retag(&retagged.header.creation_date);
        for seg in &mut retagged.segments {
            seg.metadata.start_time = retag(&seg.metadata.start_time);
            seg.metadata.stop_time = retag(&seg.metadata.stop_time);
            seg.metadata.useable_start_time = seg.metadata.useable_start_time.as_ref().map(retag);
            seg.metadata.useable_stop_time = seg.metadata.useable_stop_time.as_ref().map(retag);
            for sv in &mut seg.states {
                sv.epoch = retag(&sv.epoch);
            }
            for cov in &mut seg.covariances {
                cov.epoch = cov.epoch.as_ref().map(retag);
            }
        }
        for format in formats {
            assert_eq!(
                retagged.to_string(format).unwrap(),
                oem.to_string(format).unwrap(),
                "OEM {:?} output depends on the epoch's own time system",
                format
            );
        }

        // OMM
        let src = std::fs::read_to_string("test_assets/ccsds/omm/OMMExample2.txt").unwrap();
        let mut omm = OMM::from_str(&src).unwrap();
        // The fixture leaves these unset, so populate them: every epoch the
        // writers can emit has to follow the declared system, not just the
        // ones a stock message happens to carry.
        omm.metadata.ref_frame_epoch = Some(omm.mean_elements.epoch);
        omm.covariance = Some(CCSDSCovariance {
            epoch: Some(omm.mean_elements.epoch),
            cov_ref_frame: None,
            matrix: SMatrix::<f64, 6, 6>::identity(),
            comments: Vec::new(),
        });
        let omm = omm;
        let mut retagged = omm.clone();
        retagged.header.creation_date = retag(&retagged.header.creation_date);
        retagged.mean_elements.epoch = retag(&retagged.mean_elements.epoch);
        retagged.metadata.ref_frame_epoch = retagged.metadata.ref_frame_epoch.as_ref().map(retag);
        if let Some(ref mut cov) = retagged.covariance {
            cov.epoch = cov.epoch.as_ref().map(retag);
        }
        for format in formats {
            assert_eq!(
                retagged.to_string(format).unwrap(),
                omm.to_string(format).unwrap(),
                "OMM {:?} output depends on the epoch's own time system",
                format
            );
        }

        // OPM
        let src = std::fs::read_to_string("test_assets/ccsds/opm/OPMExample2.txt").unwrap();
        let mut opm = OPM::from_str(&src).unwrap();
        assert!(!opm.maneuvers.is_empty(), "fixture has maneuvers");
        opm.metadata.ref_frame_epoch = Some(opm.state_vector.epoch);
        opm.covariance = Some(CCSDSCovariance {
            epoch: Some(opm.state_vector.epoch),
            cov_ref_frame: None,
            matrix: SMatrix::<f64, 6, 6>::identity(),
            comments: Vec::new(),
        });
        let opm = opm;
        let mut retagged = opm.clone();
        retagged.header.creation_date = retag(&retagged.header.creation_date);
        retagged.state_vector.epoch = retag(&retagged.state_vector.epoch);
        retagged.metadata.ref_frame_epoch = retagged.metadata.ref_frame_epoch.as_ref().map(retag);
        if let Some(ref mut cov) = retagged.covariance {
            cov.epoch = cov.epoch.as_ref().map(retag);
        }
        for man in &mut retagged.maneuvers {
            man.epoch_ignition = retag(&man.epoch_ignition);
        }
        for format in formats {
            assert_eq!(
                retagged.to_string(format).unwrap(),
                opm.to_string(format).unwrap(),
                "OPM {:?} output depends on the epoch's own time system",
                format
            );
        }

        // CDM
        let src = std::fs::read_to_string("test_assets/ccsds/cdm/CDMExample1.txt").unwrap();
        let mut cdm = CDM::from_str(&src).unwrap();
        cdm.relative_metadata.previous_message_epoch = Some(cdm.relative_metadata.tca);
        cdm.relative_metadata.next_message_epoch = Some(cdm.relative_metadata.tca);
        let cdm = cdm;
        let mut retagged = cdm.clone();
        retagged.header.creation_date = retag(&retagged.header.creation_date);
        retagged.relative_metadata.tca = retag(&retagged.relative_metadata.tca);
        retagged.relative_metadata.previous_message_epoch = retagged
            .relative_metadata
            .previous_message_epoch
            .as_ref()
            .map(retag);
        retagged.relative_metadata.next_message_epoch = retagged
            .relative_metadata
            .next_message_epoch
            .as_ref()
            .map(retag);
        for format in formats {
            assert_eq!(
                retagged.to_string(format).unwrap(),
                cdm.to_string(format).unwrap(),
                "CDM {:?} output depends on the epoch's own time system",
                format
            );
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_ref_frame_epoch_survives_a_non_utc_time_system() {
        // Under a non-UTC TIME_SYSTEM the message has to come back declaring
        // that same system, with every epoch on it. REF_FRAME_EPOCH precedes
        // TIME_SYSTEM in the fixed keyword order and JSON object keys put
        // START_TIME ahead of it, so a reader that resolves either eagerly
        // falls back to UTC and shifts the instant by the scale offset.
        let source = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let mut oem = OEM::from_str(&source).unwrap();
        let segment = &mut oem.segments[0];
        segment.metadata.ref_frame_epoch = Some(segment.metadata.start_time);
        segment.metadata.time_system = CCSDSTimeSystem::TAI;
        let expected = segment.metadata.ref_frame_epoch.unwrap();

        let start = segment.metadata.start_time;

        for format in [CCSDSFormat::KVN, CCSDSFormat::XML, CCSDSFormat::JSON] {
            let reparsed = OEM::from_str(&oem.to_string(format).unwrap()).unwrap();

            assert_eq!(
                reparsed.segments[0].metadata.time_system,
                CCSDSTimeSystem::TAI,
                "{:?} did not round-trip the declared time system",
                format
            );

            let got = reparsed.segments[0].metadata.ref_frame_epoch.unwrap();
            assert!(
                (got - expected).abs() < 1e-9,
                "{:?} shifted REF_FRAME_EPOCH by {} s",
                format,
                got - expected
            );

            let got_start = reparsed.segments[0].metadata.start_time;
            assert!(
                (got_start - start).abs() < 1e-9,
                "{:?} shifted START_TIME by {} s",
                format,
                got_start - start
            );
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_write_oem_epochs_follow_the_declared_time_system() {
        let src = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let mut oem = OEM::from_str(&src).unwrap();
        assert_eq!(oem.segments[0].metadata.time_system, CCSDSTimeSystem::UTC);
        assert!(
            oem.to_string(CCSDSFormat::KVN)
                .unwrap()
                .contains("START_TIME = 1996-12-18T12:00:00.331")
        );

        // Redeclaring TIME_SYSTEM moves the written epochs onto that scale.
        oem.segments[0].metadata.time_system = CCSDSTimeSystem::TAI;
        assert!(
            oem.to_string(CCSDSFormat::KVN)
                .unwrap()
                .contains("START_TIME = 1996-12-18T12:00:30.331")
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_format_ccsds_datetime_writes_a_whole_second_as_whole() {
        // An epoch built on a whole second is written with no fractional
        // nanoseconds. This used to emit `.000000001`, because the formatter
        // wrote a sub-nanosecond remainder that `Epoch::to_datetime` reported
        // but the epoch did not hold; brahe/pull/488 fixed that at the source.
        let epoch =
            Epoch::from_datetime(1996, 11, 4, 17, 22, 31.0, 0.0, crate::time::TimeSystem::UTC);
        let (_, _, _, _, _, _, nanosecond) = epoch.to_datetime();
        assert_eq!(nanosecond, 0.0);

        assert_eq!(format_ccsds_datetime(&epoch), "1996-11-04T17:22:31.000");
    }

    #[test]
    #[serial_test::parallel]
    fn test_ccsds_datetime_round_trip_is_a_fixed_point() {
        // Writing and re-reading must converge; before whole nanoseconds were
        // isolated, each generation added one nanosecond without bound.
        for start in [
            "1996-11-04T17:22:31",
            "1996-12-18T12:00:00.331",
            "2024-01-15T12:00:00.5",
            "2024-06-01T00:00:10.000000100",
        ] {
            let first =
                format_ccsds_datetime(&parse_ccsds_datetime(start, &CCSDSTimeSystem::UTC).unwrap());
            let mut current = first.clone();
            for _ in 0..4 {
                current = format_ccsds_datetime(
                    &parse_ccsds_datetime(&current, &CCSDSTimeSystem::UTC).unwrap(),
                );
                assert_eq!(
                    current, first,
                    "'{}' drifts across write/read cycles",
                    start
                );
            }
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_kvn_message_round_trip_is_a_fixed_point() {
        use crate::ccsds::cdm::CDM;
        use crate::ccsds::oem::OEM;
        use crate::ccsds::omm::OMM;
        use crate::ccsds::opm::OPM;

        type KvnRoundTrip = fn(&str) -> String;
        let cases: [(&str, KvnRoundTrip); 4] = [
            ("test_assets/ccsds/oem/OEMExample1.txt", |s| {
                OEM::from_str(s)
                    .unwrap()
                    .to_string(CCSDSFormat::KVN)
                    .unwrap()
            }),
            ("test_assets/ccsds/omm/OMMExample2.txt", |s| {
                OMM::from_str(s)
                    .unwrap()
                    .to_string(CCSDSFormat::KVN)
                    .unwrap()
            }),
            ("test_assets/ccsds/opm/OPMExample1.txt", |s| {
                OPM::from_str(s)
                    .unwrap()
                    .to_string(CCSDSFormat::KVN)
                    .unwrap()
            }),
            ("test_assets/ccsds/cdm/CDMExample1.txt", |s| {
                CDM::from_str(s)
                    .unwrap()
                    .to_string(CCSDSFormat::KVN)
                    .unwrap()
            }),
        ];

        for (path, write) in cases {
            let source = std::fs::read_to_string(path).unwrap();
            let written = write(&source);
            let mut current = written.clone();
            for _ in 0..3 {
                current = write(&current);
                assert_eq!(current, written, "{} drifts across write/read cycles", path);
            }
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_oem_epochs_survive_every_output_format() {
        use crate::ccsds::oem::OEM;

        let source = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = OEM::from_str(&source).unwrap();
        let epochs = |o: &OEM| -> Vec<String> {
            o.segments
                .iter()
                .flat_map(|seg| {
                    std::iter::once(format_ccsds_datetime(&seg.metadata.start_time))
                        .chain(std::iter::once(format_ccsds_datetime(
                            &seg.metadata.stop_time,
                        )))
                        .chain(seg.states.iter().map(|sv| format_ccsds_datetime(&sv.epoch)))
                })
                .collect()
        };

        for format in [CCSDSFormat::KVN, CCSDSFormat::XML, CCSDSFormat::JSON] {
            let reparsed = OEM::from_str(&oem.to_string(format).unwrap()).unwrap();
            assert_eq!(
                epochs(&reparsed),
                epochs(&oem),
                "OEM epochs shifted across a {:?} round trip",
                format
            );
        }
    }

    #[test]
    #[serial_test::parallel]
    fn test_round_ccsds_value_is_a_conversion_fixed_point() {
        // The metre/kilometre round trip is off by one unit in the last place,
        // so a value written at full precision never settles.
        for km in [3.3313494e-4f64, 4.6189273e-4, -2.2118325e-7, 2.6088992e-10] {
            let drifted = km * 1e6 * 1e-6;
            assert_eq!(round_ccsds_value(drifted), km);
            // Rounding is idempotent under further cycles.
            assert_eq!(
                round_ccsds_value(round_ccsds_value(drifted) * 1e6 * 1e-6),
                km
            );
        }

        // Degenerate values pass through untouched.
        assert_eq!(round_ccsds_value(0.0), 0.0);
        assert!(round_ccsds_value(f64::NAN).is_nan());
        assert_eq!(round_ccsds_value(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    #[serial_test::parallel]
    fn test_covariance_extractor_does_not_round() {
        // Rounding belongs to the writers; the public extractor keeps its
        // documented multiply-and-extract semantics so a caller asking for
        // scale 1.0 gets the matrix back exactly.
        let exact = 1.2345678901234567_f64;
        let mut matrix = SMatrix::<f64, 6, 6>::zeros();
        matrix[(0, 0)] = exact;

        let values = covariance_to_lower_triangular(&matrix, 1.0);
        assert_eq!(values[0], exact);
        assert_ne!(round_ccsds_value(exact), exact);
    }

    #[test]
    #[serial_test::parallel]
    fn test_message_writes_are_stable_in_every_encoding() {
        use crate::ccsds::cdm::CDM;
        use crate::ccsds::oem::OEM;
        use crate::ccsds::omm::OMM;
        use crate::ccsds::opm::OPM;

        let formats = [CCSDSFormat::KVN, CCSDSFormat::XML, CCSDSFormat::JSON];

        macro_rules! assert_stable {
            ($ty:ty, $path:expr) => {
                let source = std::fs::read_to_string($path).unwrap();
                let message = <$ty>::from_str(&source).unwrap();
                for format in formats {
                    let written = message.to_string(format).unwrap();
                    let rewritten = <$ty>::from_str(&written)
                        .unwrap()
                        .to_string(format)
                        .unwrap();
                    assert_eq!(
                        rewritten, written,
                        "{} {:?} output is not stable across a reparse",
                        $path, format
                    );
                }
            };
        }

        assert_stable!(OEM, "test_assets/ccsds/oem/OEMExample1.txt");
        assert_stable!(OMM, "test_assets/ccsds/omm/OMMExample2.txt");
        assert_stable!(OPM, "test_assets/ccsds/opm/OPMExample3.txt");
        assert_stable!(CDM, "test_assets/ccsds/cdm/CDMExample2.txt");
    }
}
