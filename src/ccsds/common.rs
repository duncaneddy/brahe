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
    let ts = time_system
        .to_time_system()
        .unwrap_or(crate::time::TimeSystem::UTC);

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

            return Ok(Epoch::from_day_of_year(year, fractional_day, ts));
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

/// Format an Epoch as a CCSDS datetime string.
///
/// Output format: `YYYY-MM-DDThh:mm:ss.sssssssss`
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

    #[test]
    fn test_ccsds_format_display() {
        assert_eq!(format!("{}", CCSDSFormat::KVN), "KVN");
        assert_eq!(format!("{}", CCSDSFormat::XML), "XML");
        assert_eq!(format!("{}", CCSDSFormat::JSON), "JSON");
    }

    #[test]
    fn test_ccsds_time_system_parse() {
        assert_eq!(CCSDSTimeSystem::parse("UTC").unwrap(), CCSDSTimeSystem::UTC);
        assert_eq!(CCSDSTimeSystem::parse("TAI").unwrap(), CCSDSTimeSystem::TAI);
        assert_eq!(CCSDSTimeSystem::parse("GPS").unwrap(), CCSDSTimeSystem::GPS);
        assert_eq!(CCSDSTimeSystem::parse("TT").unwrap(), CCSDSTimeSystem::TT);
        assert_eq!(CCSDSTimeSystem::parse("UT1").unwrap(), CCSDSTimeSystem::UT1);
        assert_eq!(CCSDSTimeSystem::parse("TDB").unwrap(), CCSDSTimeSystem::TDB);
        assert_eq!(CCSDSTimeSystem::parse("MET").unwrap(), CCSDSTimeSystem::MET);
        assert_eq!(CCSDSTimeSystem::parse("MRT").unwrap(), CCSDSTimeSystem::MRT);
        assert!(CCSDSTimeSystem::parse("INVALID").is_err());
    }

    #[test]
    fn test_ccsds_time_system_to_brahe() {
        assert!(CCSDSTimeSystem::UTC.to_time_system().is_some());
        assert!(CCSDSTimeSystem::TAI.to_time_system().is_some());
        assert!(CCSDSTimeSystem::TDB.to_time_system().is_none());
        assert!(CCSDSTimeSystem::MET.to_time_system().is_none());
    }

    #[test]
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
    fn test_ccsds_ref_frame_display() {
        assert_eq!(format!("{}", CCSDSRefFrame::EME2000), "EME2000");
        assert_eq!(format!("{}", CCSDSRefFrame::RTN), "RTN");
        assert_eq!(
            format!("{}", CCSDSRefFrame::Other("CUSTOM".to_string())),
            "CUSTOM"
        );
    }

    #[test]
    fn test_strip_units() {
        assert_eq!(strip_units("6655.9942 [km]"), "6655.9942");
        assert_eq!(strip_units("3.11548208 [km/s]"), "3.11548208");
        assert_eq!(strip_units("0.020842611"), "0.020842611");
        assert_eq!(strip_units("  1913.000   [kg]  "), "1913.000");
    }

    #[test]
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
    fn test_detect_format_kvn() {
        assert_eq!(detect_format("CCSDS_OEM_VERS = 3.0\n"), CCSDSFormat::KVN);
    }

    #[test]
    fn test_detect_format_xml() {
        assert_eq!(
            detect_format("<?xml version=\"1.0\"?>\n<oem>"),
            CCSDSFormat::XML
        );
        assert_eq!(detect_format("<oem>"), CCSDSFormat::XML);
    }

    #[test]
    fn test_detect_format_json() {
        assert_eq!(detect_format("{\"header\": {}}"), CCSDSFormat::JSON);
        assert_eq!(detect_format("[{\"header\": {}}]"), CCSDSFormat::JSON);
    }

    #[test]
    fn test_detect_format_whitespace() {
        assert_eq!(
            detect_format("  \n  CCSDS_OEM_VERS = 3.0"),
            CCSDSFormat::KVN
        );
        assert_eq!(detect_format("  \n  <?xml"), CCSDSFormat::XML);
    }
}
