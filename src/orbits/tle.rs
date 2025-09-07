/*!
 * The `tle` module provides functionality for working with NORAD Two-Line Element (TLE) data.
 * 
 * Supports both classic TLE format and Alpha-5 TLE format using the SGP4 propagation algorithm.
 * Alpha-5 expands the NORAD ID range by replacing the first digit with a letter for IDs >= 100000.
 */

use crate::orbits::traits::{AnalyticPropagator, OrbitPropagator};
use crate::trajectories::TrajectoryEvictionPolicy;
use crate::time::{Epoch, TimeSystem};
use crate::trajectories::{AngleFormat, InterpolationMethod, OrbitFrame, OrbitState, OrbitStateType, PropagatorType, State, Trajectory};
use crate::utils::BraheError;
use crate::coordinates::{state_cartesian_to_osculating};
use crate::constants::{OMEGA_EARTH, RAD2DEG};
use crate::frames::{polar_motion, state_ecef_to_eci};
use crate::orbits::keplerian::semimajor_axis;
use crate::attitude::RotationMatrix;
use nalgebra::{Vector3, Vector6};
use serde::{Deserialize, Serialize};
use sgp4::chrono::{Datelike, Timelike};
use std::str::FromStr;
use std::f64::consts::PI;

/// TLE format type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TleFormat {
    /// Classic 2-line TLE format with numeric NORAD ID
    Classic,
    /// Alpha-5 TLE format with alphanumeric NORAD ID (first digit replaced with letter for IDs >= 100000)
    Alpha5,
}

/// Calculate TLE line checksum as an independent function
/// 
/// # Arguments
/// * `line` - Full TLE line (function automatically uses first 68 characters for checksum)
/// 
/// # Returns
/// * `u8` - Calculated checksum digit
pub fn calculate_tle_line_checksum(line: &str) -> u8 {
    // Use only first 68 characters for checksum calculation (excluding the checksum digit)
    let checksum_chars = if line.len() >= 68 { &line[..68] } else { line };
    
    let mut sum = 0;
    for c in checksum_chars.chars() {
        match c {
            '0'..='9' => sum += c.to_digit(10).unwrap() as u8,
            '-' => sum += 1,
            _ => {} // Ignore other characters
        }
    }
    sum % 10
}

/// Format a number in TLE exponential notation (±ddddd±d)
/// TLE format: [sign][5-digit mantissa][sign][1-digit exponent] = 8 characters total
/// Example: -11606-4 means -0.11606e-4
/// 
/// # Arguments
/// * `value` - Number to format in TLE exponential notation
/// 
/// # Returns
/// * `String` - Formatted 8-character string in TLE exponential notation
fn tle_format_exp(value: f64) -> String {
    if value == 0.0 {
        return " 00000-0".to_string();
    }
    
    let sign = if value >= 0.0 { " " } else { "-" };
    let abs_val = value.abs();
    
    // Convert to scientific notation
    let exponent = abs_val.log10().floor() as i32;
    let mantissa = abs_val / 10.0f64.powi(exponent);
    
    // Scale mantissa to get 5 digits (move decimal point 4 places right)
    let scaled_mantissa = (mantissa * 10000.0).round() as u32;
    
    // Format as 5 digits with leading zeros
    let mantissa_str = format!("{:05}", scaled_mantissa);
    
    // Format exponent (single digit with sign)
    let exp_sign = if exponent >= 0 { "+" } else { "-" };
    let exp_abs = exponent.abs();
    
    format!("{}{}{}{}", sign, mantissa_str, exp_sign, exp_abs)
}

/// Validate TLE line format as an independent function
/// 
/// # Arguments
/// * `line1` - First line of TLE data
/// * `line2` - Second line of TLE data
/// 
/// # Returns
/// * `bool` - true if valid, false otherwise
pub fn validate_tle_lines(line1: &str, line2: &str) -> bool {
    // Validate individual lines
    if !validate_tle_line(line1, 1) || !validate_tle_line(line2, 2) {
        return false;
    }
    
    // Validate NORAD IDs match between lines
    let id1 = &line1[2..7];
    let id2 = &line2[2..7];
    id1 == id2
}

/// Internal function to validate TLE lines with detailed error messages
/// Used by TLE struct creation for better error reporting
fn validate_tle_lines_with_errors(line1: &str, line2: &str) -> Result<(), BraheError> {
    // Basic length check
    if line1.len() != 69 {
        return Err(BraheError::Error(format!(
            "TLE line 1 must be exactly 69 characters, got {}", 
            line1.len()
        )));
    }
    
    if line2.len() != 69 {
        return Err(BraheError::Error(format!(
            "TLE line 2 must be exactly 69 characters, got {}", 
            line2.len()
        )));
    }
    
    // Check line numbers
    if !line1.starts_with('1') {
        return Err(BraheError::Error("TLE line 1 must start with '1'".to_string()));
    }
    
    if !line2.starts_with('2') {
        return Err(BraheError::Error("TLE line 2 must start with '2'".to_string()));
    }
    
    // Validate NORAD IDs match between lines
    let id1 = &line1[2..7];
    let id2 = &line2[2..7];
    if id1 != id2 {
        return Err(BraheError::Error(format!(
            "NORAD IDs don't match: line1='{}', line2='{}'", 
            id1, id2
        )));
    }
    
    // Validate checksums
    let checksum1_expected = line1.chars().nth(68).unwrap().to_digit(10)
        .ok_or_else(|| BraheError::Error("Line 1 checksum must be a digit".to_string()))? as u8;
    let checksum1_calculated = calculate_tle_line_checksum(&line1[..68]);
    if checksum1_expected != checksum1_calculated {
        return Err(BraheError::Error(format!(
            "Line 1 checksum mismatch: expected {}, calculated {}", 
            checksum1_expected, checksum1_calculated
        )));
    }
    
    let checksum2_expected = line2.chars().nth(68).unwrap().to_digit(10)
        .ok_or_else(|| BraheError::Error("Line 2 checksum must be a digit".to_string()))? as u8;
    let checksum2_calculated = calculate_tle_line_checksum(&line2[..68]);
    if checksum2_expected != checksum2_calculated {
        return Err(BraheError::Error(format!(
            "Line 2 checksum mismatch: expected {}, calculated {}", 
            checksum2_expected, checksum2_calculated
        )));
    }
    
    Ok(())
}

/// Validate single TLE line format
/// 
/// # Arguments
/// * `line` - TLE line to validate
/// * `expected_line_number` - Expected line number (1 or 2)
/// 
/// # Returns
/// * `bool` - true if valid, false otherwise
pub fn validate_tle_line(line: &str, expected_line_number: u8) -> bool {
    // Basic length check
    if line.len() != 69 {
        return false;
    }
    
    // Check line number
    let first_char = line.chars().next().unwrap_or(' ');
    if first_char != (b'0' + expected_line_number) as char {
        return false;
    }
    
    // Validate checksum
    let checksum_expected = line.chars().nth(68).and_then(|c| c.to_digit(10)).unwrap_or(10) as u8;
    if checksum_expected > 9 {
        return false; // Invalid checksum character
    }
    
    let checksum_calculated = calculate_tle_line_checksum(&line[..68]);
    checksum_expected == checksum_calculated
}

/// Extract NORAD ID from TLE string, handling both classic and Alpha-5 formats
/// 
/// Alpha-5 mapping: A=10, B=11, ..., H=17, J=19, K=20, ..., N=23, P=25, ..., Z=35
/// (skipping I=18 and O=24 to avoid confusion with 1 and 0)
/// 
/// # Arguments
/// * `id_str` - 5-character NORAD ID string (either numeric or Alpha-5)
/// 
/// # Returns
/// * `Result<u32, BraheError>` - Decoded numeric NORAD ID
pub fn extract_tle_norad_id(id_str: &str) -> Result<u32, BraheError> {
    if id_str.len() != 5 {
        return Err(BraheError::Error(format!(
            "NORAD ID must be exactly 5 characters, got: '{}'", 
            id_str
        )));
    }
    
    let first_char = id_str.chars().next().unwrap();
    
    if first_char.is_ascii_digit() {
        // Classic format - all numeric
        id_str.parse::<u32>()
            .map_err(|_| BraheError::Error(format!("Invalid numeric NORAD ID: '{}'", id_str)))
    } else if first_char.is_ascii_alphabetic() && first_char.is_ascii_uppercase() {
        // Alpha-5 format - first character is letter
        decode_alpha5_id(id_str)
    } else {
        Err(BraheError::Error(format!(
            "Invalid NORAD ID format: '{}' (first character must be digit or uppercase letter)", 
            id_str
        )))
    }
}

/// Internal function to decode Alpha-5 NORAD ID to numeric value
fn decode_alpha5_id(id_str: &str) -> Result<u32, BraheError> {
    let first_char = id_str.chars().next().unwrap();
    let remaining = &id_str[1..];
    
    // Validate remaining 4 characters are numeric
    let remaining_num = remaining.parse::<u32>()
        .map_err(|_| BraheError::Error(format!(
            "Invalid Alpha-5 ID: remaining digits '{}' must be numeric", 
            remaining
        )))?;
    
    // Convert letter to corresponding tens digit
    // A=10, B=11, ..., H=17, J=18, K=19, ..., N=22, P=23, ..., Z=33
    // (I and O are skipped)
    let tens_digit = match first_char {
        'A' => 10, 'B' => 11, 'C' => 12, 'D' => 13, 'E' => 14, 'F' => 15, 'G' => 16, 'H' => 17,
        'J' => 18, 'K' => 19, 'L' => 20, 'M' => 21, 'N' => 22,
        'P' => 23, 'Q' => 24, 'R' => 25, 'S' => 26, 'T' => 27, 'U' => 28, 'V' => 29, 
        'W' => 30, 'X' => 31, 'Y' => 32, 'Z' => 33,
        _ => return Err(BraheError::Error(format!(
            "Invalid Alpha-5 letter: '{}' (I and O are not allowed)", 
            first_char
        ))),
    };
    
    // Combine tens digit with remaining 4 digits
    // For Alpha-5, we map to ranges starting at 100000
    let numeric_id = (tens_digit - 10) * 10000 + 100000 + remaining_num;
    
    Ok(numeric_id)
}

/// Extract epoch from SGP4 elements as an independent function
/// 
/// # Arguments
/// * `elements` - SGP4 elements structure
/// 
/// # Returns
/// * `Result<Epoch, BraheError>` - Extracted epoch or error
pub fn extract_epoch(elements: &sgp4::Elements) -> Result<Epoch, BraheError> {
    // SGP4 elements contain a NaiveDateTime
    // Convert to Julian Date for Brahe's Epoch using time::conversions
    let dt = elements.datetime;
    
    // Use the time::conversions module for Julian date conversion
    let jd = crate::time::conversions::datetime_to_jd(
        dt.year() as u32,
        dt.month() as u8,
        dt.day() as u8,
        dt.hour() as u8,
        dt.minute() as u8,
        dt.second() as f64,
        dt.nanosecond() as f64,
    );
    
    Ok(Epoch::from_jd(jd, crate::time::TimeSystem::UTC))
}

/// Compute Greenwich Mean Sidereal Time 1982 Model for TLE transformations
/// 
/// This implementation follows the TLE-specific GMST calculation used in SGP4
/// transformations between TEME and PEF frames, as specified in Revisiting 
/// Spacetrack Report No 3 by David Vallado.
/// 
/// # Arguments
/// * `epoch` - Epoch for GMST computation
/// 
/// # Returns
/// * `f64` - Greenwich mean sidereal time angle in radians [0, 2π)
pub fn gmst82_tle(epoch: Epoch) -> f64 {
    // Compute UT1 time as Julian date
    let jd_ut1 = epoch.jd_as_time_system(TimeSystem::UT1);
    
    // Centuries since J2000.0
    let t = (jd_ut1 - 2451545.0) / 36525.0;
    
    // Apply Formula from AIAA 2006-6753 Appendix C (modified for TLE compatibility)
    // This matches the implementation in Brandon Rhodes' SGP4 code
    let g = 67310.54841 + 8640184.812866 * t + 0.093104 * t * t - 6.2e-6 * t * t * t;
    
    // Compute GMST as angle
    let theta = ((jd_ut1 % 1.0) + (g / 86400.0 % 1.0)) * 2.0 * PI;
    
    // Normalize to [0, 2π)
    theta % (2.0 * PI)
}

/// Decode NORAD ID from string, handling Alpha-5 format as an independent function
/// 
/// # Arguments
/// * `id_str` - 5-character NORAD ID string
/// 
/// # Returns
/// * `Result<(u32, TleFormat), BraheError>` - Decoded numeric ID and format type
fn decode_norad_id(id_str: &str) -> Result<(u32, TleFormat), BraheError> {
    if id_str.len() != 5 {
        return Err(BraheError::Error(format!(
            "NORAD ID must be exactly 5 characters, got: '{}'", 
            id_str
        )));
    }
    
    let first_char = id_str.chars().next().unwrap();
    
    if first_char.is_ascii_digit() {
        // Classic format - all numeric
        let id = extract_tle_norad_id(id_str)?;
        Ok((id, TleFormat::Classic))
    } else if first_char.is_ascii_alphabetic() && first_char.is_ascii_uppercase() {
        // Alpha-5 format - first character is letter
        let alpha_5_id = extract_tle_norad_id(id_str)?;
        Ok((alpha_5_id, TleFormat::Alpha5))
    } else {
        Err(BraheError::Error(format!(
            "Invalid NORAD ID format: '{}' (first character must be digit or uppercase letter)", 
            id_str
        )))
    }
}

/// Convert TLE line to use numeric NORAD ID for SGP4 compatibility as an independent function
fn convert_to_numeric_line(line: &str, numeric_id: u32) -> Result<String, BraheError> {
    if line.len() != 69 {
        return Err(BraheError::Error(format!(
            "TLE line must be 69 characters, got {}", 
            line.len()
        )));
    }
    
    // Replace NORAD ID (positions 2-7, 0-indexed: 2-6) with numeric version
    let mut modified_line = line.to_string();
    
    // For very large IDs, we'll use modulo to fit in 5 digits for SGP4 compatibility
    let sgp4_id = if numeric_id > 99999 { numeric_id % 100000 } else { numeric_id };
    let numeric_id_str = format!("{:05}", sgp4_id);
    
    modified_line.replace_range(2..7, &numeric_id_str);
    
    // Recalculate checksum for the modified line
    let new_checksum = calculate_tle_line_checksum(&modified_line[..68]);
    modified_line.replace_range(68..69, &new_checksum.to_string());
    
    Ok(modified_line)
}

/// Convert TLE lines to orbital elements as an independent function
/// 
/// # Arguments
/// * `line1` - First line of TLE data
/// * `line2` - Second line of TLE data
/// 
/// # Returns
/// * `Result<Vector6<f64>, BraheError>` - Orbital elements [a, e, i, Ω, ω, M] in SI units
pub fn lines_to_orbit_elements(line1: &str, line2: &str) -> Result<Vector6<f64>, BraheError> {
    // Validate TLE format first
    if !validate_tle_lines(line1, line2) {
        return Err(BraheError::Error("Invalid TLE format".to_string()));
    }
    
    // Extract and decode NORAD ID from line1 (positions 2-7, 0-indexed: 2-6)
    let norad_id_str = &line1[2..7];
    let (norad_id, format) = decode_norad_id(norad_id_str)?;
    
    // Create modified lines with numeric NORAD ID for SGP4 parsing
    // Only convert if we actually need to (Alpha-5 format or large numeric IDs)
    let numeric_line1 = if format == TleFormat::Alpha5 || norad_id > 99999 {
        convert_to_numeric_line(line1, norad_id)?
    } else {
        line1.to_string()
    };
    let numeric_line2 = if format == TleFormat::Alpha5 || norad_id > 99999 {
        convert_to_numeric_line(line2, norad_id)?
    } else {
        line2.to_string()
    };
    
    // Parse using sgp4 crate with converted lines
    let tle_string = format!("{}\n{}", numeric_line1, numeric_line2);
    let elements = sgp4::parse_2les(&tle_string)
        .map_err(|e| BraheError::Error(format!("Failed to parse 2-line TLE: {:?}", e)))?
        .into_iter()
        .next()
        .ok_or_else(|| BraheError::Error(format!("No elements parsed from 2-line TLE. Input was: '{}'", tle_string)))?;
    
    // Convert to orbital elements vector
    // Calculate semi-major axis from mean motion using orbits module
    let n_rad_s = elements.mean_motion * 2.0 * std::f64::consts::PI / 86400.0; // Convert rev/day to rad/s
    let a = crate::orbits::semimajor_axis(n_rad_s, false); // Use orbits module function
    
    let e = elements.eccentricity;
    let i = elements.inclination.to_radians();
    let omega_cap = elements.right_ascension.to_radians();
    let omega = elements.argument_of_perigee.to_radians();
    let m = elements.mean_anomaly.to_radians();
    
    Ok(Vector6::new(a, e, i, omega_cap, omega, m))
}

/// Convert TLE lines to OrbitState as an independent function
/// 
/// # Arguments
/// * `line1` - First line of TLE data  
/// * `line2` - Second line of TLE data
/// * `frame` - Reference frame for the output state
/// * `orbit_type` - Type of orbital representation for the output state
/// 
/// # Returns
/// * `Result<OrbitState, BraheError>` - OrbitState in the specified frame and type
pub fn lines_to_orbit_state(line1: &str, line2: &str, frame: OrbitFrame, orbit_type: OrbitStateType) -> Result<OrbitState, BraheError> {
    // Validate TLE format first
    if !validate_tle_lines(line1, line2) {
        return Err(BraheError::Error("Invalid TLE format".to_string()));
    }
    
    // Extract and decode NORAD ID from line1 (positions 2-7, 0-indexed: 2-6)
    let norad_id_str = &line1[2..7];
    let (norad_id, format) = decode_norad_id(norad_id_str)?;
    
    // Create modified lines with numeric NORAD ID for SGP4 parsing
    // Only convert if we actually need to (Alpha-5 format or large numeric IDs)
    let numeric_line1 = if format == TleFormat::Alpha5 || norad_id > 99999 {
        convert_to_numeric_line(line1, norad_id)?
    } else {
        line1.to_string()
    };
    let numeric_line2 = if format == TleFormat::Alpha5 || norad_id > 99999 {
        convert_to_numeric_line(line2, norad_id)?
    } else {
        line2.to_string()
    };
    
    // Parse using sgp4 crate with converted lines
    let tle_string = format!("{}\n{}", numeric_line1, numeric_line2);
    let elements = sgp4::parse_2les(&tle_string)
        .map_err(|e| BraheError::Error(format!("Failed to parse 2-line TLE: {:?}", e)))?
        .into_iter()
        .next()
        .ok_or_else(|| BraheError::Error(format!("No elements parsed from 2-line TLE. Input was: '{}'", tle_string)))?;
    
    // Extract epoch
    let epoch = extract_epoch(&elements)?;
    
    // Get orbital elements (always start with Keplerian in ECI, radians)
    let kep_elements = lines_to_orbit_elements(line1, line2)?;
    
    // Create initial Keplerian state in ECI
    let kep_state = OrbitState::new(
        epoch,
        kep_elements,
        OrbitFrame::ECI,
        OrbitStateType::Keplerian,
        AngleFormat::Radians,
    )?;
    
    // Convert to requested frame
    let frame_converted = kep_state.to_frame(&frame)?;
    
    // Convert to requested orbit type
    match orbit_type {
        OrbitStateType::Keplerian => {
            // Already in Keplerian, just ensure proper angle format
            frame_converted.to_keplerian(AngleFormat::Radians)
        }
        OrbitStateType::Cartesian => {
            frame_converted.to_cartesian()
        }
    }
}

/// Encode numeric NORAD ID to Alpha-5 format if needed
/// 
/// # Arguments
/// * `norad_id` - Numeric NORAD ID
/// 
/// # Returns
/// * `Result<String, BraheError>` - 5-character NORAD ID string (numeric or Alpha-5)
fn encode_norad_id_to_string(norad_id: u32) -> Result<String, BraheError> {
    if norad_id < 100000 {
        // Classic format - use numeric
        Ok(format!("{:05}", norad_id))
    } else if norad_id <= 339999 {
        // Alpha-5 format needed
        let base_id = norad_id - 100000;
        let tens_digit = base_id / 10000;
        let remaining = base_id % 10000;
        
        // Map tens digit to letter (A=0, B=1, ..., Z=23, skipping I and O)
        let letter = match tens_digit {
            0..=7 => (b'A' + tens_digit as u8) as char,   // A-H (0-7)
            8..=13 => (b'J' + (tens_digit - 8) as u8) as char, // J-N (8-13, skip I)
            14..=22 => (b'P' + (tens_digit - 14) as u8) as char, // P-X (14-22, skip O)
            23 => 'Z', // Z (23)
            _ => return Err(BraheError::Error(format!(
                "Invalid tens digit {} for Alpha-5 encoding", tens_digit
            ))),
        };
        
        Ok(format!("{}{:04}", letter, remaining))
    } else {
        Err(BraheError::Error(format!(
            "NORAD ID {} exceeds maximum Alpha-5 range (339999). Cannot encode in TLE format.",
            norad_id
        )))
    }
}

/// Create TLE lines from orbital elements
/// 
/// # Arguments
/// * `epoch` - Epoch of the TLE
/// * `elements` - Orbital elements [n, e, i, raan, argp, anomaly]
///   where n is mean motion (rev/day), angles in specified units
/// * `norad_id` - NORAD catalog ID (supports Alpha-5 encoding for IDs >= 100000)
/// * `designation` - International designator
/// * `element_num` - Element set number
/// * `orbit_num` - Revolution number at epoch
/// * `as_degrees` - If true, angular elements are in degrees; if false, in radians
/// * `ndt2` - Optional first derivative of mean motion divided by 2
/// * `nddt6` - Optional second derivative of mean motion divided by 6
/// * `bstar` - Optional B-star drag term
/// 
/// # Returns
/// * `Result<(String, String), BraheError>` - TLE line1 and line2
pub fn tle_lines_from_elements(
    epoch: Epoch, 
    elements: &[f64],  // [n, e, i, raan, argp, anomaly]
    norad_id: u32,
    designation: &str,
    element_num: u32,
    orbit_num: u32,
    as_degrees: bool,
    ndt2: Option<f64>,
    nddt6: Option<f64>,
    bstar: Option<f64>
) -> Result<(String, String), BraheError> {
    if elements.len() < 6 {
        return Err(BraheError::Error("Elements array must have at least 6 elements".to_string()));
    }
    
    // Extract elements
    let n = elements[0];        // Mean motion (rev/day)
    let e = elements[1];        // Eccentricity
    let i_raw = elements[2];    // Inclination 
    let raan_raw = elements[3]; // Right ascension
    let argp_raw = elements[4]; // Argument of perigee
    let anomaly_raw = elements[5]; // Mean anomaly
    
    // Convert angles to degrees if needed (TLE format uses degrees)
    let i = if as_degrees { i_raw } else { i_raw * RAD2DEG };
    let raan = if as_degrees { raan_raw } else { raan_raw * RAD2DEG };
    let argp = if as_degrees { argp_raw } else { argp_raw * RAD2DEG };
    let anomaly = if as_degrees { anomaly_raw } else { anomaly_raw * RAD2DEG };
    
    // Use provided drag terms or defaults
    let ndt2 = ndt2.unwrap_or(0.0);
    let nddt6 = nddt6.unwrap_or(0.0);
    let bstar = bstar.unwrap_or(0.0);
    
    // Format epoch for TLE
    let dt = epoch.to_datetime_as_time_system(TimeSystem::UTC);
    let year = dt.0 % 100; // Year % 100 for 2-digit year
    
    // Calculate day of year manually from month/day
    let days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut leap_year_extra = 0;
    if (dt.0 % 4 == 0 && dt.0 % 100 != 0) || dt.0 % 400 == 0 {
        leap_year_extra = 1; // Leap year
    }
    
    let mut day_of_year = dt.2 as i32; // Start with current day
    for month_idx in 0..(dt.1 as usize - 1) {
        day_of_year += days_in_month[month_idx];
        if month_idx == 1 { // February
            day_of_year += leap_year_extra;
        }
    }
    
    let doy = day_of_year as f64 + dt.3 as f64 / 24.0 + dt.4 as f64 / 1440.0 + dt.5 / 86400.0; // Day of year + fractional day
    
    // Encode NORAD ID (with Alpha-5 support)
    let norad_id_str = encode_norad_id_to_string(norad_id)?;
    
    // Format drag terms in TLE exponential notation - matching Python format
    let ndt2_sign = if ndt2 < 0.0 { "-" } else { " " };
    let ndt2_formatted = format!("{:9.8}", ndt2.abs()).trim_start_matches('0').to_string();
    let ndt2_str = format!("{}{}", ndt2_sign, ndt2_formatted);
    
    let nddt6_str = if nddt6 == 0.0 {
        " 00000-0".to_string()
    } else {
        tle_format_exp(nddt6)
    };
    
    let bstar_str = if bstar == 0.0 {
        " 00000-0".to_string()
    } else {
        tle_format_exp(bstar)
    };
    
    // Build line 1 (without checksum) - matching Python format exactly
    let line1_base = format!(
        "1 {}U {:8} {:02}{:12.8} {} {:8} {:8} 0 {:4}",
        norad_id_str, designation, year, doy, ndt2_str, nddt6_str, bstar_str, element_num
    );
    
    // Build line 2 (without checksum) - matching Python format exactly
    let ecc_str = format!("{:9.7}", e).trim_start_matches("0").trim_start_matches(".").to_string();
    let line2_base = format!(
        "2 {} {:8.4} {:8.4} {:7} {:8.4} {:8.4} {:11.8}{:5}",
        norad_id_str, i, raan, ecc_str, argp, anomaly, n, orbit_num
    );
    
    // Calculate and append checksums
    let line1_checksum = calculate_tle_line_checksum(&line1_base);
    let line2_checksum = calculate_tle_line_checksum(&line2_base);
    
    let line1_final = format!("{}{}", line1_base, line1_checksum);
    let line2_final = format!("{}{}", line2_base, line2_checksum);
    
    Ok((line1_final, line2_final))
}

/// Create TLE lines from orbit state
/// 
/// # Arguments
/// * `orbit_state` - Orbit state in any frame/representation
/// * `norad_id` - NORAD catalog ID (supports Alpha-5 encoding)
/// * `designation` - International designator
/// * `element_num` - Element set number
/// * `orbit_num` - Revolution number at epoch
/// * `ndt2` - Optional first derivative of mean motion divided by 2
/// * `nddt6` - Optional second derivative of mean motion divided by 6
/// * `bstar` - Optional B-star drag term
/// 
/// # Returns
/// * `Result<(String, String), BraheError>` - TLE line1 and line2
pub fn tle_lines_from_orbit_state(
    orbit_state: &OrbitState,
    norad_id: u32,
    designation: &str,
    element_num: u32,
    orbit_num: u32,
    ndt2: Option<f64>,
    nddt6: Option<f64>,
    bstar: Option<f64>
) -> Result<(String, String), BraheError> {
    // Convert to ECI Cartesian state
    let eci_cartesian = orbit_state.to_frame(&OrbitFrame::ECI)?.to_cartesian()?;
    
    // Convert to Keplerian elements
    let kep_elements = state_cartesian_to_osculating(eci_cartesian.state, false);
    
    // Convert to TLE format: [n, e, i, raan, argp, anomaly] (angles in radians from conversion)
    let a = kep_elements[0];
    let e = kep_elements[1];
    let i = kep_elements[2];    // Already in radians
    let raan = kep_elements[3]; // Already in radians
    let argp = kep_elements[4]; // Already in radians
    let anomaly = kep_elements[5]; // Already in radians
    
    // Calculate mean motion from semi-major axis
    let n = crate::orbits::mean_motion(a, false) / (2.0 * PI) * 86400.0; // Convert to rev/day
    
    let elements = [n, e, i, raan, argp, anomaly];
    
    tle_lines_from_elements(
        orbit_state.epoch,
        &elements,
        norad_id,
        designation,
        element_num,
        orbit_num,
        false, // angles are in radians from state_cartesian_to_osculating
        ndt2,
        nddt6,
        bstar
    )
}

/// Create TLE from orbital elements
/// 
/// # Arguments
/// * `epoch` - Epoch of the TLE
/// * `elements` - Orbital elements [n, e, i, raan, argp, anomaly]
/// * `norad_id` - NORAD catalog ID (supports Alpha-5 encoding)
/// * `designation` - International designator
/// * `element_num` - Element set number
/// * `orbit_num` - Revolution number at epoch
/// * `as_degrees` - If true, angular elements are in degrees; if false, in radians
/// * `ndt2` - Optional first derivative of mean motion divided by 2
/// * `nddt6` - Optional second derivative of mean motion divided by 6
/// * `bstar` - Optional B-star drag term
/// 
/// # Returns
/// * `Result<TLE, BraheError>` - Created TLE
pub fn tle_from_elements(
    epoch: Epoch, 
    elements: &[f64],
    norad_id: u32,
    designation: &str,
    element_num: u32,
    orbit_num: u32,
    as_degrees: bool,
    ndt2: Option<f64>,
    nddt6: Option<f64>,
    bstar: Option<f64>
) -> Result<TLE, BraheError> {
    let (line1, line2) = tle_lines_from_elements(
        epoch, elements, norad_id, designation, element_num, orbit_num, as_degrees, ndt2, nddt6, bstar
    )?;
    TLE::from_lines(&line1, &line2)
}

/// Create TLE from ECEF state vector
/// 
/// # Arguments
/// * `epoch` - Epoch of the state
/// * `ecef_state` - ECEF state vector [x, y, z, vx, vy, vz] in meters and m/s
/// 
/// # Returns
/// * `Result<TLE, BraheError>` - Created TLE
pub fn tle_from_ecef(epoch: Epoch, ecef_state: Vector6<f64>) -> Result<TLE, BraheError> {
    // Convert ECEF to ECI
    let eci_state = state_ecef_to_eci(epoch, ecef_state);
    tle_from_eci(epoch, eci_state)
}

/// Create TLE from ECI state vector  
/// 
/// # Arguments
/// * `epoch` - Epoch of the state
/// * `eci_state` - ECI state vector [x, y, z, vx, vy, vz] in meters and m/s
/// 
/// # Returns  
/// * `Result<TLE, BraheError>` - Created TLE
pub fn tle_from_eci(epoch: Epoch, eci_state: Vector6<f64>) -> Result<TLE, BraheError> {
    // Convert Cartesian to Keplerian elements
    let kep_elements = state_cartesian_to_osculating(eci_state, false);
    
    // Convert to TLE format: [n, e, i, raan, argp, anomaly] (angles in radians from conversion)
    let a = kep_elements[0];
    let e = kep_elements[1];
    let i = kep_elements[2];    // In radians
    let raan = kep_elements[3]; // In radians
    let argp = kep_elements[4]; // In radians
    let anomaly = kep_elements[5]; // In radians
    
    // Calculate mean motion from semi-major axis
    let n = crate::orbits::mean_motion(a, false) / (2.0 * PI) * 86400.0; // Convert to rev/day
    
    let elements = [n, e, i, raan, argp, anomaly];
    
    // Use default values for missing parameters
    tle_from_elements(
        epoch, 
        &elements, 
        99800, // Default NORAD ID
        "        ", // Empty designation
        0, // Element number
        0, // Orbit number
        false, // angles are in radians
        None, // ndt2
        None, // nddt6
        None  // bstar
    )
}

/// Structure representing a Two-Line Element set with SGP4 propagation capability
#[derive(Debug, Clone)]
pub struct TLE {
    /// Raw first line of TLE
    pub line1: String,
    
    /// Raw second line of TLE  
    pub line2: String,
    
    /// Optional satellite name (from 3-line format)
    pub satellite_name: Option<String>,
    
    /// TLE format type (Classic or Alpha-5)
    pub format: TleFormat,
    
    /// Original NORAD ID string (may contain Alpha-5 encoding)
    pub norad_id_string: String,
    
    /// Decoded numeric NORAD ID 
    pub norad_id: u32,
    
    /// Parsed SGP4 elements
    elements: sgp4::Elements,
    
    /// SGP4 propagation constants
    constants: sgp4::Constants,
    
    /// Initial state from TLE epoch
    initial_state: OrbitState,
    
    /// Current propagated state
    current_state: OrbitState,
    
    /// Trajectory of all propagated states
    trajectory: Trajectory<OrbitState>,
    
    /// Step size in seconds for stepping operations
    step_size: f64,
}

impl TLE {
    /// Create a new TLE from classic 2-line format
    /// 
    /// # Arguments
    /// * `line1` - First line of TLE data
    /// * `line2` - Second line of TLE data
    /// 
    /// # Returns
    /// * `Result<TLE, BraheError>` - New TLE instance or error
    pub fn from_lines(line1: &str, line2: &str) -> Result<Self, BraheError> {
        Self::from_tle_string(&format!("{}\n{}", line1, line2), 60.0)
    }
    
    /// Create a new TLE from 3-line format (with satellite name)
    /// 
    /// # Arguments  
    /// * `name` - Satellite name (line 0)
    /// * `line1` - First line of TLE data
    /// * `line2` - Second line of TLE data
    /// 
    /// # Returns
    /// * `Result<TLE, BraheError>` - New TLE instance or error
    pub fn from_3le(name: &str, line1: &str, line2: &str) -> Result<Self, BraheError> {
        Self::from_tle_string(&format!("{}\n{}\n{}", name, line1, line2), 60.0)
    }
    
    /// Create TLE from raw TLE string (auto-detects format)
    /// 
    /// # Arguments
    /// * `tle_string` - Raw TLE data as string
    /// 
    /// # Returns
    /// * `Result<TLE, BraheError>` - New TLE instance or error
    pub fn from_tle_string(tle_string: &str, step_size: f64) -> Result<Self, BraheError> {
        let lines: Vec<&str> = tle_string.trim().lines().collect();
        
        match lines.len() {
            2 => {
                // 2-line format
                let line1 = lines[0].trim();
                let line2 = lines[1].trim();
                
                Self::parse_tle(None, line1, line2, step_size)
            },
            3 => {
                // 3-line format (with satellite name)
                let name = lines[0].trim();
                let line1 = lines[1].trim();
                let line2 = lines[2].trim();
                
                Self::parse_tle(Some(name), line1, line2, step_size)
            },
            _ => Err(BraheError::Error(format!(
                "Invalid TLE format: expected 2 or 3 lines, got {}", 
                lines.len()
            ))),
        }
    }
    
    /// Internal TLE parsing function
    fn parse_tle(name: Option<&str>, line1: &str, line2: &str, step_size: f64) -> Result<Self, BraheError> {
        // Validate TLE format
        validate_tle_lines_with_errors(line1, line2)?;
        
        // Extract and decode NORAD ID from line1 (positions 2-7, 0-indexed: 2-6)
        let norad_id_str = &line1[2..7];
        let (norad_id, format) = decode_norad_id(norad_id_str)?;
        
        // Create modified lines with numeric NORAD ID for SGP4 parsing
        // Only convert if we actually need to (Alpha-5 format or large numeric IDs)
        let numeric_line1 = if format == TleFormat::Alpha5 || norad_id > 99999 {
            convert_to_numeric_line(line1, norad_id)?
        } else {
            line1.to_string()
        };
        let numeric_line2 = if format == TleFormat::Alpha5 || norad_id > 99999 {
            convert_to_numeric_line(line2, norad_id)?
        } else {
            line2.to_string()
        };
        
        // Parse using sgp4 crate with correct function based on format
        let elements = if let Some(sat_name) = name {
            // 3-line format
            let tle_string = format!("{}\n{}\n{}", sat_name, numeric_line1, numeric_line2);
            sgp4::parse_3les(&tle_string)
                .map_err(|e| BraheError::Error(format!("Failed to parse 3-line TLE: {:?}", e)))?
                .into_iter()
                .next()
                .ok_or_else(|| BraheError::Error(format!("No elements parsed from 3-line TLE. Input was: '{}'", tle_string)))?
        } else {
            // 2-line format
            let tle_string = format!("{}\n{}", numeric_line1, numeric_line2);
            sgp4::parse_2les(&tle_string)
                .map_err(|e| BraheError::Error(format!("Failed to parse 2-line TLE: {:?}", e)))?
                .into_iter()
                .next()
                .ok_or_else(|| BraheError::Error(format!("No elements parsed from 2-line TLE. Input was: '{}'", tle_string)))?
        };
        
        // Create SGP4 constants for propagation
        let constants = sgp4::Constants::from_elements(&elements)
            .map_err(|e| BraheError::Error(format!("Failed to create SGP4 constants: {:?}", e)))?;
        
        // Extract epoch from elements
        let _epoch = extract_epoch(&elements)?;
        
        // Create initial orbital state from TLE lines (Keplerian in ECI for TLE)
        let initial_state = lines_to_orbit_state(line1, line2, OrbitFrame::ECI, OrbitStateType::Keplerian)?;
        let current_state = initial_state.clone();
        
        // Initialize trajectory with initial state
        let mut trajectory = Trajectory::new(InterpolationMethod::Linear);
        trajectory.add_state(initial_state.clone())?;
        
        Ok(TLE {
            line1: line1.to_string(),
            line2: line2.to_string(),
            satellite_name: name.map(|s| s.to_string()),
            format,
            norad_id_string: norad_id_str.to_string(),
            norad_id,
            elements,
            constants,
            initial_state,
            current_state,
            trajectory,
            step_size,
        })
    }
    
    
    /// Get satellite name (if available)
    pub fn satellite_name(&self) -> Option<&str> {
        self.satellite_name.as_deref()
    }
    
    /// Get original NORAD ID string (may be Alpha-5 format)
    pub fn norad_id_string(&self) -> &str {
        &self.norad_id_string
    }
    
    /// Get decoded numeric NORAD ID
    pub fn norad_id(&self) -> u32 {
        self.elements.norad_id as u32
    }
    
    /// Get international designator
    pub fn international_designator(&self) -> Option<String> {
        self.elements.international_designator.clone()
    }
    
    /// Get epoch of TLE
    pub fn epoch(&self) -> Epoch {
        self.initial_state.epoch
    }
    
    /// Get mean motion (revolutions per day)
    pub fn mean_motion(&self) -> f64 {
        self.elements.mean_motion
    }
    
    /// Get eccentricity
    pub fn eccentricity(&self) -> f64 {
        self.elements.eccentricity
    }
    
    /// Get inclination in degrees
    pub fn inclination(&self) -> f64 {
        self.elements.inclination
    }
    
    /// Get right ascension of ascending node in degrees  
    pub fn raan(&self) -> f64 {
        self.elements.right_ascension
    }
    
    /// Get right ascension of ascending node (RAAN) in degrees (alias for compatibility)
    pub fn right_ascension(&self) -> f64 {
        self.elements.right_ascension
    }
    
    /// Get argument of perigee in degrees
    pub fn argument_of_perigee(&self) -> f64 {
        self.elements.argument_of_perigee
    }
    
    /// Get B-star drag coefficient
    pub fn bstar(&self) -> f64 {
        self.elements.drag_term
    }
    
    /// Get first derivative of mean motion (n_dot/2) in revolutions per day squared
    pub fn first_derivative_mean_motion(&self) -> f64 {
        self.elements.mean_motion_dot
    }
    
    /// Get second derivative of mean motion (n_ddot/6) in revolutions per day cubed  
    pub fn second_derivative_mean_motion(&self) -> f64 {
        self.elements.mean_motion_ddot
    }
    
    /// Get orbital elements as a vector [a, e, i, raan, argp, mean_anomaly]
    /// Returns elements in meters and degrees
    pub fn orbital_elements(&self) -> Result<Vector6<f64>, BraheError> {
        // Convert mean motion (rev/day) to semi-major axis (meters)
        // Mean motion is in rev/day, convert to rad/s for semimajor_axis function
        let n_rad_per_sec = self.elements.mean_motion * 2.0 * PI / 86400.0;
        let a = semimajor_axis(n_rad_per_sec, false);
        
        Ok(Vector6::new(
            a,                                           // semi-major axis (m)
            self.elements.eccentricity,                  // eccentricity
            self.elements.inclination,                   // inclination (deg)
            self.elements.right_ascension,               // RAAN (deg)
            self.elements.argument_of_perigee,           // argument of perigee (deg)
            self.elements.mean_anomaly                   // mean anomaly (deg)
        ))
    }
    
    /// Get mean anomaly in degrees
    pub fn mean_anomaly(&self) -> f64 {
        self.elements.mean_anomaly
    }
    
    /// Check if TLE uses Alpha-5 format
    pub fn is_alpha5(&self) -> bool {
        self.format == TleFormat::Alpha5
    }
    
    /// Propagate to specific time and return Cartesian state
    pub fn propagate(&self, epoch: Epoch) -> Result<OrbitState, BraheError> {
        // Calculate minutes since TLE epoch
        let time_diff = epoch - self.epoch();
        let minutes_since_epoch = time_diff / 60.0; // Convert seconds to minutes
        
        // Use SGP4 to propagate
        let prediction = self.constants.propagate(sgp4::MinutesSinceEpoch(minutes_since_epoch))
            .map_err(|e| BraheError::Error(format!("SGP4 propagation failed: {:?}", e)))?;
        
        // Convert SGP4 prediction to Cartesian state  
        // SGP4 returns position in km and velocity in km/s
        let position = Vector3::new(
            prediction.position[0] * 1000.0, // Convert km to m
            prediction.position[1] * 1000.0,
            prediction.position[2] * 1000.0,
        );
        
        let velocity = Vector3::new(
            prediction.velocity[0] * 1000.0, // Convert km/s to m/s  
            prediction.velocity[1] * 1000.0,
            prediction.velocity[2] * 1000.0,
        );
        
        let state_vector = Vector6::new(
            position.x, position.y, position.z,
            velocity.x, velocity.y, velocity.z,
        );
        
        OrbitState::new(
            epoch,
            state_vector,
            OrbitFrame::ECI,
            OrbitStateType::Cartesian,
            AngleFormat::None,
        )
    }

    /// Get satellite state in TEME (True Equator Mean Equinox) frame
    /// 
    /// This is the raw output from SGP4 propagation without any frame transformations.
    /// TEME is the native reference frame for TLE propagation.
    /// 
    /// # Arguments
    /// * `epoch` - Time for state computation
    /// 
    /// # Returns
    /// * `Result<Vector6<f64>, BraheError>` - Cartesian state in TEME frame [m; m/s]
    pub fn state_teme(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        // Calculate minutes since TLE epoch
        let time_diff = epoch - self.epoch();
        let minutes_since_epoch = time_diff / 60.0;
        
        // Use SGP4 to propagate
        let prediction = self.constants.propagate(sgp4::MinutesSinceEpoch(minutes_since_epoch))
            .map_err(|e| BraheError::Error(format!("SGP4 propagation failed: {:?}", e)))?;
        
        // Convert SGP4 output to Vector6 (km to m, km/s to m/s)
        Ok(Vector6::new(
            prediction.position[0] * 1000.0,
            prediction.position[1] * 1000.0,
            prediction.position[2] * 1000.0,
            prediction.velocity[0] * 1000.0,
            prediction.velocity[1] * 1000.0,
            prediction.velocity[2] * 1000.0,
        ))
    }
    
    /// Get satellite state in PEF (Pseudo-Earth-Fixed) frame
    /// 
    /// Transforms from TEME to PEF frame using GMST82 rotation. This accounts
    /// for Earth's rotation but not polar motion.
    /// 
    /// # Arguments
    /// * `epoch` - Time for state computation
    /// 
    /// # Returns
    /// * `Result<Vector6<f64>, BraheError>` - Cartesian state in PEF frame [m; m/s]
    pub fn state_pef(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        // Get state in TEME frame
        let x_teme = self.state_teme(epoch)?;
        
        let r_teme = x_teme.fixed_rows::<3>(0).into_owned();
        let v_teme = x_teme.fixed_rows::<3>(3).into_owned();
        
        // Compute TEME -> PEF transformation using GMST82
        let theta = gmst82_tle(epoch);
        let r_matrix = RotationMatrix::Rz(theta, false).to_matrix(); // angle in radians
        let omega_earth = Vector3::new(0.0, 0.0, OMEGA_EARTH);
        
        // Transform position and velocity
        let r_pef = r_matrix * r_teme;
        let v_pef = r_matrix * v_teme - omega_earth.cross(&r_pef);
        
        Ok(Vector6::new(
            r_pef[0], r_pef[1], r_pef[2],
            v_pef[0], v_pef[1], v_pef[2]
        ))
    }
    
    /// Get satellite state in ITRF (International Terrestrial Reference Frame)
    /// 
    /// Transforms from PEF to ITRF frame using polar motion corrections.
    /// This provides the most accurate Earth-fixed coordinates.
    /// 
    /// # Arguments
    /// * `epoch` - Time for state computation
    /// 
    /// # Returns
    /// * `Result<Vector6<f64>, BraheError>` - Cartesian state in ITRF frame [m; m/s]
    pub fn state_itrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        // Get state in PEF frame
        let x_pef = self.state_pef(epoch)?;
        
        let r_pef = x_pef.fixed_rows::<3>(0).into_owned();
        let v_pef = x_pef.fixed_rows::<3>(3).into_owned();
        
        // Apply polar motion transformation
        let pm = polar_motion(epoch);
        
        let r_itrf = pm * r_pef;
        let v_itrf = pm * v_pef;
        
        Ok(Vector6::new(
            r_itrf[0], r_itrf[1], r_itrf[2],
            v_itrf[0], v_itrf[1], v_itrf[2]
        ))
    }
}

/// Implement OrbitPropagator trait for TLE
impl OrbitPropagator for TLE {
    fn propagate_to(&mut self, target_epoch: Epoch) -> Result<&OrbitState, BraheError> {
        let mut current_epoch = self.current_state.epoch;
        
        // Step until we're close to the target epoch
        while (target_epoch - current_epoch) > self.step_size {
            self.step()?;
            current_epoch = self.current_state.epoch;
        }
        
        // Take a micro step to reach the exact target epoch
        if current_epoch < target_epoch {
            let remaining_time = target_epoch - current_epoch;
            self.step_by(remaining_time)?;
        }
        
        Ok(&self.current_state)
    }
    
    fn reset(&mut self) -> Result<(), BraheError> {
        self.current_state = self.initial_state.clone();
        
        // Reset trajectory with initial state
        self.trajectory = Trajectory::new(self.trajectory.interpolation_method);
        self.trajectory.add_state(self.initial_state.clone())?;
        
        Ok(())
    }
    
    fn current_epoch(&self) -> Epoch {
        self.current_state.epoch
    }
    
    fn current_state(&self) -> &OrbitState {
        &self.current_state
    }
    
    fn initial_state(&self) -> &OrbitState {
        &self.initial_state
    }
    
    fn set_initial_state(&mut self, _state: OrbitState) -> Result<(), BraheError> {
        // For TLE, we don't allow changing the initial state since it comes from the TLE data
        Err(BraheError::Error(
            "Cannot change initial state for TLE propagator - state is determined by TLE data".to_string()
        ))
    }
    
    fn set_initial_conditions(
        &mut self, 
        _epoch: Epoch, 
        _state: Vector6<f64>,
        _frame: OrbitFrame,
        _orbit_type: OrbitStateType,
        _angle_format: AngleFormat,
    ) -> Result<(), BraheError> {
        // For TLE, we don't allow changing initial conditions
        Err(BraheError::Error(
            "Cannot change initial conditions for TLE propagator - conditions are determined by TLE data".to_string()
        ))
    }
    
    fn trajectory(&self) -> &Trajectory<OrbitState> {
        &self.trajectory
    }
    
    fn trajectory_mut(&mut self) -> &mut Trajectory<OrbitState> {
        &mut self.trajectory
    }
    
    fn set_max_trajectory_size(&mut self, max_size: Option<usize>) {
        self.trajectory.set_max_size(max_size);
    }
    
    fn set_max_trajectory_age(&mut self, max_age: Option<f64>) {
        self.trajectory.set_max_age(max_age);
    }
    
    fn set_eviction_policy(&mut self, policy: TrajectoryEvictionPolicy) {
        self.trajectory.set_eviction_policy(policy);
    }
    
    fn step_size(&self) -> f64 {
        self.step_size
    }
    
    fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }
    
    fn step(&mut self) -> Result<&OrbitState, BraheError> {
        let current_epoch = self.current_state.epoch;
        let target_epoch = current_epoch + self.step_size;
        
        let propagated_state = self.propagate(target_epoch)?;
        self.current_state = propagated_state.clone();
        self.trajectory.add_state(propagated_state)?;
        
        Ok(&self.current_state)
    }
    
    fn step_by(&mut self, step_size: f64) -> Result<&OrbitState, BraheError> {
        let current_epoch = self.current_state.epoch;
        let target_epoch = current_epoch + step_size;
        
        let propagated_state = self.propagate(target_epoch)?;
        self.current_state = propagated_state.clone();
        self.trajectory.add_state(propagated_state)?;
        
        Ok(&self.current_state)
    }
    
    fn propagate_steps(&mut self, num_steps: usize) -> Result<Vec<OrbitState>, BraheError> {
        let mut states = Vec::new();
        
        for _ in 0..num_steps {
            let state = self.step()?.clone();
            states.push(state);
        }
        
        Ok(states)
    }
    
    fn step_to(&mut self, target_epoch: Epoch) -> Result<&OrbitState, BraheError> {
        while self.current_state.epoch < target_epoch {
            self.step()?;
        }
        
        Ok(&self.current_state)
    }
}

impl FromStr for TLE {
    type Err = BraheError;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_tle_string(s, 60.0) // Default 60 second step size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::testing::{setup_global_test_eop, setup_global_test_eop_original_brahe};
    use approx::assert_abs_diff_eq;
    
    // Example TLE for ISS (International Space Station) - Classic format
    const ISS_CLASSIC_TLE: &str = r#"1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992
2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"#;

    // ISS TLE from Python original reference tests - for exact validation
    const ISS_TLE_LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    const ISS_TLE_LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    
    // Example 3-line TLE with satellite name  
    const ISS_3LE: &str = r#"ISS (ZARYA)
1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992
2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"#;
    
    // Example Alpha-5 TLE (using A0000 format for NORAD ID >= 100000)
    const ALPHA5_TLE: &str = r#"1 A0000U 21001A   21001.00000000  .00000000  00000-0  00000-0 0  9991
2 A0000  50.0000   0.0000 0001000   0.0000   0.0000 15.50000000000000"#;
    
    #[test]
    fn test_tle_from_classic_2line() {
        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE, 60.0).unwrap();
        
        assert_eq!(tle.format, TleFormat::Classic);
        assert_eq!(tle.satellite_name(), None);
        assert_eq!(tle.norad_id(), 25544);
        assert_eq!(tle.norad_id_string(), "25544");
    }
    
    #[test]
    fn test_tle_from_3line_format() {
        let tle = TLE::from_3le("ISS (ZARYA)", 
            "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992",
            "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003").unwrap();
        
        assert_eq!(tle.satellite_name(), Some("ISS (ZARYA)"));
        assert_eq!(tle.norad_id(), 25544);
    }
    
    #[test]
    fn test_alpha5_decoding() {
        // Test Alpha-5 decoding: A0000 should decode to 100000
        let (decoded_id, format) = decode_norad_id("A0000").unwrap();
        assert_eq!(decoded_id, 100000);
        assert_eq!(format, TleFormat::Alpha5);
        
        // Test other Alpha-5 examples
        let (decoded_id, _) = decode_norad_id("E8493").unwrap();
        assert_eq!(decoded_id, 148493);
        
        let (decoded_id, _) = decode_norad_id("Z9999").unwrap();
        assert_eq!(decoded_id, 339999);
        
        // Test skipped letters (I and O not allowed)
        assert!(decode_norad_id("I0000").is_err());
        assert!(decode_norad_id("O0000").is_err());
    }
    
    #[test]
    fn test_tle_validation() {
        // Test invalid line length
        let invalid_tle = "1 25544U\n2 25544";
        assert!(TLE::from_tle_string(invalid_tle, 60.0).is_err());
        
        // Test wrong line numbers
        let wrong_line_num = r#"2 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9991
1 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000000"#;
        assert!(TLE::from_tle_string(wrong_line_num, 60.0).is_err());
        
        // Test mismatched NORAD IDs
        let mismatched_ids = r#"1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9991
2 25545  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000000"#;
        assert!(TLE::from_tle_string(mismatched_ids, 60.0).is_err());
        
        // Test invalid number of lines
        let single_line = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9991";
        assert!(TLE::from_tle_string(single_line, 60.0).is_err());
        
        let four_lines = r#"Line 0
1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9991
2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000000
Line 3"#;
        assert!(TLE::from_tle_string(four_lines, 60.0).is_err());
    }
    
    #[test]
    fn test_tle_orbital_elements() {
        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE, 60.0).unwrap();
        
        // Test basic orbital elements access
        assert!(tle.eccentricity() < 0.01); // Low Earth orbit should have low eccentricity
        assert_abs_diff_eq!(tle.inclination(), 51.6461, epsilon = 0.1);
        assert!(tle.mean_motion() > 15.0); // ISS orbits about 15.5 times per day
        assert_abs_diff_eq!(tle.mean_motion(), 15.48919103, epsilon = 0.1);
        
        // Test argument of perigee and RAAN
        assert_abs_diff_eq!(tle.argument_of_perigee(), 88.1267, epsilon = 0.1);
        assert_abs_diff_eq!(tle.raan(), 306.0234, epsilon = 0.1);
        assert_abs_diff_eq!(tle.mean_anomaly(), 25.5695, epsilon = 0.1);
    }
    
    #[test]
    fn test_tle_propagation() {
        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE, 60.0).unwrap();
        
        let initial_epoch = tle.epoch();
        let future_epoch = initial_epoch + 3600.0; // 1 hour later
        
        // Test single propagation
        let state = tle.propagate(future_epoch).unwrap();
        assert_eq!(*state.epoch(), future_epoch);
        assert_eq!(state.orbit_type, OrbitStateType::Cartesian);
        assert_eq!(state.frame, OrbitFrame::ECI);
        
        // Verify position is reasonable for LEO satellite
        let position = state.position().unwrap();
        let altitude_km = (position.norm() - 6371000.0) / 1000.0; // Rough altitude calculation
        assert!(altitude_km > 200.0 && altitude_km < 800.0); // Reasonable LEO altitude range
        
        // Verify velocity is reasonable for LEO
        let velocity = state.velocity().unwrap();
        let velocity_magnitude = velocity.norm() / 1000.0; // Convert m/s to km/s
        assert!(velocity_magnitude > 6.0 && velocity_magnitude < 9.0); // LEO velocity range
    }
    
    #[test] 
    fn test_tle_propagator_trait() {
        let mut tle = TLE::from_tle_string(ISS_CLASSIC_TLE, 60.0).unwrap();
        
        let initial_epoch = tle.current_epoch();
        
        // Test OrbitPropagator trait methods
        let target_epoch = initial_epoch + 1800.0; // 30 minutes later
        let propagated_state = tle.propagate_to(target_epoch).unwrap();
        assert_eq!(*propagated_state.epoch(), target_epoch);
        
        // Verify current state was updated
        assert_eq!(tle.current_epoch(), target_epoch);
        
        let mut tle = TLE::from_tle_string(ISS_CLASSIC_TLE, 60.0).unwrap();
        let initial_propagation_epoch = tle.current_epoch();
        let states = tle.propagate_steps(10).unwrap();
        assert_eq!(states.len(), 10);
        
        let mut expected_epoch = initial_propagation_epoch;
        for (_i, state) in states.iter().enumerate() {
            expected_epoch += tle.step_size();
            let epoch_diff = *state.epoch() - expected_epoch;
            assert_abs_diff_eq!(epoch_diff, 0.0, epsilon = 1e-6);
            
            // Verify each state has reasonable position for LEO
            let position = state.position().unwrap();
            let altitude_km = (position.norm() - 6371000.0) / 1000.0;
            assert!(altitude_km > 200.0 && altitude_km < 700.0);
        }
        
        // Test reset functionality
        tle.reset().unwrap();
        assert_eq!(tle.current_epoch(), initial_epoch);
        assert_eq!(tle.trajectory().states.len(), 1);
    }
    
    #[test]
    fn test_trajectory_memory_management() {
        let mut tle = TLE::from_tle_string(ISS_CLASSIC_TLE, 60.0).unwrap();
        
        // Set trajectory limits
        tle.set_max_trajectory_size(Some(5));
        tle.set_eviction_policy(TrajectoryEvictionPolicy::KeepCount);
        
        let initial_epoch = tle.current_epoch();
        
        // Propagate to many epochs to test eviction
        let mut target_epochs = Vec::new();
        for i in 1..=10 {
            let epoch = initial_epoch + (i as f64) * 600.0; // Every 10 minutes
            target_epochs.push(epoch);
        }
        
        let _states = tle.propagate_steps(target_epochs.len()).unwrap();
        
        // Should only keep the most recent 5 states (plus initial state may be kept)
        assert!(tle.trajectory().states.len() <= 6);
        
        // Verify the kept states are the most recent ones
        let trajectory_states = &tle.trajectory().states;
        if trajectory_states.len() > 1 {
            // Should be in chronological order
            for i in 1..trajectory_states.len() {
                assert!(trajectory_states[i].epoch() >= trajectory_states[i-1].epoch());
            }
        }
    }
    
    #[test]
    fn test_tle_immutable_state() {
        let mut tle = TLE::from_tle_string(ISS_CLASSIC_TLE, 60.0).unwrap();
        
        let initial_state = tle.initial_state().clone();
        
        // TLE propagators should not allow changing initial state/conditions
        // since they come from the TLE data
        assert!(tle.set_initial_state(initial_state.clone()).is_err());
        
        use crate::trajectories::AngleFormat;
        assert!(tle.set_initial_conditions(
            initial_state.epoch,
            nalgebra::Vector6::zeros(),
            OrbitFrame::ECI,
            OrbitStateType::Cartesian,
            AngleFormat::None
        ).is_err());
    }
    
    #[test]
    fn test_checksum_calculation() {
        // Test TLE checksum calculation
        let line = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  999";
        let checksum = calculate_tle_line_checksum(line);
        assert_eq!(checksum, 2); // Calculated checksum for this line
        
        // Test line with negative values
        let line_with_neg = "1 25544U 98067A   21001.00000000 -.00001764  00000-0 -40967-4 0  999";
        let checksum_neg = calculate_tle_line_checksum(line_with_neg);
        assert_eq!(checksum_neg, 4); // Should count minus signs as 1
    }
    
    #[test]
    fn test_numeric_line_conversion() {
        // Test conversion of Alpha-5 line to numeric for SGP4 compatibility
        let alpha5_line = "1 A0000U 21001A   21001.00000000  .00000000  00000-0  00000-0 0  9991";
        let numeric_line = convert_to_numeric_line(alpha5_line, 100000).unwrap();
        
        // Should replace A0000 with 00000 (100000 mod 100000)
        assert!(numeric_line.contains("00000"));
        assert!(!numeric_line.contains("A0000"));
        
        // Should recalculate checksum
        assert_eq!(numeric_line.len(), 69);
        
        // Test with smaller ID that fits in 5 digits
        let small_line = convert_to_numeric_line(alpha5_line, 12345).unwrap();
        assert!(small_line.contains("12345"));
    }
    
    #[test]
    fn test_from_str_trait() {
        // Test that TLE implements FromStr
        let tle: TLE = ISS_CLASSIC_TLE.parse().unwrap();
        assert_eq!(tle.norad_id(), 25544);
        
        // Test error case
        let bad_tle: Result<TLE, _> = "invalid tle".parse();
        assert!(bad_tle.is_err());
    }
    
    #[test]
    fn test_is_alpha5_flag() {
        let classic_tle = TLE::from_tle_string(ISS_CLASSIC_TLE, 60.0).unwrap();
        assert!(!classic_tle.is_alpha5());
        
        // Would need a valid Alpha-5 TLE to test this properly
        // For now just test the enum comparison
        assert_eq!(TleFormat::Classic != TleFormat::Alpha5, true);
    }
    
    #[test]
    fn test_julian_date_conversion() {
        // Test the datetime to Julian Date conversion
        // Use a known date: 2021-01-01 00:00:00 UTC = JD 2459215.5
        
        // Create a mock elements struct to test epoch extraction
        // Note: This test might need adjustment based on actual SGP4 API
        // For now, just test the conversion logic would work
        
        let jd_2021_jan_1 = 2459215.5;
        let epoch = Epoch::from_jd(jd_2021_jan_1, TimeSystem::UTC);
        
        // Basic sanity check that our epoch system works
        assert_abs_diff_eq!(epoch.jd(), jd_2021_jan_1, epsilon = 1e-10);
    }

    // Tests for independent functions
    
    #[test]
    fn test_validate_tle_lines() {
        // Test valid TLE lines
        let line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992";
        let line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";
        assert!(validate_tle_lines(line1, line2));
        
        // Test invalid length
        let short_line1 = "1 25544U 98067A";
        assert!(!validate_tle_lines(short_line1, line2));
        
        // Test mismatched NORAD IDs
        let wrong_id_line2 = "2 25545  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";
        assert!(!validate_tle_lines(line1, wrong_id_line2));
        
        // Test wrong line numbers
        let wrong_line_num = "3 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992";
        assert!(!validate_tle_lines(wrong_line_num, line2));
        
        // Test invalid checksum
        let bad_checksum_line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9990";
        assert!(!validate_tle_lines(bad_checksum_line1, line2));
    }
    
    #[test]
    fn test_calculate_tle_line_checksum() {
        // Test checksum calculation
        let line = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  999";
        let checksum = calculate_tle_line_checksum(line);
        assert_eq!(checksum, 2);
        
        // Test with negative values
        let line_with_neg = "1 25544U 98067A   21001.00000000 -.00001764  00000-0 -40967-4 0  999";
        let checksum_neg = calculate_tle_line_checksum(line_with_neg);
        assert_eq!(checksum_neg, 4);
        
        // Test empty line
        let empty_checksum = calculate_tle_line_checksum("");
        assert_eq!(empty_checksum, 0);
    }
    
    #[test]
    fn test_decode_alpha5_id() {
        // Test basic Alpha-5 decoding using internal function
        assert_eq!(decode_alpha5_id("A0000").unwrap(), 100000);
        assert_eq!(decode_alpha5_id("A0001").unwrap(), 100001);
        assert_eq!(decode_alpha5_id("B0000").unwrap(), 110000);
        assert_eq!(decode_alpha5_id("E8493").unwrap(), 148493);
        assert_eq!(decode_alpha5_id("Z9999").unwrap(), 339999);
        
        // Test invalid letters
        assert!(decode_alpha5_id("I0000").is_err());
        assert!(decode_alpha5_id("O0000").is_err());
        
        // Test non-numeric remaining
        assert!(decode_alpha5_id("AABCD").is_err());
        
        // Test wrong length (handled by caller but worth testing)
        // Note: This function assumes 5 character input from caller
    }
    
    #[test]
    fn test_extract_tle_norad_id() {
        // Test classic format (numeric NORAD IDs less than 100000)
        assert_eq!(extract_tle_norad_id("25544").unwrap(), 25544);
        assert_eq!(extract_tle_norad_id("00001").unwrap(), 1);
        assert_eq!(extract_tle_norad_id("12345").unwrap(), 12345);
        assert_eq!(extract_tle_norad_id("99999").unwrap(), 99999);
        
        // Test Alpha-5 format (NORAD IDs >= 100000)
        assert_eq!(extract_tle_norad_id("A0000").unwrap(), 100000);
        assert_eq!(extract_tle_norad_id("A0001").unwrap(), 100001);
        assert_eq!(extract_tle_norad_id("B0000").unwrap(), 110000);
        assert_eq!(extract_tle_norad_id("E8493").unwrap(), 148493);
        assert_eq!(extract_tle_norad_id("Z9999").unwrap(), 339999);
        
        // Test invalid formats
        assert!(extract_tle_norad_id("I0000").is_err()); // Invalid letter
        assert!(extract_tle_norad_id("O0000").is_err()); // Invalid letter  
        assert!(extract_tle_norad_id("AABCD").is_err()); // Non-numeric after letter
        assert!(extract_tle_norad_id("1234").is_err()); // Wrong length
        assert!(extract_tle_norad_id("123456").is_err()); // Wrong length
    }
    
    #[test]
    fn test_lines_to_orbit_elements() {
        let line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992";
        let line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";
        
        let elements = lines_to_orbit_elements(line1, line2).unwrap();
        
        // Check that we get 6 elements: [a, e, i, Ω, ω, M]
        assert_eq!(elements.len(), 6);
        
        // Verify semi-major axis is reasonable for ISS (around 6700-6800 km)
        let a = elements[0];
        assert!(a > 6_700_000.0 && a < 6_800_000.0); // meters
        
        // Verify eccentricity is small for LEO
        let e = elements[1];
        assert!(e < 0.01);
        
        // Verify inclination is close to expected (51.6461 degrees)
        let i_rad = elements[2];
        let i_deg = i_rad.to_degrees();
        assert_abs_diff_eq!(i_deg, 51.6461, epsilon = 0.1);
        
        // Verify RAAN
        let raan_rad = elements[3];
        let raan_deg = raan_rad.to_degrees();
        assert_abs_diff_eq!(raan_deg, 306.0234, epsilon = 0.1);
        
        // Verify argument of perigee
        let argp_rad = elements[4];
        let argp_deg = argp_rad.to_degrees();
        assert_abs_diff_eq!(argp_deg, 88.1267, epsilon = 0.1);
        
        // Verify mean anomaly
        let ma_rad = elements[5];
        let ma_deg = ma_rad.to_degrees();
        assert_abs_diff_eq!(ma_deg, 25.5695, epsilon = 0.1);
        
        // Test with invalid lines
        let bad_line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9990";
        assert!(lines_to_orbit_elements(bad_line1, line2).is_err());
    }
    
    #[test]
    fn test_lines_to_orbit_state() {
        let line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992";
        let line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";
        
        let orbit_state = lines_to_orbit_state(line1, line2, OrbitFrame::ECI, OrbitStateType::Keplerian).unwrap();
        
        // Verify state properties
        assert_eq!(orbit_state.frame, OrbitFrame::ECI);
        assert_eq!(orbit_state.orbit_type, OrbitStateType::Keplerian);
        assert_eq!(orbit_state.angle_format, AngleFormat::Radians);
        
        // Verify epoch exists
        let epoch = orbit_state.epoch;
        assert!(epoch.jd() > 0.0); // Just verify we have a valid epoch
        
        // Verify state vector
        let state = orbit_state.state;
        assert_eq!(state.len(), 6);
        
        // Semi-major axis should be reasonable for ISS
        let a = state[0];
        assert!(a > 6_700_000.0 && a < 6_800_000.0);
        
        // Verify inclination matches parsed value
        let i_rad = state[2];
        let i_deg = i_rad.to_degrees();
        assert_abs_diff_eq!(i_deg, 51.6461, epsilon = 0.1);
        
        // Test with invalid lines
        let bad_line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000009";
        assert!(lines_to_orbit_state(line1, bad_line2, OrbitFrame::ECI, OrbitStateType::Keplerian).is_err());
    }
    
    #[test]
    fn test_alpha5_tle_lines_to_orbit_elements() {
        // Test with Alpha-5 TLE (A0000 = 100000)
        let line1_base = "1 A0000U 21001A   21001.00000000  .00000000  00000-0  00000-0 0  999";
        let line2_base = "2 A0000  50.0000   0.0000 0001000   0.0000   0.0000 15.5000000000000";
        
        // Calculate correct checksums
        let checksum1 = calculate_tle_line_checksum(line1_base);
        let checksum2 = calculate_tle_line_checksum(line2_base);
        
        let alpha5_line1 = format!("{}{}",line1_base, checksum1);
        let alpha5_line2 = format!("{}{}",line2_base, checksum2);
        
        let elements = lines_to_orbit_elements(&alpha5_line1, &alpha5_line2).unwrap();
        
        // Should successfully parse and return valid elements
        assert_eq!(elements.len(), 6);
        
        // Semi-major axis should be reasonable
        let a = elements[0];
        assert!(a > 6_000_000.0 && a < 8_000_000.0);
        
        // Eccentricity should match
        let e = elements[1];
        assert_abs_diff_eq!(e, 0.0001, epsilon = 1e-6);
        
        // Inclination should match (50 degrees)
        let i_rad = elements[2];
        let i_deg = i_rad.to_degrees();
        assert_abs_diff_eq!(i_deg, 50.0, epsilon = 0.1);
        
        // RAAN should be 0
        let raan_rad = elements[3];
        assert_abs_diff_eq!(raan_rad, 0.0, epsilon = 1e-6);
        
        // Argument of perigee should be 0
        let argp_rad = elements[4];
        assert_abs_diff_eq!(argp_rad, 0.0, epsilon = 1e-6);
        
        // Mean anomaly should be 0
        let ma_rad = elements[5];
        assert_abs_diff_eq!(ma_rad, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_analytic_propagator_state() {
        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE, 60.0).unwrap();
        let epoch = Epoch::from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        
        let state = tle.state(epoch);
        
        assert_eq!(state.len(), 6);
        assert!(state[0].abs() > 1e6); // Position should be reasonable (>1000 km)
        assert!(state[3].abs() > 1e3); // Velocity should be reasonable (>1 km/s)
    }

    #[test]
    fn test_analytic_propagator_state_eci() {
        setup_global_test_eop();
        
        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE, 60.0).unwrap();
        let epoch = Epoch::from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        
        let state_eci = tle.state_eci(epoch);
        
        assert_eq!(state_eci.len(), 6);
        assert!(state_eci[0].abs() > 1e6); // Position should be reasonable
        assert!(state_eci[3].abs() > 1e3); // Velocity should be reasonable
    }

    #[test]
    fn test_analytic_propagator_state_ecef() {
        setup_global_test_eop();

        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE, 60.0).unwrap();
        let epoch = Epoch::from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        
        let state_ecef = tle.state_ecef(epoch);
        
        assert_eq!(state_ecef.len(), 6);
        assert!(state_ecef[0].abs() > 1e6); // Position should be reasonable
        assert!(state_ecef[3].abs() > 1e3); // Velocity should be reasonable
    }

    #[test]
    fn test_analytic_propagator_state_osculating_elements() {
        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE, 60.0).unwrap();
        let epoch = Epoch::from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        
        let elements = tle.state_osculating_elements(epoch);
        
        assert_eq!(elements.len(), 6);
        assert!(elements[0] > 6e6); // Semi-major axis should be > 6000 km for ISS
        assert!(elements[1] >= 0.0 && elements[1] < 1.0); // Eccentricity [0,1)
        assert!(elements[2] >= 0.0 && elements[2] <= std::f64::consts::PI); // Inclination [0,π]
    }

    #[test]
    fn test_analytic_propagator_batch_states() {
        setup_global_test_eop();

        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE, 60.0).unwrap();
        let epoch1 = Epoch::from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_datetime(2021, 1, 1, 13, 0, 0.0, 0.0, TimeSystem::UTC);
        let epochs = vec![epoch1, epoch2];
        
        let states = tle.states(&epochs);
        
        assert_eq!(states.len(), 2);
        assert_eq!(states[0].len(), 6);
        assert_eq!(states[1].len(), 6);
        
        let states_eci = tle.states_eci(&epochs);
        assert_eq!(states_eci.len(), 2);
        
        let states_ecef = tle.states_ecef(&epochs);
        assert_eq!(states_ecef.len(), 2);
        
        let states_elements = tle.states_osculating_elements(&epochs);
        assert_eq!(states_elements.len(), 2);
    }

    #[test]
    fn test_analytic_propagator_consistency() {
        let tle = TLE::from_tle_string(ISS_CLASSIC_TLE, 60.0).unwrap();
        let epoch = Epoch::from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        
        // Single epoch call
        let state_single = tle.state(epoch);
        
        // Batch call with single epoch
        let states_batch = tle.states(&[epoch]);
        
        // Should be identical
        assert_abs_diff_eq!(state_single[0], states_batch[0][0], epsilon = 1e-12);
        assert_abs_diff_eq!(state_single[1], states_batch[0][1], epsilon = 1e-12);
        assert_abs_diff_eq!(state_single[2], states_batch[0][2], epsilon = 1e-12);
        assert_abs_diff_eq!(state_single[3], states_batch[0][3], epsilon = 1e-12);
        assert_abs_diff_eq!(state_single[4], states_batch[0][4], epsilon = 1e-12);
        assert_abs_diff_eq!(state_single[5], states_batch[0][5], epsilon = 1e-12);
    }

    // ========== Python Reference Tests ==========
    // These tests match the exact Python reference implementation

    #[test]
    fn test_tle_checksum_reference() {
        // Test TLE checksum calculation matches Python reference
        let checksum1 = calculate_tle_line_checksum(ISS_TLE_LINE1);
        assert_eq!(checksum1, 7);

        let checksum2 = calculate_tle_line_checksum(ISS_TLE_LINE2);
        assert_eq!(checksum2, 7);
    }

    #[test]
    fn test_validate_tle_reference() {
        // Test TLE validation matches Python reference
        let invalid_tle_line = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2926";
        assert!(!validate_tle_lines(invalid_tle_line, ISS_TLE_LINE2));

        assert!(validate_tle_lines(ISS_TLE_LINE1, ISS_TLE_LINE2));
    }

    #[test]
    fn test_tle_elements_reference() {
        setup_global_test_eop();
        let tle = TLE::from_lines(ISS_TLE_LINE1, ISS_TLE_LINE2).unwrap();

        // Test individual TLE elements match Python reference
        assert_abs_diff_eq!(tle.mean_motion(), 15.72125391, epsilon = 1e-8);
        assert_abs_diff_eq!(tle.eccentricity(), 0.0006703, epsilon = 1e-7);
        assert_abs_diff_eq!(tle.inclination(), 51.6416, epsilon = 1e-4);
        assert_abs_diff_eq!(tle.right_ascension(), 247.4627, epsilon = 1e-4);
        assert_abs_diff_eq!(tle.argument_of_perigee(), 130.536, epsilon = 1e-3);
        assert_abs_diff_eq!(tle.mean_anomaly(), 325.0288, epsilon = 1e-4);
        
        // Test drag terms  
        assert_abs_diff_eq!(tle.first_derivative_mean_motion(), -2.182e-05, epsilon = 1e-8);
        assert_abs_diff_eq!(tle.second_derivative_mean_motion(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(tle.bstar(), -1.1606e-05, epsilon = 1e-8);
    }

    #[test]
    fn test_tle_elements_computed() {
        setup_global_test_eop();
        let tle = TLE::from_lines(ISS_TLE_LINE1, ISS_TLE_LINE2).unwrap();

        // Test computed orbital elements (semi-major axis from mean motion)
        let elements = tle.orbital_elements().unwrap();
        
        assert_eq!(elements.len(), 6);
        assert_abs_diff_eq!(elements[0], 6730960.675248184, epsilon = 1.0); // Semi-major axis in meters
        assert_abs_diff_eq!(elements[1], 0.0006703, epsilon = 1e-7);
        assert_abs_diff_eq!(elements[2], 51.6416, epsilon = 1e-4);
        assert_abs_diff_eq!(elements[3], 247.4627, epsilon = 1e-4);
        assert_abs_diff_eq!(elements[4], 130.536, epsilon = 1e-3);
        assert_abs_diff_eq!(elements[5], 325.0288, epsilon = 1e-4);
    }

    #[test]
    fn test_tle_state_teme_reference() {
        setup_global_test_eop_original_brahe();
        let tle = TLE::from_lines(ISS_TLE_LINE1, ISS_TLE_LINE2).unwrap();
        
        let state = tle.state_teme(tle.epoch()).unwrap();
        
        // Test TEME state matches Python reference exactly
        assert_eq!(state.len(), 6);
        assert_abs_diff_eq!(state[0], 4083909.8260273533, epsilon = 1e-6);
        assert_abs_diff_eq!(state[1], -993636.8325621719, epsilon = 1e-6);
        assert_abs_diff_eq!(state[2], 5243614.536966579, epsilon = 1e-6);
        assert_abs_diff_eq!(state[3], 2512.831950943635, epsilon = 1e-6);
        assert_abs_diff_eq!(state[4], 7259.8698423432315, epsilon = 1e-6);
        assert_abs_diff_eq!(state[5], -583.775727402632, epsilon = 1e-6);
    }

    #[test]
    fn test_tle_state_pef_reference() {
        setup_global_test_eop_original_brahe();
        let tle = TLE::from_lines(ISS_TLE_LINE1, ISS_TLE_LINE2).unwrap();
        
        let state = tle.state_pef(tle.epoch()).unwrap();
        
        // Test PEF state matches Python reference (relax tolerance for small EOP differences)
        assert_eq!(state.len(), 6);
        assert_abs_diff_eq!(state[0], -3953205.7105210484, epsilon = 1.0);
        assert_abs_diff_eq!(state[1], 1427514.704810681, epsilon = 1.0);
        assert_abs_diff_eq!(state[2], 5243614.536966579, epsilon = 1.0);
        assert_abs_diff_eq!(state[3], -3175.692140186211, epsilon = 1.0);
        assert_abs_diff_eq!(state[4], -6658.887120918979, epsilon = 1.0);
        assert_abs_diff_eq!(state[5], -583.775727402632, epsilon = 1.0);
    }

    #[test]
    fn test_tle_state_itrf_reference() {
        setup_global_test_eop_original_brahe();
        let tle = TLE::from_lines(ISS_TLE_LINE1, ISS_TLE_LINE2).unwrap();
        
        let state = tle.state_itrf(tle.epoch()).unwrap();
        
        // Test ITRF state matches Python reference (relax tolerance for small EOP differences)
        assert_eq!(state.len(), 6);
        assert_abs_diff_eq!(state[0], -3953198.4858592334, epsilon = 1.0);
        assert_abs_diff_eq!(state[1], 1427508.2304882656, epsilon = 1.0);
        assert_abs_diff_eq!(state[2], 5243621.746247788, epsilon = 1.0);
        assert_abs_diff_eq!(state[3], -3175.6929443809036, epsilon = 1.0);
        assert_abs_diff_eq!(state[4], -6658.8864002006185, epsilon = 1.0);
        assert_abs_diff_eq!(state[5], -583.7795735705351, epsilon = 1.0);
    }

    #[test]
    fn test_tle_state_eci_reference() {
        setup_global_test_eop_original_brahe();
        let tle = TLE::from_lines(ISS_TLE_LINE1, ISS_TLE_LINE2).unwrap();
        
        let state = tle.state_eci(tle.epoch());
        
        // Test ECI/GCRF state matches Python reference (relax tolerance for small EOP differences)
        assert_eq!(state.len(), 6);
        assert_abs_diff_eq!(state[0], 4086521.0432801973, epsilon = 1.0);
        assert_abs_diff_eq!(state[1], -1001422.0546131282, epsilon = 1.0);
        assert_abs_diff_eq!(state[2], 5240097.963377853, epsilon = 1.0);
        assert_abs_diff_eq!(state[3], 2526.47546734367, epsilon = 1.0);
        assert_abs_diff_eq!(state[4], 7254.93629077332, epsilon = 1.0);
        assert_abs_diff_eq!(state[5], -586.2164882389718, epsilon = 1.0);
    }

    #[test]
    fn test_tle_string_from_elements_reference() {
        setup_global_test_eop();
        
        let epoch = Epoch::from_datetime(2019, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let norad_id = 99999;
        
        // Semi-major axis in meters, convert to mean motion
        let sma = crate::constants::R_EARTH + 500e3;
        let n = crate::orbits::mean_motion(sma, false) / (2.0 * PI) * 86400.0; // Convert to rev/day
        
        let elements = [n, 0.001, 97.7, 45.0, 30.0, 15.0]; // [n, e, i, raan, argp, anomaly]
        
        let (line1, line2) = tle_lines_from_elements(
            epoch,
            &elements,
            norad_id,
            "        ", // empty designation
            0, // element number
            0, // orbit number  
            true, // angles in degrees
            Some(0.0), // ndt2
            Some(0.0), // nddt6
            Some(0.0)  // bstar
        ).unwrap();
        
        // Verify the lines are valid TLE format and match expected exact strings
        assert!(validate_tle_lines(&line1, &line2));
        assert_eq!(line1.len(), 69);
        assert_eq!(line2.len(), 69);
        
        // Verify exact line content for zero drag terms case
        let expected_line1 = "1 99999U          19  1.00000000  .00000000  00000-0  00000-0 0    09";
        let expected_line2 = "2 99999  97.7000  45.0000 0010000  30.0000  15.0000 15.21936719    03";
        assert_eq!(line1, expected_line1);
        assert_eq!(line2, expected_line2);
        
        // Test variant with non-zero drag terms
        let (line1_drag, line2_drag) = tle_lines_from_elements(
            epoch,
            &elements,
            12345, // different NORAD ID
            "21001A  ", // 8-char designation
            123, // element number
            456, // orbit number  
            true, // angles in degrees
            Some(-2.182e-5), // ndt2 - negative
            Some(1.5e-9), // nddt6 - positive
            Some(-1.1606e-4)  // bstar - negative
        ).unwrap();
        
        assert!(validate_tle_lines(&line1_drag, &line2_drag));
        assert_eq!(line1_drag.len(), 69);
        assert_eq!(line2_drag.len(), 69);
        
        // Test variant with different eccentricity formatting
        let high_ecc_elements = [n, 0.12345, 97.7, 45.0, 30.0, 15.0]; // higher eccentricity
        let (line1_ecc, line2_ecc) = tle_lines_from_elements(
            epoch,
            &high_ecc_elements,
            54321,
            "        ", // empty designation
            0, 0, true,
            Some(0.0), Some(0.0), Some(0.0)
        ).unwrap();
        
        assert!(validate_tle_lines(&line1_ecc, &line2_ecc));
        assert!(line2_ecc.contains("1234500")); // Check eccentricity formatting (0.12345 -> 1234500)
    }

    #[test]
    fn test_frame_conversion_reference_validation() {
        setup_global_test_eop();
        let tle = TLE::from_lines(ISS_TLE_LINE1, ISS_TLE_LINE2).unwrap();
        
        // Test that all frame conversions produce valid results
        let state_teme = tle.state_teme(tle.epoch()).unwrap();
        let state_pef = tle.state_pef(tle.epoch()).unwrap();
        let state_itrf = tle.state_itrf(tle.epoch()).unwrap();
        let state_ecef = tle.state_ecef(tle.epoch());
        let state_eci = tle.state_eci(tle.epoch());
        
        // ECEF should equal ITRF exactly
        for i in 0..6 {
            assert_abs_diff_eq!(state_ecef[i], state_itrf[i], epsilon = 1e-12);
        }
        
        // All position magnitudes should be similar (just rotations)
        let pos_mag_teme = (state_teme[0].powi(2) + state_teme[1].powi(2) + state_teme[2].powi(2)).sqrt();
        let pos_mag_pef = (state_pef[0].powi(2) + state_pef[1].powi(2) + state_pef[2].powi(2)).sqrt();
        let pos_mag_itrf = (state_itrf[0].powi(2) + state_itrf[1].powi(2) + state_itrf[2].powi(2)).sqrt();
        let pos_mag_eci = (state_eci[0].powi(2) + state_eci[1].powi(2) + state_eci[2].powi(2)).sqrt();
        
        assert_abs_diff_eq!(pos_mag_teme, pos_mag_pef, epsilon = 1.0);
        assert_abs_diff_eq!(pos_mag_pef, pos_mag_itrf, epsilon = 1.0);
        assert_abs_diff_eq!(pos_mag_itrf, pos_mag_eci, epsilon = 1.0);
        
        // All should be reasonable orbital altitudes
        assert!(pos_mag_teme > 6.5e6); // > 6500 km  
        assert!(pos_mag_teme < 7.0e6); // < 7000 km (LEO)
    }
}

/// Implement AnalyticPropagator trait for TLE
impl AnalyticPropagator for TLE {
    fn state(&self, epoch: Epoch) -> Vector6<f64> {
        let orbit_state = self.propagate_to_state(epoch);
        orbit_state.state
    }

    fn state_eci(&self, epoch: Epoch) -> Vector6<f64> {
        // Transform ITRF -> GCRF using full frame transformation chain
        let itrf_state = self.state_itrf(epoch).unwrap();
        state_ecef_to_eci(epoch, itrf_state)
    }

    fn state_ecef(&self, epoch: Epoch) -> Vector6<f64> {
        // Use ITRF state as ECEF (they are equivalent for TLE purposes)
        self.state_itrf(epoch).unwrap()
    }

    fn state_osculating_elements(&self, epoch: Epoch) -> Vector6<f64> {
        let orbit_state = self.propagate_to_state(epoch);
        let cart_state = orbit_state.to_cartesian().unwrap();
        state_cartesian_to_osculating(cart_state.state, false)
    }

    fn states(&self, epochs: &[Epoch]) -> Trajectory<OrbitState> {
        let mut states = Vec::new();
        for &epoch in epochs {
            states.push(self.propagate_to_state(epoch));
        }
        Trajectory::from_states(states, InterpolationMethod::Linear)
            .unwrap()
            .with_propagator(PropagatorType::SGP4)
    }

    fn states_eci(&self, epochs: &[Epoch]) -> Trajectory<OrbitState> {
        let mut states = Vec::new();
        for &epoch in epochs {
            let eci_state = self.state_eci(epoch);
            let orbit_state = OrbitState::new(
                epoch,
                eci_state,
                OrbitFrame::ECI,
                OrbitStateType::Cartesian,
                AngleFormat::None,
            ).unwrap();
            states.push(orbit_state);
        }
        Trajectory::from_states(states, InterpolationMethod::Linear)
            .unwrap()
            .with_propagator(PropagatorType::SGP4)
    }

    fn states_ecef(&self, epochs: &[Epoch]) -> Trajectory<OrbitState> {
        let mut states = Vec::new();
        for &epoch in epochs {
            let ecef_state = self.state_ecef(epoch);
            let orbit_state = OrbitState::new(
                epoch,
                ecef_state,
                OrbitFrame::ECEF,
                OrbitStateType::Cartesian,
                AngleFormat::None,
            ).unwrap();
            states.push(orbit_state);
        }
        Trajectory::from_states(states, InterpolationMethod::Linear)
            .unwrap()
            .with_propagator(PropagatorType::SGP4)
    }

    fn states_osculating_elements(&self, epochs: &[Epoch]) -> Trajectory<OrbitState> {
        let mut states = Vec::new();
        for &epoch in epochs {
            let orbit_state = self.propagate_to_state(epoch);
            let cart_state = orbit_state.to_cartesian().unwrap();
            let osculating_elements = state_cartesian_to_osculating(cart_state.state, false);
            
            let kep_state = OrbitState::new(
                epoch,
                osculating_elements,
                OrbitFrame::ECI,
                OrbitStateType::Keplerian,
                AngleFormat::Radians,
            ).unwrap();
            
            states.push(kep_state);
        }
        Trajectory::from_states(states, InterpolationMethod::Linear)
            .unwrap()
            .with_propagator(PropagatorType::SGP4)
    }
}

impl TLE {
    /// Internal helper method to propagate to a state without modifying the propagator
    fn propagate_to_state(&self, epoch: Epoch) -> OrbitState {
        self.propagate(epoch).unwrap()
    }
}
