use crate::constants::GM_EARTH;
#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
use crate::constants::RADIANS;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;
use nalgebra::Vector6;
use std::f64::consts::PI;

/// TLE format type
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TleFormat {
    /// Classic 2-line TLE format with numeric NORAD ID
    Classic,
    /// Alpha-5 TLE format with alphanumeric NORAD ID (first digit replaced with letter for IDs >= 100000)
    Alpha5,
}

/// Calculate checksum for a TLE line
///
/// # Arguments
/// * `line` - TLE line (without checksum digit)
///
/// # Returns
/// * `u32` - Checksum value (0-9)
pub fn calculate_tle_line_checksum(line: &str) -> u32 {
    let mut checksum = 0u32;

    for ch in line.chars().take(68) {
        match ch {
            '0'..='9' => checksum += ch.to_digit(10).unwrap_or(0),
            '-' => checksum += 1,
            _ => {} // Spaces, letters, and other characters contribute 0
        }
    }

    checksum % 10
}

/// Validate a single TLE line
///
/// # Arguments
/// * `line` - TLE line to validate
///
/// # Returns
/// * `bool` - True if line is valid
pub fn validate_tle_line(line: &str) -> bool {
    // Confirm Length
    if line.len() != 69 {
        return false;
    }

    // Confirm line starts with '1' or '2'
    if !line.starts_with('1') && !line.starts_with('2') {
        return false;
    }

    let expected_checksum = calculate_tle_line_checksum(line);
    let actual_checksum = line
        .chars()
        .nth(68)
        .and_then(|c| c.to_digit(10))
        .unwrap_or(10); // Invalid if not a digit

    expected_checksum == actual_checksum
}

/// Validate TLE line pair
///
/// # Arguments
/// * `line1` - First TLE line
/// * `line2` - Second TLE line
///
/// # Returns
/// * `bool` - True if both lines are valid and have matching NORAD IDs
pub fn validate_tle_lines(line1: &str, line2: &str) -> bool {
    if !validate_tle_line(line1) || !validate_tle_line(line2) {
        return false;
    }

    // Check line numbers
    if !line1.starts_with('1') || !line2.starts_with('2') {
        return false;
    }

    // Extract NORAD IDs and compare
    let norad1 = line1.get(2..7).unwrap_or("").trim();
    let norad2 = line2.get(2..7).unwrap_or("").trim();

    norad1 == norad2
}

/// Parse NORAD ID from string, handling both classic and Alpha-5 formats
///
/// # Arguments
/// * `norad_str` - NORAD ID string from TLE
///
/// # Returns
/// * `Result<u32, BraheError>` - Parsed numeric NORAD ID
pub fn parse_norad_id(norad_str: &str) -> Result<u32, BraheError> {
    // Confirm ID is right length
    if norad_str.len() > 5 {
        return Err(BraheError::Error(format!(
            "NORAD ID too long: {}. Expected 5 characters found {}",
            norad_str,
            norad_str.len()
        )));
    }
    if norad_str.len() < 5 {
        return Err(BraheError::Error(format!(
            "NORAD ID too short: {}. Expected 5 characters found {}",
            norad_str,
            norad_str.len()
        )));
    }

    let trimmed = norad_str.trim();

    if trimmed.is_empty() {
        return Err(BraheError::Error("Empty NORAD ID".to_string()));
    }

    let first_char = trimmed.chars().next().unwrap();

    if first_char.is_ascii_digit() {
        // Classic format - direct numeric conversion
        trimmed
            .parse::<u32>()
            .map_err(|_| BraheError::Error(format!("Invalid numeric NORAD ID: {}", trimmed)))
    } else if first_char.is_ascii_alphabetic() {
        // Alpha-5 format conversion
        norad_id_alpha5_to_numeric(trimmed)
    } else {
        Err(BraheError::Error(format!(
            "Invalid NORAD ID format: {}",
            trimmed
        )))
    }
}

/// Convert numeric NORAD ID to Alpha-5 format or pass through if in legacy range
///
/// # Arguments
/// * `norad_id` - Numeric NORAD ID (0-339999)
///
/// # Returns
/// * `Result<String, BraheError>` - For IDs 0-99999: numeric string (e.g., "42").
///   For IDs 100000-339999: Alpha-5 format ID (e.g., "A0001")
pub fn norad_id_numeric_to_alpha5(norad_id: u32) -> Result<String, BraheError> {
    // IDs 0-99999: Pass through as numeric string
    if norad_id < 100000 {
        return Ok(norad_id.to_string());
    }

    // IDs > 339999: Error
    if norad_id > 339999 {
        return Err(BraheError::Error(format!(
            "NORAD ID {} is out of valid range (0-339999)",
            norad_id
        )));
    }

    // IDs 100000-339999: Convert to Alpha-5
    let first_value = norad_id / 10000;
    let numeric_part = norad_id % 10000;

    // Convert first value to character (10=A, 11=B, ..., 33=Z, skip I and O)
    let first_char = match first_value {
        10..=17 => char::from(b'A' + (first_value - 10) as u8), // A-H
        18..=22 => char::from(b'J' + (first_value - 18) as u8), // J-N (skip I)
        23..=33 => char::from(b'P' + (first_value - 23) as u8), // P-Z (skip O)
        _ => {
            return Err(BraheError::Error(format!(
                "Invalid Alpha-5 first value: {}",
                first_value
            )));
        }
    };

    Ok(format!("{}{:04}", first_char, numeric_part))
}

/// Convert Alpha-5 NORAD ID to numeric format
///
/// # Arguments
/// * `alpha5_id` - Alpha-5 format ID (e.g., "A0001")
///
/// # Returns
/// * `Result<u32, BraheError>` - Converted numeric ID
pub fn norad_id_alpha5_to_numeric(alpha5_id: &str) -> Result<u32, BraheError> {
    if alpha5_id.len() != 5 {
        return Err(BraheError::Error(
            "Alpha-5 ID must be exactly 5 characters".to_string(),
        ));
    }

    let chars: Vec<char> = alpha5_id.chars().collect();
    let first_char = chars[0];

    // Convert first character (A=10, B=11, ..., Z=35, skip I and O)
    let first_value = match first_char {
        'A'..='H' => (first_char as u32) - ('A' as u32) + 10,
        'J'..='N' => (first_char as u32) - ('A' as u32) + 9, // Skip I
        'P'..='Z' => (first_char as u32) - ('A' as u32) + 8, // Skip I and O
        _ => {
            return Err(BraheError::Error(format!(
                "Invalid Alpha-5 first character: {}",
                first_char
            )));
        }
    };

    // Parse remaining 4 digits
    let remaining: String = chars[1..].iter().collect();
    let numeric_part = remaining
        .parse::<u32>()
        .map_err(|_| BraheError::Error(format!("Invalid Alpha-5 numeric part: {}", remaining)))?;

    if numeric_part > 9999 {
        return Err(BraheError::Error(
            "Alpha-5 numeric part cannot exceed 9999".to_string(),
        ));
    }

    Ok(first_value * 10000 + numeric_part)
}

/// Extract Epoch from TLE line 1
///
/// # Arguments
/// * `line1` - First TLE line
///
/// # Returns
/// * `Result<Epoch, BraheError>` - Extracted epoch
pub fn epoch_from_tle(line1: &str) -> Result<Epoch, BraheError> {
    if line1.len() < 32 {
        return Err(BraheError::Error(
            "TLE line 1 too short to extract epoch".to_string(),
        ));
    }

    let epoch_str = &line1[18..32];

    let year_2digit: u32 = epoch_str[0..2]
        .parse()
        .map_err(|_| BraheError::Error("Invalid year in TLE".to_string()))?;
    let year = if year_2digit < 57 {
        2000 + year_2digit
    } else {
        1900 + year_2digit
    };

    let day_of_year: f64 = epoch_str[2..]
        .parse()
        .map_err(|_| BraheError::Error("Invalid day of year in TLE".to_string()))?;

    Ok(Epoch::from_day_of_year(year, day_of_year, TimeSystem::UTC))
}

/// Extract Keplerian orbital elements from TLE lines
///
/// # Arguments
/// * `line1` - First TLE line
/// * `line2` - Second TLE line
///
/// # Returns
/// * `Result<(Vector6<f64>, Epoch), BraheError>` - epoch and Keplerian elements [a, e, i, Ω, ω, M]
///
/// Elements are returned in standard units:
/// - a: semi-major axis [m]
/// - e: eccentricity [dimensionless]
/// - i: inclination [degrees]
/// - Ω: right ascension of ascending node [degrees]
/// - ω: argument of periapsis [degrees]
/// - M: mean anomaly [degrees]
pub fn keplerian_elements_from_tle(
    line1: &str,
    line2: &str,
) -> Result<(Epoch, Vector6<f64>), BraheError> {
    if !validate_tle_lines(line1, line2) {
        return Err(BraheError::Error("Invalid TLE lines".to_string()));
    }

    let epoch = epoch_from_tle(line1)?;

    // Parse orbital elements from line 2
    let inclination: f64 = line2[8..16]
        .trim()
        .parse()
        .map_err(|_| BraheError::Error("Invalid inclination in TLE".to_string()))?;

    let raan: f64 = line2[17..25]
        .trim()
        .parse()
        .map_err(|_| BraheError::Error("Invalid RAAN in TLE".to_string()))?;

    // Parse eccentricity (format: "0.1234567" but stored as "1234567")
    let ecc_str = line2[26..33].trim();
    let eccentricity: f64 = format!("0.{}", ecc_str)
        .parse()
        .map_err(|_| BraheError::Error("Invalid eccentricity in TLE".to_string()))?;

    let arg_perigee: f64 = line2[34..42]
        .trim()
        .parse()
        .map_err(|_| BraheError::Error("Invalid argument of perigee in TLE".to_string()))?;

    let mean_anomaly: f64 = line2[43..51]
        .trim()
        .parse()
        .map_err(|_| BraheError::Error("Invalid mean anomaly in TLE".to_string()))?;

    let mean_motion_revs_per_day: f64 = line2[52..63]
        .trim()
        .parse()
        .map_err(|_| BraheError::Error("Invalid mean motion in TLE".to_string()))?;

    // Convert mean motion from rev/day to rad/s, then compute semi-major axis
    let mean_motion_rad_per_sec = mean_motion_revs_per_day * 2.0 * PI / 86400.0;
    let semi_major_axis =
        (GM_EARTH / (mean_motion_rad_per_sec * mean_motion_rad_per_sec)).powf(1.0 / 3.0);

    let elements = Vector6::new(
        semi_major_axis, // [m]
        eccentricity,    // [dimensionless]
        inclination,     // [degrees]
        raan,            // [degrees]
        arg_perigee,     // [degrees]
        mean_anomaly,    // [degrees]
    );

    Ok((epoch, elements))
}

/// Convert Keplerian elements to TLE format (simplified interface)
///
/// # Arguments
/// * `epoch` - Epoch of the elements
/// * `elements` - Keplerian elements [a (m), e, i (deg), Ω (deg), ω (deg), M (deg)]
/// * `norad_id` - NORAD catalog number as string (supports numeric and Alpha-5 format)
///
/// # Returns
/// * `Result<(String, String), BraheError>` - TLE line 1 and line 2
pub fn keplerian_elements_to_tle(
    epoch: &Epoch,
    elements: &Vector6<f64>,
    norad_id: &str,
) -> Result<(String, String), BraheError> {
    // Convert semi-major axis to mean motion (rev/day)
    let semi_major_axis = elements[0]; // meters
    let mean_motion_rad_per_sec =
        (GM_EARTH / (semi_major_axis * semi_major_axis * semi_major_axis)).sqrt();
    let mean_motion_revs_per_day = mean_motion_rad_per_sec * 86400.0 / (2.0 * PI);

    create_tle_lines(
        epoch,
        norad_id,
        'U', // Classification: Unclassified
        "",  // International designator: empty
        mean_motion_revs_per_day,
        elements[1], // eccentricity
        elements[2], // inclination (degrees)
        elements[3], // RAAN (degrees)
        elements[4], // argument of periapsis (degrees)
        elements[5], // mean anomaly (degrees)
        0.0,         // first_derivative: 0.0
        0.0,         // second_derivative: 0.0
        0.0,         // bstar: 0.0
        0,           // ephemeris_type: 0
        0,           // element_set_number: 0
        0,           // revolution_number: 0
    )
}

/// Create TLE lines from orbital elements and all required TLE metadata
///
/// # Arguments
/// * `epoch` - Epoch of the orbital elements
/// * `norad_id` - NORAD catalog number as string (supports numeric and Alpha-5 format)
/// * `classification` - Classification character ('U', 'C', or 'S')
/// * `intl_designator` - International designator (e.g., "98067A")
/// * `mean_motion` - Mean motion in revolutions per day
/// * `eccentricity` - Eccentricity (0 <= e < 1)
/// * `inclination` - Inclination in degrees
/// * `raan` - Right ascension of ascending node in degrees
/// * `arg_periapsis` - Argument of periapsis in degrees
/// * `mean_anomaly` - Mean anomaly in degrees
/// * `first_derivative` - First derivative of mean motion divided by 2
/// * `second_derivative` - Second derivative of mean motion divided by 6
/// * `bstar` - B* drag term
/// * `ephemeris_type` - Ephemeris type (usually 0)
/// * `element_set_number` - Element set number (0-9999)
/// * `revolution_number` - Revolution number at epoch (0-99999)
///
/// # Returns
/// * `Result<(String, String), BraheError>` - TLE line 1 and line 2
#[allow(clippy::too_many_arguments)]
pub fn create_tle_lines(
    epoch: &Epoch,
    norad_id: &str,
    classification: char,
    intl_designator: &str,
    mean_motion: f64,
    eccentricity: f64,
    inclination: f64,
    raan: f64,
    arg_periapsis: f64,
    mean_anomaly: f64,
    first_derivative: f64,
    second_derivative: f64,
    bstar: f64,
    ephemeris_type: u8,
    element_set_number: u16,
    revolution_number: u32,
) -> Result<(String, String), BraheError> {
    // Validate and parse NORAD ID
    let norad_id_trimmed = norad_id.trim();
    if norad_id_trimmed.len() > 5 {
        return Err(BraheError::Error(format!(
            "NORAD ID too long: {}. Expected 5 characters max",
            norad_id_trimmed
        )));
    }

    // Validate that it's a valid NORAD ID (numeric or Alpha-5)
    parse_norad_id(norad_id_trimmed)?;
    if !(0.0..1.0).contains(&eccentricity) {
        return Err(BraheError::Error(
            "Eccentricity must be in range [0, 1)".to_string(),
        ));
    }
    if mean_motion <= 0.0 {
        return Err(BraheError::Error(
            "Mean motion must be positive".to_string(),
        ));
    }
    if !(0.0..1.0).contains(&eccentricity) {
        return Err(BraheError::Error(
            "Eccentricity must be in range [0, 1)".to_string(),
        ));
    }
    if !(0.0..=180.0).contains(&inclination) {
        return Err(BraheError::Error(
            "Inclination must be in range [0, 180] degrees".to_string(),
        ));
    }
    if !(0.0..360.0).contains(&arg_periapsis) {
        return Err(BraheError::Error(
            "Argument of periapsis must be in range [0, 360) degrees".to_string(),
        ));
    }
    if !(0.0..360.0).contains(&mean_anomaly) {
        return Err(BraheError::Error(
            "Mean anomaly must be in range [0, 360) degrees".to_string(),
        ));
    }
    if element_set_number > 9999 {
        return Err(BraheError::Error(
            "Element set number cannot exceed 9999".to_string(),
        ));
    }
    if revolution_number > 99999 {
        return Err(BraheError::Error(
            "Revolution number cannot exceed 99999".to_string(),
        ));
    }
    if !matches!(classification, 'U' | 'C' | 'S' | ' ') {
        return Err(BraheError::Error(
            "Classification must be 'U', 'C', 'S', or ' '".to_string(),
        ));
    }
    if intl_designator.len() > 8 {
        return Err(BraheError::Error(
            "International designator cannot exceed 8 characters".to_string(),
        ));
    }

    // Get epoch components
    let mut epoch = *epoch;
    epoch.time_system = TimeSystem::UTC; // TLEs use UTC
    let year = epoch.year();
    let day_of_year = epoch.day_of_year();
    let year_2digit = year % 100;

    // Format first derivative with sign
    let ndt2_sign = if first_derivative < 0.0 { "-" } else { " " };
    let ndt2_abs_formatted = format!("{:9.8}", first_derivative.abs());
    let ndt2_no_leading_zero = if ndt2_abs_formatted.starts_with("0.") {
        &ndt2_abs_formatted[1..] // Remove leading zero, keep the dot
    } else {
        &ndt2_abs_formatted
    };
    let ndt2_formatted = format!("{}{}", ndt2_sign, ndt2_no_leading_zero);
    let ndt2_final = if ndt2_formatted.len() > 10 {
        &ndt2_formatted[..10]
    } else {
        &ndt2_formatted
    };

    // Format second derivative in exponential notation
    let nddt6_formatted = format_exponential(second_derivative);

    // Format B* term in exponential notation
    let bstar_formatted = format_exponential(bstar);

    // Ensure international designator is properly formatted (8 characters max)
    let intl_des_formatted = format!("{:8}", intl_designator);
    let intl_des_final = if intl_des_formatted.len() > 8 {
        &intl_des_formatted[..8]
    } else {
        &intl_des_formatted
    };

    // Create line 1 (without checksum)
    let line1_base = format!(
        "1 {:5}{} {} {:02}{:012.8} {} {} {} {} {:04}",
        norad_id_trimmed,
        classification,
        intl_des_final,
        year_2digit,
        day_of_year,
        ndt2_final,
        nddt6_formatted,
        bstar_formatted,
        ephemeris_type,
        element_set_number
    );

    // Calculate checksum and complete line 1
    let line1_checksum = calculate_tle_line_checksum(&line1_base);
    let line1 = format!("{}{}", line1_base, line1_checksum);

    // Format eccentricity (remove "0." prefix)
    let ecc_formatted = format!("{:.7}", eccentricity);
    let ecc_final = if let Some(stripped) = ecc_formatted.strip_prefix("0.") {
        stripped
    } else {
        &ecc_formatted
    };

    // Normalize angles to [0, 360)
    let incl_norm = inclination.rem_euclid(360.0);
    let raan_norm = raan.rem_euclid(360.0);
    let argp_norm = arg_periapsis.rem_euclid(360.0);
    let mean_anom_norm = mean_anomaly.rem_euclid(360.0);

    // Create line 2 (without checksum)
    let line2_base = format!(
        "2 {:5} {:8.4} {:8.4} {} {:8.4} {:8.4} {:11.8}{:05}",
        norad_id_trimmed,
        incl_norm,
        raan_norm,
        ecc_final,
        argp_norm,
        mean_anom_norm,
        mean_motion,
        revolution_number
    );

    // Calculate checksum and complete line 2
    let line2_checksum = calculate_tle_line_checksum(&line2_base);
    let line2 = format!("{}{}", line2_base, line2_checksum);

    Ok((line1, line2))
}

/// Format a number in TLE exponential notation
///
/// # Arguments
/// * `value` - Number to format
///
/// # Returns
/// * `String` - Formatted exponential notation (8 characters)
fn format_exponential(value: f64) -> String {
    if value == 0.0 {
        return " 00000+0".to_string();
    }

    let abs_val = value.abs();
    let sign = if value >= 0.0 { " " } else { "-" };

    // Calculate exponent using TLE standard format
    // TLE uses scientific notation where mantissa is between 0.1 and 1.0
    let log_val = abs_val.log10();
    let exponent = log_val.floor() as i32;

    // Calculate 5-digit mantissa
    let mantissa = abs_val / 10_f64.powi(exponent);
    let body = (mantissa * 10000.0).round() as u32;

    // Format exponent sign
    let exp_sign = if exponent >= 0 { "+" } else { "-" };

    format!("{}{:05}{}{}", sign, body, exp_sign, exponent.abs())
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::orbits::keplerian::semimajor_axis;
    use approx::assert_abs_diff_eq;
    use rstest::rstest;

    #[rstest]
    #[case(
        "1 20580U 90037B   25261.05672437  .00006481  00000+0  23415-3 0  9990",
        0
    )]
    #[case(
        "1 24920U 97047A   25261.00856804  .00000165  00000+0  89800-4 0  9991",
        1
    )]
    #[case(
        "1 00900U 64063C   25261.21093924  .00000602  00000+0  60787-3 0  9992",
        2
    )]
    #[case(
        "1 26605U 00071A   25260.44643294  .00000025  00000+0  00000+0 0  9993",
        3
    )]
    #[case(
        "2 26410 146.0803  17.8086 8595307 233.2516   0.1184  0.44763667 19104",
        4
    )]
    #[case(
        "1 28414U 04035B   25261.30628127  .00003436  00000+0  25400-3 0  9995",
        5
    )]
    #[case(
        "1 28371U 04025F   25260.92882365  .00000356  00000+0  90884-4 0  9996",
        6
    )]
    #[case(
        "1 19751U 89001C   25260.63997541  .00000045  00000+0  00000+0 0  9997",
        7
    )]
    #[case(
        "1 29228U 06021A   25261.14661065  .00002029  00000+0  12599-3 0  9998",
        8
    )]
    #[case(
        "2 31127  98.3591 223.5782 0064856  30.4095 330.0844 14.63937036981529",
        9
    )]
    fn test_calculate_tle_line_checksum(#[case] line: &str, #[case] expected: u32) {
        let checksum = calculate_tle_line_checksum(line);
        assert_eq!(checksum, expected);
    }

    #[rstest]
    #[case("1 20580U 90037B   25261.05672437  .00006481  00000+0  23415-3 0  9990")]
    #[case("1 24920U 97047A   25261.00856804  .00000165  00000+0  89800-4 0  9991")]
    #[case("1 00900U 64063C   25261.21093924  .00000602  00000+0  60787-3 0  9992")]
    #[case("1 26605U 00071A   25260.44643294  .00000025  00000+0  00000+0 0  9993")]
    #[case("2 26410 146.0803  17.8086 8595307 233.2516   0.1184  0.44763667 19104")]
    #[case("1 28414U 04035B   25261.30628127  .00003436  00000+0  25400-3 0  9995")]
    #[case("1 28371U 04025F   25260.92882365  .00000356  00000+0  90884-4 0  9996")]
    #[case("1 19751U 89001C   25260.63997541  .00000045  00000+0  00000+0 0  9997")]
    #[case("1 29228U 06021A   25261.14661065  .00002029  00000+0  12599-3 0  9998")]
    #[case("2 31127  98.3591 223.5782 0064856  30.4095 330.0844 14.63937036981529")]
    fn test_validate_tle_line_valid(#[case] line: &str) {
        assert!(validate_tle_line(line));
    }

    #[rstest]
    #[case("1 20580U 90037B   25261.05672437  .00006481  00000+0  23415-3 0  9980")]
    #[case("1 24920U 97047A   25261.00856804  .00000165  00000+0  89800-4 0  9931")]
    #[case("1 00900U 64063C   25261.21093924  .00000602  00000+0  60787-3 0  9912")]
    #[case("1 26605U 00071A   25260.44643294  .00000025  00000+0  00000+0 0  9983")]
    #[case("2 26410 146.0803  17.8086 8595307 233.2516   0.1184  19104")]
    #[case("1 28414U 04035B   25261.30628127  .00003436  00000+0  25400-3 0  9923421295")]
    #[case("3 28371U 04025F   25260.92882365  .00000356  00000+0  90884-4 0  9996")]
    #[case("3 19751U 89001C   25260.63997541  .00000045  00000+0  00000+0 0  9999")]
    fn test_validate_tle_invalid(#[case] line: &str) {
        assert!(!validate_tle_line(line));
    }

    #[rstest]
    #[case(
        "1 22195U 92070B   25260.83452377 -.00000009  00000+0  00000+0 0  9999",
        "2 22195  52.6519  78.7552 0137761  68.4365 290.4819  6.47293897777784"
    )]
    #[case(
        "1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9997",
        "2 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110516"
    )]
    fn test_validate_tle_lines_valid(#[case] line1: &str, #[case] line2: &str) {
        assert!(validate_tle_lines(line1, line2));
    }

    #[rstest]
    #[case(
        "1 22195U 92070B   25260.83452377 -.00000009  00000+0  00000+0 0  9999",
        "2 22196  52.6519  78.7552 0137761  68.4365 290.4819  6.47293897777784"
    )]
    #[case(
        "1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9997",
        "1 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110516"
    )]
    #[case(
        "1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  999",
        "2 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110516"
    )]
    #[case(
        "1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9997",
        "2 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.0027772611051"
    )]
    #[case(
        "1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9997",
        "3 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110517"
    )]
    #[case(
        "2 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9998",
        "2 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110516"
    )]
    #[case(
        "1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9997",
        "2 23614  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110517"
    )]
    fn test_validate_tle_lines_invalid(#[case] line1: &str, #[case] line2: &str) {
        assert!(!validate_tle_lines(line1, line2));
    }

    #[rstest]
    #[case("25544", 25544)]
    #[case("00001", 1)]
    #[case("99999", 99999)]
    #[case("    1", 1)]
    fn test_parse_norad_id(#[case] id_str: &str, #[case] expected: u32) {
        assert_eq!(parse_norad_id(id_str).unwrap(), expected);
    }

    #[rstest]
    #[case("A0000", 100000)]
    #[case("A0001", 100001)]
    #[case("A9999", 109999)]
    #[case("B0000", 110000)]
    #[case("Z9999", 339999)]
    #[case("B1234", 111234)]
    #[case("C5678", 125678)]
    #[case("D9012", 139012)]
    #[case("E3456", 143456)]
    #[case("F7890", 157890)]
    #[case("G1234", 161234)]
    #[case("H2345", 172345)]
    #[case("J6789", 186789)]
    #[case("K0123", 190123)]
    #[case("L4567", 204567)]
    #[case("M8901", 218901)]
    #[case("N2345", 222345)]
    #[case("P6789", 236789)]
    #[case("Q0123", 240123)]
    #[case("R4567", 254567)]
    #[case("S8901", 268901)]
    #[case("T2345", 272345)]
    #[case("U6789", 286789)]
    #[case("V0123", 290123)]
    #[case("W4567", 304567)]
    #[case("X8901", 318901)]
    #[case("Y2345", 322345)]
    #[case("Z6789", 336789)]
    fn test_parse_norad_id_alpha5_valid(#[case] id_str: &str, #[case] expected: u32) {
        assert_eq!(parse_norad_id(id_str).unwrap(), expected);
    }

    #[rstest]
    #[case("I0001")] // 'I' is invalid
    #[case("O1234")] // 'O' is invalid
    #[case("A123")] // Too short
    #[case("A12345")] // Too long
    #[case("1234A")] // Invalid format
    #[case("!2345")] // Invalid character
    #[case("")] // Empty string
    #[case("     ")] // Only spaces
    fn test_parse_norad_id_invalid(#[case] id_str: &str) {
        assert!(parse_norad_id(id_str).is_err());
    }

    #[test]
    fn test_keplerian_elements_from_tle() {
        let line1 = "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997";
        let line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";

        let result = keplerian_elements_from_tle(line1, line2);
        assert!(result.is_ok());

        let (epoch, elements) = result.unwrap();

        assert_eq!(epoch.year(), 2021);
        assert_eq!(epoch.month(), 1);
        assert_eq!(epoch.day(), 1);
        assert_eq!(epoch.hour(), 12);
        assert_eq!(epoch.minute(), 0);
        assert_abs_diff_eq!(epoch.second(), 0.0, epsilon = 1e-6);

        let n_rad_per_sec = 15.48919103 * 2.0 * PI / 86400.0;
        let a = semimajor_axis(n_rad_per_sec, RADIANS);
        assert_abs_diff_eq!(elements[0], a, epsilon = 1.0e-3); // Semi-major axis in meters
        assert_abs_diff_eq!(elements[1], 0.0003417, epsilon = 1.0e-7); // Eccentricity
        assert_abs_diff_eq!(elements[2], 51.6461, epsilon = 1.0e-4); // Inclination
        assert_abs_diff_eq!(elements[3], 306.0234, epsilon = 1.0e-4); // RAAN
        assert_abs_diff_eq!(elements[4], 88.1267, epsilon = 1.0e-4); // Argument of periapsis
        assert_abs_diff_eq!(elements[5], 25.5695, epsilon = 1.0e-4); // Mean anomaly
    }

    #[test]
    fn test_create_tle_lines() {
        let epoch = Epoch::from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let semi_major_axis = 6786000.0; // meters
        let eccentricity = 0.12345; // dimensionless
        let inclination = 51.6461; // degrees
        let raan = 306.0234; // degrees
        let arg_periapsis = 88.1267; // degrees
        let mean_anomaly = 25.5695; // degrees

        // Convert semi-major axis to mean motion (rev/day)
        let mean_motion_rad_per_sec =
            (GM_EARTH / (semi_major_axis * semi_major_axis * semi_major_axis)).sqrt();
        let mean_motion_revs_per_day = mean_motion_rad_per_sec * 86400.0 / (2.0 * PI);

        let (line1, line2) = create_tle_lines(
            &epoch,
            "25544",
            'U',
            "98067A",
            mean_motion_revs_per_day,
            eccentricity,
            inclination,
            raan,
            arg_periapsis,
            mean_anomaly,
            -0.00001764,
            -0.00000067899,
            -0.00012345,
            0,
            999,
            12345,
        )
        .unwrap();

        assert_eq!(
            line1,
            "1 25544U 98067A   21001.50000000 -.00001764 -67899-7 -12345-4 0 09995"
        );
        assert_eq!(
            line2,
            "2 25544  51.6461 306.0234 1234500  88.1267  25.5695 15.53037630123450"
        );

        let (line1, line2) = create_tle_lines(
            &epoch,
            "25544",
            'U',
            "98067A",
            mean_motion_revs_per_day,
            eccentricity,
            inclination,
            raan,
            arg_periapsis,
            mean_anomaly,
            0.00001764,
            0.00000067899,
            0.00012345,
            0,
            999,
            12345,
        )
        .unwrap();

        assert_eq!(
            line1,
            "1 25544U 98067A   21001.50000000  .00001764  67899-7  12345-4 0 09992"
        );
        assert_eq!(
            line2,
            "2 25544  51.6461 306.0234 1234500  88.1267  25.5695 15.53037630123450"
        );
    }

    #[test]
    fn test_keplerian_elements_to_tle() {
        let epoch = Epoch::from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let semi_major_axis = 6786000.0; // meters
        let eccentricity = 0.12345; // dimensionless
        let inclination = 51.6461; // degrees
        let raan = 306.0234; // degrees
        let arg_periapsis = 88.1267; // degrees
        let mean_anomaly = 25.5695; // degrees

        let elements = Vector6::new(
            semi_major_axis,
            eccentricity,
            inclination,
            raan,
            arg_periapsis,
            mean_anomaly,
        );

        let (line1, line2) = keplerian_elements_to_tle(&epoch, &elements, "25544").unwrap();

        assert_eq!(
            line1,
            "1 25544U          21001.50000000  .00000000  00000+0  00000+0 0 00000"
        );
        assert_eq!(
            line2,
            "2 25544  51.6461 306.0234 1234500  88.1267  25.5695 15.53037630000005"
        );
    }

    #[rstest]
    #[case(0.0, " 00000+0")]
    #[case(0.00012345, " 12345-4")]
    #[case(-0.00012345, "-12345-4")]
    #[case(12345.0, " 12345+4")]
    #[case(-12345.0, "-12345+4")]
    fn test_format_exponential(#[case] value: f64, #[case] expected: &str) {
        assert_eq!(format_exponential(value), expected);
    }

    #[rstest]
    #[case(0, "0")] // Pass through
    #[case(1, "1")] // Pass through
    #[case(42, "42")] // Pass through
    #[case(12345, "12345")] // Pass through
    #[case(99999, "99999")] // Pass through (boundary)
    #[case(100000, "A0000")] // Alpha-5 conversion starts
    #[case(100001, "A0001")]
    #[case(109999, "A9999")]
    #[case(110000, "B0000")]
    #[case(111234, "B1234")]
    #[case(125678, "C5678")]
    #[case(186789, "J6789")] // Skip I
    #[case(236789, "P6789")] // Skip O
    #[case(339999, "Z9999")] // Alpha-5 boundary
    fn test_norad_id_numeric_to_alpha5_valid(#[case] norad_id: u32, #[case] expected: &str) {
        assert_eq!(norad_id_numeric_to_alpha5(norad_id).unwrap(), expected);
    }

    #[rstest]
    #[case(340000)] // Too high
    #[case(999999)] // Way too high
    fn test_norad_id_numeric_to_alpha5_invalid(#[case] norad_id: u32) {
        assert!(norad_id_numeric_to_alpha5(norad_id).is_err());
    }

    #[rstest]
    #[case("A0000", 100000)]
    #[case("A0001", 100001)]
    #[case("A9999", 109999)]
    #[case("B0000", 110000)]
    #[case("B1234", 111234)]
    #[case("C5678", 125678)]
    #[case("J6789", 186789)] // Skip I
    #[case("P6789", 236789)] // Skip O
    #[case("Z9999", 339999)]
    fn test_norad_id_alpha5_to_numeric_valid(#[case] alpha5_id: &str, #[case] expected: u32) {
        assert_eq!(norad_id_alpha5_to_numeric(alpha5_id).unwrap(), expected);
    }

    #[rstest]
    #[case("I0001")] // Invalid letter I
    #[case("O0001")] // Invalid letter O
    #[case("@0001")] // Invalid character
    #[case("A00012")] // Too long
    #[case("A00")] // Too short
    #[case("")] // Empty
    #[case("AAAAA")] // All letters
    fn test_norad_id_alpha5_to_numeric_invalid(#[case] alpha5_id: &str) {
        assert!(norad_id_alpha5_to_numeric(alpha5_id).is_err());
    }

    #[rstest]
    #[case(100000)]
    #[case(100001)]
    #[case(109999)]
    #[case(110000)]
    #[case(125678)]
    #[case(186789)]
    #[case(236789)]
    #[case(339999)]
    fn test_norad_id_alpha5_numeric_round_trip(#[case] id: u32) {
        let alpha5 = norad_id_numeric_to_alpha5(id).unwrap();
        let parsed_id = norad_id_alpha5_to_numeric(&alpha5).unwrap();
        assert_eq!(
            id, parsed_id,
            "Round trip failed for ID {}: {} -> {}",
            id, alpha5, parsed_id
        );
    }

    #[test]
    fn test_keplerian_tle_circularity() {
        // Test circularity: Keplerian elements -> TLE -> Keplerian elements
        let original_epoch = Epoch::from_datetime(2021, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let original_elements = Vector6::new(
            6786000.0, // Semi-major axis (m)
            0.12345,   // Eccentricity
            51.6461,   // Inclination (degrees)
            306.0234,  // RAAN (degrees)
            88.1267,   // Argument of periapsis (degrees)
            25.5695,   // Mean anomaly (degrees)
        );
        let norad_id = "25544";

        // Convert Keplerian elements to TLE
        let (line1, line2) =
            keplerian_elements_to_tle(&original_epoch, &original_elements, norad_id).unwrap();

        // Convert TLE back to Keplerian elements
        let (recovered_epoch, recovered_elements) =
            keplerian_elements_from_tle(&line1, &line2).unwrap();

        // Check that epoch matches (within reasonable precision)
        assert_eq!(recovered_epoch.year(), original_epoch.year());
        assert_eq!(recovered_epoch.month(), original_epoch.month());
        assert_eq!(recovered_epoch.day(), original_epoch.day());
        assert_eq!(recovered_epoch.hour(), original_epoch.hour());
        assert_eq!(recovered_epoch.minute(), original_epoch.minute());
        assert_abs_diff_eq!(
            recovered_epoch.second(),
            original_epoch.second(),
            epsilon = 1e-6
        );

        // Check that elements match (within reasonable precision for TLE format limitations)
        assert_abs_diff_eq!(recovered_elements[0], original_elements[0], epsilon = 1.0); // Semi-major axis (m)
        assert_abs_diff_eq!(recovered_elements[1], original_elements[1], epsilon = 1e-6); // Eccentricity
        assert_abs_diff_eq!(recovered_elements[2], original_elements[2], epsilon = 1e-3); // Inclination (degrees)
        assert_abs_diff_eq!(recovered_elements[3], original_elements[3], epsilon = 1e-3); // RAAN (degrees)
        assert_abs_diff_eq!(recovered_elements[4], original_elements[4], epsilon = 1e-3); // Argument of periapsis (degrees)
        assert_abs_diff_eq!(recovered_elements[5], original_elements[5], epsilon = 1e-3); // Mean anomaly (degrees)
    }

    #[test]
    fn test_tle_keplerian_circularity() {
        // Test circularity: TLE -> Keplerian elements -> TLE
        let original_line1 =
            "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997";
        let original_line2 =
            "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";

        // Convert TLE to Keplerian elements
        let (epoch, elements) =
            keplerian_elements_from_tle(original_line1, original_line2).unwrap();

        // Convert back to TLE
        let (recovered_line1, recovered_line2) =
            keplerian_elements_to_tle(&epoch, &elements, "25544").unwrap();

        // Extract NORAD ID from both TLEs to compare (columns 2-6 of line 1)
        let original_norad_id = original_line1[2..7].trim();
        let recovered_norad_id = recovered_line1[2..7].trim();
        assert_eq!(original_norad_id, recovered_norad_id);

        // Parse both TLEs and compare elements (since exact string match is not expected due to formatting differences)
        let (_, original_elements) =
            keplerian_elements_from_tle(original_line1, original_line2).unwrap();
        let (_, recovered_elements) =
            keplerian_elements_from_tle(&recovered_line1, &recovered_line2).unwrap();

        // Elements should match within TLE precision limits
        assert_abs_diff_eq!(recovered_elements[0], original_elements[0], epsilon = 1.0); // Semi-major axis (m)
        assert_abs_diff_eq!(recovered_elements[1], original_elements[1], epsilon = 1e-6); // Eccentricity
        assert_abs_diff_eq!(recovered_elements[2], original_elements[2], epsilon = 1e-3); // Inclination (degrees)
        assert_abs_diff_eq!(recovered_elements[3], original_elements[3], epsilon = 1e-3); // RAAN (degrees)
        assert_abs_diff_eq!(recovered_elements[4], original_elements[4], epsilon = 1e-3); // Argument of periapsis (degrees)
        assert_abs_diff_eq!(recovered_elements[5], original_elements[5], epsilon = 1e-3); // Mean anomaly (degrees)
    }

    #[test]
    fn test_epoch_from_tle_basic() {
        // Test basic epoch extraction from ISS TLE
        let line1 = "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997";
        let result = epoch_from_tle(line1);
        assert!(result.is_ok());

        let epoch = result.unwrap();
        assert_eq!(epoch.year(), 2021);
        assert_eq!(epoch.month(), 1);
        assert_eq!(epoch.day(), 1);
        assert_eq!(epoch.hour(), 12);
        assert_eq!(epoch.minute(), 0);
        assert_abs_diff_eq!(epoch.second(), 0.0, epsilon = 1e-6);
        assert_eq!(epoch.time_system, TimeSystem::UTC);
    }

    #[rstest]
    #[case(
        "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997",
        2021,
        1,
        1,
        12,
        0,
        0.0
    )]
    #[case(
        "1 25544U 98067A   21032.25000000  .00001764  00000-0  40967-4 0  9997",
        2021,
        2,
        1,
        6,
        0,
        0.0
    )]
    #[case(
        "1 25544U 98067A   21365.00000000  .00001764  00000-0  40967-4 0  9997",
        2021,
        12,
        31,
        0,
        0,
        0.0
    )]
    #[case(
        "1 25544U 98067A   56001.00000000  .00001764  00000-0  40967-4 0  9997",
        2056,
        1,
        1,
        0,
        0,
        0.0
    )] // 2000s epoch (56 < 57)
    #[case(
        "1 25544U 98067A   57001.00000000  .00001764  00000-0  40967-4 0  9997",
        1957,
        1,
        1,
        0,
        0,
        0.0
    )] // Boundary: 57 -> 1957
    #[case(
        "1 25544U 98067A   00001.00000000  .00001764  00000-0  40967-4 0  9997",
        2000,
        1,
        1,
        0,
        0,
        0.0
    )] // Y2K
    #[case(
        "1 25544U 98067A   99365.00000000  .00001764  00000-0  40967-4 0  9997",
        1999,
        12,
        31,
        0,
        0,
        0.0
    )] // Last day of 1999
    fn test_epoch_from_tle_various_dates(
        #[case] line1: &str,
        #[case] year: u32,
        #[case] month: u8,
        #[case] day: u8,
        #[case] hour: u8,
        #[case] minute: u8,
        #[case] second: f64,
    ) {
        let epoch = epoch_from_tle(line1).unwrap();
        assert_eq!(epoch.year(), year);
        assert_eq!(epoch.month(), month);
        assert_eq!(epoch.day(), day);
        assert_eq!(epoch.hour(), hour);
        assert_eq!(epoch.minute(), minute);
        assert_abs_diff_eq!(epoch.second(), second, epsilon = 1e-6);
        assert_eq!(epoch.time_system, TimeSystem::UTC);
    }

    #[test]
    fn test_epoch_from_tle_fractional_day() {
        // Test with fractional day of year
        let line1 = "1 25544U 98067A   21001.75000000  .00001764  00000-0  40967-4 0  9997";
        let epoch = epoch_from_tle(line1).unwrap();

        assert_eq!(epoch.year(), 2021);
        assert_eq!(epoch.month(), 1);
        assert_eq!(epoch.day(), 1);
        assert_eq!(epoch.hour(), 18);
        assert_eq!(epoch.minute(), 0);
        assert_abs_diff_eq!(epoch.second(), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_epoch_from_tle_with_seconds() {
        // Test with fractional seconds
        // 0.00069444 days * 86400 s/day = 60.0 seconds exactly
        let line1 = "1 25544U 98067A   21001.50069444  .00001764  00000-0  40967-4 0  9997";
        let epoch = epoch_from_tle(line1).unwrap();

        assert_eq!(epoch.year(), 2021);
        assert_eq!(epoch.month(), 1);
        assert_eq!(epoch.day(), 1);
        assert_eq!(epoch.hour(), 12);
        // 60 seconds should be represented as 0 minutes, ~60 seconds
        // Due to floating point precision in day_of_year conversion
        assert_eq!(epoch.minute(), 0);
        assert_abs_diff_eq!(epoch.second(), 60.0, epsilon = 1.0);
    }

    #[test]
    fn test_epoch_from_tle_leap_year() {
        // Test leap year day 366
        let line1 = "1 25544U 98067A   20366.00000000  .00001764  00000-0  40967-4 0  9997";
        let epoch = epoch_from_tle(line1).unwrap();

        assert_eq!(epoch.year(), 2020);
        assert_eq!(epoch.month(), 12);
        assert_eq!(epoch.day(), 31);
        assert_eq!(epoch.hour(), 0);
        assert_eq!(epoch.minute(), 0);
        assert_abs_diff_eq!(epoch.second(), 0.0, epsilon = 1e-6);
    }

    #[rstest]
    #[case("1 25544U 98067A   21001.5000000")] // Too short
    #[case("Too short")] // Way too short
    #[case("")] // Empty
    fn test_epoch_from_tle_invalid_lines(#[case] line1: &str) {
        assert!(epoch_from_tle(line1).is_err());
    }

    #[test]
    fn test_epoch_from_tle_consistency_with_keplerian() {
        // Verify epoch_from_tle matches the epoch from keplerian_elements_from_tle
        let line1 = "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997";
        let line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";

        let epoch_direct = epoch_from_tle(line1).unwrap();
        let (epoch_from_keplerian, _) = keplerian_elements_from_tle(line1, line2).unwrap();

        assert_eq!(epoch_direct.year(), epoch_from_keplerian.year());
        assert_eq!(epoch_direct.month(), epoch_from_keplerian.month());
        assert_eq!(epoch_direct.day(), epoch_from_keplerian.day());
        assert_eq!(epoch_direct.hour(), epoch_from_keplerian.hour());
        assert_eq!(epoch_direct.minute(), epoch_from_keplerian.minute());
        assert_abs_diff_eq!(
            epoch_direct.second(),
            epoch_from_keplerian.second(),
            epsilon = 1e-6
        );
    }
}
