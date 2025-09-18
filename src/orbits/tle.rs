use crate::time::Epoch;
use crate::constants::GM_EARTH;
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
            _ => {}, // Spaces, letters, and other characters contribute 0
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
    let actual_checksum = line.chars().nth(68)
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
        return Err(BraheError::Error(format!("NORAD ID too long: {}. Expected 5 characters found {}", norad_str, norad_str.len())));
    }
    if norad_str.len() < 5 {
        return Err(BraheError::Error(format!("NORAD ID too short: {}. Expected 5 characters found {}", norad_str, norad_str.len())));
    }


    let trimmed = norad_str.trim();

    if trimmed.is_empty() {
        return Err(BraheError::Error("Empty NORAD ID".to_string()));
    }

    let first_char = trimmed.chars().next().unwrap();

    if first_char.is_ascii_digit() {
        // Classic format - direct numeric conversion
        trimmed.parse::<u32>()
            .map_err(|_| BraheError::Error(format!("Invalid numeric NORAD ID: {}", trimmed)))
    } else if first_char.is_ascii_alphabetic() {
        // Alpha-5 format conversion
        convert_alpha5_to_numeric(trimmed)
    } else {
        Err(BraheError::Error(format!("Invalid NORAD ID format: {}", trimmed)))
    }
}

/// Convert Alpha-5 NORAD ID to numeric format
///
/// # Arguments
/// * `alpha5_id` - Alpha-5 format ID (e.g., "A0001")
///
/// # Returns
/// * `Result<u32, BraheError>` - Converted numeric ID
fn convert_alpha5_to_numeric(alpha5_id: &str) -> Result<u32, BraheError> {
    if alpha5_id.len() != 5 {
        return Err(BraheError::Error("Alpha-5 ID must be exactly 5 characters".to_string()));
    }

    let chars: Vec<char> = alpha5_id.chars().collect();
    let first_char = chars[0];

    // Convert first character (A=10, B=11, ..., Z=35, skip I and O)
    let first_value = match first_char {
        'A'..='H' => (first_char as u32) - ('A' as u32) + 10,
        'J'..='N' => (first_char as u32) - ('A' as u32) + 9,  // Skip I
        'P'..='Z' => (first_char as u32) - ('A' as u32) + 8,  // Skip I and O
        _ => return Err(BraheError::Error(format!("Invalid Alpha-5 first character: {}", first_char))),
    };

    // Parse remaining 4 digits
    let remaining: String = chars[1..].iter().collect();
    let numeric_part = remaining.parse::<u32>()
        .map_err(|_| BraheError::Error(format!("Invalid Alpha-5 numeric part: {}", remaining)))?;

    if numeric_part > 9999 {
        return Err(BraheError::Error("Alpha-5 numeric part cannot exceed 9999".to_string()));
    }

    Ok(first_value * 10000 + numeric_part)
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
pub fn keplerian_elements_from_tle(line1: &str, line2: &str) -> Result<(Epoch, Vector6<f64>), BraheError> {
    if !validate_tle_lines(line1, line2) {
        return Err(BraheError::Error("Invalid TLE lines".to_string()));
    }

    // Parse epoch from line 1
    let epoch_str = &line1[18..32];
    let year_2digit: u32 = epoch_str[0..2].parse()
        .map_err(|_| BraheError::Error("Invalid year in TLE".to_string()))?;
    let year = if year_2digit < 57 { 2000 + year_2digit } else { 1900 + year_2digit };
    let day_of_year: f64 = epoch_str[2..].parse()
        .map_err(|_| BraheError::Error("Invalid day of year in TLE".to_string()))?;

    // Convert day of year to month and day
    let day_of_year_int = day_of_year.floor() as u32;
    let fractional_day = day_of_year - day_of_year_int as f64;

    // Simple algorithm to convert day of year to month/day
    let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let days_in_month = [31, if is_leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    let mut month = 1;
    let mut remaining_days = day_of_year_int;

    for (i, &days) in days_in_month.iter().enumerate() {
        if remaining_days <= days {
            month = (i + 1) as u8;
            break;
        }
        remaining_days -= days;
    }

    let day = remaining_days.max(1) as u8;
    let hours = fractional_day * 24.0;
    let hour = hours.floor() as u8;
    let minutes = (hours - hour as f64) * 60.0;
    let minute = minutes.floor() as u8;
    let seconds = (minutes - minute as f64) * 60.0;

    let epoch = Epoch::from_datetime(year, month, day, hour, minute, seconds, 0.0, crate::time::TimeSystem::UTC);

    // Parse orbital elements from line 2
    let inclination: f64 = line2[8..16].trim().parse()
        .map_err(|_| BraheError::Error("Invalid inclination in TLE".to_string()))?;

    let raan: f64 = line2[17..25].trim().parse()
        .map_err(|_| BraheError::Error("Invalid RAAN in TLE".to_string()))?;

    // Parse eccentricity (format: "0.1234567" but stored as "1234567")
    let ecc_str = line2[26..33].trim();
    let eccentricity: f64 = format!("0.{}", ecc_str).parse()
        .map_err(|_| BraheError::Error("Invalid eccentricity in TLE".to_string()))?;

    let arg_perigee: f64 = line2[34..42].trim().parse()
        .map_err(|_| BraheError::Error("Invalid argument of perigee in TLE".to_string()))?;

    let mean_anomaly: f64 = line2[43..51].trim().parse()
        .map_err(|_| BraheError::Error("Invalid mean anomaly in TLE".to_string()))?;

    let mean_motion_revs_per_day: f64 = line2[52..63].trim().parse()
        .map_err(|_| BraheError::Error("Invalid mean motion in TLE".to_string()))?;

    // Convert mean motion from rev/day to rad/s, then compute semi-major axis
    let mean_motion_rad_per_sec = mean_motion_revs_per_day * 2.0 * PI / 86400.0;
    let semi_major_axis = (GM_EARTH / (mean_motion_rad_per_sec * mean_motion_rad_per_sec)).powf(1.0/3.0);

    let elements = Vector6::new(
        semi_major_axis,  // [m]
        eccentricity,     // [dimensionless]
        inclination,      // [degrees]
        raan,             // [degrees]
        arg_perigee,      // [degrees]
        mean_anomaly,     // [degrees]
    );

    Ok((epoch, elements))
}

/// Convert Keplerian elements to TLE format
///
/// # Arguments
/// * `epoch` - Epoch of the elements
/// * `elements` - Keplerian elements [a, e, i, Ω, ω, M]
/// * `norad_id` - NORAD catalog number (optional)
/// * `classification` - Security classification ('U', 'C', or 'S')
/// * `intl_designator` - International designator (optional)
/// * `first_derivative` - First derivative of mean motion divided by 2 [rev/day²]
/// * `second_derivative` - Second derivative of mean motion divided by 6 [rev/day³]
/// * `bstar` - B* drag term [1/earth radii]
///
/// # Returns
/// * `Result<(String, String), BraheError>` - TLE line 1 and line 2
pub fn keplerian_elements_to_tle(
    epoch: &Epoch,
    elements: &Vector6<f64>,
    norad_id: Option<u32>,
    classification: Option<char>,
    intl_designator: Option<&str>,
    first_derivative: Option<f64>,
    second_derivative: Option<f64>,
    bstar: Option<f64>,
) -> Result<(String, String), BraheError> {

    create_tle_lines(
        epoch,
        elements[0], // semi-major axis
        elements[1], // eccentricity
        elements[2], // inclination (degrees)
        elements[3], // RAAN (degrees)
        elements[4], // argument of periapsis (degrees)
        elements[5], // mean anomaly (degrees)
        norad_id,
        classification,
        intl_designator,
        first_derivative,
        second_derivative,
        bstar,
    )
}

/// Create complete TLE lines from all parameters
///
/// # Arguments
/// * `epoch` - Epoch of the elements
/// * `semi_major_axis` - Semi-major axis [m]
/// * `eccentricity` - Eccentricity [dimensionless]
/// * `inclination` - Inclination [degrees]
/// * `raan` - Right ascension of ascending node [degrees]
/// * `arg_periapsis` - Argument of periapsis [degrees]
/// * `mean_anomaly` - Mean anomaly [degrees]
/// * `norad_id` - NORAD catalog number (optional, defaults to 99999)
/// * `classification` - Security classification (optional, defaults to 'U')
/// * `intl_designator` - International designator (optional, defaults to empty)
/// * `first_derivative` - First derivative of mean motion divided by 2 (optional, defaults to 0.0)
/// * `second_derivative` - Second derivative of mean motion divided by 6 (optional, defaults to 0.0)
/// * `bstar` - B* drag term (optional, defaults to 0.0)
///
/// # Returns
/// * `Result<(String, String), BraheError>` - TLE line 1 and line 2
pub fn create_tle_lines(
    epoch: &Epoch,
    semi_major_axis: f64,
    eccentricity: f64,
    inclination: f64,
    raan: f64,
    arg_periapsis: f64,
    mean_anomaly: f64,
    norad_id: Option<u32>,
    classification: Option<char>,
    intl_designator: Option<&str>,
    first_derivative: Option<f64>,
    second_derivative: Option<f64>,
    bstar: Option<f64>,
) -> Result<(String, String), BraheError> {

    let norad_id = norad_id.unwrap_or(99999);
    let classification = classification.unwrap_or('U');
    let intl_designator = intl_designator.unwrap_or("");
    let ndt2 = first_derivative.unwrap_or(0.0);
    let nddt6 = second_derivative.unwrap_or(0.0);
    let bstar_val = bstar.unwrap_or(0.0);

    // Validate inputs
    if norad_id > 99999 {
        return Err(BraheError::Error("NORAD ID cannot exceed 99999".to_string()));
    }
    if eccentricity < 0.0 || eccentricity >= 1.0 {
        return Err(BraheError::Error("Eccentricity must be in range [0, 1)".to_string()));
    }
    if semi_major_axis <= 0.0 {
        return Err(BraheError::Error("Semi-major axis must be positive".to_string()));
    }

    // Convert semi-major axis to mean motion (rev/day)
    let mean_motion_rad_per_sec = (GM_EARTH / (semi_major_axis * semi_major_axis * semi_major_axis)).sqrt();
    let mean_motion_revs_per_day = mean_motion_rad_per_sec * 86400.0 / (2.0 * PI);

    // Format epoch
    // Get the datetime in UTC
    let (year, month, day, hour, minute, second, _nanosecond) = epoch.to_datetime_as_time_system(crate::time::TimeSystem::UTC);

    let year_2digit = (year % 100) as u32;

    // Calculate day of year
    let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let days_in_month = [31, if is_leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

    let mut day_of_year = day as f64;
    for i in 0..(month - 1) as usize {
        day_of_year += days_in_month[i] as f64;
    }

    // Add fractional part for hour, minute, second
    day_of_year += (hour as f64 * 3600.0 + minute as f64 * 60.0 + second) / 86400.0;

    // Format first derivative with sign
    let ndt2_sign = if ndt2 < 0.0 { "-" } else { " " };
    let ndt2_formatted = format!("{}{:9.8}", ndt2_sign, ndt2.abs()).replace("0.", ".");
    let ndt2_final = if ndt2_formatted.len() > 10 {
        &ndt2_formatted[..10]
    } else {
        &ndt2_formatted
    };

    // Format second derivative in exponential notation
    let nddt6_formatted = format_exponential(nddt6);

    // Format B* term in exponential notation
    let bstar_formatted = format_exponential(bstar_val);

    // Ensure international designator is properly formatted (8 characters max)
    let intl_des_formatted = format!("{:8}", intl_designator);
    let intl_des_final = if intl_des_formatted.len() > 8 {
        &intl_des_formatted[..8]
    } else {
        &intl_des_formatted
    };

    // Create line 1 (without checksum)
    let line1_base = format!(
        "1 {:5}{} {} {:02}{:12.8} {} {} {} 0    0",
        norad_id,
        classification,
        intl_des_final,
        year_2digit,
        day_of_year,
        ndt2_final,
        nddt6_formatted,
        bstar_formatted
    );

    // Calculate checksum and complete line 1
    let line1_checksum = calculate_tle_line_checksum(&line1_base);
    let line1 = format!("{}{}", line1_base, line1_checksum);

    // Format eccentricity (remove "0." prefix)
    let ecc_formatted = format!("{:.7}", eccentricity);
    let ecc_final = if ecc_formatted.starts_with("0.") {
        &ecc_formatted[2..]
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
        "2 {:5} {:8.4} {:8.4} {} {:8.4} {:8.4} {:11.8}    0",
        norad_id,
        incl_norm,
        raan_norm,
        ecc_final,
        argp_norm,
        mean_anom_norm,
        mean_motion_revs_per_day
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
        return " 00000-0".to_string();
    }

    let abs_val = value.abs();
    let sign = if value >= 0.0 { " " } else { "-" };

    // Calculate exponent using TLE standard format
    let log_val = abs_val.log10();
    let exponent = log_val.floor() as i32 + 1;

    // Calculate mantissa body
    let body = (abs_val * 10_f64.powi(-exponent + 5)) as u32;

    // TLE format always uses "-" for exponent sign
    format!("{}{:05}-{}", sign, body, exponent.abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use crate::time::Epoch;
    use crate::orbits::keplerian::semimajor_axis;
    use approx::assert_abs_diff_eq;

    #[rstest]
    #[case("1 20580U 90037B   25261.05672437  .00006481  00000+0  23415-3 0  9990", 0)]
    #[case("1 24920U 97047A   25261.00856804  .00000165  00000+0  89800-4 0  9991", 1)]
    #[case("1 00900U 64063C   25261.21093924  .00000602  00000+0  60787-3 0  9992", 2)]
    #[case("1 26605U 00071A   25260.44643294  .00000025  00000+0  00000+0 0  9993", 3)]
    #[case("2 26410 146.0803  17.8086 8595307 233.2516   0.1184  0.44763667 19104", 4)]
    #[case("1 28414U 04035B   25261.30628127  .00003436  00000+0  25400-3 0  9995", 5)]
    #[case("1 28371U 04025F   25260.92882365  .00000356  00000+0  90884-4 0  9996", 6)]
    #[case("1 19751U 89001C   25260.63997541  .00000045  00000+0  00000+0 0  9997", 7)]
    #[case("1 29228U 06021A   25261.14661065  .00002029  00000+0  12599-3 0  9998", 8)]
    #[case("2 31127  98.3591 223.5782 0064856  30.4095 330.0844 14.63937036981529", 9)]
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
    #[case("1 22195U 92070B   25260.83452377 -.00000009  00000+0  00000+0 0  9999", "2 22195  52.6519  78.7552 0137761  68.4365 290.4819  6.47293897777784")]
    #[case("1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9997", "2 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110516")]
    fn test_validate_tle_lines_valid(#[case] line1: &str, #[case] line2: &str) {
        assert!(validate_tle_lines(line1, line2));
    }

    #[rstest]
    #[case("1 22195U 92070B   25260.83452377 -.00000009  00000+0  00000+0 0  9999", "2 22196  52.6519  78.7552 0137761  68.4365 290.4819  6.47293897777784")]
    #[case("1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9997", "1 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110516")]
    #[case("1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  999", "2 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110516")]
    #[case("1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9997", "2 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.0027772611051")]
    #[case("1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9997", "3 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110517")]
    #[case("2 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9998", "2 23613  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110516")]
    #[case("1 23613U 95035B   25260.68951341 -.00000252  00000+0  00000+0 0  9997", "2 23614  13.4910 350.0515 0007963 105.8217 238.1991  1.00277726110517")]
    fn test_validate_tle_lines_invalid(#[case] line1: &str, #[case] line2: &str) {
        assert!(!validate_tle_lines(line1, line2));
    }

    #[rstest]
    #[case("25544", 25544)]
    #[case("00001", 1)]
    #[case("99999", 99999)]
    #[case("    1", 1)]
    fn test_parse_norad_id_classic(#[case] id_str: &str, #[case] expected: u32) {
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
    #[case("A123")]  // Too short
    #[case("A12345")] // Too long
    #[case("1234A")] // Invalid format
    #[case("!2345")] // Invalid character
    #[case("")]      // Empty string
    #[case("     ")] // Only spaces
    fn test_parse_norad_id_invalid(#[case] id_str: &str) {
        assert!(parse_norad_id(id_str).is_err());
    }

    // #[test]
    // fn test_keplerian_elements_from_tle() {
    //     let line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992";
    //     let line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";

    //     let result = keplerian_elements_from_tle(line1, line2);
    //     assert!(result.is_ok());

    //     let (epoch, elements) = result.unwrap();
        
    //     assert_eq!(epoch.year(), 2021);
    //     assert_eq!(epoch.month(), 1);
    //     assert_eq!(epoch.day(), 1);
    //     assert_eq!(epoch.hour(), 0);
    //     assert_eq!(epoch.minute(), 0);
    //     assert_abs_diff_eq!(epoch.second(), 0.0, epsilon = 1e-6);

    //     let n_rad_per_sec = 15.48919103000003 * 2.0 * PI / 86400.0;
    //     let a = semimajor_axis( n_rad_per_sec, false);
    //     assert_abs_diff_eq!(elements[0], a, epsilon = 1.0e-3); // Semi-major axis in meters
    //     assert_abs_diff_eq!(elements[1], 0.0003417, epsilon = 1.0e-7); // Eccentricity
    //     assert_abs_diff_eq!(elements[2], 51.6461, epsilon = 1.0e-4); // Inclination
    //     assert_abs_diff_eq!(elements[3], 306.0234, epsilon = 1.0e-4); // RAAN
    //     assert_abs_diff_eq!(elements[4], 88.1267, epsilon = 1.0e-4); // Argument of periapsis
    //     assert_abs_diff_eq!(elements[5], 25.5695, epsilon = 1.0e-4); // Mean anomaly

    // }

    // #[rstest]
    // fn test_keplerian_elements_from_tle() {
    //     let line1 = "1 25544U 98067A   21001.00000000  .00001764  00000-0  40967-4 0  9992";
    //     let line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";

    //     let result = keplerian_elements_from_tle(line1, line2);
    //     assert!(result.is_ok());

    //     let (elements, _epoch) = result.unwrap();
    //     assert!(elements[0] > 6.0e6); // Semi-major axis should be reasonable for ISS
    //     assert!(elements[1] >= 0.0 && elements[1] < 1.0); // Valid eccentricity
    //     assert!(elements[2] > 50.0 && elements[2] < 60.0); // ISS inclination ~51.6°
    // }

    #[test]
    fn test_format_exponential() {
        assert_eq!(format_exponential(0.0), " 00000-0");
        // Test other values...
    }
}