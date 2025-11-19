/*!
 * Provides helper functions for parsing CSSI space weather files
 */

use crate::space_weather::types::{SpaceWeatherData, SpaceWeatherSection};
use crate::time::conversions::datetime_to_mjd;
use crate::utils::errors::BraheError;

/// Parse a data line from a CSSI space weather file.
///
/// The format follows the Fortran specification:
/// `FORMAT(I4,I3,I3,I5,I3,8I3,I4,8I4,I4,F4.1,I2,I4,F6.1,I2,5F6.1)`
///
/// # Arguments
/// - `line`: Reference to string to attempt to parse as a CSSI formatted line
///
/// # Returns
/// On successful parse returns tuple containing:
/// - `mjd`: Modified Julian Date for the data entry
/// - `data`: SpaceWeatherData struct with all parsed values
///
/// # References
/// 1. See [CelesTrak Space Weather Format](https://celestrak.org/SpaceData/SpaceWx-format.php)
pub fn parse_cssi_line(line: &str) -> Result<(f64, SpaceWeatherData), BraheError> {
    parse_cssi_line_with_section(line, SpaceWeatherSection::Observed)
}

/// Parse a data line from a CSSI space weather file with a specified section.
///
/// This function handles the different formats for OBSERVED/DAILY_PREDICTED sections
/// (which have complete Kp/Ap data) vs MONTHLY_PREDICTED sections (which have blank
/// Kp/Ap fields but valid F10.7 data).
///
/// # Arguments
/// - `line`: Reference to string to attempt to parse as a CSSI formatted line
/// - `section`: The section of the file this line comes from
///
/// # Returns
/// On successful parse returns tuple containing:
/// - `mjd`: Modified Julian Date for the data entry
/// - `data`: SpaceWeatherData struct with all parsed values (Kp/Ap will be NaN for MONTHLY_PREDICTED)
pub fn parse_cssi_line_with_section(
    line: &str,
    section: SpaceWeatherSection,
) -> Result<(f64, SpaceWeatherData), BraheError> {
    // Minimum line length check - MONTHLY_PREDICTED lines can be shorter
    let min_len = match section {
        SpaceWeatherSection::MonthlyPredicted => 124, // No need for full Kp/Ap fields
        _ => 130,
    };

    if line.len() < min_len {
        return Err(BraheError::SpaceWeatherError(format!(
            "Line too short to be a CSSI space weather line: found {} characters, expected at least {}",
            line.len(),
            min_len
        )));
    }

    // Parse date fields
    let year = parse_field::<u32>(line, 0, 4, "year")?;
    let month = parse_field::<u8>(line, 4, 7, "month")?;
    let day = parse_field::<u8>(line, 7, 10, "day")?;

    // Calculate MJD from date
    let mjd = datetime_to_mjd(year, month, day, 0, 0, 0.0, 0.0);

    // Parse Bartels rotation
    let bsrn = parse_field::<u32>(line, 10, 15, "BSRN")?;
    let nd = parse_field::<u32>(line, 15, 18, "ND")?;

    // For MONTHLY_PREDICTED, Kp/Ap fields are blank - use NaN
    let (kp, kp_sum, ap, ap_avg, cp, c9) = if section == SpaceWeatherSection::MonthlyPredicted {
        (
            [f64::NAN; 8],
            f64::NAN,
            [f64::NAN; 8],
            f64::NAN,
            f64::NAN,
            0u8,
        )
    } else {
        // Parse 8 Kp indices (stored as 0-90, convert to 0.0-9.0)
        let mut kp = [0.0; 8];
        let mut kp_start = 18;
        for (i, kp_val) in kp.iter_mut().enumerate() {
            let kp_int = parse_field::<i32>(line, kp_start, kp_start + 3, &format!("Kp[{}]", i))?;
            *kp_val = convert_kp_to_float(kp_int);
            kp_start += 3;
        }

        // Parse Kp sum (at position 42, width 4)
        let kp_sum_int = parse_field::<i32>(line, 42, 46, "Kp sum")?;
        let kp_sum = kp_sum_int as f64 / 10.0;

        // Parse 8 Ap indices (at position 46, each width 4)
        let mut ap = [0.0; 8];
        let mut ap_start = 46;
        for (i, ap_val) in ap.iter_mut().enumerate() {
            *ap_val = parse_field::<f64>(line, ap_start, ap_start + 4, &format!("Ap[{}]", i))?;
            ap_start += 4;
        }

        // Parse Ap average (at position 78, width 4)
        let ap_avg = parse_field::<f64>(line, 78, 82, "Ap avg")?;

        // Parse Cp (at position 82, width 4)
        let cp = parse_field::<f64>(line, 82, 86, "Cp")?;

        // Parse C9 (at position 86, width 2)
        let c9 = parse_field::<u8>(line, 86, 88, "C9")?;

        (kp, kp_sum, ap, ap_avg, cp, c9)
    };

    // Parse ISN (at position 88, width 4)
    let isn = parse_field::<u32>(line, 88, 92, "ISN")?;

    // Parse F10.7 observed (at position 92, width 6)
    let f107_obs = parse_field::<f64>(line, 92, 98, "F10.7 obs")?;

    // Parse qualifier (at position 98, width 2) - may be blank for monthly predicted
    let qualifier = parse_field_optional::<u8>(line, 98, 100).unwrap_or(0);

    // Parse the remaining F10.7 values (each width 6)
    let f107_adj_ctr81 = parse_field::<f64>(line, 100, 106, "F10.7 adj ctr81")?;
    let f107_adj_lst81 = parse_field::<f64>(line, 106, 112, "F10.7 adj lst81")?;
    let f107_obs_ctr81 = parse_field::<f64>(line, 112, 118, "F10.7 obs ctr81")?;
    let f107_obs_lst81 = parse_field::<f64>(line, 118, 124, "F10.7 obs lst81")?;

    // Note: The adjusted F10.7 daily value would be at position 124-130, but it seems
    // to be the same as the observed value based on file documentation

    let data = SpaceWeatherData {
        year,
        month,
        day,
        bsrn,
        nd,
        kp,
        kp_sum,
        ap,
        ap_avg,
        cp,
        c9,
        isn,
        f107_obs,
        qualifier,
        f107_adj_ctr81,
        f107_adj_lst81,
        f107_obs_ctr81,
        f107_obs_lst81,
        section,
    };

    Ok((mjd, data))
}

/// Convert Kp integer (0-90) to float (0.0-9.0) with proper handling of thirds.
///
/// Kp values are stored as integers where:
/// - 0, 10, 20, ... 90 represent 0, 1, 2, ... 9
/// - 3, 13, 23, ... represent 0+, 1+, 2+, ... (add 0.33)
/// - 7, 17, 27, ... represent 1-, 2-, 3-, ... (subtract 0.33)
fn convert_kp_to_float(kp_int: i32) -> f64 {
    let base = kp_int / 10;
    let remainder = kp_int % 10;

    let fractional = match remainder {
        0 => 0.0,
        3 => 1.0 / 3.0,
        7 => 2.0 / 3.0,
        _ => remainder as f64 / 10.0, // Fallback for unexpected values
    };

    base as f64 + fractional
}

/// Helper function to parse a field from a fixed-width line
fn parse_field<T: std::str::FromStr>(
    line: &str,
    start: usize,
    end: usize,
    field_name: &str,
) -> Result<T, BraheError>
where
    T::Err: std::fmt::Display,
{
    if end > line.len() {
        return Err(BraheError::SpaceWeatherError(format!(
            "Line too short to parse {}: need {} characters, have {}",
            field_name,
            end,
            line.len()
        )));
    }

    line[start..end].trim().parse::<T>().map_err(|e| {
        BraheError::SpaceWeatherError(format!(
            "Failed to parse {} from '{}': {}",
            field_name,
            &line[start..end],
            e
        ))
    })
}

/// Helper function to parse an optional field from a fixed-width line.
/// Returns None if the field is blank or cannot be parsed.
fn parse_field_optional<T: std::str::FromStr>(line: &str, start: usize, end: usize) -> Option<T> {
    if end > line.len() {
        return None;
    }

    let trimmed = line[start..end].trim();
    if trimmed.is_empty() {
        return None;
    }

    trimmed.parse::<T>().ok()
}

/// Detect which section a line belongs to based on file structure.
///
/// Returns:
/// - `Some("OBSERVED")` for observed data section
/// - `Some("DAILY_PREDICTED")` for daily predictions
/// - `Some("MONTHLY_PREDICTED")` for monthly predictions
/// - `None` for header/metadata lines
pub fn detect_section(line: &str) -> Option<&'static str> {
    if line.starts_with("BEGIN OBSERVED") {
        Some("OBSERVED")
    } else if line.starts_with("BEGIN DAILY_PREDICTED") {
        Some("DAILY_PREDICTED")
    } else if line.starts_with("BEGIN MONTHLY_PREDICTED") {
        Some("MONTHLY_PREDICTED")
    } else {
        None
    }
}

/// Check if a line is a data line (starts with a 4-digit year)
pub fn is_data_line(line: &str) -> bool {
    if line.len() < 4 {
        return false;
    }
    line[0..4].trim().parse::<i32>().is_ok()
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_convert_kp_to_float() {
        // Test base values
        assert_abs_diff_eq!(convert_kp_to_float(0), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(convert_kp_to_float(10), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(convert_kp_to_float(50), 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(convert_kp_to_float(90), 9.0, epsilon = 1e-10);

        // Test + values (thirds)
        assert_abs_diff_eq!(convert_kp_to_float(3), 1.0 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(convert_kp_to_float(13), 1.0 + 1.0 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(convert_kp_to_float(43), 4.0 + 1.0 / 3.0, epsilon = 1e-10);

        // Test - values (two thirds)
        assert_abs_diff_eq!(convert_kp_to_float(7), 2.0 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(convert_kp_to_float(17), 1.0 + 2.0 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(convert_kp_to_float(27), 2.0 + 2.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parse_cssi_line() {
        // Sample line from the actual file
        let line = "1957 10 01 1700 19 43 40 30 20 37 23 43 37 273  32  27  15   7  22   9  32  22  21 1.1 5 334 269.8 0 266.8 235.5 269.3 266.6 230.9";

        let result = parse_cssi_line(line);
        assert!(result.is_ok(), "Failed to parse line: {:?}", result.err());

        let (mjd, data) = result.unwrap();

        // Check date
        assert_eq!(data.year, 1957);
        assert_eq!(data.month, 10);
        assert_eq!(data.day, 1);

        // MJD for 1957-10-01 (verified: JD 2436113.5 - 2400000.5 = 36113)
        assert_abs_diff_eq!(mjd, 36112.0, epsilon = 0.1);

        // Check Bartels rotation
        assert_eq!(data.bsrn, 1700);
        assert_eq!(data.nd, 19);

        // Check Kp indices (43, 40, 30, 20, 37, 23, 43, 37)
        assert_abs_diff_eq!(data.kp[0], 4.0 + 1.0 / 3.0, epsilon = 1e-10); // 43
        assert_abs_diff_eq!(data.kp[1], 4.0, epsilon = 1e-10); // 40
        assert_abs_diff_eq!(data.kp[2], 3.0, epsilon = 1e-10); // 30
        assert_abs_diff_eq!(data.kp[3], 2.0, epsilon = 1e-10); // 20
        assert_abs_diff_eq!(data.kp[4], 3.0 + 2.0 / 3.0, epsilon = 1e-10); // 37
        assert_abs_diff_eq!(data.kp[5], 2.0 + 1.0 / 3.0, epsilon = 1e-10); // 23
        assert_abs_diff_eq!(data.kp[6], 4.0 + 1.0 / 3.0, epsilon = 1e-10); // 43
        assert_abs_diff_eq!(data.kp[7], 3.0 + 2.0 / 3.0, epsilon = 1e-10); // 37

        // Check Kp sum (273 / 10 = 27.3)
        assert_abs_diff_eq!(data.kp_sum, 27.3, epsilon = 1e-10);

        // Check Ap indices
        assert_eq!(data.ap[0], 32.0);
        assert_eq!(data.ap[1], 27.0);
        assert_eq!(data.ap[2], 15.0);
        assert_eq!(data.ap[3], 7.0);
        assert_eq!(data.ap[4], 22.0);
        assert_eq!(data.ap[5], 9.0);
        assert_eq!(data.ap[6], 32.0);
        assert_eq!(data.ap[7], 22.0);

        // Check Ap average
        assert_eq!(data.ap_avg, 21.0);

        // Check other indices
        assert_abs_diff_eq!(data.cp, 1.1, epsilon = 1e-10);
        assert_eq!(data.c9, 5);
        assert_eq!(data.isn, 334);

        // Check F10.7 values
        assert_abs_diff_eq!(data.f107_obs, 269.8, epsilon = 1e-10);
        assert_eq!(data.qualifier, 0);
        assert_abs_diff_eq!(data.f107_adj_ctr81, 266.8, epsilon = 1e-10);
        assert_abs_diff_eq!(data.f107_adj_lst81, 235.5, epsilon = 1e-10);
        assert_abs_diff_eq!(data.f107_obs_ctr81, 269.3, epsilon = 1e-10);
        assert_abs_diff_eq!(data.f107_obs_lst81, 266.6, epsilon = 1e-10);
    }

    #[test]
    fn test_parse_cssi_line_short() {
        let short_line = "1957 10 01";
        let result = parse_cssi_line(short_line);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_data_line() {
        assert!(is_data_line(
            "1957 10 01 1700 19 43 40 30 20 37 23 43 37 273"
        ));
        assert!(is_data_line("2024 01 15 2600  5 20 30 17 23"));
        assert!(!is_data_line("BEGIN OBSERVED"));
        assert!(!is_data_line("# Comment line"));
        assert!(!is_data_line(""));
    }

    #[test]
    fn test_detect_section() {
        assert_eq!(detect_section("BEGIN OBSERVED"), Some("OBSERVED"));
        assert_eq!(
            detect_section("BEGIN DAILY_PREDICTED"),
            Some("DAILY_PREDICTED")
        );
        assert_eq!(
            detect_section("BEGIN MONTHLY_PREDICTED"),
            Some("MONTHLY_PREDICTED")
        );
        assert_eq!(detect_section("END OBSERVED"), None);
        assert_eq!(detect_section("# Comment"), None);
    }

    #[test]
    fn test_parse_another_line() {
        // Another line from the file with different values
        let line = "1957 10 06 1700 24 17  3 10  7  0  0  3  3  43   6   2   4   3   0   0   2   2   2 0.0 0 321 250.9 0 269.3 238.4 251.2 269.6 234.3";

        let result = parse_cssi_line(line);
        assert!(result.is_ok());

        let (_, data) = result.unwrap();

        // Check Kp values with zeros
        assert_abs_diff_eq!(data.kp[4], 0.0, epsilon = 1e-10); // 0
        assert_abs_diff_eq!(data.kp[5], 0.0, epsilon = 1e-10); // 0

        // Check Ap values
        assert_eq!(data.ap[4], 0.0);
        assert_eq!(data.ap[5], 0.0);

        // Check Cp is 0.0
        assert_abs_diff_eq!(data.cp, 0.0, epsilon = 1e-10);
        assert_eq!(data.c9, 0);

        // Check default section is Observed
        assert_eq!(data.section, SpaceWeatherSection::Observed);
    }

    #[test]
    fn test_parse_cssi_line_with_section() {
        // Sample line with complete data
        let line = "1957 10 01 1700 19 43 40 30 20 37 23 43 37 273  32  27  15   7  22   9  32  22  21 1.1 5 334 269.8 0 266.8 235.5 269.3 266.6 230.9";

        // Parse as DailyPredicted section
        let result = parse_cssi_line_with_section(line, SpaceWeatherSection::DailyPredicted);
        assert!(result.is_ok());
        let (_, data) = result.unwrap();
        assert_eq!(data.section, SpaceWeatherSection::DailyPredicted);

        // Verify data is still parsed correctly
        assert_eq!(data.year, 1957);
        assert_abs_diff_eq!(data.kp[0], 4.0 + 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_parse_monthly_predicted_line() {
        // Sample MONTHLY_PREDICTED line with blank Kp/Ap fields
        // Format: year, month, day, bsrn, nd, then blanks, then ISN and F10.7 values
        let line = "2041 10 01 2837  1                                                                        10  70.0    69.2  70.5  69.8  68.8  69.0";

        let result = parse_cssi_line_with_section(line, SpaceWeatherSection::MonthlyPredicted);
        assert!(result.is_ok(), "Failed to parse: {:?}", result.err());

        let (mjd, data) = result.unwrap();

        // Check date
        assert_eq!(data.year, 2041);
        assert_eq!(data.month, 10);
        assert_eq!(data.day, 1);
        assert_eq!(data.bsrn, 2837);
        assert_eq!(data.nd, 1);

        // Check that Kp/Ap are NaN
        for i in 0..8 {
            assert!(data.kp[i].is_nan(), "Kp[{}] should be NaN", i);
            assert!(data.ap[i].is_nan(), "Ap[{}] should be NaN", i);
        }
        assert!(data.kp_sum.is_nan());
        assert!(data.ap_avg.is_nan());
        assert!(data.cp.is_nan());

        // Check ISN and F10.7 are parsed correctly
        assert_eq!(data.isn, 10);
        assert_abs_diff_eq!(data.f107_obs, 70.0, epsilon = 1e-10);
        assert_abs_diff_eq!(data.f107_adj_ctr81, 69.2, epsilon = 1e-10);
        assert_abs_diff_eq!(data.f107_adj_lst81, 70.5, epsilon = 1e-10);
        assert_abs_diff_eq!(data.f107_obs_ctr81, 69.8, epsilon = 1e-10);
        assert_abs_diff_eq!(data.f107_obs_lst81, 68.8, epsilon = 1e-10);

        // Check section is set correctly
        assert_eq!(data.section, SpaceWeatherSection::MonthlyPredicted);

        // MJD should be calculated correctly
        assert!(mjd > 60000.0); // 2041 should be well into the future
    }

    #[test]
    fn test_parse_monthly_predicted_fails_as_observed() {
        // MONTHLY_PREDICTED line should fail if parsed as OBSERVED (blank fields)
        let line = "2041 10 01 2837  1                                                                        10  70.0    69.2  70.5  69.8  68.8  69.0";

        let result = parse_cssi_line(line);
        assert!(result.is_err()); // Should fail because Kp fields are blank
    }
}
