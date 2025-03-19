/*!
*
*/

use crate::orbits::semimajor_axis;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;
use std::f64::consts::PI;
use std::str::FromStr;

use crate::sgp::{SGP4Data, SGPGravityModel};

/// Two-Line Element set representation
pub struct TLE {
    /// First line of the TLE
    pub line1: String,
    /// Second line of the TLE
    pub line2: String,
    /// Epoch of the TLE
    pub epoch: Epoch,

    // Satellite parameters parsed from TLE
    pub satcat_id: String,
    pub classification: char,
    pub international_designator: String,
    pub year: u32,
    pub day_of_year: f64,
    pub first_derivative_mean_motion: f64,
    pub second_derivative_mean_motion: f64,
    pub bstar: f64,
    pub element_number: u32,
    pub inclination: f64,
    pub raan: f64,
    pub eccentricity: f64,
    pub argument_of_perigee: f64,
    pub mean_anomaly: f64,
    pub mean_motion_revs_per_day: f64,
    pub semimajor_axis: f64,
    pub revolution_number: u32,

    // SGP4 propagator state (initialize on first use)
    sgp4_data: SGP4Data,
}

impl TLE {
    /// Create a new TLE from raw TLE lines
    pub fn new(line1: &str, line2: &str) -> Result<Self, BraheError> {
        // Validate line lengths
        if line1.len() < 69 || line2.len() < 69 {
            return Err(BraheError::Error(
                "TLE lines must be at least 69 characters".to_string(),
            ));
        }

        // Validate checksum
        if !valid_tle_checksum(line1) || !valid_tle_checksum(line2) {
            return Err(BraheError::Error(
                "TLE checksum validation failed".to_string(),
            ));
        }

        // Parse line 1
        let satcat_id = line1[2..7].trim().to_string();
        let classification = line1.chars().nth(7).unwrap_or(' ');
        let international_designator = line1[9..17].trim().to_string();

        let year = match line1[18..20].trim().parse::<u32>() {
            Ok(year) => {
                if year < 57 {
                    year + 2000
                } else {
                    year + 1900
                }
            }
            Err(e) => {
                return Err(BraheError::Error(format!(
                    "Failed to parse epoch year: {}",
                    e
                )));
            }
        };

        let day_of_year = match line1[20..32].trim().parse::<f64>() {
            Ok(day) => day,
            Err(e) => {
                return Err(BraheError::Error(format!(
                    "Failed to parse epoch day: {}",
                    e
                )));
            }
        };

        let first_derivative_raw = &line1[33..43];
        let first_derivative_sign = match first_derivative_raw.chars().nth(0) {
            Some('+') => 1.0,
            Some('-') => -1.0,
            Some(' ') => 1.0,
            _ => {
                return Err(BraheError::Error("Failed to parse ndot sign".to_string()));
            }
        };
        let first_derivative_string = format!("0.{}", &first_derivative_raw[1..7]);
        let first_derivative = match first_derivative_string.parse::<f64>() {
            Ok(val) => val,
            Err(e) => {
                return Err(BraheError::Error(format!(
                    "Failed to parse ndot mantissa: {}",
                    e
                )));
            }
        };
        let first_derivative_mean_motion = first_derivative_sign * first_derivative;

        let second_derivative_raw = &line1[44..52];
        let second_derivative_mantissa = match second_derivative_raw[..6].parse::<f64>() {
            Ok(val) => val,
            Err(e) => {
                return Err(BraheError::Error(format!(
                    "Failed to parse nddot mantissa: {}",
                    e
                )));
            }
        };
        let second_derivative_exponent = match second_derivative_raw[6..].parse::<i32>() {
            Ok(val) => val,
            Err(e) => {
                return Err(BraheError::Error(format!(
                    "Failed to parse nddot exponent: {}",
                    e
                )));
            }
        };
        let second_derivative_mean_motion =
            second_derivative_mantissa * 10.0_f64.powi(second_derivative_exponent);

        // bstar (need to handle sign and exponent correctly)
        let bstar_sign = match line1.chars().nth(53) {
            Some('+') => 1.0,
            Some('-') => -1.0,
            _ => {
                return Err(BraheError::Error("Failed to parse bstar sign".to_string()));
            }
        };
        let bstar_raw = line1[54..61].trim();
        // Add '0.' prefix to bstar_raw to handle implied decimal point
        let bstar_mantissa = format!("0.{}", bstar_raw[..5]);

        let bstar_mantissa = match bstar_mantissa.parse::<f64>() {
            Ok(val) => val,
            Err(e) => {
                return Err(BraheError::Error(format!(
                    "Failed to parse bstar mantissa: {}",
                    e
                )));
            }
        };
        let bstar_exponent = match bstar_raw[5..].parse::<i32>() {
            Ok(val) => val,
            Err(e) => {
                return Err(BraheError::Error(format!(
                    "Failed to parse bstar exponent: {}",
                    e
                )));
            }
        };
        let bstar = bstar_sign * bstar_mantissa * 10.0_f64.powi(bstar_exponent);

        let element_number = match line1[64..68].trim().parse::<u32>() {
            Ok(val) => val,
            Err(e) => {
                return Err(BraheError::Error(format!(
                    "Failed to parse element number: {}",
                    e
                )));
            }
        };

        // Parse Line 2

        let inclination = match line2[8..16].trim().parse::<f64>() {
            Ok(val) => val,
            Err(e) => {
                return Err(BraheError::Error(format!(
                    "Failed to parse inclination: {}",
                    e
                )));
            }
        };

        let raan = match line2[17..25].trim().parse::<f64>() {
            Ok(val) => val,
            Err(e) => return Err(BraheError::Error(format!("Failed to parse RAAN: {}", e))),
        };

        // Eccentricity has an implied decimal point (0. prefix)
        let eccentricity_string = format!("0.{}", &line2[26..33].trim());
        let eccentricity = match eccentricity_string.parse::<f64>() {
            Ok(val) => val,
            Err(e) => {
                return Err(BraheError::Error(format!(
                    "Failed to parse eccentricity: {}",
                    e
                )));
            }
        };

        let argument_of_perigee = match line2[34..42].trim().parse::<f64>() {
            Ok(val) => val,
            Err(e) => {
                return Err(BraheError::Error(format!(
                    "Failed to parse argument of perigee: {}",
                    e
                )));
            }
        };

        let mean_anomaly = match line2[43..51].trim().parse::<f64>() {
            Ok(val) => val,
            Err(e) => {
                return Err(BraheError::Error(format!(
                    "Failed to parse mean anomaly: {}",
                    e
                )));
            }
        };

        let mean_motion_revs_per_day = match line2[52..63].trim().parse::<f64>() {
            Ok(val) => val,
            Err(e) => {
                return Err(BraheError::Error(format!(
                    "Failed to parse mean motion: {}",
                    e
                )));
            }
        };

        // Calculate semimajor axis from mean motion
        let sma = semimajor_axis(mean_motion_revs_per_day * 2.0 * PI / 86400.0, false);

        let revolution_number = match line2[63..68].trim().parse::<u32>() {
            Ok(val) => val,
            Err(e) => {
                return Err(BraheError::Error(format!(
                    "Failed to parse revolution number: {}",
                    e
                )));
            }
        };

        // Create epoch
        let mut epoch = Epoch::from_date(year, 1, 1, TimeSystem::UTC);
        epoch += (day_of_year - 1.0) * 86400.0;

        // Initialize SGP4 propagator data structure
        let sgp4_data = SGP4Data::from_tle(
            satcat_id.clone().as_str(),
            epoch.jd_as_time_system(TimeSystem::UTC) - 2433281.5, // Need days since 1950
            bstar,
            first_derivative_mean_motion,
            second_derivative_mean_motion,
            eccentricity,
            argument_of_perigee * PI / 180.0,
            inclination * PI / 180.0,
            mean_anomaly * PI / 180.0,
            mean_motion_revs_per_day,
            raan * PI / 180.0,
            SGPGravityModel::WGS84,
        )?;

        // Create TLE object
        Ok(TLE {
            line1: line1.to_string(),
            line2: line2.to_string(),
            epoch,
            satcat_id,
            classification,
            international_designator,
            year,
            day_of_year,
            first_derivative_mean_motion,
            second_derivative_mean_motion,
            bstar,
            element_number,
            inclination,
            raan,
            eccentricity,
            argument_of_perigee,
            mean_anomaly,
            mean_motion_revs_per_day,
            semimajor_axis: sma,
            revolution_number,
            sgp4_data,
        })
    }
}

impl FromStr for TLE {
    type Err = BraheError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Split by newlines, handle both Windows and Unix line endings
        let lines: Vec<&str> = s
            .split(&['\n', '\r'][..])
            .filter(|s| !s.is_empty())
            .collect();

        if lines.len() < 2 {
            return Err(BraheError::Error(
                "TLE must have at least two lines".to_string(),
            ));
        }

        // Handle case where there's a title line
        let (line1, line2) =
            if lines.len() >= 3 && !lines[0].starts_with('1') && !lines[0].starts_with('2') {
                (lines[1], lines[2])
            } else {
                (lines[0], lines[1])
            };

        TLE::new(line1, line2)
    }
}

/// Calculate TLE checksum
fn tle_checksum(line: &str) -> u8 {
    let mut sum = 0;
    for c in line.chars().take(68) {
        if c.is_ascii_digit() {
            sum += c as u8 - b'0';
        } else if c == '-' {
            sum += 1;
        }
    }
    sum % 10
}

/// Validate TLE checksum
fn valid_tle_checksum(line: &str) -> bool {
    if line.len() < 69 {
        return false;
    }

    let checksum = match line.chars().nth(68) {
        Some(c) if c.is_ascii_digit() => c as u8 - b'0',
        _ => return false,
    };

    tle_checksum(line) == checksum
}

// /// The `tle_checksum` function calculates the checksum for a given TLE line. The checksum is the
// /// modulo 10 sum of the digits in the line, excluding the last character. Any '-' characters are
// /// treated as 1.
// ///
// /// # Arguments
// /// * `line` - A string slice containing the TLE line to calculate the checksum for.
// ///
// /// # Returns
// /// A u8 value representing the checksum of the TLE line.
// ///
// /// # Example
// /// ```
// /// use brahe::tle_checksum;
// ///
// /// let line = "1 25544U 98067A   20274.51782528  .00000500  00000-0  15574-4 0  9993";
// /// let checksum = tle_checksum(line);
// /// assert_eq!(checksum, 3);
// /// ```
// pub fn tle_checksum(line: &str) -> u8 {
//     let mut sum = 0;
//     for (i, c) in line.chars().enumerate() {
//         if c.is_digit(10) {
//             sum += c.to_digit(10).unwrap() * (i + 1) as u32;
//         } else if c == '-' {
//             sum += 1;
//         }
//     }
//     (sum % 10) as u8
// }
//
// /// The `valid_tle_checksum` function checks if the checksum of a given TLE line is valid. The
// /// checksum is the modulo 10 sum of the digits in the line, excluding the last character. Any '-'
// /// characters are treated as 1.
// ///
// /// # Arguments
// /// * `line` - A string slice containing the TLE line to check the checksum for.
// ///
// /// # Returns
// /// A boolean value indicating if the checksum is valid.
// ///
// /// # Example
// /// ```
// /// use brahe::valid_tle_checksum;
// ///
// /// let line = "1 25544U 98067A   20274.51782528  .00000500  00000-0  15574-4 0  9993";
// /// assert!(valid_tle_checksum(line));
// /// ```
// pub fn valid_tle_checksum(line: &str) -> bool {
//     let checksum = tle_checksum(line);
//
//     // Check if the line length is at least 68 characters
//     if line.len() < 69 {
//         return false;
//     }
//
//     // Check if the last character is a digit matching the checksum
//     if let Some(last) = line.chars().last() {
//         if last.is_digit(10) {
//             checksum == last.to_digit(10).unwrap() as u8
//         } else {
//             false
//         }
//     } else {
//         false
//     }
// }
//
// // struct TLE {
// //     pub line1: String,
// //     pub line2: String,
// //     pub epoch: Epoch,
// // }
// //
// // impl TLE {
// //     pub fn new(line1: &str, line2: &str) -> Result<TLE, String> {
// //         // Check the checksums
// //         let checksum1 = tle_checksum(line1);
// //         let checksum2 = tle_checksum(line2);
// //
// //         if checksum1 != 0 || checksum2 != 0 {
// //             return Err("Checksum failed".to_string());
// //         }
// //
// //         // Parse the epoch
// //         // let epoch = Epoch::from_tle(line1)?;
// //
// //         Ok(TLE {
// //             line1: line1.to_string(),
// //             line2: line2.to_string(),
// //             epoch,
// //         })
// //     }
// // }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tle_checksum() {
        let line = "1 25544U 98067A   20274.51782528  .00000500  00000-0  15574-4 0  9993";
        let checksum = tle_checksum(line);
        assert_eq!(checksum, 3);
    }

    #[test]
    fn test_valid_tle_checksum() {
        let line = "1 25544U 98067A   24287.94238119  .00024791  00000-0  44322-3 0  9991";
        assert!(valid_tle_checksum(line));
    }
}
