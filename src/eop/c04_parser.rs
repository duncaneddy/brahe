/*!
 * Provides helper functions for parsing IERS C04-formatted files
 */

use crate::constants::AS2RAD;
use crate::utils::errors::BraheError;

// Type alias for complex EOP parse result
type EOPParseResult =
    Result<(f64, f64, f64, f64, Option<f64>, Option<f64>, Option<f64>), BraheError>;

/// Parse a line out of a C04 file and return the resulting data.
///
/// # Arguments
/// - `line`: Reference to string to attempt to parse as a C04 formatted line
///
/// # Returns
/// On successful parse returns tuple containing:
/// - `mjd`: Modified Julian date of data point
/// - `pm_x`: x-component of polar motion correction. Units: (radians)
/// - `pm_y`: y-component of polar motion correction. Units: (radians)
/// - `ut1_utc`: Offset of UT1 time scale from UTC time scale. Units: (seconds)
/// - `dX`: "X" component of Celestial Intermediate Pole (CIP) offset. Units: (radians)
/// - `dY`: "Y" component of Celestial Intermediate Pole (CIP) offset. Units: (radians)
/// - `lod`: Difference between astronomically determined length of day and 86400 second TAI. Units: (seconds)
///
/// # References
/// 1. See [EOP 20 C04 Series Metadata](https://datacenter.iers.org/versionMetadata.php?filename=latestVersionMeta/234_EOP_C04_20.62-NOW234.txt) for more information on the C04 file format.
///
/// # Examples
/// ```ignore
/// use brahe::constants::AS2RAD;
/// use brahe::eop::parse_c04_line;
///
/// let line = "2023  11  21   0  60269.00    0.244498    0.234480   0.0111044    0.000305   -0.000100   -0.000720   -0.001318   0.0002867    0.000052    0.000051   0.0000171    0.000054    0.000044    0.000094    0.000068   0.0000298";
///
/// let result = parse_c04_line(line.to_string());
///
/// assert!(result.is_ok());
/// let (mjd, pm_x, pm_y, ut1_utc, dX, dY, lod) = result.unwrap();
///
/// assert_eq!(mjd, 60269.0);
/// assert_eq!(pm_x, 0.244498 * AS2RAD);
/// assert_eq!(pm_y, 0.234480 * AS2RAD);
/// assert_eq!(ut1_utc, 0.0111044);
/// assert_eq!(dX, Some(0.000305 * AS2RAD));
/// assert_eq!(dY, Some(-0.000100 * AS2RAD));
/// assert_eq!(lod, Some(0.0002867));
/// ```
#[allow(non_snake_case)]
pub fn parse_c04_line(line: String) -> EOPParseResult {
    const MJD_RANGE: std::ops::Range<usize> = 16..26;
    const PM_X_RANGE: std::ops::Range<usize> = 26..38;
    const PM_Y_RANGE: std::ops::Range<usize> = 38..50;
    const UT1_UTC_RANGE: std::ops::Range<usize> = 50..62;
    const DX_RANGE: std::ops::Range<usize> = 62..74;
    const DY_RANGE: std::ops::Range<usize> = 74..86;
    const LOD_RANGE: std::ops::Range<usize> = 110..122;
    const C04_LINE_LENGTH: usize = 218;

    if line.len() != C04_LINE_LENGTH {
        return Err(BraheError::EOPError(format!(
            "Line too short to be a standard line: found {} characters, expected {}",
            line.len(),
            C04_LINE_LENGTH
        )));
    }

    let mjd = match line[MJD_RANGE].trim().parse::<f64>() {
        Ok(mjd) => mjd,
        Err(e) => {
            return Err(BraheError::EOPError(format!(
                "Failed to parse mjd from '{}': {}",
                &line[MJD_RANGE], e
            )));
        }
    };
    let pm_x = match line[PM_X_RANGE].trim().parse::<f64>() {
        Ok(pm_x) => pm_x * AS2RAD,
        Err(e) => {
            return Err(BraheError::EOPError(format!(
                "Failed to parse pm_x from '{}': {}",
                &line[PM_X_RANGE], e
            )));
        }
    };
    let pm_y = match line[PM_Y_RANGE].trim().parse::<f64>() {
        Ok(pm_y) => pm_y * AS2RAD,
        Err(e) => {
            return Err(BraheError::EOPError(format!(
                "Failed to parse pm_y from '{}': {}",
                &line[PM_Y_RANGE], e
            )));
        }
    };
    let ut1_utc = match line[UT1_UTC_RANGE].trim().parse::<f64>() {
        Ok(ut1_utc) => ut1_utc,
        Err(e) => {
            return Err(BraheError::EOPError(format!(
                "Failed to parse ut1_utc from '{}': {}",
                &line[UT1_UTC_RANGE], e
            )));
        }
    };
    let lod = match line[LOD_RANGE].trim().parse::<f64>() {
        Ok(lod) => lod,
        Err(e) => {
            return Err(BraheError::EOPError(format!(
                "Failed to parse lod from '{}': {}",
                &line[LOD_RANGE], e
            )));
        }
    };
    let dX = match line[DX_RANGE].trim().parse::<f64>() {
        Ok(dX) => dX * AS2RAD,
        Err(e) => {
            return Err(BraheError::EOPError(format!(
                "Failed to parse dX from '{}': {}",
                &line[DX_RANGE], e
            )));
        }
    };
    let dY = match line[DY_RANGE].trim().parse::<f64>() {
        Ok(dY) => dY * AS2RAD,
        Err(e) => {
            return Err(BraheError::EOPError(format!(
                "Failed to parse dY from '{}': {}",
                &line[DY_RANGE], e
            )));
        }
    };

    Ok((mjd, pm_x, pm_y, ut1_utc, Some(dX), Some(dY), Some(lod)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(non_snake_case)]
    fn test_parse_c04_line() {
        let line = "2023  11  21   0  60269.00    0.244498    0.234480   0.0111044    0.000305   -0.000100   -0.000720   -0.001318   0.0002867    0.000052    0.000051   0.0000171    0.000054    0.000044    0.000094    0.000068   0.0000298";

        let result = parse_c04_line(line.to_string());

        assert!(result.is_ok());
        let (mjd, pm_x, pm_y, ut1_utc, dX, dY, lod) = result.unwrap();

        assert_eq!(mjd, 60269.0);
        assert_eq!(pm_x, 0.244498 * AS2RAD);
        assert_eq!(pm_y, 0.234480 * AS2RAD);
        assert_eq!(ut1_utc, 0.0111044);
        assert_eq!(dX, Some(0.000305 * AS2RAD));
        assert_eq!(dY, Some(-0.000100 * AS2RAD));
        assert_eq!(lod, Some(0.0002867));
    }

    #[test]
    fn test_parse_c04_line_wrong_length_too_short() {
        let line = "short line";
        let result = parse_c04_line(line.to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Line too short to be a standard line")
        );
    }

    #[test]
    fn test_parse_c04_line_wrong_length_too_long() {
        let line = "2023  11  21   0  60269.00    0.244498    0.234480   0.0111044    0.000305   -0.000100   -0.000720   -0.001318   0.0002867    0.000052    0.000051   0.0000171    0.000054    0.000044    0.000094    0.000068   0.0000298EXTRACHARACTERS";
        let result = parse_c04_line(line.to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_c04_line_invalid_mjd() {
        // Create a line with invalid MJD (letters instead of numbers)
        let line = "2023  11  21   0  XXXXX.XX    0.244498    0.234480   0.0111044    0.000305   -0.000100   -0.000720   -0.001318   0.0002867    0.000052    0.000051   0.0000171    0.000054    0.000044    0.000094    0.000068   0.0000298";
        let result = parse_c04_line(line.to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to parse mjd")
        );
    }

    #[test]
    fn test_parse_c04_line_invalid_pm_x() {
        // Create a line with invalid pm_x
        let line = "2023  11  21   0  60269.00    XXXXXXXX    0.234480   0.0111044    0.000305   -0.000100   -0.000720   -0.001318   0.0002867    0.000052    0.000051   0.0000171    0.000054    0.000044    0.000094    0.000068   0.0000298";
        let result = parse_c04_line(line.to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to parse pm_x")
        );
    }

    #[test]
    fn test_parse_c04_line_invalid_pm_y() {
        // Create a line with invalid pm_y
        let line = "2023  11  21   0  60269.00    0.244498    XXXXXXXX   0.0111044    0.000305   -0.000100   -0.000720   -0.001318   0.0002867    0.000052    0.000051   0.0000171    0.000054    0.000044    0.000094    0.000068   0.0000298";
        let result = parse_c04_line(line.to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to parse pm_y")
        );
    }

    #[test]
    fn test_parse_c04_line_invalid_ut1_utc() {
        // Create a line with invalid ut1_utc
        let line = "2023  11  21   0  60269.00    0.244498    0.234480   XXXXXXXXX    0.000305   -0.000100   -0.000720   -0.001318   0.0002867    0.000052    0.000051   0.0000171    0.000054    0.000044    0.000094    0.000068   0.0000298";
        let result = parse_c04_line(line.to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to parse ut1_utc")
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_parse_c04_line_invalid_dX() {
        // Create a line with invalid dX
        let line = "2023  11  21   0  60269.00    0.244498    0.234480   0.0111044    XXXXXXXX   -0.000100   -0.000720   -0.001318   0.0002867    0.000052    0.000051   0.0000171    0.000054    0.000044    0.000094    0.000068   0.0000298";
        let result = parse_c04_line(line.to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to parse dX")
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_parse_c04_line_invalid_dY() {
        // Create a line with invalid dY
        let line = "2023  11  21   0  60269.00    0.244498    0.234480   0.0111044    0.000305   XXXXXXXXX   -0.000720   -0.001318   0.0002867    0.000052    0.000051   0.0000171    0.000054    0.000044    0.000094    0.000068   0.0000298";
        let result = parse_c04_line(line.to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to parse dY")
        );
    }

    #[test]
    fn test_parse_c04_line_invalid_lod() {
        // Create a line with invalid lod
        let line = "2023  11  21   0  60269.00    0.244498    0.234480   0.0111044    0.000305   -0.000100   -0.000720   -0.001318   XXXXXXXXX    0.000052    0.000051   0.0000171    0.000054    0.000044    0.000094    0.000068   0.0000298";
        let result = parse_c04_line(line.to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to parse lod")
        );
    }

    #[test]
    fn test_parse_c04_line_empty_string() {
        let line = "";
        let result = parse_c04_line(line.to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Line too short to be a standard line")
        );
    }
}
