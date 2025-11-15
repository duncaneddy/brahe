/*!
 * Provides helper functions for parsing IERS standard-formatted files
 */

use crate::constants::AS2RAD;
use crate::utils::errors::BraheError;

// Type alias for complex EOP parse result
type EOPParseResult =
    Result<(f64, f64, f64, f64, Option<f64>, Option<f64>, Option<f64>), BraheError>;

/// Parse a line out of a standard file and return the resulting data.
///
/// # Arguments
/// - `line`: Reference to string to attempt to parse as a standard formatted line
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
/// 1. See [Standard Series Metadata](https://datacenter.iers.org/versionMetadata.php?filename=latestVersionMeta/9_FINALS.ALL_IAU2000_V2013_019.txt) for more information on the standard file format.
#[allow(non_snake_case)]
pub fn parse_standard_line(line: String) -> EOPParseResult {
    const MJD_RANGE: std::ops::Range<usize> = 6..15;
    const PM_X_RANGE: std::ops::Range<usize> = 17..27;
    const PM_Y_RANGE: std::ops::Range<usize> = 36..46;
    const UT1_UTC_RANGE: std::ops::Range<usize> = 58..68;
    const DX_RANGE: std::ops::Range<usize> = 96..106;
    const DY_RANGE: std::ops::Range<usize> = 115..125;
    const LOD_RANGE: std::ops::Range<usize> = 78..86;
    const STANDARD_LINE_LENGTH: usize = 187;

    if line.len() != STANDARD_LINE_LENGTH {
        return Err(BraheError::EOPError(format!(
            "Line too short to be a standard line: found {} characters, expected {}",
            line.len(),
            STANDARD_LINE_LENGTH
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
        Ok(lod) => Some(lod * 1.0e-3),
        Err(_e) => None,
    };
    let dX = match line[DX_RANGE].trim().parse::<f64>() {
        Ok(dX) => Some(dX * 1.0e-3 * AS2RAD),
        Err(_e) => None,
    };
    let dY = match line[DY_RANGE].trim().parse::<f64>() {
        Ok(dY) => Some(dY * 1.0e-3 * AS2RAD),
        Err(_e) => None,
    };

    Ok((mjd, pm_x, pm_y, ut1_utc, dX, dY, lod))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    #[allow(non_snake_case)]
    fn test_parse_standard_line_full() {
        let line = "2311 1 60249.00 I  0.274620 0.000020  0.268283 0.000018  I 0.0113205 0.0000039 -0.3630 0.0029  I     0.293    0.290    -0.045    0.041  0.274569  0.268315  0.0113342     0.238    -0.039  ";

        let result = parse_standard_line(line.to_string());

        assert!(result.is_ok());
        let (mjd, pm_x, pm_y, ut1_utc, dX, dY, lod) = result.unwrap();

        assert_eq!(mjd, 60249.0);
        assert_eq!(pm_x, 0.274620 * AS2RAD);
        assert_eq!(pm_y, 0.268283 * AS2RAD);
        assert_eq!(ut1_utc, 0.0113205);
        assert_eq!(dX, Some(0.293 * 1.0e-3 * AS2RAD));
        assert_eq!(dY, Some(-0.045 * 1.0e-3 * AS2RAD));
        assert_eq!(lod, Some(-0.3630 * 1.0e-3));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_parse_standard_line_no_bulletin_b() {
        let line = "231220 60298.00 I  0.167496 0.000091  0.200643 0.000091  I 0.0109716 0.0000102  0.7706 0.0069  P     0.103    0.128    -0.193    0.160                                                     ";

        let result = parse_standard_line(line.to_string());

        assert!(result.is_ok());
        let (mjd, pm_x, pm_y, ut1_utc, dX, dY, lod) = result.unwrap();

        assert_eq!(mjd, 60298.0);
        assert_eq!(pm_x, 0.167496 * AS2RAD);
        assert_eq!(pm_y, 0.200643 * AS2RAD);
        assert_eq!(ut1_utc, 0.0109716);
        assert_eq!(dX, Some(0.103 * 1.0e-3 * AS2RAD));
        assert_eq!(dY, Some(-0.193 * 1.0e-3 * AS2RAD));
        assert_eq!(lod, Some(0.7706 * 1.0e-3));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_parse_standard_line_no_bulletin_b_no_lod() {
        let line = "24 3 4 60373.00 P  0.026108 0.007892  0.289637 0.008989  P 0.0110535 0.0072179                 P     0.006    0.128    -0.118    0.160                                                     ";

        let result = parse_standard_line(line.to_string());

        assert!(result.is_ok());
        let (mjd, pm_x, pm_y, ut1_utc, dX, dY, lod) = result.unwrap();

        assert_eq!(mjd, 60373.0);
        assert_eq!(pm_x, 0.026108 * AS2RAD);
        assert_eq!(pm_y, 0.289637 * AS2RAD);
        assert_eq!(ut1_utc, 0.0110535);
        assert_eq!(dX, Some(0.006 * 1.0e-3 * AS2RAD));
        assert_eq!(dY, Some(-0.118 * 1.0e-3 * AS2RAD));
        assert_eq!(lod, None);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_parse_standard_line_no_bulletin_b_no_lod_no_dxdy() {
        let line = "241228 60672.00 P  0.173369 0.019841  0.266914 0.028808  P 0.0420038 0.0254096                                                                                                             ";

        let result = parse_standard_line(line.to_string());

        assert!(result.is_ok());
        let (mjd, pm_x, pm_y, ut1_utc, dX, dY, lod) = result.unwrap();

        assert_eq!(mjd, 60672.0);
        assert_eq!(pm_x, 0.173369 * AS2RAD);
        assert_eq!(pm_y, 0.266914 * AS2RAD);
        assert_eq!(ut1_utc, 0.0420038);
        assert_eq!(dX, None);
        assert_eq!(dY, None);
        assert_eq!(lod, None);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_parse_standard_line_only_mjd() {
        let line = "241229 60673.00                                                                                                                                                                            ";

        let result = parse_standard_line(line.to_string());

        assert!(result.is_err());
    }

    #[test]
    fn test_parse_standard_line_wrong_length_too_short() {
        let line = "short line";
        let result = parse_standard_line(line.to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Line too short to be a standard line")
        );
    }

    #[test]
    fn test_parse_standard_line_wrong_length_too_long() {
        let line = "2311 1 60249.00 I  0.274620 0.000020  0.268283 0.000018  I 0.0113205 0.0000039 -0.3630 0.0029  I     0.293    0.290    -0.045    0.041  0.274569  0.268315  0.0113342     0.238    -0.039  EXTRA";
        let result = parse_standard_line(line.to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_standard_line_invalid_mjd() {
        let line = "2311 1 XXXXX.XX I  0.274620 0.000020  0.268283 0.000018  I 0.0113205 0.0000039 -0.3630 0.0029  I     0.293    0.290    -0.045    0.041  0.274569  0.268315  0.0113342     0.238    -0.039  ";
        let result = parse_standard_line(line.to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to parse mjd")
        );
    }

    #[test]
    fn test_parse_standard_line_invalid_pm_x() {
        let line = "2311 1 60249.00 I  XXXXXXXX 0.000020  0.268283 0.000018  I 0.0113205 0.0000039 -0.3630 0.0029  I     0.293    0.290    -0.045    0.041  0.274569  0.268315  0.0113342     0.238    -0.039  ";
        let result = parse_standard_line(line.to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to parse pm_x")
        );
    }

    #[test]
    fn test_parse_standard_line_invalid_pm_y() {
        let line = "2311 1 60249.00 I  0.274620 0.000020  XXXXXXXX 0.000018  I 0.0113205 0.0000039 -0.3630 0.0029  I     0.293    0.290    -0.045    0.041  0.274569  0.268315  0.0113342     0.238    -0.039  ";
        let result = parse_standard_line(line.to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to parse pm_y")
        );
    }

    #[test]
    fn test_parse_standard_line_invalid_ut1_utc() {
        let line = "2311 1 60249.00 I  0.274620 0.000020  0.268283 0.000018  I XXXXXXXXX 0.0000039 -0.3630 0.0029  I     0.293    0.290    -0.045    0.041  0.274569  0.268315  0.0113342     0.238    -0.039  ";
        let result = parse_standard_line(line.to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Failed to parse ut1_utc")
        );
    }

    #[test]
    fn test_parse_standard_line_empty_string() {
        let line = "";
        let result = parse_standard_line(line.to_string());
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Line too short to be a standard line")
        );
    }
}
