/*!
 * Modified ITC decoder.
 */

use std::path::Path;

use nalgebra::SMatrix;

use super::types::{ITC, ITCCovarianceFrame, ITCHeader, ITCStateVector};
use crate::clients::spacetrack::EphemerisFileName;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

const KM_TO_M: f64 = 1.0e3;
const KM2_TO_M2: f64 = 1.0e6;

fn parse_error(detail: impl AsRef<str>) -> BraheError {
    BraheError::ParseError(format!("Modified ITC: {}", detail.as_ref()))
}

fn month_day_from_day_of_year(year: u32, day_of_year: u32) -> Result<(u8, u8), BraheError> {
    let leap = (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400);
    let days_in_month: [u32; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let days_in_year: u32 = days_in_month.iter().sum();
    if day_of_year < 1 || day_of_year > days_in_year {
        return Err(parse_error(format!(
            "day of year {} out of range for {}",
            day_of_year, year
        )));
    }
    let mut remaining = day_of_year;
    for (index, days) in days_in_month.iter().enumerate() {
        if remaining <= *days {
            return Ok((index as u8 + 1, remaining as u8));
        }
        remaining -= days;
    }
    Err(parse_error(format!(
        "day of year {} out of range for {}",
        day_of_year, year
    )))
}

/// Whether a token is a record epoch (`YYYYDDDHHMMSS` with an optional fraction).
fn is_epoch_token(token: &str) -> bool {
    let integer = token.split_once('.').map(|(i, _)| i).unwrap_or(token);
    integer.len() == 13 && integer.chars().all(|c| c.is_ascii_digit())
}

/// Parses a `YYYYDDDHHMMSS.sss` UTC epoch token.
pub(crate) fn parse_itc_epoch(token: &str) -> Result<Epoch, BraheError> {
    let (integer, fraction) = token.split_once('.').unwrap_or((token, ""));
    if !is_epoch_token(token) {
        return Err(parse_error(format!(
            "invalid epoch '{}'; expected YYYYDDDHHMMSS.sss",
            token
        )));
    }
    let year: u32 = integer[0..4].parse().unwrap();
    let day_of_year: u32 = integer[4..7].parse().unwrap();
    let hour: u32 = integer[7..9].parse().unwrap();
    let minute: u32 = integer[9..11].parse().unwrap();
    let second: u32 = integer[11..13].parse().unwrap();
    if hour > 23 || minute > 59 || second > 60 {
        return Err(parse_error(format!(
            "invalid time of day in epoch '{}'",
            token
        )));
    }
    let fractional: f64 = if fraction.is_empty() {
        0.0
    } else if fraction.chars().all(|c| c.is_ascii_digit()) {
        format!("0.{}", fraction).parse().unwrap()
    } else {
        return Err(parse_error(format!(
            "invalid fractional seconds in epoch '{}'",
            token
        )));
    };
    let (month, day) = month_day_from_day_of_year(year, day_of_year)?;
    Ok(Epoch::from_datetime(
        year,
        month,
        day,
        hour as u8,
        minute as u8,
        second as f64 + fractional,
        0.0,
        TimeSystem::UTC,
    ))
}

fn parse_header_epoch(value: &str, field: &str) -> Result<Epoch, BraheError> {
    Epoch::from_string(value)
        .ok_or_else(|| parse_error(format!("invalid {} value '{}'", field, value)))
}

/// Text between `key` and the nearest following key in `line`, trimmed.
fn header_field<'a>(line: &'a str, key: &str, other_keys: &[&str]) -> Option<&'a str> {
    let start = line.find(key)? + key.len();
    let rest = &line[start..];
    let end = other_keys
        .iter()
        .filter_map(|k| rest.find(k))
        .min()
        .unwrap_or(rest.len());
    Some(rest[..end].trim())
}

fn parse_header(lines: &[&str]) -> Result<ITCHeader, BraheError> {
    let mut header = ITCHeader::new();

    if let Some(value) = header_field(lines[0], "created:", &[])
        && !value.is_empty()
    {
        header.created = Some(parse_header_epoch(value, "created")?);
    }

    let keys = ["ephemeris_start:", "ephemeris_stop:", "step_size:"];
    if let Some(value) = header_field(lines[1], keys[0], &[keys[1], keys[2]])
        && !value.is_empty()
    {
        header.ephemeris_start = Some(parse_header_epoch(value, "ephemeris_start")?);
    }
    if let Some(value) = header_field(lines[1], keys[1], &[keys[0], keys[2]])
        && !value.is_empty()
    {
        header.ephemeris_stop = Some(parse_header_epoch(value, "ephemeris_stop")?);
    }
    if let Some(value) = header_field(lines[1], keys[2], &[keys[0], keys[1]])
        && !value.is_empty()
    {
        header.step_size = Some(
            value
                .parse::<f64>()
                .map_err(|_| parse_error(format!("invalid step_size value '{}'", value)))?,
        );
    }

    if let Some(value) = header_field(lines[2], "ephemeris_source:", &[])
        && !value.is_empty()
    {
        header.ephemeris_source = Some(value.to_string());
    }

    header.covariance_frame = ITCCovarianceFrame::parse(lines[3])?;
    Ok(header)
}

fn parse_state_line(line: &str) -> Result<ITCStateVector, BraheError> {
    let tokens: Vec<&str> = line.split_whitespace().collect();
    if tokens.len() != 7 {
        return Err(parse_error(format!(
            "state line '{}' has {} fields; expected epoch, three positions and three velocities",
            line,
            tokens.len()
        )));
    }
    let epoch = parse_itc_epoch(tokens[0])?;
    let mut values = [0.0_f64; 6];
    for (index, token) in tokens[1..].iter().enumerate() {
        values[index] = token.parse::<f64>().map_err(|_| {
            parse_error(format!(
                "invalid state component '{}' at epoch {}",
                token, epoch
            ))
        })? * KM_TO_M;
    }
    Ok(ITCStateVector::new(
        epoch,
        [values[0], values[1], values[2]],
        [values[3], values[4], values[5]],
    ))
}

fn parse_covariance_lines(lines: &[&str], epoch: Epoch) -> Result<SMatrix<f64, 6, 6>, BraheError> {
    let mut values = Vec::with_capacity(21);
    for line in lines {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() != 7 {
            return Err(parse_error(format!(
                "covariance line '{}' at epoch {} has {} fields; expected 7",
                line,
                epoch,
                tokens.len()
            )));
        }
        for token in tokens {
            values.push(
                token.parse::<f64>().map_err(|_| {
                    parse_error(format!(
                        "invalid covariance value '{}' at epoch {}",
                        token, epoch
                    ))
                })? * KM2_TO_M2,
            );
        }
    }
    let mut matrix = SMatrix::<f64, 6, 6>::zeros();
    let mut index = 0;
    for row in 0..6 {
        for col in 0..=row {
            matrix[(row, col)] = values[index];
            matrix[(col, row)] = values[index];
            index += 1;
        }
    }
    Ok(matrix)
}

fn finish_record(
    itc: &mut ITC,
    state: ITCStateVector,
    covariance_lines: &[&str],
) -> Result<(), BraheError> {
    let count = covariance_lines.len();
    if count != 0 && count != 3 {
        return Err(parse_error(format!(
            "record at {} has {} covariance lines; expected 0 or 3",
            state.epoch, count
        )));
    }
    if count == 3 {
        let covariance = parse_covariance_lines(covariance_lines, state.epoch)?;
        itc.push_state_with_covariance(state, covariance)
    } else {
        itc.push_state(state)
    }
}

impl ITC {
    /// Parses a Modified ITC message from text.
    ///
    /// The state frame is left at the header default (`EME2000`) because the
    /// text carries no frame token; use [`ITC::from_file`] to infer it from
    /// the file name.
    ///
    /// # Arguments
    /// * `content` - Full file text
    ///
    /// # Returns
    /// * `Ok(ITC)`: The parsed message
    /// * `Err(BraheError)`: If the header or any record is malformed
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::ITC;
    ///
    /// let text = std::fs::read_to_string(
    ///     "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt",
    /// ).unwrap();
    /// let itc = ITC::from_str(&text).unwrap();
    /// assert_eq!(itc.len(), 50);
    /// assert!(itc.source_name.is_none());
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(content: &str) -> Result<Self, BraheError> {
        let lines: Vec<&str> = content
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect();
        if lines.len() < 4 {
            return Err(parse_error(format!(
                "expected at least four header lines, found {}",
                lines.len()
            )));
        }
        let header = parse_header(&lines[..4])?;
        let mut itc = ITC::new(header);

        let mut pending: Option<ITCStateVector> = None;
        let mut covariance_lines: Vec<&str> = Vec::with_capacity(3);
        for line in &lines[4..] {
            let first = line.split_whitespace().next().unwrap_or("");
            if is_epoch_token(first) {
                if let Some(state) = pending.take() {
                    finish_record(&mut itc, state, &covariance_lines)?;
                    covariance_lines.clear();
                }
                pending = Some(parse_state_line(line)?);
            } else if pending.is_some() {
                covariance_lines.push(line);
            } else {
                return Err(parse_error(format!(
                    "data line '{}' precedes the first state line",
                    line
                )));
            }
        }
        if let Some(state) = pending.take() {
            finish_record(&mut itc, state, &covariance_lines)?;
        }
        if itc.is_empty() {
            return Err(parse_error("no ephemeris records"));
        }
        Ok(itc)
    }

    /// Reads and parses a Modified ITC file.
    ///
    /// When the file name follows the Space-Track convention it is stored in
    /// `source_name` and its DataType sets `header.state_frame` (`MEME`,
    /// `EME2000` and `J2000` give `EME2000`; `TEME` and `ITRF` give those
    /// frames). A name that does not follow the convention leaves the
    /// defaults.
    ///
    /// # Arguments
    /// * `path` - Path to the file
    ///
    /// # Returns
    /// * `Ok(ITC)`: The parsed message
    /// * `Err(BraheError)`: If the file cannot be read or parsed, or names an unsupported DataType
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::ITC;
    ///
    /// let itc = ITC::from_file(
    ///     "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt",
    /// ).unwrap();
    /// assert_eq!(itc.source_name.as_ref().unwrap().norad_cat_id, 100002);
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, BraheError> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| {
            BraheError::IoError(format!(
                "failed to read Modified ITC file {}: {}",
                path.display(),
                e
            ))
        })?;
        let mut itc = Self::from_str(&content)?;
        if let Some(name) = path.file_name().and_then(|n| n.to_str())
            && let Ok(parsed) = EphemerisFileName::parse(name)
        {
            itc.header.state_frame = super::frames::state_frame_for_data_type(&parsed.data_type)?;
            itc.source_name = Some(parsed);
        }
        Ok(itc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frames::CelestialFrame;
    use crate::time::TimeSystem;
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    const FULL: &str = "test_assets/starlink/MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt";
    const TRUNCATED: &str = "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt";

    fn utc(y: u32, mo: u8, d: u8, h: u8, mi: u8, s: f64) -> Epoch {
        Epoch::from_datetime(y, mo, d, h, mi, s, 0.0, TimeSystem::UTC)
    }

    #[test]
    #[parallel]
    fn test_parse_itc_epoch() {
        let e = parse_itc_epoch("2026254014242.000").unwrap();
        assert_eq!(e, utc(2026, 9, 11, 1, 42, 42.0));
        let e = parse_itc_epoch("2024060235959.500").unwrap();
        assert_eq!(e, utc(2024, 2, 29, 23, 59, 59.5));
        let e = parse_itc_epoch("2025001000000").unwrap();
        assert_eq!(e, utc(2025, 1, 1, 0, 0, 0.0));
        assert!(parse_itc_epoch("2026254014242.").is_ok());
        assert!(parse_itc_epoch("202625401424.000").is_err());
        assert!(parse_itc_epoch("2026366014242.000").is_err());
        assert!(parse_itc_epoch("2026254244242.000").is_err());
        assert!(parse_itc_epoch("2026254016042.000").is_err());
        assert!(parse_itc_epoch("2026254014299.000").is_err());
        assert!(parse_itc_epoch("4.6343390768e-07").is_err());
        assert!(parse_itc_epoch("2026254014242.1a2").is_err());
    }

    #[test]
    #[parallel]
    fn test_parse_full_asset_header() {
        let itc = ITC::from_file(FULL).unwrap();
        assert_eq!(itc.header.created, Some(utc(2026, 9, 11, 1, 55, 52.0)));
        assert_eq!(
            itc.header.ephemeris_start,
            Some(utc(2026, 9, 11, 1, 42, 42.0))
        );
        assert_eq!(
            itc.header.ephemeris_stop,
            Some(utc(2026, 9, 14, 1, 42, 42.0))
        );
        assert_eq!(itc.header.step_size, Some(60.0));
        assert_eq!(itc.header.ephemeris_source.as_deref(), Some("blend"));
        assert_eq!(itc.header.covariance_frame, ITCCovarianceFrame::RTN);
        assert_eq!(itc.header.state_frame, CelestialFrame::EME2000);
        let name = itc.source_name.as_ref().unwrap();
        assert_eq!(name.norad_cat_id, 100001);
        assert_eq!(name.object_name, "STARLINK-38128");
    }

    #[test]
    #[parallel]
    fn test_parse_full_asset_records() {
        let itc = ITC::from_file(FULL).unwrap();
        assert_eq!(itc.len(), 4321);
        assert_eq!(itc.covariances.len(), 4321);

        let first = &itc.states[0];
        assert_eq!(first.epoch, utc(2026, 9, 11, 1, 42, 42.0));
        assert_abs_diff_eq!(first.position[0], 4244.3465367594e3, epsilon = 1e-6);
        assert_abs_diff_eq!(first.position[1], 1264.3254891872e3, epsilon = 1e-6);
        assert_abs_diff_eq!(first.position[2], 5043.9826441325e3, epsilon = 1e-6);
        assert_abs_diff_eq!(first.velocity[0], 3.5951547629e3, epsilon = 1e-9);
        assert_abs_diff_eq!(first.velocity[1], 5.2587956583e3, epsilon = 1e-9);
        assert_abs_diff_eq!(first.velocity[2], -4.3350352914e3, epsilon = 1e-9);

        let c = &itc.covariances[0];
        assert_abs_diff_eq!(c[(0, 0)], 4.6343390768e-07 * 1e6, epsilon = 1e-12);
        assert_abs_diff_eq!(c[(1, 0)], -3.7963809271e-07 * 1e6, epsilon = 1e-12);
        assert_abs_diff_eq!(c[(1, 1)], 7.7770281867e-07 * 1e6, epsilon = 1e-12);
        assert_abs_diff_eq!(c[(2, 2)], 1.2050744950e-06 * 1e6, epsilon = 1e-12);
        assert_abs_diff_eq!(c[(3, 0)], 8.2617476650e-10 * 1e6, epsilon = 1e-15);
        assert_abs_diff_eq!(c[(5, 5)], 5.4287251909e-12 * 1e6, epsilon = 1e-17);
        for i in 0..6 {
            for j in 0..6 {
                assert_eq!(c[(i, j)], c[(j, i)]);
            }
        }

        let last = itc.states.last().unwrap();
        assert_eq!(last.epoch, utc(2026, 9, 14, 1, 42, 42.0));
        for w in itc.states.windows(2) {
            assert_abs_diff_eq!(w[1].epoch - w[0].epoch, 60.0, epsilon = 1e-6);
        }
    }

    #[test]
    #[parallel]
    fn test_parse_truncated_asset() {
        let itc = ITC::from_file(TRUNCATED).unwrap();
        assert_eq!(itc.len(), 50);
        assert!(itc.has_covariance());
        assert_eq!(itc.header.created, Some(utc(2026, 9, 11, 2, 4, 18.0)));
        assert_eq!(itc.states[0].epoch, utc(2026, 9, 11, 1, 49, 42.0));
        assert_eq!(itc.source_name.as_ref().unwrap().norad_cat_id, 100002);
    }

    #[test]
    #[parallel]
    fn test_parse_from_str_has_no_source_name() {
        let content = std::fs::read_to_string(TRUNCATED).unwrap();
        let itc = ITC::from_str(&content).unwrap();
        assert!(itc.source_name.is_none());
        assert_eq!(itc.header.state_frame, CelestialFrame::EME2000);
        assert_eq!(itc.len(), 50);
    }

    #[test]
    #[parallel]
    fn test_parse_without_covariance() {
        let content = "created:\n\
ephemeris_start:2026-09-11 01:42:42 UTC ephemeris_stop:2026-09-11 01:44:42 UTC step_size:60\n\
ephemeris_source:test\n\
UVW\n\
2026254014242.000 4244.3465367594 1264.3254891872 5043.9826441325 3.5951547629 5.2587956583 -4.3350352914\n\
2026254014342.000 4449.8462744669 1576.6138914245 4772.1198724309 3.2521129034 5.1467007555 -4.7234817117\n\
2026254014442.000 4640.0000000000 1880.0000000000 4480.0000000000 2.9000000000 5.0000000000 -5.0000000000\n";
        let itc = ITC::from_str(content).unwrap();
        assert_eq!(itc.len(), 3);
        assert!(!itc.has_covariance());
        assert!(itc.header.created.is_none());
        assert_eq!(itc.header.step_size, Some(60.0));
    }

    #[test]
    #[parallel]
    fn test_parse_header_fields_in_any_order() {
        let content = "created:2026-09-11 01:55:52 UTC\n\
step_size:60 ephemeris_stop:2026-09-14 01:42:42 UTC ephemeris_start:2026-09-11 01:42:42 UTC\n\
ephemeris_source:blend\n\
UVW\n\
2026254014242.000 4244.3465367594 1264.3254891872 5043.9826441325 3.5951547629 5.2587956583 -4.3350352914\n";
        let itc = ITC::from_str(content).unwrap();
        assert_eq!(
            itc.header.ephemeris_start,
            Some(utc(2026, 9, 11, 1, 42, 42.0))
        );
        assert_eq!(
            itc.header.ephemeris_stop,
            Some(utc(2026, 9, 14, 1, 42, 42.0))
        );
        assert_eq!(itc.header.step_size, Some(60.0));
    }

    #[test]
    #[parallel]
    fn test_parse_tolerates_unrecognized_descriptive_lines() {
        let content = "Generated by an operator tool\n\
Some other note\n\
And a third\n\
J2000\n\
2026254014242.000 4244.3465367594 1264.3254891872 5043.9826441325 3.5951547629 5.2587956583 -4.3350352914\n";
        let itc = ITC::from_str(content).unwrap();
        assert!(itc.header.created.is_none());
        assert!(itc.header.ephemeris_start.is_none());
        assert!(itc.header.ephemeris_source.is_none());
        assert_eq!(itc.header.covariance_frame, ITCCovarianceFrame::EME2000);
        assert_eq!(itc.len(), 1);
    }

    fn header_and(records: &str) -> String {
        format!(
            "created:2026-09-11 01:55:52 UTC\n\
ephemeris_start:2026-09-11 01:42:42 UTC ephemeris_stop:2026-09-14 01:42:42 UTC step_size:60\n\
ephemeris_source:blend\n\
UVW\n{}",
            records
        )
    }

    const REC0: &str = "2026254014242.000 4244.3465367594 1264.3254891872 5043.9826441325 3.5951547629 5.2587956583 -4.3350352914\n";
    const COV0: &str = "4.6343390768e-07 -3.7963809271e-07 7.7770281867e-07 1.8398914684e-10 2.3515746188e-10 1.2050744950e-06 8.2617476650e-10\n\
-9.3402950908e-10 6.5056423006e-13 2.0019807553e-12 -4.6727451721e-10 4.1362166473e-10 -1.2117302280e-12 -8.4570284181e-13\n\
5.1950545462e-13 2.6473144477e-13 8.7817244924e-13 1.6102297802e-09 -2.1049889128e-16 -1.5343530772e-15 5.4287251909e-12\n";
    const REC1: &str = "2026254014342.000 4449.8462744669 1576.6138914245 4772.1198724309 3.2521129034 5.1467007555 -4.7234817117\n";

    #[test]
    #[parallel]
    fn test_parse_errors() {
        assert!(ITC::from_str("created:\nephemeris_source:x\nUVW\n").is_err());
        assert!(ITC::from_str(&header_and("")).is_err());
        assert!(ITC::from_str(&header_and(&REC0.replace(" -4.3350352914", ""))).is_err());
        assert!(
            ITC::from_str(&header_and(&format!(
                "{}{}",
                REC0,
                &COV0[..COV0.len() - 20]
            )))
            .is_err()
        );
        assert!(ITC::from_str(&header_and(&format!("{}{}{}", REC0, COV0, REC1))).is_err());
        assert!(
            ITC::from_str(&header_and(&format!(
                "{}{}{}{}",
                REC0,
                COV0,
                REC1,
                COV0.lines().take(2).collect::<Vec<_>>().join("\n")
            )))
            .is_err()
        );
        assert!(ITC::from_str(&header_and(&format!("{}{}", REC1, REC0))).is_err());
        assert!(ITC::from_str(&header_and(&format!("{}{}", COV0, REC0))).is_err());
        let bad_step = header_and(REC0).replace("step_size:60", "step_size:abc");
        assert!(ITC::from_str(&bad_step).is_err());
        let bad_created =
            header_and(REC0).replace("created:2026-09-11 01:55:52 UTC", "created:yesterday");
        assert!(ITC::from_str(&bad_created).is_err());
        let bad_frame = header_and(REC0).replace("\nUVW\n", "\nTEME\n");
        assert!(ITC::from_str(&bad_frame).is_err());
        assert!(ITC::from_file("test_assets/starlink/does_not_exist.txt").is_err());
        let bad_state_component = REC0.replace("4244.3465367594", "abc");
        assert!(ITC::from_str(&header_and(&bad_state_component)).is_err());
        let bad_covariance_value = COV0.replace("4.6343390768e-07", "abc");
        assert!(ITC::from_str(&header_and(&format!("{}{}", REC0, bad_covariance_value))).is_err());
    }

    #[test]
    #[parallel]
    fn test_parse_covariance_lower_triangle_order() {
        let itc = ITC::from_str(&header_and(&format!("{}{}", REC0, COV0))).unwrap();
        let c = &itc.covariances[0];
        assert_abs_diff_eq!(c[(3, 1)], -9.3402950908e-10 * 1e6, epsilon = 1e-16);
        assert_abs_diff_eq!(c[(4, 4)], 5.1950545462e-13 * 1e6, epsilon = 1e-18);
        assert_abs_diff_eq!(c[(5, 0)], 2.6473144477e-13 * 1e6, epsilon = 1e-18);
        assert_abs_diff_eq!(c[(5, 4)], -1.5343530772e-15 * 1e6, epsilon = 1e-20);
    }
}
