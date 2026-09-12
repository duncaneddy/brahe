/*!
 * Modified ITC encoder.
 */

use std::path::Path;

use super::types::{ITC, is_symmetric};
use crate::time::conversions::day_of_year_from_calendar;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

const M_TO_KM: f64 = 1.0e-3;
const M2_TO_KM2: f64 = 1.0e-6;

/// Formats `value` like C's `%.10e`: ten fraction digits, a signed
/// exponent of at least two digits.
///
/// # Arguments
/// * `value` - Value to format
///
/// # Returns
/// * `String`: The C-style scientific representation
fn format_scientific(value: f64) -> String {
    let rust = format!("{:.10e}", value);
    let (mantissa, exponent) = rust.split_once('e').unwrap();
    let (sign, digits) = match exponent.strip_prefix('-') {
        Some(d) => ("-", d),
        None => ("+", exponent),
    };
    format!("{}e{}{:0>2}", mantissa, sign, digits)
}

/// Formats an epoch as `YYYY-MM-DD HH:MM:SS UTC` for the descriptive header lines.
///
/// # Arguments
/// * `epoch` - Epoch to format
///
/// # Returns
/// * `String`: The descriptive header timestamp
fn format_header_epoch(epoch: &Epoch) -> String {
    let (year, month, day, hour, minute, second, _) =
        epoch.to_datetime_as_time_system(TimeSystem::UTC);
    format!(
        "{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC",
        year,
        month,
        day,
        hour,
        minute,
        second.floor() as u32
    )
}

/// Formats an epoch as the record token `YYYYDDDHHMMSS.sss`.
///
/// The epoch is rounded to the nearest millisecond before it is split into
/// calendar fields, so a fraction that rounds up carries into the minute,
/// hour and day instead of printing a seconds field of `60.000`.
fn format_record_epoch(epoch: &Epoch) -> String {
    let (_, _, _, _, _, second, nanosecond) = epoch.to_datetime_as_time_system(TimeSystem::UTC);
    let raw_seconds = second + nanosecond * 1.0e-9;
    let mut rounded = *epoch + ((raw_seconds * 1.0e3).round() * 1.0e-3 - raw_seconds);
    let (mut year, mut month, mut day, mut hour, mut minute, mut second, mut nanosecond) =
        rounded.to_datetime_as_time_system(TimeSystem::UTC);
    if ((second + nanosecond * 1.0e-9) * 1.0e3).round() >= 60.0e3 {
        rounded += 1.0e-6;
        (year, month, day, hour, minute, second, nanosecond) =
            rounded.to_datetime_as_time_system(TimeSystem::UTC);
    }
    let day_of_year = day_of_year_from_calendar(year, month, day);
    format!(
        "{:04}{:03}{:02}{:02}{:06.3}",
        year,
        day_of_year,
        hour,
        minute,
        second + nanosecond * 1.0e-9
    )
}

/// Formats the header step size, printing whole seconds without a decimal point.
fn format_step(step: f64) -> String {
    let step = (step * 1.0e6).round() * 1.0e-6;
    if step.fract() == 0.0 {
        format!("{}", step as i64)
    } else {
        format!("{}", step)
    }
}

impl ITC {
    /// Renders the message as Modified ITC text.
    ///
    /// Header lines follow Starlink's convention: `created:`,
    /// `ephemeris_start: ... ephemeris_stop: ... step_size:`,
    /// `ephemeris_source:`, then the covariance frame token. Start, stop and
    /// step fall back to the first epoch, last epoch and first epoch
    /// difference when the header does not set them. Positions and
    /// velocities are written in km and km/s with ten decimals; covariance
    /// elements in km-based units as `%.10e`.
    ///
    /// # Returns
    /// * `Ok(String)`: The file text, newline-terminated
    /// * `Err(BraheError)`: If the message has no records
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::itc::ITC;
    ///
    /// let path = "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt";
    /// let itc = ITC::from_file(path).unwrap();
    /// let text = itc.to_string().unwrap();
    /// assert_eq!(text.trim_end(), std::fs::read_to_string(path).unwrap().trim_end());
    /// ```
    pub fn to_string(&self) -> Result<String, BraheError> {
        if self.states.is_empty() {
            return Err(BraheError::Error(
                "cannot write a Modified ITC message with no records".to_string(),
            ));
        }
        if !self.covariances.is_empty() && self.covariances.len() != self.states.len() {
            return Err(BraheError::Error(format!(
                "Modified ITC covariance is all-or-none; {} records but {} covariance matrices",
                self.states.len(),
                self.covariances.len()
            )));
        }
        let non_finite = self.states.iter().any(|state| {
            !state.position.iter().all(|v| v.is_finite())
                || !state.velocity.iter().all(|v| v.is_finite())
        }) || self
            .covariances
            .iter()
            .any(|covariance| !covariance.iter().all(|v| v.is_finite()));
        if non_finite {
            return Err(BraheError::Error(
                "cannot write a Modified ITC message with a non-finite position, velocity or covariance element"
                    .to_string(),
            ));
        }
        for (index, covariance) in self.covariances.iter().enumerate() {
            if !is_symmetric(covariance) {
                return Err(BraheError::Error(format!(
                    "Modified ITC covariance at record {} is not symmetric",
                    index
                )));
            }
        }
        let start = self.header.ephemeris_start.or_else(|| self.start_epoch());
        let stop = self.header.ephemeris_stop.or_else(|| self.end_epoch());
        let step = self.header.step_size.or_else(|| {
            (self.states.len() >= 2).then(|| self.states[1].epoch - self.states[0].epoch)
        });

        let mut out = String::with_capacity(self.states.len() * 480 + 256);
        out.push_str("created:");
        if let Some(created) = &self.header.created {
            out.push_str(&format_header_epoch(created));
        }
        out.push('\n');
        out.push_str("ephemeris_start:");
        if let Some(start) = &start {
            out.push_str(&format_header_epoch(start));
        }
        out.push_str(" ephemeris_stop:");
        if let Some(stop) = &stop {
            out.push_str(&format_header_epoch(stop));
        }
        out.push_str(" step_size:");
        if let Some(step) = step {
            out.push_str(&format_step(step));
        }
        out.push('\n');
        out.push_str("ephemeris_source:");
        if let Some(source) = &self.header.ephemeris_source {
            out.push_str(source);
        }
        out.push('\n');
        out.push_str(self.header.covariance_frame.token());
        out.push('\n');

        let mut previous_token: Option<(Epoch, String)> = None;
        for (index, state) in self.states.iter().enumerate() {
            let epoch_token = format_record_epoch(&state.epoch);
            if let Some((previous_epoch, previous_epoch_token)) = &previous_token
                && epoch_token.as_str() <= previous_epoch_token.as_str()
            {
                return Err(BraheError::Error(format!(
                    "records at {} and {} render to the same millisecond {}",
                    previous_epoch, state.epoch, epoch_token
                )));
            }
            previous_token = Some((state.epoch, epoch_token.clone()));
            out.push_str(&format!(
                "{} {:.10} {:.10} {:.10} {:.10} {:.10} {:.10}\n",
                epoch_token,
                state.position[0] * M_TO_KM,
                state.position[1] * M_TO_KM,
                state.position[2] * M_TO_KM,
                state.velocity[0] * M_TO_KM,
                state.velocity[1] * M_TO_KM,
                state.velocity[2] * M_TO_KM,
            ));
            if let Some(covariance) = self.covariances.get(index) {
                let mut values = Vec::with_capacity(21);
                for row in 0..6 {
                    for col in 0..=row {
                        values.push(format_scientific(covariance[(row, col)] * M2_TO_KM2));
                    }
                }
                for chunk in values.chunks(7) {
                    out.push_str(&chunk.join(" "));
                    out.push('\n');
                }
            }
        }
        Ok(out)
    }

    /// Writes the message to a file in Modified ITC text form.
    ///
    /// The file name is not checked against the Space-Track convention;
    /// use [`ITC::file_name`] to build a compliant one.
    ///
    /// # Arguments
    /// * `path` - Destination path
    ///
    /// # Returns
    /// * `Ok(())`: File written
    /// * `Err(BraheError)`: If the message is empty or the file cannot be written
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use brahe::itc::ITC;
    ///
    /// let itc = ITC::from_file(
    ///     "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt",
    /// ).unwrap();
    /// itc.to_file("/tmp/copy.txt").unwrap();
    /// ```
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), BraheError> {
        let text = self.to_string()?;
        std::fs::write(path.as_ref(), text).map_err(|e| {
            BraheError::IoError(format!(
                "failed to write Modified ITC file {}: {}",
                path.as_ref().display(),
                e
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::itc::{ITCCovarianceFrame, ITCHeader, ITCStateVector};
    use crate::time::TimeSystem;
    use nalgebra::SMatrix;
    use serial_test::parallel;

    const FULL: &str = "test_assets/starlink/MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt";
    const TRUNCATED: &str = "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt";

    fn utc(y: u32, mo: u8, d: u8, h: u8, mi: u8, s: f64) -> Epoch {
        Epoch::from_datetime(y, mo, d, h, mi, s, 0.0, TimeSystem::UTC)
    }

    #[test]
    #[parallel]
    fn test_format_scientific_matches_c_style() {
        assert_eq!(format_scientific(4.6343390768e-07), "4.6343390768e-07");
        assert_eq!(format_scientific(-9.3402950908e-10), "-9.3402950908e-10");
        assert_eq!(format_scientific(1.444e1), "1.4440000000e+01");
        assert_eq!(format_scientific(9.0e-2), "9.0000000000e-02");
        assert_eq!(format_scientific(0.0), "0.0000000000e+00");
        assert_eq!(format_scientific(1.0e-100), "1.0000000000e-100");
    }

    #[test]
    #[parallel]
    fn test_format_record_epoch() {
        assert_eq!(
            format_record_epoch(&utc(2026, 9, 11, 1, 42, 42.0)),
            "2026254014242.000"
        );
        assert_eq!(
            format_record_epoch(&utc(2024, 2, 29, 23, 59, 59.5)),
            "2024060235959.500"
        );
        assert_eq!(
            format_record_epoch(&utc(2025, 1, 1, 0, 0, 0.0)),
            "2025001000000.000"
        );
        assert_eq!(
            format_record_epoch(&(utc(2026, 9, 11, 1, 42, 59.0) + 0.9999996)),
            "2026254014300.000"
        );
        assert_eq!(
            format_record_epoch(&(utc(2026, 12, 31, 23, 59, 59.0) + 0.9996)),
            "2027001000000.000"
        );
        assert_eq!(
            format_record_epoch(&(utc(2026, 9, 11, 1, 42, 42.0) + 0.0004)),
            "2026254014242.000"
        );
        assert_eq!(
            format_record_epoch(&(utc(2026, 9, 11, 1, 42, 42.0) + 0.0006)),
            "2026254014242.001"
        );
    }

    #[test]
    #[parallel]
    fn test_format_record_epoch_leap_second() {
        let leap: Epoch = Epoch::from_datetime(2017, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC) - 1.0;
        assert_eq!(format_record_epoch(&leap), "2016366235960.000");
    }

    #[test]
    #[parallel]
    fn test_write_truncated_asset_matches_source_text() {
        let original = std::fs::read_to_string(TRUNCATED).unwrap();
        let itc = ITC::from_file(TRUNCATED).unwrap();
        let written = itc.to_string().unwrap();
        assert_eq!(written.trim_end(), original.trim_end());
    }

    #[test]
    #[parallel]
    fn test_write_full_asset_matches_source_text() {
        let original = std::fs::read_to_string(FULL).unwrap();
        let itc = ITC::from_file(FULL).unwrap();
        let written = itc.to_string().unwrap();
        assert_eq!(written.trim_end(), original.trim_end());
    }

    #[test]
    #[parallel]
    fn test_write_round_trip_preserves_values() {
        let itc = ITC::from_file(TRUNCATED).unwrap();
        let again = ITC::from_str(&itc.to_string().unwrap()).unwrap();
        assert_eq!(again.header, itc.header);
        assert_eq!(again.states, itc.states);
        assert_eq!(again.covariances, itc.covariances);
    }

    #[test]
    #[parallel]
    fn test_write_derives_header_fields_from_states() {
        let mut itc = ITC::new(ITCHeader::new().with_ephemeris_source("unit"));
        for i in 0..3 {
            itc.push_state(ITCStateVector::new(
                utc(2026, 9, 11, 1, 42, 42.0) + 30.0 * i as f64,
                [7.0e6, 1.0e3, -2.0e3],
                [1.0, 7.5e3, -3.0],
            ))
            .unwrap();
        }
        let text = itc.to_string().unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "created:");
        assert_eq!(
            lines[1],
            "ephemeris_start:2026-09-11 01:42:42 UTC ephemeris_stop:2026-09-11 01:43:42 UTC step_size:30"
        );
        assert_eq!(lines[2], "ephemeris_source:unit");
        assert_eq!(lines[3], "UVW");
        assert_eq!(
            lines[4],
            "2026254014242.000 7000.0000000000 1.0000000000 -2.0000000000 0.0010000000 7.5000000000 -0.0030000000"
        );
        assert_eq!(lines.len(), 7);
    }

    #[test]
    #[parallel]
    fn test_write_rounds_step_size_to_microseconds() {
        let mut itc = ITC::new(ITCHeader::new());
        itc.push_state(ITCStateVector::new(
            utc(2026, 9, 11, 1, 42, 42.0),
            [7.0e6, 0.0, 0.0],
            [0.0, 7.5e3, 0.0],
        ))
        .unwrap();
        itc.push_state(ITCStateVector::new(
            utc(2026, 9, 11, 1, 42, 42.0) + (30.0 + 1.0e-9),
            [7.0e6, 0.0, 0.0],
            [0.0, 7.5e3, 0.0],
        ))
        .unwrap();
        let text = itc.to_string().unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[1].ends_with("step_size:30"));
    }

    #[test]
    #[parallel]
    fn test_write_fractional_step_and_other_frames() {
        let mut itc = ITC::new(ITCHeader::new().with_covariance_frame(ITCCovarianceFrame::ITRF));
        itc.push_state_with_covariance(
            ITCStateVector::new(
                utc(2026, 1, 1, 0, 0, 0.0),
                [7.0e6, 0.0, 0.0],
                [0.0, 7.5e3, 0.0],
            ),
            SMatrix::<f64, 6, 6>::identity() * 1.0e6,
        )
        .unwrap();
        itc.push_state_with_covariance(
            ITCStateVector::new(
                utc(2026, 1, 1, 0, 0, 2.5),
                [7.0e6, 0.0, 0.0],
                [0.0, 7.5e3, 0.0],
            ),
            SMatrix::<f64, 6, 6>::identity() * 1.0e6,
        )
        .unwrap();
        let text = itc.to_string().unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[1].ends_with("step_size:2.5"));
        assert_eq!(lines[3], "ITRF");
        assert_eq!(
            lines[5],
            "1.0000000000e+00 0.0000000000e+00 1.0000000000e+00 0.0000000000e+00 0.0000000000e+00 1.0000000000e+00 0.0000000000e+00"
        );
        assert_eq!(lines.len(), 4 + 2 * 4);
    }

    #[test]
    #[parallel]
    fn test_write_empty_is_error_and_to_file_writes() {
        assert!(ITC::new(ITCHeader::new()).to_string().is_err());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.txt");
        let itc = ITC::from_file(TRUNCATED).unwrap();
        itc.to_file(&path).unwrap();
        let reread = ITC::from_file(&path).unwrap();
        assert_eq!(reread.states, itc.states);
        assert!(reread.source_name.is_none());

        let missing_dir = dir.path().join("no-such-dir").join("out.txt");
        assert!(itc.to_file(&missing_dir).is_err());
    }

    #[test]
    #[parallel]
    fn test_write_rejects_partial_covariance() {
        let mut itc = ITC::new(ITCHeader::new());
        itc.push_state_with_covariance(
            ITCStateVector::new(
                utc(2026, 9, 11, 1, 42, 42.0),
                [7.0e6, 0.0, 0.0],
                [0.0, 7.5e3, 0.0],
            ),
            SMatrix::<f64, 6, 6>::identity(),
        )
        .unwrap();
        itc.push_state_with_covariance(
            ITCStateVector::new(
                utc(2026, 9, 11, 1, 43, 42.0),
                [7.0e6, 0.0, 0.0],
                [0.0, 7.5e3, 0.0],
            ),
            SMatrix::<f64, 6, 6>::identity(),
        )
        .unwrap();
        itc.states.push(ITCStateVector::new(
            utc(2026, 9, 11, 1, 44, 42.0),
            [7.0e6, 0.0, 0.0],
            [0.0, 7.5e3, 0.0],
        ));
        assert!(itc.to_string().is_err());
    }

    #[test]
    #[parallel]
    fn test_write_rejects_non_finite_state() {
        let mut itc = ITC::new(ITCHeader::new());
        itc.push_state(ITCStateVector::new(
            utc(2026, 9, 11, 1, 42, 42.0),
            [7.0e6, 0.0, 0.0],
            [0.0, 7.5e3, 0.0],
        ))
        .unwrap();
        itc.states.push(ITCStateVector::new(
            utc(2026, 9, 11, 1, 43, 42.0),
            [f64::INFINITY, 0.0, 0.0],
            [0.0, 7.5e3, 0.0],
        ));
        assert!(itc.to_string().is_err());
    }

    #[test]
    #[parallel]
    fn test_write_rejects_asymmetric_covariance() {
        let mut itc = ITC::new(ITCHeader::new());
        itc.push_state_with_covariance(
            ITCStateVector::new(
                utc(2026, 9, 11, 1, 42, 42.0),
                [7.0e6, 0.0, 0.0],
                [0.0, 7.5e3, 0.0],
            ),
            SMatrix::<f64, 6, 6>::identity(),
        )
        .unwrap();
        itc.covariances[0][(0, 1)] = 1.0;
        itc.covariances[0][(1, 0)] = 2.0;
        assert!(itc.to_string().is_err());
    }

    #[test]
    #[parallel]
    fn test_write_rejects_colliding_rounded_epochs() {
        let mut colliding = ITC::new(ITCHeader::new());
        colliding
            .push_state(ITCStateVector::new(
                utc(2026, 9, 11, 1, 42, 42.0),
                [7.0e6, 0.0, 0.0],
                [0.0, 7.5e3, 0.0],
            ))
            .unwrap();
        colliding
            .push_state(ITCStateVector::new(
                utc(2026, 9, 11, 1, 42, 42.0) + 0.0001,
                [7.0e6, 0.0, 0.0],
                [0.0, 7.5e3, 0.0],
            ))
            .unwrap();
        assert!(colliding.to_string().is_err());

        let mut distinct = ITC::new(ITCHeader::new());
        distinct
            .push_state(ITCStateVector::new(
                utc(2026, 9, 11, 1, 42, 42.0),
                [7.0e6, 0.0, 0.0],
                [0.0, 7.5e3, 0.0],
            ))
            .unwrap();
        distinct
            .push_state(ITCStateVector::new(
                utc(2026, 9, 11, 1, 42, 42.0) + 0.001,
                [7.0e6, 0.0, 0.0],
                [0.0, 7.5e3, 0.0],
            ))
            .unwrap();
        assert!(distinct.to_string().is_ok());
    }
}
