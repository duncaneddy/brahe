//! SP3-d text writer.
//!
//! Formats an [`SP3`] as SP3-d text following the fixed columns of the
//! format specification. SI values are converted to the file units (km,
//! dm/s, µs, 1e-4 µs/s, and the standard deviation units) here.

use std::fmt::Write as _;
use std::path::Path;

use super::parse::{
    ACCURACY_BASE, BAD_CLOCK_VALUE, CLOCK_RATE_TO_S_PER_S, CORRELATION_CLOCK_STD_TOO_LARGE,
    CORRELATION_SCALE, CORRELATION_STD_TOO_LARGE, DM_PER_S_TO_M_PER_S, KM_TO_M, MM_TO_M,
    POSITION_STD_UNITS, SATELLITES_PER_LINE, StdUnits, US_TO_S, VELOCITY_STD_UNITS, sp3_error,
};
use super::types::{
    MAX_SATELLITES, SP3, SP3Correlation, SP3Header, SP3Record, SP3SatelliteId, check_header_labels,
    epoch_grid_error, header_time_fields,
};
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

/// Maximum number of characters of a comment after the `/* ` prefix.
const MAX_COMMENT_LENGTH: usize = 77;
/// Minimum number of `+` and of `++` lines.
const MIN_SATELLITE_LINES: usize = 5;
/// Minimum number of `/*` lines.
const MIN_COMMENT_LINES: usize = 4;
/// Clock and clock-rate value written for an unknown clock.
const BAD_CLOCK: f64 = 999_999.999_999;
/// Width of the component standard deviation exponent fields of `P` and `V` lines.
const COMPONENT_EXPONENT_WIDTH: u32 = 2;
/// Width of the clock standard deviation exponent field of `P` and `V` lines.
const CLOCK_EXPONENT_WIDTH: u32 = 3;
/// Largest magnitude of a correlation coefficient in units of 1e-7; the
/// coefficient columns hold values in [-0.9999999, 0.9999999].
const MAX_CORRELATION: f64 = 9_999_999.0;
/// Width and precision of the `%f` position/velocity base field.
const POSITION_BASE_FIELD: (usize, usize) = (10, 7);
/// Width and precision of the `%f` clock/clock-rate base field.
const CLOCK_BASE_FIELD: (usize, usize) = (12, 9);
/// Largest `++` accuracy exponent; the field is I3 and 0 means unknown.
const MAX_ACCURACY_EXPONENT: f64 = 999.0;
/// Distance from an integer within which a computed exponent is taken as
/// that integer, so a value read as `base^n` is written back as `n`.
const EXPONENT_SNAP: f64 = 1.0e-9;
/// `%c` placeholder line.
const TIME_SYSTEM_PLACEHOLDER: &str =
    "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc";
/// `%i` placeholder line.
const CUSTOM_PARAMETERS_LINE: &str = "%i    0    0    0    0      0      0      0      0         0";

impl SP3 {
    /// Format the file as SP3-d text.
    ///
    /// The file is always written as version d, whatever version it was
    /// read as. The start epoch, number of epochs, GPS week, seconds of
    /// week, MJD, and fractional day of the header are computed from
    /// [`SP3::epochs`] in the file's time system; the stored header values
    /// are not used and `self` is not modified. Epochs are written in the
    /// file's time system. A satellite without a record at an epoch is
    /// written as the bad value: a zero position with the clock
    /// `999999.999999` (and a zero velocity with the same clock rate when
    /// the file has velocities). An unknown clock or clock rate is written as
    /// `999999.999999`. `V` and `EV` lines are written only when
    /// `has_velocity` is set. An unknown (`None`) correlation coefficient is
    /// written as blank columns.
    ///
    /// Standard deviations are written as the exponent
    /// `ceil(log_base(sigma))` of the `%f` base as written in the file. An
    /// exponent that reaches the field limit, and infinity, are written as
    /// the "too large" marker (99, 999, 9999, or 9999999), so the written
    /// value never understates the stored one. Accuracies are written as `ceil(log2(sigma_mm))`, at least 1.
    /// The header labels are right-aligned in their fixed columns of the first
    /// line: data used and coordinate system 5 characters, orbit type 3, and
    /// agency 4. A longer label, such as the coordinate system `ITRF14` of
    /// some ILRS files, is an error; shorten it with [`SP3::header_mut`]
    /// before writing. Comment lines are padded to four.
    ///
    /// # Returns
    ///
    /// * `Result<String, BraheError>` - The SP3-d text with `\n` line endings, or an error
    ///   when the file cannot be represented: no epochs, more than 999 satellites, a time system
    ///   without a Brahe equivalent (`GLO`, `QZS`, `IRN`), an epoch interval that rounds to
    ///   zero or does not fit its 14-column field at 8 decimal places, epochs that are not
    ///   ascending or not on the grid of the interval as written, both as written at 1e-8 s, a record table that does not match the epochs and
    ///   satellites, an accuracy list that does not match the satellites, a comment longer
    ///   than 77 characters or containing a line break, an empty or whitespace-holding
    ///   header label, a header label longer than its field, a satellite number outside
    ///   1..=99, a `%f` base that is not finite or not greater than 1 as written, a standard
    ///   deviation whose `%f` base is not set, a negative or NaN standard deviation, a clock
    ///   or clock rate that would read back as the missing-value marker `999999.999999`,
    ///   a correlation coefficient that is NaN or outside [-0.9999999, 0.9999999] on the
    ///   1e-7 grid, or a value that does not fit its column
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let sat = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
    /// let options = SP3WriteOptions::default().with_velocity(false);
    /// let mut sp3 = SP3::new(vec![sat], 900.0, options).unwrap();
    /// let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let k = sp3.push_epoch(t0).unwrap();
    /// sp3.set_record(sat, k, SP3Record::new([1.5e7, 2.0e7, 5.0e6]).with_clock(2.5e-5)).unwrap();
    ///
    /// let text = sp3.to_string().unwrap();
    /// assert!(text.starts_with("#dP2024 11 10  0  0  0.00000000       1 ORBIT IGS20 EXT BRHE\n"));
    /// assert!(text.contains("\nPG01  15000.000000  20000.000000   5000.000000     25.000000\n"));
    /// assert!(text.ends_with("EOF\n"));
    /// ```
    pub fn to_string(&self) -> Result<String, BraheError> {
        let time_system = self.check_writable()?;
        let header = self.header();
        let bases = Bases::new(header)?;
        let mut out = String::new();
        self.write_header(&mut out, time_system, &bases)?;
        for (k, (epoch, row)) in self.epochs().iter().zip(self.rows()).enumerate() {
            let (year, month, day, hour, minute, second) = calendar_fields(*epoch, time_system);
            push_line(
                &mut out,
                &format!(
                    "*  {:4} {:2} {:2} {:2} {:2} {:11.8}",
                    year, month, day, hour, minute, second
                ),
            );
            for (id, record) in self.satellite_ids().iter().zip(row) {
                write_record(&mut out, header.has_velocity, &bases, *id, record.as_ref()).map_err(
                    |e| match e {
                        BraheError::ParseError(msg) => sp3_error(&format!(
                            "{} at epoch {}: {}",
                            id,
                            k,
                            msg.strip_prefix("SP3: ").unwrap_or(&msg)
                        )),
                        other => other,
                    },
                )?;
            }
        }
        out.push_str("EOF\n");
        Ok(out)
    }

    /// Write the file to disk as SP3-d text.
    ///
    /// The text is produced by [`SP3::to_string`].
    ///
    /// # Arguments
    ///
    /// * `path` - Destination file path
    ///
    /// # Returns
    ///
    /// * `Result<(), BraheError>` - An error from [`SP3::to_string`], or an `IoError` if
    ///   the file cannot be written
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{GNSSSystem, SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let sat = SP3SatelliteId::new(GNSSSystem::GPS, 1).unwrap();
    /// let mut sp3 = SP3::new(vec![sat], 900.0, SP3WriteOptions::default()).unwrap();
    /// let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// sp3.push_epoch(t0).unwrap();
    /// sp3.set_record(sat, 0, SP3Record::new([1.5e7, 2.0e7, 5.0e6])).unwrap();
    ///
    /// let path = std::env::temp_dir().join("brahe_sp3_to_file_example.sp3");
    /// sp3.to_file(&path).unwrap();
    /// assert_eq!(SP3::from_file(&path).unwrap().num_epochs(), 1);
    /// std::fs::remove_file(&path).unwrap();
    /// ```
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), BraheError> {
        let path = path.as_ref();
        let text = self.to_string()?;
        std::fs::write(path, text).map_err(|e| {
            BraheError::IoError(format!("SP3: failed to write {}: {}", path.display(), e))
        })
    }

    /// Check the parts of the file that the writer cannot express and
    /// return the Brahe time system of the epochs.
    fn check_writable(&self) -> Result<TimeSystem, BraheError> {
        let header = self.header();
        let satellites = self.num_satellites();
        if satellites > MAX_SATELLITES {
            return Err(sp3_error(&format!(
                "{} satellites exceeds the maximum of {}",
                satellites, MAX_SATELLITES
            )));
        }
        for id in self.satellite_ids() {
            SP3SatelliteId::new(id.system, id.number)?;
        }
        let time_system = header.time_system.to_time_system().ok_or_else(|| {
            sp3_error(&format!(
                "time system {} has no Brahe equivalent",
                header.time_system
            ))
        })?;
        if header.accuracies.len() != satellites {
            return Err(sp3_error(&format!(
                "header has {} accuracies for {} satellites",
                header.accuracies.len(),
                satellites
            )));
        }
        let interval = header.epoch_interval;
        if !(interval.is_finite() && interval > 0.0) {
            return Err(sp3_error(&format!(
                "epoch interval must be positive and finite, got {}",
                interval
            )));
        }
        let written = format!("{:14.8}", interval);
        if written.len() > 14 {
            return Err(sp3_error(&format!(
                "epoch interval {} s does not fit the 14-column ## field",
                interval
            )));
        }
        let interval: f64 = written.trim().parse().unwrap_or(0.0);
        if interval <= 0.0 {
            return Err(sp3_error(&format!(
                "epoch interval {} s rounds to zero at 8 decimal places",
                header.epoch_interval
            )));
        }
        let epochs = self.epochs();
        if epochs.is_empty() {
            return Err(sp3_error(
                "file has no epochs; an SP3 file needs at least one epoch",
            ));
        }
        let written: Vec<Epoch> = epochs
            .iter()
            .map(|epoch| written_epoch(*epoch, time_system))
            .collect();
        for pair in written.windows(2) {
            if let Some(detail) = epoch_grid_error(written[0], pair[0], pair[1], interval) {
                return Err(sp3_error(&detail));
            }
        }
        let rows = self.rows();
        if rows.len() != epochs.len() {
            return Err(sp3_error(&format!(
                "{} record rows for {} epochs",
                rows.len(),
                epochs.len()
            )));
        }
        for (k, row) in rows.iter().enumerate() {
            if row.len() != satellites {
                return Err(sp3_error(&format!(
                    "epoch {} has {} records for {} satellites",
                    k,
                    row.len(),
                    satellites
                )));
            }
        }
        for (k, comment) in header.comments.iter().enumerate() {
            if comment.contains(['\n', '\r']) {
                return Err(sp3_error(&format!(
                    "comment {} contains a line break",
                    k + 1
                )));
            }
            if comment.trim_end().chars().count() > MAX_COMMENT_LENGTH {
                return Err(sp3_error(&format!(
                    "comment {} is longer than {} characters",
                    k + 1,
                    MAX_COMMENT_LENGTH
                )));
            }
        }
        Ok(time_system)
    }

    /// Write every line before the first epoch.
    fn write_header(
        &self,
        out: &mut String,
        time_system: TimeSystem,
        bases: &Bases,
    ) -> Result<(), BraheError> {
        let header = self.header();
        let start = self.start_epoch();
        let (year, month, day, hour, minute, second) = calendar_fields(start, time_system);
        check_header_labels(
            &header.data_used,
            &header.coordinate_system,
            &header.orbit_type,
            &header.agency,
            "shorten it with SP3::header_mut (in Python, assign a modified SP3.header)",
        )?;
        let orbit_type = header.orbit_type.to_string();
        let first = format!(
            "#d{}{:4} {:2} {:2} {:2} {:2} {:11.8} {:7} {:>5} {:>5} {:>3} {:>4}",
            if header.has_velocity { 'V' } else { 'P' },
            year,
            month,
            day,
            hour,
            minute,
            second,
            self.num_epochs(),
            header.data_used,
            header.coordinate_system,
            orbit_type,
            header.agency
        );
        push_line(out, &first);

        let (gps_week, seconds_of_week, mjd, fractional_day) =
            header_time_fields(&written_epoch(start, time_system), time_system);
        push_line(
            out,
            &format!(
                "## {:4} {:15.8} {:14.8} {:5} {:15.13}",
                gps_week, seconds_of_week, header.epoch_interval, mjd, fractional_day
            ),
        );

        let satellites = self.satellite_ids();
        let lines = satellites
            .len()
            .div_ceil(SATELLITES_PER_LINE)
            .max(MIN_SATELLITE_LINES);
        for line in 0..lines {
            let mut text = if line == 0 {
                format!("+  {:3}   ", satellites.len())
            } else {
                "+        ".to_string()
            };
            for slot in 0..SATELLITES_PER_LINE {
                match satellites.get(line * SATELLITES_PER_LINE + slot) {
                    Some(id) => write!(text, "{}", id).unwrap(),
                    None => text.push_str("  0"),
                }
            }
            push_line(out, &text);
        }
        for line in 0..lines {
            let mut text = "++       ".to_string();
            for slot in 0..SATELLITES_PER_LINE {
                let k = line * SATELLITES_PER_LINE + slot;
                let exponent = match header.accuracies.get(k).copied().flatten() {
                    None => 0,
                    Some(accuracy) => accuracy_exponent(accuracy).ok_or_else(|| {
                        sp3_error(&format!(
                            "accuracy {} of {} must be positive and finite",
                            accuracy, satellites[k]
                        ))
                    })?,
                };
                write!(text, "{:3}", exponent).unwrap();
            }
            push_line(out, &text);
        }

        push_line(
            out,
            &format!(
                "%c {:<2} cc {:3} ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
                header.file_type.code(),
                header.time_system
            ),
        );
        push_line(out, TIME_SYSTEM_PLACEHOLDER);
        push_line(
            out,
            &format!(
                "%f {} {} {:14.11} {:18.15}",
                bases.position_text, bases.clock_text, 0.0, 0.0
            ),
        );
        push_line(
            out,
            &format!("%f {:10.7} {:12.9} {:14.11} {:18.15}", 0.0, 0.0, 0.0, 0.0),
        );
        push_line(out, CUSTOM_PARAMETERS_LINE);
        push_line(out, CUSTOM_PARAMETERS_LINE);

        for comment in &header.comments {
            push_line(out, &format!("/* {}", comment));
        }
        for _ in header.comments.len()..MIN_COMMENT_LINES {
            push_line(out, "/*");
        }
        Ok(())
    }
}

/// `%f` bases as written in the file.
///
/// The exponents of the standard deviations are computed with the base
/// values that the written text reads back as, so a re-read file decodes
/// them consistently.
struct Bases {
    /// Position/velocity base as written, `None` if the file declares none
    position: Option<f64>,
    /// Clock/clock-rate base as written, `None` if the file declares none
    clock: Option<f64>,
    /// `F10.7` text of the position/velocity base
    position_text: String,
    /// `F12.9` text of the clock/clock-rate base
    clock_text: String,
}

impl Bases {
    /// Format the bases of a header, checking that each declared base is
    /// finite and greater than 1 both before and after formatting.
    fn new(header: &SP3Header) -> Result<Self, BraheError> {
        let (position_text, position) = base_field(
            header.position_base,
            POSITION_BASE_FIELD,
            "position/velocity",
        )?;
        let (clock_text, clock) = base_field(header.clock_base, CLOCK_BASE_FIELD, "clock")?;
        Ok(Bases {
            position,
            clock,
            position_text,
            clock_text,
        })
    }
}

/// Text of a `%f` base field and the base value the text reads back as.
/// `None` is written as zero, which reads back as no base.
fn base_field(
    base: Option<f64>,
    (width, precision): (usize, usize),
    name: &str,
) -> Result<(String, Option<f64>), BraheError> {
    let what = format!("{} base", name);
    let Some(base) = base else {
        return Ok((fixed(0.0, width, precision, &what)?, None));
    };
    if !(base.is_finite() && base > 1.0) {
        return Err(sp3_error(&format!(
            "{} {} must be finite and greater than 1",
            what, base
        )));
    }
    let text = fixed(base, width, precision, &what)?;
    let written: f64 = text.trim().parse().unwrap_or(f64::NAN);
    if written.is_nan() || written <= 1.0 {
        return Err(sp3_error(&format!(
            "{} {} is written as {}, which is not greater than 1",
            what,
            base,
            text.trim()
        )));
    }
    Ok((text, Some(written)))
}

/// Append a line with trailing whitespace removed and a `\n` ending.
fn push_line(out: &mut String, line: &str) {
    out.push_str(line.trim_end());
    out.push('\n');
}

/// Calendar fields of an epoch in a time system, with the second rounded
/// to the 1e-8 s resolution of the format. A second that rounds up to 60 is
/// carried into the minute.
fn calendar_fields(epoch: Epoch, time_system: TimeSystem) -> (u32, u8, u8, u8, u8, f64) {
    let (mut year, mut month, mut day, mut hour, mut minute, mut second, mut nanosecond) =
        epoch.to_datetime_as_time_system(time_system);
    // Units of 1e-8 s within the second.
    let mut ticks = (nanosecond / 10.0).round();
    if ticks >= 1.0e8 {
        (year, month, day, hour, minute, second, nanosecond) =
            (epoch + (1.0e9 - nanosecond) * 1.0e-9).to_datetime_as_time_system(time_system);
        ticks = (nanosecond / 10.0).round().min(1.0e8 - 1.0);
    }
    (year, month, day, hour, minute, second + ticks * 1.0e-8)
}

/// Epoch as written on an epoch line: its calendar fields in `time_system`
/// with seconds rounded to 1e-8 s, read back as the parser reads them.
fn written_epoch(epoch: Epoch, time_system: TimeSystem) -> Epoch {
    let (year, month, day, hour, minute, second) = calendar_fields(epoch, time_system);
    Epoch::from_datetime(year, month, day, hour, minute, second, 0.0, time_system)
}

/// Exponent `n` of the smallest `base^n` not below `value`.
///
/// A value within 1e-9 of an exact power gives that power, so values read
/// from a file are written back unchanged.
fn ceil_exponent(value: f64, base: f64) -> f64 {
    let exponent = value.ln() / base.ln();
    let nearest = exponent.round();
    if (exponent - nearest).abs() <= EXPONENT_SNAP {
        nearest
    } else {
        exponent.ceil()
    }
}

/// `++` exponent of an accuracy in meters, or `None` if the accuracy is not
/// positive and finite. Accuracies of 1 mm or better are written as 1.
fn accuracy_exponent(accuracy: f64) -> Option<i32> {
    if !(accuracy.is_finite() && accuracy > 0.0) {
        return None;
    }
    let exponent = ceil_exponent(accuracy / MM_TO_M, ACCURACY_BASE);
    Some(exponent.clamp(1.0, MAX_ACCURACY_EXPONENT) as i32)
}

/// Right-aligned `F{width}.{precision}` field.
fn fixed(value: f64, width: usize, precision: usize, what: &str) -> Result<String, BraheError> {
    let text = format!("{:width$.precision$}", value);
    if value.is_finite() && text.len() <= width {
        Ok(text)
    } else {
        Err(sp3_error(&format!(
            "{} {} does not fit the F{}.{} field",
            what, value, width, precision
        )))
    }
}

/// Exponent field of a `P` or `V` standard deviation: blanks for `None`,
/// and otherwise the exponent clamped to the range of the field. Infinity
/// and any exponent reaching `10^width - 1` are written as that "too large"
/// marker, so the written standard deviation is never below the stored one.
///
/// # Arguments
///
/// * `std` - Standard deviation. Units: SI
/// * `base` - `%f` base of the exponent as written, greater than 1, or `None` if the file
///   declares none
/// * `scale` - File unit of the standard deviation in SI
/// * `width` - Width of the field
/// * `what` - Name of the quantity, for errors
/// * `base_name` - Name of the `%f` base, for errors
fn exponent_field(
    std: Option<f64>,
    base: Option<f64>,
    scale: f64,
    width: u32,
    what: &str,
    base_name: &str,
) -> Result<String, BraheError> {
    let width_chars = width as usize;
    let Some(std) = std else {
        return Ok(" ".repeat(width_chars));
    };
    let base = base.ok_or_else(|| {
        sp3_error(&format!(
            "{} standard deviation is set but the header declares no {} base",
            what, base_name
        ))
    })?;
    let too_large = 10_i32.pow(width) - 1;
    let exponent = if std == f64::INFINITY {
        too_large
    } else {
        let exponent = ceil_exponent(std / scale, base);
        if std < 0.0 || exponent.is_nan() {
            return Err(sp3_error(&format!(
                "{} standard deviation {} cannot be written with base {}",
                what, std, base
            )));
        }
        let smallest = 1 - 10_i32.pow(width - 1);
        exponent.clamp(f64::from(smallest), f64::from(too_large)) as i32
    };
    Ok(format!("{:width$}", exponent, width = width_chars))
}

/// Integer standard deviation field of an `EP` or `EV` line: blanks for
/// `None`, `too_large` for infinity or a value that reaches it.
fn integer_std_field(
    std: Option<f64>,
    scale: f64,
    too_large: i64,
    width: usize,
    what: &str,
) -> Result<String, BraheError> {
    let Some(std) = std else {
        return Ok(" ".repeat(width));
    };
    if std.is_nan() || std < 0.0 {
        return Err(sp3_error(&format!(
            "{} standard deviation {} is not a non-negative number",
            what, std
        )));
    }
    let value = (std / scale).round().min(too_large as f64) as i64;
    Ok(format!("{:width$}", value, width = width))
}

/// Clock or clock-rate field in file units. `None` is the missing-value
/// marker; a present value whose text the parser would read as that marker
/// is an error.
fn clock_field(clock: Option<f64>, name: &str) -> Result<String, BraheError> {
    let Some(value) = clock else {
        return fixed(BAD_CLOCK, 14, 6, name);
    };
    let text = fixed(value, 14, 6, name)?;
    if text
        .trim()
        .parse::<f64>()
        .is_ok_and(|v| v >= BAD_CLOCK_VALUE)
    {
        return Err(sp3_error(&format!(
            "{} {} in file units reaches the missing-value marker {} and would read back as missing",
            name,
            text.trim(),
            BAD_CLOCK_VALUE
        )));
    }
    Ok(text)
}

/// Shared layout of `P` and `V` lines up to column 73.
#[allow(clippy::too_many_arguments)]
fn state_line(
    tag: char,
    id: SP3SatelliteId,
    values: [f64; 3],
    clock: Option<f64>,
    clock_name: &str,
    std: [Option<f64>; 3],
    clock_std: Option<f64>,
    bases: &Bases,
    units: StdUnits,
) -> Result<String, BraheError> {
    let mut line = format!("{}{}", tag, id);
    for (axis, value) in ["x", "y", "z"].iter().zip(values) {
        line.push_str(&fixed(value, 14, 6, axis)?);
    }
    line.push_str(&clock_field(clock, clock_name)?);
    for s in std {
        line.push(' ');
        line.push_str(&exponent_field(
            s,
            bases.position,
            units.component,
            COMPONENT_EXPONENT_WIDTH,
            "position/velocity",
            "position/velocity",
        )?);
    }
    line.push(' ');
    line.push_str(&exponent_field(
        clock_std,
        bases.clock,
        units.clock,
        CLOCK_EXPONENT_WIDTH,
        "clock",
        "clock",
    )?);
    Ok(line)
}

/// `EP` or `EV` line. `None` coefficients are blank columns.
fn correlation_line(
    tag: &str,
    correlation: &SP3Correlation,
    units: StdUnits,
) -> Result<String, BraheError> {
    let mut line = format!("{}  ", tag);
    for (k, s) in correlation.std.iter().enumerate() {
        if k > 0 {
            line.push(' ');
        }
        line.push_str(&integer_std_field(
            *s,
            units.component,
            CORRELATION_STD_TOO_LARGE,
            4,
            "correlation record",
        )?);
    }
    line.push(' ');
    line.push_str(&integer_std_field(
        correlation.clock_std,
        units.clock,
        CORRELATION_CLOCK_STD_TOO_LARGE,
        7,
        "correlation record clock",
    )?);
    let coefficients = [
        correlation.xy,
        correlation.xz,
        correlation.xc,
        correlation.yz,
        correlation.yc,
        correlation.zc,
    ];
    for c in coefficients {
        let Some(c) = c else {
            line.push_str(&" ".repeat(9));
            continue;
        };
        if c.is_nan() {
            return Err(sp3_error("correlation coefficient is NaN"));
        }
        let value = (c / CORRELATION_SCALE).round();
        if value.abs() > MAX_CORRELATION {
            return Err(sp3_error(&format!(
                "correlation coefficient {} is outside [-0.9999999, 0.9999999]",
                c
            )));
        }
        write!(line, " {:8}", value as i64).unwrap();
    }
    Ok(line)
}

/// Write the `P`, `EP`, `V`, and `EV` lines of one satellite at one epoch.
/// A missing record is written as the bad value.
fn write_record(
    out: &mut String,
    has_velocity: bool,
    bases: &Bases,
    id: SP3SatelliteId,
    record: Option<&SP3Record>,
) -> Result<(), BraheError> {
    let absent = SP3Record::default();
    let record = record.unwrap_or(&absent);
    let mut line = state_line(
        'P',
        id,
        record.position.map(|p| p / KM_TO_M),
        record.clock.map(|c| c / US_TO_S),
        "clock",
        record.position_std,
        record.clock_std,
        bases,
        POSITION_STD_UNITS,
    )?;
    let flag = |set: bool, c: char| if set { c } else { ' ' };
    write!(
        line,
        " {}{}  {}{}",
        flag(record.clock_event, 'E'),
        flag(record.clock_predicted, 'P'),
        flag(record.maneuver, 'M'),
        flag(record.orbit_predicted, 'P')
    )
    .unwrap();
    push_line(out, &line);
    if let Some(correlation) = &record.position_correlation {
        push_line(
            out,
            &correlation_line("EP", correlation, POSITION_STD_UNITS)?,
        );
    }

    if has_velocity {
        let line = state_line(
            'V',
            id,
            record
                .velocity
                .unwrap_or([0.0; 3])
                .map(|v| v / DM_PER_S_TO_M_PER_S),
            record.clock_rate.map(|r| r / CLOCK_RATE_TO_S_PER_S),
            "clock rate",
            record.velocity_std,
            record.clock_rate_std,
            bases,
            VELOCITY_STD_UNITS,
        )?;
        push_line(out, &line);
        if let Some(correlation) = &record.velocity_correlation {
            push_line(
                out,
                &correlation_line("EV", correlation, VELOCITY_STD_UNITS)?,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::datafiles::sp3::types::header_time_fields;
    use crate::datafiles::sp3::{
        GNSSSystem, SP3Correlation, SP3OrbitType, SP3Record, SP3SatelliteId, SP3TimeSystem,
        SP3Version, SP3WriteOptions,
    };
    use crate::time::{Epoch, TimeSystem};
    use approx::assert_abs_diff_eq;
    use rstest::rstest;
    use serial_test::parallel;

    fn fixture(name: &str) -> SP3 {
        SP3::from_file(format!("test_assets/sp3/{name}")).unwrap()
    }

    fn sat(code: &str) -> SP3SatelliteId {
        SP3SatelliteId::parse(code).unwrap()
    }

    fn write_error(sp3: &SP3) -> String {
        match sp3.to_string() {
            Err(BraheError::ParseError(msg)) => msg,
            other => panic!("expected a write error, got {other:?}"),
        }
    }

    fn gps_t0() -> Epoch {
        Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS)
    }

    /// File with satellites G01 and G02 at two 900 s epochs, G01 present at both.
    fn two_satellite_file(options: SP3WriteOptions) -> SP3 {
        let mut sp3 = SP3::new(vec![sat("G01"), sat("G02")], 900.0, options).unwrap();
        for k in 0..2 {
            sp3.push_epoch(gps_t0() + 900.0 * k as f64).unwrap();
            sp3.set_record(
                sat("G01"),
                k,
                SP3Record::new([15_000_000.0, 20_000_000.0, 5_000_000.0])
                    .with_velocity([1000.0, -2000.0, 500.0])
                    .with_clock(2.5e-5)
                    .with_clock_rate(1e-12),
            )
            .unwrap();
        }
        sp3
    }

    fn assert_option_close(read: Option<f64>, written: Option<f64>, tol: f64, what: &str) {
        match (read, written) {
            (None, None) => {}
            (Some(r), Some(w)) => assert!((r - w).abs() <= tol, "{what}: {r} != {w}"),
            _ => panic!("{what}: {read:?} != {written:?}"),
        }
    }

    /// A written standard deviation is the smallest power of the base not
    /// below the read value, so it lies in `[read, read * base)`.
    fn assert_quantized(read: Option<f64>, written: Option<f64>, base: Option<f64>, what: &str) {
        match (read, written) {
            (None, None) => {}
            (Some(r), Some(w)) if r.is_infinite() => assert!(w.is_infinite(), "{what}: {w}"),
            (Some(r), Some(w)) => {
                let base = base.unwrap();
                assert!(
                    w >= r * (1.0 - 1e-12) && w < r * base,
                    "{what}: written {w} outside [{r}, {r} * {base})"
                );
            }
            _ => panic!("{what}: {read:?} != {written:?}"),
        }
    }

    fn assert_correlation_close(
        read: &Option<SP3Correlation>,
        written: &Option<SP3Correlation>,
        what: &str,
    ) {
        match (read, written) {
            (None, None) => {}
            (Some(r), Some(w)) => {
                for k in 0..3 {
                    assert_option_close(r.std[k], w.std[k], 1e-15, what);
                }
                assert_option_close(r.clock_std, w.clock_std, 1e-22, what);
                let pairs = [
                    (r.xy, w.xy),
                    (r.xz, w.xz),
                    (r.xc, w.xc),
                    (r.yz, w.yz),
                    (r.yc, w.yc),
                    (r.zc, w.zc),
                ];
                for (a, b) in pairs {
                    assert_option_close(a, b, 1e-12, what);
                }
            }
            _ => panic!("{what}: {read:?} != {written:?}"),
        }
    }

    fn assert_record_close(
        read: &SP3Record,
        written: &SP3Record,
        pb: Option<f64>,
        cb: Option<f64>,
    ) {
        for k in 0..3 {
            assert_abs_diff_eq!(read.position[k], written.position[k], epsilon = 1e-6);
            assert_quantized(
                read.position_std[k],
                written.position_std[k],
                pb,
                "position std",
            );
            assert_quantized(
                read.velocity_std[k],
                written.velocity_std[k],
                pb,
                "velocity std",
            );
        }
        assert_option_close(read.clock, written.clock, 1e-12, "clock");
        assert_quantized(read.clock_std, written.clock_std, cb, "clock std");
        match (read.velocity, written.velocity) {
            (None, None) => {}
            (Some(r), Some(w)) => {
                for k in 0..3 {
                    assert_abs_diff_eq!(r[k], w[k], epsilon = 1e-7);
                }
            }
            (r, w) => panic!("velocity: {r:?} != {w:?}"),
        }
        assert_option_close(read.clock_rate, written.clock_rate, 1e-16, "clock rate");
        assert_quantized(
            read.clock_rate_std,
            written.clock_rate_std,
            cb,
            "clock rate std",
        );
        assert_eq!(
            (
                read.clock_event,
                read.clock_predicted,
                read.maneuver,
                read.orbit_predicted
            ),
            (
                written.clock_event,
                written.clock_predicted,
                written.maneuver,
                written.orbit_predicted
            )
        );
        assert_correlation_close(
            &read.position_correlation,
            &written.position_correlation,
            "EP",
        );
        assert_correlation_close(
            &read.velocity_correlation,
            &written.velocity_correlation,
            "EV",
        );
    }

    #[rstest]
    #[case("example-a-1.sp3")]
    #[case("example-a-2.sp3")]
    #[case("example-c-1.sp3")]
    #[case("example-c-2.sp3")]
    #[case("example-d-1.sp3")]
    #[case("example-d-2.sp3")]
    #[case("example-d-3.sp3")]
    #[case("example-d-4.sp3")]
    #[case("example-d-5.sp3")]
    #[case("issue895-hours-increment.sp3")]
    #[case("issue895-minutes-increment.sp3")]
    #[case("issue895-second-digits.sp3")]
    #[case("issue895-no-eof.sp3")]
    #[case("issue895-clock-record.sp3")]
    #[case("issue895-header-comment.sp3")]
    #[case("three-hours.sp3")]
    #[case("missing-agency.sp3")]
    #[case("issue1327-136-sats.sp3")]
    #[case("missing-standard-deviation.sp3")]
    #[case("truncated-nsgf.orb.lageos2.160305.v35.sp3")]
    #[case("esaBHN.sp3")]
    #[case("esaPRO.sp3")]
    #[case("unknownType.sp3")]
    #[case("nsgf.orb.stella.v00.sp3")]
    #[case("too-many-comments-a.sp3")]
    #[case("too-long-comment-a.sp3")]
    #[case("exceeded-sat-count.sp3")]
    #[case("gbm19500_after_drop.sp3")]
    #[case("gbm19500_after_no_drop.sp3")]
    #[case("gbm19500_large_gap.sp3")]
    #[case("gbm19500_truncated.sp3")]
    #[case("gbm19500_wrong_agency.sp3")]
    #[case("gbm19500_wrong_coordinate_system.sp3")]
    #[case("gbm19500_wrong_data_used.sp3")]
    #[case("gbm19500_wrong_derivatives.sp3")]
    #[case("gbm19500_wrong_frame.sp3")]
    #[case("gbm19500_wrong_mu.sp3")]
    #[case("gbm19500_wrong_orbit_type.sp3")]
    #[case("gbm19500_wrong_satellite_count.sp3")]
    #[case("gbm19500_wrong_satellite_list.sp3")]
    #[case("gbm19500_wrong_time_system.sp3")]
    #[case("gbm19500_wrong_type.sp3")]
    #[parallel]
    fn test_SP3_round_trip(#[case] f: &str) {
        let read = fixture(f);
        let text = read.to_string().unwrap();
        let written = SP3::from_str(&text).unwrap();

        assert_eq!(written.epochs(), read.epochs());
        assert_eq!(written.satellite_ids(), read.satellite_ids());

        let (rh, wh) = (read.header(), written.header());
        for (k, row) in read.rows().iter().enumerate() {
            for (j, record) in row.iter().enumerate() {
                match (record, &written.rows()[k][j]) {
                    (None, None) => {}
                    (Some(r), Some(w)) => {
                        assert_record_close(r, w, rh.position_base, rh.clock_base)
                    }
                    (r, w) => panic!("epoch {k} satellite {j}: {r:?} != {w:?}"),
                }
            }
        }

        let ts = rh.time_system.to_time_system().unwrap();
        let (week, sow, mjd, fd) = header_time_fields(&read.start_epoch(), ts);
        assert_eq!(wh.version, SP3Version::D);
        assert_eq!(wh.start_epoch, read.start_epoch());
        assert_eq!(wh.number_of_epochs, read.num_epochs());
        assert_eq!((wh.gps_week, wh.mjd), (week, mjd));
        assert_abs_diff_eq!(wh.seconds_of_week, sow, epsilon = 1e-8);
        assert_abs_diff_eq!(wh.fractional_day, fd, epsilon = 1e-13);
        for (r, w) in rh.accuracies.iter().zip(&wh.accuracies) {
            assert_quantized(*r, *w, Some(2.0), "accuracy");
        }

        let mut expected = rh.clone();
        expected.version = SP3Version::D;
        expected.start_epoch = wh.start_epoch;
        expected.number_of_epochs = wh.number_of_epochs;
        expected.gps_week = wh.gps_week;
        expected.seconds_of_week = wh.seconds_of_week;
        expected.mjd = wh.mjd;
        expected.fractional_day = wh.fractional_day;
        expected.accuracies = wh.accuracies.clone();
        if expected.comments.len() < 4 {
            expected.comments.resize(4, String::new());
        }
        assert_eq!(wh, &expected);
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_too_long_comment_fixture() {
        let msg = write_error(&fixture("too-long-comment-d.sp3"));
        assert!(
            msg.contains("comment 3 is longer than 77 characters"),
            "{msg}"
        );
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_layout() {
        let text = fixture("example-d-1.sp3").to_string().unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].starts_with("#dP"), "{}", lines[0]);
        let count = |pred: &dyn Fn(&str) -> bool| lines.iter().filter(|l| pred(l)).count();
        assert!(count(&|l| l.starts_with('+') && !l.starts_with("++")) >= 5);
        assert!(count(&|l| l.starts_with("++")) >= 5);
        assert!(count(&|l| l.starts_with("/*")) >= 4);
        assert_eq!(*lines.last().unwrap(), "EOF");
        assert!(text.ends_with("EOF\n"));
        for line in &lines {
            assert!(line.chars().count() <= 80, "line too long: {line}");
            assert_eq!(line.trim_end(), *line, "trailing whitespace: {line:?}");
        }
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_header_lines() {
        let text = fixture("example-d-2.sp3").to_string().unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines[0],
            "#dV2001  8  8  0  0  0.00000000       2 ORBIT IGS97 HLM MGEX"
        );
        assert_eq!(
            lines[1],
            "## 1126 259200.00000000   900.00000000 52129 0.0000000000000"
        );
        assert_eq!(
            lines[2],
            "+   26   G01G02G03G04G05G06G07G08G09G10G11G13G14G17G18G20G21"
        );
        assert_eq!(
            lines[3],
            "+        G23G24G25G26G27G28G29G30G31  0  0  0  0  0  0  0  0"
        );
        assert_eq!(
            lines[6],
            "+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0"
        );
        assert_eq!(
            lines[7],
            "++         7  8  7  8  6  7  7  7  7  7  7  7  7  8  8  7  9"
        );
        assert_eq!(
            lines[12],
            "%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc"
        );
        assert_eq!(
            lines[13],
            "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc"
        );
        assert_eq!(
            lines[14],
            "%f  1.2500000  1.025000000  0.00000000000  0.000000000000000"
        );
        assert_eq!(
            lines[15],
            "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000"
        );
        assert_eq!(
            lines[16],
            "%i    0    0    0    0      0      0      0      0         0"
        );
        assert_eq!(lines[22], "*  2001  8  8  0  0  0.00000000");
        assert_eq!(
            lines[23],
            "PG01 -11044.805800 -10475.672350  21929.418200    189.163300 18 18 18 219"
        );
        assert_eq!(
            lines[24],
            "EP    55   55   55     222  1234567 -1234567  5999999      -30       21 -1230000"
        );
        assert_eq!(
            lines[25],
            "VG01  20298.880364 -18462.044804   1381.387685     -4.534317 14 14 14 191"
        );
        assert_eq!(
            lines[26],
            "EV    22   22   22     111  1234567  1234567  1234567  1234567  1234567  1234567"
        );
        assert_eq!(
            lines[27],
            "PG02 -12593.593500  10170.327650 -20354.534400    -55.976000 18 18 18 219     M"
        );
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_long_coordinate_system() {
        let mut read = fixture("issue1014-days-increment.sp3");
        assert_eq!(read.header().coordinate_system, "ITRF14");
        let msg = write_error(&read);
        assert!(
            msg.contains("coordinate system 'ITRF14' is longer than 5 characters"),
            "{msg}"
        );
        assert!(msg.contains("header_mut"), "{msg}");

        read.header_mut().coordinate_system = "ITR14".to_string();
        let text = read.to_string().unwrap();
        assert_eq!(
            text.lines().next().unwrap(),
            "#dV2021 12 30 23  0  0.00000000      62   SLR ITR14 FIT JCET"
        );
        let written = SP3::from_str(&text).unwrap();
        assert_eq!(written.header().coordinate_system, "ITR14");
        assert_eq!(written.epochs(), read.epochs());
        assert_eq!(written.satellite_ids(), read.satellite_ids());
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_recomputes_derived_header_fields() {
        let read = fixture("issue1327-136-sats.sp3");
        assert_eq!((read.header().gps_week, read.header().mjd), (0, 0));
        let text = read.to_string().unwrap();
        assert_eq!(
            text.lines().nth(1).unwrap(),
            "## 2302 172800.00000000   300.00000000 60360 0.0000000000000"
        );
        assert_eq!((read.header().gps_week, read.header().mjd), (0, 0));

        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        sp3.header_mut().number_of_epochs = 7;
        sp3.header_mut().start_epoch = gps_t0() + 86400.0;
        let text = sp3.to_string().unwrap();
        assert_eq!(
            text.lines().next().unwrap(),
            "#dV2024 11 10  0  0  0.00000000       2 ORBIT IGS20 EXT BRHE"
        );
        assert_eq!(sp3.header().number_of_epochs, 7);
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_absent_record() {
        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        sp3.set_record(sat("G01"), 1, SP3Record::new([1.0e7, 2.0e7, 3.0e7]))
            .unwrap();
        let text = sp3.to_string().unwrap();
        let lines: Vec<&str> = text.lines().collect();
        let g02: Vec<&str> = lines
            .iter()
            .copied()
            .filter(|l| l.get(1..4) == Some("G02"))
            .collect();
        assert_eq!(
            g02,
            [
                "PG02      0.000000      0.000000      0.000000 999999.999999",
                "VG02      0.000000      0.000000      0.000000 999999.999999",
                "PG02      0.000000      0.000000      0.000000 999999.999999",
                "VG02      0.000000      0.000000      0.000000 999999.999999",
            ]
        );
        assert!(lines.contains(&"PG01  10000.000000  20000.000000  30000.000000 999999.999999"));
        assert!(lines.contains(&"VG01      0.000000      0.000000      0.000000 999999.999999"));

        let back = SP3::from_str(&text).unwrap();
        assert!(back.record(sat("G02"), 0).unwrap().is_none());
        let g01 = back.record(sat("G01"), 1).unwrap().unwrap();
        assert_eq!(
            (g01.clock, g01.velocity, g01.clock_rate),
            (None, None, None)
        );
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_standard_deviations() {
        let mut sp3 = two_satellite_file(SP3WriteOptions::default().with_velocity(false));
        sp3.header_mut().position_base = Some(1.25);
        sp3.header_mut().clock_base = Some(1.025);
        let mut record = SP3Record::new([1.0e7, 2.0e7, 3.0e7]).with_clock(1e-4);
        record.position_std = [
            Some(0.010),
            Some(1.25f64.powi(18) * 1e-3),
            Some(f64::INFINITY),
        ];
        record.clock_std = Some(1.025f64.powi(219) * 1e-12);
        record.clock_event = true;
        record.orbit_predicted = true;
        sp3.set_record(sat("G01"), 0, record).unwrap();
        let text = sp3.to_string().unwrap();
        // 10 mm lies between 1.25^10 = 9.31 mm and 1.25^11 = 11.64 mm.
        assert!(
            text.lines().any(|l| l
                == "PG01  10000.000000  20000.000000  30000.000000    100.000000 11 18 99 219 E    P"),
            "{text}"
        );
    }

    #[test]
    #[parallel]
    fn test_SP3_to_file() {
        let sp3 = two_satellite_file(SP3WriteOptions::default());
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("out.sp3");
        sp3.to_file(&path).unwrap();
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            sp3.to_string().unwrap()
        );
        assert_eq!(
            SP3::from_file(&path).unwrap(),
            SP3::from_str(&sp3.to_string().unwrap()).unwrap()
        );

        match sp3.to_file(dir.path().join("missing").join("out.sp3")) {
            Err(BraheError::IoError(msg)) => assert!(msg.starts_with("SP3: "), "{msg}"),
            other => panic!("expected an I/O error, got {other:?}"),
        }
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_comment_length() {
        let options = |len: usize| SP3WriteOptions::default().with_comments(vec!["C".repeat(len)]);
        let text = two_satellite_file(options(77)).to_string().unwrap();
        let back = SP3::from_str(&text).unwrap();
        assert_eq!(back.header().comments[0], "C".repeat(77));

        let msg = write_error(&two_satellite_file(options(78)));
        assert!(
            msg.contains("comment 1 is longer than 77 characters"),
            "{msg}"
        );
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_standard_deviation_without_base() {
        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        let mut record = SP3Record::new([1.0e7, 2.0e7, 3.0e7]);
        record.position_std[1] = Some(0.01);
        sp3.set_record(sat("G02"), 1, record).unwrap();
        let msg = write_error(&sp3);
        assert!(msg.contains("G02 at epoch 1"), "{msg}");
        assert!(msg.contains("no position/velocity base"), "{msg}");

        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        sp3.header_mut().position_base = Some(1.25);
        let mut record = SP3Record::new([1.0e7, 2.0e7, 3.0e7]);
        record.position_std = [Some(0.01); 3];
        record.clock_rate_std = Some(1e-16);
        sp3.set_record(sat("G02"), 0, record).unwrap();
        let msg = write_error(&sp3);
        assert!(msg.contains("no clock base"), "{msg}");
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_unmapped_time_system() {
        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        sp3.header_mut().time_system = SP3TimeSystem::GLO;
        let msg = write_error(&sp3);
        assert!(msg.contains("time system GLO"), "{msg}");
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_too_many_satellites() {
        let template = two_satellite_file(SP3WriteOptions::default());
        let mut header = template.header().clone();
        let systems = [GNSSSystem::GPS, GNSSSystem::GLONASS, GNSSSystem::Galileo];
        let satellites: Vec<SP3SatelliteId> = (0..1000)
            .map(|k| SP3SatelliteId::new(systems[k % 3], (k % 99) as u16 + 1).unwrap())
            .collect();
        header.accuracies = vec![None; satellites.len()];
        let sp3 = SP3::from_parts(header, Vec::new(), satellites, Vec::new());
        let msg = write_error(&sp3);
        assert!(
            msg.contains("1000 satellites exceeds the maximum of 999"),
            "{msg}"
        );
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_invalid_tables() {
        let template = two_satellite_file(SP3WriteOptions::default());
        let header = template.header().clone();
        let satellites = template.satellite_ids().to_vec();
        let rows = template.rows().to_vec();
        let build = |epochs: Vec<Epoch>, rows: Vec<Vec<Option<SP3Record>>>| {
            SP3::from_parts(header.clone(), epochs, satellites.clone(), rows)
        };

        let off_grid = build(vec![gps_t0(), gps_t0() + 950.0], rows.clone());
        assert!(write_error(&off_grid).contains("is not on the 900 s grid"));

        let descending = build(vec![gps_t0(), gps_t0() - 900.0], rows.clone());
        assert!(write_error(&descending).contains("is not after the last epoch"));

        let mut short_row = rows.clone();
        short_row[1].pop();
        let msg = write_error(&build(vec![gps_t0(), gps_t0() + 900.0], short_row));
        assert!(
            msg.contains("epoch 1 has 1 records for 2 satellites"),
            "{msg}"
        );

        let msg = write_error(&build(vec![gps_t0()], rows.clone()));
        assert!(msg.contains("2 record rows for 1 epochs"), "{msg}");

        let mut bad_interval = template.clone();
        bad_interval.header_mut().epoch_interval = 0.0;
        assert!(write_error(&bad_interval).contains("epoch interval"));

        let mut bad_accuracies = template.clone();
        bad_accuracies.header_mut().accuracies.push(None);
        assert!(write_error(&bad_accuracies).contains("3 accuracies for 2 satellites"));
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_header_label_errors() {
        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        sp3.header_mut().coordinate_system = String::new();
        assert!(write_error(&sp3).contains("coordinate system"));

        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        sp3.header_mut().data_used = "u U".to_string();
        assert!(write_error(&sp3).contains("data used"));

        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        sp3.header_mut().orbit_type = SP3OrbitType::Other("X".repeat(30));
        assert!(write_error(&sp3).contains("longer than 3 characters"));

        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        sp3.header_mut().comments = vec!["first\nsecond".to_string()];
        assert!(write_error(&sp3).contains("line break"));
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_value_errors() {
        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        sp3.set_record(sat("G02"), 0, SP3Record::new([1.0e13, 0.0, 0.0]))
            .unwrap();
        assert!(write_error(&sp3).contains("does not fit"));

        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        sp3.set_record(sat("G02"), 0, SP3Record::new([f64::NAN, 0.0, 0.0]))
            .unwrap();
        assert!(write_error(&sp3).contains("does not fit"));

        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        sp3.header_mut().accuracies[0] = Some(f64::INFINITY);
        assert!(write_error(&sp3).contains("accuracy"));

        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        sp3.header_mut().position_base = Some(1.25);
        let mut record = SP3Record::new([1.0e7, 2.0e7, 3.0e7]);
        record.position_std[0] = Some(-1.0);
        sp3.set_record(sat("G02"), 0, record).unwrap();
        assert!(write_error(&sp3).contains("standard deviation"));
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_correlation_errors() {
        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        sp3.header_mut().position_base = Some(1.25);
        sp3.header_mut().clock_base = Some(1.025);
        let blank = || SP3Correlation {
            std: [None; 3],
            clock_std: None,
            xy: Some(0.0),
            xz: Some(0.0),
            xc: Some(0.0),
            yz: Some(0.0),
            yc: Some(0.0),
            zc: Some(0.0),
        };
        let record = |correlation: SP3Correlation| {
            let mut record = SP3Record::new([1.0e7, 2.0e7, 3.0e7]);
            record.position_correlation = Some(correlation);
            record
        };

        sp3.set_record(sat("G02"), 0, record(blank())).unwrap();
        let text = sp3.to_string().unwrap();
        assert!(text.contains("\nEP "), "{text}");

        let negative = SP3Correlation {
            std: [Some(-0.1), None, None],
            ..blank()
        };
        sp3.set_record(sat("G02"), 0, record(negative)).unwrap();
        assert!(write_error(&sp3).contains("not a non-negative number"));

        let not_a_number = SP3Correlation {
            xy: Some(f64::NAN),
            ..blank()
        };
        sp3.set_record(sat("G02"), 0, record(not_a_number)).unwrap();
        assert!(write_error(&sp3).contains("correlation coefficient is NaN"));
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_epoch_second_rounding() {
        let mut sp3 = SP3::new(vec![sat("G01")], 30.0, SP3WriteOptions::default()).unwrap();
        let t0 = gps_t0() + 59.999_999_999_9;
        sp3.push_epoch(t0).unwrap();
        let text = sp3.to_string().unwrap();
        assert!(
            text.contains("\n*  2024 11 10  0  1  0.00000000\n"),
            "{text}"
        );
        assert!(
            text.starts_with("#dV2024 11 10  0  1  0.00000000"),
            "{text}"
        );
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_utc_leap_second_round_trip() {
        let options = SP3WriteOptions {
            time_system: SP3TimeSystem::UTC,
            ..SP3WriteOptions::default()
        };
        let mut sp3 = SP3::new(vec![sat("G01")], 60.0, options).unwrap();
        let t0 = Epoch::from_datetime(2016, 12, 31, 23, 59, 0.0, 0.0, TimeSystem::UTC);
        for k in 0..3 {
            sp3.push_epoch(t0 + 60.0 * k as f64).unwrap();
        }
        let text = sp3.to_string().unwrap();
        assert!(
            text.contains("\n*  2016 12 31 23 59 60.00000000\n"),
            "{text}"
        );
        assert!(
            text.contains("\n*  2017  1  1  0  0 59.00000000\n"),
            "{text}"
        );

        let parsed = SP3::from_str(&text).unwrap();
        assert_eq!(parsed.epochs().len(), 3);
        for (actual, expected) in parsed.epochs().iter().zip(sp3.epochs()) {
            assert_abs_diff_eq!(*actual - *expected, 0.0, epsilon = 1e-9);
        }
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_interval_at_written_precision() {
        let mut sp3 = SP3::new(vec![sat("G01")], 1.0e-9, SP3WriteOptions::default()).unwrap();
        sp3.push_epoch(gps_t0()).unwrap();
        let msg = write_error(&sp3);
        assert!(
            msg.contains("epoch interval 0.000000001 s rounds to zero at 8 decimal places"),
            "{msg}"
        );

        let mut sp3 = SP3::new(vec![sat("G01")], 1.0e6, SP3WriteOptions::default()).unwrap();
        sp3.push_epoch(gps_t0()).unwrap();
        let msg = write_error(&sp3);
        assert!(
            msg.contains("epoch interval 1000000 s does not fit the 14-column ## field"),
            "{msg}"
        );

        let mut sp3 = SP3::new(vec![sat("G01")], 99_999.0, SP3WriteOptions::default()).unwrap();
        sp3.push_epoch(gps_t0()).unwrap();
        assert!(sp3.to_string().unwrap().contains(" 99999.00000000 "));

        let mut sp3 = SP3::new(vec![sat("G01")], 6.0e-9, SP3WriteOptions::default()).unwrap();
        sp3.push_epoch(gps_t0() + 6.0e-9).unwrap();
        sp3.push_epoch(gps_t0() + 12.0e-9).unwrap();
        let msg = write_error(&sp3);
        assert!(msg.contains("is not after the last epoch"), "{msg}");
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_date_reference_uses_written_start() {
        let line = |start: Epoch| {
            let mut sp3 = SP3::new(vec![sat("G01")], 900.0, SP3WriteOptions::default()).unwrap();
            sp3.push_epoch(start).unwrap();
            sp3.to_string().unwrap().lines().nth(1).unwrap().to_string()
        };
        let rounded = line(gps_t0() - 4.0e-9);
        assert_eq!(rounded, line(gps_t0()));
        assert!(rounded.starts_with("## 2340      0.00000000"), "{rounded}");
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_saturates_large_standard_deviations() {
        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        sp3.header_mut().position_base = Some(1.25);
        sp3.header_mut().clock_base = Some(1.025);
        let mut record = SP3Record::new([1.0e7, 2.0e7, 3.0e7])
            .with_clock(1e-4)
            .with_velocity([1.0, 2.0, 3.0])
            .with_clock_rate(1e-12);
        // 1.25^98 mm is about 3.2e6 m and 1.025^998 ps about 5.1e-2 s; larger
        // values need the "too large" marker.
        record.position_std = [Some(1.0e7), Some(1.25f64.powi(98) * 1e-3), None];
        record.clock_std = Some(1.0);
        record.velocity_std = [Some(1.0e4), None, None];
        record.clock_rate_std = Some(1.0e-1);
        sp3.set_record(sat("G02"), 0, record.clone()).unwrap();
        let text = sp3.to_string().unwrap();
        assert!(
            text.lines().any(|l| l
                == "PG02  10000.000000  20000.000000  30000.000000    100.000000 99 98    999"),
            "{text}"
        );
        assert!(
            text.lines().any(|l| l
                == "VG02     10.000000     20.000000     30.000000      0.010000 99       999"),
            "{text}"
        );

        let back = SP3::from_str(&text).unwrap();
        let written = back.record(sat("G02"), 0).unwrap().unwrap();
        assert_eq!(written.position_std[0], Some(f64::INFINITY));
        assert_eq!(written.clock_std, Some(f64::INFINITY));
        assert_eq!(written.velocity_std[0], Some(f64::INFINITY));
        assert_eq!(written.clock_rate_std, Some(f64::INFINITY));
        assert_quantized(
            record.position_std[1],
            written.position_std[1],
            Some(1.25),
            "position std",
        );
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_invalid_bases() {
        let with_std = |position_base: Option<f64>, clock_base: Option<f64>| {
            let mut sp3 = two_satellite_file(SP3WriteOptions::default());
            sp3.header_mut().position_base = position_base;
            sp3.header_mut().clock_base = clock_base;
            let mut record = SP3Record::new([1.0e7, 2.0e7, 3.0e7]).with_clock(1e-4);
            record.position_std = [Some(0.010); 3];
            record.clock_std = Some(1e-10);
            sp3.set_record(sat("G02"), 0, record).unwrap();
            sp3
        };
        for base in [0.0, 1.0, 0.5, -1.25, f64::NAN, f64::INFINITY] {
            let msg = write_error(&with_std(Some(base), Some(1.025)));
            assert!(msg.contains("position/velocity base"), "{base}: {msg}");
            let msg = write_error(&with_std(Some(1.25), Some(base)));
            assert!(msg.contains("clock base"), "{base}: {msg}");
        }
        let msg = write_error(&with_std(Some(1.000_000_01), Some(1.025)));
        assert!(
            msg.contains("position/velocity base 1.00000001 is written as 1.0000000"),
            "{msg}"
        );
        let msg = write_error(&with_std(Some(1.25), Some(1.000_000_000_1)));
        assert!(msg.contains("is written as 1.000000000"), "{msg}");

        // An unused base is still validated, since the file must re-parse.
        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        sp3.header_mut().position_base = Some(1.0);
        assert!(write_error(&sp3).contains("position/velocity base"));

        // Exponents use the base as written, so they stay consistent on re-read.
        let sp3 = with_std(Some(1.250_000_04), Some(1.025));
        let back = SP3::from_str(&sp3.to_string().unwrap()).unwrap();
        assert_eq!(back.header().position_base, Some(1.25));
        let written = back.record(sat("G02"), 0).unwrap().unwrap();
        assert_quantized(Some(0.010), written.position_std[0], Some(1.25), "std");
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_clock_in_missing_value_range() {
        let clock_file = |clock: f64| {
            let mut sp3 = two_satellite_file(SP3WriteOptions::default());
            sp3.set_record(
                sat("G02"),
                0,
                SP3Record::new([1.0e7, 2.0e7, 3.0e7]).with_clock(clock),
            )
            .unwrap();
            sp3
        };
        let msg = write_error(&clock_file(1.0));
        assert!(msg.contains("G02 at epoch 0"), "{msg}");
        assert!(msg.contains("clock 1000000.000000"), "{msg}");
        assert!(msg.contains("missing-value marker"), "{msg}");
        // Rounds to 999999.000000 in the file.
        assert!(write_error(&clock_file(0.999_998_999_999_9)).contains("missing-value marker"));

        let text = clock_file(0.999_998_9).to_string().unwrap();
        let back = SP3::from_str(&text).unwrap();
        let clock = back.record(sat("G02"), 0).unwrap().unwrap().clock.unwrap();
        assert_abs_diff_eq!(clock, 0.999_998_9, epsilon = 1e-12);

        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        sp3.set_record(
            sat("G02"),
            1,
            SP3Record::new([1.0e7, 2.0e7, 3.0e7])
                .with_velocity([1.0, 2.0, 3.0])
                .with_clock_rate(1.0e-4),
        )
        .unwrap();
        let msg = write_error(&sp3);
        assert!(msg.contains("clock rate 1000000.000000"), "{msg}");
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_invalid_satellite_number() {
        let template = two_satellite_file(SP3WriteOptions::default());
        let bad = SP3SatelliteId {
            system: GNSSSystem::GPS,
            number: 100,
        };
        let sp3 = SP3::from_parts(
            template.header().clone(),
            template.epochs().to_vec(),
            vec![sat("G01"), bad],
            template.rows().to_vec(),
        );
        let msg = write_error(&sp3);
        assert!(
            msg.contains("satellite number 100 out of range 1..=99"),
            "{msg}"
        );
    }

    #[test]
    #[parallel]
    fn test_SP3_round_trip_blank_correlation_coefficients() {
        let content = std::fs::read_to_string("test_assets/sp3/example-d-2.sp3")
            .unwrap()
            .replacen(
                "EP    55   55   55     222  1234567 -1234567  5999999      -30       21 -1230000",
                "EP    10   10   10",
                1,
            );
        let read = SP3::from_str(&content).unwrap();
        let text = read.to_string().unwrap();
        assert!(text.lines().any(|l| l == "EP    10   10   10"), "{text}");

        let written = SP3::from_str(&text).unwrap();
        let blank = written
            .record(sat("G01"), 0)
            .unwrap()
            .unwrap()
            .position_correlation
            .clone()
            .unwrap();
        assert_eq!(
            [blank.xy, blank.xz, blank.xc, blank.yz, blank.yc, blank.zc],
            [None; 6]
        );
        let full = written
            .record(sat("G02"), 0)
            .unwrap()
            .unwrap()
            .position_correlation
            .clone()
            .unwrap();
        let coefficients = [full.xy, full.xz, full.xc, full.yz, full.yc, full.zc];
        let expected = [
            0.1234567, -0.1234567, 0.5999999, -0.0000030, 0.0000021, -0.1230000,
        ];
        for (c, e) in coefficients.iter().zip(expected) {
            assert_abs_diff_eq!(c.unwrap(), e, epsilon = 1e-15);
        }

        // Interior blanks keep the columns of the coefficients after them.
        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        let mut record = SP3Record::new([1.0e7, 2.0e7, 3.0e7]);
        record.position_correlation = Some(SP3Correlation {
            std: [None; 3],
            clock_std: None,
            xy: None,
            xz: Some(0.5),
            xc: None,
            yz: None,
            yc: None,
            zc: Some(-0.25),
        });
        sp3.set_record(sat("G02"), 0, record).unwrap();
        let text = sp3.to_string().unwrap();
        assert!(
            text.lines().any(|l| l
                == "EP                                   5000000                            -2500000"),
            "{text}"
        );
        let back = SP3::from_str(&text).unwrap();
        let c = back
            .record(sat("G02"), 0)
            .unwrap()
            .unwrap()
            .position_correlation
            .clone()
            .unwrap();
        assert_eq!(
            [c.xy, c.xz, c.xc, c.yz, c.yc, c.zc],
            [None, Some(0.5), None, None, None, Some(-0.25)]
        );
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_correlation_out_of_range() {
        let with_xy = |xy: f64| {
            let mut sp3 = two_satellite_file(SP3WriteOptions::default());
            let mut record = SP3Record::new([1.0e7, 2.0e7, 3.0e7]);
            record.position_correlation = Some(SP3Correlation {
                std: [None; 3],
                clock_std: None,
                xy: Some(xy),
                xz: None,
                xc: None,
                yz: None,
                yc: None,
                zc: None,
            });
            sp3.set_record(sat("G02"), 0, record).unwrap();
            sp3
        };
        for xy in [2.0, -2.0, 1.0, -1.0, 0.999_999_96, f64::INFINITY] {
            let msg = write_error(&with_xy(xy));
            assert!(
                msg.contains("is outside [-0.9999999, 0.9999999]"),
                "{xy}: {msg}"
            );
        }
        for xy in [0.9999999, -0.9999999, 0.999_999_94] {
            let text = with_xy(xy).to_string().unwrap();
            let back = SP3::from_str(&text).unwrap();
            let c = back
                .record(sat("G02"), 0)
                .unwrap()
                .unwrap()
                .position_correlation
                .clone()
                .unwrap();
            assert_abs_diff_eq!(c.xy.unwrap(), xy, epsilon = 1e-7);
        }
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_no_epochs() {
        let sp3 = SP3::new(vec![sat("G01")], 900.0, SP3WriteOptions::default()).unwrap();
        assert!(write_error(&sp3).contains("file has no epochs"));
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.sp3");
        assert!(sp3.to_file(&path).is_err());
        assert!(!path.exists());
    }

    #[test]
    #[parallel]
    fn test_SP3_to_string_header_label_widths() {
        type SetLabel = fn(&mut SP3);
        let cases: [(&str, SetLabel); 4] = [
            ("data used 'ORBITS' is longer than 5 characters", |s| {
                s.header_mut().data_used = "ORBITS".to_string()
            }),
            (
                "coordinate system 'IGS20X' is longer than 5 characters",
                |s| s.header_mut().coordinate_system = "IGS20X".to_string(),
            ),
            ("orbit type 'FITS' is longer than 3 characters", |s| {
                s.header_mut().orbit_type = SP3OrbitType::Other("FITS".to_string())
            }),
            ("agency 'BRAHE' is longer than 4 characters", |s| {
                s.header_mut().agency = "BRAHE".to_string()
            }),
        ];
        for (expected, set) in cases {
            let mut sp3 = two_satellite_file(SP3WriteOptions::default());
            set(&mut sp3);
            let msg = write_error(&sp3);
            assert!(msg.contains(expected), "{msg}");
            assert!(msg.contains("header_mut"), "{msg}");
        }

        let mut sp3 = two_satellite_file(SP3WriteOptions::default());
        sp3.header_mut().data_used = "ABCDE".to_string();
        sp3.header_mut().coordinate_system = "FGHIJ".to_string();
        sp3.header_mut().orbit_type = SP3OrbitType::Other("KLM".to_string());
        sp3.header_mut().agency = "NOPQ".to_string();
        let text = sp3.to_string().unwrap();
        assert!(
            text.starts_with("#dV2024 11 10  0  0  0.00000000       2 ABCDE FGHIJ KLM NOPQ\n"),
            "{text}"
        );
        let header = SP3::from_str(&text).unwrap().header().clone();
        assert_eq!(header.orbit_type, SP3OrbitType::Other("KLM".to_string()));
        assert_eq!(header.agency, "NOPQ");
    }
}
