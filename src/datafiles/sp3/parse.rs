//! SP3 text parser.
//!
//! Reads SP3 versions a, b, c, and d into an [`SP3`]. The version line, the
//! `##` line, and epoch lines are split on whitespace; satellite lists,
//! `%c`/`%f` lines, and the `P`, `EP`, `V`, and `EV` records are read from
//! the fixed columns of the format specification. File units (km, dm/s, µs,
//! 1e-4 µs/s, and the standard deviation units) are converted to SI here.

use std::collections::HashMap;
use std::path::Path;
use std::str::FromStr;

use super::types::{
    SP3, SP3Correlation, SP3FileType, SP3Header, SP3OrbitType, SP3Record, SP3SatelliteId,
    SP3TimeSystem, SP3Version, file_type_from_satellites,
};
use crate::time::conversions::days_in_month;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

/// Kilometers to meters.
const KM_TO_M: f64 = 1.0e3;
/// Decimeters per second to meters per second.
const DM_PER_S_TO_M_PER_S: f64 = 1.0e-1;
/// Microseconds to seconds.
const US_TO_S: f64 = 1.0e-6;
/// Clock rate unit of 1e-4 microseconds per second to seconds per second.
const CLOCK_RATE_TO_S_PER_S: f64 = 1.0e-10;
/// Millimeters to meters.
const MM_TO_M: f64 = 1.0e-3;
/// Velocity standard deviation unit of 1e-4 millimeters per second to meters per second.
const VELOCITY_STD_TO_M_PER_S: f64 = 1.0e-7;
/// Picoseconds to seconds.
const PS_TO_S: f64 = 1.0e-12;
/// Clock rate standard deviation unit of 1e-4 picoseconds per second to seconds per second.
const CLOCK_RATE_STD_TO_S_PER_S: f64 = 1.0e-16;
/// Scale of the integer correlation coefficients of `EP` and `EV` records.
const CORRELATION_SCALE: f64 = 1.0e-7;
/// Clock and clock-rate values at or above this are the bad-value marker `999999.999999`.
const BAD_CLOCK_VALUE: f64 = 999_999.0;
/// Base of the header accuracy exponents. Units: mm
const ACCURACY_BASE: f64 = 2.0;
/// Number of satellite fields on one `+` or `++` line.
const SATELLITES_PER_LINE: usize = 17;
/// Column (zero-based) of the first satellite field on `+` and `++` lines.
const FIRST_SATELLITE_COLUMN: usize = 9;

/// Create an SP3 error that is not tied to a line.
///
/// # Arguments
///
/// * `detail` - Description of the problem
///
/// # Returns
///
/// * `BraheError` - `ParseError("SP3: <detail>")`
pub(crate) fn sp3_error(detail: &str) -> BraheError {
    BraheError::ParseError(format!("SP3: {}", detail))
}

/// Create an SP3 parse error for a line of the file.
///
/// # Arguments
///
/// * `line` - One-based line number
/// * `detail` - Description of the problem
///
/// # Returns
///
/// * `BraheError` - `ParseError("SP3: line <line>: <detail>")`
pub(crate) fn sp3_parse_error(line: usize, detail: &str) -> BraheError {
    BraheError::ParseError(format!("SP3: line {}: {}", line, detail))
}

/// Attach a line number to an SP3 error raised by a field parser.
fn at_line(line: usize, err: BraheError) -> BraheError {
    match err {
        BraheError::ParseError(msg) => {
            sp3_parse_error(line, msg.strip_prefix("SP3: ").unwrap_or(&msg))
        }
        other => other,
    }
}

impl SP3 {
    /// Parse an SP3 file from text.
    ///
    /// Versions a, b, c, and d are accepted. All values are converted to SI
    /// units. A record whose position is the bad value `0.000000` is stored
    /// as absent, and a clock or clock rate of `999999.999999` is stored as
    /// `None`. Epochs are expressed in the Brahe time system matching the
    /// `%c` time system label; a label without a Brahe equivalent (`GLO`,
    /// `QZS`, `IRN`) is an error. Header fields are stored as read.
    ///
    /// # Arguments
    ///
    /// * `content` - SP3 file contents
    ///
    /// # Returns
    ///
    /// * `Result<SP3, BraheError>` - The parsed file, or a `ParseError` of the form
    ///   `"SP3: line N: <detail>"` naming the offending line
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{SP3, SP3SatelliteId, SP3Version};
    ///
    /// let content = "\
    /// #cP2001  8  8  0  0  0.00000000       1 ORBIT IGS97 HLM  IGS
    /// ### 1126 259200.00000000   900.00000000 52129 0.0000000000000
    /// +    1   G01
    /// ++         7
    /// %c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
    /// %f  1.2500000  1.025000000  0.00000000000  0.000000000000000
    /// %i    0    0    0    0      0      0      0      0         0
    /// /* EXAMPLE
    /// *  2001  8  8  0  0  0.00000000
    /// PG01 -11044.805800 -10475.672350  21929.418200    189.163300 18 18 18 219
    /// EOF
    /// ";
    /// let sp3 = SP3::from_str(content).unwrap();
    /// assert_eq!(sp3.header().version, SP3Version::C);
    /// let record = sp3.record(SP3SatelliteId::parse("G01").unwrap(), 0).unwrap().unwrap();
    /// assert!((record.position[0] + 11_044_805.8).abs() < 1e-6);
    /// assert!((record.clock.unwrap() - 189.1633e-6).abs() < 1e-15);
    /// ```
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(content: &str) -> Result<Self, BraheError> {
        let mut lines = content.lines().enumerate().map(|(i, l)| (i + 1, l));
        let mut parser = match lines.next() {
            Some((n, line)) if classify(line) == Some(LineKind::Version) => Parser::new(n, line)?,
            Some((n, _)) => {
                return Err(sp3_parse_error(
                    n,
                    "the first line is not an SP3 version line",
                ));
            }
            None => return Err(sp3_error("empty file")),
        };

        let mut previous = LineKind::Version;
        for (n, line) in lines {
            let kind = classify(line).ok_or_else(|| {
                let prefix: String = line.chars().take(2).collect();
                sp3_parse_error(n, &format!("unrecognized line identifier '{}'", prefix))
            })?;
            if !previous.allowed_next().contains(&kind) {
                return Err(sp3_parse_error(
                    n,
                    &format!(
                        "unexpected {} line after a {} line",
                        kind.label(),
                        previous.label()
                    ),
                ));
            }
            if kind == LineKind::EndOfFile {
                break;
            }
            parser.handle(kind, n, line)?;
            previous = kind;
        }

        parser.finish()
    }

    /// Parse an SP3 file from disk.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to a plain-text SP3 file
    ///
    /// # Returns
    ///
    /// * `Result<SP3, BraheError>` - The parsed file, an `IoError` if the file cannot be
    ///   read, or a `ParseError` as for [`SP3::from_str`]
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::SP3;
    ///
    /// let path = std::env::temp_dir().join("brahe_sp3_from_file_example.sp3");
    /// std::fs::write(&path, "\
    /// #dP2001  8  8  0  0  0.00000000       1 ORBIT IGS97 HLM MGEX
    /// ### 1126 259200.00000000   900.00000000 52129 0.0000000000000
    /// +    1   E11
    /// ++         0
    /// %c E  cc GAL ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
    /// %f  0.0000000  0.000000000  0.00000000000  0.000000000000000
    /// %i    0    0    0    0      0      0      0      0         0
    /// /* EXAMPLE
    /// *  2001  8  8  0  0  0.00000000
    /// PE11  15000.000000  20000.000000   5000.000000 999999.999999
    /// EOF
    /// ").unwrap();
    ///
    /// let sp3 = SP3::from_file(&path).unwrap();
    /// assert_eq!(sp3.num_satellites(), 1);
    /// assert_eq!(sp3.num_epochs(), 1);
    /// std::fs::remove_file(&path).unwrap();
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, BraheError> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| {
            BraheError::IoError(format!("SP3: failed to read {}: {}", path.display(), e))
        })?;
        Self::from_str(&content)
    }
}

/// Kind of an SP3 line, identified by its leading characters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LineKind {
    /// `#` version, start epoch, and file descriptors
    Version,
    /// `##` GPS week, interval, and MJD
    DateReference,
    /// `+` satellite identifiers
    SatelliteIds,
    /// `++` satellite accuracy exponents
    Accuracy,
    /// `%c` file type and time system
    TimeSystem,
    /// `%f` standard deviation bases
    Bases,
    /// `%i` custom integer parameters
    CustomParameters,
    /// `/*` comment
    Comment,
    /// `*` epoch header
    Epoch,
    /// `P` position and clock record
    Position,
    /// `EP` position and clock correlation record
    PositionCorrelation,
    /// `V` velocity and clock rate record
    Velocity,
    /// `EV` velocity and clock rate correlation record
    VelocityCorrelation,
    /// `EOF` end-of-file marker
    EndOfFile,
}

impl LineKind {
    /// Line kinds allowed to follow this one.
    fn allowed_next(self) -> &'static [LineKind] {
        use LineKind::*;
        match self {
            Version => &[DateReference],
            DateReference => &[SatelliteIds],
            SatelliteIds => &[SatelliteIds, Accuracy],
            Accuracy => &[Accuracy, TimeSystem],
            TimeSystem => &[TimeSystem, Bases],
            Bases => &[Bases, CustomParameters],
            CustomParameters => &[CustomParameters, Comment, Epoch],
            Comment => &[Comment, Epoch],
            Epoch => &[Epoch, Position, EndOfFile],
            Position => &[Epoch, Position, PositionCorrelation, Velocity, EndOfFile],
            PositionCorrelation => &[Epoch, Position, Velocity, EndOfFile],
            Velocity => &[Epoch, Position, VelocityCorrelation, EndOfFile],
            VelocityCorrelation => &[Epoch, Position, EndOfFile],
            EndOfFile => &[],
        }
    }

    /// Line identifier as written in the file.
    fn label(self) -> &'static str {
        match self {
            LineKind::Version => "#",
            LineKind::DateReference => "##",
            LineKind::SatelliteIds => "+",
            LineKind::Accuracy => "++",
            LineKind::TimeSystem => "%c",
            LineKind::Bases => "%f",
            LineKind::CustomParameters => "%i",
            LineKind::Comment => "/*",
            LineKind::Epoch => "*",
            LineKind::Position => "P",
            LineKind::PositionCorrelation => "EP",
            LineKind::Velocity => "V",
            LineKind::VelocityCorrelation => "EV",
            LineKind::EndOfFile => "EOF",
        }
    }

    /// Whether the line is read by fixed byte columns.
    fn has_fixed_columns(self) -> bool {
        matches!(
            self,
            LineKind::SatelliteIds
                | LineKind::Accuracy
                | LineKind::TimeSystem
                | LineKind::Bases
                | LineKind::Position
                | LineKind::PositionCorrelation
                | LineKind::Velocity
                | LineKind::VelocityCorrelation
        )
    }
}

/// Identify the kind of a line from its leading characters.
fn classify(line: &str) -> Option<LineKind> {
    let kind = if line.starts_with("##") {
        LineKind::DateReference
    } else if line.starts_with('#') {
        LineKind::Version
    } else if line.starts_with("++") {
        LineKind::Accuracy
    } else if line.starts_with('+') {
        LineKind::SatelliteIds
    } else if line.starts_with("%c") {
        LineKind::TimeSystem
    } else if line.starts_with("%f") {
        LineKind::Bases
    } else if line.starts_with("%i") {
        LineKind::CustomParameters
    } else if line.starts_with("/*") {
        LineKind::Comment
    } else if line.starts_with('*') {
        LineKind::Epoch
    } else if line.starts_with("EP") {
        LineKind::PositionCorrelation
    } else if line.starts_with("EV") {
        LineKind::VelocityCorrelation
    } else if line.trim_end() == "EOF" {
        LineKind::EndOfFile
    } else if line.starts_with('P') {
        LineKind::Position
    } else if line.starts_with('V') {
        LineKind::Velocity
    } else {
        return None;
    };
    Some(kind)
}

/// Fixed-column field `[start, end)` of an ASCII line.
///
/// A line ending at or before `start` yields an empty field. A line ending
/// strictly inside the field is truncated and is an error.
fn field(n: usize, line: &str, start: usize, end: usize) -> Result<&str, BraheError> {
    let len = line.len();
    if len <= start {
        Ok("")
    } else if len >= end {
        Ok(&line[start..end])
    } else {
        Err(sp3_parse_error(
            n,
            &format!("record truncated in columns {}-{}", start + 1, end),
        ))
    }
}

/// Parse a required number, ignoring surrounding blanks.
fn parse_number<T: FromStr>(n: usize, text: &str, what: &str) -> Result<T, BraheError> {
    let text = text.trim();
    text.parse::<T>()
        .map_err(|_| sp3_parse_error(n, &format!("invalid {} '{}'", what, text)))
}

/// Parse a required finite floating-point number, ignoring surrounding blanks.
fn parse_f64(n: usize, text: &str, what: &str) -> Result<f64, BraheError> {
    let value: f64 = parse_number(n, text, what)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(sp3_parse_error(
            n,
            &format!("invalid {} '{}'", what, text.trim()),
        ))
    }
}

/// Parse an optional number; a blank field is `None`.
fn parse_optional<T: FromStr>(n: usize, text: &str, what: &str) -> Result<Option<T>, BraheError> {
    if text.trim().is_empty() {
        Ok(None)
    } else {
        parse_number(n, text, what).map(Some)
    }
}

/// Parse an optional finite floating-point number; a blank field is `None`.
fn parse_optional_f64(n: usize, text: &str, what: &str) -> Result<Option<f64>, BraheError> {
    if text.trim().is_empty() {
        Ok(None)
    } else {
        parse_f64(n, text, what).map(Some)
    }
}

/// Calendar fields of an epoch as written in the file.
#[derive(Debug, Clone, Copy)]
struct DateTimeFields {
    year: u32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
}

impl DateTimeFields {
    /// Parse and validate year, month, day, hour, minute, and second fields.
    fn parse(n: usize, fields: [&str; 6]) -> Result<Self, BraheError> {
        let [year, month, day, hour, minute, second] = fields;
        let dt = DateTimeFields {
            year: parse_number(n, year, "year")?,
            month: parse_number(n, month, "month")?,
            day: parse_number(n, day, "day")?,
            hour: parse_number(n, hour, "hour")?,
            minute: parse_number(n, minute, "minute")?,
            second: parse_f64(n, second, "second")?,
        };
        dt.validate(n)?;
        Ok(dt)
    }

    /// Check that every field is within its calendar range.
    fn validate(&self, n: usize) -> Result<(), BraheError> {
        let max_day =
            days_in_month(self.year, self.month).map_err(|_| self.out_of_range(n, "month"))?;
        if !(1..=max_day).contains(&self.day) {
            return Err(self.out_of_range(n, "day"));
        }
        if self.hour > 23 {
            return Err(self.out_of_range(n, "hour"));
        }
        if self.minute > 59 {
            return Err(self.out_of_range(n, "minute"));
        }
        if !(0.0..60.0).contains(&self.second) {
            return Err(self.out_of_range(n, "second"));
        }
        Ok(())
    }

    fn out_of_range(&self, n: usize, what: &str) -> BraheError {
        sp3_parse_error(
            n,
            &format!(
                "{} out of range in date {:04}-{:02}-{:02} {:02}:{:02}:{:011.8}",
                what, self.year, self.month, self.day, self.hour, self.minute, self.second
            ),
        )
    }

    /// Epoch of these fields in a time system.
    fn to_epoch(self, time_system: TimeSystem) -> Epoch {
        Epoch::from_datetime(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            0.0,
            time_system,
        )
    }
}

/// Unit scales of the standard deviations of one record type.
#[derive(Debug, Clone, Copy)]
struct StdUnits {
    /// File unit of the x, y, z standard deviations to SI
    component: f64,
    /// File unit of the clock standard deviation to SI
    clock: f64,
}

/// Standard deviation units of `P` and `EP` records: mm and ps.
const POSITION_STD_UNITS: StdUnits = StdUnits {
    component: MM_TO_M,
    clock: PS_TO_S,
};

/// Standard deviation units of `V` and `EV` records: 1e-4 mm/s and 1e-4 ps/s.
const VELOCITY_STD_UNITS: StdUnits = StdUnits {
    component: VELOCITY_STD_TO_M_PER_S,
    clock: CLOCK_RATE_STD_TO_S_PER_S,
};

/// Values of a `P` or `V` record, converted to SI.
#[derive(Debug)]
struct StateFields {
    values: [f64; 3],
    clock: Option<f64>,
    std: [Option<f64>; 3],
    clock_std: Option<f64>,
}

/// Read the shared layout of `P` and `V` records.
///
/// Columns 5-46 hold x, y, z, columns 47-60 the clock term, columns 62-69
/// the two-digit component exponents, and columns 71-73 the three-digit clock
/// exponent. Standard deviations are `base^n` in the units of `units`; an
/// exponent of 99 (component) or 999 (clock) is stored as infinity.
fn parse_state_fields(
    n: usize,
    line: &str,
    value_scale: f64,
    clock_scale: f64,
    units: StdUnits,
    bases: (Option<f64>, Option<f64>),
) -> Result<StateFields, BraheError> {
    let axes = ["x", "y", "z"];
    let mut values = [0.0; 3];
    for (k, value) in values.iter_mut().enumerate() {
        let start = 4 + 14 * k;
        let text = field(n, line, start, start + 14)?;
        if text.trim().is_empty() {
            return Err(sp3_parse_error(
                n,
                &format!("missing {} component", axes[k]),
            ));
        }
        *value = parse_f64(n, text, axes[k])? * value_scale;
    }

    let clock = parse_optional_f64(n, field(n, line, 46, 60)?, "clock")?
        .filter(|c| *c < BAD_CLOCK_VALUE)
        .map(|c| c * clock_scale);

    let (component_base, clock_base) = bases;
    let mut std = [None; 3];
    for (k, s) in std.iter_mut().enumerate() {
        let start = 61 + 3 * k;
        *s = exponent_std(
            n,
            field(n, line, start, start + 2)?,
            component_base,
            "position/velocity",
            99,
            units.component,
        )?;
    }
    let clock_std = exponent_std(
        n,
        field(n, line, 70, 73)?,
        clock_base,
        "clock",
        999,
        units.clock,
    )?;

    Ok(StateFields {
        values,
        clock,
        std,
        clock_std,
    })
}

/// Standard deviation `base^n * scale` from an exponent field.
///
/// `base_name` names the `%f` base the exponent refers to; an exponent with
/// no such base is an error.
fn exponent_std(
    n: usize,
    text: &str,
    base: Option<f64>,
    base_name: &str,
    too_large: i32,
    scale: f64,
) -> Result<Option<f64>, BraheError> {
    let Some(exponent) = parse_optional::<i32>(n, text, "standard deviation exponent")? else {
        return Ok(None);
    };
    let base = base.ok_or_else(|| {
        sp3_parse_error(
            n,
            &format!(
                "{} standard deviation exponent but the %f line declares no {} base",
                base_name, base_name
            ),
        )
    })?;
    if exponent == too_large {
        Ok(Some(f64::INFINITY))
    } else {
        Ok(Some(base.powi(exponent) * scale))
    }
}

/// Read an `EP` or `EV` correlation record.
///
/// Columns 5-18 hold the component standard deviations (I4, 9999 is too
/// large), columns 20-26 the clock standard deviation (I7, 9999999 is too
/// large), and columns 28-80 six I8 correlation coefficients scaled by 1e7.
fn parse_correlation(n: usize, line: &str, units: StdUnits) -> Result<SP3Correlation, BraheError> {
    let std_value = |start: usize, end: usize, too_large: i64, scale: f64| {
        let value = parse_optional::<i64>(n, field(n, line, start, end)?, "standard deviation")?;
        Ok::<_, BraheError>(value.map(|v| {
            if v == too_large {
                f64::INFINITY
            } else {
                v as f64 * scale
            }
        }))
    };
    let correlation = |k: usize| {
        let start = 27 + 9 * k;
        let value = parse_optional::<i64>(n, field(n, line, start, start + 8)?, "correlation")?;
        Ok::<_, BraheError>(value.map_or(0.0, |v| v as f64 * CORRELATION_SCALE))
    };

    Ok(SP3Correlation {
        std: [
            std_value(4, 8, 9999, units.component)?,
            std_value(9, 13, 9999, units.component)?,
            std_value(14, 18, 9999, units.component)?,
        ],
        clock_std: std_value(19, 26, 9_999_999, units.clock)?,
        xy: correlation(0)?,
        xz: correlation(1)?,
        xc: correlation(2)?,
        yz: correlation(3)?,
        yc: correlation(4)?,
        zc: correlation(5)?,
    })
}

/// File type from a `%c` file type field, if it is a single valid code.
fn parse_file_type(code: &str) -> Option<SP3FileType> {
    let mut chars = code.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => SP3FileType::from_code(c).ok(),
        _ => None,
    }
}

/// Record of the most recent `P` line, completed by the `EP`, `V`, and `EV`
/// lines that follow it.
struct PendingRecord {
    id: SP3SatelliteId,
    /// Header index of the satellite; `None` for a bad-value record of a
    /// satellite missing from the header list, which is discarded
    satellite: Option<usize>,
    record: SP3Record,
}

/// Line-by-line SP3 parser state.
struct Parser {
    header: SP3Header,
    start: DateTimeFields,
    time_system: Option<TimeSystem>,
    satellite_count: Option<usize>,
    satellites: Vec<SP3SatelliteId>,
    index: HashMap<SP3SatelliteId, usize>,
    accuracy_started: bool,
    bases_read: bool,
    epochs: Vec<Epoch>,
    records: Vec<Vec<Option<SP3Record>>>,
    seen: Vec<bool>,
    pending: Option<PendingRecord>,
}

impl Parser {
    /// Start parsing from the version line.
    ///
    /// The time system, file type, and start epoch are completed when the
    /// first `%c` line is read.
    fn new(n: usize, line: &str) -> Result<Self, BraheError> {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 10 {
            return Err(sp3_parse_error(
                n,
                &format!(
                    "version line has {} fields, expected at least 10",
                    tokens.len()
                ),
            ));
        }

        let mut chars = tokens[0].chars();
        chars.next();
        let version = chars
            .next()
            .ok_or_else(|| sp3_parse_error(n, "missing version character"))
            .and_then(|c| SP3Version::from_char(c).map_err(|e| at_line(n, e)))?;
        let has_velocity = match chars.next() {
            Some('P') => false,
            Some('V') => true,
            other => {
                return Err(sp3_parse_error(
                    n,
                    &format!(
                        "invalid position/velocity flag '{}'",
                        other.map(String::from).unwrap_or_default()
                    ),
                ));
            }
        };
        let start = DateTimeFields::parse(
            n,
            [
                chars.as_str(),
                tokens[1],
                tokens[2],
                tokens[3],
                tokens[4],
                tokens[5],
            ],
        )?;

        let header = SP3Header {
            version,
            has_velocity,
            start_epoch: start.to_epoch(TimeSystem::GPS),
            number_of_epochs: parse_number(n, tokens[6], "number of epochs")?,
            data_used: tokens[7].to_string(),
            coordinate_system: tokens[8].to_string(),
            orbit_type: SP3OrbitType::parse(tokens[9]),
            agency: tokens.get(10).copied().unwrap_or("").to_string(),
            gps_week: 0,
            seconds_of_week: 0.0,
            epoch_interval: 0.0,
            mjd: 0,
            fractional_day: 0.0,
            file_type: SP3FileType::Mixed,
            time_system: SP3TimeSystem::GPS,
            position_base: None,
            clock_base: None,
            accuracies: Vec::new(),
            comments: Vec::new(),
        };

        Ok(Parser {
            header,
            start,
            time_system: None,
            satellite_count: None,
            satellites: Vec::new(),
            index: HashMap::new(),
            accuracy_started: false,
            bases_read: false,
            epochs: Vec::new(),
            records: Vec::new(),
            seen: Vec::new(),
            pending: None,
        })
    }

    /// Process one line after the version line.
    fn handle(&mut self, kind: LineKind, n: usize, line: &str) -> Result<(), BraheError> {
        if kind.has_fixed_columns() && !line.is_ascii() {
            return Err(sp3_parse_error(
                n,
                "non-ASCII characters in a fixed-column line",
            ));
        }
        match kind {
            LineKind::Version | LineKind::EndOfFile => Err(sp3_parse_error(
                n,
                &format!("unexpected {} line", kind.label()),
            )),
            LineKind::DateReference => self.date_reference(n, line),
            LineKind::SatelliteIds => self.satellite_ids(n, line),
            LineKind::Accuracy => self.accuracy(n, line),
            LineKind::TimeSystem => self.time_system_line(n, line),
            LineKind::Bases => self.bases(n, line),
            LineKind::CustomParameters => Ok(()),
            LineKind::Comment => {
                let body = &line[2..];
                let body = body.strip_prefix(' ').unwrap_or(body);
                self.header.comments.push(body.trim_end().to_string());
                Ok(())
            }
            LineKind::Epoch => self.epoch(n, line),
            LineKind::Position => self.position(n, line),
            LineKind::PositionCorrelation => self.position_correlation(n, line),
            LineKind::Velocity => self.velocity(n, line),
            LineKind::VelocityCorrelation => self.velocity_correlation(n, line),
        }
    }

    /// `##` line: GPS week, seconds of week, interval, MJD, fractional day.
    fn date_reference(&mut self, n: usize, line: &str) -> Result<(), BraheError> {
        let tokens: Vec<&str> = line[2..].split_whitespace().collect();
        if tokens.len() < 5 {
            return Err(sp3_parse_error(
                n,
                &format!("## line has {} fields, expected 5", tokens.len()),
            ));
        }
        self.header.gps_week = parse_number(n, tokens[0], "GPS week")?;
        self.header.seconds_of_week = parse_f64(n, tokens[1], "seconds of week")?;
        self.header.epoch_interval = parse_f64(n, tokens[2], "epoch interval")?;
        self.header.mjd = parse_number(n, tokens[3], "modified Julian day")?;
        self.header.fractional_day = parse_f64(n, tokens[4], "fractional day")?;
        Ok(())
    }

    /// `+` line: the satellite count (columns 4-6 of the first line) and
    /// identifiers in three-column fields starting at column 10. Blank and
    /// zero padding fields (`  0`, ` 00`) are skipped.
    fn satellite_ids(&mut self, n: usize, line: &str) -> Result<(), BraheError> {
        let count = match self.satellite_count {
            Some(count) => count,
            None => {
                let count = parse_number(n, field(n, line, 3, 6)?, "satellite count")?;
                self.satellite_count = Some(count);
                count
            }
        };
        for k in 0..SATELLITES_PER_LINE {
            let start = FIRST_SATELLITE_COLUMN + 3 * k;
            if self.satellites.len() >= count || start + 3 > line.len() {
                break;
            }
            let raw = &line[start..start + 3];
            if raw.trim().trim_start_matches('0').is_empty() {
                continue;
            }
            let id = SP3SatelliteId::parse(raw).map_err(|e| at_line(n, e))?;
            if self.index.insert(id, self.satellites.len()).is_some() {
                return Err(sp3_parse_error(
                    n,
                    &format!("satellite {} is listed twice", id),
                ));
            }
            self.satellites.push(id);
        }
        Ok(())
    }

    /// `++` line: accuracy exponents aligned with the satellite fields. An
    /// exponent `n` is an accuracy of `2^n` mm; `0` is unknown.
    fn accuracy(&mut self, n: usize, line: &str) -> Result<(), BraheError> {
        let count = self.satellite_count.unwrap_or(0);
        if !self.accuracy_started {
            self.accuracy_started = true;
            if self.satellites.len() != count {
                return Err(sp3_parse_error(
                    n,
                    &format!(
                        "header declares {} satellites but lists {}",
                        count,
                        self.satellites.len()
                    ),
                ));
            }
        }
        for k in 0..SATELLITES_PER_LINE {
            let start = FIRST_SATELLITE_COLUMN + 3 * k;
            if self.header.accuracies.len() >= count || start + 3 > line.len() {
                break;
            }
            let exponent = parse_optional::<i32>(n, &line[start..start + 3], "accuracy exponent")?;
            self.header.accuracies.push(match exponent {
                None | Some(0) => None,
                Some(e) => Some(ACCURACY_BASE.powi(e) * MM_TO_M),
            });
        }
        Ok(())
    }

    /// `%c` line: file type (columns 4-5) and time system (columns 10-12).
    /// Only the first `%c` line carries values.
    fn time_system_line(&mut self, n: usize, line: &str) -> Result<(), BraheError> {
        if self.time_system.is_some() {
            return Ok(());
        }

        let code = field(n, line, 3, 5)?.trim();
        self.header.file_type = match parse_file_type(code) {
            Some(file_type) => file_type,
            None if matches!(self.header.version, SP3Version::A | SP3Version::B) => {
                file_type_from_satellites(&self.satellites)
            }
            None => {
                return Err(sp3_parse_error(n, &format!("unknown file type '{}'", code)));
            }
        };

        let label = SP3TimeSystem::parse(field(n, line, 9, 12)?).map_err(|e| at_line(n, e))?;
        let time_system = label.to_time_system().ok_or_else(|| {
            sp3_parse_error(n, &format!("time system '{}' is not supported", label))
        })?;
        self.header.time_system = label;
        self.header.start_epoch = self.start.to_epoch(time_system);
        self.time_system = Some(time_system);
        Ok(())
    }

    /// `%f` line: bases of the position/velocity (columns 4-13) and
    /// clock/clock-rate (columns 15-26) standard deviation exponents. Only
    /// the first `%f` line carries values; a zero base means none.
    fn bases(&mut self, n: usize, line: &str) -> Result<(), BraheError> {
        if self.bases_read {
            return Ok(());
        }
        self.bases_read = true;
        let nonzero = |b: Option<f64>| b.filter(|b| *b != 0.0);
        self.header.position_base = nonzero(parse_optional_f64(
            n,
            field(n, line, 3, 13)?,
            "position base",
        )?);
        self.header.clock_base = nonzero(parse_optional_f64(
            n,
            field(n, line, 14, 26)?,
            "clock base",
        )?);
        Ok(())
    }

    /// `*` line: a new epoch, split on whitespace.
    fn epoch(&mut self, n: usize, line: &str) -> Result<(), BraheError> {
        self.flush();
        let time_system = self
            .time_system
            .ok_or_else(|| sp3_parse_error(n, "epoch line before the %c time system line"))?;
        let tokens: Vec<&str> = line[1..].split_whitespace().collect();
        if tokens.len() < 6 {
            return Err(sp3_parse_error(
                n,
                &format!("epoch line has {} fields, expected 6", tokens.len()),
            ));
        }
        let fields = DateTimeFields::parse(
            n,
            [
                tokens[0], tokens[1], tokens[2], tokens[3], tokens[4], tokens[5],
            ],
        )?;
        self.epochs.push(fields.to_epoch(time_system));
        self.records.push(vec![None; self.satellites.len()]);
        self.seen.clear();
        self.seen.resize(self.satellites.len(), false);
        Ok(())
    }

    /// `P` line: position, clock, their standard deviations, and the flags
    /// in columns 75, 76, 79, and 80.
    ///
    /// A satellite missing from the header list is accepted only with the
    /// bad-value position `0.000000`, and its record is discarded.
    fn position(&mut self, n: usize, line: &str) -> Result<(), BraheError> {
        self.flush();
        let id = SP3SatelliteId::parse(field(n, line, 1, 4)?).map_err(|e| at_line(n, e))?;
        let state = parse_state_fields(
            n,
            line,
            KM_TO_M,
            US_TO_S,
            POSITION_STD_UNITS,
            (self.header.position_base, self.header.clock_base),
        )?;
        let satellite = self.index.get(&id).copied();
        match satellite {
            Some(k) if std::mem::replace(&mut self.seen[k], true) => {
                return Err(sp3_parse_error(
                    n,
                    &format!("second P record for {} at this epoch", id),
                ));
            }
            None if state.values != [0.0; 3] => {
                return Err(sp3_parse_error(
                    n,
                    &format!("satellite {} is not in the header satellite list", id),
                ));
            }
            _ => {}
        }
        let flag = |column: usize, c: u8| line.as_bytes().get(column) == Some(&c);
        self.pending = Some(PendingRecord {
            id,
            satellite,
            record: SP3Record {
                position: state.values,
                clock: state.clock,
                position_std: state.std,
                clock_std: state.clock_std,
                clock_event: flag(74, b'E'),
                clock_predicted: flag(75, b'P'),
                maneuver: flag(78, b'M'),
                orbit_predicted: flag(79, b'P'),
                ..SP3Record::default()
            },
        });
        Ok(())
    }

    /// The record started by the latest `P` line.
    fn pending_record(&mut self, n: usize, kind: LineKind) -> Result<&mut SP3Record, BraheError> {
        self.pending.as_mut().map(|p| &mut p.record).ok_or_else(|| {
            sp3_parse_error(
                n,
                &format!("{} line without a preceding P line", kind.label()),
            )
        })
    }

    /// `EP` line: position and clock correlation of the latest `P` record.
    fn position_correlation(&mut self, n: usize, line: &str) -> Result<(), BraheError> {
        let correlation = parse_correlation(n, line, POSITION_STD_UNITS)?;
        self.pending_record(n, LineKind::PositionCorrelation)?
            .position_correlation = Some(correlation);
        Ok(())
    }

    /// `V` line: velocity, clock rate, and their standard deviations for the
    /// satellite of the latest `P` record. A zero velocity is the bad value
    /// and is stored as `None`.
    fn velocity(&mut self, n: usize, line: &str) -> Result<(), BraheError> {
        let id = SP3SatelliteId::parse(field(n, line, 1, 4)?).map_err(|e| at_line(n, e))?;
        if self.pending.as_ref().map(|p| p.id) != Some(id) {
            return Err(sp3_parse_error(
                n,
                &format!("V record for {} does not follow its P record", id),
            ));
        }
        let state = parse_state_fields(
            n,
            line,
            DM_PER_S_TO_M_PER_S,
            CLOCK_RATE_TO_S_PER_S,
            VELOCITY_STD_UNITS,
            (self.header.position_base, self.header.clock_base),
        )?;
        let record = self.pending_record(n, LineKind::Velocity)?;
        record.velocity = (state.values != [0.0; 3]).then_some(state.values);
        record.clock_rate = state.clock;
        record.velocity_std = state.std;
        record.clock_rate_std = state.clock_std;
        Ok(())
    }

    /// `EV` line: velocity and clock rate correlation of the latest record.
    fn velocity_correlation(&mut self, n: usize, line: &str) -> Result<(), BraheError> {
        let correlation = parse_correlation(n, line, VELOCITY_STD_UNITS)?;
        self.pending_record(n, LineKind::VelocityCorrelation)?
            .velocity_correlation = Some(correlation);
        Ok(())
    }

    /// Store the pending record in the current epoch row. A record whose
    /// position is exactly zero is the bad value and leaves the satellite
    /// absent.
    fn flush(&mut self) {
        if let Some(pending) = self.pending.take()
            && let Some(satellite) = pending.satellite
            && pending.record.position != [0.0; 3]
            && let Some(row) = self.records.last_mut()
        {
            row[satellite] = Some(pending.record);
        }
    }

    /// Assemble the parsed file.
    fn finish(mut self) -> Result<SP3, BraheError> {
        self.flush();
        if self.time_system.is_none() {
            return Err(sp3_error("file ends before the %c time system line"));
        }
        self.header.accuracies.resize(self.satellites.len(), None);
        Ok(SP3::from_parts(
            self.header,
            self.epochs,
            self.satellites,
            self.records,
        ))
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::datafiles::sp3::{
        GNSSSystem, SP3, SP3FileType, SP3OrbitType, SP3SatelliteId, SP3TimeSystem, SP3Version,
    };
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::BraheError;
    use approx::assert_abs_diff_eq;
    use rstest::rstest;
    use serial_test::parallel;

    fn fixture(name: &str) -> String {
        format!("test_assets/sp3/{name}")
    }

    fn sat(code: &str) -> SP3SatelliteId {
        SP3SatelliteId::parse(code).unwrap()
    }

    fn assert_rel(actual: f64, expected: f64, tol: f64) {
        assert!(
            ((actual - expected) / expected).abs() <= tol,
            "{actual} differs from {expected} by more than {tol} relative"
        );
    }

    fn read_fixture(name: &str) -> String {
        std::fs::read_to_string(fixture(name)).unwrap()
    }

    fn parse_error(content: &str) -> String {
        match SP3::from_str(content) {
            Err(BraheError::ParseError(msg)) => msg,
            other => panic!("expected a parse error, got {other:?}"),
        }
    }

    fn assert_epoch(actual: Epoch, expected: Epoch) {
        assert_eq!(actual.time_system, expected.time_system);
        assert_abs_diff_eq!(actual - expected, 0.0, epsilon = 1e-9);
    }

    #[rstest]
    #[case("example-a-1.sp3", SP3Version::A, 25, 3)]
    #[case("example-a-2.sp3", SP3Version::A, 25, 3)]
    #[case("example-c-1.sp3", SP3Version::C, 26, 2)]
    #[case("example-c-2.sp3", SP3Version::C, 26, 2)]
    #[case("example-d-1.sp3", SP3Version::D, 140, 3)]
    #[case("example-d-2.sp3", SP3Version::D, 26, 2)]
    #[case("example-d-3.sp3", SP3Version::D, 26, 1)]
    #[case("example-d-4.sp3", SP3Version::D, 26, 1)]
    #[case("example-d-5.sp3", SP3Version::D, 26, 1)]
    #[parallel]
    fn test_SP3_from_file_spec_examples(
        #[case] f: &str,
        #[case] v: SP3Version,
        #[case] nsat: usize,
        #[case] nep: usize,
    ) {
        let sp3 = SP3::from_file(format!("test_assets/sp3/{f}")).unwrap();
        assert_eq!(
            (sp3.header().version, sp3.num_satellites(), sp3.num_epochs()),
            (v, nsat, nep)
        );
    }

    #[test]
    #[parallel]
    fn test_SP3_from_file_example_a_1() {
        let sp3 = SP3::from_file(fixture("example-a-1.sp3")).unwrap();
        let h = sp3.header();
        assert_eq!(h.version, SP3Version::A);
        assert!(!h.has_velocity);
        assert_eq!(h.orbit_type, SP3OrbitType::FIT);
        assert_eq!(h.time_system, SP3TimeSystem::GPS);
        assert_eq!(h.file_type, SP3FileType::GPS);
        assert_eq!(h.agency, "NGS");
        assert_eq!(h.coordinate_system, "ITR92");
        assert_eq!(h.data_used, "d");
        assert_eq!(h.number_of_epochs, 3);
        assert_eq!(h.gps_week, 779);
        assert_abs_diff_eq!(h.seconds_of_week, 518400.0, epsilon = 1e-10);
        assert_abs_diff_eq!(h.epoch_interval, 900.0, epsilon = 1e-15);
        assert_eq!(h.mjd, 49703);
        assert_abs_diff_eq!(h.fractional_day, 0.0, epsilon = 1e-15);
        assert_eq!(h.position_base, None);
        assert_eq!(h.clock_base, None);
        assert_eq!(h.comments.len(), 4);
        assert_eq!(h.accuracies.len(), 25);
        assert_abs_diff_eq!(h.accuracies[0].unwrap(), 0.128, epsilon = 1e-15);
        assert_abs_diff_eq!(h.accuracies[2].unwrap(), 0.032, epsilon = 1e-15);

        let t0 = Epoch::from_datetime(1994, 12, 17, 0, 0, 0.0, 0.0, TimeSystem::GPS);
        assert_epoch(h.start_epoch, t0);
        assert_epoch(sp3.start_epoch(), t0);
        assert_epoch(sp3.epochs()[1], t0 + 900.0);

        let g01 = sat("G01");
        assert_eq!(sp3.satellite_ids()[0], g01);
        assert_eq!(sp3.satellite_ids()[24], sat("G31"));
        assert!(!sp3.contains(sat("G03")));
        assert!(sp3.record(sat("G04"), 2).unwrap().is_some());
        let r = sp3.record(g01, 0).unwrap().unwrap();
        assert_abs_diff_eq!(r.position[0], 16258524.75, epsilon = 1e-6);
        assert_abs_diff_eq!(r.position[1], -3529015.75, epsilon = 1e-6);
        assert_abs_diff_eq!(r.position[2], -20611427.05, epsilon = 1e-6);
        assert_abs_diff_eq!(r.clock.unwrap(), -62.5406e-6, epsilon = 1e-15);
        assert_eq!(r.velocity, None);
        assert_eq!(r.clock_rate, None);
        assert_eq!(r.position_std, [None; 3]);
        assert_eq!(r.clock_std, None);
        assert!(!r.clock_event && !r.clock_predicted && !r.maneuver && !r.orbit_predicted);
    }

    #[test]
    #[parallel]
    fn test_SP3_from_file_example_a_2_velocity() {
        let sp3 = SP3::from_file(fixture("example-a-2.sp3")).unwrap();
        assert!(sp3.header().has_velocity);
        let r = sp3.record(sat("G01"), 0).unwrap().unwrap();
        let v = r.velocity.unwrap();
        assert_abs_diff_eq!(v[0], -656.0373522, epsilon = 1e-9);
        assert_abs_diff_eq!(v[1], 2560.5954994, epsilon = 1e-9);
        assert_abs_diff_eq!(v[2], -946.0427179, epsilon = 1e-9);
        assert_abs_diff_eq!(r.clock_rate.unwrap(), -2.4236e-12, epsilon = 1e-22);
    }

    #[test]
    #[parallel]
    fn test_SP3_from_file_example_c_1_flags() {
        let sp3 = SP3::from_file(fixture("example-c-1.sp3")).unwrap();
        let h = sp3.header();
        assert_eq!(h.orbit_type, SP3OrbitType::HLM);
        assert_eq!(h.agency, "IGS");
        assert_eq!(h.position_base, Some(1.25));
        assert_eq!(h.clock_base, Some(1.025));

        let r = sp3.record(sat("G01"), 0).unwrap().unwrap();
        assert_abs_diff_eq!(r.position[0], -11044805.8, epsilon = 1e-6);
        assert_abs_diff_eq!(r.position[1], -10475672.35, epsilon = 1e-6);
        assert_abs_diff_eq!(r.position[2], 21929418.2, epsilon = 1e-6);
        assert_abs_diff_eq!(r.clock.unwrap(), 189.1633e-6, epsilon = 1e-15);

        let g02 = sp3.record(sat("G02"), 0).unwrap().unwrap();
        assert!(g02.maneuver);
        assert!(!g02.clock_event && !g02.clock_predicted && !g02.orbit_predicted);

        let g04 = sp3.record(sat("G04"), 1).unwrap().unwrap();
        assert!(g04.clock_event);
        assert!(g04.clock_predicted);
        assert!(!g04.maneuver);
        assert!(g04.orbit_predicted);

        let g01 = sp3.record(sat("G01"), 1).unwrap().unwrap();
        assert!(!g01.clock_event);
        assert!(g01.clock_predicted && g01.orbit_predicted);

        assert!(sp3.record(sat("G07"), 0).unwrap().is_none());
        assert!(sp3.record(sat("G05"), 1).unwrap().is_none());
    }

    #[test]
    #[parallel]
    fn test_SP3_from_file_example_c_2_correlations() {
        let sp3 = SP3::from_file(fixture("example-c-2.sp3")).unwrap();
        let r = sp3.record(sat("G01"), 0).unwrap().unwrap();
        let v = r.velocity.unwrap();
        assert_abs_diff_eq!(v[0], 2029.8880364, epsilon = 1e-9);
        assert_abs_diff_eq!(v[1], -1846.2044804, epsilon = 1e-9);
        assert_abs_diff_eq!(v[2], 138.1387685, epsilon = 1e-9);
        assert_abs_diff_eq!(r.clock_rate.unwrap(), -4.534317e-10, epsilon = 1e-22);

        let ep = r.position_correlation.as_ref().unwrap();
        for s in ep.std {
            assert_abs_diff_eq!(s.unwrap(), 0.055, epsilon = 1e-15);
        }
        assert_abs_diff_eq!(ep.clock_std.unwrap(), 222e-12, epsilon = 1e-24);
        assert_abs_diff_eq!(ep.xy, 0.1234567, epsilon = 1e-15);
        assert_abs_diff_eq!(ep.xz, -0.1234567, epsilon = 1e-15);
        assert_abs_diff_eq!(ep.xc, 0.5999999, epsilon = 1e-15);
        assert_abs_diff_eq!(ep.yz, -0.0000030, epsilon = 1e-15);
        assert_abs_diff_eq!(ep.yc, 0.0000021, epsilon = 1e-15);
        assert_abs_diff_eq!(ep.zc, -0.1230000, epsilon = 1e-15);

        let ev = r.velocity_correlation.as_ref().unwrap();
        for s in ev.std {
            assert_abs_diff_eq!(s.unwrap(), 22e-7, epsilon = 1e-19);
        }
        assert_abs_diff_eq!(ev.clock_std.unwrap(), 111e-16, epsilon = 1e-28);
        for c in [ev.xy, ev.xz, ev.xc, ev.yz, ev.yc, ev.zc] {
            assert_abs_diff_eq!(c, 0.1234567, epsilon = 1e-15);
        }

        let g02 = sp3.record(sat("G02"), 0).unwrap().unwrap();
        assert!(g02.maneuver);
        assert!(g02.position_correlation.is_some());
    }

    #[test]
    #[parallel]
    fn test_SP3_from_file_example_d_1() {
        let sp3 = SP3::from_file(fixture("example-d-1.sp3")).unwrap();
        let h = sp3.header();
        assert_eq!(h.orbit_type, SP3OrbitType::BCT);
        assert_eq!(h.time_system, SP3TimeSystem::GPS);
        assert_eq!(h.file_type, SP3FileType::Mixed);
        assert_eq!(h.coordinate_system, "WGS84");
        assert_eq!(h.agency, "MGEX");
        assert_eq!(h.accuracies, vec![None; 140]);
        assert_eq!(
            h.comments,
            vec![
                "Note: This is a simulated file, meant to illustrate what an SP3-d header",
                "might look like with more than 85 satellites. Source for GPS and SBAS satel-",
                "lite positions: BRDM0930.13N. G=GPS,R=GLONASS,E=Galileo,C=BeiDou,J=QZSS,",
                "I=IRNSS,S=SBAS. For definitions of SBAS satellites, refer to the website:",
                "http://igs.org/mgex/status-SBAS",
            ]
        );

        let ids = sp3.satellite_ids();
        assert_eq!(ids[0], sat("G01"));
        assert_eq!(ids[139], sat("S38"));
        assert_eq!(
            ids.iter()
                .filter(|s| s.system == GNSSSystem::BeiDou)
                .count(),
            35
        );

        let s37 = sat("S37");
        let r = sp3.record(s37, 0).unwrap().unwrap();
        assert_abs_diff_eq!(r.position[0], -34534904.566, epsilon = 1e-6);
        assert_abs_diff_eq!(r.position[1], 24164610.955, epsilon = 1e-6);
        assert_abs_diff_eq!(r.position[2], 29812.840, epsilon = 1e-6);
        assert_abs_diff_eq!(r.clock.unwrap(), 0.29942e-6, epsilon = 1e-15);
        assert!(sp3.record(s37, 1).unwrap().is_none());
        assert!(sp3.record(s37, 2).unwrap().is_some());
        assert_epoch(
            sp3.end_epoch(),
            Epoch::from_datetime(2013, 4, 3, 23, 45, 0.0, 0.0, TimeSystem::GPS),
        );
    }

    #[test]
    #[parallel]
    fn test_SP3_from_file_example_d_2() {
        let sp3 = SP3::from_file(fixture("example-d-2.sp3")).unwrap();
        let h = sp3.header();
        assert_eq!(h.time_system, SP3TimeSystem::GPS);
        assert_eq!(h.orbit_type, SP3OrbitType::HLM);
        assert_eq!(h.file_type, SP3FileType::GPS);
        assert!(h.has_velocity);

        let g01 = sat("G01");
        let r = sp3.record(g01, 0).unwrap().unwrap();
        assert_abs_diff_eq!(r.position[0], -11044.805800e3, epsilon = 1e-6);
        assert_abs_diff_eq!(r.position[1], -10475.672350e3, epsilon = 1e-6);
        assert_abs_diff_eq!(r.position[2], 21929.418200e3, epsilon = 1e-6);
        assert_abs_diff_eq!(r.clock.unwrap(), 189.163300e-6, epsilon = 1e-15);
        for s in r.position_std {
            assert_rel(s.unwrap(), 1.25f64.powi(18) * 1e-3, 1e-15);
        }
        assert_rel(r.clock_std.unwrap(), 1.025f64.powi(219) * 1e-12, 1e-15);
        for s in r.velocity_std {
            assert_rel(s.unwrap(), 1.25f64.powi(14) * 1e-7, 1e-15);
        }
        assert_rel(r.clock_rate_std.unwrap(), 1.025f64.powi(191) * 1e-16, 1e-15);
        assert!(!r.clock_event && !r.clock_predicted && !r.maneuver && !r.orbit_predicted);

        let later = sp3.record(g01, 1).unwrap().unwrap();
        assert!(!later.clock_event);
        assert!(later.clock_predicted);
        assert!(!later.maneuver);
        assert!(later.orbit_predicted);
        assert_epoch(
            sp3.epochs()[1],
            Epoch::from_datetime(2001, 8, 9, 23, 45, 0.0, 0.0, TimeSystem::GPS),
        );

        assert!(sp3.record(sat("G06"), 0).unwrap().is_none());
        assert!(sp3.record(sat("G03"), 1).unwrap().is_none());
    }

    #[test]
    #[parallel]
    fn test_SP3_from_str_bad_clock() {
        let content = std::fs::read_to_string(fixture("example-d-2.sp3"))
            .unwrap()
            .replace(
                "PG01 -11044.805800 -10475.672350  21929.418200    189.163300 18 18 18 219\n",
                "PG01 -11044.805800 -10475.672350  21929.418200 999999.999999 18 18 18 219\n",
            )
            .replace(
                "VG01  20298.880364 -18462.044804   1381.387685     -4.534317 14 14 14 191\n",
                "VG01  20298.880364 -18462.044804   1381.387685 999999.999999 14 14 14 191\n",
            );
        let sp3 = SP3::from_str(&content).unwrap();
        let r = sp3.record(sat("G01"), 0).unwrap().unwrap();
        assert_eq!(r.clock, None);
        assert_eq!(r.clock_rate, None);
        assert_abs_diff_eq!(r.position[0], -11044805.8, epsilon = 1e-6);
        assert!(r.velocity.is_some());
        let g02 = sp3.record(sat("G02"), 0).unwrap().unwrap();
        assert!(g02.clock.is_some());
    }

    #[rstest]
    #[case(
        "example-d-3.sp3",
        SP3TimeSystem::UTC,
        TimeSystem::UTC,
        SP3FileType::IRNSS
    )]
    #[case(
        "example-d-4.sp3",
        SP3TimeSystem::UTC,
        TimeSystem::UTC,
        SP3FileType::SBAS
    )]
    #[case(
        "example-d-5.sp3",
        SP3TimeSystem::GAL,
        TimeSystem::GST,
        SP3FileType::SBAS
    )]
    #[parallel]
    fn test_SP3_from_file_time_systems(
        #[case] f: &str,
        #[case] sp3_ts: SP3TimeSystem,
        #[case] ts: TimeSystem,
        #[case] file_type: SP3FileType,
    ) {
        let sp3 = SP3::from_file(fixture(f)).unwrap();
        assert_eq!(sp3.header().time_system, sp3_ts);
        assert_eq!(sp3.header().file_type, file_type);
        let t0 = Epoch::from_datetime(2001, 8, 8, 0, 0, 0.0, 0.0, ts);
        assert_epoch(sp3.header().start_epoch, t0);
        assert_epoch(sp3.start_epoch(), t0);
    }

    #[test]
    #[parallel]
    fn test_SP3_from_file_unsupported_time_system() {
        match SP3::from_file(fixture("example-d-2-glo.sp3")) {
            Err(BraheError::ParseError(msg)) => {
                assert_eq!(msg, "SP3: line 13: time system 'GLO' is not supported");
            }
            other => panic!("expected a parse error, got {other:?}"),
        }
    }

    #[test]
    #[parallel]
    fn test_SP3_from_file_missing_file() {
        assert!(matches!(
            SP3::from_file(fixture("does-not-exist.sp3")),
            Err(BraheError::IoError(_))
        ));
    }

    #[test]
    #[parallel]
    fn test_SP3_from_str_unknown_line() {
        let content = std::fs::read_to_string(fixture("example-c-1.sp3"))
            .unwrap()
            .replacen("PG03", "XG03", 1);
        match SP3::from_str(&content) {
            Err(BraheError::ParseError(msg)) => {
                assert!(msg.starts_with("SP3: line 26: "), "{msg}");
            }
            other => panic!("expected a parse error, got {other:?}"),
        }
    }

    #[test]
    #[parallel]
    fn test_SP3_from_str_undeclared_satellite() {
        let content = std::fs::read_to_string(fixture("example-c-1.sp3"))
            .unwrap()
            .replacen("PG03", "PG32", 1);
        match SP3::from_str(&content) {
            Err(BraheError::ParseError(msg)) => {
                assert!(msg.starts_with("SP3: line 26: "), "{msg}");
                assert!(msg.contains("G32"), "{msg}");
            }
            other => panic!("expected a parse error, got {other:?}"),
        }
    }

    #[test]
    #[parallel]
    fn test_sp3_parse_error() {
        match sp3_parse_error(7, "bad field") {
            BraheError::ParseError(msg) => assert_eq!(msg, "SP3: line 7: bad field"),
            other => panic!("unexpected error {other:?}"),
        }
        match sp3_error("bad file") {
            BraheError::ParseError(msg) => assert_eq!(msg, "SP3: bad file"),
            other => panic!("unexpected error {other:?}"),
        }
    }

    #[test]
    #[parallel]
    fn test_SP3_from_str_too_large_standard_deviations() {
        let content = read_fixture("example-d-2.sp3")
            .replacen(
                "    189.163300 18 18 18 219\n",
                "    189.163300 99 18    999\n",
                1,
            )
            .replacen(
                "EP    55   55   55     222",
                "EP  9999        55 9999999",
                1,
            );
        let sp3 = SP3::from_str(&content).unwrap();
        let r = sp3.record(sat("G01"), 0).unwrap().unwrap();
        assert_eq!(r.position_std[0], Some(f64::INFINITY));
        assert_rel(r.position_std[1].unwrap(), 1.25f64.powi(18) * 1e-3, 1e-15);
        assert_eq!(r.position_std[2], None);
        assert_eq!(r.clock_std, Some(f64::INFINITY));
        let ep = r.position_correlation.as_ref().unwrap();
        assert_eq!(ep.std[0], Some(f64::INFINITY));
        assert_eq!(ep.std[1], None);
        assert_abs_diff_eq!(ep.std[2].unwrap(), 0.055, epsilon = 1e-15);
        assert_eq!(ep.clock_std, Some(f64::INFINITY));
    }

    #[test]
    #[parallel]
    fn test_SP3_from_str_exponent_without_base() {
        let content = read_fixture("example-d-2.sp3").replacen(
            "%f  1.2500000  1.025000000",
            "%f  1.2500000  0.000000000",
            1,
        );
        let msg = parse_error(&content);
        assert!(msg.starts_with("SP3: line 24: "), "{msg}");
        assert!(msg.contains("no clock base"), "{msg}");
    }

    #[test]
    #[parallel]
    fn test_SP3_from_str_zero_velocity() {
        let content = read_fixture("example-d-2.sp3").replacen(
            "VG01  20298.880364 -18462.044804   1381.387685",
            "VG01      0.000000      0.000000      0.000000",
            1,
        );
        let sp3 = SP3::from_str(&content).unwrap();
        let r = sp3.record(sat("G01"), 0).unwrap().unwrap();
        assert_eq!(r.velocity, None);
        assert_abs_diff_eq!(r.clock_rate.unwrap(), -4.534317e-10, epsilon = 1e-22);
    }

    #[test]
    #[parallel]
    fn test_SP3_from_str_record_errors() {
        let c1 = read_fixture("example-c-1.sp3");
        let c2 = read_fixture("example-c-2.sp3");

        let msg = parse_error(&c1.replacen("PG02", "PG01", 1));
        assert!(msg.starts_with("SP3: line 25: "), "{msg}");
        assert!(msg.contains("second P record for G01"), "{msg}");

        let msg = parse_error(&c2.replacen("VG01", "VG02", 1));
        assert!(msg.starts_with("SP3: line 26: "), "{msg}");
        assert!(msg.contains("V record for G02"), "{msg}");

        let msg = parse_error(&c1.replacen("    189.163300 18 18 18 219", "    189.16", 1));
        assert_eq!(msg, "SP3: line 24: record truncated in columns 47-60");

        let msg = parse_error(&c1.replacen("PG01 -11044.805800", "PG01             ", 1));
        assert_eq!(msg, "SP3: line 24: missing x component");

        let msg = parse_error(&c1.replacen("PG01 -11044.805800", "PG01 -11044.8058\u{b5}", 1));
        assert!(msg.starts_with("SP3: line 24: non-ASCII"), "{msg}");

        let msg = parse_error(&c1.replacen("*  2001  8  8", "*  2001  2 30", 1));
        assert!(msg.starts_with("SP3: line 23: day out of range"), "{msg}");

        let msg = parse_error(&c1.replacen("*  2001", "EP", 1));
        assert!(
            msg.starts_with("SP3: line 23: unexpected EP line after a /* line"),
            "{msg}"
        );
    }

    #[test]
    #[parallel]
    fn test_SP3_from_str_header_errors() {
        let c1 = read_fixture("example-c-1.sp3");

        assert_eq!(parse_error(""), "SP3: empty file");
        assert_eq!(
            parse_error("## 1126"),
            "SP3: line 1: the first line is not an SP3 version line"
        );
        let header_only: String = c1.lines().take(14).map(|l| format!("{l}\n")).collect();
        assert_eq!(
            parse_error(&header_only[..header_only.find("%c").unwrap()]),
            "SP3: file ends before the %c time system line"
        );
        assert!(
            parse_error(&c1.replacen("#cP", "#cX", 1))
                .starts_with("SP3: line 1: invalid position/velocity flag 'X'")
        );
        assert!(
            parse_error(&c1.replacen("+   26", "+   27", 1))
                .starts_with("SP3: line 8: header declares 27 satellites but lists 26")
        );
        assert!(
            parse_error(&c1.replacen("G02G03", "G02G02", 1))
                .starts_with("SP3: line 3: satellite G02 is listed twice")
        );
        assert!(
            parse_error(&c1.replacen("%c G ", "%c cc", 1))
                .starts_with("SP3: line 13: unknown file type 'cc'")
        );
        assert!(
            parse_error(&c1.replacen("%c G  cc GPS", "%c G  cc XYZ", 1))
                .starts_with("SP3: line 13: unknown time system 'XYZ'")
        );
    }

    #[test]
    #[parallel]
    fn test_SP3_from_str_version_a_file_type_placeholder() {
        let content = read_fixture("example-c-1.sp3")
            .replacen("#cP", "#bP", 1)
            .replace("PG03", "PR03")
            .replacen("G03G04", "R03G04", 1)
            .replacen("%c G ", "%c cc", 1);
        let sp3 = SP3::from_str(&content).unwrap();
        assert_eq!(sp3.header().version, SP3Version::B);
        assert_eq!(sp3.header().file_type, SP3FileType::Mixed);
        let sp3 = SP3::from_file(fixture("example-a-2.sp3")).unwrap();
        assert_eq!(sp3.header().file_type, SP3FileType::GPS);
    }
}
