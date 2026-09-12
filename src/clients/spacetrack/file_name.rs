/*!
 * Space-Track ephemeris file-name convention.
 *
 * Space-Track requires operator ephemeris submissions to be named
 * `<DataType>_<Catalog#>_<CommonName>_<DayTimeGroup>_<Operational/Special>_<MetaData>_<Classification>.<Extension>`
 * (Spaceflight Safety Handbook for Operators, "How to Name Ephemeris Files").
 * [`EphemerisFileName`] parses and generates names in that convention.
 */

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::time::conversions::day_of_year_from_calendar;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

/// Whether an ephemeris file describes the planned (operational) trajectory
/// or a special-case alternative.
///
/// Space-Track accepts one operational file per satellite at a time and any
/// number of special files.
///
/// # Examples
///
/// ```
/// use brahe::spacetrack::EphemerisFileCategory;
///
/// assert_eq!(EphemerisFileCategory::parse("oper").unwrap(), EphemerisFileCategory::Operational);
/// assert_eq!(EphemerisFileCategory::Special.to_string(), "Special");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EphemerisFileCategory {
    /// The trajectory the satellite is planned to fly, including routine maneuvers.
    Operational,
    /// A planning-only trajectory that assumes a special maneuver.
    Special,
}

impl EphemerisFileCategory {
    /// Parses the category field of a file name.
    ///
    /// Matching is case-insensitive on the prefix: any token starting with
    /// `oper` is `Operational` and any token starting with `special` is
    /// `Special`, which covers the handbook's `oper`, `operational` and
    /// `special` spellings.
    ///
    /// # Arguments
    /// * `token` - The category field of a file name
    ///
    /// # Returns
    /// * `Ok(EphemerisFileCategory)`: The parsed category
    /// * `Err(BraheError)`: If the token is not a recognized category
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::spacetrack::EphemerisFileCategory;
    ///
    /// assert_eq!(EphemerisFileCategory::parse("Operational").unwrap(), EphemerisFileCategory::Operational);
    /// assert!(EphemerisFileCategory::parse("planned").is_err());
    /// ```
    pub fn parse(token: &str) -> Result<Self, BraheError> {
        let lower = token.to_ascii_lowercase();
        if lower.starts_with("oper") {
            Ok(Self::Operational)
        } else if lower.starts_with("special") {
            Ok(Self::Special)
        } else {
            Err(BraheError::ParseError(format!(
                "unknown ephemeris file category '{}'; expected Operational or Special",
                token
            )))
        }
    }
}

impl fmt::Display for EphemerisFileCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operational => write!(f, "Operational"),
            Self::Special => write!(f, "Special"),
        }
    }
}

/// A Space-Track ephemeris file name, field by field.
///
/// `Display` renders the compliant name. The catalog number is zero-padded to
/// five digits, which matches both the handbook's `00900` example and
/// Starlink's six-digit and nine-digit identifiers.
///
/// # Examples
///
/// ```
/// use brahe::spacetrack::{EphemerisFileCategory, EphemerisFileName};
/// use brahe::time::{Epoch, TimeSystem};
///
/// let start = Epoch::from_datetime(2026, 9, 11, 1, 42, 42.0, 0.0, TimeSystem::UTC);
/// let name = EphemerisFileName::new(100001, "STARLINK-38128", start, EphemerisFileCategory::Operational, "").unwrap();
/// assert_eq!(name.to_string(), "MEME_100001_STARLINK-38128_2540142_Operational__UNCLASSIFIED.txt");
///
/// let parsed = EphemerisFileName::parse(&name.to_string()).unwrap();
/// assert_eq!(parsed.norad_cat_id, 100001);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EphemerisFileName {
    /// Data type field; normatively `MEME`.
    pub data_type: String,
    /// NORAD catalog number, or the nine-digit analyst number for uncataloged objects.
    pub norad_cat_id: u32,
    /// Common name of the object.
    pub object_name: String,
    /// Day of year of the ephemeris start, UTC (1 to 366).
    pub day_of_year: u16,
    /// Hour of the ephemeris start, UTC.
    pub hour: u8,
    /// Minute of the ephemeris start, UTC.
    pub minute: u8,
    /// Operational or Special.
    pub category: EphemerisFileCategory,
    /// Operator-defined metadata; may be empty.
    pub metadata: String,
    /// Classification field, kept as written.
    pub classification: String,
    /// File extension without the dot.
    pub extension: String,
}

/// Characters that turn a field into a path rather than part of a file name.
const FORBIDDEN_FIELD_CHARACTERS: [char; 3] = ['/', '\\', '\0'];

/// Validates that a file-name field names no path of its own, so that a name
/// assembled from the fields always addresses a file inside a single
/// directory.
///
/// # Arguments
/// * `field` - Field name, used in the error message
/// * `value` - Field value, which may be empty or contain `_`
///
/// # Returns
/// * `Ok(())`: The value is safe to place in a file name
/// * `Err(BraheError)`: If the value contains `/`, `\` or NUL, or is `.` or `..`
fn validate_path_field(field: &str, value: &str) -> Result<(), BraheError> {
    if value.contains(FORBIDDEN_FIELD_CHARACTERS) {
        return Err(BraheError::Error(format!(
            "invalid ephemeris file name field {}: '{}' must not contain '/', '\\' or NUL",
            field, value
        )));
    }
    if value == "." || value == ".." {
        return Err(BraheError::Error(format!(
            "invalid ephemeris file name field {}: '{}' must not be '.' or '..'",
            field, value
        )));
    }
    Ok(())
}

/// Validates that a single-token file-name field is non-empty, does not
/// contain the `_` delimiter, and names no path of its own.
///
/// # Arguments
/// * `field` - Field name, used in the error message
/// * `value` - Field value
///
/// # Returns
/// * `Ok(())`: The value is a valid single-token field
/// * `Err(BraheError)`: If the value is empty, contains `_`, `/`, `\` or NUL, or is `.` or `..`
fn validate_field(field: &str, value: &str) -> Result<(), BraheError> {
    if value.is_empty() {
        return Err(BraheError::Error(format!(
            "invalid ephemeris file name field {}: value must not be empty",
            field
        )));
    }
    if value.contains('_') {
        return Err(BraheError::Error(format!(
            "invalid ephemeris file name field {}: '{}' must not contain '_', '/', '\\' or NUL",
            field, value
        )));
    }
    validate_path_field(field, value)
}

impl EphemerisFileName {
    /// Default data type, the mean equator and mean equinox of J2000.0.
    pub const DEFAULT_DATA_TYPE: &'static str = "MEME";
    /// Default classification.
    pub const DEFAULT_CLASSIFICATION: &'static str = "UNCLASSIFIED";
    /// Default extension.
    pub const DEFAULT_EXTENSION: &'static str = "txt";

    /// Builds a file name with the `MEME` data type, `UNCLASSIFIED`
    /// classification and `txt` extension.
    ///
    /// # Arguments
    /// * `norad_cat_id` - NORAD catalog number or analyst number
    /// * `object_name` - Common name of the object
    /// * `start_epoch` - Ephemeris start; the day-time group is taken in UTC
    /// * `category` - Operational or Special
    /// * `metadata` - Operator-defined metadata, may be empty
    ///
    /// # Returns
    /// * `Ok(EphemerisFileName)`: The populated name
    /// * `Err(BraheError)`: If `object_name` is empty, if `metadata` contains `_`, or if either contains `/`, `\` or NUL or is `.` or `..`
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::spacetrack::{EphemerisFileCategory, EphemerisFileName};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let start = Epoch::from_datetime(2020, 10, 26, 12, 24, 0.0, 0.0, TimeSystem::UTC);
    /// let name = EphemerisFileName::new(25544, "ISS", start, EphemerisFileCategory::Operational, "nomnvr").unwrap();
    /// assert_eq!(name.to_string(), "MEME_25544_ISS_3001224_Operational_nomnvr_UNCLASSIFIED.txt");
    /// ```
    pub fn new(
        norad_cat_id: u32,
        object_name: &str,
        start_epoch: Epoch,
        category: EphemerisFileCategory,
        metadata: &str,
    ) -> Result<Self, BraheError> {
        if object_name.is_empty() {
            return Err(BraheError::Error(
                "invalid ephemeris file name field object_name: value must not be empty"
                    .to_string(),
            ));
        }
        validate_path_field("object_name", object_name)?;
        if metadata.contains('_') {
            return Err(BraheError::Error(format!(
                "invalid ephemeris file name field metadata: '{}' must not contain '_'",
                metadata
            )));
        }
        validate_path_field("metadata", metadata)?;
        let (year, month, day, hour, minute, _, _) =
            start_epoch.to_datetime_as_time_system(TimeSystem::UTC);
        let day_of_year = day_of_year_from_calendar(year, month, day) as u16;
        Ok(Self {
            data_type: Self::DEFAULT_DATA_TYPE.to_string(),
            norad_cat_id,
            object_name: object_name.to_string(),
            day_of_year,
            hour,
            minute,
            category,
            metadata: metadata.to_string(),
            classification: Self::DEFAULT_CLASSIFICATION.to_string(),
            extension: Self::DEFAULT_EXTENSION.to_string(),
        })
    }

    /// Sets the data type field.
    ///
    /// # Arguments
    /// * `data_type` - Data type token, for example `MEME` or `TEME`
    ///
    /// # Returns
    /// * `Ok(EphemerisFileName)`: The name with the data type replaced
    /// * `Err(BraheError)`: If `data_type` is empty, contains `_`, `/`, `\` or NUL, or is `.` or `..`
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::spacetrack::{EphemerisFileCategory, EphemerisFileName};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let start = Epoch::from_datetime(2026, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let name = EphemerisFileName::new(1, "A", start, EphemerisFileCategory::Special, "").unwrap().with_data_type("TEME").unwrap();
    /// assert!(name.to_string().starts_with("TEME_"));
    /// ```
    pub fn with_data_type(mut self, data_type: &str) -> Result<Self, BraheError> {
        validate_field("data_type", data_type)?;
        self.data_type = data_type.to_string();
        Ok(self)
    }

    /// Sets the classification field.
    ///
    /// # Arguments
    /// * `classification` - Classification token
    ///
    /// # Returns
    /// * `Ok(EphemerisFileName)`: The name with the classification replaced
    /// * `Err(BraheError)`: If `classification` is empty, contains `_`, `/`, `\` or NUL, or is `.` or `..`
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::spacetrack::{EphemerisFileCategory, EphemerisFileName};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let start = Epoch::from_datetime(2026, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let name = EphemerisFileName::new(1, "A", start, EphemerisFileCategory::Special, "").unwrap().with_classification("unclassified").unwrap();
    /// assert!(name.to_string().ends_with("_unclassified.txt"));
    /// ```
    pub fn with_classification(mut self, classification: &str) -> Result<Self, BraheError> {
        validate_field("classification", classification)?;
        self.classification = classification.to_string();
        Ok(self)
    }

    /// Sets the file extension (without the dot).
    ///
    /// # Arguments
    /// * `extension` - Extension without a leading dot
    ///
    /// # Returns
    /// * `Ok(EphemerisFileName)`: The name with the extension replaced
    /// * `Err(BraheError)`: If `extension` is empty or contains `_`, `.`, `/`, `\` or NUL
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::spacetrack::{EphemerisFileCategory, EphemerisFileName};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let start = Epoch::from_datetime(2026, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let name = EphemerisFileName::new(1, "A", start, EphemerisFileCategory::Special, "").unwrap().with_extension("dat").unwrap();
    /// assert!(name.to_string().ends_with(".dat"));
    /// ```
    pub fn with_extension(mut self, extension: &str) -> Result<Self, BraheError> {
        validate_field("extension", extension)?;
        if extension.contains('.') {
            return Err(BraheError::Error(format!(
                "invalid ephemeris file name field extension: '{}' must not contain '.'",
                extension
            )));
        }
        self.extension = extension.to_string();
        Ok(self)
    }

    /// Parses a file name in the Space-Track convention.
    ///
    /// The first two fields and the last four fields are fixed; anything
    /// between them, underscores included, is the object name. The day-time
    /// group must be exactly seven digits (`DDDHHMM`) with a valid day of
    /// year, hour and minute. Every text field is held to the same rules as
    /// the builders: no `/`, `\` or NUL, and no field that is `.` or `..`, so
    /// a parsed name always addresses a file inside a single directory.
    ///
    /// # Arguments
    /// * `name` - File name with extension, without directory components
    ///
    /// # Returns
    /// * `Ok(EphemerisFileName)`: The parsed fields
    /// * `Err(BraheError)`: If the name does not follow the convention or a field names a path
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::spacetrack::{EphemerisFileCategory, EphemerisFileName};
    ///
    /// let name = EphemerisFileName::parse("MEME_25544_ISS(ZARYA)_1651200_operational_nomnvr_UNCLASSIFIED.txt").unwrap();
    /// assert_eq!(name.object_name, "ISS(ZARYA)");
    /// assert_eq!((name.day_of_year, name.hour, name.minute), (165, 12, 0));
    /// assert_eq!(name.category, EphemerisFileCategory::Operational);
    /// ```
    pub fn parse(name: &str) -> Result<Self, BraheError> {
        let err = |detail: String| {
            BraheError::ParseError(format!(
                "invalid ephemeris file name '{}': {}",
                name, detail
            ))
        };

        let (stem, extension) = name
            .rsplit_once('.')
            .ok_or_else(|| err("missing file extension".to_string()))?;
        if extension.is_empty() {
            return Err(err("empty file extension".to_string()));
        }

        let parts: Vec<&str> = stem.split('_').collect();
        if parts.len() < 7 {
            return Err(err(format!(
                "expected at least seven underscore-separated fields, found {}",
                parts.len()
            )));
        }
        let n = parts.len();

        let data_type = parts[0];
        if data_type.is_empty() {
            return Err(err("empty data type".to_string()));
        }
        validate_field("data_type", data_type).map_err(|e| err(e.to_string()))?;
        let norad_cat_id: u32 = parts[1]
            .parse()
            .map_err(|_| err(format!("catalog number '{}' is not an integer", parts[1])))?;
        let object_name = parts[2..n - 4].join("_");
        if object_name.is_empty() {
            return Err(err("empty object name".to_string()));
        }
        validate_path_field("object_name", &object_name).map_err(|e| err(e.to_string()))?;

        let day_time_group = parts[n - 4];
        if day_time_group.len() != 7 || !day_time_group.chars().all(|c| c.is_ascii_digit()) {
            return Err(err(format!(
                "day-time group '{}' must be seven digits DDDHHMM",
                day_time_group
            )));
        }
        let day_of_year: u16 = day_time_group[0..3].parse().unwrap();
        let hour: u8 = day_time_group[3..5].parse().unwrap();
        let minute: u8 = day_time_group[5..7].parse().unwrap();
        if !(1..=366).contains(&day_of_year) {
            return Err(err(format!(
                "day of year {} out of range 1..=366",
                day_of_year
            )));
        }
        if hour > 23 {
            return Err(err(format!("hour {} out of range 0..=23", hour)));
        }
        if minute > 59 {
            return Err(err(format!("minute {} out of range 0..=59", minute)));
        }

        let category =
            EphemerisFileCategory::parse(parts[n - 3]).map_err(|e| err(e.to_string()))?;
        let metadata = parts[n - 2];
        validate_path_field("metadata", metadata).map_err(|e| err(e.to_string()))?;
        let classification = parts[n - 1];
        if classification.is_empty() {
            return Err(err("empty classification".to_string()));
        }
        validate_field("classification", classification).map_err(|e| err(e.to_string()))?;
        validate_field("extension", extension).map_err(|e| err(e.to_string()))?;

        Ok(Self {
            data_type: data_type.to_string(),
            norad_cat_id,
            object_name,
            day_of_year,
            hour,
            minute,
            category,
            metadata: metadata.to_string(),
            classification: classification.to_string(),
            extension: extension.to_string(),
        })
    }
}

impl fmt::Display for EphemerisFileName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}_{:05}_{}_{:03}{:02}{:02}_{}_{}_{}.{}",
            self.data_type,
            self.norad_cat_id,
            self.object_name,
            self.day_of_year,
            self.hour,
            self.minute,
            self.category,
            self.metadata,
            self.classification,
            self.extension
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use serial_test::parallel;

    #[test]
    #[parallel]
    fn test_ephemeris_file_name_parse_starlink() {
        let name = EphemerisFileName::parse(
            "MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt",
        )
        .unwrap();
        assert_eq!(name.data_type, "MEME");
        assert_eq!(name.norad_cat_id, 100001);
        assert_eq!(name.object_name, "STARLINK-38128");
        assert_eq!(name.day_of_year, 254);
        assert_eq!(name.hour, 1);
        assert_eq!(name.minute, 42);
        assert_eq!(name.category, EphemerisFileCategory::Operational);
        assert_eq!(name.metadata, "1473385380");
        assert_eq!(name.classification, "UNCLASSIFIED");
        assert_eq!(name.extension, "txt");
    }

    #[test]
    #[parallel]
    fn test_ephemeris_file_name_parse_handbook_examples() {
        let a = EphemerisFileName::parse("MEME_25544_ISS_1651200_oper__unclassified.txt").unwrap();
        assert_eq!(a.norad_cat_id, 25544);
        assert_eq!(a.object_name, "ISS");
        assert_eq!((a.day_of_year, a.hour, a.minute), (165, 12, 0));
        assert_eq!(a.category, EphemerisFileCategory::Operational);
        assert_eq!(a.metadata, "");
        assert_eq!(a.classification, "unclassified");

        let b = EphemerisFileName::parse(
            "MEME_25544_ISS(ZARYA)_1651200_operational_nomnvr_UNCLASSIFIED.txt",
        )
        .unwrap();
        assert_eq!(b.object_name, "ISS(ZARYA)");
        assert_eq!(b.metadata, "nomnvr");

        let c = EphemerisFileName::parse(
            "MEME_799500234_Sat1_1651200_special_separation_unclassified.txt",
        )
        .unwrap();
        assert_eq!(c.norad_cat_id, 799500234);
        assert_eq!(c.category, EphemerisFileCategory::Special);
        assert_eq!(c.metadata, "separation");
    }

    #[test]
    #[parallel]
    fn test_ephemeris_file_name_parse_object_name_with_underscores() {
        let name =
            EphemerisFileName::parse("MEME_12345_MY_SAT_A_0010530_Special_burn02_UNCLASSIFIED.txt")
                .unwrap();
        assert_eq!(name.object_name, "MY_SAT_A");
        assert_eq!((name.day_of_year, name.hour, name.minute), (1, 5, 30));
        assert_eq!(name.metadata, "burn02");
    }

    #[test]
    #[parallel]
    fn test_ephemeris_file_name_parse_errors() {
        assert!(EphemerisFileName::parse("MEME_25544_ISS_1651200_oper__unclassified").is_err());
        assert!(EphemerisFileName::parse("MEME_25544_ISS_1651200_oper.txt").is_err());
        assert!(EphemerisFileName::parse("MEME_abc_ISS_1651200_oper__unclassified.txt").is_err());
        assert!(EphemerisFileName::parse("MEME_25544_ISS_165120_oper__unclassified.txt").is_err());
        assert!(EphemerisFileName::parse("MEME_25544_ISS_1652500_oper__unclassified.txt").is_err());
        assert!(EphemerisFileName::parse("MEME_25544_ISS_1651260_oper__unclassified.txt").is_err());
        assert!(EphemerisFileName::parse("MEME_25544_ISS_3671200_oper__unclassified.txt").is_err());
        assert!(
            EphemerisFileName::parse("MEME_25544_ISS_1651200_planned__unclassified.txt").is_err()
        );
        assert!(EphemerisFileName::parse("MEME_25544__1651200_oper__unclassified.txt").is_err());
        assert!(EphemerisFileName::parse("MEME_25544_ISS_1651200_oper__unclassified.").is_err());
        assert!(EphemerisFileName::parse("_25544_ISS_1651200_oper__unclassified.txt").is_err());
        assert!(EphemerisFileName::parse("MEME_25544_ISS_1651200_oper_meta_.txt").is_err());
    }

    #[test]
    #[parallel]
    fn test_ephemeris_file_name_parse_rejects_path_traversal() {
        for bad in [
            "MEME_100001_../../../../tmp/evil_2540142_Operational__UNCLASSIFIED.txt",
            "/tmp/pwn_100001_X_2540142_Operational__UNCLASSIFIED.txt",
            "MEME_100001_..\\..\\tmp\\evil_2540142_Operational__UNCLASSIFIED.txt",
            "MEME_100001_.._2540142_Operational__UNCLASSIFIED.txt",
            "MEME_100001_X_2540142_Operational_../evil_UNCLASSIFIED.txt",
            "MEME_100001_X_2540142_Operational__UNCLASSIFIED.txt/../evil",
            "MEME_100001_X_2540142_Operational__UNCLASS/IFIED.txt",
            "MEME_100001_X_2540142_Operational__UNCLASSIFIED.t\0xt",
        ] {
            assert!(
                EphemerisFileName::parse(bad).is_err(),
                "expected '{bad}' to be rejected"
            );
        }
    }

    #[test]
    #[parallel]
    fn test_ephemeris_file_name_builders_reject_path_traversal() {
        let start = Epoch::from_datetime(2026, 9, 11, 1, 42, 42.0, 0.0, TimeSystem::UTC);
        assert!(
            EphemerisFileName::new(1, "..", start, EphemerisFileCategory::Operational, "").is_err()
        );
        assert!(
            EphemerisFileName::new(
                1,
                "../../tmp/evil",
                start,
                EphemerisFileCategory::Operational,
                ""
            )
            .is_err()
        );
        assert!(
            EphemerisFileName::new(1, "A\\B", start, EphemerisFileCategory::Operational, "")
                .is_err()
        );
        assert!(
            EphemerisFileName::new(1, "A", start, EphemerisFileCategory::Operational, "../evil")
                .is_err()
        );

        let name =
            EphemerisFileName::new(1, "A", start, EphemerisFileCategory::Operational, "").unwrap();
        assert!(name.clone().with_data_type("..").is_err());
        assert!(name.clone().with_classification("A\\B").is_err());
        assert!(name.with_extension("t\0xt").is_err());
    }

    #[test]
    #[parallel]
    fn test_ephemeris_file_name_new_and_display() {
        let start = Epoch::from_datetime(2026, 9, 11, 1, 42, 42.0, 0.0, TimeSystem::UTC);
        let name = EphemerisFileName::new(
            100001,
            "STARLINK-38128",
            start,
            EphemerisFileCategory::Operational,
            "1473385380",
        )
        .unwrap();
        assert_eq!(
            name.to_string(),
            "MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt"
        );

        let padded = EphemerisFileName::new(
            900,
            "CALSPHERE 1",
            start,
            EphemerisFileCategory::Special,
            "",
        )
        .unwrap();
        assert_eq!(
            padded.to_string(),
            "MEME_00900_CALSPHERE 1_2540142_Special__UNCLASSIFIED.txt"
        );

        let custom = EphemerisFileName::new(
            25544,
            "ISS",
            start,
            EphemerisFileCategory::Operational,
            "nomnvr",
        )
        .unwrap()
        .with_data_type("TEME")
        .unwrap()
        .with_classification("unclassified")
        .unwrap()
        .with_extension("dat")
        .unwrap();
        assert_eq!(
            custom.to_string(),
            "TEME_25544_ISS_2540142_Operational_nomnvr_unclassified.dat"
        );
    }

    #[test]
    #[parallel]
    fn test_ephemeris_file_name_round_trip() {
        for original in [
            "MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt",
            "MEME_69995_STARLINK-38084_2540139_Operational_1473385200_UNCLASSIFIED.txt",
            "MEME_799501571_STARLINK-36331_2540207_Operational_1473386880_UNCLASSIFIED.txt",
        ] {
            assert_eq!(
                EphemerisFileName::parse(original).unwrap().to_string(),
                original
            );
        }
    }

    #[test]
    #[parallel]
    fn test_ephemeris_file_name_uses_utc_for_day_time_group() {
        let start = Epoch::from_datetime(2026, 9, 11, 1, 42, 42.0, 0.0, TimeSystem::UTC);
        let in_tai = Epoch::from_datetime(2026, 9, 11, 1, 43, 19.0, 0.0, TimeSystem::TAI);
        let a =
            EphemerisFileName::new(1, "A", start, EphemerisFileCategory::Operational, "").unwrap();
        let b =
            EphemerisFileName::new(1, "A", in_tai, EphemerisFileCategory::Operational, "").unwrap();
        assert_eq!(a.to_string(), b.to_string());
    }

    #[test]
    #[parallel]
    fn test_ephemeris_file_name_new_leap_second_day_of_year() {
        let leap: Epoch = Epoch::from_datetime(2017, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC) - 1.0;
        let name =
            EphemerisFileName::new(1, "A", leap, EphemerisFileCategory::Operational, "").unwrap();
        assert_eq!((name.day_of_year, name.hour, name.minute), (366, 23, 59));
        assert_eq!(
            name.to_string(),
            "MEME_00001_A_3662359_Operational__UNCLASSIFIED.txt"
        );
    }

    #[test]
    #[parallel]
    fn test_ephemeris_file_category_parse_and_display() {
        assert_eq!(
            EphemerisFileCategory::parse("oper").unwrap(),
            EphemerisFileCategory::Operational
        );
        assert_eq!(
            EphemerisFileCategory::parse("OPERATIONAL").unwrap(),
            EphemerisFileCategory::Operational
        );
        assert_eq!(
            EphemerisFileCategory::parse("Special").unwrap(),
            EphemerisFileCategory::Special
        );
        assert!(EphemerisFileCategory::parse("planned").is_err());
        assert_eq!(
            EphemerisFileCategory::Operational.to_string(),
            "Operational"
        );
        assert_eq!(EphemerisFileCategory::Special.to_string(), "Special");
    }

    #[test]
    #[parallel]
    fn test_ephemeris_file_name_rejects_delimiter_in_fields() {
        let start = Epoch::from_datetime(2026, 9, 11, 1, 42, 42.0, 0.0, TimeSystem::UTC);
        assert!(
            EphemerisFileName::new(1, "A", start, EphemerisFileCategory::Operational, "burn_02")
                .is_err()
        );
        assert!(
            EphemerisFileName::new(1, "", start, EphemerisFileCategory::Operational, "").is_err()
        );

        let name =
            EphemerisFileName::new(1, "A", start, EphemerisFileCategory::Operational, "").unwrap();
        assert!(name.clone().with_extension("txt.bak").is_err());
        assert!(name.clone().with_classification("UN_CLASS").is_err());
        assert!(name.with_data_type("MEME/EXTRA").is_err());
    }
}
