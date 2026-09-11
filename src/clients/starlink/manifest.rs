/*!
 * Typed view of Starlink's `MANIFEST.txt`.
 */

use polars::prelude::*;

use crate::clients::spacetrack::{EphemerisFileCategory, EphemerisFileName};
use crate::time::conversions::calendar_from_day_of_year;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

const SECONDS_PER_DAY: f64 = 86400.0;
const GPS_STOP_WINDOW_SECONDS: f64 = 30.0 * SECONDS_PER_DAY;

/// Parses an RFC 7231 HTTP date such as `Fri, 11 Sep 2026 05:15:30 GMT`.
///
/// # Arguments
/// * `value` - Header value
///
/// # Returns
/// * `Some(Epoch)`: The instant in UTC
/// * `None`: If the value is not in the fixed-length IMF-fixdate form
pub(crate) fn parse_http_date(value: &str) -> Option<Epoch> {
    let parts: Vec<&str> = value.split_whitespace().collect();
    if parts.len() != 6 || !parts[0].ends_with(',') || parts[5] != "GMT" {
        return None;
    }
    let day: u8 = parts[1].parse().ok()?;
    let month = match parts[2] {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    };
    let year: u32 = parts[3].parse().ok()?;
    let clock: Vec<&str> = parts[4].split(':').collect();
    if clock.len() != 3 {
        return None;
    }
    let hour: u8 = clock[0].parse().ok()?;
    let minute: u8 = clock[1].parse().ok()?;
    let second: u8 = clock[2].parse().ok()?;
    if !(1..=31).contains(&day) || hour > 23 || minute > 59 || second > 60 {
        return None;
    }
    Some(Epoch::from_datetime(
        year,
        month,
        day,
        hour,
        minute,
        second as f64,
        0.0,
        TimeSystem::UTC,
    ))
}

/// One line of the Starlink manifest with the epochs decoded from the name.
///
/// `ephemeris_start` has minute resolution (the file name carries only
/// `DDDHHMM`); its year comes from `ephemeris_stop` when Starlink's metadata
/// field decodes as GPS seconds, otherwise from the manifest's reference
/// epoch. Starlink starts each file about fifteen minutes before creating
/// it, so `ephemeris_start` is the practical update time per satellite.
///
/// # Examples
///
/// ```
/// use brahe::starlink::StarlinkManifest;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let text = std::fs::read_to_string("test_assets/starlink/MANIFEST.txt").unwrap();
/// let reference = Epoch::from_datetime(2026, 9, 11, 6, 0, 0.0, 0.0, TimeSystem::UTC);
/// let manifest = StarlinkManifest::parse(&text, reference, None).unwrap();
/// let entry = manifest.find_by_norad_id(100001).unwrap();
/// assert_eq!(entry.object_name, "STARLINK-38128");
/// assert!(entry.ephemeris_stop.is_some());
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct StarlinkManifestEntry {
    /// Parsed file name.
    pub file_name: EphemerisFileName,
    /// NORAD catalog number.
    pub norad_cat_id: u32,
    /// Common name, for example `STARLINK-38128`.
    pub object_name: String,
    /// Operational or Special.
    pub category: EphemerisFileCategory,
    /// Ephemeris start, UTC, minute resolution.
    pub ephemeris_start: Epoch,
    /// Ephemeris stop decoded from the metadata field, when present.
    pub ephemeris_stop: Option<Epoch>,
}

impl StarlinkManifestEntry {
    /// The file name as listed in the manifest.
    ///
    /// # Returns
    /// * `String`: Compliant file name with extension
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkManifest;
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let text = std::fs::read_to_string("test_assets/starlink/MANIFEST.txt").unwrap();
    /// let reference = Epoch::from_datetime(2026, 9, 11, 6, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let manifest = StarlinkManifest::parse(&text, reference, None).unwrap();
    /// assert!(manifest.entries()[0].file_name_string().starts_with("MEME_100001_"));
    /// ```
    pub fn file_name_string(&self) -> String {
        self.file_name.to_string()
    }

    /// Builds an entry from a parsed file name, decoding the stop epoch from
    /// the metadata field when possible and placing the start epoch in a year.
    ///
    /// # Arguments
    /// * `file_name` - Parsed Space-Track file name
    /// * `reference` - Epoch used to validate GPS-second metadata and to place the start in a year
    ///
    /// # Returns
    /// * `Ok(StarlinkManifestEntry)`: The decoded entry
    /// * `Err(BraheError)`: If the day of year cannot be placed in a calendar year
    fn from_file_name(file_name: EphemerisFileName, reference: Epoch) -> Result<Self, BraheError> {
        let ephemeris_stop = decode_gps_stop(&file_name.metadata, reference);
        let anchor = ephemeris_stop.unwrap_or(reference);
        let ephemeris_start = start_epoch(&file_name, anchor)?;
        Ok(Self {
            norad_cat_id: file_name.norad_cat_id,
            object_name: file_name.object_name.clone(),
            category: file_name.category,
            ephemeris_start,
            ephemeris_stop,
            file_name,
        })
    }
}

/// Decodes a manifest entry's metadata field as GPS seconds, when it is
/// entirely digits and the resulting epoch falls near `reference`.
///
/// # Arguments
/// * `metadata` - Metadata field of the file name
/// * `reference` - Epoch used to bound plausible GPS-second values
///
/// # Returns
/// * `Some(Epoch)`: The decoded stop epoch
/// * `None`: If the field is not numeric or does not decode near `reference`
fn decode_gps_stop(metadata: &str, reference: Epoch) -> Option<Epoch> {
    if metadata.is_empty() || !metadata.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let seconds: f64 = metadata.parse().ok()?;
    let stop = Epoch::from_gps_seconds(seconds);
    ((stop - reference).abs() <= GPS_STOP_WINDOW_SECONDS).then_some(stop)
}

/// Start epoch for a day-time group, choosing the anchor's calendar year
/// when that places the start no later than a day after `anchor`, and the
/// year before otherwise. A day of year such as 60 is valid in any year
/// (February 29 in a leap year, March 1 otherwise), so the choice is made on
/// ordering alone; the one-day tolerance absorbs clock differences between
/// the file's creation and the manifest's retrieval.
///
/// # Arguments
/// * `name` - Parsed file name carrying the day-of-year, hour and minute
/// * `anchor` - Epoch the start should precede (the stop epoch, or the manifest reference)
///
/// # Returns
/// * `Ok(Epoch)`: The start epoch in UTC
/// * `Err(BraheError)`: If the day of year is invalid in both candidate years
fn start_epoch(name: &EphemerisFileName, anchor: Epoch) -> Result<Epoch, BraheError> {
    let (anchor_year, _, _, _, _, _, _) = anchor.to_datetime_as_time_system(TimeSystem::UTC);
    let candidate = |year: u32| -> Result<Epoch, BraheError> {
        let (month, day) = calendar_from_day_of_year(year, name.day_of_year as u32)?;
        Ok(Epoch::from_datetime(
            year,
            month,
            day,
            name.hour,
            name.minute,
            0.0,
            0.0,
            TimeSystem::UTC,
        ))
    };
    let this_year = candidate(anchor_year);
    let start = match this_year {
        Ok(start) if start <= anchor + SECONDS_PER_DAY => Ok(start),
        _ => candidate(anchor_year.saturating_sub(1)),
    };
    start.map_err(|e| {
        BraheError::ParseError(format!(
            "cannot place day-time group {:03}{:02}{:02} of '{}' near {}: {}",
            name.day_of_year, name.hour, name.minute, name, anchor, e
        ))
    })
}

/// Starlink's manifest as a table of entries.
///
/// # Examples
///
/// ```
/// use brahe::starlink::StarlinkManifest;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let text = std::fs::read_to_string("test_assets/starlink/MANIFEST.txt").unwrap();
/// let reference = Epoch::from_datetime(2026, 9, 11, 6, 0, 0.0, 0.0, TimeSystem::UTC);
/// let manifest = StarlinkManifest::parse(&text, reference, None).unwrap();
/// assert_eq!(manifest.len(), 5);
/// let df = manifest.to_dataframe().unwrap();
/// assert_eq!(df.height(), 5);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct StarlinkManifest {
    entries: Vec<StarlinkManifestEntry>,
    /// `Last-Modified` reported by the server for this listing, when known.
    pub last_modified: Option<Epoch>,
    /// When this copy of the listing was retrieved or cached.
    pub retrieved: Epoch,
}

impl StarlinkManifest {
    /// Parses manifest text; blank lines are skipped and any other line must
    /// be a Space-Track file name.
    ///
    /// # Arguments
    /// * `text` - Manifest contents
    /// * `reference` - Epoch used to place start dates in a year and to validate GPS-second metadata
    /// * `last_modified` - Server `Last-Modified` for the listing, if known
    ///
    /// # Returns
    /// * `Ok(StarlinkManifest)`: The parsed table
    /// * `Err(BraheError)`: If any non-blank line is not a compliant file name
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkManifest;
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let reference = Epoch::from_datetime(2026, 9, 11, 6, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let manifest = StarlinkManifest::parse(
    ///     "MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt\n",
    ///     reference,
    ///     None,
    /// ).unwrap();
    /// assert_eq!(manifest.entries()[0].norad_cat_id, 100001);
    /// ```
    pub fn parse(
        text: &str,
        reference: Epoch,
        last_modified: Option<Epoch>,
    ) -> Result<Self, BraheError> {
        let mut entries = Vec::new();
        for (index, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let name = EphemerisFileName::parse(line).map_err(|e| {
                BraheError::ParseError(format!("Starlink manifest line {}: {}", index + 1, e))
            })?;
            entries.push(StarlinkManifestEntry::from_file_name(name, reference)?);
        }
        Ok(Self {
            entries,
            last_modified,
            retrieved: reference,
        })
    }

    /// All entries in manifest order.
    ///
    /// # Returns
    /// * `&[StarlinkManifestEntry]`: The entries
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkManifest;
    /// use brahe::time::Epoch;
    ///
    /// let manifest = StarlinkManifest::parse("", Epoch::now(), None).unwrap();
    /// assert!(manifest.entries().is_empty());
    /// ```
    pub fn entries(&self) -> &[StarlinkManifestEntry] {
        &self.entries
    }

    /// Iterates over the entries.
    ///
    /// # Returns
    /// * `impl Iterator<Item = &StarlinkManifestEntry>`: Entries in manifest order
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkManifest;
    /// use brahe::time::Epoch;
    ///
    /// let manifest = StarlinkManifest::parse("", Epoch::now(), None).unwrap();
    /// assert_eq!(manifest.iter().count(), 0);
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = &StarlinkManifestEntry> {
        self.entries.iter()
    }

    /// Number of entries.
    ///
    /// # Returns
    /// * `usize`: Entry count
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkManifest;
    /// use brahe::time::Epoch;
    ///
    /// assert_eq!(StarlinkManifest::parse("", Epoch::now(), None).unwrap().len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the manifest has no entries.
    ///
    /// # Returns
    /// * `bool`: `true` when empty
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkManifest;
    /// use brahe::time::Epoch;
    ///
    /// assert!(StarlinkManifest::parse("", Epoch::now(), None).unwrap().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Entry for a NORAD catalog number.
    ///
    /// # Arguments
    /// * `norad_cat_id` - Catalog number
    ///
    /// # Returns
    /// * `Option<&StarlinkManifestEntry>`: The entry, or `None` if absent
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkManifest;
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let text = std::fs::read_to_string("test_assets/starlink/MANIFEST.txt").unwrap();
    /// let reference = Epoch::from_datetime(2026, 9, 11, 6, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let manifest = StarlinkManifest::parse(&text, reference, None).unwrap();
    /// assert!(manifest.find_by_norad_id(100003).is_some());
    /// ```
    pub fn find_by_norad_id(&self, norad_cat_id: u32) -> Option<&StarlinkManifestEntry> {
        self.entries.iter().find(|e| e.norad_cat_id == norad_cat_id)
    }

    /// Entry whose object name matches exactly.
    ///
    /// # Arguments
    /// * `object_name` - Common name, case-sensitive
    ///
    /// # Returns
    /// * `Option<&StarlinkManifestEntry>`: The entry, or `None` if absent
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkManifest;
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let text = std::fs::read_to_string("test_assets/starlink/MANIFEST.txt").unwrap();
    /// let reference = Epoch::from_datetime(2026, 9, 11, 6, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let manifest = StarlinkManifest::parse(&text, reference, None).unwrap();
    /// assert_eq!(manifest.find_by_object_name("STARLINK-38128").unwrap().norad_cat_id, 100001);
    /// ```
    pub fn find_by_object_name(&self, object_name: &str) -> Option<&StarlinkManifestEntry> {
        self.entries.iter().find(|e| e.object_name == object_name)
    }

    /// Entries whose file name differs from, or is absent in, `other`.
    ///
    /// # Arguments
    /// * `other` - An earlier manifest to compare against
    ///
    /// # Returns
    /// * `Vec<&StarlinkManifestEntry>`: New or updated entries, in manifest order
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkManifest;
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let reference = Epoch::from_datetime(2026, 9, 11, 6, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let old = StarlinkManifest::parse("MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt\n", reference, None).unwrap();
    /// let new = StarlinkManifest::parse("MEME_100001_STARLINK-38128_2540942_Operational_1473414180_UNCLASSIFIED.txt\n", reference + 3600.0, None).unwrap();
    /// assert_eq!(new.changed_since(&old).len(), 1);
    /// assert!(old.changed_since(&old).is_empty());
    /// ```
    pub fn changed_since<'a>(&'a self, other: &StarlinkManifest) -> Vec<&'a StarlinkManifestEntry> {
        self.entries
            .iter()
            .filter(|e| {
                other
                    .find_by_norad_id(e.norad_cat_id)
                    .map(|o| o.file_name != e.file_name)
                    .unwrap_or(true)
            })
            .collect()
    }

    /// The manifest as a polars `DataFrame`.
    ///
    /// Columns: `norad_cat_id` (u32), `object_name` (str), `category` (str),
    /// `ephemeris_start` (datetime, milliseconds, UTC), `ephemeris_stop`
    /// (datetime, null when the name carries no stop), `file_name` (str).
    ///
    /// # Returns
    /// * `Ok(DataFrame)`: One row per entry
    /// * `Err(BraheError)`: If polars rejects the columns
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::starlink::StarlinkManifest;
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let text = std::fs::read_to_string("test_assets/starlink/MANIFEST.txt").unwrap();
    /// let reference = Epoch::from_datetime(2026, 9, 11, 6, 0, 0.0, 0.0, TimeSystem::UTC);
    /// let df = StarlinkManifest::parse(&text, reference, None).unwrap().to_dataframe().unwrap();
    /// assert_eq!(df.height(), 5);
    /// ```
    pub fn to_dataframe(&self) -> Result<DataFrame, BraheError> {
        let unix_epoch = Epoch::from_datetime(1970, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let millis = |e: &Epoch| -> i64 { ((*e - unix_epoch) * 1000.0).round() as i64 };
        let datetime = DataType::Datetime(TimeUnit::Milliseconds, None);

        let norad: Column = Series::new(
            "norad_cat_id".into(),
            self.entries
                .iter()
                .map(|e| e.norad_cat_id)
                .collect::<Vec<u32>>(),
        )
        .into();
        let names: Column = Series::new(
            "object_name".into(),
            self.entries
                .iter()
                .map(|e| e.object_name.as_str())
                .collect::<Vec<_>>(),
        )
        .into();
        let categories: Column = Series::new(
            "category".into(),
            self.entries
                .iter()
                .map(|e| e.category.to_string())
                .collect::<Vec<_>>(),
        )
        .into();
        let starts: Column = Series::new(
            "ephemeris_start".into(),
            self.entries
                .iter()
                .map(|e| millis(&e.ephemeris_start))
                .collect::<Vec<i64>>(),
        )
        .cast(&datetime)
        .map_err(|e| BraheError::Error(format!("Failed to build ephemeris_start column: {}", e)))?
        .into();
        let stops: Column = Series::new(
            "ephemeris_stop".into(),
            self.entries
                .iter()
                .map(|e| e.ephemeris_stop.as_ref().map(millis))
                .collect::<Vec<Option<i64>>>(),
        )
        .cast(&datetime)
        .map_err(|e| BraheError::Error(format!("Failed to build ephemeris_stop column: {}", e)))?
        .into();
        let files: Column = Series::new(
            "file_name".into(),
            self.entries
                .iter()
                .map(|e| e.file_name.to_string())
                .collect::<Vec<_>>(),
        )
        .into();

        DataFrame::new(
            self.entries.len(),
            vec![norad, names, categories, starts, stops, files],
        )
        .map_err(|e| {
            BraheError::Error(format!(
                "Failed to create Starlink manifest DataFrame: {}",
                e
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use serial_test::parallel;

    use super::*;
    use crate::time::TimeSystem;

    const FIXTURE: &str = "test_assets/starlink/MANIFEST.txt";

    fn utc(y: u32, mo: u8, d: u8, h: u8, mi: u8, s: f64) -> Epoch {
        Epoch::from_datetime(y, mo, d, h, mi, s, 0.0, TimeSystem::UTC)
    }

    fn reference() -> Epoch {
        utc(2026, 9, 11, 6, 30, 0.0)
    }

    #[test]
    #[parallel]
    fn test_parse_http_date() {
        assert_eq!(
            parse_http_date("Fri, 11 Sep 2026 05:15:30 GMT"),
            Some(utc(2026, 9, 11, 5, 15, 30.0))
        );
        assert_eq!(
            parse_http_date("Mon, 01 Jan 2024 00:00:00 GMT"),
            Some(utc(2024, 1, 1, 0, 0, 0.0))
        );
        assert_eq!(parse_http_date("Fri, 11 Sep 2026 05:15:30"), None);
        assert_eq!(parse_http_date("11 Sep 2026 05:15:30 GMT"), None);
        assert_eq!(parse_http_date("Fri, 11 Xyz 2026 05:15:30 GMT"), None);
        assert_eq!(parse_http_date(""), None);
    }

    #[test]
    #[parallel]
    fn test_manifest_parse_fixture() {
        let text = std::fs::read_to_string(FIXTURE).unwrap();
        let manifest =
            StarlinkManifest::parse(&text, reference(), Some(utc(2026, 9, 11, 5, 15, 30.0)))
                .unwrap();
        assert_eq!(manifest.len(), 5);
        assert_eq!(manifest.retrieved, reference());
        assert_eq!(manifest.last_modified, Some(utc(2026, 9, 11, 5, 15, 30.0)));
        let first = &manifest.entries()[0];
        assert_eq!(first.norad_cat_id, 100001);
        assert_eq!(first.object_name, "STARLINK-38128");
        assert_eq!(first.category, EphemerisFileCategory::Operational);
        assert_eq!(first.ephemeris_start, utc(2026, 9, 11, 1, 42, 0.0));
        assert_eq!(first.ephemeris_stop, Some(utc(2026, 9, 14, 1, 42, 42.0)));
        assert_eq!(
            first.file_name_string(),
            "MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt"
        );
        assert_eq!(
            manifest.find_by_norad_id(100002).unwrap().object_name,
            "STARLINK-37711"
        );
        assert!(manifest.find_by_norad_id(1).is_none());
        assert_eq!(
            manifest
                .find_by_object_name("STARLINK-38123")
                .unwrap()
                .norad_cat_id,
            100003
        );
        assert!(manifest.find_by_object_name("starlink-38123").is_none());
        assert_eq!(manifest.iter().count(), 5);
    }

    #[test]
    #[parallel]
    fn test_manifest_parse_skips_blank_lines_and_rejects_malformed() {
        let text = "MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt\n\n   \nMEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt\n";
        assert_eq!(
            StarlinkManifest::parse(text, reference(), None)
                .unwrap()
                .len(),
            2
        );
        let bad = "MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt\nnot-a-file-name\n";
        assert!(StarlinkManifest::parse(bad, reference(), None).is_err());
        assert!(
            StarlinkManifest::parse("", reference(), None)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    #[parallel]
    fn test_manifest_entry_without_gps_metadata_infers_year_from_reference() {
        let text = "MEME_25544_ISS_2540142_Operational_nomnvr_UNCLASSIFIED.txt\n";
        let m = StarlinkManifest::parse(text, reference(), None).unwrap();
        let e = &m.entries()[0];
        assert!(e.ephemeris_stop.is_none());
        assert_eq!(e.ephemeris_start, utc(2026, 9, 11, 1, 42, 0.0));

        let december = "MEME_25544_ISS_3652300_Operational_nomnvr_UNCLASSIFIED.txt\n";
        let m = StarlinkManifest::parse(december, utc(2027, 1, 2, 0, 0, 0.0), None).unwrap();
        assert_eq!(
            m.entries()[0].ephemeris_start,
            utc(2026, 12, 31, 23, 0, 0.0)
        );

        let numeric_but_not_time = "MEME_25544_ISS_2540142_Operational_42_UNCLASSIFIED.txt\n";
        let m = StarlinkManifest::parse(numeric_but_not_time, reference(), None).unwrap();
        assert!(m.entries()[0].ephemeris_stop.is_none());
        assert_eq!(m.entries()[0].ephemeris_start, utc(2026, 9, 11, 1, 42, 0.0));

        let leap = "MEME_25544_ISS_0600000_Operational_nomnvr_UNCLASSIFIED.txt\n";
        let m = StarlinkManifest::parse(leap, utc(2024, 3, 1, 0, 0, 0.0), None).unwrap();
        assert_eq!(m.entries()[0].ephemeris_start, utc(2024, 2, 29, 0, 0, 0.0));
        let m = StarlinkManifest::parse(leap, utc(2026, 3, 1, 0, 0, 0.0), None).unwrap();
        assert_eq!(m.entries()[0].ephemeris_start, utc(2026, 3, 1, 0, 0, 0.0));

        let day_366 = "MEME_25544_ISS_3660000_Operational_nomnvr_UNCLASSIFIED.txt\n";
        let m = StarlinkManifest::parse(day_366, utc(2025, 1, 2, 0, 0, 0.0), None).unwrap();
        assert_eq!(m.entries()[0].ephemeris_start, utc(2024, 12, 31, 0, 0, 0.0));
        assert!(StarlinkManifest::parse(day_366, utc(2027, 1, 2, 0, 0, 0.0), None).is_err());

        let old_listing = "MEME_25544_ISS_0010000_Operational_nomnvr_UNCLASSIFIED.txt\n";
        let m = StarlinkManifest::parse(old_listing, utc(2026, 12, 30, 0, 0, 0.0), None).unwrap();
        assert_eq!(m.entries()[0].ephemeris_start, utc(2026, 1, 1, 0, 0, 0.0));
    }

    #[test]
    #[parallel]
    fn test_manifest_year_from_gps_stop_across_new_year() {
        let stop = utc(2027, 1, 2, 1, 42, 42.0);
        let gps = stop.gps_seconds().round() as u64;
        let text = format!(
            "MEME_100001_STARLINK-38128_3650142_Operational_{}_UNCLASSIFIED.txt\n",
            gps
        );
        let m = StarlinkManifest::parse(&text, utc(2027, 1, 2, 3, 0, 0.0), None).unwrap();
        let e = &m.entries()[0];
        assert_eq!(e.ephemeris_stop, Some(stop));
        assert_eq!(e.ephemeris_start, utc(2026, 12, 31, 1, 42, 0.0));
    }

    #[test]
    #[parallel]
    fn test_manifest_changed_since() {
        let old = std::fs::read_to_string(FIXTURE).unwrap();
        let a = StarlinkManifest::parse(&old, reference(), None).unwrap();
        let mut lines: Vec<String> = old.lines().map(str::to_string).collect();
        lines[1] = "MEME_100002_STARLINK-37711_2540949_Operational_1473414600_UNCLASSIFIED.txt"
            .to_string();
        lines.remove(4);
        lines.push(
            "MEME_100006_STARLINK-38117_2540137_Operational_1473385080_UNCLASSIFIED.txt"
                .to_string(),
        );
        let b = StarlinkManifest::parse(&lines.join("\n"), reference() + 3600.0, None).unwrap();
        let changed: Vec<u32> = b.changed_since(&a).iter().map(|e| e.norad_cat_id).collect();
        assert_eq!(changed, vec![100002, 100006]);
        let reverse: Vec<u32> = a.changed_since(&b).iter().map(|e| e.norad_cat_id).collect();
        assert_eq!(reverse, vec![100002, 100005]);
        assert!(a.changed_since(&a).is_empty());
    }

    #[test]
    #[parallel]
    fn test_manifest_to_dataframe() {
        let text = std::fs::read_to_string(FIXTURE).unwrap();
        let m = StarlinkManifest::parse(&text, reference(), None).unwrap();
        let df = m.to_dataframe().unwrap();
        assert_eq!(df.height(), 5);
        let names: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            names,
            vec![
                "norad_cat_id",
                "object_name",
                "category",
                "ephemeris_start",
                "ephemeris_stop",
                "file_name"
            ]
        );
        assert_eq!(
            df.column("norad_cat_id").unwrap().u32().unwrap().get(0),
            Some(100001)
        );
        assert_eq!(
            df.column("object_name").unwrap().str().unwrap().get(0),
            Some("STARLINK-38128")
        );
        assert_eq!(
            df.column("category").unwrap().str().unwrap().get(0),
            Some("Operational")
        );
        assert!(matches!(
            df.column("ephemeris_start").unwrap().dtype(),
            DataType::Datetime(TimeUnit::Milliseconds, _)
        ));
        let start_ms = df
            .column("ephemeris_start")
            .unwrap()
            .datetime()
            .unwrap()
            .phys
            .get(0)
            .unwrap();
        let expected_ms =
            ((utc(2026, 9, 11, 1, 42, 0.0) - utc(1970, 1, 1, 0, 0, 0.0)) * 1000.0).round() as i64;
        assert_eq!(start_ms, expected_ms);
        assert_eq!(df.column("ephemeris_stop").unwrap().null_count(), 0);

        let no_stop = StarlinkManifest::parse(
            "MEME_25544_ISS_2540142_Operational_nomnvr_UNCLASSIFIED.txt\n",
            reference(),
            None,
        )
        .unwrap();
        let df2 = no_stop.to_dataframe().unwrap();
        assert_eq!(df2.column("ephemeris_stop").unwrap().null_count(), 1);
    }
}
