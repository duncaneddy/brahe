/*!
 * Typed response structs for SpaceTrack API responses.
 *
 * Provides strongly-typed Rust structs for commonly queried
 * SpaceTrack request classes (SATCAT, FileShare, SP Ephemeris).
 *
 * The [`GPRecord`] type is defined in [`crate::types`] and re-exported
 * from this module's parent for backward compatibility.
 */

use crate::types::serde_flex::*;
use serde::{Deserialize, Serialize};

/// Satellite Catalog (SATCAT) record.
///
/// Contains metadata about a cataloged space object including its name,
/// country of origin, launch information, and current orbital parameters.
///
/// # Examples
///
/// ```
/// use brahe::spacetrack::SATCATRecord;
///
/// let json = r#"[{
///     "SATNAME": "ISS (ZARYA)",
///     "NORAD_CAT_ID": "25544",
///     "OBJECT_TYPE": "PAY",
///     "COUNTRY": "ISS"
/// }]"#;
///
/// let records: Vec<SATCATRecord> = serde_json::from_str(json).unwrap();
/// assert_eq!(records[0].satname.as_deref(), Some("ISS (ZARYA)"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub struct SATCATRecord {
    /// International designator
    #[serde(rename = "INTLDES", default)]
    pub intldes: Option<String>,
    /// NORAD catalog identifier
    #[serde(
        rename = "NORAD_CAT_ID",
        default,
        deserialize_with = "flex_u32::deserialize"
    )]
    pub norad_cat_id: Option<u32>,
    /// Object type code
    #[serde(rename = "OBJECT_TYPE", default)]
    pub object_type: Option<String>,
    /// Satellite name
    #[serde(rename = "SATNAME", default)]
    pub satname: Option<String>,
    /// Country/organization code
    #[serde(rename = "COUNTRY", default)]
    pub country: Option<String>,
    /// Launch date
    #[serde(rename = "LAUNCH", default)]
    pub launch: Option<String>,
    /// Launch site code
    #[serde(rename = "SITE", default)]
    pub site: Option<String>,
    /// Decay date (if decayed)
    #[serde(rename = "DECAY", default)]
    pub decay: Option<String>,
    /// Orbital period in minutes
    #[serde(rename = "PERIOD", default)]
    pub period: Option<String>,
    /// Orbital inclination in degrees
    #[serde(rename = "INCLINATION", default)]
    pub inclination: Option<String>,
    /// Apogee altitude in kilometers
    #[serde(rename = "APOGEE", default)]
    pub apogee: Option<String>,
    /// Perigee altitude in kilometers
    #[serde(rename = "PERIGEE", default)]
    pub perigee: Option<String>,
    /// Comment field
    #[serde(rename = "COMMENT", default)]
    pub comment: Option<String>,
    /// Comment code
    #[serde(rename = "COMMENTCODE", default)]
    pub commentcode: Option<String>,
    /// Radar cross-section value
    #[serde(rename = "RCSVALUE", default)]
    pub rcsvalue: Option<String>,
    /// Radar cross-section size category
    #[serde(rename = "RCS_SIZE", default)]
    pub rcs_size: Option<String>,
    /// File number
    #[serde(rename = "FILE", default)]
    pub file: Option<String>,
    /// Launch year
    #[serde(rename = "LAUNCH_YEAR", default)]
    pub launch_year: Option<String>,
    /// Launch number within year
    #[serde(rename = "LAUNCH_NUM", default)]
    pub launch_num: Option<String>,
    /// Launch piece designator
    #[serde(rename = "LAUNCH_PIECE", default)]
    pub launch_piece: Option<String>,
    /// Current status flag
    #[serde(rename = "CURRENT", default)]
    pub current: Option<String>,
    /// Object name
    #[serde(rename = "OBJECT_NAME", default)]
    pub object_name: Option<String>,
    /// Object identifier
    #[serde(rename = "OBJECT_ID", default)]
    pub object_id: Option<String>,
    /// Object number
    #[serde(rename = "OBJECT_NUMBER", default)]
    pub object_number: Option<String>,
}

/// FileShare file record from the fileshare/file request class.
///
/// Contains metadata about a file in a user's Space-Track file share.
/// All fields are optional strings since Space-Track may omit fields
/// or return null values.
///
/// # Examples
///
/// ```
/// use brahe::spacetrack::FileShareFileRecord;
///
/// let json = r#"[{
///     "FILE_ID": "12345",
///     "FILE_NAME": "data.txt",
///     "FILE_LINK": "/fileshare/download/file_id/12345",
///     "FOLDER_ID": "100"
/// }]"#;
///
/// let records: Vec<FileShareFileRecord> = serde_json::from_str(json).unwrap();
/// assert_eq!(records[0].file_id.as_deref(), Some("12345"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileShareFileRecord {
    /// File identifier
    #[serde(rename = "FILE_ID", default)]
    pub file_id: Option<String>,
    /// File name
    #[serde(rename = "FILE_NAME", default)]
    pub file_name: Option<String>,
    /// File download link
    #[serde(rename = "FILE_LINK", default)]
    pub file_link: Option<String>,
    /// File size in bytes
    #[serde(rename = "FILE_SIZE", default)]
    pub file_size: Option<String>,
    /// File content type
    #[serde(rename = "FILE_CONTTYPE", default)]
    pub file_conttype: Option<String>,
    /// Folder identifier
    #[serde(rename = "FOLDER_ID", default)]
    pub folder_id: Option<String>,
    /// Creation date
    #[serde(rename = "CREATED", default)]
    pub created: Option<String>,
}

/// FileShare folder record from the fileshare/folder request class.
///
/// Contains metadata about a folder in a user's Space-Track file share.
///
/// # Examples
///
/// ```
/// use brahe::spacetrack::FolderRecord;
///
/// let json = r#"[{
///     "FOLDER_ID": "100",
///     "FOLDER_NAME": "my_data"
/// }]"#;
///
/// let records: Vec<FolderRecord> = serde_json::from_str(json).unwrap();
/// assert_eq!(records[0].folder_id.as_deref(), Some("100"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FolderRecord {
    /// Folder identifier
    #[serde(rename = "FOLDER_ID", default)]
    pub folder_id: Option<String>,
    /// Folder name
    #[serde(rename = "FOLDER_NAME", default)]
    pub folder_name: Option<String>,
    /// Parent folder identifier
    #[serde(rename = "PARENT_FOLDER_ID", default)]
    pub parent_folder_id: Option<String>,
    /// Creation date
    #[serde(rename = "CREATED", default)]
    pub created: Option<String>,
}

/// SP Ephemeris file record from the spephemeris/file request class.
///
/// Contains metadata about an SP ephemeris file on Space-Track.
///
/// # Examples
///
/// ```
/// use brahe::spacetrack::SPEphemerisFileRecord;
///
/// let json = r#"[{
///     "FILE_ID": "99999",
///     "NORAD_CAT_ID": "25544"
/// }]"#;
///
/// let records: Vec<SPEphemerisFileRecord> = serde_json::from_str(json).unwrap();
/// assert_eq!(records[0].file_id.as_deref(), Some("99999"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SPEphemerisFileRecord {
    /// File identifier
    #[serde(rename = "FILE_ID", default)]
    pub file_id: Option<String>,
    /// NORAD catalog identifier
    #[serde(
        rename = "NORAD_CAT_ID",
        default,
        deserialize_with = "flex_u32::deserialize"
    )]
    pub norad_cat_id: Option<u32>,
    /// File name
    #[serde(rename = "FILE_NAME", default)]
    pub file_name: Option<String>,
    /// File download link
    #[serde(rename = "FILE_LINK", default)]
    pub file_link: Option<String>,
    /// File size in bytes
    #[serde(rename = "FILE_SIZE", default)]
    pub file_size: Option<String>,
    /// Creation date
    #[serde(rename = "CREATED", default)]
    pub created: Option<String>,
    /// Epoch start
    #[serde(rename = "EPOCH_START", default)]
    pub epoch_start: Option<String>,
    /// Epoch stop
    #[serde(rename = "EPOCH_STOP", default)]
    pub epoch_stop: Option<String>,
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_satcat_record_deserialize() {
        let json = r#"[{
            "INTLDES": "1998-067A",
            "NORAD_CAT_ID": "25544",
            "OBJECT_TYPE": "PAY",
            "SATNAME": "ISS (ZARYA)",
            "COUNTRY": "ISS",
            "LAUNCH": "1998-11-20",
            "SITE": "TTMTR",
            "DECAY": null,
            "PERIOD": "92.87",
            "INCLINATION": "51.64",
            "APOGEE": "415",
            "PERIGEE": "414",
            "COMMENT": "",
            "COMMENTCODE": "",
            "RCSVALUE": "0",
            "RCS_SIZE": "LARGE",
            "FILE": "1234",
            "LAUNCH_YEAR": "1998",
            "LAUNCH_NUM": "067",
            "LAUNCH_PIECE": "A",
            "CURRENT": "Y",
            "OBJECT_NAME": "ISS (ZARYA)",
            "OBJECT_ID": "1998-067A",
            "OBJECT_NUMBER": "25544"
        }]"#;

        let records: Vec<SATCATRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records.len(), 1);

        let record = &records[0];
        assert_eq!(record.satname.as_deref(), Some("ISS (ZARYA)"));
        assert_eq!(record.norad_cat_id, Some(25544));
        assert_eq!(record.object_type.as_deref(), Some("PAY"));
        assert_eq!(record.country.as_deref(), Some("ISS"));
        assert!(record.decay.is_none());
    }

    #[test]
    fn test_satcat_record_unknown_fields_ignored() {
        let json = r#"[{
            "SATNAME": "ISS",
            "FUTURE_FIELD": "value"
        }]"#;

        let records: Vec<SATCATRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records[0].satname.as_deref(), Some("ISS"));
    }

    // -- FileShareFileRecord tests --

    #[test]
    fn test_fileshare_file_record_deserialize() {
        let json = r#"[{
            "FILE_ID": "12345",
            "FILE_NAME": "data.txt",
            "FILE_LINK": "/fileshare/download/file_id/12345",
            "FILE_SIZE": "1024",
            "FILE_CONTTYPE": "text/plain",
            "FOLDER_ID": "100",
            "CREATED": "2024-01-15"
        }]"#;

        let records: Vec<FileShareFileRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].file_id.as_deref(), Some("12345"));
        assert_eq!(records[0].file_name.as_deref(), Some("data.txt"));
        assert_eq!(records[0].folder_id.as_deref(), Some("100"));
    }

    #[test]
    fn test_fileshare_file_record_minimal() {
        let json = r#"[{"FILE_ID": "12345"}]"#;
        let records: Vec<FileShareFileRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records[0].file_id.as_deref(), Some("12345"));
        assert!(records[0].file_name.is_none());
    }

    #[test]
    fn test_fileshare_file_record_unknown_fields() {
        let json = r#"[{"FILE_ID": "12345", "UNKNOWN": "value"}]"#;
        let records: Vec<FileShareFileRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records[0].file_id.as_deref(), Some("12345"));
    }

    // -- FolderRecord tests --

    #[test]
    fn test_folder_record_deserialize() {
        let json = r#"[{
            "FOLDER_ID": "100",
            "FOLDER_NAME": "my_data",
            "PARENT_FOLDER_ID": "50",
            "CREATED": "2024-01-15"
        }]"#;

        let records: Vec<FolderRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].folder_id.as_deref(), Some("100"));
        assert_eq!(records[0].folder_name.as_deref(), Some("my_data"));
        assert_eq!(records[0].parent_folder_id.as_deref(), Some("50"));
    }

    #[test]
    fn test_folder_record_minimal() {
        let json = r#"[{"FOLDER_ID": "100"}]"#;
        let records: Vec<FolderRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records[0].folder_id.as_deref(), Some("100"));
        assert!(records[0].folder_name.is_none());
    }

    // -- SPEphemerisFileRecord tests --

    #[test]
    fn test_spephemeris_file_record_deserialize() {
        let json = r#"[{
            "FILE_ID": "99999",
            "NORAD_CAT_ID": "25544",
            "FILE_NAME": "iss_sp.e",
            "FILE_LINK": "/spephemeris/download/99999",
            "FILE_SIZE": "2048",
            "CREATED": "2024-01-15",
            "EPOCH_START": "2024-01-14T00:00:00",
            "EPOCH_STOP": "2024-01-15T00:00:00"
        }]"#;

        let records: Vec<SPEphemerisFileRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].file_id.as_deref(), Some("99999"));
        assert_eq!(records[0].norad_cat_id, Some(25544));
        assert_eq!(records[0].file_name.as_deref(), Some("iss_sp.e"));
        assert_eq!(
            records[0].epoch_start.as_deref(),
            Some("2024-01-14T00:00:00")
        );
    }

    #[test]
    fn test_spephemeris_file_record_minimal() {
        let json = r#"[{"FILE_ID": "99999"}]"#;
        let records: Vec<SPEphemerisFileRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records[0].file_id.as_deref(), Some("99999"));
        assert!(records[0].norad_cat_id.is_none());
    }

    #[test]
    fn test_spephemeris_file_record_numeric_norad_cat_id() {
        let json = r#"[{"FILE_ID": "99999", "NORAD_CAT_ID": 25544}]"#;
        let records: Vec<SPEphemerisFileRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records[0].norad_cat_id, Some(25544));
    }

    #[test]
    fn test_satcat_record_numeric_norad_cat_id() {
        let json = r#"[{"SATNAME": "ISS", "NORAD_CAT_ID": 25544}]"#;
        let records: Vec<SATCATRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records[0].norad_cat_id, Some(25544));
    }
}
