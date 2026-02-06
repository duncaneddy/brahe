/*!
 * Typed response structs for SpaceTrack API responses.
 *
 * Provides strongly-typed Rust structs for the most commonly queried
 * SpaceTrack request classes. All fields are `Option<String>` since
 * the API may omit fields or return nulls.
 */

use serde::{Deserialize, Serialize};

/// General Perturbations (OMM) record from the GP request class.
///
/// Contains orbital elements and metadata for a single satellite.
/// All fields are optional strings since Space-Track may omit fields
/// or return null values. Use the field values by parsing them to the
/// desired numeric type.
///
/// # Examples
///
/// ```
/// use brahe::spacetrack::GPRecord;
///
/// let json = r#"[{
///     "CCSDS_OMM_VERS": "3.0",
///     "OBJECT_NAME": "ISS (ZARYA)",
///     "NORAD_CAT_ID": "25544",
///     "EPOCH": "2024-01-15T12:00:00.000",
///     "MEAN_MOTION": "15.50000000",
///     "ECCENTRICITY": "0.00010000",
///     "INCLINATION": "51.6400"
/// }]"#;
///
/// let records: Vec<GPRecord> = serde_json::from_str(json).unwrap();
/// assert_eq!(records[0].object_name.as_deref(), Some("ISS (ZARYA)"));
/// assert_eq!(records[0].norad_cat_id.as_deref(), Some("25544"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub struct GPRecord {
    /// CCSDS OMM version
    #[serde(rename = "CCSDS_OMM_VERS", default)]
    pub ccsds_omm_vers: Option<String>,
    /// Comment field
    #[serde(rename = "COMMENT", default)]
    pub comment: Option<String>,
    /// Record creation date
    #[serde(rename = "CREATION_DATE", default)]
    pub creation_date: Option<String>,
    /// Data originator
    #[serde(rename = "ORIGINATOR", default)]
    pub originator: Option<String>,
    /// Satellite common name
    #[serde(rename = "OBJECT_NAME", default)]
    pub object_name: Option<String>,
    /// International designator
    #[serde(rename = "OBJECT_ID", default)]
    pub object_id: Option<String>,
    /// Center name (typically "EARTH")
    #[serde(rename = "CENTER_NAME", default)]
    pub center_name: Option<String>,
    /// Reference frame (typically "TEME")
    #[serde(rename = "REF_FRAME", default)]
    pub ref_frame: Option<String>,
    /// Time system (typically "UTC")
    #[serde(rename = "TIME_SYSTEM", default)]
    pub time_system: Option<String>,
    /// Mean element theory (typically "SGP4")
    #[serde(rename = "MEAN_ELEMENT_THEORY", default)]
    pub mean_element_theory: Option<String>,
    /// Epoch of the orbital elements
    #[serde(rename = "EPOCH", default)]
    pub epoch: Option<String>,
    /// Mean motion in revolutions per day
    #[serde(rename = "MEAN_MOTION", default)]
    pub mean_motion: Option<String>,
    /// Orbital eccentricity
    #[serde(rename = "ECCENTRICITY", default)]
    pub eccentricity: Option<String>,
    /// Orbital inclination in degrees
    #[serde(rename = "INCLINATION", default)]
    pub inclination: Option<String>,
    /// Right ascension of ascending node in degrees
    #[serde(rename = "RA_OF_ASC_NODE", default)]
    pub ra_of_asc_node: Option<String>,
    /// Argument of pericenter in degrees
    #[serde(rename = "ARG_OF_PERICENTER", default)]
    pub arg_of_pericenter: Option<String>,
    /// Mean anomaly in degrees
    #[serde(rename = "MEAN_ANOMALY", default)]
    pub mean_anomaly: Option<String>,
    /// Ephemeris type
    #[serde(rename = "EPHEMERIS_TYPE", default)]
    pub ephemeris_type: Option<String>,
    /// Classification type (U=Unclassified, C=Classified, S=Secret)
    #[serde(rename = "CLASSIFICATION_TYPE", default)]
    pub classification_type: Option<String>,
    /// NORAD catalog identifier
    #[serde(rename = "NORAD_CAT_ID", default)]
    pub norad_cat_id: Option<String>,
    /// Element set number
    #[serde(rename = "ELEMENT_SET_NO", default)]
    pub element_set_no: Option<String>,
    /// Revolution number at epoch
    #[serde(rename = "REV_AT_EPOCH", default)]
    pub rev_at_epoch: Option<String>,
    /// BSTAR drag coefficient
    #[serde(rename = "BSTAR", default)]
    pub bstar: Option<String>,
    /// First derivative of mean motion
    #[serde(rename = "MEAN_MOTION_DOT", default)]
    pub mean_motion_dot: Option<String>,
    /// Second derivative of mean motion
    #[serde(rename = "MEAN_MOTION_DDOT", default)]
    pub mean_motion_ddot: Option<String>,
    /// Semi-major axis in kilometers
    #[serde(rename = "SEMIMAJOR_AXIS", default)]
    pub semimajor_axis: Option<String>,
    /// Orbital period in minutes
    #[serde(rename = "PERIOD", default)]
    pub period: Option<String>,
    /// Apoapsis altitude in kilometers
    #[serde(rename = "APOAPSIS", default)]
    pub apoapsis: Option<String>,
    /// Periapsis altitude in kilometers
    #[serde(rename = "PERIAPSIS", default)]
    pub periapsis: Option<String>,
    /// Object type (PAYLOAD, ROCKET BODY, DEBRIS, etc.)
    #[serde(rename = "OBJECT_TYPE", default)]
    pub object_type: Option<String>,
    /// Radar cross-section size category (SMALL, MEDIUM, LARGE)
    #[serde(rename = "RCS_SIZE", default)]
    pub rcs_size: Option<String>,
    /// Country code of the launching state
    #[serde(rename = "COUNTRY_CODE", default)]
    pub country_code: Option<String>,
    /// Launch date
    #[serde(rename = "LAUNCH_DATE", default)]
    pub launch_date: Option<String>,
    /// Launch site code
    #[serde(rename = "SITE", default)]
    pub site: Option<String>,
    /// Decay date (if decayed)
    #[serde(rename = "DECAY_DATE", default)]
    pub decay_date: Option<String>,
    /// File number
    #[serde(rename = "FILE", default)]
    pub file: Option<String>,
    /// GP record identifier
    #[serde(rename = "GP_ID", default)]
    pub gp_id: Option<String>,
    /// TLE line 0 (object name)
    #[serde(rename = "TLE_LINE0", default)]
    pub tle_line0: Option<String>,
    /// TLE line 1
    #[serde(rename = "TLE_LINE1", default)]
    pub tle_line1: Option<String>,
    /// TLE line 2
    #[serde(rename = "TLE_LINE2", default)]
    pub tle_line2: Option<String>,
}

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
    #[serde(rename = "NORAD_CAT_ID", default)]
    pub norad_cat_id: Option<String>,
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
/// use brahe::spacetrack::SpEphemerisFileRecord;
///
/// let json = r#"[{
///     "FILE_ID": "99999",
///     "NORAD_CAT_ID": "25544"
/// }]"#;
///
/// let records: Vec<SpEphemerisFileRecord> = serde_json::from_str(json).unwrap();
/// assert_eq!(records[0].file_id.as_deref(), Some("99999"));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpEphemerisFileRecord {
    /// File identifier
    #[serde(rename = "FILE_ID", default)]
    pub file_id: Option<String>,
    /// NORAD catalog identifier
    #[serde(rename = "NORAD_CAT_ID", default)]
    pub norad_cat_id: Option<String>,
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
    fn test_gp_record_deserialize_full() {
        let json = r#"[{
            "CCSDS_OMM_VERS": "3.0",
            "COMMENT": "GENERATED VIA SPACE-TRACK.ORG API",
            "CREATION_DATE": "2024-01-15 12:00:00",
            "ORIGINATOR": "18 SDS",
            "OBJECT_NAME": "ISS (ZARYA)",
            "OBJECT_ID": "1998-067A",
            "CENTER_NAME": "EARTH",
            "REF_FRAME": "TEME",
            "TIME_SYSTEM": "UTC",
            "MEAN_ELEMENT_THEORY": "SGP4",
            "EPOCH": "2024-01-15T12:00:00.000000",
            "MEAN_MOTION": "15.50000000",
            "ECCENTRICITY": "0.00010000",
            "INCLINATION": "51.6400",
            "RA_OF_ASC_NODE": "200.0000",
            "ARG_OF_PERICENTER": "100.0000",
            "MEAN_ANOMALY": "260.0000",
            "EPHEMERIS_TYPE": "0",
            "CLASSIFICATION_TYPE": "U",
            "NORAD_CAT_ID": "25544",
            "ELEMENT_SET_NO": "999",
            "REV_AT_EPOCH": "45000",
            "BSTAR": "0.00034100",
            "MEAN_MOTION_DOT": "0.00001000",
            "MEAN_MOTION_DDOT": "0.00000000",
            "SEMIMAJOR_AXIS": "6793.000",
            "PERIOD": "92.87",
            "APOAPSIS": "415.000",
            "PERIAPSIS": "414.000",
            "OBJECT_TYPE": "PAYLOAD",
            "RCS_SIZE": "LARGE",
            "COUNTRY_CODE": "ISS",
            "LAUNCH_DATE": "1998-11-20",
            "SITE": "TTMTR",
            "DECAY_DATE": null,
            "FILE": "1234",
            "GP_ID": "567890",
            "TLE_LINE0": "0 ISS (ZARYA)",
            "TLE_LINE1": "1 25544U 98067A   24015.50000000  .00001000  00000+0  34100-4 0  9999",
            "TLE_LINE2": "2 25544  51.6400 200.0000 0001000 100.0000 260.0000 15.50000000450001"
        }]"#;

        let records: Vec<GPRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records.len(), 1);

        let record = &records[0];
        assert_eq!(record.object_name.as_deref(), Some("ISS (ZARYA)"));
        assert_eq!(record.norad_cat_id.as_deref(), Some("25544"));
        assert_eq!(record.eccentricity.as_deref(), Some("0.00010000"));
        assert_eq!(record.inclination.as_deref(), Some("51.6400"));
        assert!(record.decay_date.is_none());
        assert!(record.tle_line1.is_some());
    }

    #[test]
    fn test_gp_record_deserialize_minimal() {
        let json = r#"[{
            "OBJECT_NAME": "ISS (ZARYA)",
            "NORAD_CAT_ID": "25544"
        }]"#;

        let records: Vec<GPRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].object_name.as_deref(), Some("ISS (ZARYA)"));
        assert!(records[0].epoch.is_none());
        assert!(records[0].mean_motion.is_none());
    }

    #[test]
    fn test_gp_record_unknown_fields_ignored() {
        let json = r#"[{
            "OBJECT_NAME": "ISS (ZARYA)",
            "UNKNOWN_FIELD": "some_value",
            "ANOTHER_UNKNOWN": 42
        }]"#;

        let records: Vec<GPRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].object_name.as_deref(), Some("ISS (ZARYA)"));
    }

    #[test]
    fn test_gp_record_empty_array() {
        let json = "[]";
        let records: Vec<GPRecord> = serde_json::from_str(json).unwrap();
        assert!(records.is_empty());
    }

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
        assert_eq!(record.norad_cat_id.as_deref(), Some("25544"));
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

    #[test]
    fn test_gp_record_clone() {
        let json = r#"[{"OBJECT_NAME": "ISS", "NORAD_CAT_ID": "25544"}]"#;
        let records: Vec<GPRecord> = serde_json::from_str(json).unwrap();
        let cloned = records[0].clone();
        assert_eq!(cloned.object_name, records[0].object_name);
    }

    #[test]
    fn test_gp_record_debug() {
        let json = r#"[{"OBJECT_NAME": "ISS"}]"#;
        let records: Vec<GPRecord> = serde_json::from_str(json).unwrap();
        let debug = format!("{:?}", records[0]);
        assert!(debug.contains("ISS"));
    }

    #[test]
    fn test_gp_record_serialize() {
        let json = r#"[{"OBJECT_NAME":"ISS","NORAD_CAT_ID":"25544"}]"#;
        let records: Vec<GPRecord> = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&records[0]).unwrap();
        assert!(serialized.contains("OBJECT_NAME"));
        assert!(serialized.contains("ISS"));
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

    // -- SpEphemerisFileRecord tests --

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

        let records: Vec<SpEphemerisFileRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].file_id.as_deref(), Some("99999"));
        assert_eq!(records[0].norad_cat_id.as_deref(), Some("25544"));
        assert_eq!(records[0].file_name.as_deref(), Some("iss_sp.e"));
        assert_eq!(
            records[0].epoch_start.as_deref(),
            Some("2024-01-14T00:00:00")
        );
    }

    #[test]
    fn test_spephemeris_file_record_minimal() {
        let json = r#"[{"FILE_ID": "99999"}]"#;
        let records: Vec<SpEphemerisFileRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records[0].file_id.as_deref(), Some("99999"));
        assert!(records[0].norad_cat_id.is_none());
    }
}
