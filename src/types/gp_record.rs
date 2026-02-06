/*!
 * General Perturbations (OMM) record shared by SpaceTrack and Celestrak.
 *
 * This struct is the canonical representation of GP data from both data
 * sources. Fields are properly typed per the Space-Track database schema,
 * with flexible serde deserializers that accept both string and numeric
 * JSON representations.
 */

use serde::{Deserialize, Serialize};
use std::borrow::Cow;

use super::serde_flex::*;

/// Trait for accessing record fields by name as strings.
///
/// Enables the filter engine to work with any record type that can
/// provide field values by their API field name. Returns `Cow<str>`
/// so string fields can borrow and numeric fields can convert.
pub(crate) trait FieldAccessor {
    fn get_field(&self, name: &str) -> Option<Cow<'_, str>>;
}

/// General Perturbations (OMM) record from the GP request class.
///
/// Contains orbital elements and metadata for a single satellite.
/// Numeric fields are properly typed (f64, u32, etc.) per the
/// Space-Track database schema. Flexible deserializers handle both
/// string values (SpaceTrack) and numeric values (Celestrak).
///
/// # Examples
///
/// ```
/// use brahe::types::GPRecord;
///
/// // SpaceTrack format (all strings)
/// let json = r#"[{
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
/// assert_eq!(records[0].norad_cat_id, Some(25544));
/// assert_eq!(records[0].inclination, Some(51.64));
///
/// // Celestrak format (native numbers)
/// let json = r#"[{
///     "OBJECT_NAME": "ISS (ZARYA)",
///     "NORAD_CAT_ID": 25544,
///     "MEAN_MOTION": 15.5,
///     "ECCENTRICITY": 0.0001,
///     "INCLINATION": 51.64
/// }]"#;
///
/// let records: Vec<GPRecord> = serde_json::from_str(json).unwrap();
/// assert_eq!(records[0].norad_cat_id, Some(25544));
/// assert_eq!(records[0].eccentricity, Some(0.0001));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub struct GPRecord {
    /// CCSDS OMM version
    #[serde(
        rename = "CCSDS_OMM_VERS",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub ccsds_omm_vers: Option<String>,
    /// Comment field
    #[serde(
        rename = "COMMENT",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub comment: Option<String>,
    /// Record creation date
    #[serde(
        rename = "CREATION_DATE",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub creation_date: Option<String>,
    /// Data originator
    #[serde(
        rename = "ORIGINATOR",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub originator: Option<String>,
    /// Satellite common name
    #[serde(
        rename = "OBJECT_NAME",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub object_name: Option<String>,
    /// International designator
    #[serde(
        rename = "OBJECT_ID",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub object_id: Option<String>,
    /// Center name (typically "EARTH")
    #[serde(
        rename = "CENTER_NAME",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub center_name: Option<String>,
    /// Reference frame (typically "TEME")
    #[serde(
        rename = "REF_FRAME",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub ref_frame: Option<String>,
    /// Time system (typically "UTC")
    #[serde(
        rename = "TIME_SYSTEM",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub time_system: Option<String>,
    /// Mean element theory (typically "SGP4")
    #[serde(
        rename = "MEAN_ELEMENT_THEORY",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub mean_element_theory: Option<String>,
    /// Epoch of the orbital elements
    #[serde(
        rename = "EPOCH",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub epoch: Option<String>,
    /// Mean motion in revolutions per day
    #[serde(
        rename = "MEAN_MOTION",
        default,
        deserialize_with = "flex_f64::deserialize"
    )]
    pub mean_motion: Option<f64>,
    /// Orbital eccentricity
    #[serde(
        rename = "ECCENTRICITY",
        default,
        deserialize_with = "flex_f64::deserialize"
    )]
    pub eccentricity: Option<f64>,
    /// Orbital inclination in degrees
    #[serde(
        rename = "INCLINATION",
        default,
        deserialize_with = "flex_f64::deserialize"
    )]
    pub inclination: Option<f64>,
    /// Right ascension of ascending node in degrees
    #[serde(
        rename = "RA_OF_ASC_NODE",
        default,
        deserialize_with = "flex_f64::deserialize"
    )]
    pub ra_of_asc_node: Option<f64>,
    /// Argument of pericenter in degrees
    #[serde(
        rename = "ARG_OF_PERICENTER",
        default,
        deserialize_with = "flex_f64::deserialize"
    )]
    pub arg_of_pericenter: Option<f64>,
    /// Mean anomaly in degrees
    #[serde(
        rename = "MEAN_ANOMALY",
        default,
        deserialize_with = "flex_f64::deserialize"
    )]
    pub mean_anomaly: Option<f64>,
    /// Ephemeris type
    #[serde(
        rename = "EPHEMERIS_TYPE",
        default,
        deserialize_with = "flex_u8::deserialize"
    )]
    pub ephemeris_type: Option<u8>,
    /// Classification type (U=Unclassified, C=Classified, S=Secret)
    #[serde(
        rename = "CLASSIFICATION_TYPE",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub classification_type: Option<String>,
    /// NORAD catalog identifier
    #[serde(
        rename = "NORAD_CAT_ID",
        default,
        deserialize_with = "flex_u32::deserialize"
    )]
    pub norad_cat_id: Option<u32>,
    /// Element set number
    #[serde(
        rename = "ELEMENT_SET_NO",
        default,
        deserialize_with = "flex_u16::deserialize"
    )]
    pub element_set_no: Option<u16>,
    /// Revolution number at epoch
    #[serde(
        rename = "REV_AT_EPOCH",
        default,
        deserialize_with = "flex_u32::deserialize"
    )]
    pub rev_at_epoch: Option<u32>,
    /// BSTAR drag coefficient
    #[serde(rename = "BSTAR", default, deserialize_with = "flex_f64::deserialize")]
    pub bstar: Option<f64>,
    /// First derivative of mean motion
    #[serde(
        rename = "MEAN_MOTION_DOT",
        default,
        deserialize_with = "flex_f64::deserialize"
    )]
    pub mean_motion_dot: Option<f64>,
    /// Second derivative of mean motion
    #[serde(
        rename = "MEAN_MOTION_DDOT",
        default,
        deserialize_with = "flex_f64::deserialize"
    )]
    pub mean_motion_ddot: Option<f64>,
    /// Semi-major axis in kilometers
    #[serde(
        rename = "SEMIMAJOR_AXIS",
        default,
        deserialize_with = "flex_f64::deserialize"
    )]
    pub semimajor_axis: Option<f64>,
    /// Orbital period in minutes
    #[serde(rename = "PERIOD", default, deserialize_with = "flex_f64::deserialize")]
    pub period: Option<f64>,
    /// Apoapsis altitude in kilometers
    #[serde(
        rename = "APOAPSIS",
        default,
        deserialize_with = "flex_f64::deserialize"
    )]
    pub apoapsis: Option<f64>,
    /// Periapsis altitude in kilometers
    #[serde(
        rename = "PERIAPSIS",
        default,
        deserialize_with = "flex_f64::deserialize"
    )]
    pub periapsis: Option<f64>,
    /// Object type (PAYLOAD, ROCKET BODY, DEBRIS, etc.)
    #[serde(
        rename = "OBJECT_TYPE",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub object_type: Option<String>,
    /// Radar cross-section size category (SMALL, MEDIUM, LARGE)
    #[serde(
        rename = "RCS_SIZE",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub rcs_size: Option<String>,
    /// Country code of the launching state
    #[serde(
        rename = "COUNTRY_CODE",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub country_code: Option<String>,
    /// Launch date
    #[serde(
        rename = "LAUNCH_DATE",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub launch_date: Option<String>,
    /// Launch site code
    #[serde(
        rename = "SITE",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub site: Option<String>,
    /// Decay date (if decayed)
    #[serde(
        rename = "DECAY_DATE",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub decay_date: Option<String>,
    /// File number
    #[serde(rename = "FILE", default, deserialize_with = "flex_u64::deserialize")]
    pub file: Option<u64>,
    /// GP record identifier
    #[serde(rename = "GP_ID", default, deserialize_with = "flex_u32::deserialize")]
    pub gp_id: Option<u32>,
    /// TLE line 0 (object name)
    #[serde(
        rename = "TLE_LINE0",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub tle_line0: Option<String>,
    /// TLE line 1
    #[serde(
        rename = "TLE_LINE1",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub tle_line1: Option<String>,
    /// TLE line 2
    #[serde(
        rename = "TLE_LINE2",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub tle_line2: Option<String>,
}

impl FieldAccessor for GPRecord {
    fn get_field(&self, name: &str) -> Option<Cow<'_, str>> {
        match name {
            "CCSDS_OMM_VERS" => self.ccsds_omm_vers.as_deref().map(Cow::Borrowed),
            "COMMENT" => self.comment.as_deref().map(Cow::Borrowed),
            "CREATION_DATE" => self.creation_date.as_deref().map(Cow::Borrowed),
            "ORIGINATOR" => self.originator.as_deref().map(Cow::Borrowed),
            "OBJECT_NAME" => self.object_name.as_deref().map(Cow::Borrowed),
            "OBJECT_ID" => self.object_id.as_deref().map(Cow::Borrowed),
            "CENTER_NAME" => self.center_name.as_deref().map(Cow::Borrowed),
            "REF_FRAME" => self.ref_frame.as_deref().map(Cow::Borrowed),
            "TIME_SYSTEM" => self.time_system.as_deref().map(Cow::Borrowed),
            "MEAN_ELEMENT_THEORY" => self.mean_element_theory.as_deref().map(Cow::Borrowed),
            "EPOCH" => self.epoch.as_deref().map(Cow::Borrowed),
            "MEAN_MOTION" => self.mean_motion.map(|v| Cow::Owned(v.to_string())),
            "ECCENTRICITY" => self.eccentricity.map(|v| Cow::Owned(v.to_string())),
            "INCLINATION" => self.inclination.map(|v| Cow::Owned(v.to_string())),
            "RA_OF_ASC_NODE" => self.ra_of_asc_node.map(|v| Cow::Owned(v.to_string())),
            "ARG_OF_PERICENTER" => self.arg_of_pericenter.map(|v| Cow::Owned(v.to_string())),
            "MEAN_ANOMALY" => self.mean_anomaly.map(|v| Cow::Owned(v.to_string())),
            "EPHEMERIS_TYPE" => self.ephemeris_type.map(|v| Cow::Owned(v.to_string())),
            "CLASSIFICATION_TYPE" => self.classification_type.as_deref().map(Cow::Borrowed),
            "NORAD_CAT_ID" => self.norad_cat_id.map(|v| Cow::Owned(v.to_string())),
            "ELEMENT_SET_NO" => self.element_set_no.map(|v| Cow::Owned(v.to_string())),
            "REV_AT_EPOCH" => self.rev_at_epoch.map(|v| Cow::Owned(v.to_string())),
            "BSTAR" => self.bstar.map(|v| Cow::Owned(v.to_string())),
            "MEAN_MOTION_DOT" => self.mean_motion_dot.map(|v| Cow::Owned(v.to_string())),
            "MEAN_MOTION_DDOT" => self.mean_motion_ddot.map(|v| Cow::Owned(v.to_string())),
            "SEMIMAJOR_AXIS" => self.semimajor_axis.map(|v| Cow::Owned(v.to_string())),
            "PERIOD" => self.period.map(|v| Cow::Owned(v.to_string())),
            "APOAPSIS" => self.apoapsis.map(|v| Cow::Owned(v.to_string())),
            "PERIAPSIS" => self.periapsis.map(|v| Cow::Owned(v.to_string())),
            "OBJECT_TYPE" => self.object_type.as_deref().map(Cow::Borrowed),
            "RCS_SIZE" => self.rcs_size.as_deref().map(Cow::Borrowed),
            "COUNTRY_CODE" => self.country_code.as_deref().map(Cow::Borrowed),
            "LAUNCH_DATE" => self.launch_date.as_deref().map(Cow::Borrowed),
            "SITE" => self.site.as_deref().map(Cow::Borrowed),
            "DECAY_DATE" => self.decay_date.as_deref().map(Cow::Borrowed),
            "FILE" => self.file.map(|v| Cow::Owned(v.to_string())),
            "GP_ID" => self.gp_id.map(|v| Cow::Owned(v.to_string())),
            "TLE_LINE0" => self.tle_line0.as_deref().map(Cow::Borrowed),
            "TLE_LINE1" => self.tle_line1.as_deref().map(Cow::Borrowed),
            "TLE_LINE2" => self.tle_line2.as_deref().map(Cow::Borrowed),
            _ => None,
        }
    }
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
        assert_eq!(record.norad_cat_id, Some(25544));
        assert_eq!(record.eccentricity, Some(0.0001));
        assert_eq!(record.inclination, Some(51.64));
        assert!(record.decay_date.is_none());
        assert!(record.tle_line1.is_some());
        assert_eq!(record.ephemeris_type, Some(0));
        assert_eq!(record.element_set_no, Some(999));
        assert_eq!(record.rev_at_epoch, Some(45000));
        assert_eq!(record.file, Some(1234));
        assert_eq!(record.gp_id, Some(567890));
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
        assert_eq!(records[0].norad_cat_id, Some(25544));
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
    fn test_gp_record_deserialize_numeric_values() {
        // Celestrak returns numeric values as JSON numbers, not strings
        let json = r#"[{
            "OBJECT_NAME": "ISS (ZARYA)",
            "OBJECT_ID": "1998-067A",
            "EPOCH": "2026-02-06T05:05:23.180640",
            "MEAN_MOTION": 15.48432747,
            "ECCENTRICITY": 0.0011199,
            "INCLINATION": 51.6316,
            "RA_OF_ASC_NODE": 227.9611,
            "ARG_OF_PERICENTER": 69.2605,
            "MEAN_ANOMALY": 290.9582,
            "EPHEMERIS_TYPE": 0,
            "CLASSIFICATION_TYPE": "U",
            "NORAD_CAT_ID": 25544,
            "ELEMENT_SET_NO": 999,
            "REV_AT_EPOCH": 55145,
            "BSTAR": 0.00024775,
            "MEAN_MOTION_DOT": 0.00012979,
            "MEAN_MOTION_DDOT": 0
        }]"#;

        let records: Vec<GPRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records.len(), 1);

        let record = &records[0];
        assert_eq!(record.object_name.as_deref(), Some("ISS (ZARYA)"));
        assert_eq!(record.norad_cat_id, Some(25544));
        assert_eq!(record.inclination, Some(51.6316));
        assert_eq!(record.eccentricity, Some(0.0011199));
        assert_eq!(record.ephemeris_type, Some(0));
        assert_eq!(record.mean_motion_ddot, Some(0.0));
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

    #[test]
    fn test_field_accessor_string_fields() {
        let json = r#"{"OBJECT_NAME": "ISS (ZARYA)", "OBJECT_TYPE": "PAYLOAD"}"#;
        let record: GPRecord = serde_json::from_str(json).unwrap();
        assert_eq!(
            record.get_field("OBJECT_NAME").as_deref(),
            Some("ISS (ZARYA)")
        );
        assert_eq!(record.get_field("OBJECT_TYPE").as_deref(), Some("PAYLOAD"));
    }

    #[test]
    fn test_field_accessor_numeric_fields() {
        let json = r#"{"NORAD_CAT_ID": 25544, "INCLINATION": 51.64, "ECCENTRICITY": 0.0001}"#;
        let record: GPRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.get_field("NORAD_CAT_ID").as_deref(), Some("25544"));
        assert_eq!(record.get_field("INCLINATION").as_deref(), Some("51.64"));
        assert_eq!(record.get_field("ECCENTRICITY").as_deref(), Some("0.0001"));
    }

    #[test]
    fn test_field_accessor_missing_field() {
        let json = r#"{"OBJECT_NAME": "ISS"}"#;
        let record: GPRecord = serde_json::from_str(json).unwrap();
        assert!(record.get_field("EPOCH").is_none());
        assert!(record.get_field("NONEXISTENT").is_none());
    }
}
