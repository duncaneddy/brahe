/*!
 * Typed response structs for CelestrakClient API responses.
 *
 * Provides a strongly-typed Rust struct for CelestrakClient's SATCAT endpoint.
 * GP query responses use [`crate::types::GPRecord`] for interoperability
 * with SpaceTrack GP data.
 *
 * All fields are `Option<String>` (except NORAD_CAT_ID which is `Option<u32>`)
 * since the API may omit fields or return nulls.
 */

use crate::types::serde_flex::*;
use serde::{Deserialize, Serialize};

/// CelestrakClient Satellite Catalog (SATCAT) record.
///
/// Contains metadata about a cataloged space object from CelestrakClient's
/// SATCAT endpoint (`/satcat/records.php`). This has different field names
/// than SpaceTrack's SATCAT, reflecting CelestrakClient's own schema.
///
/// # Examples
///
/// ```
/// use brahe::celestrak::CelestrakSATCATRecord;
///
/// let json = r#"[{
///     "OBJECT_NAME": "ISS (ZARYA)",
///     "NORAD_CAT_ID": "25544",
///     "OBJECT_TYPE": "PAY",
///     "OPS_STATUS_CODE": "+",
///     "OWNER": "ISS"
/// }]"#;
///
/// let records: Vec<CelestrakSATCATRecord> = serde_json::from_str(json).unwrap();
/// assert_eq!(records[0].object_name.as_deref(), Some("ISS (ZARYA)"));
/// assert_eq!(records[0].norad_cat_id, Some(25544));
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub struct CelestrakSATCATRecord {
    /// Object common name
    #[serde(
        rename = "OBJECT_NAME",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub object_name: Option<String>,
    /// International designator (e.g. "1998-067A")
    #[serde(
        rename = "OBJECT_ID",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub object_id: Option<String>,
    /// NORAD catalog identifier
    #[serde(
        rename = "NORAD_CAT_ID",
        default,
        deserialize_with = "flex_u32::deserialize"
    )]
    pub norad_cat_id: Option<u32>,
    /// Object type code (PAY, R/B, DEB, etc.)
    #[serde(
        rename = "OBJECT_TYPE",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub object_type: Option<String>,
    /// Operational status code (+, -, P, B, S, X, D, ?)
    #[serde(
        rename = "OPS_STATUS_CODE",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub ops_status_code: Option<String>,
    /// Owner/operator country or organization code
    #[serde(
        rename = "OWNER",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub owner: Option<String>,
    /// Launch date (YYYY-MM-DD)
    #[serde(
        rename = "LAUNCH_DATE",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub launch_date: Option<String>,
    /// Launch site code
    #[serde(
        rename = "LAUNCH_SITE",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub launch_site: Option<String>,
    /// Decay date (YYYY-MM-DD), if decayed
    #[serde(
        rename = "DECAY_DATE",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub decay_date: Option<String>,
    /// Orbital period in minutes
    #[serde(
        rename = "PERIOD",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub period: Option<String>,
    /// Orbital inclination in degrees
    #[serde(
        rename = "INCLINATION",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub inclination: Option<String>,
    /// Apogee altitude in kilometers
    #[serde(
        rename = "APOGEE",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub apogee: Option<String>,
    /// Perigee altitude in kilometers
    #[serde(
        rename = "PERIGEE",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub perigee: Option<String>,
    /// Radar cross-section in square meters
    #[serde(rename = "RCS", default, deserialize_with = "flex_string::deserialize")]
    pub rcs: Option<String>,
    /// Data status code
    #[serde(
        rename = "DATA_STATUS_CODE",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub data_status_code: Option<String>,
    /// Orbit center (e.g. "EA" for Earth)
    #[serde(
        rename = "ORBIT_CENTER",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub orbit_center: Option<String>,
    /// Orbit type (e.g. "ORB" for orbiting)
    #[serde(
        rename = "ORBIT_TYPE",
        default,
        deserialize_with = "flex_string::deserialize"
    )]
    pub orbit_type: Option<String>,
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_celestrak_satcat_record_deserialize_full() {
        let json = r#"[{
            "OBJECT_NAME": "ISS (ZARYA)",
            "OBJECT_ID": "1998-067A",
            "NORAD_CAT_ID": "25544",
            "OBJECT_TYPE": "PAY",
            "OPS_STATUS_CODE": "+",
            "OWNER": "ISS",
            "LAUNCH_DATE": "1998-11-20",
            "LAUNCH_SITE": "TTMTR",
            "DECAY_DATE": null,
            "PERIOD": "92.87",
            "INCLINATION": "51.64",
            "APOGEE": "415",
            "PERIGEE": "414",
            "RCS": "0.0000",
            "DATA_STATUS_CODE": "",
            "ORBIT_CENTER": "EA",
            "ORBIT_TYPE": "ORB"
        }]"#;

        let records: Vec<CelestrakSATCATRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records.len(), 1);

        let record = &records[0];
        assert_eq!(record.object_name.as_deref(), Some("ISS (ZARYA)"));
        assert_eq!(record.object_id.as_deref(), Some("1998-067A"));
        assert_eq!(record.norad_cat_id, Some(25544));
        assert_eq!(record.object_type.as_deref(), Some("PAY"));
        assert_eq!(record.ops_status_code.as_deref(), Some("+"));
        assert_eq!(record.owner.as_deref(), Some("ISS"));
        assert_eq!(record.launch_date.as_deref(), Some("1998-11-20"));
        assert_eq!(record.launch_site.as_deref(), Some("TTMTR"));
        assert!(record.decay_date.is_none());
        assert_eq!(record.period.as_deref(), Some("92.87"));
        assert_eq!(record.inclination.as_deref(), Some("51.64"));
        assert_eq!(record.apogee.as_deref(), Some("415"));
        assert_eq!(record.perigee.as_deref(), Some("414"));
        assert_eq!(record.orbit_center.as_deref(), Some("EA"));
        assert_eq!(record.orbit_type.as_deref(), Some("ORB"));
    }

    #[test]
    fn test_celestrak_satcat_record_deserialize_minimal() {
        let json = r#"[{
            "OBJECT_NAME": "ISS (ZARYA)",
            "NORAD_CAT_ID": "25544"
        }]"#;

        let records: Vec<CelestrakSATCATRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].object_name.as_deref(), Some("ISS (ZARYA)"));
        assert_eq!(records[0].norad_cat_id, Some(25544));
        assert!(records[0].period.is_none());
        assert!(records[0].decay_date.is_none());
    }

    #[test]
    fn test_celestrak_satcat_record_unknown_fields_ignored() {
        let json = r#"[{
            "OBJECT_NAME": "ISS (ZARYA)",
            "UNKNOWN_FIELD": "some_value",
            "ANOTHER_UNKNOWN": 42
        }]"#;

        let records: Vec<CelestrakSATCATRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].object_name.as_deref(), Some("ISS (ZARYA)"));
    }

    #[test]
    fn test_celestrak_satcat_record_empty_array() {
        let json = "[]";
        let records: Vec<CelestrakSATCATRecord> = serde_json::from_str(json).unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn test_celestrak_satcat_record_clone() {
        let json = r#"[{"OBJECT_NAME": "ISS", "NORAD_CAT_ID": "25544"}]"#;
        let records: Vec<CelestrakSATCATRecord> = serde_json::from_str(json).unwrap();
        let cloned = records[0].clone();
        assert_eq!(cloned.object_name, records[0].object_name);
    }

    #[test]
    fn test_celestrak_satcat_record_debug() {
        let json = r#"[{"OBJECT_NAME": "ISS"}]"#;
        let records: Vec<CelestrakSATCATRecord> = serde_json::from_str(json).unwrap();
        let debug = format!("{:?}", records[0]);
        assert!(debug.contains("ISS"));
    }

    #[test]
    fn test_celestrak_satcat_record_serialize() {
        let json = r#"[{"OBJECT_NAME":"ISS","NORAD_CAT_ID":"25544"}]"#;
        let records: Vec<CelestrakSATCATRecord> = serde_json::from_str(json).unwrap();
        let serialized = serde_json::to_string(&records[0]).unwrap();
        assert!(serialized.contains("OBJECT_NAME"));
        assert!(serialized.contains("ISS"));
    }

    #[test]
    fn test_celestrak_satcat_record_numeric_norad_cat_id() {
        let json = r#"[{"OBJECT_NAME": "ISS", "NORAD_CAT_ID": 25544}]"#;
        let records: Vec<CelestrakSATCATRecord> = serde_json::from_str(json).unwrap();
        assert_eq!(records[0].norad_cat_id, Some(25544));
    }
}
