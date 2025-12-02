/*!
 * SATCAT (Satellite Catalog) request class
 *
 * The SATCAT class provides access to the satellite catalog containing
 * information about all cataloged space objects.
 */

use serde::{Deserialize, Serialize};

use crate::define_request_class;
use crate::spacetrack::serde_helpers::{
    deserialize_optional_f64, deserialize_optional_i32, deserialize_optional_u32,
    deserialize_optional_u64,
};

/// SATCAT record containing satellite catalog information.
///
/// This struct represents a single satellite catalog entry as returned
/// by the SpaceTrack API.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "UPPERCASE")]
pub struct SATCATRecord {
    /// International designator (e.g., "98067A")
    #[serde(default)]
    pub intldes: Option<String>,
    /// NORAD catalog ID
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub norad_cat_id: Option<u32>,
    /// Object type (PAYLOAD, ROCKET BODY, DEBRIS, etc.)
    #[serde(default)]
    pub object_type: Option<String>,
    /// Satellite name
    #[serde(default)]
    pub satname: Option<String>,
    /// Country/organization code
    #[serde(default)]
    pub country: Option<String>,
    /// Launch date
    #[serde(default)]
    pub launch: Option<String>,
    /// Launch site code
    #[serde(default)]
    pub site: Option<String>,
    /// Decay date (if decayed)
    #[serde(default)]
    pub decay: Option<String>,
    /// Orbital period (minutes)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub period: Option<f64>,
    /// Inclination (degrees)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub inclination: Option<f64>,
    /// Apogee altitude (km)
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub apogee: Option<u32>,
    /// Perigee altitude (km)
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub perigee: Option<u32>,
    /// Comment
    #[serde(default)]
    pub comment: Option<String>,
    /// Comment code
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub commentcode: Option<u32>,
    /// RCS value (m^2)
    #[serde(default, deserialize_with = "deserialize_optional_i32")]
    pub rcsvalue: Option<i32>,
    /// RCS size category (SMALL, MEDIUM, LARGE)
    #[serde(default)]
    pub rcs_size: Option<String>,
    /// File number
    #[serde(default, deserialize_with = "deserialize_optional_u64")]
    pub file: Option<u64>,
    /// Launch year
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub launch_year: Option<u32>,
    /// Launch number within the year
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub launch_num: Option<u32>,
    /// Launch piece
    #[serde(default)]
    pub launch_piece: Option<String>,
    /// Current status (Y = active tracking, N = not tracked)
    #[serde(default)]
    pub current: Option<String>,
    /// Object name (same as satname in most cases)
    #[serde(default)]
    pub object_name: Option<String>,
    /// Object ID
    #[serde(default)]
    pub object_id: Option<String>,
    /// Object number
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub object_number: Option<u32>,
}

define_request_class! {
    /// SATCAT request builder for querying satellite catalog entries.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{SATCATRequest, SpaceTrackClient};
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&SATCATRequest::new()
    ///     .norad_cat_id(25544)
    /// ).await?;
    /// ```
    name: SATCATRequest,
    class_name: "satcat",
    controller: "basicspacedata",
    record: SATCATRecord,
    predicates: {
        /// International designator
        intldes: String,
        /// NORAD catalog ID
        norad_cat_id: u32,
        /// Object type (PAYLOAD, ROCKET BODY, DEBRIS)
        object_type: String,
        /// Satellite name
        satname: String,
        /// Country/organization code
        country: String,
        /// Launch date
        launch: String,
        /// Launch site
        site: String,
        /// Decay date
        decay: String,
        /// Orbital period (minutes)
        period: f64,
        /// Inclination (degrees)
        inclination: f64,
        /// Apogee altitude (km)
        apogee: u32,
        /// Perigee altitude (km)
        perigee: u32,
        /// RCS size category
        rcs_size: String,
        /// Current tracking status
        current: String,
        /// Object name
        object_name: String,
        /// Object ID
        object_id: String,
        /// Launch year
        launch_year: u32,
        /// Launch number
        launch_num: u32,
        /// Launch piece
        launch_piece: String,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::spacetrack::operators::like;
    use crate::spacetrack::request_classes::RequestClass;

    #[test]
    fn test_satcat_request_new() {
        let req = SATCATRequest::new();
        assert!(req.norad_cat_id.is_none());
        assert!(req.satname.is_none());
    }

    #[test]
    fn test_satcat_request_builder() {
        let req = SATCATRequest::new()
            .country("US")
            .object_type("PAYLOAD")
            .current("Y")
            .limit(100);

        assert!(req.country.is_some());
        assert!(req.object_type.is_some());
        assert!(req.current.is_some());
        assert_eq!(req.limit, Some(100));
    }

    #[test]
    fn test_satcat_request_class_name() {
        assert_eq!(SATCATRequest::class_name(), "satcat");
    }

    #[test]
    fn test_satcat_request_controller() {
        assert_eq!(SATCATRequest::controller(), "basicspacedata");
    }

    #[test]
    fn test_satcat_request_with_like() {
        let req = SATCATRequest::new().satname(like("%ISS%"));

        let predicates = req.predicates();
        assert!(!predicates.is_empty());
    }

    #[test]
    fn test_satcat_record_deserialize() {
        let json = r#"{
            "NORAD_CAT_ID": 25544,
            "SATNAME": "ISS (ZARYA)",
            "COUNTRY": "ISS",
            "OBJECT_TYPE": "PAYLOAD",
            "LAUNCH": "1998-11-20",
            "PERIOD": 92.9,
            "INCLINATION": 51.6,
            "APOGEE": 422,
            "PERIGEE": 418,
            "RCS_SIZE": "LARGE",
            "CURRENT": "Y"
        }"#;

        let record: SATCATRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.norad_cat_id, Some(25544));
        assert_eq!(record.satname, Some("ISS (ZARYA)".to_string()));
        assert_eq!(record.country, Some("ISS".to_string()));
        assert_eq!(record.object_type, Some("PAYLOAD".to_string()));
        assert_eq!(record.current, Some("Y".to_string()));
    }

    #[test]
    fn test_satcat_record_default() {
        let record = SATCATRecord::default();
        assert!(record.norad_cat_id.is_none());
        assert!(record.satname.is_none());
    }
}
