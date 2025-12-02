/*!
 * Maneuver request classes
 *
 * The Maneuver classes provide access to maneuver data.
 * This is part of the expandedspacedata controller.
 */

use serde::{Deserialize, Serialize};

use crate::define_request_class;

/// Maneuver record containing satellite maneuver information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct ManeuverRecord {
    /// Maneuver ID
    #[serde(default)]
    pub maneuver_id: Option<u64>,
    /// NORAD catalog ID
    #[serde(default)]
    pub norad_cat_id: Option<u32>,
    /// Satellite name
    #[serde(default)]
    pub satname: Option<String>,
    /// Source
    #[serde(default)]
    pub source: Option<String>,
    /// Maneuver type
    #[serde(default)]
    pub maneuverable: Option<String>,
    /// Maneuver start time
    #[serde(default)]
    pub start_time: Option<String>,
    /// Maneuver stop time
    #[serde(default)]
    pub stop_time: Option<String>,
    /// Data status
    #[serde(default)]
    pub data_status: Option<String>,
}

define_request_class! {
    /// Maneuver request builder for querying satellite maneuver data.
    ///
    /// This requires expanded spacedata access.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{ManeuverRequest, SpaceTrackClient};
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&ManeuverRequest::new()
    ///     .norad_cat_id(25544)
    ///     .limit(10)
    /// ).await?;
    /// ```
    name: ManeuverRequest,
    class_name: "maneuver",
    controller: "expandedspacedata",
    record: ManeuverRecord,
    predicates: {
        /// Maneuver ID
        maneuver_id: u64,
        /// NORAD catalog ID
        norad_cat_id: u32,
        /// Satellite name
        satname: String,
        /// Source
        source: String,
        /// Maneuverable flag
        maneuverable: String,
        /// Start time
        start_time: String,
        /// Stop time
        stop_time: String,
        /// Data status
        data_status: String,
    }
}

/// Maneuver History record containing historical maneuver data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct ManeuverHistoryRecord {
    /// Maneuver History ID
    #[serde(default)]
    pub maneuver_history_id: Option<u64>,
    /// NORAD catalog ID
    #[serde(default)]
    pub norad_cat_id: Option<u32>,
    /// Satellite name
    #[serde(default)]
    pub satname: Option<String>,
    /// Object type
    #[serde(default)]
    pub object_type: Option<String>,
    /// Source
    #[serde(default)]
    pub source: Option<String>,
    /// Maneuver start time
    #[serde(default)]
    pub start_time: Option<String>,
    /// Maneuver stop time
    #[serde(default)]
    pub stop_time: Option<String>,
    /// Delta V (km/s)
    #[serde(default)]
    pub delta_v: Option<f64>,
    /// Thrust uncertainty
    #[serde(default)]
    pub thrust_uncertainty: Option<f64>,
}

define_request_class! {
    /// Maneuver History request builder for querying historical maneuver data.
    ///
    /// This requires expanded spacedata access.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{ManeuverHistoryRequest, SpaceTrackClient};
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&ManeuverHistoryRequest::new()
    ///     .norad_cat_id(25544)
    ///     .limit(100)
    /// ).await?;
    /// ```
    name: ManeuverHistoryRequest,
    class_name: "maneuver_history",
    controller: "expandedspacedata",
    record: ManeuverHistoryRecord,
    predicates: {
        /// Maneuver History ID
        maneuver_history_id: u64,
        /// NORAD catalog ID
        norad_cat_id: u32,
        /// Satellite name
        satname: String,
        /// Object type
        object_type: String,
        /// Source
        source: String,
        /// Start time
        start_time: String,
        /// Stop time
        stop_time: String,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::spacetrack::request_classes::RequestClass;

    #[test]
    fn test_maneuver_request_class_name() {
        assert_eq!(ManeuverRequest::class_name(), "maneuver");
    }

    #[test]
    fn test_maneuver_request_controller() {
        assert_eq!(ManeuverRequest::controller(), "expandedspacedata");
    }

    #[test]
    fn test_maneuver_history_request_class_name() {
        assert_eq!(ManeuverHistoryRequest::class_name(), "maneuver_history");
    }

    #[test]
    fn test_maneuver_history_request_controller() {
        assert_eq!(ManeuverHistoryRequest::controller(), "expandedspacedata");
    }

    #[test]
    fn test_maneuver_record_deserialize() {
        let json = r#"{
            "MANEUVER_ID": 12345,
            "NORAD_CAT_ID": 25544,
            "SATNAME": "ISS (ZARYA)",
            "START_TIME": "2024-01-15 12:00:00",
            "STOP_TIME": "2024-01-15 12:30:00"
        }"#;

        let record: ManeuverRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.maneuver_id, Some(12345));
        assert_eq!(record.satname, Some("ISS (ZARYA)".to_string()));
    }

    #[test]
    fn test_maneuver_history_record_deserialize() {
        let json = r#"{
            "MANEUVER_HISTORY_ID": 12345,
            "NORAD_CAT_ID": 25544,
            "SATNAME": "ISS (ZARYA)",
            "DELTA_V": 0.005
        }"#;

        let record: ManeuverHistoryRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.maneuver_history_id, Some(12345));
        assert_eq!(record.delta_v, Some(0.005));
    }
}
