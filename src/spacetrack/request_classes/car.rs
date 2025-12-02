/*!
 * CAR (Conjunction Assessment Report) request class
 *
 * The CAR class provides access to Conjunction Assessment Reports.
 * This is part of the expandedspacedata controller.
 */

use serde::{Deserialize, Serialize};

use crate::define_request_class;

/// CAR record containing conjunction assessment information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct CARRecord {
    /// Message ID
    #[serde(default)]
    pub message_id: Option<u64>,
    /// Message epoch
    #[serde(default)]
    pub message_epoch: Option<String>,
    /// Collision probability
    #[serde(default)]
    pub collision_probability: Option<f64>,
    /// Miss distance (km)
    #[serde(default)]
    pub miss_distance: Option<f64>,
    /// Time of closest approach
    #[serde(default)]
    pub tca: Option<String>,
    /// SAT1 NORAD catalog ID
    #[serde(default)]
    pub sat1_norad_cat_id: Option<u32>,
    /// SAT1 name
    #[serde(default)]
    pub sat1_name: Option<String>,
    /// SAT2 NORAD catalog ID
    #[serde(default)]
    pub sat2_norad_cat_id: Option<u32>,
    /// SAT2 name
    #[serde(default)]
    pub sat2_name: Option<String>,
    /// Emergency reportable flag
    #[serde(default)]
    pub emergency_reportable: Option<String>,
}

define_request_class! {
    /// CAR request builder for querying Conjunction Assessment Reports.
    ///
    /// This requires expanded spacedata access.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{CARRequest, SpaceTrackClient};
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&CARRequest::new()
    ///     .sat1_norad_cat_id(25544)
    ///     .limit(10)
    /// ).await?;
    /// ```
    name: CARRequest,
    class_name: "car",
    controller: "expandedspacedata",
    record: CARRecord,
    predicates: {
        /// Message ID
        message_id: u64,
        /// Message epoch
        message_epoch: String,
        /// Collision probability
        collision_probability: f64,
        /// Miss distance (km)
        miss_distance: f64,
        /// Time of closest approach
        tca: String,
        /// SAT1 NORAD catalog ID
        sat1_norad_cat_id: u32,
        /// SAT1 name
        sat1_name: String,
        /// SAT2 NORAD catalog ID
        sat2_norad_cat_id: u32,
        /// SAT2 name
        sat2_name: String,
        /// Emergency reportable flag
        emergency_reportable: String,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::spacetrack::request_classes::RequestClass;

    #[test]
    fn test_car_request_class_name() {
        assert_eq!(CARRequest::class_name(), "car");
    }

    #[test]
    fn test_car_request_controller() {
        assert_eq!(CARRequest::controller(), "expandedspacedata");
    }

    #[test]
    fn test_car_record_deserialize() {
        let json = r#"{
            "MESSAGE_ID": 12345,
            "TCA": "2024-01-15 12:30:00",
            "MISS_DISTANCE": 0.5,
            "COLLISION_PROBABILITY": 0.0001,
            "SAT1_NORAD_CAT_ID": 25544
        }"#;

        let record: CARRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.message_id, Some(12345));
        assert_eq!(record.miss_distance, Some(0.5));
    }
}
