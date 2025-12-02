/*!
 * CDM (Conjunction Data Message) request class - Expanded Space Data
 *
 * The CDM class provides access to full Conjunction Data Messages.
 * This is part of the expandedspacedata controller and has more detail
 * than cdm_public.
 */

use serde::{Deserialize, Serialize};

use crate::define_request_class;

/// CDM record containing full conjunction data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct CDMRecord {
    /// CDM ID
    #[serde(default)]
    pub cdm_id: Option<u64>,
    /// Constellation
    #[serde(default)]
    pub constellation: Option<String>,
    /// CDM message ID
    #[serde(default)]
    pub message_id: Option<String>,
    /// CDM message for
    #[serde(default)]
    pub message_for: Option<String>,
    /// Creation date
    #[serde(default)]
    pub creation_date: Option<String>,
    /// Emergency reportable flag
    #[serde(default)]
    pub emergency_reportable: Option<String>,
    /// Time of closest approach (TCA)
    #[serde(default)]
    pub tca: Option<String>,
    /// Miss distance (km)
    #[serde(default)]
    pub miss_distance: Option<f64>,
    /// Relative speed (km/s)
    #[serde(default)]
    pub relative_speed: Option<f64>,
    /// Relative position R (km)
    #[serde(default)]
    pub relative_position_r: Option<f64>,
    /// Relative position T (km)
    #[serde(default)]
    pub relative_position_t: Option<f64>,
    /// Relative position N (km)
    #[serde(default)]
    pub relative_position_n: Option<f64>,
    /// Relative velocity R (km/s)
    #[serde(default)]
    pub relative_velocity_r: Option<f64>,
    /// Relative velocity T (km/s)
    #[serde(default)]
    pub relative_velocity_t: Option<f64>,
    /// Relative velocity N (km/s)
    #[serde(default)]
    pub relative_velocity_n: Option<f64>,
    /// Collision probability method
    #[serde(default)]
    pub collision_probability_method: Option<String>,
    /// Collision probability
    #[serde(default)]
    pub collision_probability: Option<f64>,
    /// SAT1 ID
    #[serde(default)]
    pub sat1_id: Option<u32>,
    /// SAT1 name
    #[serde(default)]
    pub sat1_name: Option<String>,
    /// SAT1 NORAD catalog ID
    #[serde(default)]
    pub sat1_norad_cat_id: Option<u32>,
    /// SAT1 object designator
    #[serde(default)]
    pub sat1_object_designator: Option<String>,
    /// SAT1 object type
    #[serde(default)]
    pub sat1_object_type: Option<String>,
    /// SAT1 operator organization
    #[serde(default)]
    pub sat1_operator_organization: Option<String>,
    /// SAT2 ID
    #[serde(default)]
    pub sat2_id: Option<u32>,
    /// SAT2 name
    #[serde(default)]
    pub sat2_name: Option<String>,
    /// SAT2 NORAD catalog ID
    #[serde(default)]
    pub sat2_norad_cat_id: Option<u32>,
    /// SAT2 object designator
    #[serde(default)]
    pub sat2_object_designator: Option<String>,
    /// SAT2 object type
    #[serde(default)]
    pub sat2_object_type: Option<String>,
    /// SAT2 operator organization
    #[serde(default)]
    pub sat2_operator_organization: Option<String>,
}

define_request_class! {
    /// CDM request builder for querying full Conjunction Data Messages.
    ///
    /// This requires expanded spacedata access. For public CDM data, use
    /// `CDMPublicRequest` instead.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{CDMRequest, SpaceTrackClient};
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&CDMRequest::new()
    ///     .sat1_norad_cat_id(25544)
    ///     .limit(10)
    /// ).await?;
    /// ```
    name: CDMRequest,
    class_name: "cdm",
    controller: "expandedspacedata",
    record: CDMRecord,
    predicates: {
        /// CDM ID
        cdm_id: u64,
        /// Constellation
        constellation: String,
        /// Creation date
        creation_date: String,
        /// Time of closest approach
        tca: String,
        /// Miss distance (km)
        miss_distance: f64,
        /// Relative speed (km/s)
        relative_speed: f64,
        /// Collision probability
        collision_probability: f64,
        /// Emergency reportable flag
        emergency_reportable: String,
        /// SAT1 NORAD catalog ID
        sat1_norad_cat_id: u32,
        /// SAT1 name
        sat1_name: String,
        /// SAT1 object type
        sat1_object_type: String,
        /// SAT2 NORAD catalog ID
        sat2_norad_cat_id: u32,
        /// SAT2 name
        sat2_name: String,
        /// SAT2 object type
        sat2_object_type: String,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::spacetrack::request_classes::RequestClass;

    #[test]
    fn test_cdm_request_class_name() {
        assert_eq!(CDMRequest::class_name(), "cdm");
    }

    #[test]
    fn test_cdm_request_controller() {
        assert_eq!(CDMRequest::controller(), "expandedspacedata");
    }

    #[test]
    fn test_cdm_record_deserialize() {
        let json = r#"{
            "CDM_ID": 123456,
            "TCA": "2024-01-15 12:30:00",
            "MISS_DISTANCE": 0.5,
            "COLLISION_PROBABILITY": 0.0001,
            "SAT1_NORAD_CAT_ID": 25544,
            "SAT1_NAME": "ISS (ZARYA)"
        }"#;

        let record: CDMRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.cdm_id, Some(123456));
        assert_eq!(record.sat1_name, Some("ISS (ZARYA)".to_string()));
    }
}
