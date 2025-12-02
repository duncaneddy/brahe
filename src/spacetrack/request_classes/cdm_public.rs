/*!
 * CDM Public request class
 *
 * The CDM Public class provides access to public Conjunction Data Messages.
 */

use serde::{Deserialize, Serialize};

use crate::define_request_class;

/// CDM Public record containing conjunction data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct CDMPublicRecord {
    /// CDM ID
    #[serde(default)]
    pub cdm_id: Option<u64>,
    /// Creation date
    #[serde(default)]
    pub creation_date: Option<String>,
    /// Time of closest approach (TCA)
    #[serde(default)]
    pub tca: Option<String>,
    /// Miss distance (km)
    #[serde(default)]
    pub miss_distance: Option<f64>,
    /// Relative speed (km/s)
    #[serde(default)]
    pub relative_speed: Option<f64>,
    /// Probability of collision
    #[serde(default)]
    pub collision_probability: Option<f64>,
    /// SAT1 object designator
    #[serde(default)]
    pub sat1_object_designator: Option<String>,
    /// SAT1 NORAD catalog ID
    #[serde(default)]
    pub sat1_norad_cat_id: Option<u32>,
    /// SAT1 object name
    #[serde(default)]
    pub sat1_object_name: Option<String>,
    /// SAT1 object type
    #[serde(default)]
    pub sat1_object_type: Option<String>,
    /// SAT1 RCS size
    #[serde(default)]
    pub sat1_rcs: Option<String>,
    /// SAT1 excl volume frame
    #[serde(default)]
    pub sat1_excl_vol: Option<String>,
    /// SAT2 object designator
    #[serde(default)]
    pub sat2_object_designator: Option<String>,
    /// SAT2 NORAD catalog ID
    #[serde(default)]
    pub sat2_norad_cat_id: Option<u32>,
    /// SAT2 object name
    #[serde(default)]
    pub sat2_object_name: Option<String>,
    /// SAT2 object type
    #[serde(default)]
    pub sat2_object_type: Option<String>,
    /// SAT2 RCS size
    #[serde(default)]
    pub sat2_rcs: Option<String>,
    /// SAT2 excl volume frame
    #[serde(default)]
    pub sat2_excl_vol: Option<String>,
}

define_request_class! {
    /// CDM Public request builder for querying public conjunction data.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{CDMPublicRequest, SpaceTrackClient};
    /// use brahe::spacetrack::operators::greater_than;
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&CDMPublicRequest::new()
    ///     .sat1_norad_cat_id(25544)
    ///     .limit(10)
    /// ).await?;
    /// ```
    name: CDMPublicRequest,
    class_name: "cdm_public",
    controller: "basicspacedata",
    record: CDMPublicRecord,
    predicates: {
        /// CDM ID
        cdm_id: u64,
        /// Creation date
        creation_date: String,
        /// Time of closest approach
        tca: String,
        /// Miss distance (km)
        miss_distance: f64,
        /// Relative speed (km/s)
        relative_speed: f64,
        /// Probability of collision
        collision_probability: f64,
        /// SAT1 NORAD catalog ID
        sat1_norad_cat_id: u32,
        /// SAT1 object name
        sat1_object_name: String,
        /// SAT1 object type
        sat1_object_type: String,
        /// SAT2 NORAD catalog ID
        sat2_norad_cat_id: u32,
        /// SAT2 object name
        sat2_object_name: String,
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
    fn test_cdm_public_request_new() {
        let req = CDMPublicRequest::new();
        assert!(req.cdm_id.is_none());
    }

    #[test]
    fn test_cdm_public_request_class_name() {
        assert_eq!(CDMPublicRequest::class_name(), "cdm_public");
    }

    #[test]
    fn test_cdm_public_request_controller() {
        assert_eq!(CDMPublicRequest::controller(), "basicspacedata");
    }

    #[test]
    fn test_cdm_public_record_deserialize() {
        let json = r#"{
            "CDM_ID": 123456,
            "TCA": "2024-01-15 12:30:00",
            "MISS_DISTANCE": 0.5,
            "RELATIVE_SPEED": 12.5,
            "COLLISION_PROBABILITY": 0.0001,
            "SAT1_NORAD_CAT_ID": 25544,
            "SAT1_OBJECT_NAME": "ISS (ZARYA)",
            "SAT2_NORAD_CAT_ID": 99999,
            "SAT2_OBJECT_NAME": "DEBRIS"
        }"#;

        let record: CDMPublicRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.cdm_id, Some(123456));
        assert_eq!(record.miss_distance, Some(0.5));
        assert_eq!(record.sat1_norad_cat_id, Some(25544));
    }
}
