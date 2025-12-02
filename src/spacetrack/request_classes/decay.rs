/*!
 * Decay request class
 *
 * The Decay class provides information about predicted and actual
 * object decay (re-entry) events.
 */

use serde::{Deserialize, Serialize};

use crate::define_request_class;

/// Decay record containing re-entry/decay information.
///
/// This struct represents a single decay record as returned by the SpaceTrack API.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct DecayRecord {
    /// NORAD catalog ID
    #[serde(default)]
    pub norad_cat_id: Option<u32>,
    /// Object number
    #[serde(default)]
    pub object_number: Option<u32>,
    /// Object name
    #[serde(default)]
    pub object_name: Option<String>,
    /// International designator
    #[serde(default)]
    pub intldes: Option<String>,
    /// Object ID
    #[serde(default)]
    pub object_id: Option<String>,
    /// RCS category
    #[serde(default)]
    pub rcs: Option<String>,
    /// RCS size
    #[serde(default)]
    pub rcs_size: Option<String>,
    /// Country code
    #[serde(default)]
    pub country: Option<String>,
    /// Message epoch (when the decay message was issued)
    #[serde(default)]
    pub msg_epoch: Option<String>,
    /// Decay epoch (predicted or actual decay time)
    #[serde(default)]
    pub decay_epoch: Option<String>,
    /// Source of the decay data
    #[serde(default)]
    pub source: Option<String>,
    /// Message type
    #[serde(default)]
    pub msg_type: Option<String>,
    /// Precedence
    #[serde(default)]
    pub precedence: Option<u32>,
}

define_request_class! {
    /// Decay request builder for querying object decay/re-entry information.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{DecayRequest, SpaceTrackClient};
    /// use brahe::spacetrack::operators::greater_than;
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&DecayRequest::new()
    ///     .decay_epoch(greater_than("2024-01-01"))
    ///     .limit(10)
    /// ).await?;
    /// ```
    name: DecayRequest,
    class_name: "decay",
    controller: "basicspacedata",
    record: DecayRecord,
    predicates: {
        /// NORAD catalog ID
        norad_cat_id: u32,
        /// Object number
        object_number: u32,
        /// Object name
        object_name: String,
        /// International designator
        intldes: String,
        /// Object ID
        object_id: String,
        /// RCS category
        rcs: String,
        /// RCS size
        rcs_size: String,
        /// Country code
        country: String,
        /// Message epoch
        msg_epoch: String,
        /// Decay epoch
        decay_epoch: String,
        /// Source
        source: String,
        /// Message type
        msg_type: String,
        /// Precedence
        precedence: u32,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::spacetrack::operators::greater_than;
    use crate::spacetrack::request_classes::RequestClass;

    #[test]
    fn test_decay_request_new() {
        let req = DecayRequest::new();
        assert!(req.norad_cat_id.is_none());
    }

    #[test]
    fn test_decay_request_class_name() {
        assert_eq!(DecayRequest::class_name(), "decay");
    }

    #[test]
    fn test_decay_request_controller() {
        assert_eq!(DecayRequest::controller(), "basicspacedata");
    }

    #[test]
    fn test_decay_request_with_predicates() {
        let req = DecayRequest::new()
            .country("US")
            .decay_epoch(greater_than("2024-01-01"))
            .limit(10);

        let predicates = req.predicates();
        assert!(!predicates.is_empty());
    }

    #[test]
    fn test_decay_record_deserialize() {
        let json = r#"{
            "NORAD_CAT_ID": 12345,
            "OBJECT_NAME": "TEST SAT",
            "COUNTRY": "US",
            "DECAY_EPOCH": "2024-01-15 12:30:00",
            "MSG_TYPE": "Decay",
            "SOURCE": "18 SPCS"
        }"#;

        let record: DecayRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.norad_cat_id, Some(12345));
        assert_eq!(record.object_name, Some("TEST SAT".to_string()));
        assert_eq!(record.country, Some("US".to_string()));
    }
}
