/*!
 * TIP (Tracking and Impact Prediction) request class
 *
 * The TIP class provides tracking and impact prediction messages
 * for decaying objects.
 */

use serde::{Deserialize, Serialize};

use crate::define_request_class;

/// TIP record containing tracking and impact prediction information.
///
/// This struct represents a single TIP message as returned by the SpaceTrack API.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct TIPRecord {
    /// NORAD catalog ID
    #[serde(default)]
    pub norad_cat_id: Option<u32>,
    /// Message epoch
    #[serde(default)]
    pub msg_epoch: Option<String>,
    /// Insert epoch
    #[serde(default)]
    pub insert_epoch: Option<String>,
    /// Decay epoch (predicted)
    #[serde(default)]
    pub decay_epoch: Option<String>,
    /// Window size (hours)
    #[serde(default)]
    pub window: Option<f64>,
    /// Revolution number
    #[serde(default)]
    pub rev: Option<u32>,
    /// Propagation direction
    #[serde(default)]
    pub direction: Option<String>,
    /// Latitude at predicted impact
    #[serde(default)]
    pub lat: Option<f64>,
    /// Longitude at predicted impact
    #[serde(default)]
    pub lon: Option<f64>,
    /// Inclination (degrees)
    #[serde(default)]
    pub incl: Option<f64>,
    /// Next report time
    #[serde(default)]
    pub next_report: Option<String>,
    /// TIP ID
    #[serde(default)]
    pub id: Option<String>,
    /// High interest flag
    #[serde(default)]
    pub high_interest: Option<String>,
    /// Object number
    #[serde(default)]
    pub object_number: Option<u32>,
}

define_request_class! {
    /// TIP request builder for querying Tracking and Impact Prediction messages.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{TIPRequest, SpaceTrackClient};
    /// use brahe::spacetrack::operators::greater_than;
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&TIPRequest::new()
    ///     .decay_epoch(greater_than("2024-01-01"))
    ///     .limit(10)
    /// ).await?;
    /// ```
    name: TIPRequest,
    class_name: "tip",
    controller: "basicspacedata",
    record: TIPRecord,
    predicates: {
        /// NORAD catalog ID
        norad_cat_id: u32,
        /// Object number
        object_number: u32,
        /// Message epoch
        msg_epoch: String,
        /// Insert epoch
        insert_epoch: String,
        /// Decay epoch
        decay_epoch: String,
        /// Window size (hours)
        window: f64,
        /// Revolution number
        rev: u32,
        /// Propagation direction
        direction: String,
        /// Latitude at predicted impact
        lat: f64,
        /// Longitude at predicted impact
        lon: f64,
        /// Inclination (degrees)
        incl: f64,
        /// High interest flag
        high_interest: String,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::spacetrack::operators::greater_than;
    use crate::spacetrack::request_classes::RequestClass;

    #[test]
    fn test_tip_request_new() {
        let req = TIPRequest::new();
        assert!(req.norad_cat_id.is_none());
    }

    #[test]
    fn test_tip_request_class_name() {
        assert_eq!(TIPRequest::class_name(), "tip");
    }

    #[test]
    fn test_tip_request_controller() {
        assert_eq!(TIPRequest::controller(), "basicspacedata");
    }

    #[test]
    fn test_tip_request_with_predicates() {
        let req = TIPRequest::new()
            .decay_epoch(greater_than("2024-01-01"))
            .high_interest("Y")
            .limit(10);

        let predicates = req.predicates();
        assert!(!predicates.is_empty());
    }

    #[test]
    fn test_tip_record_deserialize() {
        let json = r#"{
            "NORAD_CAT_ID": 12345,
            "DECAY_EPOCH": "2024-01-15 12:30:00",
            "WINDOW": 6.5,
            "LAT": 45.5,
            "LON": -122.5,
            "INCL": 51.6,
            "HIGH_INTEREST": "Y"
        }"#;

        let record: TIPRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.norad_cat_id, Some(12345));
        assert_eq!(record.window, Some(6.5));
        assert_eq!(record.high_interest, Some("Y".to_string()));
    }
}
