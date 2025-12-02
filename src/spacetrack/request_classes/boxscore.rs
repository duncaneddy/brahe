/*!
 * Boxscore request class
 *
 * The Boxscore class provides summary statistics about objects in the catalog.
 */

use serde::{Deserialize, Serialize};

use crate::define_request_class;
use crate::spacetrack::serde_helpers::deserialize_optional_u32;

/// Boxscore record containing catalog summary statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct BoxscoreRecord {
    /// Country code
    #[serde(default)]
    pub country: Option<String>,
    /// Spadoc country code
    #[serde(default)]
    pub spadoc_cd: Option<String>,
    /// Orbital total count
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub orbital_tba: Option<u32>,
    /// Orbital payload count
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub orbital_payload_count: Option<u32>,
    /// Orbital rocket body count
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub orbital_rocket_body_count: Option<u32>,
    /// Orbital debris count
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub orbital_debris_count: Option<u32>,
    /// Orbital total count
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub orbital_total_count: Option<u32>,
    /// Decayed payload count
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub decayed_payload_count: Option<u32>,
    /// Decayed rocket body count
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub decayed_rocket_body_count: Option<u32>,
    /// Decayed debris count
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub decayed_debris_count: Option<u32>,
    /// Decayed total count
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub decayed_total_count: Option<u32>,
    /// Country total count
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub country_total: Option<u32>,
}

define_request_class! {
    /// Boxscore request builder for querying catalog statistics.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{BoxscoreRequest, SpaceTrackClient};
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&BoxscoreRequest::new()
    ///     .country("US")
    /// ).await?;
    /// ```
    name: BoxscoreRequest,
    class_name: "boxscore",
    controller: "basicspacedata",
    record: BoxscoreRecord,
    predicates: {
        /// Country code
        country: String,
        /// Spadoc country code
        spadoc_cd: String,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::spacetrack::request_classes::RequestClass;

    #[test]
    fn test_boxscore_request_new() {
        let req = BoxscoreRequest::new();
        assert!(req.country.is_none());
    }

    #[test]
    fn test_boxscore_request_class_name() {
        assert_eq!(BoxscoreRequest::class_name(), "boxscore");
    }

    #[test]
    fn test_boxscore_request_controller() {
        assert_eq!(BoxscoreRequest::controller(), "basicspacedata");
    }

    #[test]
    fn test_boxscore_record_deserialize() {
        let json = r#"{
            "COUNTRY": "US",
            "ORBITAL_PAYLOAD_COUNT": 1500,
            "ORBITAL_ROCKET_BODY_COUNT": 500,
            "ORBITAL_DEBRIS_COUNT": 3000,
            "ORBITAL_TOTAL_COUNT": 5000
        }"#;

        let record: BoxscoreRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.country, Some("US".to_string()));
        assert_eq!(record.orbital_payload_count, Some(1500));
        assert_eq!(record.orbital_total_count, Some(5000));
    }
}
