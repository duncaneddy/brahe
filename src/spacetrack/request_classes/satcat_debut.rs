/*!
 * SATCAT Debut request class
 *
 * The SATCAT Debut class provides information about new objects
 * added to the satellite catalog.
 */

use serde::{Deserialize, Serialize};

use crate::define_request_class;

/// SATCAT Debut record containing information about new catalog entries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct SATCATDebutRecord {
    /// International designator
    #[serde(default)]
    pub intldes: Option<String>,
    /// NORAD catalog ID
    #[serde(default)]
    pub norad_cat_id: Option<u32>,
    /// Object type
    #[serde(default)]
    pub object_type: Option<String>,
    /// Satellite name
    #[serde(default)]
    pub satname: Option<String>,
    /// Debut date (when added to catalog)
    #[serde(default)]
    pub debut: Option<String>,
    /// Country code
    #[serde(default)]
    pub country: Option<String>,
    /// Launch date
    #[serde(default)]
    pub launch: Option<String>,
    /// Launch site
    #[serde(default)]
    pub site: Option<String>,
    /// Decay date
    #[serde(default)]
    pub decay: Option<String>,
    /// Period (minutes)
    #[serde(default)]
    pub period: Option<f64>,
    /// Inclination (degrees)
    #[serde(default)]
    pub inclination: Option<f64>,
    /// Apogee (km)
    #[serde(default)]
    pub apogee: Option<u32>,
    /// Perigee (km)
    #[serde(default)]
    pub perigee: Option<u32>,
    /// RCS size
    #[serde(default)]
    pub rcs_size: Option<String>,
    /// Current status
    #[serde(default)]
    pub current: Option<String>,
}

define_request_class! {
    /// SATCAT Debut request builder for querying new catalog entries.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{SATCATDebutRequest, SpaceTrackClient};
    /// use brahe::spacetrack::operators::greater_than;
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&SATCATDebutRequest::new()
    ///     .debut(greater_than("2024-01-01"))
    ///     .limit(100)
    /// ).await?;
    /// ```
    name: SATCATDebutRequest,
    class_name: "satcat_debut",
    controller: "basicspacedata",
    record: SATCATDebutRecord,
    predicates: {
        /// International designator
        intldes: String,
        /// NORAD catalog ID
        norad_cat_id: u32,
        /// Object type
        object_type: String,
        /// Satellite name
        satname: String,
        /// Debut date
        debut: String,
        /// Country code
        country: String,
        /// Launch date
        launch: String,
        /// Launch site
        site: String,
        /// Decay date
        decay: String,
        /// Period (minutes)
        period: f64,
        /// Inclination (degrees)
        inclination: f64,
        /// Apogee (km)
        apogee: u32,
        /// Perigee (km)
        perigee: u32,
        /// RCS size
        rcs_size: String,
        /// Current status
        current: String,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::spacetrack::operators::greater_than;
    use crate::spacetrack::request_classes::RequestClass;

    #[test]
    fn test_satcat_debut_request_new() {
        let req = SATCATDebutRequest::new();
        assert!(req.norad_cat_id.is_none());
    }

    #[test]
    fn test_satcat_debut_request_class_name() {
        assert_eq!(SATCATDebutRequest::class_name(), "satcat_debut");
    }

    #[test]
    fn test_satcat_debut_request_controller() {
        assert_eq!(SATCATDebutRequest::controller(), "basicspacedata");
    }

    #[test]
    fn test_satcat_debut_request_with_predicates() {
        let req = SATCATDebutRequest::new()
            .debut(greater_than("2024-01-01"))
            .object_type("PAYLOAD")
            .limit(100);

        let predicates = req.predicates();
        assert!(!predicates.is_empty());
    }

    #[test]
    fn test_satcat_debut_record_deserialize() {
        let json = r#"{
            "NORAD_CAT_ID": 99999,
            "SATNAME": "NEW SAT",
            "OBJECT_TYPE": "PAYLOAD",
            "DEBUT": "2024-01-15 00:00:00",
            "COUNTRY": "US",
            "LAUNCH": "2024-01-10"
        }"#;

        let record: SATCATDebutRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.norad_cat_id, Some(99999));
        assert_eq!(record.satname, Some("NEW SAT".to_string()));
        assert_eq!(record.debut, Some("2024-01-15 00:00:00".to_string()));
    }
}
