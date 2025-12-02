/*!
 * SATCAT Change request class
 *
 * The SATCAT Change class provides information about changes to
 * satellite catalog entries.
 */

use serde::{Deserialize, Serialize};

use crate::define_request_class;

/// SATCAT Change record containing information about catalog changes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct SATCATChangeRecord {
    /// NORAD catalog ID
    #[serde(default)]
    pub norad_cat_id: Option<u32>,
    /// Object number
    #[serde(default)]
    pub object_number: Option<u32>,
    /// Current name
    #[serde(default)]
    pub current_name: Option<String>,
    /// Previous name
    #[serde(default)]
    pub previous_name: Option<String>,
    /// Current international designator
    #[serde(default)]
    pub current_intldes: Option<String>,
    /// Previous international designator
    #[serde(default)]
    pub previous_intldes: Option<String>,
    /// Current country
    #[serde(default)]
    pub current_country: Option<String>,
    /// Previous country
    #[serde(default)]
    pub previous_country: Option<String>,
    /// Current launch date
    #[serde(default)]
    pub current_launch: Option<String>,
    /// Previous launch date
    #[serde(default)]
    pub previous_launch: Option<String>,
    /// Current decay date
    #[serde(default)]
    pub current_decay: Option<String>,
    /// Previous decay date
    #[serde(default)]
    pub previous_decay: Option<String>,
    /// Change made timestamp
    #[serde(default)]
    pub change_made: Option<String>,
}

define_request_class! {
    /// SATCAT Change request builder for querying catalog changes.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{SATCATChangeRequest, SpaceTrackClient};
    /// use brahe::spacetrack::operators::greater_than;
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&SATCATChangeRequest::new()
    ///     .change_made(greater_than("2024-01-01"))
    ///     .limit(100)
    /// ).await?;
    /// ```
    name: SATCATChangeRequest,
    class_name: "satcat_change",
    controller: "basicspacedata",
    record: SATCATChangeRecord,
    predicates: {
        /// NORAD catalog ID
        norad_cat_id: u32,
        /// Object number
        object_number: u32,
        /// Current name
        current_name: String,
        /// Previous name
        previous_name: String,
        /// Current international designator
        current_intldes: String,
        /// Previous international designator
        previous_intldes: String,
        /// Current country
        current_country: String,
        /// Previous country
        previous_country: String,
        /// Current launch date
        current_launch: String,
        /// Previous launch date
        previous_launch: String,
        /// Current decay date
        current_decay: String,
        /// Previous decay date
        previous_decay: String,
        /// Change made timestamp
        change_made: String,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::spacetrack::operators::greater_than;
    use crate::spacetrack::request_classes::RequestClass;

    #[test]
    fn test_satcat_change_request_new() {
        let req = SATCATChangeRequest::new();
        assert!(req.norad_cat_id.is_none());
    }

    #[test]
    fn test_satcat_change_request_class_name() {
        assert_eq!(SATCATChangeRequest::class_name(), "satcat_change");
    }

    #[test]
    fn test_satcat_change_request_controller() {
        assert_eq!(SATCATChangeRequest::controller(), "basicspacedata");
    }

    #[test]
    fn test_satcat_change_request_with_predicates() {
        let req = SATCATChangeRequest::new()
            .change_made(greater_than("2024-01-01"))
            .limit(100);

        let predicates = req.predicates();
        assert!(!predicates.is_empty());
    }

    #[test]
    fn test_satcat_change_record_deserialize() {
        let json = r#"{
            "NORAD_CAT_ID": 25544,
            "CURRENT_NAME": "ISS (ZARYA)",
            "PREVIOUS_NAME": "ISS",
            "CURRENT_COUNTRY": "ISS",
            "CHANGE_MADE": "2024-01-15 12:00:00"
        }"#;

        let record: SATCATChangeRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.norad_cat_id, Some(25544));
        assert_eq!(record.current_name, Some("ISS (ZARYA)".to_string()));
        assert_eq!(record.previous_name, Some("ISS".to_string()));
    }
}
