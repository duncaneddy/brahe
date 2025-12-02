/*!
 * Launch Site request class
 *
 * The Launch Site class provides information about launch facilities.
 */

use serde::{Deserialize, Serialize};

use crate::define_request_class;

/// Launch site record containing launch facility information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct LaunchSiteRecord {
    /// Site code
    #[serde(default)]
    pub site_code: Option<String>,
    /// Launch site name
    #[serde(default)]
    pub launch_site: Option<String>,
}

define_request_class! {
    /// Launch Site request builder for querying launch facility information.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{LaunchSiteRequest, SpaceTrackClient};
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&LaunchSiteRequest::new()).await?;
    /// ```
    name: LaunchSiteRequest,
    class_name: "launch_site",
    controller: "basicspacedata",
    record: LaunchSiteRecord,
    predicates: {
        /// Site code
        site_code: String,
        /// Launch site name
        launch_site: String,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::spacetrack::request_classes::RequestClass;

    #[test]
    fn test_launch_site_request_new() {
        let req = LaunchSiteRequest::new();
        assert!(req.site_code.is_none());
    }

    #[test]
    fn test_launch_site_request_class_name() {
        assert_eq!(LaunchSiteRequest::class_name(), "launch_site");
    }

    #[test]
    fn test_launch_site_request_controller() {
        assert_eq!(LaunchSiteRequest::controller(), "basicspacedata");
    }

    #[test]
    fn test_launch_site_record_deserialize() {
        let json = r#"{
            "SITE_CODE": "AFETR",
            "LAUNCH_SITE": "Cape Canaveral, Florida"
        }"#;

        let record: LaunchSiteRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.site_code, Some("AFETR".to_string()));
        assert_eq!(
            record.launch_site,
            Some("Cape Canaveral, Florida".to_string())
        );
    }
}
