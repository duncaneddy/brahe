/*!
 * Organization request class
 *
 * The Organization class provides information about satellite operators.
 * This is part of the expandedspacedata controller.
 */

use serde::{Deserialize, Serialize};

use crate::define_request_class;

/// Organization record containing operator information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct OrganizationRecord {
    /// Organization ID
    #[serde(default)]
    pub organization_id: Option<u64>,
    /// Organization name
    #[serde(default)]
    pub organization_name: Option<String>,
    /// Organization type
    #[serde(default)]
    pub organization_type: Option<String>,
    /// Country
    #[serde(default)]
    pub country: Option<String>,
    /// Parent organization ID
    #[serde(default)]
    pub parent_id: Option<u64>,
    /// Primary phone
    #[serde(default)]
    pub primary_phone: Option<String>,
    /// Primary email
    #[serde(default)]
    pub primary_email: Option<String>,
    /// Info link
    #[serde(default)]
    pub info_link: Option<String>,
}

define_request_class! {
    /// Organization request builder for querying satellite operator information.
    ///
    /// This requires expanded spacedata access.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{OrganizationRequest, SpaceTrackClient};
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&OrganizationRequest::new()
    ///     .country("US")
    ///     .limit(100)
    /// ).await?;
    /// ```
    name: OrganizationRequest,
    class_name: "organization",
    controller: "expandedspacedata",
    record: OrganizationRecord,
    predicates: {
        /// Organization ID
        organization_id: u64,
        /// Organization name
        organization_name: String,
        /// Organization type
        organization_type: String,
        /// Country
        country: String,
        /// Parent organization ID
        parent_id: u64,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::spacetrack::request_classes::RequestClass;

    #[test]
    fn test_organization_request_class_name() {
        assert_eq!(OrganizationRequest::class_name(), "organization");
    }

    #[test]
    fn test_organization_request_controller() {
        assert_eq!(OrganizationRequest::controller(), "expandedspacedata");
    }

    #[test]
    fn test_organization_record_deserialize() {
        let json = r#"{
            "ORGANIZATION_ID": 12345,
            "ORGANIZATION_NAME": "NASA",
            "ORGANIZATION_TYPE": "Government",
            "COUNTRY": "US"
        }"#;

        let record: OrganizationRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.organization_id, Some(12345));
        assert_eq!(record.organization_name, Some("NASA".to_string()));
        assert_eq!(record.country, Some("US".to_string()));
    }
}
