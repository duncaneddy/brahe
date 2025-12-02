/*!
 * Announcement request class
 *
 * The Announcement class provides access to Space-Track announcements.
 */

use serde::{Deserialize, Serialize};

use crate::define_request_class;

/// Announcement record containing Space-Track announcement information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct AnnouncementRecord {
    /// Announcement ID
    #[serde(default)]
    pub announcement_id: Option<u64>,
    /// Announcement type
    #[serde(default)]
    pub announcement_type: Option<String>,
    /// Announcement text
    #[serde(default)]
    pub announcement_text: Option<String>,
    /// Announcement start time
    #[serde(default)]
    pub announcement_start: Option<String>,
    /// Announcement end time
    #[serde(default)]
    pub announcement_end: Option<String>,
}

define_request_class! {
    /// Announcement request builder for querying Space-Track announcements.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{AnnouncementRequest, SpaceTrackClient};
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&AnnouncementRequest::new()
    ///     .limit(10)
    /// ).await?;
    /// ```
    name: AnnouncementRequest,
    class_name: "announcement",
    controller: "basicspacedata",
    record: AnnouncementRecord,
    predicates: {
        /// Announcement ID
        announcement_id: u64,
        /// Announcement type
        announcement_type: String,
        /// Announcement start time
        announcement_start: String,
        /// Announcement end time
        announcement_end: String,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::spacetrack::request_classes::RequestClass;

    #[test]
    fn test_announcement_request_new() {
        let req = AnnouncementRequest::new();
        assert!(req.announcement_id.is_none());
    }

    #[test]
    fn test_announcement_request_class_name() {
        assert_eq!(AnnouncementRequest::class_name(), "announcement");
    }

    #[test]
    fn test_announcement_request_controller() {
        assert_eq!(AnnouncementRequest::controller(), "basicspacedata");
    }

    #[test]
    fn test_announcement_record_deserialize() {
        let json = r#"{
            "ANNOUNCEMENT_ID": 123,
            "ANNOUNCEMENT_TYPE": "General",
            "ANNOUNCEMENT_TEXT": "System maintenance scheduled",
            "ANNOUNCEMENT_START": "2024-01-01 00:00:00",
            "ANNOUNCEMENT_END": "2024-01-02 00:00:00"
        }"#;

        let record: AnnouncementRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.announcement_id, Some(123));
        assert_eq!(record.announcement_type, Some("General".to_string()));
    }
}
