/*!
 * Type definitions for the SpaceTrack API client.
 *
 * Defines the core enums used to construct SpaceTrack queries:
 * request controllers, request classes, sort orders, and output formats.
 */

use std::fmt;

/// SpaceTrack API request controller.
///
/// Controllers determine which API endpoint is used. Most queries use
/// `BasicSpaceData`, which provides access to GP, SATCAT, decay, and other
/// commonly used request classes.
///
/// # Examples
///
/// ```
/// use brahe::spacetrack::RequestController;
///
/// let controller = RequestController::BasicSpaceData;
/// assert_eq!(controller.as_str(), "basicspacedata");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestController {
    /// Basic space data controller - most commonly used endpoint.
    /// Provides access to GP, SATCAT, decay, TIP, and other standard data.
    BasicSpaceData,
    /// Expanded space data controller for additional datasets.
    ExpandedSpaceData,
    /// File share controller for bulk file downloads.
    FileShare,
    /// SP Ephemeris controller for special perturbations ephemeris data.
    SPEphemeris,
    /// Public files controller.
    PublicFiles,
}

impl RequestController {
    /// Returns the URL path segment for this controller.
    pub fn as_str(&self) -> &'static str {
        match self {
            RequestController::BasicSpaceData => "basicspacedata",
            RequestController::ExpandedSpaceData => "expandedspacedata",
            RequestController::FileShare => "fileshare",
            RequestController::SPEphemeris => "spephemeris",
            RequestController::PublicFiles => "publicfiles",
        }
    }
}

impl fmt::Display for RequestController {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// SpaceTrack API request class.
///
/// Each request class corresponds to a specific type of data available from
/// Space-Track.org. The most commonly used classes are `GP` (General
/// Perturbations / OMM data) and `SATCAT` (Satellite Catalog).
///
/// # Examples
///
/// ```
/// use brahe::spacetrack::{RequestClass, RequestController};
///
/// let class = RequestClass::GP;
/// assert_eq!(class.as_str(), "gp");
/// assert_eq!(class.default_controller(), RequestController::BasicSpaceData);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum RequestClass {
    /// General Perturbations (OMM) data - current orbital elements.
    GP,
    /// GP history - historical orbital element sets.
    GPHistory,
    /// Satellite Catalog - object metadata (name, country, type, etc.).
    SATCAT,
    /// SATCAT changes - modifications to catalog entries.
    SATCATChange,
    /// SATCAT debut - newly cataloged objects.
    SATCATDebut,
    /// Decay predictions and actual decay data.
    Decay,
    /// Tracking and Impact Prediction messages.
    TIP,
    /// Public Conjunction Data Messages.
    CDMPublic,
    /// Boxscore summary statistics.
    Boxscore,
    /// Space-Track announcements.
    Announcement,
    /// Launch site information.
    LaunchSite,
}

impl RequestClass {
    /// Returns the URL path segment for this request class.
    pub fn as_str(&self) -> &'static str {
        match self {
            RequestClass::GP => "gp",
            RequestClass::GPHistory => "gp_history",
            RequestClass::SATCAT => "satcat",
            RequestClass::SATCATChange => "satcat_change",
            RequestClass::SATCATDebut => "satcat_debut",
            RequestClass::Decay => "decay",
            RequestClass::TIP => "tip",
            RequestClass::CDMPublic => "cdm_public",
            RequestClass::Boxscore => "boxscore",
            RequestClass::Announcement => "announcement",
            RequestClass::LaunchSite => "launch_site",
        }
    }

    /// Returns the default controller for this request class.
    ///
    /// Most classes default to `BasicSpaceData`. This allows users to
    /// construct queries without needing to know which controller to use.
    pub fn default_controller(&self) -> RequestController {
        RequestController::BasicSpaceData
    }
}

impl fmt::Display for RequestClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Sort order for query results.
///
/// Used with `SpaceTrackQuery::order_by()` to control result ordering.
///
/// # Examples
///
/// ```
/// use brahe::spacetrack::SortOrder;
///
/// assert_eq!(SortOrder::Asc.as_str(), "asc");
/// assert_eq!(SortOrder::Desc.as_str(), "desc");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    /// Ascending order (smallest/earliest first).
    Asc,
    /// Descending order (largest/latest first).
    Desc,
}

impl SortOrder {
    /// Returns the URL path segment for this sort order.
    pub fn as_str(&self) -> &'static str {
        match self {
            SortOrder::Asc => "asc",
            SortOrder::Desc => "desc",
        }
    }
}

impl fmt::Display for SortOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Output format for query results.
///
/// Controls the format of the response data from Space-Track.
/// JSON is the default and most commonly used format.
///
/// # Examples
///
/// ```
/// use brahe::spacetrack::OutputFormat;
///
/// let format = OutputFormat::JSON;
/// assert!(format.is_json());
/// assert_eq!(format.as_str(), "json");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum OutputFormat {
    /// JSON format (default).
    JSON,
    /// XML format.
    XML,
    /// HTML format.
    HTML,
    /// CSV format.
    CSV,
    /// Two-Line Element format.
    TLE,
    /// Three-Line Element format (includes object name).
    ThreeLe,
    /// CCSDS Keyword-Value Notation format.
    KVN,
}

impl OutputFormat {
    /// Returns the URL path segment for this output format.
    pub fn as_str(&self) -> &'static str {
        match self {
            OutputFormat::JSON => "json",
            OutputFormat::XML => "xml",
            OutputFormat::HTML => "html",
            OutputFormat::CSV => "csv",
            OutputFormat::TLE => "tle",
            OutputFormat::ThreeLe => "3le",
            OutputFormat::KVN => "kvn",
        }
    }

    /// Returns true if this format produces JSON output.
    pub fn is_json(&self) -> bool {
        matches!(self, OutputFormat::JSON)
    }
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    // -- RequestController tests --

    #[test]
    fn test_request_controller_as_str() {
        assert_eq!(RequestController::BasicSpaceData.as_str(), "basicspacedata");
        assert_eq!(
            RequestController::ExpandedSpaceData.as_str(),
            "expandedspacedata"
        );
        assert_eq!(RequestController::FileShare.as_str(), "fileshare");
        assert_eq!(RequestController::SPEphemeris.as_str(), "spephemeris");
        assert_eq!(RequestController::PublicFiles.as_str(), "publicfiles");
    }

    #[test]
    fn test_request_controller_display() {
        assert_eq!(
            format!("{}", RequestController::BasicSpaceData),
            "basicspacedata"
        );
        assert_eq!(
            format!("{}", RequestController::ExpandedSpaceData),
            "expandedspacedata"
        );
        assert_eq!(format!("{}", RequestController::FileShare), "fileshare");
        assert_eq!(format!("{}", RequestController::SPEphemeris), "spephemeris");
        assert_eq!(format!("{}", RequestController::PublicFiles), "publicfiles");
    }

    // -- RequestClass tests --

    #[test]
    fn test_request_class_as_str() {
        assert_eq!(RequestClass::GP.as_str(), "gp");
        assert_eq!(RequestClass::GPHistory.as_str(), "gp_history");
        assert_eq!(RequestClass::SATCAT.as_str(), "satcat");
        assert_eq!(RequestClass::SATCATChange.as_str(), "satcat_change");
        assert_eq!(RequestClass::SATCATDebut.as_str(), "satcat_debut");
        assert_eq!(RequestClass::Decay.as_str(), "decay");
        assert_eq!(RequestClass::TIP.as_str(), "tip");
        assert_eq!(RequestClass::CDMPublic.as_str(), "cdm_public");
        assert_eq!(RequestClass::Boxscore.as_str(), "boxscore");
        assert_eq!(RequestClass::Announcement.as_str(), "announcement");
        assert_eq!(RequestClass::LaunchSite.as_str(), "launch_site");
    }

    #[test]
    fn test_request_class_display() {
        assert_eq!(format!("{}", RequestClass::GP), "gp");
        assert_eq!(format!("{}", RequestClass::GPHistory), "gp_history");
        assert_eq!(format!("{}", RequestClass::SATCAT), "satcat");
        assert_eq!(format!("{}", RequestClass::SATCATChange), "satcat_change");
        assert_eq!(format!("{}", RequestClass::SATCATDebut), "satcat_debut");
        assert_eq!(format!("{}", RequestClass::Decay), "decay");
        assert_eq!(format!("{}", RequestClass::TIP), "tip");
        assert_eq!(format!("{}", RequestClass::CDMPublic), "cdm_public");
        assert_eq!(format!("{}", RequestClass::Boxscore), "boxscore");
        assert_eq!(format!("{}", RequestClass::Announcement), "announcement");
        assert_eq!(format!("{}", RequestClass::LaunchSite), "launch_site");
    }

    #[test]
    fn test_request_class_default_controller() {
        assert_eq!(
            RequestClass::GP.default_controller(),
            RequestController::BasicSpaceData
        );
        assert_eq!(
            RequestClass::GPHistory.default_controller(),
            RequestController::BasicSpaceData
        );
        assert_eq!(
            RequestClass::SATCAT.default_controller(),
            RequestController::BasicSpaceData
        );
        assert_eq!(
            RequestClass::SATCATChange.default_controller(),
            RequestController::BasicSpaceData
        );
        assert_eq!(
            RequestClass::SATCATDebut.default_controller(),
            RequestController::BasicSpaceData
        );
        assert_eq!(
            RequestClass::Decay.default_controller(),
            RequestController::BasicSpaceData
        );
        assert_eq!(
            RequestClass::TIP.default_controller(),
            RequestController::BasicSpaceData
        );
        assert_eq!(
            RequestClass::CDMPublic.default_controller(),
            RequestController::BasicSpaceData
        );
        assert_eq!(
            RequestClass::Boxscore.default_controller(),
            RequestController::BasicSpaceData
        );
        assert_eq!(
            RequestClass::Announcement.default_controller(),
            RequestController::BasicSpaceData
        );
        assert_eq!(
            RequestClass::LaunchSite.default_controller(),
            RequestController::BasicSpaceData
        );
    }

    // -- SortOrder tests --

    #[test]
    fn test_sort_order_as_str() {
        assert_eq!(SortOrder::Asc.as_str(), "asc");
        assert_eq!(SortOrder::Desc.as_str(), "desc");
    }

    #[test]
    fn test_sort_order_display() {
        assert_eq!(format!("{}", SortOrder::Asc), "asc");
        assert_eq!(format!("{}", SortOrder::Desc), "desc");
    }

    // -- OutputFormat tests --

    #[test]
    fn test_output_format_as_str() {
        assert_eq!(OutputFormat::JSON.as_str(), "json");
        assert_eq!(OutputFormat::XML.as_str(), "xml");
        assert_eq!(OutputFormat::HTML.as_str(), "html");
        assert_eq!(OutputFormat::CSV.as_str(), "csv");
        assert_eq!(OutputFormat::TLE.as_str(), "tle");
        assert_eq!(OutputFormat::ThreeLe.as_str(), "3le");
        assert_eq!(OutputFormat::KVN.as_str(), "kvn");
    }

    #[test]
    fn test_output_format_display() {
        assert_eq!(format!("{}", OutputFormat::JSON), "json");
        assert_eq!(format!("{}", OutputFormat::XML), "xml");
        assert_eq!(format!("{}", OutputFormat::HTML), "html");
        assert_eq!(format!("{}", OutputFormat::CSV), "csv");
        assert_eq!(format!("{}", OutputFormat::TLE), "tle");
        assert_eq!(format!("{}", OutputFormat::ThreeLe), "3le");
        assert_eq!(format!("{}", OutputFormat::KVN), "kvn");
    }

    #[test]
    fn test_output_format_is_json() {
        assert!(OutputFormat::JSON.is_json());
        assert!(!OutputFormat::XML.is_json());
        assert!(!OutputFormat::HTML.is_json());
        assert!(!OutputFormat::CSV.is_json());
        assert!(!OutputFormat::TLE.is_json());
        assert!(!OutputFormat::ThreeLe.is_json());
        assert!(!OutputFormat::KVN.is_json());
    }

    // -- Shared trait tests --

    #[test]
    fn test_enum_equality() {
        assert_eq!(
            RequestController::BasicSpaceData,
            RequestController::BasicSpaceData
        );
        assert_ne!(
            RequestController::BasicSpaceData,
            RequestController::FileShare
        );
        assert_eq!(RequestClass::GP, RequestClass::GP);
        assert_ne!(RequestClass::GP, RequestClass::SATCAT);
    }

    #[test]
    fn test_enum_clone() {
        let controller = RequestController::BasicSpaceData;
        let cloned = controller;
        assert_eq!(controller, cloned);

        let class = RequestClass::GP;
        let cloned = class;
        assert_eq!(class, cloned);
    }

    #[test]
    fn test_enum_debug() {
        assert_eq!(
            format!("{:?}", RequestController::BasicSpaceData),
            "BasicSpaceData"
        );
        assert_eq!(format!("{:?}", RequestClass::GP), "GP");
        assert_eq!(format!("{:?}", SortOrder::Asc), "Asc");
        assert_eq!(format!("{:?}", OutputFormat::JSON), "JSON");
    }
}
