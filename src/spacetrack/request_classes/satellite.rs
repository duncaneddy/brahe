/*!
 * Satellite request class
 *
 * The Satellite class provides detailed satellite information.
 * This is part of the expandedspacedata controller.
 */

use serde::{Deserialize, Serialize};

use crate::define_request_class;

/// Satellite record containing detailed satellite information.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct SatelliteRecord {
    /// NORAD catalog ID
    #[serde(default)]
    pub norad_cat_id: Option<u32>,
    /// Satellite ID
    #[serde(default)]
    pub satid: Option<u32>,
    /// Satellite name
    #[serde(default)]
    pub satname: Option<String>,
    /// International designator
    #[serde(default)]
    pub intldes: Option<String>,
    /// Satellite status
    #[serde(default)]
    pub status: Option<String>,
    /// Active flag
    #[serde(default)]
    pub active: Option<String>,
    /// Country
    #[serde(default)]
    pub country: Option<String>,
    /// Launch date
    #[serde(default)]
    pub launch_date: Option<String>,
    /// Launch site
    #[serde(default)]
    pub launch_site: Option<String>,
    /// Launch vehicle
    #[serde(default)]
    pub launch_vehicle: Option<String>,
    /// Launch mass (kg)
    #[serde(default)]
    pub launch_mass: Option<f64>,
    /// Dry mass (kg)
    #[serde(default)]
    pub dry_mass: Option<f64>,
    /// Power (watts)
    #[serde(default)]
    pub power: Option<f64>,
    /// Expected lifetime (years)
    #[serde(default)]
    pub expected_lifetime: Option<f64>,
    /// Mission type
    #[serde(default)]
    pub mission_type: Option<String>,
    /// Mission description
    #[serde(default)]
    pub mission: Option<String>,
    /// Contractor
    #[serde(default)]
    pub contractor: Option<String>,
    /// Operator organization name
    #[serde(default)]
    pub operator: Option<String>,
    /// Users
    #[serde(default)]
    pub users: Option<String>,
}

define_request_class! {
    /// Satellite request builder for querying detailed satellite information.
    ///
    /// This requires expanded spacedata access.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{SatelliteRequest, SpaceTrackClient};
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&SatelliteRequest::new()
    ///     .norad_cat_id(25544)
    /// ).await?;
    /// ```
    name: SatelliteRequest,
    class_name: "satellite",
    controller: "expandedspacedata",
    record: SatelliteRecord,
    predicates: {
        /// NORAD catalog ID
        norad_cat_id: u32,
        /// Satellite ID
        satid: u32,
        /// Satellite name
        satname: String,
        /// International designator
        intldes: String,
        /// Status
        status: String,
        /// Active flag
        active: String,
        /// Country
        country: String,
        /// Launch date
        launch_date: String,
        /// Launch site
        launch_site: String,
        /// Launch vehicle
        launch_vehicle: String,
        /// Mission type
        mission_type: String,
        /// Operator
        operator: String,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::spacetrack::request_classes::RequestClass;

    #[test]
    fn test_satellite_request_class_name() {
        assert_eq!(SatelliteRequest::class_name(), "satellite");
    }

    #[test]
    fn test_satellite_request_controller() {
        assert_eq!(SatelliteRequest::controller(), "expandedspacedata");
    }

    #[test]
    fn test_satellite_record_deserialize() {
        let json = r#"{
            "NORAD_CAT_ID": 25544,
            "SATNAME": "ISS (ZARYA)",
            "INTLDES": "1998-067A",
            "COUNTRY": "ISS",
            "LAUNCH_DATE": "1998-11-20",
            "LAUNCH_MASS": 419725.0,
            "MISSION_TYPE": "Human spaceflight"
        }"#;

        let record: SatelliteRecord = serde_json::from_str(json).unwrap();
        assert_eq!(record.norad_cat_id, Some(25544));
        assert_eq!(record.satname, Some("ISS (ZARYA)".to_string()));
        assert_eq!(record.launch_mass, Some(419725.0));
    }
}
