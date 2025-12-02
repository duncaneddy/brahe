/*!
 * TLE (Two-Line Element) request classes
 *
 * The TLE classes provide access to historical and latest TLE data.
 * Note: These classes are deprecated in favor of GP, but still functional.
 */

use serde::{Deserialize, Serialize};

use crate::define_request_class;
use crate::propagators::SGPPropagator;
use crate::spacetrack::serde_helpers::{
    deserialize_optional_f64, deserialize_optional_i32, deserialize_optional_u32,
    deserialize_optional_u64,
};
use crate::utils::BraheError;

/// TLE record containing Two-Line Element data.
///
/// This struct represents a single TLE record as returned by the SpaceTrack API.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct TLERecord {
    /// Comment field
    #[serde(default)]
    pub comment: Option<String>,
    /// Originator
    #[serde(default)]
    pub originator: Option<String>,
    /// NORAD catalog ID
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub norad_cat_id: Option<u32>,
    /// Object name
    #[serde(default)]
    pub object_name: Option<String>,
    /// Object type
    #[serde(default)]
    pub object_type: Option<String>,
    /// Classification type
    #[serde(default)]
    pub classification_type: Option<String>,
    /// International designator
    #[serde(default)]
    pub intldes: Option<String>,
    /// Epoch
    #[serde(default)]
    pub epoch: Option<String>,
    /// Epoch year (2-digit)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub epoch_microseconds: Option<f64>,
    /// Mean motion (rev/day)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub mean_motion: Option<f64>,
    /// Eccentricity
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub eccentricity: Option<f64>,
    /// Inclination (degrees)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub inclination: Option<f64>,
    /// Right ascension of ascending node (degrees)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub ra_of_asc_node: Option<f64>,
    /// Argument of perigee (degrees)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub arg_of_pericenter: Option<f64>,
    /// Mean anomaly (degrees)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub mean_anomaly: Option<f64>,
    /// Ephemeris type
    #[serde(default, deserialize_with = "deserialize_optional_i32")]
    pub ephemeris_type: Option<i32>,
    /// Element set number
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub element_set_no: Option<u32>,
    /// Revolution number at epoch
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub rev_at_epoch: Option<u32>,
    /// BSTAR drag term
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub bstar: Option<f64>,
    /// First derivative of mean motion
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub mean_motion_dot: Option<f64>,
    /// Second derivative of mean motion
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub mean_motion_ddot: Option<f64>,
    /// File ID
    #[serde(default, deserialize_with = "deserialize_optional_u64")]
    pub file: Option<u64>,
    /// TLE line 1
    #[serde(default)]
    pub tle_line1: Option<String>,
    /// TLE line 2
    #[serde(default)]
    pub tle_line2: Option<String>,
    /// TLE line 0 (name)
    #[serde(default)]
    pub tle_line0: Option<String>,
    /// Object ID
    #[serde(default)]
    pub object_id: Option<String>,
    /// Object number
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub object_number: Option<u32>,
    /// Semi-major axis (km)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub semimajor_axis: Option<f64>,
    /// Period (minutes)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub period: Option<f64>,
    /// Apoapsis (km)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub apoapsis: Option<f64>,
    /// Periapsis (km)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub periapsis: Option<f64>,
    /// Data status code
    #[serde(default)]
    pub data_status_code: Option<String>,
    /// Ordinal
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub ordinal: Option<u32>,
}

impl TLERecord {
    /// Convert this TLE record to an SGP propagator.
    ///
    /// # Arguments
    ///
    /// * `step_size` - The propagator step size in seconds
    ///
    /// # Returns
    ///
    /// An `SGPPropagator` initialized with this TLE record's data.
    ///
    /// # Errors
    ///
    /// Returns an error if the TLE record doesn't contain valid TLE data.
    pub fn to_sgp_propagator(&self, step_size: f64) -> Result<SGPPropagator, BraheError> {
        let name = self.tle_line0.as_ref().or(self.object_name.as_ref());
        let line1 = self
            .tle_line1
            .as_ref()
            .ok_or_else(|| BraheError::ParseError("TLE record missing line 1".to_string()))?;
        let line2 = self
            .tle_line2
            .as_ref()
            .ok_or_else(|| BraheError::ParseError("TLE record missing line 2".to_string()))?;

        SGPPropagator::from_3le(name.map(|s| s.as_str()), line1, line2, step_size)
    }
}

impl crate::spacetrack::query::HasTLEData for TLERecord {
    fn to_sgp_propagator(&self, step_size: f64) -> Result<SGPPropagator, BraheError> {
        TLERecord::to_sgp_propagator(self, step_size)
    }
}

define_request_class! {
    /// TLE request builder for querying Two-Line Element sets.
    ///
    /// **Note**: This class is deprecated. Use `GPRequest` instead.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{TLERequest, SpaceTrackClient};
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&TLERequest::new()
    ///     .norad_cat_id(25544)
    ///     .limit(1)
    /// ).await?;
    /// ```
    name: TLERequest,
    class_name: "tle",
    controller: "basicspacedata",
    record: TLERecord,
    predicates: {
        /// NORAD catalog ID
        norad_cat_id: u32,
        /// Object name
        object_name: String,
        /// Object ID
        object_id: String,
        /// Epoch
        epoch: String,
        /// Eccentricity
        eccentricity: f64,
        /// Inclination (degrees)
        inclination: f64,
        /// Mean motion (rev/day)
        mean_motion: f64,
        /// Right ascension of ascending node (degrees)
        ra_of_asc_node: f64,
        /// Argument of perigee (degrees)
        arg_of_pericenter: f64,
        /// Mean anomaly (degrees)
        mean_anomaly: f64,
        /// Object type
        object_type: String,
        /// File ID
        file: u64,
        /// Ordinal
        ordinal: u32,
    }
}

define_request_class! {
    /// TLE Latest request builder for querying the most recent TLE for each object.
    ///
    /// **Note**: This class is deprecated. Use `GPRequest` instead.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{TLELatestRequest, SpaceTrackClient};
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&TLELatestRequest::new()
    ///     .norad_cat_id(25544)
    /// ).await?;
    /// ```
    name: TLELatestRequest,
    class_name: "tle_latest",
    controller: "basicspacedata",
    record: TLERecord,
    predicates: {
        /// NORAD catalog ID
        norad_cat_id: u32,
        /// Object name
        object_name: String,
        /// Object ID
        object_id: String,
        /// Ordinal (1 = latest, 2 = 2nd latest, etc.)
        ordinal: u32,
        /// Object type
        object_type: String,
    }
}

define_request_class! {
    /// TLE Publish request builder for querying TLE publication information.
    ///
    /// **Note**: This class is deprecated. Use `GPRequest` instead.
    name: TLEPublishRequest,
    class_name: "tle_publish",
    controller: "basicspacedata",
    record: TLERecord,
    predicates: {
        /// Publish epoch
        publish_epoch: String,
        /// NORAD catalog ID
        norad_cat_id: u32,
        /// Object name
        object_name: String,
    }
}

/// Extension trait for converting vectors of TLE records to propagators.
pub trait TLERecordVecExt {
    /// Convert all TLE records to SGP propagators.
    fn to_sgp_propagators(&self, step_size: f64) -> Result<Vec<SGPPropagator>, BraheError>;

    /// Convert all TLE records to SGP propagators, skipping failures.
    fn to_sgp_propagators_skip_errors(&self, step_size: f64) -> Vec<SGPPropagator>;
}

impl TLERecordVecExt for Vec<TLERecord> {
    fn to_sgp_propagators(&self, step_size: f64) -> Result<Vec<SGPPropagator>, BraheError> {
        self.iter()
            .map(|r| r.to_sgp_propagator(step_size))
            .collect()
    }

    fn to_sgp_propagators_skip_errors(&self, step_size: f64) -> Vec<SGPPropagator> {
        self.iter()
            .filter_map(|r| r.to_sgp_propagator(step_size).ok())
            .collect()
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::spacetrack::request_classes::RequestClass;
    use crate::utils::testing::setup_global_test_eop;

    const ISS_LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    const ISS_LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    #[test]
    fn test_tle_request_new() {
        let req = TLERequest::new();
        assert!(req.norad_cat_id.is_none());
    }

    #[test]
    fn test_tle_request_class_name() {
        assert_eq!(TLERequest::class_name(), "tle");
    }

    #[test]
    fn test_tle_latest_request_class_name() {
        assert_eq!(TLELatestRequest::class_name(), "tle_latest");
    }

    #[test]
    fn test_tle_publish_request_class_name() {
        assert_eq!(TLEPublishRequest::class_name(), "tle_publish");
    }

    #[test]
    fn test_tle_request_controller() {
        assert_eq!(TLERequest::controller(), "basicspacedata");
    }

    #[test]
    fn test_tle_record_to_propagator() {
        setup_global_test_eop();
        let record = TLERecord {
            norad_cat_id: Some(25544),
            object_name: Some("ISS (ZARYA)".to_string()),
            tle_line1: Some(ISS_LINE1.to_string()),
            tle_line2: Some(ISS_LINE2.to_string()),
            ..Default::default()
        };

        let result = record.to_sgp_propagator(60.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tle_record_to_propagator_missing_line() {
        let record = TLERecord {
            norad_cat_id: Some(25544),
            ..Default::default()
        };

        let result = record.to_sgp_propagator(60.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_tle_record_vec_to_propagators() {
        setup_global_test_eop();
        let records = vec![TLERecord {
            norad_cat_id: Some(25544),
            tle_line1: Some(ISS_LINE1.to_string()),
            tle_line2: Some(ISS_LINE2.to_string()),
            ..Default::default()
        }];

        let result = records.to_sgp_propagators(60.0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }
}
