/*!
 * GP (General Perturbations) request class
 *
 * The GP class provides access to the latest GP element sets for all
 * cataloged objects.
 */

use serde::{Deserialize, Serialize};

use crate::define_request_class;
use crate::propagators::SGPPropagator;
use crate::spacetrack::serde_helpers::{
    deserialize_optional_f64, deserialize_optional_i32, deserialize_optional_u32,
    deserialize_optional_u64,
};
use crate::utils::BraheError;

/// GP record containing orbital elements and metadata.
///
/// This struct represents a single GP element set as returned by the
/// SpaceTrack API.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct GPRecord {
    /// CCSDS OMM version
    #[serde(default)]
    pub ccsds_omm_vers: Option<String>,
    /// Comment field
    #[serde(default)]
    pub comment: Option<String>,
    /// Creation date of the element set
    #[serde(default)]
    pub creation_date: Option<String>,
    /// Originator of the element set
    #[serde(default)]
    pub originator: Option<String>,
    /// Object name
    #[serde(default)]
    pub object_name: Option<String>,
    /// Object ID (international designator)
    #[serde(default)]
    pub object_id: Option<String>,
    /// Center name (typically "EARTH")
    #[serde(default)]
    pub center_name: Option<String>,
    /// Reference frame
    #[serde(default)]
    pub ref_frame: Option<String>,
    /// Time system
    #[serde(default)]
    pub time_system: Option<String>,
    /// Mean element theory
    #[serde(default)]
    pub mean_element_theory: Option<String>,
    /// Epoch of the element set
    #[serde(default)]
    pub epoch: Option<String>,
    /// Mean motion (revolutions per day)
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
    /// Argument of pericenter (degrees)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub arg_of_pericenter: Option<f64>,
    /// Mean anomaly (degrees)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub mean_anomaly: Option<f64>,
    /// Ephemeris type
    #[serde(default, deserialize_with = "deserialize_optional_i32")]
    pub ephemeris_type: Option<i32>,
    /// Classification type
    #[serde(default)]
    pub classification_type: Option<String>,
    /// NORAD catalog ID
    #[serde(default, deserialize_with = "deserialize_optional_u32")]
    pub norad_cat_id: Option<u32>,
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
    /// Semi-major axis (km)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub semimajor_axis: Option<f64>,
    /// Orbital period (minutes)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub period: Option<f64>,
    /// Apoapsis altitude (km)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub apoapsis: Option<f64>,
    /// Periapsis altitude (km)
    #[serde(default, deserialize_with = "deserialize_optional_f64")]
    pub periapsis: Option<f64>,
    /// Object type (PAYLOAD, ROCKET BODY, DEBRIS, etc.)
    #[serde(default)]
    pub object_type: Option<String>,
    /// RCS size category (SMALL, MEDIUM, LARGE)
    #[serde(default)]
    pub rcs_size: Option<String>,
    /// Country code
    #[serde(default)]
    pub country_code: Option<String>,
    /// Launch date
    #[serde(default)]
    pub launch_date: Option<String>,
    /// Launch site
    #[serde(default)]
    pub site: Option<String>,
    /// Decay date (if decayed)
    #[serde(default)]
    pub decay_date: Option<String>,
    /// Data decay flag
    #[serde(default, deserialize_with = "deserialize_optional_i32")]
    pub decayed: Option<i32>,
    /// File ID
    #[serde(default, deserialize_with = "deserialize_optional_u64")]
    pub file: Option<u64>,
    /// GP ID
    #[serde(default, deserialize_with = "deserialize_optional_u64")]
    pub gp_id: Option<u64>,
    /// TLE line 0 (name)
    #[serde(default)]
    pub tle_line0: Option<String>,
    /// TLE line 1
    #[serde(default)]
    pub tle_line1: Option<String>,
    /// TLE line 2
    #[serde(default)]
    pub tle_line2: Option<String>,
}

impl GPRecord {
    /// Convert this GP record to an SGP propagator.
    ///
    /// # Arguments
    ///
    /// * `step_size` - The propagator step size in seconds
    ///
    /// # Returns
    ///
    /// An `SGPPropagator` initialized with this GP record's TLE data.
    ///
    /// # Errors
    ///
    /// Returns an error if the GP record doesn't contain valid TLE data.
    pub fn to_sgp_propagator(&self, step_size: f64) -> Result<SGPPropagator, BraheError> {
        let name = self.tle_line0.as_ref().or(self.object_name.as_ref());
        let line1 = self
            .tle_line1
            .as_ref()
            .ok_or_else(|| BraheError::ParseError("GP record missing TLE line 1".to_string()))?;
        let line2 = self
            .tle_line2
            .as_ref()
            .ok_or_else(|| BraheError::ParseError("GP record missing TLE line 2".to_string()))?;

        SGPPropagator::from_3le(name.map(|s| s.as_str()), line1, line2, step_size)
    }
}

impl crate::spacetrack::query::HasTLEData for GPRecord {
    fn to_sgp_propagator(&self, step_size: f64) -> Result<SGPPropagator, BraheError> {
        GPRecord::to_sgp_propagator(self, step_size)
    }
}

define_request_class! {
    /// GP request builder for querying General Perturbations element sets.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{GPRequest, SpaceTrackClient};
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&GPRequest::new()
    ///     .norad_cat_id(25544)
    ///     .limit(1)
    /// ).await?;
    /// ```
    name: GPRequest,
    class_name: "gp",
    controller: "basicspacedata",
    record: GPRecord,
    predicates: {
        /// NORAD catalog ID
        norad_cat_id: u32,
        /// Object name
        object_name: String,
        /// Object ID (international designator)
        object_id: String,
        /// Epoch of element set
        epoch: String,
        /// Eccentricity
        eccentricity: f64,
        /// Inclination (degrees)
        inclination: f64,
        /// Mean motion (rev/day)
        mean_motion: f64,
        /// Right ascension of ascending node (degrees)
        ra_of_asc_node: f64,
        /// Argument of pericenter (degrees)
        arg_of_pericenter: f64,
        /// Mean anomaly (degrees)
        mean_anomaly: f64,
        /// Semi-major axis (km)
        semimajor_axis: f64,
        /// Orbital period (minutes)
        period: f64,
        /// Apoapsis altitude (km)
        apoapsis: f64,
        /// Periapsis altitude (km)
        periapsis: f64,
        /// Object type (PAYLOAD, ROCKET BODY, DEBRIS)
        object_type: String,
        /// RCS size category (SMALL, MEDIUM, LARGE)
        rcs_size: String,
        /// Country code
        country_code: String,
        /// Launch date
        launch_date: String,
        /// Decay date
        decay_date: String,
        /// Creation date
        creation_date: String,
        /// File ID
        file: u64,
        /// GP ID
        gp_id: u64,
        /// Decayed flag
        decayed: i32,
    }
}

/// Extension trait for converting vectors of GP records to propagators.
pub trait GPRecordVecExt {
    /// Convert all GP records to SGP propagators.
    ///
    /// # Arguments
    ///
    /// * `step_size` - The propagator step size in seconds
    ///
    /// # Returns
    ///
    /// A vector of `SGPPropagator` instances. Records that fail to convert
    /// are returned as errors.
    fn to_sgp_propagators(&self, step_size: f64) -> Result<Vec<SGPPropagator>, BraheError>;

    /// Convert all GP records to SGP propagators, skipping failures.
    ///
    /// # Arguments
    ///
    /// * `step_size` - The propagator step size in seconds
    ///
    /// # Returns
    ///
    /// A vector of `SGPPropagator` instances. Records that fail to convert
    /// are silently skipped.
    fn to_sgp_propagators_skip_errors(&self, step_size: f64) -> Vec<SGPPropagator>;
}

impl GPRecordVecExt for Vec<GPRecord> {
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
    use crate::spacetrack::operators::greater_than;
    use crate::spacetrack::request_classes::RequestClass;
    use crate::utils::testing::setup_global_test_eop;

    // Valid TLE data for testing
    const ISS_LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    const ISS_LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    #[test]
    fn test_gp_request_new() {
        let req = GPRequest::new();
        assert!(req.norad_cat_id.is_none());
        assert!(req.limit.is_none());
    }

    #[test]
    fn test_gp_request_builder() {
        let req = GPRequest::new()
            .norad_cat_id(25544)
            .limit(10)
            .orderby_desc("epoch");

        assert!(req.norad_cat_id.is_some());
        assert_eq!(req.limit, Some(10));
        assert!(req.orderby.is_some());
    }

    #[test]
    fn test_gp_request_class_name() {
        assert_eq!(GPRequest::class_name(), "gp");
    }

    #[test]
    fn test_gp_request_controller() {
        assert_eq!(GPRequest::controller(), "basicspacedata");
    }

    #[test]
    fn test_gp_request_predicates() {
        let req = GPRequest::new().norad_cat_id(25544).limit(5);

        let predicates = req.predicates();
        assert!(!predicates.is_empty());
    }

    #[test]
    fn test_gp_request_with_operators() {
        let req = GPRequest::new()
            .epoch(greater_than("2024-01-01"))
            .eccentricity(crate::spacetrack::operators::less_than(0.01));

        let predicates = req.predicates();
        assert_eq!(predicates.len(), 2);
    }

    #[test]
    fn test_gp_record_deserialize() {
        let json = format!(
            r#"{{
            "NORAD_CAT_ID": 25544,
            "OBJECT_NAME": "ISS (ZARYA)",
            "EPOCH": "2024-01-01 00:00:00",
            "MEAN_MOTION": 15.5,
            "ECCENTRICITY": 0.0001234,
            "INCLINATION": 51.6416,
            "TLE_LINE1": "{}",
            "TLE_LINE2": "{}"
        }}"#,
            ISS_LINE1, ISS_LINE2
        );

        let record: GPRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(record.norad_cat_id, Some(25544));
        assert_eq!(record.object_name, Some("ISS (ZARYA)".to_string()));
        assert!(record.tle_line1.is_some());
        assert!(record.tle_line2.is_some());
    }

    #[test]
    fn test_gp_record_to_propagator() {
        setup_global_test_eop();
        let record = GPRecord {
            norad_cat_id: Some(25544),
            object_name: Some("ISS (ZARYA)".to_string()),
            tle_line0: None,
            tle_line1: Some(ISS_LINE1.to_string()),
            tle_line2: Some(ISS_LINE2.to_string()),
            ..Default::default()
        };

        let result = record.to_sgp_propagator(60.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gp_record_to_propagator_missing_tle() {
        let record = GPRecord {
            norad_cat_id: Some(25544),
            ..Default::default()
        };

        let result = record.to_sgp_propagator(60.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_gp_record_vec_to_propagators() {
        setup_global_test_eop();
        let records = vec![GPRecord {
            norad_cat_id: Some(25544),
            object_name: Some("ISS".to_string()),
            tle_line1: Some(ISS_LINE1.to_string()),
            tle_line2: Some(ISS_LINE2.to_string()),
            ..Default::default()
        }];

        let result = records.to_sgp_propagators(60.0);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_gp_record_vec_skip_errors() {
        setup_global_test_eop();
        let records = vec![
            GPRecord {
                norad_cat_id: Some(25544),
                object_name: Some("ISS".to_string()),
                tle_line1: Some(ISS_LINE1.to_string()),
                tle_line2: Some(ISS_LINE2.to_string()),
                ..Default::default()
            },
            GPRecord {
                norad_cat_id: Some(99999),
                // Missing TLE lines - should be skipped
                ..Default::default()
            },
        ];

        let propagators = records.to_sgp_propagators_skip_errors(60.0);
        assert_eq!(propagators.len(), 1);
    }
}
