/*!
 * GP History request class
 *
 * The GP History class provides access to historical GP element sets.
 */

use serde::{Deserialize, Serialize};

use crate::define_request_class;
use crate::propagators::SGPPropagator;
use crate::utils::BraheError;

/// GP History record containing historical orbital elements.
///
/// This struct represents a single historical GP record as returned by the SpaceTrack API.
/// It has the same structure as GPRecord.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct GPHistoryRecord {
    /// CCSDS OMM version
    #[serde(default)]
    pub ccsds_omm_vers: Option<String>,
    /// Comment
    #[serde(default)]
    pub comment: Option<String>,
    /// Creation date
    #[serde(default)]
    pub creation_date: Option<String>,
    /// Originator
    #[serde(default)]
    pub originator: Option<String>,
    /// Object name
    #[serde(default)]
    pub object_name: Option<String>,
    /// Object ID
    #[serde(default)]
    pub object_id: Option<String>,
    /// Center name
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
    /// Epoch
    #[serde(default)]
    pub epoch: Option<String>,
    /// Mean motion (rev/day)
    #[serde(default)]
    pub mean_motion: Option<f64>,
    /// Eccentricity
    #[serde(default)]
    pub eccentricity: Option<f64>,
    /// Inclination (degrees)
    #[serde(default)]
    pub inclination: Option<f64>,
    /// Right ascension of ascending node (degrees)
    #[serde(default)]
    pub ra_of_asc_node: Option<f64>,
    /// Argument of pericenter (degrees)
    #[serde(default)]
    pub arg_of_pericenter: Option<f64>,
    /// Mean anomaly (degrees)
    #[serde(default)]
    pub mean_anomaly: Option<f64>,
    /// Ephemeris type
    #[serde(default)]
    pub ephemeris_type: Option<i32>,
    /// Classification type
    #[serde(default)]
    pub classification_type: Option<String>,
    /// NORAD catalog ID
    #[serde(default)]
    pub norad_cat_id: Option<u32>,
    /// Element set number
    #[serde(default)]
    pub element_set_no: Option<u32>,
    /// Revolution number at epoch
    #[serde(default)]
    pub rev_at_epoch: Option<u32>,
    /// BSTAR drag term
    #[serde(default)]
    pub bstar: Option<f64>,
    /// First derivative of mean motion
    #[serde(default)]
    pub mean_motion_dot: Option<f64>,
    /// Second derivative of mean motion
    #[serde(default)]
    pub mean_motion_ddot: Option<f64>,
    /// Semi-major axis (km)
    #[serde(default)]
    pub semimajor_axis: Option<f64>,
    /// Period (minutes)
    #[serde(default)]
    pub period: Option<f64>,
    /// Apoapsis (km)
    #[serde(default)]
    pub apoapsis: Option<f64>,
    /// Periapsis (km)
    #[serde(default)]
    pub periapsis: Option<f64>,
    /// Object type
    #[serde(default)]
    pub object_type: Option<String>,
    /// RCS size
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
    /// Decay date
    #[serde(default)]
    pub decay_date: Option<String>,
    /// Decayed flag
    #[serde(default)]
    pub decayed: Option<i32>,
    /// File ID
    #[serde(default)]
    pub file: Option<u64>,
    /// GP ID
    #[serde(default)]
    pub gp_id: Option<u64>,
    /// TLE line 0
    #[serde(default)]
    pub tle_line0: Option<String>,
    /// TLE line 1
    #[serde(default)]
    pub tle_line1: Option<String>,
    /// TLE line 2
    #[serde(default)]
    pub tle_line2: Option<String>,
}

impl GPHistoryRecord {
    /// Convert this GP History record to an SGP propagator.
    ///
    /// # Arguments
    ///
    /// * `step_size` - The propagator step size in seconds
    ///
    /// # Returns
    ///
    /// An `SGPPropagator` initialized with this record's TLE data.
    pub fn to_sgp_propagator(&self, step_size: f64) -> Result<SGPPropagator, BraheError> {
        let name = self.tle_line0.as_ref().or(self.object_name.as_ref());
        let line1 = self.tle_line1.as_ref().ok_or_else(|| {
            BraheError::ParseError("GP History record missing TLE line 1".to_string())
        })?;
        let line2 = self.tle_line2.as_ref().ok_or_else(|| {
            BraheError::ParseError("GP History record missing TLE line 2".to_string())
        })?;

        SGPPropagator::from_3le(name.map(|s| s.as_str()), line1, line2, step_size)
    }
}

impl crate::spacetrack::query::HasTLEData for GPHistoryRecord {
    fn to_sgp_propagator(&self, step_size: f64) -> Result<SGPPropagator, BraheError> {
        GPHistoryRecord::to_sgp_propagator(self, step_size)
    }
}

define_request_class! {
    /// GP History request builder for querying historical GP element sets.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use brahe::spacetrack::{GPHistoryRequest, SpaceTrackClient};
    /// use brahe::spacetrack::operators::inclusive_range;
    ///
    /// let client = SpaceTrackClient::new("user", "pass").await?;
    /// let records = client.query(&GPHistoryRequest::new()
    ///     .norad_cat_id(25544)
    ///     .epoch(inclusive_range("2024-01-01", "2024-01-31"))
    ///     .limit(100)
    /// ).await?;
    /// ```
    name: GPHistoryRequest,
    class_name: "gp_history",
    controller: "basicspacedata",
    record: GPHistoryRecord,
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
        /// Argument of pericenter (degrees)
        arg_of_pericenter: f64,
        /// Mean anomaly (degrees)
        mean_anomaly: f64,
        /// Semi-major axis (km)
        semimajor_axis: f64,
        /// Period (minutes)
        period: f64,
        /// Apoapsis (km)
        apoapsis: f64,
        /// Periapsis (km)
        periapsis: f64,
        /// Object type
        object_type: String,
        /// RCS size
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

/// Extension trait for converting vectors of GP History records to propagators.
pub trait GPHistoryRecordVecExt {
    /// Convert all GP History records to SGP propagators.
    fn to_sgp_propagators(&self, step_size: f64) -> Result<Vec<SGPPropagator>, BraheError>;

    /// Convert all GP History records to SGP propagators, skipping failures.
    fn to_sgp_propagators_skip_errors(&self, step_size: f64) -> Vec<SGPPropagator>;
}

impl GPHistoryRecordVecExt for Vec<GPHistoryRecord> {
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
    use crate::spacetrack::operators::inclusive_range;
    use crate::spacetrack::request_classes::RequestClass;
    use crate::utils::testing::setup_global_test_eop;

    const ISS_LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    const ISS_LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    #[test]
    fn test_gp_history_request_new() {
        let req = GPHistoryRequest::new();
        assert!(req.norad_cat_id.is_none());
    }

    #[test]
    fn test_gp_history_request_class_name() {
        assert_eq!(GPHistoryRequest::class_name(), "gp_history");
    }

    #[test]
    fn test_gp_history_request_controller() {
        assert_eq!(GPHistoryRequest::controller(), "basicspacedata");
    }

    #[test]
    fn test_gp_history_request_with_range() {
        let req = GPHistoryRequest::new()
            .norad_cat_id(25544)
            .epoch(inclusive_range("2024-01-01", "2024-01-31"))
            .limit(100);

        let predicates = req.predicates();
        assert!(!predicates.is_empty());
    }

    #[test]
    fn test_gp_history_record_to_propagator() {
        setup_global_test_eop();
        let record = GPHistoryRecord {
            norad_cat_id: Some(25544),
            object_name: Some("ISS (ZARYA)".to_string()),
            tle_line1: Some(ISS_LINE1.to_string()),
            tle_line2: Some(ISS_LINE2.to_string()),
            ..Default::default()
        };

        let result = record.to_sgp_propagator(60.0);
        assert!(result.is_ok());
    }
}
