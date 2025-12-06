/*!
 * SGP4 orbital propagator with integrated Two-Line Element (TLE) parsing and validation.
 *
 * This module provides a complete implementation of the SGP4 (Simplified General Perturbations 4)
 * orbital propagator, designed for propagating satellite orbits from Two-Line Element sets.
 * SGP4 is the standard propagation model for near-Earth satellites and is widely used
 * for operational satellite tracking.
 *
 * # Key Features
 * - Full SGP4/SDP4 propagation model implementation
 * - Integrated TLE parsing with checksum validation
 * - Support for both classic and Alpha-5 TLE formats
 * - High-precision epoch extraction and conversion
 * - Integration with trajectory management system
 * - Error handling for malformed TLE data
 *
 * # TLE Format Support
 * - **Classic**: Traditional 2-line format with numeric NORAD catalog numbers
 * - **Alpha-5**: Extended format supporting alphanumeric catalog numbers (>= 100000)
 *
 * # Accuracy and Limitations
 * - Best accuracy for near-Earth satellites (altitude < 2000 km)
 * - Suitable for short to medium-term propagation (days to weeks)
 * - Not recommended for high-precision applications or long-term propagation
 * - Accuracy degrades for highly eccentric or deep-space orbits
 *
 * # References
 * - Hoots, F. R., & Roehrich, R. L. (1980). Models for Propagation of NORAD Element Sets.
 * - Vallado, D. A., et al. (2006). Revisiting Spacetrack Report #3.
 */

use crate::attitude::RotationMatrix;
use crate::constants::{AngleFormat, DEG2RAD, OMEGA_EARTH, RAD2DEG};
use crate::coordinates::state_eci_to_koe;
use crate::frames::{polar_motion, state_ecef_to_eci, state_gcrf_to_eme2000, state_itrf_to_gcrf};
use crate::orbits::tle::{
    TleFormat, calculate_tle_line_checksum, create_tle_lines, epoch_from_tle,
    norad_id_numeric_to_alpha5, parse_norad_id, validate_tle_lines,
};
use crate::propagators::traits::{SOrbitStateProvider, SStatePropagator, SStateProvider};
use crate::time::{Epoch, TimeSystem};
use crate::trajectories::DOrbitTrajectory;
use crate::trajectories::traits::{OrbitFrame, OrbitRepresentation, Trajectory};
use crate::utils::{BraheError, Identifiable};
use nalgebra::{DVector, Vector3, Vector6};
use sgp4::chrono::{Datelike, NaiveDateTime, Timelike};

// Event detection imports
use crate::events::{DDetectedEvent, DEventDetector, EventAction, EventQuery, dscan_for_event};

/// Helper functions
/// Compute Greenwich Mean Sidereal Time 1982 Model. Formulae taken from
/// `Revisiting Spacetrack Report No 3` by David Vallado for use in transforming
/// between the TEME and PEF frames.
///
/// # Arguments:
/// * epoch (:obj:`Epoch`): Epoch of transformation
///
/// # Returns:
/// * Greenwich mean sidereal time as angle. Units: Radians [0, 2pi)
/// * Rate of change of Greenwich mean sidereal time as angle. Units: Radians/second [0, 2pi)
fn tle_gmst82(epoch: Epoch, angle_format: AngleFormat) -> f64 {
    // Calculate Julian Date in UT1
    let jd_ut1 = epoch.jd_as_time_system(TimeSystem::UT1);
    let tut1 = (jd_ut1 - 2451545.0) / 36525.0;

    // GMST in seconds
    let gmst_sec =
        67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * tut1 + 0.093104 * tut1 * tut1
            - 6.2e-6 * tut1 * tut1 * tut1;

    // Normalize to [0, 86400)
    let theta = (gmst_sec * DEG2RAD / 240.0) % (2.0 * std::f64::consts::PI);

    // Convert to radians or degrees
    match angle_format {
        AngleFormat::Radians => theta,
        AngleFormat::Degrees => theta * RAD2DEG,
    }
}

fn convert_state_from_spg4_frame(
    epoch: Epoch,
    tle_state: Vector6<f64>,
    frame: OrbitFrame,
    representation: OrbitRepresentation,
    angle_format: Option<AngleFormat>,
) -> Vector6<f64> {
    // SGP4 outputs state in TEME
    // Conversion chain is TEME -> PEF -> ECEF -> ECI

    // Step 1: TEME to PEF
    let gmst = tle_gmst82(epoch, AngleFormat::Radians);
    #[allow(non_snake_case)]
    let R = RotationMatrix::Rz(gmst, AngleFormat::Radians);
    let omega_earth = Vector3::new(0.0, 0.0, OMEGA_EARTH); // rad/s

    let r_pef: Vector3<f64> = R * Vector3::<f64>::from(tle_state.fixed_rows::<3>(0));
    let v_pef: Vector3<f64> =
        R * Vector3::<f64>::from(tle_state.fixed_rows::<3>(3)) - omega_earth.cross(&r_pef);

    // Step 2: PEF to ECEF
    #[allow(non_snake_case)]
    let PM = polar_motion(epoch);

    let r_ecef = PM * r_pef;
    let v_ecef = PM * v_pef;
    let ecef_state = Vector6::new(
        r_ecef[0], r_ecef[1], r_ecef[2], v_ecef[0], v_ecef[1], v_ecef[2],
    );

    match representation {
        OrbitRepresentation::Cartesian => match frame {
            OrbitFrame::ECI => state_ecef_to_eci(epoch, ecef_state),
            OrbitFrame::GCRF => state_ecef_to_eci(epoch, ecef_state),
            OrbitFrame::EME2000 => {
                let gcrf_state = state_ecef_to_eci(epoch, ecef_state);
                state_gcrf_to_eme2000(gcrf_state)
            }
            OrbitFrame::ECEF => ecef_state,
            OrbitFrame::ITRF => ecef_state,
        },
        OrbitRepresentation::Keplerian => {
            if frame != OrbitFrame::ECI && frame != OrbitFrame::GCRF {
                panic!("Keplerian elements must be in ECI or GCRF frame");
            }

            if let Some(format) = angle_format {
                state_eci_to_koe(state_ecef_to_eci(epoch, ecef_state), format)
            } else {
                panic!("Angle format must be specified for Keplerian elements");
            }
        }
    }
}

/// Convert Vector6 to DVector for trajectory storage
#[inline]
fn svec6_to_dvec(sv: &Vector6<f64>) -> DVector<f64> {
    DVector::from_column_slice(sv.as_slice())
}

/// Convert DVector to Vector6 for internal SGP4 usage
#[inline]
fn dvec_to_svec6(dv: &DVector<f64>) -> Vector6<f64> {
    Vector6::from_column_slice(&dv.as_slice()[0..6])
}

/// SGP4 propagator
#[allow(non_camel_case_types)]
pub struct SGPPropagator {
    /// Raw first line of TLE
    pub line1: String,

    /// Raw second line of TLE
    pub line2: String,

    /// Optional satellite name (from 3-line format)
    pub satellite_name: Option<String>,

    /// TLE format type (Classic or Alpha-5)
    pub format: TleFormat,

    /// Original NORAD ID string (may contain Alpha-5 encoding)
    pub norad_id_string: String,

    /// Decoded numeric NORAD ID
    pub norad_id: u32,

    /// SGP4 propagation constants
    constants: sgp4::Constants,

    /// Epoch from TLE
    pub epoch: Epoch,

    /// Initial state vector (always ECI Cartesian from SGP4)
    initial_state: Vector6<f64>,

    /// Accumulated trajectory with configurable management
    pub trajectory: DOrbitTrajectory,

    /// Step size in seconds for stepping operations
    pub step_size: f64,

    /// Output frame (default: ECI)
    pub frame: OrbitFrame,

    /// Output representation (default: Cartesian)
    pub representation: OrbitRepresentation,

    /// Output angle format (default: Radians)
    pub angle_format: Option<AngleFormat>,

    /// Optional user-defined name for identification
    pub name: Option<String>,

    /// Optional user-defined numeric ID for identification
    pub id: Option<u64>,

    /// Optional UUID for unique identification
    pub uuid: Option<uuid::Uuid>,

    // ===== Event Detection =====
    /// Event detectors for monitoring propagation
    event_detectors: Vec<Box<dyn DEventDetector>>,

    /// Log of detected events
    event_log: Vec<DDetectedEvent>,

    /// Termination flag (set by terminal events)
    terminated: bool,
}

impl Clone for SGPPropagator {
    /// Clone the propagator.
    ///
    /// Note: Event detectors are NOT cloned (they contain trait objects).
    /// The cloned propagator starts fresh with no event detectors,
    /// an empty event log, and terminated = false.
    fn clone(&self) -> Self {
        SGPPropagator {
            line1: self.line1.clone(),
            line2: self.line2.clone(),
            satellite_name: self.satellite_name.clone(),
            format: self.format,
            norad_id_string: self.norad_id_string.clone(),
            norad_id: self.norad_id,
            constants: self.constants.clone(),
            epoch: self.epoch,
            initial_state: self.initial_state,
            trajectory: self.trajectory.clone(),
            step_size: self.step_size,
            frame: self.frame,
            representation: self.representation,
            angle_format: self.angle_format,
            name: self.name.clone(),
            id: self.id,
            uuid: self.uuid,
            // Event detection fields are reset on clone
            // (trait objects cannot be cloned)
            event_detectors: Vec::new(),
            event_log: Vec::new(),
            terminated: false,
        }
    }
}

impl std::fmt::Debug for SGPPropagator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SGPPropagator")
            .field("line1", &self.line1)
            .field("line2", &self.line2)
            .field("satellite_name", &self.satellite_name)
            .field("format", &self.format)
            .field("norad_id_string", &self.norad_id_string)
            .field("norad_id", &self.norad_id)
            .field("epoch", &self.epoch)
            .field("initial_state", &self.initial_state)
            .field("step_size", &self.step_size)
            .field("frame", &self.frame)
            .field("representation", &self.representation)
            .field("angle_format", &self.angle_format)
            .field("name", &self.name)
            .field("id", &self.id)
            .field("uuid", &self.uuid)
            .field(
                "event_detectors",
                &format!("[{} detectors]", self.event_detectors.len()),
            )
            .field("event_log", &self.event_log)
            .field("terminated", &self.terminated)
            .finish()
    }
}

impl SGPPropagator {
    /// Create a new SGP propagator from TLE lines
    ///
    /// # Arguments
    /// * `line1` - First line of TLE data
    /// * `line2` - Second line of TLE data
    /// * `step_size` - Default step size in seconds
    ///
    /// # Returns
    /// * `Result<SGPPropagator, BraheError>` - New SGP propagator instance or error
    pub fn from_tle(line1: &str, line2: &str, step_size: f64) -> Result<Self, BraheError> {
        Self::from_3le(None, line1, line2, step_size)
    }

    /// Create a new SGP propagator from 3-line TLE format
    ///
    /// # Arguments
    /// * `name` - Optional satellite name (line 0)
    /// * `line1` - First line of TLE data
    /// * `line2` - Second line of TLE data
    /// * `step_size` - Default step size in seconds
    ///
    /// # Returns
    /// * `Result<SGPPropagator, BraheError>` - New SGP propagator instance or error
    pub fn from_3le(
        name: Option<&str>,
        line1: &str,
        line2: &str,
        step_size: f64,
    ) -> Result<Self, BraheError> {
        // Validate TLE format
        if !validate_tle_lines(line1, line2) {
            return Err(BraheError::Error("Invalid TLE format".to_string()));
        }

        // Extract NORAD ID and determine format
        let norad_id_string = line1[2..7].trim().to_string();
        let norad_id = parse_norad_id(&norad_id_string)?;
        let format = if norad_id_string
            .chars()
            .next()
            .unwrap_or('0')
            .is_alphabetic()
        {
            TleFormat::Alpha5
        } else {
            TleFormat::Classic
        };

        // For Alpha-5 format, zero out NORAD ID for SGP4 library compatibility
        let (sgp4_line1, sgp4_line2) = if format == TleFormat::Alpha5 {
            // Replace Alpha-5 NORAD ID with zeros for SGP4
            let mut line1_chars: Vec<char> = line1.chars().collect();
            let mut line2_chars: Vec<char> = line2.chars().collect();

            // Zero out positions 2-6 (NORAD ID field)
            for i in 2..7 {
                if i < line1_chars.len() {
                    line1_chars[i] = '0';
                }
                if i < line2_chars.len() {
                    line2_chars[i] = '0';
                }
            }

            // Recalculate checksums for modified lines
            let mut modified_line1: String = line1_chars.into_iter().collect();
            let mut modified_line2: String = line2_chars.into_iter().collect();

            // Replace the last character (checksum) with recalculated value
            if modified_line1.len() >= 69 {
                let new_checksum1 = calculate_tle_line_checksum(&modified_line1);
                modified_line1.replace_range(68..69, &new_checksum1.to_string());
            }
            if modified_line2.len() >= 69 {
                let new_checksum2 = calculate_tle_line_checksum(&modified_line2);
                modified_line2.replace_range(68..69, &new_checksum2.to_string());
            }

            (modified_line1, modified_line2)
        } else {
            (line1.to_string(), line2.to_string())
        };

        // Parse TLE using sgp4 library
        let elements = sgp4::Elements::from_tle(
            Some(norad_id.to_string()),
            sgp4_line1.as_bytes(),
            sgp4_line2.as_bytes(),
        )
        .map_err(|e| BraheError::Error(format!("SGP4 parsing error: {:?}", e)))?;

        let constants = sgp4::Constants::from_elements(&elements)
            .map_err(|e| BraheError::Error(format!("SGP4 constants error: {:?}", e)))?;

        // Extract epoch
        let epoch = epoch_from_tle(line1)?;

        // Compute initial state in ECI
        let prediction = constants
            .propagate(sgp4::MinutesSinceEpoch(0.0))
            .map_err(|e| BraheError::Error(format!("SGP4 propagation error: {:?}", e)))?;

        // Convert from km to m and km/s to m/s
        let tle_state = Vector6::new(
            prediction.position[0] * 1000.0,
            prediction.position[1] * 1000.0,
            prediction.position[2] * 1000.0,
            prediction.velocity[0] * 1000.0,
            prediction.velocity[1] * 1000.0,
            prediction.velocity[2] * 1000.0,
        );

        // Convert initial state to ECI Cartesian
        let initial_state = convert_state_from_spg4_frame(
            epoch,
            tle_state,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None, // angle_format is not meaningful for Cartesian
        );

        // Create trajectory with initial state
        let mut trajectory =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        // Set trajectory identity from propagator identity
        if let Some(n) = name {
            trajectory = trajectory.with_name(n);
        }
        // NORAD ID is always available (norad_id: u32)
        trajectory = trajectory.with_id(norad_id as u64);

        trajectory.add(epoch, svec6_to_dvec(&initial_state));

        Ok(SGPPropagator {
            line1: line1.to_string(),
            line2: line2.to_string(),
            satellite_name: name.map(|s| s.to_string()),
            format,
            norad_id_string,
            norad_id,
            constants,
            epoch,
            initial_state,
            trajectory,
            step_size,
            frame: OrbitFrame::ECI,
            representation: OrbitRepresentation::Cartesian,
            angle_format: None, // angle_format is not meaningful for Cartesian
            name: name.map(|s| s.to_string()),
            id: Some(norad_id as u64),
            uuid: None,
            // Event detection fields
            event_detectors: Vec::new(),
            event_log: Vec::new(),
            terminated: false,
        })
    }

    /// Create a new SGP propagator from CCSDS OMM (Orbit Mean-elements Message) fields.
    ///
    /// This method directly constructs an SGP4 propagator from OMM orbital elements,
    /// bypassing TLE parsing. It creates synthetic TLE lines for API consistency.
    ///
    /// # Arguments
    /// * `epoch` - ISO 8601 datetime string (e.g., "2025-11-29T20:01:44.058144")
    /// * `mean_motion` - Mean motion in revolutions per day
    /// * `eccentricity` - Orbital eccentricity (dimensionless)
    /// * `inclination` - Orbital inclination in degrees
    /// * `raan` - Right ascension of ascending node in degrees
    /// * `arg_of_pericenter` - Argument of pericenter in degrees
    /// * `mean_anomaly` - Mean anomaly in degrees
    /// * `norad_id` - NORAD catalog ID
    /// * `step_size` - Default step size in seconds
    /// * `object_name` - Optional satellite name (OBJECT_NAME)
    /// * `object_id` - Optional international designator (OBJECT_ID, e.g., "1998-067A")
    /// * `classification` - Classification character ('U', 'C', or 'S'), defaults to 'U'
    /// * `bstar` - B* drag term, defaults to 0.0
    /// * `mean_motion_dot` - First derivative of mean motion / 2, defaults to 0.0
    /// * `mean_motion_ddot` - Second derivative of mean motion / 6, defaults to 0.0
    /// * `ephemeris_type` - Ephemeris type (usually 0), defaults to 0
    /// * `element_set_no` - Element set number, defaults to 999
    /// * `rev_at_epoch` - Revolution number at epoch, defaults to 0
    ///
    /// # Returns
    /// * `Result<SGPPropagator, BraheError>` - New SGP propagator instance or error
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::SGPPropagator;
    /// use brahe::eop::{StaticEOPProvider, set_global_eop_provider};
    ///
    /// // Initialize EOP provider
    /// let eop = StaticEOPProvider::from_zero();
    /// set_global_eop_provider(eop);
    ///
    /// // ISS OMM data
    /// let prop = SGPPropagator::from_omm_elements(
    ///     "2025-11-29T20:01:44.058144",  // EPOCH
    ///     15.49193835,                    // MEAN_MOTION (rev/day)
    ///     0.0003723,                      // ECCENTRICITY
    ///     51.6312,                        // INCLINATION (degrees)
    ///     206.3646,                       // RA_OF_ASC_NODE (degrees)
    ///     184.1118,                       // ARG_OF_PERICENTER (degrees)
    ///     175.9840,                       // MEAN_ANOMALY (degrees)
    ///     25544,                          // NORAD_CAT_ID
    ///     60.0,                           // step_size (seconds)
    ///     Some("ISS (ZARYA)"),            // OBJECT_NAME
    ///     Some("1998-067A"),              // OBJECT_ID
    ///     Some('U'),                      // CLASSIFICATION_TYPE
    ///     Some(0.15237e-3),               // BSTAR
    ///     Some(0.801e-4),                 // MEAN_MOTION_DOT
    ///     Some(0.0),                      // MEAN_MOTION_DDOT
    ///     Some(0),                        // EPHEMERIS_TYPE
    ///     Some(999),                      // ELEMENT_SET_NO
    ///     Some(54085),                    // REV_AT_EPOCH
    /// ).unwrap();
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub fn from_omm_elements(
        epoch: &str,
        mean_motion: f64,
        eccentricity: f64,
        inclination: f64,
        raan: f64,
        arg_of_pericenter: f64,
        mean_anomaly: f64,
        norad_id: u64,
        step_size: f64,
        object_name: Option<&str>,
        object_id: Option<&str>,
        classification: Option<char>,
        bstar: Option<f64>,
        mean_motion_dot: Option<f64>,
        mean_motion_ddot: Option<f64>,
        ephemeris_type: Option<u8>,
        element_set_no: Option<u64>,
        rev_at_epoch: Option<u64>,
    ) -> Result<Self, BraheError> {
        // Parse epoch (OMM format: "2025-11-29T20:01:44.058144")
        let datetime = NaiveDateTime::parse_from_str(epoch, "%Y-%m-%dT%H:%M:%S%.f")
            .or_else(|_| NaiveDateTime::parse_from_str(epoch, "%Y-%m-%dT%H:%M:%S"))
            .map_err(|e| BraheError::Error(format!("Invalid epoch format '{}': {}", epoch, e)))?;

        // Map classification character to sgp4::Classification
        let classification_char = classification.unwrap_or('U');
        let sgp4_classification = match classification_char {
            'U' | ' ' => sgp4::Classification::Unclassified,
            'C' => sgp4::Classification::Classified,
            'S' => sgp4::Classification::Secret,
            c => {
                return Err(BraheError::Error(format!(
                    "Invalid classification character: '{}'. Must be 'U', 'C', or 'S'",
                    c
                )));
            }
        };

        // Apply defaults for optional fields
        let bstar_val = bstar.unwrap_or(0.0);
        let mean_motion_dot_val = mean_motion_dot.unwrap_or(0.0);
        let mean_motion_ddot_val = mean_motion_ddot.unwrap_or(0.0);
        let ephemeris_type_val = ephemeris_type.unwrap_or(0);
        let element_set_no_val = element_set_no.unwrap_or(999);
        let rev_at_epoch_val = rev_at_epoch.unwrap_or(0);

        // Directly construct sgp4::Elements
        let elements = sgp4::Elements {
            object_name: object_name.map(|s| s.to_string()),
            international_designator: object_id.map(|s| s.to_string()),
            norad_id,
            classification: sgp4_classification,
            datetime,
            mean_motion_dot: mean_motion_dot_val,
            mean_motion_ddot: mean_motion_ddot_val,
            drag_term: bstar_val,
            element_set_number: element_set_no_val,
            inclination,
            right_ascension: raan,
            eccentricity,
            argument_of_perigee: arg_of_pericenter,
            mean_anomaly,
            mean_motion,
            revolution_number: rev_at_epoch_val,
            ephemeris_type: ephemeris_type_val,
        };

        // Create SGP4 constants from elements
        let constants = sgp4::Constants::from_elements(&elements)
            .map_err(|e| BraheError::Error(format!("SGP4 constants error: {:?}", e)))?;

        // Convert chrono::NaiveDateTime to brahe Epoch
        let brahe_epoch = Epoch::from_datetime(
            datetime.year() as u32,
            datetime.month() as u8,
            datetime.day() as u8,
            datetime.hour() as u8,
            datetime.minute() as u8,
            datetime.second() as f64 + datetime.nanosecond() as f64 / 1e9,
            0.0, // nanoseconds already included in seconds
            TimeSystem::UTC,
        );

        // Generate synthetic TLE lines for API consistency
        // Convert OBJECT_ID from OMM format (e.g., "1998-067A") to TLE format (e.g., "98067A")
        let intl_designator = object_id.map(|id| {
            // OMM format: YYYY-NNNP (year-number-piece)
            // TLE format: YYNNNP (2-digit year, no hyphen)
            if id.len() >= 5 && id.chars().nth(4) == Some('-') {
                // Has a hyphen at position 4, convert 4-digit year to 2-digit
                let year_2digit = &id[2..4]; // Take last 2 digits of year
                let rest = &id[5..]; // Everything after the hyphen
                format!("{}{}", year_2digit, rest)
            } else {
                // Already in TLE format or unknown format, use as-is
                id.to_string()
            }
        });
        let intl_designator_ref = intl_designator.as_deref().unwrap_or("");

        let norad_id_str = norad_id_numeric_to_alpha5(norad_id as u32)?;
        let (line1, line2) = create_tle_lines(
            &brahe_epoch,
            &norad_id_str,
            classification_char,
            intl_designator_ref,
            mean_motion,
            eccentricity,
            inclination,
            raan,
            arg_of_pericenter,
            mean_anomaly,
            mean_motion_dot_val / 2.0,
            mean_motion_ddot_val / 6.0,
            bstar_val,
            ephemeris_type_val,
            element_set_no_val as u16,
            rev_at_epoch_val as u32,
        )?;

        // Determine TLE format based on NORAD ID
        let format = if norad_id >= 100000 {
            TleFormat::Alpha5
        } else {
            TleFormat::Classic
        };

        // Compute initial state at epoch
        let prediction = constants
            .propagate(sgp4::MinutesSinceEpoch(0.0))
            .map_err(|e| BraheError::Error(format!("SGP4 propagation error: {:?}", e)))?;

        // Convert from km to m and km/s to m/s
        let tle_state = Vector6::new(
            prediction.position[0] * 1000.0,
            prediction.position[1] * 1000.0,
            prediction.position[2] * 1000.0,
            prediction.velocity[0] * 1000.0,
            prediction.velocity[1] * 1000.0,
            prediction.velocity[2] * 1000.0,
        );

        // Convert initial state to ECI Cartesian
        let initial_state = convert_state_from_spg4_frame(
            brahe_epoch,
            tle_state,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
        );

        // Create trajectory with initial state
        let mut trajectory =
            DOrbitTrajectory::new(6, OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        // Set trajectory identity from propagator identity
        if let Some(n) = object_name {
            trajectory = trajectory.with_name(n);
        }
        trajectory = trajectory.with_id(norad_id);

        trajectory.add(brahe_epoch, svec6_to_dvec(&initial_state));

        Ok(SGPPropagator {
            line1,
            line2,
            satellite_name: object_name.map(|s| s.to_string()),
            format,
            norad_id_string: norad_id_str,
            norad_id: norad_id as u32,
            constants,
            epoch: brahe_epoch,
            initial_state,
            trajectory,
            step_size,
            frame: OrbitFrame::ECI,
            representation: OrbitRepresentation::Cartesian,
            angle_format: None,
            name: object_name.map(|s| s.to_string()),
            id: Some(norad_id),
            uuid: None,
            event_detectors: Vec::new(),
            event_log: Vec::new(),
            terminated: false,
        })
    }

    /// Configure output format for propagated states (builder pattern).
    ///
    /// Sets the reference frame, representation type, and angle units for propagation output.
    ///
    /// # Arguments
    /// - `frame`: Target reference frame (ECI or ECEF)
    /// - `representation`: State representation (Cartesian or Keplerian)
    /// - `angle_format`: Angle units (Degrees/Radians) - required for Keplerian, None for Cartesian
    ///
    /// # Panics
    /// - If Keplerian representation requested without angle_format
    /// - If Keplerian representation requested in ECEF frame
    /// - If Cartesian representation given with angle_format
    ///
    /// # Returns
    /// Self for method chaining
    pub fn with_output_format(
        mut self,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: Option<AngleFormat>,
    ) -> Self {
        // Validate inputs
        if representation == OrbitRepresentation::Keplerian && angle_format.is_none() {
            panic!("Angle format must be specified for Keplerian elements");
        }

        if representation == OrbitRepresentation::Keplerian && frame != OrbitFrame::ECI {
            panic!("Keplerian elements must be in ECI frame");
        }

        if representation == OrbitRepresentation::Cartesian && angle_format.is_some() {
            panic!("Angle format should be None for Cartesian representation");
        }

        self.frame = frame;
        self.representation = representation;
        self.angle_format = angle_format;

        // Reset trajectory to initial state only, preserving identity
        let name = self.trajectory.get_name().map(|s| s.to_string());
        let uuid = self.trajectory.get_uuid();
        let id = self.trajectory.get_id();

        self.trajectory = DOrbitTrajectory::new(6, frame, representation, angle_format)
            .with_identity(name.as_deref(), uuid, id);

        // Propagate to initial epoch and add to trajectory
        let prediction = self
            .constants
            .propagate(sgp4::MinutesSinceEpoch(0.0))
            .expect("SGP4 propagation failed");

        // Convert from km to m and km/s to m/s
        let tle_state = Vector6::new(
            prediction.position[0] * 1000.0,
            prediction.position[1] * 1000.0,
            prediction.position[2] * 1000.0,
            prediction.velocity[0] * 1000.0,
            prediction.velocity[1] * 1000.0,
            prediction.velocity[2] * 1000.0,
        );

        let initial_state = convert_state_from_spg4_frame(
            self.epoch,
            tle_state,
            frame,
            representation,
            angle_format,
        );
        self.trajectory
            .add(self.epoch, svec6_to_dvec(&initial_state));

        self
    }

    /// Internal propagation to target epoch, returning state in the internal
    /// TEME frame that is the output of SGP4.
    fn propagate_internal(&self, target_epoch: Epoch) -> Vector6<f64> {
        // Calculate minutes since TLE epoch
        let dt = (target_epoch - self.epoch) / 60.0; // Convert seconds to minutes

        // Propagate using SGP4
        let prediction = self
            .constants
            .propagate(sgp4::MinutesSinceEpoch(dt))
            .expect("SGP4 propagation failed");

        // Convert from km to m and km/s to m/s
        Vector6::new(
            prediction.position[0] * 1000.0,
            prediction.position[1] * 1000.0,
            prediction.position[2] * 1000.0,
            prediction.velocity[0] * 1000.0,
            prediction.velocity[1] * 1000.0,
            prediction.velocity[2] * 1000.0,
        )
    }

    /// Get propagated state in Pseudo-Earth-Fixed (PEF) frame.
    ///
    /// Propagates to the given epoch and transforms from TEME to PEF using simplified
    /// rotation (GMST only, no polar motion or nutation). For higher accuracy, use
    /// propagate() with ECEF output format.
    ///
    /// # Arguments
    /// - `epoch`: Time to propagate to
    ///
    /// # Returns
    /// State vector [x, y, z, vx, vy, vz] in PEF frame. Units: meters, meters/second.
    pub fn state_pef(&self, epoch: Epoch) -> Vector6<f64> {
        let tle_state = self.propagate_internal(epoch);
        // SGP4 outputs state in TEME
        // Conversion chain is TEME -> PEF

        // Step 1: TEME to PEF
        let gmst = tle_gmst82(epoch, AngleFormat::Radians);
        #[allow(non_snake_case)]
        let R = RotationMatrix::Rz(gmst, AngleFormat::Radians);
        let omega_earth = Vector3::new(0.0, 0.0, OMEGA_EARTH); // rad/s

        let r_pef: Vector3<f64> = R * Vector3::<f64>::from(tle_state.fixed_rows::<3>(0));
        let v_pef: Vector3<f64> =
            R * Vector3::<f64>::from(tle_state.fixed_rows::<3>(3)) - omega_earth.cross(&r_pef);

        Vector6::new(r_pef[0], r_pef[1], r_pef[2], v_pef[0], v_pef[1], v_pef[2])
    }

    /// Get Keplerian orbital elements from TLE data
    ///
    /// Extracts the Keplerian elements directly from the TLE lines used to initialize
    /// this propagator. This method uses the `keplerian_elements_from_tle` function
    /// to parse the TLE data.
    ///
    /// # Arguments
    /// * `angle_format` - Format for angular elements (degrees or radians)
    ///
    /// # Returns
    /// * `Result<Vector6<f64>, BraheError>` - Keplerian elements [a, e, i, Ω, ω, M]
    ///
    /// Elements are returned in the specified angle format:
    /// - a: semi-major axis [m]
    /// - e: eccentricity [dimensionless]
    /// - i: inclination [radians or degrees]
    /// - Ω: right ascension of ascending node [radians or degrees]
    /// - ω: argument of periapsis [radians or degrees]
    /// - M: mean anomaly [radians or degrees]
    pub fn get_elements(&self, angle_format: AngleFormat) -> Result<Vector6<f64>, BraheError> {
        use crate::orbits::keplerian_elements_from_tle;

        // Extract elements from TLE (returns in degrees)
        let (_epoch, mut elements) = keplerian_elements_from_tle(&self.line1, &self.line2)?;

        // Convert angular elements to radians if requested
        if angle_format == AngleFormat::Radians {
            elements[2] *= DEG2RAD; // inclination
            elements[3] *= DEG2RAD; // RAAN
            elements[4] *= DEG2RAD; // argument of periapsis
            elements[5] *= DEG2RAD; // mean anomaly
        }

        Ok(elements)
    }

    /// Get orbital elements at TLE epoch.
    ///
    /// Returns Keplerian orbital elements extracted from the TLE data at epoch.
    /// Angular elements are returned in degrees to match the native TLE format.
    ///
    /// # Returns
    ///
    /// * `elements` - Orbital elements vector [a, e, i, Ω, ω, M]
    ///   - a: semi-major axis. Units: (m)
    ///   - e: eccentricity (dimensionless)
    ///   - i: inclination. Units: (deg)
    ///   - Ω: right ascension of ascending node. Units: (deg)
    ///   - ω: argument of periapsis. Units: (deg)
    ///   - M: mean anomaly. Units: (deg)
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::SGPPropagator;
    ///
    /// brahe::initialize_eop().unwrap();
    ///
    /// let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    /// let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    ///
    /// let prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();
    /// let elements = prop.elements();
    ///
    /// // Elements are returned in degrees
    /// println!("Inclination: {:.4} deg", elements[2]);
    /// ```
    pub fn elements(&self) -> Vector6<f64> {
        self.get_elements(AngleFormat::Degrees)
            .expect("Failed to extract elements from TLE")
    }

    /// Get semi-major axis at TLE epoch.
    ///
    /// # Returns
    ///
    /// * Semi-major axis. Units: (m)
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::SGPPropagator;
    ///
    /// brahe::initialize_eop().unwrap();
    ///
    /// let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    /// let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    ///
    /// let prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();
    /// let sma = prop.semi_major_axis();
    ///
    /// println!("Semi-major axis: {:.3} m", sma);
    /// ```
    pub fn semi_major_axis(&self) -> f64 {
        self.elements()[0]
    }

    /// Get eccentricity at TLE epoch.
    ///
    /// # Returns
    ///
    /// * Eccentricity (dimensionless)
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::SGPPropagator;
    ///
    /// brahe::initialize_eop().unwrap();
    ///
    /// let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    /// let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    ///
    /// let prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();
    /// let ecc = prop.eccentricity();
    ///
    /// println!("Eccentricity: {:.6}", ecc);
    /// ```
    pub fn eccentricity(&self) -> f64 {
        self.elements()[1]
    }

    /// Get inclination at TLE epoch.
    ///
    /// # Returns
    ///
    /// * Inclination. Units: (deg)
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::SGPPropagator;
    ///
    /// brahe::initialize_eop().unwrap();
    ///
    /// let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    /// let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    ///
    /// let prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();
    /// let inc = prop.inclination();
    ///
    /// println!("Inclination: {:.4} deg", inc);
    /// ```
    pub fn inclination(&self) -> f64 {
        self.elements()[2]
    }

    /// Get right ascension of ascending node at TLE epoch.
    ///
    /// # Returns
    ///
    /// * Right ascension of ascending node (RAAN). Units: (deg)
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::SGPPropagator;
    ///
    /// brahe::initialize_eop().unwrap();
    ///
    /// let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    /// let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    ///
    /// let prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();
    /// let raan = prop.right_ascension();
    ///
    /// println!("RAAN: {:.4} deg", raan);
    /// ```
    pub fn right_ascension(&self) -> f64 {
        self.elements()[3]
    }

    /// Get argument of periapsis at TLE epoch.
    ///
    /// # Returns
    ///
    /// * Argument of periapsis. Units: (deg)
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::SGPPropagator;
    ///
    /// brahe::initialize_eop().unwrap();
    ///
    /// let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    /// let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    ///
    /// let prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();
    /// let argp = prop.arg_perigee();
    ///
    /// println!("Argument of periapsis: {:.4} deg", argp);
    /// ```
    pub fn arg_perigee(&self) -> f64 {
        self.elements()[4]
    }

    /// Get mean anomaly at TLE epoch.
    ///
    /// # Returns
    ///
    /// * Mean anomaly. Units: (deg)
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::SGPPropagator;
    ///
    /// brahe::initialize_eop().unwrap();
    ///
    /// let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    /// let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    ///
    /// let prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();
    /// let ma = prop.mean_anomaly();
    ///
    /// println!("Mean anomaly: {:.4} deg", ma);
    /// ```
    pub fn mean_anomaly(&self) -> f64 {
        self.elements()[5]
    }

    /// Get age of ephemeris data (time since TLE epoch).
    ///
    /// Returns the difference between the current system time and the TLE epoch.
    /// A positive value indicates the TLE is in the past; a negative value indicates
    /// a future TLE epoch.
    ///
    /// # Returns
    ///
    /// * Time since TLE epoch. Units: (s)
    ///
    /// # Examples
    /// ```
    /// use brahe::propagators::SGPPropagator;
    ///
    /// brahe::initialize_eop().unwrap();
    ///
    /// let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    /// let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    ///
    /// let prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();
    /// let age = prop.ephemeris_age();
    ///
    /// println!("Ephemeris age: {:.1} s", age);
    /// ```
    pub fn ephemeris_age(&self) -> f64 {
        Epoch::now() - self.epoch
    }

    // =========================================================================
    // Event Detection Internal Methods
    // =========================================================================

    /// Get ECI Cartesian state at any epoch (for event detection)
    ///
    /// This internal method always returns the state in ECI Cartesian format,
    /// regardless of the configured output format. This is used for event detection
    /// since all event detectors expect ECI Cartesian state.
    fn state_eci_cartesian(&self, epoch: Epoch) -> Vector6<f64> {
        let tle_state = self.propagate_internal(epoch);
        convert_state_from_spg4_frame(
            epoch,
            tle_state,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None,
        )
    }

    /// Scan all event detectors for events in the interval [epoch_prev, epoch_curr]
    ///
    /// Returns a vector of detected events sorted chronologically.
    fn scan_all_events(
        &self,
        epoch_prev: Epoch,
        epoch_curr: Epoch,
        state_prev: &DVector<f64>,
        state_curr: &DVector<f64>,
    ) -> Vec<DDetectedEvent> {
        let mut events = Vec::new();

        // Create state function for bisection - uses exact SGP4 propagation
        // (no interpolation needed since SGP4 is analytical)
        let state_fn =
            |epoch: Epoch| -> DVector<f64> { svec6_to_dvec(&self.state_eci_cartesian(epoch)) };

        // No parameters for SGP4
        let params: Option<&DVector<f64>> = None;

        // Scan each detector
        for (idx, detector) in self.event_detectors.iter().enumerate() {
            // Skip detectors that have been processed (one-shot events)
            if detector.is_processed() {
                continue;
            }

            if let Some(event) = dscan_for_event(
                detector.as_ref(),
                idx,
                &state_fn,
                epoch_prev,
                epoch_curr,
                state_prev,
                state_curr,
                params,
            ) {
                events.push(event);
            }
        }

        // Sort events chronologically
        events.sort_by(|a, b| a.window_open.partial_cmp(&b.window_open).unwrap());

        events
    }

    // =========================================================================
    // Event Detection API
    // =========================================================================

    /// Add an event detector to monitor during propagation
    ///
    /// Events are detected during each propagation step and processed according
    /// to their action (Continue or Stop). Note that for SGP4 propagation,
    /// callback state mutations are ignored since SGP4 is an analytical propagator.
    ///
    /// # Arguments
    /// * `detector` - Event detector implementing the `DEventDetector` trait
    ///
    /// # Example
    /// ```
    /// use brahe::propagators::SGPPropagator;
    /// use brahe::propagators::traits::SStatePropagator;  // for initial_epoch()
    /// use brahe::events::DTimeEvent;
    ///
    /// brahe::initialize_eop().unwrap();
    ///
    /// let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    /// let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
    ///
    /// let mut prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();
    /// let epoch = prop.initial_epoch();
    ///
    /// // Add a time event detector
    /// let detector = DTimeEvent::new(epoch + 300.0, "5 minute mark");
    /// prop.add_event_detector(Box::new(detector));
    /// ```
    pub fn add_event_detector(&mut self, detector: Box<dyn DEventDetector>) {
        self.event_detectors.push(detector);
    }

    /// Take ownership of all event detectors, leaving the propagator with none.
    ///
    /// This is useful for transferring event detectors to/from cloned propagators,
    /// since event detectors (trait objects) cannot be cloned.
    ///
    /// # Returns
    ///
    /// The event detectors that were attached to this propagator.
    pub fn take_event_detectors(&mut self) -> Vec<Box<dyn DEventDetector>> {
        std::mem::take(&mut self.event_detectors)
    }

    /// Set event detectors, replacing any existing detectors.
    ///
    /// # Arguments
    ///
    /// * `detectors` - The event detectors to attach to this propagator.
    pub fn set_event_detectors(&mut self, detectors: Vec<Box<dyn DEventDetector>>) {
        self.event_detectors = detectors;
    }

    /// Take ownership of the event log, leaving an empty log.
    ///
    /// This is useful for transferring detected events from cloned propagators.
    ///
    /// # Returns
    ///
    /// The detected events from this propagator.
    pub fn take_event_log(&mut self) -> Vec<DDetectedEvent> {
        std::mem::take(&mut self.event_log)
    }

    /// Set the event log, replacing any existing log.
    ///
    /// # Arguments
    ///
    /// * `log` - The event log to set.
    pub fn set_event_log(&mut self, log: Vec<DDetectedEvent>) {
        self.event_log = log;
    }

    /// Set the terminated flag.
    ///
    /// # Arguments
    ///
    /// * `terminated` - Whether propagation has been terminated by an event.
    pub fn set_terminated(&mut self, terminated: bool) {
        self.terminated = terminated;
    }

    /// Get all detected events
    ///
    /// Returns a slice of all events that have been detected during propagation.
    pub fn event_log(&self) -> &[DDetectedEvent] {
        &self.event_log
    }

    /// Get events by name (substring match)
    ///
    /// Returns all events whose name contains the specified substring.
    ///
    /// # Arguments
    /// * `name` - Substring to search for in event names
    pub fn events_by_name(&self, name: &str) -> Vec<&DDetectedEvent> {
        self.event_log
            .iter()
            .filter(|e| e.name.contains(name))
            .collect()
    }

    /// Get the most recently detected event
    ///
    /// Returns `None` if no events have been detected.
    pub fn latest_event(&self) -> Option<&DDetectedEvent> {
        self.event_log.last()
    }

    /// Get events in a time range
    ///
    /// Returns all events that occurred between the start and end epochs (inclusive).
    ///
    /// # Arguments
    /// * `start` - Start of time range
    /// * `end` - End of time range
    pub fn events_in_range(&self, start: Epoch, end: Epoch) -> Vec<&DDetectedEvent> {
        self.event_log
            .iter()
            .filter(|e| e.window_open >= start && e.window_open <= end)
            .collect()
    }

    /// Query events with flexible filtering
    ///
    /// Returns an EventQuery that supports chainable filters and
    /// standard iterator methods.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use brahe::propagators::SGPPropagator;
    /// use brahe::time::Epoch;
    ///
    /// // Get events from detector 1 in time range
    /// let events: Vec<_> = prop.query_events()
    ///     .by_detector_index(1)
    ///     .in_time_range(epoch, epoch + 3600.0)
    ///     .collect();
    ///
    /// // Count altitude events
    /// let count = prop.query_events()
    ///     .by_name_contains("Altitude")
    ///     .count();
    /// ```
    pub fn query_events(&self) -> EventQuery<'_, std::slice::Iter<'_, DDetectedEvent>> {
        EventQuery::new(self.event_log.iter())
    }

    /// Get events by detector index
    ///
    /// Returns all events detected by the specified detector.
    ///
    /// # Arguments
    /// * `index` - Detector index (0-based, order detectors were added)
    pub fn events_by_detector_index(&self, index: usize) -> Vec<&DDetectedEvent> {
        self.event_log
            .iter()
            .filter(|e| e.detector_index == index)
            .collect()
    }

    /// Get events by detector index within time range
    ///
    /// Returns events from the specified detector that occurred in the time range.
    ///
    /// # Arguments
    /// * `index` - Detector index (0-based)
    /// * `start` - Start of time range (inclusive)
    /// * `end` - End of time range (inclusive)
    pub fn events_by_detector_index_in_range(
        &self,
        index: usize,
        start: Epoch,
        end: Epoch,
    ) -> Vec<&DDetectedEvent> {
        self.event_log
            .iter()
            .filter(|e| e.detector_index == index && e.window_open >= start && e.window_open <= end)
            .collect()
    }

    /// Get events by name within time range
    ///
    /// Returns events matching name (substring) that occurred in the time range.
    ///
    /// # Arguments
    /// * `name` - Substring to search for in event names
    /// * `start` - Start of time range (inclusive)
    /// * `end` - End of time range (inclusive)
    pub fn events_by_name_in_range(
        &self,
        name: &str,
        start: Epoch,
        end: Epoch,
    ) -> Vec<&DDetectedEvent> {
        self.event_log
            .iter()
            .filter(|e| e.name.contains(name) && e.window_open >= start && e.window_open <= end)
            .collect()
    }

    /// Clear the event log
    ///
    /// Removes all detected events from the log. Event detectors are not removed.
    pub fn clear_events(&mut self) {
        self.event_log.clear();
    }

    /// Reset the termination flag
    ///
    /// Allows propagation to continue after a terminal event. The event log
    /// is not cleared.
    pub fn reset_termination(&mut self) {
        self.terminated = false;
    }

    /// Check if propagation was terminated by an event
    pub fn is_terminated(&self) -> bool {
        self.terminated
    }
}

impl SStatePropagator for SGPPropagator {
    fn step_by(&mut self, step_size: f64) {
        // Check termination flag - stop immediately if already terminated
        if self.terminated {
            return;
        }

        let current_epoch = self.current_epoch();
        let target_epoch = current_epoch + step_size; // step_size is in seconds

        // Get ECI Cartesian states as DVector for event detection
        let state_prev_eci = svec6_to_dvec(&self.state_eci_cartesian(current_epoch));
        let state_curr_eci = svec6_to_dvec(&self.state_eci_cartesian(target_epoch));

        // Scan for events if we have any detectors
        if !self.event_detectors.is_empty() {
            let detected_events = self.scan_all_events(
                current_epoch,
                target_epoch,
                &state_prev_eci,
                &state_curr_eci,
            );

            // Process each event
            for event in detected_events {
                // Determine the action to take
                let action = if let Some(detector) = self.event_detectors.get(event.detector_index)
                {
                    if let Some(callback) = detector.callback() {
                        // Execute callback - ignore state/param mutations (SGP4 is analytical)
                        let (_, _, callback_action) =
                            callback(event.window_open, &event.entry_state, None);
                        callback_action
                    } else {
                        detector.action()
                    }
                } else {
                    event.action
                };

                // Mark detector as processed (for one-shot events like STimeEvent)
                if let Some(detector) = self.event_detectors.get(event.detector_index) {
                    detector.mark_processed();
                }

                // Log the event with the determined action
                let mut logged_event = event.clone();
                logged_event.action = action;
                self.event_log.push(logged_event);

                // Add event state to trajectory at exact event time (in user's output format)
                let event_state_output = convert_state_from_spg4_frame(
                    event.window_open,
                    self.propagate_internal(event.window_open),
                    self.frame,
                    self.representation,
                    self.angle_format,
                );
                self.trajectory
                    .add(event.window_open, svec6_to_dvec(&event_state_output));

                // Check for terminal action
                if action == EventAction::Stop {
                    self.terminated = true;
                    return; // Stop propagation, don't add target state
                }
            }
        }

        // If not terminated, add the target state to trajectory
        let tle_state = self.propagate_internal(target_epoch);
        let new_state = convert_state_from_spg4_frame(
            target_epoch,
            tle_state,
            self.frame,
            self.representation,
            self.angle_format,
        );
        self.trajectory.add(target_epoch, svec6_to_dvec(&new_state))
    }

    // Default implementation from trait is used for:
    // - step()
    // - step_past()
    // - propagate_steps()
    // - propagate_to()

    fn initial_epoch(&self) -> Epoch {
        self.epoch
    }

    fn initial_state(&self) -> Vector6<f64> {
        self.initial_state
    }

    fn current_epoch(&self) -> Epoch {
        self.trajectory.last().unwrap().0
    }

    fn current_state(&self) -> Vector6<f64> {
        let (_, dstate) = self.trajectory.last().unwrap();
        dvec_to_svec6(&dstate)
    }

    fn step_size(&self) -> f64 {
        self.step_size
    }

    fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    fn reset(&mut self) {
        self.trajectory.clear();
        self.trajectory
            .add(self.epoch, svec6_to_dvec(&self.initial_state));

        // Clear event detection state
        self.event_log.clear();
        self.terminated = false;

        // Reset processed state on all detectors (for one-shot events)
        for detector in &self.event_detectors {
            detector.reset_processed();
        }
    }

    fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_size(max_size)
    }

    fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_age(max_age)
    }

    fn propagate_to(&mut self, target_epoch: Epoch) {
        let step = self.step_size();

        if step >= 0.0 {
            // Forward propagation - only proceed if target is in the future
            if target_epoch <= self.current_epoch() {
                return;
            }
            while !self.terminated && self.current_epoch() < target_epoch {
                let remaining_time = target_epoch - self.current_epoch();
                let step_size = remaining_time.min(step);

                // Guard against very small steps to avoid infinite loops
                if step_size <= 1e-9 {
                    break;
                }

                self.step_by(step_size);
            }
        } else {
            // Backward propagation - only proceed if target is in the past
            if target_epoch >= self.current_epoch() {
                return;
            }
            while !self.terminated && self.current_epoch() > target_epoch {
                let remaining_time = self.current_epoch() - target_epoch;
                let step_size = -(remaining_time.min(step.abs()));

                // Guard against very small steps to avoid infinite loops
                if step_size.abs() <= 1e-9 {
                    break;
                }

                self.step_by(step_size);
            }
        }
    }
}

impl SStateProvider for SGPPropagator {
    fn state(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        Ok(self.propagate_internal(epoch))
    }
}

impl SOrbitStateProvider for SGPPropagator {
    fn state_eci(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        let state_ecef = self.state_ecef(epoch)?;

        // Step 3: ECEF to ECI
        Ok(state_ecef_to_eci(epoch, state_ecef))
    }

    fn state_ecef(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.state_itrf(epoch)
    }

    fn state_itrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        let state_pef = self.state_pef(epoch);

        // Step 2: PEF to ECEF
        #[allow(non_snake_case)]
        let PM = polar_motion(epoch);

        let r_itrf = PM * Vector3::<f64>::from(state_pef.fixed_rows::<3>(0));
        let v_itrf = PM * Vector3::<f64>::from(state_pef.fixed_rows::<3>(3));

        Ok(Vector6::new(
            r_itrf[0], r_itrf[1], r_itrf[2], v_itrf[0], v_itrf[1], v_itrf[2],
        ))
    }

    fn state_gcrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        // Get ECEF state and convert to GCRF/ECI
        let state_itrf = self.state_itrf(epoch)?;
        Ok(state_itrf_to_gcrf(epoch, state_itrf))
    }

    fn state_eme2000(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        // Get GCRF state and convert to EME2000
        let gcrf_state = self.state_gcrf(epoch)?;
        Ok(state_gcrf_to_eme2000(gcrf_state))
    }

    fn state_koe_osc(
        &self,
        epoch: Epoch,
        angle_format: AngleFormat,
    ) -> Result<Vector6<f64>, BraheError> {
        let state_eci = self.state_eci(epoch)?;

        Ok(state_eci_to_koe(state_eci, angle_format))
    }

    // Default implementations from trait are used for:
    // - states()
    // - states_eci()
    // - states_ecef()
    // - states_koe()
}

// Implement DStateProvider for SGPPropagator
impl crate::utils::DStateProvider for SGPPropagator {
    fn state(&self, epoch: Epoch) -> Result<nalgebra::DVector<f64>, BraheError> {
        let state_vec6 = <Self as SStateProvider>::state(self, epoch)?;
        Ok(nalgebra::DVector::from_column_slice(state_vec6.as_slice()))
    }

    fn state_dim(&self) -> usize {
        6
    }

    fn states(&self, epochs: &[Epoch]) -> Result<Vec<nalgebra::DVector<f64>>, BraheError> {
        epochs
            .iter()
            .map(|&epoch| <Self as crate::utils::DStateProvider>::state(self, epoch))
            .collect()
    }
}

// Implement DOrbitStateProvider for SGPPropagator
impl crate::utils::DOrbitStateProvider for SGPPropagator {
    fn state_eci(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        <Self as SOrbitStateProvider>::state_eci(self, epoch)
    }

    fn state_ecef(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        <Self as SOrbitStateProvider>::state_ecef(self, epoch)
    }

    fn state_itrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        <Self as SOrbitStateProvider>::state_itrf(self, epoch)
    }

    fn state_gcrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        <Self as SOrbitStateProvider>::state_gcrf(self, epoch)
    }

    fn state_eme2000(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        <Self as SOrbitStateProvider>::state_eme2000(self, epoch)
    }

    fn state_koe_osc(
        &self,
        epoch: Epoch,
        angle_format: AngleFormat,
    ) -> Result<Vector6<f64>, BraheError> {
        <Self as SOrbitStateProvider>::state_koe_osc(self, epoch, angle_format)
    }

    // Default batch implementations from trait are used for:
    // - states_eci()
    // - states_ecef()
    // - states_koe()
}

impl Identifiable for SGPPropagator {
    fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self.trajectory = self.trajectory.with_name(name);
        self
    }

    fn with_uuid(mut self, uuid: uuid::Uuid) -> Self {
        self.uuid = Some(uuid);
        self
    }

    fn with_new_uuid(mut self) -> Self {
        self.uuid = Some(uuid::Uuid::new_v4());
        self
    }

    fn with_id(mut self, id: u64) -> Self {
        self.id = Some(id);
        self.trajectory = self.trajectory.with_id(id);
        self
    }

    fn with_identity(
        mut self,
        name: Option<&str>,
        uuid: Option<uuid::Uuid>,
        id: Option<u64>,
    ) -> Self {
        self.name = name.map(|s| s.to_string());
        self.uuid = uuid;
        self.id = id;
        self.trajectory = self.trajectory.with_identity(name, uuid, id);
        self
    }

    fn set_identity(&mut self, name: Option<&str>, uuid: Option<uuid::Uuid>, id: Option<u64>) {
        self.name = name.map(|s| s.to_string());
        self.uuid = uuid;
        self.id = id;
        self.trajectory.set_identity(name, uuid, id);
    }

    fn set_id(&mut self, id: Option<u64>) {
        self.id = id;
        self.trajectory.set_id(id);
    }

    fn set_name(&mut self, name: Option<&str>) {
        self.name = name.map(|s| s.to_string());
        self.trajectory.set_name(name);
    }

    fn generate_uuid(&mut self) {
        self.uuid = Some(uuid::Uuid::new_v4());
        self.trajectory.generate_uuid();
    }

    fn get_id(&self) -> Option<u64> {
        self.id
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_uuid(&self) -> Option<uuid::Uuid> {
        self.uuid
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::RADIANS;
    use crate::utils::testing::{setup_global_test_eop, setup_global_test_eop_original_brahe};
    use approx::assert_abs_diff_eq;

    // Test TLE data
    const ISS_LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    const ISS_LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    // SGPPropagator Method Tests

    #[test]
    fn test_sgppropagator_from_tle() {
        setup_global_test_eop();
        let propagator = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0);
        assert!(propagator.is_ok());

        let prop = propagator.unwrap();
        assert_eq!(prop.step_size, 60.0);
        assert_eq!(prop.line1, ISS_LINE1);
        assert_eq!(prop.line2, ISS_LINE2);
    }

    #[test]
    fn test_sgppropagator_from_3le() {
        setup_global_test_eop();
        let name = "ISS (ZARYA)";
        let propagator = SGPPropagator::from_3le(Some(name), ISS_LINE1, ISS_LINE2, 60.0);
        assert!(propagator.is_ok());

        let prop = propagator.unwrap();
        assert_eq!(prop.satellite_name, Some(name.to_string()));

        // Verify identity fields are automatically set
        assert_eq!(prop.get_name(), Some(name));
        assert_eq!(prop.get_id(), Some(25544));
    }

    #[test]
    fn test_sgppropagator_from_omm_elements() {
        setup_global_test_eop();

        // ISS OMM data
        let propagator = SGPPropagator::from_omm_elements(
            "2025-11-29T20:01:44.058144",
            15.49193835, // mean_motion (rev/day)
            0.0003723,   // eccentricity
            51.6312,     // inclination (degrees)
            206.3646,    // raan (degrees)
            184.1118,    // arg_of_pericenter (degrees)
            175.9840,    // mean_anomaly (degrees)
            25544,       // norad_id
            60.0,        // step_size
            Some("ISS (ZARYA)"),
            Some("1998-067A"),
            Some('U'),
            Some(0.15237e-3),
            Some(0.801e-4),
            Some(0.0),
            Some(0),
            Some(999),
            Some(54085),
        );

        assert!(
            propagator.is_ok(),
            "Failed to create propagator: {:?}",
            propagator.as_ref().err()
        );
        let prop = propagator.unwrap();

        // Verify basic properties
        assert_eq!(prop.norad_id, 25544);
        assert_eq!(prop.step_size, 60.0);
        assert_eq!(prop.satellite_name, Some("ISS (ZARYA)".to_string()));
        assert_eq!(prop.epoch.year(), 2025);
        assert_eq!(prop.epoch.month(), 11);
        assert_eq!(prop.epoch.day(), 29);

        // Verify orbital elements (all angles returned in degrees)
        assert_abs_diff_eq!(prop.eccentricity(), 0.0003723, epsilon = 1e-7);
        assert_abs_diff_eq!(prop.inclination(), 51.6312, epsilon = 1e-4);
        assert_abs_diff_eq!(prop.right_ascension(), 206.3646, epsilon = 1e-4);
        assert_abs_diff_eq!(prop.arg_perigee(), 184.1118, epsilon = 1e-4);
        assert_abs_diff_eq!(prop.mean_anomaly(), 175.9840, epsilon = 1e-4);
    }

    #[test]
    fn test_sgppropagator_from_omm_elements_minimal() {
        setup_global_test_eop();

        // Test with minimal required parameters (all optionals as None)
        let propagator = SGPPropagator::from_omm_elements(
            "2025-11-29T20:01:44.058144",
            15.49193835,
            0.0003723,
            51.6312,
            206.3646,
            184.1118,
            175.9840,
            25544,
            60.0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        assert!(propagator.is_ok());
        let prop = propagator.unwrap();
        assert_eq!(prop.norad_id, 25544);
        assert_eq!(prop.satellite_name, None);
    }

    #[test]
    fn test_sgppropagator_from_omm_elements_propagation() {
        setup_global_test_eop();

        let mut prop = SGPPropagator::from_omm_elements(
            "2025-11-29T20:01:44.058144",
            15.49193835,
            0.0003723,
            51.6312,
            206.3646,
            184.1118,
            175.9840,
            25544,
            60.0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Verify initial state is valid
        let initial_state = prop.current_state();
        assert!(initial_state.iter().all(|x| x.is_finite()));

        // Propagate forward
        prop.step();
        let new_state = prop.current_state();
        assert!(new_state.iter().all(|x| x.is_finite()));
        assert_ne!(new_state, initial_state);
    }

    #[test]
    fn test_sgppropagator_from_omm_elements_invalid_epoch() {
        setup_global_test_eop();

        let propagator = SGPPropagator::from_omm_elements(
            "not-a-valid-date",
            15.49193835,
            0.0003723,
            51.6312,
            206.3646,
            184.1118,
            175.9840,
            25544,
            60.0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        );

        assert!(propagator.is_err());
        let err = propagator.unwrap_err();
        assert!(err.to_string().contains("Invalid epoch format"));
    }

    #[test]
    fn test_sgppropagator_from_tle_sets_id_without_name() {
        setup_global_test_eop();
        // Test that from_tle (2-line TLE without name) still sets ID from NORAD catalog number
        let propagator = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0);
        assert!(propagator.is_ok());

        let prop = propagator.unwrap();

        // Verify no name is set
        assert_eq!(prop.get_name(), None);

        // Verify ID is set from NORAD catalog number (25544)
        assert_eq!(prop.get_id(), Some(25544));

        // Verify trajectory also has the ID
        assert_eq!(prop.trajectory.get_id(), Some(25544));
        assert_eq!(prop.trajectory.get_name(), None);
    }

    // OrbitPropagator Trait Tests

    #[test]
    fn test_sgppropagator_orbitpropagator_step() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.current_epoch();

        prop.step();
        let new_epoch = prop.current_epoch();

        assert_abs_diff_eq!(new_epoch - initial_epoch, 60.0, epsilon = 0.1);
        assert_eq!(prop.trajectory.len(), 2); // Initial + 1 step

        // State should have changed after propagation
        let new_state = prop.current_state();
        assert_ne!(new_state, prop.initial_state);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_step_by() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.current_epoch();

        prop.step_by(120.0);
        let new_epoch = prop.current_epoch();

        assert_abs_diff_eq!(new_epoch - initial_epoch, 120.0, epsilon = 0.1);

        // Confirm only 2 states in trajectory (initial + 1 step)
        assert_eq!(prop.trajectory.len(), 2);

        // State should have changed after propagation
        let new_state = prop.current_state();
        assert_ne!(new_state, prop.initial_state);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_propagate_steps() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.current_epoch();

        prop.propagate_steps(5);
        let new_epoch = prop.current_epoch();

        assert_abs_diff_eq!(new_epoch - initial_epoch, 300.0, epsilon = 0.1);
        assert_eq!(prop.trajectory.len(), 6); // Initial + 5 steps

        // State should have changed after propagation
        let new_state = prop.current_state();
        assert_ne!(new_state, prop.initial_state);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_step_past() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.initial_epoch();

        let target_epoch = initial_epoch + 250.0;
        prop.step_past(target_epoch);

        let current_epoch = prop.current_epoch();
        assert!(current_epoch > target_epoch);

        // Should have 6 steps: initial + 5 steps of 60s
        assert_eq!(prop.trajectory.len(), 6);
        assert_abs_diff_eq!(current_epoch - initial_epoch, 300.0, epsilon = 0.1);

        // State should have changed after propagation
        let new_state = prop.current_state();
        assert_ne!(new_state, prop.initial_state);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_propagate_to() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.initial_epoch();
        let target_epoch = initial_epoch + 90.0; // 90 seconds forward

        prop.propagate_to(target_epoch);
        let current_epoch = prop.current_epoch();

        assert_eq!(current_epoch, target_epoch);

        // Should have 3 steps: initial + 1 step of 60s + 1 step of 30s
        assert_eq!(prop.trajectory.len(), 3);

        // State should have changed after propagation
        let new_state = prop.current_state();
        assert_ne!(new_state, prop.initial_state);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_current_state() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let state = prop.current_state();

        // State should be non-zero for valid TLE
        assert!(state.norm() > 0.0);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_current_epoch() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.current_epoch();

        // Epoch should match TLE epoch
        assert_eq!(epoch, prop.initial_epoch());
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_initial_state() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let state = prop.initial_state();

        // State should be non-zero
        assert!(state.norm() > 0.0);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_initial_epoch() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Should be around 2008-09-20 based on TLE epoch
        assert!(epoch.jd() > 2454700.0 && epoch.jd() < 2454800.0);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_step_size() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        assert_eq!(prop.step_size(), 60.0);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_set_step_size() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        prop.set_step_size(120.0);
        assert_eq!(prop.step_size(), 120.0);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_reset() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        // Propagate forward
        prop.propagate_steps(5);
        assert_eq!(prop.trajectory.len(), 6);

        // Reset
        prop.reset();
        assert_eq!(prop.trajectory.len(), 1);
        assert_eq!(prop.current_epoch(), prop.initial_epoch());
    }

    #[test]
    fn test_sgppropagator_statepropagator_set_eviction_policy_max_size() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        prop.set_eviction_policy_max_size(5).unwrap();

        // Propagate 10 steps
        prop.propagate_steps(10);

        // Should only keep 5 states
        assert_eq!(prop.trajectory.len(), 5);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_set_eviction_policy_max_age() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        // Set eviction policy - only keep states within 120 seconds of current
        let result = prop.set_eviction_policy_max_age(120.0);
        assert!(result.is_ok());

        // Propagate several steps (10 * 60s = 600s total)
        prop.propagate_steps(10);

        // Should have evicted old states - should keep only last ~3 states (120s / 60s step)
        // Plus current state: 3 previous + current = 4 states max
        assert!(prop.trajectory.len() <= 4);
        assert!(prop.trajectory.len() > 0);
    }

    #[test]
    fn test_sgppropagator_get_elements_radians() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        let elements = prop.get_elements(RADIANS).unwrap();

        // Expected values from ISS TLE
        assert_abs_diff_eq!(elements[0], 6730960.676936833, epsilon = 1.0); // a [m]
        assert_abs_diff_eq!(elements[1], 0.0006703, epsilon = 1e-10); // e
        assert_abs_diff_eq!(elements[2], 0.9013159509979036, epsilon = 1e-10); // i [rad]
        assert_abs_diff_eq!(elements[3], 4.319038890874972, epsilon = 1e-10); // raan [rad]
        assert_abs_diff_eq!(elements[4], 2.278282992383318, epsilon = 1e-10); // argp [rad]
        assert_abs_diff_eq!(elements[5], 5.672822723806145, epsilon = 1e-10); // M [rad]
    }

    #[test]
    fn test_sgppropagator_get_elements_degrees() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        let elements = prop.get_elements(AngleFormat::Degrees).unwrap();

        // Expected values from ISS TLE
        assert_abs_diff_eq!(elements[0], 6730960.676936833, epsilon = 1.0); // a [m]
        assert_abs_diff_eq!(elements[1], 0.0006703, epsilon = 1e-10); // e
        assert_abs_diff_eq!(elements[2], 51.6416, epsilon = 1e-10); // i [deg]
        assert_abs_diff_eq!(elements[3], 247.4627, epsilon = 1e-10); // raan [deg]
        assert_abs_diff_eq!(elements[4], 130.5360, epsilon = 1e-10); // argp [deg]
        assert_abs_diff_eq!(elements[5], 325.0288, epsilon = 1e-10); // M [deg]
    }

    #[test]
    fn test_sgppropagator_elements() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        let elements = prop.elements();

        // Should return same values as get_elements with Degrees
        assert_abs_diff_eq!(elements[0], 6730960.676936833, epsilon = 1.0); // a [m]
        assert_abs_diff_eq!(elements[1], 0.0006703, epsilon = 1e-10); // e
        assert_abs_diff_eq!(elements[2], 51.6416, epsilon = 1e-10); // i [deg]
        assert_abs_diff_eq!(elements[3], 247.4627, epsilon = 1e-10); // raan [deg]
        assert_abs_diff_eq!(elements[4], 130.5360, epsilon = 1e-10); // argp [deg]
        assert_abs_diff_eq!(elements[5], 325.0288, epsilon = 1e-10); // M [deg]
    }

    #[test]
    fn test_sgppropagator_semi_major_axis() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        let sma = prop.semi_major_axis();

        assert_abs_diff_eq!(sma, 6730960.676936833, epsilon = 1.0);
    }

    #[test]
    fn test_sgppropagator_eccentricity() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        let ecc = prop.eccentricity();

        assert_abs_diff_eq!(ecc, 0.0006703, epsilon = 1e-10);
    }

    #[test]
    fn test_sgppropagator_inclination() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        let inc = prop.inclination();

        // Should return degrees
        assert_abs_diff_eq!(inc, 51.6416, epsilon = 1e-10);
    }

    #[test]
    fn test_sgppropagator_right_ascension() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        let raan = prop.right_ascension();

        // Should return degrees
        assert_abs_diff_eq!(raan, 247.4627, epsilon = 1e-10);
    }

    #[test]
    fn test_sgppropagator_arg_perigee() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        let argp = prop.arg_perigee();

        // Should return degrees
        assert_abs_diff_eq!(argp, 130.5360, epsilon = 1e-10);
    }

    #[test]
    fn test_sgppropagator_mean_anomaly() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        let ma = prop.mean_anomaly();

        // Should return degrees
        assert_abs_diff_eq!(ma, 325.0288, epsilon = 1e-10);
    }

    #[test]
    fn test_sgppropagator_ephemeris_age() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        let age = prop.ephemeris_age();

        // TLE epoch is 2008-09-20, so age should be positive and large (years worth of seconds)
        assert!(age > 0.0);
        // Should be at least 15 years worth of seconds (from 2008 to 2023+)
        assert!(age > 15.0 * 365.25 * 86400.0);
    }

    // Identifiable Trait Tests

    #[test]
    fn test_sgppropagator_identifiable_with_name() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0)
            .unwrap()
            .with_name("My Satellite");

        assert_eq!(prop.get_name(), Some("My Satellite"));
        // ID should be set from NORAD catalog number (25544), even when created from 2-line TLE
        assert_eq!(prop.get_id(), Some(25544));
        assert_eq!(prop.get_uuid(), None);
    }

    #[test]
    fn test_sgppropagator_identifiable_with_id() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0)
            .unwrap()
            .with_id(12345);

        assert_eq!(prop.get_id(), Some(12345));
        assert_eq!(prop.get_name(), None);
        assert_eq!(prop.get_uuid(), None);
    }

    #[test]
    fn test_sgppropagator_identifiable_with_uuid() {
        setup_global_test_eop();
        let test_uuid = uuid::Uuid::new_v4();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0)
            .unwrap()
            .with_uuid(test_uuid);

        assert_eq!(prop.get_uuid(), Some(test_uuid));
        assert_eq!(prop.get_name(), None);
        // ID should be set from NORAD catalog number (25544), even when created from 2-line TLE
        assert_eq!(prop.get_id(), Some(25544));
    }

    #[test]
    fn test_sgppropagator_identifiable_with_new_uuid() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0)
            .unwrap()
            .with_new_uuid();

        assert!(prop.get_uuid().is_some());
        assert_eq!(prop.get_name(), None);
        // ID should be set from NORAD catalog number (25544), even when created from 2-line TLE
        assert_eq!(prop.get_id(), Some(25544));
    }

    #[test]
    fn test_sgppropagator_identifiable_with_identity() {
        setup_global_test_eop();
        let test_uuid = uuid::Uuid::new_v4();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0)
            .unwrap()
            .with_identity(Some("Satellite A"), Some(test_uuid), Some(999));

        assert_eq!(prop.get_name(), Some("Satellite A"));
        assert_eq!(prop.get_id(), Some(999));
        assert_eq!(prop.get_uuid(), Some(test_uuid));
    }

    #[test]
    fn test_sgppropagator_identifiable_set_name() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        prop.set_name(Some("Test Name"));
        assert_eq!(prop.get_name(), Some("Test Name"));

        prop.set_name(None);
        assert_eq!(prop.get_name(), None);
    }

    #[test]
    fn test_sgppropagator_identifiable_set_id() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        prop.set_id(Some(42));
        assert_eq!(prop.get_id(), Some(42));

        prop.set_id(None);
        assert_eq!(prop.get_id(), None);
    }

    #[test]
    fn test_sgppropagator_identifiable_generate_uuid() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        assert_eq!(prop.get_uuid(), None);

        prop.generate_uuid();
        let uuid1 = prop.get_uuid();
        assert!(uuid1.is_some());

        // Generate another UUID and verify it's different
        prop.generate_uuid();
        let uuid2 = prop.get_uuid();
        assert!(uuid2.is_some());
        assert_ne!(uuid1, uuid2);
    }

    #[test]
    fn test_sgppropagator_identifiable_set_identity() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let test_uuid = uuid::Uuid::new_v4();

        prop.set_identity(Some("Updated Name"), Some(test_uuid), Some(777));

        assert_eq!(prop.get_name(), Some("Updated Name"));
        assert_eq!(prop.get_id(), Some(777));
        assert_eq!(prop.get_uuid(), Some(test_uuid));

        // Clear all
        prop.set_identity(None, None, None);
        assert_eq!(prop.get_name(), None);
        assert_eq!(prop.get_id(), None);
        assert_eq!(prop.get_uuid(), None);
    }

    #[test]
    fn test_sgppropagator_identifiable_chaining() {
        setup_global_test_eop();
        let test_uuid = uuid::Uuid::new_v4();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0)
            .unwrap()
            .with_name("Chained Satellite")
            .with_id(123)
            .with_uuid(test_uuid);

        assert_eq!(prop.get_name(), Some("Chained Satellite"));
        assert_eq!(prop.get_id(), Some(123));
        assert_eq!(prop.get_uuid(), Some(test_uuid));
    }

    // StateProvider Trait Tests

    #[test]
    fn test_sgppropagator_analyticpropagator_state_koe_osc() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        let elements = prop.state_koe_osc(epoch, RADIANS).unwrap();

        // Verify we got keplerian elements (all finite)
        assert!(elements.iter().all(|&x| x.is_finite()));

        // Semi-major axis should be positive
        assert!(elements[0] > 0.0);

        // Eccentricity should be non-negative
        assert!(elements[1] >= 0.0);

        // Inclination should be around 51.6 degrees (in radians)
        assert_abs_diff_eq!(elements[2], 51.6_f64.to_radians(), epsilon = 0.1);
    }

    #[test]
    fn test_sgppropagator_analyticpropagator_states() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.initial_epoch();

        let epochs = vec![initial_epoch, initial_epoch + 0.01, initial_epoch + 0.02];

        let states = prop.states(&epochs).unwrap();
        assert_eq!(states.len(), 3);
    }

    #[test]
    fn test_sgppropagator_analyticpropagator_states_eci() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.initial_epoch();

        let epochs = vec![initial_epoch, initial_epoch + 0.01];

        let states = prop.states_eci(&epochs).unwrap();
        assert_eq!(states.len(), 2);
        // Verify states are valid Cartesian vectors
        for state in &states {
            assert!(state.norm() > 0.0);
        }
    }

    #[test]
    fn test_sgppropagator_analyticpropagator_states_ecef() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.initial_epoch();

        let epochs = vec![initial_epoch, initial_epoch + 0.01];

        let states = prop.states_ecef(&epochs).unwrap();
        assert_eq!(states.len(), 2);
        // Verify states are valid Cartesian vectors
        for state in &states {
            assert!(state.norm() > 0.0);
        }
    }

    #[test]
    fn test_sgppropagator_analyticpropagator_states_koe() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.initial_epoch();

        let epochs = vec![initial_epoch, initial_epoch + 0.01];

        let elements = prop.states_koe_osc(&epochs, RADIANS).unwrap();
        assert_eq!(elements.len(), 2);
        // Verify elements are valid Keplerian elements
        for elem in &elements {
            assert!(elem[0] > 0.0); // Semi-major axis positive
            assert!(elem[1] >= 0.0); // Eccentricity non-negative
        }
    }

    // State Output Tests - From Older Brahe Versions (for validation)

    #[test]
    fn test_sgppropagator_state_teme() {
        setup_global_test_eop_original_brahe();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // State in TEME frame (native SGP4 output)
        let state = prop.state(epoch).unwrap();

        assert_eq!(state.len(), 6);
        // TEME is the native SGP4 output frame
        assert_abs_diff_eq!(state[0], 4083909.8260273533, epsilon = 1e-8);
        assert_abs_diff_eq!(state[1], -993636.8325621719, epsilon = 1e-8);
        assert_abs_diff_eq!(state[2], 5243614.536966579, epsilon = 1e-8);
        assert_abs_diff_eq!(state[3], 2512.831950943635, epsilon = 1e-8);
        assert_abs_diff_eq!(state[4], 7259.8698423432315, epsilon = 1e-8);
        assert_abs_diff_eq!(state[5], -583.775727402632, epsilon = 1e-8);
    }

    #[test]
    fn test_tle_gmst82() {
        setup_global_test_eop_original_brahe();
        let epoch = epoch_from_tle(ISS_LINE1).unwrap();
        let gmst = tle_gmst82(epoch, AngleFormat::Radians);
        assert_abs_diff_eq!(gmst, 3.2494565064865406, epsilon = 1e-6);
    }

    #[test]
    fn test_sgppropagator_state_pef() {
        setup_global_test_eop_original_brahe();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // State in TEME frame (native SGP4 output)
        let state = prop.state_pef(epoch);

        assert_eq!(state.len(), 6);
        // TEME is the native SGP4 output frame

        // Differences from tighter tolerances have been primarily attributed
        // to differences in UT1-UTC calclation
        assert_abs_diff_eq!(state[0], -3953205.7105210484, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[1], 1427514.704810681, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[2], 5243614.536966579, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[3], -3175.692140186211, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[4], -6658.887120918979, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[5], -583.775727402632, epsilon = 1.5e-1);
    }

    #[test]
    #[ignore] // TODO: Velocity error is higher than desired - Need to do deeper-dive validation of frame transformations
    fn test_sgppropagator_state_ecef_values() {
        setup_global_test_eop_original_brahe();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // State in ECEF/ITRF frame
        let state = prop.state_ecef(epoch).unwrap();

        assert_eq!(state.len(), 6);
        // ECEF/ITRF frame
        assert_abs_diff_eq!(state[0], -3953198.5496517573, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[1], 1427508.1713723878, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[2], 5243621.714247745, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[3], -3414.313706718372, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[4], -7222.549343535009, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[5], -583.7798954042405, epsilon = 1.5e-1);
    }

    #[test]
    #[ignore] // TODO: Velocity error is higher than desired - Need to do deeper-dive validation of frame transformations
    fn test_sgppropagator_state_itrf_values() {
        setup_global_test_eop_original_brahe();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // State in ECEF/ITRF frame
        let state = prop.state_itrf(epoch).unwrap();

        assert_eq!(state.len(), 6);
        // ECEF/ITRF frame
        assert_abs_diff_eq!(state[0], -3953198.5496517573, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[1], 1427508.1713723878, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[2], 5243621.714247745, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[3], -3414.313706718372, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[4], -7222.549343535009, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[5], -583.7798954042405, epsilon = 1.5e-1);
    }

    #[test]
    #[ignore] // TODO: Velocity error is higher than desired - Need to do deeper-dive validation of frame transformations
    fn test_sgppropagator_state_eci_values() {
        setup_global_test_eop_original_brahe();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // State in ECI/GCRF frame
        let state = prop.state_eci(epoch).unwrap();

        assert_eq!(state.len(), 6);
        // ECI/GCRF frame (after TEME -> PEF -> ECEF -> ECI conversion)
        assert_abs_diff_eq!(state[0], 4086521.040536244, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[1], -1001422.0787863219, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[2], 5240097.960898061, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[3], 2704.171077071122, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[4], 7840.6666110244705, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[5], -586.3906587951877, epsilon = 1.5e-1);
    }

    #[test]
    #[ignore] // TODO: Velocity error is higher than desired - Need to do deeper-dive validation of frame transformations
    fn test_sgppropagator_state_gcrf_values() {
        setup_global_test_eop_original_brahe();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // State in ECI/GCRF frame
        let state = prop.state_gcrf(epoch).unwrap();

        assert_eq!(state.len(), 6);
        // ECI/GCRF frame (after TEME -> PEF -> ECEF -> ECI conversion)
        assert_abs_diff_eq!(state[0], 4086521.040536244, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[1], -1001422.0787863219, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[2], 5240097.960898061, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[3], 2704.171077071122, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[4], 7840.6666110244705, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[5], -586.3906587951877, epsilon = 1.5e-1);
    }

    #[test]
    #[ignore] // TODO: Velocity error is higher than desired - Need to do deeper-dive validation of frame transformations
    fn test_sgppropagator_state_eme2000_values() {
        setup_global_test_eop_original_brahe();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // State in EME2000 frame
        let state = prop.state_eme2000(epoch).unwrap();

        assert_eq!(state.len(), 6);
        // EME2000 frame (GCRF with bias transformation applied)
        // Expected values computed from GCRF state with bias matrix
        assert_abs_diff_eq!(state[0], 4086547.890843119, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[1], -1001422.5866752749, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[2], 5240072.135733086, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[3], 2704.1707451936, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[4], 7840.6666131931, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[5], -586.3938863063, epsilon = 1.5e-1);
    }

    // with_output_format Method Tests

    #[test]
    fn test_sgppropagator_with_output_format_eci_cartesian() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0)
            .unwrap()
            .with_output_format(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        assert_eq!(prop.frame, OrbitFrame::ECI);
        assert_eq!(prop.representation, OrbitRepresentation::Cartesian);
        assert_eq!(prop.angle_format, None);
        assert_eq!(prop.trajectory.len(), 1); // Only initial state
    }

    #[test]
    fn test_sgppropagator_with_output_format_ecef_cartesian() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0)
            .unwrap()
            .with_output_format(OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None);

        assert_eq!(prop.frame, OrbitFrame::ECEF);
        assert_eq!(prop.representation, OrbitRepresentation::Cartesian);
        assert_eq!(prop.angle_format, None);

        // Initial state should be in ECEF frame
        let state = prop.current_state();
        assert!(state.norm() > 0.0);
    }

    #[test]
    fn test_sgppropagator_with_output_format_gcrf_cartesian() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0)
            .unwrap()
            .with_output_format(OrbitFrame::GCRF, OrbitRepresentation::Cartesian, None);

        assert_eq!(prop.frame, OrbitFrame::GCRF);
        assert_eq!(prop.representation, OrbitRepresentation::Cartesian);
        assert_eq!(prop.angle_format, None);

        // Initial state should be in GCRF frame
        let state = prop.current_state();
        assert!(state.norm() > 0.0);
    }

    #[test]
    fn test_sgppropagator_with_output_format_eme2000_cartesian() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0)
            .unwrap()
            .with_output_format(OrbitFrame::EME2000, OrbitRepresentation::Cartesian, None);

        assert_eq!(prop.frame, OrbitFrame::EME2000);
        assert_eq!(prop.representation, OrbitRepresentation::Cartesian);
        assert_eq!(prop.angle_format, None);

        // Initial state should be in EME2000 frame
        let state = prop.current_state();
        assert!(state.norm() > 0.0);
    }

    #[test]
    fn test_sgppropagator_with_output_format_itrf_cartesian() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0)
            .unwrap()
            .with_output_format(OrbitFrame::ITRF, OrbitRepresentation::Cartesian, None);

        assert_eq!(prop.frame, OrbitFrame::ITRF);
        assert_eq!(prop.representation, OrbitRepresentation::Cartesian);
        assert_eq!(prop.angle_format, None);

        // Initial state should be in ITRF frame
        let state = prop.current_state();
        assert!(state.norm() > 0.0);
    }

    #[test]
    fn test_sgppropagator_with_output_format_eci_keplerian_degrees() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0)
            .unwrap()
            .with_output_format(
                OrbitFrame::ECI,
                OrbitRepresentation::Keplerian,
                Some(AngleFormat::Degrees),
            );

        assert_eq!(prop.frame, OrbitFrame::ECI);
        assert_eq!(prop.representation, OrbitRepresentation::Keplerian);
        assert_eq!(prop.angle_format, Some(AngleFormat::Degrees));

        // Initial state should be Keplerian elements
        let state = prop.current_state();
        assert!(state[0] > 0.0); // Semi-major axis positive
        assert!(state[1] >= 0.0 && state[1] < 1.0); // Eccentricity in valid range
    }

    #[test]
    fn test_sgppropagator_with_output_format_eci_keplerian_radians() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0)
            .unwrap()
            .with_output_format(
                OrbitFrame::ECI,
                OrbitRepresentation::Keplerian,
                Some(AngleFormat::Radians),
            );

        assert_eq!(prop.frame, OrbitFrame::ECI);
        assert_eq!(prop.representation, OrbitRepresentation::Keplerian);
        assert_eq!(prop.angle_format, Some(AngleFormat::Radians));

        // Initial state should be Keplerian elements
        let state = prop.current_state();
        assert!(state[0] > 0.0); // Semi-major axis positive
        assert!(state[1] >= 0.0 && state[1] < 1.0); // Eccentricity in valid range
        // Inclination should be in radians (less than pi)
        assert!(state[2] < std::f64::consts::PI);
    }

    #[test]
    fn test_sgppropagator_with_output_format_resets_trajectory() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        // Propagate to add states
        prop.propagate_steps(5);
        assert_eq!(prop.trajectory.len(), 6);

        // Change output format - should reset trajectory
        let prop = prop.with_output_format(OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None);
        assert_eq!(prop.trajectory.len(), 1); // Only initial state in new format
    }

    #[test]
    fn test_sgppropagator_with_output_format_propagate_in_new_format() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0)
            .unwrap()
            .with_output_format(OrbitFrame::ECEF, OrbitRepresentation::Cartesian, None);

        // Propagate in new format
        prop.propagate_steps(3);
        assert_eq!(prop.trajectory.len(), 4);

        // All states should be valid
        let state = prop.current_state();
        assert!(state.norm() > 0.0);
    }

    #[test]
    #[should_panic(expected = "Angle format must be specified for Keplerian elements")]
    fn test_sgppropagator_with_output_format_keplerian_without_angle_format() {
        setup_global_test_eop();
        let _prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0)
            .unwrap()
            .with_output_format(OrbitFrame::ECI, OrbitRepresentation::Keplerian, None);
    }

    #[test]
    #[should_panic(expected = "Keplerian elements must be in ECI frame")]
    fn test_sgppropagator_with_output_format_keplerian_non_eci_frame() {
        setup_global_test_eop();
        let _prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0)
            .unwrap()
            .with_output_format(
                OrbitFrame::ECEF,
                OrbitRepresentation::Keplerian,
                Some(AngleFormat::Degrees),
            );
    }

    #[test]
    #[should_panic(expected = "Angle format should be None for Cartesian representation")]
    fn test_sgppropagator_with_output_format_cartesian_with_angle_format() {
        setup_global_test_eop();
        let _prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0)
            .unwrap()
            .with_output_format(
                OrbitFrame::ECI,
                OrbitRepresentation::Cartesian,
                Some(AngleFormat::Degrees),
            );
    }

    // state_gcrf and state_eme2000 Tests (non-ignored basic tests)

    #[test]
    fn test_sgppropagator_state_gcrf() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        let state = prop.state_gcrf(epoch).unwrap();

        // State should be a valid 6D vector
        assert_eq!(state.len(), 6);
        assert!(state.norm() > 0.0);

        // Position magnitude should be reasonable for LEO (6000-7000 km from Earth center)
        let r = (state[0].powi(2) + state[1].powi(2) + state[2].powi(2)).sqrt();
        assert!(r > 6_000_000.0 && r < 7_000_000.0);

        // Velocity magnitude should be reasonable for LEO (~7-8 km/s)
        let v = (state[3].powi(2) + state[4].powi(2) + state[5].powi(2)).sqrt();
        assert!(v > 7_000.0 && v < 8_000.0);
    }

    #[test]
    fn test_sgppropagator_state_gcrf_at_different_epochs() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.initial_epoch();

        let state1 = prop.state_gcrf(initial_epoch).unwrap();
        let state2 = prop.state_gcrf(initial_epoch + 60.0).unwrap();

        // States should be different after propagation
        assert_ne!(state1, state2);

        // But magnitudes should be similar (same orbit)
        let r1 = (state1[0].powi(2) + state1[1].powi(2) + state1[2].powi(2)).sqrt();
        let r2 = (state2[0].powi(2) + state2[1].powi(2) + state2[2].powi(2)).sqrt();
        assert_abs_diff_eq!(r1, r2, epsilon = 10_000.0); // Within 10 km
    }

    #[test]
    fn test_sgppropagator_state_eme2000() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        let state = prop.state_eme2000(epoch).unwrap();

        // State should be a valid 6D vector
        assert_eq!(state.len(), 6);
        assert!(state.norm() > 0.0);

        // Position magnitude should be reasonable for LEO (6000-7000 km from Earth center)
        let r = (state[0].powi(2) + state[1].powi(2) + state[2].powi(2)).sqrt();
        assert!(r > 6_000_000.0 && r < 7_000_000.0);

        // Velocity magnitude should be reasonable for LEO (~7-8 km/s)
        let v = (state[3].powi(2) + state[4].powi(2) + state[5].powi(2)).sqrt();
        assert!(v > 7_000.0 && v < 8_000.0);
    }

    #[test]
    fn test_sgppropagator_state_eme2000_at_different_epochs() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.initial_epoch();

        let state1 = prop.state_eme2000(initial_epoch).unwrap();
        let state2 = prop.state_eme2000(initial_epoch + 60.0).unwrap();

        // States should be different after propagation
        assert_ne!(state1, state2);

        // But magnitudes should be similar (same orbit)
        let r1 = (state1[0].powi(2) + state1[1].powi(2) + state1[2].powi(2)).sqrt();
        let r2 = (state2[0].powi(2) + state2[1].powi(2) + state2[2].powi(2)).sqrt();
        assert_abs_diff_eq!(r1, r2, epsilon = 10_000.0); // Within 10 km
    }

    #[test]
    fn test_sgppropagator_state_gcrf_vs_eme2000() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        let gcrf_state = prop.state_gcrf(epoch).unwrap();
        let eme2000_state = prop.state_eme2000(epoch).unwrap();

        // GCRF and EME2000 should be very close (frame bias is small)
        // Difference should be less than 100 meters in position
        let dr = ((gcrf_state[0] - eme2000_state[0]).powi(2)
            + (gcrf_state[1] - eme2000_state[1]).powi(2)
            + (gcrf_state[2] - eme2000_state[2]).powi(2))
        .sqrt();
        assert!(dr < 100.0, "Position difference {} m > 100 m", dr);

        // Velocity difference should be very small (less than 0.1 m/s)
        let dv = ((gcrf_state[3] - eme2000_state[3]).powi(2)
            + (gcrf_state[4] - eme2000_state[4]).powi(2)
            + (gcrf_state[5] - eme2000_state[5]).powi(2))
        .sqrt();
        assert!(dv < 0.1, "Velocity difference {} m/s > 0.1 m/s", dv);
    }

    #[test]
    fn test_sgppropagator_state_gcrf_consistency_with_eci() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        let gcrf_state = prop.state_gcrf(epoch).unwrap();
        let eci_state = prop.state_eci(epoch).unwrap();

        // GCRF and ECI should be identical (ECI is implemented as GCRF)
        assert_abs_diff_eq!(gcrf_state[0], eci_state[0], epsilon = 1.0);
        assert_abs_diff_eq!(gcrf_state[1], eci_state[1], epsilon = 1.0);
        assert_abs_diff_eq!(gcrf_state[2], eci_state[2], epsilon = 1.0);
        assert_abs_diff_eq!(gcrf_state[3], eci_state[3], epsilon = 1e-6);
        assert_abs_diff_eq!(gcrf_state[4], eci_state[4], epsilon = 1e-6);
        assert_abs_diff_eq!(gcrf_state[5], eci_state[5], epsilon = 1e-6);
    }

    // ===== Event Detection Tests =====

    use crate::events::{DAscendingNodeEvent, DTimeEvent};

    #[test]
    fn test_sgppropagator_add_event_detector() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Add a time event
        let detector = DTimeEvent::new(epoch + 300.0, "5 minute mark");
        prop.add_event_detector(Box::new(detector));

        // Initial state: no events detected yet
        assert!(prop.event_log().is_empty());
        assert!(!prop.is_terminated());
    }

    #[test]
    fn test_sgppropagator_time_event_detection() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Add a time event at 150 seconds
        let target_time = epoch + 150.0;
        let detector = DTimeEvent::new(target_time, "Time Target");
        prop.add_event_detector(Box::new(detector));

        // Propagate past the event
        prop.propagate_to(epoch + 300.0);

        // Event should have been detected
        let events = prop.event_log();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "Time Target");

        // Event time should be very close to target
        assert_abs_diff_eq!(events[0].window_open.jd(), target_time.jd(), epsilon = 1e-8);
    }

    #[test]
    fn test_sgppropagator_multiple_events() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Add multiple time events
        let detector1 = DTimeEvent::new(epoch + 100.0, "Event 1");
        let detector2 = DTimeEvent::new(epoch + 200.0, "Event 2");
        let detector3 = DTimeEvent::new(epoch + 300.0, "Event 3");

        prop.add_event_detector(Box::new(detector1));
        prop.add_event_detector(Box::new(detector2));
        prop.add_event_detector(Box::new(detector3));

        // Propagate past all events
        prop.propagate_to(epoch + 400.0);

        // All events should be detected
        let events = prop.event_log();
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn test_sgppropagator_events_by_name() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Add events with different names
        let detector1 = DTimeEvent::new(epoch + 100.0, "Alpha Event");
        let detector2 = DTimeEvent::new(epoch + 200.0, "Beta Event");
        let detector3 = DTimeEvent::new(epoch + 300.0, "Alpha Prime");

        prop.add_event_detector(Box::new(detector1));
        prop.add_event_detector(Box::new(detector2));
        prop.add_event_detector(Box::new(detector3));

        prop.propagate_to(epoch + 400.0);

        // Search for "Alpha" events
        let alpha_events = prop.events_by_name("Alpha");
        assert_eq!(alpha_events.len(), 2);

        // Search for "Beta" events
        let beta_events = prop.events_by_name("Beta");
        assert_eq!(beta_events.len(), 1);
    }

    #[test]
    fn test_sgppropagator_latest_event() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Initially no events
        assert!(prop.latest_event().is_none());

        // Add events
        let detector1 = DTimeEvent::new(epoch + 100.0, "First");
        let detector2 = DTimeEvent::new(epoch + 200.0, "Second");

        prop.add_event_detector(Box::new(detector1));
        prop.add_event_detector(Box::new(detector2));

        prop.propagate_to(epoch + 250.0);

        // Latest event should be "Second"
        let latest = prop.latest_event().unwrap();
        assert_eq!(latest.name, "Second");
    }

    #[test]
    fn test_sgppropagator_events_in_range() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Add events at 100, 200, 300 seconds
        let detector1 = DTimeEvent::new(epoch + 100.0, "Event 1");
        let detector2 = DTimeEvent::new(epoch + 200.0, "Event 2");
        let detector3 = DTimeEvent::new(epoch + 300.0, "Event 3");

        prop.add_event_detector(Box::new(detector1));
        prop.add_event_detector(Box::new(detector2));
        prop.add_event_detector(Box::new(detector3));

        prop.propagate_to(epoch + 400.0);

        // Events between 150-250 seconds
        let range_events = prop.events_in_range(epoch + 150.0, epoch + 250.0);
        assert_eq!(range_events.len(), 1);
        assert_eq!(range_events[0].name, "Event 2");
    }

    #[test]
    fn test_sgppropagator_events_by_detector_index() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Add two detectors
        let detector1 = DTimeEvent::new(epoch + 100.0, "Detector 0 Event");
        let detector2 = DTimeEvent::new(epoch + 200.0, "Detector 1 Event");

        prop.add_event_detector(Box::new(detector1));
        prop.add_event_detector(Box::new(detector2));

        prop.propagate_to(epoch + 300.0);

        // Get events by detector index
        let events_0 = prop.events_by_detector_index(0);
        let events_1 = prop.events_by_detector_index(1);

        assert_eq!(events_0.len(), 1);
        assert_eq!(events_0[0].name, "Detector 0 Event");
        assert_eq!(events_1.len(), 1);
        assert_eq!(events_1[0].name, "Detector 1 Event");
    }

    #[test]
    fn test_sgppropagator_terminal_event() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Add a terminal event at 150 seconds
        let detector = DTimeEvent::new(epoch + 150.0, "Stop Here").set_terminal();

        prop.add_event_detector(Box::new(detector));

        // Try to propagate to 300 seconds
        prop.propagate_to(epoch + 300.0);

        // Should be terminated
        assert!(prop.is_terminated());

        // Propagation should have stopped at the event
        let current = prop.current_epoch();
        assert!(
            current < epoch + 200.0,
            "Propagation should have stopped before 200s"
        );
    }

    #[test]
    fn test_sgppropagator_reset_clears_events() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Add and trigger an event
        let detector = DTimeEvent::new(epoch + 100.0, "Test Event");
        prop.add_event_detector(Box::new(detector));
        prop.propagate_to(epoch + 200.0);

        assert_eq!(prop.event_log().len(), 1);

        // Reset the propagator
        prop.reset();

        // Events should be cleared
        assert!(prop.event_log().is_empty());
        assert!(!prop.is_terminated());
    }

    #[test]
    fn test_sgppropagator_clear_events() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Add and trigger events
        let detector = DTimeEvent::new(epoch + 100.0, "Test Event");
        prop.add_event_detector(Box::new(detector));
        prop.propagate_to(epoch + 200.0);

        assert_eq!(prop.event_log().len(), 1);

        // Clear only events (not trajectory)
        prop.clear_events();

        assert!(prop.event_log().is_empty());
        // Trajectory should still have data
        assert!(prop.trajectory.len() > 1);
    }

    #[test]
    fn test_sgppropagator_reset_termination() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Add a terminal event
        let detector = DTimeEvent::new(epoch + 100.0, "Terminal").set_terminal();
        prop.add_event_detector(Box::new(detector));

        prop.propagate_to(epoch + 200.0);
        assert!(prop.is_terminated());

        // Reset termination
        prop.reset_termination();
        assert!(!prop.is_terminated());

        // Should be able to propagate again
        prop.propagate_to(epoch + 300.0);
        assert!(prop.current_epoch() > epoch + 250.0);
    }

    #[test]
    fn test_sgppropagator_ascending_node_detection() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Add ascending node detector
        let detector = DAscendingNodeEvent::new("Ascending Node");
        prop.add_event_detector(Box::new(detector));

        // Propagate for one full orbit (~92 minutes for ISS)
        prop.propagate_to(epoch + 5600.0);

        // Should detect at least one ascending node crossing
        let events = prop.event_log();
        assert!(!events.is_empty(), "Should detect ascending node crossings");
    }

    #[test]
    fn test_sgppropagator_clone_does_not_copy_events() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Add an event
        let detector = DTimeEvent::new(epoch + 100.0, "Test Event");
        prop.add_event_detector(Box::new(detector));
        prop.propagate_to(epoch + 200.0);

        assert_eq!(prop.event_log().len(), 1);

        // Clone the propagator
        let cloned = prop.clone();

        // Cloned propagator should NOT have the event detector or log
        assert!(cloned.event_log().is_empty());
        assert!(!cloned.is_terminated());
    }

    #[test]
    fn test_sgppropagator_query_events() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Add events
        let detector1 = DTimeEvent::new(epoch + 100.0, "First");
        let detector2 = DTimeEvent::new(epoch + 200.0, "Second");
        let detector3 = DTimeEvent::new(epoch + 300.0, "Third");

        prop.add_event_detector(Box::new(detector1));
        prop.add_event_detector(Box::new(detector2));
        prop.add_event_detector(Box::new(detector3));

        prop.propagate_to(epoch + 400.0);

        // Use query builder
        let filtered: Vec<_> = prop
            .query_events()
            .after(epoch + 150.0)
            .before(epoch + 350.0)
            .collect();

        assert_eq!(filtered.len(), 2);
    }

    // ===== Event Detector Management Tests =====

    #[test]
    fn test_sgppropagator_take_event_detectors() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Add two event detectors
        let detector1 = DTimeEvent::new(epoch + 100.0, "Event1");
        let detector2 = DTimeEvent::new(epoch + 200.0, "Event2");
        prop.add_event_detector(Box::new(detector1));
        prop.add_event_detector(Box::new(detector2));

        // Take the detectors
        let taken = prop.take_event_detectors();

        // Verify we got the right number
        assert_eq!(taken.len(), 2);

        // Verify propagator now has empty detectors
        // (can't directly access, but propagating should not detect events)
        prop.propagate_to(epoch + 300.0);
        assert!(prop.event_log().is_empty());
    }

    #[test]
    fn test_sgppropagator_set_event_detectors() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Create detectors in a vector
        let detectors: Vec<Box<dyn crate::events::DEventDetector>> = vec![
            Box::new(DTimeEvent::new(epoch + 100.0, "Event1")),
            Box::new(DTimeEvent::new(epoch + 200.0, "Event2")),
        ];

        // Set the detectors
        prop.set_event_detectors(detectors);

        // Propagate and verify events are detected
        prop.propagate_to(epoch + 300.0);
        assert_eq!(prop.event_log().len(), 2);
    }

    #[test]
    fn test_sgppropagator_take_event_log() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Add detector and propagate to trigger event
        let detector = DTimeEvent::new(epoch + 100.0, "TestEvent");
        prop.add_event_detector(Box::new(detector));
        prop.propagate_to(epoch + 200.0);

        // Verify event was detected
        assert_eq!(prop.event_log().len(), 1);

        // Take the event log
        let taken_log = prop.take_event_log();

        // Verify we got the events
        assert_eq!(taken_log.len(), 1);
        assert_eq!(taken_log[0].name, "TestEvent");

        // Verify propagator now has empty log
        assert!(prop.event_log().is_empty());
    }

    #[test]
    fn test_sgppropagator_set_terminated_is_terminated() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        // Initial state should be false
        assert!(!prop.is_terminated());

        // Set to true
        prop.set_terminated(true);
        assert!(prop.is_terminated());

        // Set back to false
        prop.set_terminated(false);
        assert!(!prop.is_terminated());
    }

    #[test]
    fn test_sgppropagator_event_detector_roundtrip() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Add detector
        let detector = DTimeEvent::new(epoch + 150.0, "RoundtripEvent");
        prop.add_event_detector(Box::new(detector));

        // Take detectors
        let taken = prop.take_event_detectors();
        assert_eq!(taken.len(), 1);

        // Propagate - should not detect (no detectors)
        prop.propagate_to(epoch + 100.0);
        assert!(prop.event_log().is_empty());

        // Set detectors back
        prop.set_event_detectors(taken);

        // Continue propagation past event time
        prop.propagate_to(epoch + 200.0);

        // Now event should be detected
        assert_eq!(prop.event_log().len(), 1);
        assert!(prop.event_log()[0].name.contains("RoundtripEvent"));
    }
}
