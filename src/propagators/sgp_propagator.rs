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
#[cfg(test)]
use crate::constants::RADIANS;
use crate::constants::{AngleFormat, DEG2RAD, OMEGA_EARTH, RAD2DEG};
use crate::coordinates::state_cartesian_to_osculating;
use crate::frames::{polar_motion, state_ecef_to_eci};
use crate::orbits::tle::{
    TleFormat, calculate_tle_line_checksum, epoch_from_tle, parse_norad_id, validate_tle_lines,
};
use crate::propagators::traits::{OrbitPropagator, StateProvider};
use crate::time::{Epoch, TimeSystem};
use crate::trajectories::OrbitTrajectory;
use crate::trajectories::traits::{OrbitFrame, OrbitRepresentation, Trajectory};
use crate::utils::{BraheError, Identifiable};
use nalgebra::{Vector3, Vector6};

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
            OrbitFrame::ECEF => ecef_state,
        },
        OrbitRepresentation::Keplerian => {
            if frame != OrbitFrame::ECI {
                panic!("Keplerian elements must be in ECI frame");
            }

            if let Some(format) = angle_format {
                state_cartesian_to_osculating(state_ecef_to_eci(epoch, ecef_state), format)
            } else {
                panic!("Angle format must be specified for Keplerian elements");
            }
        }
    }
}

/// SGP4 propagator
#[derive(Debug, Clone)]
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
    pub trajectory: OrbitTrajectory,

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
            OrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, None);

        // Set trajectory identity from propagator identity
        if let Some(n) = name {
            trajectory = trajectory.with_name(n);
        }
        // NORAD ID is always available (norad_id: u32)
        trajectory = trajectory.with_id(norad_id as u64);

        trajectory.add(epoch, initial_state);

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

        // Reset trajectory to initial state only
        self.trajectory = OrbitTrajectory::new(frame, representation, angle_format);

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
        self.trajectory.add(self.epoch, initial_state);

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
}

impl OrbitPropagator for SGPPropagator {
    fn step_by(&mut self, step_size: f64) {
        let current_epoch = self.current_epoch();
        let target_epoch = current_epoch + step_size; // step_size is in seconds

        let tle_state = self.propagate_internal(target_epoch);
        let new_state = convert_state_from_spg4_frame(
            target_epoch,
            tle_state,
            self.frame,
            self.representation,
            self.angle_format,
        );
        self.trajectory.add(target_epoch, new_state)
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
        self.trajectory.last().unwrap().1
    }

    fn step_size(&self) -> f64 {
        self.step_size
    }

    fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    fn reset(&mut self) {
        self.trajectory.clear();
        self.trajectory.add(self.epoch, self.initial_state);
    }

    fn set_initial_conditions(
        &mut self,
        _epoch: Epoch,
        _state: Vector6<f64>,
        _frame: OrbitFrame,
        _representation: OrbitRepresentation,
        _angle_format: Option<AngleFormat>,
    ) {
        // For SGP propagator, initial conditions come from TLE and cannot be changed
        panic!(
            "Cannot change initial conditions for SGP propagator - state is determined by TLE data"
        );
    }

    fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_size(max_size)
    }

    fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_age(max_age)
    }
}

impl StateProvider for SGPPropagator {
    fn state(&self, epoch: Epoch) -> Vector6<f64> {
        self.propagate_internal(epoch)
    }

    fn state_eci(&self, epoch: Epoch) -> Vector6<f64> {
        let state_ecef = self.state_ecef(epoch);

        // Step 3: ECEF to ECI
        state_ecef_to_eci(epoch, state_ecef)
    }

    fn state_ecef(&self, epoch: Epoch) -> Vector6<f64> {
        let state_pef = self.state_pef(epoch);

        // Step 2: PEF to ECEF

        #[allow(non_snake_case)]
        let PM = polar_motion(epoch);

        let r_ecef = PM * Vector3::<f64>::from(state_pef.fixed_rows::<3>(0));
        let v_ecef = PM * Vector3::<f64>::from(state_pef.fixed_rows::<3>(3));

        Vector6::new(
            r_ecef[0], r_ecef[1], r_ecef[2], v_ecef[0], v_ecef[1], v_ecef[2],
        )
    }

    fn state_as_osculating_elements(
        &self,
        epoch: Epoch,
        angle_format: AngleFormat,
    ) -> Vector6<f64> {
        let state_eci = self.state_eci(epoch);

        state_cartesian_to_osculating(state_eci, angle_format)
    }

    // Default implementations from trait are used for:
    // - states()
    // - states_eci()
    // - states_ecef()
    // - states_as_osculating_elements()
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
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
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
    #[should_panic(expected = "Cannot change initial conditions for SGP propagator")]
    fn test_sgppropagator_orbitpropagator_set_initial_conditions() {
        setup_global_test_eop();
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = Epoch::from_datetime(2023, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);

        // Should panic - SGP propagator doesn't allow changing initial conditions
        prop.set_initial_conditions(
            epoch,
            state,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            None, // None for Cartesian
        );
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_set_eviction_policy_max_size() {
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
    fn test_sgppropagator_analyticpropagator_state_as_osculating_elements() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        let elements = prop.state_as_osculating_elements(epoch, RADIANS);

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

        let states = prop.states(&epochs);
        assert_eq!(states.len(), 3);
    }

    #[test]
    fn test_sgppropagator_analyticpropagator_states_eci() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.initial_epoch();

        let epochs = vec![initial_epoch, initial_epoch + 0.01];

        let states = prop.states_eci(&epochs);
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

        let states = prop.states_ecef(&epochs);
        assert_eq!(states.len(), 2);
        // Verify states are valid Cartesian vectors
        for state in &states {
            assert!(state.norm() > 0.0);
        }
    }

    #[test]
    fn test_sgppropagator_analyticpropagator_states_as_osculating_elements() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.initial_epoch();

        let epochs = vec![initial_epoch, initial_epoch + 0.01];

        let elements = prop.states_as_osculating_elements(&epochs, RADIANS);
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
        let state = prop.state(epoch);

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
    #[ignore] // Velocity error is higher than desired - Need to do deeper-dive validation of frame transformations
    fn test_sgppropagator_state_ecef_values() {
        setup_global_test_eop_original_brahe();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // State in ECEF/ITRF frame
        let state = prop.state_ecef(epoch);

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
    #[ignore] // Velocity error is higher than desired - Need to do deeper-dive validation of frame transformations
    fn test_sgppropagator_state_eci_values() {
        setup_global_test_eop_original_brahe();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // State in ECI/GCRF frame
        let state = prop.state_eci(epoch);

        assert_eq!(state.len(), 6);
        // ECI/GCRF frame (after TEME -> PEF -> ECEF -> ECI conversion)
        assert_abs_diff_eq!(state[0], 4086521.040536244, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[1], -1001422.0787863219, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[2], 5240097.960898061, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[3], 2704.171077071122, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[4], 7840.6666110244705, epsilon = 1.5e-1);
        assert_abs_diff_eq!(state[5], -586.3906587951877, epsilon = 1.5e-1);
    }
}
