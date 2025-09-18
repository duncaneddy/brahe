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

use nalgebra::Vector6;
use serde::{Deserialize, Serialize};
use sgp4::chrono::{Datelike, Timelike};

use crate::coordinates::state_cartesian_to_osculating;
use crate::frames::state_eci_to_ecef;
use crate::orbits::traits::{AnalyticPropagator, OrbitPropagator};
use crate::orbits::tle::{calculate_tle_line_checksum, validate_tle_lines, parse_norad_id, TleFormat};
use crate::time::Epoch;
use crate::trajectories::{TrajectoryEvictionPolicy, AngleFormat, OrbitFrame, OrbitRepresentation, OrbitalTrajectory, InterpolationMethod};
use crate::utils::BraheError;


/// SGP4 propagator using TLE data with the new architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    /// Parsed SGP4 elements
    elements: sgp4::Elements,

    /// SGP4 propagation constants
    constants: sgp4::Constants,

    /// Initial epoch from TLE
    initial_epoch: Epoch,

    /// Initial state vector (always ECI Cartesian from SGP4)
    initial_state: Vector6<f64>,

    /// Accumulated trajectory with configurable management
    trajectory: OrbitalTrajectory,

    /// Step size in seconds for stepping operations
    step_size: f64,

    /// Output frame (default: ECI)
    output_frame: OrbitFrame,

    /// Output representation (default: Cartesian)
    output_representation: OrbitRepresentation,

    /// Output angle format (default: Radians)
    output_angle_format: AngleFormat,
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
    pub fn from_3le(name: Option<&str>, line1: &str, line2: &str, step_size: f64) -> Result<Self, BraheError> {
        // Validate TLE format
        if !validate_tle_lines(line1, line2) {
            return Err(BraheError::Error("Invalid TLE format".to_string()));
        }

        // Extract NORAD ID and determine format
        let norad_id_string = line1[2..7].trim().to_string();
        let norad_id = parse_norad_id(&norad_id_string)?;
        let format = if norad_id_string.chars().next().unwrap_or('0').is_alphabetic() {
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
                if i < line1_chars.len() { line1_chars[i] = '0'; }
                if i < line2_chars.len() { line2_chars[i] = '0'; }
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
        ).map_err(|e| BraheError::Error(format!("SGP4 parsing error: {:?}", e)))?;

        let constants = sgp4::Constants::from_elements(&elements)
            .map_err(|e| BraheError::Error(format!("SGP4 constants error: {:?}", e)))?;

        // Extract initial epoch
        let initial_epoch = Self::extract_epoch_from_elements(&elements)?;

        // Compute initial state in ECI
        let prediction = constants.propagate(sgp4::MinutesSinceEpoch(0.0))
            .map_err(|e| BraheError::Error(format!("SGP4 propagation error: {:?}", e)))?;

        let initial_state = Vector6::new(
            prediction.position[0],
            prediction.position[1],
            prediction.position[2],
            prediction.velocity[0],
            prediction.velocity[1],
            prediction.velocity[2],
        );

        // Create trajectory with initial state
        let mut trajectory = OrbitalTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,  // Cartesian representation should use None for angle format
            InterpolationMethod::Linear,
        )?;
        trajectory.add_state(initial_epoch, initial_state)?;

        Ok(SGPPropagator {
            line1: line1.to_string(),
            line2: line2.to_string(),
            satellite_name: name.map(|s| s.to_string()),
            format,
            norad_id_string,
            norad_id,
            elements,
            constants,
            initial_epoch,
            initial_state,
            trajectory,
            step_size,
            output_frame: OrbitFrame::ECI,
            output_representation: OrbitRepresentation::Cartesian,
            output_angle_format: AngleFormat::None,
        })
    }

    /// Extract epoch from SGP4 elements
    fn extract_epoch_from_elements(elements: &sgp4::Elements) -> Result<Epoch, BraheError> {
        let dt = elements.datetime;
        let year = dt.year() as f64;
        let month = dt.month() as f64;
        let day = dt.day() as f64;
        let hour = dt.hour() as f64;
        let minute = dt.minute() as f64;
        let second = dt.second() as f64 + dt.nanosecond() as f64 / 1e9;

        // Convert to Julian date and then to Epoch
        let jd = crate::time::conversions::datetime_to_jd(year as u32, month as u8, day as u8, hour as u8, minute as u8, second, 0.0);
        Ok(Epoch::from_jd(jd, crate::time::TimeSystem::UTC))
    }

    /// Set output to Cartesian coordinates
    pub fn set_output_cartesian(&mut self) {
        self.output_representation = OrbitRepresentation::Cartesian;
    }

    /// Set output to Keplerian elements
    pub fn set_output_keplerian(&mut self) {
        self.output_representation = OrbitRepresentation::Keplerian;
    }

    /// Set output frame
    pub fn set_output_frame(&mut self, frame: OrbitFrame) {
        self.output_frame = frame;
    }

    /// Set output angle format
    pub fn set_output_angle_format(&mut self, angle_format: AngleFormat) {
        self.output_angle_format = angle_format;
    }
}

impl OrbitPropagator for SGPPropagator {
    fn step(&mut self) -> Result<(), BraheError> {
        self.step_by(self.step_size)
    }

    fn step_by(&mut self, step_size: f64) -> Result<(), BraheError> {
        let current_epoch = self.current_epoch();
        let target_epoch = current_epoch + step_size / 86400.0; // Convert seconds to days
        self.propagate_to(target_epoch)
    }

    fn propagate_steps(&mut self, num_steps: usize) -> Result<(), BraheError> {
        for _ in 0..num_steps {
            self.step()?;
        }
        Ok(())
    }

    fn propagate_to(&mut self, target_epoch: Epoch) -> Result<(), BraheError> {
        // Compute state at target epoch
        let state = self.state(target_epoch);

        // Convert to desired output format
        let output_state = self.trajectory.convert_state_to_format(
            state,
            target_epoch,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::Radians,
            self.output_frame,
            self.output_representation,
            self.output_angle_format,
        )?;

        // Add to trajectory
        self.trajectory.add_state(target_epoch, output_state)?;
        Ok(())
    }

    fn current_state(&self) -> Vector6<f64> {
        self.trajectory.current_state_vector()
    }

    fn current_epoch(&self) -> Epoch {
        self.trajectory.current_epoch()
    }

    fn initial_state(&self) -> Vector6<f64> {
        self.initial_state
    }

    fn initial_epoch(&self) -> Epoch {
        self.initial_epoch
    }

    fn step_size(&self) -> f64 {
        self.step_size
    }

    fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }

    fn reset(&mut self) -> Result<(), BraheError> {
        self.trajectory.clear();
        self.trajectory.add_state(self.initial_epoch, self.initial_state)?;
        Ok(())
    }

    fn set_initial_conditions(
        &mut self,
        _epoch: Epoch,
        _state: Vector6<f64>,
        _frame: OrbitFrame,
        _representation: OrbitRepresentation,
        _angle_format: AngleFormat,
    ) -> Result<(), BraheError> {
        // For SGP propagator, initial conditions come from TLE and cannot be changed
        Err(BraheError::Error(
            "Cannot change initial conditions for SGP propagator - state is determined by TLE data".to_string()
        ))
    }

    fn trajectory(&self) -> &OrbitalTrajectory {
        &self.trajectory
    }

    fn trajectory_mut(&mut self) -> &mut OrbitalTrajectory {
        &mut self.trajectory
    }

    fn set_max_trajectory_size(&mut self, max_size: Option<usize>) {
        self.trajectory.set_max_size(max_size);
    }

    fn set_max_trajectory_age(&mut self, max_age: Option<f64>) {
        self.trajectory.set_max_age(max_age);
    }

    fn set_eviction_policy(&mut self, policy: TrajectoryEvictionPolicy) {
        self.trajectory.set_eviction_policy(policy);
    }
}

impl AnalyticPropagator for SGPPropagator {
    fn state(&self, epoch: Epoch) -> Vector6<f64> {
        self.state_eci(epoch)
    }

    fn state_eci(&self, epoch: Epoch) -> Vector6<f64> {
        // Calculate minutes since TLE epoch
        let time_diff = (epoch.jd() - self.initial_epoch.jd()) * 1440.0; // Convert days to minutes

        // Propagate using SGP4
        let prediction = self.constants.propagate(sgp4::MinutesSinceEpoch(time_diff))
            .unwrap_or_else(|_| {
                // Return zero state on propagation error
                sgp4::Prediction {
                    position: [0.0, 0.0, 0.0],
                    velocity: [0.0, 0.0, 0.0],
                }
            });

        Vector6::new(
            prediction.position[0],
            prediction.position[1],
            prediction.position[2],
            prediction.velocity[0],
            prediction.velocity[1],
            prediction.velocity[2],
        )
    }

    fn state_ecef(&self, epoch: Epoch) -> Vector6<f64> {
        let eci_state = self.state_eci(epoch);
        state_eci_to_ecef(epoch, eci_state)
    }

    fn state_osculating_elements(&self, epoch: Epoch) -> Vector6<f64> {
        let eci_state = self.state_eci(epoch);
        state_cartesian_to_osculating(eci_state, false)
    }

    fn states(&self, epochs: &[Epoch]) -> OrbitalTrajectory {
        let mut trajectory = OrbitalTrajectory::new(
            self.output_frame,
            self.output_representation,
            self.output_angle_format,
            InterpolationMethod::Linear,
        ).unwrap();

        for &epoch in epochs {
            let state = self.state(epoch);
            let _ = trajectory.add_state(epoch, state);
        }

        trajectory
    }

    fn states_eci(&self, epochs: &[Epoch]) -> OrbitalTrajectory {
        let mut trajectory = OrbitalTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::Radians,
            InterpolationMethod::Linear,
        ).unwrap();

        for &epoch in epochs {
            let state = self.state_eci(epoch);
            let _ = trajectory.add_state(epoch, state);
        }

        trajectory
    }

    fn states_ecef(&self, epochs: &[Epoch]) -> OrbitalTrajectory {
        let mut trajectory = OrbitalTrajectory::new(
            OrbitFrame::ECEF,
            OrbitRepresentation::Cartesian,
            AngleFormat::Radians,
            InterpolationMethod::Linear,
        ).unwrap();

        for &epoch in epochs {
            let state = self.state_ecef(epoch);
            let _ = trajectory.add_state(epoch, state);
        }

        trajectory
    }

    fn states_osculating_elements(&self, epochs: &[Epoch]) -> OrbitalTrajectory {
        let mut trajectory = OrbitalTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Radians,
            InterpolationMethod::Linear,
        ).unwrap();

        for &epoch in epochs {
            let state = self.state_osculating_elements(epoch);
            let _ = trajectory.add_state(epoch, state);
        }

        trajectory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tle_checksum() {
        // Test cases provided by user - known correct ISS TLE data
        let iss_tle_line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
        let iss_tle_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

        // Debug: let's see what our function actually returns
        let result1 = calculate_tle_line_checksum(iss_tle_line1);
        let result2 = calculate_tle_line_checksum(iss_tle_line2);

        let expected_checksum1 = iss_tle_line1.chars().last().unwrap().to_digit(10).unwrap();
        let expected_checksum2 = iss_tle_line2.chars().last().unwrap().to_digit(10).unwrap();

        assert_eq!(result1, expected_checksum1, "Checksum for line 1 should be {}, got {}", expected_checksum1, result1);
        assert_eq!(result2, expected_checksum2, "Checksum for line 2 should be {}, got {}", expected_checksum2, result2);
    }

    #[test]
    fn test_validate_tle_line() {
        let iss_tle_line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
        assert!(validate_tle_line(iss_tle_line1));

        // Invalid line (wrong length)
        let invalid_line = "1 25544U 98067A   08264";
        assert!(!validate_tle_line(invalid_line));
    }

    #[test]
    fn test_validate_tle_lines() {
        let iss_tle_line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
        let iss_tle_line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
        assert!(validate_tle_lines(iss_tle_line1, iss_tle_line2));

        // Invalid lines (mismatched NORAD IDs)
        let invalid_line2 = "2 12345  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";
        assert!(!validate_tle_lines(iss_tle_line1, invalid_line2));

        // Invalid lines (wrong checksum)
        let invalid_line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4  0  2928"; // Changed last digit
        assert!(!validate_tle_lines(invalid_line1, iss_tle_line2));

        // Invalid lines (wrong line numbers)
        let invalid_line1_num = "2 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2928"; // Changed line number to 2
        assert!(!validate_tle_lines(invalid_line1_num, iss_tle_line2));
    }

    #[test]
    fn test_extract_classic_norad_id() {
        assert_eq!(parse_norad_id("25544").unwrap(), 25544);
    }

    #[test]
    fn test_extract_alpha5_norad_id() {
        assert_eq!(parse_norad_id("A0001").unwrap(), 100001);
    }

    #[test]
    fn test_sgp_propagator_creation() {
        // Use a real ISS TLE that should work with SGP4
        let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
        let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

        let propagator = SGPPropagator::from_tle(line1, line2, 60.0);
        if let Err(ref err) = propagator {
            println!("Error creating propagator: {:?}", err);
        }
        assert!(propagator.is_ok());
    }
}