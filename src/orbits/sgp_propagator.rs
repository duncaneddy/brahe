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
use sgp4::chrono::{Datelike, Timelike};

use crate::coordinates::state_cartesian_to_osculating;
use crate::frames::state_eci_to_ecef;
use crate::orbits::tle::{
    calculate_tle_line_checksum, parse_norad_id, validate_tle_lines, TleFormat,
};
use crate::orbits::traits::{AnalyticPropagator, OrbitPropagator};
use crate::time::Epoch;
use crate::trajectories::{
    AngleFormat, OrbitFrame, OrbitRepresentation, OrbitTrajectory, OrbitalTrajectory, Trajectory,
};
use crate::utils::BraheError;

/// SGP4 propagator using TLE data with the new architecture
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

    /// Parsed SGP4 elements
    #[allow(dead_code)]
    elements: sgp4::Elements,

    /// SGP4 propagation constants
    constants: sgp4::Constants,

    /// Initial epoch from TLE
    initial_epoch: Epoch,

    /// Initial state vector (always ECI Cartesian from SGP4)
    initial_state: Vector6<f64>,

    /// Accumulated trajectory with configurable management
    trajectory: OrbitTrajectory,

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

        // Extract initial epoch
        let initial_epoch = Self::extract_epoch_from_elements(&elements)?;

        // Compute initial state in ECI
        let prediction = constants
            .propagate(sgp4::MinutesSinceEpoch(0.0))
            .map_err(|e| BraheError::Error(format!("SGP4 propagation error: {:?}", e)))?;

        // Convert from km to m and km/s to m/s
        let initial_state = Vector6::new(
            prediction.position[0] * 1000.0,
            prediction.position[1] * 1000.0,
            prediction.position[2] * 1000.0,
            prediction.velocity[0] * 1000.0,
            prediction.velocity[1] * 1000.0,
            prediction.velocity[2] * 1000.0,
        );

        // Create trajectory with initial state
        let mut trajectory = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None, // Cartesian representation should use None for angle format
        );
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
        let jd = crate::time::conversions::datetime_to_jd(
            year as u32,
            month as u8,
            day as u8,
            hour as u8,
            minute as u8,
            second,
            0.0,
        );
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
        let target_epoch = current_epoch + step_size; // step_size is in seconds
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
            target_epoch,
            state,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
            self.output_frame,
            self.output_representation,
            self.output_angle_format,
        )?;

        // Add to trajectory
        self.trajectory.add_state(target_epoch, output_state)?;
        Ok(())
    }

    fn current_epoch(&self) -> Epoch {
        self.trajectory
            .epoch(self.trajectory.len() - 1)
            .unwrap_or(self.initial_epoch)
    }

    fn current_state(&self) -> Vector6<f64> {
        self.trajectory
            .state(self.trajectory.len() - 1)
            .unwrap_or(self.initial_state)
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
        self.trajectory
            .add_state(self.initial_epoch, self.initial_state)?;
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
            "Cannot change initial conditions for SGP propagator - state is determined by TLE data"
                .to_string(),
        ))
    }

    fn trajectory(&self) -> &OrbitTrajectory {
        &self.trajectory
    }

    fn trajectory_mut(&mut self) -> &mut OrbitTrajectory {
        &mut self.trajectory
    }

    fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_size(max_size)
    }

    fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_age(max_age)
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
        let prediction = self
            .constants
            .propagate(sgp4::MinutesSinceEpoch(time_diff))
            .unwrap_or_else(|_| {
                // Return zero state on propagation error
                sgp4::Prediction {
                    position: [0.0, 0.0, 0.0],
                    velocity: [0.0, 0.0, 0.0],
                }
            });

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

    fn state_ecef(&self, epoch: Epoch) -> Vector6<f64> {
        let eci_state = self.state_eci(epoch);
        state_eci_to_ecef(epoch, eci_state)
    }

    fn state_osculating_elements(&self, epoch: Epoch) -> Vector6<f64> {
        let eci_state = self.state_eci(epoch);
        state_cartesian_to_osculating(eci_state, false)
    }

    fn states(&self, epochs: &[Epoch]) -> OrbitTrajectory {
        let mut states = Vec::new();
        for &epoch in epochs {
            states.push(self.state(epoch));
        }

        OrbitTrajectory::from_orbital_data(
            epochs.to_vec(),
            states,
            self.output_frame,
            self.output_representation,
            self.output_angle_format,
        )
    }

    fn states_eci(&self, epochs: &[Epoch]) -> OrbitTrajectory {
        let mut states = Vec::new();
        for &epoch in epochs {
            states.push(self.state_eci(epoch));
        }

        OrbitTrajectory::from_orbital_data(
            epochs.to_vec(),
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        )
    }

    fn states_ecef(&self, epochs: &[Epoch]) -> OrbitTrajectory {
        let mut states = Vec::new();
        for &epoch in epochs {
            states.push(self.state_ecef(epoch));
        }

        OrbitTrajectory::from_orbital_data(
            epochs.to_vec(),
            states,
            OrbitFrame::ECEF,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        )
    }

    fn states_osculating_elements(&self, epochs: &[Epoch]) -> OrbitTrajectory {
        let mut states = Vec::new();
        for &epoch in epochs {
            states.push(self.state_osculating_elements(epoch));
        }

        OrbitTrajectory::from_orbital_data(
            epochs.to_vec(),
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Radians,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::testing::setup_global_test_eop;
    use approx::assert_abs_diff_eq;

    // Test TLE data
    const ISS_LINE1: &str = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    const ISS_LINE2: &str = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    // SGPPropagator Method Tests

    #[test]
    fn test_sgppropagator_from_tle() {
        let propagator = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0);
        assert!(propagator.is_ok());

        let prop = propagator.unwrap();
        assert_eq!(prop.step_size, 60.0);
        assert_eq!(prop.line1, ISS_LINE1);
        assert_eq!(prop.line2, ISS_LINE2);
    }

    #[test]
    fn test_sgppropagator_from_3le() {
        let name = "ISS (ZARYA)";
        let propagator = SGPPropagator::from_3le(Some(name), ISS_LINE1, ISS_LINE2, 60.0);
        assert!(propagator.is_ok());

        let prop = propagator.unwrap();
        assert_eq!(prop.satellite_name, Some(name.to_string()));
    }

    #[test]
    fn test_sgppropagator_set_output_cartesian() {
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        prop.set_output_keplerian();
        assert_eq!(prop.output_representation, OrbitRepresentation::Keplerian);

        prop.set_output_cartesian();
        assert_eq!(prop.output_representation, OrbitRepresentation::Cartesian);
    }

    #[test]
    fn test_sgppropagator_set_output_keplerian() {
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        assert_eq!(prop.output_representation, OrbitRepresentation::Cartesian);

        prop.set_output_keplerian();
        assert_eq!(prop.output_representation, OrbitRepresentation::Keplerian);
    }

    #[test]
    fn test_sgppropagator_set_output_frame() {
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        assert_eq!(prop.output_frame, OrbitFrame::ECI);

        prop.set_output_frame(OrbitFrame::ECEF);
        assert_eq!(prop.output_frame, OrbitFrame::ECEF);
    }

    #[test]
    fn test_sgppropagator_set_output_angle_format() {
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        assert_eq!(prop.output_angle_format, AngleFormat::None);

        prop.set_output_angle_format(AngleFormat::Degrees);
        assert_eq!(prop.output_angle_format, AngleFormat::Degrees);
    }

    // OrbitPropagator Trait Tests

    #[test]
    fn test_sgppropagator_orbitpropagator_step() {
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.current_epoch();

        prop.step().unwrap();
        let new_epoch = prop.current_epoch();

        assert_abs_diff_eq!(new_epoch - initial_epoch, 60.0, epsilon = 0.1);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_step_by() {
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.current_epoch();

        prop.step_by(120.0).unwrap();
        let new_epoch = prop.current_epoch();

        assert_abs_diff_eq!(new_epoch - initial_epoch, 120.0, epsilon = 0.1);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_propagate_steps() {
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.current_epoch();

        prop.propagate_steps(5).unwrap();
        let new_epoch = prop.current_epoch();

        assert_abs_diff_eq!(new_epoch - initial_epoch, 300.0, epsilon = 0.1);
        assert_eq!(prop.trajectory().len(), 6); // Initial + 5 steps
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_propagate_to() {
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.initial_epoch();
        let target_epoch = initial_epoch + 86400.0; // 1 day forward (in seconds)

        prop.propagate_to(target_epoch).unwrap();
        let current_epoch = prop.current_epoch();

        assert_abs_diff_eq!(current_epoch.jd(), target_epoch.jd(), epsilon = 1e-9);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_current_state() {
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let state = prop.current_state();

        // State should be non-zero for valid TLE
        assert!(state.norm() > 0.0);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_current_epoch() {
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.current_epoch();

        // Epoch should match TLE epoch
        assert_eq!(epoch, prop.initial_epoch());
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_initial_state() {
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let state = prop.initial_state();

        // State should be non-zero
        assert!(state.norm() > 0.0);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_initial_epoch() {
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        // Should be around 2008-09-20 based on TLE epoch
        assert!(epoch.jd() > 2454700.0 && epoch.jd() < 2454800.0);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_step_size() {
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        assert_eq!(prop.step_size(), 60.0);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_set_step_size() {
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        prop.set_step_size(120.0);
        assert_eq!(prop.step_size(), 120.0);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_reset() {
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        // Propagate forward
        prop.propagate_steps(5).unwrap();
        assert_eq!(prop.trajectory().len(), 6);

        // Reset
        prop.reset().unwrap();
        assert_eq!(prop.trajectory().len(), 1);
        assert_eq!(prop.current_epoch(), prop.initial_epoch());
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_set_initial_conditions() {
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = Epoch::from_datetime(2023, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);

        // Should return error - SGP propagator doesn't allow changing initial conditions
        let result = prop.set_initial_conditions(
            epoch,
            state,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_trajectory() {
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let traj = prop.trajectory();

        assert_eq!(traj.len(), 1); // Should have initial state
        assert_eq!(traj.frame, OrbitFrame::ECI);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_trajectory_mut() {
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let traj = prop.trajectory_mut();

        assert_eq!(traj.len(), 1);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_set_eviction_policy_max_size() {
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        prop.set_eviction_policy_max_size(5).unwrap();

        // Propagate 10 steps
        prop.propagate_steps(10).unwrap();

        // Should only keep 5 states
        assert_eq!(prop.trajectory().len(), 5);
    }

    #[test]
    fn test_sgppropagator_orbitpropagator_set_eviction_policy_max_age() {
        let mut prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();

        // Set eviction policy - should succeed
        let result = prop.set_eviction_policy_max_age(120.0);
        assert!(result.is_ok());

        // Propagate several steps
        prop.propagate_steps(10).unwrap();

        // Verify trajectory has states (eviction policy is applied)
        assert!(prop.trajectory().len() > 0);
    }

    // AnalyticPropagator Trait Tests

    #[test]
    fn test_sgppropagator_analyticpropagator_state() {
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch() + 0.01; // 0.01 days forward

        let state = prop.state(epoch);
        assert!(state.norm() > 0.0);
    }

    #[test]
    fn test_sgppropagator_analyticpropagator_state_eci() {
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        let state = prop.state_eci(epoch);

        // Should be close to initial state
        assert_abs_diff_eq!(state[0], prop.initial_state()[0], epsilon = 100.0);
    }

    #[test]
    fn test_sgppropagator_analyticpropagator_state_ecef() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        let state = prop.state_ecef(epoch);

        // ECEF state should be different from ECI due to frame rotation
        let eci_state = prop.state_eci(epoch);
        assert!((state - eci_state).norm() > 0.0);
    }

    #[test]
    fn test_sgppropagator_analyticpropagator_state_osculating_elements() {
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let epoch = prop.initial_epoch();

        let elements = prop.state_osculating_elements(epoch);

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
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.initial_epoch();

        let epochs = vec![initial_epoch, initial_epoch + 0.01, initial_epoch + 0.02];

        let traj = prop.states(&epochs);
        assert_eq!(traj.len(), 3);
    }

    #[test]
    fn test_sgppropagator_analyticpropagator_states_eci() {
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.initial_epoch();

        let epochs = vec![initial_epoch, initial_epoch + 0.01];

        let traj = prop.states_eci(&epochs);
        assert_eq!(traj.len(), 2);
        assert_eq!(traj.frame, OrbitFrame::ECI);
        assert_eq!(traj.representation, OrbitRepresentation::Cartesian);
    }

    #[test]
    fn test_sgppropagator_analyticpropagator_states_ecef() {
        setup_global_test_eop();
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.initial_epoch();

        let epochs = vec![initial_epoch, initial_epoch + 0.01];

        let traj = prop.states_ecef(&epochs);
        assert_eq!(traj.len(), 2);
        assert_eq!(traj.frame, OrbitFrame::ECEF);
    }

    #[test]
    fn test_sgppropagator_analyticpropagator_states_osculating_elements() {
        let prop = SGPPropagator::from_tle(ISS_LINE1, ISS_LINE2, 60.0).unwrap();
        let initial_epoch = prop.initial_epoch();

        let epochs = vec![initial_epoch, initial_epoch + 0.01];

        let traj = prop.states_osculating_elements(&epochs);
        assert_eq!(traj.len(), 2);
        assert_eq!(traj.representation, OrbitRepresentation::Keplerian);
        assert_eq!(traj.angle_format, AngleFormat::Radians);
    }
}
