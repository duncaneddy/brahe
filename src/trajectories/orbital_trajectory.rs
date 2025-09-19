/*!
 * Orbital trajectory implementation with frame and representation metadata.
 *
 * This module extends the base trajectory with orbital mechanics-specific properties
 * including reference frames (ECI, ECEF), coordinate representations (Cartesian, Keplerian),
 * and angle formats. It provides automatic coordinate transformations and maintains
 * physical consistency across different representations.
 *
 * # Coordinate Systems
 * - **ECI (Earth-Centered Inertial)**: J2000 inertial frame for space mechanics
 * - **ECEF (Earth-Centered Earth-Fixed)**: Rotating frame fixed to Earth's surface
 *
 * # State Representations
 * - **Cartesian**: Position and velocity vectors [x, y, z, vx, vy, vz] in meters and m/s
 * - **Keplerian**: Classical orbital elements [a, e, i, Ω, ω, M] in SI units and radians
 *
 * # Thread Safety
 * This module is not thread-safe. Use appropriate synchronization for concurrent access.
 */

use nalgebra::Vector6;
use serde::{Deserialize, Serialize};

use crate::time::Epoch;
use crate::utils::BraheError;
use crate::trajectories::{Trajectory6, InterpolationMethod};
use crate::coordinates::{state_cartesian_to_osculating, state_osculating_to_cartesian};
use crate::frames::{state_eci_to_ecef, state_ecef_to_eci};
use crate::constants::{DEG2RAD, RAD2DEG};

/// Trait representing a generic reference frame
pub trait ReferenceFrame: std::fmt::Debug + Clone + PartialEq {
    /// Get the name of the reference frame
    fn name(&self) -> &str;
}

/// Enumeration of orbit reference frames
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrbitFrame {
    /// Earth-Centered Inertial frame (J2000)
    ECI,
    /// Earth-Centered Earth-Fixed frame
    ECEF,
}

impl ReferenceFrame for OrbitFrame {
    fn name(&self) -> &str {
        match self {
            OrbitFrame::ECI => "Earth-Centered Inertial (J2000)",
            OrbitFrame::ECEF => "Earth-Centered Earth-Fixed",
        }
    }
}

/// Enumeration of orbit state representations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrbitRepresentation {
    /// Cartesian position and velocity (x, y, z, vx, vy, vz)
    Cartesian,
    /// Keplerian elements (a, e, i, Ω, ω, M)
    Keplerian,
}

/// Enumeration of angle formats for orbital elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AngleFormat {
    /// Angles represented in radians
    Radians,
    /// Angles represented in degrees
    Degrees,
    /// No angle representation or not applicable
    None,
}

/// Orbital trajectory with frame, representation, and angle format properties
#[derive(Debug, Clone, PartialEq)]
pub struct OrbitalTrajectory {
    /// Base trajectory containing epochs and states
    pub trajectory: Trajectory6,

    /// Reference frame of the trajectory
    pub frame: OrbitFrame,

    /// Representation type of the states
    pub representation: OrbitRepresentation,

    /// Format for angular quantities (only relevant for Keplerian)
    pub angle_format: AngleFormat,
}

impl OrbitalTrajectory {
    /// Create a new empty orbital trajectory
    pub fn new(
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: AngleFormat,
        interpolation_method: InterpolationMethod,
    ) -> Result<Self, BraheError> {
        // Validate angle format for representation
        if representation == OrbitRepresentation::Keplerian && angle_format == AngleFormat::None {
            return Err(BraheError::Error(
                "Angle format must be specified for Keplerian elements".to_string(),
            ));
        }

        if representation == OrbitRepresentation::Cartesian && angle_format != AngleFormat::None {
            return Err(BraheError::Error(
                "Angle format should be None for Cartesian representation".to_string(),
            ));
        }

        Ok(Self {
            trajectory: Trajectory6::with_interpolation(interpolation_method),
            frame,
            representation,
            angle_format,
        })
    }

    /// Create an orbital trajectory from data
    pub fn from_data(
        epochs: Vec<Epoch>,
        states: Vec<Vector6<f64>>,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: AngleFormat,
        interpolation_method: InterpolationMethod,
    ) -> Result<Self, BraheError> {
        // Validate inputs
        if representation == OrbitRepresentation::Keplerian && angle_format == AngleFormat::None {
            return Err(BraheError::Error(
                "Angle format must be specified for Keplerian elements".to_string(),
            ));
        }

        if representation == OrbitRepresentation::Cartesian && angle_format != AngleFormat::None {
            return Err(BraheError::Error(
                "Angle format should be None for Cartesian representation".to_string(),
            ));
        }

        let trajectory = Trajectory6::from_data(epochs, states, interpolation_method)?;

        Ok(Self {
            trajectory,
            frame,
            representation,
            angle_format,
        })
    }

    /// Add a state to the orbital trajectory
    pub fn add_state(&mut self, epoch: Epoch, state: Vector6<f64>) -> Result<(), BraheError> {
        self.trajectory.add_state(epoch, state)
    }

    /// Get the state at a specific epoch using interpolation
    pub fn state_at_epoch(&self, epoch: &Epoch) -> Result<Vector6<f64>, BraheError> {
        self.trajectory.state_at_epoch(epoch)
    }

    /// Find the nearest state to the specified epoch
    pub fn nearest_state(&self, epoch: &Epoch) -> Result<(Epoch, Vector6<f64>), BraheError> {
        self.trajectory.nearest_state(epoch)
    }

    /// Convert the trajectory to a different reference frame
    pub fn to_frame(&self, target_frame: OrbitFrame) -> Result<Self, BraheError> {
        if self.frame == target_frame {
            return Ok(self.clone());
        }

        // Ensure we're working with Cartesian coordinates for frame transformations
        let cartesian_traj = if self.representation != OrbitRepresentation::Cartesian {
            self.to_cartesian()?
        } else {
            self.clone()
        };

        let mut new_epochs = Vec::new();
        let mut new_states = Vec::new();

        for (epoch, state) in cartesian_traj.trajectory.epochs.iter().zip(cartesian_traj.trajectory.states.iter()) {
            let transformed_state = match (cartesian_traj.frame, target_frame) {
                (OrbitFrame::ECI, OrbitFrame::ECEF) => {
                    state_eci_to_ecef(*epoch, *state)
                }
                (OrbitFrame::ECEF, OrbitFrame::ECI) => {
                    state_ecef_to_eci(*epoch, *state)
                }
                _ => {
                    return Err(BraheError::Error(format!(
                        "Unsupported frame transformation: {:?} to {:?}",
                        cartesian_traj.frame, target_frame
                    )));
                }
            };

            new_epochs.push(*epoch);
            new_states.push(transformed_state);
        }

        Self::from_data(
            new_epochs,
            new_states,
            target_frame,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
            self.trajectory.interpolation_method,
        )
    }

    /// Convert to Earth-Centered Inertial (ECI) frame
    pub fn to_eci(&self) -> Result<Self, BraheError> {
        self.to_frame(OrbitFrame::ECI)
    }

    /// Convert to Earth-Centered Earth-Fixed (ECEF) frame
    pub fn to_ecef(&self) -> Result<Self, BraheError> {
        self.to_frame(OrbitFrame::ECEF)
    }

    /// Convert the trajectory to a different representation
    pub fn to_representation(&self, target_representation: OrbitRepresentation, target_angle_format: AngleFormat) -> Result<Self, BraheError> {
        if self.representation == target_representation {
            // If same representation but different angle format, convert angles
            if target_representation == OrbitRepresentation::Keplerian && self.angle_format != target_angle_format {
                return self.to_angle_format(target_angle_format);
            }
            return Ok(self.clone());
        }

        // Validate target parameters
        if target_representation == OrbitRepresentation::Keplerian && target_angle_format == AngleFormat::None {
            return Err(BraheError::Error(
                "Angle format must be specified for Keplerian elements".to_string(),
            ));
        }

        if target_representation == OrbitRepresentation::Cartesian && target_angle_format != AngleFormat::None {
            return Err(BraheError::Error(
                "Angle format should be None for Cartesian representation".to_string(),
            ));
        }

        match (self.representation, target_representation) {
            (OrbitRepresentation::Cartesian, OrbitRepresentation::Keplerian) => {
                // For Cartesian to Keplerian conversion, we need to be in ECI frame
                let eci_traj = if self.frame != OrbitFrame::ECI {
                    self.to_eci()?
                } else {
                    self.clone()
                };

                let mut new_epochs = Vec::new();
                let mut new_states = Vec::new();

                for (epoch, state) in eci_traj.trajectory.epochs.iter().zip(eci_traj.trajectory.states.iter()) {
                    let as_degrees = target_angle_format == AngleFormat::Degrees;
                    let keplerian_state = state_cartesian_to_osculating(*state, as_degrees);
                    new_epochs.push(*epoch);
                    new_states.push(keplerian_state);
                }

                Self::from_data(
                    new_epochs,
                    new_states,
                    OrbitFrame::ECI, // Keplerian elements are always in ECI
                    OrbitRepresentation::Keplerian,
                    target_angle_format,
                    self.trajectory.interpolation_method,
                )
            }
            (OrbitRepresentation::Keplerian, OrbitRepresentation::Cartesian) => {
                // Keplerian should already be in ECI frame
                if self.frame != OrbitFrame::ECI {
                    return Err(BraheError::Error(
                        "Keplerian elements should be in ECI frame".to_string(),
                    ));
                }

                let mut new_epochs = Vec::new();
                let mut new_states = Vec::new();

                for (epoch, state) in self.trajectory.epochs.iter().zip(self.trajectory.states.iter()) {
                    let as_degrees = self.angle_format == AngleFormat::Degrees;
                    let cartesian_state = state_osculating_to_cartesian(*state, as_degrees);
                    new_epochs.push(*epoch);
                    new_states.push(cartesian_state);
                }

                Self::from_data(
                    new_epochs,
                    new_states,
                    OrbitFrame::ECI, // Convert to ECI, user can then convert to ECEF if needed
                    OrbitRepresentation::Cartesian,
                    AngleFormat::None,
                    self.trajectory.interpolation_method,
                )
            }
            _ => {
                Err(BraheError::Error(format!(
                    "Unsupported representation conversion: {:?} to {:?}",
                    self.representation, target_representation
                )))
            }
        }
    }

    /// Convert to Cartesian representation
    pub fn to_cartesian(&self) -> Result<Self, BraheError> {
        self.to_representation(OrbitRepresentation::Cartesian, AngleFormat::None)
    }

    /// Convert to Keplerian elements
    pub fn to_keplerian(&self, angle_format: AngleFormat) -> Result<Self, BraheError> {
        if angle_format == AngleFormat::None {
            return Err(BraheError::Error(
                "Angle format must be specified when converting to Keplerian elements".to_string(),
            ));
        }

        self.to_representation(OrbitRepresentation::Keplerian, angle_format)
    }

    /// Convert the trajectory to a different angle format (only for Keplerian)
    pub fn to_angle_format(&self, target_format: AngleFormat) -> Result<Self, BraheError> {
        if self.representation != OrbitRepresentation::Keplerian {
            return Err(BraheError::Error(
                "Angle format conversion only applies to Keplerian elements".to_string(),
            ));
        }

        if self.angle_format == target_format {
            return Ok(self.clone());
        }

        if target_format == AngleFormat::None {
            return Err(BraheError::Error(
                "Cannot convert Keplerian elements to None angle format".to_string(),
            ));
        }

        let mut new_epochs = Vec::new();
        let mut new_states = Vec::new();

        let conversion_factor = match (self.angle_format, target_format) {
            (AngleFormat::Radians, AngleFormat::Degrees) => RAD2DEG,
            (AngleFormat::Degrees, AngleFormat::Radians) => DEG2RAD,
            _ => {
                return Err(BraheError::Error(format!(
                    "Unsupported angle format conversion: {:?} to {:?}",
                    self.angle_format, target_format
                )));
            }
        };

        for (epoch, state) in self.trajectory.epochs.iter().zip(self.trajectory.states.iter()) {
            let mut converted_state = *state;

            // Convert angular elements (i, Ω, ω, M) - elements 2-5
            for i in 2..6 {
                converted_state[i] = converted_state[i] * conversion_factor;
            }

            new_epochs.push(*epoch);
            new_states.push(converted_state);
        }

        Self::from_data(
            new_epochs,
            new_states,
            self.frame,
            self.representation,
            target_format,
            self.trajectory.interpolation_method,
        )
    }

    /// Convert to degrees representation (only for Keplerian)
    pub fn to_degrees(&self) -> Result<Self, BraheError> {
        self.to_angle_format(AngleFormat::Degrees)
    }

    /// Convert to radians representation (only for Keplerian)
    pub fn to_radians(&self) -> Result<Self, BraheError> {
        self.to_angle_format(AngleFormat::Radians)
    }

    /// Delegate methods to underlying trajectory
    pub fn len(&self) -> usize {
        self.trajectory.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trajectory.is_empty()
    }

    pub fn start_epoch(&self) -> Option<Epoch> {
        self.trajectory.start_epoch()
    }

    pub fn end_epoch(&self) -> Option<Epoch> {
        self.trajectory.end_epoch()
    }

    pub fn timespan(&self) -> Option<f64> {
        self.trajectory.timespan()
    }

    pub fn to_matrix(&self) -> Result<nalgebra::DMatrix<f64>, BraheError> {
        self.trajectory.to_matrix()
    }

    pub fn clear(&mut self) {
        self.trajectory.clear();
    }

    /// Get position component of the state if it's in Cartesian form
    pub fn position_at_epoch(&self, epoch: &Epoch) -> Result<nalgebra::Vector3<f64>, BraheError> {
        if self.representation != OrbitRepresentation::Cartesian {
            return Err(BraheError::Error(
                "Cannot extract position from non-Cartesian representation".to_string(),
            ));
        }

        let state = self.state_at_epoch(epoch)?;
        Ok(nalgebra::Vector3::new(state[0], state[1], state[2]))
    }

    /// Get velocity component of the state if it's in Cartesian form
    pub fn velocity_at_epoch(&self, epoch: &Epoch) -> Result<nalgebra::Vector3<f64>, BraheError> {
        if self.representation != OrbitRepresentation::Cartesian {
            return Err(BraheError::Error(
                "Cannot extract velocity from non-Cartesian representation".to_string(),
            ));
        }

        let state = self.state_at_epoch(epoch)?;
        Ok(nalgebra::Vector3::new(state[3], state[4], state[5]))
    }

    /// Set trajectory memory management parameters
    pub fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_size(max_size)
    }

    pub fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
        self.trajectory.set_eviction_policy_max_age(max_age)
    }

    /// Get current state vector (most recent state in trajectory)
    pub fn current_state_vector(&self) -> Vector6<f64> {
        if let Some(last_state) = self.trajectory.states.last() {
            *last_state
        } else {
            Vector6::zeros()
        }
    }

    /// Get current epoch (most recent epoch in trajectory)
    pub fn current_epoch(&self) -> Epoch {
        if let Some(last_epoch) = self.trajectory.epochs.last() {
            *last_epoch
        } else {
            Epoch::from_jd(0.0, crate::time::TimeSystem::UTC)
        }
    }

    /// Convert state between different coordinate frames and representations
    pub fn convert_state_to_format(
        &self,
        state: Vector6<f64>,
        epoch: Epoch,
        from_frame: OrbitFrame,
        from_representation: OrbitRepresentation,
        from_angle_format: AngleFormat,
        to_frame: OrbitFrame,
        to_representation: OrbitRepresentation,
        to_angle_format: AngleFormat,
    ) -> Result<Vector6<f64>, BraheError> {
        let mut converted_state = state;

        // Step 1: Convert to ECI Cartesian as intermediate format
        if from_frame != OrbitFrame::ECI || from_representation != OrbitRepresentation::Cartesian {
            // Convert representation first (if needed)
            if from_representation == OrbitRepresentation::Keplerian {
                let degrees = from_angle_format == AngleFormat::Degrees;
                converted_state = state_osculating_to_cartesian(converted_state, degrees);
            }

            // Convert frame (if needed)
            if from_frame == OrbitFrame::ECEF {
                converted_state = state_ecef_to_eci(epoch, converted_state);
            }
        }

        // Step 2: Convert from ECI Cartesian to target format
        if to_frame != OrbitFrame::ECI || to_representation != OrbitRepresentation::Cartesian {
            // Convert frame first (if needed)
            if to_frame == OrbitFrame::ECEF {
                converted_state = state_eci_to_ecef(epoch, converted_state);
            }

            // Convert representation (if needed)
            if to_representation == OrbitRepresentation::Keplerian {
                let degrees = to_angle_format == AngleFormat::Degrees;
                converted_state = state_cartesian_to_osculating(converted_state, degrees);
            }
        }

        Ok(converted_state)
    }

    /// Convert trajectory to different frame, representation, and angle format in one operation
    pub fn convert_to(
        &self,
        target_frame: OrbitFrame,
        target_representation: OrbitRepresentation,
        target_angle_format: AngleFormat,
    ) -> Result<Self, BraheError> {
        // Create new trajectory with target properties
        let mut new_trajectory = OrbitalTrajectory::new(
            target_frame,
            target_representation,
            target_angle_format,
            self.trajectory.interpolation_method,
        )?;

        // Convert all states to the new format
        for (epoch, state) in self.trajectory.epochs.iter().zip(self.trajectory.states.iter()) {
            let converted_state = self.convert_state_to_format(
                *state,
                *epoch,
                self.frame,
                self.representation,
                self.angle_format,
                target_frame,
                target_representation,
                target_angle_format,
            )?;
            new_trajectory.add_state(*epoch, converted_state)?;
        }

        Ok(new_trajectory)
    }

    /// Get all epochs in the trajectory
    pub fn epochs(&self) -> &[Epoch] {
        &self.trajectory.epochs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_orbital_trajectory_creation() {
        let trajectory = OrbitalTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
            InterpolationMethod::Linear,
        ).unwrap();

        assert_eq!(trajectory.frame, OrbitFrame::ECI);
        assert_eq!(trajectory.representation, OrbitRepresentation::Cartesian);
        assert_eq!(trajectory.angle_format, AngleFormat::None);
        assert!(trajectory.is_empty());
    }

    #[test]
    fn test_orbital_trajectory_validation() {
        // Should fail: Keplerian without angle format
        assert!(OrbitalTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::None,
            InterpolationMethod::Linear,
        ).is_err());

        // Should fail: Cartesian with angle format
        assert!(OrbitalTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::Degrees,
            InterpolationMethod::Linear,
        ).is_err());

        // Should now succeed: We can have Keplerian in ECEF frame (will convert via ECI)
        assert!(OrbitalTrajectory::new(
            OrbitFrame::ECEF,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
            InterpolationMethod::Linear,
        ).is_ok());
    }

    #[test]
    fn test_orbital_trajectory_with_data() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];

        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 0.0, 100.0, 7.6e3, 0.0),
        ];

        let trajectory = OrbitalTrajectory::from_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
            InterpolationMethod::Linear,
        ).unwrap();

        assert_eq!(trajectory.len(), 2);
        assert_eq!(trajectory.frame, OrbitFrame::ECI);
        assert_eq!(trajectory.representation, OrbitRepresentation::Cartesian);
    }

    #[test]
    fn test_orbital_trajectory_position_velocity() {
        let epochs = vec![Epoch::from_jd(2451545.0, TimeSystem::UTC)];
        let states = vec![Vector6::new(7000e3, 1000e3, 2000e3, 100.0, 200.0, 300.0)];

        let trajectory = OrbitalTrajectory::from_data(
            epochs.clone(),
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
            InterpolationMethod::Linear,
        ).unwrap();

        let position = trajectory.position_at_epoch(&epochs[0]).unwrap();
        let velocity = trajectory.velocity_at_epoch(&epochs[0]).unwrap();

        assert_eq!(position.x, 7000e3);
        assert_eq!(position.y, 1000e3);
        assert_eq!(position.z, 2000e3);

        assert_eq!(velocity.x, 100.0);
        assert_eq!(velocity.y, 200.0);
        assert_eq!(velocity.z, 300.0);
    }

    #[test]
    fn test_orbital_trajectory_angle_format_conversion() {
        let epochs = vec![Epoch::from_jd(2451545.0, TimeSystem::UTC)];
        let states = vec![Vector6::new(7000e3, 0.01, 1.0, 2.0, 3.0, 4.0)]; // angles in radians

        let trajectory_rad = OrbitalTrajectory::from_data(
            epochs.clone(),
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Radians,
            InterpolationMethod::Linear,
        ).unwrap();

        let trajectory_deg = trajectory_rad.to_degrees().unwrap();

        assert_eq!(trajectory_deg.angle_format, AngleFormat::Degrees);

        let state_deg = trajectory_deg.state_at_epoch(&epochs[0]).unwrap();

        // Semi-major axis and eccentricity should be unchanged
        assert_eq!(state_deg[0], 7000e3);
        assert_eq!(state_deg[1], 0.01);

        // Angular elements should be converted to degrees
        assert_abs_diff_eq!(state_deg[2], 1.0 * RAD2DEG, epsilon = 1e-10);
        assert_abs_diff_eq!(state_deg[3], 2.0 * RAD2DEG, epsilon = 1e-10);
        assert_abs_diff_eq!(state_deg[4], 3.0 * RAD2DEG, epsilon = 1e-10);
        assert_abs_diff_eq!(state_deg[5], 4.0 * RAD2DEG, epsilon = 1e-10);
    }

    #[test]
    fn test_ecef_to_keplerian_conversion() {
        // Test that we can convert ECEF Cartesian to Keplerian via ECI
        let epochs = vec![Epoch::from_jd(2451545.0, TimeSystem::UTC)];
        let states = vec![Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0)];

        let ecef_trajectory = OrbitalTrajectory::from_data(
            epochs.clone(),
            states,
            OrbitFrame::ECEF,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
            InterpolationMethod::Linear,
        ).unwrap();

        // This should work: ECEF Cartesian → ECI Cartesian → ECI Keplerian
        let keplerian_trajectory = ecef_trajectory.to_keplerian(AngleFormat::Radians).unwrap();

        assert_eq!(keplerian_trajectory.frame, OrbitFrame::ECI);
        assert_eq!(keplerian_trajectory.representation, OrbitRepresentation::Keplerian);
        assert_eq!(keplerian_trajectory.angle_format, AngleFormat::Radians);
    }
}