/*!
 * Orbital trajectory implementation for 6-dimensional orbital state vectors.
 *
 * This module provides a specialized trajectory container for orbital mechanics applications,
 * wrapping `STrajectory<6>` with orbital-specific functionality including reference frame
 * conversions, state representation transformations, and angle format handling.
 *
 * # Key Features
 * - Reference frame conversions (ECI ↔ ECEF)
 * - State representation conversions (Cartesian ↔ Keplerian)
 * - Angle format conversions (Radians ↔ Degrees)
 * - Position and velocity extraction from Cartesian states
 * - Combined conversions for efficiency
 *
 * # Examples
 * ```rust
 * use brahe::trajectories::{OrbitTrajectory, OrbitFrame, OrbitRepresentation, AngleFormat};
 * use brahe::trajectories::{Trajectory, OrbitalTrajectory};
 * use brahe::time::{Epoch, TimeSystem};
 * use nalgebra::Vector6;
 *
 * // Create orbital trajectory in ECI Cartesian coordinates
 * let mut traj = OrbitTrajectory::new(
 *     OrbitFrame::ECI,
 *     OrbitRepresentation::Cartesian,
 *     AngleFormat::None,
 * ).unwrap();
 *
 * // Add state
 * let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
 * let state = Vector6::new(6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0);
 * traj.add_state(epoch, state).unwrap();
 *
 * // Convert to Keplerian in degrees
 * let kep_traj = traj.to_keplerian(AngleFormat::Degrees).unwrap();
 * ```
 */

use nalgebra::{SVector, Vector3, Vector6};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::time::Epoch;
use crate::utils::BraheError;
use crate::coordinates::{state_cartesian_to_osculating, state_osculating_to_cartesian};
use crate::frames::{state_eci_to_ecef, state_ecef_to_eci};
use crate::constants::{DEG2RAD, RAD2DEG};

use super::strajectory::STrajectory;
use super::traits::{Trajectory, Interpolatable, OrbitalTrajectory, InterpolationMethod, TrajectoryEvictionPolicy};

/// Enumeration of orbit reference frames
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrbitFrame {
    /// Earth-Centered Inertial frame (J2000)
    ECI,
    /// Earth-Centered Earth-Fixed frame
    ECEF,
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

/// Metadata keys for orbital trajectories stored in the generic metadata HashMap
pub const ORBITAL_FRAME_KEY: &str = "orbital_frame";
pub const ORBITAL_REPRESENTATION_KEY: &str = "orbital_representation";
pub const ORBITAL_ANGLE_FORMAT_KEY: &str = "orbital_angle_format";

/// Specialized orbital trajectory container.
///
/// This is a newtype wrapper around `STrajectory<6>` that provides orbital-specific
/// functionality including conversions between reference frames (ECI/ECEF), state
/// representations (Cartesian/Keplerian), and angle formats (radians/degrees).
///
/// The newtype pattern is used to provide a clean API while delegating most functionality
/// to the underlying `STrajectory<6>` implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct OrbitTrajectory(STrajectory<6>);

impl OrbitTrajectory {
    /// Creates a new orbital trajectory with specified frame, representation, and angle format.
    ///
    /// # Arguments
    /// * `frame` - Reference frame (ECI or ECEF)
    /// * `representation` - State representation (Cartesian or Keplerian)
    /// * `angle_format` - Angle format (None for Cartesian, Radians/Degrees for Keplerian)
    ///
    /// # Returns
    /// * `Ok(OrbitTrajectory)` - New empty orbital trajectory
    /// * `Err(BraheError)` - If parameters are invalid
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::{OrbitTrajectory, OrbitFrame, OrbitRepresentation, AngleFormat};
    ///
    /// let traj = OrbitTrajectory::new(
    ///     OrbitFrame::ECI,
    ///     OrbitRepresentation::Cartesian,
    ///     AngleFormat::None,
    /// ).unwrap();
    /// ```
    pub fn new(
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: AngleFormat,
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

        let mut metadata = HashMap::new();
        metadata.insert(ORBITAL_FRAME_KEY.to_string(), serde_json::to_value(frame).unwrap());
        metadata.insert(ORBITAL_REPRESENTATION_KEY.to_string(), serde_json::to_value(representation).unwrap());
        metadata.insert(ORBITAL_ANGLE_FORMAT_KEY.to_string(), serde_json::to_value(angle_format).unwrap());

        let mut inner = STrajectory::<6>::new();
        inner.metadata = metadata;

        Ok(OrbitTrajectory(inner))
    }

    /// Create orbital trajectory from data with specified orbital properties.
    ///
    /// # Arguments
    /// * `epochs` - Vector of epochs
    /// * `states` - Vector of state vectors
    /// * `frame` - Reference frame (ECI or ECEF)
    /// * `representation` - State representation (Cartesian or Keplerian)
    /// * `angle_format` - Angle format (None for Cartesian, Radians/Degrees for Keplerian)
    ///
    /// # Returns
    /// * `Ok(OrbitTrajectory)` - New orbital trajectory with data
    /// * `Err(BraheError)` - If parameters are invalid or data validation fails
    pub fn from_orbital_data(
        epochs: Vec<Epoch>,
        states: Vec<Vector6<f64>>,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: AngleFormat,
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

        let mut trajectory = STrajectory::<6>::from_data(epochs, states)?;
        trajectory.metadata.insert(ORBITAL_FRAME_KEY.to_string(), serde_json::to_value(frame).unwrap());
        trajectory.metadata.insert(ORBITAL_REPRESENTATION_KEY.to_string(), serde_json::to_value(representation).unwrap());
        trajectory.metadata.insert(ORBITAL_ANGLE_FORMAT_KEY.to_string(), serde_json::to_value(angle_format).unwrap());

        Ok(OrbitTrajectory(trajectory))
    }

    /// Helper to get orbital frame from metadata
    fn get_orbital_frame(&self) -> Result<OrbitFrame, BraheError> {
        self.0.metadata.get(ORBITAL_FRAME_KEY)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .ok_or_else(|| BraheError::Error("Not an orbital trajectory - missing frame metadata".to_string()))
    }

    /// Helper to get orbital representation from metadata
    fn get_orbital_representation(&self) -> Result<OrbitRepresentation, BraheError> {
        self.0.metadata.get(ORBITAL_REPRESENTATION_KEY)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .ok_or_else(|| BraheError::Error("Not an orbital trajectory - missing representation metadata".to_string()))
    }

    /// Helper to get angle format from metadata
    fn get_angle_format(&self) -> Result<AngleFormat, BraheError> {
        self.0.metadata.get(ORBITAL_ANGLE_FORMAT_KEY)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .ok_or_else(|| BraheError::Error("Not an orbital trajectory - missing angle format metadata".to_string()))
    }

    /// Get the state vector at a specific epoch using interpolation
    pub fn state_at_epoch(&self, epoch: &Epoch) -> Result<Vector6<f64>, BraheError> {
        self.0.state_at_epoch(epoch)
    }

    /// Sets the interpolation method using builder pattern.
    ///
    /// This method consumes self and returns a new trajectory with the specified
    /// interpolation method, allowing for method chaining.
    ///
    /// # Arguments
    /// * `interpolation_method` - Method to use for state interpolation between epochs
    ///
    /// # Returns
    /// Self with updated interpolation method
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::{OrbitTrajectory, OrbitFrame, OrbitRepresentation, AngleFormat, InterpolationMethod};
    /// let traj = OrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, AngleFormat::None).unwrap()
    ///     .with_interpolation_method(InterpolationMethod::Linear);
    /// ```
    pub fn with_interpolation_method(mut self, interpolation_method: InterpolationMethod) -> Self {
        self.0.interpolation_method = interpolation_method;
        self
    }

    /// Sets the eviction policy to keep a maximum number of states using builder pattern.
    ///
    /// This method consumes self and returns a new trajectory with the specified
    /// eviction policy, allowing for method chaining.
    ///
    /// # Arguments
    /// * `max_size` - Maximum number of states to retain (must be >= 1)
    ///
    /// # Returns
    /// Self with updated eviction policy
    ///
    /// # Panics
    /// Panics if max_size is less than 1
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::{OrbitTrajectory, OrbitFrame, OrbitRepresentation, AngleFormat};
    /// let traj = OrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, AngleFormat::None).unwrap()
    ///     .with_eviction_policy_max_size(100);
    /// ```
    pub fn with_eviction_policy_max_size(mut self, max_size: usize) -> Self {
        // Use the setter method which handles validation and applies eviction
        self.0.set_eviction_policy_max_size(max_size)
            .expect("Failed to set eviction policy");
        self
    }

    /// Sets the eviction policy to keep states within a maximum age using builder pattern.
    ///
    /// This method consumes self and returns a new trajectory with the specified
    /// eviction policy, allowing for method chaining.
    ///
    /// # Arguments
    /// * `max_age` - Maximum age of states to retain in seconds (must be > 0.0)
    ///
    /// # Returns
    /// Self with updated eviction policy
    ///
    /// # Panics
    /// Panics if max_age is not positive
    ///
    /// # Examples
    /// ```rust
    /// use brahe::trajectories::{OrbitTrajectory, OrbitFrame, OrbitRepresentation, AngleFormat};
    /// let traj = OrbitTrajectory::new(OrbitFrame::ECI, OrbitRepresentation::Cartesian, AngleFormat::None).unwrap()
    ///     .with_eviction_policy_max_age(3600.0);
    /// ```
    pub fn with_eviction_policy_max_age(mut self, max_age: f64) -> Self {
        // Use the setter method which handles validation and applies eviction
        self.0.set_eviction_policy_max_age(max_age)
            .expect("Failed to set eviction policy");
        self
    }

    /// Set eviction policy to keep a maximum number of states
    pub fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError> {
        self.0.set_eviction_policy_max_size(max_size)
    }

    /// Set eviction policy to keep states within a maximum age
    pub fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
        self.0.set_eviction_policy_max_age(max_age)
    }

    /// Get current state vector (most recent state in trajectory)
    pub fn current_state_vector(&self) -> Vector6<f64> {
        if let Some(last_state) = self.0.states.last() {
            *last_state
        } else {
            Vector6::zeros()
        }
    }

    /// Get current epoch (most recent epoch in trajectory)
    pub fn current_epoch(&self) -> Epoch {
        if let Some(last_epoch) = self.0.epochs.last() {
            *last_epoch
        } else {
            Epoch::from_jd(0.0, crate::time::TimeSystem::UTC)
        }
    }

    /// Get all epochs in the trajectory
    pub fn epochs(&self) -> &[Epoch] {
        &self.0.epochs
    }

    /// Convert the trajectory to a matrix representation
    /// Returns a matrix where columns are time points and rows are state elements
    pub fn to_matrix(&self) -> Result<nalgebra::DMatrix<f64>, BraheError> {
        self.0.to_matrix()
    }

    /// Convert state between different coordinate frames and representations
    pub fn convert_state_to_format(
        &self,
        state: SVector<f64, 6>,
        epoch: Epoch,
        from_frame: OrbitFrame,
        from_representation: OrbitRepresentation,
        from_angle_format: AngleFormat,
        to_frame: OrbitFrame,
        to_representation: OrbitRepresentation,
        to_angle_format: AngleFormat,
    ) -> Result<SVector<f64, 6>, BraheError> {
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
}

/// Index implementation returns state vector at given index
///
/// # Panics
/// Panics if index is out of bounds
impl std::ops::Index<usize> for OrbitTrajectory {
    type Output = Vector6<f64>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

/// Iterator over trajectory (epoch, state) pairs
pub struct OrbitTrajectoryIterator<'a> {
    trajectory: &'a OrbitTrajectory,
    index: usize,
}

impl<'a> Iterator for OrbitTrajectoryIterator<'a> {
    type Item = (Epoch, Vector6<f64>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.trajectory.len() {
            let result = self.trajectory.get(self.index).ok();
            self.index += 1;
            result
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.trajectory.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for OrbitTrajectoryIterator<'a> {
    fn len(&self) -> usize {
        self.trajectory.len() - self.index
    }
}

/// IntoIterator implementation for iterating over (epoch, state) pairs
impl<'a> IntoIterator for &'a OrbitTrajectory {
    type Item = (Epoch, Vector6<f64>);
    type IntoIter = OrbitTrajectoryIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        OrbitTrajectoryIterator {
            trajectory: self,
            index: 0,
        }
    }
}

// Passthrough implementations for Trajectory trait
impl Trajectory for OrbitTrajectory {
    type StateVector = Vector6<f64>;

    fn from_data(epochs: Vec<Epoch>, states: Vec<Self::StateVector>) -> Result<Self, BraheError> {
        // Create a basic STrajectory and wrap it
        // Note: This doesn't set orbital metadata, use `new()` followed by `add_state()` instead
        STrajectory::<6>::from_data(epochs, states).map(OrbitTrajectory)
    }

    fn add_state(&mut self, epoch: Epoch, state: Self::StateVector) -> Result<(), BraheError> {
        self.0.add_state(epoch, state)
    }

    fn state(&self, index: usize) -> Result<Self::StateVector, BraheError> {
        self.0.state(index)
    }

    fn epoch(&self, index: usize) -> Result<Epoch, BraheError> {
        self.0.epoch(index)
    }

    fn nearest_state(&self, epoch: &Epoch) -> Result<(Epoch, Self::StateVector), BraheError> {
        self.0.nearest_state(epoch)
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn start_epoch(&self) -> Option<Epoch> {
        self.0.start_epoch()
    }

    fn end_epoch(&self) -> Option<Epoch> {
        self.0.end_epoch()
    }

    fn timespan(&self) -> Option<f64> {
        self.0.timespan()
    }

    fn first(&self) -> Option<(Epoch, Self::StateVector)> {
        self.0.first()
    }

    fn last(&self) -> Option<(Epoch, Self::StateVector)> {
        self.0.last()
    }

    fn clear(&mut self) {
        self.0.clear()
    }

    fn remove_state(&mut self, epoch: &Epoch) -> Result<Self::StateVector, BraheError> {
        self.0.remove_state(epoch)
    }

    fn remove_state_at_index(&mut self, index: usize) -> Result<(Epoch, Self::StateVector), BraheError> {
        self.0.remove_state_at_index(index)
    }

    fn get(&self, index: usize) -> Result<(Epoch, Self::StateVector), BraheError> {
        self.0.get(index)
    }

    fn index_before_epoch(&self, epoch: &Epoch) -> Result<usize, BraheError> {
        self.0.index_before_epoch(epoch)
    }

    fn index_after_epoch(&self, epoch: &Epoch) -> Result<usize, BraheError> {
        self.0.index_after_epoch(epoch)
    }

    fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError> {
        self.0.set_eviction_policy_max_size(max_size)
    }

    fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
        self.0.set_eviction_policy_max_age(max_age)
    }

    fn get_eviction_policy(&self) -> TrajectoryEvictionPolicy {
        self.0.get_eviction_policy()
    }
}

// Passthrough implementations for Interpolatable trait
impl Interpolatable for OrbitTrajectory {
    fn set_interpolation_method(&mut self, method: InterpolationMethod) {
        self.0.set_interpolation_method(method)
    }

    fn get_interpolation_method(&self) -> InterpolationMethod {
        self.0.get_interpolation_method()
    }
}

// Implementation of OrbitalTrajectory trait
impl OrbitalTrajectory for OrbitTrajectory {
    fn to_frame(&self, target_frame: OrbitFrame) -> Result<Self, BraheError> {
        let frame = self.get_orbital_frame()?;
        let representation = self.get_orbital_representation()?;

        if frame == target_frame {
            return Ok(self.clone());
        }

        // Ensure we're working with Cartesian coordinates for frame transformations
        let cartesian_traj = if representation != OrbitRepresentation::Cartesian {
            self.to_cartesian()?
        } else {
            self.clone()
        };

        let mut new_trajectory = Self::new(
            target_frame,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        )?;
        new_trajectory.set_interpolation_method(self.get_interpolation_method());

        for (epoch, state) in cartesian_traj.0.epochs.iter().zip(cartesian_traj.0.states.iter()) {
            let cartesian_frame = cartesian_traj.get_orbital_frame()?;

            let transformed_state = match (cartesian_frame, target_frame) {
                (OrbitFrame::ECI, OrbitFrame::ECEF) => {
                    state_eci_to_ecef(*epoch, *state)
                }
                (OrbitFrame::ECEF, OrbitFrame::ECI) => {
                    state_ecef_to_eci(*epoch, *state)
                }
                _ => {
                    return Err(BraheError::Error(format!(
                        "Unsupported frame transformation: {:?} to {:?}",
                        cartesian_frame, target_frame
                    )));
                }
            };

            new_trajectory.add_state(*epoch, transformed_state)?;
        }

        Ok(new_trajectory)
    }

    fn to_representation(&self, target_representation: OrbitRepresentation,
                        target_angle_format: AngleFormat) -> Result<Self, BraheError> {
        let frame = self.get_orbital_frame()?;
        let representation = self.get_orbital_representation()?;
        let angle_format = self.get_angle_format()?;

        if representation == target_representation {
            // If same representation but different angle format, convert angles
            if target_representation == OrbitRepresentation::Keplerian && angle_format != target_angle_format {
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

        match (representation, target_representation) {
            (OrbitRepresentation::Cartesian, OrbitRepresentation::Keplerian) => {
                // For Cartesian to Keplerian conversion, we need to be in ECI frame
                let eci_traj = if frame != OrbitFrame::ECI {
                    self.to_eci()?
                } else {
                    self.clone()
                };

                let mut new_trajectory = Self::new(
                    OrbitFrame::ECI, // Keplerian elements are always in ECI
                    OrbitRepresentation::Keplerian,
                    target_angle_format,
                )?;
                new_trajectory.set_interpolation_method(self.get_interpolation_method());

                for (epoch, state) in eci_traj.0.epochs.iter().zip(eci_traj.0.states.iter()) {
                    let as_degrees = target_angle_format == AngleFormat::Degrees;
                    let keplerian_state = state_cartesian_to_osculating(*state, as_degrees);
                    new_trajectory.add_state(*epoch, keplerian_state)?;
                }

                Ok(new_trajectory)
            }
            (OrbitRepresentation::Keplerian, OrbitRepresentation::Cartesian) => {
                // Keplerian should already be in ECI frame
                if frame != OrbitFrame::ECI {
                    return Err(BraheError::Error(
                        "Keplerian elements should be in ECI frame".to_string(),
                    ));
                }

                let mut new_trajectory = Self::new(
                    OrbitFrame::ECI, // Convert to ECI, user can then convert to ECEF if needed
                    OrbitRepresentation::Cartesian,
                    AngleFormat::None,
                )?;
                new_trajectory.set_interpolation_method(self.get_interpolation_method());

                for (epoch, state) in self.0.epochs.iter().zip(self.0.states.iter()) {
                    let as_degrees = angle_format == AngleFormat::Degrees;
                    let cartesian_state = state_osculating_to_cartesian(*state, as_degrees);
                    new_trajectory.add_state(*epoch, cartesian_state)?;
                }

                Ok(new_trajectory)
            }
            _ => {
                Err(BraheError::Error(format!(
                    "Unsupported representation conversion: {:?} to {:?}",
                    representation, target_representation
                )))
            }
        }
    }

    fn to_angle_format(&self, target_format: AngleFormat) -> Result<Self, BraheError> {
        let frame = self.get_orbital_frame()?;
        let representation = self.get_orbital_representation()?;
        let angle_format = self.get_angle_format()?;

        if representation != OrbitRepresentation::Keplerian {
            return Err(BraheError::Error(
                "Angle format conversion only applies to Keplerian elements".to_string(),
            ));
        }

        if angle_format == target_format {
            return Ok(self.clone());
        }

        if target_format == AngleFormat::None {
            return Err(BraheError::Error(
                "Cannot convert Keplerian elements to None angle format".to_string(),
            ));
        }

        let conversion_factor = match (angle_format, target_format) {
            (AngleFormat::Radians, AngleFormat::Degrees) => RAD2DEG,
            (AngleFormat::Degrees, AngleFormat::Radians) => DEG2RAD,
            _ => {
                return Err(BraheError::Error(format!(
                    "Unsupported angle format conversion: {:?} to {:?}",
                    angle_format, target_format
                )));
            }
        };

        let mut new_trajectory = Self::new(
            frame,
            representation,
            target_format,
        )?;
        new_trajectory.set_interpolation_method(self.get_interpolation_method());

        for (epoch, state) in self.0.epochs.iter().zip(self.0.states.iter()) {
            let mut converted_state = *state;

            // Convert angular elements (i, Ω, ω, M) - elements 2-5
            for i in 2..6 {
                converted_state[i] = converted_state[i] * conversion_factor;
            }

            new_trajectory.add_state(*epoch, converted_state)?;
        }

        Ok(new_trajectory)
    }

    fn position_at_epoch(&self, epoch: &Epoch) -> Result<Vector3<f64>, BraheError> {
        let representation = self.get_orbital_representation()?;

        if representation != OrbitRepresentation::Cartesian {
            return Err(BraheError::Error(
                "Cannot extract position from non-Cartesian representation".to_string(),
            ));
        }

        let state = self.interpolate(epoch)?;
        Ok(Vector3::new(state[0], state[1], state[2]))
    }

    fn velocity_at_epoch(&self, epoch: &Epoch) -> Result<Vector3<f64>, BraheError> {
        let representation = self.get_orbital_representation()?;

        if representation != OrbitRepresentation::Cartesian {
            return Err(BraheError::Error(
                "Cannot extract velocity from non-Cartesian representation".to_string(),
            ));
        }

        let state = self.interpolate(epoch)?;
        Ok(Vector3::new(state[3], state[4], state[5]))
    }

    fn orbital_frame(&self) -> OrbitFrame {
        self.get_orbital_frame().unwrap_or(OrbitFrame::ECI)
    }

    fn orbital_representation(&self) -> OrbitRepresentation {
        self.get_orbital_representation().unwrap_or(OrbitRepresentation::Cartesian)
    }

    fn angle_format(&self) -> AngleFormat {
        self.get_angle_format().unwrap_or(AngleFormat::None)
    }

    fn convert_to(
        &self,
        target_frame: OrbitFrame,
        target_representation: OrbitRepresentation,
        target_angle_format: AngleFormat,
    ) -> Result<Self, BraheError> {
        let frame = self.get_orbital_frame()?;
        let representation = self.get_orbital_representation()?;
        let angle_format = self.get_angle_format()?;

        // Create new trajectory with target properties
        let mut new_trajectory = Self::new(
            target_frame,
            target_representation,
            target_angle_format,
        )?;
        new_trajectory.set_interpolation_method(self.get_interpolation_method());

        // Convert all states to the new format
        for (epoch, state) in self.0.epochs.iter().zip(self.0.states.iter()) {
            let converted_state = self.convert_state_to_format(
                *state,
                *epoch,
                frame,
                representation,
                angle_format,
                target_frame,
                target_representation,
                target_angle_format,
            )?;
            new_trajectory.add_state(*epoch, converted_state)?;
        }

        Ok(new_trajectory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::testing::setup_global_test_eop;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_orbittrajectory_new() {
        let traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        assert_eq!(traj.len(), 0);
        assert_eq!(traj.orbital_frame(), OrbitFrame::ECI);
        assert_eq!(traj.orbital_representation(), OrbitRepresentation::Cartesian);
        assert_eq!(traj.angle_format(), AngleFormat::None);
    }

    #[test]
    fn test_orbittrajectory_from_orbital_data() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
        ];

        let traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        assert_eq!(traj.len(), 2);
        assert_eq!(traj.orbital_frame(), OrbitFrame::ECI);
        assert_eq!(traj.orbital_representation(), OrbitRepresentation::Cartesian);
    }

    #[test]
    fn test_orbittrajectory_to_frame() {
        setup_global_test_eop();

        let mut eci_traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        eci_traj.add_state(epoch, state).unwrap();

        let ecef_traj = eci_traj.to_frame(OrbitFrame::ECEF).unwrap();
        assert_eq!(ecef_traj.orbital_frame(), OrbitFrame::ECEF);

        // Convert back
        let eci_traj2 = ecef_traj.to_frame(OrbitFrame::ECI).unwrap();
        assert_eq!(eci_traj2.orbital_frame(), OrbitFrame::ECI);
    }

    #[test]
    fn test_orbittrajectory_to_keplerian() {
        let mut cart_traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        cart_traj.add_state(epoch, state).unwrap();

        let kep_traj = cart_traj.to_keplerian(AngleFormat::Degrees).unwrap();
        assert_eq!(kep_traj.orbital_representation(), OrbitRepresentation::Keplerian);
        assert_eq!(kep_traj.angle_format(), AngleFormat::Degrees);
    }

    #[test]
    fn test_orbittrajectory_to_cartesian() {
        let mut kep_traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Radians,
        ).unwrap();

        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        // a, e, i, raan, argp, M
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 0.0, 0.0);
        kep_traj.add_state(epoch, state).unwrap();

        let cart_traj = kep_traj.to_cartesian().unwrap();
        assert_eq!(cart_traj.orbital_representation(), OrbitRepresentation::Cartesian);
        assert_eq!(cart_traj.angle_format(), AngleFormat::None);
    }

    #[test]
    fn test_orbittrajectory_position_velocity_at_epoch() {
        let mut cart_traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 1000e3, 2000e3, 100.0, 200.0, 300.0);
        cart_traj.add_state(epoch, state).unwrap();

        let pos = cart_traj.position_at_epoch(&epoch).unwrap();
        assert_abs_diff_eq!(pos[0], 7000e3, epsilon = 1.0);
        assert_abs_diff_eq!(pos[1], 1000e3, epsilon = 1.0);
        assert_abs_diff_eq!(pos[2], 2000e3, epsilon = 1.0);

        let vel = cart_traj.velocity_at_epoch(&epoch).unwrap();
        assert_abs_diff_eq!(vel[0], 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(vel[1], 200.0, epsilon = 1e-10);
        assert_abs_diff_eq!(vel[2], 300.0, epsilon = 1e-10);
    }

    // Additional Trajectory Trait Tests

    #[test]
    fn test_orbittrajectory_trajectory_add_state() {
        let mut traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        // Add states in order
        let epoch1 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state1 = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add_state(epoch1, state1).unwrap();

        let epoch3 = Epoch::from_jd(2451545.2, TimeSystem::UTC);
        let state3 = Vector6::new(7200e3, 0.0, 0.0, 0.0, 7.7e3, 0.0);
        traj.add_state(epoch3, state3).unwrap();

        // Add a state in between
        let epoch2 = Epoch::from_jd(2451545.1, TimeSystem::UTC);
        let state2 = Vector6::new(7100e3, 0.0, 0.0, 0.0, 7.6e3, 0.0);
        traj.add_state(epoch2, state2).unwrap();

        assert_eq!(traj.len(), 3);
        let epochs = traj.epochs();
        assert_eq!(epochs[0].jd(), 2451545.0);
        assert_eq!(epochs[1].jd(), 2451545.1);
        assert_eq!(epochs[2].jd(), 2451545.2);
    }

    #[test]
    fn test_orbittrajectory_trajectory_state_at_index() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
            Vector6::new(7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0),
        ];
        let traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        // Test valid indices
        let state0 = traj.state(0).unwrap();
        assert_eq!(state0[0], 7000e3);

        let state1 = traj.state(1).unwrap();
        assert_eq!(state1[0], 7100e3);

        let state2 = traj.state(2).unwrap();
        assert_eq!(state2[0], 7200e3);

        // Test invalid index
        assert!(traj.state(10).is_err());
    }

    #[test]
    fn test_orbittrajectory_trajectory_epoch_at_index() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
            Vector6::new(7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0),
        ];
        let traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        // Test valid indices
        let epoch0 = traj.epoch(0).unwrap();
        assert_eq!(epoch0.jd(), 2451545.0);

        let epoch1 = traj.epoch(1).unwrap();
        assert_eq!(epoch1.jd(), 2451545.1);

        let epoch2 = traj.epoch(2).unwrap();
        assert_eq!(epoch2.jd(), 2451545.2);

        // Test invalid index
        assert!(traj.epoch(10).is_err());
    }

    #[test]
    fn test_orbittrajectory_trajectory_nearest_state() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
            Vector6::new(7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0),
        ];
        let traj = OrbitTrajectory::from_orbital_data(
            epochs.clone(),
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        // Test finding nearest to exact epoch
        let query_epoch = epochs[1];
        let (nearest_epoch, nearest_state) = traj.nearest_state(&query_epoch).unwrap();
        assert_eq!(nearest_epoch, epochs[1]);
        assert_eq!(nearest_state[0], 7100e3);

        // Test finding nearest to mid-point (closer to second epoch)
        let mid_epoch = Epoch::from_jd(2451545.06, TimeSystem::UTC);
        let (nearest_epoch, _) = traj.nearest_state(&mid_epoch).unwrap();
        assert_eq!(nearest_epoch, epochs[1]);
    }

    #[test]
    fn test_orbittrajectory_trajectory_len() {
        let mut traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        assert_eq!(traj.len(), 0);
        assert!(traj.is_empty());

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add_state(epoch, state).unwrap();

        assert_eq!(traj.len(), 1);
        assert!(!traj.is_empty());
    }

    #[test]
    fn test_orbittrajectory_trajectory_is_empty() {
        let mut traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        assert!(traj.is_empty());

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add_state(epoch, state).unwrap();

        assert!(!traj.is_empty());
    }

    #[test]
    fn test_orbittrajectory_trajectory_start_epoch() {
        let mut traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        assert!(traj.start_epoch().is_none());

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add_state(epoch, state).unwrap();

        assert_eq!(traj.start_epoch().unwrap(), epoch);
    }

    #[test]
    fn test_orbittrajectory_trajectory_end_epoch() {
        let mut traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        assert!(traj.end_epoch().is_none());

        let epoch1 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let epoch2 = Epoch::from_jd(2451545.1, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add_state(epoch1, state).unwrap();
        traj.add_state(epoch2, state).unwrap();

        assert_eq!(traj.end_epoch().unwrap(), epoch2);
    }

    #[test]
    fn test_orbittrajectory_trajectory_timespan() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
        ];
        let traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        let timespan = traj.timespan().unwrap();
        assert_abs_diff_eq!(timespan, 0.1 * 86400.0, epsilon = 1e-5);
    }

    #[test]
    fn test_orbittrajectory_trajectory_first() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
        ];
        let traj = OrbitTrajectory::from_orbital_data(
            epochs.clone(),
            states.clone(),
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        let (first_epoch, first_state) = traj.first().unwrap();
        assert_eq!(first_epoch, epochs[0]);
        assert_eq!(first_state, states[0]);
    }

    #[test]
    fn test_orbittrajectory_trajectory_last() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
        ];
        let traj = OrbitTrajectory::from_orbital_data(
            epochs.clone(),
            states.clone(),
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        let (last_epoch, last_state) = traj.last().unwrap();
        assert_eq!(last_epoch, epochs[1]);
        assert_eq!(last_state, states[1]);
    }

    #[test]
    fn test_orbittrajectory_trajectory_clear() {
        let mut traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add_state(epoch, state).unwrap();

        assert_eq!(traj.len(), 1);
        traj.clear();
        assert_eq!(traj.len(), 0);
    }

    #[test]
    fn test_orbittrajectory_trajectory_remove_state() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
        ];
        let mut traj = OrbitTrajectory::from_orbital_data(
            epochs.clone(),
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        let removed_state = traj.remove_state(&epochs[0]).unwrap();
        assert_eq!(removed_state[0], 7000e3);
        assert_eq!(traj.len(), 1);
    }

    #[test]
    fn test_orbittrajectory_trajectory_remove_state_at_index() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
        ];
        let mut traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        let (removed_epoch, removed_state) = traj.remove_state_at_index(0).unwrap();
        assert_eq!(removed_epoch.jd(), 2451545.0);
        assert_eq!(removed_state[0], 7000e3);
        assert_eq!(traj.len(), 1);
    }

    #[test]
    fn test_orbittrajectory_trajectory_get() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
        ];
        let traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        let (epoch, state) = traj.get(1).unwrap();
        assert_eq!(epoch.jd(), 2451545.1);
        assert_eq!(state[0], 7100e3);
    }

    #[test]
    fn test_orbittrajectory_trajectory_index_before_epoch() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            Vector6::new(11.0, 12.0, 13.0, 14.0, 15.0, 16.0),
            Vector6::new(21.0, 22.0, 23.0, 24.0, 25.0, 26.0),
        ];

        let traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        // Test finding index before t0 (should error - before all states)
        let before_t0 = t0 - 10.0;
        assert!(traj.index_before_epoch(&before_t0).is_err());

        // Test finding index before t0+30s (should return index 0)
        let t0_plus_30 = t0 + 30.0;
        assert_eq!(traj.index_before_epoch(&t0_plus_30).unwrap(), 0);

        // Test finding index before t0+60s (should return index 1 - exact match)
        assert_eq!(traj.index_before_epoch(&t1).unwrap(), 1);

        // Test finding index before t0+90s (should return index 1)
        let t0_plus_90 = t0 + 90.0;
        assert_eq!(traj.index_before_epoch(&t0_plus_90).unwrap(), 1);

        // Test finding index before t0+120s (should return index 2 - exact match)
        assert_eq!(traj.index_before_epoch(&t2).unwrap(), 2);

        // Test finding index before t0+150s (should return index 2)
        let t0_plus_150 = t0 + 150.0;
        assert_eq!(traj.index_before_epoch(&t0_plus_150).unwrap(), 2);
    }

    #[test]
    fn test_orbittrajectory_trajectory_index_after_epoch() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            Vector6::new(11.0, 12.0, 13.0, 14.0, 15.0, 16.0),
            Vector6::new(21.0, 22.0, 23.0, 24.0, 25.0, 26.0),
        ];

        let traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        // Test finding index after t0-30s (should return index 0)
        let t0_minus_30 = t0 - 30.0;
        assert_eq!(traj.index_after_epoch(&t0_minus_30).unwrap(), 0);

        // Test finding index after t0 (should return index 0 - exact match)
        assert_eq!(traj.index_after_epoch(&t0).unwrap(), 0);

        // Test finding index after t0+30s (should return index 1)
        let t0_plus_30 = t0 + 30.0;
        assert_eq!(traj.index_after_epoch(&t0_plus_30).unwrap(), 1);

        // Test finding index after t0+60s (should return index 1 - exact match)
        assert_eq!(traj.index_after_epoch(&t1).unwrap(), 1);

        // Test finding index after t0+90s (should return index 2)
        let t0_plus_90 = t0 + 90.0;
        assert_eq!(traj.index_after_epoch(&t0_plus_90).unwrap(), 2);

        // Test finding index after t0+120s (should return index 2 - exact match)
        assert_eq!(traj.index_after_epoch(&t2).unwrap(), 2);

        // Test finding index after t0+150s (should error - after all states)
        let t0_plus_150 = t0 + 150.0;
        assert!(traj.index_after_epoch(&t0_plus_150).is_err());
    }

    #[test]
    fn test_orbittrajectory_trajectory_state_before_epoch() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            Vector6::new(11.0, 12.0, 13.0, 14.0, 15.0, 16.0),
            Vector6::new(21.0, 22.0, 23.0, 24.0, 25.0, 26.0),
        ];

        let traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        // Test that state_before_epoch returns correct (epoch, state) tuples
        let t0_plus_30 = t0 + 30.0;
        let (epoch, state) = traj.state_before_epoch(&t0_plus_30).unwrap();
        assert_eq!(epoch, t0);
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1e-10);

        let t0_plus_90 = t0 + 90.0;
        let (epoch, state) = traj.state_before_epoch(&t0_plus_90).unwrap();
        assert_eq!(epoch, t1);
        assert_abs_diff_eq!(state[0], 11.0, epsilon = 1e-10);

        // Test error case for epoch before all states
        let before_t0 = t0 - 10.0;
        assert!(traj.state_before_epoch(&before_t0).is_err());

        // Verify it uses the default trait implementation correctly
        let (epoch, state) = traj.state_before_epoch(&t2).unwrap();
        assert_eq!(epoch, t2);
        assert_abs_diff_eq!(state[0], 21.0, epsilon = 1e-10);
    }

    #[test]
    fn test_orbittrajectory_trajectory_state_after_epoch() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            Vector6::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            Vector6::new(11.0, 12.0, 13.0, 14.0, 15.0, 16.0),
            Vector6::new(21.0, 22.0, 23.0, 24.0, 25.0, 26.0),
        ];

        let traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        // Test that state_after_epoch returns correct (epoch, state) tuples
        let t0_plus_30 = t0 + 30.0;
        let (epoch, state) = traj.state_after_epoch(&t0_plus_30).unwrap();
        assert_eq!(epoch, t1);
        assert_abs_diff_eq!(state[0], 11.0, epsilon = 1e-10);

        let t0_plus_90 = t0 + 90.0;
        let (epoch, state) = traj.state_after_epoch(&t0_plus_90).unwrap();
        assert_eq!(epoch, t2);
        assert_abs_diff_eq!(state[0], 21.0, epsilon = 1e-10);

        // Test error case for epoch after all states
        let after_t2 = t2 + 10.0;
        assert!(traj.state_after_epoch(&after_t2).is_err());

        // Verify it uses the default trait implementation correctly
        let (epoch, state) = traj.state_after_epoch(&t0).unwrap();
        assert_eq!(epoch, t0);
        assert_abs_diff_eq!(state[0], 1.0, epsilon = 1e-10);
    }

    // Interpolatable Trait Tests

    #[test]
    fn test_orbittrajectory_interpolatable_set_interpolation_method() {
        let mut traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);

        traj.set_interpolation_method(InterpolationMethod::Linear);
        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);
    }

    #[test]
    fn test_orbittrajectory_interpolatable_get_interpolation_method() {
        let mut traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        // Test that get_interpolation_method returns Linear
        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);

        // Set it to different methods and verify get_interpolation_method returns the correct value



        traj.set_interpolation_method(InterpolationMethod::Linear);
        assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);
    }

    #[test]
    fn test_orbittrajectory_interpolatable_interpolate_linear() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            Vector6::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Vector6::new(60.0, 120.0, 180.0, 240.0, 300.0, 360.0),
            Vector6::new(120.0, 240.0, 360.0, 480.0, 600.0, 720.0),
        ];

        let traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        // Test interpolate_linear at midpoints and exact epochs
        let state_at_t0 = traj.interpolate_linear(&t0).unwrap();
        assert_abs_diff_eq!(state_at_t0[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_t0[1], 0.0, epsilon = 1e-10);

        let state_at_t1 = traj.interpolate_linear(&t1).unwrap();
        assert_abs_diff_eq!(state_at_t1[0], 60.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_t1[1], 120.0, epsilon = 1e-10);

        let state_at_t2 = traj.interpolate_linear(&t2).unwrap();
        assert_abs_diff_eq!(state_at_t2[0], 120.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_t2[1], 240.0, epsilon = 1e-10);

        // Test interpolation at midpoint between t0 and t1
        let t0_plus_30 = t0 + 30.0;
        let state_at_midpoint = traj.interpolate_linear(&t0_plus_30).unwrap();
        assert_abs_diff_eq!(state_at_midpoint[0], 30.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint[1], 60.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint[2], 90.0, epsilon = 1e-10);

        // Test interpolation at midpoint between t1 and t2
        let t1_plus_30 = t1 + 30.0;
        let state_at_midpoint2 = traj.interpolate_linear(&t1_plus_30).unwrap();
        assert_abs_diff_eq!(state_at_midpoint2[0], 90.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint2[1], 180.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_at_midpoint2[2], 270.0, epsilon = 1e-10);

        // Test edge case: single state trajectory
        let single_epoch = vec![t0];
        let single_state = vec![Vector6::new(100.0, 200.0, 300.0, 400.0, 500.0, 600.0)];
        let single_traj = OrbitTrajectory::from_orbital_data(
            single_epoch,
            single_state,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        let state_single = single_traj.interpolate_linear(&t0).unwrap();
        assert_abs_diff_eq!(state_single[0], 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state_single[1], 200.0, epsilon = 1e-10);
    }

    #[test]
    fn test_orbittrajectory_interpolatable_interpolate() {
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let t1 = t0 + 60.0;
        let t2 = t0 + 120.0;

        let epochs = vec![t0, t1, t2];
        let states = vec![
            Vector6::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            Vector6::new(60.0, 120.0, 180.0, 240.0, 300.0, 360.0),
            Vector6::new(120.0, 240.0, 360.0, 480.0, 600.0, 720.0),
        ];

        let traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        // Test that interpolate() with Linear method returns same result as interpolate_linear()
        let t0_plus_30 = t0 + 30.0;
        let state_interpolate = traj.interpolate(&t0_plus_30).unwrap();
        let state_interpolate_linear = traj.interpolate_linear(&t0_plus_30).unwrap();

        for i in 0..6 {
            assert_abs_diff_eq!(state_interpolate[i], state_interpolate_linear[i], epsilon = 1e-10);
        }
    }

    // OrbitalTrajectory Trait Tests - Angle Format Conversions

    #[test]
    fn test_orbittrajectory_orbitaltrajectory_to_angle_format() {
        // Create a Keplerian trajectory in radians
        let mut kep_rad_traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Radians,
        ).unwrap();

        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        // a, e, i, raan, argp, M (angles in radians)
        let state_rad = Vector6::new(7000e3, 0.0, 0.5, 1.0, 1.5, 2.0);
        kep_rad_traj.add_state(epoch, state_rad).unwrap();

        // Convert to degrees
        let kep_deg_traj = kep_rad_traj.to_angle_format(AngleFormat::Degrees).unwrap();
        assert_eq!(kep_deg_traj.angle_format(), AngleFormat::Degrees);

        let state_deg = kep_deg_traj.state(0).unwrap();
        assert_abs_diff_eq!(state_deg[0], 7000e3, epsilon = 1.0); // a unchanged
        assert_abs_diff_eq!(state_deg[1], 0.0, epsilon = 1e-10); // e unchanged
        assert_abs_diff_eq!(state_deg[2], 0.5 * RAD2DEG, epsilon = 1e-8); // i converted
        assert_abs_diff_eq!(state_deg[3], 1.0 * RAD2DEG, epsilon = 1e-8); // raan converted
        assert_abs_diff_eq!(state_deg[4], 1.5 * RAD2DEG, epsilon = 1e-8); // argp converted
        assert_abs_diff_eq!(state_deg[5], 2.0 * RAD2DEG, epsilon = 1e-8); // M converted

        // Convert back to radians
        let kep_rad_traj2 = kep_deg_traj.to_angle_format(AngleFormat::Radians).unwrap();
        assert_eq!(kep_rad_traj2.angle_format(), AngleFormat::Radians);

        let state_rad2 = kep_rad_traj2.state(0).unwrap();
        assert_abs_diff_eq!(state_rad2[2], 0.5, epsilon = 1e-8);
        assert_abs_diff_eq!(state_rad2[3], 1.0, epsilon = 1e-8);
        assert_abs_diff_eq!(state_rad2[4], 1.5, epsilon = 1e-8);
        assert_abs_diff_eq!(state_rad2[5], 2.0, epsilon = 1e-8);

        // Test error case: converting Cartesian should fail
        let mut cart_traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();
        cart_traj.add_state(epoch, Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0)).unwrap();
        assert!(cart_traj.to_angle_format(AngleFormat::Degrees).is_err());

        // Test error case: converting to None should fail
        assert!(kep_rad_traj.to_angle_format(AngleFormat::None).is_err());
    }

    #[test]
    fn test_orbittrajectory_orbitaltrajectory_to_degrees() {
        // Create a Keplerian trajectory in radians
        let mut kep_rad_traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Radians,
        ).unwrap();

        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state_rad = Vector6::new(7000e3, 0.0, 1.0, 2.0, 3.0, 4.0);
        kep_rad_traj.add_state(epoch, state_rad).unwrap();

        // Convert to degrees using convenience method
        let kep_deg_traj = kep_rad_traj.to_degrees().unwrap();
        assert_eq!(kep_deg_traj.angle_format(), AngleFormat::Degrees);

        let state_deg = kep_deg_traj.state(0).unwrap();
        assert_abs_diff_eq!(state_deg[2], 1.0 * RAD2DEG, epsilon = 1e-8);
        assert_abs_diff_eq!(state_deg[3], 2.0 * RAD2DEG, epsilon = 1e-8);
        assert_abs_diff_eq!(state_deg[4], 3.0 * RAD2DEG, epsilon = 1e-8);
        assert_abs_diff_eq!(state_deg[5], 4.0 * RAD2DEG, epsilon = 1e-8);
    }

    #[test]
    fn test_orbittrajectory_orbitaltrajectory_to_radians() {
        // Create a Keplerian trajectory in degrees
        let mut kep_deg_traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Degrees,
        ).unwrap();

        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state_deg = Vector6::new(7000e3, 0.0, 45.0, 90.0, 180.0, 270.0);
        kep_deg_traj.add_state(epoch, state_deg).unwrap();

        // Convert to radians using convenience method
        let kep_rad_traj = kep_deg_traj.to_radians().unwrap();
        assert_eq!(kep_rad_traj.angle_format(), AngleFormat::Radians);

        let state_rad = kep_rad_traj.state(0).unwrap();
        assert_abs_diff_eq!(state_rad[2], 45.0 * DEG2RAD, epsilon = 1e-8);
        assert_abs_diff_eq!(state_rad[3], 90.0 * DEG2RAD, epsilon = 1e-8);
        assert_abs_diff_eq!(state_rad[4], 180.0 * DEG2RAD, epsilon = 1e-8);
        assert_abs_diff_eq!(state_rad[5], 270.0 * DEG2RAD, epsilon = 1e-8);
    }

    #[test]
    fn test_orbittrajectory_orbitaltrajectory_convert_to() {
        setup_global_test_eop();

        // Create a trajectory in ECI Cartesian
        let mut eci_cart_traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        eci_cart_traj.add_state(epoch, state).unwrap();

        // Convert to ECEF Keplerian Degrees in one operation
        let ecef_kep_deg_traj = eci_cart_traj.convert_to(
            OrbitFrame::ECEF,
            OrbitRepresentation::Keplerian,
            AngleFormat::Degrees,
        ).unwrap();

        // Verify all properties changed
        assert_eq!(ecef_kep_deg_traj.orbital_frame(), OrbitFrame::ECEF);
        assert_eq!(ecef_kep_deg_traj.orbital_representation(), OrbitRepresentation::Keplerian);
        assert_eq!(ecef_kep_deg_traj.angle_format(), AngleFormat::Degrees);
        assert_eq!(ecef_kep_deg_traj.len(), 1);

        // Convert back to original format
        let eci_cart_traj2 = ecef_kep_deg_traj.convert_to(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        assert_eq!(eci_cart_traj2.orbital_frame(), OrbitFrame::ECI);
        assert_eq!(eci_cart_traj2.orbital_representation(), OrbitRepresentation::Cartesian);
        assert_eq!(eci_cart_traj2.angle_format(), AngleFormat::None);

        // States should be approximately the same after round-trip
        let original_state = eci_cart_traj.state(0).unwrap();
        let final_state = eci_cart_traj2.state(0).unwrap();
        for i in 0..6 {
            assert_abs_diff_eq!(original_state[i], final_state[i], epsilon = 1e-3);
        }
    }

    // Eviction Policy Tests

    #[test]
    fn test_orbittrajectory_set_eviction_policy_max_size() {
        let mut traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        // Add 5 states
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..5 {
            let epoch = t0 + (i as f64 * 60.0);
            let state = Vector6::new(7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0);
            traj.add_state(epoch, state).unwrap();
        }

        assert_eq!(traj.len(), 5);

        // Set max size to 3
        traj.set_eviction_policy_max_size(3).unwrap();

        // Should only have 3 most recent states
        assert_eq!(traj.len(), 3);

        // First state should be the 3rd original state (oldest 2 evicted)
        let first_state = traj.state(0).unwrap();
        assert_abs_diff_eq!(first_state[0], 7000e3 + 2000.0, epsilon = 1.0);

        // Add another state - should still maintain max size
        let new_epoch = t0 + 5.0 * 60.0;
        let new_state = Vector6::new(7000e3 + 5000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add_state(new_epoch, new_state).unwrap();

        assert_eq!(traj.len(), 3);

        // Test error case
        assert!(traj.set_eviction_policy_max_size(0).is_err());
    }

    #[test]
    fn test_orbittrajectory_set_eviction_policy_max_age() {
        let mut traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        // Add states spanning 5 minutes
        let t0 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        for i in 0..6 {
            let epoch = t0 + (i as f64 * 60.0); // 0, 60, 120, 180, 240, 300 seconds
            let state = Vector6::new(7000e3 + i as f64 * 1000.0, 0.0, 0.0, 0.0, 7.5e3, 0.0);
            traj.add_state(epoch, state).unwrap();
        }

        assert_eq!(traj.len(), 6);

        // Set max age to 150 seconds - should keep states within 150s of the last epoch
        traj.set_eviction_policy_max_age(150.0).unwrap();

        // Should keep states at 180s, 240s, and 300s (within 150s of 300s)
        assert_eq!(traj.len(), 3);

        let first_state = traj.state(0).unwrap();
        assert_abs_diff_eq!(first_state[0], 7000e3 + 3000.0, epsilon = 1.0);

        // Test error case
        assert!(traj.set_eviction_policy_max_age(0.0).is_err());
        assert!(traj.set_eviction_policy_max_age(-10.0).is_err());
    }

    #[test]
    fn test_orbittrajectory_current_state_vector() {
        let mut traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        // Add a single state
        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 100e3, 200e3, 1.0, 7.5e3, 0.5);
        traj.add_state(epoch, state).unwrap();

        // Verify current_state_vector returns the most recent state
        let current = traj.current_state_vector();
        assert_abs_diff_eq!(current, state, epsilon = 1e-6);

        // Add another state
        let epoch2 = epoch + 60.0;
        let state2 = Vector6::new(7100e3, 150e3, 250e3, 1.5, 7.6e3, 0.6);
        traj.add_state(epoch2, state2).unwrap();

        // Verify it returns the new most recent state
        let current = traj.current_state_vector();
        assert_abs_diff_eq!(current, state2, epsilon = 1e-6);
    }

    #[test]
    fn test_orbittrajectory_current_epoch() {
        let mut traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        // Add a single state
        let epoch1 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state1 = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add_state(epoch1, state1).unwrap();

        // Verify current_epoch returns the most recent epoch
        let current = traj.current_epoch();
        assert_eq!(current, epoch1);

        // Add another state
        let epoch2 = epoch1 + 60.0;
        let state2 = Vector6::new(7100e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        traj.add_state(epoch2, state2).unwrap();

        // Verify it returns the new most recent epoch
        let current = traj.current_epoch();
        assert_eq!(current, epoch2);
    }

    #[test]
    fn test_orbittrajectory_to_matrix() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
            Vector6::new(7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0),
        ];
        let traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states.clone(),
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        // Convert to matrix
        let matrix = traj.to_matrix().unwrap();

        // Verify dimensions: 6 rows (state elements) x 3 columns (time points)
        assert_eq!(matrix.nrows(), 6);
        assert_eq!(matrix.ncols(), 3);

        // Verify first column matches first state
        assert_abs_diff_eq!(matrix[(0, 0)], states[0][0], epsilon = 1e-6);
        assert_abs_diff_eq!(matrix[(1, 0)], states[0][1], epsilon = 1e-6);
        assert_abs_diff_eq!(matrix[(2, 0)], states[0][2], epsilon = 1e-6);

        // Verify second column matches second state
        assert_abs_diff_eq!(matrix[(0, 1)], states[1][0], epsilon = 1e-6);
        assert_abs_diff_eq!(matrix[(1, 1)], states[1][1], epsilon = 1e-6);
        assert_abs_diff_eq!(matrix[(2, 1)], states[1][2], epsilon = 1e-6);

        // Verify third column matches third state
        assert_abs_diff_eq!(matrix[(0, 2)], states[2][0], epsilon = 1e-6);
        assert_abs_diff_eq!(matrix[(1, 2)], states[2][1], epsilon = 1e-6);
        assert_abs_diff_eq!(matrix[(2, 2)], states[2][2], epsilon = 1e-6);
    }

    // Index Trait Tests

    #[test]
    fn test_orbittrajectory_index_index() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
            Vector6::new(7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0),
        ];
        let traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        // Test indexing returns state vectors
        let state0 = &traj[0];
        assert_eq!(state0[0], 7000e3);

        let state1 = &traj[1];
        assert_eq!(state1[0], 7100e3);

        let state2 = &traj[2];
        assert_eq!(state2[0], 7200e3);
    }

    #[test]
    #[should_panic]
    fn test_orbittrajectory_index_index_out_of_bounds() {
        let epochs = vec![Epoch::from_jd(2451545.0, TimeSystem::UTC)];
        let states = vec![Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0)];
        let traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        let _ = &traj[10]; // Should panic
    }

    // IntoIterator Trait Tests

    #[test]
    fn test_orbittrajectory_intoiterator_into_iter() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
            Vector6::new(7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0),
        ];
        let traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        let mut count = 0;
        for (epoch, state) in &traj {
            match count {
                0 => {
                    assert_eq!(epoch.jd(), 2451545.0);
                    assert_eq!(state[0], 7000e3);
                }
                1 => {
                    assert_eq!(epoch.jd(), 2451545.1);
                    assert_eq!(state[0], 7100e3);
                }
                2 => {
                    assert_eq!(epoch.jd(), 2451545.2);
                    assert_eq!(state[0], 7200e3);
                }
                _ => panic!("Too many iterations"),
            }
            count += 1;
        }
        assert_eq!(count, 3);
    }

    #[test]
    fn test_orbittrajectory_intoiterator_into_iter_empty() {
        let traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        let mut count = 0;
        for _ in &traj {
            count += 1;
        }
        assert_eq!(count, 0);
    }

    #[test]
    fn test_orbittrajectory_iterator_iterator_size_hint() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
            Vector6::new(7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0),
        ];
        let traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        let iter = traj.into_iter();
        let (lower, upper) = iter.size_hint();
        assert_eq!(lower, 3);
        assert_eq!(upper, Some(3));
    }

    #[test]
    fn test_orbittrajectory_iterator_iterator_len() {
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.1, TimeSystem::UTC),
            Epoch::from_jd(2451545.2, TimeSystem::UTC),
        ];
        let states = vec![
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
            Vector6::new(7200e3, 2000e3, 1000e3, 200.0, 7.7e3, 100.0),
        ];
        let traj = OrbitTrajectory::from_orbital_data(
            epochs,
            states,
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        ).unwrap();

        let iter = traj.into_iter();
        assert_eq!(iter.len(), 3);
    }
}
