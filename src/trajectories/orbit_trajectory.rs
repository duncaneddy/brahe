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

use nalgebra::{SVector, Vector6};
use serde_json::Value;
use std::collections::HashMap;

use crate::time::Epoch;
use crate::utils::BraheError;
use crate::coordinates::{state_cartesian_to_osculating, state_osculating_to_cartesian};
use crate::frames::{state_eci_to_ecef, state_ecef_to_eci};
use crate::constants::{DEG2RAD, RAD2DEG};

use super::traits::{Trajectory, Interpolatable, OrbitalTrajectory, InterpolationMethod, TrajectoryEvictionPolicy, AngleFormat, OrbitFrame, OrbitRepresentation};

/// Specialized orbital trajectory container.
///
/// This is a newtype wrapper around `STrajectory<6>` that provides orbital-specific
/// functionality including conversions between reference frames (ECI/ECEF), state
/// representations (Cartesian/Keplerian), and angle formats (radians/degrees).
///
/// The newtype pattern is used to provide a clean API while delegating most functionality
/// to the underlying `STrajectory<6>` implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct OrbitTrajectory{
        /// Time epochs for each state, maintained in chronological order.
    /// All epochs should use consistent time systems for meaningful interpolation.
    pub epochs: Vec<Epoch>,

    /// R-dimensional state vectors corresponding to epochs.
    /// Units and interpretation depend on the specific use case:
    /// - Cartesian: [m, m, m, m/s, m/s, m/s]
    /// - Keplerian: [m, dimensionless, rad, rad, rad, rad]
    pub states: Vec<SVector<f64, 6>>,

    /// Interpolation method for state retrieval at arbitrary epochs.
    /// Default is linear interpolation for optimal performance/accuracy balance.
    pub interpolation_method: InterpolationMethod,

    /// Memory management policy for automatic state eviction.
    /// Controls how states are removed when limits are exceeded.
    pub eviction_policy: TrajectoryEvictionPolicy,

    /// Maximum number of states to retain (for KeepCount policy).
    max_size: Option<usize>,

    /// Maximum age of states to retain in seconds (for KeepWithinDuration policy).
    max_age: Option<f64>,

    /// Reference frame of the orbital states (ECI or ECEF).
    pub frame: OrbitFrame,

    /// State representation (Cartesian or Keplerian).
    /// Keplerian elements are always in ECI frame.
    /// Cartesian can be in ECI or ECEF.
    pub representation: OrbitRepresentation,

    /// Angle format for angular elements
    /// None is used for Cartesian representation.
    /// Radians or Degrees can be used for Keplerian elements.
    pub angle_format: AngleFormat,

    /// Generic metadata storage supporting arbitrary key-value pairs.
    /// Can store any JSON-serializable data including strings, numbers, booleans,
    /// arrays, and nested objects. For orbital trajectories, use ORBITAL_*_KEY constants.
    pub metadata: HashMap<String, Value>,
}

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
    ) -> Self {
        // Validate angle format for representation
        if representation == OrbitRepresentation::Keplerian && angle_format == AngleFormat::None {
            panic!("Angle format must be specified for Keplerian elements");
        }

        if representation == OrbitRepresentation::Cartesian && angle_format != AngleFormat::None {
            panic!("Angle format should be None for Cartesian representation");
        }

        if frame == OrbitFrame::ECEF && representation == OrbitRepresentation::Keplerian {
            panic!("Keplerian elements should be in ECI frame");
        }

        Self {
            epochs: Vec::new(),
            states: Vec::new(),
            interpolation_method: InterpolationMethod::Linear,
            eviction_policy: TrajectoryEvictionPolicy::None,
            max_size: None,
            max_age: None,
            frame: frame,
            representation: representation,
            angle_format: angle_format,
            metadata: HashMap::new(),
        }
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
        self.interpolation_method = interpolation_method;
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
        if max_size < 1 {
            panic!("Maximum size must be >= 1");
        }
        self.eviction_policy = TrajectoryEvictionPolicy::KeepCount;
        self.max_size = Some(max_size);
        self.max_age = None;
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
        if max_age <= 0.0 {
            panic!("Maximum age must be > 0.0");
        }
        self.eviction_policy = TrajectoryEvictionPolicy::KeepWithinDuration;
        self.max_age = Some(max_age);
        self.max_size = None;
        self
    }

    pub fn dimension(&self) -> usize {
        6
    }

    /// Convert the trajectory to a matrix representation
    /// Returns a matrix where columns are time points and rows are state elements
    /// The matrix has shape (6, n_epochs) for a 6-element state vector
    pub fn to_matrix(&self) -> Result<nalgebra::DMatrix<f64>, BraheError> {
        if self.states.is_empty() {
            return Err(BraheError::Error(
                "Cannot convert empty trajectory to matrix".to_string(),
            ));
        }

        let n_epochs = self.states.len();
        let n_elements = 6;

        let mut matrix = nalgebra::DMatrix::<f64>::zeros(n_elements, n_epochs);

        for (col_idx, state) in self.states.iter().enumerate() {
            for row_idx in 0..n_elements {
                matrix[(row_idx, col_idx)] = state[row_idx];
            }
        }

        Ok(matrix)
    }

    fn apply_eviction_policy(&mut self) -> Result<(), BraheError> {
        match self.eviction_policy {
            TrajectoryEvictionPolicy::None => {
                // No eviction
            },
            TrajectoryEvictionPolicy::KeepCount => {
                if let Some(max_size) = self.max_size {
                    if self.epochs.len() > max_size {
                        let to_remove = self.epochs.len() - max_size;
                        self.epochs.drain(0..to_remove);
                        self.states.drain(0..to_remove);
                    }
                }
            },
            TrajectoryEvictionPolicy::KeepWithinDuration => {
                if let Some(max_age) = self.max_age {
                    if let Some(&last_epoch) = self.epochs.last() {
                        let mut indices_to_keep = Vec::new();
                        for (i, &epoch) in self.epochs.iter().enumerate() {
                            if (last_epoch - epoch).abs() <= max_age {
                                indices_to_keep.push(i);
                            }
                        }

                        let new_epochs: Vec<Epoch> = indices_to_keep.iter().map(|&i| self.epochs[i]).collect();
                        let new_states: Vec<SVector<f64, 6>> = indices_to_keep.iter().map(|&i| self.states[i].clone()).collect();

                        self.epochs = new_epochs;
                        self.states = new_states;
                    }
                }
            },
        }
        Ok(())
    }

    /// Internal method for converting a single state vector between formats.
    pub(crate) fn convert_state_to_format(
        &self,
        epoch: Epoch,
        state: SVector<f64, 6>,
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
                // Immediately return so you can't request an ECEF Keplerian state
                return Ok(state_eci_to_ecef(epoch, converted_state));
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

impl Default for OrbitTrajectory {

    /// Creates a default orbital trajectory in ECI Cartesian with no angle format.
    fn default() -> Self {
        Self::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        )
    }
}

/// Index implementation returns state vector at given index
///
/// # Panics
/// Panics if index is out of bounds
impl std::ops::Index<usize> for OrbitTrajectory {
    type Output = SVector<f64, 6>;

    fn index(&self, index: usize) -> &Self::Output {
       &self.states[index]
    }
}

/// Iterator over trajectory (epoch, state) pairs
pub struct OrbitTrajectoryIterator<'a> {
    trajectory: &'a OrbitTrajectory,
    index: usize,
}

impl<'a> Iterator for OrbitTrajectoryIterator<'a> {
    type Item = (Epoch, SVector<f64, 6>);

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
    type Item = (Epoch, SVector<f64, 6>);
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

    /// Create trajectory from data. Assumes all data is in ECI Cartesian format.
    fn from_data(epochs: Vec<Epoch>, states: Vec<Self::StateVector>) -> Result<Self, BraheError> {
        if epochs.len() != states.len() {
            return Err(BraheError::Error(
                "Epochs and states vectors must have the same length".to_string(),
            ));
        }

        if epochs.is_empty() {
            return Err(BraheError::Error(
                "Cannot create trajectory from empty data".to_string(),
            ));
        }

        // Ensure epochs are sorted
        let mut indices: Vec<usize> = (0..epochs.len()).collect();
        indices.sort_by(|&i, &j| epochs[i].partial_cmp(&epochs[j]).unwrap());

        let sorted_epochs: Vec<Epoch> = indices.iter().map(|&i| epochs[i]).collect();
        let sorted_states: Vec<SVector<f64, 6>> = indices.iter().map(|&i| states[i].clone()).collect();

        Ok(Self {
            epochs: sorted_epochs,
            states: sorted_states,
            interpolation_method: InterpolationMethod::Linear,  // Default to Linear
            eviction_policy: TrajectoryEvictionPolicy::None,
            max_size: None,
            max_age: None,
            frame: OrbitFrame::ECI,  // Default to ECI Cartesian
            representation: OrbitRepresentation::Cartesian,
            angle_format: AngleFormat::None,
            metadata: HashMap::new(),
        })
    }

    fn add_state(&mut self, epoch: Epoch, state: Self::StateVector) -> Result<(), BraheError> {
        // Find the correct position to insert based on epoch
        let mut insert_idx = self.epochs.len();
        for (i, existing_epoch) in self.epochs.iter().enumerate() {
            if epoch < *existing_epoch {
                insert_idx = i;
                break;
            } else if epoch == *existing_epoch {
                // Replace state if epochs are equal
                self.states[i] = state;
                self.apply_eviction_policy()?;
                return Ok(());
            }
        }

        // Insert at the correct position
        self.epochs.insert(insert_idx, epoch);
        self.states.insert(insert_idx, state);

        // Apply eviction policy after adding state
        self.apply_eviction_policy()?;
        Ok(())
    }

    fn epoch(&self, index: usize) -> Result<Epoch, BraheError> {
        if index >= self.epochs.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} epochs",
                index,
                self.epochs.len()
            )));
        }

        Ok(self.epochs[index])
    }

    fn state(&self, index: usize) -> Result<Self::StateVector, BraheError> {
        if index >= self.states.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            )));
        }

        Ok(self.states[index].clone())
    }


    fn nearest_state(&self, epoch: &Epoch) -> Result<(Epoch, Self::StateVector), BraheError> {
        if self.epochs.is_empty() {
            return Err(BraheError::Error(
                "Cannot find nearest state in empty trajectory".to_string(),
            ));
        }

        let mut nearest_idx = 0;
        let mut min_diff = f64::MAX;

        for (i, existing_epoch) in self.epochs.iter().enumerate() {
            let diff = (*epoch - *existing_epoch).abs();
            if diff < min_diff {
                min_diff = diff;
                nearest_idx = i;
            }

            // Optimization: if we're past the epoch and moving away, we can stop
            if i > 0 && existing_epoch > epoch && diff > min_diff {
                break;
            }
        }

        Ok((self.epochs[nearest_idx], self.states[nearest_idx].clone()))
    }

    fn len(&self) -> usize {
        self.states.len()
    }

    fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    fn start_epoch(&self) -> Option<Epoch> {
        self.epochs.first().copied()
    }

    fn end_epoch(&self) -> Option<Epoch> {
        self.epochs.last().copied()
    }

    fn timespan(&self) -> Option<f64> {
        if self.epochs.len() < 2 {
            return None;
        } else {
            Some(*self.epochs.last().unwrap() - *self.epochs.first().unwrap())
        }
    }

    fn first(&self) -> Option<(Epoch, Self::StateVector)> {
        if self.epochs.is_empty() {
            None
        } else {
            Some((self.epochs[0], self.states[0].clone()))
        }
    }

    fn last(&self) -> Option<(Epoch, Self::StateVector)> {
        if self.epochs.is_empty() {
            None
        } else {
            let last_index = self.epochs.len() - 1;
            Some((self.epochs[last_index], self.states[last_index].clone()))
        }
    }

    fn clear(&mut self) {
        self.epochs.clear();
        self.states.clear();
    }

    fn remove_state(&mut self, epoch: &Epoch) -> Result<Self::StateVector, BraheError> {
        // This could be improved with binary search since epochs are sorted
        if let Some(index) = self.epochs.iter().position(|e| e == epoch) {
            let removed_state = self.states.remove(index);
            self.epochs.remove(index);
            Ok(removed_state)
        } else {
            Err(BraheError::Error(
                "Epoch not found in trajectory".to_string(),
            ))
        }
    }

    fn remove_state_at_index(&mut self, index: usize) -> Result<(Epoch, Self::StateVector), BraheError> {
        if index >= self.states.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            )));
        }

        let removed_epoch = self.epochs.remove(index);
        let removed_state = self.states.remove(index);
        Ok((removed_epoch, removed_state))
    }

    fn get(&self, index: usize) -> Result<(Epoch, Self::StateVector), BraheError> {
        if index >= self.states.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            )));
        }

        Ok((self.epochs[index], self.states[index].clone()))
    }

    fn index_before_epoch(&self, epoch: &Epoch) -> Result<usize, BraheError> {
        if self.epochs.is_empty() {
            return Err(BraheError::Error(
                "Cannot get index from empty trajectory".to_string(),
            ));
        }

        // If epoch is before the first state, error
        if epoch < &self.epochs[0] {
            return Err(BraheError::Error(
                "Epoch is before all states in trajectory".to_string(),
            ));
        }

        // Find the index at or before the epoch
        for i in (0..self.epochs.len()).rev() {
            if &self.epochs[i] <= epoch {
                return Ok(i);
            }
        }

        // Should never reach here given the checks above
        Err(BraheError::Error(
            "Failed to find index before epoch".to_string(),
        ))
    }

    fn index_after_epoch(&self, epoch: &Epoch) -> Result<usize, BraheError> {
        if self.epochs.is_empty() {
            return Err(BraheError::Error(
                "Cannot get index from empty trajectory".to_string(),
            ));
        }

        // If epoch is after the last state, error
        if epoch > self.epochs.last().unwrap() {
            return Err(BraheError::Error(
                "Epoch is after all states in trajectory".to_string(),
            ));
        }

        // Find the index at or after the epoch
        for i in 0..self.epochs.len() {
            if &self.epochs[i] >= epoch {
                return Ok(i);
            }
        }

        // Should never reach here given the checks above
        Err(BraheError::Error(
            "Failed to find index after epoch".to_string(),
        ))
    }

    fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError> {
        if max_size < 1 {
            return Err(BraheError::Error(
                "Maximum size must be >= 1".to_string(),
            ));
        }
        self.eviction_policy = TrajectoryEvictionPolicy::KeepCount;
        self.max_size = Some(max_size);
        self.max_age = None;
        self.apply_eviction_policy()?;
        Ok(())
    }

    fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError> {
        if max_age <= 0.0 {
            return Err(BraheError::Error(
                "Maximum age must be > 0.0".to_string(),
            ));
        }
        self.eviction_policy = TrajectoryEvictionPolicy::KeepWithinDuration;
        self.max_age = Some(max_age);
        self.max_size = None;
        self.apply_eviction_policy()?;
        Ok(())
    }

    fn get_eviction_policy(&self) -> TrajectoryEvictionPolicy {
        self.eviction_policy
    }
}

// Passthrough implementations for Interpolatable trait
impl Interpolatable for OrbitTrajectory {
    fn set_interpolation_method(&mut self, method: InterpolationMethod) {
        self.interpolation_method = method;
    }

    fn get_interpolation_method(&self) -> InterpolationMethod {
        self.interpolation_method
    }
}

// Implementation of OrbitalTrajectory trait
impl OrbitalTrajectory for OrbitTrajectory {
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
    fn from_orbital_data(
        epochs: Vec<Epoch>,
        states: Vec<Vector6<f64>>,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: AngleFormat,
    ) -> Self {
        // Validate inputs
        if representation == OrbitRepresentation::Keplerian && angle_format == AngleFormat::None {
            panic!("Angle format must be specified for Keplerian elements");
        }

        if representation == OrbitRepresentation::Cartesian && angle_format != AngleFormat::None {
            panic!("Angle format should be None for Cartesian representation");
        }

        if frame == OrbitFrame::ECEF && representation == OrbitRepresentation::Keplerian {
            panic!("Keplerian elements should be in ECI frame");
        }

        Self {
            epochs,
            states,
            interpolation_method: InterpolationMethod::Linear,
            eviction_policy: TrajectoryEvictionPolicy::None,
            max_size: None,
            max_age: None,
            frame,
            representation,
            angle_format,
            metadata: HashMap::new(),
        }
    }

    fn to_eci(&self) -> Self where Self: Sized {
        if self.frame == OrbitFrame::ECI {
            // Already in ECI frame
            return self.clone();
        }

        // We know we're doing a conversion, so prepare new states vector
        let mut states_converted = Vec::with_capacity(self.states.len());

        if self.representation == OrbitRepresentation::Keplerian {
            // Keplerian to Cartesian first
            for (e, s) in self.into_iter() {
                let state_cartesian = state_osculating_to_cartesian(s, self.angle_format == AngleFormat::Degrees);
                let state_eci = state_ecef_to_eci(e, state_cartesian);
                states_converted.push(state_eci);
            }
        } else {
            // ECEF Cartesian to ECI Cartesian
            for (e, s) in self.into_iter() {
                let state_eci = state_ecef_to_eci(e, s);
                states_converted.push(state_eci);
            }
        }

        Self {
            epochs: self.epochs.clone(),
            states: states_converted,
            interpolation_method: self.interpolation_method,
            eviction_policy: self.eviction_policy,
            max_size: self.max_size,
            max_age: self.max_age,
            frame: OrbitFrame::ECI,
            representation: OrbitRepresentation::Cartesian,
            angle_format: AngleFormat::None,
            metadata: self.metadata.clone(),
        }
    }

    fn to_ecef(&self) -> Self where Self: Sized {
        if self.frame == OrbitFrame::ECEF {
            // Already in ECEF frame
            return self.clone();
        }

        // We know we're doing a conversion, so prepare new states vector
        let mut states_converted = Vec::with_capacity(self.states.len());

        if self.representation == OrbitRepresentation::Keplerian {
            // Keplerian to Cartesian first
            for (e, s) in self.into_iter() {
                let state_cartesian = state_osculating_to_cartesian(s, self.angle_format == AngleFormat::Degrees);
                let state_ecef = state_eci_to_ecef(e, state_cartesian);
                states_converted.push(state_ecef);
            }
        } else {
            // ECI Cartesian to ECEF Cartesian
            for (e, s) in self.into_iter() {
                let state_ecef = state_eci_to_ecef(e, s);
                states_converted.push(state_ecef);
            }
        }

        Self {
            epochs: self.epochs.clone(),
            states: states_converted,
            interpolation_method: self.interpolation_method,
            eviction_policy: self.eviction_policy,
            max_size: self.max_size,
            max_age: self.max_age,
            frame: OrbitFrame::ECEF,
            representation: OrbitRepresentation::Cartesian,
            angle_format: AngleFormat::None,
            metadata: self.metadata.clone(),
        }
    }

    fn to_keplerian(&self, angle_format: AngleFormat) -> Self where Self: Sized {
        if self.representation == OrbitRepresentation::Keplerian && self.angle_format == angle_format {
            // Already in desired format
            return self.clone();
        }

        // We know we're doing a conversion, so prepare new states vector
        let mut states_converted = Vec::with_capacity(self.states.len());

        // If Keplerian, but wrong angle format, convert angles
        if self.representation == OrbitRepresentation::Keplerian {
            for (_e, s) in self.into_iter() {
                let mut state_converted = s;
                if self.angle_format == AngleFormat::Degrees && angle_format == AngleFormat::Radians {
                    // Degrees to Radians
                    state_converted[2] *= DEG2RAD;
                    state_converted[3] *= DEG2RAD;
                    state_converted[4] *= DEG2RAD;
                    state_converted[5] *= DEG2RAD;
                } else if self.angle_format == AngleFormat::Radians && angle_format == AngleFormat::Degrees {
                    // Radians to Degrees
                    state_converted[2] *= RAD2DEG;
                    state_converted[3] *= RAD2DEG;
                    state_converted[4] *= RAD2DEG;
                    state_converted[5] *= RAD2DEG;
                }
                states_converted.push(state_converted);
            }
        }

        // If ECEF, convert to ECI first
        if self.frame == OrbitFrame::ECEF {
            for (e, s) in self.into_iter() {
                let state_eci = state_ecef_to_eci(e, s);
                let state_kep = state_cartesian_to_osculating(state_eci, angle_format == AngleFormat::Degrees);
                states_converted.push(state_kep);
            }
        } else {
            // ECI Cartesian to Keplerian
            for (_e, s) in self.into_iter() {
                let state_kep = state_cartesian_to_osculating(s, angle_format == AngleFormat::Degrees);
                states_converted.push(state_kep);
            }
        }

        Self {
            epochs: self.epochs.clone(),
            states: states_converted,
            interpolation_method: self.interpolation_method,
            eviction_policy: self.eviction_policy,
            max_size: self.max_size,
            max_age: self.max_age,
            frame: OrbitFrame::ECI,
            representation: OrbitRepresentation::Keplerian,
            angle_format,
            metadata: self.metadata.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::R_EARTH;
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::testing::setup_global_test_eop;
    use approx::assert_abs_diff_eq;

    fn create_test_trajectory() -> OrbitTrajectory {
        let mut traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Degrees,
        ).unwrap();

        let epoch1 = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state1 = Vector6::new(R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0);
        traj.add_state(epoch1, state1).unwrap();

        let epoch2 = Epoch::from_datetime(2023, 1, 1, 12, 10, 0.0, 0.0, TimeSystem::UTC);
        let state2 = Vector6::new(R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 60.0);
        traj.add_state(epoch2, state2).unwrap();

        let epoch3 = Epoch::from_datetime(2023, 1, 1, 12, 20, 0.0, 0.0, TimeSystem::UTC);
        let state3 = Vector6::new(R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 75.0);
        traj.add_state(epoch3, state3).unwrap();

        traj
    }

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
    fn test_orbittrajectory_new_invalid() {
        // Invalid: Keplerian with None angle format
        let result = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::None,
        );
        assert!(result.is_err());

        // Invalid: Cartesian with Degrees angle format
        let result = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::Degrees,
        );
        assert!(result.is_err());

        // Invalid: Cartesian with Radians angle format
        let result = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::Radians,
        );
        assert!(result.is_err());

        // Invalid: Keplerian in ECEF frame
        let result = OrbitTrajectory::new(
            OrbitFrame::ECEF,
            OrbitRepresentation::Keplerian,
            AngleFormat::Degrees,
        );
        assert!(result.is_err());

        let result = OrbitTrajectory::new(
            OrbitFrame::ECEF,
            OrbitRepresentation::Keplerian,
            AngleFormat::Radians,
        );
        assert!(result.is_err());

        let result = OrbitTrajectory::new(
            OrbitFrame::ECEF,
            OrbitRepresentation::Keplerian,
            AngleFormat::None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_orbittrajetory_dimension() {
        let traj = create_test_trajectory();
        assert_eq!(traj.dimension(), 6);
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
        assert_eq!(matrix[(0, 0)], states[0][0]);
        assert_eq!(matrix[(1, 0)], states[0][1]);
        assert_eq!(matrix[(2, 0)], states[0][2]);
        assert_eq!(matrix[(3, 0)], states[0][3]);
        assert_eq!(matrix[(4, 0)], states[0][4]);
        assert_eq!(matrix[(5, 0)], states[0][5]);

        // Verify second column matches second state
        assert_eq!(matrix[(0, 1)], states[1][0]);
        assert_eq!(matrix[(1, 1)], states[1][1]);
        assert_eq!(matrix[(2, 1)], states[1][2]);
        assert_eq!(matrix[(3, 1)], states[1][3]);
        assert_eq!(matrix[(4, 1)], states[1][4]);
        assert_eq!(matrix[(5, 1)], states[1][5]);

        // Verify third column matches third state
        assert_eq!(matrix[(0, 2)], states[2][0]);
        assert_eq!(matrix[(1, 2)], states[2][1]);
        assert_eq!(matrix[(2, 2)], states[2][2]);
        assert_eq!(matrix[(3, 2)], states[2][3]);
        assert_eq!(matrix[(4, 2)], states[2][4]);
        assert_eq!(matrix[(5, 2)], states[2][5]);
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

        // Test before first epoch
        let test_epoch = Epoch::from_jd(2451544.9, TimeSystem::UTC);
        let (nearest_epoch, nearest_state) = traj.nearest_state(&test_epoch).unwrap();
        assert_eq!(nearest_epoch, epochs[0]);
        assert_eq!(nearest_state[0], 7000e3);

        // Test after last epoch
        let test_epoch = Epoch::from_jd(2451545.3, TimeSystem::UTC);
        let (nearest_epoch, nearest_state) = traj.nearest_state(&test_epoch).unwrap();
        assert_eq!(nearest_epoch, epochs[2]);
        assert_eq!(nearest_state[0], 7200e3);   

        // Test between epochs
        let test_epoch = Epoch::from_jd(2451545.15, TimeSystem::UTC);
        let (nearest_epoch, nearest_state) = traj.nearest_state(&test_epoch).unwrap();
        assert_eq!(nearest_epoch, epochs[1]);
        assert_eq!(nearest_state[0], 7100e3);

        // Test exact match
        let test_epoch = Epoch::from_jd(2451545.1, TimeSystem::UTC);
        let (nearest_epoch, nearest_state) = traj.nearest_state(&test_epoch).unwrap();
        assert_eq!(nearest_epoch, epochs[1]);
        assert_eq!(nearest_state[0], 7100e3);

        // Test just before second epoch
        let test_epoch = Epoch::from_jd(2451545.0999, TimeSystem::UTC);
        let (nearest_epoch, nearest_state) = traj.nearest_state(&test_epoch).unwrap();
        assert_eq!(nearest_epoch, epochs[1]);
        assert_eq!(nearest_state[1], 7100e3);
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

    #[test]
    fn test_orbittrajectory_trajectory_set_eviction_policy_max_size() {
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
    fn test_orbittrajectory_trajectory_set_eviction_policy_max_age() {
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

        // Set max age to 240 seconds 
        traj.set_eviction_policy_max_age(240.0).unwrap();
        assert_eq!(traj.len(), 4);

        let first_state = traj.state(0).unwrap();
        assert_abs_diff_eq!(first_state[0], 7000e3 + 1000.0, epsilon = 1.0);

        // Set max age to 239 seconds
        traj.set_eviction_policy_max_age(239.0).unwrap();

        assert_eq!(traj.len(), 3);
        let first_state = traj.state(0).unwrap();
        assert_abs_diff_eq!(first_state[0], 7000e3 + 2000.0, epsilon = 1.0);

        // Test error case
        assert!(traj.set_eviction_policy_max_age(0.0).is_err());
        assert!(traj.set_eviction_policy_max_age(-10.0).is_err());
    }

    // Default Trait Tests

    #[test]
    fn test_orbittrajectory_default() {
        let traj = OrbitTrajectory::default();
        assert_eq!(traj.len(), 0);
        assert!(traj.is_empty());
        assert_eq!(traj.frame(), OrbitFrame::ECI);
        assert_eq!(traj.representation(), OrbitRepresentation::Cartesian);
        assert_eq!(traj.angle_format(), AngleFormat::None);
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

    // OrbitalTrajectory Trait Tests

    #[test]
    fn test_orbittrajectory_orbitaltrajectory_from_orbital_data() {
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
    fn test_orbittrajectory_orbitaltrajectory_to_eci() {
        let state_base = state_osculating_to_cartesian(na::SVector6::new(
           R_EARTH + 500e3, 0.01, 97.0, 15.0, 30.0, 45.0
        ), true).unwrap();

        // No transformation needed if already in ECI
        let mut traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        );

        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        traj.add_state(epoch, state_base).unwrap();
        
        let eci_traj = traj.to_eci();
        assert_eq!(eci_traj.frame, OrbitFrame::ECI);
        assert_eq!(eci_traj.representation, OrbitRepresentation::Cartesian);
        assert_eq!(eci_traj.angle_format, AngleFormat::None);
        assert_eq!(eci_traj.len(), 1);
        let (epoch_out, state_out) = eci_traj.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = 1e-9);
        }

        // Convert Keplerian to ECI - Radians
        let mut kep_traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Radians,
        );
        let kep_state_rad = state_cartesian_to_osculating(state_base, false);
        kep_traj.add_state(epoch, kep_state_rad).unwrap();

        let eci_from_kep_rad = kep_traj.to_eci();
        assert_eq!(eci_from_kep_rad.frame, OrbitFrame::ECI);
        assert_eq!(eci_from_kep_rad.representation, OrbitRepresentation::Cartesian);
        assert_eq!(eci_from_kep_rad.angle_format, AngleFormat::None);
        assert_eq!(eci_from_kep_rad.len(), 1);
        let (epoch_out, state_out) = eci_from_kep_rad.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = 1e-9);
        }

        // Convert Keplerian to ECI - Degrees
        let mut kep_traj_deg = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Degrees,
        );
        let kep_state_deg = state_cartesian_to_osculating(state_base, true);
        kep_traj_deg.add_state(epoch, kep_state_deg).unwrap();
        let eci_from_kep_deg = kep_traj_deg.to_eci().unwrap();
        assert_eq!(eci_from_kep_deg.frame, OrbitFrame::ECI);
        assert_eq!(eci_from_kep_deg.representation, OrbitRepresentation::Cartesian);
        assert_eq!(eci_from_kep_deg.angle_format, AngleFormat::None);
        assert_eq!(eci_from_kep_deg.len(), 1);
        let (epoch_out, state_out) = eci_from_kep_deg.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = 1e-9);
        }

        // Convert ECEF to ECI
        let mut ecef_traj = OrbitTrajectory::new(
            OrbitFrame::ECEF,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        );
        let ecef_state = state_eci_to_ecef(epoch, state_base);
        ecef_traj.add_state(epoch, ecef_state).unwrap();
        let eci_from_ecef = ecef_traj.to_eci();
        assert_eq!(eci_from_ecef.frame, OrbitFrame::ECI);
        assert_eq!(eci_from_ecef.representation, OrbitRepresentation::Cartesian);
        assert_eq!(eci_from_ecef.angle_format, AngleFormat::None);
        assert_eq!(eci_from_ecef.len(), 1);
        let (epoch_out, state_out) = eci_from_ecef.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_orbittrajectory_orbitaltrajectory_to_ecef() {
        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state_base = state_eci_to_ecef(epoch, state_osculating_to_cartesian(na::SVector6::new(
           R_EARTH + 500e3, 0.01, 97.0, 15.0, 30.0, 45.0
        ), true).unwrap());

        // No transformation needed if already in ECEF
        let mut traj = OrbitTrajectory::new(
            OrbitFrame::ECEF,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        );

        traj.add_state(epoch, state_base).unwrap();
        let ecef_traj = traj.to_ecef();
        assert_eq!(ecef_traj.frame, OrbitFrame::ECEF);
        assert_eq!(ecef_traj.representation, OrbitRepresentation::Cartesian);
        assert_eq!(ecef_traj.angle_format, AngleFormat::None);
        assert_eq!(ecef_traj.len(), 1);
        let (epoch_out, state_out) = ecef_traj.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = 1e-9);
        }

        // Convert ECI to ECEF
        let mut eci_traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        );
        let eci_state = state_ecef_to_eci(epoch, state_base);
        eci_traj.add_state(epoch, eci_state).unwrap();
        let ecef_from_eci = eci_traj.to_ecef();
        assert_eq!(ecef_from_eci.frame, OrbitFrame::ECEF);
        assert_eq!(ecef_from_eci.representation, OrbitRepresentation::Cartesian);
        assert_eq!(ecef_from_eci.angle_format, AngleFormat::None);
        assert_eq!(ecef_from_eci.len(), 1);
        let (epoch_out, state_out) = ecef_from_eci.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = 1e-9);
        }

        // Convert Keplerian to ECEF - Radians
        let mut kep_traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Radians,
        );
        let kep_state_rad = state_cartesian_to_osculating(eci_state, false);
        kep_traj.add_state(epoch, kep_state_rad).unwrap();
        let ecef_from_kep_rad = kep_traj.to_ecef();
        assert_eq!(ecef_from_kep_rad.frame, OrbitFrame::ECEF);
        assert_eq!(ecef_from_kep_rad.representation, OrbitRepresentation::Cartesian);
        assert_eq!(ecef_from_kep_rad.angle_format, AngleFormat::None);
        assert_eq!(ecef_from_kep_rad.len(), 1);
        let (epoch_out, state_out) = ecef_from_kep_rad.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = 1e-9);
        }

        // Convert Keplerian to ECEF - Degrees
        let mut kep_traj_deg = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Degrees,
        );
        let kep_state_deg = state_cartesian_to_osculating(eci_state, true);
        kep_traj_deg.add_state(epoch, kep_state_deg).unwrap();
        let ecef_from_kep_deg = kep_traj_deg.to_ecef();
        assert_eq!(ecef_from_kep_deg.frame, OrbitFrame::ECEF);
        assert_eq!(ecef_from_kep_deg.representation, OrbitRepresentation::Cartesian);
        assert_eq!(ecef_from_kep_deg.angle_format, AngleFormat::None);
        assert_eq!(ecef_from_kep_deg.len(), 1);
        let (epoch_out, state_out) = ecef_from_kep_deg.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_base[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_orbittrajectory_orbitaltrajectory_to_keplerian_deg() {
        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let state_kep_deg = na::SVector6::new(7000e3, 0.01, 97.0, 15.0, 30.0, 45.0);

        // No transformation needed if already in Keplerian Degrees
        let mut traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Degrees,
        );
        traj.add_state(epoch, state_kep_deg).unwrap();
        let kep_traj = traj.to_keplerian(AngleFormat::Degrees);
        assert_eq!(kep_traj.frame, OrbitFrame::ECI);
        assert_eq!(kep_traj.representation, OrbitRepresentation::Keplerian);
        assert_eq!(kep_traj.angle_format, AngleFormat::Degrees);
        assert_eq!(kep_traj.len(), 1);
        let (epoch_out, state_out) = kep_traj.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_kep_deg[i], epsilon = 1e-9);
        }

        // Convert Keplerian Radians to Keplerian Degrees
        let mut kep_rad_traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Radians,
        );
        let mut state_kep_rad = state_kep_deg.clone();
        for i in 2..6 {
            state_kep_rad[i] = state_kep_deg[i] * DEG2RAD;
        }
        kep_rad_traj.add_state(epoch, state_kep_rad).unwrap();
        let kep_from_rad = kep_rad_traj.to_keplerian(AngleFormat::Degrees);
        assert_eq!(kep_from_rad.frame, OrbitFrame::ECI);
        assert_eq!(kep_from_rad.representation, OrbitRepresentation::Keplerian);
        assert_eq!(kep_from_rad.angle_format, AngleFormat::Degrees);
        assert_eq!(kep_from_rad.len(), 1);
        let (epoch_out, state_out) = kep_from_rad.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_kep_deg[i], epsilon = 1e-9);
        }

        // Convert ECI to Keplerian Degrees
        let mut cart_traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        );
        let cart_state = state_osculating_to_cartesian(state_kep_deg, true);
        cart_traj.add_state(epoch, cart_state).unwrap();
        let kep_from_cart = cart_traj.to_keplerian(AngleFormat::Degrees);
        assert_eq!(kep_from_cart.frame, OrbitFrame::ECI);
        assert_eq!(kep_from_cart.representation, OrbitRepresentation::Keplerian);
        assert_eq!(kep_from_cart.angle_format, AngleFormat::Degrees);
        assert_eq!(kep_from_cart.len(), 1);
        let (epoch_out, state_out) = kep_from_cart.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_kep_deg[i], epsilon = 1e-9);
        }

        // Convert ECEF to Keplerian Degrees
        let mut ecef_traj = OrbitTrajectory::new(
            OrbitFrame::ECEF,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        );
        let ecef_state = state_eci_to_ecef(epoch, cart_state);
        ecef_traj.add_state(epoch, ecef_state).unwrap();
        let kep_from_ecef = ecef_traj.to_keplerian(AngleFormat::Degrees);
        assert_eq!(kep_from_ecef.frame, OrbitFrame::ECI);
        assert_eq!(kep_from_ecef.representation, OrbitRepresentation::Keplerian);
        assert_eq!(kep_from_ecef.angle_format, AngleFormat::Degrees);
        assert_eq!(kep_from_ecef.len(), 1);
        let (epoch_out, state_out) = kep_from_ecef.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_kep_deg[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn test_orbittrajectory_orbitaltrajectory_to_keplerian_rad() {
        let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let mut state_kep_rad = na::SVector6::new(7000e3, 0.01, 97.0, 15.0, 30.0, 45.0);
        for i in 2..6 {
            state_kep_rad[i] = state_kep_deg[i] * DEG2RAD;
        }

        // No transformation needed if already in Keplerian Radians
        let mut traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Radians,
        );
        traj.add_state(epoch, state_kep_deg).unwrap();
        let kep_traj = traj.to_keplerian(AngleFormat::Radians);
        assert_eq!(kep_traj.frame, OrbitFrame::ECI);
        assert_eq!(kep_traj.representation, OrbitRepresentation::Keplerian);
        assert_eq!(kep_traj.angle_format, AngleFormat::Radians);
        assert_eq!(kep_traj.len(), 1);
        let (epoch_out, state_out) = kep_traj.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_kep_rad[i], epsilon = 1e-9);
        }

        // Convert Keplerian Degrees to Keplerian Radians
        let mut kep_deg_traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            AngleFormat::Degrees,
        );
        let mut state_kep_deg = state_kep_rad.clone();
        for i in 2..6 {
            state_kep_deg[i] = state_kep_deg[i] * RAD2DEG;
        }
        kep_deg_traj.add_state(epoch, state_kep_deg).unwrap();
        let kep_from_deg = kep_deg_traj.to_keplerian(AngleFormat::Radians);
        assert_eq!(kep_from_deg.frame, OrbitFrame::ECI);
        assert_eq!(kep_from_deg.representation, OrbitRepresentation::Keplerian);
        assert_eq!(kep_from_deg.angle_format, AngleFormat::Radians);
        assert_eq!(kep_from_deg.len(), 1);
        let (epoch_out, state_out) = kep_from_deg.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_kep_rad[i], epsilon = 1e-9);
        }

        // Convert ECI to Keplerian Radians
        let mut cart_traj = OrbitTrajectory::new(
            OrbitFrame::ECI,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        );
        let cart_state = state_osculating_to_cartesian(state_kep_deg, true);
        cart_traj.add_state(epoch, cart_state).unwrap();
        let kep_from_cart = cart_traj.to_keplerian(AngleFormat::Radians);
        assert_eq!(kep_from_cart.frame, OrbitFrame::ECI);
        assert_eq!(kep_from_cart.representation, OrbitRepresentation::Keplerian);
        assert_eq!(kep_from_cart.angle_format, AngleFormat::Radians);
        assert_eq!(kep_from_cart.len(), 1);
        let (epoch_out, state_out) = kep_from_cart.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_kep_rad[i], epsilon = 1e-9);
        }

        // Convert ECEF to Keplerian Radians
        let mut ecef_traj = OrbitTrajectory::new(
            OrbitFrame::ECEF,
            OrbitRepresentation::Cartesian,
            AngleFormat::None,
        );
        let ecef_state = state_eci_to_ecef(epoch, cart_state);
        ecef_traj.add_state(epoch, ecef_state).unwrap();
        let kep_from_ecef = ecef_traj.to_keplerian(AngleFormat::Radians);
        assert_eq!(kep_from_ecef.frame, OrbitFrame::ECI);
        assert_eq!(kep_from_ecef.representation, OrbitRepresentation::Keplerian);
        assert_eq!(kep_from_ecef.angle_format, AngleFormat::Radians);
        assert_eq!(kep_from_ecef.len(), 1);
        let (epoch_out, state_out) = kep_from_ecef.get(0).unwrap();
        assert_eq!(epoch_out, epoch);
        for i in 0..6 {
            assert_abs_diff_eq!(state_out[i], state_kep_rad[i], epsilon = 1e-9);
        }
    }
}
