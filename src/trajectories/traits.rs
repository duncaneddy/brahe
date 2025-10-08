/*!
 * Common traits for trajectory implementations.
 *
 * This module defines the core traits that both static (compile-time sized) and dynamic
 * (runtime sized) trajectory implementations must implement to ensure a consistent interface.
 */

use crate::time::Epoch;
use crate::utils::BraheError;
use serde::{Deserialize, Serialize};

/// Interpolation methods for retrieving trajectory states at arbitrary epochs.
///
/// Different methods provide varying trade-offs between computational cost and accuracy.
/// For most applications, linear interpolation provides sufficient accuracy with minimal
/// computational overhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationMethod {
    /// Linear interpolation between adjacent states.
    /// Good balance of speed and accuracy for smooth trajectories.
    Linear,
    /// Cubic spline interpolation using natural boundary conditions.
    /// Higher accuracy for smooth trajectories but requires more computation.
    CubicSpline,
    /// Lagrange polynomial interpolation using nearby points.
    /// High accuracy but can exhibit oscillatory behavior with noisy data.
    Lagrange,
    /// Hermite interpolation preserving first derivatives.
    /// Excellent for smooth trajectories with known velocity information.
    Hermite,
}

impl Default for InterpolationMethod {
    fn default() -> Self {
        InterpolationMethod::Linear
    }
}

/// Enumeration of trajectory eviction policies for memory management
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrajectoryEvictionPolicy {
    /// No eviction - trajectory grows unbounded
    None,
    /// Keep most recent states, evict oldest when limit reached
    KeepCount,
    /// Keep states within a time duration from current epoch
    KeepWithinDuration,
}

impl Default for TrajectoryEvictionPolicy {
    fn default() -> Self {
        TrajectoryEvictionPolicy::None
    }
}

/// Core trajectory functionality that all trajectory implementations must provide.
///
/// This trait defines the complete interface for storing, retrieving, and managing
/// trajectory state data over time, regardless of the underlying storage mechanism
/// (compile-time vs runtime sized vectors).
///
/// # Implementations
/// - [`STrajectory<N>`](super::strajectory::STrajectory) - Compile-time sized trajectories
/// - [`DynamicTrajectory`](super::trajectory::DynamicTrajectory) - Runtime-sized trajectories
pub trait Trajectory {
    /// The type used to represent state vectors
    type StateVector;

    /// Create a trajectory from vectors of epochs and states
    ///
    /// Interpolation method defaults to Linear. Use `set_interpolation_method` to change.
    ///
    /// # Arguments
    /// * `epochs` - Vector of epochs (must be non-empty and same length as states)
    /// * `states` - Vector of state vectors (all must have consistent dimension)
    ///
    /// # Returns
    /// * `Ok(Self)` - Trajectory successfully created with sorted data
    /// * `Err(BraheError)` - If validation fails (length mismatch, empty vectors, inconsistent dimensions)
    fn from_data(epochs: Vec<Epoch>, states: Vec<Self::StateVector>) -> Result<Self, BraheError> where Self: Sized;

    /// Add a state vector at a specific epoch
    ///
    /// # Arguments
    /// * `epoch` - Time epoch for the state
    /// * `state` - State vector to add
    ///
    /// # Returns
    /// * `Ok(())` - State successfully added
    /// * `Err(BraheError)` - If addition fails (e.g., dimension mismatch)
    fn add_state(&mut self, epoch: Epoch, state: Self::StateVector) -> Result<(), BraheError>;

    /// Get the state vector at a specific index
    ///
    /// # Arguments
    /// * `index` - Index of the state to retrieve
    ///
    /// # Returns
    /// * `Ok(state)` - State vector at the index
    /// * `Err(BraheError)` - If index is out of bounds
    fn state_at_index(&self, index: usize) -> Result<Self::StateVector, BraheError>;

    /// Get the epoch at a specific index
    ///
    /// # Arguments
    /// * `index` - Index of the epoch to retrieve
    ///
    /// # Returns
    /// * `Ok(epoch)` - Epoch at the index
    /// * `Err(BraheError)` - If index is out of bounds
    fn epoch_at_index(&self, index: usize) -> Result<Epoch, BraheError>;

    /// Find the nearest state to a given epoch
    ///
    /// # Arguments
    /// * `epoch` - Target epoch to find nearest state for
    ///
    /// # Returns
    /// * `Ok((epoch, state))` - Nearest epoch and corresponding state
    /// * `Err(BraheError)` - If trajectory is empty
    fn nearest_state(&self, epoch: &Epoch) -> Result<(Epoch, Self::StateVector), BraheError>;

    /// Get the number of states in the trajectory
    fn len(&self) -> usize;

    /// Check if the trajectory is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the first epoch in the trajectory, if any
    fn start_epoch(&self) -> Option<Epoch>;

    /// Get the last epoch in the trajectory, if any
    fn end_epoch(&self) -> Option<Epoch>;

    /// Get the time span covered by the trajectory in seconds
    fn timespan(&self) -> Option<f64>;

    /// Get the first (epoch, state) pair in the trajectory, if any
    fn first(&self) -> Option<(Epoch, Self::StateVector)>;

    /// Get the last (epoch, state) pair in the trajectory, if any
    fn last(&self) -> Option<(Epoch, Self::StateVector)>;

    /// Clear all states from the trajectory
    fn clear(&mut self);

    /// Remove a state at a specific epoch
    ///
    /// # Arguments
    /// * `epoch` - Epoch of the state to remove
    ///
    /// # Returns
    /// * `Ok(state)` - The removed state vector
    /// * `Err(BraheError)` - If epoch not found
    fn remove_state(&mut self, epoch: &Epoch) -> Result<Self::StateVector, BraheError>;

    /// Remove a state at a specific index
    ///
    /// # Arguments
    /// * `index` - Index of the state to remove
    ///
    /// # Returns
    /// * `Ok((epoch, state))` - The removed epoch and state
    /// * `Err(BraheError)` - If index is out of bounds
    fn remove_state_at_index(&mut self, index: usize) -> Result<(Epoch, Self::StateVector), BraheError>;

    /// Get both epoch and state at a specific index
    ///
    /// # Arguments
    /// * `index` - Index to retrieve
    ///
    /// # Returns
    /// * `Ok((epoch, state))` - Epoch and state at the index
    /// * `Err(BraheError)` - If index is out of bounds
    fn get(&self, index: usize) -> Result<(Epoch, Self::StateVector), BraheError>;

    /// Get the index of the state at or before the given epoch
    ///
    /// Returns the index of the state at the exact epoch if it exists, otherwise the index of the closest state before it.
    ///
    /// # Arguments
    /// * `epoch` - Target epoch
    ///
    /// # Returns
    /// * `Ok(index)` - Index of the state at or before the target epoch
    /// * `Err(BraheError)` - If trajectory is empty or epoch is before all states
    fn index_before_epoch(&self, epoch: &Epoch) -> Result<usize, BraheError>;

    /// Get the index of the state at or after the given epoch
    ///
    /// Returns the index of the state at the exact epoch if it exists, otherwise the index of the closest state after it.
    ///
    /// # Arguments
    /// * `epoch` - Target epoch
    ///
    /// # Returns
    /// * `Ok(index)` - Index of the state at or after the target epoch
    /// * `Err(BraheError)` - If trajectory is empty or epoch is after all states
    fn index_after_epoch(&self, epoch: &Epoch) -> Result<usize, BraheError>;

    /// Get the state at or before the given epoch
    ///
    /// Returns the state at the exact epoch if it exists, otherwise the closest state before it.
    ///
    /// # Arguments
    /// * `epoch` - Target epoch
    ///
    /// # Returns
    /// * `Ok((epoch, state))` - The epoch and state at or before the target epoch
    /// * `Err(BraheError)` - If trajectory is empty or epoch is before all states
    fn state_before_epoch(&self, epoch: &Epoch) -> Result<(Epoch, Self::StateVector), BraheError> {
        let index = self.index_before_epoch(epoch)?;
        self.get(index)
    }

    /// Get the state at or after the given epoch
    ///
    /// Returns the state at the exact epoch if it exists, otherwise the closest state after it.
    ///
    /// # Arguments
    /// * `epoch` - Target epoch
    ///
    /// # Returns
    /// * `Ok((epoch, state))` - The epoch and state at or after the target epoch
    /// * `Err(BraheError)` - If trajectory is empty or epoch is after all states
    fn state_after_epoch(&self, epoch: &Epoch) -> Result<(Epoch, Self::StateVector), BraheError> {
        let index = self.index_after_epoch(epoch)?;
        self.get(index)
    }
}

/// Trait for trajectory interpolation functionality.
///
/// This trait provides interpolation methods for retrieving trajectory states at arbitrary epochs.
/// It requires the implementing type to also implement `Trajectory` to access the underlying state data.
///
/// # Default Implementations
/// The trait provides default implementations for `interpolate_linear` and `interpolate` methods
/// that use the `Trajectory` trait methods to perform interpolation.
///
/// # Examples
/// ```rust
/// use brahe::trajectories::{STrajectory6, Trajectory, Interpolatable, InterpolationMethod};
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector6;
///
/// let epochs = vec![
///     Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC),
///     Epoch::from_datetime(2023, 1, 1, 13, 0, 0.0, 0.0, TimeSystem::UTC),
/// ];
/// let states = vec![
///     Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
///     Vector6::new(7100e3, 1000e3, 500e3, 100.0, 7.6e3, 50.0),
/// ];
/// let traj = STrajectory6::from_data(epochs, states).unwrap();
///
/// // Interpolate at an intermediate epoch
/// let epoch = Epoch::from_datetime(2023, 1, 1, 12, 30, 0.0, 0.0, TimeSystem::UTC);
/// let state = traj.interpolate(&epoch).unwrap();
/// ```
pub trait Interpolatable: Trajectory {
    /// Set the interpolation method for the trajectory
    ///
    /// # Arguments
    /// * `method` - The interpolation method to use
    fn set_interpolation_method(&mut self, method: InterpolationMethod);

    /// Get the current interpolation method
    ///
    /// # Returns
    /// The current interpolation method (defaults to Linear if not set)
    fn get_interpolation_method(&self) -> InterpolationMethod;

    /// Interpolate state at a given epoch using linear interpolation
    ///
    /// This is a default implementation that uses the `Trajectory` methods to
    /// perform linear interpolation between bracketing states.
    ///
    /// # Arguments
    /// * `epoch` - Target epoch for interpolation
    ///
    /// # Returns
    /// * `Ok(state)` - Interpolated state vector
    /// * `Err(BraheError)` - If interpolation fails or epoch is out of range
    fn interpolate_linear(&self, epoch: &Epoch) -> Result<Self::StateVector, BraheError>
    where
        Self::StateVector: Clone + std::ops::Mul<f64, Output = Self::StateVector> + std::ops::Add<Output = Self::StateVector>,
    {
        if self.is_empty() {
            return Err(BraheError::Error(
                "Cannot interpolate state from empty trajectory".to_string(),
            ));
        }

        // If only one state, return it
        if self.len() == 1 {
            return self.state_at_index(0);
        }

        // Get indices before and after the target epoch (single search operation each)
        let idx1 = self.index_before_epoch(epoch)?;
        let idx2 = self.index_after_epoch(epoch)?;

        // If indices are the same, we have an exact match
        if idx1 == idx2 {
            return self.state_at_index(idx1);
        }

        // Get the bracketing epochs and states
        let (epoch1, state1) = self.get(idx1)?;
        let (epoch2, state2) = self.get(idx2)?;

        // Linear interpolation: state = state1 * (1 - t) + state2 * t
        // where t = (epoch - epoch1) / (epoch2 - epoch1)
        let t = (*epoch - epoch1) / (epoch2 - epoch1);
        let interpolated = state1.clone() * (1.0 - t) + state2 * t;

        Ok(interpolated)
    }

    /// Interpolate state at a given epoch using the configured interpolation method
    ///
    /// This is a default implementation that dispatches to the appropriate interpolation
    /// method based on the current `interpolation_method` setting.
    ///
    /// # Arguments
    /// * `epoch` - Target epoch for interpolation
    ///
    /// # Returns
    /// * `Ok(state)` - Interpolated state vector
    /// * `Err(BraheError)` - If interpolation fails or epoch is out of range
    fn interpolate(&self, epoch: &Epoch) -> Result<Self::StateVector, BraheError>
    where
        Self::StateVector: Clone + std::ops::Mul<f64, Output = Self::StateVector> + std::ops::Add<Output = Self::StateVector>,
    {
        match self.get_interpolation_method() {
            InterpolationMethod::Linear => self.interpolate_linear(epoch),
            InterpolationMethod::CubicSpline => Err(BraheError::Error(
                "Cubic spline interpolation not yet implemented".to_string(),
            )),
            InterpolationMethod::Lagrange => Err(BraheError::Error(
                "Lagrange interpolation not yet implemented".to_string(),
            )),
            InterpolationMethod::Hermite => Err(BraheError::Error(
                "Hermite interpolation not yet implemented".to_string(),
            )),
        }
    }
}

/// Trait for orbital-specific functionality on 6-dimensional trajectories.
///
/// This trait provides methods for working with orbital state trajectories, including
/// conversions between reference frames (ECI/ECEF), state representations (Cartesian/Keplerian),
/// and angle formats (radians/degrees). It also provides convenient accessors for position
/// and velocity components.
///
/// This trait requires both `Trajectory` and `Interpolatable` to be implemented, enabling
/// both basic trajectory operations and state interpolation.
///
/// # Reference Frames
/// - **ECI (Earth-Centered Inertial)**: GCRF inertial reference frame
/// - **ECEF (Earth-Centered Earth-Fixed)**: Earth-fixed rotating frame
///
/// # State Representations
/// - **Cartesian**: Position and velocity vectors [x, y, z, vx, vy, vz] in meters and m/s
/// - **Keplerian**: Classical orbital elements [a, e, i, Ω, ω, M] where angles use specified format
///
/// # Angle Formats
/// - **Radians**: Angular elements in radians (i, Ω, ω, M)
/// - **Degrees**: Angular elements in degrees (i, Ω, ω, M)
/// - **None**: No angular representation (for Cartesian states)
///
/// # Examples
/// ```rust
/// use brahe::trajectories::{STrajectory6, OrbitalTrajectory, OrbitFrame, OrbitRepresentation, AngleFormat, Trajectory};
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector6;
///
/// // Create orbital trajectory in ECI Cartesian coordinates
/// let mut traj = STrajectory6::new_orbital_trajectory(
///     OrbitFrame::ECI,
///     OrbitRepresentation::Cartesian,
///     AngleFormat::None,
/// ).unwrap();
///
/// // Add state
/// let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let state = Vector6::new(6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0);
/// traj.add_state(epoch, state).unwrap();
///
/// // Convert to Keplerian in degrees
/// let kep_traj = traj.to_keplerian(AngleFormat::Degrees).unwrap();
/// ```
pub trait OrbitalTrajectory: Interpolatable {
    /// Convert the trajectory to a different reference frame.
    ///
    /// # Arguments
    /// * `target_frame` - Target reference frame (ECI or ECEF)
    ///
    /// # Returns
    /// * `Ok(Self)` - New trajectory in target frame
    /// * `Err(BraheError)` - If conversion fails
    ///
    /// # Note
    /// If the trajectory is in Keplerian representation, it will first be converted to
    /// Cartesian for the frame transformation, then converted back to Keplerian.
    fn to_frame(&self, target_frame: super::strajectory::OrbitFrame) -> Result<Self, BraheError>
    where Self: Sized;

    /// Convert to Earth-Centered Inertial (ECI) frame.
    ///
    /// Convenience method equivalent to `to_frame(OrbitFrame::ECI)`.
    fn to_eci(&self) -> Result<Self, BraheError> where Self: Sized {
        self.to_frame(super::strajectory::OrbitFrame::ECI)
    }

    /// Convert to Earth-Centered Earth-Fixed (ECEF) frame.
    ///
    /// Convenience method equivalent to `to_frame(OrbitFrame::ECEF)`.
    fn to_ecef(&self) -> Result<Self, BraheError> where Self: Sized {
        self.to_frame(super::strajectory::OrbitFrame::ECEF)
    }

    /// Convert the trajectory to a different state representation.
    ///
    /// # Arguments
    /// * `target_representation` - Target representation (Cartesian or Keplerian)
    /// * `target_angle_format` - Angle format for Keplerian (None for Cartesian)
    ///
    /// # Returns
    /// * `Ok(Self)` - New trajectory in target representation
    /// * `Err(BraheError)` - If conversion fails or parameters are invalid
    fn to_representation(&self, target_representation: super::strajectory::OrbitRepresentation,
                        target_angle_format: super::strajectory::AngleFormat) -> Result<Self, BraheError>
    where Self: Sized;

    /// Convert to Cartesian representation.
    ///
    /// Convenience method equivalent to `to_representation(OrbitRepresentation::Cartesian, AngleFormat::None)`.
    fn to_cartesian(&self) -> Result<Self, BraheError> where Self: Sized {
        self.to_representation(super::strajectory::OrbitRepresentation::Cartesian, super::strajectory::AngleFormat::None)
    }

    /// Convert to Keplerian elements with specified angle format.
    ///
    /// # Arguments
    /// * `angle_format` - Format for angular elements (Radians or Degrees, cannot be None)
    ///
    /// # Returns
    /// * `Ok(Self)` - New trajectory in Keplerian representation
    /// * `Err(BraheError)` - If angle_format is None or conversion fails
    fn to_keplerian(&self, angle_format: super::strajectory::AngleFormat) -> Result<Self, BraheError> where Self: Sized {
        if angle_format == super::strajectory::AngleFormat::None {
            return Err(BraheError::Error(
                "Angle format must be specified when converting to Keplerian elements".to_string(),
            ));
        }
        self.to_representation(super::strajectory::OrbitRepresentation::Keplerian, angle_format)
    }

    /// Convert the trajectory to a different angle format (only for Keplerian).
    ///
    /// # Arguments
    /// * `target_format` - Target angle format (Radians or Degrees)
    ///
    /// # Returns
    /// * `Ok(Self)` - New trajectory with converted angles
    /// * `Err(BraheError)` - If trajectory is not Keplerian or format is invalid
    fn to_angle_format(&self, target_format: super::strajectory::AngleFormat) -> Result<Self, BraheError>
    where Self: Sized;

    /// Convert to degrees representation (only for Keplerian).
    ///
    /// Convenience method equivalent to `to_angle_format(AngleFormat::Degrees)`.
    fn to_degrees(&self) -> Result<Self, BraheError> where Self: Sized {
        self.to_angle_format(super::strajectory::AngleFormat::Degrees)
    }

    /// Convert to radians representation (only for Keplerian).
    ///
    /// Convenience method equivalent to `to_angle_format(AngleFormat::Radians)`.
    fn to_radians(&self) -> Result<Self, BraheError> where Self: Sized {
        self.to_angle_format(super::strajectory::AngleFormat::Radians)
    }

    /// Get position component of the state at a specific epoch (Cartesian only).
    ///
    /// # Arguments
    /// * `epoch` - Target epoch for position extraction
    ///
    /// # Returns
    /// * `Ok(Vector3<f64>)` - Position vector [x, y, z] in meters
    /// * `Err(BraheError)` - If trajectory is not Cartesian or interpolation fails
    fn position_at_epoch(&self, epoch: &Epoch) -> Result<nalgebra::Vector3<f64>, BraheError>;

    /// Get velocity component of the state at a specific epoch (Cartesian only).
    ///
    /// # Arguments
    /// * `epoch` - Target epoch for velocity extraction
    ///
    /// # Returns
    /// * `Ok(Vector3<f64>)` - Velocity vector [vx, vy, vz] in m/s
    /// * `Err(BraheError)` - If trajectory is not Cartesian or interpolation fails
    fn velocity_at_epoch(&self, epoch: &Epoch) -> Result<nalgebra::Vector3<f64>, BraheError>;

    /// Get the current orbital reference frame.
    ///
    /// # Returns
    /// Current reference frame (ECI or ECEF)
    fn orbital_frame(&self) -> super::strajectory::OrbitFrame;

    /// Get the current orbital state representation.
    ///
    /// # Returns
    /// Current representation (Cartesian or Keplerian)
    fn orbital_representation(&self) -> super::strajectory::OrbitRepresentation;

    /// Get the current angle format.
    ///
    /// # Returns
    /// Current angle format (Radians, Degrees, or None for Cartesian)
    fn angle_format(&self) -> super::strajectory::AngleFormat;

    /// Convert trajectory to different frame, representation, and angle format in one operation.
    ///
    /// This is more efficient than chaining multiple conversions as it minimizes
    /// intermediate transformations.
    ///
    /// # Arguments
    /// * `target_frame` - Target reference frame
    /// * `target_representation` - Target state representation
    /// * `target_angle_format` - Target angle format
    ///
    /// # Returns
    /// * `Ok(Self)` - New trajectory with all conversions applied
    /// * `Err(BraheError)` - If any conversion fails or parameters are invalid
    fn convert_to(
        &self,
        target_frame: super::strajectory::OrbitFrame,
        target_representation: super::strajectory::OrbitRepresentation,
        target_angle_format: super::strajectory::AngleFormat,
    ) -> Result<Self, BraheError>
    where Self: Sized;
}