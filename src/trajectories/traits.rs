/*!
 * Common traits for trajectory implementations.
 *
 * This module defines the core traits that both static (compile-time sized) and dynamic
 * (runtime sized) trajectory implementations must implement to ensure a consistent interface.
 */

use crate::constants::AngleFormat;
use crate::time::Epoch;
use crate::utils::BraheError;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Interpolation methods for retrieving trajectory states at arbitrary epochs.
///
/// Different methods provide varying trade-offs between computational cost and accuracy.
/// For most applications, linear interpolation provides sufficient accuracy with minimal
/// computational overhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum InterpolationMethod {
    /// Linear interpolation between adjacent states.
    /// Good balance of speed and accuracy for smooth trajectories.
    #[default]
    Linear,
}

/// Enumeration of trajectory eviction policies for memory management
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TrajectoryEvictionPolicy {
    /// No eviction - trajectory grows unbounded
    #[default]
    None,
    /// Keep most recent states, evict oldest when limit reached
    KeepCount,
    /// Keep states within a time duration from current epoch
    KeepWithinDuration,
}

/// Enumeration of orbit reference frames
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrbitFrame {
    /// Earth-Centered Inertial (legacy, ambiguous - prefer GCRF or EME2000)
    ECI,
    /// Earth-Centered Earth-Fixed (legacy, ambiguous - prefer ITRF)
    ECEF,
    /// Geocentric Celestial Reference Frame (IAU 2006/2000A)
    GCRF,
    /// International Terrestrial Reference Frame
    ITRF,
    /// Earth Mean Equator and Equinox of J2000.0
    EME2000,
}

impl fmt::Display for OrbitFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrbitFrame::ECI => write!(f, "ECI"),
            OrbitFrame::ECEF => write!(f, "ECEF"),
            OrbitFrame::GCRF => write!(f, "GCRF"),
            OrbitFrame::ITRF => write!(f, "ITRF"),
            OrbitFrame::EME2000 => write!(f, "EME2000"),
        }
    }
}

impl fmt::Debug for OrbitFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrbitFrame::ECI => write!(f, "OrbitFrame(Earth-Centered Inertial)"),
            OrbitFrame::ECEF => write!(f, "OrbitFrame(Earth-Centered Earth-Fixed)"),
            OrbitFrame::GCRF => write!(f, "OrbitFrame(Geocentric Celestial Reference Frame)"),
            OrbitFrame::ITRF => write!(f, "OrbitFrame(International Terrestrial Reference Frame)"),
            OrbitFrame::EME2000 => {
                write!(f, "OrbitFrame(Earth Mean Equator and Equinox of J2000.0)")
            }
        }
    }
}

/// Enumeration of orbit state representations
#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrbitRepresentation {
    /// Cartesian position and velocity (x, y, z, vx, vy, vz)
    Cartesian,
    /// Keplerian elements (a, e, i, Ω, ω, M)
    Keplerian,
}

impl fmt::Display for OrbitRepresentation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrbitRepresentation::Cartesian => write!(f, "Cartesian"),
            OrbitRepresentation::Keplerian => write!(f, "Keplerian"),
        }
    }
}

impl fmt::Debug for OrbitRepresentation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrbitRepresentation::Cartesian => write!(f, "OrbitRepresentation(Cartesian)"),
            OrbitRepresentation::Keplerian => write!(f, "OrbitRepresentation(Keplerian)"),
        }
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
    fn from_data(epochs: Vec<Epoch>, states: Vec<Self::StateVector>) -> Result<Self, BraheError>
    where
        Self: Sized;

    /// Add a state vector at a specific epoch
    ///
    /// # Arguments
    /// * `epoch` - Time epoch for the state
    /// * `state` - State vector to add
    ///
    /// # Returns
    /// * `Ok(())` - State successfully added
    /// * `Err(BraheError)` - If addition fails (e.g., dimension mismatch)
    fn add(&mut self, epoch: Epoch, state: Self::StateVector) -> ();

    /// Get the epoch at a specific index
    ///
    /// # Arguments
    /// * `index` - Index of the epoch to retrieve
    ///
    /// # Returns
    /// * `Ok(epoch)` - Epoch at the index
    /// * `Err(BraheError)` - If index is out of bounds
    fn epoch_at_idx(&self, index: usize) -> Result<Epoch, BraheError>;

    /// Get the state vector at a specific index
    ///
    /// # Arguments
    /// * `index` - Index of the state to retrieve
    ///
    /// # Returns
    /// * `Ok(state)` - State vector at the index
    /// * `Err(BraheError)` - If index is out of bounds
    fn state_at_idx(&self, index: usize) -> Result<Self::StateVector, BraheError>;

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
    fn remove_epoch(&mut self, epoch: &Epoch) -> Result<Self::StateVector, BraheError>;

    /// Remove a state at a specific index
    ///
    /// # Arguments
    /// * `index` - Index of the state to remove
    ///
    /// # Returns
    /// * `Ok((epoch, state))` - The removed epoch and state
    /// * `Err(BraheError)` - If index is out of bounds
    fn remove(&mut self, index: usize) -> Result<(Epoch, Self::StateVector), BraheError>;

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

    /// Set eviction policy to keep a maximum number of states.
    ///
    /// When the number of states exceeds `max_size`, the oldest states are evicted first.
    ///
    /// # Arguments
    /// * `max_size` - Maximum number of states to retain (must be >= 1)
    ///
    /// # Returns
    /// * `Ok(())` - Policy successfully set and applied
    /// * `Err(BraheError)` - If max_size is less than 1
    fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError>;

    /// Set eviction policy to keep states within a maximum age from the most recent state.
    ///
    /// States older than `max_age` seconds from the most recent state are evicted.
    ///position_at_epoch
    /// # Arguments
    /// * `max_age` - Maximum age of states to retain in seconds (must be > 0.0)
    ///
    /// # Returns
    /// * `Ok(())` - Policy successfully set and applied
    /// * `Err(BraheError)` - If max_age is not positive
    fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError>;

    /// Get the current eviction policy
    ///
    /// # Returns
    /// The current eviction policy (None, KeepCount, or KeepWithinDuration)
    fn get_eviction_policy(&self) -> TrajectoryEvictionPolicy;
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
/// use brahe::trajectories::STrajectory6;
/// use brahe::traits::{Trajectory, Interpolatable, InterpolationMethod};
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
        Self::StateVector: Clone
            + std::ops::Mul<f64, Output = Self::StateVector>
            + std::ops::Add<Output = Self::StateVector>,
    {
        if self.is_empty() {
            return Err(BraheError::Error(
                "Cannot interpolate state from empty trajectory".to_string(),
            ));
        }

        // If only one state, also error if epoch does not match
        if self.len() == 1 {
            let (only_epoch, only_state) = self.first().unwrap();
            if *epoch != only_epoch {
                return Err(BraheError::Error(
                    "Cannot interpolate state: single state trajectory and epoch does not match"
                        .to_string(),
                ));
            }
            return Ok(only_state);
        }

        // Explicit bounds checking
        if let Some(start) = self.start_epoch()
            && *epoch < start
        {
            return Err(BraheError::OutOfBoundsError(format!(
                "Cannot interpolate: epoch {} is before trajectory start {}",
                epoch, start
            )));
        }

        if let Some(end) = self.end_epoch()
            && *epoch > end
        {
            return Err(BraheError::OutOfBoundsError(format!(
                "Cannot interpolate: epoch {} is after trajectory end {}",
                epoch, end
            )));
        }

        // Get indices before and after the target epoch (single search operation each)
        let idx1 = self.index_before_epoch(epoch)?;
        let idx2 = self.index_after_epoch(epoch)?;

        // If indices are the same, we have an exact match
        if idx1 == idx2 {
            return self.state_at_idx(idx1);
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
        Self::StateVector: Clone
            + std::ops::Mul<f64, Output = Self::StateVector>
            + std::ops::Add<Output = Self::StateVector>,
    {
        // Explicit bounds checking
        if let Some(start) = self.start_epoch()
            && *epoch < start
        {
            return Err(BraheError::OutOfBoundsError(format!(
                "Cannot interpolate: epoch {} is before trajectory start {}",
                epoch, start
            )));
        }

        if let Some(end) = self.end_epoch()
            && *epoch > end
        {
            return Err(BraheError::OutOfBoundsError(format!(
                "Cannot interpolate: epoch {} is after trajectory end {}",
                epoch, end
            )));
        }

        self.interpolate_linear(epoch)
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
/// use brahe::trajectories::OrbitTrajectory;
/// use brahe::traits::{OrbitalTrajectory, OrbitFrame, OrbitRepresentation, Trajectory};
/// use brahe::AngleFormat;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector6;
///
/// // Create orbital trajectory in ECI Cartesian coordinates
/// let mut traj = OrbitTrajectory::new(
///     OrbitFrame::ECI,
///     OrbitRepresentation::Cartesian,
///     None,
/// );
///
/// // Add state
/// let epoch = Epoch::from_datetime(2023, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let state = Vector6::new(6.678e6, 0.0, 0.0, 0.0, 7.726e3, 0.0);
/// traj.add(epoch, state);
///
/// // Convert to Keplerian in degrees
/// let kep_traj = traj.to_keplerian(AngleFormat::Degrees);
/// ```
pub trait OrbitalTrajectory: Interpolatable {
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
    /// New orbital trajectory with data
    ///
    /// # Panics
    /// Panics if parameters are invalid (e.g., None angle_format with Keplerian, or Keplerian with ECEF)
    fn from_orbital_data(
        epochs: Vec<Epoch>,
        states: Vec<Self::StateVector>,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: Option<AngleFormat>,
    ) -> Self
    where
        Self: Sized;

    /// Convert to Earth-Centered Inertial (ECI) frame.
    ///
    /// Returns a new trajectory in the ECI frame.
    ///
    /// # Returns
    /// * `Ok(Self)` - New trajectory in ECI frame
    /// * `Err(BraheError)` - If conversion fails
    fn to_eci(&self) -> Self
    where
        Self: Sized;

    /// Convert to Earth-Centered Earth-Fixed (ECEF) frame.
    ///
    /// Returns a new trajectory in the ECEF frame.
    ///
    /// # Returns
    /// * `Ok(Self)` - New trajectory in ECEF frame
    /// * `Err(BraheError)` - If conversion fails
    fn to_ecef(&self) -> Self
    where
        Self: Sized;

    /// Convert to Geocentric Celestial Reference Frame (GCRF).
    ///
    /// Returns a new trajectory in the GCRF frame.
    ///
    /// # Returns
    /// * `Self` - New trajectory in GCRF frame
    fn to_gcrf(&self) -> Self
    where
        Self: Sized;

    /// Convert to Earth Mean Equator and Equinox of J2000.0 (EME2000) frame.
    ///
    /// Returns a new trajectory in the EME2000 frame.
    ///
    /// # Returns
    /// * `Self` - New trajectory in EME2000 frame
    fn to_eme2000(&self) -> Self
    where
        Self: Sized;

    /// Convert to International Terrestrial Reference Frame (ITRF).
    ///
    /// Returns a new trajectory in the ITRF frame.
    ///
    /// # Returns
    /// * `Self` - New trajectory in ITRF frame
    fn to_itrf(&self) -> Self
    where
        Self: Sized;

    /// Convert to Keplerian elements with specified angle format.
    ///
    /// Returns a new trajectory in Keplerian representation.
    ///
    /// # Arguments
    /// * `angle_format` - Format for angular elements (Radians or Degrees, cannot be None)
    ///
    /// # Returns
    /// * `Ok(Self)` - New trajectory in Keplerian representation
    /// * `Err(BraheError)` - If angle_format is None or conversion fails
    fn to_keplerian(&self, angle_format: AngleFormat) -> Self
    where
        Self: Sized;
}
