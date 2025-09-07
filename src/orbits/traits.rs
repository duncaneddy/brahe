/*!
* The orbits `traits` module provides traits for working with orbital state information.
* It includes the `OrbitalState` trait, which defines methods for retrieving position and velocity
* vectors, as well as the `OrbitalStateWithTime` trait, which extends `OrbitalState` to include
* time information. These traits are designed to be implemented by types that represent
* orbital states, such as those derived from Two-Line Element (TLE) data or other
* orbital models.
*/

use crate::time::Epoch;
use crate::trajectories::{AngleFormat, OrbitFrame, OrbitState, OrbitStateType, Trajectory, TrajectoryEvictionPolicy};
use crate::utils::BraheError;
use nalgebra as na;
use nalgebra::Vector6;

trait OrbitalState {
    /// Returns the time of the orbital state.
    fn epoch(&self) -> Option<Epoch>;

    /// Returns the position vector of the orbital state.
    fn position(&self) -> na::Vector3<f64>;

    /// Returns the velocity vector of the orbital state.
    fn velocity(&self) -> na::Vector3<f64>;

    /// Returns the full state vector, which includes position and velocity.
    fn state(&self) -> na::Vector6<f64>;

    /// Returns the state as osculating elements.
    fn state_elements(&self) -> Option<na::Vector6<f64>> {
        None
    }

    /// Returns the state as Earth-centered inertial (ECI) coordinates.
    fn eci_state(&self) -> Option<na::Vector6<f64>> {
        None
    }

    /// Returns the state as Earth-centered Earth-fixed (ECEF) coordinates.
    fn ecef_state(&self) -> Option<na::Vector6<f64>> {
        None
    }
}

trait OrbitalStateInterpolator: OrbitalState {
    /// Returns number of epochs in the interpolator.
    ///
    /// # Returns
    /// The number of epochs in the interpolator.
    ///
    fn num_epochs(&self) -> usize;

    /// Returns the number of states in the interpolator.
    /// This is the same as `num_epochs()`.
    ///
    /// # Returns
    /// The number of states in the interpolator.
    fn num_states(&self) -> usize {
        self.num_epochs()
    }

    /// Interpolates the orbital state at a given epoch.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to interpolate the orbital state.
    ///
    /// # Returns
    /// An `Option` containing the interpolated orbital state, or `None` if the interpolation fails.
    fn state_at(&self, epoch: Epoch) -> Option<na::Vector6<f64>>;

    /// Returns an iterator over the states in the interpolator in their
    /// default frame of reference.
    fn states(&self) -> impl Iterator<Item = impl OrbitalState>;

    /// Returns an interator over the states as Keplerian elements.
    fn states_elements(&self) -> impl Iterator<Item = Option<na::Vector6<f64>>> {
        self.states().map(|state| state.state_elements())
    }

    /// Returns an iterator over the states in Earth-centered inertial (ECI) coordinates.
    fn states_eci(&self) -> impl Iterator<Item = Option<na::Vector6<f64>>> {
        self.states().map(|state| state.eci_state())
    }

    /// Returns an iterator over the states in Earth-centered Earth-fixed (ECEF) coordinates.
    fn states_ecef(&self) -> impl Iterator<Item = Option<na::Vector6<f64>>> {
        self.states().map(|state| state.ecef_state())
    }
}

/// Trait for analytic orbital propagators that can compute states directly at any epoch
/// without requiring numerical integration. This trait is designed for propagators like
/// SGP4/TLE that have closed-form solutions.
pub trait AnalyticPropagator {
    /// Returns the state at the given epoch in the propagator's default coordinate frame.
    /// 
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    /// 
    /// # Returns
    /// A 6-element vector containing position (km) and velocity (km/s) components
    /// in the propagator's default frame (typically TEME for SGP4).
    fn state(&self, epoch: Epoch) -> na::Vector6<f64>;

    /// Returns the state at the given epoch in Earth-Centered Inertial (ECI) coordinates.
    /// 
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    /// 
    /// # Returns
    /// A 6-element vector containing position (km) and velocity (km/s) components
    /// in the ECI frame.
    fn state_eci(&self, epoch: Epoch) -> na::Vector6<f64>;

    /// Returns the state at the given epoch in Earth-Centered Earth-Fixed (ECEF) coordinates.
    /// 
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    /// 
    /// # Returns
    /// A 6-element vector containing position (km) and velocity (km/s) components
    /// in the ECEF frame.
    fn state_ecef(&self, epoch: Epoch) -> na::Vector6<f64>;

    /// Returns the state at the given epoch as osculating orbital elements.
    /// 
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    /// 
    /// # Returns
    /// A 6-element vector containing osculating Keplerian elements:
    /// [semi-major axis (km), eccentricity, inclination (rad), RAAN (rad), 
    ///  argument of perigee (rad), true anomaly (rad)]
    fn state_osculating_elements(&self, epoch: Epoch) -> na::Vector6<f64>;

    /// Returns states at multiple epochs in the propagator's default coordinate frame as a Trajectory.
    /// 
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    /// 
    /// # Returns
    /// Trajectory containing states in the propagator's default frame
    fn states(&self, epochs: &[Epoch]) -> Trajectory<OrbitState>;

    /// Returns states at multiple epochs in Earth-Centered Inertial (ECI) coordinates as a Trajectory.
    /// 
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    /// 
    /// # Returns
    /// Trajectory containing states in the ECI frame
    fn states_eci(&self, epochs: &[Epoch]) -> Trajectory<OrbitState>;

    /// Returns states at multiple epochs in Earth-Centered Earth-Fixed (ECEF) coordinates as a Trajectory.
    /// 
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    /// 
    /// # Returns
    /// Trajectory containing states in the ECEF frame
    fn states_ecef(&self, epochs: &[Epoch]) -> Trajectory<OrbitState>;

    /// Returns states at multiple epochs as osculating orbital elements as a Trajectory.
    /// 
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    /// 
    /// # Returns
    /// Trajectory containing states as osculating Keplerian elements
    fn states_osculating_elements(&self, epochs: &[Epoch]) -> Trajectory<OrbitState>;
}

/// Core trait for orbit propagators providing unified interface
pub trait OrbitPropagator {
    /// Propagate to a specific target epoch
    /// 
    /// # Arguments
    /// * `target_epoch` - The epoch to propagate to
    /// 
    /// # Returns
    /// Reference to the propagated state at target epoch
    fn propagate_to(&mut self, target_epoch: Epoch) -> Result<&OrbitState, BraheError>;
    
    /// Reset propagator to initial conditions
    fn reset(&mut self) -> Result<(), BraheError>;
    
    /// Get current epoch of the propagator
    fn current_epoch(&self) -> Epoch;
    
    /// Get reference to current state
    fn current_state(&self) -> &OrbitState;
    
    /// Get reference to initial state
    fn initial_state(&self) -> &OrbitState;
    
    /// Set initial state and reset propagator
    fn set_initial_state(&mut self, state: OrbitState) -> Result<(), BraheError>;
    
    /// Set initial conditions from components
    /// 
    /// # Arguments
    /// * `epoch` - Initial epoch
    /// * `state` - 6-element state vector
    /// * `frame` - Reference frame
    /// * `orbit_type` - Type of orbital representation
    /// * `angle_format` - Format for angular elements
    fn set_initial_conditions(
        &mut self, 
        epoch: Epoch, 
        state: Vector6<f64>,
        frame: crate::trajectories::OrbitFrame,
        orbit_type: crate::trajectories::OrbitStateType,
        angle_format: crate::trajectories::AngleFormat,
    ) -> Result<(), BraheError>;
    
    /// Get step size in seconds
    fn step_size(&self) -> f64;
    
    /// Set step size in seconds
    fn set_step_size(&mut self, step_size: f64);
    
    /// Step forward by the default step size
    /// 
    /// # Returns
    /// Reference to the updated state after stepping
    fn step(&mut self) -> Result<&OrbitState, BraheError>;
    
    /// Step forward by a specified time duration
    /// 
    /// # Arguments
    /// * `step_size` - Time step in seconds
    /// 
    /// # Returns
    /// Reference to the updated state after stepping
    fn step_by(&mut self, step_size: f64) -> Result<&OrbitState, BraheError>;
    
    /// Step forward by default step size for a specified number of steps
    /// 
    /// # Arguments
    /// * `num_steps` - Number of steps to take
    /// 
    /// # Returns
    /// Vector of states after each step
    fn propagate_steps(&mut self, num_steps: usize) -> Result<Vec<OrbitState>, BraheError>;
    
    /// Step forward by default step size until current epoch is past target epoch
    /// 
    /// # Arguments
    /// * `target_epoch` - The epoch to step past using default step size
    /// 
    /// # Returns
    /// Reference to the final state (which will be past target_epoch)
    /// 
    /// # Note
    /// Unlike `propagate_to()` which propagates to the exact epoch, this method
    /// steps using the default step size until the target is exceeded.
    fn step_to(&mut self, target_epoch: Epoch) -> Result<&OrbitState, BraheError>;
    
    /// Get reference to accumulated trajectory
    fn trajectory(&self) -> &Trajectory<OrbitState>;
    
    /// Get mutable reference to accumulated trajectory
    fn trajectory_mut(&mut self) -> &mut Trajectory<OrbitState>;
    
    /// Set maximum trajectory size for memory management
    fn set_max_trajectory_size(&mut self, max_size: Option<usize>);
    
    /// Set maximum age of states to keep (in seconds) for time-based eviction
    fn set_max_trajectory_age(&mut self, max_age: Option<f64>);
    
    /// Set eviction policy for trajectory memory management
    fn set_eviction_policy(&mut self, policy: TrajectoryEvictionPolicy);
}

/// Trait for orbit interpolators providing trajectory-based state access
pub trait OrbitInterpolator {
    /// Get the number of states in the interpolator
    fn num_states(&self) -> usize;
    
    /// Get state at a specific epoch using interpolation
    fn state_at_epoch(&self, epoch: &Epoch) -> Result<OrbitState, BraheError>;
    
    /// Get the time span covered by the interpolator
    fn time_span(&self) -> Result<(Epoch, Epoch), BraheError>;
    
    /// Check if an epoch is within the interpolation range
    fn contains_epoch(&self, epoch: &Epoch) -> bool;
}
