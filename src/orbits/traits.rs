/*!
* The orbits `traits` module provides traits for working with orbital state information.
* It includes the `OrbitalState` trait, which defines methods for retrieving position and velocity
* vectors, as well as the `OrbitalStateWithTime` trait, which extends `OrbitalState` to include
* time information. These traits are designed to be implemented by types that represent
* orbital states, such as those derived from Two-Line Element (TLE) data or other
* orbital models.
*/

use crate::time::Epoch;
use crate::trajectories::{AngleFormat, OrbitFrame, OrbitState, OrbitStateType, Trajectory};
use nalgebra as na;

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
