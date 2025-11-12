/*!
 * Propagator traits with clean interfaces and vector-based operations
 */

use nalgebra::Vector6;

use crate::constants::AngleFormat;
use crate::time::Epoch;
use crate::trajectories::traits::{OrbitFrame, OrbitRepresentation};
use crate::utils::BraheError;
use crate::utils::identifiable::Identifiable;

/// Core trait for orbit propagators with clean interface
pub trait OrbitPropagator {
    /// Step forward by the default step size
    /// Returns Result indicating success/failure, use getters to access state
    fn step(&mut self) {
        self.step_by(self.step_size());
    }

    /// Step forward by a specified time duration
    ///
    /// # Arguments
    /// * `step_size` - Time step in seconds
    fn step_by(&mut self, step_size: f64);

    /// Step past a specified target epoch
    /// If the target epoch is before or equal to the current epoch, no action is taken
    fn step_past(&mut self, target_epoch: Epoch) {
        while self.current_epoch() < target_epoch {
            self.step();
        }
    }

    /// Step forward by default step size for a specified number of steps
    ///
    /// # Arguments
    /// * `num_steps` - Number of steps to take
    fn propagate_steps(&mut self, num_steps: usize) {
        for _ in 0..num_steps {
            self.step();
        }
    }

    /// Propagate to a specific target epoch
    ///
    /// # Arguments
    /// * `target_epoch` - The epoch to propagate to
    fn propagate_to(&mut self, target_epoch: Epoch) {
        let mut current_epoch = self.current_epoch();

        while current_epoch < target_epoch {
            // Calculate step size to not overshoot
            let remaining_time = target_epoch - current_epoch;
            let step_size = remaining_time.min(self.step_size());

            // Guard against very small steps to avoid infinite loops
            if step_size <= 1e-9 {
                break;
            }

            self.step_by(step_size);
            current_epoch = self.current_epoch();
        }
    }

    // Getter methods for accessing state

    /// Get current epoch
    fn current_epoch(&self) -> Epoch;

    /// Get current state as a 6D vector
    fn current_state(&self) -> Vector6<f64>;

    /// Get initial epoch
    fn initial_epoch(&self) -> Epoch;

    /// Get initial state as a 6D vector
    fn initial_state(&self) -> Vector6<f64>;

    // Configuration methods
    /// Get step size in seconds
    fn step_size(&self) -> f64;

    /// Set step size in seconds
    fn set_step_size(&mut self, step_size: f64);

    /// Reset propagator to initial conditions
    fn reset(&mut self);

    /// Set initial conditions from components
    ///
    /// # Arguments
    /// * `epoch` - Initial epoch
    /// * `state` - 6-element state vector
    /// * `frame` - Reference frame
    /// * `representation` - Type of orbital representation
    /// * `angle_format` - Format for angular elements (None for Cartesian, Some(format) for Keplerian)
    ///
    /// # Panics
    /// May panic if the combination of frame, representation, and angle_format is incompatible
    fn set_initial_conditions(
        &mut self,
        epoch: Epoch,
        state: Vector6<f64>,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: Option<AngleFormat>,
    );

    /// Propagate and populate trajectory at multiple epochs
    ///
    /// # Arguments
    /// * `epochs` - Epochs to propagate to and add to trajectory
    fn propagate_trajectory(&mut self, epochs: &[Epoch]) {
        for &epoch in epochs {
            self.propagate_to(epoch);
        }
    }

    // Memory management for trajectory
    /// Set eviction policy to keep a maximum number of states
    fn set_eviction_policy_max_size(&mut self, max_size: usize) -> Result<(), BraheError>;

    /// Set eviction policy to keep states within a maximum age
    fn set_eviction_policy_max_age(&mut self, max_age: f64) -> Result<(), BraheError>;
}

/// Trait for analytic orbital propagators that can compute states directly at any epoch
/// without requiring numerical integration. This trait is designed for propagators like
/// SGP4/TLE that have closed-form solutions.
pub trait StateProvider {
    /// Returns the state at the given epoch as a 6-element vector in the propagator's
    /// native coordinate frame and representation.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// A 6-element vector containing the state in the propagator's native output format
    fn state(&self, epoch: Epoch) -> Vector6<f64>;

    /// Returns the state at the given epoch in Earth-Centered Inertial (ECI)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// A 6-element vector containing position (km) and velocity (km/s) components
    /// in the ECI frame.
    fn state_eci(&self, epoch: Epoch) -> Vector6<f64>;

    /// Returns the state at the given epoch in Earth-Centered Earth-Fixed (ECEF)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// A 6-element vector containing position (km) and velocity (km/s) components
    /// in the ECEF frame.
    fn state_ecef(&self, epoch: Epoch) -> Vector6<f64>;

    /// Returns the state at the given epoch in Geocentric Celestial Reference Frame (GCRF)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// A 6-element vector containing position (m) and velocity (m/s) components
    /// in the GCRF frame.
    fn state_gcrf(&self, epoch: Epoch) -> Vector6<f64>;

    /// Returns the state at the given epoch in International Terrestrial Reference Frame (ITRF)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// A 6-element vector containing position (m) and velocity (m/s) components
    /// in the ITRF frame.
    fn state_itrf(&self, epoch: Epoch) -> Vector6<f64>;

    /// Returns the state at the given epoch in Earth Mean Equator and Equinox of J2000.0 (EME2000)
    /// Cartesian coordinates.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// A 6-element vector containing position (m) and velocity (m/s) components
    /// in the EME2000 frame.
    fn state_eme2000(&self, epoch: Epoch) -> Vector6<f64>;

    /// Returns the state at the given epoch as osculating orbital elements.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    /// * `angle_format` - Angle format for angular elements (Degrees or Radians)
    ///
    /// # Returns
    /// A 6-element vector containing osculating Keplerian elements [a, e, i, RAAN, arg_periapsis, mean_anomaly]
    /// where angles are in the specified format
    fn state_as_osculating_elements(&self, epoch: Epoch, angle_format: AngleFormat)
    -> Vector6<f64>;

    /// Returns states at multiple epochs in the propagator's native coordinate frame
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * Vector of 6-element vectors containing states in the propagator's native output format
    fn states(&self, epochs: &[Epoch]) -> Vec<Vector6<f64>> {
        epochs.iter().map(|&epoch| self.state(epoch)).collect()
    }

    /// Returns states at multiple epochs in Earth-Centered Inertial (ECI)
    /// Cartesian coordinates as a STrajectory6.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * Vector of 6-element vectors containing position (m) and velocity (m/s) components
    fn states_eci(&self, epochs: &[Epoch]) -> Vec<Vector6<f64>> {
        epochs.iter().map(|&epoch| self.state_eci(epoch)).collect()
    }

    /// Returns states at multiple epochs in Earth-Centered Earth-Fixed (ECEF)
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * Vector of 6-element vectors containing position (m) and velocity (m/s) components
    fn states_ecef(&self, epochs: &[Epoch]) -> Vec<Vector6<f64>> {
        epochs.iter().map(|&epoch| self.state_ecef(epoch)).collect()
    }

    /// Returns states at multiple epochs in Geocentric Celestial Reference Frame (GCRF)
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * Vector of 6-element vectors containing position (m) and velocity (m/s) components
    ///   in the GCRF frame.
    fn states_gcrf(&self, epochs: &[Epoch]) -> Vec<Vector6<f64>> {
        epochs.iter().map(|&epoch| self.state_gcrf(epoch)).collect()
    }

    /// Returns states at multiple epochs in International Terrestrial Reference Frame (ITRF)
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * Vector of 6-element vectors containing position (m) and velocity (m/s) components
    ///   in the ITRF frame.
    fn states_itrf(&self, epochs: &[Epoch]) -> Vec<Vector6<f64>> {
        epochs.iter().map(|&epoch| self.state_itrf(epoch)).collect()
    }

    /// Returns states at multiple epochs in Earth Mean Equator and Equinox of J2000.0 (EME2000)
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * Vector of 6-element vectors containing position (m) and velocity (m/s) components
    ///   in the EME2000 frame.
    fn states_eme2000(&self, epochs: &[Epoch]) -> Vec<Vector6<f64>> {
        epochs
            .iter()
            .map(|&epoch| self.state_eme2000(epoch))
            .collect()
    }

    /// Returns states at multiple epochs as osculating orbital elements.
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    /// * `angle_format` - Angle format for angular elements (Degrees or Radians)
    ///
    /// # Returns
    /// * Vector of 6-element vectors containing osculating Keplerian elements
    fn states_as_osculating_elements(
        &self,
        epochs: &[Epoch],
        angle_format: AngleFormat,
    ) -> Vec<Vector6<f64>> {
        epochs
            .iter()
            .map(|&epoch| self.state_as_osculating_elements(epoch, angle_format))
            .collect()
    }
}

/// Combined trait for state providers with identity tracking.
///
/// This supertrait combines `StateProvider` and `Identifiable`, used primarily
/// in access computation where satellite identity needs to be tracked alongside
/// orbital state computation.
///
/// # Automatic Implementation
///
/// This trait is automatically implemented for any type that implements both
/// `StateProvider` and `Identifiable` via a blanket implementation.
///
/// # Examples
///
/// ```
/// use brahe::propagators::{KeplerianPropagator, SGPPropagator};
/// use brahe::traits::IdentifiableStateProvider;
///
/// // Both propagators implement IdentifiableStateProvider automatically
/// fn accepts_identified_provider<P: IdentifiableStateProvider>(provider: &P) {
///     // Can use both StateProvider and Identifiable methods
/// }
/// ```
pub trait IdentifiableStateProvider: StateProvider + Identifiable {}

// Blanket implementation for any type implementing both traits
impl<T: StateProvider + Identifiable> IdentifiableStateProvider for T {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbits;
    use crate::propagators::KeplerianPropagator;
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::testing::setup_global_test_eop;
    use nalgebra::Vector6;

    fn create_test_propagator() -> KeplerianPropagator {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        // LEO orbit: a=6878 km, e=0.01, i=45 deg
        let state = Vector6::new(
            6878000.0, // m
            0.01,
            45.0_f64.to_radians(),
            0.0,
            0.0,
            0.0,
        );

        KeplerianPropagator::new(
            epoch,
            state,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Radians),
            60.0,
        )
    }

    #[test]
    fn test_orbit_propagator_step() {
        let mut prop = create_test_propagator();
        let initial_epoch = prop.current_epoch();
        let step_size = prop.step_size();

        // Step forward using default step() method
        prop.step();

        // Verify epoch advanced by step_size
        let new_epoch = prop.current_epoch();
        assert!((new_epoch - initial_epoch - step_size).abs() < 1e-6);
    }

    #[test]
    fn test_orbit_propagator_step_past() {
        let mut prop = create_test_propagator();
        let initial_epoch = prop.current_epoch();
        let target = initial_epoch + 250.0; // 250 seconds in the future

        // Use step_past to reach target
        prop.step_past(target);

        // Verify we've gone past the target
        assert!(prop.current_epoch() >= target);
    }

    #[test]
    fn test_orbit_propagator_step_past_already_past() {
        let mut prop = create_test_propagator();
        let initial_epoch = prop.current_epoch();

        // Step forward first
        prop.step_by(120.0);
        let current = prop.current_epoch();

        // Try to step_past to an epoch in the past
        prop.step_past(initial_epoch);

        // Should not have changed (already past)
        assert_eq!(prop.current_epoch(), current);
    }

    #[test]
    fn test_orbit_propagator_propagate_steps() {
        let mut prop = create_test_propagator();
        let initial_epoch = prop.current_epoch();
        let step_size = prop.step_size();
        let num_steps = 5;

        // Propagate for 5 steps
        prop.propagate_steps(num_steps);

        // Verify epoch advanced by num_steps * step_size
        let new_epoch = prop.current_epoch();
        let expected_time = step_size * num_steps as f64;
        assert!((new_epoch - initial_epoch - expected_time).abs() < 1e-3);
    }

    #[test]
    fn test_orbit_propagator_propagate_to() {
        let mut prop = create_test_propagator();
        let initial_epoch = prop.current_epoch();
        let target = initial_epoch + 157.0; // Not a multiple of step_size

        // Propagate to exact target
        prop.propagate_to(target);

        // Verify we reached the target (within tolerance)
        let final_epoch = prop.current_epoch();
        assert!((final_epoch - target).abs() < 1e-6);
    }

    #[test]
    fn test_orbit_propagator_propagate_to_past_epoch() {
        let mut prop = create_test_propagator();
        let initial_epoch = prop.current_epoch();

        // Try to propagate to a past epoch
        let past = initial_epoch - 100.0;
        prop.propagate_to(past);

        // Should not have changed
        assert_eq!(prop.current_epoch(), initial_epoch);
    }

    #[test]
    fn test_orbit_propagator_propagate_trajectory() {
        let mut prop = create_test_propagator();
        let initial_epoch = prop.current_epoch();

        // Create array of target epochs
        let epochs = vec![
            initial_epoch + 60.0,
            initial_epoch + 120.0,
            initial_epoch + 180.0,
        ];

        // Propagate through all epochs
        prop.propagate_trajectory(&epochs);

        // Verify final epoch is the last target
        let final_epoch = prop.current_epoch();
        assert!((final_epoch - epochs[2]).abs() < 1e-6);
    }

    #[test]
    fn test_state_provider_states() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        // Initialize in degrees: a, e, i, raan, argp, M
        let elements = Vector6::new(6878000.0, 0.01, 45.0, 15.0, 30.0, 60.0);
        let prop = KeplerianPropagator::new(
            epoch,
            elements,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
            60.0,
        );

        // Create multiple epochs
        let epochs = vec![epoch, epoch + 120.0, epoch + 240.0];

        // Get states for all epochs (should be in degrees since that's the output format)
        let states = prop.states(&epochs);

        // Verify we got the right number of states
        assert_eq!(states.len(), 3);

        // Calculate mean motion using library function
        let a = elements[0];
        let mean_motion_deg_per_sec = orbits::mean_motion(a, AngleFormat::Degrees);

        // For each state, verify Keplerian elements behavior
        for (idx, &state) in states.iter().enumerate() {
            let time_elapsed = 120.0 * idx as f64; // seconds

            // Orbital elements should remain constant (a, e, i, raan, argp)
            assert!((state[0] - elements[0]).abs() < 1.0); // a within 1 m
            assert!((state[1] - elements[1]).abs() < 1e-6); // e constant
            assert!((state[2] - elements[2]).abs() < 1e-6); // i constant (deg)
            assert!((state[3] - elements[3]).abs() < 1e-6); // raan constant (deg)
            assert!((state[4] - elements[4]).abs() < 1e-6); // argp constant (deg)

            // Mean anomaly should advance by mean_motion * time
            let expected_ma = (elements[5] + mean_motion_deg_per_sec * time_elapsed) % 360.0;
            let actual_ma = state[5] % 360.0;
            // Allow for wrapping around 360
            let ma_diff = (expected_ma - actual_ma).abs();
            let ma_diff_wrapped = ma_diff.min((360.0 - ma_diff).abs());
            assert!(ma_diff_wrapped < 0.01); // within 0.01 degrees
        }
    }

    #[test]
    fn test_state_provider_states_eci() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let elements = Vector6::new(6878000.0, 0.01, 45.0, 15.0, 30.0, 60.0);
        let prop = KeplerianPropagator::new(
            epoch,
            elements,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
            60.0,
        );

        let epochs = vec![epoch, epoch + 120.0, epoch + 240.0];
        let states = prop.states_eci(&epochs);

        assert_eq!(states.len(), 3);

        // Verify all 6 state elements are different from the first state
        let first_state = states[0];
        for state in states.iter().skip(1) {
            // At least one element must be different
            let mut all_same = true;
            for i in 0..6 {
                if (state[i] - first_state[i]).abs() > 1e-9 {
                    all_same = false;
                    break;
                }
            }
            assert!(!all_same, "State should be different from first state");
        }
    }

    #[test]
    fn test_state_provider_states_ecef() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let elements = Vector6::new(6878000.0, 0.01, 45.0, 15.0, 30.0, 60.0);
        let prop = KeplerianPropagator::new(
            epoch,
            elements,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
            60.0,
        );

        let epochs = vec![epoch, epoch + 120.0, epoch + 240.0];
        let states = prop.states_ecef(&epochs);

        assert_eq!(states.len(), 3);

        // Verify every state vector is different
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                assert!(!states[i].relative_eq(&states[j], 1e-9, 1e-9));
            }
        }
    }

    #[test]
    fn test_state_provider_states_gcrf() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let elements = Vector6::new(6878000.0, 0.01, 45.0, 15.0, 30.0, 60.0);
        let prop = KeplerianPropagator::new(
            epoch,
            elements,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
            60.0,
        );

        let epochs = vec![epoch, epoch + 120.0, epoch + 240.0];
        let states = prop.states_gcrf(&epochs);

        assert_eq!(states.len(), 3);

        // Verify every state vector is different
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                assert!(!states[i].relative_eq(&states[j], 1e-9, 1e-9));
            }
        }
    }

    #[test]
    fn test_state_provider_states_itrf() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let elements = Vector6::new(6878000.0, 0.01, 45.0, 15.0, 30.0, 60.0);
        let prop = KeplerianPropagator::new(
            epoch,
            elements,
            OrbitFrame::ECI,
            OrbitRepresentation::Keplerian,
            Some(AngleFormat::Degrees),
            60.0,
        );

        let epochs = vec![epoch, epoch + 120.0, epoch + 240.0];
        let states = prop.states_itrf(&epochs);

        assert_eq!(states.len(), 3);

        // Verify every state vector is different
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                assert!(!states[i].relative_eq(&states[j], 1e-9, 1e-9));
            }
        }
    }
}
