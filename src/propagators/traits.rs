/*!
 * Propagator traits with clean interfaces and vector-based operations
 */

use nalgebra::{DVector, Vector6};

use crate::constants::AngleFormat;
use crate::time::Epoch;
use crate::trajectories::traits::{OrbitFrame, OrbitRepresentation};
use crate::utils::BraheError;

// Re-export state provider traits for convenience
pub use crate::utils::state_providers::{
    DCovarianceProvider, DIdentifiableStateProvider, DOrbitCovarianceProvider, DOrbitStateProvider,
    DStateProvider, SCovarianceProvider, SIdentifiableStateProvider, SOrbitCovarianceProvider,
    SOrbitStateProvider, SStateProvider,
};

/// Core trait for state propagators with static-sized (6D) state vectors
///
/// This trait provides a clean interface for state propagators that work with
/// compile-time sized 6-element state vectors (typically position and velocity).
/// It focuses purely on state propagation without orbit-specific initialization.
///
/// See also: [`DStatePropagator`] for dynamic-sized version, [`SOrbitPropagator`] for orbit-specific initialization
pub trait SStatePropagator {
    /// Step forward by the default step size
    /// Returns Result indicating success/failure, use getters to access state
    fn step(&mut self) {
        self.step_by(self.step_size());
    }

    /// Step by a specified time duration (positive or negative)
    ///
    /// # Arguments
    /// * `step_size` - Time step in seconds (positive for forward, negative for backward)
    fn step_by(&mut self, step_size: f64);

    /// Step past a specified target epoch in the current propagation direction.
    ///
    /// The propagation direction is determined by the sign of `step_size()`:
    /// - For forward propagation (step_size > 0): steps until current_epoch >= target_epoch
    /// - For backward propagation (step_size < 0): steps until current_epoch <= target_epoch
    ///
    /// If the target is in the opposite direction of propagation, no action is taken.
    fn step_past(&mut self, target_epoch: Epoch) {
        let current = self.current_epoch();
        let step = self.step_size();

        if step >= 0.0 {
            // Forward propagation - only proceed if target is in the future
            if target_epoch <= current {
                return;
            }
            while self.current_epoch() < target_epoch {
                self.step();
            }
        } else {
            // Backward propagation - only proceed if target is in the past
            if target_epoch >= current {
                return;
            }
            while self.current_epoch() > target_epoch {
                self.step();
            }
        }
    }

    /// Step by default step size for a specified number of steps
    ///
    /// # Arguments
    /// * `num_steps` - Number of steps to take
    fn propagate_steps(&mut self, num_steps: usize) {
        for _ in 0..num_steps {
            self.step();
        }
    }

    /// Propagate to a specific target epoch, respecting propagation direction.
    ///
    /// The propagation direction is determined by the sign of `step_size()`:
    /// - For forward propagation (step_size > 0): propagates forward to target
    /// - For backward propagation (step_size < 0): propagates backward to target
    ///
    /// If the target is in the opposite direction of propagation, no action is taken.
    ///
    /// # Arguments
    /// * `target_epoch` - The epoch to propagate to
    fn propagate_to(&mut self, target_epoch: Epoch) {
        let mut current_epoch = self.current_epoch();
        let step = self.step_size();

        if step >= 0.0 {
            // Forward propagation - only proceed if target is in the future
            if target_epoch <= current_epoch {
                return;
            }
            while current_epoch < target_epoch {
                let remaining_time = target_epoch - current_epoch;
                let step_size = remaining_time.min(step);

                // Guard against very small steps to avoid infinite loops
                if step_size <= 1e-9 {
                    break;
                }

                self.step_by(step_size);
                current_epoch = self.current_epoch();
            }
        } else {
            // Backward propagation - only proceed if target is in the past
            if target_epoch >= current_epoch {
                return;
            }
            while current_epoch > target_epoch {
                let remaining_time = current_epoch - target_epoch;
                let step_size = -(remaining_time.min(step.abs()));

                // Guard against very small steps to avoid infinite loops
                if step_size.abs() <= 1e-9 {
                    break;
                }

                self.step_by(step_size);
                current_epoch = self.current_epoch();
            }
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

/// Core trait for state propagators with dynamic-sized state vectors
///
/// This trait provides a clean interface for state propagators that work with
/// runtime-sized state vectors (DVector). Useful for propagators with non-standard
/// state dimensions (e.g., including STM, variational equations, etc.).
/// It focuses purely on state propagation without orbit-specific initialization.
///
/// See also: [`SStatePropagator`] for static-sized (6D) version, [`DOrbitPropagator`] for orbit-specific initialization
pub trait DStatePropagator {
    /// Step forward by the default step size
    /// Returns Result indicating success/failure, use getters to access state
    fn step(&mut self) {
        self.step_by(self.step_size());
    }

    /// Step by a specified time duration (positive or negative)
    ///
    /// # Arguments
    /// * `step_size` - Time step in seconds (positive for forward, negative for backward)
    fn step_by(&mut self, step_size: f64);

    /// Step past a specified target epoch in the current propagation direction.
    ///
    /// The propagation direction is determined by the sign of `step_size()`:
    /// - For forward propagation (step_size > 0): steps until current_epoch >= target_epoch
    /// - For backward propagation (step_size < 0): steps until current_epoch <= target_epoch
    ///
    /// If the target is in the opposite direction of propagation, no action is taken.
    fn step_past(&mut self, target_epoch: Epoch) {
        let current = self.current_epoch();
        let step = self.step_size();

        if step >= 0.0 {
            // Forward propagation - only proceed if target is in the future
            if target_epoch <= current {
                return;
            }
            while self.current_epoch() < target_epoch {
                self.step();
            }
        } else {
            // Backward propagation - only proceed if target is in the past
            if target_epoch >= current {
                return;
            }
            while self.current_epoch() > target_epoch {
                self.step();
            }
        }
    }

    /// Step by default step size for a specified number of steps
    ///
    /// # Arguments
    /// * `num_steps` - Number of steps to take
    fn propagate_steps(&mut self, num_steps: usize) {
        for _ in 0..num_steps {
            self.step();
        }
    }

    /// Propagate to a specific target epoch, respecting propagation direction.
    ///
    /// The propagation direction is determined by the sign of `step_size()`:
    /// - For forward propagation (step_size > 0): propagates forward to target
    /// - For backward propagation (step_size < 0): propagates backward to target
    ///
    /// If the target is in the opposite direction of propagation, no action is taken.
    ///
    /// # Arguments
    /// * `target_epoch` - The epoch to propagate to
    fn propagate_to(&mut self, target_epoch: Epoch) {
        let mut current_epoch = self.current_epoch();
        let step = self.step_size();

        if step >= 0.0 {
            // Forward propagation - only proceed if target is in the future
            if target_epoch <= current_epoch {
                return;
            }
            while current_epoch < target_epoch {
                let remaining_time = target_epoch - current_epoch;
                let step_size = remaining_time.min(step);

                // Guard against very small steps to avoid infinite loops
                if step_size <= 1e-9 {
                    break;
                }

                self.step_by(step_size);
                current_epoch = self.current_epoch();
            }
        } else {
            // Backward propagation - only proceed if target is in the past
            if target_epoch >= current_epoch {
                return;
            }
            while current_epoch > target_epoch {
                let remaining_time = current_epoch - target_epoch;
                let step_size = -(remaining_time.min(step.abs()));

                // Guard against very small steps to avoid infinite loops
                if step_size.abs() <= 1e-9 {
                    break;
                }

                self.step_by(step_size);
                current_epoch = self.current_epoch();
            }
        }
    }

    // Getter methods for accessing state

    /// Get current epoch
    fn current_epoch(&self) -> Epoch;

    /// Get current state as a dynamic vector
    fn current_state(&self) -> DVector<f64>;

    /// Get initial epoch
    fn initial_epoch(&self) -> Epoch;

    /// Get initial state as a dynamic vector
    fn initial_state(&self) -> DVector<f64>;

    /// Get state dimension
    fn state_dim(&self) -> usize;

    // Configuration methods
    /// Get step size in seconds
    fn step_size(&self) -> f64;

    /// Set step size in seconds
    fn set_step_size(&mut self, step_size: f64);

    /// Reset propagator to initial conditions
    fn reset(&mut self);

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

/// Orbit-specific propagator trait that extends [`SStatePropagator`] with orbital initialization
///
/// This trait adds orbit-specific initialization capabilities to the base state propagator.
/// Types implementing this trait can accept initial conditions in various orbital frames
/// and representations (Cartesian, Keplerian, etc.).
///
/// Not all propagators support changing initial conditions (e.g., SGP4/TLE-based propagators
/// derive their state from TLE data). Such propagators should only implement [`SStatePropagator`].
///
/// See also: [`DOrbitPropagator`] for dynamic-sized version
pub trait SOrbitPropagator: SStatePropagator {
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
}

/// Orbit-specific propagator trait that extends [`DStatePropagator`] with orbital initialization
///
/// This trait adds orbit-specific initialization capabilities to the base state propagator.
/// Types implementing this trait can accept initial conditions in various orbital frames
/// and representations (Cartesian, Keplerian, etc.).
///
/// Not all propagators support changing initial conditions (e.g., SGP4/TLE-based propagators
/// derive their state from TLE data). Such propagators should only implement [`DStatePropagator`].
///
/// See also: [`SOrbitPropagator`] for static-sized (6D) version
pub trait DOrbitPropagator: DStatePropagator {
    /// Set initial conditions from components
    ///
    /// # Arguments
    /// * `epoch` - Initial epoch
    /// * `state` - State vector
    /// * `frame` - Reference frame
    /// * `representation` - Type of orbital representation
    /// * `angle_format` - Format for angular elements (None for Cartesian, Some(format) for Keplerian)
    ///
    /// # Panics
    /// May panic if the combination of frame, representation, and angle_format is incompatible
    fn set_initial_conditions(
        &mut self,
        epoch: Epoch,
        state: DVector<f64>,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: Option<AngleFormat>,
    );
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
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
    fn test_sorbit_propagator_step() {
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
    fn test_sorbit_propagator_step_past() {
        let mut prop = create_test_propagator();
        let initial_epoch = prop.current_epoch();
        let target = initial_epoch + 250.0; // 250 seconds in the future

        // Use step_past to reach target
        prop.step_past(target);

        // Verify we've gone past the target
        assert!(prop.current_epoch() >= target);
    }

    #[test]
    fn test_sorbit_propagator_step_past_already_past() {
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
    fn test_sorbit_propagator_propagate_steps() {
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
    fn test_sorbit_propagator_propagate_to() {
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
    fn test_sorbit_propagator_propagate_to_past_epoch() {
        let mut prop = create_test_propagator();
        let initial_epoch = prop.current_epoch();

        // Try to propagate to a past epoch
        let past = initial_epoch - 100.0;
        prop.propagate_to(past);

        // Should not have changed
        assert_eq!(prop.current_epoch(), initial_epoch);
    }

    #[test]
    fn test_sorbit_propagator_propagate_trajectory() {
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
    fn test_sorbit_state_provider_states() {
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
        let states = prop.states(&epochs).unwrap();

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
    fn test_sorbit_state_provider_states_eci() {
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
        let states = prop.states_eci(&epochs).unwrap();

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
    fn test_sorbit_state_provider_states_ecef() {
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
        let states = prop.states_ecef(&epochs).unwrap();

        assert_eq!(states.len(), 3);

        // Verify every state vector is different
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                assert!(!states[i].relative_eq(&states[j], 1e-9, 1e-9));
            }
        }
    }

    #[test]
    fn test_sorbit_state_provider_states_gcrf() {
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
        let states = prop.states_gcrf(&epochs).unwrap();

        assert_eq!(states.len(), 3);

        // Verify every state vector is different
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                assert!(!states[i].relative_eq(&states[j], 1e-9, 1e-9));
            }
        }
    }

    #[test]
    fn test_sorbit_state_provider_states_itrf() {
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
        let states = prop.states_itrf(&epochs).unwrap();

        assert_eq!(states.len(), 3);

        // Verify every state vector is different
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                assert!(!states[i].relative_eq(&states[j], 1e-9, 1e-9));
            }
        }
    }
}
