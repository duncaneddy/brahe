use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::ops::Index;

use crate::time::Epoch;
use crate::trajectories::state::State;
use crate::utils::BraheError;

/// Enumeration of interpolation methods for trajectory states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterpolationMethod {
    /// No interpolation, returns the nearest state
    None,
    /// Linear interpolation between states
    Linear,
    /// Cubic spline interpolation
    CubicSpline,
    /// Lagrange polynomial interpolation
    Lagrange,
    /// Hermite interpolation
    Hermite,
}

/// Enumeration of propagator types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropagatorType {
    /// SGP4 propagator
    SGP4,
    /// Numerical propagator
    Numerical,
    /// Analytical propagator (e.g., J2)
    Analytical,
    /// Ephemeris (not propagated, loaded from file)
    Ephemeris,
}

/// Structure representing a collection of states over time
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory<S: State> {
    /// Collection of states
    pub states: Vec<S>,

    /// Propagator that generated this trajectory (optional)
    pub propagator_type: Option<PropagatorType>,

    /// Interpolation method to use when retrieving states
    pub interpolation_method: InterpolationMethod,
}

impl<S: State + Serialize + DeserializeOwned> Trajectory<S> {
    /// Create a new empty trajectory
    pub fn new(interpolation_method: InterpolationMethod) -> Self {
        Self {
            states: Vec::new(),
            propagator_type: None,
            interpolation_method,
        }
    }

    /// Create a trajectory from a vector of states
    pub fn from_states(
        states: Vec<S>,
        interpolation_method: InterpolationMethod,
    ) -> Result<Self, BraheError> {
        if states.is_empty() {
            return Err(BraheError::Error(
                "Cannot create trajectory from empty states".to_string(),
            ));
        }

        // Ensure all states have the same frame
        let first_frame = states[0].frame();

        for state in &states {
            if state.frame() != first_frame {
                return Err(BraheError::Error(
                    "All states in a trajectory must have the same reference frame".to_string(),
                ));
            }
        }

        // Ensure states are sorted by epoch
        let mut sorted_states = states;
        sorted_states.sort_by(|a, b| a.epoch().partial_cmp(b.epoch()).unwrap());

        Ok(Self {
            states: sorted_states,
            propagator_type: None,
            interpolation_method,
        })
    }

    /// Set the propagator type
    pub fn with_propagator(mut self, propagator_type: PropagatorType) -> Self {
        self.propagator_type = Some(propagator_type);
        self
    }

    /// Add a state to the trajectory
    pub fn add_state(&mut self, state: S) -> Result<(), BraheError> {
        // If the trajectory is empty, just add the state
        if self.states.is_empty() {
            self.states.push(state);
            return Ok(());
        }

        // Check if the state is compatible with existing states
        if state.frame() != self.states[0].frame() {
            return Err(BraheError::Error(
                "Cannot add state with different frame to trajectory".to_string(),
            ));
        }

        // Find the correct position to insert based on epoch
        let mut insert_idx = self.states.len();
        for (i, existing) in self.states.iter().enumerate() {
            if state.epoch() < existing.epoch() {
                insert_idx = i;
                break;
            } else if state.epoch() == existing.epoch() {
                // Replace state if epochs are equal
                self.states[i] = state;
                return Ok(());
            }
        }

        // Insert at the correct position
        self.states.insert(insert_idx, state);
        Ok(())
    }

    /// Get the state at a specific epoch using interpolation - primary API method
    pub fn state_at_epoch(&self, epoch: &Epoch) -> Result<S, BraheError> {
        self.interpolate_to(epoch)
    }

    /// Get the state at a specific epoch using interpolation
    pub fn interpolate_to(&self, epoch: &Epoch) -> Result<S, BraheError> {
        if self.states.is_empty() {
            return Err(BraheError::Error(
                "Cannot interpolate state from empty trajectory".to_string(),
            ));
        }

        // If only one state, return it
        if self.states.len() == 1 {
            return Ok(self.states[0].clone());
        }

        // If epoch is before the first state or after the last state
        if epoch < self.states[0].epoch() {
            return Err(BraheError::Error(
                "Requested epoch is before the first state in trajectory".to_string(),
            ));
        }
        if epoch > self.states.last().unwrap().epoch() {
            return Err(BraheError::Error(
                "Requested epoch is after the last state in trajectory".to_string(),
            ));
        }

        // Find the closest state or states for interpolation
        match self.interpolation_method {
            InterpolationMethod::None => self.nearest_state(epoch),
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

    /// Find the nearest state to the specified epoch
    pub fn nearest_state(&self, epoch: &Epoch) -> Result<S, BraheError> {
        if self.states.is_empty() {
            return Err(BraheError::Error(
                "Cannot find nearest state in empty trajectory".to_string(),
            ));
        }

        let mut nearest_idx = 0;
        let mut min_diff = f64::MAX;

        for (i, state) in self.states.iter().enumerate() {
            let diff = (*epoch - *state.epoch()).abs();
            if diff < min_diff {
                min_diff = diff;
                nearest_idx = i;
            }

            // Optimization: if we're past the epoch and moving away, we can stop
            if i > 0 && state.epoch() > epoch && diff > min_diff {
                break;
            }
        }

        Ok(self.states[nearest_idx].clone())
    }

    /// Find the state occurring before the specified epoch
    pub fn state_before(&self, epoch: &Epoch) -> Result<S, BraheError> {
        if self.states.is_empty() {
            return Err(BraheError::Error(
                "Cannot find state in empty trajectory".to_string(),
            ));
        }

        // If the epoch is before the first state
        if epoch <= self.states[0].epoch() {
            return Err(BraheError::Error(
                "Requested epoch is before or equal to the first state in trajectory".to_string(),
            ));
        }

        let mut before_idx = 0;
        for (i, state) in self.states.iter().enumerate() {
            if state.epoch() < epoch {
                before_idx = i;
            } else {
                break;
            }
        }

        Ok(self.states[before_idx].clone())
    }

    /// Find the state occurring after the specified epoch
    pub fn state_after(&self, epoch: &Epoch) -> Result<S, BraheError> {
        if self.states.is_empty() {
            return Err(BraheError::Error(
                "Cannot find state in empty trajectory".to_string(),
            ));
        }

        // If the epoch is after the last state
        if epoch >= self.states.last().unwrap().epoch() {
            return Err(BraheError::Error(
                "Requested epoch is after or equal to the last state in trajectory".to_string(),
            ));
        }

        for state in self.states.iter() {
            if state.epoch() > epoch {
                return Ok(state.clone());
            }
        }

        // This should never happen given the checks above
        Err(BraheError::Error(
            "Could not find state after the specified epoch".to_string(),
        ))
    }

    /// Get the state at the specified index
    pub fn state_at_index(&self, index: usize) -> Result<S, BraheError> {
        if index >= self.states.len() {
            return Err(BraheError::Error(format!(
                "Index {} out of bounds for trajectory with {} states",
                index,
                self.states.len()
            )));
        }

        Ok(self.states[index].clone())
    }

    /// Interpolate between states using linear interpolation
    fn interpolate_linear(&self, epoch: &Epoch) -> Result<S, BraheError> {
        if self.states.is_empty() {
            return Err(BraheError::Error(
                "Cannot interpolate state from empty trajectory".to_string(),
            ));
        }

        // If only one state, return it
        if self.states.len() == 1 {
            return Ok(self.states[0].clone());
        }

        // Handle boundary cases
        if epoch < self.states[0].epoch() {
            return Err(BraheError::Error(
                "Requested epoch is before the first state in trajectory".to_string(),
            ));
        }

        if epoch > self.states.last().unwrap().epoch() {
            return Err(BraheError::Error(
                "Requested epoch is after the last state in trajectory".to_string(),
            ));
        }

        // Find the two states that bracket the requested epoch
        for i in 0..self.states.len() - 1 {
            let state1 = &self.states[i];
            let state2 = &self.states[i + 1];

            // Check if the requested epoch is between these two states
            if epoch >= state1.epoch() && epoch <= state2.epoch() {
                // Calculate interpolation factor (t)
                let t1 = *state1.epoch();
                let t2 = *state2.epoch();
                let t = *epoch;

                // This computes the normalized interpolation factor (0 to 1)
                let alpha = (t - t1) / (t2 - t1);

                // Use the state's own interpolation method
                return state1.interpolate_with(state2, alpha, epoch);
            }
        }

        // If we reach here, something went wrong with our epoch comparison logic
        Err(BraheError::Error(
            "Failed to find bracketing states for interpolation".to_string(),
        ))
    }

    /// Converts the trajectory to a different reference frame
    pub fn to_frame(&self, frame: &S::Frame) -> Result<Self, BraheError> {
        if self.states.is_empty() {
            return Ok(self.clone());
        }

        if self.states[0].frame() == frame {
            return Ok(self.clone());
        }

        let mut new_states = Vec::with_capacity(self.states.len());

        for state in &self.states {
            new_states.push(state.to_frame(frame)?);
        }

        Trajectory::from_states(new_states, self.interpolation_method).map(|mut traj| {
            traj.propagator_type = self.propagator_type;
            traj
        })
    }

    /// Convert the trajectory to degrees representation
    pub fn as_degrees(&self) -> Result<Self, BraheError> {
        // Convert all states to degrees
        let mut new_states = Vec::with_capacity(self.states.len());
        for state in &self.states {
            new_states.push(state.as_degrees());
        }

        let mut new_trajectory = self.clone();
        new_trajectory.states = new_states;

        Ok(new_trajectory)
    }

    /// Convert the trajectory to radians representation
    pub fn as_radians(&self) -> Result<Self, BraheError> {
        // Convert all states to radians
        let mut new_states = Vec::with_capacity(self.states.len());
        for state in &self.states {
            new_states.push(state.as_radians());
        }

        let mut new_trajectory = self.clone();
        new_trajectory.states = new_states;

        Ok(new_trajectory)
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
        let n_elements = 6; // All states should have 6 elements

        let mut matrix = nalgebra::DMatrix::<f64>::zeros(n_elements, n_epochs);

        for (col_idx, state) in self.states.iter().enumerate() {
            for row_idx in 0..n_elements {
                matrix[(row_idx, col_idx)] = state[row_idx];
            }
        }

        Ok(matrix)
    }

    /// Convert the trajectory to JSON format
    pub fn to_json(&self) -> Result<String, BraheError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| BraheError::Error(format!("Failed to serialize trajectory: {}", e)))
    }

    /// Load a trajectory from JSON format
    pub fn from_json(json: &str) -> Result<Self, BraheError> {
        serde_json::from_str(json)
            .map_err(|e| BraheError::Error(format!("Failed to deserialize trajectory: {}", e)))
    }
}

// Allow indexing into the trajectory directly
impl<S: State> Index<usize> for Trajectory<S> {
    type Output = S;

    fn index(&self, index: usize) -> &Self::Output {
        &self.states[index]
    }
}

// Implement serialization for Trajectory
impl<S: State + Serialize> Serialize for Trajectory<S> {
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        use serde::ser::SerializeStruct;

        let mut s = serializer.serialize_struct("Trajectory", 3)?;
        s.serialize_field("states", &self.states)?;
        s.serialize_field("propagator_type", &self.propagator_type)?;
        s.serialize_field("interpolation_method", &self.interpolation_method)?;
        s.end()
    }
}

// Implement deserialization for Trajectory
impl<'de, S: State + Deserialize<'de>> Deserialize<'de> for Trajectory<S> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct TrajectoryHelper<S> {
            states: Vec<S>,
            propagator_type: Option<PropagatorType>,
            interpolation_method: InterpolationMethod,
        }

        let helper = TrajectoryHelper::deserialize(deserializer)?;

        Ok(Trajectory {
            states: helper.states,
            propagator_type: helper.propagator_type,
            interpolation_method: helper.interpolation_method,
        })
    }
}

/// Iterator for traversing trajectory states
pub struct TrajectoryIter<'a, S: State> {
    trajectory: &'a Trajectory<S>,
    current_index: usize,
}

impl<'a, S: State> Iterator for TrajectoryIter<'a, S> {
    type Item = &'a S;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.trajectory.states.len() {
            let state = &self.trajectory.states[self.current_index];
            self.current_index += 1;
            Some(state)
        } else {
            None
        }
    }
}

impl<S: State> Trajectory<S> {
    /// Returns an iterator over the states in the trajectory
    pub fn iter(&self) -> TrajectoryIter<'_, S> {
        TrajectoryIter {
            trajectory: self,
            current_index: 0,
        }
    }

    /// Returns the number of states in the trajectory
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Returns true if the trajectory is empty
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

/// Mutable iterator for traversing and modifying trajectory states
pub struct TrajectoryIterMut<'a, S: State> {
    states: &'a mut [S],
    current_index: usize,
}

impl<'a, S: State> Iterator for TrajectoryIterMut<'a, S> {
    type Item = &'a mut S;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.states.len() {
            // Safety: We're ensuring the indexes are valid and non-overlapping
            let state = unsafe {
                let ptr = self.states.as_mut_ptr().add(self.current_index);
                &mut *ptr
            };
            self.current_index += 1;
            Some(state)
        } else {
            None
        }
    }
}

impl<S: State> Trajectory<S> {
    /// Returns a mutable iterator over the states in the trajectory
    pub fn iter_mut(&mut self) -> TrajectoryIterMut<'_, S> {
        TrajectoryIterMut {
            states: &mut self.states,
            current_index: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::DEG2RAD;
    use crate::time::{Epoch, TimeSystem};
    use crate::trajectories::orbit_state::{OrbitFrame, OrbitState, OrbitStateType};
    use nalgebra::Vector6;

    use crate::{AngleFormat, RAD2DEG};
    use approx::assert_abs_diff_eq;

    fn create_test_state(time_offset: f64) -> OrbitState {
        // Create a test state at J2000 + time_offset
        let epoch = Epoch::from_jd(2451545.0 + time_offset, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        OrbitState::new(
            epoch,
            state,
            OrbitFrame::ECI,
            OrbitStateType::Cartesian,
            AngleFormat::None,
        )
    }

    #[test]
    fn test_trajectory_creation() {
        let states = vec![
            create_test_state(0.0),
            create_test_state(0.1),
            create_test_state(0.2),
        ];

        let trajectory = Trajectory::from_states(states, InterpolationMethod::Linear).unwrap();

        assert_eq!(trajectory.states.len(), 3);
        assert_eq!(trajectory.interpolation_method, InterpolationMethod::Linear);
        assert_eq!(trajectory.propagator_type, None);
    }

    #[test]
    fn test_trajectory_add_state() {
        let mut trajectory = Trajectory::new(InterpolationMethod::Linear);

        // Add states in order
        trajectory.add_state(create_test_state(0.0)).unwrap();
        trajectory.add_state(create_test_state(0.2)).unwrap();

        // Add a state in between
        trajectory.add_state(create_test_state(0.1)).unwrap();

        assert_eq!(trajectory.states.len(), 3);
        assert_eq!(trajectory.states[0].epoch().jd(), 2451545.0);
        assert_eq!(trajectory.states[1].epoch().jd(), 2451545.1);
        assert_eq!(trajectory.states[2].epoch().jd(), 2451545.2);
    }

    #[test]
    fn test_trajectory_nearest_state() {
        let states = vec![
            create_test_state(0.0),
            create_test_state(1.0),
            create_test_state(2.0),
        ];

        let trajectory = Trajectory::from_states(states, InterpolationMethod::None).unwrap();

        // Request a time exactly at a state
        let state_at_0 = trajectory
            .nearest_state(&Epoch::from_jd(2451545.0, TimeSystem::UTC))
            .unwrap();
        assert_eq!(state_at_0.epoch().jd(), 2451545.0);

        // Request a time halfway between two states
        let state_at_0_5 = trajectory
            .nearest_state(&Epoch::from_jd(2451545.5, TimeSystem::UTC))
            .unwrap();
        // Should return the closest state (0.0)
        assert_eq!(state_at_0_5.epoch().jd(), 2451545.0);

        // Request a time nearer one state
        let state_at_0_25 = trajectory
            .nearest_state(&Epoch::from_jd(2451545.25, TimeSystem::UTC))
            .unwrap();
        // Should return the closest state (0.0)
        assert_eq!(state_at_0_25.epoch().jd(), 2451545.0);

        // Request a time nearer another state
        let state_at_1_75 = trajectory
            .nearest_state(&Epoch::from_jd(2451545.75, TimeSystem::UTC))
            .unwrap();
        // Should return the closest state (1.0)
        assert_eq!(state_at_1_75.epoch().jd(), 2451546.0);
    }

    #[test]
    fn test_trajectory_state_before_after() {
        let states = vec![
            create_test_state(0.0),
            create_test_state(0.1),
            create_test_state(0.2),
        ];

        let trajectory = Trajectory::from_states(states, InterpolationMethod::None).unwrap();

        // Test state_before
        let state_before = trajectory
            .state_before(&Epoch::from_jd(2451545.15, TimeSystem::UTC))
            .unwrap();
        assert_eq!(state_before.epoch().jd(), 2451545.1);

        // Test state_after
        let state_after = trajectory
            .state_after(&Epoch::from_jd(2451545.15, TimeSystem::UTC))
            .unwrap();
        assert_eq!(state_after.epoch().jd(), 2451545.2);

        // Test out of bounds
        assert!(
            trajectory
                .state_before(&Epoch::from_jd(2451545.0, TimeSystem::UTC))
                .is_err()
        );
        assert!(
            trajectory
                .state_after(&Epoch::from_jd(2451545.2, TimeSystem::UTC))
                .is_err()
        );
    }

    #[test]
    fn test_trajectory_state_at_index() {
        let states = vec![
            create_test_state(0.0),
            create_test_state(0.1),
            create_test_state(0.2),
        ];

        let trajectory = Trajectory::from_states(states, InterpolationMethod::None).unwrap();

        // Test valid indices
        let state_0 = trajectory.state_at_index(0).unwrap();
        assert_eq!(state_0.epoch().jd(), 2451545.0);

        let state_2 = trajectory.state_at_index(2).unwrap();
        assert_eq!(state_2.epoch().jd(), 2451545.2);

        // Test out of bounds
        assert!(trajectory.state_at_index(3).is_err());
    }

    #[test]
    fn test_trajectory_indexing() {
        let states = vec![
            create_test_state(0.0),
            create_test_state(0.1),
            create_test_state(0.2),
        ];

        let trajectory = Trajectory::from_states(states, InterpolationMethod::None).unwrap();

        assert_eq!(trajectory[0].epoch().jd(), 2451545.0);
        assert_eq!(trajectory[1].epoch().jd(), 2451545.1);
        assert_eq!(trajectory[2].epoch().jd(), 2451545.2);
    }

    #[test]
    fn test_trajectory_iterator() {
        let states = vec![
            create_test_state(0.0),
            create_test_state(0.1),
            create_test_state(0.2),
        ];

        let trajectory = Trajectory::from_states(states, InterpolationMethod::None).unwrap();

        // Test iterator
        let mut count = 0;
        for state in trajectory.iter() {
            assert_eq!(state.epoch().jd(), 2451545.0 + (count as f64) * 0.1);
            count += 1;
        }
        assert_eq!(count, 3);

        // Test that we can iterate multiple times (creating new iterators)
        count = 0;
        for _state in trajectory.iter() {
            count += 1;
        }
        assert_eq!(count, 3);

        // Test that we can have multiple independent iterators
        let mut iter1 = trajectory.iter();
        let mut iter2 = trajectory.iter();

        // Advance the first iterator
        let _ = iter1.next();
        let _ = iter1.next();

        // The second iterator should still be at the beginning
        assert_eq!(iter2.next().unwrap().epoch().jd(), 2451545.0);
    }

    #[test]
    fn test_trajectory_mutable_iterator() {
        let states = vec![
            create_test_state(0.0),
            create_test_state(0.1),
            create_test_state(0.2),
        ];

        let mut trajectory = Trajectory::from_states(states, InterpolationMethod::None).unwrap();

        // Use mutable iterator to modify states
        // For each state, modify the x position by adding 1000 km
        for state in trajectory.iter_mut() {
            if let OrbitStateType::Cartesian = state.orbit_type {
                // Update x-position by adding 1000 km
                state.state[0] += 1000e3;
            }
        }

        // Verify modifications were applied
        for state in trajectory.iter() {
            let position = state.position().unwrap();
            assert_eq!(position.x, 8000e3); // Original 7000e3 + 1000e3
            // Other components should remain unchanged
            assert_eq!(position.y, 0.0);
            assert_eq!(position.z, 0.0);
        }
    }

    #[test]
    fn test_trajectory_linear_interpolation_cartesian() {
        // Create a trajectory with states having different positions and velocities
        let epoch0 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let epoch1 = Epoch::from_jd(2451546.0, TimeSystem::UTC);

        // Create state with increasing position and velocity
        let state0 = OrbitState::new(
            epoch0.clone(),
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            OrbitFrame::ECI,
            OrbitStateType::Cartesian,
            AngleFormat::None,
        );

        let state1 = OrbitState::new(
            epoch1.clone(),
            Vector6::new(8000e3, 1000e3, 500e3, 100.0, 8.5e3, 50.0),
            OrbitFrame::ECI,
            OrbitStateType::Cartesian,
            AngleFormat::None,
        );

        let trajectory =
            Trajectory::from_states(vec![state0, state1], InterpolationMethod::Linear).unwrap();

        // Test interpolation at 25% between the two states
        let epoch_25 = Epoch::from_jd(2451545.25, TimeSystem::UTC);
        let state_at_25 = trajectory.state_at_epoch(&epoch_25).unwrap();

        // Verify epoch
        assert_eq!(state_at_25.epoch().jd(), 2451545.25);

        // Get the position and velocity
        let position = state_at_25.position().unwrap();
        let velocity = state_at_25.velocity().unwrap();

        // The interpolated values should be 25% between the original states
        assert_abs_diff_eq!(position.x, 7250e3, epsilon = 1.0); // 7000 + 0.25*(8000-7000)
        assert_abs_diff_eq!(position.y, 250e3, epsilon = 1.0); // 0 + 0.25*(1000-0)
        assert_abs_diff_eq!(position.z, 125e3, epsilon = 1.0); // 0 + 0.25*(500-0)

        assert_abs_diff_eq!(velocity.x, 25.0, epsilon = 0.1); // 0 + 0.25*(100-0)
        assert_abs_diff_eq!(velocity.y, 7.75e3, epsilon = 1.0); // 7.5 + 0.25*(8.5-7.5)
        assert_abs_diff_eq!(velocity.z, 12.5, epsilon = 0.1); // 0 + 0.25*(50-0)

        // Test interpolation at 50% between the two states
        let epoch_50 = Epoch::from_jd(2451545.5, TimeSystem::UTC);
        let state_at_50 = trajectory.state_at_epoch(&epoch_50).unwrap();

        // Get the position and velocity
        let position = state_at_50.position().unwrap();
        let velocity = state_at_50.velocity().unwrap();

        // The interpolated values should be 50% between the original states
        assert_abs_diff_eq!(position.x, 7500e3, epsilon = 1.0); // 7000 + 0.5*(8000-7000)
        assert_abs_diff_eq!(position.y, 500e3, epsilon = 1.0); // 0 + 0.5*(1000-0)
        assert_abs_diff_eq!(position.z, 250e3, epsilon = 1.0); // 0 + 0.5*(500-0)

        assert_abs_diff_eq!(velocity.x, 50.0, epsilon = 0.1); // 0 + 0.5*(100-0)
        assert_abs_diff_eq!(velocity.y, 8.0e3, epsilon = 1.0); // 7.5 + 0.5*(8.5-7.5)
        assert_abs_diff_eq!(velocity.z, 25.0, epsilon = 0.1); // 0 + 0.5*(50-0)

        // Test interpolation at 75% between the two states
        let epoch_75 = Epoch::from_jd(2451545.75, TimeSystem::UTC);
        let state_at_75 = trajectory.state_at_epoch(&epoch_75).unwrap();

        // Get the position and velocity
        let position = state_at_75.position().unwrap();
        let velocity = state_at_75.velocity().unwrap();

        // The interpolated values should be 75% between the original states
        assert_abs_diff_eq!(position.x, 7750e3, epsilon = 1.0); // 7000 + 0.75*(8000-7000)
        assert_abs_diff_eq!(position.y, 750e3, epsilon = 1.0); // 0 + 0.75*(1000-0)
        assert_abs_diff_eq!(position.z, 375e3, epsilon = 1.0); // 0 + 0.75*(500-0)

        assert_abs_diff_eq!(velocity.x, 75.0, epsilon = 0.1); // 0 + 0.75*(100-0)
        assert_abs_diff_eq!(velocity.y, 8.25e3, epsilon = 1.0); // 7.5 + 0.75*(8.5-7.5)
        assert_abs_diff_eq!(velocity.z, 37.5, epsilon = 0.1); // 0 + 0.75*(50-0)
    }

    #[test]
    fn test_trajectory_linear_interpolation_keplerian() {
        let epoch0 = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let epoch1 = Epoch::from_jd(2451546.0, TimeSystem::UTC);

        // Test the method specifically handles Keplerian elements correctly
        let kep_state0 = OrbitState::new(
            epoch0.clone(),
            Vector6::new(7000e3, 0.01, 0.0, 359.0 * DEG2RAD, 3.0 * DEG2RAD, 0.0),
            OrbitFrame::ECI,
            OrbitStateType::Keplerian,
            AngleFormat::Radians,
        );

        let kep_state1 = OrbitState::new(
            epoch1.clone(),
            Vector6::new(7200e3, 0.02, 0.1, 3.0 * DEG2RAD, 359.0 * DEG2RAD, 0.0),
            OrbitFrame::ECI,
            OrbitStateType::Keplerian,
            AngleFormat::Radians,
        );

        let kep_trajectory =
            Trajectory::from_states(vec![kep_state0, kep_state1], InterpolationMethod::Linear)
                .unwrap();

        // Test interpolation with Keplerian elements
        let epoch_50 = Epoch::from_jd(2451545.5, TimeSystem::UTC);
        let kep_state_at_50 = kep_trajectory.state_at_epoch(&epoch_50).unwrap();

        // The semi-major axis and eccentricity should be linearly interpolated
        assert_abs_diff_eq!(kep_state_at_50.state[0], 7100e3, epsilon = 1.0); // 7000 + 0.5*(7200-7000)
        assert_abs_diff_eq!(kep_state_at_50.state[1], 0.015, epsilon = 0.0001); // 0.01 + 0.5*(0.02-0.01)

        // The angular elements should be correctly interpolated, respecting angle wrapping
        assert_abs_diff_eq!(kep_state_at_50.state[2], 0.05, epsilon = 0.0001); // 0.0 + 0.5*(0.1-0.0)

        // Test angle wrap handling: mean anomaly wraps from 6.0 (close to 2π) back to 0
        assert_abs_diff_eq!(kep_state_at_50.state[3], 1.0 * DEG2RAD, epsilon = 0.0001); // 359.0 + 0.5*(3.0-359.0) (wrapped)
        assert_abs_diff_eq!(kep_state_at_50.state[4], 1.0 * DEG2RAD, epsilon = 0.0001); // 3.0 + 0.5*(359.0-3.0) (wrapped)

        assert_abs_diff_eq!(kep_state_at_50.state[5], 0.0, epsilon = 0.0001);
    }

    #[test]
    fn test_trajectory_as_degrees() {
        let states = vec![
            create_test_state(0.0),
            create_test_state(0.1),
            create_test_state(0.2),
        ];

        let mut trajectory = Trajectory::from_states(states, InterpolationMethod::None).unwrap();

        // Convert to degrees
        trajectory = trajectory.as_degrees().unwrap();

        // Verify all states are unchanged
        for state in trajectory.iter() {
            let position = state.position().unwrap();
            assert_eq!(position.x, 7000.0e3);
            assert_eq!(position.y, 0.0);
            assert_eq!(position.z, 0.0);

            let velocity = state.velocity().unwrap();
            assert_eq!(velocity.x, 0.0);
            assert_eq!(velocity.y, 7.5e3);
            assert_eq!(velocity.z, 0.0);
        }

        let states = vec![
            OrbitState::new(
                Epoch::from_jd(2451545.0, TimeSystem::UTC),
                Vector6::new(7000.0, 0.0, 10.0, 20.0, 30.0, 40.0),
                OrbitFrame::ECI,
                OrbitStateType::Keplerian,
                AngleFormat::Degrees,
            ),
            OrbitState::new(
                Epoch::from_jd(2451545.1, TimeSystem::UTC),
                Vector6::new(7000.0, 0.0, 10.0, 20.0, 30.0, 40.0),
                OrbitFrame::ECI,
                OrbitStateType::Keplerian,
                AngleFormat::Degrees,
            ),
        ];

        let mut trajectory = Trajectory::from_states(states, InterpolationMethod::None).unwrap();

        // Convert to degrees
        trajectory = trajectory.as_degrees().unwrap();

        // Verify all states are unchanged
        for state in trajectory.iter() {
            assert_eq!(state[0], 7000.0);
            assert_eq!(state[1], 0.0);
            assert_eq!(state[2], 10.0);
            assert_eq!(state[3], 20.0);
            assert_eq!(state[4], 30.0);
            assert_eq!(state[5], 40.0);
        }

        // Create Radians trajectory
        let states = vec![
            OrbitState::new(
                Epoch::from_jd(2451545.0, TimeSystem::UTC),
                Vector6::new(
                    7000.0,
                    0.0,
                    10.0 * DEG2RAD,
                    20.0 * DEG2RAD,
                    30.0 * DEG2RAD,
                    40.0 * DEG2RAD,
                ),
                OrbitFrame::ECI,
                OrbitStateType::Keplerian,
                AngleFormat::Radians,
            ),
            OrbitState::new(
                Epoch::from_jd(2451545.1, TimeSystem::UTC),
                Vector6::new(
                    7000.0,
                    0.0,
                    10.0 * DEG2RAD,
                    20.0 * DEG2RAD,
                    30.0 * DEG2RAD,
                    40.0 * DEG2RAD,
                ),
                OrbitFrame::ECI,
                OrbitStateType::Keplerian,
                AngleFormat::Radians,
            ),
        ];

        let mut trajectory = Trajectory::from_states(states, InterpolationMethod::None).unwrap();

        // Convert to degrees
        trajectory = trajectory.as_degrees().unwrap();

        // Verify all states are what we expect
        for state in trajectory.iter() {
            assert_eq!(state[0], 7000.0);
            assert_eq!(state[1], 0.0);
            assert_abs_diff_eq!(state[2], 10.0, epsilon = 1e-12);
            assert_abs_diff_eq!(state[3], 20.0, epsilon = 1e-12);
            assert_abs_diff_eq!(state[4], 30.0, epsilon = 1e-12);
            assert_abs_diff_eq!(state[5], 40.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_trajectory_as_radians() {
        let states = vec![
            create_test_state(0.0),
            create_test_state(0.1),
            create_test_state(0.2),
        ];

        let mut trajectory = Trajectory::from_states(states, InterpolationMethod::None).unwrap();

        // Convert to degrees
        trajectory = trajectory.as_radians().unwrap();

        // Verify all states are unchanged
        for state in trajectory.iter() {
            let position = state.position().unwrap();
            assert_eq!(position.x, 7000.0e3);
            assert_eq!(position.y, 0.0);
            assert_eq!(position.z, 0.0);

            let velocity = state.velocity().unwrap();
            assert_eq!(velocity.x, 0.0);
            assert_eq!(velocity.y, 7.5e3);
            assert_eq!(velocity.z, 0.0);
        }

        let states = vec![
            OrbitState::new(
                Epoch::from_jd(2451545.0, TimeSystem::UTC),
                Vector6::new(7000.0, 0.0, 1.0, 2.0, 3.0, 4.0),
                OrbitFrame::ECI,
                OrbitStateType::Keplerian,
                AngleFormat::Radians,
            ),
            OrbitState::new(
                Epoch::from_jd(2451545.1, TimeSystem::UTC),
                Vector6::new(7000.0, 0.0, 1.0, 2.0, 3.0, 4.0),
                OrbitFrame::ECI,
                OrbitStateType::Keplerian,
                AngleFormat::Radians,
            ),
        ];

        let mut trajectory = Trajectory::from_states(states, InterpolationMethod::None).unwrap();

        // Convert to degrees
        trajectory = trajectory.as_radians().unwrap();

        // Verify all states are unchanged
        for state in trajectory.iter() {
            assert_eq!(state[0], 7000.0);
            assert_eq!(state[1], 0.0);
            assert_eq!(state[2], 1.0);
            assert_eq!(state[3], 2.0);
            assert_eq!(state[4], 3.0);
            assert_eq!(state[5], 4.0);
        }

        // Create Radians trajectory
        let states = vec![
            OrbitState::new(
                Epoch::from_jd(2451545.0, TimeSystem::UTC),
                Vector6::new(
                    7000.0,
                    0.0,
                    1.0 * RAD2DEG,
                    2.0 * RAD2DEG,
                    3.0 * RAD2DEG,
                    4.0 * RAD2DEG,
                ),
                OrbitFrame::ECI,
                OrbitStateType::Keplerian,
                AngleFormat::Degrees,
            ),
            OrbitState::new(
                Epoch::from_jd(2451545.1, TimeSystem::UTC),
                Vector6::new(
                    7000.0,
                    0.0,
                    1.0 * RAD2DEG,
                    2.0 * RAD2DEG,
                    3.0 * RAD2DEG,
                    4.0 * RAD2DEG,
                ),
                OrbitFrame::ECI,
                OrbitStateType::Keplerian,
                AngleFormat::Degrees,
            ),
        ];

        let mut trajectory = Trajectory::from_states(states, InterpolationMethod::None).unwrap();

        // Convert to degrees
        trajectory = trajectory.as_radians().unwrap();

        // Verify all states are what we expect
        for state in trajectory.iter() {
            assert_eq!(state[0], 7000.0);
            assert_eq!(state[1], 0.0);
            assert_abs_diff_eq!(state[2], 1.0, epsilon = 1e-12);
            assert_abs_diff_eq!(state[3], 2.0, epsilon = 1e-12);
            assert_abs_diff_eq!(state[4], 3.0, epsilon = 1e-12);
            assert_abs_diff_eq!(state[5], 4.0, epsilon = 1e-12);
        }
    }
}
