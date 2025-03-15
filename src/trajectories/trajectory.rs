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

    /// Current iterator position
    current_index: usize,
}

impl<S: State + Serialize + DeserializeOwned> Trajectory<S> {
    /// Create a new empty trajectory
    pub fn new(interpolation_method: InterpolationMethod) -> Self {
        Self {
            states: Vec::new(),
            propagator_type: None,
            interpolation_method,
            current_index: 0,
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
            current_index: 0,
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

        for (i, state) in self.states.iter().enumerate() {
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
        // This method depends on the specifics of how to interpolate between your state types
        // For now, return a not implemented error
        Err(BraheError::Error(
            "Linear interpolation between arbitrary state types not yet implemented".to_string(),
        ))

        // Implementation would follow this approach:
        // 1. Find the two states that bracket the requested epoch
        // 2. Calculate interpolation factor based on epoch
        // 3. Linearly interpolate each element of the state
        // 4. Create a new state with the interpolated values
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

    /// Reset the iterator to the beginning
    pub fn reset_iterator(&mut self) {
        self.current_index = 0;
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

// Implement Iterator trait for Trajectory
impl<S: State> Iterator for Trajectory<S> {
    type Item = S;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.states.len() {
            let state = self.states[self.current_index].clone();
            self.current_index += 1;
            Some(state)
        } else {
            None
        }
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
            current_index: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use crate::trajectories::orbit_state::{OrbitFrame, OrbitState, OrbitStateType};
    use nalgebra::Vector6;

    fn create_test_state(time_offset: f64) -> OrbitState {
        // Create a test state at J2000 + time_offset
        let epoch = Epoch::from_jd(2451545.0 + time_offset, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        OrbitState::new(epoch, state, OrbitFrame::ECI, OrbitStateType::Cartesian)
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

        let mut trajectory = Trajectory::from_states(states, InterpolationMethod::None).unwrap();

        // Test iterator
        let mut count = 0;
        for state in &mut trajectory {
            assert_eq!(state.epoch().jd(), 2451545.0 + (count as f64) * 0.1);
            count += 1;
        }
        assert_eq!(count, 3);

        // Test reset and re-iteration
        trajectory.reset_iterator();
        count = 0;
        for state in &mut trajectory {
            count += 1;
        }
        assert_eq!(count, 3);
    }
}
