/*!
 * Implementation of a generic trajectory that can contain any state type.
 */

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
                // Insert after if epochs are equal
                insert_idx = i + 1;
                break;
            }
        }

        // Insert at the correct position
        self.states.insert(insert_idx, state);
        Ok(())
    }

    /// Get the state at a specific epoch using interpolation
    pub fn state_at(&self, epoch: &Epoch) -> Result<S, BraheError> {
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
    fn nearest_state(&self, epoch: &Epoch) -> Result<S, BraheError> {
        let mut nearest_idx = 0;
        let mut min_diff = f64::MAX;

        for (i, state) in self.states.iter().enumerate() {
            let diff = (*epoch - *state.epoch()).abs();
            if diff < min_diff {
                min_diff = diff;
                nearest_idx = i;
            }

            // NOTE: This could be improved by exiting early if the epochs are sorted
            // and the current epoch is greater than the requested epoch
        }

        Ok(self.states[nearest_idx].clone())
    }

    /// Interpolate between states using linear interpolation
    fn interpolate_linear(&self, _epoch: &Epoch) -> Result<S, BraheError> {
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

/// Trait for propagating orbital states into trajectories
pub trait Propagator<S: State> {
    /// Propagate to a single epoch
    fn propagate_to(&self, epoch: &Epoch) -> Result<S, BraheError>;

    /// Propagate over a time span with a specific step
    fn propagate_range(
        &self,
        start: &Epoch,
        end: &Epoch,
        step: f64,
    ) -> Result<Trajectory<S>, BraheError>;
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
            create_test_state(0.1),
            create_test_state(0.2),
        ];

        let trajectory = Trajectory::from_states(states, InterpolationMethod::None).unwrap();

        // Request a time exactly at a state
        let state_at_0 = trajectory
            .state_at(&Epoch::from_jd(2451545.0, TimeSystem::UTC))
            .unwrap();
        assert_eq!(state_at_0.epoch().jd(), 2451545.0);

        // Request a time between states
        let state_at_0_05 = trajectory
            .state_at(&Epoch::from_jd(2451545.05, TimeSystem::UTC))
            .unwrap();
        // Should return the closest state (0.0)
        assert_eq!(state_at_0_05.epoch().jd(), 2451545.0);
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
}
