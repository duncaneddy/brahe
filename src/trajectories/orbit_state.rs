/*!
 * Implementation of orbit state types based on the State trait.
 */

use nalgebra::{Vector3, Vector6};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::time::Epoch;
use crate::trajectories::state::{ReferenceFrame, State};
use crate::utils::BraheError;
use crate::{coordinates, frames};

/// Enumeration of orbit reference frames
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrbitFrame {
    /// Earth-Centered Inertial frame (J2000)
    ECI,
    /// Earth-Centered Earth-Fixed frame
    ECEF,
    // Additional frames can be added as needed
}

impl ReferenceFrame for OrbitFrame {
    fn name(&self) -> &str {
        match self {
            OrbitFrame::ECI => "Earth-Centered Inertial (J2000)",
            OrbitFrame::ECEF => "Earth-Centered Earth-Fixed",
        }
    }
}

/// Enumeration of orbit state types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrbitStateType {
    /// Cartesian position and velocity (x, y, z, vx, vy, vz)
    Cartesian,
    /// Keplerian elements (a, e, i, Ω, ω, M)
    Keplerian,
    /// TLE mean elements (a, e, i, Ω, ω, M)
    TLEMean,
}

/// Structure representing an orbital state
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OrbitState {
    /// Time of the state
    pub epoch: Epoch,

    /// The state as a 6-dimensional vector (position and velocity for Cartesian, or elements for other types)
    pub state: Vector6<f64>,

    /// The reference frame in which this state is expressed
    pub frame: OrbitFrame,

    /// The type of state vector (Cartesian, Keplerian, etc.)
    pub orbit_type: OrbitStateType,

    /// Optional additional state information (could be propagator-specific)
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl OrbitState {
    /// Create a new OrbitState
    pub fn new(
        epoch: Epoch,
        state: Vector6<f64>,
        frame: OrbitFrame,
        orbit_type: OrbitStateType,
    ) -> Self {
        Self {
            epoch,
            state,
            frame,
            orbit_type,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the state
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Get the position component of the state if it's in Cartesian form
    pub fn position(&self) -> Result<Vector3<f64>, BraheError> {
        if self.orbit_type == OrbitStateType::Cartesian {
            Ok(Vector3::new(self.state[0], self.state[1], self.state[2]))
        } else {
            Err(BraheError::Error(format!(
                "Cannot extract Cartesian position from state type: {:?}",
                self.orbit_type
            )))
        }
    }

    /// Get the velocity component of the state if it's in Cartesian form
    pub fn velocity(&self) -> Result<Vector3<f64>, BraheError> {
        if self.orbit_type == OrbitStateType::Cartesian {
            Ok(Vector3::new(self.state[3], self.state[4], self.state[5]))
        } else {
            Err(BraheError::Error(format!(
                "Cannot extract Cartesian velocity from state type: {:?}",
                self.orbit_type
            )))
        }
    }

    /// Convert to Cartesian state if not already
    pub fn to_cartesian(&self) -> Result<Self, BraheError> {
        match self.orbit_type {
            OrbitStateType::Cartesian => Ok(self.clone()),
            OrbitStateType::Keplerian => {
                let cart_state = coordinates::state_osculating_to_cartesian(self.state, false);
                Ok(Self::new(
                    self.epoch,
                    cart_state,
                    self.frame,
                    OrbitStateType::Cartesian,
                ))
            }
            OrbitStateType::TLEMean => Err(BraheError::Error(
                "Conversion from TLE mean elements to Cartesian not yet implemented".to_string(),
            )),
        }
    }

    /// Convert to Keplerian elements if not already
    pub fn to_keplerian(&self) -> Result<Self, BraheError> {
        match self.orbit_type {
            OrbitStateType::Keplerian => Ok(self.clone()),
            OrbitStateType::Cartesian => {
                let kep_state = coordinates::state_cartesian_to_osculating(self.state, false);
                Ok(Self::new(
                    self.epoch,
                    kep_state,
                    self.frame,
                    OrbitStateType::Keplerian,
                ))
            }
            OrbitStateType::TLEMean => Err(BraheError::Error(
                "Conversion from TLE mean elements to Keplerian not yet implemented".to_string(),
            )),
        }
    }

    /// Convert the state to JSON format
    pub fn to_json(&self) -> Result<String, BraheError> {
        serde_json::to_string(self)
            .map_err(|e| BraheError::Error(format!("Failed to serialize state: {}", e)))
    }

    /// Load a state from JSON format
    pub fn from_json(json: &str) -> Result<Self, BraheError> {
        serde_json::from_str(json)
            .map_err(|e| BraheError::Error(format!("Failed to deserialize state: {}", e)))
    }
}

impl State for OrbitState {
    type Frame = OrbitFrame;

    fn epoch(&self) -> &Epoch {
        &self.epoch
    }

    fn frame(&self) -> &Self::Frame {
        &self.frame
    }

    fn get_element(&self, index: usize) -> Result<f64, BraheError> {
        if index < self.len() {
            Ok(self.state[index])
        } else {
            Err(BraheError::Error(format!(
                "Index {} out of bounds for OrbitState with length {}",
                index,
                self.len()
            )))
        }
    }

    fn len(&self) -> usize {
        6 // All orbit states have 6 elements
    }

    fn to_frame(&self, frame: &Self::Frame) -> Result<Self, BraheError> {
        if self.frame == *frame {
            return Ok(self.clone());
        }

        // Ensure we're working with Cartesian coordinates for frame transformations
        let cart_state = self.to_cartesian()?;

        match (self.frame, frame) {
            (OrbitFrame::ECI, OrbitFrame::ECEF) => {
                // Convert ECI to ECEF
                let ecef_state = frames::state_eci_to_ecef(self.epoch, cart_state.state);
                Ok(OrbitState::new(
                    cart_state.epoch,
                    ecef_state,
                    OrbitFrame::ECEF,
                    OrbitStateType::Cartesian,
                ))
            }
            (OrbitFrame::ECEF, OrbitFrame::ECI) => {
                // Convert ECEF to ECI
                let eci_state = frames::state_ecef_to_eci(self.epoch, cart_state.state);
                Ok(OrbitState::new(
                    cart_state.epoch,
                    eci_state,
                    OrbitFrame::ECI,
                    OrbitStateType::Cartesian,
                ))
            }
            _ => Err(BraheError::Error(format!(
                "Unsupported frame transformation: {:?} to {:?}",
                self.frame, frame
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use approx::assert_abs_diff_eq;

    fn create_test_state(time_offset: f64) -> OrbitState {
        // Create a test state at J2000 + time_offset
        let epoch = Epoch::from_jd(2451545.0 + time_offset, TimeSystem::UTC);
        let state = Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0);
        OrbitState::new(epoch, state, OrbitFrame::ECI, OrbitStateType::Cartesian)
    }

    #[test]
    fn test_orbit_state_creation() {
        let state = create_test_state(0.0);

        assert_eq!(state.epoch.jd(), 2451545.0);
        assert_eq!(state.frame, OrbitFrame::ECI);
        assert_eq!(state.orbit_type, OrbitStateType::Cartesian);

        // Test position and velocity extraction
        let pos = state.position().unwrap();
        let vel = state.velocity().unwrap();

        assert_eq!(pos.x, 7000e3);
        assert_eq!(pos.y, 0.0);
        assert_eq!(pos.z, 0.0);

        assert_eq!(vel.x, 0.0);
        assert_eq!(vel.y, 7.5e3);
        assert_eq!(vel.z, 0.0);
    }

    #[test]
    fn test_orbit_state_trait_implementation() {
        let state = create_test_state(0.0);

        // Test State trait methods
        assert_eq!(state.epoch().jd(), 2451545.0);
        assert_eq!(state.frame(), &OrbitFrame::ECI);
        assert_eq!(state.len(), 6);
        assert_eq!(state.is_empty(), false);

        // Test element access
        assert_eq!(state.get_element(0).unwrap(), 7000e3);
        assert_eq!(state.get_element(4).unwrap(), 7.5e3);
        assert!(state.get_element(6).is_err());
    }

    #[test]
    fn test_orbit_state_serialization() {
        let state = create_test_state(0.0);

        // Test JSON serialization/deserialization roundtrip
        let json = state.to_json().unwrap();

        // Print the JSON string for debugging
        println!("{}", json);

        let state2 = OrbitState::from_json(&json).unwrap();

        // Compare the original and deserialized states
        assert_abs_diff_eq!(
            (state2.epoch().clone() - state.epoch().clone()),
            0.0,
            epsilon = 1e-9
        );
        assert_eq!(state2.frame, state.frame);
        assert_eq!(state2.orbit_type, state.orbit_type);
        assert_eq!(state2.state, state.state);
    }
}
