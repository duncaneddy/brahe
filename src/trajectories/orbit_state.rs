/*!
 * Implementation of orbit state types based on the State trait.
 */

use std::collections::HashMap;
use std::ops::{Index, IndexMut};

use nalgebra::{Vector3, Vector6};
use serde::{Deserialize, Serialize};

use crate::constants::{DEG2RAD, RAD2DEG};
use crate::time::Epoch;
use crate::trajectories::state::{AngleFormat, ReferenceFrame, State};
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

    /// The format of angular quantities in the state
    pub angle_format: AngleFormat,

    /// Optional additional state information (could be propagator-specific)
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

// Implement Index trait for OrbitState
impl Index<usize> for OrbitState {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        if index >= self.len() {
            panic!("Index out of bounds: {} (len: {})", index, self.len());
        }

        &self.state[index]
    }
}

impl IndexMut<usize> for OrbitState {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index >= self.len() {
            panic!("Index out of bounds: {} (len: {})", index, self.len());
        }

        &mut self.state[index]
    }
}

impl OrbitState {
    /// Create a new OrbitState
    pub fn new(
        epoch: Epoch,
        state: Vector6<f64>,
        frame: OrbitFrame,
        orbit_type: OrbitStateType,
        angle_format: AngleFormat,
    ) -> OrbitState {
        if orbit_type == OrbitStateType::Keplerian && angle_format == AngleFormat::None {
            panic!("Angle format must be specified for Keplerian elements");
        }

        Self {
            epoch,
            state,
            frame,
            orbit_type,
            angle_format,
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
                let as_degrees = if self.angle_format == AngleFormat::Degrees {
                    true
                } else {
                    false
                };
                let cart_state = coordinates::state_osculating_to_cartesian(self.state, as_degrees);
                Ok(Self::new(
                    self.epoch,
                    cart_state,
                    self.frame,
                    OrbitStateType::Cartesian,
                    AngleFormat::None,
                )) // We know this will succeed because we're converting to Cartesian
            }
            OrbitStateType::TLEMean => Err(BraheError::Error(
                "Conversion from TLE mean elements to Cartesian not yet implemented".to_string(),
            )),
        }
    }

    /// Convert to Keplerian elements if not already
    pub fn to_keplerian(&self, angle_format: AngleFormat) -> Result<Self, BraheError> {
        match self.orbit_type {
            OrbitStateType::Keplerian => Ok(self.clone()),
            OrbitStateType::Cartesian => {
                let as_degrees = if self.angle_format == AngleFormat::Degrees {
                    true
                } else {
                    false
                };
                let kep_state = coordinates::state_cartesian_to_osculating(self.state, as_degrees);
                Ok(Self::new(
                    self.epoch,
                    kep_state,
                    self.frame,
                    OrbitStateType::Keplerian,
                    angle_format,
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

    fn angle_format(&self) -> AngleFormat {
        self.angle_format
    }

    /// Convert the state to degrees representation
    fn as_degrees(&self) -> Self {
        if self.angle_format == AngleFormat::Degrees || self.angle_format == AngleFormat::None {
            return self.clone();
        }

        let mut new_state = self.clone();
        new_state.angle_format = AngleFormat::Degrees;

        // Convert i, Ω, ω, M from radians to degrees (elements 2-5)
        for i in 2..6 {
            new_state.state[i] = new_state.state[i] * RAD2DEG;
        }

        new_state
    }

    /// Convert the state to radians representation
    fn as_radians(&self) -> Self {
        if self.angle_format == AngleFormat::Radians || self.angle_format == AngleFormat::None {
            return self.clone();
        }

        let mut new_state = self.clone();
        new_state.angle_format = AngleFormat::Radians;

        // Convert i, Ω, ω, M from degrees to radians (elements 2-5)
        for i in 2..6 {
            new_state.state[i] = new_state.state[i] * DEG2RAD;
        }

        new_state
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
                    self.angle_format,
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
                    self.angle_format,
                ))
            }
            _ => Err(BraheError::Error(format!(
                "Unsupported frame transformation: {:?} to {:?}",
                self.frame, frame
            ))),
        }
    }

    /// Interpolate between two states
    ///
    /// This method performs linear interpolation between two states, `self` and `other`, at a given
    /// interpolation factor `alpha`. The resulting state will be in the same frame and orbit type as
    /// the input states.
    ///
    /// # Arguments
    /// - `other`: The other state to interpolate with
    /// - `alpha`: The interpolation factor (0.0 to 1.0)
    /// - `epoch`: The epoch at which to create the interpolated state
    ///
    /// # Returns
    /// A new state that is the linear interpolation between `self` and `other`
    ///
    /// # Errors
    /// This method will return an error if the two states have different orbit types or frames
    fn interpolate_with(
        &self,
        other: &Self,
        alpha: f64,
        epoch: &Epoch,
    ) -> Result<Self, BraheError> {
        // Check that both states have the same type and frame
        if self.orbit_type != other.orbit_type {
            return Err(BraheError::Error(format!(
                "Cannot interpolate between different orbit types: {:?} and {:?}",
                self.orbit_type, other.orbit_type
            )));
        }

        if self.frame != other.frame {
            return Err(BraheError::Error(format!(
                "Cannot interpolate between different frames: {:?} and {:?}",
                self.frame, other.frame
            )));
        }

        // For certain state types like Keplerian elements, we might need
        // special interpolation logic to handle wraparound of angular values

        // For now, we'll do simple linear interpolation on each element
        let mut interpolated_state = Vector6::zeros();

        for i in 0..6 {
            let val1 = self.state[i];
            let val2 = other.state[i];

            // Special handling for angular elements in Keplerian orbits
            if self.orbit_type == OrbitStateType::Keplerian
                && (i == 2 || i == 3 || i == 4 || i == 5)
            {
                // Handle angular wraparound for i, Ω, ω, and M
                // to ensure small angle difference around 0 is handled correctly
                let mut diff = val2 - val1;

                if diff > std::f64::consts::PI {
                    diff -= 2.0 * std::f64::consts::PI;
                } else if diff < -std::f64::consts::PI {
                    diff += 2.0 * std::f64::consts::PI;
                }

                interpolated_state[i] = val1 + alpha * diff;

                // Ensure the angle stays in the correct range
                if i == 2 || i == 3 || i == 4 || i == 5 {
                    while interpolated_state[i] < 0.0 {
                        interpolated_state[i] += 2.0 * std::f64::consts::PI;
                    }
                    while interpolated_state[i] >= 2.0 * std::f64::consts::PI {
                        interpolated_state[i] -= 2.0 * std::f64::consts::PI;
                    }
                }
            } else {
                // Standard linear interpolation for non-angular elements
                interpolated_state[i] = val1 * (1.0 - alpha) + val2 * alpha;
            }
        }

        // Create a new state with the interpolated elements
        Ok(OrbitState::new(
            epoch.clone(),
            interpolated_state,
            self.frame,
            self.orbit_type,
            self.angle_format,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::DEG2RAD;
    use crate::time::{Epoch, TimeSystem};
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

    #[test]
    fn test_orbit_state_angle_format() {
        // Create a Keplerian state in radians
        let kep_state_rad = OrbitState::new(
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Vector6::new(
                7000e3,
                0.01,
                30.0 * DEG2RAD,
                60.0 * DEG2RAD,
                45.0 * DEG2RAD,
                90.0 * DEG2RAD,
            ),
            OrbitFrame::ECI,
            OrbitStateType::Keplerian,
            AngleFormat::Radians,
        );

        // By default, it should be in radians
        assert_eq!(kep_state_rad.angle_format, AngleFormat::Radians);

        // Convert to degrees
        let kep_state_deg = kep_state_rad.as_degrees();
        assert_eq!(kep_state_deg.angle_format, AngleFormat::Degrees);

        // Check the values were converted properly
        assert_abs_diff_eq!(kep_state_deg.state[0], 7000e3); // a doesn't change
        assert_abs_diff_eq!(kep_state_deg.state[1], 0.01); // e doesn't change
        assert_abs_diff_eq!(kep_state_deg.state[2], 30.0, epsilon = 1e-12); // i converted to degrees
        assert_abs_diff_eq!(kep_state_deg.state[3], 60.0, epsilon = 1e-12); // Ω converted to degrees
        assert_abs_diff_eq!(kep_state_deg.state[4], 45.0, epsilon = 1e-12); // ω converted to degrees
        assert_abs_diff_eq!(kep_state_deg.state[5], 90.0, epsilon = 1e-12); // M converted to degrees

        // Convert back to radians
        let kep_state_rad2 = kep_state_deg.as_radians();
        assert_eq!(kep_state_rad2.angle_format, AngleFormat::Radians);

        // Check the values match the original
        assert_abs_diff_eq!(kep_state_rad2.state[0], kep_state_rad.state[0]);
        assert_abs_diff_eq!(kep_state_rad2.state[1], kep_state_rad.state[1]);
        assert_abs_diff_eq!(
            kep_state_rad2.state[2],
            kep_state_rad.state[2],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            kep_state_rad2.state[3],
            kep_state_rad.state[3],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            kep_state_rad2.state[4],
            kep_state_rad.state[4],
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            kep_state_rad2.state[5],
            kep_state_rad.state[5],
            epsilon = 1e-12
        );

        // Test with Cartesian state
        let cart_state = OrbitState::new(
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            OrbitFrame::ECI,
            OrbitStateType::Cartesian,
            AngleFormat::None,
        );

        // Should be None for Cartesian
        assert_eq!(cart_state.angle_format, AngleFormat::None);

        // Converting to degrees should result in passing through
        let cart_state_deg = cart_state.as_degrees();
        assert_eq!(cart_state_deg.angle_format, AngleFormat::None);
        assert_eq!(cart_state_deg.state, cart_state.state);
    }

    // Add to tests in src/trajectories/orbit_state.rs
    #[test]
    fn test_orbit_state_indexing() {
        // Create a test Cartesian state
        let cart_state = OrbitState::new(
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Vector6::new(7000e3, 1000e3, 2000e3, 100.0, 200.0, 300.0),
            OrbitFrame::ECI,
            OrbitStateType::Cartesian,
            AngleFormat::None,
        );

        // Test direct indexing
        assert_eq!(cart_state[0], 7000e3);
        assert_eq!(cart_state[1], 1000e3);
        assert_eq!(cart_state[2], 2000e3);
        assert_eq!(cart_state[3], 100.0);
        assert_eq!(cart_state[4], 200.0);
        assert_eq!(cart_state[5], 300.0);

        // Create a Keplerian state
        let kep_state = OrbitState::new(
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Vector6::new(7000e3, 0.01, 0.2, 0.3, 0.4, 0.5),
            OrbitFrame::ECI,
            OrbitStateType::Keplerian,
            AngleFormat::Degrees,
        );

        // Test direct indexing
        assert_eq!(kep_state[0], 7000e3); // a
        assert_eq!(kep_state[1], 0.01); // e
        assert_eq!(kep_state[2], 0.2); // i
        assert_eq!(kep_state[3], 0.3); // Ω
        assert_eq!(kep_state[4], 0.4); // ω
        assert_eq!(kep_state[5], 0.5); // M
    }

    #[should_panic]
    #[test]
    fn test_index_out_of_bounds() {
        let state = OrbitState::new(
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Vector6::new(7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0),
            OrbitFrame::ECI,
            OrbitStateType::Cartesian,
            AngleFormat::None,
        );

        let _value = state[6]; // This should panic
    }

    #[test]
    fn test_orbit_state_mutable_indexing() {
        // Create a mutable Cartesian state
        let mut cart_state = OrbitState::new(
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Vector6::new(7000e3, 1000e3, 2000e3, 100.0, 200.0, 300.0),
            OrbitFrame::ECI,
            OrbitStateType::Cartesian,
            AngleFormat::None,
        );

        // Modify elements using mutable indexing
        cart_state[0] = 8000e3;
        cart_state[3] = 150.0;

        // Verify changes
        assert_eq!(cart_state[0], 8000e3);
        assert_eq!(cart_state[3], 150.0);

        // Original values shouldn't have changed
        assert_eq!(cart_state[1], 1000e3);
        assert_eq!(cart_state[2], 2000e3);

        // Verify that changes are reflected in position/velocity methods
        let position = cart_state.position().unwrap();
        assert_eq!(position.x, 8000e3);

        // Create a mutable Keplerian state
        let mut kep_state = OrbitState::new(
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Vector6::new(7000e3, 0.01, 0.2, 0.3, 0.4, 0.5),
            OrbitFrame::ECI,
            OrbitStateType::Keplerian,
            AngleFormat::Degrees,
        );

        // Modify elements
        kep_state[0] = 7500e3; // Change semi-major axis
        kep_state[1] = 0.02; // Change eccentricity

        // Verify changes
        assert_eq!(kep_state[0], 7500e3);
        assert_eq!(kep_state[1], 0.02);

        // Converting to Cartesian should reflect the changes
        let cart_converted = kep_state.to_cartesian().unwrap();
        // The position/velocity will be different from the original values
        assert_ne!(cart_converted[0], 7000e3);
    }
}
