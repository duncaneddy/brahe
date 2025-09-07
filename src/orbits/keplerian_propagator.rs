/*!
 * Keplerian propagator implementation using analytical orbital mechanics
 * for two-body problem motion. This propagator solves Kepler's equation
 * to propagate orbital elements forward in time.
 */

use crate::coordinates::{state_cartesian_to_osculating, state_osculating_to_cartesian};
use crate::orbits::{propagation::{OrbitPropagator, TrajectoryEvictionPolicy}, traits::AnalyticPropagator};
use crate::time::Epoch;
use crate::trajectories::{AngleFormat, InterpolationMethod, OrbitFrame, OrbitState, OrbitStateType, PropagatorType, State, Trajectory};
use crate::utils::BraheError;
use nalgebra::Vector6;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Keplerian propagator for analytical two-body orbital motion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeplerianPropagator {
    /// Initial orbital state
    initial_state: OrbitState,
    
    /// Current orbital state
    current_state: OrbitState,
    
    /// Accumulated trajectory
    trajectory: Trajectory<OrbitState>,
    
    /// Maximum trajectory size for memory management
    max_trajectory_size: Option<usize>,
    
    /// Eviction policy for trajectory memory management
    eviction_policy: TrajectoryEvictionPolicy,
}

impl KeplerianPropagator {
    /// Create a new KeplerianPropagator from Keplerian orbital elements
    /// 
    /// # Arguments
    /// * `epoch` - Initial epoch
    /// * `elements` - Keplerian orbital elements [a, e, i, raan, argp, anomaly] (km, rad)
    /// * `frame` - Reference frame (typically ECI)
    /// * `anomaly_type` - Type of anomaly (mean or true anomaly)
    /// * `angle_format` - Format for angular elements
    /// 
    /// # Returns
    /// New KeplerianPropagator instance
    pub fn new(
        epoch: Epoch,
        elements: Vector6<f64>,
        frame: OrbitFrame,
        angle_format: AngleFormat,
    ) -> Result<Self, BraheError> {
        let state = OrbitState::new(
            epoch,
            elements,
            frame,
            OrbitStateType::Keplerian,
            angle_format,
        );

        let mut trajectory = Trajectory::new(InterpolationMethod::Linear);
        trajectory.add_state(state.clone())?;
        
        Ok(Self {
            initial_state: state.clone(),
            current_state: state,
            trajectory: trajectory.with_propagator(PropagatorType::Analytical),
            max_trajectory_size: None,
            eviction_policy: TrajectoryEvictionPolicy::None,
        })
    }

    /// Create a new KeplerianPropagator from Cartesian state
    /// 
    /// # Arguments
    /// * `epoch` - Initial epoch
    /// * `cartesian_state` - Cartesian position/velocity [x, y, z, vx, vy, vz] (km, km/s)
    /// * `frame` - Reference frame
    /// 
    /// # Returns
    /// New KeplerianPropagator instance
    pub fn from_cartesian(
        epoch: Epoch,
        cartesian_state: Vector6<f64>,
        frame: OrbitFrame,
    ) -> Result<Self, BraheError> {
        // Convert Cartesian to Keplerian elements
        let elements = state_cartesian_to_osculating(cartesian_state, false);
        
        Self::new(epoch, elements, frame, AngleFormat::Radians)
    }

    /// Propagate Keplerian elements to a target epoch
    /// 
    /// # Arguments
    /// * `target_epoch` - Epoch to propagate to
    /// 
    /// # Returns
    /// Propagated orbital state
    fn propagate_keplerian(&self, target_epoch: Epoch) -> Result<OrbitState, BraheError> {
        let dt = target_epoch - *self.initial_state.epoch();
        
        // Get initial Keplerian elements
        let initial_kep = if self.initial_state.orbit_type == OrbitStateType::Keplerian {
            self.initial_state.state
        } else {
            // Convert to Keplerian if needed
            let cart_state = self.initial_state.to_cartesian()?;
            state_cartesian_to_osculating(cart_state.state, false)
        };

        let a = initial_kep[0]; // Semi-major axis (km)
        let e = initial_kep[1]; // Eccentricity
        let i = initial_kep[2]; // Inclination (rad)
        let raan = initial_kep[3]; // Right Ascension of Ascending Node (rad)
        let argp = initial_kep[4]; // Argument of perigee (rad)
        let M0 = initial_kep[5]; // Initial mean anomaly (rad)

        // Compute mean motion (rad/s)
        let mu = 3.986004418e14; // Earth's gravitational parameter (m^3/s^2)
        let a_m = a * 1000.0; // Convert to meters
        let n = (mu / (a_m * a_m * a_m)).sqrt(); // Mean motion in rad/s

        // Propagate mean anomaly
        let M = M0 + n * dt;
        let M_normalized = ((M % (2.0 * PI)) + 2.0 * PI) % (2.0 * PI);

        // Create new state with propagated mean anomaly
        let new_elements = Vector6::new(a, e, i, raan, argp, M_normalized);
        
        Ok(OrbitState::new(
            target_epoch,
            new_elements,
            self.initial_state.frame,
            OrbitStateType::Keplerian,
            self.initial_state.angle_format,
        ))
    }

    /// Apply eviction policy to manage trajectory memory
    fn apply_eviction_policy(&mut self) -> Result<(), BraheError> {
        if let Some(max_size) = self.max_trajectory_size {
            if self.trajectory.states.len() > max_size {
                match self.eviction_policy {
                    TrajectoryEvictionPolicy::None => {
                        // Do nothing, let it grow
                    },
                    TrajectoryEvictionPolicy::KeepRecent => {
                        // Remove oldest states
                        let to_remove = self.trajectory.states.len() - max_size;
                        self.trajectory.states.drain(0..to_remove);
                    },
                    TrajectoryEvictionPolicy::KeepWithinDuration => {
                        // Keep states within some duration from current epoch
                        let current_epoch = *self.current_state.epoch();
                        let duration_limit = 86400.0; // 1 day in seconds
                        
                        self.trajectory.states.retain(|state| {
                            (current_epoch - *state.epoch()).abs() <= duration_limit
                        });
                    },
                    TrajectoryEvictionPolicy::MemoryBased => {
                        // Simple memory-based eviction - remove half when limit reached
                        let to_remove = self.trajectory.states.len() / 2;
                        self.trajectory.states.drain(0..to_remove);
                    },
                }
            }
        }
        
        Ok(())
    }
}

impl OrbitPropagator for KeplerianPropagator {
    fn propagate_to(&mut self, target_epoch: Epoch) -> Result<&OrbitState, BraheError> {
        self.current_state = self.propagate_keplerian(target_epoch)?;
        self.trajectory.add_state(self.current_state.clone())?;
        self.apply_eviction_policy()?;
        
        Ok(&self.current_state)
    }
    
    fn reset(&mut self) -> Result<(), BraheError> {
        self.current_state = self.initial_state.clone();
        self.trajectory = Trajectory::new(InterpolationMethod::Linear)
            .with_propagator(PropagatorType::Analytical);
        self.trajectory.add_state(self.initial_state.clone())?;
        
        Ok(())
    }
    
    fn current_epoch(&self) -> Epoch {
        *self.current_state.epoch()
    }
    
    fn current_state(&self) -> &OrbitState {
        &self.current_state
    }
    
    fn initial_state(&self) -> &OrbitState {
        &self.initial_state
    }
    
    fn set_initial_state(&mut self, state: OrbitState) -> Result<(), BraheError> {
        self.initial_state = state.clone();
        self.current_state = state;
        self.reset()
    }
    
    fn set_initial_conditions(
        &mut self,
        epoch: Epoch,
        state: Vector6<f64>,
        frame: OrbitFrame,
        orbit_type: OrbitStateType,
        angle_format: AngleFormat,
    ) -> Result<(), BraheError> {
        let new_state = OrbitState::new(epoch, state, frame, orbit_type, angle_format);
        self.set_initial_state(new_state)
    }
    
    fn propagate_batch(&mut self, epochs: &[Epoch]) -> Result<Vec<OrbitState>, BraheError> {
        let mut states = Vec::new();
        
        for &epoch in epochs {
            let state = self.propagate_keplerian(epoch)?;
            states.push(state);
        }
        
        // Add all states to trajectory
        for state in &states {
            self.trajectory.add_state(state.clone())?;
        }
        
        self.apply_eviction_policy()?;
        
        // Update current state to the last propagated state
        if let Some(last_state) = states.last() {
            self.current_state = last_state.clone();
        }
        
        Ok(states)
    }
    
    fn trajectory(&self) -> &Trajectory<OrbitState> {
        &self.trajectory
    }
    
    fn trajectory_mut(&mut self) -> &mut Trajectory<OrbitState> {
        &mut self.trajectory
    }
    
    fn set_max_trajectory_size(&mut self, max_size: Option<usize>) {
        self.max_trajectory_size = max_size;
    }
    
    fn set_eviction_policy(&mut self, policy: TrajectoryEvictionPolicy) {
        self.eviction_policy = policy;
    }
}

impl AnalyticPropagator for KeplerianPropagator {
    fn state(&self, epoch: Epoch) -> Vector6<f64> {
        let orbit_state = self.propagate_keplerian(epoch).unwrap();
        orbit_state.state
    }

    fn state_eci(&self, epoch: Epoch) -> Vector6<f64> {
        let orbit_state = self.propagate_keplerian(epoch).unwrap();
        let eci_state = orbit_state.to_frame(&OrbitFrame::ECI).unwrap();
        eci_state.state
    }

    fn state_ecef(&self, epoch: Epoch) -> Vector6<f64> {
        let orbit_state = self.propagate_keplerian(epoch).unwrap();
        let ecef_state = orbit_state.to_frame(&OrbitFrame::ECEF).unwrap();
        ecef_state.state
    }

    fn state_osculating_elements(&self, epoch: Epoch) -> Vector6<f64> {
        let orbit_state = self.propagate_keplerian(epoch).unwrap();
        if orbit_state.orbit_type == OrbitStateType::Keplerian {
            orbit_state.state
        } else {
            let cart_state = orbit_state.to_cartesian().unwrap();
            state_cartesian_to_osculating(cart_state.state, false)
        }
    }

    fn states(&self, epochs: &[Epoch]) -> Trajectory<OrbitState> {
        let mut states = Vec::new();
        for &epoch in epochs {
            if let Ok(state) = self.propagate_keplerian(epoch) {
                states.push(state);
            }
        }
        
        Trajectory::from_states(states, InterpolationMethod::Linear)
            .unwrap()
            .with_propagator(PropagatorType::Analytical)
    }

    fn states_eci(&self, epochs: &[Epoch]) -> Trajectory<OrbitState> {
        let mut states = Vec::new();
        for &epoch in epochs {
            if let Ok(orbit_state) = self.propagate_keplerian(epoch) {
                if let Ok(eci_state) = orbit_state.to_frame(&OrbitFrame::ECI) {
                    states.push(eci_state);
                }
            }
        }
        
        Trajectory::from_states(states, InterpolationMethod::Linear)
            .unwrap()
            .with_propagator(PropagatorType::Analytical)
    }

    fn states_ecef(&self, epochs: &[Epoch]) -> Trajectory<OrbitState> {
        let mut states = Vec::new();
        for &epoch in epochs {
            if let Ok(orbit_state) = self.propagate_keplerian(epoch) {
                if let Ok(ecef_state) = orbit_state.to_frame(&OrbitFrame::ECEF) {
                    states.push(ecef_state);
                }
            }
        }
        
        Trajectory::from_states(states, InterpolationMethod::Linear)
            .unwrap()
            .with_propagator(PropagatorType::Analytical)
    }

    fn states_osculating_elements(&self, epochs: &[Epoch]) -> Trajectory<OrbitState> {
        let mut states = Vec::new();
        for &epoch in epochs {
            if let Ok(orbit_state) = self.propagate_keplerian(epoch) {
                let kep_state = if orbit_state.orbit_type == OrbitStateType::Keplerian {
                    orbit_state
                } else {
                    let cart_state = orbit_state.to_cartesian().unwrap();
                    let osculating_elements = state_cartesian_to_osculating(cart_state.state, false);
                    
                    OrbitState::new(
                        epoch,
                        osculating_elements,
                        OrbitFrame::ECI,
                        OrbitStateType::Keplerian,
                        AngleFormat::Radians,
                    )
                };
                
                states.push(kep_state);
            }
        }
        
        Trajectory::from_states(states, InterpolationMethod::Linear)
            .unwrap()
            .with_propagator(PropagatorType::Analytical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_keplerian_propagator_creation() {
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let elements = Vector6::new(7000e3, 0.01, 0.1, 0.0, 0.0, 0.0);
        
        let propagator = KeplerianPropagator::new(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
        ).unwrap();

        assert_eq!(*propagator.initial_state().epoch(), epoch);
        assert_abs_diff_eq!(propagator.initial_state().state[0], 7000e3, epsilon = 1.0);
        assert_abs_diff_eq!(propagator.initial_state().state[1], 0.01, epsilon = 1e-10);
    }

    #[test]
    fn test_keplerian_propagator_propagation() {
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let elements = Vector6::new(7000e3, 0.0, 0.0, 0.0, 0.0, 0.0); // Circular equatorial orbit
        
        let mut propagator = KeplerianPropagator::new(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
        ).unwrap();

        // Propagate by half an orbital period
        let mu = 3.986004418e14;
        let a = 7000e3;
        let period = 2.0 * std::f64::consts::PI * ((a * 1000.0_f64).powi(3) / mu).sqrt();
        let half_period_epoch = Epoch::from_jd(2451545.0 + period / (2.0 * 86400.0), TimeSystem::UTC);
        
        let propagated_state = propagator.propagate_to(half_period_epoch).unwrap();
        
        // Mean anomaly should have advanced by π
        let expected_mean_anomaly = std::f64::consts::PI;
        assert_abs_diff_eq!(
            propagated_state.state[5],
            expected_mean_anomaly,
            epsilon = 1e-6
        );
    }

    #[test]
    fn test_keplerian_propagator_analytic_interface() {
        let epoch = Epoch::from_jd(2451545.0, TimeSystem::UTC);
        let elements = Vector6::new(7000e3, 0.01, 0.1, 0.0, 0.0, 0.0);
        
        let propagator = KeplerianPropagator::new(
            epoch,
            elements,
            OrbitFrame::ECI,
            AngleFormat::Radians,
        ).unwrap();

        // Test single state propagation
        let future_epoch = Epoch::from_jd(2451546.0, TimeSystem::UTC);
        let state = propagator.state(future_epoch);
        assert_eq!(state.len(), 6);

        // Test batch states propagation
        let epochs = vec![
            Epoch::from_jd(2451545.0, TimeSystem::UTC),
            Epoch::from_jd(2451545.5, TimeSystem::UTC),
            Epoch::from_jd(2451546.0, TimeSystem::UTC),
        ];
        
        let trajectory = propagator.states(&epochs);
        assert_eq!(trajectory.states.len(), 3);
        
        // Test trajectory to matrix
        let matrix = trajectory.to_matrix().unwrap();
        assert_eq!(matrix.nrows(), 6);
        assert_eq!(matrix.ncols(), 3);
    }
}