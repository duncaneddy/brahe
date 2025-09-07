/*!
 * Keplerian propagator implementation using analytical orbital mechanics
 * for two-body problem motion. This propagator solves Kepler's equation
 * to propagate orbital elements forward in time.
 */

use crate::coordinates::{state_cartesian_to_osculating, state_osculating_to_cartesian};
use crate::orbits::traits::{AnalyticPropagator, OrbitPropagator};
use crate::orbits::keplerian::mean_motion;
use crate::trajectories::TrajectoryEvictionPolicy;
use crate::time::Epoch;
use crate::trajectories::{AngleFormat, InterpolationMethod, OrbitFrame, OrbitState, OrbitStateType, PropagatorType, State, Trajectory};
use crate::constants::DEG2RAD;
use crate::utils::BraheError;
use nalgebra::Vector6;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Keplerian propagator for analytical two-body orbital motion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeplerianPropagator {
    /// Initial orbital state (in original representation and frame)
    initial_state: OrbitState,
    
    /// Current orbital state (in original representation and frame)
    current_state: OrbitState,
    
    /// Internal osculating orbital elements (always in radians, ECI frame)
    internal_osculating_elements: Vector6<f64>,
    
    /// Reference epoch for internal propagation
    reference_epoch: Epoch,
    
    /// Accumulated trajectory
    trajectory: Trajectory<OrbitState>,
    
    /// Step size in seconds for stepping operations
    step_size: f64,

    /// Mean motion in radians per second
    n: f64,
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
    /// * `step_size` - Step size in seconds for propagation
    /// 
    /// # Returns
    /// New KeplerianPropagator instance
    pub fn new(
        epoch: Epoch,
        elements: Vector6<f64>,
        frame: OrbitFrame,
        angle_format: AngleFormat,
        step_size: f64,
    ) -> Result<Self, BraheError> {
        let state = OrbitState::new(
            epoch,
            elements,
            frame,
            OrbitStateType::Keplerian,
            angle_format,
        )?;

        // Convert input state to internal osculating elements in ECI frame with radians
        let internal_elements = Self::convert_to_internal_osculating(&state)?;

        let mut trajectory = Trajectory::new(InterpolationMethod::Linear);
        trajectory.add_state(state.clone())?;

        let n = mean_motion(internal_elements[0], false);
        
        Ok(Self {
            initial_state: state.clone(),
            current_state: state,
            internal_osculating_elements: internal_elements,
            reference_epoch: epoch,
            trajectory: trajectory.with_propagator(PropagatorType::Analytical),
            step_size,
            n,
        })
    }

    /// Create a new KeplerianPropagator from Cartesian state
    /// 
    /// # Arguments
    /// * `epoch` - Initial epoch
    /// * `cartesian_state` - Cartesian position/velocity [x, y, z, vx, vy, vz] (km, km/s)
    /// * `frame` - Reference frame
    /// * `step_size` - Step size in seconds for propagation
    /// 
    /// # Returns
    /// New KeplerianPropagator instance
    pub fn from_cartesian(
        epoch: Epoch,
        cartesian_state: Vector6<f64>,
        frame: OrbitFrame,
        step_size: f64,
    ) -> Result<Self, BraheError> {
        let state = OrbitState::new(
            epoch,
            cartesian_state,
            frame,
            OrbitStateType::Cartesian,
            AngleFormat::None,
        )?;

        // Convert input state to internal osculating elements in ECI frame with radians
        let internal_elements = Self::convert_to_internal_osculating(&state)?;

        let mut trajectory = Trajectory::new(InterpolationMethod::Linear);
        trajectory.add_state(state.clone())?;

        let n = mean_motion(internal_elements[0], false);
        
        Ok(Self {
            initial_state: state.clone(),
            current_state: state,
            internal_osculating_elements: internal_elements,
            reference_epoch: epoch,
            trajectory: trajectory.with_propagator(PropagatorType::Analytical),
            step_size,
            n,
        })
    }
    
    /// Convert any orbit state to internal osculating elements (ECI, radians)
    fn convert_to_internal_osculating(state: &OrbitState) -> Result<Vector6<f64>, BraheError> {
        // First convert to ECI frame
        let eci_state = state.to_frame(&OrbitFrame::ECI)?;
        
        // Then convert to osculating elements if needed
        match eci_state.orbit_type {
            OrbitStateType::Cartesian => {
                Ok(state_cartesian_to_osculating(eci_state.state, false))
            }
            OrbitStateType::Keplerian => {
                // Convert angles to radians if needed
                if eci_state.angle_format == AngleFormat::Radians {
                    Ok(eci_state.state)
                } else {
                    let mut elements = eci_state.state;
                    // Convert angles from degrees to radians (i, RAAN, argp, mean_anomaly)
                    for i in 2..6 {
                        elements[i] = elements[i] * DEG2RAD;
                    }
                    Ok(elements)
                }
            }
        }
    }
    
    /// Convert internal osculating elements back to original state format
    fn convert_from_internal_osculating(&self, epoch: Epoch, internal_elements: Vector6<f64>) -> Result<OrbitState, BraheError> {
        // Create osculating state in ECI, radians
        let osculating_state = OrbitState::new(
            epoch,
            internal_elements,
            OrbitFrame::ECI,
            OrbitStateType::Keplerian,
            AngleFormat::Radians,
        )?;
        
        // Convert to original frame
        let frame_converted = osculating_state.to_frame(&self.initial_state.frame)?;
        
        // Convert to original representation type
        match self.initial_state.orbit_type {
            OrbitStateType::Cartesian => {
                // to_frame() already converted to Cartesian, no need to call to_cartesian()
                Ok(frame_converted)
            }
            OrbitStateType::Keplerian => {
                // Convert to original angle format
                frame_converted.to_keplerian(self.initial_state.angle_format)
            }
        }
    }

    /// Propagate Keplerian elements to a target epoch
    /// 
    /// # Arguments
    /// * `target_epoch` - Epoch to propagate to
    /// 
    /// # Returns
    /// Propagated orbital state in original format
    fn propagate_keplerian(&self, target_epoch: Epoch) -> Result<OrbitState, BraheError> {
        let dt = target_epoch - self.reference_epoch;
    
        // Use internal osculating elements (always in radians, ECI)
        let a = self.internal_osculating_elements[0]; // Semi-major axis (m)
        let e = self.internal_osculating_elements[1]; // Eccentricity
        let i = self.internal_osculating_elements[2]; // Inclination (rad)
        let raan = self.internal_osculating_elements[3]; // Right Ascension of Ascending Node (rad)
        let argp = self.internal_osculating_elements[4]; // Argument of perigee (rad)
        let m0 = self.internal_osculating_elements[5]; // Initial mean anomaly (rad)

        // Propagate mean anomaly and normalize to [0, 2π]
        let m = (m0 + self.n * dt) % (2.0 * PI);

        // Create new internal state with propagated mean anomaly
        let propagated_elements = Vector6::new(a, e, i, raan, argp, m);
        
        // Convert back to original state format and frame
        self.convert_from_internal_osculating(target_epoch, propagated_elements)
    }

}

impl OrbitPropagator for KeplerianPropagator {
    fn propagate_to(&mut self, target_epoch: Epoch) -> Result<&OrbitState, BraheError> {
        let mut current_epoch = *self.current_state.epoch();
        
        // Step until we're close to the target epoch
        while (target_epoch - current_epoch) > self.step_size {
            self.step()?;
            current_epoch = *self.current_state.epoch();
        }
        
        // Take a micro step to reach the exact target epoch
        if current_epoch < target_epoch {
            let remaining_time = target_epoch - current_epoch;
            self.step_by(remaining_time)?;
        }
        
        Ok(&self.current_state)
    }
    
    fn reset(&mut self) -> Result<(), BraheError> {
        self.current_state = self.initial_state.clone();
        self.reference_epoch = self.initial_state.epoch;
        self.internal_osculating_elements = Self::convert_to_internal_osculating(&self.initial_state)?;
        self.n = mean_motion(self.internal_osculating_elements[0], false);
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
        self.current_state = state.clone();
        self.reference_epoch = state.epoch;
        self.internal_osculating_elements = Self::convert_to_internal_osculating(&state)?;
        self.n = mean_motion(self.internal_osculating_elements[0], false);
        self.trajectory = Trajectory::new(InterpolationMethod::Linear)
            .with_propagator(PropagatorType::Analytical);
        self.trajectory.add_state(state)?;
        Ok(())
    }
    
    fn set_initial_conditions(
        &mut self,
        epoch: Epoch,
        state: Vector6<f64>,
        frame: OrbitFrame,
        orbit_type: OrbitStateType,
        angle_format: AngleFormat,
    ) -> Result<(), BraheError> {
        let new_state = OrbitState::new(epoch, state, frame, orbit_type, angle_format)?;
        self.set_initial_state(new_state)
    }
    
    fn step_size(&self) -> f64 {
        self.step_size
    }
    
    fn set_step_size(&mut self, step_size: f64) {
        self.step_size = step_size;
    }
    
    fn step(&mut self) -> Result<&OrbitState, BraheError> {
        let current_epoch = *self.current_state.epoch();
        let target_epoch = current_epoch + self.step_size;
        
        self.current_state = self.propagate_keplerian(target_epoch)?;
        self.trajectory.add_state(self.current_state.clone())?;
        
        Ok(&self.current_state)
    }
    
    fn step_by(&mut self, step_size: f64) -> Result<&OrbitState, BraheError> {
        let current_epoch = *self.current_state.epoch();
        let target_epoch = current_epoch + step_size;
        
        self.current_state = self.propagate_keplerian(target_epoch)?;
        self.trajectory.add_state(self.current_state.clone())?;
        
        Ok(&self.current_state)
    }
    
    fn propagate_steps(&mut self, num_steps: usize) -> Result<Vec<OrbitState>, BraheError> {
        let mut states = Vec::new();
        
        for _ in 0..num_steps {
            let state = self.step()?.clone();
            states.push(state);
        }
        
        Ok(states)
    }
    
    fn step_to(&mut self, target_epoch: Epoch) -> Result<&OrbitState, BraheError> {
        while *self.current_state.epoch() < target_epoch {
            self.step()?;
        }
        
        Ok(&self.current_state)
    }
    
    fn trajectory(&self) -> &Trajectory<OrbitState> {
        &self.trajectory
    }
    
    fn trajectory_mut(&mut self) -> &mut Trajectory<OrbitState> {
        &mut self.trajectory
    }
    
    fn set_max_trajectory_size(&mut self, max_size: Option<usize>) {
        self.trajectory.set_max_size(max_size);
    }
    
    fn set_max_trajectory_age(&mut self, max_age: Option<f64>) {
        self.trajectory.set_max_age(max_age);
    }
    
    fn set_eviction_policy(&mut self, policy: TrajectoryEvictionPolicy) {
        self.trajectory.set_eviction_policy(policy);
    }
}

impl AnalyticPropagator for KeplerianPropagator {
    fn state(&self, epoch: Epoch) -> Vector6<f64> {
        let orbit_state = self.propagate_keplerian(epoch).unwrap();
        orbit_state.state
    }

    fn state_eci(&self, epoch: Epoch) -> Vector6<f64> {
        // Propagate internally and get ECI cartesian
        let dt = epoch - self.reference_epoch;
        let a = self.internal_osculating_elements[0];
        let e = self.internal_osculating_elements[1];
        let i = self.internal_osculating_elements[2];
        let raan = self.internal_osculating_elements[3];
        let argp = self.internal_osculating_elements[4];
        let m0 = self.internal_osculating_elements[5];
        let m = (m0 + self.n * dt) % (2.0 * PI);
        let propagated_elements = Vector6::new(a, e, i, raan, argp, m);
        
        // Convert to cartesian in ECI
        state_osculating_to_cartesian(propagated_elements, false)
    }

    fn state_ecef(&self, epoch: Epoch) -> Vector6<f64> {
        let eci_state = self.state_eci(epoch);
        let eci_orbit_state = OrbitState::new(
            epoch,
            eci_state,
            OrbitFrame::ECI,
            OrbitStateType::Cartesian,
            AngleFormat::None,
        ).unwrap();
        let ecef_state = eci_orbit_state.to_frame(&OrbitFrame::ECEF).unwrap();
        ecef_state.state
    }

    fn state_osculating_elements(&self, epoch: Epoch) -> Vector6<f64> {
        // Propagate internal elements
        let dt = epoch - self.reference_epoch;
        let a = self.internal_osculating_elements[0];
        let e = self.internal_osculating_elements[1];
        let i = self.internal_osculating_elements[2];
        let raan = self.internal_osculating_elements[3];
        let argp = self.internal_osculating_elements[4];
        let m0 = self.internal_osculating_elements[5];
        let m = (m0 + self.n * dt) % (2.0 * PI);
        
        Vector6::new(a, e, i, raan, argp, m)
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
                    ).unwrap()
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
    use crate::orbits::keplerian::orbital_period;
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
            60.0, // 60 second step size
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
            60.0, // 60 second step size
        ).unwrap();

        // Propagate by half an orbital period
        let _mu = 3.986004418e14;
        let a = 7000e3;
        let period = orbital_period(a);
        let half_period_epoch = epoch + period / 2.0;

        println!("Orbital period: {} seconds", period);
        println!("Orbit state: {:?}", propagator.current_state());

        let propagated_state = propagator.propagate_to(half_period_epoch).unwrap();

        println!("Propagated state: {:?}", propagated_state);

        // Confirm the current epoch is what is expected
        assert_eq!(*propagated_state.epoch(), epoch + period / 2.0);
        
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
            60.0, // 60 second step size
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