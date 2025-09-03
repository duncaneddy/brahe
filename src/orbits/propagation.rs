/*!
 * Orbit propagation traits and supporting types for unified propagator interfaces.
 * 
 * This module provides the core `OrbitPropagator` trait that enables consistent
 * interfaces across analytical, semi-analytical, and numerical propagators.
 */

use crate::time::Epoch;
use crate::trajectories::{OrbitState, Trajectory};
use crate::utils::BraheError;
use nalgebra::Vector6;
use serde::{Deserialize, Serialize};

/// Enumeration of trajectory eviction policies for memory management
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrajectoryEvictionPolicy {
    /// No eviction - trajectory grows unbounded
    None,
    /// Keep most recent states, evict oldest when limit reached
    KeepRecent,
    /// Keep states within a time duration from current epoch
    KeepWithinDuration,
    /// Memory-based eviction using approximate memory usage
    MemoryBased,
}

// BasePropagator removed - propagators now implement OrbitPropagator trait directly
// This reduces unnecessary abstraction layers and simplifies the architecture

/// Core trait for orbit propagators providing unified interface
pub trait OrbitPropagator {
    /// Propagate to a specific target epoch
    /// 
    /// # Arguments
    /// * `target_epoch` - The epoch to propagate to
    /// 
    /// # Returns
    /// Reference to the propagated state at target epoch
    fn propagate_to(&mut self, target_epoch: Epoch) -> Result<&OrbitState, BraheError>;
    
    /// Reset propagator to initial conditions
    fn reset(&mut self) -> Result<(), BraheError>;
    
    /// Get current epoch of the propagator
    fn current_epoch(&self) -> Epoch;
    
    /// Get reference to current state
    fn current_state(&self) -> &OrbitState;
    
    /// Get reference to initial state
    fn initial_state(&self) -> &OrbitState;
    
    /// Set initial state and reset propagator
    fn set_initial_state(&mut self, state: OrbitState) -> Result<(), BraheError>;
    
    /// Set initial conditions from components
    /// 
    /// # Arguments
    /// * `epoch` - Initial epoch
    /// * `state` - 6-element state vector
    /// * `frame` - Reference frame
    /// * `orbit_type` - Type of orbital representation
    /// * `angle_format` - Format for angular elements
    fn set_initial_conditions(
        &mut self, 
        epoch: Epoch, 
        state: Vector6<f64>,
        frame: crate::trajectories::OrbitFrame,
        orbit_type: crate::trajectories::OrbitStateType,
        angle_format: crate::trajectories::AngleFormat,
    ) -> Result<(), BraheError>;
    
    /// Propagate to multiple epochs at once
    /// 
    /// # Arguments
    /// * `epochs` - Vector of epochs to propagate to
    /// 
    /// # Returns
    /// Vector of states at each requested epoch
    fn propagate_batch(&mut self, epochs: &[Epoch]) -> Result<Vec<OrbitState>, BraheError>;
    
    /// Get reference to accumulated trajectory
    fn trajectory(&self) -> &Trajectory<OrbitState>;
    
    /// Get mutable reference to accumulated trajectory
    fn trajectory_mut(&mut self) -> &mut Trajectory<OrbitState>;
    
    /// Set maximum trajectory size for memory management
    fn set_max_trajectory_size(&mut self, max_size: Option<usize>);
    
    /// Set eviction policy for trajectory memory management
    fn set_eviction_policy(&mut self, policy: TrajectoryEvictionPolicy);
}

/// Trait for orbit interpolators providing trajectory-based state access
pub trait OrbitInterpolator {
    /// Get the number of states in the interpolator
    fn num_states(&self) -> usize;
    
    /// Get state at a specific epoch using interpolation
    fn state_at_epoch(&self, epoch: &Epoch) -> Result<OrbitState, BraheError>;
    
    /// Get the time span covered by the interpolator
    fn time_span(&self) -> Result<(Epoch, Epoch), BraheError>;
    
    /// Check if an epoch is within the interpolation range
    fn contains_epoch(&self, epoch: &Epoch) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trajectory_eviction_policy_enum() {
        // Test that eviction policy enum can be created and compared
        assert_eq!(TrajectoryEvictionPolicy::None, TrajectoryEvictionPolicy::None);
        assert_eq!(TrajectoryEvictionPolicy::KeepRecent, TrajectoryEvictionPolicy::KeepRecent);
        assert_ne!(TrajectoryEvictionPolicy::None, TrajectoryEvictionPolicy::KeepRecent);
    }
}