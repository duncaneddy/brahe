/*!
This module contains the `StatePropagator` trait, which is used to define the interface for
state propagators. State propagators are used to propagate the state of a system forward in time.
 */

use nalgebra::SVector;

use crate::time::Epoch;
use crate::utils::BraheError;

/// The `StatePropagator` trait defines the interface for state propagators. State propagators are
/// used to propagate the state of a system forward in time. The state is represented as a
/// `SVector<f64, S>`, where `S` is the size of the state vector.
///
/// The state propagator should store the initial state and epoch, and should be able to propagate
/// the state forward in time using the `step` method.
pub trait StatePropagator<const S: usize> {
    fn get_state_size(&self) -> usize;
    fn get_num_states(&self) -> usize;
    fn get_initial_state(&self) -> &SVector<f64, S>;
    fn get_initial_epoch(&self) -> Epoch;
    fn get_last_step_size(&self) -> Option<f64>;
    fn get_last_epoch(&self) -> Epoch;
    fn get_last_state(&self) -> &SVector<f64, S>;
    fn get_final_epoch(&self) -> Option<Epoch>;
    fn get_step_size(&self) -> Option<f64>;
    fn get_state_by_index(&self, index: usize) -> Option<&SVector<f64, S>>;
    fn get_state_by_epoch(&self, epoch: Epoch) -> Option<&SVector<f64, S>>;
    fn set_final_epoch(&mut self, epoch: Epoch) -> Result<(), BraheError>;
    fn set_step_size(&mut self, step_size: f64) -> Result<(), BraheError>;
    fn reinitialize(&mut self);
    fn step(&mut self) -> Result<(), BraheError>;
    fn step_by(&mut self, dt: f64) -> Result<(), BraheError>;
    fn step_to_epoch(&mut self, epoch: Epoch) -> Result<(), BraheError>;
    fn step_to_final_epoch(&mut self) -> Result<(), BraheError>;
}