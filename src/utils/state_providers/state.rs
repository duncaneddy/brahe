/*!
 * Frame-agnostic state provider traits.
 *
 * [`SStateProvider`] and [`DStateProvider`] give epoch-parameterized access
 * to a state vector, static-sized (6D) and dynamic-sized respectively,
 * without assuming the state is orbital. Frame-aware extensions live in
 * `orbit_state`.
 */

use nalgebra::{DVector, Vector6};

use crate::time::Epoch;
use crate::utils::errors::BraheError;

/// Trait for types that can provide state vectors at arbitrary epochs.
///
/// This is the base trait for static-sized (6D) state access without any
/// frame-specific operations. Useful for:
/// - Non-orbital trajectories (e.g., attitude, ground tracks)
/// - Generic state access without orbital mechanics assumptions
/// - Building blocks for more specialized traits
///
/// For orbital-specific state providers with frame conversions, see [`SOrbitStateProvider`].
pub trait SStateProvider {
    /// Returns the state at the given epoch as a 6-element vector in the provider's
    /// native coordinate frame and representation.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(Vector6<f64>)` - 6-element vector containing the state in the provider's native output format
    /// * `Err(BraheError)` - If the state cannot be computed (e.g., epoch out of bounds)
    fn state(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError>;

    /// Returns states at multiple epochs in the propagator's native coordinate frame
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<Vector6<f64>>)` - Vector of 6-element vectors containing states
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states(&self, epochs: &[Epoch]) -> Result<Vec<Vector6<f64>>, BraheError> {
        epochs.iter().map(|&epoch| self.state(epoch)).collect()
    }
}

/// Trait for types that can provide dynamic-sized state vectors at arbitrary epochs.
///
/// This is the base trait for dynamic-sized state access without any
/// frame-specific operations. Useful for:
/// - Non-standard state dimensions (e.g., including STM, sensitivity matrices)
/// - Runtime-determined state sizes
/// - Generic state access without orbital mechanics assumptions
///
/// For orbital-specific state providers, see [`DOrbitStateProvider`].
pub trait DStateProvider {
    /// Returns the state at the given epoch as a dynamic vector in the provider's
    /// native coordinate frame and representation.
    ///
    /// # Arguments
    /// * `epoch` - The epoch at which to compute the state
    ///
    /// # Returns
    /// * `Ok(DVector<f64>)` - Dynamic vector containing the state in the provider's native output format
    /// * `Err(BraheError)` - If the state cannot be computed (e.g., epoch out of bounds)
    fn state(&self, epoch: Epoch) -> Result<DVector<f64>, BraheError>;

    /// Returns the dimension of the state vector
    fn state_dim(&self) -> usize;

    /// Returns states at multiple epochs in the propagator's native coordinate frame
    ///
    /// # Arguments
    /// * `epochs` - Slice of epochs at which to compute states
    ///
    /// # Returns
    /// * `Ok(Vec<DVector<f64>>)` - Vector of dynamic vectors containing states
    /// * `Err(BraheError)` - If any state cannot be computed
    fn states(&self, epochs: &[Epoch]) -> Result<Vec<DVector<f64>>, BraheError> {
        epochs.iter().map(|&epoch| self.state(epoch)).collect()
    }
}
