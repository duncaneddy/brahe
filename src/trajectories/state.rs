/*!
 * This module defines the base State trait shared by all state types.
 */

use crate::time::Epoch;
use crate::utils::BraheError;

/// Trait representing a generic reference frame
pub trait ReferenceFrame: std::fmt::Debug + Clone + PartialEq {
    /// Get the name of the reference frame
    fn name(&self) -> &str;
}

/// Base trait for all state types (orbit, attitude, etc.)
pub trait State: Clone {
    /// The reference frame type used by this state
    type Frame: ReferenceFrame;

    /// Get the epoch of the state
    fn epoch(&self) -> &Epoch;

    /// Get the reference frame of the state
    fn frame(&self) -> &Self::Frame;

    /// Access a specific element by index
    fn get_element(&self, index: usize) -> Result<f64, BraheError>;

    /// Get the number of elements in the state
    fn len(&self) -> usize;

    /// Check if state contains any elements
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert to another reference frame
    fn to_frame(&self, frame: &Self::Frame) -> Result<Self, BraheError>;
}
