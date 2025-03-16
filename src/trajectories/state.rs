/*!
 * This module defines the base State trait shared by all state types.
 */

use std::ops::{Index, IndexMut};

use crate::time::Epoch;
use crate::utils::BraheError;
use serde::{Deserialize, Serialize};

/// Trait representing a generic reference frame
pub trait ReferenceFrame: std::fmt::Debug + Clone + PartialEq {
    /// Get the name of the reference frame
    fn name(&self) -> &str;
}

/// Enumeration of angle formats for state representations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AngleFormat {
    /// Angles represented in radians
    Radians,
    /// Angles represented in degrees
    Degrees,
    /// No angle representation or not applicable
    None,
}

/// Base trait for all state types (orbit, attitude, etc.)
pub trait State: Clone + Index<usize, Output = f64> + IndexMut<usize, Output = f64> {
    /// The reference frame type used by this state
    type Frame: ReferenceFrame;

    /// Get the epoch of the state
    fn epoch(&self) -> &Epoch;

    /// Get the reference frame of the state
    fn frame(&self) -> &Self::Frame;

    /// Get the angle format of the state
    fn angle_format(&self) -> AngleFormat;

    /// Convert the state to degrees representation
    fn as_degrees(&self) -> Self;

    /// Convert the state to radians representation
    fn as_radians(&self) -> Self;

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

    /// Create a new state at a different epoch with linearly interpolated elements
    fn interpolate_with(&self, other: &Self, alpha: f64, epoch: &Epoch)
    -> Result<Self, BraheError>;
}
