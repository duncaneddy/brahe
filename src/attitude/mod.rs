/*!
The `attitude` module provides types and traits for representing and converting between different attitude representations.
*/

#![allow(unused_imports)]

use crate::coordinates::SMatrix3;

pub mod attitude_types;
pub mod euler_angle;
pub mod euler_axis;
pub mod quaternion;
pub mod rotation_matrix;
pub mod traits;

pub use attitude_types::*;
pub use euler_angle::*;
pub use euler_axis::*;
pub use quaternion::*;
pub use rotation_matrix::*;
pub use traits::*;
