/*!
 * Module to provide coordinate system transformations.
 */

use nalgebra::{SMatrix, SVector};

/// 6-dimensional static vector type for Cartesian state vectors
pub type SVector6 = SVector<f64, 6>;

/// 3x3 static matrix type for rotation matrices and transformations
pub type SMatrix3 = SMatrix<f64, 3, 3>;

pub mod cartesian;
pub mod coordinate_types;
pub mod geocentric;
pub mod geodetic;
pub mod topocentric;

pub use cartesian::*;
pub use coordinate_types::*;
pub use geocentric::*;
pub use geodetic::*;
pub use topocentric::*;
