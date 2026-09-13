/*!
 * Mathematical utilities and algorithms for Brahe.
 *
 * This module provides core mathematical functionality including:
 * - Angle conversions and utilities
 * - Linear algebra operations
 * - Jacobian computation for numerical integration
 * - Interpolation utilities
 */

pub mod angles;
// Not `pub`: `crate::frames::covariance` is the addressable `covariance`
// module, and a second public module of that name would make
// `pub use math::*;` (in `lib.rs`) collide with `pub use frames::*;` on the
// module name itself. The glob re-export below still surfaces every public
// item as `crate::math::*`.
mod covariance;
pub mod interpolation;
pub mod jacobian;
pub mod linalg;
pub mod sensitivity;
pub mod spherical_geometry;
pub mod traits;

// Re-export commonly used items
pub use angles::*;
pub use covariance::*;
pub use interpolation::*;
pub use jacobian::*;
pub use linalg::*;
pub use sensitivity::*;
pub use spherical_geometry::*;
pub use traits::*;
