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
pub mod interpolation;
pub mod jacobian;
pub mod linalg;

// Re-export commonly used items
pub use angles::*;
pub use interpolation::*;
pub use jacobian::*;
pub use linalg::*;
