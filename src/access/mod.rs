/*!
 * Access computation module for satellite ground coverage analysis
 *
 * This module provides functionality for computing when and how satellites
 * can access ground locations or imaging targets. It supports:
 *
 * - Point and polygon locations with GeoJSON interoperability
 * - Extensible constraint system (Rust and Python-defined)
 * - Flexible property computation
 * - Polygon tessellation via along-track strips
 * - Parallel computation with configurable threading
 */

pub mod compute;
pub mod constraints;
pub mod geometry;
pub mod location;
pub mod properties;
pub mod windows;

// Re-exports
pub use compute::*;
pub use constraints::*;
pub use geometry::*;
pub use location::*;
pub use properties::*;
pub use windows::*;
