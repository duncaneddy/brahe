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

pub mod constraints;
pub mod location;

// Re-exports
pub use constraints::{
    AccessConstraint, AscDsc, AscDscConstraint, ConstraintComposite, ElevationConstraint,
    ElevationMaskConstraint, LocalTimeConstraint, LookDirection, LookDirectionConstraint,
    OffNadirConstraint,
};
pub use location::{AccessibleLocation, PointLocation, PolygonLocation};
