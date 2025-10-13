/*!
 * Centralized trait re-exports for the Brahe library.
 *
 * This module provides a single location to import all public traits used throughout
 * the library, making it easier for users to discover and import trait functionality.
 *
 * # Usage
 * ```rust
 * use brahe::traits::*;
 * ```
 *
 * # Available Traits
 *
 * ## Orbital Propagation
 * - `OrbitPropagator` - Core trait for orbit propagators with clean interface
 * - `AnalyticPropagator` - Trait for analytic orbital propagators (SGP4, TLE)
 *
 * ## Trajectories
 * - `Trajectory` - Core trajectory functionality for storing and managing state data
 * - `Interpolatable` - Trajectory interpolation functionality
 * - `OrbitalTrajectory` - Orbital-specific trajectory functionality
 *
 * ## Attitude Representations
 * - `ToAttitude` - Convert from attitude representation to another
 * - `FromAttitude` - Create attitude representation from another
 *
 * ## Earth Orientation Parameters
 * - `EarthOrientationProvider` - Trait for EOP data providers
 *
 * ## Numerical Integration
 * - `NumericalIntegrator` - Generic numerical integrator trait
 * - `StateInterpolator` - State interpolation for numerical integrators
 */

// Orbit propagator traits
pub use crate::orbits::traits::{AnalyticPropagator, OrbitPropagator};

// Trajectory traits and types
pub use crate::trajectories::traits::{
    // Traits
    Interpolatable,
    OrbitalTrajectory,
    Trajectory,
    // Types and Enums
    AngleFormat,
    InterpolationMethod,
    OrbitFrame,
    OrbitRepresentation,
    TrajectoryEvictionPolicy,
};

// Attitude traits
pub use crate::attitude::attitude_representation::{FromAttitude, ToAttitude};

// EOP traits (imported through public re-export)
pub use crate::eop::EarthOrientationProvider;

// Numerical integration traits
pub use crate::integrators::numerical_integrator::NumericalIntegrator;
pub use crate::utils::interpolation::StateInterpolator;
