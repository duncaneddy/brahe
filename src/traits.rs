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
 * - `StateProvider` - Trait for analytic orbital propagators (SGP4, TLE)
 * - `IdentifiableStateProvider` - Combined trait for state providers with identity tracking
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
 *
 * ## Identification
 * - `Identifiable` - Trait for objects that can be identified by name, ID, and/or UUID
 */

// Orbit propagator traits
pub use crate::orbits::traits::{IdentifiableStateProvider, OrbitPropagator, StateProvider};

// Trajectory traits and types
pub use crate::trajectories::traits::{
    // Traits
    Interpolatable,
    // Types and Enums
    InterpolationMethod,
    OrbitFrame,
    OrbitRepresentation,
    OrbitalTrajectory,
    Trajectory,
    TrajectoryEvictionPolicy,
};

// Attitude traits
pub use crate::attitude::traits::{FromAttitude, ToAttitude};

// EOP traits (imported through public re-export)
pub use crate::eop::EarthOrientationProvider;

// Numerical integration traits
pub use crate::integrators::numerical_integrator::NumericalIntegrator;
pub use crate::utils::interpolation::StateInterpolator;

// Identification trait
pub use crate::utils::identifiable::Identifiable;
