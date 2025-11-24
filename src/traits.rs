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
 * - `SStatePropagator` - Core trait for static-sized (6D) state propagators
 * - `DStatePropagator` - Core trait for dynamic-sized state propagators
 * - `SOrbitPropagator` - Orbit-specific propagator trait with orbital initialization (extends `SStatePropagator`)
 * - `DOrbitPropagator` - Orbit-specific propagator trait with orbital initialization (extends `DStatePropagator`)
 *
 * ## State Providers
 * - `SStateProvider` - Base trait for static-sized state access (frame-agnostic)
 * - `DStateProvider` - Base trait for dynamic-sized state access (frame-agnostic)
 * - `SOrbitStateProvider` - Trait for static-sized state with orbital frame conversions (extends `SStateProvider`)
 * - `DOrbitStateProvider` - Trait for dynamic-sized state with orbital capabilities (extends `DStateProvider`)
 * - `SIdentifiableStateProvider` - Combined trait for static-sized state providers with identity tracking
 * - `DIdentifiableStateProvider` - Combined trait for dynamic-sized state providers with identity tracking
 *
 * ## Covariance Providers
 * - `SCovarianceProvider` - Base trait for static-sized covariance access (frame-agnostic)
 * - `DCovarianceProvider` - Base trait for dynamic-sized covariance access (frame-agnostic)
 * - `SOrbitCovarianceProvider` - Trait for static-sized covariance with frame conversions (extends `SCovarianceProvider`)
 * - `DOrbitCovarianceProvider` - Trait for dynamic-sized covariance with frame conversions (extends `DCovarianceProvider`)
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
 * - `FixedStepIntegrator` - Trait for fixed-step numerical integrators
 * - `AdaptiveStepIntegrator` - Trait for adaptive-step numerical integrators
 *
 * ## Identification
 * - `Identifiable` - Trait for objects that can be identified by name, ID, and/or UUID
 */

// Orbit propagator traits
pub use crate::propagators::traits::{
    DOrbitPropagator, DStatePropagator, SOrbitPropagator, SStatePropagator,
};

// State and covariance provider traits
pub use crate::utils::state_providers::{
    DCovarianceProvider, DIdentifiableStateProvider, DOrbitCovarianceProvider, DOrbitStateProvider,
    DStateProvider, SCovarianceProvider, SIdentifiableStateProvider, SOrbitCovarianceProvider,
    SOrbitStateProvider, SStateProvider,
};

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
pub use crate::integrators::traits::{DIntegrator, SIntegrator};

// Identification trait
pub use crate::utils::identifiable::Identifiable;
