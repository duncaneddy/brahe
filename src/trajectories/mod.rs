/*!
 * The trajectories module provides both static (compile-time) and dynamic (runtime)
 * trajectory implementations for representing various types of states over time.
 *
 * # Trajectory Types
 * - `Trajectory` trait: Core trajectory interface implemented by all trajectory types
 * - `STrajectory<R>`: Static, compile-time sized trajectories for high performance
 * - `DTrajectory`: Dynamic, runtime sized trajectories for flexibility
 * - `OrbitTrajectory`: Specialized orbital trajectory with frame/representation conversions
 * - `OrbitalTrajectory` trait: Orbital-specific functionality trait
 *
 * # Examples
 * ```rust
 * use brahe::trajectories::{DTrajectory, STrajectory6, OrbitTrajectory, OrbitFrame, OrbitRepresentation, AngleFormat, Trajectory};
 *
 * // Dynamic trajectory - any dimension
 * let mut dyn_traj = DTrajectory::new(7); // 7-dimensional
 *
 * // Static trajectory - compile-time sized
 * let mut static_traj = STrajectory6::new(); // 6-dimensional
 *
 * // Orbital trajectory - 6D with orbital-specific features
 * let mut orbit_traj = OrbitTrajectory::new(
 *     OrbitFrame::ECI,
 *     OrbitRepresentation::Cartesian,
 *     AngleFormat::None,
 * ).unwrap();
 * ```
 */

mod traits;
mod strajectory;
mod dtrajectory;
mod orbit_trajectory;

// Re-export everything from traits
pub use traits::*;

// Re-export everything from strajectory
pub use strajectory::*;

// Re-export everything from dtrajectory
pub use dtrajectory::*;

// Re-export everything from orbit_trajectory
pub use orbit_trajectory::*;