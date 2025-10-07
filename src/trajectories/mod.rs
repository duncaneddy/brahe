/*!
 * The trajectories module provides both static (compile-time) and dynamic (runtime)
 * trajectory implementations for representing various types of states over time.
 *
 * # Trajectory Types
 * - `Trajectory` trait: Core trajectory interface implemented by all trajectory types
 * - `STrajectory<R>`: Static, compile-time sized trajectories for high performance
 * - `DynamicTrajectory`: Dynamic, runtime sized trajectories for flexibility
 * - `OrbitalTrajectory` trait: Orbital-specific functionality for 6D trajectories
 *
 * # Examples
 * ```rust
 * use brahe::trajectories::{DynamicTrajectory, STrajectory6, Trajectory};
 *
 * // Dynamic trajectory - any dimension
 * let mut dyn_traj = DynamicTrajectory::new(7); // 7-dimensional
 *
 * // Static trajectory - compile-time sized
 * let mut static_traj = STrajectory6::new(); // 6-dimensional
 * ```
 */

mod traits;
mod strajectory;
mod trajectory;

// Re-export everything from traits
pub use traits::*;

// Re-export everything from strajectory
pub use strajectory::*;

// Re-export everything from trajectory (this includes DynamicTrajectory struct)
pub use trajectory::*;