/*!
 * The trajectories module provides both static (compile-time) and dynamic (runtime)
 * trajectory implementations for representing various types of states over time.
 *
 * # Trajectory Types
 * - `Trajectory` trait: Core trajectory interface implemented by all trajectory types
 * - `STrajectory<R>`: Static, compile-time sized trajectories for high performance
 * - `DTrajectory`: Dynamic, runtime sized trajectories for flexibility
 * - `SOrbitTrajectory`: Static orbital trajectory with frame/representation conversions
 * - `DOrbitTrajectory`: Dynamic orbital trajectory with frame/representation conversions
 * - `OrbitalTrajectory` trait: Orbital-specific functionality trait
 *
 * # Examples
 * ```rust
 * use brahe::trajectories::{DTrajectory, STrajectory6, SOrbitTrajectory, DOrbitTrajectory};
 * use brahe::traits::{Trajectory, OrbitFrame, OrbitRepresentation};
 *
 * // Dynamic trajectory - any dimension
 * let mut dyn_traj = DTrajectory::new(7); // 7-dimensional
 *
 * // Static trajectory - compile-time sized
 * let mut static_traj = STrajectory6::new(); // 6-dimensional
 *
 * // Static orbital trajectory - 6D with orbital-specific features
 * let mut sorbit_traj = SOrbitTrajectory::new(
 *     OrbitFrame::ECI,
 *     OrbitRepresentation::Cartesian,
 *     None,
 * );
 *
 * // Dynamic orbital trajectory - 6D with orbital-specific features
 * let mut dorbit_traj = DOrbitTrajectory::new(
 *     6,  // dimension
 *     OrbitFrame::ECI,
 *     OrbitRepresentation::Cartesian,
 *     None,
 * );
 * ```
 */

pub mod dorbit_trajectory;
pub mod dtrajectory;
pub mod sorbit_trajectory;
pub mod strajectory;
pub mod traits;

// Note: traits and their types are not re-exported here to avoid ambiguous glob re-exports
// Use `use brahe::traits::*` or `use brahe::trajectories::traits::*` instead

// Re-export everything from strajectory
pub use strajectory::*;

// Re-export everything from dtrajectory
pub use dtrajectory::*;

// Re-export everything from sorbit_trajectory
pub use sorbit_trajectory::*;

// Re-export everything from dorbit_trajectory
pub use dorbit_trajectory::*;
