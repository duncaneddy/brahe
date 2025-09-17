/*!
 * The trajectories module provides a unified interface for representing various types
 * of states (orbital, attitude, etc.) and trajectories over time.
 */

mod trajectory;
mod orbital_trajectory;

pub use trajectory::*;
pub use orbital_trajectory::*;