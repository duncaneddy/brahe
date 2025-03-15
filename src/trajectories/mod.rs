/*!
 * The trajectories module provides a unified interface for representing various types
 * of states (orbital, attitude, etc.) and trajectories over time.
 */

mod orbit_state;
mod state;
mod trajectory;

pub use orbit_state::*;
pub use state::*;
pub use trajectory::*;
