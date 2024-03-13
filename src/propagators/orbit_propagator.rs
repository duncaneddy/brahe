/*!
This module defines the `OrbitPropagator` trait, which is used to define the interface for
orbit propagators. Orbit propagators are used to propagate the state of an orbit forward in time.
All orbit propagators are also state propagators, and so they inherit the `StatePropagator` trait.
 */

use crate::propagators::StatePropagator;

// We should also define an "interpolate" tr
pub trait OrbitPropagator<const S: usize>: StatePropagator<S> {
    // TODO: Add methods to convert to OrbitDataFrame
    // fn to_orbit_data_frame(&self) -> OrbitDataFrame;
}