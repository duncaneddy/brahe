/*!
The Constants module provides a set of constants that are used throughout the
Brahe library. These constants are primarily used for defining fixed values
that are used in the implementation of various algorithms and models.

They are specifically grouped into the following categories:
- math: Constants related to mathematical operations
- time: Constants related to time systems and conversions
- physical: Constants related to physical properties of the Earth, celestial bodies, and space
- units: Constants and enumerations for unit handling (angles, etc.)
*/

pub mod math;
pub mod physical;
pub mod time;
pub mod units;

pub use math::*;
pub use physical::*;
pub use time::*;
pub use units::*;
