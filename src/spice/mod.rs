/*!
 * Shared ANISE/SPICE context manager and DE ephemeris functions.
 *
 * This module owns the global ANISE Almanac instance and provides almanac initialization,
 * thread-safe access, and DE-based position queries.
 *
 * The analytical ephemerides (`sun_position`, `moon_position`) remain in `orbit_dynamics::ephemerides`.
 * Kernel downloading is handled by `datasets::naif`.
 */

pub mod almanac;
pub(crate) mod daf;
pub mod kernels;
pub mod pck;
pub mod positions;
pub(crate) mod segments;
pub mod spk;

pub use almanac::*;
pub use kernels::*;
pub use pck::BPCK;
pub use positions::*;
pub use spk::SPK;
