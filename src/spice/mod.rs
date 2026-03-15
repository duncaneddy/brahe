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
pub mod kernels;
pub mod positions;

pub use almanac::*;
pub use kernels::*;
pub use positions::*;
