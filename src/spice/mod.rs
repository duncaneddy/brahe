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
pub mod registry;
pub(crate) mod segments;
pub mod spk;

pub use almanac::*;
pub use kernels::*;
pub use pck::BPCK;
pub use positions::*;
// Explicit re-export list (rather than `registry::*`): `initialize_ephemeris`
// and `initialize_ephemeris_with_kernel` are also defined in `almanac`
// (ANISE-backed) and would otherwise collide. `almanac`/`positions` are
// removed once callers migrate to the native registry, at which point this
// can become a plain glob export.
pub use registry::{
    NAIF_EARTH, NAIF_EMB, NAIF_JUPITER_BARYCENTER, NAIF_MARS, NAIF_MARS_BARYCENTER, NAIF_MERCURY,
    NAIF_MERCURY_BARYCENTER, NAIF_MOON, NAIF_NEPTUNE_BARYCENTER, NAIF_PLUTO_BARYCENTER,
    NAIF_SATURN_BARYCENTER, NAIF_SSB, NAIF_SUN, NAIF_URANUS_BARYCENTER, NAIF_VENUS,
    NAIF_VENUS_BARYCENTER, clear_kernels, load_kernel, loaded_kernels, pck_euler_angles,
    pck_rotation_matrix, spk_position, spk_position_in_kernel, spk_state, spk_state_in_kernel,
    spk_velocity, spk_velocity_in_kernel, unload_kernel,
};
pub use spk::SPK;
