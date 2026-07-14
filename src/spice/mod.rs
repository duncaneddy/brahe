/*!
 * Native SPICE kernel support: DAF/SPK/PCK parsing, a global multi-kernel
 * registry, and solar-system body ephemeris queries.
 *
 * The analytical ephemerides (`sun_position`, `moon_position`) remain in
 * `orbit_dynamics::ephemerides`. Kernel downloading is handled by
 * `datasets::naif`.
 */

pub(crate) mod daf;
pub mod kernels;
pub mod naif_id;
pub mod pck;
pub mod positions;
pub mod registry;
pub(crate) mod segments;
pub mod spk;
#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod validation;

pub use kernels::*;
pub use naif_id::*;
pub use pck::BPCK;
pub use positions::*;
pub use registry::*;
pub use spk::SPK;
