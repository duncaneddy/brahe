/*!
 * Reference frame transformations for Mars: Mars-Centered Inertial (MCI),
 * Mars-Centered Mars-Fixed (MCMF), and their relationship to the
 * Earth-Centered Inertial (ECI) frame.
 *
 * MCI is aligned with the ICRF (treated here as equivalent to J2000, as
 * elsewhere in this crate) but centered on Mars. MCMF is the body-fixed
 * frame defined by the IAU/WGCCRE pole and prime-meridian model for Mars
 * (NAIF ID 499), evaluated by [`rotation_icrf_to_body_fixed_iau`].
 *
 * # MCI origin
 *
 * The MCI origin is the Mars body center (NAIF 499), matching the IAU
 * rotation model. The DE kernels only carry the Mars *system* barycenter
 * (NAIF 4); the body-center leg comes from the `mar099s` satellite
 * ephemeris kernel, which the translation functions in this module
 * auto-download and load on first use (mirroring the lunar PCK
 * auto-load in [`super::lunar`]).
 */

use nalgebra::Vector3;

use crate::math::{SMatrix3, SVector6};
use crate::spice::{NAIFId, spk_position, spk_state};
use crate::time::Epoch;
use crate::utils::BraheError;
use crate::utils::batch::{batch_map, batch_map_epochs};

use super::iau_rotation::{
    body_fixed_iau_angles_and_rates, euler313_omega_body, rotation_icrf_to_body_fixed_iau,
};
use super::transform::{
    RotatingFrameContext, apply_state_icrf_to_rotating, apply_state_rotating_to_icrf,
};

/// Idempotently loads the `mar099s` Mars satellite ephemeris kernel
/// (downloading it to `~/.cache/brahe/naif` if needed) into the global
/// SPICE kernel registry. The DE kernels only carry the Mars *system*
/// barycenter (NAIF 4); `mar099s` provides the Mars body-center (NAIF
/// 499) leg the MCI translation functions require.
///
/// The registry is only consulted on the first call (`OnceLock`); unloading
/// a kernel afterwards is not re-detected.
///
/// Called automatically by every MCI translation in this module; not
/// normally called directly. Mirrors
/// [`super::lunar::ensure_lunar_pck_loaded`].
///
/// # Panics
/// Panics with an actionable message if the kernel cannot be loaded (e.g.
/// no network access and no cached copy).
pub(crate) fn ensure_mars_spk_loaded() {
    // OnceLock latch: the registry is checked (and the kernels loaded if
    // absent) on the first call only. Unloading a kernel mid-operation
    // afterwards is not re-detected; subsequent ephemeris queries will
    // error instead. If the load fails the latch stays unset and the next
    // call retries.
    static MARS_SPK_LOADED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    MARS_SPK_LOADED.get_or_init(|| {
        if crate::spice::kernel_is_loaded("mar099s") {
            return;
        }
        // Loads the default DE ephemeris first when the registry is empty (a
        // satellite kernel alone cannot resolve the Earth leg and would
        // suppress the DE auto-initialization), then mar099s.
        crate::spice::registry::ensure_bodies_loadable(&[NAIFId::Mars.id()]).unwrap_or_else(|e| {
            panic!(
                "Failed to auto-load Mars ephemeris kernels (de440s/mar099s): {}. \
                 Download them with brahe::datasets::naif::download_spice_kernel \
                 (SPICEKernel::DE440s / SPICEKernel::Mar099s, None) and call \
                 brahe::spice::load_spice_kernel(<path>).",
                e
            )
        });
        if !crate::spice::kernel_is_loaded("mar099s") {
            panic!(
                "Failed to auto-load Mars satellite ephemeris 'mar099s'. \
                 Download it with brahe::datasets::naif::download_spice_kernel\
                 (SPICEKernel::Mar099s, None) and call \
                 brahe::spice::load_spice_kernel(<path>)."
            );
        }
    });
}

/// Computes the rotation matrix from Mars-Centered Inertial (MCI) to
/// Mars-Centered Mars-Fixed (MCMF), using the IAU/WGCCRE pole and
/// prime-meridian model for Mars (NAIF ID 499).
///
/// # Arguments:
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming MCI -> MCMF
///
/// # Examples:
/// ```
/// use brahe::frames::rotation_mci_to_mcmf;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_mci_to_mcmf(epc);
/// ```
///
/// # References:
/// - [Archinal, B.A., et al., "Report of the IAU Working Group on Cartographic
///   Coordinates and Rotational Elements: 2015", Celestial Mechanics and
///   Dynamical Astronomy 130, 22 (2018)](https://doi.org/10.1007/s10569-017-9805-5)
pub fn rotation_mci_to_mcmf(epc: Epoch) -> SMatrix3 {
    rotation_icrf_to_body_fixed_iau(NAIFId::Mars.id(), epc)
        .expect("IAU Mars rotation model missing from embedded WGCCRE table — this is a bug")
}

/// Computes the rotation matrix from Mars-Centered Mars-Fixed (MCMF) to
/// Mars-Centered Inertial (MCI), using the IAU/WGCCRE pole and
/// prime-meridian model for Mars (NAIF ID 499).
///
/// # Arguments:
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming MCMF -> MCI
///
/// # Examples:
/// ```
/// use brahe::frames::rotation_mcmf_to_mci;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_mcmf_to_mci(epc);
/// ```
///
/// # References:
/// - [Archinal, B.A., et al., "Report of the IAU Working Group on Cartographic
///   Coordinates and Rotational Elements: 2015", Celestial Mechanics and
///   Dynamical Astronomy 130, 22 (2018)](https://doi.org/10.1007/s10569-017-9805-5)
pub fn rotation_mcmf_to_mci(epc: Epoch) -> SMatrix3 {
    rotation_mci_to_mcmf(epc).transpose()
}

/// Transforms a Cartesian Mars-inertial position into the equivalent
/// Cartesian Mars-fixed position.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_mci`: Cartesian Mars-inertial (MCI) position. Units: (*m*)
///
/// # Returns
/// - `x_mcmf`: Cartesian Mars-fixed (MCMF) position. Units: (*m*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_MARS;
/// use brahe::frames::position_mci_to_mcmf;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_mci = Vector3::new(R_MARS + 400e3, 0.0, 0.0);
/// let x_mcmf = position_mci_to_mcmf(epc, x_mci);
/// ```
pub fn position_mci_to_mcmf(epc: Epoch, x_mci: Vector3<f64>) -> Vector3<f64> {
    rotation_mci_to_mcmf(epc) * x_mci
}

/// Transforms a Cartesian Mars-fixed position into the equivalent
/// Cartesian Mars-inertial position.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_mcmf`: Cartesian Mars-fixed (MCMF) position. Units: (*m*)
///
/// # Returns
/// - `x_mci`: Cartesian Mars-inertial (MCI) position. Units: (*m*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_MARS;
/// use brahe::frames::position_mcmf_to_mci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_mcmf = Vector3::new(R_MARS, 0.0, 0.0);
/// let x_mci = position_mcmf_to_mci(epc, x_mcmf);
/// ```
pub fn position_mcmf_to_mci(epc: Epoch, x_mcmf: Vector3<f64>) -> Vector3<f64> {
    rotation_mcmf_to_mci(epc) * x_mcmf
}

/// Rotation matrix and body-frame angular velocity of the Mars-fixed frame
/// (MCMF) at `epc`.
///
/// # Arguments
/// - `epc`: Epoch instant
///
/// # Returns
/// - MCI -> MCMF rotation matrix and MCMF angular velocity (rad/s)
///
/// # Examples
///
/// ```ignore
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let c = mcmf_context(epc);
/// // c.r_mat rotates MCI -> MCMF; c.omega_b is the MCMF angular velocity
/// ```
fn mcmf_context(epc: Epoch) -> RotatingFrameContext {
    let (angles, rates) = body_fixed_iau_angles_and_rates(NAIFId::Mars.id(), epc)
        .expect("IAU Mars rotation model missing from embedded WGCCRE table — this is a bug");
    let r_mat = rotation_mci_to_mcmf(epc);
    let omega_b = euler313_omega_body(angles, rates);
    RotatingFrameContext { r_mat, omega_b }
}

/// Mars position relative to the Earth in ICRF axes.
///
/// # Arguments
/// - `epc`: Epoch instant
///
/// # Returns
/// - Mars position relative to the Earth. Units: (*m*)
///
/// # Examples
///
/// ```ignore
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let offset = mars_earth_offset_position(epc);
/// // x_mci = x_eci - offset
/// ```
fn mars_earth_offset_position(epc: Epoch) -> Vector3<f64> {
    ensure_mars_spk_loaded();
    spk_position(NAIFId::Mars, NAIFId::Earth, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)")
}

/// Mars state relative to the Earth in ICRF axes.
///
/// # Arguments
/// - `epc`: Epoch instant
///
/// # Returns
/// - Mars state relative to the Earth (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples
///
/// ```ignore
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let offset = mars_earth_offset_state(epc);
/// // x_mci = x_eci - offset
/// ```
fn mars_earth_offset_state(epc: Epoch) -> SVector6 {
    ensure_mars_spk_loaded();
    spk_state(NAIFId::Mars, NAIFId::Earth, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)")
}

/// Transforms a Cartesian Mars-inertial state (position and velocity) into
/// the equivalent Cartesian Mars-fixed state.
///
/// The velocity transformation accounts for the transport term induced by
/// Mars' rotation: `v_mcmf = R * v_mci - omega_mcmf x (R * r_mci)`, where
/// `R` is the MCI -> MCMF rotation and `omega_mcmf` is Mars' angular
/// velocity, expressed in the MCMF frame.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_mci`: Cartesian Mars-inertial (MCI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_mcmf`: Cartesian Mars-fixed (MCMF) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_MARS;
/// use brahe::frames::state_mci_to_mcmf;
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_mci = vector6_from_array([R_MARS + 400e3, 0.0, 0.0, 0.0, 3.4e3, 0.0]);
/// let x_mcmf = state_mci_to_mcmf(epc, x_mci);
/// ```
pub fn state_mci_to_mcmf(epc: Epoch, x_mci: SVector6) -> SVector6 {
    apply_state_icrf_to_rotating(&mcmf_context(epc), &x_mci)
}

/// Transforms a Cartesian Mars-fixed state (position and velocity) into
/// the equivalent Cartesian Mars-inertial state.
///
/// Inverse of [`state_mci_to_mcmf`]: `v_mci = R^T * (v_mcmf + omega_mcmf x
/// r_mcmf)`, where `R` is the MCI -> MCMF rotation and `omega_mcmf` is
/// Mars' angular velocity, expressed in the MCMF frame.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_mcmf`: Cartesian Mars-fixed (MCMF) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_mci`: Cartesian Mars-inertial (MCI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_MARS;
/// use brahe::frames::{state_mci_to_mcmf, state_mcmf_to_mci};
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_mci = vector6_from_array([R_MARS + 400e3, 0.0, 0.0, 0.0, 3.4e3, 0.0]);
/// let x_mcmf = state_mci_to_mcmf(epc, x_mci);
///
/// // Convert back to MCI
/// let x_mci2 = state_mcmf_to_mci(epc, x_mcmf);
/// ```
pub fn state_mcmf_to_mci(epc: Epoch, x_mcmf: SVector6) -> SVector6 {
    apply_state_rotating_to_icrf(&mcmf_context(epc), &x_mcmf)
}

/// Transforms a Cartesian Earth-inertial (ECI) position into the
/// equivalent Cartesian Mars-inertial (MCI) position.
///
/// The MCI origin is the Mars body center (NAIF ID 499); see the
/// module-level documentation.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded and auto-loads the `mar099s` satellite ephemeris kernel for
/// the body-center leg; see [`crate::spice::spk_position`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_eci`: Cartesian Earth-inertial (ECI) position. Units: (*m*)
///
/// # Returns
/// - `x_mci`: Cartesian Mars-inertial (MCI) position. Units: (*m*)
///
/// # Examples:
/// ```
/// use brahe::frames::position_eci_to_mci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_eci = Vector3::new(1e7, 2e7, 3e7);
/// let x_mci = position_eci_to_mci(epc, x_eci);
/// ```
pub fn position_eci_to_mci(epc: Epoch, x_eci: Vector3<f64>) -> Vector3<f64> {
    x_eci - mars_earth_offset_position(epc)
}

/// Transforms a Cartesian Mars-inertial (MCI) position into the
/// equivalent Cartesian Earth-inertial (ECI) position.
///
/// The MCI origin is the Mars body center (NAIF ID 499); see the
/// module-level documentation.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded and auto-loads the `mar099s` satellite ephemeris kernel for
/// the body-center leg; see [`crate::spice::spk_position`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_mci`: Cartesian Mars-inertial (MCI) position. Units: (*m*)
///
/// # Returns
/// - `x_eci`: Cartesian Earth-inertial (ECI) position. Units: (*m*)
///
/// # Examples:
/// ```
/// use brahe::frames::position_mci_to_eci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_mci = Vector3::new(1e7, 2e7, 3e7);
/// let x_eci = position_mci_to_eci(epc, x_mci);
/// ```
pub fn position_mci_to_eci(epc: Epoch, x_mci: Vector3<f64>) -> Vector3<f64> {
    x_mci + mars_earth_offset_position(epc)
}

/// Transforms a Cartesian Earth-inertial (ECI) state (position and
/// velocity) into the equivalent Cartesian Mars-inertial (MCI) state.
///
/// The MCI origin is the Mars body center (NAIF ID 499); see the
/// module-level documentation.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded and auto-loads the `mar099s` satellite ephemeris kernel for
/// the body-center leg; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_eci`: Cartesian Earth-inertial (ECI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_mci`: Cartesian Mars-inertial (MCI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples:
/// ```
/// use brahe::frames::state_eci_to_mci;
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_eci = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
/// let x_mci = state_eci_to_mci(epc, x_eci);
/// ```
pub fn state_eci_to_mci(epc: Epoch, x_eci: SVector6) -> SVector6 {
    x_eci - mars_earth_offset_state(epc)
}

/// Transforms a Cartesian Mars-inertial (MCI) state (position and
/// velocity) into the equivalent Cartesian Earth-inertial (ECI) state.
///
/// The MCI origin is the Mars body center (NAIF ID 499); see the
/// module-level documentation.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded and auto-loads the `mar099s` satellite ephemeris kernel for
/// the body-center leg; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_mci`: Cartesian Mars-inertial (MCI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_eci`: Cartesian Earth-inertial (ECI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples:
/// ```
/// use brahe::frames::state_mci_to_eci;
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_mci = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
/// let x_eci = state_mci_to_eci(epc, x_mci);
/// ```
pub fn state_mci_to_eci(epc: Epoch, x_mci: SVector6) -> SVector6 {
    x_mci + mars_earth_offset_state(epc)
}

/// Computes the MCI-to-MCMF rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_mci_to_mcmf`]. Evaluation runs on the global thread pool
/// for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming MCI -> MCMF, one per epoch, in input order
///
/// # Examples:
/// ```
/// use brahe::frames::rotations_mci_to_mcmf;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0];
/// let r = rotations_mci_to_mcmf(&epochs);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_mci_to_mcmf(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(epochs, |epc| rotation_mci_to_mcmf(*epc))
}

/// Computes the MCMF-to-MCI rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_mcmf_to_mci`]. Evaluation runs on the global thread pool
/// for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming MCMF -> MCI, one per epoch, in input order
///
/// # Examples:
/// ```
/// use brahe::frames::rotations_mcmf_to_mci;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0];
/// let r = rotations_mcmf_to_mci(&epochs);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_mcmf_to_mci(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(epochs, |epc| rotation_mcmf_to_mci(*epc))
}

/// Transforms a batch of Cartesian positions from MCI to MCMF.
///
/// Batch form of [`position_mci_to_mcmf`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_mci`: Cartesian MCI positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian MCMF positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::positions_mci_to_mcmf;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(3.6e6, 0.0, 0.0); 3];
/// let out = positions_mci_to_mcmf(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_mci_to_mcmf(
    epochs: &[Epoch],
    x_mci: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(epochs, x_mci, rotation_mci_to_mcmf, |r, x| r * x)
}

/// Transforms a batch of Cartesian positions from MCMF to MCI.
///
/// Batch form of [`position_mcmf_to_mci`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_mcmf`: Cartesian MCMF positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian MCI positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::positions_mcmf_to_mci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(3.6e6, 0.0, 0.0); 3];
/// let out = positions_mcmf_to_mci(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_mcmf_to_mci(
    epochs: &[Epoch],
    x_mcmf: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(epochs, x_mcmf, rotation_mcmf_to_mci, |r, x| r * x)
}

/// Transforms a batch of Cartesian states from MCI to MCMF.
///
/// Batch form of [`state_mci_to_mcmf`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_mci`: Cartesian MCI states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian MCMF states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::states_mci_to_mcmf;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let states = vec![vector6_from_array([3.6e6, 0.0, 0.0, 0.0, 3.4e3, 0.0]); 3];
/// let out = states_mci_to_mcmf(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_mci_to_mcmf(
    epochs: &[Epoch],
    x_mci: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(epochs, x_mci, mcmf_context, apply_state_icrf_to_rotating)
}

/// Transforms a batch of Cartesian states from MCMF to MCI.
///
/// Batch form of [`state_mcmf_to_mci`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_mcmf`: Cartesian MCMF states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian MCI states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::states_mcmf_to_mci;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let states = vec![vector6_from_array([3.6e6, 0.0, 0.0, 0.0, 3.4e3, 0.0]); 3];
/// let out = states_mcmf_to_mci(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_mcmf_to_mci(
    epochs: &[Epoch],
    x_mcmf: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(epochs, x_mcmf, mcmf_context, apply_state_rotating_to_icrf)
}

/// Transforms a batch of Cartesian positions from ECI to MCI.
///
/// Batch form of [`position_eci_to_mci`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_eci`: Cartesian ECI positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian MCI positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::positions_eci_to_mci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(7.0e6, 1.0e6, -2.0e6); 3];
/// let out = positions_eci_to_mci(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_eci_to_mci(
    epochs: &[Epoch],
    x_eci: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(epochs, x_eci, mars_earth_offset_position, |offset, x| {
        x - offset
    })
}

/// Transforms a batch of Cartesian positions from MCI to ECI.
///
/// Batch form of [`position_mci_to_eci`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_mci`: Cartesian MCI positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian ECI positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::positions_mci_to_eci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(7.0e6, 1.0e6, -2.0e6); 3];
/// let out = positions_mci_to_eci(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_mci_to_eci(
    epochs: &[Epoch],
    x_mci: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(epochs, x_mci, mars_earth_offset_position, |offset, x| {
        x + offset
    })
}

/// Transforms a batch of Cartesian states from ECI to MCI.
///
/// Batch form of [`state_eci_to_mci`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_eci`: Cartesian ECI states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian MCI states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::states_eci_to_mci;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let states = vec![vector6_from_array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]); 3];
/// let out = states_eci_to_mci(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_eci_to_mci(
    epochs: &[Epoch],
    x_eci: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(epochs, x_eci, mars_earth_offset_state, |offset, x| {
        x - offset
    })
}

/// Transforms a batch of Cartesian states from MCI to ECI.
///
/// Batch form of [`state_mci_to_eci`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_mci`: Cartesian MCI states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian ECI states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::states_mci_to_eci;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let states = vec![vector6_from_array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]); 3];
/// let out = states_mci_to_eci(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_mci_to_eci(
    epochs: &[Epoch],
    x_mci: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(epochs, x_mci, mars_earth_offset_state, |offset, x| {
        x + offset
    })
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;
    use serial_test::serial;

    use super::*;
    use crate::constants::R_MARS;
    use crate::math::vector6_from_array;
    use crate::spice::{load_spice_kernel, unload_spice_kernel};
    use crate::time::TimeSystem;
    use crate::utils::testing::{
        CacheRedirect, setup_global_test_spice, synthetic_spk_kernel_bytes,
    };

    #[test]
    fn test_state_mci_to_mcmf_roundtrip() {
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x = vector6_from_array([R_MARS + 400e3, 0.0, 0.0, 0.0, 3.4e3, 0.0]);
        let x2 = state_mcmf_to_mci(epc, state_mci_to_mcmf(epc, x));
        for i in 0..6 {
            assert_abs_diff_eq!(x2[i], x[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_state_mci_to_mcmf_transport_term() {
        // Velocity of a body-fixed point: numerically differentiate R(t)*r and
        // compare with the analytic transport term. Catches sign/frame errors.
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r_inertial = Vector3::new(R_MARS + 400e3, 1e6, 2e6);
        let x = vector6_from_array([r_inertial[0], r_inertial[1], r_inertial[2], 0.0, 0.0, 0.0]);
        let dt = 1.0; // s
        let p0 = position_mci_to_mcmf(epc, r_inertial);
        let p1 = position_mci_to_mcmf(epc + dt, r_inertial);
        let v_fd = (p1 - p0) / dt;
        let v_analytic = state_mci_to_mcmf(epc, x).fixed_rows::<3>(3).into_owned();
        // A 1-second forward difference carries an O(dt) curvature
        // (omega x (omega x r)) truncation term on the order of ~1 cm/s
        // here; verified by sweeping dt down to 0.1 s (error shrinks
        // proportionally with dt, confirming this is truncation, not a
        // sign/frame bug in the analytic transport term).
        for i in 0..3 {
            assert_abs_diff_eq!(v_analytic[i], v_fd[i], epsilon = 1e-2);
        }
    }

    #[test]
    fn test_mcmf_surface_point_is_stationary() {
        // A point rotating with Mars has near-zero MCMF velocity
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r_mcmf = Vector3::new(R_MARS, 0.0, 0.0);
        let x_mci = state_mcmf_to_mci(
            epc,
            vector6_from_array([r_mcmf[0], r_mcmf[1], r_mcmf[2], 0.0, 0.0, 0.0]),
        );
        let back = state_mci_to_mcmf(epc, x_mci);
        for i in 3..6 {
            assert_abs_diff_eq!(back[i], 0.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_position_mcmf_to_mci_roundtrip() {
        // Exercises position_mcmf_to_mci and rotation_mcmf_to_mci, which the
        // state-based tests above don't touch directly.
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_mcmf = Vector3::new(R_MARS, 1e6, 2e6);
        let x_mci = position_mcmf_to_mci(epc, x_mcmf);
        let x_mcmf2 = position_mci_to_mcmf(epc, x_mci);
        for i in 0..3 {
            assert_abs_diff_eq!(x_mcmf2[i], x_mcmf[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[serial]
    fn test_state_eci_to_mci_matches_spk() {
        // x_mci = x_eci - state_of_mars_relative_to_earth
        setup_global_test_spice();
        // The 499 reference query below needs mar099s loaded (the transform
        // under test auto-loads it, but the reference is computed first).
        ensure_mars_spk_loaded();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
        let offset = crate::spice::spk_state(NAIFId::Mars, NAIFId::Earth, epc).unwrap();
        let expected = x - offset;
        let got = state_eci_to_mci(epc, x);
        for i in 0..6 {
            assert_abs_diff_eq!(got[i], expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[serial]
    fn test_state_eci_to_mci_roundtrip() {
        // Exercises position_eci_to_mci, position_mci_to_eci, and
        // state_mci_to_eci, which test_state_eci_to_mci_matches_spk doesn't
        // touch directly.
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_eci = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);

        let x_mci = state_eci_to_mci(epc, x_eci);
        let x_eci2 = state_mci_to_eci(epc, x_mci);
        for i in 0..6 {
            assert_abs_diff_eq!(x_eci2[i], x_eci[i], epsilon = 1e-6);
        }

        let p_eci = x_eci.fixed_rows::<3>(0).into_owned();
        let p_mci = position_eci_to_mci(epc, p_eci);
        let p_eci2 = position_mci_to_eci(epc, p_mci);
        for i in 0..3 {
            assert_abs_diff_eq!(p_eci2[i], p_eci[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[serial]
    fn test_eci_mci_transforms_offline() {
        // The MCI translation functions need the Mars body-center (NAIF 499)
        // leg, which the DE kernels don't carry — auto-loaded from `mar099s`.
        // Seed a synthetic mar099s providing the (499, 4) leg into a redirected
        // cache; the real de440s stays resident (never cleared) for the
        // barycenter chain. Only mar099s is unloaded/reloaded here.
        setup_global_test_spice();
        load_spice_kernel("de440s").unwrap();
        let _ = unload_spice_kernel("mar099s");
        {
            let cache = CacheRedirect::new();
            cache.seed_real_de440s();
            cache.seed("mar099s.bsp", &synthetic_spk_kernel_bytes(&[(499, 4, 2.0)]));

            let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
            let x_eci = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);

            // Auto-load path: `ensure_mars_spk_loaded` is a OnceLock latch,
            // so it only loads the kernel if no earlier test in this process
            // fired it. Load explicitly afterwards so the test is
            // deterministic regardless of latch state.
            assert!(!crate::spice::kernel_is_loaded("mar099s"));
            ensure_mars_spk_loaded();
            if !crate::spice::kernel_is_loaded("mar099s") {
                load_spice_kernel("mar099s").unwrap();
            }
            let x_mci = state_eci_to_mci(epc, x_eci);
            assert!(crate::spice::kernel_is_loaded("mar099s"));

            let x_eci2 = state_mci_to_eci(epc, x_mci);
            for i in 0..6 {
                assert_abs_diff_eq!(x_eci2[i], x_eci[i], epsilon = 1e-6);
            }

            let p_eci = x_eci.fixed_rows::<3>(0).into_owned();
            let p_mci = position_eci_to_mci(epc, p_eci);
            let p_eci2 = position_mci_to_eci(epc, p_mci);
            for i in 0..3 {
                assert_abs_diff_eq!(p_eci2[i], p_eci[i], epsilon = 1e-6);
            }

            // Offset consistency: MCI is the Earth->Mars-body-center translation.
            let offset = crate::spice::spk_state(NAIFId::Mars, NAIFId::Earth, epc).unwrap();
            for i in 0..6 {
                assert_abs_diff_eq!(x_mci[i], x_eci[i] - offset[i], epsilon = 1e-6);
            }

            // ensure_mars_spk_loaded is idempotent while loaded.
            ensure_mars_spk_loaded();

            unload_spice_kernel("mar099s").unwrap();
        }
        // The latch is now set but the kernel was just unloaded, so later
        // latch-relying tests would see it missing. Best-effort restore of
        // the real mar099s (real cache; tolerated failure keeps this test
        // offline-safe when nothing later needs the kernel).
        let _ = load_spice_kernel("mar099s");
    }

    #[test]
    #[serial]
    fn test_batch_mars_frames_match_scalar() {
        setup_global_test_spice();
        let epc0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let epochs: Vec<Epoch> = (0..3).map(|i| epc0 + 3600.0 * i as f64).collect();
        let states: Vec<SVector6> = (0..3)
            .map(|i| {
                vector6_from_array([R_MARS + 300e3 + 1e3 * i as f64, 0.0, 0.0, 0.0, 3.4e3, 0.0])
            })
            .collect();
        let positions: Vec<Vector3<f64>> = states
            .iter()
            .map(|s| Vector3::new(s[0], s[1], s[2]))
            .collect();

        for i in 0..3 {
            let e = epochs[i];
            assert_eq!(rotations_mci_to_mcmf(&epochs)[i], rotation_mci_to_mcmf(e));
            assert_eq!(rotations_mcmf_to_mci(&epochs)[i], rotation_mcmf_to_mci(e));
            assert_eq!(
                positions_mci_to_mcmf(&epochs, &positions).unwrap()[i],
                position_mci_to_mcmf(e, positions[i])
            );
            assert_eq!(
                positions_mcmf_to_mci(&epochs, &positions).unwrap()[i],
                position_mcmf_to_mci(e, positions[i])
            );
            assert_eq!(
                positions_mci_to_mcmf(&epochs[..1], &positions).unwrap()[i],
                position_mci_to_mcmf(epochs[0], positions[i])
            );
            assert_eq!(
                states_mci_to_mcmf(&epochs, &states).unwrap()[i],
                state_mci_to_mcmf(e, states[i])
            );
            assert_eq!(
                states_mcmf_to_mci(&epochs, &states).unwrap()[i],
                state_mcmf_to_mci(e, states[i])
            );
            assert_eq!(
                states_mci_to_mcmf(&epochs[..1], &states).unwrap()[i],
                state_mci_to_mcmf(epochs[0], states[i])
            );
            assert_eq!(
                positions_eci_to_mci(&epochs, &positions).unwrap()[i],
                position_eci_to_mci(e, positions[i])
            );
            assert_eq!(
                positions_mci_to_eci(&epochs, &positions).unwrap()[i],
                position_mci_to_eci(e, positions[i])
            );
            assert_eq!(
                states_eci_to_mci(&epochs, &states).unwrap()[i],
                state_eci_to_mci(e, states[i])
            );
            assert_eq!(
                states_mci_to_eci(&epochs, &states).unwrap()[i],
                state_mci_to_eci(e, states[i])
            );
            assert_eq!(
                states_eci_to_mci(&epochs, &states[..1]).unwrap()[i],
                state_eci_to_mci(e, states[0])
            );
        }

        let mcmf = states_mci_to_mcmf(&epochs, &states).unwrap();
        let back = states_mcmf_to_mci(&epochs, &mcmf).unwrap();
        for i in 0..3 {
            for k in 0..3 {
                assert_abs_diff_eq!(back[i][k], states[i][k], epsilon = 1e-6);
            }
        }
        assert!(states_mci_to_mcmf(&epochs[..2], &states).is_err());
        assert!(rotations_mci_to_mcmf(&[]).is_empty());
    }
}
