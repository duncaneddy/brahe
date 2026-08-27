/*!
 * Reference frame transformations for the Moon: Lunar-Centered Inertial
 * (LCI), Lunar-Fixed Principal Axis (LFPA), Lunar-Fixed Mean
 * Earth/polar-axis (LFME), and their relationship to the Earth-Centered
 * Inertial (ECI) frame.
 *
 * LCI is aligned with the ICRF (treated here as equivalent to J2000, as
 * elsewhere in this crate) but centered on the Moon (NAIF ID 301). LFPA is
 * the DE440 lunar principal-axis frame (NAIF frame class ID 31008,
 * `MOON_PA_DE440`), evaluated from the binary PCK `moon_pa_de440` via
 * [`crate::spice::pck_rotation_matrix`] / [`crate::spice::pck_euler_angles`].
 * LFME is the "mean Earth/polar axis" frame in which the Moon's mean pole
 * and mean prime meridian (facing Earth) are nominally aligned with the
 * frame axes; it differs from LFPA by a small constant rotation, described
 * below.
 *
 * # PA -> ME rotation convention
 *
 * The constant LFPA <-> LFME rotation is transcribed from NAIF's lunar
 * frames kernel `moon_de440_220930.tf`
 * (<https://naif.jpl.nasa.gov/pub/naif/generic_kernels/fk/satellites/a_old_versions/moon_de440_220930.tf>),
 * which defines frame `MOON_ME_DE440_ME421` (class ID 31009) as a "text
 * kernel" (TK) frame relative to `MOON_PA_DE440` (31008):
 *
 * ```text
 * TKFRAME_31009_SPEC     = 'ANGLES'
 * TKFRAME_31009_RELATIVE = 'MOON_PA_DE440'
 * TKFRAME_31009_ANGLES   = ( 67.8526   78.6944   0.2785 )
 * TKFRAME_31009_AXES     = (   3,        2,        1    )
 * TKFRAME_31009_UNITS    = 'ARCSECONDS'
 * ```
 *
 * Per the SPICE Frames Required Reading (`frames.req`, "Defining a TK Frame
 * Using Euler Angles"): for `TKFRAME_<id>_SPEC = 'ANGLES'`, the matrix `M`
 * satisfying `V_relative = M * V_tkframe` (i.e. `M` converts a vector's
 * components in the TK frame into its components in the RELATIVE frame)
 * is `M = [angle_1]_axis_1 [angle_2]_axis_2 [angle_3]_axis_3`, where
 * `[A]_i` is a coordinate rotation by angle `A` about axis `i` (`1=x,
 * 2=y, 3=z`) using the same right-handed convention as this crate's
 * `rx`/`ry`/`rz` helpers (verified against `frames.req`'s worked
 * topocentric-frame example, whose stated `M = TP2BF` reproduces the
 * angle/axis-ordered product literally, left to right).
 *
 * Here the TK frame is `MOON_ME_DE440_ME421` (LFME) and the RELATIVE frame
 * is `MOON_PA_DE440` (LFPA), so
 *
 * `M = Rz(67.8526") * Ry(78.6944") * Rx(0.2785")`
 *
 * converts LFME vector components into LFPA components: `X_lfpa = M *
 * X_lfme`. In this crate's `rotation_<a>_to_<b>` naming (`X_b = R *
 * X_a`), `M` is therefore [`rotation_lfme_to_lfpa`], and
 * [`rotation_lfpa_to_lfme`] is its transpose — the reverse of the
 * superficially similar-looking `Rz * Ry * Rx` product one might guess
 * belongs to `lfpa_to_lfme` directly. Because the total rotation angle is
 * only ~104 arcsec (~5e-4 rad), a magnitude-only check (e.g. the ~875 m
 * surface displacement below) cannot distinguish the two directions to
 * first order; the direction used here follows directly from the kernel
 * comment block's stated `V_relative = M * V_tkframe` relationship, not
 * from the displacement test.
 */

use nalgebra::Vector3;

use crate::constants::AS2RAD;
use crate::math::{SMatrix3, SVector6};
use crate::spice::{NAIFId, spk_position, spk_state};
use crate::time::Epoch;
use crate::utils::BraheError;
use crate::utils::batch::{batch_map, batch_map_epochs};

use super::iau_rotation::{euler313_omega_body, rx, ry, rz};
use super::transform::{
    RotatingFrameContext, apply_state_icrf_to_rotating, apply_state_rotating_to_icrf,
};

/// NAIF frame class ID of the DE440 lunar principal-axis frame
/// (`MOON_PA_DE440`), as defined in NAIF's lunar frames kernel.
const MOON_PA_FRAME_ID: i32 = 31008;

/// Idempotently loads the `moon_pa_de440` binary PCK (downloading it to
/// `~/.cache/brahe/naif` if needed) into the global SPICE kernel registry.
/// The registry is only consulted on the first call (`OnceLock`); unloading
/// the kernel afterwards is not re-detected.
///
/// Called automatically by every LFPA/LFME transformation in this module;
/// not normally called directly.
///
/// # Panics
/// Panics with an actionable message if the kernel cannot be loaded (e.g.
/// no network access and no cached copy).
pub(crate) fn ensure_lunar_pck_loaded() {
    // OnceLock latch: the registry is checked (and the kernel loaded if
    // absent) on the first call only. Unloading the kernel mid-operation
    // afterwards is not re-detected; subsequent orientation queries will
    // error instead. If the load fails the latch stays unset and the next
    // call retries.
    static LUNAR_PCK_LOADED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    LUNAR_PCK_LOADED.get_or_init(|| {
        if crate::spice::kernel_is_loaded("moon_pa_de440") {
            return;
        }
        crate::spice::load_spice_kernel("moon_pa_de440").unwrap_or_else(|e| {
            panic!(
                "Failed to auto-load lunar PCK 'moon_pa_de440': {}. \
                 Download it with brahe::datasets::naif::download_spice_kernel\
                 (SPICEKernel::MoonPaDe440, None) and call \
                 brahe::spice::load_spice_kernel(<path>).",
                e
            )
        });
    });
}

/// Computes the rotation matrix from Lunar-Centered Inertial (LCI) to
/// Lunar-Fixed Principal Axis (LFPA), using the DE440 lunar principal-axis
/// binary PCK (`moon_pa_de440`, NAIF frame class ID 31008).
///
/// Auto-loads the `moon_pa_de440` PCK (downloading it to
/// `~/.cache/brahe/naif` if needed) via [`ensure_lunar_pck_loaded`].
///
/// # Arguments:
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming LCI -> LFPA
///
/// # Examples:
/// ```
/// use brahe::frames::rotation_lci_to_lfpa;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_lci_to_lfpa(epc);
/// ```
pub fn rotation_lci_to_lfpa(epc: Epoch) -> SMatrix3 {
    ensure_lunar_pck_loaded();
    crate::spice::pck_rotation_matrix(MOON_PA_FRAME_ID, epc)
        .unwrap_or_else(|e| panic!("Lunar PCK orientation query failed: {}", e))
        .to_matrix()
}

/// Computes the rotation matrix from Lunar-Fixed Principal Axis (LFPA) to
/// Lunar-Centered Inertial (LCI). Inverse of [`rotation_lci_to_lfpa`].
///
/// # Arguments:
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming LFPA -> LCI
///
/// # Examples:
/// ```
/// use brahe::frames::rotation_lfpa_to_lci;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_lfpa_to_lci(epc);
/// ```
pub fn rotation_lfpa_to_lci(epc: Epoch) -> SMatrix3 {
    rotation_lci_to_lfpa(epc).transpose()
}

/// Computes the constant rotation matrix from Lunar-Fixed Mean
/// Earth/polar-axis (LFME) to Lunar-Fixed Principal Axis (LFPA).
///
/// Transcribed from NAIF's lunar frames kernel `moon_de440_220930.tf`
/// (frame `MOON_ME_DE440_ME421`, TKFRAME relative to `MOON_PA_DE440`); see
/// the module-level documentation for the full TKFRAME reading and sign
/// convention.
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming LFME -> LFPA
///
/// # Examples:
/// ```
/// use brahe::frames::rotation_lfme_to_lfpa;
///
/// let r = rotation_lfme_to_lfpa();
/// ```
pub fn rotation_lfme_to_lfpa() -> SMatrix3 {
    rz(67.8526 * AS2RAD) * ry(78.6944 * AS2RAD) * rx(0.2785 * AS2RAD)
}

/// Computes the constant rotation matrix from Lunar-Fixed Principal Axis
/// (LFPA) to Lunar-Fixed Mean Earth/polar-axis (LFME). Inverse of
/// [`rotation_lfme_to_lfpa`].
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming LFPA -> LFME
///
/// # Examples:
/// ```
/// use brahe::frames::rotation_lfpa_to_lfme;
/// use nalgebra::Vector3;
///
/// let r = rotation_lfpa_to_lfme();
/// // Surface displacement between LFPA and LFME representations of the
/// // same body-fixed point is on the order of the mean lunar radius times
/// // the ~104 arcsec PA/ME misalignment angle (~875 m).
/// let v = Vector3::new(1737.4e3, 0.0, 0.0);
/// let displacement = (r * v - v).norm();
/// assert!(displacement > 850.0 && displacement < 900.0);
/// ```
pub fn rotation_lfpa_to_lfme() -> SMatrix3 {
    rotation_lfme_to_lfpa().transpose()
}

/// Computes the rotation matrix from Lunar-Centered Inertial (LCI) to
/// Lunar-Fixed Mean Earth/polar-axis (LFME).
///
/// Auto-loads the `moon_pa_de440` PCK (downloading it to
/// `~/.cache/brahe/naif` if needed) via [`ensure_lunar_pck_loaded`].
///
/// # Arguments:
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming LCI -> LFME
///
/// # Examples:
/// ```
/// use brahe::frames::rotation_lci_to_lfme;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_lci_to_lfme(epc);
/// ```
pub fn rotation_lci_to_lfme(epc: Epoch) -> SMatrix3 {
    rotation_lfpa_to_lfme() * rotation_lci_to_lfpa(epc)
}

/// Computes the rotation matrix from Lunar-Fixed Mean Earth/polar-axis
/// (LFME) to Lunar-Centered Inertial (LCI). Inverse of
/// [`rotation_lci_to_lfme`].
///
/// # Arguments:
/// - `epc`: Epoch instant for computation of the transformation matrix
///
/// # Returns:
/// - `r`: 3x3 Rotation matrix transforming LFME -> LCI
///
/// # Examples:
/// ```
/// use brahe::frames::rotation_lfme_to_lci;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_lfme_to_lci(epc);
/// ```
pub fn rotation_lfme_to_lci(epc: Epoch) -> SMatrix3 {
    rotation_lci_to_lfme(epc).transpose()
}

/// Transforms a Cartesian Lunar-inertial (LCI) position into the
/// equivalent Cartesian Lunar-Fixed Principal Axis (LFPA) position.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_lci`: Cartesian Lunar-inertial (LCI) position. Units: (*m*)
///
/// # Returns
/// - `x_lfpa`: Cartesian Lunar-Fixed Principal Axis (LFPA) position. Units: (*m*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_MOON;
/// use brahe::frames::position_lci_to_lfpa;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_lci = Vector3::new(R_MOON + 100e3, 0.0, 0.0);
/// let x_lfpa = position_lci_to_lfpa(epc, x_lci);
/// ```
pub fn position_lci_to_lfpa(epc: Epoch, x_lci: Vector3<f64>) -> Vector3<f64> {
    rotation_lci_to_lfpa(epc) * x_lci
}

/// Transforms a Cartesian Lunar-Fixed Principal Axis (LFPA) position into
/// the equivalent Cartesian Lunar-inertial (LCI) position.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_lfpa`: Cartesian Lunar-Fixed Principal Axis (LFPA) position. Units: (*m*)
///
/// # Returns
/// - `x_lci`: Cartesian Lunar-inertial (LCI) position. Units: (*m*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_MOON;
/// use brahe::frames::position_lfpa_to_lci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_lfpa = Vector3::new(R_MOON, 0.0, 0.0);
/// let x_lci = position_lfpa_to_lci(epc, x_lfpa);
/// ```
pub fn position_lfpa_to_lci(epc: Epoch, x_lfpa: Vector3<f64>) -> Vector3<f64> {
    rotation_lfpa_to_lci(epc) * x_lfpa
}

/// Transforms a Cartesian Lunar-inertial (LCI) position into the
/// equivalent Cartesian Lunar-Fixed Mean Earth/polar-axis (LFME) position.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_lci`: Cartesian Lunar-inertial (LCI) position. Units: (*m*)
///
/// # Returns
/// - `x_lfme`: Cartesian Lunar-Fixed Mean Earth/polar-axis (LFME) position. Units: (*m*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_MOON;
/// use brahe::frames::position_lci_to_lfme;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_lci = Vector3::new(R_MOON + 100e3, 0.0, 0.0);
/// let x_lfme = position_lci_to_lfme(epc, x_lci);
/// ```
pub fn position_lci_to_lfme(epc: Epoch, x_lci: Vector3<f64>) -> Vector3<f64> {
    rotation_lci_to_lfme(epc) * x_lci
}

/// Transforms a Cartesian Lunar-Fixed Mean Earth/polar-axis (LFME)
/// position into the equivalent Cartesian Lunar-inertial (LCI) position.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_lfme`: Cartesian Lunar-Fixed Mean Earth/polar-axis (LFME) position. Units: (*m*)
///
/// # Returns
/// - `x_lci`: Cartesian Lunar-inertial (LCI) position. Units: (*m*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_MOON;
/// use brahe::frames::position_lfme_to_lci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_lfme = Vector3::new(R_MOON, 0.0, 0.0);
/// let x_lci = position_lfme_to_lci(epc, x_lfme);
/// ```
pub fn position_lfme_to_lci(epc: Epoch, x_lfme: Vector3<f64>) -> Vector3<f64> {
    rotation_lfme_to_lci(epc) * x_lfme
}

/// Rotation matrix and body-frame angular velocity of the lunar
/// principal-axis frame (LFPA) at `epc`.
///
/// # Arguments
/// - `epc`: Epoch instant
///
/// # Returns
/// - LCI -> LFPA rotation matrix and LFPA angular velocity (rad/s)
///
/// # Examples
///
/// ```ignore
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let c = lfpa_context(epc);
/// // c.r_mat rotates LCI -> LFPA; c.omega_b is the LFPA angular velocity
/// ```
fn lfpa_context(epc: Epoch) -> RotatingFrameContext {
    ensure_lunar_pck_loaded();
    let (angles, rates) = crate::spice::pck_euler_angles(MOON_PA_FRAME_ID, epc)
        .unwrap_or_else(|e| panic!("Lunar PCK orientation query failed: {}", e));
    let r_mat = crate::spice::pck_rotation_matrix(MOON_PA_FRAME_ID, epc)
        .unwrap_or_else(|e| panic!("Lunar PCK orientation query failed: {}", e))
        .to_matrix();
    let omega_b = euler313_omega_body(angles, rates);
    RotatingFrameContext { r_mat, omega_b }
}

/// Rotation matrix and body-frame angular velocity of the lunar mean
/// Earth/polar-axis frame (LFME) at `epc`.
///
/// # Arguments
/// - `epc`: Epoch instant
///
/// # Returns
/// - LCI -> LFME rotation matrix and LFME angular velocity (rad/s)
///
/// # Examples
///
/// ```ignore
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let c = lfme_context(epc);
/// // c.r_mat rotates LCI -> LFME; c.omega_b is the LFME angular velocity
/// ```
fn lfme_context(epc: Epoch) -> RotatingFrameContext {
    ensure_lunar_pck_loaded();
    let (angles, rates) = crate::spice::pck_euler_angles(MOON_PA_FRAME_ID, epc)
        .unwrap_or_else(|e| panic!("Lunar PCK orientation query failed: {}", e));
    let r_pa_to_me = rotation_lfpa_to_lfme();
    let r_mat = r_pa_to_me * rotation_lci_to_lfpa(epc);
    let omega_b = r_pa_to_me * euler313_omega_body(angles, rates);
    RotatingFrameContext { r_mat, omega_b }
}

/// Moon position relative to the Earth in ICRF axes.
///
/// # Arguments
/// - `epc`: Epoch instant
///
/// # Returns
/// - Moon position relative to the Earth. Units: (*m*)
///
/// # Examples
///
/// ```ignore
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let offset = moon_earth_offset_position(epc);
/// // x_lci = x_eci - offset
/// ```
fn moon_earth_offset_position(epc: Epoch) -> Vector3<f64> {
    spk_position(NAIFId::Moon, NAIFId::Earth, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)")
}

/// Moon state relative to the Earth in ICRF axes.
///
/// # Arguments
/// - `epc`: Epoch instant
///
/// # Returns
/// - Moon state relative to the Earth (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples
///
/// ```ignore
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let offset = moon_earth_offset_state(epc);
/// // x_lci = x_eci - offset
/// ```
fn moon_earth_offset_state(epc: Epoch) -> SVector6 {
    spk_state(NAIFId::Moon, NAIFId::Earth, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)")
}

/// Transforms a Cartesian Lunar-inertial (LCI) state (position and
/// velocity) into the equivalent Cartesian Lunar-Fixed Principal Axis
/// (LFPA) state.
///
/// The velocity transformation accounts for the transport term induced by
/// the Moon's rotation: `v_lfpa = R * v_lci - omega_lfpa x (R * r_lci)`,
/// where `R` is the LCI -> LFPA rotation and `omega_lfpa` is the Moon's
/// angular velocity (from the PA rotation model), expressed in the LFPA
/// frame.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_lci`: Cartesian Lunar-inertial (LCI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_lfpa`: Cartesian Lunar-Fixed Principal Axis (LFPA) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_MOON;
/// use brahe::frames::state_lci_to_lfpa;
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_lci = vector6_from_array([R_MOON + 100e3, 0.0, 0.0, 0.0, 1.6e3, 0.0]);
/// let x_lfpa = state_lci_to_lfpa(epc, x_lci);
/// ```
pub fn state_lci_to_lfpa(epc: Epoch, x_lci: SVector6) -> SVector6 {
    apply_state_icrf_to_rotating(&lfpa_context(epc), &x_lci)
}

/// Transforms a Cartesian Lunar-Fixed Principal Axis (LFPA) state
/// (position and velocity) into the equivalent Cartesian Lunar-inertial
/// (LCI) state.
///
/// Inverse of [`state_lci_to_lfpa`]: `v_lci = R^T * (v_lfpa + omega_lfpa x
/// r_lfpa)`, where `R` is the LCI -> LFPA rotation and `omega_lfpa` is the
/// Moon's angular velocity, expressed in the LFPA frame.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_lfpa`: Cartesian Lunar-Fixed Principal Axis (LFPA) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_lci`: Cartesian Lunar-inertial (LCI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_MOON;
/// use brahe::frames::{state_lci_to_lfpa, state_lfpa_to_lci};
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_lci = vector6_from_array([R_MOON + 100e3, 0.0, 0.0, 0.0, 1.6e3, 0.0]);
/// let x_lfpa = state_lci_to_lfpa(epc, x_lci);
///
/// // Convert back to LCI
/// let x_lci2 = state_lfpa_to_lci(epc, x_lfpa);
/// ```
pub fn state_lfpa_to_lci(epc: Epoch, x_lfpa: SVector6) -> SVector6 {
    apply_state_rotating_to_icrf(&lfpa_context(epc), &x_lfpa)
}

/// Transforms a Cartesian Lunar-inertial (LCI) state (position and
/// velocity) into the equivalent Cartesian Lunar-Fixed Mean Earth/polar-axis
/// (LFME) state.
///
/// The LFME frame is rigidly offset from LFPA by a constant rotation (see
/// the module-level documentation), so its angular velocity, expressed in
/// LFME, is `omega_lfme = rotation_lfpa_to_lfme() * omega_lfpa`. The
/// velocity transport term is otherwise identical in form to
/// [`state_lci_to_lfpa`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_lci`: Cartesian Lunar-inertial (LCI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_lfme`: Cartesian Lunar-Fixed Mean Earth/polar-axis (LFME) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_MOON;
/// use brahe::frames::state_lci_to_lfme;
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_lci = vector6_from_array([R_MOON + 100e3, 0.0, 0.0, 0.0, 1.6e3, 0.0]);
/// let x_lfme = state_lci_to_lfme(epc, x_lci);
/// ```
pub fn state_lci_to_lfme(epc: Epoch, x_lci: SVector6) -> SVector6 {
    apply_state_icrf_to_rotating(&lfme_context(epc), &x_lci)
}

/// Transforms a Cartesian Lunar-Fixed Mean Earth/polar-axis (LFME) state
/// (position and velocity) into the equivalent Cartesian Lunar-inertial
/// (LCI) state. Inverse of [`state_lci_to_lfme`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_lfme`: Cartesian Lunar-Fixed Mean Earth/polar-axis (LFME) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_lci`: Cartesian Lunar-inertial (LCI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples:
/// ```
/// use brahe::constants::R_MOON;
/// use brahe::frames::{state_lci_to_lfme, state_lfme_to_lci};
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_lci = vector6_from_array([R_MOON + 100e3, 0.0, 0.0, 0.0, 1.6e3, 0.0]);
/// let x_lfme = state_lci_to_lfme(epc, x_lci);
///
/// // Convert back to LCI
/// let x_lci2 = state_lfme_to_lci(epc, x_lfme);
/// ```
pub fn state_lfme_to_lci(epc: Epoch, x_lfme: SVector6) -> SVector6 {
    apply_state_rotating_to_icrf(&lfme_context(epc), &x_lfme)
}

/// Transforms a Cartesian Earth-inertial (ECI) position into the
/// equivalent Cartesian Lunar-inertial (LCI) position.
///
/// The LCI origin is the Moon's body center (NAIF ID 301), directly
/// available from the bundled `de440s` ephemeris (no barycenter offset,
/// unlike [`crate::frames::mars`]).
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_position`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_eci`: Cartesian Earth-inertial (ECI) position. Units: (*m*)
///
/// # Returns
/// - `x_lci`: Cartesian Lunar-inertial (LCI) position. Units: (*m*)
///
/// # Examples:
/// ```
/// use brahe::frames::position_eci_to_lci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_eci = Vector3::new(1e7, 2e7, 3e7);
/// let x_lci = position_eci_to_lci(epc, x_eci);
/// ```
pub fn position_eci_to_lci(epc: Epoch, x_eci: Vector3<f64>) -> Vector3<f64> {
    x_eci - moon_earth_offset_position(epc)
}

/// Transforms a Cartesian Lunar-inertial (LCI) position into the
/// equivalent Cartesian Earth-inertial (ECI) position.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_position`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_lci`: Cartesian Lunar-inertial (LCI) position. Units: (*m*)
///
/// # Returns
/// - `x_eci`: Cartesian Earth-inertial (ECI) position. Units: (*m*)
///
/// # Examples:
/// ```
/// use brahe::frames::position_lci_to_eci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_lci = Vector3::new(1e7, 2e7, 3e7);
/// let x_eci = position_lci_to_eci(epc, x_lci);
/// ```
pub fn position_lci_to_eci(epc: Epoch, x_lci: Vector3<f64>) -> Vector3<f64> {
    x_lci + moon_earth_offset_position(epc)
}

/// Transforms a Cartesian Earth-inertial (ECI) state (position and
/// velocity) into the equivalent Cartesian Lunar-inertial (LCI) state.
///
/// The LCI origin is the Moon's body center (NAIF ID 301), directly
/// available from the bundled `de440s` ephemeris.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_eci`: Cartesian Earth-inertial (ECI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_lci`: Cartesian Lunar-inertial (LCI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples:
/// ```
/// use brahe::frames::state_eci_to_lci;
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_eci = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
/// let x_lci = state_eci_to_lci(epc, x_eci);
/// ```
pub fn state_eci_to_lci(epc: Epoch, x_eci: SVector6) -> SVector6 {
    x_eci - moon_earth_offset_state(epc)
}

/// Transforms a Cartesian Lunar-inertial (LCI) state (position and
/// velocity) into the equivalent Cartesian Earth-inertial (ECI) state.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_lci`: Cartesian Lunar-inertial (LCI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_eci`: Cartesian Earth-inertial (ECI) state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples:
/// ```
/// use brahe::frames::state_lci_to_eci;
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_lci = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
/// let x_eci = state_lci_to_eci(epc, x_lci);
/// ```
pub fn state_lci_to_eci(epc: Epoch, x_lci: SVector6) -> SVector6 {
    x_lci + moon_earth_offset_state(epc)
}

/// Computes the LCI-to-LFPA rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_lci_to_lfpa`]. Evaluation runs on the global thread pool
/// for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming LCI -> LFPA, one per epoch, in input order
///
/// # Examples:
/// ```
/// use brahe::frames::rotations_lci_to_lfpa;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0];
/// let r = rotations_lci_to_lfpa(&epochs);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_lci_to_lfpa(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(|epc| rotation_lci_to_lfpa(*epc), epochs)
}

/// Computes the LFPA-to-LCI rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_lfpa_to_lci`]. Evaluation runs on the global thread pool
/// for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming LFPA -> LCI, one per epoch, in input order
///
/// # Examples:
/// ```
/// use brahe::frames::rotations_lfpa_to_lci;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0];
/// let r = rotations_lfpa_to_lci(&epochs);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_lfpa_to_lci(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(|epc| rotation_lfpa_to_lci(*epc), epochs)
}

/// Computes the LCI-to-LFME rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_lci_to_lfme`]. Evaluation runs on the global thread pool
/// for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming LCI -> LFME, one per epoch, in input order
///
/// # Examples:
/// ```
/// use brahe::frames::rotations_lci_to_lfme;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0];
/// let r = rotations_lci_to_lfme(&epochs);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_lci_to_lfme(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(|epc| rotation_lci_to_lfme(*epc), epochs)
}

/// Computes the LFME-to-LCI rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_lfme_to_lci`]. Evaluation runs on the global thread pool
/// for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming LFME -> LCI, one per epoch, in input order
///
/// # Examples:
/// ```
/// use brahe::frames::rotations_lfme_to_lci;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0];
/// let r = rotations_lfme_to_lci(&epochs);
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_lfme_to_lci(epochs: &[Epoch]) -> Vec<SMatrix3> {
    batch_map(|epc| rotation_lfme_to_lci(*epc), epochs)
}

/// Transforms a batch of Cartesian positions from LCI to LFPA.
///
/// Batch form of [`position_lci_to_lfpa`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_lci`: Cartesian LCI positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian LFPA positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::positions_lci_to_lfpa;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(1.9e6, 0.0, 0.0); 3];
/// let out = positions_lci_to_lfpa(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_lci_to_lfpa(
    epochs: &[Epoch],
    x_lci: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(rotation_lci_to_lfpa, |r, x| r * x, epochs, x_lci)
}

/// Transforms a batch of Cartesian positions from LFPA to LCI.
///
/// Batch form of [`position_lfpa_to_lci`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_lfpa`: Cartesian LFPA positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian LCI positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::positions_lfpa_to_lci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(1.9e6, 0.0, 0.0); 3];
/// let out = positions_lfpa_to_lci(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_lfpa_to_lci(
    epochs: &[Epoch],
    x_lfpa: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(rotation_lfpa_to_lci, |r, x| r * x, epochs, x_lfpa)
}

/// Transforms a batch of Cartesian positions from LCI to LFME.
///
/// Batch form of [`position_lci_to_lfme`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_lci`: Cartesian LCI positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian LFME positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::positions_lci_to_lfme;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(1.9e6, 0.0, 0.0); 3];
/// let out = positions_lci_to_lfme(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_lci_to_lfme(
    epochs: &[Epoch],
    x_lci: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(rotation_lci_to_lfme, |r, x| r * x, epochs, x_lci)
}

/// Transforms a batch of Cartesian positions from LFME to LCI.
///
/// Batch form of [`position_lfme_to_lci`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_lfme`: Cartesian LFME positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian LCI positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::positions_lfme_to_lci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(1.9e6, 0.0, 0.0); 3];
/// let out = positions_lfme_to_lci(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_lfme_to_lci(
    epochs: &[Epoch],
    x_lfme: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(rotation_lfme_to_lci, |r, x| r * x, epochs, x_lfme)
}

/// Transforms a batch of Cartesian states from LCI to LFPA.
///
/// Batch form of [`state_lci_to_lfpa`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_lci`: Cartesian LCI states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian LFPA states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::states_lci_to_lfpa;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let states = vec![vector6_from_array([1.9e6, 0.0, 0.0, 0.0, 1.6e3, 0.0]); 3];
/// let out = states_lci_to_lfpa(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_lci_to_lfpa(
    epochs: &[Epoch],
    x_lci: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(lfpa_context, apply_state_icrf_to_rotating, epochs, x_lci)
}

/// Transforms a batch of Cartesian states from LFPA to LCI.
///
/// Batch form of [`state_lfpa_to_lci`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_lfpa`: Cartesian LFPA states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian LCI states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::states_lfpa_to_lci;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let states = vec![vector6_from_array([1.9e6, 0.0, 0.0, 0.0, 1.6e3, 0.0]); 3];
/// let out = states_lfpa_to_lci(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_lfpa_to_lci(
    epochs: &[Epoch],
    x_lfpa: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(lfpa_context, apply_state_rotating_to_icrf, epochs, x_lfpa)
}

/// Transforms a batch of Cartesian states from LCI to LFME.
///
/// Batch form of [`state_lci_to_lfme`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_lci`: Cartesian LCI states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian LFME states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::states_lci_to_lfme;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let states = vec![vector6_from_array([1.9e6, 0.0, 0.0, 0.0, 1.6e3, 0.0]); 3];
/// let out = states_lci_to_lfme(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_lci_to_lfme(
    epochs: &[Epoch],
    x_lci: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(lfme_context, apply_state_icrf_to_rotating, epochs, x_lci)
}

/// Transforms a batch of Cartesian states from LFME to LCI.
///
/// Batch form of [`state_lfme_to_lci`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_lfme`: Cartesian LFME states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian LCI states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::states_lfme_to_lci;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let states = vec![vector6_from_array([1.9e6, 0.0, 0.0, 0.0, 1.6e3, 0.0]); 3];
/// let out = states_lfme_to_lci(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_lfme_to_lci(
    epochs: &[Epoch],
    x_lfme: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(lfme_context, apply_state_rotating_to_icrf, epochs, x_lfme)
}

/// Transforms a batch of Cartesian positions from ECI to LCI.
///
/// Batch form of [`position_eci_to_lci`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_eci`: Cartesian ECI positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian LCI positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::positions_eci_to_lci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(7.0e6, 1.0e6, -2.0e6); 3];
/// let out = positions_eci_to_lci(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_eci_to_lci(
    epochs: &[Epoch],
    x_eci: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(
        moon_earth_offset_position,
        |offset, x| x - offset,
        epochs,
        x_eci,
    )
}

/// Transforms a batch of Cartesian positions from LCI to ECI.
///
/// Batch form of [`position_lci_to_eci`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_lci`: Cartesian LCI positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian ECI positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::positions_lci_to_eci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(7.0e6, 1.0e6, -2.0e6); 3];
/// let out = positions_lci_to_eci(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_lci_to_eci(
    epochs: &[Epoch],
    x_lci: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    batch_map_epochs(
        moon_earth_offset_position,
        |offset, x| x + offset,
        epochs,
        x_lci,
    )
}

/// Transforms a batch of Cartesian states from ECI to LCI.
///
/// Batch form of [`state_eci_to_lci`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_eci`: Cartesian ECI states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian LCI states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::states_eci_to_lci;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let states = vec![vector6_from_array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]); 3];
/// let out = states_eci_to_lci(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_eci_to_lci(
    epochs: &[Epoch],
    x_eci: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(
        moon_earth_offset_state,
        |offset, x| x - offset,
        epochs,
        x_eci,
    )
}

/// Transforms a batch of Cartesian states from LCI to ECI.
///
/// Batch form of [`state_lci_to_eci`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the transformation context once and applies it to every
/// element. Evaluation runs on the global thread pool for large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_lci`: Cartesian LCI states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian ECI states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule
///
/// # Examples:
/// ```
/// use brahe::frames::states_lci_to_eci;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 60.0, epc + 120.0];
/// let states = vec![vector6_from_array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]); 3];
/// let out = states_lci_to_eci(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_lci_to_eci(
    epochs: &[Epoch],
    x_lci: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    batch_map_epochs(
        moon_earth_offset_state,
        |offset, x| x + offset,
        epochs,
        x_lci,
    )
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector3;
    use serial_test::serial;

    use super::*;
    use crate::constants::R_MOON;
    use crate::math::vector6_from_array;
    use crate::spice::{load_spice_kernel, unload_spice_kernel};
    use crate::time::TimeSystem;
    use crate::utils::testing::{
        CacheRedirect, setup_global_test_spice, synthetic_pck_kernel_bytes,
    };

    /// Epoch at ET ~500 s past J2000 TDB, inside the [0, 1000] coverage of the
    /// synthetic lunar PCK seeded by the offline tests below.
    fn epc_synth() -> Epoch {
        Epoch::from_jd(
            crate::constants::JD_J2000 + 500.0 / crate::constants::SECONDS_PER_DAY,
            TimeSystem::TDB,
        )
    }

    #[test]
    fn test_rotation_lfpa_to_lfme_is_small_constant() {
        let r = rotation_lfpa_to_lfme();
        // Total rotation angle ~ sqrt(0.2785^2 + 78.6944^2 + 67.8526^2)
        // arcsec ~ 1.04e2 arcsec ~ 5.04e-4 rad
        let angle = ((r.trace() - 1.0) / 2.0).acos();
        assert!(
            angle > 4.0e-4 && angle < 6.0e-4,
            "PA->ME angle {} rad out of range",
            angle
        );
        // Orthonormal, proper rotation
        assert_abs_diff_eq!(r.determinant(), 1.0, epsilon = 1e-12);
        let should_be_identity = r * r.transpose();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    should_be_identity[(i, j)],
                    if i == j { 1.0 } else { 0.0 },
                    epsilon = 1e-12
                );
            }
        }
    }

    #[test]
    fn test_rotation_lfpa_to_lfme_surface_displacement() {
        // NASA/TP-20220014814 Sec. 4.2: PA/ME surface displacement ~875 m.
        let r = rotation_lfpa_to_lfme();
        let v = Vector3::new(1737.4e3, 0.0, 0.0);
        let displacement = (r * v - v).norm();
        assert!(
            displacement > 850.0 && displacement < 900.0,
            "displacement {} m out of range",
            displacement
        );
    }

    #[test]
    fn test_rotation_lfme_to_lfpa_is_lfpa_to_lfme_transpose() {
        let r_pa_to_me = rotation_lfpa_to_lfme();
        let r_me_to_pa = rotation_lfme_to_lfpa();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(r_pa_to_me[(i, j)], r_me_to_pa[(j, i)], epsilon = 1e-15);
            }
        }
    }

    #[test]
    #[serial]
    fn test_rotation_lci_to_lfpa_matches_pck() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r = rotation_lci_to_lfpa(epc);
        let r_pck = crate::spice::pck_rotation_matrix(MOON_PA_FRAME_ID, epc).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(r[(i, j)], r_pck[(i, j)]); // bit-identical: same code path
            }
        }
    }

    #[test]
    #[serial]
    fn test_state_lci_to_lfpa_roundtrip() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x = vector6_from_array([R_MOON + 100e3, 1e5, 2e5, 0.0, 1.6e3, 0.0]);
        let x2 = state_lfpa_to_lci(epc, state_lci_to_lfpa(epc, x));
        for i in 0..6 {
            assert_abs_diff_eq!(x2[i], x[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[serial]
    fn test_state_lci_to_lfpa_transport_term() {
        // Same finite-difference pattern as the Mars module: numerically
        // differentiate R(t)*r and compare with the analytic transport term.
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r_inertial = Vector3::new(R_MOON + 100e3, 1e5, 2e5);
        let x = vector6_from_array([r_inertial[0], r_inertial[1], r_inertial[2], 0.0, 0.0, 0.0]);
        let dt = 1.0; // s
        let p0 = position_lci_to_lfpa(epc, r_inertial);
        let p1 = position_lci_to_lfpa(epc + dt, r_inertial);
        let v_fd = (p1 - p0) / dt;
        let v_analytic = state_lci_to_lfpa(epc, x).fixed_rows::<3>(3).into_owned();
        for i in 0..3 {
            assert_abs_diff_eq!(v_analytic[i], v_fd[i], epsilon = 1e-2);
        }
    }

    #[test]
    #[serial]
    fn test_lfpa_surface_point_is_stationary() {
        // A point rotating with the Moon (in the PA frame) has near-zero
        // LFPA velocity.
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r_lfpa = Vector3::new(R_MOON, 0.0, 0.0);
        let x_lci = state_lfpa_to_lci(
            epc,
            vector6_from_array([r_lfpa[0], r_lfpa[1], r_lfpa[2], 0.0, 0.0, 0.0]),
        );
        let back = state_lci_to_lfpa(epc, x_lci);
        for i in 3..6 {
            assert_abs_diff_eq!(back[i], 0.0, epsilon = 1e-9);
        }
    }

    #[test]
    #[serial]
    fn test_lci_lfme_roundtrip() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_lci = vector6_from_array([R_MOON + 100e3, 1e5, 2e5, 0.0, 1.6e3, 0.0]);

        let x_lfme = state_lci_to_lfme(epc, x_lci);
        let x_lci2 = state_lfme_to_lci(epc, x_lfme);
        for i in 0..6 {
            assert_abs_diff_eq!(x_lci2[i], x_lci[i], epsilon = 1e-6);
        }

        let p_lci = x_lci.fixed_rows::<3>(0).into_owned();
        let p_lfme = position_lci_to_lfme(epc, p_lci);
        let p_lci2 = position_lfme_to_lci(epc, p_lfme);
        for i in 0..3 {
            assert_abs_diff_eq!(p_lci2[i], p_lci[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[serial]
    fn test_lfme_surface_point_is_nearly_stationary() {
        // A point rotating with the Moon (in the LFME frame) has near-zero
        // LFME velocity, same as the LFPA case (LFME is rigidly offset from
        // LFPA by a constant rotation).
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r_lfme = Vector3::new(R_MOON, 0.0, 0.0);
        let x_lci = state_lfme_to_lci(
            epc,
            vector6_from_array([r_lfme[0], r_lfme[1], r_lfme[2], 0.0, 0.0, 0.0]),
        );
        let back = state_lci_to_lfme(epc, x_lci);
        for i in 3..6 {
            assert_abs_diff_eq!(back[i], 0.0, epsilon = 1e-9);
        }
    }

    #[test]
    #[serial]
    fn test_state_eci_to_lci_matches_spk() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x = vector6_from_array([1e8, 2e8, 3e8, 1.0, 2.0, 3.0]);
        let offset = crate::spice::spk_state(NAIFId::Moon, NAIFId::Earth, epc).unwrap();
        let expected = x - offset;
        let got = state_eci_to_lci(epc, x);
        for i in 0..6 {
            assert_abs_diff_eq!(got[i], expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[serial]
    fn test_state_eci_to_lci_roundtrip() {
        // Exercises position_eci_to_lci, position_lci_to_eci, and
        // state_lci_to_eci, which test_state_eci_to_lci_matches_spk doesn't
        // touch directly.
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_eci = vector6_from_array([1e8, 2e8, 3e8, 1.0, 2.0, 3.0]);

        let x_lci = state_eci_to_lci(epc, x_eci);
        let x_eci2 = state_lci_to_eci(epc, x_lci);
        for i in 0..6 {
            assert_abs_diff_eq!(x_eci2[i], x_eci[i], epsilon = 1e-6);
        }

        let p_eci = x_eci.fixed_rows::<3>(0).into_owned();
        let p_lci = position_eci_to_lci(epc, p_eci);
        let p_eci2 = position_lci_to_eci(epc, p_lci);
        for i in 0..3 {
            assert_abs_diff_eq!(p_eci2[i], p_eci[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[serial]
    fn test_eci_lci_transforms_offline() {
        // ECI <-> LCI translation needs only the (real) de440s ephemeris; no
        // lunar PCK required. Exercises state_eci_to_lci/state_lci_to_eci and
        // position_eci_to_lci/position_lci_to_eci offline.
        setup_global_test_spice();
        load_spice_kernel("de440s").unwrap();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_eci = vector6_from_array([1e8, 2e8, 3e8, 1.0, 2.0, 3.0]);

        let x_lci = state_eci_to_lci(epc, x_eci);
        let x_eci2 = state_lci_to_eci(epc, x_lci);
        for i in 0..6 {
            assert_abs_diff_eq!(x_eci2[i], x_eci[i], epsilon = 1e-6);
        }

        let p_eci = x_eci.fixed_rows::<3>(0).into_owned();
        let p_lci = position_eci_to_lci(epc, p_eci);
        let p_eci2 = position_lci_to_eci(epc, p_lci);
        for i in 0..3 {
            assert_abs_diff_eq!(p_eci2[i], p_eci[i], epsilon = 1e-6);
        }
        // Offset consistency: LCI is Earth->Moon translated.
        let offset = crate::spice::spk_position(NAIFId::Moon, NAIFId::Earth, epc).unwrap();
        for i in 0..3 {
            assert_abs_diff_eq!(p_lci[i], p_eci[i] - offset[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[serial]
    fn test_lunar_pck_transforms_offline() {
        // LFPA/LFME rotations and states auto-load the moon_pa_de440 PCK. Seed
        // a synthetic PCK (frame class 31008, ET coverage [0, 1000]) into a
        // redirected cache so the auto-load path resolves offline. The real
        // de440s stays resident throughout (never cleared); only the lunar PCK
        // is unloaded/reloaded so a prior panicked run cannot poison us.
        setup_global_test_spice();
        load_spice_kernel("de440s").unwrap();
        let _ = unload_spice_kernel("moon_pa_de440");
        {
            let cache = CacheRedirect::new();
            cache.seed_real_de440s();
            cache.seed(
                "moon_pa_de440_200625.bpc",
                &synthetic_pck_kernel_bytes(MOON_PA_FRAME_ID),
            );

            let epc = epc_synth();

            // Auto-load path: `ensure_lunar_pck_loaded` is a OnceLock latch,
            // so it only loads the kernel if no earlier test in this process
            // fired it. Load explicitly afterwards so the test is
            // deterministic regardless of latch state.
            assert!(!crate::spice::kernel_is_loaded("moon_pa_de440"));
            ensure_lunar_pck_loaded();
            if !crate::spice::kernel_is_loaded("moon_pa_de440") {
                load_spice_kernel("moon_pa_de440").unwrap();
            }
            let r_lci_lfpa = rotation_lci_to_lfpa(epc);
            assert!(crate::spice::kernel_is_loaded("moon_pa_de440"));

            // rotation_lfpa_to_lci is the transpose of rotation_lci_to_lfpa.
            let r_lfpa_lci = rotation_lfpa_to_lci(epc);
            for i in 0..3 {
                for j in 0..3 {
                    assert_abs_diff_eq!(r_lfpa_lci[(i, j)], r_lci_lfpa[(j, i)], epsilon = 1e-15);
                }
            }

            // rotation_lci_to_lfme = rotation_lfpa_to_lfme * rotation_lci_to_lfpa,
            // and rotation_lfme_to_lci is its transpose.
            let r_lci_lfme = rotation_lci_to_lfme(epc);
            let expected_lfme = rotation_lfpa_to_lfme() * r_lci_lfpa;
            for i in 0..3 {
                for j in 0..3 {
                    assert_abs_diff_eq!(r_lci_lfme[(i, j)], expected_lfme[(i, j)], epsilon = 1e-15);
                }
            }
            let r_lfme_lci = rotation_lfme_to_lci(epc);
            for i in 0..3 {
                for j in 0..3 {
                    assert_abs_diff_eq!(r_lfme_lci[(i, j)], r_lci_lfme[(j, i)], epsilon = 1e-15);
                }
            }

            // Position round trips (LCI <-> LFPA, LCI <-> LFME).
            let p_lci = Vector3::new(R_MOON + 100e3, 1e5, 2e5);
            let p_lfpa = position_lci_to_lfpa(epc, p_lci);
            let p_lci_back = position_lfpa_to_lci(epc, p_lfpa);
            for i in 0..3 {
                assert_abs_diff_eq!(p_lci_back[i], p_lci[i], epsilon = 1e-6);
            }
            let p_lfme = position_lci_to_lfme(epc, p_lci);
            let p_lci_back2 = position_lfme_to_lci(epc, p_lfme);
            for i in 0..3 {
                assert_abs_diff_eq!(p_lci_back2[i], p_lci[i], epsilon = 1e-6);
            }

            // State round trips exercise the velocity transport terms.
            let x_lci = vector6_from_array([R_MOON + 100e3, 1e5, 2e5, 0.0, 1.6e3, 0.0]);
            let x_lfpa = state_lci_to_lfpa(epc, x_lci);
            let x_lci_back = state_lfpa_to_lci(epc, x_lfpa);
            for i in 0..6 {
                assert_abs_diff_eq!(x_lci_back[i], x_lci[i], epsilon = 1e-6);
            }
            let x_lfme = state_lci_to_lfme(epc, x_lci);
            let x_lci_back2 = state_lfme_to_lci(epc, x_lfme);
            for i in 0..6 {
                assert_abs_diff_eq!(x_lci_back2[i], x_lci[i], epsilon = 1e-6);
            }

            // ensure_lunar_pck_loaded is idempotent: a second call while loaded
            // is a no-op (does not error).
            ensure_lunar_pck_loaded();

            unload_spice_kernel("moon_pa_de440").unwrap();
        }
        // The latch is now set but the kernel was just unloaded, so later
        // latch-relying tests would see it missing. Best-effort restore of
        // the real PCK (real cache; tolerated failure keeps this test
        // offline-safe when nothing later needs the kernel).
        let _ = load_spice_kernel("moon_pa_de440");
    }

    #[test]
    #[serial]
    fn test_batch_lunar_frames_match_scalar() {
        setup_global_test_spice();
        let epc0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let epochs: Vec<Epoch> = (0..3).map(|i| epc0 + 3600.0 * i as f64).collect();
        let states: Vec<SVector6> = (0..3)
            .map(|i| {
                vector6_from_array([R_MOON + 100e3 + 1e3 * i as f64, 0.0, 0.0, 0.0, 1.6e3, 0.0])
            })
            .collect();
        let positions: Vec<Vector3<f64>> = states
            .iter()
            .map(|s| Vector3::new(s[0], s[1], s[2]))
            .collect();

        for i in 0..3 {
            let e = epochs[i];
            assert_eq!(rotations_lci_to_lfpa(&epochs)[i], rotation_lci_to_lfpa(e));
            assert_eq!(rotations_lfpa_to_lci(&epochs)[i], rotation_lfpa_to_lci(e));
            assert_eq!(rotations_lci_to_lfme(&epochs)[i], rotation_lci_to_lfme(e));
            assert_eq!(rotations_lfme_to_lci(&epochs)[i], rotation_lfme_to_lci(e));

            assert_eq!(
                positions_lci_to_lfpa(&epochs, &positions).unwrap()[i],
                position_lci_to_lfpa(e, positions[i])
            );
            assert_eq!(
                positions_lfpa_to_lci(&epochs, &positions).unwrap()[i],
                position_lfpa_to_lci(e, positions[i])
            );
            assert_eq!(
                positions_lci_to_lfme(&epochs, &positions).unwrap()[i],
                position_lci_to_lfme(e, positions[i])
            );
            assert_eq!(
                positions_lfme_to_lci(&epochs, &positions).unwrap()[i],
                position_lfme_to_lci(e, positions[i])
            );
            assert_eq!(
                positions_lci_to_lfpa(&epochs[..1], &positions).unwrap()[i],
                position_lci_to_lfpa(epochs[0], positions[i])
            );

            assert_eq!(
                states_lci_to_lfpa(&epochs, &states).unwrap()[i],
                state_lci_to_lfpa(e, states[i])
            );
            assert_eq!(
                states_lfpa_to_lci(&epochs, &states).unwrap()[i],
                state_lfpa_to_lci(e, states[i])
            );
            assert_eq!(
                states_lci_to_lfme(&epochs, &states).unwrap()[i],
                state_lci_to_lfme(e, states[i])
            );
            assert_eq!(
                states_lfme_to_lci(&epochs, &states).unwrap()[i],
                state_lfme_to_lci(e, states[i])
            );
            assert_eq!(
                states_lci_to_lfme(&epochs[..1], &states).unwrap()[i],
                state_lci_to_lfme(epochs[0], states[i])
            );

            assert_eq!(
                positions_eci_to_lci(&epochs, &positions).unwrap()[i],
                position_eci_to_lci(e, positions[i])
            );
            assert_eq!(
                positions_lci_to_eci(&epochs, &positions).unwrap()[i],
                position_lci_to_eci(e, positions[i])
            );
            assert_eq!(
                states_eci_to_lci(&epochs, &states).unwrap()[i],
                state_eci_to_lci(e, states[i])
            );
            assert_eq!(
                states_lci_to_eci(&epochs, &states).unwrap()[i],
                state_lci_to_eci(e, states[i])
            );
            assert_eq!(
                states_eci_to_lci(&epochs, &states[..1]).unwrap()[i],
                state_eci_to_lci(e, states[0])
            );
        }

        let lfpa = states_lci_to_lfpa(&epochs, &states).unwrap();
        let back = states_lfpa_to_lci(&epochs, &lfpa).unwrap();
        for i in 0..3 {
            for k in 0..3 {
                assert_abs_diff_eq!(back[i][k], states[i][k], epsilon = 1e-6);
            }
        }
        assert!(states_lci_to_lfpa(&epochs[..2], &states).is_err());
        assert!(rotations_lci_to_lfpa(&[]).is_empty());
    }
}
