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

use super::iau_rotation::{euler313_omega_body, rx, ry, rz};

/// NAIF frame class ID of the DE440 lunar principal-axis frame
/// (`MOON_PA_DE440`), as defined in NAIF's lunar frames kernel.
const MOON_PA_FRAME_ID: i32 = 31008;

/// Idempotently loads the `moon_pa_de440` binary PCK (downloading it to
/// `~/.cache/brahe/naif` if needed) into the global SPICE kernel registry.
///
/// Called automatically by every LFPA/LFME transformation in this module;
/// not normally called directly.
///
/// # Panics
/// Panics with an actionable message if the kernel cannot be loaded (e.g.
/// no network access and no cached copy).
pub(crate) fn ensure_lunar_pck_loaded() {
    // Allocation-free registry check. Not a `OnceLock` latch: the registry
    // can be cleared (`clear_kernels`) or the kernel unloaded at runtime, so
    // the check must consult live registry state on every call.
    if crate::spice::kernel_is_loaded("moon_pa_de440") {
        return;
    }
    crate::spice::load_kernel("moon_pa_de440").unwrap_or_else(|e| {
        panic!(
            "Failed to auto-load lunar PCK 'moon_pa_de440': {}. \
             Download manually and call brahe::spice::load_kernel(<path>).",
            e
        )
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
/// ```no_run
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
/// ```no_run
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
/// ```no_run
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
/// ```no_run
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
/// ```no_run
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
/// ```no_run
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
/// ```no_run
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
/// ```no_run
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
/// ```no_run
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
    ensure_lunar_pck_loaded();
    let (angles, rates) = crate::spice::pck_euler_angles(MOON_PA_FRAME_ID, epc)
        .unwrap_or_else(|e| panic!("Lunar PCK orientation query failed: {}", e));
    let r_mat = crate::spice::pck_rotation_matrix(MOON_PA_FRAME_ID, epc)
        .unwrap_or_else(|e| panic!("Lunar PCK orientation query failed: {}", e))
        .to_matrix();
    let omega_b = euler313_omega_body(angles, rates);

    let r = x_lci.fixed_rows::<3>(0);
    let v = x_lci.fixed_rows::<3>(3);

    let r_b: Vector3<f64> = r_mat * r;
    let v_b: Vector3<f64> = r_mat * v - omega_b.cross(&r_b);

    SVector6::new(r_b[0], r_b[1], r_b[2], v_b[0], v_b[1], v_b[2])
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
/// ```no_run
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
    ensure_lunar_pck_loaded();
    let (angles, rates) = crate::spice::pck_euler_angles(MOON_PA_FRAME_ID, epc)
        .unwrap_or_else(|e| panic!("Lunar PCK orientation query failed: {}", e));
    let r_mat = crate::spice::pck_rotation_matrix(MOON_PA_FRAME_ID, epc)
        .unwrap_or_else(|e| panic!("Lunar PCK orientation query failed: {}", e))
        .to_matrix();
    let omega_b = euler313_omega_body(angles, rates);

    let r_b: Vector3<f64> = x_lfpa.fixed_rows::<3>(0).into_owned();
    let v_b: Vector3<f64> = x_lfpa.fixed_rows::<3>(3).into_owned();

    let r: Vector3<f64> = r_mat.transpose() * r_b;
    let v: Vector3<f64> = r_mat.transpose() * (v_b + omega_b.cross(&r_b));

    SVector6::new(r[0], r[1], r[2], v[0], v[1], v[2])
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
/// ```no_run
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
    ensure_lunar_pck_loaded();
    let (angles, rates) = crate::spice::pck_euler_angles(MOON_PA_FRAME_ID, epc)
        .unwrap_or_else(|e| panic!("Lunar PCK orientation query failed: {}", e));
    let r_pa_to_me = rotation_lfpa_to_lfme();
    let r_mat = r_pa_to_me * rotation_lci_to_lfpa(epc);
    let omega_b = r_pa_to_me * euler313_omega_body(angles, rates);

    let r = x_lci.fixed_rows::<3>(0);
    let v = x_lci.fixed_rows::<3>(3);

    let r_b: Vector3<f64> = r_mat * r;
    let v_b: Vector3<f64> = r_mat * v - omega_b.cross(&r_b);

    SVector6::new(r_b[0], r_b[1], r_b[2], v_b[0], v_b[1], v_b[2])
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
/// ```no_run
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
    ensure_lunar_pck_loaded();
    let (angles, rates) = crate::spice::pck_euler_angles(MOON_PA_FRAME_ID, epc)
        .unwrap_or_else(|e| panic!("Lunar PCK orientation query failed: {}", e));
    let r_pa_to_me = rotation_lfpa_to_lfme();
    let r_mat = r_pa_to_me * rotation_lci_to_lfpa(epc);
    let omega_b = r_pa_to_me * euler313_omega_body(angles, rates);

    let r_b: Vector3<f64> = x_lfme.fixed_rows::<3>(0).into_owned();
    let v_b: Vector3<f64> = x_lfme.fixed_rows::<3>(3).into_owned();

    let r: Vector3<f64> = r_mat.transpose() * r_b;
    let v: Vector3<f64> = r_mat.transpose() * (v_b + omega_b.cross(&r_b));

    SVector6::new(r[0], r[1], r[2], v[0], v[1], v[2])
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
/// ```no_run
/// use brahe::frames::position_eci_to_lci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_eci = Vector3::new(1e7, 2e7, 3e7);
/// let x_lci = position_eci_to_lci(epc, x_eci);
/// ```
pub fn position_eci_to_lci(epc: Epoch, x_eci: Vector3<f64>) -> Vector3<f64> {
    let offset = spk_position(NAIFId::Moon, NAIFId::Earth, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)");
    x_eci - offset
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
/// ```no_run
/// use brahe::frames::position_lci_to_eci;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_lci = Vector3::new(1e7, 2e7, 3e7);
/// let x_eci = position_lci_to_eci(epc, x_lci);
/// ```
pub fn position_lci_to_eci(epc: Epoch, x_lci: Vector3<f64>) -> Vector3<f64> {
    let offset = spk_position(NAIFId::Moon, NAIFId::Earth, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)");
    x_lci + offset
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
/// ```no_run
/// use brahe::frames::state_eci_to_lci;
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_eci = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
/// let x_lci = state_eci_to_lci(epc, x_eci);
/// ```
pub fn state_eci_to_lci(epc: Epoch, x_eci: SVector6) -> SVector6 {
    let offset = spk_state(NAIFId::Moon, NAIFId::Earth, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)");
    x_eci - offset
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
/// ```no_run
/// use brahe::frames::state_lci_to_eci;
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_lci = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
/// let x_eci = state_lci_to_eci(epc, x_lci);
/// ```
pub fn state_lci_to_eci(epc: Epoch, x_lci: SVector6) -> SVector6 {
    let offset = spk_state(NAIFId::Moon, NAIFId::Earth, epc)
        .expect("SPK query failed: ensure a DE kernel is available (auto-init de440s)");
    x_lci + offset
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
    use crate::time::TimeSystem;
    use crate::utils::testing::setup_global_test_spice;

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
    #[cfg_attr(not(feature = "integration"), ignore)]
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
    #[cfg_attr(not(feature = "integration"), ignore)]
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
    #[cfg_attr(not(feature = "integration"), ignore)]
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
    #[cfg_attr(not(feature = "integration"), ignore)]
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
    #[cfg_attr(not(feature = "integration"), ignore)]
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
    #[cfg_attr(not(feature = "integration"), ignore)]
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
    #[cfg_attr(not(feature = "integration"), ignore)]
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
    #[cfg_attr(not(feature = "integration"), ignore)]
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
}
