/*!
 * Centralized reference frame router.
 *
 * This module defines [`ReferenceFrame`], an enum spanning every reference
 * frame implemented elsewhere in [`crate::frames`], and a small set of
 * router functions (`rotation_frame_to_frame`, `position_frame_to_frame`,
 * `state_frame_to_frame`) that convert between *any* two frames in the
 * enum, dispatching to the pairwise transformation functions defined in the
 * rest of this module (`gcrf_itrf`, `eme_2000`, `lunar`, `mars`,
 * `iau_rotation`) and to [`crate::spice`] for cross-body translations.
 *
 * # Hub-and-spoke design
 *
 * Every conversion is performed in two independent steps:
 *
 * 1. **Orientation**: the state is rotated from the source frame's axes
 *    into ICRF axes (still centered on the source frame's origin), using
 *    the exact pairwise rotation/transport-velocity transform for that
 *    frame. [`ReferenceFrame::GCRF`], [`ReferenceFrame::LCI`],
 *    [`ReferenceFrame::MCI`], [`ReferenceFrame::EMBI`],
 *    [`ReferenceFrame::SSBI`], and [`ReferenceFrame::BodyCenteredICRF`] are
 *    already ICRF-aligned (this crate treats GCRF, LCI, MCI, and the ICRF
 *    as equivalent axes, consistent with [`crate::frames::lunar`] and
 *    [`crate::frames::mars`]), so this step is the identity for them.
 * 2. **Translation**: if the source and target frames are centered on
 *    different bodies, the ICRF-axis state is re-centered using
 *    `center_offset_state`, the single seam where a body-to-body
 *    translation is looked up (currently always via SPK; a future
 *    analytic ephemeris provider would plug in here without touching any
 *    other code in this module).
 *
 * The target frame's axes are then produced by inverting step 1 for the
 * target frame. Same-center conversions (e.g. GCRF <-> ITRF, LCI <->
 * LFPA) skip the translation step entirely and never query SPK — the
 * router is bit-identical to the underlying pairwise function in that
 * case.
 *
 * # Frame centers
 *
 * | Frame | Center (NAIF ID) |
 * |---|---|
 * | GCRF, ITRF, EME2000 | Earth (399) |
 * | LCI, LFPA, LFME | Moon (301) |
 * | MCI, MCMF | Mars system barycenter (4) |
 * | EMBI | Earth-Moon barycenter (3) |
 * | SSBI | Solar System barycenter (0) |
 * | `BodyCenteredICRF(id)` | `id` |
 * | `BodyFixedIAU(id)` | `id` (the body itself, e.g. 499 for Mars — see caveat below) |
 * | `BodyFixedPCK { center, .. }` | `center` |
 *
 * `BodyFixedIAU(id)` is centered on the body itself rather than its
 * system barycenter. For bodies without a direct SPK ephemeris segment
 * for that body ID (e.g. Mars, NAIF 499, whose ephemeris is only
 * available for its barycenter, NAIF 4 — see
 * [`crate::frames::mars`]), a translation into or out of
 * `BodyFixedIAU(id)` from a differently-centered frame will surface the
 * underlying SPK lookup error. Use [`ReferenceFrame::MCMF`] (or another
 * named, barycenter-anchored frame) for translated Mars-system
 * conversions; `BodyFixedIAU(499)` remains exact and always available for
 * *rotation-only* queries (its orientation dispatch is identical to
 * `MCMF`'s).
 */

use nalgebra::Vector3;
use std::fmt;
use std::str::FromStr;

use crate::math::{SMatrix3, SVector6};
use crate::spice::{NAIF_EARTH, NAIF_EMB, NAIF_MARS_BARYCENTER, NAIF_MOON, NAIF_SSB};
use crate::time::Epoch;
use crate::utils::BraheError;

use super::eme_2000::rotation_gcrf_to_eme2000;
use super::gcrf_itrf::rotation_gcrf_to_itrf;
use super::iau_rotation::{
    body_fixed_iau_angles_and_rates, euler313_omega_body, rotation_icrf_to_body_fixed_iau,
};
use super::lunar::{rotation_lci_to_lfme, rotation_lci_to_lfpa};
use super::mars::rotation_mci_to_mcmf;

/// A reference frame supported by the centralized frame router
/// ([`rotation_frame_to_frame`], [`position_frame_to_frame`],
/// [`state_frame_to_frame`]).
///
/// Includes every named frame defined elsewhere in [`crate::frames`]
/// (`GCRF`, `ITRF`, `EME2000`, the lunar frames `LCI`/`LFPA`/`LFME`, and
/// the Mars frames `MCI`/`MCMF`), the Earth-Moon and Solar System
/// barycentric inertial frames (`EMBI`, `SSBI`), and three generic
/// variants for bodies without a dedicated named frame:
///
/// - `BodyCenteredICRF(naif_id)`: ICRF-aligned axes centered on `naif_id`.
/// - `BodyFixedIAU(naif_id)`: the IAU/WGCCRE body-fixed frame of
///   `naif_id` (see [`rotation_icrf_to_body_fixed_iau`]), centered on
///   `naif_id` itself.
/// - `BodyFixedPCK { center, frame_id }`: a body-fixed frame evaluated
///   from a loaded binary PCK's `frame_id` (see
///   [`crate::spice::pck_rotation_matrix`]), centered on `center`.
///
/// See the module-level documentation for the full center table and the
/// hub-and-spoke conversion design.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ReferenceFrame {
    /// Geocentric Celestial Reference Frame (ICRF-aligned, Earth-centered).
    GCRF,
    /// International Terrestrial Reference Frame (Earth-fixed).
    ITRF,
    /// Earth Mean Equator and Equinox of J2000.0.
    EME2000,
    /// Lunar-Centered Inertial (ICRF-aligned, Moon-centered).
    LCI,
    /// Lunar-Fixed Principal Axis (DE440 `MOON_PA_DE440`).
    LFPA,
    /// Lunar-Fixed Mean Earth/polar-axis.
    LFME,
    /// Mars-Centered Inertial (ICRF-aligned, Mars system barycenter-centered).
    MCI,
    /// Mars-Centered Mars-Fixed (IAU/WGCCRE Mars rotation model).
    MCMF,
    /// Earth-Moon Barycentric Inertial (ICRF-aligned).
    EMBI,
    /// Solar System Barycentric Inertial (ICRF-aligned).
    SSBI,
    /// ICRF-aligned axes centered on the given NAIF ID.
    BodyCenteredICRF(i32),
    /// IAU/WGCCRE body-fixed frame of the given NAIF ID, centered on that
    /// same NAIF ID.
    BodyFixedIAU(i32),
    /// Body-fixed frame evaluated from a loaded binary PCK's `frame_id`,
    /// centered on `center`.
    BodyFixedPCK {
        /// NAIF ID of the frame's center.
        center: i32,
        /// NAIF binary PCK frame class ID (e.g. 31008 for `MOON_PA_DE440`).
        frame_id: i32,
    },
}

impl ReferenceFrame {
    /// NAIF ID of this frame's origin.
    ///
    /// See the module-level documentation for the full frame-to-center
    /// table, including the `BodyFixedIAU` caveat.
    ///
    /// # Returns:
    /// - `naif_id`: NAIF ID of the frame's center
    ///
    /// # Examples:
    /// ```
    /// use brahe::frames::ReferenceFrame;
    ///
    /// assert_eq!(ReferenceFrame::GCRF.center_naif_id(), 399);
    /// assert_eq!(ReferenceFrame::LCI.center_naif_id(), 301);
    /// ```
    pub fn center_naif_id(&self) -> i32 {
        match self {
            ReferenceFrame::GCRF | ReferenceFrame::ITRF | ReferenceFrame::EME2000 => NAIF_EARTH,
            ReferenceFrame::LCI | ReferenceFrame::LFPA | ReferenceFrame::LFME => NAIF_MOON,
            ReferenceFrame::MCI | ReferenceFrame::MCMF => NAIF_MARS_BARYCENTER,
            ReferenceFrame::EMBI => NAIF_EMB,
            ReferenceFrame::SSBI => NAIF_SSB,
            ReferenceFrame::BodyCenteredICRF(id) => *id,
            ReferenceFrame::BodyFixedIAU(id) => *id,
            ReferenceFrame::BodyFixedPCK { center, .. } => *center,
        }
    }

    /// Rotates a state from this frame's own axes (still centered on this
    /// frame's origin) into ICRF axes. Identity for frames already
    /// ICRF-aligned; inverts the frame's body-fixed transport transform
    /// otherwise.
    fn state_to_icrf_axes(&self, epc: Epoch, x: SVector6) -> Result<SVector6, BraheError> {
        match self {
            ReferenceFrame::GCRF
            | ReferenceFrame::LCI
            | ReferenceFrame::MCI
            | ReferenceFrame::EMBI
            | ReferenceFrame::SSBI
            | ReferenceFrame::BodyCenteredICRF(_) => Ok(x),
            ReferenceFrame::ITRF => Ok(super::gcrf_itrf::state_itrf_to_gcrf(epc, x)),
            ReferenceFrame::EME2000 => Ok(super::eme_2000::state_eme2000_to_gcrf(x)),
            ReferenceFrame::LFPA => Ok(super::lunar::state_lfpa_to_lci(epc, x)),
            ReferenceFrame::LFME => Ok(super::lunar::state_lfme_to_lci(epc, x)),
            ReferenceFrame::MCMF => Ok(super::mars::state_mcmf_to_mci(epc, x)),
            ReferenceFrame::BodyFixedIAU(id) => state_iau_body_to_icrf(*id, epc, x),
            ReferenceFrame::BodyFixedPCK { frame_id, .. } => {
                state_pck_body_to_icrf(*frame_id, epc, x)
            }
        }
    }

    /// Rotates an ICRF-axis state (already translated to this frame's
    /// origin) into this frame's own axes. Inverse of
    /// [`ReferenceFrame::state_to_icrf_axes`].
    fn state_from_icrf_axes(&self, epc: Epoch, x_icrf: SVector6) -> Result<SVector6, BraheError> {
        match self {
            ReferenceFrame::GCRF
            | ReferenceFrame::LCI
            | ReferenceFrame::MCI
            | ReferenceFrame::EMBI
            | ReferenceFrame::SSBI
            | ReferenceFrame::BodyCenteredICRF(_) => Ok(x_icrf),
            ReferenceFrame::ITRF => Ok(super::gcrf_itrf::state_gcrf_to_itrf(epc, x_icrf)),
            ReferenceFrame::EME2000 => Ok(super::eme_2000::state_gcrf_to_eme2000(x_icrf)),
            ReferenceFrame::LFPA => Ok(super::lunar::state_lci_to_lfpa(epc, x_icrf)),
            ReferenceFrame::LFME => Ok(super::lunar::state_lci_to_lfme(epc, x_icrf)),
            ReferenceFrame::MCMF => Ok(super::mars::state_mci_to_mcmf(epc, x_icrf)),
            ReferenceFrame::BodyFixedIAU(id) => state_icrf_to_iau_body(*id, epc, x_icrf),
            ReferenceFrame::BodyFixedPCK { frame_id, .. } => {
                state_icrf_to_pck_body(*frame_id, epc, x_icrf)
            }
        }
    }
}

impl fmt::Display for ReferenceFrame {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ReferenceFrame::GCRF => write!(f, "GCRF"),
            ReferenceFrame::ITRF => write!(f, "ITRF"),
            ReferenceFrame::EME2000 => write!(f, "EME2000"),
            ReferenceFrame::LCI => write!(f, "LCI"),
            ReferenceFrame::LFPA => write!(f, "LFPA"),
            ReferenceFrame::LFME => write!(f, "LFME"),
            ReferenceFrame::MCI => write!(f, "MCI"),
            ReferenceFrame::MCMF => write!(f, "MCMF"),
            ReferenceFrame::EMBI => write!(f, "EMBI"),
            ReferenceFrame::SSBI => write!(f, "SSBI"),
            ReferenceFrame::BodyCenteredICRF(id) => write!(f, "BodyCenteredICRF({})", id),
            ReferenceFrame::BodyFixedIAU(id) => write!(f, "BodyFixedIAU({})", id),
            ReferenceFrame::BodyFixedPCK { center, frame_id } => {
                write!(f, "BodyFixedPCK(center={}, frame_id={})", center, frame_id)
            }
        }
    }
}

impl FromStr for ReferenceFrame {
    type Err = BraheError;

    /// Parses a [`ReferenceFrame`] from its [`Display`](fmt::Display)
    /// representation (named variants only, case-insensitive), plus the
    /// common aliases `"ECI"` (-> [`ReferenceFrame::GCRF`]) and `"ECEF"`
    /// (-> [`ReferenceFrame::ITRF`]).
    ///
    /// The generic variants (`BodyCenteredICRF`, `BodyFixedIAU`,
    /// `BodyFixedPCK`) are not parseable from a string; construct them
    /// directly.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_uppercase().as_str() {
            "ECI" => Ok(ReferenceFrame::GCRF),
            "ECEF" => Ok(ReferenceFrame::ITRF),
            "GCRF" => Ok(ReferenceFrame::GCRF),
            "ITRF" => Ok(ReferenceFrame::ITRF),
            "EME2000" => Ok(ReferenceFrame::EME2000),
            "LCI" => Ok(ReferenceFrame::LCI),
            "LFPA" => Ok(ReferenceFrame::LFPA),
            "LFME" => Ok(ReferenceFrame::LFME),
            "MCI" => Ok(ReferenceFrame::MCI),
            "MCMF" => Ok(ReferenceFrame::MCMF),
            "EMBI" => Ok(ReferenceFrame::EMBI),
            "SSBI" => Ok(ReferenceFrame::SSBI),
            _ => Err(BraheError::ParseError(format!(
                "Unknown reference frame '{}'. Supported: GCRF (alias ECI), ITRF (alias ECEF), \
                 EME2000, LCI, LFPA, LFME, MCI, MCMF, EMBI, SSBI",
                s
            ))),
        }
    }
}

/// Rotates an ICRF-axis state into the IAU/WGCCRE body-fixed frame of
/// `naif_id`, including the velocity transport term induced by the
/// body's rotation. Generic form of [`super::mars::state_mci_to_mcmf`],
/// parameterized over `naif_id` for use by [`ReferenceFrame::BodyFixedIAU`].
fn state_icrf_to_iau_body(
    naif_id: i32,
    epc: Epoch,
    x_icrf: SVector6,
) -> Result<SVector6, BraheError> {
    let (angles, rates) = body_fixed_iau_angles_and_rates(naif_id, epc)?;
    let r_mat = rotation_icrf_to_body_fixed_iau(naif_id, epc)?;
    let omega_b = euler313_omega_body(angles, rates);

    let r = x_icrf.fixed_rows::<3>(0);
    let v = x_icrf.fixed_rows::<3>(3);

    let r_b: Vector3<f64> = r_mat * r;
    let v_b: Vector3<f64> = r_mat * v - omega_b.cross(&r_b);

    Ok(SVector6::new(
        r_b[0], r_b[1], r_b[2], v_b[0], v_b[1], v_b[2],
    ))
}

/// Rotates a state in the IAU/WGCCRE body-fixed frame of `naif_id` into
/// ICRF axes. Inverse of [`state_icrf_to_iau_body`].
fn state_iau_body_to_icrf(
    naif_id: i32,
    epc: Epoch,
    x_body: SVector6,
) -> Result<SVector6, BraheError> {
    let (angles, rates) = body_fixed_iau_angles_and_rates(naif_id, epc)?;
    let r_mat = rotation_icrf_to_body_fixed_iau(naif_id, epc)?;
    let omega_b = euler313_omega_body(angles, rates);

    let r_b: Vector3<f64> = x_body.fixed_rows::<3>(0).into_owned();
    let v_b: Vector3<f64> = x_body.fixed_rows::<3>(3).into_owned();

    let r: Vector3<f64> = r_mat.transpose() * r_b;
    let v: Vector3<f64> = r_mat.transpose() * (v_b + omega_b.cross(&r_b));

    Ok(SVector6::new(r[0], r[1], r[2], v[0], v[1], v[2]))
}

/// Rotates an ICRF-axis state into the body-fixed frame defined by binary
/// PCK `frame_id`, including the velocity transport term induced by the
/// frame's rotation. Generic form of
/// [`super::lunar::state_lci_to_lfpa`], parameterized over `frame_id` for
/// use by [`ReferenceFrame::BodyFixedPCK`]. Unlike the lunar-specific
/// helpers, this does not auto-load any PCK — the caller's kernel must
/// already be loaded (see [`crate::spice::pck_rotation_matrix`]).
fn state_icrf_to_pck_body(
    frame_id: i32,
    epc: Epoch,
    x_icrf: SVector6,
) -> Result<SVector6, BraheError> {
    let (angles, rates) = crate::spice::pck_euler_angles(frame_id, epc)?;
    let r_mat = crate::spice::pck_rotation_matrix(frame_id, epc)?;
    let omega_b = euler313_omega_body(angles, rates);

    let r = x_icrf.fixed_rows::<3>(0);
    let v = x_icrf.fixed_rows::<3>(3);

    let r_b: Vector3<f64> = r_mat * r;
    let v_b: Vector3<f64> = r_mat * v - omega_b.cross(&r_b);

    Ok(SVector6::new(
        r_b[0], r_b[1], r_b[2], v_b[0], v_b[1], v_b[2],
    ))
}

/// Rotates a state in the body-fixed frame defined by binary PCK
/// `frame_id` into ICRF axes. Inverse of [`state_icrf_to_pck_body`].
fn state_pck_body_to_icrf(
    frame_id: i32,
    epc: Epoch,
    x_body: SVector6,
) -> Result<SVector6, BraheError> {
    let (angles, rates) = crate::spice::pck_euler_angles(frame_id, epc)?;
    let r_mat = crate::spice::pck_rotation_matrix(frame_id, epc)?;
    let omega_b = euler313_omega_body(angles, rates);

    let r_b: Vector3<f64> = x_body.fixed_rows::<3>(0).into_owned();
    let v_b: Vector3<f64> = x_body.fixed_rows::<3>(3).into_owned();

    let r: Vector3<f64> = r_mat.transpose() * r_b;
    let v: Vector3<f64> = r_mat.transpose() * (v_b + omega_b.cross(&r_b));

    Ok(SVector6::new(r[0], r[1], r[2], v[0], v[1], v[2]))
}

/// Rotation matrix from ICRF axes to `frame`'s own axes at `epc`. Identity
/// for ICRF-aligned frames.
fn icrf_to_frame_dcm(frame: ReferenceFrame, epc: Epoch) -> Result<SMatrix3, BraheError> {
    match frame {
        ReferenceFrame::GCRF
        | ReferenceFrame::LCI
        | ReferenceFrame::MCI
        | ReferenceFrame::EMBI
        | ReferenceFrame::SSBI
        | ReferenceFrame::BodyCenteredICRF(_) => Ok(SMatrix3::identity()),
        ReferenceFrame::ITRF => Ok(rotation_gcrf_to_itrf(epc)),
        ReferenceFrame::EME2000 => Ok(rotation_gcrf_to_eme2000()),
        ReferenceFrame::LFPA => Ok(rotation_lci_to_lfpa(epc)),
        ReferenceFrame::LFME => Ok(rotation_lci_to_lfme(epc)),
        ReferenceFrame::MCMF => Ok(rotation_mci_to_mcmf(epc)),
        ReferenceFrame::BodyFixedIAU(id) => rotation_icrf_to_body_fixed_iau(id, epc),
        ReferenceFrame::BodyFixedPCK { frame_id, .. } => {
            crate::spice::pck_rotation_matrix(frame_id, epc)
        }
    }
}

/// Computes the rotation matrix transforming `from` axes into `to` axes
/// at `epc`.
///
/// Purely an orientation query: does not depend on, and does not query,
/// either frame's center (in particular, this never touches SPK).
/// Equivalent to `R_to(epc) * R_from(epc)^T`, where `R_x` is `x`'s
/// ICRF -> `x` rotation matrix (identity for ICRF-aligned frames).
///
/// # Arguments
/// - `from`: Source reference frame
/// - `to`: Target reference frame
/// - `epc`: Epoch instant for computation of the transformation
///
/// # Returns
/// - `r`: 3x3 rotation matrix transforming `from` -> `to`
///
/// # Examples:
/// ```
/// use brahe::frames::{ReferenceFrame, rotation_frame_to_frame};
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_frame_to_frame(ReferenceFrame::MCI, ReferenceFrame::MCMF, epc).unwrap();
/// ```
pub fn rotation_frame_to_frame(
    from: ReferenceFrame,
    to: ReferenceFrame,
    epc: Epoch,
) -> Result<SMatrix3, BraheError> {
    let r_to = icrf_to_frame_dcm(to, epc)?;
    let r_from = icrf_to_frame_dcm(from, epc)?;
    Ok(r_to * r_from.transpose())
}

/// Transforms a Cartesian position from `from` to `to` at `epc`.
///
/// Same hub-and-spoke design as [`state_frame_to_frame`], without the
/// velocity transport terms. Same-center conversions never touch SPK.
///
/// # Arguments
/// - `from`: Source reference frame
/// - `to`: Target reference frame
/// - `epc`: Epoch instant for computation of the transformation
/// - `x`: Cartesian position in `from` axes/center. Units: (*m*)
///
/// # Returns
/// - `x`: Cartesian position in `to` axes/center. Units: (*m*)
///
/// # Examples:
/// ```no_run
/// use brahe::constants::R_EARTH;
/// use brahe::frames::{ReferenceFrame, position_frame_to_frame};
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = Vector3::new(R_EARTH + 500e3, 0.0, 0.0);
/// let x_itrf =
///     position_frame_to_frame(ReferenceFrame::GCRF, ReferenceFrame::ITRF, epc, x_gcrf).unwrap();
/// ```
pub fn position_frame_to_frame(
    from: ReferenceFrame,
    to: ReferenceFrame,
    epc: Epoch,
    x: Vector3<f64>,
) -> Result<Vector3<f64>, BraheError> {
    if from == to {
        return Ok(x);
    }
    let r_from = icrf_to_frame_dcm(from, epc)?;
    let x_icrf = r_from.transpose() * x;
    let x_translated = if from.center_naif_id() == to.center_naif_id() {
        x_icrf
    } else {
        let offset = center_offset_state(from.center_naif_id(), to.center_naif_id(), epc)?;
        x_icrf - offset.fixed_rows::<3>(0).into_owned()
    };
    let r_to = icrf_to_frame_dcm(to, epc)?;
    Ok(r_to * x_translated)
}

/// Transforms a Cartesian state (position and velocity) from `from` to
/// `to` at `epc`.
///
/// Uses a hub-and-spoke design: the state is first rotated from `from`
/// axes into ICRF axes (an exact orientation + velocity-transport
/// transform, still centered on `from`'s origin), then re-centered onto
/// `to`'s origin via `center_offset_state` if the two frames have
/// different centers, then rotated into `to` axes. Same-center
/// conversions (e.g. GCRF <-> ITRF) skip the re-centering step and never
/// touch SPK — the result is bit-identical to the underlying pairwise
/// function.
///
/// # Arguments
/// - `from`: Source reference frame
/// - `to`: Target reference frame
/// - `epc`: Epoch instant for computation of the transformation
/// - `x`: Cartesian state (position, velocity) in `from` axes/center. Units: (*m*; *m/s*)
///
/// # Returns
/// - `x`: Cartesian state (position, velocity) in `to` axes/center. Units: (*m*; *m/s*)
///
/// # Examples:
/// ```no_run
/// use brahe::constants::R_MOON;
/// use brahe::frames::{ReferenceFrame, state_frame_to_frame};
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = vector6_from_array([1e8, -2e8, 5e7, 1.0e3, -2.0e3, 0.5e3]);
/// let x_lfpa =
///     state_frame_to_frame(ReferenceFrame::GCRF, ReferenceFrame::LFPA, epc, x_gcrf).unwrap();
/// ```
pub fn state_frame_to_frame(
    from: ReferenceFrame,
    to: ReferenceFrame,
    epc: Epoch,
    x: SVector6,
) -> Result<SVector6, BraheError> {
    if from == to {
        return Ok(x);
    }
    let x_icrf = from.state_to_icrf_axes(epc, x)?;
    let x_translated = if from.center_naif_id() == to.center_naif_id() {
        x_icrf
    } else {
        x_icrf - center_offset_state(from.center_naif_id(), to.center_naif_id(), epc)?
    };
    to.state_from_icrf_axes(epc, x_translated)
}

/// State of `to_center` relative to `from_center` at `epc`, in ICRF axes.
///
/// The single translation seam used by [`position_frame_to_frame`] and
/// [`state_frame_to_frame`] whenever two frames are centered on
/// different bodies. Backed by [`crate::spice::spk_state`] today; a
/// future analytic ephemeris provider (e.g. for center pairs without SPK
/// coverage) would plug in here without any change to the router
/// functions above.
///
/// # Arguments
/// - `from_center`: NAIF ID of the origin frame's center
/// - `to_center`: NAIF ID of the destination frame's center
/// - `epc`: Epoch at which to evaluate the offset
///
/// # Returns
/// - `x`: State `[x, y, z, vx, vy, vz]` of `to_center` relative to
///   `from_center`, in ICRF axes. Units: (*m*; *m/s*)
pub(crate) fn center_offset_state(
    from_center: i32,
    to_center: i32,
    epc: Epoch,
) -> Result<SVector6, BraheError> {
    crate::spice::spk_state(to_center, from_center, epc)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use serial_test::serial;

    use super::*;
    use crate::constants::{DEGREES, R_EARTH};
    use crate::coordinates::state_koe_to_eci;
    use crate::math::vector6_from_array;
    use crate::spice::{NAIF_EARTH, NAIF_MARS_BARYCENTER};
    use crate::time::TimeSystem;
    use crate::utils::testing::{setup_global_test_eop, setup_global_test_spice};

    #[test]
    #[serial] // EOP global
    fn test_router_matches_pairwise_gcrf_itrf() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x = state_koe_to_eci(
            vector6_from_array([R_EARTH + 500e3, 1e-3, 97.8, 75.0, 25.0, 45.0]),
            DEGREES,
        );
        let via_router =
            state_frame_to_frame(ReferenceFrame::GCRF, ReferenceFrame::ITRF, epc, x).unwrap();
        let pairwise = crate::frames::state_gcrf_to_itrf(epc, x);
        for i in 0..6 {
            // bit-identical: same-center path must not touch SPK
            assert_eq!(via_router[i], pairwise[i]);
        }
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial]
    fn test_router_matches_pairwise_eci_lci_and_lfpa() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x = vector6_from_array([1e8, -2e8, 5e7, 1.0e3, -2.0e3, 0.5e3]);

        let via_router =
            state_frame_to_frame(ReferenceFrame::GCRF, ReferenceFrame::LCI, epc, x).unwrap();
        let pairwise = crate::frames::state_eci_to_lci(epc, x);
        for i in 0..6 {
            assert_abs_diff_eq!(via_router[i], pairwise[i], epsilon = 1e-9);
        }

        let x_lci = via_router;
        let via_router_lfpa =
            state_frame_to_frame(ReferenceFrame::LCI, ReferenceFrame::LFPA, epc, x_lci).unwrap();
        let pairwise_lfpa = crate::frames::state_lci_to_lfpa(epc, x_lci);
        for i in 0..6 {
            assert_eq!(via_router_lfpa[i], pairwise_lfpa[i]); // same-center, bit-identical
        }

        let via_router_composed =
            state_frame_to_frame(ReferenceFrame::GCRF, ReferenceFrame::LFPA, epc, x).unwrap();
        let composed =
            crate::frames::state_lci_to_lfpa(epc, crate::frames::state_eci_to_lci(epc, x));
        for i in 0..6 {
            assert_abs_diff_eq!(via_router_composed[i], composed[i], epsilon = 1e-9);
        }
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial]
    fn test_router_roundtrip_all_pairs() {
        setup_global_test_eop();
        setup_global_test_spice();
        let frames = [
            ReferenceFrame::GCRF,
            ReferenceFrame::ITRF,
            ReferenceFrame::EME2000,
            ReferenceFrame::LCI,
            ReferenceFrame::LFPA,
            ReferenceFrame::LFME,
            ReferenceFrame::MCI,
            ReferenceFrame::MCMF,
            ReferenceFrame::EMBI,
            ReferenceFrame::SSBI,
        ];
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x = vector6_from_array([1e8, -2e8, 5e7, 1.0e3, -2.0e3, 0.5e3]);
        // Position tolerance is looser than a same-magnitude round trip would
        // suggest because some pairs (e.g. MCI <-> EME2000) compose the
        // approximate (not exactly orthogonal to machine precision:
        // `B^T*B - I` ~ 4e-15) `bias_eme2000` rotation with the ~3.3e11 m
        // Earth-Mars SPK center offset; that tiny orthogonality residual,
        // amplified by the offset magnitude, is a ~1e-3 m floating-point
        // noise floor inherent to `bias_eme2000` (already reviewed,
        // out of scope here) and unrelated to the router's own logic.
        for &a in &frames {
            for &b in &frames {
                let there = state_frame_to_frame(a, b, epc, x).unwrap();
                let back = state_frame_to_frame(b, a, epc, there).unwrap();
                for i in 0..3 {
                    assert_abs_diff_eq!(back[i], x[i], epsilon = 1e-2);
                }
                for i in 3..6 {
                    assert_abs_diff_eq!(back[i], x[i], epsilon = 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_reference_frame_from_str_aliases() {
        assert_eq!(
            "ECI".parse::<ReferenceFrame>().unwrap(),
            ReferenceFrame::GCRF
        );
        assert_eq!(
            "ECEF".parse::<ReferenceFrame>().unwrap(),
            ReferenceFrame::ITRF
        );
        assert_eq!(
            "LFPA".parse::<ReferenceFrame>().unwrap(),
            ReferenceFrame::LFPA
        );
        assert_eq!(
            "eci".parse::<ReferenceFrame>().unwrap(),
            ReferenceFrame::GCRF
        );
        assert!("bogus".parse::<ReferenceFrame>().is_err());
    }

    #[test]
    fn test_generic_variants_equal_named() {
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let a = rotation_frame_to_frame(ReferenceFrame::MCI, ReferenceFrame::MCMF, epc).unwrap();
        let b = rotation_frame_to_frame(
            ReferenceFrame::BodyCenteredICRF(4),
            ReferenceFrame::BodyFixedIAU(499),
            epc,
        )
        .unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(a[(i, j)], b[(i, j)]);
            }
        }
    }

    #[test]
    fn test_reference_frame_display() {
        assert_eq!(ReferenceFrame::GCRF.to_string(), "GCRF");
        assert_eq!(ReferenceFrame::LFPA.to_string(), "LFPA");
        assert_eq!(
            ReferenceFrame::BodyCenteredICRF(4).to_string(),
            "BodyCenteredICRF(4)"
        );
        assert_eq!(
            ReferenceFrame::BodyFixedIAU(599).to_string(),
            "BodyFixedIAU(599)"
        );
        assert_eq!(
            ReferenceFrame::BodyFixedPCK {
                center: 301,
                frame_id: 31008
            }
            .to_string(),
            "BodyFixedPCK(center=301, frame_id=31008)"
        );
    }

    #[test]
    fn test_reference_frame_display_from_str_roundtrip() {
        let frames = [
            ReferenceFrame::GCRF,
            ReferenceFrame::ITRF,
            ReferenceFrame::EME2000,
            ReferenceFrame::LCI,
            ReferenceFrame::LFPA,
            ReferenceFrame::LFME,
            ReferenceFrame::MCI,
            ReferenceFrame::MCMF,
            ReferenceFrame::EMBI,
            ReferenceFrame::SSBI,
        ];
        for f in frames {
            assert_eq!(f.to_string().parse::<ReferenceFrame>().unwrap(), f);
        }
    }

    #[test]
    fn test_reference_frame_center_naif_id() {
        assert_eq!(ReferenceFrame::GCRF.center_naif_id(), NAIF_EARTH);
        assert_eq!(ReferenceFrame::ITRF.center_naif_id(), NAIF_EARTH);
        assert_eq!(ReferenceFrame::EME2000.center_naif_id(), NAIF_EARTH);
        assert_eq!(ReferenceFrame::LCI.center_naif_id(), 301);
        assert_eq!(ReferenceFrame::LFPA.center_naif_id(), 301);
        assert_eq!(ReferenceFrame::LFME.center_naif_id(), 301);
        assert_eq!(ReferenceFrame::MCI.center_naif_id(), NAIF_MARS_BARYCENTER);
        assert_eq!(ReferenceFrame::MCMF.center_naif_id(), NAIF_MARS_BARYCENTER);
        assert_eq!(ReferenceFrame::EMBI.center_naif_id(), 3);
        assert_eq!(ReferenceFrame::SSBI.center_naif_id(), 0);
        assert_eq!(ReferenceFrame::BodyCenteredICRF(599).center_naif_id(), 599);
        assert_eq!(ReferenceFrame::BodyFixedIAU(499).center_naif_id(), 499);
        assert_eq!(
            ReferenceFrame::BodyFixedPCK {
                center: 301,
                frame_id: 31008
            }
            .center_naif_id(),
            301
        );
    }

    #[test]
    fn test_reference_frame_serde_roundtrip() {
        let frames = [
            ReferenceFrame::GCRF,
            ReferenceFrame::LFPA,
            ReferenceFrame::BodyCenteredICRF(4),
            ReferenceFrame::BodyFixedIAU(499),
            ReferenceFrame::BodyFixedPCK {
                center: 301,
                frame_id: 31008,
            },
        ];
        for f in frames {
            let json = serde_json::to_string(&f).unwrap();
            let back: ReferenceFrame = serde_json::from_str(&json).unwrap();
            assert_eq!(back, f);
        }
    }

    #[test]
    fn test_position_frame_to_frame_same_frame_is_identity() {
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x = Vector3::new(1.0, 2.0, 3.0);
        let out =
            position_frame_to_frame(ReferenceFrame::GCRF, ReferenceFrame::GCRF, epc, x).unwrap();
        assert_eq!(out, x);
    }

    #[test]
    fn test_position_frame_to_frame_matches_pairwise_gcrf_itrf() {
        setup_global_test_eop();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x = Vector3::new(R_EARTH + 500e3, 1e5, 2e5);
        let via_router =
            position_frame_to_frame(ReferenceFrame::GCRF, ReferenceFrame::ITRF, epc, x).unwrap();
        let pairwise = crate::frames::position_gcrf_to_itrf(epc, x);
        for i in 0..3 {
            assert_eq!(via_router[i], pairwise[i]);
        }
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial]
    fn test_body_fixed_iau_translation_surfaces_spk_error() {
        // BodyFixedIAU(499) is centered on Mars itself (NAIF 499), which
        // has no direct SPK segment in the bundled de440s ephemeris (only
        // the Mars system barycenter, NAIF 4, does) -- translating into
        // it from a differently-centered frame must surface that error
        // rather than silently substituting the barycenter.
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x = vector6_from_array([1e8, -2e8, 5e7, 1.0e3, -2.0e3, 0.5e3]);
        let result = state_frame_to_frame(
            ReferenceFrame::GCRF,
            ReferenceFrame::BodyFixedIAU(499),
            epc,
            x,
        );
        assert!(result.is_err());
    }
}
