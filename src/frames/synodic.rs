/*!
 * Synodic (two-body rotating) reference frames: the Earth-Moon Rotating
 * (EMR), Sun-Earth Rotating (SER), and Geocentric Solar Ecliptic (GSE)
 * frames defined in NASA TP-20220014814 §2.5, with the exact
 * (GTDS/STK-convention) rotation-matrix time derivative of §4.6.1 —
 * including the dẑ/dt term, evaluated from native SPK acceleration —
 * rather than the GMAT dẑ/dt ≈ 0 approximation.
 */

use nalgebra::Vector3;

use crate::constants::{
    GM_EARTH, GM_JUPITER, GM_JUPITER_SYSTEM, GM_MARS, GM_MARS_SYSTEM, GM_MERCURY, GM_MOON,
    GM_NEPTUNE, GM_NEPTUNE_SYSTEM, GM_PLUTO, GM_PLUTO_SYSTEM, GM_SATURN, GM_SATURN_SYSTEM, GM_SUN,
    GM_URANUS, GM_URANUS_SYSTEM, GM_VENUS,
};
use crate::math::{SMatrix3, SVector6};
use crate::spice::{NAIFId, spk_acceleration, spk_position, spk_state};
use crate::time::Epoch;
use crate::utils::BraheError;
use crate::utils::batch::{try_batch_map, try_batch_map_epochs};

/// Computes the inertial→synodic rotation matrix `R` and its exact time
/// derivative `Ṙ` from the relative state of the two primaries
/// (NASA TP-20220014814 Eq. 66/69).
///
/// Axes: x̂ = r₁₂/‖r₁₂‖, ẑ = (r₁₂×v₁₂)/‖r₁₂×v₁₂‖, ŷ = ẑ×x̂; `R` has rows
/// x̂ᵀ, ŷᵀ, ẑᵀ. The derivative rows use dẑ/dt evaluated from `a12` (exact
/// GTDS/STK convention).
///
/// # Arguments
/// - `r12`: Position of the secondary relative to the primary, inertial axes. Units: [m]
/// - `v12`: Velocity of the secondary relative to the primary. Units: [m/s]
/// - `a12`: Acceleration of the secondary relative to the primary. Units: [m/s²]
///
/// # Returns
/// - `(R, Ṙ)`: Rotation matrix from inertial axes to synodic axes, and its
///   time derivative. Units: [-], [1/s]
///
/// # Errors
/// Returns [`BraheError::Error`] if `r12`, `v12`, or `a12` is non-finite,
/// if `r12` is zero/negligibly small (undefined x̂), or if `r12` and `v12`
/// are zero/negligibly non-collinear (undefined ẑ, i.e. `r12 × v12 ≈ 0`).
/// The angular-momentum check is relative-scale
/// (`‖r₁₂×v₁₂‖ <= f64::EPSILON * ‖r₁₂‖ * ‖v₁₂‖`, i.e. comparing against
/// the maximum possible cross-product magnitude) rather than an absolute
/// threshold, so legitimate small-scale two-body systems are not
/// incorrectly rejected.
///
/// # Examples
/// ```ignore
/// // Crate-internal: circular orbit in the xy-plane => R = I and Ṙ is the
/// // instantaneous rotation rate about ẑ.
/// let (r_mat, r_dot_mat) = synodic_axes(r12, v12, a12).unwrap();
/// ```
pub(crate) fn synodic_axes(
    r12: Vector3<f64>,
    v12: Vector3<f64>,
    a12: Vector3<f64>,
) -> Result<(SMatrix3, SMatrix3), BraheError> {
    if !r12.iter().all(|c| c.is_finite())
        || !v12.iter().all(|c| c.is_finite())
        || !a12.iter().all(|c| c.is_finite())
    {
        return Err(BraheError::Error(
            "synodic_axes: r12, v12, and a12 must all be finite".to_string(),
        ));
    }

    let r_norm = r12.norm();
    if r_norm <= f64::EPSILON * r12.amax() {
        return Err(BraheError::Error(
            "synodic_axes: r12 is zero/negligibly small — x̂ is undefined".to_string(),
        ));
    }
    let x_hat = r12 / r_norm;

    let h = r12.cross(&v12);
    let h_norm = h.norm();
    if h_norm <= f64::EPSILON * r_norm * v12.norm() {
        return Err(BraheError::Error(
            "synodic_axes: r12 and v12 are collinear (r12 x v12 ~ 0) — ẑ is undefined".to_string(),
        ));
    }
    let z_hat = h / h_norm;

    let y_hat = z_hat.cross(&x_hat);

    // TP Eq. 69: dx̂/dt = (v₁₂ - x̂(x̂·v₁₂))/‖r₁₂‖ — the component of the
    // relative velocity perpendicular to x̂, divided by the separation.
    let x_hat_dot = (v12 - x_hat * x_hat.dot(&v12)) / r_norm;

    // dh/dt = d/dt(r₁₂×v₁₂) = v₁₂×v₁₂ + r₁₂×a₁₂ = r₁₂×a₁₂; dẑ/dt is its
    // component perpendicular to ẑ, divided by ‖h‖ (exact GTDS/STK form).
    let h_dot = r12.cross(&a12);
    let z_hat_dot = (h_dot - z_hat * z_hat.dot(&h_dot)) / h_norm;

    let y_hat_dot = z_hat_dot.cross(&x_hat) + z_hat.cross(&x_hat_dot);

    let r_mat = SMatrix3::from_rows(&[x_hat.transpose(), y_hat.transpose(), z_hat.transpose()]);
    let r_dot_mat = SMatrix3::from_rows(&[
        x_hat_dot.transpose(),
        y_hat_dot.transpose(),
        z_hat_dot.transpose(),
    ]);
    Ok((r_mat, r_dot_mat))
}

/// Transforms an inertial-axis state (already translated to the synodic
/// frame's origin) into synodic axes: `r_s = R r`, `v_s = R v + Ṙ r`
/// (NASA TP-20220014814 Eq. 67/70, translation handled by the caller).
///
/// # Arguments
/// - `r_mat`: Inertial→synodic rotation matrix from [`synodic_axes`]
/// - `r_dot_mat`: Its time derivative. Units: [1/s]
/// - `x`: Cartesian state (position, velocity), inertial axes. Units: [m; m/s]
///
/// # Returns
/// - Cartesian state in synodic axes. Units: [m; m/s]
pub(crate) fn state_inertial_to_synodic(
    r_mat: &SMatrix3,
    r_dot_mat: &SMatrix3,
    x: SVector6,
) -> SVector6 {
    let r = x.fixed_rows::<3>(0).into_owned();
    let v = x.fixed_rows::<3>(3).into_owned();
    let r_s: Vector3<f64> = r_mat * r;
    let v_s: Vector3<f64> = r_mat * v + r_dot_mat * r;
    SVector6::new(r_s[0], r_s[1], r_s[2], v_s[0], v_s[1], v_s[2])
}

/// Inverse of [`state_inertial_to_synodic`]: `r = Rᵀ r_s`,
/// `v = Rᵀ v_s + Ṙᵀ r_s` (using Ṙᵀ = −RᵀṘRᵀ from d/dt(RᵀR) = 0, which
/// reduces TP Eq. 68/71 to this form).
///
/// # Arguments
/// - `r_mat`: Inertial→synodic rotation matrix from [`synodic_axes`]
/// - `r_dot_mat`: Its time derivative. Units: [1/s]
/// - `x`: Cartesian state (position, velocity), synodic axes. Units: [m; m/s]
///
/// # Returns
/// - Cartesian state in inertial axes. Units: [m; m/s]
pub(crate) fn state_synodic_to_inertial(
    r_mat: &SMatrix3,
    r_dot_mat: &SMatrix3,
    x: SVector6,
) -> SVector6 {
    let r_s = x.fixed_rows::<3>(0).into_owned();
    let v_s = x.fixed_rows::<3>(3).into_owned();
    let r: Vector3<f64> = r_mat.transpose() * r_s;
    let v: Vector3<f64> = r_mat.transpose() * v_s + r_dot_mat.transpose() * r_s;
    SVector6::new(r[0], r[1], r[2], v[0], v[1], v[2])
}

/// Gravitational parameter for the NAIF IDs supported as generic synodic
/// frame primaries, from the crate's packaged constants.
///
/// # Arguments
/// - `naif_id`: NAIF ID of the body (planet, Moon, Sun, or planetary
///   barycenter alias)
///
/// # Returns
/// - `gm`: Gravitational parameter. Units: [m³/s²]
///
/// # Errors
/// Returns [`BraheError::Error`] for IDs without a packaged GM constant.
pub(crate) fn body_gm(naif_id: i32) -> Result<f64, BraheError> {
    match naif_id {
        10 => Ok(GM_SUN),
        1 | 199 => Ok(GM_MERCURY),
        2 | 299 => Ok(GM_VENUS),
        399 => Ok(GM_EARTH),
        301 => Ok(GM_MOON),
        3 => Ok(GM_EARTH + GM_MOON),
        4 => Ok(GM_MARS_SYSTEM),
        499 => Ok(GM_MARS),
        5 => Ok(GM_JUPITER_SYSTEM),
        599 => Ok(GM_JUPITER),
        6 => Ok(GM_SATURN_SYSTEM),
        699 => Ok(GM_SATURN),
        7 => Ok(GM_URANUS_SYSTEM),
        799 => Ok(GM_URANUS),
        8 => Ok(GM_NEPTUNE_SYSTEM),
        899 => Ok(GM_NEPTUNE),
        9 => Ok(GM_PLUTO_SYSTEM),
        999 => Ok(GM_PLUTO),
        id => Err(BraheError::Error(format!(
            "No packaged GM constant for NAIF ID {id}; synodic barycenter \
             origins are only supported for the Sun, planets, planetary \
             barycenters, and the Moon"
        ))),
    }
}

/// Synodic frame axes for an arbitrary primary/secondary pair at `epc`:
/// the inertial→synodic rotation matrix and its exact time derivative,
/// built from the secondary's SPK state and acceleration relative to the
/// primary (NASA TP-20220014814 §4.6.1). x̂ points primary→secondary.
/// Generic form of [`emr_axes`]/[`ser_axes`]/[`gse_axes`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `primary`: NAIF ID of the primary body
/// - `secondary`: NAIF ID of the secondary body
///
/// # Returns
/// - `(R, Ṙ)`: Rotation matrix from ICRF-aligned axes to synodic axes,
///   and its time derivative. Units: [-], [1/s]
pub(crate) fn generic_synodic_axes(
    epc: Epoch,
    primary: i32,
    secondary: i32,
) -> Result<(SMatrix3, SMatrix3), BraheError> {
    crate::spice::registry::ensure_bodies_loadable(&[primary, secondary])?;
    let x12 = spk_state(secondary, primary, epc)?;
    let a12 = spk_acceleration(secondary, primary, epc)?;
    synodic_axes(
        x12.fixed_rows::<3>(0).into_owned(),
        x12.fixed_rows::<3>(3).into_owned(),
        a12,
    )
}

/// SSB-relative state of the GM-weighted barycenter of a two-body pair
/// (ICRF axes). Generic form of [`sun_earth_barycenter_state`].
///
/// # Arguments
/// - `epc`: Epoch instant
/// - `primary`: NAIF ID of the primary body
/// - `secondary`: NAIF ID of the secondary body
///
/// # Returns
/// - Cartesian state of the pair barycenter relative to the SSB.
///   Units: [m; m/s]
pub(crate) fn pair_barycenter_state(
    epc: Epoch,
    primary: i32,
    secondary: i32,
) -> Result<SVector6, BraheError> {
    let gm1 = body_gm(primary)?;
    let gm2 = body_gm(secondary)?;
    crate::spice::registry::ensure_bodies_loadable(&[primary, secondary])?;
    let x1 = spk_state(primary, NAIFId::SolarSystemBarycenter, epc)?;
    let x2 = spk_state(secondary, NAIFId::SolarSystemBarycenter, epc)?;
    Ok((x1 * gm1 + x2 * gm2) / (gm1 + gm2))
}

/// EMR (Earth-Moon Rotating) frame axes at `epc`: the inertial→EMR
/// rotation matrix and its exact time derivative, built from the Moon's
/// SPK state and acceleration relative to Earth (NASA TP-20220014814
/// §2.5.1/§4.6.2).
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
///
/// # Returns
/// - `(R, Ṙ)`: Rotation matrix from GCRF to EMR axes, and its time
///   derivative. Units: [-], [1/s]
pub(crate) fn emr_axes(epc: Epoch) -> Result<(SMatrix3, SMatrix3), BraheError> {
    generic_synodic_axes(epc, NAIFId::Earth.id(), NAIFId::Moon.id())
}

/// Synodic rotation matrix and origin offset (ICRF axes, from Earth) for a
/// position transformation at one epoch.
struct SynodicPositionContext {
    r_mat: SMatrix3,
    offset: Vector3<f64>,
}

/// Synodic rotation matrix, its time derivative, and origin offset state
/// (ICRF axes, from Earth) for a state transformation at one epoch.
struct SynodicStateContext {
    r_mat: SMatrix3,
    r_dot_mat: SMatrix3,
    offset: SVector6,
}

/// Apply a synodic position context: re-center from Earth to the synodic
/// origin, then rotate into synodic axes.
fn apply_position_inertial_to_synodic(
    c: &SynodicPositionContext,
    x: &Vector3<f64>,
) -> Vector3<f64> {
    c.r_mat * (x - c.offset)
}

/// Apply a synodic position context in the inverse direction: rotate to ICRF
/// axes, then re-center from the synodic origin to Earth.
fn apply_position_synodic_to_inertial(
    c: &SynodicPositionContext,
    x: &Vector3<f64>,
) -> Vector3<f64> {
    c.r_mat.transpose() * x + c.offset
}

/// Apply a synodic state context: re-center from Earth to the synodic origin,
/// then rotate into synodic axes with the transport term.
fn apply_state_inertial_to_synodic(c: &SynodicStateContext, x: &SVector6) -> SVector6 {
    state_inertial_to_synodic(&c.r_mat, &c.r_dot_mat, x - c.offset)
}

/// Apply a synodic state context in the inverse direction.
fn apply_state_synodic_to_inertial(c: &SynodicStateContext, x: &SVector6) -> SVector6 {
    state_synodic_to_inertial(&c.r_mat, &c.r_dot_mat, *x) + c.offset
}

/// EMR position context: rotation from [`emr_axes`] and the Earth -> EMB
/// offset.
fn emr_position_context(epc: Epoch) -> Result<SynodicPositionContext, BraheError> {
    let (r_mat, _) = emr_axes(epc)?;
    let offset = spk_position(NAIFId::EarthMoonBarycenter, NAIFId::Earth, epc)?;
    Ok(SynodicPositionContext { r_mat, offset })
}

/// EMR state context: rotation and rate from [`emr_axes`] and the Earth ->
/// EMB offset state.
fn emr_state_context(epc: Epoch) -> Result<SynodicStateContext, BraheError> {
    let (r_mat, r_dot_mat) = emr_axes(epc)?;
    // EMB relative to Earth in ICRF axes: re-center Earth → EMB, then rotate.
    let offset = spk_state(NAIFId::EarthMoonBarycenter, NAIFId::Earth, epc)?;
    Ok(SynodicStateContext {
        r_mat,
        r_dot_mat,
        offset,
    })
}

/// Computes the rotation matrix from Geocentric Celestial Reference Frame
/// (GCRF) to Earth-Moon Rotating (EMR) axes.
///
/// EMR is the two-body rotating frame defined by the instantaneous
/// Earth-Moon geometry (NASA TP-20220014814 §2.5.1): x̂ points from Earth
/// to the Moon, ẑ is along the instantaneous orbit normal, and ŷ completes
/// the right-handed triad. The rotation matrix is built from the Moon's
/// SPK state and acceleration relative to Earth, using the exact
/// (GTDS/STK-convention) time derivative of §4.6.2.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
///
/// # Returns
/// - `r`: 3x3 Rotation matrix transforming GCRF -> EMR
///
/// # Examples
/// ```
/// use brahe::frames::rotation_gcrf_to_emr;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_gcrf_to_emr(epc).unwrap();
/// ```
pub fn rotation_gcrf_to_emr(epc: Epoch) -> Result<SMatrix3, BraheError> {
    Ok(emr_axes(epc)?.0)
}

/// Computes the rotation matrix from Earth-Moon Rotating (EMR) axes to
/// Geocentric Celestial Reference Frame (GCRF). Inverse of
/// [`rotation_gcrf_to_emr`].
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
///
/// # Returns
/// - `r`: 3x3 Rotation matrix transforming EMR -> GCRF
///
/// # Examples
/// ```
/// use brahe::frames::rotation_emr_to_gcrf;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_emr_to_gcrf(epc).unwrap();
/// ```
pub fn rotation_emr_to_gcrf(epc: Epoch) -> Result<SMatrix3, BraheError> {
    Ok(emr_axes(epc)?.0.transpose())
}

/// Transforms a Cartesian GCRF position into the equivalent Cartesian
/// Earth-Moon Rotating (EMR) position.
///
/// The EMR origin is the Earth-Moon Barycenter (NASA TP-20220014814
/// §2.5.1); the input is re-centered from Earth to the barycenter before
/// rotating into EMR axes.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_position`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_gcrf`: Cartesian GCRF position. Units: (*m*)
///
/// # Returns
/// - `x_emr`: Cartesian EMR position. Units: (*m*)
///
/// # Examples
/// ```
/// use brahe::frames::position_gcrf_to_emr;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = Vector3::new(1e7, 2e7, 3e7);
/// let x_emr = position_gcrf_to_emr(epc, x_gcrf).unwrap();
/// ```
pub fn position_gcrf_to_emr(epc: Epoch, x_gcrf: Vector3<f64>) -> Result<Vector3<f64>, BraheError> {
    Ok(apply_position_inertial_to_synodic(
        &emr_position_context(epc)?,
        &x_gcrf,
    ))
}

/// Transforms a Cartesian Earth-Moon Rotating (EMR) position into the
/// equivalent Cartesian GCRF position. Inverse of
/// [`position_gcrf_to_emr`].
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_position`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_emr`: Cartesian EMR position. Units: (*m*)
///
/// # Returns
/// - `x_gcrf`: Cartesian GCRF position. Units: (*m*)
///
/// # Examples
/// ```
/// use brahe::frames::{position_emr_to_gcrf, position_gcrf_to_emr};
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = Vector3::new(1e7, 2e7, 3e7);
/// let x_emr = position_gcrf_to_emr(epc, x_gcrf).unwrap();
///
/// // Convert back to GCRF
/// let x_gcrf2 = position_emr_to_gcrf(epc, x_emr).unwrap();
/// ```
pub fn position_emr_to_gcrf(epc: Epoch, x_emr: Vector3<f64>) -> Result<Vector3<f64>, BraheError> {
    Ok(apply_position_synodic_to_inertial(
        &emr_position_context(epc)?,
        &x_emr,
    ))
}

/// Transforms a Cartesian GCRF state (position and velocity) into the
/// equivalent Cartesian Earth-Moon Rotating (EMR) state.
///
/// The EMR origin is the Earth-Moon Barycenter (NASA TP-20220014814
/// §2.5.1); the input is re-centered from Earth to the barycenter, then
/// rotated into EMR axes using the exact (GTDS/STK-convention) rotation
/// rate of §4.6.2 (including the transport term from `Ṙ`).
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_gcrf`: Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_emr`: Cartesian EMR state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples
/// ```
/// use brahe::frames::state_gcrf_to_emr;
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
/// let x_emr = state_gcrf_to_emr(epc, x_gcrf).unwrap();
/// ```
pub fn state_gcrf_to_emr(epc: Epoch, x_gcrf: SVector6) -> Result<SVector6, BraheError> {
    Ok(apply_state_inertial_to_synodic(
        &emr_state_context(epc)?,
        &x_gcrf,
    ))
}

/// Transforms a Cartesian Earth-Moon Rotating (EMR) state (position and
/// velocity) into the equivalent Cartesian GCRF state. Inverse of
/// [`state_gcrf_to_emr`].
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_emr`: Cartesian EMR state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_gcrf`: Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples
/// ```
/// use brahe::frames::{state_emr_to_gcrf, state_gcrf_to_emr};
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
/// let x_emr = state_gcrf_to_emr(epc, x_gcrf).unwrap();
///
/// // Convert back to GCRF
/// let x_gcrf2 = state_emr_to_gcrf(epc, x_emr).unwrap();
/// ```
pub fn state_emr_to_gcrf(epc: Epoch, x_emr: SVector6) -> Result<SVector6, BraheError> {
    Ok(apply_state_synodic_to_inertial(
        &emr_state_context(epc)?,
        &x_emr,
    ))
}

/// State of the Sun-Earth barycenter (SEB) relative to the Solar System
/// Barycenter, in ICRF axes, computed as the GM-weighted combination of
/// the Sun and Earth SPK states (NASA TP-20220014814 §2.5.3). The SEB is
/// a derived quantity with no NAIF ID or SPK segment of its own.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the barycenter state
///
/// # Returns
/// - `x_seb`: Cartesian SEB state (position, velocity) relative to the
///   Solar System Barycenter, ICRF axes. Units: (*m*; *m/s*)
///
/// Delegates to [`pair_barycenter_state`], the generic form of this
/// computation.
pub(crate) fn sun_earth_barycenter_state(epc: Epoch) -> Result<SVector6, BraheError> {
    pair_barycenter_state(epc, NAIFId::Sun.id(), NAIFId::Earth.id())
}

/// SER (Sun-Earth Rotating) frame axes at `epc`: the inertial→SER rotation
/// matrix and its exact time derivative, built from the Earth's SPK state
/// and acceleration relative to the Sun (NASA TP-20220014814
/// §2.5.3/§4.6.4). x̂ points Sun→Earth.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
///
/// # Returns
/// - `(R, Ṙ)`: Rotation matrix from GCRF to SER axes, and its time
///   derivative. Units: [-], [1/s]
pub(crate) fn ser_axes(epc: Epoch) -> Result<(SMatrix3, SMatrix3), BraheError> {
    generic_synodic_axes(epc, NAIFId::Sun.id(), NAIFId::Earth.id())
}

/// GSE (Geocentric Solar Ecliptic) frame axes at `epc`: the inertial→GSE
/// rotation matrix and its exact time derivative, built from the Sun's
/// SPK state and acceleration relative to Earth (NASA TP-20220014814
/// §2.5.4/§4.6.5). x̂ points Earth→Sun — the reversed sense relative to
/// SER.
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
///
/// # Returns
/// - `(R, Ṙ)`: Rotation matrix from GCRF to GSE axes, and its time
///   derivative. Units: [-], [1/s]
pub(crate) fn gse_axes(epc: Epoch) -> Result<(SMatrix3, SMatrix3), BraheError> {
    generic_synodic_axes(epc, NAIFId::Earth.id(), NAIFId::Sun.id())
}

/// Earth→SEB origin offset in ICRF axes (SEB state minus Earth state,
/// both relative to the SSB).
fn seb_offset_from_earth(epc: Epoch) -> Result<SVector6, BraheError> {
    let seb = sun_earth_barycenter_state(epc)?;
    let earth = spk_state(NAIFId::Earth, NAIFId::SolarSystemBarycenter, epc)?;
    Ok(seb - earth)
}

/// SER position context: rotation from [`ser_axes`] and the Earth -> SEB
/// offset.
fn ser_position_context(epc: Epoch) -> Result<SynodicPositionContext, BraheError> {
    let (r_mat, _) = ser_axes(epc)?;
    let offset = seb_offset_from_earth(epc)?.fixed_rows::<3>(0).into_owned();
    Ok(SynodicPositionContext { r_mat, offset })
}

/// SER state context: rotation and rate from [`ser_axes`] and the Earth ->
/// SEB offset state.
fn ser_state_context(epc: Epoch) -> Result<SynodicStateContext, BraheError> {
    let (r_mat, r_dot_mat) = ser_axes(epc)?;
    // SEB relative to Earth in ICRF axes: re-center Earth → SEB, then rotate.
    let offset = seb_offset_from_earth(epc)?;
    Ok(SynodicStateContext {
        r_mat,
        r_dot_mat,
        offset,
    })
}

/// Computes the rotation matrix from Geocentric Celestial Reference Frame
/// (GCRF) to Sun-Earth Rotating (SER) axes.
///
/// SER is the two-body rotating frame defined by the instantaneous
/// Sun-Earth geometry (NASA TP-20220014814 §2.5.3): x̂ points from the Sun
/// to Earth, ẑ is along the instantaneous orbit normal, and ŷ completes
/// the right-handed triad. The rotation matrix is built from the Earth's
/// SPK state and acceleration relative to the Sun, using the exact
/// (GTDS/STK-convention) time derivative of §4.6.4.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
///
/// # Returns
/// - `r`: 3x3 Rotation matrix transforming GCRF -> SER
///
/// # Examples
/// ```
/// use brahe::frames::rotation_gcrf_to_ser;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_gcrf_to_ser(epc).unwrap();
/// ```
pub fn rotation_gcrf_to_ser(epc: Epoch) -> Result<SMatrix3, BraheError> {
    Ok(ser_axes(epc)?.0)
}

/// Computes the rotation matrix from Sun-Earth Rotating (SER) axes to
/// Geocentric Celestial Reference Frame (GCRF). Inverse of
/// [`rotation_gcrf_to_ser`].
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
///
/// # Returns
/// - `r`: 3x3 Rotation matrix transforming SER -> GCRF
///
/// # Examples
/// ```
/// use brahe::frames::rotation_ser_to_gcrf;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_ser_to_gcrf(epc).unwrap();
/// ```
pub fn rotation_ser_to_gcrf(epc: Epoch) -> Result<SMatrix3, BraheError> {
    Ok(ser_axes(epc)?.0.transpose())
}

/// Transforms a Cartesian GCRF position into the equivalent Cartesian
/// Sun-Earth Rotating (SER) position.
///
/// The SER origin is the (true, GM-weighted) Sun-Earth Barycenter (NASA
/// TP-20220014814 §2.5.3); the input is re-centered from Earth to the SEB
/// before rotating into SER axes.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_gcrf`: Cartesian GCRF position. Units: (*m*)
///
/// # Returns
/// - `x_ser`: Cartesian SER position. Units: (*m*)
///
/// # Examples
/// ```
/// use brahe::frames::position_gcrf_to_ser;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = Vector3::new(1e7, 2e7, 3e7);
/// let x_ser = position_gcrf_to_ser(epc, x_gcrf).unwrap();
/// ```
pub fn position_gcrf_to_ser(epc: Epoch, x_gcrf: Vector3<f64>) -> Result<Vector3<f64>, BraheError> {
    Ok(apply_position_inertial_to_synodic(
        &ser_position_context(epc)?,
        &x_gcrf,
    ))
}

/// Transforms a Cartesian Sun-Earth Rotating (SER) position into the
/// equivalent Cartesian GCRF position. Inverse of
/// [`position_gcrf_to_ser`].
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_ser`: Cartesian SER position. Units: (*m*)
///
/// # Returns
/// - `x_gcrf`: Cartesian GCRF position. Units: (*m*)
///
/// # Examples
/// ```
/// use brahe::frames::{position_gcrf_to_ser, position_ser_to_gcrf};
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = Vector3::new(1e7, 2e7, 3e7);
/// let x_ser = position_gcrf_to_ser(epc, x_gcrf).unwrap();
///
/// // Convert back to GCRF
/// let x_gcrf2 = position_ser_to_gcrf(epc, x_ser).unwrap();
/// ```
pub fn position_ser_to_gcrf(epc: Epoch, x_ser: Vector3<f64>) -> Result<Vector3<f64>, BraheError> {
    Ok(apply_position_synodic_to_inertial(
        &ser_position_context(epc)?,
        &x_ser,
    ))
}

/// Transforms a Cartesian GCRF state (position and velocity) into the
/// equivalent Cartesian Sun-Earth Rotating (SER) state.
///
/// The SER origin is the (true, GM-weighted) Sun-Earth Barycenter (NASA
/// TP-20220014814 §2.5.3); the input is re-centered from Earth to the
/// SEB, then rotated into SER axes using the exact (GTDS/STK-convention)
/// rotation rate of §4.6.4 (including the transport term from `Ṙ`).
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_gcrf`: Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_ser`: Cartesian SER state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples
/// ```
/// use brahe::frames::state_gcrf_to_ser;
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
/// let x_ser = state_gcrf_to_ser(epc, x_gcrf).unwrap();
/// ```
pub fn state_gcrf_to_ser(epc: Epoch, x_gcrf: SVector6) -> Result<SVector6, BraheError> {
    Ok(apply_state_inertial_to_synodic(
        &ser_state_context(epc)?,
        &x_gcrf,
    ))
}

/// Transforms a Cartesian Sun-Earth Rotating (SER) state (position and
/// velocity) into the equivalent Cartesian GCRF state. Inverse of
/// [`state_gcrf_to_ser`].
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_ser`: Cartesian SER state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_gcrf`: Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples
/// ```
/// use brahe::frames::{state_gcrf_to_ser, state_ser_to_gcrf};
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
/// let x_ser = state_gcrf_to_ser(epc, x_gcrf).unwrap();
///
/// // Convert back to GCRF
/// let x_gcrf2 = state_ser_to_gcrf(epc, x_ser).unwrap();
/// ```
pub fn state_ser_to_gcrf(epc: Epoch, x_ser: SVector6) -> Result<SVector6, BraheError> {
    Ok(apply_state_synodic_to_inertial(
        &ser_state_context(epc)?,
        &x_ser,
    ))
}

/// Computes the rotation matrix from Geocentric Celestial Reference Frame
/// (GCRF) to Geocentric Solar Ecliptic (GSE) axes.
///
/// GSE is the two-body rotating frame defined by the instantaneous
/// Earth-Sun geometry (NASA TP-20220014814 §2.5.4): x̂ points from Earth
/// to the Sun — the reversed sense relative to SER — ẑ is along the
/// instantaneous orbit normal (near the ecliptic pole), and ŷ completes
/// the right-handed triad. The rotation matrix is built from the Sun's
/// SPK state and acceleration relative to Earth, using the exact
/// (GTDS/STK-convention) time derivative of §4.6.5.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
///
/// # Returns
/// - `r`: 3x3 Rotation matrix transforming GCRF -> GSE
///
/// # Examples
/// ```
/// use brahe::frames::rotation_gcrf_to_gse;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_gcrf_to_gse(epc).unwrap();
/// ```
pub fn rotation_gcrf_to_gse(epc: Epoch) -> Result<SMatrix3, BraheError> {
    Ok(gse_axes(epc)?.0)
}

/// Computes the rotation matrix from Geocentric Solar Ecliptic (GSE) axes
/// to Geocentric Celestial Reference Frame (GCRF). Inverse of
/// [`rotation_gcrf_to_gse`].
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
///
/// # Returns
/// - `r`: 3x3 Rotation matrix transforming GSE -> GCRF
///
/// # Examples
/// ```
/// use brahe::frames::rotation_gse_to_gcrf;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let r = rotation_gse_to_gcrf(epc).unwrap();
/// ```
pub fn rotation_gse_to_gcrf(epc: Epoch) -> Result<SMatrix3, BraheError> {
    Ok(gse_axes(epc)?.0.transpose())
}

/// Transforms a Cartesian GCRF position into the equivalent Cartesian
/// Geocentric Solar Ecliptic (GSE) position.
///
/// GSE is Earth-centered (NASA TP-20220014814 §2.5.4); unlike EMR and
/// SER, no translation is applied — only the rotation into GSE axes.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_gcrf`: Cartesian GCRF position. Units: (*m*)
///
/// # Returns
/// - `x_gse`: Cartesian GSE position. Units: (*m*)
///
/// # Examples
/// ```
/// use brahe::frames::position_gcrf_to_gse;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = Vector3::new(1e7, 2e7, 3e7);
/// let x_gse = position_gcrf_to_gse(epc, x_gcrf).unwrap();
/// ```
pub fn position_gcrf_to_gse(epc: Epoch, x_gcrf: Vector3<f64>) -> Result<Vector3<f64>, BraheError> {
    Ok(gse_axes(epc)?.0 * x_gcrf)
}

/// Transforms a Cartesian Geocentric Solar Ecliptic (GSE) position into
/// the equivalent Cartesian GCRF position. Inverse of
/// [`position_gcrf_to_gse`].
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_gse`: Cartesian GSE position. Units: (*m*)
///
/// # Returns
/// - `x_gcrf`: Cartesian GCRF position. Units: (*m*)
///
/// # Examples
/// ```
/// use brahe::frames::{position_gcrf_to_gse, position_gse_to_gcrf};
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = Vector3::new(1e7, 2e7, 3e7);
/// let x_gse = position_gcrf_to_gse(epc, x_gcrf).unwrap();
///
/// // Convert back to GCRF
/// let x_gcrf2 = position_gse_to_gcrf(epc, x_gse).unwrap();
/// ```
pub fn position_gse_to_gcrf(epc: Epoch, x_gse: Vector3<f64>) -> Result<Vector3<f64>, BraheError> {
    Ok(gse_axes(epc)?.0.transpose() * x_gse)
}

/// Transforms a Cartesian GCRF state (position and velocity) into the
/// equivalent Cartesian Geocentric Solar Ecliptic (GSE) state.
///
/// GSE is Earth-centered (NASA TP-20220014814 §2.5.4); unlike EMR and
/// SER, no translation is applied — only the rotation into GSE axes,
/// using the exact (GTDS/STK-convention) rotation rate of §4.6.5
/// (including the transport term from `Ṙ`).
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_gcrf`: Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_gse`: Cartesian GSE state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples
/// ```
/// use brahe::frames::state_gcrf_to_gse;
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
/// let x_gse = state_gcrf_to_gse(epc, x_gcrf).unwrap();
/// ```
pub fn state_gcrf_to_gse(epc: Epoch, x_gcrf: SVector6) -> Result<SVector6, BraheError> {
    let (r_mat, r_dot_mat) = gse_axes(epc)?;
    Ok(state_inertial_to_synodic(&r_mat, &r_dot_mat, x_gcrf))
}

/// Transforms a Cartesian Geocentric Solar Ecliptic (GSE) state (position
/// and velocity) into the equivalent Cartesian GCRF state. Inverse of
/// [`state_gcrf_to_gse`].
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded; see [`crate::spice::spk_state`].
///
/// # Arguments
/// - `epc`: Epoch instant for computation of the transformation
/// - `x_gse`: Cartesian GSE state (position, velocity). Units: (*m*; *m/s*)
///
/// # Returns
/// - `x_gcrf`: Cartesian GCRF state (position, velocity). Units: (*m*; *m/s*)
///
/// # Examples
/// ```
/// use brahe::frames::{state_gcrf_to_gse, state_gse_to_gcrf};
/// use brahe::math::vector6_from_array;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_gcrf = vector6_from_array([1e7, 2e7, 3e7, 1.0, 2.0, 3.0]);
/// let x_gse = state_gcrf_to_gse(epc, x_gcrf).unwrap();
///
/// // Convert back to GCRF
/// let x_gcrf2 = state_gse_to_gcrf(epc, x_gse).unwrap();
/// ```
pub fn state_gse_to_gcrf(epc: Epoch, x_gse: SVector6) -> Result<SVector6, BraheError> {
    let (r_mat, r_dot_mat) = gse_axes(epc)?;
    Ok(state_synodic_to_inertial(&r_mat, &r_dot_mat, x_gse))
}

/// Computes the GCRF to Earth-Moon Rotating (EMR) rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_gcrf_to_emr`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming GCRF -> EMR, one per epoch, in input order
/// - Error if the ephemeris cannot be evaluated at any epoch
///
/// # Examples
/// ```
/// use brahe::frames::rotations_gcrf_to_emr;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 3600.0];
/// let r = rotations_gcrf_to_emr(&epochs).unwrap();
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_gcrf_to_emr(epochs: &[Epoch]) -> Result<Vec<SMatrix3>, BraheError> {
    try_batch_map(epochs, |epc| rotation_gcrf_to_emr(*epc))
}

/// Computes the Earth-Moon Rotating (EMR) to GCRF rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_emr_to_gcrf`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming EMR -> GCRF, one per epoch, in input order
/// - Error if the ephemeris cannot be evaluated at any epoch
///
/// # Examples
/// ```
/// use brahe::frames::rotations_emr_to_gcrf;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 3600.0];
/// let r = rotations_emr_to_gcrf(&epochs).unwrap();
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_emr_to_gcrf(epochs: &[Epoch]) -> Result<Vec<SMatrix3>, BraheError> {
    try_batch_map(epochs, |epc| rotation_emr_to_gcrf(*epc))
}

/// Computes the GCRF to Sun-Earth Rotating (SER) rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_gcrf_to_ser`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming GCRF -> SER, one per epoch, in input order
/// - Error if the ephemeris cannot be evaluated at any epoch
///
/// # Examples
/// ```
/// use brahe::frames::rotations_gcrf_to_ser;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 3600.0];
/// let r = rotations_gcrf_to_ser(&epochs).unwrap();
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_gcrf_to_ser(epochs: &[Epoch]) -> Result<Vec<SMatrix3>, BraheError> {
    try_batch_map(epochs, |epc| rotation_gcrf_to_ser(*epc))
}

/// Computes the Sun-Earth Rotating (SER) to GCRF rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_ser_to_gcrf`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming SER -> GCRF, one per epoch, in input order
/// - Error if the ephemeris cannot be evaluated at any epoch
///
/// # Examples
/// ```
/// use brahe::frames::rotations_ser_to_gcrf;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 3600.0];
/// let r = rotations_ser_to_gcrf(&epochs).unwrap();
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_ser_to_gcrf(epochs: &[Epoch]) -> Result<Vec<SMatrix3>, BraheError> {
    try_batch_map(epochs, |epc| rotation_ser_to_gcrf(*epc))
}

/// Computes the GCRF to Geocentric Solar Ecliptic (GSE) rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_gcrf_to_gse`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming GCRF -> GSE, one per epoch, in input order
/// - Error if the ephemeris cannot be evaluated at any epoch
///
/// # Examples
/// ```
/// use brahe::frames::rotations_gcrf_to_gse;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 3600.0];
/// let r = rotations_gcrf_to_gse(&epochs).unwrap();
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_gcrf_to_gse(epochs: &[Epoch]) -> Result<Vec<SMatrix3>, BraheError> {
    try_batch_map(epochs, |epc| rotation_gcrf_to_gse(*epc))
}

/// Computes the Geocentric Solar Ecliptic (GSE) to GCRF rotation matrix for each epoch in `epochs`.
///
/// Batch form of [`rotation_gse_to_gcrf`]. Evaluation runs on the global thread pool for
/// large inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants for computation of the transformation matrices
///
/// # Returns
/// - Rotation matrices transforming GSE -> GCRF, one per epoch, in input order
/// - Error if the ephemeris cannot be evaluated at any epoch
///
/// # Examples
/// ```
/// use brahe::frames::rotations_gse_to_gcrf;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 3600.0];
/// let r = rotations_gse_to_gcrf(&epochs).unwrap();
/// assert_eq!(r.len(), 2);
/// ```
pub fn rotations_gse_to_gcrf(epochs: &[Epoch]) -> Result<Vec<SMatrix3>, BraheError> {
    try_batch_map(epochs, |epc| rotation_gse_to_gcrf(*epc))
}

/// Transforms a batch of Cartesian positions from GCRF to Earth-Moon Rotating (EMR).
///
/// Batch form of [`position_gcrf_to_emr`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the synodic axes and origin offset once and applies them
/// to every element. Evaluation runs on the global thread pool for large
/// inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_gcrf`: Cartesian GCRF positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian EMR positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule or the ephemeris cannot be evaluated
///
/// # Examples
/// ```
/// use brahe::frames::positions_gcrf_to_emr;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(7.0e6, 1.0e6, -2.0e6); 3];
/// let out = positions_gcrf_to_emr(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_gcrf_to_emr(
    epochs: &[Epoch],
    x_gcrf: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    try_batch_map_epochs(epochs, x_gcrf, emr_position_context, |c, x| {
        Ok(apply_position_inertial_to_synodic(c, x))
    })
}

/// Transforms a batch of Cartesian positions from Earth-Moon Rotating (EMR) to GCRF.
///
/// Batch form of [`position_emr_to_gcrf`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the synodic axes and origin offset once and applies them
/// to every element. Evaluation runs on the global thread pool for large
/// inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_emr`: Cartesian EMR positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian GCRF positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule or the ephemeris cannot be evaluated
///
/// # Examples
/// ```
/// use brahe::frames::positions_emr_to_gcrf;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(7.0e6, 1.0e6, -2.0e6); 3];
/// let out = positions_emr_to_gcrf(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_emr_to_gcrf(
    epochs: &[Epoch],
    x_emr: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    try_batch_map_epochs(epochs, x_emr, emr_position_context, |c, x| {
        Ok(apply_position_synodic_to_inertial(c, x))
    })
}

/// Transforms a batch of Cartesian positions from GCRF to Sun-Earth Rotating (SER).
///
/// Batch form of [`position_gcrf_to_ser`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the synodic axes and origin offset once and applies them
/// to every element. Evaluation runs on the global thread pool for large
/// inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_gcrf`: Cartesian GCRF positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian SER positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule or the ephemeris cannot be evaluated
///
/// # Examples
/// ```
/// use brahe::frames::positions_gcrf_to_ser;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(7.0e6, 1.0e6, -2.0e6); 3];
/// let out = positions_gcrf_to_ser(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_gcrf_to_ser(
    epochs: &[Epoch],
    x_gcrf: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    try_batch_map_epochs(epochs, x_gcrf, ser_position_context, |c, x| {
        Ok(apply_position_inertial_to_synodic(c, x))
    })
}

/// Transforms a batch of Cartesian positions from Sun-Earth Rotating (SER) to GCRF.
///
/// Batch form of [`position_ser_to_gcrf`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the synodic axes and origin offset once and applies them
/// to every element. Evaluation runs on the global thread pool for large
/// inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_ser`: Cartesian SER positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian GCRF positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule or the ephemeris cannot be evaluated
///
/// # Examples
/// ```
/// use brahe::frames::positions_ser_to_gcrf;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(7.0e6, 1.0e6, -2.0e6); 3];
/// let out = positions_ser_to_gcrf(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_ser_to_gcrf(
    epochs: &[Epoch],
    x_ser: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    try_batch_map_epochs(epochs, x_ser, ser_position_context, |c, x| {
        Ok(apply_position_synodic_to_inertial(c, x))
    })
}

/// Transforms a batch of Cartesian positions from GCRF to Geocentric Solar Ecliptic (GSE).
///
/// Batch form of [`position_gcrf_to_gse`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the synodic axes and origin offset once and applies them
/// to every element. Evaluation runs on the global thread pool for large
/// inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_gcrf`: Cartesian GCRF positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian GSE positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule or the ephemeris cannot be evaluated
///
/// # Examples
/// ```
/// use brahe::frames::positions_gcrf_to_gse;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(7.0e6, 1.0e6, -2.0e6); 3];
/// let out = positions_gcrf_to_gse(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_gcrf_to_gse(
    epochs: &[Epoch],
    x_gcrf: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    try_batch_map_epochs(epochs, x_gcrf, gse_axes, |(r_mat, _), x| Ok(r_mat * x))
}

/// Transforms a batch of Cartesian positions from Geocentric Solar Ecliptic (GSE) to GCRF.
///
/// Batch form of [`position_gse_to_gcrf`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the synodic axes and origin offset once and applies them
/// to every element. Evaluation runs on the global thread pool for large
/// inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_gse`: Cartesian GSE positions, length 1 or the batch length. Units: (*m*)
///
/// # Returns
/// - Cartesian GCRF positions in input order. Units: (*m*)
/// - Error if the lengths do not satisfy the broadcast rule or the ephemeris cannot be evaluated
///
/// # Examples
/// ```
/// use brahe::frames::positions_gse_to_gcrf;
/// use brahe::time::{Epoch, TimeSystem};
/// use nalgebra::Vector3;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let positions = vec![Vector3::new(7.0e6, 1.0e6, -2.0e6); 3];
/// let out = positions_gse_to_gcrf(&[epc], &positions).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn positions_gse_to_gcrf(
    epochs: &[Epoch],
    x_gse: &[Vector3<f64>],
) -> Result<Vec<Vector3<f64>>, BraheError> {
    try_batch_map_epochs(epochs, x_gse, gse_axes, |(r_mat, _), x| {
        Ok(r_mat.transpose() * x)
    })
}

/// Transforms a batch of Cartesian states from GCRF to Earth-Moon Rotating (EMR).
///
/// Batch form of [`state_gcrf_to_emr`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the synodic axes and origin offset once and applies them
/// to every element. Evaluation runs on the global thread pool for large
/// inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_gcrf`: Cartesian GCRF states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian EMR states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule or the ephemeris cannot be evaluated
///
/// # Examples
/// ```
/// use brahe::frames::states_gcrf_to_emr;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 3600.0, epc + 7200.0];
/// let states = vec![vector6_from_array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]); 3];
/// let out = states_gcrf_to_emr(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_gcrf_to_emr(
    epochs: &[Epoch],
    x_gcrf: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    try_batch_map_epochs(epochs, x_gcrf, emr_state_context, |c, x| {
        Ok(apply_state_inertial_to_synodic(c, x))
    })
}

/// Transforms a batch of Cartesian states from Earth-Moon Rotating (EMR) to GCRF.
///
/// Batch form of [`state_emr_to_gcrf`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the synodic axes and origin offset once and applies them
/// to every element. Evaluation runs on the global thread pool for large
/// inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_emr`: Cartesian EMR states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian GCRF states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule or the ephemeris cannot be evaluated
///
/// # Examples
/// ```
/// use brahe::frames::states_emr_to_gcrf;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 3600.0, epc + 7200.0];
/// let states = vec![vector6_from_array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]); 3];
/// let out = states_emr_to_gcrf(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_emr_to_gcrf(
    epochs: &[Epoch],
    x_emr: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    try_batch_map_epochs(epochs, x_emr, emr_state_context, |c, x| {
        Ok(apply_state_synodic_to_inertial(c, x))
    })
}

/// Transforms a batch of Cartesian states from GCRF to Sun-Earth Rotating (SER).
///
/// Batch form of [`state_gcrf_to_ser`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the synodic axes and origin offset once and applies them
/// to every element. Evaluation runs on the global thread pool for large
/// inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_gcrf`: Cartesian GCRF states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian SER states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule or the ephemeris cannot be evaluated
///
/// # Examples
/// ```
/// use brahe::frames::states_gcrf_to_ser;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 3600.0, epc + 7200.0];
/// let states = vec![vector6_from_array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]); 3];
/// let out = states_gcrf_to_ser(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_gcrf_to_ser(
    epochs: &[Epoch],
    x_gcrf: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    try_batch_map_epochs(epochs, x_gcrf, ser_state_context, |c, x| {
        Ok(apply_state_inertial_to_synodic(c, x))
    })
}

/// Transforms a batch of Cartesian states from Sun-Earth Rotating (SER) to GCRF.
///
/// Batch form of [`state_ser_to_gcrf`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the synodic axes and origin offset once and applies them
/// to every element. Evaluation runs on the global thread pool for large
/// inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_ser`: Cartesian SER states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian GCRF states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule or the ephemeris cannot be evaluated
///
/// # Examples
/// ```
/// use brahe::frames::states_ser_to_gcrf;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 3600.0, epc + 7200.0];
/// let states = vec![vector6_from_array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]); 3];
/// let out = states_ser_to_gcrf(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_ser_to_gcrf(
    epochs: &[Epoch],
    x_ser: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    try_batch_map_epochs(epochs, x_ser, ser_state_context, |c, x| {
        Ok(apply_state_synodic_to_inertial(c, x))
    })
}

/// Transforms a batch of Cartesian states from GCRF to Geocentric Solar Ecliptic (GSE).
///
/// Batch form of [`state_gcrf_to_gse`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the synodic axes and origin offset once and applies them
/// to every element. Evaluation runs on the global thread pool for large
/// inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_gcrf`: Cartesian GCRF states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian GSE states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule or the ephemeris cannot be evaluated
///
/// # Examples
/// ```
/// use brahe::frames::states_gcrf_to_gse;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 3600.0, epc + 7200.0];
/// let states = vec![vector6_from_array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]); 3];
/// let out = states_gcrf_to_gse(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_gcrf_to_gse(
    epochs: &[Epoch],
    x_gcrf: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    try_batch_map_epochs(epochs, x_gcrf, gse_axes, |(r_mat, r_dot_mat), x| {
        Ok(state_inertial_to_synodic(r_mat, r_dot_mat, *x))
    })
}

/// Transforms a batch of Cartesian states from Geocentric Solar Ecliptic (GSE) to GCRF.
///
/// Batch form of [`state_gse_to_gcrf`]. `epochs` and the vector argument follow the
/// broadcast rule: each has length 1 or the common batch length. A single
/// epoch evaluates the synodic axes and origin offset once and applies them
/// to every element. Evaluation runs on the global thread pool for large
/// inputs.
///
/// # Arguments
/// - `epochs`: Epoch instants, length 1 or the batch length
/// - `x_gse`: Cartesian GSE states (position, velocity), length 1 or the batch length. Units: (*m*; *m/s*)
///
/// # Returns
/// - Cartesian GCRF states (position, velocity) in input order. Units: (*m*; *m/s*)
/// - Error if the lengths do not satisfy the broadcast rule or the ephemeris cannot be evaluated
///
/// # Examples
/// ```
/// use brahe::frames::states_gse_to_gcrf;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::vector6_from_array;
///
/// let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let epochs = vec![epc, epc + 3600.0, epc + 7200.0];
/// let states = vec![vector6_from_array([7.0e6, 0.0, 0.0, 0.0, 7.5e3, 0.0]); 3];
/// let out = states_gse_to_gcrf(&epochs, &states).unwrap();
/// assert_eq!(out.len(), 3);
/// ```
pub fn states_gse_to_gcrf(
    epochs: &[Epoch],
    x_gse: &[SVector6],
) -> Result<Vec<SVector6>, BraheError> {
    try_batch_map_epochs(epochs, x_gse, gse_axes, |(r_mat, r_dot_mat), x| {
        Ok(state_synodic_to_inertial(r_mat, r_dot_mat, *x))
    })
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use serial_test::{parallel, serial};

    use super::*;
    use crate::constants::{GM_EARTH, GM_MOON, GM_SUN};
    use crate::spice::{NAIFId, spk_state};
    use crate::time::TimeSystem;
    use crate::utils::testing::setup_global_test_spice;

    /// Analytic circular-orbit relative state in the xy-plane at phase
    /// `theta`: r = R(cosθ, sinθ, 0), v = RΩ(−sinθ, cosθ, 0), a = −Ω²r.
    fn circular_rel_state(
        radius: f64,
        omega: f64,
        theta: f64,
    ) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>) {
        let r = Vector3::new(radius * theta.cos(), radius * theta.sin(), 0.0);
        let v = Vector3::new(
            -radius * omega * theta.sin(),
            radius * omega * theta.cos(),
            0.0,
        );
        let a = -r * omega * omega;
        (r, v, a)
    }

    #[test]
    #[parallel]
    fn test_synodic_axes_circular_orbit() {
        let radius = 3.844e8;
        let omega = 2.66e-6;
        let (r, v, a) = circular_rel_state(radius, omega, 0.0);
        let (r_mat, r_dot_mat) = synodic_axes(r, v, a).unwrap();

        // At theta = 0 the synodic axes coincide with the inertial axes.
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    r_mat[(i, j)],
                    if i == j { 1.0 } else { 0.0 },
                    epsilon = 1e-14
                );
            }
        }
        // Ṙ is the instantaneous rotation about ẑ at rate omega.
        assert_abs_diff_eq!(r_dot_mat[(0, 1)], omega, epsilon = 1e-18);
        assert_abs_diff_eq!(r_dot_mat[(1, 0)], -omega, epsilon = 1e-18);
        assert_abs_diff_eq!(r_dot_mat[(2, 0)], 0.0, epsilon = 1e-18);
        assert_abs_diff_eq!(r_dot_mat[(2, 1)], 0.0, epsilon = 1e-18);
    }

    #[test]
    #[parallel]
    fn test_synodic_axes_orthonormal_and_skew() {
        // Generic inclined, non-circular input: orthonormality and the
        // rigid-rotation identity ṘRᵀ + RṘᵀ = 0 must hold regardless.
        let r = Vector3::new(2.5e8, -1.2e8, 0.9e8);
        let v = Vector3::new(300.0, 800.0, -150.0);
        let a = Vector3::new(-1.9e-3, 0.9e-3, -0.7e-3);
        let (r_mat, r_dot_mat) = synodic_axes(r, v, a).unwrap();

        let identity = r_mat * r_mat.transpose();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    identity[(i, j)],
                    if i == j { 1.0 } else { 0.0 },
                    epsilon = 1e-14
                );
            }
        }
        assert_abs_diff_eq!(r_mat.determinant(), 1.0, epsilon = 1e-14);

        let skew = r_dot_mat * r_mat.transpose() + r_mat * r_dot_mat.transpose();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(skew[(i, j)], 0.0, epsilon = 1e-18);
            }
        }
    }

    #[test]
    #[parallel]
    fn test_synodic_axes_derivative_matches_finite_difference() {
        let radius = 3.844e8;
        let omega = 2.66e-6;
        let theta = 0.7;
        let dt = 1.0;

        let (r, v, a) = circular_rel_state(radius, omega, theta);
        let (_, r_dot_mat) = synodic_axes(r, v, a).unwrap();

        let (rp, vp, ap) = circular_rel_state(radius, omega, theta + omega * dt);
        let (r_mat_p, _) = synodic_axes(rp, vp, ap).unwrap();
        let (rm, vm, am) = circular_rel_state(radius, omega, theta - omega * dt);
        let (r_mat_m, _) = synodic_axes(rm, vm, am).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    r_dot_mat[(i, j)],
                    (r_mat_p[(i, j)] - r_mat_m[(i, j)]) / (2.0 * dt),
                    epsilon = 1e-12
                );
            }
        }
    }

    #[test]
    #[parallel]
    fn test_synodic_axes_zero_separation_errs() {
        // r12 = 0 leaves x_hat = r12/||r12|| undefined.
        let r = Vector3::zeros();
        let v = Vector3::new(300.0, 800.0, -150.0);
        let a = Vector3::new(-1.9e-3, 0.9e-3, -0.7e-3);
        let err = synodic_axes(r, v, a).unwrap_err().to_string();
        assert!(err.contains("r12"), "error should name r12: {err}");
    }

    #[test]
    #[parallel]
    fn test_synodic_axes_collinear_errs() {
        // r12 parallel to v12 => r12 x v12 = 0 leaves z_hat undefined.
        let r = Vector3::new(1.0e8, 0.0, 0.0);
        let v = Vector3::new(2.0e3, 0.0, 0.0);
        let a = Vector3::new(-1.9e-3, 0.9e-3, -0.7e-3);
        let err = synodic_axes(r, v, a).unwrap_err().to_string();
        assert!(err.contains("v12"), "error should name v12: {err}");
    }

    #[test]
    #[parallel]
    fn test_state_transform_roundtrip() {
        let r = Vector3::new(2.5e8, -1.2e8, 0.9e8);
        let v = Vector3::new(300.0, 800.0, -150.0);
        let a = Vector3::new(-1.9e-3, 0.9e-3, -0.7e-3);
        let (r_mat, r_dot_mat) = synodic_axes(r, v, a).unwrap();

        let x = SVector6::new(1.0e8, -2.0e8, 5.0e7, 1.0e3, -2.0e3, 0.5e3);
        let x_syn = state_inertial_to_synodic(&r_mat, &r_dot_mat, x);
        let x_back = state_synodic_to_inertial(&r_mat, &r_dot_mat, x_syn);
        for i in 0..3 {
            assert_abs_diff_eq!(x_back[i], x[i], epsilon = 1e-6);
        }
        for i in 3..6 {
            assert_abs_diff_eq!(x_back[i], x[i], epsilon = 1e-10);
        }
    }

    #[test]
    #[serial] // global SPICE registry
    fn test_emr_moon_on_x_axis() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // The Moon's own state, expressed in EMR, must lie on +x̂ with zero
        // y/z position and zero y/z velocity (it defines the frame).
        let x_moon_gcrf = spk_state(NAIFId::Moon, NAIFId::Earth, epc).unwrap();
        let x_moon_emr = state_gcrf_to_emr(epc, x_moon_gcrf).unwrap();

        // Moon distance from EMB = d * GM_EARTH/(GM_EARTH+GM_MOON) ~ 0.9879 d.
        assert!(x_moon_emr[0] > 3.4e8 && x_moon_emr[0] < 4.1e8);
        assert_abs_diff_eq!(x_moon_emr[1], 0.0, epsilon = 1e-3);
        assert_abs_diff_eq!(x_moon_emr[2], 0.0, epsilon = 1e-3);
        assert_abs_diff_eq!(x_moon_emr[4], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(x_moon_emr[5], 0.0, epsilon = 1e-6);

        // Earth sits on −x̂ at the EMB offset (~4.7e6 m).
        let x_earth_emr = state_gcrf_to_emr(epc, SVector6::zeros()).unwrap();
        assert!(x_earth_emr[0] < -4.0e6 && x_earth_emr[0] > -5.5e6);
        assert_abs_diff_eq!(x_earth_emr[1], 0.0, epsilon = 1e-3);
        assert_abs_diff_eq!(x_earth_emr[2], 0.0, epsilon = 1e-3);
    }

    #[test]
    #[serial]
    fn test_emr_roundtrip() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x = SVector6::new(1.0e8, -2.0e8, 5.0e7, 1.0e3, -2.0e3, 0.5e3);
        let x_back = state_emr_to_gcrf(epc, state_gcrf_to_emr(epc, x).unwrap()).unwrap();
        for i in 0..3 {
            assert_abs_diff_eq!(x_back[i], x[i], epsilon = 1e-4);
        }
        for i in 3..6 {
            assert_abs_diff_eq!(x_back[i], x[i], epsilon = 1e-9);
        }

        let x3 = Vector3::new(1.0e8, -2.0e8, 5.0e7);
        let x3_back = position_emr_to_gcrf(epc, position_gcrf_to_emr(epc, x3).unwrap()).unwrap();
        for i in 0..3 {
            assert_abs_diff_eq!(x3_back[i], x3[i], epsilon = 1e-4);
        }
    }

    #[test]
    #[serial]
    fn test_rotation_gcrf_to_emr_derivative_consistency() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let dt = 10.0;

        let (_, r_dot_mat) = emr_axes(epc).unwrap();
        let r_mat_p = rotation_gcrf_to_emr(epc + dt).unwrap();
        let r_mat_m = rotation_gcrf_to_emr(epc + (-dt)).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(
                    r_dot_mat[(i, j)],
                    (r_mat_p[(i, j)] - r_mat_m[(i, j)]) / (2.0 * dt),
                    epsilon = 1e-11
                );
            }
        }
        // Proper rotation
        let r_mat = rotation_gcrf_to_emr(epc).unwrap();
        assert_abs_diff_eq!(r_mat.determinant(), 1.0, epsilon = 1e-12);
    }

    #[test]
    #[serial]
    fn test_ser_earth_position() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Earth (GCRF origin) in SER: on +x̂ at ~GM_SUN/(GM_SUN+GM_EARTH) of
        // the Sun-Earth distance from the SEB, with zero y/z.
        let x_earth_ser = state_gcrf_to_ser(epc, SVector6::zeros()).unwrap();
        let d = spk_state(NAIFId::Earth, NAIFId::Sun, epc)
            .unwrap()
            .fixed_rows::<3>(0)
            .norm();
        let expected_x = d * GM_SUN / (GM_SUN + GM_EARTH);
        assert_abs_diff_eq!(x_earth_ser[0], expected_x, epsilon = 1.0);
        assert_abs_diff_eq!(x_earth_ser[1], 0.0, epsilon = 1e-2);
        assert_abs_diff_eq!(x_earth_ser[2], 0.0, epsilon = 1e-2);

        // The Sun sits on −x̂ at the small SEB offset (~4.5e5 m).
        let x_sun_gcrf = spk_state(NAIFId::Sun, NAIFId::Earth, epc).unwrap();
        let x_sun_ser = state_gcrf_to_ser(epc, x_sun_gcrf).unwrap();
        assert!(x_sun_ser[0] < -3.0e5 && x_sun_ser[0] > -6.0e5);
        assert_abs_diff_eq!(x_sun_ser[1], 0.0, epsilon = 1e-2);
        assert_abs_diff_eq!(x_sun_ser[2], 0.0, epsilon = 1e-2);
    }

    #[test]
    #[serial]
    fn test_ser_roundtrip() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x = SVector6::new(1.0e8, -2.0e8, 5.0e7, 1.0e3, -2.0e3, 0.5e3);
        let x_back = state_ser_to_gcrf(epc, state_gcrf_to_ser(epc, x).unwrap()).unwrap();
        for i in 0..3 {
            assert_abs_diff_eq!(x_back[i], x[i], epsilon = 1e-2);
        }
        for i in 3..6 {
            assert_abs_diff_eq!(x_back[i], x[i], epsilon = 1e-7);
        }
    }

    #[test]
    #[serial]
    fn test_gse_sun_on_x_axis() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // GSE x̂ points Earth→Sun: the Sun lies on +x̂ with zero y/z and zero
        // transverse velocity; GSE is Earth-centered (no translation).
        let x_sun_gcrf = spk_state(NAIFId::Sun, NAIFId::Earth, epc).unwrap();
        let x_sun_gse = state_gcrf_to_gse(epc, x_sun_gcrf).unwrap();
        let d = x_sun_gcrf.fixed_rows::<3>(0).norm();
        assert_abs_diff_eq!(x_sun_gse[0], d, epsilon = 1e-3);
        assert_abs_diff_eq!(x_sun_gse[1], 0.0, epsilon = 1e-2);
        assert_abs_diff_eq!(x_sun_gse[2], 0.0, epsilon = 1e-2);
        assert_abs_diff_eq!(x_sun_gse[4], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(x_sun_gse[5], 0.0, epsilon = 1e-6);

        // Earth stays at the origin.
        let x_earth_gse = state_gcrf_to_gse(epc, SVector6::zeros()).unwrap();
        for i in 0..6 {
            assert_abs_diff_eq!(x_earth_gse[i], 0.0, epsilon = 1e-9);
        }
    }

    #[test]
    #[serial]
    fn test_gse_z_axis_near_ecliptic_pole() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // GSE ẑ is the instantaneous ecliptic normal: ~23.44 deg (the mean
        // obliquity) from the GCRF z-axis. The Moon-induced wobble of the
        // Earth's heliocentric velocity perturbs this by well under 0.5 deg.
        let r_mat = rotation_gcrf_to_gse(epc).unwrap();
        let cos_angle = r_mat[(2, 2)]; // ẑ_gse · ẑ_gcrf
        let angle_deg = cos_angle.acos().to_degrees();
        assert!(
            (angle_deg - 23.439).abs() < 0.5,
            "GSE z-axis {} deg from GCRF z, expected ~23.44",
            angle_deg
        );
    }

    #[test]
    #[serial]
    fn test_gse_roundtrip() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x = SVector6::new(1.0e8, -2.0e8, 5.0e7, 1.0e3, -2.0e3, 0.5e3);
        let x_back = state_gse_to_gcrf(epc, state_gcrf_to_gse(epc, x).unwrap()).unwrap();
        for i in 0..3 {
            assert_abs_diff_eq!(x_back[i], x[i], epsilon = 1e-4);
        }
        for i in 3..6 {
            assert_abs_diff_eq!(x_back[i], x[i], epsilon = 1e-9);
        }
    }

    #[test]
    #[serial]
    fn test_sun_earth_barycenter_between_bodies() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let seb = sun_earth_barycenter_state(epc).unwrap();
        let sun = spk_state(NAIFId::Sun, NAIFId::SolarSystemBarycenter, epc).unwrap();
        // SEB is ~GM_EARTH/(GM_SUN+GM_EARTH) of the way from Sun to Earth:
        // ~449 km from the Sun's center.
        let offset = (seb - sun).fixed_rows::<3>(0).norm();
        assert!(
            offset > 3.0e5 && offset < 6.0e5,
            "SEB-Sun offset {} m",
            offset
        );
    }

    #[test]
    #[parallel]
    fn test_body_gm_known_bodies() {
        assert_eq!(body_gm(10).unwrap(), GM_SUN);
        assert_eq!(body_gm(1).unwrap(), GM_MERCURY);
        assert_eq!(body_gm(199).unwrap(), GM_MERCURY);
        assert_eq!(body_gm(2).unwrap(), GM_VENUS);
        assert_eq!(body_gm(299).unwrap(), GM_VENUS);
        assert_eq!(body_gm(399).unwrap(), GM_EARTH);
        assert_eq!(body_gm(301).unwrap(), GM_MOON);
        assert_eq!(body_gm(3).unwrap(), GM_EARTH + GM_MOON);
        assert_eq!(body_gm(4).unwrap(), GM_MARS_SYSTEM);
        assert_eq!(body_gm(499).unwrap(), GM_MARS);
        assert_eq!(body_gm(5).unwrap(), GM_JUPITER_SYSTEM);
        assert_eq!(body_gm(599).unwrap(), GM_JUPITER);
        assert_eq!(body_gm(6).unwrap(), GM_SATURN_SYSTEM);
        assert_eq!(body_gm(699).unwrap(), GM_SATURN);
        assert_eq!(body_gm(7).unwrap(), GM_URANUS_SYSTEM);
        assert_eq!(body_gm(799).unwrap(), GM_URANUS);
        assert_eq!(body_gm(8).unwrap(), GM_NEPTUNE_SYSTEM);
        assert_eq!(body_gm(899).unwrap(), GM_NEPTUNE);
        assert_eq!(body_gm(9).unwrap(), GM_PLUTO_SYSTEM);
        assert_eq!(body_gm(999).unwrap(), GM_PLUTO);
        assert!(body_gm(502).is_err()); // Europa: no packaged GM constant
    }

    #[test]
    #[serial] // SPICE registry global
    fn test_generic_synodic_axes_matches_emr() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let (s_g, sd_g) = generic_synodic_axes(epc, 399, 301).unwrap();
        let (s_e, sd_e) = emr_axes(epc).unwrap();
        assert_abs_diff_eq!(s_g, s_e, epsilon = 0.0);
        assert_abs_diff_eq!(sd_g, sd_e, epsilon = 0.0);
    }

    #[test]
    #[serial]
    fn test_pair_barycenter_state_matches_seb() {
        setup_global_test_spice();
        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_pair = pair_barycenter_state(epc, 10, 399).unwrap();
        let x_seb = sun_earth_barycenter_state(epc).unwrap();
        assert_abs_diff_eq!(x_pair, x_seb, epsilon = 0.0);
    }

    #[test]
    #[serial]
    fn test_batch_synodic_frames_match_scalar() {
        setup_global_test_spice();
        let epc0 = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let epochs: Vec<Epoch> = (0..3).map(|i| epc0 + 3600.0 * i as f64).collect();
        let states: Vec<SVector6> = (0..3)
            .map(|i| SVector6::new(7.0e6 + 1e3 * i as f64, 1.0e6, -2.0e6, 0.0, 7.5e3, 0.0))
            .collect();
        let positions: Vec<Vector3<f64>> = states
            .iter()
            .map(|s| Vector3::new(s[0], s[1], s[2]))
            .collect();

        macro_rules! check_family {
            ($rot_f:ident, $rot_b:ident, $pos_f:ident, $pos_b:ident, $st_f:ident, $st_b:ident,
             $rots_f:ident, $rots_b:ident, $poss_f:ident, $poss_b:ident, $sts_f:ident, $sts_b:ident) => {{
                let rf = $rots_f(&epochs).unwrap();
                let rb = $rots_b(&epochs).unwrap();
                let pf = $poss_f(&epochs, &positions).unwrap();
                let pf1 = $poss_f(&epochs[..1], &positions).unwrap();
                let pb = $poss_b(&epochs, &pf).unwrap();
                let sf = $sts_f(&epochs, &states).unwrap();
                let sf1 = $sts_f(&epochs[..1], &states).unwrap();
                let sb = $sts_b(&epochs, &sf).unwrap();
                let one = $sts_b(&epochs, &states[..1]).unwrap();
                for i in 0..3 {
                    assert_eq!(rf[i], $rot_f(epochs[i]).unwrap());
                    assert_eq!(rb[i], $rot_b(epochs[i]).unwrap());
                    assert_eq!(pf[i], $pos_f(epochs[i], positions[i]).unwrap());
                    assert_eq!(pf1[i], $pos_f(epochs[0], positions[i]).unwrap());
                    assert_eq!(pb[i], $pos_b(epochs[i], pf[i]).unwrap());
                    assert_eq!(sf[i], $st_f(epochs[i], states[i]).unwrap());
                    assert_eq!(sf1[i], $st_f(epochs[0], states[i]).unwrap());
                    assert_eq!(sb[i], $st_b(epochs[i], sf[i]).unwrap());
                    assert_eq!(one[i], $st_b(epochs[i], states[0]).unwrap());
                    for k in 0..3 {
                        assert_abs_diff_eq!(sb[i][k], states[i][k], epsilon = 1e-3);
                    }
                }
                assert!($sts_f(&epochs[..2], &states).is_err());
                assert!($rots_f(&[]).unwrap().is_empty());
            }};
        }

        check_family!(
            rotation_gcrf_to_emr,
            rotation_emr_to_gcrf,
            position_gcrf_to_emr,
            position_emr_to_gcrf,
            state_gcrf_to_emr,
            state_emr_to_gcrf,
            rotations_gcrf_to_emr,
            rotations_emr_to_gcrf,
            positions_gcrf_to_emr,
            positions_emr_to_gcrf,
            states_gcrf_to_emr,
            states_emr_to_gcrf
        );
        check_family!(
            rotation_gcrf_to_ser,
            rotation_ser_to_gcrf,
            position_gcrf_to_ser,
            position_ser_to_gcrf,
            state_gcrf_to_ser,
            state_ser_to_gcrf,
            rotations_gcrf_to_ser,
            rotations_ser_to_gcrf,
            positions_gcrf_to_ser,
            positions_ser_to_gcrf,
            states_gcrf_to_ser,
            states_ser_to_gcrf
        );
        check_family!(
            rotation_gcrf_to_gse,
            rotation_gse_to_gcrf,
            position_gcrf_to_gse,
            position_gse_to_gcrf,
            state_gcrf_to_gse,
            state_gse_to_gcrf,
            rotations_gcrf_to_gse,
            rotations_gse_to_gcrf,
            positions_gcrf_to_gse,
            positions_gse_to_gcrf,
            states_gcrf_to_gse,
            states_gse_to_gcrf
        );
    }
}
