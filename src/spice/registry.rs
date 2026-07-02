/*!
 * Global SPICE kernel registry and generic SPK/PCK query API.
 *
 * Owns a process-wide set of loaded SPK/PCK kernels keyed by the
 * name-or-path string used to load them. Kernels are auto-detected as SPK
 * or PCK from the DAF ID word. Cross-kernel `spk_*` queries build a chain
 * of segments spanning all loaded SPK kernels, collected in natural load
 * order (each kernel's segments in file order); segment selection at
 * evaluation time picks the last covering candidate, so more recently
 * loaded kernels take precedence for overlapping body pairs (matching
 * SPICE's own "last loaded wins" convention). `pck_*` queries search
 * loaded PCK kernels newest-first for a frame with coverage at the
 * requested epoch.
 */

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use nalgebra::{Matrix3, Vector3, Vector6};
use once_cell::sync::Lazy;

use crate::datasets::naif::{
    SUPPORTED_KERNELS, SUPPORTED_PCK_KERNELS, download_de_kernel, download_pck_kernel,
};
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

use super::daf::DafFile;
use super::pck::BPCK;
use super::segments::{ChebyshevSegment, is_coverage_error};
use super::spk::{
    ChainLink, SPK, evaluate_chain_position, evaluate_chain_state, evaluate_chain_velocity,
    evaluate_with_epoch_fallback, resolve_chain,
};

// ============================================================================
// NAIF ID Constants
// ============================================================================

/// NAIF ID of the Solar System Barycenter.
pub const NAIF_SSB: i32 = 0;
/// NAIF ID of the Mercury Barycenter.
pub const NAIF_MERCURY_BARYCENTER: i32 = 1;
/// NAIF ID of the Venus Barycenter.
pub const NAIF_VENUS_BARYCENTER: i32 = 2;
/// NAIF ID of the Earth-Moon Barycenter.
pub const NAIF_EMB: i32 = 3;
/// NAIF ID of the Mars Barycenter.
pub const NAIF_MARS_BARYCENTER: i32 = 4;
/// NAIF ID of the Jupiter Barycenter.
pub const NAIF_JUPITER_BARYCENTER: i32 = 5;
/// NAIF ID of the Saturn Barycenter.
pub const NAIF_SATURN_BARYCENTER: i32 = 6;
/// NAIF ID of the Uranus Barycenter.
pub const NAIF_URANUS_BARYCENTER: i32 = 7;
/// NAIF ID of the Neptune Barycenter.
pub const NAIF_NEPTUNE_BARYCENTER: i32 = 8;
/// NAIF ID of the Pluto Barycenter.
pub const NAIF_PLUTO_BARYCENTER: i32 = 9;
/// NAIF ID of the Sun.
pub const NAIF_SUN: i32 = 10;
/// NAIF ID of the Mercury body center.
pub const NAIF_MERCURY: i32 = 199;
/// NAIF ID of the Venus body center.
pub const NAIF_VENUS: i32 = 299;
/// NAIF ID of the Earth body center.
pub const NAIF_EARTH: i32 = 399;
/// NAIF ID of the Moon body center.
pub const NAIF_MOON: i32 = 301;
/// NAIF ID of the Mars body center.
pub const NAIF_MARS: i32 = 499;

// ============================================================================
// Global Registry State
// ============================================================================

/// One loaded kernel, tagged by its parsed type.
enum Kernel {
    Spk(Arc<SPK>),
    Pck(Arc<BPCK>),
}

/// Process-wide set of loaded SPICE kernels plus derived caches.
struct KernelRegistry {
    /// Kernels keyed by name-or-path, plus load order for precedence.
    kernels: HashMap<String, Kernel>,
    load_order: Vec<String>,
    /// Cross-kernel chain cache, invalidated on load/unload.
    chain_cache: HashMap<(i32, i32), Arc<Vec<ChainLink>>>,
}

static GLOBAL_SPICE: Lazy<RwLock<KernelRegistry>> = Lazy::new(|| {
    RwLock::new(KernelRegistry {
        kernels: HashMap::new(),
        load_order: Vec::new(),
        chain_cache: HashMap::new(),
    })
});

/// Convert a brahe [`Epoch`] to SPICE ephemeris time (TDB seconds past
/// J2000).
pub(crate) fn epoch_to_et(epc: Epoch) -> f64 {
    epc.seconds_past_j2000_as_time_system(TimeSystem::TDB)
}

/// Resolve a kernel name or file path to a local file path, downloading
/// (and caching) known DE kernel names via the NAIF dataset cache.
///
/// Known kernel names are looked up in `naif::SUPPORTED_KERNELS` (DE SPK
/// kernels) and `naif::SUPPORTED_PCK_KERNELS` (binary PCK kernels) — the
/// same lists `datasets::naif` validates against — rather than a second,
/// independently maintained set of literals, so the two cannot drift out
/// of sync.
fn resolve_kernel_source(name_or_path: &str) -> Result<std::path::PathBuf, BraheError> {
    if SUPPORTED_KERNELS.contains(&name_or_path) {
        download_de_kernel(name_or_path, None)
    } else if SUPPORTED_PCK_KERNELS
        .iter()
        .any(|(name, _)| *name == name_or_path)
    {
        download_pck_kernel(name_or_path, None)
    } else {
        let path = Path::new(name_or_path);
        if path.exists() {
            Ok(path.to_path_buf())
        } else {
            Err(BraheError::IoError(format!(
                "Kernel '{}' is neither a known kernel name nor an existing file path",
                name_or_path
            )))
        }
    }
}

// ============================================================================
// Load/Unload API
// ============================================================================

/// Load a SPICE kernel into the global registry, auto-detecting SPK vs.
/// PCK from the DAF ID word.
///
/// Idempotent: calling with a `name_or_path` that is already loaded is a
/// no-op. Known DE kernel names (`"de440s"`, `"de440"`, etc.) are
/// downloaded and cached via [`crate::datasets::naif::download_de_kernel`];
/// the known binary PCK name `"moon_pa_de440"` is downloaded and cached via
/// [`crate::datasets::naif::download_pck_kernel`]; any other string is
/// treated as a file path.
///
/// # Arguments
/// - `name_or_path`: A known DE kernel or PCK kernel name, or a path to a `.bsp`/`.bpc` file
///
/// # Returns
/// - `Ok(())` on success, or `BraheError` if the kernel cannot be resolved,
///   read, or parsed
///
/// # Examples
/// ```no_run
/// use brahe::spice::load_kernel;
///
/// load_kernel("de440s").expect("Failed to load DE440s");
/// ```
pub fn load_kernel(name_or_path: &str) -> Result<(), BraheError> {
    // Idempotent short-circuit (read lock only) for the common case.
    {
        let reg = GLOBAL_SPICE.read().unwrap();
        if reg.kernels.contains_key(name_or_path) {
            return Ok(());
        }
    }

    let path = resolve_kernel_source(name_or_path)?;
    let daf = DafFile::from_file(&path)?;
    let kernel = match daf.id_word.as_str() {
        "DAF/SPK" => Kernel::Spk(Arc::new(SPK::from_daf(daf)?)),
        "DAF/PCK" => Kernel::Pck(Arc::new(BPCK::from_daf(daf)?)),
        other => {
            return Err(BraheError::IoError(format!(
                "Unrecognized DAF ID word '{}' in kernel '{}'; expected 'DAF/SPK' or 'DAF/PCK'",
                other, name_or_path
            )));
        }
    };

    let mut reg = GLOBAL_SPICE.write().unwrap();
    // Double-checked locking: another thread may have loaded it while we
    // resolved and parsed the file.
    if reg.kernels.contains_key(name_or_path) {
        return Ok(());
    }
    reg.kernels.insert(name_or_path.to_string(), kernel);
    reg.load_order.push(name_or_path.to_string());
    reg.chain_cache.clear();
    Ok(())
}

/// Unload a kernel previously loaded via [`load_kernel`].
///
/// # Arguments
/// - `name_or_path`: The same string originally passed to [`load_kernel`]
///
/// # Returns
/// - `Ok(())` on success, or `BraheError` if `name_or_path` is not loaded
///
/// # Examples
/// ```no_run
/// use brahe::spice::{load_kernel, unload_kernel};
///
/// load_kernel("de440s").unwrap();
/// unload_kernel("de440s").expect("Failed to unload DE440s");
/// ```
pub fn unload_kernel(name_or_path: &str) -> Result<(), BraheError> {
    let mut reg = GLOBAL_SPICE.write().unwrap();
    if reg.kernels.remove(name_or_path).is_none() {
        return Err(BraheError::Error(format!(
            "Kernel '{}' is not currently loaded",
            name_or_path
        )));
    }
    reg.load_order.retain(|k| k != name_or_path);
    reg.chain_cache.clear();
    Ok(())
}

/// Unload all kernels from the global registry.
///
/// # Examples
/// ```no_run
/// use brahe::spice::{load_kernel, clear_kernels};
///
/// load_kernel("de440s").unwrap();
/// clear_kernels();
/// ```
pub fn clear_kernels() {
    let mut reg = GLOBAL_SPICE.write().unwrap();
    reg.kernels.clear();
    reg.load_order.clear();
    reg.chain_cache.clear();
}

/// Names/paths of currently loaded kernels, in load order.
///
/// # Returns
/// - Kernel names/paths in the order they were passed to [`load_kernel`]
///
/// # Examples
/// ```no_run
/// use brahe::spice::{load_kernel, loaded_kernels};
///
/// load_kernel("de440s").unwrap();
/// assert_eq!(loaded_kernels(), vec!["de440s".to_string()]);
/// ```
pub fn loaded_kernels() -> Vec<String> {
    GLOBAL_SPICE.read().unwrap().load_order.clone()
}

/// Initialize the global ephemeris with the default DE440s kernel.
///
/// Equivalent to `load_kernel("de440s")`. Optional: `spk_position` and
/// related queries lazily auto-initialize DE440s if no kernels are loaded.
///
/// # Returns
/// - `Ok(())` on success, or `BraheError` on download/parse failure
///
/// # Examples
/// ```no_run
/// use brahe::spice::initialize_ephemeris;
///
/// initialize_ephemeris().expect("Failed to initialize ephemeris");
/// ```
pub fn initialize_ephemeris() -> Result<(), BraheError> {
    load_kernel("de440s")
}

/// Initialize the global ephemeris with a specific DE kernel.
///
/// Equivalent to `load_kernel(kernel)`.
///
/// # Arguments
/// - `kernel`: A known DE kernel name (e.g. `"de440s"`, `"de440"`)
///
/// # Returns
/// - `Ok(())` on success, or `BraheError` on download/parse failure
///
/// # Examples
/// ```no_run
/// use brahe::spice::initialize_ephemeris_with_kernel;
///
/// initialize_ephemeris_with_kernel("de440").expect("Failed to initialize DE440");
/// ```
pub fn initialize_ephemeris_with_kernel(kernel: &str) -> Result<(), BraheError> {
    load_kernel(kernel)
}

// ============================================================================
// Cross-Kernel Chain Resolution
// ============================================================================

/// Collect the segments of every currently loaded SPK kernel in natural
/// order (kernels in load order, each kernel's segments in file order).
/// Combined with select-last in `covering_segment`, this makes the most
/// recently loaded covering segment win.
fn collect_spk_segments(reg: &KernelRegistry) -> Vec<Arc<ChebyshevSegment>> {
    let mut segments = Vec::new();
    for name in reg.load_order.iter() {
        if let Some(Kernel::Spk(spk)) = reg.kernels.get(name) {
            segments.extend(spk.segments().iter().cloned());
        }
    }
    segments
}

/// Resolve (and cache) the cross-kernel segment chain for `target` rel
/// `center`, spanning all loaded SPK kernels. Segments are offered to
/// [`resolve_chain`] in natural order (kernels in load order, each
/// kernel's segments in file order); the evaluators then select the last
/// covering candidate per link, so the most recently loaded kernel wins
/// for overlapping body pairs.
fn global_chain(target: i32, center: i32) -> Result<Arc<Vec<ChainLink>>, BraheError> {
    {
        let reg = GLOBAL_SPICE.read().unwrap();
        if let Some(chain) = reg.chain_cache.get(&(target, center)) {
            return Ok(Arc::clone(chain));
        }
    }
    let mut reg = GLOBAL_SPICE.write().unwrap();
    if let Some(chain) = reg.chain_cache.get(&(target, center)) {
        return Ok(Arc::clone(chain)); // double-checked
    }
    let segments = collect_spk_segments(&reg);
    if segments.is_empty() {
        return Err(BraheError::InitializationError(
            "No SPK kernels loaded; call load_kernel(...) or initialize_ephemeris() first"
                .to_string(),
        ));
    }
    let chain = Arc::new(resolve_chain(&segments, target, center)?);
    reg.chain_cache.insert((target, center), Arc::clone(&chain));
    Ok(chain)
}

/// Evaluate a cached `chain` at `et`; on an out-of-coverage failure (the
/// cached topology-only chain's segments don't cover `et` while a
/// different, longer path through the currently loaded kernels might),
/// fall back to an epoch-aware re-resolution across every loaded SPK
/// kernel's segments. Non-coverage errors (e.g. corrupt record data)
/// propagate unchanged. See [`evaluate_with_epoch_fallback`] for the
/// gating and why the fallback chain is never cached.
fn global_eval<T>(
    target: i32,
    center: i32,
    et: f64,
    chain: &[ChainLink],
    eval: impl Fn(&[ChainLink], f64) -> Result<T, BraheError>,
) -> Result<T, BraheError> {
    evaluate_with_epoch_fallback(
        chain,
        target,
        center,
        et,
        || collect_spk_segments(&GLOBAL_SPICE.read().unwrap()),
        eval,
    )
}

/// Load `de440s` if no SPK kernel is currently loaded, preserving the
/// library's historical lazy-initialization behavior for `spk_*` queries.
///
/// Checks for an SPK specifically (not just any loaded kernel) so that a
/// registry holding only PCK kernels (e.g. after `load_kernel("moon_pa_de440")`)
/// still auto-initializes the default ephemeris.
fn ensure_default_ephemeris_loaded() -> Result<(), BraheError> {
    let has_spk = GLOBAL_SPICE
        .read()
        .unwrap()
        .kernels
        .values()
        .any(|k| matches!(k, Kernel::Spk(_)));
    if !has_spk {
        load_kernel("de440s")?;
    }
    Ok(())
}

// ============================================================================
// Generic SPK Queries
// ============================================================================

/// Position of `target` relative to `center` at `epc`, resolved across all
/// loaded SPK kernels.
///
/// Auto-initializes with `de440s` if no kernels are loaded. The
/// cross-kernel chain is resolved once per `(target, center)` pair and
/// cached (topology only, ignoring epoch); if the cached chain's segments
/// don't cover `epc` while a different path through the loaded kernels
/// does, this transparently falls back to an epoch-aware re-resolution
/// (not cached) — see `evaluate_with_epoch_fallback`.
///
/// # Arguments
/// - `target`: NAIF ID of the target body
/// - `center`: NAIF ID of the center body
/// - `epc`: Epoch at which to evaluate the position
///
/// # Returns
/// - Position of `target` relative to `center` in the kernel's inertial
///   frame (ICRF axes). Units: [m]
///
/// # Examples
/// ```
/// use brahe::spice::{NAIF_MOON, NAIF_EARTH, spk_position, initialize_ephemeris};
/// use brahe::time::{Epoch, TimeSystem};
///
/// initialize_ephemeris().unwrap();
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let r_moon = spk_position(NAIF_MOON, NAIF_EARTH, epc).unwrap();
/// ```
pub fn spk_position(target: i32, center: i32, epc: Epoch) -> Result<Vector3<f64>, BraheError> {
    ensure_default_ephemeris_loaded()?;
    let et = epoch_to_et(epc);
    let chain = global_chain(target, center)?;
    Ok(global_eval(target, center, et, &chain, evaluate_chain_position)? * 1.0e3)
}

/// Velocity of `target` relative to `center` at `epc`, resolved across all
/// loaded SPK kernels.
///
/// Auto-initializes with `de440s` if no kernels are loaded. The
/// cross-kernel chain is resolved once per `(target, center)` pair and
/// cached (topology only, ignoring epoch); if the cached chain's segments
/// don't cover `epc` while a different path through the loaded kernels
/// does, this transparently falls back to an epoch-aware re-resolution
/// (not cached) — see `evaluate_with_epoch_fallback`.
///
/// # Arguments
/// - `target`: NAIF ID of the target body
/// - `center`: NAIF ID of the center body
/// - `epc`: Epoch at which to evaluate the velocity
///
/// # Returns
/// - Velocity of `target` relative to `center` in the kernel's inertial
///   frame (ICRF axes). Units: [m/s]
///
/// # Examples
/// ```
/// use brahe::spice::{NAIF_MOON, NAIF_EARTH, spk_velocity, initialize_ephemeris};
/// use brahe::time::{Epoch, TimeSystem};
///
/// initialize_ephemeris().unwrap();
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let v_moon = spk_velocity(NAIF_MOON, NAIF_EARTH, epc).unwrap();
/// ```
pub fn spk_velocity(target: i32, center: i32, epc: Epoch) -> Result<Vector3<f64>, BraheError> {
    ensure_default_ephemeris_loaded()?;
    let et = epoch_to_et(epc);
    let chain = global_chain(target, center)?;
    Ok(global_eval(target, center, et, &chain, evaluate_chain_velocity)? * 1.0e3)
}

/// Position and velocity of `target` relative to `center` at `epc`,
/// resolved across all loaded SPK kernels.
///
/// Auto-initializes with `de440s` if no kernels are loaded. The
/// cross-kernel chain is resolved once per `(target, center)` pair and
/// cached (topology only, ignoring epoch); if the cached chain's segments
/// don't cover `epc` while a different path through the loaded kernels
/// does, this transparently falls back to an epoch-aware re-resolution
/// (not cached) — see `evaluate_with_epoch_fallback`.
///
/// # Arguments
/// - `target`: NAIF ID of the target body
/// - `center`: NAIF ID of the center body
/// - `epc`: Epoch at which to evaluate the state
///
/// # Returns
/// - State `[x, y, z, vx, vy, vz]` of `target` relative to `center` in the
///   kernel's inertial frame (ICRF axes). Units: [m, m/s]
///
/// # Examples
/// ```
/// use brahe::spice::{NAIF_MOON, NAIF_EARTH, spk_state, initialize_ephemeris};
/// use brahe::time::{Epoch, TimeSystem};
///
/// initialize_ephemeris().unwrap();
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let x_moon = spk_state(NAIF_MOON, NAIF_EARTH, epc).unwrap();
/// ```
pub fn spk_state(target: i32, center: i32, epc: Epoch) -> Result<Vector6<f64>, BraheError> {
    ensure_default_ephemeris_loaded()?;
    let et = epoch_to_et(epc);
    let chain = global_chain(target, center)?;
    let (r, v) = global_eval(target, center, et, &chain, evaluate_chain_state)?;
    Ok(Vector6::new(
        r[0] * 1.0e3,
        r[1] * 1.0e3,
        r[2] * 1.0e3,
        v[0] * 1.0e3,
        v[1] * 1.0e3,
        v[2] * 1.0e3,
    ))
}

/// Fetch the loaded `Kernel::Spk` entry for `kernel_name`, loading it first
/// if absent.
fn spk_kernel_for_query(kernel_name: &str) -> Result<Arc<SPK>, BraheError> {
    load_kernel(kernel_name)?;
    let reg = GLOBAL_SPICE.read().unwrap();
    match reg.kernels.get(kernel_name) {
        Some(Kernel::Spk(spk)) => Ok(Arc::clone(spk)),
        Some(Kernel::Pck(_)) => Err(BraheError::Error(format!(
            "Kernel '{}' is a binary PCK, not an SPK",
            kernel_name
        ))),
        None => unreachable!("load_kernel just ensured '{kernel_name}' is present"),
    }
}

/// Position of `target` relative to `center` at `epc`, queried from a
/// single named kernel.
///
/// Queries **that kernel only**: no cross-kernel chaining is performed and
/// the registry's last-loaded-wins precedence semantics do not apply. The
/// kernel is auto-loaded by name or path if not already loaded.
///
/// # Arguments
/// - `kernel_name`: A known DE kernel name (e.g. `"de440s"`, `"de440"`), or
///   a path to a `.bsp` file
/// - `target`: NAIF ID of the target body
/// - `center`: NAIF ID of the center body
/// - `epc`: Epoch at which to evaluate the position
///
/// # Returns
/// - Position of `target` relative to `center` in the kernel's inertial
///   frame (ICRF axes). Units: [m]
///
/// # Examples
/// ```no_run
/// use brahe::spice::{NAIF_MOON, NAIF_EARTH, spk_position_in_kernel};
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let r_moon = spk_position_in_kernel("de440s", NAIF_MOON, NAIF_EARTH, epc).unwrap();
/// ```
pub fn spk_position_in_kernel(
    kernel_name: &str,
    target: i32,
    center: i32,
    epc: Epoch,
) -> Result<Vector3<f64>, BraheError> {
    let spk = spk_kernel_for_query(kernel_name)?;
    spk.position(target, center, epoch_to_et(epc))
}

/// Velocity of `target` relative to `center` at `epc`, queried from a
/// single named kernel.
///
/// Queries **that kernel only**: no cross-kernel chaining is performed and
/// the registry's last-loaded-wins precedence semantics do not apply. The
/// kernel is auto-loaded by name or path if not already loaded.
///
/// # Arguments
/// - `kernel_name`: A known DE kernel name (e.g. `"de440s"`, `"de440"`), or
///   a path to a `.bsp` file
/// - `target`: NAIF ID of the target body
/// - `center`: NAIF ID of the center body
/// - `epc`: Epoch at which to evaluate the velocity
///
/// # Returns
/// - Velocity of `target` relative to `center` in the kernel's inertial
///   frame (ICRF axes). Units: [m/s]
///
/// # Examples
/// ```no_run
/// use brahe::spice::{NAIF_MOON, NAIF_EARTH, spk_velocity_in_kernel};
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let v_moon = spk_velocity_in_kernel("de440s", NAIF_MOON, NAIF_EARTH, epc).unwrap();
/// ```
pub fn spk_velocity_in_kernel(
    kernel_name: &str,
    target: i32,
    center: i32,
    epc: Epoch,
) -> Result<Vector3<f64>, BraheError> {
    let spk = spk_kernel_for_query(kernel_name)?;
    spk.velocity(target, center, epoch_to_et(epc))
}

/// Position and velocity of `target` relative to `center` at `epc`,
/// queried from a single named kernel.
///
/// Queries **that kernel only**: no cross-kernel chaining is performed and
/// the registry's last-loaded-wins precedence semantics do not apply. The
/// kernel is auto-loaded by name or path if not already loaded.
///
/// # Arguments
/// - `kernel_name`: A known DE kernel name (e.g. `"de440s"`, `"de440"`), or
///   a path to a `.bsp` file
/// - `target`: NAIF ID of the target body
/// - `center`: NAIF ID of the center body
/// - `epc`: Epoch at which to evaluate the state
///
/// # Returns
/// - State `[x, y, z, vx, vy, vz]` of `target` relative to `center` in the
///   kernel's inertial frame (ICRF axes). Units: [m, m/s]
///
/// # Examples
/// ```no_run
/// use brahe::spice::{NAIF_MOON, NAIF_EARTH, spk_state_in_kernel};
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let x_moon = spk_state_in_kernel("de440s", NAIF_MOON, NAIF_EARTH, epc).unwrap();
/// ```
pub fn spk_state_in_kernel(
    kernel_name: &str,
    target: i32,
    center: i32,
    epc: Epoch,
) -> Result<Vector6<f64>, BraheError> {
    let spk = spk_kernel_for_query(kernel_name)?;
    spk.state(target, center, epoch_to_et(epc))
}

// ============================================================================
// Generic PCK Queries
// ============================================================================

/// Search loaded PCK kernels newest-first for `frame_id` with coverage at
/// `et`, applying `query` to the first kernel that both contains the frame
/// and covers the epoch.
///
/// A per-kernel coverage/frame-lookup miss (identified via
/// `is_coverage_error`) moves on to the next kernel; any other error (e.g.
/// a segment record with corrupt data) is a genuine data problem, not just
/// "try the next kernel", and is propagated immediately rather than being
/// swallowed into the generic "not covered" error below.
fn pck_query<T>(
    frame_id: i32,
    epc: Epoch,
    query: impl Fn(&BPCK, i32, f64) -> Result<T, BraheError>,
) -> Result<T, BraheError> {
    let et = epoch_to_et(epc);
    let reg = GLOBAL_SPICE.read().unwrap();
    let mut available_frames: Vec<i32> = Vec::new();
    for name in reg.load_order.iter().rev() {
        if let Some(Kernel::Pck(pck)) = reg.kernels.get(name) {
            for f in pck.frame_ids() {
                if !available_frames.contains(&f) {
                    available_frames.push(f);
                }
            }
            match query(pck, frame_id, et) {
                Ok(result) => return Ok(result),
                Err(err) if is_coverage_error(&err) => continue,
                Err(err) => return Err(err),
            }
        }
    }
    Err(BraheError::Error(format!(
        "Frame class ID {} not covered by any loaded PCK kernel at epoch ET {} \
         (available frame IDs: {:?})",
        frame_id, et, available_frames
    )))
}

/// 3-1-3 Euler angles and rates of the body-fixed frame `frame_id`
/// relative to its segment reference frame (ICRF for DE440-era kernels),
/// searching loaded PCK kernels newest-first.
///
/// PCKs are never auto-initialized; a PCK kernel must be explicitly loaded
/// via [`load_kernel`] first.
///
/// # Arguments
/// - `frame_id`: Body-frame class ID (e.g. 31008 for MOON_PA_DE440)
/// - `epc`: Epoch at which to evaluate the orientation
///
/// # Returns
/// - `(angles, rates)`: `angles = [phi, delta, w]` in [rad]; `rates` are
///   their time derivatives in [rad/s]
///
/// # Examples
/// ```no_run
/// use brahe::spice::{load_kernel, pck_euler_angles};
/// use brahe::time::{Epoch, TimeSystem};
///
/// load_kernel("/path/to/moon_pa_de440_200625.bpc").unwrap();
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let (angles, rates) = pck_euler_angles(31008, epc).unwrap();
/// ```
pub fn pck_euler_angles(
    frame_id: i32,
    epc: Epoch,
) -> Result<(Vector3<f64>, Vector3<f64>), BraheError> {
    pck_query(frame_id, epc, |pck, f, et| pck.euler_angles(f, et))
}

/// Rotation matrix from the segment reference frame (ICRF) to the
/// body-fixed frame `frame_id`, searching loaded PCK kernels newest-first.
///
/// PCKs are never auto-initialized; a PCK kernel must be explicitly loaded
/// via [`load_kernel`] first.
///
/// # Arguments
/// - `frame_id`: Body-frame class ID (e.g. 31008 for MOON_PA_DE440)
/// - `epc`: Epoch at which to evaluate the orientation
///
/// # Returns
/// - 3x3 rotation matrix (ICRF to body-fixed). Dimensionless.
///
/// # Examples
/// ```no_run
/// use brahe::spice::{load_kernel, pck_rotation_matrix};
/// use brahe::time::{Epoch, TimeSystem};
///
/// load_kernel("/path/to/moon_pa_de440_200625.bpc").unwrap();
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let r = pck_rotation_matrix(31008, epc).unwrap();
/// ```
pub fn pck_rotation_matrix(frame_id: i32, epc: Epoch) -> Result<Matrix3<f64>, BraheError> {
    pck_query(frame_id, epc, |pck, f, et| pck.rotation_matrix(f, et))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use serial_test::serial;

    use super::*;
    use crate::utils::testing::setup_global_test_spice;

    fn epc_2025() -> Epoch {
        Epoch::from_date(2025, 1, 1, TimeSystem::UTC)
    }

    /// Build a minimal little-endian SPK with one type-2 segment for target
    /// 10 rel center 0 whose x-position is the constant `x_km` over ET in
    /// [-2e9, 2e9] (covers 2025); y = z = 0.
    fn synthetic_spk_bytes(x_km: f64) -> Vec<u8> {
        let degree = 1usize;
        let rsize = 2 + 3 * (degree + 1); // 8
        // Data: record [MID, RADIUS, x0, x1, y0, y1, z0, z1] + trailer
        let data: Vec<f64> = vec![
            0.0,
            2.0e9, // MID, RADIUS
            x_km,
            0.0, // x
            0.0,
            0.0, // y
            0.0,
            0.0, // z
            -2.0e9,
            4.0e9,
            rsize as f64,
            1.0, // INIT, INTLEN, RSIZE, N
        ];

        let mut file = vec![0u8; 4 * 1024];
        file[..8].copy_from_slice(b"DAF/SPK ");
        file[8..12].copy_from_slice(&2i32.to_le_bytes()); // ND
        file[12..16].copy_from_slice(&6i32.to_le_bytes()); // NI
        file[76..80].copy_from_slice(&2i32.to_le_bytes()); // FWARD
        file[80..84].copy_from_slice(&2i32.to_le_bytes()); // BWARD
        file[84..88].copy_from_slice(&500i32.to_le_bytes()); // FREE
        file[88..96].copy_from_slice(b"LTL-IEEE");

        // Summary record (record 2): NEXT=0, PREV=0, NSUM=1
        let rec = 1024;
        file[rec + 16..rec + 24].copy_from_slice(&1f64.to_le_bytes());
        // doubles: start_et, end_et
        file[rec + 24..rec + 32].copy_from_slice(&(-2.0e9f64).to_le_bytes());
        file[rec + 32..rec + 40].copy_from_slice(&2.0e9f64.to_le_bytes());
        // ints: [target, center, frame, type, start_addr, end_addr]
        // Data record = record 4 => word address 385 ((4-1)*128 + 1)
        let start_addr = 385i32;
        let end_addr = start_addr + data.len() as i32 - 1;
        for (i, v) in [10i32, 0, 1, 2, start_addr, end_addr].iter().enumerate() {
            let off = rec + 40 + i * 4;
            file[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        // Name record (record 3): spaces
        for b in &mut file[2048..2048 + 40] {
            *b = b' ';
        }
        // Data (record 4)
        for (i, v) in data.iter().enumerate() {
            let off = 3 * 1024 + i * 8;
            file[off..off + 8].copy_from_slice(&v.to_le_bytes());
        }
        file
    }

    /// Build a minimal little-endian SPK with one type-2 segment for
    /// `target` rel `center` whose x-position is the constant `x_km` over
    /// ET in `[start_et, end_et]`; y = z = 0. Generalizes
    /// [`synthetic_spk_bytes`] (fixed to target 10, center 0, and a
    /// [-2e9, 2e9] span) to arbitrary body pairs and coverage intervals,
    /// for constructing multi-kernel/multi-pair chain topologies.
    fn synthetic_spk_bytes_for(
        target: i32,
        center: i32,
        start_et: f64,
        end_et: f64,
        x_km: f64,
    ) -> Vec<u8> {
        let degree = 1usize;
        let rsize = 2 + 3 * (degree + 1); // 8
        let mid = (start_et + end_et) / 2.0;
        let radius = (end_et - start_et) / 2.0;
        // Data: record [MID, RADIUS, x0, x1, y0, y1, z0, z1] + trailer
        let data: Vec<f64> = vec![
            mid,
            radius,
            x_km,
            0.0, // x
            0.0,
            0.0, // y
            0.0,
            0.0, // z
            start_et,
            end_et - start_et,
            rsize as f64,
            1.0, // INIT, INTLEN, RSIZE, N
        ];

        let mut file = vec![0u8; 4 * 1024];
        file[..8].copy_from_slice(b"DAF/SPK ");
        file[8..12].copy_from_slice(&2i32.to_le_bytes()); // ND
        file[12..16].copy_from_slice(&6i32.to_le_bytes()); // NI
        file[76..80].copy_from_slice(&2i32.to_le_bytes()); // FWARD
        file[80..84].copy_from_slice(&2i32.to_le_bytes()); // BWARD
        file[84..88].copy_from_slice(&500i32.to_le_bytes()); // FREE
        file[88..96].copy_from_slice(b"LTL-IEEE");

        // Summary record (record 2): NEXT=0, PREV=0, NSUM=1
        let rec = 1024;
        file[rec + 16..rec + 24].copy_from_slice(&1f64.to_le_bytes());
        // doubles: start_et, end_et
        file[rec + 24..rec + 32].copy_from_slice(&start_et.to_le_bytes());
        file[rec + 32..rec + 40].copy_from_slice(&end_et.to_le_bytes());
        // ints: [target, center, frame, type, start_addr, end_addr]
        // Data record = record 4 => word address 385 ((4-1)*128 + 1)
        let start_addr = 385i32;
        let end_addr = start_addr + data.len() as i32 - 1;
        for (i, v) in [target, center, 1, 2, start_addr, end_addr]
            .iter()
            .enumerate()
        {
            let off = rec + 40 + i * 4;
            file[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        // Name record (record 3): spaces
        for b in &mut file[2048..2048 + 40] {
            *b = b' ';
        }
        // Data (record 4)
        for (i, v) in data.iter().enumerate() {
            let off = 3 * 1024 + i * 8;
            file[off..off + 8].copy_from_slice(&v.to_le_bytes());
        }
        file
    }

    /// Epoch corresponding to `et` TDB seconds past J2000 (approximate:
    /// converts via the JD relationship rather than round-tripping through
    /// `epoch_to_et`, close enough for the second-scale synthetic-segment
    /// coverage boundaries used in these tests).
    fn epc_from_et(et: f64) -> Epoch {
        Epoch::from_jd(
            crate::constants::JD_J2000 + et / crate::constants::SECONDS_PER_DAY,
            TimeSystem::TDB,
        )
    }

    /// Build a minimal little-endian binary PCK with one type-2 segment for
    /// frame 31006 rel frame 1, covering et in [0, 1000] (does not cover
    /// `epc_2025()`; only used to populate the registry with a PCK-typed
    /// kernel, not to be queried).
    fn synthetic_bpck_bytes() -> Vec<u8> {
        let degree = 1usize;
        let rsize = 2 + 3 * (degree + 1); // 8
        let data: Vec<f64> = vec![
            500.0,
            500.0, // MID, RADIUS
            0.1,
            0.0, // phi
            0.3,
            0.0, // delta
            0.4,
            0.0, // w
            0.0,
            1000.0,
            rsize as f64,
            1.0, // INIT, INTLEN, RSIZE, N
        ];

        let mut file = vec![0u8; 4 * 1024];
        file[..8].copy_from_slice(b"DAF/PCK ");
        file[8..12].copy_from_slice(&2i32.to_le_bytes()); // ND
        file[12..16].copy_from_slice(&5i32.to_le_bytes()); // NI
        file[76..80].copy_from_slice(&2i32.to_le_bytes()); // FWARD
        file[80..84].copy_from_slice(&2i32.to_le_bytes()); // BWARD
        file[84..88].copy_from_slice(&500i32.to_le_bytes()); // FREE
        file[88..96].copy_from_slice(b"LTL-IEEE");

        // Summary record (record 2): NEXT=0, PREV=0, NSUM=1
        let rec = 1024;
        file[rec + 16..rec + 24].copy_from_slice(&1f64.to_le_bytes());
        file[rec + 24..rec + 32].copy_from_slice(&0f64.to_le_bytes());
        file[rec + 32..rec + 40].copy_from_slice(&1000f64.to_le_bytes());
        // ints: [frame_class_id, reference_frame, type, start_addr, end_addr]
        let start_addr = 385i32;
        let end_addr = start_addr + data.len() as i32 - 1;
        for (i, v) in [31006i32, 1, 2, start_addr, end_addr].iter().enumerate() {
            let off = rec + 40 + i * 4;
            file[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        // Name record (record 3): spaces
        for b in &mut file[2048..2048 + 40] {
            *b = b' ';
        }
        // Data (record 4)
        for (i, v) in data.iter().enumerate() {
            let off = 3 * 1024 + i * 8;
            file[off..off + 8].copy_from_slice(&v.to_le_bytes());
        }
        file
    }

    /// Build a minimal little-endian binary PCK with one type-2 segment for
    /// `frame_id` rel frame 1, covering et in `[0, 1000]`, whose single
    /// record has `RADIUS = 0.0` — a corrupt-data condition (invalid
    /// RADIUS) distinct from "frame not found" or "out of coverage".
    fn synthetic_bpck_bytes_corrupt(frame_id: i32) -> Vec<u8> {
        let degree = 1usize;
        let rsize = 2 + 3 * (degree + 1); // 8
        let data: Vec<f64> = vec![
            500.0,
            0.0, // MID, RADIUS=0 (invalid)
            0.1,
            0.0, // phi
            0.3,
            0.0, // delta
            0.4,
            0.0, // w
            0.0,
            1000.0,
            rsize as f64,
            1.0, // INIT, INTLEN, RSIZE, N
        ];

        let mut file = vec![0u8; 4 * 1024];
        file[..8].copy_from_slice(b"DAF/PCK ");
        file[8..12].copy_from_slice(&2i32.to_le_bytes()); // ND
        file[12..16].copy_from_slice(&5i32.to_le_bytes()); // NI
        file[76..80].copy_from_slice(&2i32.to_le_bytes()); // FWARD
        file[80..84].copy_from_slice(&2i32.to_le_bytes()); // BWARD
        file[84..88].copy_from_slice(&500i32.to_le_bytes()); // FREE
        file[88..96].copy_from_slice(b"LTL-IEEE");

        // Summary record (record 2): NEXT=0, PREV=0, NSUM=1
        let rec = 1024;
        file[rec + 16..rec + 24].copy_from_slice(&1f64.to_le_bytes());
        file[rec + 24..rec + 32].copy_from_slice(&0f64.to_le_bytes());
        file[rec + 32..rec + 40].copy_from_slice(&1000f64.to_le_bytes());
        // ints: [frame_class_id, reference_frame, type, start_addr, end_addr]
        let start_addr = 385i32;
        let end_addr = start_addr + data.len() as i32 - 1;
        for (i, v) in [frame_id, 1, 2, start_addr, end_addr].iter().enumerate() {
            let off = rec + 40 + i * 4;
            file[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        // Name record (record 3): spaces
        for b in &mut file[2048..2048 + 40] {
            *b = b' ';
        }
        // Data (record 4)
        for (i, v) in data.iter().enumerate() {
            let off = 3 * 1024 + i * 8;
            file[off..off + 8].copy_from_slice(&v.to_le_bytes());
        }
        file
    }

    #[test]
    #[serial]
    fn test_pck_query_propagates_corrupt_data_error_instead_of_masking_as_not_covered() {
        // Regression test: before the fix, `pck_query` collapsed every
        // per-kernel failure (including a corrupt record's invalid-RADIUS
        // IoError) into the generic "not covered" aggregate error. A newer
        // kernel providing frame 31099 with a corrupt record must have its
        // error propagate immediately rather than being swallowed while
        // falling through to an older kernel that doesn't have the frame
        // at all.
        setup_global_test_spice();
        clear_kernels();

        let dir = tempfile::tempdir().unwrap();
        let path_old = dir.path().join("old.bpc");
        let path_new = dir.path().join("new.bpc");
        std::fs::write(&path_old, synthetic_bpck_bytes()).unwrap(); // frame 31006 only
        std::fs::write(&path_new, synthetic_bpck_bytes_corrupt(31099)).unwrap();

        load_kernel(path_old.to_str().unwrap()).unwrap();
        load_kernel(path_new.to_str().unwrap()).unwrap();

        let err = pck_euler_angles(31099, epc_from_et(500.0)).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("RADIUS"),
            "expected RADIUS error, got: {}",
            msg
        );
        assert!(!msg.contains("not covered"), "error was masked: {}", msg);

        // Restore global state for other tests.
        clear_kernels();
        load_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_registry_load_unload_idempotent() {
        setup_global_test_spice();
        clear_kernels();
        assert!(loaded_kernels().is_empty());

        load_kernel("de440s").unwrap();
        load_kernel("de440s").unwrap(); // idempotent
        assert_eq!(loaded_kernels(), vec!["de440s".to_string()]);

        unload_kernel("de440s").unwrap();
        assert!(loaded_kernels().is_empty());
        assert!(unload_kernel("de440s").is_err()); // not loaded

        // Restore for other tests
        load_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_spk_position_generic() {
        setup_global_test_spice();
        let r = spk_position(NAIF_MOON, NAIF_EARTH, epc_2025()).unwrap();
        let d = r.norm();
        assert!(d > 3.5e8 && d < 4.1e8);
    }

    #[test]
    #[serial]
    fn test_spk_state_matches_position_velocity() {
        setup_global_test_spice();
        let epc = epc_2025();
        let x = spk_state(NAIF_SUN, NAIF_EARTH, epc).unwrap();
        let r = spk_position(NAIF_SUN, NAIF_EARTH, epc).unwrap();
        let v = spk_velocity(NAIF_SUN, NAIF_EARTH, epc).unwrap();
        assert_abs_diff_eq!((x.fixed_rows::<3>(0) - r).norm(), 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!((x.fixed_rows::<3>(3) - v).norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[serial]
    fn test_query_with_no_kernels_auto_initializes() {
        setup_global_test_spice(); // ensures de440s is in the local cache
        clear_kernels();
        // Auto-init loads de440s from cache without an explicit load_kernel
        let r = spk_position(NAIF_SUN, NAIF_EARTH, epc_2025()).unwrap();
        assert!(r.norm() > 1.4e11);
        assert_eq!(loaded_kernels(), vec!["de440s".to_string()]);
    }

    #[test]
    #[serial]
    fn test_query_with_only_pck_loaded_still_auto_initializes_spk() {
        // Regression test: `ensure_default_ephemeris_loaded` must check for
        // an *SPK* kernel specifically, not just any loaded kernel, so that
        // a registry holding only a PCK (e.g. after
        // `load_kernel("moon_pa_de440")`) still auto-loads de440s for
        // `spk_*` queries instead of erroring with "No SPK kernels loaded".
        setup_global_test_spice(); // ensures de440s is in the local cache
        clear_kernels();

        let dir = tempfile::tempdir().unwrap();
        let pck_path = dir.path().join("synthetic.bpc");
        std::fs::write(&pck_path, synthetic_bpck_bytes()).unwrap();
        load_kernel(pck_path.to_str().unwrap()).unwrap();
        assert_eq!(
            loaded_kernels(),
            vec![pck_path.to_str().unwrap().to_string()]
        );

        let r = spk_position(NAIF_SUN, NAIF_EARTH, epc_2025()).unwrap();
        assert!(r.norm() > 1.4e11);
        assert!(loaded_kernels().contains(&"de440s".to_string()));

        // Restore global state for other tests.
        clear_kernels();
        load_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_registry_last_loaded_kernel_wins() {
        // SPICE precedence: the most recently loaded kernel wins for
        // overlapping body pairs. Regression test for the cross-kernel
        // segment ordering in `global_chain`.
        setup_global_test_spice();
        clear_kernels();

        let dir = tempfile::tempdir().unwrap();
        let path_a = dir.path().join("a.bsp");
        let path_b = dir.path().join("b.bsp");
        std::fs::write(&path_a, synthetic_spk_bytes(1.0)).unwrap();
        std::fs::write(&path_b, synthetic_spk_bytes(2.0)).unwrap();

        load_kernel(path_a.to_str().unwrap()).unwrap();
        load_kernel(path_b.to_str().unwrap()).unwrap();

        // Later-loaded kernel B takes precedence (2.0 km = 2.0e3 m).
        let r = spk_position(10, NAIF_SSB, epc_2025()).unwrap();
        assert_abs_diff_eq!(r[0], 2.0e3, epsilon = 1e-9);

        // Unloading B invalidates the chain cache and falls back to A.
        unload_kernel(path_b.to_str().unwrap()).unwrap();
        let r = spk_position(10, NAIF_SSB, epc_2025()).unwrap();
        assert_abs_diff_eq!(r[0], 1.0e3, epsilon = 1e-9);

        // Restore global state for other tests.
        clear_kernels();
        load_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_global_chain_falls_back_to_epoch_aware_chain() {
        // Cross-kernel counterpart of `test_spk_position_falls_back_to_epoch_aware_chain`
        // (spk.rs): three kernels providing a direct A rel B link [0,100]
        // plus a two-hop A rel C [0,200] / C rel B [0,200] alternative.
        // `global_chain`'s cached topology-only resolution always prefers
        // the 1-hop direct link; querying an epoch only the two-hop path
        // covers must transparently fall back instead of erroring.
        setup_global_test_spice();
        clear_kernels();

        let dir = tempfile::tempdir().unwrap();
        let path_direct = dir.path().join("direct.bsp");
        let path_ac = dir.path().join("ac.bsp");
        let path_cb = dir.path().join("cb.bsp");
        std::fs::write(
            &path_direct,
            synthetic_spk_bytes_for(10, 0, 0.0, 100.0, 7.0),
        )
        .unwrap();
        std::fs::write(&path_ac, synthetic_spk_bytes_for(10, 3, 0.0, 200.0, 2.0)).unwrap();
        std::fs::write(&path_cb, synthetic_spk_bytes_for(3, 0, 0.0, 200.0, 3.0)).unwrap();

        load_kernel(path_direct.to_str().unwrap()).unwrap();
        load_kernel(path_ac.to_str().unwrap()).unwrap();
        load_kernel(path_cb.to_str().unwrap()).unwrap();

        // et=50: covered by the (cached) direct link -> direct value.
        let r = spk_position(10, 0, epc_from_et(50.0)).unwrap();
        assert_abs_diff_eq!(r[0], 7.0e3, epsilon = 1e-6);

        // et=150: direct link out of coverage -> falls back to the
        // two-hop path (2.0 + 3.0 = 5.0 km).
        let r = spk_position(10, 0, epc_from_et(150.0)).unwrap();
        assert_abs_diff_eq!(r[0], 5.0e3, epsilon = 1e-6);

        // et=300: neither path covers -> error names target, center, and
        // mentions coverage.
        let err = spk_position(10, 0, epc_from_et(300.0)).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("10") && msg.contains("coverage"));

        // Restore global state for other tests.
        clear_kernels();
        load_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_spk_position_in_kernel_scoped() {
        setup_global_test_spice();
        let r = spk_position_in_kernel("de440s", NAIF_MOON, NAIF_EARTH, epc_2025()).unwrap();
        assert!(r.norm() > 3.5e8 && r.norm() < 4.1e8);
    }

    #[test]
    #[serial]
    fn test_concurrent_queries() {
        setup_global_test_spice();
        load_kernel("de440s").unwrap();
        let handles: Vec<_> = (0..8)
            .map(|i| {
                std::thread::spawn(move || {
                    let epc = Epoch::from_date(2025, 1, 1 + i, TimeSystem::UTC);
                    spk_position(NAIF_MOON, NAIF_EARTH, epc).unwrap().norm()
                })
            })
            .collect();
        for h in handles {
            let d = h.join().unwrap();
            assert!(d > 3.4e8 && d < 4.1e8);
        }
    }

    #[test]
    #[serial]
    fn test_pck_query_without_pck_errors() {
        setup_global_test_spice();
        let err = pck_euler_angles(31006, epc_2025()).unwrap_err();
        assert!(format!("{}", err).contains("31006"));
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial]
    fn test_load_kernel_resolves_moon_pa_de440_network() {
        setup_global_test_spice();
        clear_kernels();

        load_kernel("moon_pa_de440").unwrap();
        assert_eq!(loaded_kernels(), vec!["moon_pa_de440".to_string()]);

        // Frame class 31008 (MOON_PA_DE440) has coverage at 2025-01-01.
        let r = pck_rotation_matrix(31008, epc_2025()).unwrap();
        let rtr = r.transpose() * r;
        assert_abs_diff_eq!((rtr - Matrix3::identity()).norm(), 0.0, epsilon = 1e-9);

        // Restore global state for other tests.
        clear_kernels();
        load_kernel("de440s").unwrap();
    }

    #[test]
    fn test_naif_id_constants() {
        assert_eq!(NAIF_SSB, 0);
        assert_eq!(NAIF_EMB, 3);
        assert_eq!(NAIF_SUN, 10);
        assert_eq!(NAIF_MOON, 301);
        assert_eq!(NAIF_EARTH, 399);
    }

    #[test]
    fn test_spk_kernel_name() {
        use super::super::kernels::SPKKernel;
        assert_eq!(SPKKernel::DE440s.name(), "de440s");
        assert_eq!(SPKKernel::DE440.name(), "de440");
    }
}
