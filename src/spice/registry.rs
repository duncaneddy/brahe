/*!
 * Global SPICE kernel registry and generic SPK/PCK query API.
 *
 * Owns a process-wide set of loaded SPK/PCK kernels keyed by the
 * name-or-path string used to load them. Kernels are auto-detected as SPK
 * or PCK from the DAF ID word. Cross-kernel `spk_*` queries build a chain
 * of segments spanning all loaded SPK kernels, with more recently loaded
 * kernels taking precedence for overlapping body pairs (matching SPICE's
 * own "last loaded wins" convention). `pck_*` queries search loaded PCK
 * kernels newest-first for a frame with coverage at the requested epoch.
 */

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};

use nalgebra::{Matrix3, Vector3, Vector6};
use once_cell::sync::Lazy;

use crate::datasets::naif::download_de_kernel;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

use super::daf::DafFile;
use super::pck::BPCK;
use super::spk::{
    ChainLink, SPK, evaluate_chain_position, evaluate_chain_state, evaluate_chain_velocity,
    resolve_chain,
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
fn resolve_kernel_source(name_or_path: &str) -> Result<std::path::PathBuf, BraheError> {
    match name_or_path {
        // Known downloadable kernel names resolve via the NAIF dataset cache
        "de430" | "de432s" | "de435" | "de438" | "de440" | "de440s" | "de442" | "de442s" => {
            download_de_kernel(name_or_path, None)
        }
        other => {
            let path = Path::new(other);
            if path.exists() {
                Ok(path.to_path_buf())
            } else {
                Err(BraheError::IoError(format!(
                    "Kernel '{}' is neither a known kernel name nor an existing file path",
                    other
                )))
            }
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
/// any other string is treated as a file path.
///
/// # Arguments
/// - `name_or_path`: A known DE kernel name, or a path to a `.bsp`/`.bpc` file
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

/// Resolve (and cache) the cross-kernel segment chain for `target` rel
/// `center`, spanning all loaded SPK kernels. Later-loaded kernels take
/// precedence: both across kernels and within a single kernel, segments
/// are offered to [`resolve_chain`] newest-first so ties resolve to the
/// most recently loaded data.
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
    // Later-loaded kernels take precedence: collect segments newest-first,
    // and within each kernel reverse its own segment order too, so
    // last-listed-wins semantics apply consistently at both levels.
    let mut segments = Vec::new();
    for name in reg.load_order.iter().rev() {
        if let Some(Kernel::Spk(spk)) = reg.kernels.get(name) {
            segments.extend(spk.segments().iter().rev().cloned());
        }
    }
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

/// Load `de440s` if no kernels are currently loaded, preserving the
/// library's historical lazy-initialization behavior for `spk_*` queries.
fn ensure_default_ephemeris_loaded() -> Result<(), BraheError> {
    let has_kernels = !GLOBAL_SPICE.read().unwrap().load_order.is_empty();
    if !has_kernels {
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
/// Auto-initializes with `de440s` if no kernels are loaded.
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
    Ok(evaluate_chain_position(&global_chain(target, center)?, et)? * 1.0e3)
}

/// Velocity of `target` relative to `center` at `epc`, resolved across all
/// loaded SPK kernels.
///
/// Auto-initializes with `de440s` if no kernels are loaded.
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
    Ok(evaluate_chain_velocity(&global_chain(target, center)?, et)? * 1.0e3)
}

/// Position and velocity of `target` relative to `center` at `epc`,
/// resolved across all loaded SPK kernels.
///
/// Auto-initializes with `de440s` if no kernels are loaded.
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
    let (r, v) = evaluate_chain_state(&global_chain(target, center)?, et)?;
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
/// single named kernel only (no cross-kernel chain resolution).
///
/// Loads `kernel_name` first if not already loaded.
///
/// # Arguments
/// - `kernel_name`: A known DE kernel name, or a path to a `.bsp` file
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
/// single named kernel only (no cross-kernel chain resolution).
///
/// Loads `kernel_name` first if not already loaded.
///
/// # Arguments
/// - `kernel_name`: A known DE kernel name, or a path to a `.bsp` file
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
/// queried from a single named kernel only (no cross-kernel chain
/// resolution).
///
/// Loads `kernel_name` first if not already loaded.
///
/// # Arguments
/// - `kernel_name`: A known DE kernel name, or a path to a `.bsp` file
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
            let frame_ids = pck.frame_ids();
            if frame_ids.contains(&frame_id)
                && let Ok(result) = query(pck, frame_id, et)
            {
                return Ok(result);
            }
            for f in frame_ids {
                if !available_frames.contains(&f) {
                    available_frames.push(f);
                }
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
/// - `frame_id`: Body-frame class ID (e.g. 31006 for MOON_PA_DE440)
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
/// let (angles, rates) = pck_euler_angles(31006, epc).unwrap();
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
/// - `frame_id`: Body-frame class ID (e.g. 31006 for MOON_PA_DE440)
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
/// let r = pck_rotation_matrix(31006, epc).unwrap();
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
