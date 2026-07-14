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

use nalgebra::{Vector3, Vector6};
use once_cell::sync::Lazy;

use crate::attitude::{EulerAngle, Quaternion, RotationMatrix};
use crate::datasets::naif::download_spice_kernel;
use crate::time::Epoch;
use crate::utils::BraheError;

use super::daf::DAFFile;
use super::kernels::{KernelSource, SPICEKernel};
use super::naif_id::{FrameId, NAIFId};
use super::pck::BPCK;
use super::segments::{ChebyshevSegment, is_coverage_error};
use super::spk::{
    ChainLink, SPK, evaluate_chain_position, evaluate_chain_state, evaluate_chain_velocity,
    evaluate_with_epoch_fallback, resolve_chain,
};

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
    epc.spice_et()
}

/// Resolve a [`KernelSource`] to a local file path, downloading (and
/// caching) a known [`SPICEKernel`] via the NAIF dataset cache or validating
/// that a bring-your-own path exists.
fn resolve_kernel_path(source: &KernelSource) -> Result<std::path::PathBuf, BraheError> {
    match source {
        KernelSource::Kernel(kernel) => download_spice_kernel(*kernel, None),
        KernelSource::Path(path) => {
            let p = Path::new(path);
            if p.exists() {
                Ok(p.to_path_buf())
            } else {
                Err(BraheError::IoError(format!(
                    "Kernel '{}' is neither a known kernel name nor an existing file path",
                    path
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
/// Idempotent: calling with a source that is already loaded is a no-op.
/// Accepts anything convertible into a [`KernelSource`]: a known kernel
/// name string (`"de440s"`, `"moon_pa_de440"`, ...) or [`SPICEKernel`] is
/// downloaded and cached via [`crate::datasets::naif::download_spice_kernel`]; any
/// other string is treated as a file path. The registry is keyed by
/// [`KernelSource::key`] (the kernel name or the path string), so loading
/// by name and by [`SPICEKernel`] refer to the same entry.
///
/// # Arguments
/// - `kernel`: A known kernel name/[`SPICEKernel`], or a path to a `.bsp`/`.bpc` file
///
/// # Returns
/// - `Ok(())` on success, or `BraheError` if the kernel cannot be resolved,
///   read, or parsed
///
/// # Examples
/// ```no_run
/// use brahe::spice::load_spice_kernel;
///
/// load_spice_kernel("de440s").expect("Failed to load DE440s");
/// ```
pub fn load_spice_kernel(kernel: impl Into<KernelSource>) -> Result<(), BraheError> {
    let source = kernel.into();
    let key = source.key().to_string();

    // Idempotent short-circuit (read lock only) for the common case.
    {
        let reg = GLOBAL_SPICE.read().unwrap();
        if reg.kernels.contains_key(&key) {
            return Ok(());
        }
    }

    let path = resolve_kernel_path(&source)?;
    let daf = DAFFile::from_file(&path)?;
    let kernel = match daf.id_word.as_str() {
        "DAF/SPK" => Kernel::Spk(Arc::new(SPK::from_daf(daf)?)),
        "DAF/PCK" => Kernel::Pck(Arc::new(BPCK::from_daf(daf)?)),
        other => {
            return Err(BraheError::IoError(format!(
                "Unrecognized DAF ID word '{}' in kernel '{}'; expected 'DAF/SPK' or 'DAF/PCK'",
                other, key
            )));
        }
    };

    let mut reg = GLOBAL_SPICE.write().unwrap();
    // Double-checked locking: another thread may have loaded it while we
    // resolved and parsed the file.
    if reg.kernels.contains_key(&key) {
        return Ok(());
    }
    reg.kernels.insert(key.clone(), kernel);
    reg.load_order.push(key);
    reg.chain_cache.clear();
    Ok(())
}

/// Unload a kernel previously loaded via [`load_spice_kernel`].
///
/// # Arguments
/// - `kernel`: The same kernel name/[`SPICEKernel`]/path originally passed to
///   [`load_spice_kernel`]
///
/// # Returns
/// - `Ok(())` on success, or `BraheError` if the kernel is not loaded
///
/// # Examples
/// ```no_run
/// use brahe::spice::{load_spice_kernel, unload_spice_kernel};
///
/// load_spice_kernel("de440s").unwrap();
/// unload_spice_kernel("de440s").expect("Failed to unload DE440s");
/// ```
pub fn unload_spice_kernel(kernel: impl Into<KernelSource>) -> Result<(), BraheError> {
    let key = kernel.into().key().to_string();
    let mut reg = GLOBAL_SPICE.write().unwrap();
    if reg.kernels.remove(&key).is_none() {
        return Err(BraheError::Error(format!(
            "Kernel '{}' is not currently loaded",
            key
        )));
    }
    reg.load_order.retain(|k| k != &key);
    reg.chain_cache.clear();
    Ok(())
}

/// Unload all kernels from the global registry.
///
/// # Examples
/// ```no_run
/// use brahe::spice::{load_spice_kernel, clear_spice_kernels};
///
/// load_spice_kernel("de440s").unwrap();
/// clear_spice_kernels();
/// ```
pub fn clear_spice_kernels() {
    let mut reg = GLOBAL_SPICE.write().unwrap();
    reg.kernels.clear();
    reg.load_order.clear();
    reg.chain_cache.clear();
}

/// Names/paths of currently loaded kernels, in load order.
///
/// # Returns
/// - Kernel names/paths in the order they were passed to [`load_spice_kernel`]
///
/// # Examples
/// ```no_run
/// use brahe::spice::{load_spice_kernel, loaded_spice_kernels};
///
/// load_spice_kernel("de440s").unwrap();
/// assert_eq!(loaded_spice_kernels(), vec!["de440s".to_string()]);
/// ```
pub fn loaded_spice_kernels() -> Vec<String> {
    GLOBAL_SPICE.read().unwrap().load_order.clone()
}

/// True when a loaded kernel's registry key contains `fragment`.
///
/// A registry key is the known-kernel name (`"de440s"`, `"moon_pa_de440"`,
/// ...) or, for bring-your-own kernels, the file-path string it was loaded
/// from — so a fragment like `"moon_pa_de440"` matches both a name-loaded
/// kernel and a path-loaded copy of the same file. Allocation-free
/// read-lock check, cheap enough for per-call guards (unlike cloning
/// [`loaded_spice_kernels`]); unlike a `OnceLock` latch it stays correct across
/// [`clear_spice_kernels`]/[`unload_spice_kernel`].
///
/// # Arguments
/// - `fragment`: Substring to search for in each loaded kernel's key
///
/// # Returns
/// - `true` if any loaded kernel's key contains `fragment`
///
/// # Examples
/// ```no_run
/// use brahe::spice::{kernel_is_loaded, load_spice_kernel};
///
/// load_spice_kernel("de440s").unwrap();
/// assert!(kernel_is_loaded("de440s"));
/// assert!(!kernel_is_loaded("de440x"));
/// ```
pub fn kernel_is_loaded(fragment: &str) -> bool {
    GLOBAL_SPICE
        .read()
        .unwrap()
        .load_order
        .iter()
        .any(|k| k.contains(fragment))
}

/// Kernels loaded by [`load_common_spice_kernels`]: `de440s` (planetary ephemeris)
/// and `moon_pa_de440` (lunar principal-axes orientation).
pub(crate) const COMMON_SPICE_KERNELS: &[SPICEKernel] =
    &[SPICEKernel::DE440s, SPICEKernel::MoonPaDe440];

/// Kernels loaded by [`load_all_spice_kernels`]: [`COMMON_SPICE_KERNELS`] plus every
/// satellite ephemeris kernel brahe knows how to download.
pub(crate) const ALL_SPICE_KERNELS: &[SPICEKernel] = &[
    SPICEKernel::DE440s,
    SPICEKernel::MoonPaDe440,
    SPICEKernel::Mar099s,
    SPICEKernel::Jup365,
    SPICEKernel::Sat441,
    SPICEKernel::Ura184,
    SPICEKernel::Nep097,
    SPICEKernel::Plu060,
];

/// Load the kernels most applications need: `de440s` (planetary ephemeris)
/// and `moon_pa_de440` (lunar principal-axes orientation).
///
/// ~46 MB total on first download; cached thereafter. Each kernel load is
/// idempotent, so calling this alongside other [`load_spice_kernel`] calls is
/// safe.
///
/// Loading is not atomic: on error, kernels already loaded before the
/// failure remain resident, so the call can be safely retried.
///
/// # Returns
/// - `Ok(())` on success, or `BraheError` if a kernel cannot be downloaded,
///   read, or parsed
///
/// # Examples
/// ```no_run
/// use brahe::spice::load_common_spice_kernels;
///
/// load_common_spice_kernels().expect("Failed to load common kernels");
/// ```
pub fn load_common_spice_kernels() -> Result<(), BraheError> {
    for kernel in COMMON_SPICE_KERNELS {
        load_spice_kernel(*kernel)?;
    }
    Ok(())
}

/// Load every kernel brahe knows how to download: `de440s`, `moon_pa_de440`,
/// and the satellite ephemeris kernels `mar099s`, `jup365`, `sat441`, `ura184`,
/// `nep097`, `plu060`.
///
/// ~2.5 GB total on first download; cached thereafter. Prefer
/// [`load_common_spice_kernels`] unless outer-planet body centers or moons are
/// needed.
///
/// Loading is not atomic: on error, kernels already loaded before the
/// failure remain resident, so the call can be safely retried.
///
/// # Returns
/// - `Ok(())` on success, or `BraheError` if a kernel cannot be downloaded,
///   read, or parsed
///
/// # Examples
/// ```no_run
/// use brahe::spice::load_all_spice_kernels;
///
/// load_all_spice_kernels().expect("Failed to load all kernels");
/// ```
pub fn load_all_spice_kernels() -> Result<(), BraheError> {
    for kernel in ALL_SPICE_KERNELS {
        load_spice_kernel(*kernel)?;
    }
    Ok(())
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
            "No SPK kernels loaded; call load_spice_kernel(...) or load_common_spice_kernels() first"
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
/// registry holding only PCK kernels (e.g. after `load_spice_kernel("moon_pa_de440")`)
/// still auto-initializes the default ephemeris.
fn ensure_default_ephemeris_loaded() -> Result<(), BraheError> {
    let has_spk = GLOBAL_SPICE
        .read()
        .unwrap()
        .kernels
        .values()
        .any(|k| matches!(k, Kernel::Spk(_)));
    if !has_spk {
        load_spice_kernel("de440s")?;
    }
    Ok(())
}

/// Ensures the kernels needed to resolve queries among `ids` are loaded:
/// the default planetary DE ephemeris when no SPK is resident yet
/// (**before** any satellite kernel, so a satellite load cannot suppress
/// the DE auto-initialization), then each satellite-system ephemeris
/// kernel for satellite-range IDs.
///
/// Satellite-kernel load failures are ignored: the ID may instead be
/// covered by a bring-your-own kernel already in the registry, in which
/// case the subsequent query succeeds; if nothing covers it, the query
/// surfaces the chain-resolution error.
///
/// An explicitly loaded alternative DE kernel (e.g. `de440`) keeps its
/// precedence — the default is only loaded into an SPK-empty registry.
///
/// # Arguments
/// - `ids`: NAIF IDs the caller is about to query through the registry
///
/// # Returns
/// - `Ok(())` on success, or `BraheError` if the default DE ephemeris
///   cannot be loaded
pub(crate) fn ensure_bodies_loadable(ids: &[i32]) -> Result<(), BraheError> {
    use super::positions::{de_kernel_covers, satellite_system_kernel};

    ensure_default_ephemeris_loaded()?;
    for &id in ids {
        if !de_kernel_covers(id)
            && (400..=999).contains(&id)
            && let Some(kernel) = satellite_system_kernel(id / 100)
        {
            let _ = load_spice_kernel(kernel);
        }
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
/// use brahe::spice::{NAIFId, spk_position, load_common_spice_kernels};
/// use brahe::time::{Epoch, TimeSystem};
///
/// load_common_spice_kernels().unwrap();
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let r_moon = spk_position(NAIFId::Moon, NAIFId::Earth, epc).unwrap();
/// ```
pub fn spk_position(
    target: impl Into<NAIFId>,
    center: impl Into<NAIFId>,
    epc: Epoch,
) -> Result<Vector3<f64>, BraheError> {
    let target = target.into().id();
    let center = center.into().id();
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
/// use brahe::spice::{NAIFId, spk_velocity, load_common_spice_kernels};
/// use brahe::time::{Epoch, TimeSystem};
///
/// load_common_spice_kernels().unwrap();
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let v_moon = spk_velocity(NAIFId::Moon, NAIFId::Earth, epc).unwrap();
/// ```
pub fn spk_velocity(
    target: impl Into<NAIFId>,
    center: impl Into<NAIFId>,
    epc: Epoch,
) -> Result<Vector3<f64>, BraheError> {
    let target = target.into().id();
    let center = center.into().id();
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
/// use brahe::spice::{NAIFId, spk_state, load_common_spice_kernels};
/// use brahe::time::{Epoch, TimeSystem};
///
/// load_common_spice_kernels().unwrap();
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let x_moon = spk_state(NAIFId::Moon, NAIFId::Earth, epc).unwrap();
/// ```
pub fn spk_state(
    target: impl Into<NAIFId>,
    center: impl Into<NAIFId>,
    epc: Epoch,
) -> Result<Vector6<f64>, BraheError> {
    let target = target.into().id();
    let center = center.into().id();
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

/// Fetch the loaded `Kernel::Spk` entry for `kernel`, loading it first if
/// absent.
fn spk_kernel_for_query(kernel: KernelSource) -> Result<Arc<SPK>, BraheError> {
    let key = kernel.key().to_string();
    load_spice_kernel(kernel)?;
    let reg = GLOBAL_SPICE.read().unwrap();
    match reg.kernels.get(&key) {
        Some(Kernel::Spk(spk)) => Ok(Arc::clone(spk)),
        Some(Kernel::Pck(_)) => Err(BraheError::Error(format!(
            "Kernel '{}' is a binary PCK, not an SPK",
            key
        ))),
        // Reachable if another thread unloads the kernel between the
        // load_spice_kernel call above and this lookup.
        None => Err(BraheError::Error(format!(
            "Kernel '{}' was unloaded concurrently during query",
            key
        ))),
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
/// - `kernel`: A known kernel name/[`SPICEKernel`], or a path to a `.bsp`
///   file
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
/// use brahe::spice::{NAIFId, spk_position_from_kernel};
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let r_moon = spk_position_from_kernel("de440s", NAIFId::Moon, NAIFId::Earth, epc).unwrap();
/// ```
pub fn spk_position_from_kernel(
    kernel: impl Into<KernelSource>,
    target: impl Into<NAIFId>,
    center: impl Into<NAIFId>,
    epc: Epoch,
) -> Result<Vector3<f64>, BraheError> {
    let target = target.into().id();
    let center = center.into().id();
    let spk = spk_kernel_for_query(kernel.into())?;
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
/// - `kernel`: A known kernel name/[`SPICEKernel`], or a path to a `.bsp`
///   file
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
/// use brahe::spice::{NAIFId, spk_velocity_from_kernel};
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let v_moon = spk_velocity_from_kernel("de440s", NAIFId::Moon, NAIFId::Earth, epc).unwrap();
/// ```
pub fn spk_velocity_from_kernel(
    kernel: impl Into<KernelSource>,
    target: impl Into<NAIFId>,
    center: impl Into<NAIFId>,
    epc: Epoch,
) -> Result<Vector3<f64>, BraheError> {
    let target = target.into().id();
    let center = center.into().id();
    let spk = spk_kernel_for_query(kernel.into())?;
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
/// - `kernel`: A known kernel name/[`SPICEKernel`], or a path to a `.bsp`
///   file
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
/// use brahe::spice::{NAIFId, spk_state_from_kernel};
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let x_moon = spk_state_from_kernel("de440s", NAIFId::Moon, NAIFId::Earth, epc).unwrap();
/// ```
pub fn spk_state_from_kernel(
    kernel: impl Into<KernelSource>,
    target: impl Into<NAIFId>,
    center: impl Into<NAIFId>,
    epc: Epoch,
) -> Result<Vector6<f64>, BraheError> {
    let target = target.into().id();
    let center = center.into().id();
    let spk = spk_kernel_for_query(kernel.into())?;
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
/// via [`load_spice_kernel`] first.
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
/// use brahe::spice::{FrameId, load_spice_kernel, pck_euler_angles};
/// use brahe::time::{Epoch, TimeSystem};
///
/// load_spice_kernel("/path/to/moon_pa_de440_200625.bpc").unwrap();
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let (angles, rates) = pck_euler_angles(FrameId::MoonPaDe440, epc).unwrap();
/// ```
pub fn pck_euler_angles(
    frame_id: impl Into<FrameId>,
    epc: Epoch,
) -> Result<(Vector3<f64>, Vector3<f64>), BraheError> {
    let frame_id = frame_id.into().id();
    pck_query(frame_id, epc, |pck, f, et| pck.euler_angles(f, et))
}

/// 3-1-3 Euler angle of the body-fixed frame `frame_id` relative to its
/// segment reference frame (ICRF for DE440-era kernels), searching loaded
/// PCK kernels newest-first.
///
/// PCKs are never auto-initialized; a PCK kernel must be explicitly loaded
/// via [`load_spice_kernel`] first.
///
/// # Arguments
/// - `frame_id`: Body-frame class ID (e.g. 31008 for MOON_PA_DE440)
/// - `epc`: Epoch at which to evaluate the orientation
///
/// # Returns
/// - `EulerAngle` (order `ZXZ`, radians): ICRF to body-fixed orientation
///
/// # Examples
/// ```no_run
/// use brahe::spice::{FrameId, load_spice_kernel, pck_euler_angle};
/// use brahe::time::{Epoch, TimeSystem};
///
/// load_spice_kernel("/path/to/moon_pa_de440_200625.bpc").unwrap();
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let e = pck_euler_angle(FrameId::MoonPaDe440, epc).unwrap();
/// ```
pub fn pck_euler_angle(frame_id: impl Into<FrameId>, epc: Epoch) -> Result<EulerAngle, BraheError> {
    let frame_id = frame_id.into().id();
    pck_query(frame_id, epc, |pck, f, et| pck.euler_angle(f, et))
}

/// Time derivatives of the 3-1-3 Euler angles of the body-fixed frame
/// `frame_id`, searching loaded PCK kernels newest-first.
///
/// PCKs are never auto-initialized; a PCK kernel must be explicitly loaded
/// via [`load_spice_kernel`] first.
///
/// # Arguments
/// - `frame_id`: Body-frame class ID (e.g. 31008 for MOON_PA_DE440)
/// - `epc`: Epoch at which to evaluate the orientation
///
/// # Returns
/// - `[phi_dot, delta_dot, w_dot]` in [rad/s]
///
/// # Examples
/// ```no_run
/// use brahe::spice::{FrameId, load_spice_kernel, pck_euler_rates};
/// use brahe::time::{Epoch, TimeSystem};
///
/// load_spice_kernel("/path/to/moon_pa_de440_200625.bpc").unwrap();
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let rates = pck_euler_rates(FrameId::MoonPaDe440, epc).unwrap();
/// ```
pub fn pck_euler_rates(
    frame_id: impl Into<FrameId>,
    epc: Epoch,
) -> Result<Vector3<f64>, BraheError> {
    let frame_id = frame_id.into().id();
    pck_query(frame_id, epc, |pck, f, et| pck.euler_rates(f, et))
}

/// Typed Euler angle and its rates for the body-fixed frame `frame_id`,
/// from a single shared segment lookup, searching loaded PCK kernels
/// newest-first.
///
/// PCKs are never auto-initialized; a PCK kernel must be explicitly loaded
/// via [`load_spice_kernel`] first.
///
/// # Arguments
/// - `frame_id`: Body-frame class ID (e.g. 31008 for MOON_PA_DE440)
/// - `epc`: Epoch at which to evaluate the orientation
///
/// # Returns
/// - `(angle, rates)`: `angle` is the `EulerAngle` (order `ZXZ`, radians);
///   `rates` are `[phi_dot, delta_dot, w_dot]` in [rad/s]
///
/// # Examples
/// ```no_run
/// use brahe::spice::{FrameId, load_spice_kernel, pck_euler_angle_and_rates};
/// use brahe::time::{Epoch, TimeSystem};
///
/// load_spice_kernel("/path/to/moon_pa_de440_200625.bpc").unwrap();
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let (angle, rates) = pck_euler_angle_and_rates(FrameId::MoonPaDe440, epc).unwrap();
/// ```
pub fn pck_euler_angle_and_rates(
    frame_id: impl Into<FrameId>,
    epc: Epoch,
) -> Result<(EulerAngle, Vector3<f64>), BraheError> {
    let frame_id = frame_id.into().id();
    pck_query(frame_id, epc, |pck, f, et| pck.euler_angle_and_rates(f, et))
}

/// Orientation of the body-fixed frame `frame_id` relative to its segment
/// reference frame (ICRF for DE440-era kernels), as a unit `Quaternion`,
/// searching loaded PCK kernels newest-first.
///
/// PCKs are never auto-initialized; a PCK kernel must be explicitly loaded
/// via [`load_spice_kernel`] first.
///
/// # Arguments
/// - `frame_id`: Body-frame class ID (e.g. 31008 for MOON_PA_DE440)
/// - `epc`: Epoch at which to evaluate the orientation
///
/// # Returns
/// - Unit `Quaternion` (ICRF to body-fixed). Dimensionless.
///
/// # Examples
/// ```no_run
/// use brahe::spice::{FrameId, load_spice_kernel, pck_quaternion};
/// use brahe::time::{Epoch, TimeSystem};
///
/// load_spice_kernel("/path/to/moon_pa_de440_200625.bpc").unwrap();
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let q = pck_quaternion(FrameId::MoonPaDe440, epc).unwrap();
/// ```
pub fn pck_quaternion(frame_id: impl Into<FrameId>, epc: Epoch) -> Result<Quaternion, BraheError> {
    let frame_id = frame_id.into().id();
    pck_query(frame_id, epc, |pck, f, et| pck.quaternion(f, et))
}

/// Rotation matrix from the segment reference frame (ICRF) to the
/// body-fixed frame `frame_id`, searching loaded PCK kernels newest-first.
///
/// PCKs are never auto-initialized; a PCK kernel must be explicitly loaded
/// via [`load_spice_kernel`] first.
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
/// use brahe::spice::{FrameId, load_spice_kernel, pck_rotation_matrix};
/// use brahe::time::{Epoch, TimeSystem};
///
/// load_spice_kernel("/path/to/moon_pa_de440_200625.bpc").unwrap();
/// let epc = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
/// let r = pck_rotation_matrix(FrameId::MoonPaDe440, epc).unwrap();
/// ```
pub fn pck_rotation_matrix(
    frame_id: impl Into<FrameId>,
    epc: Epoch,
) -> Result<RotationMatrix, BraheError> {
    let frame_id = frame_id.into().id();
    pck_query(frame_id, epc, |pck, f, et| pck.rotation_matrix(f, et))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use nalgebra::Matrix3;
    use serial_test::serial;

    use super::*;
    use crate::attitude::ToAttitude;
    use crate::time::TimeSystem;
    use crate::utils::testing::{CacheRedirect, setup_global_test_spice};

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
        clear_spice_kernels();

        let dir = tempfile::tempdir().unwrap();
        let path_old = dir.path().join("old.bpc");
        let path_new = dir.path().join("new.bpc");
        std::fs::write(&path_old, synthetic_bpck_bytes()).unwrap(); // frame 31006 only
        std::fs::write(&path_new, synthetic_bpck_bytes_corrupt(31099)).unwrap();

        load_spice_kernel(path_old.to_str().unwrap()).unwrap();
        load_spice_kernel(path_new.to_str().unwrap()).unwrap();

        let err = pck_euler_angles(31099, epc_from_et(500.0)).unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("RADIUS"),
            "expected RADIUS error, got: {}",
            msg
        );
        assert!(!msg.contains("not covered"), "error was masked: {}", msg);

        // Restore global state for other tests.
        clear_spice_kernels();
        load_spice_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_registry_load_unload_idempotent() {
        setup_global_test_spice();
        clear_spice_kernels();
        assert!(loaded_spice_kernels().is_empty());

        load_spice_kernel("de440s").unwrap();
        load_spice_kernel("de440s").unwrap(); // idempotent
        assert_eq!(loaded_spice_kernels(), vec!["de440s".to_string()]);

        unload_spice_kernel("de440s").unwrap();
        assert!(loaded_spice_kernels().is_empty());
        assert!(unload_spice_kernel("de440s").is_err()); // not loaded

        // Restore for other tests
        load_spice_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_spk_position_generic() {
        setup_global_test_spice();
        let r = spk_position(NAIFId::Moon, NAIFId::Earth, epc_2025()).unwrap();
        let d = r.norm();
        assert!(d > 3.5e8 && d < 4.1e8);
    }

    #[test]
    #[serial]
    fn test_spk_state_matches_position_velocity() {
        setup_global_test_spice();
        let epc = epc_2025();
        let x = spk_state(NAIFId::Sun, NAIFId::Earth, epc).unwrap();
        let r = spk_position(NAIFId::Sun, NAIFId::Earth, epc).unwrap();
        let v = spk_velocity(NAIFId::Sun, NAIFId::Earth, epc).unwrap();
        assert_abs_diff_eq!((x.fixed_rows::<3>(0) - r).norm(), 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!((x.fixed_rows::<3>(3) - v).norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[serial]
    fn test_query_with_no_kernels_auto_initializes() {
        setup_global_test_spice(); // ensures de440s is in the local cache
        clear_spice_kernels();
        // Auto-init loads de440s from cache without an explicit load_spice_kernel
        let r = spk_position(NAIFId::Sun, NAIFId::Earth, epc_2025()).unwrap();
        assert!(r.norm() > 1.4e11);
        assert_eq!(loaded_spice_kernels(), vec!["de440s".to_string()]);
    }

    #[test]
    #[serial]
    fn test_query_with_only_pck_loaded_still_auto_initializes_spk() {
        // Regression test: `ensure_default_ephemeris_loaded` must check for
        // an *SPK* kernel specifically, not just any loaded kernel, so that
        // a registry holding only a PCK (e.g. after
        // `load_spice_kernel("moon_pa_de440")`) still auto-loads de440s for
        // `spk_*` queries instead of erroring with "No SPK kernels loaded".
        setup_global_test_spice(); // ensures de440s is in the local cache
        clear_spice_kernels();

        let dir = tempfile::tempdir().unwrap();
        let pck_path = dir.path().join("synthetic.bpc");
        std::fs::write(&pck_path, synthetic_bpck_bytes()).unwrap();
        load_spice_kernel(pck_path.to_str().unwrap()).unwrap();
        assert_eq!(
            loaded_spice_kernels(),
            vec![pck_path.to_str().unwrap().to_string()]
        );

        let r = spk_position(NAIFId::Sun, NAIFId::Earth, epc_2025()).unwrap();
        assert!(r.norm() > 1.4e11);
        assert!(loaded_spice_kernels().contains(&"de440s".to_string()));

        // Restore global state for other tests.
        clear_spice_kernels();
        load_spice_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_registry_last_loaded_kernel_wins() {
        // SPICE precedence: the most recently loaded kernel wins for
        // overlapping body pairs. Regression test for the cross-kernel
        // segment ordering in `global_chain`.
        setup_global_test_spice();
        clear_spice_kernels();

        let dir = tempfile::tempdir().unwrap();
        let path_a = dir.path().join("a.bsp");
        let path_b = dir.path().join("b.bsp");
        std::fs::write(&path_a, synthetic_spk_bytes(1.0)).unwrap();
        std::fs::write(&path_b, synthetic_spk_bytes(2.0)).unwrap();

        load_spice_kernel(path_a.to_str().unwrap()).unwrap();
        load_spice_kernel(path_b.to_str().unwrap()).unwrap();

        // Later-loaded kernel B takes precedence (2.0 km = 2.0e3 m).
        let r = spk_position(10, NAIFId::SolarSystemBarycenter, epc_2025()).unwrap();
        assert_abs_diff_eq!(r[0], 2.0e3, epsilon = 1e-9);

        // Unloading B invalidates the chain cache and falls back to A.
        unload_spice_kernel(path_b.to_str().unwrap()).unwrap();
        let r = spk_position(10, NAIFId::SolarSystemBarycenter, epc_2025()).unwrap();
        assert_abs_diff_eq!(r[0], 1.0e3, epsilon = 1e-9);

        // Restore global state for other tests.
        clear_spice_kernels();
        load_spice_kernel("de440s").unwrap();
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
        clear_spice_kernels();

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

        load_spice_kernel(path_direct.to_str().unwrap()).unwrap();
        load_spice_kernel(path_ac.to_str().unwrap()).unwrap();
        load_spice_kernel(path_cb.to_str().unwrap()).unwrap();

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
        clear_spice_kernels();
        load_spice_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_spk_position_from_kernel_scoped() {
        setup_global_test_spice();
        let r =
            spk_position_from_kernel("de440s", NAIFId::Moon, NAIFId::Earth, epc_2025()).unwrap();
        assert!(r.norm() > 3.5e8 && r.norm() < 4.1e8);
    }

    #[test]
    #[serial]
    fn test_spk_state_and_velocity_from_kernel_scoped() {
        // Scoped state/velocity queries against a single named kernel: the
        // state's position/velocity halves must equal the position-only and
        // velocity-only scoped queries.
        setup_global_test_spice();
        let epc = epc_2025();
        let x = spk_state_from_kernel("de440s", NAIFId::Moon, NAIFId::Earth, epc).unwrap();
        let r = spk_position_from_kernel("de440s", NAIFId::Moon, NAIFId::Earth, epc).unwrap();
        let v = spk_velocity_from_kernel("de440s", NAIFId::Moon, NAIFId::Earth, epc).unwrap();
        assert_abs_diff_eq!((x.fixed_rows::<3>(0) - r).norm(), 0.0, epsilon = 1e-9);
        assert_abs_diff_eq!((x.fixed_rows::<3>(3) - v).norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    #[serial]
    fn test_concurrent_queries() {
        setup_global_test_spice();
        load_spice_kernel("de440s").unwrap();
        let handles: Vec<_> = (0..8)
            .map(|i| {
                std::thread::spawn(move || {
                    let epc = Epoch::from_date(2025, 1, 1 + i, TimeSystem::UTC);
                    spk_position(NAIFId::Moon, NAIFId::Earth, epc)
                        .unwrap()
                        .norm()
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
        clear_spice_kernels();

        load_spice_kernel("moon_pa_de440").unwrap();
        assert_eq!(loaded_spice_kernels(), vec!["moon_pa_de440".to_string()]);

        // Frame class 31008 (MOON_PA_DE440) has coverage at 2025-01-01.
        let r = pck_rotation_matrix(31008, epc_2025()).unwrap().to_matrix();
        let rtr = r.transpose() * r;
        assert_abs_diff_eq!((rtr - Matrix3::identity()).norm(), 0.0, epsilon = 1e-9);

        // Typed accessors agree with the rotation matrix at the same epoch.
        let e = pck_euler_angle(31008, epc_2025()).unwrap();
        assert_abs_diff_eq!(e.to_rotation_matrix().to_matrix(), r, epsilon = 1e-9);
        let q = pck_quaternion(31008, epc_2025()).unwrap();
        assert_abs_diff_eq!(q.to_rotation_matrix().to_matrix(), r, epsilon = 1e-9);

        // Restore global state for other tests.
        clear_spice_kernels();
        load_spice_kernel("de440s").unwrap();
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)]
    #[serial]
    fn test_load_mar099s_and_query_phobos_deimos_network() {
        setup_global_test_spice();
        clear_spice_kernels();

        load_spice_kernel("mar099s").unwrap();
        assert_eq!(loaded_spice_kernels(), vec!["mar099s".to_string()]);

        let epc = Epoch::from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        // Phobos (401) relative to Mars barycenter (4): |r| ~ 9376 km
        let r = spk_position(NAIFId::Phobos, NAIFId::MarsBarycenter, epc).unwrap();
        assert!((r.norm() - 9.376e6).abs() < 0.2e6);
        // Deimos (402) relative to Mars barycenter (4): |r| ~ 23463 km
        let r = spk_position(NAIFId::Deimos, NAIFId::MarsBarycenter, epc).unwrap();
        assert!((r.norm() - 2.3463e7).abs() < 0.5e6);

        // Restore global state for other tests.
        clear_spice_kernels();
        load_spice_kernel("de440s").unwrap();
    }

    #[test]
    fn test_naif_kernel_name() {
        use super::super::kernels::SPICEKernel;
        assert_eq!(SPICEKernel::DE440s.name(), "de440s");
        assert_eq!(SPICEKernel::DE440.name(), "de440");
    }

    /// The kernel sets are compile-time lists; assert contents so docs stay
    /// honest.
    #[test]
    fn test_common_and_all_kernel_lists() {
        assert_eq!(
            COMMON_SPICE_KERNELS,
            &[SPICEKernel::DE440s, SPICEKernel::MoonPaDe440]
        );
        assert_eq!(
            ALL_SPICE_KERNELS,
            &[
                SPICEKernel::DE440s,
                SPICEKernel::MoonPaDe440,
                SPICEKernel::Mar099s,
                SPICEKernel::Jup365,
                SPICEKernel::Sat441,
                SPICEKernel::Ura184,
                SPICEKernel::Nep097,
                SPICEKernel::Plu060,
            ]
        );
    }

    #[test]
    #[cfg_attr(not(feature = "integration"), ignore)] // downloads moon_pa_de440
    #[serial]
    fn test_load_common_kernels() {
        setup_global_test_spice();
        clear_spice_kernels();

        load_common_spice_kernels().unwrap();
        let loaded = loaded_spice_kernels();
        assert!(loaded.contains(&"de440s".to_string()));
        assert!(loaded.contains(&"moon_pa_de440".to_string()));

        // Restore global state for other tests.
        clear_spice_kernels();
        load_spice_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_resolve_kernel_path_missing_file_errors() {
        // A path source that is neither a known kernel name nor an existing
        // file surfaces `resolve_kernel_path`'s IoError.
        setup_global_test_spice();
        let err = load_spice_kernel("/nonexistent/path/to/kernel.bsp").unwrap_err();
        assert!(format!("{}", err).contains("neither a known kernel name"));
    }

    #[test]
    #[serial]
    fn test_ensure_bodies_loadable_loads_de_before_satellite_kernel() {
        // From an SPK-empty registry, the default DE ephemeris must load
        // BEFORE the satellite kernel: a satellite kernel alone would
        // suppress the DE auto-initialization and leave e.g. the Earth leg
        // of a 499<->399 query unresolvable.
        setup_global_test_spice();
        load_spice_kernel("de440s").unwrap();
        {
            let cache = CacheRedirect::new();
            cache.seed_real_de440s();
            cache.seed(
                "mar099s.bsp",
                &crate::utils::testing::synthetic_spk_kernel_bytes(&[(499, 4, 2.0)]),
            );

            clear_spice_kernels();
            ensure_bodies_loadable(&[499]).unwrap();
            assert_eq!(
                loaded_spice_kernels(),
                vec!["de440s".to_string(), "mar099s".to_string()]
            );

            // Idempotent: a second call adds nothing.
            ensure_bodies_loadable(&[499]).unwrap();
            assert_eq!(loaded_spice_kernels().len(), 2);

            // DE-covered IDs load no satellite kernel.
            clear_spice_kernels();
            ensure_bodies_loadable(&[301, 399]).unwrap();
            assert_eq!(loaded_spice_kernels(), vec!["de440s".to_string()]);
        }
        // Restore global state for other tests.
        clear_spice_kernels();
        load_spice_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_kernel_is_loaded() {
        setup_global_test_spice();
        load_spice_kernel("de440s").unwrap();
        assert!(kernel_is_loaded("de440s"));
        // Substring of a loaded key matches; an unknown fragment does not.
        assert!(kernel_is_loaded("de440"));
        assert!(!kernel_is_loaded("nonexistent_kernel"));
    }

    #[test]
    #[serial]
    fn test_load_kernel_rejects_unrecognized_daf_id_word() {
        // A structurally valid DAF whose ID word is neither DAF/SPK nor
        // DAF/PCK hits `load_spice_kernel`'s `other` arm.
        setup_global_test_spice();
        clear_spice_kernels();

        let mut bytes = crate::utils::testing::synthetic_spk_kernel_bytes(&[(10, 0, 1.0)]);
        bytes[..8].copy_from_slice(b"DAF/CK  ");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mystery.bin");
        std::fs::write(&path, &bytes).unwrap();

        let err = load_spice_kernel(path.to_str().unwrap()).unwrap_err();
        assert!(format!("{}", err).contains("Unrecognized DAF ID word"));

        // Restore global state for other tests.
        clear_spice_kernels();
        load_spice_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_load_common_kernels_offline() {
        // Offline counterpart of `test_load_common_kernels`. The real de440s
        // is kept resident so `load_common_spice_kernels`' de440s load hits the
        // idempotent short-circuit (and any concurrent cache read stays
        // valid); only the synthetic moon_pa_de440 PCK is fetched, from the
        // redirected cache, so no download occurs.
        setup_global_test_spice();
        load_spice_kernel("de440s").unwrap();
        {
            let cache = CacheRedirect::new();
            cache.seed_real_de440s();
            cache.seed("moon_pa_de440_200625.bpc", &synthetic_bpck_bytes());

            load_common_spice_kernels().unwrap();
            let loaded = loaded_spice_kernels();
            assert!(loaded.contains(&"de440s".to_string()));
            assert!(loaded.contains(&"moon_pa_de440".to_string()));
        }
        clear_spice_kernels();
        load_spice_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_load_all_kernels_offline() {
        // Every kernel `load_all_spice_kernels` pulls beyond de440s (the lunar PCK
        // and one satellite ephemeris kernel per outer planet) is seeded as a
        // synthetic file; de440s stays resident and short-circuits.
        setup_global_test_spice();
        load_spice_kernel("de440s").unwrap();
        {
            let cache = CacheRedirect::new();
            cache.seed_real_de440s();
            cache.seed("moon_pa_de440_200625.bpc", &synthetic_bpck_bytes());
            cache.seed("mar099s.bsp", &synthetic_spk_bytes(1.0));
            cache.seed("jup365.bsp", &synthetic_spk_bytes(1.0));
            cache.seed("sat441.bsp", &synthetic_spk_bytes(1.0));
            cache.seed("ura184_part-3.bsp", &synthetic_spk_bytes(1.0));
            cache.seed("nep097.bsp", &synthetic_spk_bytes(1.0));
            cache.seed("plu060.bsp", &synthetic_spk_bytes(1.0));

            load_all_spice_kernels().unwrap();
            let loaded = loaded_spice_kernels();
            for name in [
                "de440s",
                "moon_pa_de440",
                "mar099s",
                "jup365",
                "sat441",
                "ura184",
                "nep097",
                "plu060",
            ] {
                assert!(loaded.contains(&name.to_string()), "missing {}", name);
            }
        }
        clear_spice_kernels();
        load_spice_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_global_chain_errors_with_no_spk_loaded() {
        // `global_chain`'s `return Err` branch: with no SPK kernel loaded the
        // collected segment set is empty.
        setup_global_test_spice();
        clear_spice_kernels();

        let err = global_chain(1, 0).unwrap_err();
        assert!(format!("{}", err).contains("No SPK kernels loaded"));

        // Restore global state for other tests.
        load_spice_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_global_chain_cache_miss_then_hit() {
        // First resolution is a cache miss (resolve + insert -> final `Ok`);
        // the second identical query is a cache hit (early `return Ok`).
        setup_global_test_spice();
        clear_spice_kernels();
        load_spice_kernel("de440s").unwrap();

        let c1 = global_chain(10, 0).unwrap();
        let c2 = global_chain(10, 0).unwrap();
        assert_eq!(c1.len(), c2.len());
    }

    #[test]
    #[serial]
    fn test_spk_kernel_for_query_rejects_pck() {
        // Scoped SPK query against a loaded PCK hits `spk_kernel_for_query`'s
        // PCK error arm.
        setup_global_test_spice();
        clear_spice_kernels();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("orient.bpc");
        std::fs::write(&path, synthetic_bpck_bytes()).unwrap();

        let err = spk_position_from_kernel(
            path.to_str().unwrap(),
            NAIFId::Moon,
            NAIFId::Earth,
            epc_2025(),
        )
        .unwrap_err();
        assert!(format!("{}", err).contains("binary PCK"));

        // Restore global state for other tests.
        clear_spice_kernels();
        load_spice_kernel("de440s").unwrap();
    }

    #[test]
    #[serial]
    fn test_pck_typed_registry_functions_offline() {
        // Exercises `pck_query`'s `Ok` path and every typed PCK registry
        // accessor against a path-loaded synthetic PCK (frame 31006 covering
        // ET [0, 1000]).
        setup_global_test_spice();
        clear_spice_kernels();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("synthetic.bpc");
        std::fs::write(&path, synthetic_bpck_bytes()).unwrap();
        load_spice_kernel(path.to_str().unwrap()).unwrap();

        let epc = epc_from_et(500.0);
        let angle = pck_euler_angle(31006, epc).unwrap();
        let rates = pck_euler_rates(31006, epc).unwrap();
        let (angle2, rates2) = pck_euler_angle_and_rates(31006, epc).unwrap();
        assert_eq!(angle2, angle);
        assert_abs_diff_eq!(rates2, rates, epsilon = 0.0);

        let q = pck_quaternion(31006, epc).unwrap();
        let r = pck_rotation_matrix(31006, epc).unwrap();
        // Quaternion, rotation matrix, and typed EulerAngle all agree.
        assert_abs_diff_eq!(
            q.to_rotation_matrix().to_matrix(),
            r.to_matrix(),
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(
            angle.to_rotation_matrix().to_matrix(),
            r.to_matrix(),
            epsilon = 1e-12
        );

        // Restore global state for other tests.
        clear_spice_kernels();
        load_spice_kernel("de440s").unwrap();
    }
}
