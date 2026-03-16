/*!
 * Owns global Almanac
 */

use std::sync::{Arc, RwLock};

use once_cell::sync::Lazy;

use anise::prelude as anise_prelude;

use crate::datasets::naif::download_de_kernel;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

use super::kernels::SPKKernel;

// ============================================================================
// Epoch Conversion
// ============================================================================

/// Convert a Brahe [`Epoch`] to an ANISE Epoch using Gregorian calendar components.
///
/// Converts via datetime components
#[inline]
pub(super) fn brahe_epoch_to_anise(epc: Epoch) -> anise_prelude::Epoch {
    let (yy, mm, dd, h, m, s, ns) = epc.to_datetime_as_time_system(TimeSystem::UTC);
    anise_prelude::Epoch::from_gregorian_utc(yy as i32, mm, dd, h, m, s as u8, ns as u32)
}

// ============================================================================
// Global Almanac State
// ============================================================================

/// Global ANISE Almanac instance for DE position queries.
///
/// Thread-safe, lazily initialized. Use `initialize_ephemeris` to pre-load,
/// or it will be loaded automatically on the first DE position call.
static GLOBAL_ALMANAC: Lazy<Arc<RwLock<Option<Arc<anise_prelude::Almanac>>>>> =
    Lazy::new(|| Arc::new(RwLock::new(None)));

/// TODO: Replace with a kernel-set tracker once multiple kernel types are supported simultaneously.
/// Tracks which SPK kernel is currently loaded
static GLOBAL_KERNEL_TYPE: Lazy<Arc<RwLock<Option<String>>>> =
    Lazy::new(|| Arc::new(RwLock::new(None)));

// ============================================================================
// Public API
// ============================================================================

/// Set a custom ANISE Almanac as the global ephemeris provider.
///
/// # Example
///
/// ```no_run
/// use brahe::spice::set_global_almanac;
/// use anise::prelude::{SPK, Almanac};
///
/// let spk = SPK::load("path/to/custom.bsp").unwrap();
/// let almanac = Almanac::from_spk(spk);
/// set_global_almanac(almanac);
/// ```
pub fn set_global_almanac(almanac: anise_prelude::Almanac) {
    *GLOBAL_ALMANAC.write().unwrap() = Some(Arc::new(almanac));
}

/// Initialize the global ephemeris provider with the default DE440s kernel.
///
/// Optional. The almanac is lazily initialized on first use if not called.
/// Call explicitly to control when the download/load latency occurs.
///
/// # Example
///
/// ```
/// use brahe::spice::initialize_ephemeris;
///
/// initialize_ephemeris().expect("Failed to initialize ephemeris");
/// ```
pub fn initialize_ephemeris() -> Result<(), BraheError> {
    initialize_ephemeris_with_kernel("de440s")
}

/// Initialize the global ephemeris provider with a specific JPL DE kernel.
///
/// Supported kernels: `"de440s"` and `"de440"`.
///
/// # Arguments
///
/// * `kernel` - Name of the DE kernel to download and load as the global ephemeris provider
///
/// # Example
///
/// ```
/// use brahe::spice::initialize_ephemeris_with_kernel;
///
/// initialize_ephemeris_with_kernel("de440s").expect("Failed to initialize DE440s");
/// ```
pub fn initialize_ephemeris_with_kernel(kernel: &str) -> Result<(), BraheError> {
    let de_path = download_de_kernel(kernel, None)?;
    let de_path_str = de_path.to_str().ok_or_else(|| {
        BraheError::IoError(format!("Failed to convert {} path to string", kernel))
    })?;

    let spk = anise_prelude::SPK::load(de_path_str)
        .map_err(|e| BraheError::IoError(format!("Failed to load {} kernel: {}", kernel, e)))?;
    let almanac = anise_prelude::Almanac::from_spk(spk);

    set_global_almanac(almanac);
    *GLOBAL_KERNEL_TYPE.write().unwrap() = Some(kernel.to_string());

    Ok(())
}

/// Return the name of the currently loaded kernel, or `None` if none is loaded.
///
/// # Example
///
/// ```
/// use brahe::spice::{initialize_ephemeris_with_kernel, get_loaded_kernel_type};
///
/// initialize_ephemeris_with_kernel("de440s").unwrap();
/// assert_eq!(get_loaded_kernel_type(), Some("de440s".to_string()));
/// ```
pub fn get_loaded_kernel_type() -> Option<String> {
    GLOBAL_KERNEL_TYPE.read().unwrap().clone()
}

// ============================================================================
// Internal Helpers
// ============================================================================

/// Get the global almanac, auto-initializing with DE440s if not yet loaded.
pub(super) fn get_almanac() -> Result<Arc<anise_prelude::Almanac>, BraheError> {
    {
        let reader = GLOBAL_ALMANAC.read().unwrap();
        if let Some(ref almanac) = *reader {
            return Ok(Arc::clone(almanac));
        }
    }

    let mut writer = GLOBAL_ALMANAC.write().unwrap();

    // Double-checked locking: another thread may have initialized while we waited
    if let Some(ref almanac) = *writer {
        return Ok(Arc::clone(almanac));
    }

    let path = download_de_kernel("de440s", None)?;
    let path_str = path.to_str().ok_or_else(|| {
        BraheError::IoError("Failed to convert DE440s path to string".to_string())
    })?;

    let spk = anise_prelude::SPK::load(path_str)
        .map_err(|e| BraheError::IoError(format!("Failed to load DE440s kernel: {}", e)))?;
    let almanac_arc = Arc::new(anise_prelude::Almanac::from_spk(spk));
    *writer = Some(Arc::clone(&almanac_arc));
    *GLOBAL_KERNEL_TYPE.write().unwrap() = Some("de440s".to_string());

    Ok(almanac_arc)
}

/// Ensure the correct kernel is loaded for `kernel`, switching if needed.
pub(super) fn ensure_kernel_loaded(
    kernel: SPKKernel,
) -> Result<Arc<anise_prelude::Almanac>, BraheError> {
    let required = match kernel {
        SPKKernel::DE440s => "de440s",
        SPKKernel::DE440 => "de440",
    };

    // Fast path: correct kernel already loaded (read lock only)
    {
        let loaded = GLOBAL_KERNEL_TYPE.read().unwrap();
        if let Some(ref k) = *loaded
            && k == required
        {
            return get_almanac();
        }
    }

    // Slow path: switch kernels
    initialize_ephemeris_with_kernel(required)?;
    get_almanac()
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::Arc;

    use serial_test::serial;

    use super::*;
    use crate::utils::cache::get_naif_cache_dir;

    fn reset_global_almanac_state() {
        *GLOBAL_ALMANAC.write().unwrap() = None;
        *GLOBAL_KERNEL_TYPE.write().unwrap() = None;
    }

    fn setup_test_kernel() -> Option<PathBuf> {
        let test_asset_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_assets")
            .join("de440s.bsp");

        if !test_asset_path.exists() {
            return None;
        }

        let cache_dir = get_naif_cache_dir().expect("Failed to get NAIF cache dir");
        let cache_path = PathBuf::from(&cache_dir).join("de440s.bsp");

        if !cache_path.exists() {
            fs::copy(&test_asset_path, &cache_path).expect("Failed to copy test asset to cache");
        }

        Some(cache_path)
    }

    fn load_test_almanac() -> Option<anise_prelude::Almanac> {
        let cache_path = setup_test_kernel()?;
        let cache_path_str = cache_path
            .to_str()
            .expect("Failed to convert path to string");
        let spk =
            anise_prelude::SPK::load(cache_path_str).expect("Failed to load DE440s test kernel");
        Some(anise_prelude::Almanac::from_spk(spk))
    }

    #[test]
    #[serial]
    fn test_initialize_ephemeris_with_invalid_kernel_returns_err() {
        reset_global_almanac_state();

        let result = initialize_ephemeris_with_kernel("invalid");

        assert!(result.is_err());
        assert_eq!(get_loaded_kernel_type(), None);
    }

    #[test]
    #[serial]
    fn test_initialize_ephemeris_sets_loaded_kernel_type() {
        let Some(_) = setup_test_kernel() else {
            return;
        };

        reset_global_almanac_state();
        initialize_ephemeris().expect("Failed to initialize DE440s test kernel");

        assert_eq!(get_loaded_kernel_type(), Some("de440s".to_string()));

        reset_global_almanac_state();
    }

    #[test]
    #[serial]
    fn test_get_almanac_returns_seeded_global_almanac() {
        let Some(almanac) = load_test_almanac() else {
            return;
        };

        reset_global_almanac_state();

        let seeded = Arc::new(almanac);
        *GLOBAL_ALMANAC.write().unwrap() = Some(Arc::clone(&seeded));

        let retrieved = get_almanac().expect("Failed to get seeded almanac");

        assert!(Arc::ptr_eq(&retrieved, &seeded));

        reset_global_almanac_state();
    }

    #[test]
    #[serial]
    fn test_ensure_kernel_loaded_returns_existing_almanac_for_matching_kernel() {
        let Some(almanac) = load_test_almanac() else {
            return;
        };

        reset_global_almanac_state();

        let seeded = Arc::new(almanac);
        *GLOBAL_ALMANAC.write().unwrap() = Some(Arc::clone(&seeded));
        *GLOBAL_KERNEL_TYPE.write().unwrap() = Some("de440s".to_string());

        let retrieved =
            ensure_kernel_loaded(SPKKernel::DE440s).expect("Failed to get existing almanac");

        assert!(Arc::ptr_eq(&retrieved, &seeded));
        assert_eq!(get_loaded_kernel_type(), Some("de440s".to_string()));

        reset_global_almanac_state();
    }
}
