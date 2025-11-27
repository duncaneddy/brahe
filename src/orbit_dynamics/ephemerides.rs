/*!
Provide low-accuracy ephemerides for various celestial bodies.
 */

use nalgebra::Vector3;
use once_cell::sync::Lazy;
use std::sync::{Arc, RwLock};

use anise::constants::frames as anise_frames;
use anise::prelude as anise_prelude;

use crate::DEG2RAD;
use crate::attitude::RotationMatrix;
use crate::constants::{AS2RAD, MJD2000, RADIANS};
use crate::datasets::naif::download_de_kernel;
use crate::frames::rotation_eme2000_to_gcrf;
use crate::propagators::force_model_config::EphemerisSource;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

// ============================================================================
// Epoch Conversion Utilities
// ============================================================================

/// Convert a Brahe Epoch to an ANISE Epoch using Gregorian calendar components.
///
/// This method converts via datetime components rather than ISO string parsing,
/// providing more direct and stable conversion.
///
/// # Arguments
///
/// * `epc` - Brahe Epoch to convert
///
/// # Returns
///
/// * ANISE Epoch in UTC time scale
#[inline]
fn brahe_epoch_to_anise(epc: Epoch) -> anise_prelude::Epoch {
    let (yy, mm, dd, h, m, s, ns) = epc.to_datetime_as_time_system(TimeSystem::UTC);
    anise_prelude::Epoch::from_gregorian_utc(yy as i32, mm, dd, h, m, s as u8, ns as u32)
}

// ============================================================================
// Global Almanac Management
// ============================================================================

/// Global ANISE Almanac instance for high-precision ephemeris computations.
///
/// This static provides thread-safe, shared access to a single ANISE Almanac
/// context loaded with ephemeris kernels (DE440s or DE440). The Almanac is lazily
/// initialized on first use or can be pre-initialized via `initialize_ephemeris()`.
static GLOBAL_ALMANAC: Lazy<Arc<RwLock<Option<Arc<anise_prelude::Almanac>>>>> =
    Lazy::new(|| Arc::new(RwLock::new(None)));

/// Global tracker for which kernel is currently loaded.
///
/// Tracks which NAIF kernel ("de440s" or "de440") is currently loaded in the
/// global Almanac. This is useful for debugging and validation.
static GLOBAL_KERNEL_TYPE: Lazy<Arc<RwLock<Option<String>>>> =
    Lazy::new(|| Arc::new(RwLock::new(None)));

/// Set a custom ANISE Almanac as the global ephemeris provider.
///
/// This function allows users to configure a custom Almanac (e.g., with different
/// kernels or settings) for use by all DE440s ephemeris functions.
///
/// # Arguments
///
/// * `almanac` - The ANISE Almanac instance to set as the global provider
///
/// # Example
///
/// ```no_run
/// use brahe::ephemerides::set_global_almanac;
/// use anise::prelude::{SPK, Almanac};
///
/// // Load custom kernel
/// let spk = SPK::load("path/to/custom.bsp").unwrap();
/// let almanac = Almanac::from_spk(spk);
///
/// // Set as global
/// set_global_almanac(almanac);
/// ```
pub fn set_global_almanac(almanac: anise_prelude::Almanac) {
    *GLOBAL_ALMANAC.write().unwrap() = Some(Arc::new(almanac));
}

/// Initialize the global ephemeris provider with the default DE440s kernel.
///
/// This function downloads (or uses a cached copy of) the NAIF DE440s ephemeris
/// kernel and sets it as the global Almanac provider. This initialization is
/// optional - if not called, the Almanac will be lazily initialized on the first
/// call to `sun_position_de()` or `moon_position_de()`.
///
/// Calling this function explicitly is recommended when you want to:
/// - Control when the kernel download/loading occurs (avoid latency on first use)
/// - Handle initialization errors explicitly
/// - Pre-load the kernel during application startup
///
/// # Returns
///
/// * `Ok(())` if the Almanac was successfully initialized
/// * `Err(BraheError)` if kernel download or loading failed
///
/// # Example
///
/// ```
/// use brahe::ephemerides::initialize_ephemeris;
///
/// // Initialize at application startup
/// initialize_ephemeris().expect("Failed to initialize ephemeris");
/// ```
pub fn initialize_ephemeris() -> Result<(), BraheError> {
    initialize_ephemeris_with_kernel("de440s")
}

/// Initialize the global ephemeris provider with a specific JPL DE kernel.
///
/// This function downloads (or uses a cached copy of) the specified NAIF DE
/// ephemeris kernel and sets it as the global Almanac provider. Supported
/// kernels include "de440s" (smaller, 1550-2650 CE) and "de440" (full, 13200
/// BCE-17191 CE).
///
/// # Arguments
///
/// * `kernel` - Name of the kernel to load (typically "de440s" or "de440")
///
/// # Returns
///
/// * `Ok(())` if the Almanac was successfully initialized
/// * `Err(BraheError)` if kernel download or loading failed
///
/// # Example
///
/// ```
/// use brahe::ephemerides::initialize_ephemeris_with_kernel;
///
/// // Initialize with full DE440 kernel
/// initialize_ephemeris_with_kernel("de440").expect("Failed to initialize DE440");
///
/// // Or use smaller DE440s kernel
/// initialize_ephemeris_with_kernel("de440s").expect("Failed to initialize DE440s");
/// ```
pub fn initialize_ephemeris_with_kernel(kernel: &str) -> Result<(), BraheError> {
    // Download or get cached kernel
    let de_path = download_de_kernel(kernel, None)?;
    let de_path_str = de_path.to_str().ok_or_else(|| {
        BraheError::IoError(format!("Failed to convert {} path to string", kernel))
    })?;

    // Load SPK and create Almanac context
    let spk = anise_prelude::SPK::load(de_path_str)
        .map_err(|e| BraheError::IoError(format!("Failed to load {} kernel: {}", kernel, e)))?;
    let almanac = anise_prelude::Almanac::from_spk(spk);

    // Set as global and track which kernel is loaded
    set_global_almanac(almanac);
    *GLOBAL_KERNEL_TYPE.write().unwrap() = Some(kernel.to_string());

    Ok(())
}

/// Get the name of the currently loaded ephemeris kernel.
///
/// Returns the kernel name (e.g., "de440s" or "de440") if an ephemeris has been
/// initialized, or `None` if no kernel has been loaded yet.
///
/// # Returns
///
/// * `Some(String)` - Name of the loaded kernel
/// * `None` - No kernel has been loaded
///
/// # Example
///
/// ```
/// use brahe::ephemerides::{initialize_ephemeris_with_kernel, get_loaded_kernel_type};
///
/// initialize_ephemeris_with_kernel("de440").unwrap();
/// assert_eq!(get_loaded_kernel_type(), Some("de440".to_string()));
/// ```
pub fn get_loaded_kernel_type() -> Option<String> {
    GLOBAL_KERNEL_TYPE.read().unwrap().clone()
}

/// Internal helper to get the global Almanac, initializing it if necessary.
///
/// This function implements the lazy initialization pattern: if the Almanac
/// has not been explicitly initialized via `initialize_ephemeris()`, it will
/// be automatically loaded on first access.
///
/// # Returns
///
/// * `Ok(Arc<Almanac>)` - Shared reference to the global Almanac
/// * `Err(BraheError)` - If initialization failed
fn get_almanac() -> Result<Arc<anise_prelude::Almanac>, BraheError> {
    // Try to get existing almanac
    {
        let reader = GLOBAL_ALMANAC.read().unwrap();
        if let Some(ref almanac) = *reader {
            return Ok(Arc::clone(almanac));
        }
    }

    // Need to initialize - acquire write lock
    let mut writer = GLOBAL_ALMANAC.write().unwrap();

    // Double-check pattern: another thread might have initialized while we waited
    if let Some(ref almanac) = *writer {
        return Ok(Arc::clone(almanac));
    }

    // Initialize with default DE440s kernel
    let de440s_path = download_de_kernel("de440s", None)?;
    let de440s_path_str = de440s_path.to_str().ok_or_else(|| {
        BraheError::IoError("Failed to convert DE440s path to string".to_string())
    })?;

    let spk = anise_prelude::SPK::load(de440s_path_str)
        .map_err(|e| BraheError::IoError(format!("Failed to load DE440s kernel: {}", e)))?;
    let almanac = anise_prelude::Almanac::from_spk(spk);

    let almanac_arc = Arc::new(almanac);
    *writer = Some(Arc::clone(&almanac_arc));

    // Track that DE440s was loaded
    *GLOBAL_KERNEL_TYPE.write().unwrap() = Some("de440s".to_string());

    Ok(almanac_arc)
}

/// Ensure the correct ephemeris kernel is loaded for the given source.
///
/// Automatically loads the appropriate kernel if a different one is currently loaded.
/// Uses thread-safe double-checked locking for minimal overhead.
///
/// # Arguments
///
/// * `source` - The ephemeris source requiring validation
///
/// # Returns
///
/// * `Ok(Arc<Almanac>)` - Reference to the loaded almanac
/// * `Err(BraheError)` - If LowPrecision source is used (not supported) or kernel loading fails
fn ensure_kernel_loaded(
    source: EphemerisSource,
) -> Result<Arc<anise_prelude::Almanac>, BraheError> {
    let required_kernel = match source {
        EphemerisSource::DE440s => "de440s",
        EphemerisSource::DE440 => "de440",
        EphemerisSource::LowPrecision => {
            return Err(BraheError::Error(
                "Low-precision ephemeris source does not use NAIF kernels. \
                 Use sun_position() or moon_position() instead."
                    .to_string(),
            ));
        }
    };

    // Fast path: Check if correct kernel already loaded (read lock only)
    {
        let loaded = GLOBAL_KERNEL_TYPE.read().unwrap();
        if let Some(ref kernel_type) = *loaded
            && kernel_type == required_kernel
        {
            // Already loaded, return quickly
            return get_almanac();
        }
    }

    // Slow path: Need to switch kernels (write lock)
    initialize_ephemeris_with_kernel(required_kernel)?;
    get_almanac()
}

// ============================================================================
// Ephemeris Functions
// ============================================================================

/// Calculate the position of the Sun in the GCRF inertial frame using low-precision analytical
/// methods.
///
/// # Arguments
///
/// - `epc`: Epoch at which to calculate the Sun's position
///
/// # Returns
///
/// - `r`: Position of the Sun in the GCRF frame. Units: [m]
///
/// # References
///
/// - O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, 2012, p.70-73.
///
/// # Example
///
/// ```
/// use brahe::ephemerides::sun_position;
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
/// use brahe::constants::AU;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
///
/// // Get Sun position in GCRF frame
/// let r_sun = sun_position(epc);
/// ```
#[allow(non_snake_case)]
pub fn sun_position(epc: Epoch) -> Vector3<f64> {
    // Constants
    let pi = std::f64::consts::PI;
    let mjd_tt = epc.mjd_as_time_system(TimeSystem::TT);
    let epsilon = 23.43929111 * DEG2RAD; // Obliquity of J2000 ecliptic
    let T = (mjd_tt - MJD2000) / 36525.0; // Julian cent. since J2000

    // Mean anomaly, ecliptic longitude and radius
    let M = 2.0 * pi * (0.9931267 + 99.9973583 * T).fract(); // [rad]
    let L = 2.0
        * pi
        * (0.7859444 + M / (2.0 * pi) + (6892.0 * M.sin() + 72.0 * (2.0 * M).sin()) / 1296.0e3)
            .fract(); // [rad]
    let r = 149.619e9 - 2.499e9 * M.cos() - 0.021e9 * (2.0 * M).cos(); // [m]

    // Equatorial position vector
    let r_run_eme_2000 =
        RotationMatrix::Rx(-epsilon, RADIANS) * Vector3::new(r * L.cos(), r * L.sin(), 0.0);

    // Convert to GCRF frame
    rotation_eme2000_to_gcrf() * r_run_eme_2000
}

/// Calculate the position of the Moon in the GCRF inertial frame using low-precision analytical
/// methods.
///
/// # Arguments
///
/// - `epc`: Epoch at which to calculate the Moon's position
///
/// # Returns
///
/// - `r`: Position of the Moon in the GCRF ecliptic frame. Units: [m]
///
/// # References
///
/// - O. Montenbruck, and E. Gill, *Satellite Orbits: Models, Methods and Applications*, 2012, p.70-73.
///
/// # Example
///
/// ```
/// use brahe::ephemerides::moon_position;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::constants::AU;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
///
/// // Moon position in GCRF frame
/// let r_moon = moon_position(epc);
/// ```
#[allow(non_snake_case)]
pub fn moon_position(epc: Epoch) -> Vector3<f64> {
    // Constants
    let pi = std::f64::consts::PI;
    let mjd_tt = epc.mjd_as_time_system(TimeSystem::TT);
    let epsilon = 23.43929111 * DEG2RAD; // Obliquity of J2000 ecliptic
    let T = (mjd_tt - MJD2000) / 36525.0; // Julian cent. since J2000

    // Mean elements of lunar orbit
    let L_0 = (0.606433 + 1336.851344 * T).fract(); // Mean longitude [rev] w.r.t. J2000 equinox
    let l = 2.0 * pi * (0.374897 + 1325.552410 * T).fract(); // Moon's mean anomaly [rad]
    let lp = 2.0 * pi * (0.993133 + 99.997361 * T).fract(); // Sun's mean anomaly [rad]
    let D = 2.0 * pi * (0.827361 + 1236.853086 * T).fract(); // Diff. long. Moon-Sun [rad]
    let F = 2.0 * pi * (0.259086 + 1342.227825 * T).fract(); // Argument of latitude

    // Ecliptic longitude (w.r.t. equinox of J2000)
    let dL = 22640.0 * l.sin() - 4586.0 * (l - 2.0 * D).sin()
        + 2370.0 * (2.0 * D).sin()
        + 769.0 * (2.0 * l).sin()
        - 668.0 * (lp).sin()
        - 412.0 * (2.0 * F).sin()
        - 212.0 * (2.0 * l - 2.0 * D).sin()
        - 206.0 * (l + lp - 2.0 * D).sin()
        + 192.0 * (l + 2.0 * D).sin()
        - 165.0 * (lp - 2.0 * D).sin()
        - 125.0 * D.sin()
        - 110.0 * (l + lp).sin()
        + 148.0 * (l - lp).sin()
        - 55.0 * (2.0 * F - 2.0 * D).sin();

    let L = 2.0 * pi * (L_0 + dL / 1296.0e3).fract(); // [rad]

    // Ecliptic latitude
    let S = F + (dL + 412.0 * (2.0 * F).sin() + 541.0 * lp.sin()) * AS2RAD;
    let h = F - 2.0 * D;
    let N = -526.0 * h.sin() + 44.0 * (l + h).sin() - 31.0 * (-l + h).sin() - 23.0 * (lp + h).sin()
        + 11.0 * (-lp + h).sin()
        - 25.0 * (-2.0 * l + F).sin()
        + 21.0 * (-l + F).sin();
    let B = (18520.0 * S.sin() + N) * AS2RAD; // [rad]

    // Distance [m]
    let r = 385000e3
        - 20905e3 * l.cos()
        - 3699e3 * (2.0 * D - l).cos()
        - 2956e3 * (2.0 * D).cos()
        - 570e3 * (2.0 * l).cos()
        + 246e3 * (2.0 * l - 2.0 * D).cos()
        - 205e3 * (lp - 2.0 * D).cos()
        - 171e3 * (l + 2.0 * D).cos()
        - 152e3 * (l + lp - 2.0 * D).cos();

    // Equatorial coordinates
    let r_moon_eme2000 = RotationMatrix::Rx(-epsilon, RADIANS)
        * Vector3::new(r * L.cos() * B.cos(), r * L.sin() * B.cos(), r * B.sin());

    // Convert to GCRF frame
    rotation_eme2000_to_gcrf() * r_moon_eme2000
}

/// Calculate the position of the Sun in the GCRF inertial frame using NAIF DE ephemeris (DE440s or DE440).
///
/// This function uses high-precision NAIF DE ephemeris kernels for solar position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate the Sun's position
/// * `source` - Ephemeris source (DE440s or DE440)
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of the Sun in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If LowPrecision source is specified or ephemeris query fails
///
/// # Errors
///
/// * Returns error if LowPrecision source is specified (not supported)
/// * Returns error if ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::ephemerides::sun_position_de;
/// use brahe::propagators::force_model_config::EphemerisSource;
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
///
/// // Get Sun position in GCRF frame using DE440s
/// let r_sun = sun_position_de(epc, EphemerisSource::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn sun_position_de(epc: Epoch, source: EphemerisSource) -> Result<Vector3<f64>, BraheError> {
    // Ensure correct kernel is loaded
    let ctx = ensure_kernel_loaded(source)?;

    // Convert Brahe Epoch to Anise Epoch
    let anise_epoch = brahe_epoch_to_anise(epc);

    // Get Sun position from ephemeris
    let r_sun_eme2000 = ctx
        .translate(
            anise_frames::SUN_J2000, // Target
            anise_frames::EME2000,   // Observer
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Sun position: {}", e)))?;

    // Convert from km to meters
    let r_sun_eme2000_m = Vector3::new(
        r_sun_eme2000.radius_km[0] * 1.0e3,
        r_sun_eme2000.radius_km[1] * 1.0e3,
        r_sun_eme2000.radius_km[2] * 1.0e3,
    );

    // Transform to GCRF frame
    Ok(rotation_eme2000_to_gcrf() * r_sun_eme2000_m)
}

/// Calculate the position of the Moon in the GCRF inertial frame using NAIF DE ephemeris (DE440s or DE440).
///
/// This function uses high-precision NAIF DE ephemeris kernels for lunar position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate the Moon's position
/// * `source` - Ephemeris source (DE440s or DE440)
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of the Moon in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If LowPrecision source is specified or ephemeris query fails
///
/// # Errors
///
/// * Returns error if LowPrecision source is specified (not supported)
/// * Returns error if ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::ephemerides::moon_position_de;
/// use brahe::propagators::force_model_config::EphemerisSource;
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
///
/// // Get Moon position in GCRF frame using DE440s
/// let r_moon = moon_position_de(epc, EphemerisSource::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn moon_position_de(epc: Epoch, source: EphemerisSource) -> Result<Vector3<f64>, BraheError> {
    // Ensure correct kernel is loaded
    let ctx = ensure_kernel_loaded(source)?;

    // Convert Brahe Epoch to Anise Epoch
    let anise_epoch = brahe_epoch_to_anise(epc);

    // Get Moon position from ephemeris
    let r_moon_eme2000 = ctx
        .translate(
            anise_frames::IAU_MOON_FRAME, // Target
            anise_frames::EME2000,        // Observer
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Moon position: {}", e)))?;

    // Convert from km to meters
    let r_moon_eme2000_m = Vector3::new(
        r_moon_eme2000.radius_km[0] * 1.0e3,
        r_moon_eme2000.radius_km[1] * 1.0e3,
        r_moon_eme2000.radius_km[2] * 1.0e3,
    );

    // Transform to GCRF frame
    Ok(rotation_eme2000_to_gcrf() * r_moon_eme2000_m)
}

/// Calculate the position of Jupiter in the GCRF inertial frame using NAIF DE ephemeris (DE440s or DE440).
///
/// This function uses high-precision NAIF DE ephemeris kernels for Jupiter position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Jupiter's position
/// * `source` - Ephemeris source (DE440s or DE440)
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of Jupiter in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If LowPrecision source is specified or ephemeris query fails
///
/// # Errors
///
/// * Returns error if LowPrecision source is specified (not supported)
/// * Returns error if ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::ephemerides::jupiter_position_de;
/// use brahe::propagators::force_model_config::EphemerisSource;
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
///
/// // Get Jupiter's position in GCRF frame using DE440s
/// let r_jupiter = jupiter_position_de(epc, EphemerisSource::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn jupiter_position_de(
    epc: Epoch,
    source: EphemerisSource,
) -> Result<Vector3<f64>, BraheError> {
    // Ensure correct kernel is loaded
    let ctx = ensure_kernel_loaded(source)?;

    // Convert Brahe Epoch to Anise Epoch
    let anise_epoch = brahe_epoch_to_anise(epc);

    // Get Jupiter's position from ephemeris
    let r_jupiter_eme2000 = ctx
        .translate(
            anise_frames::JUPITER_BARYCENTER_J2000, // Target
            anise_frames::EME2000,                  // Observer
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Jupiter position: {}", e)))?;

    // Convert from km to meters
    let r_jupiter_eme2000_m = Vector3::new(
        r_jupiter_eme2000.radius_km[0] * 1.0e3,
        r_jupiter_eme2000.radius_km[1] * 1.0e3,
        r_jupiter_eme2000.radius_km[2] * 1.0e3,
    );

    // Transform to GCRF frame
    Ok(rotation_eme2000_to_gcrf() * r_jupiter_eme2000_m)
}

/// Calculate the position of Mars in the GCRF inertial frame using NAIF DE ephemeris (DE440s or DE440).
///
/// This function uses high-precision NAIF DE ephemeris kernels for Mars position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Mars' position
/// * `source` - Ephemeris source (DE440s or DE440)
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of Mars in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If LowPrecision source is specified or ephemeris query fails
///
/// # Errors
///
/// * Returns error if LowPrecision source is specified (not supported)
/// * Returns error if ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::ephemerides::mars_position_de;
/// use brahe::propagators::force_model_config::EphemerisSource;
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
///
/// // Get Mars's position in GCRF frame using DE440s
/// let r_mars = mars_position_de(epc, EphemerisSource::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn mars_position_de(epc: Epoch, source: EphemerisSource) -> Result<Vector3<f64>, BraheError> {
    // Ensure correct kernel is loaded
    let ctx = ensure_kernel_loaded(source)?;

    // Convert Brahe Epoch to Anise Epoch
    let anise_epoch = brahe_epoch_to_anise(epc);

    // Get Mars's position from ephemeris
    let r_mars_eme2000 = ctx
        .translate(
            anise_frames::MARS_BARYCENTER_J2000, // Target
            anise_frames::EME2000,               // Observer
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Mars position: {}", e)))?;

    // Convert from km to meters
    let r_mars_eme2000_m = Vector3::new(
        r_mars_eme2000.radius_km[0] * 1.0e3,
        r_mars_eme2000.radius_km[1] * 1.0e3,
        r_mars_eme2000.radius_km[2] * 1.0e3,
    );

    // Transform to GCRF frame
    Ok(rotation_eme2000_to_gcrf() * r_mars_eme2000_m)
}

/// Calculate the position of Mercury in the GCRF inertial frame using NAIF DE ephemeris (DE440s or DE440).
///
/// This function uses high-precision NAIF DE ephemeris kernels for Mercury position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Mercury's position
/// * `source` - Ephemeris source (DE440s or DE440)
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of Mercury in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If LowPrecision source is specified or ephemeris query fails
///
/// # Errors
///
/// * Returns error if LowPrecision source is specified (not supported)
/// * Returns error if ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::ephemerides::mercury_position_de;
/// use brahe::propagators::force_model_config::EphemerisSource;
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
///
/// // Get Mercury's position in GCRF frame using DE440s
/// let r_mercury = mercury_position_de(epc, EphemerisSource::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn mercury_position_de(
    epc: Epoch,
    source: EphemerisSource,
) -> Result<Vector3<f64>, BraheError> {
    // Ensure correct kernel is loaded
    let ctx = ensure_kernel_loaded(source)?;

    // Convert Brahe Epoch to Anise Epoch
    let anise_epoch = brahe_epoch_to_anise(epc);

    // Get Mercury's position from ephemeris
    let r_mercury_eme2000 = ctx
        .translate(
            anise_frames::MERCURY_J2000, // Target
            anise_frames::EME2000,       // Observer
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Mercury position: {}", e)))?;

    // Convert from km to meters
    let r_mercury_eme2000_m = Vector3::new(
        r_mercury_eme2000.radius_km[0] * 1.0e3,
        r_mercury_eme2000.radius_km[1] * 1.0e3,
        r_mercury_eme2000.radius_km[2] * 1.0e3,
    );

    // Transform to GCRF frame
    Ok(rotation_eme2000_to_gcrf() * r_mercury_eme2000_m)
}

/// Calculate the position of Neptune in the GCRF inertial frame using NAIF DE ephemeris (DE440s or DE440).
///
/// This function uses high-precision NAIF DE ephemeris kernels for Neptune position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Neptune's position
/// * `source` - Ephemeris source (DE440s or DE440)
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of Neptune in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If LowPrecision source is specified or ephemeris query fails
///
/// # Errors
///
/// * Returns error if LowPrecision source is specified (not supported)
/// * Returns error if ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::ephemerides::neptune_position_de;
/// use brahe::propagators::force_model_config::EphemerisSource;
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
///
/// // Get Neptune's position in GCRF frame using DE440s
/// let r_neptune = neptune_position_de(epc, EphemerisSource::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn neptune_position_de(
    epc: Epoch,
    source: EphemerisSource,
) -> Result<Vector3<f64>, BraheError> {
    // Ensure correct kernel is loaded
    let ctx = ensure_kernel_loaded(source)?;

    // Convert Brahe Epoch to Anise Epoch
    let anise_epoch = brahe_epoch_to_anise(epc);

    // Get Neptune's position from ephemeris
    let r_neptune_eme2000 = ctx
        .translate(
            anise_frames::NEPTUNE_BARYCENTER_J2000, // Target
            anise_frames::EME2000,                  // Observer
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Neptune position: {}", e)))?;

    // Convert from km to meters
    let r_neptune_eme2000_m = Vector3::new(
        r_neptune_eme2000.radius_km[0] * 1.0e3,
        r_neptune_eme2000.radius_km[1] * 1.0e3,
        r_neptune_eme2000.radius_km[2] * 1.0e3,
    );

    // Transform to GCRF frame
    Ok(rotation_eme2000_to_gcrf() * r_neptune_eme2000_m)
}

/// Calculate the position of Saturn in the GCRF inertial frame using NAIF DE ephemeris (DE440s or DE440).
///
/// This function uses high-precision NAIF DE ephemeris kernels for Saturn position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Saturn's position
/// * `source` - Ephemeris source (DE440s or DE440)
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of Saturn in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If LowPrecision source is specified or ephemeris query fails
///
/// # Errors
///
/// * Returns error if LowPrecision source is specified (not supported)
/// * Returns error if ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::ephemerides::saturn_position_de;
/// use brahe::propagators::force_model_config::EphemerisSource;
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
///
/// // Get Saturn's position in GCRF frame using DE440s
/// let r_saturn = saturn_position_de(epc, EphemerisSource::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn saturn_position_de(epc: Epoch, source: EphemerisSource) -> Result<Vector3<f64>, BraheError> {
    // Ensure correct kernel is loaded
    let ctx = ensure_kernel_loaded(source)?;

    // Convert Brahe Epoch to Anise Epoch
    let anise_epoch = brahe_epoch_to_anise(epc);

    // Get Saturn's position from ephemeris
    let r_saturn_eme2000 = ctx
        .translate(
            anise_frames::SATURN_BARYCENTER_J2000, // Target
            anise_frames::EME2000,                 // Observer
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Saturn position: {}", e)))?;

    // Convert from km to meters
    let r_saturn_eme2000_m = Vector3::new(
        r_saturn_eme2000.radius_km[0] * 1.0e3,
        r_saturn_eme2000.radius_km[1] * 1.0e3,
        r_saturn_eme2000.radius_km[2] * 1.0e3,
    );

    // Transform to GCRF frame
    Ok(rotation_eme2000_to_gcrf() * r_saturn_eme2000_m)
}

/// Calculate the position of Uranus in the GCRF inertial frame using NAIF DE ephemeris (DE440s or DE440).
///
/// This function uses high-precision NAIF DE ephemeris kernels for Uranus position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Uranus' position
/// * `source` - Ephemeris source (DE440s or DE440)
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of Uranus in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If LowPrecision source is specified or ephemeris query fails
///
/// # Errors
///
/// * Returns error if LowPrecision source is specified (not supported)
/// * Returns error if ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::ephemerides::uranus_position_de;
/// use brahe::propagators::force_model_config::EphemerisSource;
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
///
/// // Get Uranus' position in GCRF frame using DE440s
/// let r_uranus = uranus_position_de(epc, EphemerisSource::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn uranus_position_de(epc: Epoch, source: EphemerisSource) -> Result<Vector3<f64>, BraheError> {
    // Ensure correct kernel is loaded
    let ctx = ensure_kernel_loaded(source)?;

    // Convert Brahe Epoch to Anise Epoch
    let anise_epoch = brahe_epoch_to_anise(epc);

    // Get Uranus' position from ephemeris
    let r_uranus_eme2000 = ctx
        .translate(
            anise_frames::URANUS_BARYCENTER_J2000, // Target
            anise_frames::EME2000,                 // Observer
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Uranus position: {}", e)))?;

    // Convert from km to meters
    let r_uranus_eme2000_m = Vector3::new(
        r_uranus_eme2000.radius_km[0] * 1.0e3,
        r_uranus_eme2000.radius_km[1] * 1.0e3,
        r_uranus_eme2000.radius_km[2] * 1.0e3,
    );

    // Transform to GCRF frame
    Ok(rotation_eme2000_to_gcrf() * r_uranus_eme2000_m)
}

/// Calculate the position of Venus in the GCRF inertial frame using NAIF DE ephemeris (DE440s or DE440).
///
/// This function uses high-precision NAIF DE ephemeris kernels for Venus position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate Venus' position
/// * `source` - Ephemeris source (DE440s or DE440)
///
/// # Returns
///
/// * `Ok(Vector3<f64>)` - Position of Venus in the GCRF frame. Units: [m]
/// * `Err(BraheError)` - If LowPrecision source is specified or ephemeris query fails
///
/// # Errors
///
/// * Returns error if LowPrecision source is specified (not supported)
/// * Returns error if ephemeris kernel cannot be loaded or queried
///
/// # Example
///
/// ```
/// use brahe::ephemerides::venus_position_de;
/// use brahe::propagators::force_model_config::EphemerisSource;
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
///
/// // Get Venus' position in GCRF frame using DE440s
/// let r_venus = venus_position_de(epc, EphemerisSource::DE440s)?;
/// # Ok::<(), brahe::utils::BraheError>(())
/// ```
pub fn venus_position_de(epc: Epoch, source: EphemerisSource) -> Result<Vector3<f64>, BraheError> {
    // Ensure correct kernel is loaded
    let ctx = ensure_kernel_loaded(source)?;

    // Convert Brahe Epoch to Anise Epoch
    let anise_epoch = brahe_epoch_to_anise(epc);

    // Get Venus' position from ephemeris
    let r_venus_eme2000 = ctx
        .translate(
            anise_frames::VENUS_J2000, // Target
            anise_frames::EME2000,     // Observer
            anise_epoch,
            None,
        )
        .map_err(|e| BraheError::Error(format!("Failed to query Venus position: {}", e)))?;

    // Convert from km to meters
    let r_venus_eme2000_m = Vector3::new(
        r_venus_eme2000.radius_km[0] * 1.0e3,
        r_venus_eme2000.radius_km[1] * 1.0e3,
        r_venus_eme2000.radius_km[2] * 1.0e3,
    );

    // Transform to GCRF frame
    Ok(rotation_eme2000_to_gcrf() * r_venus_eme2000_m)
}

/// Calculate the position of the solar system barycenter in the GCRF inertial frame using NAIF DE440s ephemeris.
///
/// This function uses the high-precision NAIF DE ephemeris kernel (DE440s or DE440) for solar
/// system barycenter position computation. The kernel is loaded once and cached in a global
/// thread-safe context, making subsequent calls very efficient.
///
/// # Arguments
///
/// - `epc`: Epoch at which to calculate the solar system barycenter's position
/// - `source`: Ephemeris source to use (DE440s or DE440)
///
/// # Returns
///
/// - `Ok(r)`: Position of the solar system barycenter in the GCRF frame. Units: [m]
/// - `Err`: If the ephemeris kernel cannot be loaded or the query fails
///
/// # Errors
///
/// Returns an error if the DE kernel cannot be loaded or if the ephemeris query fails.
///
/// # Example
///
/// ```
/// use brahe::ephemerides::solar_system_barycenter_position_de;
/// use brahe::propagators::force_model_config::EphemerisSource;
/// use brahe::time::Epoch;
/// use brahe::TimeSystem;
///
/// let epc = Epoch::from_date(2024, 2, 25, TimeSystem::UTC);
///
/// // Get the solar system barycenter's position in GCRF frame using DE440s
/// let r_solar_system_barycenter = solar_system_barycenter_position_de(epc, EphemerisSource::DE440s).unwrap();
/// ```
pub fn solar_system_barycenter_position_de(
    epc: Epoch,
    source: EphemerisSource,
) -> Result<Vector3<f64>, BraheError> {
    // Ensure the correct kernel is loaded
    let ctx = ensure_kernel_loaded(source)?;

    // Convert Brahe Epoch to Anise Epoch
    let anise_epoch = brahe_epoch_to_anise(epc);

    // Get the solar system barycenter's position from ephemeris
    let r_solar_system_barycenter_eme2000 = ctx
        .translate(
            anise_frames::SSB_J2000, // Target
            anise_frames::EME2000,   // Observer
            anise_epoch,
            None,
        )
        .map_err(|e| {
            BraheError::Error(format!(
                "Failed to query solar system barycenter position: {}",
                e
            ))
        })?;

    // Convert from km to meters
    let r_solar_system_barycenter_eme2000_m = Vector3::new(
        r_solar_system_barycenter_eme2000.radius_km[0] * 1.0e3,
        r_solar_system_barycenter_eme2000.radius_km[1] * 1.0e3,
        r_solar_system_barycenter_eme2000.radius_km[2] * 1.0e3,
    );

    // Transform to GCRF frame
    Ok(rotation_eme2000_to_gcrf() * r_solar_system_barycenter_eme2000_m)
}

/// Convenience alias for `solar_system_barycenter_position_de`.
///
/// Calculate the position of the Solar System Barycenter in the GCRF inertial frame using
/// NAIF DE ephemeris.
///
/// # Arguments
///
/// * `epc` - Epoch at which to calculate the Solar System Barycenter position
/// * `source` - Ephemeris source to use (DE440s or DE440)
///
/// # Returns
///
/// * `Ok`: Position of the Solar System Barycenter in the GCRF frame. Units: (m)
/// * `Err`: If the ephemeris kernel cannot be loaded or the query fails
pub fn ssb_position_de(epc: Epoch, source: EphemerisSource) -> Result<Vector3<f64>, BraheError> {
    solar_system_barycenter_position_de(epc, source)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use rstest::rstest;

    use super::*;
    use crate::frames::rotation_gcrf_to_eme2000;
    use crate::utils::testing::setup_global_test_almanac;

    #[rstest]
    #[case(60310.0, 24622331959.5803, - 133060326832.922, - 57688711921.8327)]
    #[case(60310.0416666667, 24729796439.5928, - 133043454385.773, - 57681396820.7343)]
    #[case(60310.0833333333, 24837247557.3983, - 133026510053.894, - 57674050553.7908)]
    #[case(60310.125, 24944685254.8774, - 133009493846.297, - 57666673124.91)]
    #[case(60310.1666666667, 25052109473.9791, - 132992405772.026, - 57659264538.0129)]
    #[case(60310.2083333333, 25159520156.6597, - 132975245840.167, - 57651824797.0383)]
    #[case(60310.25, 25266917244.8224, - 132958014059.854, - 57644353905.9469)]
    #[case(60310.2916666667, 25374300680.4381, - 132940710440.254, - 57636851868.7128)]
    #[case(60310.3333333333, 25481670405.4859, - 132923334990.575, - 57629318689.3278)]
    #[case(60310.375, 25589026361.8913, - 132905887720.074, - 57621754371.8058)]
    #[case(60310.4166666667, 25696368491.6483, - 132888368638.04, - 57614158920.1738)]
    #[case(60310.4583333333, 25803696736.7582, - 132870777753.803, - 57606532338.4766)]
    #[case(60310.5, 25911011039.1696, - 132853115076.742, - 57598874630.7812)]
    #[case(60310.5416666667, 26018311340.903, - 132835380616.268, - 57591185801.1675)]
    #[case(60310.5833333333, 26125597583.9727, - 132817574381.834, - 57583465853.7339)]
    #[case(60310.625, 26232869710.364, - 132799696382.941, - 57575714792.5993)]
    #[case(60310.6666666667, 26340127662.107, - 132781746629.124, - 57567932621.8976)]
    #[case(60310.7083333333, 26447371381.2497, - 132763725129.956, - 57560119345.7796)]
    #[case(60310.75, 26554600809.7973, - 132745631895.061, - 57552274968.4175)]
    #[case(60310.7916666667, 26661815889.8041, - 132727466934.095, - 57544399493.998)]
    #[case(60310.8333333333, 26769016563.345, - 132709230256.754, - 57536492926.7245)]
    #[case(60310.875, 26876202772.4389, - 132690921872.785, - 57528555270.8231)]
    #[case(60310.9166666667, 26983374459.1741, - 132672541791.966, - 57520586530.5326)]
    #[case(60310.9583333333, 27090531565.6459, - 132654090024.114, - 57512586710.1098)]
    #[case(60311.0, 27197674033.8982, - 132635566579.098, - 57504555813.8335)]
    fn test_sun_position(#[case] mjd_tt: f64, #[case] px: f64, #[case] py: f64, #[case] pz: f64) {
        let epc = Epoch::from_mjd(mjd_tt, TimeSystem::TT);
        // Need to convert from GCRF to EME2000 for comparison with reference data
        let p = rotation_gcrf_to_eme2000() * sun_position(epc);

        assert_abs_diff_eq!(
            epc.mjd_as_time_system(TimeSystem::TT),
            mjd_tt,
            epsilon = 1e-9
        );
        // Given slight differences from how time is initialized (via floating point conversion)
        // We consider these two equivalent if they are within 1.0 m
        assert_abs_diff_eq!(p[0], px, epsilon = 1.0);
        assert_abs_diff_eq!(p[1], py, epsilon = 1.0);
        assert_abs_diff_eq!(p[2], pz, epsilon = 1.0);
    }

    #[rstest]
    #[case(60310.0, - 367995522.308997, 142596488.428594, 89284714.7899626)]
    #[case(60310.0416666667, - 369455605.1617, 139781983.996656, 87830337.122483)]
    #[case(60310.0833333333, - 370886358.974318, 136956734.105683, 86369129.9895069)]
    #[case(60310.125, - 372287676.366997, 134120963.564647, 84901210.1179443)]
    #[case(60310.1666666667, - 373659451.967227, 131274897.728386, 83426694.6122531)]
    #[case(60310.2083333333, - 375001582.408992, 128418762.486828, 81945700.9487947)]
    #[case(60310.25, - 376313966.331919, 125552784.254334, 80458346.9703463)]
    #[case(60310.2916666667, - 377596504.381675, 122677189.956447, 78964750.8790742)]
    #[case(60310.3333333333, - 378849099.209013, 119792207.019449, 77465031.2310408)]
    #[case(60310.375, - 380071655.468602, 116898063.360552, 75959306.9311214)]
    #[case(60310.4166666667, - 381264079.819666, 113994987.373979, 74447697.2256373)]
    #[case(60310.4583333333, - 382426280.924473, 111083207.921993, 72930321.6975378)]
    #[case(60310.5, - 383558169.447324, 108162954.324949, 71407300.261266)]
    #[case(60310.5416666667, - 384659658.054943, 105234456.347874, 69878753.155555)]
    #[case(60310.5833333333, - 385730661.414956, 102297944.191818, 68344800.9388653)]
    #[case(60310.625, - 386771096.194816, 99353648.4843795, 66805564.4843179)]
    #[case(60310.6666666667, - 387780881.061889, 96401800.267068, 65261164.9729798)]
    #[case(60310.7083333333, - 388759936.682298, 93442630.9860015, 63711723.8888798)]
    #[case(60310.75, - 389708185.719446, 90476372.4840198, 62157363.0147493)]
    #[case(60310.7916666667, - 390625552.834187, 87503256.987822, 60598204.425146)]
    #[case(60310.8333333333, - 391511964.683671, 84523517.0991419, 59034370.4817298)]
    #[case(60310.875, - 392367349.919748, 81537385.7878371, 57465983.8294244)]
    #[case(60310.9166666667, - 393191639.189418, 78545096.3782326, 55893167.3891384)]
    #[case(60310.9583333333, - 393984765.133349, 75546882.5419835, 54316044.3538419)]
    #[case(60311.0, - 394746662.3846, 72542978.2908859, 52734738.1846248)]
    fn test_moon_position(#[case] mjd_tt: f64, #[case] px: f64, #[case] py: f64, #[case] pz: f64) {
        let epc = Epoch::from_mjd(mjd_tt, TimeSystem::TT);
        // Need to convert from GCRF to EME2000 for comparison with reference data
        let p = rotation_gcrf_to_eme2000() * moon_position(epc);

        assert_abs_diff_eq!(
            epc.mjd_as_time_system(TimeSystem::TT),
            mjd_tt,
            epsilon = 1e-9
        );
        // Given slight differences from how time is initialized (via floating point conversion)
        // We consider these two equivalent if they are within 1.0 m
        assert_abs_diff_eq!(p[0], px, epsilon = 1.0);
        assert_abs_diff_eq!(p[1], py, epsilon = 1.0);
        assert_abs_diff_eq!(p[2], pz, epsilon = 1.0);
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_sun_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();

        let epc = Epoch::from_date(year, month, day, TimeSystem::UTC);
        let r_sun_analytical = sun_position(epc);
        let r_sun_de = sun_position_de(epc, EphemerisSource::DE440s).unwrap();

        // Compute the dot product and confirm the angle is less than 1 degree
        let dot_product =
            r_sun_analytical.dot(&r_sun_de) / (r_sun_analytical.norm() * r_sun_de.norm());
        let angle = dot_product.acos() * (180.0 / std::f64::consts::PI);

        assert_abs_diff_eq!(angle, 0.0, epsilon = 1.0e-1);
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_moon_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();

        let epc = Epoch::from_date(year, month, day, TimeSystem::UTC);
        let r_moon_analytical = moon_position(epc);
        let r_moon_de = moon_position_de(epc, EphemerisSource::DE440s).unwrap();

        // Compute the dot product and confirm the angle is less than 1 degree
        let dot_product =
            r_moon_analytical.dot(&r_moon_de) / (r_moon_analytical.norm() * r_moon_de.norm());
        let angle = dot_product.acos() * (180.0 / std::f64::consts::PI);

        assert_abs_diff_eq!(angle, 0.0, epsilon = 1.0e-1);
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_jupiter_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();

        let epc = Epoch::from_date(year, month, day, TimeSystem::UTC);
        // Just call and ensure no panic occurs
        let _r_jupiter_de = jupiter_position_de(epc, EphemerisSource::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_mars_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();

        let epc = Epoch::from_date(year, month, day, TimeSystem::UTC);
        // Just call and ensure no panic occurs
        let _r_mars_de = mars_position_de(epc, EphemerisSource::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_mercury_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();

        let epc = Epoch::from_date(year, month, day, TimeSystem::UTC);
        // Just call and ensure no panic occurs
        let _r_mercury_de = mercury_position_de(epc, EphemerisSource::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_neptune_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();

        let epc = Epoch::from_date(year, month, day, TimeSystem::UTC);
        // Just call and ensure no panic occurs
        let _r_neptune_de = neptune_position_de(epc, EphemerisSource::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_saturn_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();

        let epc = Epoch::from_date(year, month, day, TimeSystem::UTC);
        // Just call and ensure no panic occurs
        let _r_saturn_de = saturn_position_de(epc, EphemerisSource::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_uranus_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();

        let epc = Epoch::from_date(year, month, day, TimeSystem::UTC);
        // Just call and ensure no panic occurs
        let _r_uranus_de = uranus_position_de(epc, EphemerisSource::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_venus_position_de(#[case] year: u32, #[case] month: u8, #[case] day: u8) {
        setup_global_test_almanac();

        let epc = Epoch::from_date(year, month, day, TimeSystem::UTC);
        // Just call and ensure no panic occurs
        let _r_venus_de = venus_position_de(epc, EphemerisSource::DE440s).unwrap();
    }

    #[rstest]
    #[case(2025, 1, 1)]
    #[case(2025, 2, 15)]
    #[case(2025, 3, 30)]
    #[case(2025, 5, 15)]
    #[case(2025, 7, 1)]
    #[case(2025, 8, 15)]
    #[case(2025, 10, 1)]
    #[case(2025, 11, 15)]
    #[case(2025, 12, 31)]
    fn test_solar_system_barycenter_position_de(
        #[case] year: u32,
        #[case] month: u8,
        #[case] day: u8,
    ) {
        setup_global_test_almanac();

        let epc = Epoch::from_date(year, month, day, TimeSystem::UTC);
        // Just call and ensure no panic occurs
        let _r_ssb_de = solar_system_barycenter_position_de(epc, EphemerisSource::DE440s).unwrap();
    }

    #[test]
    fn test_kernel_auto_switching() {
        // Initialize with DE440s
        initialize_ephemeris_with_kernel("de440s").unwrap();
        assert_eq!(get_loaded_kernel_type(), Some("de440s".to_string()));

        // Request DE440 - should auto-switch
        let result = ensure_kernel_loaded(EphemerisSource::DE440);
        assert!(result.is_ok());
        assert_eq!(get_loaded_kernel_type(), Some("de440".to_string()));

        // Request DE440s again - should switch back
        let result = ensure_kernel_loaded(EphemerisSource::DE440s);
        assert!(result.is_ok());
        assert_eq!(get_loaded_kernel_type(), Some("de440s".to_string()));
    }

    #[test]
    fn test_low_precision_source_error() {
        let result = ensure_kernel_loaded(EphemerisSource::LowPrecision);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(e.to_string().contains("Low-precision"));
        }
    }
}
