//! Shared ANISE Almanac for benchmark tasks that need PCK kernel data,
//! plus helpers for the controlled EOP and space-weather input files.
//!
//! `MetaAlmanac::latest()` auto-downloads ANISE's default kernel set on
//! first use (~50 MB to ~/.local/share/nyx-space/anise/ or platform
//! equivalent) and caches them for subsequent runs. Construction cost is
//! paid once per process via OnceLock; per-task timing only measures task
//! work.
//!
//! Kernel set (from MetaAlmanac::default()):
//!   - de440s.bsp  — solar system ephemerides
//!   - pck11.pca   — planetary constants (ellipsoid a, f for Earth etc.)
//!   - moon_fk_de440.epa — Moon frame kernel
//!   - moon_pa_de440_200625.bpc — Moon orientation BPC
//!   - earth_latest_high_prec.bpc — Earth rotation (ITRF93↔J2000)
//!
//! # Controlled inputs
//!
//! The benchmark suite uses the same EOP and space-weather files as the
//! Orekit (Java) and brahe-Rust reference implementations, both sourced
//! from `$OREKIT_DATA`:
//!
//!   - EOP:  `Earth-Orientation-Parameters/IAU-2000/finals2000A.all`
//!   - SW:   `CSSI-Space-Weather-Data/SpaceWeather-All-v1.2.txt`
//!
//! ## ANISE 0.10.1 EOP support
//!
//! ANISE 0.10.1 does **not** expose an IERS finals2000A loader. Earth
//! rotation corrections (UT1, polar motion) are embedded in the high-
//! precision BPC kernel (`earth_latest_high_prec.bpc`) which is part of
//! the default kernel set downloaded by `MetaAlmanac::latest()`. No
//! additional attachment is possible.
//!
//! ## hifitime 4.3.0 EOP support
//!
//! hifitime's `Ut1Provider` accepts the JPL EOP2 CSV format (comma-
//! separated, with an `EOP2=` header). The IERS finals2000A file uses a
//! different fixed-width format. No direct `from_finals2000a` constructor
//! exists. Conversion is non-trivial (requires leap-second accounting for
//! each daily record); the spec permits accepting this confound and using
//! `Ut1Provider::default()` (zero DUT1).
//!
//! ## Nyx 2.4.0 space-weather support
//!
//! Nyx's atmosphere/drag models (`ConstantDrag`, `Drag`) use internally
//! specified density models (exponential, standard 1976) and do not
//! expose a CSSI space-weather file loader. The CSSI SW file path helper
//! is exposed here for future tasks.

use anise::almanac::Almanac;
use platform_dirs::AppDirs;
use std::path::PathBuf;
use std::sync::OnceLock;

static ALMANAC: OnceLock<Almanac> = OnceLock::new();

/// Return the path to the ANISE kernel cache directory.
fn anise_cache_dir() -> PathBuf {
    AppDirs::new(Some("nyx-space/anise"), true)
        .map(|d| d.data_dir)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_default();
            PathBuf::from(home).join(".local/share/nyx-space/anise")
        })
}

/// Return a reference to the lazily-initialized global Almanac.
///
/// On first call this loads ANISE kernels from the local cache directory
/// (populated by a prior `MetaAlmanac::latest()` call or manual placement).
/// Subsequent calls return the cached instance immediately.
///
/// The Almanac includes `earth_latest_high_prec.bpc` which embeds IERS
/// EOP corrections (UT1, polar motion) as of the kernel's publish date.
/// No additional finals2000A attachment is performed because ANISE 0.10.1
/// has no IERS-format EOP loader (see module-level docs).
///
/// Kernel loading order matches `MetaAlmanac::default()`:
///   de440s.bsp → pck11.pca → moon_fk_de440.epa → moon_pa_de440_200625.bpc
///   → earth_latest_high_prec.bpc
pub fn almanac() -> &'static Almanac {
    ALMANAC.get_or_init(|| {
        let cache = anise_cache_dir();
        let kernel_files = [
            "de440s.bsp",
            "pck11.pca",
            "moon_fk_de440.epa",
            "moon_pa_de440_200625.bpc",
            "earth_latest_high_prec.bpc",
        ];

        let mut alm = Almanac::default();
        for file in &kernel_files {
            let path = cache.join(file);
            let path_str = path.to_string_lossy();
            alm = alm.load(&path_str).unwrap_or_else(|e| {
                panic!(
                    "Failed to load ANISE kernel '{}': {e}\n\
                     Ensure kernels are cached at {}\n\
                     Run `bench_nyx` once with network access to populate the cache.",
                    path_str,
                    cache.display()
                )
            });
        }
        let alm = alm;

        // Attempt to attach IERS EOP from OREKIT_DATA.
        // ANISE 0.10.1 has no finals2000A loader; EOP is already baked into
        // earth_latest_high_prec.bpc. Emit an informational message and proceed.
        match find_orekit_eop_file() {
            Some(eop_path) => {
                eprintln!(
                    "INFO [data.rs]: IERS EOP file found at {}. \
                     ANISE 0.10.1 does not expose a finals2000A ingest API; \
                     EOP corrections are embedded in earth_latest_high_prec.bpc \
                     (the ANISE default). Controlled EOP confound accepted per spec.",
                    eop_path.display()
                );
            }
            None => {
                eprintln!(
                    "INFO [data.rs]: OREKIT_DATA finals2000A.all not found \
                     (OREKIT_DATA env var unset or file missing). \
                     Using ANISE default EOP from earth_latest_high_prec.bpc."
                );
            }
        }

        // Attempt to note CSSI space-weather availability.
        // Nyx 2.4.0's drag/atmosphere models use internally specified density
        // tables and do not accept a CSSI SW file. Emit an informational message.
        match find_orekit_sw_file() {
            Some(sw_path) => {
                eprintln!(
                    "INFO [data.rs]: CSSI space-weather file found at {}. \
                     Nyx 2.4.0 atmosphere models do not expose a CSSI SW loader; \
                     using internally specified density tables. SW confound accepted per spec.",
                    sw_path.display()
                );
            }
            None => {
                eprintln!(
                    "INFO [data.rs]: OREKIT_DATA SpaceWeather-All-v1.2.txt not found. \
                     Using Nyx default atmosphere density tables."
                );
            }
        }

        alm
    })
}

/// Path to the controlled IERS EOP file (`finals2000A.all`) from
/// `$OREKIT_DATA`, if present.
///
/// `$OREKIT_DATA` defaults to `~/.orekit/orekit-data` when unset.
///
/// Tasks that can ingest this file directly (e.g., a future hifitime
/// version with a `from_finals2000a` constructor) may call this helper
/// and load the file themselves.
pub fn find_orekit_eop_file() -> Option<PathBuf> {
    let dir = std::env::var("OREKIT_DATA").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.orekit/orekit-data")
    });
    let p = PathBuf::from(dir)
        .join("Earth-Orientation-Parameters")
        .join("IAU-2000")
        .join("finals2000A.all");
    if p.exists() { Some(p) } else { None }
}

/// Path to the controlled CSSI space-weather file
/// (`SpaceWeather-All-v1.2.txt`) from `$OREKIT_DATA`, if present.
///
/// `$OREKIT_DATA` defaults to `~/.orekit/orekit-data` when unset.
///
/// Drag tasks may load this directly into an atmosphere model when a
/// suitable loader is available (not yet supported in Nyx 2.4.0).
pub fn find_orekit_sw_file() -> Option<PathBuf> {
    let dir = std::env::var("OREKIT_DATA").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.orekit/orekit-data")
    });
    let p = PathBuf::from(dir)
        .join("CSSI-Space-Weather-Data")
        .join("SpaceWeather-All-v1.2.txt");
    if p.exists() { Some(p) } else { None }
}
