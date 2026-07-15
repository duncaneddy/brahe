/*!
FES2004 ocean tide geopotential corrections (IERS Conventions 2010, TN36 §6.3).

Data source: FES2004 normalized Stokes-coefficient amplitude file, downloaded
once into `$BRAHE_CACHE/tides/` from
<https://iers-conventions.obspm.fr/content/chapter6/additional_info/tidemodels/fes2004_Cnm-Snm.dat>.
*/

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::orbit_dynamics::ocean_tides_admittance::ADMITTANCE_TABLE;
use crate::orbit_dynamics::tides::TideDeltas;
use crate::utils::BraheError;
use crate::utils::cache::get_tides_cache_dir;
use crate::utils::download::download_bytes;
use crate::utils::fs::atomic_write;

const FES2004_URL: &str = "https://iers-conventions.obspm.fr/content/chapter6/additional_info/tidemodels/fes2004_Cnm-Snm.dat";
const FES2004_FILENAME: &str = "fes2004_Cnm-Snm.dat";

/// Minimum line count of a complete FES2004 coefficient file (the canonical
/// IERS file has 59,466 lines). A freshly downloaded file shorter than this is
/// truncated and is rejected before being written to the cache.
const FES2004_MIN_LINES: usize = 59_000;

/// The 18 FES2004 main-wave Doodson numbers present in the coefficient file
/// (TN36 §6.3.2; S1 is excluded — it has no admittance pivots and is not in the
/// file). Every one must be present for a cached file to be considered intact,
/// and each is an admittance pivot referenced by Table 6.7.
pub(crate) const FES2004_MAIN_WAVE_DOODSON: [&str; 18] = [
    "55.565", "55.575", "56.554", "57.555", "65.455", "75.555", "85.455", "93.555", "135.655",
    "145.555", "163.555", "165.555", "235.755", "245.655", "255.555", "273.555", "275.555",
    "455.555",
];

/// Path to the cached FES2004 ocean tide coefficient file, downloading it
/// (one-time, ~3.7 MB) from IERS if not already cached.
///
/// # Returns
///
/// * `PathBuf` - Location of `fes2004_Cnm-Snm.dat` inside `$BRAHE_CACHE/tides/`.
///
/// # Errors
///
/// Returns `BraheError` if the tides cache directory cannot be created, or if
/// no cached copy exists and the download from IERS fails. The error message
/// names the URL and the target cache path so the file can be fetched
/// manually to proceed offline.
pub fn fes2004_coefficients_path() -> Result<PathBuf, BraheError> {
    let dir = PathBuf::from(get_tides_cache_dir()?);
    let path = dir.join(FES2004_FILENAME);
    if path.exists() {
        return Ok(path);
    }

    // Serialize downloads within this process so two threads that both find an
    // empty cache don't fetch concurrently and collide on the PID-based temp
    // name used by `atomic_write`. Cross-process races remain possible but are
    // safe: distinct PIDs produce distinct temp names and the final rename is
    // atomic, so the worst case is redundant downloads, never a corrupt file.
    static FES2004_DOWNLOAD_LOCK: Mutex<()> = Mutex::new(());
    let _download_guard = FES2004_DOWNLOAD_LOCK.lock().unwrap();
    // Another thread may have completed the download while we waited for the lock.
    if path.exists() {
        return Ok(path);
    }

    let buf = download_bytes(FES2004_URL).map_err(|e| {
        BraheError::Error(format!(
            "FES2004 ocean tide coefficients are not cached and the download from {FES2004_URL} \
             failed: {e}. Download the file manually to {} to proceed offline.",
            path.display()
        ))
    })?;
    if buf.is_empty() {
        return Err(BraheError::Error(format!(
            "Empty response downloading FES2004 ocean tide coefficients from {FES2004_URL}"
        )));
    }
    // Validate the freshly downloaded file is complete before promoting it into
    // the cache, so a truncated download is never atomically written and later
    // parsed as if it were valid.
    let line_count = buf.iter().filter(|&&b| b == b'\n').count();
    if line_count < FES2004_MIN_LINES {
        return Err(BraheError::Error(format!(
            "Downloaded FES2004 ocean tide file from {FES2004_URL} is truncated \
             ({line_count} lines, expected at least {FES2004_MIN_LINES}); it was not \
             cached. Retry, or download it manually to {}.",
            path.display()
        )));
    }
    atomic_write(&path, &buf).map_err(|e| {
        BraheError::Error(format!(
            "Failed to write FES2004 cache {}: {}",
            path.display(),
            e
        ))
    })?;
    Ok(path)
}

/// One `(n, m)` coefficient row of a tidal constituent: prograde (+) and
/// retrograde (−) fully-normalized geopotential amplitudes C±/S± (IERS TN36
/// Eq. 6.15 inputs), dimensionless (file units of 1e-11 already applied).
#[derive(Debug, Clone, Copy)]
pub(crate) struct OceanTideLine {
    pub n: u16,
    pub m: u16,
    pub c_plus: f64,
    pub s_plus: f64,
    pub c_minus: f64,
    pub s_minus: f64,
}

/// One tidal constituent: Doodson multipliers, Darwin name, precomputed
/// multipliers of `[γ, l, l′, F, D, Ω]` (Task 6), and coefficient lines.
#[derive(Debug, Clone)]
pub(crate) struct OceanTideConstituent {
    pub doodson: [i8; 6],
    #[allow(dead_code)] // diagnostics only: Darwin name (e.g. "M2") carried for debugging
    pub name: String,
    pub delaunay: [i32; 6],
    pub lines: Vec<OceanTideLine>,
}

/// Parse a Doodson number string ("255.555", " 55.565") into the six
/// fundamental-variable multipliers `[kτ, ks, kh, kp, kN′, kps]`: first digit
/// raw, digits 2..6 biased by −5 (Doodson's convention).
///
/// # Arguments
///
/// * `s` - Doodson number string, e.g. `"255.555"` (dimensionless digits).
///
/// # Returns
///
/// * `[i8; 6]` - Multipliers `[kτ, ks, kh, kp, kN′, kps]` (dimensionless).
///
/// # Errors
///
/// Returns `BraheError` if `s` is not of the form `DDD.DDD` (three digits, a
/// literal `.`, then three digits).
pub(crate) fn parse_doodson(s: &str) -> Result<[i8; 6], BraheError> {
    let t = s.trim();
    let (head, tail) = t
        .split_once('.')
        .ok_or_else(|| BraheError::Error(format!("invalid Doodson number '{s}': missing '.'")))?;
    if tail.len() != 3 || head.is_empty() || head.len() > 3 {
        return Err(BraheError::Error(format!("invalid Doodson number '{s}'")));
    }
    let digits: Vec<i8> = format!("{:0>3}{}", head, tail)
        .chars()
        .map(|c| {
            c.to_digit(10)
                .map(|d| d as i8)
                .ok_or_else(|| BraheError::Error(format!("invalid Doodson number '{s}'")))
        })
        .collect::<Result<_, _>>()?;
    Ok([
        digits[0],
        digits[1] - 5,
        digits[2] - 5,
        digits[3] - 5,
        digits[4] - 5,
        digits[5] - 5,
    ])
}

/// Convert Doodson multipliers `[kτ, ks, kh, kp, kN′, kps]` to additive
/// multipliers of `[γ = GMST+π, l, l′, F, D, Ω]`, the argument vector
/// returned by [`crate::orbit_dynamics::tides::doodson_delaunay_args`].
///
/// The identities are s = F + Ω, h = s − D, p = s − l, N′ = −Ω,
/// ps = s − D − l′, τ = γ − s (IERS TN36 §6.2.1 explanatory text below
/// Eq. 6.8e). Matches the conversion in Orekit's `OceanTidesWave`
/// constructor (independently cross-checked).
///
/// # Arguments
///
/// * `k` - Doodson multipliers `[kτ, ks, kh, kp, kN′, kps]` (dimensionless).
///
/// # Returns
///
/// * `[i32; 6]` - Multipliers of `[γ, l, l′, F, D, Ω]` (dimensionless).
pub(crate) fn doodson_to_delaunay(k: [i8; 6]) -> [i32; 6] {
    let (kt, ks, kh, kp, kn, kps) = (
        k[0] as i32,
        k[1] as i32,
        k[2] as i32,
        k[3] as i32,
        k[4] as i32,
        k[5] as i32,
    );
    [
        kt,                            // γ
        -kp,                           // l
        -kps,                          // l′
        -kt + ks + kh + kp + kps,      // F
        -kh - kps,                     // D
        -kt + ks + kh + kp - kn + kps, // Ω
    ]
}

/// Mean rates of the Doodson fundamental variables (τ, s, h, p, N′, ps) in
/// degrees per hour, used only for the time-independent admittance
/// interpolation weights of Eq. (6.16). Values reproduce the "deg/hr" column
/// of TN36 Table 6.5c (e.g. M2 = 2τ̇ = 28.98410).
const DOODSON_RATES_DEG_PER_HOUR: [f64; 6] = [
    14.49205211,
    0.54901653,
    0.04106864,
    0.00464183,
    0.00220641,
    0.00000196,
];

/// Rate θ̇f of a constituent's Doodson argument.
///
/// # Arguments
///
/// * `k` - Doodson multipliers `[kτ, ks, kh, kp, kN′, kps]` (dimensionless).
///
/// # Returns
///
/// * `f64` - Argument rate θ̇f [deg/hour].
pub(crate) fn theta_dot_deg_per_hour(k: [i8; 6]) -> f64 {
    k.iter()
        .zip(DOODSON_RATES_DEG_PER_HOUR.iter())
        .map(|(ki, ri)| *ki as f64 * ri)
        .sum()
}

/// Doodson argument θf from precomputed `[γ, l, l′, F, D, Ω]` multipliers and
/// the argument vector of
/// [`crate::orbit_dynamics::tides::doodson_delaunay_args`].
///
/// # Arguments
///
/// * `delaunay` - Multipliers of `[γ, l, l′, F, D, Ω]` (dimensionless), from
///   [`doodson_to_delaunay`].
/// * `args` - `[γ, l, l′, F, D, Ω]` (radians), from
///   [`crate::orbit_dynamics::tides::doodson_delaunay_args`].
///
/// # Returns
///
/// * `f64` - Doodson argument θf [rad].
pub(crate) fn tide_argument(delaunay: &[i32; 6], args: &[f64; 6]) -> f64 {
    delaunay
        .iter()
        .zip(args.iter())
        .map(|(ki, ai)| *ki as f64 * ai)
        .sum()
}

/// Parse the FES2004 coefficient file (IERS TN36 §6.3.2), truncated to
/// `n_max`/`m_max`, grouped by constituent in file order. Degree 0 and 1 rows
/// are skipped: degree-1 terms represent geocenter motion, not geopotential
/// coefficients about the Earth center of mass (consistent with Orekit's
/// OceanTidesWave, START_DEGREE = 2). Units of 1e-11 are applied.
///
/// # Arguments
///
/// * `path` - Path to the FES2004 `Cnm`/`Snm` coefficient file.
/// * `n_max` - Maximum spherical harmonic degree to retain (dimensionless).
/// * `m_max` - Maximum spherical harmonic order to retain (dimensionless).
///
/// # Returns
///
/// * `Vec<OceanTideConstituent>` - Constituents in file order, each holding
///   its truncated set of coefficient lines.
///
/// # Errors
///
/// Returns `BraheError` if the file cannot be read, if a data row has a
/// malformed degree, order, or coefficient field, or if no constituents were
/// parsed (indicating a corrupt or truncated file).
pub(crate) fn parse_fes2004_file(
    path: &Path,
    n_max: usize,
    m_max: usize,
) -> Result<Vec<OceanTideConstituent>, BraheError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| BraheError::Error(format!("cannot read {}: {e}", path.display())))?;
    const SCALE: f64 = 1e-11;
    let mut waves: Vec<OceanTideConstituent> = Vec::new();
    for (lineno, line) in content.lines().enumerate() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        // Data rows: "55.565 Om1 2 0 -6.58128 0.00000 -0.00000 -0.00000".
        // Header lines fail the Doodson parse / field count and are skipped.
        if fields.len() != 8 {
            continue;
        }
        let Ok(doodson) = parse_doodson(fields[0]) else {
            continue;
        };
        let n: u16 = fields[2].parse().map_err(|e| {
            BraheError::Error(format!(
                "{}:{}: bad degree: {e}",
                path.display(),
                lineno + 1
            ))
        })?;
        let m: u16 = fields[3].parse().map_err(|e| {
            BraheError::Error(format!("{}:{}: bad order: {e}", path.display(), lineno + 1))
        })?;
        // Reject physically impossible rows: an order exceeding the degree would
        // alias another packed (n, m) entry in the accumulator (silently, in
        // release builds), corrupting the coefficient set.
        if m > n {
            return Err(BraheError::Error(format!(
                "{}:{}: invalid coefficient row: order m={m} exceeds degree n={n}",
                path.display(),
                lineno + 1
            )));
        }
        if n < 2 || (n as usize) > n_max || (m as usize) > m_max {
            continue;
        }
        let mut vals = [0.0f64; 4];
        for (i, f) in fields[4..8].iter().enumerate() {
            vals[i] = f.parse::<f64>().map_err(|e| {
                BraheError::Error(format!(
                    "{}:{}: bad coefficient: {e}",
                    path.display(),
                    lineno + 1
                ))
            })? * SCALE;
        }
        let entry = OceanTideLine {
            n,
            m,
            c_plus: vals[0],
            s_plus: vals[1],
            c_minus: vals[2],
            s_minus: vals[3],
        };
        match waves.last_mut() {
            Some(w) if w.doodson == doodson => w.lines.push(entry),
            _ => waves.push(OceanTideConstituent {
                doodson,
                name: fields[1].to_string(),
                delaunay: doodson_to_delaunay(doodson),
                lines: vec![entry],
            }),
        }
    }
    if waves.is_empty() {
        return Err(BraheError::Error(format!(
            "no FES2004 constituents parsed from {} — file corrupt or truncated",
            path.display()
        )));
    }
    Ok(waves)
}

/// Expand the FES2004 main waves with the secondary (admittance) waves of
/// TN36 Table 6.7 by linear admittance interpolation between the two pivot
/// main waves (Eq. 6.16):
///
/// ```text
/// C±f = (θ̇f−θ̇1)/(θ̇2−θ̇1)·(Hf/H2)·C±2 + (θ̇2−θ̇f)/(θ̇2−θ̇1)·(Hf/H1)·C±1
/// ```
///
/// The weights are time-independent, so the expansion happens once at model
/// load. Rows without pivots are main waves already present in `main` (or
/// S1, excluded per §6.3.2) and are skipped.
///
/// # Arguments
///
/// * `main` - FES2004 main-wave constituents, e.g. from [`parse_fes2004_file`].
///
/// # Returns
///
/// * `Vec<OceanTideConstituent>` - The 63 secondary constituents only; the
///   caller appends these to `main`.
///
/// # Errors
///
/// Returns `BraheError` if a pivot wave named in Table 6.7 is missing from
/// `main` (indicating the file was truncated or parsed incorrectly).
pub(crate) fn expand_admittance(
    main: &[OceanTideConstituent],
) -> Result<Vec<OceanTideConstituent>, BraheError> {
    let find = |doodson: [i8; 6]| main.iter().find(|w| w.doodson == doodson);
    let hf_of = |doodson_str: &str| -> Result<f64, BraheError> {
        ADMITTANCE_TABLE
            .iter()
            .find(|r| parse_doodson(r.doodson).ok() == parse_doodson(doodson_str).ok())
            .map(|r| r.hf)
            .ok_or_else(|| BraheError::Error(format!("pivot {doodson_str} missing from Table 6.7")))
    };

    let mut out = Vec::new();
    for row in ADMITTANCE_TABLE.iter() {
        let Some((p1, p2)) = row.pivots else { continue };
        let kf = parse_doodson(row.doodson)?;
        let k1 = parse_doodson(p1)?;
        let k2 = parse_doodson(p2)?;
        let (w1c, w2c) = {
            let (tf, t1, t2) = (
                theta_dot_deg_per_hour(kf),
                theta_dot_deg_per_hour(k1),
                theta_dot_deg_per_hour(k2),
            );
            let frac = (tf - t1) / (t2 - t1);
            let (h1, h2) = (hf_of(p1)?, hf_of(p2)?);
            ((1.0 - frac) * row.hf / h1, frac * row.hf / h2)
        };
        let c1 = find(k1).ok_or_else(|| {
            BraheError::Error(format!("pivot {p1} of {} not in FES2004 file", row.doodson))
        })?;
        let c2 = find(k2).ok_or_else(|| {
            BraheError::Error(format!("pivot {p2} of {} not in FES2004 file", row.doodson))
        })?;

        // Union of pivot (n, m) grids, weighted combination per line.
        let mut lines: std::collections::BTreeMap<(u16, u16), OceanTideLine> =
            std::collections::BTreeMap::new();
        for (w, c) in [(w1c, c1), (w2c, c2)] {
            for l in &c.lines {
                let e = lines.entry((l.n, l.m)).or_insert(OceanTideLine {
                    n: l.n,
                    m: l.m,
                    c_plus: 0.0,
                    s_plus: 0.0,
                    c_minus: 0.0,
                    s_minus: 0.0,
                });
                e.c_plus += w * l.c_plus;
                e.s_plus += w * l.s_plus;
                e.c_minus += w * l.c_minus;
                e.s_minus += w * l.s_minus;
            }
        }
        out.push(OceanTideConstituent {
            doodson: kf,
            name: row.name.to_string(),
            delaunay: doodson_to_delaunay(kf),
            lines: lines.into_values().collect(),
        });
    }
    Ok(out)
}

/// FES2004 ocean tide model (IERS TN36 §6.3): per-constituent prograde and
/// retrograde geopotential amplitudes, truncated to a degree/order, optionally
/// completed by the Table 6.7 admittance waves.
///
/// # References
/// - IERS Conventions (2010), TN36 §6.3, Eq. (6.15)–(6.16); §6.3.2 (FES2004).
/// - Lyard et al. (2006), FES2004.
pub struct OceanTideModel {
    n_max: usize,
    m_max: usize,
    constituents: Vec<OceanTideConstituent>,
}

impl OceanTideModel {
    /// Enforce the documented truncation bounds (2 <= m_max <= n_max <= 100);
    /// out-of-range values would otherwise silently truncate the model.
    ///
    /// Shared by [`OceanTideModel::from_file`] and [`OceanTideModel::new`]
    /// so that `new` can reject bad bounds *before* touching the cache
    /// file: this is a caller-input error, not evidence of a corrupt cache, and
    /// must never trigger cache deletion.
    ///
    /// # Arguments
    ///
    /// * `n_max`, `m_max` - Truncation degree/order (2 <= m_max <= n_max <= 100).
    ///
    /// # Errors
    ///
    /// Returns `BraheError` if the bounds are out of range.
    fn validate_truncation_bounds(n_max: usize, m_max: usize) -> Result<(), BraheError> {
        if m_max < 2 || n_max > 100 || m_max > n_max {
            return Err(BraheError::Error(format!(
                "invalid FES2004 truncation: require 2 <= m_max <= n_max <= 100, \
                 got n_max={n_max}, m_max={m_max}"
            )));
        }
        Ok(())
    }

    /// Build from an explicit FES2004 coefficient file path.
    ///
    /// # Arguments
    ///
    /// * `path` - FES2004 `fes2004_Cnm-Snm.dat`-format file.
    /// * `n_max`, `m_max` - Truncation degree/order (2 <= m_max <= n_max <= 100).
    /// * `include_admittance` - Expand secondary waves per Eq. (6.16).
    ///
    /// # Returns
    ///
    /// * `OceanTideModel` - Parsed and (optionally) admittance-expanded model.
    ///
    /// # Errors
    ///
    /// Returns `BraheError` if the file cannot be parsed (see
    /// [`parse_fes2004_file`]) or, when `include_admittance` is set, if a
    /// Table 6.7 pivot wave is missing from the file (see
    /// [`expand_admittance`]).
    ///
    /// # Example
    ///
    /// ```no_run
    /// use brahe::OceanTideModel;
    /// use std::path::Path;
    ///
    /// let model = OceanTideModel::from_file(
    ///     Path::new("fes2004_Cnm-Snm.dat"),
    ///     20,
    ///     20,
    ///     true,
    /// )
    /// .unwrap();
    /// assert_eq!(model.n_max(), 20);
    /// ```
    pub fn from_file(
        path: &Path,
        n_max: usize,
        m_max: usize,
        include_admittance: bool,
    ) -> Result<Self, BraheError> {
        Self::validate_truncation_bounds(n_max, m_max)?;
        let mut constituents = parse_fes2004_file(path, n_max, m_max)?;
        if include_admittance {
            let secondary = expand_admittance(&constituents)?;
            constituents.extend(secondary);
        }
        Ok(Self::from_constituents(constituents, n_max, m_max))
    }

    /// Build from the cached (downloading once if needed) IERS FES2004 file.
    /// See [`fes2004_coefficients_path`] for cache/download semantics.
    ///
    /// # Arguments
    ///
    /// * `n_max`, `m_max` - Truncation degree/order (2 <= m_max <= n_max <= 100).
    /// * `include_admittance` - Expand secondary waves per Eq. (6.16).
    ///
    /// # Returns
    ///
    /// * `OceanTideModel` - Parsed and (optionally) admittance-expanded model.
    ///
    /// # Errors
    ///
    /// Returns `BraheError` if `n_max`/`m_max` are out of the documented bounds
    /// (a plain error; the cache is left untouched), if the FES2004
    /// coefficients cannot be located or downloaded (see
    /// [`fes2004_coefficients_path`], likewise left untouched), or if the
    /// cached file itself fails to parse, is missing main constituents, or
    /// does not cover the requested degree — in which case it is treated as
    /// corrupt, removed, and the error notes that it will be re-downloaded on
    /// the next call.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use brahe::OceanTideModel;
    ///
    /// // Downloads FES2004 coefficients into the cache on first use.
    /// let model = OceanTideModel::new(20, 20, true).unwrap();
    /// assert_eq!(model.n_max(), 20);
    /// ```
    pub fn new(n_max: usize, m_max: usize, include_admittance: bool) -> Result<Self, BraheError> {
        // Reject bad truncation bounds up front: this is a caller-input error,
        // not evidence that the cache is corrupt, so it must propagate as a
        // plain error without touching (and, in particular, without deleting)
        // the cache file.
        Self::validate_truncation_bounds(n_max, m_max)?;
        let path = fes2004_coefficients_path()?;
        // A truncated download is rejected before caching (see
        // `fes2004_coefficients_path`), but a pre-existing cache file may have
        // been corrupted out-of-band (partial write from an older build,
        // manual tampering, disk error, or a degree-truncated file placed at
        // the canonical cache path). Validate it and, on failure, remove it so
        // the next call re-downloads a fresh copy.
        Self::from_file(&path, n_max, m_max, include_admittance)
            .and_then(|model| {
                model.validate_main_waves()?;
                model.validate_degree_coverage(n_max)?;
                Ok(model)
            })
            .map_err(|e| {
                let _ = std::fs::remove_file(&path);
                BraheError::Error(format!(
                    "The cached FES2004 file {} is corrupt or truncated ({e}); it has \
                     been removed and will be re-downloaded on the next call, or can be \
                     fetched manually from {FES2004_URL}.",
                    path.display()
                ))
            })
    }

    /// Verify that all 18 FES2004 main-wave constituents (the pivots required by
    /// the Table 6.7 admittance expansion) are present, guarding against a
    /// corrupt or truncated cache that dropped constituents.
    fn validate_main_waves(&self) -> Result<(), BraheError> {
        let present: std::collections::HashSet<[i8; 6]> =
            self.constituents.iter().map(|c| c.doodson).collect();
        for d in FES2004_MAIN_WAVE_DOODSON {
            if !present.contains(&parse_doodson(d)?) {
                return Err(BraheError::Error(format!("missing main constituent {d}")));
            }
        }
        Ok(())
    }

    /// Verify the parsed coefficient lines actually reach the requested
    /// `n_max`, guarding against a cache file that holds all 18 main waves but
    /// only a low-degree subset (e.g. a degree-truncated fixture placed under
    /// the canonical cache name), which would otherwise silently zero out
    /// every high-degree correction instead of erroring.
    ///
    /// [`parse_fes2004_file`] truncates at `n_max`, so a genuine, complete
    /// FES2004 file (degree 100) always yields a maximum parsed degree equal
    /// to `n_max`; anything less means the cached file itself stops short.
    fn validate_degree_coverage(&self, n_max: usize) -> Result<(), BraheError> {
        let max_parsed_degree = self
            .constituents
            .iter()
            .flat_map(|c| &c.lines)
            .map(|l| l.n as usize)
            .max()
            .unwrap_or(0);
        if max_parsed_degree < n_max {
            return Err(BraheError::Error(format!(
                "requested degree coverage n_max={n_max} but the file only contains \
                 coefficients up to degree {max_parsed_degree}"
            )));
        }
        Ok(())
    }

    /// Assemble from already-built constituents (test seam and internal use).
    pub(crate) fn from_constituents(
        constituents: Vec<OceanTideConstituent>,
        n_max: usize,
        m_max: usize,
    ) -> Self {
        OceanTideModel {
            n_max,
            m_max,
            constituents,
        }
    }

    /// Truncation degree of the model.
    ///
    /// # Returns
    ///
    /// * `usize` - The `n_max` the model was constructed with.
    pub fn n_max(&self) -> usize {
        self.n_max
    }

    /// Truncation order of the model.
    ///
    /// # Returns
    ///
    /// * `usize` - The `m_max` the model was constructed with.
    pub fn m_max(&self) -> usize {
        self.m_max
    }

    /// Number of tidal constituents (main plus, if enabled, admittance) held
    /// by the model.
    ///
    /// # Returns
    ///
    /// * `usize` - Constituent count.
    #[allow(dead_code)] // test-only diagnostics: asserted by unit tests, not used in the library build
    pub(crate) fn num_constituents(&self) -> usize {
        self.constituents.len()
    }

    /// Accumulate ΔC̄nm/ΔS̄nm at the given fundamental arguments into `deltas`
    /// per IERS TN36 Eq. (6.15), expanded to real arithmetic:
    ///
    ///   ΔC̄nm += (C⁺+C⁻)·cos θf + (S⁺+S⁻)·sin θf
    ///   ΔS̄nm += (S⁺−S⁻)·cos θf − (C⁺−C⁻)·sin θf
    ///
    /// (identical to Orekit `OceanTidesWave.addContribution`, cross-checked).
    ///
    /// # Arguments
    ///
    /// * `args` - `[γ = GMST+π, l, l′, F, D, Ω]` [rad] from
    ///   [`crate::orbit_dynamics::tides::doodson_delaunay_args`].
    /// * `deltas` - Accumulator; must satisfy `deltas.n_max() >= self.n_max`
    ///   and `deltas.m_max() >= self.m_max`.
    ///
    /// # Returns
    ///
    /// (none)
    pub(crate) fn accumulate_deltas(&self, args: &[f64; 6], deltas: &mut TideDeltas) {
        for wave in &self.constituents {
            let theta = tide_argument(&wave.delaunay, args);
            let (sin_t, cos_t) = theta.sin_cos();
            for l in &wave.lines {
                let dc = (l.c_plus + l.c_minus) * cos_t + (l.s_plus + l.s_minus) * sin_t;
                let ds = (l.s_plus - l.s_minus) * cos_t - (l.c_plus - l.c_minus) * sin_t;
                deltas.add(l.n as usize, l.m as usize, dc, ds);
            }
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::time::{Epoch, TimeSystem};

    #[test]
    #[serial_test::parallel]
    fn test_doodson_to_delaunay_matches_iers_tables() {
        // M2 (255.555): Table 6.5c row τ s h p N' ps = 2 0 0 0 0 0,
        // Delaunay l l' F D Ω = 0 0 2 0 2 (subtracted) => additive multipliers:
        assert_eq!(
            doodson_to_delaunay(parse_doodson("255.555").unwrap()),
            [2, 0, 0, -2, 0, -2]
        );
        // N2 (245.655): Table 6.5c row 2 -1 0 1 0 0 / Delaunay 1 0 2 0 2:
        assert_eq!(
            doodson_to_delaunay(parse_doodson("245.655").unwrap()),
            [2, -1, 0, -2, 0, -2]
        );
        // K1 (165.555): pure γ.
        assert_eq!(
            doodson_to_delaunay(parse_doodson("165.555").unwrap()),
            [1, 0, 0, 0, 0, 0]
        );
        // O1 (145.555): γ - 2s => F, Ω pick up the s expansion.
        assert_eq!(
            doodson_to_delaunay(parse_doodson("145.555").unwrap()),
            [1, 0, 0, -2, 0, -2]
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_theta_dot_matches_table_65c() {
        // Table 6.5c "deg/hr" column: M2 = 28.98410, N2 = 28.43973.
        let m2 = theta_dot_deg_per_hour(parse_doodson("255.555").unwrap());
        assert!((m2 - 28.98410).abs() < 1e-4, "M2 rate {m2}");
        let n2 = theta_dot_deg_per_hour(parse_doodson("245.655").unwrap());
        assert!((n2 - 28.43973).abs() < 1e-4, "N2 rate {n2}");
        // K1 = τ̇ + ṡ = rate of GMST+π ≈ 15.041069 deg/hr.
        let k1 = theta_dot_deg_per_hour(parse_doodson("165.555").unwrap());
        assert!((k1 - 15.04107).abs() < 1e-4, "K1 rate {k1}");
    }

    #[test]
    #[serial_test::serial]
    fn test_tide_argument_m2_period() {
        // Finite-difference the M2 argument over one hour: expect ~28.984 deg/hr.
        crate::utils::testing::setup_global_test_eop();
        let epoch = Epoch::from_datetime(2020, 3, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let delaunay = doodson_to_delaunay(parse_doodson("255.555").unwrap());
        let a0 = crate::orbit_dynamics::tides::doodson_delaunay_args(epoch);
        let a1 = crate::orbit_dynamics::tides::doodson_delaunay_args(epoch + 3600.0);
        let rate_deg_hr =
            (tide_argument(&delaunay, &a1) - tide_argument(&delaunay, &a0)).to_degrees();
        assert!(
            (rate_deg_hr - 28.98410).abs() < 1e-3,
            "M2 rate {rate_deg_hr}"
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_parse_doodson() {
        assert_eq!(parse_doodson("255.555").unwrap(), [2, 0, 0, 0, 0, 0]);
        assert_eq!(parse_doodson("55.565").unwrap(), [0, 0, 0, 0, 1, 0]);
        assert_eq!(parse_doodson("56.554").unwrap(), [0, 0, 1, 0, 0, -1]);
        assert_eq!(parse_doodson("135.655").unwrap(), [1, -2, 0, 1, 0, 0]);
        assert_eq!(parse_doodson("455.555").unwrap(), [4, 0, 0, 0, 0, 0]);
        assert!(parse_doodson("bogus").is_err());
    }

    #[test]
    #[serial_test::parallel]
    fn test_parse_fes2004_fixture() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test_data/fes2004_Cnm-Snm_n30.dat");
        let waves = parse_fes2004_file(&path, 30, 30).unwrap();
        // 18 constituents in the file (TN36 §6.3.2 prose lists T2, but the actual
        // coefficient file does not include it; T2 arrives via admittance).
        assert_eq!(waves.len(), 18);
        let total_lines: usize = waves.iter().map(|w| w.lines.len()).sum();
        assert_eq!(total_lines, 7890); // n in 2..=30, m <= 30 (degree-1 rows skipped)

        // Anchor: M2 (2,0) row is "-0.00298 -15.54137 0.00000 0.00000" x 1e-11.
        let m2 = waves
            .iter()
            .find(|w| w.doodson == [2, 0, 0, 0, 0, 0])
            .unwrap();
        let l = m2.lines.iter().find(|l| l.n == 2 && l.m == 0).unwrap();
        assert!((l.c_plus - (-0.00298e-11)).abs() < 1e-19);
        assert!((l.s_plus - (-15.54137e-11)).abs() < 1e-19);
        assert_eq!(l.c_minus, 0.0);

        // Anchor: Om1 (55.565) degree-2 zonal, C+ = -6.58128e-11.
        let om1 = waves
            .iter()
            .find(|w| w.doodson == [0, 0, 0, 0, 1, 0])
            .unwrap();
        let l = om1.lines.iter().find(|l| l.n == 2 && l.m == 0).unwrap();
        assert!((l.c_plus - (-6.58128e-11)).abs() < 1e-19);

        // Truncation: parsing at (4, 2) keeps n <= 4 and m <= 2 only.
        let waves42 = parse_fes2004_file(&path, 4, 2).unwrap();
        assert!(
            waves42
                .iter()
                .flat_map(|w| &w.lines)
                .all(|l| l.n <= 4 && l.m <= 2)
        );
        assert!(waves42.iter().flat_map(|w| &w.lines).all(|l| l.n >= 2));
    }

    #[test]
    #[serial_test::serial]
    fn test_fes2004_path_uses_cache_without_network() {
        let _guard = crate::utils::testing::setup_test_fes2004_cache();
        let path = fes2004_coefficients_path().unwrap();
        assert!(path.exists());
        assert!(path.ends_with("tides/fes2004_Cnm-Snm.dat"));
    }

    #[test]
    #[serial_test::serial]
    #[cfg_attr(not(feature = "integration"), ignore)]
    fn test_fes2004_download() {
        // Network-gated: clean cache, real download, file is complete.
        let dir = tempfile::tempdir().unwrap();
        let prev = std::env::var("BRAHE_CACHE").ok();
        unsafe {
            std::env::set_var("BRAHE_CACHE", dir.path());
        }
        let path = fes2004_coefficients_path().unwrap();
        let len = std::fs::metadata(&path).unwrap().len();
        assert!(len > 3_500_000, "downloaded file too small: {len}");
        unsafe {
            match prev {
                Some(v) => std::env::set_var("BRAHE_CACHE", v),
                None => std::env::remove_var("BRAHE_CACHE"),
            }
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_new_rejects_bad_bounds_without_deleting_cache() {
        // Out-of-range truncation bounds are a caller-input error, not cache
        // corruption: `new` must reject them without ever touching a
        // valid, already-cached file.
        let _guard = crate::utils::testing::setup_test_fes2004_cache();
        let cached = std::path::PathBuf::from(crate::utils::cache::get_tides_cache_dir().unwrap())
            .join("fes2004_Cnm-Snm.dat");
        assert!(cached.exists(), "test fixture must be seeded");

        let result = OceanTideModel::new(20, 1, true);
        let msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("m_max=1 must be rejected"),
        };
        assert!(
            msg.contains("truncation"),
            "error should name the bounds violation, got: {msg}"
        );
        assert!(
            !msg.contains("corrupt"),
            "bounds error must not be misreported as cache corruption, got: {msg}"
        );
        assert!(
            cached.exists(),
            "valid cache file must survive a bounds-validation error"
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_parse_fes2004_rejects_order_gt_degree() {
        // A data row with m > n would alias another packed (n, m) entry and must
        // be rejected outright rather than silently corrupting the model.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.dat");
        std::fs::write(&path, "255.555 M2 2 3 1.0 2.0 3.0 4.0\n").unwrap();
        let result = parse_fes2004_file(&path, 30, 30);
        assert!(result.is_err(), "row with m > n must error");
    }

    #[test]
    #[serial_test::parallel]
    fn test_from_file_rejects_out_of_range_bounds() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test_data/fes2004_Cnm-Snm_n30.dat");
        // m_max < 2, n_max > 100, and m_max > n_max are all rejected.
        assert!(OceanTideModel::from_file(&path, 1, 1, false).is_err());
        assert!(OceanTideModel::from_file(&path, 101, 20, false).is_err());
        assert!(OceanTideModel::from_file(&path, 10, 20, false).is_err());
    }

    #[test]
    #[serial_test::serial]
    fn test_new_rejects_and_removes_truncated_cache() {
        // A pre-existing cache file missing main constituents must be rejected
        // and deleted so the next call re-downloads.
        let dir = tempfile::tempdir().unwrap();
        let tides = dir.path().join("tides");
        std::fs::create_dir_all(&tides).unwrap();
        let cached = tides.join("fes2004_Cnm-Snm.dat");
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test_data/fes2004_Cnm-Snm_n30.dat");
        let content = std::fs::read_to_string(&fixture).unwrap();
        // Keep only the first few hundred lines: far fewer than the 18 required
        // main constituents, so `validate_main_waves` fails.
        let truncated: String = content.lines().take(400).collect::<Vec<_>>().join("\n");
        std::fs::write(&cached, &truncated).unwrap();

        let prev = std::env::var("BRAHE_CACHE").ok();
        unsafe {
            std::env::set_var("BRAHE_CACHE", dir.path());
        }
        let result = OceanTideModel::new(20, 20, false);
        assert!(result.is_err(), "truncated cache must error");
        assert!(!cached.exists(), "corrupt cache file must be removed");
        unsafe {
            match prev {
                Some(v) => std::env::set_var("BRAHE_CACHE", v),
                None => std::env::remove_var("BRAHE_CACHE"),
            }
        }
    }

    #[test]
    #[serial_test::serial]
    fn test_new_rejects_degree_truncated_cache() {
        // The n30 fixture holds all 18 main waves (so `validate_main_waves`
        // alone would pass it) but only up to degree 30. Requesting a higher
        // degree must be rejected as a truncated cache, not silently return a
        // model with high-degree corrections zeroed out.
        let _guard = crate::utils::testing::setup_test_fes2004_cache();
        let cached = std::path::PathBuf::from(crate::utils::cache::get_tides_cache_dir().unwrap())
            .join("fes2004_Cnm-Snm.dat");
        assert!(cached.exists(), "test fixture must be seeded");

        let result = OceanTideModel::new(50, 50, true);
        let msg = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("degree-truncated cache must be rejected"),
        };
        assert!(
            msg.contains("degree"),
            "error should mention degree coverage, got: {msg}"
        );
        assert!(
            !cached.exists(),
            "degree-truncated cache file must be removed"
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_admittance_table_integrity() {
        use crate::orbit_dynamics::ocean_tides_admittance::ADMITTANCE_TABLE;
        // 18 main rows (in the FES2004 file), S1 excluded (no pivots, not in file),
        // 63 secondary rows with pivots. M4 is main with no Hf dependence.
        let with_pivots = ADMITTANCE_TABLE
            .iter()
            .filter(|r| r.pivots.is_some())
            .count();
        assert_eq!(with_pivots, 63);
        // Every pivot must be a main wave present in the FES2004 file.
        let file_waves: std::collections::HashSet<[i8; 6]> = FES2004_MAIN_WAVE_DOODSON
            .iter()
            .map(|s| parse_doodson(s).unwrap())
            .collect();
        for row in ADMITTANCE_TABLE.iter() {
            if let Some((p1, p2)) = row.pivots {
                assert!(
                    file_waves.contains(&parse_doodson(p1).unwrap()),
                    "pivot {p1} of {} is not a FES2004 main wave",
                    row.doodson
                );
                assert!(
                    file_waves.contains(&parse_doodson(p2).unwrap()),
                    "pivot {p2} of {} is not a FES2004 main wave",
                    row.doodson
                );
            }
        }
        // Anchor amplitudes (Table 6.7): K1 .36878, O1 -.26221, M2 .63192.
        let hf = |d: &str| ADMITTANCE_TABLE.iter().find(|r| r.doodson == d).unwrap().hf;
        assert_eq!(hf("165.555"), 0.36878);
        assert_eq!(hf("145.555"), -0.26221);
        assert_eq!(hf("255.555"), 0.63192);
    }

    #[test]
    #[serial_test::serial]
    fn test_expand_admittance_155555_weights() {
        // 155.555 (Hf = -0.00399) pivots O1 (145.555, H1 = -0.26221) and
        // K1 (165.555, H2 = 0.36878). Its frequency is exactly midway:
        // (θ̇f-θ̇1)/(θ̇2-θ̇1) = ṡ/(2ṡ) = 0.5, so per Eq. (6.16)
        //   C± = 0.5·(Hf/H2)·C±_K1 + 0.5·(Hf/H1)·C±_O1.
        let _guard = crate::utils::testing::setup_test_fes2004_cache();
        let path = fes2004_coefficients_path().unwrap();
        let main = parse_fes2004_file(&path, 30, 30).unwrap();
        let secondary = expand_admittance(&main).unwrap();
        let w = secondary
            .iter()
            .find(|w| w.doodson == parse_doodson("155.555").unwrap())
            .unwrap();
        let k1 = main
            .iter()
            .find(|w| w.doodson == parse_doodson("165.555").unwrap())
            .unwrap();
        let o1 = main
            .iter()
            .find(|w| w.doodson == parse_doodson("145.555").unwrap())
            .unwrap();
        let (hf, h1, h2) = (-0.00399, -0.26221, 0.36878);
        let pick = |c: &OceanTideConstituent, n, m| {
            *c.lines.iter().find(|l| l.n == n && l.m == m).unwrap()
        };
        let (lf, l1, l2) = (pick(w, 2, 1), pick(o1, 2, 1), pick(k1, 2, 1));
        let expected = 0.5 * (hf / h2) * l2.c_plus + 0.5 * (hf / h1) * l1.c_plus;
        assert!(
            (lf.c_plus - expected).abs() < 1e-18,
            "got {} want {expected}",
            lf.c_plus
        );

        // Expanded set covers all pivot rows: 63 secondary constituents.
        assert_eq!(secondary.len(), 63);
    }

    #[test]
    #[serial_test::parallel]
    fn test_ocean_accumulate_single_constituent() {
        // Synthetic wave with known argument: multipliers all zero except γ, so
        // θ = γ. Check the Eq. (6.15) real expansion at two argument values.
        let wave = OceanTideConstituent {
            doodson: [1, 0, 0, 0, 0, 0],
            name: "TEST".to_string(),
            delaunay: [1, 0, 0, 0, 0, 0],
            lines: vec![OceanTideLine {
                n: 2,
                m: 1,
                c_plus: 3.0e-11,
                s_plus: 5.0e-11,
                c_minus: 7.0e-11,
                s_minus: 11.0e-11,
            }],
        };
        let model = OceanTideModel::from_constituents(vec![wave], 2, 1);

        // θ = 0: ΔC = C+ + C- = 10e-11; ΔS = S+ - S- = -6e-11.
        let mut d = TideDeltas::new(2, 1);
        model.accumulate_deltas(&[0.0; 6], &mut d);
        let (dc, ds) = d.get(2, 1);
        assert!((dc - 10.0e-11).abs() < 1e-24);
        assert!((ds - (-6.0e-11)).abs() < 1e-24);

        // θ = π/2: ΔC = S+ + S- = 16e-11; ΔS = -(C+ - C-) = 4e-11.
        let mut d = TideDeltas::new(2, 1);
        model.accumulate_deltas(
            &[std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, 0.0, 0.0],
            &mut d,
        );
        let (dc, ds) = d.get(2, 1);
        assert!((dc - 16.0e-11).abs() < 1e-24);
        assert!((ds - 4.0e-11).abs() < 1e-24);
    }

    #[test]
    #[serial_test::serial]
    fn test_ocean_model_new_magnitude() {
        crate::utils::testing::setup_global_test_eop();
        let _guard = crate::utils::testing::setup_test_fes2004_cache();
        let model = OceanTideModel::new(20, 20, true).unwrap();
        assert_eq!(model.n_max(), 20);
        assert_eq!(model.num_constituents(), 81); // 18 main + 63 admittance

        let epoch = Epoch::from_datetime(2020, 3, 1, 6, 0, 0.0, 0.0, TimeSystem::UTC);
        let args = crate::orbit_dynamics::tides::doodson_delaunay_args(epoch);
        let mut d = TideDeltas::new(20, 20);
        model.accumulate_deltas(&args, &mut d);
        // Dominant ocean tide corrections are ~1e-10..1e-8 on low-degree terms
        // (M2 alone carries ~1.5e-10 on C22/S22 scale amplitudes).
        let (dc22, ds22) = d.get(2, 2);
        let mag = (dc22 * dc22 + ds22 * ds22).sqrt();
        assert!(mag > 1e-11 && mag < 1e-8, "|ΔC̄22,ΔS̄22| = {mag:e}");
    }

    #[test]
    #[serial_test::serial]
    fn test_ocean_model_admittance_toggle() {
        let _guard = crate::utils::testing::setup_test_fes2004_cache();
        let with = OceanTideModel::new(20, 20, true).unwrap();
        let without = OceanTideModel::new(20, 20, false).unwrap();
        assert_eq!(with.num_constituents(), 81);
        assert_eq!(without.num_constituents(), 18);
    }
}
