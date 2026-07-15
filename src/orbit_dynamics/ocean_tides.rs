/*!
FES2004 ocean tide geopotential corrections (IERS Conventions 2010, TN36 §6.3).

Data source: FES2004 normalized Stokes-coefficient amplitude file, downloaded
once into `$BRAHE_CACHE/tides/` from
<https://iers-conventions.obspm.fr/content/chapter6/additional_info/tidemodels/fes2004_Cnm-Snm.dat>.
*/

use std::io::Read;
use std::path::{Path, PathBuf};

use crate::utils::BraheError;
use crate::utils::cache::get_tides_cache_dir;
use crate::utils::fs::atomic_write;

const FES2004_URL: &str = "https://iers-conventions.obspm.fr/content/chapter6/additional_info/tidemodels/fes2004_Cnm-Snm.dat";
const FES2004_FILENAME: &str = "fes2004_Cnm-Snm.dat";

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

    let response = ureq::get(FES2004_URL).call().map_err(|e| {
        BraheError::Error(format!(
            "FES2004 ocean tide coefficients are not cached and the download from {FES2004_URL} \
             failed: {e}. Download the file manually to {} to proceed offline.",
            path.display()
        ))
    })?;
    let mut buf = Vec::new();
    response
        .into_body()
        .into_reader()
        .read_to_end(&mut buf)
        .map_err(|e| BraheError::Error(format!("FES2004 download read failed: {e}")))?;
    if buf.is_empty() {
        return Err(BraheError::Error(format!(
            "Empty response downloading FES2004 ocean tide coefficients from {FES2004_URL}"
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
#[allow(dead_code)] // consumed by Task 6+: ocean tide acceleration model
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
#[allow(dead_code)] // consumed by Task 6+: ocean tide acceleration model
pub(crate) struct OceanTideConstituent {
    pub doodson: [i8; 6],
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
#[allow(dead_code)] // consumed by Task 6+: ocean tide acceleration model
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
#[allow(dead_code)] // consumed by Task 6+: ocean tide acceleration model
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
                delaunay: [0; 6],
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

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
        unsafe {
            std::env::set_var("BRAHE_CACHE", dir.path());
        }
        let path = fes2004_coefficients_path().unwrap();
        let len = std::fs::metadata(&path).unwrap().len();
        assert!(len > 3_500_000, "downloaded file too small: {len}");
        unsafe {
            std::env::remove_var("BRAHE_CACHE");
        }
    }
}
