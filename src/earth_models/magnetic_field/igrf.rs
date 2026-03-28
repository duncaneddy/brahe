use nalgebra::Vector3;
use once_cell::sync::Lazy;

use crate::AngleFormat;
use crate::time::Epoch;
use crate::utils::errors::BraheError;

use super::spherical_harmonics::{
    RE_MAGNETIC, nm_to_idx, num_coefficients, synth_field_geocentric,
};
use super::transforms::{
    epoch_to_decimal_year, field_geocentric_to_geocentric_enz, field_geocentric_to_geodetic_enz,
    field_geodetic_enz_to_ecef, geodetic_to_geocentric_mag,
};

/// Internal result type: (b_r, b_theta, b_phi, theta_rad, lat_geod_rad, lon_deg, lat_deg)
type MagFieldCoreResult = (f64, f64, f64, f64, f64, f64, f64);

/// Parsed IGRF-14 model coefficients.
///
/// Stores Gauss coefficients at 27 five-year epochs from 1900.0 to 2030.0,
/// for spherical harmonic degrees 1-13.
pub(crate) struct IGRFCoefficients {
    /// Epoch times as decimal years
    pub epochs: Vec<f64>,
    /// Gauss g coefficients: `g[epoch_idx][coeff_idx]`
    pub g: Vec<Vec<f64>>,
    /// Gauss h coefficients: `h[epoch_idx][coeff_idx]`
    pub h: Vec<Vec<f64>>,
    /// Maximum spherical harmonic degree
    pub n_max: usize,
}

/// Lazily parsed IGRF-14 coefficients, loaded once from the embedded SHC file.
static IGRF_COEFFICIENTS: Lazy<IGRFCoefficients> =
    Lazy::new(|| parse_shc(super::data::IGRF14_SHC).expect("Failed to parse embedded IGRF14.shc"));

/// Parse an SHC (Spherical Harmonic Coefficient) file.
///
/// File format:
/// - Lines starting with `#` are comments
/// - First non-comment line: `N_MIN N_MAX NTIMES SP_ORDER N_STEPS [START END]`
/// - Second non-comment line: space-separated epoch years
/// - Remaining lines: `n m coeff_1 coeff_2 ... coeff_NTIMES`
///   - Positive m → g coefficient for (n, m)
///   - Negative m → h coefficient for (n, |m|)
fn parse_shc(content: &str) -> Result<IGRFCoefficients, BraheError> {
    let mut header_lines_remaining = 2;
    let mut n_max: usize = 0;
    let mut epochs = Vec::new();
    let mut g_by_epoch: Vec<Vec<f64>> = Vec::new();
    let mut h_by_epoch: Vec<Vec<f64>> = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        if header_lines_remaining == 2 {
            // Parameter line: N_MIN N_MAX NTIMES SP_ORDER N_STEPS [START END]
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() < 5 {
                return Err(BraheError::ParseError(
                    "SHC header line has fewer than 5 fields".to_string(),
                ));
            }
            n_max = parts[1]
                .parse::<usize>()
                .map_err(|e| BraheError::ParseError(format!("Failed to parse N_MAX: {e}")))?;
            let n_times = parts[2]
                .parse::<usize>()
                .map_err(|e| BraheError::ParseError(format!("Failed to parse NTIMES: {e}")))?;

            let n_c = num_coefficients(n_max);
            g_by_epoch = vec![vec![0.0; n_c]; n_times];
            h_by_epoch = vec![vec![0.0; n_c]; n_times];

            header_lines_remaining -= 1;
            continue;
        }

        if header_lines_remaining == 1 {
            // Epoch years line
            epochs = trimmed
                .split_whitespace()
                .map(|s| {
                    s.parse::<f64>()
                        .map_err(|e| BraheError::ParseError(format!("Failed to parse epoch: {e}")))
                })
                .collect::<Result<Vec<f64>, _>>()?;

            header_lines_remaining -= 1;
            continue;
        }

        // Coefficient line: n m coeff_1 coeff_2 ... coeff_NTIMES
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 3 {
            continue;
        }

        let n = parts[0]
            .parse::<i32>()
            .map_err(|e| BraheError::ParseError(format!("Failed to parse n: {e}")))?;
        let m = parts[1]
            .parse::<i32>()
            .map_err(|e| BraheError::ParseError(format!("Failed to parse m: {e}")))?;

        let coeffs: Vec<f64> = parts[2..]
            .iter()
            .map(|s| {
                s.parse::<f64>()
                    .map_err(|e| BraheError::ParseError(format!("Failed to parse coeff: {e}")))
            })
            .collect::<Result<Vec<f64>, _>>()?;

        let n_abs = n as usize;
        let m_abs = m.unsigned_abs() as usize;
        let idx = nm_to_idx(n_abs, m_abs);

        if m >= 0 {
            // g coefficient
            for (epoch_idx, &val) in coeffs.iter().enumerate() {
                if epoch_idx < g_by_epoch.len() {
                    g_by_epoch[epoch_idx][idx] = val;
                }
            }
        } else {
            // h coefficient (negative m in file)
            for (epoch_idx, &val) in coeffs.iter().enumerate() {
                if epoch_idx < h_by_epoch.len() {
                    h_by_epoch[epoch_idx][idx] = val;
                }
            }
        }
    }

    Ok(IGRFCoefficients {
        epochs,
        g: g_by_epoch,
        h: h_by_epoch,
        n_max,
    })
}

/// Interpolate IGRF coefficients to a target decimal year.
///
/// Uses linear interpolation between the two bracketing 5-year epochs.
///
/// # Returns
///
/// `(g_interp, h_interp)` - Interpolated coefficient vectors
fn interpolate_igrf(coeffs: &IGRFCoefficients, decimal_year: f64) -> (Vec<f64>, Vec<f64>) {
    let n = coeffs.epochs.len();
    let n_coeffs = coeffs.g[0].len();

    // Find bracketing epochs
    let mut i = 0;
    while i < n - 1 && coeffs.epochs[i + 1] < decimal_year {
        i += 1;
    }
    // Clamp to valid range
    if i >= n - 1 {
        i = n - 2;
    }

    let t0 = coeffs.epochs[i];
    let t1 = coeffs.epochs[i + 1];
    let dt = t1 - t0;

    let frac = if dt.abs() > 1e-10 {
        (decimal_year - t0) / dt
    } else {
        0.0
    };

    let mut g = vec![0.0; n_coeffs];
    let mut h = vec![0.0; n_coeffs];

    for j in 0..n_coeffs {
        g[j] = coeffs.g[i][j] + frac * (coeffs.g[i + 1][j] - coeffs.g[i][j]);
        h[j] = coeffs.h[i][j] + frac * (coeffs.h[i + 1][j] - coeffs.h[i][j]);
    }

    (g, h)
}

/// Compute IGRF-14 magnetic field in the geodetic ENZ (East-North-Zenith) frame.
///
/// The geodetic ENZ frame has zenith perpendicular to the WGS84 ellipsoid surface.
///
/// # Arguments
///
/// * `epoch` - Time of evaluation
/// * `x_geod` - Geodetic position as (longitude, latitude, altitude) where altitude is in meters.
///   Longitude and latitude units are controlled by `angle_format`.
/// * `angle_format` - Whether longitude/latitude are in degrees or radians
///
/// # Returns
///
/// `Vector3<f64>` containing (B_east, B_north, B_zenith) in nanoTesla
///
/// # Errors
///
/// Returns `BraheError::OutOfBoundsError` if the epoch is outside the model range [1900, 2030].
///
/// # Examples
///
/// ```
/// use brahe::earth_models::magnetic_field::igrf_geodetic_enz;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::AngleFormat;
/// use nalgebra::Vector3;
///
/// let epoch = Epoch::from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_geod = Vector3::new(0.0, 80.0, 0.0); // lon=0, lat=80 deg, alt=0 m
/// let b_enz = igrf_geodetic_enz(&epoch, x_geod, AngleFormat::Degrees).unwrap();
/// // b_enz contains (B_east, B_north, B_zenith) in nT
/// ```
pub fn igrf_geodetic_enz(
    epoch: &Epoch,
    x_geod: Vector3<f64>,
    angle_format: AngleFormat,
) -> Result<Vector3<f64>, BraheError> {
    let (b_r, b_theta, _b_phi, theta, lat_geod_rad, _lon_deg, _lat_deg) =
        igrf_core(epoch, x_geod, angle_format)?;

    Ok(field_geocentric_to_geodetic_enz(
        b_r,
        b_theta,
        _b_phi,
        theta,
        lat_geod_rad,
    ))
}

/// Compute IGRF-14 magnetic field in the geocentric ENZ (East-North-Zenith) frame.
///
/// The geocentric ENZ frame has zenith along the geocentric radial direction.
///
/// # Arguments
///
/// * `epoch` - Time of evaluation
/// * `x_geod` - Geodetic position as (longitude, latitude, altitude) where altitude is in meters.
///   Longitude and latitude units are controlled by `angle_format`.
/// * `angle_format` - Whether longitude/latitude are in degrees or radians
///
/// # Returns
///
/// `Vector3<f64>` containing (B_east, B_north, B_zenith) in nanoTesla
///
/// # Errors
///
/// Returns `BraheError::OutOfBoundsError` if the epoch is outside the model range [1900, 2030].
pub fn igrf_geocentric_enz(
    epoch: &Epoch,
    x_geod: Vector3<f64>,
    angle_format: AngleFormat,
) -> Result<Vector3<f64>, BraheError> {
    let (b_r, b_theta, b_phi, _theta, _lat_geod_rad, _lon_deg, _lat_deg) =
        igrf_core(epoch, x_geod, angle_format)?;

    Ok(field_geocentric_to_geocentric_enz(b_r, b_theta, b_phi))
}

/// Compute IGRF-14 magnetic field in the ECEF frame.
///
/// # Arguments
///
/// * `epoch` - Time of evaluation
/// * `x_geod` - Geodetic position as (longitude, latitude, altitude) where altitude is in meters.
///   Longitude and latitude units are controlled by `angle_format`.
/// * `angle_format` - Whether longitude/latitude are in degrees or radians
///
/// # Returns
///
/// `Vector3<f64>` containing (B_x, B_y, B_z) in ECEF frame, in nanoTesla
///
/// # Errors
///
/// Returns `BraheError::OutOfBoundsError` if the epoch is outside the model range [1900, 2030].
pub fn igrf_ecef(
    epoch: &Epoch,
    x_geod: Vector3<f64>,
    angle_format: AngleFormat,
) -> Result<Vector3<f64>, BraheError> {
    let (b_r, b_theta, b_phi, theta, lat_geod_rad, lon_deg, lat_deg) =
        igrf_core(epoch, x_geod, angle_format)?;

    let b_enz = field_geocentric_to_geodetic_enz(b_r, b_theta, b_phi, theta, lat_geod_rad);
    Ok(field_geodetic_enz_to_ecef(b_enz, lon_deg, lat_deg))
}

/// Core IGRF computation returning raw geocentric spherical components and geometry.
///
/// This is the shared implementation used by all public IGRF functions.
///
/// # Returns
///
/// `(b_r, b_theta, b_phi, theta_rad, lat_geod_rad, lon_deg, lat_deg)`
fn igrf_core(
    epoch: &Epoch,
    x_geod: Vector3<f64>,
    angle_format: AngleFormat,
) -> Result<MagFieldCoreResult, BraheError> {
    let coeffs = &*IGRF_COEFFICIENTS;

    // Convert angles to degrees
    let (lon_deg, lat_deg) = match angle_format {
        AngleFormat::Degrees => (x_geod[0], x_geod[1]),
        AngleFormat::Radians => (x_geod[0].to_degrees(), x_geod[1].to_degrees()),
    };
    let alt_m = x_geod[2];
    let alt_km = alt_m / 1000.0;

    // Convert epoch to decimal year and validate
    let decimal_year = epoch_to_decimal_year(epoch);
    let first_epoch = coeffs.epochs[0];
    let last_epoch = *coeffs.epochs.last().unwrap();

    if decimal_year < first_epoch || decimal_year > last_epoch {
        return Err(BraheError::OutOfBoundsError(format!(
            "IGRF epoch {:.2} is outside valid range [{:.1}, {:.1}]",
            decimal_year, first_epoch, last_epoch
        )));
    }

    // Interpolate coefficients
    let (g, h) = interpolate_igrf(coeffs, decimal_year);

    // Convert geodetic to geocentric
    let (r, theta, phi) = geodetic_to_geocentric_mag(lat_deg, lon_deg, alt_km);

    // Compute field in geocentric spherical coordinates
    let (b_r, b_theta, b_phi) =
        synth_field_geocentric(r, theta, phi, &g, &h, coeffs.n_max, RE_MAGNETIC);

    let lat_geod_rad = lat_deg.to_radians();

    Ok((b_r, b_theta, b_phi, theta, lat_geod_rad, lon_deg, lat_deg))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::TimeSystem;
    use crate::utils::testing::setup_global_test_eop;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_parse_shc_basic() {
        let coeffs = &*IGRF_COEFFICIENTS;
        assert_eq!(coeffs.n_max, 13);
        assert_eq!(coeffs.epochs.len(), 27);
        assert_abs_diff_eq!(coeffs.epochs[0], 1900.0, epsilon = 1e-6);
        assert_abs_diff_eq!(*coeffs.epochs.last().unwrap(), 2030.0, epsilon = 1e-6);

        // Spot check: g(1,0) at epoch 2025.0 should be -29350.0 (approximately)
        // From file line: 1   0 ... -29350.0 -29287.0
        let idx_10 = nm_to_idx(1, 0);
        // 2025.0 is epoch index 25 (0-based: 1900, 1905, ..., 2025 is index 25)
        assert_abs_diff_eq!(coeffs.g[25][idx_10], -29350.0, epsilon = 1.0);

        // g(1,1) at 2025.0 should be ~ -1410.3
        let idx_11 = nm_to_idx(1, 1);
        assert_abs_diff_eq!(coeffs.g[25][idx_11], -1410.3, epsilon = 1.0);

        // h(1,1) at 2025.0 should be ~ 4545.5
        assert_abs_diff_eq!(coeffs.h[25][idx_11], 4545.5, epsilon = 1.0);

        // h(n, 0) should always be 0
        for epoch_idx in 0..coeffs.epochs.len() {
            assert_abs_diff_eq!(coeffs.h[epoch_idx][idx_10], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_igrf_geodetic_enz_known_values() {
        setup_global_test_eop();

        // Test against ppigrf reference: IGRF at (lon=0, lat=80, h=0 km) at 2025.0
        // Expected values from WMMHR test table (which should be close to IGRF at low degree):
        // X (north) ≈ 6517.4 nT, Y (east) ≈ 144.8 nT, Z (down) ≈ 54701.3 nT
        // In our ENZ: B_east ≈ Y, B_north ≈ X, B_zenith ≈ -Z
        let epoch = Epoch::from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_geod = Vector3::new(0.0, 80.0, 0.0);
        let b = igrf_geodetic_enz(&epoch, x_geod, AngleFormat::Degrees).unwrap();

        // IGRF has lower resolution than WMMHR, so tolerances are wider
        // But the field should be in the right ballpark (within ~100 nT for crustal differences)
        assert!(
            b[0].abs() < 500.0,
            "B_east should be small at this location, got {}",
            b[0]
        );
        assert!(b[1] > 5000.0, "B_north should be > 5000 nT, got {}", b[1]);
        assert!(
            b[2] < -50000.0,
            "B_zenith should be strongly negative, got {}",
            b[2]
        );
    }

    #[test]
    fn test_igrf_epoch_out_of_range() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(1899, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_geod = Vector3::new(0.0, 45.0, 0.0);
        let result = igrf_geodetic_enz(&epoch, x_geod, AngleFormat::Degrees);
        assert!(result.is_err());

        let epoch = Epoch::from_datetime(2031, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let result = igrf_geodetic_enz(&epoch, x_geod, AngleFormat::Degrees);
        assert!(result.is_err());
    }

    #[test]
    fn test_igrf_radians_input() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);

        // Same point in degrees and radians should give same result
        let x_deg = Vector3::new(10.0, 45.0, 100000.0);
        let x_rad = Vector3::new(10.0_f64.to_radians(), 45.0_f64.to_radians(), 100000.0);

        let b_deg = igrf_geodetic_enz(&epoch, x_deg, AngleFormat::Degrees).unwrap();
        let b_rad = igrf_geodetic_enz(&epoch, x_rad, AngleFormat::Radians).unwrap();

        assert_abs_diff_eq!(b_deg[0], b_rad[0], epsilon = 1e-6);
        assert_abs_diff_eq!(b_deg[1], b_rad[1], epsilon = 1e-6);
        assert_abs_diff_eq!(b_deg[2], b_rad[2], epsilon = 1e-6);
    }

    #[test]
    fn test_igrf_ecef_enz_consistency() {
        setup_global_test_eop();

        // ECEF output should have the same magnitude as ENZ output
        let epoch = Epoch::from_datetime(2020, 6, 15, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_geod = Vector3::new(15.0, 50.0, 500000.0); // 500 km altitude

        let b_enz = igrf_geodetic_enz(&epoch, x_geod, AngleFormat::Degrees).unwrap();
        let b_ecef = igrf_ecef(&epoch, x_geod, AngleFormat::Degrees).unwrap();

        let mag_enz = b_enz.norm();
        let mag_ecef = b_ecef.norm();
        assert_abs_diff_eq!(mag_enz, mag_ecef, epsilon = 1e-6);
    }
}
