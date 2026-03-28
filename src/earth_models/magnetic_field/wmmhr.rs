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

/// Maximum spherical harmonic degree available in the WMMHR model.
pub const WMMHR_MAX_DEGREE: usize = 133;

/// Parsed WMMHR-2025 model coefficients.
///
/// Stores Gauss coefficients at a single epoch with secular variation rates
/// for spherical harmonic degrees 1-133.
pub(crate) struct WMMHRCoefficients {
    /// Model epoch as decimal year
    pub epoch: f64,
    /// Gauss g coefficients at epoch
    pub g: Vec<f64>,
    /// Gauss h coefficients at epoch
    pub h: Vec<f64>,
    /// Secular variation dg/dt [nT/year]
    pub g_sv: Vec<f64>,
    /// Secular variation dh/dt [nT/year]
    pub h_sv: Vec<f64>,
    /// Maximum spherical harmonic degree
    pub n_max: usize,
}

/// Lazily parsed WMMHR-2025 coefficients, loaded once from the embedded COF file.
static WMMHR_COEFFICIENTS: Lazy<WMMHRCoefficients> =
    Lazy::new(|| parse_cof(super::data::WMMHR_COF).expect("Failed to parse embedded WMMHR.COF"));

/// Parse a COF (Coefficient) file for the WMMHR model.
///
/// File format:
/// - Line 1: `epoch_year model_name date_string`
/// - Data lines: `n m g_nm h_nm dg_nm dh_nm`
/// - Sentinel: lines starting with `999`
fn parse_cof(content: &str) -> Result<WMMHRCoefficients, BraheError> {
    let mut epoch: f64 = 0.0;
    let mut n_max: usize = 0;
    let mut first_line = true;

    // First pass: determine n_max and epoch
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if first_line {
            // Header: epoch year, model name, date
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            epoch = parts[0]
                .parse::<f64>()
                .map_err(|e| BraheError::ParseError(format!("Failed to parse COF epoch: {e}")))?;
            first_line = false;
            continue;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 6 {
            continue;
        }

        let n = parts[0].parse::<usize>().unwrap_or(999);
        if n >= 999 {
            break; // sentinel
        }
        if n > n_max {
            n_max = n;
        }
    }

    // Allocate arrays
    let n_coeffs = num_coefficients(n_max);
    let mut g = vec![0.0; n_coeffs];
    let mut h = vec![0.0; n_coeffs];
    let mut g_sv = vec![0.0; n_coeffs];
    let mut h_sv = vec![0.0; n_coeffs];

    // Second pass: read coefficients
    let mut skip_header = true;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if skip_header {
            skip_header = false;
            continue;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 6 {
            continue;
        }

        let n = parts[0].parse::<usize>().unwrap_or(999);
        if n >= 999 {
            break;
        }

        let m = parts[1]
            .parse::<usize>()
            .map_err(|e| BraheError::ParseError(format!("Failed to parse m: {e}")))?;
        let g_nm = parts[2]
            .parse::<f64>()
            .map_err(|e| BraheError::ParseError(format!("Failed to parse g: {e}")))?;
        let h_nm = parts[3]
            .parse::<f64>()
            .map_err(|e| BraheError::ParseError(format!("Failed to parse h: {e}")))?;
        let dg_nm = parts[4]
            .parse::<f64>()
            .map_err(|e| BraheError::ParseError(format!("Failed to parse dg: {e}")))?;
        let dh_nm = parts[5]
            .parse::<f64>()
            .map_err(|e| BraheError::ParseError(format!("Failed to parse dh: {e}")))?;

        let idx = nm_to_idx(n, m);
        g[idx] = g_nm;
        h[idx] = h_nm;
        g_sv[idx] = dg_nm;
        h_sv[idx] = dh_nm;
    }

    Ok(WMMHRCoefficients {
        epoch,
        g,
        h,
        g_sv,
        h_sv,
        n_max,
    })
}

/// Interpolate WMMHR coefficients to a target decimal year using secular variation.
///
/// Computes `g(t) = g_epoch + dg/dt * (t - epoch)` for each coefficient,
/// truncated to the requested maximum degree.
///
/// # Returns
///
/// `(g_interp, h_interp)` - Interpolated coefficient vectors (truncated to nmax)
fn interpolate_wmmhr(
    coeffs: &WMMHRCoefficients,
    decimal_year: f64,
    nmax: usize,
) -> (Vec<f64>, Vec<f64>) {
    let dt = decimal_year - coeffs.epoch;
    let n_coeffs = num_coefficients(nmax);

    let mut g = vec![0.0; n_coeffs];
    let mut h = vec![0.0; n_coeffs];

    for n in 1..=nmax {
        for m in 0..=n {
            let idx = nm_to_idx(n, m);
            g[idx] = coeffs.g[idx] + coeffs.g_sv[idx] * dt;
            h[idx] = coeffs.h[idx] + coeffs.h_sv[idx] * dt;
        }
    }

    (g, h)
}

/// Compute WMMHR-2025 magnetic field in the geodetic ENZ (East-North-Zenith) frame.
///
/// The geodetic ENZ frame has zenith perpendicular to the WGS84 ellipsoid surface.
///
/// # Arguments
///
/// * `epoch` - Time of evaluation
/// * `x_geod` - Geodetic position as (longitude, latitude, altitude) where altitude is in meters.
///   Longitude and latitude units are controlled by `angle_format`.
/// * `angle_format` - Whether longitude/latitude are in degrees or radians
/// * `nmax` - Maximum spherical harmonic degree (1..=133). `None` uses full resolution (133).
///
/// # Returns
///
/// `Vector3<f64>` containing (B_east, B_north, B_zenith) in nanoTesla
///
/// # Errors
///
/// Returns `BraheError::OutOfBoundsError` if the epoch is outside the model range
/// or if `nmax` is outside [1, 133].
///
/// # Examples
///
/// ```
/// use brahe::earth_models::magnetic_field::wmmhr_geodetic_enz;
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::AngleFormat;
/// use nalgebra::Vector3;
///
/// let epoch = Epoch::from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let x_geod = Vector3::new(0.0, 80.0, 0.0); // lon=0, lat=80 deg, alt=0 m
/// let b_enz = wmmhr_geodetic_enz(&epoch, x_geod, AngleFormat::Degrees, None).unwrap();
/// ```
pub fn wmmhr_geodetic_enz(
    epoch: &Epoch,
    x_geod: Vector3<f64>,
    angle_format: AngleFormat,
    nmax: Option<usize>,
) -> Result<Vector3<f64>, BraheError> {
    let (b_r, b_theta, b_phi, theta, lat_geod_rad, _lon_deg, _lat_deg) =
        wmmhr_core(epoch, x_geod, angle_format, nmax)?;

    Ok(field_geocentric_to_geodetic_enz(
        b_r,
        b_theta,
        b_phi,
        theta,
        lat_geod_rad,
    ))
}

/// Compute WMMHR-2025 magnetic field in the geocentric ENZ (East-North-Zenith) frame.
///
/// The geocentric ENZ frame has zenith along the geocentric radial direction.
///
/// # Arguments
///
/// * `epoch` - Time of evaluation
/// * `x_geod` - Geodetic position as (longitude, latitude, altitude) where altitude is in meters.
///   Longitude and latitude units are controlled by `angle_format`.
/// * `angle_format` - Whether longitude/latitude are in degrees or radians
/// * `nmax` - Maximum spherical harmonic degree (1..=133). `None` uses full resolution (133).
///
/// # Returns
///
/// `Vector3<f64>` containing (B_east, B_north, B_zenith) in nanoTesla
///
/// # Errors
///
/// Returns `BraheError::OutOfBoundsError` if the epoch is outside the model range
/// or if `nmax` is outside [1, 133].
pub fn wmmhr_geocentric_enz(
    epoch: &Epoch,
    x_geod: Vector3<f64>,
    angle_format: AngleFormat,
    nmax: Option<usize>,
) -> Result<Vector3<f64>, BraheError> {
    let (b_r, b_theta, b_phi, _theta, _lat_geod_rad, _lon_deg, _lat_deg) =
        wmmhr_core(epoch, x_geod, angle_format, nmax)?;

    Ok(field_geocentric_to_geocentric_enz(b_r, b_theta, b_phi))
}

/// Compute WMMHR-2025 magnetic field in the ECEF frame.
///
/// # Arguments
///
/// * `epoch` - Time of evaluation
/// * `x_geod` - Geodetic position as (longitude, latitude, altitude) where altitude is in meters.
///   Longitude and latitude units are controlled by `angle_format`.
/// * `angle_format` - Whether longitude/latitude are in degrees or radians
/// * `nmax` - Maximum spherical harmonic degree (1..=133). `None` uses full resolution (133).
///
/// # Returns
///
/// `Vector3<f64>` containing (B_x, B_y, B_z) in ECEF frame, in nanoTesla
///
/// # Errors
///
/// Returns `BraheError::OutOfBoundsError` if the epoch is outside the model range
/// or if `nmax` is outside [1, 133].
pub fn wmmhr_ecef(
    epoch: &Epoch,
    x_geod: Vector3<f64>,
    angle_format: AngleFormat,
    nmax: Option<usize>,
) -> Result<Vector3<f64>, BraheError> {
    let (b_r, b_theta, b_phi, theta, lat_geod_rad, lon_deg, lat_deg) =
        wmmhr_core(epoch, x_geod, angle_format, nmax)?;

    let b_enz = field_geocentric_to_geodetic_enz(b_r, b_theta, b_phi, theta, lat_geod_rad);
    Ok(field_geodetic_enz_to_ecef(b_enz, lon_deg, lat_deg))
}

/// Core WMMHR computation returning raw geocentric spherical components and geometry.
fn wmmhr_core(
    epoch: &Epoch,
    x_geod: Vector3<f64>,
    angle_format: AngleFormat,
    nmax: Option<usize>,
) -> Result<MagFieldCoreResult, BraheError> {
    let coeffs = &*WMMHR_COEFFICIENTS;
    let nmax = nmax.unwrap_or(coeffs.n_max);

    // Validate nmax
    if nmax < 1 || nmax > coeffs.n_max {
        return Err(BraheError::OutOfBoundsError(format!(
            "WMMHR nmax {} is outside valid range [1, {}]",
            nmax, coeffs.n_max
        )));
    }

    // Convert angles to degrees
    let (lon_deg, lat_deg) = match angle_format {
        AngleFormat::Degrees => (x_geod[0], x_geod[1]),
        AngleFormat::Radians => (x_geod[0].to_degrees(), x_geod[1].to_degrees()),
    };
    let alt_m = x_geod[2];
    let alt_km = alt_m / 1000.0;

    // Convert epoch to decimal year and validate
    let decimal_year = epoch_to_decimal_year(epoch);

    // WMMHR valid range: approximately model epoch - 0.13 to 2030.0
    let min_year = coeffs.epoch - 0.15; // small buffer before epoch
    let max_year = 2030.0;

    if decimal_year < min_year || decimal_year > max_year {
        return Err(BraheError::OutOfBoundsError(format!(
            "WMMHR epoch {:.2} is outside valid range [{:.2}, {:.1}]",
            decimal_year, min_year, max_year
        )));
    }

    // Interpolate coefficients using secular variation
    let (g, h) = interpolate_wmmhr(coeffs, decimal_year, nmax);

    // Convert geodetic to geocentric
    let (r, theta, phi) = geodetic_to_geocentric_mag(lat_deg, lon_deg, alt_km);

    // Compute field in geocentric spherical coordinates
    let (b_r, b_theta, b_phi) = synth_field_geocentric(r, theta, phi, &g, &h, nmax, RE_MAGNETIC);

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
    fn test_parse_cof_basic() {
        let coeffs = &*WMMHR_COEFFICIENTS;
        assert_eq!(coeffs.n_max, 133);
        assert_abs_diff_eq!(coeffs.epoch, 2025.0, epsilon = 1e-6);

        // Spot check: g(1,0) should be ~ -29351.7976
        let idx_10 = nm_to_idx(1, 0);
        assert_abs_diff_eq!(coeffs.g[idx_10], -29351.7976, epsilon = 0.01);

        // g(1,1) should be ~ -1410.7694
        let idx_11 = nm_to_idx(1, 1);
        assert_abs_diff_eq!(coeffs.g[idx_11], -1410.7694, epsilon = 0.01);

        // h(1,1) should be ~ 4545.3934
        assert_abs_diff_eq!(coeffs.h[idx_11], 4545.3934, epsilon = 0.01);

        // Secular variation: dg(1,0)/dt should be ~ 11.9581
        assert_abs_diff_eq!(coeffs.g_sv[idx_10], 11.9581, epsilon = 0.01);

        // h(n, 0) should be 0 for all n
        for n in 1..=coeffs.n_max {
            let idx = nm_to_idx(n, 0);
            assert_abs_diff_eq!(coeffs.h[idx], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_wmmhr_geodetic_enz_known_values() {
        setup_global_test_eop();

        // Test against WMMHR official test values:
        // 2025.0, h=0 km, lat=80, lon=0
        // X (north) = 6517.4, Y (east) = 144.8, Z (down) = 54701.3
        // ENZ: B_east = Y = 144.8, B_north = X = 6517.4, B_zenith = -Z = -54701.3
        let epoch = Epoch::from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_geod = Vector3::new(0.0, 80.0, 0.0);
        let b = wmmhr_geodetic_enz(&epoch, x_geod, AngleFormat::Degrees, None).unwrap();

        assert_abs_diff_eq!(b[0], 144.8, epsilon = 1.0); // B_east
        assert_abs_diff_eq!(b[1], 6517.4, epsilon = 1.0); // B_north
        assert_abs_diff_eq!(b[2], -54701.3, epsilon = 1.0); // B_zenith
    }

    #[test]
    fn test_wmmhr_geodetic_enz_equator() {
        setup_global_test_eop();

        // 2025.0, h=0 km, lat=0, lon=120
        // X = 39643.1, Y = -100.3, Z (down) = -10580.7
        // ENZ: B_east = -100.3, B_north = 39643.1, B_zenith = 10580.7
        let epoch = Epoch::from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_geod = Vector3::new(120.0, 0.0, 0.0);
        let b = wmmhr_geodetic_enz(&epoch, x_geod, AngleFormat::Degrees, None).unwrap();

        assert_abs_diff_eq!(b[0], -100.3, epsilon = 1.0); // B_east
        assert_abs_diff_eq!(b[1], 39643.1, epsilon = 1.0); // B_north
        assert_abs_diff_eq!(b[2], 10580.7, epsilon = 1.0); // B_zenith
    }

    #[test]
    fn test_wmmhr_geodetic_enz_south() {
        setup_global_test_eop();

        // 2025.0, h=0 km, lat=-80, lon=240
        // X = 6136.3, Y = 15740.2, Z (down) = -52096.7
        // ENZ: B_east = 15740.2, B_north = 6136.3, B_zenith = 52096.7
        let epoch = Epoch::from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_geod = Vector3::new(240.0, -80.0, 0.0);
        let b = wmmhr_geodetic_enz(&epoch, x_geod, AngleFormat::Degrees, None).unwrap();

        assert_abs_diff_eq!(b[0], 15740.2, epsilon = 1.0); // B_east
        assert_abs_diff_eq!(b[1], 6136.3, epsilon = 1.0); // B_north
        assert_abs_diff_eq!(b[2], 52096.7, epsilon = 1.0); // B_zenith
    }

    #[test]
    fn test_wmmhr_at_altitude() {
        setup_global_test_eop();

        // 2025.0, h=100 km, lat=80, lon=0
        // X = 6218.6, Y = 81.8, Z (down) = 52567.3
        let epoch = Epoch::from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_geod = Vector3::new(0.0, 80.0, 100000.0); // 100 km = 100000 m
        let b = wmmhr_geodetic_enz(&epoch, x_geod, AngleFormat::Degrees, None).unwrap();

        assert_abs_diff_eq!(b[0], 81.8, epsilon = 1.0); // B_east
        assert_abs_diff_eq!(b[1], 6218.6, epsilon = 1.0); // B_north
        assert_abs_diff_eq!(b[2], -52567.3, epsilon = 1.0); // B_zenith
    }

    #[test]
    fn test_wmmhr_secular_variation() {
        setup_global_test_eop();

        // 2027.5, h=0 km, lat=80, lon=0
        // X = 6494.8, Y = 293.5, Z (down) = 54779.0
        let epoch = Epoch::from_datetime(2027, 7, 2, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_geod = Vector3::new(0.0, 80.0, 0.0);
        let b = wmmhr_geodetic_enz(&epoch, x_geod, AngleFormat::Degrees, None).unwrap();

        assert_abs_diff_eq!(b[0], 293.5, epsilon = 2.0); // B_east
        assert_abs_diff_eq!(b[1], 6494.8, epsilon = 2.0); // B_north
        assert_abs_diff_eq!(b[2], -54779.0, epsilon = 2.0); // B_zenith
    }

    #[test]
    fn test_wmmhr_nmax_truncation() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_geod = Vector3::new(0.0, 45.0, 0.0);

        let b_full = wmmhr_geodetic_enz(&epoch, x_geod, AngleFormat::Degrees, None).unwrap();
        let b_low = wmmhr_geodetic_enz(&epoch, x_geod, AngleFormat::Degrees, Some(13)).unwrap();

        // Lower nmax should give different (less detailed) results
        // But the overall magnitude should be similar (within ~200 nT)
        let mag_full = b_full.norm();
        let mag_low = b_low.norm();
        assert!(
            (mag_full - mag_low).abs() < 500.0,
            "Full vs truncated nmax should be similar, diff = {} nT",
            (mag_full - mag_low).abs()
        );

        // They should NOT be exactly equal
        assert!(
            (b_full - b_low).norm() > 0.1,
            "Full and truncated should differ"
        );
    }

    #[test]
    fn test_wmmhr_invalid_nmax() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2025, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_geod = Vector3::new(0.0, 45.0, 0.0);

        let result = wmmhr_geodetic_enz(&epoch, x_geod, AngleFormat::Degrees, Some(0));
        assert!(result.is_err());

        let result = wmmhr_geodetic_enz(&epoch, x_geod, AngleFormat::Degrees, Some(134));
        assert!(result.is_err());
    }

    #[test]
    fn test_wmmhr_ecef_enz_consistency() {
        setup_global_test_eop();

        let epoch = Epoch::from_datetime(2025, 6, 15, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let x_geod = Vector3::new(15.0, 50.0, 500000.0);

        let b_enz = wmmhr_geodetic_enz(&epoch, x_geod, AngleFormat::Degrees, None).unwrap();
        let b_ecef = wmmhr_ecef(&epoch, x_geod, AngleFormat::Degrees, None).unwrap();

        // Magnitudes should be equal (rotation preserves norm)
        assert_abs_diff_eq!(b_enz.norm(), b_ecef.norm(), epsilon = 1e-6);
    }
}
