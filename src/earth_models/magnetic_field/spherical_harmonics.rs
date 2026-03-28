use std::f64::consts::PI;

/// Geomagnetic reference radius [km] used by IGRF and WMMHR models.
pub const RE_MAGNETIC: f64 = 6371.2;

/// Compute Schmidt semi-normalized associated Legendre polynomials P_n^m(cos theta)
/// and their derivatives dP_n^m/d(theta).
///
/// Implementation follows the recursive algorithm from "Spacecraft Attitude Determination
/// and Control" (Wertz), as used in the ppigrf reference implementation.
///
/// # Arguments
///
/// * `theta` - Colatitude in radians (0 = north pole, PI = south pole)
/// * `n_max` - Maximum spherical harmonic degree
///
/// # Returns
///
/// `(P, dP)` where `P[n][m]` and `dP[n][m]` are the Schmidt semi-normalized associated
/// Legendre function and its theta-derivative at the given colatitude.
/// Arrays are indexed as `[n][m]` with n from 0..=n_max and m from 0..=n.
pub(crate) fn legendre_schmidt(theta: f64, n_max: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let cos_theta = theta.cos();
    let sin_theta = theta.sin();

    // Allocate P, dP, and S arrays. Each row has n_max+1 elements so that
    // the recursion p[n-2][m] never goes out of bounds (p[n][m] = 0 for m > n).
    let size = n_max + 1;
    let mut p = vec![vec![0.0; size]; size];
    let mut dp = vec![vec![0.0; size]; size];
    let mut s = vec![vec![0.0; size]; size];
    s[0][0] = 1.0;

    // Initialize P[0][0] = 1
    p[0][0] = 1.0;

    for n in 1..=n_max {
        for m in 0..=n {
            if n == m {
                // Sectoral recursion: P[n][n] = sin(theta) * P[n-1][n-1]
                p[n][n] = sin_theta * p[n - 1][m - 1];
                dp[n][n] = sin_theta * dp[n - 1][m - 1] + cos_theta * p[n - 1][n - 1];
            } else if n == 1 {
                // Special case for n=1, m=0 (no n-2 term)
                p[n][m] = cos_theta * p[n - 1][m];
                dp[n][m] = cos_theta * dp[n - 1][m] - sin_theta * p[n - 1][m];
            } else {
                // Zonal/tesseral recursion
                let knm = ((n - 1) * (n - 1) - m * m) as f64 / ((2 * n - 1) * (2 * n - 3)) as f64;
                p[n][m] = cos_theta * p[n - 1][m] - knm * p[n - 2][m];
                dp[n][m] = cos_theta * dp[n - 1][m] - sin_theta * p[n - 1][m] - knm * dp[n - 2][m];
            }

            // Compute Schmidt normalization factors
            if m == 0 {
                s[n][0] = s[n - 1][0] * (2 * n - 1) as f64 / n as f64;
            } else {
                let delta_m1 = if m == 1 { 1.0 } else { 0.0 };
                s[n][m] =
                    s[n][m - 1] * ((n - m + 1) as f64 * (1.0 + delta_m1) / (n + m) as f64).sqrt();
            }
        }
    }

    // Apply Schmidt normalization
    for n in 1..=n_max {
        for m in 0..=n {
            p[n][m] *= s[n][m];
            dp[n][m] *= s[n][m];
        }
    }

    (p, dp)
}

/// Compute magnetic field in geocentric spherical coordinates from spherical harmonic coefficients.
///
/// This is the core field synthesis function shared by IGRF and WMMHR. It evaluates the
/// magnetic scalar potential gradient using time-interpolated Gauss coefficients.
///
/// # Arguments
///
/// * `r` - Geocentric radius in km
/// * `theta` - Geocentric colatitude in radians
/// * `phi` - East longitude in radians
/// * `g` - Time-interpolated Gauss g coefficients, flat-indexed by (n,m)
/// * `h` - Time-interpolated Gauss h coefficients, flat-indexed by (n,m)
/// * `n_max` - Maximum degree of expansion
/// * `re` - Reference radius in km (typically [`RE_MAGNETIC`] = 6371.2)
///
/// # Returns
///
/// `(Br, Btheta, Bphi)` - Magnetic field components in nanoTesla:
/// - `Br`: radial (outward positive)
/// - `Btheta`: colatitude direction (southward positive)
/// - `Bphi`: east longitude direction (eastward positive)
///
/// # Coefficient indexing
///
/// Coefficients `g` and `h` are indexed by a flat index for each `(n, m)` pair:
/// `idx(n, m) = n*(n+1)/2 + m - 1` for n=1..=n_max, m=0..=n.
pub(crate) fn synth_field_geocentric(
    r: f64,
    theta: f64,
    phi: f64,
    g: &[f64],
    h: &[f64],
    n_max: usize,
    re: f64,
) -> (f64, f64, f64) {
    // Clamp theta to avoid singularity at poles (sin(theta) = 0)
    let theta = theta.clamp(1e-10, PI - 1e-10);

    // Compute Legendre polynomials
    let (p, dp) = legendre_schmidt(theta, n_max);

    let sin_theta = theta.sin();

    // Precompute cos(m*phi) and sin(m*phi) for m = 0..=n_max
    let mut cos_m_phi = vec![0.0; n_max + 1];
    let mut sin_m_phi = vec![0.0; n_max + 1];
    for m in 0..=n_max {
        cos_m_phi[m] = (m as f64 * phi).cos();
        sin_m_phi[m] = (m as f64 * phi).sin();
    }

    let mut b_r = 0.0;
    let mut b_theta = 0.0;
    let mut b_phi = 0.0;

    for n in 1..=n_max {
        // (RE/r)^(n+2) = (RE/r)^(n+1) * (RE/r)
        let ratio = re / r;
        let ratio_n2 = ratio.powi(n as i32 + 2);

        for m in 0..=n {
            let idx = n * (n + 1) / 2 + m - 1;

            let g_nm = g[idx];
            let h_nm = h[idx];

            let p_nm = p[n][m];
            let dp_nm = dp[n][m];

            let cos_mp = cos_m_phi[m];
            let sin_mp = sin_m_phi[m];

            // gh_cos = g * cos(m*phi) + h * sin(m*phi)
            let gh_cos = g_nm * cos_mp + h_nm * sin_mp;
            // gh_sin = -g * sin(m*phi) + h * cos(m*phi)
            let gh_sin = -g_nm * sin_mp + h_nm * cos_mp;

            // Br = sum[(RE/r)^(n+2) * (n+1) * gh_cos * P_n^m]
            b_r += ratio_n2 * (n + 1) as f64 * gh_cos * p_nm;

            // Btheta = -sum[(RE/r)^(n+2) * gh_cos * dP_n^m]
            b_theta -= ratio_n2 * gh_cos * dp_nm;

            // Bphi = -sum[(RE/r)^(n+2) * m * gh_sin * P_n^m / sin(theta)]
            b_phi -= ratio_n2 * m as f64 * gh_sin * p_nm / sin_theta;
        }
    }

    (b_r, b_theta, b_phi)
}

/// Convert flat coefficient index to (n, m) pair.
///
/// `idx(n, m) = n*(n+1)/2 + m - 1` for n >= 1.
#[allow(dead_code)]
pub(crate) fn idx_to_nm(idx: usize) -> (usize, usize) {
    // n*(n+1)/2 - 1 <= idx, so n ~ sqrt(2*idx)
    let n = ((((2 * (idx + 1)) as f64).sqrt() + 0.5) as usize).max(1);
    let base = n * (n + 1) / 2 - 1;
    if idx >= base && idx < base + n + 1 {
        (n, idx - base)
    } else {
        let n = n - 1;
        let base = n * (n + 1) / 2 - 1;
        (n, idx - base)
    }
}

/// Convert (n, m) pair to flat coefficient index.
///
/// Returns the flat index for the given degree `n` and order `m`.
pub(crate) fn nm_to_idx(n: usize, m: usize) -> usize {
    n * (n + 1) / 2 + m - 1
}

/// Total number of coefficients for a given maximum degree.
///
/// This is the number of `(n, m)` pairs for n=1..=n_max, m=0..=n,
/// equal to `n_max * (n_max + 3) / 2`.
///
/// For IGRF (n_max=13): 104 coefficients.
/// For WMMHR (n_max=133): 9044 coefficients.
pub(crate) fn num_coefficients(n_max: usize) -> usize {
    n_max * (n_max + 3) / 2
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_legendre_p00_is_one() {
        // P[0][0] = 1.0 for any theta
        for &theta_deg in &[0.1_f64, 30.0, 45.0, 60.0, 89.9, 179.9] {
            let theta = theta_deg.to_radians();
            let (p, _) = legendre_schmidt(theta, 3);
            assert_abs_diff_eq!(p[0][0], 1.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_legendre_p10_is_cos_theta() {
        // After Schmidt normalization, P[1][0] = cos(theta)
        for &theta_deg in &[10.0_f64, 30.0, 45.0, 60.0, 80.0, 150.0] {
            let theta = theta_deg.to_radians();
            let (p, _) = legendre_schmidt(theta, 3);
            assert_abs_diff_eq!(p[1][0], theta.cos(), epsilon = 1e-12);
        }
    }

    #[test]
    fn test_legendre_p11_is_sin_theta() {
        // P[1][1] = sin(theta) (Schmidt semi-normalized)
        for &theta_deg in &[10.0_f64, 30.0, 45.0, 60.0, 80.0, 150.0] {
            let theta = theta_deg.to_radians();
            let (p, _) = legendre_schmidt(theta, 3);
            assert_abs_diff_eq!(p[1][1], theta.sin(), epsilon = 1e-12);
        }
    }

    #[test]
    fn test_legendre_p20() {
        // Verify P[2][0] at theta = 60 degrees using recursion math.
        // From recursion (before normalization): P[2][0] = cos*P[1][0] - K20*P[0][0]
        // K20 = (1-0)/((3)(1)) = 1/3
        // cos(60°) = 0.5, P[1][0] = 0.5
        // P[2][0] = 0.5*0.5 - 1/3*1 = -0.0833... (before normalization)
        // S[2][0] = 1.5
        // P_schmidt[2][0] = -0.0833 * 1.5 = -0.125
        let theta = 60.0_f64.to_radians();
        let (p, _) = legendre_schmidt(theta, 2);
        assert_abs_diff_eq!(p[2][0], -0.125, epsilon = 1e-12);

        // Near north pole: cos(0) ≈ 1, recursion gives 1*1 - 1/3 = 2/3, times S=1.5 = 1.0
        let theta0 = 0.001_f64.to_radians();
        let (p0, _) = legendre_schmidt(theta0, 2);
        assert_abs_diff_eq!(p0[2][0], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_nm_to_idx_roundtrip() {
        // Verify index mapping
        assert_eq!(nm_to_idx(1, 0), 0);
        assert_eq!(nm_to_idx(1, 1), 1);
        assert_eq!(nm_to_idx(2, 0), 2);
        assert_eq!(nm_to_idx(2, 1), 3);
        assert_eq!(nm_to_idx(2, 2), 4);
        assert_eq!(nm_to_idx(3, 0), 5);
    }

    #[test]
    fn test_num_coefficients() {
        // n_max=1: (1,0), (1,1) = 2 coefficients
        assert_eq!(num_coefficients(1), 2);
        // n_max=2: (1,0),(1,1), (2,0),(2,1),(2,2) = 5
        assert_eq!(num_coefficients(2), 5);
        // n_max=13 (IGRF): 104
        assert_eq!(num_coefficients(13), 104);
        // n_max=133 (WMMHR): 9044
        assert_eq!(num_coefficients(133), 9044);
    }
}
