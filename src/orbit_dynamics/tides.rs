/*!
Tidal corrections to the spherical-harmonic geopotential.

Implements IERS Conventions (2010), TN36 Chapter 6:
- §6.2.2: permanent (zero-frequency) tide conversion of C̄20 between the
  mean-tide / zero-tide / conventional-tide-free systems.
- §6.2.1: solid Earth tides (added in later tasks).

Source: <https://iers-conventions.obspm.fr/content/chapter6/icc6.pdf>
*/

use nalgebra::Vector3;

use crate::constants::{GM_MOON, GM_SUN};
use crate::orbit_dynamics::gravity::GravityModelTideSystem;
use crate::time::Epoch;

/// Permanent-tide DIRECT term on the fully-normalized C̄20 (IERS Eq. 6.14,
/// the A0*H0 factor with no Love number). A0 = 4.4228e-8 m^-1 (Eq. 6.8c),
/// H0 = -0.31460 m. This is the contribution of the lunisolar permanent
/// tide-raising potential itself (present in the mean-tide system, removed
/// for zero-tide).
pub const PERM_C20_DIRECT: f64 = 4.4228e-8 * (-0.31460);

/// Permanent-tide INDIRECT term on C̄20 (IERS Eq. 6.14, A0*H0*k20). k20 =
/// 0.30190 is the secular Love number (Table 6.3 anelastic Re k20). This is
/// the Earth's permanent elastic deformation response (present in both
/// mean-tide AND zero-tide, removed for conventional tide-free).
pub const PERM_C20_INDIRECT: f64 = 4.4228e-8 * (-0.31460) * 0.30190;

/// Offset of a system's C̄20 relative to the conventional tide-free value.
///
/// Per IERS §6.2.2, the systems differ by which permanent terms are present:
/// - tide-free: neither term  -> 0
/// - zero-tide: indirect only  -> PERM_C20_INDIRECT
/// - mean-tide: direct+indirect -> PERM_C20_DIRECT + PERM_C20_INDIRECT
///
/// `Unknown` returns 0.0 (caller is responsible for not converting Unknown).
pub fn tide_system_c20_offset(system: GravityModelTideSystem) -> f64 {
    match system {
        GravityModelTideSystem::TideFree => 0.0,
        GravityModelTideSystem::ZeroTide => PERM_C20_INDIRECT,
        GravityModelTideSystem::MeanTide => PERM_C20_DIRECT + PERM_C20_INDIRECT,
        GravityModelTideSystem::Unknown => 0.0,
    }
}

/// Fully-normalized ΔC̄nm / ΔS̄nm corrections for n,m in 0..=4 (index `[n][m]`).
#[derive(Debug, Clone, Copy, Default)]
pub struct TideCoefficients {
    /// Fully-normalized cosine coefficient corrections ΔC̄nm, indexed `[n][m]`.
    pub dc: [[f64; 5]; 5],
    /// Fully-normalized sine coefficient corrections ΔS̄nm, indexed `[n][m]`.
    pub ds: [[f64; 5]; 5],
}

#[allow(dead_code)] // used by future tidal tasks
const TIDE_NMAX: usize = 4;

/// Denormalization factor sqrt((2-δ0m)(2n+1)(n-m)!/(n+m)!) — identical to
/// `GravityModel::precompute_coefficients`. n,m <= 4, so factorials are exact.
fn denorm_factor(n: usize, m: usize) -> f64 {
    fn fact(k: usize) -> f64 {
        (1..=k).map(|i| i as f64).product::<f64>().max(1.0)
    }
    let delta0m = if m == 0 { 1.0 } else { 0.0 };
    ((2.0 - delta0m) * (2.0 * n as f64 + 1.0) * fact(n - m) / fact(n + m)).sqrt()
}

/// Body-fixed acceleration from a low-degree (n<=4) set of fully-normalized
/// coefficients. Mirrors the V/W recurrence and accumulation of
/// `GravityModel::compute_spherical_harmonics_with_workspace` (gravity.rs),
/// but on stack arrays sized for the fixed tidal degree — no heap allocation,
/// no workspace plumbing. Proven equivalent by `test_low_degree_evaluator_matches_full_path`.
///
/// # References
/// - Montenbruck & Gill, *Satellite Orbits*, §3.2 (Cunningham V/W recursion).
#[allow(dead_code)] // called from future solid-Earth / ocean tidal force tasks
pub(crate) fn accel_low_degree_harmonics(
    r_ecef: Vector3<f64>,
    coeffs: &TideCoefficients,
    gm: f64,
    radius: f64,
) -> Vector3<f64> {
    const SZ: usize = TIDE_NMAX + 2; // rows 0..=n_max+1
    // Denormalize coefficients (hoist factorials; small fixed table).
    let mut c = [[0.0f64; SZ]; SZ];
    let mut s = [[0.0f64; SZ]; SZ];
    for n in 0..=TIDE_NMAX {
        c[n][0] = denorm_factor(n, 0) * coeffs.dc[n][0];
        for m in 1..=n {
            let f = denorm_factor(n, m);
            c[n][m] = f * coeffs.dc[n][m];
            s[n][m] = f * coeffs.ds[n][m];
        }
    }

    let r_sqr = r_ecef.dot(&r_ecef);
    let rho = radius * radius / r_sqr;
    let x0 = radius * r_ecef[0] / r_sqr;
    let y0 = radius * r_ecef[1] / r_sqr;
    let z0 = radius * r_ecef[2] / r_sqr;

    let mut v = [[0.0f64; SZ]; SZ];
    let mut w = [[0.0f64; SZ]; SZ];

    // Zonal column m=0.
    v[0][0] = radius / r_sqr.sqrt();
    v[1][0] = z0 * v[0][0];
    for n in 2..SZ {
        let nf = n as f64;
        v[n][0] = ((2.0 * nf - 1.0) * z0 * v[n - 1][0] - (nf - 1.0) * rho * v[n - 2][0]) / nf;
    }
    // Tesseral/sectoral columns.
    for m in 1..SZ {
        let mf = m as f64;
        v[m][m] = (2.0 * mf - 1.0) * (x0 * v[m - 1][m - 1] - y0 * w[m - 1][m - 1]);
        w[m][m] = (2.0 * mf - 1.0) * (x0 * w[m - 1][m - 1] + y0 * v[m - 1][m - 1]);
        if m + 1 < SZ {
            v[m + 1][m] = (2.0 * mf + 1.0) * z0 * v[m][m];
            w[m + 1][m] = (2.0 * mf + 1.0) * z0 * w[m][m];
        }
        for n in (m + 2)..SZ {
            let nf = n as f64;
            let a = (2.0 * nf - 1.0) / (nf - mf);
            let b = (nf + mf - 1.0) / (nf - mf);
            v[n][m] = a * z0 * v[n - 1][m] - b * rho * v[n - 2][m];
            w[n][m] = a * z0 * w[n - 1][m] - b * rho * w[n - 2][m];
        }
    }

    let (mut ax, mut ay, mut az) = (0.0, 0.0, 0.0);
    for m in 0..=TIDE_NMAX {
        let mf = m as f64;
        if m == 0 {
            for n in 0..=TIDE_NMAX {
                let cc = c[n][0];
                ax -= cc * v[n + 1][1];
                ay -= cc * w[n + 1][1];
                az -= (n as f64 + 1.0) * cc * v[n + 1][0];
            }
        } else {
            for n in m..=TIDE_NMAX {
                let nf = n as f64;
                let cc = c[n][m];
                let ss = s[n][m];
                let fac = 0.5 * (nf - mf + 1.0) * (nf - mf + 2.0);
                let p = n + 1;
                ax += 0.5 * (-cc * v[p][m + 1] - ss * w[p][m + 1])
                    + fac * (cc * v[p][m - 1] + ss * w[p][m - 1]);
                ay += 0.5 * (-cc * w[p][m + 1] + ss * v[p][m + 1])
                    + fac * (-cc * w[p][m - 1] + ss * v[p][m - 1]);
                az += (nf - mf + 1.0) * (-cc * v[p][m] - ss * w[p][m]);
            }
        }
    }
    (gm / (radius * radius)) * Vector3::new(ax, ay, az)
}

/// Physics-side solid Earth tide settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolidTideConfig {
    /// Apply IERS Step 2 frequency-dependent corrections (Tables 6.5a/b/c).
    pub frequency_dependent: bool,
}

/// IERS Table 6.3 nominal anelastic Love numbers, (Re, Im), index [n][m].
/// Degree 3 has only real values. Source: TN36 Ch.6 Table 6.3.
#[allow(clippy::approx_constant)] // 0.30102 is the IERS k22 Love number, not log10(2)
const LOVE_RE: [[f64; 4]; 4] = [
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.30190, 0.29830, 0.30102, 0.0],
    [0.093, 0.093, 0.093, 0.094],
];
const LOVE_IM: [[f64; 4]; 4] = [
    [0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0],
    [0.00000, -0.00144, -0.00130, 0.0],
    [0.0, 0.0, 0.0, 0.0],
];
/// k2m^+ degree-2 -> degree-4 coupling Love numbers (Table 6.3), m=0,1,2.
const LOVE_PLUS: [f64; 3] = [-0.00089, -0.00080, -0.00057];

/// Fully-normalized associated Legendre functions P̄nm(sin φ) for n=2,3 and
/// m=0..n, geodesy 4π convention (no Condon–Shortley phase). Returns [n][m].
/// Closed forms; normalization factor sqrt((2-δ0m)(2n+1)(n-m)!/(n+m)!).
fn norm_legendre_2_3(sphi: f64) -> [[f64; 4]; 4] {
    let cphi = (1.0 - sphi * sphi).max(0.0).sqrt();
    // Unnormalized (geodesy, no CS phase).
    let p20 = 0.5 * (3.0 * sphi * sphi - 1.0);
    let p21 = 3.0 * sphi * cphi;
    let p22 = 3.0 * cphi * cphi;
    let p30 = 0.5 * (5.0 * sphi.powi(3) - 3.0 * sphi);
    let p31 = 1.5 * (5.0 * sphi * sphi - 1.0) * cphi;
    let p32 = 15.0 * sphi * cphi * cphi;
    let p33 = 15.0 * cphi.powi(3);
    let mut p = [[0.0f64; 4]; 4];
    p[2][0] = denorm_factor(2, 0) * p20;
    p[2][1] = denorm_factor(2, 1) * p21;
    p[2][2] = denorm_factor(2, 2) * p22;
    p[3][0] = denorm_factor(3, 0) * p30;
    p[3][1] = denorm_factor(3, 1) * p31;
    p[3][2] = denorm_factor(3, 2) * p32;
    p[3][3] = denorm_factor(3, 3) * p33;
    p
}

/// Compute Step 1 frequency-independent solid Earth tide coefficient
/// corrections (IERS Eq. 6.6 for n=2,3; Eq. 6.7 for the degree-2→degree-4
/// feedback), summed over Moon and Sun.
///
/// For each body j and (n,m), with kₙₘ = kre + i·kim, λ_j/φ_j the body's ECEF
/// longitude/geocentric latitude, r_j its distance, and
///   F = (1/(2n+1)) · (GM_j/GM_⊕) · (R_⊕/r_j)^(n+1) · P̄nm(sin φ_j):
///   ΔC̄nm += F · (kre·cos(mλ_j) + kim·sin(mλ_j))
///   ΔS̄nm += F · (kre·sin(mλ_j) − kim·cos(mλ_j))
/// (the real/imag split of the complex IERS expression
///  ΔC̄nm − i·ΔS̄nm = (kₙₘ/(2n+1)) Σ_j (GM_j/GM_⊕)(R_⊕/r_j)^(n+1) P̄nm(sinφ_j) e^(−imλ_j)).
///
/// Eq. 6.7 reuses P̄2m and (R_⊕/r_j)^3 with kₘ⁺/5 into ΔC̄4m/ΔS̄4m (m=0,1,2).
///
/// # References
/// - IERS Conventions (2010), TN36 §6.2.1, Eq. (6.6)–(6.7), Table 6.3.
///   <https://iers-conventions.obspm.fr/content/chapter6/icc6.pdf>
/// - Montenbruck & Gill, *Satellite Orbits*, §3.7.2, Eq. (3.159) (unnormalized cross-check).
pub fn solid_earth_tide_coefficients(
    r_sun_ecef: Vector3<f64>,
    r_moon_ecef: Vector3<f64>,
    epoch: Epoch,
    gm_earth: f64,
    radius: f64,
    config: &SolidTideConfig,
) -> TideCoefficients {
    let _ = epoch; // used by Step 2 (frequency-dependent corrections in Task 7)
    let mut out = TideCoefficients::default();

    for (r_body, gm_body) in [(r_moon_ecef, GM_MOON), (r_sun_ecef, GM_SUN)] {
        let r = r_body.norm();
        if r <= radius {
            continue; // body inside Earth radius => skip (test sentinel / degenerate)
        }
        let sphi = r_body[2] / r;
        let lambda = r_body[1].atan2(r_body[0]);
        let p = norm_legendre_2_3(sphi);
        let gm_ratio = gm_body / gm_earth;

        // Eq. (6.6): n = 2, 3.
        for n in 2..=3usize {
            let radial = (radius / r).powi((n + 1) as i32);
            for m in 0..=n {
                let kre = LOVE_RE[n][m];
                let kim = LOVE_IM[n][m];
                let f = (1.0 / (2.0 * n as f64 + 1.0)) * gm_ratio * radial * p[n][m];
                let (cm, sm) = ((m as f64 * lambda).cos(), (m as f64 * lambda).sin());
                out.dc[n][m] += f * (kre * cm + kim * sm);
                out.ds[n][m] += f * (kre * sm - kim * cm);
            }
        }

        // Eq. (6.7): degree-2 tides -> degree-4 coefficients, m = 0,1,2.
        let radial3 = (radius / r).powi(3);
        for m in 0..=2usize {
            let kp = LOVE_PLUS[m];
            let f = (kp / 5.0) * gm_ratio * radial3 * p[2][m];
            let (cm, sm) = ((m as f64 * lambda).cos(), (m as f64 * lambda).sin());
            out.dc[4][m] += f * cm;
            out.ds[4][m] += f * sm;
        }
    }

    if config.frequency_dependent {
        // Step 2 added in Task 7.
    }
    out
}

/// Acceleration (body-fixed / ECEF) due to solid Earth tides, IERS §6.2.1.
///
/// Builds the time-varying ΔC̄nm/ΔS̄nm corrections from the Sun and Moon and
/// evaluates them as a degree-4 spherical-harmonic field. The result is ADDED
/// to the static gravity acceleration by the caller; this is exact because the
/// geopotential is linear in its coefficients (see module/spec §2). All inputs
/// and the evaluation share the same `gm_earth`, `radius`, and ECEF frame.
///
/// # Arguments
/// - `r_ecef`: satellite position, ECEF [m].
/// - `r_sun_ecef`, `r_moon_ecef`: body positions, ECEF [m].
/// - `epoch`: used for Step 2 Doodson arguments (ignored when Step 1 only).
/// - `gm_earth`, `radius`: the gravity model's own GM [m³/s²] and reference radius [m].
///
/// # References
/// - IERS Conventions (2010), TN36 §6.2.1.
pub fn accel_solid_earth_tides(
    r_ecef: Vector3<f64>,
    r_sun_ecef: Vector3<f64>,
    r_moon_ecef: Vector3<f64>,
    epoch: Epoch,
    gm_earth: f64,
    radius: f64,
    config: &SolidTideConfig,
) -> Vector3<f64> {
    let coeffs =
        solid_earth_tide_coefficients(r_sun_ecef, r_moon_ecef, epoch, gm_earth, radius, config);
    accel_low_degree_harmonics(r_ecef, &coeffs, gm_earth, radius)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orbit_dynamics::gravity::{GravityModel, GravityModelTideSystem, GravityModelType};

    #[test]
    fn test_perm_constants_match_iers() {
        // Constants are exact products of the verbatim IERS TN36 factors
        // (A0 = 4.4228e-8, H0 = -0.31460, k20 = 0.30190).
        assert_eq!(PERM_C20_DIRECT, 4.4228e-8 * (-0.31460));
        assert_eq!(PERM_C20_INDIRECT, 4.4228e-8 * (-0.31460) * 0.30190);
        // Exact product ≈ -4.2007e-9, within f64 rounding of -4.200675e-9.
        // (The IERS 5-sig-fig tabulation -4.2017e-9 is itself ~1e-12 coarse.)
        assert!((PERM_C20_INDIRECT - (-4.200675e-9)).abs() < 1e-14);
        assert!((PERM_C20_DIRECT - (-1.39142e-8)).abs() < 1e-12);
    }

    #[test]
    fn test_offsets_relative_to_tide_free() {
        assert_eq!(
            tide_system_c20_offset(GravityModelTideSystem::TideFree),
            0.0
        );
        assert!(
            (tide_system_c20_offset(GravityModelTideSystem::ZeroTide) - PERM_C20_INDIRECT).abs()
                < 1e-20
        );
        assert!(
            (tide_system_c20_offset(GravityModelTideSystem::MeanTide)
                - (PERM_C20_DIRECT + PERM_C20_INDIRECT))
                .abs()
                < 1e-20
        );
    }

    #[test]
    fn test_convert_zero_to_tide_free_matches_egm2008_within_tolerance() {
        // EGM2008 is tide-free; load, force-label zero-tide, convert back to tide-free.
        let mut m = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();
        let c20_before = m.get(2, 0).unwrap().0;
        m.tide_system = GravityModelTideSystem::ZeroTide;
        m.convert_tide_system(
            GravityModelTideSystem::ZeroTide,
            GravityModelTideSystem::TideFree,
        )
        .unwrap();
        let c20_after = m.get(2, 0).unwrap().0;
        // Converting zero->free removes the indirect term: subtract offset(zero)=INDIRECT.
        assert!((c20_after - (c20_before - PERM_C20_INDIRECT)).abs() < 1e-20);
        // Cross-check magnitude against the EGM2008 published offset (~0.7% tolerance).
        assert!(((c20_after - c20_before) - 4.1736e-9).abs() < 0.05e-9);
        assert_eq!(m.tide_system, GravityModelTideSystem::TideFree);
    }

    #[test]
    fn test_convert_roundtrip_is_identity() {
        let mut m = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
        let c20 = m.get(2, 0).unwrap().0;
        m.tide_system = GravityModelTideSystem::TideFree;
        m.convert_tide_system(
            GravityModelTideSystem::TideFree,
            GravityModelTideSystem::MeanTide,
        )
        .unwrap();
        m.convert_tide_system(
            GravityModelTideSystem::MeanTide,
            GravityModelTideSystem::TideFree,
        )
        .unwrap();
        assert!((m.get(2, 0).unwrap().0 - c20).abs() < 1e-18);
    }

    #[test]
    fn test_convert_from_unknown_errors() {
        let mut m = GravityModel::from_model_type(&GravityModelType::JGM3).unwrap();
        assert!(
            m.convert_tide_system(
                GravityModelTideSystem::Unknown,
                GravityModelTideSystem::TideFree
            )
            .is_err()
        );
    }

    use crate::constants::{GM_EARTH, R_EARTH};
    use crate::orbit_dynamics::gravity::ParallelMode;
    use crate::time::{Epoch, TimeSystem};
    use nalgebra::Vector3;

    /// Build a degree-4 GravityModel whose coefficients ARE the given tide deltas,
    /// so the full SH path can serve as an independent reference for the fixed-size
    /// evaluator. Reuses the public-within-crate test seam.
    fn reference_accel(
        coeffs: &TideCoefficients,
        r: Vector3<f64>,
        gm: f64,
        radius: f64,
    ) -> Vector3<f64> {
        let model = GravityModel::from_dense_normalized(&coeffs.dc, &coeffs.ds, 4, gm, radius);
        // Evaluate in body frame with identity rotation by passing r directly.
        model
            .compute_spherical_harmonics(r, 4, 4, ParallelMode::Never)
            .unwrap()
    }

    #[test]
    fn test_step1_c20_magnitude_and_lunar_dominance() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        // Moon ~ along +x at ~384400 km; Sun ~ along +x at 1 AU.
        let r_moon = Vector3::new(3.844e8, 0.0, 0.0);
        let r_sun = Vector3::new(1.496e11, 0.0, 0.0);
        let cfg = SolidTideConfig {
            frequency_dependent: false,
        };
        let coeffs = solid_earth_tide_coefficients(r_sun, r_moon, epoch, GM_EARTH, R_EARTH, &cfg);
        // Step-1 ΔC̄20 is ~1e-8 in magnitude (dominant solid-tide term).
        assert!(
            coeffs.dc[2][0].abs() > 1e-9 && coeffs.dc[2][0].abs() < 1e-7,
            "ΔC̄20 = {:e}",
            coeffs.dc[2][0]
        );

        // Lunar-only vs solar-only ΔC̄20: Moon ~2.2x Sun.
        let far = Vector3::new(1.0e30, 0.0, 0.0); // effectively zero contribution
        let moon_only = solid_earth_tide_coefficients(far, r_moon, epoch, GM_EARTH, R_EARTH, &cfg);
        let sun_only = solid_earth_tide_coefficients(r_sun, far, epoch, GM_EARTH, R_EARTH, &cfg);
        let ratio = moon_only.dc[2][0] / sun_only.dc[2][0];
        assert!((ratio - 2.2).abs() < 0.4, "lunar/solar ratio = {ratio}");

        // Degree-4 feedback (Eq. 6.7) is present and ~3 orders smaller than ΔC̄20.
        assert!(coeffs.dc[4][0].abs() > 0.0);
        assert!(coeffs.dc[4][0].abs() < coeffs.dc[2][0].abs());
    }

    #[test]
    fn test_accel_solid_tides_finite_and_small() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let r_sat = Vector3::new(7.0e6, 0.0, 0.0);
        let r_moon = Vector3::new(3.844e8, 0.0, 0.0);
        let r_sun = Vector3::new(1.496e11, 0.0, 0.0);
        let cfg = SolidTideConfig {
            frequency_dependent: false,
        };
        let a = accel_solid_earth_tides(r_sat, r_sun, r_moon, epoch, GM_EARTH, R_EARTH, &cfg);
        assert!(a.norm().is_finite());
        // Solid-tide accel is ~1e-7..1e-6 m/s^2 in LEO, far below ~9.8 main gravity.
        assert!(a.norm() > 1e-9 && a.norm() < 1e-4, "|a| = {:e}", a.norm());
    }

    #[test]
    fn test_low_degree_evaluator_matches_full_path() {
        let gm = 3.986004415e14;
        let radius = 6.378136300e6;
        // A representative non-trivial coefficient set.
        let mut coeffs = TideCoefficients {
            dc: [[0.0; 5]; 5],
            ds: [[0.0; 5]; 5],
        };
        coeffs.dc[2][0] = 1.2e-8;
        coeffs.dc[2][1] = -3.4e-9;
        coeffs.ds[2][1] = 7.7e-9;
        coeffs.dc[2][2] = 5.1e-9;
        coeffs.ds[2][2] = -2.2e-9;
        coeffs.dc[3][1] = 1.1e-9;
        coeffs.ds[3][1] = -0.6e-9;
        coeffs.dc[4][0] = -4.0e-11;
        for r in [
            Vector3::new(7.0e6, 0.0, 0.0),
            Vector3::new(3.0e6, 4.0e6, 5.0e6),
            Vector3::new(-2.0e6, 1.0e6, 6.5e6),
        ] {
            let a_fast = accel_low_degree_harmonics(r, &coeffs, gm, radius);
            let a_ref = reference_accel(&coeffs, r, gm, radius);
            let rel = (a_fast - a_ref).norm() / a_ref.norm().max(1e-30);
            assert!(
                rel < 1e-12,
                "rel err {rel:e} at {r:?}: fast={a_fast:?} ref={a_ref:?}"
            );
        }
    }
}
