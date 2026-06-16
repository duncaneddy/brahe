/*!
Tidal corrections to the spherical-harmonic geopotential.

Implements IERS Conventions (2010), TN36 Chapter 6:
- §6.2.2: permanent (zero-frequency) tide conversion of C̄20 between the
  mean-tide / zero-tide / conventional-tide-free systems.
- §6.2.1: solid Earth tides (added in later tasks).

Source: <https://iers-conventions.obspm.fr/content/chapter6/icc6.pdf>
*/

use nalgebra::Vector3;

use crate::orbit_dynamics::gravity::GravityModelTideSystem;

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

    use crate::orbit_dynamics::gravity::ParallelMode;
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
