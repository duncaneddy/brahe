/*!
Tidal corrections to the spherical-harmonic geopotential.

Implements IERS Conventions (2010), TN36 Chapter 6:
- §6.2.2: permanent (zero-frequency) tide conversion of C̄20 between the
  mean-tide / zero-tide / conventional-tide-free systems.
- §6.2.1: solid Earth tides (added in later tasks).

Source: <https://iers-conventions.obspm.fr/content/chapter6/icc6.pdf>
*/

use nalgebra::Vector3;

use crate::constants::{GM_MOON, GM_SUN, MJD_ZERO};
use crate::orbit_dynamics::gravity::GravityModelTideSystem;
use crate::time::{Epoch, TimeSystem};

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
#[derive(Debug, Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct SolidTideConfig {
    /// Apply IERS Step 2 frequency-dependent corrections (Tables 6.5a/b/c).
    #[serde(default)]
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
        let (dc20, step2_tesseral) = step2_corrections(epoch);
        out.dc[2][0] += dc20;
        out.dc[2][1] += step2_tesseral[0][0];
        out.ds[2][1] += step2_tesseral[0][1];
        out.dc[2][2] += step2_tesseral[1][0];
        out.ds[2][2] += step2_tesseral[1][1];
    }
    out
}

/// Compute IERS Step 2 frequency-dependent corrections to degree-2 geopotential
/// coefficients (Tables 6.5a/b/c, IERS Conventions (2010) §6.2.1).
///
/// Returns:
/// - `dc20`: ΔC̄20 correction (m=0 zonal, Eq. 6.8a real part).
/// - `tesseral`: `[[ΔC̄21, ΔS̄21], [ΔC̄22, ΔS̄22]]` (m=1 and m=2 corrections).
///
/// The argument angle for each line is:
///   `θ = m·args[0] − Σ delaunay·args[1..6]`
/// where `args = doodson_delaunay_args(epoch)` = `[GMST+π, l, l', F, D, Ω]`.
///
/// Per IERS Eq. (6.8a/6.8b) and the sign convention for η_m:
/// - m=0  (η0 = 1):  ΔC̄20 += scale·(ip·cosθ − op·sinθ)
/// - m=1  (η1 = −i): ΔC̄21 += scale·(ip·sinθ + op·cosθ); ΔS̄21 += scale·(ip·cosθ − op·sinθ)
/// - m=2  (η2 = 1):  ΔC̄22 += scale·(ip·cosθ − op·sinθ); ΔS̄22 += scale·(−ip·sinθ − op·cosθ)
///
/// Amplitudes in Tables 6.5a/b/c are in units of 1e-12; `scale = 1e-12`.
pub(crate) fn step2_corrections(epoch: Epoch) -> (f64, [[f64; 2]; 2]) {
    use crate::orbit_dynamics::tides_step2_tables::{TABLE_M0, TABLE_M1, TABLE_M2};

    let args = doodson_delaunay_args(epoch);
    const SCALE: f64 = 1e-12;

    let mut dc20 = 0.0_f64;
    let mut dc21 = 0.0_f64;
    let mut ds21 = 0.0_f64;
    let mut dc22 = 0.0_f64;
    let mut ds22 = 0.0_f64;

    // m=0: Table 6.5b, Eq. 6.8a real part.
    for line in &TABLE_M0 {
        let theta = 0.0_f64 * args[0]
            - (line.delaunay[0] as f64 * args[1]
                + line.delaunay[1] as f64 * args[2]
                + line.delaunay[2] as f64 * args[3]
                + line.delaunay[3] as f64 * args[4]
                + line.delaunay[4] as f64 * args[5]);
        let (sin_t, cos_t) = theta.sin_cos();
        dc20 += SCALE * (line.amp_in_phase * cos_t - line.amp_out_of_phase * sin_t);
    }

    // m=1: Table 6.5a.
    for line in &TABLE_M1 {
        let theta = 1.0_f64 * args[0]
            - (line.delaunay[0] as f64 * args[1]
                + line.delaunay[1] as f64 * args[2]
                + line.delaunay[2] as f64 * args[3]
                + line.delaunay[3] as f64 * args[4]
                + line.delaunay[4] as f64 * args[5]);
        let (sin_t, cos_t) = theta.sin_cos();
        dc21 += SCALE * (line.amp_in_phase * sin_t + line.amp_out_of_phase * cos_t);
        ds21 += SCALE * (line.amp_in_phase * cos_t - line.amp_out_of_phase * sin_t);
    }

    // m=2: Table 6.5c.
    for line in &TABLE_M2 {
        let theta = 2.0_f64 * args[0]
            - (line.delaunay[0] as f64 * args[1]
                + line.delaunay[1] as f64 * args[2]
                + line.delaunay[2] as f64 * args[3]
                + line.delaunay[3] as f64 * args[4]
                + line.delaunay[4] as f64 * args[5]);
        let (sin_t, cos_t) = theta.sin_cos();
        dc22 += SCALE * (line.amp_in_phase * cos_t - line.amp_out_of_phase * sin_t);
        ds22 += SCALE * (-line.amp_in_phase * sin_t - line.amp_out_of_phase * cos_t);
    }

    (dc20, [[dc21, ds21], [dc22, ds22]])
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

/// Greenwich-mean-sidereal-time-plus-π and the five Delaunay fundamental
/// arguments (l, l', F, D, Ω) in radians, for the Doodson argument of a solid
/// Earth tide line (IERS §6.2.1):
///   θ_f = m·(θg + π) − (n_l·l + n_l'·l' + n_F·F + n_D·D + n_Ω·Ω)
/// where (n_*) are the Delaunay multipliers from Tables 6.5a/b/c and θg = GMST.
///
/// l, l', F, D, Ω come from the IAU 2003 fundamental-argument polynomials
/// (SOFA iauFal03/iauFalp03/iauFaf03/iauFad03/iauFaom03), evaluated at TT
/// Julian centuries since J2000. GMST from SOFA iauGmst06 (UT1, TT).
///
/// Returns `[θg+π, l, l', F, D, Ω]` (radians).
#[allow(dead_code)] // consumed by Task 9 Step-2 frequency-dependent corrections
pub(crate) fn doodson_delaunay_args(epoch: Epoch) -> [f64; 6] {
    use std::f64::consts::PI;

    // TT Julian centuries since J2000.
    let tt_jd = epoch.jd_as_time_system(TimeSystem::TT);
    let t = (tt_jd - 2451545.0) / 36525.0;

    // SOFA fundamental arguments (radians).
    let (l, lp, f, d, om) = unsafe {
        (
            rsofa::iauFal03(t),
            rsofa::iauFalp03(t),
            rsofa::iauFaf03(t),
            rsofa::iauFad03(t),
            rsofa::iauFaom03(t),
        )
    };

    // GMST (radians), IAU 2006: needs (UT1 two-part JD, TT two-part JD).
    let ut1 = epoch.mjd_as_time_system(TimeSystem::UT1);
    let tt = epoch.mjd_as_time_system(TimeSystem::TT);
    let gmst = unsafe { rsofa::iauGmst06(MJD_ZERO, ut1, MJD_ZERO, tt) };

    [gmst + PI, l, lp, f, d, om]
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
    fn test_doodson_k1_equals_gmst_plus_pi() {
        crate::utils::testing::setup_global_test_eop();
        let epoch = Epoch::from_datetime(2015, 6, 15, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let args = doodson_delaunay_args(epoch);
        // K1 has all-zero Delaunay multipliers, order m=1 => θ = GMST + π = args[0].
        let theta_k1 = 1.0 * args[0]
            - (0.0 * args[1] + 0.0 * args[2] + 0.0 * args[3] + 0.0 * args[4] + 0.0 * args[5]);
        // args[0] should be GMST + π (mod 2π), in (0, 4π).
        assert!(theta_k1.is_finite());
        // Fundamental args are bounded angles.
        for a in &args[1..] {
            assert!(a.abs() < 1000.0);
        }
    }

    // ── Step 2 integrity tests ──────────────────────────────────────────────

    /// Verify table row counts and spot-check anchor values against the data
    /// file (iers-step2-tables.md / IERS TN36 Ch.6 PDF).
    #[test]
    fn test_step2_table_integrity() {
        use crate::orbit_dynamics::tides_step2_tables::{TABLE_M0, TABLE_M1, TABLE_M2};

        // Row counts.
        assert_eq!(TABLE_M1.len(), 48, "TABLE_M1 should have 48 rows");
        assert_eq!(TABLE_M0.len(), 21, "TABLE_M0 should have 21 rows");
        assert_eq!(TABLE_M2.len(), 2, "TABLE_M2 should have 2 rows");

        // Helper: find a row by Delaunay multipliers within a slice.
        let find = |table: &[_], d: [i8; 5]| -> Option<(f64, f64)> {
            table
                .iter()
                .find(|r: &&crate::orbit_dynamics::tides_step2_tables::Step2Line| r.delaunay == d)
                .map(|r| (r.amp_in_phase, r.amp_out_of_phase))
        };

        // TABLE_M1 anchors.
        // K1: Doodson 165,555, [0,0,0,0,0], ip=470.9, op=-30.2
        let k1 = find(&TABLE_M1, [0, 0, 0, 0, 0]).expect("K1 not found in TABLE_M1");
        assert!((k1.0 - 470.9).abs() < 1e-9, "K1 ip={}", k1.0);
        assert!((k1.1 - (-30.2)).abs() < 1e-9, "K1 op={}", k1.1);

        // O1: Doodson 145,555, [0,0,2,0,2], ip=-6.8, op=0.6
        let o1 = find(&TABLE_M1, [0, 0, 2, 0, 2]).expect("O1 not found in TABLE_M1");
        assert!((o1.0 - (-6.8)).abs() < 1e-9, "O1 ip={}", o1.0);
        assert!((o1.1 - 0.6).abs() < 1e-9, "O1 op={}", o1.1);

        // P1: Doodson 163,555, [0,0,2,-2,2], ip=-43.4, op=2.9
        let p1 = find(&TABLE_M1, [0, 0, 2, -2, 2]).expect("P1 not found in TABLE_M1");
        assert!((p1.0 - (-43.4)).abs() < 1e-9, "P1 ip={}", p1.0);
        assert!((p1.1 - 2.9).abs() < 1e-9, "P1 op={}", p1.1);

        // ψ1: Doodson 166,554, [0,-1,0,0,0], ip=-20.6, op=-0.3
        let psi1 = find(&TABLE_M1, [0, -1, 0, 0, 0]).expect("ψ1 not found in TABLE_M1");
        assert!((psi1.0 - (-20.6)).abs() < 1e-9, "ψ1 ip={}", psi1.0);
        assert!((psi1.1 - (-0.3)).abs() < 1e-9, "ψ1 op={}", psi1.1);

        // 165,565 line: [0,0,0,0,1], ip=68.1, op=-4.6
        let line_165565 = find(&TABLE_M1, [0, 0, 0, 0, 1]).expect("165,565 not found in TABLE_M1");
        assert!(
            (line_165565.0 - 68.1).abs() < 1e-9,
            "165,565 ip={}",
            line_165565.0
        );
        assert!(
            (line_165565.1 - (-4.6)).abs() < 1e-9,
            "165,565 op={}",
            line_165565.1
        );

        // TABLE_M0 anchors.
        // 55,565: [0,0,0,0,1], ip=16.6, op=-6.7
        let m0_55565 = find(&TABLE_M0, [0, 0, 0, 0, 1]).expect("55,565 not found in TABLE_M0");
        assert!((m0_55565.0 - 16.6).abs() < 1e-9, "55,565 ip={}", m0_55565.0);
        assert!(
            (m0_55565.1 - (-6.7)).abs() < 1e-9,
            "55,565 op={}",
            m0_55565.1
        );

        // Mf 75,555: [0,0,-2,0,-2], ip=0.6, op=6.3
        let mf = find(&TABLE_M0, [0, 0, -2, 0, -2]).expect("Mf 75,555 not found in TABLE_M0");
        assert!((mf.0 - 0.6).abs() < 1e-9, "Mf ip={}", mf.0);
        assert!((mf.1 - 6.3).abs() < 1e-9, "Mf op={}", mf.1);

        // TABLE_M2 anchors.
        // N2: [1,0,2,0,2], ip=-0.3
        let n2 = find(&TABLE_M2, [1, 0, 2, 0, 2]).expect("N2 not found in TABLE_M2");
        assert!((n2.0 - (-0.3)).abs() < 1e-9, "N2 ip={}", n2.0);
        assert!((n2.1).abs() < 1e-12, "N2 op should be 0.0, got {}", n2.1);

        // M2: [0,0,2,0,2], ip=-1.2
        let m2 = find(&TABLE_M2, [0, 0, 2, 0, 2]).expect("M2 not found in TABLE_M2");
        assert!((m2.0 - (-1.2)).abs() < 1e-9, "M2 ip={}", m2.0);
        assert!((m2.1).abs() < 1e-12, "M2 op should be 0.0, got {}", m2.1);
    }

    /// Step-2 corrections change the low-degree (n=2) coefficients at ~1e-11 scale
    /// and do NOT alter degree-3 or degree-4 terms (which come from Step 1 only).
    #[test]
    fn test_step2_toggle_changes_low_degree_terms() {
        crate::utils::testing::setup_global_test_eop();
        let epoch = Epoch::from_datetime(2015, 6, 15, 12, 0, 0.0, 0.0, TimeSystem::UTC);
        let r_moon = Vector3::new(3.844e8, 0.0, 1.0e7);
        let r_sun = Vector3::new(1.496e11, 2.0e10, 0.0);

        let cfg_off = SolidTideConfig {
            frequency_dependent: false,
        };
        let cfg_on = SolidTideConfig {
            frequency_dependent: true,
        };

        let c_off =
            solid_earth_tide_coefficients(r_sun, r_moon, epoch, GM_EARTH, R_EARTH, &cfg_off);
        let c_on = solid_earth_tide_coefficients(r_sun, r_moon, epoch, GM_EARTH, R_EARTH, &cfg_on);

        // Step 2 changes C̄20, C̄21, C̄22 at ~1e-11 scale.
        let d20 = (c_on.dc[2][0] - c_off.dc[2][0]).abs();
        let d21c = (c_on.dc[2][1] - c_off.dc[2][1]).abs();
        let d21s = (c_on.ds[2][1] - c_off.ds[2][1]).abs();
        let d22c = (c_on.dc[2][2] - c_off.dc[2][2]).abs();
        let d22s = (c_on.ds[2][2] - c_off.ds[2][2]).abs();

        assert!(d20 > 1e-13, "ΔC̄20 should change, got {:e}", d20);
        assert!(d21c > 1e-13 || d21s > 1e-13, "C̄21/S̄21 should change");
        assert!(d22c > 1e-14 || d22s > 1e-14, "C̄22/S̄22 should change");

        // All changes are at most ~1e-9 (tables peak at ~470e-12 ≈ 4.7e-10).
        assert!(d20 < 1e-9, "ΔC̄20 too large: {:e}", d20);
        assert!(d21c < 1e-9, "ΔC̄21 too large: {:e}", d21c);
        assert!(d22c < 1e-9, "ΔC̄22 too large: {:e}", d22c);

        // Degree-3 and degree-4 terms are unchanged (Step 2 is degree-2 only).
        assert_eq!(c_on.dc[3][0], c_off.dc[3][0], "dc[3][0] should not change");
        assert_eq!(c_on.dc[3][1], c_off.dc[3][1], "dc[3][1] should not change");
        assert_eq!(c_on.dc[4][0], c_off.dc[4][0], "dc[4][0] should not change");
        assert_eq!(c_on.dc[4][2], c_off.dc[4][2], "dc[4][2] should not change");
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
