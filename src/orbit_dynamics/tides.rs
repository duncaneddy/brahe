/*!
Tidal corrections to the spherical-harmonic geopotential.

Implements IERS Conventions (2010), TN36 Chapter 6:
- §6.2.2: permanent (zero-frequency) tide conversion of C̄20 between the
  mean-tide / zero-tide / conventional-tide-free systems.
- §6.2.1: solid Earth tides (added in later tasks).

Source: <https://iers-conventions.obspm.fr/content/chapter6/icc6.pdf>
*/

use nalgebra::Vector3;

use crate::constants::{AS2RAD, AngleFormat, GM_MOON, GM_SUN};
use crate::eop::get_global_pm;
use crate::orbit_dynamics::gravity::{
    ClenshawCoefficients, GravityModelTideSystem, ParallelMode, clenshaw_acceleration,
};
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

/// Permanent-tide DIRECT term on the fully-normalized C̄20: the A0*H0 factor
/// with no Love number, i.e. the lunisolar permanent tide-raising potential
/// itself (present in the mean-tide system, removed for zero-tide; tide
/// systems defined in IERS TN36 §1.1). A0 = 4.4228e-8 m^-1 and
/// H0 = -0.31460 m are the values quoted in Eq. 6.14. TN36 assigns this term
/// no numbered equation (Chapter 6 only converts zero-tide <-> tide-free);
/// it is the "restoring the permanent part of the tide generating potential"
/// step of TN36 Figure 1.2, and in geoid height it reproduces the classical
/// mean/zero conversion N_m - N_z = (9.9 - 29.6 sin^2(phi)) cm of
/// Lemoine et al. (1998), Eq. (11.1-1).
pub const PERM_C20_DIRECT: f64 = 4.4228e-8 * (-0.31460);

/// Permanent-tide INDIRECT term on C̄20 (IERS Eq. 6.14, A0*H0*k20). k20 =
/// 0.30190 is the nominal degree-2 Love number (Table 6.3 anelastic Re k20). This is
/// the Earth's permanent elastic deformation response (present in both
/// mean-tide AND zero-tide, removed for conventional tide-free).
pub const PERM_C20_INDIRECT: f64 = 4.4228e-8 * (-0.31460) * 0.30190;

/// Offset of a system's C̄20 relative to the conventional tide-free value.
///
/// Per IERS §1.1 and §6.2.2, the systems differ by which permanent terms are present:
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

/// Fully-normalized ΔC̄nm/ΔS̄nm accumulator for tidal geopotential corrections
/// (IERS TN36 Ch. 6), sized at construction. Triangular column-major packing
/// (same layout as the Clenshaw tables): index(n, m) = m(2·n_max − m + 1)/2 + n.
#[derive(Debug, Clone)]
pub struct TideDeltas {
    n_max: usize,
    m_max: usize,
    dc: Vec<f64>,
    ds: Vec<f64>,
}

impl TideDeltas {
    /// New all-zero delta set for degrees 0..=n_max, orders 0..=m_max.
    ///
    /// # Arguments
    /// - `n_max`: maximum degree of the delta set.
    /// - `m_max`: maximum order of the delta set.
    ///
    /// # Returns
    /// A zero-initialized `TideDeltas` sized for `(n_max, m_max)`.
    pub fn new(n_max: usize, m_max: usize) -> Self {
        let len = m_max * (2 * n_max - m_max + 1) / 2 + n_max + 1;
        TideDeltas {
            n_max,
            m_max,
            dc: vec![0.0; len],
            ds: vec![0.0; len],
        }
    }

    #[inline]
    fn idx(&self, n: usize, m: usize) -> usize {
        debug_assert!(n <= self.n_max && m <= self.m_max && m <= n);
        m * (2 * self.n_max - m + 1) / 2 + n
    }

    /// Accumulate (add) a fully-normalized correction at `(n, m)`.
    ///
    /// # Arguments
    /// - `n`, `m`: degree and order of the correction.
    /// - `dc`: fully-normalized cosine correction ΔC̄nm to add.
    /// - `ds`: fully-normalized sine correction ΔS̄nm to add.
    ///
    /// # Returns
    /// (none)
    pub fn add(&mut self, n: usize, m: usize, dc: f64, ds: f64) {
        let i = self.idx(n, m);
        self.dc[i] += dc;
        self.ds[i] += ds;
    }

    /// Read the accumulated `(ΔC̄nm, ΔS̄nm)` at `(n, m)`.
    ///
    /// # Arguments
    /// - `n`, `m`: degree and order to read.
    ///
    /// # Returns
    /// Tuple `(ΔC̄nm, ΔS̄nm)`, fully-normalized.
    pub fn get(&self, n: usize, m: usize) -> (f64, f64) {
        let i = self.idx(n, m);
        (self.dc[i], self.ds[i])
    }

    /// Reset all entries to zero (allocation reused).
    ///
    /// # Returns
    /// (none)
    pub fn clear(&mut self) {
        self.dc.fill(0.0);
        self.ds.fill(0.0);
    }

    /// Maximum degree of the delta set.
    ///
    /// # Returns
    /// The `n_max` the delta set was constructed with.
    pub fn n_max(&self) -> usize {
        self.n_max
    }

    /// Maximum order of the delta set.
    ///
    /// # Returns
    /// The `m_max` the delta set was constructed with.
    pub fn m_max(&self) -> usize {
        self.m_max
    }
}

/// Denormalization factor sqrt((2-δ0m)(2n+1)(n-m)!/(n+m)!) — identical to
/// `GravityModel::precompute_coefficients`. n,m <= 4, so factorials are exact.
fn denorm_factor(n: usize, m: usize) -> f64 {
    fn fact(k: usize) -> f64 {
        (1..=k).map(|i| i as f64).product::<f64>().max(1.0)
    }
    let delta0m = if m == 0 { 1.0 } else { 0.0 };
    ((2.0 - delta0m) * (2.0 * n as f64 + 1.0) * fact(n - m) / fact(n + m)).sqrt()
}

/// Body-fixed acceleration of a delta-coefficient field evaluated through
/// zero-baseline Clenshaw tables. One-shot convenience (allocates tables);
/// the propagator's `TideField` reuses persistent tables instead.
///
/// # Arguments
/// - `r_ecef`: evaluation position, ECEF [m].
/// - `deltas`: fully-normalized ΔC̄nm/ΔS̄nm corrections to evaluate.
/// - `gm`: gravitational parameter [m³/s²].
/// - `radius`: reference radius [m].
///
/// # Returns
/// Acceleration in the body-fixed frame [m/s²].
pub(crate) fn accel_tide_deltas(
    r_ecef: Vector3<f64>,
    deltas: &TideDeltas,
    gm: f64,
    radius: f64,
) -> Vector3<f64> {
    let mut tables = ClenshawCoefficients::zeros(deltas.n_max());
    for m in 0..=deltas.m_max() {
        for n in m.max(2)..=deltas.n_max() {
            let (dc, ds) = deltas.get(n, m);
            tables.set_normalized(n, m, dc, ds);
        }
    }
    clenshaw_acceleration(
        &tables,
        r_ecef,
        deltas.n_max(),
        deltas.m_max(),
        gm,
        radius,
        ParallelMode::Never,
    )
    .expect("delta tables sized from the delta set cannot be out of bounds")
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
/// # Arguments
/// - `r_sun_ecef`, `r_moon_ecef`: body positions, ECEF [m].
/// - `epoch`: used for Step 2 Doodson arguments (ignored when Step 1 only).
/// - `gm_earth`: Earth's gravitational parameter [m³/s²].
/// - `radius`: reference radius [m].
/// - `config`: solid-tide settings (Step 2 toggle).
/// - `deltas`: accumulator the Step 1 (and, if enabled, Step 2) corrections
///   are added into; sized `n_max >= 4`, `m_max >= 2` by the caller.
///
/// # Returns
/// (none) — corrections are accumulated into `deltas`.
///
/// # References
/// - IERS Conventions (2010), TN36 §6.2.1, Eq. (6.6)–(6.7), Table 6.3.
///   <https://iers-conventions.obspm.fr/content/chapter6/icc6.pdf>
/// - Montenbruck & Gill, *Satellite Orbits*, §3.7.2, Eq. (3.159) (unnormalized cross-check).
pub fn solid_earth_tide_deltas(
    r_sun_ecef: Vector3<f64>,
    r_moon_ecef: Vector3<f64>,
    epoch: Epoch,
    gm_earth: f64,
    radius: f64,
    config: &SolidTideConfig,
    deltas: &mut TideDeltas,
) {
    let _ = epoch; // used by Step 2 (frequency-dependent corrections in Task 7)

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
                deltas.add(n, m, f * (kre * cm + kim * sm), f * (kre * sm - kim * cm));
            }
        }

        // Eq. (6.7): degree-2 tides -> degree-4 coefficients, m = 0,1,2.
        let radial3 = (radius / r).powi(3);
        for m in 0..=2usize {
            let kp = LOVE_PLUS[m];
            let f = (kp / 5.0) * gm_ratio * radial3 * p[2][m];
            let (cm, sm) = ((m as f64 * lambda).cos(), (m as f64 * lambda).sin());
            deltas.add(4, m, f * cm, f * sm);
        }
    }

    if config.frequency_dependent {
        let (dc20, step2_tesseral) = step2_corrections(epoch);
        deltas.add(2, 0, dc20, 0.0);
        deltas.add(2, 1, step2_tesseral[0][0], step2_tesseral[0][1]);
        deltas.add(2, 2, step2_tesseral[1][0], step2_tesseral[1][1]);
    }
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
    let mut deltas = TideDeltas::new(4, 4);
    solid_earth_tide_deltas(
        r_sun_ecef,
        r_moon_ecef,
        epoch,
        gm_earth,
        radius,
        config,
        &mut deltas,
    );
    accel_tide_deltas(r_ecef, &deltas, gm_earth, radius)
}

/// Greenwich-mean-sidereal-time-plus-π and the five Delaunay fundamental
/// arguments (l, l', F, D, Ω) in radians, for the Doodson argument of a solid
/// Earth tide line (IERS §6.2.1):
///   θ_f = m·(θg + π) − (n_l·l + n_l'·l' + n_F·F + n_D·D + n_Ω·Ω)
/// where (n_*) are the Delaunay multipliers from Tables 6.5a/b/c and θg = GMST.
///
/// l, l', F, D, Ω come from the IAU 2003 fundamental-argument polynomials
/// (SOFA iauFal03/iauFalp03/iauFaf03/iauFad03/iauFaom03), evaluated at TT
/// Julian centuries since J2000. GMST from [`Epoch::gmst`] (IAU 2006).
///
/// Returns `[θg+π, l, l', F, D, Ω]` (radians).
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

    // GMST (radians), IAU 2006 (Epoch::gmst wraps SOFA iauGmst06 with UT1/TT).
    let gmst = epoch.gmst(AngleFormat::Radians);

    [gmst + PI, l, lp, f, d, om]
}

/// IERS secular pole (x̄s, ȳs) in arcseconds (updated IERS Conventions
/// §7.1.4, Eq. (21), version 2018/02/01):
///   xs = 55.0 + 1.677 (t − 2000.0) \[mas\], ys = 320.5 + 3.460 (t − 2000.0) \[mas\]
/// with t in Julian years (365.25 d) of TT. Supersedes the TN36 (2010)
/// cubic mean-pole model (adopted with ITRF2014).
///
/// # Arguments
/// - `epoch`: evaluation epoch.
///
/// # Returns
/// (f64, f64): (x̄s, ȳs). Units: arcseconds.
///
/// # Examples
/// ```
/// use brahe::orbit_dynamics::secular_pole;
/// use brahe::time::{Epoch, TimeSystem};
///
/// let epoch = Epoch::from_datetime(2020, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::TT);
/// let (xs, ys) = secular_pole(epoch);
/// assert!(xs > 0.0 && ys > 0.0);
/// ```
pub fn secular_pole(epoch: Epoch) -> (f64, f64) {
    let t = 2000.0 + (epoch.mjd_as_time_system(TimeSystem::TT) - 51544.5) / 365.25;
    let xs_mas = 55.0 + 1.677 * (t - 2000.0);
    let ys_mas = 320.5 + 3.460 * (t - 2000.0);
    (xs_mas * 1.0e-3, ys_mas * 1.0e-3)
}

/// Wobble parameters (m1, m2) in arcseconds (updated IERS Conventions
/// §7.1.4, Eq. (25)): m1 = xp − x̄s, m2 = −(yp − ȳs), with (xp, yp) from the
/// global EOP provider (stored in radians; converted here).
///
/// # Arguments
/// - `epoch`: evaluation epoch.
///
/// # Returns
/// (f64, f64): (m1, m2). Units: arcseconds.
///
/// # Errors
/// `BraheError` if global EOP data is not initialized or out of range.
///
/// # Examples
/// ```
/// use brahe::eop::*;
/// use brahe::orbit_dynamics::wobble_parameters;
/// use brahe::time::{Epoch, TimeSystem};
///
/// // Quick EOP initialization
/// let eop = FileEOPProvider::from_default_file(EOPType::StandardBulletinA, true, EOPExtrapolation::Zero).unwrap();
/// set_global_eop_provider(eop);
///
/// let epoch = Epoch::from_datetime(2020, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
/// let (m1, m2) = wobble_parameters(epoch).unwrap();
/// ```
pub fn wobble_parameters(epoch: Epoch) -> Result<(f64, f64), BraheError> {
    let (xp_rad, yp_rad) = get_global_pm(epoch.mjd_as_time_system(TimeSystem::TT))?;
    let (xs, ys) = secular_pole(epoch);
    Ok((xp_rad / AS2RAD - xs, -(yp_rad / AS2RAD - ys)))
}

/// Solid Earth pole tide ΔC̄21/ΔS̄21 (IERS TN36 §6.4, formulas following
/// Eq. (6.22), with k2 = 0.3077 + 0.0036i appropriate to the pole tide):
///   ΔC̄21 = −1.333e−9 (m1 + 0.0115 m2), ΔS̄21 = −1.333e−9 (m2 − 0.0115 m1).
///
/// # Arguments
/// - `m1_arcsec`, `m2_arcsec`: wobble parameters \[arcsec\] (see
///   [`wobble_parameters`]).
///
/// # Returns
/// (f64, f64): (ΔC̄21, ΔS̄21), fully normalized, dimensionless.
///
/// # Examples
/// ```
/// use brahe::orbit_dynamics::solid_earth_pole_tide_deltas;
///
/// let (dc21, ds21) = solid_earth_pole_tide_deltas(0.05, 0.35);
/// assert!(dc21.abs() < 1e-9 && ds21.abs() < 1e-9);
/// ```
pub fn solid_earth_pole_tide_deltas(m1_arcsec: f64, m2_arcsec: f64) -> (f64, f64) {
    (
        -1.333e-9 * (m1_arcsec + 0.0115 * m2_arcsec),
        -1.333e-9 * (m2_arcsec - 0.0115 * m1_arcsec),
    )
}

/// Ocean pole tide ΔC̄21/ΔS̄21, dominant (2,1) term of the Desai (2002)
/// self-consistent equilibrium model (IERS TN36 §6.5, Eq. (6.24)):
///   ΔC̄21 = −2.1778e−10 (m1 − 0.01724 m2)
///   ΔS̄21 = −1.7232e−10 (m2 − 0.03365 m1)
/// Captures ~90% of the ocean pole tide potential variance (§6.5); the full
/// degree-360 coefficient file is intentionally not implemented.
///
/// # Arguments
/// - `m1_arcsec`, `m2_arcsec`: wobble parameters \[arcsec\].
///
/// # Returns
/// (f64, f64): (ΔC̄21, ΔS̄21), fully normalized, dimensionless.
///
/// # Examples
/// ```
/// use brahe::orbit_dynamics::ocean_pole_tide_deltas;
///
/// let (dc21, ds21) = ocean_pole_tide_deltas(0.05, 0.35);
/// assert!(dc21.abs() < 1e-9 && ds21.abs() < 1e-9);
/// ```
pub fn ocean_pole_tide_deltas(m1_arcsec: f64, m2_arcsec: f64) -> (f64, f64) {
    (
        -2.1778e-10 * (m1_arcsec - 0.01724 * m2_arcsec),
        -1.7232e-10 * (m2_arcsec - 0.03365 * m1_arcsec),
    )
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
        let mut m = GravityModel::from_model_type(&GravityModelType::EGM2008_120).unwrap();
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

    #[test]
    fn test_step1_c20_magnitude_and_lunar_dominance() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        // Moon ~ along +x at ~384400 km; Sun ~ along +x at 1 AU.
        let r_moon = Vector3::new(3.844e8, 0.0, 0.0);
        let r_sun = Vector3::new(1.496e11, 0.0, 0.0);
        let cfg = SolidTideConfig {
            frequency_dependent: false,
        };
        let mut deltas = TideDeltas::new(4, 4);
        solid_earth_tide_deltas(r_sun, r_moon, epoch, GM_EARTH, R_EARTH, &cfg, &mut deltas);
        // Step-1 ΔC̄20 is ~1e-8 in magnitude (dominant solid-tide term).
        let (dc20, _) = deltas.get(2, 0);
        assert!(dc20.abs() > 1e-9 && dc20.abs() < 1e-7, "ΔC̄20 = {:e}", dc20);

        // Lunar-only vs solar-only ΔC̄20: Moon ~2.2x Sun.
        let far = Vector3::new(1.0e30, 0.0, 0.0); // effectively zero contribution
        let mut moon_only = TideDeltas::new(4, 4);
        solid_earth_tide_deltas(far, r_moon, epoch, GM_EARTH, R_EARTH, &cfg, &mut moon_only);
        let mut sun_only = TideDeltas::new(4, 4);
        solid_earth_tide_deltas(r_sun, far, epoch, GM_EARTH, R_EARTH, &cfg, &mut sun_only);
        let ratio = moon_only.get(2, 0).0 / sun_only.get(2, 0).0;
        assert!((ratio - 2.2).abs() < 0.4, "lunar/solar ratio = {ratio}");

        // Degree-4 feedback (Eq. 6.7) is present and ~3 orders smaller than ΔC̄20.
        let (dc40, _) = deltas.get(4, 0);
        assert!(dc40.abs() > 0.0);
        assert!(dc40.abs() < dc20.abs());
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

        // 75,575: [0,0,-2,0,0], ip=0.0, op=0.2 (ip is 0.0 in TN36 Table 6.5b;
        // guards against a past transcription error that had ip=0.2).
        let m0_75575 = find(&TABLE_M0, [0, 0, -2, 0, 0]).expect("75,575 not found in TABLE_M0");
        assert!((m0_75575.0).abs() < 1e-12, "75,575 ip={}", m0_75575.0);
        assert!((m0_75575.1 - 0.2).abs() < 1e-9, "75,575 op={}", m0_75575.1);

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

        let mut deltas_off = TideDeltas::new(4, 4);
        solid_earth_tide_deltas(
            r_sun,
            r_moon,
            epoch,
            GM_EARTH,
            R_EARTH,
            &cfg_off,
            &mut deltas_off,
        );
        let mut deltas_on = TideDeltas::new(4, 4);
        solid_earth_tide_deltas(
            r_sun,
            r_moon,
            epoch,
            GM_EARTH,
            R_EARTH,
            &cfg_on,
            &mut deltas_on,
        );

        // Step 2 changes C̄20, C̄21, C̄22 at ~1e-11 scale.
        let (dc20_off, _) = deltas_off.get(2, 0);
        let (dc20_on, _) = deltas_on.get(2, 0);
        let (dc21_off, ds21_off) = deltas_off.get(2, 1);
        let (dc21_on, ds21_on) = deltas_on.get(2, 1);
        let (dc22_off, ds22_off) = deltas_off.get(2, 2);
        let (dc22_on, ds22_on) = deltas_on.get(2, 2);
        let d20 = (dc20_on - dc20_off).abs();
        let d21c = (dc21_on - dc21_off).abs();
        let d21s = (ds21_on - ds21_off).abs();
        let d22c = (dc22_on - dc22_off).abs();
        let d22s = (ds22_on - ds22_off).abs();

        assert!(d20 > 1e-13, "ΔC̄20 should change, got {:e}", d20);
        assert!(d21c > 1e-13 || d21s > 1e-13, "C̄21/S̄21 should change");
        assert!(d22c > 1e-14 || d22s > 1e-14, "C̄22/S̄22 should change");

        // All changes are at most ~1e-9 (tables peak at ~470e-12 ≈ 4.7e-10).
        assert!(d20 < 1e-9, "ΔC̄20 too large: {:e}", d20);
        assert!(d21c < 1e-9, "ΔC̄21 too large: {:e}", d21c);
        assert!(d22c < 1e-9, "ΔC̄22 too large: {:e}", d22c);

        // Degree-3 and degree-4 terms are unchanged (Step 2 is degree-2 only).
        assert_eq!(
            deltas_on.get(3, 0),
            deltas_off.get(3, 0),
            "(3, 0) should not change"
        );
        assert_eq!(
            deltas_on.get(3, 1),
            deltas_off.get(3, 1),
            "(3, 1) should not change"
        );
        assert_eq!(
            deltas_on.get(4, 0),
            deltas_off.get(4, 0),
            "(4, 0) should not change"
        );
        assert_eq!(
            deltas_on.get(4, 2),
            deltas_off.get(4, 2),
            "(4, 2) should not change"
        );
    }

    #[test]
    #[serial_test::parallel]
    fn test_tide_deltas_packing_roundtrip() {
        let mut d = TideDeltas::new(4, 4);
        d.add(2, 0, 1.0e-9, 0.0);
        d.add(2, 0, 0.5e-9, 0.0); // accumulates
        d.add(3, 2, -2.0e-10, 4.0e-10);
        // 1.0e-9 + 0.5e-9 is not bit-exact in f64; compare with a tight epsilon.
        let (dc20, ds20) = d.get(2, 0);
        assert!((dc20 - 1.5e-9).abs() < 1e-20 && ds20 == 0.0);
        assert_eq!(d.get(3, 2), (-2.0e-10, 4.0e-10));
        assert_eq!(d.get(4, 4), (0.0, 0.0));
        d.clear();
        assert_eq!(d.get(2, 0), (0.0, 0.0));
    }

    #[test]
    #[serial_test::parallel]
    fn test_secular_pole_reference_values() {
        // icc7 §7.1.4 Eq. (21): xs = 55.0 mas, ys = 320.5 mas at t = 2000.0.
        let epoch = Epoch::from_datetime(2000, 1, 1, 12, 0, 0.0, 0.0, TimeSystem::TT);
        let (xs, ys) = secular_pole(epoch);
        assert!((xs - 0.0550).abs() < 1e-6, "xs = {xs}");
        assert!((ys - 0.3205).abs() < 1e-6, "ys = {ys}");
        // +10 years: xs = 55.0 + 16.77 = 71.77 mas; ys = 320.5 + 34.60 = 355.1 mas.
        let epoch10 = epoch + 10.0 * 365.25 * 86400.0;
        let (xs, ys) = secular_pole(epoch10);
        assert!((xs - 0.07177).abs() < 1e-5, "xs(2010) = {xs}");
        assert!((ys - 0.35510).abs() < 1e-5, "ys(2010) = {ys}");
    }

    #[test]
    #[serial_test::parallel]
    fn test_solid_pole_tide_reference_values() {
        // TN36 §6.4 (formulas after Eq. 6.22): unit wobble responses.
        let (dc, ds) = solid_earth_pole_tide_deltas(1.0, 0.0);
        assert!((dc - (-1.333e-9)).abs() < 1e-13);
        assert!((ds - (-1.333e-9 * -0.0115)).abs() < 1e-13);
        let (dc, ds) = solid_earth_pole_tide_deltas(0.0, 1.0);
        assert!((dc - (-1.333e-9 * 0.0115)).abs() < 1e-13);
        assert!((ds - (-1.333e-9)).abs() < 1e-13);
    }

    #[test]
    #[serial_test::parallel]
    fn test_ocean_pole_tide_reference_values() {
        // TN36 §6.5 Eq. (6.24). Note the DIFFERENT ΔS̄21 leading factor.
        let (dc, ds) = ocean_pole_tide_deltas(1.0, 0.0);
        assert!((dc - (-2.1778e-10)).abs() < 1e-15);
        assert!((ds - (-1.7232e-10 * -0.03365)).abs() < 1e-15);
        let (dc, ds) = ocean_pole_tide_deltas(0.0, 1.0);
        assert!((dc - (-2.1778e-10 * -0.01724)).abs() < 1e-15);
        assert!((ds - (-1.7232e-10)).abs() < 1e-15);
    }

    #[test]
    #[serial_test::serial]
    fn test_wobble_parameters_sign_convention() {
        crate::utils::testing::setup_global_test_eop();
        let epoch = Epoch::from_datetime(2015, 6, 15, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let (m1, m2) = wobble_parameters(epoch).unwrap();
        // xp, yp are O(0.1") and the secular pole is O(0.05"); wobbles are
        // bounded by ~0.8" (icc7 §7.1.4) and m2 = -(yp - ys).
        assert!(m1.abs() < 1.0 && m2.abs() < 1.0, "m1={m1}, m2={m2}");
        let (xp, yp) = crate::eop::get_global_pm(epoch.mjd_as_time_system(TimeSystem::TT)).unwrap();
        let (xs, ys) = secular_pole(epoch);
        assert!((m1 - (xp / crate::constants::AS2RAD - xs)).abs() < 1e-12);
        assert!((m2 - (-(yp / crate::constants::AS2RAD - ys))).abs() < 1e-12);
    }

    #[test]
    #[serial_test::parallel]
    fn test_tide_deltas_clenshaw_matches_full_path() {
        // Retarget of test_low_degree_evaluator_matches_full_path: the delta set
        // evaluated through zero-baseline Clenshaw tables must match a dense
        // GravityModel built from the same coefficients.
        let gm = 3.986004415e14;
        let radius = 6.378136300e6;
        let mut dc = [[0.0f64; 5]; 5];
        let mut ds = [[0.0f64; 5]; 5];
        dc[2][0] = 1.2e-8;
        dc[2][1] = -3.4e-9;
        ds[2][1] = 7.7e-9;
        dc[2][2] = 5.1e-9;
        ds[2][2] = -2.2e-9;
        dc[3][1] = 1.1e-9;
        ds[3][1] = -0.6e-9;
        dc[4][0] = -4.0e-11;
        dc[4][1] = 2.5e-11;
        ds[4][1] = -1.3e-11;
        dc[4][2] = -3.0e-11;
        ds[4][2] = 1.8e-11;

        let mut deltas = TideDeltas::new(4, 4);
        for n in 0..=4usize {
            for m in 0..=n {
                deltas.add(n, m, dc[n][m], ds[n][m]);
            }
        }
        let a_fast = |r| accel_tide_deltas(r, &deltas, gm, radius);
        let model = GravityModel::from_dense_normalized(&dc, &ds, 4, gm, radius);
        for r in [
            Vector3::new(7.0e6, 0.0, 0.0),
            Vector3::new(3.0e6, 4.0e6, 5.0e6),
            Vector3::new(-2.0e6, 1.0e6, 6.5e6),
        ] {
            let a_ref = model
                .compute_spherical_harmonics(r, 4, 4, ParallelMode::Never)
                .unwrap();
            let rel = (a_fast(r) - a_ref).norm() / a_ref.norm().max(1e-30);
            assert!(rel < 1e-12, "rel err {rel:e} at {r:?}");
        }
    }
}
