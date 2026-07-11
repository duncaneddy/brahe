//! Validation of the native SPICE implementation against ANISE.
//!
//! ANISE is a dev-dependency used purely as an independent oracle. All
//! comparisons are performed at matched ephemeris time (ET) so kernel
//! evaluation is isolated from time-conversion differences; a separate
//! end-to-end test measures the epoch-conversion difference.

use anise::astro::Aberration;
use anise::math::Vector3 as AniseVector3;
use anise::prelude::{Almanac, Epoch as AniseEpoch, Frame, SPK as AniseSPK};
use approx::assert_abs_diff_eq;
use std::path::PathBuf;

use crate::spice::naif_id::NAIFId;
use crate::spice::spk::SPK;
use crate::time::{Epoch, TimeSystem};

const BODIES: &[(i32, &str)] = &[
    (1, "Mercury barycenter"),
    (2, "Venus barycenter"),
    (4, "Mars barycenter"),
    (5, "Jupiter barycenter"),
    (6, "Saturn barycenter"),
    (7, "Uranus barycenter"),
    (8, "Neptune barycenter"),
    (9, "Pluto barycenter"),
    (10, "Sun"),
    (199, "Mercury"),
    (299, "Venus"),
    (301, "Moon"),
];

fn de440s_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_assets")
        .join("de440s.bsp");
    p.exists().then_some(p)
}

fn load_both() -> Option<(SPK, Almanac)> {
    let path = de440s_path()?;
    let native = SPK::from_file(&path).unwrap();
    let anise_spk = AniseSPK::load(path.to_str().unwrap()).unwrap();
    let almanac = Almanac::from_spk(anise_spk);
    Some((native, almanac))
}

/// ~100 ETs spanning DE440s' actual coverage (verified segment bounds:
/// [-4_734_072_000, 4_735_368_000] s past J2000 TDB), staying 1 year
/// inside each end.
fn sample_ets() -> Vec<f64> {
    let start = -4_734_072_000.0_f64 + 3.2e7;
    let end = 4_735_368_000.0_f64 - 3.2e7;
    let n = 100;
    (0..n)
        .map(|i| start + (end - start) * i as f64 / (n - 1) as f64)
        .collect()
}

fn anise_state_km(
    almanac: &Almanac,
    target: i32,
    center: i32,
    et: f64,
) -> (AniseVector3, AniseVector3) {
    let state = almanac
        .translate(
            Frame::from_ephem_j2000(target),
            Frame::from_ephem_j2000(center),
            AniseEpoch::from_et_seconds(et),
            Aberration::NONE,
        )
        .unwrap();
    (state.radius_km, state.velocity_km_s)
}

#[test]
fn test_validation_position_velocity_vs_anise_matched_et() {
    let Some((native, almanac)) = load_both() else {
        return;
    };

    let mut max_dr = 0.0_f64;
    let mut max_dv = 0.0_f64;

    for &(body, name) in BODIES {
        for center in [NAIFId::Earth.id(), NAIFId::SolarSystemBarycenter.id()] {
            for &et in &sample_ets() {
                let r_native = native.position(body, center, et).unwrap();
                let v_native = native.velocity(body, center, et).unwrap();
                let (r_km, v_km) = anise_state_km(&almanac, body, center, et);

                // Component-wise comparison: 1 mm / 1e-6 m/s absolute,
                // floored at a few ULPs of the component magnitude. At
                // outer-body distances (e.g. Pluto barycenter, ~7e12 m) 1 mm
                // is below f64 resolution (1 ULP at 7e12 m is ~9.8e-4 m;
                // eps·|x| ~ 1.6e-3 m is ~1.6 ULP), so an absolute-only bound
                // would demand agreement below machine precision and pass or
                // fail on toolchain/FMA summation order alone. For all inner
                // bodies the ULP floor is far below 1e-3/1e-6 and the
                // effective tolerance is unchanged.
                for i in 0..3 {
                    let r_expected = r_km[i] * 1.0e3;
                    let v_expected = v_km[i] * 1.0e3;
                    let dr = (r_native[i] - r_expected).abs();
                    let dv = (v_native[i] - v_expected).abs();
                    max_dr = max_dr.max(dr);
                    max_dv = max_dv.max(dv);

                    let tol_r = (1.0e-3_f64).max(4.0 * f64::EPSILON * r_expected.abs());
                    let tol_v = (1.0e-6_f64).max(4.0 * f64::EPSILON * v_expected.abs());
                    assert!(
                        dr < tol_r,
                        "{} (body {}) center {} et {} axis {}: |Δr| = {} m, tol = {} m",
                        name,
                        body,
                        center,
                        et,
                        i,
                        dr,
                        tol_r
                    );
                    assert!(
                        dv < tol_v,
                        "{} (body {}) center {} et {} axis {}: |Δv| = {} m/s, tol = {} m/s",
                        name,
                        body,
                        center,
                        et,
                        i,
                        dv,
                        tol_v
                    );
                }
            }
        }
    }

    println!(
        "matched-ET sweep: max |dr| = {:.3e} m, max |dv| = {:.3e} m/s",
        max_dr, max_dv
    );
}

#[test]
fn test_validation_et_conversion_vs_anise() {
    // Our Epoch -> ET vs hifitime's UTC -> ET. Both apply UTC leap seconds
    // identically, but the TT->TDB periodic term uses two different
    // analytic approximations: brahe's `tdb_tt_offset` (a 7-term truncated
    // Fairhead & Bretagnon 1990 series, matching SOFA's iauDtdb to ~microseconds)
    // vs hifitime's single-sinusoid formula (`Epoch::inner_g`, the classic
    // ~30 µs-accurate approximation). Neither is exact; measured disagreement
    // between the two models is ~20 µs, so require agreement to < 30 µs.
    let epc = Epoch::from_datetime(2025, 3, 15, 6, 30, 21.0, 0.0, TimeSystem::UTC);
    let et_native = epc.seconds_past_j2000_as_time_system(TimeSystem::TDB);
    let et_anise = AniseEpoch::from_gregorian_utc(2025, 3, 15, 6, 30, 21, 0).to_et_seconds();
    assert_abs_diff_eq!(et_native, et_anise, epsilon = 3.0e-5);
}

#[test]
fn test_validation_end_to_end_epoch_path() {
    // Same UTC epoch through both full stacks (native: Epoch -> ET -> SPK;
    // ANISE: gregorian UTC -> hifitime -> SPK). Differences are bounded by
    // the ET model disagreement (< 30 µs, see
    // test_validation_et_conversion_vs_anise) times relative body/Earth
    // velocity (< 40 km/s) => < 1.2 m; measured max is 0.81 m (Saturn
    // barycenter).
    let Some((native, almanac)) = load_both() else {
        return;
    };

    let epc = Epoch::from_datetime(2025, 3, 15, 6, 30, 21.0, 0.0, TimeSystem::UTC);
    let et = epc.seconds_past_j2000_as_time_system(TimeSystem::TDB);
    let anise_epoch = AniseEpoch::from_gregorian_utc(2025, 3, 15, 6, 30, 21, 0);

    let mut max_dr = 0.0_f64;
    for &(body, _) in BODIES {
        let r_native = native.position(body, NAIFId::Earth.id(), et).unwrap();
        let state = almanac
            .translate(
                Frame::from_ephem_j2000(body),
                Frame::from_ephem_j2000(NAIFId::Earth.id()),
                anise_epoch,
                Aberration::NONE,
            )
            .unwrap();
        let dr = (r_native
            - AniseVector3::new(
                state.radius_km[0] * 1.0e3,
                state.radius_km[1] * 1.0e3,
                state.radius_km[2] * 1.0e3,
            ))
        .norm();
        max_dr = max_dr.max(dr);
        assert!(dr < 1.2, "body {} end-to-end |Δr| = {} m", body, dr);
    }
    println!("end-to-end epoch path: max |dr| = {:.3} m", max_dr);
}

#[test]
fn test_validation_frame_bias_removal_documented() {
    // Documents the effect of dropping the J2000->ICRF bias rotation that
    // the pre-native implementation applied: |R_bias·r - r| for the Sun is
    // ~asin(23 mas)·1 AU ≈ 1.7e4 m — i.e. the OLD outputs differed from
    // raw kernel output by this much. The angle equivalent (~23 mas) is far
    // below the 0.1 deg tolerance used by the sun/moon direction tests.
    use crate::constants::AS2RAD;
    use nalgebra::Matrix3;

    let Some((native, _)) = load_both() else {
        return;
    };
    let r = native.position(10, NAIFId::Earth.id(), 0.0).unwrap();

    // The bias matrix formerly in positions.rs (IERS 2010 frame bias)
    let dxi = -16.6170e-3 * AS2RAD;
    let deta = -6.8192e-3 * AS2RAD;
    let dalpha = -14.6e-3 * AS2RAD;
    let b = Matrix3::new(
        1.0 - 0.5 * (dxi * dxi + deta * deta),
        dalpha,
        -dxi,
        -dalpha - dxi * deta,
        1.0 - 0.5 * (dalpha * dalpha + deta * deta),
        -deta,
        dxi + dalpha * deta,
        deta + dalpha * dxi,
        1.0 - 0.5 * (deta * deta + dxi * dxi),
    )
    .transpose();

    let delta = (b * r - r).norm();
    // ~1.2-2.2e4 m at 1 AU; assert the order of magnitude so the doc claim stays honest
    assert!(delta > 1.0e3 && delta < 1.0e5, "bias delta = {} m", delta);
}

#[test]
#[cfg_attr(not(feature = "integration"), ignore)]
fn test_validation_pck_moon_pa_vs_anise() {
    use crate::datasets::naif::download_pck_kernel;
    use crate::spice::pck::BPCK;

    let path = download_pck_kernel("moon_pa_de440", None).unwrap();
    let bpck = BPCK::from_file(&path).unwrap();

    // The real moon_pa_de440_200625.bpc kernel stores frame class ID 31008
    // (MOON_PA_DE440); 31006 is the older MOON_PA_DE403 frame and is absent
    // from this file. Confirmed by parsing the downloaded kernel's DAF
    // summaries directly (ints[0] = 31008 for both segments).
    //
    // ANISE side: load the BPC and request the ICRF (J2000) -> MOON_PA
    // (frame class 31008) DCM directly via `Almanac::rotation_to_parent`,
    // which is the single-hop primitive `Almanac::rotate` composes from.
    // ANISE's `rotation_to_parent` evaluates the same PCK type-2 Chebyshev
    // triplet (ra, dec, twist) as brahe's `BPCK::euler_angles` and builds
    // `DCM { rot_mat: r3(twist) * r1(dec) * r3(ra), from: <inertial frame
    // read from the segment summary>, to: <source.orientation_id> }`.
    // Both ANISE's `r1`/`r3` and brahe's `RotationMatrix::Rx`/`Rz` are the
    // identical elementwise-defined active rotation matrices, so this is
    // directly the ICRF -> MOON_PA DCM with the same phi/delta/w -> ra/dec/
    // twist angle correspondence as brahe's `BPCK::rotation_matrix`; no
    // transpose is needed on either side.
    let almanac = Almanac::default().load(path.to_str().unwrap()).unwrap();
    let moon_pa_frame = Frame::from_ephem_j2000(301).with_orient(31008);

    let mut max_delta = 0.0_f64;
    for &et in &[0.0_f64, 1.0e8, 5.0e8, -1.0e8] {
        let r_native = bpck.rotation_matrix(31008, et).unwrap();
        let epoch = AniseEpoch::from_et_seconds(et);
        let dcm = almanac.rotation_to_parent(moon_pa_frame, epoch).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let delta = (r_native[(i, j)] - dcm.rot_mat[(i, j)]).abs();
                max_delta = max_delta.max(delta);
                assert_abs_diff_eq!(r_native[(i, j)], dcm.rot_mat[(i, j)], epsilon = 1.0e-9);
            }
        }
    }
    println!("MOON_PA DCM vs ANISE: max |delta| = {:.3e}", max_delta);
}
