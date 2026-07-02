#![allow(missing_docs)]

//! Native SPICE implementation vs ANISE benchmarks.
//!
//! Cases (spec acceptance: native >= ANISE on every case):
//! 1. Single-epoch queries (Sun/Moon/Mars rel Earth), position-only and state,
//!    plus a direct-SPK variant that bypasses the global registry lock.
//! 2. Sequential propagation pattern: 10k epochs at 60 s steps.
//! 3. Third-body acceleration loop through accel_third_body.

use std::hint::black_box;
use std::path::PathBuf;

use criterion::Criterion;
use nalgebra::Vector3;

use anise::astro::Aberration;
use anise::prelude::{Almanac, Epoch as AniseEpoch, Frame, SPK as AniseSPK};

use brahe::orbit_dynamics::third_body::accel_third_body;
use brahe::propagators::force_model_config::{EphemerisSource, ThirdBody};
use brahe::spice::{self, NAIF_EARTH, NAIF_MARS_BARYCENTER, NAIF_MOON, NAIF_SUN, SPK};
use brahe::time::{Epoch, TimeSystem};
use brahe::utils::cache::get_naif_cache_dir;

fn kernel_path() -> PathBuf {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_assets")
        .join("de440s.bsp");
    assert!(p.exists(), "benchmark requires test_assets/de440s.bsp");
    p
}

fn setup_native() {
    // Seed the NAIF cache from the test asset (so `load_kernel("de440s")`
    // resolves locally) and register it in the global kernel registry.
    let cache_dir = get_naif_cache_dir().unwrap();
    let cache_path = PathBuf::from(cache_dir).join("de440s.bsp");
    if !cache_path.exists() {
        std::fs::copy(kernel_path(), &cache_path).unwrap();
    }
    spice::load_kernel("de440s").unwrap();
}

fn setup_anise() -> Almanac {
    let spk = AniseSPK::load(kernel_path().to_str().unwrap()).unwrap();
    Almanac::from_spk(spk)
}

fn bench_single_query(c: &mut Criterion) {
    setup_native();
    let almanac = setup_anise();
    let direct_spk = SPK::from_file(&kernel_path()).unwrap();
    let epc = Epoch::from_date(2025, 6, 1, TimeSystem::UTC);
    let et = epc.seconds_past_j2000_as_time_system(TimeSystem::TDB);

    let mut group = c.benchmark_group("spice_single_query");
    for (name, target) in [
        ("sun", NAIF_SUN),
        ("moon", NAIF_MOON),
        ("mars", NAIF_MARS_BARYCENTER),
    ] {
        group.bench_function(format!("native_position_{name}"), |b| {
            b.iter(|| spice::spk_position(black_box(target), NAIF_EARTH, black_box(epc)).unwrap())
        });
        group.bench_function(format!("native_state_{name}"), |b| {
            b.iter(|| spice::spk_state(black_box(target), NAIF_EARTH, black_box(epc)).unwrap())
        });
        group.bench_function(format!("native_direct_position_{name}"), |b| {
            b.iter(|| {
                direct_spk
                    .position(black_box(target), NAIF_EARTH, black_box(et))
                    .unwrap()
            })
        });
        group.bench_function(format!("anise_state_{name}"), |b| {
            let target_frame = Frame::from_ephem_j2000(target);
            let center_frame = Frame::from_ephem_j2000(NAIF_EARTH);
            let anise_epoch = AniseEpoch::from_et_seconds(et);
            b.iter(|| {
                almanac
                    .translate(
                        black_box(target_frame),
                        center_frame,
                        anise_epoch,
                        Aberration::NONE,
                    )
                    .unwrap()
            })
        });
    }
    group.finish();
}

fn bench_sequential_pattern(c: &mut Criterion) {
    setup_native();
    let almanac = setup_anise();
    let epc0 = Epoch::from_date(2025, 1, 1, TimeSystem::UTC);
    let et0 = epc0.seconds_past_j2000_as_time_system(TimeSystem::TDB);
    const N: usize = 10_000;
    const DT: f64 = 60.0;

    let mut group = c.benchmark_group("spice_sequential_10k");
    group.sample_size(10);
    group.bench_function("native_position", |b| {
        b.iter(|| {
            let mut acc = Vector3::zeros();
            for i in 0..N {
                let epc = epc0 + (i as f64) * DT;
                acc += spice::spk_position(NAIF_SUN, NAIF_EARTH, epc).unwrap();
            }
            black_box(acc)
        })
    });
    group.bench_function("native_state", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for i in 0..N {
                let epc = epc0 + (i as f64) * DT;
                acc += spice::spk_state(NAIF_SUN, NAIF_EARTH, epc).unwrap().norm();
            }
            black_box(acc)
        })
    });
    group.bench_function("anise_state", |b| {
        let target = Frame::from_ephem_j2000(NAIF_SUN);
        let center = Frame::from_ephem_j2000(NAIF_EARTH);
        b.iter(|| {
            let mut acc = 0.0;
            for i in 0..N {
                let anise_epoch = AniseEpoch::from_et_seconds(et0 + (i as f64) * DT);
                let s = almanac
                    .translate(target, center, anise_epoch, Aberration::NONE)
                    .unwrap();
                acc += s.radius_km.norm();
            }
            black_box(acc)
        })
    });
    group.finish();
}

fn bench_third_body_accel(c: &mut Criterion) {
    setup_native();
    let epc = Epoch::from_date(2025, 6, 1, TimeSystem::UTC);
    let r_sat = Vector3::new(7.0e6, 0.0, 0.0);

    let mut group = c.benchmark_group("spice_third_body");
    group.bench_function("accel_sun_moon_de440s", |b| {
        b.iter(|| {
            let a_sun = accel_third_body(
                black_box(ThirdBody::Sun),
                EphemerisSource::DE440s,
                black_box(epc),
                black_box(r_sat),
            );
            let a_moon = accel_third_body(
                black_box(ThirdBody::Moon),
                EphemerisSource::DE440s,
                black_box(epc),
                black_box(r_sat),
            );
            black_box((a_sun, a_moon))
        })
    });
    group.finish();
}

fn main() {
    let mut c = Criterion::default().configure_from_args();
    bench_single_query(&mut c);
    bench_sequential_pattern(&mut c);
    bench_third_body_accel(&mut c);
    c.final_summary();
}
