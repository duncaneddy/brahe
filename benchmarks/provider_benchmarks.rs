#![allow(missing_docs)]

use std::hint::black_box;

use criterion::Criterion;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

use brahe::eop::{EOPExtrapolation, EarthOrientationProvider, FileEOPProvider};
use brahe::space_weather::{FileSpaceWeatherProvider, SpaceWeatherProvider};

fn bench_eop_ut1_utc_repeated(c: &mut Criterion) {
    let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    let mjd = 59580.0; // 2022-01-01

    c.bench_function("eop_get_ut1_utc_repeated_same_mjd", |b| {
        b.iter(|| eop.get_ut1_utc(black_box(mjd)).unwrap())
    });
}

fn bench_eop_ut1_utc_sequential(c: &mut Criterion) {
    let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    let mjd_min = eop.mjd_min();
    let mjd_max = eop.mjd_max();
    let range = mjd_max - mjd_min;

    c.bench_function("eop_get_ut1_utc_sequential_scan", |b| {
        let mut i: u64 = 0;
        b.iter(|| {
            let frac = (i as f64) / 10000.0;
            let mjd = mjd_min + (frac % 1.0) * range;
            i += 1;
            eop.get_ut1_utc(black_box(mjd)).unwrap()
        })
    });
}

fn bench_eop_get_eop_repeated(c: &mut Criterion) {
    let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    let mjd = 59580.0;

    c.bench_function("eop_get_eop_repeated_same_mjd", |b| {
        b.iter(|| eop.get_eop(black_box(mjd)).unwrap())
    });
}

fn bench_eop_get_eop_sequential(c: &mut Criterion) {
    let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    let mjd_min = eop.mjd_min();
    let mjd_max = eop.mjd_max();
    let range = mjd_max - mjd_min;

    c.bench_function("eop_get_eop_sequential_scan", |b| {
        let mut i: u64 = 0;
        b.iter(|| {
            let frac = (i as f64) / 10000.0;
            let mjd = mjd_min + (frac % 1.0) * range;
            i += 1;
            eop.get_eop(black_box(mjd)).unwrap()
        })
    });
}

fn bench_eop_frame_transform_pattern(c: &mut Criterion) {
    let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    let mjd = 59580.0;

    c.bench_function("eop_frame_transform_pattern", |b| {
        b.iter(|| {
            let mjd = black_box(mjd);
            let _dxdy = eop.get_dxdy(mjd).unwrap();
            let _pm = eop.get_pm(mjd).unwrap();
            let _ut1_utc = eop.get_ut1_utc(mjd).unwrap();
        })
    });
}

fn bench_sw_kp_repeated(c: &mut Criterion) {
    let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
    let mjd = 60000.0;

    c.bench_function("sw_get_kp_repeated_same_mjd", |b| {
        b.iter(|| sw.get_kp(black_box(mjd)).unwrap())
    });
}

fn bench_sw_kp_sequential(c: &mut Criterion) {
    let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
    let mjd_min = sw.mjd_min();
    let mjd_last_obs = sw.mjd_last_observed();
    let range = mjd_last_obs - mjd_min;

    c.bench_function("sw_get_kp_sequential_scan", |b| {
        let mut i: u64 = 0;
        b.iter(|| {
            let frac = (i as f64) / 10000.0;
            let mjd = mjd_min + (frac % 1.0) * range;
            i += 1;
            sw.get_kp(black_box(mjd)).unwrap()
        })
    });
}

fn bench_sw_f107_repeated(c: &mut Criterion) {
    let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
    let mjd = 60000.0;

    c.bench_function("sw_get_f107_observed_repeated", |b| {
        b.iter(|| sw.get_f107_observed(black_box(mjd)).unwrap())
    });
}

// Custom main instead of criterion_main! so the dhat profiler runs when enabled.
// Run with --features dhat-heap to profile memory allocations.
fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let mut c = criterion::Criterion::default().configure_from_args();
    bench_eop_ut1_utc_repeated(&mut c);
    bench_eop_ut1_utc_sequential(&mut c);
    bench_eop_get_eop_repeated(&mut c);
    bench_eop_get_eop_sequential(&mut c);
    bench_eop_frame_transform_pattern(&mut c);
    bench_sw_kp_repeated(&mut c);
    bench_sw_kp_sequential(&mut c);
    bench_sw_f107_repeated(&mut c);
    c.final_summary();
}
