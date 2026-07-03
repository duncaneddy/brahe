#![allow(missing_docs)]

use std::hint::black_box;

use brahe::gravity::{GravityModel, GravityModelType, GravityTables, ParallelMode};
use criterion::Criterion;
use nalgebra::{DMatrix, Vector3};

/// Above this degree, the Cunningham kernel overflows (non-finite V/W) at
/// `r_body`'s altitude/latitude — see the kernel's `# Numerical limits` docs.
/// The Cunningham benches are skipped above this degree; Clenshaw benches run
/// at all sizes.
const CUNNINGHAM_MAX_VALID_N: usize = 120;

fn bench_spherical_harmonics(c: &mut Criterion) {
    let model =
        GravityModel::from_model_type_with_tables(&GravityModelType::EGM2008_360, GravityTables::Both).unwrap();
    let r_body = Vector3::new(6.5e6_f64, 1.2e6_f64, 3.1e6_f64);

    let mut group = c.benchmark_group("spherical_harmonics");
    for &n in &[2usize, 20, 50, 90, 120, 180, 240, 360] {
        if n <= CUNNINGHAM_MAX_VALID_N {
            group.bench_with_input(
                criterion::BenchmarkId::new("cunningham_serial", n),
                &n,
                |b, &n| {
                    b.iter(|| {
                        model
                            .compute_spherical_harmonics_cunningham(black_box(r_body), black_box(n), black_box(n), ParallelMode::Never)
                            .unwrap()
                    });
                },
            );
            group.bench_with_input(
                criterion::BenchmarkId::new("cunningham_parallel", n),
                &n,
                |b, &n| {
                    b.iter(|| {
                        model
                            .compute_spherical_harmonics_cunningham(black_box(r_body), black_box(n), black_box(n), ParallelMode::Always)
                            .unwrap()
                    });
                },
            );
        }
        group.bench_with_input(
            criterion::BenchmarkId::new("clenshaw_serial", n),
            &n,
            |b, &n| {
                b.iter(|| {
                    model
                        .compute_spherical_harmonics_clenshaw(black_box(r_body), black_box(n), black_box(n), ParallelMode::Never)
                        .unwrap()
                });
            },
        );
        group.bench_with_input(
            criterion::BenchmarkId::new("clenshaw_parallel", n),
            &n,
            |b, &n| {
                b.iter(|| {
                    model
                        .compute_spherical_harmonics_clenshaw(black_box(r_body), black_box(n), black_box(n), ParallelMode::Always)
                        .unwrap()
                });
            },
        );
    }
    group.finish();
}

/// Isolates the small-n serial path common to LEO propagation (n = 5, 20, 80).
///
/// Three variants per degree:
/// - `alloc`: `compute_spherical_harmonics` — includes the two per-call
///   `DMatrix::zeros((n+2)²)` allocations.
/// - `workspace`: `compute_spherical_harmonics_cunningham_with_workspace` reusing a
///   pre-sized workspace — this is what the propagator hot path actually does,
///   so it measures compute overhead with allocation amortized away.
/// - `clenshaw`: `compute_spherical_harmonics_clenshaw` — the propagator's
///   default kernel; no caller-managed workspace to amortize.
///
/// Comparing the two tells us whether small-n cost is allocation-bound or
/// compute-bound before we optimize the recurrence/accumulation loops.
fn bench_small_n_serial(c: &mut Criterion) {
    let model =
        GravityModel::from_model_type_with_tables(&GravityModelType::EGM2008_360, GravityTables::Both).unwrap();
    let r_body = Vector3::new(6.5e6_f64, 1.2e6_f64, 3.1e6_f64);

    let mut group = c.benchmark_group("spherical_harmonics_small_n");
    for &n in &[5usize, 20, 80] {
        group.bench_with_input(criterion::BenchmarkId::new("alloc", n), &n, |b, &n| {
            b.iter(|| {
                model
                    .compute_spherical_harmonics(
                        black_box(r_body),
                        black_box(n),
                        black_box(n),
                        ParallelMode::Never,
                    )
                    .unwrap()
            });
        });

        let mut v = DMatrix::<f64>::zeros(n + 2, n + 2);
        let mut w = DMatrix::<f64>::zeros(n + 2, n + 2);
        group.bench_with_input(criterion::BenchmarkId::new("workspace", n), &n, |b, &n| {
            b.iter(|| {
                model
                    .compute_spherical_harmonics_cunningham_with_workspace(
                        black_box(r_body),
                        black_box(n),
                        black_box(n),
                        ParallelMode::Never,
                        &mut v,
                        &mut w,
                    )
                    .unwrap()
            });
        });

        group.bench_with_input(criterion::BenchmarkId::new("clenshaw", n), &n, |b, &n| {
            b.iter(|| {
                model
                    .compute_spherical_harmonics_clenshaw(
                        black_box(r_body),
                        black_box(n),
                        black_box(n),
                        ParallelMode::Never,
                    )
                    .unwrap()
            });
        });
    }
    group.finish();
}

fn main() {
    let mut c = criterion::Criterion::default().configure_from_args();
    bench_spherical_harmonics(&mut c);
    bench_small_n_serial(&mut c);
    c.final_summary();
}
