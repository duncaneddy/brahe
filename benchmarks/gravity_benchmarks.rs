#![allow(missing_docs)]

use std::hint::black_box;

use brahe::gravity::{GravityModel, GravityModelType, ParallelMode};
use criterion::Criterion;
use nalgebra::Vector3;

fn bench_spherical_harmonics(c: &mut Criterion) {
    let model = GravityModel::from_model_type(&GravityModelType::EGM2008_360).unwrap();
    let r_body = Vector3::new(6.5e6_f64, 1.2e6_f64, 3.1e6_f64);

    let mut group = c.benchmark_group("spherical_harmonics");
    for &n in &[2usize, 20, 50, 90, 120, 180, 360] {
        group.bench_with_input(
            criterion::BenchmarkId::new("serial", n),
            &n,
            |b, &n| {
                b.iter(|| {
                    model
                        .compute_spherical_harmonics(black_box(r_body), black_box(n), black_box(n), ParallelMode::Never)
                        .unwrap()
                });
            },
        );
        group.bench_with_input(
            criterion::BenchmarkId::new("parallel", n),
            &n,
            |b, &n| {
                b.iter(|| {
                    model
                        .compute_spherical_harmonics(black_box(r_body), black_box(n), black_box(n), ParallelMode::Always)
                        .unwrap()
                });
            },
        );
    }
    group.finish();
}

fn main() {
    let mut c = criterion::Criterion::default().configure_from_args();
    bench_spherical_harmonics(&mut c);
    c.final_summary();
}
