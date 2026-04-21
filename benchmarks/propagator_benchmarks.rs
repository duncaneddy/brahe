#![allow(missing_docs)]

use brahe::{GravityConfiguration, GravityModelSource, ZonalHarmonicsDegrees};
use criterion::Criterion;

use brahe::constants::AngleFormat;
use brahe::coordinates::state_koe_to_eci;
use brahe::eop::{EOPExtrapolation, FileEOPProvider, set_global_eop_provider};
use brahe::math::SVector6;
use brahe::propagators::{
    DNumericalOrbitPropagator, ForceModelConfig, NumericalPropagationConfig, SGPPropagator,
};
use brahe::space_weather::{FileSpaceWeatherProvider, set_global_space_weather_provider};
use brahe::time::{Epoch, TimeSystem};
use brahe::traits::{DStatePropagator, SStatePropagator};
use nalgebra::DVector;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn setup_providers() {
    let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
    set_global_eop_provider(eop);

    let sw = FileSpaceWeatherProvider::from_default_file().unwrap();
    set_global_space_weather_provider(sw);
}

fn bench_sgp4_24hour(c: &mut Criterion) {
    setup_providers();

    let line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927";
    let line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537";

    let mut group = c.benchmark_group("propagator_sgp4");
    group.sample_size(10);

    group.bench_function("sgp4_24h_propagation", |b| {
        b.iter(|| {
            let mut prop = SGPPropagator::from_tle(line1, line2, 60.0).unwrap();
            let target = prop.current_epoch() + 86400.0;
            prop.propagate_to(target);
        })
    });

    group.finish();
}

fn bench_numerical_conservative_24hour(c: &mut Criterion) {
    setup_providers();

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let oe = SVector6::new(
        brahe::constants::R_EARTH + 500e3,
        0.01,
        97.8,
        15.0,
        30.0,
        45.0,
    );
    let state = state_koe_to_eci(oe, AngleFormat::Degrees);
    let dstate = DVector::from_column_slice(state.as_slice());

    let mut group = c.benchmark_group("propagator_numerical");
    group.sample_size(10);

    group.bench_function("numerical_conservative_24h", |b| {
        b.iter(|| {
            let mut prop = DNumericalOrbitPropagator::new(
                epoch,
                dstate.clone(),
                NumericalPropagationConfig::default(),
                ForceModelConfig::conservative_forces(),
                None,
                None,
                None,
                None,
            )
            .unwrap();
            let target = epoch + 86400.0;
            prop.propagate_to(target);
        })
    });

    group.finish();
}

fn bench_numerical_fast_j6_zonal_24hour(c: &mut Criterion) {
    setup_providers();

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let oe = SVector6::new(
        brahe::constants::R_EARTH + 500e3,
        0.01,
        97.8,
        15.0,
        30.0,
        45.0,
    );

    let degree = ZonalHarmonicsDegrees::J6;
    let state = state_koe_to_eci(oe, AngleFormat::Degrees);
    let dstate = DVector::from_column_slice(state.as_slice());
    let force = ForceModelConfig {
        gravity: GravityConfiguration::Zonal { degree },
        drag: None,
        srp: None,
        third_body: None,
        relativity: false,
        mass: None,
    };

    let mut group = c.benchmark_group("propagator_numerical");
    group.sample_size(10);

    group.bench_function("numerical_fast_j6_zonal_24hour", |b| {
        b.iter(|| {
            let mut prop = DNumericalOrbitPropagator::new(
                epoch,
                dstate.clone(),
                NumericalPropagationConfig::default(),
                force.clone(),
                None,
                None,
                None,
                None,
            )
            .unwrap();
            let target = epoch + 86400.0;
            prop.propagate_to(target);
        })
    });

    group.finish();
}

fn bench_spherical_harmonic_numerical_j6_equivalent_24hour(c: &mut Criterion) {
    setup_providers();

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let oe = SVector6::new(
        brahe::constants::R_EARTH + 500e3,
        0.01,
        97.8,
        15.0,
        30.0,
        45.0,
    );
    let state = state_koe_to_eci(oe, AngleFormat::Degrees);
    let dstate = DVector::from_column_slice(state.as_slice());
    let force = ForceModelConfig {
        gravity: GravityConfiguration::SphericalHarmonic {
            source: GravityModelSource::default(),
            degree: ZonalHarmonicsDegrees::J6.into(),
            order: 0,
        },
        drag: None,
        srp: None,
        third_body: None,
        relativity: false,
        mass: None,
    };

    let mut group = c.benchmark_group("propagator_numerical");
    group.sample_size(10);

    group.bench_function(
        "numerical_spherical_harmonic_numerical_j6_equivalent_24hour",
        |b| {
            b.iter(|| {
                let mut prop = DNumericalOrbitPropagator::new(
                    epoch,
                    dstate.clone(),
                    NumericalPropagationConfig::default(),
                    force.clone(),
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap();
                let target = epoch + 86400.0;
                prop.propagate_to(target);
            })
        },
    );

    group.finish();
}

// Custom main instead of criterion_main! so the dhat profiler runs when enabled
// Run bench with --features dhat-heap to profile memory allocations
fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let mut c = criterion::Criterion::default().configure_from_args();
    bench_sgp4_24hour(&mut c);
    bench_numerical_conservative_24hour(&mut c);
    bench_numerical_fast_j6_zonal_24hour(&mut c);
    bench_spherical_harmonic_numerical_j6_equivalent_24hour(&mut c);
    c.final_summary();
}
