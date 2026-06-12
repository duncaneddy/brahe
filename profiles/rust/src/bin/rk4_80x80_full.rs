//! Profile: RK4 numerical propagation, full force model — 80×80 gravity +
//! sun/moon third-body + NRLMSISE-00 drag + SRP. Heaviest baseline; matches
//! the structure used in `benchmarks/propagator_benchmarks.rs` for the
//! conservative-forces case but adds drag and SRP for full fidelity.

#![allow(missing_docs)]

use brahe::propagators::{
    AtmosphericModel, DNumericalOrbitPropagator, DragConfiguration, EclipseModel,
    EphemerisSource, ForceModelConfig, GravityConfiguration, GravityModelSource,
    NumericalPropagationConfig, ParameterSource, SolarRadiationPressureConfiguration,
    ThirdBody, ThirdBodyConfiguration,
};
use brahe::traits::DStatePropagator;
use nalgebra::DVector;
use profiles::common::{
    default_leo_state, duration_from_env, run_until_elapsed, setup_providers,
};

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    setup_providers();
    let (epoch, dstate) = default_leo_state();

    // Parameter vector layout (see DefaultParameterLayout):
    //   [mass, drag_area, Cd, srp_area, Cr]
    let params = DVector::from_vec(vec![1000.0, 10.0, 2.2, 10.0, 1.3]);

    let force = ForceModelConfig {
        gravity: GravityConfiguration::SphericalHarmonic {
            source: GravityModelSource::default(),
            degree: 80,
            order: 80,
            parallel: brahe::orbit_dynamics::ParallelMode::Auto,
        },
        drag: Some(DragConfiguration {
            model: AtmosphericModel::NRLMSISE00,
            area: ParameterSource::ParameterIndex(1),
            cd: ParameterSource::ParameterIndex(2),
        }),
        srp: Some(SolarRadiationPressureConfiguration {
            area: ParameterSource::ParameterIndex(3),
            cr: ParameterSource::ParameterIndex(4),
            eclipse_model: EclipseModel::Conical,
        }),
        third_body: Some(ThirdBodyConfiguration {
            ephemeris_source: EphemerisSource::DE440s,
            bodies: vec![ThirdBody::Sun, ThirdBody::Moon],
        }),
        relativity: false,
        mass: Some(ParameterSource::ParameterIndex(0)),
        frame_transform: Default::default(),
    };
    let prop_cfg = NumericalPropagationConfig::default();
    let duration_s = duration_from_env();

    let iters = run_until_elapsed(duration_s, || {
        let mut prop = DNumericalOrbitPropagator::new(
            epoch,
            dstate.clone(),
            prop_cfg.clone(),
            force.clone(),
            Some(params.clone()),
            None,
            None,
            None,
        )
        .expect("propagator construction must succeed");
        prop.propagate_to(epoch + 86400.0);
        std::hint::black_box(prop);
    });

    eprintln!("rk4_80x80_full: {iters} iterations in ~{duration_s:.1}s");
}
