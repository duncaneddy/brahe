//! Enable the precomputed frame-rotation and ephemeris tables and compare against
//! exact evaluation.

use brahe as bh;
use bh::traits::DStatePropagator;
use nalgebra as na;

fn main() {
    bh::initialize_eop().unwrap();

    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let oe = na::SVector::<f64, 6>::new(bh::R_EARTH + 500e3, 0.001, 97.8, 15.0, 30.0, 45.0);
    let state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Fixed-step RK4 gives both propagations an identical step sequence
    let prop_config = bh::NumericalPropagationConfig {
        method: bh::IntegratorMethod::RK4,
        integrator: bh::IntegratorConfig {
            initial_step: Some(30.0),
            ..bh::IntegratorConfig::fixed_step(30.0)
        },
        ..Default::default()
    };

    // Exact rotation and DE440s ephemeris queries at every dynamics call (the default)
    let exact_config = bh::ForceModelConfig {
        gravity: bh::GravityConfiguration::SphericalHarmonic {
            source: bh::GravityModelSource::ModelType(bh::GravityModelType::EGM2008_120),
            degree: 20,
            order: 20,
            parallel: bh::ParallelMode::Auto,
        },
        third_body: Some(vec![bh::ThirdBody::Sun.into(), bh::ThirdBody::Moon.into()]),
        ..bh::ForceModelConfig::two_body_gravity()
    };

    // Tabulate bias-precession-nutation and polar motion every 600 s and the
    // Sun and Moon positions every 60 s
    let precomputed_config = bh::ForceModelConfig {
        precompute: Some(bh::PrecomputeConfig {
            frame_rotation_step: Some(600.0),
            ephemeris_step: Some(60.0),
            cache_space_weather: false,
            ..bh::PrecomputeConfig::default()
        }),
        ..exact_config.clone()
    };

    let build = |force_config: bh::ForceModelConfig| {
        bh::DNumericalOrbitPropagator::new(
            epoch,
            na::DVector::from_column_slice(state.as_slice()),
            prop_config.clone(),
            force_config,
            None,
            None,
            None,
            None,
        )
        .unwrap()
    };

    let mut exact = build(exact_config);
    let mut precomputed = build(precomputed_config.clone());

    exact.propagate_to(epoch + 86400.0).unwrap();
    precomputed.propagate_to(epoch + 86400.0).unwrap();

    let difference =
        (exact.current_state().rows(0, 3) - precomputed.current_state().rows(0, 3)).norm();
    println!("Position difference after one day: {:.2e} m", difference);

    assert!(difference < 1e-3);

    // par_propagate_to_d builds one set of tables for propagators with matching table
    // steps and force-model settings, and shares it across all of them
    let target = epoch + 86400.0;
    let mut constellation: Vec<bh::DNumericalOrbitPropagator> = (0..4)
        .map(|_| build(precomputed_config.clone()))
        .collect();
    bh::par_propagate_to_d(&mut constellation, target).unwrap();

    for prop in &constellation {
        assert_eq!(prop.current_epoch(), target);
    }
    println!(
        "Propagated {} propagators with shared tables",
        constellation.len()
    );
}
