//! Full-force-model propagation handlers (5x5 spherical harmonic).

use brahe::FrameTransformationModel;
use brahe::propagators::{
    DNumericalOrbitPropagator, ForceModelConfig, GravityConfiguration, GravityModelSource,
    NumericalPropagationConfig,
};
use brahe::time::{Epoch, TimeSystem};
use brahe::traits::DStatePropagator;
use nalgebra::DVector;
use rayon::prelude::*;
use serde::Deserialize;
use std::hint::black_box;
use std::time::Instant;

use crate::{BenchmarkInput, BenchmarkOutput, make_output};

#[derive(Deserialize)]
struct Grav5x5Params {
    states_eci: Vec<[f64; 6]>,
    step_size: f64,
    n_steps: usize,
    gravity_degree: usize,
    gravity_order: usize,
}

pub fn grav_5x5(input: &BenchmarkInput) -> Result<BenchmarkOutput, String> {
    let p: Grav5x5Params = serde_json::from_value(input.params.clone())
        .map_err(|e| format!("params parse: {e}"))?;
    assert_eq!(p.states_eci.len(), input.batch_size);

    let epoch_0 = Epoch::from_datetime(2024, 6, 15, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    let dt_total = p.step_size * p.n_steps as f64;
    let force = ForceModelConfig {
        gravity: GravityConfiguration::SphericalHarmonic {
            source: GravityModelSource::default(),
            degree: p.gravity_degree,
            order: p.gravity_order,
            parallel: brahe::orbit_dynamics::ParallelMode::Auto,
        },
        drag: None,
        srp: None,
        third_bodies: None,
        relativity: false,
        mass: None,
        frame_transform: FrameTransformationModel::EarthRotationOnly,
    };
    let cfg = NumericalPropagationConfig::default();

    let propagate = |x0: &[f64; 6]| -> DVector<f64> {
        let dstate = DVector::from_vec(x0.to_vec());
        let mut prop = DNumericalOrbitPropagator::new(
            epoch_0, dstate, cfg.clone(), force.clone(),
            None, None, None, None,
        ).expect("propagator init");
        prop.propagate_to(epoch_0 + dt_total);
        prop.current_state().clone()
    };

    for _ in 0..input.warmup_iterations {
        let _: Vec<DVector<f64>> = p.states_eci.par_iter().map(|s| black_box(propagate(s))).collect();
    }

    let mut times = Vec::with_capacity(input.iterations);
    for _ in 0..input.iterations {
        let start = Instant::now();
        let _: Vec<DVector<f64>> = p.states_eci.par_iter().map(|s| black_box(propagate(s))).collect();
        times.push(start.elapsed().as_secs_f64().max(1e-12));
    }
    Ok(make_output(input, times))
}
