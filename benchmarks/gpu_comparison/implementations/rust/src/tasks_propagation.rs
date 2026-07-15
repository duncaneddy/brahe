//! SGP4 + numerical propagation handlers for bench_gpu_rust.

use brahe::math::SVector6;
use brahe::propagators::SGPPropagator;
use brahe::traits::SStatePropagator;
use nalgebra::Vector6;
use rayon::prelude::*;
use serde::Deserialize;
use std::hint::black_box;
use std::time::Instant;

use crate::{BenchmarkInput, BenchmarkOutput, make_output};

#[derive(Deserialize)]
struct Sgp4Params {
    line1: String,
    line2: String,
    tsince_minutes: Vec<f64>,
}

pub fn sgp4_iss_sweep(input: &BenchmarkInput) -> Result<BenchmarkOutput, String> {
    let p: Sgp4Params = serde_json::from_value(input.params.clone())
        .map_err(|e| format!("params parse: {e}"))?;
    assert_eq!(p.tsince_minutes.len(), input.batch_size);

    // Initialise the propagator once; each parallel iter builds its own instance
    // to avoid mutable shared state. SGPPropagator is cheap to construct.
    let line1 = p.line1;
    let line2 = p.line2;
    let offsets = p.tsince_minutes;

    // Capture the TLE epoch from one construction.
    let probe = SGPPropagator::from_tle(&line1, &line2, 60.0)
        .map_err(|e| format!("SGP init: {e:?}"))?;
    let tle_epoch = probe.initial_epoch();
    drop(probe);

    let propagate = |t: f64| -> Vector6<f64> {
        let prop = SGPPropagator::from_tle(&line1, &line2, 60.0).expect("SGP init");
        let target = tle_epoch + t * 60.0;
        prop.state_pef(target).expect("state_pef")
    };

    for _ in 0..input.warmup_iterations {
        let _: Vec<Vector6<f64>> = offsets.par_iter().map(|t| black_box(propagate(*t))).collect();
    }

    let mut times = Vec::with_capacity(input.iterations);
    for _ in 0..input.iterations {
        let start = Instant::now();
        let _: Vec<Vector6<f64>> = offsets.par_iter().map(|t| black_box(propagate(*t))).collect();
        times.push(start.elapsed().as_secs_f64().max(1e-12));
    }
    Ok(make_output(input, times))
}

// ───────────────────────────── Numerical two-body / J2 ──────────────────────────

use brahe::eop::{EOPExtrapolation, FileEOPProvider, set_global_eop_provider};
use brahe::propagators::{
    DNumericalOrbitPropagator, ForceModelConfig, GravityConfiguration, NumericalPropagationConfig,
    ZonalHarmonicsDegree,
};
use brahe::FrameTransformationModel;
use brahe::time::{Epoch, TimeSystem};
use brahe::traits::DStatePropagator;
use nalgebra::DVector;
use std::sync::Once;

static EOP_INIT: Once = Once::new();

fn ensure_eop_loaded() {
    EOP_INIT.call_once(|| {
        if let Ok(path) = std::env::var("BRAHE_EOP_FILE") {
            let p = std::path::Path::new(&path);
            if let Ok(eop) = FileEOPProvider::from_file(p, true, EOPExtrapolation::Hold) {
                set_global_eop_provider(eop);
            }
        }
    });
}

#[derive(Deserialize)]
struct NumericalJ2Params {
    states_eci: Vec<[f64; 6]>,
    step_size: f64,
    n_steps: usize,
}

pub fn numerical_twobody_j2(input: &BenchmarkInput) -> Result<BenchmarkOutput, String> {
    ensure_eop_loaded();
    let p: NumericalJ2Params = serde_json::from_value(input.params.clone())
        .map_err(|e| format!("params parse: {e}"))?;
    assert_eq!(p.states_eci.len(), input.batch_size);

    let epoch_0 = Epoch::from_datetime(2024, 6, 15, 12, 0, 0.0, 0.0, TimeSystem::UTC);
    let dt_total = p.step_size * p.n_steps as f64;
    let force = ForceModelConfig {
        gravity: GravityConfiguration::EarthZonal {
            degree: ZonalHarmonicsDegree::J2,
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
            epoch_0,
            dstate,
            cfg.clone(),
            force.clone(),
            None, None, None, None,
        )
        .expect("propagator init");
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
