use brahe::AngleFormat;
use brahe::coordinates::state_koe_to_eci;
use brahe::propagators::traits::{DStatePropagator, SStatePropagator, SStateProvider};
use brahe::traits::DOrbitStateProvider;
use brahe::propagators::{
    DNumericalOrbitPropagator, ForceModelConfig, KeplerianPropagator, NumericalPropagationConfig,
    SGPPropagator,
};
use brahe::time::{Epoch, TimeSystem};
use brahe::TrajectoryMode;
use nalgebra::{DVector, SVector};
use std::time::Instant;

pub fn keplerian_single(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    #[derive(serde::Deserialize)]
    struct Case {
        jd: f64,
        elements: Vec<f64>,
        dt: f64,
    }
    let cases: Vec<Case> = serde_json::from_value(params["cases"].clone()).unwrap();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(cases.len());

        for case in &cases {
            let epc = Epoch::from_jd(case.jd, TimeSystem::UTC);
            let oe = SVector::<f64, 6>::new(
                case.elements[0],
                case.elements[1],
                case.elements[2],
                case.elements[3],
                case.elements[4],
                case.elements[5],
            );
            let target = epc + case.dt;

            let prop = KeplerianPropagator::from_keplerian(epc, oe, AngleFormat::Degrees, 60.0);
            let state = DOrbitStateProvider::state_eci(&prop, target).unwrap();
            results.push(vec![
                state[0], state[1], state[2], state[3], state[4], state[5],
            ]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn keplerian_trajectory(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let jd: f64 = serde_json::from_value(params["jd"].clone()).unwrap();
    let elements: Vec<f64> = serde_json::from_value(params["elements"].clone()).unwrap();
    let step_size: f64 = serde_json::from_value(params["step_size"].clone()).unwrap();
    let n_steps: usize = serde_json::from_value(params["n_steps"].clone()).unwrap();

    let epc = Epoch::from_jd(jd, TimeSystem::UTC);
    let oe = SVector::<f64, 6>::new(
        elements[0],
        elements[1],
        elements[2],
        elements[3],
        elements[4],
        elements[5],
    );

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let prop = KeplerianPropagator::from_keplerian(epc, oe, AngleFormat::Degrees, step_size);
        let mut results = Vec::with_capacity(n_steps);

        for step_idx in 0..n_steps {
            let target = epc + (step_idx as f64 + 1.0) * step_size;
            let state = DOrbitStateProvider::state_eci(&prop, target).unwrap();
            results.push(vec![
                state[0], state[1], state[2], state[3], state[4], state[5],
            ]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn sgp4_single(params: &serde_json::Value, iterations: usize) -> (Vec<f64>, serde_json::Value) {
    let line1: String = serde_json::from_value(params["line1"].clone()).unwrap();
    let line2: String = serde_json::from_value(params["line2"].clone()).unwrap();
    let offsets: Vec<f64> = serde_json::from_value(params["time_offsets_seconds"].clone()).unwrap();

    let base_prop = SGPPropagator::from_tle(&line1, &line2, 60.0).unwrap();
    let base_epoch = base_prop.initial_epoch();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(offsets.len());

        for dt in &offsets {
            let target = base_epoch + *dt;
            // Use SStateProvider::state() to get TEME output directly (matches Java's TEME frame)
            let state = SStateProvider::state(&base_prop, target).unwrap();
            results.push(vec![
                state[0], state[1], state[2], state[3], state[4], state[5],
            ]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn sgp4_trajectory(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let line1: String = serde_json::from_value(params["line1"].clone()).unwrap();
    let line2: String = serde_json::from_value(params["line2"].clone()).unwrap();
    let step_size: f64 = serde_json::from_value(params["step_size"].clone()).unwrap();
    let n_steps: usize = serde_json::from_value(params["n_steps"].clone()).unwrap();

    let prop = SGPPropagator::from_tle(&line1, &line2, step_size).unwrap();
    let base_epoch = prop.initial_epoch();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(n_steps);

        for step_idx in 0..n_steps {
            let target = base_epoch + (step_idx as f64 + 1.0) * step_size;
            // Use SStateProvider::state() to get TEME output directly (matches Java's TEME frame)
            let state = SStateProvider::state(&prop, target).unwrap();
            results.push(vec![
                state[0], state[1], state[2], state[3], state[4], state[5],
            ]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn numerical_twobody(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let jd: f64 = serde_json::from_value(params["jd"].clone()).unwrap();
    let elements: Vec<f64> = serde_json::from_value(params["elements"].clone()).unwrap();
    let step_size: f64 = serde_json::from_value(params["step_size"].clone()).unwrap();
    let n_steps: usize = serde_json::from_value(params["n_steps"].clone()).unwrap();

    let epc = Epoch::from_jd(jd, TimeSystem::UTC);
    let oe = SVector::<f64, 6>::new(
        elements[0],
        elements[1],
        elements[2],
        elements[3],
        elements[4],
        elements[5],
    );
    let cart = state_koe_to_eci(oe, AngleFormat::Degrees);
    let state_dv = DVector::from_vec(vec![cart[0], cart[1], cart[2], cart[3], cart[4], cart[5]]);

    let prop_config = NumericalPropagationConfig::default();
    let force_config = ForceModelConfig::two_body_gravity();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut prop = DNumericalOrbitPropagator::new(
            epc,
            state_dv.clone(),
            prop_config.clone(),
            force_config.clone(),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        prop.set_trajectory_mode(TrajectoryMode::Disabled);

        let mut results = Vec::with_capacity(n_steps);
        for step_idx in 0..n_steps {
            let target = epc + (step_idx as f64 + 1.0) * step_size;
            prop.propagate_to(target);
            let state = prop.current_state();
            results.push(vec![
                state[0], state[1], state[2], state[3], state[4], state[5],
            ]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}
