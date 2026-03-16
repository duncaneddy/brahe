use brahe::frames::{state_ecef_to_eci, state_eci_to_ecef};
use brahe::time::{Epoch, TimeSystem};
use nalgebra::SVector;
use std::time::Instant;

#[derive(serde::Deserialize)]
struct FrameCase {
    jd: f64,
    state: Vec<f64>,
}

fn parse_cases(params: &serde_json::Value) -> Vec<FrameCase> {
    serde_json::from_value(params["cases"].clone()).unwrap()
}

pub fn eci_to_ecef(params: &serde_json::Value, iterations: usize) -> (Vec<f64>, serde_json::Value) {
    let cases = parse_cases(params);

    // Pre-build epochs and state vectors
    let epoch_states: Vec<(Epoch, SVector<f64, 6>)> = cases
        .iter()
        .map(|c| {
            let epc = Epoch::from_jd(c.jd, TimeSystem::UTC);
            let sv = SVector::<f64, 6>::new(
                c.state[0], c.state[1], c.state[2], c.state[3], c.state[4], c.state[5],
            );
            (epc, sv)
        })
        .collect();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(epoch_states.len());

        for (epc, state) in &epoch_states {
            let ecef = state_eci_to_ecef(*epc, *state);
            results.push(vec![ecef[0], ecef[1], ecef[2], ecef[3], ecef[4], ecef[5]]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn ecef_to_eci(params: &serde_json::Value, iterations: usize) -> (Vec<f64>, serde_json::Value) {
    let cases = parse_cases(params);

    let epoch_states: Vec<(Epoch, SVector<f64, 6>)> = cases
        .iter()
        .map(|c| {
            let epc = Epoch::from_jd(c.jd, TimeSystem::UTC);
            let sv = SVector::<f64, 6>::new(
                c.state[0], c.state[1], c.state[2], c.state[3], c.state[4], c.state[5],
            );
            (epc, sv)
        })
        .collect();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(epoch_states.len());

        for (epc, state) in &epoch_states {
            let eci = state_ecef_to_eci(*epc, *state);
            results.push(vec![eci[0], eci[1], eci[2], eci[3], eci[4], eci[5]]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}
