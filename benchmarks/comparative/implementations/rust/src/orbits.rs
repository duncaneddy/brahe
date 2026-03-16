use brahe::coordinates::{state_eci_to_koe, state_koe_to_eci};
use brahe::AngleFormat;
use nalgebra::SVector;
use std::time::Instant;

pub fn keplerian_to_cartesian(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let elements: Vec<Vec<f64>> = serde_json::from_value(params["elements"].clone()).unwrap();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(elements.len());

        for oe in &elements {
            let x_oe = SVector::<f64, 6>::new(oe[0], oe[1], oe[2], oe[3], oe[4], oe[5]);
            let cart = state_koe_to_eci(x_oe, AngleFormat::Degrees);
            results.push(vec![cart[0], cart[1], cart[2], cart[3], cart[4], cart[5]]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}

pub fn cartesian_to_keplerian(
    params: &serde_json::Value,
    iterations: usize,
) -> (Vec<f64>, serde_json::Value) {
    let states: Vec<Vec<f64>> = serde_json::from_value(params["states"].clone()).unwrap();

    let mut all_times = Vec::with_capacity(iterations);
    let mut first_results = Vec::new();

    for iter in 0..iterations {
        let start = Instant::now();
        let mut results = Vec::with_capacity(states.len());

        for state in &states {
            let x_cart =
                SVector::<f64, 6>::new(state[0], state[1], state[2], state[3], state[4], state[5]);
            let oe = state_eci_to_koe(x_cart, AngleFormat::Degrees);
            results.push(vec![oe[0], oe[1], oe[2], oe[3], oe[4], oe[5]]);
        }

        all_times.push(start.elapsed().as_secs_f64());

        if iter == 0 {
            first_results = results;
        }
    }

    (all_times, serde_json::to_value(first_results).unwrap())
}
