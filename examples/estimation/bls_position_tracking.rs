//! Batch Least Squares orbit determination from position measurements.

#[allow(unused_imports)]
use brahe as bh;
use bh::propagators::traits::DStatePropagator;
use nalgebra::{DMatrix, DVector};

fn main() {
    bh::initialize_eop().unwrap();

    // Define truth orbit: LEO circular at ~500km altitude
    let epoch = bh::time::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::time::TimeSystem::UTC);
    let r = bh::constants::physical::R_EARTH + 500e3;
    let v = (bh::constants::physical::GM_EARTH / r).sqrt();
    let true_state = DVector::from_vec(vec![r, 0.0, 0.0, 0.0, v, 0.0]);

    // Create a truth propagator for generating observations
    let mut truth_prop = bh::propagators::DNumericalOrbitPropagator::new(
        epoch,
        true_state.clone(),
        bh::propagators::NumericalPropagationConfig::default(),
        bh::propagators::force_model_config::ForceModelConfig::two_body_gravity(),
        None, None, None, None,
    ).unwrap();

    // Generate 20 noise-free position observations at 30-second intervals
    let mut observations = Vec::new();
    for i in 1..=20 {
        let obs_epoch = epoch + 30.0 * i as f64;
        truth_prop.propagate_to(obs_epoch);
        let truth_pos = truth_prop.current_state().rows(0, 3).into_owned();
        observations.push(bh::estimation::Observation::new(obs_epoch, truth_pos, 0));
    }

    // Initial guess: perturbed by 1km in x-position
    let mut initial_state = true_state.clone();
    initial_state[0] += 1000.0;

    // A priori covariance
    let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![
        1e6, 1e6, 1e6, 1e2, 1e2, 1e2,
    ]));

    // Create and run BLS
    let models: Vec<Box<dyn bh::estimation::MeasurementModel>> = vec![
        Box::new(bh::estimation::InertialPositionMeasurementModel::new(10.0)),
    ];

    let mut bls = bh::estimation::BatchLeastSquares::new(
        epoch,
        initial_state,
        p0,
        bh::propagators::NumericalPropagationConfig::default(),
        bh::propagators::force_model_config::ForceModelConfig::two_body_gravity(),
        None, None, None,
        models,
        bh::estimation::BLSConfig::default(),
    ).unwrap();

    bls.solve(&observations).unwrap();

    // Results
    println!("Converged: {}", bls.converged());
    println!("Iterations: {}", bls.iterations_completed());
    println!("Final cost: {:.6e}", bls.final_cost());

    let pos_error = (bls.current_state().rows(0, 3) - true_state.rows(0, 3)).norm();
    println!("Position error: {:.6} m", pos_error);

    // Iteration history
    for rec in bls.iteration_records() {
        println!("  Iter {}: cost={:.6e}, ||dx||={:.6e}",
            rec.iteration, rec.cost, rec.state_correction_norm);
    }
}
