//! Track a satellite with an Unscented Kalman Filter using position measurements.

#[allow(unused_imports)]
use brahe as bh;
use nalgebra::{DMatrix, DVector};

fn main() {
    bh::initialize_eop().unwrap();

    // Define a LEO circular orbit
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

    // Perturbed initial state: 1 km position error, 1 m/s velocity error
    let mut initial_state = true_state.clone();
    initial_state[0] += 1000.0;
    initial_state[4] += 1.0;

    // Initial covariance reflecting uncertainty
    let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![
        1e6, 1e6, 1e6, 1e2, 1e2, 1e2,
    ]));

    // Create the UKF with flat constructor
    let models: Vec<Box<dyn bh::estimation::MeasurementModel>> = vec![
        Box::new(bh::estimation::InertialPositionMeasurementModel::new(10.0)),
    ];

    let mut ukf = bh::estimation::UnscentedKalmanFilter::new(
        epoch,
        initial_state,
        p0,
        bh::propagators::NumericalPropagationConfig::default(),
        bh::propagators::force_model_config::ForceModelConfig::two_body_gravity(),
        None, None, None,
        models,
        bh::estimation::UKFConfig::default(),
    ).unwrap();

    // Process 30 observations at 60-second intervals
    let dt = 60.0;
    for i in 1..=30 {
        let obs_epoch = epoch + dt * i as f64;
        truth_prop.propagate_to(obs_epoch);
        let truth_pos = truth_prop.current_state().rows(0, 3).into_owned();

        let obs = bh::estimation::Observation::new(obs_epoch, truth_pos, 0);
        ukf.process_observation(&obs).unwrap();
    }

    // Compare final state to truth
    use bh::propagators::traits::DStatePropagator;
    truth_prop.propagate_to(ukf.current_epoch());
    let truth_final = truth_prop.current_state();
    let final_state = ukf.current_state();

    let pos_error = (final_state.rows(0, 3) - truth_final.rows(0, 3)).norm();
    let vel_error = (final_state.rows(3, 3) - truth_final.rows(3, 3)).norm();

    println!("Initial position error: 1000.0 m");
    println!("Final position error:   {:.2} m", pos_error);
    println!("Final velocity error:   {:.4} m/s", vel_error);
    println!("Observations processed: {}", ukf.records().len());

    // Show final covariance diagonal (1-sigma uncertainties)
    let cov = ukf.current_covariance();
    println!("\n1-sigma uncertainties:");
    println!("  Position: [{:.1}, {:.1}, {:.1}] m",
        cov[(0,0)].sqrt(), cov[(1,1)].sqrt(), cov[(2,2)].sqrt());
    println!("  Velocity: [{:.4}, {:.4}, {:.4}] m/s",
        cov[(3,3)].sqrt(), cov[(4,4)].sqrt(), cov[(5,5)].sqrt());
}
