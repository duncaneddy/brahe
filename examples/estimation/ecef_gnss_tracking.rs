//! Track a satellite using ECEF position measurements from a GNSS receiver.

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

    // Truth propagator for generating simulated GNSS observations
    let mut truth_prop = bh::propagators::DNumericalOrbitPropagator::new(
        epoch,
        true_state.clone(),
        bh::propagators::NumericalPropagationConfig::default(),
        bh::propagators::force_model_config::ForceModelConfig::two_body_gravity(),
        None, None, None, None,
    ).unwrap();

    // Perturbed initial state: 1 km position error
    let mut initial_state = true_state.clone();
    initial_state[0] += 1000.0;
    initial_state[4] += 1.0;
    let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![
        1e6, 1e6, 1e6, 1e2, 1e2, 1e2,
    ]));

    // ECEF position model with typical GNSS accuracy (5 m noise)
    let ecef_model = bh::estimation::EcefPositionMeasurementModel::new(5.0);
    let models: Vec<Box<dyn bh::estimation::MeasurementModel>> = vec![
        Box::new(ecef_model),
    ];

    let mut ekf = bh::estimation::ExtendedKalmanFilter::new(
        epoch,
        initial_state,
        p0,
        bh::propagators::NumericalPropagationConfig::default(),
        bh::propagators::force_model_config::ForceModelConfig::two_body_gravity(),
        None, None, None,
        models,
        bh::estimation::EKFConfig::default(),
    ).unwrap();

    // Simulate GNSS observations: get truth ECI state, convert to ECEF
    let dt = 60.0;
    for i in 1..=20 {
        let obs_epoch = epoch + dt * i as f64;
        truth_prop.propagate_to(obs_epoch);
        let truth_eci = truth_prop.current_state();

        // Simulate GNSS: convert truth position to ECEF
        let truth_eci_pos = nalgebra::Vector3::new(truth_eci[0], truth_eci[1], truth_eci[2]);
        let truth_ecef_pos = bh::frames::position_eci_to_ecef(obs_epoch, truth_eci_pos);
        let z = DVector::from_vec(vec![truth_ecef_pos[0], truth_ecef_pos[1], truth_ecef_pos[2]]);

        let obs = bh::estimation::Observation::new(obs_epoch, z, 0);
        ekf.process_observation(&obs).unwrap();
    }

    // Compare final state to truth
    use bh::propagators::traits::DStatePropagator;
    truth_prop.propagate_to(ekf.current_epoch());
    let truth_final = truth_prop.current_state();
    let final_state = ekf.current_state();
    let pos_error = (final_state.rows(0, 3) - truth_final.rows(0, 3)).norm();
    let vel_error = (final_state.rows(3, 3) - truth_final.rows(3, 3)).norm();

    println!("ECEF GNSS tracking with EcefPositionMeasurementModel:");
    println!("  Initial position error: 1000.0 m");
    println!("  Final position error:   {:.2} m", pos_error);
    println!("  Final velocity error:   {:.4} m/s", vel_error);
    println!("  Observations processed: {}", ekf.records().len());
}
