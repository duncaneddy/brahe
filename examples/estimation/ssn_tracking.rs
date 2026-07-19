//! Simulate SSN radar tracking of a LEO object and estimate its state with an EKF.
//!
//! Loads the Vallado SSN sensor dataset, finds passes with the access module,
//! simulates az/el/range measurements during passes, and processes them with
//! an Extended Kalman Filter, propagating through gaps between passes.

#[allow(unused_imports)]
use brahe as bh;
use bh::access::{ElevationConstraint, location_accesses};
use bh::datasets::ssn_sensors::load_ssn_sensors;
use bh::estimation::{EKFConfig, ExtendedKalmanFilter, MeasurementModel, Observation, SimpleSSNSensor};
use bh::traits::{DStatePropagator, InterpolatableTrajectory};
use nalgebra::{DMatrix, DVector, SVector};

fn main() {
    bh::initialize_eop().unwrap();

    // Configuration
    let meas_interval = 15.0; // seconds between measurements during a pass
    let duration = 2.0 * 3600.0; // tracking duration (seconds)
    let seed = 42u64;

    // Truth orbit: LEO at 700 km, 72 degree inclination
    let epoch = bh::Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh::TimeSystem::UTC);
    let oe = SVector::<f64, 6>::new(bh::R_EARTH + 700e3, 0.001, 72.0, 30.0, 0.0, 0.0);
    let true_state = bh::state_koe_to_eci(oe, bh::AngleFormat::Degrees);

    // Hermite-cubic interpolation (the propagator default) lets the
    // trajectory, stored at the ~60 s adaptive-step cadence, be sampled
    // accurately at the much finer measurement cadence used for measurement
    // simulation.
    let truth_config = bh::NumericalPropagationConfig::default();
    let mut truth_prop = bh::DNumericalOrbitPropagator::new(
        epoch,
        DVector::from_column_slice(true_state.as_slice()),
        truth_config,
        bh::ForceModelConfig::two_body_gravity(),
        None, None, None, None,
    ).unwrap();
    let epoch_end = epoch + duration;
    truth_prop.propagate_to(epoch_end);
    let truth_traj = truth_prop.trajectory().clone();

    // Build sensors from the Vallado SSN dataset (radar sites with calibration)
    let sites = load_ssn_sensors().unwrap();
    let mut sensors = SimpleSSNSensor::from_locations_calibrated(&sites, Some(seed));
    println!(
        "Loaded {} SSN sites, {} az/el/range sensors",
        sites.len(),
        sensors.len()
    );

    // Find passes and simulate measurements only inside them
    let mut observations: Vec<Observation> = Vec::new();
    let mut pass_count = 0;
    for (i, sensor) in sensors.iter_mut().enumerate() {
        let constraint = ElevationConstraint::new(Some(sensor.el_min().max(1.0)), None).unwrap();
        let windows = location_accesses(
            sensor.location(),
            &truth_prop,
            epoch,
            epoch_end,
            &constraint,
            None,
            None,
        ).unwrap();
        for w in windows {
            let obs = sensor
                .simulate_observations(&truth_traj, w.start(), w.end(), meas_interval, i)
                .unwrap();
            if !obs.is_empty() {
                pass_count += 1;
            }
            observations.extend(obs);
        }
    }

    observations.sort_by(|a, b| a.epoch.partial_cmp(&b.epoch).unwrap());
    println!(
        "Simulated {} measurements over {} passes",
        observations.len(),
        pass_count
    );

    // EKF from a perturbed initial state, using each sensor's matching model
    let mut initial_state = DVector::from_column_slice(true_state.as_slice());
    initial_state[0] += 1000.0;
    initial_state[4] += 1.0;
    let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![
        1e6, 1e6, 1e6, 1e2, 1e2, 1e2,
    ]));

    let models: Vec<Box<dyn MeasurementModel>> = sensors
        .iter()
        .map(|s| Box::new(s.measurement_model()) as Box<dyn MeasurementModel>)
        .collect();

    let mut ekf = ExtendedKalmanFilter::new(
        epoch,
        initial_state,
        p0,
        bh::NumericalPropagationConfig::default(),
        bh::ForceModelConfig::two_body_gravity(),
        None,
        None,
        None,
        models,
        EKFConfig::default(),
    ).unwrap();

    // Process observations in order; propagate through gaps between passes
    let gap_split = 600.0; // start a new arc when consecutive obs are > 10 min apart
    let mut prev_epoch = epoch;
    for obs in &observations {
        if obs.epoch - prev_epoch > gap_split {
            // advance through the gap in 60 s steps to record covariance growth
            let mut t = prev_epoch + 60.0;
            while t < obs.epoch {
                ekf.propagate_to(t).unwrap();
                t += 60.0;
            }
        }
        ekf.process_observation(obs).unwrap();
        prev_epoch = obs.epoch;
    }

    // Compare final estimate to truth
    let truth_final = truth_traj.interpolate(&ekf.current_epoch()).unwrap();
    let final_state = ekf.current_state();
    let err = (final_state.rows(0, 3) - truth_final.rows(0, 3)).norm();
    println!("Final position error: {:.1} m", err);

    let cov = ekf.current_covariance();
    println!(
        "Final position 1-sigma: [{:.1}, {:.1}, {:.1}] m",
        cov[(0, 0)].sqrt(),
        cov[(1, 1)].sqrt(),
        cov[(2, 2)].sqrt()
    );

    assert!(err < 500.0, "EKF should converge to a small position error");
    println!("Example validated successfully!");
}
