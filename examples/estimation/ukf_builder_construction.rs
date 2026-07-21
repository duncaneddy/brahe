//! Constructing an UnscentedKalmanFilter using the builder API.
//!
//! The builder takes the five required inputs -- epoch, state,
//! initial_covariance, force_config, and config -- directly as arguments
//! to `builder()`. Measurement models and remaining optional inputs are
//! set through chained setters.

use brahe::estimation::*;
use brahe::propagators::*;
use brahe::time::{Epoch, TimeSystem};
use nalgebra::{DMatrix, DVector};

fn main() {
    brahe::initialize_eop().unwrap();

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let state = DVector::from_vec(vec![brahe::R_EARTH + 500e3, 0.0, 0.0, 0.0, 7612.0, 0.0]);
    let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

    let ukf = UnscentedKalmanFilter::builder(
        epoch,
        state,
        p0,
        ForceModelConfig::two_body_gravity(),
        UKFConfig::default(),
    )
    .measurement_model(Box::new(InertialPositionMeasurementModel::new(10.0)))
    .build()
    .unwrap();

    println!("UKF state dimension: {}", ukf.current_state().len());
    println!("Records so far: {}", ukf.records().len());
}
