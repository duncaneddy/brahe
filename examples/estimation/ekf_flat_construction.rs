//! Constructing an ExtendedKalmanFilter using the flat constructor.
//!
//! The flat constructor takes every field positionally, as an alternative
//! to the builder when all inputs are already at hand.

use brahe::estimation::*;
use brahe::propagators::*;
use brahe::time::{Epoch, TimeSystem};
use nalgebra::{DMatrix, DVector};

fn main() {
    brahe::initialize_eop().unwrap();

    let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
    let state = DVector::from_vec(vec![brahe::R_EARTH + 500e3, 0.0, 0.0, 0.0, 7612.0, 0.0]);
    let p0 = DMatrix::from_diagonal(&DVector::from_vec(vec![1e6, 1e6, 1e6, 1e2, 1e2, 1e2]));

    let ekf = ExtendedKalmanFilter::new(
        epoch,
        state,
        p0,
        NumericalPropagationConfig::default(),
        ForceModelConfig::two_body_gravity(),
        None,
        None,
        None,
        vec![Box::new(InertialPositionMeasurementModel::new(10.0))],
        EKFConfig::default(),
    )
    .unwrap();

    println!("EKF state dimension: {}", ekf.current_state().len());
    println!("EKF current epoch: {}", ekf.current_epoch());
}
