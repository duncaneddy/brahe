/*!
 * Data structures for observations and filter records.
 *
 * Defines the core types used to represent observations and filter update records
 * for sequential (EKF/UKF) estimators.
 */

use nalgebra::{DMatrix, DVector};

use crate::time::Epoch;

/// A single observation (measurement) at a specific epoch.
///
/// Pairs a measurement vector with a reference to the measurement model that
/// should be used to process it. Different measurement types at different times
/// are supported by setting different `model_index` values.
///
/// # Arguments
///
/// * `epoch` - Time of the observation
/// * `measurement` - Measurement vector (z) in SI units
/// * `model_index` - Index into the estimator's measurement model list
#[derive(Clone, Debug)]
pub struct Observation {
    /// Time of the observation
    pub epoch: Epoch,
    /// Measurement vector (z)
    pub measurement: DVector<f64>,
    /// Index into the measurement model list
    pub model_index: usize,
}

impl Observation {
    /// Create a new observation.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Time of the observation
    /// * `measurement` - Measurement vector
    /// * `model_index` - Index into the estimator's measurement model list
    pub fn new(epoch: Epoch, measurement: DVector<f64>, model_index: usize) -> Self {
        Self {
            epoch,
            measurement,
            model_index,
        }
    }
}

/// Record of a single sequential filter update step.
///
/// Contains all intermediate products for analysis of filter performance,
/// including pre-fit and post-fit states, residuals, covariances, and Kalman gain.
#[derive(Clone, Debug)]
pub struct FilterRecord {
    /// Epoch of this record
    pub epoch: Epoch,

    /// State before measurement update (after prediction)
    pub state_predicted: DVector<f64>,
    /// Covariance before measurement update (after prediction)
    pub covariance_predicted: DMatrix<f64>,

    /// State after measurement update
    pub state_updated: DVector<f64>,
    /// Covariance after measurement update
    pub covariance_updated: DMatrix<f64>,

    /// Pre-fit residual: z - h(x_predicted)
    pub prefit_residual: DVector<f64>,
    /// Post-fit residual: z - h(x_updated)
    pub postfit_residual: DVector<f64>,

    /// Kalman gain matrix
    pub kalman_gain: DMatrix<f64>,

    /// Measurement model name
    pub measurement_name: String,
}
