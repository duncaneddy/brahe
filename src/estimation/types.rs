/*!
 * Data structures for observations, filter records, and batch estimation records.
 *
 * Defines the core types used to represent observations, sequential filter update
 * records (EKF/UKF), and batch estimation diagnostic records (BLS).
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

/// Record of a single batch least squares iteration.
#[derive(Clone, Debug)]
pub struct BLSIterationRecord {
    /// Iteration number (0-indexed)
    pub iteration: usize,
    /// Reference epoch for this iteration
    pub epoch: Epoch,
    /// State estimate at this iteration
    pub state: DVector<f64>,
    /// Covariance at this iteration (formal, solve-for only)
    pub covariance: DMatrix<f64>,
    /// State correction δx applied at this iteration
    pub state_correction: DVector<f64>,
    /// Norm of the state correction ||δx||
    pub state_correction_norm: f64,
    /// Cost function value J at this iteration
    pub cost: f64,
    /// RMS of all pre-fit residuals at this iteration
    pub rms_prefit_residual: f64,
    /// RMS of all post-fit residuals at this iteration
    pub rms_postfit_residual: f64,
}

/// Per-observation residual from a batch least squares iteration.
#[derive(Clone, Debug)]
pub struct BLSObservationResidual {
    /// Epoch of the observation
    pub epoch: Epoch,
    /// Name of the measurement model used
    pub model_name: String,
    /// Pre-fit residual: y - h(x_k, t) before state correction
    pub prefit_residual: DVector<f64>,
    /// Post-fit residual: y - h(x_{k+1}, t) after state correction
    pub postfit_residual: DVector<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::TimeSystem;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_bls_iteration_record_construction() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0]);
        let cov = DMatrix::identity(6, 6);
        let correction = DVector::from_vec(vec![100.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        let record = BLSIterationRecord {
            iteration: 0,
            epoch,
            state: state.clone(),
            covariance: cov,
            state_correction: correction.clone(),
            state_correction_norm: correction.norm(),
            cost: 1234.5,
            rms_prefit_residual: 50.0,
            rms_postfit_residual: 5.0,
        };

        assert_eq!(record.iteration, 0);
        assert_eq!(record.state_correction_norm, 100.0);
        assert_eq!(record.cost, 1234.5);
    }

    #[test]
    fn test_bls_observation_residual_construction() {
        let epoch = Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let residual = BLSObservationResidual {
            epoch,
            model_name: "InertialPosition".to_string(),
            prefit_residual: DVector::from_vec(vec![10.0, 20.0, 30.0]),
            postfit_residual: DVector::from_vec(vec![0.1, 0.2, 0.3]),
        };

        assert_eq!(residual.model_name, "InertialPosition");
        assert_eq!(residual.prefit_residual.len(), 3);
    }
}
