/*!
 * Inertial-frame measurement models.
 *
 * These models directly observe components of the inertial (ECI) state vector.
 * They assume the estimator's state is in an inertial frame and produce
 * measurements in that same frame. All Jacobians are analytical (identity
 * sub-matrices).
 *
 * For ECEF-frame measurements (e.g., from a GNSS receiver), see the
 * [`ecef`](super::ecef) module.
 */

use nalgebra::{DMatrix, DVector, Vector3};

use crate::estimation::traits::MeasurementModel;
use crate::time::Epoch;
use crate::utils::errors::BraheError;

// =============================================================================
// InertialPositionMeasurementModel
// =============================================================================

/// Inertial position measurement model.
///
/// Directly observes the 3D inertial (ECI) position component of the state
/// vector with configurable Gaussian noise. The state vector is assumed to be
/// in an inertial frame with layout `[x, y, z, vx, vy, vz, ...]`.
///
/// For ECEF-frame position measurements (e.g., from a GNSS receiver), use
/// [`EcefPositionMeasurementModel`] instead.
///
/// Measurement: `z = [x, y, z]` (inertial)
/// Jacobian: `H = [I₃ₓ₃ | 0₃ₓ₍ₙ₋₃₎]`
///
/// [`EcefPositionMeasurementModel`]: crate::estimation::EcefPositionMeasurementModel
///
/// # Examples
///
/// ```
/// use brahe::estimation::{InertialPositionMeasurementModel, MeasurementModel};
///
/// // Isotropic 10m noise on all axes
/// let model = InertialPositionMeasurementModel::new(10.0);
/// assert_eq!(model.measurement_dim(), 3);
///
/// // Per-axis noise
/// let model = InertialPositionMeasurementModel::new_per_axis(10.0, 10.0, 15.0);
/// ```
#[derive(Clone)]
pub struct InertialPositionMeasurementModel {
    #[allow(dead_code)]
    noise_sigma: Vector3<f64>,
    noise_cov: DMatrix<f64>,
}

impl InertialPositionMeasurementModel {
    /// Create an inertial position model with isotropic noise.
    ///
    /// # Arguments
    ///
    /// * `sigma` - Position noise standard deviation (meters), applied to all axes
    pub fn new(sigma: f64) -> Self {
        Self::new_per_axis(sigma, sigma, sigma)
    }

    /// Create an inertial position model with per-axis noise.
    ///
    /// # Arguments
    ///
    /// * `sigma_x` - X-axis position noise standard deviation (meters)
    /// * `sigma_y` - Y-axis position noise standard deviation (meters)
    /// * `sigma_z` - Z-axis position noise standard deviation (meters)
    pub fn new_per_axis(sigma_x: f64, sigma_y: f64, sigma_z: f64) -> Self {
        let noise_sigma = Vector3::new(sigma_x, sigma_y, sigma_z);
        let noise_cov = DMatrix::from_diagonal(&DVector::from_vec(vec![
            sigma_x * sigma_x,
            sigma_y * sigma_y,
            sigma_z * sigma_z,
        ]));
        Self {
            noise_sigma,
            noise_cov,
        }
    }
}

impl MeasurementModel for InertialPositionMeasurementModel {
    fn predict(
        &self,
        _epoch: &Epoch,
        state: &DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> Result<DVector<f64>, BraheError> {
        if state.len() < 3 {
            return Err(BraheError::Error(format!(
                "InertialPositionMeasurementModel requires state dimension >= 3, got {}",
                state.len()
            )));
        }
        Ok(state.rows(0, 3).into_owned())
    }

    fn jacobian(
        &self,
        _epoch: &Epoch,
        state: &DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> Result<DMatrix<f64>, BraheError> {
        let n = state.len();
        let mut h = DMatrix::zeros(3, n);
        h[(0, 0)] = 1.0;
        h[(1, 1)] = 1.0;
        h[(2, 2)] = 1.0;
        Ok(h)
    }

    fn noise_covariance(&self) -> DMatrix<f64> {
        self.noise_cov.clone()
    }

    fn measurement_dim(&self) -> usize {
        3
    }

    fn name(&self) -> &str {
        "InertialPosition"
    }
}

// =============================================================================
// InertialVelocityMeasurementModel
// =============================================================================

/// Inertial velocity measurement model.
///
/// Directly observes the 3D inertial (ECI) velocity component of the state
/// vector with configurable Gaussian noise. The state vector is assumed to be
/// in an inertial frame with layout `[x, y, z, vx, vy, vz, ...]`.
///
/// For ECEF-frame velocity measurements (e.g., from a GNSS receiver), use
/// [`EcefVelocityMeasurementModel`] instead.
///
/// Measurement: `z = [vx, vy, vz]` (inertial)
/// Jacobian: `H = [0₃ₓ₃ | I₃ₓ₃ | 0₃ₓ₍ₙ₋₆₎]`
///
/// [`EcefVelocityMeasurementModel`]: crate::estimation::EcefVelocityMeasurementModel
///
/// # Examples
///
/// ```
/// use brahe::estimation::{InertialVelocityMeasurementModel, MeasurementModel};
///
/// // Isotropic 0.1 m/s noise
/// let model = InertialVelocityMeasurementModel::new(0.1);
/// assert_eq!(model.measurement_dim(), 3);
/// ```
#[derive(Clone)]
pub struct InertialVelocityMeasurementModel {
    #[allow(dead_code)]
    noise_sigma: Vector3<f64>,
    noise_cov: DMatrix<f64>,
}

impl InertialVelocityMeasurementModel {
    /// Create an inertial velocity model with isotropic noise.
    ///
    /// # Arguments
    ///
    /// * `sigma` - Velocity noise standard deviation (m/s), applied to all axes
    pub fn new(sigma: f64) -> Self {
        Self::new_per_axis(sigma, sigma, sigma)
    }

    /// Create an inertial velocity model with per-axis noise.
    ///
    /// # Arguments
    ///
    /// * `sigma_x` - X-axis velocity noise standard deviation (m/s)
    /// * `sigma_y` - Y-axis velocity noise standard deviation (m/s)
    /// * `sigma_z` - Z-axis velocity noise standard deviation (m/s)
    pub fn new_per_axis(sigma_x: f64, sigma_y: f64, sigma_z: f64) -> Self {
        let noise_sigma = Vector3::new(sigma_x, sigma_y, sigma_z);
        let noise_cov = DMatrix::from_diagonal(&DVector::from_vec(vec![
            sigma_x * sigma_x,
            sigma_y * sigma_y,
            sigma_z * sigma_z,
        ]));
        Self {
            noise_sigma,
            noise_cov,
        }
    }
}

impl MeasurementModel for InertialVelocityMeasurementModel {
    fn predict(
        &self,
        _epoch: &Epoch,
        state: &DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> Result<DVector<f64>, BraheError> {
        if state.len() < 6 {
            return Err(BraheError::Error(format!(
                "InertialVelocityMeasurementModel requires state dimension >= 6, got {}",
                state.len()
            )));
        }
        Ok(state.rows(3, 3).into_owned())
    }

    fn jacobian(
        &self,
        _epoch: &Epoch,
        state: &DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> Result<DMatrix<f64>, BraheError> {
        let n = state.len();
        let mut h = DMatrix::zeros(3, n);
        h[(0, 3)] = 1.0;
        h[(1, 4)] = 1.0;
        h[(2, 5)] = 1.0;
        Ok(h)
    }

    fn noise_covariance(&self) -> DMatrix<f64> {
        self.noise_cov.clone()
    }

    fn measurement_dim(&self) -> usize {
        3
    }

    fn name(&self) -> &str {
        "InertialVelocity"
    }
}

// =============================================================================
// InertialStateMeasurementModel
// =============================================================================

/// Inertial state measurement model (position + velocity).
///
/// Directly observes the full 6D inertial (ECI) state with configurable
/// Gaussian noise. The state vector is assumed to be in an inertial frame
/// with layout `[x, y, z, vx, vy, vz, ...]`.
///
/// For ECEF-frame state measurements (e.g., from a GNSS receiver), use
/// [`EcefStateMeasurementModel`] instead.
///
/// Measurement: `z = [x, y, z, vx, vy, vz]` (inertial)
/// Jacobian: `H = [I₆ₓ₆ | 0₆ₓ₍ₙ₋₆₎]`
///
/// [`EcefStateMeasurementModel`]: crate::estimation::EcefStateMeasurementModel
///
/// # Examples
///
/// ```
/// use brahe::estimation::{InertialStateMeasurementModel, MeasurementModel};
///
/// // 10m position noise, 0.1 m/s velocity noise
/// let model = InertialStateMeasurementModel::new(10.0, 0.1);
/// assert_eq!(model.measurement_dim(), 6);
/// ```
#[derive(Clone)]
pub struct InertialStateMeasurementModel {
    #[allow(dead_code)]
    pos_sigma: Vector3<f64>,
    #[allow(dead_code)]
    vel_sigma: Vector3<f64>,
    noise_cov: DMatrix<f64>,
}

impl InertialStateMeasurementModel {
    /// Create an inertial state model with isotropic noise per component type.
    ///
    /// # Arguments
    ///
    /// * `pos_sigma` - Position noise standard deviation (meters), applied to all position axes
    /// * `vel_sigma` - Velocity noise standard deviation (m/s), applied to all velocity axes
    pub fn new(pos_sigma: f64, vel_sigma: f64) -> Self {
        Self::new_per_axis(
            pos_sigma, pos_sigma, pos_sigma, vel_sigma, vel_sigma, vel_sigma,
        )
    }

    /// Create an inertial state model with per-axis noise.
    ///
    /// # Arguments
    ///
    /// * `pos_sigma_x` - X-axis position noise (meters)
    /// * `pos_sigma_y` - Y-axis position noise (meters)
    /// * `pos_sigma_z` - Z-axis position noise (meters)
    /// * `vel_sigma_x` - X-axis velocity noise (m/s)
    /// * `vel_sigma_y` - Y-axis velocity noise (m/s)
    /// * `vel_sigma_z` - Z-axis velocity noise (m/s)
    pub fn new_per_axis(
        pos_sigma_x: f64,
        pos_sigma_y: f64,
        pos_sigma_z: f64,
        vel_sigma_x: f64,
        vel_sigma_y: f64,
        vel_sigma_z: f64,
    ) -> Self {
        let pos_sigma = Vector3::new(pos_sigma_x, pos_sigma_y, pos_sigma_z);
        let vel_sigma = Vector3::new(vel_sigma_x, vel_sigma_y, vel_sigma_z);
        let noise_cov = DMatrix::from_diagonal(&DVector::from_vec(vec![
            pos_sigma_x * pos_sigma_x,
            pos_sigma_y * pos_sigma_y,
            pos_sigma_z * pos_sigma_z,
            vel_sigma_x * vel_sigma_x,
            vel_sigma_y * vel_sigma_y,
            vel_sigma_z * vel_sigma_z,
        ]));
        Self {
            pos_sigma,
            vel_sigma,
            noise_cov,
        }
    }
}

impl MeasurementModel for InertialStateMeasurementModel {
    fn predict(
        &self,
        _epoch: &Epoch,
        state: &DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> Result<DVector<f64>, BraheError> {
        if state.len() < 6 {
            return Err(BraheError::Error(format!(
                "InertialStateMeasurementModel requires state dimension >= 6, got {}",
                state.len()
            )));
        }
        Ok(state.rows(0, 6).into_owned())
    }

    fn jacobian(
        &self,
        _epoch: &Epoch,
        state: &DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> Result<DMatrix<f64>, BraheError> {
        let n = state.len();
        let mut h = DMatrix::zeros(6, n);
        for i in 0..6 {
            h[(i, i)] = 1.0;
        }
        Ok(h)
    }

    fn noise_covariance(&self) -> DMatrix<f64> {
        self.noise_cov.clone()
    }

    fn measurement_dim(&self) -> usize {
        6
    }

    fn name(&self) -> &str {
        "InertialState"
    }
}
