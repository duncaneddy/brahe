/*!
 * ECEF-frame measurement models.
 *
 * These models represent measurements expressed in the Earth-fixed (ECEF)
 * frame. The estimator's state is assumed to be in an inertial (ECI) frame,
 * so these models internally perform the ECI→ECEF rotation when computing
 * predicted measurements.
 *
 * The Jacobians use the default finite-difference implementation since the
 * ECI→ECEF rotation matrix is epoch-dependent, making analytical Jacobians
 * complex. The adaptive perturbation strategy in the default implementation
 * handles the large state magnitudes correctly.
 *
 * For direct inertial-frame measurements, see the
 * [`inertial`](super::inertial) module.
 */

use nalgebra::{DMatrix, DVector, Vector3};

use crate::estimation::traits::MeasurementModel;
use crate::frames::{position_eci_to_ecef, state_eci_to_ecef};
use crate::math::linalg::SVector6;
use crate::time::Epoch;
use crate::utils::errors::BraheError;

// =============================================================================
// EcefPositionMeasurementModel
// =============================================================================

/// GNSS position measurement model (ECEF frame).
///
/// Observes 3D position in the Earth-fixed (ECEF)
/// frame. The estimator state is assumed to be in an inertial (ECI) frame;
/// this model internally converts ECI position to ECEF for the measurement
/// prediction.
///
/// Measurement: `z = R_eci2ecef(epoch) · [x, y, z]_eci` (ECEF position)
///
/// The Jacobian is computed via finite differences (default implementation)
/// since the rotation matrix is epoch-dependent.
///
/// # Examples
///
/// ```no_run
/// use brahe::estimation::{EcefPositionMeasurementModel, MeasurementModel};
///
/// // Isotropic 5m noise (typical GNSS accuracy)
/// let model = EcefPositionMeasurementModel::new(5.0);
/// assert_eq!(model.measurement_dim(), 3);
/// ```
#[derive(Clone)]
pub struct EcefPositionMeasurementModel {
    #[allow(dead_code)]
    noise_sigma: Vector3<f64>,
    noise_cov: DMatrix<f64>,
}

impl EcefPositionMeasurementModel {
    /// Create an ECEF position model with isotropic noise.
    ///
    /// # Arguments
    ///
    /// * `sigma` - Position noise standard deviation (meters), applied to all axes
    pub fn new(sigma: f64) -> Self {
        Self::new_per_axis(sigma, sigma, sigma)
    }

    /// Create an ECEF position model with per-axis noise.
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

impl MeasurementModel for EcefPositionMeasurementModel {
    fn predict(
        &self,
        epoch: &Epoch,
        state: &DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> Result<DVector<f64>, BraheError> {
        if state.len() < 3 {
            return Err(BraheError::Error(format!(
                "EcefPositionMeasurementModel requires state dimension >= 3, got {}",
                state.len()
            )));
        }

        // Extract ECI position and convert to ECEF
        let pos_eci = nalgebra::Vector3::new(state[0], state[1], state[2]);
        let pos_ecef = position_eci_to_ecef(*epoch, pos_eci);

        Ok(DVector::from_vec(vec![
            pos_ecef[0],
            pos_ecef[1],
            pos_ecef[2],
        ]))
    }

    // Uses default finite-difference Jacobian since ECI→ECEF rotation is epoch-dependent

    fn noise_covariance(&self) -> DMatrix<f64> {
        self.noise_cov.clone()
    }

    fn measurement_dim(&self) -> usize {
        3
    }

    fn name(&self) -> &str {
        "EcefPosition"
    }
}

// =============================================================================
// EcefVelocityMeasurementModel
// =============================================================================

/// GNSS velocity measurement model (ECEF frame).
///
/// Observes 3D velocity in the Earth-fixed (ECEF)
/// frame. The estimator state is assumed to be in an inertial (ECI) frame;
/// this model internally converts the full ECI state to ECEF (accounting for
/// Earth rotation effects on velocity) and extracts the velocity component.
///
/// Measurement: `z = v_ecef` from `state_eci_to_ecef(epoch, [x,y,z,vx,vy,vz])`
///
/// # Examples
///
/// ```no_run
/// use brahe::estimation::{EcefVelocityMeasurementModel, MeasurementModel};
///
/// // Isotropic 0.05 m/s noise
/// let model = EcefVelocityMeasurementModel::new(0.05);
/// assert_eq!(model.measurement_dim(), 3);
/// ```
#[derive(Clone)]
pub struct EcefVelocityMeasurementModel {
    #[allow(dead_code)]
    noise_sigma: Vector3<f64>,
    noise_cov: DMatrix<f64>,
}

impl EcefVelocityMeasurementModel {
    /// Create an ECEF velocity model with isotropic noise.
    ///
    /// # Arguments
    ///
    /// * `sigma` - Velocity noise standard deviation (m/s), applied to all axes
    pub fn new(sigma: f64) -> Self {
        Self::new_per_axis(sigma, sigma, sigma)
    }

    /// Create an ECEF velocity model with per-axis noise.
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

impl MeasurementModel for EcefVelocityMeasurementModel {
    fn predict(
        &self,
        epoch: &Epoch,
        state: &DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> Result<DVector<f64>, BraheError> {
        if state.len() < 6 {
            return Err(BraheError::Error(format!(
                "EcefVelocityMeasurementModel requires state dimension >= 6, got {}",
                state.len()
            )));
        }

        // Convert full ECI state to ECEF (velocity conversion requires position + velocity)
        let state_eci = SVector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
        let state_ecef = state_eci_to_ecef(*epoch, state_eci);

        Ok(DVector::from_vec(vec![
            state_ecef[3],
            state_ecef[4],
            state_ecef[5],
        ]))
    }

    // Uses default finite-difference Jacobian

    fn noise_covariance(&self) -> DMatrix<f64> {
        self.noise_cov.clone()
    }

    fn measurement_dim(&self) -> usize {
        3
    }

    fn name(&self) -> &str {
        "EcefVelocity"
    }
}

// =============================================================================
// EcefStateMeasurementModel
// =============================================================================

/// GNSS state measurement model (ECEF frame, position + velocity).
///
/// Observes full 6D state (position and velocity)
/// in the Earth-fixed (ECEF) frame. The estimator state is assumed to be in
/// an inertial (ECI) frame; this model internally converts the full ECI state
/// to ECEF.
///
/// Measurement: `z = state_eci_to_ecef(epoch, [x,y,z,vx,vy,vz])`
///
/// # Examples
///
/// ```no_run
/// use brahe::estimation::{EcefStateMeasurementModel, MeasurementModel};
///
/// // 5m position noise, 0.05 m/s velocity noise
/// let model = EcefStateMeasurementModel::new(5.0, 0.05);
/// assert_eq!(model.measurement_dim(), 6);
/// ```
#[derive(Clone)]
pub struct EcefStateMeasurementModel {
    #[allow(dead_code)]
    pos_sigma: Vector3<f64>,
    #[allow(dead_code)]
    vel_sigma: Vector3<f64>,
    noise_cov: DMatrix<f64>,
}

impl EcefStateMeasurementModel {
    /// Create an ECEF state model with isotropic noise per component type.
    ///
    /// # Arguments
    ///
    /// * `pos_sigma` - Position noise standard deviation (meters)
    /// * `vel_sigma` - Velocity noise standard deviation (m/s)
    pub fn new(pos_sigma: f64, vel_sigma: f64) -> Self {
        Self::new_per_axis(
            pos_sigma, pos_sigma, pos_sigma, vel_sigma, vel_sigma, vel_sigma,
        )
    }

    /// Create an ECEF state model with per-axis noise.
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

impl MeasurementModel for EcefStateMeasurementModel {
    fn predict(
        &self,
        epoch: &Epoch,
        state: &DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> Result<DVector<f64>, BraheError> {
        if state.len() < 6 {
            return Err(BraheError::Error(format!(
                "EcefStateMeasurementModel requires state dimension >= 6, got {}",
                state.len()
            )));
        }

        let state_eci = SVector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
        let state_ecef = state_eci_to_ecef(*epoch, state_eci);

        Ok(DVector::from_iterator(6, state_ecef.iter().copied()))
    }

    // Uses default finite-difference Jacobian

    fn noise_covariance(&self) -> DMatrix<f64> {
        self.noise_cov.clone()
    }

    fn measurement_dim(&self) -> usize {
        6
    }

    fn name(&self) -> &str {
        "EcefState"
    }
}
