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

use nalgebra::{DMatrix, DVector};

use crate::estimation::traits::MeasurementModel;
use crate::math::covariance::{
    covariance_from_upper_triangular, diagonal_covariance, isotropic_covariance,
    validate_covariance,
};
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
    noise_cov: DMatrix<f64>,
}

impl InertialPositionMeasurementModel {
    /// Create an inertial position model with isotropic noise.
    ///
    /// # Arguments
    ///
    /// * `sigma` - Position noise standard deviation (meters), applied to all axes
    pub fn new(sigma: f64) -> Self {
        Self {
            noise_cov: isotropic_covariance(3, sigma),
        }
    }

    /// Create an inertial position model with per-axis noise.
    ///
    /// # Arguments
    ///
    /// * `sigma_x` - X-axis position noise standard deviation (meters)
    /// * `sigma_y` - Y-axis position noise standard deviation (meters)
    /// * `sigma_z` - Z-axis position noise standard deviation (meters)
    pub fn new_per_axis(sigma_x: f64, sigma_y: f64, sigma_z: f64) -> Self {
        Self {
            noise_cov: diagonal_covariance(&[sigma_x, sigma_y, sigma_z]),
        }
    }

    /// Create an inertial position model from a full 3×3 noise covariance matrix.
    ///
    /// Allows specifying correlated measurement noise (off-diagonal terms).
    /// The matrix must be 3×3 and symmetric.
    ///
    /// # Arguments
    ///
    /// * `noise_cov` - 3×3 noise covariance matrix (meters²)
    pub fn from_covariance(noise_cov: DMatrix<f64>) -> Result<Self, BraheError> {
        let cov = validate_covariance(noise_cov)?;
        if cov.nrows() != 3 {
            return Err(BraheError::Error(format!(
                "InertialPositionMeasurementModel requires 3x3 covariance, got {}x{}",
                cov.nrows(),
                cov.ncols()
            )));
        }
        Ok(Self { noise_cov: cov })
    }

    /// Create an inertial position model from upper-triangular covariance elements.
    ///
    /// Elements are in row-major packed order: `[c₀₀, c₀₁, c₀₂, c₁₁, c₁₂, c₂₂]`
    /// (6 elements for a 3×3 matrix).
    ///
    /// # Arguments
    ///
    /// * `upper` - Upper-triangular elements in row-major packed order (meters²)
    pub fn from_upper_triangular(upper: &[f64]) -> Result<Self, BraheError> {
        Ok(Self {
            noise_cov: covariance_from_upper_triangular(3, upper)?,
        })
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
    noise_cov: DMatrix<f64>,
}

impl InertialVelocityMeasurementModel {
    /// Create an inertial velocity model with isotropic noise.
    ///
    /// # Arguments
    ///
    /// * `sigma` - Velocity noise standard deviation (m/s), applied to all axes
    pub fn new(sigma: f64) -> Self {
        Self {
            noise_cov: isotropic_covariance(3, sigma),
        }
    }

    /// Create an inertial velocity model with per-axis noise.
    ///
    /// # Arguments
    ///
    /// * `sigma_x` - X-axis velocity noise standard deviation (m/s)
    /// * `sigma_y` - Y-axis velocity noise standard deviation (m/s)
    /// * `sigma_z` - Z-axis velocity noise standard deviation (m/s)
    pub fn new_per_axis(sigma_x: f64, sigma_y: f64, sigma_z: f64) -> Self {
        Self {
            noise_cov: diagonal_covariance(&[sigma_x, sigma_y, sigma_z]),
        }
    }

    /// Create an inertial velocity model from a full 3×3 noise covariance matrix.
    ///
    /// # Arguments
    ///
    /// * `noise_cov` - 3×3 noise covariance matrix ((m/s)²)
    pub fn from_covariance(noise_cov: DMatrix<f64>) -> Result<Self, BraheError> {
        let cov = validate_covariance(noise_cov)?;
        if cov.nrows() != 3 {
            return Err(BraheError::Error(format!(
                "InertialVelocityMeasurementModel requires 3x3 covariance, got {}x{}",
                cov.nrows(),
                cov.ncols()
            )));
        }
        Ok(Self { noise_cov: cov })
    }

    /// Create an inertial velocity model from upper-triangular covariance elements.
    ///
    /// Elements are in row-major packed order: `[c₀₀, c₀₁, c₀₂, c₁₁, c₁₂, c₂₂]`
    /// (6 elements for a 3×3 matrix).
    ///
    /// # Arguments
    ///
    /// * `upper` - Upper-triangular elements in row-major packed order ((m/s)²)
    pub fn from_upper_triangular(upper: &[f64]) -> Result<Self, BraheError> {
        Ok(Self {
            noise_cov: covariance_from_upper_triangular(3, upper)?,
        })
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
        Self {
            noise_cov: diagonal_covariance(&[
                pos_sigma_x,
                pos_sigma_y,
                pos_sigma_z,
                vel_sigma_x,
                vel_sigma_y,
                vel_sigma_z,
            ]),
        }
    }

    /// Create an inertial state model from a full 6×6 noise covariance matrix.
    ///
    /// Allows specifying correlated measurement noise (off-diagonal terms),
    /// including position-velocity cross-correlations.
    ///
    /// # Arguments
    ///
    /// * `noise_cov` - 6×6 noise covariance matrix
    pub fn from_covariance(noise_cov: DMatrix<f64>) -> Result<Self, BraheError> {
        let cov = validate_covariance(noise_cov)?;
        if cov.nrows() != 6 {
            return Err(BraheError::Error(format!(
                "InertialStateMeasurementModel requires 6x6 covariance, got {}x{}",
                cov.nrows(),
                cov.ncols()
            )));
        }
        Ok(Self { noise_cov: cov })
    }

    /// Create an inertial state model from upper-triangular covariance elements.
    ///
    /// Elements are in row-major packed order (21 elements for a 6×6 matrix).
    ///
    /// # Arguments
    ///
    /// * `upper` - Upper-triangular elements in row-major packed order
    pub fn from_upper_triangular(upper: &[f64]) -> Result<Self, BraheError> {
        Ok(Self {
            noise_cov: covariance_from_upper_triangular(6, upper)?,
        })
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::traits::measurement_jacobian_numerical;
    use crate::math::jacobian::{DifferenceMethod, PerturbationStrategy};
    use crate::time::TimeSystem;
    use approx::assert_abs_diff_eq;

    fn test_epoch() -> Epoch {
        Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC)
    }

    fn test_state() -> DVector<f64> {
        DVector::from_vec(vec![6878.0e3, 1000.0e3, 500.0e3, 100.0, 7500.0, 200.0])
    }

    #[test]
    fn test_inertial_position_jacobian_matches_numerical() {
        let model = InertialPositionMeasurementModel::new(10.0);
        let epoch = test_epoch();
        let state = test_state();

        let analytical = model.jacobian(&epoch, &state, None).unwrap();
        let numerical = measurement_jacobian_numerical(
            &model,
            &epoch,
            &state,
            None,
            DifferenceMethod::Central,
            PerturbationStrategy::Fixed(1.0),
        )
        .unwrap();

        assert_eq!(analytical.nrows(), 3);
        assert_eq!(analytical.ncols(), 6);
        assert_abs_diff_eq!(analytical, numerical, epsilon = 1e-8);

        // Verify structure: [I_3 | 0_3]
        assert_abs_diff_eq!(analytical[(0, 0)], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(analytical[(1, 1)], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(analytical[(2, 2)], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(analytical[(0, 3)], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_inertial_velocity_jacobian_matches_numerical() {
        let model = InertialVelocityMeasurementModel::new(0.05);
        let epoch = test_epoch();
        let state = test_state();

        let analytical = model.jacobian(&epoch, &state, None).unwrap();
        let numerical = measurement_jacobian_numerical(
            &model,
            &epoch,
            &state,
            None,
            DifferenceMethod::Central,
            PerturbationStrategy::Fixed(1.0),
        )
        .unwrap();

        assert_eq!(analytical.nrows(), 3);
        assert_eq!(analytical.ncols(), 6);
        assert_abs_diff_eq!(analytical, numerical, epsilon = 1e-8);

        // Verify structure: [0_3 | I_3]
        assert_abs_diff_eq!(analytical[(0, 0)], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(analytical[(0, 3)], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(analytical[(1, 4)], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(analytical[(2, 5)], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_inertial_state_jacobian_matches_numerical() {
        let model = InertialStateMeasurementModel::new(5.0, 0.05);
        let epoch = test_epoch();
        let state = test_state();

        let analytical = model.jacobian(&epoch, &state, None).unwrap();
        let numerical = measurement_jacobian_numerical(
            &model,
            &epoch,
            &state,
            None,
            DifferenceMethod::Central,
            PerturbationStrategy::Fixed(1.0),
        )
        .unwrap();

        assert_eq!(analytical.nrows(), 6);
        assert_eq!(analytical.ncols(), 6);
        assert_abs_diff_eq!(analytical, numerical, epsilon = 1e-8);

        // Verify structure: I_6
        for i in 0..6 {
            assert_abs_diff_eq!(analytical[(i, i)], 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_inertial_position_constructors() {
        let m = InertialPositionMeasurementModel::new(10.0);
        assert_eq!(m.measurement_dim(), 3);
        assert_eq!(m.name(), "InertialPosition");

        let m = InertialPositionMeasurementModel::new_per_axis(1.0, 2.0, 3.0);
        let r = m.noise_covariance();
        assert_abs_diff_eq!(r[(0, 0)], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(1, 1)], 4.0, epsilon = 1e-12);

        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 2.0, 3.0]));
        let m = InertialPositionMeasurementModel::from_covariance(cov).unwrap();
        assert_eq!(m.noise_covariance().nrows(), 3);

        let m = InertialPositionMeasurementModel::from_upper_triangular(&[
            1.0, 0.0, 0.0, 2.0, 0.0, 3.0,
        ])
        .unwrap();
        assert_abs_diff_eq!(m.noise_covariance()[(1, 1)], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_inertial_velocity_constructors() {
        let m = InertialVelocityMeasurementModel::new(0.05);
        assert_eq!(m.measurement_dim(), 3);
        assert_eq!(m.name(), "InertialVelocity");

        let m = InertialVelocityMeasurementModel::new_per_axis(0.01, 0.02, 0.03);
        let r = m.noise_covariance();
        assert_abs_diff_eq!(r[(0, 0)], 0.0001, epsilon = 1e-12);

        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![0.01, 0.02, 0.03]));
        let m = InertialVelocityMeasurementModel::from_covariance(cov).unwrap();
        assert_eq!(m.noise_covariance().nrows(), 3);

        let m = InertialVelocityMeasurementModel::from_upper_triangular(&[
            0.01, 0.0, 0.0, 0.02, 0.0, 0.03,
        ])
        .unwrap();
        assert_abs_diff_eq!(m.noise_covariance()[(2, 2)], 0.03, epsilon = 1e-12);
    }

    #[test]
    fn test_inertial_state_constructors() {
        let m = InertialStateMeasurementModel::new(5.0, 0.05);
        assert_eq!(m.measurement_dim(), 6);
        assert_eq!(m.name(), "InertialState");

        let m = InertialStateMeasurementModel::new_per_axis(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        let r = m.noise_covariance();
        assert_abs_diff_eq!(r[(0, 0)], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(5, 5)], 0.09, epsilon = 1e-12);

        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0; 6]));
        let m = InertialStateMeasurementModel::from_covariance(cov).unwrap();
        assert_eq!(m.noise_covariance().nrows(), 6);

        let upper: Vec<f64> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
        ];
        let m = InertialStateMeasurementModel::from_upper_triangular(&upper).unwrap();
        assert_eq!(m.noise_covariance().nrows(), 6);
    }

    #[test]
    fn test_inertial_position_predict() {
        let model = InertialPositionMeasurementModel::new(10.0);
        let epoch = test_epoch();
        let state = test_state();
        let z = model.predict(&epoch, &state, None).unwrap();
        assert_eq!(z.len(), 3);
        assert_abs_diff_eq!(z[0], state[0], epsilon = 1e-12);
        assert_abs_diff_eq!(z[1], state[1], epsilon = 1e-12);
        assert_abs_diff_eq!(z[2], state[2], epsilon = 1e-12);
    }

    #[test]
    fn test_inertial_velocity_predict() {
        let model = InertialVelocityMeasurementModel::new(0.05);
        let epoch = test_epoch();
        let state = test_state();
        let z = model.predict(&epoch, &state, None).unwrap();
        assert_eq!(z.len(), 3);
        assert_abs_diff_eq!(z[0], state[3], epsilon = 1e-12);
        assert_abs_diff_eq!(z[1], state[4], epsilon = 1e-12);
        assert_abs_diff_eq!(z[2], state[5], epsilon = 1e-12);
    }

    #[test]
    fn test_inertial_state_predict() {
        let model = InertialStateMeasurementModel::new(5.0, 0.05);
        let epoch = test_epoch();
        let state = test_state();
        let z = model.predict(&epoch, &state, None).unwrap();
        assert_eq!(z.len(), 6);
        for i in 0..6 {
            assert_abs_diff_eq!(z[i], state[i], epsilon = 1e-12);
        }
    }
}
