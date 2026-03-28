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

use nalgebra::{DMatrix, DVector};

use crate::estimation::traits::MeasurementModel;
use crate::frames::{position_eci_to_ecef, state_eci_to_ecef};
use crate::math::covariance::{
    covariance_from_upper_triangular, diagonal_covariance, isotropic_covariance,
    validate_covariance,
};
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
#[derive(Clone, Debug)]
pub struct EcefPositionMeasurementModel {
    noise_cov: DMatrix<f64>,
}

impl EcefPositionMeasurementModel {
    /// Create an ECEF position model with isotropic noise.
    ///
    /// # Arguments
    ///
    /// * `sigma` - Position noise standard deviation (meters), applied to all axes
    pub fn new(sigma: f64) -> Self {
        Self {
            noise_cov: isotropic_covariance(3, sigma),
        }
    }

    /// Create an ECEF position model with per-axis noise.
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

    /// Create an ECEF position model from a full 3×3 noise covariance matrix.
    ///
    /// # Arguments
    ///
    /// * `noise_cov` - 3×3 noise covariance matrix (meters²)
    pub fn from_covariance(noise_cov: DMatrix<f64>) -> Result<Self, BraheError> {
        let cov = validate_covariance(noise_cov)?;
        if cov.nrows() != 3 {
            return Err(BraheError::Error(format!(
                "EcefPositionMeasurementModel requires 3x3 covariance, got {}x{}",
                cov.nrows(),
                cov.ncols()
            )));
        }
        Ok(Self { noise_cov: cov })
    }

    /// Create an ECEF position model from upper-triangular covariance elements.
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
#[derive(Clone, Debug)]
pub struct EcefVelocityMeasurementModel {
    noise_cov: DMatrix<f64>,
}

impl EcefVelocityMeasurementModel {
    /// Create an ECEF velocity model with isotropic noise.
    ///
    /// # Arguments
    ///
    /// * `sigma` - Velocity noise standard deviation (m/s), applied to all axes
    pub fn new(sigma: f64) -> Self {
        Self {
            noise_cov: isotropic_covariance(3, sigma),
        }
    }

    /// Create an ECEF velocity model with per-axis noise.
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

    /// Create an ECEF velocity model from a full 3×3 noise covariance matrix.
    ///
    /// # Arguments
    ///
    /// * `noise_cov` - 3×3 noise covariance matrix ((m/s)²)
    pub fn from_covariance(noise_cov: DMatrix<f64>) -> Result<Self, BraheError> {
        let cov = validate_covariance(noise_cov)?;
        if cov.nrows() != 3 {
            return Err(BraheError::Error(format!(
                "EcefVelocityMeasurementModel requires 3x3 covariance, got {}x{}",
                cov.nrows(),
                cov.ncols()
            )));
        }
        Ok(Self { noise_cov: cov })
    }

    /// Create an ECEF velocity model from upper-triangular covariance elements.
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
#[derive(Clone, Debug)]
pub struct EcefStateMeasurementModel {
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

    /// Create an ECEF state model from a full 6×6 noise covariance matrix.
    ///
    /// # Arguments
    ///
    /// * `noise_cov` - 6×6 noise covariance matrix
    pub fn from_covariance(noise_cov: DMatrix<f64>) -> Result<Self, BraheError> {
        let cov = validate_covariance(noise_cov)?;
        if cov.nrows() != 6 {
            return Err(BraheError::Error(format!(
                "EcefStateMeasurementModel requires 6x6 covariance, got {}x{}",
                cov.nrows(),
                cov.ncols()
            )));
        }
        Ok(Self { noise_cov: cov })
    }

    /// Create an ECEF state model from upper-triangular covariance elements.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eop::{EOPExtrapolation, FileEOPProvider, set_global_eop_provider};
    use crate::frames::{position_eci_to_ecef, state_eci_to_ecef};
    use crate::time::TimeSystem;
    use approx::assert_abs_diff_eq;
    use serial_test::serial;

    fn setup_global_test_eop() {
        let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
        set_global_eop_provider(eop);
    }

    fn test_epoch() -> Epoch {
        Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC)
    }

    /// A 6D ECI state for a LEO orbit (r ≈ 6878 km, circular)
    fn test_eci_state() -> DVector<f64> {
        let r = 6878.0e3;
        let v = (crate::constants::physical::GM_EARTH / r).sqrt();
        DVector::from_vec(vec![r, 0.0, 0.0, 0.0, v, 0.0])
    }

    // =========================================================================
    // Position model tests
    // =========================================================================

    #[test]
    fn test_ecef_position_model_constructors() {
        // Isotropic
        let m = EcefPositionMeasurementModel::new(5.0);
        assert_eq!(m.measurement_dim(), 3);
        assert_eq!(m.name(), "EcefPosition");
        let r = m.noise_covariance();
        assert_eq!(r.nrows(), 3);
        assert_abs_diff_eq!(r[(0, 0)], 25.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(1, 1)], 25.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(0, 1)], 0.0, epsilon = 1e-12);

        // Per-axis
        let m = EcefPositionMeasurementModel::new_per_axis(1.0, 2.0, 3.0);
        let r = m.noise_covariance();
        assert_abs_diff_eq!(r[(0, 0)], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(1, 1)], 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(2, 2)], 9.0, epsilon = 1e-12);

        // From covariance
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![4.0, 9.0, 16.0]));
        let m = EcefPositionMeasurementModel::from_covariance(cov).unwrap();
        let r = m.noise_covariance();
        assert_abs_diff_eq!(r[(0, 0)], 4.0, epsilon = 1e-12);

        // From upper triangular
        let m =
            EcefPositionMeasurementModel::from_upper_triangular(&[4.0, 0.0, 0.0, 9.0, 0.0, 16.0])
                .unwrap();
        let r = m.noise_covariance();
        assert_abs_diff_eq!(r[(2, 2)], 16.0, epsilon = 1e-12);
    }

    #[test]
    #[serial]
    fn test_ecef_position_model_predict() {
        setup_global_test_eop();
        let epoch = test_epoch();
        let state = test_eci_state();

        let model = EcefPositionMeasurementModel::new(5.0);
        let z = model.predict(&epoch, &state, None).unwrap();
        assert_eq!(z.len(), 3);

        // Verify against direct frame conversion
        let pos_eci = nalgebra::Vector3::new(state[0], state[1], state[2]);
        let pos_ecef = position_eci_to_ecef(epoch, pos_eci);
        assert_abs_diff_eq!(z[0], pos_ecef[0], epsilon = 1e-6);
        assert_abs_diff_eq!(z[1], pos_ecef[1], epsilon = 1e-6);
        assert_abs_diff_eq!(z[2], pos_ecef[2], epsilon = 1e-6);

        // ECEF position norm should equal ECI position norm (rotation preserves magnitude)
        let eci_norm = pos_eci.norm();
        let ecef_norm = z.norm();
        assert_abs_diff_eq!(eci_norm, ecef_norm, epsilon = 1e-6);
    }

    #[test]
    fn test_ecef_position_model_from_covariance_wrong_dim() {
        // 2x2 instead of 3x3
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0]));
        let result = EcefPositionMeasurementModel::from_covariance(cov);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("3x3"));
    }

    #[test]
    #[serial]
    fn test_ecef_position_model_jacobian() {
        setup_global_test_eop();
        let epoch = test_epoch();
        let state = test_eci_state();

        let model = EcefPositionMeasurementModel::new(5.0);
        let h = model.jacobian(&epoch, &state, None).unwrap();

        // H should be 3x6 (3 measurement dims, 6 state dims)
        assert_eq!(h.nrows(), 3);
        assert_eq!(h.ncols(), 6);

        // Position columns should be non-zero (rotation matrix entries)
        let pos_block_norm = h.view((0, 0), (3, 3)).norm();
        assert!(
            pos_block_norm > 0.1,
            "Position Jacobian block should be non-zero"
        );

        // Velocity columns should be ~zero (position doesn't depend on velocity)
        let vel_block_norm = h.view((0, 3), (3, 3)).norm();
        assert!(
            vel_block_norm < 1e-6,
            "Velocity Jacobian block should be ~zero, got {}",
            vel_block_norm
        );
    }

    // =========================================================================
    // Velocity model tests
    // =========================================================================

    #[test]
    fn test_ecef_velocity_model_constructors() {
        let m = EcefVelocityMeasurementModel::new(0.05);
        assert_eq!(m.measurement_dim(), 3);
        assert_eq!(m.name(), "EcefVelocity");
        let r = m.noise_covariance();
        assert_abs_diff_eq!(r[(0, 0)], 0.0025, epsilon = 1e-12);

        let m = EcefVelocityMeasurementModel::new_per_axis(0.01, 0.02, 0.03);
        let r = m.noise_covariance();
        assert_abs_diff_eq!(r[(0, 0)], 0.0001, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(1, 1)], 0.0004, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(2, 2)], 0.0009, epsilon = 1e-12);

        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![0.01, 0.02, 0.03]));
        let m = EcefVelocityMeasurementModel::from_covariance(cov).unwrap();
        assert_abs_diff_eq!(m.noise_covariance()[(2, 2)], 0.03, epsilon = 1e-12);

        let m =
            EcefVelocityMeasurementModel::from_upper_triangular(&[0.01, 0.0, 0.0, 0.02, 0.0, 0.03])
                .unwrap();
        assert_abs_diff_eq!(m.noise_covariance()[(1, 1)], 0.02, epsilon = 1e-12);
    }

    #[test]
    #[serial]
    fn test_ecef_velocity_model_predict() {
        setup_global_test_eop();
        let epoch = test_epoch();
        let state = test_eci_state();

        let model = EcefVelocityMeasurementModel::new(0.05);
        let z = model.predict(&epoch, &state, None).unwrap();
        assert_eq!(z.len(), 3);

        // Verify against direct frame conversion
        let state_eci = SVector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
        let state_ecef = state_eci_to_ecef(epoch, state_eci);
        assert_abs_diff_eq!(z[0], state_ecef[3], epsilon = 1e-6);
        assert_abs_diff_eq!(z[1], state_ecef[4], epsilon = 1e-6);
        assert_abs_diff_eq!(z[2], state_ecef[5], epsilon = 1e-6);
    }

    #[test]
    fn test_ecef_velocity_model_from_covariance_wrong_dim() {
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0; 6]));
        let result = EcefVelocityMeasurementModel::from_covariance(cov);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("3x3"));
    }

    #[test]
    fn test_ecef_velocity_model_predict_short_state_errors() {
        let model = EcefVelocityMeasurementModel::new(0.05);
        let short_state = DVector::from_vec(vec![1.0, 2.0, 3.0]); // only 3 elements
        let epoch = test_epoch();
        let result = model.predict(&epoch, &short_state, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains(">= 6"));
    }

    // =========================================================================
    // State model tests
    // =========================================================================

    #[test]
    fn test_ecef_state_model_constructors() {
        let m = EcefStateMeasurementModel::new(5.0, 0.05);
        assert_eq!(m.measurement_dim(), 6);
        assert_eq!(m.name(), "EcefState");
        let r = m.noise_covariance();
        assert_eq!(r.nrows(), 6);
        assert_abs_diff_eq!(r[(0, 0)], 25.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(3, 3)], 0.0025, epsilon = 1e-12);

        let m = EcefStateMeasurementModel::new_per_axis(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        let r = m.noise_covariance();
        assert_abs_diff_eq!(r[(0, 0)], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r[(5, 5)], 0.09, epsilon = 1e-12);

        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0; 6]));
        let m = EcefStateMeasurementModel::from_covariance(cov).unwrap();
        assert_eq!(m.noise_covariance().nrows(), 6);

        let upper: Vec<f64> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // row 0
            1.0, 0.0, 0.0, 0.0, 0.0, // row 1
            1.0, 0.0, 0.0, 0.0, // row 2
            1.0, 0.0, 0.0, // row 3
            1.0, 0.0, // row 4
            1.0, // row 5
        ];
        let m = EcefStateMeasurementModel::from_upper_triangular(&upper).unwrap();
        assert_eq!(m.noise_covariance().nrows(), 6);
    }

    #[test]
    #[serial]
    fn test_ecef_state_model_predict() {
        setup_global_test_eop();
        let epoch = test_epoch();
        let state = test_eci_state();

        let model = EcefStateMeasurementModel::new(5.0, 0.05);
        let z = model.predict(&epoch, &state, None).unwrap();
        assert_eq!(z.len(), 6);

        // Verify against direct frame conversion
        let state_eci = SVector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
        let state_ecef = state_eci_to_ecef(epoch, state_eci);
        for i in 0..6 {
            assert_abs_diff_eq!(z[i], state_ecef[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_ecef_state_model_from_covariance_wrong_dim() {
        let cov = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0; 3]));
        let result = EcefStateMeasurementModel::from_covariance(cov);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("6x6"));
    }

    #[test]
    fn test_ecef_state_model_predict_short_state_errors() {
        let model = EcefStateMeasurementModel::new(5.0, 0.05);
        let short_state = DVector::from_vec(vec![1.0, 2.0, 3.0]); // only 3 elements
        let epoch = test_epoch();
        let result = model.predict(&epoch, &short_state, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains(">= 6"));
    }
}
