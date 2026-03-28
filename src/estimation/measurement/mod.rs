/*!
 * Built-in measurement models for common observation types.
 *
 * Models are organized by the frame in which measurements are expressed:
 *
 * - **[`inertial`]**: Direct observations of the inertial (ECI) state vector.
 *   Analytical Jacobians (identity sub-matrices).
 *
 * - **[`ecef`]**: Measurements in the Earth-fixed (ECEF) frame.
 *   Internally converts ECI→ECEF. Uses finite-difference Jacobians since the
 *   rotation is epoch-dependent.
 *
 * New measurement models (e.g., range, range-rate, Doppler) can be added as
 * separate files in this module.
 *
 * [`MeasurementModel`]: crate::estimation::MeasurementModel
 */

mod ecef;
mod inertial;

pub use ecef::{
    EcefPositionMeasurementModel, EcefStateMeasurementModel, EcefVelocityMeasurementModel,
};
pub use inertial::{
    InertialPositionMeasurementModel, InertialStateMeasurementModel,
    InertialVelocityMeasurementModel,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eop::{EOPExtrapolation, FileEOPProvider, set_global_eop_provider};
    use crate::estimation::traits::MeasurementModel;
    use crate::frames::state_eci_to_ecef;
    use crate::math::linalg::SVector6;
    use crate::time::Epoch;
    use crate::utils::errors::BraheError;
    use approx::assert_abs_diff_eq;
    use nalgebra::{DMatrix, DVector};
    use serial_test::serial;

    fn setup_global_test_eop() {
        let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
        set_global_eop_provider(eop);
    }

    fn test_state() -> DVector<f64> {
        DVector::from_vec(vec![
            6878.0e3, 100.0e3, 50.0e3, // position [m]
            0.0, 7612.0, 100.0, // velocity [m/s]
        ])
    }

    fn test_epoch() -> Epoch {
        Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, crate::time::TimeSystem::UTC)
    }

    // =========================================================================
    // InertialPositionMeasurementModel tests
    // =========================================================================

    #[test]
    fn test_inertial_position_predict() {
        let model = InertialPositionMeasurementModel::new(10.0);
        let state = test_state();
        let epoch = test_epoch();

        let z = model.predict(&epoch, &state, None).unwrap();
        assert_eq!(z.len(), 3);
        assert_abs_diff_eq!(z[0], 6878.0e3, epsilon = 1e-10);
        assert_abs_diff_eq!(z[1], 100.0e3, epsilon = 1e-10);
        assert_abs_diff_eq!(z[2], 50.0e3, epsilon = 1e-10);
    }

    #[test]
    fn test_inertial_position_jacobian() {
        let model = InertialPositionMeasurementModel::new(10.0);
        let state = test_state();
        let epoch = test_epoch();

        let h = model.jacobian(&epoch, &state, None).unwrap();
        assert_eq!(h.nrows(), 3);
        assert_eq!(h.ncols(), 6);

        // Should be [I_3 | 0_3x3]
        assert_abs_diff_eq!(h[(0, 0)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(h[(1, 1)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(h[(2, 2)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(h[(0, 3)], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inertial_position_noise_covariance() {
        let model = InertialPositionMeasurementModel::new(10.0);
        let r = model.noise_covariance();
        assert_eq!(r.nrows(), 3);
        assert_abs_diff_eq!(r[(0, 0)], 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(0, 1)], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inertial_position_per_axis() {
        let model = InertialPositionMeasurementModel::new_per_axis(5.0, 10.0, 15.0);
        let r = model.noise_covariance();
        assert_abs_diff_eq!(r[(0, 0)], 25.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(1, 1)], 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(2, 2)], 225.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inertial_position_name_and_dim() {
        let model = InertialPositionMeasurementModel::new(10.0);
        assert_eq!(model.name(), "InertialPosition");
        assert_eq!(model.measurement_dim(), 3);
    }

    #[test]
    fn test_inertial_position_analytical_jacobian_correctness() {
        let model = InertialPositionMeasurementModel::new(10.0);
        let state = test_state();
        let epoch = test_epoch();

        let h = model.jacobian(&epoch, &state, None).unwrap();
        let dx = DVector::from_vec(vec![1.0, 2.0, 3.0, 0.1, 0.2, 0.3]);
        let z_base = model.predict(&epoch, &state, None).unwrap();
        let z_pert = model.predict(&epoch, &(&state + &dx), None).unwrap();
        let dz_actual = &z_pert - &z_base;
        let dz_predicted = &h * &dx;
        for i in 0..3 {
            assert_abs_diff_eq!(dz_actual[i], dz_predicted[i], epsilon = 1e-10);
        }
    }

    // =========================================================================
    // InertialVelocityMeasurementModel tests
    // =========================================================================

    #[test]
    fn test_inertial_velocity_predict() {
        let model = InertialVelocityMeasurementModel::new(0.1);
        let state = test_state();
        let epoch = test_epoch();

        let z = model.predict(&epoch, &state, None).unwrap();
        assert_eq!(z.len(), 3);
        assert_abs_diff_eq!(z[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(z[1], 7612.0, epsilon = 1e-10);
        assert_abs_diff_eq!(z[2], 100.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inertial_velocity_jacobian() {
        let model = InertialVelocityMeasurementModel::new(0.1);
        let state = test_state();
        let epoch = test_epoch();

        let h = model.jacobian(&epoch, &state, None).unwrap();
        assert_abs_diff_eq!(h[(0, 3)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(h[(1, 4)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(h[(2, 5)], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(h[(0, 0)], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inertial_velocity_name_and_dim() {
        let model = InertialVelocityMeasurementModel::new(0.1);
        assert_eq!(model.name(), "InertialVelocity");
        assert_eq!(model.measurement_dim(), 3);
    }

    // =========================================================================
    // InertialStateMeasurementModel tests
    // =========================================================================

    #[test]
    fn test_inertial_state_predict() {
        let model = InertialStateMeasurementModel::new(10.0, 0.1);
        let state = test_state();
        let epoch = test_epoch();

        let z = model.predict(&epoch, &state, None).unwrap();
        assert_eq!(z.len(), 6);
        assert_abs_diff_eq!(z[0], 6878.0e3, epsilon = 1e-10);
        assert_abs_diff_eq!(z[4], 7612.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inertial_state_jacobian_is_identity() {
        let model = InertialStateMeasurementModel::new(10.0, 0.1);
        let state = test_state();
        let epoch = test_epoch();

        let h = model.jacobian(&epoch, &state, None).unwrap();
        for i in 0..6 {
            for j in 0..6 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(h[(i, j)], expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_inertial_state_noise_covariance() {
        let model = InertialStateMeasurementModel::new(10.0, 0.1);
        let r = model.noise_covariance();
        assert_abs_diff_eq!(r[(0, 0)], 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(3, 3)], 0.01, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(0, 3)], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inertial_state_per_axis() {
        let model = InertialStateMeasurementModel::new_per_axis(5.0, 10.0, 15.0, 0.05, 0.1, 0.15);
        let r = model.noise_covariance();
        assert_abs_diff_eq!(r[(0, 0)], 25.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(5, 5)], 0.0225, epsilon = 1e-10);
    }

    #[test]
    fn test_inertial_state_name_and_dim() {
        let model = InertialStateMeasurementModel::new(10.0, 0.1);
        assert_eq!(model.name(), "InertialState");
        assert_eq!(model.measurement_dim(), 6);
    }

    // =========================================================================
    // Extended state & error tests
    // =========================================================================

    #[test]
    fn test_inertial_position_extended_state() {
        let model = InertialPositionMeasurementModel::new(10.0);
        let state = DVector::from_vec(vec![6878.0e3, 100.0e3, 50.0e3, 0.0, 7612.0, 100.0, 1000.0]);
        let epoch = test_epoch();

        let h = model.jacobian(&epoch, &state, None).unwrap();
        assert_eq!(h.ncols(), 7);
        assert_abs_diff_eq!(h[(0, 6)], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inertial_position_state_too_short() {
        let model = InertialPositionMeasurementModel::new(10.0);
        let state = DVector::from_vec(vec![1.0, 2.0]);
        let epoch = test_epoch();
        assert!(model.predict(&epoch, &state, None).is_err());
    }

    #[test]
    fn test_inertial_velocity_state_too_short() {
        let model = InertialVelocityMeasurementModel::new(0.1);
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let epoch = test_epoch();
        assert!(model.predict(&epoch, &state, None).is_err());
    }

    // =========================================================================
    // Default Jacobian (finite difference) test
    // =========================================================================

    #[test]
    fn test_default_jacobian_finite_difference() {
        struct RangeModel;
        impl MeasurementModel for RangeModel {
            fn predict(
                &self,
                _epoch: &Epoch,
                state: &DVector<f64>,
                _params: Option<&DVector<f64>>,
            ) -> Result<DVector<f64>, BraheError> {
                let range =
                    (state[0] * state[0] + state[1] * state[1] + state[2] * state[2]).sqrt();
                Ok(DVector::from_vec(vec![range]))
            }
            fn noise_covariance(&self) -> DMatrix<f64> {
                DMatrix::from_element(1, 1, 100.0)
            }
            fn measurement_dim(&self) -> usize {
                1
            }
            fn name(&self) -> &str {
                "Range"
            }
        }

        let model = RangeModel;
        let state = test_state();
        let epoch = test_epoch();

        let h = model.jacobian(&epoch, &state, None).unwrap();
        let r = (state[0] * state[0] + state[1] * state[1] + state[2] * state[2]).sqrt();
        assert_abs_diff_eq!(h[(0, 0)], state[0] / r, epsilon = 1e-5);
        assert_abs_diff_eq!(h[(0, 1)], state[1] / r, epsilon = 1e-5);
        assert_abs_diff_eq!(h[(0, 2)], state[2] / r, epsilon = 1e-5);
        assert_abs_diff_eq!(h[(0, 3)], 0.0, epsilon = 1e-5);
    }

    // =========================================================================
    // GNSS measurement model tests
    // =========================================================================

    #[test]
    #[serial]
    fn test_gnss_position_predict() {
        setup_global_test_eop();

        let model = EcefPositionMeasurementModel::new(5.0);
        let state = test_state();
        let epoch = test_epoch();

        let z = model.predict(&epoch, &state, None).unwrap();
        assert_eq!(z.len(), 3);

        // The ECEF position should differ from ECI due to Earth rotation
        // but have the same magnitude
        let eci_pos_mag = (state[0] * state[0] + state[1] * state[1] + state[2] * state[2]).sqrt();
        let ecef_pos_mag = (z[0] * z[0] + z[1] * z[1] + z[2] * z[2]).sqrt();
        assert_abs_diff_eq!(eci_pos_mag, ecef_pos_mag, epsilon = 1e-3);
    }

    #[test]
    fn test_gnss_position_name_and_dim() {
        let model = EcefPositionMeasurementModel::new(5.0);
        assert_eq!(model.name(), "EcefPosition");
        assert_eq!(model.measurement_dim(), 3);
    }

    #[test]
    #[serial]
    fn test_gnss_state_predict() {
        setup_global_test_eop();

        let model = EcefStateMeasurementModel::new(5.0, 0.05);
        let state = test_state();
        let epoch = test_epoch();

        let z = model.predict(&epoch, &state, None).unwrap();
        assert_eq!(z.len(), 6);

        // Compare with direct state_eci_to_ecef
        let state_eci = SVector6::new(state[0], state[1], state[2], state[3], state[4], state[5]);
        let state_ecef = state_eci_to_ecef(epoch, state_eci);
        for i in 0..6 {
            assert_abs_diff_eq!(z[i], state_ecef[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_gnss_state_name_and_dim() {
        let model = EcefStateMeasurementModel::new(5.0, 0.05);
        assert_eq!(model.name(), "EcefState");
        assert_eq!(model.measurement_dim(), 6);
    }

    #[test]
    fn test_gnss_velocity_name_and_dim() {
        let model = EcefVelocityMeasurementModel::new(0.05);
        assert_eq!(model.name(), "EcefVelocity");
        assert_eq!(model.measurement_dim(), 3);
    }

    #[test]
    fn test_gnss_position_noise() {
        let model = EcefPositionMeasurementModel::new_per_axis(3.0, 5.0, 8.0);
        let r = model.noise_covariance();
        assert_abs_diff_eq!(r[(0, 0)], 9.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(1, 1)], 25.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(2, 2)], 64.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gnss_state_too_short() {
        let model = EcefStateMeasurementModel::new(5.0, 0.05);
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let epoch = test_epoch();
        assert!(model.predict(&epoch, &state, None).is_err());
    }

    // =========================================================================
    // from_covariance / from_upper_triangular tests
    // =========================================================================

    #[test]
    fn test_inertial_position_from_covariance() {
        // Build a 3×3 covariance with off-diagonal terms
        let mut cov = DMatrix::zeros(3, 3);
        cov[(0, 0)] = 100.0;
        cov[(1, 1)] = 225.0;
        cov[(2, 2)] = 400.0;
        cov[(0, 1)] = 5.0;
        cov[(1, 0)] = 5.0;

        let model = InertialPositionMeasurementModel::from_covariance(cov.clone()).unwrap();
        let r = model.noise_covariance();
        assert_eq!(r.nrows(), 3);
        assert_abs_diff_eq!(r[(0, 0)], 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(0, 1)], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(1, 0)], 5.0, epsilon = 1e-10);
        assert_eq!(model.measurement_dim(), 3);
    }

    #[test]
    fn test_inertial_position_from_covariance_wrong_dim() {
        let cov = DMatrix::from_diagonal_element(6, 6, 100.0);
        assert!(InertialPositionMeasurementModel::from_covariance(cov).is_err());
    }

    #[test]
    fn test_inertial_position_from_upper_triangular() {
        // [c00, c01, c02, c11, c12, c22]
        let model = InertialPositionMeasurementModel::from_upper_triangular(&[
            100.0, 5.0, 0.0, 225.0, 10.0, 400.0,
        ])
        .unwrap();
        let r = model.noise_covariance();
        assert_abs_diff_eq!(r[(0, 0)], 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(0, 1)], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(1, 0)], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(1, 2)], 10.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(2, 1)], 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_inertial_state_from_covariance() {
        let cov = DMatrix::from_diagonal_element(6, 6, 50.0);
        let model = InertialStateMeasurementModel::from_covariance(cov).unwrap();
        let r = model.noise_covariance();
        assert_eq!(r.nrows(), 6);
        assert_abs_diff_eq!(r[(0, 0)], 50.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(5, 5)], 50.0, epsilon = 1e-10);
        assert_eq!(model.measurement_dim(), 6);
    }

    #[test]
    fn test_inertial_state_from_covariance_wrong_dim() {
        let cov = DMatrix::from_diagonal_element(3, 3, 100.0);
        assert!(InertialStateMeasurementModel::from_covariance(cov).is_err());
    }

    #[test]
    fn test_ecef_position_from_covariance() {
        let mut cov = DMatrix::zeros(3, 3);
        cov[(0, 0)] = 25.0;
        cov[(1, 1)] = 25.0;
        cov[(2, 2)] = 100.0; // larger vertical uncertainty
        cov[(0, 1)] = 2.0;
        cov[(1, 0)] = 2.0;

        let model = EcefPositionMeasurementModel::from_covariance(cov).unwrap();
        let r = model.noise_covariance();
        assert_abs_diff_eq!(r[(2, 2)], 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(0, 1)], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ecef_state_from_upper_triangular() {
        // 6×6 → 21 elements, all diagonal
        let mut upper = vec![0.0; 21];
        // Diagonal indices: 0, 6, 11, 15, 18, 20
        upper[0] = 100.0;
        upper[6] = 100.0;
        upper[11] = 400.0;
        upper[15] = 0.01;
        upper[18] = 0.01;
        upper[20] = 0.04;

        let model = EcefStateMeasurementModel::from_upper_triangular(&upper).unwrap();
        let r = model.noise_covariance();
        assert_eq!(r.nrows(), 6);
        assert_abs_diff_eq!(r[(0, 0)], 100.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(2, 2)], 400.0, epsilon = 1e-10);
        assert_abs_diff_eq!(r[(5, 5)], 0.04, epsilon = 1e-10);
    }
}
