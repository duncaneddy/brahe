/*!
 * Core trait definitions for the estimation module.
 *
 * Defines the [`MeasurementModel`] trait that all measurement models must implement.
 * This follows the same trait-based extensibility pattern used in the access module
 * (e.g., [`AccessConstraintComputer`], [`AccessPropertyComputer`]).
 *
 * [`AccessConstraintComputer`]: crate::access::AccessConstraintComputer
 * [`AccessPropertyComputer`]: crate::access::AccessPropertyComputer
 */

use nalgebra::{DMatrix, DVector};

use crate::math::jacobian::{DifferenceMethod, PerturbationStrategy, numerical_jacobian};
use crate::time::Epoch;
use crate::utils::errors::BraheError;

/// Compute measurement Jacobian H = dh/dx via finite differences.
///
/// Delegates to the shared finite-difference engine
/// [`numerical_jacobian`](crate::math::jacobian::numerical_jacobian) in
/// `math::jacobian`, which also backs the dynamics Jacobian and parameter
/// sensitivity providers, so perturbation sizing and differencing are
/// consistent across the library. The measurement Jacobian is (m × n) where
/// m = measurement_dim and n = state_dim.
///
/// This is also the default implementation of [`MeasurementModel::jacobian`],
/// with central differences and adaptive perturbation.
pub fn measurement_jacobian_numerical<M: MeasurementModel + ?Sized>(
    model: &M,
    epoch: &Epoch,
    state: &DVector<f64>,
    params: Option<&DVector<f64>>,
    method: DifferenceMethod,
    perturbation: PerturbationStrategy,
) -> Result<DMatrix<f64>, BraheError> {
    let m = model.measurement_dim();

    numerical_jacobian(
        |x| {
            let z = model.predict(epoch, x, params)?;
            // predict() is a user-extension boundary; validate the output
            // length so a mis-shaped model surfaces as an error naming the
            // model, not a dimension panic.
            if z.len() != m {
                return Err(BraheError::Error(format!(
                    "Model '{}' predict() returned {} elements, expected measurement_dim {}",
                    model.name(),
                    z.len(),
                    m
                )));
            }
            Ok(z)
        },
        state,
        method,
        perturbation,
    )
}

/// Validate the shapes of measurement model outputs against the model's
/// declared `measurement_dim` and the filter state dimension.
///
/// Measurement models are a user-extension boundary (including Python
/// subclasses); a wrong-shaped output should surface as a structured error
/// naming the model, not as an nalgebra dimension panic mid-update.
pub(crate) fn validate_model_outputs(
    model: &dyn MeasurementModel,
    measurement: &DVector<f64>,
    z_predicted: &DVector<f64>,
    h: Option<&DMatrix<f64>>,
    r: &DMatrix<f64>,
    state_dim: usize,
) -> Result<(), BraheError> {
    let m = model.measurement_dim();
    if measurement.len() != m {
        return Err(BraheError::Error(format!(
            "Observation measurement has {} elements but model '{}' declares measurement_dim {}",
            measurement.len(),
            model.name(),
            m
        )));
    }
    if z_predicted.len() != m {
        return Err(BraheError::Error(format!(
            "Model '{}' predict() returned {} elements, expected measurement_dim {}",
            model.name(),
            z_predicted.len(),
            m
        )));
    }
    if let Some(h) = h
        && (h.nrows() != m || h.ncols() != state_dim)
    {
        return Err(BraheError::Error(format!(
            "Model '{}' jacobian() returned a {}x{} matrix, expected {}x{}",
            model.name(),
            h.nrows(),
            h.ncols(),
            m,
            state_dim
        )));
    }
    if r.nrows() != m || r.ncols() != m {
        return Err(BraheError::Error(format!(
            "Model '{}' noise_covariance() is {}x{}, expected {}x{}",
            model.name(),
            r.nrows(),
            r.ncols(),
            m,
            m
        )));
    }
    Ok(())
}

/// Compute a measurement residual through the model, validating both inputs
/// and the output against `measurement_dim`.
///
/// [`MeasurementModel::residual`] is a user-extension boundary (including
/// Python subclasses and wrap-aware overrides). Both input vectors are checked
/// *before* the call so the default subtraction (and any override) can never
/// hit an nalgebra dimension panic, and the result length is checked after, so
/// no filter call site — pre-fit, post-fit, or UKF sigma-point deviation — can
/// panic on a mis-shaped measurement, prediction, or override output.
pub(crate) fn compute_residual(
    model: &dyn MeasurementModel,
    measured: &DVector<f64>,
    predicted: &DVector<f64>,
) -> Result<DVector<f64>, BraheError> {
    let m = model.measurement_dim();
    if measured.len() != m || predicted.len() != m {
        return Err(BraheError::Error(format!(
            "Model '{}' residual() inputs have lengths {} (measured) and {} (predicted), \
             expected measurement_dim {}",
            model.name(),
            measured.len(),
            predicted.len(),
            m
        )));
    }
    let residual = model.residual(measured, predicted)?;
    if residual.len() != m {
        return Err(BraheError::Error(format!(
            "Model '{}' residual() returned {} elements, expected measurement_dim {}",
            model.name(),
            residual.len(),
            m
        )));
    }
    Ok(residual)
}

/// Trait for defining measurement models used in estimation.
///
/// Implement this trait to define how observations relate to the state vector.
/// Built-in implementations are provided for GPS position, velocity, and state.
/// Custom measurement models can be defined in Rust or Python.
///
/// # Type Convention
///
/// - State vectors use SI base units (meters, m/s)
/// - Measurement vectors use SI base units
/// - Noise covariance R is in measurement units squared
///
/// # Examples
///
/// ```
/// use brahe::estimation::MeasurementModel;
/// use brahe::time::Epoch;
/// use brahe::utils::errors::BraheError;
/// use nalgebra::{DMatrix, DVector};
///
/// struct RangeModel {
///     station_ecef: DVector<f64>,
///     sigma: f64,
/// }
///
/// impl MeasurementModel for RangeModel {
///     fn predict(
///         &self,
///         _epoch: &Epoch,
///         state: &DVector<f64>,
///         _params: Option<&DVector<f64>>,
///     ) -> Result<DVector<f64>, BraheError> {
///         let range = (state.rows(0, 3) - &self.station_ecef).norm();
///         Ok(DVector::from_vec(vec![range]))
///     }
///
///     fn noise_covariance(&self) -> DMatrix<f64> {
///         DMatrix::from_element(1, 1, self.sigma * self.sigma)
///     }
///
///     fn measurement_dim(&self) -> usize { 1 }
///     fn name(&self) -> &str { "Range" }
/// }
/// ```
pub trait MeasurementModel: Send + Sync {
    /// Compute the predicted measurement from the current state.
    ///
    /// h(x, t, p) -> z_predicted
    ///
    /// # Arguments
    ///
    /// * `epoch` - Current epoch
    /// * `state` - Current state vector (meters, m/s for orbital states)
    /// * `params` - Optional parameter vector. The estimation filters pass
    ///   through the propagator's force-model / consider parameters, so
    ///   consider values (e.g., measurement biases) can affect the predicted
    ///   measurement.
    ///
    /// # Returns
    ///
    /// Predicted measurement vector
    fn predict(
        &self,
        epoch: &Epoch,
        state: &DVector<f64>,
        params: Option<&DVector<f64>>,
    ) -> Result<DVector<f64>, BraheError>;

    /// Compute the measurement Jacobian H = dh/dx.
    ///
    /// Default implementation uses central finite differences with adaptive
    /// perturbation via [`measurement_jacobian_numerical`]. Override for
    /// analytical Jacobians when available for better performance and accuracy.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Current epoch
    /// * `state` - Current state vector
    /// * `params` - Optional parameter vector (see [`predict`](Self::predict))
    ///
    /// # Returns
    ///
    /// Measurement Jacobian matrix (m x n) where m = measurement_dim, n = state_dim
    fn jacobian(
        &self,
        epoch: &Epoch,
        state: &DVector<f64>,
        params: Option<&DVector<f64>>,
    ) -> Result<DMatrix<f64>, BraheError> {
        measurement_jacobian_numerical(
            self,
            epoch,
            state,
            params,
            DifferenceMethod::Central,
            PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_value: 1.0,
            },
        )
    }

    /// Measurement noise covariance matrix R.
    ///
    /// R is constant per model: the method takes no epoch and estimators may
    /// read it once per observation or cache it. Use separate model instances
    /// (via `Observation::model_index`) for measurements with different noise.
    ///
    /// # Returns
    ///
    /// Noise covariance matrix (m x m) where m = measurement_dim
    fn noise_covariance(&self) -> DMatrix<f64>;

    /// Compute the measurement residual `measured - predicted`.
    ///
    /// The default implementation is plain vector subtraction. Models with
    /// angular components that wrap (e.g., azimuth crossing 0°/360°) should
    /// override this to wrap the corresponding residual components so
    /// residuals near the discontinuity stay small. All estimators (EKF,
    /// UKF, BLS) compute pre-fit and post-fit residuals through this method.
    ///
    /// # Arguments
    ///
    /// * `measured` - Observed measurement vector z
    /// * `predicted` - Predicted measurement vector h(x)
    ///
    /// # Returns
    ///
    /// Residual vector (m elements)
    ///
    /// # Errors
    ///
    /// Returns an error if a custom override fails (e.g., a Python subclass
    /// whose `residual()` raises). The default implementation is infallible.
    fn residual(
        &self,
        measured: &DVector<f64>,
        predicted: &DVector<f64>,
    ) -> Result<DVector<f64>, BraheError> {
        Ok(measured - predicted)
    }

    /// Dimension of the measurement vector.
    fn measurement_dim(&self) -> usize;

    /// Human-readable name for this measurement model.
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::InertialPositionMeasurementModel;
    use crate::time::TimeSystem;
    use approx::assert_abs_diff_eq;

    fn test_epoch() -> Epoch {
        Epoch::from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem::UTC)
    }

    /// A 6D state vector at LEO altitude
    fn test_state() -> DVector<f64> {
        DVector::from_vec(vec![6878.0e3, 1000.0e3, 500.0e3, 0.0, 7500.0, 100.0])
    }

    /// Analytical Jacobian for InertialPositionMeasurementModel is [I_3 | 0_3]
    fn expected_jacobian() -> DMatrix<f64> {
        let mut h = DMatrix::zeros(3, 6);
        h[(0, 0)] = 1.0;
        h[(1, 1)] = 1.0;
        h[(2, 2)] = 1.0;
        h
    }

    #[test]
    fn test_numerical_jacobian_central_difference() {
        let model = InertialPositionMeasurementModel::new(10.0);
        let epoch = test_epoch();
        let state = test_state();
        let expected = expected_jacobian();

        let h = measurement_jacobian_numerical(
            &model,
            &epoch,
            &state,
            None,
            DifferenceMethod::Central,
            PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_value: 1.0,
            },
        )
        .unwrap();

        assert_eq!(h.nrows(), 3);
        assert_eq!(h.ncols(), 6);
        assert_abs_diff_eq!(h, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_numerical_jacobian_forward_difference() {
        let model = InertialPositionMeasurementModel::new(10.0);
        let epoch = test_epoch();
        let state = test_state();
        let expected = expected_jacobian();

        let h = measurement_jacobian_numerical(
            &model,
            &epoch,
            &state,
            None,
            DifferenceMethod::Forward,
            PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_value: 1.0,
            },
        )
        .unwrap();

        assert_abs_diff_eq!(h, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_numerical_jacobian_backward_difference() {
        let model = InertialPositionMeasurementModel::new(10.0);
        let epoch = test_epoch();
        let state = test_state();
        let expected = expected_jacobian();

        let h = measurement_jacobian_numerical(
            &model,
            &epoch,
            &state,
            None,
            DifferenceMethod::Backward,
            PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_value: 1.0,
            },
        )
        .unwrap();

        assert_abs_diff_eq!(h, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_numerical_jacobian_perturbation_strategies() {
        let model = InertialPositionMeasurementModel::new(10.0);
        let epoch = test_epoch();
        let state = test_state();
        let expected = expected_jacobian();

        // Fixed perturbation
        let h_fixed = measurement_jacobian_numerical(
            &model,
            &epoch,
            &state,
            None,
            DifferenceMethod::Central,
            PerturbationStrategy::Fixed(1.0),
        )
        .unwrap();
        assert_abs_diff_eq!(h_fixed, expected, epsilon = 1e-8);

        // Percentage perturbation (needs all non-zero state components to avoid 0/0)
        let nonzero_state =
            DVector::from_vec(vec![6878.0e3, 1000.0e3, 500.0e3, 100.0, 7500.0, 100.0]);
        let h_pct = measurement_jacobian_numerical(
            &model,
            &epoch,
            &nonzero_state,
            None,
            DifferenceMethod::Central,
            PerturbationStrategy::Percentage(1e-6),
        )
        .unwrap();
        assert_abs_diff_eq!(h_pct, expected, epsilon = 1e-8);

        // Adaptive perturbation with different parameters
        let h_adaptive = measurement_jacobian_numerical(
            &model,
            &epoch,
            &state,
            None,
            DifferenceMethod::Central,
            PerturbationStrategy::Adaptive {
                scale_factor: 2.0,
                min_value: 0.1,
            },
        )
        .unwrap();
        assert_abs_diff_eq!(h_adaptive, expected, epsilon = 1e-8);
    }

    #[test]
    fn test_residual_default_is_subtraction() {
        let model = InertialPositionMeasurementModel::new(10.0);
        let measured = DVector::from_vec(vec![10.0, 20.0, 30.0]);
        let predicted = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let r = model.residual(&measured, &predicted).unwrap();
        assert_abs_diff_eq!(r, DVector::from_vec(vec![9.0, 18.0, 27.0]), epsilon = 1e-12);
    }

    #[test]
    fn test_compute_residual_rejects_wrong_input_lengths() {
        // compute_residual must validate both inputs before any subtraction so
        // no filter call site (pre-fit, post-fit, or UKF sigma-point deviation)
        // can panic in nalgebra on a mis-shaped measurement or prediction.
        let model = InertialPositionMeasurementModel::new(10.0); // measurement_dim 3
        let good = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let bad = DVector::from_vec(vec![1.0, 2.0]);

        // Wrong measured length
        let e = compute_residual(&model, &bad, &good).unwrap_err();
        assert!(e.to_string().contains("residual() inputs"), "{}", e);
        // Wrong predicted length
        assert!(compute_residual(&model, &good, &bad).is_err());
        // Both correct
        let r = compute_residual(&model, &good, &good).unwrap();
        assert_eq!(r.len(), 3);
    }
}
