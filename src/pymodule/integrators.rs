/*
 * Python bindings for numerical integrators.
 *
 * Provides dynamic-sized (runtime-determined dimension) integrators for solving
 * ordinary differential equations (ODEs). Includes both fixed-step and adaptive-step
 * integration methods.
 */

// NOTE: Imports are handled by mod.rs since this file is included via include! macro

use std::sync::Arc;

use crate::integrators::{
    AdaptiveStepDIntegrator, AdaptiveStepDResult, DormandPrince54DIntegrator, FixedStepDIntegrator,
    IntegratorConfig, RK4DIntegrator, RKF45DIntegrator, RKN1210DIntegrator,
};
use crate::math::jacobian::DJacobianProvider;

// ============================================================================
// Configuration Types
// ============================================================================

/// Configuration for numerical integrators.
///
/// Controls error tolerances, step size limits, and other integration parameters.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create default configuration
///     config = bh.IntegratorConfig()
///
///     # Create fixed-step configuration
///     config = bh.IntegratorConfig.fixed_step(1.0)
///
///     # Create adaptive configuration with custom tolerances
///     config = bh.IntegratorConfig.adaptive(abs_tol=1e-9, rel_tol=1e-6)
///
///     # Customize configuration
///     config = bh.IntegratorConfig(
///         abs_tol=1e-8,
///         rel_tol=1e-6,
///         max_step=100.0,
///         min_step=0.001
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "IntegratorConfig")]
#[derive(Clone)]
pub struct PyIntegratorConfig {
    pub(crate) inner: IntegratorConfig,
}

#[pymethods]
impl PyIntegratorConfig {
    /// Create a new integrator configuration.
    ///
    /// Args:
    ///     abs_tol (float, optional): Absolute error tolerance. Defaults to 1e-6.
    ///     rel_tol (float, optional): Relative error tolerance. Defaults to 1e-3.
    ///     initial_step (float, optional): Initial step size. Defaults to None (auto).
    ///     min_step (float, optional): Minimum step size. Defaults to 1e-12.
    ///     max_step (float, optional): Maximum step size. Defaults to 900.0.
    ///     step_safety_factor (float, optional): Safety factor for step control. Defaults to 0.9.
    ///     min_step_scale_factor (float, optional): Minimum step scaling. Defaults to 0.2.
    ///     max_step_scale_factor (float, optional): Maximum step scaling. Defaults to 10.0.
    ///     max_step_attempts (int, optional): Maximum step attempts. Defaults to 10.
    ///     fixed_step_size (float, optional): Fixed step size for fixed-step integrators. Defaults to None.
    ///
    /// Returns:
    ///     IntegratorConfig: New configuration
    #[new]
    #[pyo3(signature = (
        abs_tol=1e-6,
        rel_tol=1e-3,
        initial_step=None,
        min_step=Some(1e-12),
        max_step=Some(900.0),
        step_safety_factor=Some(0.9),
        min_step_scale_factor=Some(0.2),
        max_step_scale_factor=Some(10.0),
        max_step_attempts=10,
        fixed_step_size=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        abs_tol: f64,
        rel_tol: f64,
        initial_step: Option<f64>,
        min_step: Option<f64>,
        max_step: Option<f64>,
        step_safety_factor: Option<f64>,
        min_step_scale_factor: Option<f64>,
        max_step_scale_factor: Option<f64>,
        max_step_attempts: usize,
        fixed_step_size: Option<f64>,
    ) -> Self {
        PyIntegratorConfig {
            inner: IntegratorConfig {
                abs_tol,
                rel_tol,
                initial_step,
                min_step,
                max_step,
                step_safety_factor,
                min_step_scale_factor,
                max_step_scale_factor,
                max_step_attempts,
                fixed_step_size,
            },
        }
    }

    /// Create a configuration for fixed-step integration.
    ///
    /// Args:
    ///     step_size (float): Fixed timestep in seconds
    ///
    /// Returns:
    ///     IntegratorConfig: Configuration for fixed-step integration
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     config = bh.IntegratorConfig.fixed_step(1.0)
    ///     ```
    #[classmethod]
    fn fixed_step(_cls: &Bound<'_, PyType>, step_size: f64) -> Self {
        PyIntegratorConfig {
            inner: IntegratorConfig::fixed_step(step_size),
        }
    }

    /// Create a configuration for adaptive-step integration.
    ///
    /// Args:
    ///     abs_tol (float): Absolute error tolerance
    ///     rel_tol (float): Relative error tolerance
    ///
    /// Returns:
    ///     IntegratorConfig: Configuration for adaptive-step integration
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     config = bh.IntegratorConfig.adaptive(1e-9, 1e-6)
    ///     ```
    #[classmethod]
    fn adaptive(_cls: &Bound<'_, PyType>, abs_tol: f64, rel_tol: f64) -> Self {
        PyIntegratorConfig {
            inner: IntegratorConfig::adaptive(abs_tol, rel_tol),
        }
    }

    /// Get absolute error tolerance.
    #[getter]
    fn abs_tol(&self) -> f64 {
        self.inner.abs_tol
    }

    /// Get relative error tolerance.
    #[getter]
    fn rel_tol(&self) -> f64 {
        self.inner.rel_tol
    }

    /// Get initial step size.
    #[getter]
    fn initial_step(&self) -> Option<f64> {
        self.inner.initial_step
    }

    /// Get minimum step size.
    #[getter]
    fn min_step(&self) -> Option<f64> {
        self.inner.min_step
    }

    /// Get maximum step size.
    #[getter]
    fn max_step(&self) -> Option<f64> {
        self.inner.max_step
    }

    /// Get step safety factor.
    #[getter]
    fn step_safety_factor(&self) -> Option<f64> {
        self.inner.step_safety_factor
    }

    /// Get minimum step scale factor.
    #[getter]
    fn min_step_scale_factor(&self) -> Option<f64> {
        self.inner.min_step_scale_factor
    }

    /// Get maximum step scale factor.
    #[getter]
    fn max_step_scale_factor(&self) -> Option<f64> {
        self.inner.max_step_scale_factor
    }

    /// Get maximum step attempts.
    #[getter]
    fn max_step_attempts(&self) -> usize {
        self.inner.max_step_attempts
    }

    fn __repr__(&self) -> String {
        format!(
            "IntegratorConfig(abs_tol={}, rel_tol={}, max_step={:?})",
            self.inner.abs_tol, self.inner.rel_tol, self.inner.max_step
        )
    }
}

/// Result from an adaptive integration step.
///
/// Contains the new state, actual timestep used, error estimate, and suggested next step.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     def dynamics(t, state):
///         return np.array([state[1], -state[0]])
///
///     integrator = bh.RKF45Integrator(2, dynamics)
///     state = np.array([1.0, 0.0])
///
///     result = integrator.step(0.0, state, 0.1)
///     print(f"New state: {result.state}")
///     print(f"Step used: {result.dt_used}")
///     print(f"Error: {result.error_estimate}")
///     print(f"Next step: {result.dt_next}")
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "AdaptiveStepResult")]
pub struct PyAdaptiveStepDResult {
    inner: AdaptiveStepDResult,
}

#[pymethods]
impl PyAdaptiveStepDResult {
    /// Get the new state vector.
    ///
    /// Returns:
    ///     ndarray: State vector at time t + dt_used
    #[getter]
    fn state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, numpy::Ix1>> {
        self.inner.state.as_slice().to_pyarray(py)
    }

    /// Get the actual timestep used.
    ///
    /// Returns:
    ///     float: Timestep actually used (may be smaller than requested)
    #[getter]
    fn dt_used(&self) -> f64 {
        self.inner.dt_used
    }

    /// Get the estimated truncation error.
    ///
    /// Returns:
    ///     float: Estimated local truncation error
    #[getter]
    fn error_estimate(&self) -> f64 {
        self.inner.error_estimate
    }

    /// Get the suggested next timestep.
    ///
    /// Returns:
    ///     float: Suggested timestep for next iteration
    #[getter]
    fn dt_next(&self) -> f64 {
        self.inner.dt_next
    }

    fn __repr__(&self) -> String {
        format!(
            "AdaptiveStepResult(dt_used={}, error={}, dt_next={})",
            self.inner.dt_used, self.inner.error_estimate, self.inner.dt_next
        )
    }
}

// ============================================================================
// RK4 Fixed-Step Integrator
// ============================================================================

/// 4th-order Runge-Kutta fixed-step integrator.
///
/// Classical RK4 method with fixed timesteps. Provides good accuracy for most problems
/// with 4th-order error convergence.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Define dynamics: simple harmonic oscillator
///     def dynamics(t, state):
///         return np.array([state[1], -state[0]])
///
///     # Create integrator for 2D system
///     integrator = bh.RK4Integrator(2, dynamics)
///
///     # Integrate one step
///     state = np.array([1.0, 0.0])
///     new_state = integrator.step(0.0, state, 0.01)
///
///     # With variational matrix (state transition matrix)
///     def jacobian_fn(t, state):
///         return np.array([[0.0, 1.0], [-1.0, 0.0]])
///
///     jacobian = bh.DAnalyticJacobian(jacobian_fn)
///     integrator_varmat = bh.RK4Integrator(2, dynamics, jacobian)
///
///     phi = np.eye(2)
///     new_state, new_phi = integrator_varmat.step_with_varmat(0.0, state, phi, 0.01)
///     ```
#[pyclass(module = "brahe._brahe", unsendable)]
#[pyo3(name = "RK4Integrator")]
pub struct PyRK4DIntegrator {
    inner: RK4DIntegrator,
    dimension: usize,
}

#[pymethods]
impl PyRK4DIntegrator {
    /// Create a new RK4 integrator.
    ///
    /// Args:
    ///     dimension (int): State vector dimension
    ///     dynamics_fn (callable): Dynamics function with signature (t: float, state: ndarray) -> ndarray
    ///     jacobian (DAnalyticJacobian or DNumericalJacobian, optional): Jacobian provider for variational matrix propagation
    ///     config (IntegratorConfig, optional): Integration configuration
    ///
    /// Returns:
    ///     RK4Integrator: New RK4 integrator
    #[allow(deprecated)]
    #[new]
    #[pyo3(signature = (dimension, dynamics_fn, jacobian=None, config=None))]
    pub fn new(
        py: Python<'_>,
        dimension: usize,
        dynamics_fn: Py<PyAny>,
        jacobian: Option<Py<PyAny>>,
        config: Option<PyIntegratorConfig>,
    ) -> PyResult<Self> {
        // Create Rust closure for dynamics (Arc for thread safety)
        let dynamics_closure = {
            let dynamics_fn_arc = Arc::new(dynamics_fn.clone_ref(py));
            move |t: f64, state: na::DVector<f64>| -> na::DVector<f64> {
                Python::with_gil(|py| {
                    let state_py = state.as_slice().to_pyarray(py);
                    let result = dynamics_fn_arc
                        .call1(py, (t, state_py))
                        .expect("Failed to call dynamics function");
                    let result_vec = pyany_to_f64_array1(result.bind(py), Some(dimension))
                        .expect("Dynamics function returned invalid array");
                    na::DVector::from_vec(result_vec)
                })
            }
        };

        // Handle optional Jacobian provider
        let varmat: Option<Box<dyn DJacobianProvider>> = if let Some(jac_obj) = jacobian {
            // Check if it's a DNumericalJacobian or DAnalyticJacobian
            Python::with_gil(|py| {
                let jac = jac_obj.bind(py);

                // Try to extract as PyDNumericalJacobian
                if let Ok(num_jac) = jac.extract::<PyRef<PyDNumericalJacobian>>() {
                    let jac_arc = Arc::new(num_jac.dynamics_fn.clone_ref(py));
                    let jac_method = num_jac.method;
                    let jac_pert = num_jac.perturbation;

                    let jac_closure = move |t: f64, state: na::DVector<f64>| -> na::DVector<f64> {
                        Python::with_gil(|py| {
                            let state_py = state.as_slice().to_pyarray(py);
                            let result = jac_arc
                                .call1(py, (t, state_py))
                                .expect("Failed to call Jacobian dynamics");
                            let result_vec = pyany_to_f64_array1(result.bind(py), Some(dimension))
                                .expect("Jacobian dynamics returned invalid array");
                            na::DVector::from_vec(result_vec)
                        })
                    };

                    let mut provider = match jac_method {
                        crate::math::jacobian::DifferenceMethod::Forward => {
                            crate::math::jacobian::DNumericalJacobian::forward(Box::new(jac_closure))
                        }
                        crate::math::jacobian::DifferenceMethod::Central => {
                            crate::math::jacobian::DNumericalJacobian::central(Box::new(jac_closure))
                        }
                        crate::math::jacobian::DifferenceMethod::Backward => {
                            crate::math::jacobian::DNumericalJacobian::backward(Box::new(jac_closure))
                        }
                    };

                    provider = match jac_pert {
                        crate::math::jacobian::PerturbationStrategy::Adaptive {
                            scale_factor,
                            min_threshold,
                        } => provider.with_adaptive(scale_factor, min_threshold),
                        crate::math::jacobian::PerturbationStrategy::Fixed(offset) => {
                            provider.with_fixed_offset(offset)
                        }
                        crate::math::jacobian::PerturbationStrategy::Percentage(pct) => {
                            provider.with_percentage(pct)
                        }
                    };

                    Ok(Some(Box::new(provider) as Box<dyn DJacobianProvider>))
                } else if let Ok(ana_jac) = jac.extract::<PyRef<PyDAnalyticJacobian>>() {
                    let jac_arc = Arc::new(ana_jac.jacobian_fn.clone_ref(py));

                    let jac_closure = move |t: f64, state: na::DVector<f64>| -> na::DMatrix<f64> {
                        Python::with_gil(|py| {
                            let state_py = state.as_slice().to_pyarray(py);
                            let result = jac_arc
                                .call1(py, (t, state_py))
                                .expect("Failed to call Jacobian function");
                            let jac_matrix_vec =
                                pyany_to_f64_array2(result.bind(py), Some((dimension, dimension)))
                                    .expect("Jacobian function returned invalid array");

                            let mut jac_matrix = na::DMatrix::<f64>::zeros(dimension, dimension);
                            for i in 0..dimension {
                                for j in 0..dimension {
                                    jac_matrix[(i, j)] = jac_matrix_vec[i][j];
                                }
                            }
                            jac_matrix
                        })
                    };

                    let provider = crate::math::jacobian::DAnalyticJacobian::new(Box::new(jac_closure));
                    Ok(Some(Box::new(provider) as Box<dyn DJacobianProvider>))
                } else {
                    Err(exceptions::PyTypeError::new_err(
                        "jacobian must be DNumericalJacobian or DAnalyticJacobian",
                    ))
                }
            })?
        } else {
            None
        };

        // Create integrator
        let inner = if let Some(cfg) = config {
            RK4DIntegrator::with_config(dimension, Box::new(dynamics_closure), varmat, cfg.inner)
        } else {
            RK4DIntegrator::new(dimension, Box::new(dynamics_closure), varmat)
        };

        Ok(PyRK4DIntegrator { inner, dimension })
    }

    /// Perform one integration step.
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (ndarray): State vector at time t
    ///     dt (float, optional): Timestep. If None, uses the step size from configuration.
    ///
    /// Returns:
    ///     ndarray: State vector at time t + dt
    ///
    /// Example:
    ///     ```python
    ///     # Using explicit dt
    ///     new_state = integrator.step(0.0, state, 0.01)
    ///
    ///     # Using config-based dt
    ///     config = bh.IntegratorConfig.fixed_step(0.01)
    ///     integrator = bh.RK4Integrator(2, dynamics, config=config)
    ///     new_state = integrator.step(0.0, state)
    ///     ```
    #[pyo3(signature = (t, state, dt=None))]
    pub fn step<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        state: &Bound<'py, PyAny>,
        dt: Option<f64>,
    ) -> PyResult<Bound<'py, PyArray<f64, numpy::Ix1>>> {
        let state_vec = pyany_to_f64_array1(state, Some(self.dimension))?;
        let state_dvec = na::DVector::from_vec(state_vec);

        let new_state = self.inner.step(t, state_dvec, dt);

        Ok(new_state.as_slice().to_pyarray(py))
    }

    /// Perform one integration step with variational matrix propagation.
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (ndarray): State vector at time t
    ///     phi (ndarray): State transition matrix at time t (dimension × dimension)
    ///     dt (float, optional): Timestep. If None, uses the step size from configuration.
    ///
    /// Returns:
    ///     tuple: (new_state, new_phi) - State vector and STM at time t + dt
    ///
    /// Example:
    ///     ```python
    ///     phi = np.eye(2)
    ///     # Using explicit dt
    ///     new_state, new_phi = integrator.step_with_varmat(0.0, state, phi, 0.01)
    ///
    ///     # Using config-based dt
    ///     new_state, new_phi = integrator.step_with_varmat(0.0, state, phi)
    ///     ```
    #[pyo3(signature = (t, state, phi, dt=None))]
    #[allow(clippy::type_complexity)]
    pub fn step_with_varmat<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        state: &Bound<'py, PyAny>,
        phi: &Bound<'py, PyAny>,
        dt: Option<f64>,
    ) -> PyResult<(Bound<'py, PyArray<f64, numpy::Ix1>>, Bound<'py, PyArray<f64, numpy::Ix2>>)> {
        let state_vec = pyany_to_f64_array1(state, Some(self.dimension))?;
        let state_dvec = na::DVector::from_vec(state_vec);

        let phi_vec = pyany_to_f64_array2(phi, Some((self.dimension, self.dimension)))?;
        let mut phi_dmat = na::DMatrix::<f64>::zeros(self.dimension, self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_dmat[(i, j)] = phi_vec[i][j];
            }
        }

        let (new_state, new_phi) = self.inner.step_with_varmat(t, state_dvec, phi_dmat, dt);

        // Convert results to NumPy
        let state_np = new_state.as_slice().to_pyarray(py);

        let mut phi_flat = Vec::with_capacity(self.dimension * self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_flat.push(new_phi[(i, j)]);
            }
        }
        let phi_np = phi_flat
            .into_pyarray(py)
            .reshape([self.dimension, self.dimension])?;

        Ok((state_np, phi_np))
    }

    /// Get state vector dimension.
    #[getter]
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn __repr__(&self) -> String {
        format!("RK4Integrator(dimension={})", self.dimension)
    }
}
// ============================================================================
// Adaptive Integrators
// ============================================================================

/// Runge-Kutta-Fehlberg 4(5) adaptive integrator.
///
/// Embedded 5th/4th order method with automatic step size control. The integrator
/// uses error estimation from the embedded solution to adapt the timestep for
/// efficiency and accuracy.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     def dynamics(t, state):
///         # dy/dt = -y
///         return -state
///
///     config = bh.IntegratorConfig.adaptive(abs_tol=1e-9, rel_tol=1e-6)
///     integrator = bh.RKF45Integrator(dimension=1, dynamics_fn=dynamics, config=config)
///
///     # Adaptive step (tolerances from config)
///     result = integrator.step(0.0, np.array([1.0]), 0.1)
///     print(f"New state: {result.state}")
///     print(f"Step used: {result.dt_used}")
///     print(f"Next step suggestion: {result.dt_next}")
///     ```
#[pyclass(module = "brahe._brahe", unsendable)]
#[pyo3(name = "RKF45Integrator")]
pub struct PyRKF45DIntegrator {
    inner: RKF45DIntegrator,
    dimension: usize,
}

#[pymethods]
impl PyRKF45DIntegrator {
    /// Create a new RKF45 integrator.
    ///
    /// Args:
    ///     dimension (int): State vector dimension
    ///     dynamics_fn (callable): Dynamics function with signature (t: float, state: ndarray) -> ndarray
    ///     jacobian (DAnalyticJacobian or DNumericalJacobian, optional): Jacobian provider for variational matrix propagation
    ///     config (IntegratorConfig, optional): Integration configuration
    ///
    /// Returns:
    ///     RKF45Integrator: New RKF45 integrator
    #[allow(deprecated)]
    #[new]
    #[pyo3(signature = (dimension, dynamics_fn, jacobian=None, config=None))]
    pub fn new(
        py: Python<'_>,
        dimension: usize,
        dynamics_fn: Py<PyAny>,
        jacobian: Option<Py<PyAny>>,
        config: Option<PyIntegratorConfig>,
    ) -> PyResult<Self> {
        // Create Rust closure for dynamics (Arc for thread safety)
        let dynamics_closure = {
            let dynamics_fn_arc = Arc::new(dynamics_fn.clone_ref(py));
            move |t: f64, state: na::DVector<f64>| -> na::DVector<f64> {
                Python::with_gil(|py| {
                    let state_py = state.as_slice().to_pyarray(py);
                    let result = dynamics_fn_arc
                        .call1(py, (t, state_py))
                        .expect("Failed to call dynamics function");
                    let result_vec = pyany_to_f64_array1(result.bind(py), Some(dimension))
                        .expect("Dynamics function returned invalid array");
                    na::DVector::from_vec(result_vec)
                })
            }
        };

        // Handle optional Jacobian provider (same as RK4)
        let varmat: Option<Box<dyn DJacobianProvider>> = if let Some(jac_obj) = jacobian {
            Python::with_gil(|py| {
                let jac = jac_obj.bind(py);

                if let Ok(num_jac) = jac.extract::<PyRef<PyDNumericalJacobian>>() {
                    let jac_arc = Arc::new(num_jac.dynamics_fn.clone_ref(py));
                    let jac_method = num_jac.method;
                    let jac_pert = num_jac.perturbation;

                    let jac_closure = move |t: f64, state: na::DVector<f64>| -> na::DVector<f64> {
                        Python::with_gil(|py| {
                            let state_py = state.as_slice().to_pyarray(py);
                            let result = jac_arc
                                .call1(py, (t, state_py))
                                .expect("Failed to call Jacobian dynamics");
                            let result_vec = pyany_to_f64_array1(result.bind(py), Some(dimension))
                                .expect("Jacobian dynamics returned invalid array");
                            na::DVector::from_vec(result_vec)
                        })
                    };

                    let mut provider = match jac_method {
                        crate::math::jacobian::DifferenceMethod::Forward => {
                            crate::math::jacobian::DNumericalJacobian::forward(Box::new(jac_closure))
                        }
                        crate::math::jacobian::DifferenceMethod::Central => {
                            crate::math::jacobian::DNumericalJacobian::central(Box::new(jac_closure))
                        }
                        crate::math::jacobian::DifferenceMethod::Backward => {
                            crate::math::jacobian::DNumericalJacobian::backward(Box::new(jac_closure))
                        }
                    };

                    provider = match jac_pert {
                        crate::math::jacobian::PerturbationStrategy::Adaptive {
                            scale_factor,
                            min_threshold,
                        } => provider.with_adaptive(scale_factor, min_threshold),
                        crate::math::jacobian::PerturbationStrategy::Fixed(offset) => {
                            provider.with_fixed_offset(offset)
                        }
                        crate::math::jacobian::PerturbationStrategy::Percentage(pct) => {
                            provider.with_percentage(pct)
                        }
                    };

                    Ok(Some(Box::new(provider) as Box<dyn DJacobianProvider>))
                } else if let Ok(ana_jac) = jac.extract::<PyRef<PyDAnalyticJacobian>>() {
                    let jac_arc = Arc::new(ana_jac.jacobian_fn.clone_ref(py));

                    let jac_closure = move |t: f64, state: na::DVector<f64>| -> na::DMatrix<f64> {
                        Python::with_gil(|py| {
                            let state_py = state.as_slice().to_pyarray(py);
                            let result = jac_arc
                                .call1(py, (t, state_py))
                                .expect("Failed to call Jacobian function");
                            let jac_matrix_vec =
                                pyany_to_f64_array2(result.bind(py), Some((dimension, dimension)))
                                    .expect("Jacobian function returned invalid array");

                            let mut jac_matrix = na::DMatrix::<f64>::zeros(dimension, dimension);
                            for i in 0..dimension {
                                for j in 0..dimension {
                                    jac_matrix[(i, j)] = jac_matrix_vec[i][j];
                                }
                            }
                            jac_matrix
                        })
                    };

                    let provider = crate::math::jacobian::DAnalyticJacobian::new(Box::new(jac_closure));
                    Ok(Some(Box::new(provider) as Box<dyn DJacobianProvider>))
                } else {
                    Err(exceptions::PyTypeError::new_err(
                        "jacobian must be DNumericalJacobian or DAnalyticJacobian",
                    ))
                }
            })?
        } else {
            None
        };

        // Create integrator
        let inner = if let Some(cfg) = config {
            RKF45DIntegrator::with_config(dimension, Box::new(dynamics_closure), varmat, cfg.inner)
        } else {
            RKF45DIntegrator::new(dimension, Box::new(dynamics_closure), varmat)
        };

        Ok(PyRKF45DIntegrator { inner, dimension })
    }

    /// Perform one adaptive integration step.
    ///
    /// Tolerances are read from the integrator's configuration.
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (ndarray): State vector at time t
    ///     dt (float): Requested timestep
    ///
    /// Returns:
    ///     AdaptiveStepResult: Result containing new state, actual dt used, error estimate, and suggested next dt
    pub fn step<'py>(
        &self,
        _py: Python<'py>,
        t: f64,
        state: PyReadonlyArray1<f64>,
        dt: f64,
    ) -> PyResult<PyAdaptiveStepDResult> {
        // Convert state to DVector
        let state_vec = state.as_slice()?;
        let state_dvec = na::DVector::from_vec(state_vec.to_vec());

        // Perform step
        let result = self.inner.step(t, state_dvec, dt);

        Ok(PyAdaptiveStepDResult { inner: result })
    }

    /// Perform one adaptive integration step with variational matrix.
    ///
    /// Tolerances are read from the integrator's configuration.
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (ndarray): State vector at time t
    ///     phi (ndarray): State transition matrix (dimension x dimension)
    ///     dt (float): Requested timestep
    ///
    /// Returns:
    ///     tuple: (new_state, new_phi, dt_used, error_estimate, dt_next)
    #[allow(clippy::type_complexity)]
    pub fn step_with_varmat<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        state: PyReadonlyArray1<f64>,
        phi: PyReadonlyArray2<f64>,
        dt: f64,
    ) -> PyResult<(
        Bound<'py, PyArray<f64, Ix1>>,
        Bound<'py, PyArray<f64, Ix2>>,
        f64,
        f64,
        f64,
    )> {
        // Convert inputs
        let state_vec = state.as_slice()?;
        let state_dvec = na::DVector::from_vec(state_vec.to_vec());

        let phi_array = phi.as_array();
        let mut phi_dmat = na::DMatrix::<f64>::zeros(self.dimension, self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_dmat[(i, j)] = phi_array[[i, j]];
            }
        }

        // Perform step
        let (new_state, new_phi, dt_used, error_est, dt_next) =
            self.inner
                .step_with_varmat(t, state_dvec, phi_dmat, dt);

        // Convert results to NumPy
        let state_np = new_state.as_slice().to_pyarray(py);

        let mut phi_flat = Vec::with_capacity(self.dimension * self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_flat.push(new_phi[(i, j)]);
            }
        }
        let phi_np = phi_flat
            .into_pyarray(py)
            .reshape([self.dimension, self.dimension])?;

        Ok((state_np, phi_np, dt_used, error_est, dt_next))
    }

    /// Get state vector dimension.
    #[getter]
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn __repr__(&self) -> String {
        format!("RKF45Integrator(dimension={})", self.dimension)
    }
}

/// Dormand-Prince 5(4) adaptive integrator (MATLAB's ode45).
///
/// More efficient than RKF45 due to FSAL (First Same As Last) property. This is
/// the industry-standard general-purpose adaptive integrator.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     def dynamics(t, state):
///         return -state
///
///     config = bh.IntegratorConfig.adaptive(abs_tol=1e-9, rel_tol=1e-6)
///     integrator = bh.DP54Integrator(dimension=1, dynamics_fn=dynamics, config=config)
///
///     # Adaptive step (tolerances from config)
///     result = integrator.step(0.0, np.array([1.0]), 0.1)
///     ```
#[pyclass(module = "brahe._brahe", unsendable)]
#[pyo3(name = "DP54Integrator")]
pub struct PyDP54DIntegrator {
    inner: DormandPrince54DIntegrator,
    dimension: usize,
}

#[pymethods]
impl PyDP54DIntegrator {
    /// Create a new DP54 integrator.
    ///
    /// Args:
    ///     dimension (int): State vector dimension
    ///     dynamics_fn (callable): Dynamics function with signature (t: float, state: ndarray) -> ndarray
    ///     jacobian (DAnalyticJacobian or DNumericalJacobian, optional): Jacobian provider for variational matrix propagation
    ///     config (IntegratorConfig, optional): Integration configuration
    ///
    /// Returns:
    ///     DP54Integrator: New DP54 integrator
    #[allow(deprecated)]
    #[new]
    #[pyo3(signature = (dimension, dynamics_fn, jacobian=None, config=None))]
    pub fn new(
        py: Python<'_>,
        dimension: usize,
        dynamics_fn: Py<PyAny>,
        jacobian: Option<Py<PyAny>>,
        config: Option<PyIntegratorConfig>,
    ) -> PyResult<Self> {
        // Create Rust closure for dynamics
        let dynamics_closure = {
            let dynamics_fn_arc = Arc::new(dynamics_fn.clone_ref(py));
            move |t: f64, state: na::DVector<f64>| -> na::DVector<f64> {
                Python::with_gil(|py| {
                    let state_py = state.as_slice().to_pyarray(py);
                    let result = dynamics_fn_arc
                        .call1(py, (t, state_py))
                        .expect("Failed to call dynamics function");
                    let result_vec = pyany_to_f64_array1(result.bind(py), Some(dimension))
                        .expect("Dynamics function returned invalid array");
                    na::DVector::from_vec(result_vec)
                })
            }
        };

        // Handle optional Jacobian provider
        let varmat: Option<Box<dyn DJacobianProvider>> = if let Some(jac_obj) = jacobian {
            Python::with_gil(|py| {
                let jac = jac_obj.bind(py);

                if let Ok(num_jac) = jac.extract::<PyRef<PyDNumericalJacobian>>() {
                    let jac_arc = Arc::new(num_jac.dynamics_fn.clone_ref(py));
                    let jac_method = num_jac.method;
                    let jac_pert = num_jac.perturbation;

                    let jac_closure = move |t: f64, state: na::DVector<f64>| -> na::DVector<f64> {
                        Python::with_gil(|py| {
                            let state_py = state.as_slice().to_pyarray(py);
                            let result = jac_arc
                                .call1(py, (t, state_py))
                                .expect("Failed to call Jacobian dynamics");
                            let result_vec = pyany_to_f64_array1(result.bind(py), Some(dimension))
                                .expect("Jacobian dynamics returned invalid array");
                            na::DVector::from_vec(result_vec)
                        })
                    };

                    let mut provider = match jac_method {
                        crate::math::jacobian::DifferenceMethod::Forward => {
                            crate::math::jacobian::DNumericalJacobian::forward(Box::new(jac_closure))
                        }
                        crate::math::jacobian::DifferenceMethod::Central => {
                            crate::math::jacobian::DNumericalJacobian::central(Box::new(jac_closure))
                        }
                        crate::math::jacobian::DifferenceMethod::Backward => {
                            crate::math::jacobian::DNumericalJacobian::backward(Box::new(jac_closure))
                        }
                    };

                    provider = match jac_pert {
                        crate::math::jacobian::PerturbationStrategy::Adaptive {
                            scale_factor,
                            min_threshold,
                        } => provider.with_adaptive(scale_factor, min_threshold),
                        crate::math::jacobian::PerturbationStrategy::Fixed(offset) => {
                            provider.with_fixed_offset(offset)
                        }
                        crate::math::jacobian::PerturbationStrategy::Percentage(pct) => {
                            provider.with_percentage(pct)
                        }
                    };

                    Ok(Some(Box::new(provider) as Box<dyn DJacobianProvider>))
                } else if let Ok(ana_jac) = jac.extract::<PyRef<PyDAnalyticJacobian>>() {
                    let jac_arc = Arc::new(ana_jac.jacobian_fn.clone_ref(py));

                    let jac_closure = move |t: f64, state: na::DVector<f64>| -> na::DMatrix<f64> {
                        Python::with_gil(|py| {
                            let state_py = state.as_slice().to_pyarray(py);
                            let result = jac_arc
                                .call1(py, (t, state_py))
                                .expect("Failed to call Jacobian function");
                            let jac_matrix_vec =
                                pyany_to_f64_array2(result.bind(py), Some((dimension, dimension)))
                                    .expect("Jacobian function returned invalid array");

                            let mut jac_matrix = na::DMatrix::<f64>::zeros(dimension, dimension);
                            for i in 0..dimension {
                                for j in 0..dimension {
                                    jac_matrix[(i, j)] = jac_matrix_vec[i][j];
                                }
                            }
                            jac_matrix
                        })
                    };

                    let provider = crate::math::jacobian::DAnalyticJacobian::new(Box::new(jac_closure));
                    Ok(Some(Box::new(provider) as Box<dyn DJacobianProvider>))
                } else {
                    Err(exceptions::PyTypeError::new_err(
                        "jacobian must be DNumericalJacobian or DAnalyticJacobian",
                    ))
                }
            })?
        } else {
            None
        };

        // Create integrator
        let inner = if let Some(cfg) = config {
            DormandPrince54DIntegrator::with_config(dimension, Box::new(dynamics_closure), varmat, cfg.inner)
        } else {
            DormandPrince54DIntegrator::new(dimension, Box::new(dynamics_closure), varmat)
        };

        Ok(PyDP54DIntegrator { inner, dimension })
    }

    /// Perform one adaptive integration step.
    ///
    /// Tolerances are read from the integrator's configuration.
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (ndarray): State vector at time t
    ///     dt (float): Requested timestep
    ///
    /// Returns:
    ///     AdaptiveStepResult: Result containing new state, actual dt used, error estimate, and suggested next dt
    pub fn step<'py>(
        &self,
        _py: Python<'py>,
        t: f64,
        state: PyReadonlyArray1<f64>,
        dt: f64,
    ) -> PyResult<PyAdaptiveStepDResult> {
        let state_vec = state.as_slice()?;
        let state_dvec = na::DVector::from_vec(state_vec.to_vec());

        let result = self.inner.step(t, state_dvec, dt);

        Ok(PyAdaptiveStepDResult { inner: result })
    }

    /// Perform one adaptive integration step with variational matrix.
    ///
    /// Tolerances are read from the integrator's configuration.
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (ndarray): State vector at time t
    ///     phi (ndarray): State transition matrix (dimension x dimension)
    ///     dt (float): Requested timestep
    ///
    /// Returns:
    ///     tuple: (new_state, new_phi, dt_used, error_estimate, dt_next)
    #[allow(clippy::type_complexity)]
    pub fn step_with_varmat<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        state: PyReadonlyArray1<f64>,
        phi: PyReadonlyArray2<f64>,
        dt: f64,
    ) -> PyResult<(
        Bound<'py, PyArray<f64, Ix1>>,
        Bound<'py, PyArray<f64, Ix2>>,
        f64,
        f64,
        f64,
    )> {
        let state_vec = state.as_slice()?;
        let state_dvec = na::DVector::from_vec(state_vec.to_vec());

        let phi_array = phi.as_array();
        let mut phi_dmat = na::DMatrix::<f64>::zeros(self.dimension, self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_dmat[(i, j)] = phi_array[[i, j]];
            }
        }

        let (new_state, new_phi, dt_used, error_est, dt_next) =
            self.inner
                .step_with_varmat(t, state_dvec, phi_dmat, dt);

        let state_np = new_state.as_slice().to_pyarray(py);

        let mut phi_flat = Vec::with_capacity(self.dimension * self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_flat.push(new_phi[(i, j)]);
            }
        }
        let phi_np = phi_flat
            .into_pyarray(py)
            .reshape([self.dimension, self.dimension])?;

        Ok((state_np, phi_np, dt_used, error_est, dt_next))
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn __repr__(&self) -> String {
        format!("DP54Integrator(dimension={})", self.dimension)
    }
}

/// RKN12(10) Runge-Kutta-Nyström adaptive integrator (EXPERIMENTAL).
///
/// High-order specialized integrator for second-order ODEs (like orbital mechanics).
/// More efficient and accurate than general-purpose methods for position-velocity systems.
///
/// WARNING: This integrator is experimental and may have stability issues.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     def dynamics(t, state):
///         # Two-body dynamics: [r, v] -> [v, a]
///         r = state[:3]
///         v = state[3:]
///         r_norm = np.linalg.norm(r)
///         a = -bh.GM_EARTH / (r_norm**3) * r
///         return np.concatenate([v, a])
///
///     config = bh.IntegratorConfig.adaptive(abs_tol=1e-9, rel_tol=1e-6)
///     integrator = bh.RKN1210Integrator(dimension=6, dynamics_fn=dynamics, config=config)
///
///     # Adaptive step (tolerances from config)
///     result = integrator.step(0.0, state, 1.0)
///     ```
#[pyclass(module = "brahe._brahe", unsendable)]
#[pyo3(name = "RKN1210Integrator")]
pub struct PyRKN1210DIntegrator {
    inner: RKN1210DIntegrator,
    dimension: usize,
}

#[pymethods]
impl PyRKN1210DIntegrator {
    /// Create a new RKN1210 integrator.
    ///
    /// Args:
    ///     dimension (int): State vector dimension (must be even: position + velocity)
    ///     dynamics_fn (callable): Dynamics function with signature (t: float, state: ndarray) -> ndarray
    ///     jacobian (DAnalyticJacobian or DNumericalJacobian, optional): Jacobian provider for variational matrix propagation
    ///     config (IntegratorConfig, optional): Integration configuration
    ///
    /// Returns:
    ///     RKN1210Integrator: New RKN1210 integrator
    ///
    /// Raises:
    ///     ValueError: If dimension is not even
    #[allow(deprecated)]
    #[new]
    #[pyo3(signature = (dimension, dynamics_fn, jacobian=None, config=None))]
    pub fn new(
        py: Python<'_>,
        dimension: usize,
        dynamics_fn: Py<PyAny>,
        jacobian: Option<Py<PyAny>>,
        config: Option<PyIntegratorConfig>,
    ) -> PyResult<Self> {
        // Check dimension is even
        if !dimension.is_multiple_of(2) {
            return Err(exceptions::PyValueError::new_err(
                "RKN1210 requires even dimension (position + velocity)",
            ));
        }

        // Create Rust closure for dynamics
        let dynamics_closure = {
            let dynamics_fn_arc = Arc::new(dynamics_fn.clone_ref(py));
            move |t: f64, state: na::DVector<f64>| -> na::DVector<f64> {
                Python::with_gil(|py| {
                    let state_py = state.as_slice().to_pyarray(py);
                    let result = dynamics_fn_arc
                        .call1(py, (t, state_py))
                        .expect("Failed to call dynamics function");
                    let result_vec = pyany_to_f64_array1(result.bind(py), Some(dimension))
                        .expect("Dynamics function returned invalid array");
                    na::DVector::from_vec(result_vec)
                })
            }
        };

        // Handle optional Jacobian provider
        let varmat: Option<Box<dyn DJacobianProvider>> = if let Some(jac_obj) = jacobian {
            Python::with_gil(|py| {
                let jac = jac_obj.bind(py);

                if let Ok(num_jac) = jac.extract::<PyRef<PyDNumericalJacobian>>() {
                    let jac_arc = Arc::new(num_jac.dynamics_fn.clone_ref(py));
                    let jac_method = num_jac.method;
                    let jac_pert = num_jac.perturbation;

                    let jac_closure = move |t: f64, state: na::DVector<f64>| -> na::DVector<f64> {
                        Python::with_gil(|py| {
                            let state_py = state.as_slice().to_pyarray(py);
                            let result = jac_arc
                                .call1(py, (t, state_py))
                                .expect("Failed to call Jacobian dynamics");
                            let result_vec = pyany_to_f64_array1(result.bind(py), Some(dimension))
                                .expect("Jacobian dynamics returned invalid array");
                            na::DVector::from_vec(result_vec)
                        })
                    };

                    let mut provider = match jac_method {
                        crate::math::jacobian::DifferenceMethod::Forward => {
                            crate::math::jacobian::DNumericalJacobian::forward(Box::new(jac_closure))
                        }
                        crate::math::jacobian::DifferenceMethod::Central => {
                            crate::math::jacobian::DNumericalJacobian::central(Box::new(jac_closure))
                        }
                        crate::math::jacobian::DifferenceMethod::Backward => {
                            crate::math::jacobian::DNumericalJacobian::backward(Box::new(jac_closure))
                        }
                    };

                    provider = match jac_pert {
                        crate::math::jacobian::PerturbationStrategy::Adaptive {
                            scale_factor,
                            min_threshold,
                        } => provider.with_adaptive(scale_factor, min_threshold),
                        crate::math::jacobian::PerturbationStrategy::Fixed(offset) => {
                            provider.with_fixed_offset(offset)
                        }
                        crate::math::jacobian::PerturbationStrategy::Percentage(pct) => {
                            provider.with_percentage(pct)
                        }
                    };

                    Ok(Some(Box::new(provider) as Box<dyn DJacobianProvider>))
                } else if let Ok(ana_jac) = jac.extract::<PyRef<PyDAnalyticJacobian>>() {
                    let jac_arc = Arc::new(ana_jac.jacobian_fn.clone_ref(py));

                    let jac_closure = move |t: f64, state: na::DVector<f64>| -> na::DMatrix<f64> {
                        Python::with_gil(|py| {
                            let state_py = state.as_slice().to_pyarray(py);
                            let result = jac_arc
                                .call1(py, (t, state_py))
                                .expect("Failed to call Jacobian function");
                            let jac_matrix_vec =
                                pyany_to_f64_array2(result.bind(py), Some((dimension, dimension)))
                                    .expect("Jacobian function returned invalid array");

                            let mut jac_matrix = na::DMatrix::<f64>::zeros(dimension, dimension);
                            for i in 0..dimension {
                                for j in 0..dimension {
                                    jac_matrix[(i, j)] = jac_matrix_vec[i][j];
                                }
                            }
                            jac_matrix
                        })
                    };

                    let provider = crate::math::jacobian::DAnalyticJacobian::new(Box::new(jac_closure));
                    Ok(Some(Box::new(provider) as Box<dyn DJacobianProvider>))
                } else {
                    Err(exceptions::PyTypeError::new_err(
                        "jacobian must be DNumericalJacobian or DAnalyticJacobian",
                    ))
                }
            })?
        } else {
            None
        };

        // Create integrator
        let inner = if let Some(cfg) = config {
            RKN1210DIntegrator::with_config(dimension, Box::new(dynamics_closure), varmat, cfg.inner)
        } else {
            RKN1210DIntegrator::new(dimension, Box::new(dynamics_closure), varmat)
        };

        Ok(PyRKN1210DIntegrator { inner, dimension })
    }

    /// Perform one adaptive integration step.
    ///
    /// Tolerances are read from the integrator's configuration.
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (ndarray): State vector at time t [position, velocity]
    ///     dt (float): Requested timestep
    ///
    /// Returns:
    ///     AdaptiveStepResult: Result containing new state, actual dt used, error estimate, and suggested next dt
    pub fn step<'py>(
        &self,
        _py: Python<'py>,
        t: f64,
        state: PyReadonlyArray1<f64>,
        dt: f64,
    ) -> PyResult<PyAdaptiveStepDResult> {
        let state_vec = state.as_slice()?;
        let state_dvec = na::DVector::from_vec(state_vec.to_vec());

        let result = self.inner.step(t, state_dvec, dt);

        Ok(PyAdaptiveStepDResult { inner: result })
    }

    /// Perform one adaptive integration step with variational matrix.
    ///
    /// Tolerances are read from the integrator's configuration.
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (ndarray): State vector at time t
    ///     phi (ndarray): State transition matrix (dimension x dimension)
    ///     dt (float): Requested timestep
    ///
    /// Returns:
    ///     tuple: (new_state, new_phi, dt_used, error_estimate, dt_next)
    #[allow(clippy::type_complexity)]
    pub fn step_with_varmat<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        state: PyReadonlyArray1<f64>,
        phi: PyReadonlyArray2<f64>,
        dt: f64,
    ) -> PyResult<(
        Bound<'py, PyArray<f64, Ix1>>,
        Bound<'py, PyArray<f64, Ix2>>,
        f64,
        f64,
        f64,
    )> {
        let state_vec = state.as_slice()?;
        let state_dvec = na::DVector::from_vec(state_vec.to_vec());

        let phi_array = phi.as_array();
        let mut phi_dmat = na::DMatrix::<f64>::zeros(self.dimension, self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_dmat[(i, j)] = phi_array[[i, j]];
            }
        }

        let (new_state, new_phi, dt_used, error_est, dt_next) =
            self.inner
                .step_with_varmat(t, state_dvec, phi_dmat, dt);

        let state_np = new_state.as_slice().to_pyarray(py);

        let mut phi_flat = Vec::with_capacity(self.dimension * self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_flat.push(new_phi[(i, j)]);
            }
        }
        let phi_np = phi_flat
            .into_pyarray(py)
            .reshape([self.dimension, self.dimension])?;

        Ok((state_np, phi_np, dt_used, error_est, dt_next))
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn __repr__(&self) -> String {
        format!("RKN1210Integrator(dimension={})", self.dimension)
    }
}
