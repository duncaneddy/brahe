/*
 * Python bindings for numerical integrators.
 *
 * Provides dynamic-sized (runtime-determined dimension) integrators for solving
 * ordinary differential equations (ODEs). Includes both fixed-step and adaptive-step
 * integration methods.
 */

// NOTE: Imports are handled by mod.rs since this file is included via include! macro


// ============================================================================
// Configuration Types
// ============================================================================

/// Configuration for numerical integrators.
///
/// Controls error tolerances, step size limits, and other integration parameters.
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
    /// Create a new integrator configuration with custom parameters.
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
    inner: DIntegratorStepResult,
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
    ///     float | None: Estimated local truncation error (None for fixed-step integrators)
    #[getter]
    fn error_estimate(&self) -> Option<f64> {
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
        let error_str = match self.inner.error_estimate {
            Some(err) => format!("{}", err),
            None => "None".to_string(),
        };
        format!(
            "AdaptiveStepResult(dt_used={}, error={}, dt_next={})",
            self.inner.dt_used, error_str, self.inner.dt_next
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
/// Args:
///     dimension (int): State vector dimension
///     dynamics_fn (callable): Dynamics function with signature (t: float, state: ndarray) -> ndarray
///     jacobian (DAnalyticJacobian or DNumericalJacobian, optional): Jacobian provider for variational matrix propagation
///     config (IntegratorConfig, optional): Integration configuration
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
    #[allow(deprecated)]
    #[new]
    #[pyo3(signature = (dimension, dynamics_fn, jacobian=None, sensitivity=None, control_fn=None, config=None))]
    pub fn new(
        py: Python<'_>,
        dimension: usize,
        dynamics_fn: Py<PyAny>,
        jacobian: Option<Py<PyAny>>,
        sensitivity: Option<Py<PyAny>>,
        control_fn: Option<Py<PyAny>>,
        config: Option<PyIntegratorConfig>,
    ) -> PyResult<Self> {
        // Create Rust closure for dynamics (Arc for thread safety)
        // Note: Uses 3-argument closure (t, state, params) for sensitivity support
        let dynamics_closure = {
            let dynamics_fn_arc = Arc::new(dynamics_fn.clone_ref(py));
            move |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| -> na::DVector<f64> {
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

        // Create Rust closure for control input if provided
        let control: crate::integrators::traits::DControlInput = if let Some(ctrl_fn) = control_fn {
            let ctrl_fn_arc = Arc::new(ctrl_fn.clone_ref(py));
            Some(Box::new(move |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| -> na::DVector<f64> {
                Python::with_gil(|py| {
                    let state_py = state.as_slice().to_pyarray(py);
                    let result = ctrl_fn_arc
                        .call1(py, (t, state_py))
                        .expect("Failed to call control function");
                    let result_vec = pyany_to_f64_array1(result.bind(py), Some(dimension))
                        .expect("Control function returned invalid array");
                    na::DVector::from_vec(result_vec)
                })
            }))
        } else {
            None
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

                    let jac_closure = move |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| -> na::DVector<f64> {
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

                    let jac_closure = move |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| -> na::DMatrix<f64> {
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

        // Handle optional sensitivity provider
        let sensmat: Option<Box<dyn DSensitivityProvider>> = if let Some(sens_obj) = sensitivity {
            Python::with_gil(|py| {
                let sens = sens_obj.bind(py);

                // Try to extract as PyDNumericalSensitivity
                if let Ok(num_sens) = sens.extract::<PyRef<PyDNumericalSensitivity>>() {
                    let sens_arc = Arc::new(num_sens.dynamics_fn.clone_ref(py));
                    let sens_method = num_sens.method;
                    let sens_pert = num_sens.perturbation;

                    let sens_closure = move |t: f64, state: &na::DVector<f64>, params: &na::DVector<f64>| -> na::DVector<f64> {
                        Python::with_gil(|py| {
                            let state_py = state.as_slice().to_pyarray(py);
                            let params_py = params.as_slice().to_pyarray(py);
                            let result = sens_arc
                                .call1(py, (t, state_py, params_py))
                                .expect("Failed to call sensitivity dynamics");
                            let result_vec = pyany_to_f64_array1(result.bind(py), Some(state.len()))
                                .expect("Sensitivity dynamics returned invalid array");
                            na::DVector::from_vec(result_vec)
                        })
                    };

                    let mut provider = match sens_method {
                        crate::math::jacobian::DifferenceMethod::Forward => {
                            crate::math::sensitivity::DNumericalSensitivity::forward(Box::new(sens_closure))
                        }
                        crate::math::jacobian::DifferenceMethod::Central => {
                            crate::math::sensitivity::DNumericalSensitivity::central(Box::new(sens_closure))
                        }
                        crate::math::jacobian::DifferenceMethod::Backward => {
                            // Backward not implemented for sensitivity, use central
                            crate::math::sensitivity::DNumericalSensitivity::central(Box::new(sens_closure))
                        }
                    };

                    provider = match sens_pert {
                        crate::math::jacobian::PerturbationStrategy::Adaptive {
                            scale_factor,
                            min_threshold,
                        } => provider.with_strategy(crate::math::jacobian::PerturbationStrategy::Adaptive {
                            scale_factor,
                            min_threshold,
                        }),
                        crate::math::jacobian::PerturbationStrategy::Fixed(offset) => {
                            provider.with_strategy(crate::math::jacobian::PerturbationStrategy::Fixed(offset))
                        }
                        crate::math::jacobian::PerturbationStrategy::Percentage(pct) => {
                            provider.with_strategy(crate::math::jacobian::PerturbationStrategy::Percentage(pct))
                        }
                    };

                    Ok(Some(Box::new(provider) as Box<dyn DSensitivityProvider>))
                } else if let Ok(ana_sens) = sens.extract::<PyRef<PyDAnalyticSensitivity>>() {
                    let sens_arc = Arc::new(ana_sens.sensitivity_fn.clone_ref(py));

                    let sens_closure = move |t: f64, state: &na::DVector<f64>, params: &na::DVector<f64>| -> na::DMatrix<f64> {
                        Python::with_gil(|py| {
                            let state_py = state.as_slice().to_pyarray(py);
                            let params_py = params.as_slice().to_pyarray(py);
                            let result = sens_arc
                                .call1(py, (t, state_py, params_py))
                                .expect("Failed to call sensitivity function");
                            let state_dim = state.len();
                            let param_dim = params.len();
                            let sens_matrix_vec =
                                pyany_to_f64_array2(result.bind(py), Some((state_dim, param_dim)))
                                    .expect("Sensitivity function returned invalid array");

                            let mut sens_matrix = na::DMatrix::<f64>::zeros(state_dim, param_dim);
                            for i in 0..state_dim {
                                for j in 0..param_dim {
                                    sens_matrix[(i, j)] = sens_matrix_vec[i][j];
                                }
                            }
                            sens_matrix
                        })
                    };

                    let provider = crate::math::sensitivity::DAnalyticSensitivity::new(Box::new(sens_closure));
                    Ok(Some(Box::new(provider) as Box<dyn DSensitivityProvider>))
                } else {
                    Err(exceptions::PyTypeError::new_err(
                        "sensitivity must be NumericalSensitivity or AnalyticSensitivity",
                    ))
                }
            })?
        } else {
            None
        };

        // Create integrator
        let inner = if let Some(cfg) = config {
            RK4DIntegrator::with_config(dimension, Box::new(dynamics_closure), varmat, sensmat, control, cfg.inner)
        } else {
            RK4DIntegrator::new(dimension, Box::new(dynamics_closure), varmat, sensmat, control)
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

        let result = self.inner.step(t, state_dvec, None, dt);

        Ok(result.state.as_slice().to_pyarray(py))
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

        let result = self.inner.step_with_varmat(t, state_dvec, None, phi_dmat, dt);

        // Convert results to NumPy
        let state_np = result.state.as_slice().to_pyarray(py);

        let new_phi = result.phi.expect("Variational matrix should be present");
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

    /// Advance state and sensitivity matrix by one timestep.
    ///
    /// Propagates the sensitivity matrix S that maps parameter uncertainties
    /// to state uncertainties. The sensitivity evolves according to dS/dt = A*S + B.
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (np.ndarray): State vector at time t
    ///     sens (np.ndarray): Sensitivity matrix at time t (state_dim x param_dim)
    ///     params (np.ndarray): Parameter vector
    ///     dt (float, optional): Integration timestep
    ///
    /// Returns:
    ///     tuple: (new_state, new_sensitivity)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     # Setup integrator with Jacobian and sensitivity providers
    ///     integrator = bh.RK4Integrator(6, dynamics, jacobian, sensitivity)
    ///     state = np.array([...])
    ///     sens = np.zeros((6, 2))  # 6 state dims, 2 params
    ///     params = np.array([1.0, 2.0])
    ///
    ///     new_state, new_sens = integrator.step_with_sensmat(0.0, state, sens, params, 1.0)
    ///     ```
    #[pyo3(signature = (t, state, sens, params, dt=None))]
    #[allow(clippy::type_complexity)]
    pub fn step_with_sensmat<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        state: PyReadonlyArray1<f64>,
        sens: PyReadonlyArray2<f64>,
        params: PyReadonlyArray1<f64>,
        dt: Option<f64>,
    ) -> PyResult<(Bound<'py, PyArray<f64, Ix1>>, Bound<'py, PyArray<f64, Ix2>>)> {
        // Convert state to DVector
        let state_vec = state.as_slice()?;
        let state_dvec = na::DVector::from_vec(state_vec.to_vec());

        // Convert sensitivity matrix to DMatrix
        let sens_array = sens.as_array();
        let sens_shape = sens_array.shape();
        let num_params = sens_shape[1];
        let mut sens_dmat = na::DMatrix::<f64>::zeros(self.dimension, num_params);
        for i in 0..self.dimension {
            for j in 0..num_params {
                sens_dmat[(i, j)] = sens_array[[i, j]];
            }
        }

        // Convert params to DVector
        let params_vec = params.as_slice()?;
        let params_dvec = na::DVector::from_vec(params_vec.to_vec());

        // Perform step
        let result = self.inner.step_with_sensmat(
            t, state_dvec, sens_dmat, &params_dvec, dt
        );

        // Convert results to NumPy
        let state_np = result.state.as_slice().to_pyarray(py);

        let new_sens = result.sens.expect("Sensitivity matrix should be present");
        let mut sens_flat = Vec::with_capacity(self.dimension * num_params);
        for i in 0..self.dimension {
            for j in 0..num_params {
                sens_flat.push(new_sens[(i, j)]);
            }
        }
        let sens_np = sens_flat
            .into_pyarray(py)
            .reshape([self.dimension, num_params])?;

        Ok((state_np, sens_np))
    }

    /// Advance state, variational matrix (STM), and sensitivity matrix by one timestep.
    ///
    /// Propagates both matrices simultaneously for complete uncertainty quantification:
    /// - STM (Phi): Maps initial state uncertainties to current state uncertainties
    /// - Sensitivity (S): Maps parameter uncertainties to state uncertainties
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (np.ndarray): State vector at time t
    ///     phi (np.ndarray): State transition matrix at time t (state_dim x state_dim)
    ///     sens (np.ndarray): Sensitivity matrix at time t (state_dim x param_dim)
    ///     params (np.ndarray): Parameter vector
    ///     dt (float, optional): Integration timestep
    ///
    /// Returns:
    ///     tuple: (new_state, new_phi, new_sensitivity)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     # Setup integrator with Jacobian and sensitivity providers
    ///     integrator = bh.RK4Integrator(6, dynamics, jacobian, sensitivity)
    ///     state = np.array([...])
    ///     phi = np.eye(6)
    ///     sens = np.zeros((6, 2))  # 6 state dims, 2 params
    ///     params = np.array([1.0, 2.0])
    ///
    ///     new_state, new_phi, new_sens = integrator.step_with_varmat_sensmat(
    ///         0.0, state, phi, sens, params, 1.0
    ///     )
    ///     ```
    #[pyo3(signature = (t, state, phi, sens, params, dt=None))]
    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    pub fn step_with_varmat_sensmat<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        state: PyReadonlyArray1<f64>,
        phi: PyReadonlyArray2<f64>,
        sens: PyReadonlyArray2<f64>,
        params: PyReadonlyArray1<f64>,
        dt: Option<f64>,
    ) -> PyResult<(
        Bound<'py, PyArray<f64, Ix1>>,
        Bound<'py, PyArray<f64, Ix2>>,
        Bound<'py, PyArray<f64, Ix2>>,
    )> {
        // Convert state to DVector
        let state_vec = state.as_slice()?;
        let state_dvec = na::DVector::from_vec(state_vec.to_vec());

        // Convert phi to DMatrix
        let phi_array = phi.as_array();
        let mut phi_dmat = na::DMatrix::<f64>::zeros(self.dimension, self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_dmat[(i, j)] = phi_array[[i, j]];
            }
        }

        // Convert sensitivity matrix to DMatrix
        let sens_array = sens.as_array();
        let sens_shape = sens_array.shape();
        let num_params = sens_shape[1];
        let mut sens_dmat = na::DMatrix::<f64>::zeros(self.dimension, num_params);
        for i in 0..self.dimension {
            for j in 0..num_params {
                sens_dmat[(i, j)] = sens_array[[i, j]];
            }
        }

        // Convert params to DVector
        let params_vec = params.as_slice()?;
        let params_dvec = na::DVector::from_vec(params_vec.to_vec());

        // Perform step
        let result = self.inner.step_with_varmat_sensmat(
            t, state_dvec, phi_dmat, sens_dmat, &params_dvec, dt
        );

        // Convert results to NumPy
        let state_np = result.state.as_slice().to_pyarray(py);

        let new_phi = result.phi.expect("Variational matrix should be present");
        let mut phi_flat = Vec::with_capacity(self.dimension * self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_flat.push(new_phi[(i, j)]);
            }
        }
        let phi_np = phi_flat
            .into_pyarray(py)
            .reshape([self.dimension, self.dimension])?;

        let new_sens = result.sens.expect("Sensitivity matrix should be present");
        let mut sens_flat = Vec::with_capacity(self.dimension * num_params);
        for i in 0..self.dimension {
            for j in 0..num_params {
                sens_flat.push(new_sens[(i, j)]);
            }
        }
        let sens_np = sens_flat
            .into_pyarray(py)
            .reshape([self.dimension, num_params])?;

        Ok((state_np, phi_np, sens_np))
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
/// Args:
///     dimension (int): State vector dimension
///     dynamics_fn (callable): Dynamics function with signature (t: float, state: ndarray) -> ndarray
///     jacobian (DAnalyticJacobian or DNumericalJacobian, optional): Jacobian provider for variational matrix propagation
///     config (IntegratorConfig, optional): Integration configuration
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
    #[allow(deprecated)]
    #[new]
    #[pyo3(signature = (dimension, dynamics_fn, jacobian=None, sensitivity=None, control_fn=None, config=None))]
    pub fn new(
        py: Python<'_>,
        dimension: usize,
        dynamics_fn: Py<PyAny>,
        jacobian: Option<Py<PyAny>>,
        sensitivity: Option<Py<PyAny>>,
        control_fn: Option<Py<PyAny>>,
        config: Option<PyIntegratorConfig>,
    ) -> PyResult<Self> {
        // Create Rust closure for dynamics (Arc for thread safety)
        // Note: Uses 3-argument closure (t, state, params) for sensitivity support
        let dynamics_closure = {
            let dynamics_fn_arc = Arc::new(dynamics_fn.clone_ref(py));
            move |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| -> na::DVector<f64> {
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

        // Create Rust closure for control input if provided
        let control: crate::integrators::traits::DControlInput = if let Some(ctrl_fn) = control_fn {
            let ctrl_fn_arc = Arc::new(ctrl_fn.clone_ref(py));
            Some(Box::new(move |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| -> na::DVector<f64> {
                Python::with_gil(|py| {
                    let state_py = state.as_slice().to_pyarray(py);
                    let result = ctrl_fn_arc
                        .call1(py, (t, state_py))
                        .expect("Failed to call control function");
                    let result_vec = pyany_to_f64_array1(result.bind(py), Some(dimension))
                        .expect("Control function returned invalid array");
                    na::DVector::from_vec(result_vec)
                })
            }))
        } else {
            None
        };

        // Handle optional Jacobian provider (same as RK4)
        let varmat: Option<Box<dyn DJacobianProvider>> = if let Some(jac_obj) = jacobian {
            Python::with_gil(|py| {
                let jac = jac_obj.bind(py);

                if let Ok(num_jac) = jac.extract::<PyRef<PyDNumericalJacobian>>() {
                    let jac_arc = Arc::new(num_jac.dynamics_fn.clone_ref(py));
                    let jac_method = num_jac.method;
                    let jac_pert = num_jac.perturbation;

                    let jac_closure = move |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| -> na::DVector<f64> {
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

                    let jac_closure = move |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| -> na::DMatrix<f64> {
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

        // Handle optional sensitivity provider
        let sensmat: Option<Box<dyn DSensitivityProvider>> = if let Some(sens_obj) = sensitivity {
            Python::with_gil(|py| {
                let sens = sens_obj.bind(py);

                // Try to extract as PyDNumericalSensitivity
                if let Ok(num_sens) = sens.extract::<PyRef<PyDNumericalSensitivity>>() {
                    let sens_arc = Arc::new(num_sens.dynamics_fn.clone_ref(py));
                    let sens_method = num_sens.method;
                    let sens_pert = num_sens.perturbation;

                    let sens_closure = move |t: f64, state: &na::DVector<f64>, params: &na::DVector<f64>| -> na::DVector<f64> {
                        Python::with_gil(|py| {
                            let state_py = state.as_slice().to_pyarray(py);
                            let params_py = params.as_slice().to_pyarray(py);
                            let result = sens_arc
                                .call1(py, (t, state_py, params_py))
                                .expect("Failed to call sensitivity dynamics");
                            let result_vec = pyany_to_f64_array1(result.bind(py), Some(state.len()))
                                .expect("Sensitivity dynamics returned invalid array");
                            na::DVector::from_vec(result_vec)
                        })
                    };

                    let mut provider = match sens_method {
                        crate::math::jacobian::DifferenceMethod::Forward => {
                            crate::math::sensitivity::DNumericalSensitivity::forward(Box::new(sens_closure))
                        }
                        crate::math::jacobian::DifferenceMethod::Central => {
                            crate::math::sensitivity::DNumericalSensitivity::central(Box::new(sens_closure))
                        }
                        crate::math::jacobian::DifferenceMethod::Backward => {
                            crate::math::sensitivity::DNumericalSensitivity::central(Box::new(sens_closure))
                        }
                    };

                    provider = match sens_pert {
                        crate::math::jacobian::PerturbationStrategy::Adaptive {
                            scale_factor,
                            min_threshold,
                        } => provider.with_strategy(crate::math::jacobian::PerturbationStrategy::Adaptive {
                            scale_factor,
                            min_threshold,
                        }),
                        crate::math::jacobian::PerturbationStrategy::Fixed(offset) => {
                            provider.with_strategy(crate::math::jacobian::PerturbationStrategy::Fixed(offset))
                        }
                        crate::math::jacobian::PerturbationStrategy::Percentage(pct) => {
                            provider.with_strategy(crate::math::jacobian::PerturbationStrategy::Percentage(pct))
                        }
                    };

                    Ok(Some(Box::new(provider) as Box<dyn DSensitivityProvider>))
                } else if let Ok(ana_sens) = sens.extract::<PyRef<PyDAnalyticSensitivity>>() {
                    let sens_arc = Arc::new(ana_sens.sensitivity_fn.clone_ref(py));

                    let sens_closure = move |t: f64, state: &na::DVector<f64>, params: &na::DVector<f64>| -> na::DMatrix<f64> {
                        Python::with_gil(|py| {
                            let state_py = state.as_slice().to_pyarray(py);
                            let params_py = params.as_slice().to_pyarray(py);
                            let result = sens_arc
                                .call1(py, (t, state_py, params_py))
                                .expect("Failed to call sensitivity function");
                            let state_dim = state.len();
                            let param_dim = params.len();
                            let sens_matrix_vec =
                                pyany_to_f64_array2(result.bind(py), Some((state_dim, param_dim)))
                                    .expect("Sensitivity function returned invalid array");

                            let mut sens_matrix = na::DMatrix::<f64>::zeros(state_dim, param_dim);
                            for i in 0..state_dim {
                                for j in 0..param_dim {
                                    sens_matrix[(i, j)] = sens_matrix_vec[i][j];
                                }
                            }
                            sens_matrix
                        })
                    };

                    let provider = crate::math::sensitivity::DAnalyticSensitivity::new(Box::new(sens_closure));
                    Ok(Some(Box::new(provider) as Box<dyn DSensitivityProvider>))
                } else {
                    Err(exceptions::PyTypeError::new_err(
                        "sensitivity must be NumericalSensitivity or AnalyticSensitivity",
                    ))
                }
            })?
        } else {
            None
        };

        // Create integrator
        let inner = if let Some(cfg) = config {
            RKF45DIntegrator::with_config(dimension, Box::new(dynamics_closure), varmat, sensmat, control, cfg.inner)
        } else {
            RKF45DIntegrator::new(dimension, Box::new(dynamics_closure), varmat, sensmat, control)
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
        let result = self.inner.step(t, state_dvec, None, Some(dt));

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
        let result = self.inner.step_with_varmat(t, state_dvec, None, phi_dmat, Some(dt));

        // Convert results to NumPy
        let state_np = result.state.as_slice().to_pyarray(py);

        let new_phi = result.phi.expect("Variational matrix should be present");
        let mut phi_flat = Vec::with_capacity(self.dimension * self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_flat.push(new_phi[(i, j)]);
            }
        }
        let dt_used = result.dt_used;
        let error_est = result.error_estimate.expect("Error estimate should be present for adaptive integrator");
        let dt_next = result.dt_next;
        let phi_np = phi_flat
            .into_pyarray(py)
            .reshape([self.dimension, self.dimension])?;

        Ok((state_np, phi_np, dt_used, error_est, dt_next))
    }

    /// Advance state and sensitivity matrix with adaptive step control.
    ///
    /// Propagates the sensitivity matrix S that maps parameter uncertainties
    /// to state uncertainties. The sensitivity evolves according to dS/dt = A*S + B.
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (np.ndarray): State vector at time t
    ///     sens (np.ndarray): Sensitivity matrix at time t (state_dim x param_dim)
    ///     params (np.ndarray): Parameter vector
    ///     dt (float): Requested timestep
    ///
    /// Returns:
    ///     tuple: (new_state, new_sensitivity, dt_used, error_estimate, dt_next)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     # Setup integrator with Jacobian and sensitivity providers
    ///     config = bh.IntegratorConfig.adaptive(abs_tol=1e-9, rel_tol=1e-6)
    ///     integrator = bh.RKF45Integrator(6, dynamics, jacobian, sensitivity, config)
    ///     state = np.array([...])
    ///     sens = np.zeros((6, 2))  # 6 state dims, 2 params
    ///     params = np.array([1.0, 2.0])
    ///
    ///     new_state, new_sens, dt_used, error, dt_next = integrator.step_with_sensmat(
    ///         0.0, state, sens, params, 1.0
    ///     )
    ///     ```
    #[allow(clippy::type_complexity)]
    pub fn step_with_sensmat<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        state: PyReadonlyArray1<f64>,
        sens: PyReadonlyArray2<f64>,
        params: PyReadonlyArray1<f64>,
        dt: f64,
    ) -> PyResult<(
        Bound<'py, PyArray<f64, Ix1>>,
        Bound<'py, PyArray<f64, Ix2>>,
        f64,
        f64,
        f64,
    )> {
        // Convert state to DVector
        let state_vec = state.as_slice()?;
        let state_dvec = na::DVector::from_vec(state_vec.to_vec());

        // Convert sensitivity matrix to DMatrix
        let sens_array = sens.as_array();
        let sens_shape = sens_array.shape();
        let num_params = sens_shape[1];
        let mut sens_dmat = na::DMatrix::<f64>::zeros(self.dimension, num_params);
        for i in 0..self.dimension {
            for j in 0..num_params {
                sens_dmat[(i, j)] = sens_array[[i, j]];
            }
        }

        // Convert params to DVector
        let params_vec = params.as_slice()?;
        let params_dvec = na::DVector::from_vec(params_vec.to_vec());

        // Perform step
        let result = self.inner.step_with_sensmat(
            t, state_dvec, sens_dmat, &params_dvec, Some(dt)
        );

        // Convert results to NumPy
        let state_np = result.state.as_slice().to_pyarray(py);

        let new_sens = result.sens.expect("Sensitivity matrix should be present");
        let mut sens_flat = Vec::with_capacity(self.dimension * num_params);
        for i in 0..self.dimension {
            for j in 0..num_params {
                sens_flat.push(new_sens[(i, j)]);
            }
        }
        let dt_used = result.dt_used;
        let error_est = result.error_estimate.expect("Error estimate should be present for adaptive integrator");
        let dt_next = result.dt_next;
        let sens_np = sens_flat
            .into_pyarray(py)
            .reshape([self.dimension, num_params])?;

        Ok((state_np, sens_np, dt_used, error_est, dt_next))
    }

    /// Advance state, variational matrix (STM), and sensitivity matrix with adaptive step control.
    ///
    /// Propagates both matrices simultaneously for complete uncertainty quantification:
    /// - STM (Phi): Maps initial state uncertainties to current state uncertainties
    /// - Sensitivity (S): Maps parameter uncertainties to state uncertainties
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (np.ndarray): State vector at time t
    ///     phi (np.ndarray): State transition matrix at time t (state_dim x state_dim)
    ///     sens (np.ndarray): Sensitivity matrix at time t (state_dim x param_dim)
    ///     params (np.ndarray): Parameter vector
    ///     dt (float): Requested timestep
    ///
    /// Returns:
    ///     tuple: (new_state, new_phi, new_sensitivity, dt_used, error_estimate, dt_next)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     # Setup integrator with Jacobian and sensitivity providers
    ///     config = bh.IntegratorConfig.adaptive(abs_tol=1e-9, rel_tol=1e-6)
    ///     integrator = bh.RKF45Integrator(6, dynamics, jacobian, sensitivity, config)
    ///     state = np.array([...])
    ///     phi = np.eye(6)
    ///     sens = np.zeros((6, 2))  # 6 state dims, 2 params
    ///     params = np.array([1.0, 2.0])
    ///
    ///     new_state, new_phi, new_sens, dt_used, error, dt_next = integrator.step_with_varmat_sensmat(
    ///         0.0, state, phi, sens, params, 1.0
    ///     )
    ///     ```
    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    pub fn step_with_varmat_sensmat<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        state: PyReadonlyArray1<f64>,
        phi: PyReadonlyArray2<f64>,
        sens: PyReadonlyArray2<f64>,
        params: PyReadonlyArray1<f64>,
        dt: f64,
    ) -> PyResult<(
        Bound<'py, PyArray<f64, Ix1>>,
        Bound<'py, PyArray<f64, Ix2>>,
        Bound<'py, PyArray<f64, Ix2>>,
        f64,
        f64,
        f64,
    )> {
        // Convert state to DVector
        let state_vec = state.as_slice()?;
        let state_dvec = na::DVector::from_vec(state_vec.to_vec());

        // Convert phi to DMatrix
        let phi_array = phi.as_array();
        let mut phi_dmat = na::DMatrix::<f64>::zeros(self.dimension, self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_dmat[(i, j)] = phi_array[[i, j]];
            }
        }

        // Convert sensitivity matrix to DMatrix
        let sens_array = sens.as_array();
        let sens_shape = sens_array.shape();
        let num_params = sens_shape[1];
        let mut sens_dmat = na::DMatrix::<f64>::zeros(self.dimension, num_params);
        for i in 0..self.dimension {
            for j in 0..num_params {
                sens_dmat[(i, j)] = sens_array[[i, j]];
            }
        }

        // Convert params to DVector
        let params_vec = params.as_slice()?;
        let params_dvec = na::DVector::from_vec(params_vec.to_vec());

        // Perform step
        let result = self.inner.step_with_varmat_sensmat(
            t, state_dvec, phi_dmat, sens_dmat, &params_dvec, Some(dt)
        );

        // Convert results to NumPy
        let state_np = result.state.as_slice().to_pyarray(py);

        let new_phi = result.phi.expect("Variational matrix should be present");
        let mut phi_flat = Vec::with_capacity(self.dimension * self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_flat.push(new_phi[(i, j)]);
            }
        }
        let phi_np = phi_flat
            .into_pyarray(py)
            .reshape([self.dimension, self.dimension])?;

        let new_sens = result.sens.expect("Sensitivity matrix should be present");
        let mut sens_flat = Vec::with_capacity(self.dimension * num_params);
        for i in 0..self.dimension {
            for j in 0..num_params {
                sens_flat.push(new_sens[(i, j)]);
            }
        }
        let dt_used = result.dt_used;
        let error_est = result.error_estimate.expect("Error estimate should be present for adaptive integrator");
        let dt_next = result.dt_next;
        let sens_np = sens_flat
            .into_pyarray(py)
            .reshape([self.dimension, num_params])?;

        Ok((state_np, phi_np, sens_np, dt_used, error_est, dt_next))
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
/// Args:
///     dimension (int): State vector dimension
///     dynamics_fn (callable): Dynamics function with signature (t: float, state: ndarray) -> ndarray
///     jacobian (DAnalyticJacobian or DNumericalJacobian, optional): Jacobian provider for variational matrix propagation
///     config (IntegratorConfig, optional): Integration configuration
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
    #[allow(deprecated)]
    #[new]
    #[pyo3(signature = (dimension, dynamics_fn, jacobian=None, sensitivity=None, control_fn=None, config=None))]
    pub fn new(
        py: Python<'_>,
        dimension: usize,
        dynamics_fn: Py<PyAny>,
        jacobian: Option<Py<PyAny>>,
        sensitivity: Option<Py<PyAny>>,
        control_fn: Option<Py<PyAny>>,
        config: Option<PyIntegratorConfig>,
    ) -> PyResult<Self> {
        // Create Rust closure for dynamics
        // Note: DP54 uses 3-argument closure (t, state, params) for sensitivity support
        let dynamics_closure = {
            let dynamics_fn_arc = Arc::new(dynamics_fn.clone_ref(py));
            move |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| -> na::DVector<f64> {
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

        // Create Rust closure for control input if provided
        let control: crate::integrators::traits::DControlInput = if let Some(ctrl_fn) = control_fn {
            let ctrl_fn_arc = Arc::new(ctrl_fn.clone_ref(py));
            Some(Box::new(move |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| -> na::DVector<f64> {
                Python::with_gil(|py| {
                    let state_py = state.as_slice().to_pyarray(py);
                    let result = ctrl_fn_arc
                        .call1(py, (t, state_py))
                        .expect("Failed to call control function");
                    let result_vec = pyany_to_f64_array1(result.bind(py), Some(dimension))
                        .expect("Control function returned invalid array");
                    na::DVector::from_vec(result_vec)
                })
            }))
        } else {
            None
        };

        // Handle optional Jacobian provider
        let varmat: Option<Box<dyn DJacobianProvider>> = if let Some(jac_obj) = jacobian {
            Python::with_gil(|py| {
                let jac = jac_obj.bind(py);

                if let Ok(num_jac) = jac.extract::<PyRef<PyDNumericalJacobian>>() {
                    let jac_arc = Arc::new(num_jac.dynamics_fn.clone_ref(py));
                    let jac_method = num_jac.method;
                    let jac_pert = num_jac.perturbation;

                    let jac_closure = move |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| -> na::DVector<f64> {
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

                    let jac_closure = move |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| -> na::DMatrix<f64> {
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

        // Handle optional sensitivity provider
        let sensmat: Option<Box<dyn DSensitivityProvider>> = if let Some(sens_obj) = sensitivity {
            Python::with_gil(|py| {
                let sens = sens_obj.bind(py);

                if let Ok(num_sens) = sens.extract::<PyRef<PyDNumericalSensitivity>>() {
                    let sens_arc = Arc::new(num_sens.dynamics_fn.clone_ref(py));
                    let sens_method = num_sens.method;
                    let sens_pert = num_sens.perturbation;

                    let sens_closure = move |t: f64, state: &na::DVector<f64>, params: &na::DVector<f64>| -> na::DVector<f64> {
                        Python::with_gil(|py| {
                            let state_py = state.as_slice().to_pyarray(py);
                            let params_py = params.as_slice().to_pyarray(py);
                            let result = sens_arc
                                .call1(py, (t, state_py, params_py))
                                .expect("Failed to call sensitivity dynamics");
                            let result_vec = pyany_to_f64_array1(result.bind(py), Some(state.len()))
                                .expect("Sensitivity dynamics returned invalid array");
                            na::DVector::from_vec(result_vec)
                        })
                    };

                    let mut provider = match sens_method {
                        crate::math::jacobian::DifferenceMethod::Forward => {
                            crate::math::sensitivity::DNumericalSensitivity::forward(Box::new(sens_closure))
                        }
                        crate::math::jacobian::DifferenceMethod::Central => {
                            crate::math::sensitivity::DNumericalSensitivity::central(Box::new(sens_closure))
                        }
                        crate::math::jacobian::DifferenceMethod::Backward => {
                            crate::math::sensitivity::DNumericalSensitivity::central(Box::new(sens_closure))
                        }
                    };

                    provider = match sens_pert {
                        crate::math::jacobian::PerturbationStrategy::Adaptive {
                            scale_factor,
                            min_threshold,
                        } => provider.with_strategy(crate::math::jacobian::PerturbationStrategy::Adaptive {
                            scale_factor,
                            min_threshold,
                        }),
                        crate::math::jacobian::PerturbationStrategy::Fixed(offset) => {
                            provider.with_strategy(crate::math::jacobian::PerturbationStrategy::Fixed(offset))
                        }
                        crate::math::jacobian::PerturbationStrategy::Percentage(pct) => {
                            provider.with_strategy(crate::math::jacobian::PerturbationStrategy::Percentage(pct))
                        }
                    };

                    Ok(Some(Box::new(provider) as Box<dyn DSensitivityProvider>))
                } else if let Ok(ana_sens) = sens.extract::<PyRef<PyDAnalyticSensitivity>>() {
                    let sens_arc = Arc::new(ana_sens.sensitivity_fn.clone_ref(py));

                    let sens_closure = move |t: f64, state: &na::DVector<f64>, params: &na::DVector<f64>| -> na::DMatrix<f64> {
                        Python::with_gil(|py| {
                            let state_py = state.as_slice().to_pyarray(py);
                            let params_py = params.as_slice().to_pyarray(py);
                            let result = sens_arc
                                .call1(py, (t, state_py, params_py))
                                .expect("Failed to call sensitivity function");
                            let state_dim = state.len();
                            let param_dim = params.len();
                            let sens_matrix_vec =
                                pyany_to_f64_array2(result.bind(py), Some((state_dim, param_dim)))
                                    .expect("Sensitivity function returned invalid array");

                            let mut sens_matrix = na::DMatrix::<f64>::zeros(state_dim, param_dim);
                            for i in 0..state_dim {
                                for j in 0..param_dim {
                                    sens_matrix[(i, j)] = sens_matrix_vec[i][j];
                                }
                            }
                            sens_matrix
                        })
                    };

                    let provider = crate::math::sensitivity::DAnalyticSensitivity::new(Box::new(sens_closure));
                    Ok(Some(Box::new(provider) as Box<dyn DSensitivityProvider>))
                } else {
                    Err(exceptions::PyTypeError::new_err(
                        "sensitivity must be NumericalSensitivity or AnalyticSensitivity",
                    ))
                }
            })?
        } else {
            None
        };

        // Create integrator
        let inner = if let Some(cfg) = config {
            DormandPrince54DIntegrator::with_config(dimension, Box::new(dynamics_closure), varmat, sensmat, control, cfg.inner)
        } else {
            DormandPrince54DIntegrator::new(dimension, Box::new(dynamics_closure), varmat, sensmat, control)
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

        let result = self.inner.step(t, state_dvec, None, Some(dt));

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

        let result = self.inner.step_with_varmat(t, state_dvec, None, phi_dmat, Some(dt));

        let state_np = result.state.as_slice().to_pyarray(py);

        let new_phi = result.phi.expect("Variational matrix should be present");
        let mut phi_flat = Vec::with_capacity(self.dimension * self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_flat.push(new_phi[(i, j)]);
            }
        }
        let dt_used = result.dt_used;
        let error_est = result.error_estimate.expect("Error estimate should be present for adaptive integrator");
        let dt_next = result.dt_next;
        let phi_np = phi_flat
            .into_pyarray(py)
            .reshape([self.dimension, self.dimension])?;

        Ok((state_np, phi_np, dt_used, error_est, dt_next))
    }

    /// Advance state and sensitivity matrix with adaptive step control.
    ///
    /// Propagates the sensitivity matrix S that maps parameter uncertainties
    /// to state uncertainties. The sensitivity evolves according to dS/dt = A*S + B.
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (np.ndarray): State vector at time t
    ///     sens (np.ndarray): Sensitivity matrix at time t (state_dim x param_dim)
    ///     params (np.ndarray): Parameter vector
    ///     dt (float): Requested timestep
    ///
    /// Returns:
    ///     tuple: (new_state, new_sensitivity, dt_used, error_estimate, dt_next)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     # Setup integrator with Jacobian and sensitivity providers
    ///     config = bh.IntegratorConfig.adaptive(abs_tol=1e-9, rel_tol=1e-6)
    ///     integrator = bh.DP54Integrator(6, dynamics, jacobian, sensitivity, config)
    ///     state = np.array([...])
    ///     sens = np.zeros((6, 2))  # 6 state dims, 2 params
    ///     params = np.array([1.0, 2.0])
    ///
    ///     new_state, new_sens, dt_used, error, dt_next = integrator.step_with_sensmat(
    ///         0.0, state, sens, params, 1.0
    ///     )
    ///     ```
    #[allow(clippy::type_complexity)]
    pub fn step_with_sensmat<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        state: PyReadonlyArray1<f64>,
        sens: PyReadonlyArray2<f64>,
        params: PyReadonlyArray1<f64>,
        dt: f64,
    ) -> PyResult<(
        Bound<'py, PyArray<f64, Ix1>>,
        Bound<'py, PyArray<f64, Ix2>>,
        f64,
        f64,
        f64,
    )> {
        // Convert state to DVector
        let state_vec = state.as_slice()?;
        let state_dvec = na::DVector::from_vec(state_vec.to_vec());

        // Convert sensitivity matrix to DMatrix
        let sens_array = sens.as_array();
        let sens_shape = sens_array.shape();
        let num_params = sens_shape[1];
        let mut sens_dmat = na::DMatrix::<f64>::zeros(self.dimension, num_params);
        for i in 0..self.dimension {
            for j in 0..num_params {
                sens_dmat[(i, j)] = sens_array[[i, j]];
            }
        }

        // Convert params to DVector
        let params_vec = params.as_slice()?;
        let params_dvec = na::DVector::from_vec(params_vec.to_vec());

        // Perform step
        let result = self.inner.step_with_sensmat(
            t, state_dvec, sens_dmat, &params_dvec, Some(dt)
        );

        // Convert results to NumPy
        let state_np = result.state.as_slice().to_pyarray(py);

        let new_sens = result.sens.expect("Sensitivity matrix should be present");
        let mut sens_flat = Vec::with_capacity(self.dimension * num_params);
        for i in 0..self.dimension {
            for j in 0..num_params {
                sens_flat.push(new_sens[(i, j)]);
            }
        }
        let dt_used = result.dt_used;
        let error_est = result.error_estimate.expect("Error estimate should be present for adaptive integrator");
        let dt_next = result.dt_next;
        let sens_np = sens_flat
            .into_pyarray(py)
            .reshape([self.dimension, num_params])?;

        Ok((state_np, sens_np, dt_used, error_est, dt_next))
    }

    /// Advance state, variational matrix (STM), and sensitivity matrix with adaptive step control.
    ///
    /// Propagates both matrices simultaneously for complete uncertainty quantification:
    /// - STM (Phi): Maps initial state uncertainties to current state uncertainties
    /// - Sensitivity (S): Maps parameter uncertainties to state uncertainties
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (np.ndarray): State vector at time t
    ///     phi (np.ndarray): State transition matrix at time t (state_dim x state_dim)
    ///     sens (np.ndarray): Sensitivity matrix at time t (state_dim x param_dim)
    ///     params (np.ndarray): Parameter vector
    ///     dt (float): Requested timestep
    ///
    /// Returns:
    ///     tuple: (new_state, new_phi, new_sensitivity, dt_used, error_estimate, dt_next)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     # Setup integrator with Jacobian and sensitivity providers
    ///     config = bh.IntegratorConfig.adaptive(abs_tol=1e-9, rel_tol=1e-6)
    ///     integrator = bh.DP54Integrator(6, dynamics, jacobian, sensitivity, config)
    ///     state = np.array([...])
    ///     phi = np.eye(6)
    ///     sens = np.zeros((6, 2))  # 6 state dims, 2 params
    ///     params = np.array([1.0, 2.0])
    ///
    ///     new_state, new_phi, new_sens, dt_used, error, dt_next = integrator.step_with_varmat_sensmat(
    ///         0.0, state, phi, sens, params, 1.0
    ///     )
    ///     ```
    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    pub fn step_with_varmat_sensmat<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        state: PyReadonlyArray1<f64>,
        phi: PyReadonlyArray2<f64>,
        sens: PyReadonlyArray2<f64>,
        params: PyReadonlyArray1<f64>,
        dt: f64,
    ) -> PyResult<(
        Bound<'py, PyArray<f64, Ix1>>,
        Bound<'py, PyArray<f64, Ix2>>,
        Bound<'py, PyArray<f64, Ix2>>,
        f64,
        f64,
        f64,
    )> {
        // Convert state to DVector
        let state_vec = state.as_slice()?;
        let state_dvec = na::DVector::from_vec(state_vec.to_vec());

        // Convert phi to DMatrix
        let phi_array = phi.as_array();
        let mut phi_dmat = na::DMatrix::<f64>::zeros(self.dimension, self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_dmat[(i, j)] = phi_array[[i, j]];
            }
        }

        // Convert sensitivity matrix to DMatrix
        let sens_array = sens.as_array();
        let sens_shape = sens_array.shape();
        let num_params = sens_shape[1];
        let mut sens_dmat = na::DMatrix::<f64>::zeros(self.dimension, num_params);
        for i in 0..self.dimension {
            for j in 0..num_params {
                sens_dmat[(i, j)] = sens_array[[i, j]];
            }
        }

        // Convert params to DVector
        let params_vec = params.as_slice()?;
        let params_dvec = na::DVector::from_vec(params_vec.to_vec());

        // Perform step
        let result = self.inner.step_with_varmat_sensmat(
            t, state_dvec, phi_dmat, sens_dmat, &params_dvec, Some(dt)
        );

        // Convert results to NumPy
        let state_np = result.state.as_slice().to_pyarray(py);

        let new_phi = result.phi.expect("Variational matrix should be present");
        let mut phi_flat = Vec::with_capacity(self.dimension * self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_flat.push(new_phi[(i, j)]);
            }
        }
        let phi_np = phi_flat
            .into_pyarray(py)
            .reshape([self.dimension, self.dimension])?;

        let new_sens = result.sens.expect("Sensitivity matrix should be present");
        let mut sens_flat = Vec::with_capacity(self.dimension * num_params);
        for i in 0..self.dimension {
            for j in 0..num_params {
                sens_flat.push(new_sens[(i, j)]);
            }
        }
        let dt_used = result.dt_used;
        let error_est = result.error_estimate.expect("Error estimate should be present for adaptive integrator");
        let dt_next = result.dt_next;
        let sens_np = sens_flat
            .into_pyarray(py)
            .reshape([self.dimension, num_params])?;

        Ok((state_np, phi_np, sens_np, dt_used, error_est, dt_next))
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
/// Args:
///     dimension (int): State vector dimension (must be even: position + velocity)
///     dynamics_fn (callable): Dynamics function with signature (t: float, state: ndarray) -> ndarray
///     jacobian (DAnalyticJacobian or DNumericalJacobian, optional): Jacobian provider for variational matrix propagation
///     config (IntegratorConfig, optional): Integration configuration
///
/// Raises:
///     ValueError: If dimension is not even
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
    #[allow(deprecated)]
    #[new]
    #[pyo3(signature = (dimension, dynamics_fn, jacobian=None, sensitivity=None, config=None))]
    pub fn new(
        py: Python<'_>,
        dimension: usize,
        dynamics_fn: Py<PyAny>,
        jacobian: Option<Py<PyAny>>,
        sensitivity: Option<Py<PyAny>>,
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
            move |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| -> na::DVector<f64> {
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

                    let jac_closure = move |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| -> na::DVector<f64> {
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

                    let jac_closure = move |t: f64, state: &na::DVector<f64>, _params: Option<&na::DVector<f64>>| -> na::DMatrix<f64> {
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

        // Handle optional sensitivity provider
        let sensmat: Option<Box<dyn DSensitivityProvider>> = if let Some(sens_obj) = sensitivity {
            Python::with_gil(|py| {
                let sens = sens_obj.bind(py);

                if let Ok(num_sens) = sens.extract::<PyRef<PyDNumericalSensitivity>>() {
                    let sens_arc = Arc::new(num_sens.dynamics_fn.clone_ref(py));
                    let sens_method = num_sens.method;
                    let sens_pert = num_sens.perturbation;

                    let sens_closure = move |t: f64, state: &na::DVector<f64>, params: &na::DVector<f64>| -> na::DVector<f64> {
                        Python::with_gil(|py| {
                            let state_py = state.as_slice().to_pyarray(py);
                            let params_py = params.as_slice().to_pyarray(py);
                            let result = sens_arc
                                .call1(py, (t, state_py, params_py))
                                .expect("Failed to call sensitivity dynamics");
                            let result_vec = pyany_to_f64_array1(result.bind(py), Some(state.len()))
                                .expect("Sensitivity dynamics returned invalid array");
                            na::DVector::from_vec(result_vec)
                        })
                    };

                    let mut provider = match sens_method {
                        crate::math::jacobian::DifferenceMethod::Forward => {
                            crate::math::sensitivity::DNumericalSensitivity::forward(Box::new(sens_closure))
                        }
                        crate::math::jacobian::DifferenceMethod::Central => {
                            crate::math::sensitivity::DNumericalSensitivity::central(Box::new(sens_closure))
                        }
                        crate::math::jacobian::DifferenceMethod::Backward => {
                            crate::math::sensitivity::DNumericalSensitivity::central(Box::new(sens_closure))
                        }
                    };

                    provider = match sens_pert {
                        crate::math::jacobian::PerturbationStrategy::Adaptive {
                            scale_factor,
                            min_threshold,
                        } => provider.with_strategy(crate::math::jacobian::PerturbationStrategy::Adaptive {
                            scale_factor,
                            min_threshold,
                        }),
                        crate::math::jacobian::PerturbationStrategy::Fixed(offset) => {
                            provider.with_strategy(crate::math::jacobian::PerturbationStrategy::Fixed(offset))
                        }
                        crate::math::jacobian::PerturbationStrategy::Percentage(pct) => {
                            provider.with_strategy(crate::math::jacobian::PerturbationStrategy::Percentage(pct))
                        }
                    };

                    Ok(Some(Box::new(provider) as Box<dyn DSensitivityProvider>))
                } else if let Ok(ana_sens) = sens.extract::<PyRef<PyDAnalyticSensitivity>>() {
                    let sens_arc = Arc::new(ana_sens.sensitivity_fn.clone_ref(py));

                    let sens_closure = move |t: f64, state: &na::DVector<f64>, params: &na::DVector<f64>| -> na::DMatrix<f64> {
                        Python::with_gil(|py| {
                            let state_py = state.as_slice().to_pyarray(py);
                            let params_py = params.as_slice().to_pyarray(py);
                            let result = sens_arc
                                .call1(py, (t, state_py, params_py))
                                .expect("Failed to call sensitivity function");
                            let state_dim = state.len();
                            let param_dim = params.len();
                            let sens_matrix_vec =
                                pyany_to_f64_array2(result.bind(py), Some((state_dim, param_dim)))
                                    .expect("Sensitivity function returned invalid array");

                            let mut sens_matrix = na::DMatrix::<f64>::zeros(state_dim, param_dim);
                            for i in 0..state_dim {
                                for j in 0..param_dim {
                                    sens_matrix[(i, j)] = sens_matrix_vec[i][j];
                                }
                            }
                            sens_matrix
                        })
                    };

                    let provider = crate::math::sensitivity::DAnalyticSensitivity::new(Box::new(sens_closure));
                    Ok(Some(Box::new(provider) as Box<dyn DSensitivityProvider>))
                } else {
                    Err(exceptions::PyTypeError::new_err(
                        "sensitivity must be NumericalSensitivity or AnalyticSensitivity",
                    ))
                }
            })?
        } else {
            None
        };

        // Create integrator
        let inner = if let Some(cfg) = config {
            RKN1210DIntegrator::with_config(dimension, Box::new(dynamics_closure), varmat, sensmat, None, cfg.inner)
        } else {
            RKN1210DIntegrator::new(dimension, Box::new(dynamics_closure), varmat, sensmat, None)
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

        let result = self.inner.step(t, state_dvec, None, Some(dt));

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

        let result = self.inner.step_with_varmat(t, state_dvec, None, phi_dmat, Some(dt));

        let state_np = result.state.as_slice().to_pyarray(py);

        let new_phi = result.phi.expect("Variational matrix should be present");
        let mut phi_flat = Vec::with_capacity(self.dimension * self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_flat.push(new_phi[(i, j)]);
            }
        }
        let dt_used = result.dt_used;
        let error_est = result.error_estimate.expect("Error estimate should be present for adaptive integrator");
        let dt_next = result.dt_next;
        let phi_np = phi_flat
            .into_pyarray(py)
            .reshape([self.dimension, self.dimension])?;

        Ok((state_np, phi_np, dt_used, error_est, dt_next))
    }

    /// Advance state and sensitivity matrix with adaptive step control.
    ///
    /// Propagates the sensitivity matrix S that maps parameter uncertainties
    /// to state uncertainties. The sensitivity evolves according to dS/dt = A*S + B.
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (np.ndarray): State vector at time t
    ///     sens (np.ndarray): Sensitivity matrix at time t (state_dim x param_dim)
    ///     params (np.ndarray): Parameter vector
    ///     dt (float): Requested timestep
    ///
    /// Returns:
    ///     tuple: (new_state, new_sensitivity, dt_used, error_estimate, dt_next)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     # Setup integrator with Jacobian and sensitivity providers
    ///     config = bh.IntegratorConfig.adaptive(abs_tol=1e-9, rel_tol=1e-6)
    ///     integrator = bh.RKN1210Integrator(6, dynamics, jacobian, sensitivity, config)
    ///     state = np.array([...])
    ///     sens = np.zeros((6, 2))  # 6 state dims, 2 params
    ///     params = np.array([1.0, 2.0])
    ///
    ///     new_state, new_sens, dt_used, error, dt_next = integrator.step_with_sensmat(
    ///         0.0, state, sens, params, 1.0
    ///     )
    ///     ```
    #[allow(clippy::type_complexity)]
    pub fn step_with_sensmat<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        state: PyReadonlyArray1<f64>,
        sens: PyReadonlyArray2<f64>,
        params: PyReadonlyArray1<f64>,
        dt: f64,
    ) -> PyResult<(
        Bound<'py, PyArray<f64, Ix1>>,
        Bound<'py, PyArray<f64, Ix2>>,
        f64,
        f64,
        f64,
    )> {
        // Convert state to DVector
        let state_vec = state.as_slice()?;
        let state_dvec = na::DVector::from_vec(state_vec.to_vec());

        // Convert sensitivity matrix to DMatrix
        let sens_array = sens.as_array();
        let sens_shape = sens_array.shape();
        let num_params = sens_shape[1];
        let mut sens_dmat = na::DMatrix::<f64>::zeros(self.dimension, num_params);
        for i in 0..self.dimension {
            for j in 0..num_params {
                sens_dmat[(i, j)] = sens_array[[i, j]];
            }
        }

        // Convert params to DVector
        let params_vec = params.as_slice()?;
        let params_dvec = na::DVector::from_vec(params_vec.to_vec());

        // Perform step
        let result = self.inner.step_with_sensmat(
            t, state_dvec, sens_dmat, &params_dvec, Some(dt)
        );

        // Convert results to NumPy
        let state_np = result.state.as_slice().to_pyarray(py);

        let new_sens = result.sens.expect("Sensitivity matrix should be present");
        let mut sens_flat = Vec::with_capacity(self.dimension * num_params);
        for i in 0..self.dimension {
            for j in 0..num_params {
                sens_flat.push(new_sens[(i, j)]);
            }
        }
        let dt_used = result.dt_used;
        let error_est = result.error_estimate.expect("Error estimate should be present for adaptive integrator");
        let dt_next = result.dt_next;
        let sens_np = sens_flat
            .into_pyarray(py)
            .reshape([self.dimension, num_params])?;

        Ok((state_np, sens_np, dt_used, error_est, dt_next))
    }

    /// Advance state, variational matrix (STM), and sensitivity matrix with adaptive step control.
    ///
    /// Propagates both matrices simultaneously for complete uncertainty quantification:
    /// - STM (Phi): Maps initial state uncertainties to current state uncertainties
    /// - Sensitivity (S): Maps parameter uncertainties to state uncertainties
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (np.ndarray): State vector at time t
    ///     phi (np.ndarray): State transition matrix at time t (state_dim x state_dim)
    ///     sens (np.ndarray): Sensitivity matrix at time t (state_dim x param_dim)
    ///     params (np.ndarray): Parameter vector
    ///     dt (float): Requested timestep
    ///
    /// Returns:
    ///     tuple: (new_state, new_phi, new_sensitivity, dt_used, error_estimate, dt_next)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     # Setup integrator with Jacobian and sensitivity providers
    ///     config = bh.IntegratorConfig.adaptive(abs_tol=1e-9, rel_tol=1e-6)
    ///     integrator = bh.RKN1210Integrator(6, dynamics, jacobian, sensitivity, config)
    ///     state = np.array([...])
    ///     phi = np.eye(6)
    ///     sens = np.zeros((6, 2))  # 6 state dims, 2 params
    ///     params = np.array([1.0, 2.0])
    ///
    ///     new_state, new_phi, new_sens, dt_used, error, dt_next = integrator.step_with_varmat_sensmat(
    ///         0.0, state, phi, sens, params, 1.0
    ///     )
    ///     ```
    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    pub fn step_with_varmat_sensmat<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        state: PyReadonlyArray1<f64>,
        phi: PyReadonlyArray2<f64>,
        sens: PyReadonlyArray2<f64>,
        params: PyReadonlyArray1<f64>,
        dt: f64,
    ) -> PyResult<(
        Bound<'py, PyArray<f64, Ix1>>,
        Bound<'py, PyArray<f64, Ix2>>,
        Bound<'py, PyArray<f64, Ix2>>,
        f64,
        f64,
        f64,
    )> {
        // Convert state to DVector
        let state_vec = state.as_slice()?;
        let state_dvec = na::DVector::from_vec(state_vec.to_vec());

        // Convert phi to DMatrix
        let phi_array = phi.as_array();
        let mut phi_dmat = na::DMatrix::<f64>::zeros(self.dimension, self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_dmat[(i, j)] = phi_array[[i, j]];
            }
        }

        // Convert sensitivity matrix to DMatrix
        let sens_array = sens.as_array();
        let sens_shape = sens_array.shape();
        let num_params = sens_shape[1];
        let mut sens_dmat = na::DMatrix::<f64>::zeros(self.dimension, num_params);
        for i in 0..self.dimension {
            for j in 0..num_params {
                sens_dmat[(i, j)] = sens_array[[i, j]];
            }
        }

        // Convert params to DVector
        let params_vec = params.as_slice()?;
        let params_dvec = na::DVector::from_vec(params_vec.to_vec());

        // Perform step
        let result = self.inner.step_with_varmat_sensmat(
            t, state_dvec, phi_dmat, sens_dmat, &params_dvec, Some(dt)
        );

        // Convert results to NumPy
        let state_np = result.state.as_slice().to_pyarray(py);

        let new_phi = result.phi.expect("Variational matrix should be present");
        let mut phi_flat = Vec::with_capacity(self.dimension * self.dimension);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                phi_flat.push(new_phi[(i, j)]);
            }
        }
        let phi_np = phi_flat
            .into_pyarray(py)
            .reshape([self.dimension, self.dimension])?;

        let new_sens = result.sens.expect("Sensitivity matrix should be present");
        let mut sens_flat = Vec::with_capacity(self.dimension * num_params);
        for i in 0..self.dimension {
            for j in 0..num_params {
                sens_flat.push(new_sens[(i, j)]);
            }
        }
        let dt_used = result.dt_used;
        let error_est = result.error_estimate.expect("Error estimate should be present for adaptive integrator");
        let dt_next = result.dt_next;
        let sens_np = sens_flat
            .into_pyarray(py)
            .reshape([self.dimension, num_params])?;

        Ok((state_np, phi_np, sens_np, dt_used, error_est, dt_next))
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.dimension
    }

    fn __repr__(&self) -> String {
        format!("RKN1210Integrator(dimension={})", self.dimension)
    }
}
