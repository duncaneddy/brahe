/*
 * Python bindings for the Jacobian computation module.
 *
 * Provides numerical and analytical Jacobian providers for dynamic-sized systems.
 */


// ============================================================================
// Linear algebra
// ============================================================================

/// Skew-symmetric cross-product matrix `[v]x` such that `[v]x @ w == v x w`.
///
/// Args:
///     v (np.ndarray): 3-element vector whose cross product is to be written as a matrix
///
/// Returns:
///     np.ndarray: The 3x3 skew-symmetric matrix `[v]x`, shape `(3, 3)`
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     v = np.array([1.0, 2.0, 3.0])
///     s = bh.skew_symmetric(v)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(v)")]
#[pyo3(name = "skew_symmetric")]
fn py_skew_symmetric<'py>(
    py: Python<'py>,
    v: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let vec = numpy_to_vector3!(v);
    let mat = skew_symmetric(&vec);
    Ok(matrix_to_numpy!(py, mat, 3, 3, f64).to_owned())
}

/// 6x6 block-diagonal matrix `blockdiag(a, b)`.
///
/// The result carries `a` in the leading 3x3 block, `b` in the trailing 3x3
/// block, and zeros in the two off-diagonal blocks.
///
/// Args:
///     a (np.ndarray): Leading 3x3 block, placed at rows and columns 0-2
///     b (np.ndarray): Trailing 3x3 block, placed at rows and columns 3-5
///
/// Returns:
///     np.ndarray: The 6x6 block-diagonal matrix, shape `(6, 6)`
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     a = np.eye(3)
///     b = np.eye(3) * 2.0
///     m = bh.block_diagonal(a, b)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(a, b)")]
#[pyo3(name = "block_diagonal")]
fn py_block_diagonal<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    b: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let a_mat = numpy_to_smatrix3!(a);
    let b_mat = numpy_to_smatrix3!(b);
    let m = block_diagonal(&a_mat, &b_mat);
    Ok(matrix_to_numpy!(py, m, 6, 6, f64).to_owned())
}

/// The symmetric part `(m + m.T) / 2` of a square matrix.
///
/// A congruence `J @ P @ J.T` of a symmetric `P` is symmetric in exact
/// arithmetic but accumulates asymmetry of the order of the rounding error.
/// Averaging the matrix with its transpose removes that drift without
/// changing the result to within the same rounding.
///
/// Args:
///     m (np.ndarray): Square matrix
///
/// Returns:
///     np.ndarray: The symmetric part of `m`
///
/// Raises:
///     PanicException: If `m` is not square
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     m = np.array([[1.0, 0.4], [0.6, 2.0]])
///     s = bh.symmetrize(m)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(m)")]
#[pyo3(name = "symmetrize")]
fn py_symmetrize<'py>(py: Python<'py>, m: PyReadonlyArray2<'py, f64>) -> Bound<'py, PyArray<f64, Ix2>> {
    let array = m.as_array();
    let (rows, cols) = (array.shape()[0], array.shape()[1]);
    let mat = DMatrix::<f64>::from_row_iterator(rows, cols, array.iter().copied());
    let s = symmetrize(&mat);
    let s_ref = &s;
    matrix_to_numpy!(py, s_ref, rows, cols, f64).to_owned()
}

/// Whether a square matrix is symmetric to within a relative tolerance.
///
/// Args:
///     m (np.ndarray): Square matrix
///     rtol (float): Relative tolerance applied to the larger magnitude of each mirrored pair. Default `1e-9`
///
/// Returns:
///     bool: True when every pair `(i, k)`, `(k, i)` agrees within `rtol`, or when `m` is not square is False
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     m = np.array([[1.0, 0.5], [0.5, 2.0]])
///     assert bh.is_symmetric(m)
///
///     n = np.array([[1.0, 0.5], [0.4, 2.0]])
///     assert not bh.is_symmetric(n)
///     ```
#[pyfunction]
#[pyo3(signature = (m, rtol=1e-9))]
#[pyo3(text_signature = "(m, rtol=1e-9)")]
#[pyo3(name = "is_symmetric")]
fn py_is_symmetric<'py>(m: PyReadonlyArray2<'py, f64>, rtol: f64) -> bool {
    let array = m.as_array();
    let (rows, cols) = (array.shape()[0], array.shape()[1]);
    let mat = DMatrix::<f64>::from_row_iterator(rows, cols, array.iter().copied());
    is_symmetric(&mat, rtol)
}

// ============================================================================
// Enums
// ============================================================================

/// Finite difference method for numerical Jacobian approximation.
///
/// Different methods trade off accuracy vs computational cost:
/// - **Forward**: O(h) error, dimension+1 function evaluations
/// - **Central**: O(h²) error, 2*dimension function evaluations (more accurate)
/// - **Backward**: O(h) error, dimension+1 function evaluations
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Use central differences (most accurate)
///     jacobian = bh.NumericalJacobian.central(dynamics_fn)
///
///     # Or explicitly set the method
///     jacobian = bh.NumericalJacobian.new(dynamics_fn).with_method(bh.DifferenceMethod.CENTRAL)
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "DifferenceMethod")]
#[derive(Clone)]
pub struct PyDifferenceMethod {
    pub(crate) value: jacobian::DifferenceMethod,
}

#[pymethods]
impl PyDifferenceMethod {
    /// Forward finite difference: df/dx ≈ (f(x+h) - f(x)) / h
    ///
    /// First-order accurate but cheaper (dimension+1 evaluations).
    #[classattr]
    #[allow(non_snake_case)]
    fn FORWARD() -> Self {
        PyDifferenceMethod {
            value: jacobian::DifferenceMethod::Forward,
        }
    }

    /// Central finite difference: df/dx ≈ (f(x+h) - f(x-h)) / (2h)
    ///
    /// Second-order accurate but more expensive (2*dimension evaluations).
    /// Generally preferred for high-precision applications.
    #[classattr]
    #[allow(non_snake_case)]
    fn CENTRAL() -> Self {
        PyDifferenceMethod {
            value: jacobian::DifferenceMethod::Central,
        }
    }

    /// Backward finite difference: df/dx ≈ (f(x) - f(x-h)) / h
    ///
    /// First-order accurate, same cost as forward differences (dimension+1 evaluations).
    /// Useful when forward perturbations are problematic.
    #[classattr]
    #[allow(non_snake_case)]
    fn BACKWARD() -> Self {
        PyDifferenceMethod {
            value: jacobian::DifferenceMethod::Backward,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.value)
    }

    fn __repr__(&self) -> String {
        format!("DifferenceMethod.{:?}", self.value)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.value == other.value),
            CompareOp::Ne => Ok(self.value != other.value),
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Only == and != are supported for DifferenceMethod",
            )),
        }
    }
}

/// Strategy for computing perturbation sizes in finite differences.
///
/// The choice of perturbation size balances truncation error (wants large h)
/// vs roundoff error (wants small h). Different strategies suit different problems.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Adaptive perturbation (recommended for most cases)
///     jacobian = bh.NumericalJacobian.new(dynamics_fn).with_adaptive(1.0, 1.0)
///
///     # Fixed absolute perturbation for all components
///     jacobian = bh.NumericalJacobian.new(dynamics_fn).with_fixed_offset(1e-6)
///
///     # Percentage-based perturbation
///     jacobian = bh.NumericalJacobian.new(dynamics_fn).with_percentage(1e-6)
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "PerturbationStrategy")]
#[derive(Clone)]
pub struct PyPerturbationStrategy {
    pub(crate) value: jacobian::PerturbationStrategy,
}

#[pymethods]
impl PyPerturbationStrategy {
    /// Create an adaptive perturbation strategy.
    ///
    /// Args:
    ///     scale_factor (float): Multiplier on sqrt(ε), typically 1.0
    ///     min_value (float): Minimum reference value (prevents tiny perturbations near zero)
    ///
    /// Returns:
    ///     PerturbationStrategy: Adaptive perturbation strategy
    #[classmethod]
    #[pyo3(signature = (scale_factor=1.0, min_value=1.0))]
    fn adaptive(
        _cls: &Bound<'_, pyo3::types::PyType>,
        scale_factor: f64,
        min_value: f64,
    ) -> Self {
        PyPerturbationStrategy {
            value: jacobian::PerturbationStrategy::Adaptive {
                scale_factor,
                min_value,
            },
        }
    }

    /// Create a fixed absolute perturbation strategy.
    ///
    /// Args:
    ///     offset (float): Fixed perturbation for all state components
    ///
    /// Returns:
    ///     PerturbationStrategy: Fixed perturbation strategy
    #[classmethod]
    fn fixed(_cls: &Bound<'_, pyo3::types::PyType>, offset: f64) -> Self {
        PyPerturbationStrategy {
            value: jacobian::PerturbationStrategy::Fixed(offset),
        }
    }

    /// Create a percentage-based perturbation strategy.
    ///
    /// Args:
    ///     percentage (float): Percentage of state value to use as perturbation
    ///
    /// Returns:
    ///     PerturbationStrategy: Percentage perturbation strategy
    #[classmethod]
    fn percentage(_cls: &Bound<'_, pyo3::types::PyType>, percentage: f64) -> Self {
        PyPerturbationStrategy {
            value: jacobian::PerturbationStrategy::Percentage(percentage),
        }
    }

    fn __str__(&self) -> String {
        match self.value {
            jacobian::PerturbationStrategy::Adaptive {
                scale_factor,
                min_value,
            } => format!("Adaptive({}, {})", scale_factor, min_value),
            jacobian::PerturbationStrategy::Fixed(offset) => format!("Fixed({})", offset),
            jacobian::PerturbationStrategy::Percentage(pct) => format!("Percentage({})", pct),
        }
    }

    fn __repr__(&self) -> String {
        format!("PerturbationStrategy.{}", self.__str__())
    }
}

// ============================================================================
// Numerical Jacobian Provider
// ============================================================================

/// Numerical Jacobian provider for dynamic-sized systems using finite differences.
///
/// Computes the Jacobian numerically by perturbing the state and evaluating the dynamics.
/// Supports forward, central, and backward finite difference methods with various
/// perturbation strategies.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Simple harmonic oscillator dynamics
///     def dynamics(t, state):
///         return np.array([state[1], -state[0]])
///
///     # Default: central differences with adaptive perturbations
///     jacobian = bh.NumericalJacobian.new(dynamics)
///
///     # Or with custom settings:
///     jacobian = bh.NumericalJacobian.forward(dynamics).with_fixed_offset(1e-6)
///
///     state = np.array([1.0, 0.0])
///     jac_matrix = jacobian.compute(0.0, state)
///     print(jac_matrix)  # [[0, 1], [-1, 0]] approximately
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "NumericalJacobian")]
pub struct PyDNumericalJacobian {
    /// Store the Python callable (dynamics function)
    dynamics_fn: Py<PyAny>,
    method: jacobian::DifferenceMethod,
    perturbation: jacobian::PerturbationStrategy,
}

#[pymethods]
impl PyDNumericalJacobian {
    /// Create a numerical Jacobian provider with default settings.
    ///
    /// Default: central differences with adaptive perturbations.
    ///
    /// Args:
    ///     dynamics_fn (callable): Function with signature (t: float, state: ndarray) -> ndarray
    ///
    /// Returns:
    ///     NumericalJacobian: New numerical Jacobian provider
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     def dynamics(t, state):
    ///         return np.array([state[1], -state[0]])
    ///
    ///     jacobian = bh.NumericalJacobian.new(dynamics)
    ///     ```
    #[new]
    pub fn new(dynamics_fn: Py<PyAny>) -> Self {
        Self {
            dynamics_fn,
            method: jacobian::DifferenceMethod::Central,
            perturbation: jacobian::PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_value: 1.0,
            },
        }
    }

    /// Create with forward finite differences.
    ///
    /// Args:
    ///     dynamics_fn (callable): Function with signature (t: float, state: ndarray) -> ndarray
    ///
    /// Returns:
    ///     NumericalJacobian: Jacobian provider using forward differences
    #[classmethod]
    pub fn forward(_cls: &Bound<'_, pyo3::types::PyType>, dynamics_fn: Py<PyAny>) -> Self {
        Self {
            dynamics_fn,
            method: jacobian::DifferenceMethod::Forward,
            perturbation: jacobian::PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_value: 1.0,
            },
        }
    }

    /// Create with central finite differences.
    ///
    /// Args:
    ///     dynamics_fn (callable): Function with signature (t: float, state: ndarray) -> ndarray
    ///
    /// Returns:
    ///     NumericalJacobian: Jacobian provider using central differences
    #[classmethod]
    pub fn central(_cls: &Bound<'_, pyo3::types::PyType>, dynamics_fn: Py<PyAny>) -> Self {
        Self::new(dynamics_fn)
    }

    /// Create with backward finite differences.
    ///
    /// Args:
    ///     dynamics_fn (callable): Function with signature (t: float, state: ndarray) -> ndarray
    ///
    /// Returns:
    ///     NumericalJacobian: Jacobian provider using backward differences
    #[classmethod]
    pub fn backward(_cls: &Bound<'_, pyo3::types::PyType>, dynamics_fn: Py<PyAny>) -> Self {
        Self {
            dynamics_fn,
            method: jacobian::DifferenceMethod::Backward,
            perturbation: jacobian::PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_value: 1.0,
            },
        }
    }

    /// Set fixed absolute perturbation for all components.
    ///
    /// Args:
    ///     offset (float): Fixed perturbation size
    ///
    /// Returns:
    ///     NumericalJacobian: Self for method chaining
    pub fn with_fixed_offset(mut slf: PyRefMut<'_, Self>, offset: f64) -> PyRefMut<'_, Self> {
        slf.perturbation = jacobian::PerturbationStrategy::Fixed(offset);
        slf
    }

    /// Set percentage-based perturbation.
    ///
    /// Args:
    ///     percentage (float): Percentage of state value (e.g., 1e-6 for 0.0001%)
    ///
    /// Returns:
    ///     NumericalJacobian: Self for method chaining
    pub fn with_percentage(mut slf: PyRefMut<'_, Self>, percentage: f64) -> PyRefMut<'_, Self> {
        slf.perturbation = jacobian::PerturbationStrategy::Percentage(percentage);
        slf
    }

    /// Set adaptive perturbation with custom parameters.
    ///
    /// Args:
    ///     scale_factor (float): Multiplier on sqrt(ε), typically 1.0
    ///     min_value (float): Minimum reference value
    ///
    /// Returns:
    ///     NumericalJacobian: Self for method chaining
    pub fn with_adaptive(
        mut slf: PyRefMut<'_, Self>,
        scale_factor: f64,
        min_value: f64,
    ) -> PyRefMut<'_, Self> {
        slf.perturbation = jacobian::PerturbationStrategy::Adaptive {
            scale_factor,
            min_value,
        };
        slf
    }

    /// Set the difference method.
    ///
    /// Args:
    ///     method (DifferenceMethod): Finite difference method to use
    ///
    /// Returns:
    ///     NumericalJacobian: Self for method chaining
    pub fn with_method(
        mut slf: PyRefMut<'_, Self>,
        method: PyDifferenceMethod,
    ) -> PyRefMut<'_, Self> {
        slf.method = method.value;
        slf
    }

    /// Compute the Jacobian matrix at the given time and state.
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (ndarray): State vector at time t
    ///
    /// Returns:
    ///     ndarray: Jacobian matrix ∂f/∂x (dimension × dimension)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     def dynamics(t, state):
    ///         return np.array([state[1], -state[0]])
    ///
    ///     jacobian = bh.NumericalJacobian.new(dynamics)
    ///     state = np.array([1.0, 0.5])
    ///     jac = jacobian.compute(0.0, state)
    ///     # Expected: [[0, 1], [-1, 0]] approximately
    ///     ```
    ///
    /// Raises:
    ///     Exception: Propagates the original exception raised by the dynamics
    ///         callback, or a BraheError if the finite-difference evaluation fails.
    #[allow(deprecated)]
    pub fn compute<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        state: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray<f64, numpy::Ix2>>> {
        // Convert Python state to DVector
        let state_vec = pyany_to_f64_array1(state, None)?;
        let dimension = state_vec.len();
        let state_dvec = DVector::from_vec(state_vec);

        // Create a Rust closure that calls the Python function. Any exception
        // raised by the callback is stashed in `err_slot` and surfaced to the
        // Rust core as a BraheError, then re-raised verbatim below.
        let err_slot: PyErrSlot = Arc::new(Mutex::new(None));
        let dynamics_closure = {
            let dynamics_fn_clone = self.dynamics_fn.clone_ref(py);
            let err_slot = err_slot.clone();
            move |t: f64,
                  state: &DVector<f64>,
                  _params: Option<&DVector<f64>>|
                  -> Result<DVector<f64>, RustBraheError> {
                Python::attach(|py| {
                    // Convert state to NumPy array
                    let state_py = state.as_slice().to_pyarray(py);

                    // Call Python function
                    let result = dynamics_fn_clone
                        .call1(py, (t, state_py))
                        .map_err(|e| stash_callback_err(&err_slot, e))?;

                    // Convert result back to DVector
                    let result_vec = pyany_to_f64_array1(result.bind(py), Some(dimension))
                        .map_err(|e| stash_callback_err(&err_slot, e))?;
                    Ok(DVector::from_vec(result_vec))
                })
            }
        };

        // Create Rust numerical Jacobian provider
        let mut provider = match self.method {
            jacobian::DifferenceMethod::Forward => {
                jacobian::DNumericalJacobian::forward(Box::new(dynamics_closure))
            }
            jacobian::DifferenceMethod::Central => {
                jacobian::DNumericalJacobian::central(Box::new(dynamics_closure))
            }
            jacobian::DifferenceMethod::Backward => {
                jacobian::DNumericalJacobian::backward(Box::new(dynamics_closure))
            }
        };

        // Apply perturbation strategy
        provider = match self.perturbation {
            jacobian::PerturbationStrategy::Adaptive {
                scale_factor,
                min_value,
            } => provider.with_adaptive(scale_factor, min_value),
            jacobian::PerturbationStrategy::Fixed(offset) => provider.with_fixed_offset(offset),
            jacobian::PerturbationStrategy::Percentage(pct) => provider.with_percentage(pct),
        };

        // Compute Jacobian
        let jac_matrix = provider
            .compute(t, &state_dvec, None)
            .map_err(|e| raise_callback_err(&err_slot, e))?;

        // Convert DMatrix to NumPy 2D array (row-major order)
        let rows = jac_matrix.nrows();
        let cols = jac_matrix.ncols();
        let mut flat_vec = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                flat_vec.push(jac_matrix[(i, j)]);
            }
        }

        flat_vec.into_pyarray(py).reshape([rows, cols])
    }
}

// ============================================================================
// Analytical Jacobian Provider
// ============================================================================

/// Analytical Jacobian provider for dynamic-sized systems.
///
/// Uses a user-provided function that directly computes the analytical Jacobian.
/// This is the most accurate and efficient method when the analytical Jacobian is known.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Simple harmonic oscillator: dx/dt = v, dv/dt = -x
///     # Jacobian is [[0, 1], [-1, 0]]
///     def jacobian_fn(t, state):
///         return np.array([[0.0, 1.0], [-1.0, 0.0]])
///
///     jacobian = bh.AnalyticJacobian.new(jacobian_fn)
///     state = np.array([1.0, 0.0])
///     jac = jacobian.compute(0.0, state)
///     print(jac)  # [[0, 1], [-1, 0]]
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "AnalyticJacobian")]
pub struct PyDAnalyticJacobian {
    /// Store the Python callable (Jacobian function)
    jacobian_fn: Py<PyAny>,
}

#[pymethods]
impl PyDAnalyticJacobian {
    /// Create a new analytical Jacobian provider.
    ///
    /// Args:
    ///     jacobian_fn (callable): Function with signature (t: float, state: ndarray) -> ndarray
    ///         Must return a 2D array (dimension × dimension)
    ///
    /// Returns:
    ///     AnalyticJacobian: New analytical Jacobian provider
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     def jacobian_fn(t, state):
    ///         # For harmonic oscillator
    ///         return np.array([[0.0, 1.0], [-1.0, 0.0]])
    ///
    ///     jacobian = bh.AnalyticJacobian.new(jacobian_fn)
    ///     ```
    #[new]
    pub fn new(jacobian_fn: Py<PyAny>) -> Self {
        Self { jacobian_fn }
    }

    /// Compute the Jacobian matrix at the given time and state.
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (ndarray): State vector at time t
    ///
    /// Returns:
    ///     ndarray: Jacobian matrix ∂f/∂x (dimension × dimension)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     def jacobian_fn(t, state):
    ///         return np.array([[0.0, 1.0], [-1.0, 0.0]])
    ///
    ///     jacobian = bh.AnalyticJacobian.new(jacobian_fn)
    ///     state = np.array([1.0, 0.5])
    ///     jac = jacobian.compute(0.0, state)
    ///     ```
    pub fn compute<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        state: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray<f64, numpy::Ix2>>> {
        // Convert Python state to DVector
        let state_vec = pyany_to_f64_array1(state, None)?;
        let dimension = state_vec.len();
        let state_py = state_vec.into_pyarray(py);

        // Call Python function
        let result = self.jacobian_fn.call1(py, (t, state_py))?;

        // Convert result to 2D array
        let jac_matrix_vec = pyany_to_f64_array2(result.bind(py), Some((dimension, dimension)))?;

        // Convert to NumPy 2D array (row-major order)
        let flat_vec: Vec<f64> = jac_matrix_vec.into_iter().flatten().collect();

        flat_vec.into_pyarray(py).reshape([dimension, dimension])
    }
}

// ============================================================================
// Sensitivity Providers
// ============================================================================

use brahe::math::sensitivity;

/// Numerical sensitivity provider for dynamic-sized systems using finite differences.
///
/// Computes the sensitivity matrix ∂f/∂p numerically by perturbing the parameters
/// and evaluating the dynamics function.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Dynamics with consider parameters
///     def dynamics(t, state, params):
///         cd_area_m = params[0]
///         # ... compute derivatives using cd_area_m
///         return np.array([...])
///
///     sensitivity = bh.NumericalSensitivity(dynamics)
///     state = np.array([7000e3, 0, 0, 0, 7.5e3, 0])
///     params = np.array([0.044])  # cd*A/m
///     sens_matrix = sensitivity.compute(0.0, state, params)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "NumericalSensitivity")]
pub struct PyDNumericalSensitivity {
    dynamics_fn: Py<PyAny>,
    method: jacobian::DifferenceMethod,
    perturbation: jacobian::PerturbationStrategy,
}

#[pymethods]
impl PyDNumericalSensitivity {
    /// Create a numerical sensitivity provider with default settings.
    ///
    /// Uses central differences with adaptive perturbation strategy.
    ///
    /// Args:
    ///     dynamics_fn (callable): Function with signature (t: float, state: ndarray, params: ndarray) -> ndarray
    ///
    /// Returns:
    ///     NumericalSensitivity: New numerical sensitivity provider
    #[new]
    pub fn new(dynamics_fn: Py<PyAny>) -> Self {
        Self {
            dynamics_fn,
            method: jacobian::DifferenceMethod::Central,
            perturbation: jacobian::PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_value: 1.0,
            },
        }
    }

    /// Create with forward finite differences.
    ///
    /// Args:
    ///     dynamics_fn (callable): Function with signature (t: float, state: ndarray, params: ndarray) -> ndarray
    ///
    /// Returns:
    ///     NumericalSensitivity: Sensitivity provider using forward differences
    #[classmethod]
    pub fn forward(_cls: &Bound<'_, pyo3::types::PyType>, dynamics_fn: Py<PyAny>) -> Self {
        Self {
            dynamics_fn,
            method: jacobian::DifferenceMethod::Forward,
            perturbation: jacobian::PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_value: 1.0,
            },
        }
    }

    /// Create with central finite differences.
    ///
    /// Args:
    ///     dynamics_fn (callable): Function with signature (t: float, state: ndarray, params: ndarray) -> ndarray
    ///
    /// Returns:
    ///     NumericalSensitivity: Sensitivity provider using central differences
    #[classmethod]
    pub fn central(_cls: &Bound<'_, pyo3::types::PyType>, dynamics_fn: Py<PyAny>) -> Self {
        Self::new(dynamics_fn)
    }

    /// Create with backward finite differences.
    ///
    /// Args:
    ///     dynamics_fn (callable): Function with signature (t: float, state: ndarray, params: ndarray) -> ndarray
    ///
    /// Returns:
    ///     NumericalSensitivity: Sensitivity provider using backward differences
    #[classmethod]
    pub fn backward(_cls: &Bound<'_, pyo3::types::PyType>, dynamics_fn: Py<PyAny>) -> Self {
        Self {
            dynamics_fn,
            method: jacobian::DifferenceMethod::Backward,
            perturbation: jacobian::PerturbationStrategy::Adaptive {
                scale_factor: 1.0,
                min_value: 1.0,
            },
        }
    }

    /// Set fixed absolute perturbation for all parameters.
    ///
    /// Args:
    ///     offset (float): Fixed perturbation size
    ///
    /// Returns:
    ///     NumericalSensitivity: Self for method chaining
    pub fn with_fixed_offset(mut slf: PyRefMut<'_, Self>, offset: f64) -> PyRefMut<'_, Self> {
        slf.perturbation = jacobian::PerturbationStrategy::Fixed(offset);
        slf
    }

    /// Set percentage-based perturbation.
    ///
    /// Args:
    ///     percentage (float): Percentage of parameter value (e.g., 1e-6 for 0.0001%)
    ///
    /// Returns:
    ///     NumericalSensitivity: Self for method chaining
    pub fn with_percentage(mut slf: PyRefMut<'_, Self>, percentage: f64) -> PyRefMut<'_, Self> {
        slf.perturbation = jacobian::PerturbationStrategy::Percentage(percentage);
        slf
    }

    /// Set adaptive perturbation with custom parameters.
    ///
    /// Args:
    ///     scale_factor (float): Multiplier on sqrt(ε), typically 1.0
    ///     min_value (float): Minimum reference value
    ///
    /// Returns:
    ///     NumericalSensitivity: Self for method chaining
    pub fn with_adaptive(
        mut slf: PyRefMut<'_, Self>,
        scale_factor: f64,
        min_value: f64,
    ) -> PyRefMut<'_, Self> {
        slf.perturbation = jacobian::PerturbationStrategy::Adaptive {
            scale_factor,
            min_value,
        };
        slf
    }

    /// Set the difference method.
    ///
    /// Args:
    ///     method (DifferenceMethod): Finite difference method to use
    ///
    /// Returns:
    ///     NumericalSensitivity: Self for method chaining
    pub fn with_method(
        mut slf: PyRefMut<'_, Self>,
        method: PyDifferenceMethod,
    ) -> PyRefMut<'_, Self> {
        slf.method = method.value;
        slf
    }

    /// Compute the sensitivity matrix at the given time, state, and parameters.
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (ndarray): State vector at time t
    ///     params (ndarray): Consider parameters
    ///
    /// Returns:
    ///     ndarray: Sensitivity matrix ∂f/∂p (state_dim × param_dim)
    ///
    /// Raises:
    ///     Exception: Propagates the original exception raised by the dynamics
    ///         callback, or a BraheError if the finite-difference evaluation fails.
    #[allow(deprecated)]
    pub fn compute<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        state: &Bound<'py, PyAny>,
        params: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray<f64, numpy::Ix2>>> {
        // Convert Python arrays to vectors
        let state_vec = pyany_to_f64_array1(state, None)?;
        let params_vec = pyany_to_f64_array1(params, None)?;
        let state_dim = state_vec.len();
        let param_dim = params_vec.len();
        let state_dvec = DVector::from_vec(state_vec);
        let params_dvec = DVector::from_vec(params_vec);

        // Create a Rust closure that calls the Python function. Any exception
        // raised by the callback is stashed in `err_slot` and surfaced to the
        // Rust core as a BraheError, then re-raised verbatim below.
        let err_slot: PyErrSlot = Arc::new(Mutex::new(None));
        let dynamics_closure = {
            let dynamics_fn_clone = self.dynamics_fn.clone_ref(py);
            let err_slot = err_slot.clone();
            move |t: f64,
                  state: &DVector<f64>,
                  params: &DVector<f64>|
                  -> Result<DVector<f64>, RustBraheError> {
                Python::attach(|py| {
                    let state_py = state.as_slice().to_pyarray(py);
                    let params_py = params.as_slice().to_pyarray(py);

                    let result = dynamics_fn_clone
                        .call1(py, (t, state_py, params_py))
                        .map_err(|e| stash_callback_err(&err_slot, e))?;

                    let result_vec = pyany_to_f64_array1(result.bind(py), Some(state.len()))
                        .map_err(|e| stash_callback_err(&err_slot, e))?;
                    Ok(DVector::from_vec(result_vec))
                })
            }
        };

        // Create Rust numerical sensitivity provider
        let mut provider = match self.method {
            jacobian::DifferenceMethod::Forward => {
                sensitivity::DNumericalSensitivity::forward(Box::new(dynamics_closure))
            }
            jacobian::DifferenceMethod::Central => {
                sensitivity::DNumericalSensitivity::central(Box::new(dynamics_closure))
            }
            jacobian::DifferenceMethod::Backward => {
                sensitivity::DNumericalSensitivity::backward(Box::new(dynamics_closure))
            }
        };

        // Apply perturbation strategy
        provider = match self.perturbation {
            jacobian::PerturbationStrategy::Adaptive {
                scale_factor,
                min_value,
            } => provider.with_strategy(jacobian::PerturbationStrategy::Adaptive {
                scale_factor,
                min_value,
            }),
            jacobian::PerturbationStrategy::Fixed(offset) => {
                provider.with_strategy(jacobian::PerturbationStrategy::Fixed(offset))
            }
            jacobian::PerturbationStrategy::Percentage(pct) => {
                provider.with_strategy(jacobian::PerturbationStrategy::Percentage(pct))
            }
        };

        // Compute sensitivity
        let sens_matrix = provider
            .compute(t, &state_dvec, &params_dvec)
            .map_err(|e| raise_callback_err(&err_slot, e))?;

        // Convert DMatrix to NumPy 2D array
        let rows = sens_matrix.nrows();
        let cols = sens_matrix.ncols();
        let mut flat_vec = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                flat_vec.push(sens_matrix[(i, j)]);
            }
        }

        flat_vec.into_pyarray(py).reshape([state_dim, param_dim])
    }
}

/// Analytical sensitivity provider for dynamic-sized systems.
///
/// Uses a user-provided function that directly computes the analytical sensitivity matrix.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     def sensitivity_fn(t, state, params):
///         # Return ∂f/∂p matrix
///         sens = np.zeros((6, 1))
///         # ... compute analytical sensitivity
///         return sens
///
///     sensitivity = bh.AnalyticSensitivity(sensitivity_fn)
///     state = np.array([7000e3, 0, 0, 0, 7.5e3, 0])
///     params = np.array([0.044])
///     sens_matrix = sensitivity.compute(0.0, state, params)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "AnalyticSensitivity")]
pub struct PyDAnalyticSensitivity {
    sensitivity_fn: Py<PyAny>,
}

#[pymethods]
impl PyDAnalyticSensitivity {
    /// Create a new analytical sensitivity provider.
    ///
    /// Args:
    ///     sensitivity_fn (callable): Function with signature (t: float, state: ndarray, params: ndarray) -> ndarray
    ///         Must return a 2D array (state_dim × param_dim)
    ///
    /// Returns:
    ///     AnalyticSensitivity: New analytical sensitivity provider
    #[new]
    pub fn new(sensitivity_fn: Py<PyAny>) -> Self {
        Self { sensitivity_fn }
    }

    /// Compute the sensitivity matrix at the given time, state, and parameters.
    ///
    /// Args:
    ///     t (float): Current time
    ///     state (ndarray): State vector at time t
    ///     params (ndarray): Consider parameters
    ///
    /// Returns:
    ///     ndarray: Sensitivity matrix ∂f/∂p (state_dim × param_dim)
    pub fn compute<'py>(
        &self,
        py: Python<'py>,
        t: f64,
        state: &Bound<'py, PyAny>,
        params: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray<f64, numpy::Ix2>>> {
        // Convert Python arrays
        let state_vec = pyany_to_f64_array1(state, None)?;
        let params_vec = pyany_to_f64_array1(params, None)?;
        let state_dim = state_vec.len();
        let param_dim = params_vec.len();
        let state_py = state_vec.into_pyarray(py);
        let params_py = params_vec.into_pyarray(py);

        // Call Python function
        let result = self.sensitivity_fn.call1(py, (t, state_py, params_py))?;

        // Convert result to 2D array
        let sens_matrix_vec = pyany_to_f64_array2(result.bind(py), Some((state_dim, param_dim)))?;

        // Convert to NumPy 2D array
        let flat_vec: Vec<f64> = sens_matrix_vec.into_iter().flatten().collect();

        flat_vec.into_pyarray(py).reshape([state_dim, param_dim])
    }
}
