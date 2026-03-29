// Monte Carlo simulation framework Python bindings.
//
// Exposes the Rust Monte Carlo types to Python via PyO3:
// - Distribution types: PyGaussian, PyUniformDist, PyTruncatedGaussian, PyMultivariateNormal
// - Variable identification: PyMonteCarloVariableId
// - Configuration: PyMonteCarloStoppingCondition
// - Simulation engine: PyMonteCarloSimulation
// - Results: PyMonteCarloResults, PyMonteCarloRun

// ---------------------------------------------------------------------------
// Distribution wrappers
// ---------------------------------------------------------------------------

/// Gaussian (normal) probability distribution for Monte Carlo sampling.
///
/// Produces scalar samples from N(mean, std^2).
///
/// Args:
///     mean (float): Mean of the distribution
///     std (float): Standard deviation of the distribution (must be non-negative)
///
/// Returns:
///     Gaussian: A new Gaussian distribution instance
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create a Gaussian distribution with mean=0 and std=1
///     dist = bh.Gaussian(mean=0.0, std=1.0)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "Gaussian")]
pub struct PyGaussian {
    pub(crate) inner: monte_carlo::Gaussian,
}

#[pymethods]
impl PyGaussian {
    /// Create a new Gaussian distribution.
    ///
    /// Args:
    ///     mean (float): Mean of the distribution
    ///     std (float): Standard deviation of the distribution
    ///
    /// Returns:
    ///     Gaussian: A new Gaussian distribution instance
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     dist = bh.Gaussian(mean=5.0, std=2.0)
    ///     ```
    #[new]
    #[pyo3(signature = (mean, std))]
    pub fn new(mean: f64, std: f64) -> Self {
        PyGaussian {
            inner: monte_carlo::Gaussian { mean, std },
        }
    }

    /// Mean of the distribution.
    ///
    /// Returns:
    ///     float: The mean value
    #[getter]
    pub fn mean(&self) -> f64 {
        self.inner.mean
    }

    /// Standard deviation of the distribution.
    ///
    /// Returns:
    ///     float: The standard deviation
    #[getter]
    pub fn std(&self) -> f64 {
        self.inner.std
    }

    pub fn __repr__(&self) -> String {
        format!("Gaussian(mean={}, std={})", self.inner.mean, self.inner.std)
    }
}

/// Continuous uniform distribution over [low, high).
///
/// Produces scalar samples uniformly distributed in the half-open interval.
///
/// Args:
///     low (float): Lower bound (inclusive)
///     high (float): Upper bound (exclusive)
///
/// Returns:
///     UniformDist: A new uniform distribution instance
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create a uniform distribution between -1 and 1
///     dist = bh.UniformDist(low=-1.0, high=1.0)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "UniformDist")]
pub struct PyUniformDist {
    pub(crate) inner: monte_carlo::UniformDist,
}

#[pymethods]
impl PyUniformDist {
    /// Create a new uniform distribution.
    ///
    /// Args:
    ///     low (float): Lower bound (inclusive)
    ///     high (float): Upper bound (exclusive)
    ///
    /// Returns:
    ///     UniformDist: A new uniform distribution instance
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     dist = bh.UniformDist(low=0.0, high=10.0)
    ///     ```
    #[new]
    #[pyo3(signature = (low, high))]
    pub fn new(low: f64, high: f64) -> Self {
        PyUniformDist {
            inner: monte_carlo::UniformDist { low, high },
        }
    }

    /// Lower bound (inclusive).
    ///
    /// Returns:
    ///     float: The lower bound
    #[getter]
    pub fn low(&self) -> f64 {
        self.inner.low
    }

    /// Upper bound (exclusive).
    ///
    /// Returns:
    ///     float: The upper bound
    #[getter]
    pub fn high(&self) -> f64 {
        self.inner.high
    }

    pub fn __repr__(&self) -> String {
        format!(
            "UniformDist(low={}, high={})",
            self.inner.low, self.inner.high
        )
    }
}

/// Gaussian distribution truncated to [low, high] via rejection sampling.
///
/// Draws from N(mean, std^2) and discards values outside the bounds.
/// Efficient when the truncation interval covers a reasonable fraction
/// of the distribution's probability mass.
///
/// Args:
///     mean (float): Mean of the underlying Gaussian
///     std (float): Standard deviation of the underlying Gaussian
///     low (float): Lower truncation bound (inclusive)
///     high (float): Upper truncation bound (inclusive)
///
/// Returns:
///     TruncatedGaussian: A new truncated Gaussian distribution instance
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Cd between 1.8 and 2.6 centered at 2.2
///     dist = bh.TruncatedGaussian(mean=2.2, std=0.1, low=1.8, high=2.6)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "TruncatedGaussian")]
pub struct PyTruncatedGaussian {
    pub(crate) inner: monte_carlo::TruncatedGaussian,
}

#[pymethods]
impl PyTruncatedGaussian {
    /// Create a new truncated Gaussian distribution.
    ///
    /// Args:
    ///     mean (float): Mean of the underlying Gaussian
    ///     std (float): Standard deviation of the underlying Gaussian
    ///     low (float): Lower truncation bound (inclusive)
    ///     high (float): Upper truncation bound (inclusive)
    ///
    /// Returns:
    ///     TruncatedGaussian: A new truncated Gaussian distribution instance
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     dist = bh.TruncatedGaussian(mean=2.2, std=0.1, low=1.8, high=2.6)
    ///     ```
    #[new]
    #[pyo3(signature = (mean, std, low, high))]
    pub fn new(mean: f64, std: f64, low: f64, high: f64) -> Self {
        PyTruncatedGaussian {
            inner: monte_carlo::TruncatedGaussian {
                mean,
                std,
                low,
                high,
            },
        }
    }

    /// Mean of the underlying Gaussian.
    ///
    /// Returns:
    ///     float: The mean value
    #[getter]
    pub fn mean(&self) -> f64 {
        self.inner.mean
    }

    /// Standard deviation of the underlying Gaussian.
    ///
    /// Returns:
    ///     float: The standard deviation
    #[getter]
    pub fn std(&self) -> f64 {
        self.inner.std
    }

    /// Lower truncation bound.
    ///
    /// Returns:
    ///     float: The lower bound
    #[getter]
    pub fn low(&self) -> f64 {
        self.inner.low
    }

    /// Upper truncation bound.
    ///
    /// Returns:
    ///     float: The upper bound
    #[getter]
    pub fn high(&self) -> f64 {
        self.inner.high
    }

    pub fn __repr__(&self) -> String {
        format!(
            "TruncatedGaussian(mean={}, std={}, low={}, high={})",
            self.inner.mean, self.inner.std, self.inner.low, self.inner.high
        )
    }
}

/// Multivariate normal distribution with full covariance matrix.
///
/// Produces vector samples from N(mean, cov). The covariance matrix must
/// be symmetric positive-definite. Internally, the Cholesky decomposition
/// is computed at construction time.
///
/// Args:
///     mean (numpy.ndarray): Mean vector (1-D, n elements)
///     cov (numpy.ndarray): Covariance matrix (n x n, symmetric positive-definite)
///
/// Returns:
///     MultivariateNormal: A new multivariate normal distribution instance
///
/// Raises:
///     BraheError: If the covariance matrix is not positive-definite or dimensions mismatch
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     mean = np.array([0.0, 0.0, 0.0])
///     cov = np.diag([100.0, 100.0, 100.0])  # 10m 1-sigma per axis
///     dist = bh.MultivariateNormal(mean=mean, cov=cov)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "MultivariateNormal")]
pub struct PyMultivariateNormal {
    pub(crate) inner: monte_carlo::MultivariateNormal,
}

#[pymethods]
impl PyMultivariateNormal {
    /// Create a new multivariate normal distribution.
    ///
    /// Computes the Cholesky decomposition of the covariance matrix at
    /// construction time.
    ///
    /// Args:
    ///     mean (numpy.ndarray): Mean vector (1-D array of length n)
    ///     cov (numpy.ndarray): Covariance matrix (n x n, symmetric positive-definite)
    ///
    /// Returns:
    ///     MultivariateNormal: A new multivariate normal distribution instance
    ///
    /// Raises:
    ///     BraheError: If the covariance matrix is not positive-definite or dimensions mismatch
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     mean = np.array([0.0, 0.0])
    ///     cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    ///     dist = bh.MultivariateNormal(mean=mean, cov=cov)
    ///     ```
    #[new]
    #[pyo3(signature = (mean, cov))]
    pub fn new(mean: &Bound<'_, PyAny>, cov: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mean_vec = pyany_to_f64_array1(mean, None)?;
        let n = mean_vec.len();

        let cov_data = pyany_to_f64_array2(cov, Some((n, n)))?;

        let mean_dv = DVector::from_vec(mean_vec);

        // Build nalgebra DMatrix from row-major 2D vec
        let mut cov_flat = Vec::with_capacity(n * n);
        for row in &cov_data {
            cov_flat.extend_from_slice(row);
        }
        let cov_dm = DMatrix::from_row_slice(n, n, &cov_flat);

        let inner = monte_carlo::MultivariateNormal::new(mean_dv, cov_dm)
            .map_err(|e| BraheError::new_err(e.to_string()))?;

        Ok(PyMultivariateNormal { inner })
    }

    /// Dimension of the distribution (length of mean vector).
    ///
    /// Returns:
    ///     int: Number of dimensions
    #[getter]
    pub fn ndim(&self) -> usize {
        self.inner.mean.nrows()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "MultivariateNormal(ndim={})",
            self.inner.mean.nrows()
        )
    }
}

// ---------------------------------------------------------------------------
// MonteCarloVariableId
// ---------------------------------------------------------------------------

/// Typed identifier for a Monte Carlo simulation variable.
///
/// Well-known spacecraft parameters map to specific indices in the propagator's
/// ``params`` vector via :meth:`param_index`. Use :meth:`custom` for user-defined
/// variables.
///
/// Attributes:
///     INITIAL_STATE: Initial state vector variable
///     MASS: Spacecraft mass (params[0])
///     DRAG_AREA: Drag reference area (params[1])
///     DRAG_COEFFICIENT: Drag coefficient Cd (params[2])
///     SRP_AREA: Solar radiation pressure area (params[3])
///     REFLECTIVITY_COEFFICIENT: Reflectivity coefficient Cr (params[4])
///     EOP_TABLE: Earth orientation parameter table
///     SPACE_WEATHER_TABLE: Space weather data table
///
/// Example:
///     ```python
///     import brahe as bh
///
///     var_id = bh.MonteCarloVariableId.MASS
///     print(var_id.param_index())  # 0
///
///     custom_id = bh.MonteCarloVariableId.custom("my_param")
///     print(custom_id.param_index())  # None
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "MonteCarloVariableId")]
#[derive(Clone)]
pub struct PyMonteCarloVariableId {
    pub(crate) inner: monte_carlo::MonteCarloVariableId,
}

#[pymethods]
impl PyMonteCarloVariableId {
    /// Initial state vector variable.
    #[classattr]
    #[pyo3(name = "INITIAL_STATE")]
    fn initial_state() -> Self {
        Self {
            inner: monte_carlo::MonteCarloVariableId::InitialState,
        }
    }

    /// Spacecraft mass variable (params[0]).
    #[classattr]
    #[pyo3(name = "MASS")]
    fn mass() -> Self {
        Self {
            inner: monte_carlo::MonteCarloVariableId::Mass,
        }
    }

    /// Drag reference area variable (params[1]).
    #[classattr]
    #[pyo3(name = "DRAG_AREA")]
    fn drag_area() -> Self {
        Self {
            inner: monte_carlo::MonteCarloVariableId::DragArea,
        }
    }

    /// Drag coefficient variable (params[2]).
    #[classattr]
    #[pyo3(name = "DRAG_COEFFICIENT")]
    fn drag_coefficient() -> Self {
        Self {
            inner: monte_carlo::MonteCarloVariableId::DragCoefficient,
        }
    }

    /// Solar radiation pressure area variable (params[3]).
    #[classattr]
    #[pyo3(name = "SRP_AREA")]
    fn srp_area() -> Self {
        Self {
            inner: monte_carlo::MonteCarloVariableId::SrpArea,
        }
    }

    /// Reflectivity coefficient variable (params[4]).
    #[classattr]
    #[pyo3(name = "REFLECTIVITY_COEFFICIENT")]
    fn reflectivity_coefficient() -> Self {
        Self {
            inner: monte_carlo::MonteCarloVariableId::ReflectivityCoefficient,
        }
    }

    /// Earth orientation parameter table variable.
    #[classattr]
    #[pyo3(name = "EOP_TABLE")]
    fn eop_table() -> Self {
        Self {
            inner: monte_carlo::MonteCarloVariableId::EopTable,
        }
    }

    /// Space weather data table variable.
    #[classattr]
    #[pyo3(name = "SPACE_WEATHER_TABLE")]
    fn space_weather_table() -> Self {
        Self {
            inner: monte_carlo::MonteCarloVariableId::SpaceWeatherTable,
        }
    }

    /// Create a custom variable identifier.
    ///
    /// Args:
    ///     name (str): Name for the custom variable
    ///
    /// Returns:
    ///     MonteCarloVariableId: A custom variable identifier
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     var_id = bh.MonteCarloVariableId.custom("thrust_magnitude")
    ///     ```
    #[classmethod]
    pub fn custom(_cls: &Bound<'_, PyType>, name: String) -> Self {
        Self {
            inner: monte_carlo::MonteCarloVariableId::Custom(name),
        }
    }

    /// Get the params vector index for well-known spacecraft parameters.
    ///
    /// Returns:
    ///     int | None: The index into the propagator params vector, or None
    ///         for variables that are not scalar spacecraft parameters.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     print(bh.MonteCarloVariableId.MASS.param_index())  # 0
    ///     print(bh.MonteCarloVariableId.custom("x").param_index())  # None
    ///     ```
    pub fn param_index(&self) -> Option<usize> {
        self.inner.param_index()
    }

    pub fn __repr__(&self) -> String {
        format!("MonteCarloVariableId({})", self.inner)
    }

    pub fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.inner == other.inner),
            CompareOp::Ne => Ok(self.inner != other.inner),
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Only == and != comparisons are supported",
            )),
        }
    }

    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }
}

// ---------------------------------------------------------------------------
// MonteCarloStoppingCondition
// ---------------------------------------------------------------------------

/// Stopping condition for a Monte Carlo simulation.
///
/// Two strategies are supported:
/// - ``fixed_runs(n)``: Run exactly N simulations
/// - ``convergence(...)``: Run until the standard error of monitored outputs
///   drops below a threshold
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Fixed number of runs
///     stop = bh.MonteCarloStoppingCondition.fixed_runs(1000)
///
///     # Convergence-based stopping
///     stop = bh.MonteCarloStoppingCondition.convergence(
///         targets=["final_altitude"],
///         threshold=0.01,
///         min_runs=100,
///         max_runs=10000,
///         check_interval=50,
///     )
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "MonteCarloStoppingCondition")]
#[derive(Clone)]
pub struct PyMonteCarloStoppingCondition {
    pub(crate) inner: monte_carlo::MonteCarloStoppingCondition,
}

#[pymethods]
impl PyMonteCarloStoppingCondition {
    /// Create a fixed-runs stopping condition.
    ///
    /// Args:
    ///     n (int): Number of simulation runs to execute
    ///
    /// Returns:
    ///     MonteCarloStoppingCondition: A fixed-runs stopping condition
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     stop = bh.MonteCarloStoppingCondition.fixed_runs(1000)
    ///     ```
    #[classmethod]
    pub fn fixed_runs(_cls: &Bound<'_, PyType>, n: usize) -> Self {
        Self {
            inner: monte_carlo::MonteCarloStoppingCondition::FixedRuns(n),
        }
    }

    /// Create a convergence-based stopping condition.
    ///
    /// The simulation runs until the standard error of all monitored output
    /// variables drops below the threshold, or until ``max_runs`` is reached.
    ///
    /// Args:
    ///     targets (list[str]): Output variable names to monitor for convergence
    ///     threshold (float): Standard error threshold (convergence criterion)
    ///     min_runs (int): Minimum number of runs before checking convergence
    ///     max_runs (int): Maximum number of runs (safety limit)
    ///     check_interval (int): Check convergence every N runs
    ///
    /// Returns:
    ///     MonteCarloStoppingCondition: A convergence stopping condition
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     stop = bh.MonteCarloStoppingCondition.convergence(
    ///         targets=["position_error"],
    ///         threshold=0.01,
    ///         min_runs=100,
    ///         max_runs=10000,
    ///         check_interval=50,
    ///     )
    ///     ```
    #[classmethod]
    #[pyo3(signature = (targets, threshold, min_runs, max_runs, check_interval))]
    pub fn convergence(
        _cls: &Bound<'_, PyType>,
        targets: Vec<String>,
        threshold: f64,
        min_runs: usize,
        max_runs: usize,
        check_interval: usize,
    ) -> Self {
        Self {
            inner: monte_carlo::MonteCarloStoppingCondition::Convergence {
                targets,
                threshold,
                min_runs,
                max_runs,
                check_interval,
            },
        }
    }

    pub fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }
}

// ---------------------------------------------------------------------------
// MonteCarloSimulation
// ---------------------------------------------------------------------------

/// Monte Carlo simulation engine.
///
/// Orchestrates variable sampling and execution of a user-provided simulation
/// function, collecting results with statistical analysis.
///
/// Args:
///     stopping_condition (MonteCarloStoppingCondition): When to stop running simulations
///     seed (int): Random seed for reproducibility. Defaults to 42.
///     num_workers (int): Number of parallel workers (0 = auto-detect). Defaults to 0.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     stop = bh.MonteCarloStoppingCondition.fixed_runs(100)
///     sim = bh.MonteCarloSimulation(stop, seed=42)
///
///     sim.add_variable(
///         bh.MonteCarloVariableId.custom("x"),
///         bh.Gaussian(mean=0.0, std=1.0),
///     )
///
///     def run_fn(run_index, variables):
///         x = variables["x"]
///         return {"x_squared": x * x}
///
///     results = sim.run(run_fn)
///     print(f"Mean x^2: {results.mean('x_squared')}")
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "MonteCarloSimulation")]
pub struct PyMonteCarloSimulation {
    config: monte_carlo::MonteCarloConfig,
    variables: Vec<(PyMonteCarloVariableId, PyMCDistributionWrapper)>,
    presampled: Vec<(PyMonteCarloVariableId, Vec<PyMCSampledValueHolder>)>,
    callbacks: Vec<(PyMonteCarloVariableId, Py<PyAny>)>,
}

/// Internal wrapper to hold different Python distribution types.
enum PyMCDistributionWrapper {
    Gaussian(monte_carlo::Gaussian),
    Uniform(monte_carlo::UniformDist),
    TruncatedGaussian(monte_carlo::TruncatedGaussian),
    MultivariateNormal(monte_carlo::MultivariateNormal),
}

/// Internal holder for pre-sampled values from Python.
enum PyMCSampledValueHolder {
    Scalar(f64),
    Vector(DVector<f64>),
}

#[pymethods]
impl PyMonteCarloSimulation {
    /// Create a new Monte Carlo simulation.
    ///
    /// Args:
    ///     stopping_condition (MonteCarloStoppingCondition): When to stop the simulation
    ///     seed (int): Random seed for reproducibility. Defaults to 42.
    ///     num_workers (int): Number of parallel workers (0 = auto-detect). Defaults to 0.
    ///
    /// Returns:
    ///     MonteCarloSimulation: A new simulation engine instance
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     stop = bh.MonteCarloStoppingCondition.fixed_runs(100)
    ///     sim = bh.MonteCarloSimulation(stop, seed=42)
    ///     ```
    #[new]
    #[pyo3(signature = (stopping_condition, seed=42, num_workers=0))]
    pub fn new(
        stopping_condition: PyMonteCarloStoppingCondition,
        seed: u64,
        num_workers: usize,
    ) -> Self {
        let config = monte_carlo::MonteCarloConfig {
            stopping_condition: stopping_condition.inner,
            seed,
            num_workers,
        };
        PyMonteCarloSimulation {
            config,
            variables: Vec::new(),
            presampled: Vec::new(),
            callbacks: Vec::new(),
        }
    }

    /// Add a variable with a distribution to sample from.
    ///
    /// The distribution determines how values are generated for this variable
    /// on each simulation run. Supported distribution types:
    /// :class:`Gaussian`, :class:`UniformDist`, :class:`TruncatedGaussian`,
    /// :class:`MultivariateNormal`.
    ///
    /// Args:
    ///     var_id (MonteCarloVariableId): Typed identifier for this variable
    ///     distribution: Distribution to sample from (Gaussian, UniformDist,
    ///         TruncatedGaussian, or MultivariateNormal)
    ///
    /// Raises:
    ///     TypeError: If the distribution is not a recognized type
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sim = bh.MonteCarloSimulation(
    ///         bh.MonteCarloStoppingCondition.fixed_runs(100), seed=42
    ///     )
    ///     sim.add_variable(
    ///         bh.MonteCarloVariableId.MASS,
    ///         bh.Gaussian(mean=500.0, std=5.0),
    ///     )
    ///     ```
    pub fn add_variable(
        &mut self,
        var_id: PyMonteCarloVariableId,
        distribution: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let wrapper = if let Ok(g) = distribution.extract::<PyRef<'_, PyGaussian>>() {
            PyMCDistributionWrapper::Gaussian(g.inner.clone())
        } else if let Ok(u) = distribution.extract::<PyRef<'_, PyUniformDist>>() {
            PyMCDistributionWrapper::Uniform(u.inner.clone())
        } else if let Ok(t) = distribution.extract::<PyRef<'_, PyTruncatedGaussian>>() {
            PyMCDistributionWrapper::TruncatedGaussian(t.inner.clone())
        } else if let Ok(m) = distribution.extract::<PyRef<'_, PyMultivariateNormal>>() {
            PyMCDistributionWrapper::MultivariateNormal(m.inner.clone())
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected a distribution type: Gaussian, UniformDist, TruncatedGaussian, or MultivariateNormal",
            ));
        };

        self.variables.push((var_id, wrapper));
        Ok(())
    }

    /// Add a variable with pre-computed sample values.
    ///
    /// Each run uses ``samples[run_index]``. The list must contain at least
    /// as many values as the number of runs.
    ///
    /// Args:
    ///     var_id (MonteCarloVariableId): Typed identifier for this variable
    ///     samples (list): List of float values or numpy arrays (one per run)
    ///
    /// Raises:
    ///     TypeError: If a sample value is neither a float nor a numpy array
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     sim = bh.MonteCarloSimulation(
    ///         bh.MonteCarloStoppingCondition.fixed_runs(3), seed=42
    ///     )
    ///     sim.add_presampled(
    ///         bh.MonteCarloVariableId.MASS,
    ///         [500.0, 510.0, 490.0],
    ///     )
    ///     ```
    pub fn add_presampled(
        &mut self,
        var_id: PyMonteCarloVariableId,
        samples: &Bound<'_, PyList>,
    ) -> PyResult<()> {
        let mut holders = Vec::with_capacity(samples.len());
        for item in samples.iter() {
            if let Ok(val) = item.extract::<f64>() {
                holders.push(PyMCSampledValueHolder::Scalar(val));
            } else {
                // Try as numpy array / list -> vector
                let vec = pyany_to_f64_array1(&item, None)?;
                holders.push(PyMCSampledValueHolder::Vector(DVector::from_vec(vec)));
            }
        }
        self.presampled.push((var_id, holders));
        Ok(())
    }

    /// Add a variable with a Python callback for custom sampling.
    ///
    /// The callback is called once per run with ``(run_index, seed)`` and must
    /// return either a float (scalar) or a numpy array (vector).
    ///
    /// Args:
    ///     var_id (MonteCarloVariableId): Typed identifier for this variable
    ///     callback (callable): Function with signature ``(run_index: int, seed: int) -> float | numpy.ndarray``
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     sim = bh.MonteCarloSimulation(
    ///         bh.MonteCarloStoppingCondition.fixed_runs(100), seed=42
    ///     )
    ///
    ///     def sample_state(run_index, seed):
    ///         rng = np.random.default_rng(seed)
    ///         nominal = np.array([7e6, 0, 0, 0, 7.5e3, 0])
    ///         perturbation = rng.normal(0, [100, 100, 100, 0.1, 0.1, 0.1])
    ///         return nominal + perturbation
    ///
    ///     sim.add_callback(bh.MonteCarloVariableId.INITIAL_STATE, sample_state)
    ///     ```
    pub fn add_callback(
        &mut self,
        var_id: PyMonteCarloVariableId,
        callback: Py<PyAny>,
    ) -> PyResult<()> {
        self.callbacks.push((var_id, callback));
        Ok(())
    }

    /// Run the Monte Carlo simulation with a Python callable.
    ///
    /// The callable receives ``(run_index: int, variables: dict)`` where
    /// ``variables`` is a dictionary mapping variable names to their sampled
    /// values (float for scalars, numpy array for vectors). The callable
    /// must return a ``dict[str, float | None]`` of output values.
    ///
    /// Note:
    ///     Execution is sequential on the Python thread. For parallel orbit
    ///     propagation, use :meth:`run_orbit_propagation` instead.
    ///
    /// Args:
    ///     simulation_fn (callable): Function with signature
    ///         ``(run_index: int, variables: dict) -> dict[str, float | None]``
    ///
    /// Returns:
    ///     MonteCarloResults: Collected results with statistical accessors
    ///
    /// Raises:
    ///     BraheError: If parameter sampling fails or the callable raises an exception
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     stop = bh.MonteCarloStoppingCondition.fixed_runs(100)
    ///     sim = bh.MonteCarloSimulation(stop, seed=42)
    ///     sim.add_variable(
    ///         bh.MonteCarloVariableId.custom("x"),
    ///         bh.Gaussian(mean=0.0, std=1.0),
    ///     )
    ///
    ///     def run_fn(run_index, variables):
    ///         x = variables["x"]
    ///         return {"x_squared": x * x}
    ///
    ///     results = sim.run(run_fn)
    ///     print(f"Mean: {results.mean('x_squared')}")
    ///     ```
    #[pyo3(signature = (simulation_fn))]
    pub fn run(&self, py: Python<'_>, simulation_fn: Py<PyAny>) -> PyResult<PyMonteCarloResults> {
        // Build the Rust simulation
        let mut sim = monte_carlo::MonteCarloSimulation::new(self.config.clone());

        // Add distribution variables
        for (var_id, wrapper) in &self.variables {
            let dist: Box<dyn monte_carlo::MonteCarloDistribution> = match wrapper {
                PyMCDistributionWrapper::Gaussian(g) => Box::new(g.clone()),
                PyMCDistributionWrapper::Uniform(u) => Box::new(u.clone()),
                PyMCDistributionWrapper::TruncatedGaussian(t) => Box::new(t.clone()),
                PyMCDistributionWrapper::MultivariateNormal(m) => Box::new(m.clone()),
            };
            sim.add_variable(
                var_id.inner.clone(),
                monte_carlo::MonteCarloSamplingSource::Distribution(dist),
            );
        }

        // Add pre-sampled variables
        for (var_id, holders) in &self.presampled {
            let samples: Vec<monte_carlo::MonteCarloSampledValue> = holders
                .iter()
                .map(|h| match h {
                    PyMCSampledValueHolder::Scalar(v) => {
                        monte_carlo::MonteCarloSampledValue::Scalar(*v)
                    }
                    PyMCSampledValueHolder::Vector(v) => {
                        monte_carlo::MonteCarloSampledValue::Vector(v.clone())
                    }
                })
                .collect();
            sim.add_presampled(var_id.inner.clone(), samples);
        }

        // Add callback variables - call Python callbacks during sampling
        for (var_id, callback) in &self.callbacks {
            let cb = callback.clone_ref(py);
            sim.add_variable(
                var_id.inner.clone(),
                monte_carlo::MonteCarloSamplingSource::Callback(Box::new(
                    move |run_index, rng| {
                        // Derive a seed from the RNG to pass to Python
                        let seed = rng.next_u64();

                        Python::attach(|py| {
                            let result = cb
                                .call1(py, (run_index, seed))
                                .expect("Monte Carlo callback failed");

                            // Try to extract as f64 (scalar)
                            if let Ok(val) = result.extract::<f64>(py) {
                                return monte_carlo::MonteCarloSampledValue::Scalar(val);
                            }

                            // Try as numpy array / list -> vector
                            let bound = result.bind(py);
                            if let Ok(vec) = pyany_to_f64_array1(bound, None) {
                                return monte_carlo::MonteCarloSampledValue::Vector(
                                    DVector::from_vec(vec),
                                );
                            }

                            panic!("Monte Carlo callback must return a float or numpy array");
                        })
                    },
                )),
            );
        }

        // Run the simulation sequentially, calling the Python function for each run
        // We use the Rust sim.run() but wrap it to call back into Python
        let sim_fn = simulation_fn.clone_ref(py);
        let results = sim
            .run(move |run_index, params| {
                Python::attach(|py| {
                    // Convert sampled parameters to a Python dict
                    let dict = PyDict::new(py);

                    for var_id in params.ids() {
                        let key = match var_id {
                            monte_carlo::MonteCarloVariableId::InitialState => {
                                "initial_state".to_string()
                            }
                            monte_carlo::MonteCarloVariableId::Mass => "mass".to_string(),
                            monte_carlo::MonteCarloVariableId::DragArea => "drag_area".to_string(),
                            monte_carlo::MonteCarloVariableId::DragCoefficient => {
                                "drag_coefficient".to_string()
                            }
                            monte_carlo::MonteCarloVariableId::SrpArea => "srp_area".to_string(),
                            monte_carlo::MonteCarloVariableId::ReflectivityCoefficient => {
                                "reflectivity_coefficient".to_string()
                            }
                            monte_carlo::MonteCarloVariableId::EopTable => "eop_table".to_string(),
                            monte_carlo::MonteCarloVariableId::SpaceWeatherTable => {
                                "space_weather_table".to_string()
                            }
                            monte_carlo::MonteCarloVariableId::Custom(name) => name.clone(),
                        };

                        if let Some(value) = params.get(var_id) {
                            match value {
                                monte_carlo::MonteCarloSampledValue::Scalar(v) => {
                                    dict.set_item(&key, v).map_err(|e| {
                                        RustBraheError::Error(format!(
                                            "Failed to set dict item: {}",
                                            e
                                        ))
                                    })?;
                                }
                                monte_carlo::MonteCarloSampledValue::Vector(v) => {
                                    let n = v.nrows();
                                    let np_arr = vector_to_numpy!(py, v, n, f64);
                                    dict.set_item(&key, np_arr).map_err(|e| {
                                        RustBraheError::Error(format!(
                                            "Failed to set dict item: {}",
                                            e
                                        ))
                                    })?;
                                }
                                monte_carlo::MonteCarloSampledValue::Matrix(m) => {
                                    let r = m.nrows();
                                    let c = m.ncols();
                                    let np_arr = matrix_to_numpy!(py, m, r, c, f64);
                                    dict.set_item(&key, np_arr).map_err(|e| {
                                        RustBraheError::Error(format!(
                                            "Failed to set dict item: {}",
                                            e
                                        ))
                                    })?;
                                }
                                monte_carlo::MonteCarloSampledValue::Table(_) => {
                                    // Tables are not passed to the Python callable
                                    // (they are used internally for EOP/SW overrides)
                                }
                            }
                        }
                    }

                    // Call the Python simulation function
                    let py_result = sim_fn
                        .call1(py, (run_index, dict))
                        .map_err(|e| RustBraheError::Error(format!("Simulation function error: {}", e)))?;

                    // Convert returned dict to MonteCarloOutputs
                    let result_dict = py_result
                        .bind(py).cast::<PyDict>()
                        .map_err(|_| {
                            RustBraheError::Error(
                                "Simulation function must return a dict".to_string(),
                            )
                        })?;

                    let mut outputs = monte_carlo::MonteCarloOutputs::new();
                    for (key, value) in result_dict.iter() {
                        let name: String = key.extract().map_err(|e| {
                            RustBraheError::Error(format!("Dict key must be a string: {}", e))
                        })?;

                        if value.is_none() {
                            outputs.insert_optional_scalar(&name, None);
                        } else if let Ok(v) = value.extract::<f64>() {
                            outputs.insert_scalar(&name, v);
                        } else if let Ok(vec) = pyany_to_f64_array1(&value, None) {
                            outputs.insert_vector(&name, DVector::from_vec(vec));
                        } else {
                            outputs.insert_optional_scalar(&name, None);
                        }
                    }

                    Ok(outputs)
                })
            })
            .map_err(|e| BraheError::new_err(e.to_string()))?;

        Ok(PyMonteCarloResults { inner: results })
    }

    /// Run a built-in orbit propagation Monte Carlo simulation.
    ///
    /// Each run builds a numerical orbit propagator from the nominal
    /// configuration plus sampled parameter overrides, propagates to the
    /// target epoch, and extracts configured outputs. This executes entirely
    /// in Rust and uses rayon for parallel execution.
    ///
    /// Note:
    ///     Thread-local EOP and space weather providers are automatically
    ///     set from sampled table data when ``MonteCarloVariableId.EOP_TABLE``
    ///     or ``MonteCarloVariableId.SPACE_WEATHER_TABLE`` variables are registered.
    ///
    /// Args:
    ///     epoch (Epoch): Initial epoch for propagation
    ///     propagation_config (NumericalPropagationConfig): Integrator configuration
    ///     force_config (ForceModelConfig): Force model configuration
    ///     target_epoch (Epoch): Target epoch to propagate to
    ///     outputs (list[str]): Output names to extract. Supported values:
    ///         ``"final_state"``, ``"final_epoch"``, ``"simulation_duration"``,
    ///         ``"final_altitude"``, ``"final_semi_major_axis"``, ``"final_eccentricity"``
    ///     params (numpy.ndarray | None): Optional spacecraft parameter vector
    ///         ``[mass, drag_area, Cd, srp_area, Cr, ...]``. Defaults to None.
    ///     events (list[dict] | None): Optional event detector configurations.
    ///         Each dict should have keys: ``"type"`` (``"altitude"``),
    ///         ``"altitude"`` (float, meters), ``"name"`` (str),
    ///         ``"terminal"`` (bool), ``"direction"`` (``"increasing"``,
    ///         ``"decreasing"``, or ``"any"``). Defaults to None.
    ///
    /// Returns:
    ///     MonteCarloResults: Collected results with statistical accessors
    ///
    /// Raises:
    ///     BraheError: If parameter sampling or propagation fails
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     eop = bh.StaticEOPProvider.from_zero()
    ///     bh.set_global_eop_provider_from_static_provider(eop)
    ///
    ///     epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.UTC)
    ///
    ///     stop = bh.MonteCarloStoppingCondition.fixed_runs(10)
    ///     sim = bh.MonteCarloSimulation(stop, seed=42)
    ///
    ///     # Add initial state with callback
    ///     def sample_state(run_index, seed):
    ///         rng = np.random.default_rng(seed)
    ///         r = bh.R_EARTH + 500e3
    ///         v = (bh.GM_EARTH / r) ** 0.5
    ///         state = np.array([r, 0, 0, 0, v, 0])
    ///         state[:3] += rng.normal(0, 100, 3)
    ///         return state
    ///
    ///     sim.add_callback(bh.MonteCarloVariableId.INITIAL_STATE, sample_state)
    ///
    ///     results = sim.run_orbit_propagation(
    ///         epoch=epoch,
    ///         propagation_config=bh.NumericalPropagationConfig(),
    ///         force_config=bh.ForceModelConfig.two_body(),
    ///         target_epoch=epoch + 3600.0,
    ///         outputs=["final_altitude", "final_semi_major_axis"],
    ///     )
    ///     print(f"Mean altitude: {results.mean('final_altitude')}")
    ///     ```
    #[pyo3(signature = (epoch, propagation_config, force_config, target_epoch, outputs, params=None, events=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn run_orbit_propagation(
        &self,
        py: Python<'_>,
        epoch: &PyEpoch,
        propagation_config: &PyNumericalPropagationConfig,
        force_config: &PyForceModelConfig,
        target_epoch: &PyEpoch,
        outputs: Vec<String>,
        params: Option<&Bound<'_, PyAny>>,
        events: Option<&Bound<'_, PyList>>,
    ) -> PyResult<PyMonteCarloResults> {
        // Build the Rust simulation
        let mut sim = monte_carlo::MonteCarloSimulation::new(self.config.clone());

        // Add distribution variables
        for (var_id, wrapper) in &self.variables {
            let dist: Box<dyn monte_carlo::MonteCarloDistribution> = match wrapper {
                PyMCDistributionWrapper::Gaussian(g) => Box::new(g.clone()),
                PyMCDistributionWrapper::Uniform(u) => Box::new(u.clone()),
                PyMCDistributionWrapper::TruncatedGaussian(t) => Box::new(t.clone()),
                PyMCDistributionWrapper::MultivariateNormal(m) => Box::new(m.clone()),
            };
            sim.add_variable(
                var_id.inner.clone(),
                monte_carlo::MonteCarloSamplingSource::Distribution(dist),
            );
        }

        // Add pre-sampled variables
        for (var_id, holders) in &self.presampled {
            let samples: Vec<monte_carlo::MonteCarloSampledValue> = holders
                .iter()
                .map(|h| match h {
                    PyMCSampledValueHolder::Scalar(v) => {
                        monte_carlo::MonteCarloSampledValue::Scalar(*v)
                    }
                    PyMCSampledValueHolder::Vector(v) => {
                        monte_carlo::MonteCarloSampledValue::Vector(v.clone())
                    }
                })
                .collect();
            sim.add_presampled(var_id.inner.clone(), samples);
        }

        // Add callback variables
        for (var_id, callback) in &self.callbacks {
            let cb = callback.clone_ref(py);
            sim.add_variable(
                var_id.inner.clone(),
                monte_carlo::MonteCarloSamplingSource::Callback(Box::new(
                    move |run_index, rng| {
                        let seed = rng.next_u64();
                        Python::attach(|py| {
                            let result = cb
                                .call1(py, (run_index, seed))
                                .expect("Monte Carlo callback failed");
                            if let Ok(val) = result.extract::<f64>(py) {
                                return monte_carlo::MonteCarloSampledValue::Scalar(val);
                            }
                            let bound = result.bind(py);
                            if let Ok(vec) = pyany_to_f64_array1(bound, None) {
                                return monte_carlo::MonteCarloSampledValue::Vector(
                                    DVector::from_vec(vec),
                                );
                            }
                            panic!("Monte Carlo callback must return a float or numpy array");
                        })
                    },
                )),
            );
        }

        // Convert params
        let rust_params = if let Some(p) = params {
            let vec = pyany_to_f64_array1(p, None)?;
            Some(DVector::from_vec(vec))
        } else {
            None
        };

        // Convert outputs
        let rust_outputs: Vec<monte_carlo::OrbitMonteCarloOutput> = outputs
            .iter()
            .map(|name| match name.as_str() {
                "final_state" => Ok(monte_carlo::OrbitMonteCarloOutput::FinalState),
                "final_epoch" => Ok(monte_carlo::OrbitMonteCarloOutput::FinalEpoch),
                "simulation_duration" => {
                    Ok(monte_carlo::OrbitMonteCarloOutput::SimulationDuration)
                }
                "final_altitude" => Ok(monte_carlo::OrbitMonteCarloOutput::FinalAltitude),
                "final_semi_major_axis" => {
                    Ok(monte_carlo::OrbitMonteCarloOutput::FinalSemiMajorAxis)
                }
                "final_eccentricity" => {
                    Ok(monte_carlo::OrbitMonteCarloOutput::FinalEccentricity)
                }
                other => {
                    // Check for event-related outputs
                    if let Some(event_name) = other.strip_prefix("first_event_duration_") {
                        Ok(monte_carlo::OrbitMonteCarloOutput::FirstEventDuration(
                            event_name.to_string(),
                        ))
                    } else if let Some(event_name) = other.strip_prefix("last_event_duration_") {
                        Ok(monte_carlo::OrbitMonteCarloOutput::LastEventDuration(
                            event_name.to_string(),
                        ))
                    } else if let Some(event_name) = other.strip_prefix("event_count_") {
                        Ok(monte_carlo::OrbitMonteCarloOutput::EventCount(
                            event_name.to_string(),
                        ))
                    } else {
                        Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Unknown output name: '{}'. Supported: final_state, final_epoch, \
                             simulation_duration, final_altitude, final_semi_major_axis, \
                             final_eccentricity, first_event_duration_<name>, \
                             last_event_duration_<name>, event_count_<name>",
                            other
                        )))
                    }
                }
            })
            .collect::<PyResult<Vec<_>>>()?;

        // Convert events
        let rust_events = if let Some(event_list) = events {
            let mut evts = Vec::new();
            for item in event_list.iter() {
                let dict = item.cast::<PyDict>().map_err(|_| {
                    pyo3::exceptions::PyTypeError::new_err("Each event must be a dict")
                })?;

                let event_type: String = dict
                    .get_item("type")?
                    .ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err("Event dict must have 'type' key")
                    })?
                    .extract()?;

                match event_type.as_str() {
                    "altitude" => {
                        let altitude: f64 = dict
                            .get_item("altitude")?
                            .ok_or_else(|| {
                                pyo3::exceptions::PyValueError::new_err(
                                    "Altitude event must have 'altitude' key",
                                )
                            })?
                            .extract()?;
                        let name: String = dict
                            .get_item("name")?
                            .ok_or_else(|| {
                                pyo3::exceptions::PyValueError::new_err(
                                    "Altitude event must have 'name' key",
                                )
                            })?
                            .extract()?;
                        let terminal: bool = dict
                            .get_item("terminal")?
                            .map(|v| v.extract::<bool>().unwrap_or(false))
                            .unwrap_or(false);
                        let direction_str: String = dict
                            .get_item("direction")?
                            .map(|v| v.extract::<String>().unwrap_or_else(|_| "any".to_string()))
                            .unwrap_or_else(|| "any".to_string());

                        let direction = match direction_str.as_str() {
                            "increasing" => crate::events::EventDirection::Increasing,
                            "decreasing" => crate::events::EventDirection::Decreasing,
                            _ => crate::events::EventDirection::Any,
                        };

                        evts.push(monte_carlo::OrbitMonteCarloEventConfig::Altitude {
                            altitude,
                            name,
                            terminal,
                            direction,
                        });
                    }
                    other => {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "Unknown event type: '{}'. Supported: 'altitude'",
                            other
                        )));
                    }
                }
            }
            evts
        } else {
            Vec::new()
        };

        // Build orbit config
        let orbit_config = monte_carlo::OrbitMonteCarloConfig {
            epoch: epoch.obj,
            propagation_config: propagation_config.config.clone(),
            force_config: force_config.config.clone(),
            params: rust_params,
            target_epoch: target_epoch.obj,
            events: rust_events,
            outputs: rust_outputs,
        };

        // Run orbit propagation (uses rayon internally for parallel execution)
        let results = sim
            .run_orbit_propagation(orbit_config)
            .map_err(|e: RustBraheError| BraheError::new_err(e.to_string()))?;

        Ok(PyMonteCarloResults { inner: results })
    }

    pub fn __repr__(&self) -> String {
        format!(
            "MonteCarloSimulation(variables={}, config={})",
            self.variables.len() + self.presampled.len() + self.callbacks.len(),
            self.config
        )
    }
}

// ---------------------------------------------------------------------------
// MonteCarloResults
// ---------------------------------------------------------------------------

/// Results from a complete Monte Carlo simulation.
///
/// Provides access to individual run data and statistical analysis methods
/// for computing means, standard deviations, percentiles, and covariances
/// across all successful runs.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # After running a simulation...
///     results = sim.run(run_fn)
///
///     print(f"Successful: {results.num_successful}")
///     print(f"Failed: {results.num_failed}")
///     print(f"Mean altitude: {results.mean('final_altitude')}")
///     print(f"Std altitude: {results.std('final_altitude')}")
///     print(f"Median: {results.percentile('final_altitude', 0.5)}")
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "MonteCarloResults")]
pub struct PyMonteCarloResults {
    pub(crate) inner: monte_carlo::MonteCarloResults,
}

#[pymethods]
impl PyMonteCarloResults {
    /// Compute the arithmetic mean of a scalar output across successful runs.
    ///
    /// Args:
    ///     name (str): Name of the scalar output
    ///
    /// Returns:
    ///     float: The mean value
    ///
    /// Raises:
    ///     BraheError: If no values are available for the given output name
    ///
    /// Example:
    ///     ```python
    ///     mean_alt = results.mean("final_altitude")
    ///     ```
    pub fn mean(&self, name: &str) -> PyResult<f64> {
        self.inner
            .mean(name)
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Compute the sample standard deviation (ddof=1) of a scalar output.
    ///
    /// Args:
    ///     name (str): Name of the scalar output
    ///
    /// Returns:
    ///     float: The sample standard deviation
    ///
    /// Raises:
    ///     BraheError: If fewer than 2 values are available
    ///
    /// Example:
    ///     ```python
    ///     std_alt = results.std("final_altitude")
    ///     ```
    pub fn std(&self, name: &str) -> PyResult<f64> {
        self.inner
            .std(name)
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Compute a percentile of a scalar output using linear interpolation.
    ///
    /// Args:
    ///     name (str): Name of the scalar output
    ///     p (float): Percentile in [0.0, 1.0] (e.g., 0.5 for median)
    ///
    /// Returns:
    ///     float: The interpolated percentile value
    ///
    /// Raises:
    ///     BraheError: If no values are available or p is out of range
    ///
    /// Example:
    ///     ```python
    ///     median = results.percentile("final_altitude", 0.5)
    ///     p95 = results.percentile("final_altitude", 0.95)
    ///     ```
    pub fn percentile(&self, name: &str, p: f64) -> PyResult<f64> {
        self.inner
            .percentile(name, p)
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Get the minimum value of a scalar output across successful runs.
    ///
    /// Args:
    ///     name (str): Name of the scalar output
    ///
    /// Returns:
    ///     float: The minimum value
    ///
    /// Raises:
    ///     BraheError: If no values are available
    ///
    /// Example:
    ///     ```python
    ///     min_alt = results.min("final_altitude")
    ///     ```
    pub fn min(&self, name: &str) -> PyResult<f64> {
        self.inner
            .min(name)
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Get the maximum value of a scalar output across successful runs.
    ///
    /// Args:
    ///     name (str): Name of the scalar output
    ///
    /// Returns:
    ///     float: The maximum value
    ///
    /// Raises:
    ///     BraheError: If no values are available
    ///
    /// Example:
    ///     ```python
    ///     max_alt = results.max("final_altitude")
    ///     ```
    pub fn max(&self, name: &str) -> PyResult<f64> {
        self.inner
            .max(name)
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Compute the standard error of the mean for a scalar output.
    ///
    /// Standard error = sample std / sqrt(n).
    ///
    /// Args:
    ///     name (str): Name of the scalar output
    ///
    /// Returns:
    ///     float: The standard error
    ///
    /// Raises:
    ///     BraheError: If fewer than 2 values are available
    ///
    /// Example:
    ///     ```python
    ///     se = results.standard_error("final_altitude")
    ///     ```
    pub fn standard_error(&self, name: &str) -> PyResult<f64> {
        self.inner
            .standard_error(name)
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Compute the element-wise mean of a vector output across successful runs.
    ///
    /// Args:
    ///     name (str): Name of the vector output
    ///
    /// Returns:
    ///     numpy.ndarray: The mean vector
    ///
    /// Raises:
    ///     BraheError: If no vector values are available or dimensions mismatch
    ///
    /// Example:
    ///     ```python
    ///     mean_state = results.mean_vector("final_state")
    ///     ```
    pub fn mean_vector<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
        let v = self
            .inner
            .mean_vector(name)
            .map_err(|e| BraheError::new_err(e.to_string()))?;
        let flat_vec: Vec<f64> = (0..v.nrows()).map(|i| v[i]).collect();
        Ok(flat_vec.into_pyarray(py))
    }

    /// Compute the sample covariance matrix of a vector output.
    ///
    /// Args:
    ///     name (str): Name of the vector output
    ///
    /// Returns:
    ///     numpy.ndarray: The covariance matrix (n x n)
    ///
    /// Raises:
    ///     BraheError: If fewer than 2 vectors are available
    ///
    /// Example:
    ///     ```python
    ///     cov = results.covariance("final_state")
    ///     ```
    pub fn covariance<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
        let m = self
            .inner
            .covariance(name)
            .map_err(|e| BraheError::new_err(e.to_string()))?;
        let r = m.nrows();
        let c = m.ncols();
        // Convert DMatrix to numpy array manually (macro doesn't work with dynamic sizes due to move semantics)
        let m_ref = &m;
        let flat_vec: Vec<f64> = (0..r)
            .flat_map(|i| (0..c).map(move |j| m_ref[(i, j)]))
            .collect();
        let arr = flat_vec.into_pyarray(py).reshape([r, c]).unwrap();
        Ok(arr)
    }

    /// Collect all scalar values for a named output from successful runs.
    ///
    /// Args:
    ///     name (str): Name of the scalar output
    ///
    /// Returns:
    ///     numpy.ndarray: 1-D array of scalar values from successful runs
    ///
    /// Example:
    ///     ```python
    ///     altitudes = results.scalar_values("final_altitude")
    ///     ```
    pub fn scalar_values<'py>(&self, py: Python<'py>, name: &str) -> Bound<'py, PyArray<f64, Ix1>> {
        let values = self.inner.scalar_values(name);
        values.into_pyarray(py)
    }

    /// Number of runs that completed successfully.
    ///
    /// Returns:
    ///     int: Count of successful runs
    #[getter]
    pub fn num_successful(&self) -> usize {
        self.inner.num_successful()
    }

    /// Number of runs that failed.
    ///
    /// Returns:
    ///     int: Count of failed runs
    #[getter]
    pub fn num_failed(&self) -> usize {
        self.inner.num_failed()
    }

    /// Whether the simulation converged (if convergence checking was enabled).
    ///
    /// Returns:
    ///     bool | None: True if converged, False if not, None if convergence
    ///         was not checked (fixed-runs mode)
    #[getter]
    pub fn converged(&self) -> Option<bool> {
        self.inner.converged
    }

    /// Total number of runs executed.
    ///
    /// Returns:
    ///     int: Total number of runs
    #[getter]
    pub fn num_runs(&self) -> usize {
        self.inner.runs.len()
    }

    /// Master seed used for reproducibility.
    ///
    /// Returns:
    ///     int: The master seed
    #[getter]
    pub fn master_seed(&self) -> u64 {
        self.inner.master_seed
    }

    /// Final standard errors for monitored outputs (convergence mode only).
    ///
    /// Returns:
    ///     dict[str, float] | None: Map of output name to final standard error,
    ///         or None if convergence checking was not used
    #[getter]
    pub fn final_standard_errors(&self) -> Option<HashMap<String, f64>> {
        self.inner.final_standard_errors.clone()
    }

    /// Get a list of individual run results.
    ///
    /// Returns:
    ///     list[MonteCarloRun]: List of all simulation runs
    ///
    /// Example:
    ///     ```python
    ///     for run in results.runs:
    ///         if run.succeeded:
    ///             print(f"Run {run.run_index}: {run.outputs}")
    ///     ```
    #[getter]
    pub fn runs(&self, py: Python<'_>) -> PyResult<Vec<Py<PyMonteCarloRun>>> {
        let mut py_runs = Vec::with_capacity(self.inner.runs.len());
        for run in &self.inner.runs {
            let succeeded = run.succeeded();
            let run_index = run.run_index;

            // Convert outputs to Python dict
            let outputs_dict = if let Some(outputs) = run.outputs() {
                let dict = PyDict::new(py);
                for (name, value) in &outputs.values {
                    match value {
                        monte_carlo::MonteCarloOutputValue::Scalar(v) => {
                            dict.set_item(name, v)?;
                        }
                        monte_carlo::MonteCarloOutputValue::Vector(v) => {
                            let n = v.nrows();
                            let np_arr = vector_to_numpy!(py, v, n, f64);
                            dict.set_item(name, np_arr)?;
                        }
                        monte_carlo::MonteCarloOutputValue::OptionalScalar(opt) => {
                            match opt {
                                Some(v) => dict.set_item(name, v)?,
                                None => dict.set_item(name, py.None())?,
                            }
                        }
                    }
                }
                Some(dict.unbind())
            } else {
                None
            };

            // Convert sampled parameters to Python dict
            let params_dict = PyDict::new(py);
            for var_id in run.sampled_parameters.ids() {
                let key = format!("{}", var_id);
                if let Some(value) = run.sampled_parameters.get(var_id) {
                    match value {
                        monte_carlo::MonteCarloSampledValue::Scalar(v) => {
                            params_dict.set_item(&key, v)?;
                        }
                        monte_carlo::MonteCarloSampledValue::Vector(v) => {
                            let n = v.nrows();
                            let np_arr = vector_to_numpy!(py, v, n, f64);
                            params_dict.set_item(&key, np_arr)?;
                        }
                        monte_carlo::MonteCarloSampledValue::Matrix(m) => {
                            let r = m.nrows();
                            let c = m.ncols();
                            let np_arr = matrix_to_numpy!(py, m, r, c, f64);
                            params_dict.set_item(&key, np_arr)?;
                        }
                        monte_carlo::MonteCarloSampledValue::Table(t) => {
                            params_dict
                                .set_item(&key, format!("Table({} rows)", t.len()))?;
                        }
                    }
                }
            }

            // Get error message if failed
            let error_message = if !succeeded {
                run.result
                    .as_ref()
                    .err()
                    .map(|e| e.to_string())
            } else {
                None
            };

            let py_run = Py::new(
                py,
                PyMonteCarloRun {
                    run_index,
                    succeeded,
                    outputs: outputs_dict,
                    sampled_parameters: params_dict.unbind(),
                    error_message,
                    run_seed: run.sampled_parameters.run_seed,
                },
            )?;
            py_runs.push(py_run);
        }
        Ok(py_runs)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "MonteCarloResults(runs={}, successful={}, failed={}, converged={:?})",
            self.inner.runs.len(),
            self.inner.num_successful(),
            self.inner.num_failed(),
            self.inner.converged
        )
    }

    pub fn __len__(&self) -> usize {
        self.inner.runs.len()
    }
}

// ---------------------------------------------------------------------------
// MonteCarloRun
// ---------------------------------------------------------------------------

/// Data from a single Monte Carlo simulation run.
///
/// Contains the sampled input parameters, the simulation outputs (if the run
/// succeeded), and metadata about the run.
///
/// Attributes:
///     run_index (int): Zero-based index of this run
///     succeeded (bool): Whether the run completed successfully
///     outputs (dict | None): Output values from the run, or None if the run failed
///     sampled_parameters (dict): The sampled input parameters used for this run
///     error_message (str | None): Error message if the run failed
///     run_seed (int): The random seed used for this specific run
///
/// Example:
///     ```python
///     for run in results.runs:
///         if run.succeeded:
///             print(f"Run {run.run_index}: altitude={run.outputs['final_altitude']}")
///         else:
///             print(f"Run {run.run_index} failed: {run.error_message}")
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "MonteCarloRun")]
pub struct PyMonteCarloRun {
    /// Zero-based index of this run in the simulation.
    #[pyo3(get)]
    pub run_index: usize,

    /// Whether this run completed successfully.
    #[pyo3(get)]
    pub succeeded: bool,

    /// Output values as a dict, or None if the run failed.
    #[pyo3(get)]
    pub outputs: Option<Py<PyDict>>,

    /// The sampled input parameters used for this run.
    #[pyo3(get)]
    pub sampled_parameters: Py<PyDict>,

    /// Error message if the run failed, None otherwise.
    #[pyo3(get)]
    pub error_message: Option<String>,

    /// Random seed used for this specific run.
    #[pyo3(get)]
    pub run_seed: u64,
}

#[pymethods]
impl PyMonteCarloRun {
    pub fn __repr__(&self) -> String {
        if self.succeeded {
            format!("MonteCarloRun(index={}, succeeded=True)", self.run_index)
        } else {
            format!(
                "MonteCarloRun(index={}, succeeded=False, error={:?})",
                self.run_index, self.error_message
            )
        }
    }
}
