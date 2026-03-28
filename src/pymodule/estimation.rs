// =============================================================================
// Python bindings for the estimation module.
//
// Provides Python wrappers for:
// - Configuration types: ProcessNoiseConfig, EKFConfig, UKFConfig, BLSConfig
// - Data types: Observation, FilterRecord, BLSIterationRecord, BLSObservationResidual
// - Built-in measurement models (6 types, inertial + ECEF)
// - MeasurementModel base class for Python-defined custom models
// - ExtendedKalmanFilter
// - UnscentedKalmanFilter
// - BatchLeastSquares
// =============================================================================

use crate::estimation;
use crate::estimation::MeasurementModel;
use crate::estimation::DynamicsSource;
use crate::math::jacobian::{DifferenceMethod, PerturbationStrategy};

// =============================================================================
// ProcessNoiseConfig
// =============================================================================

/// Process noise configuration for sequential filters.
///
/// Controls how process noise Q is applied to the predicted covariance.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     q = np.diag([1e-6]*3 + [1e-8]*3)
///     pn = bh.ProcessNoiseConfig(q, scale_with_dt=True)
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "ProcessNoiseConfig")]
#[derive(Clone)]
pub struct PyProcessNoiseConfig {
    pub(crate) config: estimation::ProcessNoiseConfig,
}

#[pymethods]
impl PyProcessNoiseConfig {
    /// Create a new ProcessNoiseConfig.
    ///
    /// Args:
    ///     q_matrix (numpy.ndarray): Process noise matrix Q (state_dim x state_dim).
    ///     scale_with_dt (bool): If True, Q scales with time step (continuous-time model).
    ///         Defaults to False.
    ///
    /// Returns:
    ///     ProcessNoiseConfig: New process noise configuration.
    #[new]
    #[pyo3(signature = (q_matrix, scale_with_dt=false))]
    fn new(q_matrix: PyReadonlyArray2<f64>, scale_with_dt: bool) -> PyResult<Self> {
        let shape = q_matrix.shape();
        let n = shape[0];
        if shape[1] != n {
            return Err(exceptions::PyValueError::new_err(
                "q_matrix must be a square matrix",
            ));
        }
        let data: Vec<f64> = q_matrix.as_slice()?.to_vec();
        let q = DMatrix::from_row_slice(n, n, &data);

        Ok(PyProcessNoiseConfig {
            config: estimation::ProcessNoiseConfig {
                q_matrix: q,
                scale_with_dt,
            },
        })
    }

    /// Get the process noise matrix Q.
    ///
    /// Returns:
    ///     numpy.ndarray: Process noise matrix (state_dim x state_dim).
    #[getter]
    fn q_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, numpy::Ix2>> {
        let n = self.config.q_matrix.nrows();
        let c = self.config.q_matrix.ncols();
        matrix_to_numpy!(py, self.config.q_matrix, n, c, f64)
    }

    /// Whether Q scales with the time step.
    ///
    /// Returns:
    ///     bool: True if continuous-time model.
    #[getter]
    fn scale_with_dt(&self) -> bool {
        self.config.scale_with_dt
    }

    fn __repr__(&self) -> String {
        format!(
            "ProcessNoiseConfig({}x{}, scale_with_dt={})",
            self.config.q_matrix.nrows(),
            self.config.q_matrix.ncols(),
            self.config.scale_with_dt,
        )
    }
}

// =============================================================================
// EKFConfig
// =============================================================================

/// Configuration for the Extended Kalman Filter.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     config = bh.EKFConfig()  # Defaults
///     config = bh.EKFConfig(process_noise=bh.ProcessNoiseConfig(np.eye(6) * 1e-6))
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "EKFConfig")]
#[derive(Clone)]
pub struct PyEKFConfig {
    pub(crate) config: estimation::EKFConfig,
}

#[pymethods]
impl PyEKFConfig {
    /// Create a new EKFConfig.
    ///
    /// Args:
    ///     process_noise (ProcessNoiseConfig or None): Optional process noise. Defaults to None.
    ///     store_records (bool): Whether to store filter records. Defaults to True.
    ///
    /// Returns:
    ///     EKFConfig: New EKF configuration.
    #[new]
    #[pyo3(signature = (process_noise=None, store_records=true))]
    fn new(process_noise: Option<PyProcessNoiseConfig>, store_records: bool) -> Self {
        PyEKFConfig {
            config: estimation::EKFConfig {
                process_noise: process_noise.map(|pn| pn.config),
                store_records,
            },
        }
    }

    /// Create a default EKF configuration.
    ///
    /// Returns:
    ///     EKFConfig: Default configuration (no process noise, records stored).
    #[staticmethod]
    #[pyo3(name = "default")]
    fn py_default() -> Self {
        PyEKFConfig {
            config: estimation::EKFConfig::default(),
        }
    }

    /// Whether filter records are stored.
    ///
    /// Returns:
    ///     bool: True if records are stored.
    #[getter]
    fn store_records(&self) -> bool {
        self.config.store_records
    }

    fn __repr__(&self) -> String {
        format!(
            "EKFConfig(process_noise={}, store_records={})",
            if self.config.process_noise.is_some() { "set" } else { "None" },
            self.config.store_records,
        )
    }
}

// =============================================================================
// Observation
// =============================================================================

/// A single observation (measurement) at a specific epoch.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     obs = bh.Observation(epoch, np.array([6878e3, 0.0, 0.0]), model_index=0)
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "Observation")]
#[derive(Clone)]
pub struct PyObservation {
    pub(crate) observation: estimation::Observation,
}

#[pymethods]
impl PyObservation {
    /// Create a new Observation.
    ///
    /// Args:
    ///     epoch (Epoch): Time of the observation.
    ///     measurement (numpy.ndarray): Measurement vector z in SI units.
    ///     model_index (int): Index into the estimator's measurement model list.
    ///
    /// Returns:
    ///     Observation: New observation instance.
    #[new]
    #[pyo3(signature = (epoch, measurement, model_index=0))]
    fn new(
        epoch: &PyEpoch,
        measurement: PyReadonlyArray1<f64>,
        model_index: usize,
    ) -> PyResult<Self> {
        let meas_vec = DVector::from_column_slice(measurement.as_slice()?);
        Ok(PyObservation {
            observation: estimation::Observation::new(epoch.obj, meas_vec, model_index),
        })
    }

    /// Epoch of this observation.
    ///
    /// Returns:
    ///     Epoch: Observation time.
    #[getter]
    fn epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.observation.epoch,
        }
    }

    /// Measurement vector.
    ///
    /// Returns:
    ///     numpy.ndarray: Measurement vector.
    #[getter]
    fn measurement<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let n = self.observation.measurement.len();
        vector_to_numpy!(py, self.observation.measurement, n, f64)
    }

    /// Index into the estimator's measurement model list.
    ///
    /// Returns:
    ///     int: Model index.
    #[getter]
    fn model_index(&self) -> usize {
        self.observation.model_index
    }

    fn __repr__(&self) -> String {
        format!(
            "Observation(epoch={}, dim={}, model_index={})",
            self.observation.epoch,
            self.observation.measurement.len(),
            self.observation.model_index,
        )
    }
}

// =============================================================================
// FilterRecord
// =============================================================================

/// Record of a single filter update step.
///
/// Contains pre-fit and post-fit states, covariances, residuals, and Kalman gain.
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "FilterRecord")]
#[derive(Clone)]
pub struct PyFilterRecord {
    pub(crate) record: estimation::FilterRecord,
}

#[pymethods]
impl PyFilterRecord {
    /// Epoch of this record.
    ///
    /// Returns:
    ///     Epoch: Record epoch.
    #[getter]
    fn epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.record.epoch,
        }
    }

    /// State before measurement update (after prediction).
    ///
    /// Returns:
    ///     numpy.ndarray: Predicted state vector.
    #[getter]
    fn state_predicted<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let n = self.record.state_predicted.len();
        vector_to_numpy!(py, self.record.state_predicted, n, f64)
    }

    /// Covariance before measurement update (after prediction).
    ///
    /// Returns:
    ///     numpy.ndarray: Predicted covariance matrix.
    #[getter]
    fn covariance_predicted<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, numpy::Ix2>> {
        let r = self.record.covariance_predicted.nrows();
        let c = self.record.covariance_predicted.ncols();
        matrix_to_numpy!(py, self.record.covariance_predicted, r, c, f64)
    }

    /// State after measurement update.
    ///
    /// Returns:
    ///     numpy.ndarray: Updated state vector.
    #[getter]
    fn state_updated<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let n = self.record.state_updated.len();
        vector_to_numpy!(py, self.record.state_updated, n, f64)
    }

    /// Covariance after measurement update.
    ///
    /// Returns:
    ///     numpy.ndarray: Updated covariance matrix.
    #[getter]
    fn covariance_updated<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, numpy::Ix2>> {
        let r = self.record.covariance_updated.nrows();
        let c = self.record.covariance_updated.ncols();
        matrix_to_numpy!(py, self.record.covariance_updated, r, c, f64)
    }

    /// Pre-fit residual: z - h(x_predicted).
    ///
    /// Returns:
    ///     numpy.ndarray: Pre-fit residual vector.
    #[getter]
    fn prefit_residual<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let n = self.record.prefit_residual.len();
        vector_to_numpy!(py, self.record.prefit_residual, n, f64)
    }

    /// Post-fit residual: z - h(x_updated).
    ///
    /// Returns:
    ///     numpy.ndarray: Post-fit residual vector.
    #[getter]
    fn postfit_residual<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let n = self.record.postfit_residual.len();
        vector_to_numpy!(py, self.record.postfit_residual, n, f64)
    }

    /// Kalman gain matrix.
    ///
    /// Returns:
    ///     numpy.ndarray: Kalman gain matrix.
    #[getter]
    fn kalman_gain<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, numpy::Ix2>> {
        let r = self.record.kalman_gain.nrows();
        let c = self.record.kalman_gain.ncols();
        matrix_to_numpy!(py, self.record.kalman_gain, r, c, f64)
    }

    /// Measurement model name.
    ///
    /// Returns:
    ///     str: Name of the measurement model used.
    #[getter]
    fn measurement_name(&self) -> &str {
        &self.record.measurement_name
    }

    fn __repr__(&self) -> String {
        format!(
            "FilterRecord(epoch={}, model={})",
            self.record.epoch, self.record.measurement_name,
        )
    }
}


// =============================================================================
// Built-in Measurement Model Bindings
// =============================================================================

macro_rules! impl_measurement_model_binding {
    (
        $py_name:ident, $rust_type:ty, $python_name:expr,
        constructors: [$($ctor:tt)*],
        doc: $doc:expr
    ) => {
        #[doc = $doc]
        #[pyclass(module = "brahe._brahe")]
        #[pyo3(name = $python_name)]
        pub struct $py_name {
            pub(crate) model: $rust_type,
        }

        #[pymethods]
        impl $py_name {
            $($ctor)*

            /// Compute predicted measurement from state.
            ///
            /// Args:
            ///     epoch (Epoch): Current epoch.
            ///     state (numpy.ndarray): Current state vector.
            ///
            /// Returns:
            ///     numpy.ndarray: Predicted measurement vector.
            fn predict<'py>(
                &self,
                py: Python<'py>,
                epoch: &PyEpoch,
                state: PyReadonlyArray1<f64>,
            ) -> PyResult<Bound<'py, PyArray<f64, numpy::Ix1>>> {
                let state_vec = nalgebra::DVector::from_column_slice(state.as_slice()?);
                let vec = self.model.predict(&epoch.obj, &state_vec, None)
                    .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
                let flat: Vec<f64> = (0..vec.len()).map(|i| vec[i]).collect();
                Ok(flat.into_pyarray(py))
            }

            /// Compute measurement Jacobian H = dh/dx.
            ///
            /// Args:
            ///     epoch (Epoch): Current epoch.
            ///     state (numpy.ndarray): Current state vector.
            ///
            /// Returns:
            ///     numpy.ndarray: Measurement Jacobian matrix (m x n).
            fn jacobian<'py>(
                &self,
                py: Python<'py>,
                epoch: &PyEpoch,
                state: PyReadonlyArray1<f64>,
            ) -> PyResult<Bound<'py, PyArray<f64, numpy::Ix2>>> {
                let state_vec = nalgebra::DVector::from_column_slice(state.as_slice()?);
                let mat = self.model.jacobian(&epoch.obj, &state_vec, None)
                    .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
                let rows = mat.nrows();
                let cols = mat.ncols();
                let mut flat = Vec::with_capacity(rows * cols);
                for i in 0..rows {
                    for j in 0..cols {
                        flat.push(mat[(i, j)]);
                    }
                }
                Ok(flat.into_pyarray(py).reshape([rows, cols]).unwrap())
            }

            /// Get the measurement noise covariance matrix R.
            ///
            /// Returns:
            ///     numpy.ndarray: Noise covariance matrix (m x m).
            fn noise_covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, numpy::Ix2>> {
                let mat = self.model.noise_covariance();
                let rows = mat.nrows();
                let cols = mat.ncols();
                let mut flat = Vec::with_capacity(rows * cols);
                for i in 0..rows {
                    for j in 0..cols {
                        flat.push(mat[(i, j)]);
                    }
                }
                flat.into_pyarray(py).reshape([rows, cols]).unwrap()
            }

            /// Get the measurement dimension.
            ///
            /// Returns:
            ///     int: Measurement vector dimension.
            fn measurement_dim(&self) -> usize {
                self.model.measurement_dim()
            }

            /// Get the measurement model name.
            ///
            /// Returns:
            ///     str: Model name.
            fn name(&self) -> &str {
                self.model.name()
            }

            fn __repr__(&self) -> String {
                format!("{}(dim={})", self.model.name(), self.model.measurement_dim())
            }
        }
    };
}

impl_measurement_model_binding!(
    PyInertialPositionMeasurementModel,
    estimation::InertialPositionMeasurementModel,
    "InertialPositionMeasurementModel",
    constructors: [
        /// Create an inertial position measurement model.
        ///
        /// Args:
        ///     sigma (float): Position noise standard deviation (meters).
        ///
        /// Returns:
        ///     InertialPositionMeasurementModel: New model instance.
        ///
        /// Example:
        ///     ```python
        ///     import brahe as bh
        ///     model = bh.InertialPositionMeasurementModel(10.0)
        ///     ```
        #[new]
        fn new(sigma: f64) -> Self {
            PyInertialPositionMeasurementModel {
                model: estimation::InertialPositionMeasurementModel::new(sigma),
            }
        }

        /// Create with per-axis noise.
        ///
        /// Args:
        ///     sigma_x (float): X-axis position noise (meters).
        ///     sigma_y (float): Y-axis position noise (meters).
        ///     sigma_z (float): Z-axis position noise (meters).
        ///
        /// Returns:
        ///     InertialPositionMeasurementModel: New model instance.
        #[staticmethod]
        fn per_axis(sigma_x: f64, sigma_y: f64, sigma_z: f64) -> Self {
            PyInertialPositionMeasurementModel {
                model: estimation::InertialPositionMeasurementModel::new_per_axis(sigma_x, sigma_y, sigma_z),
            }
        }

        /// Create from a full 3x3 noise covariance matrix.
        ///
        /// Allows specifying correlated measurement noise (off-diagonal terms).
        ///
        /// Args:
        ///     noise_cov (numpy.ndarray): 3x3 noise covariance matrix (meters²).
        ///
        /// Returns:
        ///     InertialPositionMeasurementModel: New model instance.
        ///
        /// Example:
        ///     ```python
        ///     import brahe as bh
        ///     import numpy as np
        ///     cov = np.diag([100.0, 225.0, 400.0])
        ///     model = bh.InertialPositionMeasurementModel.from_covariance(cov)
        ///     ```
        #[staticmethod]
        fn from_covariance(noise_cov: PyReadonlyArray2<f64>) -> PyResult<Self> {
            let shape = noise_cov.shape();
            let data = noise_cov.as_slice().map_err(|e| {
                exceptions::PyValueError::new_err(format!("Failed to read array: {}", e))
            })?;
            let mat = nalgebra::DMatrix::from_row_slice(shape[0], shape[1], data);
            let model = estimation::InertialPositionMeasurementModel::from_covariance(mat)
                .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyInertialPositionMeasurementModel { model })
        }

        /// Create from upper-triangular covariance elements.
        ///
        /// Elements in row-major packed order: [c00, c01, c02, c11, c12, c22].
        ///
        /// Args:
        ///     upper (numpy.ndarray): Upper-triangular elements (6 for 3x3).
        ///
        /// Returns:
        ///     InertialPositionMeasurementModel: New model instance.
        #[staticmethod]
        fn from_upper_triangular(upper: PyReadonlyArray1<f64>) -> PyResult<Self> {
            let data = upper.as_slice().map_err(|e| {
                exceptions::PyValueError::new_err(format!("Failed to read array: {}", e))
            })?;
            let model = estimation::InertialPositionMeasurementModel::from_upper_triangular(data)
                .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyInertialPositionMeasurementModel { model })
        }
    ],
    doc: "Inertial position measurement model (3D ECI position).\n\nDirectly observes [x, y, z] from the state vector with Gaussian noise."
);

impl_measurement_model_binding!(
    PyInertialVelocityMeasurementModel,
    estimation::InertialVelocityMeasurementModel,
    "InertialVelocityMeasurementModel",
    constructors: [
        /// Create an inertial velocity measurement model.
        ///
        /// Args:
        ///     sigma (float): Velocity noise standard deviation (m/s).
        ///
        /// Returns:
        ///     InertialVelocityMeasurementModel: New model instance.
        ///
        /// Example:
        ///     ```python
        ///     import brahe as bh
        ///     model = bh.InertialVelocityMeasurementModel(0.1)
        ///     ```
        #[new]
        fn new(sigma: f64) -> Self {
            PyInertialVelocityMeasurementModel {
                model: estimation::InertialVelocityMeasurementModel::new(sigma),
            }
        }

        /// Create with per-axis noise.
        ///
        /// Args:
        ///     sigma_x (float): X-axis velocity noise (m/s).
        ///     sigma_y (float): Y-axis velocity noise (m/s).
        ///     sigma_z (float): Z-axis velocity noise (m/s).
        ///
        /// Returns:
        ///     InertialVelocityMeasurementModel: New model instance.
        #[staticmethod]
        fn per_axis(sigma_x: f64, sigma_y: f64, sigma_z: f64) -> Self {
            PyInertialVelocityMeasurementModel {
                model: estimation::InertialVelocityMeasurementModel::new_per_axis(sigma_x, sigma_y, sigma_z),
            }
        }

        /// Create from a full 3x3 noise covariance matrix.
        ///
        /// Args:
        ///     noise_cov (numpy.ndarray): 3x3 noise covariance matrix ((m/s)²).
        ///
        /// Returns:
        ///     InertialVelocityMeasurementModel: New model instance.
        #[staticmethod]
        fn from_covariance(noise_cov: PyReadonlyArray2<f64>) -> PyResult<Self> {
            let shape = noise_cov.shape();
            let data = noise_cov.as_slice().map_err(|e| {
                exceptions::PyValueError::new_err(format!("Failed to read array: {}", e))
            })?;
            let mat = nalgebra::DMatrix::from_row_slice(shape[0], shape[1], data);
            let model = estimation::InertialVelocityMeasurementModel::from_covariance(mat)
                .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyInertialVelocityMeasurementModel { model })
        }

        /// Create from upper-triangular covariance elements.
        ///
        /// Args:
        ///     upper (numpy.ndarray): Upper-triangular elements (6 for 3x3).
        ///
        /// Returns:
        ///     InertialVelocityMeasurementModel: New model instance.
        #[staticmethod]
        fn from_upper_triangular(upper: PyReadonlyArray1<f64>) -> PyResult<Self> {
            let data = upper.as_slice().map_err(|e| {
                exceptions::PyValueError::new_err(format!("Failed to read array: {}", e))
            })?;
            let model = estimation::InertialVelocityMeasurementModel::from_upper_triangular(data)
                .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyInertialVelocityMeasurementModel { model })
        }
    ],
    doc: "Inertial velocity measurement model (3D ECI velocity).\n\nDirectly observes [vx, vy, vz] from the state vector with Gaussian noise."
);

impl_measurement_model_binding!(
    PyInertialStateMeasurementModel,
    estimation::InertialStateMeasurementModel,
    "InertialStateMeasurementModel",
    constructors: [
        /// Create an inertial state measurement model.
        ///
        /// Args:
        ///     pos_sigma (float): Position noise standard deviation (meters).
        ///     vel_sigma (float): Velocity noise standard deviation (m/s).
        ///
        /// Returns:
        ///     InertialStateMeasurementModel: New model instance.
        ///
        /// Example:
        ///     ```python
        ///     import brahe as bh
        ///     model = bh.InertialStateMeasurementModel(10.0, 0.1)
        ///     ```
        #[new]
        fn new(pos_sigma: f64, vel_sigma: f64) -> Self {
            PyInertialStateMeasurementModel {
                model: estimation::InertialStateMeasurementModel::new(pos_sigma, vel_sigma),
            }
        }

        /// Create with per-axis noise.
        ///
        /// Args:
        ///     pos_sigma_x (float): X position noise (meters).
        ///     pos_sigma_y (float): Y position noise (meters).
        ///     pos_sigma_z (float): Z position noise (meters).
        ///     vel_sigma_x (float): X velocity noise (m/s).
        ///     vel_sigma_y (float): Y velocity noise (m/s).
        ///     vel_sigma_z (float): Z velocity noise (m/s).
        ///
        /// Returns:
        ///     InertialStateMeasurementModel: New model instance.
        #[staticmethod]
        fn per_axis(
            pos_sigma_x: f64, pos_sigma_y: f64, pos_sigma_z: f64,
            vel_sigma_x: f64, vel_sigma_y: f64, vel_sigma_z: f64,
        ) -> Self {
            PyInertialStateMeasurementModel {
                model: estimation::InertialStateMeasurementModel::new_per_axis(
                    pos_sigma_x, pos_sigma_y, pos_sigma_z,
                    vel_sigma_x, vel_sigma_y, vel_sigma_z,
                ),
            }
        }

        /// Create from a full 6x6 noise covariance matrix.
        ///
        /// Allows specifying correlated noise including position-velocity
        /// cross-correlations.
        ///
        /// Args:
        ///     noise_cov (numpy.ndarray): 6x6 noise covariance matrix.
        ///
        /// Returns:
        ///     InertialStateMeasurementModel: New model instance.
        #[staticmethod]
        fn from_covariance(noise_cov: PyReadonlyArray2<f64>) -> PyResult<Self> {
            let shape = noise_cov.shape();
            let data = noise_cov.as_slice().map_err(|e| {
                exceptions::PyValueError::new_err(format!("Failed to read array: {}", e))
            })?;
            let mat = nalgebra::DMatrix::from_row_slice(shape[0], shape[1], data);
            let model = estimation::InertialStateMeasurementModel::from_covariance(mat)
                .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyInertialStateMeasurementModel { model })
        }

        /// Create from upper-triangular covariance elements.
        ///
        /// Args:
        ///     upper (numpy.ndarray): Upper-triangular elements (21 for 6x6).
        ///
        /// Returns:
        ///     InertialStateMeasurementModel: New model instance.
        #[staticmethod]
        fn from_upper_triangular(upper: PyReadonlyArray1<f64>) -> PyResult<Self> {
            let data = upper.as_slice().map_err(|e| {
                exceptions::PyValueError::new_err(format!("Failed to read array: {}", e))
            })?;
            let model = estimation::InertialStateMeasurementModel::from_upper_triangular(data)
                .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyInertialStateMeasurementModel { model })
        }
    ],
    doc: "Inertial state measurement model (6D ECI state).\n\nDirectly observes [x, y, z, vx, vy, vz] from the state vector with Gaussian noise."
);

impl_measurement_model_binding!(
    PyEcefPositionMeasurementModel,
    estimation::EcefPositionMeasurementModel,
    "ECEFPositionMeasurementModel",
    constructors: [
        /// Create an ECEF position measurement model.
        ///
        /// Args:
        ///     sigma (float): Position noise standard deviation (meters).
        ///
        /// Returns:
        ///     EcefPositionMeasurementModel: New model instance.
        ///
        /// Example:
        ///     ```python
        ///     import brahe as bh
        ///     model = bh.EcefPositionMeasurementModel(5.0)
        ///     ```
        #[new]
        fn new(sigma: f64) -> Self {
            PyEcefPositionMeasurementModel {
                model: estimation::EcefPositionMeasurementModel::new(sigma),
            }
        }

        /// Create with per-axis noise.
        ///
        /// Args:
        ///     sigma_x (float): X-axis position noise (meters).
        ///     sigma_y (float): Y-axis position noise (meters).
        ///     sigma_z (float): Z-axis position noise (meters).
        ///
        /// Returns:
        ///     EcefPositionMeasurementModel: New model instance.
        #[staticmethod]
        fn per_axis(sigma_x: f64, sigma_y: f64, sigma_z: f64) -> Self {
            PyEcefPositionMeasurementModel {
                model: estimation::EcefPositionMeasurementModel::new_per_axis(sigma_x, sigma_y, sigma_z),
            }
        }

        /// Create from a full 3x3 noise covariance matrix.
        ///
        /// Args:
        ///     noise_cov (numpy.ndarray): 3x3 noise covariance matrix (meters²).
        ///
        /// Returns:
        ///     ECEFPositionMeasurementModel: New model instance.
        #[staticmethod]
        fn from_covariance(noise_cov: PyReadonlyArray2<f64>) -> PyResult<Self> {
            let shape = noise_cov.shape();
            let data = noise_cov.as_slice().map_err(|e| {
                exceptions::PyValueError::new_err(format!("Failed to read array: {}", e))
            })?;
            let mat = nalgebra::DMatrix::from_row_slice(shape[0], shape[1], data);
            let model = estimation::EcefPositionMeasurementModel::from_covariance(mat)
                .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyEcefPositionMeasurementModel { model })
        }

        /// Create from upper-triangular covariance elements.
        ///
        /// Args:
        ///     upper (numpy.ndarray): Upper-triangular elements (6 for 3x3).
        ///
        /// Returns:
        ///     ECEFPositionMeasurementModel: New model instance.
        #[staticmethod]
        fn from_upper_triangular(upper: PyReadonlyArray1<f64>) -> PyResult<Self> {
            let data = upper.as_slice().map_err(|e| {
                exceptions::PyValueError::new_err(format!("Failed to read array: {}", e))
            })?;
            let model = estimation::EcefPositionMeasurementModel::from_upper_triangular(data)
                .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyEcefPositionMeasurementModel { model })
        }
    ],
    doc: "ECEF position measurement model (3D ECEF position from GNSS).\n\nInternally converts ECI state to ECEF position."
);

impl_measurement_model_binding!(
    PyEcefVelocityMeasurementModel,
    estimation::EcefVelocityMeasurementModel,
    "ECEFVelocityMeasurementModel",
    constructors: [
        /// Create an ECEF velocity measurement model.
        ///
        /// Args:
        ///     sigma (float): Velocity noise standard deviation (m/s).
        ///
        /// Returns:
        ///     EcefVelocityMeasurementModel: New model instance.
        ///
        /// Example:
        ///     ```python
        ///     import brahe as bh
        ///     model = bh.EcefVelocityMeasurementModel(0.05)
        ///     ```
        #[new]
        fn new(sigma: f64) -> Self {
            PyEcefVelocityMeasurementModel {
                model: estimation::EcefVelocityMeasurementModel::new(sigma),
            }
        }

        /// Create with per-axis noise.
        ///
        /// Args:
        ///     sigma_x (float): X-axis velocity noise (m/s).
        ///     sigma_y (float): Y-axis velocity noise (m/s).
        ///     sigma_z (float): Z-axis velocity noise (m/s).
        ///
        /// Returns:
        ///     EcefVelocityMeasurementModel: New model instance.
        #[staticmethod]
        fn per_axis(sigma_x: f64, sigma_y: f64, sigma_z: f64) -> Self {
            PyEcefVelocityMeasurementModel {
                model: estimation::EcefVelocityMeasurementModel::new_per_axis(sigma_x, sigma_y, sigma_z),
            }
        }

        /// Create from a full 3x3 noise covariance matrix.
        ///
        /// Args:
        ///     noise_cov (numpy.ndarray): 3x3 noise covariance matrix ((m/s)²).
        ///
        /// Returns:
        ///     ECEFVelocityMeasurementModel: New model instance.
        #[staticmethod]
        fn from_covariance(noise_cov: PyReadonlyArray2<f64>) -> PyResult<Self> {
            let shape = noise_cov.shape();
            let data = noise_cov.as_slice().map_err(|e| {
                exceptions::PyValueError::new_err(format!("Failed to read array: {}", e))
            })?;
            let mat = nalgebra::DMatrix::from_row_slice(shape[0], shape[1], data);
            let model = estimation::EcefVelocityMeasurementModel::from_covariance(mat)
                .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyEcefVelocityMeasurementModel { model })
        }

        /// Create from upper-triangular covariance elements.
        ///
        /// Args:
        ///     upper (numpy.ndarray): Upper-triangular elements (6 for 3x3).
        ///
        /// Returns:
        ///     ECEFVelocityMeasurementModel: New model instance.
        #[staticmethod]
        fn from_upper_triangular(upper: PyReadonlyArray1<f64>) -> PyResult<Self> {
            let data = upper.as_slice().map_err(|e| {
                exceptions::PyValueError::new_err(format!("Failed to read array: {}", e))
            })?;
            let model = estimation::EcefVelocityMeasurementModel::from_upper_triangular(data)
                .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyEcefVelocityMeasurementModel { model })
        }
    ],
    doc: "ECEF velocity measurement model (3D ECEF velocity from GNSS).\n\nInternally converts ECI state to ECEF velocity."
);

impl_measurement_model_binding!(
    PyEcefStateMeasurementModel,
    estimation::EcefStateMeasurementModel,
    "ECEFStateMeasurementModel",
    constructors: [
        /// Create an ECEF state measurement model.
        ///
        /// Args:
        ///     pos_sigma (float): Position noise standard deviation (meters).
        ///     vel_sigma (float): Velocity noise standard deviation (m/s).
        ///
        /// Returns:
        ///     EcefStateMeasurementModel: New model instance.
        ///
        /// Example:
        ///     ```python
        ///     import brahe as bh
        ///     model = bh.EcefStateMeasurementModel(5.0, 0.05)
        ///     ```
        #[new]
        fn new(pos_sigma: f64, vel_sigma: f64) -> Self {
            PyEcefStateMeasurementModel {
                model: estimation::EcefStateMeasurementModel::new(pos_sigma, vel_sigma),
            }
        }

        /// Create with per-axis noise.
        ///
        /// Args:
        ///     pos_sigma_x (float): X position noise (meters).
        ///     pos_sigma_y (float): Y position noise (meters).
        ///     pos_sigma_z (float): Z position noise (meters).
        ///     vel_sigma_x (float): X velocity noise (m/s).
        ///     vel_sigma_y (float): Y velocity noise (m/s).
        ///     vel_sigma_z (float): Z velocity noise (m/s).
        ///
        /// Returns:
        ///     EcefStateMeasurementModel: New model instance.
        #[staticmethod]
        fn per_axis(
            pos_sigma_x: f64, pos_sigma_y: f64, pos_sigma_z: f64,
            vel_sigma_x: f64, vel_sigma_y: f64, vel_sigma_z: f64,
        ) -> Self {
            PyEcefStateMeasurementModel {
                model: estimation::EcefStateMeasurementModel::new_per_axis(
                    pos_sigma_x, pos_sigma_y, pos_sigma_z,
                    vel_sigma_x, vel_sigma_y, vel_sigma_z,
                ),
            }
        }

        /// Create from a full 6x6 noise covariance matrix.
        ///
        /// Args:
        ///     noise_cov (numpy.ndarray): 6x6 noise covariance matrix.
        ///
        /// Returns:
        ///     ECEFStateMeasurementModel: New model instance.
        #[staticmethod]
        fn from_covariance(noise_cov: PyReadonlyArray2<f64>) -> PyResult<Self> {
            let shape = noise_cov.shape();
            let data = noise_cov.as_slice().map_err(|e| {
                exceptions::PyValueError::new_err(format!("Failed to read array: {}", e))
            })?;
            let mat = nalgebra::DMatrix::from_row_slice(shape[0], shape[1], data);
            let model = estimation::EcefStateMeasurementModel::from_covariance(mat)
                .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyEcefStateMeasurementModel { model })
        }

        /// Create from upper-triangular covariance elements.
        ///
        /// Args:
        ///     upper (numpy.ndarray): Upper-triangular elements (21 for 6x6).
        ///
        /// Returns:
        ///     ECEFStateMeasurementModel: New model instance.
        #[staticmethod]
        fn from_upper_triangular(upper: PyReadonlyArray1<f64>) -> PyResult<Self> {
            let data = upper.as_slice().map_err(|e| {
                exceptions::PyValueError::new_err(format!("Failed to read array: {}", e))
            })?;
            let model = estimation::EcefStateMeasurementModel::from_upper_triangular(data)
                .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(PyEcefStateMeasurementModel { model })
        }
    ],
    doc: "ECEF state measurement model (6D ECEF state from GNSS).\n\nInternally converts ECI state to ECEF state."
);

// =============================================================================
// Covariance Matrix Helper Functions
// =============================================================================

/// Create an isotropic covariance matrix: sigma^2 * I.
///
/// Builds a dim x dim diagonal matrix where every diagonal element is sigma^2.
///
/// Args:
///     dim (int): Matrix dimension.
///     sigma (float): Standard deviation applied to all axes.
///
/// Returns:
///     numpy.ndarray: dim x dim diagonal covariance matrix.
///
/// Example:
///     ```python
///     import brahe as bh
///     r = bh.isotropic_covariance(3, 10.0)
///     # r = diag([100, 100, 100])
///     ```
#[pyfunction]
#[pyo3(name = "isotropic_covariance")]
fn py_isotropic_covariance<'py>(
    py: Python<'py>,
    dim: usize,
    sigma: f64,
) -> Bound<'py, PyArray<f64, numpy::Ix2>> {
    let mat = crate::math::covariance::isotropic_covariance(dim, sigma);
    let rows = mat.nrows();
    let cols = mat.ncols();
    let mut flat = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            flat.push(mat[(i, j)]);
        }
    }
    flat.into_pyarray(py).reshape([rows, cols]).unwrap()
}

/// Create a diagonal covariance matrix from per-axis standard deviations.
///
/// Each sigma value is squared to produce the corresponding diagonal element.
///
/// Args:
///     sigmas (numpy.ndarray): Array of standard deviations, one per axis.
///
/// Returns:
///     numpy.ndarray: n x n diagonal covariance matrix.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///     r = bh.diagonal_covariance(np.array([5.0, 10.0, 15.0]))
///     # r = diag([25, 100, 225])
///     ```
#[pyfunction]
#[pyo3(name = "diagonal_covariance")]
fn py_diagonal_covariance<'py>(
    py: Python<'py>,
    sigmas: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray<f64, numpy::Ix2>>> {
    let data = sigmas.as_slice().map_err(|e| {
        exceptions::PyValueError::new_err(format!("Failed to read array: {}", e))
    })?;
    let mat = crate::math::covariance::diagonal_covariance(data);
    let rows = mat.nrows();
    let cols = mat.ncols();
    let mut flat = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            flat.push(mat[(i, j)]);
        }
    }
    Ok(flat.into_pyarray(py).reshape([rows, cols]).unwrap())
}

// =============================================================================
// Custom Measurement Model Support (Python subclassing)
// =============================================================================

/// Base class for Python-defined measurement models.
///
/// Subclass this to define custom measurement models that work with the EKF.
/// You must implement ``predict()``, ``noise_covariance()``, ``measurement_dim()``,
/// and ``name()``. Override ``jacobian()`` to provide an analytical Jacobian;
/// return ``None`` to use automatic finite-difference computation.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     class RangeModel(bh.MeasurementModel):
///         def __init__(self, station_ecef, sigma):
///             super().__init__()
///             self.station_ecef = np.array(station_ecef)
///             self.sigma = sigma
///
///         def predict(self, epoch, state):
///             pos = state[:3]
///             return np.array([np.linalg.norm(pos - self.station_ecef)])
///
///         def noise_covariance(self):
///             return np.array([[self.sigma**2]])
///
///         def measurement_dim(self):
///             return 1
///
///         def name(self):
///             return "Range"
///
///         # jacobian() not overridden - uses automatic finite differences
///     ```
#[pyclass(module = "brahe._brahe", subclass)]
#[pyo3(name = "MeasurementModel")]
pub struct PyMeasurementModel {}

#[pymethods]
impl PyMeasurementModel {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> Self {
        PyMeasurementModel {}
    }

    /// Compute predicted measurement from state.
    ///
    /// Override this method in your subclass.
    ///
    /// Args:
    ///     epoch (Epoch): Current epoch.
    ///     state (numpy.ndarray): Current state vector.
    ///
    /// Returns:
    ///     numpy.ndarray: Predicted measurement vector.
    #[allow(unused_variables)]
    fn predict(
        &self,
        epoch: &PyEpoch,
        state: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyAny>> {
        Err(exceptions::PyNotImplementedError::new_err(
            "Subclasses must implement predict()",
        ))
    }

    /// Compute measurement Jacobian H = dh/dx.
    ///
    /// Override this method to provide an analytical Jacobian.
    /// Return None to use automatic finite-difference computation.
    ///
    /// Args:
    ///     epoch (Epoch): Current epoch.
    ///     state (numpy.ndarray): Current state vector.
    ///
    /// Returns:
    ///     numpy.ndarray or None: Jacobian matrix, or None for finite-diff.
    #[allow(unused_variables)]
    fn jacobian(
        &self,
        epoch: &PyEpoch,
        state: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyAny>> {
        // Default: return None to signal finite-diff fallback
        Python::attach(|py| Ok(py.None()))
    }

    /// Get measurement noise covariance R.
    ///
    /// Override this method in your subclass.
    ///
    /// Returns:
    ///     numpy.ndarray: Noise covariance matrix (m x m).
    fn noise_covariance(&self) -> PyResult<Py<PyAny>> {
        Err(exceptions::PyNotImplementedError::new_err(
            "Subclasses must implement noise_covariance()",
        ))
    }

    /// Get measurement vector dimension.
    ///
    /// Override this method in your subclass.
    ///
    /// Returns:
    ///     int: Measurement dimension.
    fn measurement_dim(&self) -> PyResult<usize> {
        Err(exceptions::PyNotImplementedError::new_err(
            "Subclasses must implement measurement_dim()",
        ))
    }

    /// Get human-readable model name.
    ///
    /// Override this method in your subclass.
    ///
    /// Returns:
    ///     str: Model name.
    fn name(&self) -> PyResult<String> {
        Err(exceptions::PyNotImplementedError::new_err(
            "Subclasses must implement name()",
        ))
    }
}

// Internal wrapper that implements the Rust MeasurementModel trait
// by calling Python methods via GIL acquisition.
pub(crate) struct RustMeasurementModelWrapper {
    py_model: Py<PyAny>,
    cached_name: String,
    cached_dim: usize,
    cached_noise_cov: DMatrix<f64>,
}

// Py<PyAny> is Send + Sync in PyO3, satisfying the trait bound
unsafe impl Send for RustMeasurementModelWrapper {}
unsafe impl Sync for RustMeasurementModelWrapper {}

impl RustMeasurementModelWrapper {
    pub fn new(py: Python<'_>, py_model: Py<PyAny>) -> PyResult<Self> {
        let obj = py_model.bind(py);

        // Cache static properties at construction
        let cached_name: String = obj
            .call_method0("name")
            .and_then(|r| r.extract())
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!(
                    "MeasurementModel.name() failed: {}",
                    e
                ))
            })?;

        let cached_dim: usize = obj
            .call_method0("measurement_dim")
            .and_then(|r| r.extract())
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!(
                    "MeasurementModel.measurement_dim() failed: {}",
                    e
                ))
            })?;

        let noise_cov_result = obj.call_method0("noise_covariance").map_err(|e| {
            exceptions::PyValueError::new_err(format!(
                "MeasurementModel.noise_covariance() failed: {}",
                e
            ))
        })?;

        let noise_arr: PyReadonlyArray2<f64> = noise_cov_result.extract().map_err(|e| {
            exceptions::PyValueError::new_err(format!(
                "noise_covariance() must return a 2D numpy array: {}",
                e
            ))
        })?;
        let shape = noise_arr.shape();
        let data: Vec<f64> = noise_arr.as_slice()
            .map_err(|e| exceptions::PyValueError::new_err(format!("Failed to read noise_covariance: {}", e)))?
            .to_vec();
        let cached_noise_cov = DMatrix::from_row_slice(shape[0], shape[1], &data);

        Ok(Self {
            py_model,
            cached_name,
            cached_dim,
            cached_noise_cov,
        })
    }
}

impl MeasurementModel for RustMeasurementModelWrapper {
    fn predict(
        &self,
        epoch: &crate::time::Epoch,
        state: &DVector<f64>,
        _params: Option<&DVector<f64>>,
    ) -> Result<DVector<f64>, crate::utils::errors::BraheError> {
        Python::attach(|py| {
            let py_epoch = Py::new(py, PyEpoch { obj: *epoch })
                .map_err(|e| crate::utils::errors::BraheError::Error(e.to_string()))?;
            let state_np = state.as_slice().to_pyarray(py);

            let result = self
                .py_model
                .bind(py)
                .call_method1("predict", (py_epoch, state_np))
                .map_err(|e| {
                    crate::utils::errors::BraheError::Error(format!(
                        "Python predict() failed: {}",
                        e
                    ))
                })?;

            let res_arr: PyReadonlyArray1<f64> = result.extract().map_err(|e| {
                crate::utils::errors::BraheError::Error(format!(
                    "predict() must return a numpy array: {}",
                    e
                ))
            })?;

            Ok(DVector::from_column_slice(
                res_arr
                    .as_slice()
                    .map_err(|e| crate::utils::errors::BraheError::Error(e.to_string()))?,
            ))
        })
    }

    fn jacobian(
        &self,
        epoch: &crate::time::Epoch,
        state: &DVector<f64>,
        params: Option<&DVector<f64>>,
    ) -> Result<DMatrix<f64>, crate::utils::errors::BraheError> {
        // Try calling Python jacobian() first
        let py_result: Result<Option<DMatrix<f64>>, crate::utils::errors::BraheError> =
            Python::attach(|py| {
                let py_epoch = Py::new(py, PyEpoch { obj: *epoch }).map_err(|e| {
                    crate::utils::errors::BraheError::Error(format!(
                        "Failed to create PyEpoch: {}",
                        e
                    ))
                })?;
                let state_np = state.as_slice().to_pyarray(py);

                let result = self
                    .py_model
                    .bind(py)
                    .call_method1("jacobian", (py_epoch, state_np))
                    .map_err(|e| {
                        crate::utils::errors::BraheError::Error(format!(
                            "Python jacobian() raised an exception: {}",
                            e
                        ))
                    })?;

                // If the Python method returned None, use finite-diff fallback
                if result.is_none() {
                    return Ok(None);
                }

                // Extract as 2D numpy array
                let arr: PyReadonlyArray2<f64> = result.extract().map_err(|e| {
                    crate::utils::errors::BraheError::Error(format!(
                        "jacobian() must return a 2D numpy array or None: {}",
                        e
                    ))
                })?;
                let shape = arr.shape();
                let data: Vec<f64> = arr
                    .as_slice()
                    .map_err(|e| {
                        crate::utils::errors::BraheError::Error(format!(
                            "Failed to read jacobian array data: {}",
                            e
                        ))
                    })?
                    .to_vec();
                Ok(Some(DMatrix::from_row_slice(shape[0], shape[1], &data)))
            });

        match py_result {
            Ok(Some(jacobian)) => Ok(jacobian),
            Ok(None) => {
                // Fall back to numerical Jacobian using the math::jacobian infrastructure
                estimation::measurement_jacobian_numerical(
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
            Err(e) => Err(e),
        }
    }

    fn noise_covariance(&self) -> DMatrix<f64> {
        self.cached_noise_cov.clone()
    }

    fn measurement_dim(&self) -> usize {
        self.cached_dim
    }

    fn name(&self) -> &str {
        &self.cached_name
    }
}

// =============================================================================
// MeasurementModelHolder enum (dispatch between Rust native and Python wrapper)
// =============================================================================

enum MeasurementModelHolder {
    RustNative(Box<dyn MeasurementModel>),
    PythonWrapper(RustMeasurementModelWrapper),
}

impl MeasurementModel for MeasurementModelHolder {
    fn predict(
        &self,
        epoch: &crate::time::Epoch,
        state: &DVector<f64>,
        params: Option<&DVector<f64>>,
    ) -> Result<DVector<f64>, crate::utils::errors::BraheError> {
        match self {
            MeasurementModelHolder::RustNative(m) => m.predict(epoch, state, params),
            MeasurementModelHolder::PythonWrapper(m) => m.predict(epoch, state, params),
        }
    }

    fn jacobian(
        &self,
        epoch: &crate::time::Epoch,
        state: &DVector<f64>,
        params: Option<&DVector<f64>>,
    ) -> Result<DMatrix<f64>, crate::utils::errors::BraheError> {
        match self {
            MeasurementModelHolder::RustNative(m) => m.jacobian(epoch, state, params),
            MeasurementModelHolder::PythonWrapper(m) => m.jacobian(epoch, state, params),
        }
    }

    fn noise_covariance(&self) -> DMatrix<f64> {
        match self {
            MeasurementModelHolder::RustNative(m) => m.noise_covariance(),
            MeasurementModelHolder::PythonWrapper(m) => m.noise_covariance(),
        }
    }

    fn measurement_dim(&self) -> usize {
        match self {
            MeasurementModelHolder::RustNative(m) => m.measurement_dim(),
            MeasurementModelHolder::PythonWrapper(m) => m.measurement_dim(),
        }
    }

    fn name(&self) -> &str {
        match self {
            MeasurementModelHolder::RustNative(m) => m.name(),
            MeasurementModelHolder::PythonWrapper(m) => m.name(),
        }
    }
}

/// Dispatch measurement models: try extracting as built-in types first,
/// fall back to Python wrapper for custom models.
fn process_measurement_models(
    py: Python<'_>,
    models: Vec<Py<PyAny>>,
) -> PyResult<Vec<Box<dyn MeasurementModel>>> {
    let mut result: Vec<Box<dyn MeasurementModel>> = Vec::with_capacity(models.len());

    for py_model in models {
        let obj = py_model.bind(py);

        // Try each built-in type first (pure Rust execution, no Python overhead)
        if let Ok(m) = obj.extract::<PyRef<PyInertialPositionMeasurementModel>>() {
            result.push(Box::new(MeasurementModelHolder::RustNative(Box::new(
                m.model.clone(),
            ))));
        } else if let Ok(m) = obj.extract::<PyRef<PyInertialVelocityMeasurementModel>>() {
            result.push(Box::new(MeasurementModelHolder::RustNative(Box::new(
                m.model.clone(),
            ))));
        } else if let Ok(m) = obj.extract::<PyRef<PyInertialStateMeasurementModel>>() {
            result.push(Box::new(MeasurementModelHolder::RustNative(Box::new(
                m.model.clone(),
            ))));
        } else if let Ok(m) = obj.extract::<PyRef<PyEcefPositionMeasurementModel>>() {
            result.push(Box::new(MeasurementModelHolder::RustNative(Box::new(
                m.model.clone(),
            ))));
        } else if let Ok(m) = obj.extract::<PyRef<PyEcefVelocityMeasurementModel>>() {
            result.push(Box::new(MeasurementModelHolder::RustNative(Box::new(
                m.model.clone(),
            ))));
        } else if let Ok(m) = obj.extract::<PyRef<PyEcefStateMeasurementModel>>() {
            result.push(Box::new(MeasurementModelHolder::RustNative(Box::new(
                m.model.clone(),
            ))));
        } else {
            // Custom Python measurement model
            let wrapper = RustMeasurementModelWrapper::new(py, py_model)?;
            result.push(Box::new(MeasurementModelHolder::PythonWrapper(wrapper)));
        }
    }

    Ok(result)
}

// =============================================================================
// ExtendedKalmanFilter
// =============================================================================

/// Extended Kalman Filter for sequential state estimation.
///
/// Processes observations one at a time, propagating state and covariance
/// between observation epochs using a numerical propagator. Supports both
/// built-in and custom Python measurement models.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
///
///     epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
///     state = np.array([6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0])
///     p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
///
///     ekf = bh.ExtendedKalmanFilter(
///         epoch, state, p0,
///         propagation_config=bh.NumericalPropagationConfig.default(),
///         force_config=bh.ForceModelConfig.two_body_gravity(),
///         measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "ExtendedKalmanFilter")]
pub struct PyExtendedKalmanFilter {
    ekf: estimation::ExtendedKalmanFilter,
}

#[pymethods]
impl PyExtendedKalmanFilter {
    /// Create a new ExtendedKalmanFilter.
    ///
    /// Args:
    ///     epoch (Epoch): Initial epoch.
    ///     state (numpy.ndarray): Initial state vector in ECI [x,y,z,vx,vy,vz,...] (meters, m/s).
    ///     initial_covariance (numpy.ndarray): Initial covariance matrix (n x n).
    ///     propagation_config (NumericalPropagationConfig): Propagation configuration.
    ///     force_config (ForceModelConfig): Force model configuration.
    ///     measurement_models (list): List of measurement models (built-in or custom).
    ///     config (EKFConfig or None): EKF configuration. Defaults to EKFConfig.default().
    ///     params (numpy.ndarray or None): Parameter vector for force models.
    ///     additional_dynamics (callable or None): Additional dynamics function f(t, state, params) -> derivative.
    ///     control_input (callable or None): Control input function f(t, state, params) -> acceleration.
    ///
    /// Returns:
    ///     ExtendedKalmanFilter: New EKF instance.
    #[new]
    #[pyo3(signature = (
        epoch, state, initial_covariance,
        propagation_config, force_config,
        measurement_models,
        config=None, params=None, additional_dynamics=None, control_input=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        epoch: &PyEpoch,
        state: PyReadonlyArray1<f64>,
        initial_covariance: PyReadonlyArray2<f64>,
        propagation_config: &PyNumericalPropagationConfig,
        force_config: &PyForceModelConfig,
        measurement_models: Vec<Py<PyAny>>,
        config: Option<&PyEKFConfig>,
        params: Option<PyReadonlyArray1<f64>>,
        additional_dynamics: Option<Py<PyAny>>,
        control_input: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let state_vec = DVector::from_column_slice(state.as_slice()?);
        let state_dim = state_vec.len();

        // Parse covariance
        let cov_shape = initial_covariance.shape();
        if cov_shape[0] != state_dim || cov_shape[1] != state_dim {
            return Err(exceptions::PyValueError::new_err(format!(
                "initial_covariance must be {}x{}, got {}x{}",
                state_dim, state_dim, cov_shape[0], cov_shape[1]
            )));
        }
        let cov_data: Vec<f64> = initial_covariance.as_slice()?.to_vec();
        let cov_matrix = DMatrix::from_row_slice(state_dim, state_dim, &cov_data);

        // Clone propagation config and force STM enabled
        let mut prop_config = propagation_config.config.clone();
        prop_config.variational.enable_stm = true;

        let params_vec =
            params.map(|p| DVector::from_column_slice(p.as_slice().unwrap()));

        // Wrap additional_dynamics callable
        let additional_dynamics_fn: Option<crate::integrators::traits::DStateDynamics> =
            additional_dynamics.map(|dyn_py| {
                let dyn_py = dyn_py.clone_ref(py);
                Box::new(
                    move |t: f64,
                          x: &DVector<f64>,
                          p: Option<&DVector<f64>>|
                          -> DVector<f64> {
                        Python::attach(|py| {
                            let x_np = x.as_slice().to_pyarray(py);
                            let p_np: Option<Bound<'_, PyArray<f64, Ix1>>> =
                                p.map(|pv| pv.as_slice().to_pyarray(py).to_owned());

                            let result = match p_np {
                                Some(params_arr) => dyn_py.call1(py, (t, x_np, params_arr)),
                                None => dyn_py.call1(py, (t, x_np, py.None())),
                            };

                            match result {
                                Ok(res) => {
                                    let res_arr: PyReadonlyArray1<f64> =
                                        res.extract(py).unwrap();
                                    DVector::from_column_slice(
                                        res_arr.as_slice().unwrap(),
                                    )
                                }
                                Err(e) => {
                                    panic!("Error calling additional_dynamics: {e}")
                                }
                            }
                        })
                    },
                ) as crate::integrators::traits::DStateDynamics
            });

        // Wrap control_input callable
        let control_input_fn: crate::integrators::traits::DControlInput =
            control_input.map(|ctrl_py| {
                let ctrl_py = ctrl_py.clone_ref(py);
                Box::new(
                    move |t: f64,
                          x: &DVector<f64>,
                          p: Option<&DVector<f64>>|
                          -> DVector<f64> {
                        Python::attach(|py| {
                            let x_np = x.as_slice().to_pyarray(py);
                            let p_np: Option<Bound<'_, PyArray<f64, Ix1>>> =
                                p.map(|pv| pv.as_slice().to_pyarray(py).to_owned());

                            let result = match p_np {
                                Some(params_arr) => ctrl_py.call1(py, (t, x_np, params_arr)),
                                None => ctrl_py.call1(py, (t, x_np, py.None())),
                            };

                            match result {
                                Ok(res) => {
                                    let res_arr: PyReadonlyArray1<f64> =
                                        res.extract(py).unwrap();
                                    DVector::from_column_slice(
                                        res_arr.as_slice().unwrap(),
                                    )
                                }
                                Err(e) => {
                                    panic!("Error calling control_input: {e}")
                                }
                            }
                        })
                    },
                )
                    as Box<
                        dyn Fn(f64, &DVector<f64>, Option<&DVector<f64>>) -> DVector<f64>
                            + Send
                            + Sync,
                    >
            });

        // Build the orbit propagator internally
        let prop = propagators::DNumericalOrbitPropagator::new(
            epoch.obj,
            state_vec,
            prop_config,
            force_config.config.clone(),
            params_vec,
            additional_dynamics_fn,
            control_input_fn,
            Some(cov_matrix),
        )
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let dynamics = DynamicsSource::OrbitPropagator(prop);

        // Process measurement models
        let models = process_measurement_models(py, measurement_models)?;

        // Build EKF config
        let ekf_config = config
            .map(|c| c.config.clone())
            .unwrap_or_default();

        let ekf = estimation::ExtendedKalmanFilter::from_propagator(dynamics, models, ekf_config)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyExtendedKalmanFilter { ekf })
    }

    /// Process a single observation.
    ///
    /// Performs predict (propagate to observation epoch) then update (incorporate measurement).
    ///
    /// Args:
    ///     observation (Observation): The observation to process.
    ///
    /// Returns:
    ///     FilterRecord: Record containing pre/post-fit residuals, Kalman gain, etc.
    fn process_observation(
        &mut self,
        observation: &PyObservation,
    ) -> PyResult<PyFilterRecord> {
        let record = self
            .ekf
            .process_observation(&observation.observation)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyFilterRecord { record })
    }

    /// Process multiple observations (auto-sorted by epoch).
    ///
    /// Args:
    ///     observations (list[Observation]): List of observations.
    fn process_observations(
        &mut self,
        observations: Vec<PyRef<PyObservation>>,
    ) -> PyResult<()> {
        let obs_vec: Vec<estimation::Observation> = observations
            .iter()
            .map(|o| o.observation.clone())
            .collect();
        self.ekf
            .process_observations(&obs_vec)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Get current state estimate.
    ///
    /// Returns:
    ///     numpy.ndarray: Current state vector.
    fn current_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let state = self.ekf.current_state();
        let n = state.len();
        vector_to_numpy!(py, state, n, f64)
    }

    /// Get current covariance estimate.
    ///
    /// Returns:
    ///     numpy.ndarray or None: Current covariance matrix, or None if unavailable.
    fn current_covariance<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<Bound<'py, PyArray<f64, numpy::Ix2>>> {
        self.ekf.current_covariance().map(|cov| {
            let r = cov.nrows();
            let c = cov.ncols();
            matrix_to_numpy!(py, cov, r, c, f64)
        })
    }

    /// Get current epoch.
    ///
    /// Returns:
    ///     Epoch: Current filter epoch.
    fn current_epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.ekf.current_epoch(),
        }
    }

    /// Get all stored filter records.
    ///
    /// Returns:
    ///     list[FilterRecord]: List of filter records.
    fn records(&self) -> Vec<PyFilterRecord> {
        self.ekf
            .records()
            .iter()
            .map(|r| PyFilterRecord { record: r.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ExtendedKalmanFilter(epoch={}, state_dim={}, records={})",
            self.ekf.current_epoch(),
            self.ekf.current_state().len(),
            self.ekf.records().len(),
        )
    }
}

// =============================================================================
// UKFConfig
// =============================================================================

/// Configuration for the Unscented Kalman Filter.
///
/// Example:
///     ```python
///     import brahe as bh
///     config = bh.UKFConfig(state_dim=6, alpha=1e-3, beta=2.0, kappa=0.0)
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "UKFConfig")]
#[derive(Clone)]
pub struct PyUKFConfig {
    pub(crate) config: estimation::UKFConfig,
}

#[pymethods]
impl PyUKFConfig {
    /// Create a new UKFConfig.
    ///
    /// Args:
    ///     state_dim (int): State vector dimension. Defaults to 6.
    ///     alpha (float): Sigma point spread parameter (typically 1e-3). Defaults to 1e-3.
    ///     beta (float): Distribution parameter (2.0 for Gaussian). Defaults to 2.0.
    ///     kappa (float): Secondary scaling parameter (typically 0.0). Defaults to 0.0.
    ///     process_noise (ProcessNoiseConfig or None): Optional process noise. Defaults to None.
    ///     store_records (bool): Whether to store filter records. Defaults to True.
    ///
    /// Returns:
    ///     UKFConfig: New UKF configuration.
    #[new]
    #[pyo3(signature = (state_dim=6, alpha=1e-3, beta=2.0, kappa=0.0, process_noise=None, store_records=true))]
    fn new(
        state_dim: usize,
        alpha: f64,
        beta: f64,
        kappa: f64,
        process_noise: Option<PyProcessNoiseConfig>,
        store_records: bool,
    ) -> Self {
        PyUKFConfig {
            config: estimation::UKFConfig {
                process_noise: process_noise.map(|pn| pn.config),
                state_dim,
                alpha,
                beta,
                kappa,
                store_records,
            },
        }
    }

    /// Create a default UKF configuration.
    ///
    /// Returns:
    ///     UKFConfig: Default configuration (state_dim=6, alpha=1e-3, beta=2.0, kappa=0.0).
    #[staticmethod]
    #[pyo3(name = "default")]
    fn py_default() -> Self {
        PyUKFConfig {
            config: estimation::UKFConfig::default(),
        }
    }

    #[getter]
    fn state_dim(&self) -> usize {
        self.config.state_dim
    }

    #[getter]
    fn alpha(&self) -> f64 {
        self.config.alpha
    }

    #[getter]
    fn beta(&self) -> f64 {
        self.config.beta
    }

    #[getter]
    fn kappa(&self) -> f64 {
        self.config.kappa
    }

    #[getter]
    fn store_records(&self) -> bool {
        self.config.store_records
    }

    fn __repr__(&self) -> String {
        format!(
            "UKFConfig(state_dim={}, alpha={}, beta={}, kappa={})",
            self.config.state_dim,
            self.config.alpha,
            self.config.beta,
            self.config.kappa,
        )
    }
}

// =============================================================================
// UnscentedKalmanFilter
// =============================================================================

/// Unscented Kalman Filter for sequential state estimation.
///
/// Uses sigma points to propagate state statistics through nonlinear dynamics
/// and measurement models without linearization. Does not require Jacobians
/// or STM propagation.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
///
///     epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
///     state = np.array([6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0])
///     p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
///
///     ukf = bh.UnscentedKalmanFilter(
///         epoch, state, p0,
///         propagation_config=bh.NumericalPropagationConfig.default(),
///         force_config=bh.ForceModelConfig.two_body(),
///         measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "UnscentedKalmanFilter")]
pub struct PyUnscentedKalmanFilter {
    ukf: estimation::UnscentedKalmanFilter,
}

#[pymethods]
impl PyUnscentedKalmanFilter {
    /// Create a new UnscentedKalmanFilter.
    ///
    /// Args:
    ///     epoch (Epoch): Initial epoch.
    ///     state (numpy.ndarray): Initial state vector in ECI [x,y,z,vx,vy,vz,...] (meters, m/s).
    ///     initial_covariance (numpy.ndarray): Initial covariance matrix (n x n).
    ///     propagation_config (NumericalPropagationConfig): Propagation configuration.
    ///     force_config (ForceModelConfig): Force model configuration.
    ///     measurement_models (list): List of measurement models (built-in or custom).
    ///     config (UKFConfig or None): UKF configuration. Defaults to UKFConfig.default().
    ///     params (numpy.ndarray or None): Parameter vector for force models.
    ///     additional_dynamics (callable or None): Additional dynamics function.
    ///     control_input (callable or None): Control input function.
    ///
    /// Returns:
    ///     UnscentedKalmanFilter: New UKF instance.
    #[new]
    #[pyo3(signature = (
        epoch, state, initial_covariance,
        propagation_config, force_config,
        measurement_models,
        config=None, params=None, additional_dynamics=None, control_input=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        epoch: &PyEpoch,
        state: PyReadonlyArray1<f64>,
        initial_covariance: PyReadonlyArray2<f64>,
        propagation_config: &PyNumericalPropagationConfig,
        force_config: &PyForceModelConfig,
        measurement_models: Vec<Py<PyAny>>,
        config: Option<&PyUKFConfig>,
        params: Option<PyReadonlyArray1<f64>>,
        additional_dynamics: Option<Py<PyAny>>,
        control_input: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let state_vec = nalgebra::DVector::from_column_slice(state.as_slice()?);
        let state_dim = state_vec.len();

        let cov_shape = initial_covariance.shape();
        if cov_shape[0] != state_dim || cov_shape[1] != state_dim {
            return Err(exceptions::PyValueError::new_err(format!(
                "initial_covariance must be {}x{}, got {}x{}",
                state_dim, state_dim, cov_shape[0], cov_shape[1]
            )));
        }
        let cov_data: Vec<f64> = initial_covariance.as_slice()?.to_vec();
        let cov_matrix =
            nalgebra::DMatrix::from_row_slice(state_dim, state_dim, &cov_data);

        // UKF does not need STM — use propagation config as-is
        let prop_config = propagation_config.config.clone();

        let params_vec =
            params.map(|p| nalgebra::DVector::from_column_slice(p.as_slice().unwrap()));

        // Wrap additional_dynamics callable
        let additional_dynamics_fn: Option<crate::integrators::traits::DStateDynamics> =
            additional_dynamics.map(|dyn_py| {
                let dyn_py = dyn_py.clone_ref(py);
                Box::new(
                    move |t: f64,
                          x: &nalgebra::DVector<f64>,
                          p: Option<&nalgebra::DVector<f64>>|
                          -> nalgebra::DVector<f64> {
                        Python::attach(|py| {
                            let x_np = x.as_slice().to_pyarray(py);
                            let p_np: Option<Bound<'_, numpy::PyArray<f64, numpy::Ix1>>> =
                                p.map(|pv| pv.as_slice().to_pyarray(py).to_owned());
                            let result = match p_np {
                                Some(params_arr) => dyn_py.call1(py, (t, x_np, params_arr)),
                                None => dyn_py.call1(py, (t, x_np, py.None())),
                            };
                            match result {
                                Ok(res) => {
                                    let res_arr: PyReadonlyArray1<f64> =
                                        res.extract(py).unwrap();
                                    nalgebra::DVector::from_column_slice(
                                        res_arr.as_slice().unwrap(),
                                    )
                                }
                                Err(e) => {
                                    panic!("Error calling additional_dynamics: {e}")
                                }
                            }
                        })
                    },
                ) as crate::integrators::traits::DStateDynamics
            });

        // Wrap control_input callable
        let control_input_fn: crate::integrators::traits::DControlInput =
            control_input.map(|ctrl_py| {
                let ctrl_py = ctrl_py.clone_ref(py);
                Box::new(
                    move |t: f64,
                          x: &nalgebra::DVector<f64>,
                          p: Option<&nalgebra::DVector<f64>>|
                          -> nalgebra::DVector<f64> {
                        Python::attach(|py| {
                            let x_np = x.as_slice().to_pyarray(py);
                            let p_np: Option<Bound<'_, numpy::PyArray<f64, numpy::Ix1>>> =
                                p.map(|pv| pv.as_slice().to_pyarray(py).to_owned());
                            let result = match p_np {
                                Some(params_arr) => ctrl_py.call1(py, (t, x_np, params_arr)),
                                None => ctrl_py.call1(py, (t, x_np, py.None())),
                            };
                            match result {
                                Ok(res) => {
                                    let res_arr: PyReadonlyArray1<f64> =
                                        res.extract(py).unwrap();
                                    nalgebra::DVector::from_column_slice(
                                        res_arr.as_slice().unwrap(),
                                    )
                                }
                                Err(e) => {
                                    panic!("Error calling control_input: {e}")
                                }
                            }
                        })
                    },
                )
                    as Box<
                        dyn Fn(
                                f64,
                                &nalgebra::DVector<f64>,
                                Option<&nalgebra::DVector<f64>>,
                            ) -> nalgebra::DVector<f64>
                            + Send
                            + Sync,
                    >
            });

        let prop = propagators::DNumericalOrbitPropagator::new(
            epoch.obj,
            state_vec,
            prop_config,
            force_config.config.clone(),
            params_vec,
            additional_dynamics_fn,
            control_input_fn,
            Some(cov_matrix),
        )
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let dynamics = DynamicsSource::OrbitPropagator(prop);
        let models = process_measurement_models(py, measurement_models)?;

        let ukf_config = config
            .map(|c| {
                let mut cfg = c.config.clone();
                cfg.state_dim = state_dim;
                cfg
            })
            .unwrap_or(estimation::UKFConfig {
                state_dim,
                ..estimation::UKFConfig::default()
            });

        let ukf = estimation::UnscentedKalmanFilter::from_propagator(dynamics, models, ukf_config)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyUnscentedKalmanFilter { ukf })
    }

    /// Process a single observation.
    ///
    /// Args:
    ///     observation (Observation): The observation to process.
    ///
    /// Returns:
    ///     FilterRecord: Record containing pre/post-fit residuals, gain, etc.
    fn process_observation(
        &mut self,
        observation: &PyObservation,
    ) -> PyResult<PyFilterRecord> {
        let record = self
            .ukf
            .process_observation(&observation.observation)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(PyFilterRecord { record })
    }

    /// Process multiple observations (auto-sorted by epoch).
    ///
    /// Args:
    ///     observations (list[Observation]): List of observations.
    fn process_observations(
        &mut self,
        observations: Vec<PyRef<PyObservation>>,
    ) -> PyResult<()> {
        let obs_vec: Vec<estimation::Observation> = observations
            .iter()
            .map(|o| o.observation.clone())
            .collect();
        self.ukf
            .process_observations(&obs_vec)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Get current state estimate.
    ///
    /// Returns:
    ///     numpy.ndarray: Current state vector.
    fn current_state<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray<f64, numpy::Ix1>> {
        let state = self.ukf.current_state();
        let n = state.len();
        vector_to_numpy!(py, state, n, f64)
    }

    /// Get current covariance estimate.
    ///
    /// Returns:
    ///     numpy.ndarray: Current covariance matrix.
    fn current_covariance<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, numpy::PyArray<f64, numpy::Ix2>> {
        let cov = self.ukf.current_covariance();
        let r = cov.nrows();
        let c = cov.ncols();
        matrix_to_numpy!(py, cov, r, c, f64)
    }

    /// Get current epoch.
    ///
    /// Returns:
    ///     Epoch: Current filter epoch.
    fn current_epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.ukf.current_epoch(),
        }
    }

    /// Get all stored filter records.
    ///
    /// Returns:
    ///     list[FilterRecord]: List of filter records.
    fn records(&self) -> Vec<PyFilterRecord> {
        self.ukf
            .records()
            .iter()
            .map(|r| PyFilterRecord { record: r.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "UnscentedKalmanFilter(epoch={}, state_dim={}, records={})",
            self.ukf.current_epoch(),
            self.ukf.current_state().len(),
            self.ukf.records().len(),
        )
    }
}

// =============================================================================
// BLSSolverMethod
// =============================================================================

/// Solver formulation for Batch Least Squares.
///
/// Available methods:
///     - ``BLSSolverMethod.NORMAL_EQUATIONS`` (0): Accumulate information matrix,
///       solve via Cholesky. Memory-efficient O(n^2).
///     - ``BLSSolverMethod.STACKED_OBSERVATION_MATRIX`` (1): Build full H matrix,
///       better numerical conditioning. Memory O(m*n).
///
/// Example:
///     ```python
///     import brahe as bh
///     method = bh.BLSSolverMethod.NORMAL_EQUATIONS
///     config = bh.BLSConfig(solver_method=method)
///     ```
#[pyclass(module = "brahe._brahe", skip_from_py_object)]
#[pyo3(name = "BLSSolverMethod")]
#[derive(Clone)]
pub struct PyBLSSolverMethod {}

#[pymethods]
impl PyBLSSolverMethod {
    #[classattr]
    const NORMAL_EQUATIONS: u8 = 0;
    #[classattr]
    const STACKED_OBSERVATION_MATRIX: u8 = 1;

    fn __repr__(&self) -> String {
        "BLSSolverMethod".to_string()
    }
}

/// Map a u8 solver method value to the Rust enum.
fn map_solver_method(value: u8) -> PyResult<estimation::BLSSolverMethod> {
    match value {
        0 => Ok(estimation::BLSSolverMethod::NormalEquations),
        1 => Ok(estimation::BLSSolverMethod::StackedObservationMatrix),
        _ => Err(exceptions::PyValueError::new_err(format!(
            "Invalid solver_method: {}. Use BLSSolverMethod.NORMAL_EQUATIONS (0) or \
             BLSSolverMethod.STACKED_OBSERVATION_MATRIX (1)",
            value
        ))),
    }
}

// =============================================================================
// ConsiderParameterConfig
// =============================================================================

/// Configuration for consider parameters in batch estimation.
///
/// Partitions the state into solve-for (first ``n_solve`` elements) and
/// consider (remaining elements) parameters. The consider parameters are
/// not estimated but their uncertainty is accounted for in the covariance.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # 6D state: first 6 are solve-for, next 1 is consider (e.g., Cd)
///     consider_cov = np.array([[0.01]])
///     config = bh.ConsiderParameterConfig(n_solve=6, consider_covariance=consider_cov)
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "ConsiderParameterConfig")]
#[derive(Clone)]
pub struct PyConsiderParameterConfig {
    pub(crate) config: estimation::ConsiderParameterConfig,
}

#[pymethods]
impl PyConsiderParameterConfig {
    /// Create a new ConsiderParameterConfig.
    ///
    /// Args:
    ///     n_solve (int): Number of solve-for parameters (first n elements of state).
    ///     consider_covariance (numpy.ndarray): A priori covariance for the consider
    ///         parameters (n_c x n_c), where n_c = state_dim - n_solve.
    ///
    /// Returns:
    ///     ConsiderParameterConfig: New consider parameter configuration.
    #[new]
    fn new(n_solve: usize, consider_covariance: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let shape = consider_covariance.shape();
        if shape[0] != shape[1] {
            return Err(exceptions::PyValueError::new_err(
                "consider_covariance must be a square matrix",
            ));
        }
        let data: Vec<f64> = consider_covariance.as_slice()?.to_vec();
        let cov = DMatrix::from_row_slice(shape[0], shape[1], &data);

        Ok(PyConsiderParameterConfig {
            config: estimation::ConsiderParameterConfig {
                n_solve,
                consider_covariance: cov,
            },
        })
    }

    /// Number of solve-for parameters.
    ///
    /// Returns:
    ///     int: Number of solve-for parameters.
    #[getter]
    fn n_solve(&self) -> usize {
        self.config.n_solve
    }

    /// A priori covariance for consider parameters.
    ///
    /// Returns:
    ///     numpy.ndarray: Consider covariance matrix (n_c x n_c).
    #[getter]
    fn consider_covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, numpy::Ix2>> {
        let r = self.config.consider_covariance.nrows();
        let c = self.config.consider_covariance.ncols();
        matrix_to_numpy!(py, self.config.consider_covariance, r, c, f64)
    }

    fn __repr__(&self) -> String {
        format!(
            "ConsiderParameterConfig(n_solve={}, consider_dim={})",
            self.config.n_solve,
            self.config.consider_covariance.nrows(),
        )
    }
}

// =============================================================================
// BLSConfig
// =============================================================================

/// Configuration for the Batch Least Squares estimator.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     config = bh.BLSConfig()  # All defaults
///     config = bh.BLSConfig(max_iterations=20, solver_method=bh.BLSSolverMethod.STACKED_OBSERVATION_MATRIX)
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "BLSConfig")]
#[derive(Clone)]
pub struct PyBLSConfig {
    pub(crate) config: estimation::BLSConfig,
}

#[pymethods]
impl PyBLSConfig {
    /// Create a new BLSConfig.
    ///
    /// Args:
    ///     solver_method (int): Solver formulation. Use ``BLSSolverMethod.NORMAL_EQUATIONS`` (0)
    ///         or ``BLSSolverMethod.STACKED_OBSERVATION_MATRIX`` (1). Defaults to 0.
    ///     max_iterations (int): Maximum Gauss-Newton iterations. Defaults to 10.
    ///     state_correction_threshold (float or None): Convergence threshold on ||delta_x||.
    ///         Defaults to 1e-8.
    ///     cost_convergence_threshold (float or None): Convergence threshold on relative cost change.
    ///         Defaults to None.
    ///     consider_params (ConsiderParameterConfig or None): Consider parameter configuration.
    ///         Defaults to None.
    ///     store_iteration_records (bool): Whether to store per-iteration diagnostics. Defaults to True.
    ///     store_observation_residuals (bool): Whether to store per-observation residuals. Defaults to True.
    ///
    /// Returns:
    ///     BLSConfig: New BLS configuration.
    #[new]
    #[pyo3(signature = (
        solver_method=0,
        max_iterations=10,
        state_correction_threshold=Some(1e-8),
        cost_convergence_threshold=None,
        consider_params=None,
        store_iteration_records=true,
        store_observation_residuals=true
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        solver_method: u8,
        max_iterations: usize,
        state_correction_threshold: Option<f64>,
        cost_convergence_threshold: Option<f64>,
        consider_params: Option<PyConsiderParameterConfig>,
        store_iteration_records: bool,
        store_observation_residuals: bool,
    ) -> PyResult<Self> {
        let method = map_solver_method(solver_method)?;
        Ok(PyBLSConfig {
            config: estimation::BLSConfig {
                solver_method: method,
                max_iterations,
                state_correction_threshold,
                cost_convergence_threshold,
                consider_params: consider_params.map(|cp| cp.config),
                store_iteration_records,
                store_observation_residuals,
            },
        })
    }

    /// Create a default BLS configuration.
    ///
    /// Returns:
    ///     BLSConfig: Default configuration (normal equations, 10 iterations, threshold 1e-8).
    #[staticmethod]
    #[pyo3(name = "default")]
    fn py_default() -> Self {
        PyBLSConfig {
            config: estimation::BLSConfig::default(),
        }
    }

    /// Solver method (0 = NormalEquations, 1 = StackedObservationMatrix).
    ///
    /// Returns:
    ///     int: Solver method identifier.
    #[getter]
    fn solver_method(&self) -> u8 {
        match self.config.solver_method {
            estimation::BLSSolverMethod::NormalEquations => 0,
            estimation::BLSSolverMethod::StackedObservationMatrix => 1,
        }
    }

    /// Maximum number of iterations.
    ///
    /// Returns:
    ///     int: Maximum iterations.
    #[getter]
    fn max_iterations(&self) -> usize {
        self.config.max_iterations
    }

    /// State correction convergence threshold.
    ///
    /// Returns:
    ///     float or None: Threshold value.
    #[getter]
    fn state_correction_threshold(&self) -> Option<f64> {
        self.config.state_correction_threshold
    }

    /// Cost convergence threshold.
    ///
    /// Returns:
    ///     float or None: Threshold value.
    #[getter]
    fn cost_convergence_threshold(&self) -> Option<f64> {
        self.config.cost_convergence_threshold
    }

    /// Whether iteration records are stored.
    ///
    /// Returns:
    ///     bool: True if records are stored.
    #[getter]
    fn store_iteration_records(&self) -> bool {
        self.config.store_iteration_records
    }

    /// Whether observation residuals are stored.
    ///
    /// Returns:
    ///     bool: True if residuals are stored.
    #[getter]
    fn store_observation_residuals(&self) -> bool {
        self.config.store_observation_residuals
    }

    fn __repr__(&self) -> String {
        let method_str = match self.config.solver_method {
            estimation::BLSSolverMethod::NormalEquations => "NormalEquations",
            estimation::BLSSolverMethod::StackedObservationMatrix => "StackedObservationMatrix",
        };
        format!(
            "BLSConfig(solver={}, max_iter={}, state_thresh={:?}, cost_thresh={:?})",
            method_str,
            self.config.max_iterations,
            self.config.state_correction_threshold,
            self.config.cost_convergence_threshold,
        )
    }
}

// =============================================================================
// BLSIterationRecord
// =============================================================================

/// Record of a single BLS iteration.
///
/// Contains the state estimate, covariance, state correction, cost, and
/// residual statistics at each Gauss-Newton iteration.
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "BLSIterationRecord")]
#[derive(Clone)]
pub struct PyBLSIterationRecord {
    pub(crate) record: estimation::BLSIterationRecord,
}

#[pymethods]
impl PyBLSIterationRecord {
    /// Iteration number (0-indexed).
    ///
    /// Returns:
    ///     int: Iteration number.
    #[getter]
    fn iteration(&self) -> usize {
        self.record.iteration
    }

    /// Reference epoch for this iteration.
    ///
    /// Returns:
    ///     Epoch: Iteration epoch.
    #[getter]
    fn epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.record.epoch,
        }
    }

    /// State estimate at this iteration.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector.
    #[getter]
    fn state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let n = self.record.state.len();
        vector_to_numpy!(py, self.record.state, n, f64)
    }

    /// Covariance at this iteration (formal, solve-for only).
    ///
    /// Returns:
    ///     numpy.ndarray: Covariance matrix.
    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, numpy::Ix2>> {
        let r = self.record.covariance.nrows();
        let c = self.record.covariance.ncols();
        matrix_to_numpy!(py, self.record.covariance, r, c, f64)
    }

    /// State correction applied at this iteration.
    ///
    /// Returns:
    ///     numpy.ndarray: State correction vector delta_x.
    #[getter]
    fn state_correction<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let n = self.record.state_correction.len();
        vector_to_numpy!(py, self.record.state_correction, n, f64)
    }

    /// Norm of the state correction ||delta_x||.
    ///
    /// Returns:
    ///     float: State correction norm.
    #[getter]
    fn state_correction_norm(&self) -> f64 {
        self.record.state_correction_norm
    }

    /// Cost function value J at this iteration.
    ///
    /// Returns:
    ///     float: Cost function value.
    #[getter]
    fn cost(&self) -> f64 {
        self.record.cost
    }

    /// RMS of all pre-fit residuals at this iteration.
    ///
    /// Returns:
    ///     float: Pre-fit RMS.
    #[getter]
    fn rms_prefit_residual(&self) -> f64 {
        self.record.rms_prefit_residual
    }

    /// RMS of all post-fit residuals at this iteration.
    ///
    /// Returns:
    ///     float: Post-fit RMS.
    #[getter]
    fn rms_postfit_residual(&self) -> f64 {
        self.record.rms_postfit_residual
    }

    fn __repr__(&self) -> String {
        format!(
            "BLSIterationRecord(iter={}, cost={:.6e}, dx_norm={:.6e}, rms_prefit={:.3e}, rms_postfit={:.3e})",
            self.record.iteration,
            self.record.cost,
            self.record.state_correction_norm,
            self.record.rms_prefit_residual,
            self.record.rms_postfit_residual,
        )
    }
}

// =============================================================================
// BLSObservationResidual
// =============================================================================

/// Per-observation residual from a BLS iteration.
///
/// Contains pre-fit and post-fit residuals for a single observation.
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "BLSObservationResidual")]
#[derive(Clone)]
pub struct PyBLSObservationResidual {
    pub(crate) record: estimation::BLSObservationResidual,
}

#[pymethods]
impl PyBLSObservationResidual {
    /// Epoch of the observation.
    ///
    /// Returns:
    ///     Epoch: Observation epoch.
    #[getter]
    fn epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.record.epoch,
        }
    }

    /// Name of the measurement model used.
    ///
    /// Returns:
    ///     str: Model name.
    #[getter]
    fn model_name(&self) -> &str {
        &self.record.model_name
    }

    /// Pre-fit residual: y - h(x_k, t) before state correction.
    ///
    /// Returns:
    ///     numpy.ndarray: Pre-fit residual vector.
    #[getter]
    fn prefit_residual<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let n = self.record.prefit_residual.len();
        vector_to_numpy!(py, self.record.prefit_residual, n, f64)
    }

    /// Post-fit residual: y - h(x_{k+1}, t) after state correction.
    ///
    /// Returns:
    ///     numpy.ndarray: Post-fit residual vector.
    #[getter]
    fn postfit_residual<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let n = self.record.postfit_residual.len();
        vector_to_numpy!(py, self.record.postfit_residual, n, f64)
    }

    fn __repr__(&self) -> String {
        format!(
            "BLSObservationResidual(epoch={}, model={})",
            self.record.epoch, self.record.model_name,
        )
    }
}

// =============================================================================
// BatchLeastSquares
// =============================================================================

/// Batch Least Squares estimator for orbit determination.
///
/// Processes all observations simultaneously through an iterative
/// Gauss-Newton algorithm. Supports both normal equations and stacked
/// observation matrix solver formulations.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
///
///     epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
///     state = np.array([6878e3, 0.0, 0.0, 0.0, 7612.0, 0.0])
///     p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
///
///     bls = bh.BatchLeastSquares(
///         epoch, state, p0,
///         propagation_config=bh.NumericalPropagationConfig.default(),
///         force_config=bh.ForceModelConfig.two_body_gravity(),
///         measurement_models=[bh.InertialPositionMeasurementModel(10.0)],
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "BatchLeastSquares")]
pub struct PyBatchLeastSquares {
    bls: estimation::BatchLeastSquares,
}

#[pymethods]
impl PyBatchLeastSquares {
    /// Create a new BatchLeastSquares estimator.
    ///
    /// Args:
    ///     epoch (Epoch): Initial (reference) epoch.
    ///     initial_state (numpy.ndarray): Initial state vector in ECI [x,y,z,vx,vy,vz,...] (meters, m/s).
    ///     initial_covariance (numpy.ndarray): A priori covariance matrix (n x n).
    ///     propagation_config (NumericalPropagationConfig): Propagation configuration.
    ///     force_config (ForceModelConfig): Force model configuration.
    ///     measurement_models (list): List of measurement models (built-in or custom).
    ///     config (BLSConfig or None): BLS configuration. Defaults to BLSConfig.default().
    ///     params (numpy.ndarray or None): Parameter vector for force models.
    ///     additional_dynamics (callable or None): Additional dynamics function f(t, state, params) -> derivative.
    ///     control_input (callable or None): Control input function f(t, state, params) -> acceleration.
    ///
    /// Returns:
    ///     BatchLeastSquares: New BLS instance.
    #[new]
    #[pyo3(signature = (
        epoch, initial_state, initial_covariance,
        propagation_config, force_config,
        measurement_models,
        config=None, params=None, additional_dynamics=None, control_input=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        epoch: &PyEpoch,
        initial_state: PyReadonlyArray1<f64>,
        initial_covariance: PyReadonlyArray2<f64>,
        propagation_config: &PyNumericalPropagationConfig,
        force_config: &PyForceModelConfig,
        measurement_models: Vec<Py<PyAny>>,
        config: Option<&PyBLSConfig>,
        params: Option<PyReadonlyArray1<f64>>,
        additional_dynamics: Option<Py<PyAny>>,
        control_input: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let state_vec = DVector::from_column_slice(initial_state.as_slice()?);
        let state_dim = state_vec.len();

        // Parse covariance
        let cov_shape = initial_covariance.shape();
        if cov_shape[0] != state_dim || cov_shape[1] != state_dim {
            return Err(exceptions::PyValueError::new_err(format!(
                "initial_covariance must be {}x{}, got {}x{}",
                state_dim, state_dim, cov_shape[0], cov_shape[1]
            )));
        }
        let cov_data: Vec<f64> = initial_covariance.as_slice()?.to_vec();
        let cov_matrix = DMatrix::from_row_slice(state_dim, state_dim, &cov_data);

        let params_vec =
            params.map(|p| DVector::from_column_slice(p.as_slice().unwrap()));

        // Wrap additional_dynamics callable
        let additional_dynamics_fn: Option<crate::integrators::traits::DStateDynamics> =
            additional_dynamics.map(|dyn_py| {
                let dyn_py = dyn_py.clone_ref(py);
                Box::new(
                    move |t: f64,
                          x: &DVector<f64>,
                          p: Option<&DVector<f64>>|
                          -> DVector<f64> {
                        Python::attach(|py| {
                            let x_np = x.as_slice().to_pyarray(py);
                            let p_np: Option<Bound<'_, PyArray<f64, Ix1>>> =
                                p.map(|pv| pv.as_slice().to_pyarray(py).to_owned());

                            let result = match p_np {
                                Some(params_arr) => dyn_py.call1(py, (t, x_np, params_arr)),
                                None => dyn_py.call1(py, (t, x_np, py.None())),
                            };

                            match result {
                                Ok(res) => {
                                    let res_arr: PyReadonlyArray1<f64> =
                                        res.extract(py).unwrap();
                                    DVector::from_column_slice(
                                        res_arr.as_slice().unwrap(),
                                    )
                                }
                                Err(e) => {
                                    panic!("Error calling additional_dynamics: {e}")
                                }
                            }
                        })
                    },
                ) as crate::integrators::traits::DStateDynamics
            });

        // Wrap control_input callable
        let control_input_fn: crate::integrators::traits::DControlInput =
            control_input.map(|ctrl_py| {
                let ctrl_py = ctrl_py.clone_ref(py);
                Box::new(
                    move |t: f64,
                          x: &DVector<f64>,
                          p: Option<&DVector<f64>>|
                          -> DVector<f64> {
                        Python::attach(|py| {
                            let x_np = x.as_slice().to_pyarray(py);
                            let p_np: Option<Bound<'_, PyArray<f64, Ix1>>> =
                                p.map(|pv| pv.as_slice().to_pyarray(py).to_owned());

                            let result = match p_np {
                                Some(params_arr) => ctrl_py.call1(py, (t, x_np, params_arr)),
                                None => ctrl_py.call1(py, (t, x_np, py.None())),
                            };

                            match result {
                                Ok(res) => {
                                    let res_arr: PyReadonlyArray1<f64> =
                                        res.extract(py).unwrap();
                                    DVector::from_column_slice(
                                        res_arr.as_slice().unwrap(),
                                    )
                                }
                                Err(e) => {
                                    panic!("Error calling control_input: {e}")
                                }
                            }
                        })
                    },
                )
                    as Box<
                        dyn Fn(f64, &DVector<f64>, Option<&DVector<f64>>) -> DVector<f64>
                            + Send
                            + Sync,
                    >
            });

        // Process measurement models
        let models = process_measurement_models(py, measurement_models)?;

        // Build BLS config
        let bls_config = config
            .map(|c| c.config.clone())
            .unwrap_or_default();

        // Build BLS (this internally creates the propagator with STM enabled)
        let bls = estimation::BatchLeastSquares::new(
            epoch.obj,
            state_vec,
            cov_matrix,
            propagation_config.config.clone(),
            force_config.config.clone(),
            params_vec,
            additional_dynamics_fn,
            control_input_fn,
            models,
            bls_config,
        )
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyBatchLeastSquares { bls })
    }

    /// Solve the batch least squares problem.
    ///
    /// Iteratively processes all observations to find the state that minimizes
    /// the weighted sum of squared residuals.
    ///
    /// Args:
    ///     observations (list[Observation]): List of observations to process.
    ///
    /// Example:
    ///     ```python
    ///     bls.solve(observations)
    ///     print(f"Converged: {bls.converged()}")
    ///     print(f"Iterations: {bls.iterations_completed()}")
    ///     ```
    fn solve(&mut self, observations: Vec<PyRef<PyObservation>>) -> PyResult<()> {
        let obs_vec: Vec<estimation::Observation> = observations
            .iter()
            .map(|o| o.observation.clone())
            .collect();
        self.bls
            .solve(&obs_vec)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    /// Get current state estimate at the reference epoch.
    ///
    /// Returns:
    ///     numpy.ndarray: Current state vector.
    fn current_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let state = self.bls.current_state();
        let n = state.len();
        vector_to_numpy!(py, state, n, f64)
    }

    /// Get current total covariance (formal + consider contribution).
    ///
    /// Returns:
    ///     numpy.ndarray: Total covariance matrix (n x n).
    fn current_covariance<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray<f64, numpy::Ix2>> {
        let cov = self.bls.total_covariance();
        let r = cov.nrows();
        let c = cov.ncols();
        let cov_ref = &cov;
        matrix_to_numpy!(py, cov_ref, r, c, f64)
    }

    /// Get current epoch (reference epoch for the batch).
    ///
    /// Returns:
    ///     Epoch: Current epoch.
    fn current_epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.bls.current_epoch(),
        }
    }

    /// Whether the solver has converged.
    ///
    /// Returns:
    ///     bool: True if converged.
    fn converged(&self) -> bool {
        self.bls.converged()
    }

    /// Number of Gauss-Newton iterations completed.
    ///
    /// Returns:
    ///     int: Number of iterations.
    fn iterations_completed(&self) -> usize {
        self.bls.iterations_completed()
    }

    /// Final cost function value J.
    ///
    /// Returns:
    ///     float: Final cost.
    fn final_cost(&self) -> f64 {
        self.bls.final_cost()
    }

    /// Formal covariance (solve-for partition only).
    ///
    /// Returns:
    ///     numpy.ndarray: Formal covariance matrix.
    fn formal_covariance<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray<f64, numpy::Ix2>> {
        let cov = self.bls.formal_covariance();
        let r = cov.nrows();
        let c = cov.ncols();
        matrix_to_numpy!(py, cov, r, c, f64)
    }

    /// Consider covariance contribution.
    ///
    /// Returns:
    ///     numpy.ndarray or None: Consider covariance matrix, or None if
    ///         consider parameters are not configured.
    fn consider_covariance<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<Bound<'py, PyArray<f64, numpy::Ix2>>> {
        self.bls.consider_covariance().map(|cov| {
            let r = cov.nrows();
            let c = cov.ncols();
            let cov_ref = &cov;
            matrix_to_numpy!(py, cov_ref, r, c, f64)
        })
    }

    /// Per-iteration diagnostic records.
    ///
    /// Only populated when ``config.store_iteration_records`` is True.
    ///
    /// Returns:
    ///     list[BLSIterationRecord]: List of iteration records.
    fn iteration_records(&self) -> Vec<PyBLSIterationRecord> {
        self.bls
            .iteration_records()
            .iter()
            .map(|r| PyBLSIterationRecord { record: r.clone() })
            .collect()
    }

    /// Per-observation residuals for each iteration.
    ///
    /// Only populated when ``config.store_observation_residuals`` is True.
    /// Outer list is indexed by iteration, inner list by observation.
    ///
    /// Returns:
    ///     list[list[BLSObservationResidual]]: Nested list of observation residuals.
    fn observation_residuals(&self) -> Vec<Vec<PyBLSObservationResidual>> {
        self.bls
            .observation_residuals()
            .iter()
            .map(|iter_residuals| {
                iter_residuals
                    .iter()
                    .map(|r| PyBLSObservationResidual { record: r.clone() })
                    .collect()
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchLeastSquares(epoch={}, state_dim={}, converged={}, iterations={})",
            self.bls.current_epoch(),
            self.bls.current_state().len(),
            self.bls.converged(),
            self.bls.iterations_completed(),
        )
    }
}
