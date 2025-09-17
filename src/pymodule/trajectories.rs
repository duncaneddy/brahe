/// Python wrapper for OrbitFrame enum
#[pyclass]
#[pyo3(name = "OrbitFrame")]
#[derive(Clone)]
pub struct PyOrbitFrame {
    frame: crate::trajectories::OrbitFrame,
}

#[pymethods]
impl PyOrbitFrame {
    /// Create ECI frame
    #[classmethod]
    fn eci(_cls: &Bound<'_, PyType>) -> Self {
        PyOrbitFrame { frame: crate::trajectories::OrbitFrame::ECI }
    }

    /// Create ECEF frame
    #[classmethod]
    fn ecef(_cls: &Bound<'_, PyType>) -> Self {
        PyOrbitFrame { frame: crate::trajectories::OrbitFrame::ECEF }
    }

    /// Get frame name
    fn name(&self) -> &str {
        use crate::ReferenceFrame;
        self.frame.name()
    }

    fn __str__(&self) -> String {
        self.name().to_string()
    }

    fn __repr__(&self) -> String {
        format!("OrbitFrame.{}", match self.frame {
            crate::trajectories::OrbitFrame::ECI => "ECI",
            crate::trajectories::OrbitFrame::ECEF => "ECEF",
        })
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.frame == other.frame),
            CompareOp::Ne => Ok(self.frame != other.frame),
            _ => Err(exceptions::PyNotImplementedError::new_err("Comparison not supported")),
        }
    }
}

/// Python wrapper for OrbitStateType enum
#[pyclass]
#[pyo3(name = "OrbitStateType")]
#[derive(Clone)]
pub struct PyOrbitStateType {
    state_type: crate::trajectories::OrbitStateType,
}

#[pymethods]
impl PyOrbitStateType {
    /// Create Cartesian state type
    #[classmethod]
    fn cartesian(_cls: &Bound<'_, PyType>) -> Self {
        PyOrbitStateType { state_type: crate::trajectories::OrbitStateType::Cartesian }
    }

    /// Create Keplerian state type
    #[classmethod]
    fn keplerian(_cls: &Bound<'_, PyType>) -> Self {
        PyOrbitStateType { state_type: crate::trajectories::OrbitStateType::Keplerian }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.state_type)
    }

    fn __repr__(&self) -> String {
        format!("OrbitStateType.{:?}", self.state_type)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.state_type == other.state_type),
            CompareOp::Ne => Ok(self.state_type != other.state_type),
            _ => Err(exceptions::PyNotImplementedError::new_err("Comparison not supported")),
        }
    }
}

/// Python wrapper for AngleFormat enum
#[pyclass]
#[pyo3(name = "AngleFormat")]
#[derive(Clone)]
pub struct PyAngleFormat {
    format: crate::trajectories::AngleFormat,
}

#[pymethods]
impl PyAngleFormat {
    /// Create radians angle format
    #[classmethod]
    fn radians(_cls: &Bound<'_, PyType>) -> Self {
        PyAngleFormat { format: crate::trajectories::AngleFormat::Radians }
    }

    /// Create degrees angle format
    #[classmethod]
    fn degrees(_cls: &Bound<'_, PyType>) -> Self {
        PyAngleFormat { format: crate::trajectories::AngleFormat::Degrees }
    }

    /// Create none angle format
    #[classmethod]
    fn none(_cls: &Bound<'_, PyType>) -> Self {
        PyAngleFormat { format: crate::trajectories::AngleFormat::None }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.format)
    }

    fn __repr__(&self) -> String {
        format!("AngleFormat.{:?}", self.format)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.format == other.format),
            CompareOp::Ne => Ok(self.format != other.format),
            _ => Err(exceptions::PyNotImplementedError::new_err("Comparison not supported")),
        }
    }
}

/// Python wrapper for InterpolationMethod enum
#[pyclass]
#[pyo3(name = "InterpolationMethod")]
#[derive(Clone)]
pub struct PyInterpolationMethod {
    method: crate::trajectories::InterpolationMethod,
}

#[pymethods]
impl PyInterpolationMethod {
    /// No interpolation
    #[classmethod]
    fn none(_cls: &Bound<'_, PyType>) -> Self {
        PyInterpolationMethod { method: crate::trajectories::InterpolationMethod::None }
    }

    /// Linear interpolation
    #[classmethod]
    fn linear(_cls: &Bound<'_, PyType>) -> Self {
        PyInterpolationMethod { method: crate::trajectories::InterpolationMethod::Linear }
    }

    /// Cubic spline interpolation (not yet implemented)
    #[classmethod]
    fn cubic_spline(_cls: &Bound<'_, PyType>) -> Self {
        PyInterpolationMethod { method: crate::trajectories::InterpolationMethod::CubicSpline }
    }

    /// Lagrange interpolation (not yet implemented)
    #[classmethod]
    fn lagrange(_cls: &Bound<'_, PyType>) -> Self {
        PyInterpolationMethod { method: crate::trajectories::InterpolationMethod::Lagrange }
    }

    /// Hermite interpolation (not yet implemented)
    #[classmethod]
    fn hermite(_cls: &Bound<'_, PyType>) -> Self {
        PyInterpolationMethod { method: crate::trajectories::InterpolationMethod::Hermite }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.method)
    }

    fn __repr__(&self) -> String {
        format!("InterpolationMethod.{:?}", self.method)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.method == other.method),
            CompareOp::Ne => Ok(self.method != other.method),
            _ => Err(exceptions::PyNotImplementedError::new_err("Comparison not supported")),
        }
    }
}

/// Python wrapper for OrbitState
#[pyclass]
#[pyo3(name = "OrbitState")]
pub struct PyOrbitState {
    state: crate::trajectories::OrbitState,
}

#[pymethods]
impl PyOrbitState {
    /// Create a new OrbitState
    ///
    /// Arguments:
    ///     epoch (Epoch): Time of the state
    ///     state (numpy.ndarray): 6-element state vector
    ///     frame (OrbitFrame): Reference frame
    ///     orbit_type (OrbitStateType): Type of orbit representation
    ///     angle_format (AngleFormat): Format for angular quantities
    ///
    /// Returns:
    ///     OrbitState: New orbit state instance
    #[new]
    #[pyo3(text_signature = "(epoch, state, frame, orbit_type, angle_format)")]
    pub fn new(
        epoch: PyRef<PyEpoch>,
        state: PyReadonlyArray1<f64>,
        frame: PyRef<PyOrbitFrame>,
        orbit_type: PyRef<PyOrbitStateType>,
        angle_format: PyRef<PyAngleFormat>,
    ) -> PyResult<Self> {
        let state_array = state.as_array();
        if state_array.len() != 6 {
            return Err(exceptions::PyValueError::new_err(
                "State vector must have exactly 6 elements"
            ));
        }

        let state_vec = na::Vector6::from_row_slice(state_array.as_slice().unwrap());

        match crate::trajectories::OrbitState::new(
            epoch.obj,
            state_vec,
            frame.frame,
            orbit_type.state_type,
            angle_format.format,
        ) {
            Ok(orbit_state) => Ok(PyOrbitState { state: orbit_state }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the epoch of the state
    fn epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.state.epoch }
    }

    /// Get the state vector as numpy array
    fn state<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let flat_vec: Vec<f64> = (0..6).map(|i| self.state.state[i]).collect();
        Ok(flat_vec.into_pyarray(py))
    }

    /// Get the reference frame
    #[getter]
    fn frame(&self) -> PyOrbitFrame {
        PyOrbitFrame { frame: self.state.frame }
    }

    /// Get the orbit type
    #[getter]
    fn orbit_type(&self) -> PyOrbitStateType {
        PyOrbitStateType { state_type: self.state.orbit_type }
    }

    /// Get the angle format
    #[getter]
    fn angle_format(&self) -> PyAngleFormat {
        PyAngleFormat { format: self.state.angle_format }
    }

    /// Get metadata dictionary
    #[getter]
    fn metadata(&self, py: Python) -> PyResult<Py<pyo3::PyAny>> {
        use pyo3::types::PyDict;
        let dict = PyDict::new(py);
        for (key, value) in &self.state.metadata {
            dict.set_item(key, value)?;
        }
        Ok(dict.into())
    }

    /// Add metadata to the state
    ///
    /// Arguments:
    ///     key (str): Metadata key
    ///     value (str): Metadata value
    ///
    /// Returns:
    ///     OrbitState: New state with added metadata
    #[pyo3(text_signature = "(key, value)")]
    fn with_metadata(&self, key: &str, value: &str) -> Self {
        PyOrbitState {
            state: self.state.clone().with_metadata(key, value)
        }
    }

    /// Get position component (Cartesian only)
    ///
    /// Returns:
    ///     numpy.ndarray: Position vector [x, y, z] in meters
    ///
    /// Raises:
    ///     RuntimeError: If not in Cartesian representation
    #[pyo3(text_signature = "()")]
    fn position<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.state.position() {
            Ok(pos) => {
                let pos_vec = vec![pos.x, pos.y, pos.z];
                Ok(pos_vec.into_pyarray(py))
            },
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get velocity component (Cartesian only)
    ///
    /// Returns:
    ///     numpy.ndarray: Velocity vector [vx, vy, vz] in m/s
    ///
    /// Raises:
    ///     RuntimeError: If not in Cartesian representation
    #[pyo3(text_signature = "()")]
    fn velocity<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.state.velocity() {
            Ok(vel) => {
                let vel_vec = vec![vel.x, vel.y, vel.z];
                Ok(vel_vec.into_pyarray(py))
            },
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert to Cartesian representation
    ///
    /// Returns:
    ///     OrbitState: State in Cartesian representation
    #[pyo3(text_signature = "()")]
    fn to_cartesian(&self) -> PyResult<Self> {
        match self.state.to_cartesian() {
            Ok(cart_state) => Ok(PyOrbitState { state: cart_state }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert to Keplerian representation
    ///
    /// Arguments:
    ///     angle_format (AngleFormat): Desired angle format
    ///
    /// Returns:
    ///     OrbitState: State in Keplerian representation
    #[pyo3(text_signature = "(angle_format)")]
    fn to_keplerian(&self, angle_format: PyRef<PyAngleFormat>) -> PyResult<Self> {
        match self.state.to_keplerian(angle_format.format) {
            Ok(kep_state) => Ok(PyOrbitState { state: kep_state }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert to degrees representation
    ///
    /// Returns:
    ///     OrbitState: State with angles in degrees
    #[pyo3(text_signature = "()")]
    fn as_degrees(&self) -> Self {
        use crate::State;
        PyOrbitState { state: self.state.as_degrees() }
    }

    /// Convert to radians representation
    ///
    /// Returns:
    ///     OrbitState: State with angles in radians
    #[pyo3(text_signature = "()")]
    fn as_radians(&self) -> Self {
        use crate::State;
        PyOrbitState { state: self.state.as_radians() }
    }

    /// Convert to different reference frame
    ///
    /// Arguments:
    ///     frame (OrbitFrame): Target reference frame
    ///
    /// Returns:
    ///     OrbitState: State in target frame
    #[pyo3(text_signature = "(frame)")]
    fn to_frame(&self, frame: PyRef<PyOrbitFrame>) -> PyResult<Self> {
        use crate::State;
        match self.state.to_frame(&frame.frame) {
            Ok(new_state) => Ok(PyOrbitState { state: new_state }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get element by index
    ///
    /// Arguments:
    ///     index (int): Element index (0-5)
    ///
    /// Returns:
    ///     float: Element value
    #[pyo3(text_signature = "(index)")]
    fn get_element(&self, index: usize) -> PyResult<f64> {
        use crate::State;
        match self.state.get_element(index) {
            Ok(val) => Ok(val),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get number of elements
    ///
    /// Returns:
    ///     int: Number of elements (always 6)
    #[pyo3(text_signature = "()")]
    fn len(&self) -> usize {
        use crate::State;
        self.state.len()
    }

    /// Check if empty
    ///
    /// Returns:
    ///     bool: Always False for valid states
    #[pyo3(text_signature = "()")]
    fn is_empty(&self) -> bool {
        use crate::State;
        self.state.is_empty()
    }

    /// Linear interpolation with another state
    ///
    /// Arguments:
    ///     other (OrbitState): Other state to interpolate with
    ///     alpha (float): Interpolation factor (0.0 to 1.0)
    ///     epoch (Epoch): Epoch for interpolated state
    ///
    /// Returns:
    ///     OrbitState: Interpolated state
    #[pyo3(text_signature = "(other, alpha, epoch)")]
    fn interpolate_with(
        &self,
        other: PyRef<PyOrbitState>,
        alpha: f64,
        epoch: PyRef<PyEpoch>,
    ) -> PyResult<Self> {
        use crate::State;
        match self.state.interpolate_with(&other.state, alpha, &epoch.obj) {
            Ok(interp_state) => Ok(PyOrbitState { state: interp_state }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert to JSON string
    ///
    /// Returns:
    ///     str: JSON representation
    #[pyo3(text_signature = "()")]
    fn to_json(&self) -> PyResult<String> {
        match self.state.to_json() {
            Ok(json) => Ok(json),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Create OrbitState from JSON string
    ///
    /// Arguments:
    ///     json (str): JSON string
    ///
    /// Returns:
    ///     OrbitState: Parsed state
    #[classmethod]
    #[pyo3(text_signature = "(json)")]
    fn from_json(_cls: &Bound<'_, PyType>, json: &str) -> PyResult<Self> {
        match crate::trajectories::OrbitState::from_json(json) {
            Ok(state) => Ok(PyOrbitState { state }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    // Python special methods

    /// Get item by index
    fn __getitem__(&self, index: usize) -> PyResult<f64> {
        if index >= 6 {
            return Err(exceptions::PyIndexError::new_err("Index out of range"));
        }
        Ok(self.state[index])
    }

    /// Length
    fn __len__(&self) -> usize {
        6
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "OrbitState(epoch={:?}, frame={:?}, type={:?})",
            self.state.epoch, self.state.frame, self.state.orbit_type
        )
    }

    /// String conversion
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Python wrapper for Trajectory
#[pyclass]
#[pyo3(name = "Trajectory")]
pub struct PyTrajectory {
    trajectory: crate::trajectories::Trajectory<crate::trajectories::OrbitState>,
}

#[pymethods]
impl PyTrajectory {
    /// Create a new empty trajectory
    ///
    /// Arguments:
    ///     interpolation_method (InterpolationMethod): Interpolation method to use
    ///
    /// Returns:
    ///     Trajectory: New empty trajectory
    #[new]
    #[pyo3(text_signature = "(interpolation_method)")]
    pub fn new(interpolation_method: PyRef<PyInterpolationMethod>) -> Self {
        PyTrajectory {
            trajectory: crate::trajectories::Trajectory::new(interpolation_method.method)
        }
    }

    /// Create trajectory from states
    ///
    /// Arguments:
    ///     states (list[OrbitState]): List of orbit states
    ///     interpolation_method (InterpolationMethod): Interpolation method to use
    ///
    /// Returns:
    ///     Trajectory: New trajectory with states
    #[classmethod]
    #[pyo3(text_signature = "(states, interpolation_method)")]
    fn from_states(
        _cls: &Bound<'_, PyType>,
        states: Vec<PyRef<PyOrbitState>>,
        interpolation_method: PyRef<PyInterpolationMethod>,
    ) -> PyResult<Self> {
        let orbit_states: Vec<crate::trajectories::OrbitState> = states.iter().map(|s| s.state.clone()).collect();

        match crate::trajectories::Trajectory::from_states(orbit_states, interpolation_method.method) {
            Ok(trajectory) => Ok(PyTrajectory { trajectory }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Set propagator type
    ///
    /// Arguments:
    ///     propagator_type (str): Propagator type ("SGP4", "Numerical", "Analytical", "Ephemeris")
    ///
    /// Returns:
    ///     Trajectory: Trajectory with propagator type set
    #[pyo3(text_signature = "(propagator_type)")]
    fn with_propagator(&self, propagator_type: &str) -> PyResult<Self> {
        let prop_type = match propagator_type {
            "SGP4" => crate::trajectories::PropagatorType::SGP4,
            "Numerical" => crate::trajectories::PropagatorType::Numerical,
            "Analytical" => crate::trajectories::PropagatorType::Analytical,
            "Ephemeris" => crate::trajectories::PropagatorType::Ephemeris,
            _ => return Err(exceptions::PyValueError::new_err("Invalid propagator type")),
        };

        Ok(PyTrajectory {
            trajectory: self.trajectory.clone().with_propagator(prop_type)
        })
    }

    /// Set maximum trajectory size
    ///
    /// Arguments:
    ///     max_size (Optional[int]): Maximum number of states (None for unlimited)
    #[pyo3(text_signature = "(max_size)")]
    fn set_max_size(&mut self, max_size: Option<usize>) {
        self.trajectory.set_max_size(max_size);
    }

    /// Set maximum age of states
    ///
    /// Arguments:
    ///     max_age (Optional[float]): Maximum age in seconds (None for unlimited)
    #[pyo3(text_signature = "(max_age)")]
    fn set_max_age(&mut self, max_age: Option<f64>) {
        self.trajectory.set_max_age(max_age);
    }

    /// Set eviction policy
    ///
    /// Arguments:
    ///     policy (str): Eviction policy ("None", "KeepCount", "KeepWithinDuration")
    #[pyo3(text_signature = "(policy)")]
    fn set_eviction_policy(&mut self, policy: &str) -> PyResult<()> {
        let eviction_policy = match policy {
            "None" => crate::trajectories::TrajectoryEvictionPolicy::None,
            "KeepCount" => crate::trajectories::TrajectoryEvictionPolicy::KeepCount,
            "KeepWithinDuration" => crate::trajectories::TrajectoryEvictionPolicy::KeepWithinDuration,
            _ => return Err(exceptions::PyValueError::new_err("Invalid eviction policy")),
        };

        self.trajectory.set_eviction_policy(eviction_policy);
        Ok(())
    }

    /// Add a state to the trajectory
    ///
    /// Arguments:
    ///     state (OrbitState): State to add
    #[pyo3(text_signature = "(state)")]
    fn add_state(&mut self, state: PyRef<PyOrbitState>) -> PyResult<()> {
        match self.trajectory.add_state(state.state.clone()) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get state at specific epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     OrbitState: Interpolated state at epoch
    #[pyo3(text_signature = "(epoch)")]
    fn state_at_epoch(&self, epoch: PyRef<PyEpoch>) -> PyResult<PyOrbitState> {
        match self.trajectory.state_at_epoch(&epoch.obj) {
            Ok(state) => Ok(PyOrbitState { state }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Find nearest state to epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     OrbitState: Nearest state
    #[pyo3(text_signature = "(epoch)")]
    fn nearest_state(&self, epoch: PyRef<PyEpoch>) -> PyResult<PyOrbitState> {
        match self.trajectory.nearest_state(&epoch.obj) {
            Ok(state) => Ok(PyOrbitState { state }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get state before epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     OrbitState: State before epoch
    #[pyo3(text_signature = "(epoch)")]
    fn state_before(&self, epoch: PyRef<PyEpoch>) -> PyResult<PyOrbitState> {
        match self.trajectory.state_before(&epoch.obj) {
            Ok(state) => Ok(PyOrbitState { state }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get state after epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     OrbitState: State after epoch
    #[pyo3(text_signature = "(epoch)")]
    fn state_after(&self, epoch: PyRef<PyEpoch>) -> PyResult<PyOrbitState> {
        match self.trajectory.state_after(&epoch.obj) {
            Ok(state) => Ok(PyOrbitState { state }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get state at index
    ///
    /// Arguments:
    ///     index (int): State index
    ///
    /// Returns:
    ///     OrbitState: State at index
    #[pyo3(text_signature = "(index)")]
    fn state_at_index(&self, index: usize) -> PyResult<PyOrbitState> {
        match self.trajectory.state_at_index(index) {
            Ok(state) => Ok(PyOrbitState { state }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert to different frame
    ///
    /// Arguments:
    ///     frame (OrbitFrame): Target frame
    ///
    /// Returns:
    ///     Trajectory: Trajectory in target frame
    #[pyo3(text_signature = "(frame)")]
    fn to_frame(&self, frame: PyRef<PyOrbitFrame>) -> PyResult<Self> {
        match self.trajectory.to_frame(&frame.frame) {
            Ok(traj) => Ok(PyTrajectory { trajectory: traj }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert to degrees representation
    ///
    /// Returns:
    ///     Trajectory: Trajectory with angles in degrees
    #[pyo3(text_signature = "()")]
    fn as_degrees(&self) -> PyResult<Self> {
        match self.trajectory.as_degrees() {
            Ok(traj) => Ok(PyTrajectory { trajectory: traj }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert to radians representation
    ///
    /// Returns:
    ///     Trajectory: Trajectory with angles in radians
    #[pyo3(text_signature = "()")]
    fn as_radians(&self) -> PyResult<Self> {
        match self.trajectory.as_radians() {
            Ok(traj) => Ok(PyTrajectory { trajectory: traj }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert to matrix representation
    ///
    /// Returns:
    ///     numpy.ndarray: Matrix (6, N) where columns are time points
    #[pyo3(text_signature = "()")]
    fn to_matrix<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyArray<f64, Ix2>>> {
        match self.trajectory.to_matrix() {
            Ok(matrix) => {
                let (rows, cols) = matrix.shape();
                let mut flat_vec = Vec::with_capacity(rows * cols);
                for row_idx in 0..rows {
                    for col_idx in 0..cols {
                        flat_vec.push(matrix[(row_idx, col_idx)]);
                    }
                }
                Ok(flat_vec.into_pyarray(py).reshape([rows, cols]).unwrap())
            },
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert to JSON string
    ///
    /// Returns:
    ///     str: JSON representation
    #[pyo3(text_signature = "()")]
    fn to_json(&self) -> PyResult<String> {
        match self.trajectory.to_json() {
            Ok(json) => Ok(json),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Create trajectory from JSON string
    ///
    /// Arguments:
    ///     json (str): JSON string
    ///
    /// Returns:
    ///     Trajectory: Parsed trajectory
    #[classmethod]
    #[pyo3(text_signature = "(json)")]
    fn from_json(_cls: &Bound<'_, PyType>, json: &str) -> PyResult<Self> {
        match crate::trajectories::Trajectory::from_json(json) {
            Ok(trajectory) => Ok(PyTrajectory { trajectory }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get trajectory length
    ///
    /// Returns:
    ///     int: Number of states in trajectory
    #[pyo3(text_signature = "()")]
    fn len(&self) -> usize {
        self.trajectory.len()
    }

    /// Check if trajectory is empty
    ///
    /// Returns:
    ///     bool: True if empty
    #[pyo3(text_signature = "()")]
    fn is_empty(&self) -> bool {
        self.trajectory.is_empty()
    }

    // Python special methods

    /// Get item by index
    fn __getitem__(&self, index: usize) -> PyResult<PyOrbitState> {
        if index >= self.trajectory.len() {
            return Err(exceptions::PyIndexError::new_err("Index out of range"));
        }
        Ok(PyOrbitState { state: self.trajectory[index].clone() })
    }

    /// Length
    fn __len__(&self) -> usize {
        self.trajectory.len()
    }

    /// Iterator support
    fn __iter__(slf: PyRef<'_, Self>) -> PyTrajectoryIterator {
        PyTrajectoryIterator {
            trajectory: slf.trajectory.clone(),
            index: 0,
        }
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Trajectory(len={}, interpolation={:?})",
            self.trajectory.len(),
            self.trajectory.interpolation_method
        )
    }

    /// String conversion
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Iterator for PyTrajectory
#[pyclass]
pub struct PyTrajectoryIterator {
    trajectory: crate::trajectories::Trajectory<crate::trajectories::OrbitState>,
    index: usize,
}

#[pymethods]
impl PyTrajectoryIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyOrbitState> {
        if slf.index < slf.trajectory.len() {
            let state = slf.trajectory[slf.index].clone();
            slf.index += 1;
            Some(PyOrbitState { state })
        } else {
            None
        }
    }
}