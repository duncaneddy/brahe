/// Python bindings for the new trajectory architecture

// Import traits needed by trajectory metho

/// Python wrapper for InterpolationMethod enum
#[pyclass]
#[pyo3(name = "InterpolationMethod")]
#[derive(Clone)]
pub struct PyInterpolationMethod {
    pub(crate) method: trajectories::traits::InterpolationMethod,
}

#[pymethods]
impl PyInterpolationMethod {
    #[classattr]
    fn LINEAR() -> Self {
        PyInterpolationMethod { method: trajectories::traits::InterpolationMethod::Linear }
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


/// Python wrapper for OrbitFrame enum
#[pyclass]
#[pyo3(name = "OrbitFrame")]
#[derive(Clone)]
pub struct PyOrbitFrame {
    pub(crate) frame: trajectories::traits::OrbitFrame,
}

#[pymethods]
impl PyOrbitFrame {
    /// ECI frame
    #[classattr]
    fn ECI() -> Self {
        PyOrbitFrame { frame: trajectories::traits::OrbitFrame::ECI }
    }

    /// ECEF frame
    #[classattr]
    fn ECEF() -> Self {
        PyOrbitFrame { frame: trajectories::traits::OrbitFrame::ECEF }
    }

    /// Get frame name
    fn name(&self) -> &str {
        match self.frame {
            trajectories::traits::OrbitFrame::ECI => "Earth-Centered Inertial (J2000)",
            trajectories::traits::OrbitFrame::ECEF => "Earth-Centered Earth-Fixed",
        }
    }

    fn __str__(&self) -> String {
        self.name().to_string()
    }

    fn __repr__(&self) -> String {
        format!("OrbitFrame.{}", match self.frame {
            trajectories::traits::OrbitFrame::ECI => "ECI",
            trajectories::traits::OrbitFrame::ECEF => "ECEF",
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

/// Python wrapper for OrbitRepresentation enum
#[pyclass]
#[pyo3(name = "OrbitRepresentation")]
#[derive(Clone)]
pub struct PyOrbitRepresentation {
    pub(crate) representation: trajectories::traits::OrbitRepresentation,
}

#[pymethods]
impl PyOrbitRepresentation {
    /// Cartesian representation
    #[classattr]
    fn CARTESIAN() -> Self {
        PyOrbitRepresentation { representation: trajectories::traits::OrbitRepresentation::Cartesian }
    }

    /// Keplerian representation
    #[classattr]
    fn KEPLERIAN() -> Self {
        PyOrbitRepresentation { representation: trajectories::traits::OrbitRepresentation::Keplerian }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.representation)
    }

    fn __repr__(&self) -> String {
        format!("OrbitRepresentation.{:?}", self.representation)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.representation == other.representation),
            CompareOp::Ne => Ok(self.representation != other.representation),
            _ => Err(exceptions::PyNotImplementedError::new_err("Comparison not supported")),
        }
    }
}

/// Python wrapper for OrbitTrajectory (unified design)
#[pyclass]
#[pyo3(name = "OrbitTrajectory")]
pub struct PyOrbitalTrajectory {
    pub(crate) trajectory: trajectories::OrbitTrajectory,
}

#[pymethods]
impl PyOrbitalTrajectory {
    /// Create a new empty orbital trajectory
    ///
    /// Arguments:
    ///     frame (OrbitFrame): Reference frame
    ///     representation (OrbitRepresentation): Orbital representation
    ///     angle_format (AngleFormat | None): Format for angular quantities (None for Cartesian)
    ///
    /// Returns:
    ///     OrbitTrajectory: New trajectory instance
    #[new]
    #[pyo3(signature = (frame, representation, angle_format=None), text_signature = "(frame, representation, angle_format=None)")]
    pub fn new(
        frame: PyRef<PyOrbitFrame>,
        representation: PyRef<PyOrbitRepresentation>,
        angle_format: Option<PyRef<PyAngleFormat>>,
    ) -> PyResult<Self> {
        // Validate: Cartesian must have None, Keplerian must have Some
        match (representation.representation, &angle_format) {
            (trajectories::traits::OrbitRepresentation::Cartesian, Some(_)) => {
                return Err(exceptions::PyValueError::new_err(
                    "Angle format must be None for Cartesian representation"
                ));
            },
            (trajectories::traits::OrbitRepresentation::Keplerian, None) => {
                return Err(exceptions::PyValueError::new_err(
                    "Angle format must be specified for Keplerian representation"
                ));
            },
            _ => {}
        }

        let angle_fmt = angle_format.as_ref().map(|af| af.value);

        let trajectory = trajectories::OrbitTrajectory::new(
            frame.frame,
            representation.representation,
            angle_fmt,
        );
        Ok(PyOrbitalTrajectory { trajectory })
    }

    /// Create a default empty orbital trajectory (ECI Cartesian)
    ///
    /// Returns:
    ///     OrbitTrajectory: New trajectory instance with ECI frame, Cartesian representation
    #[classmethod]
    #[pyo3(text_signature = "()")]
    pub fn default(_cls: &Bound<'_, PyType>) -> Self {
        PyOrbitalTrajectory {
            trajectory: trajectories::OrbitTrajectory::default(),
        }
    }

    /// Create orbital trajectory from data
    ///
    /// Arguments:
    ///     epochs (list[Epoch]): List of epochs
    ///     states: Flattened array of 6-element state vectors (Nx6 total elements)
    ///     frame (OrbitFrame): Reference frame
    ///     representation (OrbitRepresentation): Orbital representation
    ///     angle_format (AngleFormat | None): Format for angular quantities (None for Cartesian)
    ///
    /// Returns:
    ///     OrbitalTrajectory: New trajectory instance with data
    #[classmethod]
    #[pyo3(signature = (epochs, states, frame, representation, angle_format=None), text_signature = "(epochs, states, frame, representation, angle_format=None)")]
    pub fn from_orbital_data(
        _cls: &Bound<'_, PyType>,
        epochs: Vec<PyRef<PyEpoch>>,
        states: PyReadonlyArray1<f64>,
        frame: PyRef<PyOrbitFrame>,
        representation: PyRef<PyOrbitRepresentation>,
        angle_format: Option<PyRef<PyAngleFormat>>,
    ) -> PyResult<Self> {
        // Validate: Cartesian must have None, Keplerian must have Some
        match (representation.representation, &angle_format) {
            (trajectories::traits::OrbitRepresentation::Cartesian, Some(_)) => {
                return Err(exceptions::PyValueError::new_err(
                    "Angle format must be None for Cartesian representation"
                ));
            },
            (trajectories::traits::OrbitRepresentation::Keplerian, None) => {
                return Err(exceptions::PyValueError::new_err(
                    "Angle format must be specified for Keplerian representation"
                ));
            },
            _ => {}
        }

        let epochs_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states_array = states.as_array();

        if states_array.len() % 6 != 0 {
            return Err(exceptions::PyValueError::new_err(
                "States array length must be a multiple of 6"
            ));
        }

        let num_states = states_array.len() / 6;
        if num_states != epochs_vec.len() {
            return Err(exceptions::PyValueError::new_err(
                "Number of epochs must match number of states"
            ));
        }

        let mut states_vec = Vec::new();
        for i in 0..num_states {
            let start_idx = i * 6;
            let state_slice = &states_array.as_slice().unwrap()[start_idx..start_idx + 6];
            states_vec.push(na::Vector6::from_row_slice(state_slice));
        }

        let angle_fmt = angle_format.as_ref().map(|af| af.value);

        let trajectory = trajectories::OrbitTrajectory::from_orbital_data(
            epochs_vec,
            states_vec,
            frame.frame,
            representation.representation,
            angle_fmt,
        );
        Ok(PyOrbitalTrajectory { trajectory })
    }

    /// Set interpolation method using builder pattern
    ///
    /// Arguments:
    ///     interpolation_method (InterpolationMethod): Interpolation method to use
    ///
    /// Returns:
    ///     OrbitalTrajectory: Self with updated interpolation method
    #[pyo3(text_signature = "(interpolation_method)")]
    pub fn with_interpolation_method(mut slf: PyRefMut<'_, Self>, method: PyRef<PyInterpolationMethod>) -> Self {
        slf.trajectory = slf.trajectory.clone().with_interpolation_method(method.method);
        Self { trajectory: slf.trajectory.clone() }
    }

    /// Set eviction policy to keep maximum number of states using builder pattern
    ///
    /// Arguments:
    ///     max_size (int): Maximum number of states to retain
    ///
    /// Returns:
    ///     OrbitalTrajectory: Self with updated eviction policy
    #[pyo3(text_signature = "(max_size)")]
    pub fn with_eviction_policy_max_size(mut slf: PyRefMut<'_, Self>, max_size: usize) -> Self {
        slf.trajectory = slf.trajectory.clone().with_eviction_policy_max_size(max_size);
        Self { trajectory: slf.trajectory.clone() }
    }

    /// Set eviction policy to keep states within maximum age using builder pattern
    ///
    /// Arguments:
    ///     max_age (float): Maximum age of states in seconds
    ///
    /// Returns:
    ///     OrbitalTrajectory: Self with updated eviction policy
    #[pyo3(text_signature = "(max_age)")]
    pub fn with_eviction_policy_max_age(mut slf: PyRefMut<'_, Self>, max_age: f64) -> Self {
        slf.trajectory = slf.trajectory.clone().with_eviction_policy_max_age(max_age);
        Self { trajectory: slf.trajectory.clone() }
    }

    /// Get trajectory dimension (always 6 for orbital trajectories)
    #[pyo3(text_signature = "()")]
    pub fn dimension(&self) -> usize {
        6
    }

    /// Add a state to the trajectory
    ///
    /// Arguments:
    ///     epoch (Epoch): Time of the state
    ///     state (numpy.ndarray): 6-element state vector
    ///
    /// Returns:
    ///     None
    #[pyo3(text_signature = "(epoch, state)")]
    pub fn add(&mut self, epoch: PyRef<PyEpoch>, state: PyReadonlyArray1<f64>) -> PyResult<()> {
        let state_array = state.as_array();
        if state_array.len() != 6 {
            return Err(exceptions::PyValueError::new_err(
                "State vector must have exactly 6 elements"
            ));
        }

        let state_vec = na::Vector6::from_row_slice(state_array.as_slice().unwrap());

        self.trajectory.add(epoch.obj, state_vec);
        Ok(())
    }

    /// Get the nearest state to a given epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: (Epoch, numpy.ndarray) of nearest state
    #[pyo3(text_signature = "(epoch)")]
    pub fn nearest_state<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.nearest_state(&epoch.obj) {
            Ok((nearest_epoch, nearest_state)) => {
                Ok((PyEpoch { obj: nearest_epoch }, nearest_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the index of the state at or before the given epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     int: Index of the state at or before the target epoch
    #[pyo3(text_signature = "(epoch)")]
    pub fn index_before_epoch(&self, epoch: PyRef<PyEpoch>) -> PyResult<usize> {
        match self.trajectory.index_before_epoch(&epoch.obj) {
            Ok(index) => Ok(index),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the index of the state at or after the given epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     int: Index of the state at or after the target epoch
    #[pyo3(text_signature = "(epoch)")]
    pub fn index_after_epoch(&self, epoch: PyRef<PyEpoch>) -> PyResult<usize> {
        match self.trajectory.index_after_epoch(&epoch.obj) {
            Ok(index) => Ok(index),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the state at or before the given epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: (Epoch, numpy.ndarray) of state at or before the target epoch
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_before_epoch<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.state_before_epoch(&epoch.obj) {
            Ok((ret_epoch, ret_state)) => {
                Ok((PyEpoch { obj: ret_epoch }, ret_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the state at or after the given epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: (Epoch, numpy.ndarray) of state at or after the target epoch
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_after_epoch<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.state_after_epoch(&epoch.obj) {
            Ok((ret_epoch, ret_state)) => {
                Ok((PyEpoch { obj: ret_epoch }, ret_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Set the interpolation method for the trajectory
    ///
    /// Arguments:
    ///     method (InterpolationMethod): New interpolation method
    ///
    /// Returns:
    ///     None
    #[pyo3(text_signature = "(method)")]
    pub fn set_interpolation_method(&mut self, method: PyRef<PyInterpolationMethod>) {
        self.trajectory.set_interpolation_method(method.method);
    }

    /// Get the current interpolation method
    ///
    /// Returns:
    ///     InterpolationMethod: Current interpolation method
    #[pyo3(text_signature = "()")]
    pub fn get_interpolation_method(&self) -> PyInterpolationMethod {
        PyInterpolationMethod { method: self.trajectory.get_interpolation_method() }
    }

    /// Interpolate state at a given epoch using linear interpolation
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     numpy.ndarray: Linearly interpolated state vector
    #[pyo3(text_signature = "(epoch)")]
    pub fn interpolate_linear<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.interpolate_linear(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Interpolate state at a given epoch using the configured interpolation method
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     numpy.ndarray: Interpolated state vector
    #[pyo3(text_signature = "(epoch)")]
    pub fn interpolate<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.interpolate(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the number of states in the trajectory
    #[getter]
    pub fn length(&self) -> usize {
        self.trajectory.len()
    }

    /// Get the number of states in the trajectory (alias for length)
    #[pyo3(text_signature = "()")]
    pub fn len(&self) -> usize {
        self.trajectory.len()
    }

    /// Get trajectory frame
    #[getter]
    pub fn frame(&self) -> PyOrbitFrame {
        PyOrbitFrame { frame: self.trajectory.frame }
    }

    /// Get trajectory representation
    #[getter]
    pub fn representation(&self) -> PyOrbitRepresentation {
        PyOrbitRepresentation { representation: self.trajectory.representation }
    }

    /// Get trajectory angle format
    #[getter]
    pub fn angle_format(&self) -> Option<PyAngleFormat> {
        self.trajectory.angle_format.map(|af| PyAngleFormat { value: af })
    }

    /// Clear all states from the trajectory
    #[pyo3(text_signature = "()")]
    pub fn clear(&mut self) {
        self.trajectory.clear();
    }

    /// Get start epoch of trajectory
    ///
    /// Returns:
    ///     Epoch or None: First epoch if trajectory is not empty, None otherwise
    #[pyo3(text_signature = "()")]
    pub fn start_epoch(&self) -> Option<PyEpoch> {
        self.trajectory.start_epoch().map(|epoch| PyEpoch { obj: epoch })
    }

    /// Get end epoch of trajectory
    ///
    /// Returns:
    ///     Epoch or None: Last epoch if trajectory is not empty, None otherwise
    #[pyo3(text_signature = "()")]
    pub fn end_epoch(&self) -> Option<PyEpoch> {
        self.trajectory.end_epoch().map(|epoch| PyEpoch { obj: epoch })
    }

    /// Get time span of trajectory in seconds
    ///
    /// Returns:
    ///     float or None: Time span between first and last epochs, or None if less than 2 states
    #[pyo3(text_signature = "()")]
    pub fn timespan(&self) -> Option<f64> {
        self.trajectory.timespan()
    }

    /// Get the first (epoch, state) tuple in the trajectory, if any exists
    ///
    /// Returns:
    ///     tuple or None: (Epoch, numpy.ndarray) of first state, or None if empty
    #[pyo3(text_signature = "()")]
    pub fn first<'a>(&self, py: Python<'a>) -> Option<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        self.trajectory.first().map(|(epoch, state)| {
            (PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned())
        })
    }

    /// Get the last (epoch, state) tuple in the trajectory, if any exists
    ///
    /// Returns:
    ///     tuple or None: (Epoch, numpy.ndarray) of last state, or None if empty
    #[pyo3(text_signature = "()")]
    pub fn last<'a>(&self, py: Python<'a>) -> Option<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        self.trajectory.last().map(|(epoch, state)| {
            (PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned())
        })
    }

    /// Get all epochs as a numpy array
    #[pyo3(text_signature = "()")]
    pub fn epochs<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray<f64, Ix1>> {
        let epochs: Vec<f64> = self.trajectory.epochs.iter().map(|e| e.jd()).collect();
        epochs.to_pyarray(py).to_owned()
    }

    /// Get all states as a numpy array
    #[pyo3(text_signature = "()")]
    pub fn states<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyArray<f64, numpy::Ix2>>> {
        match self.trajectory.to_matrix() {
            Ok(states_matrix) => {
                let data: Vec<f64> = states_matrix.iter().cloned().collect();
                let shape = (states_matrix.nrows(), states_matrix.ncols());
                Ok(numpy::PyArray::from_vec(py, data).reshape(shape).unwrap().to_owned())
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Check if trajectory is empty
    #[pyo3(text_signature = "()")]
    pub fn is_empty(&self) -> bool {
        self.trajectory.len() == 0
    }

    /// Get state at specific index
    ///
    /// Arguments:
    ///     index (int): Index of the state
    ///
    /// Returns:
    ///     numpy.ndarray: State vector at given index
    #[pyo3(text_signature = "(index)")]
    pub fn state<'a>(&self, py: Python<'a>, index: usize) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        if index >= self.trajectory.len() {
            return Err(exceptions::PyIndexError::new_err("Index out of range"));
        }

        let state = &self.trajectory.states[index];
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Get epoch at specific index
    ///
    /// Arguments:
    ///     index (int): Index of the epoch
    ///
    /// Returns:
    ///     Epoch: Epoch at given index
    #[pyo3(text_signature = "(index)")]
    pub fn epoch(&self, index: usize) -> PyResult<PyEpoch> {
        let epochs = &self.trajectory.epochs;
        if index >= epochs.len() {
            return Err(exceptions::PyIndexError::new_err("Index out of range"));
        }
        Ok(PyEpoch { obj: epochs[index] })
    }

    /// Convert to ECI (Earth-Centered Inertial) frame in Cartesian representation
    ///
    /// Returns:
    ///     OrbitalTrajectory: Trajectory in ECI Cartesian frame
    #[pyo3(text_signature = "()")]
    pub fn to_eci(&self) -> Self {
        let new_trajectory = self.trajectory.to_eci();
        PyOrbitalTrajectory { trajectory: new_trajectory }
    }

    /// Convert to ECEF (Earth-Centered Earth-Fixed) frame in Cartesian representation
    ///
    /// Returns:
    ///     OrbitalTrajectory: Trajectory in ECEF Cartesian frame
    #[pyo3(text_signature = "()")]
    pub fn to_ecef(&self) -> Self {
        let new_trajectory = self.trajectory.to_ecef();
        PyOrbitalTrajectory { trajectory: new_trajectory }
    }

    /// Convert to Keplerian representation in ECI frame
    ///
    /// Arguments:
    ///     angle_format (AngleFormat): Angle format for the result (Radians or Degrees)
    ///
    /// Returns:
    ///     OrbitalTrajectory: Trajectory in ECI Keplerian representation
    #[pyo3(text_signature = "(angle_format)")]
    pub fn to_keplerian(&self, angle_format: PyRef<PyAngleFormat>) -> Self {
        let new_trajectory = self.trajectory.to_keplerian(angle_format.value);
        PyOrbitalTrajectory { trajectory: new_trajectory }
    }

    /// Convert trajectory to matrix representation
    ///
    /// Returns:
    ///     numpy.ndarray: Matrix with shape (6, N) where N is number of states.
    ///                   Each column represents a state at a specific time.
    #[pyo3(text_signature = "()")]
    pub fn to_matrix<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyArray<f64, numpy::Ix2>>> {
        match self.trajectory.to_matrix() {
            Ok(states_matrix) => {
                // Nalgebra uses column-major storage, but numpy expects row-major
                // Iterate explicitly by row then column to build row-major data
                let nrows = states_matrix.nrows();
                let ncols = states_matrix.ncols();
                let mut data = Vec::with_capacity(nrows * ncols);
                for i in 0..nrows {
                    for j in 0..ncols {
                        data.push(states_matrix[(i, j)]);
                    }
                }
                Ok(numpy::PyArray::from_vec(py, data).reshape((nrows, ncols)).unwrap().to_owned())
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Remove a state at a specific epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Epoch of the state to remove
    ///
    /// Returns:
    ///     numpy.ndarray: The removed state vector
    #[pyo3(text_signature = "(epoch)")]
    pub fn remove_epoch<'a>(&mut self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.remove_epoch(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Remove a state at a specific index
    ///
    /// Arguments:
    ///     index (int): Index of the state to remove
    ///
    /// Returns:
    ///     tuple: (Epoch, numpy.ndarray) of the removed epoch and state
    #[pyo3(text_signature = "(index)")]
    pub fn remove<'a>(&mut self, py: Python<'a>, index: usize) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.remove(index) {
            Ok((epoch, state)) => {
                Ok((PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get both epoch and state at a specific index
    ///
    /// Arguments:
    ///     index (int): Index to retrieve
    ///
    /// Returns:
    ///     tuple: (Epoch, numpy.ndarray) of epoch and state at the index
    #[pyo3(text_signature = "(index)")]
    pub fn get<'a>(&self, py: Python<'a>, index: usize) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.get(index) {
            Ok((epoch, state)) => {
                Ok((PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Python length
    fn __len__(&self) -> usize {
        self.trajectory.len()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "OrbitalTrajectory(frame={:?}, representation={:?}, states={})",
            self.trajectory.frame, self.trajectory.representation, self.trajectory.len()
        )
    }

    /// String conversion
    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Set eviction policy to keep maximum number of states
    ///
    /// Arguments:
    ///     max_size (int): Maximum number of states to retain
    ///
    /// Returns:
    ///     None
    #[pyo3(text_signature = "(max_size)")]
    pub fn set_eviction_policy_max_size(&mut self, max_size: usize) -> PyResult<()> {
        match self.trajectory.set_eviction_policy_max_size(max_size) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Set eviction policy to keep states within maximum age
    ///
    /// Arguments:
    ///     max_age (float): Maximum age in seconds relative to most recent state
    ///
    /// Returns:
    ///     None
    #[pyo3(text_signature = "(max_age)")]
    pub fn set_eviction_policy_max_age(&mut self, max_age: f64) -> PyResult<()> {
        match self.trajectory.set_eviction_policy_max_age(max_age) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get current eviction policy
    ///
    /// Returns:
    ///     str: String representation of eviction policy ("None", "KeepCount", or "KeepWithinDuration")
    #[pyo3(text_signature = "()")]
    pub fn get_eviction_policy(&self) -> String {
        format!("{:?}", self.trajectory.get_eviction_policy())
    }

    /// Index access returns state vector at given index
    ///
    /// Arguments:
    ///     index (int): Index of the state
    ///
    /// Returns:
    ///     numpy.ndarray: State vector at index
    fn __getitem__<'a>(&self, py: Python<'a>, index: isize) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let len = self.trajectory.len() as isize;
        let actual_index = if index < 0 {
            (len + index) as usize
        } else {
            index as usize
        };

        if actual_index >= self.trajectory.len() {
            return Err(exceptions::PyIndexError::new_err("Index out of range"));
        }

        let state = &self.trajectory[actual_index];
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Iterator over (epoch, state) pairs
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyOrbitalTrajectoryIterator>> {
        let py = slf.py();
        let iter = PyOrbitalTrajectoryIterator {
            trajectory: slf.into(),
            index: 0,
        };
        Py::new(py, iter)
    }
}

/// Iterator for OrbitTrajectory
#[pyclass]
struct PyOrbitalTrajectoryIterator {
    trajectory: Py<PyOrbitalTrajectory>,
    index: usize,
}

#[pymethods]
impl PyOrbitalTrajectoryIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(&mut self, py: Python<'a>) -> PyResult<Option<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)>> {
        let traj = self.trajectory.borrow(py);
        if self.index < traj.trajectory.len() {
            let (epoch, state) = traj.trajectory.get(self.index)
                .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
            self.index += 1;
            Ok(Some((
                PyEpoch { obj: epoch },
                state.as_slice().to_pyarray(py).to_owned()
            )))
        } else {
            Ok(None)
        }
    }
}

/// Python wrapper for dynamic Trajectory class
#[pyclass]
#[pyo3(name = "DTrajectory")]
pub struct PyTrajectory {
    pub(crate) trajectory: trajectories::DTrajectory,
}

#[pymethods]
impl PyTrajectory {
    /// Create a new empty trajectory
    ///
    /// Arguments:
    ///     dimension (int, optional): Trajectory dimension (default: 6, must be greater than 0)
    ///
    /// Returns:
    ///     DTrajectory: New trajectory instance with linear interpolation
    ///
    /// Examples:
    ///     DTrajectory()    # 6D trajectory (default)
    ///     DTrajectory(3)   # 3D trajectory
    ///     DTrajectory(6)   # 6D trajectory
    ///     DTrajectory(12)  # 12D trajectory
    #[new]
    #[pyo3(signature = (dimension=6))]
    pub fn new(dimension: usize) -> PyResult<Self> {
        if dimension == 0 {
            return Err(exceptions::PyValueError::new_err(
                "Trajectory dimension must be greater than 0"
            ));
        }

        let trajectory = trajectories::DTrajectory::new(dimension);
        Ok(PyTrajectory { trajectory })
    }

    /// Create a trajectory from vectors of epochs and states
    ///
    /// Arguments:
    ///     epochs (list[Epoch]): List of time epochs
    ///     states (numpy.ndarray): 2D array of states with shape (num_epochs, dimension)
    ///         where each row is a state vector
    ///     interpolation_method (InterpolationMethod): Interpolation method (default: Linear)
    ///
    /// Returns:
    ///     Trajectory: New trajectory instance
    #[classmethod]
    #[pyo3(signature = (epochs, states, interpolation_method=None))]
    pub fn from_data(
        _cls: &Bound<'_, PyType>,
        epochs: Vec<PyRef<PyEpoch>>,
        states: PyReadonlyArray2<f64>,
        interpolation_method: Option<PyRef<PyInterpolationMethod>>,
    ) -> PyResult<Self> {
        let method = interpolation_method
            .map(|m| m.method)
            .unwrap_or(trajectories::traits::InterpolationMethod::Linear);

        let epochs_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states_array = states.as_array();

        let num_epochs = epochs_vec.len();
        if num_epochs == 0 {
            return Err(exceptions::PyValueError::new_err(
                "At least one epoch is required"
            ));
        }

        // Check that number of states (rows) matches number of epochs
        if states_array.nrows() != num_epochs {
            return Err(exceptions::PyValueError::new_err(
                format!("Number of state rows ({}) must match number of epochs ({})",
                    states_array.nrows(), num_epochs)
            ));
        }

        let dimension = states_array.ncols();
        if dimension == 0 {
            return Err(exceptions::PyValueError::new_err(
                "State dimension must be greater than 0"
            ));
        }

        let mut trajectory = trajectories::DTrajectory::new(dimension)
            .with_interpolation_method(method);

        for i in 0..num_epochs {
            let state_row = states_array.row(i);
            let state_vec = na::DVector::from_iterator(dimension, state_row.iter().copied());

            trajectory.add(epochs_vec[i], state_vec)
        }

        Ok(PyTrajectory { trajectory })
    }

    /// Set interpolation method using builder pattern
    ///
    /// Arguments:
    ///     interpolation_method (InterpolationMethod): Interpolation method to use
    ///
    /// Returns:
    ///     DTrajectory: Self with updated interpolation method
    #[pyo3(text_signature = "(interpolation_method)")]
    pub fn with_interpolation_method(mut slf: PyRefMut<'_, Self>, method: PyRef<PyInterpolationMethod>) -> Self {
        slf.trajectory = slf.trajectory.clone().with_interpolation_method(method.method);
        Self { trajectory: slf.trajectory.clone() }
    }

    /// Set eviction policy to keep maximum number of states using builder pattern
    ///
    /// Arguments:
    ///     max_size (int): Maximum number of states to retain
    ///
    /// Returns:
    ///     DTrajectory: Self with updated eviction policy
    #[pyo3(text_signature = "(max_size)")]
    pub fn with_eviction_policy_max_size(mut slf: PyRefMut<'_, Self>, max_size: usize) -> Self {
        slf.trajectory = slf.trajectory.clone().with_eviction_policy_max_size(max_size);
        Self { trajectory: slf.trajectory.clone() }
    }

    /// Set eviction policy to keep states within maximum age using builder pattern
    ///
    /// Arguments:
    ///     max_age (float): Maximum age of states in seconds
    ///
    /// Returns:
    ///     DTrajectory: Self with updated eviction policy
    #[pyo3(text_signature = "(max_age)")]
    pub fn with_eviction_policy_max_age(mut slf: PyRefMut<'_, Self>, max_age: f64) -> Self {
        slf.trajectory = slf.trajectory.clone().with_eviction_policy_max_age(max_age);
        Self { trajectory: slf.trajectory.clone() }
    }

    /// Get the trajectory dimension (property)
    #[getter]
    pub fn dimension(&self) -> usize {
        self.trajectory.dimension
    }

    /// Get the trajectory dimension (method for Rust compatibility)
    ///
    /// Returns:
    ///     int: Trajectory dimension
    #[pyo3(name = "dimension")]
    #[pyo3(text_signature = "()")]
    pub fn dimension_method(&self) -> usize {
        self.trajectory.dimension
    }

    /// Add a state to the trajectory
    ///
    /// Arguments:
    ///     epoch (Epoch): Time of the state
    ///     state (numpy.ndarray): N-element state vector (where N is the trajectory dimension)
    ///
    /// Returns:
    ///     None
    #[pyo3(text_signature = "(epoch, state)")]
    pub fn add(&mut self, epoch: PyRef<PyEpoch>, state: PyReadonlyArray1<f64>) -> PyResult<()> {
        let state_array = state.as_array();
        if state_array.len() != self.trajectory.dimension {
            return Err(exceptions::PyValueError::new_err(
                format!("State vector must have exactly {} elements for {}D trajectory",
                    self.trajectory.dimension, self.trajectory.dimension)
            ));
        }

        let state_vec = na::DVector::from_column_slice(state_array.as_slice().unwrap());
        self.trajectory.add(epoch.obj, state_vec);
        Ok(())
    }


    /// Get the nearest state to a given epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: (Epoch, numpy.ndarray) of nearest state
    #[pyo3(text_signature = "(epoch)")]
    pub fn nearest_state<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.nearest_state(&epoch.obj) {
            Ok((nearest_epoch, nearest_state)) => {
                Ok((PyEpoch { obj: nearest_epoch }, nearest_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the index of the state at or before the given epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     int: Index of the state at or before the target epoch
    #[pyo3(text_signature = "(epoch)")]
    pub fn index_before_epoch(&self, epoch: PyRef<PyEpoch>) -> PyResult<usize> {
        match self.trajectory.index_before_epoch(&epoch.obj) {
            Ok(index) => Ok(index),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the index of the state at or after the given epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     int: Index of the state at or after the target epoch
    #[pyo3(text_signature = "(epoch)")]
    pub fn index_after_epoch(&self, epoch: PyRef<PyEpoch>) -> PyResult<usize> {
        match self.trajectory.index_after_epoch(&epoch.obj) {
            Ok(index) => Ok(index),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the state at or before the given epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: (Epoch, numpy.ndarray) of state at or before the target epoch
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_before_epoch<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.state_before_epoch(&epoch.obj) {
            Ok((ret_epoch, ret_state)) => {
                Ok((PyEpoch { obj: ret_epoch }, ret_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the state at or after the given epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: (Epoch, numpy.ndarray) of state at or after the target epoch
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_after_epoch<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.state_after_epoch(&epoch.obj) {
            Ok((ret_epoch, ret_state)) => {
                Ok((PyEpoch { obj: ret_epoch }, ret_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Interpolate state at a given epoch using linear interpolation
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     numpy.ndarray: Linearly interpolated state vector
    #[pyo3(text_signature = "(epoch)")]
    pub fn interpolate_linear<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.interpolate_linear(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Interpolate state at a given epoch using the configured interpolation method
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     numpy.ndarray: Interpolated state vector
    #[pyo3(text_signature = "(epoch)")]
    pub fn interpolate<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.interpolate(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get state at a specific index
    ///
    /// Arguments:
    ///     index (int): Index of the state
    ///
    /// Returns:
    ///     numpy.ndarray: State vector at index
    #[pyo3(text_signature = "(index)")]
    pub fn state<'a>(&self, py: Python<'a>, index: usize) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.state(index) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get epoch at a specific index
    ///
    /// Arguments:
    ///     index (int): Index of the epoch
    ///
    /// Returns:
    ///     Epoch: Epoch at index
    #[pyo3(text_signature = "(index)")]
    pub fn epoch(&self, index: usize) -> PyResult<PyEpoch> {
        match self.trajectory.epoch(index) {
            Ok(epoch) => Ok(PyEpoch { obj: epoch }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the number of states in the trajectory
    #[getter]
    pub fn length(&self) -> usize {
        self.trajectory.len()
    }

    /// Get the number of states in the trajectory (alias for length)
    #[pyo3(text_signature = "()")]
    pub fn len(&self) -> usize {
        self.trajectory.len()
    }

    /// Check if trajectory is empty
    #[pyo3(text_signature = "()")]
    pub fn is_empty(&self) -> bool {
        self.trajectory.is_empty()
    }

    /// Get interpolation method (callable method)
    ///
    /// Returns:
    ///     InterpolationMethod: Current interpolation method
    #[pyo3(text_signature = "()")]
    pub fn get_interpolation_method(&self) -> PyInterpolationMethod {
        PyInterpolationMethod { method: self.trajectory.interpolation_method }
    }

    /// Set interpolation method
    ///
    /// Arguments:
    ///     method (InterpolationMethod): New interpolation method
    ///
    /// Returns:
    ///     None
    #[pyo3(text_signature = "(method)")]
    pub fn set_interpolation_method(&mut self, method: PyRef<PyInterpolationMethod>) {
        self.trajectory.set_interpolation_method(method.method);
    }

    /// Set maximum trajectory size
    #[pyo3(text_signature = "(max_size)")]
    pub fn set_eviction_policy_max_size(&mut self, max_size: usize) -> PyResult<()> {
        match self.trajectory.set_eviction_policy_max_size(max_size) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Set maximum age for trajectory states (in seconds)
    #[pyo3(text_signature = "(max_age)")]
    pub fn set_eviction_policy_max_age(&mut self, max_age: f64) -> PyResult<()> {
        match self.trajectory.set_eviction_policy_max_age(max_age) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get current eviction policy
    ///
    /// Returns:
    ///     str: String representation of eviction policy ("None", "KeepCount", or "KeepWithinDuration")
    #[pyo3(text_signature = "()")]
    pub fn get_eviction_policy(&self) -> String {
        format!("{:?}", self.trajectory.get_eviction_policy())
    }

    /// Get start epoch of trajectory
    #[pyo3(text_signature = "()")]
    pub fn start_epoch(&self) -> Option<PyEpoch> {
        self.trajectory.start_epoch().map(|epoch| PyEpoch { obj: epoch })
    }

    /// Get end epoch of trajectory
    #[pyo3(text_signature = "()")]
    pub fn end_epoch(&self) -> Option<PyEpoch> {
        self.trajectory.end_epoch().map(|epoch| PyEpoch { obj: epoch })
    }

    /// Get time span of trajectory in seconds
    #[pyo3(text_signature = "()")]
    pub fn timespan(&self) -> Option<f64> {
        self.trajectory.timespan()
    }

    /// Clear all states from the trajectory
    #[pyo3(text_signature = "()")]
    pub fn clear(&mut self) {
        self.trajectory.clear();
    }

    /// Get the first (epoch, state) tuple in the trajectory, if any exists
    ///
    /// Returns:
    ///     tuple or None: (Epoch, numpy.ndarray) of first state, or None if empty
    #[pyo3(text_signature = "()")]
    pub fn first<'a>(&self, py: Python<'a>) -> Option<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        self.trajectory.first().map(|(epoch, state)| {
            (PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned())
        })
    }

    /// Get the last (epoch, state) tuple in the trajectory, if any exists
    ///
    /// Returns:
    ///     tuple or None: (Epoch, numpy.ndarray) of last state, or None if empty
    #[pyo3(text_signature = "()")]
    pub fn last<'a>(&self, py: Python<'a>) -> Option<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        self.trajectory.last().map(|(epoch, state)| {
            (PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned())
        })
    }

    /// Get all states as a numpy array
    #[pyo3(text_signature = "()")]
    pub fn to_matrix<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyArray<f64, numpy::Ix2>>> {
        match self.trajectory.to_matrix() {
            Ok(states_matrix) => {
                // Nalgebra uses column-major storage, but numpy expects row-major
                // Iterate explicitly by row then column to build row-major data
                let nrows = states_matrix.nrows();
                let ncols = states_matrix.ncols();
                let mut data = Vec::with_capacity(nrows * ncols);
                for i in 0..nrows {
                    for j in 0..ncols {
                        data.push(states_matrix[(i, j)]);
                    }
                }
                Ok(numpy::PyArray::from_vec(py, data).reshape((nrows, ncols)).unwrap().to_owned())
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Remove a state at a specific epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Epoch of the state to remove
    ///
    /// Returns:
    ///     numpy.ndarray: The removed state vector
    #[pyo3(text_signature = "(epoch)")]
    pub fn remove_epoch<'a>(&mut self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.remove_epoch(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Remove a state at a specific index
    ///
    /// Arguments:
    ///     index (int): Index of the state to remove
    ///
    /// Returns:
    ///     tuple: (Epoch, numpy.ndarray) of the removed epoch and state
    #[pyo3(text_signature = "(index)")]
    pub fn remove<'a>(&mut self, py: Python<'a>, index: usize) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.remove(index) {
            Ok((epoch, state)) => {
                Ok((PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get both epoch and state at a specific index
    ///
    /// Arguments:
    ///     index (int): Index to retrieve
    ///
    /// Returns:
    ///     tuple: (Epoch, numpy.ndarray) of epoch and state at the index
    #[pyo3(text_signature = "(index)")]
    pub fn get<'a>(&self, py: Python<'a>, index: usize) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.get(index) {
            Ok((epoch, state)) => {
                Ok((PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Python length
    fn __len__(&self) -> usize {
        self.trajectory.len()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Trajectory(dimension={}, interpolation_method={:?}, states={})",
            self.trajectory.dimension,
            self.trajectory.interpolation_method,
            self.trajectory.len()
        )
    }

    /// String conversion
    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Index access returns state vector at given index
    ///
    /// Arguments:
    ///     index (int): Index of the state
    ///
    /// Returns:
    ///     numpy.ndarray: State vector at index
    fn __getitem__<'a>(&self, py: Python<'a>, index: isize) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let len = self.trajectory.len() as isize;
        let actual_index = if index < 0 {
            (len + index) as usize
        } else {
            index as usize
        };

        if actual_index >= self.trajectory.len() {
            return Err(exceptions::PyIndexError::new_err("Index out of range"));
        }

        let state = &self.trajectory[actual_index];
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Iterator over (epoch, state) pairs
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyTrajectoryIterator>> {
        let py = slf.py();
        let iter = PyTrajectoryIterator {
            trajectory: slf.into(),
            index: 0,
        };
        Py::new(py, iter)
    }
}

/// Iterator for DTrajectory
#[pyclass]
struct PyTrajectoryIterator {
    trajectory: Py<PyTrajectory>,
    index: usize,
}

#[pymethods]
impl PyTrajectoryIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(&mut self, py: Python<'a>) -> PyResult<Option<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)>> {
        let traj = self.trajectory.borrow(py);
        if self.index < traj.trajectory.len() {
            let (epoch, state) = traj.trajectory.get(self.index)
                .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
            self.index += 1;
            Ok(Some((
                PyEpoch { obj: epoch },
                state.as_slice().to_pyarray(py).to_owned()
            )))
        } else {
            Ok(None)
        }
    }
}


/// Python wrapper for STrajectory<6> - compile-time sized 6D trajectory
#[pyclass]
#[pyo3(name = "STrajectory6")]
pub struct PySTrajectory6 {
    pub(crate) trajectory: trajectories::STrajectory6,
}

#[pymethods]
impl PySTrajectory6 {
    /// Create a new empty 6D static trajectory
    ///
    /// Returns:
    ///     STrajectory6: New trajectory instance
    #[new]
    #[pyo3(signature = (interpolation_method=None))]
    pub fn new(
        interpolation_method: Option<PyRef<PyInterpolationMethod>>,
    ) -> Self {
        let method = interpolation_method
            .map(|m| m.method)
            .unwrap_or(trajectories::traits::InterpolationMethod::Linear);

        let trajectory = trajectories::STrajectory6::new()
            .with_interpolation_method(method);

        PySTrajectory6 { trajectory }
    }

    /// Create a trajectory from vectors of epochs and states
    ///
    /// Arguments:
    ///     epochs (list[Epoch]): List of time epochs
    ///     states (numpy.ndarray): Flattened array of 6D state vectors (Nx6 total elements)
    ///     interpolation_method (InterpolationMethod): Interpolation method (default: Linear)
    ///
    /// Returns:
    ///     STrajectory6: New trajectory instance
    #[classmethod]
    #[pyo3(signature = (epochs, states, interpolation_method=None))]
    pub fn from_data(
        _cls: &Bound<'_, PyType>,
        epochs: Vec<PyRef<PyEpoch>>,
        states: PyReadonlyArray1<f64>,
        interpolation_method: Option<PyRef<PyInterpolationMethod>>,
    ) -> PyResult<Self> {
        let epochs_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states_array = states.as_array();

        if states_array.len() % 6 != 0 {
            return Err(exceptions::PyValueError::new_err(
                "States array length must be a multiple of 6"
            ));
        }

        let num_states = states_array.len() / 6;
        if num_states != epochs_vec.len() {
            return Err(exceptions::PyValueError::new_err(
                "Number of epochs must match number of states"
            ));
        }

        let mut states_vec = Vec::new();
        for i in 0..num_states {
            let start_idx = i * 6;
            let state_slice = &states_array.as_slice().unwrap()[start_idx..start_idx + 6];
            states_vec.push(na::Vector6::from_row_slice(state_slice));
        }

        let mut trajectory = trajectories::STrajectory6::from_data(epochs_vec, states_vec)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

        if let Some(method) = interpolation_method {
            trajectory.set_interpolation_method(method.method);
        }

        Ok(PySTrajectory6 { trajectory })
    }

    /// Set interpolation method using builder pattern
    ///
    /// Arguments:
    ///     interpolation_method (InterpolationMethod): Interpolation method to use
    ///
    /// Returns:
    ///     STrajectory6: Self with updated interpolation method
    #[pyo3(text_signature = "(interpolation_method)")]
    pub fn with_interpolation_method(mut slf: PyRefMut<'_, Self>, method: PyRef<PyInterpolationMethod>) -> Self {
        slf.trajectory = slf.trajectory.clone().with_interpolation_method(method.method);
        Self { trajectory: slf.trajectory.clone() }
    }

    /// Set eviction policy to keep maximum number of states using builder pattern
    ///
    /// Arguments:
    ///     max_size (int): Maximum number of states to retain
    ///
    /// Returns:
    ///     STrajectory6: Self with updated eviction policy
    #[pyo3(text_signature = "(max_size)")]
    pub fn with_eviction_policy_max_size(mut slf: PyRefMut<'_, Self>, max_size: usize) -> Self {
        slf.trajectory = slf.trajectory.clone().with_eviction_policy_max_size(max_size);
        Self { trajectory: slf.trajectory.clone() }
    }

    /// Set eviction policy to keep states within maximum age using builder pattern
    ///
    /// Arguments:
    ///     max_age (float): Maximum age of states in seconds
    ///
    /// Returns:
    ///     STrajectory6: Self with updated eviction policy
    #[pyo3(text_signature = "(max_age)")]
    pub fn with_eviction_policy_max_age(mut slf: PyRefMut<'_, Self>, max_age: f64) -> Self {
        slf.trajectory = slf.trajectory.clone().with_eviction_policy_max_age(max_age);
        Self { trajectory: slf.trajectory.clone() }
    }

    /// Get trajectory dimension (always 6)
    #[pyo3(text_signature = "()")]
    pub fn dimension(&self) -> usize {
        6
    }

    /// Add a state to the trajectory
    ///
    /// Arguments:
    ///     epoch (Epoch): Time of the state
    ///     state (numpy.ndarray): 6-element state vector
    ///
    /// Returns:
    ///     None
    #[pyo3(text_signature = "(epoch, state)")]
    pub fn add(&mut self, epoch: PyRef<PyEpoch>, state: PyReadonlyArray1<f64>) -> PyResult<()> {
        let state_array = state.as_array();
        if state_array.len() != 6 {
            return Err(exceptions::PyValueError::new_err(
                "State vector must have exactly 6 elements"
            ));
        }

        let state_vec = na::Vector6::from_row_slice(state_array.as_slice().unwrap());

        self.trajectory.add(epoch.obj, state_vec);
        Ok(())
    }

    /// Get the nearest state to a given epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: (Epoch, numpy.ndarray) of nearest state
    #[pyo3(text_signature = "(epoch)")]
    pub fn nearest_state<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.nearest_state(&epoch.obj) {
            Ok((nearest_epoch, nearest_state)) => {
                Ok((PyEpoch { obj: nearest_epoch }, nearest_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the index of the state at or before the given epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     int: Index of the state at or before the target epoch
    #[pyo3(text_signature = "(epoch)")]
    pub fn index_before_epoch(&self, epoch: PyRef<PyEpoch>) -> PyResult<usize> {
        match self.trajectory.index_before_epoch(&epoch.obj) {
            Ok(index) => Ok(index),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the index of the state at or after the given epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     int: Index of the state at or after the target epoch
    #[pyo3(text_signature = "(epoch)")]
    pub fn index_after_epoch(&self, epoch: PyRef<PyEpoch>) -> PyResult<usize> {
        match self.trajectory.index_after_epoch(&epoch.obj) {
            Ok(index) => Ok(index),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the state at or before the given epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: (Epoch, numpy.ndarray) of state at or before the target epoch
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_before_epoch<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.state_before_epoch(&epoch.obj) {
            Ok((ret_epoch, ret_state)) => {
                Ok((PyEpoch { obj: ret_epoch }, ret_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the state at or after the given epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: (Epoch, numpy.ndarray) of state at or after the target epoch
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_after_epoch<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.state_after_epoch(&epoch.obj) {
            Ok((ret_epoch, ret_state)) => {
                Ok((PyEpoch { obj: ret_epoch }, ret_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Set the interpolation method for the trajectory
    ///
    /// Arguments:
    ///     method (InterpolationMethod): New interpolation method
    ///
    /// Returns:
    ///     None
    #[pyo3(text_signature = "(method)")]
    pub fn set_interpolation_method(&mut self, method: PyRef<PyInterpolationMethod>) {
        self.trajectory.set_interpolation_method(method.method);
    }

    /// Interpolate state at a given epoch using linear interpolation
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     numpy.ndarray: Linearly interpolated state vector
    #[pyo3(text_signature = "(epoch)")]
    pub fn interpolate_linear<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.interpolate_linear(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Interpolate state at a given epoch using the configured interpolation method
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     numpy.ndarray: Interpolated state vector
    #[pyo3(text_signature = "(epoch)")]
    pub fn interpolate<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.interpolate(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get state at a specific index
    ///
    /// Arguments:
    ///     index (int): Index of the state
    ///
    /// Returns:
    ///     numpy.ndarray: State vector at index
    #[pyo3(text_signature = "(index)")]
    pub fn state<'a>(&self, py: Python<'a>, index: usize) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.state(index) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get epoch at a specific index
    ///
    /// Arguments:
    ///     index (int): Index of the epoch
    ///
    /// Returns:
    ///     Epoch: Epoch at index
    #[pyo3(text_signature = "(index)")]
    pub fn epoch(&self, index: usize) -> PyResult<PyEpoch> {
        match self.trajectory.epoch(index) {
            Ok(epoch) => Ok(PyEpoch { obj: epoch }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the number of states in the trajectory
    #[getter]
    pub fn length(&self) -> usize {
        self.trajectory.len()
    }

    /// Get the number of states in the trajectory (alias for length)
    #[pyo3(text_signature = "()")]
    pub fn len(&self) -> usize {
        self.trajectory.len()
    }

    /// Get interpolation method as property
    #[getter]
    pub fn interpolation_method(&self) -> PyInterpolationMethod {
        PyInterpolationMethod { method: self.trajectory.get_interpolation_method() }
    }

    /// Set maximum trajectory size
    #[pyo3(text_signature = "(max_size)")]
    pub fn set_eviction_policy_max_size(&mut self, max_size: usize) -> PyResult<()> {
        match self.trajectory.set_eviction_policy_max_size(max_size) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Set maximum age for trajectory states (in seconds)
    #[pyo3(text_signature = "(max_age)")]
    pub fn set_eviction_policy_max_age(&mut self, max_age: f64) -> PyResult<()> {
        match self.trajectory.set_eviction_policy_max_age(max_age) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get current eviction policy
    ///
    /// Returns:
    ///     str: String representation of eviction policy ("None", "KeepCount", or "KeepWithinDuration")
    #[pyo3(text_signature = "()")]
    pub fn get_eviction_policy(&self) -> String {
        format!("{:?}", self.trajectory.get_eviction_policy())
    }

    /// Get start epoch of trajectory
    #[getter]
    pub fn start_epoch(&self) -> Option<PyEpoch> {
        self.trajectory.start_epoch().map(|epoch| PyEpoch { obj: epoch })
    }

    /// Get end epoch of trajectory
    #[getter]
    pub fn end_epoch(&self) -> Option<PyEpoch> {
        self.trajectory.end_epoch().map(|epoch| PyEpoch { obj: epoch })
    }

    /// Get time span of trajectory in seconds
    #[getter]
    pub fn time_span(&self) -> Option<f64> {
        self.trajectory.timespan()
    }

    /// Clear all states from the trajectory
    #[pyo3(text_signature = "()")]
    pub fn clear(&mut self) {
        self.trajectory.clear();
    }

    /// Get the first (epoch, state) tuple in the trajectory, if any exists
    ///
    /// Returns:
    ///     tuple or None: (Epoch, numpy.ndarray) of first state, or None if empty
    #[pyo3(text_signature = "()")]
    pub fn first<'a>(&self, py: Python<'a>) -> Option<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        self.trajectory.first().map(|(epoch, state)| {
            (PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned())
        })
    }

    /// Get the last (epoch, state) tuple in the trajectory, if any exists
    ///
    /// Returns:
    ///     tuple or None: (Epoch, numpy.ndarray) of last state, or None if empty
    #[pyo3(text_signature = "()")]
    pub fn last<'a>(&self, py: Python<'a>) -> Option<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        self.trajectory.last().map(|(epoch, state)| {
            (PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned())
        })
    }

    /// Get all states as a numpy array
    #[pyo3(text_signature = "()")]
    pub fn to_matrix<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyArray<f64, numpy::Ix2>>> {
        match self.trajectory.to_matrix() {
            Ok(states_matrix) => {
                // Nalgebra uses column-major storage, but numpy expects row-major
                // Iterate explicitly by row then column to build row-major data
                let nrows = states_matrix.nrows();
                let ncols = states_matrix.ncols();
                let mut data = Vec::with_capacity(nrows * ncols);
                for i in 0..nrows {
                    for j in 0..ncols {
                        data.push(states_matrix[(i, j)]);
                    }
                }
                Ok(numpy::PyArray::from_vec(py, data).reshape((nrows, ncols)).unwrap().to_owned())
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Remove a state at a specific epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Epoch of the state to remove
    ///
    /// Returns:
    ///     numpy.ndarray: The removed state vector
    #[pyo3(text_signature = "(epoch)")]
    pub fn remove_epoch<'a>(&mut self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.remove_epoch(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Remove a state at a specific index
    ///
    /// Arguments:
    ///     index (int): Index of the state to remove
    ///
    /// Returns:
    ///     tuple: (Epoch, numpy.ndarray) of the removed epoch and state
    #[pyo3(text_signature = "(index)")]
    pub fn remove<'a>(&mut self, py: Python<'a>, index: usize) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.remove(index) {
            Ok((epoch, state)) => {
                Ok((PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get both epoch and state at a specific index
    ///
    /// Arguments:
    ///     index (int): Index to retrieve
    ///
    /// Returns:
    ///     tuple: (Epoch, numpy.ndarray) of epoch and state at the index
    #[pyo3(text_signature = "(index)")]
    pub fn get<'a>(&self, py: Python<'a>, index: usize) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.get(index) {
            Ok((epoch, state)) => {
                Ok((PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Check if trajectory is empty
    #[pyo3(text_signature = "()")]
    pub fn is_empty(&self) -> bool {
        self.trajectory.is_empty()
    }

    /// Python length
    fn __len__(&self) -> usize {
        self.trajectory.len()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "STrajectory6(interpolation_method={:?}, states={})",
            self.trajectory.get_interpolation_method(),
            self.trajectory.len()
        )
    }

    /// String conversion
    fn __str__(&self) -> String {
        self.__repr__()
    }

    /// Index access returns state vector at given index
    ///
    /// Arguments:
    ///     index (int): Index of the state
    ///
    /// Returns:
    ///     numpy.ndarray: State vector at index
    fn __getitem__<'a>(&self, py: Python<'a>, index: isize) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let len = self.trajectory.len() as isize;
        let actual_index = if index < 0 {
            (len + index) as usize
        } else {
            index as usize
        };

        if actual_index >= self.trajectory.len() {
            return Err(exceptions::PyIndexError::new_err("Index out of range"));
        }

        let state = &self.trajectory[actual_index];
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Iterator over (epoch, state) pairs
    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PySTrajectory6Iterator>> {
        let py = slf.py();
        let iter = PySTrajectory6Iterator {
            trajectory: slf.into(),
            index: 0,
        };
        Py::new(py, iter)
    }
}

/// Iterator for STrajectory6
#[pyclass]
struct PySTrajectory6Iterator {
    trajectory: Py<PySTrajectory6>,
    index: usize,
}

#[pymethods]
impl PySTrajectory6Iterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(&mut self, py: Python<'a>) -> PyResult<Option<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)>> {
        let traj = self.trajectory.borrow(py);
        if self.index < traj.trajectory.len() {
            let (epoch, state) = traj.trajectory.get(self.index)
                .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
            self.index += 1;
            Ok(Some((
                PyEpoch { obj: epoch },
                state.as_slice().to_pyarray(py).to_owned()
            )))
        } else {
            Ok(None)
        }
    }
}
