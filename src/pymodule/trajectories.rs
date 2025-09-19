/// Python bindings for the new trajectory architecture

/// Python wrapper for OrbitFrame enum
#[pyclass]
#[pyo3(name = "OrbitFrame")]
#[derive(Clone)]
pub struct PyOrbitFrame {
    pub(crate) frame: trajectories::OrbitFrame,
}

#[pymethods]
impl PyOrbitFrame {
    /// ECI frame
    #[classattr]
    fn eci() -> Self {
        PyOrbitFrame { frame: trajectories::OrbitFrame::ECI }
    }

    /// ECEF frame
    #[classattr]
    fn ecef() -> Self {
        PyOrbitFrame { frame: trajectories::OrbitFrame::ECEF }
    }

    /// Get frame name
    fn name(&self) -> &str {
        use crate::trajectories::ReferenceFrame;
        self.frame.name()
    }

    fn __str__(&self) -> String {
        self.name().to_string()
    }

    fn __repr__(&self) -> String {
        format!("OrbitFrame.{}", match self.frame {
            trajectories::OrbitFrame::ECI => "ECI",
            trajectories::OrbitFrame::ECEF => "ECEF",
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
    pub(crate) representation: trajectories::OrbitRepresentation,
}

#[pymethods]
impl PyOrbitRepresentation {
    /// Cartesian representation
    #[classattr]
    fn cartesian() -> Self {
        PyOrbitRepresentation { representation: trajectories::OrbitRepresentation::Cartesian }
    }

    /// Keplerian representation
    #[classattr]
    fn keplerian() -> Self {
        PyOrbitRepresentation { representation: trajectories::OrbitRepresentation::Keplerian }
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

/// Python wrapper for OrbitalTrajectory (unified design)
#[pyclass]
#[pyo3(name = "OrbitalTrajectory")]
pub struct PyOrbitalTrajectory {
    pub(crate) trajectory: trajectories::STrajectory6,
}

#[pymethods]
impl PyOrbitalTrajectory {
    /// Create a new empty orbital trajectory
    ///
    /// Arguments:
    ///     frame (OrbitFrame): Reference frame
    ///     representation (OrbitRepresentation): Orbital representation
    ///     angle_format (AngleFormat): Format for angular quantities
    ///     interpolation_method (InterpolationMethod): Interpolation method
    ///
    /// Returns:
    ///     OrbitalTrajectory: New trajectory instance
    #[new]
    #[pyo3(text_signature = "(frame, representation, angle_format, interpolation_method)")]
    pub fn new(
        frame: PyRef<PyOrbitFrame>,
        representation: PyRef<PyOrbitRepresentation>,
        angle_format: PyRef<PyAngleFormat>,
        interpolation_method: PyRef<PyInterpolationMethod>,
    ) -> PyResult<Self> {
        match trajectories::STrajectory6::new_orbital_trajectory(
            frame.frame,
            representation.representation,
            angle_format.format,
            interpolation_method.method,
        ) {
            Ok(trajectory) => Ok(PyOrbitalTrajectory { trajectory }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Create orbital trajectory from data
    ///
    /// Arguments:
    ///     epochs (list[Epoch]): List of epochs
    ///     states: Flattened array of 6-element state vectors (Nx6 total elements)
    ///     frame (OrbitFrame): Reference frame
    ///     representation (OrbitRepresentation): Orbital representation
    ///     angle_format (AngleFormat): Format for angular quantities
    ///     interpolation_method (InterpolationMethod): Interpolation method
    ///
    /// Returns:
    ///     OrbitalTrajectory: New trajectory instance with data
    #[classmethod]
    #[pyo3(text_signature = "(epochs, states, frame, representation, angle_format, interpolation_method)")]
    pub fn from_orbital_data(
        _cls: &Bound<'_, PyType>,
        epochs: Vec<PyRef<PyEpoch>>,
        states: PyReadonlyArray1<f64>,
        frame: PyRef<PyOrbitFrame>,
        representation: PyRef<PyOrbitRepresentation>,
        angle_format: PyRef<PyAngleFormat>,
        interpolation_method: PyRef<PyInterpolationMethod>,
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

        match trajectories::STrajectory6::from_orbital_data(
            epochs_vec,
            states_vec,
            frame.frame,
            representation.representation,
            angle_format.format,
            interpolation_method.method,
        ) {
            Ok(trajectory) => Ok(PyOrbitalTrajectory { trajectory }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
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
    pub fn add_state(&mut self, epoch: PyRef<PyEpoch>, state: PyReadonlyArray1<f64>) -> PyResult<()> {
        let state_array = state.as_array();
        if state_array.len() != 6 {
            return Err(exceptions::PyValueError::new_err(
                "State vector must have exactly 6 elements"
            ));
        }

        let state_vec = na::Vector6::from_row_slice(state_array.as_slice().unwrap());

        match self.trajectory.add_state(epoch.obj, state_vec) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get state at a specific epoch using interpolation
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     numpy.ndarray: Interpolated state vector
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_at_epoch<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.state_at_epoch(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
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

    /// Get the number of states in the trajectory
    #[getter]
    pub fn length(&self) -> usize {
        self.trajectory.len()
    }

    /// Get trajectory frame
    #[getter]
    pub fn frame(&self) -> PyOrbitFrame {
        use crate::trajectories::OrbitalTrajectory;
        PyOrbitFrame { frame: self.trajectory.orbital_frame() }
    }

    /// Get trajectory representation
    #[getter]
    pub fn representation(&self) -> PyOrbitRepresentation {
        use crate::trajectories::OrbitalTrajectory;
        PyOrbitRepresentation { representation: self.trajectory.orbital_representation() }
    }

    /// Get trajectory angle format
    #[getter]
    pub fn angle_format(&self) -> PyAngleFormat {
        use crate::trajectories::OrbitalTrajectory;
        PyAngleFormat { format: self.trajectory.angle_format() }
    }

    /// Clear all states from the trajectory
    #[pyo3(text_signature = "()")]
    pub fn clear(&mut self) {
        self.trajectory.clear();
    }

    /// Convert trajectory to different frame/representation
    ///
    /// Arguments:
    ///     frame (OrbitFrame): Target reference frame
    ///     representation (OrbitRepresentation): Target representation
    ///     angle_format (AngleFormat): Target angle format
    ///
    /// Returns:
    ///     OrbitalTrajectory: Converted trajectory
    #[pyo3(text_signature = "(frame, representation, angle_format)")]
    pub fn convert_to(
        &self,
        frame: PyRef<PyOrbitFrame>,
        representation: PyRef<PyOrbitRepresentation>,
        angle_format: PyRef<PyAngleFormat>,
    ) -> PyResult<Self> {
        use crate::trajectories::OrbitalTrajectory;
        match self.trajectory.convert_to(frame.frame, representation.representation, angle_format.format) {
            Ok(new_trajectory) => Ok(PyOrbitalTrajectory { trajectory: new_trajectory }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get all epochs as a numpy array
    #[pyo3(text_signature = "()")]
    pub fn epochs<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray<f64, Ix1>> {
        let epochs: Vec<f64> = self.trajectory.epochs().iter().map(|e| e.jd()).collect();
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
    pub fn state_at_index<'a>(&self, py: Python<'a>, index: usize) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        if index >= self.trajectory.len() {
            return Err(exceptions::PyIndexError::new_err("Index out of range"));
        }

        match self.trajectory.to_matrix() {
            Ok(states_matrix) => {
                let state = states_matrix.column(index);
                let state_vec: Vec<f64> = state.iter().cloned().collect();
                Ok(state_vec.to_pyarray(py).to_owned())
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get epoch at specific index
    ///
    /// Arguments:
    ///     index (int): Index of the epoch
    ///
    /// Returns:
    ///     Epoch: Epoch at given index
    #[pyo3(text_signature = "(index)")]
    pub fn epoch_at_index(&self, index: usize) -> PyResult<PyEpoch> {
        let epochs = self.trajectory.epochs();
        if index >= epochs.len() {
            return Err(exceptions::PyIndexError::new_err("Index out of range"));
        }
        Ok(PyEpoch { obj: epochs[index] })
    }

    /// Convert to Cartesian representation
    ///
    /// Returns:
    ///     OrbitalTrajectory: Trajectory in Cartesian representation
    #[pyo3(text_signature = "()")]
    pub fn to_cartesian(&self) -> PyResult<Self> {
        use crate::trajectories::OrbitalTrajectory;
        match self.trajectory.to_cartesian() {
            Ok(new_trajectory) => Ok(PyOrbitalTrajectory { trajectory: new_trajectory }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert to Keplerian representation
    ///
    /// Arguments:
    ///     angle_format (AngleFormat): Angle format for the result
    ///
    /// Returns:
    ///     OrbitalTrajectory: Trajectory in Keplerian representation
    #[pyo3(text_signature = "(angle_format)")]
    pub fn to_keplerian(&self, angle_format: PyRef<PyAngleFormat>) -> PyResult<Self> {
        use crate::trajectories::OrbitalTrajectory;
        match self.trajectory.to_keplerian(angle_format.format) {
            Ok(new_trajectory) => Ok(PyOrbitalTrajectory { trajectory: new_trajectory }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert to different reference frame
    ///
    /// Arguments:
    ///     frame (OrbitFrame): Target reference frame
    ///
    /// Returns:
    ///     OrbitalTrajectory: Trajectory in new reference frame
    #[pyo3(text_signature = "(frame)")]
    pub fn to_frame(&self, frame: PyRef<PyOrbitFrame>) -> PyResult<Self> {
        use crate::trajectories::OrbitalTrajectory;
        match self.trajectory.to_frame(frame.frame) {
            Ok(new_trajectory) => Ok(PyOrbitalTrajectory { trajectory: new_trajectory }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert to different angle format
    ///
    /// Arguments:
    ///     angle_format (AngleFormat): Target angle format
    ///
    /// Returns:
    ///     OrbitalTrajectory: Trajectory in new angle format
    #[pyo3(text_signature = "(angle_format)")]
    pub fn to_angle_format(&self, angle_format: PyRef<PyAngleFormat>) -> PyResult<Self> {
        use crate::trajectories::OrbitalTrajectory;
        match self.trajectory.to_angle_format(angle_format.format) {
            Ok(new_trajectory) => Ok(PyOrbitalTrajectory { trajectory: new_trajectory }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert to ECI (Earth-Centered Inertial) frame
    ///
    /// Returns:
    ///     OrbitalTrajectory: Trajectory in ECI frame
    #[pyo3(text_signature = "()")]
    pub fn to_eci(&self) -> PyResult<Self> {
        use crate::trajectories::OrbitalTrajectory;
        match self.trajectory.to_eci() {
            Ok(new_trajectory) => Ok(PyOrbitalTrajectory { trajectory: new_trajectory }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert to ECEF (Earth-Centered Earth-Fixed) frame
    ///
    /// Returns:
    ///     OrbitalTrajectory: Trajectory in ECEF frame
    #[pyo3(text_signature = "()")]
    pub fn to_ecef(&self) -> PyResult<Self> {
        use crate::trajectories::OrbitalTrajectory;
        match self.trajectory.to_ecef() {
            Ok(new_trajectory) => Ok(PyOrbitalTrajectory { trajectory: new_trajectory }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert to specific representation
    ///
    /// Arguments:
    ///     representation (OrbitRepresentation): Target representation
    ///     angle_format (AngleFormat): Target angle format
    ///
    /// Returns:
    ///     OrbitalTrajectory: Trajectory in new representation
    #[pyo3(text_signature = "(representation, angle_format)")]
    pub fn to_representation(&self, representation: PyRef<PyOrbitRepresentation>, angle_format: PyRef<PyAngleFormat>) -> PyResult<Self> {
        use crate::trajectories::OrbitalTrajectory;
        match self.trajectory.to_representation(representation.representation, angle_format.format) {
            Ok(new_trajectory) => Ok(PyOrbitalTrajectory { trajectory: new_trajectory }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert angles to degrees
    ///
    /// Returns:
    ///     OrbitalTrajectory: Trajectory with angles in degrees
    #[pyo3(text_signature = "()")]
    pub fn to_degrees(&self) -> PyResult<Self> {
        use crate::trajectories::OrbitalTrajectory;
        match self.trajectory.to_degrees() {
            Ok(new_trajectory) => Ok(PyOrbitalTrajectory { trajectory: new_trajectory }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert angles to radians
    ///
    /// Returns:
    ///     OrbitalTrajectory: Trajectory with angles in radians
    #[pyo3(text_signature = "()")]
    pub fn to_radians(&self) -> PyResult<Self> {
        use crate::trajectories::OrbitalTrajectory;
        match self.trajectory.to_radians() {
            Ok(new_trajectory) => Ok(PyOrbitalTrajectory { trajectory: new_trajectory }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get position component at specific epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Epoch at which to get position
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element position vector [x, y, z] in km
    #[pyo3(text_signature = "(epoch)")]
    pub fn position_at_epoch<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        use crate::trajectories::OrbitalTrajectory;
        match self.trajectory.position_at_epoch(&epoch.obj) {
            Ok(position) => Ok(position.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get velocity component at specific epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Epoch at which to get velocity
    ///
    /// Returns:
    ///     numpy.ndarray: 3-element velocity vector [vx, vy, vz] in km/s
    #[pyo3(text_signature = "(epoch)")]
    pub fn velocity_at_epoch<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        use crate::trajectories::OrbitalTrajectory;
        match self.trajectory.velocity_at_epoch(&epoch.obj) {
            Ok(velocity) => Ok(velocity.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert trajectory to matrix representation
    ///
    /// Returns:
    ///     numpy.ndarray: Matrix with states as rows
    #[pyo3(text_signature = "()")]
    pub fn to_matrix<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyArray<f64, numpy::Ix2>>> {
        match self.trajectory.to_matrix() {
            Ok(states_matrix) => {
                let data: Vec<f64> = states_matrix.iter().cloned().collect();
                let shape = (states_matrix.nrows(), states_matrix.ncols());
                Ok(numpy::PyArray::from_vec(py, data).reshape(shape).unwrap().to_owned())
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Convert trajectory to JSON representation
    ///
    /// Returns:
    ///     str: JSON string representation
    #[pyo3(text_signature = "()")]
    pub fn to_json(&self) -> PyResult<String> {
        // Create a simple JSON representation
        let epochs: Vec<f64> = self.trajectory.epochs().iter().map(|e| e.jd()).collect();

        match self.trajectory.to_matrix() {
            Ok(states_matrix) => {
                let states: Vec<Vec<f64>> = (0..states_matrix.nrows())
                    .map(|i| states_matrix.row(i).iter().cloned().collect())
                    .collect();

                use crate::trajectories::OrbitalTrajectory;
                let json_obj = serde_json::json!({
                    "frame": format!("{:?}", self.trajectory.orbital_frame()),
                    "representation": format!("{:?}", self.trajectory.orbital_representation()),
                    "angle_format": format!("{:?}", self.trajectory.angle_format()),
                    "epochs": epochs,
                    "states": states
                });

                match serde_json::to_string(&json_obj) {
                    Ok(json_str) => Ok(json_str),
                    Err(e) => Err(exceptions::PyRuntimeError::new_err(format!("JSON serialization error: {}", e))),
                }
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get current state vector (most recent state in trajectory)
    #[pyo3(text_signature = "()")]
    pub fn current_state_vector<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let current_state = self.trajectory.current_state_vector();
        Ok(current_state.as_slice().to_pyarray(py).to_owned())
    }

    /// Get current epoch (most recent epoch in trajectory)
    #[pyo3(text_signature = "()")]
    pub fn current_epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.trajectory.current_epoch() }
    }

    /// Convert state between different coordinate frames and representations
    ///
    /// Arguments:
    ///     state: State vector to convert
    ///     epoch: Epoch of the state
    ///     from_frame: Source reference frame
    ///     from_representation: Source representation
    ///     from_angle_format: Source angle format
    ///     to_frame: Target reference frame
    ///     to_representation: Target representation
    ///     to_angle_format: Target angle format
    ///
    /// Returns:
    ///     numpy.ndarray: Converted state vector
    #[pyo3(text_signature = "(state, epoch, from_frame, from_representation, from_angle_format, to_frame, to_representation, to_angle_format)")]
    pub fn convert_state_to_format<'a>(
        &self,
        py: Python<'a>,
        state: PyReadonlyArray1<f64>,
        epoch: PyRef<PyEpoch>,
        from_frame: PyRef<PyOrbitFrame>,
        from_representation: PyRef<PyOrbitRepresentation>,
        from_angle_format: PyRef<PyAngleFormat>,
        to_frame: PyRef<PyOrbitFrame>,
        to_representation: PyRef<PyOrbitRepresentation>,
        to_angle_format: PyRef<PyAngleFormat>,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state_slice = state.as_slice()?;
        if state_slice.len() != 6 {
            return Err(exceptions::PyValueError::new_err("State vector must have 6 elements"));
        }

        let state_vector = na::SVector::<f64, 6>::from_column_slice(state_slice);

        match self.trajectory.convert_state_to_format(
            state_vector,
            epoch.obj,
            from_frame.frame,
            from_representation.representation,
            from_angle_format.format,
            to_frame.frame,
            to_representation.representation,
            to_angle_format.format,
        ) {
            Ok(converted_state) => Ok(converted_state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Python length
    fn __len__(&self) -> usize {
        self.trajectory.len()
    }

    /// Python indexing support
    fn __getitem__(&self, index: isize) -> PyResult<PyEpoch> {
        let len = self.trajectory.len() as isize;
        let idx = if index < 0 { len + index } else { index };

        if idx < 0 || idx >= len {
            return Err(exceptions::PyIndexError::new_err("Index out of range"));
        }

        let epochs = self.trajectory.epochs();
        Ok(PyEpoch { obj: epochs[idx as usize] })
    }

    /// String representation
    fn __repr__(&self) -> String {
        use crate::trajectories::OrbitalTrajectory;
        format!(
            "OrbitalTrajectory(frame={:?}, representation={:?}, states={})",
            self.trajectory.orbital_frame(), self.trajectory.orbital_representation(), self.trajectory.len()
        )
    }

    /// String conversion
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Legacy trajectory classes for backward compatibility

/// Python wrapper for OrbitStateType enum (legacy)
#[pyclass]
#[pyo3(name = "OrbitStateType")]
#[derive(Clone)]
pub struct PyOrbitStateType {
    state_type: String,
}

#[pymethods]
impl PyOrbitStateType {
    #[classattr]
    fn cartesian() -> Self {
        PyOrbitStateType { state_type: "Cartesian".to_string() }
    }

    #[classattr]
    fn keplerian() -> Self {
        PyOrbitStateType { state_type: "Keplerian".to_string() }
    }

    fn __str__(&self) -> String {
        self.state_type.clone()
    }

    fn __repr__(&self) -> String {
        format!("OrbitStateType.{}", self.state_type)
    }
}

/// Python wrapper for AngleFormat enum
#[pyclass]
#[pyo3(name = "AngleFormat")]
#[derive(Clone)]
pub struct PyAngleFormat {
    pub(crate) format: trajectories::AngleFormat,
}

#[pymethods]
impl PyAngleFormat {
    #[classattr]
    fn radians() -> Self {
        PyAngleFormat { format: trajectories::AngleFormat::Radians }
    }

    #[classattr]
    fn degrees() -> Self {
        PyAngleFormat { format: trajectories::AngleFormat::Degrees }
    }

    #[classattr]
    fn none() -> Self {
        PyAngleFormat { format: trajectories::AngleFormat::None }
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
    pub(crate) method: trajectories::InterpolationMethod,
}

#[pymethods]
impl PyInterpolationMethod {
    #[classattr]
    fn linear() -> Self {
        PyInterpolationMethod { method: trajectories::InterpolationMethod::Linear }
    }

    #[classattr]
    fn lagrange() -> Self {
        PyInterpolationMethod { method: trajectories::InterpolationMethod::Lagrange }
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


/// Python wrapper for dynamic Trajectory class
#[pyclass]
#[pyo3(name = "Trajectory")]
pub struct PyTrajectory {
    pub(crate) trajectory: trajectories::Trajectory,
}

#[pymethods]
impl PyTrajectory {
    /// Create a new empty trajectory
    ///
    /// Arguments:
    ///     dimension (int): Trajectory dimension (default: 6)
    ///         OR interpolation_method (InterpolationMethod): For backward compatibility
    ///     interpolation_method (InterpolationMethod): Interpolation method (default: Linear)
    ///
    /// Returns:
    ///     Trajectory: New trajectory instance
    ///
    /// Examples:
    ///     Trajectory()                           # 6D, Linear
    ///     Trajectory(7)                         # 7D, Linear
    ///     Trajectory(12)                        # 12D, Linear
    ///     Trajectory(InterpolationMethod.lagrange)  # 6D, Lagrange (backward compatibility)
    #[new]
    #[pyo3(signature = (*args, **kwargs))]
    pub fn new(args: &Bound<'_, pyo3::types::PyTuple>, kwargs: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
        // Handle different calling patterns for backward compatibility
        let (dimension, method) = if args.len() == 0 {
            // Default case: Trajectory()
            (6, trajectories::InterpolationMethod::Linear)
        } else if args.len() == 1 {
            // Check if first argument is dimension (int) or interpolation_method
            let first_arg = args.get_item(0)?;
            if let Ok(dim) = first_arg.extract::<usize>() {
                // New API: Trajectory(dimension)
                if dim == 0 {
                    return Err(exceptions::PyValueError::new_err(
                        "Trajectory dimension must be greater than 0"
                    ));
                }
                (dim, trajectories::InterpolationMethod::Linear)
            } else if let Ok(interp_method) = first_arg.extract::<PyRef<PyInterpolationMethod>>() {
                // Old API: Trajectory(interpolation_method) - backward compatibility
                (6, interp_method.method)
            } else {
                return Err(exceptions::PyTypeError::new_err(
                    "First argument must be either dimension (int) or interpolation_method"
                ));
            }
        } else if args.len() == 2 {
            // New API: Trajectory(dimension, interpolation_method)
            let dimension = args.get_item(0)?.extract::<usize>()?;
            if dimension == 0 {
                return Err(exceptions::PyValueError::new_err(
                    "Trajectory dimension must be greater than 0"
                ));
            }
            let interp_method = args.get_item(1)?.extract::<PyRef<PyInterpolationMethod>>()?;
            (dimension, interp_method.method)
        } else {
            return Err(exceptions::PyTypeError::new_err(
                "Too many positional arguments"
            ));
        };

        // Handle keyword arguments
        let final_method = if let Some(kwargs) = kwargs {
            if let Some(interp_arg) = kwargs.get_item("interpolation_method")? {
                interp_arg.extract::<PyRef<PyInterpolationMethod>>()?.method
            } else {
                method
            }
        } else {
            method
        };

        let trajectory = trajectories::Trajectory::with_interpolation(dimension, final_method);

        Ok(PyTrajectory { trajectory })
    }

    /// Create a trajectory from vectors of epochs and states
    ///
    /// Arguments:
    ///     epochs (list[Epoch]): List of time epochs
    ///     states (numpy.ndarray): Array of 6D state vectors (Nx6)
    ///     interpolation_method (InterpolationMethod): Interpolation method (default: Linear)
    ///
    /// Returns:
    ///     Trajectory: New trajectory instance
    #[classmethod]
    #[pyo3(signature = (epochs, states, interpolation_method=None))]
    pub fn from_data(
        _cls: &Bound<'_, PyType>,
        epochs: Vec<PyRef<PyEpoch>>,
        states: PyReadonlyArray1<f64>,
        interpolation_method: Option<PyRef<PyInterpolationMethod>>,
    ) -> PyResult<Self> {
        let method = interpolation_method
            .map(|m| m.method)
            .unwrap_or(trajectories::InterpolationMethod::Linear);

        let epochs_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states_array = states.as_array();

        // Auto-detect dimension from data
        let num_epochs = epochs_vec.len();
        if num_epochs == 0 {
            return Err(exceptions::PyValueError::new_err(
                "At least one epoch is required"
            ));
        }

        let dimension = states_array.len() / num_epochs;
        if states_array.len() % num_epochs != 0 {
            return Err(exceptions::PyValueError::new_err(
                "States array length must be evenly divisible by number of epochs"
            ));
        }

        let mut trajectory = trajectories::Trajectory::with_interpolation(dimension, method);

        for i in 0..num_epochs {
            let start_idx = i * dimension;
            let state_slice = &states_array.as_slice().unwrap()[start_idx..start_idx + dimension];
            let state_vec = na::DVector::from_column_slice(state_slice);

            if let Err(e) = trajectory.add_state(epochs_vec[i], state_vec) {
                return Err(exceptions::PyRuntimeError::new_err(e.to_string()));
            }
        }

        Ok(PyTrajectory { trajectory })
    }

    /// Get the trajectory dimension
    #[getter]
    pub fn dimension(&self) -> usize {
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
    pub fn add_state(&mut self, epoch: PyRef<PyEpoch>, state: PyReadonlyArray1<f64>) -> PyResult<()> {
        let state_array = state.as_array();
        if state_array.len() != self.trajectory.dimension {
            return Err(exceptions::PyValueError::new_err(
                format!("State vector must have exactly {} elements for {}D trajectory",
                    self.trajectory.dimension, self.trajectory.dimension)
            ));
        }

        let state_vec = na::DVector::from_column_slice(state_array.as_slice().unwrap());
        match self.trajectory.add_state(epoch.obj, state_vec) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get state at a specific epoch using interpolation
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     numpy.ndarray: Interpolated state vector
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_at_epoch<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.state_at_epoch(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
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

    /// Get state at a specific index
    ///
    /// Arguments:
    ///     index (int): Index of the state
    ///
    /// Returns:
    ///     numpy.ndarray: State vector at index
    #[pyo3(text_signature = "(index)")]
    pub fn state_at_index<'a>(&self, py: Python<'a>, index: usize) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.state_at_index(index) {
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
    pub fn epoch_at_index(&self, index: usize) -> PyResult<PyEpoch> {
        match self.trajectory.epoch_at_index(index) {
            Ok(epoch) => Ok(PyEpoch { obj: epoch }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the number of states in the trajectory
    #[getter]
    pub fn length(&self) -> usize {
        self.trajectory.len()
    }

    /// Get interpolation method
    #[getter]
    pub fn interpolation_method(&self) -> PyInterpolationMethod {
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
    pub fn set_max_size(&mut self, max_size: usize) -> PyResult<()> {
        match self.trajectory.set_eviction_policy_max_size(max_size) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Set maximum age for trajectory states (in seconds)
    #[pyo3(text_signature = "(max_age)")]
    pub fn set_max_age(&mut self, max_age: f64) -> PyResult<()> {
        match self.trajectory.set_eviction_policy_max_age(max_age) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
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
                let data: Vec<f64> = states_matrix.iter().cloned().collect();
                let shape = (states_matrix.nrows(), states_matrix.ncols());
                Ok(numpy::PyArray::from_vec(py, data).reshape(shape).unwrap().to_owned())
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
}

