/// Python bindings for trajectory traits and orbital trajectory.
// Import traits needed by trajectory metho
/// Interpolation method for trajectory state estimation.
///
/// Specifies the algorithm used to estimate states at epochs between
/// discrete trajectory points.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "InterpolationMethod")]
#[derive(Clone)]
pub struct PyInterpolationMethod {
    pub(crate) method: trajectories::traits::InterpolationMethod,
}

#[pymethods]
impl PyInterpolationMethod {
    /// Linear interpolation method.
    ///
    /// Returns:
    ///     InterpolationMethod: Linear interpolation constant
    #[classattr]
    #[pyo3(name = "LINEAR")]
    fn linear() -> Self {
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


/// Reference frame for orbital trajectory representation.
///
/// Specifies the coordinate reference frame for position and velocity states.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OrbitFrame")]
#[derive(Clone)]
pub struct PyOrbitFrame {
    pub(crate) frame: trajectories::traits::OrbitFrame,
}

#[pymethods]
impl PyOrbitFrame {
    /// Earth-Centered Inertial (J2000) frame.
    ///
    /// Returns:
    ///     OrbitFrame: ECI frame constant
    #[classattr]
    #[pyo3(name = "ECI")]
    fn eci() -> Self {
        PyOrbitFrame { frame: trajectories::traits::OrbitFrame::ECI }
    }

    /// Earth-Centered Earth-Fixed frame.
    ///
    /// Returns:
    ///     OrbitFrame: ECEF frame constant
    #[classattr]
    #[pyo3(name = "ECEF")]
    fn ecef() -> Self {
        PyOrbitFrame { frame: trajectories::traits::OrbitFrame::ECEF }
    }

    /// Get the full name of the reference frame.
    ///
    /// Returns:
    ///     str: Human-readable frame name
    fn name(&self) -> &str {
        match self.frame {
            trajectories::traits::OrbitFrame::ECI => "Earth-Centered Inertial",
            trajectories::traits::OrbitFrame::ECEF => "Earth-Centered Earth-Fixed",
        }
    }

    fn __str__(&self) -> String {
        match self.frame {
            trajectories::traits::OrbitFrame::ECI => "ECI".to_string(),
            trajectories::traits::OrbitFrame::ECEF => "ECEF".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("OrbitFrame({})", self.name())
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.frame == other.frame),
            CompareOp::Ne => Ok(self.frame != other.frame),
            _ => Err(exceptions::PyNotImplementedError::new_err("Comparison not supported")),
        }
    }
}

/// Orbital state representation format.
///
/// Specifies how orbital states are represented in the trajectory.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OrbitRepresentation")]
#[derive(Clone)]
pub struct PyOrbitRepresentation {
    pub(crate) representation: trajectories::traits::OrbitRepresentation,
}

#[pymethods]
impl PyOrbitRepresentation {
    /// Cartesian position and velocity representation.
    ///
    /// States are represented as [x, y, z, vx, vy, vz] in meters and meters/second.
    ///
    /// Returns:
    ///     OrbitRepresentation: Cartesian representation constant
    #[classattr]
    #[pyo3(name = "CARTESIAN")]
    fn cartesian() -> Self {
        PyOrbitRepresentation { representation: trajectories::traits::OrbitRepresentation::Cartesian }
    }

    /// Keplerian orbital elements representation.
    ///
    /// States are represented as [a, e, i, raan, argp, nu] where angles are
    /// in radians or degrees depending on the angle format.
    ///
    /// Returns:
    ///     OrbitRepresentation: Keplerian representation constant
    #[classattr]
    #[pyo3(name = "KEPLERIAN")]
    fn keplerian() -> Self {
        PyOrbitRepresentation { representation: trajectories::traits::OrbitRepresentation::Keplerian }
    }

    fn __str__(&self) -> String {
        match self.representation {
            trajectories::traits::OrbitRepresentation::Cartesian => "Cartesian".to_string(),
            trajectories::traits::OrbitRepresentation::Keplerian => "Keplerian".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        format!("OrbitRepresentation({})", match self.representation {
            trajectories::traits::OrbitRepresentation::Cartesian => "Cartesian",
            trajectories::traits::OrbitRepresentation::Keplerian => "Keplerian",
        })
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.representation == other.representation),
            CompareOp::Ne => Ok(self.representation != other.representation),
            _ => Err(exceptions::PyNotImplementedError::new_err("Comparison not supported")),
        }
    }
}

/// Orbital trajectory with frame and representation awareness.
///
/// Stores a sequence of orbital states at specific epochs with support for
/// interpolation, frame conversions, and representation transformations.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OrbitTrajectory")]
pub struct PyOrbitalTrajectory {
    pub(crate) trajectory: trajectories::OrbitTrajectory,
}

#[pymethods]
impl PyOrbitalTrajectory {
    /// Create a new empty orbital trajectory.
    ///
    /// Args:
    ///     frame (OrbitFrame): Reference frame for the trajectory
    ///     representation (OrbitRepresentation): State representation format
    ///     angle_format (AngleFormat or None): Angle format for Keplerian states,
    ///         must be None for Cartesian representation
    ///
    /// Returns:
    ///     OrbitTrajectory: New empty trajectory instance
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     # Create trajectory in ECI Cartesian frame
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///
    ///     # Define Keplerian elements for a 500 km circular orbit
    ///     oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])  # a, e, i, raan, argp, M
    ///     state_cart = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
    ///
    ///     # Add states to trajectory
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     traj.add(epc, state_cart)
    ///     traj.add(epc + 60.0, state_cart)  # Add another state 60 seconds later
    ///     print(f"Trajectory has {traj.len()} states")
    ///     ```
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

    /// Create a default empty orbital trajectory (ECI Cartesian).
    ///
    /// Returns:
    ///     OrbitTrajectory: New trajectory with ECI frame and Cartesian representation
    #[classmethod]
    #[pyo3(text_signature = "()")]
    pub fn default(_cls: &Bound<'_, PyType>) -> Self {
        PyOrbitalTrajectory {
            trajectory: trajectories::OrbitTrajectory::default(),
        }
    }

    /// Create orbital trajectory from existing data.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of time epochs for each state
    ///     states (numpy.ndarray): 2D array of 6-element state vectors with shape (N, 6)
    ///         where N is the number of epochs. Each row is one state vector.
    ///     frame (OrbitFrame): Reference frame for the states
    ///     representation (OrbitRepresentation): State representation format
    ///     angle_format (AngleFormat or None): Angle format for Keplerian states,
    ///         must be None for Cartesian representation
    ///
    /// Returns:
    ///     OrbitTrajectory: New trajectory instance populated with data
    #[classmethod]
    #[pyo3(signature = (epochs, states, frame, representation, angle_format=None), text_signature = "(epochs, states, frame, representation, angle_format=None)")]
    pub fn from_orbital_data(
        _cls: &Bound<'_, PyType>,
        epochs: Vec<PyRef<PyEpoch>>,
        states: PyReadonlyArray2<f64>,
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

        // Check that state dimension is 6
        if states_array.ncols() != 6 {
            return Err(exceptions::PyValueError::new_err(
                format!("State dimension must be 6, got {}", states_array.ncols())
            ));
        }

        let mut states_vec = Vec::new();
        for i in 0..num_epochs {
            let state_row = states_array.row(i);
            let state_vec = na::Vector6::from_iterator(state_row.iter().copied());
            states_vec.push(state_vec);
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

    /// Set interpolation method using builder pattern.
    ///
    /// Args:
    ///     interpolation_method (InterpolationMethod): Interpolation method to use
    ///
    /// Returns:
    ///     OrbitTrajectory: Self with updated interpolation method
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     traj = traj.with_interpolation_method(bh.InterpolationMethod.LINEAR)
    ///     ```
    #[pyo3(text_signature = "(interpolation_method)")]
    pub fn with_interpolation_method(mut slf: PyRefMut<'_, Self>, method: PyRef<PyInterpolationMethod>) -> Self {
        slf.trajectory = slf.trajectory.clone().with_interpolation_method(method.method);
        Self { trajectory: slf.trajectory.clone() }
    }

    /// Set eviction policy to keep maximum number of states using builder pattern.
    ///
    /// Args:
    ///     max_size (int): Maximum number of states to retain
    ///
    /// Returns:
    ///     OrbitTrajectory: Self with updated eviction policy
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     traj = traj.with_eviction_policy_max_size(1000)
    ///     ```
    #[pyo3(text_signature = "(max_size)")]
    pub fn with_eviction_policy_max_size(mut slf: PyRefMut<'_, Self>, max_size: usize) -> Self {
        slf.trajectory = slf.trajectory.clone().with_eviction_policy_max_size(max_size);
        Self { trajectory: slf.trajectory.clone() }
    }

    /// Set eviction policy to keep states within maximum age using builder pattern.
    ///
    /// Args:
    ///     max_age (float): Maximum age of states in seconds
    ///
    /// Returns:
    ///     OrbitTrajectory: Self with updated eviction policy
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     traj = traj.with_eviction_policy_max_age(3600.0)
    ///     ```
    #[pyo3(text_signature = "(max_age)")]
    pub fn with_eviction_policy_max_age(mut slf: PyRefMut<'_, Self>, max_age: f64) -> Self {
        slf.trajectory = slf.trajectory.clone().with_eviction_policy_max_age(max_age);
        Self { trajectory: slf.trajectory.clone() }
    }

    /// Get trajectory dimension (always 6 for orbital trajectories).
    ///
    /// Returns:
    ///     int: Dimension of the trajectory (always 6)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     print(f"Dimension: {traj.dimension()}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn dimension(&self) -> usize {
        6
    }

    /// Add a state to the trajectory.
    ///
    /// Args:
    ///     epoch (Epoch): Time of the state
    ///     state (numpy.ndarray): 6-element state vector
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     ```
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

    /// Get the nearest state to a given epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: Tuple of (Epoch, numpy.ndarray) containing the nearest state
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 30.0, 0.0, bh.TimeSystem.UTC)
    ///     nearest_epc, nearest_state = traj.nearest_state(epc2)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn nearest_state<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.nearest_state(&epoch.obj) {
            Ok((nearest_epoch, nearest_state)) => {
                Ok((PyEpoch { obj: nearest_epoch }, nearest_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the index of the state at or before the given epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     int: Index of the state at or before the target epoch
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     index = traj.index_before_epoch(epc2)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn index_before_epoch(&self, epoch: PyRef<PyEpoch>) -> PyResult<usize> {
        match self.trajectory.index_before_epoch(&epoch.obj) {
            Ok(index) => Ok(index),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the index of the state at or after the given epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     int: Index of the state at or after the target epoch
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 11, 59, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     index = traj.index_after_epoch(epc2)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn index_after_epoch(&self, epoch: PyRef<PyEpoch>) -> PyResult<usize> {
        match self.trajectory.index_after_epoch(&epoch.obj) {
            Ok(index) => Ok(index),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the state at or before the given epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: Tuple of (Epoch, numpy.ndarray) containing state at or before the target epoch
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     ret_epc, ret_state = traj.state_before_epoch(epc2)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_before_epoch<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.state_before_epoch(&epoch.obj) {
            Ok((ret_epoch, ret_state)) => {
                Ok((PyEpoch { obj: ret_epoch }, ret_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the state at or after the given epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: Tuple of (Epoch, numpy.ndarray) containing state at or after the target epoch
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 11, 59, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     ret_epc, ret_state = traj.state_after_epoch(epc2)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_after_epoch<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.state_after_epoch(&epoch.obj) {
            Ok((ret_epoch, ret_state)) => {
                Ok((PyEpoch { obj: ret_epoch }, ret_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Set the interpolation method for the trajectory.
    ///
    /// Args:
    ///     method (InterpolationMethod): New interpolation method
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     traj.set_interpolation_method(bh.InterpolationMethod.LINEAR)
    ///     ```
    #[pyo3(text_signature = "(method)")]
    pub fn set_interpolation_method(&mut self, method: PyRef<PyInterpolationMethod>) {
        self.trajectory.set_interpolation_method(method.method);
    }

    /// Get the current interpolation method.
    ///
    /// Returns:
    ///     InterpolationMethod: Current interpolation method
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     method = traj.get_interpolation_method()
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn get_interpolation_method(&self) -> PyInterpolationMethod {
        PyInterpolationMethod { method: self.trajectory.get_interpolation_method() }
    }

    /// Interpolate state at a given epoch using linear interpolation.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     numpy.ndarray: Linearly interpolated state vector
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state1 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state1)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 2, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state2 = np.array([bh.R_EARTH + 510e3, 0.0, 0.0, 0.0, 7650.0, 0.0])
    ///     traj.add(epc2, state2)
    ///     epc_mid = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state_interp = traj.interpolate_linear(epc_mid)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn interpolate_linear<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.interpolate_linear(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Interpolate state at a given epoch using the configured interpolation method.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     numpy.ndarray: Interpolated state vector
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state1 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state1)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 2, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state2 = np.array([bh.R_EARTH + 510e3, 0.0, 0.0, 0.0, 7650.0, 0.0])
    ///     traj.add(epc2, state2)
    ///     epc_mid = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state_interp = traj.interpolate(epc_mid)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn interpolate<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.interpolate(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the number of states in the trajectory.
    ///
    /// Returns:
    ///     int: Number of states in the trajectory
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.DTrajectory(6)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     print(f"Trajectory length: {traj.length}")
    ///     ```
    #[getter]
    pub fn length(&self) -> usize {
        self.trajectory.len()
    }

    /// Get the number of states in the trajectory (alias for length).
    ///
    /// Returns:
    ///     int: Number of states in the trajectory
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.DTrajectory(6)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     print(f"Number of states: {traj.len()}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn len(&self) -> usize {
        self.trajectory.len()
    }

    /// Get trajectory reference frame.
    ///
    /// Returns:
    ///     OrbitFrame: Reference frame of the trajectory
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     print(f"Frame: {traj.frame}")
    ///     ```
    #[getter]
    pub fn frame(&self) -> PyOrbitFrame {
        PyOrbitFrame { frame: self.trajectory.frame }
    }

    /// Get trajectory state representation.
    ///
    /// Returns:
    ///     OrbitRepresentation: State representation format of the trajectory
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     print(f"Representation: {traj.representation}")
    ///     ```
    #[getter]
    pub fn representation(&self) -> PyOrbitRepresentation {
        PyOrbitRepresentation { representation: self.trajectory.representation }
    }

    /// Get trajectory angle format for Keplerian states.
    ///
    /// Returns:
    ///     AngleFormat or None: Angle format for Keplerian representation, None for Cartesian
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     print(f"Angle format: {traj.angle_format}")
    ///     ```
    #[getter]
    pub fn angle_format(&self) -> Option<PyAngleFormat> {
        self.trajectory.angle_format.map(|af| PyAngleFormat { value: af })
    }

    /// Clear all states from the trajectory.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     traj.clear()
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn clear(&mut self) {
        self.trajectory.clear();
    }

    /// Get start epoch of trajectory.
    ///
    /// Returns:
    ///     Epoch or None: First epoch if trajectory is not empty, None otherwise
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     print(f"Start epoch: {traj.start_epoch()}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn start_epoch(&self) -> Option<PyEpoch> {
        self.trajectory.start_epoch().map(|epoch| PyEpoch { obj: epoch })
    }

    /// Get end epoch of trajectory.
    ///
    /// Returns:
    ///     Epoch or None: Last epoch if trajectory is not empty, None otherwise
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     print(f"End epoch: {traj.end_epoch()}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn end_epoch(&self) -> Option<PyEpoch> {
        self.trajectory.end_epoch().map(|epoch| PyEpoch { obj: epoch })
    }

    /// Get time span of trajectory in seconds.
    ///
    /// Returns:
    ///     float or None: Time span between first and last epochs, or None if less than 2 states
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     traj.add(epc + 3600.0, state)
    ///     print(f"Timespan: {traj.timespan()} seconds")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn timespan(&self) -> Option<f64> {
        self.trajectory.timespan()
    }

    /// Get the first (epoch, state) tuple in the trajectory, if any exists.
    ///
    /// Returns:
    ///     tuple or None: Tuple of (Epoch, numpy.ndarray) for first state, or None if empty
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     first_epc, first_state = traj.first()
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn first<'a>(&self, py: Python<'a>) -> Option<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        self.trajectory.first().map(|(epoch, state)| {
            (PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned())
        })
    }

    /// Get the last (epoch, state) tuple in the trajectory, if any exists.
    ///
    /// Returns:
    ///     tuple or None: Tuple of (Epoch, numpy.ndarray) for last state, or None if empty
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     last_epc, last_state = traj.last()
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn last<'a>(&self, py: Python<'a>) -> Option<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        self.trajectory.last().map(|(epoch, state)| {
            (PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned())
        })
    }

    /// Get all epochs as a list of Epoch objects.
    ///
    /// Returns:
    ///     list[Epoch]: List of Epoch objects for all trajectory points
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     traj.add(epc + 60.0, state)
    ///     epochs_list = traj.epochs()
    ///     print(f"First epoch: {epochs_list[0]}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn epochs(&self) -> Vec<PyEpoch> {
        self.trajectory.epochs.iter().map(|e| PyEpoch { obj: *e }).collect()
    }

    /// Get all states as a numpy array.
    ///
    /// Returns:
    ///     numpy.ndarray: 2D array of states with shape (N, 6) where N is the number of states
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     traj.add(epc + 60.0, state)
    ///     states_array = traj.states()
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn states<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyArray<f64, numpy::Ix2>>> {
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

    /// Check if trajectory is empty.
    ///
    /// Returns:
    ///     bool: True if trajectory contains no states, False otherwise
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     print(f"Is empty: {traj.is_empty()}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn is_empty(&self) -> bool {
        self.trajectory.len() == 0
    }

    /// Convert to ECI (Earth-Centered Inertial) frame in Cartesian representation.
    ///
    /// Returns:
    ///     OrbitTrajectory: Trajectory in ECI Cartesian frame
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECEF, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0])
    ///     traj.add(epc, state)
    ///     traj_eci = traj.to_eci()
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn to_eci(&self) -> Self {
        let new_trajectory = self.trajectory.to_eci();
        PyOrbitalTrajectory { trajectory: new_trajectory }
    }

    /// Convert to ECEF (Earth-Centered Earth-Fixed) frame in Cartesian representation.
    ///
    /// Returns:
    ///     OrbitTrajectory: Trajectory in ECEF Cartesian frame
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     traj_ecef = traj.to_ecef()
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn to_ecef(&self) -> Self {
        let new_trajectory = self.trajectory.to_ecef();
        PyOrbitalTrajectory { trajectory: new_trajectory }
    }

    /// Convert to Keplerian representation in ECI frame.
    ///
    /// Args:
    ///     angle_format (AngleFormat): Angle format for the result (Radians or Degrees)
    ///
    /// Returns:
    ///     OrbitTrajectory: Trajectory in ECI Keplerian representation
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     traj_kep = traj.to_keplerian(bh.AngleFormat.RADIANS)
    ///     ```
    #[pyo3(text_signature = "(angle_format)")]
    pub fn to_keplerian(&self, angle_format: PyRef<PyAngleFormat>) -> Self {
        let new_trajectory = self.trajectory.to_keplerian(angle_format.value);
        PyOrbitalTrajectory { trajectory: new_trajectory }
    }

    /// Convert trajectory to matrix representation.
    ///
    /// Returns:
    ///     numpy.ndarray: 2D array with shape (6, N) where N is number of states
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     matrix = traj.to_matrix()
    ///     ```
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

    /// Remove a state at a specific epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Epoch of the state to remove
    ///
    /// Returns:
    ///     numpy.ndarray: The removed state vector
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     removed_state = traj.remove_epoch(epc)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn remove_epoch<'a>(&mut self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.remove_epoch(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Remove a state at a specific index.
    ///
    /// Args:
    ///     index (int): Index of the state to remove
    ///
    /// Returns:
    ///     tuple: Tuple of (Epoch, numpy.ndarray) for the removed epoch and state
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     removed_epc, removed_state = traj.remove(0)
    ///     ```
    #[pyo3(text_signature = "(index)")]
    pub fn remove<'a>(&mut self, py: Python<'a>, index: usize) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.remove(index) {
            Ok((epoch, state)) => {
                Ok((PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get both epoch and state at a specific index.
    ///
    /// Args:
    ///     index (int): Index to retrieve
    ///
    /// Returns:
    ///     tuple: Tuple of (Epoch, numpy.ndarray) for epoch and state at the index
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     ret_epc, ret_state = traj.get(0)
    ///     ```
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
            "OrbitTrajectory(frame={}, representation={}, states={})",
            PyOrbitFrame { frame: self.trajectory.frame }.__repr__(),
            PyOrbitRepresentation { representation: self.trajectory.representation }.__repr__(),
            self.trajectory.len()
        )
    }

    /// String conversion
    fn __str__(&self) -> String {
        format!(
            "OrbitTrajectory(frame={}, representation={}, states={})",
            PyOrbitFrame { frame: self.trajectory.frame }.__str__(),
            PyOrbitRepresentation { representation: self.trajectory.representation }.__str__(),
            self.trajectory.len()
        )
    }

    /// Set eviction policy to keep maximum number of states.
    ///
    /// Args:
    ///     max_size (int): Maximum number of states to retain
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     traj.set_eviction_policy_max_size(1000)
    ///     ```
    #[pyo3(text_signature = "(max_size)")]
    pub fn set_eviction_policy_max_size(&mut self, max_size: usize) -> PyResult<()> {
        match self.trajectory.set_eviction_policy_max_size(max_size) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Set eviction policy to keep states within maximum age.
    ///
    /// Args:
    ///     max_age (float): Maximum age in seconds relative to most recent state
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     traj.set_eviction_policy_max_age(3600.0)
    ///     ```
    #[pyo3(text_signature = "(max_age)")]
    pub fn set_eviction_policy_max_age(&mut self, max_age: f64) -> PyResult<()> {
        match self.trajectory.set_eviction_policy_max_age(max_age) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get current eviction policy.
    ///
    /// Returns:
    ///     str: String representation of eviction policy
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     policy = traj.get_eviction_policy()
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn get_eviction_policy(&self) -> String {
        format!("{:?}", self.trajectory.get_eviction_policy())
    }

    /// Index access returns state vector at given index.
    ///
    /// Args:
    ///     index (int): Index of the state (supports negative indexing)
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

    /// Get the epoch at a specific index.
    ///
    /// Args:
    ///     index (int): Index of the epoch to retrieve
    ///
    /// Returns:
    ///     Epoch: Epoch at the specified index
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.UTC)
    ///     state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    ///     traj.add(epc, state)
    ///
    ///     # Get epoch at index
    ///     epoch_0 = traj.epoch_at_idx(0)
    ///     ```
    #[pyo3(text_signature = "(index)")]
    pub fn epoch_at_idx(&self, index: usize) -> PyResult<PyEpoch> {
        use crate::traits::Trajectory;
        match self.trajectory.epoch_at_idx(index) {
            Ok(epoch) => Ok(PyEpoch { obj: epoch }),
            Err(e) => Err(exceptions::PyIndexError::new_err(e.to_string())),
        }
    }

    /// Get the state vector at a specific index.
    ///
    /// Args:
    ///     index (int): Index of the state to retrieve
    ///
    /// Returns:
    ///     numpy.ndarray: State vector at the specified index
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.UTC)
    ///     state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    ///     traj.add(epc, state)
    ///
    ///     # Get state at index
    ///     state_0 = traj.state_at_idx(0)
    ///     ```
    #[pyo3(text_signature = "(index)")]
    pub fn state_at_idx<'a>(&self, py: Python<'a>, index: usize) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        use crate::traits::Trajectory;
        match self.trajectory.state_at_idx(index) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyIndexError::new_err(e.to_string())),
        }
    }

    /// Get state at specified epoch (in native frame/representation).
    ///
    /// Args:
    ///     epoch (Epoch): Time for state query
    ///
    /// Returns:
    ///     numpy.ndarray: State vector in trajectory's native frame and representation
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     # Create ECI Cartesian trajectory
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.UTC)
    ///     state1 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    ///     traj.add(epc1, state1)
    ///
    ///     # Query state at epoch
    ///     state = traj.state(epc1)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn state<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> Bound<'a, PyArray<f64, Ix1>> {
        use crate::propagators::traits::StateProvider;
        let state = StateProvider::state(&self.trajectory, epoch.obj);
        state.as_slice().to_pyarray(py).to_owned()
    }

    /// Get state in ECI Cartesian frame at specified epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Time for state query
    ///
    /// Returns:
    ///     numpy.ndarray: State vector in ECI Cartesian [x, y, z, vx, vy, vz] (meters, m/s)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     # Create trajectory in any frame/representation
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.KEPLERIAN, bh.AngleFormat.DEGREES)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.UTC)
    ///     oe = np.array([bh.R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0])
    ///     traj.add(epc, oe)
    ///
    ///     # Get ECI Cartesian state (automatically converted from Keplerian)
    ///     state_eci = traj.state_eci(epc)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_eci<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> Bound<'a, PyArray<f64, Ix1>> {
        use crate::propagators::traits::StateProvider;
        let state = StateProvider::state_eci(&self.trajectory, epoch.obj);
        state.as_slice().to_pyarray(py).to_owned()
    }

    /// Get state in ECEF Cartesian frame at specified epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Time for state query
    ///
    /// Returns:
    ///     numpy.ndarray: State vector in ECEF Cartesian [x, y, z, vx, vy, vz] (meters, m/s)
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     # Create ECI trajectory
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.UTC)
    ///     state_eci = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    ///     traj.add(epc, state_eci)
    ///
    ///     # Get ECEF state (automatically converted from ECI)
    ///     state_ecef = traj.state_ecef(epc)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_ecef<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> Bound<'a, PyArray<f64, Ix1>> {
        use crate::propagators::traits::StateProvider;
        let state = StateProvider::state_ecef(&self.trajectory, epoch.obj);
        state.as_slice().to_pyarray(py).to_owned()
    }

    /// Get state as osculating Keplerian elements at specified epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Time for state query
    ///     angle_format (AngleFormat): Desired angle format for output
    ///
    /// Returns:
    ///     numpy.ndarray: Osculating Keplerian elements [a, e, i, raan, argp, M]
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     # Create Cartesian trajectory
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.UTC)
    ///     state_cart = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
    ///     traj.add(epc, state_cart)
    ///
    ///     # Get osculating elements in degrees
    ///     elements = traj.state_as_osculating_elements(epc, bh.AngleFormat.DEGREES)
    ///     print(f"Semi-major axis: {elements[0]/1000:.2f} km")
    ///     print(f"Inclination: {elements[2]:.2f} degrees")
    ///     ```
    #[pyo3(text_signature = "(epoch, angle_format)")]
    pub fn state_as_osculating_elements<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
        angle_format: &PyAngleFormat,
    ) -> Bound<'a, PyArray<f64, Ix1>> {
        use crate::propagators::traits::StateProvider;
        let state = StateProvider::state_as_osculating_elements(&self.trajectory, epoch.obj, angle_format.value);
        state.as_slice().to_pyarray(py).to_owned()
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

    /// Set the trajectory name and return self (builder pattern).
    ///
    /// Args:
    ///     name (str): Name to assign to the trajectory
    ///
    /// Returns:
    ///     OrbitTrajectory: Self with name set
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     traj = traj.with_name("My Trajectory")
    ///     ```
    fn with_name(slf: PyRefMut<'_, Self>, name: &str) -> Py<Self> {
        let py = slf.py();
        let mut traj = slf.trajectory.clone();
        traj = Identifiable::with_name(traj, name);
        Py::new(py, PyOrbitalTrajectory { trajectory: traj }).unwrap()
    }

    /// Set the trajectory numeric ID and return self (builder pattern).
    ///
    /// Args:
    ///     id (int): Numeric ID to assign to the trajectory
    ///
    /// Returns:
    ///     OrbitTrajectory: Self with ID set
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     traj = traj.with_id(12345)
    ///     ```
    fn with_id(slf: PyRefMut<'_, Self>, id: u64) -> Py<Self> {
        let py = slf.py();
        let mut traj = slf.trajectory.clone();
        traj = Identifiable::with_id(traj, id);
        Py::new(py, PyOrbitalTrajectory { trajectory: traj }).unwrap()
    }

    /// Get the trajectory name.
    ///
    /// Returns:
    ///     str | None: The trajectory name, or None if not set
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     traj = traj.with_name("My Trajectory")
    ///     print(traj.get_name())  # "My Trajectory"
    ///     ```
    fn get_name(&self) -> Option<&str> {
        Identifiable::get_name(&self.trajectory)
    }

    /// Get the trajectory numeric ID.
    ///
    /// Returns:
    ///     int | None: The trajectory ID, or None if not set
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     traj = traj.with_id(12345)
    ///     print(traj.get_id())  # 12345
    ///     ```
    fn get_id(&self) -> Option<u64> {
        Identifiable::get_id(&self.trajectory)
    }

    /// Get the trajectory UUID.
    ///
    /// Returns:
    ///     str | None: The trajectory UUID as a string, or None if not set
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     traj = traj.with_new_uuid()
    ///     print(traj.get_uuid())  # e.g., "550e8400-e29b-41d4-a716-446655440000"
    ///     ```
    fn get_uuid(&self) -> Option<String> {
        Identifiable::get_uuid(&self.trajectory).map(|u| u.to_string())
    }

    /// Generate a new UUID and set it on the trajectory (builder pattern).
    ///
    /// Returns:
    ///     OrbitTrajectory: Self with new UUID set
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     traj = traj.with_new_uuid()
    ///     ```
    fn with_new_uuid(slf: PyRefMut<'_, Self>) -> Py<Self> {
        let py = slf.py();
        let mut traj = slf.trajectory.clone();
        traj = Identifiable::with_new_uuid(traj);
        Py::new(py, PyOrbitalTrajectory { trajectory: traj }).unwrap()
    }
}

/// Iterator for OrbitTrajectory
#[pyclass(module = "brahe._brahe")]
struct PyOrbitalTrajectoryIterator {
    trajectory: Py<PyOrbitalTrajectory>,
    index: usize,
}

type PyTrajectoryIterItem<'a> = PyResult<Option<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)>>;

#[pymethods]
impl PyOrbitalTrajectoryIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(&mut self, py: Python<'a>) -> PyTrajectoryIterItem<'a> {
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

/// Dynamic-dimension trajectory container.
///
/// Stores a sequence of N-dimensional states at specific epochs with support
/// for interpolation and automatic state eviction policies. Dimension is
/// determined at runtime.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "DTrajectory")]
pub struct PyTrajectory {
    pub(crate) trajectory: trajectories::DTrajectory,
}

#[pymethods]
impl PyTrajectory {
    /// Create a new empty trajectory with specified dimension.
    ///
    /// Args:
    ///     dimension (int): Trajectory dimension (default 6, must be greater than 0)
    ///
    /// Returns:
    ///     DTrajectory: New empty trajectory instance with linear interpolation
    ///
    /// Examples:
    ///     DTrajectory()    # 6D trajectory (default)
    ///     DTrajectory(3)   # 3D trajectory
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

    /// Create a trajectory from existing data.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of time epochs
    ///     states (numpy.ndarray): 2D array of states with shape (num_epochs, dimension)
    ///         where each row is a state vector
    ///     interpolation_method (InterpolationMethod): Interpolation method (default Linear)
    ///
    /// Returns:
    ///     DTrajectory: New trajectory instance populated with data
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

        for (i, &epoch) in epochs_vec.iter().enumerate().take(num_epochs) {
            let state_row = states_array.row(i);
            let state_vec = na::DVector::from_iterator(dimension, state_row.iter().copied());

            trajectory.add(epoch, state_vec)
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.DTrajectory(6)
    ///     traj = traj.with_interpolation_method(bh.InterpolationMethod.LINEAR)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.DTrajectory(6)
    ///     traj = traj.with_eviction_policy_max_size(1000)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.DTrajectory(6)
    ///     traj = traj.with_eviction_policy_max_age(3600.0)
    ///     ```
    #[pyo3(text_signature = "(max_age)")]
    pub fn with_eviction_policy_max_age(mut slf: PyRefMut<'_, Self>, max_age: f64) -> Self {
        slf.trajectory = slf.trajectory.clone().with_eviction_policy_max_age(max_age);
        Self { trajectory: slf.trajectory.clone() }
    }

    /// Get the trajectory dimension.
    ///
    /// Returns:
    ///     int: Dimension of the trajectory
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.DTrajectory(6)
    ///     print(f"Dimension: {traj.dimension}")
    ///     ```
    #[getter]
    pub fn dimension(&self) -> usize {
        self.trajectory.dimension
    }

    /// Get the trajectory dimension (method form).
    ///
    /// Returns:
    ///     int: Dimension of the trajectory
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.DTrajectory(6)
    ///     print(f"Dimension: {traj.dimension()}")
    ///     ```
    #[pyo3(name = "dimension")]
    #[pyo3(text_signature = "()")]
    pub fn dimension_method(&self) -> usize {
        self.trajectory.dimension
    }

    /// Add a state to the trajectory.
    ///
    /// Args:
    ///     epoch (Epoch): Time of the state
    ///     state (numpy.ndarray): N-element state vector where N is the trajectory dimension
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.DTrajectory(6)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     ```
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


    /// Get the nearest state to a given epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: Tuple of (Epoch, numpy.ndarray) containing the nearest state
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 30.0, 0.0, bh.TimeSystem.UTC)
    ///     nearest_epc, nearest_state = traj.nearest_state(epc2)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn nearest_state<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.nearest_state(&epoch.obj) {
            Ok((nearest_epoch, nearest_state)) => {
                Ok((PyEpoch { obj: nearest_epoch }, nearest_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the index of the state at or before the given epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     int: Index of the state at or before the target epoch
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     index = traj.index_before_epoch(epc2)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn index_before_epoch(&self, epoch: PyRef<PyEpoch>) -> PyResult<usize> {
        match self.trajectory.index_before_epoch(&epoch.obj) {
            Ok(index) => Ok(index),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the index of the state at or after the given epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     int: Index of the state at or after the target epoch
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 11, 59, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     index = traj.index_after_epoch(epc2)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn index_after_epoch(&self, epoch: PyRef<PyEpoch>) -> PyResult<usize> {
        match self.trajectory.index_after_epoch(&epoch.obj) {
            Ok(index) => Ok(index),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the state at or before the given epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: Tuple of (Epoch, numpy.ndarray) containing state at or before the target epoch
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     ret_epc, ret_state = traj.state_before_epoch(epc2)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_before_epoch<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.state_before_epoch(&epoch.obj) {
            Ok((ret_epoch, ret_state)) => {
                Ok((PyEpoch { obj: ret_epoch }, ret_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the state at or after the given epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: Tuple of (Epoch, numpy.ndarray) containing state at or after the target epoch
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 11, 59, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     ret_epc, ret_state = traj.state_after_epoch(epc2)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_after_epoch<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.state_after_epoch(&epoch.obj) {
            Ok((ret_epoch, ret_state)) => {
                Ok((PyEpoch { obj: ret_epoch }, ret_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Interpolate state at a given epoch using linear interpolation.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     numpy.ndarray: Linearly interpolated state vector
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state1 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state1)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 2, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state2 = np.array([bh.R_EARTH + 510e3, 0.0, 0.0, 0.0, 7650.0, 0.0])
    ///     traj.add(epc2, state2)
    ///     epc_mid = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state_interp = traj.interpolate_linear(epc_mid)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn interpolate_linear<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.interpolate_linear(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Interpolate state at a given epoch using the configured interpolation method.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     numpy.ndarray: Interpolated state vector
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state1 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state1)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 2, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state2 = np.array([bh.R_EARTH + 510e3, 0.0, 0.0, 0.0, 7650.0, 0.0])
    ///     traj.add(epc2, state2)
    ///     epc_mid = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state_interp = traj.interpolate(epc_mid)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.DTrajectory(6)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     retrieved_state = traj.state_at_idx(0)
    ///     ```
    #[pyo3(text_signature = "(index)")]
    pub fn state_at_idx<'a>(&self, py: Python<'a>, index: usize) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        use crate::traits::Trajectory;
        match self.trajectory.state_at_idx(index) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyIndexError::new_err(e.to_string())),
        }
    }

    /// Get epoch at a specific index
    ///
    /// Arguments:
    ///     index (int): Index of the epoch
    ///
    /// Returns:
    ///     Epoch: Epoch at index
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.DTrajectory(6)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     retrieved_epc = traj.epoch_at_idx(0)
    ///     ```
    #[pyo3(text_signature = "(index)")]
    pub fn epoch_at_idx(&self, index: usize) -> PyResult<PyEpoch> {
        use crate::traits::Trajectory;
        match self.trajectory.epoch_at_idx(index) {
            Ok(epoch) => Ok(PyEpoch { obj: epoch }),
            Err(e) => Err(exceptions::PyIndexError::new_err(e.to_string())),
        }
    }

    /// Get the number of states in the trajectory.
    ///
    /// Returns:
    ///     int: Number of states in the trajectory
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.DTrajectory(6)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     print(f"Trajectory length: {traj.length}")
    ///     ```
    #[getter]
    pub fn length(&self) -> usize {
        self.trajectory.len()
    }

    /// Get the number of states in the trajectory (alias for length).
    ///
    /// Returns:
    ///     int: Number of states in the trajectory
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.DTrajectory(6)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     print(f"Number of states: {traj.len()}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn len(&self) -> usize {
        self.trajectory.len()
    }

    /// Check if trajectory is empty.
    ///
    /// Returns:
    ///     bool: True if trajectory contains no states, False otherwise
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.DTrajectory(6)
    ///     print(f"Is empty: {traj.is_empty()}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn is_empty(&self) -> bool {
        self.trajectory.is_empty()
    }

    /// Get interpolation method.
    ///
    /// Returns:
    ///     InterpolationMethod: Current interpolation method
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.DTrajectory(6)
    ///     method = traj.get_interpolation_method()
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn get_interpolation_method(&self) -> PyInterpolationMethod {
        PyInterpolationMethod { method: self.trajectory.interpolation_method }
    }

    /// Set interpolation method.
    ///
    /// Args:
    ///     method (InterpolationMethod): New interpolation method
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.DTrajectory(6)
    ///     method = bh.InterpolationMethod.LINEAR
    ///     traj.set_interpolation_method(method)
    ///     ```
    #[pyo3(text_signature = "(method)")]
    pub fn set_interpolation_method(&mut self, method: PyRef<PyInterpolationMethod>) {
        self.trajectory.set_interpolation_method(method.method);
    }

    /// Set maximum trajectory size.
    ///
    /// Args:
    ///     max_size (int): Maximum number of states to retain
    #[pyo3(text_signature = "(max_size)")]
    pub fn set_eviction_policy_max_size(&mut self, max_size: usize) -> PyResult<()> {
        match self.trajectory.set_eviction_policy_max_size(max_size) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Set maximum age for trajectory states.
    ///
    /// Args:
    ///     max_age (float): Maximum age in seconds relative to most recent state
    #[pyo3(text_signature = "(max_age)")]
    pub fn set_eviction_policy_max_age(&mut self, max_age: f64) -> PyResult<()> {
        match self.trajectory.set_eviction_policy_max_age(max_age) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get current eviction policy.
    ///
    /// Returns:
    ///     str: String representation of eviction policy
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     policy = traj.get_eviction_policy()
    ///     ```
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

    /// Clear all states from the trajectory.
    #[pyo3(text_signature = "()")]
    pub fn clear(&mut self) {
        self.trajectory.clear();
    }

    /// Get the first (epoch, state) tuple in the trajectory, if any exists.
    ///
    /// Returns:
    ///     tuple or None: Tuple of (Epoch, numpy.ndarray) for first state, or None if empty
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     first_epc, first_state = traj.first()
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn first<'a>(&self, py: Python<'a>) -> Option<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        self.trajectory.first().map(|(epoch, state)| {
            (PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned())
        })
    }

    /// Get the last (epoch, state) tuple in the trajectory, if any exists.
    ///
    /// Returns:
    ///     tuple or None: Tuple of (Epoch, numpy.ndarray) for last state, or None if empty
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     last_epc, last_state = traj.last()
    ///     ```
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

    /// Remove a state at a specific epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Epoch of the state to remove
    ///
    /// Returns:
    ///     numpy.ndarray: The removed state vector
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     removed_state = traj.remove_epoch(epc)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn remove_epoch<'a>(&mut self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.remove_epoch(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Remove a state at a specific index.
    ///
    /// Args:
    ///     index (int): Index of the state to remove
    ///
    /// Returns:
    ///     tuple: Tuple of (Epoch, numpy.ndarray) for the removed epoch and state
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     removed_epc, removed_state = traj.remove(0)
    ///     ```
    #[pyo3(text_signature = "(index)")]
    pub fn remove<'a>(&mut self, py: Python<'a>, index: usize) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.remove(index) {
            Ok((epoch, state)) => {
                Ok((PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get both epoch and state at a specific index.
    ///
    /// Args:
    ///     index (int): Index to retrieve
    ///
    /// Returns:
    ///     tuple: Tuple of (Epoch, numpy.ndarray) for epoch and state at the index
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     ret_epc, ret_state = traj.get(0)
    ///     ```
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

    /// Index access returns state vector at given index.
    ///
    /// Args:
    ///     index (int): Index of the state (supports negative indexing)
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
#[pyclass(module = "brahe._brahe")]
struct PyTrajectoryIterator {
    trajectory: Py<PyTrajectory>,
    index: usize,
}

#[pymethods]
impl PyTrajectoryIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(&mut self, py: Python<'a>) -> PyTrajectoryIterItem<'a> {
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


/// Static-dimension 6D trajectory container.
///
/// Stores a sequence of 6-dimensional states at specific epochs with support
/// for interpolation and automatic state eviction policies. Dimension is fixed
/// at compile time for performance.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "STrajectory6")]
pub struct PySTrajectory6 {
    pub(crate) trajectory: trajectories::STrajectory6,
}

#[pymethods]
impl PySTrajectory6 {
    /// Create a new empty 6D trajectory.
    ///
    /// Args:
    ///     interpolation_method (InterpolationMethod): Interpolation method (default Linear)
    ///
    /// Returns:
    ///     STrajectory6: New empty 6D trajectory instance
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.STrajectory6()
    ///     ```
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

    /// Create a trajectory from existing data.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of time epochs
    ///     states (numpy.ndarray): 2D array of 6D state vectors with shape (N, 6)
    ///         where N is the number of epochs. Each row is one state vector.
    ///     interpolation_method (InterpolationMethod): Interpolation method (default Linear)
    ///
    /// Returns:
    ///     STrajectory6: New 6D trajectory instance populated with data
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     states = np.array([[bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0],
    ///                        [bh.R_EARTH + 510e3, 0.0, 0.0, 0.0, 7650.0, 0.0]])
    ///     traj = bh.STrajectory6.from_data([epc1, epc2], states)
    ///     ```
    #[classmethod]
    #[pyo3(signature = (epochs, states, interpolation_method=None))]
    pub fn from_data(
        _cls: &Bound<'_, PyType>,
        epochs: Vec<PyRef<PyEpoch>>,
        states: PyReadonlyArray2<f64>,
        interpolation_method: Option<PyRef<PyInterpolationMethod>>,
    ) -> PyResult<Self> {
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

        // Check that state dimension is 6
        if states_array.ncols() != 6 {
            return Err(exceptions::PyValueError::new_err(
                format!("State dimension must be 6, got {}", states_array.ncols())
            ));
        }

        let mut states_vec = Vec::new();
        for i in 0..num_epochs {
            let state_row = states_array.row(i);
            let state_vec = na::Vector6::from_iterator(state_row.iter().copied());
            states_vec.push(state_vec);
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

    /// Get trajectory dimension (always 6).
    ///
    /// Returns:
    ///     int: Dimension of the trajectory (always 6)
    #[pyo3(text_signature = "()")]
    pub fn dimension(&self) -> usize {
        6
    }

    /// Add a state to the trajectory.
    ///
    /// Args:
    ///     epoch (Epoch): Time of the state
    ///     state (numpy.ndarray): 6-element state vector
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

    /// Get the nearest state to a given epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: Tuple of (Epoch, numpy.ndarray) containing the nearest state
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 30.0, 0.0, bh.TimeSystem.UTC)
    ///     nearest_epc, nearest_state = traj.nearest_state(epc2)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn nearest_state<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.nearest_state(&epoch.obj) {
            Ok((nearest_epoch, nearest_state)) => {
                Ok((PyEpoch { obj: nearest_epoch }, nearest_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the index of the state at or before the given epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     int: Index of the state at or before the target epoch
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     index = traj.index_before_epoch(epc2)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn index_before_epoch(&self, epoch: PyRef<PyEpoch>) -> PyResult<usize> {
        match self.trajectory.index_before_epoch(&epoch.obj) {
            Ok(index) => Ok(index),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the index of the state at or after the given epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     int: Index of the state at or after the target epoch
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 11, 59, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     index = traj.index_after_epoch(epc2)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn index_after_epoch(&self, epoch: PyRef<PyEpoch>) -> PyResult<usize> {
        match self.trajectory.index_after_epoch(&epoch.obj) {
            Ok(index) => Ok(index),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the state at or before the given epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: Tuple of (Epoch, numpy.ndarray) containing state at or before the target epoch
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     ret_epc, ret_state = traj.state_before_epoch(epc2)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_before_epoch<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.state_before_epoch(&epoch.obj) {
            Ok((ret_epoch, ret_state)) => {
                Ok((PyEpoch { obj: ret_epoch }, ret_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get the state at or after the given epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     tuple: Tuple of (Epoch, numpy.ndarray) containing state at or after the target epoch
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 11, 59, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     ret_epc, ret_state = traj.state_after_epoch(epc2)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_after_epoch<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.state_after_epoch(&epoch.obj) {
            Ok((ret_epoch, ret_state)) => {
                Ok((PyEpoch { obj: ret_epoch }, ret_state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Set the interpolation method for the trajectory.
    ///
    /// Args:
    ///     method (InterpolationMethod): New interpolation method
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     traj.set_interpolation_method(bh.InterpolationMethod.LINEAR)
    ///     ```
    #[pyo3(text_signature = "(method)")]
    pub fn set_interpolation_method(&mut self, method: PyRef<PyInterpolationMethod>) {
        self.trajectory.set_interpolation_method(method.method);
    }

    /// Interpolate state at a given epoch using linear interpolation.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     numpy.ndarray: Linearly interpolated state vector
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state1 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state1)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 2, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state2 = np.array([bh.R_EARTH + 510e3, 0.0, 0.0, 0.0, 7650.0, 0.0])
    ///     traj.add(epc2, state2)
    ///     epc_mid = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state_interp = traj.interpolate_linear(epc_mid)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn interpolate_linear<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.interpolate_linear(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Interpolate state at a given epoch using the configured interpolation method.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch
    ///
    /// Returns:
    ///     numpy.ndarray: Interpolated state vector
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state1 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc1, state1)
    ///     epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 2, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state2 = np.array([bh.R_EARTH + 510e3, 0.0, 0.0, 0.0, 7650.0, 0.0])
    ///     traj.add(epc2, state2)
    ///     epc_mid = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state_interp = traj.interpolate(epc_mid)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.DTrajectory(6)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     retrieved_state = traj.state_at_idx(0)
    ///     ```
    #[pyo3(text_signature = "(index)")]
    pub fn state_at_idx<'a>(&self, py: Python<'a>, index: usize) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        use crate::traits::Trajectory;
        match self.trajectory.state_at_idx(index) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyIndexError::new_err(e.to_string())),
        }
    }

    /// Get epoch at a specific index
    ///
    /// Arguments:
    ///     index (int): Index of the epoch
    ///
    /// Returns:
    ///     Epoch: Epoch at index
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.DTrajectory(6)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     retrieved_epc = traj.epoch_at_idx(0)
    ///     ```
    #[pyo3(text_signature = "(index)")]
    pub fn epoch_at_idx(&self, index: usize) -> PyResult<PyEpoch> {
        use crate::traits::Trajectory;
        match self.trajectory.epoch_at_idx(index) {
            Ok(epoch) => Ok(PyEpoch { obj: epoch }),
            Err(e) => Err(exceptions::PyIndexError::new_err(e.to_string())),
        }
    }

    /// Get the number of states in the trajectory.
    ///
    /// Returns:
    ///     int: Number of states in the trajectory
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.DTrajectory(6)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     print(f"Trajectory length: {traj.length}")
    ///     ```
    #[getter]
    pub fn length(&self) -> usize {
        self.trajectory.len()
    }

    /// Get the number of states in the trajectory (alias for length).
    ///
    /// Returns:
    ///     int: Number of states in the trajectory
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.DTrajectory(6)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     print(f"Number of states: {traj.len()}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn len(&self) -> usize {
        self.trajectory.len()
    }

    /// Get interpolation method.
    ///
    /// Returns:
    ///     InterpolationMethod: Current interpolation method
    #[getter]
    pub fn interpolation_method(&self) -> PyInterpolationMethod {
        PyInterpolationMethod { method: self.trajectory.get_interpolation_method() }
    }

    /// Set maximum trajectory size.
    ///
    /// Args:
    ///     max_size (int): Maximum number of states to retain
    #[pyo3(text_signature = "(max_size)")]
    pub fn set_eviction_policy_max_size(&mut self, max_size: usize) -> PyResult<()> {
        match self.trajectory.set_eviction_policy_max_size(max_size) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Set maximum age for trajectory states.
    ///
    /// Args:
    ///     max_age (float): Maximum age in seconds relative to most recent state
    #[pyo3(text_signature = "(max_age)")]
    pub fn set_eviction_policy_max_age(&mut self, max_age: f64) -> PyResult<()> {
        match self.trajectory.set_eviction_policy_max_age(max_age) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get current eviction policy.
    ///
    /// Returns:
    ///     str: String representation of eviction policy
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     policy = traj.get_eviction_policy()
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn get_eviction_policy(&self) -> String {
        format!("{:?}", self.trajectory.get_eviction_policy())
    }

    /// Get start epoch of trajectory.
    ///
    /// Returns:
    ///     Epoch or None: First epoch if trajectory is not empty, None otherwise
    #[getter]
    pub fn start_epoch(&self) -> Option<PyEpoch> {
        self.trajectory.start_epoch().map(|epoch| PyEpoch { obj: epoch })
    }

    /// Get end epoch of trajectory.
    ///
    /// Returns:
    ///     Epoch or None: Last epoch if trajectory is not empty, None otherwise
    #[getter]
    pub fn end_epoch(&self) -> Option<PyEpoch> {
        self.trajectory.end_epoch().map(|epoch| PyEpoch { obj: epoch })
    }

    /// Get time span of trajectory in seconds.
    ///
    /// Returns:
    ///     float or None: Time span between first and last epochs, or None if less than 2 states
    #[getter]
    pub fn time_span(&self) -> Option<f64> {
        self.trajectory.timespan()
    }

    /// Clear all states from the trajectory.
    #[pyo3(text_signature = "()")]
    pub fn clear(&mut self) {
        self.trajectory.clear();
    }

    /// Get the first (epoch, state) tuple in the trajectory, if any exists.
    ///
    /// Returns:
    ///     tuple or None: Tuple of (Epoch, numpy.ndarray) for first state, or None if empty
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     first_epc, first_state = traj.first()
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn first<'a>(&self, py: Python<'a>) -> Option<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        self.trajectory.first().map(|(epoch, state)| {
            (PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned())
        })
    }

    /// Get the last (epoch, state) tuple in the trajectory, if any exists.
    ///
    /// Returns:
    ///     tuple or None: Tuple of (Epoch, numpy.ndarray) for last state, or None if empty
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     last_epc, last_state = traj.last()
    ///     ```
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

    /// Remove a state at a specific epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Epoch of the state to remove
    ///
    /// Returns:
    ///     numpy.ndarray: The removed state vector
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     removed_state = traj.remove_epoch(epc)
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    pub fn remove_epoch<'a>(&mut self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.trajectory.remove_epoch(&epoch.obj) {
            Ok(state) => Ok(state.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Remove a state at a specific index.
    ///
    /// Args:
    ///     index (int): Index of the state to remove
    ///
    /// Returns:
    ///     tuple: Tuple of (Epoch, numpy.ndarray) for the removed epoch and state
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     removed_epc, removed_state = traj.remove(0)
    ///     ```
    #[pyo3(text_signature = "(index)")]
    pub fn remove<'a>(&mut self, py: Python<'a>, index: usize) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.remove(index) {
            Ok((epoch, state)) => {
                Ok((PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get both epoch and state at a specific index.
    ///
    /// Args:
    ///     index (int): Index to retrieve
    ///
    /// Returns:
    ///     tuple: Tuple of (Epoch, numpy.ndarray) for epoch and state at the index
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
    ///     traj.add(epc, state)
    ///     ret_epc, ret_state = traj.get(0)
    ///     ```
    #[pyo3(text_signature = "(index)")]
    pub fn get<'a>(&self, py: Python<'a>, index: usize) -> PyResult<(PyEpoch, Bound<'a, PyArray<f64, Ix1>>)> {
        match self.trajectory.get(index) {
            Ok((epoch, state)) => {
                Ok((PyEpoch { obj: epoch }, state.as_slice().to_pyarray(py).to_owned()))
            }
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Check if trajectory is empty.
    ///
    /// Returns:
    ///     bool: True if trajectory contains no states, False otherwise
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     traj = bh.DTrajectory(6)
    ///     print(f"Is empty: {traj.is_empty()}")
    ///     ```
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

    /// Index access returns state vector at given index.
    ///
    /// Args:
    ///     index (int): Index of the state (supports negative indexing)
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
#[pyclass(module = "brahe._brahe")]
struct PySTrajectory6Iterator {
    trajectory: Py<PySTrajectory6>,
    index: usize,
}

#[pymethods]
impl PySTrajectory6Iterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'a>(&mut self, py: Python<'a>) -> PyTrajectoryIterItem<'a> {
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
