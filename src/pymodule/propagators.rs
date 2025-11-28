/// SGP4/SDP4 satellite propagator using TLE data.
///
/// The SGP (Simplified General Perturbations) propagator implements the SGP4/SDP4 models
/// for propagating satellites using Two-Line Element (TLE) orbital data. This is the standard
/// model used for tracking objects in Earth orbit.
///
/// Note:
///     This class is created via class methods, not direct instantiation.
///     Use `SGPPropagator.from_tle()` or `SGPPropagator.from_elements()`.
///
/// Attributes:
///     current_epoch (Epoch): Current propagation time
///     initial_epoch (Epoch): TLE epoch
///     step_size (float): Current step size in seconds
///     norad_id (int): NORAD catalog ID
///     satellite_name (str or None): Satellite name if available
///     trajectory (OrbitTrajectory): Accumulated trajectory states
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # ISS TLE data (example)
///     line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  30000-3 0  9005"
///     line2 = "2 25544  51.6400 150.0000 0003000 100.0000 260.0000 15.50000000300000"
///
///     # Create propagator
///     prop = bh.SGPPropagator.from_tle(line1, line2, step_size=60.0)
///
///     # Propagate to a specific epoch
///     epc = bh.Epoch.from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///     state_eci = prop.state(epc)
///     print(f"Position: {state_eci[:3]}")
///     print(f"Velocity: {state_eci[3:]}")
///
///     # Propagate multiple epochs
///     epochs = [epc + i*60.0 for i in range(10)]  # 10 minutes
///     states = prop.states(epochs)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "SGPPropagator")]
pub struct PySGPPropagator {
    pub propagator: propagators::SGPPropagator,
}

#[pymethods]
impl PySGPPropagator {
    /// Create a new SGP propagator from TLE lines.
    ///
    /// Args:
    ///     line1 (str): First line of TLE data.
    ///     line2 (str): Second line of TLE data.
    ///     step_size (float): Step size in seconds for propagation. Defaults to 60.0.
    ///
    /// Returns:
    ///     SGPPropagator: New SGP propagator instance.
    #[classmethod]
    #[pyo3(signature = (line1, line2, step_size=60.0))]
    pub fn from_tle(_cls: &Bound<'_, PyType>, line1: String, line2: String, step_size: Option<f64>) -> PyResult<Self> {
        let step_size = step_size.unwrap_or(60.0);
        match propagators::SGPPropagator::from_tle(&line1, &line2, step_size) {
            Ok(propagator) => Ok(PySGPPropagator { propagator }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Create a new SGP propagator from 3-line TLE format (with satellite name).
    ///
    /// Args:
    ///     name (str): Satellite name (line 0).
    ///     line1 (str): First line of TLE data.
    ///     line2 (str): Second line of TLE data.
    ///     step_size (float): Step size in seconds for propagation. Defaults to 60.0.
    ///
    /// Returns:
    ///     SGPPropagator: New SGP propagator instance.
    #[classmethod]
    #[pyo3(signature = (name, line1, line2, step_size=60.0))]
    pub fn from_3le(_cls: &Bound<'_, PyType>, name: String, line1: String, line2: String, step_size: Option<f64>) -> PyResult<Self> {
        let step_size = step_size.unwrap_or(60.0);
        match propagators::SGPPropagator::from_3le(Some(&name), &line1, &line2, step_size) {
            Ok(propagator) => Ok(PySGPPropagator { propagator }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get NORAD ID.
    ///
    /// Returns:
    ///     int: NORAD catalog ID.
    #[getter]
    pub fn norad_id(&self) -> u32 {
        self.propagator.norad_id
    }

    /// Get satellite name (if available).
    ///
    /// Returns:
    ///     str or None: Satellite name if provided.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     name = "ISS (ZARYA)"
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     propagator = bh.SGPPropagator.from_3le(name, line1, line2)
    ///     print(f"Satellite: {propagator.satellite_name}")
    ///     ```
    #[getter]
    pub fn satellite_name(&self) -> Option<String> {
        self.propagator.satellite_name.clone()
    }

    /// Get TLE epoch.
    ///
    /// Returns:
    ///     Epoch: Epoch of the TLE data.
    #[getter]
    pub fn epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.propagator.epoch }
    }

    /// Get current epoch.
    ///
    /// Returns:
    ///     Epoch: Current propagator epoch.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     propagator = bh.SGPPropagator.from_tle(line1, line2)
    ///     propagator.step()
    ///     print(f"Current epoch: {propagator.current_epoch}")
    ///     ```
    #[getter]
    pub fn current_epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.propagator.current_epoch() }
    }

    /// Get step size in seconds.
    ///
    /// Returns:
    ///     float: Step size in seconds.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     propagator = bh.SGPPropagator.from_tle(line1, line2)
    ///     print(f"Step size: {propagator.step_size} seconds")
    ///     ```
    #[getter]
    pub fn step_size(&self) -> f64 {
        self.propagator.step_size()
    }

    /// Set step size in seconds.
    ///
    /// Args:
    ///     step_size (float): New step size in seconds.
    #[setter]
    pub fn set_step_size(&mut self, step_size: f64) {
        self.propagator.set_step_size(step_size);
    }

    /// Set step size in seconds (explicit method).
    ///
    /// Args:
    ///     new_step_size (float): New step size in seconds.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     propagator = bh.SGPPropagator.from_tle(line1, line2)
    ///     propagator.set_step_size(120.0)  # Can use explicit method
    ///     # or propagator.step_size = 120.0  # Can use property
    ///     ```
    #[pyo3(name = "set_step_size", text_signature = "(new_step_size)")]
    pub fn set_step_size_explicit(&mut self, new_step_size: f64) {
        self.propagator.set_step_size(new_step_size);
    }

    /// Set output format (frame, representation, and angle format).
    ///
    /// Args:
    ///     frame (OrbitFrame): Output frame (ECI or ECEF).
    ///     representation (OrbitRepresentation): Output representation (Cartesian or Keplerian).
    ///     angle_format (AngleFormat or None): Angle format for Keplerian (None for Cartesian).
    #[pyo3(text_signature = "(frame, representation, angle_format)")]
    pub fn set_output_format(
        &mut self,
        frame: PyRef<PyOrbitFrame>,
        representation: PyRef<PyOrbitRepresentation>,
        angle_format: Option<PyRef<PyAngleFormat>>,
    ) {
        let angle_fmt = angle_format.map(|af| af.value);
        self.propagator = self.propagator.clone().with_output_format(
            frame.frame,
            representation.representation,
            angle_fmt,
        );
    }

    /// Get current state vector.
    ///
    /// Returns:
    ///     numpy.ndarray: Current state vector in the propagator's output format.
    #[pyo3(text_signature = "()")]
    pub fn current_state<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray<f64, Ix1>> {
        let state = self.propagator.current_state();
        state.as_slice().to_pyarray(py).to_owned()
    }

    /// Get initial state vector.
    ///
    /// Returns:
    ///     numpy.ndarray: Initial state vector in the propagator's output format.
    #[pyo3(text_signature = "()")]
    pub fn initial_state<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray<f64, Ix1>> {
        let state = self.propagator.initial_state();
        state.as_slice().to_pyarray(py).to_owned()
    }

    /// Compute state at a specific epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector in the propagator's current output format.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = SStateProvider::state(&self.propagator, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in PEF coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in PEF frame.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_pef<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self.propagator.state_pef(epoch.obj);
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in ECI coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ECI frame.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_eci<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = SOrbitStateProvider::state_eci(&self.propagator, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in ECEF coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ECEF frame.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_ecef<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = SOrbitStateProvider::state_ecef(&self.propagator, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in GCRF coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in GCRF frame (meters, m/s).
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_gcrf<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = SOrbitStateProvider::state_gcrf(&self.propagator, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in EME2000 coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in EME2000 frame (meters, m/s).
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_eme2000<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = SOrbitStateProvider::state_eme2000(&self.propagator, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in ITRF coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ITRF frame (meters, m/s).
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_itrf<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = SOrbitStateProvider::state_itrf(&self.propagator, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute states at multiple epochs.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of state vectors in the propagator's current output format.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = SStateProvider::states(&self.propagator, &epoch_vec)?;
        Ok(states.iter().map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned()).collect())
    }

    /// Compute states at multiple epochs in ECI coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of ECI state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_eci<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = SOrbitStateProvider::states_eci(&self.propagator, &epoch_vec)?;
        Ok(states.iter().map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned()).collect())
    }

    /// Compute states at multiple epochs in GCRF coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of GCRF state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_gcrf<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = SOrbitStateProvider::states_gcrf(&self.propagator, &epoch_vec)?;
        Ok(states.iter().map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned()).collect())
    }

    /// Compute states at multiple epochs in ITRF coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of ITRF state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_itrf<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = SOrbitStateProvider::states_itrf(&self.propagator, &epoch_vec)?;
        Ok(states.iter().map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned()).collect())
    }

    /// Step forward by the default step size.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///     prop.step()  # Advance by default step_size
    ///     print(f"Advanced to: {prop.current_epoch}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn step(&mut self) {
        self.propagator.step();
    }

    /// Step forward by a specified time duration.
    ///
    /// Args:
    ///     step_size (float): Time step in seconds.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///     prop.step_by(120.0)  # Advance by 2 minutes
    ///     print(f"Advanced to: {prop.current_epoch}")
    ///     ```
    #[pyo3(text_signature = "(step_size)")]
    pub fn step_by(&mut self, step_size: f64) {
        self.propagator.step_by(step_size);
    }

    /// Step past a specified target epoch.
    ///
    /// Args:
    ///     target_epoch (Epoch): The epoch to step past.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///     target = prop.epoch + 3600.0  # 1 hour later
    ///     prop.step_past(target)
    ///     print(f"Stepped past target")
    ///     ```
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn step_past(&mut self, target_epoch: PyRef<PyEpoch>) {
        self.propagator.step_past(target_epoch.obj);
    }

    /// Propagate forward by specified number of steps.
    ///
    /// Args:
    ///     num_steps (int): Number of steps to take.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2, step_size=60.0)
    ///     prop.propagate_steps(10)  # Advance by 10 steps (600 seconds)
    ///     print(f"After 10 steps: {prop.current_epoch}")
    ///     ```
    #[pyo3(text_signature = "(num_steps)")]
    pub fn propagate_steps(&mut self, num_steps: usize) {
        self.propagator.propagate_steps(num_steps);
    }

    /// Propagate to a specific target epoch.
    ///
    /// Args:
    ///     target_epoch (Epoch): The epoch to propagate to.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///     target = prop.epoch + 7200.0  # 2 hours later
    ///     prop.propagate_to(target)
    ///     print(f"Propagated to: {prop.current_epoch}")
    ///     ```
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn propagate_to(&mut self, target_epoch: PyRef<PyEpoch>) {
        self.propagator.propagate_to(target_epoch.obj);
    }

    /// Reset propagator to initial conditions.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///     initial_epoch = prop.epoch
    ///     prop.propagate_steps(100)
    ///     prop.reset()
    ///     print(f"Reset to: {prop.current_epoch == initial_epoch}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn reset(&mut self) {
        self.propagator.reset();
    }

    /// Set trajectory eviction policy based on maximum size.
    ///
    /// Args:
    ///     max_size (int): Maximum number of states to keep in trajectory.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///     prop.set_eviction_policy_max_size(1000)
    ///     print("Trajectory limited to 1000 states")
    ///     ```
    #[pyo3(text_signature = "(max_size)")]
    pub fn set_eviction_policy_max_size(&mut self, max_size: usize) -> PyResult<()> {
        self.propagator.set_eviction_policy_max_size(max_size)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Set trajectory eviction policy based on maximum age.
    ///
    /// Args:
    ///     max_age (float): Maximum age in seconds to keep states in trajectory.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///     prop.set_eviction_policy_max_age(86400.0)  # Keep 1 day of history
    ///     print("Trajectory limited to 24 hours of states")
    ///     ```
    #[pyo3(text_signature = "(max_age)")]
    pub fn set_eviction_policy_max_age(&mut self, max_age: f64) -> PyResult<()> {
        self.propagator.set_eviction_policy_max_age(max_age)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get accumulated trajectory.
    ///
    /// Returns:
    ///     OrbitalTrajectory: The accumulated trajectory.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///     prop.propagate_steps(100)
    ///     traj = prop.trajectory
    ///     print(f"Trajectory has {traj.len()} states")
    ///     ```
    #[getter]
    pub fn trajectory(&self) -> PyOrbitalTrajectory {
        let traj = &self.propagator.trajectory;

        // Convert states from SVector to DVector
        let states: Vec<DVector<f64>> = traj.states.iter()
            .map(|s| DVector::from_column_slice(s.as_slice()))
            .collect();

        // Convert covariances from SMatrix to DMatrix if present
        let covariances = traj.covariances.as_ref().map(|covs| {
            covs.iter()
                .map(|c| DMatrix::from_row_slice(6, 6, c.as_slice()))
                .collect()
        });

        let mut d_traj = trajectories::DOrbitTrajectory::from_orbital_data(
            traj.epochs.clone(),
            states,
            traj.frame,
            traj.representation,
            traj.angle_format,
            covariances,
        );

        // Copy identity from original trajectory
        d_traj.set_identity(
            traj.get_name(),
            traj.get_uuid(),
            traj.get_id()
        );

        PyOrbitalTrajectory { trajectory: d_traj }
    }

    /// Get Keplerian orbital elements from TLE data.
    ///
    /// Extracts the Keplerian elements directly from the TLE lines used to
    /// initialize this propagator.
    ///
    /// Args:
    ///     angle_format (AngleFormat): Format for angular elements (DEGREES or RADIANS).
    ///
    /// Returns:
    ///     numpy.ndarray: Keplerian elements [a, e, i, Ω, ω, M] where:
    ///         - a: semi-major axis [m]
    ///         - e: eccentricity [dimensionless]
    ///         - i: inclination [rad or deg]
    ///         - Ω: right ascension of ascending node [rad or deg]
    ///         - ω: argument of periapsis [rad or deg]
    ///         - M: mean anomaly [rad or deg]
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    ///     line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///
    ///     # Get elements in degrees
    ///     oe_deg = prop.get_elements(bh.AngleFormat.DEGREES)
    ///     print(f"Inclination: {oe_deg[2]:.4f} degrees")
    ///
    ///     # Get elements in radians
    ///     oe_rad = prop.get_elements(bh.AngleFormat.RADIANS)
    ///     print(f"Inclination: {oe_rad[2]:.4f} radians")
    ///     ```
    #[pyo3(text_signature = "(angle_format)")]
    pub fn get_elements<'a>(&self, py: Python<'a>, angle_format: &PyAngleFormat) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.propagator.get_elements(angle_format.value) {
            Ok(elements) => Ok(elements.as_slice().to_pyarray(py).to_owned()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get semi-major axis at TLE epoch.
    ///
    /// Returns:
    ///     float: Semi-major axis in meters.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    ///     line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///
    ///     sma = prop.semi_major_axis
    ///     print(f"Semi-major axis: {sma:.3f} m")
    ///     ```
    #[getter]
    pub fn semi_major_axis(&self) -> f64 {
        self.propagator.semi_major_axis()
    }

    /// Get eccentricity at TLE epoch.
    ///
    /// Returns:
    ///     float: Eccentricity (dimensionless).
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    ///     line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///
    ///     ecc = prop.eccentricity
    ///     print(f"Eccentricity: {ecc:.6f}")
    ///     ```
    #[getter]
    pub fn eccentricity(&self) -> f64 {
        self.propagator.eccentricity()
    }

    /// Get inclination at TLE epoch.
    ///
    /// Returns:
    ///     float: Inclination in degrees.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    ///     line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///
    ///     inc = prop.inclination
    ///     print(f"Inclination: {inc:.4f} deg")
    ///     ```
    #[getter]
    pub fn inclination(&self) -> f64 {
        self.propagator.inclination()
    }

    /// Get right ascension of ascending node at TLE epoch.
    ///
    /// Returns:
    ///     float: Right ascension of ascending node (RAAN) in degrees.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    ///     line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///
    ///     raan = prop.right_ascension
    ///     print(f"RAAN: {raan:.4f} deg")
    ///     ```
    #[getter]
    pub fn right_ascension(&self) -> f64 {
        self.propagator.right_ascension()
    }

    /// Get argument of periapsis at TLE epoch.
    ///
    /// Returns:
    ///     float: Argument of periapsis in degrees.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    ///     line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///
    ///     argp = prop.arg_perigee
    ///     print(f"Argument of periapsis: {argp:.4f} deg")
    ///     ```
    #[getter]
    pub fn arg_perigee(&self) -> f64 {
        self.propagator.arg_perigee()
    }

    /// Get mean anomaly at TLE epoch.
    ///
    /// Returns:
    ///     float: Mean anomaly in degrees.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    ///     line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///
    ///     ma = prop.mean_anomaly
    ///     print(f"Mean anomaly: {ma:.4f} deg")
    ///     ```
    #[getter]
    pub fn mean_anomaly(&self) -> f64 {
        self.propagator.mean_anomaly()
    }

    // Identity methods

    /// Set the name and return self (consuming constructor pattern).
    ///
    /// Args:
    ///     name (str): Name to assign to this propagator.
    ///
    /// Returns:
    ///     SGPPropagator: Self with name set.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2).with_name("My Satellite")
    ///     print(f"Name: {prop.name}")
    ///     ```
    pub fn with_name(mut slf: PyRefMut<'_, Self>, name: String) -> PyRefMut<'_, Self> {

        slf.propagator = slf.propagator.clone().with_name(&name);
        slf
    }

    /// Set the UUID and return self (consuming constructor pattern).
    ///
    /// Args:
    ///     uuid_str (str): UUID string to assign to this propagator.
    ///
    /// Returns:
    ///     SGPPropagator: Self with UUID set.
    pub fn with_uuid(mut slf: PyRefMut<'_, Self>, uuid_str: String) -> PyResult<PyRefMut<'_, Self>> {

        let uuid = uuid::Uuid::parse_str(&uuid_str)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        slf.propagator = slf.propagator.clone().with_uuid(uuid);
        Ok(slf)
    }

    /// Generate a new UUID, set it, and return self (consuming constructor pattern).
    ///
    /// Returns:
    ///     SGPPropagator: Self with new UUID set.
    pub fn with_new_uuid(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {

        slf.propagator = slf.propagator.clone().with_new_uuid();
        slf
    }

    /// Set the numeric ID and return self (consuming constructor pattern).
    ///
    /// Args:
    ///     id (int): Numeric ID to assign to this propagator.
    ///
    /// Returns:
    ///     SGPPropagator: Self with ID set.
    pub fn with_id(mut slf: PyRefMut<'_, Self>, id: u64) -> PyRefMut<'_, Self> {

        slf.propagator = slf.propagator.clone().with_id(id);
        slf
    }

    /// Set all identity fields at once and return self (consuming constructor pattern).
    ///
    /// Args:
    ///     name (str or None): Optional name to assign.
    ///     uuid_str (str or None): Optional UUID string to assign.
    ///     id (int or None): Optional numeric ID to assign.
    ///
    /// Returns:
    ///     SGPPropagator: Self with identity set.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import uuid
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     my_uuid = str(uuid.uuid4())
    ///     prop = bh.SGPPropagator.from_tle(line1, line2).with_identity("ISS", my_uuid, 25544)
    ///     print(f"Name: {prop.name}, ID: {prop.id}, UUID: {prop.uuid}")
    ///     ```
    pub fn with_identity(mut slf: PyRefMut<'_, Self>, name: Option<String>, uuid_str: Option<String>, id: Option<u64>) -> PyResult<PyRefMut<'_, Self>> {
        let uuid = match uuid_str {            
            Some(s) => Some(uuid::Uuid::parse_str(&s)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?),
            None => None,
        };
        slf.propagator = slf.propagator.clone().with_identity(name.as_deref(), uuid, id);
        Ok(slf)
    }

    /// Set all identity fields in-place (mutating).
    ///
    /// Args:
    ///     name (str or None): Optional name to assign.
    ///     uuid_str (str or None): Optional UUID string to assign.
    ///     id (int or None): Optional numeric ID to assign.
    pub fn set_identity(&mut self, name: Option<String>, uuid_str: Option<String>, id: Option<u64>) -> PyResult<()> {

        let uuid = match uuid_str {       
            Some(s) => Some(uuid::Uuid::parse_str(&s)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?),
            None => None,
        };
        self.propagator.set_identity(name.as_deref(), uuid, id);
        Ok(())
    }

    /// Set the numeric ID in-place (mutating).
    ///
    /// Args:
    ///     id (int or None): Numeric ID to assign, or None to clear.
    pub fn set_id(&mut self, id: Option<u64>) {
        self.propagator.set_id(id)
    }

    /// Set the name in-place (mutating).
    ///
    /// Args:
    ///     name (str or None): Name to assign, or None to clear.
    pub fn set_name(&mut self, name: Option<String>) {
        self.propagator.set_name(name.as_deref());    
    }

    /// Generate a new UUID and set it in-place (mutating).
    pub fn generate_uuid(&mut self) {
        self.propagator.generate_uuid()   
     }

    /// Get the current numeric ID.
    ///
    /// Returns:
    ///     int or None: The numeric ID, or None if not set.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2).with_id(25544)
    ///     print(f"ID: {prop.get_id()}")
    ///     ```
    pub fn get_id(&self) -> Option<u64> {
        self.propagator.get_id()
    }

    /// Get the current name.
    ///
    /// Returns:
    ///     str or None: The name, or None if not set.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     name = "ISS (ZARYA)"
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     prop = bh.SGPPropagator.from_3le(name, line1, line2)
    ///     print(f"Name: {prop.get_name()}")
    ///     ```
    pub fn get_name(&self) -> Option<String> {
        self.propagator.get_name().map(|s| s.to_string())
    }

    /// Get the current UUID.
    ///
    /// Returns:
    ///     str or None: The UUID as a string, or None if not set.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2).with_new_uuid()
    ///     print(f"UUID: {prop.get_uuid()}")
    ///     ```
    pub fn get_uuid(&self) -> Option<String> {
        self.propagator.get_uuid().map(|u| u.to_string())
    }

    /// Compute state as osculating elements at a specific epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///     angle_format (AngleFormat): If AngleFormat.DEGREES, angular elements are returned in degrees, otherwise in radians.
    ///
    /// Returns:
    ///     numpy.ndarray: Osculating elements [a, e, i, raan, argp, mean_anomaly].
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    ///     line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///
    ///     # Get osculating elements at initial epoch
    ///     epoch = prop.epoch
    ///     elements_deg = prop.state_koe(epoch, bh.AngleFormat.DEGREES)
    ///     print(f"Semi-major axis: {elements_deg[0]/1000:.3f} km")
    ///     print(f"Inclination: {elements_deg[2]:.4f} degrees")
    ///
    ///     # Get elements in radians
    ///     elements_rad = prop.state_koe(epoch, bh.AngleFormat.RADIANS)
    ///     print(f"Inclination: {elements_rad[2]:.4f} radians")
    ///     ```
    #[pyo3(text_signature = "(epoch, angle_format)")]
    pub fn state_koe<'a>(
        &self,
        py: Python<'a>,
        epoch: PyRef<PyEpoch>,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = SOrbitStateProvider::state_koe(&self.propagator, epoch.obj, angle_format.value)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute states as osculating elements at multiple epochs.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///     angle_format (AngleFormat): If AngleFormat.DEGREES, angular elements are returned in degrees, otherwise in radians.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of osculating element vectors [a, e, i, raan, argp, mean_anomaly].
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    ///     line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///
    ///     # Get elements at multiple epochs
    ///     epoch0 = prop.epoch
    ///     epochs = [epoch0 + i*3600.0 for i in range(10)]  # Every hour for 10 hours
    ///     elements_list = prop.states_koe(epochs, bh.AngleFormat.DEGREES)
    ///
    ///     for i, elements in enumerate(elements_list):
    ///         print(f"Hour {i}: a={elements[0]/1000:.3f} km, e={elements[1]:.6f}")
    ///     ```
    #[pyo3(text_signature = "(epochs, angle_format)")]
    pub fn states_koe<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = SOrbitStateProvider::states_koe(&self.propagator, &epoch_vec, angle_format.value)?;
        Ok(states.iter().map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned()).collect())
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!("SGPPropagator(norad_id={}, name={:?}, epoch={:?})",
                self.propagator.norad_id,
                self.propagator.satellite_name,
                self.propagator.epoch)
    }

    /// String conversion.
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Keplerian orbit propagator using two-body dynamics.
///
/// The Keplerian propagator implements ideal two-body orbital mechanics without
/// perturbations. It's fast and accurate for short time spans but doesn't account
/// for real-world effects like drag, J2, solar radiation pressure, etc.
///
/// Args:
///     epoch (Epoch): Initial epoch.
///     state (numpy.ndarray): 6-element state vector.
///     frame (OrbitFrame): Reference frame (ECI or ECEF).
///     representation (OrbitRepresentation): State representation (Cartesian or Keplerian).
///     angle_format (AngleFormat): Angle format for Keplerian elements.
///     step_size (float): Step size in seconds for propagation.
///
/// Attributes:
///     current_epoch (Epoch): Current propagation time
///     initial_epoch (Epoch): Initial epoch from propagator creation
///     step_size (float): Current step size in seconds
///     trajectory (OrbitTrajectory): Accumulated trajectory states
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Initial epoch and orbital elements
///     epc0 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///     oe = np.array([7000000.0, 0.001, 0.9, 0.0, 0.0, 0.0])  # a, e, i, RAAN, omega, M
///
///     # Create propagator from Keplerian elements
///     prop = bh.KeplerianPropagator.from_keplerian(
///         epc0, oe, bh.AngleFormat.RADIANS, step_size=60.0
///     )
///
///     # Propagate forward one orbit
///     period = bh.orbital_period(oe[0])
///     epc_future = epc0 + period
///     state = prop.state(epc_future)
///     print(f"State after one orbit: {state}")
///
///     # Create from Cartesian state
///     x_cart = np.array([7000000.0, 0.0, 0.0, 0.0, 7546.0, 0.0])
///     prop2 = bh.KeplerianPropagator(
///         epc0, x_cart, bh.OrbitFrame.ECI,
///         bh.OrbitRepresentation.CARTESIAN,
///         bh.AngleFormat.RADIANS, 60.0
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "KeplerianPropagator")]
pub struct PyKeplerianPropagator {
    pub propagator: propagators::KeplerianPropagator,
}

#[pymethods]
impl PyKeplerianPropagator {
    /// Create a new Keplerian propagator from state vector.
    ///
    /// Args:
    ///     epoch (Epoch): Initial epoch.
    ///     state (numpy.ndarray): 6-element state vector.
    ///     frame (OrbitFrame): Reference frame.
    ///     representation (OrbitRepresentation): State representation.
    ///     angle_format (AngleFormat): Angle format (only for Keplerian).
    ///     step_size (float): Step size in seconds for propagation.
    ///
    /// Returns:
    ///     KeplerianPropagator: New propagator instance.
    #[new]
    #[pyo3(signature = (epoch, state, frame, representation, angle_format, step_size))]
    pub fn new(
        epoch: PyRef<PyEpoch>,
        state: PyReadonlyArray1<f64>,
        frame: PyRef<PyOrbitFrame>,
        representation: PyRef<PyOrbitRepresentation>,
        angle_format: PyRef<PyAngleFormat>,
        step_size: f64,
    ) -> PyResult<Self> {
        let state_array = state.as_array();
        if state_array.len() != 6 {
            return Err(exceptions::PyValueError::new_err(
                "State vector must have exactly 6 elements"
            ));
        }

        let state_vec = na::Vector6::from_row_slice(state_array.as_slice().unwrap());

        let propagator = propagators::KeplerianPropagator::new(
            epoch.obj,
            state_vec,
            frame.frame,
            representation.representation,
            Some(angle_format.value),
            step_size,
        );

        Ok(PyKeplerianPropagator { propagator })
    }

    /// Create a new Keplerian propagator from Keplerian orbital elements.
    ///
    /// Args:
    ///     epoch (Epoch): Initial epoch.
    ///     elements (numpy.ndarray): 6-element Keplerian elements [a, e, i, raan, argp, mean_anomaly].
    ///     angle_format (AngleFormat): Angle format (Degrees or Radians).
    ///     step_size (float): Step size in seconds for propagation.
    ///
    /// Returns:
    ///     KeplerianPropagator: New propagator instance.
    #[classmethod]
    #[pyo3(signature = (epoch, elements, angle_format, step_size))]
    pub fn from_keplerian(
        _cls: &Bound<'_, PyType>,
        epoch: PyRef<PyEpoch>,
        elements: PyReadonlyArray1<f64>,
        angle_format: PyRef<PyAngleFormat>,
        step_size: f64,
    ) -> PyResult<Self> {
        let elements_array = elements.as_array();
        if elements_array.len() != 6 {
            return Err(exceptions::PyValueError::new_err(
                "Elements vector must have exactly 6 elements"
            ));
        }

        let elements_vec = na::Vector6::from_row_slice(elements_array.as_slice().unwrap());

        let propagator = propagators::KeplerianPropagator::from_keplerian(
            epoch.obj,
            elements_vec,
            angle_format.value,
            step_size,
        );

        Ok(PyKeplerianPropagator { propagator })
    }

    /// Create a new Keplerian propagator from Cartesian state in ECI frame.
    ///
    /// Args:
    ///     epoch (Epoch): Initial epoch.
    ///     state (numpy.ndarray): 6-element Cartesian state [x, y, z, vx, vy, vz] in ECI frame.
    ///     step_size (float): Step size in seconds for propagation.
    ///
    /// Returns:
    ///     KeplerianPropagator: New propagator instance.
    #[classmethod]
    #[pyo3(signature = (epoch, state, step_size))]
    pub fn from_eci(
        _cls: &Bound<'_, PyType>,
        epoch: PyRef<PyEpoch>,
        state: PyReadonlyArray1<f64>,
        step_size: f64,
    ) -> PyResult<Self> {
        let state_array = state.as_array();
        if state_array.len() != 6 {
            return Err(exceptions::PyValueError::new_err(
                "State vector must have exactly 6 elements"
            ));
        }

        let state_vec = na::Vector6::from_row_slice(state_array.as_slice().unwrap());

        let propagator = propagators::KeplerianPropagator::from_eci(
            epoch.obj,
            state_vec,
            step_size,
        );

        Ok(PyKeplerianPropagator { propagator })
    }

    /// Create a new Keplerian propagator from Cartesian state in ECEF frame.
    ///
    /// Args:
    ///     epoch (Epoch): Initial epoch.
    ///     state (numpy.ndarray): 6-element Cartesian state [x, y, z, vx, vy, vz] in ECEF frame.
    ///     step_size (float): Step size in seconds for propagation.
    ///
    /// Returns:
    ///     KeplerianPropagator: New propagator instance.
    #[classmethod]
    #[pyo3(signature = (epoch, state, step_size))]
    pub fn from_ecef(
        _cls: &Bound<'_, PyType>,
        epoch: PyRef<PyEpoch>,
        state: PyReadonlyArray1<f64>,
        step_size: f64,
    ) -> PyResult<Self> {
        let state_array = state.as_array();
        if state_array.len() != 6 {
            return Err(exceptions::PyValueError::new_err(
                "State vector must have exactly 6 elements"
            ));
        }

        let state_vec = na::Vector6::from_row_slice(state_array.as_slice().unwrap());

        let propagator = propagators::KeplerianPropagator::from_ecef(
            epoch.obj,
            state_vec,
            step_size,
        );

        Ok(PyKeplerianPropagator { propagator })
    }

    /// Get current epoch.
    ///
    /// Returns:
    ///     Epoch: Current propagator epoch.
    #[getter]
    pub fn current_epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.propagator.current_epoch() }
    }

    /// Get initial epoch.
    ///
    /// Returns:
    ///     Epoch: Initial propagator epoch.
    #[getter]
    pub fn initial_epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.propagator.initial_epoch() }
    }

    /// Get initial state.
    ///
    /// Returns:
    ///     numpy.ndarray: Initial state vector.
    #[pyo3(text_signature = "()")]
    pub fn initial_state<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray<f64, Ix1>> {
        let state = self.propagator.initial_state();
        state.as_slice().to_pyarray(py).to_owned()
    }

    /// Get step size in seconds.
    ///
    /// Returns:
    ///     float: Step size in seconds.
    #[getter]
    pub fn step_size(&self) -> f64 {
        self.propagator.step_size()
    }

    /// Set step size in seconds.
    ///
    /// Args:
    ///     step_size (float): New step size in seconds.
    #[setter]
    pub fn set_step_size(&mut self, step_size: f64) {
        self.propagator.set_step_size(step_size);
    }

    /// Set step size in seconds (explicit method).
    ///
    /// Args:
    ///     new_step_size (float): New step size in seconds.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     oe = np.array([bh.R_EARTH + 500e3, 0.01, 97.8, 15.0, 30.0, 45.0])
    ///     state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
    ///     propagator = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
    ///     propagator.set_step_size(120.0)  # Can use explicit method
    ///     # or propagator.step_size = 120.0  # Can use property
    ///     ```
    #[pyo3(name = "set_step_size", text_signature = "(new_step_size)")]
    pub fn set_step_size_explicit(&mut self, new_step_size: f64) {
        self.propagator.set_step_size(new_step_size);
    }

    /// Get current state vector.
    ///
    /// Returns:
    ///     numpy.ndarray: Current state vector.
    #[pyo3(text_signature = "()")]
    pub fn current_state<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray<f64, Ix1>> {
        let state = self.propagator.current_state();
        state.as_slice().to_pyarray(py).to_owned()
    }

    /// Step forward by the default step size.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
    ///     state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
    ///     prop.step()  # Advance by default step_size (60 seconds)
    ///     print(f"Advanced to: {prop.current_epoch}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn step(&mut self) {
        self.propagator.step();
    }

    /// Step forward by a specified time duration.
    ///
    /// Args:
    ///     step_size (float): Time step in seconds.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
    ///     state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
    ///     prop.step_by(120.0)  # Advance by 120 seconds
    ///     print(f"Advanced to: {prop.current_epoch}")
    ///     ```
    #[pyo3(text_signature = "(step_size)")]
    pub fn step_by(&mut self, step_size: f64) {
        self.propagator.step_by(step_size);
    }

    /// Step past a specified target epoch.
    ///
    /// Args:
    ///     target_epoch (Epoch): The epoch to step past.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
    ///     state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
    ///     target = epc + 300.0  # Target 5 minutes ahead
    ///     prop.step_past(target)
    ///     print(f"Advanced to: {prop.current_epoch}")
    ///     ```
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn step_past(&mut self, target_epoch: PyRef<PyEpoch>) {
        self.propagator.step_past(target_epoch.obj);
    }

    /// Propagate forward by specified number of steps.
    ///
    /// Args:
    ///     num_steps (int): Number of steps to take.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
    ///     state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
    ///     prop.propagate_steps(10)  # Take 10 steps (600 seconds total)
    ///     print(f"Advanced to: {prop.current_epoch}")
    ///     ```
    #[pyo3(text_signature = "(num_steps)")]
    pub fn propagate_steps(&mut self, num_steps: usize) {
        self.propagator.propagate_steps(num_steps);
    }

    /// Propagate to a specific target epoch.
    ///
    /// Args:
    ///     target_epoch (Epoch): The epoch to propagate to.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
    ///     state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
    ///     target = epc + 3600.0  # Propagate to 1 hour ahead
    ///     prop.propagate_to(target)
    ///     print(f"Propagated to: {prop.current_epoch}")
    ///     ```
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn propagate_to(&mut self, target_epoch: PyRef<PyEpoch>) {
        self.propagator.propagate_to(target_epoch.obj);
    }

    /// Reset propagator to initial conditions.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
    ///     state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
    ///     prop.propagate_steps(10)
    ///     prop.reset()  # Return to initial epoch and state
    ///     print(f"Reset to: {prop.current_epoch}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn reset(&mut self) {
        self.propagator.reset();
    }

    /// Set initial conditions.
    ///
    /// Args:
    ///     epoch (Epoch): Initial epoch.
    ///     state (numpy.ndarray): Initial state vector.
    ///     frame (OrbitFrame): Reference frame.
    ///     representation (OrbitRepresentation): State representation.
    ///     angle_format (AngleFormat): Angle format.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
    ///     state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
    ///
    ///     # Change initial conditions to a different orbit
    ///     new_oe = np.array([bh.R_EARTH + 800e3, 0.02, 1.2, 0.5, 0.3, 0.0])
    ///     new_state = bh.state_koe_to_eci(new_oe, bh.AngleFormat.RADIANS)
    ///     new_epc = bh.Epoch.from_datetime(2024, 1, 2, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     prop.set_initial_conditions(new_epc, new_state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, bh.AngleFormat.RADIANS)
    ///     print(f"New initial epoch: {prop.initial_epoch}")
    ///     ```
    #[pyo3(text_signature = "(epoch, state, frame, representation, angle_format)")]
    pub fn set_initial_conditions(
        &mut self,
        epoch: PyRef<PyEpoch>,
        state: PyReadonlyArray1<f64>,
        frame: PyRef<PyOrbitFrame>,
        representation: PyRef<PyOrbitRepresentation>,
        angle_format: PyRef<PyAngleFormat>,
    ) -> PyResult<()> {
        let state_array = state.as_array();
        if state_array.len() != 6 {
            return Err(exceptions::PyValueError::new_err(
                "State vector must have exactly 6 elements"
            ));
        }

        let state_vec = na::Vector6::from_row_slice(state_array.as_slice().unwrap());

        self.propagator.set_initial_conditions(
            epoch.obj,
            state_vec,
            frame.frame,
            representation.representation,
            Some(angle_format.value),
        );

        Ok(())
    }

    /// Set eviction policy to keep maximum number of states.
    ///
    /// Args:
    ///     max_size (int): Maximum number of states to retain.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
    ///     state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
    ///     prop.set_eviction_policy_max_size(100)  # Keep only 100 most recent states
    ///     prop.propagate_steps(200)
    ///     print(f"Trajectory length: {prop.trajectory.len()}")
    ///     ```
    #[pyo3(text_signature = "(max_size)")]
    pub fn set_eviction_policy_max_size(&mut self, max_size: usize) -> PyResult<()> {
        match self.propagator.set_eviction_policy_max_size(max_size) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Set eviction policy to keep states within maximum age.
    ///
    /// Args:
    ///     max_age (float): Maximum age in seconds.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
    ///     state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
    ///     prop.set_eviction_policy_max_age(3600.0)  # Keep only states within 1 hour
    ///     prop.propagate_to(epc + 7200.0)  # Propagate 2 hours
    ///     print(f"Trajectory length: {prop.trajectory.len()}")
    ///     ```
    #[pyo3(text_signature = "(max_age)")]
    pub fn set_eviction_policy_max_age(&mut self, max_age: f64) -> PyResult<()> {
        match self.propagator.set_eviction_policy_max_age(max_age) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Compute state at a specific epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector in the propagator's native format.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self.propagator.state(epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in ECI coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ECI frame.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_eci<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self.propagator.state_eci(epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in ECEF coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ECEF frame.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_ecef<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self.propagator.state_ecef(epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in GCRF coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in GCRF frame (meters, m/s).
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_gcrf<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self.propagator.state_gcrf(epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in EME2000 coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in EME2000 frame (meters, m/s).
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_eme2000<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self.propagator.state_eme2000(epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in ITRF coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ITRF frame (meters, m/s).
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_itrf<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self.propagator.state_itrf(epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state as osculating elements at a specific epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///     angle_format (AngleFormat): If AngleFormat.DEGREES, angular elements are returned in degrees, otherwise in radians.
    ///
    /// Returns:
    ///     numpy.ndarray: Osculating elements [a, e, i, raan, argp, mean_anomaly].
    #[pyo3(text_signature = "(epoch, angle_format)")]
    pub fn state_koe<'a>(
        &self,
        py: Python<'a>,
        epoch: PyRef<PyEpoch>,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_koe(&self.propagator, epoch.obj, angle_format.value)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute states at multiple epochs.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of state vectors in the propagator's native format.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DStateProvider::states(&self.propagator, &epoch_vec)?;
        Ok(states.iter().map(|s| s.as_slice().to_pyarray(py).to_owned()).collect())
    }

    /// Compute states at multiple epochs in ECI coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of ECI state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_eci<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_eci(&self.propagator, &epoch_vec)?;
        Ok(states.iter().map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned()).collect())
    }

    /// Compute states at multiple epochs in ECEF coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of ECEF state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_ecef<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_ecef(&self.propagator, &epoch_vec)?;
        Ok(states.iter().map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned()).collect())
    }

    /// Compute states at multiple epochs in GCRF coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of GCRF state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_gcrf<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_gcrf(&self.propagator, &epoch_vec)?;
        Ok(states.iter().map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned()).collect())
    }

    /// Compute states at multiple epochs in ITRF coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of ITRF state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_itrf<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_itrf(&self.propagator, &epoch_vec)?;
        Ok(states.iter().map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned()).collect())
    }

    /// Compute states as osculating elements at multiple epochs.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///     angle_format (AngleFormat): If AngleFormat.DEGREES, angular elements are returned in degrees, otherwise in radians.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of osculating element vectors.
    #[pyo3(text_signature = "(epochs, angle_format)")]
    pub fn states_koe<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_koe(&self.propagator, &epoch_vec, angle_format.value)?;
        Ok(states.iter().map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned()).collect())
    }

    /// Get accumulated trajectory.
    ///
    /// Returns:
    ///     OrbitalTrajectory: The accumulated trajectory.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
    ///     state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
    ///     prop.propagate_steps(10)
    ///     traj = prop.trajectory
    ///     print(f"Trajectory contains {traj.len()} states")
    ///     ```
    #[getter]
    pub fn trajectory(&self) -> PyOrbitalTrajectory {
        let traj = &self.propagator.trajectory;

        // Convert states from SVector to DVector
        let states: Vec<DVector<f64>> = traj.states.iter()
            .map(|s| DVector::from_column_slice(s.as_slice()))
            .collect();

        // Convert covariances from SMatrix to DMatrix if present
        let covariances = traj.covariances.as_ref().map(|covs| {
            covs.iter()
                .map(|c| DMatrix::from_row_slice(6, 6, c.as_slice()))
                .collect()
        });

        let mut d_traj = trajectories::DOrbitTrajectory::from_orbital_data(
            traj.epochs.clone(),
            states,
            traj.frame,
            traj.representation,
            traj.angle_format,
            covariances,
        );

        // Copy identity from original trajectory
        d_traj.set_identity(
            traj.get_name(),
            traj.get_uuid(),
            traj.get_id()
        );

        PyOrbitalTrajectory { trajectory: d_traj }
    }

    // Identity methods

    /// Set the name and return self (consuming constructor pattern).
    ///
    /// Args:
    ///     name (str): Name to assign to this propagator.
    ///
    /// Returns:
    ///     KeplerianPropagator: Self with name set.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     oe = np.array([7000e3, 0.001, 0.9, 0.0, 0.0, 0.0])
    ///     prop = bh.KeplerianPropagator.from_keplerian(
    ///         epc, oe, bh.AngleFormat.RADIANS, 60.0
    ///     ).with_name("My Orbit")
    ///     print(f"Name: {prop.name}")
    ///     ```
    pub fn with_name(mut slf: PyRefMut<'_, Self>, name: String) -> PyRefMut<'_, Self> {
        slf.propagator = slf.propagator.clone().with_name(&name);
        slf
    }

    /// Set the UUID and return self (consuming constructor pattern).
    ///
    /// Args:
    ///     uuid_str (str): UUID string to assign to this propagator.
    ///
    /// Returns:
    ///     KeplerianPropagator: Self with UUID set.
    pub fn with_uuid(mut slf: PyRefMut<'_, Self>, uuid_str: String) -> PyResult<PyRefMut<'_, Self>> {
        let uuid = uuid::Uuid::parse_str(&uuid_str)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        slf.propagator = slf.propagator.clone().with_uuid(uuid);
        Ok(slf)
    }

    /// Generate a new UUID, set it, and return self (consuming constructor pattern).
    ///
    /// Returns:
    ///     KeplerianPropagator: Self with new UUID set.
    pub fn with_new_uuid(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {

        slf.propagator = slf.propagator.clone().with_new_uuid();
        slf
    }

    /// Set the numeric ID and return self (consuming constructor pattern).
    ///
    /// Args:
    ///     id (int): Numeric ID to assign to this propagator.
    ///
    /// Returns:
    ///     KeplerianPropagator: Self with ID set.
    pub fn with_id(mut slf: PyRefMut<'_, Self>, id: u64) -> PyRefMut<'_, Self> {

        slf.propagator = slf.propagator.clone().with_id(id);
        slf
    }

    /// Set all identity fields at once and return self (consuming constructor pattern).
    ///
    /// Args:
    ///     name (str or None): Optional name to assign.
    ///     uuid_str (str or None): Optional UUID string to assign.
    ///     id (int or None): Optional numeric ID to assign.
    ///
    /// Returns:
    ///     KeplerianPropagator: Self with identity set.
    pub fn with_identity(mut slf: PyRefMut<'_, Self>, name: Option<String>, uuid_str: Option<String>, id: Option<u64>) -> PyResult<PyRefMut<'_, Self>> {

        let uuid = match uuid_str {
            Some(s) => Some(uuid::Uuid::parse_str(&s)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?),
            None => None,
        };
        slf.propagator = slf.propagator.clone().with_identity(name.as_deref(), uuid, id);
        Ok(slf)
    }

    /// Set all identity fields in-place (mutating).
    ///
    /// Args:
    ///     name (str or None): Optional name to assign.
    ///     uuid_str (str or None): Optional UUID string to assign.
    ///     id (int or None): Optional numeric ID to assign.
    pub fn set_identity(&mut self, name: Option<String>, uuid_str: Option<String>, id: Option<u64>) -> PyResult<()> {

        let uuid = match uuid_str {
            Some(s) => Some(uuid::Uuid::parse_str(&s)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?),
            None => None,
        };
        self.propagator.set_identity(name.as_deref(), uuid, id);
        Ok(())
    }

    /// Set the numeric ID in-place (mutating).
    ///
    /// Args:
    ///     id (int or None): Numeric ID to assign, or None to clear.
    pub fn set_id(&mut self, id: Option<u64>) {

        self.propagator.set_id(id)    }

    /// Set the name in-place (mutating).
    ///
    /// Args:
    ///     name (str or None): Name to assign, or None to clear.
    pub fn set_name(&mut self, name: Option<String>) {

        self.propagator.set_name(name.as_deref())    }

    /// Generate a new UUID and set it in-place (mutating).
    pub fn generate_uuid(&mut self) {

        self.propagator.generate_uuid()    }

    /// Get the current numeric ID.
    ///
    /// Returns:
    ///     int or None: The numeric ID, or None if not set.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
    ///     state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator.from_eci(epc, state, 60.0).with_id(12345)
    ///     print(f"ID: {prop.get_id()}")
    ///     ```
    pub fn get_id(&self) -> Option<u64> {
        self.propagator.get_id()
    }

    /// Get the current name.
    ///
    /// Returns:
    ///     str or None: The name, or None if not set.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
    ///     state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator.from_eci(epc, state, 60.0).with_name("MySat")
    ///     print(f"Name: {prop.get_name()}")
    ///     ```
    pub fn get_name(&self) -> Option<String> {
        self.propagator.get_name().map(|s| s.to_string())
    }

    /// Get the current UUID.
    ///
    /// Returns:
    ///     str or None: The UUID as a string, or None if not set.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
    ///     state = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator.from_eci(epc, state, 60.0).with_new_uuid()
    ///     print(f"UUID: {prop.get_uuid()}")
    ///     ```
    pub fn get_uuid(&self) -> Option<String> {
        self.propagator.get_uuid().map(|u| u.to_string())
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!("KeplerianPropagator(epoch={:?}, step_size={})",
                self.propagator.current_epoch(),
                self.propagator.step_size())
    }

    /// String conversion.
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Propagate multiple propagators to a target epoch in parallel.
///
/// This function takes a list of propagators and calls `propagate_to` on each one
/// in parallel using the global thread pool. Each propagator's internal state is updated
/// to reflect the new epoch.
///
/// All propagators in the list must be of the same type (either all `KeplerianPropagator`
/// or all `SGPPropagator`). Mixing propagator types is not supported.
///
/// Args:
///     propagators (List[KeplerianPropagator] or List[SGPPropagator]): List of propagators to update.
///     target_epoch (Epoch): The epoch to propagate all propagators to.
///
/// Returns:
///     None: Propagators are updated in place.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
///
///     epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///
///     # Create multiple propagators
///     propagators = []
///     for i in range(10):
///         oe = np.array([bh.R_EARTH + 500e3 + i*10e3, 0.001, 98.0, i*10.0, 0.0, 0.0])
///         state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
///         prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
///         propagators.append(prop)
///
///     # Propagate all to target epoch in parallel
///     target = epoch + 3600.0  # 1 hour later
///     bh.par_propagate_to(propagators, target)
///
///     # All propagators are now at target epoch
///     for prop in propagators:
///         assert prop.current_epoch() == target
///     ```
#[pyfunction(name = "par_propagate_to")]
fn py_par_propagate_to(
    propagators: &Bound<'_, PyAny>,
    target_epoch: &PyEpoch,
) -> PyResult<()> {
    use pyo3::types::PyList;

    // Check if propagators is a list
    if !propagators.is_instance_of::<PyList>() {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "propagators must be a list of KeplerianPropagator or SGPPropagator"
        ));
    }

    let prop_list = propagators.cast::<PyList>()?;
    if prop_list.is_empty() {
        return Ok(()); // No propagators to process
    }

    // Determine propagator type from first element
    let first = prop_list.get_item(0)?;

    if first.is_instance_of::<PyKeplerianPropagator>() {
        // Process as Keplerian propagators
        let mut props: Vec<propagators::KeplerianPropagator> = Vec::new();

        for item in prop_list.iter() {
            let py_prop = item.cast::<PyKeplerianPropagator>()?;
            props.push(py_prop.borrow().propagator.clone());
        }

        // Call Rust parallel propagation function
        propagators::par_propagate_to(&mut props, target_epoch.obj);

        // Update Python objects with new state
        for (i, item) in prop_list.iter().enumerate() {
            let mut py_prop = item.cast::<PyKeplerianPropagator>()?.borrow_mut();
            py_prop.propagator = props[i].clone();
        }

        Ok(())
    } else if first.is_instance_of::<PySGPPropagator>() {
        // Process as SGP propagators
        let mut props: Vec<propagators::SGPPropagator> = Vec::new();

        for item in prop_list.iter() {
            let py_prop = item.cast::<PySGPPropagator>()?;
            props.push(py_prop.borrow().propagator.clone());
        }

        // Call Rust parallel propagation function
        propagators::par_propagate_to(&mut props, target_epoch.obj);

        // Update Python objects with new state
        for (i, item) in prop_list.iter().enumerate() {
            let mut py_prop = item.cast::<PySGPPropagator>()?.borrow_mut();
            py_prop.propagator = props[i].clone();
        }

        Ok(())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "propagators must be a list of KeplerianPropagator or SGPPropagator"
        ))
    }
}

// =============================================================================
// Configuration Classes for Numerical Propagation
// =============================================================================

/// Integration method for numerical orbit propagation.
///
/// Specifies which numerical integrator to use. Different methods trade off
/// accuracy, efficiency, and applicability.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Use default (DP54)
///     method = bh.IntegrationMethod.DP54
///
///     # Create config with specific method
///     config = bh.NumericalPropagationConfig.with_method(bh.IntegrationMethod.RKF45)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "IntegrationMethod")]
#[derive(Clone)]
pub struct PyIntegrationMethod {
    pub method: propagators::IntegratorMethod,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyIntegrationMethod {
    /// Classical 4th-order Runge-Kutta (fixed-step)
    #[classattr]
    fn RK4() -> Self {
        PyIntegrationMethod { method: propagators::IntegratorMethod::RK4 }
    }

    /// Runge-Kutta-Fehlberg 4(5) adaptive
    #[classattr]
    fn RKF45() -> Self {
        PyIntegrationMethod { method: propagators::IntegratorMethod::RKF45 }
    }

    /// Dormand-Prince 5(4) adaptive (default, MATLAB's ode45)
    #[classattr]
    fn DP54() -> Self {
        PyIntegrationMethod { method: propagators::IntegratorMethod::DP54 }
    }

    /// Runge-Kutta-Nystrom 12(10) adaptive (high-precision)
    #[classattr]
    fn RKN1210() -> Self {
        PyIntegrationMethod { method: propagators::IntegratorMethod::RKN1210 }
    }

    /// Returns true if this integrator uses adaptive step size control.
    pub fn is_adaptive(&self) -> bool {
        self.method.is_adaptive()
    }

    fn __repr__(&self) -> String {
        format!("IntegrationMethod.{:?}", self.method)
    }
}

/// Configuration for STM and sensitivity matrix propagation.
///
/// Controls whether the propagator computes and stores variational matrices
/// (State Transition Matrix and Sensitivity Matrix) during propagation.
///
/// Args:
///     enable_stm (bool): Enable State Transition Matrix propagation. Defaults to False.
///     enable_sensitivity (bool): Enable sensitivity matrix propagation. Defaults to False.
///     store_stm_history (bool): Store STM at output times. Defaults to False.
///     store_sensitivity_history (bool): Store sensitivity at output times. Defaults to False.
///
/// Attributes:
///     enable_stm (bool): Enable State Transition Matrix propagation
///     enable_sensitivity (bool): Enable sensitivity matrix propagation
///     store_stm_history (bool): Store STM at output times
///     store_sensitivity_history (bool): Store sensitivity at output times
///
/// Example:
///     ```python
///     import brahe as bh
///
///     config = bh.VariationalConfig(enable_stm=True, store_stm_history=True)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "VariationalConfig")]
#[derive(Clone)]
pub struct PyVariationalConfig {
    pub config: propagators::VariationalConfig,
}

#[pymethods]
impl PyVariationalConfig {
    /// Create a new variational configuration.
    ///
    /// Args:
    ///     enable_stm (bool): Enable State Transition Matrix propagation. Defaults to False.
    ///     enable_sensitivity (bool): Enable sensitivity matrix propagation. Defaults to False.
    ///     store_stm_history (bool): Store STM at output times. Defaults to False.
    ///     store_sensitivity_history (bool): Store sensitivity at output times. Defaults to False.
    ///
    /// Returns:
    ///     VariationalConfig: A new configuration with the specified settings
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Default (all disabled)
    ///     config = bh.VariationalConfig()
    ///
    ///     # Enable STM propagation
    ///     config = bh.VariationalConfig(enable_stm=True)
    ///
    ///     # Enable STM with history storage
    ///     config = bh.VariationalConfig(enable_stm=True, store_stm_history=True)
    ///     ```
    #[new]
    #[pyo3(signature = (enable_stm=false, enable_sensitivity=false, store_stm_history=false, store_sensitivity_history=false))]
    fn new(
        enable_stm: bool,
        enable_sensitivity: bool,
        store_stm_history: bool,
        store_sensitivity_history: bool,
    ) -> Self {
        PyVariationalConfig {
            config: propagators::VariationalConfig {
                enable_stm,
                enable_sensitivity,
                store_stm_history,
                store_sensitivity_history,
                ..Default::default()
            },
        }
    }

    /// Enable State Transition Matrix (STM) propagation.
    #[getter]
    fn get_enable_stm(&self) -> bool {
        self.config.enable_stm
    }

    #[setter]
    fn set_enable_stm(&mut self, value: bool) {
        self.config.enable_stm = value;
    }

    /// Enable sensitivity matrix propagation.
    #[getter]
    fn get_enable_sensitivity(&self) -> bool {
        self.config.enable_sensitivity
    }

    #[setter]
    fn set_enable_sensitivity(&mut self, value: bool) {
        self.config.enable_sensitivity = value;
    }

    /// Store STM at output times in trajectory.
    #[getter]
    fn get_store_stm_history(&self) -> bool {
        self.config.store_stm_history
    }

    #[setter]
    fn set_store_stm_history(&mut self, value: bool) {
        self.config.store_stm_history = value;
    }

    /// Store sensitivity matrix at output times in trajectory.
    #[getter]
    fn get_store_sensitivity_history(&self) -> bool {
        self.config.store_sensitivity_history
    }

    #[setter]
    fn set_store_sensitivity_history(&mut self, value: bool) {
        self.config.store_sensitivity_history = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "VariationalConfig(enable_stm={}, enable_sensitivity={}, store_stm_history={}, store_sensitivity_history={})",
            self.config.enable_stm, self.config.enable_sensitivity,
            self.config.store_stm_history, self.config.store_sensitivity_history
        )
    }
}

/// Atmospheric density model selection.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     model = bh.AtmosphericModel.HARRIS_PRIESTER
///     model = bh.AtmosphericModel.NRLMSISE00
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "AtmosphericModel")]
#[derive(Clone)]
pub struct PyAtmosphericModel {
    pub model: propagators::AtmosphericModel,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyAtmosphericModel {
    /// Harris-Priester atmospheric model (fast, no space weather required)
    #[classattr]
    fn HARRIS_PRIESTER() -> Self {
        PyAtmosphericModel { model: propagators::AtmosphericModel::HarrisPriester }
    }

    /// NRLMSISE-00 empirical model (high-fidelity, requires space weather)
    #[classattr]
    fn NRLMSISE00() -> Self {
        PyAtmosphericModel { model: propagators::AtmosphericModel::NRLMSISE00 }
    }

    /// Create exponential atmosphere model with custom parameters.
    ///
    /// Args:
    ///     scale_height (float): Scale height in meters.
    ///     rho0 (float): Reference density in kg/m³.
    ///     h0 (float): Reference altitude in meters.
    #[classmethod]
    fn exponential(_cls: &Bound<'_, PyType>, scale_height: f64, rho0: f64, h0: f64) -> Self {
        PyAtmosphericModel {
            model: propagators::AtmosphericModel::Exponential { scale_height, rho0, h0 }
        }
    }

    fn __repr__(&self) -> String {
        format!("AtmosphericModel({:?})", self.model)
    }
}

/// Eclipse model for solar radiation pressure calculations.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     eclipse = bh.EclipseModel.CONICAL
///     eclipse = bh.EclipseModel.CYLINDRICAL
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "EclipseModel")]
#[derive(Clone)]
pub struct PyEclipseModel {
    pub model: propagators::EclipseModel,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyEclipseModel {
    /// Conical eclipse model (accurate, accounts for penumbra)
    #[classattr]
    fn CONICAL() -> Self {
        PyEclipseModel { model: propagators::EclipseModel::Conical }
    }

    /// Cylindrical eclipse model (simpler, faster)
    #[classattr]
    fn CYLINDRICAL() -> Self {
        PyEclipseModel { model: propagators::EclipseModel::Cylindrical }
    }

    fn __repr__(&self) -> String {
        format!("EclipseModel.{:?}", self.model)
    }
}

// =============================================================================
// Parameter Source
// =============================================================================

/// Source for a parameter value (fixed or from parameter vector).
///
/// Allows specifying whether a parameter comes from a fixed value or from
/// an index in the parameter vector.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Fixed drag coefficient
///     cd = bh.ParameterSource.value(2.2)
///
///     # Variable mass from parameter vector index 0
///     mass = bh.ParameterSource.parameter_index(0)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "ParameterSource")]
#[derive(Clone)]
pub struct PyParameterSource {
    pub source: propagators::ParameterSource,
}

#[pymethods]
impl PyParameterSource {
    /// Create a fixed value parameter source.
    ///
    /// Args:
    ///     value (float): The fixed parameter value.
    ///
    /// Returns:
    ///     ParameterSource: A parameter source with a fixed value.
    #[classmethod]
    fn value(_cls: &Bound<'_, PyType>, value: f64) -> Self {
        PyParameterSource {
            source: propagators::ParameterSource::Value(value),
        }
    }

    /// Create a parameter index source.
    ///
    /// Args:
    ///     index (int): Index into the parameter vector.
    ///
    /// Returns:
    ///     ParameterSource: A parameter source referencing a parameter vector index.
    #[classmethod]
    fn parameter_index(_cls: &Bound<'_, PyType>, index: usize) -> Self {
        PyParameterSource {
            source: propagators::ParameterSource::ParameterIndex(index),
        }
    }

    /// Check if this source is a fixed value.
    fn is_value(&self) -> bool {
        matches!(self.source, propagators::ParameterSource::Value(_))
    }

    /// Check if this source references a parameter index.
    fn is_parameter_index(&self) -> bool {
        matches!(self.source, propagators::ParameterSource::ParameterIndex(_))
    }

    /// Get the fixed value (if this is a Value source).
    ///
    /// Returns:
    ///     float or None: The fixed value, or None if this is a ParameterIndex.
    fn get_value(&self) -> Option<f64> {
        match self.source {
            propagators::ParameterSource::Value(v) => Some(v),
            _ => None,
        }
    }

    /// Get the parameter index (if this is a ParameterIndex source).
    ///
    /// Returns:
    ///     int or None: The parameter index, or None if this is a Value.
    fn get_index(&self) -> Option<usize> {
        match self.source {
            propagators::ParameterSource::ParameterIndex(i) => Some(i),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.source {
            propagators::ParameterSource::Value(v) => format!("ParameterSource.value({})", v),
            propagators::ParameterSource::ParameterIndex(i) => format!("ParameterSource.parameter_index({})", i),
        }
    }
}

// =============================================================================
// Gravity Configuration
// =============================================================================

/// Gravity model configuration.
///
/// Specifies the gravity model: point mass or spherical harmonic expansion.
///
/// Args:
///     degree (int, optional): Maximum degree of spherical harmonic expansion.
///         If None, uses point mass gravity.
///     order (int, optional): Maximum order of spherical harmonic expansion.
///         If None, uses point mass gravity.
///     use_global (bool): If True, use global gravity model. Otherwise load EGM2008.
///         Default is False.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Simple two-body point mass gravity (default)
///     gravity = bh.GravityConfiguration()
///
///     # Spherical harmonic with 20x20 degree/order
///     gravity = bh.GravityConfiguration(degree=20, order=20)
///
///     # Alternative: use class methods
///     gravity = bh.GravityConfiguration.point_mass()
///     gravity = bh.GravityConfiguration.spherical_harmonic(degree=20, order=20)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "GravityConfiguration")]
#[derive(Clone)]
pub struct PyGravityConfiguration {
    pub config: propagators::GravityConfiguration,
}

#[pymethods]
impl PyGravityConfiguration {
    /// Create a gravity configuration.
    ///
    /// Args:
    ///     degree (int, optional): Maximum degree of spherical harmonic expansion.
    ///         If None, uses point mass gravity.
    ///     order (int, optional): Maximum order of spherical harmonic expansion.
    ///         If None, uses point mass gravity.
    ///     use_global (bool): If True, use global gravity model. Otherwise load EGM2008.
    ///
    /// Returns:
    ///     GravityConfiguration: Gravity configuration (point mass or spherical harmonic).
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Point mass (default)
    ///     gravity = bh.GravityConfiguration()
    ///
    ///     # Spherical harmonic
    ///     gravity = bh.GravityConfiguration(degree=20, order=20)
    ///     ```
    #[new]
    #[pyo3(signature = (degree=None, order=None, use_global=false))]
    fn new(degree: Option<usize>, order: Option<usize>, use_global: bool) -> Self {
        match (degree, order) {
            (Some(d), Some(o)) => {
                let source = if use_global {
                    propagators::GravityModelSource::Global
                } else {
                    propagators::GravityModelSource::ModelType(crate::orbit_dynamics::gravity::GravityModelType::EGM2008_360)
                };
                PyGravityConfiguration {
                    config: propagators::GravityConfiguration::SphericalHarmonic { source, degree: d, order: o },
                }
            }
            _ => PyGravityConfiguration {
                config: propagators::GravityConfiguration::PointMass,
            },
        }
    }

    /// Create a point mass gravity configuration.
    ///
    /// Returns:
    ///     GravityConfiguration: Point mass (two-body) gravity.
    #[classmethod]
    fn point_mass(_cls: &Bound<'_, PyType>) -> Self {
        PyGravityConfiguration {
            config: propagators::GravityConfiguration::PointMass,
        }
    }

    /// Create a spherical harmonic gravity configuration.
    ///
    /// Args:
    ///     degree (int): Maximum degree of expansion.
    ///     order (int): Maximum order of expansion.
    ///     use_global (bool): If True, use global gravity model. Otherwise load EGM2008.
    ///
    /// Returns:
    ///     GravityConfiguration: Spherical harmonic gravity.
    #[classmethod]
    #[pyo3(signature = (degree, order, use_global=false))]
    fn spherical_harmonic(_cls: &Bound<'_, PyType>, degree: usize, order: usize, use_global: bool) -> Self {
        let source = if use_global {
            propagators::GravityModelSource::Global
        } else {
            propagators::GravityModelSource::ModelType(crate::orbit_dynamics::gravity::GravityModelType::EGM2008_360)
        };
        PyGravityConfiguration {
            config: propagators::GravityConfiguration::SphericalHarmonic { source, degree, order },
        }
    }

    /// Check if this is point mass gravity.
    fn is_point_mass(&self) -> bool {
        matches!(self.config, propagators::GravityConfiguration::PointMass)
    }

    /// Check if this is spherical harmonic gravity.
    fn is_spherical_harmonic(&self) -> bool {
        matches!(self.config, propagators::GravityConfiguration::SphericalHarmonic { .. })
    }

    /// Get the degree (for spherical harmonic).
    ///
    /// Returns:
    ///     int or None: Degree if spherical harmonic, None otherwise.
    fn get_degree(&self) -> Option<usize> {
        match &self.config {
            propagators::GravityConfiguration::SphericalHarmonic { degree, .. } => Some(*degree),
            _ => None,
        }
    }

    /// Get the order (for spherical harmonic).
    ///
    /// Returns:
    ///     int or None: Order if spherical harmonic, None otherwise.
    fn get_order(&self) -> Option<usize> {
        match &self.config {
            propagators::GravityConfiguration::SphericalHarmonic { order, .. } => Some(*order),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.config {
            propagators::GravityConfiguration::PointMass => "GravityConfiguration.point_mass()".to_string(),
            propagators::GravityConfiguration::SphericalHarmonic { degree, order, .. } => {
                format!("GravityConfiguration.spherical_harmonic(degree={}, order={})", degree, order)
            }
        }
    }
}

// =============================================================================
// Drag Configuration
// =============================================================================

/// Atmospheric drag configuration.
///
/// Defines the atmospheric model and drag parameters.
///
/// Args:
///     model (AtmosphericModel): Atmospheric density model.
///     area (ParameterSource): Drag cross-sectional area source [m²].
///     cd (ParameterSource): Drag coefficient source (dimensionless).
///
/// Attributes:
///     model (AtmosphericModel): Atmospheric density model
///     area (ParameterSource): Drag area source
///     cd (ParameterSource): Drag coefficient source
///
/// Example:
///     ```python
///     import brahe as bh
///
///     drag = bh.DragConfiguration(
///         model=bh.AtmosphericModel.HARRIS_PRIESTER,
///         area=bh.ParameterSource.parameter_index(1),
///         cd=bh.ParameterSource.value(2.2)
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "DragConfiguration")]
#[derive(Clone)]
pub struct PyDragConfiguration {
    pub config: propagators::DragConfiguration,
}

#[pymethods]
impl PyDragConfiguration {
    /// Create a new drag configuration.
    ///
    /// Args:
    ///     model (AtmosphericModel): Atmospheric density model.
    ///     area (ParameterSource): Drag cross-sectional area source [m²].
    ///     cd (ParameterSource): Drag coefficient source (dimensionless).
    #[new]
    #[pyo3(signature = (model, area, cd))]
    fn new(model: &PyAtmosphericModel, area: &PyParameterSource, cd: &PyParameterSource) -> Self {
        PyDragConfiguration {
            config: propagators::DragConfiguration {
                model: model.model.clone(),
                area: area.source.clone(),
                cd: cd.source.clone(),
            },
        }
    }

    /// Get the atmospheric model.
    #[getter]
    fn model(&self) -> PyAtmosphericModel {
        PyAtmosphericModel { model: self.config.model.clone() }
    }

    /// Set the atmospheric model.
    #[setter]
    fn set_model(&mut self, model: &PyAtmosphericModel) {
        self.config.model = model.model.clone();
    }

    /// Get the drag area parameter source.
    #[getter]
    fn area(&self) -> PyParameterSource {
        PyParameterSource { source: self.config.area.clone() }
    }

    /// Set the drag area parameter source.
    #[setter]
    fn set_area(&mut self, area: &PyParameterSource) {
        self.config.area = area.source.clone();
    }

    /// Get the drag coefficient parameter source.
    #[getter]
    fn cd(&self) -> PyParameterSource {
        PyParameterSource { source: self.config.cd.clone() }
    }

    /// Set the drag coefficient parameter source.
    #[setter]
    fn set_cd(&mut self, cd: &PyParameterSource) {
        self.config.cd = cd.source.clone();
    }

    fn __repr__(&self) -> String {
        format!("DragConfiguration(model={:?}, area={:?}, cd={:?})",
                self.config.model, self.config.area, self.config.cd)
    }
}

// =============================================================================
// SRP Configuration
// =============================================================================

/// Solar radiation pressure configuration.
///
/// Defines the SRP parameters and eclipse model.
///
/// Args:
///     area (ParameterSource): SRP cross-sectional area source [m²].
///     cr (ParameterSource): Coefficient of reflectivity source (dimensionless).
///     eclipse_model (EclipseModel): Eclipse model for shadow effects.
///
/// Attributes:
///     area (ParameterSource): SRP area source
///     cr (ParameterSource): Reflectivity coefficient source
///     eclipse_model (EclipseModel): Eclipse model
///
/// Example:
///     ```python
///     import brahe as bh
///
///     srp = bh.SolarRadiationPressureConfiguration(
///         area=bh.ParameterSource.parameter_index(3),
///         cr=bh.ParameterSource.parameter_index(4),
///         eclipse_model=bh.EclipseModel.CONICAL
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "SolarRadiationPressureConfiguration")]
#[derive(Clone)]
pub struct PySolarRadiationPressureConfiguration {
    pub config: propagators::SolarRadiationPressureConfiguration,
}

#[pymethods]
impl PySolarRadiationPressureConfiguration {
    /// Create a new SRP configuration.
    ///
    /// Args:
    ///     area (ParameterSource): SRP cross-sectional area source [m²].
    ///     cr (ParameterSource): Coefficient of reflectivity source (dimensionless).
    ///     eclipse_model (EclipseModel): Eclipse model for shadow effects.
    #[new]
    #[pyo3(signature = (area, cr, eclipse_model))]
    fn new(area: &PyParameterSource, cr: &PyParameterSource, eclipse_model: &PyEclipseModel) -> Self {
        PySolarRadiationPressureConfiguration {
            config: propagators::SolarRadiationPressureConfiguration {
                area: area.source.clone(),
                cr: cr.source.clone(),
                eclipse_model: eclipse_model.model.clone(),
            },
        }
    }

    /// Get the SRP area parameter source.
    #[getter]
    fn area(&self) -> PyParameterSource {
        PyParameterSource { source: self.config.area.clone() }
    }

    /// Set the SRP area parameter source.
    #[setter]
    fn set_area(&mut self, area: &PyParameterSource) {
        self.config.area = area.source.clone();
    }

    /// Get the coefficient of reflectivity parameter source.
    #[getter]
    fn cr(&self) -> PyParameterSource {
        PyParameterSource { source: self.config.cr.clone() }
    }

    /// Set the coefficient of reflectivity parameter source.
    #[setter]
    fn set_cr(&mut self, cr: &PyParameterSource) {
        self.config.cr = cr.source.clone();
    }

    /// Get the eclipse model.
    #[getter]
    fn eclipse_model(&self) -> PyEclipseModel {
        PyEclipseModel { model: self.config.eclipse_model.clone() }
    }

    /// Set the eclipse model.
    #[setter]
    fn set_eclipse_model(&mut self, eclipse_model: &PyEclipseModel) {
        self.config.eclipse_model = eclipse_model.model.clone();
    }

    fn __repr__(&self) -> String {
        format!("SolarRadiationPressureConfiguration(area={:?}, cr={:?}, eclipse_model={:?})",
                self.config.area, self.config.cr, self.config.eclipse_model)
    }
}

// =============================================================================
// Third Body Configuration
// =============================================================================

/// Third-body perturber.
///
/// Celestial bodies that can act as gravitational perturbers.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     sun = bh.ThirdBody.SUN
///     moon = bh.ThirdBody.MOON
///     ```
#[pyclass(module = "brahe._brahe", eq, eq_int)]
#[pyo3(name = "ThirdBody")]
#[derive(Clone, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyThirdBody {
    SUN,
    MOON,
    MERCURY,
    VENUS,
    MARS,
    JUPITER,
    SATURN,
    URANUS,
    NEPTUNE,
}

impl From<PyThirdBody> for propagators::ThirdBody {
    fn from(body: PyThirdBody) -> Self {
        match body {
            PyThirdBody::SUN => propagators::ThirdBody::Sun,
            PyThirdBody::MOON => propagators::ThirdBody::Moon,
            PyThirdBody::MERCURY => propagators::ThirdBody::Mercury,
            PyThirdBody::VENUS => propagators::ThirdBody::Venus,
            PyThirdBody::MARS => propagators::ThirdBody::Mars,
            PyThirdBody::JUPITER => propagators::ThirdBody::Jupiter,
            PyThirdBody::SATURN => propagators::ThirdBody::Saturn,
            PyThirdBody::URANUS => propagators::ThirdBody::Uranus,
            PyThirdBody::NEPTUNE => propagators::ThirdBody::Neptune,
        }
    }
}

impl From<propagators::ThirdBody> for PyThirdBody {
    fn from(body: propagators::ThirdBody) -> Self {
        match body {
            propagators::ThirdBody::Sun => PyThirdBody::SUN,
            propagators::ThirdBody::Moon => PyThirdBody::MOON,
            propagators::ThirdBody::Mercury => PyThirdBody::MERCURY,
            propagators::ThirdBody::Venus => PyThirdBody::VENUS,
            propagators::ThirdBody::Mars => PyThirdBody::MARS,
            propagators::ThirdBody::Jupiter => PyThirdBody::JUPITER,
            propagators::ThirdBody::Saturn => PyThirdBody::SATURN,
            propagators::ThirdBody::Uranus => PyThirdBody::URANUS,
            propagators::ThirdBody::Neptune => PyThirdBody::NEPTUNE,
        }
    }
}

/// Third-body perturbations configuration.
///
/// Defines which celestial bodies to include and ephemeris source.
///
/// Args:
///     ephemeris_source (EphemerisSource): Source for celestial body ephemerides.
///     bodies (list[ThirdBody]): List of bodies to include as perturbers.
///
/// Attributes:
///     ephemeris_source (EphemerisSource): Ephemeris source
///     bodies (list[ThirdBody]): List of perturbing bodies
///
/// Example:
///     ```python
///     import brahe as bh
///
///     third_body = bh.ThirdBodyConfiguration(
///         ephemeris_source=bh.EphemerisSource.DE440s,
///         bodies=[bh.ThirdBody.SUN, bh.ThirdBody.MOON]
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "ThirdBodyConfiguration")]
#[derive(Clone)]
pub struct PyThirdBodyConfiguration {
    pub config: propagators::ThirdBodyConfiguration,
}

#[pymethods]
impl PyThirdBodyConfiguration {
    /// Create a new third-body configuration.
    ///
    /// Args:
    ///     ephemeris_source (EphemerisSource): Source for celestial body ephemerides.
    ///     bodies (list[ThirdBody]): List of bodies to include as perturbers.
    #[new]
    #[pyo3(signature = (ephemeris_source, bodies))]
    fn new(ephemeris_source: PyEphemerisSource, bodies: Vec<PyThirdBody>) -> Self {
        PyThirdBodyConfiguration {
            config: propagators::ThirdBodyConfiguration {
                ephemeris_source: ephemeris_source.into(),
                bodies: bodies.into_iter().map(|b| b.into()).collect(),
            },
        }
    }

    /// Get the ephemeris source.
    #[getter]
    fn ephemeris_source(&self) -> PyEphemerisSource {
        match self.config.ephemeris_source {
            propagators::EphemerisSource::LowPrecision => PyEphemerisSource::LowPrecision,
            propagators::EphemerisSource::DE440s => PyEphemerisSource::DE440s,
            propagators::EphemerisSource::DE440 => PyEphemerisSource::DE440,
        }
    }

    /// Set the ephemeris source.
    #[setter]
    fn set_ephemeris_source(&mut self, source: PyEphemerisSource) {
        self.config.ephemeris_source = source.into();
    }

    /// Get the list of third bodies.
    #[getter]
    fn bodies(&self) -> Vec<PyThirdBody> {
        self.config.bodies.iter().map(|b| b.clone().into()).collect()
    }

    /// Set the list of third bodies.
    #[setter]
    fn set_bodies(&mut self, bodies: Vec<PyThirdBody>) {
        self.config.bodies = bodies.into_iter().map(|b| b.into()).collect();
    }

    fn __repr__(&self) -> String {
        format!("ThirdBodyConfiguration(ephemeris_source={:?}, bodies={:?})",
                self.config.ephemeris_source, self.config.bodies)
    }
}

/// Configuration for numerical propagation.
///
/// Controls the integrator settings, tolerances, and variational equation options.
///
/// Note:
///     This class is created via class methods or static methods:
///     - `NumericalPropagationConfig.default()` - DP54 with standard tolerances
///     - `NumericalPropagationConfig.high_precision()` - RKN1210 with tight tolerances
///     - `NumericalPropagationConfig.with_method(method)` - Custom method with default settings
///     - `NumericalPropagationConfig.new(method, integrator, variational)` - Full customization
///
/// Attributes:
///     method (IntegrationMethod): Integration method
///     integrator (IntegratorConfig): Integrator configuration (tolerances, step sizes)
///     variational (VariationalConfig): Variational configuration (STM/sensitivity settings)
///     trajectory_mode (TrajectoryMode): Trajectory storage mode
///     sampling (SamplingConfig): Output sampling configuration
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Default configuration (DP54 with standard tolerances)
///     config = bh.NumericalPropagationConfig.default()
///
///     # High precision configuration
///     config = bh.NumericalPropagationConfig.high_precision()
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "NumericalPropagationConfig")]
#[derive(Clone)]
pub struct PyNumericalPropagationConfig {
    pub config: propagators::NumericalPropagationConfig,
}

#[pymethods]
impl PyNumericalPropagationConfig {
    /// Create a default configuration (DP54 with standard tolerances).
    #[classmethod]
    fn default(_cls: &Bound<'_, PyType>) -> Self {
        PyNumericalPropagationConfig {
            config: propagators::NumericalPropagationConfig::default(),
        }
    }

    /// Create a configuration with a specific integration method.
    ///
    /// Args:
    ///     method (IntegrationMethod): The integration method to use.
    #[classmethod]
    fn with_method(_cls: &Bound<'_, PyType>, method: &PyIntegrationMethod) -> Self {
        PyNumericalPropagationConfig {
            config: propagators::NumericalPropagationConfig::with_method(method.method),
        }
    }

    /// Create a high-precision configuration (RKN1210 with tight tolerances).
    #[classmethod]
    fn high_precision(_cls: &Bound<'_, PyType>) -> Self {
        PyNumericalPropagationConfig {
            config: propagators::NumericalPropagationConfig::high_precision(),
        }
    }

    /// Creates a new numerical propagation configuration with all components specified.
    ///
    /// Args:
    ///     method (IntegrationMethod): Integration method to use
    ///     integrator (IntegratorConfig): Integrator configuration (tolerances, step sizes)
    ///     variational (VariationalConfig): Variational configuration (STM/sensitivity settings)
    ///
    /// Returns:
    ///     NumericalPropagationConfig: A new configuration with the specified settings
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     config = bh.NumericalPropagationConfig(
    ///         bh.IntegrationMethod.DP54,
    ///         bh.IntegratorConfig.adaptive(1e-10, 1e-8),
    ///         bh.VariationalConfig(),
    ///     )
    ///     ```
    #[new]
    #[pyo3(signature = (method, integrator, variational))]
    fn new(
        method: &PyIntegrationMethod,
        integrator: &PyIntegratorConfig,
        variational: &PyVariationalConfig,
    ) -> Self {
        PyNumericalPropagationConfig {
            config: propagators::NumericalPropagationConfig::new(
                method.method,
                integrator.inner.clone(),
                variational.config.clone(),
            ),
        }
    }

    /// Set absolute tolerance for adaptive integrators.
    ///
    /// Args:
    ///     abs_tol (float): Absolute tolerance.
    ///
    /// Returns:
    ///     NumericalPropagationConfig: Self with updated tolerance.
    fn with_abs_tol(mut slf: PyRefMut<'_, Self>, abs_tol: f64) -> PyRefMut<'_, Self> {
        slf.config.integrator.abs_tol = abs_tol;
        slf
    }

    /// Set relative tolerance for adaptive integrators.
    ///
    /// Args:
    ///     rel_tol (float): Relative tolerance.
    ///
    /// Returns:
    ///     NumericalPropagationConfig: Self with updated tolerance.
    fn with_rel_tol(mut slf: PyRefMut<'_, Self>, rel_tol: f64) -> PyRefMut<'_, Self> {
        slf.config.integrator.rel_tol = rel_tol;
        slf
    }

    /// Set initial step size.
    ///
    /// Args:
    ///     step (float): Initial step size in seconds.
    ///
    /// Returns:
    ///     NumericalPropagationConfig: Self with updated step size.
    fn with_initial_step(mut slf: PyRefMut<'_, Self>, step: f64) -> PyRefMut<'_, Self> {
        slf.config.integrator.initial_step = Some(step);
        slf
    }

    /// Set maximum step size.
    ///
    /// Args:
    ///     step (float): Maximum step size in seconds.
    ///
    /// Returns:
    ///     NumericalPropagationConfig: Self with updated step size.
    fn with_max_step(mut slf: PyRefMut<'_, Self>, step: f64) -> PyRefMut<'_, Self> {
        slf.config.integrator.max_step = Some(step);
        slf
    }

    /// Enable STM (State Transition Matrix) propagation.
    ///
    /// Returns:
    ///     NumericalPropagationConfig: Self with STM enabled.
    fn with_stm(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.config.variational.enable_stm = true;
        slf
    }

    /// Enable sensitivity matrix propagation (requires params).
    ///
    /// Returns:
    ///     NumericalPropagationConfig: Self with sensitivity enabled.
    fn with_sensitivity(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.config.variational.enable_sensitivity = true;
        slf
    }

    /// Enable STM history storage in trajectory.
    ///
    /// Returns:
    ///     NumericalPropagationConfig: Self with STM history enabled.
    fn with_stm_history(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.config.variational.store_stm_history = true;
        slf
    }

    /// Enable sensitivity history storage in trajectory.
    ///
    /// Returns:
    ///     NumericalPropagationConfig: Self with sensitivity history enabled.
    fn with_sensitivity_history(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.config.variational.store_sensitivity_history = true;
        slf
    }

    /// Get the integration method.
    #[getter]
    fn method(&self) -> PyIntegrationMethod {
        PyIntegrationMethod { method: self.config.method }
    }

    /// Get absolute tolerance.
    #[getter]
    fn abs_tol(&self) -> f64 {
        self.config.integrator.abs_tol
    }

    /// Get relative tolerance.
    #[getter]
    fn rel_tol(&self) -> f64 {
        self.config.integrator.rel_tol
    }

    /// Get variational configuration (STM/sensitivity settings).
    #[getter]
    fn variational(&self) -> PyVariationalConfig {
        PyVariationalConfig { config: self.config.variational.clone() }
    }

    /// Set variational configuration (STM/sensitivity settings).
    #[setter]
    fn set_variational(&mut self, value: PyVariationalConfig) {
        self.config.variational = value.config;
    }

    fn __repr__(&self) -> String {
        format!("NumericalPropagationConfig(method={:?}, abs_tol={}, rel_tol={})",
                self.config.method, self.config.integrator.abs_tol, self.config.integrator.rel_tol)
    }
}

/// Force model configuration for numerical orbit propagation.
///
/// Defines all perturbation forces to be included: gravity, drag, SRP, third-body, relativity.
///
/// Args:
///     gravity (GravityConfiguration, optional): Gravity model configuration.
///         Default is point mass gravity.
///     drag (DragConfiguration, optional): Atmospheric drag configuration.
///         Default is None (disabled).
///     srp (SolarRadiationPressureConfiguration, optional): Solar radiation pressure configuration.
///         Default is None (disabled).
///     third_body (ThirdBodyConfiguration, optional): Third-body perturbations configuration.
///         Default is None (disabled).
///     relativity (bool): Enable relativistic corrections. Default is False.
///     mass (ParameterSource, optional): Spacecraft mass source. Default is None.
///
/// Attributes:
///     gravity (GravityConfiguration): Gravity model configuration
///     drag (DragConfiguration or None): Atmospheric drag configuration
///     srp (SolarRadiationPressureConfiguration or None): Solar radiation pressure configuration
///     third_body (ThirdBodyConfiguration or None): Third-body perturbations configuration
///     relativity (bool): Enable relativistic corrections
///     mass (ParameterSource or None): Spacecraft mass source
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create with explicit parameters
///     config = bh.ForceModelConfig(
///         gravity=bh.GravityConfiguration(degree=20, order=20),
///         relativity=True,
///     )
///
///     # Or use convenience class methods
///     config = bh.ForceModelConfig.default()
///     config = bh.ForceModelConfig.two_body()
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "ForceModelConfig")]
#[derive(Clone)]
pub struct PyForceModelConfig {
    pub config: propagators::ForceModelConfig,
}

#[pymethods]
impl PyForceModelConfig {
    /// Create a force model configuration.
    ///
    /// Args:
    ///     gravity (GravityConfiguration, optional): Gravity model configuration.
    ///         Default is point mass gravity.
    ///     drag (DragConfiguration, optional): Atmospheric drag configuration.
    ///     srp (SolarRadiationPressureConfiguration, optional): Solar radiation pressure configuration.
    ///     third_body (ThirdBodyConfiguration, optional): Third-body perturbations configuration.
    ///     relativity (bool): Enable relativistic corrections. Default is False.
    ///     mass (ParameterSource, optional): Spacecraft mass source.
    ///
    /// Returns:
    ///     ForceModelConfig: A force model configuration.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     config = bh.ForceModelConfig(
    ///         gravity=bh.GravityConfiguration(degree=20, order=20),
    ///         relativity=True,
    ///     )
    ///     ```
    #[new]
    #[pyo3(signature = (gravity=None, drag=None, srp=None, third_body=None, relativity=false, mass=None))]
    fn new(
        gravity: Option<&PyGravityConfiguration>,
        drag: Option<&PyDragConfiguration>,
        srp: Option<&PySolarRadiationPressureConfiguration>,
        third_body: Option<&PyThirdBodyConfiguration>,
        relativity: bool,
        mass: Option<&PyParameterSource>,
    ) -> Self {
        PyForceModelConfig {
            config: propagators::ForceModelConfig {
                gravity: gravity.map(|g| g.config.clone()).unwrap_or(propagators::GravityConfiguration::PointMass),
                drag: drag.map(|d| d.config.clone()),
                srp: srp.map(|s| s.config.clone()),
                third_body: third_body.map(|t| t.config.clone()),
                relativity,
                mass: mass.map(|m| m.source.clone()),
            },
        }
    }

    /// Create a default force model configuration.
    ///
    /// Includes:
    /// - 20x20 EGM2008 gravity
    /// - Harris-Priester atmospheric drag
    /// - Solar radiation pressure with conical eclipse
    /// - Sun and Moon third-body perturbations
    ///
    /// Requires parameter vector: [mass, drag_area, Cd, srp_area, Cr]
    #[classmethod]
    fn default(_cls: &Bound<'_, PyType>) -> Self {
        PyForceModelConfig {
            config: propagators::ForceModelConfig::default(),
        }
    }

    /// Create a high-fidelity force model configuration.
    ///
    /// Includes:
    /// - 120x120 EGM2008 gravity
    /// - NRLMSISE-00 atmospheric model
    /// - SRP with conical eclipse
    /// - Sun, Moon, and all planets (DE440s ephemerides)
    /// - Relativistic corrections
    ///
    /// Requires parameter vector: [mass, drag_area, Cd, srp_area, Cr]
    #[classmethod]
    fn high_fidelity(_cls: &Bound<'_, PyType>) -> Self {
        PyForceModelConfig {
            config: propagators::ForceModelConfig::high_fidelity(),
        }
    }

    /// Create an Earth gravity-only configuration (no drag, SRP, or third-body).
    ///
    /// Uses 20x20 EGM2008 gravity. No parameter vector required.
    #[classmethod]
    fn earth_gravity(_cls: &Bound<'_, PyType>) -> Self {
        PyForceModelConfig {
            config: propagators::ForceModelConfig::earth_gravity(),
        }
    }

    /// Create a two-body (point mass) gravity configuration.
    ///
    /// Uses only central body gravity with no perturbations.
    /// Produces results equivalent to Keplerian propagation.
    /// No parameter vector required.
    #[classmethod]
    fn two_body(_cls: &Bound<'_, PyType>) -> Self {
        PyForceModelConfig {
            config: propagators::ForceModelConfig::two_body_gravity(),
        }
    }

    /// Create a conservative forces configuration (gravity + third-body + relativity, no drag/SRP).
    #[classmethod]
    fn conservative_forces(_cls: &Bound<'_, PyType>) -> Self {
        PyForceModelConfig {
            config: propagators::ForceModelConfig::conservative_forces(),
        }
    }

    /// Create a configuration suitable for LEO satellites.
    ///
    /// Includes drag and higher-order gravity, plus SRP and third-body.
    /// Requires parameter vector: [mass, drag_area, Cd, srp_area, Cr]
    #[classmethod]
    fn leo_default(_cls: &Bound<'_, PyType>) -> Self {
        PyForceModelConfig {
            config: propagators::ForceModelConfig::leo_default(),
        }
    }

    /// Create a configuration suitable for GEO satellites.
    ///
    /// Includes SRP and third-body perturbations, omits drag.
    /// Requires parameter vector: [mass, _, _, srp_area, Cr]
    #[classmethod]
    fn geo_default(_cls: &Bound<'_, PyType>) -> Self {
        PyForceModelConfig {
            config: propagators::ForceModelConfig::geo_default(),
        }
    }

    /// Check if this configuration requires a parameter vector.
    pub fn requires_params(&self) -> bool {
        self.config.requires_params()
    }

    // =========================================================================
    // Field Accessors
    // =========================================================================

    /// Get the gravity configuration.
    #[getter]
    fn gravity(&self) -> PyGravityConfiguration {
        PyGravityConfiguration { config: self.config.gravity.clone() }
    }

    /// Set the gravity configuration.
    #[setter]
    fn set_gravity(&mut self, gravity: &PyGravityConfiguration) {
        self.config.gravity = gravity.config.clone();
    }

    /// Get the drag configuration (None if disabled).
    #[getter]
    fn drag(&self) -> Option<PyDragConfiguration> {
        self.config.drag.as_ref().map(|d| PyDragConfiguration { config: d.clone() })
    }

    /// Set the drag configuration (None to disable).
    #[setter]
    fn set_drag(&mut self, drag: Option<&PyDragConfiguration>) {
        self.config.drag = drag.map(|d| d.config.clone());
    }

    /// Get the solar radiation pressure configuration (None if disabled).
    #[getter]
    fn srp(&self) -> Option<PySolarRadiationPressureConfiguration> {
        self.config.srp.as_ref().map(|s| PySolarRadiationPressureConfiguration { config: s.clone() })
    }

    /// Set the solar radiation pressure configuration (None to disable).
    #[setter]
    fn set_srp(&mut self, srp: Option<&PySolarRadiationPressureConfiguration>) {
        self.config.srp = srp.map(|s| s.config.clone());
    }

    /// Get the third-body configuration (None if disabled).
    #[getter]
    fn third_body(&self) -> Option<PyThirdBodyConfiguration> {
        self.config.third_body.as_ref().map(|t| PyThirdBodyConfiguration { config: t.clone() })
    }

    /// Set the third-body configuration (None to disable).
    #[setter]
    fn set_third_body(&mut self, third_body: Option<&PyThirdBodyConfiguration>) {
        self.config.third_body = third_body.map(|t| t.config.clone());
    }

    /// Get whether relativistic corrections are enabled.
    #[getter]
    fn relativity(&self) -> bool {
        self.config.relativity
    }

    /// Set whether relativistic corrections are enabled.
    #[setter]
    fn set_relativity(&mut self, enabled: bool) {
        self.config.relativity = enabled;
    }

    /// Get the mass parameter source (None if not required).
    #[getter]
    fn mass(&self) -> Option<PyParameterSource> {
        self.config.mass.as_ref().map(|m| PyParameterSource { source: m.clone() })
    }

    /// Set the mass parameter source (None to not track mass).
    #[setter]
    fn set_mass(&mut self, mass: Option<&PyParameterSource>) {
        self.config.mass = mass.map(|m| m.source.clone());
    }

    fn __repr__(&self) -> String {
        format!("ForceModelConfig(requires_params={})", self.config.requires_params())
    }
}

// =============================================================================
// NumericalOrbitPropagator
// =============================================================================

/// High-fidelity numerical orbit propagator with configurable force models.
///
/// This propagator uses numerical integration with built-in orbital force models:
/// - Gravity (point mass or spherical harmonic)
/// - Atmospheric drag (Harris-Priester, NRLMSISE-00, or exponential)
/// - Solar radiation pressure with eclipse modeling
/// - Third-body perturbations (Sun, Moon, planets)
/// - Relativistic corrections
///
/// Args:
///     epoch (Epoch): Initial epoch.
///     state (numpy.ndarray): Initial state vector in ECI Cartesian [x, y, z, vx, vy, vz] (meters, m/s).
///         Can be 6D or 6+N dimensional for extended state.
///     propagation_config (NumericalPropagationConfig): Propagation configuration.
///     force_config (ForceModelConfig): Force model configuration.
///     params (numpy.ndarray or None): Parameter vector [mass, drag_area, Cd, srp_area, Cr, ...].
///         Required if force_config references parameter indices.
///     initial_covariance (numpy.ndarray or None): Optional 6x6 initial covariance matrix (enables STM).
///     additional_dynamics (callable or None): Optional function for extended state dynamics.
///         Signature: f(t, state, params) -> derivative.
///     control_input (callable or None): Optional control input function for thrust accelerations.
///         Signature: f(t, state, params) -> 3D acceleration vector.
///
/// Attributes:
///     current_epoch (Epoch): Current propagation time
///     initial_epoch (Epoch): Initial epoch from propagator creation
///     state_dim (int): Dimension of state vector (6 for basic, 6+N for extended)
///     step_size (float): Current integration step size in seconds
///     trajectory (OrbitTrajectory): Accumulated trajectory states
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
///
///     # Create initial state (ECI Cartesian)
///     epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///     oe = np.array([bh.R_EARTH + 500e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
///
///     # Parameters: [mass, drag_area, Cd, srp_area, Cr]
///     params = np.array([1000.0, 10.0, 2.2, 10.0, 1.3])
///
///     # Create propagator with default configs
///     prop = bh.NumericalOrbitPropagator(
///         epoch, state,
///         bh.NumericalPropagationConfig.default(),
///         bh.ForceModelConfig.default(),
///         params
///     )
///
///     # Propagate
///     prop.propagate_to(epoch + 3600.0)  # 1 hour
///     print(f"Final state: {prop.current_state()}")
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "NumericalOrbitPropagator")]
pub struct PyNumericalOrbitPropagator {
    pub propagator: propagators::DNumericalOrbitPropagator,
}

#[pymethods]
impl PyNumericalOrbitPropagator {
    /// Create a new numerical orbit propagator.
    ///
    /// Args:
    ///     epoch (Epoch): Initial epoch.
    ///     state (numpy.ndarray): Initial state vector in ECI Cartesian [x, y, z, vx, vy, vz] (meters, m/s).
    ///                           Can be 6D or 6+N dimensional for extended state.
    ///     propagation_config (NumericalPropagationConfig): Propagation configuration.
    ///     force_config (ForceModelConfig): Force model configuration.
    ///     params (numpy.ndarray or None): Parameter vector [mass, drag_area, Cd, srp_area, Cr, ...].
    ///                                    Required if force_config references parameter indices.
    ///     initial_covariance (numpy.ndarray or None): Optional 6x6 initial covariance matrix (enables STM).
    ///     additional_dynamics (callable or None): Optional function for extended state dynamics beyond 6D.
    ///                                            Signature: f(t, state, params) -> derivative.
    ///                                            Should return derivatives for extended state elements only.
    ///     control_input (callable or None): Optional control input function for continuous control accelerations.
    ///                                      Signature: f(t, state, params) -> acceleration.
    ///                                      Should return 3D acceleration vector to add to orbital dynamics.
    ///
    /// Returns:
    ///     NumericalOrbitPropagator: New propagator instance.
    ///
    /// Raises:
    ///     RuntimeError: If configuration is invalid or params are missing when required.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     # Extended state with additional dynamics
    ///     def additional_dyn(t, state, params):
    ///         # state[6:] contains extended state
    ///         return np.array([state[6] * 0.1])  # Simple decay
    ///
    ///     # Control input (thrust acceleration)
    ///     def control(t, state, params):
    ///         return np.array([0.0, 0.0, 1e-6])  # Small z-acceleration
    ///
    ///     prop = bh.NumericalOrbitPropagator(
    ///         epoch, extended_state, prop_config, force_config,
    ///         params=params,
    ///         additional_dynamics=additional_dyn,
    ///         control_input=control
    ///     )
    ///     ```
    #[new]
    #[pyo3(signature = (epoch, state, propagation_config, force_config, params=None, initial_covariance=None, additional_dynamics=None, control_input=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        epoch: &PyEpoch,
        state: PyReadonlyArray1<f64>,
        propagation_config: &PyNumericalPropagationConfig,
        force_config: &PyForceModelConfig,
        params: Option<PyReadonlyArray1<f64>>,
        initial_covariance: Option<PyReadonlyArray2<f64>>,
        additional_dynamics: Option<Py<PyAny>>,
        control_input: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let state_vec = nalgebra::DVector::from_column_slice(state.as_slice()?);

        let params_vec = params.map(|p| nalgebra::DVector::from_column_slice(p.as_slice().unwrap()));

        let cov_matrix = if let Some(cov) = initial_covariance {
            let cov_shape = cov.shape();
            if cov_shape[0] != 6 || cov_shape[1] != 6 {
                return Err(exceptions::PyValueError::new_err(
                    "Initial covariance must be a 6x6 matrix"
                ));
            }
            let cov_data: Vec<f64> = cov.as_slice()?.to_vec();
            Some(nalgebra::DMatrix::from_row_slice(6, 6, &cov_data))
        } else {
            None
        };

        // Wrap additional_dynamics Python callable if provided
        let additional_dynamics_fn: Option<crate::integrators::traits::DStateDynamics> =
            additional_dynamics.map(|dyn_py| {
                let dyn_py = dyn_py.clone_ref(py);
                Box::new(
                    move |t: f64, x: &nalgebra::DVector<f64>, p: Option<&nalgebra::DVector<f64>>| {
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
                                    let res_arr: PyReadonlyArray1<f64> = res.extract(py).unwrap();
                                    nalgebra::DVector::from_column_slice(res_arr.as_slice().unwrap())
                                }
                                Err(e) => {
                                    eprintln!("Error calling additional_dynamics function: {}", e);
                                    nalgebra::DVector::zeros(x.len())
                                }
                            }
                        })
                    },
                ) as crate::integrators::traits::DStateDynamics
            });

        // Wrap control_input Python callable if provided
        let control_input_fn: crate::integrators::traits::DControlInput =
            control_input.map(|ctrl_py| {
                let ctrl_py = ctrl_py.clone_ref(py);
                Box::new(
                    move |t: f64, x: &nalgebra::DVector<f64>, p: Option<&nalgebra::DVector<f64>>| {
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
                                    let res_arr: PyReadonlyArray1<f64> = res.extract(py).unwrap();
                                    nalgebra::DVector::from_column_slice(res_arr.as_slice().unwrap())
                                }
                                Err(e) => {
                                    eprintln!("Error calling control_input function: {}", e);
                                    nalgebra::DVector::zeros(3) // Control should return 3D acceleration
                                }
                            }
                        })
                    },
                ) as Box<dyn Fn(f64, &nalgebra::DVector<f64>, Option<&nalgebra::DVector<f64>>) -> nalgebra::DVector<f64> + Send + Sync>
            });

        let prop = propagators::DNumericalOrbitPropagator::new(
            epoch.obj,
            state_vec,
            propagation_config.config.clone(),
            force_config.config.clone(),
            params_vec,
            additional_dynamics_fn,
            control_input_fn,
            cov_matrix,
        ).map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyNumericalOrbitPropagator { propagator: prop })
    }

    /// Create a propagator from ECI Cartesian state with simplified configuration.
    ///
    /// Args:
    ///     epoch (Epoch): Initial epoch.
    ///     state (numpy.ndarray): Initial ECI Cartesian state [x, y, z, vx, vy, vz].
    ///     params (numpy.ndarray or None): Parameter vector.
    ///     force_config (ForceModelConfig or None): Force model config (default if None).
    ///
    /// Returns:
    ///     NumericalOrbitPropagator: New propagator instance.
    #[classmethod]
    #[pyo3(signature = (epoch, state, params=None, force_config=None))]
    pub fn from_eci(
        _cls: &Bound<'_, PyType>,
        epoch: &PyEpoch,
        state: PyReadonlyArray1<f64>,
        params: Option<PyReadonlyArray1<f64>>,
        force_config: Option<&PyForceModelConfig>,
    ) -> PyResult<Self> {
        let state_slice = state.as_slice()?;
        if state_slice.len() < 6 {
            return Err(exceptions::PyValueError::new_err(
                "State vector must have at least 6 elements"
            ));
        }

        let state_vec = nalgebra::DVector::from_column_slice(state_slice);
        let params_vec = params.map(|p| nalgebra::DVector::from_column_slice(p.as_slice().unwrap()));

        let fc = force_config
            .map(|c| c.config.clone())
            .unwrap_or_default();

        let prop = propagators::DNumericalOrbitPropagator::new(
            epoch.obj,
            state_vec,
            propagators::NumericalPropagationConfig::default(),
            fc,
            params_vec,
            None,
            None,
            None,
        ).map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyNumericalOrbitPropagator { propagator: prop })
    }

    // =========================================================================
    // DStatePropagator trait methods
    // =========================================================================

    /// Get current epoch.
    #[getter]
    pub fn current_epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.propagator.current_epoch() }
    }

    /// Get initial epoch.
    #[getter]
    pub fn initial_epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.propagator.initial_epoch() }
    }

    /// Get current state vector.
    #[pyo3(text_signature = "()")]
    pub fn current_state<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray<f64, Ix1>> {
        let state = self.propagator.current_state();
        state.as_slice().to_pyarray(py).to_owned()
    }

    /// Get initial state vector.
    #[pyo3(text_signature = "()")]
    pub fn initial_state<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray<f64, Ix1>> {
        let state = DStatePropagator::initial_state(&self.propagator);
        state.as_slice().to_pyarray(py).to_owned()
    }

    /// Get state dimension.
    #[getter]
    pub fn state_dim(&self) -> usize {
        DStatePropagator::state_dim(&self.propagator)
    }

    /// Get current step size.
    #[getter]
    pub fn step_size(&self) -> f64 {
        DStatePropagator::step_size(&self.propagator)
    }

    /// Set step size.
    #[setter]
    pub fn set_step_size(&mut self, step_size: f64) {
        DStatePropagator::set_step_size(&mut self.propagator, step_size);
    }

    /// Step forward by the default step size.
    #[pyo3(text_signature = "()")]
    pub fn step(&mut self) {
        DStatePropagator::step(&mut self.propagator);
    }

    /// Step forward by a specified time duration.
    ///
    /// Args:
    ///     step_size (float): Time step in seconds.
    #[pyo3(text_signature = "(step_size)")]
    pub fn step_by(&mut self, step_size: f64) {
        DStatePropagator::step_by(&mut self.propagator, step_size);
    }

    /// Step past a specified target epoch.
    ///
    /// Args:
    ///     target_epoch (Epoch): The epoch to step past.
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn step_past(&mut self, target_epoch: &PyEpoch) {
        DStatePropagator::step_past(&mut self.propagator, target_epoch.obj);
    }

    /// Propagate forward by specified number of steps.
    ///
    /// Args:
    ///     num_steps (int): Number of steps to take.
    #[pyo3(text_signature = "(num_steps)")]
    pub fn propagate_steps(&mut self, num_steps: usize) {
        DStatePropagator::propagate_steps(&mut self.propagator, num_steps);
    }

    /// Propagate to a specific target epoch.
    ///
    /// Args:
    ///     target_epoch (Epoch): The epoch to propagate to.
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn propagate_to(&mut self, target_epoch: &PyEpoch) {
        DStatePropagator::propagate_to(&mut self.propagator, target_epoch.obj);
    }

    /// Reset propagator to initial conditions.
    #[pyo3(text_signature = "()")]
    pub fn reset(&mut self) {
        DStatePropagator::reset(&mut self.propagator);
    }

    // =========================================================================
    // State provider methods
    // =========================================================================

    /// Compute state at a specific epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector at the requested epoch.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = DStateProvider::state(&self.propagator, epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in ECI coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ECI frame.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_eci<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_eci(&self.propagator, epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in ECEF coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ECEF frame.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_ecef<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_ecef(&self.propagator, epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in GCRF coordinates.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_gcrf<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_gcrf(&self.propagator, epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in ITRF coordinates.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_itrf<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_itrf(&self.propagator, epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in EME2000 coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in EME2000 frame.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_eme2000<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_eme2000(&self.propagator, epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state as Keplerian elements at a specific epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch.
    ///     angle_format (AngleFormat): Format for angular elements.
    ///
    /// Returns:
    ///     numpy.ndarray: Keplerian elements [a, e, i, Ω, ω, M].
    #[pyo3(text_signature = "(epoch, angle_format)")]
    pub fn state_koe<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_koe(&self.propagator, epoch.obj, angle_format.value)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    // =========================================================================
    // Trajectory and variational methods
    // =========================================================================

    /// Get accumulated trajectory.
    ///
    /// Returns:
    ///     OrbitTrajectory: The accumulated trajectory.
    #[getter]
    pub fn trajectory(&self) -> PyOrbitalTrajectory {
        PyOrbitalTrajectory { trajectory: self.propagator.trajectory().clone() }
    }

    /// Get current STM (State Transition Matrix) if enabled.
    ///
    /// Returns:
    ///     numpy.ndarray or None: The current STM (n x n matrix), or None if STM not enabled.
    #[pyo3(text_signature = "()")]
    pub fn stm<'a>(&self, py: Python<'a>) -> Option<Bound<'a, PyArray<f64, Ix2>>> {
        self.propagator.stm().map(|stm| {
            let n = stm.nrows();
            let flat: Vec<f64> = (0..n).flat_map(|i| (0..n).map(move |j| stm[(i, j)])).collect();
            flat.into_pyarray(py).reshape([n, n]).unwrap()
        })
    }

    /// Get current sensitivity matrix if enabled.
    ///
    /// Returns:
    ///     numpy.ndarray or None: The current sensitivity (n x p matrix), or None if not enabled.
    #[pyo3(text_signature = "()")]
    pub fn sensitivity<'a>(&self, py: Python<'a>) -> Option<Bound<'a, PyArray<f64, Ix2>>> {
        self.propagator.sensitivity().map(|sens| {
            let n = sens.nrows();
            let p = sens.ncols();
            let flat: Vec<f64> = (0..n).flat_map(|i| (0..p).map(move |j| sens[(i, j)])).collect();
            flat.into_pyarray(py).reshape([n, p]).unwrap()
        })
    }

    /// Get covariance at a specific epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch.
    ///
    /// Returns:
    ///     numpy.ndarray: Covariance matrix at the requested epoch.
    #[pyo3(text_signature = "(epoch)")]
    pub fn covariance<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix2>>> {
        let cov = DCovarianceProvider::covariance(&self.propagator, epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let n = cov.nrows();
        let mut flat = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                flat.push(cov[(i, j)]);
            }
        }
        Ok(flat.into_pyarray(py).reshape([n, n]).unwrap())
    }

    // =========================================================================
    // Eviction policy
    // =========================================================================

    /// Set trajectory eviction policy based on maximum size.
    ///
    /// Args:
    ///     max_size (int): Maximum number of states to keep in trajectory.
    #[pyo3(text_signature = "(max_size)")]
    pub fn set_eviction_policy_max_size(&mut self, max_size: usize) -> PyResult<()> {
        DStatePropagator::set_eviction_policy_max_size(&mut self.propagator, max_size)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Set trajectory eviction policy based on maximum age.
    ///
    /// Args:
    ///     max_age (float): Maximum age in seconds to keep states in trajectory.
    #[pyo3(text_signature = "(max_age)")]
    pub fn set_eviction_policy_max_age(&mut self, max_age: f64) -> PyResult<()> {
        DStatePropagator::set_eviction_policy_max_age(&mut self.propagator, max_age)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    // =========================================================================
    // Identity methods
    // =========================================================================

    /// Set the name and return self.
    pub fn with_name(mut slf: PyRefMut<'_, Self>, name: String) -> PyRefMut<'_, Self> {
        slf.propagator.name = Some(name);
        slf
    }

    /// Set the UUID and return self.
    pub fn with_uuid(mut slf: PyRefMut<'_, Self>, uuid_str: String) -> PyResult<PyRefMut<'_, Self>> {
        let uuid = uuid::Uuid::parse_str(&uuid_str)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        slf.propagator.uuid = Some(uuid);
        Ok(slf)
    }

    /// Generate a new UUID, set it, and return self.
    pub fn with_new_uuid(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.propagator.uuid = Some(uuid::Uuid::new_v4());
        slf
    }

    /// Set the numeric ID and return self.
    pub fn with_id(mut slf: PyRefMut<'_, Self>, id: u64) -> PyRefMut<'_, Self> {
        slf.propagator.id = Some(id);
        slf
    }

    /// Get the current name.
    pub fn get_name(&self) -> Option<String> {
        self.propagator.name.clone()
    }

    /// Get the current UUID.
    pub fn get_uuid(&self) -> Option<String> {
        self.propagator.uuid.map(|u| u.to_string())
    }

    /// Get the current numeric ID.
    pub fn get_id(&self) -> Option<u64> {
        self.propagator.id
    }

    fn __repr__(&self) -> String {
        format!("NumericalOrbitPropagator(epoch={:?}, state_dim={})",
                DStatePropagator::current_epoch(&self.propagator),
                DStatePropagator::state_dim(&self.propagator))
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    // =========================================================================
    // Event detection methods
    // =========================================================================

    /// Add an event detector to this propagator.
    ///
    /// Args:
    ///     event (TimeEvent or ValueEvent or BinaryEvent or AltitudeEvent): Event detector
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     prop = bh.NumericalOrbitPropagator.from_eci(epoch, state)
    ///     event = bh.TimeEvent(epoch + 1800.0, "30 min mark")
    ///     prop.add_event_detector(event)
    ///     ```
    #[pyo3(text_signature = "(event)")]
    pub fn add_event_detector(&mut self, event: &Bound<'_, PyAny>) -> PyResult<()> {
        // Try each event type
        if let Ok(mut time_event) = event.extract::<PyRefMut<PyTimeEvent>>() {
            if let Some(inner) = time_event.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "TimeEvent has already been consumed"
            ));
        }

        if let Ok(mut value_event) = event.extract::<PyRefMut<PyValueEvent>>() {
            if let Some(inner) = value_event.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "ValueEvent has already been consumed"
            ));
        }

        if let Ok(mut binary_event) = event.extract::<PyRefMut<PyBinaryEvent>>() {
            if let Some(inner) = binary_event.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "BinaryEvent has already been consumed"
            ));
        }

        if let Ok(mut altitude_event) = event.extract::<PyRefMut<PyAltitudeEvent>>() {
            if let Some(inner) = altitude_event.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "AltitudeEvent has already been consumed"
            ));
        }

        Err(exceptions::PyTypeError::new_err(
            "Expected event detector type (TimeEvent, ValueEvent, BinaryEvent, or AltitudeEvent)"
        ))
    }

    /// Get the event log (list of detected events).
    ///
    /// Returns:
    ///     list[DetectedEvent]: List of events detected during propagation.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     prop.propagate_to(epoch + 3600.0)
    ///     events = prop.event_log()
    ///     for event in events:
    ///         print(f"Event '{event.name}' at {event.window_open}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn event_log(&self) -> Vec<PyDetectedEvent> {
        self.propagator
            .event_log()
            .iter()
            .map(|e| PyDetectedEvent { event: e.clone() })
            .collect()
    }

    /// Create an event query builder for filtering detected events.
    ///
    /// Returns an EventQuery that allows chainable filtering of detected events.
    /// Call `.collect()` on the query to get the final list of events.
    ///
    /// Returns:
    ///     EventQuery: Query builder for filtering events
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Get events from detector 0 within a time range
    ///     events = prop.query_events() \
    ///         .by_detector_index(0) \
    ///         .in_time_range(start, end) \
    ///         .collect()
    ///
    ///     # Count events by name pattern
    ///     count = prop.query_events() \
    ///         .by_name_contains("Altitude") \
    ///         .count()
    ///
    ///     # Combined filters
    ///     events = prop.query_events() \
    ///         .by_detector_index(1) \
    ///         .in_time_range(start, end) \
    ///         .collect()
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn query_events(&self) -> PyEventQuery {
        PyEventQuery::new(self.propagator.event_log().to_vec())
    }

    /// Get events by name.
    ///
    /// Args:
    ///     name (str): Event name to filter by.
    ///
    /// Returns:
    ///     list[DetectedEvent]: Events matching the given name.
    #[pyo3(text_signature = "(name)")]
    pub fn events_by_name(&self, name: &str) -> Vec<PyDetectedEvent> {
        self.propagator
            .events_by_name(name)
            .into_iter()
            .map(|e| PyDetectedEvent { event: e.clone() })
            .collect()
    }

    /// Get latest detected event, if any.
    ///
    /// Returns:
    ///     DetectedEvent or None: The most recently detected event.
    #[pyo3(text_signature = "()")]
    pub fn latest_event(&self) -> Option<PyDetectedEvent> {
        self.propagator
            .latest_event()
            .map(|e| PyDetectedEvent { event: e.clone() })
    }

    /// Get events in time range.
    ///
    /// Args:
    ///     start (Epoch): Start of time range.
    ///     end (Epoch): End of time range.
    ///
    /// Returns:
    ///     list[DetectedEvent]: Events within the given time range.
    #[pyo3(text_signature = "(start, end)")]
    pub fn events_in_range(&self, start: &PyEpoch, end: &PyEpoch) -> Vec<PyDetectedEvent> {
        self.propagator
            .events_in_range(start.obj, end.obj)
            .into_iter()
            .map(|e| PyDetectedEvent { event: e.clone() })
            .collect()
    }

    /// Get events by detector index.
    ///
    /// Args:
    ///     index (int): Detector index (0-based, in order of add_event_detector calls).
    ///
    /// Returns:
    ///     list[DetectedEvent]: Events from the specified detector.
    #[pyo3(text_signature = "(index)")]
    pub fn events_by_detector_index(&self, index: usize) -> Vec<PyDetectedEvent> {
        self.propagator
            .events_by_detector_index(index)
            .into_iter()
            .map(|e| PyDetectedEvent { event: e.clone() })
            .collect()
    }

    /// Get events by detector index within a time range.
    ///
    /// Args:
    ///     index (int): Detector index (0-based, in order of add_event_detector calls).
    ///     start (Epoch): Start of time range (inclusive).
    ///     end (Epoch): End of time range (inclusive).
    ///
    /// Returns:
    ///     list[DetectedEvent]: Events from the specified detector within the time range.
    #[pyo3(text_signature = "(index, start, end)")]
    pub fn events_by_detector_index_in_range(
        &self,
        index: usize,
        start: &PyEpoch,
        end: &PyEpoch,
    ) -> Vec<PyDetectedEvent> {
        self.propagator
            .events_by_detector_index_in_range(index, start.obj, end.obj)
            .into_iter()
            .map(|e| PyDetectedEvent { event: e.clone() })
            .collect()
    }

    /// Get events by name pattern within a time range.
    ///
    /// Args:
    ///     name_pattern (str): Substring to search for in event names.
    ///     start (Epoch): Start of time range (inclusive).
    ///     end (Epoch): End of time range (inclusive).
    ///
    /// Returns:
    ///     list[DetectedEvent]: Events matching the name pattern within the time range.
    #[pyo3(text_signature = "(name_pattern, start, end)")]
    pub fn events_by_name_in_range(
        &self,
        name_pattern: &str,
        start: &PyEpoch,
        end: &PyEpoch,
    ) -> Vec<PyDetectedEvent> {
        self.propagator
            .events_by_name_in_range(name_pattern, start.obj, end.obj)
            .into_iter()
            .map(|e| PyDetectedEvent { event: e.clone() })
            .collect()
    }

    /// Check if propagator is terminated due to a terminal event.
    ///
    /// Returns:
    ///     bool: True if propagation was stopped by a terminal event.
    #[pyo3(text_signature = "()")]
    pub fn terminated(&self) -> bool {
        self.propagator.terminated()
    }

    /// Reset the termination flag to allow continued propagation.
    ///
    /// After calling this, propagation can continue even if it was previously
    /// stopped by a terminal event.
    #[pyo3(text_signature = "()")]
    pub fn reset_termination(&mut self) {
        self.propagator.reset_termination();
    }

    /// Clear all detected events from the event log.
    #[pyo3(text_signature = "()")]
    pub fn clear_events(&mut self) {
        self.propagator.clear_events();
    }

    // =========================================================================
    // Additional bindings for test parity
    // =========================================================================

    /// Get current parameter vector.
    ///
    /// Returns:
    ///     numpy.ndarray or None: Current parameter vector, or None if no params.
    #[pyo3(text_signature = "()")]
    pub fn current_params<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray<f64, Ix1>> {
        // DNumericalOrbitPropagator has current_params() as an inherent method
        let params = self.propagator.current_params();
        params.as_slice().to_pyarray(py).to_owned()
    }

    /// Set trajectory storage mode.
    ///
    /// Args:
    ///     mode (TrajectoryMode): The new trajectory mode.
    #[pyo3(text_signature = "(mode)")]
    pub fn set_trajectory_mode(&mut self, mode: &PyTrajectoryMode) {
        // DNumericalOrbitPropagator has set_trajectory_mode() as an inherent method
        self.propagator.set_trajectory_mode(mode.mode);
    }

    /// Get current trajectory storage mode.
    ///
    /// Returns:
    ///     TrajectoryMode: Current trajectory mode.
    #[getter]
    pub fn trajectory_mode(&self) -> PyTrajectoryMode {
        // DNumericalOrbitPropagator has trajectory_mode() as an inherent method
        PyTrajectoryMode { mode: self.propagator.trajectory_mode() }
    }

    /// Get STM (State Transition Matrix) at a specific epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for STM query.
    ///
    /// Returns:
    ///     numpy.ndarray or None: The STM at the requested epoch, or None if STM not enabled.
    #[pyo3(text_signature = "(epoch)")]
    pub fn stm_at<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> Option<Bound<'a, PyArray<f64, Ix2>>> {
        self.propagator.stm_at(epoch.obj).map(|stm| {
            let n = stm.nrows();
            let mut flat = Vec::with_capacity(n * n);
            for i in 0..n {
                for j in 0..n {
                    flat.push(stm[(i, j)]);
                }
            }
            flat.into_pyarray(py).reshape([n, n]).unwrap()
        })
    }

    /// Get sensitivity matrix at a specific epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for sensitivity query.
    ///
    /// Returns:
    ///     numpy.ndarray or None: The sensitivity matrix at the requested epoch, or None if not enabled.
    #[pyo3(text_signature = "(epoch)")]
    pub fn sensitivity_at<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> Option<Bound<'a, PyArray<f64, Ix2>>> {
        self.propagator.sensitivity_at(epoch.obj).map(|sens| {
            let n = sens.nrows();
            let p = sens.ncols();
            let mut flat = Vec::with_capacity(n * p);
            for i in 0..n {
                for j in 0..p {
                    flat.push(sens[(i, j)]);
                }
            }
            flat.into_pyarray(py).reshape([n, p]).unwrap()
        })
    }

    /// Get covariance at a specific epoch in GCRF frame.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch.
    ///
    /// Returns:
    ///     numpy.ndarray: Covariance matrix in GCRF frame.
    #[pyo3(text_signature = "(epoch)")]
    pub fn covariance_gcrf<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix2>>> {
        let cov = DOrbitCovarianceProvider::covariance_gcrf(&self.propagator, epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let n = cov.nrows();
        let mut flat = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                flat.push(cov[(i, j)]);
            }
        }
        Ok(flat.into_pyarray(py).reshape([n, n]).unwrap())
    }

    /// Get covariance at a specific epoch in RTN (Radial-Tangential-Normal) frame.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch.
    ///
    /// Returns:
    ///     numpy.ndarray: Covariance matrix in RTN frame.
    #[pyo3(text_signature = "(epoch)")]
    pub fn covariance_rtn<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix2>>> {
        let cov = DOrbitCovarianceProvider::covariance_rtn(&self.propagator, epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let n = cov.nrows();
        let mut flat = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                flat.push(cov[(i, j)]);
            }
        }
        Ok(flat.into_pyarray(py).reshape([n, n]).unwrap())
    }

    /// Set interpolation method for state trajectory queries.
    ///
    /// Args:
    ///     method (InterpolationMethod): Interpolation method to use.
    #[pyo3(text_signature = "(method)")]
    pub fn set_interpolation_method(&mut self, method: &PyInterpolationMethod) {
        InterpolationConfig::set_interpolation_method(&mut self.propagator, method.method);
    }

    /// Get current interpolation method for state trajectory queries.
    ///
    /// Returns:
    ///     InterpolationMethod: Current interpolation method.
    #[pyo3(text_signature = "()")]
    pub fn get_interpolation_method(&self) -> PyInterpolationMethod {
        PyInterpolationMethod { method: InterpolationConfig::get_interpolation_method(&self.propagator) }
    }

    /// Set interpolation method for covariance queries.
    ///
    /// Args:
    ///     method (CovarianceInterpolationMethod): Interpolation method for covariance.
    #[pyo3(text_signature = "(method)")]
    pub fn set_covariance_interpolation_method(&mut self, method: &PyCovarianceInterpolationMethod) {
        CovarianceInterpolationConfig::set_covariance_interpolation_method(&mut self.propagator, method.method);
    }

    /// Get current interpolation method for covariance queries.
    ///
    /// Returns:
    ///     CovarianceInterpolationMethod: Current covariance interpolation method.
    #[pyo3(text_signature = "()")]
    pub fn get_covariance_interpolation_method(&self) -> PyCovarianceInterpolationMethod {
        PyCovarianceInterpolationMethod { method: CovarianceInterpolationConfig::get_covariance_interpolation_method(&self.propagator) }
    }

    /// Set propagator name (mutating).
    ///
    /// Args:
    ///     name (str or None): New name for the propagator.
    #[pyo3(text_signature = "(name)")]
    pub fn set_name(&mut self, name: Option<String>) {
        self.propagator.name = name;
    }

    /// Set propagator numeric ID (mutating).
    ///
    /// Args:
    ///     id (int or None): New numeric ID for the propagator.
    #[pyo3(text_signature = "(id)")]
    pub fn set_id(&mut self, id: Option<u64>) {
        self.propagator.id = id;
    }

    /// Set propagator UUID (mutating).
    ///
    /// Args:
    ///     uuid_str (str or None): New UUID string for the propagator.
    ///
    /// Raises:
    ///     ValueError: If the UUID string is invalid.
    #[pyo3(text_signature = "(uuid_str)")]
    pub fn set_uuid(&mut self, uuid_str: Option<String>) -> PyResult<()> {
        if let Some(s) = uuid_str {
            let uuid = uuid::Uuid::parse_str(&s)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
            self.propagator.uuid = Some(uuid);
        } else {
            self.propagator.uuid = None;
        }
        Ok(())
    }

    /// Set all identity fields and return self (builder pattern).
    ///
    /// Args:
    ///     name (str or None): New name.
    ///     uuid_str (str or None): New UUID string.
    ///     id (int or None): New numeric ID.
    ///
    /// Returns:
    ///     NumericalOrbitPropagator: Self for method chaining.
    ///
    /// Raises:
    ///     ValueError: If the UUID string is invalid.
    pub fn with_identity(
        mut slf: PyRefMut<'_, Self>,
        name: Option<String>,
        uuid_str: Option<String>,
        id: Option<u64>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        slf.propagator.name = name;
        slf.propagator.id = id;
        if let Some(s) = uuid_str {
            let uuid = uuid::Uuid::parse_str(&s)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
            slf.propagator.uuid = Some(uuid);
        } else {
            slf.propagator.uuid = None;
        }
        Ok(slf)
    }

    /// Set all identity fields at once (mutating).
    ///
    /// Args:
    ///     name (str or None): New name.
    ///     uuid_str (str or None): New UUID string.
    ///     id (int or None): New numeric ID.
    ///
    /// Raises:
    ///     ValueError: If the UUID string is invalid.
    #[pyo3(text_signature = "(name, uuid_str, id)")]
    pub fn set_identity(
        &mut self,
        name: Option<String>,
        uuid_str: Option<String>,
        id: Option<u64>,
    ) -> PyResult<()> {
        self.propagator.name = name;
        self.propagator.id = id;
        if let Some(s) = uuid_str {
            let uuid = uuid::Uuid::parse_str(&s)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
            self.propagator.uuid = Some(uuid);
        } else {
            self.propagator.uuid = None;
        }
        Ok(())
    }
}

// =============================================================================
// TrajectoryMode
// =============================================================================

/// Trajectory storage mode for numerical propagators.
///
/// Controls when and whether state data is stored in the trajectory during propagation.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Create propagator
///     prop = bh.NumericalPropagator(epoch, state, dynamics, config)
///
///     # Use disabled mode to save memory
///     prop.set_trajectory_mode(bh.TrajectoryMode.DISABLED)
///
///     # Or store all integration steps for analysis
///     prop.set_trajectory_mode(bh.TrajectoryMode.ALL_STEPS)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "TrajectoryMode")]
#[derive(Clone)]
pub struct PyTrajectoryMode {
    pub(crate) mode: propagators::TrajectoryMode,
}

#[pymethods]
impl PyTrajectoryMode {
    /// Store state at every integration step.
    ///
    /// Useful for debugging or high-resolution trajectory analysis.
    /// May use significantly more memory for long propagations.
    ///
    /// Returns:
    ///     TrajectoryMode: AllSteps trajectory mode constant
    #[classattr]
    #[pyo3(name = "ALL_STEPS")]
    fn all_steps() -> Self {
        PyTrajectoryMode {
            mode: propagators::TrajectoryMode::AllSteps,
        }
    }

    /// Store state at requested output epochs only (default).
    ///
    /// Most memory-efficient. Only stores at times explicitly requested
    /// via `propagate_to_epochs()` or similar methods.
    ///
    /// Returns:
    ///     TrajectoryMode: OutputStepsOnly trajectory mode constant
    #[classattr]
    #[pyo3(name = "OUTPUT_STEPS_ONLY")]
    fn output_steps_only() -> Self {
        PyTrajectoryMode {
            mode: propagators::TrajectoryMode::OutputStepsOnly,
        }
    }

    /// Disable trajectory storage entirely.
    ///
    /// Only the current state is maintained. Useful when only the
    /// final state matters and memory is constrained.
    ///
    /// Returns:
    ///     TrajectoryMode: Disabled trajectory mode constant
    #[classattr]
    #[pyo3(name = "DISABLED")]
    fn disabled() -> Self {
        PyTrajectoryMode {
            mode: propagators::TrajectoryMode::Disabled,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.mode)
    }

    fn __repr__(&self) -> String {
        format!("TrajectoryMode.{:?}", self.mode)
    }

    fn __richcmp__(&self, other: &Self, op: pyo3::pyclass::CompareOp) -> PyResult<bool> {
        match op {
            pyo3::pyclass::CompareOp::Eq => Ok(self.mode == other.mode),
            pyo3::pyclass::CompareOp::Ne => Ok(self.mode != other.mode),
            _ => Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Comparison not supported",
            )),
        }
    }
}

// =============================================================================
// NumericalPropagator (generic N-D with Python callable dynamics)
// =============================================================================

/// Generic numerical propagator for arbitrary N-dimensional dynamical systems.
///
/// This propagator accepts a user-defined Python dynamics function and can be
/// applied to any system of ODEs: attitude dynamics, chemical kinetics, population
/// models, control systems, etc.
///
/// Args:
///     epoch (Epoch): Initial epoch.
///     state (numpy.ndarray): Initial state vector (N-dimensional).
///     dynamics (callable): Dynamics function: f(t, state, params) -> derivative.
///         Should accept (float, np.ndarray, Optional[np.ndarray]) and return np.ndarray.
///     propagation_config (NumericalPropagationConfig): Propagation configuration.
///     params (numpy.ndarray or None): Optional parameter vector for the dynamics function.
///     initial_covariance (numpy.ndarray or None): Optional initial covariance matrix (enables STM).
///
/// Attributes:
///     current_epoch (Epoch): Current propagation time
///     initial_epoch (Epoch): Initial epoch from propagator creation
///     state_dim (int): Dimension of state vector
///     step_size (float): Current integration step size in seconds
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Define dynamics: simple harmonic oscillator
///     # dx/dt = v, dv/dt = -ω²x
///     omega = 1.0
///     def sho_dynamics(t, state, params):
///         return np.array([state[1], -omega**2 * state[0]])
///
///     # Create initial state
///     epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///     state = np.array([1.0, 0.0])  # [position, velocity]
///
///     # Create propagator
///     prop = bh.NumericalPropagator(
///         epoch, state, sho_dynamics,
///         bh.NumericalPropagationConfig.default()
///     )
///
///     # Propagate one period
///     prop.propagate_to(epoch + 2.0 * np.pi)
///     print(f"Final state: {prop.current_state()}")  # Should be ~[1, 0]
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "NumericalPropagator")]
pub struct PyNumericalPropagator {
    pub propagator: propagators::DNumericalPropagator,
}

#[pymethods]
impl PyNumericalPropagator {
    /// Create a new generic numerical propagator.
    ///
    /// Args:
    ///     epoch (Epoch): Initial epoch.
    ///     state (numpy.ndarray): Initial state vector (N-dimensional).
    ///     dynamics (callable): Dynamics function: f(t, state, params) -> derivative.
    ///                         Should accept (float, np.ndarray, Optional[np.ndarray]) and return np.ndarray.
    ///     propagation_config (NumericalPropagationConfig): Propagation configuration.
    ///     params (numpy.ndarray or None): Optional parameter vector for the dynamics function.
    ///     initial_covariance (numpy.ndarray or None): Optional initial covariance matrix (enables STM).
    ///
    /// Returns:
    ///     NumericalPropagator: New propagator instance.
    #[new]
    #[pyo3(signature = (epoch, state, dynamics, propagation_config, params=None, initial_covariance=None))]
    pub fn new(
        py: Python<'_>,
        epoch: &PyEpoch,
        state: PyReadonlyArray1<f64>,
        dynamics: Py<PyAny>,
        propagation_config: &PyNumericalPropagationConfig,
        params: Option<PyReadonlyArray1<f64>>,
        initial_covariance: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<Self> {
        let state_vec = nalgebra::DVector::from_column_slice(state.as_slice()?);
        let state_dim = state_vec.len();

        let params_vec = params.map(|p| nalgebra::DVector::from_column_slice(p.as_slice().unwrap()));

        let cov_matrix = if let Some(cov) = initial_covariance {
            let cov_shape = cov.shape();
            if cov_shape[0] != state_dim || cov_shape[1] != state_dim {
                return Err(exceptions::PyValueError::new_err(
                    format!("Initial covariance must be a {}x{} matrix", state_dim, state_dim)
                ));
            }
            let cov_data: Vec<f64> = cov.as_slice()?.to_vec();
            Some(nalgebra::DMatrix::from_row_slice(state_dim, state_dim, &cov_data))
        } else {
            None
        };

        // Create a wrapper that calls the Python dynamics function
        let dynamics_py = dynamics.clone_ref(py);
        let dynamics_fn: crate::integrators::traits::DStateDynamics = Box::new(
            move |t: f64, x: &nalgebra::DVector<f64>, p: Option<&nalgebra::DVector<f64>>| {
                Python::attach(|py| {
                    // Convert state to numpy array
                    let x_np = x.as_slice().to_pyarray(py);

                    // Convert params to numpy array or None
                    let p_np: Option<Bound<'_, PyArray<f64, Ix1>>> = p.map(|pv| pv.as_slice().to_pyarray(py).to_owned());

                    // Call Python function
                    let result = match p_np {
                        Some(params_arr) => dynamics_py.call1(py, (t, x_np, params_arr)),
                        None => dynamics_py.call1(py, (t, x_np, py.None())),
                    };

                    match result {
                        Ok(res) => {
                            // Extract result as numpy array
                            let res_arr: PyReadonlyArray1<f64> = res.extract(py).unwrap();
                            nalgebra::DVector::from_column_slice(res_arr.as_slice().unwrap())
                        }
                        Err(e) => {
                            eprintln!("Error calling dynamics function: {}", e);
                            nalgebra::DVector::zeros(x.len())
                        }
                    }
                })
            }
        );

        let prop = propagators::DNumericalPropagator::new(
            epoch.obj,
            state_vec,
            dynamics_fn,
            propagation_config.config.clone(),
            params_vec,
            None, // control_input
            cov_matrix,
        ).map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyNumericalPropagator { propagator: prop })
    }

    // =========================================================================
    // DStatePropagator trait methods
    // =========================================================================

    /// Get current epoch.
    #[getter]
    pub fn current_epoch(&self) -> PyEpoch {
        PyEpoch { obj: DStatePropagator::current_epoch(&self.propagator) }
    }

    /// Get initial epoch.
    #[getter]
    pub fn initial_epoch(&self) -> PyEpoch {
        PyEpoch { obj: DStatePropagator::initial_epoch(&self.propagator) }
    }

    /// Get current state vector.
    #[pyo3(text_signature = "()")]
    pub fn current_state<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray<f64, Ix1>> {
        let state = DStatePropagator::current_state(&self.propagator);
        state.as_slice().to_pyarray(py).to_owned()
    }

    /// Get initial state vector.
    #[pyo3(text_signature = "()")]
    pub fn initial_state<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray<f64, Ix1>> {
        let state = DStatePropagator::initial_state(&self.propagator);
        state.as_slice().to_pyarray(py).to_owned()
    }

    /// Get state dimension.
    #[getter]
    pub fn state_dim(&self) -> usize {
        DStatePropagator::state_dim(&self.propagator)
    }

    /// Get current step size.
    #[getter]
    pub fn step_size(&self) -> f64 {
        DStatePropagator::step_size(&self.propagator)
    }

    /// Set step size.
    #[setter]
    pub fn set_step_size(&mut self, step_size: f64) {
        DStatePropagator::set_step_size(&mut self.propagator, step_size);
    }

    /// Step forward by the default step size.
    #[pyo3(text_signature = "()")]
    pub fn step(&mut self) {
        DStatePropagator::step(&mut self.propagator);
    }

    /// Step forward by a specified time duration.
    #[pyo3(text_signature = "(step_size)")]
    pub fn step_by(&mut self, step_size: f64) {
        DStatePropagator::step_by(&mut self.propagator, step_size);
    }

    /// Step past a specified target epoch.
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn step_past(&mut self, target_epoch: &PyEpoch) {
        DStatePropagator::step_past(&mut self.propagator, target_epoch.obj);
    }

    /// Propagate forward by specified number of steps.
    #[pyo3(text_signature = "(num_steps)")]
    pub fn propagate_steps(&mut self, num_steps: usize) {
        DStatePropagator::propagate_steps(&mut self.propagator, num_steps);
    }

    /// Propagate to a specific target epoch.
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn propagate_to(&mut self, target_epoch: &PyEpoch) {
        DStatePropagator::propagate_to(&mut self.propagator, target_epoch.obj);
    }

    /// Reset propagator to initial conditions.
    #[pyo3(text_signature = "()")]
    pub fn reset(&mut self) {
        DStatePropagator::reset(&mut self.propagator);
    }

    // =========================================================================
    // State provider methods
    // =========================================================================

    /// Compute state at a specific epoch.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = DStateProvider::state(&self.propagator, epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    // =========================================================================
    // Trajectory and variational methods
    // =========================================================================

    /// Get accumulated trajectory.
    #[getter]
    pub fn trajectory(&self) -> PyTrajectory {
        PyTrajectory { trajectory: self.propagator.trajectory().clone() }
    }

    /// Get current STM if enabled.
    #[pyo3(text_signature = "()")]
    pub fn stm<'a>(&self, py: Python<'a>) -> Option<Bound<'a, PyArray<f64, Ix2>>> {
        self.propagator.stm().map(|stm| {
            let n = stm.nrows();
            let flat: Vec<f64> = (0..n).flat_map(|i| (0..n).map(move |j| stm[(i, j)])).collect();
            flat.into_pyarray(py).reshape([n, n]).unwrap()
        })
    }

    /// Get current sensitivity matrix if enabled.
    #[pyo3(text_signature = "()")]
    pub fn sensitivity<'a>(&self, py: Python<'a>) -> Option<Bound<'a, PyArray<f64, Ix2>>> {
        self.propagator.sensitivity().map(|sens| {
            let n = sens.nrows();
            let p = sens.ncols();
            let flat: Vec<f64> = (0..n).flat_map(|i| (0..p).map(move |j| sens[(i, j)])).collect();
            flat.into_pyarray(py).reshape([n, p]).unwrap()
        })
    }

    /// Get covariance at a specific epoch.
    #[pyo3(text_signature = "(epoch)")]
    pub fn covariance<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix2>>> {
        let cov = DCovarianceProvider::covariance(&self.propagator, epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let n = cov.nrows();
        let mut flat = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                flat.push(cov[(i, j)]);
            }
        }
        Ok(flat.into_pyarray(py).reshape([n, n]).unwrap())
    }

    // =========================================================================
    // Eviction policy
    // =========================================================================

    /// Set trajectory eviction policy based on maximum size.
    #[pyo3(text_signature = "(max_size)")]
    pub fn set_eviction_policy_max_size(&mut self, max_size: usize) -> PyResult<()> {
        DStatePropagator::set_eviction_policy_max_size(&mut self.propagator, max_size)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Set trajectory eviction policy based on maximum age.
    #[pyo3(text_signature = "(max_age)")]
    pub fn set_eviction_policy_max_age(&mut self, max_age: f64) -> PyResult<()> {
        DStatePropagator::set_eviction_policy_max_age(&mut self.propagator, max_age)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Set the trajectory storage mode.
    ///
    /// Args:
    ///     mode (TrajectoryMode): The trajectory storage mode.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     prop = bh.NumericalPropagator(epoch, state, dynamics, config)
    ///     prop.set_trajectory_mode(bh.TrajectoryMode.DISABLED)
    ///     ```
    #[pyo3(text_signature = "(mode)")]
    pub fn set_trajectory_mode(&mut self, mode: &PyTrajectoryMode) {
        self.propagator.set_trajectory_mode(mode.mode);
    }

    /// Get the current trajectory storage mode.
    ///
    /// Returns:
    ///     TrajectoryMode: The current trajectory storage mode.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     prop = bh.NumericalPropagator(epoch, state, dynamics, config)
    ///     mode = prop.trajectory_mode()
    ///     print(f"Mode: {mode}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn trajectory_mode(&self) -> PyTrajectoryMode {
        PyTrajectoryMode {
            mode: self.propagator.trajectory_mode(),
        }
    }

    // =========================================================================
    // Identity methods
    // =========================================================================

    /// Set the name and return self.
    pub fn with_name(mut slf: PyRefMut<'_, Self>, name: String) -> PyRefMut<'_, Self> {
        slf.propagator.name = Some(name);
        slf
    }

    /// Set the UUID and return self.
    pub fn with_uuid(mut slf: PyRefMut<'_, Self>, uuid_str: String) -> PyResult<PyRefMut<'_, Self>> {
        let uuid = uuid::Uuid::parse_str(&uuid_str)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        slf.propagator.uuid = Some(uuid);
        Ok(slf)
    }

    /// Generate a new UUID, set it, and return self.
    pub fn with_new_uuid(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.propagator.uuid = Some(uuid::Uuid::new_v4());
        slf
    }

    /// Set the numeric ID and return self.
    pub fn with_id(mut slf: PyRefMut<'_, Self>, id: u64) -> PyRefMut<'_, Self> {
        slf.propagator.id = Some(id);
        slf
    }

    /// Get the current name.
    pub fn get_name(&self) -> Option<String> {
        self.propagator.name.clone()
    }

    /// Get the current UUID.
    pub fn get_uuid(&self) -> Option<String> {
        self.propagator.uuid.map(|u| u.to_string())
    }

    /// Get the current numeric ID.
    pub fn get_id(&self) -> Option<u64> {
        self.propagator.id
    }

    fn __repr__(&self) -> String {
        format!("NumericalPropagator(epoch={:?}, state_dim={})",
                DStatePropagator::current_epoch(&self.propagator),
                DStatePropagator::state_dim(&self.propagator))
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    // =========================================================================
    // Event detection methods
    // =========================================================================

    /// Add an event detector to this propagator.
    ///
    /// Args:
    ///     event (TimeEvent or ValueEvent or BinaryEvent): Event detector
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     prop = bh.NumericalPropagator(epoch, state, dynamics, config)
    ///     event = bh.TimeEvent(epoch + 5.0, "5 second mark")
    ///     prop.add_event_detector(event)
    ///     ```
    #[pyo3(text_signature = "(event)")]
    pub fn add_event_detector(&mut self, event: &Bound<'_, PyAny>) -> PyResult<()> {
        // Try each event type
        if let Ok(mut time_event) = event.extract::<PyRefMut<PyTimeEvent>>() {
            if let Some(inner) = time_event.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "TimeEvent has already been consumed"
            ));
        }

        if let Ok(mut value_event) = event.extract::<PyRefMut<PyValueEvent>>() {
            if let Some(inner) = value_event.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "ValueEvent has already been consumed"
            ));
        }

        if let Ok(mut binary_event) = event.extract::<PyRefMut<PyBinaryEvent>>() {
            if let Some(inner) = binary_event.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "BinaryEvent has already been consumed"
            ));
        }

        // Note: AltitudeEvent is orbit-specific and not available for generic NumericalPropagator

        Err(exceptions::PyTypeError::new_err(
            "Expected event detector type (TimeEvent, ValueEvent, or BinaryEvent)"
        ))
    }

    /// Get the event log (list of detected events).
    ///
    /// Returns:
    ///     list[DetectedEvent]: List of events detected during propagation.
    #[pyo3(text_signature = "()")]
    pub fn event_log(&self) -> Vec<PyDetectedEvent> {
        self.propagator
            .event_log()
            .iter()
            .map(|e| PyDetectedEvent { event: e.clone() })
            .collect()
    }

    /// Create an event query builder for filtering detected events.
    ///
    /// Returns an EventQuery that allows chainable filtering of detected events.
    /// Call `.collect()` on the query to get the final list of events.
    ///
    /// Returns:
    ///     EventQuery: Query builder for filtering events
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Get events from detector 0 within a time range
    ///     events = prop.query_events() \
    ///         .by_detector_index(0) \
    ///         .in_time_range(start, end) \
    ///         .collect()
    ///
    ///     # Count events by name pattern
    ///     count = prop.query_events() \
    ///         .by_name_contains("Altitude") \
    ///         .count()
    ///
    ///     # Combined filters
    ///     events = prop.query_events() \
    ///         .by_detector_index(1) \
    ///         .in_time_range(start, end) \
    ///         .collect()
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn query_events(&self) -> PyEventQuery {
        PyEventQuery::new(self.propagator.event_log().to_vec())
    }

    /// Get events by name.
    ///
    /// Args:
    ///     name (str): Event name to filter by.
    ///
    /// Returns:
    ///     list[DetectedEvent]: Events matching the given name.
    #[pyo3(text_signature = "(name)")]
    pub fn events_by_name(&self, name: &str) -> Vec<PyDetectedEvent> {
        self.propagator
            .events_by_name(name)
            .into_iter()
            .map(|e| PyDetectedEvent { event: e.clone() })
            .collect()
    }

    /// Get latest detected event, if any.
    ///
    /// Returns:
    ///     DetectedEvent or None: The most recently detected event.
    #[pyo3(text_signature = "()")]
    pub fn latest_event(&self) -> Option<PyDetectedEvent> {
        self.propagator
            .latest_event()
            .map(|e| PyDetectedEvent { event: e.clone() })
    }

    /// Get events in time range.
    ///
    /// Args:
    ///     start (Epoch): Start of time range.
    ///     end (Epoch): End of time range.
    ///
    /// Returns:
    ///     list[DetectedEvent]: Events within the given time range.
    #[pyo3(text_signature = "(start, end)")]
    pub fn events_in_range(&self, start: &PyEpoch, end: &PyEpoch) -> Vec<PyDetectedEvent> {
        self.propagator
            .events_in_range(start.obj, end.obj)
            .into_iter()
            .map(|e| PyDetectedEvent { event: e.clone() })
            .collect()
    }

    /// Get events by detector index.
    ///
    /// Args:
    ///     index (int): Detector index (0-based, in order of add_event_detector calls).
    ///
    /// Returns:
    ///     list[DetectedEvent]: Events from the specified detector.
    #[pyo3(text_signature = "(index)")]
    pub fn events_by_detector_index(&self, index: usize) -> Vec<PyDetectedEvent> {
        self.propagator
            .events_by_detector_index(index)
            .into_iter()
            .map(|e| PyDetectedEvent { event: e.clone() })
            .collect()
    }

    /// Check if propagator is terminated due to a terminal event.
    ///
    /// Returns:
    ///     bool: True if propagation was stopped by a terminal event.
    #[pyo3(text_signature = "()")]
    pub fn terminated(&self) -> bool {
        self.propagator.terminated()
    }

    /// Reset the termination flag to allow continued propagation.
    #[pyo3(text_signature = "()")]
    pub fn reset_termination(&mut self) {
        self.propagator.reset_termination();
    }

    /// Clear all detected events from the event log.
    #[pyo3(text_signature = "()")]
    pub fn clear_events(&mut self) {
        self.propagator.clear_events();
    }

    // =========================================================================
    // Additional Identity methods (matching SGPPropagator/KeplerianPropagator)
    // =========================================================================

    /// Set the name in-place (mutating).
    ///
    /// Args:
    ///     name (str or None): Name to assign, or None to clear.
    pub fn set_name(&mut self, name: Option<String>) {
        self.propagator.name = name;
    }

    /// Set the numeric ID in-place (mutating).
    ///
    /// Args:
    ///     id (int or None): Numeric ID to assign, or None to clear.
    pub fn set_id(&mut self, id: Option<u64>) {
        self.propagator.id = id;
    }

    /// Set the UUID in-place (mutating).
    ///
    /// Args:
    ///     uuid_str (str or None): UUID string to assign, or None to clear.
    pub fn set_uuid(&mut self, uuid_str: Option<String>) -> PyResult<()> {
        let uuid = match uuid_str {
            Some(s) => Some(uuid::Uuid::parse_str(&s)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?),
            None => None,
        };
        self.propagator.uuid = uuid;
        Ok(())
    }

    /// Generate a new UUID and set it in-place (mutating).
    pub fn generate_uuid(&mut self) {
        self.propagator.uuid = Some(uuid::Uuid::new_v4());
    }

    /// Set all identity fields at once and return self (consuming constructor pattern).
    ///
    /// Args:
    ///     name (str or None): Optional name to assign.
    ///     uuid_str (str or None): Optional UUID string to assign.
    ///     id (int or None): Optional numeric ID to assign.
    ///
    /// Returns:
    ///     NumericalPropagator: Self with identity set.
    pub fn with_identity(mut slf: PyRefMut<'_, Self>, name: Option<String>, uuid_str: Option<String>, id: Option<u64>) -> PyResult<PyRefMut<'_, Self>> {
        let uuid = match uuid_str {
            Some(s) => Some(uuid::Uuid::parse_str(&s)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?),
            None => None,
        };
        slf.propagator.name = name;
        slf.propagator.uuid = uuid;
        slf.propagator.id = id;
        Ok(slf)
    }

    /// Set all identity fields in-place (mutating).
    ///
    /// Args:
    ///     name (str or None): Optional name to assign.
    ///     uuid_str (str or None): Optional UUID string to assign.
    ///     id (int or None): Optional numeric ID to assign.
    pub fn set_identity(&mut self, name: Option<String>, uuid_str: Option<String>, id: Option<u64>) -> PyResult<()> {
        let uuid = match uuid_str {
            Some(s) => Some(uuid::Uuid::parse_str(&s)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?),
            None => None,
        };
        self.propagator.name = name;
        self.propagator.uuid = uuid;
        self.propagator.id = id;
        Ok(())
    }

    // =========================================================================
    // Interpolation configuration methods
    // =========================================================================

    /// Set the interpolation method using builder pattern.
    /// Note: Returns None as Python doesn't support returning mutable self with borrowed args.
    /// Use method chaining via separate calls or use set_interpolation_method instead.
    ///
    /// Args:
    ///     method (InterpolationMethod): The interpolation method to use.
    pub fn with_interpolation_method(&mut self, method: &PyInterpolationMethod) {
        self.propagator.set_interpolation_method(method.method);
    }

    /// Set the interpolation method in-place.
    ///
    /// Args:
    ///     method (InterpolationMethod): The interpolation method to use.
    pub fn set_interpolation_method(&mut self, method: &PyInterpolationMethod) {
        self.propagator.set_interpolation_method(method.method);
    }

    /// Get the current interpolation method.
    ///
    /// Returns:
    ///     InterpolationMethod: The current interpolation method.
    pub fn get_interpolation_method(&self) -> PyInterpolationMethod {
        PyInterpolationMethod { method: self.propagator.get_interpolation_method() }
    }

    /// Set the covariance interpolation method using builder pattern.
    /// Note: Returns None as Python doesn't support returning mutable self with borrowed args.
    /// Use method chaining via separate calls or use set_covariance_interpolation_method instead.
    ///
    /// Args:
    ///     method (CovarianceInterpolationMethod): The covariance interpolation method to use.
    pub fn with_covariance_interpolation_method(&mut self, method: &PyCovarianceInterpolationMethod) {
        self.propagator.set_covariance_interpolation_method(method.method);
    }

    /// Set the covariance interpolation method in-place.
    ///
    /// Args:
    ///     method (CovarianceInterpolationMethod): The covariance interpolation method to use.
    pub fn set_covariance_interpolation_method(&mut self, method: &PyCovarianceInterpolationMethod) {
        self.propagator.set_covariance_interpolation_method(method.method);
    }

    /// Get the current covariance interpolation method.
    ///
    /// Returns:
    ///     CovarianceInterpolationMethod: The current covariance interpolation method.
    pub fn get_covariance_interpolation_method(&self) -> PyCovarianceInterpolationMethod {
        PyCovarianceInterpolationMethod { method: self.propagator.get_covariance_interpolation_method() }
    }
}
