// Import traits needed by propagator methods
use crate::propagators::traits::{StateProvider, OrbitPropagator};

/// Python wrapper for SGPPropagator (replaces TLE)
/// SGP4/SDP4 satellite propagator using TLE data.
///
/// The SGP (Simplified General Perturbations) propagator implements the SGP4/SDP4 models
/// for propagating satellites using Two-Line Element (TLE) orbital data. This is the standard
/// model used for tracking objects in Earth orbit.
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
        let state = self.propagator.state(epoch.obj);
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
        let state = self.propagator.state_eci(epoch.obj);
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
        let state = self.propagator.state_ecef(epoch.obj);
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
    pub fn states<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> Vec<Bound<'a, PyArray<f64, Ix1>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = self.propagator.states(&epoch_vec);
        states.iter().map(|s| s.as_slice().to_pyarray(py).to_owned()).collect()
    }

    /// Compute states at multiple epochs in ECI coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of ECI state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_eci<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> Vec<Bound<'a, PyArray<f64, Ix1>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = self.propagator.states_eci(&epoch_vec);
        states.iter().map(|s| s.as_slice().to_pyarray(py).to_owned()).collect()
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
        PyOrbitalTrajectory { trajectory: self.propagator.trajectory.clone() }
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
        use crate::utils::Identifiable;
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
        use crate::utils::Identifiable;
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
        use crate::utils::Identifiable;
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
        use crate::utils::Identifiable;
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
        use crate::utils::Identifiable;
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
        use crate::utils::Identifiable;
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
        use crate::utils::Identifiable;
        self.propagator.set_id(id);
    }

    /// Set the name in-place (mutating).
    ///
    /// Args:
    ///     name (str or None): Name to assign, or None to clear.
    pub fn set_name(&mut self, name: Option<String>) {
        use crate::utils::Identifiable;
        self.propagator.set_name(name.as_deref());
    }

    /// Generate a new UUID and set it in-place (mutating).
    pub fn generate_uuid(&mut self) {
        use crate::utils::Identifiable;
        self.propagator.generate_uuid();
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
        use crate::utils::Identifiable;
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
        use crate::utils::Identifiable;
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
        use crate::utils::Identifiable;
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
    ///     elements_deg = prop.state_as_osculating_elements(epoch, bh.AngleFormat.DEGREES)
    ///     print(f"Semi-major axis: {elements_deg[0]/1000:.3f} km")
    ///     print(f"Inclination: {elements_deg[2]:.4f} degrees")
    ///
    ///     # Get elements in radians
    ///     elements_rad = prop.state_as_osculating_elements(epoch, bh.AngleFormat.RADIANS)
    ///     print(f"Inclination: {elements_rad[2]:.4f} radians")
    ///     ```
    #[pyo3(text_signature = "(epoch, angle_format)")]
    pub fn state_as_osculating_elements<'a>(
        &self,
        py: Python<'a>,
        epoch: PyRef<PyEpoch>,
        angle_format: &PyAngleFormat,
    ) -> Bound<'a, PyArray<f64, Ix1>> {
        let state = self.propagator.state_as_osculating_elements(epoch.obj, angle_format.value);
        state.as_slice().to_pyarray(py).to_owned()
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
    ///     elements_list = prop.states_as_osculating_elements(epochs, bh.AngleFormat.DEGREES)
    ///
    ///     for i, elements in enumerate(elements_list):
    ///         print(f"Hour {i}: a={elements[0]/1000:.3f} km, e={elements[1]:.6f}")
    ///     ```
    #[pyo3(text_signature = "(epochs, angle_format)")]
    pub fn states_as_osculating_elements<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
        angle_format: &PyAngleFormat,
    ) -> Vec<Bound<'a, PyArray<f64, Ix1>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = self.propagator.states_as_osculating_elements(&epoch_vec, angle_format.value);
        states.iter().map(|s| s.as_slice().to_pyarray(py).to_owned()).collect()
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

/// Python wrapper for KeplerianPropagator (new architecture)
/// Keplerian orbit propagator using two-body dynamics.
///
/// The Keplerian propagator implements ideal two-body orbital mechanics without
/// perturbations. It's fast and accurate for short time spans but doesn't account
/// for real-world effects like drag, J2, solar radiation pressure, etc.
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
    ///     state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)
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
    ///     state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
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
    ///     state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
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
    ///     state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
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
    ///     state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
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
    ///     state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
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
    ///     state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
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
    ///     state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
    ///
    ///     # Change initial conditions to a different orbit
    ///     new_oe = np.array([bh.R_EARTH + 800e3, 0.02, 1.2, 0.5, 0.3, 0.0])
    ///     new_state = bh.state_osculating_to_cartesian(new_oe, bh.AngleFormat.RADIANS)
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
    ///     state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
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
    ///     state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
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
    pub fn state<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> Bound<'a, PyArray<f64, Ix1>> {
        let state = self.propagator.state(epoch.obj);
        state.as_slice().to_pyarray(py).to_owned()
    }

    /// Compute state at a specific epoch in ECI coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ECI frame.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_eci<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> Bound<'a, PyArray<f64, Ix1>> {
        let state = self.propagator.state_eci(epoch.obj);
        state.as_slice().to_pyarray(py).to_owned()
    }

    /// Compute state at a specific epoch in ECEF coordinates.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ECEF frame.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_ecef<'a>(&self, py: Python<'a>, epoch: PyRef<PyEpoch>) -> Bound<'a, PyArray<f64, Ix1>> {
        let state = self.propagator.state_ecef(epoch.obj);
        state.as_slice().to_pyarray(py).to_owned()
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
    pub fn state_as_osculating_elements<'a>(
        &self,
        py: Python<'a>,
        epoch: PyRef<PyEpoch>,
        angle_format: &PyAngleFormat,
    ) -> Bound<'a, PyArray<f64, Ix1>> {
        let state = self.propagator.state_as_osculating_elements(epoch.obj, angle_format.value);
        state.as_slice().to_pyarray(py).to_owned()
    }

    /// Compute states at multiple epochs.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of state vectors in the propagator's native format.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> Vec<Bound<'a, PyArray<f64, Ix1>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = self.propagator.states(&epoch_vec);
        states.iter().map(|s| s.as_slice().to_pyarray(py).to_owned()).collect()
    }

    /// Compute states at multiple epochs in ECI coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of ECI state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_eci<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> Vec<Bound<'a, PyArray<f64, Ix1>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = self.propagator.states_eci(&epoch_vec);
        states.iter().map(|s| s.as_slice().to_pyarray(py).to_owned()).collect()
    }

    /// Compute states at multiple epochs in ECEF coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of ECEF state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_ecef<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> Vec<Bound<'a, PyArray<f64, Ix1>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = self.propagator.states_ecef(&epoch_vec);
        states.iter().map(|s| s.as_slice().to_pyarray(py).to_owned()).collect()
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
    pub fn states_as_osculating_elements<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
        angle_format: &PyAngleFormat,
    ) -> Vec<Bound<'a, PyArray<f64, Ix1>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = self.propagator.states_as_osculating_elements(&epoch_vec, angle_format.value);
        states.iter().map(|s| s.as_slice().to_pyarray(py).to_owned()).collect()
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
    ///     state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
    ///     prop.propagate_steps(10)
    ///     traj = prop.trajectory
    ///     print(f"Trajectory contains {traj.len()} states")
    ///     ```
    #[getter]
    pub fn trajectory(&self) -> PyOrbitalTrajectory {
        PyOrbitalTrajectory { trajectory: self.propagator.trajectory.clone() }
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
        use crate::utils::Identifiable;
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
        use crate::utils::Identifiable;
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
        use crate::utils::Identifiable;
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
        use crate::utils::Identifiable;
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
        use crate::utils::Identifiable;
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
        use crate::utils::Identifiable;
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
        use crate::utils::Identifiable;
        self.propagator.set_id(id);
    }

    /// Set the name in-place (mutating).
    ///
    /// Args:
    ///     name (str or None): Name to assign, or None to clear.
    pub fn set_name(&mut self, name: Option<String>) {
        use crate::utils::Identifiable;
        self.propagator.set_name(name.as_deref());
    }

    /// Generate a new UUID and set it in-place (mutating).
    pub fn generate_uuid(&mut self) {
        use crate::utils::Identifiable;
        self.propagator.generate_uuid();
    }

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
    ///     state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator.from_eci(epc, state, 60.0).with_id(12345)
    ///     print(f"ID: {prop.get_id()}")
    ///     ```
    pub fn get_id(&self) -> Option<u64> {
        use crate::utils::Identifiable;
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
    ///     state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator.from_eci(epc, state, 60.0).with_name("MySat")
    ///     print(f"Name: {prop.get_name()}")
    ///     ```
    pub fn get_name(&self) -> Option<String> {
        use crate::utils::Identifiable;
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
    ///     state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
    ///     prop = bh.KeplerianPropagator.from_eci(epc, state, 60.0).with_new_uuid()
    ///     print(f"UUID: {prop.get_uuid()}")
    ///     ```
    pub fn get_uuid(&self) -> Option<String> {
        use crate::utils::Identifiable;
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
///         state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)
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

    let prop_list = propagators.downcast::<PyList>()?;
    if prop_list.is_empty() {
        return Ok(()); // No propagators to process
    }

    // Determine propagator type from first element
    let first = prop_list.get_item(0)?;

    if first.is_instance_of::<PyKeplerianPropagator>() {
        // Process as Keplerian propagators
        let mut props: Vec<propagators::KeplerianPropagator> = Vec::new();

        for item in prop_list.iter() {
            let py_prop = item.downcast::<PyKeplerianPropagator>()?;
            props.push(py_prop.borrow().propagator.clone());
        }

        // Call Rust parallel propagation function
        propagators::par_propagate_to(&mut props, target_epoch.obj);

        // Update Python objects with new state
        for (i, item) in prop_list.iter().enumerate() {
            let mut py_prop = item.downcast::<PyKeplerianPropagator>()?.borrow_mut();
            py_prop.propagator = props[i].clone();
        }

        Ok(())
    } else if first.is_instance_of::<PySGPPropagator>() {
        // Process as SGP propagators
        let mut props: Vec<propagators::SGPPropagator> = Vec::new();

        for item in prop_list.iter() {
            let py_prop = item.downcast::<PySGPPropagator>()?;
            props.push(py_prop.borrow().propagator.clone());
        }

        // Call Rust parallel propagation function
        propagators::par_propagate_to(&mut props, target_epoch.obj);

        // Update Python objects with new state
        for (i, item) in prop_list.iter().enumerate() {
            let mut py_prop = item.downcast::<PySGPPropagator>()?.borrow_mut();
            py_prop.propagator = props[i].clone();
        }

        Ok(())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "propagators must be a list of KeplerianPropagator or SGPPropagator"
        ))
    }
}
