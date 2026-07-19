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
    pub fn from_tle(
        _cls: &Bound<'_, PyType>,
        line1: String,
        line2: String,
        step_size: Option<f64>,
    ) -> PyResult<Self> {
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
    pub fn from_3le(
        _cls: &Bound<'_, PyType>,
        name: String,
        line1: String,
        line2: String,
        step_size: Option<f64>,
    ) -> PyResult<Self> {
        let step_size = step_size.unwrap_or(60.0);
        match propagators::SGPPropagator::from_3le(Some(&name), &line1, &line2, step_size) {
            Ok(propagator) => Ok(PySGPPropagator { propagator }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Create a new SGP propagator from CCSDS OMM (Orbit Mean-elements Message) fields.
    ///
    /// This method directly constructs an SGP4 propagator from OMM orbital elements,
    /// bypassing TLE parsing. It creates synthetic TLE lines for API consistency.
    ///
    /// Args:
    ///     epoch (str): ISO 8601 datetime string (e.g., "2025-11-29T20:01:44.058144").
    ///     mean_motion (float): Mean motion in revolutions per day.
    ///     eccentricity (float): Orbital eccentricity (dimensionless).
    ///     inclination (float): Orbital inclination in degrees.
    ///     raan (float): Right ascension of ascending node in degrees.
    ///     arg_of_pericenter (float): Argument of pericenter in degrees.
    ///     mean_anomaly (float): Mean anomaly in degrees.
    ///     norad_id (int): NORAD catalog ID.
    ///     step_size (float): Step size in seconds for propagation. Defaults to 60.0.
    ///     object_name (str or None): Satellite name (OBJECT_NAME). Defaults to None.
    ///     object_id (str or None): International designator (OBJECT_ID). Defaults to None.
    ///     classification (str or None): Classification character ('U', 'C', or 'S'). Defaults to 'U'.
    ///     bstar (float or None): B* drag term. Defaults to 0.0.
    ///     mean_motion_dot (float or None): First derivative of mean motion / 2. Defaults to 0.0.
    ///     mean_motion_ddot (float or None): Second derivative of mean motion / 6. Defaults to 0.0.
    ///     ephemeris_type (int or None): Ephemeris type (usually 0). Defaults to 0.
    ///     element_set_no (int or None): Element set number. Defaults to 999.
    ///     rev_at_epoch (int or None): Revolution number at epoch. Defaults to 0.
    ///
    /// Returns:
    ///     SGPPropagator: New SGP propagator instance.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # ISS OMM data
    ///     prop = bh.SGPPropagator.from_omm_elements(
    ///         epoch="2025-11-29T20:01:44.058144",
    ///         mean_motion=15.49193835,
    ///         eccentricity=0.0003723,
    ///         inclination=51.6312,
    ///         raan=206.3646,
    ///         arg_of_pericenter=184.1118,
    ///         mean_anomaly=175.9840,
    ///         norad_id=25544,
    ///         object_name="ISS (ZARYA)",
    ///         object_id="1998-067A",
    ///         bstar=0.15237e-3,
    ///         mean_motion_dot=0.801e-4,
    ///         rev_at_epoch=54085,
    ///     )
    ///     state = prop.state(prop.epoch)
    ///     print(f"Position: {state[:3]}")
    ///     ```
    #[classmethod]
    #[pyo3(signature = (
        epoch,
        mean_motion,
        eccentricity,
        inclination,
        raan,
        arg_of_pericenter,
        mean_anomaly,
        norad_id,
        step_size=60.0,
        object_name=None,
        object_id=None,
        classification=None,
        bstar=None,
        mean_motion_dot=None,
        mean_motion_ddot=None,
        ephemeris_type=None,
        element_set_no=None,
        rev_at_epoch=None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn from_omm_elements(
        _cls: &Bound<'_, PyType>,
        epoch: &str,
        mean_motion: f64,
        eccentricity: f64,
        inclination: f64,
        raan: f64,
        arg_of_pericenter: f64,
        mean_anomaly: f64,
        norad_id: u64,
        step_size: Option<f64>,
        object_name: Option<String>,
        object_id: Option<String>,
        classification: Option<char>,
        bstar: Option<f64>,
        mean_motion_dot: Option<f64>,
        mean_motion_ddot: Option<f64>,
        ephemeris_type: Option<u8>,
        element_set_no: Option<u64>,
        rev_at_epoch: Option<u64>,
    ) -> PyResult<Self> {
        let step_size = step_size.unwrap_or(60.0);
        match propagators::SGPPropagator::from_omm_elements(
            epoch,
            mean_motion,
            eccentricity,
            inclination,
            raan,
            arg_of_pericenter,
            mean_anomaly,
            norad_id,
            step_size,
            object_name.as_deref(),
            object_id.as_deref(),
            classification,
            bstar,
            mean_motion_dot,
            mean_motion_ddot,
            ephemeris_type,
            element_set_no,
            rev_at_epoch,
        ) {
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

    /// Get TLE line 1.
    ///
    /// Returns:
    ///     str: First line of the TLE.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     propagator = bh.SGPPropagator.from_tle(line1, line2)
    ///     print(propagator.line1)
    ///     ```
    #[getter]
    pub fn line1(&self) -> String {
        self.propagator.line1.clone()
    }

    /// Get TLE line 2.
    ///
    /// Returns:
    ///     str: Second line of the TLE.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     propagator = bh.SGPPropagator.from_tle(line1, line2)
    ///     print(propagator.line2)
    ///     ```
    #[getter]
    pub fn line2(&self) -> String {
        self.propagator.line2.clone()
    }

    /// Get TLE epoch.
    ///
    /// Returns:
    ///     Epoch: Epoch of the TLE data.
    #[getter]
    pub fn epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.propagator.epoch,
        }
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
    ///     print(f"Current epoch: {propagator.current_epoch()}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn current_epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.propagator.current_epoch(),
        }
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
    pub fn state<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
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
    pub fn state_pef<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self.propagator.state_pef(epoch.obj)?;
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
    pub fn state_eci<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
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
    pub fn state_ecef<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
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
    pub fn state_gcrf<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = SOrbitStateProvider::state_gcrf(&self.propagator, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Get the state at the given epoch in the central body's body-centered
    /// inertial (BCI) frame: ICRF-aligned axes centered on the body the
    /// states are defined about (GCRF for Earth-centered sources, LCI/MCI
    /// for a Moon/Mars-centered trajectory).
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in the central body's
    ///     inertial frame (meters, m/s).
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_bci<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = SOrbitStateProvider::state_bci(&self.propagator, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Get the state at the given epoch in the central body's body-centered
    /// body-fixed (BCBF) frame (ITRF for Earth-centered sources, LFPA/MCMF
    /// for a Moon/Mars-centered trajectory).
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in the central body's
    ///     body-fixed frame (meters, m/s).
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_bcbf<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = SOrbitStateProvider::state_bcbf(&self.propagator, epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Get the state at the given epoch expressed in an arbitrary reference
    /// frame, converting from the source's native central-body frame.
    ///
    /// Args:
    ///     frame (ReferenceFrame): Reference frame to express the state in.
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in `frame` (meters, m/s).
    #[pyo3(text_signature = "(frame, epoch)")]
    pub fn state_in_frame<'a>(
        &self,
        py: Python<'a>,
        frame: &PyReferenceFrame,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = SOrbitStateProvider::state_in_frame(&self.propagator, frame.frame, epoch.obj)?;
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
    pub fn state_eme2000<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
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
    pub fn state_itrf<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
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
    pub fn states<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = SStateProvider::states(&self.propagator, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in ECI coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of ECI state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_eci<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = SOrbitStateProvider::states_eci(&self.propagator, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in ECEF coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of ECEF state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_ecef<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = SOrbitStateProvider::states_ecef(&self.propagator, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in GCRF coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of GCRF state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_gcrf<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = SOrbitStateProvider::states_gcrf(&self.propagator, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in ITRF coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of ITRF state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_itrf<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = SOrbitStateProvider::states_itrf(&self.propagator, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in EME2000 coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of EME2000 state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_eme2000<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = SOrbitStateProvider::states_eme2000(&self.propagator, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in the propagator's central body's
    /// body-centered inertial (BCI) frame.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of BCI state vectors [x, y, z, vx, vy, vz] (meters, m/s).
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_bci<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = SOrbitStateProvider::states_bci(&self.propagator, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in the propagator's central body's
    /// body-centered body-fixed (BCBF) frame.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of BCBF state vectors [x, y, z, vx, vy, vz] (meters, m/s).
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_bcbf<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = SOrbitStateProvider::states_bcbf(&self.propagator, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs expressed in an arbitrary
    /// reference frame, converting from the propagator's native
    /// central-body frame.
    ///
    /// Args:
    ///     frame (ReferenceFrame): The reference frame to express the states in.
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of state vectors [x, y, z, vx, vy, vz] in `frame` (meters, m/s).
    #[pyo3(text_signature = "(frame, epochs)")]
    pub fn states_in_frame<'a>(
        &self,
        py: Python<'a>,
        frame: &PyReferenceFrame,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states =
            SOrbitStateProvider::states_in_frame(&self.propagator, frame.frame, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
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
    ///     print(f"Advanced to: {prop.current_epoch()}")
    ///     ```
    ///
    /// Raises:
    ///     BraheError: If propagation fails.
    #[pyo3(text_signature = "()")]
    pub fn step(&mut self) -> PyResult<()> {
        Ok(self.propagator.step()?)
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
    ///     print(f"Advanced to: {prop.current_epoch()}")
    ///     ```
    ///
    /// Raises:
    ///     BraheError: If propagation fails.
    #[pyo3(text_signature = "(step_size)")]
    pub fn step_by(&mut self, step_size: f64) -> PyResult<()> {
        Ok(self.propagator.step_by(step_size)?)
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
    ///
    /// Raises:
    ///     BraheError: If propagation fails.
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn step_past(&mut self, target_epoch: PyRef<PyEpoch>) -> PyResult<()> {
        Ok(self.propagator.step_past(target_epoch.obj)?)
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
    ///     print(f"After 10 steps: {prop.current_epoch()}")
    ///     ```
    ///
    /// Raises:
    ///     BraheError: If propagation fails.
    #[pyo3(text_signature = "(num_steps)")]
    pub fn propagate_steps(&mut self, num_steps: usize) -> PyResult<()> {
        Ok(self.propagator.propagate_steps(num_steps)?)
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
    ///     print(f"Propagated to: {prop.current_epoch()}")
    ///     ```
    ///
    /// Raises:
    ///     BraheError: If propagation fails.
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn propagate_to(&mut self, target_epoch: PyRef<PyEpoch>) -> PyResult<()> {
        Ok(self.propagator.propagate_to(target_epoch.obj)?)
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
    ///     print(f"Reset to: {prop.current_epoch() == initial_epoch}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn reset(&mut self) {
        self.propagator.reset();
    }

    /// Set trajectory storage mode.
    ///
    /// Controls whether propagation states are stored in the trajectory.
    /// Use `TrajectoryMode.DISABLED` to prevent unbounded memory growth
    /// during long-duration propagations.
    ///
    /// Args:
    ///     mode (TrajectoryMode): The new trajectory mode.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///     prop.set_trajectory_mode(bh.TrajectoryMode.DISABLED)
    ///     ```
    #[pyo3(text_signature = "(mode)")]
    pub fn set_trajectory_mode(&mut self, mode: &PyTrajectoryMode) {
        self.propagator.set_trajectory_mode(mode.mode);
    }

    /// Get current trajectory storage mode.
    ///
    /// Returns:
    ///     TrajectoryMode: Current trajectory mode.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
    ///     line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///     mode = prop.trajectory_mode
    ///     ```
    #[getter]
    pub fn trajectory_mode(&self) -> PyTrajectoryMode {
        PyTrajectoryMode {
            mode: self.propagator.trajectory_mode(),
        }
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
        self.propagator
            .set_eviction_policy_max_size(max_size)
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
        self.propagator
            .set_eviction_policy_max_age(max_age)
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
        PyOrbitalTrajectory {
            trajectory: self.propagator.trajectory.clone(),
        }
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
    pub fn get_elements<'a>(
        &self,
        py: Python<'a>,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
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

    /// Get age of ephemeris data (time since TLE epoch).
    ///
    /// Returns:
    ///     float: Time since TLE epoch in seconds.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    ///     line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///
    ///     age = prop.ephemeris_age
    ///     print(f"Ephemeris age: {age:.1f} s")
    ///     ```
    #[getter]
    pub fn ephemeris_age(&self) -> f64 {
        self.propagator.ephemeris_age()
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
    pub fn with_uuid(
        mut slf: PyRefMut<'_, Self>,
        uuid_str: String,
    ) -> PyResult<PyRefMut<'_, Self>> {
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
    pub fn with_identity(
        mut slf: PyRefMut<'_, Self>,
        name: Option<String>,
        uuid_str: Option<String>,
        id: Option<u64>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let uuid =
            match uuid_str {
                Some(s) => Some(uuid::Uuid::parse_str(&s).map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e))
                })?),
                None => None,
            };
        slf.propagator = slf
            .propagator
            .clone()
            .with_identity(name.as_deref(), uuid, id);
        Ok(slf)
    }

    /// Set all identity fields in-place (mutating).
    ///
    /// Args:
    ///     name (str or None): Optional name to assign.
    ///     uuid_str (str or None): Optional UUID string to assign.
    ///     id (int or None): Optional numeric ID to assign.
    pub fn set_identity(
        &mut self,
        name: Option<String>,
        uuid_str: Option<String>,
        id: Option<u64>,
    ) -> PyResult<()> {
        let uuid =
            match uuid_str {
                Some(s) => Some(uuid::Uuid::parse_str(&s).map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e))
                })?),
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
    ///     elements_deg = prop.state_koe_osc(epoch, bh.AngleFormat.DEGREES)
    ///     print(f"Semi-major axis: {elements_deg[0]/1000:.3f} km")
    ///     print(f"Inclination: {elements_deg[2]:.4f} degrees")
    ///
    ///     # Get elements in radians
    ///     elements_rad = prop.state_koe_osc(epoch, bh.AngleFormat.RADIANS)
    ///     print(f"Inclination: {elements_rad[2]:.4f} radians")
    ///     ```
    #[pyo3(text_signature = "(epoch, angle_format)")]
    pub fn state_koe_osc<'a>(
        &self,
        py: Python<'a>,
        epoch: PyRef<PyEpoch>,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state =
            SOrbitStateProvider::state_koe_osc(&self.propagator, epoch.obj, angle_format.value)?;
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
    ///     elements_list = prop.states_koe_osc(epochs, bh.AngleFormat.DEGREES)
    ///
    ///     for i, elements in enumerate(elements_list):
    ///         print(f"Hour {i}: a={elements[0]/1000:.3f} km, e={elements[1]:.6f}")
    ///     ```
    #[pyo3(text_signature = "(epochs, angle_format)")]
    pub fn states_koe_osc<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states =
            SOrbitStateProvider::states_koe_osc(&self.propagator, &epoch_vec, angle_format.value)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute state as mean Keplerian elements at the given epoch.
    ///
    /// Mean elements are orbit-averaged elements that remove short-period and
    /// long-period J2 perturbations using first-order Brouwer-Lyddane theory.
    ///
    /// Args:
    ///     epoch (Epoch): Epoch for state computation.
    ///     angle_format (AngleFormat): If AngleFormat.DEGREES, angular elements are returned in degrees, otherwise in radians.
    ///
    /// Returns:
    ///     numpy.ndarray: Mean Keplerian elements [a, e, i, raan, argp, mean_anomaly].
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    ///     line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///
    ///     # Get mean elements at initial epoch
    ///     epoch = prop.epoch
    ///     mean_deg = prop.state_koe_mean(epoch, bh.AngleFormat.DEGREES)
    ///     osc_deg = prop.state_koe_osc(epoch, bh.AngleFormat.DEGREES)
    ///     print(f"Mean semi-major axis: {mean_deg[0]/1000:.3f} km")
    ///     print(f"Osc semi-major axis: {osc_deg[0]/1000:.3f} km")
    ///     ```
    #[pyo3(text_signature = "(epoch, angle_format)")]
    pub fn state_koe_mean<'a>(
        &self,
        py: Python<'a>,
        epoch: PyRef<PyEpoch>,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state =
            SOrbitStateProvider::state_koe_mean(&self.propagator, epoch.obj, angle_format.value)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute states as mean Keplerian elements at multiple epochs.
    ///
    /// Mean elements are orbit-averaged elements that remove short-period and
    /// long-period J2 perturbations using first-order Brouwer-Lyddane theory.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///     angle_format (AngleFormat): If AngleFormat.DEGREES, angular elements are returned in degrees, otherwise in radians.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of mean Keplerian element vectors [a, e, i, raan, argp, mean_anomaly].
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
    ///     line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///
    ///     # Get mean elements at multiple epochs
    ///     epoch0 = prop.epoch
    ///     epochs = [epoch0 + i*3600.0 for i in range(10)]  # Every hour for 10 hours
    ///     mean_list = prop.states_koe_mean(epochs, bh.AngleFormat.DEGREES)
    ///
    ///     for i, elements in enumerate(mean_list):
    ///         print(f"Hour {i}: mean a={elements[0]/1000:.3f} km")
    ///     ```
    #[pyo3(text_signature = "(epochs, angle_format)")]
    pub fn states_koe_mean<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states =
            SOrbitStateProvider::states_koe_mean(&self.propagator, &epoch_vec, angle_format.value)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    // =========================================================================
    // Event Detection Methods
    // =========================================================================

    /// Add an event detector to the propagator.
    ///
    /// Supported event types:
    /// - TimeEvent: Triggers at a specific epoch
    /// - AscendingNodeEvent: Triggers at ascending node crossings
    /// - DescendingNodeEvent: Triggers at descending node crossings
    /// - AltitudeEvent: Triggers at altitude crossings
    /// - And other orbital element events (SemiMajorAxis, Eccentricity, etc.)
    ///
    /// Note:
    ///     Custom ValueEvent and BinaryEvent with Python callbacks are not supported
    ///     for SGPPropagator. Use NumericalOrbitPropagator for custom event functions.
    ///
    /// Args:
    ///     event (TimeEvent | AscendingNodeEvent | DescendingNodeEvent | AltitudeEvent | AOIEntryEvent | AOIExitEvent): Event detector to add
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///
    ///     # Add time event
    ///     epoch = prop.epoch
    ///     event = bh.TimeEvent(epoch + 1800.0, "30 Min Mark")
    ///     prop.add_event_detector(event)
    ///
    ///     # Add node crossing event
    ///     asc_node = bh.AscendingNodeEvent("Ascending Node")
    ///     prop.add_event_detector(asc_node)
    ///
    ///     # Propagate and check events
    ///     prop.propagate_to(epoch + 6000.0)
    ///     for e in prop.event_log():
    ///         print(f"{e.name}: {e.window_open}")
    ///     ```
    #[pyo3(text_signature = "(event)")]
    pub fn add_event_detector(&mut self, event: &Bound<'_, PyAny>) -> PyResult<()> {
        // Try TimeEvent - extract and directly use D-type event
        if let Ok(mut time_event) = event.extract::<PyRefMut<PyTimeEvent>>() {
            if let Some(d_event) = time_event.take_d_event() {
                self.propagator.add_event_detector(Box::new(d_event));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "TimeEvent already consumed",
            ));
        }

        // Try ValueEvent - extract and directly use D-type event
        if let Ok(mut value_event) = event.extract::<PyRefMut<PyValueEvent>>() {
            if let Some(d_event) = value_event.take_d_event() {
                self.propagator.add_event_detector(Box::new(d_event));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "ValueEvent already consumed",
            ));
        }

        // Try BinaryEvent - extract and directly use D-type event
        if let Ok(mut binary_event) = event.extract::<PyRefMut<PyBinaryEvent>>() {
            if let Some(d_event) = binary_event.take_d_event() {
                self.propagator.add_event_detector(Box::new(d_event));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "BinaryEvent already consumed",
            ));
        }

        // Try AltitudeEvent - extract and directly use D-type event
        if let Ok(mut alt_event) = event.extract::<PyRefMut<PyAltitudeEvent>>() {
            if let Some(d_event) = alt_event.take_d_event() {
                self.propagator.add_event_detector(Box::new(d_event));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "AltitudeEvent already consumed",
            ));
        }

        // Try AscendingNodeEvent - extract and directly use D-type event
        if let Ok(mut asc_event) = event.extract::<PyRefMut<PyAscendingNodeEvent>>() {
            if let Some(d_event) = asc_event.take_d_event() {
                self.propagator.add_event_detector(Box::new(d_event));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "AscendingNodeEvent already consumed",
            ));
        }

        // Try DescendingNodeEvent - extract and directly use D-type event
        if let Ok(mut desc_event) = event.extract::<PyRefMut<PyDescendingNodeEvent>>() {
            if let Some(d_event) = desc_event.take_d_event() {
                self.propagator.add_event_detector(Box::new(d_event));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "DescendingNodeEvent already consumed",
            ));
        }

        // Try TrueAnomalyEvent - extract and directly use D-type event
        if let Ok(mut ta_event) = event.extract::<PyRefMut<PyTrueAnomalyEvent>>() {
            if let Some(d_event) = ta_event.take_d_event() {
                self.propagator.add_event_detector(Box::new(d_event));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "TrueAnomalyEvent already consumed",
            ));
        }

        // Try MeanAnomalyEvent - extract and directly use D-type event
        if let Ok(mut ma_event) = event.extract::<PyRefMut<PyMeanAnomalyEvent>>() {
            if let Some(d_event) = ma_event.take_d_event() {
                self.propagator.add_event_detector(Box::new(d_event));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "MeanAnomalyEvent already consumed",
            ));
        }

        // Try EccentricAnomalyEvent - extract and directly use D-type event
        if let Ok(mut ea_event) = event.extract::<PyRefMut<PyEccentricAnomalyEvent>>() {
            if let Some(d_event) = ea_event.take_d_event() {
                self.propagator.add_event_detector(Box::new(d_event));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "EccentricAnomalyEvent already consumed",
            ));
        }

        // Try ArgumentOfLatitudeEvent - extract and directly use D-type event
        if let Ok(mut aol_event) = event.extract::<PyRefMut<PyArgumentOfLatitudeEvent>>() {
            if let Some(d_event) = aol_event.take_d_event() {
                self.propagator.add_event_detector(Box::new(d_event));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "ArgumentOfLatitudeEvent already consumed",
            ));
        }

        // Try AOIEntryEvent - extract and directly use D-type event
        if let Ok(mut aoi_entry_event) = event.extract::<PyRefMut<PyAOIEntryEvent>>() {
            if let Some(d_event) = aoi_entry_event.take_d_event() {
                self.propagator.add_event_detector(Box::new(d_event));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "AOIEntryEvent already consumed",
            ));
        }

        // Try AOIExitEvent - extract and directly use D-type event
        if let Ok(mut aoi_exit_event) = event.extract::<PyRefMut<PyAOIExitEvent>>() {
            if let Some(d_event) = aoi_exit_event.take_d_event() {
                self.propagator.add_event_detector(Box::new(d_event));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "AOIExitEvent already consumed",
            ));
        }

        Err(exceptions::PyTypeError::new_err(
            "Unsupported event type. SGPPropagator supports: TimeEvent, ValueEvent, BinaryEvent, \
             AscendingNodeEvent, DescendingNodeEvent, AltitudeEvent, TrueAnomalyEvent, MeanAnomalyEvent, \
             EccentricAnomalyEvent, ArgumentOfLatitudeEvent, AOIEntryEvent, AOIExitEvent.",
        ))
    }

    /// Get all detected events from the event log.
    ///
    /// Returns:
    ///     list[DetectedEvent]: List of all detected events.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     prop = bh.SGPPropagator.from_tle(line1, line2)
    ///     event = bh.AscendingNodeEvent("Asc Node")
    ///     prop.add_event_detector(event)
    ///
    ///     prop.propagate_to(prop.epoch + 6000.0)
    ///
    ///     for e in prop.event_log():
    ///         print(f"{e.name} at {e.window_open}")
    ///     ```
    #[pyo3(text_signature = "()")]
    pub fn event_log(&self) -> Vec<PyDetectedEvent> {
        self.propagator
            .event_log()
            .iter()
            .map(|e| PyDetectedEvent { event: e.clone() })
            .collect()
    }

    /// Get events filtered by name.
    ///
    /// Args:
    ///     name (str): Exact event name to match.
    ///
    /// Returns:
    ///     list[DetectedEvent]: Events matching the name.
    #[pyo3(text_signature = "(name)")]
    pub fn events_by_name(&self, name: &str) -> Vec<PyDetectedEvent> {
        self.propagator
            .events_by_name(name)
            .iter()
            .map(|e| PyDetectedEvent {
                event: (*e).clone(),
            })
            .collect()
    }

    /// Get the most recent detected event.
    ///
    /// Returns:
    ///     DetectedEvent or None: Most recent event, or None if no events.
    #[pyo3(text_signature = "()")]
    pub fn latest_event(&self) -> Option<PyDetectedEvent> {
        self.propagator
            .latest_event()
            .map(|e| PyDetectedEvent { event: e.clone() })
    }

    /// Get events within a time range.
    ///
    /// Args:
    ///     start (Epoch): Start of time range.
    ///     end (Epoch): End of time range.
    ///
    /// Returns:
    ///     list[DetectedEvent]: Events within the range.
    #[pyo3(text_signature = "(start, end)")]
    pub fn events_in_range(
        &self,
        start: PyRef<PyEpoch>,
        end: PyRef<PyEpoch>,
    ) -> Vec<PyDetectedEvent> {
        self.propagator
            .events_in_range(start.obj, end.obj)
            .iter()
            .map(|e| PyDetectedEvent {
                event: (*e).clone(),
            })
            .collect()
    }

    /// Check if propagation was terminated by an event or propagation error.
    ///
    /// Returns:
    ///     bool: True if propagation was stopped by a terminal event or error.
    #[getter]
    pub fn terminated(&self) -> bool {
        self.propagator.is_terminated()
    }

    /// Get the error that caused propagation termination, if any.
    ///
    /// Returns:
    ///     str or None: Error message if terminated due to propagation error,
    ///         None if terminated by an event or not terminated.
    #[getter]
    pub fn termination_error(&self) -> Option<String> {
        self.propagator.termination_error().map(|e| e.to_string())
    }

    /// Reset the termination flag.
    ///
    /// Call this to allow propagation to continue after a terminal event or error.
    #[pyo3(text_signature = "()")]
    pub fn reset_termination(&mut self) {
        self.propagator.reset_termination();
    }

    /// Clear all events from the event log.
    #[pyo3(text_signature = "()")]
    pub fn clear_events(&mut self) {
        self.propagator.clear_events();
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "SGPPropagator(norad_id={}, name={:?}, epoch={:?})",
            self.propagator.norad_id, self.propagator.satellite_name, self.propagator.epoch
        )
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
                "State vector must have exactly 6 elements",
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
                "Elements vector must have exactly 6 elements",
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
                "State vector must have exactly 6 elements",
            ));
        }

        let state_vec = na::Vector6::from_row_slice(state_array.as_slice().unwrap());

        let propagator =
            propagators::KeplerianPropagator::from_eci(epoch.obj, state_vec, step_size);

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
                "State vector must have exactly 6 elements",
            ));
        }

        let state_vec = na::Vector6::from_row_slice(state_array.as_slice().unwrap());

        let propagator =
            propagators::KeplerianPropagator::from_ecef(epoch.obj, state_vec, step_size);

        Ok(PyKeplerianPropagator { propagator })
    }

    /// Get current epoch.
    ///
    /// Returns:
    ///     Epoch: Current propagator epoch.
    #[pyo3(text_signature = "()")]
    pub fn current_epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.propagator.current_epoch(),
        }
    }

    /// Get initial epoch.
    ///
    /// Returns:
    ///     Epoch: Initial propagator epoch.
    #[getter]
    pub fn initial_epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.propagator.initial_epoch(),
        }
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
    ///     print(f"Advanced to: {prop.current_epoch()}")
    ///     ```
    ///
    /// Raises:
    ///     BraheError: If propagation fails.
    #[pyo3(text_signature = "()")]
    pub fn step(&mut self) -> PyResult<()> {
        Ok(self.propagator.step()?)
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
    ///     print(f"Advanced to: {prop.current_epoch()}")
    ///     ```
    ///
    /// Raises:
    ///     BraheError: If propagation fails.
    #[pyo3(text_signature = "(step_size)")]
    pub fn step_by(&mut self, step_size: f64) -> PyResult<()> {
        Ok(self.propagator.step_by(step_size)?)
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
    ///     print(f"Advanced to: {prop.current_epoch()}")
    ///     ```
    ///
    /// Raises:
    ///     BraheError: If propagation fails.
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn step_past(&mut self, target_epoch: PyRef<PyEpoch>) -> PyResult<()> {
        Ok(self.propagator.step_past(target_epoch.obj)?)
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
    ///     print(f"Advanced to: {prop.current_epoch()}")
    ///     ```
    ///
    /// Raises:
    ///     BraheError: If propagation fails.
    #[pyo3(text_signature = "(num_steps)")]
    pub fn propagate_steps(&mut self, num_steps: usize) -> PyResult<()> {
        Ok(self.propagator.propagate_steps(num_steps)?)
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
    ///     print(f"Propagated to: {prop.current_epoch()}")
    ///     ```
    ///
    /// Raises:
    ///     BraheError: If propagation fails.
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn propagate_to(&mut self, target_epoch: PyRef<PyEpoch>) -> PyResult<()> {
        Ok(self.propagator.propagate_to(target_epoch.obj)?)
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
    ///     print(f"Reset to: {prop.current_epoch()}")
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
                "State vector must have exactly 6 elements",
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
    pub fn state<'a>(
        &self,
        py: Python<'a>,
        epoch: PyRef<PyEpoch>,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
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
    pub fn state_eci<'a>(
        &self,
        py: Python<'a>,
        epoch: PyRef<PyEpoch>,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
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
    pub fn state_ecef<'a>(
        &self,
        py: Python<'a>,
        epoch: PyRef<PyEpoch>,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
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
    pub fn state_gcrf<'a>(
        &self,
        py: Python<'a>,
        epoch: PyRef<PyEpoch>,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self.propagator.state_gcrf(epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Get the state at the given epoch in the central body's body-centered
    /// inertial (BCI) frame. KeplerianPropagator is Earth-centered, so this
    /// is the GCRF state.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in the central body's
    ///     inertial frame (meters, m/s).
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_bci<'a>(
        &self,
        py: Python<'a>,
        epoch: PyRef<PyEpoch>,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self.propagator.state_bci(epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Get the state at the given epoch in the central body's body-centered
    /// body-fixed (BCBF) frame. KeplerianPropagator is Earth-centered, so
    /// this is the ITRF state.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in the central body's
    ///     body-fixed frame (meters, m/s).
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_bcbf<'a>(
        &self,
        py: Python<'a>,
        epoch: PyRef<PyEpoch>,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self.propagator.state_bcbf(epoch.obj)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Get the state at the given epoch expressed in an arbitrary reference
    /// frame, converting from GCRF (this propagator's central body's
    /// inertial frame).
    ///
    /// Args:
    ///     frame (ReferenceFrame): Reference frame to express the state in.
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in `frame` (meters, m/s).
    #[pyo3(text_signature = "(frame, epoch)")]
    pub fn state_in_frame<'a>(
        &self,
        py: Python<'a>,
        frame: &PyReferenceFrame,
        epoch: PyRef<PyEpoch>,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self.propagator.state_in_frame(frame.frame, epoch.obj)?;
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
    pub fn state_eme2000<'a>(
        &self,
        py: Python<'a>,
        epoch: PyRef<PyEpoch>,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
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
    pub fn state_itrf<'a>(
        &self,
        py: Python<'a>,
        epoch: PyRef<PyEpoch>,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
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
    pub fn state_koe_osc<'a>(
        &self,
        py: Python<'a>,
        epoch: PyRef<PyEpoch>,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state =
            DOrbitStateProvider::state_koe_osc(&self.propagator, epoch.obj, angle_format.value)?;
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
    pub fn states<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DStateProvider::states(&self.propagator, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in ECI coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of ECI state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_eci<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_eci(&self.propagator, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in ECEF coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of ECEF state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_ecef<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_ecef(&self.propagator, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in GCRF coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of GCRF state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_gcrf<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_gcrf(&self.propagator, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in ITRF coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of ITRF state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_itrf<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_itrf(&self.propagator, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in EME2000 coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of EME2000 state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_eme2000<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_eme2000(&self.propagator, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in the propagator's central body's
    /// body-centered inertial (BCI) frame.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of BCI state vectors [x, y, z, vx, vy, vz] (meters, m/s).
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_bci<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_bci(&self.propagator, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in the propagator's central body's
    /// body-centered body-fixed (BCBF) frame.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of BCBF state vectors [x, y, z, vx, vy, vz] (meters, m/s).
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_bcbf<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_bcbf(&self.propagator, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs expressed in an arbitrary
    /// reference frame, converting from the propagator's native
    /// central-body frame.
    ///
    /// Args:
    ///     frame (ReferenceFrame): The reference frame to express the states in.
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of state vectors [x, y, z, vx, vy, vz] in `frame` (meters, m/s).
    #[pyo3(text_signature = "(frame, epochs)")]
    pub fn states_in_frame<'a>(
        &self,
        py: Python<'a>,
        frame: &PyReferenceFrame,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states =
            DOrbitStateProvider::states_in_frame(&self.propagator, frame.frame, &epoch_vec)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
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
    pub fn states_koe_osc<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states =
            DOrbitStateProvider::states_koe_osc(&self.propagator, &epoch_vec, angle_format.value)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute state as mean Keplerian elements at a specific epoch.
    ///
    /// Mean elements are orbit-averaged elements that remove short-period and
    /// long-period J2 perturbations using first-order Brouwer-Lyddane theory.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///     angle_format (AngleFormat): If AngleFormat.DEGREES, angular elements are returned in degrees, otherwise in radians.
    ///
    /// Returns:
    ///     numpy.ndarray: Mean Keplerian elements [a, e, i, raan, argp, mean_anomaly].
    #[pyo3(text_signature = "(epoch, angle_format)")]
    pub fn state_koe_mean<'a>(
        &self,
        py: Python<'a>,
        epoch: PyRef<PyEpoch>,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state =
            DOrbitStateProvider::state_koe_mean(&self.propagator, epoch.obj, angle_format.value)?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute states as mean Keplerian elements at multiple epochs.
    ///
    /// Mean elements are orbit-averaged elements that remove short-period and
    /// long-period J2 perturbations using first-order Brouwer-Lyddane theory.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///     angle_format (AngleFormat): If AngleFormat.DEGREES, angular elements are returned in degrees, otherwise in radians.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of mean Keplerian element vectors [a, e, i, raan, argp, mean_anomaly].
    #[pyo3(text_signature = "(epochs, angle_format)")]
    pub fn states_koe_mean<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states =
            DOrbitStateProvider::states_koe_mean(&self.propagator, &epoch_vec, angle_format.value)?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
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
        let states: Vec<DVector<f64>> = traj
            .states
            .iter()
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
        d_traj.set_identity(traj.get_name(), traj.get_uuid(), traj.get_id());

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
    pub fn with_uuid(
        mut slf: PyRefMut<'_, Self>,
        uuid_str: String,
    ) -> PyResult<PyRefMut<'_, Self>> {
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
    pub fn with_identity(
        mut slf: PyRefMut<'_, Self>,
        name: Option<String>,
        uuid_str: Option<String>,
        id: Option<u64>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let uuid =
            match uuid_str {
                Some(s) => Some(uuid::Uuid::parse_str(&s).map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e))
                })?),
                None => None,
            };
        slf.propagator = slf
            .propagator
            .clone()
            .with_identity(name.as_deref(), uuid, id);
        Ok(slf)
    }

    /// Set all identity fields in-place (mutating).
    ///
    /// Args:
    ///     name (str or None): Optional name to assign.
    ///     uuid_str (str or None): Optional UUID string to assign.
    ///     id (int or None): Optional numeric ID to assign.
    pub fn set_identity(
        &mut self,
        name: Option<String>,
        uuid_str: Option<String>,
        id: Option<u64>,
    ) -> PyResult<()> {
        let uuid =
            match uuid_str {
                Some(s) => Some(uuid::Uuid::parse_str(&s).map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e))
                })?),
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
        self.propagator.set_name(name.as_deref())
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
        format!(
            "KeplerianPropagator(epoch={:?}, step_size={})",
            self.propagator.current_epoch(),
            self.propagator.step_size()
        )
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
/// The list may freely mix `KeplerianPropagator`, `SGPPropagator`, and
/// `NumericalOrbitPropagator` instances. Propagators are grouped by type and each
/// group is propagated in parallel; results are written back to the original objects
/// in place, preserving their order.
///
/// Note: `NumericalPropagator` (with user-defined Python dynamics) is NOT supported because
/// Python callbacks cannot safely execute in parallel due to the GIL. Use `NumericalOrbitPropagator`
/// for parallel propagation of orbital dynamics.
///
/// Note: For SGPPropagator and NumericalOrbitPropagator, event detectors and event logs are
/// properly preserved during parallel propagation. Events detected during propagation will be
/// available in each propagator's `event_log()` after the call completes.
///
/// Args:
///     propagators (Sequence[KeplerianPropagator] or Sequence[SGPPropagator] or Sequence[NumericalOrbitPropagator]): List of propagators to update.
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
fn py_par_propagate_to(propagators: &Bound<'_, PyAny>, target_epoch: &PyEpoch) -> PyResult<()> {
    // Check if propagators is a list
    if !propagators.is_instance_of::<PyList>() {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "propagators must be a list of KeplerianPropagator, SGPPropagator, or NumericalOrbitPropagator",
        ));
    }

    let prop_list = propagators.cast::<PyList>()?;
    if prop_list.is_empty() {
        return Ok(()); // No propagators to process
    }

    // Classify every element by type up front. The whole list is validated before
    // any propagation runs, so a list containing an unsupported propagator fails
    // cleanly without partially mutating the others. Indices are recorded per type
    // so each homogeneous group can be propagated in parallel and results written
    // back to the correct original objects.
    let mut kep_idx: Vec<usize> = Vec::new();
    let mut sgp_idx: Vec<usize> = Vec::new();
    let mut num_orbit_idx: Vec<usize> = Vec::new();

    for (i, item) in prop_list.iter().enumerate() {
        if item.is_instance_of::<PyKeplerianPropagator>() {
            kep_idx.push(i);
        } else if item.is_instance_of::<PySGPPropagator>() {
            sgp_idx.push(i);
        } else if item.is_instance_of::<PyNumericalOrbitPropagator>() {
            num_orbit_idx.push(i);
        } else if item.is_instance_of::<PyNumericalPropagator>() {
            // NumericalPropagator uses Python callbacks for dynamics, which cannot
            // safely execute in parallel due to the GIL. Provide a helpful message.
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "NumericalPropagator cannot be used with par_propagate_to because its Python \
                 dynamics callback cannot safely execute in parallel due to the GIL. \
                 Use NumericalOrbitPropagator for parallel propagation of orbital dynamics, \
                 or call propagate_to() sequentially on each NumericalPropagator.",
            ));
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "propagators must be a list of KeplerianPropagator, SGPPropagator, or NumericalOrbitPropagator",
            ));
        }
    }

    // Process Keplerian propagators as a group.
    if !kep_idx.is_empty() {
        let mut props: Vec<propagators::KeplerianPropagator> = Vec::with_capacity(kep_idx.len());

        for &i in &kep_idx {
            let item = prop_list.get_item(i)?;
            let py_prop = item.cast::<PyKeplerianPropagator>()?;
            props.push(py_prop.borrow().propagator.clone());
        }

        // Call Rust parallel propagation function
        propagators::par_propagate_to_s(&mut props, target_epoch.obj)?;

        // Update Python objects with new state
        for (k, &i) in kep_idx.iter().enumerate() {
            let item = prop_list.get_item(i)?;
            let mut py_prop = item.cast::<PyKeplerianPropagator>()?.borrow_mut();
            py_prop.propagator = props[k].clone();
        }
    }

    // Process SGP propagators as a group.
    if !sgp_idx.is_empty() {
        // Note: SGPPropagator::Clone does NOT clone event_detectors (trait objects can't be cloned)
        // We must extract them before cloning and reattach after parallel propagation
        let mut props: Vec<propagators::SGPPropagator> = Vec::with_capacity(sgp_idx.len());
        let mut extracted_detectors: Vec<Vec<Box<dyn events::DEventDetector>>> = Vec::new();

        for &i in &sgp_idx {
            let item = prop_list.get_item(i)?;
            let mut py_prop = item.cast::<PySGPPropagator>()?.borrow_mut();
            // Take ownership of event detectors BEFORE cloning (they would be lost on clone)
            extracted_detectors.push(py_prop.propagator.take_event_detectors());
            props.push(py_prop.propagator.clone());
        }

        // Reattach event detectors to the cloned propagators
        for (k, detectors) in extracted_detectors.into_iter().enumerate() {
            props[k].set_event_detectors(detectors);
        }

        // Call Rust parallel propagation function
        propagators::par_propagate_to_s(&mut props, target_epoch.obj)?;

        // Transfer results back to Python objects
        for (k, &i) in sgp_idx.iter().enumerate() {
            let item = prop_list.get_item(i)?;
            let mut py_prop = item.cast::<PySGPPropagator>()?.borrow_mut();

            // Take event state from propagated clone before transferring
            let detectors = props[k].take_event_detectors();
            let event_log = props[k].take_event_log();
            let terminated = props[k].is_terminated();
            let termination_error = props[k].take_termination_error();

            // Transfer full propagator state (trajectory, epoch_current, state_current, etc.)
            py_prop.propagator = props[k].clone();

            // Restore event detection state lost in clone
            py_prop.propagator.set_event_detectors(detectors);
            py_prop.propagator.set_event_log(event_log);
            py_prop.propagator.set_terminated(terminated);
            py_prop.propagator.set_termination_error(termination_error);
        }
    }

    // Process NumericalOrbitPropagator instances as a group.
    if !num_orbit_idx.is_empty() {
        // Note: DNumericalOrbitPropagator cannot be cloned (has Box<dyn DIntegrator>),
        // so we work directly with mutable references using raw pointers

        // Wrapper for raw pointer that implements Send
        // SAFETY: We ensure that each pointer is accessed by only one thread
        struct SendPtr(*mut propagators::DNumericalOrbitPropagator);
        unsafe impl Send for SendPtr {}
        unsafe impl Sync for SendPtr {}

        // Collect borrow guards (must stay alive while we use raw pointers)
        let mut borrow_guards: Vec<PyRefMut<'_, PyNumericalOrbitPropagator>> =
            Vec::with_capacity(num_orbit_idx.len());

        for &i in &num_orbit_idx {
            let item = prop_list.get_item(i)?;
            borrow_guards.push(item.cast::<PyNumericalOrbitPropagator>()?.borrow_mut());
        }

        // Create wrapped raw pointers from the borrow guards
        let prop_ptrs: Vec<SendPtr> = borrow_guards
            .iter_mut()
            .map(|guard| SendPtr(&mut guard.propagator as *mut _))
            .collect();

        // Use rayon scoped parallelism with wrapped raw pointers
        // SAFETY: No Python objects are involved in the parallel execution.
        // Each propagator (DNumericalOrbitPropagator) is accessed by exactly one thread,
        // and the borrow_guards are kept alive throughout this scope to ensure validity.
        let target = target_epoch.obj;
        let result: Result<(), RustBraheError> =
            brahe::utils::threading::get_thread_pool().install(|| {
                prop_ptrs.par_iter().try_for_each(|SendPtr(ptr)| {
                    // SAFETY: ptr is valid because it points to data owned by borrow_guards
                    // which are still alive in the outer scope
                    unsafe { (*(*ptr)).propagate_to(target) }
                })
            });

        // On failure, re-raise the original Python exception stashed by whichever
        // propagator's additional-dynamics/control callback raised it, falling
        // back to the wrapped BraheError when the failure is purely numerical.
        if let Err(e) = result {
            for guard in &borrow_guards {
                if let Some(py_err) = guard.err_slot.lock().unwrap().take() {
                    return Err(py_err);
                }
            }
            return Err(PyErr::from(e));
        }

        // borrow_guards are dropped here, releasing the borrows
    }

    Ok(())
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
#[pyclass(module = "brahe._brahe", from_py_object)]
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
        PyIntegrationMethod {
            method: propagators::IntegratorMethod::RK4,
        }
    }

    /// Runge-Kutta-Fehlberg 4(5) adaptive
    #[classattr]
    fn RKF45() -> Self {
        PyIntegrationMethod {
            method: propagators::IntegratorMethod::RKF45,
        }
    }

    /// Runge-Kutta-Fehlberg 7(8) adaptive
    #[classattr]
    fn RKF78() -> Self {
        PyIntegrationMethod {
            method: propagators::IntegratorMethod::RKF78,
        }
    }

    /// Dormand-Prince 5(4) adaptive (default, MATLAB's ode45)
    #[classattr]
    fn DP54() -> Self {
        PyIntegrationMethod {
            method: propagators::IntegratorMethod::DP54,
        }
    }

    /// Runge-Kutta-Nystrom 12(10) adaptive (high-precision)
    #[classattr]
    fn RKN1210() -> Self {
        PyIntegrationMethod {
            method: propagators::IntegratorMethod::RKN1210,
        }
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
#[pyclass(module = "brahe._brahe", from_py_object)]
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
            self.config.enable_stm,
            self.config.enable_sensitivity,
            self.config.store_stm_history,
            self.config.store_sensitivity_history
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
#[pyclass(module = "brahe._brahe", from_py_object)]
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
        PyAtmosphericModel {
            model: propagators::AtmosphericModel::HarrisPriester,
        }
    }

    /// NRLMSISE-00 empirical model (high-fidelity, requires space weather)
    #[classattr]
    fn NRLMSISE00() -> Self {
        PyAtmosphericModel {
            model: propagators::AtmosphericModel::NRLMSISE00,
        }
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
            model: propagators::AtmosphericModel::Exponential {
                scale_height,
                rho0,
                h0,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!("AtmosphericModel({:?})", self.model)
    }
}

/// Maximum zonal harmonic degree for `GravityConfiguration.earth_zonal`.
///
/// Selects the highest J_n term retained when evaluating the closed-form zonal
/// gravity model. The expansion always starts at J_2.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     degree = bh.ZonalHarmonicsDegree.J2
///     degree = bh.ZonalHarmonicsDegree.J6
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "ZonalHarmonicsDegree")]
#[derive(Clone)]
pub struct PyZonalHarmonicsDegree {
    pub degree: propagators::ZonalHarmonicsDegree,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyZonalHarmonicsDegree {
    /// J2 only (oblateness term)
    #[classattr]
    fn J2() -> Self {
        PyZonalHarmonicsDegree {
            degree: propagators::ZonalHarmonicsDegree::J2,
        }
    }

    /// Through J3
    #[classattr]
    fn J3() -> Self {
        PyZonalHarmonicsDegree {
            degree: propagators::ZonalHarmonicsDegree::J3,
        }
    }

    /// Through J4
    #[classattr]
    fn J4() -> Self {
        PyZonalHarmonicsDegree {
            degree: propagators::ZonalHarmonicsDegree::J4,
        }
    }

    /// Through J5
    #[classattr]
    fn J5() -> Self {
        PyZonalHarmonicsDegree {
            degree: propagators::ZonalHarmonicsDegree::J5,
        }
    }

    /// Through J6
    #[classattr]
    fn J6() -> Self {
        PyZonalHarmonicsDegree {
            degree: propagators::ZonalHarmonicsDegree::J6,
        }
    }

    fn __repr__(&self) -> String {
        format!("ZonalHarmonicsDegree.J{}", usize::from(&self.degree))
    }
}

/// ECI-to-body-fixed rotation model used by the numerical propagator's force evaluation.
///
/// Selects the precision/speed trade-off for every body-fixed force term (spherical-harmonic
/// and zonal gravity, NRLMSISE-00 density, drag).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     transform = bh.FrameTransformationModel.FULL_EARTH_ROTATION
///     transform = bh.FrameTransformationModel.EARTH_ROTATION_ONLY
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "FrameTransformationModel")]
#[derive(Clone)]
pub struct PyFrameTransformationModel {
    pub model: propagators::FrameTransformationModel,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyFrameTransformationModel {
    /// Full IAU 2006/2000A rotation: bias-precession-nutation + ERA + polar motion (default).
    #[classattr]
    fn FULL_EARTH_ROTATION() -> Self {
        PyFrameTransformationModel {
            model: propagators::FrameTransformationModel::FullEarthRotation,
        }
    }

    /// Earth Rotation Angle only — ~1.5x faster but ignores precession, nutation, and polar motion.
    #[classattr]
    fn EARTH_ROTATION_ONLY() -> Self {
        PyFrameTransformationModel {
            model: propagators::FrameTransformationModel::EarthRotationOnly,
        }
    }

    fn __repr__(&self) -> String {
        format!("FrameTransformationModel.{:?}", self.model)
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
#[pyclass(module = "brahe._brahe", from_py_object)]
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
        PyEclipseModel {
            model: propagators::EclipseModel::Conical,
        }
    }

    /// Cylindrical eclipse model (simpler, faster)
    #[classattr]
    fn CYLINDRICAL() -> Self {
        PyEclipseModel {
            model: propagators::EclipseModel::Cylindrical,
        }
    }

    /// Cylindrical eclipse model (simpler, faster)
    #[classattr]
    fn NONE() -> Self {
        PyEclipseModel {
            model: propagators::EclipseModel::None,
        }
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
#[pyclass(module = "brahe._brahe", from_py_object)]
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
            propagators::ParameterSource::ParameterIndex(i) => {
                format!("ParameterSource.parameter_index({})", i)
            }
        }
    }
}

// =============================================================================
// Parallel Mode
// =============================================================================

/// Parallelization mode for spherical harmonic gravity evaluation.
///
/// Controls whether the spherical-harmonic computation runs in parallel
/// (via Brahe's managed Rayon thread pool) or serially.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     mode = bh.ParallelMode.Auto
///     mode = bh.ParallelMode.Always
///     mode = bh.ParallelMode.Never
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "ParallelMode")]
#[derive(Clone)]
pub struct PyParallelMode {
    pub mode: brahe::orbit_dynamics::ParallelMode,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyParallelMode {
    /// Parallelize only when the expansion degree meets the auto-threshold (default).
    #[classattr]
    fn Auto() -> Self {
        PyParallelMode {
            mode: brahe::orbit_dynamics::ParallelMode::Auto,
        }
    }

    /// Always parallelize via the global thread pool.
    #[classattr]
    fn Always() -> Self {
        PyParallelMode {
            mode: brahe::orbit_dynamics::ParallelMode::Always,
        }
    }

    /// Always run serially.
    #[classattr]
    fn Never() -> Self {
        PyParallelMode {
            mode: brahe::orbit_dynamics::ParallelMode::Never,
        }
    }

    fn __repr__(&self) -> String {
        format!("ParallelMode.{:?}", self.mode)
    }
}

// =============================================================================
// Tides Configuration
// =============================================================================

/// Solid Earth tide settings.
///
/// Controls whether Step 2 (frequency-dependent) IERS corrections are applied,
/// and whether the solid Earth pole tide is applied.
///
/// Args:
///     frequency_dependent (bool): Apply IERS Step 2 frequency-dependent corrections.
///         Default is False.
///     pole_tide (bool): Apply the solid Earth pole tide ΔC̄21/ΔS̄21 (IERS
///         TN36 Section 6.4). Requires EOP initialization. Default is False.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     solid = bh.SolidTideConfig(frequency_dependent=True, pole_tide=True)
///     assert solid.frequency_dependent is True
///     assert solid.pole_tide is True
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "SolidTideConfig")]
#[derive(Clone)]
pub struct PySolidTideConfig {
    pub config: brahe::orbit_dynamics::tides::SolidTideConfig,
}

#[pymethods]
impl PySolidTideConfig {
    #[new]
    #[pyo3(signature = (frequency_dependent=false, pole_tide=false))]
    fn new(frequency_dependent: bool, pole_tide: bool) -> Self {
        Self {
            config: brahe::orbit_dynamics::tides::SolidTideConfig {
                frequency_dependent,
                pole_tide,
            },
        }
    }

    /// Whether Step 2 frequency-dependent corrections are enabled.
    #[getter]
    fn frequency_dependent(&self) -> bool {
        self.config.frequency_dependent
    }

    /// Set whether Step 2 frequency-dependent corrections are enabled.
    #[setter]
    fn set_frequency_dependent(&mut self, v: bool) {
        self.config.frequency_dependent = v;
    }

    /// Whether the solid Earth pole tide is enabled.
    #[getter]
    fn pole_tide(&self) -> bool {
        self.config.pole_tide
    }

    /// Set whether the solid Earth pole tide is enabled.
    #[setter]
    fn set_pole_tide(&mut self, v: bool) {
        self.config.pole_tide = v;
    }

    fn __repr__(&self) -> String {
        format!(
            "SolidTideConfig(frequency_dependent={}, pole_tide={})",
            self.config.frequency_dependent, self.config.pole_tide
        )
    }
}

/// FES2004 ocean tide configuration (IERS TN36 Section 6.3) plus the ocean
/// pole tide (Section 6.5).
///
/// Requires a one-time download of the IERS FES2004 coefficient file into
/// the brahe cache on first use.
///
/// Args:
///     degree (int): Truncation degree, 2-100. Defaults to 20.
///     order (int): Truncation order, <= degree. Defaults to 20.
///     include_admittance (bool): Add secondary waves by admittance
///         interpolation (TN36 Table 6.7). Defaults to True.
///     pole_tide (bool): Apply the ocean pole tide (2,1) term (TN36
///         Eq. 6.24). Requires EOP initialization. Defaults to False.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     ocean = bh.OceanTideConfig(degree=30, order=30)
///     tides = bh.TidesConfiguration(
///         permanent=bh.PermanentTideConfig.AUTO,
///         solid=bh.SolidTideConfig(frequency_dependent=True, pole_tide=True),
///         ocean=ocean,
///     )
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "OceanTideConfig")]
#[derive(Clone)]
pub struct PyOceanTideConfig {
    pub config: propagators::OceanTideConfig,
}

#[pymethods]
impl PyOceanTideConfig {
    #[new]
    #[pyo3(signature = (degree=20, order=20, include_admittance=true, pole_tide=false))]
    fn new(degree: usize, order: usize, include_admittance: bool, pole_tide: bool) -> Self {
        Self {
            config: propagators::OceanTideConfig {
                degree,
                order,
                include_admittance,
                pole_tide,
            },
        }
    }

    /// Truncation degree of the ocean tide expansion.
    #[getter]
    fn degree(&self) -> usize {
        self.config.degree
    }

    /// Set the truncation degree of the ocean tide expansion.
    #[setter]
    fn set_degree(&mut self, v: usize) {
        self.config.degree = v;
    }

    /// Truncation order of the ocean tide expansion.
    #[getter]
    fn order(&self) -> usize {
        self.config.order
    }

    /// Set the truncation order of the ocean tide expansion.
    #[setter]
    fn set_order(&mut self, v: usize) {
        self.config.order = v;
    }

    /// Whether secondary waves are added by admittance interpolation.
    #[getter]
    fn include_admittance(&self) -> bool {
        self.config.include_admittance
    }

    /// Set whether secondary waves are added by admittance interpolation.
    #[setter]
    fn set_include_admittance(&mut self, v: bool) {
        self.config.include_admittance = v;
    }

    /// Whether the ocean pole tide is enabled.
    #[getter]
    fn pole_tide(&self) -> bool {
        self.config.pole_tide
    }

    /// Set whether the ocean pole tide is enabled.
    #[setter]
    fn set_pole_tide(&mut self, v: bool) {
        self.config.pole_tide = v;
    }

    fn __repr__(&self) -> String {
        format!(
            "OceanTideConfig(degree={}, order={}, include_admittance={}, pole_tide={})",
            self.config.degree,
            self.config.order,
            self.config.include_admittance,
            self.config.pole_tide
        )
    }
}

/// Permanent (zero-frequency) tide handling for the static gravity field.
///
/// Controls how the loaded model's C̄20 is reconciled with the solid-tide model.
///
/// Use the class attributes ``AUTO`` and ``OFF`` for the unit variants, or
/// ``PermanentTideConfig.convert_to(system)`` to force a specific tide system.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Auto-detect from model flag (default)
///     perm = bh.PermanentTideConfig.AUTO
///
///     # Disable permanent-tide correction
///     perm = bh.PermanentTideConfig.OFF
///
///     # Force convert to a specific system
///     perm = bh.PermanentTideConfig.convert_to(bh.GravityModelTideSystem.ZeroTide)
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "PermanentTideConfig")]
#[derive(Clone)]
pub struct PyPermanentTideConfig {
    pub config: propagators::PermanentTideConfig,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyPermanentTideConfig {
    /// Auto-detect tide system from the model's flag and convert C̄20 to tide-free. (default)
    #[classattr]
    fn AUTO() -> Self {
        Self {
            config: propagators::PermanentTideConfig::Auto,
        }
    }

    /// Leave C̄20 untouched.
    #[classattr]
    fn OFF() -> Self {
        Self {
            config: propagators::PermanentTideConfig::Off,
        }
    }

    /// Force the gravity field into the given tide system.
    ///
    /// Args:
    ///     system (GravityModelTideSystem): Target tide system.
    ///
    /// Returns:
    ///     PermanentTideConfig: A PermanentTideConfig that converts to the given system.
    #[staticmethod]
    fn convert_to(system: &PyGravityModelTideSystem) -> Self {
        Self {
            config: propagators::PermanentTideConfig::ConvertTo(system.tide_system),
        }
    }

    fn __repr__(&self) -> String {
        match &self.config {
            propagators::PermanentTideConfig::Auto => "PermanentTideConfig.AUTO".to_string(),
            propagators::PermanentTideConfig::Off => "PermanentTideConfig.OFF".to_string(),
            propagators::PermanentTideConfig::ConvertTo(s) => {
                format!(
                    "PermanentTideConfig.convert_to(GravityModelTideSystem.{:?})",
                    s
                )
            }
        }
    }
}

/// Tidal correction configuration for ForceModelConfig.
///
/// Args:
///     permanent (PermanentTideConfig): Permanent-tide / tide-system handling.
///     solid (SolidTideConfig, optional): Solid Earth tide configuration.
///         None disables solid tides. Default is None.
///     ocean (OceanTideConfig, optional): Ocean tide configuration. None
///         disables ocean tides. Default is None.
///     ephemeris_source (EphemerisSource, optional): Source for the Sun and
///         Moon positions the tidal corrections are computed from. Defaults to
///         EphemerisSource.LowPrecision (the analytic geocentric ephemerides),
///         which is accurate enough for the ~1e-7 m/s^2 tidal perturbation. Set
///         to a high-precision source to share positions with a third-body
///         perturbation configured against the same source.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     solid = bh.SolidTideConfig(frequency_dependent=True)
///     tides = bh.TidesConfiguration(permanent=bh.PermanentTideConfig.AUTO, solid=solid)
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "TidesConfiguration")]
#[derive(Clone)]
pub struct PyTidesConfiguration {
    pub config: propagators::TidesConfiguration,
}

#[pymethods]
impl PyTidesConfiguration {
    #[new]
    #[pyo3(signature = (permanent, solid=None, ocean=None, ephemeris_source=PyEphemerisSource::LowPrecision))]
    fn new(
        py: Python,
        permanent: &PyPermanentTideConfig,
        solid: Option<&PySolidTideConfig>,
        ocean: Option<&PyOceanTideConfig>,
        ephemeris_source: PyEphemerisSource,
    ) -> PyResult<Self> {
        if let propagators::PermanentTideConfig::ConvertTo(sys) = &permanent.config
            && *sys != orbit_dynamics::GravityModelTideSystem::TideFree
            && solid.is_some()
        {
            let msg = std::ffi::CString::new(format!(
                "PermanentTideConfig.convert_to(GravityModelTideSystem.{sys:?}) combined \
                     with solid Earth tides double-counts the permanent tide: the solid-tide \
                     model (IERS \u{a7}6.2.1) already includes the permanent part and expects a \
                     conventional tide-free background field. Use \
                     convert_to(GravityModelTideSystem.TideFree) or PermanentTideConfig.AUTO, \
                     or disable solid tides."
            ))?;
            PyErr::warn(
                py,
                &py.get_type::<pyo3::exceptions::PyUserWarning>(),
                &msg,
                2,
            )?;
        }
        Ok(Self {
            config: propagators::TidesConfiguration {
                permanent: permanent.config.clone(),
                solid: solid.map(|s| s.config),
                ocean: ocean.map(|o| o.config),
                ephemeris_source: ephemeris_source.into(),
            },
        })
    }

    /// Get the permanent tide configuration.
    #[getter]
    fn permanent(&self) -> PyPermanentTideConfig {
        PyPermanentTideConfig {
            config: self.config.permanent.clone(),
        }
    }

    /// Get the solid Earth tide configuration (None if disabled).
    #[getter]
    fn solid(&self) -> Option<PySolidTideConfig> {
        self.config.solid.map(|s| PySolidTideConfig { config: s })
    }

    /// Get the ocean tide configuration (None if disabled).
    #[getter]
    fn ocean(&self) -> Option<PyOceanTideConfig> {
        self.config.ocean.map(|o| PyOceanTideConfig { config: o })
    }

    /// Get the ephemeris source for the tidal Sun and Moon positions.
    #[getter]
    fn ephemeris_source(&self) -> PyEphemerisSource {
        match self.config.ephemeris_source {
            propagators::EphemerisSource::LowPrecision => PyEphemerisSource::LowPrecision,
            propagators::EphemerisSource::DE440s => PyEphemerisSource::DE440s,
            propagators::EphemerisSource::DE440 => PyEphemerisSource::DE440,
            propagators::EphemerisSource::SPK(spice::SPICEKernel::DE440) => {
                PyEphemerisSource::DE440
            }
            // The Python `EphemerisSource` enum exposes only LowPrecision/DE440s/DE440,
            // so it can never construct any other SPK kernel; map the remainder to
            // DE440s as the closest Python-visible source.
            propagators::EphemerisSource::SPK(_) => PyEphemerisSource::DE440s,
        }
    }

    /// Set the ephemeris source for the tidal Sun and Moon positions.
    #[setter]
    fn set_ephemeris_source(&mut self, source: PyEphemerisSource) {
        self.config.ephemeris_source = source.into();
    }

    fn __repr__(&self) -> String {
        format!(
            "TidesConfiguration(permanent={:?}, solid={:?}, ocean={:?}, ephemeris_source={:?})",
            self.config.permanent, self.config.solid, self.config.ocean, self.config.ephemeris_source
        )
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
///     model_type (GravityModelType, optional): Gravity model to use.
///         Defaults to EGM2008_120.
///     use_global (bool, optional): If True, use global gravity model.
///         Defaults to False.
///     parallel (ParallelMode, optional): Parallelization mode for the
///         spherical-harmonic acceleration computation. Defaults to Auto.
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
///     # Spherical harmonic with specific model
///     gravity = bh.GravityConfiguration(
///         degree=20, order=20, model_type=bh.GravityModelType.GGM05S
///     )
///
///     # Alternative: use class methods
///     gravity = bh.GravityConfiguration.point_mass()
///     gravity = bh.GravityConfiguration.spherical_harmonic(degree=20, order=20)
///     gravity = bh.GravityConfiguration.earth_zonal(bh.ZonalHarmonicsDegree.J6)
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
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
    ///     model_type (GravityModelType, optional): Gravity model to use.
    ///         Defaults to EGM2008_120.
    ///     use_global (bool, optional): If True, use global gravity model.
    ///         Defaults to False.
    ///     parallel (ParallelMode, optional): Parallelization mode for the
    ///         spherical-harmonic acceleration computation. Defaults to Auto.
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
    ///
    ///     # Spherical harmonic with specific model
    ///     gravity = bh.GravityConfiguration(
    ///         degree=20, order=20, model_type=bh.GravityModelType.GGM05S
    ///     )
    ///     ```
    #[new]
    #[pyo3(signature = (degree=None, order=None, model_type=None, use_global=false, parallel=None))]
    fn new(
        degree: Option<usize>,
        order: Option<usize>,
        model_type: Option<&PyGravityModelType>,
        use_global: bool,
        parallel: Option<&PyParallelMode>,
    ) -> Self {
        match (degree, order) {
            (Some(d), Some(o)) => {
                let source = if use_global {
                    propagators::GravityModelSource::Global
                } else {
                    let model = model_type
                        .map(|mt| mt.model.clone())
                        .unwrap_or(brahe::orbit_dynamics::gravity::GravityModelType::EGM2008_120);
                    propagators::GravityModelSource::ModelType(model)
                };
                let parallel_mode = parallel
                    .map(|p| p.mode)
                    .unwrap_or(brahe::orbit_dynamics::ParallelMode::Auto);
                PyGravityConfiguration {
                    config: propagators::GravityConfiguration::SphericalHarmonic {
                        source,
                        degree: d,
                        order: o,
                        parallel: parallel_mode,
                    },
                }
            }
            _ => PyGravityConfiguration {
                config: propagators::GravityConfiguration::PointMass,
            },
        }
    }

    /// Create a configuration with no gravity term from the propagation
    /// center. For barycentric propagation centers (``CentralBody.EMB``,
    /// ``CentralBody.SSB``), which have no mass of their own: every
    /// gravitational force enters through the third-body entries.
    ///
    /// Returns:
    ///     GravityConfiguration: No central gravity term.
    #[classmethod]
    fn zero(_cls: &Bound<'_, PyType>) -> Self {
        PyGravityConfiguration {
            config: propagators::GravityConfiguration::Zero,
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
    ///     model_type (GravityModelType, optional): Gravity model to use.
    ///         Defaults to EGM2008_120.
    ///     use_global (bool, optional): If True, use global gravity model.
    ///         Defaults to False.
    ///     parallel (ParallelMode, optional): Parallelization mode for the
    ///         spherical-harmonic acceleration computation. Defaults to Auto.
    ///
    /// Returns:
    ///     GravityConfiguration: Spherical harmonic gravity.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Default (EGM2008)
    ///     gravity = bh.GravityConfiguration.spherical_harmonic(degree=20, order=20)
    ///
    ///     # With specific model
    ///     gravity = bh.GravityConfiguration.spherical_harmonic(
    ///         degree=20, order=20, model_type=bh.GravityModelType.GGM05S
    ///     )
    ///     ```
    #[classmethod]
    #[pyo3(signature = (degree, order, model_type=None, use_global=false, parallel=None))]
    fn spherical_harmonic(
        _cls: &Bound<'_, PyType>,
        degree: usize,
        order: usize,
        model_type: Option<&PyGravityModelType>,
        use_global: bool,
        parallel: Option<&PyParallelMode>,
    ) -> Self {
        let source = if use_global {
            propagators::GravityModelSource::Global
        } else {
            let model = model_type
                .map(|mt| mt.model.clone())
                .unwrap_or(brahe::orbit_dynamics::gravity::GravityModelType::EGM2008_120);
            propagators::GravityModelSource::ModelType(model)
        };
        let parallel_mode = parallel
            .map(|p| p.mode)
            .unwrap_or(brahe::orbit_dynamics::ParallelMode::Auto);
        PyGravityConfiguration {
            config: propagators::GravityConfiguration::SphericalHarmonic {
                source,
                degree,
                order,
                parallel: parallel_mode,
            },
        }
    }

    /// Create an Earth zonal-only gravity configuration (J_2..=J_n, m=0).
    ///
    /// Equivalent to `spherical_harmonic` with `m = 0` against the packaged
    /// Earth gravity model, but evaluated via an explicit closed-form
    /// expansion that the compiler can vectorise — ~50% faster for the same
    /// axially-symmetric expansion. Earth-specific because the J_n
    /// coefficients and reference radius are baked into the implementation.
    ///
    /// Args:
    ///     degree (ZonalHarmonicsDegree): Maximum zonal degree (J_2 through J_6).
    ///
    /// Returns:
    ///     GravityConfiguration: Earth zonal gravity.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     gravity = bh.GravityConfiguration.earth_zonal(bh.ZonalHarmonicsDegree.J6)
    ///     ```
    #[classmethod]
    fn earth_zonal(_cls: &Bound<'_, PyType>, degree: &PyZonalHarmonicsDegree) -> Self {
        PyGravityConfiguration {
            config: propagators::GravityConfiguration::EarthZonal {
                degree: degree.degree.clone(),
            },
        }
    }

    /// Check if this is point mass gravity.
    fn is_point_mass(&self) -> bool {
        matches!(self.config, propagators::GravityConfiguration::PointMass)
    }

    /// Check if this is spherical harmonic gravity.
    fn is_spherical_harmonic(&self) -> bool {
        matches!(
            self.config,
            propagators::GravityConfiguration::SphericalHarmonic { .. }
        )
    }

    /// Check if this is Earth zonal gravity.
    fn is_earth_zonal(&self) -> bool {
        matches!(
            self.config,
            propagators::GravityConfiguration::EarthZonal { .. }
        )
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

    /// Get the parallel mode (for spherical harmonic).
    ///
    /// Returns:
    ///     ParallelMode or None: Parallel mode if spherical harmonic, None otherwise.
    fn get_parallel(&self) -> Option<PyParallelMode> {
        match &self.config {
            propagators::GravityConfiguration::SphericalHarmonic { parallel, .. } => {
                Some(PyParallelMode { mode: *parallel })
            }
            _ => None,
        }
    }

    /// Get the zonal degree (for Earth zonal gravity).
    ///
    /// Returns:
    ///     ZonalHarmonicsDegree or None: Zonal degree if Earth zonal gravity, None otherwise.
    fn get_earth_zonal_degree(&self) -> Option<PyZonalHarmonicsDegree> {
        match &self.config {
            propagators::GravityConfiguration::EarthZonal { degree } => {
                Some(PyZonalHarmonicsDegree {
                    degree: degree.clone(),
                })
            }
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.config {
            propagators::GravityConfiguration::Zero => {
                "GravityConfiguration.zero()".to_string()
            }
            propagators::GravityConfiguration::PointMass => {
                "GravityConfiguration.point_mass()".to_string()
            }
            propagators::GravityConfiguration::SphericalHarmonic {
                degree,
                order,
                parallel,
                ..
            } => {
                format!(
                    "GravityConfiguration.spherical_harmonic(degree={}, order={}, parallel=ParallelMode.{:?})",
                    degree, order, parallel
                )
            }
            propagators::GravityConfiguration::EarthZonal { degree, .. } => {
                format!("GravityConfiguration.earth_zonal(degree={})", degree)
            }
        }
    }
}

// =============================================================================
// Drag Configuration
// =============================================================================

/// Atmospheric drag configuration.
///
/// Defines the atmospheric model and drag parameters. The optional `body`
/// attributes the drag to a body other than the propagation's central body:
/// density and relative wind are then evaluated at the object's state
/// relative to that body (e.g. Earth drag on an EMB-centered cislunar
/// trajectory).
///
/// Args:
///     model (AtmosphericModel): Atmospheric density model.
///     area (ParameterSource): Drag cross-sectional area source [m²].
///     cd (ParameterSource): Drag coefficient source (dimensionless).
///     body (CentralBody, optional): Body whose atmosphere produces the
///         drag. Defaults to None, meaning the propagation's central body.
///
/// Attributes:
///     model (AtmosphericModel): Atmospheric density model
///     area (ParameterSource): Drag area source
///     cd (ParameterSource): Drag coefficient source
///     body (CentralBody or None): Attributed drag body
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
///
///     # Earth-attributed drag for an EMB-centered propagation
///     cislunar_drag = bh.DragConfiguration(
///         model=bh.AtmosphericModel.NRLMSISE00,
///         area=bh.ParameterSource.value(10.0),
///         cd=bh.ParameterSource.value(2.2),
///         body=bh.CentralBody.Earth,
///     )
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
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
    ///     body (CentralBody, optional): Body whose atmosphere produces the
    ///         drag. Defaults to None, meaning the propagation's central body.
    #[new]
    #[pyo3(signature = (model, area, cd, body=None))]
    fn new(
        model: &PyAtmosphericModel,
        area: &PyParameterSource,
        cd: &PyParameterSource,
        body: Option<&PyCentralBody>,
    ) -> Self {
        PyDragConfiguration {
            config: propagators::DragConfiguration {
                model: model.model.clone(),
                area: area.source.clone(),
                cd: cd.source.clone(),
                body: body.map(|b| b.body.clone()),
            },
        }
    }

    /// Get the attributed drag body (None = the propagation's central body).
    #[getter]
    fn body(&self) -> Option<PyCentralBody> {
        self.config.body.as_ref().map(|b| PyCentralBody { body: b.clone() })
    }

    /// Set the attributed drag body (None = the propagation's central body).
    #[setter]
    fn set_body(&mut self, body: Option<&PyCentralBody>) {
        self.config.body = body.map(|b| b.body.clone());
    }

    /// Get the atmospheric model.
    #[getter]
    fn model(&self) -> PyAtmosphericModel {
        PyAtmosphericModel {
            model: self.config.model.clone(),
        }
    }

    /// Set the atmospheric model.
    #[setter]
    fn set_model(&mut self, model: &PyAtmosphericModel) {
        self.config.model = model.model.clone();
    }

    /// Get the drag area parameter source.
    #[getter]
    fn area(&self) -> PyParameterSource {
        PyParameterSource {
            source: self.config.area.clone(),
        }
    }

    /// Set the drag area parameter source.
    #[setter]
    fn set_area(&mut self, area: &PyParameterSource) {
        self.config.area = area.source.clone();
    }

    /// Get the drag coefficient parameter source.
    #[getter]
    fn cd(&self) -> PyParameterSource {
        PyParameterSource {
            source: self.config.cd.clone(),
        }
    }

    /// Set the drag coefficient parameter source.
    #[setter]
    fn set_cd(&mut self, cd: &PyParameterSource) {
        self.config.cd = cd.source.clone();
    }

    fn __repr__(&self) -> String {
        format!(
            "DragConfiguration(model={:?}, area={:?}, cd={:?}, body={:?})",
            self.config.model, self.config.area, self.config.cd, self.config.body
        )
    }
}

// =============================================================================
// Central Body
// =============================================================================

/// The central body an orbit is propagated relative to.
///
/// `Earth`, `Moon`, and `Mars` are built in because they have dedicated named
/// inertial/fixed frame pairs elsewhere in brahe (`GCRF`/`ITRF`, `LCI`/`LFPA`,
/// `MCI`/`MCMF`). `EMB` and `SSB` are the Earth-Moon and Solar System
/// barycenters -- useful as propagation origins for heliocentric or cislunar
/// trajectories, but they have no physical radius, spin, or fixed frame. Any
/// other body is constructed via `CentralBody.Custom(...)` or
/// `CentralBody.from_naif_id(...)`.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     earth = bh.CentralBody.Earth
///     moon = bh.CentralBody.Moon
///     enceladus = bh.CentralBody.from_naif_id(602)
///     ```
#[pyclass(module = "brahe._brahe", eq, from_py_object)]
#[pyo3(name = "CentralBody")]
#[derive(Clone, PartialEq)]
pub struct PyCentralBody {
    pub(crate) body: propagators::CentralBody,
}

#[pymethods]
impl PyCentralBody {
    /// Earth (NAIF ID 399).
    #[classattr]
    #[allow(non_snake_case)]
    fn Earth() -> Self {
        PyCentralBody { body: propagators::CentralBody::Earth }
    }

    /// Moon (NAIF ID 301).
    #[classattr]
    #[allow(non_snake_case)]
    fn Moon() -> Self {
        PyCentralBody { body: propagators::CentralBody::Moon }
    }

    /// Mars (body center, NAIF ID 499).
    #[classattr]
    #[allow(non_snake_case)]
    fn Mars() -> Self {
        PyCentralBody { body: propagators::CentralBody::Mars }
    }

    /// Earth-Moon barycenter (NAIF ID 3).
    #[classattr]
    #[allow(non_snake_case)]
    fn EMB() -> Self {
        PyCentralBody { body: propagators::CentralBody::EMB }
    }

    /// Solar System barycenter (NAIF ID 0).
    #[classattr]
    #[allow(non_snake_case)]
    fn SSB() -> Self {
        PyCentralBody { body: propagators::CentralBody::SSB }
    }

    /// Construct a user-defined central body.
    ///
    /// Args:
    ///     name (str): Human-readable name (e.g. `"Enceladus"`).
    ///     naif_id (int): NAIF ID of the body.
    ///     gm (float): Gravitational parameter. Units: (m^3/s^2)
    ///     radius (float, optional): Mean or equatorial radius, if known. Units: (m)
    ///     omega (numpy.ndarray or list, optional): Body-fixed axial spin vector, if known. Units: (rad/s)
    ///     fixed_frame (ReferenceFrame, optional): Body-fixed reference frame, required for
    ///         spherical-harmonic gravity and body-fixed rotations.
    ///
    /// Returns:
    ///     CentralBody: A user-defined central body.
    #[staticmethod]
    #[pyo3(signature = (name, naif_id, gm, radius=None, omega=None, fixed_frame=None))]
    #[allow(non_snake_case)]
    fn Custom(
        name: String,
        naif_id: i32,
        gm: f64,
        radius: Option<f64>,
        omega: Option<Bound<'_, PyAny>>,
        fixed_frame: Option<PyReferenceFrame>,
    ) -> PyResult<Self> {
        let omega_vec = match omega {
            Some(o) => Some(pyany_to_svector::<3>(&o)?),
            None => None,
        };
        Ok(PyCentralBody {
            body: propagators::CentralBody::Custom(propagators::CustomBody {
                name,
                naif_id,
                gm,
                radius,
                omega: omega_vec,
                fixed_frame: fixed_frame.map(|f| f.frame),
            }),
        })
    }

    /// Construct a `CentralBody` from a NAIF ID.
    ///
    /// `399`, `301`, `4`/`499`, `3`, and `0` map to the built-in `Earth`,
    /// `Moon`, `Mars`, `EMB`, and `SSB` variants respectively. A fixed table of
    /// other commonly used bodies maps to a pre-populated `Custom` variant.
    ///
    /// Args:
    ///     naif_id (int): NAIF ID of the body.
    ///
    /// Returns:
    ///     CentralBody: The corresponding central body.
    ///
    /// Raises:
    ///     ValueError: If `naif_id` is not a built-in body or in the embedded table.
    #[staticmethod]
    fn from_naif_id(naif_id: i32) -> PyResult<Self> {
        propagators::CentralBody::from_naif_id(naif_id)
            .map(|body| PyCentralBody { body })
            .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Gravitational parameter of the central body.
    ///
    /// Returns:
    ///     float: Gravitational parameter. Units: (m^3/s^2). `0.0` for the `EMB` and `SSB` barycenters.
    fn gm(&self) -> f64 {
        self.body.gm()
    }

    /// Mean or equatorial radius of the central body.
    ///
    /// Returns:
    ///     float or None: Radius, if known. Units: (m). `None` for the `EMB` and `SSB` barycenters.
    fn radius(&self) -> Option<f64> {
        self.body.radius()
    }

    /// NAIF ID of the central body.
    ///
    /// Returns:
    ///     int: NAIF ID.
    fn naif_id(&self) -> i32 {
        self.body.naif_id()
    }

    /// Body-fixed axial spin vector of the central body.
    ///
    /// Returns:
    ///     numpy.ndarray or None: Spin vector expressed in the body's inertial frame, if
    ///     known. Units: (rad/s). `None` for the `EMB`/`SSB` barycenters and for `Custom`
    ///     bodies unless `omega` was set explicitly.
    fn omega_vector<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray<f64, Ix1>>> {
        self.body.omega_vector().map(|v| vector_to_numpy!(py, v, 3, f64))
    }

    /// ICRF-aligned inertial reference frame centered on this body.
    ///
    /// Returns:
    ///     ReferenceFrame: `GCRF` for `Earth`, `LCI` for `Moon`, `MCI` for `Mars`, `EMBI`
    ///     for `EMB`, `SSBI` for `SSB`, and `BodyCenteredICRF(naif_id)` for `Custom` bodies.
    fn inertial_frame(&self) -> PyReferenceFrame {
        PyReferenceFrame { frame: self.body.inertial_frame() }
    }

    /// Body-fixed reference frame of this body, if one is defined.
    ///
    /// Returns:
    ///     ReferenceFrame or None: `ITRF` for `Earth`, `LFPA` for `Moon`, `MCMF` for `Mars`,
    ///     `None` for `EMB`/`SSB`, and `custom.fixed_frame` for `Custom` bodies.
    fn fixed_frame(&self) -> Option<PyReferenceFrame> {
        self.body.fixed_frame().map(|frame| PyReferenceFrame { frame })
    }

    /// Whether this central body is a barycenter (`EMB` or `SSB`).
    ///
    /// Returns:
    ///     bool: `True` for `EMB`/`SSB`, `False` otherwise.
    fn is_barycenter(&self) -> bool {
        self.body.is_barycenter()
    }

    fn __str__(&self) -> String {
        self.body.to_string()
    }

    fn __repr__(&self) -> String {
        format!("CentralBody.{:?}", self.body)
    }
}

// =============================================================================
// Occulting Body
// =============================================================================

/// Occulting body for eclipse/shadow modeling in solar radiation pressure calculations.
///
/// Identifies a celestial body whose shadow may occult the sun as seen from the
/// spacecraft.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     earth = bh.OccultingBody.Earth
///     custom = bh.OccultingBody.Custom(name="Europa", naif_id=502, radius=1560.8e3)
///     ```
#[pyclass(module = "brahe._brahe", eq, from_py_object)]
#[pyo3(name = "OccultingBody")]
#[derive(Clone, PartialEq)]
pub struct PyOccultingBody {
    pub(crate) body: propagators::OccultingBody,
}

#[pymethods]
impl PyOccultingBody {
    /// Earth.
    #[classattr]
    #[allow(non_snake_case)]
    fn Earth() -> Self {
        PyOccultingBody { body: propagators::OccultingBody::Earth }
    }

    /// Moon.
    #[classattr]
    #[allow(non_snake_case)]
    fn Moon() -> Self {
        PyOccultingBody { body: propagators::OccultingBody::Moon }
    }

    /// Mars.
    #[classattr]
    #[allow(non_snake_case)]
    fn Mars() -> Self {
        PyOccultingBody { body: propagators::OccultingBody::Mars }
    }

    /// Construct a user-defined occulting body.
    ///
    /// Args:
    ///     name (str): Descriptive name of the body.
    ///     naif_id (int): NAIF ID of the physical body.
    ///     radius (float): Mean physical radius of the body. Units: (m)
    ///
    /// Returns:
    ///     OccultingBody: A user-defined occulting body.
    #[staticmethod]
    #[pyo3(signature = (name, naif_id, radius))]
    #[allow(non_snake_case)]
    fn Custom(name: String, naif_id: i32, radius: f64) -> Self {
        PyOccultingBody { body: propagators::OccultingBody::Custom { name, naif_id, radius } }
    }

    /// Mean physical radius of the occulting body.
    ///
    /// Returns:
    ///     float: Physical radius of the body. Units: (m)
    fn radius(&self) -> f64 {
        self.body.radius()
    }

    /// NAIF ID of the physical occulting body.
    ///
    /// Returns:
    ///     int: NAIF integer ID of the physical body.
    fn naif_id(&self) -> i32 {
        self.body.naif_id()
    }

    /// NAIF ID to use when resolving the occulting body's position via SPK ephemerides.
    ///
    /// Returns:
    ///     int: NAIF integer ID to use for SPK position queries. Identical to `naif_id`
    ///     for every variant (the physical body center).
    fn naif_position_id(&self) -> i32 {
        self.body.naif_position_id()
    }

    fn __repr__(&self) -> String {
        format!("OccultingBody.{:?}", self.body)
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
///     occulting_bodies (list[OccultingBody], optional): Bodies whose shadow may occult
///         the sun. Defaults to `[OccultingBody.Earth]`.
///
/// Attributes:
///     area (ParameterSource): SRP area source
///     cr (ParameterSource): Reflectivity coefficient source
///     eclipse_model (EclipseModel): Eclipse model
///     occulting_bodies (list[OccultingBody]): Bodies whose shadow may occult the sun
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
#[pyclass(module = "brahe._brahe", from_py_object)]
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
    ///     occulting_bodies (list[OccultingBody], optional): Bodies whose shadow may occult
    ///         the sun. Defaults to `[OccultingBody.Earth]`.
    #[new]
    #[pyo3(signature = (area, cr, eclipse_model, occulting_bodies=None))]
    fn new(
        area: &PyParameterSource,
        cr: &PyParameterSource,
        eclipse_model: &PyEclipseModel,
        occulting_bodies: Option<Vec<PyOccultingBody>>,
    ) -> Self {
        PySolarRadiationPressureConfiguration {
            config: propagators::SolarRadiationPressureConfiguration {
                area: area.source.clone(),
                cr: cr.source.clone(),
                eclipse_model: eclipse_model.model.clone(),
                occulting_bodies: occulting_bodies
                    .map(|bodies| bodies.into_iter().map(|b| b.body).collect())
                    .unwrap_or_else(|| vec![propagators::OccultingBody::Earth]),
            },
        }
    }

    /// Get the SRP area parameter source.
    #[getter]
    fn area(&self) -> PyParameterSource {
        PyParameterSource {
            source: self.config.area.clone(),
        }
    }

    /// Set the SRP area parameter source.
    #[setter]
    fn set_area(&mut self, area: &PyParameterSource) {
        self.config.area = area.source.clone();
    }

    /// Get the coefficient of reflectivity parameter source.
    #[getter]
    fn cr(&self) -> PyParameterSource {
        PyParameterSource {
            source: self.config.cr.clone(),
        }
    }

    /// Set the coefficient of reflectivity parameter source.
    #[setter]
    fn set_cr(&mut self, cr: &PyParameterSource) {
        self.config.cr = cr.source.clone();
    }

    /// Get the eclipse model.
    #[getter]
    fn eclipse_model(&self) -> PyEclipseModel {
        PyEclipseModel {
            model: self.config.eclipse_model.clone(),
        }
    }

    /// Set the eclipse model.
    #[setter]
    fn set_eclipse_model(&mut self, eclipse_model: &PyEclipseModel) {
        self.config.eclipse_model = eclipse_model.model.clone();
    }

    /// Get the bodies whose shadow may occult the sun.
    #[getter]
    fn occulting_bodies(&self) -> Vec<PyOccultingBody> {
        self.config.occulting_bodies.iter().map(|b| PyOccultingBody { body: b.clone() }).collect()
    }

    /// Set the bodies whose shadow may occult the sun.
    #[setter]
    fn set_occulting_bodies(&mut self, occulting_bodies: Vec<PyOccultingBody>) {
        self.config.occulting_bodies = occulting_bodies.into_iter().map(|b| b.body).collect();
    }

    fn __repr__(&self) -> String {
        format!(
            "SolarRadiationPressureConfiguration(area={:?}, cr={:?}, eclipse_model={:?}, occulting_bodies={:?})",
            self.config.area, self.config.cr, self.config.eclipse_model, self.config.occulting_bodies
        )
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
///     ceres = bh.ThirdBody.Custom(name="Ceres", naif_id=2000001, gm=6.26325e10)
///     ```
#[pyclass(module = "brahe._brahe", eq, from_py_object)]
#[pyo3(name = "ThirdBody")]
#[derive(Clone, PartialEq)]
pub struct PyThirdBody {
    pub(crate) body: propagators::ThirdBody,
}

#[pymethods]
impl PyThirdBody {
    /// Sun.
    #[classattr]
    #[allow(non_snake_case)]
    fn SUN() -> Self {
        PyThirdBody { body: propagators::ThirdBody::Sun }
    }

    /// Moon.
    #[classattr]
    #[allow(non_snake_case)]
    fn MOON() -> Self {
        PyThirdBody { body: propagators::ThirdBody::Moon }
    }

    /// Mercury.
    #[classattr]
    #[allow(non_snake_case)]
    fn MERCURY() -> Self {
        PyThirdBody { body: propagators::ThirdBody::Mercury }
    }

    /// Venus.
    #[classattr]
    #[allow(non_snake_case)]
    fn VENUS() -> Self {
        PyThirdBody { body: propagators::ThirdBody::Venus }
    }

    /// Mars (planet center, NAIF 499) with the planet-only GM. For the
    /// classical third-body formulation about Earth prefer
    /// `MARS_BARYCENTER`, which is resolvable from the DE kernels alone.
    #[classattr]
    #[allow(non_snake_case)]
    fn MARS() -> Self {
        PyThirdBody { body: propagators::ThirdBody::Mars }
    }

    /// Mars system barycenter (NAIF 4): Mars, Phobos, and Deimos with the
    /// system GM. Used by the default Earth force models.
    #[classattr]
    #[allow(non_snake_case)]
    fn MARS_BARYCENTER() -> Self {
        PyThirdBody { body: propagators::ThirdBody::MarsBarycenter }
    }

    /// Jupiter (planet center, NAIF 599) with the planet-only GM.
    #[classattr]
    #[allow(non_snake_case)]
    fn JUPITER() -> Self {
        PyThirdBody { body: propagators::ThirdBody::Jupiter }
    }

    /// Jupiter system barycenter (NAIF 5) with the system GM.
    #[classattr]
    #[allow(non_snake_case)]
    fn JUPITER_BARYCENTER() -> Self {
        PyThirdBody { body: propagators::ThirdBody::JupiterBarycenter }
    }

    /// Saturn (planet center, NAIF 699) with the planet-only GM.
    #[classattr]
    #[allow(non_snake_case)]
    fn SATURN() -> Self {
        PyThirdBody { body: propagators::ThirdBody::Saturn }
    }

    /// Saturn system barycenter (NAIF 6) with the system GM.
    #[classattr]
    #[allow(non_snake_case)]
    fn SATURN_BARYCENTER() -> Self {
        PyThirdBody { body: propagators::ThirdBody::SaturnBarycenter }
    }

    /// Uranus (planet center, NAIF 799) with the planet-only GM.
    #[classattr]
    #[allow(non_snake_case)]
    fn URANUS() -> Self {
        PyThirdBody { body: propagators::ThirdBody::Uranus }
    }

    /// Uranus system barycenter (NAIF 7) with the system GM.
    #[classattr]
    #[allow(non_snake_case)]
    fn URANUS_BARYCENTER() -> Self {
        PyThirdBody { body: propagators::ThirdBody::UranusBarycenter }
    }

    /// Neptune (planet center, NAIF 899) with the planet-only GM.
    #[classattr]
    #[allow(non_snake_case)]
    fn NEPTUNE() -> Self {
        PyThirdBody { body: propagators::ThirdBody::Neptune }
    }

    /// Neptune system barycenter (NAIF 8) with the system GM.
    #[classattr]
    #[allow(non_snake_case)]
    fn NEPTUNE_BARYCENTER() -> Self {
        PyThirdBody { body: propagators::ThirdBody::NeptuneBarycenter }
    }

    /// Earth. Only meaningful as a perturber when the central body is not
    /// Earth itself (e.g. `CentralBody.EMB` or `CentralBody.Mars`).
    #[classattr]
    #[allow(non_snake_case)]
    fn EARTH() -> Self {
        PyThirdBody { body: propagators::ThirdBody::Earth }
    }

    /// Phobos, the larger of Mars's two moons.
    #[classattr]
    #[allow(non_snake_case)]
    fn PHOBOS() -> Self {
        PyThirdBody { body: propagators::ThirdBody::Phobos }
    }

    /// Deimos, the smaller of Mars's two moons.
    #[classattr]
    #[allow(non_snake_case)]
    fn DEIMOS() -> Self {
        PyThirdBody { body: propagators::ThirdBody::Deimos }
    }

    /// Construct a user-defined perturbing body.
    ///
    /// Args:
    ///     name (str): Human-readable name (e.g. `"Ceres"`).
    ///     naif_id (int): NAIF ID of the body.
    ///     gm (float): Gravitational parameter. Units: (m^3/s^2)
    ///
    /// Returns:
    ///     ThirdBody: A user-defined perturbing body.
    #[staticmethod]
    #[pyo3(signature = (name, naif_id, gm))]
    #[allow(non_snake_case)]
    fn Custom(name: String, naif_id: i32, gm: f64) -> Self {
        PyThirdBody { body: propagators::ThirdBody::Custom { name, naif_id, gm } }
    }

    /// NAIF ID of the perturbing body.
    ///
    /// Returns:
    ///     int: NAIF ID.
    fn naif_id(&self) -> i32 {
        self.body.naif_id()
    }

    /// Gravitational parameter of the perturbing body.
    ///
    /// Returns:
    ///     float: Gravitational parameter. Units: (m^3/s^2)
    fn gm(&self) -> f64 {
        self.body.gm()
    }

    /// The CentralBody equivalent of this perturber, if it is a physical
    /// body brahe knows how to treat as a frame/parameter center.
    ///
    /// Returns:
    ///     CentralBody | None: Central-body equivalent, or None for the
    ///         barycenter variants and Custom bodies.
    fn as_central_body(&self) -> Option<PyCentralBody> {
        self.body.as_central_body().map(|b| PyCentralBody { body: b })
    }

    /// The body-fixed reference frame a gravity field attached to this body
    /// is expressed in (e.g. ITRF for Earth, LFPA for the Moon, MCMF for
    /// Mars).
    ///
    /// Returns:
    ///     ReferenceFrame | None: Body-fixed frame, or None for the
    ///         barycenter variants, Custom bodies, and bodies without a
    ///         rotation model.
    fn body_fixed_frame(&self) -> Option<PyReferenceFrame> {
        self.body.body_fixed_frame().map(|f| PyReferenceFrame { frame: f })
    }

    fn __repr__(&self) -> String {
        format!("ThirdBody.{:?}", self.body)
    }
}

/// Configuration for a single third-body perturber.
///
/// Pairs a perturbing body with its ephemeris source and the gravity model
/// used for its perturbation (point-mass by default; a spherical-harmonic or
/// Earth-zonal field evaluates at the object's position relative to the
/// body).
///
/// Args:
///     body (ThirdBody): The perturbing body.
///     ephemeris_source (EphemerisSource, optional): Source for the body's
///         position. Defaults to ``EphemerisSource.DE440s``.
///     gravity (GravityConfiguration, optional): Gravity model for this
///         perturber. Defaults to point-mass.
///
/// Attributes:
///     body (ThirdBody): The perturbing body
///     ephemeris_source (EphemerisSource): Ephemeris source
///     gravity (GravityConfiguration): Gravity model for this perturber
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Point-mass Moon from DE440s
///     moon = bh.ThirdBodyConfiguration(bh.ThirdBody.MOON)
///
///     # Earth as an 8x8 spherical-harmonic perturber (for an EMB-centered
///     # propagation)
///     earth = bh.ThirdBodyConfiguration(
///         bh.ThirdBody.EARTH,
///         gravity=bh.GravityConfiguration.spherical_harmonic(degree=8, order=8),
///     )
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "ThirdBodyConfiguration")]
#[derive(Clone)]
pub struct PyThirdBodyConfiguration {
    pub config: propagators::ThirdBodyConfiguration,
}

#[pymethods]
impl PyThirdBodyConfiguration {
    /// Create a configuration for a single third-body perturber.
    ///
    /// Args:
    ///     body (ThirdBody): The perturbing body.
    ///     ephemeris_source (EphemerisSource, optional): Source for the body's
    ///         position. Defaults to ``EphemerisSource.DE440s``.
    ///     gravity (GravityConfiguration, optional): Gravity model for this
    ///         perturber. Defaults to point-mass.
    #[new]
    #[pyo3(signature = (body, ephemeris_source=None, gravity=None))]
    fn new(
        body: PyThirdBody,
        ephemeris_source: Option<PyEphemerisSource>,
        gravity: Option<PyGravityConfiguration>,
    ) -> Self {
        PyThirdBodyConfiguration {
            config: propagators::ThirdBodyConfiguration {
                body: body.body,
                ephemeris_source: ephemeris_source
                    .map(|s| s.into())
                    .unwrap_or(propagators::EphemerisSource::DE440s),
                gravity: gravity
                    .map(|g| g.config.clone())
                    .unwrap_or(propagators::GravityConfiguration::PointMass),
            },
        }
    }

    /// Get the perturbing body.
    #[getter]
    fn body(&self) -> PyThirdBody {
        PyThirdBody { body: self.config.body.clone() }
    }

    /// Set the perturbing body.
    #[setter]
    fn set_body(&mut self, body: PyThirdBody) {
        self.config.body = body.body;
    }

    /// Get the ephemeris source.
    #[getter]
    fn ephemeris_source(&self) -> PyEphemerisSource {
        match self.config.ephemeris_source {
            propagators::EphemerisSource::LowPrecision => PyEphemerisSource::LowPrecision,
            propagators::EphemerisSource::DE440s => PyEphemerisSource::DE440s,
            propagators::EphemerisSource::DE440 => PyEphemerisSource::DE440,
            propagators::EphemerisSource::SPK(spice::SPICEKernel::DE440) => {
                PyEphemerisSource::DE440
            }
            // The Python `EphemerisSource` enum exposes only LowPrecision/DE440s/DE440,
            // so it can never construct any other SPK kernel; map the remainder to
            // DE440s as the closest Python-visible source.
            propagators::EphemerisSource::SPK(_) => PyEphemerisSource::DE440s,
        }
    }

    /// Set the ephemeris source.
    #[setter]
    fn set_ephemeris_source(&mut self, source: PyEphemerisSource) {
        self.config.ephemeris_source = source.into();
    }

    /// Get the gravity model for this perturber.
    #[getter]
    fn gravity(&self) -> PyGravityConfiguration {
        PyGravityConfiguration { config: self.config.gravity.clone() }
    }

    /// Set the gravity model for this perturber.
    #[setter]
    fn set_gravity(&mut self, gravity: PyGravityConfiguration) {
        self.config.gravity = gravity.config.clone();
    }

    fn __repr__(&self) -> String {
        format!(
            "ThirdBodyConfiguration(body={:?}, ephemeris_source={:?}, gravity={:?})",
            self.config.body, self.config.ephemeris_source, self.config.gravity
        )
    }
}

/// A single third-body entry: either a full configuration or a bare body
/// (which takes DE440s ephemerides and point-mass gravity).
#[derive(FromPyObject)]
pub enum PyThirdBodyLike {
    Config(PyThirdBodyConfiguration),
    Body(PyThirdBody),
}

impl PyThirdBodyLike {
    fn into_config(self) -> propagators::ThirdBodyConfiguration {
        match self {
            PyThirdBodyLike::Config(c) => c.config,
            PyThirdBodyLike::Body(b) => b.body.into(),
        }
    }
}

/// Accepts a single entry or a list of entries.
#[derive(FromPyObject)]
pub enum PyThirdBodiesInput {
    One(PyThirdBodyLike),
    Many(Vec<PyThirdBodyLike>),
}

impl PyThirdBodiesInput {
    fn into_configs(self) -> Vec<propagators::ThirdBodyConfiguration> {
        match self {
            PyThirdBodiesInput::One(item) => vec![item.into_config()],
            PyThirdBodiesInput::Many(items) => {
                items.into_iter().map(|i| i.into_config()).collect()
            }
        }
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
#[pyclass(module = "brahe._brahe", from_py_object)]
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

    /// Enable or disable acceleration storage in trajectory.
    ///
    /// Args:
    ///     store (bool): Whether to store accelerations.
    ///
    /// Returns:
    ///     NumericalPropagationConfig: Self with updated setting.
    fn with_store_accelerations(mut slf: PyRefMut<'_, Self>, store: bool) -> PyRefMut<'_, Self> {
        slf.config.store_accelerations = store;
        slf
    }

    /// Set the interpolation method for trajectory queries.
    ///
    /// Args:
    ///     method (InterpolationMethod): The interpolation method to use.
    ///
    /// Returns:
    ///     NumericalPropagationConfig: Self with updated interpolation method.
    fn with_interpolation_method(
        mut slf: PyRefMut<'_, Self>,
        method: PyInterpolationMethod,
    ) -> PyRefMut<'_, Self> {
        slf.config.interpolation_method = method.method;
        slf
    }

    /// Get the integration method.
    #[getter]
    fn method(&self) -> PyIntegrationMethod {
        PyIntegrationMethod {
            method: self.config.method,
        }
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
        PyVariationalConfig {
            config: self.config.variational.clone(),
        }
    }

    /// Set variational configuration (STM/sensitivity settings).
    #[setter]
    fn set_variational(&mut self, value: PyVariationalConfig) {
        self.config.variational = value.config;
    }

    /// Get whether acceleration storage is enabled.
    #[getter]
    fn store_accelerations(&self) -> bool {
        self.config.store_accelerations
    }

    /// Set whether acceleration storage is enabled.
    #[setter]
    fn set_store_accelerations(&mut self, value: bool) {
        self.config.store_accelerations = value;
    }

    /// Get the interpolation method.
    #[getter]
    fn interpolation_method(&self) -> PyInterpolationMethod {
        PyInterpolationMethod {
            method: self.config.interpolation_method,
        }
    }

    /// Set the interpolation method.
    #[setter]
    fn set_interpolation_method(&mut self, value: PyInterpolationMethod) {
        self.config.interpolation_method = value.method;
    }

    /// Validate that the configuration is internally consistent.
    ///
    /// Called automatically by the propagator constructors. Can also be called
    /// directly to pre-flight a configuration before propagation.
    ///
    /// Raises:
    ///     RuntimeError: If interpolation_method is HermiteQuintic but
    ///         store_accelerations is False. The message names the two fixes
    ///         (enable acceleration storage, or pick a different method).
    fn validate(&self) -> PyResult<()> {
        self.config
            .validate()
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "NumericalPropagationConfig(method={:?}, abs_tol={}, rel_tol={})",
            self.config.method, self.config.integrator.abs_tol, self.config.integrator.rel_tol
        )
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
///     third_body (ThirdBodyConfiguration | ThirdBody | list, optional): Third-body
///         perturbation entries; a single entry or a list mixing ThirdBody and
///         ThirdBodyConfiguration values. Default is None (disabled).
///     relativity (bool, optional): Enable relativistic corrections. Default is False.
///     mass (ParameterSource, optional): Spacecraft mass source. Default is None.
///     frame_transform (FrameTransformationModel, optional): ECI-to-body-fixed rotation
///         used by every body-fixed force term. Defaults to ``FULL_EARTH_ROTATION``.
///
/// Attributes:
///     gravity (GravityConfiguration): Gravity model configuration
///     drag (DragConfiguration or None): Atmospheric drag configuration
///     srp (SolarRadiationPressureConfiguration or None): Solar radiation pressure configuration
///     third_body (list[ThirdBodyConfiguration] or None): Third-body perturbation entries
///     relativity (bool): Enable relativistic corrections
///     mass (ParameterSource or None): Spacecraft mass source
///     frame_transform (FrameTransformationModel): ECI-to-body-fixed rotation model
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
#[pyclass(module = "brahe._brahe", from_py_object)]
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
    ///     third_body (ThirdBodyConfiguration | ThirdBody | list, optional): Third-body
    ///         perturbation entries; a single entry or a list mixing ThirdBody and
    ///         ThirdBodyConfiguration values.
    ///     relativity (bool, optional): Enable relativistic corrections. Default is False.
    ///     mass (ParameterSource, optional): Spacecraft mass source.
    ///     tides (TidesConfiguration, optional): Solid Earth tides configuration.
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
    #[pyo3(signature = (gravity=None, drag=None, srp=None, third_body=None, relativity=false, mass=None, frame_transform=None, tides=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        gravity: Option<&PyGravityConfiguration>,
        drag: Option<&PyDragConfiguration>,
        srp: Option<&PySolarRadiationPressureConfiguration>,
        third_body: Option<PyThirdBodiesInput>,
        relativity: bool,
        mass: Option<&PyParameterSource>,
        frame_transform: Option<&PyFrameTransformationModel>,
        tides: Option<&PyTidesConfiguration>,
    ) -> Self {
        PyForceModelConfig {
            config: propagators::ForceModelConfig {
                central_body: propagators::CentralBody::default(),
                gravity: gravity
                    .map(|g| g.config.clone())
                    .unwrap_or(propagators::GravityConfiguration::PointMass),
                drag: drag.map(|d| d.config.clone()),
                srp: srp.map(|s| s.config.clone()),
                third_body: third_body.map(|t| t.into_configs()),
                relativity,
                mass: mass.map(|m| m.source.clone()),
                frame_transform: frame_transform.map(|f| f.model.clone()).unwrap_or_default(),
                tides: tides.map(|t| t.config.clone()),
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
    /// - Solid Earth tides with frequency-dependent corrections and the
    ///   solid pole tide
    /// - Ocean tides (FES2004, 30x30) with admittance and the ocean pole
    ///   tide; requires a one-time cached download of the IERS FES2004
    ///   coefficient file
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

    /// Create a force model configuration for a specific central body.
    ///
    /// Convenience constructor that fills in `frame_transform` with its default
    /// (`FrameTransformationModel.FULL_EARTH_ROTATION`) so callers only need to
    /// specify the options that vary per central body. Does not validate the
    /// resulting configuration -- call `validate()` to check that the chosen
    /// options are compatible with `central_body`.
    ///
    /// Args:
    ///     central_body (CentralBody): Body the orbit is propagated relative to.
    ///     gravity (GravityConfiguration): Gravity model configuration.
    ///     drag (DragConfiguration, optional): Atmospheric drag configuration.
    ///     srp (SolarRadiationPressureConfiguration, optional): Solar radiation pressure configuration.
    ///     third_body (ThirdBodyConfiguration | ThirdBody | list, optional): Third-body
    ///         perturbation entries; a single entry or a list mixing ThirdBody and
    ///         ThirdBodyConfiguration values.
    ///     relativity (bool, optional): Enable relativistic corrections. Default is False.
    ///     mass (ParameterSource, optional): Spacecraft mass source.
    ///
    /// Returns:
    ///     ForceModelConfig: A force model configuration for `central_body`.
    #[classmethod]
    #[pyo3(signature = (central_body, gravity, drag=None, srp=None, third_body=None, relativity=false, mass=None))]
    #[allow(clippy::too_many_arguments)]
    fn for_body(
        _cls: &Bound<'_, PyType>,
        central_body: &PyCentralBody,
        gravity: &PyGravityConfiguration,
        drag: Option<&PyDragConfiguration>,
        srp: Option<&PySolarRadiationPressureConfiguration>,
        third_body: Option<PyThirdBodiesInput>,
        relativity: bool,
        mass: Option<&PyParameterSource>,
    ) -> Self {
        PyForceModelConfig {
            config: propagators::ForceModelConfig::for_body(
                central_body.body.clone(),
                gravity.config.clone(),
                drag.map(|d| d.config.clone()),
                srp.map(|s| s.config.clone()),
                third_body.map(|t| t.into_configs()),
                relativity,
                mass.map(|m| m.source.clone()),
            ),
        }
    }

    /// Create a configuration suitable for propagation about the Moon.
    ///
    /// Uses 50x50 GRGM660PRIM lunar gravity, no drag, SRP occulted by the Moon
    /// and Earth, and Earth/Sun third-body perturbations (DE440s ephemerides).
    /// Requires parameter vector: [mass, _, _, srp_area, Cr]
    ///
    /// Returns:
    ///     ForceModelConfig: Configuration with the Moon as the central body.
    #[classmethod]
    fn lunar_default(_cls: &Bound<'_, PyType>) -> Self {
        PyForceModelConfig { config: propagators::ForceModelConfig::lunar_default() }
    }

    /// Create a configuration suitable for propagation about Mars.
    ///
    /// Uses 50x50 GMM-2B Mars gravity, exponential atmospheric drag, SRP
    /// occulted by Mars, and Sun third-body perturbations (DE440s ephemerides).
    /// Requires parameter vector: [mass, drag_area, Cd, srp_area, Cr]
    ///
    /// Returns:
    ///     ForceModelConfig: Configuration with Mars as the central body.
    #[classmethod]
    fn mars_default(_cls: &Bound<'_, PyType>) -> Self {
        PyForceModelConfig { config: propagators::ForceModelConfig::mars_default() }
    }

    /// Create a configuration suitable for cislunar propagation about the Earth-Moon barycenter.
    ///
    /// Uses point mass gravity (the barycenter has no mass of its own), no
    /// drag, SRP occulted by Earth and the Moon, and Earth/Moon/Sun third-body
    /// perturbations (DE440s ephemerides).
    /// Requires parameter vector: [mass, _, _, srp_area, Cr]
    ///
    /// Returns:
    ///     ForceModelConfig: Configuration with the Earth-Moon barycenter as the central body.
    #[classmethod]
    fn cislunar_default(_cls: &Bound<'_, PyType>) -> Self {
        PyForceModelConfig { config: propagators::ForceModelConfig::cislunar_default() }
    }

    /// Validate that this configuration's options are compatible with its central body.
    ///
    /// This method is called automatically at propagator construction; it may
    /// also be called explicitly ahead of time on a standalone configuration for
    /// early feedback.
    ///
    /// Returns:
    ///     None: If the configuration is internally consistent.
    ///
    /// Raises:
    ///     RuntimeError: If the configuration is internally inconsistent (naming both
    ///         the offending option and the central body).
    pub fn validate(&self) -> PyResult<()> {
        self.config.validate().map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Check if this configuration requires a parameter vector.
    pub fn requires_params(&self) -> bool {
        self.config.requires_params()
    }

    // =========================================================================
    // Field Accessors
    // =========================================================================

    /// Get the central body this configuration propagates relative to.
    #[getter]
    fn central_body(&self) -> PyCentralBody {
        PyCentralBody { body: self.config.central_body.clone() }
    }

    /// Set the central body this configuration propagates relative to.
    #[setter]
    fn set_central_body(&mut self, central_body: &PyCentralBody) {
        self.config.central_body = central_body.body.clone();
    }

    /// Get the gravity configuration.
    #[getter]
    fn gravity(&self) -> PyGravityConfiguration {
        PyGravityConfiguration {
            config: self.config.gravity.clone(),
        }
    }

    /// Set the gravity configuration.
    #[setter]
    fn set_gravity(&mut self, gravity: &PyGravityConfiguration) {
        self.config.gravity = gravity.config.clone();
    }

    /// Get the drag configuration (None if disabled).
    #[getter]
    fn drag(&self) -> Option<PyDragConfiguration> {
        self.config
            .drag
            .as_ref()
            .map(|d| PyDragConfiguration { config: d.clone() })
    }

    /// Set the drag configuration (None to disable).
    #[setter]
    fn set_drag(&mut self, drag: Option<&PyDragConfiguration>) {
        self.config.drag = drag.map(|d| d.config.clone());
    }

    /// Get the solar radiation pressure configuration (None if disabled).
    #[getter]
    fn srp(&self) -> Option<PySolarRadiationPressureConfiguration> {
        self.config
            .srp
            .as_ref()
            .map(|s| PySolarRadiationPressureConfiguration { config: s.clone() })
    }

    /// Set the solar radiation pressure configuration (None to disable).
    #[setter]
    fn set_srp(&mut self, srp: Option<&PySolarRadiationPressureConfiguration>) {
        self.config.srp = srp.map(|s| s.config.clone());
    }

    /// Get the third-body perturbation entries (None if disabled).
    #[getter]
    fn third_body(&self) -> Option<Vec<PyThirdBodyConfiguration>> {
        self.config.third_body.as_ref().map(|v| {
            v.iter()
                .map(|t| PyThirdBodyConfiguration { config: t.clone() })
                .collect()
        })
    }

    /// Set the third-body perturbation entries (None to disable). Accepts a
    /// single ThirdBody or ThirdBodyConfiguration, or a list mixing both.
    #[setter]
    fn set_third_body(&mut self, third_body: Option<PyThirdBodiesInput>) {
        self.config.third_body = third_body.map(|t| t.into_configs());
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
        self.config
            .mass
            .as_ref()
            .map(|m| PyParameterSource { source: m.clone() })
    }

    /// Set the mass parameter source (None to not track mass).
    #[setter]
    fn set_mass(&mut self, mass: Option<&PyParameterSource>) {
        self.config.mass = mass.map(|m| m.source.clone());
    }

    /// Get the ECI-to-body-fixed frame transformation model.
    #[getter]
    fn frame_transform(&self) -> PyFrameTransformationModel {
        PyFrameTransformationModel {
            model: self.config.frame_transform.clone(),
        }
    }

    /// Set the ECI-to-body-fixed frame transformation model.
    #[setter]
    fn set_frame_transform(&mut self, frame_transform: &PyFrameTransformationModel) {
        self.config.frame_transform = frame_transform.model.clone();
    }

    /// Get the tidal correction configuration (None if disabled).
    #[getter]
    fn tides(&self) -> Option<PyTidesConfiguration> {
        self.config
            .tides
            .as_ref()
            .map(|t| PyTidesConfiguration { config: t.clone() })
    }

    /// Set the tidal correction configuration (None to disable).
    #[setter]
    fn set_tides(&mut self, tides: Option<&PyTidesConfiguration>) {
        self.config.tides = tides.map(|t| t.config.clone());
    }

    fn __repr__(&self) -> String {
        format!(
            "ForceModelConfig(requires_params={}, frame_transform={:?})",
            self.config.requires_params(),
            self.config.frame_transform,
        )
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
/// - Solid Earth tides
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
    /// Holds the original Python exception raised inside the additional-dynamics
    /// or control-input trampoline, so a driven propagation can re-raise it
    /// verbatim instead of the wrapped BraheError message.
    err_slot: PyErrSlot,
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

        let params_vec =
            params.map(|p| nalgebra::DVector::from_column_slice(p.as_slice().unwrap()));

        let cov_matrix = if let Some(cov) = initial_covariance {
            let cov_shape = cov.shape();
            if cov_shape[0] != 6 || cov_shape[1] != 6 {
                return Err(exceptions::PyValueError::new_err(
                    "Initial covariance must be a 6x6 matrix",
                ));
            }
            let cov_data: Vec<f64> = cov.as_slice()?.to_vec();
            Some(nalgebra::DMatrix::from_row_slice(6, 6, &cov_data))
        } else {
            None
        };

        // Slot shared with the Python trampolines below. A callback that raises
        // records its exception here and surfaces a BraheError to the core; the
        // driving step/propagate method re-raises the stashed exception.
        let err_slot: PyErrSlot = Arc::new(Mutex::new(None));

        // Wrap additional_dynamics Python callable if provided
        let additional_dynamics_fn: Option<brahe::integrators::traits::DStateDynamics> =
            additional_dynamics.map(|dyn_py| {
                let dyn_py = dyn_py.clone_ref(py);
                let err_slot = err_slot.clone();
                Box::new(
                    move |t: f64,
                          x: &nalgebra::DVector<f64>,
                          p: Option<&nalgebra::DVector<f64>>| {
                        Python::attach(|py| {
                            let x_np = x.as_slice().to_pyarray(py);
                            let p_np: Option<Bound<'_, PyArray<f64, Ix1>>> =
                                p.map(|pv| pv.as_slice().to_pyarray(py).to_owned());

                            let result = match p_np {
                                Some(params_arr) => dyn_py.call1(py, (t, x_np, params_arr)),
                                None => dyn_py.call1(py, (t, x_np, py.None())),
                            };

                            let res = result.map_err(|e| stash_callback_err(&err_slot, e))?;
                            let res_arr: PyReadonlyArray1<f64> =
                                res.extract(py).map_err(|e| stash_callback_err(&err_slot, PyErr::from(e)))?;
                            let res_slice = res_arr.as_slice().map_err(|e| {
                                RustBraheError::Error(format!(
                                    "callback returned non-contiguous array: {e}"
                                ))
                            })?;
                            Ok(nalgebra::DVector::from_column_slice(res_slice))
                        })
                    },
                ) as brahe::integrators::traits::DStateDynamics
            });

        // Wrap control_input Python callable if provided
        let control_input_fn: brahe::integrators::traits::DControlInput =
            control_input.map(|ctrl_py| {
                let ctrl_py = ctrl_py.clone_ref(py);
                let err_slot = err_slot.clone();
                Box::new(
                    move |t: f64,
                          x: &nalgebra::DVector<f64>,
                          p: Option<&nalgebra::DVector<f64>>| {
                        Python::attach(|py| {
                            let x_np = x.as_slice().to_pyarray(py);
                            let p_np: Option<Bound<'_, PyArray<f64, Ix1>>> =
                                p.map(|pv| pv.as_slice().to_pyarray(py).to_owned());

                            let result = match p_np {
                                Some(params_arr) => ctrl_py.call1(py, (t, x_np, params_arr)),
                                None => ctrl_py.call1(py, (t, x_np, py.None())),
                            };

                            let res = result.map_err(|e| stash_callback_err(&err_slot, e))?;
                            let res_arr: PyReadonlyArray1<f64> =
                                res.extract(py).map_err(|e| stash_callback_err(&err_slot, PyErr::from(e)))?;
                            let res_slice = res_arr.as_slice().map_err(|e| {
                                RustBraheError::Error(format!(
                                    "callback returned non-contiguous array: {e}"
                                ))
                            })?;
                            Ok(nalgebra::DVector::from_column_slice(res_slice))
                        })
                    },
                )
                    as Box<
                        dyn Fn(
                                f64,
                                &nalgebra::DVector<f64>,
                                Option<&nalgebra::DVector<f64>>,
                            )
                                -> Result<nalgebra::DVector<f64>, brahe::utils::BraheError>
                            + Send
                            + Sync,
                    >
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
        )
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyNumericalOrbitPropagator {
            propagator: prop,
            err_slot,
        })
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
                "State vector must have at least 6 elements",
            ));
        }

        let state_vec = nalgebra::DVector::from_column_slice(state_slice);
        let params_vec =
            params.map(|p| nalgebra::DVector::from_column_slice(p.as_slice().unwrap()));

        let fc = force_config.map(|c| c.config.clone()).unwrap_or_default();

        let prop = propagators::DNumericalOrbitPropagator::new(
            epoch.obj,
            state_vec,
            propagators::NumericalPropagationConfig::default(),
            fc.clone(),
            params_vec,
            None,
            None,
            None,
        )
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyNumericalOrbitPropagator {
            propagator: prop,
            err_slot: Arc::new(Mutex::new(None)),
        })
    }

    // =========================================================================
    // DStatePropagator trait methods
    // =========================================================================

    /// Get current epoch.
    #[pyo3(text_signature = "()")]
    pub fn current_epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.propagator.current_epoch(),
        }
    }

    /// Get initial epoch.
    #[getter]
    pub fn initial_epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.propagator.initial_epoch(),
        }
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
    ///
    /// Raises:
    ///     Exception: Propagates the original exception raised by an
    ///         additional-dynamics or control-input callback, or a BraheError
    ///         if propagation fails.
    #[pyo3(text_signature = "()")]
    pub fn step(&mut self) -> PyResult<()> {
        DStatePropagator::step(&mut self.propagator)
            .map_err(|e| raise_callback_err(&self.err_slot, e))
    }

    /// Step forward by a specified time duration.
    ///
    /// Args:
    ///     step_size (float): Time step in seconds.
    ///
    /// Raises:
    ///     Exception: Propagates the original exception raised by an
    ///         additional-dynamics or control-input callback, or a BraheError
    ///         if propagation fails.
    #[pyo3(text_signature = "(step_size)")]
    pub fn step_by(&mut self, step_size: f64) -> PyResult<()> {
        DStatePropagator::step_by(&mut self.propagator, step_size)
            .map_err(|e| raise_callback_err(&self.err_slot, e))
    }

    /// Step past a specified target epoch.
    ///
    /// Args:
    ///     target_epoch (Epoch): The epoch to step past.
    ///
    /// Raises:
    ///     Exception: Propagates the original exception raised by an
    ///         additional-dynamics or control-input callback, or a BraheError
    ///         if propagation fails.
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn step_past(&mut self, target_epoch: &PyEpoch) -> PyResult<()> {
        DStatePropagator::step_past(&mut self.propagator, target_epoch.obj)
            .map_err(|e| raise_callback_err(&self.err_slot, e))
    }

    /// Propagate forward by specified number of steps.
    ///
    /// Args:
    ///     num_steps (int): Number of steps to take.
    ///
    /// Raises:
    ///     Exception: Propagates the original exception raised by an
    ///         additional-dynamics or control-input callback, or a BraheError
    ///         if propagation fails.
    #[pyo3(text_signature = "(num_steps)")]
    pub fn propagate_steps(&mut self, num_steps: usize) -> PyResult<()> {
        DStatePropagator::propagate_steps(&mut self.propagator, num_steps)
            .map_err(|e| raise_callback_err(&self.err_slot, e))
    }

    /// Propagate to a specific target epoch.
    ///
    /// Args:
    ///     target_epoch (Epoch): The epoch to propagate to.
    ///
    /// Raises:
    ///     Exception: Propagates the original exception raised by an
    ///         additional-dynamics or control-input callback, or a BraheError
    ///         if propagation fails.
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn propagate_to(&mut self, target_epoch: &PyEpoch) -> PyResult<()> {
        DStatePropagator::propagate_to(&mut self.propagator, target_epoch.obj)
            .map_err(|e| raise_callback_err(&self.err_slot, e))
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
    pub fn state<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
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
    pub fn state_eci<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_eci(&self.propagator, epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Get the state at the given epoch in the central body's body-centered
    /// inertial (BCI) frame (`GCRF` for an `Earth`-centered propagator, `LCI`
    /// for a `Moon`-centered one, `MCI` for `Mars`, etc.).
    ///
    /// This is the state the integrator actually propagates: no central-body
    /// offset or axis rotation is applied. `state_eci` always returns an
    /// Earth-centered state regardless of the propagator's central body; use
    /// this method to get the state in its native frame instead, which avoids
    /// an Earth round trip for non-Earth propagators.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in the central body's
    ///     inertial frame.
    ///
    /// Raises:
    ///     RuntimeError: If `epoch` is outside the propagator's stored trajectory
    ///         and does not match the current epoch.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_bci<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self
            .propagator
            .state_bci(epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Get the state at the given epoch in the central body's body-centered
    /// body-fixed (BCBF) frame (`ITRF` for an `Earth`-centered propagator,
    /// `LFPA` for a `Moon`-centered one, `MCMF` for `Mars`, the configured
    /// fixed frame for a custom body).
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in the central body's
    ///     body-fixed frame.
    ///
    /// Raises:
    ///     RuntimeError: If `epoch` is outside the propagator's stored trajectory,
    ///         or the central body has no body-fixed frame (`EMB`/`SSB`
    ///         barycenters, custom bodies without a configured frame).
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_bcbf<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self
            .propagator
            .state_bcbf(epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute the state at the given epoch in an arbitrary reference frame.
    ///
    /// Converts directly from this propagator's own central body's inertial
    /// frame, avoiding an unnecessary Earth round trip for a lunar/Martian
    /// propagator.
    ///
    /// Args:
    ///     frame (ReferenceFrame): Reference frame to express the state in.
    ///     epoch (Epoch): Target epoch for state computation.
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in `frame`.
    ///
    /// Raises:
    ///     RuntimeError: If the state cannot be computed or the frame conversion fails.
    #[pyo3(text_signature = "(frame, epoch)")]
    pub fn state_in_frame<'a>(&self, py: Python<'a>, frame: &PyReferenceFrame, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self
            .propagator
            .state_in_frame(frame.frame, epoch.obj)
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
    pub fn state_ecef<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_ecef(&self.propagator, epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in GCRF coordinates.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_gcrf<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_gcrf(&self.propagator, epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in ITRF coordinates.
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_itrf<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
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
    pub fn state_eme2000<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = DOrbitStateProvider::state_eme2000(&self.propagator, epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute states at multiple epochs in ECI coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of ECI state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_eci<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_eci(&self.propagator, &epoch_vec)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in ECEF coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of ECEF state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_ecef<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_ecef(&self.propagator, &epoch_vec)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in GCRF coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of GCRF state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_gcrf<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_gcrf(&self.propagator, &epoch_vec)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in ITRF coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of ITRF state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_itrf<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_itrf(&self.propagator, &epoch_vec)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in EME2000 coordinates.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of EME2000 state vectors.
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_eme2000<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_eme2000(&self.propagator, &epoch_vec)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in the propagator's central body's
    /// body-centered inertial (BCI) frame.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of BCI state vectors [x, y, z, vx, vy, vz] (meters, m/s).
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_bci<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_bci(&self.propagator, &epoch_vec)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs in the propagator's central body's
    /// body-centered body-fixed (BCBF) frame.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of BCBF state vectors [x, y, z, vx, vy, vz] (meters, m/s).
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_bcbf<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DOrbitStateProvider::states_bcbf(&self.propagator, &epoch_vec)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states at multiple epochs expressed in an arbitrary
    /// reference frame, converting from the propagator's native
    /// central-body frame.
    ///
    /// Args:
    ///     frame (ReferenceFrame): The reference frame to express the states in.
    ///     epochs (list[Epoch]): List of epochs for state computation.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of state vectors [x, y, z, vx, vy, vz] in `frame` (meters, m/s).
    #[pyo3(text_signature = "(frame, epochs)")]
    pub fn states_in_frame<'a>(
        &self,
        py: Python<'a>,
        frame: &PyReferenceFrame,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states =
            DOrbitStateProvider::states_in_frame(&self.propagator, frame.frame, &epoch_vec)
                .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute state as osculating Keplerian elements at a specific epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch.
    ///     angle_format (AngleFormat): Format for angular elements.
    ///
    /// Returns:
    ///     numpy.ndarray: Osculating Keplerian elements [a, e, i, Ω, ω, M].
    #[pyo3(text_signature = "(epoch, angle_format)")]
    pub fn state_koe_osc<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state =
            DOrbitStateProvider::state_koe_osc(&self.propagator, epoch.obj, angle_format.value)
                .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state as mean Keplerian elements at a specific epoch.
    ///
    /// Mean elements are orbit-averaged elements that remove short-period and
    /// long-period J2 perturbations using first-order Brouwer-Lyddane theory.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch.
    ///     angle_format (AngleFormat): Format for angular elements.
    ///
    /// Returns:
    ///     numpy.ndarray: Mean Keplerian elements [a, e, i, Ω, ω, M].
    #[pyo3(text_signature = "(epoch, angle_format)")]
    pub fn state_koe_mean<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state =
            DOrbitStateProvider::state_koe_mean(&self.propagator, epoch.obj, angle_format.value)
                .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute states as osculating Keplerian elements at multiple epochs.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of target epochs.
    ///     angle_format (AngleFormat): Format for angular elements.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of osculating Keplerian elements [a, e, i, Ω, ω, M].
    #[pyo3(text_signature = "(epochs, angle_format)")]
    pub fn states_koe_osc<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states =
            DOrbitStateProvider::states_koe_osc(&self.propagator, &epoch_vec, angle_format.value)
                .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    /// Compute states as mean Keplerian elements at multiple epochs.
    ///
    /// Mean elements are orbit-averaged elements that remove short-period and
    /// long-period J2 perturbations using first-order Brouwer-Lyddane theory.
    ///
    /// Args:
    ///     epochs (list[Epoch]): List of target epochs.
    ///     angle_format (AngleFormat): Format for angular elements.
    ///
    /// Returns:
    ///     list[numpy.ndarray]: List of mean Keplerian elements [a, e, i, Ω, ω, M].
    #[pyo3(text_signature = "(epochs, angle_format)")]
    pub fn states_koe_mean<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states =
            DOrbitStateProvider::states_koe_mean(&self.propagator, &epoch_vec, angle_format.value)
                .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(states
            .iter()
            .map(|s: &Vector6<f64>| s.as_slice().to_pyarray(py).to_owned())
            .collect())
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
        PyOrbitalTrajectory {
            trajectory: self.propagator.trajectory().clone(),
        }
    }

    /// Get current STM (State Transition Matrix) if enabled.
    ///
    /// Returns:
    ///     numpy.ndarray or None: The current STM (n x n matrix), or None if STM not enabled.
    #[pyo3(text_signature = "()")]
    pub fn stm<'a>(&self, py: Python<'a>) -> Option<Bound<'a, PyArray<f64, Ix2>>> {
        self.propagator.stm().map(|stm| {
            let n = stm.nrows();
            let flat: Vec<f64> = (0..n)
                .flat_map(|i| (0..n).map(move |j| stm[(i, j)]))
                .collect();
            flat.into_pyarray(py).reshape([n, n]).unwrap()
        })
    }

    /// Get the parameter vector supplied at construction.
    ///
    /// These are the force-model / consider parameters passed to the dynamics,
    /// control input, and (via the estimation filters) measurement models.
    ///
    /// Returns:
    ///     numpy.ndarray or None: Parameter vector, or None if no parameters were provided.
    #[pyo3(text_signature = "()")]
    pub fn params<'a>(&self, py: Python<'a>) -> Option<Bound<'a, PyArray<f64, Ix1>>> {
        self.propagator.params().map(|p| {
            let flat: Vec<f64> = p.iter().copied().collect();
            flat.into_pyarray(py)
        })
    }

    /// Disable STM (variational equation) propagation.
    ///
    /// Providing an initial covariance at construction enables STM propagation
    /// automatically. When the STM is not needed, disabling it removes the cost
    /// of integrating the variational equations at every step. Covariance
    /// propagation requires the STM, so any covariance held by the propagator
    /// is cleared: ``stm()`` and ``current_covariance()`` return None afterwards.
    /// Sensitivity propagation, if enabled, is unaffected. No-op if STM
    /// propagation is not enabled.
    #[pyo3(text_signature = "()")]
    pub fn disable_stm_propagation(&mut self) {
        self.propagator.disable_stm_propagation();
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
            let flat: Vec<f64> = (0..n)
                .flat_map(|i| (0..p).map(move |j| sens[(i, j)]))
                .collect();
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
    pub fn covariance<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix2>>> {
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
    pub fn with_uuid(
        mut slf: PyRefMut<'_, Self>,
        uuid_str: String,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let uuid = uuid::Uuid::parse_str(&uuid_str)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        slf.propagator.uuid = Some(uuid);
        Ok(slf)
    }

    /// Generate a new UUID, set it, and return self.
    pub fn with_new_uuid(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.propagator.uuid = Some(uuid::Uuid::now_v7());
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
        format!(
            "NumericalOrbitPropagator(epoch={:?}, state_dim={})",
            DStatePropagator::current_epoch(&self.propagator),
            DStatePropagator::state_dim(&self.propagator)
        )
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
                "TimeEvent has already been consumed",
            ));
        }

        if let Ok(mut value_event) = event.extract::<PyRefMut<PyValueEvent>>() {
            if let Some(inner) = value_event.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "ValueEvent has already been consumed",
            ));
        }

        if let Ok(mut binary_event) = event.extract::<PyRefMut<PyBinaryEvent>>() {
            if let Some(inner) = binary_event.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "BinaryEvent has already been consumed",
            ));
        }

        if let Ok(mut altitude_event) = event.extract::<PyRefMut<PyAltitudeEvent>>() {
            if let Some(inner) = altitude_event.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "AltitudeEvent has already been consumed",
            ));
        }

        // Premade Orbital Element Events
        if let Ok(mut e) = event.extract::<PyRefMut<PySemiMajorAxisEvent>>() {
            if let Some(inner) = e.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "SemiMajorAxisEvent has already been consumed",
            ));
        }
        if let Ok(mut e) = event.extract::<PyRefMut<PyEccentricityEvent>>() {
            if let Some(inner) = e.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "EccentricityEvent has already been consumed",
            ));
        }
        if let Ok(mut e) = event.extract::<PyRefMut<PyInclinationEvent>>() {
            if let Some(inner) = e.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "InclinationEvent has already been consumed",
            ));
        }
        if let Ok(mut e) = event.extract::<PyRefMut<PyArgumentOfPerigeeEvent>>() {
            if let Some(inner) = e.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "ArgumentOfPerigeeEvent has already been consumed",
            ));
        }
        if let Ok(mut e) = event.extract::<PyRefMut<PyMeanAnomalyEvent>>() {
            if let Some(inner) = e.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "MeanAnomalyEvent has already been consumed",
            ));
        }
        if let Ok(mut e) = event.extract::<PyRefMut<PyEccentricAnomalyEvent>>() {
            if let Some(inner) = e.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "EccentricAnomalyEvent has already been consumed",
            ));
        }
        if let Ok(mut e) = event.extract::<PyRefMut<PyTrueAnomalyEvent>>() {
            if let Some(inner) = e.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "TrueAnomalyEvent has already been consumed",
            ));
        }
        if let Ok(mut e) = event.extract::<PyRefMut<PyArgumentOfLatitudeEvent>>() {
            if let Some(inner) = e.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "ArgumentOfLatitudeEvent has already been consumed",
            ));
        }

        // Premade Node Crossing Events
        if let Ok(mut e) = event.extract::<PyRefMut<PyAscendingNodeEvent>>() {
            if let Some(inner) = e.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "AscendingNodeEvent has already been consumed",
            ));
        }
        if let Ok(mut e) = event.extract::<PyRefMut<PyDescendingNodeEvent>>() {
            if let Some(inner) = e.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "DescendingNodeEvent has already been consumed",
            ));
        }

        // Premade State-Derived Events
        if let Ok(mut e) = event.extract::<PyRefMut<PySpeedEvent>>() {
            if let Some(inner) = e.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "SpeedEvent has already been consumed",
            ));
        }
        if let Ok(mut e) = event.extract::<PyRefMut<PyLongitudeEvent>>() {
            if let Some(inner) = e.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "LongitudeEvent has already been consumed",
            ));
        }
        if let Ok(mut e) = event.extract::<PyRefMut<PyLatitudeEvent>>() {
            if let Some(inner) = e.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "LatitudeEvent has already been consumed",
            ));
        }

        // Premade Eclipse/Shadow Events
        if let Ok(mut e) = event.extract::<PyRefMut<PyUmbraEvent>>() {
            if let Some(inner) = e.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "UmbraEvent has already been consumed",
            ));
        }
        if let Ok(mut e) = event.extract::<PyRefMut<PyPenumbraEvent>>() {
            if let Some(inner) = e.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "PenumbraEvent has already been consumed",
            ));
        }
        if let Ok(mut e) = event.extract::<PyRefMut<PyEclipseEvent>>() {
            if let Some(inner) = e.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "EclipseEvent has already been consumed",
            ));
        }
        if let Ok(mut e) = event.extract::<PyRefMut<PySunlitEvent>>() {
            if let Some(inner) = e.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "SunlitEvent has already been consumed",
            ));
        }

        Err(exceptions::PyTypeError::new_err(
            "Expected event detector type (TimeEvent, ValueEvent, BinaryEvent, AltitudeEvent, or premade event)",
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
        PyTrajectoryMode {
            mode: self.propagator.trajectory_mode(),
        }
    }

    /// Get STM (State Transition Matrix) at a specific epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Target epoch for STM query.
    ///
    /// Returns:
    ///     numpy.ndarray or None: The STM at the requested epoch, or None if STM not enabled.
    #[pyo3(text_signature = "(epoch)")]
    pub fn stm_at<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> Option<Bound<'a, PyArray<f64, Ix2>>> {
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
    pub fn sensitivity_at<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> Option<Bound<'a, PyArray<f64, Ix2>>> {
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
    pub fn covariance_gcrf<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix2>>> {
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
    pub fn covariance_rtn<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix2>>> {
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
        PyInterpolationMethod {
            method: InterpolationConfig::get_interpolation_method(&self.propagator),
        }
    }

    /// Set interpolation method for covariance queries.
    ///
    /// Args:
    ///     method (CovarianceInterpolationMethod): Interpolation method for covariance.
    #[pyo3(text_signature = "(method)")]
    pub fn set_covariance_interpolation_method(
        &mut self,
        method: &PyCovarianceInterpolationMethod,
    ) {
        CovarianceInterpolationConfig::set_covariance_interpolation_method(
            &mut self.propagator,
            method.method,
        );
    }

    /// Get current interpolation method for covariance queries.
    ///
    /// Returns:
    ///     CovarianceInterpolationMethod: Current covariance interpolation method.
    #[pyo3(text_signature = "()")]
    pub fn get_covariance_interpolation_method(&self) -> PyCovarianceInterpolationMethod {
        PyCovarianceInterpolationMethod {
            method: CovarianceInterpolationConfig::get_covariance_interpolation_method(
                &self.propagator,
            ),
        }
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
#[pyclass(module = "brahe._brahe", from_py_object)]
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
    /// Holds the original Python exception raised inside the dynamics or
    /// control-input trampoline, so a driven propagation can re-raise it
    /// verbatim instead of the wrapped BraheError message.
    err_slot: PyErrSlot,
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
    ///     control_input (callable or None): Optional control input function.
    ///                                       Signature: f(t, state, params) -> control_perturbation.
    ///                                       Should return an N-dimensional array matching state dimension.
    ///
    /// Returns:
    ///     NumericalPropagator: New propagator instance.
    #[new]
    #[pyo3(signature = (epoch, state, dynamics, propagation_config, params=None, initial_covariance=None, control_input=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        py: Python<'_>,
        epoch: &PyEpoch,
        state: PyReadonlyArray1<f64>,
        dynamics: Py<PyAny>,
        propagation_config: &PyNumericalPropagationConfig,
        params: Option<PyReadonlyArray1<f64>>,
        initial_covariance: Option<PyReadonlyArray2<f64>>,
        control_input: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let state_vec = nalgebra::DVector::from_column_slice(state.as_slice()?);
        let state_dim = state_vec.len();

        let params_vec =
            params.map(|p| nalgebra::DVector::from_column_slice(p.as_slice().unwrap()));

        let cov_matrix = if let Some(cov) = initial_covariance {
            let cov_shape = cov.shape();
            if cov_shape[0] != state_dim || cov_shape[1] != state_dim {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Initial covariance must be a {}x{} matrix",
                    state_dim, state_dim
                )));
            }
            let cov_data: Vec<f64> = cov.as_slice()?.to_vec();
            Some(nalgebra::DMatrix::from_row_slice(
                state_dim, state_dim, &cov_data,
            ))
        } else {
            None
        };

        // Slot shared with the Python trampolines below. A callback that raises
        // records its exception here and surfaces a BraheError to the core; the
        // driving step/propagate method re-raises the stashed exception.
        let err_slot: PyErrSlot = Arc::new(Mutex::new(None));

        // Create a wrapper that calls the Python dynamics function
        let dynamics_py = dynamics.clone_ref(py);
        let dynamics_fn: brahe::integrators::traits::DStateDynamics = {
            let err_slot = err_slot.clone();
            Box::new(
                move |t: f64, x: &nalgebra::DVector<f64>, p: Option<&nalgebra::DVector<f64>>| {
                    Python::attach(|py| {
                        // Convert state to numpy array
                        let x_np = x.as_slice().to_pyarray(py);

                        // Convert params to numpy array or None
                        let p_np: Option<Bound<'_, PyArray<f64, Ix1>>> =
                            p.map(|pv| pv.as_slice().to_pyarray(py).to_owned());

                        // Call Python function
                        let result = match p_np {
                            Some(params_arr) => dynamics_py.call1(py, (t, x_np, params_arr)),
                            None => dynamics_py.call1(py, (t, x_np, py.None())),
                        };

                        let res = result.map_err(|e| stash_callback_err(&err_slot, e))?;
                        let res_arr: PyReadonlyArray1<f64> =
                            res.extract(py).map_err(|e| stash_callback_err(&err_slot, PyErr::from(e)))?;
                        let res_slice = res_arr.as_slice().map_err(|e| {
                            RustBraheError::Error(format!(
                                "callback returned non-contiguous array: {e}"
                            ))
                        })?;
                        Ok(nalgebra::DVector::from_column_slice(res_slice))
                    })
                },
            )
        };

        // Wrap control_input Python callable if provided
        let control_input_fn: brahe::integrators::traits::DControlInput =
            control_input.map(|ctrl_py| {
                let ctrl_py = ctrl_py.clone_ref(py);
                let err_slot = err_slot.clone();
                Box::new(
                    move |t: f64,
                          x: &nalgebra::DVector<f64>,
                          p: Option<&nalgebra::DVector<f64>>| {
                        Python::attach(|py| {
                            let x_np = x.as_slice().to_pyarray(py);
                            let p_np: Option<Bound<'_, PyArray<f64, Ix1>>> =
                                p.map(|pv| pv.as_slice().to_pyarray(py).to_owned());

                            let result = match p_np {
                                Some(params_arr) => ctrl_py.call1(py, (t, x_np, params_arr)),
                                None => ctrl_py.call1(py, (t, x_np, py.None())),
                            };

                            let res = result.map_err(|e| stash_callback_err(&err_slot, e))?;
                            let res_arr: PyReadonlyArray1<f64> =
                                res.extract(py).map_err(|e| stash_callback_err(&err_slot, PyErr::from(e)))?;
                            let res_slice = res_arr.as_slice().map_err(|e| {
                                RustBraheError::Error(format!(
                                    "callback returned non-contiguous array: {e}"
                                ))
                            })?;
                            Ok(nalgebra::DVector::from_column_slice(res_slice))
                        })
                    },
                )
                    as Box<
                        dyn Fn(
                                f64,
                                &nalgebra::DVector<f64>,
                                Option<&nalgebra::DVector<f64>>,
                            )
                                -> Result<nalgebra::DVector<f64>, brahe::utils::BraheError>
                            + Send
                            + Sync,
                    >
            });

        let prop = propagators::DNumericalPropagator::new(
            epoch.obj,
            state_vec,
            dynamics_fn,
            propagation_config.config.clone(),
            params_vec,
            control_input_fn,
            cov_matrix,
        )
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyNumericalPropagator {
            propagator: prop,
            err_slot,
        })
    }

    // =========================================================================
    // DStatePropagator trait methods
    // =========================================================================

    /// Get current epoch.
    #[pyo3(text_signature = "()")]
    pub fn current_epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: DStatePropagator::current_epoch(&self.propagator),
        }
    }

    /// Get initial epoch.
    #[getter]
    pub fn initial_epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: DStatePropagator::initial_epoch(&self.propagator),
        }
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
    ///
    /// Raises:
    ///     Exception: Propagates the original exception raised by the dynamics
    ///         or control-input callback, or a BraheError if propagation fails.
    #[pyo3(text_signature = "()")]
    pub fn step(&mut self) -> PyResult<()> {
        DStatePropagator::step(&mut self.propagator)
            .map_err(|e| raise_callback_err(&self.err_slot, e))
    }

    /// Step forward by a specified time duration.
    ///
    /// Raises:
    ///     Exception: Propagates the original exception raised by the dynamics
    ///         or control-input callback, or a BraheError if propagation fails.
    #[pyo3(text_signature = "(step_size)")]
    pub fn step_by(&mut self, step_size: f64) -> PyResult<()> {
        DStatePropagator::step_by(&mut self.propagator, step_size)
            .map_err(|e| raise_callback_err(&self.err_slot, e))
    }

    /// Step past a specified target epoch.
    ///
    /// Raises:
    ///     Exception: Propagates the original exception raised by the dynamics
    ///         or control-input callback, or a BraheError if propagation fails.
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn step_past(&mut self, target_epoch: &PyEpoch) -> PyResult<()> {
        DStatePropagator::step_past(&mut self.propagator, target_epoch.obj)
            .map_err(|e| raise_callback_err(&self.err_slot, e))
    }

    /// Propagate forward by specified number of steps.
    ///
    /// Raises:
    ///     Exception: Propagates the original exception raised by the dynamics
    ///         or control-input callback, or a BraheError if propagation fails.
    #[pyo3(text_signature = "(num_steps)")]
    pub fn propagate_steps(&mut self, num_steps: usize) -> PyResult<()> {
        DStatePropagator::propagate_steps(&mut self.propagator, num_steps)
            .map_err(|e| raise_callback_err(&self.err_slot, e))
    }

    /// Propagate to a specific target epoch.
    ///
    /// Raises:
    ///     Exception: Propagates the original exception raised by the dynamics
    ///         or control-input callback, or a BraheError if propagation fails.
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn propagate_to(&mut self, target_epoch: &PyEpoch) -> PyResult<()> {
        DStatePropagator::propagate_to(&mut self.propagator, target_epoch.obj)
            .map_err(|e| raise_callback_err(&self.err_slot, e))
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
    pub fn state<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = DStateProvider::state(&self.propagator, epoch.obj)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
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
    pub fn states<'a>(
        &self,
        py: Python<'a>,
        epochs: Vec<PyRef<PyEpoch>>,
    ) -> PyResult<Vec<Bound<'a, PyArray<f64, Ix1>>>> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let states = DStateProvider::states(&self.propagator, &epoch_vec)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        Ok(states
            .iter()
            .map(|s| s.as_slice().to_pyarray(py).to_owned())
            .collect())
    }

    // =========================================================================
    // Trajectory and variational methods
    // =========================================================================

    /// Get accumulated trajectory.
    #[getter]
    pub fn trajectory(&self) -> PyTrajectory {
        PyTrajectory {
            trajectory: self.propagator.trajectory().clone(),
        }
    }

    /// Get current STM if enabled.
    #[pyo3(text_signature = "()")]
    pub fn stm<'a>(&self, py: Python<'a>) -> Option<Bound<'a, PyArray<f64, Ix2>>> {
        self.propagator.stm().map(|stm| {
            let n = stm.nrows();
            let flat: Vec<f64> = (0..n)
                .flat_map(|i| (0..n).map(move |j| stm[(i, j)]))
                .collect();
            flat.into_pyarray(py).reshape([n, n]).unwrap()
        })
    }

    /// Get the parameter vector supplied at construction.
    ///
    /// These are the force-model / consider parameters passed to the dynamics,
    /// control input, and (via the estimation filters) measurement models.
    ///
    /// Returns:
    ///     numpy.ndarray or None: Parameter vector, or None if no parameters were provided.
    #[pyo3(text_signature = "()")]
    pub fn params<'a>(&self, py: Python<'a>) -> Option<Bound<'a, PyArray<f64, Ix1>>> {
        self.propagator.params().map(|p| {
            let flat: Vec<f64> = p.iter().copied().collect();
            flat.into_pyarray(py)
        })
    }

    /// Disable STM (variational equation) propagation.
    ///
    /// Providing an initial covariance at construction enables STM propagation
    /// automatically. When the STM is not needed, disabling it removes the cost
    /// of integrating the variational equations at every step. Covariance
    /// propagation requires the STM, so any covariance held by the propagator
    /// is cleared: ``stm()`` and ``current_covariance()`` return None afterwards.
    /// Sensitivity propagation, if enabled, is unaffected. No-op if STM
    /// propagation is not enabled.
    #[pyo3(text_signature = "()")]
    pub fn disable_stm_propagation(&mut self) {
        self.propagator.disable_stm_propagation();
    }

    /// Get current sensitivity matrix if enabled.
    #[pyo3(text_signature = "()")]
    pub fn sensitivity<'a>(&self, py: Python<'a>) -> Option<Bound<'a, PyArray<f64, Ix2>>> {
        self.propagator.sensitivity().map(|sens| {
            let n = sens.nrows();
            let p = sens.ncols();
            let flat: Vec<f64> = (0..n)
                .flat_map(|i| (0..p).map(move |j| sens[(i, j)]))
                .collect();
            flat.into_pyarray(py).reshape([n, p]).unwrap()
        })
    }

    /// Get covariance at a specific epoch.
    #[pyo3(text_signature = "(epoch)")]
    pub fn covariance<'a>(
        &self,
        py: Python<'a>,
        epoch: &PyEpoch,
    ) -> PyResult<Bound<'a, PyArray<f64, Ix2>>> {
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
    pub fn with_uuid(
        mut slf: PyRefMut<'_, Self>,
        uuid_str: String,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let uuid = uuid::Uuid::parse_str(&uuid_str)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        slf.propagator.uuid = Some(uuid);
        Ok(slf)
    }

    /// Generate a new UUID, set it, and return self.
    pub fn with_new_uuid(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.propagator.uuid = Some(uuid::Uuid::now_v7());
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
        format!(
            "NumericalPropagator(epoch={:?}, state_dim={})",
            DStatePropagator::current_epoch(&self.propagator),
            DStatePropagator::state_dim(&self.propagator)
        )
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
                "TimeEvent has already been consumed",
            ));
        }

        if let Ok(mut value_event) = event.extract::<PyRefMut<PyValueEvent>>() {
            if let Some(inner) = value_event.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "ValueEvent has already been consumed",
            ));
        }

        if let Ok(mut binary_event) = event.extract::<PyRefMut<PyBinaryEvent>>() {
            if let Some(inner) = binary_event.event.take() {
                self.propagator.add_event_detector(Box::new(inner));
                return Ok(());
            }
            return Err(exceptions::PyValueError::new_err(
                "BinaryEvent has already been consumed",
            ));
        }

        // Note: AltitudeEvent is orbit-specific and not available for generic NumericalPropagator

        Err(exceptions::PyTypeError::new_err(
            "Expected event detector type (TimeEvent, ValueEvent, or BinaryEvent)",
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
        let uuid =
            match uuid_str {
                Some(s) => Some(uuid::Uuid::parse_str(&s).map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e))
                })?),
                None => None,
            };
        self.propagator.uuid = uuid;
        Ok(())
    }

    /// Generate a new UUID and set it in-place (mutating).
    pub fn generate_uuid(&mut self) {
        self.propagator.uuid = Some(uuid::Uuid::now_v7());
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
    pub fn with_identity(
        mut slf: PyRefMut<'_, Self>,
        name: Option<String>,
        uuid_str: Option<String>,
        id: Option<u64>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let uuid =
            match uuid_str {
                Some(s) => Some(uuid::Uuid::parse_str(&s).map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e))
                })?),
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
    pub fn set_identity(
        &mut self,
        name: Option<String>,
        uuid_str: Option<String>,
        id: Option<u64>,
    ) -> PyResult<()> {
        let uuid =
            match uuid_str {
                Some(s) => Some(uuid::Uuid::parse_str(&s).map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e))
                })?),
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
        PyInterpolationMethod {
            method: self.propagator.get_interpolation_method(),
        }
    }

    /// Set the covariance interpolation method using builder pattern.
    /// Note: Returns None as Python doesn't support returning mutable self with borrowed args.
    /// Use method chaining via separate calls or use set_covariance_interpolation_method instead.
    ///
    /// Args:
    ///     method (CovarianceInterpolationMethod): The covariance interpolation method to use.
    pub fn with_covariance_interpolation_method(
        &mut self,
        method: &PyCovarianceInterpolationMethod,
    ) {
        self.propagator
            .set_covariance_interpolation_method(method.method);
    }

    /// Set the covariance interpolation method in-place.
    ///
    /// Args:
    ///     method (CovarianceInterpolationMethod): The covariance interpolation method to use.
    pub fn set_covariance_interpolation_method(
        &mut self,
        method: &PyCovarianceInterpolationMethod,
    ) {
        self.propagator
            .set_covariance_interpolation_method(method.method);
    }

    /// Get the current covariance interpolation method.
    ///
    /// Returns:
    ///     CovarianceInterpolationMethod: The current covariance interpolation method.
    pub fn get_covariance_interpolation_method(&self) -> PyCovarianceInterpolationMethod {
        PyCovarianceInterpolationMethod {
            method: self.propagator.get_covariance_interpolation_method(),
        }
    }
}
