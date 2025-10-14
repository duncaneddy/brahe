// Import traits needed by propagator methods
use crate::orbits::traits::{AnalyticPropagator, OrbitPropagator};

/// Computes the orbital period of an object around Earth.
///
/// Uses rastro.constants.GM_EARTH as the standard gravitational parameter for the calculation.
///
/// Args:
///     a (float): The semi-major axis of the astronomical object in meters.
///
/// Returns:
///     float: The orbital period of the astronomical object in seconds.
#[pyfunction]
#[pyo3(text_signature = "(a)")]
#[pyo3(name = "orbital_period")]
fn py_orbital_period(a: f64) -> PyResult<f64> {
    Ok(orbits::orbital_period(a))
}

/// Computes the orbital period of an astronomical object around a general body.
///
/// Args:
///     a (float): The semi-major axis of the astronomical object in meters.
///     gm (float): The standard gravitational parameter of primary body in m³/s².
///
/// Returns:
///     float: The orbital period of the astronomical object in seconds.
#[pyfunction]
#[pyo3(text_signature = "(a, gm)")]
#[pyo3(name = "orbital_period_general")]
fn py_orbital_period_general(a: f64, gm: f64) -> PyResult<f64> {
    Ok(orbits::orbital_period_general(a, gm))
}

/// Computes the mean motion of an astronomical object around Earth.
///
/// Args:
///     a (float): The semi-major axis of the astronomical object in meters.
///     angle_format (AngleFormat): Return output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: The mean motion of the astronomical object in radians or degrees.
#[pyfunction]
#[pyo3(text_signature = "(a, angle_format)")]
#[pyo3(name = "mean_motion")]
fn py_mean_motion(a: f64, angle_format: &PyAngleFormat) -> PyResult<f64> {
    Ok(orbits::mean_motion(a, angle_format.value))
}

/// Computes the mean motion of an astronomical object around a general body
/// given a semi-major axis.
///
/// Args:
///     a (float): The semi-major axis of the astronomical object in meters.
///     gm (float): The standard gravitational parameter of primary body in m³/s².
///     angle_format (AngleFormat): Return output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: The mean motion of the astronomical object in radians or degrees.
#[pyfunction]
#[pyo3(text_signature = "(a, gm, angle_format)")]
#[pyo3(name = "mean_motion_general")]
fn py_mean_motion_general(a: f64, gm: f64, angle_format: &PyAngleFormat) -> PyResult<f64> {
    Ok(orbits::mean_motion_general(a, gm, angle_format.value))
}

/// Computes the semi-major axis of an astronomical object from Earth
/// given the object's mean motion.
///
/// Args:
///     n (float): The mean motion of the astronomical object in radians or degrees.
///     angle_format (AngleFormat): Interpret mean motion as AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: The semi-major axis of the astronomical object in meters.
#[pyfunction]
#[pyo3(text_signature = "(n, angle_format)")]
#[pyo3(name = "semimajor_axis")]
fn py_semimajor_axis(n: f64, angle_format: &PyAngleFormat) -> PyResult<f64> {
    Ok(orbits::semimajor_axis(n, angle_format.value))
}

/// Computes the semi-major axis of an astronomical object from a general body
/// given the object's mean motion.
///
/// Args:
///     n (float): The mean motion of the astronomical object in radians or degrees.
///     gm (float): The standard gravitational parameter of primary body in m³/s².
///     angle_format (AngleFormat): Interpret mean motion as AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: The semi-major axis of the astronomical object in meters.
#[pyfunction]
#[pyo3(text_signature = "(n, gm, angle_format)")]
#[pyo3(name = "semimajor_axis_general")]
fn py_semimajor_axis_general(n: f64, gm: f64, angle_format: &PyAngleFormat) -> PyResult<f64> {
    Ok(orbits::semimajor_axis_general(n, gm, angle_format.value))
}

/// Computes the semi-major axis from orbital period for a general body.
///
/// Args:
///     period (float): The orbital period in seconds.
///     gm (float): The standard gravitational parameter of primary body in m³/s².
///
/// Returns:
///     float: The semi-major axis in meters.
#[pyfunction]
#[pyo3(text_signature = "(period, gm)")]
#[pyo3(name = "semimajor_axis_from_orbital_period_general")]
fn py_semimajor_axis_from_orbital_period_general(period: f64, gm: f64) -> PyResult<f64> {
    Ok(orbits::semimajor_axis_from_orbital_period_general(period, gm))
}

/// Computes the semi-major axis from orbital period around Earth.
///
/// Args:
///     period (float): The orbital period in seconds.
///
/// Returns:
///     float: The semi-major axis in meters.
#[pyfunction]
#[pyo3(text_signature = "(period)")]
#[pyo3(name = "semimajor_axis_from_orbital_period")]
fn py_semimajor_axis_from_orbital_period(period: f64) -> PyResult<f64> {
    Ok(orbits::semimajor_axis_from_orbital_period(period))
}

/// Computes the perigee velocity of an astronomical object around Earth.
///
/// Args:
///     a (float): The semi-major axis of the astronomical object in meters.
///     e (float): The eccentricity of the astronomical object's orbit (dimensionless).
///
/// Returns:
///     float: The magnitude of velocity of the object at perigee in m/s.
#[pyfunction]
#[pyo3(text_signature = "(a, e)")]
#[pyo3(name = "perigee_velocity")]
fn py_perigee_velocity(a: f64, e: f64) -> PyResult<f64> {
    Ok(orbits::perigee_velocity(a, e))
}

/// Computes the periapsis velocity of an astronomical object around a general body.
///
/// Args:
///     a (float): The semi-major axis of the astronomical object in meters.
///     e (float): The eccentricity of the astronomical object's orbit (dimensionless).
///     gm (float): The standard gravitational parameter of primary body in m³/s².
///
/// Returns:
///     float: The magnitude of velocity of the object at periapsis in m/s.
#[pyfunction]
#[pyo3(text_signature = "(a, e, gm)")]
#[pyo3(name = "periapsis_velocity")]
fn py_periapsis_velocity(a: f64, e: f64, gm: f64) -> PyResult<f64> {
    Ok(orbits::periapsis_velocity(a, e, gm))
}

/// Calculate the distance of an object at its periapsis.
///
/// Args:
///     a (float): The semi-major axis of the astronomical object in meters.
///     e (float): The eccentricity of the astronomical object's orbit (dimensionless).
///
/// Returns:
///     float: The distance of the object at periapsis in meters.
#[pyfunction]
#[pyo3(text_signature = "(a, e)")]
#[pyo3(name = "periapsis_distance")]
fn py_periapsis_distance(a: f64, e: f64) -> PyResult<f64> {
    Ok(orbits::periapsis_distance(a, e))
}

/// Computes the apogee velocity of an astronomical object around Earth.
///
/// Args:
///     a (float): The semi-major axis of the astronomical object in meters.
///     e (float): The eccentricity of the astronomical object's orbit (dimensionless).
///
/// Returns:
///     float: The magnitude of velocity of the object at apogee in m/s.
#[pyfunction]
#[pyo3(text_signature = "(a, e)")]
#[pyo3(name = "apogee_velocity")]
fn py_apogee_velocity(a: f64, e: f64) -> PyResult<f64> {
    Ok(orbits::apogee_velocity(a, e))
}

/// Computes the apoapsis velocity of an astronomical object around a general body.
///
/// Args:
///     a (float): The semi-major axis of the astronomical object in meters.
///     e (float): The eccentricity of the astronomical object's orbit (dimensionless).
///     gm (float): The standard gravitational parameter of primary body in m³/s².
///
/// Returns:
///     float: The magnitude of velocity of the object at apoapsis in m/s.
#[pyfunction]
#[pyo3(text_signature = "(a, e, gm)")]
#[pyo3(name = "apoapsis_velocity")]
fn py_apoapsis_velocity(a: f64, e: f64, gm: f64) -> PyResult<f64> {
    Ok(orbits::apoapsis_velocity(a, e, gm))
}

/// Calculate the distance of an object at its apoapsis.
///
/// Args:
///     a (float): The semi-major axis of the astronomical object in meters.
///     e (float): The eccentricity of the astronomical object's orbit (dimensionless).
///
/// Returns:
///     float: The distance of the object at apoapsis in meters.
#[pyfunction]
#[pyo3(text_signature = "(a, e)")]
#[pyo3(name = "apoapsis_distance")]
fn py_apoapsis_distance(a: f64, e: f64) -> PyResult<f64> {
    Ok(orbits::apoapsis_distance(a, e))
}

/// Computes the inclination for a Sun-synchronous orbit around Earth based on
/// the J2 gravitational perturbation.
///
/// Args:
///     a (float): The semi-major axis of the astronomical object in meters.
///     e (float): The eccentricity of the astronomical object's orbit (dimensionless).
///     angle_format (AngleFormat): Return output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: Inclination for a Sun synchronous orbit in degrees or radians.
#[pyfunction]
#[pyo3(text_signature = "(a, e, angle_format)")]
#[pyo3(name = "sun_synchronous_inclination")]
fn py_sun_synchronous_inclination(a: f64, e: f64, angle_format: &PyAngleFormat) -> PyResult<f64> {
    Ok(orbits::sun_synchronous_inclination(a, e, angle_format.value))
}

/// Converts eccentric anomaly into mean anomaly.
///
/// Args:
///     anm_ecc (float): Eccentric anomaly in radians or degrees.
///     e (float): The eccentricity of the astronomical object's orbit (dimensionless).
///     angle_format (AngleFormat): Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: Mean anomaly in radians or degrees.
#[pyfunction]
#[pyo3(text_signature = "(anm_ecc, e, angle_format)")]
#[pyo3(name = "anomaly_eccentric_to_mean")]
fn py_anomaly_eccentric_to_mean(anm_ecc: f64, e: f64, angle_format: &PyAngleFormat) -> PyResult<f64> {
    Ok(orbits::anomaly_eccentric_to_mean(anm_ecc, e, angle_format.value))
}

/// Converts mean anomaly into eccentric anomaly.
///
/// Args:
///     anm_mean (float): Mean anomaly in radians or degrees.
///     e (float): The eccentricity of the astronomical object's orbit (dimensionless).
///     angle_format (AngleFormat): Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: Eccentric anomaly in radians or degrees.
#[pyfunction]
#[pyo3(text_signature = "(anm_mean, e, angle_format)")]
#[pyo3(name = "anomaly_mean_to_eccentric")]
fn py_anomaly_mean_to_eccentric(anm_mean: f64, e: f64, angle_format: &PyAngleFormat) -> PyResult<f64> {
    match orbits::anomaly_mean_to_eccentric(anm_mean, e, angle_format.value) {
        Ok(value) => Ok(value),
        Err(err) => Err(exceptions::PyRuntimeError::new_err(err)),
    }
}

/// Converts true anomaly into eccentric anomaly.
///
/// Args:
///     anm_true (float): True anomaly in radians or degrees.
///     e (float): The eccentricity of the astronomical object's orbit (dimensionless).
///     angle_format (AngleFormat): Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: Eccentric anomaly in radians or degrees.
#[pyfunction]
#[pyo3(text_signature = "(anm_true, e, angle_format)")]
#[pyo3(name = "anomaly_true_to_eccentric")]
fn py_anomaly_true_to_eccentric(anm_true: f64, e: f64, angle_format: &PyAngleFormat) -> PyResult<f64> {
    Ok(orbits::anomaly_true_to_eccentric(anm_true, e, angle_format.value))
}

/// Converts eccentric anomaly into true anomaly.
///
/// Args:
///     anm_ecc (float): Eccentric anomaly in radians or degrees.
///     e (float): The eccentricity of the astronomical object's orbit (dimensionless).
///     angle_format (AngleFormat): Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: True anomaly in radians or degrees.
#[pyfunction]
#[pyo3(text_signature = "(anm_ecc, e, angle_format)")]
#[pyo3(name = "anomaly_eccentric_to_true")]
fn py_anomaly_eccentric_to_true(anm_ecc: f64, e: f64, angle_format: &PyAngleFormat) -> PyResult<f64> {
    Ok(orbits::anomaly_eccentric_to_true(anm_ecc, e, angle_format.value))
}

/// Converts true anomaly into mean anomaly.
///
/// Args:
///     anm_true (float): True anomaly in radians or degrees.
///     e (float): The eccentricity of the astronomical object's orbit (dimensionless).
///     angle_format (AngleFormat): Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: Mean anomaly in radians or degrees.
#[pyfunction]
#[pyo3(text_signature = "(anm_true, e, angle_format)")]
#[pyo3(name = "anomaly_true_to_mean")]
fn py_anomaly_true_to_mean(anm_true: f64, e: f64, angle_format: &PyAngleFormat) -> PyResult<f64> {
    Ok(orbits::anomaly_true_to_mean(anm_true, e, angle_format.value))
}

/// Converts mean anomaly into true anomaly.
///
/// Args:
///     anm_mean (float): Mean anomaly in radians or degrees.
///     e (float): The eccentricity of the astronomical object's orbit (dimensionless).
///     angle_format (AngleFormat): Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: True anomaly in radians or degrees.
#[pyfunction]
#[pyo3(text_signature = "(anm_mean, e, angle_format)")]
#[pyo3(name = "anomaly_mean_to_true")]
fn py_anomaly_mean_to_true(anm_mean: f64, e: f64, angle_format: &PyAngleFormat) -> PyResult<f64> {
    match orbits::anomaly_mean_to_true(anm_mean, e, angle_format.value) {
        Ok(value) => Ok(value),
        Err(err) => Err(exceptions::PyRuntimeError::new_err(err)),
    }
}

// New propagator implementations

/// Python wrapper for SGPPropagator (replaces TLE)
#[pyclass]
#[pyo3(name = "SGPPropagator")]
pub struct PySGPPropagator {
    pub(crate) propagator: orbits::SGPPropagator,
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
        match orbits::SGPPropagator::from_tle(&line1, &line2, step_size) {
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
        match orbits::SGPPropagator::from_3le(Some(&name), &line1, &line2, step_size) {
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
        PyEpoch { obj: self.propagator.initial_epoch() }
    }

    /// Get current epoch.
    ///
    /// Returns:
    ///     Epoch: Current propagator epoch.
    #[getter]
    pub fn current_epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.propagator.current_epoch() }
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
    #[pyo3(text_signature = "()")]
    pub fn step(&mut self) {
        self.propagator.step();
    }

    /// Step forward by a specified time duration.
    ///
    /// Args:
    ///     step_size (float): Time step in seconds.
    #[pyo3(text_signature = "(step_size)")]
    pub fn step_by(&mut self, step_size: f64) {
        self.propagator.step_by(step_size);
    }

    /// Step past a specified target epoch.
    ///
    /// Args:
    ///     target_epoch (Epoch): The epoch to step past.
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn step_past(&mut self, target_epoch: PyRef<PyEpoch>) {
        self.propagator.step_past(target_epoch.obj);
    }

    /// Propagate forward by specified number of steps.
    ///
    /// Args:
    ///     num_steps (int): Number of steps to take.
    #[pyo3(text_signature = "(num_steps)")]
    pub fn propagate_steps(&mut self, num_steps: usize) {
        self.propagator.propagate_steps(num_steps);
    }

    /// Propagate to a specific target epoch.
    ///
    /// Args:
    ///     target_epoch (Epoch): The epoch to propagate to.
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn propagate_to(&mut self, target_epoch: PyRef<PyEpoch>) {
        self.propagator.propagate_to(target_epoch.obj);
    }

    /// Reset propagator to initial conditions.
    #[pyo3(text_signature = "()")]
    pub fn reset(&mut self) {
        self.propagator.reset();
    }

    /// Set trajectory eviction policy based on maximum size.
    ///
    /// Args:
    ///     max_size (int): Maximum number of states to keep in trajectory.
    #[pyo3(text_signature = "(max_size)")]
    pub fn set_eviction_policy_max_size(&mut self, max_size: usize) -> PyResult<()> {
        self.propagator.set_eviction_policy_max_size(max_size)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Set trajectory eviction policy based on maximum age.
    ///
    /// Args:
    ///     max_age (float): Maximum age in seconds to keep states in trajectory.
    #[pyo3(text_signature = "(max_age)")]
    pub fn set_eviction_policy_max_age(&mut self, max_age: f64) -> PyResult<()> {
        self.propagator.set_eviction_policy_max_age(max_age)
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Get accumulated trajectory.
    ///
    /// Returns:
    ///     OrbitalTrajectory: The accumulated trajectory.
    #[getter]
    pub fn trajectory(&self) -> PyOrbitalTrajectory {
        PyOrbitalTrajectory { trajectory: self.propagator.trajectory.clone() }
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!("SGPPropagator(norad_id={}, name={:?}, epoch={:?})",
                self.propagator.norad_id,
                self.propagator.satellite_name,
                self.propagator.initial_epoch())
    }

    /// String conversion.
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

/// Python wrapper for KeplerianPropagator (new architecture)
#[pyclass]
#[pyo3(name = "KeplerianPropagator")]
pub struct PyKeplerianPropagator {
    pub(crate) propagator: orbits::KeplerianPropagator,
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

        let propagator = orbits::KeplerianPropagator::new(
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

        let propagator = orbits::KeplerianPropagator::from_keplerian(
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

        let propagator = orbits::KeplerianPropagator::from_eci(
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

        let propagator = orbits::KeplerianPropagator::from_ecef(
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
    #[pyo3(text_signature = "()")]
    pub fn step(&mut self) {
        self.propagator.step();
    }

    /// Step forward by a specified time duration.
    ///
    /// Args:
    ///     step_size (float): Time step in seconds.
    #[pyo3(text_signature = "(step_size)")]
    pub fn step_by(&mut self, step_size: f64) {
        self.propagator.step_by(step_size);
    }

    /// Step past a specified target epoch.
    ///
    /// Args:
    ///     target_epoch (Epoch): The epoch to step past.
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn step_past(&mut self, target_epoch: PyRef<PyEpoch>) {
        self.propagator.step_past(target_epoch.obj);
    }

    /// Propagate forward by specified number of steps.
    ///
    /// Args:
    ///     num_steps (int): Number of steps to take.
    #[pyo3(text_signature = "(num_steps)")]
    pub fn propagate_steps(&mut self, num_steps: usize) {
        self.propagator.propagate_steps(num_steps);
    }

    /// Propagate to a specific target epoch.
    ///
    /// Args:
    ///     target_epoch (Epoch): The epoch to propagate to.
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn propagate_to(&mut self, target_epoch: PyRef<PyEpoch>) {
        self.propagator.propagate_to(target_epoch.obj);
    }

    /// Reset propagator to initial conditions.
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
    #[getter]
    pub fn trajectory(&self) -> PyOrbitalTrajectory {
        PyOrbitalTrajectory { trajectory: self.propagator.trajectory.clone() }
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

// Legacy TLE support (for backward compatibility)
/// Legacy TLE class for backward compatibility.
#[pyclass]
#[pyo3(name = "TLE")]
pub struct PyTLE {
    // Minimal implementation using SGPPropagator internally
    propagator: orbits::SGPPropagator,
}

#[pymethods]
impl PyTLE {
    /// Create a TLE from lines (legacy compatibility).
    ///
    /// Args:
    ///     line1 (str): First line of TLE data.
    ///     line2 (str): Second line of TLE data.
    ///     step_size (float): Step size in seconds for propagation. Defaults to 60.0.
    ///
    /// Returns:
    ///     TLE: New TLE instance.
    #[classmethod]
    #[pyo3(signature = (line1, line2, step_size=60.0))]
    pub fn from_lines(_cls: &Bound<'_, PyType>, line1: String, line2: String, step_size: Option<f64>) -> PyResult<Self> {
        let step_size = step_size.unwrap_or(60.0);
        match orbits::SGPPropagator::from_tle(&line1, &line2, step_size) {
            Ok(propagator) => Ok(PyTLE { propagator }),
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

    /// Get TLE epoch.
    ///
    /// Returns:
    ///     Epoch: Epoch of the TLE data.
    #[getter]
    pub fn epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.propagator.initial_epoch() }
    }
}

/// Validate TLE lines.
///
/// Args:
///     line1 (str): First line of TLE data.
///     line2 (str): Second line of TLE data.
///
/// Returns:
///     bool: True if both lines are valid.
#[pyfunction]
#[pyo3(text_signature = "(line1, line2)")]
#[pyo3(name = "validate_tle_lines")]
fn py_validate_tle_lines(line1: String, line2: String) -> PyResult<bool> {
    Ok(orbits::validate_tle_lines(&line1, &line2))
}

/// Validate single TLE line.
///
/// Args:
///     line (str): TLE line to validate.
///
/// Returns:
///     bool: True if the line is valid.
#[pyfunction]
#[pyo3(text_signature = "(line)")]
#[pyo3(name = "validate_tle_line")]
fn py_validate_tle_line(line: String) -> PyResult<bool> {
    Ok(orbits::validate_tle_line(&line))
}

/// Calculate TLE line checksum.
///
/// Args:
///     line (str): TLE line.
///
/// Returns:
///     int: Checksum value.
#[pyfunction]
#[pyo3(text_signature = "(line)")]
#[pyo3(name = "calculate_tle_line_checksum")]
fn py_calculate_tle_line_checksum(line: String) -> PyResult<u32> {
    Ok(orbits::calculate_tle_line_checksum(&line))
}





/// Extract Keplerian orbital elements from TLE lines.
///
/// Extracts the standard six Keplerian orbital elements from Two-Line Element (TLE) data.
/// Returns elements in standard order: [a, e, i, raan, argp, M] where angles are in radians.
///
/// Args:
///     line1 (str): First line of TLE data.
///     line2 (str): Second line of TLE data.
///
/// Returns:
///     tuple: A tuple containing:
///         - epoch (Epoch): Epoch of the TLE data.
///         - elements (numpy.ndarray): Six Keplerian elements [a, e, i, raan, argp, M] where
///           a is semi-major axis in meters, e is eccentricity (dimensionless), and
///           i, raan, argp, M are in radians.
#[pyfunction]
#[pyo3(text_signature = "(line1, line2)")]
#[pyo3(name = "keplerian_elements_from_tle")]
fn py_keplerian_elements_from_tle<'py>(py: Python<'py>, line1: String, line2: String) -> PyResult<(PyEpoch, Bound<'py, PyArray<f64, Ix1>>)> {
    match orbits::keplerian_elements_from_tle(&line1, &line2) {
        Ok((epoch, elements)) => {
            let elements_array = elements.as_slice().to_pyarray(py).to_owned();
            let py_epoch = PyEpoch { obj: epoch };
            Ok((py_epoch, elements_array))
        },
        Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

/// Convert Keplerian elements to TLE lines.
///
/// Converts standard Keplerian orbital elements to Two-Line Element (TLE) format.
/// Input angles should be in degrees for compatibility with TLE format.
///
/// Args:
///     epoch (Epoch): Epoch of the elements.
///     elements (numpy.ndarray): Keplerian elements [a (m), e, i (deg), raan (deg), argp (deg), M (deg)].
///     norad_id (str): NORAD catalog number (supports numeric and Alpha-5 format).
///
/// Returns:
///     tuple: A tuple containing (line1, line2) - the two TLE lines as strings.
#[pyfunction]
#[pyo3(text_signature = "(epoch, elements, norad_id)")]
#[pyo3(name = "keplerian_elements_to_tle")]
fn py_keplerian_elements_to_tle(
    epoch: &PyEpoch,
    elements: PyReadonlyArray1<f64>,
    norad_id: &str,
) -> PyResult<(String, String)> {
    let elements_array = elements.as_array();
    let elements_vec = na::Vector6::from_row_slice(elements_array.as_slice().unwrap());

    match orbits::keplerian_elements_to_tle(
        &epoch.obj,
        &elements_vec,
        norad_id,
    ) {
        Ok((line1, line2)) => Ok((line1, line2)),
        Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

/// Create complete TLE lines from all parameters.
///
/// Creates Two-Line Element (TLE) lines from complete set of orbital and administrative parameters.
/// Provides full control over all TLE fields including derivatives and drag terms.
///
/// Args:
///     epoch (Epoch): Epoch of the elements.
///     inclination (float): Inclination in degrees.
///     raan (float): Right ascension of ascending node in degrees.
///     eccentricity (float): Eccentricity (dimensionless).
///     arg_perigee (float): Argument of periapsis in degrees.
///     mean_anomaly (float): Mean anomaly in degrees.
///     mean_motion (float): Mean motion in revolutions per day.
///     norad_id (str): NORAD catalog number (supports numeric and Alpha-5 format).
///     ephemeris_type (int): Ephemeris type (0-9).
///     element_set_number (int): Element set number.
///     revolution_number (int): Revolution number at epoch.
///     classification (str, optional): Security classification. Defaults to ' '.
///     intl_designator (str, optional): International designator. Defaults to ''.
///     first_derivative (float, optional): First derivative of mean motion. Defaults to 0.0.
///     second_derivative (float, optional): Second derivative of mean motion. Defaults to 0.0.
///     bstar (float, optional): BSTAR drag term. Defaults to 0.0.
///
/// Returns:
///     tuple: A tuple containing (line1, line2) - the two TLE lines as strings.
#[pyfunction]
#[pyo3(text_signature = "(epoch, inclination, raan, eccentricity, arg_perigee, mean_anomaly, mean_motion, norad_id, ephemeris_type, element_set_number, revolution_number, classification=None, intl_designator=None, first_derivative=None, second_derivative=None, bstar=None)")]
#[pyo3(name = "create_tle_lines")]
#[allow(clippy::too_many_arguments)]
fn py_create_tle_lines(
    epoch: &PyEpoch,
    inclination: f64,
    raan: f64,
    eccentricity: f64,
    arg_perigee: f64,
    mean_anomaly: f64,
    mean_motion: f64,
    norad_id: &str,
    ephemeris_type: u8,
    element_set_number: u16,
    revolution_number: u32,
    classification: Option<char>,
    intl_designator: Option<String>,
    first_derivative: Option<f64>,
    second_derivative: Option<f64>,
    bstar: Option<f64>,
) -> PyResult<(String, String)> {
    let intl_designator_ref = intl_designator.as_deref();

    match orbits::create_tle_lines(
        &epoch.obj,
        norad_id,
        classification.unwrap_or(' '),
        intl_designator_ref.unwrap_or(""),
        mean_motion,
        eccentricity,
        inclination,
        raan,
        arg_perigee,
        mean_anomaly,
        first_derivative.unwrap_or(0.0),
        second_derivative.unwrap_or(0.0),
        bstar.unwrap_or(0.0),
        ephemeris_type,
        element_set_number,
        revolution_number,
    ) {
        Ok((line1, line2)) => Ok((line1, line2)),
        Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

/// Parse NORAD ID from string, handling both classic and Alpha-5 formats.
///
/// Args:
///     norad_str (str): NORAD ID string from TLE.
///
/// Returns:
///     int: Parsed numeric NORAD ID.
#[pyfunction]
#[pyo3(text_signature = "(norad_str)")]
#[pyo3(name = "parse_norad_id")]
fn py_parse_norad_id(norad_str: String) -> PyResult<u32> {
    match orbits::parse_norad_id(&norad_str) {
        Ok(id) => Ok(id),
        Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

/// Convert numeric NORAD ID to Alpha-5 format.
///
/// Args:
///     norad_id (int): Numeric NORAD ID (100000-339999).
///
/// Returns:
///     str: Alpha-5 format ID (e.g., "A0001").
#[pyfunction]
#[pyo3(text_signature = "(norad_id)")]
#[pyo3(name = "norad_id_numeric_to_alpha5")]
fn py_norad_id_numeric_to_alpha5(norad_id: u32) -> PyResult<String> {
    match orbits::norad_id_numeric_to_alpha5(norad_id) {
        Ok(alpha5_id) => Ok(alpha5_id),
        Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

/// Convert Alpha-5 NORAD ID to numeric format.
///
/// Args:
///     alpha5_id (str): Alpha-5 format ID (e.g., "A0001").
///
/// Returns:
///     int: Numeric NORAD ID.
#[pyfunction]
#[pyo3(text_signature = "(alpha5_id)")]
#[pyo3(name = "norad_id_alpha5_to_numeric")]
fn py_norad_id_alpha5_to_numeric(alpha5_id: String) -> PyResult<u32> {
    match orbits::norad_id_alpha5_to_numeric(&alpha5_id) {
        Ok(numeric_id) => Ok(numeric_id),
        Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

/// Extract Epoch from TLE line 1
///
/// Extracts and parses the epoch timestamp from the first line of TLE data.
/// The epoch is returned in UTC time system.
///
/// Args:
///     line1 (str): First line of TLE data
///
/// Returns:
///     Epoch: Extracted epoch in UTC time system
///
/// Examples:
///     >>> line1 = "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997"
///     >>> epoch = epoch_from_tle(line1)
///     >>> epoch.year()
///     2021
#[pyfunction]
#[pyo3(text_signature = "(line1)")]
#[pyo3(name = "epoch_from_tle")]
fn py_epoch_from_tle(line1: String) -> PyResult<PyEpoch> {
    match orbits::epoch_from_tle(&line1) {
        Ok(epoch) => Ok(PyEpoch { obj: epoch }),
        Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}