/// Computes the orbital period of an object around Earth.
///
/// Uses rastro.constants.GM_EARTH as the standard gravitational parameter for the calculation.
///
/// Arguments:
///     a (`float`): The semi-major axis of the astronomical object. Units: (m)
///
/// Returns:
///     period (`float`): The orbital period of the astronomical object. Units: (s)
#[pyfunction]
#[pyo3(text_signature = "(a)")]
#[pyo3(name = "orbital_period")]
fn py_orbital_period(a: f64) -> PyResult<f64> {
    Ok(orbits::orbital_period(a))
}

/// Computes the orbital period of an astronomical object around a general body.
///
/// Arguments:
///     a (`float`): The semi-major axis of the astronomical object. Units: (m)
///     gm (`float`): The standard gravitational parameter of primary body. Units: [m^3/s^2]
///
/// Returns:
///     period (`float`): The orbital period of the astronomical object. Units: (s)
#[pyfunction]
#[pyo3(text_signature = "(a, gm)")]
#[pyo3(name = "orbital_period_general")]
fn py_orbital_period_general(a: f64, gm: f64) -> PyResult<f64> {
    Ok(orbits::orbital_period_general(a, gm))
}

/// Computes the mean motion of an astronomical object around Earth.
///
/// Arguments:
///     a (`float`): The semi-major axis of the astronomical object. Units: (m)
///     as_degrees (`bool`): Return output in degrees instead of radians
///
/// Returns:
///     n (`float`): The mean motion of the astronomical object. Units: (rad) or (deg)
#[pyfunction]
#[pyo3(text_signature = "(a, as_degrees)")]
#[pyo3(name = "mean_motion")]
fn py_mean_motion(a: f64, as_degrees: bool) -> PyResult<f64> {
    Ok(orbits::mean_motion(a, as_degrees))
}

/// Computes the mean motion of an astronomical object around a general body
/// given a semi-major axis.
///
/// Arguments:
///     a (`float`): The semi-major axis of the astronomical object. Units: (m)
///     gm (`float`): The standard gravitational parameter of primary body. Units: [m^3/s^2]
///     as_degrees (`bool`): Return output in degrees instead of radians
///
/// Returns:
///     n (`float`): The mean motion of the astronomical object. Units: (rad) or (deg)
#[pyfunction]
#[pyo3(text_signature = "(a, gm, as_degrees)")]
#[pyo3(name = "mean_motion_general")]
fn py_mean_motion_general(a: f64, gm: f64, as_degrees: bool) -> PyResult<f64> {
    Ok(orbits::mean_motion_general(a, gm, as_degrees))
}

/// Computes the semi-major axis of an astronomical object from Earth
/// given the object's mean motion.
///
/// Arguments:
///     n (`float`): The mean motion of the astronomical object. Units: (rad) or (deg)
///     as_degrees (`bool`): Interpret mean motion as degrees if `true` or radians if `false`
///
/// Returns:
///     a (`float`): The semi-major axis of the astronomical object. Units: (m)
#[pyfunction]
#[pyo3(text_signature = "(a, as_degrees)")]
#[pyo3(name = "semimajor_axis")]
fn py_semimajor_axis(n: f64, as_degrees: bool) -> PyResult<f64> {
    Ok(orbits::semimajor_axis(n, as_degrees))
}

/// Computes the semi-major axis of an astronomical object from a general body
/// given the object's mean motion.
///
/// Arguments:
///     n (`float`): The mean motion of the astronomical object. Units: (rad) or (deg)
///     gm (`float`): The standard gravitational parameter of primary body. Units: [m^3/s^2]
///     as_degrees (`float`): Interpret mean motion as degrees if `true` or radians if `false`
///
/// Returns:
///     a (`float`): The semi-major axis of the astronomical object. Units: (m)
#[pyfunction]
#[pyo3(text_signature = "(a, gm, as_degrees)")]
#[pyo3(name = "semimajor_axis_general")]
fn py_semimajor_axis_general(n: f64, gm: f64, as_degrees: bool) -> PyResult<f64> {
    Ok(orbits::semimajor_axis_general(n, gm, as_degrees))
}

#[pyfunction]
#[pyo3(text_signature = "(T, gm)")]
#[pyo3(name = "semimajor_axis_from_orbital_period_general")]
fn py_semimajor_axis_from_orbital_period_general(period: f64, gm: f64) -> PyResult<f64> {
    Ok(orbits::semimajor_axis_from_orbital_period_general(period, gm))
}

#[pyfunction]
#[pyo3(text_signature = "(T)")]
#[pyo3(name = "semimajor_axis_from_orbital_period")]
fn py_semimajor_axis_from_orbital_period(period: f64) -> PyResult<f64> {
    Ok(orbits::semimajor_axis_from_orbital_period(period))
}

/// Computes the perigee velocity of an astronomical object around Earth.
///
/// Arguments:
///     a (`float`): The semi-major axis of the astronomical object. Units: (m)
///     e (`float`): The eccentricity of the astronomical object's orbit. Dimensionless
///
/// Returns:
///     v (`float`): The magnitude of velocity of the object at perigee. Units: (m/s)
#[pyfunction]
#[pyo3(text_signature = "(a, e)")]
#[pyo3(name = "perigee_velocity")]
fn py_perigee_velocity(a: f64, e: f64) -> PyResult<f64> {
    Ok(orbits::perigee_velocity(a, e))
}

/// Computes the periapsis velocity of an astronomical object around a general body.
///
/// Arguments:
///     a (`float`): The semi-major axis of the astronomical object. Units: (m)
///     e (`float`): The eccentricity of the astronomical object's orbit. Dimensionless
///     gm (`float`): The standard gravitational parameter of primary body. Units: [m^3/s^2]
///
/// Returns:
///     v (`float`): The magnitude of velocity of the object at periapsis. Units: (m/s)
#[pyfunction]
#[pyo3(text_signature = "(a, e)")]
#[pyo3(name = "periapsis_velocity")]
fn py_periapsis_velocity(a: f64, e: f64, gm: f64) -> PyResult<f64> {
    Ok(orbits::periapsis_velocity(a, e, gm))
}

/// Calculate the distance of an object at its periapsis
///
/// # Arguments
///
/// * `a`: The semi-major axis of the astronomical object. Units: (m)
/// * `e`: The eccentricity of the astronomical object's orbit. Dimensionless
///
/// # Returns
///
/// * `r`: The distance of the object at periapsis. Units (s)
#[pyfunction]
#[pyo3(text_signature = "(a, e)")]
#[pyo3(name = "periapsis_distance")]
fn py_periapsis_distance(a: f64, e: f64) -> PyResult<f64> {
    Ok(orbits::periapsis_distance(a, e))
}

/// Computes the apogee velocity of an astronomical object around Earth.
///
/// Arguments:
///     a (`float`): The semi-major axis of the astronomical object. Units: (m)
///     e (`float`): The eccentricity of the astronomical object's orbit. Dimensionless
///
/// Returns:
///     v (`float`): The magnitude of velocity of the object at apogee. Units: (m/s)
#[pyfunction]
#[pyo3(text_signature = "(a, e)")]
#[pyo3(name = "apogee_velocity")]
fn py_apogee_velocity(a: f64, e: f64) -> PyResult<f64> {
    Ok(orbits::apogee_velocity(a, e))
}

/// Computes the apoapsis velocity of an astronomical object around a general body.
///
/// Arguments:
///     a (`float`): The semi-major axis of the astronomical object. Units: (m)
///     e (`float`): The eccentricity of the astronomical object's orbit. Dimensionless
///     gm (`float`): The standard gravitational parameter of primary body. Units: [m^3/s^2]
///
/// Returns:
///     v (`float`): The magnitude of velocity of the object at apoapsis. Units: (m/s)
#[pyfunction]
#[pyo3(text_signature = "(a, e)")]
#[pyo3(name = "apoapsis_velocity")]
fn py_apoapsis_velocity(a: f64, e: f64, gm: f64) -> PyResult<f64> {
    Ok(orbits::apoapsis_velocity(a, e, gm))
}

/// Calculate the distance of an object at its apoapsis
///
/// # Arguments
///
/// * `a`: The semi-major axis of the astronomical object. Units: (m)
/// * `e`: The eccentricity of the astronomical object's orbit. Dimensionless
///
/// # Returns
///
/// * `r`: The distance of the object at apoapsis. Units (s)
#[pyfunction]
#[pyo3(text_signature = "(a, e)")]
#[pyo3(name = "apoapsis_distance")]
fn py_apoapsis_distance(a: f64, e: f64) -> PyResult<f64> {
    Ok(orbits::apoapsis_distance(a, e))
}

/// Computes the inclination for a Sun-synchronous orbit around Earth based on
/// the J2 gravitational perturbation.
///
/// Arguments:
///     a (`float`) The semi-major axis of the astronomical object. Units: (m)
///     e (`float`) The eccentricity of the astronomical object's orbit. Dimensionless
///     as_degrees (`bool`) Return output in degrees instead of radians
///
/// Returns:
///     inc (`float`) Inclination for a Sun synchronous orbit. Units: (deg) or (rad)
#[pyfunction]
#[pyo3(text_signature = "(a, e, as_degrees)")]
#[pyo3(name = "sun_synchronous_inclination")]
fn py_sun_synchronous_inclination(a: f64, e: f64, as_degrees: bool) -> PyResult<f64> {
    Ok(orbits::sun_synchronous_inclination(a, e, as_degrees))
}

/// Converts eccentric anomaly into mean anomaly.
///
/// Arguments:
///     anm_ecc (`float`): Eccentric anomaly. Units: (rad) or (deg)
///     e (`float`): The eccentricity of the astronomical object's orbit. Dimensionless
///     as_degrees (`bool`): Interprets input and returns output in (deg) if `true` or (rad) if `false`
///
/// Returns:
///     anm_mean (`float`): Mean anomaly. Units: (rad) or (deg)
#[pyfunction]
#[pyo3(text_signature = "(anm_ecc, e, as_degrees)")]
#[pyo3(name = "anomaly_eccentric_to_mean")]
fn py_anomaly_eccentric_to_mean(anm_ecc: f64, e: f64, as_degrees: bool) -> PyResult<f64> {
    Ok(orbits::anomaly_eccentric_to_mean(anm_ecc, e, as_degrees))
}

/// Converts mean anomaly into eccentric anomaly
///
/// Arguments:
///     anm_mean (`float`): Mean anomaly. Units: (rad) or (deg)
///     e (`float`): The eccentricity of the astronomical object's orbit. Dimensionless
///     as_degrees (`float`): Interprets input and returns output in (deg) if `true` or (rad) if `false`
///
/// Returns:
///     anm_ecc (`float`): Eccentric anomaly. Units: (rad) or (deg)
#[pyfunction]
#[pyo3(text_signature = "(anm_mean, e, as_degrees)")]
#[pyo3(name = "anomaly_mean_to_eccentric")]
fn py_anomaly_mean_to_eccentric(anm_mean: f64, e: f64, as_degrees: bool) -> PyResult<f64> {
    let res = orbits::anomaly_mean_to_eccentric(anm_mean, e, as_degrees);
    if res.is_ok() {
        Ok(res.unwrap())
    } else {
        Err(exceptions::PyRuntimeError::new_err(res.err().unwrap()))
    }
}

/// Converts true anomaly into eccentric anomaly
///
/// Arguments:
///     anm_true (`float`): true anomaly. Units: (rad) or (deg)
///     e (`float`): The eccentricity of the astronomical object's orbit. Dimensionless
///     as_degrees (`bool`): Interprets input and returns output in (deg) if `true` or (rad) if `false`
///
/// Returns:
///     anm_ecc (`float`): Eccentric anomaly. Units: (rad) or (deg)
#[pyfunction]
#[pyo3(text_signature = "(anm_true, e, as_degrees))")]
#[pyo3(name = "anomaly_true_to_eccentric")]
fn py_anomaly_true_to_eccentric(anm_true: f64, e: f64, as_degrees: bool) -> PyResult<f64> {
    Ok(orbits::anomaly_true_to_eccentric(anm_true, e, as_degrees))
}

/// Converts eccentric anomaly into true anomaly
///
/// # Arguments
///     anm_ecc (`float`): Eccentric anomaly. Units: (rad) or (deg)
///     e (`float`): The eccentricity of the astronomical object's orbit. Dimensionless
///     as_degrees (`bool`): Interprets input and returns output in (deg) if `true` or (rad) if `false`
///
/// # Returns
///     anm_true (`float`): true anomaly. Units: (rad) or (deg)
#[pyfunction]
#[pyo3(text_signature = "(anm_ecc, e, as_degrees))")]
#[pyo3(name = "anomaly_eccentric_to_true")]
fn py_anomaly_eccentric_to_true(anm_ecc: f64, e: f64, as_degrees: bool) -> PyResult<f64> {
    Ok(orbits::anomaly_eccentric_to_true(anm_ecc, e, as_degrees))
}

/// Converts true anomaly into mean anomaly.
///
/// Arguments:
///     anm_true (`float`): True anomaly. Units: (rad) or (deg)
///     e (`float`): The eccentricity of the astronomical object's orbit. Dimensionless
///     as_degrees (`bool`): Interprets input and returns output in (deg) if `true` or (rad) if `false`
///
/// Returns:
///     anm_mean (`float`): Mean anomaly. Units: (rad) or (deg)
#[pyfunction]
#[pyo3(text_signature = "(anm_ecc, e, as_degrees))")]
#[pyo3(name = "anomaly_true_to_mean")]
fn py_anomaly_true_to_mean(anm_ecc: f64, e: f64, as_degrees: bool) -> PyResult<f64> {
    Ok(orbits::anomaly_true_to_mean(anm_ecc, e, as_degrees))
}

/// Converts mean anomaly into true anomaly
///
/// Arguments:
///     anm_mean (`float`): Mean anomaly. Units: (rad) or (deg)
///     e (`float`): The eccentricity of the astronomical object's orbit. Dimensionless
///     as_degrees (`bool`): Interprets input and returns output in (deg) if `true` or (rad) if `false`
///
/// Returns:
///     anm_true (`float`): True anomaly. Units: (rad) or (deg)
#[pyfunction]
#[pyo3(text_signature = "(anm_mean, e, as_degrees)")]
#[pyo3(name = "anomaly_mean_to_true")]
fn py_anomaly_mean_to_true(anm_mean: f64, e: f64, as_degrees: bool) -> PyResult<f64> {
    let res = orbits::anomaly_mean_to_true(anm_mean, e, as_degrees);
    if res.is_ok() {
        Ok(res.unwrap())
    } else {
        Err(exceptions::PyRuntimeError::new_err(res.err().unwrap()))
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
    /// Create a new SGP propagator from TLE lines
    ///
    /// Arguments:
    ///     line1 (str): First line of TLE data
    ///     line2 (str): Second line of TLE data
    ///     step_size (float): Step size in seconds for propagation (default: 60.0)
    ///
    /// Returns:
    ///     SGPPropagator: New SGP propagator instance
    #[classmethod]
    #[pyo3(signature = (line1, line2, step_size=60.0))]
    pub fn from_tle(_cls: &Bound<'_, PyType>, line1: String, line2: String, step_size: Option<f64>) -> PyResult<Self> {
        let step_size = step_size.unwrap_or(60.0);
        match orbits::SGPPropagator::from_tle(&line1, &line2, step_size) {
            Ok(propagator) => Ok(PySGPPropagator { propagator }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Create a new SGP propagator from 3-line TLE format (with satellite name)
    ///
    /// Arguments:
    ///     name (str): Satellite name (line 0)
    ///     line1 (str): First line of TLE data
    ///     line2 (str): Second line of TLE data
    ///     step_size (float): Step size in seconds for propagation (default: 60.0)
    ///
    /// Returns:
    ///     SGPPropagator: New SGP propagator instance
    #[classmethod]
    #[pyo3(signature = (name, line1, line2, step_size=60.0))]
    pub fn from_3le(_cls: &Bound<'_, PyType>, name: String, line1: String, line2: String, step_size: Option<f64>) -> PyResult<Self> {
        let step_size = step_size.unwrap_or(60.0);
        match orbits::SGPPropagator::from_3le(Some(&name), &line1, &line2, step_size) {
            Ok(propagator) => Ok(PySGPPropagator { propagator }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get NORAD ID
    #[getter]
    pub fn norad_id(&self) -> u32 {
        self.propagator.norad_id
    }

    /// Get satellite name (if available)
    #[getter]
    pub fn satellite_name(&self) -> Option<String> {
        self.propagator.satellite_name.clone()
    }

    /// Get TLE epoch
    #[getter]
    pub fn epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.propagator.initial_epoch() }
    }

    /// Get current epoch
    #[getter]
    pub fn current_epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.propagator.current_epoch() }
    }

    /// Get step size in seconds
    #[getter]
    pub fn step_size(&self) -> f64 {
        self.propagator.step_size()
    }

    /// Set step size in seconds
    #[setter]
    pub fn set_step_size(&mut self, step_size: f64) {
        self.propagator.set_step_size(step_size);
    }

    /// Set output to Cartesian coordinates
    #[pyo3(text_signature = "()")]
    pub fn set_output_cartesian(&mut self) {
        self.propagator.set_output_cartesian();
    }

    /// Set output to Keplerian elements
    #[pyo3(text_signature = "()")]
    pub fn set_output_keplerian(&mut self) {
        self.propagator.set_output_keplerian();
    }

    /// Set output frame
    #[pyo3(text_signature = "(frame)")]
    pub fn set_output_frame(&mut self, frame: PyRef<PyOrbitFrame>) {
        self.propagator.set_output_frame(frame.frame);
    }

    /// Set output angle format
    #[pyo3(text_signature = "(angle_format)")]
    pub fn set_output_angle_format(&mut self, angle_format: PyRef<PyAngleFormat>) {
        self.propagator.set_output_angle_format(angle_format.format);
    }

    /// Get current state vector
    ///
    /// Returns:
    ///     numpy.ndarray: Current state vector in the propagator's output format
    #[pyo3(text_signature = "()")]
    pub fn current_state<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray<f64, Ix1>> {
        let state = self.propagator.current_state();
        state.as_slice().to_pyarray(py).to_owned()
    }

    /// Compute state at a specific epoch
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch for state computation
    ///
    /// Returns:
    ///     numpy.ndarray: State vector in the propagator's current output format
    #[pyo3(text_signature = "(epoch)")]
    pub fn state<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self.propagator.state(epoch.obj);
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in ECI coordinates
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch for state computation
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ECI frame
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_eci<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self.propagator.state_eci(epoch.obj);
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute state at a specific epoch in ECEF coordinates
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch for state computation
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ECEF frame
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_ecef<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        let state = self.propagator.state_ecef(epoch.obj);
        Ok(state.as_slice().to_pyarray(py).to_owned())
    }

    /// Compute states at multiple epochs
    ///
    /// Arguments:
    ///     epochs (list[Epoch]): List of epochs for state computation
    ///
    /// Returns:
    ///     OrbitalTrajectory: Trajectory containing states at requested epochs
    #[pyo3(text_signature = "(epochs)")]
    pub fn states(&self, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<PyOrbitalTrajectory> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let trajectory = self.propagator.states(&epoch_vec);
        Ok(PyOrbitalTrajectory { trajectory })
    }

    /// Compute states at multiple epochs in ECI coordinates
    ///
    /// Arguments:
    ///     epochs (list[Epoch]): List of epochs for state computation
    ///
    /// Returns:
    ///     OrbitalTrajectory: Trajectory containing ECI states at requested epochs
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_eci(&self, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<PyOrbitalTrajectory> {
        let epoch_vec: Vec<_> = epochs.iter().map(|e| e.obj).collect();
        let trajectory = self.propagator.states_eci(&epoch_vec);
        Ok(PyOrbitalTrajectory { trajectory })
    }

    /// Step forward by the default step size
    #[pyo3(text_signature = "()")]
    pub fn step(&mut self) -> PyResult<()> {
        match self.propagator.step() {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Step forward by a specified time duration
    ///
    /// Arguments:
    ///     step_size (float): Time step in seconds
    #[pyo3(text_signature = "(step_size)")]
    pub fn step_by(&mut self, step_size: f64) -> PyResult<()> {
        match self.propagator.step_by(step_size) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Propagate to a specific target epoch
    ///
    /// Arguments:
    ///     target_epoch (Epoch): The epoch to propagate to
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn propagate_to(&mut self, target_epoch: PyRef<PyEpoch>) -> PyResult<()> {
        match self.propagator.propagate_to(target_epoch.obj) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Reset propagator to initial conditions
    #[pyo3(text_signature = "()")]
    pub fn reset(&mut self) -> PyResult<()> {
        match self.propagator.reset() {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get accumulated trajectory
    #[getter]
    pub fn trajectory(&self) -> PyOrbitalTrajectory {
        PyOrbitalTrajectory { trajectory: self.propagator.trajectory().clone() }
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("SGPPropagator(norad_id={}, name={:?}, epoch={:?})",
                self.propagator.norad_id,
                self.propagator.satellite_name,
                self.propagator.initial_epoch())
    }

    /// String conversion
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
    /// Create a new Keplerian propagator from orbital elements
    ///
    /// Arguments:
    ///     epoch (Epoch): Initial epoch
    ///     elements (numpy.ndarray): 6-element orbital elements [a, e, i, Ω, ω, ν]
    ///     frame (OrbitFrame): Reference frame (default: ECI)
    ///     angle_format (AngleFormat): Angle format (default: radians)
    ///     step_size (float): Step size in seconds for propagation (default: 60.0)
    ///
    /// Returns:
    ///     KeplerianPropagator: New propagator instance
    #[new]
    #[pyo3(signature = (epoch, elements, frame=None, angle_format=None, step_size=60.0))]
    pub fn new(
        epoch: PyRef<PyEpoch>,
        elements: PyReadonlyArray1<f64>,
        frame: Option<PyRef<PyOrbitFrame>>,
        angle_format: Option<PyRef<PyAngleFormat>>,
        step_size: Option<f64>,
    ) -> PyResult<Self> {
        let elements_array = elements.as_array();
        if elements_array.len() != 6 {
            return Err(exceptions::PyValueError::new_err(
                "Elements vector must have exactly 6 elements"
            ));
        }

        let elements_vec = na::Vector6::from_row_slice(elements_array.as_slice().unwrap());
        let frame = frame.map(|f| f.frame).unwrap_or(trajectories::OrbitFrame::ECI);
        let angle_format = angle_format.map(|af| af.format).unwrap_or(trajectories::AngleFormat::Radians);
        let step_size = step_size.unwrap_or(60.0);

        let propagator = orbits::KeplerianPropagator::new(
            epoch.obj,
            elements_vec,
            frame,
            trajectories::OrbitRepresentation::Keplerian,
            angle_format,
            step_size,
        ).map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyKeplerianPropagator { propagator })
    }

    /// Create a new Keplerian propagator from Cartesian state
    ///
    /// Arguments:
    ///     epoch (Epoch): Initial epoch
    ///     cartesian_state (numpy.ndarray): 6-element Cartesian state [x, y, z, vx, vy, vz]
    ///     frame (OrbitFrame): Reference frame (default: ECI)
    ///     step_size (float): Step size in seconds for propagation (default: 60.0)
    ///
    /// Returns:
    ///     KeplerianPropagator: New propagator instance
    #[classmethod]
    #[pyo3(signature = (epoch, cartesian_state, frame=None, step_size=60.0))]
    pub fn from_cartesian(
        _cls: &Bound<'_, PyType>,
        epoch: PyRef<PyEpoch>,
        cartesian_state: PyReadonlyArray1<f64>,
        frame: Option<PyRef<PyOrbitFrame>>,
        step_size: Option<f64>,
    ) -> PyResult<Self> {
        let state_array = cartesian_state.as_array();
        if state_array.len() != 6 {
            return Err(exceptions::PyValueError::new_err(
                "State vector must have exactly 6 elements"
            ));
        }

        let state_vec = na::Vector6::from_row_slice(state_array.as_slice().unwrap());
        let frame = frame.map(|f| f.frame).unwrap_or(trajectories::OrbitFrame::ECI);
        let step_size = step_size.unwrap_or(60.0);

        let propagator = orbits::KeplerianPropagator::from_cartesian(
            epoch.obj,
            state_vec,
            frame,
            step_size,
        ).map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

        Ok(PyKeplerianPropagator { propagator })
    }

    /// Get current epoch
    #[getter]
    pub fn current_epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.propagator.current_epoch() }
    }

    /// Get initial epoch
    #[getter]
    pub fn initial_epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.propagator.initial_epoch() }
    }

    /// Get step size in seconds
    #[getter]
    pub fn step_size(&self) -> f64 {
        self.propagator.step_size()
    }

    /// Set step size in seconds
    #[setter]
    pub fn set_step_size(&mut self, step_size: f64) {
        self.propagator.set_step_size(step_size);
    }

    /// Get current state vector
    #[pyo3(text_signature = "()")]
    pub fn current_state<'a>(&self, py: Python<'a>) -> Bound<'a, PyArray<f64, Ix1>> {
        let state = self.propagator.current_state();
        state.as_slice().to_pyarray(py).to_owned()
    }

    /// Step forward by the default step size
    #[pyo3(text_signature = "()")]
    pub fn step(&mut self) -> PyResult<()> {
        match self.propagator.step() {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Propagate to a specific target epoch
    ///
    /// Arguments:
    ///     target_epoch (Epoch): The epoch to propagate to
    #[pyo3(text_signature = "(target_epoch)")]
    pub fn propagate_to(&mut self, target_epoch: PyRef<PyEpoch>) -> PyResult<()> {
        match self.propagator.propagate_to(target_epoch.obj) {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Reset propagator to initial conditions
    #[pyo3(text_signature = "()")]
    pub fn reset(&mut self) -> PyResult<()> {
        match self.propagator.reset() {
            Ok(_) => Ok(()),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get accumulated trajectory
    #[getter]
    pub fn trajectory(&self) -> PyOrbitalTrajectory {
        PyOrbitalTrajectory { trajectory: self.propagator.trajectory().clone() }
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("KeplerianPropagator(epoch={:?})", self.propagator.current_epoch())
    }

    /// String conversion
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// Legacy TLE support (for backward compatibility)
/// Legacy TLE class for backward compatibility
#[pyclass]
#[pyo3(name = "TLE")]
pub struct PyTLE {
    // Minimal implementation using SGPPropagator internally
    propagator: orbits::SGPPropagator,
}

#[pymethods]
impl PyTLE {
    /// Create a TLE from lines (legacy compatibility)
    #[classmethod]
    #[pyo3(signature = (line1, line2, step_size=60.0))]
    pub fn from_lines(_cls: &Bound<'_, PyType>, line1: String, line2: String, step_size: Option<f64>) -> PyResult<Self> {
        let step_size = step_size.unwrap_or(60.0);
        match orbits::SGPPropagator::from_tle(&line1, &line2, step_size) {
            Ok(propagator) => Ok(PyTLE { propagator }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Get NORAD ID
    #[getter]
    pub fn norad_id(&self) -> u32 {
        self.propagator.norad_id
    }

    /// Get TLE epoch
    #[getter]
    pub fn epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.propagator.initial_epoch() }
    }
}

/// Validate TLE lines
#[pyfunction]
#[pyo3(text_signature = "(line1, line2)")]
#[pyo3(name = "validate_tle_lines")]
fn py_validate_tle_lines(line1: String, line2: String) -> PyResult<bool> {
    Ok(orbits::validate_tle_lines(&line1, &line2))
}

/// Validate single TLE line
#[pyfunction]
#[pyo3(text_signature = "(line)")]
#[pyo3(name = "validate_tle_line")]
fn py_validate_tle_line(line: String) -> PyResult<bool> {
    Ok(orbits::validate_tle_line(&line))
}

/// Calculate TLE line checksum
#[pyfunction]
#[pyo3(text_signature = "(line)")]
#[pyo3(name = "calculate_tle_line_checksum")]
fn py_calculate_tle_line_checksum(line: String) -> PyResult<u32> {
    Ok(orbits::calculate_tle_line_checksum(&line) as u32)
}





/// Extract Keplerian orbital elements from TLE lines
///
/// Extracts the standard six Keplerian orbital elements from Two-Line Element (TLE) data.
/// Returns elements in standard order: [a, e, i, raan, argp, M] where angles are in radians.
///
/// Arguments:
///     line1 (`str`): First line of TLE data
///     line2 (`str`): Second line of TLE data
///
/// Returns:
///     elements (`numpy.ndarray`): Six Keplerian elements [a, e, i, raan, argp, M]
///         - a: Semi-major axis (m)
///         - e: Eccentricity (dimensionless)
///         - i: Inclination (rad)
///         - raan: Right ascension of ascending node (rad)
///         - argp: Argument of periapsis (rad)
///         - M: Mean anomaly (rad)
///     epoch (`Epoch`): Epoch of the TLE data
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

/// Convert Keplerian elements to TLE lines
///
/// Converts standard Keplerian orbital elements to Two-Line Element (TLE) format.
/// Input angles should be in degrees for compatibility with TLE format.
///
/// Arguments:
///     epoch (`Epoch`): Epoch of the elements
///     elements (`array`): Keplerian elements [a (m), e, i (deg), Ω (deg), ω (deg), M (deg)]
///     norad_id (`str`): NORAD catalog number (supports numeric and Alpha-5 format)
///
/// Returns:
///     tuple: (line1, line2) - The two TLE lines as strings
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

/// Create complete TLE lines from all parameters
///
/// Creates Two-Line Element (TLE) lines from complete set of orbital and administrative parameters.
/// Provides full control over all TLE fields including derivatives and drag terms.
///
/// Arguments:
///     epoch (`Epoch`): Epoch of the elements
///     inclination (`float`): Inclination (degrees)
///     raan (`float`): Right ascension of ascending node (degrees)
///     eccentricity (`float`): Eccentricity (dimensionless)
///     arg_perigee (`float`): Argument of periapsis (degrees)
///     mean_anomaly (`float`): Mean anomaly (degrees)
///     mean_motion (`float`): Mean motion (revolutions per day)
///     norad_id (`str`): NORAD catalog number (supports numeric and Alpha-5 format)
///     ephemeris_type (`int`): Ephemeris type (0-9)
///     element_set_number (`int`): Element set number
///     revolution_number (`int`): Revolution number at epoch
///     classification (`str`, optional): Security classification (default: ' ')
///     intl_designator (`str`, optional): International designator (default: '')
///     first_derivative (`float`, optional): First derivative of mean motion (default: 0.0)
///     second_derivative (`float`, optional): Second derivative of mean motion (default: 0.0)
///     bstar (`float`, optional): BSTAR drag term (default: 0.0)
///
/// Returns:
///     tuple: (line1, line2) - The two TLE lines as strings
#[pyfunction]
#[pyo3(text_signature = "(epoch, inclination, raan, eccentricity, arg_perigee, mean_anomaly, mean_motion, norad_id, ephemeris_type, element_set_number, revolution_number, classification=None, intl_designator=None, first_derivative=None, second_derivative=None, bstar=None)")]
#[pyo3(name = "create_tle_lines")]
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

/// Parse NORAD ID from string, handling both classic and Alpha-5 formats
///
/// # Arguments
/// * `norad_str` - NORAD ID string from TLE
///
/// # Returns
/// * `u32` - Parsed numeric NORAD ID
#[pyfunction]
#[pyo3(text_signature = "(norad_str)")]
#[pyo3(name = "parse_norad_id")]
fn py_parse_norad_id(norad_str: String) -> PyResult<u32> {
    match orbits::parse_norad_id(&norad_str) {
        Ok(id) => Ok(id),
        Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

/// Convert numeric NORAD ID to Alpha-5 format
///
/// # Arguments
/// * `norad_id` - Numeric NORAD ID (100000-339999)
///
/// # Returns
/// * `str` - Alpha-5 format ID (e.g., "A0001")
#[pyfunction]
#[pyo3(text_signature = "(norad_id)")]
#[pyo3(name = "norad_id_numeric_to_alpha5")]
fn py_norad_id_numeric_to_alpha5(norad_id: u32) -> PyResult<String> {
    match orbits::norad_id_numeric_to_alpha5(norad_id) {
        Ok(alpha5_id) => Ok(alpha5_id),
        Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

/// Convert Alpha-5 NORAD ID to numeric format
///
/// # Arguments
/// * `alpha5_id` - Alpha-5 format ID (e.g., "A0001")
///
/// # Returns
/// * `int` - Numeric NORAD ID
#[pyfunction]
#[pyo3(text_signature = "(alpha5_id)")]
#[pyo3(name = "norad_id_alpha5_to_numeric")]
fn py_norad_id_alpha5_to_numeric(alpha5_id: String) -> PyResult<u32> {
    match orbits::norad_id_alpha5_to_numeric(&alpha5_id) {
        Ok(numeric_id) => Ok(numeric_id),
        Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}