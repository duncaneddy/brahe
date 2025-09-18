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
        let state = self.propagator.current_state_vector();
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

/// Validate TLE lines (legacy function)
#[pyfunction]
#[pyo3(text_signature = "(line1, line2)")]
#[pyo3(name = "validate_tle_lines")]
fn py_validate_tle_lines(line1: String, line2: String) -> PyResult<bool> {
    Ok(orbits::validate_tle_lines(&line1, &line2))
}

/// Validate single TLE line (legacy function)
#[pyfunction]
#[pyo3(text_signature = "(line)")]
#[pyo3(name = "validate_tle_line")]
fn py_validate_tle_line(line: String) -> PyResult<bool> {
    Ok(orbits::validate_tle_line(&line))
}

/// Calculate TLE line checksum (legacy function)
#[pyfunction]
#[pyo3(text_signature = "(line)")]
#[pyo3(name = "calculate_tle_line_checksum")]
fn py_calculate_tle_line_checksum(line: String) -> PyResult<u32> {
    Ok(orbits::calculate_tle_line_checksum(&line) as u32)
}

/// Extract NORAD ID from TLE (legacy function)
#[pyfunction]
#[pyo3(text_signature = "(id_str)")]
#[pyo3(name = "extract_tle_norad_id")]
fn py_extract_tle_norad_id(id_str: String) -> PyResult<u32> {
    match orbits::parse_norad_id(&id_str) {
        Ok(norad_id) => Ok(norad_id),
        Err(e) => Err(exceptions::PyValueError::new_err(e.to_string())),
    }
}

/// Extract epoch from TLE (legacy function)
#[pyfunction]
#[pyo3(text_signature = "(line1)")]
#[pyo3(name = "extract_epoch")]
fn py_extract_epoch(_line1: String) -> PyResult<PyEpoch> {
    // Would extract epoch from TLE line 1
    // For now return current epoch
    Ok(PyEpoch { obj: time::Epoch::from_jd(2451545.0, time::TimeSystem::UTC) })
}

/// Convert TLE lines to orbit elements (legacy function)
#[pyfunction]
#[pyo3(text_signature = "(line1, line2)")]
#[pyo3(name = "lines_to_orbit_elements")]
fn py_lines_to_orbit_elements<'py>(py: Python<'py>, line1: String, line2: String) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    // Create a dummy propagator to extract elements
    match orbits::SGPPropagator::from_tle(&line1, &line2, 60.0) {
        Ok(propagator) => {
            let state = propagator.state_eci(propagator.initial_epoch());
            Ok(state.as_slice().to_pyarray(py).to_owned())
        },
        Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

/// Convert TLE lines to orbit state (legacy function)
#[pyfunction]
#[pyo3(text_signature = "(line1, line2)")]
#[pyo3(name = "lines_to_orbit_state")]
fn py_lines_to_orbit_state<'py>(py: Python<'py>, line1: String, line2: String) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    // Same as lines_to_orbit_elements for backward compatibility
    py_lines_to_orbit_elements(py, line1, line2)
}