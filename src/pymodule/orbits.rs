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

// TLE (Two-Line Element) Support

/// Python wrapper for TLE (Two-Line Element) object with SGP4 propagation capability
#[pyclass]
#[pyo3(name = "TLE")]
pub struct PyTLE {
    tle: crate::TLE,
}

#[pymethods]
impl PyTLE {
    /// Create a new TLE from 2-line format
    ///
    /// Arguments:
    ///     line1 (str): First line of TLE data
    ///     line2 (str): Second line of TLE data
    ///
    /// Returns:
    ///     TLE: New TLE instance
    #[classmethod]
    #[pyo3(text_signature = "(cls, line1, line2)")]
    pub fn from_lines(_cls: &Bound<'_, PyType>, line1: String, line2: String) -> PyResult<Self> {
        match crate::TLE::from_lines(&line1, &line2) {
            Ok(tle) => Ok(PyTLE { tle }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(format!("{}", e))),
        }
    }

    /// Create a new TLE from 3-line format (with satellite name)
    ///
    /// Arguments:
    ///     name (str): Satellite name (line 0)
    ///     line1 (str): First line of TLE data
    ///     line2 (str): Second line of TLE data
    ///
    /// Returns:
    ///     TLE: New TLE instance
    #[classmethod]
    #[pyo3(text_signature = "(cls, name, line1, line2)")]
    pub fn from_3le(_cls: &Bound<'_, PyType>, name: String, line1: String, line2: String) -> PyResult<Self> {
        match crate::TLE::from_3le(&name, &line1, &line2) {
            Ok(tle) => Ok(PyTLE { tle }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(format!("{}", e))),
        }
    }

    /// Create TLE from raw TLE string (auto-detects format)
    ///
    /// Arguments:
    ///     tle_string (str): Raw TLE data as string
    ///
    /// Returns:
    ///     TLE: New TLE instance
    #[classmethod]
    #[pyo3(text_signature = "(cls, tle_string)")]
    pub fn from_tle_string(_cls: &Bound<'_, PyType>, tle_string: String) -> PyResult<Self> {
        match crate::TLE::from_tle_string(&tle_string) {
            Ok(tle) => Ok(PyTLE { tle }),
            Err(e) => Err(exceptions::PyRuntimeError::new_err(format!("{}", e))),
        }
    }

    /// Get satellite name (if available)
    ///
    /// Returns:
    ///     Optional[str]: Satellite name or None if not set
    #[getter]
    pub fn satellite_name(&self) -> Option<String> {
        self.tle.satellite_name().map(|s| s.to_string())
    }

    /// Get original NORAD ID string (may be Alpha-5 format)
    ///
    /// Returns:
    ///     str: Original NORAD ID string
    #[getter]
    pub fn norad_id_string(&self) -> String {
        self.tle.norad_id_string().to_string()
    }

    /// Get decoded numeric NORAD ID
    ///
    /// Returns:
    ///     int: Numeric NORAD ID
    #[getter]
    pub fn norad_id(&self) -> u32 {
        self.tle.norad_id()
    }

    /// Get international designator
    ///
    /// Returns:
    ///     Optional[str]: International designator or None if not set
    #[getter]
    pub fn international_designator(&self) -> Option<String> {
        self.tle.international_designator()
    }

    /// Get epoch of TLE
    ///
    /// Returns:
    ///     Epoch: TLE epoch
    #[getter]
    pub fn epoch(&self) -> PyEpoch {
        PyEpoch { obj: self.tle.epoch() }
    }

    /// Get mean motion (revolutions per day)
    ///
    /// Returns:
    ///     float: Mean motion in rev/day
    #[getter]
    pub fn mean_motion(&self) -> f64 {
        self.tle.mean_motion()
    }

    /// Get eccentricity
    ///
    /// Returns:
    ///     float: Orbital eccentricity
    #[getter]
    pub fn eccentricity(&self) -> f64 {
        self.tle.eccentricity()
    }

    /// Get inclination in radians or degrees
    ///
    /// Arguments:
    ///     as_degrees (bool): Return in degrees if True, radians if False
    ///
    /// Returns:
    ///     float: Orbital inclination
    #[pyo3(text_signature = "(as_degrees)")]
    pub fn inclination(&self, as_degrees: bool) -> f64 {
        let inc_rad = self.tle.inclination();
        if as_degrees {
            inc_rad.to_degrees()
        } else {
            inc_rad
        }
    }

    /// Get right ascension of ascending node in radians or degrees
    ///
    /// Arguments:
    ///     as_degrees (bool): Return in degrees if True, radians if False
    ///
    /// Returns:
    ///     float: Right ascension of ascending node
    #[pyo3(text_signature = "(as_degrees)")]
    pub fn raan(&self, as_degrees: bool) -> f64 {
        let raan_rad = self.tle.raan();
        if as_degrees {
            raan_rad.to_degrees()
        } else {
            raan_rad
        }
    }

    /// Get argument of perigee in radians or degrees
    ///
    /// Arguments:
    ///     as_degrees (bool): Return in degrees if True, radians if False
    ///
    /// Returns:
    ///     float: Argument of perigee
    #[pyo3(text_signature = "(as_degrees)")]
    pub fn argument_of_perigee(&self, as_degrees: bool) -> f64 {
        let arg_per_rad = self.tle.argument_of_perigee();
        if as_degrees {
            arg_per_rad.to_degrees()
        } else {
            arg_per_rad
        }
    }

    /// Get mean anomaly in radians or degrees
    ///
    /// Arguments:
    ///     as_degrees (bool): Return in degrees if True, radians if False
    ///
    /// Returns:
    ///     float: Mean anomaly
    #[pyo3(text_signature = "(as_degrees)")]
    pub fn mean_anomaly(&self, as_degrees: bool) -> f64 {
        let mean_anom_rad = self.tle.mean_anomaly();
        if as_degrees {
            mean_anom_rad.to_degrees()
        } else {
            mean_anom_rad
        }
    }

    /// Check if TLE uses Alpha-5 format
    ///
    /// Returns:
    ///     bool: True if Alpha-5 format, False if classic format
    #[getter]
    pub fn is_alpha5(&self) -> bool {
        self.tle.is_alpha5()
    }

    /// Propagate to specific time and return Cartesian state
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch for propagation
    ///
    /// Returns:
    ///     numpy.ndarray: Cartesian state vector [x, y, z, vx, vy, vz] in meters and m/s
    #[pyo3(text_signature = "(epoch)")]
    pub fn propagate<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        match self.tle.propagate(epoch.obj) {
            Ok(orbit_state) => {
                // Extract state vector (position and velocity)
                let state_vec = orbit_state.state;
                let flat_vec: Vec<f64> = (0..6).map(|i| state_vec[i]).collect();
                Ok(flat_vec.into_pyarray(py))
            },
            Err(e) => Err(exceptions::PyRuntimeError::new_err(format!("{}", e))),
        }
    }

    /// Get state at given epoch in propagator's default frame (analytical computation)
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch for state computation
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in meters and m/s
    #[pyo3(text_signature = "(epoch)")]
    pub fn state<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        use crate::orbits::traits::AnalyticPropagator;
        let state_vec = self.tle.state(epoch.obj);
        let flat_vec: Vec<f64> = (0..6).map(|i| state_vec[i]).collect();
        Ok(flat_vec.into_pyarray(py))
    }

    /// Get state at given epoch in ECI coordinates (analytical computation)
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch for state computation
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ECI frame in meters and m/s
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_eci<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        use crate::orbits::traits::AnalyticPropagator;
        let state_vec = self.tle.state_eci(epoch.obj);
        let flat_vec: Vec<f64> = (0..6).map(|i| state_vec[i]).collect();
        Ok(flat_vec.into_pyarray(py))
    }

    /// Get state at given epoch in ECEF coordinates (analytical computation)
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch for state computation
    ///
    /// Returns:
    ///     numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ECEF frame in meters and m/s
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_ecef<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        use crate::orbits::traits::AnalyticPropagator;
        let state_vec = self.tle.state_ecef(epoch.obj);
        let flat_vec: Vec<f64> = (0..6).map(|i| state_vec[i]).collect();
        Ok(flat_vec.into_pyarray(py))
    }

    /// Get state at given epoch as osculating orbital elements (analytical computation)
    ///
    /// Arguments:
    ///     epoch (Epoch): Target epoch for state computation
    ///
    /// Returns:
    ///     numpy.ndarray: Osculating elements [a, e, i, Ω, ω, M] where angles are in radians
    #[pyo3(text_signature = "(epoch)")]
    pub fn state_osculating_elements<'a>(&self, py: Python<'a>, epoch: &PyEpoch) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
        use crate::orbits::traits::AnalyticPropagator;
        let elements_vec = self.tle.state_osculating_elements(epoch.obj);
        let flat_vec: Vec<f64> = (0..6).map(|i| elements_vec[i]).collect();
        Ok(flat_vec.into_pyarray(py))
    }

    /// Get states at multiple epochs in propagator's default frame (analytical computation)
    ///
    /// Arguments:
    ///     epochs (list[Epoch]): List of target epochs for state computation
    ///
    /// Returns:
    ///     numpy.ndarray: Array of state vectors, shape (N, 6)
    #[pyo3(text_signature = "(epochs)")]
    pub fn states<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<Bound<'a, PyArray<f64, Ix2>>> {
        use crate::orbits::traits::AnalyticPropagator;
        let epoch_vec: Vec<crate::Epoch> = epochs.iter().map(|e| e.obj).collect();
        let trajectory = self.tle.states(&epoch_vec);
        let matrix = trajectory.to_matrix().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        // Convert matrix (6, N) to array (N, 6) for Python
        let n_epochs = epoch_vec.len();
        let mut flat_vec = Vec::with_capacity(n_epochs * 6);
        for col_idx in 0..n_epochs {
            for row_idx in 0..6 {
                flat_vec.push(matrix[(row_idx, col_idx)]);
            }
        }
        
        Ok(flat_vec.into_pyarray(py).reshape([n_epochs, 6]).unwrap())
    }

    /// Get states at multiple epochs in ECI coordinates (analytical computation)
    ///
    /// Arguments:
    ///     epochs (list[Epoch]): List of target epochs for state computation
    ///
    /// Returns:
    ///     numpy.ndarray: Array of state vectors in ECI frame, shape (N, 6)
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_eci<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<Bound<'a, PyArray<f64, Ix2>>> {
        use crate::orbits::traits::AnalyticPropagator;
        let epoch_vec: Vec<crate::Epoch> = epochs.iter().map(|e| e.obj).collect();
        let trajectory = self.tle.states_eci(&epoch_vec);
        let matrix = trajectory.to_matrix().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        // Convert matrix (6, N) to array (N, 6) for Python
        let n_epochs = epoch_vec.len();
        let mut flat_vec = Vec::with_capacity(n_epochs * 6);
        for col_idx in 0..n_epochs {
            for row_idx in 0..6 {
                flat_vec.push(matrix[(row_idx, col_idx)]);
            }
        }
        
        Ok(flat_vec.into_pyarray(py).reshape([n_epochs, 6]).unwrap())
    }

    /// Get states at multiple epochs in ECEF coordinates (analytical computation)
    ///
    /// Arguments:
    ///     epochs (list[Epoch]): List of target epochs for state computation
    ///
    /// Returns:
    ///     numpy.ndarray: Array of state vectors in ECEF frame, shape (N, 6)
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_ecef<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<Bound<'a, PyArray<f64, Ix2>>> {
        use crate::orbits::traits::AnalyticPropagator;
        let epoch_vec: Vec<crate::Epoch> = epochs.iter().map(|e| e.obj).collect();
        let trajectory = self.tle.states_ecef(&epoch_vec);
        let matrix = trajectory.to_matrix().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        // Convert matrix (6, N) to array (N, 6) for Python
        let n_epochs = epoch_vec.len();
        let mut flat_vec = Vec::with_capacity(n_epochs * 6);
        for col_idx in 0..n_epochs {
            for row_idx in 0..6 {
                flat_vec.push(matrix[(row_idx, col_idx)]);
            }
        }
        
        Ok(flat_vec.into_pyarray(py).reshape([n_epochs, 6]).unwrap())
    }

    /// Get states at multiple epochs as osculating orbital elements (analytical computation)
    ///
    /// Arguments:
    ///     epochs (list[Epoch]): List of target epochs for state computation
    ///
    /// Returns:
    ///     numpy.ndarray: Array of osculating elements, shape (N, 6), angles in radians
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_osculating_elements<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<Bound<'a, PyArray<f64, Ix2>>> {
        use crate::orbits::traits::AnalyticPropagator;
        let epoch_vec: Vec<crate::Epoch> = epochs.iter().map(|e| e.obj).collect();
        let trajectory = self.tle.states_osculating_elements(&epoch_vec);
        let matrix = trajectory.to_matrix().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        // Convert matrix (6, N) to array (N, 6) for Python
        let n_epochs = epoch_vec.len();
        let mut flat_vec = Vec::with_capacity(n_epochs * 6);
        for col_idx in 0..n_epochs {
            for row_idx in 0..6 {
                flat_vec.push(matrix[(row_idx, col_idx)]);
            }
        }
        
        Ok(flat_vec.into_pyarray(py).reshape([n_epochs, 6]).unwrap())
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!("TLE(norad_id={}, name={:?}, epoch={:?})", 
                self.tle.norad_id(), 
                self.tle.satellite_name(), 
                self.tle.epoch())
    }

    /// String conversion
    fn __str__(&self) -> String {
        self.__repr__()
    }
}

// Independent TLE utility functions

/// Validate TLE line format
///
/// Arguments:
///     line1 (str): First line of TLE data
///     line2 (str): Second line of TLE data
///
/// Returns:
///     bool: True if validation passes, False otherwise
#[pyfunction]
#[pyo3(text_signature = "(line1, line2)")]
#[pyo3(name = "validate_tle_lines")]
fn py_validate_tle_lines(line1: String, line2: String) -> bool {
    crate::orbits::tle::validate_tle_lines(&line1, &line2)
}

/// Validate single TLE line format
///
/// Arguments:
///     line (str): TLE line to validate
///     expected_line_number (int): Expected line number (1 or 2)
///
/// Returns:
///     bool: True if validation passes, False otherwise
#[pyfunction]
#[pyo3(text_signature = "(line, expected_line_number)")]
#[pyo3(name = "validate_tle_line")]
fn py_validate_tle_line(line: String, expected_line_number: u8) -> bool {
    crate::orbits::tle::validate_tle_line(&line, expected_line_number)
}

/// Calculate TLE line checksum
///
/// Arguments:
///     line (str): First 68 characters of TLE line (without checksum)
///
/// Returns:
///     int: Calculated checksum digit (0-9)
#[pyfunction]
#[pyo3(text_signature = "(line)")]
#[pyo3(name = "calculate_tle_line_checksum")]
fn py_calculate_tle_line_checksum(line: String) -> PyResult<u8> {
    Ok(crate::orbits::tle::calculate_tle_line_checksum(&line))
}

/// Extract NORAD ID from string, handling both classic and Alpha-5 formats
///
/// Arguments:
///     id_str (str): 5-character NORAD ID string (numeric or Alpha-5)
///
/// Returns:
///     int: Decoded numeric NORAD ID
///
/// Raises:
///     RuntimeError: If decoding fails
#[pyfunction]
#[pyo3(text_signature = "(id_str)")]
#[pyo3(name = "extract_tle_norad_id")]
fn py_extract_tle_norad_id(id_str: String) -> PyResult<u32> {
    match crate::orbits::tle::extract_tle_norad_id(&id_str) {
        Ok(id) => Ok(id),
        Err(e) => Err(exceptions::PyRuntimeError::new_err(format!("{}", e))),
    }
}

/// Extract epoch from SGP4 elements
///
/// Arguments:
///     elements: SGP4 elements structure (internal use)
///
/// Returns:
///     Epoch: Extracted epoch
///
/// Note: This function is primarily for internal use
#[pyfunction]
#[pyo3(text_signature = "(elements)")]
#[pyo3(name = "extract_epoch")]
fn py_extract_epoch(_elements: PyObject) -> PyResult<PyEpoch> {
    // This function is mainly for internal use and would require
    // exposing SGP4 elements to Python, which is complex.
    // For now, we'll make it a placeholder that suggests using TLE.epoch() instead
    Err(exceptions::PyNotImplementedError::new_err(
        "Use TLE.epoch property instead for extracting epochs from TLE data"
    ))
}

/// Convert TLE lines to orbital elements
///
/// Arguments:
///     line1 (str): First line of TLE data
///     line2 (str): Second line of TLE data
///
/// Returns:
///     numpy.ndarray: Orbital elements [a, e, i, Ω, ω, M] in SI units (meters, radians)
///
/// Raises:
///     RuntimeError: If parsing fails
#[pyfunction]
#[pyo3(text_signature = "(line1, line2)")]
#[pyo3(name = "lines_to_orbit_elements")]
fn py_lines_to_orbit_elements<'a>(py: Python<'a>, line1: String, line2: String) -> PyResult<Bound<'a, PyArray<f64, Ix1>>> {
    match crate::orbits::tle::lines_to_orbit_elements(&line1, &line2) {
        Ok(elements) => {
            let flat_vec: Vec<f64> = (0..6).map(|i| elements[i]).collect();
            Ok(flat_vec.into_pyarray(py))
        },
        Err(e) => Err(exceptions::PyRuntimeError::new_err(format!("{}", e))),
    }
}

/// Convert TLE lines to OrbitState
///
/// Arguments:
///     line1 (str): First line of TLE data
///     line2 (str): Second line of TLE data
///
/// Returns:
///     dict: Dictionary containing orbit state information with keys:
///           - epoch: Epoch object
///           - elements: numpy array of orbital elements [a, e, i, Ω, ω, M]
///           - frame: Orbital frame ('ECI')
///           - orbit_type: Orbit state type ('TLEMean')
///
/// Raises:
///     RuntimeError: If parsing fails
#[pyfunction]
#[pyo3(text_signature = "(line1, line2)")]
#[pyo3(name = "lines_to_orbit_state")]
fn py_lines_to_orbit_state(py: Python, line1: String, line2: String) -> PyResult<PyObject> {
    match crate::orbits::tle::lines_to_orbit_state(&line1, &line2) {
        Ok(orbit_state) => {
            use pyo3::types::PyDict;
            let dict = PyDict::new(py);
            
            // Add epoch
            let py_epoch = PyEpoch { obj: orbit_state.epoch };
            dict.set_item("epoch", py_epoch)?;
            
            // Add elements as numpy array
            let elements_vec: Vec<f64> = (0..6).map(|i| orbit_state.state[i]).collect();
            let elements_array = elements_vec.into_pyarray(py);
            dict.set_item("elements", elements_array)?;
            
            // Add frame and orbit type as strings
            dict.set_item("frame", "ECI")?;
            dict.set_item("orbit_type", "TLEMean")?;
            
            Ok(dict.into())
        },
        Err(e) => Err(exceptions::PyRuntimeError::new_err(format!("{}", e))),
    }
}

/// Python wrapper for KeplerianPropagator
#[pyclass]
#[pyo3(name = "KeplerianPropagator")]
pub struct PyKeplerianPropagator {
    propagator: crate::orbits::KeplerianPropagator,
}

#[pymethods]
impl PyKeplerianPropagator {
    /// Create a new Keplerian propagator from orbital elements
    /// 
    /// Arguments:
    ///     epoch (Epoch): Initial epoch
    ///     elements (numpy.ndarray): Keplerian elements [a, e, i, raan, argp, anomaly] (km, rad)
    ///     frame (str): Reference frame ("ECI" or "ECEF")
    ///     angle_format (str): Angular format ("radians" or "degrees")
    /// 
    /// Returns:
    ///     KeplerianPropagator: New propagator instance
    #[new]
    #[pyo3(text_signature = "(epoch, elements, frame='ECI', angle_format='radians')")]
    pub fn new(
        epoch: PyRef<PyEpoch>,
        elements: PyReadonlyArray1<f64>,
        frame: Option<&str>,
        angle_format: Option<&str>,
    ) -> PyResult<Self> {
        use crate::trajectories::{AngleFormat, OrbitFrame};
        
        let elements_array = elements.as_array();
        if elements_array.len() != 6 {
            return Err(exceptions::PyValueError::new_err(
                "Elements array must have exactly 6 elements"
            ));
        }
        
        let elements_vec = nalgebra::Vector6::from_row_slice(elements_array.as_slice().unwrap());
        
        let frame = match frame.unwrap_or("ECI") {
            "ECI" => OrbitFrame::ECI,
            "ECEF" => OrbitFrame::ECEF,
            _ => return Err(exceptions::PyValueError::new_err(
                "Frame must be 'ECI' or 'ECEF'"
            )),
        };
        
        let angle_format = match angle_format.unwrap_or("radians") {
            "radians" => AngleFormat::Radians,
            "degrees" => AngleFormat::Degrees,
            _ => return Err(exceptions::PyValueError::new_err(
                "Angle format must be 'radians' or 'degrees'"
            )),
        };
        
        let propagator = crate::orbits::KeplerianPropagator::new(
            epoch.obj,
            elements_vec,
            frame,
            angle_format,
        ).map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(PyKeplerianPropagator { propagator })
    }
    
    /// Create a new Keplerian propagator from Cartesian state
    /// 
    /// Arguments:
    ///     epoch (Epoch): Initial epoch
    ///     cartesian_state (numpy.ndarray): Cartesian state [x, y, z, vx, vy, vz] (km, km/s)
    ///     frame (str): Reference frame ("ECI" or "ECEF")
    /// 
    /// Returns:
    ///     KeplerianPropagator: New propagator instance
    #[staticmethod]
    #[pyo3(text_signature = "(epoch, cartesian_state, frame='ECI')")]
    pub fn from_cartesian(
        epoch: PyRef<PyEpoch>,
        cartesian_state: PyReadonlyArray1<f64>,
        frame: Option<&str>,
    ) -> PyResult<Self> {
        use crate::trajectories::OrbitFrame;
        
        let state_array = cartesian_state.as_array();
        if state_array.len() != 6 {
            return Err(exceptions::PyValueError::new_err(
                "Cartesian state array must have exactly 6 elements"
            ));
        }
        
        let state_vec = nalgebra::Vector6::from_row_slice(state_array.as_slice().unwrap());
        
        let frame = match frame.unwrap_or("ECI") {
            "ECI" => OrbitFrame::ECI,
            "ECEF" => OrbitFrame::ECEF,
            _ => return Err(exceptions::PyValueError::new_err(
                "Frame must be 'ECI' or 'ECEF'"
            )),
        };
        
        let propagator = crate::orbits::KeplerianPropagator::from_cartesian(
            epoch.obj,
            state_vec,
            frame,
        ).map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(PyKeplerianPropagator { propagator })
    }
    
    /// Get states at multiple epochs in ECI coordinates
    ///
    /// Arguments:
    ///     epochs (list[Epoch]): List of target epochs
    ///
    /// Returns:
    ///     numpy.ndarray: Array of state vectors in ECI frame, shape (N, 6)
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_eci<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<Bound<'a, PyArray<f64, Ix2>>> {
        use crate::orbits::traits::AnalyticPropagator;
        let epoch_vec: Vec<crate::Epoch> = epochs.iter().map(|e| e.obj).collect();
        let trajectory = self.propagator.states_eci(&epoch_vec);
        let matrix = trajectory.to_matrix().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        // Convert matrix (6, N) to array (N, 6) for Python
        let n_epochs = epoch_vec.len();
        let mut flat_vec = Vec::with_capacity(n_epochs * 6);
        for col_idx in 0..n_epochs {
            for row_idx in 0..6 {
                flat_vec.push(matrix[(row_idx, col_idx)]);
            }
        }
        
        Ok(flat_vec.into_pyarray(py).reshape([n_epochs, 6]).unwrap())
    }
    
    /// Get states at multiple epochs in ECEF coordinates
    ///
    /// Arguments:
    ///     epochs (list[Epoch]): List of target epochs
    ///
    /// Returns:
    ///     numpy.ndarray: Array of state vectors in ECEF frame, shape (N, 6)
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_ecef<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<Bound<'a, PyArray<f64, Ix2>>> {
        use crate::orbits::traits::AnalyticPropagator;
        let epoch_vec: Vec<crate::Epoch> = epochs.iter().map(|e| e.obj).collect();
        let trajectory = self.propagator.states_ecef(&epoch_vec);
        let matrix = trajectory.to_matrix().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        // Convert matrix (6, N) to array (N, 6) for Python
        let n_epochs = epoch_vec.len();
        let mut flat_vec = Vec::with_capacity(n_epochs * 6);
        for col_idx in 0..n_epochs {
            for row_idx in 0..6 {
                flat_vec.push(matrix[(row_idx, col_idx)]);
            }
        }
        
        Ok(flat_vec.into_pyarray(py).reshape([n_epochs, 6]).unwrap())
    }
    
    /// Get states at multiple epochs as osculating elements
    ///
    /// Arguments:
    ///     epochs (list[Epoch]): List of target epochs
    ///
    /// Returns:
    ///     numpy.ndarray: Array of osculating elements, shape (N, 6), angles in radians
    #[pyo3(text_signature = "(epochs)")]
    pub fn states_osculating_elements<'a>(&self, py: Python<'a>, epochs: Vec<PyRef<PyEpoch>>) -> PyResult<Bound<'a, PyArray<f64, Ix2>>> {
        use crate::orbits::traits::AnalyticPropagator;
        let epoch_vec: Vec<crate::Epoch> = epochs.iter().map(|e| e.obj).collect();
        let trajectory = self.propagator.states_osculating_elements(&epoch_vec);
        let matrix = trajectory.to_matrix().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        // Convert matrix (6, N) to array (N, 6) for Python
        let n_epochs = epoch_vec.len();
        let mut flat_vec = Vec::with_capacity(n_epochs * 6);
        for col_idx in 0..n_epochs {
            for row_idx in 0..6 {
                flat_vec.push(matrix[(row_idx, col_idx)]);
            }
        }
        
        Ok(flat_vec.into_pyarray(py).reshape([n_epochs, 6]).unwrap())
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