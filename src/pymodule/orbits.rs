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