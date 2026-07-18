/// Computes the orbital period of an object around Earth.
///
/// Uses rastro.constants.GM_EARTH as the standard gravitational parameter for the calculation.
///
/// Args:
///     a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` will be extracted.
///
/// Returns:
///     float: The orbital period of the astronomical object in seconds.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar semi-major axis
///     a = bh.R_EARTH + 400e3
///     period = bh.orbital_period(a)
///     print(f"Orbital period: {period/60:.2f} minutes")
///
///     # Using Keplerian elements vector
///     oe = [bh.R_EARTH + 400e3, 0.001, np.radians(51.6), 0, 0, 0]
///     period = bh.orbital_period(oe)
///     print(f"Orbital period: {period/60:.2f} minutes")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(a_or_oe)")]
#[pyo3(name = "orbital_period")]
fn py_orbital_period(a_or_oe: &Bound<'_, PyAny>) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(a) = a_or_oe.extract::<f64>() {
        return Ok(orbits::orbital_period(a));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(a_or_oe, Some(6))?;
    Ok(orbits::orbital_period(oe[0]))
}

/// Computes the orbital period of an astronomical object around a general body.
///
/// Args:
///     a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` will be extracted.
///     gm (float): (keyword-only) The standard gravitational parameter of primary body in m³/s².
///
/// Returns:
///     float: The orbital period of the astronomical object in seconds.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar semi-major axis
///     a = 1900000.0  # 1900 km semi-major axis
///     period = bh.orbital_period_general(a, bh.GM_MOON)
///     print(f"Lunar orbital period: {period/3600:.2f} hours")
///
///     # Using Keplerian elements vector
///     oe = [1900000.0, 0.01, np.radians(45), 0, 0, 0]
///     period = bh.orbital_period_general(oe, bh.GM_MOON)
///     print(f"Lunar orbital period: {period/3600:.2f} hours")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(a_or_oe, gm)")]
#[pyo3(name = "orbital_period_general")]
fn py_orbital_period_general(a_or_oe: &Bound<'_, PyAny>, gm: f64) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(a) = a_or_oe.extract::<f64>() {
        return Ok(orbits::orbital_period_general(a, gm));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(a_or_oe, Some(6))?;
    Ok(orbits::orbital_period_general(oe[0], gm))
}

/// Computes orbital period from an ECI state vector using the vis-viva equation.
///
/// This function uses the vis-viva equation to compute the semi-major axis from the
/// position and velocity, then calculates the orbital period.
///
/// Args:
///     state_eci (np.ndarray): ECI state vector [x, y, z, vx, vy, vz] in meters and meters/second.
///     gm (float): Gravitational parameter in m³/s². Use GM_EARTH for Earth orbits.
///
/// Returns:
///     float: Orbital period in seconds.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Create a circular orbit state at 500 km altitude
///     r = bh.R_EARTH + 500e3
///     v = np.sqrt(bh.GM_EARTH / r)
///     state_eci = np.array([r, 0, 0, 0, v, 0])
///
///     # Compute orbital period from state
///     period = bh.orbital_period_from_state(state_eci, bh.GM_EARTH)
///     print(f"Period: {period/60:.2f} minutes")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(state_eci, gm)")]
#[pyo3(name = "orbital_period_from_state")]
fn py_orbital_period_from_state(
    _py: Python,
    state_eci: PyReadonlyArray1<f64>,
    gm: f64,
) -> PyResult<f64> {
    let state = state_eci.as_array();
    if state.len() != 6 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "state_eci must be a 6-element array [x, y, z, vx, vy, vz]",
        ));
    }

    let state_vec = nalgebra::Vector6::from_iterator(state.iter().copied());
    Ok(orbits::orbital_period_from_state(&state_vec, gm))
}

/// Computes the mean motion of an astronomical object around Earth.
///
/// Args:
///     a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` will be extracted.
///     angle_format (AngleFormat): (keyword-only) Return output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: The mean motion of the astronomical object in radians or degrees.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar semi-major axis
///     a = bh.R_EARTH + 35786e3
///     n = bh.mean_motion(a, bh.AngleFormat.DEGREES)
///     print(f"Mean motion: {n:.6f} deg/s")
///
///     # Using Keplerian elements vector
///     oe = [bh.R_EARTH + 35786e3, 0.001, np.radians(0), 0, 0, 0]
///     n = bh.mean_motion(oe, bh.AngleFormat.DEGREES)
///     print(f"Mean motion: {n:.6f} deg/s")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(a_or_oe, angle_format)")]
#[pyo3(name = "mean_motion")]
fn py_mean_motion(a_or_oe: &Bound<'_, PyAny>, angle_format: &PyAngleFormat) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(a) = a_or_oe.extract::<f64>() {
        return Ok(orbits::mean_motion(a, angle_format.value));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(a_or_oe, Some(6))?;
    Ok(orbits::mean_motion(oe[0], angle_format.value))
}

/// Computes the mean motion of an astronomical object around a general body
/// given a semi-major axis.
///
/// Args:
///     a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` will be extracted.
///     gm (float): (keyword-only) The standard gravitational parameter of primary body in m³/s².
///     angle_format (AngleFormat): (keyword-only) Return output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: The mean motion of the astronomical object in radians or degrees.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar semi-major axis
///     a = 4000000.0  # 4000 km semi-major axis
///     n = bh.mean_motion_general(a, bh.GM_MARS, bh.AngleFormat.RADIANS)
///     print(f"Mean motion: {n:.6f} rad/s")
///
///     # Using Keplerian elements vector
///     oe = [4000000.0, 0.01, np.radians(30), 0, 0, 0]
///     n = bh.mean_motion_general(oe, bh.GM_MARS, bh.AngleFormat.RADIANS)
///     print(f"Mean motion: {n:.6f} rad/s")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(a_or_oe, gm, angle_format)")]
#[pyo3(name = "mean_motion_general")]
fn py_mean_motion_general(a_or_oe: &Bound<'_, PyAny>, gm: f64, angle_format: &PyAngleFormat) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(a) = a_or_oe.extract::<f64>() {
        return Ok(orbits::mean_motion_general(a, gm, angle_format.value));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(a_or_oe, Some(6))?;
    Ok(orbits::mean_motion_general(oe[0], gm, angle_format.value))
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
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Calculate semi-major axis from mean motion (typical LEO satellite)
///     n = 0.001027  # radians/second (~15 revolutions/day)
///     a = bh.semimajor_axis(n, bh.AngleFormat.RADIANS)
///     print(f"Semi-major axis: {a/1000:.2f} km")
///     ```
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
///     gm (float): (keyword-only) The standard gravitational parameter of primary body in m³/s².
///     angle_format (AngleFormat): Interpret mean motion as AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: The semi-major axis of the astronomical object in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Calculate semi-major axis for Jupiter orbiter
///     n = 0.0001  # radians/second
///     a = bh.semimajor_axis_general(n, bh.GM_JUPITER, bh.AngleFormat.RADIANS)
///     print(f"Semi-major axis: {a/1000:.2f} km")
///     ```
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
///     gm (float): (keyword-only) The standard gravitational parameter of primary body in m³/s².
///
/// Returns:
///     float: The semi-major axis in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Calculate semi-major axis for 2-hour Venus orbit
///     period = 2 * 3600.0  # 2 hours in seconds
///     a = bh.semimajor_axis_from_orbital_period_general(period, bh.GM_VENUS)
///     print(f"Semi-major axis: {a/1000:.2f} km")
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Calculate semi-major axis for a 90-minute orbit
///     period = 90 * 60.0  # 90 minutes in seconds
///     a = bh.semimajor_axis_from_orbital_period(period)
///     print(f"Semi-major axis: {a/1000:.2f} km")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(period)")]
#[pyo3(name = "semimajor_axis_from_orbital_period")]
fn py_semimajor_axis_from_orbital_period(period: f64) -> PyResult<f64> {
    Ok(orbits::semimajor_axis_from_orbital_period(period))
}

/// Computes the perigee velocity of an astronomical object around Earth.
///
/// Args:
///     a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
///     e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.
///
/// Returns:
///     float: The magnitude of velocity of the object at perigee in m/s.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar parameters
///     a = 26554000.0  # meters
///     e = 0.72  # high eccentricity
///     v_peri = bh.perigee_velocity(a, e)
///     print(f"Perigee velocity: {v_peri:.2f} m/s")
///
///     # Using Keplerian elements vector
///     oe = [26554000.0, 0.72, np.radians(63.4), 0, 0, 0]
///     v_peri = bh.perigee_velocity(oe)
///     print(f"Perigee velocity: {v_peri:.2f} m/s")
///     ```
#[pyfunction]
#[pyo3(signature = (a_or_oe, e=None), text_signature = "(a_or_oe, e=None)")]
#[pyo3(name = "perigee_velocity")]
fn py_perigee_velocity(a_or_oe: &Bound<'_, PyAny>, e: Option<f64>) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(a) = a_or_oe.extract::<f64>() {
        let ecc = e.ok_or_else(|| exceptions::PyValueError::new_err(
            "Parameter 'e' is required when 'a_or_oe' is a scalar"
        ))?;
        return Ok(orbits::perigee_velocity(a, ecc));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(a_or_oe, Some(6))?;
    Ok(orbits::perigee_velocity(oe[0], oe[1]))
}

/// Computes the periapsis velocity of an astronomical object around a general body.
///
/// Args:
///     a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
///     e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.
///     gm (float): (keyword-only) The standard gravitational parameter of primary body in m³/s².
///
/// Returns:
///     float: The magnitude of velocity of the object at periapsis in m/s.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar parameters
///     a = 5e11  # 5 AU semi-major axis (meters)
///     e = 0.95  # highly elliptical
///     v_peri = bh.periapsis_velocity(a, e, bh.GM_SUN)
///     print(f"Periapsis velocity: {v_peri/1000:.2f} km/s")
///
///     # Using Keplerian elements vector
///     oe = [5e11, 0.95, np.radians(10), 0, 0, 0]
///     v_peri = bh.periapsis_velocity(oe, gm=bh.GM_SUN)
///     print(f"Periapsis velocity: {v_peri/1000:.2f} km/s")
///     ```
#[pyfunction]
#[pyo3(signature = (a_or_oe, e=None, *, gm), text_signature = "(a_or_oe, e=None, *, gm)")]
#[pyo3(name = "periapsis_velocity")]
fn py_periapsis_velocity(a_or_oe: &Bound<'_, PyAny>, e: Option<f64>, gm: f64) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(a) = a_or_oe.extract::<f64>() {
        let ecc = e.ok_or_else(|| exceptions::PyValueError::new_err(
            "Parameter 'e' is required when 'a_or_oe' is a scalar"
        ))?;
        return Ok(orbits::periapsis_velocity(a, ecc, gm));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(a_or_oe, Some(6))?;
    Ok(orbits::periapsis_velocity(oe[0], oe[1], gm))
}

/// Calculate the distance of an object at its periapsis.
///
/// Args:
///     a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
///     e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.
///
/// Returns:
///     float: The distance of the object at periapsis in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar parameters
///     a = 8000000.0  # 8000 km semi-major axis
///     e = 0.2  # moderate eccentricity
///     r_peri = bh.periapsis_distance(a, e)
///     print(f"Periapsis distance: {r_peri/1000:.2f} km")
///
///     # Using Keplerian elements vector
///     oe = [8000000.0, 0.2, np.radians(45), 0, 0, 0]
///     r_peri = bh.periapsis_distance(oe)
///     print(f"Periapsis distance: {r_peri/1000:.2f} km")
///     ```
#[pyfunction]
#[pyo3(signature = (a_or_oe, e=None), text_signature = "(a_or_oe, e=None)")]
#[pyo3(name = "periapsis_distance")]
fn py_periapsis_distance(a_or_oe: &Bound<'_, PyAny>, e: Option<f64>) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(a) = a_or_oe.extract::<f64>() {
        let ecc = e.ok_or_else(|| exceptions::PyValueError::new_err(
            "Parameter 'e' is required when 'a_or_oe' is a scalar"
        ))?;
        return Ok(orbits::periapsis_distance(a, ecc));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(a_or_oe, Some(6))?;
    Ok(orbits::periapsis_distance(oe[0], oe[1]))
}

/// Computes the apogee velocity of an astronomical object around Earth.
///
/// Args:
///     a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
///     e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.
///
/// Returns:
///     float: The magnitude of velocity of the object at apogee in m/s.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar parameters
///     a = 24400000.0  # meters
///     e = 0.73  # high eccentricity
///     v_apo = bh.apogee_velocity(a, e)
///     print(f"Apogee velocity: {v_apo:.2f} m/s")
///
///     # Using Keplerian elements vector
///     oe = [24400000.0, 0.73, np.radians(7), 0, 0, 0]
///     v_apo = bh.apogee_velocity(oe)
///     print(f"Apogee velocity: {v_apo:.2f} m/s")
///     ```
#[pyfunction]
#[pyo3(signature = (a_or_oe, e=None), text_signature = "(a_or_oe, e=None)")]
#[pyo3(name = "apogee_velocity")]
fn py_apogee_velocity(a_or_oe: &Bound<'_, PyAny>, e: Option<f64>) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(a) = a_or_oe.extract::<f64>() {
        let ecc = e.ok_or_else(|| exceptions::PyValueError::new_err(
            "Parameter 'e' is required when 'a_or_oe' is a scalar"
        ))?;
        return Ok(orbits::apogee_velocity(a, ecc));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(a_or_oe, Some(6))?;
    Ok(orbits::apogee_velocity(oe[0], oe[1]))
}

/// Computes the apoapsis velocity of an astronomical object around a general body.
///
/// Args:
///     a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
///     e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.
///     gm (float): (keyword-only) The standard gravitational parameter of primary body in m³/s².
///
/// Returns:
///     float: The magnitude of velocity of the object at apoapsis in m/s.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar parameters
///     a = 10000000.0  # 10000 km semi-major axis
///     e = 0.3
///     v_apo = bh.apoapsis_velocity(a, e, bh.GM_MARS)
///     print(f"Apoapsis velocity: {v_apo/1000:.2f} km/s")
///
///     # Using Keplerian elements vector
///     oe = [10000000.0, 0.3, np.radians(30), 0, 0, 0]
///     v_apo = bh.apoapsis_velocity(oe, gm=bh.GM_MARS)
///     print(f"Apoapsis velocity: {v_apo/1000:.2f} km/s")
///     ```
#[pyfunction]
#[pyo3(signature = (a_or_oe, e=None, *, gm), text_signature = "(a_or_oe, e=None, *, gm)")]
#[pyo3(name = "apoapsis_velocity")]
fn py_apoapsis_velocity(a_or_oe: &Bound<'_, PyAny>, e: Option<f64>, gm: f64) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(a) = a_or_oe.extract::<f64>() {
        let ecc = e.ok_or_else(|| exceptions::PyValueError::new_err(
            "Parameter 'e' is required when 'a_or_oe' is a scalar"
        ))?;
        return Ok(orbits::apoapsis_velocity(a, ecc, gm));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(a_or_oe, Some(6))?;
    Ok(orbits::apoapsis_velocity(oe[0], oe[1], gm))
}

/// Calculate the distance of an object at its apoapsis.
///
/// Args:
///     a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
///     e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.
///
/// Returns:
///     float: The distance of the object at apoapsis in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar parameters
///     a = 8000000.0  # 8000 km semi-major axis
///     e = 0.2  # moderate eccentricity
///     r_apo = bh.apoapsis_distance(a, e)
///     print(f"Apoapsis distance: {r_apo/1000:.2f} km")
///
///     # Using Keplerian elements vector
///     oe = [8000000.0, 0.2, np.radians(45), 0, 0, 0]
///     r_apo = bh.apoapsis_distance(oe)
///     print(f"Apoapsis distance: {r_apo/1000:.2f} km")
///     ```
#[pyfunction]
#[pyo3(signature = (a_or_oe, e=None), text_signature = "(a_or_oe, e=None)")]
#[pyo3(name = "apoapsis_distance")]
fn py_apoapsis_distance(a_or_oe: &Bound<'_, PyAny>, e: Option<f64>) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(a) = a_or_oe.extract::<f64>() {
        let ecc = e.ok_or_else(|| exceptions::PyValueError::new_err(
            "Parameter 'e' is required when 'a_or_oe' is a scalar"
        ))?;
        return Ok(orbits::apoapsis_distance(a, ecc));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(a_or_oe, Some(6))?;
    Ok(orbits::apoapsis_distance(oe[0], oe[1]))
}

/// Calculate the altitude above a body's surface at periapsis.
///
/// Args:
///     a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
///     e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.
///     r_body (float): (keyword-only) The radius of the central body in meters.
///
/// Returns:
///     float: The altitude above the body's surface at periapsis in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar parameters
///     a = bh.R_EARTH + 500e3  # 500 km mean altitude
///     e = 0.01  # slight eccentricity
///     alt_peri = bh.periapsis_altitude(a, e, bh.R_EARTH)
///     print(f"Periapsis altitude: {alt_peri/1000:.2f} km")
///
///     # Using Keplerian elements vector
///     oe = [bh.R_EARTH + 500e3, 0.01, np.radians(45), 0, 0, 0]
///     alt_peri = bh.periapsis_altitude(oe, r_body=bh.R_EARTH)
///     print(f"Periapsis altitude: {alt_peri/1000:.2f} km")
///     ```
#[pyfunction]
#[pyo3(signature = (a_or_oe, e=None, *, r_body), text_signature = "(a_or_oe, e=None, *, r_body)")]
#[pyo3(name = "periapsis_altitude")]
fn py_periapsis_altitude(a_or_oe: &Bound<'_, PyAny>, e: Option<f64>, r_body: f64) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(a) = a_or_oe.extract::<f64>() {
        let ecc = e.ok_or_else(|| exceptions::PyValueError::new_err(
            "Parameter 'e' is required when 'a_or_oe' is a scalar"
        ))?;
        return Ok(orbits::periapsis_altitude(a, ecc, r_body));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(a_or_oe, Some(6))?;
    Ok(orbits::periapsis_altitude(oe[0], oe[1], r_body))
}

/// Calculate the altitude above Earth's surface at perigee.
///
/// Args:
///     a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
///     e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.
///
/// Returns:
///     float: The altitude above Earth's surface at perigee in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar parameters
///     a = bh.R_EARTH + 420e3  # 420 km mean altitude
///     e = 0.0005  # very nearly circular
///     alt = bh.perigee_altitude(a, e)
///     print(f"Perigee altitude: {alt/1000:.2f} km")
///
///     # Using Keplerian elements vector
///     oe = [bh.R_EARTH + 420e3, 0.0005, np.radians(51.6), 0, 0, 0]
///     alt = bh.perigee_altitude(oe)
///     print(f"Perigee altitude: {alt/1000:.2f} km")
///     ```
#[pyfunction]
#[pyo3(signature = (a_or_oe, e=None), text_signature = "(a_or_oe, e=None)")]
#[pyo3(name = "perigee_altitude")]
fn py_perigee_altitude(a_or_oe: &Bound<'_, PyAny>, e: Option<f64>) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(a) = a_or_oe.extract::<f64>() {
        let ecc = e.ok_or_else(|| exceptions::PyValueError::new_err(
            "Parameter 'e' is required when 'a_or_oe' is a scalar"
        ))?;
        return Ok(orbits::perigee_altitude(a, ecc));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(a_or_oe, Some(6))?;
    Ok(orbits::perigee_altitude(oe[0], oe[1]))
}

/// Calculate the altitude above a body's surface at apoapsis.
///
/// Args:
///     a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
///     e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.
///     r_body (float): (keyword-only) The radius of the central body in meters.
///
/// Returns:
///     float: The altitude above the body's surface at apoapsis in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar parameters
///     a = bh.R_MOON + 100e3  # 100 km mean altitude
///     e = 0.05  # moderate eccentricity
///     alt_apo = bh.apoapsis_altitude(a, e, bh.R_MOON)
///     print(f"Apoapsis altitude: {alt_apo/1000:.2f} km")
///
///     # Using Keplerian elements vector
///     oe = [bh.R_MOON + 100e3, 0.05, np.radians(30), 0, 0, 0]
///     alt_apo = bh.apoapsis_altitude(oe, r_body=bh.R_MOON)
///     print(f"Apoapsis altitude: {alt_apo/1000:.2f} km")
///     ```
#[pyfunction]
#[pyo3(signature = (a_or_oe, e=None, *, r_body), text_signature = "(a_or_oe, e=None, *, r_body)")]
#[pyo3(name = "apoapsis_altitude")]
fn py_apoapsis_altitude(a_or_oe: &Bound<'_, PyAny>, e: Option<f64>, r_body: f64) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(a) = a_or_oe.extract::<f64>() {
        let ecc = e.ok_or_else(|| exceptions::PyValueError::new_err(
            "Parameter 'e' is required when 'a_or_oe' is a scalar"
        ))?;
        return Ok(orbits::apoapsis_altitude(a, ecc, r_body));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(a_or_oe, Some(6))?;
    Ok(orbits::apoapsis_altitude(oe[0], oe[1], r_body))
}

/// Calculate the altitude above Earth's surface at apogee.
///
/// Args:
///     a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
///     e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.
///
/// Returns:
///     float: The altitude above Earth's surface at apogee in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar parameters
///     a = 26554000.0  # ~26554 km semi-major axis
///     e = 0.7  # highly eccentric
///     alt = bh.apogee_altitude(a, e)
///     print(f"Apogee altitude: {alt/1000:.2f} km")
///
///     # Using Keplerian elements vector
///     oe = [26554000.0, 0.7, np.radians(63.4), 0, 0, 0]
///     alt = bh.apogee_altitude(oe)
///     print(f"Apogee altitude: {alt/1000:.2f} km")
///     ```
#[pyfunction]
#[pyo3(signature = (a_or_oe, e=None), text_signature = "(a_or_oe, e=None)")]
#[pyo3(name = "apogee_altitude")]
fn py_apogee_altitude(a_or_oe: &Bound<'_, PyAny>, e: Option<f64>) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(a) = a_or_oe.extract::<f64>() {
        let ecc = e.ok_or_else(|| exceptions::PyValueError::new_err(
            "Parameter 'e' is required when 'a_or_oe' is a scalar"
        ))?;
        return Ok(orbits::apogee_altitude(a, ecc));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(a_or_oe, Some(6))?;
    Ok(orbits::apogee_altitude(oe[0], oe[1]))
}

/// Computes the inclination for a Sun-synchronous orbit around Earth based on
/// the J2 gravitational perturbation.
///
/// Args:
///     a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
///     e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.
///     angle_format (AngleFormat): (keyword-only) Return output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: Inclination for a Sun synchronous orbit in degrees or radians.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar parameters
///     a = bh.R_EARTH + 600e3
///     e = 0.001  # nearly circular
///     inc = bh.sun_synchronous_inclination(a, e, bh.AngleFormat.DEGREES)
///     print(f"Sun-synchronous inclination: {inc:.2f} degrees")
///
///     # Using Keplerian elements vector
///     oe = [bh.R_EARTH + 600e3, 0.001, np.radians(97.8), 0, 0, 0]
///     inc = bh.sun_synchronous_inclination(oe, angle_format=bh.AngleFormat.DEGREES)
///     print(f"Sun-synchronous inclination: {inc:.2f} degrees")
///     ```
#[pyfunction]
#[pyo3(signature = (a_or_oe, e=None, *, angle_format), text_signature = "(a_or_oe, e=None, *, angle_format)")]
#[pyo3(name = "sun_synchronous_inclination")]
fn py_sun_synchronous_inclination(a_or_oe: &Bound<'_, PyAny>, e: Option<f64>, angle_format: &PyAngleFormat) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(a) = a_or_oe.extract::<f64>() {
        let ecc = e.ok_or_else(|| exceptions::PyValueError::new_err(
            "Parameter 'e' is required when 'a_or_oe' is a scalar"
        ))?;
        return Ok(orbits::sun_synchronous_inclination(a, ecc, angle_format.value));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(a_or_oe, Some(6))?;
    Ok(orbits::sun_synchronous_inclination(oe[0], oe[1], angle_format.value))
}

/// Returns the geostationary orbit semi-major axis around Earth.
/// 
/// Returns:
///    float: The semi-major axis in meters.
/// 
/// Example:
///    ```python
///    import brahe as bh
///    a_geo = bh.geo_sma()
///    print(f"Geostationary semi-major axis: {a_geo/1000:.2f} km")
///    ```
#[pyfunction]
#[pyo3(signature = (), text_signature = "()")]
#[pyo3(name = "geo_sma")]
fn py_geo_sma() -> PyResult<f64> {
    Ok(orbits::geo_sma())
}

/// Converts eccentric anomaly into mean anomaly.
///
/// Args:
///     anm_ecc_or_oe (float or array): Either the eccentric anomaly, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, E] from which `e` and `E` will be extracted.
///         The anomaly in the vector should match the `angle_format`.
///     e (float, optional): The eccentricity. Required if `anm_ecc_or_oe` is a scalar, ignored if vector.
///     angle_format (AngleFormat): (keyword-only) Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: Mean anomaly in radians or degrees.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar parameters
///     E = np.pi / 4  # 45 degrees eccentric anomaly
///     e = 0.1  # eccentricity
///     M = bh.anomaly_eccentric_to_mean(E, e, bh.AngleFormat.RADIANS)
///     print(f"Mean anomaly: {M:.4f} radians")
///
///     # Using Keplerian elements vector (with eccentric anomaly at index 5)
///     oe = [bh.R_EARTH + 500e3, 0.1, np.radians(45), 0, 0, np.pi/4]
///     M = bh.anomaly_eccentric_to_mean(oe, angle_format=bh.AngleFormat.RADIANS)
///     print(f"Mean anomaly: {M:.4f} radians")
///     ```
#[pyfunction]
#[pyo3(signature = (anm_ecc_or_oe, e=None, *, angle_format), text_signature = "(anm_ecc_or_oe, e=None, *, angle_format)")]
#[pyo3(name = "anomaly_eccentric_to_mean")]
fn py_anomaly_eccentric_to_mean(anm_ecc_or_oe: &Bound<'_, PyAny>, e: Option<f64>, angle_format: &PyAngleFormat) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(anm_ecc) = anm_ecc_or_oe.extract::<f64>() {
        let ecc = e.ok_or_else(|| exceptions::PyValueError::new_err(
            "Parameter 'e' is required when 'anm_ecc_or_oe' is a scalar"
        ))?;
        return Ok(orbits::anomaly_eccentric_to_mean(anm_ecc, ecc, angle_format.value));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(anm_ecc_or_oe, Some(6))?;
    Ok(orbits::anomaly_eccentric_to_mean(oe[5], oe[1], angle_format.value))
}

/// Converts mean anomaly into eccentric anomaly.
///
/// Args:
///     anm_mean_or_oe (float or array): Either the mean anomaly, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, M] from which `e` and `M` will be extracted.
///         The anomaly in the vector should match the `angle_format`.
///     e (float, optional): The eccentricity. Required if `anm_mean_or_oe` is a scalar, ignored if vector.
///     angle_format (AngleFormat): (keyword-only) Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: Eccentric anomaly in radians or degrees.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar parameters
///     M = 1.5  # mean anomaly in radians
///     e = 0.3  # eccentricity
///     E = bh.anomaly_mean_to_eccentric(M, e, bh.AngleFormat.RADIANS)
///     print(f"Eccentric anomaly: {E:.4f} radians")
///
///     # Using Keplerian elements vector (with mean anomaly at index 5)
///     oe = [bh.R_EARTH + 500e3, 0.3, np.radians(45), 0, 0, 1.5]
///     E = bh.anomaly_mean_to_eccentric(oe, angle_format=bh.AngleFormat.RADIANS)
///     print(f"Eccentric anomaly: {E:.4f} radians")
///     ```
#[pyfunction]
#[pyo3(signature = (anm_mean_or_oe, e=None, *, angle_format), text_signature = "(anm_mean_or_oe, e=None, *, angle_format)")]
#[pyo3(name = "anomaly_mean_to_eccentric")]
fn py_anomaly_mean_to_eccentric(anm_mean_or_oe: &Bound<'_, PyAny>, e: Option<f64>, angle_format: &PyAngleFormat) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(anm_mean) = anm_mean_or_oe.extract::<f64>() {
        let ecc = e.ok_or_else(|| exceptions::PyValueError::new_err(
            "Parameter 'e' is required when 'anm_mean_or_oe' is a scalar"
        ))?;
        return match orbits::anomaly_mean_to_eccentric(anm_mean, ecc, angle_format.value) {
            Ok(value) => Ok(value),
            Err(err) => Err(exceptions::PyRuntimeError::new_err(err)),
        };
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(anm_mean_or_oe, Some(6))?;
    match orbits::anomaly_mean_to_eccentric(oe[5], oe[1], angle_format.value) {
        Ok(value) => Ok(value),
        Err(err) => Err(exceptions::PyRuntimeError::new_err(err)),
    }
}

/// Converts true anomaly into eccentric anomaly.
///
/// Args:
///     anm_true_or_oe (float or array): Either the true anomaly, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, ν] from which `e` and `ν` will be extracted.
///         The anomaly in the vector should match the `angle_format`.
///     e (float, optional): The eccentricity. Required if `anm_true_or_oe` is a scalar, ignored if vector.
///     angle_format (AngleFormat): (keyword-only) Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: Eccentric anomaly in radians or degrees.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar parameters
///     nu = np.pi / 3  # 60 degrees true anomaly
///     e = 0.2  # eccentricity
///     E = bh.anomaly_true_to_eccentric(nu, e, bh.AngleFormat.RADIANS)
///     print(f"Eccentric anomaly: {E:.4f} radians")
///
///     # Using Keplerian elements vector (with true anomaly at index 5)
///     oe = [bh.R_EARTH + 500e3, 0.2, np.radians(45), 0, 0, np.pi/3]
///     E = bh.anomaly_true_to_eccentric(oe, angle_format=bh.AngleFormat.RADIANS)
///     print(f"Eccentric anomaly: {E:.4f} radians")
///     ```
#[pyfunction]
#[pyo3(signature = (anm_true_or_oe, e=None, *, angle_format), text_signature = "(anm_true_or_oe, e=None, *, angle_format)")]
#[pyo3(name = "anomaly_true_to_eccentric")]
fn py_anomaly_true_to_eccentric(anm_true_or_oe: &Bound<'_, PyAny>, e: Option<f64>, angle_format: &PyAngleFormat) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(anm_true) = anm_true_or_oe.extract::<f64>() {
        let ecc = e.ok_or_else(|| exceptions::PyValueError::new_err(
            "Parameter 'e' is required when 'anm_true_or_oe' is a scalar"
        ))?;
        return Ok(orbits::anomaly_true_to_eccentric(anm_true, ecc, angle_format.value));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(anm_true_or_oe, Some(6))?;
    Ok(orbits::anomaly_true_to_eccentric(oe[5], oe[1], angle_format.value))
}

/// Converts eccentric anomaly into true anomaly.
///
/// Args:
///     anm_ecc_or_oe (float or array): Either the eccentric anomaly, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, E] from which `e` and `E` will be extracted.
///         The anomaly in the vector should match the `angle_format`.
///     e (float, optional): The eccentricity. Required if `anm_ecc_or_oe` is a scalar, ignored if vector.
///     angle_format (AngleFormat): (keyword-only) Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: True anomaly in radians or degrees.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar parameters
///     E = np.pi / 4  # 45 degrees eccentric anomaly
///     e = 0.4  # eccentricity
///     nu = bh.anomaly_eccentric_to_true(E, e, bh.AngleFormat.RADIANS)
///     print(f"True anomaly: {nu:.4f} radians")
///
///     # Using Keplerian elements vector (with eccentric anomaly at index 5)
///     oe = [bh.R_EARTH + 500e3, 0.4, np.radians(45), 0, 0, np.pi/4]
///     nu = bh.anomaly_eccentric_to_true(oe, angle_format=bh.AngleFormat.RADIANS)
///     print(f"True anomaly: {nu:.4f} radians")
///     ```
#[pyfunction]
#[pyo3(signature = (anm_ecc_or_oe, e=None, *, angle_format), text_signature = "(anm_ecc_or_oe, e=None, *, angle_format)")]
#[pyo3(name = "anomaly_eccentric_to_true")]
fn py_anomaly_eccentric_to_true(anm_ecc_or_oe: &Bound<'_, PyAny>, e: Option<f64>, angle_format: &PyAngleFormat) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(anm_ecc) = anm_ecc_or_oe.extract::<f64>() {
        let ecc = e.ok_or_else(|| exceptions::PyValueError::new_err(
            "Parameter 'e' is required when 'anm_ecc_or_oe' is a scalar"
        ))?;
        return Ok(orbits::anomaly_eccentric_to_true(anm_ecc, ecc, angle_format.value));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(anm_ecc_or_oe, Some(6))?;
    Ok(orbits::anomaly_eccentric_to_true(oe[5], oe[1], angle_format.value))
}

/// Converts true anomaly into mean anomaly.
///
/// Args:
///     anm_true_or_oe (float or array): Either the true anomaly, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, ν] from which `e` and `ν` will be extracted.
///         The anomaly in the vector should match the `angle_format`.
///     e (float, optional): The eccentricity. Required if `anm_true_or_oe` is a scalar, ignored if vector.
///     angle_format (AngleFormat): (keyword-only) Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: Mean anomaly in radians or degrees.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar parameters
///     nu = np.pi / 2  # 90 degrees true anomaly
///     e = 0.15  # eccentricity
///     M = bh.anomaly_true_to_mean(nu, e, bh.AngleFormat.RADIANS)
///     print(f"Mean anomaly: {M:.4f} radians")
///
///     # Using Keplerian elements vector (with true anomaly at index 5)
///     oe = [bh.R_EARTH + 500e3, 0.15, np.radians(45), 0, 0, np.pi/2]
///     M = bh.anomaly_true_to_mean(oe, angle_format=bh.AngleFormat.RADIANS)
///     print(f"Mean anomaly: {M:.4f} radians")
///     ```
#[pyfunction]
#[pyo3(signature = (anm_true_or_oe, e=None, *, angle_format), text_signature = "(anm_true_or_oe, e=None, *, angle_format)")]
#[pyo3(name = "anomaly_true_to_mean")]
fn py_anomaly_true_to_mean(anm_true_or_oe: &Bound<'_, PyAny>, e: Option<f64>, angle_format: &PyAngleFormat) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(anm_true) = anm_true_or_oe.extract::<f64>() {
        let ecc = e.ok_or_else(|| exceptions::PyValueError::new_err(
            "Parameter 'e' is required when 'anm_true_or_oe' is a scalar"
        ))?;
        return Ok(orbits::anomaly_true_to_mean(anm_true, ecc, angle_format.value));
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(anm_true_or_oe, Some(6))?;
    Ok(orbits::anomaly_true_to_mean(oe[5], oe[1], angle_format.value))
}

/// Converts mean anomaly into true anomaly.
///
/// Args:
///     anm_mean_or_oe (float or array): Either the mean anomaly, or a 6-element
///         Keplerian elements array [a, e, i, Ω, ω, M] from which `e` and `M` will be extracted.
///         The anomaly in the vector should match the `angle_format`.
///     e (float, optional): The eccentricity. Required if `anm_mean_or_oe` is a scalar, ignored if vector.
///     angle_format (AngleFormat): (keyword-only) Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: True anomaly in radians or degrees.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Using scalar parameters
///     M = 2.0  # mean anomaly in radians
///     e = 0.25  # eccentricity
///     nu = bh.anomaly_mean_to_true(M, e, bh.AngleFormat.RADIANS)
///     print(f"True anomaly: {nu:.4f} radians")
///
///     # Using Keplerian elements vector (with mean anomaly at index 5)
///     oe = [bh.R_EARTH + 500e3, 0.25, np.radians(45), 0, 0, 2.0]
///     nu = bh.anomaly_mean_to_true(oe, angle_format=bh.AngleFormat.RADIANS)
///     print(f"True anomaly: {nu:.4f} radians")
///     ```
#[pyfunction]
#[pyo3(signature = (anm_mean_or_oe, e=None, *, angle_format), text_signature = "(anm_mean_or_oe, e=None, *, angle_format)")]
#[pyo3(name = "anomaly_mean_to_true")]
fn py_anomaly_mean_to_true(anm_mean_or_oe: &Bound<'_, PyAny>, e: Option<f64>, angle_format: &PyAngleFormat) -> PyResult<f64> {
    // Try to extract as scalar first
    if let Ok(anm_mean) = anm_mean_or_oe.extract::<f64>() {
        let ecc = e.ok_or_else(|| exceptions::PyValueError::new_err(
            "Parameter 'e' is required when 'anm_mean_or_oe' is a scalar"
        ))?;
        return match orbits::anomaly_mean_to_true(anm_mean, ecc, angle_format.value) {
            Ok(value) => Ok(value),
            Err(err) => Err(exceptions::PyRuntimeError::new_err(err)),
        };
    }

    // Try to extract as vector (Keplerian elements)
    let oe = pyany_to_f64_array1(anm_mean_or_oe, Some(6))?;
    match orbits::anomaly_mean_to_true(oe[5], oe[1], angle_format.value) {
        Ok(value) => Ok(value),
        Err(err) => Err(exceptions::PyRuntimeError::new_err(err)),
    }
}

// New propagator implementations


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

/// Convert numeric NORAD ID to Alpha-5 format or pass through if in legacy range.
///
/// Args:
///     norad_id (int): Numeric NORAD ID (0-339999). IDs 0-99999 are passed through
///         as numeric strings. IDs 100000-339999 are converted to Alpha-5 format.
///
/// Returns:
///     str: For IDs 0-99999: numeric string (e.g., "42"). For IDs 100000-339999:
///         Alpha-5 format ID (e.g., "A0001").
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
///     ```python
///     line1 = "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997"
///     epoch = epoch_from_tle(line1)
///     epoch.year()
///     ```
#[pyfunction]
#[pyo3(text_signature = "(line1)")]
#[pyo3(name = "epoch_from_tle")]
fn py_epoch_from_tle(line1: String) -> PyResult<PyEpoch> {
    match orbits::epoch_from_tle(&line1) {
        Ok(epoch) => Ok(PyEpoch { obj: epoch }),
        Err(e) => Err(exceptions::PyRuntimeError::new_err(e.to_string())),
    }
}

// =============================================================================
// Mean-Osculating Conversion Method Selector
// =============================================================================

/// Algorithm used to map between mean and osculating Keplerian elements.
///
/// Note:
///     This class is created via a class attribute and a class method:
///     - `MeanElementMethod.BROUWER_LYDDANE` - First-order analytical J2 mapping
///     - `MeanElementMethod.numerical(config)` - Numerical windowed averaging (batch-only)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Analytical Brouwer-Lyddane method (single-state and batch)
///     method = bh.MeanElementMethod.BROUWER_LYDDANE
///
///     # Numerical windowed-averaging method (batch-only)
///     cfg = bh.MeanElementNumericalMethodConfig(5400.0, bh.WindowAlignment.CENTERED, bh.WindowEdgeHandling.TRUNCATE)
///     method = bh.MeanElementMethod.numerical(cfg)
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "MeanElementMethod")]
#[derive(Clone)]
pub struct PyMeanElementMethod {
    pub method: orbits::MeanElementMethod,
}

#[pymethods]
impl PyMeanElementMethod {
    /// First-order analytical Brouwer-Lyddane J2 method.
    #[classattr]
    #[allow(non_snake_case)]
    fn BROUWER_LYDDANE() -> Self {
        PyMeanElementMethod {
            method: orbits::MeanElementMethod::BrouwerLyddane,
        }
    }

    /// Create a numerical windowed-averaging method selector (batch-only).
    ///
    /// Args:
    ///     config (MeanElementNumericalMethodConfig): Numerical averaging window configuration.
    ///
    /// Returns:
    ///     MeanElementMethod: A numerical method selector.
    #[classmethod]
    fn numerical(_cls: &Bound<'_, PyType>, config: &PyMeanElementNumericalMethodConfig) -> Self {
        PyMeanElementMethod {
            method: orbits::MeanElementMethod::Numerical(config.config.clone()),
        }
    }

    fn __repr__(&self) -> String {
        match &self.method {
            orbits::MeanElementMethod::BrouwerLyddane => "MeanElementMethod.BROUWER_LYDDANE".to_string(),
            orbits::MeanElementMethod::Numerical(_) => "MeanElementMethod.numerical(...)".to_string(),
        }
    }
}

/// Placement of the numerical averaging window relative to the output epoch.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     alignment = bh.WindowAlignment.CENTERED
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "WindowAlignment")]
#[derive(Clone)]
pub struct PyWindowAlignment {
    pub value: orbits::WindowAlignment,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyWindowAlignment {
    /// Symmetric window `[t - W/2, t + W/2]`.
    #[classattr]
    fn CENTERED() -> Self {
        PyWindowAlignment {
            value: orbits::WindowAlignment::Centered,
        }
    }

    /// Causal look-back window `[t - W, t]`.
    #[classattr]
    fn TRAILING() -> Self {
        PyWindowAlignment {
            value: orbits::WindowAlignment::Trailing,
        }
    }

    /// Look-ahead window `[t, t + W]`.
    #[classattr]
    fn LEADING() -> Self {
        PyWindowAlignment {
            value: orbits::WindowAlignment::Leading,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.value)
    }

    fn __repr__(&self) -> String {
        format!("WindowAlignment.{:?}", self.value)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.value == other.value),
            CompareOp::Ne => Ok(self.value != other.value),
            _ => Err(exceptions::PyNotImplementedError::new_err(
                "Comparison not supported",
            )),
        }
    }
}

/// Handling of output epochs whose averaging window runs past the data bounds.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     edge = bh.WindowEdgeHandling.TRUNCATE
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "WindowEdgeHandling")]
#[derive(Clone)]
pub struct PyWindowEdgeHandling {
    pub value: orbits::WindowEdgeHandling,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyWindowEdgeHandling {
    /// Drop unsupported output epochs; output is shorter than input.
    #[classattr]
    fn TRUNCATE() -> Self {
        PyWindowEdgeHandling {
            value: orbits::WindowEdgeHandling::Truncate,
        }
    }

    /// Hold the window length fixed and slide the anchor within it; output length preserved.
    #[classattr]
    fn PRESERVE_WINDOW() -> Self {
        PyWindowEdgeHandling {
            value: orbits::WindowEdgeHandling::PreserveWindow,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.value)
    }

    fn __repr__(&self) -> String {
        format!("WindowEdgeHandling.{:?}", self.value)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.value == other.value),
            CompareOp::Ne => Ok(self.value != other.value),
            _ => Err(exceptions::PyNotImplementedError::new_err(
                "Comparison not supported",
            )),
        }
    }
}

/// Configuration for the numerical windowed-averaging mean-element method.
///
/// Args:
///     window_seconds (float): Averaging window length in seconds.
///     alignment (WindowAlignment): Placement of the averaging window relative to the output epoch.
///     edge (WindowEdgeHandling): Handling of output epochs whose window runs past the data bounds.
///     inverse (MeanElementInverseConfig, optional): Dynamics for the iterative mean-to-osculating inverse.
///         Required for numerical mean-to-osculating conversion; unused for osculating-to-mean.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     cfg = bh.MeanElementNumericalMethodConfig(5400.0, bh.WindowAlignment.CENTERED, bh.WindowEdgeHandling.TRUNCATE)
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "MeanElementNumericalMethodConfig")]
#[derive(Clone)]
pub struct PyMeanElementNumericalMethodConfig {
    pub config: orbits::MeanElementNumericalMethodConfig,
}

#[pymethods]
impl PyMeanElementNumericalMethodConfig {
    /// Create a numerical windowed-averaging configuration.
    ///
    /// Args:
    ///     window_seconds (float): Averaging window length in seconds.
    ///     alignment (WindowAlignment): Placement of the averaging window relative to the output epoch.
    ///     edge (WindowEdgeHandling): Handling of output epochs whose window runs past the data bounds.
    ///     inverse (MeanElementInverseConfig, optional): Dynamics for the iterative mean-to-osculating inverse.
    ///
    /// Returns:
    ///     MeanElementNumericalMethodConfig: A new numerical averaging configuration.
    #[new]
    #[pyo3(signature = (window_seconds, alignment, edge, inverse=None))]
    fn new(
        window_seconds: f64,
        alignment: &PyWindowAlignment,
        edge: &PyWindowEdgeHandling,
        inverse: Option<&PyMeanElementInverseConfig>,
    ) -> Self {
        PyMeanElementNumericalMethodConfig {
            config: orbits::MeanElementNumericalMethodConfig {
                window_seconds,
                alignment: alignment.value,
                edge: edge.value,
                inverse: inverse.map(|i| i.config.clone()),
            },
        }
    }

    /// Averaging window length in seconds.
    #[getter]
    fn get_window_seconds(&self) -> f64 {
        self.config.window_seconds
    }

    #[setter]
    fn set_window_seconds(&mut self, value: f64) {
        self.config.window_seconds = value;
    }

    /// Placement of the averaging window relative to the output epoch.
    #[getter]
    fn get_alignment(&self) -> PyWindowAlignment {
        PyWindowAlignment {
            value: self.config.alignment,
        }
    }

    #[setter]
    fn set_alignment(&mut self, value: &PyWindowAlignment) {
        self.config.alignment = value.value;
    }

    /// Handling of output epochs whose window runs past the data bounds.
    #[getter]
    fn get_edge(&self) -> PyWindowEdgeHandling {
        PyWindowEdgeHandling {
            value: self.config.edge,
        }
    }

    #[setter]
    fn set_edge(&mut self, value: &PyWindowEdgeHandling) {
        self.config.edge = value.value;
    }

    /// Dynamics for the iterative mean-to-osculating inverse, if configured.
    #[getter]
    fn get_inverse(&self) -> Option<PyMeanElementInverseConfig> {
        self.config
            .inverse
            .clone()
            .map(|config| PyMeanElementInverseConfig { config })
    }

    #[setter]
    fn set_inverse(&mut self, value: Option<&PyMeanElementInverseConfig>) {
        self.config.inverse = value.map(|v| v.config.clone());
    }

    fn __repr__(&self) -> String {
        format!(
            "MeanElementNumericalMethodConfig(window_seconds={}, alignment={:?}, edge={:?})",
            self.config.window_seconds, self.config.alignment, self.config.edge
        )
    }
}

/// Iterative differential-correction dynamics for numerical mean-to-osculating conversion.
///
/// Args:
///     force_model (ForceModelConfig): Force model used to propagate the trial osculating state.
///     propagation (NumericalPropagationConfig): Integrator settings for the trial propagation.
///     tolerance (float): Convergence tolerance on the mean-element residual, compared
///         against the mixed norm ``|Δa| (m) + 1e6·|Δe| + 1e6·‖Δ(i, Ω, ω, M)‖ (rad)``.
///         For example, ``tolerance=1.0`` demands roughly 1 m in a, 1e-6 in e, and
///         1e-6 rad (~0.2 arcsec) in the angles combined.
///     max_iterations (int): Maximum number of differential-correction iterations.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     cfg = bh.MeanElementInverseConfig(
///         bh.ForceModelConfig.default(),
///         bh.NumericalPropagationConfig.default(),
///         1.0,
///         50,
///     )
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "MeanElementInverseConfig")]
#[derive(Clone)]
pub struct PyMeanElementInverseConfig {
    pub config: orbits::MeanElementInverseConfig,
}

#[pymethods]
impl PyMeanElementInverseConfig {
    /// Create an iterative differential-correction configuration.
    ///
    /// Args:
    ///     force_model (ForceModelConfig): Force model used to propagate the trial osculating state.
    ///     propagation (NumericalPropagationConfig): Integrator settings for the trial propagation.
    ///     tolerance (float): Convergence tolerance on the mean-element residual, compared
///         against the mixed norm ``|Δa| (m) + 1e6·|Δe| + 1e6·‖Δ(i, Ω, ω, M)‖ (rad)``.
///         For example, ``tolerance=1.0`` demands roughly 1 m in a, 1e-6 in e, and
///         1e-6 rad (~0.2 arcsec) in the angles combined.
    ///     max_iterations (int): Maximum number of differential-correction iterations.
    ///
    /// Returns:
    ///     MeanElementInverseConfig: A new inverse-dynamics configuration.
    #[new]
    fn new(
        force_model: &PyForceModelConfig,
        propagation: &PyNumericalPropagationConfig,
        tolerance: f64,
        max_iterations: usize,
    ) -> Self {
        PyMeanElementInverseConfig {
            config: orbits::MeanElementInverseConfig {
                force_model: force_model.config.clone(),
                propagation: propagation.config.clone(),
                tolerance,
                max_iterations,
            },
        }
    }

    /// Force model used to propagate the trial osculating state.
    #[getter]
    fn get_force_model(&self) -> PyForceModelConfig {
        PyForceModelConfig {
            config: self.config.force_model.clone(),
        }
    }

    #[setter]
    fn set_force_model(&mut self, value: &PyForceModelConfig) {
        self.config.force_model = value.config.clone();
    }

    /// Integrator settings for the trial propagation.
    #[getter]
    fn get_propagation(&self) -> PyNumericalPropagationConfig {
        PyNumericalPropagationConfig {
            config: self.config.propagation.clone(),
        }
    }

    #[setter]
    fn set_propagation(&mut self, value: &PyNumericalPropagationConfig) {
        self.config.propagation = value.config.clone();
    }

    /// Convergence tolerance on the mean-element residual (mixed norm
    /// `|Δa| (m) + 1e6·|Δe| + 1e6·‖Δ(i, Ω, ω, M)‖ (rad)`).
    #[getter]
    fn get_tolerance(&self) -> f64 {
        self.config.tolerance
    }

    #[setter]
    fn set_tolerance(&mut self, value: f64) {
        self.config.tolerance = value;
    }

    /// Maximum number of differential-correction iterations.
    #[getter]
    fn get_max_iterations(&self) -> usize {
        self.config.max_iterations
    }

    #[setter]
    fn set_max_iterations(&mut self, value: usize) {
        self.config.max_iterations = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "MeanElementInverseConfig(tolerance={}, max_iterations={})",
            self.config.tolerance, self.config.max_iterations
        )
    }
}

/// Convert osculating Keplerian elements to mean Keplerian elements.
///
/// Applies the selected mean-element method to convert osculating (instantaneous)
/// orbital elements to mean (orbit-averaged) elements. `MeanElementMethod.numerical()`
/// is batch-only and raises for single-state calls; use `MeanElementMethod.BROUWER_LYDDANE`
/// for the first-order analytical J2 mapping.
///
/// Args:
///     osc (numpy.ndarray): Osculating Keplerian elements as a 6-element array:
///         [a, e, i, Ω, ω, M] where:
///         - a: Semi-major axis (meters)
///         - e: Eccentricity (dimensionless)
///         - i: Inclination (radians or degrees, per angle_format)
///         - Ω: Right ascension of ascending node (radians or degrees, per angle_format)
///         - ω: Argument of perigee (radians or degrees, per angle_format)
///         - M: Mean anomaly (radians or degrees, per angle_format)
///     method (MeanElementMethod): Algorithm used to compute mean elements.
///     angle_format (AngleFormat): Format of angular elements (Radians or Degrees)
///
/// Returns:
///     numpy.ndarray: Mean Keplerian elements in the same format as input.
///
/// Raises:
///     BraheError: If `method` is `MeanElementMethod.numerical()`.
///
/// Note:
///     The forward and inverse transformations are not perfectly inverse due to
///     first-order truncation of the infinite series. Small errors of order J2²
///     are expected.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Define osculating elements for a LEO satellite (angles in degrees)
///     osc = np.array([
///         bh.R_EARTH + 500e3,  # a = 6878 km
///         0.001,               # e = 0.001 (near-circular)
///         45.0,                # i = 45 degrees
///         0.0,                 # Ω = 0
///         0.0,                 # ω = 0
///         0.0,                 # M = 0
///     ])
///
///     mean = bh.state_koe_osc_to_mean(osc, bh.MeanElementMethod.BROUWER_LYDDANE, bh.AngleFormat.DEGREES)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(osc, method, angle_format)")]
#[pyo3(name = "state_koe_osc_to_mean")]
fn py_state_koe_osc_to_mean<'py>(
    py: Python<'py>,
    osc: &Bound<'_, PyAny>,
    method: &PyMeanElementMethod,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let osc_vec = pyany_to_f64_array1(osc, Some(6))?;
    let osc_svec = SVector::<f64, 6>::from_row_slice(&osc_vec);
    let mean = orbits::state_koe_osc_to_mean(&osc_svec, method.method.clone(), angle_format.value)?;
    Ok(mean.as_slice().to_pyarray(py))
}

/// Convert mean Keplerian elements to osculating Keplerian elements.
///
/// Applies the selected mean-element method to convert mean (orbit-averaged)
/// orbital elements to osculating (instantaneous) elements. `MeanElementMethod.numerical()`
/// is batch-only and raises for single-state calls; use `MeanElementMethod.BROUWER_LYDDANE`
/// for the first-order analytical J2 mapping.
///
/// Args:
///     mean (numpy.ndarray): Mean Keplerian elements as a 6-element array:
///         [a, e, i, Ω, ω, M] where:
///         - a: Semi-major axis (meters)
///         - e: Eccentricity (dimensionless)
///         - i: Inclination (radians or degrees, per angle_format)
///         - Ω: Right ascension of ascending node (radians or degrees, per angle_format)
///         - ω: Argument of perigee (radians or degrees, per angle_format)
///         - M: Mean anomaly (radians or degrees, per angle_format)
///     method (MeanElementMethod): Algorithm used to compute osculating elements.
///     angle_format (AngleFormat): Format of angular elements (Radians or Degrees)
///
/// Returns:
///     numpy.ndarray: Osculating Keplerian elements in the same format as input.
///
/// Raises:
///     BraheError: If `method` is `MeanElementMethod.numerical()`.
///
/// Note:
///     The forward and inverse transformations are not perfectly inverse due to
///     first-order truncation of the infinite series. Small errors of order J2²
///     are expected.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Define mean elements for a LEO satellite (angles in degrees)
///     mean = np.array([
///         bh.R_EARTH + 500e3,  # a = 6878 km
///         0.001,               # e = 0.001 (near-circular)
///         45.0,                # i = 45 degrees
///         0.0,                 # Ω = 0
///         0.0,                 # ω = 0
///         0.0,                 # M = 0
///     ])
///
///     osc = bh.state_koe_mean_to_osc(mean, bh.MeanElementMethod.BROUWER_LYDDANE, bh.AngleFormat.DEGREES)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(mean, method, angle_format)")]
#[pyo3(name = "state_koe_mean_to_osc")]
fn py_state_koe_mean_to_osc<'py>(
    py: Python<'py>,
    mean: &Bound<'_, PyAny>,
    method: &PyMeanElementMethod,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let mean_vec = pyany_to_f64_array1(mean, Some(6))?;
    let mean_svec = SVector::<f64, 6>::from_row_slice(&mean_vec);
    let osc = orbits::state_koe_mean_to_osc(&mean_svec, method.method.clone(), angle_format.value)?;
    Ok(osc.as_slice().to_pyarray(py))
}

/// Converts an (M, 6) numpy array view into a `Vec` of 6-element state vectors.
///
/// Iterates row-by-row via the ndarray iterator rather than `as_slice()` so that
/// non-contiguous row views (e.g. from a sliced or transposed input array) are
/// handled correctly.
fn states_array2_to_svec6(states: &PyReadonlyArray2<f64>) -> PyResult<Vec<SVector<f64, 6>>> {
    let arr = states.as_array();
    if arr.ncols() != 6 {
        return Err(exceptions::PyValueError::new_err(format!(
            "States array must have 6 columns, got {}",
            arr.ncols()
        )));
    }
    Ok((0..arr.nrows())
        .map(|i| SVector::<f64, 6>::from_iterator(arr.row(i).iter().copied()))
        .collect())
}

/// Python return type shared by the batch mean-element conversion functions:
/// output epochs paired with an (N, 6) numpy array of Keplerian states.
type BatchKoeResult<'py> = (Vec<PyEpoch>, Bound<'py, PyArray<f64, Ix2>>);

/// Converts a `Vec` of `(epoch, state)` pairs into `(list[Epoch], numpy.ndarray)` for return to Python.
fn pairs_to_py<'py>(py: Python<'py>, pairs: Vec<(time::Epoch, SVector<f64, 6>)>) -> BatchKoeResult<'py> {
    let nrows = pairs.len();
    let mut epochs = Vec::with_capacity(nrows);
    let mut data = Vec::with_capacity(nrows * 6);
    for (t, s) in pairs {
        epochs.push(PyEpoch { obj: t });
        data.extend_from_slice(s.as_slice());
    }
    let states = PyArray::from_vec(py, data).reshape((nrows, 6)).unwrap();
    (epochs, states)
}

/// Convert a batch of osculating Keplerian states to mean Keplerian states.
///
/// Applies the selected mean-element method across an epoch/state series.
/// `MeanElementMethod.BROUWER_LYDDANE` maps each `(epoch, state)` pair
/// independently, so the output has the same length as the input.
/// `MeanElementMethod.numerical()` instead averages the osculating states
/// over a moving window (see `MeanElementNumericalMethodConfig`); the output may be shorter
/// than the input when `edge` is `WindowEdgeHandling.TRUNCATE` and some output
/// epochs' windows are not fully supported by the input trajectory.
///
/// Args:
///     epochs (list[Epoch]): Epochs of the input osculating states, one per `states` row.
///     states (numpy.ndarray): Osculating Keplerian elements, shape (M, 6). Each row is
///         [a, e, i, Ω, ω, M] where:
///         - a: Semi-major axis (meters)
///         - e: Eccentricity (dimensionless)
///         - i: Inclination (radians or degrees, per angle_format)
///         - Ω: Right ascension of ascending node (radians or degrees, per angle_format)
///         - ω: Argument of perigee (radians or degrees, per angle_format)
///         - M: Mean anomaly (radians or degrees, per angle_format)
///     method (MeanElementMethod): Algorithm used to compute mean elements.
///     angle_format (AngleFormat): Format of angular elements (Radians or Degrees).
///
/// Returns:
///     tuple: (list[Epoch], numpy.ndarray) of output epochs and mean Keplerian elements,
///         shape (N, 6). N equals M for `BROUWER_LYDDANE`; for `numerical()` N may be
///         less than M when `edge` is `WindowEdgeHandling.TRUNCATE`.
///
/// Raises:
///     BraheError: If `epochs` and `states` have different lengths, or if `method` is
///         `MeanElementMethod.numerical()` and `window_seconds` is not positive.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     e0 = bh.Epoch.from_gps_seconds(0.0)
///     epochs = [e0, e0 + 60.0, e0 + 120.0]
///     s = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
///     states = np.vstack([s, s, s])
///
///     out_epochs, out_states = bh.batch_state_koe_osc_to_mean(
///         epochs, states, bh.MeanElementMethod.BROUWER_LYDDANE, bh.AngleFormat.DEGREES
///     )
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epochs, states, method, angle_format)")]
#[pyo3(name = "batch_state_koe_osc_to_mean")]
fn py_batch_state_koe_osc_to_mean<'py>(
    py: Python<'py>,
    epochs: Vec<PyRef<'_, PyEpoch>>,
    states: PyReadonlyArray2<f64>,
    method: &PyMeanElementMethod,
    angle_format: &PyAngleFormat,
) -> PyResult<BatchKoeResult<'py>> {
    let epochs_vec: Vec<time::Epoch> = epochs.iter().map(|e| e.obj).collect();
    let states_vec = states_array2_to_svec6(&states)?;
    let pairs = orbits::batch_state_koe_osc_to_mean(
        &epochs_vec,
        &states_vec,
        method.method.clone(),
        angle_format.value,
    )?;
    Ok(pairs_to_py(py, pairs))
}

/// Convert a batch of mean Keplerian states to osculating Keplerian states.
///
/// Applies the selected mean-element method across an epoch/state series.
/// `MeanElementMethod.BROUWER_LYDDANE` maps each `(epoch, state)` pair
/// independently, so the output has the same length as the input.
/// `MeanElementMethod.numerical()` instead inverts the windowed averaging via
/// iterative differential correction (see `MeanElementNumericalMethodConfig` and `MeanElementInverseConfig`);
/// the output has the same length and epochs as the input.
///
/// Args:
///     epochs (list[Epoch]): Epochs of the input mean states, one per `states` row.
///     states (numpy.ndarray): Mean Keplerian elements, shape (M, 6). Each row is
///         [a, e, i, Ω, ω, M] where:
///         - a: Semi-major axis (meters)
///         - e: Eccentricity (dimensionless)
///         - i: Inclination (radians or degrees, per angle_format)
///         - Ω: Right ascension of ascending node (radians or degrees, per angle_format)
///         - ω: Argument of perigee (radians or degrees, per angle_format)
///         - M: Mean anomaly (radians or degrees, per angle_format)
///     method (MeanElementMethod): Algorithm used to compute osculating elements.
///     angle_format (AngleFormat): Format of angular elements (Radians or Degrees).
///
/// Returns:
///     tuple: (list[Epoch], numpy.ndarray) of output epochs and osculating Keplerian
///         elements, shape (M, 6).
///
/// Raises:
///     BraheError: If `epochs` and `states` have different lengths; if `method` is
///         `MeanElementMethod.numerical()` and `inverse` was not configured on the
///         `MeanElementNumericalMethodConfig`; or if the differential correction fails to converge
///         within `inverse.max_iterations` for any state.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     e0 = bh.Epoch.from_gps_seconds(0.0)
///     epochs = [e0, e0 + 60.0, e0 + 120.0]
///     s = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
///     states = np.vstack([s, s, s])
///
///     out_epochs, out_states = bh.batch_state_koe_mean_to_osc(
///         epochs, states, bh.MeanElementMethod.BROUWER_LYDDANE, bh.AngleFormat.DEGREES
///     )
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epochs, states, method, angle_format)")]
#[pyo3(name = "batch_state_koe_mean_to_osc")]
fn py_batch_state_koe_mean_to_osc<'py>(
    py: Python<'py>,
    epochs: Vec<PyRef<'_, PyEpoch>>,
    states: PyReadonlyArray2<f64>,
    method: &PyMeanElementMethod,
    angle_format: &PyAngleFormat,
) -> PyResult<BatchKoeResult<'py>> {
    let epochs_vec: Vec<time::Epoch> = epochs.iter().map(|e| e.obj).collect();
    let states_vec = states_array2_to_svec6(&states)?;
    let pairs = orbits::batch_state_koe_mean_to_osc(
        &epochs_vec,
        &states_vec,
        method.method.clone(),
        angle_format.value,
    )?;
    Ok(pairs_to_py(py, pairs))
}

/// Convert Keplerian elements to equinoctial elements (Vallado 2-99).
///
/// Args:
///     koe (numpy.ndarray): Keplerian `[a, e, i, Ω, ω, M]` (a in meters; angles per angle_format).
///     angle_format (AngleFormat): Format of angular inputs/outputs.
///     fr (int): Retrograde factor, +1 for direct orbits, -1 for near-retrograde. Defaults to 1.
///
/// Returns:
///     numpy.ndarray: Equinoctial `[a, h, k, p, q, l]` (a in meters; l per angle_format).
///
/// Example:
///     ```python
///     import brahe as bh, numpy as np
///     koe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
///     eqn = bh.state_koe_to_equinoctial(koe, bh.AngleFormat.DEGREES)
///     ```
#[pyfunction]
#[pyo3(signature = (koe, angle_format, fr=1), text_signature = "(koe, angle_format, fr=1)")]
#[pyo3(name = "state_koe_to_equinoctial")]
fn py_state_koe_to_equinoctial<'py>(
    py: Python<'py>,
    koe: &Bound<'_, PyAny>,
    angle_format: &PyAngleFormat,
    fr: i8,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let koe_vec = pyany_to_f64_array1(koe, Some(6))?;
    let koe_svec = SVector::<f64, 6>::from_row_slice(&koe_vec);
    let eqn = orbits::state_koe_to_equinoctial(&koe_svec, angle_format.value, fr);
    Ok(eqn.as_slice().to_pyarray(py))
}

/// Convert equinoctial elements to Keplerian elements (Vallado 2-99).
///
/// Args:
///     eqn (numpy.ndarray): Equinoctial `[a, h, k, p, q, l]` (a in meters; l per angle_format).
///     angle_format (AngleFormat): Format of angular input/outputs.
///     fr (int): Retrograde factor matching the forward conversion. Defaults to 1.
///
/// Returns:
///     numpy.ndarray: Keplerian `[a, e, i, Ω, ω, M]` (a in meters; angles per angle_format).
///
/// Example:
///     ```python
///     import brahe as bh, numpy as np
///     koe = np.array([bh.R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0])
///     eqn = bh.state_koe_to_equinoctial(koe, bh.AngleFormat.DEGREES)
///     back = bh.state_equinoctial_to_koe(eqn, bh.AngleFormat.DEGREES)
///     ```
#[pyfunction]
#[pyo3(signature = (eqn, angle_format, fr=1), text_signature = "(eqn, angle_format, fr=1)")]
#[pyo3(name = "state_equinoctial_to_koe")]
fn py_state_equinoctial_to_koe<'py>(
    py: Python<'py>,
    eqn: &Bound<'_, PyAny>,
    angle_format: &PyAngleFormat,
    fr: i8,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let eqn_vec = pyany_to_f64_array1(eqn, Some(6))?;
    let eqn_svec = SVector::<f64, 6>::from_row_slice(&eqn_vec);
    let koe = orbits::state_equinoctial_to_koe(&eqn_svec, angle_format.value, fr);
    Ok(koe.as_slice().to_pyarray(py))
}

// ============================================================================
// Walker Constellation Generator
// ============================================================================

/// Walker constellation pattern type.
///
/// Defines whether the constellation uses a Delta (360°) or Star (180°) RAAN spread.
///
/// Attributes:
///     DELTA: Walker Delta pattern with 360° RAAN spread (global coverage).
///         Used by GPS, Galileo, and similar navigation constellations.
///     STAR: Walker Star pattern with 180° RAAN spread (polar coverage).
///         Used by Iridium and similar communications constellations.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     delta = bh.WalkerPattern.DELTA  # 360° RAAN spread
///     star = bh.WalkerPattern.STAR    # 180° RAAN spread
///     ```
#[pyclass(name = "WalkerPattern", module = "brahe._brahe", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)] // Python convention: enum variants are UPPERCASE
pub enum PyWalkerPattern {
    /// Walker Delta pattern with 360° RAAN spread (global coverage)
    DELTA = 0,
    /// Walker Star pattern with 180° RAAN spread (polar coverage)
    STAR = 1,
}

impl From<PyWalkerPattern> for orbits::WalkerPattern {
    fn from(pattern: PyWalkerPattern) -> Self {
        match pattern {
            PyWalkerPattern::DELTA => orbits::WalkerPattern::Delta,
            PyWalkerPattern::STAR => orbits::WalkerPattern::Star,
        }
    }
}

impl From<orbits::WalkerPattern> for PyWalkerPattern {
    fn from(pattern: orbits::WalkerPattern) -> Self {
        match pattern {
            orbits::WalkerPattern::Delta => PyWalkerPattern::DELTA,
            orbits::WalkerPattern::Star => PyWalkerPattern::STAR,
        }
    }
}

/// Generator for Walker constellation patterns using T:P:F notation.
///
/// Walker constellations place satellites in circular or near-circular orbits with:
/// - T total satellites distributed across P planes
/// - Each plane inclined at the same angle
/// - Planes evenly spaced in RAAN (spread depends on pattern type)
/// - Satellites within each plane evenly spaced in mean anomaly
/// - Phase offset F controls inter-plane phasing
///
/// Two pattern types are supported via `pattern`:
/// - `WalkerPattern.DELTA`: 360° RAAN spread (global coverage, e.g., GPS, Galileo)
/// - `WalkerPattern.STAR`: 180° RAAN spread (polar coverage, e.g., Iridium)
///
/// All satellites in the constellation share the same semi-major axis, eccentricity,
/// inclination, and argument of perigee. They differ only in RAAN and mean anomaly.
///
/// Args:
///     t (int): Total number of satellites (must be divisible by p)
///     p (int): Number of orbital planes
///     f (int): Phasing factor (0 to p-1)
///     semi_major_axis (float): Semi-major axis in meters
///     eccentricity (float): Eccentricity (dimensionless)
///     inclination (float): Inclination in degrees or radians (based on angle_format)
///     argument_of_perigee (float): Argument of perigee
///     reference_raan (float): RAAN for plane 0
///     reference_mean_anomaly (float): Mean anomaly for first satellite
///     epoch (Epoch): Reference epoch for ephemeris generation
///     angle_format (AngleFormat): Format for angular inputs
///     pattern (WalkerPattern): Pattern type (DELTA for 360° RAAN, STAR for 180° RAAN)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///     # Walker Delta constellation (GPS-like)
///     gen = bh.WalkerConstellationGenerator(
///         t=24,                       # 24 total satellites
///         p=3,                        # 3 orbital planes
///         f=1,                        # phasing factor 1
///         semi_major_axis=bh.R_EARTH + 20200e3,  # GPS altitude
///         eccentricity=0.0,
///         inclination=55.0,
///         argument_of_perigee=0.0,
///         reference_raan=0.0,
///         reference_mean_anomaly=0.0,
///         epoch=epoch,
///         angle_format=bh.AngleFormat.DEGREES,
///         pattern=bh.WalkerPattern.DELTA,  # 360° RAAN spread
///     )
///
///     # Generate Keplerian propagators
///     propagators = gen.as_keplerian_propagators(60.0)
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "WalkerConstellationGenerator")]
#[derive(Clone)]
pub struct PyWalkerConstellationGenerator {
    pub(crate) generator: orbits::WalkerConstellationGenerator,
}

#[pymethods]
impl PyWalkerConstellationGenerator {
    #[new]
    #[pyo3(signature = (t, p, f, semi_major_axis, eccentricity, inclination, argument_of_perigee, reference_raan, reference_mean_anomaly, epoch, angle_format, pattern))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        t: usize,
        p: usize,
        f: usize,
        semi_major_axis: f64,
        eccentricity: f64,
        inclination: f64,
        argument_of_perigee: f64,
        reference_raan: f64,
        reference_mean_anomaly: f64,
        epoch: &PyEpoch,
        angle_format: &PyAngleFormat,
        pattern: &PyWalkerPattern,
    ) -> Self {
        Self {
            generator: orbits::WalkerConstellationGenerator::new(
                t,
                p,
                f,
                semi_major_axis,
                eccentricity,
                inclination,
                argument_of_perigee,
                reference_raan,
                reference_mean_anomaly,
                epoch.obj,
                angle_format.value,
                orbits::WalkerPattern::from(*pattern),
            ),
        }
    }

    /// Set a base name for satellite naming.
    ///
    /// When set, satellites will be named as "{base_name}-P{plane}-S{sat}"
    /// (e.g., "GPS-P0-S0", "GPS-P0-S1", "GPS-P1-S0", etc.)
    ///
    /// Args:
    ///     name (str): Base name for the constellation satellites
    ///
    /// Returns:
    ///     WalkerConstellationGenerator: Self with the base name set
    fn with_base_name(&self, name: &str) -> Self {
        Self {
            generator: self.generator.clone().with_base_name(name),
        }
    }

    /// Total number of satellites in the constellation.
    #[getter]
    fn total_satellites(&self) -> usize {
        self.generator.total_satellites
    }

    /// Number of orbital planes.
    #[getter]
    fn num_planes(&self) -> usize {
        self.generator.num_planes
    }

    /// Phasing factor.
    #[getter]
    fn phasing(&self) -> usize {
        self.generator.phasing
    }

    /// Number of satellites per plane (T/P).
    #[getter]
    fn satellites_per_plane(&self) -> usize {
        self.generator.satellites_per_plane()
    }

    /// Semi-major axis in meters.
    #[getter]
    fn semi_major_axis(&self) -> f64 {
        self.generator.semi_major_axis
    }

    /// Eccentricity.
    #[getter]
    fn eccentricity(&self) -> f64 {
        self.generator.eccentricity
    }

    /// Reference epoch.
    #[getter]
    fn epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.generator.epoch,
        }
    }

    /// Walker pattern (Delta or Star).
    #[getter]
    fn pattern(&self) -> PyWalkerPattern {
        PyWalkerPattern::from(self.generator.pattern)
    }

    /// Get Keplerian elements for a specific satellite.
    ///
    /// Args:
    ///     plane_index (int): Plane index (0 to P-1)
    ///     sat_index (int): Satellite index within plane (0 to T/P - 1)
    ///     angle_format (AngleFormat): Output angle format
    ///
    /// Returns:
    ///     np.ndarray: Keplerian elements [a, e, i, raan, argp, M]
    #[pyo3(signature = (plane_index, sat_index, angle_format))]
    fn satellite_elements<'py>(
        &self,
        py: Python<'py>,
        plane_index: usize,
        sat_index: usize,
        angle_format: &PyAngleFormat,
    ) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
        let elements = self.generator.satellite_elements(plane_index, sat_index);

        // Convert angles if needed
        let output = match angle_format.value {
            constants::AngleFormat::Degrees => Vector6::new(
                elements[0],
                elements[1],
                elements[2] * constants::RAD2DEG,
                elements[3] * constants::RAD2DEG,
                elements[4] * constants::RAD2DEG,
                elements[5] * constants::RAD2DEG,
            ),
            constants::AngleFormat::Radians => elements,
        };

        Ok(output.as_slice().to_pyarray(py))
    }

    /// Generate Keplerian propagators for all satellites in the constellation.
    ///
    /// Args:
    ///     step_size (float): Step size in seconds for propagation
    ///
    /// Returns:
    ///     list[KeplerianPropagator]: List of propagators, one per satellite
    #[pyo3(signature = (step_size))]
    fn as_keplerian_propagators(&self, step_size: f64) -> Vec<PyKeplerianPropagator> {
        self.generator
            .as_keplerian_propagators(step_size)
            .into_iter()
            .map(|p| PyKeplerianPropagator { propagator: p })
            .collect()
    }

    /// Generate SGP propagators for all satellites in the constellation.
    ///
    /// This method creates TLE data for each satellite using the provided
    /// drag and mean motion derivative parameters.
    ///
    /// Args:
    ///     step_size (float): Step size in seconds for propagation
    ///     bstar (float): B* drag term (Earth radii^-1)
    ///     ndt2 (float): First derivative of mean motion divided by 2 (rev/day²)
    ///     nddt6 (float): Second derivative of mean motion divided by 6 (rev/day³)
    ///
    /// Returns:
    ///     list[SGPPropagator]: List of propagators, one per satellite
    #[pyo3(signature = (step_size, bstar, ndt2, nddt6))]
    fn as_sgp_propagators(
        &self,
        step_size: f64,
        bstar: f64,
        ndt2: f64,
        nddt6: f64,
    ) -> PyResult<Vec<PySGPPropagator>> {
        self.generator
            .as_sgp_propagators(step_size, bstar, ndt2, nddt6)
            .map(|props| {
                props
                    .into_iter()
                    .map(|p| PySGPPropagator { propagator: p })
                    .collect()
            })
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Generate numerical orbit propagators for all satellites in the constellation.
    ///
    /// Args:
    ///     propagation_config (NumericalPropagationConfig): Propagation configuration
    ///     force_config (ForceModelConfig): Force model configuration
    ///     params (np.ndarray, optional): Parameter vector [mass, drag_area, Cd, srp_area, Cr]
    ///
    /// Returns:
    ///     list[DNumericalOrbitPropagator]: List of propagators, one per satellite
    #[pyo3(signature = (propagation_config, force_config, params=None))]
    fn as_numerical_propagators(
        &self,
        propagation_config: &PyNumericalPropagationConfig,
        force_config: &PyForceModelConfig,
        params: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Vec<PyNumericalOrbitPropagator>> {
        let params_dvec = match params {
            Some(p) => {
                let vec = pyany_to_f64_array1(p, None)?;
                Some(DVector::from_vec(vec))
            }
            None => None,
        };

        self.generator
            .as_numerical_propagators(
                propagation_config.config.clone(),
                force_config.config.clone(),
                params_dvec,
            )
            .map(|props| {
                props
                    .into_iter()
                    .map(|p| PyNumericalOrbitPropagator { propagator: p })
                    .collect()
            })
            .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}