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