// Import traits needed by propagator methods
use crate::orbits::traits::{StateProvider, OrbitPropagator};

/// Computes the orbital period of an object around Earth.
///
/// Uses rastro.constants.GM_EARTH as the standard gravitational parameter for the calculation.
///
/// Args:
///     a (float): The semi-major axis of the astronomical object in meters.
///
/// Returns:
///     (float): The orbital period of the astronomical object in seconds.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Calculate orbital period for ISS-like orbit (400 km altitude)
///     a = bh.R_EARTH + 400e3
///     period = bh.orbital_period(a)
///     print(f"Orbital period: {period/60:.2f} minutes")
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Calculate orbital period around the Moon
///     a = 1900000.0  # 1900 km semi-major axis
///     period = bh.orbital_period_general(a, bh.GM_MOON)
///     print(f"Lunar orbital period: {period/3600:.2f} hours")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(a, gm)")]
#[pyo3(name = "orbital_period_general")]
fn py_orbital_period_general(a: f64, gm: f64) -> PyResult<f64> {
    Ok(orbits::orbital_period_general(a, gm))
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
///     a (float): The semi-major axis of the astronomical object in meters.
///     angle_format (AngleFormat): Return output in AngleFormat.DEGREES or AngleFormat.RADIANS.
///
/// Returns:
///     float: The mean motion of the astronomical object in radians or degrees.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Calculate mean motion for geostationary orbit (35786 km altitude)
///     a = bh.R_EARTH + 35786e3
///     n = bh.mean_motion(a, bh.AngleFormat.DEGREES)
///     print(f"Mean motion: {n:.6f} deg/s")
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Calculate mean motion for a Mars orbiter
///     a = 4000000.0  # 4000 km semi-major axis
///     n = bh.mean_motion_general(a, bh.GM_MARS, bh.AngleFormat.RADIANS)
///     print(f"Mean motion: {n:.6f} rad/s")
///     ```
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
///     gm (float): The standard gravitational parameter of primary body in m³/s².
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
///     gm (float): The standard gravitational parameter of primary body in m³/s².
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
///     a (float): The semi-major axis of the astronomical object in meters.
///     e (float): The eccentricity of the astronomical object's orbit (dimensionless).
///
/// Returns:
///     float: The magnitude of velocity of the object at perigee in m/s.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Calculate perigee velocity for Molniya orbit (highly elliptical)
///     a = 26554000.0  # meters
///     e = 0.72  # high eccentricity
///     v_peri = bh.perigee_velocity(a, e)
///     print(f"Perigee velocity: {v_peri:.2f} m/s")
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Calculate periapsis velocity for a comet around the Sun
///     a = 5e11  # 5 AU semi-major axis (meters)
///     e = 0.95  # highly elliptical
///     v_peri = bh.periapsis_velocity(a, e, bh.GM_SUN)
///     print(f"Periapsis velocity: {v_peri/1000:.2f} km/s")
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Calculate periapsis distance for an elliptical orbit
///     a = 8000000.0  # 8000 km semi-major axis
///     e = 0.2  # moderate eccentricity
///     r_peri = bh.periapsis_distance(a, e)
///     print(f"Periapsis distance: {r_peri/1000:.2f} km")
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Calculate apogee velocity for GTO (Geostationary Transfer Orbit)
///     a = 24400000.0  # meters
///     e = 0.73  # high eccentricity
///     v_apo = bh.apogee_velocity(a, e)
///     print(f"Apogee velocity: {v_apo:.2f} m/s")
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Calculate apoapsis velocity for a Martian satellite
///     a = 10000000.0  # 10000 km semi-major axis
///     e = 0.3
///     v_apo = bh.apoapsis_velocity(a, e, bh.GM_MARS)
///     print(f"Apoapsis velocity: {v_apo/1000:.2f} km/s")
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Calculate apoapsis distance
///     a = 8000000.0  # 8000 km semi-major axis
///     e = 0.2  # moderate eccentricity
///     r_apo = bh.apoapsis_distance(a, e)
///     print(f"Apoapsis distance: {r_apo/1000:.2f} km")
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Calculate sun-synchronous inclination for typical Earth observation satellite (600 km)
///     a = bh.R_EARTH + 600e3
///     e = 0.001  # nearly circular
///     inc = bh.sun_synchronous_inclination(a, e, bh.AngleFormat.DEGREES)
///     print(f"Sun-synchronous inclination: {inc:.2f} degrees")
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///     import math
///
///     # Convert eccentric to mean anomaly
///     E = math.pi / 4  # 45 degrees eccentric anomaly
///     e = 0.1  # eccentricity
///     M = bh.anomaly_eccentric_to_mean(E, e, bh.AngleFormat.RADIANS)
///     print(f"Mean anomaly: {M:.4f} radians")
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Convert mean to eccentric anomaly (solves Kepler's equation)
///     M = 1.5  # mean anomaly in radians
///     e = 0.3  # eccentricity
///     E = bh.anomaly_mean_to_eccentric(M, e, bh.AngleFormat.RADIANS)
///     print(f"Eccentric anomaly: {E:.4f} radians")
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///     import math
///
///     # Convert true to eccentric anomaly
///     nu = math.pi / 3  # 60 degrees true anomaly
///     e = 0.2  # eccentricity
///     E = bh.anomaly_true_to_eccentric(nu, e, bh.AngleFormat.RADIANS)
///     print(f"Eccentric anomaly: {E:.4f} radians")
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///     import math
///
///     # Convert eccentric to true anomaly
///     E = math.pi / 4  # 45 degrees eccentric anomaly
///     e = 0.4  # eccentricity
///     nu = bh.anomaly_eccentric_to_true(E, e, bh.AngleFormat.RADIANS)
///     print(f"True anomaly: {nu:.4f} radians")
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///     import math
///
///     # Convert true to mean anomaly
///     nu = math.pi / 2  # 90 degrees true anomaly
///     e = 0.15  # eccentricity
///     M = bh.anomaly_true_to_mean(nu, e, bh.AngleFormat.RADIANS)
///     print(f"Mean anomaly: {M:.4f} radians")
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Convert mean to true anomaly (combines Kepler's equation + eccentric anomaly conversion)
///     M = 2.0  # mean anomaly in radians
///     e = 0.25  # eccentricity
///     nu = bh.anomaly_mean_to_true(M, e, bh.AngleFormat.RADIANS)
///     print(f"True anomaly: {nu:.4f} radians")
///     ```
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
    pub propagator: orbits::SGPPropagator,
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
    #[getter]
    pub fn id(&self) -> Option<u64> {
        use crate::utils::Identifiable;
        self.propagator.get_id()
    }

    /// Get the current name.
    ///
    /// Returns:
    ///     str or None: The name, or None if not set.
    #[getter]
    pub fn name(&self) -> Option<String> {
        use crate::utils::Identifiable;
        self.propagator.get_name().map(|s| s.to_string())
    }

    /// Get the current UUID.
    ///
    /// Returns:
    ///     str or None: The UUID as a string, or None if not set.
    #[getter]
    pub fn uuid(&self) -> Option<String> {
        use crate::utils::Identifiable;
        self.propagator.get_uuid().map(|u| u.to_string())
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
    pub propagator: orbits::KeplerianPropagator,
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
    #[getter]
    pub fn id(&self) -> Option<u64> {
        use crate::utils::Identifiable;
        self.propagator.get_id()
    }

    /// Get the current name.
    ///
    /// Returns:
    ///     str or None: The name, or None if not set.
    #[getter]
    pub fn name(&self) -> Option<String> {
        use crate::utils::Identifiable;
        self.propagator.get_name().map(|s| s.to_string())
    }

    /// Get the current UUID.
    ///
    /// Returns:
    ///     str or None: The UUID as a string, or None if not set.
    #[getter]
    pub fn uuid(&self) -> Option<String> {
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