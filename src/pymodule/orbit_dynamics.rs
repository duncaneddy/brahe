// ============================================================================
// Ephemerides Python Bindings
// ============================================================================

/// Calculate the position of the Sun in the GCRF inertial frame using low-precision analytical methods.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate the Sun's position
///
/// Returns:
///     np.ndarray: Position of the Sun in the GCRF frame. Units: (m)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_sun = bh.sun_position(epc)
///     ```
#[pyfunction]
#[pyo3(name = "sun_position")]
fn py_sun_position<'py>(py: Python<'py>, epc: &PyEpoch) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let r = orbit_dynamics::sun_position(epc.obj);
    Ok(vector_to_numpy!(py, r, 3, f64))
}

/// Calculate the position of the Moon in the GCRF inertial frame using low-precision analytical methods.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate the Moon's position
///
/// Returns:
///     np.ndarray: Position of the Moon in the GCRF frame. Units: (m)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_moon = bh.moon_position(epc)
///     ```
#[pyfunction]
#[pyo3(name = "moon_position")]
fn py_moon_position<'py>(py: Python<'py>, epc: &PyEpoch) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let r = orbit_dynamics::moon_position(epc.obj);
    Ok(vector_to_numpy!(py, r, 3, f64))
}

/// Calculate the position of the Sun in the GCRF inertial frame using NAIF DE440s ephemeris.
///
/// This function uses the high-precision NAIF DE440s ephemeris kernel for solar position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// If the ephemeris has not been initialized, it will be automatically loaded on the
/// first call. For more control over initialization and error handling, use
/// `initialize_ephemeris()` explicitly.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate the Sun's position
///
/// Returns:
///     np.ndarray: Position of the Sun in the GCRF frame. Units: (m)
///
/// Raises:
///     Exception: If the DE440s kernel cannot be loaded or ephemeris query fails
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Optional: Pre-initialize for better error handling
///     bh.initialize_ephemeris()
///
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_sun = bh.sun_position_de440s(epc)
///     ```
#[pyfunction]
#[pyo3(name = "sun_position_de440s")]
fn py_sun_position_de440s<'py>(py: Python<'py>, epc: &PyEpoch) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let r = orbit_dynamics::sun_position_de440s(epc.obj);
    Ok(vector_to_numpy!(py, r, 3, f64))
}

/// Calculate the position of the Moon in the GCRF inertial frame using NAIF DE440s ephemeris.
///
/// This function uses the high-precision NAIF DE440s ephemeris kernel for lunar position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// If the ephemeris has not been initialized, it will be automatically loaded on the
/// first call. For more control over initialization and error handling, use
/// `initialize_ephemeris()` explicitly.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate the Moon's position
///
/// Returns:
///     np.ndarray: Position of the Moon in the GCRF frame. Units: (m)
///
/// Raises:
///     Exception: If the DE440s kernel cannot be loaded or ephemeris query fails
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Optional: Pre-initialize for better error handling
///     bh.initialize_ephemeris()
///
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_moon = bh.moon_position_de440s(epc)
///     ```
#[pyfunction]
#[pyo3(name = "moon_position_de440s")]
fn py_moon_position_de440s<'py>(py: Python<'py>, epc: &PyEpoch) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let r = orbit_dynamics::moon_position_de440s(epc.obj);
    Ok(vector_to_numpy!(py, r, 3, f64))
}

/// Calculate the position of Mercury in the GCRF inertial frame using NAIF DE440s ephemeris.
///
/// This function uses the high-precision NAIF DE440s ephemeris kernel for Mercury position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// If the ephemeris has not been initialized, it will be automatically loaded on the
/// first call. For more control over initialization and error handling, use
/// `initialize_ephemeris()` explicitly.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate Mercury's position
///
/// Returns:
///     np.ndarray: Position of Mercury in the GCRF frame. Units: (m)
///
/// Raises:
///     Exception: If the DE440s kernel cannot be loaded or ephemeris query fails
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Optional: Pre-initialize for better error handling
///     bh.initialize_ephemeris()
///
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_mercury = bh.mercury_position_de440s(epc)
///     ```
#[pyfunction]
#[pyo3(name = "mercury_position_de440s")]
fn py_mercury_position_de440s<'py>(py: Python<'py>, epc: &PyEpoch) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let r = orbit_dynamics::mercury_position_de440s(epc.obj);
    Ok(vector_to_numpy!(py, r, 3, f64))
}

/// Calculate the position of Venus in the GCRF inertial frame using NAIF DE440s ephemeris.
///
/// This function uses the high-precision NAIF DE440s ephemeris kernel for Venus position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// If the ephemeris has not been initialized, it will be automatically loaded on the
/// first call. For more control over initialization and error handling, use
/// `initialize_ephemeris()` explicitly.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate Venus's position
///
/// Returns:
///     np.ndarray: Position of Venus in the GCRF frame. Units: (m)
///
/// Raises:
///     Exception: If the DE440s kernel cannot be loaded or ephemeris query fails
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Optional: Pre-initialize for better error handling
///     bh.initialize_ephemeris()
///
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_venus = bh.venus_position_de440s(epc)
///     ```
#[pyfunction]
#[pyo3(name = "venus_position_de440s")]
fn py_venus_position_de440s<'py>(py: Python<'py>, epc: &PyEpoch) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let r = orbit_dynamics::venus_position_de440s(epc.obj);
    Ok(vector_to_numpy!(py, r, 3, f64))
}

/// Calculate the position of Mars in the GCRF inertial frame using NAIF DE440s ephemeris.
///
/// This function uses the high-precision NAIF DE440s ephemeris kernel for Mars position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// If the ephemeris has not been initialized, it will be automatically loaded on the
/// first call. For more control over initialization and error handling, use
/// `initialize_ephemeris()` explicitly.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate Mars's position
///
/// Returns:
///     np.ndarray: Position of Mars in the GCRF frame. Units: (m)
///
/// Raises:
///     Exception: If the DE440s kernel cannot be loaded or ephemeris query fails
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Optional: Pre-initialize for better error handling
///     bh.initialize_ephemeris()
///
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_mars = bh.mars_position_de440s(epc)
///     ```
#[pyfunction]
#[pyo3(name = "mars_position_de440s")]
fn py_mars_position_de440s<'py>(py: Python<'py>, epc: &PyEpoch) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let r = orbit_dynamics::mars_position_de440s(epc.obj);
    Ok(vector_to_numpy!(py, r, 3, f64))
}

/// Calculate the position of Jupiter in the GCRF inertial frame using NAIF DE440s ephemeris.
///
/// This function uses the high-precision NAIF DE440s ephemeris kernel for Jupiter position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// If the ephemeris has not been initialized, it will be automatically loaded on the
/// first call. For more control over initialization and error handling, use
/// `initialize_ephemeris()` explicitly.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate Jupiter's position
///
/// Returns:
///     np.ndarray: Position of Jupiter in the GCRF frame. Units: (m)
///
/// Raises:
///     Exception: If the DE440s kernel cannot be loaded or ephemeris query fails
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Optional: Pre-initialize for better error handling
///     bh.initialize_ephemeris()
///
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_jupiter = bh.jupiter_position_de440s(epc)
///     ```
#[pyfunction]
#[pyo3(name = "jupiter_position_de440s")]
fn py_jupiter_position_de440s<'py>(py: Python<'py>, epc: &PyEpoch) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let r = orbit_dynamics::jupiter_position_de440s(epc.obj);
    Ok(vector_to_numpy!(py, r, 3, f64))
}

/// Calculate the position of Saturn in the GCRF inertial frame using NAIF DE440s ephemeris.
///
/// This function uses the high-precision NAIF DE440s ephemeris kernel for Saturn position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// If the ephemeris has not been initialized, it will be automatically loaded on the
/// first call. For more control over initialization and error handling, use
/// `initialize_ephemeris()` explicitly.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate Saturn's position
///
/// Returns:
///     np.ndarray: Position of Saturn in the GCRF frame. Units: (m)
///
/// Raises:
///     Exception: If the DE440s kernel cannot be loaded or ephemeris query fails
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Optional: Pre-initialize for better error handling
///     bh.initialize_ephemeris()
///
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_saturn = bh.saturn_position_de440s(epc)
///     ```
#[pyfunction]
#[pyo3(name = "saturn_position_de440s")]
fn py_saturn_position_de440s<'py>(py: Python<'py>, epc: &PyEpoch) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let r = orbit_dynamics::saturn_position_de440s(epc.obj);
    Ok(vector_to_numpy!(py, r, 3, f64))
}

/// Calculate the position of Uranus in the GCRF inertial frame using NAIF DE440s ephemeris.
///
/// This function uses the high-precision NAIF DE440s ephemeris kernel for Uranus position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// If the ephemeris has not been initialized, it will be automatically loaded on the
/// first call. For more control over initialization and error handling, use
/// `initialize_ephemeris()` explicitly.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate Uranus's position
///
/// Returns:
///     np.ndarray: Position of Uranus in the GCRF frame. Units: (m)
///
/// Raises:
///     Exception: If the DE440s kernel cannot be loaded or ephemeris query fails
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Optional: Pre-initialize for better error handling
///     bh.initialize_ephemeris()
///
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_uranus = bh.uranus_position_de440s(epc)
///     ```
#[pyfunction]
#[pyo3(name = "uranus_position_de440s")]
fn py_uranus_position_de440s<'py>(py: Python<'py>, epc: &PyEpoch) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let r = orbit_dynamics::uranus_position_de440s(epc.obj);
    Ok(vector_to_numpy!(py, r, 3, f64))
}

/// Calculate the position of Neptune in the GCRF inertial frame using NAIF DE440s ephemeris.
///
/// This function uses the high-precision NAIF DE440s ephemeris kernel for Neptune position
/// computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// If the ephemeris has not been initialized, it will be automatically loaded on the
/// first call. For more control over initialization and error handling, use
/// `initialize_ephemeris()` explicitly.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate Neptune's position
///
/// Returns:
///     np.ndarray: Position of Neptune in the GCRF frame. Units: (m)
///
/// Raises:
///     Exception: If the DE440s kernel cannot be loaded or ephemeris query fails
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Optional: Pre-initialize for better error handling
///     bh.initialize_ephemeris()
///
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_neptune = bh.neptune_position_de440s(epc)
///     ```
#[pyfunction]
#[pyo3(name = "neptune_position_de440s")]
fn py_neptune_position_de440s<'py>(py: Python<'py>, epc: &PyEpoch) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let r = orbit_dynamics::neptune_position_de440s(epc.obj);
    Ok(vector_to_numpy!(py, r, 3, f64))
}

/// Calculate the position of the Solar System Barycenter in the GCRF inertial frame using NAIF DE440s ephemeris.
///
/// This function uses the high-precision NAIF DE440s ephemeris kernel for Solar System Barycenter
/// position computation. The kernel is loaded once and cached in a global thread-safe context,
/// making subsequent calls very efficient.
///
/// If the ephemeris has not been initialized, it will be automatically loaded on the
/// first call. For more control over initialization and error handling, use
/// `initialize_ephemeris()` explicitly.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate the Solar System Barycenter position
///
/// Returns:
///     np.ndarray: Position of the Solar System Barycenter in the GCRF frame. Units: (m)
///
/// Raises:
///     Exception: If the DE440s kernel cannot be loaded or ephemeris query fails
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Optional: Pre-initialize for better error handling
///     bh.initialize_ephemeris()
///
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_ssb = bh.solar_system_barycenter_position_de440s(epc)
///     ```
#[pyfunction]
#[pyo3(name = "solar_system_barycenter_position_de440s")]
fn py_solar_system_barycenter_position_de440s<'py>(py: Python<'py>, epc: &PyEpoch) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let r = orbit_dynamics::solar_system_barycenter_position_de440s(epc.obj);
    Ok(vector_to_numpy!(py, r, 3, f64))
}

/// Convenience alias for `solar_system_barycenter_position_de440s`.
///
/// Calculate the position of the Solar System Barycenter in the GCRF inertial frame using
/// NAIF DE440s ephemeris.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate the Solar System Barycenter position
///
/// Returns:
///     np.ndarray: Position of the Solar System Barycenter in the GCRF frame. Units: (m)
///
/// Raises:
///     Exception: If the DE440s kernel cannot be loaded or ephemeris query fails
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Optional: Pre-initialize for better error handling
///     bh.initialize_ephemeris()
///
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_ssb = bh.ssb_position_de440s(epc)
///     ```
#[pyfunction]
#[pyo3(name = "ssb_position_de440s")]
fn py_ssb_position_de440s<'py>(py: Python<'py>, epc: &PyEpoch) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let r = orbit_dynamics::ssb_position_de440s(epc.obj);
    Ok(vector_to_numpy!(py, r, 3, f64))
}

/// Initialize the global ephemeris provider with the default DE440s kernel.
///
/// This function downloads (or uses a cached copy of) the NAIF DE440s ephemeris
/// kernel and sets it as the global Almanac provider. This initialization is
/// optional - if not called, the Almanac will be lazily initialized on the first
/// call to `sun_position_de440s()` or `moon_position_de440s()`.
///
/// Calling this function explicitly is recommended when you want to:
/// - Control when the kernel download/loading occurs (avoid latency on first use)
/// - Handle initialization errors explicitly
/// - Pre-load the kernel during application startup
///
/// Returns:
///     None: Successfully initialized the ephemeris
///
/// Raises:
///     Exception: If kernel download or loading failed
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Initialize at application startup
///     bh.initialize_ephemeris()
///
///     # Now use DE440s ephemeris functions
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_sun = bh.sun_position_de440s(epc)
///     r_moon = bh.moon_position_de440s(epc)
///     ```
#[pyfunction]
#[pyo3(name = "initialize_ephemeris")]
fn py_initialize_ephemeris() -> PyResult<()> {
    orbit_dynamics::initialize_ephemeris()
        .map_err(|e| exceptions::PyRuntimeError::new_err(format!("Failed to initialize ephemeris: {}", e)))
}

// ============================================================================
// Third-Body Acceleration Python Bindings
// ============================================================================

/// Calculate the acceleration due to the Sun on an object using low-precision analytical ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for r_object.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate the Sun's position
///     r_object (np.ndarray): Position (length 3) or state (length 6) of the object in the GCRF frame. Units: (m)
///
/// Returns:
///     np.ndarray: Acceleration due to the Sun. Units: (m/s²)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///
///     # Using position vector
///     r_object = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])
///     a = bh.accel_third_body_sun(epc, r_object)
///
///     # Or using state vector directly
///     x_object = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
///     a = bh.accel_third_body_sun(epc, x_object)
///     ```
#[pyfunction]
#[pyo3(name = "accel_third_body_sun")]
fn py_accel_third_body_sun<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    r_object: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let len = r_object.len();
    let a = if len == 3 {
        let r_obj = numpy_to_vector3!(r_object);
        orbit_dynamics::accel_third_body_sun(epc.obj, r_obj)
    } else if len == 6 {
        let x_obj = numpy_to_vector6!(r_object);
        orbit_dynamics::accel_third_body_sun(epc.obj, x_obj)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_object must be length 3 (position) or 6 (state)"
        ));
    };
    Ok(vector_to_numpy!(py, a, 3, f64))
}

/// Calculate the acceleration due to the Moon on an object using low-precision analytical ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for r_object.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate the Moon's position
///     r_object (np.ndarray): Position (length 3) or state (length 6) of the object in the GCRF frame. Units: (m)
///
/// Returns:
///     np.ndarray: Acceleration due to the Moon. Units: (m/s²)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_object = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])
///     a = bh.accel_third_body_moon(epc, r_object)
///     ```
#[pyfunction]
#[pyo3(name = "accel_third_body_moon")]
fn py_accel_third_body_moon<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    r_object: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let len = r_object.len();
    let a = if len == 3 {
        let r_obj = numpy_to_vector3!(r_object);
        orbit_dynamics::accel_third_body_moon(epc.obj, r_obj)
    } else if len == 6 {
        let x_obj = numpy_to_vector6!(r_object);
        orbit_dynamics::accel_third_body_moon(epc.obj, x_obj)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_object must be length 3 (position) or 6 (state)"
        ));
    };
    Ok(vector_to_numpy!(py, a, 3, f64))
}

/// Calculate the acceleration due to the Sun on an object using DE440s high-precision ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for r_object.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate the Sun's position
///     r_object (np.ndarray): Position (length 3) or state (length 6) of the object in the GCRF frame. Units: (m)
///
/// Returns:
///     np.ndarray: Acceleration due to the Sun. Units: (m/s²)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_ephemeris()
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_object = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])
///     a = bh.accel_third_body_sun_de440s(epc, r_object)
///     ```
#[pyfunction]
#[pyo3(name = "accel_third_body_sun_de440s")]
fn py_accel_third_body_sun_de440s<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    r_object: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let len = r_object.len();
    let a = if len == 3 {
        let r_obj = numpy_to_vector3!(r_object);
        orbit_dynamics::accel_third_body_sun_de440s(epc.obj, r_obj)
    } else if len == 6 {
        let x_obj = numpy_to_vector6!(r_object);
        orbit_dynamics::accel_third_body_sun_de440s(epc.obj, x_obj)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_object must be length 3 (position) or 6 (state)"
        ));
    };
    Ok(vector_to_numpy!(py, a, 3, f64))
}

/// Calculate the acceleration due to the Moon on an object using DE440s high-precision ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for r_object.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate the Moon's position
///     r_object (np.ndarray): Position (length 3) or state (length 6) of the object in the GCRF frame. Units: (m)
///
/// Returns:
///     np.ndarray: Acceleration due to the Moon. Units: (m/s²)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_ephemeris()
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_object = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])
///     a = bh.accel_third_body_moon_de440s(epc, r_object)
///     ```
#[pyfunction]
#[pyo3(name = "accel_third_body_moon_de440s")]
fn py_accel_third_body_moon_de440s<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    r_object: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let len = r_object.len();
    let a = if len == 3 {
        let r_obj = numpy_to_vector3!(r_object);
        orbit_dynamics::accel_third_body_moon_de440s(epc.obj, r_obj)
    } else if len == 6 {
        let x_obj = numpy_to_vector6!(r_object);
        orbit_dynamics::accel_third_body_moon_de440s(epc.obj, x_obj)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_object must be length 3 (position) or 6 (state)"
        ));
    };
    Ok(vector_to_numpy!(py, a, 3, f64))
}

/// Calculate the acceleration due to Mercury on an object using DE440s ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for r_object.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate Mercury's position
///     r_object (np.ndarray): Position (length 3) or state (length 6) of the object in the GCRF frame. Units: (m)
///
/// Returns:
///     np.ndarray: Acceleration due to Mercury. Units: (m/s²)
#[pyfunction]
#[pyo3(name = "accel_third_body_mercury_de440s")]
fn py_accel_third_body_mercury_de440s<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    r_object: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let len = r_object.len();
    let a = if len == 3 {
        let r_obj = numpy_to_vector3!(r_object);
        orbit_dynamics::accel_third_body_mercury_de440s(epc.obj, r_obj)
    } else if len == 6 {
        let x_obj = numpy_to_vector6!(r_object);
        orbit_dynamics::accel_third_body_mercury_de440s(epc.obj, x_obj)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_object must be length 3 (position) or 6 (state)"
        ));
    };
    Ok(vector_to_numpy!(py, a, 3, f64))
}

/// Calculate the acceleration due to Venus on an object using DE440s ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for r_object.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate Venus's position
///     r_object (np.ndarray): Position (length 3) or state (length 6) of the object in the GCRF frame. Units: (m)
///
/// Returns:
///     np.ndarray: Acceleration due to Venus. Units: (m/s²)
#[pyfunction]
#[pyo3(name = "accel_third_body_venus_de440s")]
fn py_accel_third_body_venus_de440s<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    r_object: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let len = r_object.len();
    let a = if len == 3 {
        let r_obj = numpy_to_vector3!(r_object);
        orbit_dynamics::accel_third_body_venus_de440s(epc.obj, r_obj)
    } else if len == 6 {
        let x_obj = numpy_to_vector6!(r_object);
        orbit_dynamics::accel_third_body_venus_de440s(epc.obj, x_obj)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_object must be length 3 (position) or 6 (state)"
        ));
    };
    Ok(vector_to_numpy!(py, a, 3, f64))
}

/// Calculate the acceleration due to Mars on an object using DE440s ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for r_object.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate Mars's position
///     r_object (np.ndarray): Position (length 3) or state (length 6) of the object in the GCRF frame. Units: (m)
///
/// Returns:
///     np.ndarray: Acceleration due to Mars. Units: (m/s²)
#[pyfunction]
#[pyo3(name = "accel_third_body_mars_de440s")]
fn py_accel_third_body_mars_de440s<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    r_object: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let len = r_object.len();
    let a = if len == 3 {
        let r_obj = numpy_to_vector3!(r_object);
        orbit_dynamics::accel_third_body_mars_de440s(epc.obj, r_obj)
    } else if len == 6 {
        let x_obj = numpy_to_vector6!(r_object);
        orbit_dynamics::accel_third_body_mars_de440s(epc.obj, x_obj)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_object must be length 3 (position) or 6 (state)"
        ));
    };
    Ok(vector_to_numpy!(py, a, 3, f64))
}

/// Calculate the acceleration due to Jupiter on an object using DE440s ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for r_object.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate Jupiter's position
///     r_object (np.ndarray): Position (length 3) or state (length 6) of the object in the GCRF frame. Units: (m)
///
/// Returns:
///     np.ndarray: Acceleration due to Jupiter. Units: (m/s²)
#[pyfunction]
#[pyo3(name = "accel_third_body_jupiter_de440s")]
fn py_accel_third_body_jupiter_de440s<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    r_object: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let len = r_object.len();
    let a = if len == 3 {
        let r_obj = numpy_to_vector3!(r_object);
        orbit_dynamics::accel_third_body_jupiter_de440s(epc.obj, r_obj)
    } else if len == 6 {
        let x_obj = numpy_to_vector6!(r_object);
        orbit_dynamics::accel_third_body_jupiter_de440s(epc.obj, x_obj)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_object must be length 3 (position) or 6 (state)"
        ));
    };
    Ok(vector_to_numpy!(py, a, 3, f64))
}

/// Calculate the acceleration due to Saturn on an object using DE440s ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for r_object.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate Saturn's position
///     r_object (np.ndarray): Position (length 3) or state (length 6) of the object in the GCRF frame. Units: (m)
///
/// Returns:
///     np.ndarray: Acceleration due to Saturn. Units: (m/s²)
#[pyfunction]
#[pyo3(name = "accel_third_body_saturn_de440s")]
fn py_accel_third_body_saturn_de440s<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    r_object: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let len = r_object.len();
    let a = if len == 3 {
        let r_obj = numpy_to_vector3!(r_object);
        orbit_dynamics::accel_third_body_saturn_de440s(epc.obj, r_obj)
    } else if len == 6 {
        let x_obj = numpy_to_vector6!(r_object);
        orbit_dynamics::accel_third_body_saturn_de440s(epc.obj, x_obj)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_object must be length 3 (position) or 6 (state)"
        ));
    };
    Ok(vector_to_numpy!(py, a, 3, f64))
}

/// Calculate the acceleration due to Uranus on an object using DE440s ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for r_object.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate Uranus's position
///     r_object (np.ndarray): Position (length 3) or state (length 6) of the object in the GCRF frame. Units: (m)
///
/// Returns:
///     np.ndarray: Acceleration due to Uranus. Units: (m/s²)
#[pyfunction]
#[pyo3(name = "accel_third_body_uranus_de440s")]
fn py_accel_third_body_uranus_de440s<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    r_object: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let len = r_object.len();
    let a = if len == 3 {
        let r_obj = numpy_to_vector3!(r_object);
        orbit_dynamics::accel_third_body_uranus_de440s(epc.obj, r_obj)
    } else if len == 6 {
        let x_obj = numpy_to_vector6!(r_object);
        orbit_dynamics::accel_third_body_uranus_de440s(epc.obj, x_obj)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_object must be length 3 (position) or 6 (state)"
        ));
    };
    Ok(vector_to_numpy!(py, a, 3, f64))
}

/// Calculate the acceleration due to Neptune on an object using DE440s ephemerides.
///
/// Accepts either a 3D position vector or a 6D state vector for r_object.
///
/// Args:
///     epc (Epoch): Epoch at which to calculate Neptune's position
///     r_object (np.ndarray): Position (length 3) or state (length 6) of the object in the GCRF frame. Units: (m)
///
/// Returns:
///     np.ndarray: Acceleration due to Neptune. Units: (m/s²)
#[pyfunction]
#[pyo3(name = "accel_third_body_neptune_de440s")]
fn py_accel_third_body_neptune_de440s<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    r_object: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let len = r_object.len();
    let a = if len == 3 {
        let r_obj = numpy_to_vector3!(r_object);
        orbit_dynamics::accel_third_body_neptune_de440s(epc.obj, r_obj)
    } else if len == 6 {
        let x_obj = numpy_to_vector6!(r_object);
        orbit_dynamics::accel_third_body_neptune_de440s(epc.obj, x_obj)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_object must be length 3 (position) or 6 (state)"
        ));
    };
    Ok(vector_to_numpy!(py, a, 3, f64))
}

// ============================================================================
// Gravity Acceleration Python Bindings
// ============================================================================

/// Compute acceleration due to point-mass gravity.
///
/// Accepts either a 3D position vector or a 6D state vector for r_object.
///
/// Args:
///     r_object (np.ndarray): Position (length 3) or state (length 6) of the object. Units: (m)
///     r_central_body (np.ndarray): Position vector of the central body. Units: (m)
///     gm (float): Gravitational parameter. Units: (m³/s²)
///
/// Returns:
///     np.ndarray: Acceleration due to gravity. Units: (m/s²)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     r_object = np.array([bh.R_EARTH, 0.0, 0.0])
///     r_central_body = np.array([0.0, 0.0, 0.0])
///     a_grav = bh.accel_point_mass_gravity(r_object, r_central_body, bh.GM_EARTH)
///     ```
#[pyfunction]
#[pyo3(name = "accel_point_mass_gravity")]
fn py_accel_point_mass_gravity<'py>(
    py: Python<'py>,
    r_object: PyReadonlyArray1<f64>,
    r_central_body: PyReadonlyArray1<f64>,
    gm: f64,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let len = r_object.len();
    let r_cb = numpy_to_vector3!(r_central_body);
    let a = if len == 3 {
        let r_obj = numpy_to_vector3!(r_object);
        orbit_dynamics::accel_point_mass_gravity(r_obj, r_cb, gm)
    } else if len == 6 {
        let x_obj = numpy_to_vector6!(r_object);
        orbit_dynamics::accel_point_mass_gravity(x_obj, r_cb, gm)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_object must be length 3 (position) or 6 (state)"
        ));
    };
    Ok(vector_to_numpy!(py, a, 3, f64))
}

// ============================================================================
// Spherical Harmonic Gravity Model Enums
// ============================================================================

/// Default gravity models packaged with Brahe.
///
/// These models provide varying levels of fidelity for Earth's gravitational field.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "DefaultGravityModel")]
#[derive(Clone)]
pub struct PyDefaultGravityModel {
    pub(crate) model: orbit_dynamics::DefaultGravityModel,
}

#[pymethods]
impl PyDefaultGravityModel {
    /// Earth Gravitational Model 2008, 360x360 degree and order.
    ///
    /// High-fidelity gravity model from the National Geospatial-Intelligence Agency (NGA).
    /// Best accuracy but computationally expensive for high degree/order evaluations.
    #[classattr]
    #[allow(non_snake_case)]
    fn EGM2008_360() -> Self {
        PyDefaultGravityModel {
            model: orbit_dynamics::DefaultGravityModel::EGM2008_360,
        }
    }

    /// Goddard Earth Model 2005S, 180x180 degree and order.
    ///
    /// Medium-fidelity gravity model from NASA Goddard Space Flight Center.
    /// Balances accuracy with computational efficiency.
    #[classattr]
    #[allow(non_snake_case)]
    fn GGM05S() -> Self {
        PyDefaultGravityModel {
            model: orbit_dynamics::DefaultGravityModel::GGM05S,
        }
    }

    /// Joint Gravity Model 3, 70x70 degree and order.
    ///
    /// Lower-fidelity gravity model suitable for fast computations.
    /// Good for preliminary analysis or real-time applications.
    #[classattr]
    #[allow(non_snake_case)]
    fn JGM3() -> Self {
        PyDefaultGravityModel {
            model: orbit_dynamics::DefaultGravityModel::JGM3,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.model)
    }

    fn __repr__(&self) -> String {
        format!("DefaultGravityModel.{:?}", self.model)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(std::mem::discriminant(&self.model) == std::mem::discriminant(&other.model)),
            CompareOp::Ne => Ok(std::mem::discriminant(&self.model) != std::mem::discriminant(&other.model)),
            _ => Err(exceptions::PyNotImplementedError::new_err("Comparison not supported")),
        }
    }
}

/// Tide system convention used in gravity model.
///
/// Different gravity models use different conventions for handling permanent tidal deformation.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "GravityModelTideSystem")]
#[derive(Clone)]
pub struct PyGravityModelTideSystem {
    pub(crate) tide_system: orbit_dynamics::GravityModelTideSystem,
}

#[pymethods]
impl PyGravityModelTideSystem {
    /// Zero-tide system.
    ///
    /// Includes permanent tidal deformation but removes the permanent indirect effect on gravity.
    #[classattr]
    #[allow(non_snake_case)]
    fn ZeroTide() -> Self {
        PyGravityModelTideSystem {
            tide_system: orbit_dynamics::GravityModelTideSystem::ZeroTide,
        }
    }

    /// Tide-free system.
    ///
    /// Permanent tidal effects are removed. This is the most common convention.
    #[classattr]
    #[allow(non_snake_case)]
    fn TideFree() -> Self {
        PyGravityModelTideSystem {
            tide_system: orbit_dynamics::GravityModelTideSystem::TideFree,
        }
    }

    /// Mean-tide system.
    ///
    /// Time-averaged permanent tide is included.
    #[classattr]
    #[allow(non_snake_case)]
    fn MeanTide() -> Self {
        PyGravityModelTideSystem {
            tide_system: orbit_dynamics::GravityModelTideSystem::MeanTide,
        }
    }

    /// Unknown tide system.
    ///
    /// Tide system convention is unspecified or not documented.
    #[classattr]
    #[allow(non_snake_case)]
    fn Unknown() -> Self {
        PyGravityModelTideSystem {
            tide_system: orbit_dynamics::GravityModelTideSystem::Unknown,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.tide_system)
    }

    fn __repr__(&self) -> String {
        format!("GravityModelTideSystem.{:?}", self.tide_system)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(std::mem::discriminant(&self.tide_system) == std::mem::discriminant(&other.tide_system)),
            CompareOp::Ne => Ok(std::mem::discriminant(&self.tide_system) != std::mem::discriminant(&other.tide_system)),
            _ => Err(exceptions::PyNotImplementedError::new_err("Comparison not supported")),
        }
    }
}

/// Error estimation type for gravity model coefficients.
///
/// Indicates what kind of uncertainty information is provided with the model.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "GravityModelErrors")]
#[derive(Clone)]
pub struct PyGravityModelErrors {
    pub(crate) errors: orbit_dynamics::GravityModelErrors,
}

#[pymethods]
impl PyGravityModelErrors {
    /// No error estimates provided.
    #[classattr]
    #[allow(non_snake_case)]
    fn No() -> Self {
        PyGravityModelErrors {
            errors: orbit_dynamics::GravityModelErrors::No,
        }
    }

    /// Calibrated error estimates.
    ///
    /// Empirically derived uncertainties based on data fit.
    #[classattr]
    #[allow(non_snake_case)]
    fn Calibrated() -> Self {
        PyGravityModelErrors {
            errors: orbit_dynamics::GravityModelErrors::Calibrated,
        }
    }

    /// Formal error estimates.
    ///
    /// Statistical uncertainties from the estimation process.
    #[classattr]
    #[allow(non_snake_case)]
    fn Formal() -> Self {
        PyGravityModelErrors {
            errors: orbit_dynamics::GravityModelErrors::Formal,
        }
    }

    /// Both calibrated and formal error estimates.
    #[classattr]
    #[allow(non_snake_case)]
    fn CalibratedAndFormal() -> Self {
        PyGravityModelErrors {
            errors: orbit_dynamics::GravityModelErrors::CalibratedAndFormal,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.errors)
    }

    fn __repr__(&self) -> String {
        format!("GravityModelErrors.{:?}", self.errors)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(std::mem::discriminant(&self.errors) == std::mem::discriminant(&other.errors)),
            CompareOp::Ne => Ok(std::mem::discriminant(&self.errors) != std::mem::discriminant(&other.errors)),
            _ => Err(exceptions::PyNotImplementedError::new_err("Comparison not supported")),
        }
    }
}

/// Normalization convention for spherical harmonic coefficients.
///
/// Different gravity models use different normalization schemes for coefficients.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "GravityModelNormalization")]
#[derive(Clone)]
pub struct PyGravityModelNormalization {
    pub(crate) normalization: orbit_dynamics::GravityModelNormalization,
}

#[pymethods]
impl PyGravityModelNormalization {
    /// Fully normalized coefficients.
    ///
    /// Modern standard using 4π normalization. Most common in recent gravity models.
    #[classattr]
    #[allow(non_snake_case)]
    fn FullyNormalized() -> Self {
        PyGravityModelNormalization {
            normalization: orbit_dynamics::GravityModelNormalization::FullyNormalized,
        }
    }

    /// Unnormalized coefficients.
    ///
    /// Legacy format without normalization. Rare in modern models.
    #[classattr]
    #[allow(non_snake_case)]
    fn Unnormalized() -> Self {
        PyGravityModelNormalization {
            normalization: orbit_dynamics::GravityModelNormalization::Unnormalized,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.normalization)
    }

    fn __repr__(&self) -> String {
        format!("GravityModelNormalization.{:?}", self.normalization)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(std::mem::discriminant(&self.normalization) == std::mem::discriminant(&other.normalization)),
            CompareOp::Ne => Ok(std::mem::discriminant(&self.normalization) != std::mem::discriminant(&other.normalization)),
            _ => Err(exceptions::PyNotImplementedError::new_err("Comparison not supported")),
        }
    }
}

// ============================================================================
// Spherical Harmonic Gravity Model Class
// ============================================================================

/// Spherical harmonic gravity model for high-fidelity gravitational acceleration computation.
///
/// This class represents a spherical harmonic expansion of Earth's gravitational potential,
/// allowing for accurate modeling of Earth's non-uniform gravity field.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "GravityModel")]
pub struct PyGravityModel {
    pub(crate) model: orbit_dynamics::GravityModel,
}

#[pymethods]
impl PyGravityModel {
    /// Tide system convention.
    #[getter]
    fn tide_system(&self) -> PyGravityModelTideSystem {
        PyGravityModelTideSystem {
            tide_system: self.model.tide_system,
        }
    }

    /// Maximum degree of spherical harmonic expansion.
    #[getter]
    fn n_max(&self) -> usize {
        self.model.n_max
    }

    /// Maximum order of spherical harmonic expansion.
    #[getter]
    fn m_max(&self) -> usize {
        self.model.m_max
    }

    /// Gravitational parameter. Units: (m³/s²)
    #[getter]
    fn gm(&self) -> f64 {
        self.model.gm
    }

    /// Reference radius. Units: (m)
    #[getter]
    fn radius(&self) -> f64 {
        self.model.radius
    }

    /// Name of the gravity model.
    #[getter]
    fn model_name(&self) -> String {
        self.model.model_name.clone()
    }

    /// Error estimation type.
    #[getter]
    fn model_errors(&self) -> PyGravityModelErrors {
        PyGravityModelErrors {
            errors: self.model.model_errors,
        }
    }

    /// Coefficient normalization convention.
    #[getter]
    fn normalization(&self) -> PyGravityModelNormalization {
        PyGravityModelNormalization {
            normalization: self.model.normalization,
        }
    }

    /// Load gravity model from a .gfc file.
    ///
    /// Args:
    ///     filepath (str): Path to the .gfc gravity model file
    ///
    /// Returns:
    ///     GravityModel: Loaded gravity model
    ///
    /// Raises:
    ///     Exception: If file cannot be loaded or parsed
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Load custom gravity model from file
    ///     model = bh.GravityModel.from_file("path/to/model.gfc")
    ///     print(f"Model: {model.model_name}, {model.n_max}x{model.m_max}")
    ///     ```
    #[classmethod]
    fn from_file(_cls: &Bound<'_, PyType>, filepath: &str) -> PyResult<Self> {
        let path = Path::new(filepath);
        let model = orbit_dynamics::GravityModel::from_file(path)
            .map_err(|e| exceptions::PyRuntimeError::new_err(format!("Failed to load gravity model: {}", e)))?;
        Ok(PyGravityModel { model })
    }

    /// Load one of the default packaged gravity models.
    ///
    /// Args:
    ///     model (DefaultGravityModel): Which default model to load
    ///
    /// Returns:
    ///     GravityModel: Loaded gravity model
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Load JGM3 70x70 model
    ///     model = bh.GravityModel.from_default(bh.DefaultGravityModel.JGM3)
    ///     print(f"Loaded {model.model_name}")
    ///
    ///     # Load EGM2008 360x360 model
    ///     model_hifi = bh.GravityModel.from_default(bh.DefaultGravityModel.EGM2008_360)
    ///     ```
    #[classmethod]
    fn from_default(_cls: &Bound<'_, PyType>, model: &PyDefaultGravityModel) -> PyResult<Self> {
        let grav_model = orbit_dynamics::GravityModel::from_default(model.model);
        Ok(PyGravityModel { model: grav_model })
    }

    /// Get spherical harmonic coefficients for a specific degree and order.
    ///
    /// Args:
    ///     n (int): Degree (0 <= n <= n_max)
    ///     m (int): Order (0 <= m <= min(n, m_max))
    ///
    /// Returns:
    ///     tuple: (C_nm, S_nm) coefficients
    ///
    /// Raises:
    ///     Exception: If n or m are out of bounds
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     model = bh.GravityModel.from_default(bh.DefaultGravityModel.JGM3)
    ///
    ///     # Get J2 coefficient (C20)
    ///     c20, s20 = model.get(2, 0)
    ///     print(f"J2 = {-c20}")
    ///     ```
    fn get(&self, n: usize, m: usize) -> PyResult<(f64, f64)> {
        self.model.get(n, m)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Failed to get coefficient: {}", e)))
    }

    /// Compute gravitational acceleration in body-fixed frame using spherical harmonics.
    ///
    /// Args:
    ///     r_body (np.ndarray): Position vector in body-fixed frame. Units: (m)
    ///     n_max (int): Maximum degree to evaluate (n_max <= model.n_max)
    ///     m_max (int): Maximum order to evaluate (m_max <= min(n_max, model.m_max))
    ///
    /// Returns:
    ///     np.ndarray: Acceleration in body-fixed frame. Units: (m/s²)
    ///
    /// Raises:
    ///     Exception: If n_max or m_max exceed model limits or if m_max > n_max
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     model = bh.GravityModel.from_default(bh.DefaultGravityModel.JGM3)
    ///     r_body = np.array([6525.919e3, 1710.416e3, 2508.886e3])
    ///
    ///     # Compute using 20x20 expansion
    ///     a_body = model.compute_spherical_harmonics(r_body, 20, 20)
    ///     ```
    fn compute_spherical_harmonics<'py>(
        &self,
        py: Python<'py>,
        r_body: PyReadonlyArray1<f64>,
        n_max: usize,
        m_max: usize,
    ) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
        let r = numpy_to_vector3!(r_body);
        let a = self.model.compute_spherical_harmonics(r, n_max, m_max)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Failed to compute spherical harmonics: {}", e)))?;
        Ok(vector_to_numpy!(py, a, 3, f64))
    }

    fn __repr__(&self) -> String {
        format!(
            "GravityModel(name='{}', n_max={}, m_max={})",
            self.model.model_name, self.model.n_max, self.model.m_max
        )
    }

    fn __str__(&self) -> String {
        format!(
            "GravityModel '{}' ({}x{}, GM={:.6e} m³/s², R={:.3e} m)",
            self.model.model_name, self.model.n_max, self.model.m_max, self.model.gm, self.model.radius
        )
    }
}

/// Compute acceleration due to spherical harmonic gravity model.
///
/// This function computes the gravitational acceleration on an object using a spherical
/// harmonic expansion of Earth's gravity field. It transforms the position to body-fixed
/// coordinates, evaluates the spherical harmonics, and transforms the acceleration back
/// to the inertial frame.
///
/// Accepts either a 3D position vector or a 6D state vector for r_eci.
///
/// Args:
///     r_eci (np.ndarray): Position (length 3) or state (length 6) in ECI frame. Units: (m)
///     R_i2b (np.ndarray): Rotation matrix from ECI to body-fixed frame (3x3)
///     gravity_model (GravityModel): Gravity model to use
///     n_max (int): Maximum degree to evaluate (n_max <= model.n_max)
///     m_max (int): Maximum order to evaluate (m_max <= min(n_max, model.m_max))
///
/// Returns:
///     np.ndarray: Acceleration in ECI frame. Units: (m/s²)
///
/// Raises:
///     Exception: If n_max or m_max exceed model limits or if m_max > n_max
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Initialize EOP for frame transformations
///     bh.initialize_eop()
///
///     # Load gravity model
///     model = bh.GravityModel.from_default(bh.DefaultGravityModel.JGM3)
///
///     # Create test position
///     r_eci = np.array([6525.919e3, 1710.416e3, 2508.886e3])
///
///     # Get rotation matrix (or use identity for simplified case)
///     R = np.eye(3)
///
///     # Compute acceleration
///     a_grav = bh.accel_gravity_spherical_harmonics(r_eci, R, model, 20, 20)
///     print(f"Acceleration: {a_grav} m/s²")
///     ```
#[pyfunction]
#[pyo3(name = "accel_gravity_spherical_harmonics")]
#[allow(non_snake_case)]
fn py_accel_gravity_spherical_harmonics<'py>(
    py: Python<'py>,
    r_eci: PyReadonlyArray1<f64>,
    R_i2b: PyReadonlyArray2<f64>,
    gravity_model: &PyGravityModel,
    n_max: usize,
    m_max: usize,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let len = r_eci.len();
    let rot = numpy_to_smatrix3!(R_i2b);
    let a = if len == 3 {
        let r = numpy_to_vector3!(r_eci);
        orbit_dynamics::accel_gravity_spherical_harmonics(r, rot, &gravity_model.model, n_max, m_max)
    } else if len == 6 {
        let x = numpy_to_vector6!(r_eci);
        orbit_dynamics::accel_gravity_spherical_harmonics(x, rot, &gravity_model.model, n_max, m_max)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_eci must be length 3 (position) or 6 (state)"
        ));
    };
    Ok(vector_to_numpy!(py, a, 3, f64))
}

// ============================================================================
// Atmospheric Density Models Python Bindings
// ============================================================================

/// Computes atmospheric density using the Harris-Priester model.
///
/// The Harris-Priester model accounts for diurnal density variations caused by solar heating.
/// Valid for altitudes between 100 km and 1000 km. Returns 0.0 outside this range.
///
/// Args:
///     r_tod (np.ndarray): Satellite position in true-of-date frame. Units: (m)
///     r_sun (np.ndarray): Sun position in true-of-date frame. Units: (m)
///
/// Returns:
///     float: Atmospheric density at the satellite position. Units: (kg/m³)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2024, 1, 1, bh.TimeSystem.UTC)
///     r_sat = np.array([bh.R_EARTH + 400e3, 0.0, 0.0])
///     r_sun = bh.sun_position(epc)
///
///     density = bh.density_harris_priester(r_sat, r_sun)
///     print(f"Density: {density:.2e} kg/m³")
///     ```
#[pyfunction]
#[pyo3(name = "density_harris_priester")]
fn py_density_harris_priester(
    r_tod: PyReadonlyArray1<f64>,
    r_sun: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let r = numpy_to_vector3!(r_tod);
    let r_s = numpy_to_vector3!(r_sun);
    Ok(orbit_dynamics::atmospheric_density_models::density_harris_priester(r, r_s))
}

// ============================================================================
// Drag Acceleration Python Bindings
// ============================================================================

/// Compute acceleration due to atmospheric drag.
///
/// Args:
///     x_object (np.ndarray): Satellite state vector in inertial frame. Units: [m; m/s]
///     density (float): Atmospheric density. Units: (kg/m³)
///     mass (float): Spacecraft mass. Units: (kg)
///     area (float): Wind-facing cross-sectional area. Units: (m²)
///     drag_coefficient (float): Coefficient of drag (dimensionless)
///     T (np.ndarray): Rotation matrix from inertial to true-of-date frame (3x3)
///
/// Returns:
///     np.ndarray: Acceleration due to drag. Units: (m/s²)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     x_object = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
///     density = 1.0e-12
///     a_drag = bh.accel_drag(x_object, density, 1000.0, 1.0, 2.3, np.eye(3))
///     ```
#[pyfunction]
#[pyo3(name = "accel_drag")]
#[allow(non_snake_case)]
fn py_accel_drag<'py>(
    py: Python<'py>,
    x_object: PyReadonlyArray1<f64>,
    density: f64,
    mass: f64,
    area: f64,
    drag_coefficient: f64,
    T: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let x_obj = numpy_to_vector6!(x_object);
    let t_mat = numpy_to_smatrix3!(T);
    let a = orbit_dynamics::accel_drag(x_obj, density, mass, area, drag_coefficient, t_mat);
    Ok(vector_to_numpy!(py, a, 3, f64))
}

// ============================================================================
// Solar Radiation Pressure Acceleration Python Bindings
// ============================================================================

/// Calculate acceleration due to solar radiation pressure.
///
/// Accepts either a 3D position vector or a 6D state vector for r_object.
///
/// Args:
///     r_object (np.ndarray): Position (length 3) or state (length 6) of the object. Units: (m)
///     r_sun (np.ndarray): Position vector of the sun. Units: (m)
///     mass (float): Mass of the object. Units: (kg)
///     cr (float): Coefficient of reflectivity (dimensionless)
///     area (float): Cross-sectional area of the object. Units: (m²)
///     p0 (float): Solar radiation pressure at 1 AU. Units: (N/m²)
///
/// Returns:
///     np.ndarray: Acceleration due to solar radiation pressure. Units: (m/s²)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2024, 2, 25, bh.TimeSystem.UTC)
///     r_object = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])
///     r_sun = bh.sun_position(epc)
///     a_srp = bh.accel_solar_radiation_pressure(r_object, r_sun, 1000.0, 1.8, 1.0, 4.56e-6)
///     ```
#[pyfunction]
#[pyo3(name = "accel_solar_radiation_pressure")]
fn py_accel_solar_radiation_pressure<'py>(
    py: Python<'py>,
    r_object: PyReadonlyArray1<f64>,
    r_sun: PyReadonlyArray1<f64>,
    mass: f64,
    cr: f64,
    area: f64,
    p0: f64,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let len = r_object.len();
    let r_s = numpy_to_vector3!(r_sun);
    let a = if len == 3 {
        let r_obj = numpy_to_vector3!(r_object);
        orbit_dynamics::accel_solar_radiation_pressure(r_obj, r_s, mass, cr, area, p0)
    } else if len == 6 {
        let x_obj = numpy_to_vector6!(r_object);
        orbit_dynamics::accel_solar_radiation_pressure(x_obj, r_s, mass, cr, area, p0)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_object must be length 3 (position) or 6 (state)"
        ));
    };
    Ok(vector_to_numpy!(py, a, 3, f64))
}

/// Calculate the fraction of the object illuminated by the sun using a conical (penumbral) shadow model.
///
/// The conical shadow model accounts for the finite size of both the Sun and Earth, modeling
/// the penumbra region where the satellite receives partial sunlight. This is more accurate
/// than the cylindrical model but computationally more expensive.
///
/// Accepts either a 3D position vector or a 6D state vector for r_object.
///
/// Args:
///     r_object (np.ndarray): Position (length 3) or state (length 6) of the object in ECI frame. Units: (m)
///     r_sun (np.ndarray): Position vector of the sun in ECI frame. Units: (m)
///
/// Returns:
///     float: Illumination fraction between 0.0 and 1.0. Values: 0.0 (full shadow/umbra),
///            0.0-1.0 (partial shadow/penumbra), 1.0 (full sunlight)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2024, 1, 1, bh.TimeSystem.UTC)
///     r_sat = np.array([bh.R_EARTH + 400e3, 0.0, 0.0])
///     r_sun = bh.sun_position(epc)
///
///     nu = bh.eclipse_conical(r_sat, r_sun)
///     print(f"Illumination fraction: {nu}")
///     ```
#[pyfunction]
#[pyo3(name = "eclipse_conical")]
fn py_eclipse_conical(
    r_object: PyReadonlyArray1<f64>,
    r_sun: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let len = r_object.len();
    let r_s = numpy_to_vector3!(r_sun);
    let nu = if len == 3 {
        let r_obj = numpy_to_vector3!(r_object);
        orbit_dynamics::eclipse_conical(r_obj, r_s)
    } else if len == 6 {
        let x_obj = numpy_to_vector6!(r_object);
        orbit_dynamics::eclipse_conical(x_obj, r_s)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_object must be length 3 (position) or 6 (state)"
        ));
    };
    Ok(nu)
}

/// Calculate the fraction of the object illuminated by the sun using a cylindrical shadow model.
///
/// The cylindrical shadow model is a simplified approach that assumes Earth casts a cylindrical
/// shadow parallel to the Sun-Earth line. This model is computationally efficient and provides
/// binary shadow determination (fully lit or fully shadowed, no penumbra).
///
/// Accepts either a 3D position vector or a 6D state vector for r_object.
///
/// Args:
///     r_object (np.ndarray): Position (length 3) or state (length 6) of the object in ECI frame. Units: (m)
///     r_sun (np.ndarray): Position vector of the sun in ECI frame. Units: (m)
///
/// Returns:
///     float: Illumination fraction, either 0.0 (full shadow) or 1.0 (full sunlight).
///            No partial illumination is returned by this model.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2024, 1, 1, bh.TimeSystem.UTC)
///     r_sat = np.array([bh.R_EARTH + 400e3, 0.0, 0.0])
///     r_sun = bh.sun_position(epc)
///
///     nu = bh.eclipse_cylindrical(r_sat, r_sun)
///     if nu == 0.0:
///         print("Satellite is in Earth's shadow")
///     else:
///         print("Satellite is in sunlight")
///     ```
#[pyfunction]
#[pyo3(name = "eclipse_cylindrical")]
fn py_eclipse_cylindrical(
    r_object: PyReadonlyArray1<f64>,
    r_sun: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let len = r_object.len();
    let r_s = numpy_to_vector3!(r_sun);
    let nu = if len == 3 {
        let r_obj = numpy_to_vector3!(r_object);
        orbit_dynamics::eclipse_cylindrical(r_obj, r_s)
    } else if len == 6 {
        let x_obj = numpy_to_vector6!(r_object);
        orbit_dynamics::eclipse_cylindrical(x_obj, r_s)
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "r_object must be length 3 (position) or 6 (state)"
        ));
    };
    Ok(nu)
}

// ============================================================================
// Relativity Acceleration Python Bindings
// ============================================================================

/// Calculate acceleration due to special and general relativity.
///
/// Args:
///     x_object (np.ndarray): State vector of the object in ECI frame. Units: [m; m/s]
///
/// Returns:
///     np.ndarray: Acceleration due to relativistic effects. Units: (m/s²)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     x_object = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7500.0, 0.0])
///     a_rel = bh.accel_relativity(x_object)
///     ```
#[pyfunction]
#[pyo3(name = "accel_relativity")]
fn py_accel_relativity<'py>(
    py: Python<'py>,
    x_object: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let x_obj = numpy_to_vector6!(x_object);
    let a = orbit_dynamics::accel_relativity(x_obj);
    Ok(vector_to_numpy!(py, a, 3, f64))
}
