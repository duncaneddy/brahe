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
