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

/// Compute atmospheric density using the NRLMSISE-00 model from ECEF coordinates.
///
/// This function computes atmospheric density using the NRLMSISE-00 empirical
/// model, automatically retrieving space weather data for the given epoch.
/// The ECEF position is converted to geodetic coordinates internally.
///
/// Args:
///     epc (Epoch): Epoch of computation (used to lookup space weather data)
///     x_ecef (np.ndarray): Position in ECEF frame. Units: (m)
///
/// Returns:
///     float: Atmospheric density. Units: (kg/m³)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Initialize EOP and space weather data
///     bh.initialize_eop()
///     bh.initialize_sw()
///
///     # Define epoch and ECEF position (400 km altitude over equator)
///     epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)
///     x_ecef = np.array([bh.R_EARTH + 400e3, 0.0, 0.0])
///
///     # Compute density
///     density = bh.density_nrlmsise00(epc, x_ecef)
///     print(f"Density: {density:.2e} kg/m³")
///     ```
#[pyfunction]
#[pyo3(name = "density_nrlmsise00")]
fn py_density_nrlmsise00(epc: &PyEpoch, x_ecef: PyReadonlyArray1<f64>) -> PyResult<f64> {
    if x_ecef.len() != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "ECEF position must have 3 elements [x, y, z]",
        ));
    }
    let pos = numpy_to_vector3!(x_ecef);

    brahe::earth_models::density_nrlmsise00(&epc.obj, pos)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Compute atmospheric density using the NRLMSISE-00 model from geodetic coordinates.
///
/// This function computes atmospheric density using the NRLMSISE-00 empirical
/// model, automatically retrieving space weather data for the given epoch.
/// Takes geodetic coordinates directly.
///
/// Args:
///     epc (Epoch): Epoch of computation (used to lookup space weather data)
///     geod (np.ndarray): Geodetic position as [longitude, latitude, altitude] where
///         longitude and latitude are in degrees, and altitude is in meters
///
/// Returns:
///     float: Atmospheric density. Units: (kg/m³)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Initialize EOP and space weather data
///     bh.initialize_eop()
///     bh.initialize_sw()
///
///     # Define epoch and geodetic position
///     epc = bh.Epoch.from_date(2020, 6, 1, bh.TimeSystem.UTC)
///     geod = np.array([-74.0, 40.7, 400e3])  # NYC area, 400 km altitude
///
///     # Compute density
///     density = bh.density_nrlmsise00_geod(epc, geod)
///     print(f"Density: {density:.2e} kg/m³")
///     ```
#[pyfunction]
#[pyo3(name = "density_nrlmsise00_geod")]
fn py_density_nrlmsise00_geod(epc: &PyEpoch, geod: PyReadonlyArray1<f64>) -> PyResult<f64> {
    if geod.len() != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Geodetic position must have 3 elements [lon, lat, alt]",
        ));
    }
    let g = geod.as_array();
    let geod_arr = [g[0], g[1], g[2]];

    brahe::earth_models::density_nrlmsise00_geod(&epc.obj, &geod_arr)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

// ============================================================================
// Magnetic Field Model Python Bindings
// ============================================================================

/// Compute IGRF-14 magnetic field in the geodetic ENZ frame.
///
/// The geodetic ENZ frame has zenith perpendicular to the WGS84 ellipsoid surface.
///
/// Args:
///     epc (Epoch): Epoch of computation
///     x_geod (numpy.ndarray): Geodetic position [longitude, latitude, altitude_m].
///         Angle units controlled by angle_format. Altitude always in meters.
///     angle_format (AngleFormat): Whether longitude/latitude are in degrees or radians
///
/// Returns:
///     numpy.ndarray: Magnetic field [B_east, B_north, B_zenith] in nT
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     x_geod = np.array([0.0, 80.0, 0.0])  # lon=0, lat=80 deg, alt=0 m
///     b = bh.igrf_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)
///     print(f"B_east={b[0]:.1f}, B_north={b[1]:.1f}, B_zenith={b[2]:.1f} nT")
///     ```
#[pyfunction]
#[pyo3(name = "igrf_geodetic_enz")]
fn py_igrf_geodetic_enz<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_geod: PyReadonlyArray1<f64>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let geod = numpy_to_vector3!(x_geod);
    let result = brahe::earth_models::magnetic_field::igrf_geodetic_enz(
        &epc.obj,
        geod,
        angle_format.value,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(vector_to_numpy!(py, result, 3, f64))
}

/// Compute IGRF-14 magnetic field in the geocentric ENZ frame.
///
/// The geocentric ENZ frame has zenith along the geocentric radial direction.
///
/// Args:
///     epc (Epoch): Epoch of computation
///     x_geod (numpy.ndarray): Geodetic position [longitude, latitude, altitude_m].
///         Angle units controlled by angle_format. Altitude always in meters.
///     angle_format (AngleFormat): Whether longitude/latitude are in degrees or radians
///
/// Returns:
///     numpy.ndarray: Magnetic field [B_east, B_north, B_zenith] in nT
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     x_geod = np.array([0.0, 80.0, 0.0])
///     b = bh.igrf_geocentric_enz(epc, x_geod, bh.AngleFormat.DEGREES)
///     ```
#[pyfunction]
#[pyo3(name = "igrf_geocentric_enz")]
fn py_igrf_geocentric_enz<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_geod: PyReadonlyArray1<f64>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let geod = numpy_to_vector3!(x_geod);
    let result = brahe::earth_models::magnetic_field::igrf_geocentric_enz(
        &epc.obj,
        geod,
        angle_format.value,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(vector_to_numpy!(py, result, 3, f64))
}

/// Compute IGRF-14 magnetic field in the ECEF frame.
///
/// Args:
///     epc (Epoch): Epoch of computation
///     x_geod (numpy.ndarray): Geodetic position [longitude, latitude, altitude_m].
///         Angle units controlled by angle_format. Altitude always in meters.
///     angle_format (AngleFormat): Whether longitude/latitude are in degrees or radians
///
/// Returns:
///     numpy.ndarray: Magnetic field [B_x, B_y, B_z] in ECEF frame, in nT
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     x_geod = np.array([0.0, 80.0, 0.0])
///     b = bh.igrf_ecef(epc, x_geod, bh.AngleFormat.DEGREES)
///     ```
#[pyfunction]
#[pyo3(name = "igrf_ecef")]
fn py_igrf_ecef<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_geod: PyReadonlyArray1<f64>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let geod = numpy_to_vector3!(x_geod);
    let result = brahe::earth_models::magnetic_field::igrf_ecef(
        &epc.obj,
        geod,
        angle_format.value,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(vector_to_numpy!(py, result, 3, f64))
}

/// Compute WMMHR-2025 magnetic field in the geodetic ENZ frame.
///
/// The geodetic ENZ frame has zenith perpendicular to the WGS84 ellipsoid surface.
///
/// Args:
///     epc (Epoch): Epoch of computation
///     x_geod (numpy.ndarray): Geodetic position [longitude, latitude, altitude_m].
///         Angle units controlled by angle_format. Altitude always in meters.
///     angle_format (AngleFormat): Whether longitude/latitude are in degrees or radians
///     nmax (int, optional): Maximum spherical harmonic degree (1-133). Default: 133 (full resolution).
///
/// Returns:
///     numpy.ndarray: Magnetic field [B_east, B_north, B_zenith] in nT
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     x_geod = np.array([0.0, 80.0, 0.0])
///     b = bh.wmmhr_geodetic_enz(epc, x_geod, bh.AngleFormat.DEGREES)
///     print(f"B_east={b[0]:.1f}, B_north={b[1]:.1f}, B_zenith={b[2]:.1f} nT")
///     ```
#[pyfunction]
#[pyo3(name = "wmmhr_geodetic_enz", signature = (epc, x_geod, angle_format, nmax=None))]
fn py_wmmhr_geodetic_enz<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_geod: PyReadonlyArray1<f64>,
    angle_format: &PyAngleFormat,
    nmax: Option<usize>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let geod = numpy_to_vector3!(x_geod);
    let result = brahe::earth_models::magnetic_field::wmmhr_geodetic_enz(
        &epc.obj,
        geod,
        angle_format.value,
        nmax,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(vector_to_numpy!(py, result, 3, f64))
}

/// Compute WMMHR-2025 magnetic field in the geocentric ENZ frame.
///
/// The geocentric ENZ frame has zenith along the geocentric radial direction.
///
/// Args:
///     epc (Epoch): Epoch of computation
///     x_geod (numpy.ndarray): Geodetic position [longitude, latitude, altitude_m].
///         Angle units controlled by angle_format. Altitude always in meters.
///     angle_format (AngleFormat): Whether longitude/latitude are in degrees or radians
///     nmax (int, optional): Maximum spherical harmonic degree (1-133). Default: 133 (full resolution).
///
/// Returns:
///     numpy.ndarray: Magnetic field [B_east, B_north, B_zenith] in nT
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     x_geod = np.array([0.0, 80.0, 0.0])
///     b = bh.wmmhr_geocentric_enz(epc, x_geod, bh.AngleFormat.DEGREES)
///     ```
#[pyfunction]
#[pyo3(name = "wmmhr_geocentric_enz", signature = (epc, x_geod, angle_format, nmax=None))]
fn py_wmmhr_geocentric_enz<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_geod: PyReadonlyArray1<f64>,
    angle_format: &PyAngleFormat,
    nmax: Option<usize>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let geod = numpy_to_vector3!(x_geod);
    let result = brahe::earth_models::magnetic_field::wmmhr_geocentric_enz(
        &epc.obj,
        geod,
        angle_format.value,
        nmax,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(vector_to_numpy!(py, result, 3, f64))
}

/// Compute WMMHR-2025 magnetic field in the ECEF frame.
///
/// Args:
///     epc (Epoch): Epoch of computation
///     x_geod (numpy.ndarray): Geodetic position [longitude, latitude, altitude_m].
///         Angle units controlled by angle_format. Altitude always in meters.
///     angle_format (AngleFormat): Whether longitude/latitude are in degrees or radians
///     nmax (int, optional): Maximum spherical harmonic degree (1-133). Default: 133 (full resolution).
///
/// Returns:
///     numpy.ndarray: Magnetic field [B_x, B_y, B_z] in ECEF frame, in nT
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     x_geod = np.array([0.0, 80.0, 0.0])
///     b = bh.wmmhr_ecef(epc, x_geod, bh.AngleFormat.DEGREES)
///     ```
#[pyfunction]
#[pyo3(name = "wmmhr_ecef", signature = (epc, x_geod, angle_format, nmax=None))]
fn py_wmmhr_ecef<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_geod: PyReadonlyArray1<f64>,
    angle_format: &PyAngleFormat,
    nmax: Option<usize>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let geod = numpy_to_vector3!(x_geod);
    let result = brahe::earth_models::magnetic_field::wmmhr_ecef(
        &epc.obj,
        geod,
        angle_format.value,
        nmax,
    )
    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(vector_to_numpy!(py, result, 3, f64))
}