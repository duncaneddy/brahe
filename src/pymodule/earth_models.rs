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
///     epoch (Epoch): Epoch of computation (used to lookup space weather data)
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

    crate::earth_models::density_nrlmsise00(&epc.obj, pos)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Compute atmospheric density using the NRLMSISE-00 model from geodetic coordinates.
///
/// This function computes atmospheric density using the NRLMSISE-00 empirical
/// model, automatically retrieving space weather data for the given epoch.
/// Takes geodetic coordinates directly.
///
/// Args:
///     epoch (Epoch): Epoch of computation (used to lookup space weather data)
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

    crate::earth_models::density_nrlmsise00_geod(&epc.obj, &geod_arr)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}