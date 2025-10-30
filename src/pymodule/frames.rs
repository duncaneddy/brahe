/// Computes the Bias-Precession-Nutation matrix transforming the `GCRS` to the
/// `CIRS` intermediate reference frame. This transformation corrects for the
/// bias, precession, and nutation of Celestial Intermediate Origin (`CIO`) with
/// respect to inertial space.
///
/// This formulation computes the Bias-Precession-Nutation correction matrix
/// according using a `CIO` based model using using the `IAU 2006`
/// precession and `IAU 2000A` nutation models.
///
/// The function will utilize the global Earth orientation and loaded data to
/// apply corrections to the Celestial Intermediate Pole (`CIP`) derived from
/// empirical observations.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of transformation matrix
///
/// Returns:
///     (numpy.ndarray): 3x3 rotation matrix transforming `GCRS` -> `CIRS`
///
/// References:
///     IAU SOFA Tools For Earth Attitude, Example 5.5
///     http://www.iausofa.org/2021_0512_C/sofa/sofa_pn_c.pdf
///     Software Version 18, 2021-04-18
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "bias_precession_nutation")]
unsafe fn py_bias_precession_nutation<'py>(py: Python<'py>, epc: &PyEpoch) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::bias_precession_nutation(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the Earth rotation matrix transforming the `CIRS` to the `TIRS`
/// intermediate reference frame. This transformation corrects for the Earth
/// rotation.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of transformation matrix
///
/// Returns:
///     (numpy.ndarray): 3x3 rotation matrix transforming `CIRS` -> `TIRS`
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "earth_rotation")]
unsafe fn py_earth_rotation<'py>(py: Python<'py>, epc: &PyEpoch) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::earth_rotation(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the Earth rotation matrix transforming the `TIRS` to the `ITRF` reference
/// frame.
///
/// The function will utilize the global Earth orientation and loaded data to
/// apply corrections to compute the polar motion correction based on empirical
/// observations of polar motion drift.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of transformation matrix
///
/// Returns:
///     (numpy.ndarray): 3x3 rotation matrix transforming `TIRS` -> `ITRF`
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "polar_motion")]
unsafe fn py_polar_motion<'py>(py: Python<'py>, epc: &PyEpoch) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::polar_motion(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the combined rotation matrix from the inertial to the Earth-fixed
/// reference frame. Applies corrections for bias, precession, nutation,
/// Earth-rotation, and polar motion.
///
/// The transformation is accomplished using the `IAU 2006/2000A`, `CIO`-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
///
/// The function will utilize the global Earth orientation and loaded data to
/// apply corrections for Celestial Intermidate Pole (`CIP`) and polar motion drift
/// derived from empirical observations.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of transformation matrix
///
/// Returns:
///     (numpy.ndarray): 3x3 rotation matrix transforming `GCRF` -> `ITRF`
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Create epoch
///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///
///     # Get rotation matrix
///     R = bh.rotation_eci_to_ecef(epc)
///     print(f"Rotation matrix shape: {R.shape}")
///     # Output: Rotation matrix shape: (3, 3)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_eci_to_ecef")]
unsafe fn py_rotation_eci_to_ecef<'py>(py: Python<'py>, epc: &PyEpoch) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::rotation_eci_to_ecef(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the combined rotation matrix from the Earth-fixed to the inertial
/// reference frame. Applies corrections for bias, precession, nutation,
/// Earth-rotation, and polar motion.
///
/// The transformation is accomplished using the `IAU 2006/2000A`, `CIO`-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
///
/// The function will utilize the global Earth orientation and loaded data to
/// apply corrections for Celestial Intermidate Pole (`CIP`) and polar motion drift
/// derived from empirical observations.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of transformation matrix
///
/// Returns:
///     (numpy.ndarray): 3x3 rotation matrix transforming `ITRF` -> `GCRF`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create epoch
///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///
///     # Get rotation matrix from ECEF to ECI
///     R = bh.rotation_ecef_to_eci(epc)
///     print(f"Rotation matrix shape: {R.shape}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_ecef_to_eci")]
unsafe fn py_rotation_ecef_to_eci<'py>(py: Python<'py>, epc: &PyEpoch) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::rotation_ecef_to_eci(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Transforms a position vector from the Earth Centered Inertial (`ECI`/`GCRF`) frame
/// to the Earth Centered Earth Fixed (`ECEF`/`ITRF`) frame.
///
/// Applies the full `IAU 2006/2000A` transformation including bias, precession,
/// nutation, Earth rotation, and polar motion corrections using global Earth
/// orientation parameters.
///
/// Args:
///     epc (Epoch): Epoch instant for the transformation
///     x (numpy.ndarray or list): Position vector in `ECI` frame (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Position vector in `ECEF` frame (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Create epoch
///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///
///     # Position vector in ECI (meters)
///     r_eci = np.array([7000000.0, 0.0, 0.0])
///
///     # Transform to ECEF
///     r_ecef = bh.position_eci_to_ecef(epc, r_eci)
///     print(f"ECEF position: {r_ecef}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x)")]
#[pyo3(name = "position_eci_to_ecef")]
fn py_position_eci_to_ecef<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_eci_to_ecef(epc.obj, pyany_to_svector::<3>(&x)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a position vector from the Earth Centered Earth Fixed (`ECEF`/`ITRF`)
/// frame to the Earth Centered Inertial (`ECI`/`GCRF`) frame.
///
/// Applies the full `IAU 2006/2000A` transformation including bias, precession,
/// nutation, Earth rotation, and polar motion corrections using global Earth
/// orientation parameters.
///
/// Args:
///     epc (Epoch): Epoch instant for the transformation
///     x (numpy.ndarray or list): Position vector in `ECEF` frame (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Position vector in `ECI` frame (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Create epoch
///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///
///     # Position in ECEF (ground station)
///     r_ecef = np.array([4000000.0, 3000000.0, 4000000.0])
///
///     # Transform to ECI
///     r_eci = bh.position_ecef_to_eci(epc, r_ecef)
///     print(f"ECI position: {r_eci}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x)")]
#[pyo3(name = "position_ecef_to_eci")]
fn py_position_ecef_to_eci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_ecef_to_eci(epc.obj, pyany_to_svector::<3>(&x)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a state vector (position and velocity) from the Earth Centered
/// Inertial (`ECI`/`GCRF`) frame to the Earth Centered Earth Fixed (`ECEF`/`ITRF`) frame.
///
/// Applies the full `IAU 2006/2000A` transformation including bias, precession,
/// nutation, Earth rotation, and polar motion corrections using global Earth
/// orientation parameters. The velocity transformation accounts for the Earth's
/// rotation rate.
///
/// Args:
///     epc (Epoch): Epoch instant for the transformation
///     x_eci (numpy.ndarray or list): State vector in `ECI` frame `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: State vector in `ECEF` frame `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Create epoch
///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///
///     # State vector in ECI [x, y, z, vx, vy, vz] (meters, m/s)
///     state_eci = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
///
///     # Transform to ECEF
///     state_ecef = bh.state_eci_to_ecef(epc, state_eci)
///     print(f"ECEF state: {state_ecef}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_eci)")]
#[pyo3(name = "state_eci_to_ecef")]
fn py_state_eci_to_ecef<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_eci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_eci_to_ecef(epc.obj, pyany_to_svector::<6>(&x_eci)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Transforms a state vector (position and velocity) from the Earth Centered
/// Earth Fixed (`ECEF`/`ITRF`) frame to the Earth Centered Inertial (`ECI`/`GCRF`) frame.
///
/// Applies the full `IAU 2006/2000A` transformation including bias, precession,
/// nutation, Earth rotation, and polar motion corrections using global Earth
/// orientation parameters. The velocity transformation accounts for the Earth's
/// rotation rate.
///
/// Args:
///     epc (Epoch): Epoch instant for the transformation
///     x_ecef (numpy.ndarray or list): State vector in `ECEF` frame `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: State vector in `ECI` frame `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Create epoch
///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///
///     # State vector in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
///     state_ecef = np.array([4000000.0, 3000000.0, 4000000.0, 100.0, -50.0, 200.0])
///
///     # Transform to ECI
///     state_eci = bh.state_ecef_to_eci(epc, state_ecef)
///     print(f"ECI state: {state_eci}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_ecef)")]
#[pyo3(name = "state_ecef_to_eci")]
fn py_state_ecef_to_eci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_ecef: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_ecef_to_eci(epc.obj, pyany_to_svector::<6>(&x_ecef)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}