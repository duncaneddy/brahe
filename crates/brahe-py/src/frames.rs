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

/// Computes the combined rotation matrix from GCRF (Geocentric Celestial Reference Frame)
/// to ITRF (International Terrestrial Reference Frame). Applies corrections for bias,
/// precession, nutation, Earth-rotation, and polar motion.
///
/// The transformation is accomplished using the `IAU 2006/2000A`, `CIO`-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
///
/// The function will utilize the global Earth orientation and loaded data to
/// apply corrections for Celestial Intermediate Pole (`CIP`) and polar motion drift
/// derived from empirical observations.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of transformation matrix
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming `GCRF` -> `ITRF`
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Create epoch
///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///
///     # Get rotation matrix from GCRF to ITRF
///     R = bh.rotation_gcrf_to_itrf(epc)
///     print(f"Rotation matrix shape: {R.shape}")
///     # Output: Rotation matrix shape: (3, 3)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_gcrf_to_itrf")]
unsafe fn py_rotation_gcrf_to_itrf<'py>(py: Python<'py>, epc: &PyEpoch) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::rotation_gcrf_to_itrf(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the combined rotation matrix from the inertial to the Earth-fixed
/// reference frame. Applies corrections for bias, precession, nutation,
/// Earth-rotation, and polar motion.
///
/// This function is an alias for rotation_gcrf_to_itrf. `ECI` refers to the
/// `GCRF` (Geocentric Celestial Reference Frame) implementation, and `ECEF` refers
/// to the `ITRF` (International Terrestrial Reference Frame) implementation.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of transformation matrix
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming `ECI` (`GCRF`) -> `ECEF` (`ITRF`)
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

/// Computes the combined rotation matrix from ITRF (International Terrestrial Reference Frame)
/// to GCRF (Geocentric Celestial Reference Frame). Applies corrections for bias,
/// precession, nutation, Earth-rotation, and polar motion.
///
/// The transformation is accomplished using the `IAU 2006/2000A`, `CIO`-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
///
/// The function will utilize the global Earth orientation and loaded data to
/// apply corrections for Celestial Intermediate Pole (`CIP`) and polar motion drift
/// derived from empirical observations.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of transformation matrix
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming `ITRF` -> `GCRF`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create epoch
///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///
///     # Get rotation matrix from ITRF to GCRF
///     R = bh.rotation_itrf_to_gcrf(epc)
///     print(f"Rotation matrix shape: {R.shape}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_itrf_to_gcrf")]
unsafe fn py_rotation_itrf_to_gcrf<'py>(py: Python<'py>, epc: &PyEpoch) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::rotation_itrf_to_gcrf(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the combined rotation matrix from the Earth-fixed to the inertial
/// reference frame. Applies corrections for bias, precession, nutation,
/// Earth-rotation, and polar motion.
///
/// This function is an alias for rotation_itrf_to_gcrf. `ECEF` refers to the
/// `ITRF` (International Terrestrial Reference Frame) implementation, and `ECI` refers
/// to the `GCRF` (Geocentric Celestial Reference Frame) implementation.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of transformation matrix
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming `ECEF` (`ITRF`) -> `ECI` (`GCRF`)
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

/// Transforms a position vector from GCRF (Geocentric Celestial Reference Frame)
/// to ITRF (International Terrestrial Reference Frame).
///
/// Applies the full `IAU 2006/2000A` transformation including bias, precession,
/// nutation, Earth rotation, and polar motion corrections using global Earth
/// orientation parameters.
///
/// Args:
///     epc (Epoch): Epoch instant for the transformation
///     x (numpy.ndarray or list): Position vector in `GCRF` frame (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Position vector in `ITRF` frame (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Create epoch
///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///
///     # Position vector in GCRF (meters)
///     r_gcrf = np.array([7000000.0, 0.0, 0.0])
///
///     # Transform to ITRF
///     r_itrf = bh.position_gcrf_to_itrf(epc, r_gcrf)
///     print(f"ITRF position: {r_itrf}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x)")]
#[pyo3(name = "position_gcrf_to_itrf")]
fn py_position_gcrf_to_itrf<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_gcrf_to_itrf(epc.obj, pyany_to_svector::<3>(&x)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a position vector from the Earth Centered Inertial (`ECI`/`GCRF`) frame
/// to the Earth Centered Earth Fixed (`ECEF`/`ITRF`) frame.
///
/// This function is an alias for position_gcrf_to_itrf. Applies the full
/// `IAU 2006/2000A` transformation including bias, precession, nutation, Earth
/// rotation, and polar motion corrections using global Earth orientation parameters.
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

/// Transforms a position vector from ITRF (International Terrestrial Reference Frame)
/// to GCRF (Geocentric Celestial Reference Frame).
///
/// Applies the full `IAU 2006/2000A` transformation including bias, precession,
/// nutation, Earth rotation, and polar motion corrections using global Earth
/// orientation parameters.
///
/// Args:
///     epc (Epoch): Epoch instant for the transformation
///     x (numpy.ndarray or list): Position vector in `ITRF` frame (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Position vector in `GCRF` frame (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Create epoch
///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///
///     # Position in ITRF (ground station)
///     r_itrf = np.array([4000000.0, 3000000.0, 4000000.0])
///
///     # Transform to GCRF
///     r_gcrf = bh.position_itrf_to_gcrf(epc, r_itrf)
///     print(f"GCRF position: {r_gcrf}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x)")]
#[pyo3(name = "position_itrf_to_gcrf")]
fn py_position_itrf_to_gcrf<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_itrf_to_gcrf(epc.obj, pyany_to_svector::<3>(&x)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a position vector from the Earth Centered Earth Fixed (`ECEF`/`ITRF`)
/// frame to the Earth Centered Inertial (`ECI`/`GCRF`) frame.
///
/// This function is an alias for position_itrf_to_gcrf. Applies the full
/// `IAU 2006/2000A` transformation including bias, precession, nutation, Earth
/// rotation, and polar motion corrections using global Earth orientation parameters.
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

/// Transforms a state vector (position and velocity) from GCRF (Geocentric Celestial
/// Reference Frame) to ITRF (International Terrestrial Reference Frame).
///
/// Applies the full `IAU 2006/2000A` transformation including bias, precession,
/// nutation, Earth rotation, and polar motion corrections using global Earth
/// orientation parameters. The velocity transformation accounts for the Earth's
/// rotation rate.
///
/// Args:
///     epc (Epoch): Epoch instant for the transformation
///     x_gcrf (numpy.ndarray or list): State vector in `GCRF` frame `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: State vector in `ITRF` frame `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Create epoch
///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///
///     # State vector in GCRF [x, y, z, vx, vy, vz] (meters, m/s)
///     state_gcrf = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
///
///     # Transform to ITRF
///     state_itrf = bh.state_gcrf_to_itrf(epc, state_gcrf)
///     print(f"ITRF state: {state_itrf}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_gcrf)")]
#[pyo3(name = "state_gcrf_to_itrf")]
fn py_state_gcrf_to_itrf<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_gcrf: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_gcrf_to_itrf(epc.obj, pyany_to_svector::<6>(&x_gcrf)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Transforms a state vector (position and velocity) from the Earth Centered
/// Inertial (`ECI`/`GCRF`) frame to the Earth Centered Earth Fixed (`ECEF`/`ITRF`) frame.
///
/// This function is an alias for state_gcrf_to_itrf. Applies the full
/// `IAU 2006/2000A` transformation including bias, precession, nutation, Earth
/// rotation, and polar motion corrections using global Earth orientation parameters.
/// The velocity transformation accounts for the Earth's rotation rate.
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

/// Transforms a state vector (position and velocity) from ITRF (International Terrestrial
/// Reference Frame) to GCRF (Geocentric Celestial Reference Frame).
///
/// Applies the full `IAU 2006/2000A` transformation including bias, precession,
/// nutation, Earth rotation, and polar motion corrections using global Earth
/// orientation parameters. The velocity transformation accounts for the Earth's
/// rotation rate.
///
/// Args:
///     epc (Epoch): Epoch instant for the transformation
///     x_itrf (numpy.ndarray or list): State vector in `ITRF` frame `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: State vector in `GCRF` frame `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Create epoch
///     epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///
///     # State vector in ITRF [x, y, z, vx, vy, vz] (meters, m/s)
///     state_itrf = np.array([4000000.0, 3000000.0, 4000000.0, 100.0, -50.0, 200.0])
///
///     # Transform to GCRF
///     state_gcrf = bh.state_itrf_to_gcrf(epc, state_itrf)
///     print(f"GCRF state: {state_gcrf}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_itrf)")]
#[pyo3(name = "state_itrf_to_gcrf")]
fn py_state_itrf_to_gcrf<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_itrf: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_itrf_to_gcrf(epc.obj, pyany_to_svector::<6>(&x_itrf)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Transforms a state vector (position and velocity) from the Earth Centered
/// Earth Fixed (`ECEF`/`ITRF`) frame to the Earth Centered Inertial (`ECI`/`GCRF`) frame.
///
/// This function is an alias for state_itrf_to_gcrf. Applies the full
/// `IAU 2006/2000A` transformation including bias, precession, nutation, Earth
/// rotation, and polar motion corrections using global Earth orientation parameters.
/// The velocity transformation accounts for the Earth's rotation rate.
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

/// Computes the frame bias matrix transforming GCRF (Geocentric Celestial Reference Frame)
/// to EME2000 (Earth Mean Equator and Equinox of J2000.0).
///
/// The bias matrix accounts for the small offset between the GCRF and the J2000.0 mean
/// equator and equinox due to the difference in their definitions. This is a constant
/// transformation that does not vary with time.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming `GCRF` -> `EME2000`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Get the bias matrix
///     B = bh.bias_eme2000()
///     print(f"Bias matrix shape: {B.shape}")
///     # Output: Bias matrix shape: (3, 3)
///     ```
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "bias_eme2000")]
unsafe fn py_bias_eme2000<'py>(py: Python<'py>) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::bias_eme2000();
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the rotation matrix from GCRF (Geocentric Celestial Reference Frame)
/// to EME2000 (Earth Mean Equator and Equinox of J2000.0).
///
/// This transformation applies the frame bias correction to account for the difference
/// between GCRF (ICRS-aligned) and EME2000 (J2000.0 mean equator/equinox). The
/// transformation is constant and does not depend on time.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming `GCRF` -> `EME2000`
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Get rotation matrix
///     R = bh.rotation_gcrf_to_eme2000()
///     print(f"Rotation matrix shape: {R.shape}")
///     # Output: Rotation matrix shape: (3, 3)
///     ```
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "rotation_gcrf_to_eme2000")]
unsafe fn py_rotation_gcrf_to_eme2000<'py>(py: Python<'py>) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::rotation_gcrf_to_eme2000();
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the rotation matrix from EME2000 (Earth Mean Equator and Equinox of J2000.0)
/// to GCRF (Geocentric Celestial Reference Frame).
///
/// This transformation applies the inverse frame bias correction to account for the
/// difference between EME2000 (J2000.0 mean equator/equinox) and GCRF (ICRS-aligned).
/// The transformation is constant and does not depend on time.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming `EME2000` -> `GCRF`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Get rotation matrix
///     R = bh.rotation_eme2000_to_gcrf()
///     print(f"Rotation matrix shape: {R.shape}")
///     # Output: Rotation matrix shape: (3, 3)
///     ```
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "rotation_eme2000_to_gcrf")]
unsafe fn py_rotation_eme2000_to_gcrf<'py>(py: Python<'py>) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::rotation_eme2000_to_gcrf();
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Transforms a position vector from GCRF (Geocentric Celestial Reference Frame)
/// to EME2000 (Earth Mean Equator and Equinox of J2000.0).
///
/// Applies the frame bias correction to account for the small offset between GCRF
/// and the J2000.0 mean equator and equinox. This is a constant transformation
/// that does not vary with time.
///
/// Args:
///     x (numpy.ndarray or list): Position vector in `GCRF` frame (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Position vector in `EME2000` frame (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Position vector in GCRF (meters)
///     r_gcrf = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])
///
///     # Transform to EME2000
///     r_eme2000 = bh.position_gcrf_to_eme2000(r_gcrf)
///     print(f"EME2000 position: {r_eme2000}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(x)")]
#[pyo3(name = "position_gcrf_to_eme2000")]
fn py_position_gcrf_to_eme2000<'py>(
    py: Python<'py>,
    x: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_gcrf_to_eme2000(pyany_to_svector::<3>(&x)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a position vector from EME2000 (Earth Mean Equator and Equinox of J2000.0)
/// to GCRF (Geocentric Celestial Reference Frame).
///
/// Applies the inverse frame bias correction to account for the small offset between
/// the J2000.0 mean equator and equinox and GCRF. This is a constant transformation
/// that does not vary with time.
///
/// Args:
///     x (numpy.ndarray or list): Position vector in `EME2000` frame (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Position vector in `GCRF` frame (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Position vector in EME2000 (meters)
///     r_eme2000 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])
///
///     # Transform to GCRF
///     r_gcrf = bh.position_eme2000_to_gcrf(r_eme2000)
///     print(f"GCRF position: {r_gcrf}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(x)")]
#[pyo3(name = "position_eme2000_to_gcrf")]
fn py_position_eme2000_to_gcrf<'py>(
    py: Python<'py>,
    x: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_eme2000_to_gcrf(pyany_to_svector::<3>(&x)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a state vector (position and velocity) from GCRF (Geocentric Celestial
/// Reference Frame) to EME2000 (Earth Mean Equator and Equinox of J2000.0).
///
/// Applies the frame bias correction to both position and velocity. Because the
/// transformation does not vary with time, the velocity is directly rotated without
/// additional correction terms.
///
/// Args:
///     x_gcrf (numpy.ndarray or list): State vector in `GCRF` frame `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: State vector in `EME2000` frame `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # State vector in GCRF [x, y, z, vx, vy, vz] (meters, m/s)
///     state_gcrf = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
///
///     # Transform to EME2000
///     state_eme2000 = bh.state_gcrf_to_eme2000(state_gcrf)
///     print(f"EME2000 state: {state_eme2000}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(x_gcrf)")]
#[pyo3(name = "state_gcrf_to_eme2000")]
fn py_state_gcrf_to_eme2000<'py>(
    py: Python<'py>,
    x_gcrf: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_gcrf_to_eme2000(pyany_to_svector::<6>(&x_gcrf)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Transforms a state vector (position and velocity) from EME2000 (Earth Mean Equator
/// and Equinox of J2000.0) to GCRF (Geocentric Celestial Reference Frame).
///
/// Applies the inverse frame bias correction to both position and velocity. Because
/// the transformation does not vary with time, the velocity is directly rotated without
/// additional correction terms.
///
/// Args:
///     x_eme2000 (numpy.ndarray or list): State vector in `EME2000` frame `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: State vector in `GCRF` frame `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # State vector in EME2000 [x, y, z, vx, vy, vz] (meters, m/s)
///     state_eme2000 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
///
///     # Transform to GCRF
///     state_gcrf = bh.state_eme2000_to_gcrf(state_eme2000)
///     print(f"GCRF state: {state_gcrf}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(x_eme2000)")]
#[pyo3(name = "state_eme2000_to_gcrf")]
fn py_state_eme2000_to_gcrf<'py>(
    py: Python<'py>,
    x_eme2000: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_eme2000_to_gcrf(pyany_to_svector::<6>(&x_eme2000)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

// ============================================================================
// IAU/WGCCRE Body Rotation Model
// ============================================================================

/// Computes the rotation matrix from the ICRF to the IAU/WGCCRE body-fixed
/// frame of `naif_id` at `epc`.
///
/// Args:
///     naif_id (int): NAIF ID of the body (see `iau_rotation_model_ids` for the supported set)
///     epc (Epoch): Epoch instant for computation of the transformation matrix
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming ICRF -> body-fixed
///
/// Raises:
///     RuntimeError: If no IAU/WGCCRE rotation model is embedded for `naif_id`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     r = bh.rotation_icrf_to_body_fixed_iau(499, epc)  # Mars
///     ```
#[pyfunction]
#[pyo3(text_signature = "(naif_id, epc)")]
#[pyo3(name = "rotation_icrf_to_body_fixed_iau")]
unsafe fn py_rotation_icrf_to_body_fixed_iau<'py>(
    py: Python<'py>,
    naif_id: i32,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let mat = frames::rotation_icrf_to_body_fixed_iau(naif_id, epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(matrix_to_numpy!(py, mat, 3, 3, f64))
}

/// Sorted list of NAIF IDs with an embedded IAU/WGCCRE rotation model.
///
/// Returns:
///     list[int]: Sorted NAIF IDs supported by `rotation_icrf_to_body_fixed_iau`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     ids = bh.iau_rotation_model_ids()
///     assert 499 in ids  # Mars
///     ```
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "iau_rotation_model_ids")]
fn py_iau_rotation_model_ids() -> Vec<i32> {
    frames::iau_rotation_model_ids()
}

// ============================================================================
// Mars Reference Frames (MCI, MCMF)
// ============================================================================

/// Computes the rotation matrix from Mars-Centered Inertial (MCI) to
/// Mars-Centered Mars-Fixed (MCMF), using the IAU/WGCCRE pole and
/// prime-meridian model for Mars.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation matrix
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming MCI -> MCMF
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     r = bh.rotation_mci_to_mcmf(epc)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_mci_to_mcmf")]
unsafe fn py_rotation_mci_to_mcmf<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::rotation_mci_to_mcmf(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the rotation matrix from Mars-Centered Mars-Fixed (MCMF) to
/// Mars-Centered Inertial (MCI). Inverse of `rotation_mci_to_mcmf`.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation matrix
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming MCMF -> MCI
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     r = bh.rotation_mcmf_to_mci(epc)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_mcmf_to_mci")]
unsafe fn py_rotation_mcmf_to_mci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::rotation_mcmf_to_mci(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Transforms a Cartesian Mars-inertial (MCI) position into the equivalent
/// Cartesian Mars-fixed (MCMF) position.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_mci (numpy.ndarray or list): Cartesian MCI position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian MCMF position (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_mcmf = bh.position_mci_to_mcmf(epc, [bh.R_MARS + 400e3, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_mci)")]
#[pyo3(name = "position_mci_to_mcmf")]
fn py_position_mci_to_mcmf<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_mci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_mci_to_mcmf(epc.obj, pyany_to_svector::<3>(&x_mci)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian Mars-fixed (MCMF) position into the equivalent
/// Cartesian Mars-inertial (MCI) position.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_mcmf (numpy.ndarray or list): Cartesian MCMF position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian MCI position (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_mci = bh.position_mcmf_to_mci(epc, [bh.R_MARS, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_mcmf)")]
#[pyo3(name = "position_mcmf_to_mci")]
fn py_position_mcmf_to_mci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_mcmf: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_mcmf_to_mci(epc.obj, pyany_to_svector::<3>(&x_mcmf)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian Mars-inertial (MCI) state (position and velocity)
/// into the equivalent Cartesian Mars-fixed (MCMF) state.
///
/// The velocity transformation accounts for the transport term induced by
/// Mars' rotation.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_mci (numpy.ndarray or list): Cartesian MCI state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian MCMF state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_mci = [bh.R_MARS + 400e3, 0.0, 0.0, 0.0, 3.4e3, 0.0]
///     x_mcmf = bh.state_mci_to_mcmf(epc, x_mci)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_mci)")]
#[pyo3(name = "state_mci_to_mcmf")]
fn py_state_mci_to_mcmf<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_mci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_mci_to_mcmf(epc.obj, pyany_to_svector::<6>(&x_mci)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Transforms a Cartesian Mars-fixed (MCMF) state (position and velocity)
/// into the equivalent Cartesian Mars-inertial (MCI) state. Inverse of
/// `state_mci_to_mcmf`.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_mcmf (numpy.ndarray or list): Cartesian MCMF state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian MCI state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_mci = [bh.R_MARS + 400e3, 0.0, 0.0, 0.0, 3.4e3, 0.0]
///     x_mcmf = bh.state_mci_to_mcmf(epc, x_mci)
///     x_mci2 = bh.state_mcmf_to_mci(epc, x_mcmf)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_mcmf)")]
#[pyo3(name = "state_mcmf_to_mci")]
fn py_state_mcmf_to_mci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_mcmf: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_mcmf_to_mci(epc.obj, pyany_to_svector::<6>(&x_mcmf)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Transforms a Cartesian Earth-inertial (ECI) position into the equivalent
/// Cartesian Earth-Moon-barycenter inertial (EMBI) position.
///
/// Both frames share the ICRF orientation, so this is a pure translation by
/// the Earth's position relative to the Earth-Moon barycenter (NAIF ID 3).
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_eci (numpy.ndarray or list): Cartesian ECI position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian EMBI position (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_emb = bh.position_eci_to_emb(epc, [7e6, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_eci)")]
#[pyo3(name = "position_eci_to_emb")]
fn py_position_eci_to_emb<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_eci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_eci_to_emb(epc.obj, pyany_to_svector::<3>(&x_eci)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian Earth-Moon-barycenter inertial (EMBI) position
/// into the equivalent Cartesian Earth-inertial (ECI) position.
///
/// Both frames share the ICRF orientation, so this is a pure translation by
/// the Earth's position relative to the Earth-Moon barycenter (NAIF ID 3).
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_emb (numpy.ndarray or list): Cartesian EMBI position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian ECI position (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_eci = bh.position_emb_to_eci(epc, [7e6, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_emb)")]
#[pyo3(name = "position_emb_to_eci")]
fn py_position_emb_to_eci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_emb: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_emb_to_eci(epc.obj, pyany_to_svector::<3>(&x_emb)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian Earth-inertial (ECI) state (position and velocity)
/// into the equivalent Cartesian Earth-Moon-barycenter inertial (EMBI)
/// state.
///
/// Both frames share the ICRF orientation, so this is a pure translation by
/// the Earth's state relative to the Earth-Moon barycenter (NAIF ID 3).
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_eci (numpy.ndarray or list): Cartesian ECI state (m; m/s), shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian EMBI state (m; m/s), shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_emb = bh.state_eci_to_emb(epc, [7e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_eci)")]
#[pyo3(name = "state_eci_to_emb")]
fn py_state_eci_to_emb<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_eci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_eci_to_emb(epc.obj, pyany_to_svector::<6>(&x_eci)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Transforms a Cartesian Earth-Moon-barycenter inertial (EMBI) state
/// (position and velocity) into the equivalent Cartesian Earth-inertial
/// (ECI) state.
///
/// Both frames share the ICRF orientation, so this is a pure translation by
/// the Earth's state relative to the Earth-Moon barycenter (NAIF ID 3).
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_emb (numpy.ndarray or list): Cartesian EMBI state (m; m/s), shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian ECI state (m; m/s), shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_eci = bh.state_emb_to_eci(epc, [7e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_emb)")]
#[pyo3(name = "state_emb_to_eci")]
fn py_state_emb_to_eci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_emb: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_emb_to_eci(epc.obj, pyany_to_svector::<6>(&x_emb)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}


/// Transforms a Cartesian Earth-inertial (ECI) position into the equivalent
/// Cartesian Mars-inertial (MCI) position.
///
/// The MCI origin is the Mars body center (NAIF ID 499); the `mar099s`
/// satellite ephemeris kernel is auto-loaded for the body-center leg. Auto-initializes
/// the default `de440s` ephemeris if no SPK kernel is loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_eci (numpy.ndarray or list): Cartesian ECI position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian MCI position (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_mci = bh.position_eci_to_mci(epc, [1e7, 2e7, 3e7])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_eci)")]
#[pyo3(name = "position_eci_to_mci")]
fn py_position_eci_to_mci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_eci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_eci_to_mci(epc.obj, pyany_to_svector::<3>(&x_eci)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian Mars-inertial (MCI) position into the equivalent
/// Cartesian Earth-inertial (ECI) position.
///
/// The MCI origin is the Mars body center (NAIF ID 499); the `mar099s`
/// satellite ephemeris kernel is auto-loaded for the body-center leg. Auto-initializes
/// the default `de440s` ephemeris if no SPK kernel is loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_mci (numpy.ndarray or list): Cartesian MCI position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian ECI position (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_eci = bh.position_mci_to_eci(epc, [1e7, 2e7, 3e7])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_mci)")]
#[pyo3(name = "position_mci_to_eci")]
fn py_position_mci_to_eci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_mci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_mci_to_eci(epc.obj, pyany_to_svector::<3>(&x_mci)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian Earth-inertial (ECI) state (position and velocity)
/// into the equivalent Cartesian Mars-inertial (MCI) state.
///
/// The MCI origin is the Mars body center (NAIF ID 499); the `mar099s`
/// satellite ephemeris kernel is auto-loaded for the body-center leg. Auto-initializes
/// the default `de440s` ephemeris if no SPK kernel is loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_eci (numpy.ndarray or list): Cartesian ECI state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian MCI state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_mci = bh.state_eci_to_mci(epc, [1e7, 2e7, 3e7, 1.0, 2.0, 3.0])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_eci)")]
#[pyo3(name = "state_eci_to_mci")]
fn py_state_eci_to_mci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_eci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_eci_to_mci(epc.obj, pyany_to_svector::<6>(&x_eci)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Transforms a Cartesian Mars-inertial (MCI) state (position and velocity)
/// into the equivalent Cartesian Earth-inertial (ECI) state.
///
/// The MCI origin is the Mars body center (NAIF ID 499); the `mar099s`
/// satellite ephemeris kernel is auto-loaded for the body-center leg. Auto-initializes
/// the default `de440s` ephemeris if no SPK kernel is loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_mci (numpy.ndarray or list): Cartesian MCI state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian ECI state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_eci = bh.state_mci_to_eci(epc, [1e7, 2e7, 3e7, 1.0, 2.0, 3.0])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_mci)")]
#[pyo3(name = "state_mci_to_eci")]
fn py_state_mci_to_eci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_mci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_mci_to_eci(epc.obj, pyany_to_svector::<6>(&x_mci)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

// ============================================================================
// Lunar Reference Frames (LCI, LFPA, LFME)
// ============================================================================

/// Computes the rotation matrix from Lunar-Centered Inertial (LCI) to
/// Lunar-Fixed Principal Axis (LFPA), using the DE440 lunar principal-axis
/// binary PCK (`moon_pa_de440`).
///
/// Auto-loads the `moon_pa_de440` PCK (downloading it to `~/.cache/brahe/naif`
/// if needed).
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation matrix
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming LCI -> LFPA
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     r = bh.rotation_lci_to_lfpa(epc)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_lci_to_lfpa")]
unsafe fn py_rotation_lci_to_lfpa<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::rotation_lci_to_lfpa(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the rotation matrix from Lunar-Fixed Principal Axis (LFPA) to
/// Lunar-Centered Inertial (LCI). Inverse of `rotation_lci_to_lfpa`.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation matrix
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming LFPA -> LCI
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     r = bh.rotation_lfpa_to_lci(epc)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_lfpa_to_lci")]
unsafe fn py_rotation_lfpa_to_lci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::rotation_lfpa_to_lci(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the constant rotation matrix from Lunar-Fixed Mean Earth/polar-axis
/// (LFME) to Lunar-Fixed Principal Axis (LFPA).
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming LFME -> LFPA
///
/// Example:
///     ```python
///     import brahe as bh
///
///     r = bh.rotation_lfme_to_lfpa()
///     ```
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "rotation_lfme_to_lfpa")]
unsafe fn py_rotation_lfme_to_lfpa<'py>(py: Python<'py>) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::rotation_lfme_to_lfpa();
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the constant rotation matrix from Lunar-Fixed Principal Axis
/// (LFPA) to Lunar-Fixed Mean Earth/polar-axis (LFME). Inverse of
/// `rotation_lfme_to_lfpa`.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming LFPA -> LFME
///
/// Example:
///     ```python
///     import brahe as bh
///
///     r = bh.rotation_lfpa_to_lfme()
///     ```
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "rotation_lfpa_to_lfme")]
unsafe fn py_rotation_lfpa_to_lfme<'py>(py: Python<'py>) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::rotation_lfpa_to_lfme();
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the rotation matrix from Lunar-Centered Inertial (LCI) to
/// Lunar-Fixed Mean Earth/polar-axis (LFME).
///
/// Auto-loads the `moon_pa_de440` PCK (downloading it to `~/.cache/brahe/naif`
/// if needed).
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation matrix
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming LCI -> LFME
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     r = bh.rotation_lci_to_lfme(epc)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_lci_to_lfme")]
unsafe fn py_rotation_lci_to_lfme<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::rotation_lci_to_lfme(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the rotation matrix from Lunar-Fixed Mean Earth/polar-axis (LFME)
/// to Lunar-Centered Inertial (LCI). Inverse of `rotation_lci_to_lfme`.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation matrix
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming LFME -> LCI
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     r = bh.rotation_lfme_to_lci(epc)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_lfme_to_lci")]
unsafe fn py_rotation_lfme_to_lci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = frames::rotation_lfme_to_lci(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Transforms a Cartesian Lunar-inertial (LCI) position into the equivalent
/// Cartesian Lunar-Fixed Principal Axis (LFPA) position.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_lci (numpy.ndarray or list): Cartesian LCI position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian LFPA position (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_lfpa = bh.position_lci_to_lfpa(epc, [bh.R_MOON + 100e3, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_lci)")]
#[pyo3(name = "position_lci_to_lfpa")]
fn py_position_lci_to_lfpa<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_lci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_lci_to_lfpa(epc.obj, pyany_to_svector::<3>(&x_lci)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian Lunar-Fixed Principal Axis (LFPA) position into the
/// equivalent Cartesian Lunar-inertial (LCI) position.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_lfpa (numpy.ndarray or list): Cartesian LFPA position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian LCI position (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_lci = bh.position_lfpa_to_lci(epc, [bh.R_MOON, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_lfpa)")]
#[pyo3(name = "position_lfpa_to_lci")]
fn py_position_lfpa_to_lci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_lfpa: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_lfpa_to_lci(epc.obj, pyany_to_svector::<3>(&x_lfpa)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian Lunar-inertial (LCI) position into the equivalent
/// Cartesian Lunar-Fixed Mean Earth/polar-axis (LFME) position.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_lci (numpy.ndarray or list): Cartesian LCI position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian LFME position (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_lfme = bh.position_lci_to_lfme(epc, [bh.R_MOON + 100e3, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_lci)")]
#[pyo3(name = "position_lci_to_lfme")]
fn py_position_lci_to_lfme<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_lci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_lci_to_lfme(epc.obj, pyany_to_svector::<3>(&x_lci)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian Lunar-Fixed Mean Earth/polar-axis (LFME) position
/// into the equivalent Cartesian Lunar-inertial (LCI) position.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_lfme (numpy.ndarray or list): Cartesian LFME position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian LCI position (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_lci = bh.position_lfme_to_lci(epc, [bh.R_MOON, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_lfme)")]
#[pyo3(name = "position_lfme_to_lci")]
fn py_position_lfme_to_lci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_lfme: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_lfme_to_lci(epc.obj, pyany_to_svector::<3>(&x_lfme)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian Lunar-inertial (LCI) state (position and velocity)
/// into the equivalent Cartesian Lunar-Fixed Principal Axis (LFPA) state.
///
/// The velocity transformation accounts for the transport term induced by
/// the Moon's rotation.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_lci (numpy.ndarray or list): Cartesian LCI state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian LFPA state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_lci = [bh.R_MOON + 100e3, 0.0, 0.0, 0.0, 1.6e3, 0.0]
///     x_lfpa = bh.state_lci_to_lfpa(epc, x_lci)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_lci)")]
#[pyo3(name = "state_lci_to_lfpa")]
fn py_state_lci_to_lfpa<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_lci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_lci_to_lfpa(epc.obj, pyany_to_svector::<6>(&x_lci)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Transforms a Cartesian Lunar-Fixed Principal Axis (LFPA) state (position
/// and velocity) into the equivalent Cartesian Lunar-inertial (LCI) state.
/// Inverse of `state_lci_to_lfpa`.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_lfpa (numpy.ndarray or list): Cartesian LFPA state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian LCI state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_lci = [bh.R_MOON + 100e3, 0.0, 0.0, 0.0, 1.6e3, 0.0]
///     x_lfpa = bh.state_lci_to_lfpa(epc, x_lci)
///     x_lci2 = bh.state_lfpa_to_lci(epc, x_lfpa)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_lfpa)")]
#[pyo3(name = "state_lfpa_to_lci")]
fn py_state_lfpa_to_lci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_lfpa: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_lfpa_to_lci(epc.obj, pyany_to_svector::<6>(&x_lfpa)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Transforms a Cartesian Lunar-inertial (LCI) state (position and velocity)
/// into the equivalent Cartesian Lunar-Fixed Mean Earth/polar-axis (LFME)
/// state.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_lci (numpy.ndarray or list): Cartesian LCI state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian LFME state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_lci = [bh.R_MOON + 100e3, 0.0, 0.0, 0.0, 1.6e3, 0.0]
///     x_lfme = bh.state_lci_to_lfme(epc, x_lci)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_lci)")]
#[pyo3(name = "state_lci_to_lfme")]
fn py_state_lci_to_lfme<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_lci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_lci_to_lfme(epc.obj, pyany_to_svector::<6>(&x_lci)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Transforms a Cartesian Lunar-Fixed Mean Earth/polar-axis (LFME) state
/// (position and velocity) into the equivalent Cartesian Lunar-inertial (LCI)
/// state. Inverse of `state_lci_to_lfme`.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_lfme (numpy.ndarray or list): Cartesian LFME state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian LCI state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_lci = [bh.R_MOON + 100e3, 0.0, 0.0, 0.0, 1.6e3, 0.0]
///     x_lfme = bh.state_lci_to_lfme(epc, x_lci)
///     x_lci2 = bh.state_lfme_to_lci(epc, x_lfme)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_lfme)")]
#[pyo3(name = "state_lfme_to_lci")]
fn py_state_lfme_to_lci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_lfme: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_lfme_to_lci(epc.obj, pyany_to_svector::<6>(&x_lfme)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Transforms a Cartesian Earth-inertial (ECI) position into the equivalent
/// Cartesian Lunar-inertial (LCI) position.
///
/// The LCI origin is the Moon's body center (NAIF ID 301). Auto-initializes
/// the default `de440s` ephemeris if no SPK kernel is loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_eci (numpy.ndarray or list): Cartesian ECI position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian LCI position (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_lci = bh.position_eci_to_lci(epc, [1e7, 2e7, 3e7])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_eci)")]
#[pyo3(name = "position_eci_to_lci")]
fn py_position_eci_to_lci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_eci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_eci_to_lci(epc.obj, pyany_to_svector::<3>(&x_eci)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian Lunar-inertial (LCI) position into the equivalent
/// Cartesian Earth-inertial (ECI) position.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_lci (numpy.ndarray or list): Cartesian LCI position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian ECI position (m), shape `(3,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_eci = bh.position_lci_to_eci(epc, [1e7, 2e7, 3e7])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_lci)")]
#[pyo3(name = "position_lci_to_eci")]
fn py_position_lci_to_eci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_lci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_lci_to_eci(epc.obj, pyany_to_svector::<3>(&x_lci)?);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian Earth-inertial (ECI) state (position and velocity)
/// into the equivalent Cartesian Lunar-inertial (LCI) state.
///
/// The LCI origin is the Moon's body center (NAIF ID 301). Auto-initializes
/// the default `de440s` ephemeris if no SPK kernel is loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_eci (numpy.ndarray or list): Cartesian ECI state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian LCI state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_lci = bh.state_eci_to_lci(epc, [1e7, 2e7, 3e7, 1.0, 2.0, 3.0])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_eci)")]
#[pyo3(name = "state_eci_to_lci")]
fn py_state_eci_to_lci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_eci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_eci_to_lci(epc.obj, pyany_to_svector::<6>(&x_eci)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Transforms a Cartesian Lunar-inertial (LCI) state (position and velocity)
/// into the equivalent Cartesian Earth-inertial (ECI) state.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_lci (numpy.ndarray or list): Cartesian LCI state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian ECI state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_eci = bh.state_lci_to_eci(epc, [1e7, 2e7, 3e7, 1.0, 2.0, 3.0])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_lci)")]
#[pyo3(name = "state_lci_to_eci")]
fn py_state_lci_to_eci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_lci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_lci_to_eci(epc.obj, pyany_to_svector::<6>(&x_lci)?);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

// ============================================================================
// Synodic Reference Frames (EMR, SER, GSE)
// ============================================================================

/// Computes the rotation matrix from GCRF axes to Earth-Moon Rotating (EMR)
/// frame axes (NASA TP-20220014814 §4.6.2): x̂ Earth→Moon, ẑ along the
/// Moon's orbital angular momentum relative to Earth.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming GCRF -> EMR axes
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     r = bh.rotation_gcrf_to_emr(epc)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_gcrf_to_emr")]
unsafe fn py_rotation_gcrf_to_emr<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let mat = frames::rotation_gcrf_to_emr(epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(matrix_to_numpy!(py, mat, 3, 3, f64))
}

/// Computes the rotation matrix from Earth-Moon Rotating (EMR) frame axes to
/// GCRF axes. Inverse of `rotation_gcrf_to_emr`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming EMR -> GCRF axes
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     r = bh.rotation_emr_to_gcrf(epc)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_emr_to_gcrf")]
unsafe fn py_rotation_emr_to_gcrf<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let mat = frames::rotation_emr_to_gcrf(epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(matrix_to_numpy!(py, mat, 3, 3, f64))
}

/// Transforms a Cartesian GCRF position into the equivalent Earth-Moon
/// Rotating (EMR) frame position. The EMR origin is the Earth-Moon
/// barycenter.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_gcrf (numpy.ndarray or list): Cartesian GCRF position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian EMR position (m), shape `(3,)`
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_emr = bh.position_gcrf_to_emr(epc, [1e8, -2e8, 5e7])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_gcrf)")]
#[pyo3(name = "position_gcrf_to_emr")]
fn py_position_gcrf_to_emr<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_gcrf: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_gcrf_to_emr(epc.obj, pyany_to_svector::<3>(&x_gcrf)?)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian Earth-Moon Rotating (EMR) frame position into the
/// equivalent Cartesian GCRF position. Inverse of `position_gcrf_to_emr`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_emr (numpy.ndarray or list): Cartesian EMR position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian GCRF position (m), shape `(3,)`
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_gcrf = bh.position_emr_to_gcrf(epc, [3.8e8, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_emr)")]
#[pyo3(name = "position_emr_to_gcrf")]
fn py_position_emr_to_gcrf<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_emr: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_emr_to_gcrf(epc.obj, pyany_to_svector::<3>(&x_emr)?)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian GCRF state (position and velocity) into the
/// equivalent Earth-Moon Rotating (EMR) frame state. The EMR origin is the
/// Earth-Moon barycenter; the velocity transform uses the exact rotation-
/// matrix time derivative (including dẑ/dt) evaluated from SPK
/// acceleration.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_gcrf (numpy.ndarray or list): Cartesian GCRF state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian EMR state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_emr = bh.state_gcrf_to_emr(epc, [1e8, -2e8, 5e7, 1.0e3, -2.0e3, 0.5e3])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_gcrf)")]
#[pyo3(name = "state_gcrf_to_emr")]
fn py_state_gcrf_to_emr<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_gcrf: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_gcrf_to_emr(epc.obj, pyany_to_svector::<6>(&x_gcrf)?)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Transforms a Cartesian Earth-Moon Rotating (EMR) frame state (position
/// and velocity) into the equivalent Cartesian GCRF state. Inverse of
/// `state_gcrf_to_emr`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_emr (numpy.ndarray or list): Cartesian EMR state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian GCRF state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_emr = [3.8e8, 0.0, 0.0, 0.0, 1.0e3, 0.0]
///     x_gcrf = bh.state_emr_to_gcrf(epc, x_emr)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_emr)")]
#[pyo3(name = "state_emr_to_gcrf")]
fn py_state_emr_to_gcrf<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_emr: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_emr_to_gcrf(epc.obj, pyany_to_svector::<6>(&x_emr)?)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Computes the rotation matrix from GCRF axes to Sun-Earth Rotating (SER)
/// frame axes (NASA TP-20220014814 §4.6.4): x̂ Sun→Earth, ẑ along Earth's
/// orbital angular momentum relative to the Sun.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming GCRF -> SER axes
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     r = bh.rotation_gcrf_to_ser(epc)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_gcrf_to_ser")]
unsafe fn py_rotation_gcrf_to_ser<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let mat = frames::rotation_gcrf_to_ser(epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(matrix_to_numpy!(py, mat, 3, 3, f64))
}

/// Computes the rotation matrix from Sun-Earth Rotating (SER) frame axes to
/// GCRF axes. Inverse of `rotation_gcrf_to_ser`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming SER -> GCRF axes
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     r = bh.rotation_ser_to_gcrf(epc)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_ser_to_gcrf")]
unsafe fn py_rotation_ser_to_gcrf<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let mat = frames::rotation_ser_to_gcrf(epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(matrix_to_numpy!(py, mat, 3, 3, f64))
}

/// Transforms a Cartesian GCRF position into the equivalent Sun-Earth
/// Rotating (SER) frame position. The SER origin is the (computed)
/// Sun-Earth barycenter.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_gcrf (numpy.ndarray or list): Cartesian GCRF position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian SER position (m), shape `(3,)`
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_ser = bh.position_gcrf_to_ser(epc, [1e8, -2e8, 5e7])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_gcrf)")]
#[pyo3(name = "position_gcrf_to_ser")]
fn py_position_gcrf_to_ser<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_gcrf: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_gcrf_to_ser(epc.obj, pyany_to_svector::<3>(&x_gcrf)?)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian Sun-Earth Rotating (SER) frame position into the
/// equivalent Cartesian GCRF position. Inverse of `position_gcrf_to_ser`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_ser (numpy.ndarray or list): Cartesian SER position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian GCRF position (m), shape `(3,)`
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_gcrf = bh.position_ser_to_gcrf(epc, [1.5e11, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_ser)")]
#[pyo3(name = "position_ser_to_gcrf")]
fn py_position_ser_to_gcrf<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_ser: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_ser_to_gcrf(epc.obj, pyany_to_svector::<3>(&x_ser)?)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian GCRF state (position and velocity) into the
/// equivalent Sun-Earth Rotating (SER) frame state. The SER origin is the
/// (computed) Sun-Earth barycenter; the velocity transform uses the exact
/// rotation-matrix time derivative (including dẑ/dt) evaluated from SPK
/// acceleration.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_gcrf (numpy.ndarray or list): Cartesian GCRF state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian SER state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_ser = bh.state_gcrf_to_ser(epc, [1e8, -2e8, 5e7, 1.0e3, -2.0e3, 0.5e3])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_gcrf)")]
#[pyo3(name = "state_gcrf_to_ser")]
fn py_state_gcrf_to_ser<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_gcrf: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_gcrf_to_ser(epc.obj, pyany_to_svector::<6>(&x_gcrf)?)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Transforms a Cartesian Sun-Earth Rotating (SER) frame state (position
/// and velocity) into the equivalent Cartesian GCRF state. Inverse of
/// `state_gcrf_to_ser`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_ser (numpy.ndarray or list): Cartesian SER state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian GCRF state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_ser = [1.5e11, 0.0, 0.0, 0.0, 1.0e3, 0.0]
///     x_gcrf = bh.state_ser_to_gcrf(epc, x_ser)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_ser)")]
#[pyo3(name = "state_ser_to_gcrf")]
fn py_state_ser_to_gcrf<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_ser: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_ser_to_gcrf(epc.obj, pyany_to_svector::<6>(&x_ser)?)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Computes the rotation matrix from GCRF axes to Geocentric Solar Ecliptic
/// (GSE) frame axes (NASA TP-20220014814 §2.5.4/§4.6.5): x̂ Earth→Sun, ẑ
/// normal to the instantaneous ecliptic plane.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming GCRF -> GSE axes
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     r = bh.rotation_gcrf_to_gse(epc)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_gcrf_to_gse")]
unsafe fn py_rotation_gcrf_to_gse<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let mat = frames::rotation_gcrf_to_gse(epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(matrix_to_numpy!(py, mat, 3, 3, f64))
}

/// Computes the rotation matrix from Geocentric Solar Ecliptic (GSE) frame
/// axes to GCRF axes. Inverse of `rotation_gcrf_to_gse`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming GSE -> GCRF axes
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     r = bh.rotation_gse_to_gcrf(epc)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_gse_to_gcrf")]
unsafe fn py_rotation_gse_to_gcrf<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let mat = frames::rotation_gse_to_gcrf(epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(matrix_to_numpy!(py, mat, 3, 3, f64))
}

/// Transforms a Cartesian GCRF position into the equivalent Geocentric
/// Solar Ecliptic (GSE) frame position. GSE is Earth-centered.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_gcrf (numpy.ndarray or list): Cartesian GCRF position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian GSE position (m), shape `(3,)`
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_gse = bh.position_gcrf_to_gse(epc, [1e8, -2e8, 5e7])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_gcrf)")]
#[pyo3(name = "position_gcrf_to_gse")]
fn py_position_gcrf_to_gse<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_gcrf: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_gcrf_to_gse(epc.obj, pyany_to_svector::<3>(&x_gcrf)?)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian Geocentric Solar Ecliptic (GSE) frame position
/// into the equivalent Cartesian GCRF position. Inverse of
/// `position_gcrf_to_gse`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_gse (numpy.ndarray or list): Cartesian GSE position (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian GCRF position (m), shape `(3,)`
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_gcrf = bh.position_gse_to_gcrf(epc, [1.5e11, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_gse)")]
#[pyo3(name = "position_gse_to_gcrf")]
fn py_position_gse_to_gcrf<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_gse: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_gse_to_gcrf(epc.obj, pyany_to_svector::<3>(&x_gse)?)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian GCRF state (position and velocity) into the
/// equivalent Geocentric Solar Ecliptic (GSE) frame state. GSE is
/// Earth-centered; the velocity transform uses the exact rotation-matrix
/// time derivative (including dẑ/dt) evaluated from SPK acceleration.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_gcrf (numpy.ndarray or list): Cartesian GCRF state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian GSE state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_gse = bh.state_gcrf_to_gse(epc, [1e8, -2e8, 5e7, 1.0e3, -2.0e3, 0.5e3])
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_gcrf)")]
#[pyo3(name = "state_gcrf_to_gse")]
fn py_state_gcrf_to_gse<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_gcrf: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_gcrf_to_gse(epc.obj, pyany_to_svector::<6>(&x_gcrf)?)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, vec, 6, f64))
}

/// Transforms a Cartesian Geocentric Solar Ecliptic (GSE) frame state
/// (position and velocity) into the equivalent Cartesian GCRF state.
/// Inverse of `state_gcrf_to_gse`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch): Epoch instant for computation of the transformation
///     x_gse (numpy.ndarray or list): Cartesian GSE state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian GCRF state `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Raises:
///     RuntimeError: If the SPK lookup fails at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_gse = [1.5e11, 0.0, 0.0, 0.0, 1.0e3, 0.0]
///     x_gcrf = bh.state_gse_to_gcrf(epc, x_gse)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(epc, x_gse)")]
#[pyo3(name = "state_gse_to_gcrf")]
fn py_state_gse_to_gcrf<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_gse: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_gse_to_gcrf(epc.obj, pyany_to_svector::<6>(&x_gse)?)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, vec, 6, f64))
}

// ============================================================================
// Reference Frame Router
// ============================================================================

/// Origin choice for a generic synodic frame (`ReferenceFrame.Synodic`):
/// `Primary`, `Secondary`, or the GM-weighted two-body `Barycenter`.
#[pyclass(module = "brahe._brahe", eq, from_py_object)]
#[pyo3(name = "SynodicOrigin")]
#[derive(Clone, PartialEq)]
pub struct PySynodicOrigin {
    pub(crate) origin: frames::SynodicOrigin,
}

#[pymethods]
impl PySynodicOrigin {
    /// Centered on the primary body.
    #[classattr]
    #[allow(non_snake_case)]
    fn Primary() -> Self {
        PySynodicOrigin { origin: frames::SynodicOrigin::Primary }
    }

    /// Centered on the secondary body.
    #[classattr]
    #[allow(non_snake_case)]
    fn Secondary() -> Self {
        PySynodicOrigin { origin: frames::SynodicOrigin::Secondary }
    }

    /// Centered on the GM-weighted two-body barycenter.
    #[classattr]
    #[allow(non_snake_case)]
    fn Barycenter() -> Self {
        PySynodicOrigin { origin: frames::SynodicOrigin::Barycenter }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.origin)
    }

    fn __repr__(&self) -> String {
        format!("SynodicOrigin.{:?}", self.origin)
    }
}

/// A reference frame supported by the centralized frame router
/// (`rotation_frame_to_frame`, `position_frame_to_frame`, `state_frame_to_frame`).
///
/// Includes every named frame defined elsewhere in this module (`GCRF`,
/// `ITRF`, `EME2000`, the lunar frames `LFPA`/`LFME`, and the Mars frame
/// `MCMF`, plus the corresponding inertial frames `LCI`/`MCI`), the
/// Earth-Moon and Solar System barycentric inertial frames (`EMBI`, `SSBI`),
/// and three generic constructors for bodies without a dedicated named
/// frame: `BodyCenteredICRF(naif_id)`, `BodyFixedIAU(naif_id)`, and
/// `BodyFixedPCK(center, frame_id)`.
///
/// Frame centers (NAIF ID): GCRF/ITRF/EME2000 -> Earth (399); LCI/LFPA/LFME
/// -> Moon (301); MCI/MCMF -> Mars (499); EMBI -> 3; SSBI ->
/// 0; `BodyCenteredICRF(id)`/`BodyFixedIAU(id)` -> `id`; `BodyFixedPCK` ->
/// its `center`.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     r = bh.rotation_frame_to_frame(bh.ReferenceFrame.MCI, bh.ReferenceFrame.MCMF, epc)
///     ```
#[pyclass(module = "brahe._brahe", eq, from_py_object)]
#[pyo3(name = "ReferenceFrame")]
#[derive(Clone, PartialEq)]
pub struct PyReferenceFrame {
    pub(crate) frame: frames::ReferenceFrame,
}

#[pymethods]
impl PyReferenceFrame {
    /// Geocentric Celestial Reference Frame (ICRF-aligned, Earth-centered).
    #[classattr]
    #[allow(non_snake_case)]
    fn GCRF() -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::GCRF }
    }

    /// International Terrestrial Reference Frame (Earth-fixed).
    #[classattr]
    #[allow(non_snake_case)]
    fn ITRF() -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::ITRF }
    }

    /// Alias for `ReferenceFrame.GCRF`: the Earth-Centered Inertial (ECI)
    /// frame is realized as GCRF.
    #[classattr]
    #[allow(non_snake_case)]
    fn ECI() -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::ECI }
    }

    /// Alias for `ReferenceFrame.ITRF`: the Earth-Centered Earth-Fixed (ECEF)
    /// frame is realized as ITRF.
    #[classattr]
    #[allow(non_snake_case)]
    fn ECEF() -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::ECEF }
    }

    /// Earth Mean Equator and Equinox of J2000.0.
    #[classattr]
    #[allow(non_snake_case)]
    fn EME2000() -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::EME2000 }
    }

    /// Lunar-Centered Inertial (ICRF-aligned, Moon-centered).
    #[classattr]
    #[allow(non_snake_case)]
    fn LCI() -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::LCI }
    }

    /// Lunar-Fixed Principal Axis (DE440 `MOON_PA_DE440`).
    #[classattr]
    #[allow(non_snake_case)]
    fn LFPA() -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::LFPA }
    }

    /// Lunar-Fixed Mean Earth/polar-axis.
    #[classattr]
    #[allow(non_snake_case)]
    fn LFME() -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::LFME }
    }

    /// Mars-Centered Inertial (ICRF-aligned, Mars-centered).
    #[classattr]
    #[allow(non_snake_case)]
    fn MCI() -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::MCI }
    }

    /// Mars-Centered Mars-Fixed (IAU/WGCCRE Mars rotation model).
    #[classattr]
    #[allow(non_snake_case)]
    fn MCMF() -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::MCMF }
    }

    /// Earth-Moon Barycentric Inertial (ICRF-aligned).
    #[classattr]
    #[allow(non_snake_case)]
    fn EMBI() -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::EMBI }
    }

    /// Solar System Barycentric Inertial (ICRF-aligned).
    #[classattr]
    #[allow(non_snake_case)]
    fn SSBI() -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::SSBI }
    }

    /// Earth-Moon Rotating frame (NASA TP-20220014814): x̂ Earth→Moon, ẑ
    /// along the Moon's orbital angular momentum relative to Earth;
    /// centered on the Earth-Moon barycenter.
    #[classattr]
    #[allow(non_snake_case)]
    fn EMR() -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::EMR }
    }

    /// Sun-Earth Rotating frame (NASA TP-20220014814): x̂ Sun→Earth, ẑ
    /// along Earth's orbital angular momentum relative to the Sun;
    /// centered on the (computed) Sun-Earth barycenter.
    #[classattr]
    #[allow(non_snake_case)]
    fn SER() -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::SER }
    }

    /// Geocentric Solar Ecliptic frame (NASA TP-20220014814): x̂ Earth→Sun,
    /// ẑ normal to the instantaneous ecliptic plane; Earth-centered.
    #[classattr]
    #[allow(non_snake_case)]
    fn GSE() -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::GSE }
    }

    /// ICRF-aligned axes centered on the given NAIF ID.
    ///
    /// Args:
    ///     naif_id (int): NAIF ID of the frame's center
    ///
    /// Returns:
    ///     ReferenceFrame: ICRF-aligned frame centered on `naif_id`
    #[staticmethod]
    #[allow(non_snake_case)]
    fn BodyCenteredICRF(naif_id: i32) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::BodyCenteredICRF(naif_id) }
    }

    /// IAU/WGCCRE body-fixed frame of the given NAIF ID, centered on that
    /// same NAIF ID.
    ///
    /// Args:
    ///     naif_id (int): NAIF ID of the body (see `iau_rotation_model_ids` for the supported set)
    ///
    /// Returns:
    ///     ReferenceFrame: IAU/WGCCRE body-fixed frame of `naif_id`
    #[staticmethod]
    #[allow(non_snake_case)]
    fn BodyFixedIAU(naif_id: i32) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::BodyFixedIAU(naif_id) }
    }

    /// Body-fixed frame evaluated from a loaded binary PCK's `frame_id`,
    /// centered on `center`.
    ///
    /// Args:
    ///     center (int): NAIF ID of the frame's center
    ///     frame_id (int): NAIF binary PCK frame class ID (e.g. 31008 for `MOON_PA_DE440`)
    ///
    /// Returns:
    ///     ReferenceFrame: Body-fixed frame for `frame_id`, centered on `center`
    #[staticmethod]
    #[allow(non_snake_case)]
    fn BodyFixedPCK(center: i32, frame_id: i32) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::BodyFixedPCK { center, frame_id } }
    }

    /// Body-fixed frame evaluated from a user-registered rotation callback
    /// (see `register_custom_frame`), centered on `center`.
    ///
    /// For a body without a catalogued NAIF ID, self-assign a unique negative
    /// `center` (mirroring NAIF's convention for non-catalogued objects):
    /// rotation-only queries never consult the center, and translations will
    /// raise unless an ephemeris covering that ID is loaded.
    ///
    /// Args:
    ///     center (int): NAIF ID of the frame's center (may be self-assigned negative)
    ///     key (int): Registry key the frame's callbacks were registered under
    ///
    /// Returns:
    ///     ReferenceFrame: Custom body-fixed frame for `key`, centered on `center`
    #[staticmethod]
    #[allow(non_snake_case)]
    fn BodyFixedCustom(center: i32, key: u32) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::BodyFixedCustom { center, key } }
    }

    /// Generic two-body synodic (rotating) frame: x̂ from `primary` toward
    /// `secondary`, ẑ along the secondary's orbital angular momentum
    /// relative to the primary, centered per `origin`. The named frames
    /// are specific configurations: `EMR == Synodic(Barycenter, 399, 301)`,
    /// `SER == Synodic(Barycenter, 10, 399)`, `GSE == Synodic(Primary, 399, 10)`.
    ///
    /// Args:
    ///     origin (SynodicOrigin): Origin choice (primary, secondary, or barycenter)
    ///     primary (int): NAIF ID of the primary body, in 0..=999
    ///     secondary (int): NAIF ID of the secondary body, in 0..=999
    ///
    /// Returns:
    ///     ReferenceFrame: Generic synodic frame for the pair
    ///
    /// Raises:
    ///     ValueError: If either NAIF ID is outside 0..=999
    #[staticmethod]
    #[allow(non_snake_case)]
    fn Synodic(origin: PySynodicOrigin, primary: i32, secondary: i32) -> PyResult<Self> {
        if !(0..=999).contains(&primary) || !(0..=999).contains(&secondary) {
            return Err(exceptions::PyValueError::new_err(
                "Synodic primary/secondary NAIF IDs must be in 0..=999",
            ));
        }
        Ok(PyReferenceFrame {
            frame: frames::ReferenceFrame::Synodic { origin: origin.origin, primary, secondary },
        })
    }

    /// Origin choice of a synodic frame (`Synodic`, `EMR`, `SER`, `GSE`).
    ///
    /// Returns:
    ///     Optional[SynodicOrigin]: Origin choice, or `None` for non-synodic frames
    #[getter]
    fn synodic_origin(&self) -> Option<PySynodicOrigin> {
        let origin = match self.frame {
            frames::ReferenceFrame::Synodic { origin, .. } => origin,
            frames::ReferenceFrame::EMR | frames::ReferenceFrame::SER => {
                frames::SynodicOrigin::Barycenter
            }
            frames::ReferenceFrame::GSE => frames::SynodicOrigin::Primary,
            _ => return None,
        };
        Some(PySynodicOrigin { origin })
    }

    /// NAIF ID of the synodic primary (`Synodic`, `EMR`, `SER`, `GSE`).
    ///
    /// Returns:
    ///     Optional[int]: NAIF ID of the primary body, or `None` for non-synodic frames
    #[getter]
    fn synodic_primary(&self) -> Option<i32> {
        match self.frame {
            frames::ReferenceFrame::Synodic { primary, .. } => Some(primary),
            frames::ReferenceFrame::EMR | frames::ReferenceFrame::GSE => Some(399),
            frames::ReferenceFrame::SER => Some(10),
            _ => None,
        }
    }

    /// NAIF ID of the synodic secondary (`Synodic`, `EMR`, `SER`, `GSE`).
    ///
    /// Returns:
    ///     Optional[int]: NAIF ID of the secondary body, or `None` for non-synodic frames
    #[getter]
    fn synodic_secondary(&self) -> Option<i32> {
        match self.frame {
            frames::ReferenceFrame::Synodic { secondary, .. } => Some(secondary),
            frames::ReferenceFrame::EMR => Some(301),
            frames::ReferenceFrame::SER => Some(399),
            frames::ReferenceFrame::GSE => Some(10),
            _ => None,
        }
    }

    /// Parses a `ReferenceFrame` from its string representation (named
    /// variants only, case-insensitive), plus the common aliases `"ECI"`
    /// (-> `GCRF`) and `"ECEF"` (-> `ITRF`).
    ///
    /// Args:
    ///     s (str): String representation of the reference frame
    ///
    /// Returns:
    ///     ReferenceFrame: Parsed reference frame
    ///
    /// Raises:
    ///     ValueError: If `s` is not a recognized reference frame name
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     assert bh.ReferenceFrame.from_string("ECI") == bh.ReferenceFrame.GCRF
    ///     ```
    #[staticmethod]
    fn from_string(s: &str) -> PyResult<Self> {
        s.parse::<frames::ReferenceFrame>()
            .map(|frame| PyReferenceFrame { frame })
            .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __str__(&self) -> String {
        self.frame.to_string()
    }

    fn __repr__(&self) -> String {
        format!("ReferenceFrame.{}", self.frame)
    }
}

/// Computes the rotation matrix transforming `from_frame` axes into
/// `to_frame` axes at `epc`.
///
/// Purely an orientation query: does not depend on, and does not query,
/// either frame's center. This does not mean SPK is never touched: EMR,
/// SER, and GSE orientations are themselves derived from SPK
/// state/acceleration (auto-loading `de440s`), so a query involving one
/// of those frames still queries SPK.
///
/// Args:
///     from_frame (ReferenceFrame): Source reference frame
///     to_frame (ReferenceFrame): Target reference frame
///     epc (Epoch): Epoch instant for computation of the transformation
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming `from_frame` -> `to_frame`
///
/// Raises:
///     RuntimeError: If either frame's orientation cannot be evaluated at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     r = bh.rotation_frame_to_frame(bh.ReferenceFrame.MCI, bh.ReferenceFrame.MCMF, epc)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(from_frame, to_frame, epc)")]
#[pyo3(name = "rotation_frame_to_frame")]
unsafe fn py_rotation_frame_to_frame<'py>(
    py: Python<'py>,
    from_frame: PyReferenceFrame,
    to_frame: PyReferenceFrame,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let mat = frames::rotation_frame_to_frame(from_frame.frame, to_frame.frame, epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(matrix_to_numpy!(py, mat, 3, 3, f64))
}

/// Registers (or replaces) a user-defined body-fixed frame under `key`.
///
/// The frame becomes usable as `ReferenceFrame.BodyFixedCustom(center, key)`
/// in every frame-router function. `rotation` must be a callable taking an
/// `Epoch` and returning a 3x3 rotation matrix (ICRF -> body-fixed, i.e.
/// `v_body = R @ v_icrf`) as a numpy array or nested list. If `omega` is
/// omitted, the angular velocity used for the velocity transport term is
/// derived numerically from `rotation` by central differencing; provide an
/// `omega` callable (Epoch -> length-3 array, rad/s, body-fixed axes) when
/// the spin model has an analytic rate.
///
/// This enables custom orientation models — e.g. asteroid spin states from
/// the DAMIT database — without any change to the frame-router API.
///
/// Args:
///     key (int): Identifier to register the frame under; the same value is
///         passed to `ReferenceFrame.BodyFixedCustom`.
///     rotation (callable): Callable `Epoch -> 3x3 array` returning the
///         ICRF -> body-fixed rotation matrix.
///     omega (callable, optional): Callable `Epoch -> length-3 array`
///         returning the frame's angular velocity in body-fixed axes (rad/s).
///
/// Example:
///     ```python
///     import numpy as np
///     import brahe as bh
///
///     t0 = bh.Epoch.from_date(2024, 1, 1, bh.TimeSystem.TDB)
///
///     def spin(epc):
///         theta = 1.0e-3 * (epc - t0)
///         c, s = np.cos(theta), np.sin(theta)
///         return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
///
///     bh.register_custom_frame(42, spin)
///     frame = bh.ReferenceFrame.BodyFixedCustom(-20001, 42)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(key, rotation, omega=None)")]
#[pyo3(name = "register_custom_frame")]
#[pyo3(signature = (key, rotation, omega=None))]
fn py_register_custom_frame(key: u32, rotation: Py<PyAny>, omega: Option<Py<PyAny>>) {
    let rotation_fn = move |epc: time::Epoch| -> SMatrix3 {
        Python::attach(|py| {
            let py_epc = PyEpoch { obj: epc };
            let result = rotation
                .call1(py, (py_epc,))
                .expect("custom frame rotation callback raised an exception");
            pyany_to_smatrix::<3, 3>(result.bind(py))
                .expect("custom frame rotation callback must return a 3x3 matrix")
        })
    };
    let omega_fn = omega.map(|omega| -> Box<frames::CustomFrameOmega> {
        Box::new(move |epc: time::Epoch| {
            Python::attach(|py| {
                let py_epc = PyEpoch { obj: epc };
                let result = omega
                    .call1(py, (py_epc,))
                    .expect("custom frame omega callback raised an exception");
                pyany_to_svector::<3>(result.bind(py))
                    .expect("custom frame omega callback must return a length-3 vector")
            })
        })
    });
    frames::register_custom_frame(key, rotation_fn, omega_fn);
}

/// Removes the custom frame registered under `key`.
///
/// Args:
///     key (int): Identifier the frame was registered under.
///
/// Returns:
///     bool: True if a frame was registered under `key` and has been removed.
#[pyfunction]
#[pyo3(text_signature = "(key)")]
#[pyo3(name = "unregister_custom_frame")]
fn py_unregister_custom_frame(key: u32) -> bool {
    frames::unregister_custom_frame(key)
}

/// Transforms a Cartesian position from `from_frame` to `to_frame` at `epc`.
///
/// Same hub-and-spoke design as `state_frame_to_frame`, without the velocity
/// transport terms. Same-center conversions skip the translation lookup,
/// but EMR/SER/GSE orientation still queries SPK ephemerides
/// (auto-loading `de440s`), so a same-center conversion involving one of
/// those frames is not SPK-free.
///
/// Args:
///     from_frame (ReferenceFrame): Source reference frame
///     to_frame (ReferenceFrame): Target reference frame
///     epc (Epoch): Epoch instant for computation of the transformation
///     x (numpy.ndarray or list): Cartesian position in `from_frame` axes/center (m), shape `(3,)`
///
/// Returns:
///     numpy.ndarray: Cartesian position in `to_frame` axes/center (m), shape `(3,)`
///
/// Raises:
///     RuntimeError: If either frame's orientation cannot be evaluated at
///         `epc`, or if the two frames have different centers and no
///         ephemeris path exists between them
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_gcrf = [bh.R_EARTH + 500e3, 0.0, 0.0]
///     x_itrf = bh.position_frame_to_frame(bh.ReferenceFrame.GCRF, bh.ReferenceFrame.ITRF, epc, x_gcrf)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(from_frame, to_frame, epc, x)")]
#[pyo3(name = "position_frame_to_frame")]
fn py_position_frame_to_frame<'py>(
    py: Python<'py>,
    from_frame: PyReferenceFrame,
    to_frame: PyReferenceFrame,
    epc: &PyEpoch,
    x: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::position_frame_to_frame(
        from_frame.frame,
        to_frame.frame,
        epc.obj,
        pyany_to_svector::<3>(&x)?,
    )
    .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

/// Transforms a Cartesian state (position and velocity) from `from_frame` to
/// `to_frame` at `epc`.
///
/// Uses a hub-and-spoke design: the state is first rotated from
/// `from_frame` axes into ICRF axes (an exact orientation +
/// velocity-transport transform, still centered on `from_frame`'s origin),
/// then re-centered onto `to_frame`'s origin if the two frames have
/// different centers, then rotated into `to_frame` axes. Same-center
/// conversions (e.g. GCRF <-> ITRF) skip the re-centering step, so that
/// step never touches SPK; EMR/SER/GSE orientation still queries SPK
/// ephemerides (auto-loading `de440s`) even for a same-center conversion
/// like GCRF <-> GSE.
///
/// Args:
///     from_frame (ReferenceFrame): Source reference frame
///     to_frame (ReferenceFrame): Target reference frame
///     epc (Epoch): Epoch instant for computation of the transformation
///     x (numpy.ndarray or list): Cartesian state in `from_frame` axes/center `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Returns:
///     numpy.ndarray: Cartesian state in `to_frame` axes/center `[position (m), velocity (m/s)]`, shape `(6,)`
///
/// Raises:
///     RuntimeError: If either frame's orientation cannot be evaluated at
///         `epc`, or if the two frames have different centers and no
///         ephemeris path exists between them
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_gcrf = [1e8, -2e8, 5e7, 1.0e3, -2.0e3, 0.5e3]
///     x_lfpa = bh.state_frame_to_frame(bh.ReferenceFrame.GCRF, bh.ReferenceFrame.LFPA, epc, x_gcrf)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(from_frame, to_frame, epc, x)")]
#[pyo3(name = "state_frame_to_frame")]
fn py_state_frame_to_frame<'py>(
    py: Python<'py>,
    from_frame: PyReferenceFrame,
    to_frame: PyReferenceFrame,
    epc: &PyEpoch,
    x: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = frames::state_frame_to_frame(
        from_frame.frame,
        to_frame.frame,
        epc.obj,
        pyany_to_svector::<6>(&x)?,
    )
    .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(vector_to_numpy!(py, vec, 6, f64))
}