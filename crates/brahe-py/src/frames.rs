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