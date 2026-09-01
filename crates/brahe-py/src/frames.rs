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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of transformation matrix. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming `GCRF` -> `ITRF`, shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
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
fn py_rotation_gcrf_to_itrf<'py>(py: Python<'py>, epc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_rotation(py, epc, frames::rotation_gcrf_to_itrf, frames::rotations_gcrf_to_itrf)
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of transformation matrix. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming `ECI` (`GCRF`) -> `ECEF` (`ITRF`), shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
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
fn py_rotation_eci_to_ecef<'py>(py: Python<'py>, epc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_rotation(py, epc, frames::rotation_eci_to_ecef, frames::rotations_eci_to_ecef)
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of transformation matrix. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming `ITRF` -> `GCRF`, shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_eop()
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
fn py_rotation_itrf_to_gcrf<'py>(py: Python<'py>, epc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_rotation(py, epc, frames::rotation_itrf_to_gcrf, frames::rotations_itrf_to_gcrf)
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of transformation matrix. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming `ECEF` (`ITRF`) -> `ECI` (`GCRF`), shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_eop()
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
fn py_rotation_ecef_to_eci<'py>(py: Python<'py>, epc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_rotation(py, epc, frames::rotation_ecef_to_eci, frames::rotations_ecef_to_eci)
}

/// Transforms a position vector from GCRF (Geocentric Celestial Reference Frame)
/// to ITRF (International Terrestrial Reference Frame).
///
/// Applies the full `IAU 2006/2000A` transformation including bias, precession,
/// nutation, Earth rotation, and polar motion corrections using global Earth
/// orientation parameters.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x (numpy.ndarray or list): Position vector in `GCRF` frame (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Position vector in `ITRF` frame (m), shape `(3,)` for a single
///         input, or the batch layout of `x` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
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
///
///     # Batch: one row per position, one shared epoch
///     positions = np.tile(r_gcrf, (10, 1))                     # shape (10, 3)
///     positions_itrf = bh.position_gcrf_to_itrf(epc, positions)  # shape (10, 3)
///
///     # One position at a sequence of epochs
///     epochs = [epc + 60.0 * i for i in range(5)]
///     track = bh.position_gcrf_to_itrf(epochs, r_gcrf)         # shape (5, 3)
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x, axis=-1))]
#[pyo3(text_signature = "(epc, x, axis=-1)")]
#[pyo3(name = "position_gcrf_to_itrf")]
fn py_position_gcrf_to_itrf<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<3>(py, epc, x, axis, frames::position_gcrf_to_itrf, frames::positions_gcrf_to_itrf)
}

/// Transforms a position vector from the Earth Centered Inertial (`ECI`/`GCRF`) frame
/// to the Earth Centered Earth Fixed (`ECEF`/`ITRF`) frame.
///
/// This function is an alias for position_gcrf_to_itrf. Applies the full
/// `IAU 2006/2000A` transformation including bias, precession, nutation, Earth
/// rotation, and polar motion corrections using global Earth orientation parameters.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x (numpy.ndarray or list): Position vector in `ECI` frame (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Position vector in `ECEF` frame (m), shape `(3,)` for a single
///         input, or the batch layout of `x` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
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
#[pyo3(signature = (epc, x, axis=-1))]
#[pyo3(text_signature = "(epc, x, axis=-1)")]
#[pyo3(name = "position_eci_to_ecef")]
fn py_position_eci_to_ecef<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<3>(py, epc, x, axis, frames::position_eci_to_ecef, frames::positions_eci_to_ecef)
}

/// Transforms a position vector from ITRF (International Terrestrial Reference Frame)
/// to GCRF (Geocentric Celestial Reference Frame).
///
/// Applies the full `IAU 2006/2000A` transformation including bias, precession,
/// nutation, Earth rotation, and polar motion corrections using global Earth
/// orientation parameters.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x (numpy.ndarray or list): Position vector in `ITRF` frame (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Position vector in `GCRF` frame (m), shape `(3,)` for a single
///         input, or the batch layout of `x` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
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
#[pyo3(signature = (epc, x, axis=-1))]
#[pyo3(text_signature = "(epc, x, axis=-1)")]
#[pyo3(name = "position_itrf_to_gcrf")]
fn py_position_itrf_to_gcrf<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<3>(py, epc, x, axis, frames::position_itrf_to_gcrf, frames::positions_itrf_to_gcrf)
}

/// Transforms a position vector from the Earth Centered Earth Fixed (`ECEF`/`ITRF`)
/// frame to the Earth Centered Inertial (`ECI`/`GCRF`) frame.
///
/// This function is an alias for position_itrf_to_gcrf. Applies the full
/// `IAU 2006/2000A` transformation including bias, precession, nutation, Earth
/// rotation, and polar motion corrections using global Earth orientation parameters.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x (numpy.ndarray or list): Position vector in `ECEF` frame (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Position vector in `ECI` frame (m), shape `(3,)` for a single
///         input, or the batch layout of `x` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
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
#[pyo3(signature = (epc, x, axis=-1))]
#[pyo3(text_signature = "(epc, x, axis=-1)")]
#[pyo3(name = "position_ecef_to_eci")]
fn py_position_ecef_to_eci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<3>(py, epc, x, axis, frames::position_ecef_to_eci, frames::positions_ecef_to_eci)
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_gcrf (numpy.ndarray or list): State vector in `GCRF` frame `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_gcrf` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: State vector in `ITRF` frame `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_gcrf` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
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
#[pyo3(signature = (epc, x_gcrf, axis=-1))]
#[pyo3(text_signature = "(epc, x_gcrf, axis=-1)")]
#[pyo3(name = "state_gcrf_to_itrf")]
fn py_state_gcrf_to_itrf<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_gcrf: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<6>(py, epc, x_gcrf, axis, frames::state_gcrf_to_itrf, frames::states_gcrf_to_itrf)
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_eci (numpy.ndarray or list): State vector in `ECI` frame `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_eci` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: State vector in `ECEF` frame `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_eci` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
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
///
///     # Batch: one row per state, one shared epoch
///     states_eci = np.tile(state_eci, (10, 1))              # shape (10, 6)
///     states_ecef = bh.state_eci_to_ecef(epc, states_eci)   # shape (10, 6)
///
///     # Column layout: components along axis 0
///     cols_ecef = bh.state_eci_to_ecef(epc, states_eci.T, axis=0)  # shape (6, 10)
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_eci, axis=-1))]
#[pyo3(text_signature = "(epc, x_eci, axis=-1)")]
#[pyo3(name = "state_eci_to_ecef")]
fn py_state_eci_to_ecef<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<6>(py, epc, x_eci, axis, frames::state_eci_to_ecef, frames::states_eci_to_ecef)
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_itrf (numpy.ndarray or list): State vector in `ITRF` frame `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_itrf` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: State vector in `GCRF` frame `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_itrf` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
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
#[pyo3(signature = (epc, x_itrf, axis=-1))]
#[pyo3(text_signature = "(epc, x_itrf, axis=-1)")]
#[pyo3(name = "state_itrf_to_gcrf")]
fn py_state_itrf_to_gcrf<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_itrf: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<6>(py, epc, x_itrf, axis, frames::state_itrf_to_gcrf, frames::states_itrf_to_gcrf)
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_ecef (numpy.ndarray or list): State vector in `ECEF` frame `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_ecef` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: State vector in `ECI` frame `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_ecef` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
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
#[pyo3(signature = (epc, x_ecef, axis=-1))]
#[pyo3(text_signature = "(epc, x_ecef, axis=-1)")]
#[pyo3(name = "state_ecef_to_eci")]
fn py_state_ecef_to_eci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_ecef: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<6>(py, epc, x_ecef, axis, frames::state_ecef_to_eci, frames::states_ecef_to_eci)
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
///     x (numpy.ndarray or list): Position vector in `GCRF` frame (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Position vector in `EME2000` frame (m), shape `(3,)` for a single
///         input, or the batch layout of `x` (shape `(n, 3)` for a single vector
///         for batched input.
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
#[pyo3(signature = (x, axis=-1))]
#[pyo3(text_signature = "(x, axis=-1)")]
#[pyo3(name = "position_gcrf_to_eme2000")]
fn py_position_gcrf_to_eme2000<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec::<3>(py, x, axis, frames::position_gcrf_to_eme2000, frames::positions_gcrf_to_eme2000)
}

/// Transforms a position vector from EME2000 (Earth Mean Equator and Equinox of J2000.0)
/// to GCRF (Geocentric Celestial Reference Frame).
///
/// Applies the inverse frame bias correction to account for the small offset between
/// the J2000.0 mean equator and equinox and GCRF. This is a constant transformation
/// that does not vary with time.
///
/// Args:
///     x (numpy.ndarray or list): Position vector in `EME2000` frame (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Position vector in `GCRF` frame (m), shape `(3,)` for a single
///         input, or the batch layout of `x` (shape `(n, 3)` for a single vector
///         for batched input.
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
#[pyo3(signature = (x, axis=-1))]
#[pyo3(text_signature = "(x, axis=-1)")]
#[pyo3(name = "position_eme2000_to_gcrf")]
fn py_position_eme2000_to_gcrf<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec::<3>(py, x, axis, frames::position_eme2000_to_gcrf, frames::positions_eme2000_to_gcrf)
}

/// Transforms a state vector (position and velocity) from GCRF (Geocentric Celestial
/// Reference Frame) to EME2000 (Earth Mean Equator and Equinox of J2000.0).
///
/// Applies the frame bias correction to both position and velocity. Because the
/// transformation does not vary with time, the velocity is directly rotated without
/// additional correction terms.
///
/// Args:
///     x_gcrf (numpy.ndarray or list): State vector in `GCRF` frame `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_gcrf` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: State vector in `EME2000` frame `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_gcrf` (shape `(n, 6)` for a single vector
///         for batched input.
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
#[pyo3(signature = (x_gcrf, axis=-1))]
#[pyo3(text_signature = "(x_gcrf, axis=-1)")]
#[pyo3(name = "state_gcrf_to_eme2000")]
fn py_state_gcrf_to_eme2000<'py>(
    py: Python<'py>,
    x_gcrf: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec::<6>(py, x_gcrf, axis, frames::state_gcrf_to_eme2000, frames::states_gcrf_to_eme2000)
}

/// Transforms a state vector (position and velocity) from EME2000 (Earth Mean Equator
/// and Equinox of J2000.0) to GCRF (Geocentric Celestial Reference Frame).
///
/// Applies the inverse frame bias correction to both position and velocity. Because
/// the transformation does not vary with time, the velocity is directly rotated without
/// additional correction terms.
///
/// Args:
///     x_eme2000 (numpy.ndarray or list): State vector in `EME2000` frame `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_eme2000` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: State vector in `GCRF` frame `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_eme2000` (shape `(n, 6)` for a single vector
///         for batched input.
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
#[pyo3(signature = (x_eme2000, axis=-1))]
#[pyo3(text_signature = "(x_eme2000, axis=-1)")]
#[pyo3(name = "state_eme2000_to_gcrf")]
fn py_state_eme2000_to_gcrf<'py>(
    py: Python<'py>,
    x_eme2000: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec::<6>(py, x_eme2000, axis, frames::state_eme2000_to_gcrf, frames::states_eme2000_to_gcrf)
}

// ============================================================================
// IAU/WGCCRE Body Rotation Model
// ============================================================================

/// Computes the rotation matrix from the ICRF to the IAU/WGCCRE body-fixed
/// frame of `naif_id` at `epc`.
///
/// Args:
///     naif_id (int): NAIF ID of the body (see `iau_rotation_model_ids` for the supported set)
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation matrix. A sequence evaluates
///         one matrix per epoch.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming ICRF -> body-fixed, shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
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
fn py_rotation_icrf_to_body_fixed_iau<'py>(
    py: Python<'py>,
    naif_id: i32,
    epc: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_rotation(
        py,
        epc,
        |e| frames::rotation_icrf_to_body_fixed_iau(naif_id, e),
        |es| frames::rotations_icrf_to_body_fixed_iau(naif_id, es),
    )
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation matrix. A sequence evaluates
///         one matrix per epoch.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming MCI -> MCMF, shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
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
fn py_rotation_mci_to_mcmf<'py>(py: Python<'py>, epc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_rotation(py, epc, frames::rotation_mci_to_mcmf, frames::rotations_mci_to_mcmf)
}

/// Computes the rotation matrix from Mars-Centered Mars-Fixed (MCMF) to
/// Mars-Centered Inertial (MCI). Inverse of `rotation_mci_to_mcmf`.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation matrix. A sequence evaluates
///         one matrix per epoch.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming MCMF -> MCI, shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
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
fn py_rotation_mcmf_to_mci<'py>(py: Python<'py>, epc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_rotation(py, epc, frames::rotation_mcmf_to_mci, frames::rotations_mcmf_to_mci)
}

/// Transforms a Cartesian Mars-inertial (MCI) position into the equivalent
/// Cartesian Mars-fixed (MCMF) position.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_mci (numpy.ndarray or list): Cartesian MCI position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_mci` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian MCMF position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_mci` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_mcmf = bh.position_mci_to_mcmf(epc, [bh.R_MARS + 400e3, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_mci, axis=-1))]
#[pyo3(text_signature = "(epc, x_mci, axis=-1)")]
#[pyo3(name = "position_mci_to_mcmf")]
fn py_position_mci_to_mcmf<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_mci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<3>(py, epc, x_mci, axis, frames::position_mci_to_mcmf, frames::positions_mci_to_mcmf)
}

/// Transforms a Cartesian Mars-fixed (MCMF) position into the equivalent
/// Cartesian Mars-inertial (MCI) position.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_mcmf (numpy.ndarray or list): Cartesian MCMF position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_mcmf` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian MCI position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_mcmf` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_mci = bh.position_mcmf_to_mci(epc, [bh.R_MARS, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_mcmf, axis=-1))]
#[pyo3(text_signature = "(epc, x_mcmf, axis=-1)")]
#[pyo3(name = "position_mcmf_to_mci")]
fn py_position_mcmf_to_mci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_mcmf: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<3>(py, epc, x_mcmf, axis, frames::position_mcmf_to_mci, frames::positions_mcmf_to_mci)
}

/// Transforms a Cartesian Mars-inertial (MCI) state (position and velocity)
/// into the equivalent Cartesian Mars-fixed (MCMF) state.
///
/// The velocity transformation accounts for the transport term induced by
/// Mars' rotation.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_mci (numpy.ndarray or list): Cartesian MCI state `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_mci` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian MCMF state `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_mci` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_mci, axis=-1))]
#[pyo3(text_signature = "(epc, x_mci, axis=-1)")]
#[pyo3(name = "state_mci_to_mcmf")]
fn py_state_mci_to_mcmf<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_mci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<6>(py, epc, x_mci, axis, frames::state_mci_to_mcmf, frames::states_mci_to_mcmf)
}

/// Transforms a Cartesian Mars-fixed (MCMF) state (position and velocity)
/// into the equivalent Cartesian Mars-inertial (MCI) state. Inverse of
/// `state_mci_to_mcmf`.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_mcmf (numpy.ndarray or list): Cartesian MCMF state `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_mcmf` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian MCI state `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_mcmf` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_mcmf, axis=-1))]
#[pyo3(text_signature = "(epc, x_mcmf, axis=-1)")]
#[pyo3(name = "state_mcmf_to_mci")]
fn py_state_mcmf_to_mci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_mcmf: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<6>(py, epc, x_mcmf, axis, frames::state_mcmf_to_mci, frames::states_mcmf_to_mci)
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_eci (numpy.ndarray or list): Cartesian ECI position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_eci` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian EMBI position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_eci` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_emb = bh.position_eci_to_emb(epc, [7e6, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_eci, axis=-1))]
#[pyo3(text_signature = "(epc, x_eci, axis=-1)")]
#[pyo3(name = "position_eci_to_emb")]
fn py_position_eci_to_emb<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<3>(py, epc, x_eci, axis, frames::position_eci_to_emb, frames::positions_eci_to_emb)
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_emb (numpy.ndarray or list): Cartesian EMBI position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_emb` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian ECI position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_emb` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_eci = bh.position_emb_to_eci(epc, [7e6, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_emb, axis=-1))]
#[pyo3(text_signature = "(epc, x_emb, axis=-1)")]
#[pyo3(name = "position_emb_to_eci")]
fn py_position_emb_to_eci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_emb: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<3>(py, epc, x_emb, axis, frames::position_emb_to_eci, frames::positions_emb_to_eci)
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_eci (numpy.ndarray or list): Cartesian ECI state (m; m/s), shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_eci` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian EMBI state (m; m/s), shape `(6,)` for a single
///         input, or the batch layout of `x_eci` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_emb = bh.state_eci_to_emb(epc, [7e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_eci, axis=-1))]
#[pyo3(text_signature = "(epc, x_eci, axis=-1)")]
#[pyo3(name = "state_eci_to_emb")]
fn py_state_eci_to_emb<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<6>(py, epc, x_eci, axis, frames::state_eci_to_emb, frames::states_eci_to_emb)
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_emb (numpy.ndarray or list): Cartesian EMBI state (m; m/s), shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_emb` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian ECI state (m; m/s), shape `(6,)` for a single
///         input, or the batch layout of `x_emb` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_eci = bh.state_emb_to_eci(epc, [7e6, 0.0, 0.0, 0.0, 7.5e3, 0.0])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_emb, axis=-1))]
#[pyo3(text_signature = "(epc, x_emb, axis=-1)")]
#[pyo3(name = "state_emb_to_eci")]
fn py_state_emb_to_eci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_emb: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<6>(py, epc, x_emb, axis, frames::state_emb_to_eci, frames::states_emb_to_eci)
}


/// Transforms a Cartesian Earth-inertial (ECI) position into the equivalent
/// Cartesian Mars-inertial (MCI) position.
///
/// The MCI origin is the Mars body center (NAIF ID 499); the `mar099s`
/// satellite ephemeris kernel is auto-loaded for the body-center leg. Auto-initializes
/// the default `de440s` ephemeris if no SPK kernel is loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_eci (numpy.ndarray or list): Cartesian ECI position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_eci` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian MCI position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_eci` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_mci = bh.position_eci_to_mci(epc, [1e7, 2e7, 3e7])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_eci, axis=-1))]
#[pyo3(text_signature = "(epc, x_eci, axis=-1)")]
#[pyo3(name = "position_eci_to_mci")]
fn py_position_eci_to_mci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<3>(py, epc, x_eci, axis, frames::position_eci_to_mci, frames::positions_eci_to_mci)
}

/// Transforms a Cartesian Mars-inertial (MCI) position into the equivalent
/// Cartesian Earth-inertial (ECI) position.
///
/// The MCI origin is the Mars body center (NAIF ID 499); the `mar099s`
/// satellite ephemeris kernel is auto-loaded for the body-center leg. Auto-initializes
/// the default `de440s` ephemeris if no SPK kernel is loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_mci (numpy.ndarray or list): Cartesian MCI position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_mci` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian ECI position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_mci` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_eci = bh.position_mci_to_eci(epc, [1e7, 2e7, 3e7])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_mci, axis=-1))]
#[pyo3(text_signature = "(epc, x_mci, axis=-1)")]
#[pyo3(name = "position_mci_to_eci")]
fn py_position_mci_to_eci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_mci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<3>(py, epc, x_mci, axis, frames::position_mci_to_eci, frames::positions_mci_to_eci)
}

/// Transforms a Cartesian Earth-inertial (ECI) state (position and velocity)
/// into the equivalent Cartesian Mars-inertial (MCI) state.
///
/// The MCI origin is the Mars body center (NAIF ID 499); the `mar099s`
/// satellite ephemeris kernel is auto-loaded for the body-center leg. Auto-initializes
/// the default `de440s` ephemeris if no SPK kernel is loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_eci (numpy.ndarray or list): Cartesian ECI state `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_eci` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian MCI state `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_eci` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_mci = bh.state_eci_to_mci(epc, [1e7, 2e7, 3e7, 1.0, 2.0, 3.0])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_eci, axis=-1))]
#[pyo3(text_signature = "(epc, x_eci, axis=-1)")]
#[pyo3(name = "state_eci_to_mci")]
fn py_state_eci_to_mci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<6>(py, epc, x_eci, axis, frames::state_eci_to_mci, frames::states_eci_to_mci)
}

/// Transforms a Cartesian Mars-inertial (MCI) state (position and velocity)
/// into the equivalent Cartesian Earth-inertial (ECI) state.
///
/// The MCI origin is the Mars body center (NAIF ID 499); the `mar099s`
/// satellite ephemeris kernel is auto-loaded for the body-center leg. Auto-initializes
/// the default `de440s` ephemeris if no SPK kernel is loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_mci (numpy.ndarray or list): Cartesian MCI state `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_mci` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian ECI state `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_mci` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_eci = bh.state_mci_to_eci(epc, [1e7, 2e7, 3e7, 1.0, 2.0, 3.0])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_mci, axis=-1))]
#[pyo3(text_signature = "(epc, x_mci, axis=-1)")]
#[pyo3(name = "state_mci_to_eci")]
fn py_state_mci_to_eci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_mci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<6>(py, epc, x_mci, axis, frames::state_mci_to_eci, frames::states_mci_to_eci)
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation matrix. A sequence evaluates
///         one matrix per epoch.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming LCI -> LFPA, shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
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
fn py_rotation_lci_to_lfpa<'py>(py: Python<'py>, epc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_rotation(py, epc, frames::rotation_lci_to_lfpa, frames::rotations_lci_to_lfpa)
}

/// Computes the rotation matrix from Lunar-Fixed Principal Axis (LFPA) to
/// Lunar-Centered Inertial (LCI). Inverse of `rotation_lci_to_lfpa`.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation matrix. A sequence evaluates
///         one matrix per epoch.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming LFPA -> LCI, shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
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
fn py_rotation_lfpa_to_lci<'py>(py: Python<'py>, epc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_rotation(py, epc, frames::rotation_lfpa_to_lci, frames::rotations_lfpa_to_lci)
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation matrix. A sequence evaluates
///         one matrix per epoch.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming LCI -> LFME, shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
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
fn py_rotation_lci_to_lfme<'py>(py: Python<'py>, epc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_rotation(py, epc, frames::rotation_lci_to_lfme, frames::rotations_lci_to_lfme)
}

/// Computes the rotation matrix from Lunar-Fixed Mean Earth/polar-axis (LFME)
/// to Lunar-Centered Inertial (LCI). Inverse of `rotation_lci_to_lfme`.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation matrix. A sequence evaluates
///         one matrix per epoch.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming LFME -> LCI, shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
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
fn py_rotation_lfme_to_lci<'py>(py: Python<'py>, epc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_rotation(py, epc, frames::rotation_lfme_to_lci, frames::rotations_lfme_to_lci)
}

/// Transforms a Cartesian Lunar-inertial (LCI) position into the equivalent
/// Cartesian Lunar-Fixed Principal Axis (LFPA) position.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_lci (numpy.ndarray or list): Cartesian LCI position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_lci` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian LFPA position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_lci` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_lfpa = bh.position_lci_to_lfpa(epc, [bh.R_MOON + 100e3, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_lci, axis=-1))]
#[pyo3(text_signature = "(epc, x_lci, axis=-1)")]
#[pyo3(name = "position_lci_to_lfpa")]
fn py_position_lci_to_lfpa<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_lci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<3>(py, epc, x_lci, axis, frames::position_lci_to_lfpa, frames::positions_lci_to_lfpa)
}

/// Transforms a Cartesian Lunar-Fixed Principal Axis (LFPA) position into the
/// equivalent Cartesian Lunar-inertial (LCI) position.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_lfpa (numpy.ndarray or list): Cartesian LFPA position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_lfpa` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian LCI position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_lfpa` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_lci = bh.position_lfpa_to_lci(epc, [bh.R_MOON, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_lfpa, axis=-1))]
#[pyo3(text_signature = "(epc, x_lfpa, axis=-1)")]
#[pyo3(name = "position_lfpa_to_lci")]
fn py_position_lfpa_to_lci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_lfpa: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<3>(py, epc, x_lfpa, axis, frames::position_lfpa_to_lci, frames::positions_lfpa_to_lci)
}

/// Transforms a Cartesian Lunar-inertial (LCI) position into the equivalent
/// Cartesian Lunar-Fixed Mean Earth/polar-axis (LFME) position.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_lci (numpy.ndarray or list): Cartesian LCI position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_lci` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian LFME position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_lci` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_lfme = bh.position_lci_to_lfme(epc, [bh.R_MOON + 100e3, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_lci, axis=-1))]
#[pyo3(text_signature = "(epc, x_lci, axis=-1)")]
#[pyo3(name = "position_lci_to_lfme")]
fn py_position_lci_to_lfme<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_lci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<3>(py, epc, x_lci, axis, frames::position_lci_to_lfme, frames::positions_lci_to_lfme)
}

/// Transforms a Cartesian Lunar-Fixed Mean Earth/polar-axis (LFME) position
/// into the equivalent Cartesian Lunar-inertial (LCI) position.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_lfme (numpy.ndarray or list): Cartesian LFME position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_lfme` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian LCI position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_lfme` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_lci = bh.position_lfme_to_lci(epc, [bh.R_MOON, 0.0, 0.0])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_lfme, axis=-1))]
#[pyo3(text_signature = "(epc, x_lfme, axis=-1)")]
#[pyo3(name = "position_lfme_to_lci")]
fn py_position_lfme_to_lci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_lfme: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<3>(py, epc, x_lfme, axis, frames::position_lfme_to_lci, frames::positions_lfme_to_lci)
}

/// Transforms a Cartesian Lunar-inertial (LCI) state (position and velocity)
/// into the equivalent Cartesian Lunar-Fixed Principal Axis (LFPA) state.
///
/// The velocity transformation accounts for the transport term induced by
/// the Moon's rotation.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_lci (numpy.ndarray or list): Cartesian LCI state `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_lci` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian LFPA state `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_lci` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_lci, axis=-1))]
#[pyo3(text_signature = "(epc, x_lci, axis=-1)")]
#[pyo3(name = "state_lci_to_lfpa")]
fn py_state_lci_to_lfpa<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_lci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<6>(py, epc, x_lci, axis, frames::state_lci_to_lfpa, frames::states_lci_to_lfpa)
}

/// Transforms a Cartesian Lunar-Fixed Principal Axis (LFPA) state (position
/// and velocity) into the equivalent Cartesian Lunar-inertial (LCI) state.
/// Inverse of `state_lci_to_lfpa`.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_lfpa (numpy.ndarray or list): Cartesian LFPA state `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_lfpa` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian LCI state `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_lfpa` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_lfpa, axis=-1))]
#[pyo3(text_signature = "(epc, x_lfpa, axis=-1)")]
#[pyo3(name = "state_lfpa_to_lci")]
fn py_state_lfpa_to_lci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_lfpa: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<6>(py, epc, x_lfpa, axis, frames::state_lfpa_to_lci, frames::states_lfpa_to_lci)
}

/// Transforms a Cartesian Lunar-inertial (LCI) state (position and velocity)
/// into the equivalent Cartesian Lunar-Fixed Mean Earth/polar-axis (LFME)
/// state.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_lci (numpy.ndarray or list): Cartesian LCI state `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_lci` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian LFME state `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_lci` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_lci, axis=-1))]
#[pyo3(text_signature = "(epc, x_lci, axis=-1)")]
#[pyo3(name = "state_lci_to_lfme")]
fn py_state_lci_to_lfme<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_lci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<6>(py, epc, x_lci, axis, frames::state_lci_to_lfme, frames::states_lci_to_lfme)
}

/// Transforms a Cartesian Lunar-Fixed Mean Earth/polar-axis (LFME) state
/// (position and velocity) into the equivalent Cartesian Lunar-inertial (LCI)
/// state. Inverse of `state_lci_to_lfme`.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_lfme (numpy.ndarray or list): Cartesian LFME state `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_lfme` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian LCI state `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_lfme` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_lfme, axis=-1))]
#[pyo3(text_signature = "(epc, x_lfme, axis=-1)")]
#[pyo3(name = "state_lfme_to_lci")]
fn py_state_lfme_to_lci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_lfme: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<6>(py, epc, x_lfme, axis, frames::state_lfme_to_lci, frames::states_lfme_to_lci)
}

/// Transforms a Cartesian Earth-inertial (ECI) position into the equivalent
/// Cartesian Lunar-inertial (LCI) position.
///
/// The LCI origin is the Moon's body center (NAIF ID 301). Auto-initializes
/// the default `de440s` ephemeris if no SPK kernel is loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_eci (numpy.ndarray or list): Cartesian ECI position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_eci` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian LCI position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_eci` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_lci = bh.position_eci_to_lci(epc, [1e7, 2e7, 3e7])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_eci, axis=-1))]
#[pyo3(text_signature = "(epc, x_eci, axis=-1)")]
#[pyo3(name = "position_eci_to_lci")]
fn py_position_eci_to_lci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<3>(py, epc, x_eci, axis, frames::position_eci_to_lci, frames::positions_eci_to_lci)
}

/// Transforms a Cartesian Lunar-inertial (LCI) position into the equivalent
/// Cartesian Earth-inertial (ECI) position.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_lci (numpy.ndarray or list): Cartesian LCI position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_lci` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian ECI position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_lci` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_eci = bh.position_lci_to_eci(epc, [1e7, 2e7, 3e7])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_lci, axis=-1))]
#[pyo3(text_signature = "(epc, x_lci, axis=-1)")]
#[pyo3(name = "position_lci_to_eci")]
fn py_position_lci_to_eci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_lci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<3>(py, epc, x_lci, axis, frames::position_lci_to_eci, frames::positions_lci_to_eci)
}

/// Transforms a Cartesian Earth-inertial (ECI) state (position and velocity)
/// into the equivalent Cartesian Lunar-inertial (LCI) state.
///
/// The LCI origin is the Moon's body center (NAIF ID 301). Auto-initializes
/// the default `de440s` ephemeris if no SPK kernel is loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_eci (numpy.ndarray or list): Cartesian ECI state `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_eci` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian LCI state `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_eci` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_lci = bh.state_eci_to_lci(epc, [1e7, 2e7, 3e7, 1.0, 2.0, 3.0])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_eci, axis=-1))]
#[pyo3(text_signature = "(epc, x_eci, axis=-1)")]
#[pyo3(name = "state_eci_to_lci")]
fn py_state_eci_to_lci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<6>(py, epc, x_eci, axis, frames::state_eci_to_lci, frames::states_eci_to_lci)
}

/// Transforms a Cartesian Lunar-inertial (LCI) state (position and velocity)
/// into the equivalent Cartesian Earth-inertial (ECI) state.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_lci (numpy.ndarray or list): Cartesian LCI state `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_lci` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian ECI state `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_lci` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_eci = bh.state_lci_to_eci(epc, [1e7, 2e7, 3e7, 1.0, 2.0, 3.0])
///     ```
#[pyfunction]
#[pyo3(signature = (epc, x_lci, axis=-1))]
#[pyo3(text_signature = "(epc, x_lci, axis=-1)")]
#[pyo3(name = "state_lci_to_eci")]
fn py_state_lci_to_eci<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_lci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_epoch_vec::<6>(py, epc, x_lci, axis, frames::state_lci_to_eci, frames::states_lci_to_eci)
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one matrix per epoch.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming GCRF -> EMR axes, shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
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
fn py_rotation_gcrf_to_emr<'py>(py: Python<'py>, epc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_rotation(py, epc, frames::rotation_gcrf_to_emr, frames::rotations_gcrf_to_emr)
}

/// Computes the rotation matrix from Earth-Moon Rotating (EMR) frame axes to
/// GCRF axes. Inverse of `rotation_gcrf_to_emr`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one matrix per epoch.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming EMR -> GCRF axes, shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
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
fn py_rotation_emr_to_gcrf<'py>(py: Python<'py>, epc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_rotation(py, epc, frames::rotation_emr_to_gcrf, frames::rotations_emr_to_gcrf)
}

/// Transforms a Cartesian GCRF position into the equivalent Earth-Moon
/// Rotating (EMR) frame position. The EMR origin is the Earth-Moon
/// barycenter.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_gcrf (numpy.ndarray or list): Cartesian GCRF position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_gcrf` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian EMR position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_gcrf` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_gcrf, axis=-1))]
#[pyo3(text_signature = "(epc, x_gcrf, axis=-1)")]
#[pyo3(name = "position_gcrf_to_emr")]
fn py_position_gcrf_to_emr<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_gcrf: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_vec::<3>(py, epc, x_gcrf, axis, frames::position_gcrf_to_emr, frames::positions_gcrf_to_emr)
}

/// Transforms a Cartesian Earth-Moon Rotating (EMR) frame position into the
/// equivalent Cartesian GCRF position. Inverse of `position_gcrf_to_emr`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_emr (numpy.ndarray or list): Cartesian EMR position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_emr` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian GCRF position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_emr` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_emr, axis=-1))]
#[pyo3(text_signature = "(epc, x_emr, axis=-1)")]
#[pyo3(name = "position_emr_to_gcrf")]
fn py_position_emr_to_gcrf<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_emr: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_vec::<3>(py, epc, x_emr, axis, frames::position_emr_to_gcrf, frames::positions_emr_to_gcrf)
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_gcrf (numpy.ndarray or list): Cartesian GCRF state `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_gcrf` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian EMR state `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_gcrf` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_gcrf, axis=-1))]
#[pyo3(text_signature = "(epc, x_gcrf, axis=-1)")]
#[pyo3(name = "state_gcrf_to_emr")]
fn py_state_gcrf_to_emr<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_gcrf: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_vec::<6>(py, epc, x_gcrf, axis, frames::state_gcrf_to_emr, frames::states_gcrf_to_emr)
}

/// Transforms a Cartesian Earth-Moon Rotating (EMR) frame state (position
/// and velocity) into the equivalent Cartesian GCRF state. Inverse of
/// `state_gcrf_to_emr`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_emr (numpy.ndarray or list): Cartesian EMR state `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_emr` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian GCRF state `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_emr` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_emr, axis=-1))]
#[pyo3(text_signature = "(epc, x_emr, axis=-1)")]
#[pyo3(name = "state_emr_to_gcrf")]
fn py_state_emr_to_gcrf<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_emr: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_vec::<6>(py, epc, x_emr, axis, frames::state_emr_to_gcrf, frames::states_emr_to_gcrf)
}

/// Computes the rotation matrix from GCRF axes to Sun-Earth Rotating (SER)
/// frame axes (NASA TP-20220014814 §4.6.4): x̂ Sun→Earth, ẑ along Earth's
/// orbital angular momentum relative to the Sun.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one matrix per epoch.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming GCRF -> SER axes, shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
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
fn py_rotation_gcrf_to_ser<'py>(py: Python<'py>, epc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_rotation(py, epc, frames::rotation_gcrf_to_ser, frames::rotations_gcrf_to_ser)
}

/// Computes the rotation matrix from Sun-Earth Rotating (SER) frame axes to
/// GCRF axes. Inverse of `rotation_gcrf_to_ser`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one matrix per epoch.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming SER -> GCRF axes, shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
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
fn py_rotation_ser_to_gcrf<'py>(py: Python<'py>, epc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_rotation(py, epc, frames::rotation_ser_to_gcrf, frames::rotations_ser_to_gcrf)
}

/// Transforms a Cartesian GCRF position into the equivalent Sun-Earth
/// Rotating (SER) frame position. The SER origin is the (computed)
/// Sun-Earth barycenter.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_gcrf (numpy.ndarray or list): Cartesian GCRF position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_gcrf` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian SER position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_gcrf` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_gcrf, axis=-1))]
#[pyo3(text_signature = "(epc, x_gcrf, axis=-1)")]
#[pyo3(name = "position_gcrf_to_ser")]
fn py_position_gcrf_to_ser<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_gcrf: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_vec::<3>(py, epc, x_gcrf, axis, frames::position_gcrf_to_ser, frames::positions_gcrf_to_ser)
}

/// Transforms a Cartesian Sun-Earth Rotating (SER) frame position into the
/// equivalent Cartesian GCRF position. Inverse of `position_gcrf_to_ser`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_ser (numpy.ndarray or list): Cartesian SER position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_ser` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian GCRF position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_ser` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_ser, axis=-1))]
#[pyo3(text_signature = "(epc, x_ser, axis=-1)")]
#[pyo3(name = "position_ser_to_gcrf")]
fn py_position_ser_to_gcrf<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_ser: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_vec::<3>(py, epc, x_ser, axis, frames::position_ser_to_gcrf, frames::positions_ser_to_gcrf)
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_gcrf (numpy.ndarray or list): Cartesian GCRF state `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_gcrf` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian SER state `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_gcrf` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_gcrf, axis=-1))]
#[pyo3(text_signature = "(epc, x_gcrf, axis=-1)")]
#[pyo3(name = "state_gcrf_to_ser")]
fn py_state_gcrf_to_ser<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_gcrf: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_vec::<6>(py, epc, x_gcrf, axis, frames::state_gcrf_to_ser, frames::states_gcrf_to_ser)
}

/// Transforms a Cartesian Sun-Earth Rotating (SER) frame state (position
/// and velocity) into the equivalent Cartesian GCRF state. Inverse of
/// `state_gcrf_to_ser`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_ser (numpy.ndarray or list): Cartesian SER state `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_ser` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian GCRF state `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_ser` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_ser, axis=-1))]
#[pyo3(text_signature = "(epc, x_ser, axis=-1)")]
#[pyo3(name = "state_ser_to_gcrf")]
fn py_state_ser_to_gcrf<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_ser: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_vec::<6>(py, epc, x_ser, axis, frames::state_ser_to_gcrf, frames::states_ser_to_gcrf)
}

/// Computes the rotation matrix from GCRF axes to Geocentric Solar Ecliptic
/// (GSE) frame axes (NASA TP-20220014814 §2.5.4/§4.6.5): x̂ Earth→Sun, ẑ
/// normal to the instantaneous ecliptic plane.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one matrix per epoch.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming GCRF -> GSE axes, shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
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
fn py_rotation_gcrf_to_gse<'py>(py: Python<'py>, epc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_rotation(py, epc, frames::rotation_gcrf_to_gse, frames::rotations_gcrf_to_gse)
}

/// Computes the rotation matrix from Geocentric Solar Ecliptic (GSE) frame
/// axes to GCRF axes. Inverse of `rotation_gcrf_to_gse`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one matrix per epoch.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming GSE -> GCRF axes, shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
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
fn py_rotation_gse_to_gcrf<'py>(py: Python<'py>, epc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_rotation(py, epc, frames::rotation_gse_to_gcrf, frames::rotations_gse_to_gcrf)
}

/// Transforms a Cartesian GCRF position into the equivalent Geocentric
/// Solar Ecliptic (GSE) frame position. GSE is Earth-centered.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_gcrf (numpy.ndarray or list): Cartesian GCRF position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_gcrf` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian GSE position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_gcrf` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_gcrf, axis=-1))]
#[pyo3(text_signature = "(epc, x_gcrf, axis=-1)")]
#[pyo3(name = "position_gcrf_to_gse")]
fn py_position_gcrf_to_gse<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_gcrf: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_vec::<3>(py, epc, x_gcrf, axis, frames::position_gcrf_to_gse, frames::positions_gcrf_to_gse)
}

/// Transforms a Cartesian Geocentric Solar Ecliptic (GSE) frame position
/// into the equivalent Cartesian GCRF position. Inverse of
/// `position_gcrf_to_gse`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_gse (numpy.ndarray or list): Cartesian GSE position (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x_gse` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian GCRF position (m), shape `(3,)` for a single
///         input, or the batch layout of `x_gse` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_gse, axis=-1))]
#[pyo3(text_signature = "(epc, x_gse, axis=-1)")]
#[pyo3(name = "position_gse_to_gcrf")]
fn py_position_gse_to_gcrf<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_gse: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_vec::<3>(py, epc, x_gse, axis, frames::position_gse_to_gcrf, frames::positions_gse_to_gcrf)
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
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_gcrf (numpy.ndarray or list): Cartesian GCRF state `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_gcrf` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian GSE state `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_gcrf` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_gcrf, axis=-1))]
#[pyo3(text_signature = "(epc, x_gcrf, axis=-1)")]
#[pyo3(name = "state_gcrf_to_gse")]
fn py_state_gcrf_to_gse<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_gcrf: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_vec::<6>(py, epc, x_gcrf, axis, frames::state_gcrf_to_gse, frames::states_gcrf_to_gse)
}

/// Transforms a Cartesian Geocentric Solar Ecliptic (GSE) frame state
/// (position and velocity) into the equivalent Cartesian GCRF state.
/// Inverse of `state_gcrf_to_gse`.
///
/// Auto-initializes the default `de440s` ephemeris if no SPK kernel is
/// loaded.
///
/// Args:
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x_gse (numpy.ndarray or list): Cartesian GSE state `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x_gse` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian GCRF state `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x_gse` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
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
#[pyo3(signature = (epc, x_gse, axis=-1))]
#[pyo3(text_signature = "(epc, x_gse, axis=-1)")]
#[pyo3(name = "state_gse_to_gcrf")]
fn py_state_gse_to_gcrf<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x_gse: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    try_dispatch_epoch_vec::<6>(py, epc, x_gse, axis, frames::state_gse_to_gcrf, frames::states_gse_to_gcrf)
}

// ============================================================================
// Reference Frame Router
// ============================================================================

/// Origin choice for a generic synodic frame (`CelestialFrame.Synodic`):
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
///     r = bh.rotation_frame_to_frame(bh.CelestialFrame.MCI, bh.CelestialFrame.MCMF, epc)
///     ```
#[pyclass(module = "brahe._brahe", eq, from_py_object)]
#[pyo3(name = "CelestialFrame")]
#[derive(Clone, PartialEq)]
pub struct PyCelestialFrame {
    pub(crate) frame: frames::CelestialFrame,
}

#[pymethods]
impl PyCelestialFrame {
    /// Geocentric Celestial Reference Frame (ICRF-aligned, Earth-centered).
    #[classattr]
    #[allow(non_snake_case)]
    fn GCRF() -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::GCRF }
    }

    /// International Terrestrial Reference Frame (Earth-fixed).
    #[classattr]
    #[allow(non_snake_case)]
    fn ITRF() -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::ITRF }
    }

    /// Alias for `CelestialFrame.GCRF`: the Earth-Centered Inertial (ECI)
    /// frame is realized as GCRF.
    #[classattr]
    #[allow(non_snake_case)]
    fn ECI() -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::ECI }
    }

    /// Alias for `CelestialFrame.ITRF`: the Earth-Centered Earth-Fixed (ECEF)
    /// frame is realized as ITRF.
    #[classattr]
    #[allow(non_snake_case)]
    fn ECEF() -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::ECEF }
    }

    /// Earth Mean Equator and Equinox of J2000.0.
    #[classattr]
    #[allow(non_snake_case)]
    fn EME2000() -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::EME2000 }
    }

    /// Lunar-Centered Inertial (ICRF-aligned, Moon-centered).
    #[classattr]
    #[allow(non_snake_case)]
    fn LCI() -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::LCI }
    }

    /// Lunar-Fixed Principal Axis (DE440 `MOON_PA_DE440`).
    #[classattr]
    #[allow(non_snake_case)]
    fn LFPA() -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::LFPA }
    }

    /// Lunar-Fixed Mean Earth/polar-axis.
    #[classattr]
    #[allow(non_snake_case)]
    fn LFME() -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::LFME }
    }

    /// Mars-Centered Inertial (ICRF-aligned, Mars-centered).
    #[classattr]
    #[allow(non_snake_case)]
    fn MCI() -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::MCI }
    }

    /// Mars-Centered Mars-Fixed (IAU/WGCCRE Mars rotation model).
    #[classattr]
    #[allow(non_snake_case)]
    fn MCMF() -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::MCMF }
    }

    /// Earth-Moon Barycentric Inertial (ICRF-aligned).
    #[classattr]
    #[allow(non_snake_case)]
    fn EMBI() -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::EMBI }
    }

    /// Solar System Barycentric Inertial (ICRF-aligned).
    #[classattr]
    #[allow(non_snake_case)]
    fn SSBI() -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::SSBI }
    }

    /// Earth-Moon Rotating frame (NASA TP-20220014814): x̂ Earth→Moon, ẑ
    /// along the Moon's orbital angular momentum relative to Earth;
    /// centered on the Earth-Moon barycenter.
    #[classattr]
    #[allow(non_snake_case)]
    fn EMR() -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::EMR }
    }

    /// Sun-Earth Rotating frame (NASA TP-20220014814): x̂ Sun→Earth, ẑ
    /// along Earth's orbital angular momentum relative to the Sun;
    /// centered on the (computed) Sun-Earth barycenter.
    #[classattr]
    #[allow(non_snake_case)]
    fn SER() -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::SER }
    }

    /// Geocentric Solar Ecliptic frame (NASA TP-20220014814): x̂ Earth→Sun,
    /// ẑ normal to the instantaneous ecliptic plane; Earth-centered.
    #[classattr]
    #[allow(non_snake_case)]
    fn GSE() -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::GSE }
    }

    /// ICRF-aligned axes centered on the given NAIF ID.
    ///
    /// Args:
    ///     naif_id (int): NAIF ID of the frame's center
    ///
    /// Returns:
    ///     CelestialFrame: ICRF-aligned frame centered on `naif_id`
    #[staticmethod]
    #[allow(non_snake_case)]
    fn BodyCenteredICRF(naif_id: i32) -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::BodyCenteredICRF(naif_id) }
    }

    /// IAU/WGCCRE body-fixed frame of the given NAIF ID, centered on that
    /// same NAIF ID.
    ///
    /// Args:
    ///     naif_id (int): NAIF ID of the body (see `iau_rotation_model_ids` for the supported set)
    ///
    /// Returns:
    ///     CelestialFrame: IAU/WGCCRE body-fixed frame of `naif_id`
    #[staticmethod]
    #[allow(non_snake_case)]
    fn BodyFixedIAU(naif_id: i32) -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::BodyFixedIAU(naif_id) }
    }

    /// Body-fixed frame evaluated from a loaded binary PCK's `frame_id`,
    /// centered on `center`.
    ///
    /// Args:
    ///     center (int): NAIF ID of the frame's center
    ///     frame_id (int): NAIF binary PCK frame class ID (e.g. 31008 for `MOON_PA_DE440`)
    ///
    /// Returns:
    ///     CelestialFrame: Body-fixed frame for `frame_id`, centered on `center`
    #[staticmethod]
    #[allow(non_snake_case)]
    fn BodyFixedPCK(center: i32, frame_id: i32) -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::BodyFixedPCK { center, frame_id } }
    }

    /// Body-fixed frame evaluated from a user-registered rotation callback
    /// (see `register_custom_frame`), centered on `center`.
    ///
    /// For a body without a catalogued NAIF ID, self-assign a unique negative
    /// `center` (mirroring NAIF's convention for non-catalogued objects):
    /// rotation-only queries never consult the center, and translations will
    /// raise unless an ephemeris covering that ID is loaded.
    ///
    /// Center IDs at or below -1_000_000_000 are reserved for synthetic synodic
    /// barycenter centers (see `Synodic`); self-assigning one to a custom body
    /// is unlikely to collide with an actual synodic pair's encoded ID, but is
    /// not rejected, since real user-defined bodies and loaded kernels may
    /// legitimately need any negative ID.
    ///
    /// Args:
    ///     center (int): NAIF ID of the frame's center (may be self-assigned negative)
    ///     key (int): Registry key the frame's callbacks were registered under
    ///
    /// Returns:
    ///     CelestialFrame: Custom body-fixed frame for `key`, centered on `center`
    #[staticmethod]
    #[allow(non_snake_case)]
    fn BodyFixedCustom(center: i32, key: u32) -> Self {
        PyCelestialFrame { frame: frames::CelestialFrame::BodyFixedCustom { center, key } }
    }

    /// Generic two-body synodic (rotating) frame: x̂ from `primary` toward
    /// `secondary`, ẑ along the secondary's orbital angular momentum
    /// relative to the primary, centered per `origin`. The named frames
    /// are specific configurations: `EMR == Synodic(Barycenter, 399, 301)`,
    /// `SER == Synodic(Barycenter, 10, 399)`, `GSE == Synodic(Primary, 399, 10)`.
    ///
    /// Any NAIF ID is accepted for `primary` and `secondary`. For a
    /// `Barycenter` origin, the pair is encoded into a synthetic negative
    /// center ID; this encoding is collision-free when both IDs are in
    /// 0..=999, and both bodies must have packaged GM constants. IDs
    /// outside 0..=999 still construct successfully but produce a
    /// different encoding that no longer maps back to a synthetic
    /// center — this surfaces as an SPK/GM lookup error at transform
    /// time rather than a silent collision.
    ///
    /// Args:
    ///     origin (SynodicOrigin): Origin choice (primary, secondary, or barycenter)
    ///     primary (int): NAIF ID of the primary body
    ///     secondary (int): NAIF ID of the secondary body
    ///
    /// Returns:
    ///     CelestialFrame: Generic synodic frame for the pair
    #[staticmethod]
    #[allow(non_snake_case)]
    fn Synodic(origin: PySynodicOrigin, primary: i32, secondary: i32) -> Self {
        PyCelestialFrame {
            frame: frames::CelestialFrame::Synodic { origin: origin.origin, primary, secondary },
        }
    }

    /// Origin choice of a synodic frame (`Synodic`, `EMR`, `SER`, `GSE`).
    ///
    /// Returns:
    ///     Optional[SynodicOrigin]: Origin choice, or `None` for non-synodic frames
    #[getter]
    fn synodic_origin(&self) -> Option<PySynodicOrigin> {
        let origin = match self.frame {
            frames::CelestialFrame::Synodic { origin, .. } => origin,
            frames::CelestialFrame::EMR | frames::CelestialFrame::SER => {
                frames::SynodicOrigin::Barycenter
            }
            frames::CelestialFrame::GSE => frames::SynodicOrigin::Primary,
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
            frames::CelestialFrame::Synodic { primary, .. } => Some(primary),
            frames::CelestialFrame::EMR | frames::CelestialFrame::GSE => Some(399),
            frames::CelestialFrame::SER => Some(10),
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
            frames::CelestialFrame::Synodic { secondary, .. } => Some(secondary),
            frames::CelestialFrame::EMR => Some(301),
            frames::CelestialFrame::SER => Some(399),
            frames::CelestialFrame::GSE => Some(10),
            _ => None,
        }
    }

    /// Parses a `CelestialFrame` from its string representation (named
    /// variants only, case-insensitive), plus the common aliases `"ECI"`
    /// (-> `GCRF`) and `"ECEF"` (-> `ITRF`).
    ///
    /// Args:
    ///     s (str): String representation of the reference frame
    ///
    /// Returns:
    ///     CelestialFrame: Parsed reference frame
    ///
    /// Raises:
    ///     ValueError: If `s` is not a recognized reference frame name
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     assert bh.CelestialFrame.from_string("ECI") == bh.CelestialFrame.GCRF
    ///     ```
    #[staticmethod]
    fn from_string(s: &str) -> PyResult<Self> {
        s.parse::<frames::CelestialFrame>()
            .map(|frame| PyCelestialFrame { frame })
            .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))
    }

    fn __str__(&self) -> String {
        self.frame.to_string()
    }

    fn __repr__(&self) -> String {
        format!("CelestialFrame.{}", self.frame)
    }
}

/// Extracts a `frames::ReferenceFrame` from a Python object that is either a
/// `CelestialFrame` or a `ReferenceFrame` (`Body`/`OrbitRelative`), for the
/// `*_frame_to_frame` functions that accept either.
fn extract_frame(obj: &Bound<'_, PyAny>) -> PyResult<frames::ReferenceFrame> {
    if let Ok(celestial) = obj.extract::<PyCelestialFrame>() {
        return Ok(frames::ReferenceFrame::Celestial(celestial.frame));
    }
    if let Ok(frame) = obj.extract::<PyReferenceFrame>() {
        return Ok(frame.frame);
    }
    Err(exceptions::PyTypeError::new_err(
        "expected a CelestialFrame or a ReferenceFrame",
    ))
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
/// `from_frame`/`to_frame` accept a `CelestialFrame` or a `ReferenceFrame`: a
/// registered body/sensor frame, or a bound orbit-relative frame (e.g.
/// `ReferenceFrame.RTN("SC")`), resolved by walking the frame registry.
///
/// Args:
///     from_frame (CelestialFrame | ReferenceFrame): Source reference frame
///     to_frame (CelestialFrame | ReferenceFrame): Target reference frame
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one matrix per epoch.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming `from_frame` -> `to_frame`, shape `(3, 3)` for a single epoch or `(n, 3, 3)`
///         for a sequence of `n` epochs.
///
/// Raises:
///     RuntimeError: If either frame's orientation cannot be evaluated at `epc`
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     r = bh.rotation_frame_to_frame(bh.CelestialFrame.MCI, bh.CelestialFrame.MCMF, epc)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(from_frame, to_frame, epc)")]
#[pyo3(name = "rotation_frame_to_frame")]
fn py_rotation_frame_to_frame<'py>(
    py: Python<'py>,
    from_frame: &Bound<'py, PyAny>,
    to_frame: &Bound<'py, PyAny>,
    epc: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let from = extract_frame(from_frame)?;
    let to = extract_frame(to_frame)?;
    let (from2, to2) = (from.clone(), to.clone());
    try_dispatch_epoch_rotation(
        py,
        epc,
        move |e| frames::rotation_frame_to_frame(from.clone(), to.clone(), e),
        move |es| frames::rotations_frame_to_frame(from2.clone(), to2.clone(), es),
    )
}

/// Registers (or replaces) a user-defined body-fixed frame under `key`.
///
/// The frame becomes usable as `CelestialFrame.BodyFixedCustom(center, key)`
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
///         passed to `CelestialFrame.BodyFixedCustom`.
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
///     frame = bh.CelestialFrame.BodyFixedCustom(-20001, 42)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(key, rotation, omega=None)")]
#[pyo3(name = "register_custom_frame")]
#[pyo3(signature = (key, rotation, omega=None))]
fn py_register_custom_frame(key: u32, rotation: Py<PyAny>, omega: Option<Py<PyAny>>) {
    let rotation_fn = move |epc: time::Epoch| -> Result<SMatrix3, RustBraheError> {
        Python::attach(|py| {
            let py_epc = PyEpoch { obj: epc };
            let result = rotation.call1(py, (py_epc,)).map_err(|e| {
                RustBraheError::Error(format!(
                    "custom frame rotation callback raised an exception: {e}"
                ))
            })?;
            pyany_to_smatrix::<3, 3>(result.bind(py)).map_err(|e| {
                RustBraheError::Error(format!(
                    "custom frame rotation callback must return a 3x3 matrix: {e}"
                ))
            })
        })
    };
    let omega_fn = omega.map(|omega| -> Box<frames::CustomFrameOmega> {
        Box::new(move |epc: time::Epoch| {
            Python::attach(|py| {
                let py_epc = PyEpoch { obj: epc };
                let result = omega.call1(py, (py_epc,)).map_err(|e| {
                    RustBraheError::Error(format!(
                        "custom frame omega callback raised an exception: {e}"
                    ))
                })?;
                pyany_to_svector::<3>(result.bind(py)).map_err(|e| {
                    RustBraheError::Error(format!(
                        "custom frame omega callback must return a length-3 vector: {e}"
                    ))
                })
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
/// `from_frame`/`to_frame` accept a `CelestialFrame` or a `ReferenceFrame`: a
/// registered body/sensor frame, or a bound orbit-relative frame (e.g.
/// `ReferenceFrame.RTN("SC")`), resolved by walking the frame registry. A
/// non-celestial frame's origin is the origin of the object it is bound to.
///
/// Args:
///     from_frame (CelestialFrame | ReferenceFrame): Source reference frame
///     to_frame (CelestialFrame | ReferenceFrame): Target reference frame
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x (numpy.ndarray or list): Cartesian position in `from_frame` axes/center (m), shape `(3,)`, or a batch
///         of vectors with the 3 components along `axis` (for example shape `(n, 3)`).
///     axis (int, optional): The axis of `x` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian position in `to_frame` axes/center (m), shape `(3,)` for a single
///         input, or the batch layout of `x` (shape `(n, 3)` for a single vector
///         with a sequence of `n` epochs).
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
///     bh.initialize_eop()
///
///     epc = bh.Epoch.from_datetime(2024, 3, 1, 0, 0, 0.0, 0.0, bh.UTC)
///     x_gcrf = [bh.R_EARTH + 500e3, 0.0, 0.0]
///     x_itrf = bh.position_frame_to_frame(bh.CelestialFrame.GCRF, bh.CelestialFrame.ITRF, epc, x_gcrf)
///     ```
#[pyfunction]
#[pyo3(signature = (from_frame, to_frame, epc, x, axis=-1))]
#[pyo3(text_signature = "(from_frame, to_frame, epc, x, axis=-1)")]
#[pyo3(name = "position_frame_to_frame")]
fn py_position_frame_to_frame<'py>(
    py: Python<'py>,
    from_frame: &Bound<'py, PyAny>,
    to_frame: &Bound<'py, PyAny>,
    epc: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let from = extract_frame(from_frame)?;
    let to = extract_frame(to_frame)?;
    let (from2, to2) = (from.clone(), to.clone());
    try_dispatch_epoch_vec::<3>(
        py,
        epc,
        x,
        axis,
        move |e, v| frames::position_frame_to_frame(from.clone(), to.clone(), e, v),
        move |es, vs| frames::positions_frame_to_frame(from2.clone(), to2.clone(), es, vs),
    )
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
/// `from_frame`/`to_frame` accept a `CelestialFrame` or a `ReferenceFrame`: a
/// registered body/sensor frame, or a bound orbit-relative frame (e.g.
/// `ReferenceFrame.RTN("SC")`), resolved by walking the frame registry. Every link
/// in a non-celestial frame's orientation chain must supply an angular
/// velocity, since the velocity transport term is otherwise undefined.
///
/// Args:
///     from_frame (CelestialFrame | ReferenceFrame): Source reference frame
///     to_frame (CelestialFrame | ReferenceFrame): Target reference frame
///     epc (Epoch or Sequence[Epoch]): Epoch instant for computation of the transformation. A sequence evaluates
///         one epoch per vector (or broadcasts a single vector across all epochs).
///     x (numpy.ndarray or list): Cartesian state in `from_frame` axes/center `[position (m), velocity (m/s)]`, shape `(6,)`, or a batch
///         of vectors with the 6 components along `axis` (for example shape `(n, 6)`).
///     axis (int, optional): The axis of `x` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian state in `to_frame` axes/center `[position (m), velocity (m/s)]`, shape `(6,)` for a single
///         input, or the batch layout of `x` (shape `(n, 6)` for a single vector
///         with a sequence of `n` epochs).
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
///     x_lfpa = bh.state_frame_to_frame(bh.CelestialFrame.GCRF, bh.CelestialFrame.LFPA, epc, x_gcrf)
///     ```
#[pyfunction]
#[pyo3(signature = (from_frame, to_frame, epc, x, axis=-1))]
#[pyo3(text_signature = "(from_frame, to_frame, epc, x, axis=-1)")]
#[pyo3(name = "state_frame_to_frame")]
fn py_state_frame_to_frame<'py>(
    py: Python<'py>,
    from_frame: &Bound<'py, PyAny>,
    to_frame: &Bound<'py, PyAny>,
    epc: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let from = extract_frame(from_frame)?;
    let to = extract_frame(to_frame)?;
    let (from2, to2) = (from.clone(), to.clone());
    try_dispatch_epoch_vec::<6>(
        py,
        epc,
        x,
        axis,
        move |e, v| frames::state_frame_to_frame(from.clone(), to.clone(), e, v),
        move |es, vs| frames::states_frame_to_frame(from2.clone(), to2.clone(), es, vs),
    )
}

// ============================================================================
// ReferenceFrame / BodyFrame and the frame/object registries
// ============================================================================

/// Local orbital frame axes definitions.
///
/// `RTN` is the frame the SANA registries call `RSW`; brahe uses its
/// existing RTN vocabulary (`state_eci_to_rtn`, `covariance_rtn`).
///
/// Every kind is a valid frame identity, which is what parsing a data file
/// needs, but only `RTN` has an axes derivation today. A transform through
/// any other kind raises until issue #452 adds the remaining derivations.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     kind = bh.OrbitRelativeFrameKind.RTN
///     ```
#[pyclass(module = "brahe._brahe", eq, from_py_object)]
#[pyo3(name = "OrbitRelativeFrameKind")]
#[derive(Clone, Copy, PartialEq)]
pub struct PyOrbitRelativeFrameKind {
    pub(crate) kind: frames::OrbitRelativeFrameKind,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyOrbitRelativeFrameKind {
    /// Local-Vertical Local-Horizontal.
    #[classattr]
    fn LVLH() -> Self {
        PyOrbitRelativeFrameKind { kind: frames::OrbitRelativeFrameKind::LVLH }
    }

    /// Radial / transverse (along-track) / normal (cross-track). SANA: RSW.
    #[classattr]
    fn RTN() -> Self {
        PyOrbitRelativeFrameKind { kind: frames::OrbitRelativeFrameKind::RTN }
    }

    /// Normal / tangential / cross-track.
    #[classattr]
    fn NTW() -> Self {
        PyOrbitRelativeFrameKind { kind: frames::OrbitRelativeFrameKind::NTW }
    }

    /// Tangential / normal / cross-track.
    #[classattr]
    fn TNW() -> Self {
        PyOrbitRelativeFrameKind { kind: frames::OrbitRelativeFrameKind::TNW }
    }

    /// Perifocal. SANA-registered only as an inertial-snapshot frame.
    #[classattr]
    fn PQW() -> Self {
        PyOrbitRelativeFrameKind { kind: frames::OrbitRelativeFrameKind::PQW }
    }

    /// Equinoctial. SANA-registered only as an inertial-snapshot frame.
    #[classattr]
    fn EQW() -> Self {
        PyOrbitRelativeFrameKind { kind: frames::OrbitRelativeFrameKind::EQW }
    }

    /// Topocentric south / east / zenith.
    #[classattr]
    fn SEZ() -> Self {
        PyOrbitRelativeFrameKind { kind: frames::OrbitRelativeFrameKind::SEZ }
    }

    /// Velocity / normal / co-normal.
    #[classattr]
    fn VNC() -> Self {
        PyOrbitRelativeFrameKind { kind: frames::OrbitRelativeFrameKind::VNC }
    }

    /// Nadir / Sun / normal.
    #[classattr]
    fn NSW() -> Self {
        PyOrbitRelativeFrameKind { kind: frames::OrbitRelativeFrameKind::NSW }
    }

    fn __str__(&self) -> String {
        self.kind.to_string()
    }

    fn __repr__(&self) -> String {
        format!("OrbitRelativeFrameKind.{}", self.kind)
    }
}

/// Rotating vs. quasi-inertial snapshot variant of a local orbital frame.
///
/// - **Rotating**: True local orbital frame, rotating with the orbit. It
///   carries the orbital angular velocity, so a state transform through it
///   picks up the corresponding velocity transport term.
/// - **Inertial**: The same axes, taken as an instantaneous snapshot of the
///   orbit state at the evaluation epoch and then treated as non-rotating.
///   Its rate is zero, so a state transform through it applies no transport
///   term. The axes still differ from epoch to epoch, since each evaluation
///   takes a fresh snapshot.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     variant = bh.OrbitRelativeFrameVariant.ROTATING
///     ```
#[pyclass(module = "brahe._brahe", eq, from_py_object)]
#[pyo3(name = "OrbitRelativeFrameVariant")]
#[derive(Clone, Copy, PartialEq)]
pub struct PyOrbitRelativeFrameVariant {
    pub(crate) variant: frames::OrbitRelativeFrameVariant,
}

#[pymethods]
#[allow(non_snake_case)]
impl PyOrbitRelativeFrameVariant {
    /// True local orbital frame, rotating with the orbit.
    #[classattr]
    fn ROTATING() -> Self {
        PyOrbitRelativeFrameVariant { variant: frames::OrbitRelativeFrameVariant::Rotating }
    }

    /// Quasi-inertial snapshot: the orbit-relative axes at the evaluation
    /// epoch, with the frame's rate taken as zero.
    #[classattr]
    fn INERTIAL() -> Self {
        PyOrbitRelativeFrameVariant { variant: frames::OrbitRelativeFrameVariant::Inertial }
    }

    fn __str__(&self) -> String {
        self.variant.to_string()
    }

    fn __repr__(&self) -> String {
        format!("OrbitRelativeFrameVariant.{}", self.variant)
    }
}

/// An object-local spacecraft body frame (spacecraft body, sensor, or
/// actuator), with an optional instance designator (e.g. `CSS("1")`).
///
/// The families are the values of the SANA spacecraft body reference frame
/// registry: https://sanaregistry.org/r/spacecraft_body_reference_frames/
///
/// Bind to an object with `ReferenceFrame.body(object, body_frame)`.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bf = bh.BodyFrame.CSS("1")
///     frame = bh.ReferenceFrame.body("SC", bf)
///     ```
#[pyclass(module = "brahe._brahe", eq, from_py_object)]
#[pyo3(name = "BodyFrame")]
#[derive(Clone, PartialEq)]
pub struct PyBodyFrame {
    pub(crate) frame: frames::BodyFrame,
}

#[pymethods]
impl PyBodyFrame {
    /// Accelerometer frame.
    ///
    /// Args:
    ///     designator (str, optional): Sensor instance designator (e.g. "1")
    ///
    /// Returns:
    ///     BodyFrame: The `ACC` body frame
    #[staticmethod]
    #[pyo3(signature = (designator=None))]
    #[allow(non_snake_case)]
    fn ACC(designator: Option<String>) -> Self {
        PyBodyFrame { frame: frames::BodyFrame::ACC(designator) }
    }

    /// Actuator frame.
    ///
    /// Args:
    ///     designator (str, optional): Actuator instance designator (e.g. "1")
    ///
    /// Returns:
    ///     BodyFrame: The `ACTUATOR` body frame
    #[staticmethod]
    #[pyo3(signature = (designator=None))]
    #[allow(non_snake_case)]
    fn ACTUATOR(designator: Option<String>) -> Self {
        PyBodyFrame { frame: frames::BodyFrame::Actuator(designator) }
    }

    /// Autonomous star tracker frame.
    ///
    /// Args:
    ///     designator (str, optional): Sensor instance designator (e.g. "1")
    ///
    /// Returns:
    ///     BodyFrame: The `AST` body frame
    #[staticmethod]
    #[pyo3(signature = (designator=None))]
    #[allow(non_snake_case)]
    fn AST(designator: Option<String>) -> Self {
        PyBodyFrame { frame: frames::BodyFrame::AST(designator) }
    }

    /// Coarse sun sensor frame.
    ///
    /// Args:
    ///     designator (str, optional): Sensor instance designator (e.g. "1")
    ///
    /// Returns:
    ///     BodyFrame: The `CSS` body frame
    #[staticmethod]
    #[pyo3(signature = (designator=None))]
    #[allow(non_snake_case)]
    fn CSS(designator: Option<String>) -> Self {
        PyBodyFrame { frame: frames::BodyFrame::CSS(designator) }
    }

    /// Digital sun sensor frame.
    ///
    /// Args:
    ///     designator (str, optional): Sensor instance designator (e.g. "1")
    ///
    /// Returns:
    ///     BodyFrame: The `DSS` body frame
    #[staticmethod]
    #[pyo3(signature = (designator=None))]
    #[allow(non_snake_case)]
    fn DSS(designator: Option<String>) -> Self {
        PyBodyFrame { frame: frames::BodyFrame::DSS(designator) }
    }

    /// Earth sensor assembly frame.
    ///
    /// Args:
    ///     designator (str, optional): Sensor instance designator (e.g. "1")
    ///
    /// Returns:
    ///     BodyFrame: The `ESA` body frame
    #[staticmethod]
    #[pyo3(signature = (designator=None))]
    #[allow(non_snake_case)]
    fn ESA(designator: Option<String>) -> Self {
        PyBodyFrame { frame: frames::BodyFrame::ESA(designator) }
    }

    /// Gyroscope frame.
    ///
    /// Args:
    ///     designator (str, optional): Sensor instance designator (e.g. "1")
    ///
    /// Returns:
    ///     BodyFrame: The `GYRO_FRAME` body frame
    #[staticmethod]
    #[pyo3(signature = (designator=None))]
    #[allow(non_snake_case)]
    fn GYRO_FRAME(designator: Option<String>) -> Self {
        PyBodyFrame { frame: frames::BodyFrame::GyroFrame(designator) }
    }

    /// Inertial measurement unit frame.
    ///
    /// Args:
    ///     designator (str, optional): Sensor instance designator (e.g. "1")
    ///
    /// Returns:
    ///     BodyFrame: The `IMU_FRAME` body frame
    #[staticmethod]
    #[pyo3(signature = (designator=None))]
    #[allow(non_snake_case)]
    fn IMU_FRAME(designator: Option<String>) -> Self {
        PyBodyFrame { frame: frames::BodyFrame::IMUFrame(designator) }
    }

    /// Instrument frame.
    ///
    /// Args:
    ///     designator (str, optional): Instrument instance designator (e.g. "A")
    ///
    /// Returns:
    ///     BodyFrame: The `INSTRUMENT` body frame
    #[staticmethod]
    #[pyo3(signature = (designator=None))]
    #[allow(non_snake_case)]
    fn INSTRUMENT(designator: Option<String>) -> Self {
        PyBodyFrame { frame: frames::BodyFrame::Instrument(designator) }
    }

    /// Magnetic torque assembly frame.
    ///
    /// Args:
    ///     designator (str, optional): Actuator instance designator (e.g. "1")
    ///
    /// Returns:
    ///     BodyFrame: The `MTA` body frame
    #[staticmethod]
    #[pyo3(signature = (designator=None))]
    #[allow(non_snake_case)]
    fn MTA(designator: Option<String>) -> Self {
        PyBodyFrame { frame: frames::BodyFrame::MTA(designator) }
    }

    /// Reaction wheel frame.
    ///
    /// Args:
    ///     designator (str, optional): Actuator instance designator (e.g. "4")
    ///
    /// Returns:
    ///     BodyFrame: The `RW` body frame
    #[staticmethod]
    #[pyo3(signature = (designator=None))]
    #[allow(non_snake_case)]
    fn RW(designator: Option<String>) -> Self {
        PyBodyFrame { frame: frames::BodyFrame::RW(designator) }
    }

    /// Solar array frame.
    ///
    /// Args:
    ///     designator (str, optional): Array instance designator (e.g. "1")
    ///
    /// Returns:
    ///     BodyFrame: The `SA` body frame
    #[staticmethod]
    #[pyo3(signature = (designator=None))]
    #[allow(non_snake_case)]
    fn SA(designator: Option<String>) -> Self {
        PyBodyFrame { frame: frames::BodyFrame::SA(designator) }
    }

    /// Spacecraft body frame.
    ///
    /// Args:
    ///     designator (str, optional): Instance designator (e.g. "1")
    ///
    /// Returns:
    ///     BodyFrame: The `SC_BODY` body frame
    #[staticmethod]
    #[pyo3(signature = (designator=None))]
    #[allow(non_snake_case)]
    fn SC_BODY(designator: Option<String>) -> Self {
        PyBodyFrame { frame: frames::BodyFrame::SCBody(designator) }
    }

    /// Generic sensor frame.
    ///
    /// Args:
    ///     designator (str, optional): Sensor instance designator (e.g. "10")
    ///
    /// Returns:
    ///     BodyFrame: The `SENSOR` body frame
    #[staticmethod]
    #[pyo3(signature = (designator=None))]
    #[allow(non_snake_case)]
    fn SENSOR(designator: Option<String>) -> Self {
        PyBodyFrame { frame: frames::BodyFrame::Sensor(designator) }
    }

    /// Star tracker frame.
    ///
    /// Args:
    ///     designator (str, optional): Sensor instance designator (e.g. "2")
    ///
    /// Returns:
    ///     BodyFrame: The `STARTRACKER` body frame
    #[staticmethod]
    #[pyo3(signature = (designator=None))]
    #[allow(non_snake_case)]
    fn STARTRACKER(designator: Option<String>) -> Self {
        PyBodyFrame { frame: frames::BodyFrame::StarTracker(designator) }
    }

    /// Three-axis magnetometer frame.
    ///
    /// Args:
    ///     designator (str, optional): Sensor instance designator (e.g. "1")
    ///
    /// Returns:
    ///     BodyFrame: The `TAM` body frame
    #[staticmethod]
    #[pyo3(signature = (designator=None))]
    #[allow(non_snake_case)]
    fn TAM(designator: Option<String>) -> Self {
        PyBodyFrame { frame: frames::BodyFrame::TAM(designator) }
    }

    fn __str__(&self) -> String {
        self.frame.to_string()
    }

    fn __repr__(&self) -> String {
        format!("BodyFrame.{}", self.frame)
    }
}

/// Unified frame identity spanning celestial, orbit-relative, and body
/// frames.
///
/// Covers three kinds of frame: a `CelestialFrame` (see
/// `ReferenceFrame.celestial`), a local orbital frame of a specific
/// object (`RTN`, `LVLH`, ...), and an object-local body/sensor frame
/// (`SC_BODY`, `CSS`, ...). Orbit-relative and body frames carry an
/// optional bound object: a frame constructed through one of the family
/// staticmethods below (`ReferenceFrame.RTN("SC")`) is always bound; `ReferenceFrame.body`
/// with `object=None` and `ReferenceFrame.orbit_relative` with `object=None`
/// construct an unbound label.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     rtn = bh.ReferenceFrame.RTN("SC")
///     css = bh.ReferenceFrame.CSS("SC", "1")
///     print(rtn.is_bound(), rtn.object())
///     ```
#[pyclass(module = "brahe._brahe", eq, from_py_object)]
#[pyo3(name = "ReferenceFrame")]
#[derive(Clone, PartialEq)]
pub struct PyReferenceFrame {
    pub(crate) frame: frames::ReferenceFrame,
}

#[pymethods]
impl PyReferenceFrame {
    /// Wraps a `CelestialFrame` as a `ReferenceFrame`. Mirrors Rust's
    /// `From<CelestialFrame> for ReferenceFrame`.
    ///
    /// Args:
    ///     frame (CelestialFrame): The celestial frame to wrap
    ///
    /// Returns:
    ///     ReferenceFrame: The wrapped celestial frame
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     gcrf = bh.ReferenceFrame.celestial(bh.CelestialFrame.GCRF)
    ///     ```
    #[staticmethod]
    fn celestial(frame: PyCelestialFrame) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::Celestial(frame.frame) }
    }

    /// Bound Radial/Transverse/Normal orbit-relative frame (rotating
    /// variant). SANA: RSW.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `RTN (rotating)` orbit-relative frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn RTN(object: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::RTN(object) }
    }

    /// Bound Local-Vertical Local-Horizontal orbit-relative frame (rotating
    /// variant).
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today, so this frame is constructible but every transform through
    /// it raises until issue #452 adds the remaining derivations.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `LVLH (rotating)` orbit-relative frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn LVLH(object: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::LVLH(object) }
    }

    /// Bound Normal/Tangential/cross-track orbit-relative frame (rotating
    /// variant).
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today, so this frame is constructible but every transform through
    /// it raises until issue #452 adds the remaining derivations.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `NTW (rotating)` orbit-relative frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn NTW(object: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::NTW(object) }
    }

    /// Bound Tangential/Normal/cross-track orbit-relative frame (rotating
    /// variant).
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today, so this frame is constructible but every transform through
    /// it raises until issue #452 adds the remaining derivations.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `TNW (rotating)` orbit-relative frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn TNW(object: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::TNW(object) }
    }

    /// Bound topocentric South/East/Zenith orbit-relative frame (rotating
    /// variant).
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today, so this frame is constructible but every transform through
    /// it raises until issue #452 adds the remaining derivations.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `SEZ (rotating)` orbit-relative frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn SEZ(object: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::SEZ(object) }
    }

    /// Bound Velocity/Normal/Co-normal orbit-relative frame (rotating
    /// variant).
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today, so this frame is constructible but every transform through
    /// it raises until issue #452 adds the remaining derivations.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `VNC (rotating)` orbit-relative frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn VNC(object: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::VNC(object) }
    }

    /// Bound Nadir/Sun/Normal orbit-relative frame (rotating variant).
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today, so this frame is constructible but every transform through
    /// it raises until issue #452 adds the remaining derivations.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `NSW (rotating)` orbit-relative frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn NSW(object: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::NSW(object) }
    }

    /// Bound Perifocal orbit-relative frame (inertial-snapshot variant;
    /// `PQW` is SANA-registered only as inertial).
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today, so this frame is constructible but every transform through
    /// it raises until issue #452 adds the remaining derivations.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `PQW (inertial)` orbit-relative frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn PQW(object: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::PQW(object) }
    }

    /// Bound Equinoctial orbit-relative frame (inertial-snapshot variant;
    /// `EQW` is SANA-registered only as inertial).
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today, so this frame is constructible but every transform through
    /// it raises until issue #452 adds the remaining derivations.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `EQW (inertial)` orbit-relative frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn EQW(object: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::EQW(object) }
    }

    /// Bound spacecraft body frame (no instance designator).
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `SC_BODY` body frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn SC_BODY(object: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::SC_BODY(object) }
    }

    /// Bound coarse sun sensor frame.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///     designator (str): Sensor instance designator (e.g. "1")
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `CSS_<designator>` body frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn CSS(object: String, designator: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::CSS(object, designator) }
    }

    /// Bound accelerometer frame.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///     designator (str): Sensor instance designator (e.g. "1")
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `ACC_<designator>` body frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn ACC(object: String, designator: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::ACC(object, designator) }
    }

    /// Bound autonomous star tracker frame.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///     designator (str): Sensor instance designator (e.g. "1")
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `AST_<designator>` body frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn AST(object: String, designator: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::AST(object, designator) }
    }

    /// Bound digital sun sensor frame.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///     designator (str): Sensor instance designator (e.g. "1")
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `DSS_<designator>` body frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn DSS(object: String, designator: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::DSS(object, designator) }
    }

    /// Bound Earth sensor assembly frame.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///     designator (str): Sensor instance designator (e.g. "1")
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `ESA_<designator>` body frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn ESA(object: String, designator: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::ESA(object, designator) }
    }

    /// Bound gyroscope frame.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///     designator (str): Sensor instance designator (e.g. "1")
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `GYRO_FRAME_<designator>` body frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn GYRO_FRAME(object: String, designator: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::GYRO_FRAME(object, designator) }
    }

    /// Bound inertial measurement unit frame.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///     designator (str): Sensor instance designator (e.g. "1")
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `IMU_FRAME_<designator>` body frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn IMU_FRAME(object: String, designator: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::IMU_FRAME(object, designator) }
    }

    /// Bound instrument frame.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///     designator (str): Instrument instance designator (e.g. "A")
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `INSTRUMENT_<designator>` body frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn INSTRUMENT(object: String, designator: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::INSTRUMENT(object, designator) }
    }

    /// Bound magnetic torque assembly frame.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///     designator (str): Actuator instance designator (e.g. "1")
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `MTA_<designator>` body frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn MTA(object: String, designator: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::MTA(object, designator) }
    }

    /// Bound reaction wheel frame.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///     designator (str): Actuator instance designator (e.g. "4")
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `RW_<designator>` body frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn RW(object: String, designator: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::RW(object, designator) }
    }

    /// Bound solar array frame.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///     designator (str): Array instance designator (e.g. "1")
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `SA_<designator>` body frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn SA(object: String, designator: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::SA(object, designator) }
    }

    /// Bound generic sensor frame.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///     designator (str): Sensor instance designator (e.g. "10")
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `SENSOR_<designator>` body frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn SENSOR(object: String, designator: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::SENSOR(object, designator) }
    }

    /// Bound star tracker frame.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///     designator (str): Sensor instance designator (e.g. "2")
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `STARTRACKER_<designator>` body frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn STARTRACKER(object: String, designator: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::STARTRACKER(object, designator) }
    }

    /// Bound three-axis magnetometer frame.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///     designator (str): Sensor instance designator (e.g. "1")
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `TAM_<designator>` body frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn TAM(object: String, designator: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::TAM(object, designator) }
    }

    /// Bound actuator frame.
    ///
    /// Args:
    ///     object (str): The object the frame is defined relative to
    ///     designator (str): Actuator instance designator (e.g. "1")
    ///
    /// Returns:
    ///     ReferenceFrame: The bound `ACTUATOR_<designator>` body frame
    #[staticmethod]
    #[allow(non_snake_case)]
    fn ACTUATOR(object: String, designator: String) -> Self {
        PyReferenceFrame { frame: frames::ReferenceFrame::ACTUATOR(object, designator) }
    }

    /// Constructs a body frame, general form. Covers designator-less and
    /// non-standard `BodyFrame` cases beyond the family-specific
    /// staticmethods (e.g. `ReferenceFrame.CSS`, `ReferenceFrame.RW`).
    ///
    /// Args:
    ///     object (str, optional): The object the frame is defined relative to.
    ///         `None` constructs an unbound label frame.
    ///     body_frame (BodyFrame): The body frame kind and optional instance designator
    ///
    /// Returns:
    ///     ReferenceFrame: The body frame, bound to `object` if given
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     frame = bh.ReferenceFrame.body("SC", bh.BodyFrame.SC_BODY())
    ///     ```
    #[staticmethod]
    #[pyo3(signature = (object, body_frame))]
    fn body(object: Option<String>, body_frame: PyBodyFrame) -> Self {
        let frame = match object {
            Some(object) => frames::ReferenceFrame::body(object, body_frame.frame),
            None => frames::ReferenceFrame::from(body_frame.frame),
        };
        PyReferenceFrame { frame }
    }

    /// Constructs an orbit-relative frame, validating the kind/variant
    /// combination. General form of the family staticmethods (`ReferenceFrame.RTN`,
    /// ...), for callers that hold a runtime kind/variant pair and an
    /// optional, not-yet-bound object.
    ///
    /// Among the orbit-relative kinds only `RTN` has an axes derivation
    /// today; the others construct successfully but every transform through
    /// them raises until issue #452 adds the remaining derivations.
    ///
    /// Args:
    ///     kind (OrbitRelativeFrameKind): Frame construction (axes definition)
    ///     variant (OrbitRelativeFrameVariant): Rotating (true local orbital frame) or
    ///         inertial (quasi-inertial snapshot)
    ///     object (str, optional): The bound object, or `None` for an unbound label
    ///
    /// Returns:
    ///     ReferenceFrame: The orbit-relative frame
    ///
    /// Raises:
    ///     ValueError: If `kind` is `OrbitRelativeFrameKind.PQW`/`OrbitRelativeFrameKind.EQW`
    ///         and `variant` is `OrbitRelativeFrameVariant.ROTATING`
    #[staticmethod]
    #[pyo3(signature = (kind, variant, object=None))]
    fn orbit_relative(
        kind: PyOrbitRelativeFrameKind,
        variant: PyOrbitRelativeFrameVariant,
        object: Option<String>,
    ) -> PyResult<Self> {
        let frame =
            frames::ReferenceFrame::orbit_relative(kind.kind, variant.variant, object.map(Into::into))
                .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyReferenceFrame { frame })
    }

    /// Whether the frame is evaluable: a celestial frame, or an
    /// orbit-relative/body frame with a bound object.
    ///
    /// Returns:
    ///     bool: True if the frame is bound (celestial frames are always bound)
    fn is_bound(&self) -> bool {
        self.frame.is_bound()
    }

    /// The bound object, if any.
    ///
    /// Returns:
    ///     Optional[str]: The bound object, or `None` for a celestial frame
    ///         or an unbound orbit-relative/body frame
    fn object(&self) -> Option<String> {
        self.frame.object().map(|o| o.to_string())
    }

    fn __str__(&self) -> String {
        self.frame.to_string()
    }

    fn __repr__(&self) -> String {
        format!("ReferenceFrame(\"{}\")", self.frame)
    }
}

/// State provider backed by an `OrbitTrajectory`, whose native state is
/// dynamically sized: the leading six elements are the Cartesian position and
/// velocity, and any further elements are ignored.
struct PyTrajectoryStateProvider {
    trajectory: trajectories::DOrbitTrajectory,
}

impl SStateProvider for PyTrajectoryStateProvider {
    fn state(&self, epoch: time::Epoch) -> Result<Vector6<f64>, RustBraheError> {
        let state = DStateProvider::state(&self.trajectory, epoch)?;
        if state.len() < 6 {
            return Err(RustBraheError::Error(format!(
                "OrbitTrajectory returned a state of {} elements; at least 6 \
                 (position and velocity) are required",
                state.len()
            )));
        }
        Ok(Vector6::from_column_slice(&state.as_slice()[..6]))
    }
}

/// State provider backed by a Python callable `epoch -> length-6 ndarray`.
struct PyCallableStateProvider {
    callback: Py<PyAny>,
}

impl SStateProvider for PyCallableStateProvider {
    fn state(&self, epoch: time::Epoch) -> Result<Vector6<f64>, RustBraheError> {
        Python::attach(|py| {
            let py_epc = PyEpoch { obj: epoch };
            let result = self.callback.call1(py, (py_epc,)).map_err(|e| {
                RustBraheError::Error(format!(
                    "object state callback raised an exception: {e}"
                ))
            })?;
            pyany_to_svector::<6>(result.bind(py)).map_err(|e| {
                RustBraheError::Error(format!(
                    "object state callback must return a length-6 vector: {e}"
                ))
            })
        })
    }
}

/// Registers (or replaces) `frame`'s orientation relative to `parent`.
///
/// `frame` must be a bound `Body` frame (e.g. `ReferenceFrame.SC_BODY("SC")`,
/// `ReferenceFrame.CSS("SC", "1")`); `parent` must resolve to a celestial root by
/// walking the registry — either `parent` is itself a `CelestialFrame`, or
/// it is a bound `Body` frame that is already registered and whose own
/// parent chain terminates at one. `provider` is either a constant
/// attitude (`Quaternion`, `RotationMatrix`, `EulerAngle`, or `EulerAxis`)
/// or a callable `Epoch -> 3x3 ndarray` returning the parent -> frame
/// rotation matrix. `omega` and `numerical_rates_step` are meaningful only
/// for a callable `provider` (a constant attitude already has zero angular
/// velocity relative to its parent): `omega` is an optional callable
/// `Epoch -> length-3 ndarray` (rad/s, expressed in `frame`); when `omega`
/// is omitted, passing `numerical_rates_step` derives the angular velocity
/// numerically by central differencing the rotation over `±step/2` seconds.
///
/// Args:
///     frame (ReferenceFrame): The bound `Body` frame being registered
///     parent (CelestialFrame | ReferenceFrame): The frame `frame`'s orientation is expressed relative to
///     provider (Quaternion or RotationMatrix or EulerAngle or EulerAxis or callable): Supplies
///         `frame`'s rotation (and, for a callable, optionally its angular velocity) relative to `parent`
///     omega (callable, optional): Callable `Epoch -> length-3 ndarray` returning the frame's
///         angular velocity relative to `parent`, expressed in `frame` (rad/s)
///     numerical_rates_step (float, optional): Central-difference step (s) used to derive the
///         angular velocity numerically when `omega` is not given
///
/// Raises:
///     BraheError: If `frame` is not a bound `Body` frame, if the parent chain does not
///         terminate at a celestial frame, or if it cycles back through `frame`
///     TypeError: If `provider` is not a constant attitude or a callable, or if `omega` is given
///         and is not callable
///     ValueError: If `omega` or `numerical_rates_step` is given for a constant-attitude
///         `provider`, or if `numerical_rates_step` is not a positive finite number
///
/// Returns:
///     None: The frame is registered in the global frame registry
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.clear_frame_registry()
///     q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
///     bh.register_frame(bh.ReferenceFrame.SC_BODY("SC"), bh.CelestialFrame.GCRF, q)
///     bh.register_frame(bh.ReferenceFrame.CSS("SC", "1"), bh.ReferenceFrame.SC_BODY("SC"), q)
///     bh.clear_frame_registry()
///     ```
#[pyfunction]
#[pyo3(signature = (frame, parent, provider, omega=None, numerical_rates_step=None))]
#[pyo3(text_signature = "(frame, parent, provider, omega=None, numerical_rates_step=None)")]
#[pyo3(name = "register_frame")]
fn py_register_frame(
    frame: PyReferenceFrame,
    parent: &Bound<'_, PyAny>,
    provider: &Bound<'_, PyAny>,
    omega: Option<&Bound<'_, PyAny>>,
    numerical_rates_step: Option<f64>,
) -> PyResult<()> {
    let parent = extract_frame(parent)?;

    let is_constant_attitude = provider.extract::<PyQuaternion>().is_ok()
        || provider.extract::<PyRotationMatrix>().is_ok()
        || provider.extract::<PyEulerAngle>().is_ok()
        || provider.extract::<PyEulerAxis>().is_ok();
    if is_constant_attitude && (omega.is_some() || numerical_rates_step.is_some()) {
        return Err(exceptions::PyValueError::new_err(
            "omega and numerical_rates_step apply only to a callable provider; a constant \
             attitude already has zero angular velocity relative to its parent",
        ));
    }

    if let Ok(q) = provider.extract::<PyQuaternion>() {
        frames::register_frame(frame.frame, parent, q.obj)?;
        return Ok(());
    }
    if let Ok(r) = provider.extract::<PyRotationMatrix>() {
        frames::register_frame(frame.frame, parent, r.obj)?;
        return Ok(());
    }
    if let Ok(e) = provider.extract::<PyEulerAngle>() {
        frames::register_frame(frame.frame, parent, e.obj)?;
        return Ok(());
    }
    if let Ok(a) = provider.extract::<PyEulerAxis>() {
        frames::register_frame(frame.frame, parent, a.obj)?;
        return Ok(());
    }
    if !provider.is_callable() {
        return Err(exceptions::PyTypeError::new_err(
            "provider must be a Quaternion, RotationMatrix, EulerAngle, EulerAxis, or a \
             callable epoch -> 3x3 ndarray",
        ));
    }
    if let Some(omega) = omega
        && !omega.is_callable()
    {
        return Err(exceptions::PyTypeError::new_err(
            "omega must be a callable epoch -> length-3 ndarray",
        ));
    }
    let rotation_py: Py<PyAny> = provider.clone().unbind();
    let rotation_fn = move |epc: time::Epoch| -> Result<SMatrix3, RustBraheError> {
        Python::attach(|py| {
            let py_epc = PyEpoch { obj: epc };
            let result = rotation_py.call1(py, (py_epc,)).map_err(|e| {
                RustBraheError::Error(format!(
                    "frame rotation callback raised an exception: {e}"
                ))
            })?;
            pyany_to_smatrix::<3, 3>(result.bind(py)).map_err(|e| {
                RustBraheError::Error(format!(
                    "frame rotation callback must return a 3x3 matrix: {e}"
                ))
            })
        })
    };
    let omega_fn = omega.map(|omega| -> Box<frames::CustomFrameOmega> {
        let omega: Py<PyAny> = omega.clone().unbind();
        Box::new(move |epc: time::Epoch| {
            Python::attach(|py| {
                let py_epc = PyEpoch { obj: epc };
                let result = omega.call1(py, (py_epc,)).map_err(|e| {
                    RustBraheError::Error(format!(
                        "frame omega callback raised an exception: {e}"
                    ))
                })?;
                pyany_to_svector::<3>(result.bind(py)).map_err(|e| {
                    RustBraheError::Error(format!(
                        "frame omega callback must return a length-3 vector: {e}"
                    ))
                })
            })
        })
    });

    match numerical_rates_step {
        Some(step) => {
            let provider = frames::CallbackOrientation::new(rotation_fn, omega_fn)
                .with_numerical_rates(step)
                .map_err(|e| {
                    exceptions::PyValueError::new_err(format!("numerical_rates_step: {e}"))
                })?;
            frames::register_frame(frame.frame, parent, provider)?;
        }
        None => {
            let provider = frames::CallbackOrientation::new(rotation_fn, omega_fn);
            frames::register_frame(frame.frame, parent, provider)?;
        }
    }
    Ok(())
}

/// Removes the registered orientation of a bound `Body` frame.
///
/// Args:
///     frame (ReferenceFrame): The bound `Body` frame to unregister
///
/// Returns:
///     bool: True if `frame` was registered and has been removed
#[pyfunction]
#[pyo3(name = "unregister_frame")]
fn py_unregister_frame(frame: PyReferenceFrame) -> bool {
    frames::unregister_frame(&frame.frame)
}

/// Removes every entry from the frame registry, including entries
/// registered through `register_custom_frame`.
///
/// Intended for test isolation.
///
/// Returns:
///     None: Every frame registry entry is removed
#[pyfunction]
#[pyo3(name = "clear_frame_registry")]
fn py_clear_frame_registry() {
    frames::clear_frame_registry();
}

/// Registers (or replaces) `name`'s state provider.
///
/// `provider` is either a Cartesian `OrbitTrajectory` or a callable
/// `Epoch -> length-6 ndarray` returning `[position (m), velocity (m/s)]`
/// in `frame` axes/center. A trajectory whose state is dynamically sized is
/// adapted on the way in: its leading six elements are read as the position
/// and velocity, and any further elements are ignored.
///
/// A parsed CCSDS OEM registers in one call with `OEM.register_for(name)`,
/// which converts the ephemeris to a trajectory and registers it in the
/// frame the OEM declares. OMM and OPM carry elements rather than an
/// ephemeris, so they reach the registry through a propagator: an OMM's
/// mean elements build an `SGPPropagator` (`SGPPropagator.from_omm_elements`),
/// and an OPM's Cartesian state builds a `KeplerianPropagator`
/// (`KeplerianPropagator.from_eci`). Register the propagated state in the
/// frame named here, for example
/// `register_object(name, lambda epc: prop.state_gcrf(epc), CelestialFrame.GCRF)`.
///
/// Args:
///     name (str): The object's identity (e.g. "LRO", "2024-123A")
///     provider (OrbitTrajectory or callable): Supplies `name`'s state at arbitrary epochs, in `frame`
///     frame (CelestialFrame): The celestial frame `provider`'s states are expressed in
///
/// Raises:
///     TypeError: If `provider` is not an `OrbitTrajectory` or a callable
///     ValueError: If `provider` is an `OrbitTrajectory` using the Keplerian representation
///         rather than Cartesian position/velocity
///
/// Returns:
///     None: The object is registered in the global object registry
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.clear_object_registry()
///     bh.register_object("SC", lambda epc: [bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0], bh.CelestialFrame.GCRF)
///     bh.clear_object_registry()
///     ```
#[pyfunction]
#[pyo3(name = "register_object")]
fn py_register_object(
    name: String,
    provider: &Bound<'_, PyAny>,
    frame: PyCelestialFrame,
) -> PyResult<()> {
    if let Ok(traj) = provider.extract::<PyRef<'_, PyOrbitalTrajectory>>() {
        if traj.trajectory.representation != trajectories::traits::OrbitRepresentation::Cartesian
        {
            return Err(exceptions::PyValueError::new_err(
                "OrbitTrajectory provider must use the Cartesian representation \
                 (position/velocity), not Keplerian elements",
            ));
        }
        let provider = PyTrajectoryStateProvider {
            trajectory: traj.trajectory.clone(),
        };
        frames::register_object(name, provider, frame.frame)?;
        return Ok(());
    }
    if !provider.is_callable() {
        return Err(exceptions::PyTypeError::new_err(
            "provider must be an OrbitTrajectory or a callable epoch -> length-6 ndarray",
        ));
    }
    let callback: Py<PyAny> = provider.clone().unbind();
    frames::register_object(name, PyCallableStateProvider { callback }, frame.frame)?;
    Ok(())
}

/// Removes the registered provider for `name`.
///
/// Args:
///     name (str): The object's identity
///
/// Returns:
///     bool: True if `name` was registered and has been removed
#[pyfunction]
#[pyo3(name = "unregister_object")]
fn py_unregister_object(name: String) -> bool {
    frames::unregister_object(&name.into())
}

/// Removes every entry from the object registry.
///
/// Intended for test isolation.
///
/// Returns:
///     None: Every object registry entry is removed
#[pyfunction]
#[pyo3(name = "clear_object_registry")]
fn py_clear_object_registry() {
    frames::clear_object_registry();
}

/// Names of every registered object, sorted for stable output.
///
/// Returns:
///     List[str]: Registered object names, sorted lexicographically
#[pyfunction]
#[pyo3(name = "registered_objects")]
fn py_registered_objects() -> Vec<String> {
    frames::registered_objects()
        .iter()
        .map(|id| id.to_string())
        .collect()
}

/// Registers `name` as a SPICE state provider for `naif_id`, in the GCRF
/// frame.
///
/// Convenience wrapper equivalent to
/// `register_object(name, SPKStateProvider(naif_id), CelestialFrame.GCRF)`.
///
/// Args:
///     name (str): The object's identity to register
///     naif_id (int): NAIF ID of the body to query from loaded SPICE kernels
///
/// Returns:
///     None: The object is registered in the global object registry
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.clear_object_registry()
///     bh.register_object_from_naif("MOON", 301)
///     bh.clear_object_registry()
///     ```
#[pyfunction]
#[pyo3(name = "register_object_from_naif")]
fn py_register_object_from_naif(name: String, naif_id: i32) -> PyResult<()> {
    frames::register_object_from_naif(name, naif_id)?;
    Ok(())
}