/// Computes the rotation matrix transforming a vector in the radial, along-track, cross-track (RTN)
/// frame to the Earth-Centered Inertial (ECI) frame.
///
/// The ECI frame can be any inertial frame centered at the Earth's center, such as GCRF or EME2000.
///
/// The RTN frame is defined as follows:
/// - R (Radial): Points from the Earth's center to the satellite's position.
/// - N (Cross-Track): Perpendicular to the orbital plane, defined by the angular momentum vector (cross product of position and velocity).
/// - T (Along-Track): Completes the right-handed coordinate system, lying in the orbital plane and perpendicular to R and N.
///
/// Args:
///     x_eci (numpy.ndarray or list): 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,)
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming from RTN to ECI frame, shape (3, 3)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Define satellite state
///     sma = bh.R_EARTH + 700e3  # Semi-major axis in meters
///     state = np.array([sma, 0.0, 0.0, 0.0, bh.perigee_velocity(sma, 0.0), 0.0])
///
///     # Get rotation matrix
///     R = bh.rotation_rtn_to_eci(state)
///     print(f"RTN to ECI rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(x_eci)")]
#[pyo3(name = "rotation_rtn_to_eci")]
unsafe fn py_rotation_rtn_to_eci<'py>(
    py: Python<'py>,
    x_eci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let mat = relative_motion::rotation_rtn_to_eci(pyany_to_svector::<6>(&x_eci)?);
    Ok(matrix_to_numpy!(py, mat, 3, 3, f64))
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Inertial (ECI)
/// frame to the radial, along-track, cross-track (RTN) frame.
///
/// This is the transpose (inverse) of the RTN-to-ECI rotation matrix.
///
/// Args:
///     x_eci (numpy.ndarray or list): 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,)
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming from ECI to RTN frame, shape (3, 3)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Define satellite state
///     sma = bh.R_EARTH + 700e3  # Semi-major axis in meters
///     state = np.array([sma, 0.0, 0.0, 0.0, bh.perigee_velocity(sma, 0.0), 0.0])
///
///     # Get rotation matrix
///     R = bh.rotation_eci_to_rtn(state)
///     print(f"ECI to RTN rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(x_eci)")]
#[pyo3(name = "rotation_eci_to_rtn")]
unsafe fn py_rotation_eci_to_rtn<'py>(
    py: Python<'py>,
    x_eci: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let mat = relative_motion::rotation_eci_to_rtn(pyany_to_svector::<6>(&x_eci)?);
    Ok(matrix_to_numpy!(py, mat, 3, 3, f64))
}

/// Transforms the absolute states of a chief and deputy satellite from the Earth-Centered Inertial (ECI)
/// frame to the relative state of the deputy with respect to the chief in the rotating
/// Radial, Along-Track, Cross-Track (RTN) frame.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,)
///     x_deputy (numpy.ndarray or list): 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,)
///
/// Returns:
///     numpy.ndarray: 6D relative state vector of the deputy with respect to the chief in the RTN frame [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N] (m, m/s), shape (6,)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
///
///     # Define chief and deputy orbital elements
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
///     oe_deputy = np.array([bh.R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])
///
///     # Convert to Cartesian states
///     x_chief = bh.state_osculating_to_cartesian(oe_chief, bh.AngleFormat.DEGREES)
///     x_deputy = bh.state_osculating_to_cartesian(oe_deputy, bh.AngleFormat.DEGREES)
///
///     # Transform to relative RTN state
///     x_rel_rtn = bh.state_eci_to_rtn(x_chief, x_deputy)
///     print(f"Relative state in RTN: {x_rel_rtn}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(x_chief, x_deputy)")]
#[pyo3(name = "state_eci_to_rtn")]
unsafe fn py_state_eci_to_rtn<'py>(
    py: Python<'py>,
    x_chief: Bound<'py, PyAny>,
    x_deputy: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let result = relative_motion::state_eci_to_rtn(
        pyany_to_svector::<6>(&x_chief)?,
        pyany_to_svector::<6>(&x_deputy)?,
    );
    Ok(vector_to_numpy!(py, result, 6, f64))
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite
/// from the rotating Radial, Along-Track, Cross-Track (RTN) frame to the absolute state
/// of the deputy in the Earth-Centered Inertial (ECI) frame.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,)
///     x_rel_rtn (numpy.ndarray or list): 6D relative state vector of the deputy with respect to the chief in the RTN frame [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N] (m, m/s), shape (6,)
///
/// Returns:
///     numpy.ndarray: 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
///
///     # Define chief state and relative RTN state
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
///     x_chief = bh.state_osculating_to_cartesian(oe_chief, bh.AngleFormat.DEGREES)
///
///     # Relative state: 1km radial, 0.5km along-track, -0.3km cross-track
///     x_rel_rtn = np.array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0])
///
///     # Transform to absolute deputy ECI state
///     x_deputy = bh.state_rtn_to_eci(x_chief, x_rel_rtn)
///     print(f"Deputy state in ECI: {x_deputy}")
///     ```
#[pyfunction]
#[pyo3(text_signature = "(x_chief, x_rel_rtn)")]
#[pyo3(name = "state_rtn_to_eci")]
unsafe fn py_state_rtn_to_eci<'py>(
    py: Python<'py>,
    x_chief: Bound<'py, PyAny>,
    x_rel_rtn: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let result = relative_motion::state_rtn_to_eci(
        pyany_to_svector::<6>(&x_chief)?,
        pyany_to_svector::<6>(&x_rel_rtn)?,
    );
    Ok(vector_to_numpy!(py, result, 6, f64))
}
