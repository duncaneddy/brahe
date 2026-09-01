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
///     x_eci (numpy.ndarray or list): 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming from RTN to ECI frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
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
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "rotation_rtn_to_eci")]
fn py_rotation_rtn_to_eci<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_eci,
        axis,
        relative_motion::rotation_rtn_to_eci,
        relative_motion::rotations_rtn_to_eci,
    )
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Inertial (ECI)
/// frame to the radial, along-track, cross-track (RTN) frame.
///
/// This is the transpose (inverse) of the RTN-to-ECI rotation matrix.
///
/// Args:
///     x_eci (numpy.ndarray or list): 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming from ECI to RTN frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
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
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "rotation_eci_to_rtn")]
fn py_rotation_eci_to_rtn<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_eci,
        axis,
        relative_motion::rotation_eci_to_rtn,
        relative_motion::rotations_eci_to_rtn,
    )
}

/// Computes the angular velocity of the radial, along-track, cross-track (RTN) frame with
/// respect to the Earth-Centered Inertial (ECI) frame, expressed in RTN axes.
///
/// The RTN frame rotates about its cross-track axis at the orbital true-anomaly rate
/// `f_dot = |r x v| / r^2`, so the angular velocity is `[0, 0, f_dot]`.
///
/// The returned vector's components are those of the RTN frame, not the ECI
/// frame: the third component is the rate about the cross-track axis.
/// Multiply by `rotation_rtn_to_eci` to express the same angular velocity in
/// ECI axes.
///
/// Args:
///     x_eci (numpy.ndarray or list): 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,)
///
/// Returns:
///     numpy.ndarray: Angular velocity of the RTN frame relative to ECI, expressed in RTN axes (rad/s), shape (3,)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     sma = bh.R_EARTH + 700e3
///     state = np.array([sma, 0.0, 0.0, 0.0, bh.perigee_velocity(sma, 0.0), 0.0])
///     omega = bh.omega_rtn(state)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(x_eci)")]
#[pyo3(name = "omega_rtn")]
fn py_omega_rtn<'py>(py: Python<'py>, x_eci: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let x = pyany_to_svector::<6>(x_eci)?;
    let omega = relative_motion::omega_rtn(x);
    Ok(vector_to_numpy!(py, omega, 3, f64))
}

/// Transforms the absolute states of a chief and deputy satellite from the Earth-Centered Inertial (ECI)
/// frame to the relative state of the deputy with respect to the chief in the rotating
/// Radial, Along-Track, Cross-Track (RTN) frame.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_deputy (numpy.ndarray or list): 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: 6D relative state vector of the deputy with respect to the chief in the RTN frame [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
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
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///     x_deputy = bh.state_koe_to_eci(oe_deputy, bh.AngleFormat.DEGREES)
///
///     # Transform to relative RTN state
///     x_rel_rtn = bh.state_eci_to_rtn(x_chief, x_deputy)
///     print(f"Relative state in RTN: {x_rel_rtn}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_deputy, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_deputy, axis=-1)")]
#[pyo3(name = "state_eci_to_rtn")]
fn py_state_eci_to_rtn<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_deputy: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        x_deputy,
        axis,
        relative_motion::state_eci_to_rtn,
        relative_motion::states_eci_to_rtn,
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite
/// from the rotating Radial, Along-Track, Cross-Track (RTN) frame to the absolute state
/// of the deputy in the Earth-Centered Inertial (ECI) frame.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_rel_rtn (numpy.ndarray or list): 6D relative state vector of the deputy with respect to the chief in the RTN frame [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
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
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///
///     # Relative state: 1km radial, 0.5km along-track, -0.3km cross-track
///     x_rel_rtn = np.array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0])
///
///     # Transform to absolute deputy ECI state
///     x_deputy = bh.state_rtn_to_eci(x_chief, x_rel_rtn)
///     print(f"Deputy state in ECI: {x_deputy}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_rel_rtn, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_rel_rtn, axis=-1)")]
#[pyo3(name = "state_rtn_to_eci")]
fn py_state_rtn_to_eci<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_rel_rtn: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        x_rel_rtn,
        axis,
        relative_motion::state_rtn_to_eci,
        relative_motion::states_rtn_to_eci,
    )
}

/// Converts chief and deputy satellite orbital elements (OE) to quasi-nonsingular relative orbital elements (ROE).
///
/// The ROE formulation provides a mean description of relative motion that is nonsingular for
/// circular and near-circular orbits. The ROE vector contains:
/// - da: Relative semi-major axis (dimensionless)
/// - dλ: Relative mean longitude (degrees or radians)
/// - dex: x-component of relative eccentricity vector (dimensionless)
/// - dey: y-component of relative eccentricity vector (dimensionless)
/// - dix: x-component of relative inclination vector (degrees or radians)
/// - diy: y-component of relative inclination vector (degrees or radians)
///
/// Args:
///     oe_chief (numpy.ndarray or list): Chief satellite orbital elements [a, e, i, Ω, ω, M] shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     oe_deputy (numpy.ndarray or list): Deputy satellite orbital elements [a, e, i, Ω, ω, M] shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     angle_format (AngleFormat): Format of angular elements (DEGREES or RADIANS)
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`. A single vector in one argument is
///         broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: Relative orbital elements [da, dλ, dex, dey, dix, diy] shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Define chief and deputy orbital elements (degrees)
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
///     oe_deputy = np.array([bh.R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])
///
///     # Convert to ROE
///     roe = bh.state_oe_to_roe(oe_chief, oe_deputy, bh.AngleFormat.DEGREES)
///     print(f"Relative orbital elements: {roe}")
///     # Relative orbital elements: [1.413e-4, 9.321e-2, 4.324e-4, 2.511e-4, 5.0e-2, 4.954e-2]
///     ```
#[pyfunction]
#[pyo3(signature = (oe_chief, oe_deputy, angle_format, axis=-1))]
#[pyo3(text_signature = "(oe_chief, oe_deputy, angle_format, axis=-1)")]
#[pyo3(name = "state_oe_to_roe")]
fn py_state_oe_to_roe<'py>(
    py: Python<'py>,
    oe_chief: &Bound<'py, PyAny>,
    oe_deputy: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec_pair::<6>(
        py,
        oe_chief,
        oe_deputy,
        axis,
        |c, d| relative_motion::state_oe_to_roe(c, d, af),
        |cs, ds| relative_motion::states_oe_to_roe(cs, ds, af),
    )
}

/// Converts chief satellite orbital elements (OE) and quasi-nonsingular relative orbital elements (ROE)
/// to deputy satellite orbital elements.
///
/// This is the inverse transformation of `state_oe_to_roe`, converting from ROE representation
/// back to classical orbital elements for the deputy satellite.
///
/// Args:
///     oe_chief (numpy.ndarray or list): Chief satellite orbital elements [a, e, i, Ω, ω, M] shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     roe (numpy.ndarray or list): Relative orbital elements [da, dλ, dex, dey, dix, diy] shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     angle_format (AngleFormat): Format of angular elements (DEGREES or RADIANS)
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`. A single vector in one argument is
///         broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: Deputy satellite orbital elements [a, e, i, Ω, ω, M] shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Define chief orbital elements and ROE (degrees)
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
///     roe = np.array([1.413e-4, 9.321e-2, 4.324e-4, 2.511e-4, 5.0e-2, 4.954e-2])
///
///     # Convert to deputy OE
///     oe_deputy = bh.state_roe_to_oe(oe_chief, roe, bh.AngleFormat.DEGREES)
///     print(f"Deputy orbital elements: {oe_deputy}")
///     # Deputy orbital elements: [7.079e6, 1.5e-3, 97.85, 15.05, 30.05, 45.05]
///     ```
#[pyfunction]
#[pyo3(signature = (oe_chief, roe, angle_format, axis=-1))]
#[pyo3(text_signature = "(oe_chief, roe, angle_format, axis=-1)")]
#[pyo3(name = "state_roe_to_oe")]
fn py_state_roe_to_oe<'py>(
    py: Python<'py>,
    oe_chief: &Bound<'py, PyAny>,
    roe: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec_pair::<6>(
        py,
        oe_chief,
        roe,
        axis,
        |c, d| relative_motion::state_roe_to_oe(c, d, af),
        |cs, ds| relative_motion::states_roe_to_oe(cs, ds, af),
    )
}

/// Converts chief and deputy satellite ECI state vectors to quasi-nonsingular Relative Orbital Elements (ROE).
///
/// This function converts both ECI states to Keplerian orbital elements, then computes
/// the quasi-nonsingular Relative Orbital Elements between them.
///
/// The ROE formulation provides a mean description of relative motion that is nonsingular for
/// circular and near-circular orbits. The ROE vector contains:
/// - da: Relative semi-major axis (dimensionless)
/// - dλ: Relative mean longitude (degrees or radians)
/// - dex: x-component of relative eccentricity vector (dimensionless)
/// - dey: y-component of relative eccentricity vector (dimensionless)
/// - dix: x-component of relative inclination vector (degrees or radians)
/// - diy: y-component of relative inclination vector (degrees or radians)
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D ECI state vector of the chief satellite [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_deputy (numpy.ndarray or list): 6D ECI state vector of the deputy satellite [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     angle_format (AngleFormat): Format of angular elements in output (DEGREES or RADIANS)
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`. A single vector in one argument is
///         broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: Relative orbital elements [da, dλ, dex, dey, dix, diy] shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
///
///     # Define chief and deputy orbital elements (degrees)
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
///     oe_deputy = np.array([bh.R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])
///
///     # Convert to ECI states
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///     x_deputy = bh.state_koe_to_eci(oe_deputy, bh.AngleFormat.DEGREES)
///
///     # Compute ROE directly from ECI states
///     roe = bh.state_eci_to_roe(x_chief, x_deputy, bh.AngleFormat.DEGREES)
///     print(f"Relative orbital elements: {roe}")
///     # Relative orbital elements: [1.413e-4, 9.321e-2, 4.324e-4, 2.511e-4, 5.0e-2, 4.954e-2]
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_deputy, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_deputy, angle_format, axis=-1)")]
#[pyo3(name = "state_eci_to_roe")]
fn py_state_eci_to_roe<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_deputy: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        x_deputy,
        axis,
        |c, d| relative_motion::state_eci_to_roe(c, d, af),
        |cs, ds| relative_motion::states_eci_to_roe(cs, ds, af),
    )
}

/// Converts chief satellite ECI state and quasi-nonsingular Relative Orbital Elements (ROE)
/// to deputy satellite ECI state.
///
/// This function converts the chief ECI state to Keplerian orbital elements, applies
/// the ROE to obtain deputy orbital elements, then converts back to ECI state.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D ECI state vector of the chief satellite [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     roe (numpy.ndarray or list): Relative orbital elements [da, dλ, dex, dey, dix, diy] shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     angle_format (AngleFormat): Format of angular elements in input ROE (DEGREES or RADIANS)
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`. A single vector in one argument is
///         broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: 6D ECI state vector of the deputy satellite [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
///
///     # Define chief orbital elements and convert to ECI
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///
///     # Define ROE (small relative orbit)
///     roe = np.array([1.413e-4, 9.321e-2, 4.324e-4, 2.511e-4, 5.0e-2, 4.954e-2])
///
///     # Compute deputy ECI state from chief and ROE
///     x_deputy = bh.state_roe_to_eci(x_chief, roe, bh.AngleFormat.DEGREES)
///     print(f"Deputy ECI state: {x_deputy}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, roe, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_chief, roe, angle_format, axis=-1)")]
#[pyo3(name = "state_roe_to_eci")]
fn py_state_roe_to_eci<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    roe: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        roe,
        axis,
        |c, d| relative_motion::state_roe_to_eci(c, d, af),
        |cs, ds| relative_motion::states_roe_to_eci(cs, ds, af),
    )
}
