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
///     x_eci (numpy.ndarray or list): 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Angular velocity of the RTN frame relative to ECI, expressed in RTN axes (rad/s), shape (3,), or the batch dimensions
///         with 3 components along `axis` for batched input.
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
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "omega_rtn")]
fn py_omega_rtn<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_map::<6, 3>(py, x_eci, axis, relative_motion::omega_rtn, relative_motion::omegas_rtn)
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

/// 6x6 Jacobian taking an RTN state covariance into ECI axes.
///
/// The RTN-to-ECI state map is `r_eci = R @ rho` and
/// `v_eci = R @ (rho_dot + omega x rho)`, with `R` the RTN-to-ECI rotation
/// and `omega` the RTN frame's angular velocity in RTN components. Its
/// Jacobian is `[[R, 0], [R @ skew(omega), R]]`. The `INERTIAL` variant
/// freezes the axes at the evaluation epoch, taking `omega = 0`, and so gives
/// the plain block diagonal; `ROTATING` carries the coupling term.
///
/// Args:
///     x_eci (numpy.ndarray or list): 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,)
///     variant (OrbitRelativeFrameVariant): Whether the RTN axes rotate with the orbit or are frozen at the epoch
///
/// Returns:
///     numpy.ndarray: 6x6 Jacobian such that `P_eci = J @ P_rtn @ J.T`, shape (6, 6)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     sma = bh.R_EARTH + 700e3
///     x_eci = np.array([sma, 0.0, 0.0, 0.0, bh.perigee_velocity(sma, 0.0), 0.0])
///     j = bh.jacobian_rtn_to_eci(x_eci, bh.OrbitRelativeFrameVariant.ROTATING)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(x_eci, variant)")]
#[pyo3(name = "jacobian_rtn_to_eci")]
fn py_jacobian_rtn_to_eci<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    variant: &PyOrbitRelativeFrameVariant,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let x = pyany_to_svector::<6>(x_eci)?;
    let j = relative_motion::jacobian_rtn_to_eci(x, variant.variant);
    Ok(matrix_to_numpy!(py, j, 6, 6, f64).to_owned())
}

/// 6x6 Jacobian taking an ECI state covariance into RTN axes.
///
/// Exact inverse of `jacobian_rtn_to_eci`: with `R.T` the ECI-to-RTN
/// rotation, the Jacobian is `[[R.T, 0], [-skew(omega) @ R.T, R.T]]`.
///
/// Args:
///     x_eci (numpy.ndarray or list): 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,)
///     variant (OrbitRelativeFrameVariant): Whether the RTN axes rotate with the orbit or are frozen at the epoch
///
/// Returns:
///     numpy.ndarray: 6x6 Jacobian such that `P_rtn = J @ P_eci @ J.T`, shape (6, 6)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     sma = bh.R_EARTH + 700e3
///     x_eci = np.array([sma, 0.0, 0.0, 0.0, bh.perigee_velocity(sma, 0.0), 0.0])
///     j = bh.jacobian_eci_to_rtn(x_eci, bh.OrbitRelativeFrameVariant.ROTATING)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(x_eci, variant)")]
#[pyo3(name = "jacobian_eci_to_rtn")]
fn py_jacobian_eci_to_rtn<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    variant: &PyOrbitRelativeFrameVariant,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let x = pyany_to_svector::<6>(x_eci)?;
    let j = relative_motion::jacobian_eci_to_rtn(x, variant.variant);
    Ok(matrix_to_numpy!(py, j, 6, 6, f64).to_owned())
}

/// Transforms a 6x6 state covariance from RTN axes into ECI axes.
///
/// Applies the congruence `P_eci = J @ P_rtn @ J.T` with `J` from
/// `jacobian_rtn_to_eci`, and symmetrizes the result.
///
/// Args:
///     x_eci (numpy.ndarray or list): 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,)
///     covariance (numpy.ndarray): 6x6 state covariance in RTN axes, shape (6, 6)
///     variant (OrbitRelativeFrameVariant): Whether the RTN axes rotate with the orbit or are frozen at the epoch
///
/// Returns:
///     numpy.ndarray: 6x6 state covariance in ECI axes, shape (6, 6)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     sma = bh.R_EARTH + 700e3
///     x_eci = np.array([sma, 0.0, 0.0, 0.0, bh.perigee_velocity(sma, 0.0), 0.0])
///     p_eci = bh.covariance_rtn_to_eci(
///         x_eci, np.eye(6) * 100.0, bh.OrbitRelativeFrameVariant.ROTATING
///     )
///     ```
#[pyfunction]
#[pyo3(text_signature = "(x_eci, covariance, variant)")]
#[pyo3(name = "covariance_rtn_to_eci")]
fn py_covariance_rtn_to_eci<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    covariance: &Bound<'py, PyAny>,
    variant: &PyOrbitRelativeFrameVariant,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let x = pyany_to_svector::<6>(x_eci)?;
    let p = pyany_to_smatrix::<6, 6>(covariance)?;
    let rotated = relative_motion::covariance_rtn_to_eci(x, &p, variant.variant);
    Ok(matrix_to_numpy!(py, rotated, 6, 6, f64).to_owned())
}

/// Transforms a 6x6 state covariance from ECI axes into RTN axes.
///
/// Applies the congruence `P_rtn = J @ P_eci @ J.T` with `J` from
/// `jacobian_eci_to_rtn`, and symmetrizes the result.
///
/// Args:
///     x_eci (numpy.ndarray or list): 6D state vector of the frame's origin in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,)
///     covariance (numpy.ndarray): 6x6 state covariance in ECI axes, shape (6, 6)
///     variant (OrbitRelativeFrameVariant): Whether the RTN axes rotate with the orbit or are frozen at the epoch
///
/// Returns:
///     numpy.ndarray: 6x6 state covariance in RTN axes, shape (6, 6)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     sma = bh.R_EARTH + 700e3
///     x_eci = np.array([sma, 0.0, 0.0, 0.0, bh.perigee_velocity(sma, 0.0), 0.0])
///     p_rtn = bh.covariance_eci_to_rtn(
///         x_eci, np.eye(6) * 100.0, bh.OrbitRelativeFrameVariant.ROTATING
///     )
///     ```
#[pyfunction]
#[pyo3(text_signature = "(x_eci, covariance, variant)")]
#[pyo3(name = "covariance_eci_to_rtn")]
fn py_covariance_eci_to_rtn<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    covariance: &Bound<'py, PyAny>,
    variant: &PyOrbitRelativeFrameVariant,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let x = pyany_to_svector::<6>(x_eci)?;
    let p = pyany_to_smatrix::<6, 6>(covariance)?;
    let rotated = relative_motion::covariance_eci_to_rtn(x, &p, variant.variant);
    Ok(matrix_to_numpy!(py, rotated, 6, 6, f64).to_owned())
}

/// Computes the rotation matrix transforming a vector in the Local-Vertical Local-Horizontal
/// (LVLH) frame to the Earth-Centered Inertial (ECI) frame.
///
/// The ECI frame can be any inertial frame centered on the orbited body, such as GCRF or EME2000.
///
/// The LVLH frame follows the CCSDS and SANA definition:
/// - Z: Unit vector collinear with and opposite to the position vector (nadir).
/// - Y: Unit vector collinear with and opposite to the orbital angular momentum `r x v`.
/// - X: `Y x Z`, completing the right-handed set (along-track for a circular orbit).
///
/// This is a signed permutation of the RTN axes: `X = T`, `Y = -N`, `Z = -R`. Vallado and
/// STK use the name LVLH for the RTN axes themselves; brahe follows the CCSDS convention.
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
///     numpy.ndarray: 3x3 rotation matrix transforming from LVLH to ECI frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     x_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
///
///     R = bh.rotation_lvlh_to_eci(x_eci)
///     print(f"LVLH to ECI rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "rotation_lvlh_to_eci")]
fn py_rotation_lvlh_to_eci<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_eci,
        axis,
        relative_motion::rotation_lvlh_to_eci,
        relative_motion::rotations_lvlh_to_eci,
    )
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Inertial (ECI)
/// frame to the Local-Vertical Local-Horizontal (LVLH) frame.
///
/// This is the transpose (inverse) of the LVLH-to-ECI rotation matrix.
///
/// The LVLH frame follows the CCSDS and SANA definition:
/// - Z: Unit vector collinear with and opposite to the position vector (nadir).
/// - Y: Unit vector collinear with and opposite to the orbital angular momentum `r x v`.
/// - X: `Y x Z`, completing the right-handed set (along-track for a circular orbit).
///
/// This is a signed permutation of the RTN axes: `X = T`, `Y = -N`, `Z = -R`. Vallado and
/// STK use the name LVLH for the RTN axes themselves; brahe follows the CCSDS convention.
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
///     numpy.ndarray: 3x3 rotation matrix transforming from ECI to LVLH frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     x_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
///
///     R = bh.rotation_eci_to_lvlh(x_eci)
///     print(f"ECI to LVLH rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "rotation_eci_to_lvlh")]
fn py_rotation_eci_to_lvlh<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_eci,
        axis,
        relative_motion::rotation_eci_to_lvlh,
        relative_motion::rotations_eci_to_lvlh,
    )
}

/// Computes the angular velocity of the Local-Vertical Local-Horizontal (LVLH) frame with
/// respect to the Earth-Centered Inertial (ECI) frame, expressed in LVLH axes.
///
/// The LVLH frame rotates about the orbit normal at the true-anomaly rate `f_dot = |r x v| / r^2`.
/// The orbit normal is the negative LVLH Y axis, so the angular velocity is `[0, -f_dot, 0]`.
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
///     numpy.ndarray: Angular velocity of the LVLH frame relative to ECI, expressed in LVLH axes (rad/s), shape (3,), or the batch dimensions
///         with 3 components along `axis` for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     x_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
///     omega = bh.omega_lvlh(x_eci)
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "omega_lvlh")]
fn py_omega_lvlh<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_map::<6, 3>(py, x_eci, axis, relative_motion::omega_lvlh, relative_motion::omegas_lvlh)
}

/// Transforms the absolute states of a chief and deputy satellite from the Earth-Centered
/// Inertial (ECI) frame to the relative state of the deputy with respect to the chief in the
/// rotating Local-Vertical Local-Horizontal (LVLH) frame.
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
///     numpy.ndarray: 6D relative state vector of the deputy with respect to the chief in the LVLH frame [ρ_X, ρ_Y, ρ_Z, ρ̇_X, ρ̇_Y, ρ̇_Z] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
///
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     oe_deputy = np.array([bh.R_EARTH + 701e3, 0.0115, 97.85, 15.05, 30.05, 45.05])
///
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///     x_deputy = bh.state_koe_to_eci(oe_deputy, bh.AngleFormat.DEGREES)
///
///     x_rel_lvlh = bh.state_eci_to_lvlh(x_chief, x_deputy)
///     print(f"Relative state in LVLH: {x_rel_lvlh}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_deputy, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_deputy, axis=-1)")]
#[pyo3(name = "state_eci_to_lvlh")]
fn py_state_eci_to_lvlh<'py>(
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
        relative_motion::state_eci_to_lvlh,
        relative_motion::states_eci_to_lvlh,
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite
/// from the rotating Local-Vertical Local-Horizontal (LVLH) frame to the absolute state of
/// the deputy in the Earth-Centered Inertial (ECI) frame.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_rel_lvlh (numpy.ndarray or list): 6D relative state vector of the deputy with respect to the chief in the LVLH frame [ρ_X, ρ_Y, ρ_Z, ρ̇_X, ρ̇_Y, ρ̇_Z] (m, m/s), shape (6,), or a batch of
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
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///
///     # Relative state: 1km X, 0.5km Y, -0.3km Z
///     x_rel_lvlh = np.array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0])
///
///     x_deputy = bh.state_lvlh_to_eci(x_chief, x_rel_lvlh)
///     print(f"Deputy state in ECI: {x_deputy}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_rel_lvlh, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_rel_lvlh, axis=-1)")]
#[pyo3(name = "state_lvlh_to_eci")]
fn py_state_lvlh_to_eci<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_rel_lvlh: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        x_rel_lvlh,
        axis,
        relative_motion::state_lvlh_to_eci,
        relative_motion::states_lvlh_to_eci,
    )
}

/// Computes the rotation matrix transforming a vector in the Normal, Tangential, Cross-track
/// (NTW) frame to the Earth-Centered Inertial (ECI) frame.
///
/// The ECI frame can be any inertial frame centered on the orbited body, such as GCRF or EME2000.
///
/// The NTW frame follows the SANA definition:
/// - Y (T): Unit vector along the inertial velocity.
/// - Z (W): Unit vector along the orbital angular momentum `r x v`.
/// - X (N): `Y x Z`, completing the right-handed set; in the orbit plane, normal to the
///   velocity, pointing outward (radial for a circular orbit).
///
/// On a circular orbit NTW coincides with RTN; on an eccentric orbit the two differ by the
/// flight-path angle. TNW and VNC use the same three directions reordered with a sign change:
/// `TNW = [Y_NTW, -X_NTW, Z_NTW]` and `VNC = [Y_NTW, Z_NTW, X_NTW]`.
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
///     numpy.ndarray: 3x3 rotation matrix transforming from NTW to ECI frame, shape (3, 3), or the batch dimensions
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
///     R = bh.rotation_ntw_to_eci(state)
///     print(f"NTW to ECI rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "rotation_ntw_to_eci")]
fn py_rotation_ntw_to_eci<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_eci,
        axis,
        relative_motion::rotation_ntw_to_eci,
        relative_motion::rotations_ntw_to_eci,
    )
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Inertial (ECI)
/// frame to the Normal, Tangential, Cross-track (NTW) frame.
///
/// This is the transpose (inverse) of the NTW-to-ECI rotation matrix.
///
/// The NTW frame follows the SANA definition:
/// - Y (T): Unit vector along the inertial velocity.
/// - Z (W): Unit vector along the orbital angular momentum `r x v`.
/// - X (N): `Y x Z`, completing the right-handed set; in the orbit plane, normal to the
///   velocity, pointing outward (radial for a circular orbit).
///
/// On a circular orbit NTW coincides with RTN; on an eccentric orbit the two differ by the
/// flight-path angle. TNW and VNC use the same three directions reordered with a sign change:
/// `TNW = [Y_NTW, -X_NTW, Z_NTW]` and `VNC = [Y_NTW, Z_NTW, X_NTW]`.
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
///     numpy.ndarray: 3x3 rotation matrix transforming from ECI to NTW frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     x_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
///
///     R = bh.rotation_eci_to_ntw(x_eci)
///     print(f"ECI to NTW rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "rotation_eci_to_ntw")]
fn py_rotation_eci_to_ntw<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_eci,
        axis,
        relative_motion::rotation_eci_to_ntw,
        relative_motion::rotations_eci_to_ntw,
    )
}

/// Computes the angular velocity of the Normal, Tangential, Cross-track (NTW) frame with
/// respect to the Earth-Centered Inertial (ECI) frame, expressed in NTW axes. Equal to
/// `omega_ntw_for_body` with `GM_EARTH`.
///
/// The NTW axes are built from the velocity direction and the orbit normal, so the frame turns
/// with the velocity vector: about the W axis at `omega_v = mu * |h| / (r^3 * v^2)`, the
/// two-body rate at which the unit velocity rotates. The rate is exact under two-body motion
/// and is the rate of the osculating frame otherwise. On a circular orbit it equals the mean
/// motion.
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
///     numpy.ndarray: Angular velocity of the NTW frame relative to ECI, expressed in NTW axes (rad/s), shape (3,), or the batch dimensions
///         with 3 components along `axis` for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     x_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
///     omega = bh.omega_ntw(x_eci)
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "omega_ntw")]
fn py_omega_ntw<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_map::<6, 3>(py, x_eci, axis, relative_motion::omega_ntw, relative_motion::omegas_ntw)
}

/// Computes the angular velocity of the Normal, Tangential, Cross-track (NTW) frame with
/// respect to an inertial frame centered on a body with gravitational parameter `gm`,
/// expressed in NTW axes.
///
/// The NTW axes are built from the velocity direction and the orbit normal, so the frame turns
/// with the velocity vector: about the W axis at `omega_v = mu * |h| / (r^3 * v^2)`, the
/// two-body rate at which the unit velocity rotates. The rate is exact under two-body motion
/// and is the rate of the osculating frame otherwise. On a circular orbit it equals the mean
/// motion.
///
/// Args:
///     x_inertial (numpy.ndarray or list): 6D state vector in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     gm (float): Gravitational parameter of the central body (m^3/s^2)
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Angular velocity of the NTW frame relative to the inertial frame, expressed in NTW axes (rad/s), shape (3,), or the batch dimensions
///         with 3 components along `axis` for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     sma = bh.R_MARS + 400e3
///     x = np.array([sma, 0.0, 0.0, 0.0, np.sqrt(bh.GM_MARS / sma), 0.0])
///     omega = bh.omega_ntw_for_body(x, bh.GM_MARS)
///     ```
#[pyfunction]
#[pyo3(signature = (x_inertial, gm, axis=-1))]
#[pyo3(text_signature = "(x_inertial, gm, axis=-1)")]
#[pyo3(name = "omega_ntw_for_body")]
fn py_omega_ntw_for_body<'py>(
    py: Python<'py>,
    x_inertial: &Bound<'py, PyAny>,
    gm: f64,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_map::<6, 3>(
        py,
        x_inertial,
        axis,
        |x| relative_motion::omega_ntw_for_body(x, gm),
        |xs| relative_motion::omegas_ntw_for_body(xs, gm),
    )
}

/// Transforms the absolute states of a chief and deputy satellite from the Earth-Centered
/// Inertial (ECI) frame to the relative state of the deputy with respect to the chief in the
/// rotating Normal, Tangential, Cross-track (NTW) frame.
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
///     numpy.ndarray: 6D relative state of the deputy with respect to the chief in the NTW frame [rho_N, rho_T, rho_W, rho_dot_N, rho_dot_T, rho_dot_W] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
///
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     oe_deputy = np.array([bh.R_EARTH + 701e3, 0.0115, 97.85, 15.05, 30.05, 45.05])
///
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///     x_deputy = bh.state_koe_to_eci(oe_deputy, bh.AngleFormat.DEGREES)
///
///     x_rel_ntw = bh.state_eci_to_ntw(x_chief, x_deputy)
///     print(f"Relative state in NTW: {x_rel_ntw}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_deputy, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_deputy, axis=-1)")]
#[pyo3(name = "state_eci_to_ntw")]
fn py_state_eci_to_ntw<'py>(
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
        relative_motion::state_eci_to_ntw,
        relative_motion::states_eci_to_ntw,
    )
}

/// Transforms the absolute states of a chief and deputy satellite from an inertial frame
/// centered on a body with gravitational parameter `gm` to the relative state of the deputy
/// with respect to the chief in the rotating Normal, Tangential, Cross-track (NTW) frame.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_deputy (numpy.ndarray or list): 6D state vector of the deputy satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     gm (float): Gravitational parameter of the central body (m^3/s^2)
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: 6D relative state of the deputy with respect to the chief in the NTW frame [rho_N, rho_T, rho_W, rho_dot_N, rho_dot_T, rho_dot_W] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe_chief = np.array([bh.R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0])
///     oe_deputy = np.array([bh.R_MARS + 401e3, 0.0515, 92.65, 45.05, 270.05, 10.05])
///
///     x_chief = bh.state_koe_to_inertial_for_body(oe_chief, bh.CentralBody.Mars, bh.AngleFormat.DEGREES)
///     x_deputy = bh.state_koe_to_inertial_for_body(oe_deputy, bh.CentralBody.Mars, bh.AngleFormat.DEGREES)
///
///     x_rel_ntw = bh.state_inertial_to_ntw_for_body(x_chief, x_deputy, bh.GM_MARS)
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_deputy, gm, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_deputy, gm, axis=-1)")]
#[pyo3(name = "state_inertial_to_ntw_for_body")]
fn py_state_inertial_to_ntw_for_body<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_deputy: &Bound<'py, PyAny>,
    gm: f64,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        x_deputy,
        axis,
        |c, d| relative_motion::state_inertial_to_ntw_for_body(c, d, gm),
        |cs, ds| relative_motion::states_inertial_to_ntw_for_body(cs, ds, gm),
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the rotating Normal, Tangential, Cross-track (NTW) frame to the absolute state of the
/// deputy in the Earth-Centered Inertial (ECI) frame.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_rel_ntw (numpy.ndarray or list): 6D relative state of the deputy with respect to the chief in the NTW frame [rho_N, rho_T, rho_W, rho_dot_N, rho_dot_T, rho_dot_W] (m, m/s), shape (6,), or a batch of
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
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///
///     # Relative state: 1km X, 0.5km Y, -0.3km Z
///     x_rel_ntw = np.array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0])
///
///     x_deputy = bh.state_ntw_to_eci(x_chief, x_rel_ntw)
///     print(f"Deputy state in ECI: {x_deputy}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_rel_ntw, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_rel_ntw, axis=-1)")]
#[pyo3(name = "state_ntw_to_eci")]
fn py_state_ntw_to_eci<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_rel_ntw: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        x_rel_ntw,
        axis,
        relative_motion::state_ntw_to_eci,
        relative_motion::states_ntw_to_eci,
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the rotating Normal, Tangential, Cross-track (NTW) frame to the absolute state of the
/// deputy in an inertial frame centered on a body with gravitational parameter `gm`.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_rel_ntw (numpy.ndarray or list): 6D relative state of the deputy with respect to the chief in the NTW frame [rho_N, rho_T, rho_W, rho_dot_N, rho_dot_T, rho_dot_W] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     gm (float): Gravitational parameter of the central body (m^3/s^2)
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: 6D state vector of the deputy satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe_chief = np.array([bh.R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0])
///     x_chief = bh.state_koe_to_inertial_for_body(oe_chief, bh.CentralBody.Mars, bh.AngleFormat.DEGREES)
///     x_rel_ntw = np.array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0])
///
///     x_deputy = bh.state_ntw_to_inertial_for_body(x_chief, x_rel_ntw, bh.GM_MARS)
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_rel_ntw, gm, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_rel_ntw, gm, axis=-1)")]
#[pyo3(name = "state_ntw_to_inertial_for_body")]
fn py_state_ntw_to_inertial_for_body<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_rel_ntw: &Bound<'py, PyAny>,
    gm: f64,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        x_rel_ntw,
        axis,
        |c, r| relative_motion::state_ntw_to_inertial_for_body(c, r, gm),
        |cs, rs| relative_motion::states_ntw_to_inertial_for_body(cs, rs, gm),
    )
}

/// Computes the rotation matrix transforming a vector in the Tangential, Normal, Cross-track
/// (TNW) frame to the Earth-Centered Inertial (ECI) frame.
///
/// The ECI frame can be any inertial frame centered on the orbited body, such as GCRF or EME2000.
///
/// The TNW frame follows the CCSDS and SANA definition:
/// - X (T): Unit vector along the inertial velocity.
/// - Z (W): Unit vector along the orbital angular momentum `r x v`.
/// - Y (N): `Z x X`, completing the right-handed set; in the orbit plane, normal to the
///   velocity, pointing inward (nadir for a circular orbit).
///
/// The matrix is assembled from the NTW axes as `[Y_NTW, -X_NTW, Z_NTW]`, which equals the
/// definition exactly.
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
///     numpy.ndarray: 3x3 rotation matrix transforming from TNW to ECI frame, shape (3, 3), or the batch dimensions
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
///     R = bh.rotation_tnw_to_eci(state)
///     print(f"TNW to ECI rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "rotation_tnw_to_eci")]
fn py_rotation_tnw_to_eci<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_eci,
        axis,
        relative_motion::rotation_tnw_to_eci,
        relative_motion::rotations_tnw_to_eci,
    )
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Inertial (ECI)
/// frame to the Tangential, Normal, Cross-track (TNW) frame.
///
/// This is the transpose (inverse) of the TNW-to-ECI rotation matrix.
///
/// The TNW frame follows the CCSDS and SANA definition:
/// - X (T): Unit vector along the inertial velocity.
/// - Z (W): Unit vector along the orbital angular momentum `r x v`.
/// - Y (N): `Z x X`, completing the right-handed set; in the orbit plane, normal to the
///   velocity, pointing inward (nadir for a circular orbit).
///
/// The matrix is assembled from the NTW axes as `[Y_NTW, -X_NTW, Z_NTW]`, which equals the
/// definition exactly.
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
///     numpy.ndarray: 3x3 rotation matrix transforming from ECI to TNW frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     x_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
///
///     R = bh.rotation_eci_to_tnw(x_eci)
///     print(f"ECI to TNW rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "rotation_eci_to_tnw")]
fn py_rotation_eci_to_tnw<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_eci,
        axis,
        relative_motion::rotation_eci_to_tnw,
        relative_motion::rotations_eci_to_tnw,
    )
}

/// Computes the angular velocity of the Tangential, Normal, Cross-track (TNW) frame with
/// respect to the Earth-Centered Inertial (ECI) frame, expressed in TNW axes. Equal to
/// `omega_tnw_for_body` with `GM_EARTH`.
///
/// The TNW axes are built from the velocity direction and the orbit normal, so the frame turns
/// with the velocity vector: about the W axis at `omega_v = mu * |h| / (r^3 * v^2)`, the
/// two-body rate at which the unit velocity rotates. The rate is exact under two-body motion
/// and is the rate of the osculating frame otherwise. On a circular orbit it equals the mean
/// motion.
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
///     numpy.ndarray: Angular velocity of the TNW frame relative to ECI, expressed in TNW axes (rad/s), shape (3,), or the batch dimensions
///         with 3 components along `axis` for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     x_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
///     omega = bh.omega_tnw(x_eci)
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "omega_tnw")]
fn py_omega_tnw<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_map::<6, 3>(py, x_eci, axis, relative_motion::omega_tnw, relative_motion::omegas_tnw)
}

/// Computes the angular velocity of the Tangential, Normal, Cross-track (TNW) frame with
/// respect to an inertial frame centered on a body with gravitational parameter `gm`,
/// expressed in TNW axes.
///
/// The TNW axes are built from the velocity direction and the orbit normal, so the frame turns
/// with the velocity vector: about the W axis at `omega_v = mu * |h| / (r^3 * v^2)`, the
/// two-body rate at which the unit velocity rotates. The rate is exact under two-body motion
/// and is the rate of the osculating frame otherwise. On a circular orbit it equals the mean
/// motion.
///
/// Args:
///     x_inertial (numpy.ndarray or list): 6D state vector in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     gm (float): Gravitational parameter of the central body (m^3/s^2)
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Angular velocity of the TNW frame relative to the inertial frame, expressed in TNW axes (rad/s), shape (3,), or the batch dimensions
///         with 3 components along `axis` for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     sma = bh.R_MARS + 400e3
///     x = np.array([sma, 0.0, 0.0, 0.0, np.sqrt(bh.GM_MARS / sma), 0.0])
///     omega = bh.omega_tnw_for_body(x, bh.GM_MARS)
///     ```
#[pyfunction]
#[pyo3(signature = (x_inertial, gm, axis=-1))]
#[pyo3(text_signature = "(x_inertial, gm, axis=-1)")]
#[pyo3(name = "omega_tnw_for_body")]
fn py_omega_tnw_for_body<'py>(
    py: Python<'py>,
    x_inertial: &Bound<'py, PyAny>,
    gm: f64,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_map::<6, 3>(
        py,
        x_inertial,
        axis,
        |x| relative_motion::omega_tnw_for_body(x, gm),
        |xs| relative_motion::omegas_tnw_for_body(xs, gm),
    )
}

/// Transforms the absolute states of a chief and deputy satellite from the Earth-Centered
/// Inertial (ECI) frame to the relative state of the deputy with respect to the chief in the
/// rotating Tangential, Normal, Cross-track (TNW) frame.
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
///     numpy.ndarray: 6D relative state of the deputy with respect to the chief in the TNW frame [rho_T, rho_N, rho_W, rho_dot_T, rho_dot_N, rho_dot_W] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
///
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     oe_deputy = np.array([bh.R_EARTH + 701e3, 0.0115, 97.85, 15.05, 30.05, 45.05])
///
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///     x_deputy = bh.state_koe_to_eci(oe_deputy, bh.AngleFormat.DEGREES)
///
///     x_rel_tnw = bh.state_eci_to_tnw(x_chief, x_deputy)
///     print(f"Relative state in TNW: {x_rel_tnw}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_deputy, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_deputy, axis=-1)")]
#[pyo3(name = "state_eci_to_tnw")]
fn py_state_eci_to_tnw<'py>(
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
        relative_motion::state_eci_to_tnw,
        relative_motion::states_eci_to_tnw,
    )
}

/// Transforms the absolute states of a chief and deputy satellite from an inertial frame
/// centered on a body with gravitational parameter `gm` to the relative state of the deputy
/// with respect to the chief in the rotating Tangential, Normal, Cross-track (TNW) frame.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_deputy (numpy.ndarray or list): 6D state vector of the deputy satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     gm (float): Gravitational parameter of the central body (m^3/s^2)
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: 6D relative state of the deputy with respect to the chief in the TNW frame [rho_T, rho_N, rho_W, rho_dot_T, rho_dot_N, rho_dot_W] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe_chief = np.array([bh.R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0])
///     oe_deputy = np.array([bh.R_MARS + 401e3, 0.0515, 92.65, 45.05, 270.05, 10.05])
///
///     x_chief = bh.state_koe_to_inertial_for_body(oe_chief, bh.CentralBody.Mars, bh.AngleFormat.DEGREES)
///     x_deputy = bh.state_koe_to_inertial_for_body(oe_deputy, bh.CentralBody.Mars, bh.AngleFormat.DEGREES)
///
///     x_rel_tnw = bh.state_inertial_to_tnw_for_body(x_chief, x_deputy, bh.GM_MARS)
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_deputy, gm, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_deputy, gm, axis=-1)")]
#[pyo3(name = "state_inertial_to_tnw_for_body")]
fn py_state_inertial_to_tnw_for_body<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_deputy: &Bound<'py, PyAny>,
    gm: f64,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        x_deputy,
        axis,
        |c, d| relative_motion::state_inertial_to_tnw_for_body(c, d, gm),
        |cs, ds| relative_motion::states_inertial_to_tnw_for_body(cs, ds, gm),
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the rotating Tangential, Normal, Cross-track (TNW) frame to the absolute state of the
/// deputy in the Earth-Centered Inertial (ECI) frame.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_rel_tnw (numpy.ndarray or list): 6D relative state of the deputy with respect to the chief in the TNW frame [rho_T, rho_N, rho_W, rho_dot_T, rho_dot_N, rho_dot_W] (m, m/s), shape (6,), or a batch of
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
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///
///     # Relative state: 1km X, 0.5km Y, -0.3km Z
///     x_rel_tnw = np.array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0])
///
///     x_deputy = bh.state_tnw_to_eci(x_chief, x_rel_tnw)
///     print(f"Deputy state in ECI: {x_deputy}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_rel_tnw, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_rel_tnw, axis=-1)")]
#[pyo3(name = "state_tnw_to_eci")]
fn py_state_tnw_to_eci<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_rel_tnw: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        x_rel_tnw,
        axis,
        relative_motion::state_tnw_to_eci,
        relative_motion::states_tnw_to_eci,
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the rotating Tangential, Normal, Cross-track (TNW) frame to the absolute state of the
/// deputy in an inertial frame centered on a body with gravitational parameter `gm`.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_rel_tnw (numpy.ndarray or list): 6D relative state of the deputy with respect to the chief in the TNW frame [rho_T, rho_N, rho_W, rho_dot_T, rho_dot_N, rho_dot_W] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     gm (float): Gravitational parameter of the central body (m^3/s^2)
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: 6D state vector of the deputy satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe_chief = np.array([bh.R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0])
///     x_chief = bh.state_koe_to_inertial_for_body(oe_chief, bh.CentralBody.Mars, bh.AngleFormat.DEGREES)
///     x_rel_tnw = np.array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0])
///
///     x_deputy = bh.state_tnw_to_inertial_for_body(x_chief, x_rel_tnw, bh.GM_MARS)
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_rel_tnw, gm, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_rel_tnw, gm, axis=-1)")]
#[pyo3(name = "state_tnw_to_inertial_for_body")]
fn py_state_tnw_to_inertial_for_body<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_rel_tnw: &Bound<'py, PyAny>,
    gm: f64,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        x_rel_tnw,
        axis,
        |c, r| relative_motion::state_tnw_to_inertial_for_body(c, r, gm),
        |cs, rs| relative_motion::states_tnw_to_inertial_for_body(cs, rs, gm),
    )
}

/// Computes the rotation matrix transforming a vector in the Velocity, Normal, Co-normal
/// (VNC) frame to the Earth-Centered Inertial (ECI) frame.
///
/// The ECI frame can be any inertial frame centered on the orbited body, such as GCRF or EME2000.
///
/// The VNC frame follows the SANA definition:
/// - X (V): Unit vector along the inertial velocity.
/// - Y (N): Unit vector along the orbital angular momentum `r x v` (normal to the orbit).
/// - Z (C): `X x Y`, the co-normal, completing the right-handed set; in the orbit plane,
///   normal to the velocity, pointing outward (radial for a circular orbit).
///
/// The matrix is assembled from the NTW axes as `[Y_NTW, Z_NTW, X_NTW]`, which equals the
/// definition exactly. TNW and VNC use the same three directions as NTW: `TNW = [Y_NTW,
/// -X_NTW, Z_NTW]`, `VNC = [Y_NTW, Z_NTW, X_NTW]`; STK calls VNC the VNB frame.
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
///     numpy.ndarray: 3x3 rotation matrix transforming from VNC to ECI frame, shape (3, 3), or the batch dimensions
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
///     R = bh.rotation_vnc_to_eci(state)
///     print(f"VNC to ECI rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "rotation_vnc_to_eci")]
fn py_rotation_vnc_to_eci<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_eci,
        axis,
        relative_motion::rotation_vnc_to_eci,
        relative_motion::rotations_vnc_to_eci,
    )
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Inertial (ECI)
/// frame to the Velocity, Normal, Co-normal (VNC) frame.
///
/// This is the transpose (inverse) of the VNC-to-ECI rotation matrix.
///
/// The VNC frame follows the SANA definition:
/// - X (V): Unit vector along the inertial velocity.
/// - Y (N): Unit vector along the orbital angular momentum `r x v` (normal to the orbit).
/// - Z (C): `X x Y`, the co-normal, completing the right-handed set; in the orbit plane,
///   normal to the velocity, pointing outward (radial for a circular orbit).
///
/// The matrix is assembled from the NTW axes as `[Y_NTW, Z_NTW, X_NTW]`, which equals the
/// definition exactly. TNW and VNC use the same three directions as NTW: `TNW = [Y_NTW,
/// -X_NTW, Z_NTW]`, `VNC = [Y_NTW, Z_NTW, X_NTW]`; STK calls VNC the VNB frame.
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
///     numpy.ndarray: 3x3 rotation matrix transforming from ECI to VNC frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     x_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
///
///     R = bh.rotation_eci_to_vnc(x_eci)
///     print(f"ECI to VNC rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "rotation_eci_to_vnc")]
fn py_rotation_eci_to_vnc<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_eci,
        axis,
        relative_motion::rotation_eci_to_vnc,
        relative_motion::rotations_eci_to_vnc,
    )
}

/// Computes the angular velocity of the Velocity, Normal, Co-normal (VNC) frame with
/// respect to the Earth-Centered Inertial (ECI) frame, expressed in VNC axes. Equal to
/// `omega_vnc_for_body` with `GM_EARTH`.
///
/// The VNC axes are built from the velocity direction and the orbit normal, and the orbit
/// normal is the Y axis, so the frame turns about Y at `omega_v = mu * |h| / (r^3 * v^2)`, the
/// two-body rate at which the unit velocity rotates. The rate is exact under two-body motion
/// and is the rate of the osculating frame otherwise. On a circular orbit it equals the mean
/// motion.
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
///     numpy.ndarray: Angular velocity of the VNC frame relative to ECI, expressed in VNC axes (rad/s), shape (3,), or the batch dimensions
///         with 3 components along `axis` for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     x_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
///     omega = bh.omega_vnc(x_eci)
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "omega_vnc")]
fn py_omega_vnc<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_map::<6, 3>(py, x_eci, axis, relative_motion::omega_vnc, relative_motion::omegas_vnc)
}

/// Computes the angular velocity of the Velocity, Normal, Co-normal (VNC) frame with
/// respect to an inertial frame centered on a body with gravitational parameter `gm`,
/// expressed in VNC axes.
///
/// The VNC axes are built from the velocity direction and the orbit normal, and the orbit
/// normal is the Y axis, so the frame turns about Y at `omega_v = mu * |h| / (r^3 * v^2)`, the
/// two-body rate at which the unit velocity rotates. The rate is exact under two-body motion
/// and is the rate of the osculating frame otherwise. On a circular orbit it equals the mean
/// motion.
///
/// Args:
///     x_inertial (numpy.ndarray or list): 6D state vector in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     gm (float): Gravitational parameter of the central body (m^3/s^2)
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Angular velocity of the VNC frame relative to the inertial frame, expressed in VNC axes (rad/s), shape (3,), or the batch dimensions
///         with 3 components along `axis` for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     sma = bh.R_MARS + 400e3
///     x = np.array([sma, 0.0, 0.0, 0.0, np.sqrt(bh.GM_MARS / sma), 0.0])
///     omega = bh.omega_vnc_for_body(x, bh.GM_MARS)
///     ```
#[pyfunction]
#[pyo3(signature = (x_inertial, gm, axis=-1))]
#[pyo3(text_signature = "(x_inertial, gm, axis=-1)")]
#[pyo3(name = "omega_vnc_for_body")]
fn py_omega_vnc_for_body<'py>(
    py: Python<'py>,
    x_inertial: &Bound<'py, PyAny>,
    gm: f64,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_map::<6, 3>(
        py,
        x_inertial,
        axis,
        |x| relative_motion::omega_vnc_for_body(x, gm),
        |xs| relative_motion::omegas_vnc_for_body(xs, gm),
    )
}

/// Transforms the absolute states of a chief and deputy satellite from the Earth-Centered
/// Inertial (ECI) frame to the relative state of the deputy with respect to the chief in the
/// rotating Velocity, Normal, Co-normal (VNC) frame.
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
///     numpy.ndarray: 6D relative state of the deputy with respect to the chief in the VNC frame [rho_V, rho_N, rho_C, rho_dot_V, rho_dot_N, rho_dot_C] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
///
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     oe_deputy = np.array([bh.R_EARTH + 701e3, 0.0115, 97.85, 15.05, 30.05, 45.05])
///
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///     x_deputy = bh.state_koe_to_eci(oe_deputy, bh.AngleFormat.DEGREES)
///
///     x_rel_vnc = bh.state_eci_to_vnc(x_chief, x_deputy)
///     print(f"Relative state in VNC: {x_rel_vnc}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_deputy, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_deputy, axis=-1)")]
#[pyo3(name = "state_eci_to_vnc")]
fn py_state_eci_to_vnc<'py>(
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
        relative_motion::state_eci_to_vnc,
        relative_motion::states_eci_to_vnc,
    )
}

/// Transforms the absolute states of a chief and deputy satellite from an inertial frame
/// centered on a body with gravitational parameter `gm` to the relative state of the deputy
/// with respect to the chief in the rotating Velocity, Normal, Co-normal (VNC) frame.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_deputy (numpy.ndarray or list): 6D state vector of the deputy satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     gm (float): Gravitational parameter of the central body (m^3/s^2)
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: 6D relative state of the deputy with respect to the chief in the VNC frame [rho_V, rho_N, rho_C, rho_dot_V, rho_dot_N, rho_dot_C] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe_chief = np.array([bh.R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0])
///     oe_deputy = np.array([bh.R_MARS + 401e3, 0.0515, 92.65, 45.05, 270.05, 10.05])
///
///     x_chief = bh.state_koe_to_inertial_for_body(oe_chief, bh.CentralBody.Mars, bh.AngleFormat.DEGREES)
///     x_deputy = bh.state_koe_to_inertial_for_body(oe_deputy, bh.CentralBody.Mars, bh.AngleFormat.DEGREES)
///
///     x_rel_vnc = bh.state_inertial_to_vnc_for_body(x_chief, x_deputy, bh.GM_MARS)
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_deputy, gm, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_deputy, gm, axis=-1)")]
#[pyo3(name = "state_inertial_to_vnc_for_body")]
fn py_state_inertial_to_vnc_for_body<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_deputy: &Bound<'py, PyAny>,
    gm: f64,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        x_deputy,
        axis,
        |c, d| relative_motion::state_inertial_to_vnc_for_body(c, d, gm),
        |cs, ds| relative_motion::states_inertial_to_vnc_for_body(cs, ds, gm),
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the rotating Velocity, Normal, Co-normal (VNC) frame to the absolute state of the
/// deputy in the Earth-Centered Inertial (ECI) frame.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_rel_vnc (numpy.ndarray or list): 6D relative state of the deputy with respect to the chief in the VNC frame [rho_V, rho_N, rho_C, rho_dot_V, rho_dot_N, rho_dot_C] (m, m/s), shape (6,), or a batch of
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
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///
///     # Relative state: 1km X, 0.5km Y, -0.3km Z
///     x_rel_vnc = np.array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0])
///
///     x_deputy = bh.state_vnc_to_eci(x_chief, x_rel_vnc)
///     print(f"Deputy state in ECI: {x_deputy}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_rel_vnc, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_rel_vnc, axis=-1)")]
#[pyo3(name = "state_vnc_to_eci")]
fn py_state_vnc_to_eci<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_rel_vnc: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        x_rel_vnc,
        axis,
        relative_motion::state_vnc_to_eci,
        relative_motion::states_vnc_to_eci,
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the rotating Velocity, Normal, Co-normal (VNC) frame to the absolute state of the
/// deputy in an inertial frame centered on a body with gravitational parameter `gm`.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_rel_vnc (numpy.ndarray or list): 6D relative state of the deputy with respect to the chief in the VNC frame [rho_V, rho_N, rho_C, rho_dot_V, rho_dot_N, rho_dot_C] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     gm (float): Gravitational parameter of the central body (m^3/s^2)
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: 6D state vector of the deputy satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe_chief = np.array([bh.R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0])
///     x_chief = bh.state_koe_to_inertial_for_body(oe_chief, bh.CentralBody.Mars, bh.AngleFormat.DEGREES)
///     x_rel_vnc = np.array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0])
///
///     x_deputy = bh.state_vnc_to_inertial_for_body(x_chief, x_rel_vnc, bh.GM_MARS)
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_rel_vnc, gm, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_rel_vnc, gm, axis=-1)")]
#[pyo3(name = "state_vnc_to_inertial_for_body")]
fn py_state_vnc_to_inertial_for_body<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_rel_vnc: &Bound<'py, PyAny>,
    gm: f64,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        x_rel_vnc,
        axis,
        |c, r| relative_motion::state_vnc_to_inertial_for_body(c, r, gm),
        |cs, rs| relative_motion::states_vnc_to_inertial_for_body(cs, rs, gm),
    )
}

/// Computes the rotation matrix transforming a vector in the perifocal (PQW) frame to the
/// Earth-Centered Inertial (ECI) frame. Equal to `rotation_pqw_to_inertial_for_body` with
/// `GM_EARTH`.
///
/// The PQW frame follows the SANA definition:
/// - P: Unit vector toward periapsis, along the eccentricity vector.
/// - W: Unit vector along the orbital angular momentum (r x v).
/// - Q: W x P, completing the right-handed set.
///
/// SANA registers PQW only as an inertial snapshot: the axes are taken from the state at the
/// evaluation epoch and treated as fixed, so there is no rate and the Jacobians are block
/// diagonal.
///
/// Degenerate orbits use the zero-angle conventions: when the eccentricity vector norm is
/// below 1e-9 (circular orbit) P is taken along the ascending node, and when the node vector
/// norm is also below 1e-9 (equatorial orbit) P is taken along the inertial x axis projected
/// into the orbit plane.
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
///     numpy.ndarray: 3x3 rotation matrix transforming from PQW to ECI frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     sma = bh.R_EARTH + 700e3
///     state = np.array([sma, 0.0, 0.0, 0.0, bh.perigee_velocity(sma, 0.0), 0.0])
///
///     R = bh.rotation_pqw_to_eci(state)
///     print(f"PQW to ECI rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "rotation_pqw_to_eci")]
fn py_rotation_pqw_to_eci<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_eci,
        axis,
        relative_motion::rotation_pqw_to_eci,
        relative_motion::rotations_pqw_to_eci,
    )
}

/// Computes the rotation matrix transforming a vector in the perifocal (PQW) frame to an
/// inertial frame centered on a body with gravitational parameter `gm`.
///
/// The PQW frame follows the SANA definition:
/// - P: Unit vector toward periapsis, along the eccentricity vector.
/// - W: Unit vector along the orbital angular momentum (r x v).
/// - Q: W x P, completing the right-handed set.
///
/// SANA registers PQW only as an inertial snapshot: the axes are taken from the state at the
/// evaluation epoch and treated as fixed, so there is no rate and the Jacobians are block
/// diagonal.
///
/// Degenerate orbits use the zero-angle conventions: when the eccentricity vector norm is
/// below 1e-9 (circular orbit) P is taken along the ascending node, and when the node vector
/// norm is also below 1e-9 (equatorial orbit) P is taken along the inertial x axis projected
/// into the orbit plane.
///
/// Args:
///     x_inertial (numpy.ndarray or list): 6D state vector in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     gm (float): Gravitational parameter of the central body (m^3/s^2)
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming from PQW to the inertial frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     sma = bh.R_MARS + 400e3
///     x = np.array([sma, 0.0, 0.0, 0.0, 1.05 * np.sqrt(bh.GM_MARS / sma), 0.0])
///
///     R = bh.rotation_pqw_to_inertial_for_body(x, bh.GM_MARS)
///     ```
#[pyfunction]
#[pyo3(signature = (x_inertial, gm, axis=-1))]
#[pyo3(text_signature = "(x_inertial, gm, axis=-1)")]
#[pyo3(name = "rotation_pqw_to_inertial_for_body")]
fn py_rotation_pqw_to_inertial_for_body<'py>(
    py: Python<'py>,
    x_inertial: &Bound<'py, PyAny>,
    gm: f64,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_inertial,
        axis,
        |x| relative_motion::rotation_pqw_to_inertial_for_body(x, gm),
        |xs| relative_motion::rotations_pqw_to_inertial_for_body(xs, gm),
    )
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Inertial (ECI)
/// frame to the perifocal (PQW) frame.
///
/// This is the transpose (inverse) of the PQW-to-ECI rotation matrix.
///
/// The PQW frame follows the SANA definition:
/// - P: Unit vector toward periapsis, along the eccentricity vector.
/// - W: Unit vector along the orbital angular momentum (r x v).
/// - Q: W x P, completing the right-handed set.
///
/// SANA registers PQW only as an inertial snapshot: the axes are taken from the state at the
/// evaluation epoch and treated as fixed, so there is no rate and the Jacobians are block
/// diagonal.
///
/// Degenerate orbits use the zero-angle conventions: when the eccentricity vector norm is
/// below 1e-9 (circular orbit) P is taken along the ascending node, and when the node vector
/// norm is also below 1e-9 (equatorial orbit) P is taken along the inertial x axis projected
/// into the orbit plane.
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
///     numpy.ndarray: 3x3 rotation matrix transforming from ECI to PQW frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     sma = bh.R_EARTH + 700e3
///     state = np.array([sma, 0.0, 0.0, 0.0, bh.perigee_velocity(sma, 0.0), 0.0])
///
///     R = bh.rotation_eci_to_pqw(state)
///     print(f"ECI to PQW rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "rotation_eci_to_pqw")]
fn py_rotation_eci_to_pqw<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_eci,
        axis,
        relative_motion::rotation_eci_to_pqw,
        relative_motion::rotations_eci_to_pqw,
    )
}

/// Computes the rotation matrix transforming a vector in an inertial frame centered on a body
/// with gravitational parameter `gm` to the perifocal (PQW) frame.
///
/// This is the transpose (inverse) of the PQW-to-inertial rotation matrix.
///
/// The PQW frame follows the SANA definition:
/// - P: Unit vector toward periapsis, along the eccentricity vector.
/// - W: Unit vector along the orbital angular momentum (r x v).
/// - Q: W x P, completing the right-handed set.
///
/// SANA registers PQW only as an inertial snapshot: the axes are taken from the state at the
/// evaluation epoch and treated as fixed, so there is no rate and the Jacobians are block
/// diagonal.
///
/// Degenerate orbits use the zero-angle conventions: when the eccentricity vector norm is
/// below 1e-9 (circular orbit) P is taken along the ascending node, and when the node vector
/// norm is also below 1e-9 (equatorial orbit) P is taken along the inertial x axis projected
/// into the orbit plane.
///
/// Args:
///     x_inertial (numpy.ndarray or list): 6D state vector in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     gm (float): Gravitational parameter of the central body (m^3/s^2)
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming from the inertial frame to PQW, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     sma = bh.R_MARS + 400e3
///     x = np.array([sma, 0.0, 0.0, 0.0, 1.05 * np.sqrt(bh.GM_MARS / sma), 0.0])
///
///     R = bh.rotation_inertial_to_pqw_for_body(x, bh.GM_MARS)
///     ```
#[pyfunction]
#[pyo3(signature = (x_inertial, gm, axis=-1))]
#[pyo3(text_signature = "(x_inertial, gm, axis=-1)")]
#[pyo3(name = "rotation_inertial_to_pqw_for_body")]
fn py_rotation_inertial_to_pqw_for_body<'py>(
    py: Python<'py>,
    x_inertial: &Bound<'py, PyAny>,
    gm: f64,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_inertial,
        axis,
        |x| relative_motion::rotation_inertial_to_pqw_for_body(x, gm),
        |xs| relative_motion::rotations_inertial_to_pqw_for_body(xs, gm),
    )
}

/// Transforms the absolute states of a chief and deputy satellite from the Earth-Centered
/// Inertial (ECI) frame to the relative state of the deputy with respect to the chief in the
/// perifocal (PQW) frame.
///
/// PQW is an inertial snapshot, so the relative velocity is a pure rotation of the ECI
/// relative velocity with no transport term.
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
///     numpy.ndarray: 6D relative state of the deputy with respect to the chief in the PQW frame [rho_P, rho_Q, rho_W, rho_dot_P, rho_dot_Q, rho_dot_W] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
///
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
///     oe_deputy = np.array([bh.R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])
///
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///     x_deputy = bh.state_koe_to_eci(oe_deputy, bh.AngleFormat.DEGREES)
///
///     x_rel_pqw = bh.state_eci_to_pqw(x_chief, x_deputy)
///     print(f"Relative state in PQW: {x_rel_pqw}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_deputy, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_deputy, axis=-1)")]
#[pyo3(name = "state_eci_to_pqw")]
fn py_state_eci_to_pqw<'py>(
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
        relative_motion::state_eci_to_pqw,
        relative_motion::states_eci_to_pqw,
    )
}

/// Transforms the absolute states of a chief and deputy satellite from an inertial frame
/// centered on a body with gravitational parameter `gm` to the relative state of the deputy
/// with respect to the chief in the perifocal (PQW) frame.
///
/// PQW is an inertial snapshot, so the relative velocity is a pure rotation of the inertial
/// relative velocity with no transport term.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_deputy (numpy.ndarray or list): 6D state vector of the deputy satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     gm (float): Gravitational parameter of the central body (m^3/s^2)
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: 6D relative state of the deputy with respect to the chief in the PQW frame [rho_P, rho_Q, rho_W, rho_dot_P, rho_dot_Q, rho_dot_W] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe_chief = np.array([bh.R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0])
///     oe_deputy = np.array([bh.R_MARS + 401e3, 0.0515, 92.65, 45.05, 270.05, 10.05])
///
///     x_chief = bh.state_koe_to_inertial_for_body(oe_chief, bh.CentralBody.Mars, bh.AngleFormat.DEGREES)
///     x_deputy = bh.state_koe_to_inertial_for_body(oe_deputy, bh.CentralBody.Mars, bh.AngleFormat.DEGREES)
///
///     x_rel_pqw = bh.state_inertial_to_pqw_for_body(x_chief, x_deputy, bh.GM_MARS)
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_deputy, gm, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_deputy, gm, axis=-1)")]
#[pyo3(name = "state_inertial_to_pqw_for_body")]
fn py_state_inertial_to_pqw_for_body<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_deputy: &Bound<'py, PyAny>,
    gm: f64,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        x_deputy,
        axis,
        |c, d| relative_motion::state_inertial_to_pqw_for_body(c, d, gm),
        |cs, ds| relative_motion::states_inertial_to_pqw_for_body(cs, ds, gm),
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the perifocal (PQW) frame to the absolute state of the deputy in the Earth-Centered
/// Inertial (ECI) frame.
///
/// PQW is an inertial snapshot, so the deputy's ECI relative velocity is a pure rotation of
/// the PQW relative velocity with no transport term.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_rel_pqw (numpy.ndarray or list): 6D relative state of the deputy with respect to the chief in the PQW frame [rho_P, rho_Q, rho_W, rho_dot_P, rho_dot_Q, rho_dot_W] (m, m/s), shape (6,), or a batch of
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
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///
///     x_rel_pqw = np.array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0])
///
///     x_deputy = bh.state_pqw_to_eci(x_chief, x_rel_pqw)
///     print(f"Deputy state in ECI: {x_deputy}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_rel_pqw, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_rel_pqw, axis=-1)")]
#[pyo3(name = "state_pqw_to_eci")]
fn py_state_pqw_to_eci<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_rel_pqw: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        x_rel_pqw,
        axis,
        relative_motion::state_pqw_to_eci,
        relative_motion::states_pqw_to_eci,
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the perifocal (PQW) frame to the absolute state of the deputy in an inertial frame
/// centered on a body with gravitational parameter `gm`.
///
/// PQW is an inertial snapshot, so the deputy's inertial relative velocity is a pure rotation
/// of the PQW relative velocity with no transport term.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_rel_pqw (numpy.ndarray or list): 6D relative state of the deputy with respect to the chief in the PQW frame [rho_P, rho_Q, rho_W, rho_dot_P, rho_dot_Q, rho_dot_W] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     gm (float): Gravitational parameter of the central body (m^3/s^2)
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: 6D state vector of the deputy satellite in the inertial frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe_chief = np.array([bh.R_MARS + 400e3, 0.05, 92.6, 45.0, 270.0, 10.0])
///     x_chief = bh.state_koe_to_inertial_for_body(oe_chief, bh.CentralBody.Mars, bh.AngleFormat.DEGREES)
///     x_rel_pqw = np.array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0])
///
///     x_deputy = bh.state_pqw_to_inertial_for_body(x_chief, x_rel_pqw, bh.GM_MARS)
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_rel_pqw, gm, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_rel_pqw, gm, axis=-1)")]
#[pyo3(name = "state_pqw_to_inertial_for_body")]
fn py_state_pqw_to_inertial_for_body<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_rel_pqw: &Bound<'py, PyAny>,
    gm: f64,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        x_rel_pqw,
        axis,
        |c, r| relative_motion::state_pqw_to_inertial_for_body(c, r, gm),
        |cs, rs| relative_motion::states_pqw_to_inertial_for_body(cs, rs, gm),
    )
}

/// Computes the rotation matrix transforming a vector in the equinoctial (EQW) frame to the
/// Earth-Centered Inertial (ECI) frame.
///
/// The EQW frame follows the SANA definition:
/// - E: Unit vector along the ascending node, z x h normalized, where z is the inertial
///   pole and h the orbit normal.
/// - W: Unit vector along the orbital angular momentum (r x v).
/// - Q: W x E, completing the right-handed set.
///
/// SANA registers EQW only as a quasi-inertial snapshot: the axes are taken from the state at
/// the evaluation epoch and treated as fixed, so there is no rate and the Jacobians are block
/// diagonal. On an equatorial orbit the node is undefined (node vector norm below 1e-9) and E
/// is taken along the inertial x axis projected into the orbit plane, matching the zero
/// right-ascension convention while keeping the matrix orthonormal.
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
///     numpy.ndarray: 3x3 rotation matrix transforming from EQW to ECI frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
///
///     R = bh.rotation_eqw_to_eci(state)
///     print(f"EQW to ECI rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "rotation_eqw_to_eci")]
fn py_rotation_eqw_to_eci<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_eci,
        axis,
        relative_motion::rotation_eqw_to_eci,
        relative_motion::rotations_eqw_to_eci,
    )
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Inertial (ECI)
/// frame to the equinoctial (EQW) frame.
///
/// This is the transpose (inverse) of the EQW-to-ECI rotation matrix.
///
/// The EQW frame follows the SANA definition:
/// - E: Unit vector along the ascending node, z x h normalized, where z is the inertial
///   pole and h the orbit normal.
/// - W: Unit vector along the orbital angular momentum (r x v).
/// - Q: W x E, completing the right-handed set.
///
/// SANA registers EQW only as a quasi-inertial snapshot: the axes are taken from the state at
/// the evaluation epoch and treated as fixed, so there is no rate and the Jacobians are block
/// diagonal. On an equatorial orbit the node is undefined (node vector norm below 1e-9) and E
/// is taken along the inertial x axis projected into the orbit plane, matching the zero
/// right-ascension convention while keeping the matrix orthonormal.
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
///     numpy.ndarray: 3x3 rotation matrix transforming from ECI to EQW frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
///
///     R = bh.rotation_eci_to_eqw(state)
///     print(f"ECI to EQW rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, axis=-1))]
#[pyo3(text_signature = "(x_eci, axis=-1)")]
#[pyo3(name = "rotation_eci_to_eqw")]
fn py_rotation_eci_to_eqw<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_eci,
        axis,
        relative_motion::rotation_eci_to_eqw,
        relative_motion::rotations_eci_to_eqw,
    )
}

/// Transforms the absolute states of a chief and deputy satellite from the Earth-Centered
/// Inertial (ECI) frame to the relative state of the deputy with respect to the chief in the
/// equinoctial (EQW) frame.
///
/// EQW is a quasi-inertial snapshot, so the relative velocity is a pure rotation of the ECI
/// relative velocity with no transport term.
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
///     numpy.ndarray: 6D relative state of the deputy with respect to the chief in the EQW frame [rho_E, rho_Q, rho_W, rho_dot_E, rho_dot_Q, rho_dot_W] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     bh.initialize_eop()
///
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
///     oe_deputy = np.array([bh.R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])
///
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///     x_deputy = bh.state_koe_to_eci(oe_deputy, bh.AngleFormat.DEGREES)
///
///     x_rel_eqw = bh.state_eci_to_eqw(x_chief, x_deputy)
///     print(f"Relative state in EQW: {x_rel_eqw}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_deputy, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_deputy, axis=-1)")]
#[pyo3(name = "state_eci_to_eqw")]
fn py_state_eci_to_eqw<'py>(
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
        relative_motion::state_eci_to_eqw,
        relative_motion::states_eci_to_eqw,
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the equinoctial (EQW) frame to the absolute state of the deputy in the Earth-Centered
/// Inertial (ECI) frame.
///
/// EQW is a quasi-inertial snapshot, so the deputy's ECI relative velocity is a pure rotation of
/// the EQW relative velocity with no transport term.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_rel_eqw (numpy.ndarray or list): 6D relative state of the deputy with respect to the chief in the EQW frame [rho_E, rho_Q, rho_W, rho_dot_E, rho_dot_Q, rho_dot_W] (m, m/s), shape (6,), or a batch of
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
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///
///     x_rel_eqw = np.array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0])
///
///     x_deputy = bh.state_eqw_to_eci(x_chief, x_rel_eqw)
///     print(f"Deputy state in ECI: {x_deputy}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_rel_eqw, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_rel_eqw, axis=-1)")]
#[pyo3(name = "state_eqw_to_eci")]
fn py_state_eqw_to_eci<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_rel_eqw: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_chief,
        x_rel_eqw,
        axis,
        relative_motion::state_eqw_to_eci,
        relative_motion::states_eqw_to_eci,
    )
}

/// Computes the rotation matrix transforming a vector in the Nadir, Sun, Normal (NSW) frame
/// to the Earth-Centered Inertial (ECI) frame.
///
/// The ECI frame can be any inertial frame centered on the orbited body, such as GCRF or EME2000.
///
/// The NSW frame follows the SANA definition:
/// - X: Unit vector toward nadir, opposite the position vector.
/// - Y: As close to the direction of the Sun as possible while normal to X: the unit vector
///   from the spacecraft to the Sun with its X component removed.
/// - Z: `X x Y`, completing the right-handed set.
///
/// `x_sun` is the Sun's state relative to the same center as `x_eci`. Because X lies along the
/// position vector, the frame is identical whether the Sun direction is taken from the center or
/// from the spacecraft. When the Sun lies along the nadir line (projected norm below 1e-9) the Y
/// axis falls back to the along-track direction.
///
/// Args:
///     x_eci (numpy.ndarray or list): 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_sun (numpy.ndarray or list): 6D state vector of the Sun relative to the same center [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis`; only the position is used here.
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming from NSW to ECI frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     x_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
///     x_sun = np.array([bh.AU, 0.0, 0.0, 0.0, 0.0, 0.0])
///
///     R = bh.rotation_nsw_to_eci(x_eci, x_sun)
///     print(f"NSW to ECI rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, x_sun, axis=-1))]
#[pyo3(text_signature = "(x_eci, x_sun, axis=-1)")]
#[pyo3(name = "rotation_nsw_to_eci")]
fn py_rotation_nsw_to_eci<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    x_sun: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair_matrix::<6, 3>(
        py,
        x_eci,
        x_sun,
        axis,
        relative_motion::rotation_nsw_to_eci,
        relative_motion::rotations_nsw_to_eci,
    )
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Inertial (ECI)
/// frame to the Nadir, Sun, Normal (NSW) frame.
///
/// This is the transpose (inverse) of the NSW-to-ECI rotation matrix.
///
/// The NSW frame follows the SANA definition:
/// - X: Unit vector toward nadir, opposite the position vector.
/// - Y: As close to the direction of the Sun as possible while normal to X: the unit vector
///   from the spacecraft to the Sun with its X component removed.
/// - Z: `X x Y`, completing the right-handed set.
///
/// `x_sun` is the Sun's state relative to the same center as `x_eci`. Because X lies along the
/// position vector, the frame is identical whether the Sun direction is taken from the center or
/// from the spacecraft. When the Sun lies along the nadir line (projected norm below 1e-9) the Y
/// axis falls back to the along-track direction.
///
/// Args:
///     x_eci (numpy.ndarray or list): 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_sun (numpy.ndarray or list): 6D state vector of the Sun relative to the same center [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis`; only the position is used here.
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming from ECI to NSW frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     x_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
///     x_sun = np.array([bh.AU, 0.0, 0.0, 0.0, 0.0, 0.0])
///
///     R = bh.rotation_eci_to_nsw(x_eci, x_sun)
///     print(f"ECI to NSW rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, x_sun, axis=-1))]
#[pyo3(text_signature = "(x_eci, x_sun, axis=-1)")]
#[pyo3(name = "rotation_eci_to_nsw")]
fn py_rotation_eci_to_nsw<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    x_sun: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair_matrix::<6, 3>(
        py,
        x_eci,
        x_sun,
        axis,
        relative_motion::rotation_eci_to_nsw,
        relative_motion::rotations_eci_to_nsw,
    )
}

/// Computes the angular velocity of the Nadir, Sun, Normal (NSW) frame with respect to the
/// Earth-Centered Inertial (ECI) frame, expressed in NSW axes.
///
/// The rate follows from the time derivatives of the NSW basis vectors and is purely
/// kinematic; it needs no gravitational parameter. Passing a Sun state with zero velocity
/// gives the fixed-Sun approximation, which omits a term of order 2e-7 rad/s for an Earth
/// orbit.
///
/// Args:
///     x_eci (numpy.ndarray or list): 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_sun (numpy.ndarray or list): 6D state vector of the Sun relative to the same center [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis`.
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: Angular velocity of the NSW frame relative to ECI, expressed in NSW axes (rad/s), shape (3,), or the batch dimensions
///         with 3 components along `axis` for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe = np.array([bh.R_EARTH + 700e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     x_eci = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
///     x_sun = np.array([bh.AU, 0.0, 0.0, 0.0, 0.0, 0.0])
///     omega = bh.omega_nsw(x_eci, x_sun)
///     ```
#[pyfunction]
#[pyo3(signature = (x_eci, x_sun, axis=-1))]
#[pyo3(text_signature = "(x_eci, x_sun, axis=-1)")]
#[pyo3(name = "omega_nsw")]
fn py_omega_nsw<'py>(
    py: Python<'py>,
    x_eci: &Bound<'py, PyAny>,
    x_sun: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair_map::<6, 3>(
        py,
        x_eci,
        x_sun,
        axis,
        relative_motion::omega_nsw,
        relative_motion::omegas_nsw,
    )
}

/// Transforms the absolute states of a chief and deputy satellite from the Earth-Centered
/// Inertial (ECI) frame to the relative state of the deputy with respect to the chief in the
/// rotating Nadir, Sun, Normal (NSW) frame.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_deputy (numpy.ndarray or list): 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_sun (numpy.ndarray or list): 6D state vector of the Sun relative to the same center [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis`.
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the others.
///
/// Returns:
///     numpy.ndarray: 6D relative state of the deputy with respect to the chief in the NSW frame [rho_X, rho_Y, rho_Z, rho_dot_X, rho_dot_Y, rho_dot_Z] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
///     oe_deputy = np.array([bh.R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])
///
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///     x_deputy = bh.state_koe_to_eci(oe_deputy, bh.AngleFormat.DEGREES)
///     x_sun = np.array([bh.AU, 0.0, 0.0, 0.0, 0.0, 0.0])
///
///     x_rel_nsw = bh.state_eci_to_nsw(x_chief, x_deputy, x_sun)
///     print(f"Relative state in NSW: {x_rel_nsw}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_deputy, x_sun, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_deputy, x_sun, axis=-1)")]
#[pyo3(name = "state_eci_to_nsw")]
fn py_state_eci_to_nsw<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_deputy: &Bound<'py, PyAny>,
    x_sun: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_triple::<6>(
        py,
        x_chief,
        x_deputy,
        x_sun,
        axis,
        relative_motion::state_eci_to_nsw,
        relative_motion::states_eci_to_nsw,
    )
}

/// Transforms the relative state of a deputy satellite with respect to a chief satellite from
/// the rotating Nadir, Sun, Normal (NSW) frame to the absolute state of the deputy in the
/// Earth-Centered Inertial (ECI) frame.
///
/// Args:
///     x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_rel_nsw (numpy.ndarray or list): 6D relative state of the deputy with respect to the chief in the NSW frame [rho_X, rho_Y, rho_Z, rho_dot_X, rho_dot_Y, rho_dot_Z] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_sun (numpy.ndarray or list): 6D state vector of the Sun relative to the same center [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis`.
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the others.
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
///     oe_chief = np.array([bh.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
///     x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)
///     x_sun = np.array([bh.AU, 0.0, 0.0, 0.0, 0.0, 0.0])
///
///     x_rel_nsw = np.array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0])
///
///     x_deputy = bh.state_nsw_to_eci(x_chief, x_rel_nsw, x_sun)
///     print(f"Deputy state in ECI: {x_deputy}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_chief, x_rel_nsw, x_sun, axis=-1))]
#[pyo3(text_signature = "(x_chief, x_rel_nsw, x_sun, axis=-1)")]
#[pyo3(name = "state_nsw_to_eci")]
fn py_state_nsw_to_eci<'py>(
    py: Python<'py>,
    x_chief: &Bound<'py, PyAny>,
    x_rel_nsw: &Bound<'py, PyAny>,
    x_sun: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_triple::<6>(
        py,
        x_chief,
        x_rel_nsw,
        x_sun,
        axis,
        relative_motion::state_nsw_to_eci,
        relative_motion::states_nsw_to_eci,
    )
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Earth-Fixed (ECEF)
/// frame to the South, East, Zenith (SEZ) topocentric horizon frame of a site.
///
/// The SEZ frame follows the SANA definition: the local horizon is the fundamental plane, S
/// points due south from the site, E points east, and Z points along the site's WGS84 geodetic
/// vertical. The site is the position part of `x_ecef`. The E axis, and the frame's rate, are
/// undefined at the poles, where longitude itself is undefined; this is not special-cased.
///
/// Args:
///     x_ecef (numpy.ndarray or list): 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)); only the position is used
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming from ECEF to SEZ frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     r_site = bh.position_geodetic_to_ecef(np.array([30.0, 45.0, 500.0]), bh.AngleFormat.DEGREES)
///     x_site = np.array([r_site[0], r_site[1], r_site[2], 0.0, 0.0, 0.0])
///
///     R = bh.rotation_ecef_to_sez(x_site)
///     print(f"ECEF to SEZ rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_ecef, axis=-1))]
#[pyo3(text_signature = "(x_ecef, axis=-1)")]
#[pyo3(name = "rotation_ecef_to_sez")]
fn py_rotation_ecef_to_sez<'py>(
    py: Python<'py>,
    x_ecef: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_ecef,
        axis,
        relative_motion::rotation_ecef_to_sez,
        relative_motion::rotations_ecef_to_sez,
    )
}

/// Computes the rotation matrix transforming a vector in the South, East, Zenith (SEZ)
/// topocentric horizon frame of a site to the Earth-Centered Earth-Fixed (ECEF) frame.
///
/// This is the transpose (inverse) of the ECEF-to-SEZ rotation matrix.
///
/// The SEZ frame follows the SANA definition: the local horizon is the fundamental plane, S
/// points due south from the site, E points east, and Z points along the site's WGS84 geodetic
/// vertical. The site is the position part of `x_ecef`. The E axis, and the frame's rate, are
/// undefined at the poles, where longitude itself is undefined; this is not special-cased.
///
/// Args:
///     x_ecef (numpy.ndarray or list): 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)); only the position is used
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming from SEZ to ECEF frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     r_site = bh.position_geodetic_to_ecef(np.array([30.0, 45.0, 500.0]), bh.AngleFormat.DEGREES)
///     x_site = np.array([r_site[0], r_site[1], r_site[2], 0.0, 0.0, 0.0])
///
///     R = bh.rotation_sez_to_ecef(x_site)
///     print(f"SEZ to ECEF rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_ecef, axis=-1))]
#[pyo3(text_signature = "(x_ecef, axis=-1)")]
#[pyo3(name = "rotation_sez_to_ecef")]
fn py_rotation_sez_to_ecef<'py>(
    py: Python<'py>,
    x_ecef: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_ecef,
        axis,
        relative_motion::rotation_sez_to_ecef,
        relative_motion::rotations_sez_to_ecef,
    )
}

/// Computes the angular velocity of a site's SEZ frame with respect to the ECEF frame,
/// expressed in SEZ axes.
///
/// The SEZ axes depend only on the site's longitude and geodetic latitude, so the frame
/// turns relative to ECEF only when the site moves: about the polar axis at the longitude rate
/// and about the east axis at the latitude rate. A stationary site has zero rate. The rate
/// relative to an inertial frame is this vector plus Earth's rotation rate rotated into SEZ,
/// which the frame graph composes.
///
/// Args:
///     x_ecef (numpy.ndarray or list): 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Angular velocity of the SEZ frame relative to ECEF, expressed in SEZ axes (rad/s), shape (3,), or the batch dimensions
///         with 3 components along `axis` for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     r_site = bh.position_geodetic_to_ecef(np.array([30.0, 45.0, 500.0]), bh.AngleFormat.DEGREES)
///     x_site = np.array([r_site[0], r_site[1], r_site[2], 0.0, 0.0, 0.0])
///     omega = bh.omega_sez(x_site)
///     ```
#[pyfunction]
#[pyo3(signature = (x_ecef, axis=-1))]
#[pyo3(text_signature = "(x_ecef, axis=-1)")]
#[pyo3(name = "omega_sez")]
fn py_omega_sez<'py>(
    py: Python<'py>,
    x_ecef: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_map::<6, 3>(
        py,
        x_ecef,
        axis,
        relative_motion::omega_sez,
        relative_motion::omegas_sez,
    )
}

/// Transforms the absolute Earth-Centered Earth-Fixed (ECEF) states of a site and a target
/// into the relative state of the target with respect to the site in the site's rotating SEZ
/// frame.
///
/// Args:
///     x_site (numpy.ndarray or list): 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_target (numpy.ndarray or list): 6D state vector of the target in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: 6D relative state of the target with respect to the site in the SEZ frame [rho_S, rho_E, rho_Z, rho_dot_S, rho_dot_E, rho_dot_Z] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     r_site = bh.position_geodetic_to_ecef(np.array([30.0, 45.0, 500.0]), bh.AngleFormat.DEGREES)
///     x_site = np.array([r_site[0], r_site[1], r_site[2], 0.0, 0.0, 0.0])
///     x_target = x_site + np.array([200e3, 300e3, 400e3, 0.0, 0.0, 0.0])
///
///     x_rel_sez = bh.state_ecef_to_sez(x_site, x_target)
///     print(f"Relative state in SEZ: {x_rel_sez}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_site, x_target, axis=-1))]
#[pyo3(text_signature = "(x_site, x_target, axis=-1)")]
#[pyo3(name = "state_ecef_to_sez")]
fn py_state_ecef_to_sez<'py>(
    py: Python<'py>,
    x_site: &Bound<'py, PyAny>,
    x_target: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_site,
        x_target,
        axis,
        relative_motion::state_ecef_to_sez,
        relative_motion::states_ecef_to_sez,
    )
}

/// Transforms the relative state of a target with respect to a site from the site's rotating
/// SEZ frame to the absolute state of the target in the Earth-Centered Earth-Fixed (ECEF)
/// frame.
///
/// Args:
///     x_site (numpy.ndarray or list): 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_rel_sez (numpy.ndarray or list): 6D relative state of the target with respect to the site in the SEZ frame [rho_S, rho_E, rho_Z, rho_dot_S, rho_dot_E, rho_dot_Z] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: 6D state vector of the target in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     r_site = bh.position_geodetic_to_ecef(np.array([30.0, 45.0, 500.0]), bh.AngleFormat.DEGREES)
///     x_site = np.array([r_site[0], r_site[1], r_site[2], 0.0, 0.0, 0.0])
///     x_rel_sez = np.array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0])
///
///     x_target = bh.state_sez_to_ecef(x_site, x_rel_sez)
///     print(f"Target state in ECEF: {x_target}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_site, x_rel_sez, axis=-1))]
#[pyo3(text_signature = "(x_site, x_rel_sez, axis=-1)")]
#[pyo3(name = "state_sez_to_ecef")]
fn py_state_sez_to_ecef<'py>(
    py: Python<'py>,
    x_site: &Bound<'py, PyAny>,
    x_rel_sez: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_site,
        x_rel_sez,
        axis,
        relative_motion::state_sez_to_ecef,
        relative_motion::states_sez_to_ecef,
    )
}

/// Computes the rotation matrix transforming a vector in the Earth-Centered Earth-Fixed (ECEF)
/// frame to the East, North, Zenith (ENZ) topocentric horizon frame of a site.
///
/// ENZ is not a SANA orbit-relative frame; it is brahe's `coordinates` topocentric vocabulary
/// (see `rotation_ellipsoid_to_enz`), extended here to a rotating axis with an angular
/// velocity. The local horizon is the fundamental plane, E points east, N points due north
/// from the site, and Z points along the site's WGS84 geodetic vertical. ENZ is a signed
/// permutation of SEZ: `E = E_SEZ`, `N = -S_SEZ`, `Z = Z_SEZ`. The site is the position part
/// of `x_ecef`. The E and N axes, and the frame's rate, are undefined at the poles, where
/// longitude itself is undefined; this is not special-cased.
///
/// Args:
///     x_ecef (numpy.ndarray or list): 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)); only the position is used
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming from ECEF to ENZ frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     r_site = bh.position_geodetic_to_ecef(np.array([30.0, 45.0, 500.0]), bh.AngleFormat.DEGREES)
///     x_site = np.array([r_site[0], r_site[1], r_site[2], 0.0, 0.0, 0.0])
///
///     R = bh.rotation_ecef_to_enz(x_site)
///     print(f"ECEF to ENZ rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_ecef, axis=-1))]
#[pyo3(text_signature = "(x_ecef, axis=-1)")]
#[pyo3(name = "rotation_ecef_to_enz")]
fn py_rotation_ecef_to_enz<'py>(
    py: Python<'py>,
    x_ecef: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_ecef,
        axis,
        relative_motion::rotation_ecef_to_enz,
        relative_motion::rotations_ecef_to_enz,
    )
}

/// Computes the rotation matrix transforming a vector in the East, North, Zenith (ENZ)
/// topocentric horizon frame of a site to the Earth-Centered Earth-Fixed (ECEF) frame.
///
/// This is the transpose (inverse) of the ECEF-to-ENZ rotation matrix.
///
/// ENZ is not a SANA orbit-relative frame; it is brahe's `coordinates` topocentric vocabulary
/// (see `rotation_ellipsoid_to_enz`), extended here to a rotating axis with an angular
/// velocity. The local horizon is the fundamental plane, E points east, N points due north
/// from the site, and Z points along the site's WGS84 geodetic vertical. ENZ is a signed
/// permutation of SEZ: `E = E_SEZ`, `N = -S_SEZ`, `Z = Z_SEZ`. The site is the position part
/// of `x_ecef`. The E and N axes, and the frame's rate, are undefined at the poles, where
/// longitude itself is undefined; this is not special-cased.
///
/// Args:
///     x_ecef (numpy.ndarray or list): 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)); only the position is used
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix transforming from ENZ to ECEF frame, shape (3, 3), or the batch dimensions
///         followed by (3, 3) for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     r_site = bh.position_geodetic_to_ecef(np.array([30.0, 45.0, 500.0]), bh.AngleFormat.DEGREES)
///     x_site = np.array([r_site[0], r_site[1], r_site[2], 0.0, 0.0, 0.0])
///
///     R = bh.rotation_enz_to_ecef(x_site)
///     print(f"ENZ to ECEF rotation matrix:\n{R}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_ecef, axis=-1))]
#[pyo3(text_signature = "(x_ecef, axis=-1)")]
#[pyo3(name = "rotation_enz_to_ecef")]
fn py_rotation_enz_to_ecef<'py>(
    py: Python<'py>,
    x_ecef: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_rotation::<6>(
        py,
        x_ecef,
        axis,
        relative_motion::rotation_enz_to_ecef,
        relative_motion::rotations_enz_to_ecef,
    )
}

/// Computes the angular velocity of a site's ENZ frame with respect to the ECEF frame,
/// expressed in ENZ axes.
///
/// The ENZ axes depend only on the site's longitude and geodetic latitude, so the frame
/// turns relative to ECEF only when the site moves: about the polar axis at the longitude rate
/// and about the east axis at the latitude rate. A stationary site has zero rate. The rate
/// relative to an inertial frame is this vector plus Earth's rotation rate rotated into ENZ,
/// which the frame graph composes. This is the same angular velocity as `omega_sez` expressed
/// in ENZ axes instead of SEZ axes.
///
/// Args:
///     x_ecef (numpy.ndarray or list): 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Angular velocity of the ENZ frame relative to ECEF, expressed in ENZ axes (rad/s), shape (3,), or the batch dimensions
///         with 3 components along `axis` for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     r_site = bh.position_geodetic_to_ecef(np.array([30.0, 45.0, 500.0]), bh.AngleFormat.DEGREES)
///     x_site = np.array([r_site[0], r_site[1], r_site[2], 0.0, 0.0, 0.0])
///     omega = bh.omega_enz(x_site)
///     ```
#[pyfunction]
#[pyo3(signature = (x_ecef, axis=-1))]
#[pyo3(text_signature = "(x_ecef, axis=-1)")]
#[pyo3(name = "omega_enz")]
fn py_omega_enz<'py>(
    py: Python<'py>,
    x_ecef: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_map::<6, 3>(
        py,
        x_ecef,
        axis,
        relative_motion::omega_enz,
        relative_motion::omegas_enz,
    )
}

/// Transforms the absolute Earth-Centered Earth-Fixed (ECEF) states of a site and a target
/// into the relative state of the target with respect to the site in the site's rotating ENZ
/// frame.
///
/// Args:
///     x_site (numpy.ndarray or list): 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_target (numpy.ndarray or list): 6D state vector of the target in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: 6D relative state of the target with respect to the site in the ENZ frame [rho_E, rho_N, rho_Z, rho_dot_E, rho_dot_N, rho_dot_Z] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     r_site = bh.position_geodetic_to_ecef(np.array([30.0, 45.0, 500.0]), bh.AngleFormat.DEGREES)
///     x_site = np.array([r_site[0], r_site[1], r_site[2], 0.0, 0.0, 0.0])
///     x_target = x_site + np.array([200e3, 300e3, 400e3, 0.0, 0.0, 0.0])
///
///     x_rel_enz = bh.state_ecef_to_enz(x_site, x_target)
///     print(f"Relative state in ENZ: {x_rel_enz}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_site, x_target, axis=-1))]
#[pyo3(text_signature = "(x_site, x_target, axis=-1)")]
#[pyo3(name = "state_ecef_to_enz")]
fn py_state_ecef_to_enz<'py>(
    py: Python<'py>,
    x_site: &Bound<'py, PyAny>,
    x_target: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_site,
        x_target,
        axis,
        relative_motion::state_ecef_to_enz,
        relative_motion::states_ecef_to_enz,
    )
}

/// Transforms the relative state of a target with respect to a site from the site's rotating
/// ENZ frame to the absolute state of the target in the Earth-Centered Earth-Fixed (ECEF)
/// frame.
///
/// Args:
///     x_site (numpy.ndarray or list): 6D state vector of the site in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     x_rel_enz (numpy.ndarray or list): 6D relative state of the target with respect to the site in the ENZ frame [rho_E, rho_N, rho_Z, rho_dot_E, rho_dot_N, rho_dot_Z] (m, m/s), shape (6,), or a batch of
///         vectors with the 6 components along `axis` (for example shape (n, 6)).
///     axis (int, optional): The axis along which the 6 components of a single vector
///         lie; the remaining axes enumerate the batch. For a batch of shape (n, 6)
///         the components lie along the last axis, so the default `-1` applies; a
///         (6, n) column layout uses `axis=0`.
///         A single vector in one argument is broadcast across a batch in the other.
///
/// Returns:
///     numpy.ndarray: 6D state vector of the target in the ECEF frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,), or the layout of the batched
///         argument for batched input.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     r_site = bh.position_geodetic_to_ecef(np.array([30.0, 45.0, 500.0]), bh.AngleFormat.DEGREES)
///     x_site = np.array([r_site[0], r_site[1], r_site[2], 0.0, 0.0, 0.0])
///     x_rel_enz = np.array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0])
///
///     x_target = bh.state_enz_to_ecef(x_site, x_rel_enz)
///     print(f"Target state in ECEF: {x_target}")
///     ```
#[pyfunction]
#[pyo3(signature = (x_site, x_rel_enz, axis=-1))]
#[pyo3(text_signature = "(x_site, x_rel_enz, axis=-1)")]
#[pyo3(name = "state_enz_to_ecef")]
fn py_state_enz_to_ecef<'py>(
    py: Python<'py>,
    x_site: &Bound<'py, PyAny>,
    x_rel_enz: &Bound<'py, PyAny>,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    dispatch_vec_pair::<6>(
        py,
        x_site,
        x_rel_enz,
        axis,
        relative_motion::state_enz_to_ecef,
        relative_motion::states_enz_to_ecef,
    )
}
