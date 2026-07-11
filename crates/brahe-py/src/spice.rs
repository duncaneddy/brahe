// ============================================================================
// SPICE Kernel Registry Python Bindings
// ============================================================================

/// Load a SPICE kernel into the global registry.
///
/// Known kernel names ("de440s", "de440", "de430", "de432s", "de435",
/// "de438", "de442", "de442s") are downloaded from NAIF and cached; any
/// other argument is treated as a local file path. SPK (.bsp) and binary
/// PCK (.bpc) kernels are detected automatically from the file header.
/// Loading is idempotent; kernels stay resident until unloaded, and later
/// loads take precedence for overlapping coverage.
///
/// Args:
///     name_or_path (str): Kernel name or filesystem path
///
/// Raises:
///     RuntimeError: If the kernel cannot be downloaded, read, or parsed
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.load_kernel("de440s")
///     print(bh.loaded_kernels())
///     ```
#[pyfunction]
#[pyo3(name = "load_kernel")]
fn py_load_kernel(name_or_path: &str) -> PyResult<()> {
    spice::load_kernel(name_or_path).map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Unload a SPICE kernel from the global registry.
///
/// Args:
///     name_or_path (str): The name or path the kernel was loaded under
///
/// Raises:
///     RuntimeError: If no kernel is loaded under that name
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.load_kernel("de440s")
///     bh.unload_kernel("de440s")
///     ```
#[pyfunction]
#[pyo3(name = "unload_kernel")]
fn py_unload_kernel(name_or_path: &str) -> PyResult<()> {
    spice::unload_kernel(name_or_path).map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Remove all kernels from the global registry.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.clear_kernels()
///     assert bh.loaded_kernels() == []
///     ```
#[pyfunction]
#[pyo3(name = "clear_kernels")]
fn py_clear_kernels() {
    spice::clear_kernels()
}

/// List the kernels currently loaded in the global registry, in load order.
///
/// Returns:
///     list[str]: Loaded kernel names/paths
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_ephemeris()
///     assert "de440s" in bh.loaded_kernels()
///     ```
#[pyfunction]
#[pyo3(name = "loaded_kernels")]
fn py_loaded_kernels() -> Vec<String> {
    spice::loaded_kernels()
}

/// Load the kernels most applications need: "de440s" (planetary ephemeris)
/// and "moon_pa_de440" (lunar principal-axes orientation).
///
/// ~150 MB total on first download; cached thereafter. Each kernel load is
/// idempotent, so calling this alongside other `load_kernel` calls is safe.
///
/// Raises:
///     RuntimeError: If a kernel cannot be downloaded, read, or parsed
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.load_common_kernels()
///     print(bh.loaded_kernels())
///     ```
#[pyfunction]
#[pyo3(name = "load_common_kernels")]
fn py_load_common_kernels() -> PyResult<()> {
    spice::load_common_kernels().map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Load every kernel brahe knows how to download: "de440s", "moon_pa_de440",
/// and the satellite-system kernels "mar099s", "jup365", "sat441", "ura184",
/// "nep097", "plu060".
///
/// ~3.5 GB total on first download; cached thereafter. Prefer
/// `load_common_kernels` unless outer-planet body centers or moons are
/// needed.
///
/// Raises:
///     RuntimeError: If a kernel cannot be downloaded, read, or parsed
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.load_all_kernels()
///     print(bh.loaded_kernels())
///     ```
#[pyfunction]
#[pyo3(name = "load_all_kernels")]
fn py_load_all_kernels() -> PyResult<()> {
    spice::load_all_kernels().map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Position of a target body relative to a center body from loaded SPK kernels.
///
/// The result is expressed in the kernel's inertial frame (ICRF axes; NAIF
/// labels this "J2000"). If no kernels are loaded, DE440s is loaded
/// automatically.
///
/// Args:
///     target (int): NAIF ID of the target body (e.g. bh.NAIFId.MOON)
///     center (int): NAIF ID of the center body (e.g. bh.NAIFId.EARTH)
///     epc (Epoch): Epoch of the query
///
/// Returns:
///     np.ndarray: Position of target relative to center. Units: (m)
///
/// Raises:
///     RuntimeError: If no ephemeris path exists between the bodies or the
///         epoch is outside kernel coverage
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     r = bh.spk_position(bh.NAIFId.MOON, bh.NAIFId.EARTH, epc)
///     ```
#[pyfunction]
#[pyo3(name = "spk_position")]
fn py_spk_position<'py>(
    py: Python<'py>,
    target: i32,
    center: i32,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let r = spice::spk_position(target, center, epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, r, 3, f64))
}

/// Velocity of a target body relative to a center body from loaded SPK kernels.
///
/// The result is expressed in the kernel's inertial frame (ICRF axes; NAIF
/// labels this "J2000"). If no kernels are loaded, DE440s is loaded
/// automatically.
///
/// Args:
///     target (int): NAIF ID of the target body (e.g. bh.NAIFId.MOON)
///     center (int): NAIF ID of the center body (e.g. bh.NAIFId.EARTH)
///     epc (Epoch): Epoch of the query
///
/// Returns:
///     np.ndarray: Velocity of target relative to center. Units: (m/s)
///
/// Raises:
///     RuntimeError: If no ephemeris path exists between the bodies or the
///         epoch is outside kernel coverage
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     v = bh.spk_velocity(bh.NAIFId.MOON, bh.NAIFId.EARTH, epc)
///     ```
#[pyfunction]
#[pyo3(name = "spk_velocity")]
fn py_spk_velocity<'py>(
    py: Python<'py>,
    target: i32,
    center: i32,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let v = spice::spk_velocity(target, center, epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, v, 3, f64))
}

/// State (position and velocity) of a target body relative to a center body
/// from loaded SPK kernels.
///
/// The result is expressed in the kernel's inertial frame (ICRF axes; NAIF
/// labels this "J2000"). Computing the state shares a single record lookup
/// between position and velocity. If no kernels are loaded, DE440s is
/// loaded automatically.
///
/// Args:
///     target (int): NAIF ID of the target body (e.g. bh.NAIFId.MOON)
///     center (int): NAIF ID of the center body (e.g. bh.NAIFId.EARTH)
///     epc (Epoch): Epoch of the query
///
/// Returns:
///     np.ndarray: State [x, y, z, vx, vy, vz] of target relative to
///         center. Units: (m, m/s)
///
/// Raises:
///     RuntimeError: If no ephemeris path exists between the bodies or the
///         epoch is outside kernel coverage
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     x = bh.spk_state(bh.NAIFId.SUN, bh.NAIFId.EARTH, epc)
///     ```
#[pyfunction]
#[pyo3(name = "spk_state")]
fn py_spk_state<'py>(
    py: Python<'py>,
    target: i32,
    center: i32,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let x = spice::spk_state(target, center, epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, x, 6, f64))
}

/// Position of a target body relative to a center body, queried from a
/// single named kernel.
///
/// Queries that kernel only — no cross-kernel chaining is performed and
/// the registry's last-loaded-wins precedence semantics do not apply. The
/// kernel is auto-loaded by name or path if not already loaded.
///
/// Args:
///     kernel_name (str): A known DE kernel name (e.g. "de440s", "de440"),
///         or a path to a .bsp file
///     target (int): NAIF ID of the target body (e.g. bh.NAIFId.MOON)
///     center (int): NAIF ID of the center body (e.g. bh.NAIFId.EARTH)
///     epc (Epoch): Epoch of the query
///
/// Returns:
///     np.ndarray: Position of target relative to center in the kernel's
///         inertial frame (ICRF axes). Units: (m)
///
/// Raises:
///     RuntimeError: If the kernel cannot be loaded, does not contain the
///         requested bodies, or does not cover the epoch
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     r = bh.spk_position_from_kernel("de440s", bh.NAIFId.MOON, bh.NAIFId.EARTH, epc)
///     ```
#[pyfunction]
#[pyo3(name = "spk_position_from_kernel")]
fn py_spk_position_from_kernel<'py>(
    py: Python<'py>,
    kernel_name: &str,
    target: i32,
    center: i32,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let r = spice::spk_position_from_kernel(kernel_name, target, center, epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, r, 3, f64))
}

/// Velocity of a target body relative to a center body, queried from a
/// single named kernel.
///
/// Queries that kernel only — no cross-kernel chaining is performed and
/// the registry's last-loaded-wins precedence semantics do not apply. The
/// kernel is auto-loaded by name or path if not already loaded.
///
/// Args:
///     kernel_name (str): A known DE kernel name (e.g. "de440s", "de440"),
///         or a path to a .bsp file
///     target (int): NAIF ID of the target body (e.g. bh.NAIFId.MOON)
///     center (int): NAIF ID of the center body (e.g. bh.NAIFId.EARTH)
///     epc (Epoch): Epoch of the query
///
/// Returns:
///     np.ndarray: Velocity of target relative to center in the kernel's
///         inertial frame (ICRF axes). Units: (m/s)
///
/// Raises:
///     RuntimeError: If the kernel cannot be loaded, does not contain the
///         requested bodies, or does not cover the epoch
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     v = bh.spk_velocity_from_kernel("de440s", bh.NAIFId.MOON, bh.NAIFId.EARTH, epc)
///     ```
#[pyfunction]
#[pyo3(name = "spk_velocity_from_kernel")]
fn py_spk_velocity_from_kernel<'py>(
    py: Python<'py>,
    kernel_name: &str,
    target: i32,
    center: i32,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let v = spice::spk_velocity_from_kernel(kernel_name, target, center, epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, v, 3, f64))
}

/// State (position and velocity) of a target body relative to a center
/// body, queried from a single named kernel.
///
/// Queries that kernel only — no cross-kernel chaining is performed and
/// the registry's last-loaded-wins precedence semantics do not apply. The
/// kernel is auto-loaded by name or path if not already loaded.
///
/// Args:
///     kernel_name (str): A known DE kernel name (e.g. "de440s", "de440"),
///         or a path to a .bsp file
///     target (int): NAIF ID of the target body (e.g. bh.NAIFId.MOON)
///     center (int): NAIF ID of the center body (e.g. bh.NAIFId.EARTH)
///     epc (Epoch): Epoch of the query
///
/// Returns:
///     np.ndarray: State [x, y, z, vx, vy, vz] of target relative to
///         center in the kernel's inertial frame (ICRF axes). Units: (m, m/s)
///
/// Raises:
///     RuntimeError: If the kernel cannot be loaded, does not contain the
///         requested bodies, or does not cover the epoch
///
/// Example:
///     ```python
///     import brahe as bh
///
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     x = bh.spk_state_from_kernel("de440s", bh.NAIFId.SUN, bh.NAIFId.EARTH, epc)
///     ```
#[pyfunction]
#[pyo3(name = "spk_state_from_kernel")]
fn py_spk_state_from_kernel<'py>(
    py: Python<'py>,
    kernel_name: &str,
    target: i32,
    center: i32,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let x = spice::spk_state_from_kernel(kernel_name, target, center, epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, x, 6, f64))
}

/// 3-1-3 Euler angles and rates of a PCK body-fixed frame relative to ICRF.
///
/// PCKs are never auto-initialized; a PCK kernel must be explicitly loaded
/// via `bh.load_kernel(...)` first.
///
/// Args:
///     frame_id (int): Body-frame class ID (e.g. 31008 for MOON_PA_DE440)
///     epc (Epoch): Epoch of the query
///
/// Returns:
///     tuple[np.ndarray, np.ndarray]: (angles [phi, delta, w] in rad,
///         rates in rad/s)
///
/// Raises:
///     RuntimeError: If no loaded binary PCK provides the frame at the epoch
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.load_kernel("moon_pa_de440")
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     angles, rates = bh.pck_euler_angles(31008, epc)
///     ```
#[pyfunction]
#[pyo3(name = "pck_euler_angles")]
#[allow(clippy::type_complexity)]
fn py_pck_euler_angles<'py>(
    py: Python<'py>,
    frame_id: i32,
    epc: &PyEpoch,
) -> PyResult<(Bound<'py, PyArray<f64, Ix1>>, Bound<'py, PyArray<f64, Ix1>>)> {
    let (a, r) = spice::pck_euler_angles(frame_id, epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok((vector_to_numpy!(py, a, 3, f64), vector_to_numpy!(py, r, 3, f64)))
}

/// 3-1-3 Euler angle of a PCK body-fixed frame relative to ICRF.
///
/// PCKs are never auto-initialized; a PCK kernel must be explicitly loaded
/// via `bh.load_kernel(...)` first.
///
/// Args:
///     frame_id (int): Body-frame class ID (e.g. 31008 for MOON_PA_DE440)
///     epc (Epoch): Epoch of the query
///
/// Returns:
///     EulerAngle: ICRF to body-fixed orientation (order ZXZ, radians)
///
/// Raises:
///     RuntimeError: If no loaded binary PCK provides the frame at the epoch
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.load_kernel("moon_pa_de440")
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     e = bh.pck_euler_angle(31008, epc)
///     ```
#[pyfunction]
#[pyo3(name = "pck_euler_angle")]
fn py_pck_euler_angle(frame_id: i32, epc: &PyEpoch) -> PyResult<PyEulerAngle> {
    let e = spice::pck_euler_angle(frame_id, epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyEulerAngle { obj: e })
}

/// Time derivatives of the 3-1-3 Euler angles of a PCK body-fixed frame.
///
/// PCKs are never auto-initialized; a PCK kernel must be explicitly loaded
/// via `bh.load_kernel(...)` first.
///
/// Args:
///     frame_id (int): Body-frame class ID (e.g. 31008 for MOON_PA_DE440)
///     epc (Epoch): Epoch of the query
///
/// Returns:
///     np.ndarray: [phi_dot, delta_dot, w_dot] in rad/s
///
/// Raises:
///     RuntimeError: If no loaded binary PCK provides the frame at the epoch
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.load_kernel("moon_pa_de440")
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     rates = bh.pck_euler_rates(31008, epc)
///     ```
#[pyfunction]
#[pyo3(name = "pck_euler_rates")]
fn py_pck_euler_rates<'py>(
    py: Python<'py>,
    frame_id: i32,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let r = spice::pck_euler_rates(frame_id, epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(vector_to_numpy!(py, r, 3, f64))
}

/// Typed Euler angle and its rates for a PCK body-fixed frame, from a
/// single shared segment lookup.
///
/// PCKs are never auto-initialized; a PCK kernel must be explicitly loaded
/// via `bh.load_kernel(...)` first.
///
/// Args:
///     frame_id (int): Body-frame class ID (e.g. 31008 for MOON_PA_DE440)
///     epc (Epoch): Epoch of the query
///
/// Returns:
///     tuple: (EulerAngle, np.ndarray) — orientation (order ZXZ, radians)
///         and rates [phi_dot, delta_dot, w_dot] in rad/s
///
/// Raises:
///     RuntimeError: If no loaded binary PCK provides the frame at the epoch
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.load_kernel("moon_pa_de440")
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     angle, rates = bh.pck_euler_angle_and_rates(31008, epc)
///     ```
#[pyfunction]
#[pyo3(name = "pck_euler_angle_and_rates")]
fn py_pck_euler_angle_and_rates<'py>(
    py: Python<'py>,
    frame_id: i32,
    epc: &PyEpoch,
) -> PyResult<(PyEulerAngle, Bound<'py, PyArray<f64, Ix1>>)> {
    let (e, r) = spice::pck_euler_angle_and_rates(frame_id, epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok((PyEulerAngle { obj: e }, vector_to_numpy!(py, r, 3, f64)))
}

/// Orientation of a PCK body-fixed frame relative to ICRF, as a unit
/// quaternion.
///
/// PCKs are never auto-initialized; a PCK kernel must be explicitly loaded
/// via `bh.load_kernel(...)` first.
///
/// Args:
///     frame_id (int): Body-frame class ID (e.g. 31008 for MOON_PA_DE440)
///     epc (Epoch): Epoch of the query
///
/// Returns:
///     Quaternion: Unit quaternion (ICRF to body-fixed). Dimensionless.
///
/// Raises:
///     RuntimeError: If no loaded binary PCK provides the frame at the epoch
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.load_kernel("moon_pa_de440")
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     q = bh.pck_quaternion(31008, epc)
///     ```
#[pyfunction]
#[pyo3(name = "pck_quaternion")]
fn py_pck_quaternion(frame_id: i32, epc: &PyEpoch) -> PyResult<PyQuaternion> {
    let q = spice::pck_quaternion(frame_id, epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyQuaternion { obj: q })
}

/// Rotation matrix from ICRF to a PCK body-fixed frame.
///
/// PCKs are never auto-initialized; a PCK kernel must be explicitly loaded
/// via `bh.load_kernel(...)` first.
///
/// Args:
///     frame_id (int): Body-frame class ID (e.g. 31008 for MOON_PA_DE440)
///     epc (Epoch): Epoch of the query
///
/// Returns:
///     RotationMatrix: 3x3 rotation matrix (ICRF to body-fixed). Dimensionless.
///
/// Raises:
///     RuntimeError: If no loaded binary PCK provides the frame at the epoch
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.load_kernel("moon_pa_de440")
///     epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
///     R = bh.pck_rotation_matrix(31008, epc)
///     ```
#[pyfunction]
#[pyo3(name = "pck_rotation_matrix")]
fn py_pck_rotation_matrix(frame_id: i32, epc: &PyEpoch) -> PyResult<PyRotationMatrix> {
    let m = spice::pck_rotation_matrix(frame_id, epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(PyRotationMatrix { obj: m })
}
