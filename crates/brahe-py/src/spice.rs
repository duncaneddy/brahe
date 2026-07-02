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

/// Position of a target body relative to a center body from loaded SPK kernels.
///
/// The result is expressed in the kernel's inertial frame (ICRF axes; NAIF
/// labels this "J2000"). If no kernels are loaded, DE440s is loaded
/// automatically.
///
/// Args:
///     target (int): NAIF ID of the target body (e.g. bh.NAIF_MOON)
///     center (int): NAIF ID of the center body (e.g. bh.NAIF_EARTH)
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
///     r = bh.spk_position(bh.NAIF_MOON, bh.NAIF_EARTH, epc)
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
///     target (int): NAIF ID of the target body (e.g. bh.NAIF_MOON)
///     center (int): NAIF ID of the center body (e.g. bh.NAIF_EARTH)
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
///     v = bh.spk_velocity(bh.NAIF_MOON, bh.NAIF_EARTH, epc)
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
///     target (int): NAIF ID of the target body (e.g. bh.NAIF_MOON)
///     center (int): NAIF ID of the center body (e.g. bh.NAIF_EARTH)
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
///     x = bh.spk_state(bh.NAIF_SUN, bh.NAIF_EARTH, epc)
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
///     target (int): NAIF ID of the target body (e.g. bh.NAIF_MOON)
///     center (int): NAIF ID of the center body (e.g. bh.NAIF_EARTH)
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
///     r = bh.spk_position_in_kernel("de440s", bh.NAIF_MOON, bh.NAIF_EARTH, epc)
///     ```
#[pyfunction]
#[pyo3(name = "spk_position_in_kernel")]
fn py_spk_position_in_kernel<'py>(
    py: Python<'py>,
    kernel_name: &str,
    target: i32,
    center: i32,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let r = spice::spk_position_in_kernel(kernel_name, target, center, epc.obj)
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
///     target (int): NAIF ID of the target body (e.g. bh.NAIF_MOON)
///     center (int): NAIF ID of the center body (e.g. bh.NAIF_EARTH)
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
///     v = bh.spk_velocity_in_kernel("de440s", bh.NAIF_MOON, bh.NAIF_EARTH, epc)
///     ```
#[pyfunction]
#[pyo3(name = "spk_velocity_in_kernel")]
fn py_spk_velocity_in_kernel<'py>(
    py: Python<'py>,
    kernel_name: &str,
    target: i32,
    center: i32,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let v = spice::spk_velocity_in_kernel(kernel_name, target, center, epc.obj)
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
///     target (int): NAIF ID of the target body (e.g. bh.NAIF_MOON)
///     center (int): NAIF ID of the center body (e.g. bh.NAIF_EARTH)
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
///     x = bh.spk_state_in_kernel("de440s", bh.NAIF_SUN, bh.NAIF_EARTH, epc)
///     ```
#[pyfunction]
#[pyo3(name = "spk_state_in_kernel")]
fn py_spk_state_in_kernel<'py>(
    py: Python<'py>,
    kernel_name: &str,
    target: i32,
    center: i32,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let x = spice::spk_state_in_kernel(kernel_name, target, center, epc.obj)
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
///     np.ndarray: 3x3 rotation matrix (ICRF to body-fixed). Dimensionless.
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
fn py_pck_rotation_matrix<'py>(
    py: Python<'py>,
    frame_id: i32,
    epc: &PyEpoch,
) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let m = spice::pck_rotation_matrix(frame_id, epc.obj)
        .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok(matrix_to_numpy!(py, m, 3, 3, f64))
}
