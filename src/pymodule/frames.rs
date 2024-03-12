/// Computes the Bias-Precession-Nutation matrix transforming the GCRS to the
/// CIRS intermediate reference frame. This transformation corrects for the
/// bias, precession, and nutation of Celestial Intermediate Origin (CIO) with
/// respect to inertial space.
///
/// This formulation computes the Bias-Precession-Nutation correction matrix
/// according using a CIO based model using using the IAU 2006
/// precession and IAU 2000A nutation models.
///
/// The function will utilize the global Earth orientation and loaded data to
/// apply corrections to the Celestial Intermediate Pole (CIP) derived from
/// empirical observations.
///
/// Arguments:
///     epc (`Epoch`): Epoch instant for computation of transformation matrix
///
/// Returns:
///     rc2i (`numpy.ndarray`): 3x3 Rotation matrix transforming GCRS -> CIRS
///
/// References:
/// - [IAU SOFA Tools For Earth Attitude, Example 5.5](http://www.iausofa.org/2021_0512_C/sofa/sofa_pn_c.pdf) Software Version 18, 2021-04-18
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "bias_precession_nutation")]
unsafe fn py_bias_precession_nutation<'py>(py: Python<'py>, epc: &PyEpoch) -> &'py PyArray<f64, Ix2> {
    let mat = frames::bias_precession_nutation(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the Earth rotation matrix transforming the CIRS to the TIRS
/// intermediate reference frame. This transformation corrects for the Earth
/// rotation.
///
/// Arguments:
///     epc (`Epoch`): Epoch instant for computation of transformation matrix
///
/// Returns:
///     r (`numpy.ndarray`): 3x3 Rotation matrix transforming CIRS -> TIRS
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "earth_rotation")]
unsafe fn py_earth_rotation<'py>(py: Python<'py>, epc: &PyEpoch) -> &'py PyArray<f64, Ix2> {
    let mat = frames::earth_rotation(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the Earth rotation matrix transforming the TIRS to the ITRF reference
/// frame.
///
/// The function will utilize the global Earth orientation and loaded data to
/// apply corrections to compute the polar motion correction based on empirical
/// observations of polar motion drift.
///
/// Arguments:
///     epc (`Epoch`): Epoch instant for computation of transformation matrix
///
/// Returns:
///     rpm (`numpy.ndarray`): 3x3 Rotation matrix transforming TIRS -> ITRF
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "polar_motion")]
unsafe fn py_polar_motion<'py>(py: Python<'py>, epc: &PyEpoch) -> &'py PyArray<f64, Ix2> {
    let mat = frames::polar_motion(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the combined rotation matrix from the inertial to the Earth-fixed
/// reference frame. Applies corrections for bias, precession, nutation,
/// Earth-rotation, and polar motion.
///
/// The transformation is accomplished using the IAU 2006/2000A, CIO-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
///
/// The function will utilize the global Earth orientation and loaded data to
/// apply corrections for Celestial Intermidate Pole (CIP) and polar motion drift
/// derived from empirical observations.
///
/// Arguments:
///     epc (`Epoch`): Epoch instant for computation of transformation matrix
///
/// Returns:
///     r (`numpy.ndarray`): 3x3 Rotation matrix transforming GCRF -> ITRF
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_eci_to_ecef")]
unsafe fn py_rotation_eci_to_ecef<'py>(py: Python<'py>, epc: &PyEpoch) -> &'py PyArray<f64, Ix2> {
    let mat = frames::rotation_eci_to_ecef(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

/// Computes the combined rotation matrix from the Earth-fixed to the inertial
/// reference frame. Applies corrections for bias, precession, nutation,
/// Earth-rotation, and polar motion.
///
/// The transformation is accomplished using the IAU 2006/2000A, CIO-based
/// theory using classical angles. The method as described in section 5.5 of
/// the SOFA C transformation cookbook.
///
/// The function will utilize the global Earth orientation and loaded data to
/// apply corrections for Celestial Intermidate Pole (CIP) and polar motion drift
/// derived from empirical observations.
///
/// Arguments:
///     epc (`Epoch`): Epoch instant for computation of transformation matrix
///
/// Returns:
///     r (`numpy.ndarray`): 3x3 Rotation matrix transforming ITRF -> GCRF
#[pyfunction]
#[pyo3(text_signature = "(epc)")]
#[pyo3(name = "rotation_ecef_to_eci")]
unsafe fn py_rotation_ecef_to_eci<'py>(py: Python<'py>, epc: &PyEpoch) -> &'py PyArray<f64, Ix2> {
    let mat = frames::rotation_ecef_to_eci(epc.obj);
    matrix_to_numpy!(py, mat, 3, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(epc, x)")]
#[pyo3(name = "position_eci_to_ecef")]
unsafe fn py_position_eci_to_ecef<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x: &'py PyArray<f64, Ix1>,
) -> &'py PyArray<f64, Ix1> {
    let vec = frames::position_eci_to_ecef(epc.obj, &numpy_to_vector!(x, 3, f64));

    vector_to_numpy!(py, vec, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(epc, x)")]
#[pyo3(name = "position_ecef_to_eci")]
unsafe fn py_position_ecef_to_eci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x: &'py PyArray<f64, Ix1>,
) -> &'py PyArray<f64, Ix1> {
    let vec = frames::position_ecef_to_eci(epc.obj, &numpy_to_vector!(x, 3, f64));

    vector_to_numpy!(py, vec, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(epc, x_eci)")]
#[pyo3(name = "state_eci_to_ecef")]
unsafe fn py_state_eci_to_ecef<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_eci: &'py PyArray<f64, Ix1>,
) -> &'py PyArray<f64, Ix1> {
    let vec = frames::state_eci_to_ecef(epc.obj, &numpy_to_vector!(x_eci, 6, f64));

    vector_to_numpy!(py, vec, 6, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(epc, x_ecef)")]
#[pyo3(name = "state_ecef_to_eci")]
unsafe fn py_state_ecef_to_eci<'py>(
    py: Python<'py>,
    epc: &PyEpoch,
    x_ecef: &'py PyArray<f64, Ix1>,
) -> &'py PyArray<f64, Ix1> {
    let vec = frames::state_ecef_to_eci(epc.obj, &numpy_to_vector!(x_ecef, 6, f64));

    vector_to_numpy!(py, vec, 6, f64)
}