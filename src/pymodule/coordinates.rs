#[pyfunction]
#[pyo3(text_signature = "(x_oe, as_degrees)")]
#[pyo3(name = "state_osculating_to_cartesian")]
unsafe fn py_state_osculating_to_cartesian<'py>(
    py: Python<'py>,
    x_oe: &'py PyArray<f64, Ix1>,
    as_degrees: bool,
) -> &'py PyArray<f64, Ix1> {
    let vec =
        coordinates::state_osculating_to_cartesian(numpy_to_vector!(x_oe, 6, f64), as_degrees);

    vector_to_numpy!(py, vec, 6, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(x_cart, as_degrees)")]
#[pyo3(name = "state_cartesian_to_osculating")]
unsafe fn py_state_cartesian_to_osculating<'py>(
    py: Python<'py>,
    x_cart: &'py PyArray<f64, Ix1>,
    as_degrees: bool,
) -> &'py PyArray<f64, Ix1> {
    let vec =
        coordinates::state_cartesian_to_osculating(numpy_to_vector!(x_cart, 6, f64), as_degrees);

    vector_to_numpy!(py, vec, 6, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(x_geoc, as_degrees)")]
#[pyo3(name = "position_geocentric_to_ecef")]
unsafe fn py_position_geocentric_to_ecef<'py>(
    py: Python<'py>,
    x_geoc: &'py PyArray<f64, Ix1>,
    as_degrees: bool,
) -> &'py PyArray<f64, Ix1> {
    let vec =
        coordinates::position_geocentric_to_ecef(numpy_to_vector!(x_geoc, 3, f64), as_degrees)
            .unwrap();

    vector_to_numpy!(py, vec, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(epc, x_ecef)")]
#[pyo3(name = "position_ecef_to_geocentric")]
unsafe fn py_position_ecef_to_geocentric<'py>(
    py: Python<'py>,
    x_ecef: &'py PyArray<f64, Ix1>,
    as_degrees: bool,
) -> &'py PyArray<f64, Ix1> {
    let vec =
        coordinates::position_ecef_to_geocentric(numpy_to_vector!(x_ecef, 3, f64), as_degrees);

    vector_to_numpy!(py, vec, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(epc, x_geod)")]
#[pyo3(name = "position_geodetic_to_ecef")]
unsafe fn py_position_geodetic_to_ecef<'py>(
    py: Python<'py>,
    x_geod: &'py PyArray<f64, Ix1>,
    as_degrees: bool,
) -> &'py PyArray<f64, Ix1> {
    let vec = coordinates::position_geodetic_to_ecef(numpy_to_vector!(x_geod, 3, f64), as_degrees)
        .unwrap();

    vector_to_numpy!(py, vec, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(epc, x_ecef)")]
#[pyo3(name = "position_ecef_to_geodetic")]
unsafe fn py_position_ecef_to_geodetic<'py>(
    py: Python<'py>,
    x_ecef: &'py PyArray<f64, Ix1>,
    as_degrees: bool,
) -> &'py PyArray<f64, Ix1> {
    let vec = coordinates::position_ecef_to_geodetic(numpy_to_vector!(x_ecef, 3, f64), as_degrees);

    vector_to_numpy!(py, vec, 3, f64)
}
