/// Helper function to parse strings into appropriate ellipsoidal conversion type
fn string_to_ellipsoidal_conversion_type(s: &str) -> Result<coordinates::EllipsoidalConversionType, PyErr> {
    match s.as_ref() {
        "Geocentric" => Ok(coordinates::EllipsoidalConversionType::Geocentric),
        "Geodetic" => Ok(coordinates::EllipsoidalConversionType::Geodetic),
        _ => Err(exceptions::PyRuntimeError::new_err(format!(
            "Unknown EllipsoidalConverstionType \"{}\". Can be either \"Geocentric\" or \"Geodetic\".",
            s
        ))),
    }
}

/// Helper function to convert time system enumerations into representative string
fn ellipsoidal_conversion_type_to_string(ts: coordinates::EllipsoidalConversionType) -> String {
    match ts {
        coordinates::EllipsoidalConversionType::Geocentric => String::from("Geocentric"),
        coordinates::EllipsoidalConversionType::Geodetic => String::from("Geodetic"),
    }
}

#[pyfunction]
#[pyo3(text_signature = "(x_oe, as_degrees)")]
#[pyo3(name = "state_osculating_to_cartesian")]
unsafe fn py_state_osculating_to_cartesian<'py>(
    py: Python<'py>,
    x_oe: &'py PyArray<f64, Ix1>,
    as_degrees: bool,
) -> &'py PyArray<f64, Ix1> {
    let vec =
        coordinates::state_osculating_to_cartesian(&numpy_to_vector!(x_oe, 6, f64), as_degrees);

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
        coordinates::state_cartesian_to_osculating(&numpy_to_vector!(x_cart, 6, f64), as_degrees);

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
        coordinates::position_geocentric_to_ecef(&numpy_to_vector!(x_geoc, 3, f64), as_degrees)
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
        coordinates::position_ecef_to_geocentric(&numpy_to_vector!(x_ecef, 3, f64), as_degrees);

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
    let vec = coordinates::position_geodetic_to_ecef(&numpy_to_vector!(x_geod, 3, f64), as_degrees)
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
    let vec = coordinates::position_ecef_to_geodetic(&numpy_to_vector!(x_ecef, 3, f64), as_degrees);

    vector_to_numpy!(py, vec, 3, f64)
}


#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "rotation_ellipsoid_to_enz")]
unsafe fn py_rotation_ellipsoid_to_enz<'py>(py: Python<'py>, x_ellipsoid: &'py PyArray<f64, Ix1>, as_degrees: bool) -> &'py PyArray<f64, Ix2> {
    let mat = coordinates::rotation_ellipsoid_to_enz(&numpy_to_vector!(x_ellipsoid, 3, f64), as_degrees);

    matrix_to_numpy!(py, mat, 3, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "rotation_enz_to_ellipsoid")]
unsafe fn py_rotation_enz_to_ellipsoid<'py>(py: Python<'py>, x_ellipsoid: &'py PyArray<f64, Ix1>, as_degrees: bool) -> &'py PyArray<f64, Ix2> {
    let mat = coordinates::rotation_enz_to_ellipsoid(&numpy_to_vector!(x_ellipsoid, 3, f64), as_degrees);

    matrix_to_numpy!(py, mat, 3, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "relative_position_ecef_to_enz")]
unsafe fn py_relative_position_ecef_to_enz<'py>(py: Python<'py>, location_ecef: &'py PyArray<f64, Ix1>, r_ecef: &'py PyArray<f64, Ix1>, conversion_type: &str) -> &'py PyArray<f64, Ix1> {
    let vec = coordinates::relative_position_ecef_to_enz(&numpy_to_vector!(location_ecef, 3, f64), &numpy_to_vector!(r_ecef, 3, f64), string_to_ellipsoidal_conversion_type(conversion_type).unwrap());

    vector_to_numpy!(py, vec, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "relative_position_enz_to_ecef")]
unsafe fn py_relative_position_enz_to_ecef<'py>(py: Python<'py>, location_ecef: &'py PyArray<f64, Ix1>, r_enz: &'py PyArray<f64, Ix1>, conversion_type: &str) -> &'py PyArray<f64, Ix1> {
    let vec = coordinates::relative_position_enz_to_ecef(&numpy_to_vector!(location_ecef, 3, f64), &numpy_to_vector!(r_enz, 3, f64), string_to_ellipsoidal_conversion_type(conversion_type).unwrap());

    vector_to_numpy!(py, vec, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "rotation_ellipsoid_to_sez")]
unsafe fn py_rotation_ellipsoid_to_sez<'py>(py: Python<'py>, x_ellipsoid: &'py PyArray<f64, Ix1>, as_degrees: bool) -> &'py PyArray<f64, Ix2> {
    let mat = coordinates::rotation_ellipsoid_to_sez(&numpy_to_vector!(x_ellipsoid, 3, f64), as_degrees);

    matrix_to_numpy!(py, mat, 3, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "rotation_sez_to_ellipsoid")]
unsafe fn py_rotation_sez_to_ellipsoid<'py>(py: Python<'py>, x_ellipsoid: &'py PyArray<f64, Ix1>, as_degrees: bool) -> &'py PyArray<f64, Ix2> {
    let mat = coordinates::rotation_sez_to_ellipsoid(&numpy_to_vector!(x_ellipsoid, 3, f64), as_degrees);

    matrix_to_numpy!(py, mat, 3, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "relative_position_ecef_to_sez")]
unsafe fn py_relative_position_ecef_to_sez<'py>(py: Python<'py>, location_ecef: &'py PyArray<f64, Ix1>, r_ecef: &'py PyArray<f64, Ix1>, conversion_type: &str) -> &'py PyArray<f64, Ix1> {
    let vec = coordinates::relative_position_ecef_to_sez(&numpy_to_vector!(location_ecef, 3, f64), &numpy_to_vector!(r_ecef, 3, f64), string_to_ellipsoidal_conversion_type(conversion_type).unwrap());

    vector_to_numpy!(py, vec, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "relative_position_sez_to_ecef")]
unsafe fn py_relative_position_sez_to_ecef<'py>(py: Python<'py>, location_ecef: &'py PyArray<f64, Ix1>, x_sez: &'py PyArray<f64, Ix1>, conversion_type: &str) -> &'py PyArray<f64, Ix1> {
    let vec = coordinates::relative_position_sez_to_ecef(&numpy_to_vector!(location_ecef, 3, f64), &numpy_to_vector!(x_sez, 3, f64), string_to_ellipsoidal_conversion_type(conversion_type).unwrap());

    vector_to_numpy!(py, vec, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "position_enz_to_azel")]
unsafe fn py_position_enz_to_azel<'py>(py: Python<'py>, x_enz: &'py PyArray<f64, Ix1>, as_degrees: bool) -> &'py PyArray<f64, Ix1> {
    let vec = coordinates::position_enz_to_azel(&numpy_to_vector!(x_enz, 3, f64), as_degrees);

    vector_to_numpy!(py, vec, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "position_sez_to_azel")]
unsafe fn py_position_sez_to_azel<'py>(py: Python<'py>, x_sez: &'py PyArray<f64, Ix1>, as_degrees: bool) -> &'py PyArray<f64, Ix1> {
    let vec = coordinates::position_sez_to_azel(&numpy_to_vector!(x_sez, 3, f64), as_degrees);

    vector_to_numpy!(py, vec, 3, f64)
}
