/// Helper function to parse strings into appropriate ellipsoidal conversion type
fn string_to_ellipsoidal_conversion_type(s: &str) -> Result<coordinates::EllipsoidalConversionType, PyErr> {
    match s {
        "Geocentric" => Ok(coordinates::EllipsoidalConversionType::Geocentric),
        "Geodetic" => Ok(coordinates::EllipsoidalConversionType::Geodetic),
        _ => Err(exceptions::PyRuntimeError::new_err(format!(
            "Unknown EllipsoidalConverstionType \"{}\". Can be either \"Geocentric\" or \"Geodetic\".",
            s
        ))),
    }
}

#[pyfunction]
#[pyo3(text_signature = "(x_oe, angle_format)")]
#[pyo3(name = "state_osculating_to_cartesian")]
/// Convert osculating orbital elements to Cartesian state.
///
/// Transforms a state vector from osculating Keplerian orbital elements to Cartesian
/// position and velocity coordinates.
///
/// Args:
///     x_oe (numpy.ndarray): Osculating orbital elements `[a, e, i, RAAN, omega, M]` where
///         `a` is semi-major axis (meters), `e` is eccentricity (dimensionless), `i` is
///         inclination (radians or degrees), `RAAN` is right ascension of ascending node
///         (radians or degrees), `omega` is argument of periapsis (radians or degrees),
///         and `M` is mean anomaly (radians or degrees).
///     angle_format (AngleFormat): Angle format for angular elements (`RADIANS` or `DEGREES`).
///
/// Returns:
///     (numpy.ndarray): Cartesian state `[x, y, z, vx, vy, vz]` where position is in meters
///         and velocity is in meters per second.
unsafe fn py_state_osculating_to_cartesian<'py>(
    py: Python<'py>,
    x_oe: Bound<'py, PyArray<f64, Ix1>>,
    angle_format: &PyAngleFormat,
) -> Bound<'py, PyArray<f64, Ix1>> {
    let vec =
        coordinates::state_osculating_to_cartesian(numpy_to_vector!(x_oe, 6, f64), angle_format.value);

    vector_to_numpy!(py, vec, 6, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(x_cart, angle_format)")]
#[pyo3(name = "state_cartesian_to_osculating")]
/// Convert Cartesian state to osculating orbital elements.
///
/// Transforms a state vector from Cartesian position and velocity coordinates to
/// osculating Keplerian orbital elements.
///
/// Args:
///     x_cart (numpy.ndarray): Cartesian state `[x, y, z, vx, vy, vz]` where position
///         is in meters and velocity is in meters per second.
///     angle_format (AngleFormat): Angle format for output angular elements (`RADIANS` or `DEGREES`).
///
/// Returns:
///     (numpy.ndarray): Osculating orbital elements `[a, e, i, RAAN, omega, M]` where `a` is
///         semi-major axis (meters), `e` is eccentricity (dimensionless), `i` is inclination
///         (radians or degrees), `RAAN` is right ascension of ascending node (radians or degrees),
///         `omega` is argument of periapsis (radians or degrees), and `M` is mean anomaly
///         (radians or degrees).
unsafe fn py_state_cartesian_to_osculating<'py>(
    py: Python<'py>,
    x_cart: Bound<'py, PyArray<f64, Ix1>>,
    angle_format: &PyAngleFormat,
) -> Bound<'py, PyArray<f64, Ix1>> {
    let vec =
        coordinates::state_cartesian_to_osculating(numpy_to_vector!(x_cart, 6, f64), angle_format.value);

    vector_to_numpy!(py, vec, 6, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(x_geoc, angle_format)")]
#[pyo3(name = "position_geocentric_to_ecef")]
/// Convert geocentric position to `ECEF` Cartesian coordinates.
///
/// Transforms a position from geocentric spherical coordinates (latitude, longitude, radius)
/// to Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates.
///
/// Args:
///     x_geoc (numpy.ndarray): Geocentric position `[latitude, longitude, radius]` where
///         latitude is in radians or degrees, longitude is in radians or degrees, and
///         radius is in meters.
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     (numpy.ndarray): `ECEF` Cartesian position `[x, y, z]` in meters.
unsafe fn py_position_geocentric_to_ecef<'py>(
    py: Python<'py>,
    x_geoc: Bound<'py, PyArray<f64, Ix1>>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec =
        coordinates::position_geocentric_to_ecef(numpy_to_vector!(x_geoc, 3, f64), angle_format.value)
            .map_err(exceptions::PyValueError::new_err)?;

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

#[pyfunction]
#[pyo3(text_signature = "(x_ecef, angle_format)")]
#[pyo3(name = "position_ecef_to_geocentric")]
/// Convert `ECEF` Cartesian position to geocentric coordinates.
///
/// Transforms a position from Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates
/// to geocentric spherical coordinates (latitude, longitude, radius).
///
/// Args:
///     x_ecef (numpy.ndarray): `ECEF` Cartesian position `[x, y, z]` in meters.
///     angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     (numpy.ndarray): Geocentric position `[latitude, longitude, radius]` where latitude
///         is in radians or degrees, longitude is in radians or degrees, and radius is in meters.
unsafe fn py_position_ecef_to_geocentric<'py>(
    py: Python<'py>,
    x_ecef: Bound<'py, PyArray<f64, Ix1>>,
    angle_format: &PyAngleFormat,
) -> Bound<'py, PyArray<f64, Ix1>> {
    let vec =
        coordinates::position_ecef_to_geocentric(numpy_to_vector!(x_ecef, 3, f64), angle_format.value);

    vector_to_numpy!(py, vec, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(x_geod, angle_format)")]
#[pyo3(name = "position_geodetic_to_ecef")]
/// Convert geodetic position to `ECEF` Cartesian coordinates.
///
/// Transforms a position from geodetic coordinates (latitude, longitude, altitude) using
/// the `WGS84` ellipsoid model to Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates.
///
/// Args:
///     x_geod (numpy.ndarray): Geodetic position `[latitude, longitude, altitude]` where
///         latitude is in radians or degrees, longitude is in radians or degrees, and
///         altitude is in meters above the `WGS84` ellipsoid.
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     (numpy.ndarray): `ECEF` Cartesian position `[x, y, z]` in meters.
unsafe fn py_position_geodetic_to_ecef<'py>(
    py: Python<'py>,
    x_geod: Bound<'py, PyArray<f64, Ix1>>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = coordinates::position_geodetic_to_ecef(numpy_to_vector!(x_geod, 3, f64), angle_format.value)
        .map_err(exceptions::PyValueError::new_err)?;

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

#[pyfunction]
#[pyo3(text_signature = "(x_ecef, angle_format)")]
#[pyo3(name = "position_ecef_to_geodetic")]
/// Convert `ECEF` Cartesian position to geodetic coordinates.
///
/// Transforms a position from Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates
/// to geodetic coordinates (latitude, longitude, altitude) using the `WGS84` ellipsoid model.
///
/// Args:
///     x_ecef (numpy.ndarray): `ECEF` Cartesian position `[x, y, z]` in meters.
///     angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     (numpy.ndarray): Geodetic position `[latitude, longitude, altitude]` where latitude
///         is in radians or degrees, longitude is in radians or degrees, and altitude
///         is in meters above the `WGS84` ellipsoid.
unsafe fn py_position_ecef_to_geodetic<'py>(
    py: Python<'py>,
    x_ecef: Bound<'py, PyArray<f64, Ix1>>,
    angle_format: &PyAngleFormat,
) -> Bound<'py, PyArray<f64, Ix1>> {
    let vec = coordinates::position_ecef_to_geodetic(numpy_to_vector!(x_ecef, 3, f64), angle_format.value);

    vector_to_numpy!(py, vec, 3, f64)
}


#[pyfunction]
#[pyo3(text_signature = "(x_ellipsoid, angle_format)")]
#[pyo3(name = "rotation_ellipsoid_to_enz")]
/// Compute rotation matrix from ellipsoidal coordinates to East-North-Up (`ENZ`) frame.
///
/// Calculates the rotation matrix that transforms vectors from an ellipsoidal coordinate
/// frame (geocentric or geodetic) to the local East-North-Up (`ENZ`) topocentric frame at
/// the specified location.
///
/// Args:
///     x_ellipsoid (numpy.ndarray): Ellipsoidal position `[latitude, longitude, altitude/radius]`
///         where latitude is in radians or degrees, longitude is in radians or degrees.
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     (numpy.ndarray): 3x3 rotation matrix from ellipsoidal frame to `ENZ` frame.
unsafe fn py_rotation_ellipsoid_to_enz<'py>(py: Python<'py>, x_ellipsoid: Bound<'py, PyArray<f64, Ix1>>, angle_format: &PyAngleFormat) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = coordinates::rotation_ellipsoid_to_enz(numpy_to_vector!(x_ellipsoid, 3, f64), angle_format.value);

    matrix_to_numpy!(py, mat, 3, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(x_ellipsoid, angle_format)")]
#[pyo3(name = "rotation_enz_to_ellipsoid")]
/// Compute rotation matrix from East-North-Up (`ENZ`) frame to ellipsoidal coordinates.
///
/// Calculates the rotation matrix that transforms vectors from the local East-North-Up
/// (`ENZ`) topocentric frame to an ellipsoidal coordinate frame (geocentric or geodetic)
/// at the specified location.
///
/// Args:
///     x_ellipsoid (numpy.ndarray): Ellipsoidal position `[latitude, longitude, altitude/radius]`
///         where latitude is in radians or degrees, longitude is in radians or degrees.
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     (numpy.ndarray): 3x3 rotation matrix from `ENZ` frame to ellipsoidal frame.
unsafe fn py_rotation_enz_to_ellipsoid<'py>(py: Python<'py>, x_ellipsoid: Bound<'py, PyArray<f64, Ix1>>, angle_format: &PyAngleFormat) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = coordinates::rotation_enz_to_ellipsoid(numpy_to_vector!(x_ellipsoid, 3, f64), angle_format.value);

    matrix_to_numpy!(py, mat, 3, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(location_ecef, r_ecef, conversion_type)")]
#[pyo3(name = "relative_position_ecef_to_enz")]
/// Convert relative position from `ECEF` to East-North-Up (`ENZ`) frame.
///
/// Transforms a relative position vector from Earth-Centered Earth-Fixed (`ECEF`) coordinates
/// to the local East-North-Up (`ENZ`) topocentric frame at the specified location.
///
/// Args:
///     location_ecef (numpy.ndarray): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
///     r_ecef (numpy.ndarray): Position vector in `ECEF` coordinates `[x, y, z]` in meters.
///     conversion_type (str): Type of ellipsoidal conversion, either `"Geocentric"` or `"Geodetic"`.
///
/// Returns:
///     (numpy.ndarray): Relative position in `ENZ` frame `[east, north, up]` in meters.
unsafe fn py_relative_position_ecef_to_enz<'py>(py: Python<'py>, location_ecef: Bound<'py, PyArray<f64, Ix1>>, r_ecef: Bound<'py, PyArray<f64, Ix1>>, conversion_type: &str) -> Bound<'py, PyArray<f64, Ix1>> {
    let vec = coordinates::relative_position_ecef_to_enz(numpy_to_vector!(location_ecef, 3, f64), numpy_to_vector!(r_ecef, 3, f64), string_to_ellipsoidal_conversion_type(conversion_type).unwrap());

    vector_to_numpy!(py, vec, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(location_ecef, r_enz, conversion_type)")]
#[pyo3(name = "relative_position_enz_to_ecef")]
/// Convert relative position from East-North-Up (`ENZ`) frame to `ECEF`.
///
/// Transforms a relative position vector from the local East-North-Up (`ENZ`) topocentric
/// frame to Earth-Centered Earth-Fixed (`ECEF`) coordinates at the specified location.
///
/// Args:
///     location_ecef (numpy.ndarray): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
///     r_enz (numpy.ndarray): Relative position in `ENZ` frame `[east, north, up]` in meters.
///     conversion_type (str): Type of ellipsoidal conversion, either `"Geocentric"` or `"Geodetic"`.
///
/// Returns:
///     (numpy.ndarray): Position vector in `ECEF` coordinates `[x, y, z]` in meters.
unsafe fn py_relative_position_enz_to_ecef<'py>(py: Python<'py>, location_ecef: Bound<'py, PyArray<f64, Ix1>>, r_enz: Bound<'py, PyArray<f64, Ix1>>, conversion_type: &str) -> Bound<'py, PyArray<f64, Ix1>> {
    let vec = coordinates::relative_position_enz_to_ecef(numpy_to_vector!(location_ecef, 3, f64), numpy_to_vector!(r_enz, 3, f64), string_to_ellipsoidal_conversion_type(conversion_type).unwrap());

    vector_to_numpy!(py, vec, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(x_ellipsoid, angle_format)")]
#[pyo3(name = "rotation_ellipsoid_to_sez")]
/// Compute rotation matrix from ellipsoidal coordinates to South-East-Zenith (`SEZ`) frame.
///
/// Calculates the rotation matrix that transforms vectors from an ellipsoidal coordinate
/// frame (geocentric or geodetic) to the local South-East-Zenith (`SEZ`) topocentric frame
/// at the specified location.
///
/// Args:
///     x_ellipsoid (numpy.ndarray): Ellipsoidal position `[latitude, longitude, altitude/radius]`
///         where latitude is in radians or degrees, longitude is in radians or degrees.
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     (numpy.ndarray): 3x3 rotation matrix from ellipsoidal frame to `SEZ` frame.
unsafe fn py_rotation_ellipsoid_to_sez<'py>(py: Python<'py>, x_ellipsoid: Bound<'py, PyArray<f64, Ix1>>, angle_format: &PyAngleFormat) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = coordinates::rotation_ellipsoid_to_sez(numpy_to_vector!(x_ellipsoid, 3, f64), angle_format.value);

    matrix_to_numpy!(py, mat, 3, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(x_ellipsoid, angle_format)")]
#[pyo3(name = "rotation_sez_to_ellipsoid")]
/// Compute rotation matrix from South-East-Zenith (`SEZ`) frame to ellipsoidal coordinates.
///
/// Calculates the rotation matrix that transforms vectors from the local South-East-Zenith
/// (`SEZ`) topocentric frame to an ellipsoidal coordinate frame (geocentric or geodetic)
/// at the specified location.
///
/// Args:
///     x_ellipsoid (numpy.ndarray): Ellipsoidal position `[latitude, longitude, altitude/radius]`
///         where latitude is in radians or degrees, longitude is in radians or degrees.
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     (numpy.ndarray): 3x3 rotation matrix from `SEZ` frame to ellipsoidal frame.
unsafe fn py_rotation_sez_to_ellipsoid<'py>(py: Python<'py>, x_ellipsoid: Bound<'py, PyArray<f64, Ix1>>, angle_format: &PyAngleFormat) -> Bound<'py, PyArray<f64, Ix2>> {
    let mat = coordinates::rotation_sez_to_ellipsoid(numpy_to_vector!(x_ellipsoid, 3, f64), angle_format.value);

    matrix_to_numpy!(py, mat, 3, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(location_ecef, r_ecef, conversion_type)")]
#[pyo3(name = "relative_position_ecef_to_sez")]
/// Convert relative position from `ECEF` to South-East-Zenith (`SEZ`) frame.
///
/// Transforms a relative position vector from Earth-Centered Earth-Fixed (`ECEF`) coordinates
/// to the local South-East-Zenith (`SEZ`) topocentric frame at the specified location.
///
/// Args:
///     location_ecef (numpy.ndarray): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
///     r_ecef (numpy.ndarray): Position vector in `ECEF` coordinates `[x, y, z]` in meters.
///     conversion_type (str): Type of ellipsoidal conversion, either `"Geocentric"` or `"Geodetic"`.
///
/// Returns:
///     (numpy.ndarray): Relative position in `SEZ` frame `[south, east, zenith]` in meters.
unsafe fn py_relative_position_ecef_to_sez<'py>(py: Python<'py>, location_ecef: Bound<'py, PyArray<f64, Ix1>>, r_ecef: Bound<'py, PyArray<f64, Ix1>>, conversion_type: &str) -> Bound<'py, PyArray<f64, Ix1>> {
    let vec = coordinates::relative_position_ecef_to_sez(numpy_to_vector!(location_ecef, 3, f64), numpy_to_vector!(r_ecef, 3, f64), string_to_ellipsoidal_conversion_type(conversion_type).unwrap());

    vector_to_numpy!(py, vec, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(location_ecef, x_sez, conversion_type)")]
#[pyo3(name = "relative_position_sez_to_ecef")]
/// Convert relative position from South-East-Zenith (`SEZ`) frame to `ECEF`.
///
/// Transforms a relative position vector from the local South-East-Zenith (`SEZ`) topocentric
/// frame to Earth-Centered Earth-Fixed (`ECEF`) coordinates at the specified location.
///
/// Args:
///     location_ecef (numpy.ndarray): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
///     x_sez (numpy.ndarray): Relative position in `SEZ` frame `[south, east, zenith]` in meters.
///     conversion_type (str): Type of ellipsoidal conversion, either `"Geocentric"` or `"Geodetic"`.
///
/// Returns:
///     (numpy.ndarray): Position vector in `ECEF` coordinates `[x, y, z]` in meters.
unsafe fn py_relative_position_sez_to_ecef<'py>(py: Python<'py>, location_ecef: Bound<'py, PyArray<f64, Ix1>>, x_sez: Bound<'py, PyArray<f64, Ix1>>, conversion_type: &str) -> Bound<'py, PyArray<f64, Ix1>> {
    let vec = coordinates::relative_position_sez_to_ecef(numpy_to_vector!(location_ecef, 3, f64), numpy_to_vector!(x_sez, 3, f64), string_to_ellipsoidal_conversion_type(conversion_type).unwrap());

    vector_to_numpy!(py, vec, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(x_enz, angle_format)")]
#[pyo3(name = "position_enz_to_azel")]
/// Convert position from East-North-Up (`ENZ`) frame to azimuth-elevation-range.
///
/// Transforms a position from the local East-North-Up (`ENZ`) topocentric frame to
/// azimuth-elevation-range spherical coordinates.
///
/// Args:
///     x_enz (numpy.ndarray): Position in `ENZ` frame `[east, north, up]` in meters.
///     angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     (numpy.ndarray): Azimuth-elevation-range `[azimuth, elevation, range]` where azimuth
///         and elevation are in radians or degrees, and range is in meters.
unsafe fn py_position_enz_to_azel<'py>(py: Python<'py>, x_enz: Bound<'py, PyArray<f64, Ix1>>, angle_format: &PyAngleFormat) -> Bound<'py, PyArray<f64, Ix1>> {
    let vec = coordinates::position_enz_to_azel(numpy_to_vector!(x_enz, 3, f64), angle_format.value);

    vector_to_numpy!(py, vec, 3, f64)
}

#[pyfunction]
#[pyo3(text_signature = "(x_sez, angle_format)")]
#[pyo3(name = "position_sez_to_azel")]
/// Convert position from South-East-Zenith (`SEZ`) frame to azimuth-elevation-range.
///
/// Transforms a position from the local South-East-Zenith (`SEZ`) topocentric frame to
/// azimuth-elevation-range spherical coordinates.
///
/// Args:
///     x_sez (numpy.ndarray): Position in `SEZ` frame `[south, east, zenith]` in meters.
///     angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     (numpy.ndarray): Azimuth-elevation-range `[azimuth, elevation, range]` where azimuth
///         and elevation are in radians or degrees, and range is in meters.
unsafe fn py_position_sez_to_azel<'py>(py: Python<'py>, x_sez: Bound<'py, PyArray<f64, Ix1>>, angle_format: &PyAngleFormat) -> Bound<'py, PyArray<f64, Ix1>> {
    let vec = coordinates::position_sez_to_azel(numpy_to_vector!(x_sez, 3, f64), angle_format.value);

    vector_to_numpy!(py, vec, 3, f64)
}
