/// Python wrapper for EllipsoidalConversionType enum
///
/// Specifies the type of ellipsoidal conversion used in coordinate transformations.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "EllipsoidalConversionType")]
#[derive(Clone)]
pub struct PyEllipsoidalConversionType {
    pub(crate) value: coordinates::EllipsoidalConversionType,
}

#[pymethods]
impl PyEllipsoidalConversionType {
    /// Geocentric ellipsoidal conversion.
    ///
    /// Uses geocentric latitude where the angle is measured from the center of the Earth.
    ///
    /// Returns:
    ///     EllipsoidalConversionType: Geocentric conversion type
    #[classattr]
    #[allow(non_snake_case)]
    fn GEOCENTRIC() -> Self {
        PyEllipsoidalConversionType {
            value: coordinates::EllipsoidalConversionType::Geocentric,
        }
    }

    /// Geodetic ellipsoidal conversion.
    ///
    /// Uses geodetic latitude where the angle is measured perpendicular to the WGS84 ellipsoid.
    ///
    /// Returns:
    ///     EllipsoidalConversionType: Geodetic conversion type
    #[classattr]
    #[allow(non_snake_case)]
    fn GEODETIC() -> Self {
        PyEllipsoidalConversionType {
            value: coordinates::EllipsoidalConversionType::Geodetic,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.value)
    }

    fn __repr__(&self) -> String {
        format!("EllipsoidalConversionType.{:?}", self.value)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.value == other.value),
            CompareOp::Ne => Ok(self.value != other.value),
            _ => Err(exceptions::PyNotImplementedError::new_err("Comparison not supported")),
        }
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
///     x_oe (numpy.ndarray or list): Osculating orbital elements `[a, e, i, RAAN, omega, M]` where
///         `a` is semi-major axis (meters), `e` is eccentricity (dimensionless), `i` is
///         inclination (radians or degrees), `RAAN` is right ascension of ascending node
///         (radians or degrees), `omega` is argument of periapsis (radians or degrees),
///         and `M` is mean anomaly (radians or degrees).
///     angle_format (AngleFormat): Angle format for angular elements (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: Cartesian state `[x, y, z, vx, vy, vz]` where position is in meters
///         and velocity is in meters per second.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Orbital elements for a circular orbit
///     oe = np.array([7000000.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # a, e, i, RAAN, omega, M
///     x_cart = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
///     print(f"Cartesian state: {x_cart}")
///     ```
fn py_state_osculating_to_cartesian<'py>(
    py: Python<'py>,
    x_oe: Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec =
        coordinates::state_osculating_to_cartesian(pyany_to_svector::<6>(&x_oe)?, angle_format.value);

    Ok(vector_to_numpy!(py, vec, 6, f64))
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
///     x_cart (numpy.ndarray or list): Cartesian state `[x, y, z, vx, vy, vz]` where position
///         is in meters and velocity is in meters per second.
///     angle_format (AngleFormat): Angle format for output angular elements (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: Osculating orbital elements `[a, e, i, RAAN, omega, M]` where `a` is
///         semi-major axis (meters), `e` is eccentricity (dimensionless), `i` is inclination
///         (radians or degrees), `RAAN` is right ascension of ascending node (radians or degrees),
///         `omega` is argument of periapsis (radians or degrees), and `M` is mean anomaly
///         (radians or degrees).
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Cartesian state vector
///     x_cart = np.array([7000000.0, 0.0, 0.0, 0.0, 7546.0, 0.0])  # [x, y, z, vx, vy, vz]
///     oe = bh.state_cartesian_to_osculating(x_cart, bh.AngleFormat.RADIANS)
///     print(f"Orbital elements: a={oe[0]:.0f}m, e={oe[1]:.6f}, i={oe[2]:.6f} rad")
///     ```
fn py_state_cartesian_to_osculating<'py>(
    py: Python<'py>,
    x_cart: Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec =
        coordinates::state_cartesian_to_osculating(pyany_to_svector::<6>(&x_cart)?, angle_format.value);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

#[pyfunction]
#[pyo3(text_signature = "(x_geoc, angle_format)")]
#[pyo3(name = "position_geocentric_to_ecef")]
/// Convert geocentric position to `ECEF` Cartesian coordinates.
///
/// Transforms a position from geocentric spherical coordinates (longitude, latitude, radius)
/// to Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates.
///
/// Args:
///     x_geoc (numpy.ndarray or list): Geocentric position `[longitude, latitude, radius]` where
///         longitude is in radians or degrees, latitude is in radians or degrees, and
///         radius is in meters.
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: `ECEF` Cartesian position `[x, y, z]` in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Convert geocentric coordinates to ECEF
///     lon, lat, r = 0.0, 0.0, 6378137.0  # Equator, prime meridian, Earth's radius
///     x_geoc = np.array([lon, lat, r])
///     x_ecef = bh.position_geocentric_to_ecef(x_geoc, bh.AngleFormat.RADIANS)
///     print(f"ECEF position: {x_ecef}")
///     ```
fn py_position_geocentric_to_ecef<'py>(
    py: Python<'py>,
    x_geoc: Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec =
        coordinates::position_geocentric_to_ecef(pyany_to_svector::<3>(&x_geoc)?, angle_format.value)
            .map_err(exceptions::PyValueError::new_err)?;

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

#[pyfunction]
#[pyo3(text_signature = "(x_ecef, angle_format)")]
#[pyo3(name = "position_ecef_to_geocentric")]
/// Convert `ECEF` Cartesian position to geocentric coordinates.
///
/// Transforms a position from Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates
/// to geocentric spherical coordinates (longitude, latitude, radius).
///
/// Args:
///     x_ecef (numpy.ndarray or list): `ECEF` Cartesian position `[x, y, z]` in meters.
///     angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: Geocentric position `[longitude, latitude, radius]` where longitude
///         is in radians or degrees, latitude is in radians or degrees, and radius is in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Convert ECEF to geocentric coordinates
///     x_ecef = np.array([6378137.0, 0.0, 0.0])  # Point on equator, prime meridian
///     x_geoc = bh.position_ecef_to_geocentric(x_ecef, bh.AngleFormat.DEGREES)
///     print(f"Geocentric: lon={x_geoc[0]:.2f}°, lat={x_geoc[1]:.2f}°, r={x_geoc[2]:.0f}m")
///     ```
fn py_position_ecef_to_geocentric<'py>(
    py: Python<'py>,
    x_ecef: Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec =
        coordinates::position_ecef_to_geocentric(pyany_to_svector::<3>(&x_ecef)?, angle_format.value);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

#[pyfunction]
#[pyo3(text_signature = "(x_geod, angle_format)")]
#[pyo3(name = "position_geodetic_to_ecef")]
/// Convert geodetic position to `ECEF` Cartesian coordinates.
///
/// Transforms a position from geodetic coordinates (longitude, latitude, altitude) using
/// the `WGS84` ellipsoid model to Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates.
///
/// Args:
///     x_geod (numpy.ndarray or list): Geodetic position `[longitude, latitude, altitude]` where
///         longitude is in radians or degrees, latitude is in radians or degrees, and
///         altitude is in meters above the `WGS84` ellipsoid.
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: `ECEF` Cartesian position `[x, y, z]` in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Convert geodetic coordinates (GPS-like) to ECEF
///     lon, lat, alt = -105.0, 40.0, 1655.0  # Boulder, CO (degrees, meters)
///     x_geod = np.array([lon, lat, alt])
///     x_ecef = bh.position_geodetic_to_ecef(x_geod, bh.AngleFormat.DEGREES)
///     print(f"ECEF position: {x_ecef}")
///     ```
fn py_position_geodetic_to_ecef<'py>(
    py: Python<'py>,
    x_geod: Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = coordinates::position_geodetic_to_ecef(pyany_to_svector::<3>(&x_geod)?, angle_format.value)
        .map_err(exceptions::PyValueError::new_err)?;

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

#[pyfunction]
#[pyo3(text_signature = "(x_ecef, angle_format)")]
#[pyo3(name = "position_ecef_to_geodetic")]
/// Convert `ECEF` Cartesian position to geodetic coordinates.
///
/// Transforms a position from Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates
/// to geodetic coordinates (longitude, latitude, altitude) using the `WGS84` ellipsoid model.
///
/// Args:
///     x_ecef (numpy.ndarray or list): `ECEF` Cartesian position `[x, y, z]` in meters.
///     angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: Geodetic position `[longitude, latitude, altitude]` where longitude
///         is in radians or degrees, latitude is in radians or degrees, and altitude
///         is in meters above the `WGS84` ellipsoid.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Convert ECEF to geodetic coordinates (GPS-like)
///     x_ecef = np.array([-1275936.0, -4797210.0, 4020109.0])  # Example location
///     x_geod = bh.position_ecef_to_geodetic(x_ecef, bh.AngleFormat.DEGREES)
///     print(f"Geodetic: lon={x_geod[0]:.4f}°, lat={x_geod[1]:.4f}°, alt={x_geod[2]:.0f}m")
///     ```
fn py_position_ecef_to_geodetic<'py>(
    py: Python<'py>,
    x_ecef: Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = coordinates::position_ecef_to_geodetic(pyany_to_svector::<3>(&x_ecef)?, angle_format.value);

    Ok(vector_to_numpy!(py, vec, 3, f64))
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
///     x_ellipsoid (numpy.ndarray or list): Ellipsoidal position `[latitude, longitude, altitude/radius]`
///         where latitude is in radians or degrees, longitude is in radians or degrees.
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix from ellipsoidal frame to `ENZ` frame.
fn py_rotation_ellipsoid_to_enz<'py>(py: Python<'py>, x_ellipsoid: Bound<'py, PyAny>, angle_format: &PyAngleFormat) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let mat = coordinates::rotation_ellipsoid_to_enz(pyany_to_svector::<3>(&x_ellipsoid)?, angle_format.value);

    Ok(matrix_to_numpy!(py, mat, 3, 3, f64))
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
///     x_ellipsoid (numpy.ndarray or list): Ellipsoidal position `[latitude, longitude, altitude/radius]`
///         where latitude is in radians or degrees, longitude is in radians or degrees.
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix from `ENZ` frame to ellipsoidal frame.
fn py_rotation_enz_to_ellipsoid<'py>(py: Python<'py>, x_ellipsoid: Bound<'py, PyAny>, angle_format: &PyAngleFormat) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let mat = coordinates::rotation_enz_to_ellipsoid(pyany_to_svector::<3>(&x_ellipsoid)?, angle_format.value);

    Ok(matrix_to_numpy!(py, mat, 3, 3, f64))
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
///     location_ecef (numpy.ndarray or list): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
///     r_ecef (numpy.ndarray or list): Position vector in `ECEF` coordinates `[x, y, z]` in meters.
///     conversion_type (EllipsoidalConversionType): Type of ellipsoidal conversion (`GEOCENTRIC` or `GEODETIC`).
///
/// Returns:
///     numpy.ndarray: Relative position in `ENZ` frame `[east, north, up]` in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Ground station and satellite positions
///     station_ecef = np.array([4000000.0, 3000000.0, 4000000.0])
///     sat_ecef = np.array([4100000.0, 3100000.0, 4100000.0])
///     enz = bh.relative_position_ecef_to_enz(station_ecef, sat_ecef, bh.EllipsoidalConversionType.GEODETIC)
///     print(f"ENZ: East={enz[0]/1000:.1f}km, North={enz[1]/1000:.1f}km, Up={enz[2]/1000:.1f}km")
///     ```
fn py_relative_position_ecef_to_enz<'py>(py: Python<'py>, location_ecef: Bound<'py, PyAny>, r_ecef: Bound<'py, PyAny>, conversion_type: &PyEllipsoidalConversionType) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = coordinates::relative_position_ecef_to_enz(pyany_to_svector::<3>(&location_ecef)?, pyany_to_svector::<3>(&r_ecef)?, conversion_type.value);

    Ok(vector_to_numpy!(py, vec, 3, f64))
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
///     location_ecef (numpy.ndarray or list): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
///     r_enz (numpy.ndarray or list): Relative position in `ENZ` frame `[east, north, up]` in meters.
///     conversion_type (EllipsoidalConversionType): Type of ellipsoidal conversion (`GEOCENTRIC` or `GEODETIC`).
///
/// Returns:
///     numpy.ndarray: Position vector in `ECEF` coordinates `[x, y, z]` in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Convert ENZ offset back to ECEF
///     station_ecef = np.array([4000000.0, 3000000.0, 4000000.0])
///     enz_offset = np.array([50000.0, 30000.0, 100000.0])  # 50km east, 30km north, 100km up
///     target_ecef = bh.relative_position_enz_to_ecef(station_ecef, enz_offset, bh.EllipsoidalConversionType.GEODETIC)
///     print(f"Target ECEF: {target_ecef}")
///     ```
fn py_relative_position_enz_to_ecef<'py>(py: Python<'py>, location_ecef: Bound<'py, PyAny>, r_enz: Bound<'py, PyAny>, conversion_type: &PyEllipsoidalConversionType) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = coordinates::relative_position_enz_to_ecef(pyany_to_svector::<3>(&location_ecef)?, pyany_to_svector::<3>(&r_enz)?, conversion_type.value);

    Ok(vector_to_numpy!(py, vec, 3, f64))
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
///     x_ellipsoid (numpy.ndarray or list): Ellipsoidal position `[latitude, longitude, altitude/radius]`
///         where latitude is in radians or degrees, longitude is in radians or degrees.
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix from ellipsoidal frame to `SEZ` frame.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Get rotation matrix for ground station in SEZ frame
///     lat, lon, alt = 0.7, -1.5, 100.0  # radians, meters
///     x_geod = np.array([lat, lon, alt])
///     R_sez = bh.rotation_ellipsoid_to_sez(x_geod, bh.AngleFormat.RADIANS)
///     print(f"Rotation matrix shape: {R_sez.shape}")
///     ```
fn py_rotation_ellipsoid_to_sez<'py>(py: Python<'py>, x_ellipsoid: Bound<'py, PyAny>, angle_format: &PyAngleFormat) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let mat = coordinates::rotation_ellipsoid_to_sez(pyany_to_svector::<3>(&x_ellipsoid)?, angle_format.value);

    Ok(matrix_to_numpy!(py, mat, 3, 3, f64))
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
///     x_ellipsoid (numpy.ndarray or list): Ellipsoidal position `[latitude, longitude, altitude/radius]`
///         where latitude is in radians or degrees, longitude is in radians or degrees.
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix from `SEZ` frame to ellipsoidal frame.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Get inverse rotation matrix from SEZ to ellipsoidal
///     lat, lon, alt = 0.7, -1.5, 100.0  # radians, meters
///     x_geod = np.array([lat, lon, alt])
///     R_ellipsoid = bh.rotation_sez_to_ellipsoid(x_geod, bh.AngleFormat.RADIANS)
///     print(f"Rotation matrix shape: {R_ellipsoid.shape}")
///     ```
fn py_rotation_sez_to_ellipsoid<'py>(py: Python<'py>, x_ellipsoid: Bound<'py, PyAny>, angle_format: &PyAngleFormat) -> PyResult<Bound<'py, PyArray<f64, Ix2>>> {
    let mat = coordinates::rotation_sez_to_ellipsoid(pyany_to_svector::<3>(&x_ellipsoid)?, angle_format.value);

    Ok(matrix_to_numpy!(py, mat, 3, 3, f64))
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
///     location_ecef (numpy.ndarray or list): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
///     r_ecef (numpy.ndarray or list): Position vector in `ECEF` coordinates `[x, y, z]` in meters.
///     conversion_type (EllipsoidalConversionType): Type of ellipsoidal conversion (`GEOCENTRIC` or `GEODETIC`).
///
/// Returns:
///     numpy.ndarray: Relative position in `SEZ` frame `[south, east, zenith]` in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Ground station and satellite positions
///     station_ecef = np.array([4000000.0, 3000000.0, 4000000.0])
///     sat_ecef = np.array([4100000.0, 3100000.0, 4100000.0])
///     sez = bh.relative_position_ecef_to_sez(station_ecef, sat_ecef, bh.EllipsoidalConversionType.GEODETIC)
///     print(f"SEZ: South={sez[0]/1000:.1f}km, East={sez[1]/1000:.1f}km, Zenith={sez[2]/1000:.1f}km")
///     ```
fn py_relative_position_ecef_to_sez<'py>(py: Python<'py>, location_ecef: Bound<'py, PyAny>, r_ecef: Bound<'py, PyAny>, conversion_type: &PyEllipsoidalConversionType) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = coordinates::relative_position_ecef_to_sez(pyany_to_svector::<3>(&location_ecef)?, pyany_to_svector::<3>(&r_ecef)?, conversion_type.value);

    Ok(vector_to_numpy!(py, vec, 3, f64))
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
///     location_ecef (numpy.ndarray or list): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
///     x_sez (numpy.ndarray or list): Relative position in `SEZ` frame `[south, east, zenith]` in meters.
///     conversion_type (EllipsoidalConversionType): Type of ellipsoidal conversion (`GEOCENTRIC` or `GEODETIC`).
///
/// Returns:
///     numpy.ndarray: Position vector in `ECEF` coordinates `[x, y, z]` in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Convert SEZ offset back to ECEF
///     station_ecef = np.array([4000000.0, 3000000.0, 4000000.0])
///     sez_offset = np.array([30000.0, 50000.0, 100000.0])  # 30km south, 50km east, 100km up
///     target_ecef = bh.relative_position_sez_to_ecef(station_ecef, sez_offset, bh.EllipsoidalConversionType.GEODETIC)
///     print(f"Target ECEF: {target_ecef}")
///     ```
fn py_relative_position_sez_to_ecef<'py>(py: Python<'py>, location_ecef: Bound<'py, PyAny>, x_sez: Bound<'py, PyAny>, conversion_type: &PyEllipsoidalConversionType) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = coordinates::relative_position_sez_to_ecef(pyany_to_svector::<3>(&location_ecef)?, pyany_to_svector::<3>(&x_sez)?, conversion_type.value);

    Ok(vector_to_numpy!(py, vec, 3, f64))
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
///     x_enz (numpy.ndarray or list): Position in `ENZ` frame `[east, north, up]` in meters.
///     angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: Azimuth-elevation-range `[azimuth, elevation, range]` where azimuth
///         and elevation are in radians or degrees, and range is in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Convert ENZ to azimuth-elevation for satellite tracking
///     enz = np.array([50000.0, 100000.0, 200000.0])  # East, North, Up (meters)
///     azel = bh.position_enz_to_azel(enz, bh.AngleFormat.DEGREES)
///     print(f"Az={azel[0]:.1f}°, El={azel[1]:.1f}°, Range={azel[2]/1000:.1f}km")
///     ```
fn py_position_enz_to_azel<'py>(py: Python<'py>, x_enz: Bound<'py, PyAny>, angle_format: &PyAngleFormat) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = coordinates::position_enz_to_azel(pyany_to_svector::<3>(&x_enz)?, angle_format.value);

    Ok(vector_to_numpy!(py, vec, 3, f64))
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
///     x_sez (numpy.ndarray or list): Position in `SEZ` frame `[south, east, zenith]` in meters.
///     angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: Azimuth-elevation-range `[azimuth, elevation, range]` where azimuth
///         and elevation are in radians or degrees, and range is in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Convert SEZ to azimuth-elevation for satellite tracking
///     sez = np.array([30000.0, 50000.0, 100000.0])  # South, East, Zenith (meters)
///     azel = bh.position_sez_to_azel(sez, bh.AngleFormat.DEGREES)
///     print(f"Az={azel[0]:.1f}°, El={azel[1]:.1f}°, Range={azel[2]/1000:.1f}km")
///     ```
fn py_position_sez_to_azel<'py>(py: Python<'py>, x_sez: Bound<'py, PyAny>, angle_format: &PyAngleFormat) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = coordinates::position_sez_to_azel(pyany_to_svector::<3>(&x_sez)?, angle_format.value);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}
