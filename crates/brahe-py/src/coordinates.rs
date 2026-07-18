/// Python wrapper for EllipsoidalConversionType enum
///
/// Specifies the type of ellipsoidal conversion used in coordinate transformations.
#[pyclass(module = "brahe._brahe", from_py_object)]
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
#[pyo3(name = "state_koe_to_eci")]
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
///     x_cart = bh.state_koe_to_eci(oe, bh.AngleFormat.RADIANS)
///     print(f"Cartesian state: {x_cart}")
///     ```
fn py_state_koe_to_eci<'py>(
    py: Python<'py>,
    x_oe: Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec =
        coordinates::state_koe_to_eci(pyany_to_svector::<6>(&x_oe)?, angle_format.value);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

#[pyfunction]
#[pyo3(text_signature = "(x_cart, angle_format)")]
#[pyo3(name = "state_eci_to_koe")]
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
///     oe = bh.state_eci_to_koe(x_cart, bh.AngleFormat.RADIANS)
///     print(f"Orbital elements: a={oe[0]:.0f}m, e={oe[1]:.6f}, i={oe[2]:.6f} rad")
///     ```
fn py_state_eci_to_koe<'py>(
    py: Python<'py>,
    x_cart: Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec =
        coordinates::state_eci_to_koe(pyany_to_svector::<6>(&x_cart)?, angle_format.value);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

#[pyfunction]
#[pyo3(text_signature = "(x_cart, central_body, angle_format)")]
#[pyo3(name = "state_inertial_to_koe_for_body")]
/// Convert a Cartesian state in a body's ICRF-aligned inertial (BCI) frame to
/// osculating orbital elements referenced to that body's mean equator at J2000.
///
/// Unlike `state_eci_to_koe` (whose elements are referenced to the ICRF axes),
/// the inclination and RAAN here are measured against the **body mean equator
/// at J2000**: the reference plane is normal to the body's IAU pole
/// `(alpha0, delta0)` evaluated at J2000 TDB, with the x-axis at the ascending
/// node of that equator on the ICRF equator - the standard IAU orientation
/// convention (Archinal et al., "Report of the IAU Working Group on
/// Cartographic Coordinates and Rotational Elements: 2015", Celest Mech Dyn
/// Astr 130, 22 (2018), <https://doi.org/10.1007/s10569-017-9805-5>). This
/// ascending node is where `z_ICRF x p_hat` points: that vector is
/// perpendicular to both poles, hence lies in both equatorial planes. This
/// is the natural frame for polar / sun-synchronous / frozen orbits about
/// the Moon, Mars, and other bodies whose spin pole is tilted relative to
/// the ICRF pole. `CentralBody.Earth` is an exact passthrough of
/// `state_eci_to_koe`. Inverse of `state_koe_to_inertial_for_body`.
///
/// Args:
///     x_cart (numpy.ndarray or list): Cartesian state `[x, y, z, vx, vy, vz]` in the
///         body-centered ICRF-aligned frame (e.g. LCI for the Moon, MCI for Mars),
///         position in meters and velocity in meters per second.
///     central_body (CentralBody): Central body (supplies the GM and the IAU pole /
///         body-fixed frame).
///     angle_format (AngleFormat): Angle format for output angular elements (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: Osculating orbital elements `[a, e, i, RAAN, omega, M]` referenced
///         to the body mean equator at J2000.
///
/// Raises:
///     RuntimeError: If `central_body` is a barycenter, has no positive GM, or is a
///         `Custom` body without a pole / `fixed_frame`.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Cartesian state in the Moon-centered inertial (LCI) frame
///     x_cart = np.array([1837.4e3, 0.0, 0.0, 0.0, 1600.0, 0.0])
///     oe = bh.state_inertial_to_koe_for_body(x_cart, bh.CentralBody.Moon, bh.AngleFormat.RADIANS)
///     ```
fn py_state_inertial_to_koe_for_body<'py>(
    py: Python<'py>,
    x_cart: Bound<'py, PyAny>,
    central_body: &PyCentralBody,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = coordinates::state_inertial_to_koe_for_body(
        pyany_to_svector::<6>(&x_cart)?,
        &central_body.body,
        angle_format.value,
    )
    .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

#[pyfunction]
#[pyo3(text_signature = "(x_oe, central_body, angle_format)")]
#[pyo3(name = "state_koe_to_inertial_for_body")]
/// Convert osculating orbital elements referenced to a body's mean equator at
/// J2000 to the equivalent Cartesian state in that body's ICRF-aligned inertial
/// (BCI) frame. Inverse of `state_inertial_to_koe_for_body`.
///
/// Unlike `state_koe_to_eci` (whose elements are referenced to the ICRF axes),
/// the inclination and RAAN here are measured against the **body mean equator
/// at J2000** (the plane normal to the body's IAU pole at J2000 TDB, x-axis at
/// the ascending node of that equator on the ICRF equator - the standard IAU
/// orientation convention (Archinal et al., "Report of the IAU Working Group
/// on Cartographic Coordinates and Rotational Elements: 2015", Celest Mech
/// Dyn Astr 130, 22 (2018), <https://doi.org/10.1007/s10569-017-9805-5>);
/// this ascending node is where `z_ICRF x p_hat` points, since that vector
/// is perpendicular to both poles and hence lies in both equatorial
/// planes). The output state is in the body-centered ICRF-aligned frame,
/// so it composes directly with the
/// body-fixed transforms (`state_bci_to_bcbf`-style) and with the numerical
/// propagators, which integrate in that frame. `CentralBody.Earth` is an exact
/// passthrough of `state_koe_to_eci`.
///
/// Args:
///     x_oe (numpy.ndarray or list): Osculating orbital elements `[a, e, i, RAAN, omega, M]`
///         referenced to the body mean equator at J2000, where the semi-major axis is in
///         meters and angles are in the given format.
///     central_body (CentralBody): Central body (supplies the GM and the IAU pole /
///         body-fixed frame).
///     angle_format (AngleFormat): Angle format for input angular elements (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: Cartesian state `[x, y, z, vx, vy, vz]` in the body-centered
///         ICRF-aligned frame. Units: (m; m/s)
///
/// Raises:
///     RuntimeError: If `central_body` is a barycenter, has no positive GM, or is a
///         `Custom` body without a pole / `fixed_frame`.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # A 90 deg polar orbit referenced to Mars's equator (not the ICRF pole)
///     oe = np.array([bh.R_MARS + 300e3, 0.01, 92.6, 45.0, 270.0, 0.0])
///     x_cart = bh.state_koe_to_inertial_for_body(oe, bh.CentralBody.Mars, bh.AngleFormat.DEGREES)
///     ```
fn py_state_koe_to_inertial_for_body<'py>(
    py: Python<'py>,
    x_oe: Bound<'py, PyAny>,
    central_body: &PyCentralBody,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = coordinates::state_koe_to_inertial_for_body(
        pyany_to_svector::<6>(&x_oe)?,
        &central_body.body,
        angle_format.value,
    )
    .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;

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

#[pyfunction]
#[pyo3(text_signature = "(x_radec, angle_format)")]
#[pyo3(name = "position_radec_to_inertial")]
/// Convert a right ascension, declination, and range into the equivalent
/// Cartesian inertial position.
///
/// Args:
///     x_radec (numpy.ndarray or list): Right ascension, declination, and range
///         `[ra, dec, range]` where right ascension and declination are in
///         radians or degrees, and range is in meters.
///     angle_format (AngleFormat): Angle format for angular elements (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: Cartesian inertial position `[x, y, z]` in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     x_radec = np.array([0.0, 0.0, 1.0])
///     x_inertial = bh.position_radec_to_inertial(x_radec, bh.AngleFormat.DEGREES)
///     print(f"Inertial position: {x_inertial}")
///     ```
fn py_position_radec_to_inertial<'py>(
    py: Python<'py>,
    x_radec: Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec =
        coordinates::position_radec_to_inertial(pyany_to_svector::<3>(&x_radec)?, angle_format.value);

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

#[pyfunction]
#[pyo3(text_signature = "(x_inertial, angle_format)")]
#[pyo3(name = "position_inertial_to_radec")]
/// Convert a Cartesian inertial position into the equivalent right ascension,
/// declination, and range.
///
/// Right ascension is normalized to the range `[0, 360)` degrees (or `[0, 2*pi)`
/// radians). At the polar singularity (`x = y = 0`) right ascension is
/// indeterminate from position alone and is returned as `0`; use
/// `state_inertial_to_radec` to resolve it from velocity instead.
///
/// Args:
///     x_inertial (numpy.ndarray or list): Cartesian inertial position `[x, y, z]` in meters.
///     angle_format (AngleFormat): Angle format for angular output (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: Right ascension, declination, and range `[ra, dec, range]` where
///         right ascension and declination are in radians or degrees, and range is in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     x_inertial = np.array([1.0, 0.0, 0.0])
///     x_radec = bh.position_inertial_to_radec(x_inertial, bh.AngleFormat.DEGREES)
///     print(f"RA/Dec: {x_radec}")
///     ```
fn py_position_inertial_to_radec<'py>(
    py: Python<'py>,
    x_inertial: Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = coordinates::position_inertial_to_radec(
        pyany_to_svector::<3>(&x_inertial)?,
        angle_format.value,
    );

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

#[pyfunction]
#[pyo3(text_signature = "(x_radec, angle_format)")]
#[pyo3(name = "state_radec_to_inertial")]
/// Convert a right ascension, declination, range, and their rates into the
/// equivalent Cartesian inertial position and velocity.
///
/// Args:
///     x_radec (numpy.ndarray or list): Right ascension, declination, range, and rates
///         `[ra, dec, range, ra_rate, dec_rate, range_rate]` where right ascension,
///         declination, and their rates are in radians (or radians/s) or degrees
///         (or degrees/s), and range/range_rate are in meters and meters/s.
///     angle_format (AngleFormat): Angle format for angular elements and rates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: Cartesian inertial position and velocity `[x, y, z, vx, vy, vz]`
///         in meters and meters per second.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     x_radec = np.array([0.0, 0.0, 7000e3, 0.0, 0.0, 0.0])
///     x_inertial = bh.state_radec_to_inertial(x_radec, bh.AngleFormat.DEGREES)
///     print(f"Inertial state: {x_inertial}")
///     ```
fn py_state_radec_to_inertial<'py>(
    py: Python<'py>,
    x_radec: Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec =
        coordinates::state_radec_to_inertial(pyany_to_svector::<6>(&x_radec)?, angle_format.value);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

#[pyfunction]
#[pyo3(text_signature = "(x_inertial, angle_format)")]
#[pyo3(name = "state_inertial_to_radec")]
/// Convert a Cartesian inertial position and velocity into the equivalent
/// right ascension, declination, range, and their rates.
///
/// Right ascension is normalized to the range `[0, 360)` degrees (or `[0, 2*pi)`
/// radians). At the polar singularity (`x = y = 0`), where right ascension is
/// indeterminate from position alone, it is instead resolved from the
/// velocity components.
///
/// Args:
///     x_inertial (numpy.ndarray or list): Cartesian inertial position and velocity
///         `[x, y, z, vx, vy, vz]` in meters and meters per second.
///     angle_format (AngleFormat): Angle format for angular output and rates (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: Right ascension, declination, range, and rates
///         `[ra, dec, range, ra_rate, dec_rate, range_rate]` where right ascension,
///         declination, and their rates are in radians (or radians/s) or degrees
///         (or degrees/s), and range/range_rate are in meters and meters/s.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     x_inertial = np.array([7000e3, 0.0, 0.0, 0.0, 0.0, 0.0])
///     x_radec = bh.state_inertial_to_radec(x_inertial, bh.AngleFormat.DEGREES)
///     print(f"RA/Dec state: {x_radec}")
///     ```
fn py_state_inertial_to_radec<'py>(
    py: Python<'py>,
    x_inertial: Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec =
        coordinates::state_inertial_to_radec(pyany_to_svector::<6>(&x_inertial)?, angle_format.value);

    Ok(vector_to_numpy!(py, vec, 6, f64))
}

#[pyfunction]
#[pyo3(text_signature = "(x_radec, site_geodetic, epc, angle_format)")]
#[pyo3(name = "position_radec_to_azel")]
/// Convert a topocentric right ascension, declination, and range into the
/// equivalent azimuth, elevation, and range as seen from a given site.
///
/// This is a direction-only rotation of the line-of-sight unit vector: no
/// parallax translation between the geocenter and the site is applied, and
/// `range` passes through unchanged. The input `(ra, dec)` must already be
/// the direction from the site: for stars (effectively at infinite distance)
/// this is the same as the geocentric catalog `(ra, dec)`, but for satellites
/// or other nearby objects the caller must first compute the topocentric
/// right ascension/declination before calling this function.
///
/// Requires a global Earth orientation parameter (EOP) provider to be
/// initialized, as with all frame conversions between inertial and
/// Earth-fixed frames.
///
/// Args:
///     x_radec (numpy.ndarray or list): Topocentric right ascension, declination, and range
///         `[ra, dec, range]` where right ascension and declination are in radians or
///         degrees, and range is in meters.
///     site_geodetic (numpy.ndarray or list): Geodetic coordinates of the observing site
///         `[lon, lat, alt]` where longitude and latitude are in radians or degrees,
///         and altitude is in meters.
///     epc (Epoch): Epoch of the observation, used to rotate between the inertial and
///         Earth-fixed frames.
///     angle_format (AngleFormat): Angle format for angular elements (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: Azimuth (clockwise from North), elevation, and range
///         `[az, el, range]` where azimuth and elevation are in radians or degrees,
///         and range is in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_datetime(2024, 3, 20, 12, 0, 0.0, 0.0, bh.UTC)
///     site = np.array([-122.17, 37.43, 100.0])  # Stanford, deg/deg/m
///     x_radec = np.array([101.28, -16.72, 1.0])
///
///     # Requires a global EOP provider to be initialized first.
///     x_azel = bh.position_radec_to_azel(x_radec, site, epc, bh.AngleFormat.DEGREES)
///     ```
fn py_position_radec_to_azel<'py>(
    py: Python<'py>,
    x_radec: Bound<'py, PyAny>,
    site_geodetic: Bound<'py, PyAny>,
    epc: &PyEpoch,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = coordinates::position_radec_to_azel(
        pyany_to_svector::<3>(&x_radec)?,
        pyany_to_svector::<3>(&site_geodetic)?,
        epc.obj,
        angle_format.value,
    );

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

#[pyfunction]
#[pyo3(text_signature = "(x_azel, site_geodetic, epc, angle_format)")]
#[pyo3(name = "position_azel_to_radec")]
/// Convert an azimuth, elevation, and range as seen from a given site into
/// the equivalent topocentric right ascension, declination, and range.
///
/// This is the inverse of `position_radec_to_azel` and is likewise a
/// direction-only rotation: no parallax translation between the site and
/// the geocenter is applied, and `range` passes through unchanged. The
/// returned `(ra, dec)` is the topocentric direction as seen from the site,
/// which for stars is the same as the geocentric catalog `(ra, dec)`.
///
/// Requires a global Earth orientation parameter (EOP) provider to be
/// initialized, as with all frame conversions between inertial and
/// Earth-fixed frames.
///
/// Args:
///     x_azel (numpy.ndarray or list): Azimuth (clockwise from North), elevation, and
///         range `[az, el, range]` where azimuth and elevation are in radians or
///         degrees, and range is in meters.
///     site_geodetic (numpy.ndarray or list): Geodetic coordinates of the observing site
///         `[lon, lat, alt]` where longitude and latitude are in radians or degrees,
///         and altitude is in meters.
///     epc (Epoch): Epoch of the observation, used to rotate between the Earth-fixed
///         and inertial frames.
///     angle_format (AngleFormat): Angle format for angular elements (`RADIANS` or `DEGREES`).
///
/// Returns:
///     numpy.ndarray: Topocentric right ascension, declination, and range
///         `[ra, dec, range]` where right ascension and declination are in radians
///         or degrees, and range is in meters.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     epc = bh.Epoch.from_datetime(2024, 3, 20, 12, 0, 0.0, 0.0, bh.UTC)
///     site = np.array([-122.17, 37.43, 100.0])  # Stanford, deg/deg/m
///     x_azel = np.array([180.0, 45.0, 1.0])
///
///     # Requires a global EOP provider to be initialized first.
///     x_radec = bh.position_azel_to_radec(x_azel, site, epc, bh.AngleFormat.DEGREES)
///     ```
fn py_position_azel_to_radec<'py>(
    py: Python<'py>,
    x_azel: Bound<'py, PyAny>,
    site_geodetic: Bound<'py, PyAny>,
    epc: &PyEpoch,
    angle_format: &PyAngleFormat,
) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
    let vec = coordinates::position_azel_to_radec(
        pyany_to_svector::<3>(&x_azel)?,
        pyany_to_svector::<3>(&site_geodetic)?,
        epc.obj,
        angle_format.value,
    );

    Ok(vector_to_numpy!(py, vec, 3, f64))
}

#[pyfunction]
#[pyo3(text_signature = "(ra, dec, pm_ra, pm_dec, parallax, radial_velocity, epoch_from, epoch_to, angle_format)")]
#[pyo3(name = "apply_proper_motion")]
#[pyo3(signature = (ra, dec, pm_ra, pm_dec, parallax, radial_velocity, epoch_from, epoch_to, angle_format))]
/// Propagate a star's catalog position from one epoch to another using the
/// rigorous (direction-only) proper-motion transformation.
///
/// The star's unit direction vector is advanced linearly in the tangent plane
/// by its proper motion, scaled by a first-order perspective-acceleration
/// correction that accounts for the change in the star's angular rate as its
/// line-of-sight distance changes (significant for high radial-velocity,
/// high-parallax stars such as Barnard's Star), and then renormalized.
///
/// `pm_ra` follows the standard catalog convention: it is
/// mu_alpha* = mu_alpha * cos(dec), not the raw coordinate rate mu_alpha. This
/// matches the `pmra`/`pmdec` columns of Hipparcos, Gaia, and most other star
/// catalogs. If `parallax` or `radial_velocity` is `None`, the
/// perspective-acceleration term is omitted (equivalent to setting it to
/// zero), reducing to a purely linear proper-motion propagation.
///
/// This function implements the direction part of the transformation only;
/// it does not apply light-time or Doppler (radial-velocity-rate) corrections.
///
/// Args:
///     ra (float): Right ascension at `epoch_from`. Units: (radians or degrees)
///     dec (float): Declination at `epoch_from`. Units: (radians or degrees)
///     pm_ra (float): Proper motion in right ascension, mu_alpha* = mu_alpha * cos(dec).
///         Units: (mas/yr)
///     pm_dec (float): Proper motion in declination, mu_delta. Units: (mas/yr)
///     parallax (float or None): Annual parallax, or `None` if unknown/unavailable.
///         Units: (mas)
///     radial_velocity (float or None): Radial velocity, or `None` if unknown/unavailable.
///         Units: (km/s)
///     epoch_from (Epoch): Epoch of the input `(ra, dec)`.
///     epoch_to (Epoch): Epoch to propagate the position to.
///     angle_format (AngleFormat): Angle format for `ra`/`dec` input and output
///         (`RADIANS` or `DEGREES`).
///
/// Returns:
///     tuple[float, float]: Right ascension and declination propagated to `epoch_to`.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Barnard's Star (HIP 87937), J1991.25 Hipparcos catalog values.
///     epoch_from = bh.Epoch.from_mjd(48348.5625, bh.TimeSystem.TT)
///     epoch_to = bh.Epoch.from_mjd(48348.5625 + 10.0 * 365.25, bh.TimeSystem.TT)
///
///     ra, dec = bh.apply_proper_motion(
///         269.45402305,
///         4.66828815,
///         -797.84,
///         10326.93,
///         549.30,
///         -106.8,
///         epoch_from,
///         epoch_to,
///         bh.AngleFormat.DEGREES,
///     )
///     ```
#[allow(clippy::too_many_arguments)]
fn py_apply_proper_motion(
    ra: f64,
    dec: f64,
    pm_ra: f64,
    pm_dec: f64,
    parallax: Option<f64>,
    radial_velocity: Option<f64>,
    epoch_from: &PyEpoch,
    epoch_to: &PyEpoch,
    angle_format: &PyAngleFormat,
) -> (f64, f64) {
    coordinates::apply_proper_motion(
        ra,
        dec,
        pm_ra,
        pm_dec,
        parallax,
        radial_velocity,
        epoch_from.obj,
        epoch_to.obj,
        angle_format.value,
    )
}
