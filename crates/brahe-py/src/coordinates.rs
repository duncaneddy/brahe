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
///         Also accepts a batch of vectors with the 6 components along `axis`
///         (for example shape `(n, 6)`).
///     angle_format (AngleFormat): Angle format for angular elements (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_oe` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian state `[x, y, z, vx, vy, vz]` where position is in meters
///         and velocity is in meters per second.
///         For batched input the output takes the batch layout of `x_oe`.
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
#[pyfunction]
#[pyo3(signature = (x_oe, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_oe, angle_format, axis=-1)")]
#[pyo3(name = "state_koe_to_eci")]
fn py_state_koe_to_eci<'py>(
    py: Python<'py>,
    x_oe: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec::<6>(
        py,
        x_oe,
        axis,
        |v| coordinates::state_koe_to_eci(v, af),
        |vs| coordinates::states_koe_to_eci(vs, af),
    )
}

/// Convert Cartesian state to osculating orbital elements.
///
/// Transforms a state vector from Cartesian position and velocity coordinates to
/// osculating Keplerian orbital elements.
///
/// Args:
///     x_cart (numpy.ndarray or list): Cartesian state `[x, y, z, vx, vy, vz]` where position
///         is in meters and velocity is in meters per second.
///         Also accepts a batch of vectors with the 6 components along `axis`
///         (for example shape `(n, 6)`).
///     angle_format (AngleFormat): Angle format for output angular elements (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_cart` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Osculating orbital elements `[a, e, i, RAAN, omega, M]` where `a` is
///         semi-major axis (meters), `e` is eccentricity (dimensionless), `i` is inclination
///         (radians or degrees), `RAAN` is right ascension of ascending node (radians or degrees),
///         `omega` is argument of periapsis (radians or degrees), and `M` is mean anomaly
///         (radians or degrees).
///         For batched input the output takes the batch layout of `x_cart`.
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
#[pyfunction]
#[pyo3(signature = (x_cart, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_cart, angle_format, axis=-1)")]
#[pyo3(name = "state_eci_to_koe")]
fn py_state_eci_to_koe<'py>(
    py: Python<'py>,
    x_cart: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec::<6>(
        py,
        x_cart,
        axis,
        |v| coordinates::state_eci_to_koe(v, af),
        |vs| coordinates::states_eci_to_koe(vs, af),
    )
}

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
///         Also accepts a batch of vectors with the 6 components along `axis`
///         (for example shape `(n, 6)`).
///     central_body (CentralBody): Central body (supplies the GM and the IAU pole /
///         body-fixed frame).
///     angle_format (AngleFormat): Angle format for output angular elements (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_cart` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Osculating orbital elements `[a, e, i, RAAN, omega, M]` referenced
///         to the body mean equator at J2000.
///         For batched input the output takes the batch layout of `x_cart`.
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
#[pyfunction]
#[pyo3(signature = (x_cart, central_body, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_cart, central_body, angle_format, axis=-1)")]
#[pyo3(name = "state_inertial_to_koe_for_body")]
fn py_state_inertial_to_koe_for_body<'py>(
    py: Python<'py>,
    x_cart: &Bound<'py, PyAny>,
    central_body: &PyCentralBody,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    let body = &central_body.body;
    try_dispatch_vec::<6>(
        py,
        x_cart,
        axis,
        |v| {
            coordinates::state_inertial_to_koe_for_body(v, body, af)
                .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
        },
        |vs| {
            coordinates::states_inertial_to_koe_for_body(vs, body, af)
                .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
        },
    )
}

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
///         Also accepts a batch of vectors with the 6 components along `axis`
///         (for example shape `(n, 6)`).
///     central_body (CentralBody): Central body (supplies the GM and the IAU pole /
///         body-fixed frame).
///     angle_format (AngleFormat): Angle format for input angular elements (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_oe` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian state `[x, y, z, vx, vy, vz]` in the body-centered
///         ICRF-aligned frame. Units: (m; m/s)
///         For batched input the output takes the batch layout of `x_oe`.
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
#[pyfunction]
#[pyo3(signature = (x_oe, central_body, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_oe, central_body, angle_format, axis=-1)")]
#[pyo3(name = "state_koe_to_inertial_for_body")]
fn py_state_koe_to_inertial_for_body<'py>(
    py: Python<'py>,
    x_oe: &Bound<'py, PyAny>,
    central_body: &PyCentralBody,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    let body = &central_body.body;
    try_dispatch_vec::<6>(
        py,
        x_oe,
        axis,
        |v| {
            coordinates::state_koe_to_inertial_for_body(v, body, af)
                .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
        },
        |vs| {
            coordinates::states_koe_to_inertial_for_body(vs, body, af)
                .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))
        },
    )
}

/// Convert geocentric position to `ECEF` Cartesian coordinates.
///
/// Transforms a position from geocentric spherical coordinates (longitude, latitude, radius)
/// to Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates.
///
/// Args:
///     x_geoc (numpy.ndarray or list): Geocentric position `[longitude, latitude, radius]` where
///         longitude is in radians or degrees, latitude is in radians or degrees, and
///         radius is in meters.
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_geoc` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: `ECEF` Cartesian position `[x, y, z]` in meters.
///         For batched input the output takes the batch layout of `x_geoc`.
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
#[pyfunction]
#[pyo3(signature = (x_geoc, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_geoc, angle_format, axis=-1)")]
#[pyo3(name = "position_geocentric_to_ecef")]
fn py_position_geocentric_to_ecef<'py>(
    py: Python<'py>,
    x_geoc: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    try_dispatch_vec::<3>(
        py,
        x_geoc,
        axis,
        |v| coordinates::position_geocentric_to_ecef(v, af).map_err(exceptions::PyValueError::new_err),
        |vs| coordinates::positions_geocentric_to_ecef(vs, af).map_err(exceptions::PyValueError::new_err),
    )
}

/// Convert `ECEF` Cartesian position to geocentric coordinates.
///
/// Transforms a position from Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates
/// to geocentric spherical coordinates (longitude, latitude, radius).
///
/// Args:
///     x_ecef (numpy.ndarray or list): `ECEF` Cartesian position `[x, y, z]` in meters.
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_ecef` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Geocentric position `[longitude, latitude, radius]` where longitude
///         is in radians or degrees, latitude is in radians or degrees, and radius is in meters.
///         For batched input the output takes the batch layout of `x_ecef`.
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
#[pyfunction]
#[pyo3(signature = (x_ecef, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_ecef, angle_format, axis=-1)")]
#[pyo3(name = "position_ecef_to_geocentric")]
fn py_position_ecef_to_geocentric<'py>(
    py: Python<'py>,
    x_ecef: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec::<3>(
        py,
        x_ecef,
        axis,
        |v| coordinates::position_ecef_to_geocentric(v, af),
        |vs| coordinates::positions_ecef_to_geocentric(vs, af),
    )
}

/// Convert geodetic position to `ECEF` Cartesian coordinates.
///
/// Transforms a position from geodetic coordinates (longitude, latitude, altitude) using
/// the `WGS84` ellipsoid model to Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates.
///
/// Args:
///     x_geod (numpy.ndarray or list): Geodetic position `[longitude, latitude, altitude]` where
///         longitude is in radians or degrees, latitude is in radians or degrees, and
///         altitude is in meters above the `WGS84` ellipsoid.
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_geod` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: `ECEF` Cartesian position `[x, y, z]` in meters.
///         For batched input the output takes the batch layout of `x_geod`.
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
#[pyfunction]
#[pyo3(signature = (x_geod, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_geod, angle_format, axis=-1)")]
#[pyo3(name = "position_geodetic_to_ecef")]
fn py_position_geodetic_to_ecef<'py>(
    py: Python<'py>,
    x_geod: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    try_dispatch_vec::<3>(
        py,
        x_geod,
        axis,
        |v| coordinates::position_geodetic_to_ecef(v, af).map_err(exceptions::PyValueError::new_err),
        |vs| coordinates::positions_geodetic_to_ecef(vs, af).map_err(exceptions::PyValueError::new_err),
    )
}

/// Convert `ECEF` Cartesian position to geodetic coordinates.
///
/// Transforms a position from Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates
/// to geodetic coordinates (longitude, latitude, altitude) using the `WGS84` ellipsoid model.
///
/// Args:
///     x_ecef (numpy.ndarray or list): `ECEF` Cartesian position `[x, y, z]` in meters.
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_ecef` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Geodetic position `[longitude, latitude, altitude]` where longitude
///         is in radians or degrees, latitude is in radians or degrees, and altitude
///         is in meters above the `WGS84` ellipsoid.
///         For batched input the output takes the batch layout of `x_ecef`.
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
#[pyfunction]
#[pyo3(signature = (x_ecef, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_ecef, angle_format, axis=-1)")]
#[pyo3(name = "position_ecef_to_geodetic")]
fn py_position_ecef_to_geodetic<'py>(
    py: Python<'py>,
    x_ecef: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec::<3>(
        py,
        x_ecef,
        axis,
        |v| coordinates::position_ecef_to_geodetic(v, af),
        |vs| coordinates::positions_ecef_to_geodetic(vs, af),
    )
}


/// Compute rotation matrix from ellipsoidal coordinates to East-North-Up (`ENZ`) frame.
///
/// Calculates the rotation matrix that transforms vectors from an ellipsoidal coordinate
/// frame (geocentric or geodetic) to the local East-North-Up (`ENZ`) topocentric frame at
/// the specified location.
///
/// Args:
///     x_ellipsoid (numpy.ndarray or list): Ellipsoidal position `[latitude, longitude, altitude/radius]`
///         where latitude is in radians or degrees, longitude is in radians or degrees.
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_ellipsoid` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix from ellipsoidal frame to `ENZ` frame.
///         For batched input the batch dimensions are followed by `(3, 3)`.
#[pyfunction]
#[pyo3(signature = (x_ellipsoid, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_ellipsoid, angle_format, axis=-1)")]
#[pyo3(name = "rotation_ellipsoid_to_enz")]
fn py_rotation_ellipsoid_to_enz<'py>(
    py: Python<'py>,
    x_ellipsoid: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec_rotation::<3>(
        py,
        x_ellipsoid,
        axis,
        |v| coordinates::rotation_ellipsoid_to_enz(v, af),
        |vs| coordinates::rotations_ellipsoid_to_enz(vs, af),
    )
}

/// Compute rotation matrix from East-North-Up (`ENZ`) frame to ellipsoidal coordinates.
///
/// Calculates the rotation matrix that transforms vectors from the local East-North-Up
/// (`ENZ`) topocentric frame to an ellipsoidal coordinate frame (geocentric or geodetic)
/// at the specified location.
///
/// Args:
///     x_ellipsoid (numpy.ndarray or list): Ellipsoidal position `[latitude, longitude, altitude/radius]`
///         where latitude is in radians or degrees, longitude is in radians or degrees.
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_ellipsoid` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix from `ENZ` frame to ellipsoidal frame.
///         For batched input the batch dimensions are followed by `(3, 3)`.
#[pyfunction]
#[pyo3(signature = (x_ellipsoid, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_ellipsoid, angle_format, axis=-1)")]
#[pyo3(name = "rotation_enz_to_ellipsoid")]
fn py_rotation_enz_to_ellipsoid<'py>(
    py: Python<'py>,
    x_ellipsoid: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec_rotation::<3>(
        py,
        x_ellipsoid,
        axis,
        |v| coordinates::rotation_enz_to_ellipsoid(v, af),
        |vs| coordinates::rotations_enz_to_ellipsoid(vs, af),
    )
}

/// Convert relative position from `ECEF` to East-North-Up (`ENZ`) frame.
///
/// Transforms a relative position vector from Earth-Centered Earth-Fixed (`ECEF`) coordinates
/// to the local East-North-Up (`ENZ`) topocentric frame at the specified location.
///
/// Args:
///     location_ecef (numpy.ndarray or list): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
///         Also accepts a batch of locations with the 3 components along `axis`;
///         a single location is broadcast across a batch of positions and vice versa.
///     r_ecef (numpy.ndarray or list): Position vector in `ECEF` coordinates `[x, y, z]` in meters.
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     conversion_type (EllipsoidalConversionType): Type of ellipsoidal conversion (`GEOCENTRIC` or `GEODETIC`).
///     axis (int, optional): The axis of `location_ecef` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Relative position in `ENZ` frame `[east, north, up]` in meters.
///         For batched input the output takes the layout of the batched argument.
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
#[pyfunction]
#[pyo3(signature = (location_ecef, r_ecef, conversion_type, axis=-1))]
#[pyo3(text_signature = "(location_ecef, r_ecef, conversion_type, axis=-1)")]
#[pyo3(name = "relative_position_ecef_to_enz")]
fn py_relative_position_ecef_to_enz<'py>(
    py: Python<'py>,
    location_ecef: &Bound<'py, PyAny>,
    r_ecef: &Bound<'py, PyAny>,
    conversion_type: &PyEllipsoidalConversionType,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let ct = conversion_type.value;
    dispatch_vec_pair::<3>(
        py,
        location_ecef,
        r_ecef,
        axis,
        |l, x| coordinates::relative_position_ecef_to_enz(l, x, ct),
        |ls, xs| coordinates::relative_positions_ecef_to_enz(ls, xs, ct),
    )
}

/// Convert relative position from East-North-Up (`ENZ`) frame to `ECEF`.
///
/// Transforms a relative position vector from the local East-North-Up (`ENZ`) topocentric
/// frame to Earth-Centered Earth-Fixed (`ECEF`) coordinates at the specified location.
///
/// Args:
///     location_ecef (numpy.ndarray or list): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
///         Also accepts a batch of locations with the 3 components along `axis`;
///         a single location is broadcast across a batch of positions and vice versa.
///     r_enz (numpy.ndarray or list): Relative position in `ENZ` frame `[east, north, up]` in meters.
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     conversion_type (EllipsoidalConversionType): Type of ellipsoidal conversion (`GEOCENTRIC` or `GEODETIC`).
///     axis (int, optional): The axis of `location_ecef` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Position vector in `ECEF` coordinates `[x, y, z]` in meters.
///         For batched input the output takes the layout of the batched argument.
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
#[pyfunction]
#[pyo3(signature = (location_ecef, r_enz, conversion_type, axis=-1))]
#[pyo3(text_signature = "(location_ecef, r_enz, conversion_type, axis=-1)")]
#[pyo3(name = "relative_position_enz_to_ecef")]
fn py_relative_position_enz_to_ecef<'py>(
    py: Python<'py>,
    location_ecef: &Bound<'py, PyAny>,
    r_enz: &Bound<'py, PyAny>,
    conversion_type: &PyEllipsoidalConversionType,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let ct = conversion_type.value;
    dispatch_vec_pair::<3>(
        py,
        location_ecef,
        r_enz,
        axis,
        |l, x| coordinates::relative_position_enz_to_ecef(l, x, ct),
        |ls, xs| coordinates::relative_positions_enz_to_ecef(ls, xs, ct),
    )
}

/// Compute rotation matrix from ellipsoidal coordinates to South-East-Zenith (`SEZ`) frame.
///
/// Calculates the rotation matrix that transforms vectors from an ellipsoidal coordinate
/// frame (geocentric or geodetic) to the local South-East-Zenith (`SEZ`) topocentric frame
/// at the specified location.
///
/// Args:
///     x_ellipsoid (numpy.ndarray or list): Ellipsoidal position `[latitude, longitude, altitude/radius]`
///         where latitude is in radians or degrees, longitude is in radians or degrees.
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_ellipsoid` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix from ellipsoidal frame to `SEZ` frame.
///         For batched input the batch dimensions are followed by `(3, 3)`.
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
#[pyfunction]
#[pyo3(signature = (x_ellipsoid, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_ellipsoid, angle_format, axis=-1)")]
#[pyo3(name = "rotation_ellipsoid_to_sez")]
fn py_rotation_ellipsoid_to_sez<'py>(
    py: Python<'py>,
    x_ellipsoid: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec_rotation::<3>(
        py,
        x_ellipsoid,
        axis,
        |v| coordinates::rotation_ellipsoid_to_sez(v, af),
        |vs| coordinates::rotations_ellipsoid_to_sez(vs, af),
    )
}

/// Compute rotation matrix from South-East-Zenith (`SEZ`) frame to ellipsoidal coordinates.
///
/// Calculates the rotation matrix that transforms vectors from the local South-East-Zenith
/// (`SEZ`) topocentric frame to an ellipsoidal coordinate frame (geocentric or geodetic)
/// at the specified location.
///
/// Args:
///     x_ellipsoid (numpy.ndarray or list): Ellipsoidal position `[latitude, longitude, altitude/radius]`
///         where latitude is in radians or degrees, longitude is in radians or degrees.
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_ellipsoid` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: 3x3 rotation matrix from `SEZ` frame to ellipsoidal frame.
///         For batched input the batch dimensions are followed by `(3, 3)`.
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
#[pyfunction]
#[pyo3(signature = (x_ellipsoid, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_ellipsoid, angle_format, axis=-1)")]
#[pyo3(name = "rotation_sez_to_ellipsoid")]
fn py_rotation_sez_to_ellipsoid<'py>(
    py: Python<'py>,
    x_ellipsoid: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec_rotation::<3>(
        py,
        x_ellipsoid,
        axis,
        |v| coordinates::rotation_sez_to_ellipsoid(v, af),
        |vs| coordinates::rotations_sez_to_ellipsoid(vs, af),
    )
}

/// Convert relative position from `ECEF` to South-East-Zenith (`SEZ`) frame.
///
/// Transforms a relative position vector from Earth-Centered Earth-Fixed (`ECEF`) coordinates
/// to the local South-East-Zenith (`SEZ`) topocentric frame at the specified location.
///
/// Args:
///     location_ecef (numpy.ndarray or list): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
///         Also accepts a batch of locations with the 3 components along `axis`;
///         a single location is broadcast across a batch of positions and vice versa.
///     r_ecef (numpy.ndarray or list): Position vector in `ECEF` coordinates `[x, y, z]` in meters.
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     conversion_type (EllipsoidalConversionType): Type of ellipsoidal conversion (`GEOCENTRIC` or `GEODETIC`).
///     axis (int, optional): The axis of `location_ecef` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Relative position in `SEZ` frame `[south, east, zenith]` in meters.
///         For batched input the output takes the layout of the batched argument.
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
#[pyfunction]
#[pyo3(signature = (location_ecef, r_ecef, conversion_type, axis=-1))]
#[pyo3(text_signature = "(location_ecef, r_ecef, conversion_type, axis=-1)")]
#[pyo3(name = "relative_position_ecef_to_sez")]
fn py_relative_position_ecef_to_sez<'py>(
    py: Python<'py>,
    location_ecef: &Bound<'py, PyAny>,
    r_ecef: &Bound<'py, PyAny>,
    conversion_type: &PyEllipsoidalConversionType,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let ct = conversion_type.value;
    dispatch_vec_pair::<3>(
        py,
        location_ecef,
        r_ecef,
        axis,
        |l, x| coordinates::relative_position_ecef_to_sez(l, x, ct),
        |ls, xs| coordinates::relative_positions_ecef_to_sez(ls, xs, ct),
    )
}

/// Convert relative position from South-East-Zenith (`SEZ`) frame to `ECEF`.
///
/// Transforms a relative position vector from the local South-East-Zenith (`SEZ`) topocentric
/// frame to Earth-Centered Earth-Fixed (`ECEF`) coordinates at the specified location.
///
/// Args:
///     location_ecef (numpy.ndarray or list): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
///         Also accepts a batch of locations with the 3 components along `axis`;
///         a single location is broadcast across a batch of positions and vice versa.
///     x_sez (numpy.ndarray or list): Relative position in `SEZ` frame `[south, east, zenith]` in meters.
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     conversion_type (EllipsoidalConversionType): Type of ellipsoidal conversion (`GEOCENTRIC` or `GEODETIC`).
///     axis (int, optional): The axis of `location_ecef` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Position vector in `ECEF` coordinates `[x, y, z]` in meters.
///         For batched input the output takes the layout of the batched argument.
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
#[pyfunction]
#[pyo3(signature = (location_ecef, x_sez, conversion_type, axis=-1))]
#[pyo3(text_signature = "(location_ecef, x_sez, conversion_type, axis=-1)")]
#[pyo3(name = "relative_position_sez_to_ecef")]
fn py_relative_position_sez_to_ecef<'py>(
    py: Python<'py>,
    location_ecef: &Bound<'py, PyAny>,
    x_sez: &Bound<'py, PyAny>,
    conversion_type: &PyEllipsoidalConversionType,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let ct = conversion_type.value;
    dispatch_vec_pair::<3>(
        py,
        location_ecef,
        x_sez,
        axis,
        |l, x| coordinates::relative_position_sez_to_ecef(l, x, ct),
        |ls, xs| coordinates::relative_positions_sez_to_ecef(ls, xs, ct),
    )
}

/// Convert position from East-North-Up (`ENZ`) frame to azimuth-elevation-range.
///
/// Transforms a position from the local East-North-Up (`ENZ`) topocentric frame to
/// azimuth-elevation-range spherical coordinates.
///
/// Args:
///     x_enz (numpy.ndarray or list): Position in `ENZ` frame `[east, north, up]` in meters.
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_enz` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Azimuth-elevation-range `[azimuth, elevation, range]` where azimuth
///         and elevation are in radians or degrees, and range is in meters.
///         For batched input the output takes the batch layout of `x_enz`.
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
#[pyfunction]
#[pyo3(signature = (x_enz, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_enz, angle_format, axis=-1)")]
#[pyo3(name = "position_enz_to_azel")]
fn py_position_enz_to_azel<'py>(
    py: Python<'py>,
    x_enz: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec::<3>(
        py,
        x_enz,
        axis,
        |v| coordinates::position_enz_to_azel(v, af),
        |vs| coordinates::positions_enz_to_azel(vs, af),
    )
}

/// Convert position from South-East-Zenith (`SEZ`) frame to azimuth-elevation-range.
///
/// Transforms a position from the local South-East-Zenith (`SEZ`) topocentric frame to
/// azimuth-elevation-range spherical coordinates.
///
/// Args:
///     x_sez (numpy.ndarray or list): Position in `SEZ` frame `[south, east, zenith]` in meters.
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_sez` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Azimuth-elevation-range `[azimuth, elevation, range]` where azimuth
///         and elevation are in radians or degrees, and range is in meters.
///         For batched input the output takes the batch layout of `x_sez`.
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
#[pyfunction]
#[pyo3(signature = (x_sez, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_sez, angle_format, axis=-1)")]
#[pyo3(name = "position_sez_to_azel")]
fn py_position_sez_to_azel<'py>(
    py: Python<'py>,
    x_sez: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec::<3>(
        py,
        x_sez,
        axis,
        |v| coordinates::position_sez_to_azel(v, af),
        |vs| coordinates::positions_sez_to_azel(vs, af),
    )
}

/// Convert a right ascension, declination, and range into the equivalent
/// Cartesian inertial position.
///
/// Args:
///     x_radec (numpy.ndarray or list): Right ascension, declination, and range
///         `[ra, dec, range]` where right ascension and declination are in
///         radians or degrees, and range is in meters.
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     angle_format (AngleFormat): Angle format for angular elements (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_radec` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian inertial position `[x, y, z]` in meters.
///         For batched input the output takes the batch layout of `x_radec`.
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
#[pyfunction]
#[pyo3(signature = (x_radec, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_radec, angle_format, axis=-1)")]
#[pyo3(name = "position_radec_to_inertial")]
fn py_position_radec_to_inertial<'py>(
    py: Python<'py>,
    x_radec: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec::<3>(
        py,
        x_radec,
        axis,
        |v| coordinates::position_radec_to_inertial(v, af),
        |vs| coordinates::positions_radec_to_inertial(vs, af),
    )
}

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
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     angle_format (AngleFormat): Angle format for angular output (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_inertial` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Right ascension, declination, and range `[ra, dec, range]` where
///         right ascension and declination are in radians or degrees, and range is in meters.
///         For batched input the output takes the batch layout of `x_inertial`.
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
#[pyfunction]
#[pyo3(signature = (x_inertial, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_inertial, angle_format, axis=-1)")]
#[pyo3(name = "position_inertial_to_radec")]
fn py_position_inertial_to_radec<'py>(
    py: Python<'py>,
    x_inertial: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec::<3>(
        py,
        x_inertial,
        axis,
        |v| coordinates::position_inertial_to_radec(v, af),
        |vs| coordinates::positions_inertial_to_radec(vs, af),
    )
}

/// Convert a right ascension, declination, range, and their rates into the
/// equivalent Cartesian inertial position and velocity.
///
/// Args:
///     x_radec (numpy.ndarray or list): Right ascension, declination, range, and rates
///         `[ra, dec, range, ra_rate, dec_rate, range_rate]` where right ascension,
///         declination, and their rates are in radians (or radians/s) or degrees
///         (or degrees/s), and range/range_rate are in meters and meters/s.
///         Also accepts a batch of vectors with the 6 components along `axis`
///         (for example shape `(n, 6)`).
///     angle_format (AngleFormat): Angle format for angular elements and rates (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_radec` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Cartesian inertial position and velocity `[x, y, z, vx, vy, vz]`
///         in meters and meters per second.
///         For batched input the output takes the batch layout of `x_radec`.
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
#[pyfunction]
#[pyo3(signature = (x_radec, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_radec, angle_format, axis=-1)")]
#[pyo3(name = "state_radec_to_inertial")]
fn py_state_radec_to_inertial<'py>(
    py: Python<'py>,
    x_radec: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec::<6>(
        py,
        x_radec,
        axis,
        |v| coordinates::state_radec_to_inertial(v, af),
        |vs| coordinates::states_radec_to_inertial(vs, af),
    )
}

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
///         Also accepts a batch of vectors with the 6 components along `axis`
///         (for example shape `(n, 6)`).
///     angle_format (AngleFormat): Angle format for angular output and rates (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_inertial` along which the 6 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 6)` the components lie along the last axis, so the default `-1`
///         applies; a `(6, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Right ascension, declination, range, and rates
///         `[ra, dec, range, ra_rate, dec_rate, range_rate]` where right ascension,
///         declination, and their rates are in radians (or radians/s) or degrees
///         (or degrees/s), and range/range_rate are in meters and meters/s.
///         For batched input the output takes the batch layout of `x_inertial`.
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
#[pyfunction]
#[pyo3(signature = (x_inertial, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_inertial, angle_format, axis=-1)")]
#[pyo3(name = "state_inertial_to_radec")]
fn py_state_inertial_to_radec<'py>(
    py: Python<'py>,
    x_inertial: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    dispatch_vec::<6>(
        py,
        x_inertial,
        axis,
        |v| coordinates::state_inertial_to_radec(v, af),
        |vs| coordinates::states_inertial_to_radec(vs, af),
    )
}

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
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     site_geodetic (numpy.ndarray or list): Geodetic coordinates of the observing site
///         `[lon, lat, alt]` where longitude and latitude are in radians or degrees,
///         and altitude is in meters.
///     epc (Epoch or Sequence[Epoch]): Epoch of the observation, used to rotate between the inertial and
///         Earth-fixed frames.
///         A sequence evaluates one epoch per vector (or broadcasts a single
///         vector across all epochs).
///     angle_format (AngleFormat): Angle format for angular elements (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_radec` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Azimuth (clockwise from North), elevation, and range
///         `[az, el, range]` where azimuth and elevation are in radians or degrees,
///         and range is in meters.
///         For batched input the output takes the batch layout of `x_radec` (shape
///         `(n, 3)` for a single vector with a sequence of `n` epochs).
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
#[pyfunction]
#[pyo3(signature = (x_radec, site_geodetic, epc, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_radec, site_geodetic, epc, angle_format, axis=-1)")]
#[pyo3(name = "position_radec_to_azel")]
fn py_position_radec_to_azel<'py>(
    py: Python<'py>,
    x_radec: &Bound<'py, PyAny>,
    site_geodetic: &Bound<'py, PyAny>,
    epc: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    let site = pyany_to_svector::<3>(site_geodetic)?;
    dispatch_epoch_vec::<3>(
        py,
        epc,
        x_radec,
        axis,
        |e, v| coordinates::position_radec_to_azel(v, site, e, af),
        |es, vs| coordinates::positions_radec_to_azel(vs, site, es, af),
    )
}

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
///         Also accepts a batch of vectors with the 3 components along `axis`
///         (for example shape `(n, 3)`).
///     site_geodetic (numpy.ndarray or list): Geodetic coordinates of the observing site
///         `[lon, lat, alt]` where longitude and latitude are in radians or degrees,
///         and altitude is in meters.
///     epc (Epoch or Sequence[Epoch]): Epoch of the observation, used to rotate between the Earth-fixed
///         and inertial frames.
///         A sequence evaluates one epoch per vector (or broadcasts a single
///         vector across all epochs).
///     angle_format (AngleFormat): Angle format for angular elements (`RADIANS` or `DEGREES`).
///     axis (int, optional): The axis of `x_azel` along which the 3 components of a
///         single vector lie; the remaining axes enumerate the batch. For a batch of
///         shape `(n, 3)` the components lie along the last axis, so the default `-1`
///         applies; a `(3, n)` column layout uses `axis=0`.
///
/// Returns:
///     numpy.ndarray: Topocentric right ascension, declination, and range
///         `[ra, dec, range]` where right ascension and declination are in radians
///         or degrees, and range is in meters.
///         For batched input the output takes the batch layout of `x_azel` (shape
///         `(n, 3)` for a single vector with a sequence of `n` epochs).
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
#[pyfunction]
#[pyo3(signature = (x_azel, site_geodetic, epc, angle_format, axis=-1))]
#[pyo3(text_signature = "(x_azel, site_geodetic, epc, angle_format, axis=-1)")]
#[pyo3(name = "position_azel_to_radec")]
fn py_position_azel_to_radec<'py>(
    py: Python<'py>,
    x_azel: &Bound<'py, PyAny>,
    site_geodetic: &Bound<'py, PyAny>,
    epc: &Bound<'py, PyAny>,
    angle_format: &PyAngleFormat,
    axis: isize,
) -> PyResult<Bound<'py, PyAny>> {
    let af = angle_format.value;
    let site = pyany_to_svector::<3>(site_geodetic)?;
    dispatch_epoch_vec::<3>(
        py,
        epc,
        x_azel,
        axis,
        |e, v| coordinates::position_azel_to_radec(v, site, e, af),
        |es, vs| coordinates::positions_azel_to_radec(vs, site, es, af),
    )
}

#[pyfunction]
#[pyo3(text_signature = "(ra, dec, pm_ra, pm_dec, parallax, radial_velocity, epoch_from, epoch_to, angle_format)")]
#[pyo3(name = "apply_proper_motion")]
#[pyo3(signature = (ra, dec, pm_ra, pm_dec, parallax, radial_velocity, epoch_from, epoch_to, angle_format))]
/// Propagate a star's catalog position from one epoch to another using IAU
/// SOFA's `iauPmsafe` space-motion transformation.
///
/// `iauPmsafe` reconstructs the star's full barycentric position/velocity
/// state from the catalog `(ra, dec)`, proper motion, parallax, and radial
/// velocity; advances it assuming straight-line motion at constant velocity
/// (including a light-time correction and the special-relativistic Doppler
/// treatment of Stumpff, 1985); and reduces the result back to catalog
/// `(ra, dec)` at `epoch_to`. To first order this is the rigorous
/// direction-only epoch transformation of ESA SP-1200 (Vol. 1, §1.5.5): a
/// tangential proper-motion displacement plus a radial "perspective
/// acceleration" term that is significant for high radial-velocity,
/// high-parallax stars such as Barnard's Star.
///
/// `pm_ra` follows the standard catalog convention: it is
/// mu_alpha* = mu_alpha * cos(dec), not the raw coordinate rate mu_alpha. This
/// matches the `pmra`/`pmdec` columns of Hipparcos, Gaia, and most other star
/// catalogs. If `parallax` or `radial_velocity` is `None` it is treated as
/// zero; `iauPmsafe` additionally applies a proper-motion-scaled
/// minimum-parallax guard so a star with a missing or tiny parallax still
/// propagates correctly rather than being clamped to a no-op.
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
