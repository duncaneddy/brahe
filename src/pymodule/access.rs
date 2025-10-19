// Access module Python bindings
//
// This file contains Python bindings for the access computation module,
// including constraints, locations, and access window finding.

use crate::access::constraints::{
    AccessConstraint, AscDsc, AscDscConstraint, ConstraintComposite, ElevationConstraint,
    ElevationMaskConstraint, LocalTimeConstraint, LookDirection, LookDirectionConstraint,
    OffNadirConstraint,
};
use crate::access::location::{PointLocation, PolygonLocation};
use crate::utils::identifiable::Identifiable;
use nalgebra::Vector3;
use pyo3::types::PyDict;

// ================================
// Properties Dict Wrapper
// ================================

/// A dictionary-like wrapper for Location properties that supports dict-style assignment.
///
/// This class provides a Pythonic dict interface for accessing and modifying location properties.
/// Changes are automatically synchronized with the underlying Location object.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     loc = bh.PointLocation(15.4, 78.2, 0.0)
///
///     # Dict-style assignment
///     loc.properties["climate"] = "Arctic"
///     loc.properties["country"] = "Norway"
///
///     # Dict-style access
///     climate = loc.properties["climate"]
///
///     # Dict methods work
///     if "country" in loc.properties:
///         print(loc.properties["country"])
///
///     # Iteration
///     for key in loc.properties:
///         print(key, loc.properties[key])
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "PropertiesDict")]
pub struct PyPropertiesDict {
    parent: Py<PyAny>,
}

#[pymethods]
impl PyPropertiesDict {
    /// Get a property value by key.
    fn __getitem__(&self, key: String, py: Python) -> PyResult<Py<PyAny>> {
        let props_dict = self.get_properties_dict(py)?;
        props_dict
            .get_item(&key)?
            .ok_or_else(|| exceptions::PyKeyError::new_err(format!("Key '{}' not found", key)))
            .map(|item| item.into())
    }

    /// Set a property value by key.
    fn __setitem__(&self, key: String, value: &Bound<'_, PyAny>, py: Python) -> PyResult<()> {
        // Convert Python value to JSON via serialization
        let json_module = py.import("json")?;
        let dumps = json_module.getattr("dumps")?;
        let json_str: String = dumps.call1((value,))?.extract()?;

        let json_value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| exceptions::PyValueError::new_err(format!("JSON error: {}", e)))?;

        // Update the parent's properties
        self.set_property(py, key, json_value)?;
        Ok(())
    }

    /// Delete a property by key.
    fn __delitem__(&self, key: String, py: Python) -> PyResult<()> {
        self.remove_property(py, key)
    }

    /// Return the number of properties.
    fn __len__(&self, py: Python) -> PyResult<usize> {
        let props_dict = self.get_properties_dict(py)?;
        Ok(props_dict.len())
    }

    /// Check if a key exists in properties.
    fn __contains__(&self, key: String, py: Python) -> PyResult<bool> {
        let props_dict = self.get_properties_dict(py)?;
        props_dict.contains(&key)
    }

    /// Return an iterator over property keys.
    fn __iter__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let props_dict = self.get_properties_dict(py)?;
        Ok(props_dict.call_method0("__iter__")?.into())
    }

    /// String representation.
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let props_dict = self.get_properties_dict(py)?;
        Ok(format!("PropertiesDict({})", props_dict.repr()?))
    }

    /// Get property value with optional default.
    #[pyo3(signature = (key, default=None))]
    fn get(&self, key: String, default: Option<Py<PyAny>>, py: Python) -> PyResult<Py<PyAny>> {
        let props_dict = self.get_properties_dict(py)?;
        match props_dict.get_item(&key)? {
            Some(value) => Ok(value.into()),
            None => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    /// Return a list of property keys.
    fn keys(&self, py: Python) -> PyResult<Py<PyAny>> {
        let props_dict = self.get_properties_dict(py)?;
        Ok(props_dict.call_method0("keys")?.into())
    }

    /// Return a list of property values.
    fn values(&self, py: Python) -> PyResult<Py<PyAny>> {
        let props_dict = self.get_properties_dict(py)?;
        Ok(props_dict.call_method0("values")?.into())
    }

    /// Return a list of (key, value) tuples.
    fn items(&self, py: Python) -> PyResult<Py<PyAny>> {
        let props_dict = self.get_properties_dict(py)?;
        Ok(props_dict.call_method0("items")?.into())
    }

    /// Remove all properties.
    fn clear(&self, py: Python) -> PyResult<()> {
        // Get all keys first (to avoid modifying during iteration)
        let props_dict = self.get_properties_dict(py)?;
        let keys_list = props_dict.call_method0("keys")?;
        let keys: Vec<String> = keys_list.extract()?;

        // Remove each key
        for key in keys {
            self.remove_property(py, key)?;
        }
        Ok(())
    }

    /// Update properties from another dict.
    fn update(&self, other: &Bound<'_, PyDict>, py: Python) -> PyResult<()> {
        for (key, value) in other.iter() {
            let key_str: String = key.extract()?;
            self.__setitem__(key_str, &value, py)?;
        }
        Ok(())
    }
}

impl PyPropertiesDict {
    /// Create a new PropertiesDict wrapping a parent Location.
    fn new(parent: Py<PyAny>) -> Self {
        Self { parent }
    }

    /// Get the properties as a Python dict.
    fn get_properties_dict<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        // Call the parent's internal _get_properties_dict method
        let parent_obj = self.parent.bind(py);
        let props_obj = parent_obj.call_method0("_get_properties_dict")?;
        props_obj.downcast::<PyDict>().cloned().map_err(|e| e.into())
    }

    /// Set a property on the parent Location.
    fn set_property(&self, py: Python, key: String, value: serde_json::Value) -> PyResult<()> {
        let parent_obj = self.parent.bind(py);
        parent_obj.call_method1("_set_property", (key, value.to_string()))?;
        Ok(())
    }

    /// Remove a property from the parent Location.
    fn remove_property(&self, py: Python, key: String) -> PyResult<()> {
        let parent_obj = self.parent.bind(py);
        parent_obj.call_method1("_remove_property", (key,))?;
        Ok(())
    }
}

// ================================
// Enums
// ================================

/// Look direction of a satellite relative to its velocity vector.
///
/// Indicates whether a satellite is looking to the left (counterclockwise from velocity),
/// right (clockwise from velocity), or either direction.
///
/// This is commonly used for imaging satellites with side-looking sensors or SAR systems
/// that have a preferred look direction.
///
/// Attributes:
///     LEFT: Left-looking (counterclockwise from velocity vector)
///     RIGHT: Right-looking (clockwise from velocity vector)
///     EITHER: Either left or right is acceptable
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create a constraint for right-looking only satellites
///     constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.RIGHT)
///
///     # Create a constraint accepting either direction
///     constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.EITHER)
///
///     # Compare look directions
///     assert bh.LookDirection.LEFT != bh.LookDirection.RIGHT
///     assert bh.LookDirection.LEFT == bh.LookDirection.LEFT
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "LookDirection")]
#[derive(Clone)]
pub struct PyLookDirection {
    pub(crate) value: LookDirection,
}

#[pymethods]
impl PyLookDirection {
    /// Left-looking (counterclockwise from velocity vector)
    #[classattr]
    #[allow(non_snake_case)]
    fn LEFT() -> Self {
        PyLookDirection {
            value: LookDirection::Left,
        }
    }

    /// Right-looking (clockwise from velocity vector)
    #[classattr]
    #[allow(non_snake_case)]
    fn RIGHT() -> Self {
        PyLookDirection {
            value: LookDirection::Right,
        }
    }

    /// Either left or right
    #[classattr]
    #[allow(non_snake_case)]
    fn EITHER() -> Self {
        PyLookDirection {
            value: LookDirection::Either,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.value)
    }

    fn __repr__(&self) -> String {
        format!("LookDirection.{:?}", self.value)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.value == other.value),
            CompareOp::Ne => Ok(self.value != other.value),
            _ => Err(exceptions::PyNotImplementedError::new_err(
                "Comparison not supported",
            )),
        }
    }
}

/// Ascending or descending pass type for satellite orbits.
///
/// Indicates whether a satellite is moving from south to north (ascending) or
/// north to south (descending) in its orbit. This is determined by the sign of
/// the Z-component of the velocity vector in ECEF coordinates.
///
/// This is useful for:
/// - Sun-synchronous orbits that prefer specific pass types
/// - Minimizing lighting variation between passes
/// - Coordinating multi-satellite observations
///
/// Attributes:
///     ASCENDING: Satellite moving from south to north (vz > 0 in ECEF)
///     DESCENDING: Satellite moving from north to south (vz < 0 in ECEF)
///     EITHER: Either ascending or descending is acceptable
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create a constraint for ascending passes only
///     constraint = bh.AscDscConstraint(allowed=bh.AscDsc.ASCENDING)
///
///     # Create a constraint for descending passes only
///     constraint = bh.AscDscConstraint(allowed=bh.AscDsc.DESCENDING)
///
///     # Accept either type
///     constraint = bh.AscDscConstraint(allowed=bh.AscDsc.EITHER)
///
///     # Compare pass types
///     assert bh.AscDsc.ASCENDING != bh.AscDsc.DESCENDING
///     assert bh.AscDsc.ASCENDING == bh.AscDsc.ASCENDING
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "AscDsc")]
#[derive(Clone)]
pub struct PyAscDsc {
    pub(crate) value: AscDsc,
}

#[pymethods]
impl PyAscDsc {
    /// Ascending pass (moving from south to north)
    #[classattr]
    #[allow(non_snake_case)]
    fn ASCENDING() -> Self {
        PyAscDsc {
            value: AscDsc::Ascending,
        }
    }

    /// Descending pass (moving from north to south)
    #[classattr]
    #[allow(non_snake_case)]
    fn DESCENDING() -> Self {
        PyAscDsc {
            value: AscDsc::Descending,
        }
    }

    /// Either ascending or descending
    #[classattr]
    #[allow(non_snake_case)]
    fn EITHER() -> Self {
        PyAscDsc {
            value: AscDsc::Either,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.value)
    }

    fn __repr__(&self) -> String {
        format!("AscDsc.{:?}", self.value)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.value == other.value),
            CompareOp::Ne => Ok(self.value != other.value),
            _ => Err(exceptions::PyNotImplementedError::new_err(
                "Comparison not supported",
            )),
        }
    }
}

// ================================
// Constraint Classes
// ================================

/// Elevation angle constraint for satellite visibility.
///
/// Constrains access based on the elevation angle of the satellite above
/// the local horizon at the ground location.
///
/// Args:
///     min_elevation_deg (float | None): Minimum elevation angle in degrees, or None for no minimum
///     max_elevation_deg (float | None): Maximum elevation angle in degrees, or None for no maximum
///
/// Raises:
///     ValueError: If both min and max are None (unbounded constraint is meaningless)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Typical ground station constraint: 5° minimum elevation
///     constraint = bh.ElevationConstraint(min_elevation_deg=5.0, max_elevation_deg=None)
///
///     # Both bounds specified
///     constraint = bh.ElevationConstraint(min_elevation_deg=5.0, max_elevation_deg=85.0)
///
///     # Only maximum (e.g., avoid zenith)
///     constraint = bh.ElevationConstraint(min_elevation_deg=None, max_elevation_deg=85.0)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "ElevationConstraint")]
pub struct PyElevationConstraint {
    pub(crate) constraint: ElevationConstraint,
}

#[pymethods]
impl PyElevationConstraint {
    #[new]
    #[pyo3(signature = (min_elevation_deg=None, max_elevation_deg=None))]
    fn new(min_elevation_deg: Option<f64>, max_elevation_deg: Option<f64>) -> PyResult<Self> {
        ElevationConstraint::new(min_elevation_deg, max_elevation_deg)
            .map(|constraint| PyElevationConstraint { constraint })
            .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Get the constraint name
    fn name(&self) -> &str {
        self.constraint.name()
    }

    /// Evaluate whether the constraint is satisfied.
    ///
    /// Args:
    ///     epoch (Epoch): Time of evaluation
    ///     sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
    ///     location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)
    ///
    /// Returns:
    ///     bool: True if constraint is satisfied, False otherwise
    fn evaluate(
        &self,
        _py: Python,
        epoch: &PyEpoch,
        sat_state_ecef: PyReadonlyArray1<f64>,
        location_ecef: PyReadonlyArray1<f64>,
    ) -> bool {
        let sat_state = numpy_to_vector!(sat_state_ecef, 6, f64);
        let location = numpy_to_vector!(location_ecef, 3, f64);
        self.constraint.evaluate(&epoch.obj, &sat_state, &location)
    }

    fn __str__(&self) -> String {
        self.constraint.name().to_string()
    }

    fn __repr__(&self) -> String {
        format!("ElevationConstraint({})", self.constraint.name())
    }
}

/// Azimuth-dependent elevation mask constraint.
///
/// Constrains access based on azimuth-dependent elevation masks.
/// Useful for ground stations with terrain obstructions or antenna limitations.
///
/// The mask is defined as a list of (azimuth, elevation) pairs in degrees.
/// Linear interpolation is used between points, and the mask wraps at 0°/360°.
///
/// Args:
///     mask (list[tuple[float, float]]): List of (azimuth_deg, min_elevation_deg) pairs
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Ground station with terrain obstruction to the north
///     mask = [
///         (0.0, 15.0),     # North: 15° minimum
///         (90.0, 5.0),     # East: 5° minimum
///         (180.0, 5.0),    # South: 5° minimum
///         (270.0, 5.0),    # West: 5° minimum
///     ]
///     constraint = bh.ElevationMaskConstraint(mask)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "ElevationMaskConstraint")]
pub struct PyElevationMaskConstraint {
    pub(crate) constraint: ElevationMaskConstraint,
}

#[pymethods]
impl PyElevationMaskConstraint {
    #[new]
    fn new(mask: Vec<(f64, f64)>) -> Self {
        PyElevationMaskConstraint {
            constraint: ElevationMaskConstraint::new(mask),
        }
    }

    /// Get the constraint name
    fn name(&self) -> &str {
        self.constraint.name()
    }

    /// Evaluate whether the constraint is satisfied.
    ///
    /// Args:
    ///     epoch (Epoch): Time of evaluation
    ///     sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
    ///     location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)
    ///
    /// Returns:
    ///     bool: True if constraint is satisfied, False otherwise
    fn evaluate(
        &self,
        _py: Python,
        epoch: &PyEpoch,
        sat_state_ecef: PyReadonlyArray1<f64>,
        location_ecef: PyReadonlyArray1<f64>,
    ) -> bool {
        let sat_state = numpy_to_vector!(sat_state_ecef, 6, f64);
        let location = numpy_to_vector!(location_ecef, 3, f64);
        self.constraint.evaluate(&epoch.obj, &sat_state, &location)
    }

    fn __str__(&self) -> String {
        self.constraint.name().to_string()
    }

    fn __repr__(&self) -> String {
        format!("ElevationMaskConstraint({})", self.constraint.name())
    }
}

/// Off-nadir angle constraint for satellite imaging.
///
/// Constrains access based on the off-nadir angle (angle between the satellite's
/// nadir vector and the line-of-sight to the location).
///
/// Args:
///     min_off_nadir_deg (float | None): Minimum off-nadir angle in degrees, or None for no minimum
///     max_off_nadir_deg (float | None): Maximum off-nadir angle in degrees, or None for no maximum
///
/// Raises:
///     ValueError: If both min and max are None, or if any angle is negative
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Imaging satellite with 45° maximum slew angle
///     constraint = bh.OffNadirConstraint(min_off_nadir_deg=None, max_off_nadir_deg=45.0)
///
///     # Minimum 10° to avoid nadir (e.g., for oblique imaging)
///     constraint = bh.OffNadirConstraint(min_off_nadir_deg=10.0, max_off_nadir_deg=45.0)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OffNadirConstraint")]
pub struct PyOffNadirConstraint {
    pub(crate) constraint: OffNadirConstraint,
}

#[pymethods]
impl PyOffNadirConstraint {
    #[new]
    #[pyo3(signature = (min_off_nadir_deg=None, max_off_nadir_deg=None))]
    fn new(min_off_nadir_deg: Option<f64>, max_off_nadir_deg: Option<f64>) -> PyResult<Self> {
        OffNadirConstraint::new(min_off_nadir_deg, max_off_nadir_deg)
            .map(|constraint| PyOffNadirConstraint { constraint })
            .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Get the constraint name
    fn name(&self) -> &str {
        self.constraint.name()
    }

    /// Evaluate whether the constraint is satisfied.
    ///
    /// Args:
    ///     epoch (Epoch): Time of evaluation
    ///     sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
    ///     location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)
    ///
    /// Returns:
    ///     bool: True if constraint is satisfied, False otherwise
    fn evaluate(
        &self,
        _py: Python,
        epoch: &PyEpoch,
        sat_state_ecef: PyReadonlyArray1<f64>,
        location_ecef: PyReadonlyArray1<f64>,
    ) -> bool {
        let sat_state = numpy_to_vector!(sat_state_ecef, 6, f64);
        let location = numpy_to_vector!(location_ecef, 3, f64);
        self.constraint.evaluate(&epoch.obj, &sat_state, &location)
    }

    fn __str__(&self) -> String {
        self.constraint.name().to_string()
    }

    fn __repr__(&self) -> String {
        format!("OffNadirConstraint({})", self.constraint.name())
    }
}

/// Local solar time constraint.
///
/// Constrains access based on the local solar time at the ground location.
/// Useful for sun-synchronous orbits or daytime-only imaging.
///
/// Time windows are specified in military time format (HHMM).
/// Wrap-around windows (e.g., 2200-0200) are supported.
///
/// Args:
///     time_windows (list[tuple[int, int]]): List of (start_military, end_military) tuples (0-2400)
///
/// Raises:
///     ValueError: If any military time is invalid (>2400 or minutes >=60)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Only daytime (6 AM to 6 PM local time)
///     constraint = bh.LocalTimeConstraint(time_windows=[(600, 1800)])
///
///     # Two windows: morning (6-9 AM) and evening (4-7 PM)
///     constraint = bh.LocalTimeConstraint(time_windows=[(600, 900), (1600, 1900)])
///
///     # Overnight window (10 PM to 2 AM) - handles wrap-around
///     constraint = bh.LocalTimeConstraint(time_windows=[(2200, 200)])
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "LocalTimeConstraint")]
pub struct PyLocalTimeConstraint {
    pub(crate) constraint: LocalTimeConstraint,
}

#[pymethods]
impl PyLocalTimeConstraint {
    #[new]
    fn new(time_windows: Vec<(u16, u16)>) -> PyResult<Self> {
        LocalTimeConstraint::new(time_windows)
            .map(|constraint| PyLocalTimeConstraint { constraint })
            .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Create from decimal hour windows instead of military time.
    ///
    /// Args:
    ///     time_windows (list[tuple[float, float]]): List of (start_hour, end_hour) tuples [0, 24)
    ///
    /// Returns:
    ///     LocalTimeConstraint: The constraint instance
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Only daytime (6 AM to 6 PM local time)
    ///     constraint = bh.LocalTimeConstraint.from_hours([(6.0, 18.0)])
    ///
    ///     # Overnight window (10 PM to 2 AM)
    ///     constraint = bh.LocalTimeConstraint.from_hours([(22.0, 2.0)])
    ///     ```
    #[classmethod]
    fn from_hours(_cls: &Bound<'_, PyType>, time_windows: Vec<(f64, f64)>) -> PyResult<Self> {
        LocalTimeConstraint::from_hours(time_windows)
            .map(|constraint| PyLocalTimeConstraint { constraint })
            .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Get the constraint name
    fn name(&self) -> &str {
        self.constraint.name()
    }

    /// Evaluate whether the constraint is satisfied.
    ///
    /// Args:
    ///     epoch (Epoch): Time of evaluation
    ///     sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
    ///     location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)
    ///
    /// Returns:
    ///     bool: True if constraint is satisfied, False otherwise
    fn evaluate(
        &self,
        _py: Python,
        epoch: &PyEpoch,
        sat_state_ecef: PyReadonlyArray1<f64>,
        location_ecef: PyReadonlyArray1<f64>,
    ) -> bool {
        let sat_state = numpy_to_vector!(sat_state_ecef, 6, f64);
        let location = numpy_to_vector!(location_ecef, 3, f64);
        self.constraint.evaluate(&epoch.obj, &sat_state, &location)
    }

    fn __str__(&self) -> String {
        self.constraint.name().to_string()
    }

    fn __repr__(&self) -> String {
        format!("LocalTimeConstraint({})", self.constraint.name())
    }
}

/// Look direction constraint (left/right relative to velocity).
///
/// Constrains access based on the look direction of the satellite relative
/// to its velocity vector.
///
/// Args:
///     allowed (LookDirection): Required look direction (LEFT, RIGHT, or EITHER)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Satellite can only look right
///     constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.RIGHT)
///
///     # Either direction is acceptable
///     constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.EITHER)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "LookDirectionConstraint")]
pub struct PyLookDirectionConstraint {
    pub(crate) constraint: LookDirectionConstraint,
}

#[pymethods]
impl PyLookDirectionConstraint {
    #[new]
    fn new(allowed: &PyLookDirection) -> Self {
        PyLookDirectionConstraint {
            constraint: LookDirectionConstraint::new(allowed.value),
        }
    }

    /// Get the constraint name
    fn name(&self) -> &str {
        self.constraint.name()
    }

    /// Evaluate whether the constraint is satisfied.
    ///
    /// Args:
    ///     epoch (Epoch): Time of evaluation
    ///     sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
    ///     location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)
    ///
    /// Returns:
    ///     bool: True if constraint is satisfied, False otherwise
    fn evaluate(
        &self,
        _py: Python,
        epoch: &PyEpoch,
        sat_state_ecef: PyReadonlyArray1<f64>,
        location_ecef: PyReadonlyArray1<f64>,
    ) -> bool {
        let sat_state = numpy_to_vector!(sat_state_ecef, 6, f64);
        let location = numpy_to_vector!(location_ecef, 3, f64);
        self.constraint.evaluate(&epoch.obj, &sat_state, &location)
    }

    fn __str__(&self) -> String {
        self.constraint.name().to_string()
    }

    fn __repr__(&self) -> String {
        format!("LookDirectionConstraint({})", self.constraint.name())
    }
}

/// Ascending/descending pass constraint.
///
/// Constrains access based on whether the satellite is on an ascending or
/// descending pass (moving north or south).
///
/// Args:
///     allowed (AscDsc): Required pass type (ASCENDING, DESCENDING, or EITHER)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Only ascending passes
///     constraint = bh.AscDscConstraint(allowed=bh.AscDsc.ASCENDING)
///
///     # Either type is acceptable
///     constraint = bh.AscDscConstraint(allowed=bh.AscDsc.EITHER)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "AscDscConstraint")]
pub struct PyAscDscConstraint {
    pub(crate) constraint: AscDscConstraint,
}

#[pymethods]
impl PyAscDscConstraint {
    #[new]
    fn new(allowed: &PyAscDsc) -> Self {
        PyAscDscConstraint {
            constraint: AscDscConstraint::new(allowed.value),
        }
    }

    /// Get the constraint name
    fn name(&self) -> &str {
        self.constraint.name()
    }

    /// Evaluate whether the constraint is satisfied.
    ///
    /// Args:
    ///     epoch (Epoch): Time of evaluation
    ///     sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
    ///     location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)
    ///
    /// Returns:
    ///     bool: True if constraint is satisfied, False otherwise
    fn evaluate(
        &self,
        _py: Python,
        epoch: &PyEpoch,
        sat_state_ecef: PyReadonlyArray1<f64>,
        location_ecef: PyReadonlyArray1<f64>,
    ) -> bool {
        let sat_state = numpy_to_vector!(sat_state_ecef, 6, f64);
        let location = numpy_to_vector!(location_ecef, 3, f64);
        self.constraint.evaluate(&epoch.obj, &sat_state, &location)
    }

    fn __str__(&self) -> String {
        self.constraint.name().to_string()
    }

    fn __repr__(&self) -> String {
        format!("AscDscConstraint({})", self.constraint.name())
    }
}

// ================================
// Constraint Composition
// ================================

/// Composite constraint combining multiple constraints with AND logic.
///
/// All constraints must be satisfied for the composite to evaluate to true.
///
/// Args:
///     constraints (list): List of constraint objects to combine with AND logic
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Ground station with multiple requirements
///     elev = bh.ElevationConstraint(min_elevation_deg=5.0, max_elevation_deg=None)
///     time = bh.LocalTimeConstraint(time_windows=[(600, 1800)])
///     combined = bh.ConstraintAll(constraints=[elev, time])
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "ConstraintAll")]
pub struct PyConstraintAll {
    pub(crate) composite: ConstraintComposite,
}

#[pymethods]
impl PyConstraintAll {
    #[new]
    fn new(constraints: Vec<Py<PyAny>>) -> PyResult<Self> {
        // Convert Python objects to Rust constraint trait objects
        // For now, we'll store them as-is and handle evaluation in Rust
        // This requires implementing conversion logic for each constraint type

        // Note: This is a simplified version. Full implementation would need
        // to check the type of each Py<PyAny> and convert to the appropriate
        // Rust constraint type
        Python::attach(|py| {
            let mut rust_constraints: Vec<Box<dyn AccessConstraint>> = Vec::new();

            for py_obj in constraints {
                // Try to extract each constraint type
                if let Ok(c) = py_obj.extract::<PyRef<PyElevationConstraint>>(py) {
                    rust_constraints.push(Box::new(c.constraint.clone()));
                } else if let Ok(c) = py_obj.extract::<PyRef<PyElevationMaskConstraint>>(py) {
                    rust_constraints.push(Box::new(c.constraint.clone()));
                } else if let Ok(c) = py_obj.extract::<PyRef<PyOffNadirConstraint>>(py) {
                    rust_constraints.push(Box::new(c.constraint.clone()));
                } else if let Ok(c) = py_obj.extract::<PyRef<PyLocalTimeConstraint>>(py) {
                    rust_constraints.push(Box::new(c.constraint.clone()));
                } else if let Ok(c) = py_obj.extract::<PyRef<PyLookDirectionConstraint>>(py) {
                    rust_constraints.push(Box::new(c.constraint.clone()));
                } else if let Ok(c) = py_obj.extract::<PyRef<PyAscDscConstraint>>(py) {
                    rust_constraints.push(Box::new(c.constraint.clone()));
                } else {
                    return Err(exceptions::PyTypeError::new_err(
                        "All constraints must be valid constraint objects"
                    ));
                }
            }

            Ok(PyConstraintAll {
                composite: ConstraintComposite::All(rust_constraints),
            })
        })
    }

    /// Get the constraint name
    fn name(&self) -> &str {
        self.composite.name()
    }

    /// Evaluate whether the constraint is satisfied.
    ///
    /// Args:
    ///     epoch (Epoch): Time of evaluation
    ///     sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
    ///     location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)
    ///
    /// Returns:
    ///     bool: True if ALL constraints are satisfied, False otherwise
    fn evaluate(
        &self,
        _py: Python,
        epoch: &PyEpoch,
        sat_state_ecef: PyReadonlyArray1<f64>,
        location_ecef: PyReadonlyArray1<f64>,
    ) -> bool {
        let sat_state = numpy_to_vector!(sat_state_ecef, 6, f64);
        let location = numpy_to_vector!(location_ecef, 3, f64);
        self.composite.evaluate(&epoch.obj, &sat_state, &location)
    }

    fn __str__(&self) -> String {
        self.composite.format_string()
    }

    fn __repr__(&self) -> String {
        format!("ConstraintAll({})", self.composite.format_string())
    }
}

/// Composite constraint combining multiple constraints with OR logic.
///
/// At least one constraint must be satisfied for the composite to evaluate to true.
///
/// Args:
///     constraints (list): List of constraint objects to combine with OR logic
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Accept either high elevation or specific time window
///     elev = bh.ElevationConstraint(min_elevation_deg=60.0, max_elevation_deg=None)
///     time = bh.LocalTimeConstraint(time_windows=[(1200, 1400)])
///     combined = bh.ConstraintAny(constraints=[elev, time])
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "ConstraintAny")]
pub struct PyConstraintAny {
    pub(crate) composite: ConstraintComposite,
}

#[pymethods]
impl PyConstraintAny {
    #[new]
    fn new(constraints: Vec<Py<PyAny>>) -> PyResult<Self> {
        Python::attach(|py| {
            let mut rust_constraints: Vec<Box<dyn AccessConstraint>> = Vec::new();

            for py_obj in constraints {
                if let Ok(c) = py_obj.extract::<PyRef<PyElevationConstraint>>(py) {
                    rust_constraints.push(Box::new(c.constraint.clone()));
                } else if let Ok(c) = py_obj.extract::<PyRef<PyElevationMaskConstraint>>(py) {
                    rust_constraints.push(Box::new(c.constraint.clone()));
                } else if let Ok(c) = py_obj.extract::<PyRef<PyOffNadirConstraint>>(py) {
                    rust_constraints.push(Box::new(c.constraint.clone()));
                } else if let Ok(c) = py_obj.extract::<PyRef<PyLocalTimeConstraint>>(py) {
                    rust_constraints.push(Box::new(c.constraint.clone()));
                } else if let Ok(c) = py_obj.extract::<PyRef<PyLookDirectionConstraint>>(py) {
                    rust_constraints.push(Box::new(c.constraint.clone()));
                } else if let Ok(c) = py_obj.extract::<PyRef<PyAscDscConstraint>>(py) {
                    rust_constraints.push(Box::new(c.constraint.clone()));
                } else {
                    return Err(exceptions::PyTypeError::new_err(
                        "All constraints must be valid constraint objects"
                    ));
                }
            }

            Ok(PyConstraintAny {
                composite: ConstraintComposite::Any(rust_constraints),
            })
        })
    }

    /// Get the constraint name
    fn name(&self) -> &str {
        self.composite.name()
    }

    /// Evaluate whether the constraint is satisfied.
    ///
    /// Args:
    ///     epoch (Epoch): Time of evaluation
    ///     sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
    ///     location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)
    ///
    /// Returns:
    ///     bool: True if AT LEAST ONE constraint is satisfied, False otherwise
    fn evaluate(
        &self,
        _py: Python,
        epoch: &PyEpoch,
        sat_state_ecef: PyReadonlyArray1<f64>,
        location_ecef: PyReadonlyArray1<f64>,
    ) -> bool {
        let sat_state = numpy_to_vector!(sat_state_ecef, 6, f64);
        let location = numpy_to_vector!(location_ecef, 3, f64);
        self.composite.evaluate(&epoch.obj, &sat_state, &location)
    }

    fn __str__(&self) -> String {
        self.composite.format_string()
    }

    fn __repr__(&self) -> String {
        format!("ConstraintAny({})", self.composite.format_string())
    }
}

/// Composite constraint negating another constraint with NOT logic.
///
/// The negated constraint must NOT be satisfied for this to evaluate to true.
///
/// Args:
///     constraint: Constraint object to negate
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Avoid low elevation angles (i.e., require high elevation)
///     low_elev = bh.ElevationConstraint(min_elevation_deg=None, max_elevation_deg=10.0)
///     high_elev = bh.ConstraintNot(constraint=low_elev)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "ConstraintNot")]
pub struct PyConstraintNot {
    pub(crate) composite: ConstraintComposite,
}

#[pymethods]
impl PyConstraintNot {
    #[new]
    fn new(constraint: Py<PyAny>) -> PyResult<Self> {
        Python::attach(|py| {
            let rust_constraint: Box<dyn AccessConstraint> =
                if let Ok(c) = constraint.extract::<PyRef<PyElevationConstraint>>(py) {
                    Box::new(c.constraint.clone())
                } else if let Ok(c) = constraint.extract::<PyRef<PyElevationMaskConstraint>>(py) {
                    Box::new(c.constraint.clone())
                } else if let Ok(c) = constraint.extract::<PyRef<PyOffNadirConstraint>>(py) {
                    Box::new(c.constraint.clone())
                } else if let Ok(c) = constraint.extract::<PyRef<PyLocalTimeConstraint>>(py) {
                    Box::new(c.constraint.clone())
                } else if let Ok(c) = constraint.extract::<PyRef<PyLookDirectionConstraint>>(py) {
                    Box::new(c.constraint.clone())
                } else if let Ok(c) = constraint.extract::<PyRef<PyAscDscConstraint>>(py) {
                    Box::new(c.constraint.clone())
                } else {
                    return Err(exceptions::PyTypeError::new_err(
                        "Constraint must be a valid constraint object"
                    ));
                };

            Ok(PyConstraintNot {
                composite: ConstraintComposite::Not(rust_constraint),
            })
        })
    }

    /// Get the constraint name
    fn name(&self) -> &str {
        self.composite.name()
    }

    /// Evaluate whether the constraint is satisfied.
    ///
    /// Args:
    ///     epoch (Epoch): Time of evaluation
    ///     sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
    ///     location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)
    ///
    /// Returns:
    ///     bool: True if the negated constraint is NOT satisfied, False otherwise
    fn evaluate(
        &self,
        _py: Python,
        epoch: &PyEpoch,
        sat_state_ecef: PyReadonlyArray1<f64>,
        location_ecef: PyReadonlyArray1<f64>,
    ) -> bool {
        let sat_state = numpy_to_vector!(sat_state_ecef, 6, f64);
        let location = numpy_to_vector!(location_ecef, 3, f64);
        self.composite.evaluate(&epoch.obj, &sat_state, &location)
    }

    fn __str__(&self) -> String {
        self.composite.format_string()
    }

    fn __repr__(&self) -> String {
        format!("ConstraintNot({})", self.composite.format_string())
    }
}

// ================================
// Location Classes
// ================================

/// A single point location on Earth's surface.
///
/// Represents a discrete point with geodetic coordinates (longitude, latitude, altitude).
/// Commonly used for ground stations, imaging targets, or tessellated polygon tiles.
///
/// Args:
///     lon (float): Longitude in degrees (-180 to 180)
///     lat (float): Latitude in degrees (-90 to 90)
///     alt (float): Altitude above ellipsoid in meters (default: 0.0)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create a ground station in Svalbard
///     svalbard = bh.PointLocation(lon=15.4, lat=78.2, alt=0.0)
///
///     # With identity
///     svalbard = bh.PointLocation(lon=15.4, lat=78.2, alt=0.0) \\
///         .with_name("Svalbard Ground Station") \\
///         .with_id(1)
///
///     # With custom properties
///     svalbard = bh.PointLocation(lon=15.4, lat=78.2, alt=0.0) \\
///         .add_property("country", "Norway") \\
///         .add_property("min_elevation_deg", 5.0)
///
///     # Access coordinates
///     lon = svalbard.lon()  # Quick accessor (always degrees)
///     lat_rad = svalbard.latitude(bh.AngleFormat.RADIANS)  # Format-aware
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "PointLocation")]
pub struct PyPointLocation {
    pub(crate) location: PointLocation,
}

#[pymethods]
impl PyPointLocation {
    #[new]
    #[pyo3(signature = (lon, lat, alt=0.0))]
    fn new(lon: f64, lat: f64, alt: f64) -> Self {
        PyPointLocation {
            location: PointLocation::new(lon, lat, alt),
        }
    }

    /// Create from GeoJSON Point Feature.
    ///
    /// Args:
    ///     geojson (dict): GeoJSON Feature object with Point geometry
    ///
    /// Returns:
    ///     PointLocation: New location instance
    ///
    /// Raises:
    ///     ValueError: If GeoJSON is invalid or not a Point Feature
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     geojson = {
    ///         "type": "Feature",
    ///         "geometry": {
    ///             "type": "Point",
    ///             "coordinates": [15.4, 78.2, 0.0]
    ///         },
    ///         "properties": {
    ///             "name": "Svalbard"
    ///         }
    ///     }
    ///
    ///     location = bh.PointLocation.from_geojson(geojson)
    ///     ```
    #[classmethod]
    fn from_geojson(_cls: &Bound<'_, PyType>, geojson: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Convert Python dict to serde_json::Value
        let json_str = Python::attach(|py| -> PyResult<String> {
            let json_module = py.import("json")?;
            let dumps = json_module.getattr("dumps")?;
            let json_str = dumps.call1((geojson,))?;
            json_str.extract()
        })?;

        let json_value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

        PointLocation::from_geojson(&json_value)
            .map(|location| PyPointLocation { location })
            .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Get longitude in degrees (quick accessor).
    ///
    /// Returns:
    ///     float: Longitude in degrees
    fn lon(&self) -> f64 {
        self.location.lon()
    }

    /// Get latitude in degrees (quick accessor).
    ///
    /// Returns:
    ///     float: Latitude in degrees
    fn lat(&self) -> f64 {
        self.location.lat()
    }

    /// Get altitude in meters (quick accessor).
    ///
    /// Returns:
    ///     float: Altitude in meters
    fn alt(&self) -> f64 {
        self.location.alt()
    }

    /// Get longitude with angle format conversion.
    ///
    /// Args:
    ///     angle_format (AngleFormat): Desired output format (DEGREES or RADIANS)
    ///
    /// Returns:
    ///     float: Longitude in specified format
    fn longitude(&self, angle_format: &PyAngleFormat) -> f64 {
        self.location.longitude(angle_format.value)
    }

    /// Get latitude with angle format conversion.
    ///
    /// Args:
    ///     angle_format (AngleFormat): Desired output format (DEGREES or RADIANS)
    ///
    /// Returns:
    ///     float: Latitude in specified format
    fn latitude(&self, angle_format: &PyAngleFormat) -> f64 {
        self.location.latitude(angle_format.value)
    }

    /// Get altitude in meters.
    ///
    /// Returns:
    ///     float: Altitude in meters
    fn altitude(&self) -> f64 {
        self.location.altitude()
    }

    /// Get center coordinates in geodetic format [lon, lat, alt].
    ///
    /// Returns:
    ///     ndarray: Geodetic coordinates [longitude_deg, latitude_deg, altitude_m]
    fn center_geodetic(&self, py: Python) -> Py<PyAny> {
        let center = self.location.center_geodetic();
        vec![center.x, center.y, center.z].into_pyarray(py).into_any().unbind()
    }

    /// Get center position in ECEF coordinates [x, y, z].
    ///
    /// Returns:
    ///     ndarray: ECEF position in meters [x, y, z]
    fn center_ecef(&self, py: Python) -> Py<PyAny> {
        let ecef = self.location.center_ecef();
        vec![ecef.x, ecef.y, ecef.z].into_pyarray(py).into_any().unbind()
    }

    /// Get custom properties dictionary.
    ///
    /// Returns:
    ///     PropertiesDict: Dictionary-like wrapper for properties that supports assignment
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     loc = bh.PointLocation(15.4, 78.2, 0.0)
    ///
    ///     # Dict-style assignment
    ///     loc.properties["climate"] = "Arctic"
    ///     loc.properties["country"] = "Norway"
    ///
    ///     # Dict-style access
    ///     print(loc.properties["climate"])  # "Arctic"
    ///
    ///     # Dict methods
    ///     if "country" in loc.properties:
    ///         del loc.properties["country"]
    ///
    ///     # Iteration
    ///     for key in loc.properties.keys():
    ///         print(key, loc.properties[key])
    ///     ```
    #[getter]
    fn properties(slf: Bound<'_, Self>) -> PyResult<Py<PyPropertiesDict>> {
        let py = slf.py();
        let parent: Py<PyAny> = slf.clone().into_any().unbind();
        Py::new(py, PyPropertiesDict::new(parent))
    }

    /// Internal method: Get properties as a plain Python dict.
    #[pyo3(name = "_get_properties_dict")]
    fn get_properties_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let props = self.location.properties();
        let py_dict = PyDict::new(py);

        for (key, value) in props.iter() {
            let py_value = serde_json::to_string(value)
                .map_err(|e| exceptions::PyValueError::new_err(format!("JSON error: {}", e)))?;
            let json_module = py.import("json")?;
            let loads = json_module.getattr("loads")?;
            let py_val = loads.call1((py_value,))?;
            py_dict.set_item(key, py_val)?;
        }

        Ok(py_dict.into())
    }

    /// Internal method: Set a property value.
    #[pyo3(name = "_set_property")]
    fn set_property(&mut self, key: String, json_str: String) -> PyResult<()> {
        let value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| exceptions::PyValueError::new_err(format!("JSON error: {}", e)))?;
        self.location.properties_mut().insert(key, value);
        Ok(())
    }

    /// Internal method: Remove a property.
    #[pyo3(name = "_remove_property")]
    fn remove_property(&mut self, key: String) -> PyResult<()> {
        self.location.properties_mut().remove(&key)
            .ok_or_else(|| exceptions::PyKeyError::new_err(format!("Key '{}' not found", key)))?;
        Ok(())
    }

    /// Export to GeoJSON Feature format.
    ///
    /// Returns:
    ///     dict: GeoJSON Feature object
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     location = bh.PointLocation(lon=15.4, lat=78.2, alt=0.0) \\
    ///         .with_name("Svalbard")
    ///
    ///     geojson = location.to_geojson()
    ///     # Returns:
    ///     # {
    ///     #     "type": "Feature",
    ///     #     "geometry": {
    ///     #         "type": "Point",
    ///     #         "coordinates": [15.4, 78.2, 0.0]
    ///     #     },
    ///     #     "properties": {
    ///     #         "name": "Svalbard"
    ///     #     }
    ///     # }
    ///     ```
    fn to_geojson(&self, py: Python) -> PyResult<Py<PyAny>> {
        let geojson = self.location.to_geojson();
        let json_str = serde_json::to_string(&geojson)
            .map_err(|e| exceptions::PyValueError::new_err(format!("JSON error: {}", e)))?;

        let json_module = py.import("json")?;
        let loads = json_module.getattr("loads")?;
        let result = loads.call1((json_str,))?;
        Ok(result.into())
    }

    /// Add a custom property (builder pattern).
    ///
    /// Args:
    ///     key (str): Property name
    ///     value: Property value (must be JSON-serializable)
    ///
    /// Returns:
    ///     PointLocation: Self for chaining
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     location = bh.PointLocation(lon=15.4, lat=78.2, alt=0.0) \\
    ///         .add_property("country", "Norway") \\
    ///         .add_property("elevation_mask_deg", 5.0)
    ///     ```
    fn add_property<'py>(mut slf: PyRefMut<'py, Self>, key: String, value: &Bound<'py, PyAny>) -> PyResult<PyRefMut<'py, Self>> {
        // Convert Python value to serde_json::Value
        let json_str = Python::attach(|py| -> PyResult<String> {
            let json_module = py.import("json")?;
            let dumps = json_module.getattr("dumps")?;
            let json_str = dumps.call1((value,))?;
            json_str.extract()
        })?;

        let json_value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

        slf.location = slf.location.clone().add_property(&key, json_value);
        Ok(slf)
    }

    // ===== Identifiable trait methods =====

    /// Set the name (builder pattern).
    ///
    /// Args:
    ///     name (str): Human-readable name
    ///
    /// Returns:
    ///     PointLocation: Self for chaining
    fn with_name(mut slf: PyRefMut<'_, Self>, name: String) -> PyRefMut<'_, Self> {
        slf.location = slf.location.clone().with_name(&name);
        slf
    }

    /// Set the numeric ID (builder pattern).
    ///
    /// Args:
    ///     id (int): Numeric identifier
    ///
    /// Returns:
    ///     PointLocation: Self for chaining
    fn with_id(mut slf: PyRefMut<'_, Self>, id: u64) -> PyRefMut<'_, Self> {
        slf.location = slf.location.clone().with_id(id);
        slf
    }

    /// Set the UUID from a string (builder pattern).
    ///
    /// Args:
    ///     uuid_str (str): UUID string
    ///
    /// Returns:
    ///     PointLocation: Self for chaining
    ///
    /// Raises:
    ///     ValueError: If UUID string is invalid
    fn with_uuid(mut slf: PyRefMut<'_, Self>, uuid_str: String) -> PyResult<PyRefMut<'_, Self>> {
        let uuid = uuid::Uuid::parse_str(&uuid_str)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        slf.location = slf.location.clone().with_uuid(uuid);
        Ok(slf)
    }

    /// Generate a new UUID (builder pattern).
    ///
    /// Returns:
    ///     PointLocation: Self for chaining
    fn with_new_uuid(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.location = slf.location.clone().with_new_uuid();
        slf
    }

    /// Get the name.
    ///
    /// Returns:
    ///     str | None: Name if set, None otherwise
    fn get_name(&self) -> Option<String> {
        self.location.get_name().map(|s| s.to_string())
    }

    /// Get the numeric ID.
    ///
    /// Returns:
    ///     int | None: ID if set, None otherwise
    fn get_id(&self) -> Option<u64> {
        self.location.get_id()
    }

    /// Get the UUID as a string.
    ///
    /// Returns:
    ///     str | None: UUID string if set, None otherwise
    fn get_uuid(&self) -> Option<String> {
        self.location.get_uuid().map(|u| u.to_string())
    }

    /// Set the name (mutating).
    ///
    /// Args:
    ///     name (str | None): Name to set, or None to clear
    fn set_name(&mut self, name: Option<String>) {
        self.location.set_name(name.as_deref());
    }

    /// Set the numeric ID (mutating).
    ///
    /// Args:
    ///     id (int | None): ID to set, or None to clear
    fn set_id(&mut self, id: Option<u64>) {
        self.location.set_id(id);
    }

    /// Generate a new UUID (mutating).
    fn generate_uuid(&mut self) {
        self.location.generate_uuid();
    }

    fn __str__(&self) -> String {
        if let Some(name) = self.location.get_name() {
            format!("PointLocation({}, lon={:.4}°, lat={:.4}°, alt={:.1}m)",
                name, self.location.lon(), self.location.lat(), self.location.alt())
        } else {
            format!("PointLocation(lon={:.4}°, lat={:.4}°, alt={:.1}m)",
                self.location.lon(), self.location.lat(), self.location.alt())
        }
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

/// A polygonal area on Earth's surface.
///
/// Represents a closed polygon with multiple vertices.
/// Commonly used for areas of interest, no-fly zones, or imaging footprints.
///
/// The polygon is automatically closed if the first and last vertices don't match.
///
/// Args:
///     vertices (list[list[float]]): List of [lon, lat, alt] vertices in degrees and meters
///
/// Raises:
///     ValueError: If polygon has fewer than 4 vertices or has validation errors
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Define a rectangular area
///     vertices = [
///         [10.0, 50.0, 0.0],  # lon, lat, alt
///         [11.0, 50.0, 0.0],
///         [11.0, 51.0, 0.0],
///         [10.0, 51.0, 0.0],
///         [10.0, 50.0, 0.0],  # Closed (first == last)
///     ]
///     polygon = bh.PolygonLocation(vertices)
///
///     # With identity
///     polygon = bh.PolygonLocation(vertices) \\
///         .with_name("AOI-1") \\
///         .add_property("region", "Europe")
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "PolygonLocation")]
pub struct PyPolygonLocation {
    pub(crate) location: PolygonLocation,
}

#[pymethods]
impl PyPolygonLocation {
    #[new]
    fn new(vertices: Vec<Vec<f64>>) -> PyResult<Self> {
        // Convert Vec<Vec<f64>> to Vec<Vector3<f64>>
        let mut vec3_vertices = Vec::new();
        for vertex in vertices {
            if vertex.len() != 3 {
                return Err(exceptions::PyValueError::new_err(
                    "Each vertex must have [lon, lat, alt]",
                ));
            }
            vec3_vertices.push(Vector3::new(vertex[0], vertex[1], vertex[2]));
        }

        PolygonLocation::new(vec3_vertices)
            .map(|location| PyPolygonLocation { location })
            .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Create from GeoJSON Polygon Feature.
    ///
    /// Args:
    ///     geojson (dict): GeoJSON Feature object with Polygon geometry
    ///
    /// Returns:
    ///     PolygonLocation: New polygon instance
    ///
    /// Raises:
    ///     ValueError: If GeoJSON is invalid or not a Polygon Feature
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     geojson = {
    ///         "type": "Feature",
    ///         "geometry": {
    ///             "type": "Polygon",
    ///             "coordinates": [[
    ///                 [10.0, 50.0, 0.0],
    ///                 [11.0, 50.0, 0.0],
    ///                 [11.0, 51.0, 0.0],
    ///                 [10.0, 51.0, 0.0],
    ///                 [10.0, 50.0, 0.0]
    ///             ]]
    ///         },
    ///         "properties": {
    ///             "name": "AOI-1"
    ///         }
    ///     }
    ///
    ///     polygon = bh.PolygonLocation.from_geojson(geojson)
    ///     ```
    #[classmethod]
    fn from_geojson(_cls: &Bound<'_, PyType>, geojson: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Convert Python dict to serde_json::Value
        let json_str = Python::attach(|py| -> PyResult<String> {
            let json_module = py.import("json")?;
            let dumps = json_module.getattr("dumps")?;
            let json_str = dumps.call1((geojson,))?;
            json_str.extract()
        })?;

        let json_value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

        PolygonLocation::from_geojson(&json_value)
            .map(|location| PyPolygonLocation { location })
            .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Get polygon vertices.
    ///
    /// Returns all vertices including the closure vertex (first == last).
    ///
    /// Returns:
    ///     ndarray: Vertices as Nx3 array [[lon, lat, alt], ...]
    fn vertices(&self, py: Python) -> Py<PyAny> {
        let verts = self.location.vertices();
        let n = verts.len();
        let mut data = Vec::new();
        for v in verts {
            data.extend_from_slice(&[v.x, v.y, v.z]);
        }
        data.into_pyarray(py).reshape([n, 3]).unwrap().into_any().unbind()
    }

    /// Get number of unique vertices (excluding closure).
    ///
    /// Returns:
    ///     int: Number of unique vertices
    fn num_vertices(&self) -> usize {
        self.location.num_vertices()
    }

    /// Get center longitude in degrees (quick accessor).
    ///
    /// Returns:
    ///     float: Center longitude in degrees
    fn lon(&self) -> f64 {
        self.location.lon()
    }

    /// Get center latitude in degrees (quick accessor).
    ///
    /// Returns:
    ///     float: Center latitude in degrees
    fn lat(&self) -> f64 {
        self.location.lat()
    }

    /// Get center altitude in meters (quick accessor).
    ///
    /// Returns:
    ///     float: Center altitude in meters
    fn alt(&self) -> f64 {
        self.location.alt()
    }

    /// Get center longitude with angle format conversion.
    ///
    /// Args:
    ///     angle_format (AngleFormat): Desired output format (DEGREES or RADIANS)
    ///
    /// Returns:
    ///     float: Center longitude in specified format
    fn longitude(&self, angle_format: &PyAngleFormat) -> f64 {
        self.location.longitude(angle_format.value)
    }

    /// Get center latitude with angle format conversion.
    ///
    /// Args:
    ///     angle_format (AngleFormat): Desired output format (DEGREES or RADIANS)
    ///
    /// Returns:
    ///     float: Center latitude in specified format
    fn latitude(&self, angle_format: &PyAngleFormat) -> f64 {
        self.location.latitude(angle_format.value)
    }

    /// Get center altitude in meters.
    ///
    /// Returns:
    ///     float: Center altitude in meters
    fn altitude(&self) -> f64 {
        self.location.altitude()
    }

    /// Get center coordinates in geodetic format [lon, lat, alt].
    ///
    /// Returns:
    ///     ndarray: Geodetic coordinates [longitude_deg, latitude_deg, altitude_m]
    fn center_geodetic(&self, py: Python) -> Py<PyAny> {
        let center = self.location.center_geodetic();
        vec![center.x, center.y, center.z].into_pyarray(py).into_any().unbind()
    }

    /// Get center position in ECEF coordinates [x, y, z].
    ///
    /// Returns:
    ///     ndarray: ECEF position in meters [x, y, z]
    fn center_ecef(&self, py: Python) -> Py<PyAny> {
        let ecef = self.location.center_ecef();
        vec![ecef.x, ecef.y, ecef.z].into_pyarray(py).into_any().unbind()
    }

    /// Get custom properties dictionary.
    ///
    /// Returns:
    ///     PropertiesDict: Dictionary-like wrapper for properties that supports assignment
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     verts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    ///     poly = bh.PolygonLocation(verts)
    ///
    ///     # Dict-style assignment
    ///     poly.properties["region"] = "Test Area"
    ///     poly.properties["area_km2"] = 123.45
    ///
    ///     # Dict-style access
    ///     print(poly.properties["region"])  # "Test Area"
    ///
    ///     # Dict methods
    ///     if "area_km2" in poly.properties:
    ///         del poly.properties["area_km2"]
    ///     ```
    #[getter]
    fn properties(slf: Bound<'_, Self>) -> PyResult<Py<PyPropertiesDict>> {
        let py = slf.py();
        let parent: Py<PyAny> = slf.clone().into_any().unbind();
        Py::new(py, PyPropertiesDict::new(parent))
    }

    /// Internal method: Get properties as a plain Python dict.
    #[pyo3(name = "_get_properties_dict")]
    fn get_properties_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        let props = self.location.properties();
        let py_dict = PyDict::new(py);

        for (key, value) in props.iter() {
            let py_value = serde_json::to_string(value)
                .map_err(|e| exceptions::PyValueError::new_err(format!("JSON error: {}", e)))?;
            let json_module = py.import("json")?;
            let loads = json_module.getattr("loads")?;
            let py_val = loads.call1((py_value,))?;
            py_dict.set_item(key, py_val)?;
        }

        Ok(py_dict.into())
    }

    /// Internal method: Set a property value.
    #[pyo3(name = "_set_property")]
    fn set_property(&mut self, key: String, json_str: String) -> PyResult<()> {
        let value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| exceptions::PyValueError::new_err(format!("JSON error: {}", e)))?;
        self.location.properties_mut().insert(key, value);
        Ok(())
    }

    /// Internal method: Remove a property.
    #[pyo3(name = "_remove_property")]
    fn remove_property(&mut self, key: String) -> PyResult<()> {
        self.location.properties_mut().remove(&key)
            .ok_or_else(|| exceptions::PyKeyError::new_err(format!("Key '{}' not found", key)))?;
        Ok(())
    }

    /// Export to GeoJSON Feature format.
    ///
    /// Returns:
    ///     dict: GeoJSON Feature object
    fn to_geojson(&self, py: Python) -> PyResult<Py<PyAny>> {
        let geojson = self.location.to_geojson();
        let json_str = serde_json::to_string(&geojson)
            .map_err(|e| exceptions::PyValueError::new_err(format!("JSON error: {}", e)))?;

        let json_module = py.import("json")?;
        let loads = json_module.getattr("loads")?;
        let result = loads.call1((json_str,))?;
        Ok(result.into())
    }

    /// Add a custom property (builder pattern).
    ///
    /// Args:
    ///     key (str): Property name
    ///     value: Property value (must be JSON-serializable)
    ///
    /// Returns:
    ///     PolygonLocation: Self for chaining
    fn add_property<'py>(mut slf: PyRefMut<'py, Self>, key: String, value: &Bound<'py, PyAny>) -> PyResult<PyRefMut<'py, Self>> {
        // Convert Python value to serde_json::Value
        let json_str = Python::attach(|py| -> PyResult<String> {
            let json_module = py.import("json")?;
            let dumps = json_module.getattr("dumps")?;
            let json_str = dumps.call1((value,))?;
            json_str.extract()
        })?;

        let json_value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid JSON: {}", e)))?;

        slf.location = slf.location.clone().add_property(&key, json_value);
        Ok(slf)
    }

    // ===== Identifiable trait methods =====

    /// Set the name (builder pattern).
    ///
    /// Args:
    ///     name (str): Human-readable name
    ///
    /// Returns:
    ///     PolygonLocation: Self for chaining
    fn with_name(mut slf: PyRefMut<'_, Self>, name: String) -> PyRefMut<'_, Self> {
        slf.location = slf.location.clone().with_name(&name);
        slf
    }

    /// Set the numeric ID (builder pattern).
    ///
    /// Args:
    ///     id (int): Numeric identifier
    ///
    /// Returns:
    ///     PolygonLocation: Self for chaining
    fn with_id(mut slf: PyRefMut<'_, Self>, id: u64) -> PyRefMut<'_, Self> {
        slf.location = slf.location.clone().with_id(id);
        slf
    }

    /// Set the UUID from a string (builder pattern).
    ///
    /// Args:
    ///     uuid_str (str): UUID string
    ///
    /// Returns:
    ///     PolygonLocation: Self for chaining
    ///
    /// Raises:
    ///     ValueError: If UUID string is invalid
    fn with_uuid(mut slf: PyRefMut<'_, Self>, uuid_str: String) -> PyResult<PyRefMut<'_, Self>> {
        let uuid = uuid::Uuid::parse_str(&uuid_str)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        slf.location = slf.location.clone().with_uuid(uuid);
        Ok(slf)
    }

    /// Generate a new UUID (builder pattern).
    ///
    /// Returns:
    ///     PolygonLocation: Self for chaining
    fn with_new_uuid(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.location = slf.location.clone().with_new_uuid();
        slf
    }

    /// Get the name.
    ///
    /// Returns:
    ///     str | None: Name if set, None otherwise
    fn get_name(&self) -> Option<String> {
        self.location.get_name().map(|s| s.to_string())
    }

    /// Get the numeric ID.
    ///
    /// Returns:
    ///     int | None: ID if set, None otherwise
    fn get_id(&self) -> Option<u64> {
        self.location.get_id()
    }

    /// Get the UUID as a string.
    ///
    /// Returns:
    ///     str | None: UUID string if set, None otherwise
    fn get_uuid(&self) -> Option<String> {
        self.location.get_uuid().map(|u| u.to_string())
    }

    /// Set the name (mutating).
    ///
    /// Args:
    ///     name (str | None): Name to set, or None to clear
    fn set_name(&mut self, name: Option<String>) {
        self.location.set_name(name.as_deref());
    }

    /// Set the numeric ID (mutating).
    ///
    /// Args:
    ///     id (int | None): ID to set, or None to clear
    fn set_id(&mut self, id: Option<u64>) {
        self.location.set_id(id);
    }

    /// Generate a new UUID (mutating).
    fn generate_uuid(&mut self) {
        self.location.generate_uuid();
    }

    fn __str__(&self) -> String {
        if let Some(name) = self.location.get_name() {
            format!("PolygonLocation({}, {} vertices, center=[{:.4}°, {:.4}°, {:.1}m])",
                name, self.location.num_vertices(),
                self.location.lon(), self.location.lat(), self.location.alt())
        } else {
            format!("PolygonLocation({} vertices, center=[{:.4}°, {:.4}°, {:.1}m])",
                self.location.num_vertices(),
                self.location.lon(), self.location.lat(), self.location.alt())
        }
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}
