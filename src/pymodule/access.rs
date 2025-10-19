// Access module Python bindings
//
// This file contains Python bindings for the access computation module,
// including constraints, locations, and access window finding.

use crate::access::constraints::{
    AccessConstraint, AscDsc, AscDscConstraint, ConstraintComposite, ElevationConstraint,
    ElevationMaskConstraint, LocalTimeConstraint, LookDirection, LookDirectionConstraint,
    OffNadirConstraint,
};
use crate::access::location::{AccessibleLocation, PointLocation, PolygonLocation};
use crate::access::properties::{
    AccessProperties, AccessPropertyComputer, PropertyValue,
};
use crate::access::windows::{AccessSearchConfig, AccessWindow};
use crate::utils::identifiable::Identifiable;
use crate::utils::BraheError;
use nalgebra::Vector3;
use pyo3::types::PyDict;
use std::collections::HashMap;

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
    pub constraint: ElevationConstraint,
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
    pub constraint: ElevationMaskConstraint,
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
    pub constraint: OffNadirConstraint,
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
    pub constraint: LocalTimeConstraint,
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
    pub constraint: LookDirectionConstraint,
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
    pub constraint: AscDscConstraint,
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
    pub composite: ConstraintComposite,
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
    pub composite: ConstraintComposite,
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
    pub composite: ConstraintComposite,
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
    pub location: PointLocation,
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
    pub location: PolygonLocation,
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

// ================================
// Access Properties
// ================================

/// A flexible value type for access properties.
///
/// PropertyValue can store different types of data associated with access windows,
/// including scalars, vectors, time series, booleans, strings, and arbitrary JSON.
///
/// Args:
///     value: The value to store (int, float, list, bool, str, or dict)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Scalar value
///     prop1 = bh.PropertyValue(42.5)
///
///     # Vector value
///     prop2 = bh.PropertyValue([1.0, 2.0, 3.0])
///
///     # Boolean value
///     prop3 = bh.PropertyValue(True)
///
///     # String value
///     prop4 = bh.PropertyValue("visible")
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "PropertyValue")]
pub struct PyPropertyValue {
    pub(crate) value: PropertyValue,
}

#[pymethods]
impl PyPropertyValue {
    #[new]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Check bool BEFORE f64, since bool is a subclass of int in Python
        let property_value = if let Ok(v) = value.extract::<bool>() {
            PropertyValue::Boolean(v)
        } else if let Ok(v) = value.extract::<f64>() {
            PropertyValue::Scalar(v)
        } else if let Ok(v) = value.extract::<Vec<f64>>() {
            PropertyValue::Vector(v)
        } else if let Ok(v) = value.extract::<String>() {
            PropertyValue::String(v)
        } else if let Ok(dict) = value.downcast::<PyDict>() {
            // Convert dict to JSON
            let json_module = value.py().import("json")?;
            let dumps = json_module.getattr("dumps")?;
            let json_str: String = dumps.call1((dict,))?.extract()?;
            let json_value: serde_json::Value = serde_json::from_str(&json_str)
                .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(format!("JSON error: {}", e)))?;
            PropertyValue::Json(json_value)
        } else {
            return Err(PyErr::new::<exceptions::PyTypeError, _>(
                "PropertyValue must be a number, list, bool, str, or dict"
            ));
        };

        Ok(Self { value: property_value })
    }

    /// Convert the property value to a Python object.
    ///
    /// Returns:
    ///     The value as a Python object (int, float, list, bool, or str)
    fn to_python(&self, py: Python) -> PyResult<Py<PyAny>> {
        use pyo3::types::{PyFloat, PyList, PyString};

        match &self.value {
            PropertyValue::Scalar(v) => {
                let float_obj = PyFloat::new(py, *v);
                Ok(float_obj.unbind().into())
            }
            PropertyValue::Vector(v) => {
                let list = PyList::new(py, v.iter())?;
                Ok(list.unbind().into())
            }
            PropertyValue::TimeSeries { times, values } => {
                let dict = PyDict::new(py);
                dict.set_item("times", times)?;
                dict.set_item("values", values)?;
                Ok(dict.into())
            }
            PropertyValue::Boolean(v) => {
                // Use Python's builtin bool function
                let builtins = py.import("builtins")?;
                let bool_func = builtins.getattr("bool")?;
                Ok(bool_func.call1((*v,))?.into())
            }
            PropertyValue::String(v) => {
                let str_obj = PyString::new(py, v);
                Ok(str_obj.unbind().into())
            }
            PropertyValue::Json(v) => {
                // Convert JSON to Python via json module
                let json_module = py.import("json")?;
                let loads = json_module.getattr("loads")?;
                let json_str = serde_json::to_string(v)
                    .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(format!("JSON error: {}", e)))?;
                Ok(loads.call1((json_str,))?.into())
            }
        }
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let value_str = self.to_python(py)?;
        Ok(format!("PropertyValue({})", value_str))
    }
}

/// An access window representing a period of time when access constraints are satisfied.
///
/// AccessWindow stores the opening and closing times of an access period, along with
/// computed properties for that window.
///
/// Args:
///     window_open (Epoch): Opening time of the access window
///     window_close (Epoch): Closing time of the access window
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create an access window
///     t_open = bh.Epoch(2024, 1, 1, 12, 0, 0.0)
///     t_close = bh.Epoch(2024, 1, 1, 12, 10, 0.0)
///     window = bh.AccessWindow(t_open, t_close)
///
///     # Access window properties
///     print(f"Duration: {window.duration()} seconds")
///     print(f"Midpoint: {window.midtime()}")
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "AccessWindow")]
pub struct PyAccessWindow {
    pub(crate) window: AccessWindow,
}

#[pymethods]
impl PyAccessWindow {
    #[new]
    fn new(window_open: &PyEpoch, window_close: &PyEpoch) -> Self {
        Self {
            window: AccessWindow {
                window_open: window_open.obj,
                window_close: window_close.obj,
                location_name: None,
                location_id: None,
                location_uuid: None,
                satellite_name: None,
                satellite_id: None,
                satellite_uuid: None,
                name: None,
                id: None,
                uuid: None,
                properties: AccessProperties {
                    azimuth_open: 0.0,
                    azimuth_close: 0.0,
                    elevation_min: 0.0,
                    elevation_max: 0.0,
                    off_nadir_min: 0.0,
                    off_nadir_max: 0.0,
                    local_time: 0.0,
                    look_direction: LookDirection::Either,
                    asc_dsc: AscDsc::Either,
                    additional: std::collections::HashMap::new(),
                },
            },
        }
    }

    // ===== Time properties (as getters) =====

    /// Get the window opening time.
    ///
    /// Returns:
    ///     Epoch: Window opening time
    #[getter]
    fn window_open(&self) -> PyEpoch {
        PyEpoch { obj: self.window.window_open }
    }

    /// Get the window closing time.
    ///
    /// Returns:
    ///     Epoch: Window closing time
    #[getter]
    fn window_close(&self) -> PyEpoch {
        PyEpoch { obj: self.window.window_close }
    }

    /// Get the start time of the access window (alias for window_open).
    ///
    /// Returns:
    ///     Epoch: Opening time of the window
    #[getter]
    fn start(&self) -> PyEpoch {
        PyEpoch { obj: self.window.start() }
    }

    /// Get the end time of the access window (alias for window_close).
    ///
    /// Returns:
    ///     Epoch: Closing time of the window
    #[getter]
    fn end(&self) -> PyEpoch {
        PyEpoch { obj: self.window.end() }
    }

    /// Get the midpoint time of the access window.
    ///
    /// Returns:
    ///     Epoch: Midpoint time (average of start and end)
    #[getter]
    fn midtime(&self) -> PyEpoch {
        PyEpoch { obj: self.window.midtime() }
    }

    /// Get the duration of the access window in seconds.
    ///
    /// Returns:
    ///     float: Duration in seconds
    #[getter]
    fn duration(&self) -> f64 {
        self.window.duration()
    }

    // ===== Identifiable properties =====

    /// Get the access window name (auto-generated or user-set).
    ///
    /// Returns:
    ///     Optional[str]: Window name, or None if not set
    #[getter]
    fn name(&self) -> Option<String> {
        self.window.name.clone()
    }

    /// Get the access window numeric ID.
    ///
    /// Returns:
    ///     Optional[int]: Window ID, or None if not set
    #[getter]
    fn id(&self) -> Option<u64> {
        self.window.id
    }

    /// Get the access window UUID.
    ///
    /// Returns:
    ///     Optional[str]: UUID as string, or None if not set
    #[getter]
    fn uuid(&self) -> Option<String> {
        self.window.uuid.map(|u| u.to_string())
    }

    // ===== Location/Satellite identification =====

    /// Get the location name if available.
    ///
    /// Returns:
    ///     Optional[str]: Name of the location, or None if not set
    #[getter]
    fn location_name(&self) -> Option<String> {
        self.window.location_name.clone()
    }

    /// Get the location ID if available.
    ///
    /// Returns:
    ///     Optional[int]: ID of the location, or None if not set
    #[getter]
    fn location_id(&self) -> Option<u64> {
        self.window.location_id
    }

    /// Get the satellite/object name if available.
    ///
    /// Returns:
    ///     Optional[str]: Name of the satellite, or None if not set
    #[getter]
    fn satellite_name(&self) -> Option<String> {
        self.window.satellite_name.clone()
    }

    /// Get the satellite/object ID if available.
    ///
    /// Returns:
    ///     Optional[int]: ID of the satellite, or None if not set
    #[getter]
    fn satellite_id(&self) -> Option<u64> {
        self.window.satellite_id
    }

    /// Get the location UUID if available.
    ///
    /// Returns:
    ///     Optional[str]: UUID of the location as string, or None if not set
    #[getter]
    fn location_uuid(&self) -> Option<String> {
        self.window.location_uuid.map(|u| u.to_string())
    }

    /// Get the satellite UUID if available.
    ///
    /// Returns:
    ///     Optional[str]: UUID of the satellite as string, or None if not set
    #[getter]
    fn satellite_uuid(&self) -> Option<String> {
        self.window.satellite_uuid.map(|u| u.to_string())
    }

    // ===== Access properties (convenience getters) =====

    /// Get the access properties object.
    ///
    /// Returns:
    ///     AccessProperties: Computed properties for this access window
    #[getter]
    fn properties(&self) -> PyAccessProperties {
        PyAccessProperties {
            properties: self.window.properties.clone(),
        }
    }

    /// Get azimuth angle at window opening (degrees, 0-360).
    ///
    /// Returns:
    ///     float: Azimuth at window open
    #[getter]
    fn azimuth_open(&self) -> f64 {
        self.window.properties.azimuth_open
    }

    /// Get azimuth angle at window closing (degrees, 0-360).
    ///
    /// Returns:
    ///     float: Azimuth at window close
    #[getter]
    fn azimuth_close(&self) -> f64 {
        self.window.properties.azimuth_close
    }

    /// Get minimum elevation angle during access (degrees).
    ///
    /// Returns:
    ///     float: Minimum elevation angle
    #[getter]
    fn elevation_min(&self) -> f64 {
        self.window.properties.elevation_min
    }

    /// Get maximum elevation angle during access (degrees).
    ///
    /// Returns:
    ///     float: Maximum elevation angle
    #[getter]
    fn elevation_max(&self) -> f64 {
        self.window.properties.elevation_max
    }

    /// Get minimum off-nadir angle during access (degrees).
    ///
    /// Returns:
    ///     float: Minimum off-nadir angle
    #[getter]
    fn off_nadir_min(&self) -> f64 {
        self.window.properties.off_nadir_min
    }

    /// Get maximum off-nadir angle during access (degrees).
    ///
    /// Returns:
    ///     float: Maximum off-nadir angle
    #[getter]
    fn off_nadir_max(&self) -> f64 {
        self.window.properties.off_nadir_max
    }

    /// Get local solar time at window midpoint (seconds since midnight, 0-86400).
    ///
    /// Returns:
    ///     float: Local time in seconds
    #[getter]
    fn local_time(&self) -> f64 {
        self.window.properties.local_time
    }

    /// Get look direction (Left or Right).
    ///
    /// Returns:
    ///     LookDirection: Look direction enum value
    #[getter]
    fn look_direction(&self) -> PyLookDirection {
        PyLookDirection { value: self.window.properties.look_direction }
    }

    /// Get ascending/descending pass type.
    ///
    /// Returns:
    ///     AscDsc: Pass type enum value
    #[getter]
    fn asc_dsc(&self) -> PyAscDsc {
        PyAscDsc { value: self.window.properties.asc_dsc }
    }

    fn __repr__(&self) -> String {
        format!(
            "AccessWindow(start={}, end={}, duration={:.2}s)",
            self.window.start(),
            self.window.end(),
            self.window.duration()
        )
    }
}

/// Properties computed for an access window.
///
/// AccessProperties contains geometric properties (azimuth, elevation, off-nadir angles,
/// local time, look direction, ascending/descending) computed over an access window,
/// plus a dictionary of additional custom properties.
///
/// Attributes:
///     azimuth_open (float): Azimuth angle at window opening (degrees, 0-360)
///     azimuth_close (float): Azimuth angle at window closing (degrees, 0-360)
///     elevation_min (float): Minimum elevation angle (degrees)
///     elevation_max (float): Maximum elevation angle (degrees)
///     off_nadir_min (float): Minimum off-nadir angle (degrees)
///     off_nadir_max (float): Maximum off-nadir angle (degrees)
///     local_time (float): Local solar time (seconds since midnight, 0-86400)
///     look_direction (LookDirection): Required look direction (Left or Right)
///     asc_dsc (AscDsc): Pass type (Ascending or Descending)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Access properties are typically computed by the access computation system
///     # This example shows accessing the properties
///     props = ...  # From access computation
///
///     print(f"Azimuth at open: {props.azimuth_open}°")
///     print(f"Max elevation: {props.elevation_max}°")
///     print(f"Look direction: {props.look_direction}")
///
///     # Access additional custom properties
///     if "signal_strength" in props.additional:
///         print(f"Signal: {props.additional['signal_strength']}")
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "AccessProperties")]
pub struct PyAccessProperties {
    pub(crate) properties: AccessProperties,
}

#[pymethods]
impl PyAccessProperties {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        azimuth_open: f64,
        azimuth_close: f64,
        elevation_min: f64,
        elevation_max: f64,
        off_nadir_min: f64,
        off_nadir_max: f64,
        local_time: f64,
        look_direction: &PyLookDirection,
        asc_dsc: &PyAscDsc,
    ) -> Self {
        Self {
            properties: AccessProperties {
                azimuth_open,
                azimuth_close,
                elevation_min,
                elevation_max,
                off_nadir_min,
                off_nadir_max,
                local_time,
                look_direction: look_direction.value,
                asc_dsc: asc_dsc.value,
                additional: std::collections::HashMap::new(),
            },
        }
    }

    #[getter]
    fn azimuth_open(&self) -> f64 {
        self.properties.azimuth_open
    }

    #[getter]
    fn azimuth_close(&self) -> f64 {
        self.properties.azimuth_close
    }

    #[getter]
    fn elevation_min(&self) -> f64 {
        self.properties.elevation_min
    }

    #[getter]
    fn elevation_max(&self) -> f64 {
        self.properties.elevation_max
    }

    #[getter]
    fn off_nadir_min(&self) -> f64 {
        self.properties.off_nadir_min
    }

    #[getter]
    fn off_nadir_max(&self) -> f64 {
        self.properties.off_nadir_max
    }

    #[getter]
    fn local_time(&self) -> f64 {
        self.properties.local_time
    }

    #[getter]
    fn look_direction(&self) -> PyLookDirection {
        PyLookDirection { value: self.properties.look_direction }
    }

    #[getter]
    fn asc_dsc(&self) -> PyAscDsc {
        PyAscDsc { value: self.properties.asc_dsc }
    }

    /// Get additional properties as a dict-like wrapper.
    ///
    /// Returns a dictionary-like object that automatically converts between
    /// Python types and internal PropertyValue representation.
    ///
    /// Supported Python types:
    /// - float -> Scalar
    /// - list[float] -> Vector
    /// - bool -> Boolean
    /// - str -> String
    /// - dict -> Json
    ///
    /// Returns:
    ///     AdditionalPropertiesDict: Dict-like wrapper for additional properties
    ///
    /// Example:
    ///     ```python
    ///     # Dict-style assignment
    ///     props.additional["doppler_shift"] = 2500.0
    ///     props.additional["snr_values"] = [10.5, 12.3, 15.1]
    ///     props.additional["has_eclipse"] = False
    ///
    ///     # Dict-style access
    ///     print(props.additional["doppler_shift"])  # 2500.0
    ///
    ///     # Dict methods
    ///     if "doppler_shift" in props.additional:
    ///         del props.additional["doppler_shift"]
    ///
    ///     # Iteration
    ///     for key in props.additional.keys():
    ///         print(key, props.additional[key])
    ///     ```
    #[getter]
    fn additional(slf: Bound<'_, Self>) -> PyResult<Py<PyAdditionalPropertiesDict>> {
        let py = slf.py();
        let dict = PyAdditionalPropertiesDict::new(slf.into_any().unbind());
        Py::new(py, dict)
    }

    // Internal methods for AdditionalPropertiesDict to call back into
    fn _get_additional_properties_dict(&self, py: Python) -> PyResult<Py<PyDict>> {
        use pyo3::types::{PyFloat, PyList, PyString};

        let dict = PyDict::new(py);
        for (key, value) in &self.properties.additional {
            let py_value: Py<PyAny> = match value {
                PropertyValue::Scalar(v) => {
                    let float_obj = PyFloat::new(py, *v);
                    float_obj.unbind().into()
                }
                PropertyValue::Vector(v) => {
                    let list = PyList::new(py, v.iter())?;
                    list.unbind().into()
                }
                PropertyValue::TimeSeries { times, values } => {
                    let ts_dict = PyDict::new(py);
                    ts_dict.set_item("times", times)?;
                    ts_dict.set_item("values", values)?;
                    ts_dict.into()
                }
                PropertyValue::Boolean(v) => {
                    // Use Python's builtin bool function
                    let builtins = py.import("builtins")?;
                    let bool_func = builtins.getattr("bool")?;
                    bool_func.call1((*v,))?.into()
                }
                PropertyValue::String(v) => {
                    let str_obj = PyString::new(py, v);
                    str_obj.unbind().into()
                }
                PropertyValue::Json(v) => {
                    let json_module = py.import("json")?;
                    let loads = json_module.getattr("loads")?;
                    let json_str = serde_json::to_string(v)
                        .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(format!("JSON error: {}", e)))?;
                    loads.call1((json_str,))?.into()
                }
            };
            dict.set_item(key, py_value)?;
        }
        Ok(dict.into())
    }

    fn _set_additional_property(&mut self, key: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        // Check bool BEFORE f64, since bool is a subclass of int in Python
        let property_value = if let Ok(v) = value.extract::<bool>() {
            PropertyValue::Boolean(v)
        } else if let Ok(v) = value.extract::<f64>() {
            PropertyValue::Scalar(v)
        } else if let Ok(v) = value.extract::<Vec<f64>>() {
            // Handle regular Python lists
            PropertyValue::Vector(v)
        } else if value.hasattr("tolist")? {
            // Handle numpy arrays by converting to list
            let as_list = value.call_method0("tolist")?;
            if let Ok(v) = as_list.extract::<Vec<f64>>() {
                PropertyValue::Vector(v)
            } else {
                return Err(PyErr::new::<exceptions::PyTypeError, _>(
                    "Numpy array must contain numeric values"
                ));
            }
        } else if let Ok(v) = value.extract::<String>() {
            PropertyValue::String(v)
        } else if let Ok(dict) = value.downcast::<PyDict>() {
            // Convert dict to JSON
            let json_module = value.py().import("json")?;
            let dumps = json_module.getattr("dumps")?;
            let json_str: String = dumps.call1((dict,))?.extract()?;
            let json_value: serde_json::Value = serde_json::from_str(&json_str)
                .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(format!("JSON error: {}", e)))?;
            PropertyValue::Json(json_value)
        } else {
            return Err(PyErr::new::<exceptions::PyTypeError, _>(
                "Property value must be a number, list, numpy array, bool, str, or dict"
            ));
        };

        self.properties.additional.insert(key, property_value);
        Ok(())
    }

    fn _remove_additional_property(&mut self, key: String) -> PyResult<()> {
        self.properties.additional.remove(&key)
            .ok_or_else(|| exceptions::PyKeyError::new_err(format!("Key '{}' not found", key)))?;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "AccessProperties(az=[{:.1}°, {:.1}°], el=[{:.1}°, {:.1}°], off_nadir=[{:.1}°, {:.1}°])",
            self.properties.azimuth_open,
            self.properties.azimuth_close,
            self.properties.elevation_min,
            self.properties.elevation_max,
            self.properties.off_nadir_min,
            self.properties.off_nadir_max
        )
    }
}

// ================================
// AdditionalPropertiesDict
// ================================

/// Python dictionary interface for additional access properties.
///
/// Provides dict-like access to additional properties with automatic type conversion.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "AdditionalPropertiesDict")]
pub struct PyAdditionalPropertiesDict {
    parent: Py<PyAny>,
}

#[pymethods]
impl PyAdditionalPropertiesDict {
    /// Get a property value by key.
    fn __getitem__(&self, key: String, py: Python) -> PyResult<Py<PyAny>> {
        let props_dict = self.get_additional_properties_dict(py)?;
        props_dict
            .get_item(&key)?
            .ok_or_else(|| exceptions::PyKeyError::new_err(format!("Key '{}' not found", key)))
            .map(|item| item.into())
    }

    /// Set a property value by key.
    fn __setitem__(&self, key: String, value: &Bound<'_, PyAny>, py: Python) -> PyResult<()> {
        // Call the parent's _set_additional_property method which handles type conversion
        self.set_additional_property(py, key, value)?;
        Ok(())
    }

    /// Delete a property by key.
    fn __delitem__(&self, key: String, py: Python) -> PyResult<()> {
        self.remove_additional_property(py, key)
    }

    /// Return the number of properties.
    fn __len__(&self, py: Python) -> PyResult<usize> {
        let props_dict = self.get_additional_properties_dict(py)?;
        Ok(props_dict.len())
    }

    /// Check if a key exists in properties.
    fn __contains__(&self, key: String, py: Python) -> PyResult<bool> {
        let props_dict = self.get_additional_properties_dict(py)?;
        props_dict.contains(&key)
    }

    /// Return an iterator over property keys.
    fn __iter__(&self, py: Python) -> PyResult<Py<PyAny>> {
        let props_dict = self.get_additional_properties_dict(py)?;
        Ok(props_dict.call_method0("__iter__")?.into())
    }

    /// String representation.
    fn __repr__(&self, py: Python) -> PyResult<String> {
        let props_dict = self.get_additional_properties_dict(py)?;
        Ok(format!("AdditionalPropertiesDict({})", props_dict.repr()?))
    }

    /// Get property value with optional default.
    #[pyo3(signature = (key, default=None))]
    fn get(&self, key: String, default: Option<Py<PyAny>>, py: Python) -> PyResult<Py<PyAny>> {
        let props_dict = self.get_additional_properties_dict(py)?;
        match props_dict.get_item(&key)? {
            Some(value) => Ok(value.into()),
            None => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    /// Return a list of property keys.
    fn keys(&self, py: Python) -> PyResult<Py<PyAny>> {
        let props_dict = self.get_additional_properties_dict(py)?;
        Ok(props_dict.call_method0("keys")?.into())
    }

    /// Return a list of property values.
    fn values(&self, py: Python) -> PyResult<Py<PyAny>> {
        let props_dict = self.get_additional_properties_dict(py)?;
        Ok(props_dict.call_method0("values")?.into())
    }

    /// Return a list of (key, value) tuples.
    fn items(&self, py: Python) -> PyResult<Py<PyAny>> {
        let props_dict = self.get_additional_properties_dict(py)?;
        Ok(props_dict.call_method0("items")?.into())
    }

    /// Remove all properties.
    fn clear(&self, py: Python) -> PyResult<()> {
        // Get all keys first (to avoid modifying during iteration)
        let props_dict = self.get_additional_properties_dict(py)?;
        let keys_view = props_dict.call_method0("keys")?;

        // Convert dict_keys to list before extracting
        let builtins = py.import("builtins")?;
        let list_fn = builtins.getattr("list")?;
        let keys_list = list_fn.call1((keys_view,))?;
        let keys: Vec<String> = keys_list.extract()?;

        // Remove each key
        for key in keys {
            self.remove_additional_property(py, key)?;
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

impl PyAdditionalPropertiesDict {
    /// Create a new AdditionalPropertiesDict wrapping a parent AccessProperties.
    fn new(parent: Py<PyAny>) -> Self {
        Self { parent }
    }

    /// Get the additional properties as a Python dict.
    fn get_additional_properties_dict<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyDict>> {
        // Call the parent's internal _get_additional_properties_dict method
        let parent_obj = self.parent.bind(py);
        let props_obj = parent_obj.call_method0("_get_additional_properties_dict")?;
        props_obj.downcast::<PyDict>().cloned().map_err(|e| e.into())
    }

    /// Set a property on the parent AccessProperties.
    fn set_additional_property(&self, py: Python, key: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let parent_obj = self.parent.bind(py);
        parent_obj.call_method1("_set_additional_property", (key, value))?;
        Ok(())
    }

    /// Remove a property from the parent AccessProperties.
    fn remove_additional_property(&self, py: Python, key: String) -> PyResult<()> {
        let parent_obj = self.parent.bind(py);
        parent_obj.call_method1("_remove_additional_property", (key,))?;
        Ok(())
    }
}

// ================================
// AccessPropertyComputer
// ================================

/// Base class for custom access property computers.
///
/// Subclass this class and implement the `compute` and `property_names` methods
/// to create custom property calculations that can be applied to access windows.
///
/// The compute method is called for each access window and should return a dictionary
/// of property names to values. Properties can be scalars, vectors, time series,
/// booleans, strings, or any JSON-serializable value.
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     class DopplerComputer(bh.AccessPropertyComputer):
///         '''Computes Doppler shift at window midtime.'''
///
///         def compute(self, window, satellite_state_ecef, location_ecef):
///             '''
///             Args:
///                 window: AccessWindow with timing information
///                 satellite_state_ecef: Satellite state [x,y,z,vx,vy,vz] in ECEF (m, m/s)
///                 location_ecef: Location position [x,y,z] in ECEF (m)
///
///             Returns:
///                 dict: Property name -> value
///             '''
///             # Extract velocity
///             vx, vy, vz = satellite_state_ecef[3:6]
///
///             # Line-of-sight vector
///             sat_pos = satellite_state_ecef[:3]
///             los = location_ecef - sat_pos
///             los_unit = los / np.linalg.norm(los)
///
///             # Radial velocity
///             sat_vel = np.array([vx, vy, vz])
///             radial_velocity = np.dot(sat_vel, los_unit)
///
///             # Doppler shift (L-band)
///             freq_hz = 1.57542e9  # GPS L1
///             doppler_hz = -radial_velocity * freq_hz / bh.C_LIGHT
///
///             return {"doppler_shift": doppler_hz}
///
///         def property_names(self):
///             '''Return list of property names this computer produces.'''
///             return ["doppler_shift"]
///
///     # Use with access computation (future)
///     computer = DopplerComputer()
///     # accesses = bh.compute_accesses(..., property_computers=[computer])
///     ```
///
/// Notes:
///     - The `compute` method receives ECEF coordinates in SI units (meters, m/s)
///     - Property values are automatically converted to appropriate Rust types
///     - The window parameter provides access to timing via:
///       - `window.window_open`: Start epoch
///       - `window.window_close`: End epoch
///       - `window.midtime()`: Midpoint epoch
///       - `window.duration()`: Duration in seconds
#[pyclass(module = "brahe._brahe", subclass)]
#[pyo3(name = "AccessPropertyComputer")]
pub struct PyAccessPropertyComputer {
    // No internal state - subclasses will override methods
}

#[pymethods]
impl PyAccessPropertyComputer {
    #[new]
    fn new() -> Self {
        PyAccessPropertyComputer {}
    }

    /// Compute custom properties for an access window.
    ///
    /// Override this method in your subclass to implement custom property calculations.
    ///
    /// Args:
    ///     window (AccessWindow): Access window with timing information
    ///     satellite_state_ecef (ndarray): Satellite state in ECEF [x,y,z,vx,vy,vz] (meters, m/s)
    ///     location_ecef (ndarray): Location position in ECEF [x,y,z] (meters)
    ///
    /// Returns:
    ///     dict: Dictionary mapping property names (str) to values (scalar, list, dict, etc.)
    fn compute(
        &self,
        _window: &PyAccessWindow,
        _satellite_state_ecef: PyReadonlyArray1<f64>,
        _location_ecef: PyReadonlyArray1<f64>,
    ) -> PyResult<Py<PyDict>> {
        Err(exceptions::PyNotImplementedError::new_err(
            "Subclasses must implement compute() method",
        ))
    }

    /// Return list of property names this computer will produce.
    ///
    /// Override this method to return the list of property names that your
    /// compute() method will include in its returned dictionary.
    ///
    /// Returns:
    ///     list[str]: List of property names
    fn property_names(&self) -> PyResult<Vec<String>> {
        Err(exceptions::PyNotImplementedError::new_err(
            "Subclasses must implement property_names() method",
        ))
    }
}

// Internal wrapper that implements the Rust AccessPropertyComputer trait
// by calling Python methods
#[allow(dead_code)]
pub(crate) struct RustAccessPropertyComputerWrapper {
    py_computer: Py<PyAny>,
}

#[allow(dead_code)]
impl RustAccessPropertyComputerWrapper {
    pub fn new(py_computer: Py<PyAny>) -> Self {
        RustAccessPropertyComputerWrapper { py_computer }
    }
}

impl AccessPropertyComputer for RustAccessPropertyComputerWrapper {
    fn compute(
        &self,
        window: &AccessWindow,
        state_provider: &dyn crate::orbits::traits::StateProvider,
        location_ecef: &nalgebra::Vector3<f64>,
        _location_geodetic: &nalgebra::Vector3<f64>,
    ) -> Result<HashMap<String, PropertyValue>, BraheError> {
        Python::attach(|py| {
            // Get satellite state at midtime
            let midtime = window.midtime();
            let sat_state = state_provider.state_ecef(midtime);

            // Convert to numpy arrays
            let sat_state_array = sat_state.as_slice().to_pyarray(py).to_owned();
            let loc_array = location_ecef.as_slice().to_pyarray(py).to_owned();

            // Create Python AccessWindow
            let py_window = Py::new(
                py,
                PyAccessWindow {
                    window: window.clone(),
                },
            )
            .map_err(|e| BraheError::Error(format!("Failed to create PyAccessWindow: {}", e)))?;

            // Call Python compute method
            let py_obj = self.py_computer.bind(py);
            let result_dict = py_obj
                .call_method1("compute", (py_window, sat_state_array, loc_array))
                .map_err(|e| {
                    BraheError::Error(format!("Python compute() method failed: {}", e))
                })?;

            // Convert Python dict to Rust HashMap<String, PropertyValue>
            let dict: &Bound<'_, PyDict> = result_dict
                .downcast()
                .map_err(|e| BraheError::Error(format!("compute() must return a dict: {}", e)))?;

            let mut props = HashMap::new();
            for (key, value) in dict.iter() {
                let key_str: String = key.extract().map_err(|e| {
                    BraheError::Error(format!("Property keys must be strings: {}", e))
                })?;

                // Convert Python value to PropertyValue
                let prop_value = python_value_to_property_value(&value)?;
                props.insert(key_str, prop_value);
            }

            Ok(props)
        })
    }

    fn property_names(&self) -> Vec<String> {
        Python::attach(|py| {
            let py_obj = self.py_computer.bind(py);
            py_obj
                .call_method0("property_names")
                .and_then(|result| result.extract())
                .unwrap_or_else(|_| Vec::new())
        })
    }
}

// Helper function to convert Python values to PropertyValue
#[allow(dead_code)]
fn python_value_to_property_value(value: &Bound<'_, PyAny>) -> Result<PropertyValue, BraheError> {
    // Try bool first (before int/float, since bool is a subclass of int in Python)
    if let Ok(b) = value.extract::<bool>() {
        return Ok(PropertyValue::Boolean(b));
    }

    // Try float/int
    if let Ok(f) = value.extract::<f64>() {
        return Ok(PropertyValue::Scalar(f));
    }

    // Try string
    if let Ok(s) = value.extract::<String>() {
        return Ok(PropertyValue::String(s));
    }

    // Try list/array
    if let Ok(vec) = value.extract::<Vec<f64>>() {
        return Ok(PropertyValue::Vector(vec));
    }

    // Try dict (could be TimeSeries or generic JSON)
    if let Ok(dict) = value.downcast::<PyDict>() {
        // Check if it's a time series format
        let has_times = dict.contains("times").map_err(|e| {
            BraheError::Error(format!("Failed to check for 'times' key: {}", e))
        })?;
        let has_values = dict.contains("values").map_err(|e| {
            BraheError::Error(format!("Failed to check for 'values' key: {}", e))
        })?;

        if has_times && has_values {
            let times: Vec<f64> = dict
                .get_item("times")
                .map_err(|e| BraheError::Error(format!("Failed to get 'times': {}", e)))?
                .ok_or_else(|| BraheError::Error("Missing 'times' key".to_string()))?
                .extract()
                .map_err(|e| BraheError::Error(format!("Failed to extract 'times': {}", e)))?;
            let values: Vec<f64> = dict
                .get_item("values")
                .map_err(|e| BraheError::Error(format!("Failed to get 'values': {}", e)))?
                .ok_or_else(|| BraheError::Error("Missing 'values' key".to_string()))?
                .extract()
                .map_err(|e| BraheError::Error(format!("Failed to extract 'values': {}", e)))?;
            return Ok(PropertyValue::TimeSeries { times, values });
        }

        // Otherwise, convert to JSON
        let json_module = value.py().import("json").map_err(|e| {
            BraheError::Error(format!("Failed to import json module: {}", e))
        })?;
        let dumps = json_module.getattr("dumps").map_err(|e| {
            BraheError::Error(format!("Failed to get json.dumps: {}", e))
        })?;
        let json_str: String = dumps
            .call1((value,))
            .and_then(|s| s.extract::<String>())
            .map_err(|e| BraheError::Error(format!("Failed to serialize to JSON: {}", e)))?;

        let json_value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| BraheError::ParseError(format!("Invalid JSON: {}", e)))?;

        return Ok(PropertyValue::Json(json_value));
    }

    // Fallback: try to convert to JSON
    Python::attach(|py| {
        let json_module = py
            .import("json")
            .map_err(|e| BraheError::Error(format!("Failed to import json: {}", e)))?;
        let dumps = json_module
            .getattr("dumps")
            .map_err(|e| BraheError::Error(format!("Failed to get json.dumps: {}", e)))?;
        let json_str: String = dumps
            .call1((value,))
            .and_then(|s| s.extract::<String>())
            .map_err(|e| BraheError::Error(format!("Failed to serialize value: {}", e)))?;

        let json_value: serde_json::Value = serde_json::from_str(&json_str)
            .map_err(|e| BraheError::ParseError(format!("Invalid JSON: {}", e)))?;

        Ok(PropertyValue::Json(json_value))
    })
}

// ================================
// Access Computation Functions
// ================================

/// Configuration for access search grid parameters.
///
/// Controls the time step and adaptive stepping behavior for access window finding.
///
/// Args:
///     initial_time_step (float): Initial time step in seconds for grid search (default: 60.0)
///     adaptive_step (bool): Enable adaptive stepping after first access (default: False)
///     adaptive_fraction (float): Fraction of orbital period to use for adaptive step (default: 0.75)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create a config with custom parameters
///     config = bh.AccessSearchConfig(
///         initial_time_step=30.0,
///         adaptive_step=True,
///         adaptive_fraction=0.5
///     )
///
///     # Use config with location_accesses
///     windows = bh.location_accesses(
///         station, prop, search_start, search_end,
///         constraint, config=config
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "AccessSearchConfig")]
#[derive(Clone)]
pub struct PyAccessSearchConfig {
    pub(crate) config: AccessSearchConfig,
}

#[pymethods]
impl PyAccessSearchConfig {
    #[new]
    #[pyo3(signature = (initial_time_step=60.0, adaptive_step=false, adaptive_fraction=0.75))]
    fn new(initial_time_step: f64, adaptive_step: bool, adaptive_fraction: f64) -> Self {
        Self {
            config: AccessSearchConfig {
                initial_time_step,
                adaptive_step,
                adaptive_fraction,
            },
        }
    }

    /// Get the initial time step in seconds.
    ///
    /// Returns:
    ///     float: Initial time step
    #[getter]
    fn initial_time_step(&self) -> f64 {
        self.config.initial_time_step
    }

    /// Set the initial time step in seconds.
    ///
    /// Args:
    ///     value (float): New initial time step
    #[setter]
    fn set_initial_time_step(&mut self, value: f64) {
        self.config.initial_time_step = value;
    }

    /// Get whether adaptive stepping is enabled.
    ///
    /// Returns:
    ///     bool: Adaptive stepping flag
    #[getter]
    fn adaptive_step(&self) -> bool {
        self.config.adaptive_step
    }

    /// Set whether adaptive stepping is enabled.
    ///
    /// Args:
    ///     value (bool): Enable/disable adaptive stepping
    #[setter]
    fn set_adaptive_step(&mut self, value: bool) {
        self.config.adaptive_step = value;
    }

    /// Get the adaptive fraction (fraction of orbital period).
    ///
    /// Returns:
    ///     float: Adaptive fraction
    #[getter]
    fn adaptive_fraction(&self) -> f64 {
        self.config.adaptive_fraction
    }

    /// Set the adaptive fraction (fraction of orbital period).
    ///
    /// Args:
    ///     value (float): New adaptive fraction
    #[setter]
    fn set_adaptive_fraction(&mut self, value: f64) {
        self.config.adaptive_fraction = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "AccessSearchConfig(initial_time_step={}, adaptive_step={}, adaptive_fraction={})",
            self.config.initial_time_step, self.config.adaptive_step, self.config.adaptive_fraction
        )
    }
}

/// Compute access windows for locations and satellites.
///
/// This function accepts either single items or lists for both locations and propagators,
/// automatically handling all combinations. All location-satellite pairs are computed
/// and results are returned sorted by window start time.
///
/// Args:
///     locations (PointLocation | PolygonLocation | List[PointLocation | PolygonLocation]):
///         Single location or list of locations
///     propagators (SGPPropagator | KeplerianPropagator | List[SGPPropagator | KeplerianPropagator]):
///         Single propagator or list of propagators
///     search_start (Epoch): Start of search window
///     search_end (Epoch): End of search window
///     constraint (AccessConstraint): Access constraints to evaluate
///     property_computers (Optional[List[AccessPropertyComputer]]): Optional property computers
///     config (Optional[AccessSearchConfig]): Search configuration (default: 60s fixed grid, no adaptation)
///     time_tolerance (Optional[float]): Bisection search tolerance in seconds (default: 0.01)
///
/// Returns:
///     List[AccessWindow]: List of access windows sorted by start time
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Create a ground station
///     station = bh.PointLocation(-75.0, 40.0, 0.0)  # Philadelphia
///
///     # Create satellite propagators
///     epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
///     oe = np.array([bh.R_EARTH + 500e3, 0.01, 97.8, 15.0, 30.0, 45.0])
///     state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)
///     prop1 = bh.KeplerianPropagator(epoch, state)
///
///     # Define access constraints
///     constraint = bh.ElevationConstraint(10.0)  # 10 degree minimum elevation
///
///     # Single location, single propagator
///     search_end = epoch + 86400.0  # 1 day
///     windows = bh.location_accesses(station, prop1, epoch, search_end, constraint)
///
///     # Single location, multiple propagators
///     prop2 = bh.KeplerianPropagator(epoch, state)  # Different satellite
///     windows = bh.location_accesses(station, [prop1, prop2], epoch, search_end, constraint)
///
///     # Multiple locations, single propagator
///     station2 = bh.PointLocation(-122.0, 37.0, 0.0)  # San Francisco
///     windows = bh.location_accesses([station, station2], prop1, epoch, search_end, constraint)
///
///     # Multiple locations, multiple propagators
///     windows = bh.location_accesses([station, station2], [prop1, prop2], epoch, search_end, constraint)
///
///     # Custom search configuration
///     config = bh.AccessSearchConfig(initial_time_step=30.0, adaptive_step=True)
///     windows = bh.location_accesses(station, prop1, epoch, search_end, constraint, config=config)
///     ```
#[pyfunction(name = "location_accesses")]
#[pyo3(signature = (locations, propagators, search_start, search_end, constraint, _property_computers=None, config=None, time_tolerance=None))]
#[allow(clippy::too_many_arguments)]
fn py_location_accesses(
    _py: Python,
    locations: &Bound<'_, PyAny>,
    propagators: &Bound<'_, PyAny>,
    search_start: &PyEpoch,
    search_end: &PyEpoch,
    constraint: &Bound<'_, PyAny>,
    _property_computers: Option<Vec<Py<PyAny>>>,
    config: Option<&PyAccessSearchConfig>,
    time_tolerance: Option<f64>,
) -> PyResult<Vec<PyAccessWindow>> {
    use crate::access::compute::location_accesses;
    use pyo3::types::PyList;

    // Use provided config or create default
    let search_config = config.map(|c| c.config).unwrap_or_default();

    // Extract constraint as trait object
    let constraint_trait: &dyn AccessConstraint = if let Ok(c) = constraint.downcast::<PyElevationConstraint>() {
        &c.borrow().constraint
    } else if let Ok(c) = constraint.downcast::<PyOffNadirConstraint>() {
        &c.borrow().constraint
    } else if let Ok(c) = constraint.downcast::<PyLocalTimeConstraint>() {
        &c.borrow().constraint
    } else if let Ok(c) = constraint.downcast::<PyLookDirectionConstraint>() {
        &c.borrow().constraint
    } else if let Ok(c) = constraint.downcast::<PyAscDscConstraint>() {
        &c.borrow().constraint
    } else if let Ok(c) = constraint.downcast::<PyElevationMaskConstraint>() {
        &c.borrow().constraint
    } else if let Ok(c) = constraint.downcast::<PyConstraintAll>() {
        &c.borrow().composite
    } else if let Ok(c) = constraint.downcast::<PyConstraintAny>() {
        &c.borrow().composite
    } else if let Ok(c) = constraint.downcast::<PyConstraintNot>() {
        &c.borrow().composite
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "constraint must be an AccessConstraint type"
        ));
    };

    // Process locations - check if it's a list or single item
    let loc_is_list = locations.is_instance_of::<PyList>();

    // Process propagators - check if it's a list or single item
    let prop_is_list = propagators.is_instance_of::<PyList>();

    // Extract locations as vectors of references
    enum LocationVec {
        Point(Vec<PointLocation>),
        Polygon(Vec<PolygonLocation>),
    }

    let locations_vec = if loc_is_list {
        let list = locations.downcast::<PyList>()?;
        let mut point_locs = Vec::new();
        let mut polygon_locs = Vec::new();
        let mut is_point = true;

        for item in list.iter() {
            if let Ok(loc) = item.downcast::<PyPointLocation>() {
                point_locs.push(loc.borrow().location.clone());
            } else if let Ok(loc) = item.downcast::<PyPolygonLocation>() {
                polygon_locs.push(loc.borrow().location.clone());
                is_point = false;
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "locations list must contain only PointLocation or PolygonLocation objects"
                ));
            }
        }

        if is_point {
            LocationVec::Point(point_locs)
        } else {
            LocationVec::Polygon(polygon_locs)
        }
    } else if let Ok(loc) = locations.downcast::<PyPointLocation>() {
        LocationVec::Point(vec![loc.borrow().location.clone()])
    } else if let Ok(loc) = locations.downcast::<PyPolygonLocation>() {
        LocationVec::Polygon(vec![loc.borrow().location.clone()])
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "locations must be PointLocation, PolygonLocation, or a list of these types"
        ));
    };

    // Extract propagators as vectors
    enum PropagatorVec {
        Sgp(Vec<crate::orbits::sgp_propagator::SGPPropagator>),
        Keplerian(Vec<crate::orbits::keplerian_propagator::KeplerianPropagator>),
    }

    let propagators_vec = if prop_is_list {
        let list = propagators.downcast::<PyList>()?;
        let mut sgp_props = Vec::new();
        let mut kep_props = Vec::new();
        let mut is_sgp = true;

        for item in list.iter() {
            if let Ok(prop) = item.downcast::<PySGPPropagator>() {
                sgp_props.push(prop.borrow().propagator.clone());
            } else if let Ok(prop) = item.downcast::<PyKeplerianPropagator>() {
                kep_props.push(prop.borrow().propagator.clone());
                is_sgp = false;
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "propagators list must contain only SGPPropagator or KeplerianPropagator objects"
                ));
            }
        }

        if is_sgp {
            PropagatorVec::Sgp(sgp_props)
        } else {
            PropagatorVec::Keplerian(kep_props)
        }
    } else if let Ok(prop) = propagators.downcast::<PySGPPropagator>() {
        PropagatorVec::Sgp(vec![prop.borrow().propagator.clone()])
    } else if let Ok(prop) = propagators.downcast::<PyKeplerianPropagator>() {
        PropagatorVec::Keplerian(vec![prop.borrow().propagator.clone()])
    } else {
        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "propagators must be SGPPropagator, KeplerianPropagator, or a list of these types"
        ));
    };

    // Call the Rust function with the extracted types
    let windows = match (&locations_vec, &propagators_vec) {
        (LocationVec::Point(locs), PropagatorVec::Sgp(props)) => {
            location_accesses(
                locs,
                props,
                search_start.obj,
                search_end.obj,
                constraint_trait,
                None,  // property_computers
                Some(&search_config),
                time_tolerance,
            )
        }
        (LocationVec::Point(locs), PropagatorVec::Keplerian(props)) => {
            location_accesses(
                locs,
                props,
                search_start.obj,
                search_end.obj,
                constraint_trait,
                None,  // property_computers
                Some(&search_config),
                time_tolerance,
            )
        }
        (LocationVec::Polygon(locs), PropagatorVec::Sgp(props)) => {
            location_accesses(
                locs,
                props,
                search_start.obj,
                search_end.obj,
                constraint_trait,
                None,  // property_computers
                Some(&search_config),
                time_tolerance,
            )
        }
        (LocationVec::Polygon(locs), PropagatorVec::Keplerian(props)) => {
            location_accesses(
                locs,
                props,
                search_start.obj,
                search_end.obj,
                constraint_trait,
                None,  // property_computers
                Some(&search_config),
                time_tolerance,
            )
        }
    };

    // Convert to Python windows
    Ok(windows.into_iter().map(|w| PyAccessWindow { window: w }).collect())
}
