// Python bindings for the SpaceTrack module.

use crate::spacetrack;
use crate::spacetrack::query::{
    SpaceTrackOperator, SpaceTrackOrder, SpaceTrackPredicate, SpaceTrackQuery, SpaceTrackValue,
};

// ============================================================================
// SpaceTrackValue - Type-safe query values
// ============================================================================

/// Value type for SpaceTrack queries.
///
/// This enum provides type-safe handling of different value types,
/// with automatic formatting for the SpaceTrack API.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Values are automatically converted from Python types
///     value = bh.SpaceTrackValue.from_int(25544)
///     value = bh.SpaceTrackValue.from_float(0.01)
///     value = bh.SpaceTrackValue.from_string("ISS")
///     value = bh.SpaceTrackValue.from_epoch(epoch)
///     ```
#[pyclass(name = "SpaceTrackValue", module = "brahe._brahe")]
#[derive(Clone)]
pub struct PySpaceTrackValue {
    pub(crate) inner: SpaceTrackValue,
}

#[pymethods]
impl PySpaceTrackValue {
    /// Create a value from an integer.
    #[staticmethod]
    fn from_int(v: i64) -> Self {
        Self {
            inner: SpaceTrackValue::Integer(v),
        }
    }

    /// Create a value from a float.
    #[staticmethod]
    fn from_float(v: f64) -> Self {
        Self {
            inner: SpaceTrackValue::Float(v),
        }
    }

    /// Create a value from a string.
    #[staticmethod]
    fn from_string(v: &str) -> Self {
        Self {
            inner: SpaceTrackValue::String(v.to_string()),
        }
    }

    /// Create a value from an Epoch (auto-formatted to ISO string).
    #[staticmethod]
    fn from_epoch(epoch: &PyEpoch) -> Self {
        Self {
            inner: SpaceTrackValue::Epoch(epoch.obj),
        }
    }

    /// Create a value from a boolean.
    #[staticmethod]
    fn from_bool(v: bool) -> Self {
        Self {
            inner: SpaceTrackValue::Boolean(v),
        }
    }

    /// Convert to query string format.
    fn to_query_string(&self) -> String {
        self.inner.to_query_string()
    }

    fn __repr__(&self) -> String {
        format!("SpaceTrackValue({})", self.inner.to_query_string())
    }
}

// ============================================================================
// SpaceTrackOrder - Query result ordering
// ============================================================================

/// Order direction for query results.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Use when setting query ordering
///     query = bh.SpaceTrackQuery.gp().order_by(predicate, bh.SpaceTrackOrder.DESCENDING)
///     ```
#[pyclass(name = "SpaceTrackOrder", module = "brahe._brahe")]
#[derive(Clone)]
pub struct PySpaceTrackOrder {
    pub(crate) inner: SpaceTrackOrder,
}

#[pymethods]
impl PySpaceTrackOrder {
    /// Ascending order (A-Z, 0-9, oldest-newest).
    #[classattr]
    #[pyo3(name = "ASCENDING")]
    fn ascending() -> Self {
        Self {
            inner: SpaceTrackOrder::Ascending,
        }
    }

    /// Descending order (Z-A, 9-0, newest-oldest).
    #[classattr]
    #[pyo3(name = "DESCENDING")]
    fn descending() -> Self {
        Self {
            inner: SpaceTrackOrder::Descending,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            SpaceTrackOrder::Ascending => "SpaceTrackOrder.ASCENDING".to_string(),
            SpaceTrackOrder::Descending => "SpaceTrackOrder.DESCENDING".to_string(),
        }
    }
}

// ============================================================================
// SpaceTrackPredicateBuilder - Fluent predicate construction
// ============================================================================

/// Builder for constructing SpaceTrack predicates with fluent syntax.
///
/// This class is created by `SpaceTrackPredicate.field()` or one of the
/// typed field constructors, and provides methods for specifying the comparison
/// operator and value.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # The builder is consumed when an operator method is called
///     predicate = bh.SpaceTrackPredicate.epoch().gt("2024-01-01")
///     ```
#[pyclass(name = "SpaceTrackPredicateBuilder", module = "brahe._brahe")]
#[derive(Clone)]
pub struct PySpaceTrackPredicateBuilder {
    pub(crate) field: String,
}

#[pymethods]
impl PySpaceTrackPredicateBuilder {
    /// Get the field name.
    #[getter]
    fn field_name(&self) -> &str {
        &self.field
    }

    /// Create an equals predicate.
    ///
    /// Args:
    ///     value: The value to compare (int, float, str, or Epoch)
    ///
    /// Returns:
    ///     SpaceTrackPredicate: The completed predicate
    #[pyo3(signature = (value))]
    fn eq(&self, value: &Bound<'_, PyAny>) -> PyResult<PySpaceTrackPredicate> {
        let v = py_to_spacetrack_value(value)?;
        Ok(PySpaceTrackPredicate {
            inner: SpaceTrackPredicate {
                field: self.field.clone(),
                operator: SpaceTrackOperator::Equals(v),
            },
        })
    }

    /// Create a greater-than predicate.
    ///
    /// Args:
    ///     value: The value to compare
    ///
    /// Returns:
    ///     SpaceTrackPredicate: The completed predicate
    #[pyo3(signature = (value))]
    fn gt(&self, value: &Bound<'_, PyAny>) -> PyResult<PySpaceTrackPredicate> {
        let v = py_to_spacetrack_value(value)?;
        Ok(PySpaceTrackPredicate {
            inner: SpaceTrackPredicate {
                field: self.field.clone(),
                operator: SpaceTrackOperator::GreaterThan(v),
            },
        })
    }

    /// Create a less-than predicate.
    ///
    /// Args:
    ///     value: The value to compare
    ///
    /// Returns:
    ///     SpaceTrackPredicate: The completed predicate
    #[pyo3(signature = (value))]
    fn lt(&self, value: &Bound<'_, PyAny>) -> PyResult<PySpaceTrackPredicate> {
        let v = py_to_spacetrack_value(value)?;
        Ok(PySpaceTrackPredicate {
            inner: SpaceTrackPredicate {
                field: self.field.clone(),
                operator: SpaceTrackOperator::LessThan(v),
            },
        })
    }

    /// Create a greater-than-or-equal predicate.
    ///
    /// Args:
    ///     value: The value to compare
    ///
    /// Returns:
    ///     SpaceTrackPredicate: The completed predicate
    #[pyo3(signature = (value))]
    fn gte(&self, value: &Bound<'_, PyAny>) -> PyResult<PySpaceTrackPredicate> {
        let v = py_to_spacetrack_value(value)?;
        Ok(PySpaceTrackPredicate {
            inner: SpaceTrackPredicate {
                field: self.field.clone(),
                operator: SpaceTrackOperator::GreaterThanOrEqual(v),
            },
        })
    }

    /// Create a less-than-or-equal predicate.
    ///
    /// Args:
    ///     value: The value to compare
    ///
    /// Returns:
    ///     SpaceTrackPredicate: The completed predicate
    #[pyo3(signature = (value))]
    fn lte(&self, value: &Bound<'_, PyAny>) -> PyResult<PySpaceTrackPredicate> {
        let v = py_to_spacetrack_value(value)?;
        Ok(PySpaceTrackPredicate {
            inner: SpaceTrackPredicate {
                field: self.field.clone(),
                operator: SpaceTrackOperator::LessThanOrEqual(v),
            },
        })
    }

    /// Create a not-equal predicate.
    ///
    /// Args:
    ///     value: The value to compare
    ///
    /// Returns:
    ///     SpaceTrackPredicate: The completed predicate
    #[pyo3(signature = (value))]
    fn ne(&self, value: &Bound<'_, PyAny>) -> PyResult<PySpaceTrackPredicate> {
        let v = py_to_spacetrack_value(value)?;
        Ok(PySpaceTrackPredicate {
            inner: SpaceTrackPredicate {
                field: self.field.clone(),
                operator: SpaceTrackOperator::NotEqual(v),
            },
        })
    }

    /// Create an inclusive range predicate.
    ///
    /// Args:
    ///     start: The start value of the range
    ///     end: The end value of the range
    ///
    /// Returns:
    ///     SpaceTrackPredicate: The completed predicate
    #[pyo3(signature = (start, end))]
    fn between(
        &self,
        start: &Bound<'_, PyAny>,
        end: &Bound<'_, PyAny>,
    ) -> PyResult<PySpaceTrackPredicate> {
        let s = py_to_spacetrack_value(start)?;
        let e = py_to_spacetrack_value(end)?;
        Ok(PySpaceTrackPredicate {
            inner: SpaceTrackPredicate {
                field: self.field.clone(),
                operator: SpaceTrackOperator::Range(s, e),
            },
        })
    }

    /// Create a SQL LIKE pattern predicate.
    ///
    /// The pattern uses SQL LIKE syntax where:
    /// - `%` matches any sequence of characters
    /// - `_` matches any single character
    ///
    /// Args:
    ///     pattern (str): The LIKE pattern
    ///
    /// Returns:
    ///     SpaceTrackPredicate: The completed predicate
    #[pyo3(signature = (pattern))]
    fn like(&self, pattern: &str) -> PySpaceTrackPredicate {
        PySpaceTrackPredicate {
            inner: SpaceTrackPredicate {
                field: self.field.clone(),
                operator: SpaceTrackOperator::Like(pattern.to_string()),
            },
        }
    }

    /// Create a starts-with predicate.
    ///
    /// Args:
    ///     prefix (str): The prefix to match
    ///
    /// Returns:
    ///     SpaceTrackPredicate: The completed predicate
    #[pyo3(signature = (prefix))]
    fn starts_with(&self, prefix: &str) -> PySpaceTrackPredicate {
        PySpaceTrackPredicate {
            inner: SpaceTrackPredicate {
                field: self.field.clone(),
                operator: SpaceTrackOperator::StartsWith(prefix.to_string()),
            },
        }
    }

    /// Create a contains predicate (wraps with `%` for LIKE pattern).
    ///
    /// Args:
    ///     substring (str): The substring to match
    ///
    /// Returns:
    ///     SpaceTrackPredicate: The completed predicate
    #[pyo3(signature = (substring))]
    fn contains(&self, substring: &str) -> PySpaceTrackPredicate {
        PySpaceTrackPredicate {
            inner: SpaceTrackPredicate {
                field: self.field.clone(),
                operator: SpaceTrackOperator::Like(format!("%{}%", substring)),
            },
        }
    }

    /// Create a null value predicate.
    ///
    /// Returns:
    ///     SpaceTrackPredicate: The completed predicate
    fn is_null(&self) -> PySpaceTrackPredicate {
        PySpaceTrackPredicate {
            inner: SpaceTrackPredicate {
                field: self.field.clone(),
                operator: SpaceTrackOperator::Null,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!("SpaceTrackPredicateBuilder(field='{}')", self.field)
    }
}

// ============================================================================
// SpaceTrackPredicate - A complete predicate filter
// ============================================================================

/// A single predicate filter for a SpaceTrack query.
///
/// Predicates combine a field name with an operator and value(s) to create
/// filter conditions for queries.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Using typed field constructors
///     predicate = bh.SpaceTrackPredicate.norad_cat_id().eq(25544)
///
///     # Using generic field constructor
///     predicate = bh.SpaceTrackPredicate.field("CUSTOM_FIELD").gt(100)
///     ```
#[pyclass(name = "SpaceTrackPredicate", module = "brahe._brahe")]
#[derive(Clone)]
pub struct PySpaceTrackPredicate {
    pub(crate) inner: SpaceTrackPredicate,
}

#[pymethods]
impl PySpaceTrackPredicate {
    /// Create a predicate builder for a generic field by name.
    ///
    /// Args:
    ///     name (str): The field name (will be uppercased)
    ///
    /// Returns:
    ///     SpaceTrackPredicateBuilder: A builder for creating the predicate
    #[staticmethod]
    fn field(name: &str) -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: name.to_uppercase(),
        }
    }

    /// Create a predicate builder for the NORAD_CAT_ID field.
    #[staticmethod]
    fn norad_cat_id() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "NORAD_CAT_ID".to_string(),
        }
    }

    /// Create a predicate builder for the OBJECT_NAME field.
    #[staticmethod]
    fn object_name() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "OBJECT_NAME".to_string(),
        }
    }

    /// Create a predicate builder for the OBJECT_ID field.
    #[staticmethod]
    fn object_id() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "OBJECT_ID".to_string(),
        }
    }

    /// Create a predicate builder for the EPOCH field.
    #[staticmethod]
    fn epoch() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "EPOCH".to_string(),
        }
    }

    /// Create a predicate builder for the COUNTRY_CODE field.
    #[staticmethod]
    fn country_code() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "COUNTRY_CODE".to_string(),
        }
    }

    /// Create a predicate builder for the LAUNCH_DATE field.
    #[staticmethod]
    fn launch_date() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "LAUNCH_DATE".to_string(),
        }
    }

    /// Create a predicate builder for the DECAY_DATE field.
    #[staticmethod]
    fn decay_date() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "DECAY_DATE".to_string(),
        }
    }

    /// Create a predicate builder for the ECCENTRICITY field.
    #[staticmethod]
    fn eccentricity() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "ECCENTRICITY".to_string(),
        }
    }

    /// Create a predicate builder for the INCLINATION field.
    #[staticmethod]
    fn inclination() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "INCLINATION".to_string(),
        }
    }

    /// Create a predicate builder for the MEAN_MOTION field.
    #[staticmethod]
    fn mean_motion() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "MEAN_MOTION".to_string(),
        }
    }

    /// Create a predicate builder for the OBJECT_TYPE field.
    #[staticmethod]
    fn object_type() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "OBJECT_TYPE".to_string(),
        }
    }

    /// Create a predicate builder for the RCS_SIZE field.
    #[staticmethod]
    fn rcs_size() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "RCS_SIZE".to_string(),
        }
    }

    /// Create a predicate builder for the PERIOD field.
    #[staticmethod]
    fn period() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "PERIOD".to_string(),
        }
    }

    /// Create a predicate builder for the APOAPSIS field.
    #[staticmethod]
    fn apoapsis() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "APOAPSIS".to_string(),
        }
    }

    /// Create a predicate builder for the PERIAPSIS field.
    #[staticmethod]
    fn periapsis() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "PERIAPSIS".to_string(),
        }
    }

    /// Create a predicate builder for the SEMIMAJOR_AXIS field.
    #[staticmethod]
    fn semimajor_axis() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "SEMIMAJOR_AXIS".to_string(),
        }
    }

    /// Create a predicate builder for the BSTAR field.
    #[staticmethod]
    fn bstar() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "BSTAR".to_string(),
        }
    }

    /// Create a predicate builder for the CREATION_DATE field.
    #[staticmethod]
    fn creation_date() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "CREATION_DATE".to_string(),
        }
    }

    /// Create a predicate builder for the DECAYED field.
    #[staticmethod]
    fn decayed() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "DECAYED".to_string(),
        }
    }

    /// Create a predicate builder for the SITE field.
    #[staticmethod]
    fn site() -> PySpaceTrackPredicateBuilder {
        PySpaceTrackPredicateBuilder {
            field: "SITE".to_string(),
        }
    }

    /// Get the field name.
    #[getter]
    fn field_name(&self) -> &str {
        &self.inner.field
    }

    fn __repr__(&self) -> String {
        format!("SpaceTrackPredicate(field='{}')", self.inner.field)
    }
}

// ============================================================================
// SpaceTrackQuery - Query builder
// ============================================================================

/// Builder for composing SpaceTrack API queries.
///
/// This class provides a fluent API for constructing queries with filters,
/// ordering, and pagination.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     query = (bh.SpaceTrackQuery.gp()
///         .filter(bh.SpaceTrackPredicate.norad_cat_id().eq(25544))
///         .filter(bh.SpaceTrackPredicate.epoch().gt("2024-01-01"))
///         .order_by_desc(bh.SpaceTrackPredicate.epoch())
///         .limit(10))
///     ```
#[pyclass(name = "SpaceTrackQuery", module = "brahe._brahe")]
#[derive(Clone)]
pub struct PySpaceTrackQuery {
    pub(crate) inner: SpaceTrackQuery,
}

#[pymethods]
impl PySpaceTrackQuery {
    /// Create a new query for the specified controller and class.
    ///
    /// Args:
    ///     controller (str): The controller name (e.g., "basicspacedata")
    ///     class_name (str): The class name (e.g., "gp")
    ///
    /// Returns:
    ///     SpaceTrackQuery: A new query builder
    #[new]
    fn new(controller: &str, class_name: &str) -> Self {
        Self {
            inner: SpaceTrackQuery::new(controller, class_name),
        }
    }

    /// Create a query for the GP (General Perturbations) class.
    #[staticmethod]
    fn gp() -> Self {
        Self {
            inner: SpaceTrackQuery::gp(),
        }
    }

    /// Create a query for the TLE class.
    #[staticmethod]
    fn tle() -> Self {
        Self {
            inner: SpaceTrackQuery::tle(),
        }
    }

    /// Create a query for the SATCAT (Satellite Catalog) class.
    #[staticmethod]
    fn satcat() -> Self {
        Self {
            inner: SpaceTrackQuery::satcat(),
        }
    }

    /// Create a query for the OMM (Orbit Mean-Elements Message) class.
    #[staticmethod]
    fn omm() -> Self {
        Self {
            inner: SpaceTrackQuery::omm(),
        }
    }

    /// Create a query for the DECAY class.
    #[staticmethod]
    fn decay() -> Self {
        Self {
            inner: SpaceTrackQuery::decay(),
        }
    }

    /// Create a query for the TIP class.
    #[staticmethod]
    fn tip() -> Self {
        Self {
            inner: SpaceTrackQuery::tip(),
        }
    }

    /// Create a query for the GP_HISTORY class.
    #[staticmethod]
    fn gp_history() -> Self {
        Self {
            inner: SpaceTrackQuery::gp_history(),
        }
    }

    /// Create a query for the SATCAT_CHANGE class.
    #[staticmethod]
    fn satcat_change() -> Self {
        Self {
            inner: SpaceTrackQuery::satcat_change(),
        }
    }

    /// Create a query for the SATCAT_DEBUT class.
    #[staticmethod]
    fn satcat_debut() -> Self {
        Self {
            inner: SpaceTrackQuery::satcat_debut(),
        }
    }

    /// Create a query for the LAUNCH_SITE class.
    #[staticmethod]
    fn launch_site() -> Self {
        Self {
            inner: SpaceTrackQuery::launch_site(),
        }
    }

    /// Create a query for the BOXSCORE class.
    #[staticmethod]
    fn boxscore() -> Self {
        Self {
            inner: SpaceTrackQuery::boxscore(),
        }
    }

    /// Create a query for the CDM_PUBLIC class.
    #[staticmethod]
    fn cdm_public() -> Self {
        Self {
            inner: SpaceTrackQuery::cdm_public(),
        }
    }

    /// Create a query for the ANNOUNCEMENT class.
    #[staticmethod]
    fn announcement() -> Self {
        Self {
            inner: SpaceTrackQuery::announcement(),
        }
    }

    /// Add a predicate filter to the query.
    ///
    /// Args:
    ///     predicate (SpaceTrackPredicate): The filter predicate
    ///
    /// Returns:
    ///     SpaceTrackQuery: Self for chaining
    fn filter(&self, predicate: &PySpaceTrackPredicate) -> Self {
        Self {
            inner: self.inner.clone().filter(predicate.inner.clone()),
        }
    }

    /// Set the maximum number of results.
    ///
    /// Args:
    ///     limit (int): Maximum number of results
    ///
    /// Returns:
    ///     SpaceTrackQuery: Self for chaining
    fn limit(&self, limit: u32) -> Self {
        Self {
            inner: self.inner.clone().limit(limit),
        }
    }

    /// Set ordering by predicate field with specified direction.
    ///
    /// Args:
    ///     field (SpaceTrackPredicateBuilder): The field to order by
    ///     order (SpaceTrackOrder): The order direction
    ///
    /// Returns:
    ///     SpaceTrackQuery: Self for chaining
    fn order_by(&self, field: &PySpaceTrackPredicateBuilder, order: &PySpaceTrackOrder) -> Self {
        Self {
            inner: self.inner.clone().order_by(
                spacetrack::query::SpaceTrackPredicateBuilder {
                    field: field.field.clone(),
                },
                order.inner,
            ),
        }
    }

    /// Set ordering by field in ascending order.
    ///
    /// Args:
    ///     field (SpaceTrackPredicateBuilder): The field to order by
    ///
    /// Returns:
    ///     SpaceTrackQuery: Self for chaining
    fn order_by_asc(&self, field: &PySpaceTrackPredicateBuilder) -> Self {
        Self {
            inner: self.inner.clone().order_by_asc(
                spacetrack::query::SpaceTrackPredicateBuilder {
                    field: field.field.clone(),
                },
            ),
        }
    }

    /// Set ordering by field in descending order.
    ///
    /// Args:
    ///     field (SpaceTrackPredicateBuilder): The field to order by
    ///
    /// Returns:
    ///     SpaceTrackQuery: Self for chaining
    fn order_by_desc(&self, field: &PySpaceTrackPredicateBuilder) -> Self {
        Self {
            inner: self.inner.clone().order_by_desc(
                spacetrack::query::SpaceTrackPredicateBuilder {
                    field: field.field.clone(),
                },
            ),
        }
    }

    /// Enable distinct results only.
    ///
    /// Returns:
    ///     SpaceTrackQuery: Self for chaining
    fn distinct(&self) -> Self {
        Self {
            inner: self.inner.clone().distinct(),
        }
    }

    /// Get the controller name.
    #[getter]
    fn controller(&self) -> &str {
        self.inner.controller()
    }

    /// Get the class name.
    #[getter]
    fn class_name(&self) -> &str {
        self.inner.class()
    }

    fn __repr__(&self) -> String {
        format!(
            "SpaceTrackQuery(controller='{}', class='{}')",
            self.inner.controller(),
            self.inner.class()
        )
    }
}

// ============================================================================
// Helper function to convert Python value to SpaceTrackValue
// ============================================================================

fn py_to_spacetrack_value(value: &Bound<'_, PyAny>) -> PyResult<SpaceTrackValue> {
    // Try to extract as PyEpoch first
    if let Ok(epoch) = value.extract::<PyEpoch>() {
        return Ok(SpaceTrackValue::Epoch(epoch.obj));
    }

    // Try to extract as PySpaceTrackValue
    if let Ok(st_value) = value.extract::<PySpaceTrackValue>() {
        return Ok(st_value.inner);
    }

    // Try bool before int (since bool is a subtype of int in Python)
    if let Ok(v) = value.extract::<bool>() {
        return Ok(SpaceTrackValue::Boolean(v));
    }

    // Try int
    if let Ok(v) = value.extract::<i64>() {
        return Ok(SpaceTrackValue::Integer(v));
    }

    // Try float
    if let Ok(v) = value.extract::<f64>() {
        return Ok(SpaceTrackValue::Float(v));
    }

    // Try string
    if let Ok(v) = value.extract::<String>() {
        return Ok(SpaceTrackValue::String(v));
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Value must be int, float, str, bool, Epoch, or SpaceTrackValue",
    ))
}

// ============================================================================
// Helper function for converting records to dicts
// ============================================================================

/// Helper function to convert a Rust record to a Python dict with snake_case keys.
#[allow(deprecated)]
fn record_to_py_dict<T: serde::Serialize>(py: Python<'_>, record: &T) -> PyResult<Py<PyDict>> {
    let json_str = serde_json::to_string(record)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    let json_module = py.import("json")?;
    let dict = json_module.call_method1("loads", (json_str,))?;
    let py_dict: &Bound<'_, PyDict> = dict.downcast()?;

    // Convert UPPERCASE keys to snake_case
    let new_dict = PyDict::new(py);
    for (key, value) in py_dict.iter() {
        let key_str: String = key.extract()?;
        let snake_key = key_str.to_lowercase();
        new_dict.set_item(snake_key, value)?;
    }
    Ok(new_dict.into())
}

// ============================================================================
// GP Record - General Perturbations orbital elements
// ============================================================================

/// GP record containing orbital elements from Space-Track.
///
/// This record represents a GP (General Perturbations) element set as returned
/// by the SpaceTrack API.
///
/// Attributes:
///     norad_cat_id (int | None): NORAD catalog ID
///     object_name (str | None): Object name
///     object_id (str | None): Object ID (international designator)
///     epoch (str | None): Epoch of element set
///     mean_motion (float | None): Mean motion (rev/day)
///     eccentricity (float | None): Eccentricity
///     inclination (float | None): Inclination (degrees)
///     ra_of_asc_node (float | None): Right ascension of ascending node (degrees)
///     arg_of_pericenter (float | None): Argument of pericenter (degrees)
///     mean_anomaly (float | None): Mean anomaly (degrees)
///     bstar (float | None): BSTAR drag term
///     semimajor_axis (float | None): Semi-major axis (km)
///     period (float | None): Orbital period (minutes)
///     apoapsis (float | None): Apoapsis altitude (km)
///     periapsis (float | None): Periapsis altitude (km)
///     object_type (str | None): Object type (PAYLOAD, ROCKET BODY, DEBRIS)
///     rcs_size (str | None): RCS size category
///     country_code (str | None): Country code
///     launch_date (str | None): Launch date
///     decay_date (str | None): Decay date (if decayed)
///     tle_line0 (str | None): TLE line 0 (name)
///     tle_line1 (str | None): TLE line 1
///     tle_line2 (str | None): TLE line 2
///
/// Example:
///     ```python
///     import brahe as bh
///
///     client = bh.SpaceTrackClient("username", "password")
///     records = client.gp(norad_cat_id=25544, limit=1)
///     record = records[0]
///     print(f"ISS epoch: {record.epoch}")
///     print(f"Inclination: {record.inclination}°")
///     ```
#[pyclass(name = "GPRecord", module = "brahe._brahe")]
pub struct PyGPRecord {
    pub(crate) record: spacetrack::GPRecord,
}

#[pymethods]
impl PyGPRecord {
    #[getter]
    fn ccsds_omm_vers(&self) -> Option<String> { self.record.ccsds_omm_vers.clone() }
    #[getter]
    fn comment(&self) -> Option<String> { self.record.comment.clone() }
    #[getter]
    fn creation_date(&self) -> Option<String> { self.record.creation_date.clone() }
    #[getter]
    fn originator(&self) -> Option<String> { self.record.originator.clone() }
    #[getter]
    fn object_name(&self) -> Option<String> { self.record.object_name.clone() }
    #[getter]
    fn object_id(&self) -> Option<String> { self.record.object_id.clone() }
    #[getter]
    fn center_name(&self) -> Option<String> { self.record.center_name.clone() }
    #[getter]
    fn ref_frame(&self) -> Option<String> { self.record.ref_frame.clone() }
    #[getter]
    fn time_system(&self) -> Option<String> { self.record.time_system.clone() }
    #[getter]
    fn mean_element_theory(&self) -> Option<String> { self.record.mean_element_theory.clone() }
    #[getter]
    fn epoch(&self) -> Option<String> { self.record.epoch.clone() }
    #[getter]
    fn mean_motion(&self) -> Option<f64> { self.record.mean_motion }
    #[getter]
    fn eccentricity(&self) -> Option<f64> { self.record.eccentricity }
    #[getter]
    fn inclination(&self) -> Option<f64> { self.record.inclination }
    #[getter]
    fn ra_of_asc_node(&self) -> Option<f64> { self.record.ra_of_asc_node }
    #[getter]
    fn arg_of_pericenter(&self) -> Option<f64> { self.record.arg_of_pericenter }
    #[getter]
    fn mean_anomaly(&self) -> Option<f64> { self.record.mean_anomaly }
    #[getter]
    fn ephemeris_type(&self) -> Option<i32> { self.record.ephemeris_type }
    #[getter]
    fn classification_type(&self) -> Option<String> { self.record.classification_type.clone() }
    #[getter]
    fn norad_cat_id(&self) -> Option<u32> { self.record.norad_cat_id }
    #[getter]
    fn element_set_no(&self) -> Option<u32> { self.record.element_set_no }
    #[getter]
    fn rev_at_epoch(&self) -> Option<u32> { self.record.rev_at_epoch }
    #[getter]
    fn bstar(&self) -> Option<f64> { self.record.bstar }
    #[getter]
    fn mean_motion_dot(&self) -> Option<f64> { self.record.mean_motion_dot }
    #[getter]
    fn mean_motion_ddot(&self) -> Option<f64> { self.record.mean_motion_ddot }
    #[getter]
    fn semimajor_axis(&self) -> Option<f64> { self.record.semimajor_axis }
    #[getter]
    fn period(&self) -> Option<f64> { self.record.period }
    #[getter]
    fn apoapsis(&self) -> Option<f64> { self.record.apoapsis }
    #[getter]
    fn periapsis(&self) -> Option<f64> { self.record.periapsis }
    #[getter]
    fn object_type(&self) -> Option<String> { self.record.object_type.clone() }
    #[getter]
    fn rcs_size(&self) -> Option<String> { self.record.rcs_size.clone() }
    #[getter]
    fn country_code(&self) -> Option<String> { self.record.country_code.clone() }
    #[getter]
    fn launch_date(&self) -> Option<String> { self.record.launch_date.clone() }
    #[getter]
    fn site(&self) -> Option<String> { self.record.site.clone() }
    #[getter]
    fn decay_date(&self) -> Option<String> { self.record.decay_date.clone() }
    #[getter]
    fn decayed(&self) -> Option<i32> { self.record.decayed }
    #[getter]
    fn file(&self) -> Option<u64> { self.record.file }
    #[getter]
    fn gp_id(&self) -> Option<u64> { self.record.gp_id }
    #[getter]
    fn tle_line0(&self) -> Option<String> { self.record.tle_line0.clone() }
    #[getter]
    fn tle_line1(&self) -> Option<String> { self.record.tle_line1.clone() }
    #[getter]
    fn tle_line2(&self) -> Option<String> { self.record.tle_line2.clone() }

    /// Convert this record to a dictionary with snake_case keys.
    ///
    /// Returns:
    ///     dict: Record data as a dictionary
    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "GPRecord(norad_cat_id={:?}, object_name={:?}, epoch={:?})",
            self.record.norad_cat_id, self.record.object_name, self.record.epoch
        )
    }
}

impl From<spacetrack::GPRecord> for PyGPRecord {
    fn from(record: spacetrack::GPRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// GP History Record - Historical GP elements
// ============================================================================

/// GP History record containing historical orbital elements.
#[pyclass(name = "GPHistoryRecord", module = "brahe._brahe")]
pub struct PyGPHistoryRecord {
    pub(crate) record: spacetrack::GPHistoryRecord,
}

#[pymethods]
impl PyGPHistoryRecord {
    #[getter]
    fn ccsds_omm_vers(&self) -> Option<String> { self.record.ccsds_omm_vers.clone() }
    #[getter]
    fn comment(&self) -> Option<String> { self.record.comment.clone() }
    #[getter]
    fn creation_date(&self) -> Option<String> { self.record.creation_date.clone() }
    #[getter]
    fn originator(&self) -> Option<String> { self.record.originator.clone() }
    #[getter]
    fn object_name(&self) -> Option<String> { self.record.object_name.clone() }
    #[getter]
    fn object_id(&self) -> Option<String> { self.record.object_id.clone() }
    #[getter]
    fn center_name(&self) -> Option<String> { self.record.center_name.clone() }
    #[getter]
    fn ref_frame(&self) -> Option<String> { self.record.ref_frame.clone() }
    #[getter]
    fn time_system(&self) -> Option<String> { self.record.time_system.clone() }
    #[getter]
    fn mean_element_theory(&self) -> Option<String> { self.record.mean_element_theory.clone() }
    #[getter]
    fn epoch(&self) -> Option<String> { self.record.epoch.clone() }
    #[getter]
    fn mean_motion(&self) -> Option<f64> { self.record.mean_motion }
    #[getter]
    fn eccentricity(&self) -> Option<f64> { self.record.eccentricity }
    #[getter]
    fn inclination(&self) -> Option<f64> { self.record.inclination }
    #[getter]
    fn ra_of_asc_node(&self) -> Option<f64> { self.record.ra_of_asc_node }
    #[getter]
    fn arg_of_pericenter(&self) -> Option<f64> { self.record.arg_of_pericenter }
    #[getter]
    fn mean_anomaly(&self) -> Option<f64> { self.record.mean_anomaly }
    #[getter]
    fn ephemeris_type(&self) -> Option<i32> { self.record.ephemeris_type }
    #[getter]
    fn classification_type(&self) -> Option<String> { self.record.classification_type.clone() }
    #[getter]
    fn norad_cat_id(&self) -> Option<u32> { self.record.norad_cat_id }
    #[getter]
    fn element_set_no(&self) -> Option<u32> { self.record.element_set_no }
    #[getter]
    fn rev_at_epoch(&self) -> Option<u32> { self.record.rev_at_epoch }
    #[getter]
    fn bstar(&self) -> Option<f64> { self.record.bstar }
    #[getter]
    fn mean_motion_dot(&self) -> Option<f64> { self.record.mean_motion_dot }
    #[getter]
    fn mean_motion_ddot(&self) -> Option<f64> { self.record.mean_motion_ddot }
    #[getter]
    fn semimajor_axis(&self) -> Option<f64> { self.record.semimajor_axis }
    #[getter]
    fn period(&self) -> Option<f64> { self.record.period }
    #[getter]
    fn apoapsis(&self) -> Option<f64> { self.record.apoapsis }
    #[getter]
    fn periapsis(&self) -> Option<f64> { self.record.periapsis }
    #[getter]
    fn object_type(&self) -> Option<String> { self.record.object_type.clone() }
    #[getter]
    fn rcs_size(&self) -> Option<String> { self.record.rcs_size.clone() }
    #[getter]
    fn country_code(&self) -> Option<String> { self.record.country_code.clone() }
    #[getter]
    fn launch_date(&self) -> Option<String> { self.record.launch_date.clone() }
    #[getter]
    fn site(&self) -> Option<String> { self.record.site.clone() }
    #[getter]
    fn decay_date(&self) -> Option<String> { self.record.decay_date.clone() }
    #[getter]
    fn decayed(&self) -> Option<i32> { self.record.decayed }
    #[getter]
    fn file(&self) -> Option<u64> { self.record.file }
    #[getter]
    fn gp_id(&self) -> Option<u64> { self.record.gp_id }
    #[getter]
    fn tle_line0(&self) -> Option<String> { self.record.tle_line0.clone() }
    #[getter]
    fn tle_line1(&self) -> Option<String> { self.record.tle_line1.clone() }
    #[getter]
    fn tle_line2(&self) -> Option<String> { self.record.tle_line2.clone() }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "GPHistoryRecord(norad_cat_id={:?}, object_name={:?}, epoch={:?})",
            self.record.norad_cat_id, self.record.object_name, self.record.epoch
        )
    }
}

impl From<spacetrack::GPHistoryRecord> for PyGPHistoryRecord {
    fn from(record: spacetrack::GPHistoryRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// OMM Record - Orbit Mean-elements Message
// ============================================================================

/// OMM record containing orbital mean elements.
#[pyclass(name = "OMMRecord", module = "brahe._brahe")]
pub struct PyOMMRecord {
    pub(crate) record: spacetrack::OMMRecord,
}

#[pymethods]
impl PyOMMRecord {
    #[getter]
    fn ccsds_omm_vers(&self) -> Option<String> { self.record.ccsds_omm_vers.clone() }
    #[getter]
    fn comment(&self) -> Option<String> { self.record.comment.clone() }
    #[getter]
    fn creation_date(&self) -> Option<String> { self.record.creation_date.clone() }
    #[getter]
    fn originator(&self) -> Option<String> { self.record.originator.clone() }
    #[getter]
    fn object_name(&self) -> Option<String> { self.record.object_name.clone() }
    #[getter]
    fn object_id(&self) -> Option<String> { self.record.object_id.clone() }
    #[getter]
    fn center_name(&self) -> Option<String> { self.record.center_name.clone() }
    #[getter]
    fn ref_frame(&self) -> Option<String> { self.record.ref_frame.clone() }
    #[getter]
    fn time_system(&self) -> Option<String> { self.record.time_system.clone() }
    #[getter]
    fn mean_element_theory(&self) -> Option<String> { self.record.mean_element_theory.clone() }
    #[getter]
    fn epoch(&self) -> Option<String> { self.record.epoch.clone() }
    #[getter]
    fn mean_motion(&self) -> Option<f64> { self.record.mean_motion }
    #[getter]
    fn eccentricity(&self) -> Option<f64> { self.record.eccentricity }
    #[getter]
    fn inclination(&self) -> Option<f64> { self.record.inclination }
    #[getter]
    fn ra_of_asc_node(&self) -> Option<f64> { self.record.ra_of_asc_node }
    #[getter]
    fn arg_of_pericenter(&self) -> Option<f64> { self.record.arg_of_pericenter }
    #[getter]
    fn mean_anomaly(&self) -> Option<f64> { self.record.mean_anomaly }
    #[getter]
    fn ephemeris_type(&self) -> Option<i32> { self.record.ephemeris_type }
    #[getter]
    fn classification_type(&self) -> Option<String> { self.record.classification_type.clone() }
    #[getter]
    fn norad_cat_id(&self) -> Option<u32> { self.record.norad_cat_id }
    #[getter]
    fn element_set_no(&self) -> Option<u32> { self.record.element_set_no }
    #[getter]
    fn rev_at_epoch(&self) -> Option<u32> { self.record.rev_at_epoch }
    #[getter]
    fn bstar(&self) -> Option<f64> { self.record.bstar }
    #[getter]
    fn mean_motion_dot(&self) -> Option<f64> { self.record.mean_motion_dot }
    #[getter]
    fn mean_motion_ddot(&self) -> Option<f64> { self.record.mean_motion_ddot }
    #[getter]
    fn semimajor_axis(&self) -> Option<f64> { self.record.semimajor_axis }
    #[getter]
    fn period(&self) -> Option<f64> { self.record.period }
    #[getter]
    fn apoapsis(&self) -> Option<f64> { self.record.apoapsis }
    #[getter]
    fn periapsis(&self) -> Option<f64> { self.record.periapsis }
    #[getter]
    fn object_type(&self) -> Option<String> { self.record.object_type.clone() }
    #[getter]
    fn rcs_size(&self) -> Option<String> { self.record.rcs_size.clone() }
    #[getter]
    fn country_code(&self) -> Option<String> { self.record.country_code.clone() }
    #[getter]
    fn launch_date(&self) -> Option<String> { self.record.launch_date.clone() }
    #[getter]
    fn site(&self) -> Option<String> { self.record.site.clone() }
    #[getter]
    fn decay_date(&self) -> Option<String> { self.record.decay_date.clone() }
    #[getter]
    fn decayed(&self) -> Option<i32> { self.record.decayed }
    #[getter]
    fn file(&self) -> Option<u64> { self.record.file }
    #[getter]
    fn gp_id(&self) -> Option<u64> { self.record.gp_id }
    #[getter]
    fn tle_line0(&self) -> Option<String> { self.record.tle_line0.clone() }
    #[getter]
    fn tle_line1(&self) -> Option<String> { self.record.tle_line1.clone() }
    #[getter]
    fn tle_line2(&self) -> Option<String> { self.record.tle_line2.clone() }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "OMMRecord(norad_cat_id={:?}, object_name={:?}, epoch={:?})",
            self.record.norad_cat_id, self.record.object_name, self.record.epoch
        )
    }
}

impl From<spacetrack::OMMRecord> for PyOMMRecord {
    fn from(record: spacetrack::OMMRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// TLE Record - Two-Line Element
// ============================================================================

/// TLE record containing Two-Line Element data.
#[pyclass(name = "TLERecord", module = "brahe._brahe")]
pub struct PyTLERecord {
    pub(crate) record: spacetrack::TLERecord,
}

#[pymethods]
impl PyTLERecord {
    #[getter]
    fn comment(&self) -> Option<String> { self.record.comment.clone() }
    #[getter]
    fn originator(&self) -> Option<String> { self.record.originator.clone() }
    #[getter]
    fn norad_cat_id(&self) -> Option<u32> { self.record.norad_cat_id }
    #[getter]
    fn object_name(&self) -> Option<String> { self.record.object_name.clone() }
    #[getter]
    fn object_type(&self) -> Option<String> { self.record.object_type.clone() }
    #[getter]
    fn classification_type(&self) -> Option<String> { self.record.classification_type.clone() }
    #[getter]
    fn intldes(&self) -> Option<String> { self.record.intldes.clone() }
    #[getter]
    fn epoch(&self) -> Option<String> { self.record.epoch.clone() }
    #[getter]
    fn epoch_microseconds(&self) -> Option<f64> { self.record.epoch_microseconds }
    #[getter]
    fn mean_motion(&self) -> Option<f64> { self.record.mean_motion }
    #[getter]
    fn eccentricity(&self) -> Option<f64> { self.record.eccentricity }
    #[getter]
    fn inclination(&self) -> Option<f64> { self.record.inclination }
    #[getter]
    fn ra_of_asc_node(&self) -> Option<f64> { self.record.ra_of_asc_node }
    #[getter]
    fn arg_of_pericenter(&self) -> Option<f64> { self.record.arg_of_pericenter }
    #[getter]
    fn mean_anomaly(&self) -> Option<f64> { self.record.mean_anomaly }
    #[getter]
    fn ephemeris_type(&self) -> Option<i32> { self.record.ephemeris_type }
    #[getter]
    fn element_set_no(&self) -> Option<u32> { self.record.element_set_no }
    #[getter]
    fn rev_at_epoch(&self) -> Option<u32> { self.record.rev_at_epoch }
    #[getter]
    fn bstar(&self) -> Option<f64> { self.record.bstar }
    #[getter]
    fn mean_motion_dot(&self) -> Option<f64> { self.record.mean_motion_dot }
    #[getter]
    fn mean_motion_ddot(&self) -> Option<f64> { self.record.mean_motion_ddot }
    #[getter]
    fn file(&self) -> Option<u64> { self.record.file }
    #[getter]
    fn tle_line1(&self) -> Option<String> { self.record.tle_line1.clone() }
    #[getter]
    fn tle_line2(&self) -> Option<String> { self.record.tle_line2.clone() }
    #[getter]
    fn tle_line0(&self) -> Option<String> { self.record.tle_line0.clone() }
    #[getter]
    fn object_id(&self) -> Option<String> { self.record.object_id.clone() }
    #[getter]
    fn object_number(&self) -> Option<u32> { self.record.object_number }
    #[getter]
    fn semimajor_axis(&self) -> Option<f64> { self.record.semimajor_axis }
    #[getter]
    fn period(&self) -> Option<f64> { self.record.period }
    #[getter]
    fn apoapsis(&self) -> Option<f64> { self.record.apoapsis }
    #[getter]
    fn periapsis(&self) -> Option<f64> { self.record.periapsis }
    #[getter]
    fn data_status_code(&self) -> Option<String> { self.record.data_status_code.clone() }
    #[getter]
    fn ordinal(&self) -> Option<u32> { self.record.ordinal }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "TLERecord(norad_cat_id={:?}, object_name={:?}, epoch={:?})",
            self.record.norad_cat_id, self.record.object_name, self.record.epoch
        )
    }
}

impl From<spacetrack::TLERecord> for PyTLERecord {
    fn from(record: spacetrack::TLERecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// SATCAT Record - Satellite Catalog
// ============================================================================

/// SATCAT record containing satellite catalog information.
#[pyclass(name = "SATCATRecord", module = "brahe._brahe")]
pub struct PySATCATRecord {
    pub(crate) record: spacetrack::SATCATRecord,
}

#[pymethods]
impl PySATCATRecord {
    #[getter]
    fn intldes(&self) -> Option<String> { self.record.intldes.clone() }
    #[getter]
    fn norad_cat_id(&self) -> Option<u32> { self.record.norad_cat_id }
    #[getter]
    fn object_type(&self) -> Option<String> { self.record.object_type.clone() }
    #[getter]
    fn satname(&self) -> Option<String> { self.record.satname.clone() }
    #[getter]
    fn country(&self) -> Option<String> { self.record.country.clone() }
    #[getter]
    fn launch(&self) -> Option<String> { self.record.launch.clone() }
    #[getter]
    fn site(&self) -> Option<String> { self.record.site.clone() }
    #[getter]
    fn decay(&self) -> Option<String> { self.record.decay.clone() }
    #[getter]
    fn period(&self) -> Option<f64> { self.record.period }
    #[getter]
    fn inclination(&self) -> Option<f64> { self.record.inclination }
    #[getter]
    fn apogee(&self) -> Option<u32> { self.record.apogee }
    #[getter]
    fn perigee(&self) -> Option<u32> { self.record.perigee }
    #[getter]
    fn comment(&self) -> Option<String> { self.record.comment.clone() }
    #[getter]
    fn commentcode(&self) -> Option<u32> { self.record.commentcode }
    #[getter]
    fn rcsvalue(&self) -> Option<i32> { self.record.rcsvalue }
    #[getter]
    fn rcs_size(&self) -> Option<String> { self.record.rcs_size.clone() }
    #[getter]
    fn file(&self) -> Option<u64> { self.record.file }
    #[getter]
    fn launch_year(&self) -> Option<u32> { self.record.launch_year }
    #[getter]
    fn launch_num(&self) -> Option<u32> { self.record.launch_num }
    #[getter]
    fn launch_piece(&self) -> Option<String> { self.record.launch_piece.clone() }
    #[getter]
    fn current(&self) -> Option<String> { self.record.current.clone() }
    #[getter]
    fn object_name(&self) -> Option<String> { self.record.object_name.clone() }
    #[getter]
    fn object_id(&self) -> Option<String> { self.record.object_id.clone() }
    #[getter]
    fn object_number(&self) -> Option<u32> { self.record.object_number }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "SATCATRecord(norad_cat_id={:?}, satname={:?})",
            self.record.norad_cat_id, self.record.satname
        )
    }
}

impl From<spacetrack::SATCATRecord> for PySATCATRecord {
    fn from(record: spacetrack::SATCATRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// SATCAT Change Record
// ============================================================================

/// SATCAT Change record containing catalog change information.
#[pyclass(name = "SATCATChangeRecord", module = "brahe._brahe")]
pub struct PySATCATChangeRecord {
    pub(crate) record: spacetrack::SATCATChangeRecord,
}

#[pymethods]
impl PySATCATChangeRecord {
    #[getter]
    fn norad_cat_id(&self) -> Option<u32> { self.record.norad_cat_id }
    #[getter]
    fn object_number(&self) -> Option<u32> { self.record.object_number }
    #[getter]
    fn current_name(&self) -> Option<String> { self.record.current_name.clone() }
    #[getter]
    fn previous_name(&self) -> Option<String> { self.record.previous_name.clone() }
    #[getter]
    fn current_intldes(&self) -> Option<String> { self.record.current_intldes.clone() }
    #[getter]
    fn previous_intldes(&self) -> Option<String> { self.record.previous_intldes.clone() }
    #[getter]
    fn current_country(&self) -> Option<String> { self.record.current_country.clone() }
    #[getter]
    fn previous_country(&self) -> Option<String> { self.record.previous_country.clone() }
    #[getter]
    fn current_launch(&self) -> Option<String> { self.record.current_launch.clone() }
    #[getter]
    fn previous_launch(&self) -> Option<String> { self.record.previous_launch.clone() }
    #[getter]
    fn current_decay(&self) -> Option<String> { self.record.current_decay.clone() }
    #[getter]
    fn previous_decay(&self) -> Option<String> { self.record.previous_decay.clone() }
    #[getter]
    fn change_made(&self) -> Option<String> { self.record.change_made.clone() }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "SATCATChangeRecord(norad_cat_id={:?}, change_made={:?})",
            self.record.norad_cat_id, self.record.change_made
        )
    }
}

impl From<spacetrack::SATCATChangeRecord> for PySATCATChangeRecord {
    fn from(record: spacetrack::SATCATChangeRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// SATCAT Debut Record
// ============================================================================

/// SATCAT Debut record containing new catalog entry information.
#[pyclass(name = "SATCATDebutRecord", module = "brahe._brahe")]
pub struct PySATCATDebutRecord {
    pub(crate) record: spacetrack::SATCATDebutRecord,
}

#[pymethods]
impl PySATCATDebutRecord {
    #[getter]
    fn intldes(&self) -> Option<String> { self.record.intldes.clone() }
    #[getter]
    fn norad_cat_id(&self) -> Option<u32> { self.record.norad_cat_id }
    #[getter]
    fn object_type(&self) -> Option<String> { self.record.object_type.clone() }
    #[getter]
    fn satname(&self) -> Option<String> { self.record.satname.clone() }
    #[getter]
    fn debut(&self) -> Option<String> { self.record.debut.clone() }
    #[getter]
    fn country(&self) -> Option<String> { self.record.country.clone() }
    #[getter]
    fn launch(&self) -> Option<String> { self.record.launch.clone() }
    #[getter]
    fn site(&self) -> Option<String> { self.record.site.clone() }
    #[getter]
    fn decay(&self) -> Option<String> { self.record.decay.clone() }
    #[getter]
    fn period(&self) -> Option<f64> { self.record.period }
    #[getter]
    fn inclination(&self) -> Option<f64> { self.record.inclination }
    #[getter]
    fn apogee(&self) -> Option<u32> { self.record.apogee }
    #[getter]
    fn perigee(&self) -> Option<u32> { self.record.perigee }
    #[getter]
    fn rcs_size(&self) -> Option<String> { self.record.rcs_size.clone() }
    #[getter]
    fn current(&self) -> Option<String> { self.record.current.clone() }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "SATCATDebutRecord(norad_cat_id={:?}, satname={:?}, debut={:?})",
            self.record.norad_cat_id, self.record.satname, self.record.debut
        )
    }
}

impl From<spacetrack::SATCATDebutRecord> for PySATCATDebutRecord {
    fn from(record: spacetrack::SATCATDebutRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// Decay Record
// ============================================================================

/// Decay record containing re-entry/decay information.
#[pyclass(name = "DecayRecord", module = "brahe._brahe")]
pub struct PyDecayRecord {
    pub(crate) record: spacetrack::DecayRecord,
}

#[pymethods]
impl PyDecayRecord {
    #[getter]
    fn norad_cat_id(&self) -> Option<u32> { self.record.norad_cat_id }
    #[getter]
    fn object_number(&self) -> Option<u32> { self.record.object_number }
    #[getter]
    fn object_name(&self) -> Option<String> { self.record.object_name.clone() }
    #[getter]
    fn intldes(&self) -> Option<String> { self.record.intldes.clone() }
    #[getter]
    fn object_id(&self) -> Option<String> { self.record.object_id.clone() }
    #[getter]
    fn rcs(&self) -> Option<String> { self.record.rcs.clone() }
    #[getter]
    fn rcs_size(&self) -> Option<String> { self.record.rcs_size.clone() }
    #[getter]
    fn country(&self) -> Option<String> { self.record.country.clone() }
    #[getter]
    fn msg_epoch(&self) -> Option<String> { self.record.msg_epoch.clone() }
    #[getter]
    fn decay_epoch(&self) -> Option<String> { self.record.decay_epoch.clone() }
    #[getter]
    fn source(&self) -> Option<String> { self.record.source.clone() }
    #[getter]
    fn msg_type(&self) -> Option<String> { self.record.msg_type.clone() }
    #[getter]
    fn precedence(&self) -> Option<u32> { self.record.precedence }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "DecayRecord(norad_cat_id={:?}, object_name={:?}, decay_epoch={:?})",
            self.record.norad_cat_id, self.record.object_name, self.record.decay_epoch
        )
    }
}

impl From<spacetrack::DecayRecord> for PyDecayRecord {
    fn from(record: spacetrack::DecayRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// TIP Record - Tracking and Impact Prediction
// ============================================================================

/// TIP record containing tracking and impact prediction data.
#[pyclass(name = "TIPRecord", module = "brahe._brahe")]
pub struct PyTIPRecord {
    pub(crate) record: spacetrack::TIPRecord,
}

#[pymethods]
impl PyTIPRecord {
    #[getter]
    fn norad_cat_id(&self) -> Option<u32> { self.record.norad_cat_id }
    #[getter]
    fn msg_epoch(&self) -> Option<String> { self.record.msg_epoch.clone() }
    #[getter]
    fn insert_epoch(&self) -> Option<String> { self.record.insert_epoch.clone() }
    #[getter]
    fn decay_epoch(&self) -> Option<String> { self.record.decay_epoch.clone() }
    #[getter]
    fn window(&self) -> Option<f64> { self.record.window }
    #[getter]
    fn rev(&self) -> Option<u32> { self.record.rev }
    #[getter]
    fn direction(&self) -> Option<String> { self.record.direction.clone() }
    #[getter]
    fn lat(&self) -> Option<f64> { self.record.lat }
    #[getter]
    fn lon(&self) -> Option<f64> { self.record.lon }
    #[getter]
    fn incl(&self) -> Option<f64> { self.record.incl }
    #[getter]
    fn next_report(&self) -> Option<String> { self.record.next_report.clone() }
    #[getter]
    fn id(&self) -> Option<String> { self.record.id.clone() }
    #[getter]
    fn high_interest(&self) -> Option<String> { self.record.high_interest.clone() }
    #[getter]
    fn object_number(&self) -> Option<u32> { self.record.object_number }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "TIPRecord(norad_cat_id={:?}, decay_epoch={:?})",
            self.record.norad_cat_id, self.record.decay_epoch
        )
    }
}

impl From<spacetrack::TIPRecord> for PyTIPRecord {
    fn from(record: spacetrack::TIPRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// CDM Public Record - Public Conjunction Data Message
// ============================================================================

/// CDM Public record containing public conjunction data.
#[pyclass(name = "CDMPublicRecord", module = "brahe._brahe")]
pub struct PyCDMPublicRecord {
    pub(crate) record: spacetrack::CDMPublicRecord,
}

#[pymethods]
impl PyCDMPublicRecord {
    #[getter]
    fn cdm_id(&self) -> Option<u64> { self.record.cdm_id }
    #[getter]
    fn creation_date(&self) -> Option<String> { self.record.creation_date.clone() }
    #[getter]
    fn tca(&self) -> Option<String> { self.record.tca.clone() }
    #[getter]
    fn miss_distance(&self) -> Option<f64> { self.record.miss_distance }
    #[getter]
    fn relative_speed(&self) -> Option<f64> { self.record.relative_speed }
    #[getter]
    fn collision_probability(&self) -> Option<f64> { self.record.collision_probability }
    #[getter]
    fn sat1_object_designator(&self) -> Option<String> { self.record.sat1_object_designator.clone() }
    #[getter]
    fn sat1_norad_cat_id(&self) -> Option<u32> { self.record.sat1_norad_cat_id }
    #[getter]
    fn sat1_object_name(&self) -> Option<String> { self.record.sat1_object_name.clone() }
    #[getter]
    fn sat1_object_type(&self) -> Option<String> { self.record.sat1_object_type.clone() }
    #[getter]
    fn sat1_rcs(&self) -> Option<String> { self.record.sat1_rcs.clone() }
    #[getter]
    fn sat1_excl_vol(&self) -> Option<String> { self.record.sat1_excl_vol.clone() }
    #[getter]
    fn sat2_object_designator(&self) -> Option<String> { self.record.sat2_object_designator.clone() }
    #[getter]
    fn sat2_norad_cat_id(&self) -> Option<u32> { self.record.sat2_norad_cat_id }
    #[getter]
    fn sat2_object_name(&self) -> Option<String> { self.record.sat2_object_name.clone() }
    #[getter]
    fn sat2_object_type(&self) -> Option<String> { self.record.sat2_object_type.clone() }
    #[getter]
    fn sat2_rcs(&self) -> Option<String> { self.record.sat2_rcs.clone() }
    #[getter]
    fn sat2_excl_vol(&self) -> Option<String> { self.record.sat2_excl_vol.clone() }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "CDMPublicRecord(cdm_id={:?}, tca={:?}, miss_distance={:?})",
            self.record.cdm_id, self.record.tca, self.record.miss_distance
        )
    }
}

impl From<spacetrack::CDMPublicRecord> for PyCDMPublicRecord {
    fn from(record: spacetrack::CDMPublicRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// Boxscore Record
// ============================================================================

/// Boxscore record containing catalog statistics.
#[pyclass(name = "BoxscoreRecord", module = "brahe._brahe")]
pub struct PyBoxscoreRecord {
    pub(crate) record: spacetrack::BoxscoreRecord,
}

#[pymethods]
impl PyBoxscoreRecord {
    #[getter]
    fn country(&self) -> Option<String> { self.record.country.clone() }
    #[getter]
    fn spadoc_cd(&self) -> Option<String> { self.record.spadoc_cd.clone() }
    #[getter]
    fn orbital_tba(&self) -> Option<u32> { self.record.orbital_tba }
    #[getter]
    fn orbital_payload_count(&self) -> Option<u32> { self.record.orbital_payload_count }
    #[getter]
    fn orbital_rocket_body_count(&self) -> Option<u32> { self.record.orbital_rocket_body_count }
    #[getter]
    fn orbital_debris_count(&self) -> Option<u32> { self.record.orbital_debris_count }
    #[getter]
    fn orbital_total_count(&self) -> Option<u32> { self.record.orbital_total_count }
    #[getter]
    fn decayed_payload_count(&self) -> Option<u32> { self.record.decayed_payload_count }
    #[getter]
    fn decayed_rocket_body_count(&self) -> Option<u32> { self.record.decayed_rocket_body_count }
    #[getter]
    fn decayed_debris_count(&self) -> Option<u32> { self.record.decayed_debris_count }
    #[getter]
    fn decayed_total_count(&self) -> Option<u32> { self.record.decayed_total_count }
    #[getter]
    fn country_total(&self) -> Option<u32> { self.record.country_total }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "BoxscoreRecord(country={:?}, orbital_total_count={:?})",
            self.record.country, self.record.orbital_total_count
        )
    }
}

impl From<spacetrack::BoxscoreRecord> for PyBoxscoreRecord {
    fn from(record: spacetrack::BoxscoreRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// Launch Site Record
// ============================================================================

/// Launch Site record containing launch facility information.
#[pyclass(name = "LaunchSiteRecord", module = "brahe._brahe")]
pub struct PyLaunchSiteRecord {
    pub(crate) record: spacetrack::LaunchSiteRecord,
}

#[pymethods]
impl PyLaunchSiteRecord {
    #[getter]
    fn site_code(&self) -> Option<String> { self.record.site_code.clone() }
    #[getter]
    fn launch_site(&self) -> Option<String> { self.record.launch_site.clone() }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "LaunchSiteRecord(site_code={:?}, launch_site={:?})",
            self.record.site_code, self.record.launch_site
        )
    }
}

impl From<spacetrack::LaunchSiteRecord> for PyLaunchSiteRecord {
    fn from(record: spacetrack::LaunchSiteRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// Announcement Record
// ============================================================================

/// Announcement record containing Space-Track announcement data.
#[pyclass(name = "AnnouncementRecord", module = "brahe._brahe")]
pub struct PyAnnouncementRecord {
    pub(crate) record: spacetrack::AnnouncementRecord,
}

#[pymethods]
impl PyAnnouncementRecord {
    #[getter]
    fn announcement_id(&self) -> Option<u64> { self.record.announcement_id }
    #[getter]
    fn announcement_type(&self) -> Option<String> { self.record.announcement_type.clone() }
    #[getter]
    fn announcement_text(&self) -> Option<String> { self.record.announcement_text.clone() }
    #[getter]
    fn announcement_start(&self) -> Option<String> { self.record.announcement_start.clone() }
    #[getter]
    fn announcement_end(&self) -> Option<String> { self.record.announcement_end.clone() }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "AnnouncementRecord(announcement_id={:?}, announcement_type={:?})",
            self.record.announcement_id, self.record.announcement_type
        )
    }
}

impl From<spacetrack::AnnouncementRecord> for PyAnnouncementRecord {
    fn from(record: spacetrack::AnnouncementRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// CDM Record - Full Conjunction Data Message (Expanded Space Data)
// ============================================================================

/// CDM record containing full conjunction data (requires expanded access).
#[pyclass(name = "CDMRecord", module = "brahe._brahe")]
pub struct PyCDMRecord {
    pub(crate) record: spacetrack::CDMRecord,
}

#[pymethods]
impl PyCDMRecord {
    #[getter]
    fn cdm_id(&self) -> Option<u64> { self.record.cdm_id }
    #[getter]
    fn constellation(&self) -> Option<String> { self.record.constellation.clone() }
    #[getter]
    fn message_id(&self) -> Option<String> { self.record.message_id.clone() }
    #[getter]
    fn message_for(&self) -> Option<String> { self.record.message_for.clone() }
    #[getter]
    fn creation_date(&self) -> Option<String> { self.record.creation_date.clone() }
    #[getter]
    fn emergency_reportable(&self) -> Option<String> { self.record.emergency_reportable.clone() }
    #[getter]
    fn tca(&self) -> Option<String> { self.record.tca.clone() }
    #[getter]
    fn miss_distance(&self) -> Option<f64> { self.record.miss_distance }
    #[getter]
    fn relative_speed(&self) -> Option<f64> { self.record.relative_speed }
    #[getter]
    fn relative_position_r(&self) -> Option<f64> { self.record.relative_position_r }
    #[getter]
    fn relative_position_t(&self) -> Option<f64> { self.record.relative_position_t }
    #[getter]
    fn relative_position_n(&self) -> Option<f64> { self.record.relative_position_n }
    #[getter]
    fn relative_velocity_r(&self) -> Option<f64> { self.record.relative_velocity_r }
    #[getter]
    fn relative_velocity_t(&self) -> Option<f64> { self.record.relative_velocity_t }
    #[getter]
    fn relative_velocity_n(&self) -> Option<f64> { self.record.relative_velocity_n }
    #[getter]
    fn collision_probability_method(&self) -> Option<String> { self.record.collision_probability_method.clone() }
    #[getter]
    fn collision_probability(&self) -> Option<f64> { self.record.collision_probability }
    #[getter]
    fn sat1_id(&self) -> Option<u32> { self.record.sat1_id }
    #[getter]
    fn sat1_name(&self) -> Option<String> { self.record.sat1_name.clone() }
    #[getter]
    fn sat1_norad_cat_id(&self) -> Option<u32> { self.record.sat1_norad_cat_id }
    #[getter]
    fn sat1_object_designator(&self) -> Option<String> { self.record.sat1_object_designator.clone() }
    #[getter]
    fn sat1_object_type(&self) -> Option<String> { self.record.sat1_object_type.clone() }
    #[getter]
    fn sat1_operator_organization(&self) -> Option<String> { self.record.sat1_operator_organization.clone() }
    #[getter]
    fn sat2_id(&self) -> Option<u32> { self.record.sat2_id }
    #[getter]
    fn sat2_name(&self) -> Option<String> { self.record.sat2_name.clone() }
    #[getter]
    fn sat2_norad_cat_id(&self) -> Option<u32> { self.record.sat2_norad_cat_id }
    #[getter]
    fn sat2_object_designator(&self) -> Option<String> { self.record.sat2_object_designator.clone() }
    #[getter]
    fn sat2_object_type(&self) -> Option<String> { self.record.sat2_object_type.clone() }
    #[getter]
    fn sat2_operator_organization(&self) -> Option<String> { self.record.sat2_operator_organization.clone() }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "CDMRecord(cdm_id={:?}, tca={:?}, miss_distance={:?})",
            self.record.cdm_id, self.record.tca, self.record.miss_distance
        )
    }
}

impl From<spacetrack::CDMRecord> for PyCDMRecord {
    fn from(record: spacetrack::CDMRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// CAR Record - Conjunction Assessment Report (Expanded Space Data)
// ============================================================================

/// CAR record containing conjunction assessment data (requires expanded access).
#[pyclass(name = "CARRecord", module = "brahe._brahe")]
pub struct PyCARRecord {
    pub(crate) record: spacetrack::CARRecord,
}

#[pymethods]
impl PyCARRecord {
    #[getter]
    fn message_id(&self) -> Option<u64> { self.record.message_id }
    #[getter]
    fn message_epoch(&self) -> Option<String> { self.record.message_epoch.clone() }
    #[getter]
    fn collision_probability(&self) -> Option<f64> { self.record.collision_probability }
    #[getter]
    fn miss_distance(&self) -> Option<f64> { self.record.miss_distance }
    #[getter]
    fn tca(&self) -> Option<String> { self.record.tca.clone() }
    #[getter]
    fn sat1_norad_cat_id(&self) -> Option<u32> { self.record.sat1_norad_cat_id }
    #[getter]
    fn sat1_name(&self) -> Option<String> { self.record.sat1_name.clone() }
    #[getter]
    fn sat2_norad_cat_id(&self) -> Option<u32> { self.record.sat2_norad_cat_id }
    #[getter]
    fn sat2_name(&self) -> Option<String> { self.record.sat2_name.clone() }
    #[getter]
    fn emergency_reportable(&self) -> Option<String> { self.record.emergency_reportable.clone() }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "CARRecord(message_id={:?}, tca={:?}, miss_distance={:?})",
            self.record.message_id, self.record.tca, self.record.miss_distance
        )
    }
}

impl From<spacetrack::CARRecord> for PyCARRecord {
    fn from(record: spacetrack::CARRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// Maneuver Record (Expanded Space Data)
// ============================================================================

/// Maneuver record containing satellite maneuver data (requires expanded access).
#[pyclass(name = "ManeuverRecord", module = "brahe._brahe")]
pub struct PyManeuverRecord {
    pub(crate) record: spacetrack::ManeuverRecord,
}

#[pymethods]
impl PyManeuverRecord {
    #[getter]
    fn maneuver_id(&self) -> Option<u64> { self.record.maneuver_id }
    #[getter]
    fn norad_cat_id(&self) -> Option<u32> { self.record.norad_cat_id }
    #[getter]
    fn satname(&self) -> Option<String> { self.record.satname.clone() }
    #[getter]
    fn source(&self) -> Option<String> { self.record.source.clone() }
    #[getter]
    fn maneuverable(&self) -> Option<String> { self.record.maneuverable.clone() }
    #[getter]
    fn start_time(&self) -> Option<String> { self.record.start_time.clone() }
    #[getter]
    fn stop_time(&self) -> Option<String> { self.record.stop_time.clone() }
    #[getter]
    fn data_status(&self) -> Option<String> { self.record.data_status.clone() }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "ManeuverRecord(maneuver_id={:?}, norad_cat_id={:?}, satname={:?})",
            self.record.maneuver_id, self.record.norad_cat_id, self.record.satname
        )
    }
}

impl From<spacetrack::ManeuverRecord> for PyManeuverRecord {
    fn from(record: spacetrack::ManeuverRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// Maneuver History Record (Expanded Space Data)
// ============================================================================

/// Maneuver History record containing historical maneuver data.
#[pyclass(name = "ManeuverHistoryRecord", module = "brahe._brahe")]
pub struct PyManeuverHistoryRecord {
    pub(crate) record: spacetrack::ManeuverHistoryRecord,
}

#[pymethods]
impl PyManeuverHistoryRecord {
    #[getter]
    fn maneuver_history_id(&self) -> Option<u64> { self.record.maneuver_history_id }
    #[getter]
    fn norad_cat_id(&self) -> Option<u32> { self.record.norad_cat_id }
    #[getter]
    fn satname(&self) -> Option<String> { self.record.satname.clone() }
    #[getter]
    fn object_type(&self) -> Option<String> { self.record.object_type.clone() }
    #[getter]
    fn source(&self) -> Option<String> { self.record.source.clone() }
    #[getter]
    fn start_time(&self) -> Option<String> { self.record.start_time.clone() }
    #[getter]
    fn stop_time(&self) -> Option<String> { self.record.stop_time.clone() }
    #[getter]
    fn delta_v(&self) -> Option<f64> { self.record.delta_v }
    #[getter]
    fn thrust_uncertainty(&self) -> Option<f64> { self.record.thrust_uncertainty }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "ManeuverHistoryRecord(maneuver_history_id={:?}, norad_cat_id={:?})",
            self.record.maneuver_history_id, self.record.norad_cat_id
        )
    }
}

impl From<spacetrack::ManeuverHistoryRecord> for PyManeuverHistoryRecord {
    fn from(record: spacetrack::ManeuverHistoryRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// Organization Record (Expanded Space Data)
// ============================================================================

/// Organization record containing operator information (requires expanded access).
#[pyclass(name = "OrganizationRecord", module = "brahe._brahe")]
pub struct PyOrganizationRecord {
    pub(crate) record: spacetrack::OrganizationRecord,
}

#[pymethods]
impl PyOrganizationRecord {
    #[getter]
    fn organization_id(&self) -> Option<u64> { self.record.organization_id }
    #[getter]
    fn organization_name(&self) -> Option<String> { self.record.organization_name.clone() }
    #[getter]
    fn organization_type(&self) -> Option<String> { self.record.organization_type.clone() }
    #[getter]
    fn country(&self) -> Option<String> { self.record.country.clone() }
    #[getter]
    fn parent_id(&self) -> Option<u64> { self.record.parent_id }
    #[getter]
    fn primary_phone(&self) -> Option<String> { self.record.primary_phone.clone() }
    #[getter]
    fn primary_email(&self) -> Option<String> { self.record.primary_email.clone() }
    #[getter]
    fn info_link(&self) -> Option<String> { self.record.info_link.clone() }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "OrganizationRecord(organization_id={:?}, organization_name={:?})",
            self.record.organization_id, self.record.organization_name
        )
    }
}

impl From<spacetrack::OrganizationRecord> for PyOrganizationRecord {
    fn from(record: spacetrack::OrganizationRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// Satellite Record (Expanded Space Data)
// ============================================================================

/// Satellite record containing detailed satellite information (requires expanded access).
#[pyclass(name = "SatelliteRecord", module = "brahe._brahe")]
pub struct PySatelliteRecord {
    pub(crate) record: spacetrack::SatelliteRecord,
}

#[pymethods]
impl PySatelliteRecord {
    #[getter]
    fn norad_cat_id(&self) -> Option<u32> { self.record.norad_cat_id }
    #[getter]
    fn satid(&self) -> Option<u32> { self.record.satid }
    #[getter]
    fn satname(&self) -> Option<String> { self.record.satname.clone() }
    #[getter]
    fn intldes(&self) -> Option<String> { self.record.intldes.clone() }
    #[getter]
    fn status(&self) -> Option<String> { self.record.status.clone() }
    #[getter]
    fn active(&self) -> Option<String> { self.record.active.clone() }
    #[getter]
    fn country(&self) -> Option<String> { self.record.country.clone() }
    #[getter]
    fn launch_date(&self) -> Option<String> { self.record.launch_date.clone() }
    #[getter]
    fn launch_site(&self) -> Option<String> { self.record.launch_site.clone() }
    #[getter]
    fn launch_vehicle(&self) -> Option<String> { self.record.launch_vehicle.clone() }
    #[getter]
    fn launch_mass(&self) -> Option<f64> { self.record.launch_mass }
    #[getter]
    fn dry_mass(&self) -> Option<f64> { self.record.dry_mass }
    #[getter]
    fn power(&self) -> Option<f64> { self.record.power }
    #[getter]
    fn expected_lifetime(&self) -> Option<f64> { self.record.expected_lifetime }
    #[getter]
    fn mission_type(&self) -> Option<String> { self.record.mission_type.clone() }
    #[getter]
    fn mission(&self) -> Option<String> { self.record.mission.clone() }
    #[getter]
    fn contractor(&self) -> Option<String> { self.record.contractor.clone() }
    #[getter]
    fn operator(&self) -> Option<String> { self.record.operator.clone() }
    #[getter]
    fn users(&self) -> Option<String> { self.record.users.clone() }

    fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        record_to_py_dict(py, &self.record)
    }

    fn __repr__(&self) -> String {
        format!(
            "SatelliteRecord(norad_cat_id={:?}, satname={:?})",
            self.record.norad_cat_id, self.record.satname
        )
    }
}

impl From<spacetrack::SatelliteRecord> for PySatelliteRecord {
    fn from(record: spacetrack::SatelliteRecord) -> Self {
        Self { record }
    }
}

// ============================================================================
// SpaceTrack Client
// ============================================================================

/// Blocking SpaceTrack API client for Python.
///
/// This client provides synchronous access to the Space-Track.org API
/// for querying orbital element data, satellite catalog information,
/// and other space surveillance data.
///
/// Args:
///     identity (str): Space-Track.org username
///     password (str): Space-Track.org password
///     base_url (str, optional): Custom base URL (default: https://www.space-track.org/)
///
/// Raises:
///     RuntimeError: If authentication fails
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create client and authenticate
///     client = bh.SpaceTrackClient("username", "password")
///
///     # Query GP data for ISS
///     records = client.gp(norad_cat_id=25544, limit=1)
///     print(f"ISS epoch: {records[0]['EPOCH']}")
///
///     # Query satellite catalog
///     sats = client.satcat(country="US", object_type="PAYLOAD", limit=10)
///     for sat in sats:
///         print(f"{sat['SATNAME']}: NORAD ID {sat['NORAD_CAT_ID']}")
///     ```
///
/// Note:
///     Requires a Space-Track.org account. Register at:
///     https://www.space-track.org/auth/createAccount
#[pyclass(name = "SpaceTrackClient", module = "brahe._brahe")]
pub struct PySpaceTrackClient {
    client: spacetrack::BlockingSpaceTrackClient,
}

#[pymethods]
impl PySpaceTrackClient {
    /// Create a new SpaceTrack client and authenticate.
    ///
    /// Args:
    ///     identity (str): Space-Track.org username
    ///     password (str): Space-Track.org password
    ///     base_url (str, optional): Custom base URL
    ///
    /// Returns:
    ///     SpaceTrackClient: Authenticated client instance
    ///
    /// Raises:
    ///     RuntimeError: If authentication fails
    #[new]
    #[pyo3(signature = (identity, password, base_url=None))]
    fn new(identity: &str, password: &str, base_url: Option<&str>) -> PyResult<Self> {
        let client = match base_url {
            Some(url) => spacetrack::BlockingSpaceTrackClient::with_base_url(identity, password, url),
            None => spacetrack::BlockingSpaceTrackClient::new(identity, password),
        }
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self { client })
    }

    /// Check if the client is authenticated.
    ///
    /// Returns:
    ///     bool: True if authenticated
    #[getter]
    fn is_authenticated(&self) -> bool {
        self.client.is_authenticated()
    }

    /// Get the base URL.
    ///
    /// Returns:
    ///     str: Base URL for API requests
    #[getter]
    fn base_url(&self) -> &str {
        self.client.base_url()
    }

    /// Query GP (General Perturbations) data.
    ///
    /// Returns the latest GP element sets for cataloged objects.
    ///
    /// Args:
    ///     norad_cat_id (int, optional): Filter by NORAD catalog ID
    ///     object_name (str, optional): Filter by object name
    ///     object_id (str, optional): Filter by object ID
    ///     epoch (str, optional): Filter by epoch (supports operators like ">2024-01-01")
    ///     object_type (str, optional): Filter by type (PAYLOAD, ROCKET BODY, DEBRIS)
    ///     country_code (str, optional): Filter by country code
    ///     limit (int, optional): Maximum number of results
    ///     orderby (str, optional): Field to order by (use "field asc" or "field desc")
    ///
    /// Returns:
    ///     list[GPRecord]: List of GP records
    ///
    /// Raises:
    ///     RuntimeError: If query fails
    ///
    /// Example:
    ///     ```python
    ///     # Get ISS GP data
    ///     records = client.gp(norad_cat_id=25544, limit=1)
    ///     print(records[0].object_name)  # Access via property
    ///     print(records[0].as_dict())    # Or as dictionary
    ///
    ///     # Get recent debris
    ///     debris = client.gp(object_type="DEBRIS", epoch=">2024-01-01", limit=100)
    ///     ```
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        norad_cat_id=None,
        object_name=None,
        object_id=None,
        epoch=None,
        object_type=None,
        country_code=None,
        limit=None,
        orderby=None
    ))]
    fn gp(
        &self,
        norad_cat_id: Option<u32>,
        object_name: Option<&str>,
        object_id: Option<&str>,
        epoch: Option<&str>,
        object_type: Option<&str>,
        country_code: Option<&str>,
        limit: Option<u32>,
        orderby: Option<&str>,
    ) -> PyResult<Vec<PyGPRecord>> {
        let mut req = spacetrack::request_classes::GPRequest::new();

        if let Some(v) = norad_cat_id {
            req = req.norad_cat_id(v);
        }
        if let Some(v) = object_name {
            req = req.object_name(parse_query_value(v));
        }
        if let Some(v) = object_id {
            req = req.object_id(parse_query_value(v));
        }
        if let Some(v) = epoch {
            req = req.epoch(parse_query_value(v));
        }
        if let Some(v) = object_type {
            req = req.object_type(parse_query_value(v));
        }
        if let Some(v) = country_code {
            req = req.country_code(parse_query_value(v));
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }
        if let Some(v) = orderby {
            req = parse_orderby(req, v);
        }

        let records = self
            .client
            .query(&req)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(records.into_iter().map(PyGPRecord::from).collect())
    }

    /// Query GP data and return as SGP propagators.
    ///
    /// Args:
    ///     step_size (float): Propagator step size in seconds
    ///     norad_cat_id (int, optional): Filter by NORAD catalog ID
    ///     limit (int, optional): Maximum number of results
    ///
    /// Returns:
    ///     list[SGPPropagator]: List of SGP propagators
    ///
    /// Example:
    ///     ```python
    ///     propagators = client.gp_as_propagators(60.0, norad_cat_id=25544, limit=1)
    ///     state = propagators[0].propagate(epoch)
    ///     ```
    #[pyo3(signature = (step_size, norad_cat_id=None, limit=None))]
    fn gp_as_propagators(
        &self,
        step_size: f64,
        norad_cat_id: Option<u32>,
        limit: Option<u32>,
    ) -> PyResult<Vec<PySGPPropagator>> {
        let mut req = spacetrack::request_classes::GPRequest::new();

        if let Some(v) = norad_cat_id {
            req = req.norad_cat_id(v);
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }

        let propagators = self
            .client
            .fetch_propagators(&req, step_size)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(propagators
            .into_iter()
            .map(|p| PySGPPropagator { propagator: p })
            .collect())
    }

    /// Query SATCAT (Satellite Catalog) data.
    ///
    /// Returns satellite catalog entries.
    ///
    /// Args:
    ///     norad_cat_id (int, optional): Filter by NORAD catalog ID
    ///     satname (str, optional): Filter by satellite name
    ///     intldes (str, optional): Filter by international designator
    ///     object_type (str, optional): Filter by type
    ///     country (str, optional): Filter by country
    ///     launch (str, optional): Filter by launch date
    ///     current (str, optional): Filter by current status (Y/N)
    ///     limit (int, optional): Maximum results
    ///     orderby (str, optional): Field to order by
    ///
    /// Returns:
    ///     list[SATCATRecord]: List of SATCAT records
    ///
    /// Example:
    ///     ```python
    ///     # Get US payloads
    ///     sats = client.satcat(country="US", object_type="PAYLOAD", limit=100)
    ///     print(sats[0].satname)  # Access via property
    ///     ```
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        norad_cat_id=None,
        satname=None,
        intldes=None,
        object_type=None,
        country=None,
        launch=None,
        current=None,
        limit=None,
        orderby=None
    ))]
    fn satcat(
        &self,
        norad_cat_id: Option<u32>,
        satname: Option<&str>,
        intldes: Option<&str>,
        object_type: Option<&str>,
        country: Option<&str>,
        launch: Option<&str>,
        current: Option<&str>,
        limit: Option<u32>,
        orderby: Option<&str>,
    ) -> PyResult<Vec<PySATCATRecord>> {
        let mut req = spacetrack::request_classes::SATCATRequest::new();

        if let Some(v) = norad_cat_id {
            req = req.norad_cat_id(v);
        }
        if let Some(v) = satname {
            req = req.satname(parse_query_value(v));
        }
        if let Some(v) = intldes {
            req = req.intldes(parse_query_value(v));
        }
        if let Some(v) = object_type {
            req = req.object_type(parse_query_value(v));
        }
        if let Some(v) = country {
            req = req.country(parse_query_value(v));
        }
        if let Some(v) = launch {
            req = req.launch(parse_query_value(v));
        }
        if let Some(v) = current {
            req = req.current(parse_query_value(v));
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }
        if let Some(v) = orderby {
            req = parse_orderby(req, v);
        }

        let records = self
            .client
            .query(&req)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(records.into_iter().map(PySATCATRecord::from).collect())
    }

    /// Query TLE (Two-Line Element) data.
    ///
    /// Note: This class is deprecated. Use gp() instead.
    ///
    /// Args:
    ///     norad_cat_id (int, optional): Filter by NORAD catalog ID
    ///     object_name (str, optional): Filter by object name
    ///     epoch (str, optional): Filter by epoch
    ///     limit (int, optional): Maximum results
    ///     orderby (str, optional): Field to order by
    ///
    /// Returns:
    ///     list[TLERecord]: List of TLE records
    #[pyo3(signature = (
        norad_cat_id=None,
        object_name=None,
        epoch=None,
        limit=None,
        orderby=None
    ))]
    fn tle(
        &self,
        norad_cat_id: Option<u32>,
        object_name: Option<&str>,
        epoch: Option<&str>,
        limit: Option<u32>,
        orderby: Option<&str>,
    ) -> PyResult<Vec<PyTLERecord>> {
        let mut req = spacetrack::request_classes::TLERequest::new();

        if let Some(v) = norad_cat_id {
            req = req.norad_cat_id(v);
        }
        if let Some(v) = object_name {
            req = req.object_name(parse_query_value(v));
        }
        if let Some(v) = epoch {
            req = req.epoch(parse_query_value(v));
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }
        if let Some(v) = orderby {
            req = parse_orderby(req, v);
        }

        let records = self
            .client
            .query(&req)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(records.into_iter().map(PyTLERecord::from).collect())
    }

    /// Query decay data.
    ///
    /// Returns predicted and actual re-entry information.
    ///
    /// Args:
    ///     norad_cat_id (int, optional): Filter by NORAD catalog ID
    ///     decay_epoch (str, optional): Filter by decay epoch
    ///     country (str, optional): Filter by country
    ///     limit (int, optional): Maximum results
    ///
    /// Returns:
    ///     list[DecayRecord]: List of decay records
    #[pyo3(signature = (norad_cat_id=None, decay_epoch=None, country=None, limit=None))]
    fn decay(
        &self,
        norad_cat_id: Option<u32>,
        decay_epoch: Option<&str>,
        country: Option<&str>,
        limit: Option<u32>,
    ) -> PyResult<Vec<PyDecayRecord>> {
        let mut req = spacetrack::request_classes::DecayRequest::new();

        if let Some(v) = norad_cat_id {
            req = req.norad_cat_id(v);
        }
        if let Some(v) = decay_epoch {
            req = req.decay_epoch(parse_query_value(v));
        }
        if let Some(v) = country {
            req = req.country(parse_query_value(v));
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }

        let records = self
            .client
            .query(&req)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(records.into_iter().map(PyDecayRecord::from).collect())
    }

    /// Query TIP (Tracking and Impact Prediction) data.
    ///
    /// Returns tracking and impact prediction messages.
    ///
    /// Args:
    ///     norad_cat_id (int, optional): Filter by NORAD catalog ID
    ///     decay_epoch (str, optional): Filter by decay epoch
    ///     high_interest (str, optional): Filter by high interest flag
    ///     limit (int, optional): Maximum results
    ///
    /// Returns:
    ///     list[TIPRecord]: List of TIP records
    #[pyo3(signature = (norad_cat_id=None, decay_epoch=None, high_interest=None, limit=None))]
    fn tip(
        &self,
        norad_cat_id: Option<u32>,
        decay_epoch: Option<&str>,
        high_interest: Option<&str>,
        limit: Option<u32>,
    ) -> PyResult<Vec<PyTIPRecord>> {
        let mut req = spacetrack::request_classes::TIPRequest::new();

        if let Some(v) = norad_cat_id {
            req = req.norad_cat_id(v);
        }
        if let Some(v) = decay_epoch {
            req = req.decay_epoch(parse_query_value(v));
        }
        if let Some(v) = high_interest {
            req = req.high_interest(parse_query_value(v));
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }

        let records = self
            .client
            .query(&req)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(records.into_iter().map(PyTIPRecord::from).collect())
    }

    /// Query public CDM (Conjunction Data Message) data.
    ///
    /// Returns public conjunction data.
    ///
    /// Args:
    ///     sat1_norad_cat_id (int, optional): Filter by SAT1 NORAD catalog ID
    ///     sat2_norad_cat_id (int, optional): Filter by SAT2 NORAD catalog ID
    ///     tca (str, optional): Filter by time of closest approach
    ///     limit (int, optional): Maximum results
    ///
    /// Returns:
    ///     list[CDMPublicRecord]: List of CDM records
    #[pyo3(signature = (sat1_norad_cat_id=None, sat2_norad_cat_id=None, tca=None, limit=None))]
    fn cdm_public(
        &self,
        sat1_norad_cat_id: Option<u32>,
        sat2_norad_cat_id: Option<u32>,
        tca: Option<&str>,
        limit: Option<u32>,
    ) -> PyResult<Vec<PyCDMPublicRecord>> {
        let mut req = spacetrack::request_classes::CDMPublicRequest::new();

        if let Some(v) = sat1_norad_cat_id {
            req = req.sat1_norad_cat_id(v);
        }
        if let Some(v) = sat2_norad_cat_id {
            req = req.sat2_norad_cat_id(v);
        }
        if let Some(v) = tca {
            req = req.tca(parse_query_value(v));
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }

        let records = self
            .client
            .query(&req)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(records.into_iter().map(PyCDMPublicRecord::from).collect())
    }

    /// Query boxscore (catalog statistics) data.
    ///
    /// Returns catalog summary statistics by country.
    ///
    /// Args:
    ///     country (str, optional): Filter by country code
    ///
    /// Returns:
    ///     list[BoxscoreRecord]: List of boxscore records
    #[pyo3(signature = (country=None))]
    fn boxscore(&self, country: Option<&str>) -> PyResult<Vec<PyBoxscoreRecord>> {
        let mut req = spacetrack::request_classes::BoxscoreRequest::new();

        if let Some(v) = country {
            req = req.country(parse_query_value(v));
        }

        let records = self
            .client
            .query(&req)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(records.into_iter().map(PyBoxscoreRecord::from).collect())
    }

    /// Query launch site data.
    ///
    /// Returns launch facility information.
    ///
    /// Args:
    ///     site_code (str, optional): Filter by site code
    ///
    /// Returns:
    ///     list[LaunchSiteRecord]: List of launch site records
    #[pyo3(signature = (site_code=None))]
    fn launch_site(&self, site_code: Option<&str>) -> PyResult<Vec<PyLaunchSiteRecord>> {
        let mut req = spacetrack::request_classes::LaunchSiteRequest::new();

        if let Some(v) = site_code {
            req = req.site_code(parse_query_value(v));
        }

        let records = self
            .client
            .query(&req)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(records.into_iter().map(PyLaunchSiteRecord::from).collect())
    }

    /// Query SATCAT change data.
    ///
    /// Returns information about catalog changes.
    ///
    /// Args:
    ///     norad_cat_id (int, optional): Filter by NORAD catalog ID
    ///     change_made (str, optional): Filter by change timestamp
    ///     limit (int, optional): Maximum results
    ///
    /// Returns:
    ///     list[SATCATChangeRecord]: List of SATCAT change records
    #[pyo3(signature = (norad_cat_id=None, change_made=None, limit=None))]
    fn satcat_change(
        &self,
        norad_cat_id: Option<u32>,
        change_made: Option<&str>,
        limit: Option<u32>,
    ) -> PyResult<Vec<PySATCATChangeRecord>> {
        let mut req = spacetrack::request_classes::SATCATChangeRequest::new();

        if let Some(v) = norad_cat_id {
            req = req.norad_cat_id(v);
        }
        if let Some(v) = change_made {
            req = req.change_made(parse_query_value(v));
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }

        let records = self
            .client
            .query(&req)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(records.into_iter().map(PySATCATChangeRecord::from).collect())
    }

    /// Query SATCAT debut (new objects) data.
    ///
    /// Returns information about new catalog entries.
    ///
    /// Args:
    ///     debut (str, optional): Filter by debut date
    ///     object_type (str, optional): Filter by type
    ///     country (str, optional): Filter by country
    ///     limit (int, optional): Maximum results
    ///
    /// Returns:
    ///     list[SATCATDebutRecord]: List of SATCAT debut records
    #[pyo3(signature = (debut=None, object_type=None, country=None, limit=None))]
    fn satcat_debut(
        &self,
        debut: Option<&str>,
        object_type: Option<&str>,
        country: Option<&str>,
        limit: Option<u32>,
    ) -> PyResult<Vec<PySATCATDebutRecord>> {
        let mut req = spacetrack::request_classes::SATCATDebutRequest::new();

        if let Some(v) = debut {
            req = req.debut(parse_query_value(v));
        }
        if let Some(v) = object_type {
            req = req.object_type(parse_query_value(v));
        }
        if let Some(v) = country {
            req = req.country(parse_query_value(v));
        }
        if let Some(v) = limit {
            req = req.limit(v);
        }

        let records = self
            .client
            .query(&req)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(records.into_iter().map(PySATCATDebutRecord::from).collect())
    }

    /// Query announcement data.
    ///
    /// Returns Space-Track announcements.
    ///
    /// Args:
    ///     limit (int, optional): Maximum results
    ///
    /// Returns:
    ///     list[AnnouncementRecord]: List of announcement records
    #[pyo3(signature = (limit=None))]
    fn announcement(&self, limit: Option<u32>) -> PyResult<Vec<PyAnnouncementRecord>> {
        let mut req = spacetrack::request_classes::AnnouncementRequest::new();

        if let Some(v) = limit {
            req = req.limit(v);
        }

        let records = self
            .client
            .query(&req)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(records.into_iter().map(PyAnnouncementRecord::from).collect())
    }

    /// Make a generic API request.
    ///
    /// For advanced use when the typed methods don't meet your needs.
    ///
    /// Args:
    ///     controller (str): Controller name (e.g., "basicspacedata")
    ///     class_name (str): Request class name (e.g., "gp")
    ///     predicates (dict): Query predicates as key-value pairs
    ///     format (str, optional): Output format (json, tle, 3le, etc.)
    ///
    /// Returns:
    ///     str: Raw response body
    ///
    /// Example:
    ///     ```python
    ///     response = client.generic_request(
    ///         "basicspacedata",
    ///         "gp",
    ///         {"NORAD_CAT_ID": "25544", "limit": "1"},
    ///         format="3le"
    ///     )
    ///     print(response)
    ///     ```
    #[pyo3(signature = (controller, class_name, predicates, format=None))]
    fn generic_request(
        &self,
        controller: &str,
        class_name: &str,
        predicates: HashMap<String, String>,
        format: Option<&str>,
    ) -> PyResult<String> {
        let pred_vec: Vec<(&str, spacetrack::QueryValue)> = predicates
            .iter()
            .map(|(k, v)| {
                // Leak the string to get a static reference
                let key: &'static str = Box::leak(k.clone().into_boxed_str());
                (key, parse_query_value(v))
            })
            .collect();

        self.client
            .generic_request(controller, class_name, &pred_vec, format)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Execute a query using the fluent query builder API.
    ///
    /// This method executes a query built with the fluent API and returns
    /// the appropriate record type based on the query class.
    ///
    /// Args:
    ///     query (SpaceTrackQuery): The query to execute
    ///
    /// Returns:
    ///     list: List of records (type depends on query class)
    ///
    /// Raises:
    ///     RuntimeError: If query execution fails
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.SpaceTrackClient("username", "password")
    ///
    ///     # Build and execute a GP query
    ///     query = (bh.SpaceTrackQuery.gp()
    ///         .filter(bh.SpaceTrackPredicate.norad_cat_id().eq(25544))
    ///         .order_by_desc(bh.SpaceTrackPredicate.epoch())
    ///         .limit(10))
    ///
    ///     records = client.execute_query(query)
    ///     for record in records:
    ///         print(f"{record.object_name}: {record.epoch}")
    ///     ```
    fn execute_query(&self, py: Python<'_>, query: &PySpaceTrackQuery) -> PyResult<Py<PyAny>> {
        let class_name = query.inner.class();

        // Route to appropriate query method based on class
        match class_name {
            "gp" => {
                let records: Vec<spacetrack::GPRecord> = self
                    .client
                    .execute_query(&query.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let py_records: Vec<PyGPRecord> =
                    records.into_iter().map(PyGPRecord::from).collect();
                Ok(py_records.into_pyobject(py)?.into_any().unbind())
            }
            "tle" => {
                let records: Vec<spacetrack::TLERecord> = self
                    .client
                    .execute_query(&query.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let py_records: Vec<PyTLERecord> =
                    records.into_iter().map(PyTLERecord::from).collect();
                Ok(py_records.into_pyobject(py)?.into_any().unbind())
            }
            "satcat" => {
                let records: Vec<spacetrack::SATCATRecord> = self
                    .client
                    .execute_query(&query.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let py_records: Vec<PySATCATRecord> =
                    records.into_iter().map(PySATCATRecord::from).collect();
                Ok(py_records.into_pyobject(py)?.into_any().unbind())
            }
            "omm" => {
                let records: Vec<spacetrack::OMMRecord> = self
                    .client
                    .execute_query(&query.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let py_records: Vec<PyOMMRecord> =
                    records.into_iter().map(PyOMMRecord::from).collect();
                Ok(py_records.into_pyobject(py)?.into_any().unbind())
            }
            "gp_history" => {
                let records: Vec<spacetrack::GPHistoryRecord> = self
                    .client
                    .execute_query(&query.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let py_records: Vec<PyGPHistoryRecord> =
                    records.into_iter().map(PyGPHistoryRecord::from).collect();
                Ok(py_records.into_pyobject(py)?.into_any().unbind())
            }
            "decay" => {
                let records: Vec<spacetrack::DecayRecord> = self
                    .client
                    .execute_query(&query.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let py_records: Vec<PyDecayRecord> =
                    records.into_iter().map(PyDecayRecord::from).collect();
                Ok(py_records.into_pyobject(py)?.into_any().unbind())
            }
            "tip" => {
                let records: Vec<spacetrack::TIPRecord> = self
                    .client
                    .execute_query(&query.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let py_records: Vec<PyTIPRecord> =
                    records.into_iter().map(PyTIPRecord::from).collect();
                Ok(py_records.into_pyobject(py)?.into_any().unbind())
            }
            "cdm_public" => {
                let records: Vec<spacetrack::CDMPublicRecord> = self
                    .client
                    .execute_query(&query.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let py_records: Vec<PyCDMPublicRecord> =
                    records.into_iter().map(PyCDMPublicRecord::from).collect();
                Ok(py_records.into_pyobject(py)?.into_any().unbind())
            }
            "boxscore" => {
                let records: Vec<spacetrack::BoxscoreRecord> = self
                    .client
                    .execute_query(&query.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let py_records: Vec<PyBoxscoreRecord> =
                    records.into_iter().map(PyBoxscoreRecord::from).collect();
                Ok(py_records.into_pyobject(py)?.into_any().unbind())
            }
            "launch_site" => {
                let records: Vec<spacetrack::LaunchSiteRecord> = self
                    .client
                    .execute_query(&query.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let py_records: Vec<PyLaunchSiteRecord> =
                    records.into_iter().map(PyLaunchSiteRecord::from).collect();
                Ok(py_records.into_pyobject(py)?.into_any().unbind())
            }
            "announcement" => {
                let records: Vec<spacetrack::AnnouncementRecord> = self
                    .client
                    .execute_query(&query.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let py_records: Vec<PyAnnouncementRecord> =
                    records.into_iter().map(PyAnnouncementRecord::from).collect();
                Ok(py_records.into_pyobject(py)?.into_any().unbind())
            }
            "satcat_change" => {
                let records: Vec<spacetrack::SATCATChangeRecord> = self
                    .client
                    .execute_query(&query.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let py_records: Vec<PySATCATChangeRecord> =
                    records.into_iter().map(PySATCATChangeRecord::from).collect();
                Ok(py_records.into_pyobject(py)?.into_any().unbind())
            }
            "satcat_debut" => {
                let records: Vec<spacetrack::SATCATDebutRecord> = self
                    .client
                    .execute_query(&query.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                let py_records: Vec<PySATCATDebutRecord> =
                    records.into_iter().map(PySATCATDebutRecord::from).collect();
                Ok(py_records.into_pyobject(py)?.into_any().unbind())
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Unknown query class: {}",
                class_name
            ))),
        }
    }

    /// Execute a GP query and return results as SGP propagators.
    ///
    /// This is a convenience method for querying GP data and converting
    /// directly to propagators.
    ///
    /// Args:
    ///     query (SpaceTrackQuery): A GP query
    ///     step_size (float): Propagator step size in seconds
    ///
    /// Returns:
    ///     list[SGPPropagator]: List of SGP propagators
    ///
    /// Raises:
    ///     RuntimeError: If query fails or records can't be converted
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     query = (bh.SpaceTrackQuery.gp()
    ///         .filter(bh.SpaceTrackPredicate.object_name().contains("STARLINK"))
    ///         .limit(10))
    ///
    ///     propagators = client.execute_query_as_sgp_propagators(query, 60.0)
    ///     ```
    #[pyo3(signature = (query, step_size))]
    fn execute_query_as_sgp_propagators(
        &self,
        query: &PySpaceTrackQuery,
        step_size: f64,
    ) -> PyResult<Vec<PySGPPropagator>> {
        let class_name = query.inner.class();

        // Only support classes that have TLE data
        match class_name {
            "gp" => {
                let records: Vec<spacetrack::GPRecord> = self
                    .client
                    .execute_query(&query.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

                let propagators: Vec<PySGPPropagator> = records
                    .iter()
                    .filter_map(|r| {
                        r.to_sgp_propagator(step_size)
                            .ok()
                            .map(|p| PySGPPropagator { propagator: p })
                    })
                    .collect();
                Ok(propagators)
            }
            "tle" => {
                let records: Vec<spacetrack::TLERecord> = self
                    .client
                    .execute_query(&query.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

                let propagators: Vec<PySGPPropagator> = records
                    .iter()
                    .filter_map(|r| {
                        r.to_sgp_propagator(step_size)
                            .ok()
                            .map(|p| PySGPPropagator { propagator: p })
                    })
                    .collect();
                Ok(propagators)
            }
            "omm" => {
                let records: Vec<spacetrack::OMMRecord> = self
                    .client
                    .execute_query(&query.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

                let propagators: Vec<PySGPPropagator> = records
                    .iter()
                    .filter_map(|r| {
                        r.to_sgp_propagator(step_size)
                            .ok()
                            .map(|p| PySGPPropagator { propagator: p })
                    })
                    .collect();
                Ok(propagators)
            }
            "gp_history" => {
                let records: Vec<spacetrack::GPHistoryRecord> = self
                    .client
                    .execute_query(&query.inner)
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

                let propagators: Vec<PySGPPropagator> = records
                    .iter()
                    .filter_map(|r| {
                        r.to_sgp_propagator(step_size)
                            .ok()
                            .map(|p| PySGPPropagator { propagator: p })
                    })
                    .collect();
                Ok(propagators)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Query class '{}' does not support propagator conversion. Use gp, tle, omm, or gp_history.",
                class_name
            ))),
        }
    }
}

/// Parse a string value into a QueryValue, handling operators.
fn parse_query_value(value: &str) -> spacetrack::QueryValue {
    let trimmed = value.trim();

    // Check for operators using strip_prefix for cleaner code
    if let Some(rest) = trimmed.strip_prefix("><").or_else(|| trimmed.strip_prefix("<>")) {
        spacetrack::not_equal(rest)
    } else if let Some(rest) = trimmed.strip_prefix(">=") {
        // Greater than or equal - use range
        spacetrack::inclusive_range(rest, "now")
    } else if let Some(rest) = trimmed.strip_prefix("<=") {
        // Less than or equal - use range
        spacetrack::inclusive_range("0000-01-01", rest)
    } else if let Some(rest) = trimmed.strip_prefix('>') {
        spacetrack::greater_than(rest)
    } else if let Some(rest) = trimmed.strip_prefix('<') {
        spacetrack::less_than(rest)
    } else if let Some(rest) = trimmed.strip_prefix("~~") {
        spacetrack::like(rest)
    } else if let Some(rest) = trimmed.strip_prefix('^') {
        spacetrack::startswith(rest)
    } else if trimmed.contains("--") {
        let parts: Vec<&str> = trimmed.split("--").collect();
        if parts.len() == 2 {
            spacetrack::inclusive_range(parts[0], parts[1])
        } else {
            spacetrack::equals(trimmed)
        }
    } else if trimmed == "null-val" {
        spacetrack::null_val()
    } else {
        spacetrack::equals(trimmed)
    }
}

/// Parse orderby string into request.
fn parse_orderby<R>(mut req: R, orderby: &str) -> R
where
    R: OrderByMethods,
{
    let lower = orderby.to_lowercase();
    if lower.ends_with(" desc") {
        let field = orderby[..orderby.len() - 5].trim();
        req.set_orderby_desc(field);
    } else if lower.ends_with(" asc") {
        let field = orderby[..orderby.len() - 4].trim();
        req.set_orderby_asc(field);
    } else {
        req.set_orderby_asc(orderby);
    }
    req
}

/// Trait for request types that support orderby.
trait OrderByMethods: Sized {
    fn set_orderby_asc(&mut self, field: &str);
    fn set_orderby_desc(&mut self, field: &str);
}

// Implement OrderByMethods for all request types
macro_rules! impl_orderby {
    ($($t:ty),*) => {
        $(
            impl OrderByMethods for $t {
                fn set_orderby_asc(&mut self, field: &str) {
                    *self = std::mem::take(self).orderby_asc(field);
                }
                fn set_orderby_desc(&mut self, field: &str) {
                    *self = std::mem::take(self).orderby_desc(field);
                }
            }
        )*
    };
}

impl_orderby!(
    spacetrack::request_classes::GPRequest,
    spacetrack::request_classes::SATCATRequest,
    spacetrack::request_classes::TLERequest
);

// Functions are registered in mod.rs via add_class() calls
