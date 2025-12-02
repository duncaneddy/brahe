/*!
 * SpaceTrack query builder types
 *
 * This module provides a fluent, type-safe API for constructing SpaceTrack queries.
 * It replaces the string-based operator syntax with composable predicate builders.
 */

use std::fmt::Display;

use crate::time::Epoch;

use super::operators::QueryValue;

// ============================================================================
// SpaceTrackValue - Type-safe query values
// ============================================================================

/// Value types that can be used in SpaceTrack queries.
///
/// This enum provides type-safe handling of different value types,
/// with automatic formatting for the SpaceTrack API.
#[derive(Debug, Clone, PartialEq)]
pub enum SpaceTrackValue {
    /// Integer value
    Integer(i64),
    /// Floating point value
    Float(f64),
    /// String value
    String(String),
    /// Epoch value (auto-formatted to ISO string)
    Epoch(Epoch),
    /// Boolean value
    Boolean(bool),
}

impl SpaceTrackValue {
    /// Convert the value to a string suitable for SpaceTrack API queries.
    pub fn to_query_string(&self) -> String {
        match self {
            SpaceTrackValue::Integer(v) => v.to_string(),
            SpaceTrackValue::Float(v) => v.to_string(),
            SpaceTrackValue::String(v) => v.clone(),
            SpaceTrackValue::Epoch(e) => e.isostring(),
            SpaceTrackValue::Boolean(v) => if *v { "true" } else { "false" }.to_string(),
        }
    }
}

impl Display for SpaceTrackValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_query_string())
    }
}

// From implementations for SpaceTrackValue
impl From<i32> for SpaceTrackValue {
    fn from(v: i32) -> Self {
        SpaceTrackValue::Integer(v as i64)
    }
}

impl From<i64> for SpaceTrackValue {
    fn from(v: i64) -> Self {
        SpaceTrackValue::Integer(v)
    }
}

impl From<u32> for SpaceTrackValue {
    fn from(v: u32) -> Self {
        SpaceTrackValue::Integer(v as i64)
    }
}

impl From<u64> for SpaceTrackValue {
    fn from(v: u64) -> Self {
        SpaceTrackValue::Integer(v as i64)
    }
}

impl From<f32> for SpaceTrackValue {
    fn from(v: f32) -> Self {
        SpaceTrackValue::Float(v as f64)
    }
}

impl From<f64> for SpaceTrackValue {
    fn from(v: f64) -> Self {
        SpaceTrackValue::Float(v)
    }
}

impl From<&str> for SpaceTrackValue {
    fn from(v: &str) -> Self {
        SpaceTrackValue::String(v.to_string())
    }
}

impl From<String> for SpaceTrackValue {
    fn from(v: String) -> Self {
        SpaceTrackValue::String(v)
    }
}

impl From<bool> for SpaceTrackValue {
    fn from(v: bool) -> Self {
        SpaceTrackValue::Boolean(v)
    }
}

impl From<Epoch> for SpaceTrackValue {
    fn from(v: Epoch) -> Self {
        SpaceTrackValue::Epoch(v)
    }
}

impl From<&Epoch> for SpaceTrackValue {
    fn from(v: &Epoch) -> Self {
        SpaceTrackValue::Epoch(*v)
    }
}

// ============================================================================
// SpaceTrackOrder - Query result ordering
// ============================================================================

/// Order direction for query results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpaceTrackOrder {
    /// Ascending order (A-Z, 0-9, oldest-newest)
    #[default]
    Ascending,
    /// Descending order (Z-A, 9-0, newest-oldest)
    Descending,
}

impl SpaceTrackOrder {
    /// Convert to SpaceTrack query string suffix.
    pub fn as_query_string(&self) -> &'static str {
        match self {
            SpaceTrackOrder::Ascending => " asc",
            SpaceTrackOrder::Descending => " desc",
        }
    }
}

// ============================================================================
// SpaceTrackOperator - Internal operator representation
// ============================================================================

/// Operator representation for query predicates.
#[derive(Debug, Clone, PartialEq)]
pub enum SpaceTrackOperator {
    /// Exact value match
    Equals(SpaceTrackValue),
    /// Greater than comparison (`>value`)
    GreaterThan(SpaceTrackValue),
    /// Less than comparison (`<value`)
    LessThan(SpaceTrackValue),
    /// Greater than or equal comparison (implemented as range with open end)
    GreaterThanOrEqual(SpaceTrackValue),
    /// Less than or equal comparison (implemented as range with open start)
    LessThanOrEqual(SpaceTrackValue),
    /// Not equal comparison (`<>value`)
    NotEqual(SpaceTrackValue),
    /// Inclusive range (`left--right`)
    Range(SpaceTrackValue, SpaceTrackValue),
    /// SQL LIKE pattern matching (`~~value`)
    Like(String),
    /// Starts with prefix (`^value`)
    StartsWith(String),
    /// Null value (`null-val`)
    Null,
}

impl SpaceTrackOperator {
    /// Convert to the internal QueryValue format for URL encoding.
    pub(crate) fn to_query_value(&self) -> QueryValue {
        match self {
            SpaceTrackOperator::Equals(v) => QueryValue::Value(v.to_query_string()),
            SpaceTrackOperator::GreaterThan(v) => QueryValue::GreaterThan(v.to_query_string()),
            SpaceTrackOperator::LessThan(v) => QueryValue::LessThan(v.to_query_string()),
            // >= is implemented as a range from value to empty (SpaceTrack interprets this as >=)
            SpaceTrackOperator::GreaterThanOrEqual(v) => {
                QueryValue::Range(v.to_query_string(), "".to_string())
            }
            // <= is implemented as a range from empty to value (SpaceTrack interprets this as <=)
            SpaceTrackOperator::LessThanOrEqual(v) => {
                QueryValue::Range("".to_string(), v.to_query_string())
            }
            SpaceTrackOperator::NotEqual(v) => QueryValue::NotEqual(v.to_query_string()),
            SpaceTrackOperator::Range(l, r) => {
                QueryValue::Range(l.to_query_string(), r.to_query_string())
            }
            SpaceTrackOperator::Like(p) => QueryValue::Like(p.clone()),
            SpaceTrackOperator::StartsWith(p) => QueryValue::StartsWith(p.clone()),
            SpaceTrackOperator::Null => QueryValue::Null,
        }
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
/// # Example
///
/// ```
/// use brahe::spacetrack::SpaceTrackPredicate;
///
/// // Using typed field constructors
/// let predicate = SpaceTrackPredicate::norad_cat_id().eq(25544);
///
/// // Using generic field constructor
/// let predicate = SpaceTrackPredicate::field("CUSTOM_FIELD").gt(100);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SpaceTrackPredicate {
    /// The field name
    pub field: String,
    /// The operator with value(s)
    pub operator: SpaceTrackOperator,
}

impl SpaceTrackPredicate {
    /// Create a predicate builder for a generic field by name.
    ///
    /// Use this for fields that don't have a typed constructor method.
    /// The field name will be converted to uppercase automatically.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::SpaceTrackPredicate;
    ///
    /// let predicate = SpaceTrackPredicate::field("FILE").gt(12345);
    /// ```
    pub fn field(name: &str) -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: name.to_uppercase(),
        }
    }

    // ========================================================================
    // Typed field constructors for common fields
    // ========================================================================

    /// Create a predicate builder for the NORAD_CAT_ID field.
    pub fn norad_cat_id() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "NORAD_CAT_ID".to_string(),
        }
    }

    /// Create a predicate builder for the OBJECT_NAME field.
    pub fn object_name() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "OBJECT_NAME".to_string(),
        }
    }

    /// Create a predicate builder for the OBJECT_ID field (international designator).
    pub fn object_id() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "OBJECT_ID".to_string(),
        }
    }

    /// Create a predicate builder for the EPOCH field.
    pub fn epoch() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "EPOCH".to_string(),
        }
    }

    /// Create a predicate builder for the COUNTRY_CODE field.
    pub fn country_code() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "COUNTRY_CODE".to_string(),
        }
    }

    /// Create a predicate builder for the LAUNCH_DATE field.
    pub fn launch_date() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "LAUNCH_DATE".to_string(),
        }
    }

    /// Create a predicate builder for the DECAY_DATE field.
    pub fn decay_date() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "DECAY_DATE".to_string(),
        }
    }

    /// Create a predicate builder for the ECCENTRICITY field.
    pub fn eccentricity() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "ECCENTRICITY".to_string(),
        }
    }

    /// Create a predicate builder for the INCLINATION field.
    pub fn inclination() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "INCLINATION".to_string(),
        }
    }

    /// Create a predicate builder for the MEAN_MOTION field.
    pub fn mean_motion() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "MEAN_MOTION".to_string(),
        }
    }

    /// Create a predicate builder for the OBJECT_TYPE field.
    pub fn object_type() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "OBJECT_TYPE".to_string(),
        }
    }

    /// Create a predicate builder for the RCS_SIZE field.
    pub fn rcs_size() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "RCS_SIZE".to_string(),
        }
    }

    /// Create a predicate builder for the PERIOD field.
    pub fn period() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "PERIOD".to_string(),
        }
    }

    /// Create a predicate builder for the APOAPSIS field.
    pub fn apoapsis() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "APOAPSIS".to_string(),
        }
    }

    /// Create a predicate builder for the PERIAPSIS field.
    pub fn periapsis() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "PERIAPSIS".to_string(),
        }
    }

    /// Create a predicate builder for the SEMIMAJOR_AXIS field.
    pub fn semimajor_axis() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "SEMIMAJOR_AXIS".to_string(),
        }
    }

    /// Create a predicate builder for the BSTAR field.
    pub fn bstar() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "BSTAR".to_string(),
        }
    }

    /// Create a predicate builder for the CREATION_DATE field.
    pub fn creation_date() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "CREATION_DATE".to_string(),
        }
    }

    /// Create a predicate builder for the FILE field.
    pub fn file() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "FILE".to_string(),
        }
    }

    /// Create a predicate builder for the GP_ID field.
    pub fn gp_id() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "GP_ID".to_string(),
        }
    }

    /// Create a predicate builder for the TLE_LINE0 field.
    pub fn tle_line0() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "TLE_LINE0".to_string(),
        }
    }

    /// Create a predicate builder for the TLE_LINE1 field.
    pub fn tle_line1() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "TLE_LINE1".to_string(),
        }
    }

    /// Create a predicate builder for the TLE_LINE2 field.
    pub fn tle_line2() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "TLE_LINE2".to_string(),
        }
    }

    /// Create a predicate builder for the DECAYED field.
    pub fn decayed() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "DECAYED".to_string(),
        }
    }

    /// Create a predicate builder for the SITE field.
    pub fn site() -> SpaceTrackPredicateBuilder {
        SpaceTrackPredicateBuilder {
            field: "SITE".to_string(),
        }
    }

    // ========================================================================
    // Internal methods
    // ========================================================================

    /// Get the field name.
    pub fn field_name(&self) -> &str {
        &self.field
    }

    /// Convert to the internal (field, QueryValue) tuple format.
    pub(crate) fn to_query_tuple(&self) -> (String, QueryValue) {
        (self.field.clone(), self.operator.to_query_value())
    }
}

// ============================================================================
// SpaceTrackPredicateBuilder - Fluent predicate construction
// ============================================================================

/// Builder for constructing SpaceTrack predicates with fluent syntax.
///
/// This struct is created by `SpaceTrackPredicate::field()` or one of the
/// typed field constructors, and provides methods for specifying the comparison
/// operator and value.
///
/// # Example
///
/// ```
/// use brahe::spacetrack::SpaceTrackPredicate;
///
/// // The builder is consumed when an operator method is called
/// let predicate = SpaceTrackPredicate::epoch().gt("2024-01-01");
/// ```
#[derive(Debug, Clone)]
pub struct SpaceTrackPredicateBuilder {
    /// The field name
    pub field: String,
}

impl SpaceTrackPredicateBuilder {
    /// Get the field name this builder is for.
    pub fn field_name(&self) -> &str {
        &self.field
    }

    // ========================================================================
    // Comparison operators
    // ========================================================================

    /// Create an equals predicate (`=value`).
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::SpaceTrackPredicate;
    ///
    /// let predicate = SpaceTrackPredicate::norad_cat_id().eq(25544);
    /// ```
    pub fn eq(self, value: impl Into<SpaceTrackValue>) -> SpaceTrackPredicate {
        SpaceTrackPredicate {
            field: self.field,
            operator: SpaceTrackOperator::Equals(value.into()),
        }
    }

    /// Create a greater-than predicate (`>value`).
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::SpaceTrackPredicate;
    ///
    /// let predicate = SpaceTrackPredicate::epoch().gt("2024-01-01");
    /// ```
    pub fn gt(self, value: impl Into<SpaceTrackValue>) -> SpaceTrackPredicate {
        SpaceTrackPredicate {
            field: self.field,
            operator: SpaceTrackOperator::GreaterThan(value.into()),
        }
    }

    /// Create a less-than predicate (`<value`).
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::SpaceTrackPredicate;
    ///
    /// let predicate = SpaceTrackPredicate::eccentricity().lt(0.01);
    /// ```
    pub fn lt(self, value: impl Into<SpaceTrackValue>) -> SpaceTrackPredicate {
        SpaceTrackPredicate {
            field: self.field,
            operator: SpaceTrackOperator::LessThan(value.into()),
        }
    }

    /// Create a greater-than-or-equal predicate (`>=value`).
    ///
    /// Note: SpaceTrack implements this as a range with an open end.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::SpaceTrackPredicate;
    ///
    /// let predicate = SpaceTrackPredicate::inclination().gte(90.0);
    /// ```
    pub fn gte(self, value: impl Into<SpaceTrackValue>) -> SpaceTrackPredicate {
        SpaceTrackPredicate {
            field: self.field,
            operator: SpaceTrackOperator::GreaterThanOrEqual(value.into()),
        }
    }

    /// Create a less-than-or-equal predicate (`<=value`).
    ///
    /// Note: SpaceTrack implements this as a range with an open start.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::SpaceTrackPredicate;
    ///
    /// let predicate = SpaceTrackPredicate::inclination().lte(90.0);
    /// ```
    pub fn lte(self, value: impl Into<SpaceTrackValue>) -> SpaceTrackPredicate {
        SpaceTrackPredicate {
            field: self.field,
            operator: SpaceTrackOperator::LessThanOrEqual(value.into()),
        }
    }

    /// Create a not-equal predicate (`<>value`).
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::SpaceTrackPredicate;
    ///
    /// let predicate = SpaceTrackPredicate::country_code().ne("US");
    /// ```
    pub fn ne(self, value: impl Into<SpaceTrackValue>) -> SpaceTrackPredicate {
        SpaceTrackPredicate {
            field: self.field,
            operator: SpaceTrackOperator::NotEqual(value.into()),
        }
    }

    /// Create an inclusive range predicate (`start--end`).
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::SpaceTrackPredicate;
    ///
    /// let predicate = SpaceTrackPredicate::epoch().between("2024-01-01", "2024-12-31");
    /// ```
    pub fn between(
        self,
        start: impl Into<SpaceTrackValue>,
        end: impl Into<SpaceTrackValue>,
    ) -> SpaceTrackPredicate {
        SpaceTrackPredicate {
            field: self.field,
            operator: SpaceTrackOperator::Range(start.into(), end.into()),
        }
    }

    // ========================================================================
    // String operators
    // ========================================================================

    /// Create a SQL LIKE pattern predicate (`~~pattern`).
    ///
    /// The pattern uses SQL LIKE syntax where:
    /// - `%` matches any sequence of characters
    /// - `_` matches any single character
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::SpaceTrackPredicate;
    ///
    /// let predicate = SpaceTrackPredicate::object_name().like("%STARLINK%");
    /// ```
    pub fn like(self, pattern: &str) -> SpaceTrackPredicate {
        SpaceTrackPredicate {
            field: self.field,
            operator: SpaceTrackOperator::Like(pattern.to_string()),
        }
    }

    /// Create a starts-with predicate (`^prefix`).
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::SpaceTrackPredicate;
    ///
    /// let predicate = SpaceTrackPredicate::object_name().starts_with("ISS");
    /// ```
    pub fn starts_with(self, prefix: &str) -> SpaceTrackPredicate {
        SpaceTrackPredicate {
            field: self.field,
            operator: SpaceTrackOperator::StartsWith(prefix.to_string()),
        }
    }

    /// Create a contains predicate (wraps with `%` for LIKE pattern).
    ///
    /// This is a convenience method equivalent to `.like("%substring%")`.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::SpaceTrackPredicate;
    ///
    /// let predicate = SpaceTrackPredicate::object_name().contains("STARLINK");
    /// // Equivalent to: .like("%STARLINK%")
    /// ```
    pub fn contains(self, substring: &str) -> SpaceTrackPredicate {
        SpaceTrackPredicate {
            field: self.field,
            operator: SpaceTrackOperator::Like(format!("%{}%", substring)),
        }
    }

    // ========================================================================
    // Null check
    // ========================================================================

    /// Create a null value predicate (`null-val`).
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::SpaceTrackPredicate;
    ///
    /// let predicate = SpaceTrackPredicate::decay_date().is_null();
    /// ```
    pub fn is_null(self) -> SpaceTrackPredicate {
        SpaceTrackPredicate {
            field: self.field,
            operator: SpaceTrackOperator::Null,
        }
    }
}

// ============================================================================
// SpaceTrackQuery - Query builder
// ============================================================================

/// Builder for composing SpaceTrack API queries.
///
/// This struct provides a fluent API for constructing queries with filters,
/// ordering, and pagination.
///
/// # Example
///
/// ```
/// use brahe::spacetrack::{SpaceTrackQuery, SpaceTrackPredicate, SpaceTrackOrder};
///
/// let query = SpaceTrackQuery::gp()
///     .filter(SpaceTrackPredicate::norad_cat_id().eq(25544))
///     .filter(SpaceTrackPredicate::epoch().gt("2024-01-01"))
///     .order_by_desc(SpaceTrackPredicate::epoch())
///     .limit(10);
/// ```
#[derive(Debug, Clone)]
pub struct SpaceTrackQuery {
    pub(crate) controller: String,
    pub(crate) class: String,
    pub(crate) predicates: Vec<SpaceTrackPredicate>,
    pub(crate) limit: Option<u32>,
    pub(crate) order_by_field: Option<String>,
    pub(crate) order: Option<SpaceTrackOrder>,
    pub(crate) distinct: bool,
}

impl SpaceTrackQuery {
    /// Create a new query for the specified controller and class.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::SpaceTrackQuery;
    ///
    /// let query = SpaceTrackQuery::new("basicspacedata", "gp");
    /// ```
    pub fn new(controller: &str, class: &str) -> Self {
        Self {
            controller: controller.to_string(),
            class: class.to_string(),
            predicates: Vec::new(),
            limit: None,
            order_by_field: None,
            order: None,
            distinct: false,
        }
    }

    // ========================================================================
    // Convenience constructors for common request classes
    // ========================================================================

    /// Create a query for the GP (General Perturbations) class.
    pub fn gp() -> Self {
        Self::new("basicspacedata", "gp")
    }

    /// Create a query for the TLE class.
    pub fn tle() -> Self {
        Self::new("basicspacedata", "tle")
    }

    /// Create a query for the SATCAT (Satellite Catalog) class.
    pub fn satcat() -> Self {
        Self::new("basicspacedata", "satcat")
    }

    /// Create a query for the OMM (Orbit Mean-Elements Message) class.
    pub fn omm() -> Self {
        Self::new("basicspacedata", "omm")
    }

    /// Create a query for the DECAY class.
    pub fn decay() -> Self {
        Self::new("basicspacedata", "decay")
    }

    /// Create a query for the TIP (Tracking and Impact Prediction) class.
    pub fn tip() -> Self {
        Self::new("basicspacedata", "tip")
    }

    /// Create a query for the GP_HISTORY class.
    pub fn gp_history() -> Self {
        Self::new("basicspacedata", "gp_history")
    }

    /// Create a query for the SATCAT_CHANGE class.
    pub fn satcat_change() -> Self {
        Self::new("basicspacedata", "satcat_change")
    }

    /// Create a query for the SATCAT_DEBUT class.
    pub fn satcat_debut() -> Self {
        Self::new("basicspacedata", "satcat_debut")
    }

    /// Create a query for the LAUNCH_SITE class.
    pub fn launch_site() -> Self {
        Self::new("basicspacedata", "launch_site")
    }

    /// Create a query for the BOXSCORE class.
    pub fn boxscore() -> Self {
        Self::new("basicspacedata", "boxscore")
    }

    /// Create a query for the CDM_PUBLIC (Conjunction Data Message) class.
    pub fn cdm_public() -> Self {
        Self::new("basicspacedata", "cdm_public")
    }

    /// Create a query for the ANNOUNCEMENT class.
    pub fn announcement() -> Self {
        Self::new("basicspacedata", "announcement")
    }

    // ========================================================================
    // Query building methods
    // ========================================================================

    /// Add a predicate filter to the query.
    ///
    /// Multiple filters are combined with AND logic.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::{SpaceTrackQuery, SpaceTrackPredicate};
    ///
    /// let query = SpaceTrackQuery::gp()
    ///     .filter(SpaceTrackPredicate::norad_cat_id().eq(25544))
    ///     .filter(SpaceTrackPredicate::epoch().gt("2024-01-01"));
    /// ```
    pub fn filter(mut self, predicate: SpaceTrackPredicate) -> Self {
        self.predicates.push(predicate);
        self
    }

    /// Set the maximum number of results to return.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::SpaceTrackQuery;
    ///
    /// let query = SpaceTrackQuery::gp().limit(100);
    /// ```
    pub fn limit(mut self, limit: u32) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Set ordering by field with specified direction.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::{SpaceTrackQuery, SpaceTrackPredicate, SpaceTrackOrder};
    ///
    /// let query = SpaceTrackQuery::gp()
    ///     .order_by(SpaceTrackPredicate::epoch(), SpaceTrackOrder::Descending);
    /// ```
    pub fn order_by(mut self, field: SpaceTrackPredicateBuilder, order: SpaceTrackOrder) -> Self {
        self.order_by_field = Some(field.field);
        self.order = Some(order);
        self
    }

    /// Set ordering by field in ascending order.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::{SpaceTrackQuery, SpaceTrackPredicate};
    ///
    /// let query = SpaceTrackQuery::gp()
    ///     .order_by_asc(SpaceTrackPredicate::epoch());
    /// ```
    pub fn order_by_asc(mut self, field: SpaceTrackPredicateBuilder) -> Self {
        self.order_by_field = Some(field.field);
        self.order = Some(SpaceTrackOrder::Ascending);
        self
    }

    /// Set ordering by field in descending order.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::{SpaceTrackQuery, SpaceTrackPredicate};
    ///
    /// let query = SpaceTrackQuery::gp()
    ///     .order_by_desc(SpaceTrackPredicate::epoch());
    /// ```
    pub fn order_by_desc(mut self, field: SpaceTrackPredicateBuilder) -> Self {
        self.order_by_field = Some(field.field);
        self.order = Some(SpaceTrackOrder::Descending);
        self
    }

    /// Enable distinct results only.
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::spacetrack::SpaceTrackQuery;
    ///
    /// let query = SpaceTrackQuery::gp().distinct();
    /// ```
    pub fn distinct(mut self) -> Self {
        self.distinct = true;
        self
    }

    // ========================================================================
    // Internal methods for query execution
    // ========================================================================

    /// Get the controller name.
    pub fn controller(&self) -> &str {
        &self.controller
    }

    /// Get the class name.
    pub fn class(&self) -> &str {
        &self.class
    }

    /// Build the predicates as (name, QueryValue) tuples for the request.
    pub(crate) fn build_predicates(&self) -> Vec<(String, QueryValue)> {
        let mut result: Vec<(String, QueryValue)> =
            self.predicates.iter().map(|p| p.to_query_tuple()).collect();

        if let Some(limit) = self.limit {
            result.push(("limit".to_string(), QueryValue::Value(limit.to_string())));
        }

        if let Some(ref field) = self.order_by_field {
            let order_str = self.order.unwrap_or_default().as_query_string();
            let order_value = format!("{}{}", field, order_str);
            result.push(("orderby".to_string(), QueryValue::Value(order_value)));
        }

        if self.distinct {
            result.push((
                "distinct".to_string(),
                QueryValue::Value("true".to_string()),
            ));
        }

        result
    }
}

// ============================================================================
// HasTLEData trait - Marker for records containing TLE data
// ============================================================================

use crate::propagators::SGPPropagator;
use crate::utils::BraheError;

/// Marker trait for record types that contain TLE data.
///
/// Types implementing this trait can be converted to SGP propagators.
/// This enables the `as_sgp_propagators()` method on query builders.
pub trait HasTLEData {
    /// Convert this record to an SGP propagator.
    ///
    /// # Arguments
    ///
    /// * `step_size` - The propagator step size in seconds
    ///
    /// # Returns
    ///
    /// An SGP propagator initialized with this record's TLE data.
    fn to_sgp_propagator(&self, step_size: f64) -> Result<SGPPropagator, BraheError>;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::time::TimeSystem;
    use crate::utils::testing::setup_global_test_eop;

    #[test]
    fn test_spacetrack_value_from_types() {
        let v: SpaceTrackValue = 42i32.into();
        assert_eq!(v.to_query_string(), "42");

        let v: SpaceTrackValue = 42i64.into();
        assert_eq!(v.to_query_string(), "42");

        let v: SpaceTrackValue = 42u32.into();
        assert_eq!(v.to_query_string(), "42");

        let v: SpaceTrackValue = 1.23f64.into();
        assert_eq!(v.to_query_string(), "1.23");

        let v: SpaceTrackValue = "test".into();
        assert_eq!(v.to_query_string(), "test");

        let v: SpaceTrackValue = "test".to_string().into();
        assert_eq!(v.to_query_string(), "test");

        let v: SpaceTrackValue = true.into();
        assert_eq!(v.to_query_string(), "true");

        let v: SpaceTrackValue = false.into();
        assert_eq!(v.to_query_string(), "false");
    }

    #[test]
    fn test_spacetrack_value_from_epoch() {
        setup_global_test_eop();
        let epoch = Epoch::from_datetime(2024, 1, 15, 12, 30, 45.0, 0.0, TimeSystem::UTC);
        let v: SpaceTrackValue = epoch.into();
        assert_eq!(v.to_query_string(), "2024-01-15T12:30:45Z");
    }

    #[test]
    fn test_spacetrack_order() {
        assert_eq!(SpaceTrackOrder::Ascending.as_query_string(), " asc");
        assert_eq!(SpaceTrackOrder::Descending.as_query_string(), " desc");
    }

    #[test]
    fn test_predicate_eq() {
        let pred = SpaceTrackPredicate::norad_cat_id().eq(25544);
        assert_eq!(pred.field, "NORAD_CAT_ID");
        let (field, value) = pred.to_query_tuple();
        assert_eq!(field, "NORAD_CAT_ID");
        assert_eq!(value.to_url_segment(), "25544");
    }

    #[test]
    fn test_predicate_gt() {
        let pred = SpaceTrackPredicate::epoch().gt("2024-01-01");
        let (_, value) = pred.to_query_tuple();
        assert_eq!(value.to_url_segment(), ">2024-01-01");
    }

    #[test]
    fn test_predicate_lt() {
        let pred = SpaceTrackPredicate::eccentricity().lt(0.01);
        let (_, value) = pred.to_query_tuple();
        assert_eq!(value.to_url_segment(), "<0.01");
    }

    #[test]
    fn test_predicate_gte() {
        let pred = SpaceTrackPredicate::inclination().gte(90.0);
        let (_, value) = pred.to_query_tuple();
        assert_eq!(value.to_url_segment(), "90--");
    }

    #[test]
    fn test_predicate_lte() {
        let pred = SpaceTrackPredicate::inclination().lte(90.0);
        let (_, value) = pred.to_query_tuple();
        assert_eq!(value.to_url_segment(), "--90");
    }

    #[test]
    fn test_predicate_ne() {
        let pred = SpaceTrackPredicate::country_code().ne("US");
        let (_, value) = pred.to_query_tuple();
        assert_eq!(value.to_url_segment(), "<>US");
    }

    #[test]
    fn test_predicate_between() {
        let pred = SpaceTrackPredicate::epoch().between("2024-01-01", "2024-12-31");
        let (_, value) = pred.to_query_tuple();
        assert_eq!(value.to_url_segment(), "2024-01-01--2024-12-31");
    }

    #[test]
    fn test_predicate_like() {
        let pred = SpaceTrackPredicate::object_name().like("%STARLINK%");
        let (_, value) = pred.to_query_tuple();
        assert_eq!(value.to_url_segment(), "~~%STARLINK%");
    }

    #[test]
    fn test_predicate_starts_with() {
        let pred = SpaceTrackPredicate::object_name().starts_with("ISS");
        let (_, value) = pred.to_query_tuple();
        assert_eq!(value.to_url_segment(), "^ISS");
    }

    #[test]
    fn test_predicate_contains() {
        let pred = SpaceTrackPredicate::object_name().contains("STARLINK");
        let (_, value) = pred.to_query_tuple();
        assert_eq!(value.to_url_segment(), "~~%STARLINK%");
    }

    #[test]
    fn test_predicate_is_null() {
        let pred = SpaceTrackPredicate::decay_date().is_null();
        let (_, value) = pred.to_query_tuple();
        assert_eq!(value.to_url_segment(), "null-val");
    }

    #[test]
    fn test_predicate_generic_field() {
        let pred = SpaceTrackPredicate::field("custom_field").eq("value");
        assert_eq!(pred.field, "CUSTOM_FIELD");
    }

    #[test]
    fn test_query_gp() {
        let query = SpaceTrackQuery::gp();
        assert_eq!(query.controller, "basicspacedata");
        assert_eq!(query.class, "gp");
    }

    #[test]
    fn test_query_with_filters() {
        let query = SpaceTrackQuery::gp()
            .filter(SpaceTrackPredicate::norad_cat_id().eq(25544))
            .filter(SpaceTrackPredicate::epoch().gt("2024-01-01"));

        assert_eq!(query.predicates.len(), 2);
    }

    #[test]
    fn test_query_with_limit() {
        let query = SpaceTrackQuery::gp().limit(100);
        assert_eq!(query.limit, Some(100));
    }

    #[test]
    fn test_query_with_order() {
        let query = SpaceTrackQuery::gp()
            .order_by(SpaceTrackPredicate::epoch(), SpaceTrackOrder::Descending);

        assert!(query.order_by_field.is_some());
        assert_eq!(query.order_by_field.as_deref(), Some("EPOCH"));
        assert_eq!(query.order, Some(SpaceTrackOrder::Descending));
    }

    #[test]
    fn test_query_order_by_asc() {
        let query = SpaceTrackQuery::gp().order_by_asc(SpaceTrackPredicate::epoch());

        assert!(query.order_by_field.is_some());
        assert_eq!(query.order_by_field.as_deref(), Some("EPOCH"));
        assert_eq!(query.order, Some(SpaceTrackOrder::Ascending));
    }

    #[test]
    fn test_query_order_by_desc() {
        let query = SpaceTrackQuery::gp().order_by_desc(SpaceTrackPredicate::epoch());

        assert!(query.order_by_field.is_some());
        assert_eq!(query.order_by_field.as_deref(), Some("EPOCH"));
        assert_eq!(query.order, Some(SpaceTrackOrder::Descending));
    }

    #[test]
    fn test_query_distinct() {
        let query = SpaceTrackQuery::gp().distinct();
        assert!(query.distinct);
    }

    #[test]
    fn test_query_build_predicates() {
        let query = SpaceTrackQuery::gp()
            .filter(SpaceTrackPredicate::norad_cat_id().eq(25544))
            .limit(10)
            .order_by_desc(SpaceTrackPredicate::epoch())
            .distinct();

        let predicates = query.build_predicates();

        // Should have: norad_cat_id, limit, orderby, distinct
        assert_eq!(predicates.len(), 4);

        // Check that predicates are present
        assert!(predicates.iter().any(|(k, _)| k == "NORAD_CAT_ID"));
        assert!(predicates.iter().any(|(k, _)| k == "limit"));
        assert!(predicates.iter().any(|(k, _)| k == "orderby"));
        assert!(predicates.iter().any(|(k, _)| k == "distinct"));
    }

    #[test]
    fn test_query_build_predicates_orderby_format() {
        let query = SpaceTrackQuery::gp().order_by_desc(SpaceTrackPredicate::epoch());

        let predicates = query.build_predicates();
        let orderby = predicates.iter().find(|(k, _)| k == "orderby");

        assert!(orderby.is_some());
        let (_, value) = orderby.unwrap();
        assert_eq!(value.to_url_segment(), "EPOCH desc");
    }

    #[test]
    fn test_all_typed_field_constructors() {
        // Just verify all typed constructors create valid builders
        let fields = vec![
            SpaceTrackPredicate::norad_cat_id(),
            SpaceTrackPredicate::object_name(),
            SpaceTrackPredicate::object_id(),
            SpaceTrackPredicate::epoch(),
            SpaceTrackPredicate::country_code(),
            SpaceTrackPredicate::launch_date(),
            SpaceTrackPredicate::decay_date(),
            SpaceTrackPredicate::eccentricity(),
            SpaceTrackPredicate::inclination(),
            SpaceTrackPredicate::mean_motion(),
            SpaceTrackPredicate::object_type(),
            SpaceTrackPredicate::rcs_size(),
            SpaceTrackPredicate::period(),
            SpaceTrackPredicate::apoapsis(),
            SpaceTrackPredicate::periapsis(),
            SpaceTrackPredicate::semimajor_axis(),
            SpaceTrackPredicate::bstar(),
            SpaceTrackPredicate::creation_date(),
            SpaceTrackPredicate::file(),
            SpaceTrackPredicate::gp_id(),
            SpaceTrackPredicate::tle_line0(),
            SpaceTrackPredicate::tle_line1(),
            SpaceTrackPredicate::tle_line2(),
            SpaceTrackPredicate::decayed(),
            SpaceTrackPredicate::site(),
        ];

        for builder in fields {
            // Each should have a non-empty field name
            assert!(!builder.field_name().is_empty());
            // And should be able to create a predicate
            let pred = builder.eq("test");
            assert!(!pred.field_name().is_empty());
        }
    }

    #[test]
    fn test_all_query_constructors() {
        let queries = vec![
            SpaceTrackQuery::gp(),
            SpaceTrackQuery::tle(),
            SpaceTrackQuery::satcat(),
            SpaceTrackQuery::omm(),
            SpaceTrackQuery::decay(),
            SpaceTrackQuery::tip(),
            SpaceTrackQuery::gp_history(),
            SpaceTrackQuery::satcat_change(),
            SpaceTrackQuery::satcat_debut(),
            SpaceTrackQuery::launch_site(),
            SpaceTrackQuery::boxscore(),
            SpaceTrackQuery::cdm_public(),
            SpaceTrackQuery::announcement(),
        ];

        for query in queries {
            assert!(!query.controller().is_empty());
            assert!(!query.class().is_empty());
        }
    }
}
