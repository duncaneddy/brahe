/*!
 * SpaceTrack query operators
 *
 * These operators are used to build query predicates for SpaceTrack API requests.
 * They translate to URL-encoded query parameters in the SpaceTrack REST API.
 */

use std::fmt::Display;

/// A query value that can be used as a predicate filter in SpaceTrack queries.
///
/// This enum represents the different types of filter operations supported
/// by the SpaceTrack API.
#[derive(Debug, Clone, PartialEq)]
pub enum QueryValue {
    /// Exact value match.
    Value(String),
    /// Greater than comparison (`>value`).
    GreaterThan(String),
    /// Less than comparison (`<value`).
    LessThan(String),
    /// Not equal comparison (`<>value`).
    NotEqual(String),
    /// Inclusive range (`left--right`).
    Range(String, String),
    /// SQL LIKE pattern matching (`~~value`).
    Like(String),
    /// Starts with prefix (`^value`).
    StartsWith(String),
    /// Null value (`null-val`).
    Null,
}

impl QueryValue {
    /// Convert the query value to a URL-safe string for the SpaceTrack API.
    pub fn to_url_segment(&self) -> String {
        match self {
            QueryValue::Value(v) => v.clone(),
            QueryValue::GreaterThan(v) => format!(">{}", v),
            QueryValue::LessThan(v) => format!("<{}", v),
            QueryValue::NotEqual(v) => format!("<>{}", v),
            QueryValue::Range(l, r) => format!("{}--{}", l, r),
            QueryValue::Like(v) => format!("~~{}", v),
            QueryValue::StartsWith(v) => format!("^{}", v),
            QueryValue::Null => "null-val".to_string(),
        }
    }
}

impl Display for QueryValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_url_segment())
    }
}

/// Create a greater-than filter (`>value`).
///
/// # Arguments
///
/// * `value` - The value to compare against
///
/// # Returns
///
/// A `QueryValue::GreaterThan` variant
///
/// # Example
///
/// ```
/// use brahe::spacetrack::operators::greater_than;
///
/// let filter = greater_than("2024-01-01");
/// assert_eq!(filter.to_url_segment(), ">2024-01-01");
/// ```
pub fn greater_than<T: Display>(value: T) -> QueryValue {
    QueryValue::GreaterThan(stringify_predicate_value(value))
}

/// Create a less-than filter (`<value`).
///
/// # Arguments
///
/// * `value` - The value to compare against
///
/// # Returns
///
/// A `QueryValue::LessThan` variant
///
/// # Example
///
/// ```
/// use brahe::spacetrack::operators::less_than;
///
/// let filter = less_than("2024-01-01");
/// assert_eq!(filter.to_url_segment(), "<2024-01-01");
/// ```
pub fn less_than<T: Display>(value: T) -> QueryValue {
    QueryValue::LessThan(stringify_predicate_value(value))
}

/// Create a not-equal filter (`<>value`).
///
/// # Arguments
///
/// * `value` - The value to compare against
///
/// # Returns
///
/// A `QueryValue::NotEqual` variant
///
/// # Example
///
/// ```
/// use brahe::spacetrack::operators::not_equal;
///
/// let filter = not_equal("US");
/// assert_eq!(filter.to_url_segment(), "<>US");
/// ```
pub fn not_equal<T: Display>(value: T) -> QueryValue {
    QueryValue::NotEqual(stringify_predicate_value(value))
}

/// Create an inclusive range filter (`left--right`).
///
/// # Arguments
///
/// * `left` - The lower bound of the range (inclusive)
/// * `right` - The upper bound of the range (inclusive)
///
/// # Returns
///
/// A `QueryValue::Range` variant
///
/// # Example
///
/// ```
/// use brahe::spacetrack::operators::inclusive_range;
///
/// let filter = inclusive_range("2024-01-01", "2024-12-31");
/// assert_eq!(filter.to_url_segment(), "2024-01-01--2024-12-31");
/// ```
pub fn inclusive_range<T: Display, U: Display>(left: T, right: U) -> QueryValue {
    QueryValue::Range(
        stringify_predicate_value(left),
        stringify_predicate_value(right),
    )
}

/// Create a LIKE pattern filter (`~~value`).
///
/// The pattern uses SQL LIKE syntax where:
/// - `%` matches any sequence of characters
/// - `_` matches any single character
///
/// # Arguments
///
/// * `pattern` - The LIKE pattern to match
///
/// # Returns
///
/// A `QueryValue::Like` variant
///
/// # Example
///
/// ```
/// use brahe::spacetrack::operators::like;
///
/// let filter = like("%STATION%");
/// assert_eq!(filter.to_url_segment(), "~~%STATION%");
/// ```
pub fn like<T: Display>(pattern: T) -> QueryValue {
    QueryValue::Like(stringify_predicate_value(pattern))
}

/// Create a starts-with filter (`^value`).
///
/// # Arguments
///
/// * `prefix` - The prefix to match
///
/// # Returns
///
/// A `QueryValue::StartsWith` variant
///
/// # Example
///
/// ```
/// use brahe::spacetrack::operators::startswith;
///
/// let filter = startswith("ISS");
/// assert_eq!(filter.to_url_segment(), "^ISS");
/// ```
pub fn startswith<T: Display>(prefix: T) -> QueryValue {
    QueryValue::StartsWith(stringify_predicate_value(prefix))
}

/// Create an exact value match filter.
///
/// # Arguments
///
/// * `value` - The exact value to match
///
/// # Returns
///
/// A `QueryValue::Value` variant
///
/// # Example
///
/// ```
/// use brahe::spacetrack::operators::equals;
///
/// let filter = equals(25544);
/// assert_eq!(filter.to_url_segment(), "25544");
/// ```
pub fn equals<T: Display>(value: T) -> QueryValue {
    QueryValue::Value(stringify_predicate_value(value))
}

/// Create a null value filter.
///
/// # Returns
///
/// A `QueryValue::Null` variant
///
/// # Example
///
/// ```
/// use brahe::spacetrack::operators::null_val;
///
/// let filter = null_val();
/// assert_eq!(filter.to_url_segment(), "null-val");
/// ```
pub fn null_val() -> QueryValue {
    QueryValue::Null
}

/// Convert a value to a SpaceTrack-compatible string.
///
/// This handles special cases like booleans which need to be lowercase.
pub fn stringify_predicate_value<T: Display>(value: T) -> String {
    // Handle boolean values (Rust Display for bool is "true"/"false" which is correct)
    // Handle other special cases as needed
    value.to_string()
}

/// Stringify a boolean value for SpaceTrack (lowercase "true" or "false").
pub fn stringify_bool(value: bool) -> String {
    if value {
        "true".to_string()
    } else {
        "false".to_string()
    }
}

/// Stringify a sequence of values as comma-separated.
pub fn stringify_sequence<T: Display>(values: &[T]) -> String {
    values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// Convert a `QueryValue` to an `Into<QueryValue>` implementation for common types.
impl From<i32> for QueryValue {
    fn from(v: i32) -> Self {
        QueryValue::Value(v.to_string())
    }
}

impl From<i64> for QueryValue {
    fn from(v: i64) -> Self {
        QueryValue::Value(v.to_string())
    }
}

impl From<u32> for QueryValue {
    fn from(v: u32) -> Self {
        QueryValue::Value(v.to_string())
    }
}

impl From<u64> for QueryValue {
    fn from(v: u64) -> Self {
        QueryValue::Value(v.to_string())
    }
}

impl From<f32> for QueryValue {
    fn from(v: f32) -> Self {
        QueryValue::Value(v.to_string())
    }
}

impl From<f64> for QueryValue {
    fn from(v: f64) -> Self {
        QueryValue::Value(v.to_string())
    }
}

impl From<&str> for QueryValue {
    fn from(v: &str) -> Self {
        QueryValue::Value(v.to_string())
    }
}

impl From<String> for QueryValue {
    fn from(v: String) -> Self {
        QueryValue::Value(v)
    }
}

impl From<bool> for QueryValue {
    fn from(v: bool) -> Self {
        QueryValue::Value(stringify_bool(v))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_greater_than() {
        assert_eq!(greater_than("test").to_url_segment(), ">test");
        assert_eq!(greater_than(100).to_url_segment(), ">100");
        assert_eq!(greater_than(1.5).to_url_segment(), ">1.5");
    }

    #[test]
    fn test_less_than() {
        assert_eq!(less_than("test").to_url_segment(), "<test");
        assert_eq!(less_than(100).to_url_segment(), "<100");
    }

    #[test]
    fn test_not_equal() {
        assert_eq!(not_equal("test").to_url_segment(), "<>test");
        assert_eq!(not_equal("US").to_url_segment(), "<>US");
    }

    #[test]
    fn test_inclusive_range() {
        assert_eq!(inclusive_range("a", "b").to_url_segment(), "a--b");
        assert_eq!(
            inclusive_range("2024-01-01", "2024-12-31").to_url_segment(),
            "2024-01-01--2024-12-31"
        );
        assert_eq!(inclusive_range(1, 10).to_url_segment(), "1--10");
    }

    #[test]
    fn test_like() {
        assert_eq!(like("test").to_url_segment(), "~~test");
        assert_eq!(like("%STATION%").to_url_segment(), "~~%STATION%");
    }

    #[test]
    fn test_startswith() {
        assert_eq!(startswith("test").to_url_segment(), "^test");
        assert_eq!(startswith("ISS").to_url_segment(), "^ISS");
    }

    #[test]
    fn test_equals() {
        assert_eq!(equals("test").to_url_segment(), "test");
        assert_eq!(equals(25544).to_url_segment(), "25544");
    }

    #[test]
    fn test_null_val() {
        assert_eq!(null_val().to_url_segment(), "null-val");
    }

    #[test]
    fn test_stringify_bool() {
        assert_eq!(stringify_bool(true), "true");
        assert_eq!(stringify_bool(false), "false");
    }

    #[test]
    fn test_stringify_sequence() {
        assert_eq!(stringify_sequence(&[1, 2, 3]), "1,2,3");
        assert_eq!(stringify_sequence(&["a", "b", "c"]), "a,b,c");
        assert_eq!(stringify_sequence(&[25544, 34602]), "25544,34602");
    }

    #[test]
    fn test_query_value_from_types() {
        let v: QueryValue = 42i32.into();
        assert_eq!(v.to_url_segment(), "42");

        let v: QueryValue = 42i64.into();
        assert_eq!(v.to_url_segment(), "42");

        let v: QueryValue = 42u32.into();
        assert_eq!(v.to_url_segment(), "42");

        let v: QueryValue = 42u64.into();
        assert_eq!(v.to_url_segment(), "42");

        let v: QueryValue = 1.23f32.into();
        assert!(v.to_url_segment().starts_with("1.23"));

        let v: QueryValue = 1.23f64.into();
        assert_eq!(v.to_url_segment(), "1.23");

        let v: QueryValue = "test".into();
        assert_eq!(v.to_url_segment(), "test");

        let v: QueryValue = "test".to_string().into();
        assert_eq!(v.to_url_segment(), "test");

        let v: QueryValue = true.into();
        assert_eq!(v.to_url_segment(), "true");

        let v: QueryValue = false.into();
        assert_eq!(v.to_url_segment(), "false");
    }

    #[test]
    fn test_query_value_display() {
        assert_eq!(format!("{}", greater_than("test")), ">test");
        assert_eq!(format!("{}", less_than("test")), "<test");
        assert_eq!(format!("{}", not_equal("test")), "<>test");
        assert_eq!(format!("{}", inclusive_range("a", "b")), "a--b");
        assert_eq!(format!("{}", like("test")), "~~test");
        assert_eq!(format!("{}", startswith("test")), "^test");
        assert_eq!(format!("{}", null_val()), "null-val");
    }
}
