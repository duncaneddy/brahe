/*!
 * Query operator functions for building filter values.
 *
 * These functions generate operator-prefixed strings for use in SpaceTrack
 * and Celestrak query filter values. SpaceTrack uses a URL-based query
 * language where operators are embedded in the value portion of field/value pairs.
 *
 * For example, filtering for NORAD_CAT_ID > 25544 would use:
 * `.filter("NORAD_CAT_ID", &greater_than("25544"))`
 *
 * Which produces the URL segment: `/NORAD_CAT_ID/>25544/`
 */

use std::fmt::Display;

/// Greater-than operator: produces `">value"`.
///
/// # Arguments
///
/// * `value` - The comparison value
///
/// # Returns
///
/// * `String` - Formatted operator string
///
/// # Examples
///
/// ```
/// use brahe::utils::operators;
///
/// assert_eq!(operators::greater_than("25544"), ">25544");
/// assert_eq!(operators::greater_than("0.5"), ">0.5");
/// ```
pub fn greater_than<T: Display>(value: T) -> String {
    format!(">{}", value)
}

/// Less-than operator: produces `"<value"`.
///
/// # Arguments
///
/// * `value` - The comparison value
///
/// # Returns
///
/// * `String` - Formatted operator string
///
/// # Examples
///
/// ```
/// use brahe::utils::operators;
///
/// assert_eq!(operators::less_than("0.01"), "<0.01");
/// ```
pub fn less_than<T: Display>(value: T) -> String {
    format!("<{}", value)
}

/// Not-equal operator: produces `"<>value"`.
///
/// # Arguments
///
/// * `value` - The comparison value
///
/// # Returns
///
/// * `String` - Formatted operator string
///
/// # Examples
///
/// ```
/// use brahe::utils::operators;
///
/// assert_eq!(operators::not_equal("DEBRIS"), "<>DEBRIS");
/// ```
pub fn not_equal<T: Display>(value: T) -> String {
    format!("<>{}", value)
}

/// Inclusive range operator: produces `"left--right"`.
///
/// Matches values where `left <= value <= right`.
///
/// # Arguments
///
/// * `left` - The lower bound (inclusive)
/// * `right` - The upper bound (inclusive)
///
/// # Returns
///
/// * `String` - Formatted operator string
///
/// # Examples
///
/// ```
/// use brahe::utils::operators;
///
/// assert_eq!(operators::inclusive_range("25544", "25600"), "25544--25600");
/// assert_eq!(
///     operators::inclusive_range("2024-01-01", "2024-12-31"),
///     "2024-01-01--2024-12-31"
/// );
/// ```
pub fn inclusive_range<T: Display, U: Display>(left: T, right: U) -> String {
    format!("{}--{}", left, right)
}

/// Like/contains operator: produces `"~~value"`.
///
/// Performs a case-insensitive substring match. The `%` character can be used
/// as a wildcard.
///
/// # Arguments
///
/// * `value` - The pattern to match
///
/// # Returns
///
/// * `String` - Formatted operator string
///
/// # Examples
///
/// ```
/// use brahe::utils::operators;
///
/// assert_eq!(operators::like("STARLINK"), "~~STARLINK");
/// ```
pub fn like<T: Display>(value: T) -> String {
    format!("~~{}", value)
}

/// Starts-with operator: produces `"^value"`.
///
/// Matches values that begin with the specified prefix.
///
/// # Arguments
///
/// * `value` - The prefix to match
///
/// # Returns
///
/// * `String` - Formatted operator string
///
/// # Examples
///
/// ```
/// use brahe::utils::operators;
///
/// assert_eq!(operators::startswith("NOAA"), "^NOAA");
/// ```
pub fn startswith<T: Display>(value: T) -> String {
    format!("^{}", value)
}

/// Current time reference: returns `"now"`.
///
/// Used in date/time filters to reference the current server time.
///
/// # Returns
///
/// * `String` - The string "now"
///
/// # Examples
///
/// ```
/// use brahe::utils::operators;
///
/// assert_eq!(operators::now(), "now");
/// ```
pub fn now() -> String {
    "now".to_string()
}

/// Time offset from now: produces `"now-N"` or `"now+N"`.
///
/// Creates a relative time reference offset by the specified number of days.
/// Negative values produce `"now-N"` (past), positive values produce `"now+N"` (future).
///
/// # Arguments
///
/// * `days` - Number of days offset (negative for past, positive for future)
///
/// # Returns
///
/// * `String` - Formatted time reference string
///
/// # Examples
///
/// ```
/// use brahe::utils::operators;
///
/// assert_eq!(operators::now_offset(-7), "now-7");
/// assert_eq!(operators::now_offset(14), "now+14");
/// assert_eq!(operators::now_offset(0), "now+0");
/// ```
pub fn now_offset(days: i32) -> String {
    if days < 0 {
        format!("now{}", days)
    } else {
        format!("now+{}", days)
    }
}

/// Null value reference: returns `"null-val"`.
///
/// Used to filter for records where a field is null/empty.
///
/// # Returns
///
/// * `String` - The string "null-val"
///
/// # Examples
///
/// ```
/// use brahe::utils::operators;
///
/// assert_eq!(operators::null_val(), "null-val");
/// ```
pub fn null_val() -> String {
    "null-val".to_string()
}

/// OR list operator: produces `"val1,val2,val3"`.
///
/// Matches records where the field equals any of the provided values.
///
/// # Arguments
///
/// * `values` - Slice of values to match against
///
/// # Returns
///
/// * `String` - Comma-separated value string
///
/// # Examples
///
/// ```
/// use brahe::utils::operators;
///
/// assert_eq!(operators::or_list(&["25544", "25545", "25546"]), "25544,25545,25546");
/// assert_eq!(operators::or_list(&["US", "PRC"]), "US,PRC");
/// ```
pub fn or_list<T: Display>(values: &[T]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_greater_than() {
        assert_eq!(greater_than("25544"), ">25544");
        assert_eq!(greater_than("0.5"), ">0.5");
        assert_eq!(greater_than(100), ">100");
    }

    #[test]
    fn test_less_than() {
        assert_eq!(less_than("0.01"), "<0.01");
        assert_eq!(less_than("2024-01-01"), "<2024-01-01");
        assert_eq!(less_than(50), "<50");
    }

    #[test]
    fn test_not_equal() {
        assert_eq!(not_equal("DEBRIS"), "<>DEBRIS");
        assert_eq!(not_equal("null-val"), "<>null-val");
    }

    #[test]
    fn test_inclusive_range() {
        assert_eq!(inclusive_range("25544", "25600"), "25544--25600");
        assert_eq!(
            inclusive_range("2024-01-01", "2024-12-31"),
            "2024-01-01--2024-12-31"
        );
        assert_eq!(inclusive_range(1, 100), "1--100");
    }

    #[test]
    fn test_like() {
        assert_eq!(like("STARLINK"), "~~STARLINK");
        assert_eq!(like("ISS%"), "~~ISS%");
    }

    #[test]
    fn test_startswith() {
        assert_eq!(startswith("NOAA"), "^NOAA");
        assert_eq!(startswith("2024"), "^2024");
    }

    #[test]
    fn test_now() {
        assert_eq!(now(), "now");
    }

    #[test]
    fn test_now_offset() {
        assert_eq!(now_offset(-7), "now-7");
        assert_eq!(now_offset(14), "now+14");
        assert_eq!(now_offset(0), "now+0");
        assert_eq!(now_offset(-1), "now-1");
        assert_eq!(now_offset(-30), "now-30");
    }

    #[test]
    fn test_null_val() {
        assert_eq!(null_val(), "null-val");
    }

    #[test]
    fn test_or_list() {
        assert_eq!(or_list(&["25544", "25545", "25546"]), "25544,25545,25546");
        assert_eq!(or_list(&["US", "PRC"]), "US,PRC");
        assert_eq!(or_list(&["single"]), "single");
        assert_eq!(or_list::<&str>(&[]), "");
    }

    #[test]
    fn test_or_list_with_integers() {
        assert_eq!(or_list(&[25544, 25545]), "25544,25545");
    }

    #[test]
    fn test_operator_composition() {
        // Operators can be composed in filter values
        let range = inclusive_range(now_offset(-7), now());
        assert_eq!(range, "now-7--now");
    }
}
