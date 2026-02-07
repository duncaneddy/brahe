/*!
 * Client-side filtering engine for CelestrakClient query results.
 *
 * Parses SpaceTrack-compatible operator strings and applies them
 * as filters to downloaded records. This enables SpaceTrack-style
 * filtering on CelestrakClient data where the API only supports
 * limited server-side filtering.
 *
 * Supported operators:
 * - `>value` - Greater than
 * - `<value` - Less than
 * - `<>value` - Not equal
 * - `min--max` - Inclusive range
 * - `~~pattern` - Case-insensitive substring match (like)
 * - `^prefix` - Case-insensitive prefix match (starts with)
 * - `value` - Exact string match
 */

use std::borrow::Cow;

use crate::celestrak::query::{Filter, OrderBy};
use crate::celestrak::responses::CelestrakSATCATRecord;
use crate::types::gp_record::FieldAccessor;

impl FieldAccessor for CelestrakSATCATRecord {
    fn get_field(&self, name: &str) -> Option<Cow<'_, str>> {
        match name {
            "OBJECT_NAME" => self.object_name.as_deref().map(Cow::Borrowed),
            "OBJECT_ID" => self.object_id.as_deref().map(Cow::Borrowed),
            "NORAD_CAT_ID" => self.norad_cat_id.map(|v| Cow::Owned(v.to_string())),
            "OBJECT_TYPE" => self.object_type.as_deref().map(Cow::Borrowed),
            "OPS_STATUS_CODE" => self.ops_status_code.as_deref().map(Cow::Borrowed),
            "OWNER" => self.owner.as_deref().map(Cow::Borrowed),
            "LAUNCH_DATE" => self.launch_date.as_deref().map(Cow::Borrowed),
            "LAUNCH_SITE" => self.launch_site.as_deref().map(Cow::Borrowed),
            "DECAY_DATE" => self.decay_date.as_deref().map(Cow::Borrowed),
            "PERIOD" => self.period.as_deref().map(Cow::Borrowed),
            "INCLINATION" => self.inclination.as_deref().map(Cow::Borrowed),
            "APOGEE" => self.apogee.as_deref().map(Cow::Borrowed),
            "PERIGEE" => self.perigee.as_deref().map(Cow::Borrowed),
            "RCS" => self.rcs.as_deref().map(Cow::Borrowed),
            "DATA_STATUS_CODE" => self.data_status_code.as_deref().map(Cow::Borrowed),
            "ORBIT_CENTER" => self.orbit_center.as_deref().map(Cow::Borrowed),
            "ORBIT_TYPE" => self.orbit_type.as_deref().map(Cow::Borrowed),
            _ => None,
        }
    }
}

/// Parsed filter operator with its value.
enum FilterOp<'a> {
    GreaterThan(&'a str),
    LessThan(&'a str),
    NotEqual(&'a str),
    Range(&'a str, &'a str),
    Like(&'a str),
    StartsWith(&'a str),
    Exact(&'a str),
}

/// Parse a filter value string into a FilterOp.
fn parse_filter_value(value: &str) -> FilterOp<'_> {
    // Check operators from most specific to least specific
    if let Some(rest) = value.strip_prefix("<>") {
        FilterOp::NotEqual(rest)
    } else if let Some(rest) = value.strip_prefix("~~") {
        FilterOp::Like(rest)
    } else if let Some(rest) = value.strip_prefix('>') {
        FilterOp::GreaterThan(rest)
    } else if let Some(rest) = value.strip_prefix('<') {
        FilterOp::LessThan(rest)
    } else if let Some(rest) = value.strip_prefix('^') {
        FilterOp::StartsWith(rest)
    } else if let Some(dash_pos) = value.find("--") {
        let min = &value[..dash_pos];
        let max = &value[dash_pos + 2..];
        FilterOp::Range(min, max)
    } else {
        FilterOp::Exact(value)
    }
}

/// Check if a single filter matches a record.
fn matches_filter<T: FieldAccessor>(record: &T, filter: &Filter) -> bool {
    let field_value = match record.get_field(&filter.field) {
        Some(v) => v,
        // Records with missing fields don't match any filter
        None => return false,
    };

    match parse_filter_value(&filter.value) {
        FilterOp::GreaterThan(threshold) => {
            compare_values(&field_value, threshold) == Some(std::cmp::Ordering::Greater)
        }
        FilterOp::LessThan(threshold) => {
            compare_values(&field_value, threshold) == Some(std::cmp::Ordering::Less)
        }
        FilterOp::NotEqual(other) => !field_value.eq_ignore_ascii_case(other),
        FilterOp::Range(min, max) => {
            let cmp_min = compare_values(&field_value, min);
            let cmp_max = compare_values(&field_value, max);
            matches!(
                (cmp_min, cmp_max),
                (
                    Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal),
                    Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
                )
            )
        }
        FilterOp::Like(pattern) => field_value
            .to_ascii_lowercase()
            .contains(&pattern.to_ascii_lowercase()),
        FilterOp::StartsWith(prefix) => field_value
            .to_ascii_lowercase()
            .starts_with(&prefix.to_ascii_lowercase()),
        FilterOp::Exact(expected) => *field_value == *expected,
    }
}

/// Compare two string values, attempting numeric comparison first.
fn compare_values(a: &str, b: &str) -> Option<std::cmp::Ordering> {
    // Try numeric comparison first
    if let (Ok(a_num), Ok(b_num)) = (a.parse::<f64>(), b.parse::<f64>()) {
        a_num.partial_cmp(&b_num)
    } else {
        // Fall back to lexicographic comparison
        Some(a.cmp(b))
    }
}

/// Apply client-side filters to a vector of records.
///
/// Records must match ALL filters (AND logic).
pub(crate) fn apply_filters<T: FieldAccessor>(records: Vec<T>, filters: &[Filter]) -> Vec<T> {
    if filters.is_empty() {
        return records;
    }
    records
        .into_iter()
        .filter(|record| filters.iter().all(|f| matches_filter(record, f)))
        .collect()
}

/// Apply client-side ordering to a vector of records.
///
/// Multiple ordering clauses are applied in order (primary sort first).
pub(crate) fn apply_order_by<T: FieldAccessor>(records: &mut [T], order_by: &[OrderBy]) {
    if order_by.is_empty() {
        return;
    }
    records.sort_by(|a, b| {
        for clause in order_by {
            let a_val = a.get_field(&clause.field);
            let b_val = b.get_field(&clause.field);
            let cmp = match (&a_val, &b_val) {
                (Some(av), Some(bv)) => compare_values(av, bv).unwrap_or(std::cmp::Ordering::Equal),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            };
            let cmp = if clause.ascending { cmp } else { cmp.reverse() };
            if cmp != std::cmp::Ordering::Equal {
                return cmp;
            }
        }
        std::cmp::Ordering::Equal
    });
}

/// Apply a client-side limit to truncate results.
pub(crate) fn apply_limit<T>(records: Vec<T>, limit: Option<u32>) -> Vec<T> {
    match limit {
        Some(n) => records.into_iter().take(n as usize).collect(),
        None => records,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::types::GPRecord;

    fn make_gp_record(
        name: &str,
        norad_id: &str,
        inclination: &str,
        eccentricity: &str,
        object_type: &str,
    ) -> GPRecord {
        let json = format!(
            r#"{{"OBJECT_NAME": "{}", "NORAD_CAT_ID": "{}", "INCLINATION": "{}", "ECCENTRICITY": "{}", "OBJECT_TYPE": "{}"}}"#,
            name, norad_id, inclination, eccentricity, object_type
        );
        serde_json::from_str(&json).unwrap()
    }

    fn make_satcat_record(
        name: &str,
        norad_id: &str,
        inclination: &str,
        owner: &str,
    ) -> CelestrakSATCATRecord {
        let json = format!(
            r#"{{"OBJECT_NAME": "{}", "NORAD_CAT_ID": "{}", "INCLINATION": "{}", "OWNER": "{}"}}"#,
            name, norad_id, inclination, owner
        );
        serde_json::from_str(&json).unwrap()
    }

    fn sample_gp_records() -> Vec<GPRecord> {
        vec![
            make_gp_record("ISS (ZARYA)", "25544", "51.64", "0.0001", "PAYLOAD"),
            make_gp_record("STARLINK-1234", "44713", "53.05", "0.0001", "PAYLOAD"),
            make_gp_record("COSMOS 2251 DEB", "33767", "74.03", "0.0200", "DEBRIS"),
            make_gp_record("NOAA 18", "28654", "98.70", "0.0014", "PAYLOAD"),
            make_gp_record("GPS BIIR-2", "24876", "55.00", "0.0050", "PAYLOAD"),
        ]
    }

    // -- Greater than tests --

    #[test]
    fn test_filter_greater_than_numeric() {
        let records = sample_gp_records();
        let filters = vec![Filter {
            field: "INCLINATION".to_string(),
            value: ">70".to_string(),
        }];
        let result = apply_filters(records, &filters);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].object_name.as_deref(), Some("COSMOS 2251 DEB"));
        assert_eq!(result[1].object_name.as_deref(), Some("NOAA 18"));
    }

    // -- Less than tests --

    #[test]
    fn test_filter_less_than_numeric() {
        let records = sample_gp_records();
        let filters = vec![Filter {
            field: "ECCENTRICITY".to_string(),
            value: "<0.001".to_string(),
        }];
        let result = apply_filters(records, &filters);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].object_name.as_deref(), Some("ISS (ZARYA)"));
        assert_eq!(result[1].object_name.as_deref(), Some("STARLINK-1234"));
    }

    // -- Not equal tests --

    #[test]
    fn test_filter_not_equal() {
        let records = sample_gp_records();
        let filters = vec![Filter {
            field: "OBJECT_TYPE".to_string(),
            value: "<>DEBRIS".to_string(),
        }];
        let result = apply_filters(records, &filters);
        assert_eq!(result.len(), 4);
        assert!(
            result
                .iter()
                .all(|r| r.object_type.as_deref() != Some("DEBRIS"))
        );
    }

    // -- Range tests --

    #[test]
    fn test_filter_range_inclusive() {
        let records = sample_gp_records();
        let filters = vec![Filter {
            field: "NORAD_CAT_ID".to_string(),
            value: "25000--30000".to_string(),
        }];
        let result = apply_filters(records, &filters);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].object_name.as_deref(), Some("ISS (ZARYA)"));
        assert_eq!(result[1].object_name.as_deref(), Some("NOAA 18"));
    }

    // -- Like tests --

    #[test]
    fn test_filter_like_case_insensitive() {
        let records = sample_gp_records();
        let filters = vec![Filter {
            field: "OBJECT_NAME".to_string(),
            value: "~~starlink".to_string(),
        }];
        let result = apply_filters(records, &filters);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].object_name.as_deref(), Some("STARLINK-1234"));
    }

    // -- Starts with tests --

    #[test]
    fn test_filter_starts_with() {
        let records = sample_gp_records();
        let filters = vec![Filter {
            field: "OBJECT_NAME".to_string(),
            value: "^NOAA".to_string(),
        }];
        let result = apply_filters(records, &filters);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].object_name.as_deref(), Some("NOAA 18"));
    }

    // -- Exact match tests --

    #[test]
    fn test_filter_exact_match() {
        let records = sample_gp_records();
        let filters = vec![Filter {
            field: "NORAD_CAT_ID".to_string(),
            value: "25544".to_string(),
        }];
        let result = apply_filters(records, &filters);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].object_name.as_deref(), Some("ISS (ZARYA)"));
    }

    // -- Multiple filters (AND logic) --

    #[test]
    fn test_multiple_filters_and_logic() {
        let records = sample_gp_records();
        let filters = vec![
            Filter {
                field: "OBJECT_TYPE".to_string(),
                value: "<>DEBRIS".to_string(),
            },
            Filter {
                field: "INCLINATION".to_string(),
                value: ">52".to_string(),
            },
        ];
        let result = apply_filters(records, &filters);
        assert_eq!(result.len(), 3);
    }

    // -- Missing field tests --

    #[test]
    fn test_filter_missing_field_excluded() {
        let records = sample_gp_records();
        let filters = vec![Filter {
            field: "EPOCH".to_string(),
            value: ">2024-01-01".to_string(),
        }];
        let result = apply_filters(records, &filters);
        assert!(result.is_empty());
    }

    // -- Empty filters --

    #[test]
    fn test_empty_filters_returns_all() {
        let records = sample_gp_records();
        let filters: Vec<Filter> = vec![];
        let result = apply_filters(records, &filters);
        assert_eq!(result.len(), 5);
    }

    // -- Ordering tests --

    #[test]
    fn test_order_by_ascending_numeric() {
        let mut records = sample_gp_records();
        let order_by = vec![OrderBy {
            field: "INCLINATION".to_string(),
            ascending: true,
        }];
        apply_order_by(&mut records, &order_by);
        assert_eq!(records[0].object_name.as_deref(), Some("ISS (ZARYA)"));
        assert_eq!(records[4].object_name.as_deref(), Some("NOAA 18"));
    }

    #[test]
    fn test_order_by_descending_numeric() {
        let mut records = sample_gp_records();
        let order_by = vec![OrderBy {
            field: "INCLINATION".to_string(),
            ascending: false,
        }];
        apply_order_by(&mut records, &order_by);
        assert_eq!(records[0].object_name.as_deref(), Some("NOAA 18"));
        assert_eq!(records[4].object_name.as_deref(), Some("ISS (ZARYA)"));
    }

    #[test]
    fn test_order_by_lexicographic() {
        let mut records = sample_gp_records();
        let order_by = vec![OrderBy {
            field: "OBJECT_NAME".to_string(),
            ascending: true,
        }];
        apply_order_by(&mut records, &order_by);
        assert_eq!(records[0].object_name.as_deref(), Some("COSMOS 2251 DEB"));
        assert_eq!(records[4].object_name.as_deref(), Some("STARLINK-1234"));
    }

    #[test]
    fn test_empty_order_by() {
        let mut records = sample_gp_records();
        let original_first = records[0].object_name.clone();
        let order_by: Vec<OrderBy> = vec![];
        apply_order_by(&mut records, &order_by);
        assert_eq!(records[0].object_name, original_first);
    }

    // -- Limit tests --

    #[test]
    fn test_limit_truncates() {
        let records = sample_gp_records();
        let result = apply_limit(records, Some(2));
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_limit_none_returns_all() {
        let records = sample_gp_records();
        let result = apply_limit(records, None);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_limit_larger_than_records() {
        let records = sample_gp_records();
        let result = apply_limit(records, Some(100));
        assert_eq!(result.len(), 5);
    }

    // -- SATCAT record filtering --

    #[test]
    fn test_satcat_filter() {
        let records = vec![
            make_satcat_record("ISS (ZARYA)", "25544", "51.64", "ISS"),
            make_satcat_record("COSMOS 2251", "22675", "74.03", "CIS"),
            make_satcat_record("NOAA 18", "28654", "98.70", "US"),
        ];
        let filters = vec![Filter {
            field: "OWNER".to_string(),
            value: "US".to_string(),
        }];
        let result = apply_filters(records, &filters);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].object_name.as_deref(), Some("NOAA 18"));
    }

    #[test]
    fn test_satcat_order_by() {
        let mut records = vec![
            make_satcat_record("NOAA 18", "28654", "98.70", "US"),
            make_satcat_record("ISS (ZARYA)", "25544", "51.64", "ISS"),
            make_satcat_record("COSMOS 2251", "22675", "74.03", "CIS"),
        ];
        let order_by = vec![OrderBy {
            field: "NORAD_CAT_ID".to_string(),
            ascending: true,
        }];
        apply_order_by(&mut records, &order_by);
        assert_eq!(records[0].norad_cat_id, Some(22675));
        assert_eq!(records[1].norad_cat_id, Some(25544));
        assert_eq!(records[2].norad_cat_id, Some(28654));
    }

    // -- Edge cases --

    #[test]
    fn test_filter_not_equal_case_insensitive() {
        let records = sample_gp_records();
        let filters = vec![Filter {
            field: "OBJECT_TYPE".to_string(),
            value: "<>debris".to_string(),
        }];
        let result = apply_filters(records, &filters);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_filter_unknown_field() {
        let records = sample_gp_records();
        let filters = vec![Filter {
            field: "NONEXISTENT".to_string(),
            value: "test".to_string(),
        }];
        let result = apply_filters(records, &filters);
        assert!(result.is_empty());
    }

    #[test]
    fn test_combined_filter_order_limit() {
        let mut records = sample_gp_records();
        let filters = vec![Filter {
            field: "OBJECT_TYPE".to_string(),
            value: "<>DEBRIS".to_string(),
        }];
        records = apply_filters(records, &filters);
        assert_eq!(records.len(), 4);

        let order_by = vec![OrderBy {
            field: "INCLINATION".to_string(),
            ascending: false,
        }];
        apply_order_by(&mut records, &order_by);

        records = apply_limit(records, Some(2));
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].object_name.as_deref(), Some("NOAA 18"));
        assert_eq!(records[1].object_name.as_deref(), Some("GPS BIIR-2"));
    }
}
