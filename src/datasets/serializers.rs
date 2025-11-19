/*!
 * Generic serialization functions for TLE/3LE data to various file formats.
 * These are not source-specific and can be used with data from any provider.
 */

use serde_json::json;

/// Serialize 3LE data to plain text format
///
/// # Arguments
/// * `data` - Vector of (name, line1, line2) tuples
/// * `include_names` - If true, output 3LE format with names; if false, output 2LE format
///
/// # Returns
/// * `String` - Plain text formatted output
///
/// # Example
/// ```
/// use brahe::datasets::serializers::serialize_3le_to_txt;
///
/// let data = vec![
///     ("ISS (ZARYA)".to_string(),
///      "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997".to_string(),
///      "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003".to_string())
/// ];
/// let txt = serialize_3le_to_txt(&data, true);
/// assert!(txt.contains("ISS (ZARYA)"));
/// ```
pub fn serialize_3le_to_txt(data: &[(String, String, String)], include_names: bool) -> String {
    let mut lines = Vec::new();

    for (name, line1, line2) in data {
        if include_names {
            lines.push(name.clone());
        }
        lines.push(line1.clone());
        lines.push(line2.clone());
    }

    lines.join("\n")
}

/// Serialize 3LE data to CSV format
///
/// # Arguments
/// * `data` - Vector of (name, line1, line2) tuples
/// * `include_names` - If true, include name column; if false, only line1 and line2
///
/// # Returns
/// * `String` - CSV formatted output with headers
///
/// # Example
/// ```
/// use brahe::datasets::serializers::serialize_3le_to_csv;
///
/// let data = vec![
///     ("ISS (ZARYA)".to_string(),
///      "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997".to_string(),
///      "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003".to_string())
/// ];
/// let csv = serialize_3le_to_csv(&data, true);
/// assert!(csv.starts_with("name,line1,line2"));
/// ```
pub fn serialize_3le_to_csv(data: &[(String, String, String)], include_names: bool) -> String {
    let mut lines = Vec::new();

    // Add header
    if include_names {
        lines.push("name,line1,line2".to_string());
    } else {
        lines.push("line1,line2".to_string());
    }

    // Add data rows
    for (name, line1, line2) in data {
        // Escape quotes in CSV fields by doubling them and wrapping in quotes if needed
        let escaped_name = escape_csv_field(name);
        let escaped_line1 = escape_csv_field(line1);
        let escaped_line2 = escape_csv_field(line2);

        if include_names {
            lines.push(format!(
                "{},{},{}",
                escaped_name, escaped_line1, escaped_line2
            ));
        } else {
            lines.push(format!("{},{}", escaped_line1, escaped_line2));
        }
    }

    lines.join("\n")
}

/// Serialize 3LE data to JSON format
///
/// # Arguments
/// * `data` - Vector of (name, line1, line2) tuples
/// * `include_names` - If true, include name field; if false, only line1 and line2
///
/// # Returns
/// * `String` - JSON formatted output as array of objects
///
/// # Example
/// ```
/// use brahe::datasets::serializers::serialize_3le_to_json;
///
/// let data = vec![
///     ("ISS (ZARYA)".to_string(),
///      "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997".to_string(),
///      "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003".to_string())
/// ];
/// let json_str = serialize_3le_to_json(&data, true);
/// assert!(json_str.contains("\"name\""));
/// assert!(json_str.contains("ISS (ZARYA)"));
/// ```
pub fn serialize_3le_to_json(data: &[(String, String, String)], include_names: bool) -> String {
    let entries: Vec<serde_json::Value> = data
        .iter()
        .map(|(name, line1, line2)| {
            if include_names {
                json!({
                    "name": name,
                    "line1": line1,
                    "line2": line2
                })
            } else {
                json!({
                    "line1": line1,
                    "line2": line2
                })
            }
        })
        .collect();

    serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string())
}

/// Helper function to escape CSV fields that contain special characters
fn escape_csv_field(field: &str) -> String {
    // If field contains comma, quote, or newline, wrap in quotes and escape existing quotes
    if field.contains(',') || field.contains('"') || field.contains('\n') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use rstest::rstest;

    fn get_test_data() -> Vec<(String, String, String)> {
        vec![
            (
                "ISS (ZARYA)".to_string(),
                "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997".to_string(),
                "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003".to_string(),
            ),
            (
                "STARLINK-1007".to_string(),
                "1 44713U 19074A   21001.50000000  .00001764  00000-0  40967-4 0  9997".to_string(),
                "2 44713  53.0000 100.0000 0001000  90.0000  10.0000 15.00000000000003".to_string(),
            ),
        ]
    }

    #[test]
    fn test_serialize_to_txt_with_names() {
        let data = get_test_data();
        let txt = serialize_3le_to_txt(&data, true);

        // Should contain satellite names
        assert!(txt.contains("ISS (ZARYA)"));
        assert!(txt.contains("STARLINK-1007"));

        // Should contain TLE lines
        assert!(txt.contains("1 25544"));
        assert!(txt.contains("2 25544"));
        assert!(txt.contains("1 44713"));
        assert!(txt.contains("2 44713"));

        // Count newlines - should have 6 lines (name + 2 TLE lines for each of 2 satellites)
        let line_count = txt.lines().count();
        assert_eq!(line_count, 6);
    }

    #[test]
    fn test_serialize_to_txt_without_names() {
        let data = get_test_data();
        let txt = serialize_3le_to_txt(&data, false);

        // Should NOT contain satellite names
        assert!(!txt.contains("ISS (ZARYA)"));
        assert!(!txt.contains("STARLINK-1007"));

        // Should contain TLE lines
        assert!(txt.contains("1 25544"));
        assert!(txt.contains("2 25544"));

        // Count newlines - should have 4 lines (2 TLE lines for each of 2 satellites)
        let line_count = txt.lines().count();
        assert_eq!(line_count, 4);
    }

    #[test]
    fn test_serialize_to_txt_empty() {
        let data: Vec<(String, String, String)> = vec![];
        let txt = serialize_3le_to_txt(&data, true);
        assert_eq!(txt, "");
    }

    #[test]
    fn test_serialize_to_csv_with_names() {
        let data = get_test_data();
        let csv = serialize_3le_to_csv(&data, true);

        // Should have header with name column
        assert!(csv.starts_with("name,line1,line2"));

        // Should contain satellite names
        assert!(csv.contains("ISS (ZARYA)"));
        assert!(csv.contains("STARLINK-1007"));

        // Should contain TLE lines
        assert!(csv.contains("1 25544"));
        assert!(csv.contains("2 25544"));

        // Count lines - should have 3 (header + 2 data rows)
        let line_count = csv.lines().count();
        assert_eq!(line_count, 3);
    }

    #[test]
    fn test_serialize_to_csv_without_names() {
        let data = get_test_data();
        let csv = serialize_3le_to_csv(&data, false);

        // Should have header without name column
        assert!(csv.starts_with("line1,line2"));

        // Should NOT contain satellite names
        assert!(!csv.contains("ISS (ZARYA)"));
        assert!(!csv.contains("STARLINK-1007"));

        // Should contain TLE lines
        assert!(csv.contains("1 25544"));
        assert!(csv.contains("2 25544"));

        // Count lines - should have 3 (header + 2 data rows)
        let line_count = csv.lines().count();
        assert_eq!(line_count, 3);
    }

    #[test]
    fn test_serialize_to_csv_empty() {
        let data: Vec<(String, String, String)> = vec![];
        let csv = serialize_3le_to_csv(&data, true);
        assert_eq!(csv, "name,line1,line2");
    }

    #[test]
    fn test_serialize_to_json_with_names() {
        let data = vec![(
            "ISS (ZARYA)".to_string(),
            "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997".to_string(),
            "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003".to_string(),
        )];

        let json_str = serialize_3le_to_json(&data, true);

        // Parse to verify valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert!(parsed.is_array());

        // Check structure
        let array = parsed.as_array().unwrap();
        assert_eq!(array.len(), 1);

        let obj = &array[0];
        assert_eq!(obj["name"], "ISS (ZARYA)");
        assert_eq!(
            obj["line1"],
            "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997"
        );
        assert_eq!(
            obj["line2"],
            "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"
        );
    }

    #[test]
    fn test_serialize_to_json_without_names() {
        let data = vec![(
            "ISS (ZARYA)".to_string(),
            "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997".to_string(),
            "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003".to_string(),
        )];

        let json_str = serialize_3le_to_json(&data, false);

        // Parse to verify valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert!(parsed.is_array());

        // Check structure - should NOT have name field
        let array = parsed.as_array().unwrap();
        let obj = &array[0];
        assert!(obj.get("name").is_none());
        assert!(obj.get("line1").is_some());
        assert!(obj.get("line2").is_some());
    }

    #[test]
    fn test_serialize_to_json_empty() {
        let data: Vec<(String, String, String)> = vec![];
        let json_str = serialize_3le_to_json(&data, true);

        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert!(parsed.is_array());
        assert_eq!(parsed.as_array().unwrap().len(), 0);
    }

    #[rstest]
    #[case("simple", "simple")]
    #[case("with space", "with space")]
    #[case("with,comma", "\"with,comma\"")]
    #[case("with\"quote", "\"with\"\"quote\"")]
    #[case("with\nnewline", "\"with\nnewline\"")]
    #[case("ISS (ZARYA)", "ISS (ZARYA)")]
    fn test_escape_csv_field(#[case] input: &str, #[case] expected: &str) {
        assert_eq!(escape_csv_field(input), expected);
    }

    #[test]
    fn test_csv_with_special_characters_in_name() {
        let data = vec![(
            "Satellite, with \"quotes\" and, commas".to_string(),
            "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997".to_string(),
            "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003".to_string(),
        )];

        let csv = serialize_3le_to_csv(&data, true);

        // Should properly escape the name field
        assert!(csv.contains("\"Satellite, with \"\"quotes\"\" and, commas\""));
    }
}
