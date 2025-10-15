/*!
 * Generic parsing functions for TLE/3LE data formats.
 * These are not source-specific and can be used with data from any provider.
 */

use crate::utils::BraheError;

/// Parse 3LE format text into structured (name, line1, line2) tuples
///
/// # Arguments
/// * `text` - Raw 3LE format text with satellite name on first line, TLE on next two lines
///
/// # Returns
/// * `Result<Vec<(String, String, String)>, BraheError>` - Vector of (name, line1, line2) tuples
///
/// # Example
/// ```
/// use brahe::datasets::parsers::parse_3le_text;
///
/// let text = "ISS (ZARYA)\n\
///             1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997\n\
///             2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";
/// let result = parse_3le_text(text).unwrap();
/// assert_eq!(result.len(), 1);
/// assert_eq!(result[0].0, "ISS (ZARYA)");
/// ```
pub fn parse_3le_text(text: &str) -> Result<Vec<(String, String, String)>, BraheError> {
    let lines: Vec<&str> = text.lines().collect();
    let mut result = Vec::new();

    let mut i = 0;
    while i < lines.len() {
        // Skip empty lines
        if lines[i].trim().is_empty() {
            i += 1;
            continue;
        }

        // Need at least 3 lines for a complete 3LE entry
        if i + 2 >= lines.len() {
            break;
        }

        let name = lines[i].trim();
        let line1 = lines[i + 1].trim();
        let line2 = lines[i + 2].trim();

        // Validate that line1 and line2 are valid TLE lines
        if !line1.starts_with('1') || !line2.starts_with('2') {
            return Err(BraheError::Error(format!(
                "Invalid TLE format at line {}: expected lines starting with '1' and '2'",
                i + 1
            )));
        }

        // Validate line lengths
        if line1.len() != 69 || line2.len() != 69 {
            return Err(BraheError::Error(format!(
                "Invalid TLE line length at line {}: expected 69 characters",
                i + 1
            )));
        }

        result.push((name.to_string(), line1.to_string(), line2.to_string()));
        i += 3;
    }

    if result.is_empty() {
        return Err(BraheError::Error(
            "No valid 3LE entries found in input text".to_string(),
        ));
    }

    Ok(result)
}

/// Convert 3LE data to 2LE format by dropping satellite names
///
/// # Arguments
/// * `data` - Vector of (name, line1, line2) tuples
///
/// # Returns
/// * `Vec<(String, String)>` - Vector of (line1, line2) tuples without names
///
/// # Example
/// ```
/// use brahe::datasets::parsers::convert_3le_to_2le;
///
/// let data_3le = vec![
///     ("ISS (ZARYA)".to_string(),
///      "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997".to_string(),
///      "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003".to_string())
/// ];
/// let data_2le = convert_3le_to_2le(&data_3le);
/// assert_eq!(data_2le.len(), 1);
/// assert_eq!(data_2le[0].0, "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997");
/// ```
pub fn convert_3le_to_2le(data: &[(String, String, String)]) -> Vec<(String, String)> {
    data.iter()
        .map(|(_name, line1, line2)| (line1.clone(), line2.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_3le_text_single_entry() {
        let text = "ISS (ZARYA)\n\
                    1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997\n\
                    2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";

        let result = parse_3le_text(text).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, "ISS (ZARYA)");
        assert!(result[0].1.starts_with("1 25544"));
        assert!(result[0].2.starts_with("2 25544"));
    }

    #[test]
    fn test_parse_3le_text_multiple_entries() {
        let text = "ISS (ZARYA)\n\
                    1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997\n\
                    2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003\n\
                    STARLINK-1007\n\
                    1 44713U 19074A   21001.50000000  .00001764  00000-0  40967-4 0  9997\n\
                    2 44713  53.0000 100.0000 0001000  90.0000  10.0000 15.00000000000003";

        let result = parse_3le_text(text).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, "ISS (ZARYA)");
        assert_eq!(result[1].0, "STARLINK-1007");
    }

    #[test]
    fn test_parse_3le_text_with_empty_lines() {
        let text = "\n\
                    ISS (ZARYA)\n\
                    1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997\n\
                    2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003\n\
                    \n\
                    STARLINK-1007\n\
                    1 44713U 19074A   21001.50000000  .00001764  00000-0  40967-4 0  9997\n\
                    2 44713  53.0000 100.0000 0001000  90.0000  10.0000 15.00000000000003\n";

        let result = parse_3le_text(text).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_parse_3le_text_invalid_line_start() {
        let text = "ISS (ZARYA)\n\
                    2 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997\n\
                    2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";

        let result = parse_3le_text(text);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("expected lines starting with '1' and '2'")
        );
    }

    #[test]
    fn test_parse_3le_text_invalid_line_length() {
        let text = "ISS (ZARYA)\n\
                    1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  999\n\
                    2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003";

        let result = parse_3le_text(text);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid TLE line length")
        );
    }

    #[test]
    fn test_parse_3le_text_empty_input() {
        let text = "";
        let result = parse_3le_text(text);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No valid 3LE entries found")
        );
    }

    #[test]
    fn test_parse_3le_text_only_empty_lines() {
        let text = "\n\n\n";
        let result = parse_3le_text(text);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No valid 3LE entries found")
        );
    }

    #[test]
    fn test_convert_3le_to_2le_single() {
        let data_3le = vec![(
            "ISS (ZARYA)".to_string(),
            "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997".to_string(),
            "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003".to_string(),
        )];

        let data_2le = convert_3le_to_2le(&data_3le);
        assert_eq!(data_2le.len(), 1);
        assert_eq!(
            data_2le[0].0,
            "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997"
        );
        assert_eq!(
            data_2le[0].1,
            "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"
        );
    }

    #[test]
    fn test_convert_3le_to_2le_multiple() {
        let data_3le = vec![
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
        ];

        let data_2le = convert_3le_to_2le(&data_3le);
        assert_eq!(data_2le.len(), 2);
    }

    #[test]
    fn test_convert_3le_to_2le_empty() {
        let data_3le: Vec<(String, String, String)> = vec![];
        let data_2le = convert_3le_to_2le(&data_3le);
        assert_eq!(data_2le.len(), 0);
    }
}
