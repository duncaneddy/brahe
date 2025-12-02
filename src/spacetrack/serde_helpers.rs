/*!
 * Serde deserialization helpers for SpaceTrack API responses.
 *
 * The SpaceTrack API returns numeric values as JSON strings, which requires
 * custom deserializers to parse them into the appropriate Rust types.
 */

use serde::{Deserialize, Deserializer};

/// Deserialize a string as an optional f64.
///
/// Handles cases where the API returns:
/// - A string like "15.49206887" -> Some(15.49206887)
/// - An empty string -> None
/// - A null value -> None
/// - An actual number -> Some(number)
pub fn deserialize_optional_f64<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrNumber {
        String(String),
        Number(f64),
        Null,
    }

    match StringOrNumber::deserialize(deserializer)? {
        StringOrNumber::String(s) if s.is_empty() => Ok(None),
        StringOrNumber::String(s) => s
            .parse()
            .map(Some)
            .map_err(|_| serde::de::Error::custom(format!("invalid f64 string: {}", s))),
        StringOrNumber::Number(n) => Ok(Some(n)),
        StringOrNumber::Null => Ok(None),
    }
}

/// Deserialize a string as an optional u32.
///
/// Handles cases where the API returns:
/// - A string like "25544" -> Some(25544)
/// - An empty string -> None
/// - A null value -> None
/// - An actual number -> Some(number)
pub fn deserialize_optional_u32<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrNumber {
        String(String),
        Number(u32),
        Null,
    }

    match StringOrNumber::deserialize(deserializer)? {
        StringOrNumber::String(s) if s.is_empty() => Ok(None),
        StringOrNumber::String(s) => s
            .parse()
            .map(Some)
            .map_err(|_| serde::de::Error::custom(format!("invalid u32 string: {}", s))),
        StringOrNumber::Number(n) => Ok(Some(n)),
        StringOrNumber::Null => Ok(None),
    }
}

/// Deserialize a string as an optional u64.
///
/// Handles cases where the API returns:
/// - A string like "1234567890" -> Some(1234567890)
/// - An empty string -> None
/// - A null value -> None
/// - An actual number -> Some(number)
pub fn deserialize_optional_u64<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrNumber {
        String(String),
        Number(u64),
        Null,
    }

    match StringOrNumber::deserialize(deserializer)? {
        StringOrNumber::String(s) if s.is_empty() => Ok(None),
        StringOrNumber::String(s) => s
            .parse()
            .map(Some)
            .map_err(|_| serde::de::Error::custom(format!("invalid u64 string: {}", s))),
        StringOrNumber::Number(n) => Ok(Some(n)),
        StringOrNumber::Null => Ok(None),
    }
}

/// Deserialize a string as an optional i32.
///
/// Handles cases where the API returns:
/// - A string like "0" -> Some(0)
/// - An empty string -> None
/// - A null value -> None
/// - An actual number -> Some(number)
pub fn deserialize_optional_i32<'de, D>(deserializer: D) -> Result<Option<i32>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrNumber {
        String(String),
        Number(i32),
        Null,
    }

    match StringOrNumber::deserialize(deserializer)? {
        StringOrNumber::String(s) if s.is_empty() => Ok(None),
        StringOrNumber::String(s) => s
            .parse()
            .map(Some)
            .map_err(|_| serde::de::Error::custom(format!("invalid i32 string: {}", s))),
        StringOrNumber::Number(n) => Ok(Some(n)),
        StringOrNumber::Null => Ok(None),
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct TestF64 {
        #[serde(deserialize_with = "deserialize_optional_f64")]
        value: Option<f64>,
    }

    #[derive(Debug, Deserialize)]
    struct TestU32 {
        #[serde(deserialize_with = "deserialize_optional_u32")]
        value: Option<u32>,
    }

    #[test]
    fn test_deserialize_f64_from_string() {
        let json = r#"{"value": "15.49206887"}"#;
        let parsed: TestF64 = serde_json::from_str(json).unwrap();
        assert!((parsed.value.unwrap() - 15.49206887).abs() < 1e-10);
    }

    #[test]
    fn test_deserialize_f64_from_number() {
        let json = r#"{"value": 15.49206887}"#;
        let parsed: TestF64 = serde_json::from_str(json).unwrap();
        assert!((parsed.value.unwrap() - 15.49206887).abs() < 1e-10);
    }

    #[test]
    fn test_deserialize_f64_from_null() {
        let json = r#"{"value": null}"#;
        let parsed: TestF64 = serde_json::from_str(json).unwrap();
        assert!(parsed.value.is_none());
    }

    #[test]
    fn test_deserialize_f64_from_empty_string() {
        let json = r#"{"value": ""}"#;
        let parsed: TestF64 = serde_json::from_str(json).unwrap();
        assert!(parsed.value.is_none());
    }

    #[test]
    fn test_deserialize_u32_from_string() {
        let json = r#"{"value": "25544"}"#;
        let parsed: TestU32 = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.value, Some(25544));
    }

    #[test]
    fn test_deserialize_u32_from_number() {
        let json = r#"{"value": 25544}"#;
        let parsed: TestU32 = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.value, Some(25544));
    }
}
