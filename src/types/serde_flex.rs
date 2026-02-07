/*!
 * Flexible serde deserializer modules for handling mixed JSON value types.
 *
 * SpaceTrack returns all values as JSON strings (e.g., `"15.48"`), while
 * Celestrak returns numeric values as JSON numbers (e.g., `15.48`). These
 * deserializer modules handle both representations transparently.
 *
 * Each module provides a `deserialize` function suitable for use with
 * `#[serde(deserialize_with = "...")]`.
 */

/// Deserializes an `Option<f64>` from a JSON string, number, or null.
///
/// Accepts:
/// - JSON number: `15.48` → `Some(15.48)`
/// - JSON string: `"15.48"` → `Some(15.48)` (parsed)
/// - JSON null or missing: → `None`
pub mod flex_f64 {
    use serde::{self, Deserialize, Deserializer};

    /// Deserialize an `Option<f64>` from a JSON string, number, or null.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: Option<serde_json::Value> = Option::deserialize(deserializer)?;
        match value {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(serde_json::Value::Number(n)) => Ok(n.as_f64()),
            Some(serde_json::Value::String(s)) => {
                if s.is_empty() {
                    Ok(None)
                } else {
                    s.parse::<f64>().map(Some).map_err(serde::de::Error::custom)
                }
            }
            Some(v) => Err(serde::de::Error::custom(format!(
                "expected number or string, got {}",
                v
            ))),
        }
    }
}

/// Deserializes an `Option<u8>` from a JSON string, number, or null.
pub mod flex_u8 {
    use serde::{self, Deserialize, Deserializer};

    /// Deserialize an `Option<u8>` from a JSON string, number, or null.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<u8>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: Option<serde_json::Value> = Option::deserialize(deserializer)?;
        match value {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(serde_json::Value::Number(n)) => n
                .as_u64()
                .and_then(|v| u8::try_from(v).ok())
                .map(Some)
                .ok_or_else(|| serde::de::Error::custom("number out of range for u8")),
            Some(serde_json::Value::String(s)) => {
                if s.is_empty() {
                    Ok(None)
                } else {
                    s.parse::<u8>().map(Some).map_err(serde::de::Error::custom)
                }
            }
            Some(v) => Err(serde::de::Error::custom(format!(
                "expected number or string, got {}",
                v
            ))),
        }
    }
}

/// Deserializes an `Option<u16>` from a JSON string, number, or null.
pub mod flex_u16 {
    use serde::{self, Deserialize, Deserializer};

    /// Deserialize an `Option<u16>` from a JSON string, number, or null.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<u16>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: Option<serde_json::Value> = Option::deserialize(deserializer)?;
        match value {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(serde_json::Value::Number(n)) => n
                .as_u64()
                .and_then(|v| u16::try_from(v).ok())
                .map(Some)
                .ok_or_else(|| serde::de::Error::custom("number out of range for u16")),
            Some(serde_json::Value::String(s)) => {
                if s.is_empty() {
                    Ok(None)
                } else {
                    s.parse::<u16>().map(Some).map_err(serde::de::Error::custom)
                }
            }
            Some(v) => Err(serde::de::Error::custom(format!(
                "expected number or string, got {}",
                v
            ))),
        }
    }
}

/// Deserializes an `Option<u32>` from a JSON string, number, or null.
pub mod flex_u32 {
    use serde::{self, Deserialize, Deserializer};

    /// Deserialize an `Option<u32>` from a JSON string, number, or null.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<u32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: Option<serde_json::Value> = Option::deserialize(deserializer)?;
        match value {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(serde_json::Value::Number(n)) => n
                .as_u64()
                .and_then(|v| u32::try_from(v).ok())
                .map(Some)
                .ok_or_else(|| serde::de::Error::custom("number out of range for u32")),
            Some(serde_json::Value::String(s)) => {
                if s.is_empty() {
                    Ok(None)
                } else {
                    s.parse::<u32>().map(Some).map_err(serde::de::Error::custom)
                }
            }
            Some(v) => Err(serde::de::Error::custom(format!(
                "expected number or string, got {}",
                v
            ))),
        }
    }
}

/// Deserializes an `Option<u64>` from a JSON string, number, or null.
pub mod flex_u64 {
    use serde::{self, Deserialize, Deserializer};

    /// Deserialize an `Option<u64>` from a JSON string, number, or null.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<u64>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: Option<serde_json::Value> = Option::deserialize(deserializer)?;
        match value {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(serde_json::Value::Number(n)) => n
                .as_u64()
                .map(Some)
                .ok_or_else(|| serde::de::Error::custom("number out of range for u64")),
            Some(serde_json::Value::String(s)) => {
                if s.is_empty() {
                    Ok(None)
                } else {
                    s.parse::<u64>().map(Some).map_err(serde::de::Error::custom)
                }
            }
            Some(v) => Err(serde::de::Error::custom(format!(
                "expected number or string, got {}",
                v
            ))),
        }
    }
}

/// Deserializes an `Option<String>` from any JSON value.
///
/// Converts numbers and booleans to their string representation.
/// This replaces the previous `string_or_any` module.
pub mod flex_string {
    use serde::{self, Deserialize, Deserializer};

    /// Deserialize an `Option<String>` from any JSON value.
    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value: Option<serde_json::Value> = Option::deserialize(deserializer)?;
        match value {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(serde_json::Value::String(s)) => Ok(Some(s)),
            Some(v) => Ok(Some(v.to_string())),
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use serde::Deserialize;

    // Test struct using flex_f64
    #[derive(Deserialize)]
    struct TestF64 {
        #[serde(default, deserialize_with = "super::flex_f64::deserialize")]
        value: Option<f64>,
    }

    // Test struct using flex_u32
    #[derive(Deserialize)]
    struct TestU32 {
        #[serde(default, deserialize_with = "super::flex_u32::deserialize")]
        value: Option<u32>,
    }

    // Test struct using flex_u8
    #[derive(Deserialize)]
    struct TestU8 {
        #[serde(default, deserialize_with = "super::flex_u8::deserialize")]
        value: Option<u8>,
    }

    // Test struct using flex_u16
    #[derive(Deserialize)]
    struct TestU16 {
        #[serde(default, deserialize_with = "super::flex_u16::deserialize")]
        value: Option<u16>,
    }

    // Test struct using flex_u64
    #[derive(Deserialize)]
    struct TestU64 {
        #[serde(default, deserialize_with = "super::flex_u64::deserialize")]
        value: Option<u64>,
    }

    // Test struct using flex_string
    #[derive(Deserialize)]
    struct TestString {
        #[serde(default, deserialize_with = "super::flex_string::deserialize")]
        value: Option<String>,
    }

    // -- flex_f64 tests --

    #[test]
    fn test_flex_f64_from_number() {
        let t: TestF64 = serde_json::from_str(r#"{"value": 15.48}"#).unwrap();
        assert_eq!(t.value, Some(15.48));
    }

    #[test]
    fn test_flex_f64_from_string() {
        let t: TestF64 = serde_json::from_str(r#"{"value": "15.48"}"#).unwrap();
        assert_eq!(t.value, Some(15.48));
    }

    #[test]
    fn test_flex_f64_from_null() {
        let t: TestF64 = serde_json::from_str(r#"{"value": null}"#).unwrap();
        assert!(t.value.is_none());
    }

    #[test]
    fn test_flex_f64_from_missing() {
        let t: TestF64 = serde_json::from_str(r#"{}"#).unwrap();
        assert!(t.value.is_none());
    }

    #[test]
    fn test_flex_f64_from_zero() {
        let t: TestF64 = serde_json::from_str(r#"{"value": 0}"#).unwrap();
        assert_eq!(t.value, Some(0.0));
    }

    #[test]
    fn test_flex_f64_from_string_zero() {
        let t: TestF64 = serde_json::from_str(r#"{"value": "0"}"#).unwrap();
        assert_eq!(t.value, Some(0.0));
    }

    #[test]
    fn test_flex_f64_from_empty_string() {
        let t: TestF64 = serde_json::from_str(r#"{"value": ""}"#).unwrap();
        assert!(t.value.is_none());
    }

    // -- flex_u32 tests --

    #[test]
    fn test_flex_u32_from_number() {
        let t: TestU32 = serde_json::from_str(r#"{"value": 25544}"#).unwrap();
        assert_eq!(t.value, Some(25544));
    }

    #[test]
    fn test_flex_u32_from_string() {
        let t: TestU32 = serde_json::from_str(r#"{"value": "25544"}"#).unwrap();
        assert_eq!(t.value, Some(25544));
    }

    #[test]
    fn test_flex_u32_from_null() {
        let t: TestU32 = serde_json::from_str(r#"{"value": null}"#).unwrap();
        assert!(t.value.is_none());
    }

    #[test]
    fn test_flex_u32_from_missing() {
        let t: TestU32 = serde_json::from_str(r#"{}"#).unwrap();
        assert!(t.value.is_none());
    }

    #[test]
    fn test_flex_u32_from_empty_string() {
        let t: TestU32 = serde_json::from_str(r#"{"value": ""}"#).unwrap();
        assert!(t.value.is_none());
    }

    // -- flex_u8 tests --

    #[test]
    fn test_flex_u8_from_number() {
        let t: TestU8 = serde_json::from_str(r#"{"value": 0}"#).unwrap();
        assert_eq!(t.value, Some(0));
    }

    #[test]
    fn test_flex_u8_from_string() {
        let t: TestU8 = serde_json::from_str(r#"{"value": "2"}"#).unwrap();
        assert_eq!(t.value, Some(2));
    }

    // -- flex_u16 tests --

    #[test]
    fn test_flex_u16_from_number() {
        let t: TestU16 = serde_json::from_str(r#"{"value": 999}"#).unwrap();
        assert_eq!(t.value, Some(999));
    }

    #[test]
    fn test_flex_u16_from_string() {
        let t: TestU16 = serde_json::from_str(r#"{"value": "999"}"#).unwrap();
        assert_eq!(t.value, Some(999));
    }

    // -- flex_u64 tests --

    #[test]
    fn test_flex_u64_from_number() {
        let t: TestU64 = serde_json::from_str(r#"{"value": 1234567890}"#).unwrap();
        assert_eq!(t.value, Some(1234567890));
    }

    #[test]
    fn test_flex_u64_from_string() {
        let t: TestU64 = serde_json::from_str(r#"{"value": "1234567890"}"#).unwrap();
        assert_eq!(t.value, Some(1234567890));
    }

    // -- flex_string tests --

    #[test]
    fn test_flex_string_from_string() {
        let t: TestString = serde_json::from_str(r#"{"value": "hello"}"#).unwrap();
        assert_eq!(t.value.as_deref(), Some("hello"));
    }

    #[test]
    fn test_flex_string_from_number() {
        let t: TestString = serde_json::from_str(r#"{"value": 25544}"#).unwrap();
        assert_eq!(t.value.as_deref(), Some("25544"));
    }

    #[test]
    fn test_flex_string_from_float() {
        let t: TestString = serde_json::from_str(r#"{"value": 15.48}"#).unwrap();
        assert_eq!(t.value.as_deref(), Some("15.48"));
    }

    #[test]
    fn test_flex_string_from_null() {
        let t: TestString = serde_json::from_str(r#"{"value": null}"#).unwrap();
        assert!(t.value.is_none());
    }

    #[test]
    fn test_flex_string_from_missing() {
        let t: TestString = serde_json::from_str(r#"{}"#).unwrap();
        assert!(t.value.is_none());
    }
}
