/*!
 * String-backed object identity for frames scoped to a specific object.
 */

use std::fmt;
use std::sync::Arc;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// String-backed object identity (e.g. `"LRO"`, `"2024-123A"`).
///
/// Cheap to clone (`Arc<str>` internally). Serializes as a plain JSON
/// string.
///
/// # Examples
///
/// ```rust
/// use brahe::frames::ObjectId;
///
/// let id: ObjectId = "LRO".into();
/// assert_eq!(id.to_string(), "LRO");
/// assert_eq!(id, ObjectId::from("LRO".to_string()));
/// ```
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct ObjectId(Arc<str>);

impl fmt::Display for ObjectId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for ObjectId {
    fn from(value: &str) -> Self {
        ObjectId(Arc::from(value))
    }
}

impl From<String> for ObjectId {
    fn from(value: String) -> Self {
        ObjectId(Arc::from(value.as_str()))
    }
}

impl Serialize for ObjectId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

struct ObjectIdVisitor;

impl Visitor<'_> for ObjectIdVisitor {
    type Value = ObjectId;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a string object identifier")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(ObjectId::from(value))
    }
}

impl<'de> Deserialize<'de> for ObjectId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_str(ObjectIdVisitor)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use serial_test::parallel;

    use super::*;

    #[test]
    #[parallel]
    fn test_object_id_from_string() {
        let id = ObjectId::from(String::from("ISS"));
        assert_eq!(id.to_string(), "ISS");
        assert_eq!(id, ObjectId::from("ISS"));
    }

    #[test]
    #[parallel]
    fn test_object_id_deserialize_rejects_non_string() {
        let err = serde_json::from_str::<ObjectId>("123").unwrap_err();
        assert!(err.to_string().contains("a string object identifier"));
    }
}
