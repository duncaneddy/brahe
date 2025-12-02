/*!
 * SpaceTrack request classes
 *
 * This module contains the request class definitions and traits for
 * building type-safe SpaceTrack API queries.
 */

use serde::de::DeserializeOwned;

use super::operators::QueryValue;

// Re-export all request classes - BasicSpaceData
pub mod announcement;
pub mod boxscore;
pub mod cdm_public;
pub mod decay;
pub mod gp;
pub mod gp_history;
pub mod launch_site;
pub mod omm;
pub mod satcat;
pub mod satcat_change;
pub mod satcat_debut;
pub mod tip;
pub mod tle;

// Re-export all request classes - ExpandedSpaceData
pub mod car;
pub mod cdm;
pub mod maneuver;
pub mod organization;
pub mod satellite;

// BasicSpaceData - public record types
pub use announcement::AnnouncementRecord;
pub use boxscore::BoxscoreRecord;
pub use cdm_public::CDMPublicRecord;
pub use decay::DecayRecord;
pub use gp::{GPRecord, GPRecordVecExt};
pub use gp_history::{GPHistoryRecord, GPHistoryRecordVecExt};
pub use launch_site::LaunchSiteRecord;
pub use omm::{OMMRecord, OMMRecordVecExt};
pub use satcat::SATCATRecord;
pub use satcat_change::SATCATChangeRecord;
pub use satcat_debut::SATCATDebutRecord;
pub use tip::TIPRecord;
pub use tle::{TLERecord, TLERecordVecExt};

// ExpandedSpaceData - public record types
pub use car::CARRecord;
pub use cdm::CDMRecord;
pub use maneuver::{ManeuverHistoryRecord, ManeuverRecord};
pub use organization::OrganizationRecord;
pub use satellite::SatelliteRecord;

// Internal request classes (used by client.rs and blocking.rs)
pub(crate) use announcement::AnnouncementRequest;
pub(crate) use boxscore::BoxscoreRequest;
pub(crate) use cdm_public::CDMPublicRequest;
pub(crate) use decay::DecayRequest;
pub(crate) use gp::GPRequest;
pub(crate) use launch_site::LaunchSiteRequest;
pub(crate) use satcat::SATCATRequest;
pub(crate) use satcat_change::SATCATChangeRequest;
pub(crate) use satcat_debut::SATCATDebutRequest;
pub(crate) use tip::TIPRequest;
pub(crate) use tle::TLERequest;

/// Trait for SpaceTrack request classes.
///
/// Each request class (GP, SATCAT, TLE, etc.) implements this trait
/// to provide type-safe query building and response parsing.
pub(crate) trait RequestClass: Sized + Send + Sync {
    /// The record type returned by this request class.
    type Record: DeserializeOwned + Clone + Send + Sync;

    /// Get the request class name (e.g., "gp", "satcat").
    fn class_name() -> &'static str;

    /// Get the controller name (e.g., "basicspacedata").
    fn controller() -> &'static str;

    /// Build the query predicates as (name, value) pairs.
    fn predicates(&self) -> Vec<(&'static str, QueryValue)>;
}

/// Order direction for sorting results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OrderDirection {
    /// Ascending order
    Asc,
    /// Descending order
    Desc,
}

impl OrderDirection {
    /// Convert to SpaceTrack query string suffix.
    pub(crate) fn as_query_string(self) -> &'static str {
        match self {
            OrderDirection::Asc => " asc",
            OrderDirection::Desc => " desc",
        }
    }
}

/// Ordering specification for query results.
#[derive(Debug, Clone)]
pub(crate) struct OrderBy {
    /// Field name to order by
    pub(crate) field: String,
    /// Order direction
    pub(crate) direction: OrderDirection,
}

impl OrderBy {
    /// Create an ascending order specification.
    pub(crate) fn asc(field: &str) -> Self {
        Self {
            field: field.to_uppercase(),
            direction: OrderDirection::Asc,
        }
    }

    /// Create a descending order specification.
    pub(crate) fn desc(field: &str) -> Self {
        Self {
            field: field.to_uppercase(),
            direction: OrderDirection::Desc,
        }
    }

    /// Convert to SpaceTrack query string.
    pub(crate) fn to_query_string(&self) -> String {
        format!("{}{}", self.field, self.direction.as_query_string())
    }
}

/// Macro to define a request class with typed predicates.
///
/// This macro generates:
/// - A request builder struct with typed predicate fields
/// - Builder methods for each predicate
/// - Implementation of the `RequestClass` trait
#[macro_export]
macro_rules! define_request_class {
    (
        $(#[$struct_meta:meta])*
        name: $name:ident,
        class_name: $class_name:literal,
        controller: $controller:literal,
        record: $record:ty,
        predicates: {
            $(
                $(#[$field_meta:meta])*
                $field:ident : $field_type:ty
            ),* $(,)?
        }
    ) => {
        $(#[$struct_meta])*
        #[derive(Debug, Clone, Default)]
        #[allow(dead_code)]
        pub(crate) struct $name {
            $(
                $(#[$field_meta])*
                pub(crate) $field: Option<$crate::spacetrack::operators::QueryValue>,
            )*
            /// Maximum number of results to return
            pub(crate) limit: Option<u32>,
            /// Field and direction to order results by
            pub(crate) orderby: Option<$crate::spacetrack::request_classes::OrderBy>,
            /// Whether to return distinct results only
            pub(crate) distinct: Option<bool>,
        }

        #[allow(dead_code)]
        impl $name {
            /// Create a new request builder with default values.
            pub(crate) fn new() -> Self {
                Self::default()
            }

            $(
                /// Set the $field predicate.
                pub(crate) fn $field(mut self, value: impl Into<$crate::spacetrack::operators::QueryValue>) -> Self {
                    self.$field = Some(value.into());
                    self
                }
            )*

            /// Set the maximum number of results to return.
            pub(crate) fn limit(mut self, limit: u32) -> Self {
                self.limit = Some(limit);
                self
            }

            /// Set the field and direction to order results by.
            pub(crate) fn orderby(mut self, orderby: $crate::spacetrack::request_classes::OrderBy) -> Self {
                self.orderby = Some(orderby);
                self
            }

            /// Order results by the specified field in ascending order.
            pub(crate) fn orderby_asc(mut self, field: &str) -> Self {
                self.orderby = Some($crate::spacetrack::request_classes::OrderBy::asc(field));
                self
            }

            /// Order results by the specified field in descending order.
            pub(crate) fn orderby_desc(mut self, field: &str) -> Self {
                self.orderby = Some($crate::spacetrack::request_classes::OrderBy::desc(field));
                self
            }

            /// Set whether to return distinct results only.
            pub(crate) fn distinct(mut self, distinct: bool) -> Self {
                self.distinct = Some(distinct);
                self
            }
        }

        impl $crate::spacetrack::request_classes::RequestClass for $name {
            type Record = $record;

            fn class_name() -> &'static str {
                $class_name
            }

            fn controller() -> &'static str {
                $controller
            }

            fn predicates(&self) -> Vec<(&'static str, $crate::spacetrack::operators::QueryValue)> {
                let mut predicates = Vec::new();

                $(
                    if let Some(ref value) = self.$field {
                        // Convert field name to uppercase for SpaceTrack API
                        predicates.push((stringify!($field).to_uppercase().leak() as &'static str, value.clone()));
                    }
                )*

                if let Some(limit) = self.limit {
                    predicates.push(("limit", $crate::spacetrack::operators::QueryValue::Value(limit.to_string())));
                }

                if let Some(ref orderby) = self.orderby {
                    predicates.push(("orderby", $crate::spacetrack::operators::QueryValue::Value(orderby.to_query_string())));
                }

                if let Some(distinct) = self.distinct {
                    if distinct {
                        predicates.push(("distinct", $crate::spacetrack::operators::QueryValue::Value("true".to_string())));
                    }
                }

                predicates
            }
        }
    };
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_order_direction() {
        assert_eq!(OrderDirection::Asc.as_query_string(), " asc");
        assert_eq!(OrderDirection::Desc.as_query_string(), " desc");
    }

    #[test]
    fn test_order_by_asc() {
        let order = OrderBy::asc("epoch");
        assert_eq!(order.field, "EPOCH");
        assert_eq!(order.direction, OrderDirection::Asc);
        assert_eq!(order.to_query_string(), "EPOCH asc");
    }

    #[test]
    fn test_order_by_desc() {
        let order = OrderBy::desc("epoch");
        assert_eq!(order.field, "EPOCH");
        assert_eq!(order.direction, OrderDirection::Desc);
        assert_eq!(order.to_query_string(), "EPOCH desc");
    }
}
