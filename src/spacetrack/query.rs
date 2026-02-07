/*!
 * SpaceTrack query builder.
 *
 * Provides a fluent builder API for constructing Space-Track.org API queries.
 * The builder produces URL path strings that are appended to the base URL
 * by the client.
 *
 * The Space-Track API uses a path-based query format:
 * `/<controller>/query/class/<class>/FIELD/value/.../format/<format>`
 */

use crate::spacetrack::types::{OutputFormat, RequestClass, RequestController, SortOrder};

/// Percent-encode characters that are invalid in URI path segments.
///
/// Encodes `>`, `<`, and `^` which are used by SpaceTrack filter operators
/// but are not valid URI path characters per RFC 3986.
fn encode_path_value(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '>' => result.push_str("%3E"),
            '<' => result.push_str("%3C"),
            '^' => result.push_str("%5E"),
            _ => result.push(c),
        }
    }
    result
}

/// A filter predicate for a SpaceTrack query.
#[derive(Debug, Clone)]
struct Filter {
    field: String,
    value: String,
}

/// An ordering clause for a SpaceTrack query.
#[derive(Debug, Clone)]
struct OrderBy {
    field: String,
    order: SortOrder,
}

/// Builder for constructing SpaceTrack API queries.
///
/// Uses a fluent API where each method returns `Self` to allow method chaining.
/// Call `build()` to produce the URL path string.
///
/// # Examples
///
/// ```
/// use brahe::spacetrack::{SpaceTrackQuery, RequestClass, SortOrder, OutputFormat};
///
/// // Query latest GP data for ISS
/// let query = SpaceTrackQuery::new(RequestClass::GP)
///     .filter("NORAD_CAT_ID", "25544")
///     .order_by("EPOCH", SortOrder::Desc)
///     .limit(1);
///
/// let url_path = query.build();
/// assert!(url_path.contains("/class/gp/"));
/// assert!(url_path.contains("/NORAD_CAT_ID/25544/"));
///
/// // Query with custom format
/// let query = SpaceTrackQuery::new(RequestClass::GP)
///     .filter("NORAD_CAT_ID", "25544")
///     .format(OutputFormat::TLE);
///
/// let url_path = query.build();
/// assert!(url_path.contains("/format/tle"));
/// ```
#[derive(Debug, Clone)]
pub struct SpaceTrackQuery {
    controller: RequestController,
    class: RequestClass,
    filters: Vec<Filter>,
    order_by: Vec<OrderBy>,
    limit_count: Option<u32>,
    limit_offset: Option<u32>,
    output_format: Option<OutputFormat>,
    predicates: Vec<String>,
    metadata: bool,
    distinct: bool,
    empty_result: bool,
    favorites: Option<String>,
}

impl SpaceTrackQuery {
    /// Create a new query builder for the specified request class.
    ///
    /// Uses the default controller for the request class.
    ///
    /// # Arguments
    ///
    /// * `class` - The request class to query
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::spacetrack::{SpaceTrackQuery, RequestClass};
    ///
    /// let query = SpaceTrackQuery::new(RequestClass::GP);
    /// ```
    pub fn new(class: RequestClass) -> Self {
        SpaceTrackQuery {
            controller: class.default_controller(),
            class,
            filters: Vec::new(),
            order_by: Vec::new(),
            limit_count: None,
            limit_offset: None,
            output_format: None,
            predicates: Vec::new(),
            metadata: false,
            distinct: false,
            empty_result: false,
            favorites: None,
        }
    }

    /// Override the default controller for this query.
    ///
    /// Most users will not need to call this method, as each request class
    /// has an appropriate default controller.
    ///
    /// # Arguments
    ///
    /// * `controller` - The controller to use
    pub fn controller(mut self, controller: RequestController) -> Self {
        self.controller = controller;
        self
    }

    /// Add a filter predicate to the query.
    ///
    /// Filters restrict the results to records where the specified field matches
    /// the given value. Use operator functions from the `operators` module to
    /// construct comparison values (e.g., `operators::greater_than("25544")`).
    ///
    /// # Arguments
    ///
    /// * `field` - The field name (e.g., "NORAD_CAT_ID", "EPOCH")
    /// * `value` - The filter value, optionally with operator prefix
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::spacetrack::{SpaceTrackQuery, RequestClass, operators};
    ///
    /// let query = SpaceTrackQuery::new(RequestClass::GP)
    ///     .filter("NORAD_CAT_ID", "25544")
    ///     .filter("EPOCH", &operators::greater_than("2024-01-01"));
    /// ```
    pub fn filter(mut self, field: &str, value: &str) -> Self {
        self.filters.push(Filter {
            field: field.to_string(),
            value: value.to_string(),
        });
        self
    }

    /// Add an ordering clause to the query.
    ///
    /// Multiple order_by calls are cumulative - records are sorted by the
    /// first field, then by subsequent fields for ties.
    ///
    /// # Arguments
    ///
    /// * `field` - The field to sort by
    /// * `order` - The sort direction (ascending or descending)
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::spacetrack::{SpaceTrackQuery, RequestClass, SortOrder};
    ///
    /// let query = SpaceTrackQuery::new(RequestClass::GP)
    ///     .order_by("EPOCH", SortOrder::Desc);
    /// ```
    pub fn order_by(mut self, field: &str, order: SortOrder) -> Self {
        self.order_by.push(OrderBy {
            field: field.to_string(),
            order,
        });
        self
    }

    /// Set the maximum number of results to return.
    ///
    /// # Arguments
    ///
    /// * `count` - Maximum number of records
    pub fn limit(mut self, count: u32) -> Self {
        self.limit_count = Some(count);
        self
    }

    /// Set the maximum number of results and an offset.
    ///
    /// # Arguments
    ///
    /// * `count` - Maximum number of records
    /// * `offset` - Number of records to skip
    pub fn limit_offset(mut self, count: u32, offset: u32) -> Self {
        self.limit_count = Some(count);
        self.limit_offset = Some(offset);
        self
    }

    /// Set the output format for query results.
    ///
    /// If not specified, defaults to JSON.
    ///
    /// # Arguments
    ///
    /// * `format` - The desired output format
    pub fn format(mut self, format: OutputFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    /// Specify which fields to include in the response.
    ///
    /// This is a predicates filter that limits which fields are returned
    /// for each record. Useful for reducing response size.
    ///
    /// # Arguments
    ///
    /// * `fields` - Slice of field names to include
    pub fn predicates_filter(mut self, fields: &[&str]) -> Self {
        self.predicates = fields.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Enable or disable metadata in the response.
    ///
    /// When enabled, the response includes query metadata (e.g., total count).
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to include metadata
    pub fn metadata(mut self, enabled: bool) -> Self {
        self.metadata = enabled;
        self
    }

    /// Enable or disable distinct results.
    ///
    /// When enabled, duplicate records are removed from the results.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to return distinct results
    pub fn distinct(mut self, enabled: bool) -> Self {
        self.distinct = enabled;
        self
    }

    /// Enable or disable empty result return.
    ///
    /// When enabled, an empty query (no results) returns an empty array/set
    /// instead of an error response.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to allow empty results
    pub fn empty_result(mut self, enabled: bool) -> Self {
        self.empty_result = enabled;
        self
    }

    /// Set a favorites filter for the query.
    ///
    /// # Arguments
    ///
    /// * `favorites` - The favorites identifier
    pub fn favorites(mut self, favorites: &str) -> Self {
        self.favorites = Some(favorites.to_string());
        self
    }

    /// Build the URL path string for this query.
    ///
    /// Produces the path portion of the Space-Track API URL, which should
    /// be appended to the base URL and authentication endpoint.
    ///
    /// # Returns
    ///
    /// * `String` - The URL path string for this query
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::spacetrack::{SpaceTrackQuery, RequestClass, SortOrder};
    ///
    /// let query = SpaceTrackQuery::new(RequestClass::GP)
    ///     .filter("NORAD_CAT_ID", "25544")
    ///     .order_by("EPOCH", SortOrder::Desc)
    ///     .limit(1);
    ///
    /// let path = query.build();
    /// assert_eq!(
    ///     path,
    ///     "/basicspacedata/query/class/gp/NORAD_CAT_ID/25544/orderby/EPOCH%20desc/limit/1/format/json"
    /// );
    /// ```
    pub fn build(&self) -> String {
        let mut parts = Vec::new();

        // Controller and query prefix
        parts.push(format!(
            "/{}/query/class/{}",
            self.controller.as_str(),
            self.class.as_str()
        ));

        // Filters (percent-encode operator characters in values)
        for filter in &self.filters {
            parts.push(format!(
                "/{}/{}",
                filter.field,
                encode_path_value(&filter.value)
            ));
        }

        // Order by
        if !self.order_by.is_empty() {
            let order_str: Vec<String> = self
                .order_by
                .iter()
                .map(|o| format!("{}%20{}", o.field, o.order.as_str()))
                .collect();
            parts.push(format!("/orderby/{}", order_str.join(",")));
        }

        // Limit
        if let Some(count) = self.limit_count {
            if let Some(offset) = self.limit_offset {
                parts.push(format!("/limit/{},{}", count, offset));
            } else {
                parts.push(format!("/limit/{}", count));
            }
        }

        // Predicates filter
        if !self.predicates.is_empty() {
            parts.push(format!("/predicates/{}", self.predicates.join(",")));
        }

        // Metadata
        if self.metadata {
            parts.push("/metadata/true".to_string());
        }

        // Distinct
        if self.distinct {
            parts.push("/distinct/true".to_string());
        }

        // Empty result
        if self.empty_result {
            parts.push("/emptyresult/show".to_string());
        }

        // Favorites
        if let Some(ref fav) = self.favorites {
            parts.push(format!("/favorites/{}", fav));
        }

        // Format (default to JSON)
        let format = self.output_format.unwrap_or(OutputFormat::JSON);
        parts.push(format!("/format/{}", format.as_str()));

        parts.concat()
    }

    /// Returns the output format for this query.
    ///
    /// Returns `OutputFormat::JSON` if no format has been explicitly set.
    pub fn output_format(&self) -> OutputFormat {
        self.output_format.unwrap_or(OutputFormat::JSON)
    }

    /// Returns the request class for this query.
    pub fn request_class(&self) -> RequestClass {
        self.class
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::spacetrack::operators;

    #[test]
    fn test_basic_query() {
        let query = SpaceTrackQuery::new(RequestClass::GP);
        let path = query.build();
        assert_eq!(path, "/basicspacedata/query/class/gp/format/json");
    }

    #[test]
    fn test_query_with_filter() {
        let query = SpaceTrackQuery::new(RequestClass::GP).filter("NORAD_CAT_ID", "25544");

        let path = query.build();
        assert_eq!(
            path,
            "/basicspacedata/query/class/gp/NORAD_CAT_ID/25544/format/json"
        );
    }

    #[test]
    fn test_query_with_multiple_filters() {
        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .filter("EPOCH", &operators::greater_than("2024-01-01"));

        let path = query.build();
        assert!(path.contains("/NORAD_CAT_ID/25544/"));
        assert!(path.contains("/EPOCH/%3E2024-01-01/"));
    }

    #[test]
    fn test_query_with_order_by() {
        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .order_by("EPOCH", SortOrder::Desc);

        let path = query.build();
        assert!(path.contains("/orderby/EPOCH%20desc/"));
    }

    #[test]
    fn test_query_with_multiple_order_by() {
        let query = SpaceTrackQuery::new(RequestClass::GP)
            .order_by("EPOCH", SortOrder::Desc)
            .order_by("NORAD_CAT_ID", SortOrder::Asc);

        let path = query.build();
        assert!(path.contains("/orderby/EPOCH%20desc,NORAD_CAT_ID%20asc/"));
    }

    #[test]
    fn test_query_with_limit() {
        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .limit(5);

        let path = query.build();
        assert!(path.contains("/limit/5/"));
    }

    #[test]
    fn test_query_with_limit_offset() {
        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .limit_offset(10, 20);

        let path = query.build();
        assert!(path.contains("/limit/10,20/"));
    }

    #[test]
    fn test_query_with_format() {
        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .format(OutputFormat::TLE);

        let path = query.build();
        assert!(path.ends_with("/format/tle"));
    }

    #[test]
    fn test_query_with_3le_format() {
        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .format(OutputFormat::ThreeLe);

        let path = query.build();
        assert!(path.ends_with("/format/3le"));
    }

    #[test]
    fn test_query_with_predicates() {
        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .predicates_filter(&["NORAD_CAT_ID", "OBJECT_NAME", "EPOCH"]);

        let path = query.build();
        assert!(path.contains("/predicates/NORAD_CAT_ID,OBJECT_NAME,EPOCH/"));
    }

    #[test]
    fn test_query_with_metadata() {
        let query = SpaceTrackQuery::new(RequestClass::GP).metadata(true);

        let path = query.build();
        assert!(path.contains("/metadata/true"));
    }

    #[test]
    fn test_query_with_distinct() {
        let query = SpaceTrackQuery::new(RequestClass::GP).distinct(true);

        let path = query.build();
        assert!(path.contains("/distinct/true"));
    }

    #[test]
    fn test_query_with_empty_result() {
        let query = SpaceTrackQuery::new(RequestClass::GP).empty_result(true);

        let path = query.build();
        assert!(path.contains("/emptyresult/show"));
    }

    #[test]
    fn test_query_with_favorites() {
        let query = SpaceTrackQuery::new(RequestClass::GP).favorites("my_favorites");

        let path = query.build();
        assert!(path.contains("/favorites/my_favorites/"));
    }

    #[test]
    fn test_query_with_custom_controller() {
        let query =
            SpaceTrackQuery::new(RequestClass::GP).controller(RequestController::ExpandedSpaceData);

        let path = query.build();
        assert!(path.starts_with("/expandedspacedata/"));
    }

    #[test]
    fn test_query_cdm_public_default_controller() {
        let query = SpaceTrackQuery::new(RequestClass::CDMPublic);

        let path = query.build();
        assert!(path.starts_with("/expandedspacedata/"));
    }

    #[test]
    fn test_query_satcat() {
        let query = SpaceTrackQuery::new(RequestClass::SATCAT)
            .filter("NORAD_CAT_ID", "25544")
            .limit(1);

        let path = query.build();
        assert_eq!(
            path,
            "/basicspacedata/query/class/satcat/NORAD_CAT_ID/25544/limit/1/format/json"
        );
    }

    #[test]
    fn test_full_query() {
        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .order_by("EPOCH", SortOrder::Desc)
            .limit(1);

        let path = query.build();
        assert_eq!(
            path,
            "/basicspacedata/query/class/gp/NORAD_CAT_ID/25544/orderby/EPOCH%20desc/limit/1/format/json"
        );
    }

    #[test]
    fn test_query_with_operators() {
        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("EPOCH", &operators::greater_than(operators::now_offset(-7)))
            .filter("ECCENTRICITY", &operators::less_than("0.01"))
            .filter("OBJECT_TYPE", &operators::not_equal("DEBRIS"))
            .filter("NORAD_CAT_ID", &operators::inclusive_range(25544, 25600))
            .format(OutputFormat::JSON);

        let path = query.build();
        assert!(path.contains("/EPOCH/%3Enow-7/"));
        assert!(path.contains("/ECCENTRICITY/%3C0.01/"));
        assert!(path.contains("/OBJECT_TYPE/%3C%3EDEBRIS/"));
        assert!(path.contains("/NORAD_CAT_ID/25544--25600/"));
    }

    #[test]
    fn test_query_output_format_accessor() {
        let query = SpaceTrackQuery::new(RequestClass::GP);
        assert_eq!(query.output_format(), OutputFormat::JSON);

        let query = SpaceTrackQuery::new(RequestClass::GP).format(OutputFormat::TLE);
        assert_eq!(query.output_format(), OutputFormat::TLE);
    }

    #[test]
    fn test_query_request_class_accessor() {
        let query = SpaceTrackQuery::new(RequestClass::GP);
        assert_eq!(query.request_class(), RequestClass::GP);

        let query = SpaceTrackQuery::new(RequestClass::SATCAT);
        assert_eq!(query.request_class(), RequestClass::SATCAT);
    }

    #[test]
    fn test_query_clone() {
        let query = SpaceTrackQuery::new(RequestClass::GP)
            .filter("NORAD_CAT_ID", "25544")
            .limit(1);

        let cloned = query.clone();
        assert_eq!(query.build(), cloned.build());
    }

    #[test]
    fn test_query_metadata_false_not_in_url() {
        let query = SpaceTrackQuery::new(RequestClass::GP).metadata(false);

        let path = query.build();
        assert!(!path.contains("metadata"));
    }

    #[test]
    fn test_query_distinct_false_not_in_url() {
        let query = SpaceTrackQuery::new(RequestClass::GP).distinct(false);

        let path = query.build();
        assert!(!path.contains("distinct"));
    }

    #[test]
    fn test_query_empty_result_false_not_in_url() {
        let query = SpaceTrackQuery::new(RequestClass::GP).empty_result(false);

        let path = query.build();
        assert!(!path.contains("emptyresult"));
    }

    #[test]
    fn test_all_request_classes() {
        let classes = vec![
            RequestClass::GP,
            RequestClass::GPHistory,
            RequestClass::SATCAT,
            RequestClass::SATCATChange,
            RequestClass::SATCATDebut,
            RequestClass::Decay,
            RequestClass::TIP,
            RequestClass::CDMPublic,
            RequestClass::Boxscore,
            RequestClass::Announcement,
            RequestClass::LaunchSite,
        ];

        for class in classes {
            let query = SpaceTrackQuery::new(class);
            let path = query.build();
            assert!(path.contains(&format!("/class/{}/", class.as_str())));
        }
    }

    #[test]
    fn test_query_with_xml_format() {
        let query = SpaceTrackQuery::new(RequestClass::GP).format(OutputFormat::XML);
        let path = query.build();
        assert!(path.ends_with("/format/xml"));
    }

    #[test]
    fn test_query_with_html_format() {
        let query = SpaceTrackQuery::new(RequestClass::GP).format(OutputFormat::HTML);
        let path = query.build();
        assert!(path.ends_with("/format/html"));
    }

    #[test]
    fn test_query_with_csv_format() {
        let query = SpaceTrackQuery::new(RequestClass::GP).format(OutputFormat::CSV);
        let path = query.build();
        assert!(path.ends_with("/format/csv"));
    }

    #[test]
    fn test_query_with_kvn_format() {
        let query = SpaceTrackQuery::new(RequestClass::GP).format(OutputFormat::KVN);
        let path = query.build();
        assert!(path.ends_with("/format/kvn"));
    }

    #[test]
    fn test_query_with_json_format_explicit() {
        let query = SpaceTrackQuery::new(RequestClass::GP).format(OutputFormat::JSON);
        let path = query.build();
        assert!(path.ends_with("/format/json"));
    }

    #[test]
    fn test_query_with_all_controllers() {
        let controllers = vec![
            (RequestController::BasicSpaceData, "basicspacedata"),
            (RequestController::ExpandedSpaceData, "expandedspacedata"),
            (RequestController::FileShare, "fileshare"),
            (RequestController::PublicFiles, "publicfiles"),
        ];

        for (controller, expected) in controllers {
            let query = SpaceTrackQuery::new(RequestClass::GP).controller(controller);
            let path = query.build();
            assert!(
                path.starts_with(&format!("/{}/", expected)),
                "Expected path to start with /{expected}/, got: {path}"
            );
        }
    }
}
