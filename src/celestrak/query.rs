/*!
 * Query builder for the CelestrakClient API.
 *
 * Provides a fluent builder pattern for constructing CelestrakClient queries
 * with both server-side parameters (sent to CelestrakClient) and client-side
 * post-processing options (filtering, ordering, limiting applied after download).
 *
 * CelestrakClient uses standard HTTP query parameters rather than path-based queries:
 * `?GROUP=stations&FORMAT=JSON`
 */

use crate::celestrak::types::{CelestrakOutputFormat, CelestrakQueryType, SupGPSource};

/// A client-side filter predicate for post-download filtering.
///
/// Uses the same operator string format as SpaceTrack operators
/// (e.g., `>50`, `<0.01`, `<>DEBRIS`, `25544--25600`, `~~STARLINK`, `^NOAA`).
#[derive(Debug, Clone)]
pub(crate) struct Filter {
    pub(crate) field: String,
    pub(crate) value: String,
}

/// An ordering clause for client-side sorting of results.
#[derive(Debug, Clone)]
pub(crate) struct OrderBy {
    pub(crate) field: String,
    pub(crate) ascending: bool,
}

/// Builder for constructing CelestrakClient API queries.
///
/// Uses a fluent API where each method returns `Self` to allow method chaining.
/// Parameters are divided into:
///
/// - **Server-side**: Sent as URL query parameters to CelestrakClient (group, catnr, etc.)
/// - **Client-side**: Applied by Brahe after downloading the data (filter, order_by, limit)
///
/// # Examples
///
/// ```
/// use brahe::celestrak::{CelestrakQuery, CelestrakOutputFormat};
///
/// // GP query by satellite group
/// let query = CelestrakQuery::gp()
///     .group("stations")
///     .format(CelestrakOutputFormat::Json);
///
/// let url = query.build_url();
/// assert!(url.contains("GROUP=stations"));
/// assert!(url.contains("FORMAT=JSON"));
///
/// // SATCAT query for active payloads
/// let query = CelestrakQuery::satcat()
///     .active(true)
///     .payloads(true);
///
/// let url = query.build_url();
/// assert!(url.contains("ACTIVE=Y"));
/// assert!(url.contains("PAYLOADS=Y"));
/// ```
#[derive(Debug, Clone)]
pub struct CelestrakQuery {
    query_type: CelestrakQueryType,
    // Server-side parameters (sent to CelestrakClient API)
    group: Option<String>,
    catnr: Option<u32>,
    intdes: Option<String>,
    name: Option<String>,
    special: Option<String>,
    source: Option<SupGPSource>,
    file: Option<String>,
    payloads: Option<bool>,
    on_orbit: Option<bool>,
    active: Option<bool>,
    max_results: Option<u32>,
    output_format: Option<CelestrakOutputFormat>,
    // Client-side post-processing (applied by Brahe after download)
    filters: Vec<Filter>,
    order_by_clauses: Vec<OrderBy>,
    limit_count: Option<u32>,
}

impl CelestrakQuery {
    /// Create a new query builder for the specified query type.
    ///
    /// # Arguments
    ///
    /// * `query_type` - The CelestrakClient API endpoint to query
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::celestrak::{CelestrakQuery, CelestrakQueryType};
    ///
    /// let query = CelestrakQuery::new(CelestrakQueryType::GP);
    /// assert_eq!(query.query_type(), CelestrakQueryType::GP);
    /// ```
    pub fn new(query_type: CelestrakQueryType) -> Self {
        CelestrakQuery {
            query_type,
            group: None,
            catnr: None,
            intdes: None,
            name: None,
            special: None,
            source: None,
            file: None,
            payloads: None,
            on_orbit: None,
            active: None,
            max_results: None,
            output_format: None,
            filters: Vec::new(),
            order_by_clauses: Vec::new(),
            limit_count: None,
        }
    }

    /// Create a new GP query builder.
    ///
    /// Shorthand for `CelestrakQuery::new(CelestrakQueryType::GP)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::celestrak::{CelestrakQuery, CelestrakQueryType};
    ///
    /// let query = CelestrakQuery::gp();
    /// assert_eq!(query.query_type(), CelestrakQueryType::GP);
    /// ```
    pub fn gp() -> Self {
        Self::new(CelestrakQueryType::GP)
    }

    /// Create a new supplemental GP query builder.
    ///
    /// Shorthand for `CelestrakQuery::new(CelestrakQueryType::SupGP)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::celestrak::{CelestrakQuery, CelestrakQueryType};
    ///
    /// let query = CelestrakQuery::sup_gp();
    /// assert_eq!(query.query_type(), CelestrakQueryType::SupGP);
    /// ```
    pub fn sup_gp() -> Self {
        Self::new(CelestrakQueryType::SupGP)
    }

    /// Create a new SATCAT query builder.
    ///
    /// Shorthand for `CelestrakQuery::new(CelestrakQueryType::SATCAT)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::celestrak::{CelestrakQuery, CelestrakQueryType};
    ///
    /// let query = CelestrakQuery::satcat();
    /// assert_eq!(query.query_type(), CelestrakQueryType::SATCAT);
    /// ```
    pub fn satcat() -> Self {
        Self::new(CelestrakQueryType::SATCAT)
    }

    // -- Server-side parameters --

    /// Set the satellite group for the query.
    ///
    /// Available for GP and SATCAT queries. Groups include "stations", "active",
    /// "gnss", "last-30-days", etc.
    ///
    /// # Arguments
    ///
    /// * `group` - The satellite group name
    pub fn group(mut self, group: &str) -> Self {
        self.group = Some(group.to_string());
        self
    }

    /// Set the NORAD catalog number filter.
    ///
    /// Available for GP, SupGP, and SATCAT queries.
    ///
    /// # Arguments
    ///
    /// * `norad_id` - The NORAD catalog number
    pub fn catnr(mut self, norad_id: u32) -> Self {
        self.catnr = Some(norad_id);
        self
    }

    /// Set the international designator filter.
    ///
    /// Available for GP, SupGP, and SATCAT queries.
    ///
    /// # Arguments
    ///
    /// * `intdes` - The international designator (e.g., "1998-067A")
    pub fn intdes(mut self, intdes: &str) -> Self {
        self.intdes = Some(intdes.to_string());
        self
    }

    /// Set the satellite name search filter.
    ///
    /// Available for GP, SupGP, and SATCAT queries.
    ///
    /// # Arguments
    ///
    /// * `name` - The satellite name (partial match supported)
    pub fn name_search(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Set the special query parameter.
    ///
    /// Available for all query types. Typically used for special collections
    /// or custom queries.
    ///
    /// # Arguments
    ///
    /// * `special` - The special query value
    pub fn special(mut self, special: &str) -> Self {
        self.special = Some(special.to_string());
        self
    }

    /// Set the supplemental GP data source.
    ///
    /// Only available for SupGP queries.
    ///
    /// # Arguments
    ///
    /// * `source` - The supplemental data source
    pub fn source(mut self, source: SupGPSource) -> Self {
        self.source = Some(source);
        self
    }

    /// Set the file parameter for supplemental GP queries.
    ///
    /// Only available for SupGP queries.
    ///
    /// # Arguments
    ///
    /// * `file` - The file identifier
    pub fn file(mut self, file: &str) -> Self {
        self.file = Some(file.to_string());
        self
    }

    /// Filter to payloads only.
    ///
    /// Only available for SATCAT queries.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to filter to payloads only
    pub fn payloads(mut self, enabled: bool) -> Self {
        self.payloads = Some(enabled);
        self
    }

    /// Filter to on-orbit objects only.
    ///
    /// Only available for SATCAT queries.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to filter to on-orbit objects only
    pub fn on_orbit(mut self, enabled: bool) -> Self {
        self.on_orbit = Some(enabled);
        self
    }

    /// Filter to active objects only.
    ///
    /// Only available for SATCAT queries.
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to filter to active objects only
    pub fn active(mut self, enabled: bool) -> Self {
        self.active = Some(enabled);
        self
    }

    /// Set the maximum number of results returned by the server.
    ///
    /// Only available for SATCAT queries. This is a server-side limit.
    /// For client-side limiting (applied after download), use `limit()`.
    ///
    /// # Arguments
    ///
    /// * `max` - Maximum number of results
    pub fn max(mut self, max: u32) -> Self {
        self.max_results = Some(max);
        self
    }

    /// Set the output format for query results.
    ///
    /// If not specified, typed query methods (`query_gp`, `query_satcat`)
    /// will automatically use JSON.
    ///
    /// # Arguments
    ///
    /// * `format` - The desired output format
    pub fn format(mut self, format: CelestrakOutputFormat) -> Self {
        self.output_format = Some(format);
        self
    }

    // -- Client-side post-processing --

    /// Add a client-side filter predicate.
    ///
    /// Filters are applied by Brahe after downloading the data. Use SpaceTrack
    /// operator functions to construct comparison values.
    ///
    /// # Arguments
    ///
    /// * `field` - The record field name (e.g., "INCLINATION", "ECCENTRICITY")
    /// * `value` - The filter value with optional operator prefix (e.g., ">50", "<0.01")
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::celestrak::CelestrakQuery;
    /// use brahe::spacetrack::operators;
    ///
    /// let query = CelestrakQuery::gp()
    ///     .group("stations")
    ///     .filter("INCLINATION", &operators::greater_than("50"))
    ///     .filter("ECCENTRICITY", &operators::less_than("0.01"));
    /// ```
    pub fn filter(mut self, field: &str, value: &str) -> Self {
        self.filters.push(Filter {
            field: field.to_string(),
            value: value.to_string(),
        });
        self
    }

    /// Add a client-side ordering clause.
    ///
    /// Ordering is applied by Brahe after downloading the data.
    /// Multiple calls are cumulative.
    ///
    /// # Arguments
    ///
    /// * `field` - The field to sort by
    /// * `ascending` - Whether to sort ascending (true) or descending (false)
    pub fn order_by(mut self, field: &str, ascending: bool) -> Self {
        self.order_by_clauses.push(OrderBy {
            field: field.to_string(),
            ascending,
        });
        self
    }

    /// Set a client-side limit on the number of results.
    ///
    /// This limit is applied after downloading and filtering. For server-side
    /// limiting on SATCAT queries, use `max()`.
    ///
    /// # Arguments
    ///
    /// * `count` - Maximum number of records to return after filtering
    pub fn limit(mut self, count: u32) -> Self {
        self.limit_count = Some(count);
        self
    }

    // -- Accessors --

    /// Returns the query type for this query.
    pub fn query_type(&self) -> CelestrakQueryType {
        self.query_type
    }

    /// Returns the output format for this query.
    ///
    /// Returns `None` if no format has been explicitly set.
    pub fn output_format(&self) -> Option<CelestrakOutputFormat> {
        self.output_format
    }

    /// Returns true if this query has any client-side filters.
    pub fn has_client_side_processing(&self) -> bool {
        !self.filters.is_empty() || !self.order_by_clauses.is_empty() || self.limit_count.is_some()
    }

    /// Returns a reference to the client-side filters.
    pub(crate) fn client_side_filters(&self) -> &[Filter] {
        &self.filters
    }

    /// Returns a reference to the client-side ordering clauses.
    pub(crate) fn client_side_order_by(&self) -> &[OrderBy] {
        &self.order_by_clauses
    }

    /// Returns the client-side limit, if set.
    pub(crate) fn client_side_limit(&self) -> Option<u32> {
        self.limit_count
    }

    // -- URL building --

    /// Build the URL query string for this query.
    ///
    /// Produces the query string portion (after `?`) to be appended to the
    /// base URL and endpoint path. Only includes server-side parameters;
    /// client-side filters are not included in the URL.
    ///
    /// # Returns
    ///
    /// * `String` - The URL query string (e.g., "GROUP=stations&FORMAT=JSON")
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::celestrak::{CelestrakQuery, CelestrakOutputFormat};
    ///
    /// let query = CelestrakQuery::gp()
    ///     .group("stations")
    ///     .format(CelestrakOutputFormat::Json);
    ///
    /// let url = query.build_url();
    /// assert_eq!(url, "GROUP=stations&FORMAT=JSON");
    /// ```
    pub fn build_url(&self) -> String {
        let mut params = Vec::new();

        // GP and SATCAT: GROUP
        if let Some(ref group) = self.group {
            params.push(format!("GROUP={}", group));
        }

        // GP, SupGP, SATCAT: CATNR
        if let Some(catnr) = self.catnr {
            params.push(format!("CATNR={}", catnr));
        }

        // GP, SupGP, SATCAT: INTDES
        if let Some(ref intdes) = self.intdes {
            params.push(format!("INTDES={}", intdes));
        }

        // GP, SupGP, SATCAT: NAME
        if let Some(ref name) = self.name {
            params.push(format!("NAME={}", name));
        }

        // All: SPECIAL
        if let Some(ref special) = self.special {
            params.push(format!("SPECIAL={}", special));
        }

        // SupGP: SOURCE
        if let Some(ref source) = self.source {
            params.push(format!("SOURCE={}", source.as_str()));
        }

        // SupGP: FILE
        if let Some(ref file) = self.file {
            params.push(format!("FILE={}", file));
        }

        // SATCAT: PAYLOADS
        if let Some(enabled) = self.payloads
            && enabled
        {
            params.push("PAYLOADS=Y".to_string());
        }

        // SATCAT: ONORBIT
        if let Some(enabled) = self.on_orbit
            && enabled
        {
            params.push("ONORBIT=Y".to_string());
        }

        // SATCAT: ACTIVE
        if let Some(enabled) = self.active
            && enabled
        {
            params.push("ACTIVE=Y".to_string());
        }

        // SATCAT: MAX
        if let Some(max) = self.max_results {
            params.push(format!("MAX={}", max));
        }

        // All: FORMAT
        if let Some(ref format) = self.output_format {
            params.push(format!("FORMAT={}", format.as_str()));
        }

        params.join("&")
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    // -- Constructor tests --

    #[test]
    fn test_gp_constructor() {
        let query = CelestrakQuery::gp();
        assert_eq!(query.query_type(), CelestrakQueryType::GP);
    }

    #[test]
    fn test_sup_gp_constructor() {
        let query = CelestrakQuery::sup_gp();
        assert_eq!(query.query_type(), CelestrakQueryType::SupGP);
    }

    #[test]
    fn test_satcat_constructor() {
        let query = CelestrakQuery::satcat();
        assert_eq!(query.query_type(), CelestrakQueryType::SATCAT);
    }

    #[test]
    fn test_new_constructor() {
        let query = CelestrakQuery::new(CelestrakQueryType::GP);
        assert_eq!(query.query_type(), CelestrakQueryType::GP);
    }

    // -- GP URL building tests --

    #[test]
    fn test_gp_by_group() {
        let query = CelestrakQuery::gp().group("stations");
        assert_eq!(query.build_url(), "GROUP=stations");
    }

    #[test]
    fn test_gp_by_catnr() {
        let query = CelestrakQuery::gp().catnr(25544);
        assert_eq!(query.build_url(), "CATNR=25544");
    }

    #[test]
    fn test_gp_by_intdes() {
        let query = CelestrakQuery::gp().intdes("1998-067A");
        assert_eq!(query.build_url(), "INTDES=1998-067A");
    }

    #[test]
    fn test_gp_by_name() {
        let query = CelestrakQuery::gp().name_search("ISS");
        assert_eq!(query.build_url(), "NAME=ISS");
    }

    #[test]
    fn test_gp_with_format() {
        let query = CelestrakQuery::gp()
            .group("stations")
            .format(CelestrakOutputFormat::Json);
        assert_eq!(query.build_url(), "GROUP=stations&FORMAT=JSON");
    }

    #[test]
    fn test_gp_with_tle_format() {
        let query = CelestrakQuery::gp()
            .group("stations")
            .format(CelestrakOutputFormat::ThreeLe);
        assert_eq!(query.build_url(), "GROUP=stations&FORMAT=3LE");
    }

    #[test]
    fn test_gp_with_special() {
        let query = CelestrakQuery::gp().special("all");
        assert_eq!(query.build_url(), "SPECIAL=all");
    }

    #[test]
    fn test_gp_multiple_params() {
        let query = CelestrakQuery::gp()
            .group("stations")
            .catnr(25544)
            .format(CelestrakOutputFormat::Json);
        assert_eq!(query.build_url(), "GROUP=stations&CATNR=25544&FORMAT=JSON");
    }

    // -- SupGP URL building tests --

    #[test]
    fn test_sup_gp_by_source() {
        let query = CelestrakQuery::sup_gp().source(SupGPSource::SpaceX);
        assert_eq!(query.build_url(), "SOURCE=spacex");
    }

    #[test]
    fn test_sup_gp_by_file() {
        let query = CelestrakQuery::sup_gp().file("starlink");
        assert_eq!(query.build_url(), "FILE=starlink");
    }

    #[test]
    fn test_sup_gp_by_catnr() {
        let query = CelestrakQuery::sup_gp().catnr(25544);
        assert_eq!(query.build_url(), "CATNR=25544");
    }

    #[test]
    fn test_sup_gp_with_source_and_format() {
        let query = CelestrakQuery::sup_gp()
            .source(SupGPSource::Starlink)
            .format(CelestrakOutputFormat::Json);
        assert_eq!(query.build_url(), "SOURCE=starlink&FORMAT=JSON");
    }

    #[test]
    fn test_sup_gp_with_name() {
        let query = CelestrakQuery::sup_gp().name_search("STARLINK-1234");
        assert_eq!(query.build_url(), "NAME=STARLINK-1234");
    }

    // -- SATCAT URL building tests --

    #[test]
    fn test_satcat_by_group() {
        let query = CelestrakQuery::satcat().group("stations");
        assert_eq!(query.build_url(), "GROUP=stations");
    }

    #[test]
    fn test_satcat_active() {
        let query = CelestrakQuery::satcat().active(true);
        assert_eq!(query.build_url(), "ACTIVE=Y");
    }

    #[test]
    fn test_satcat_payloads() {
        let query = CelestrakQuery::satcat().payloads(true);
        assert_eq!(query.build_url(), "PAYLOADS=Y");
    }

    #[test]
    fn test_satcat_on_orbit() {
        let query = CelestrakQuery::satcat().on_orbit(true);
        assert_eq!(query.build_url(), "ONORBIT=Y");
    }

    #[test]
    fn test_satcat_with_max() {
        let query = CelestrakQuery::satcat().active(true).max(100);
        assert_eq!(query.build_url(), "ACTIVE=Y&MAX=100");
    }

    #[test]
    fn test_satcat_multiple_flags() {
        let query = CelestrakQuery::satcat()
            .active(true)
            .payloads(true)
            .on_orbit(true)
            .format(CelestrakOutputFormat::Json);
        assert_eq!(
            query.build_url(),
            "PAYLOADS=Y&ONORBIT=Y&ACTIVE=Y&FORMAT=JSON"
        );
    }

    #[test]
    fn test_satcat_false_flags_not_in_url() {
        let query = CelestrakQuery::satcat()
            .active(false)
            .payloads(false)
            .on_orbit(false);
        assert_eq!(query.build_url(), "");
    }

    #[test]
    fn test_satcat_by_name() {
        let query = CelestrakQuery::satcat()
            .name_search("ISS")
            .format(CelestrakOutputFormat::Json);
        assert_eq!(query.build_url(), "NAME=ISS&FORMAT=JSON");
    }

    // -- Client-side parameter tests --

    #[test]
    fn test_client_side_filter() {
        let query = CelestrakQuery::gp()
            .group("stations")
            .filter("INCLINATION", ">50");
        assert!(query.has_client_side_processing());
        assert_eq!(query.client_side_filters().len(), 1);
        assert_eq!(query.client_side_filters()[0].field, "INCLINATION");
        assert_eq!(query.client_side_filters()[0].value, ">50");
        // Client-side filters not in URL
        assert_eq!(query.build_url(), "GROUP=stations");
    }

    #[test]
    fn test_client_side_order_by() {
        let query = CelestrakQuery::gp()
            .group("stations")
            .order_by("EPOCH", false);
        assert!(query.has_client_side_processing());
        assert_eq!(query.client_side_order_by().len(), 1);
        assert_eq!(query.client_side_order_by()[0].field, "EPOCH");
        assert!(!query.client_side_order_by()[0].ascending);
    }

    #[test]
    fn test_client_side_limit() {
        let query = CelestrakQuery::gp().group("stations").limit(10);
        assert!(query.has_client_side_processing());
        assert_eq!(query.client_side_limit(), Some(10));
    }

    #[test]
    fn test_no_client_side_processing() {
        let query = CelestrakQuery::gp().group("stations");
        assert!(!query.has_client_side_processing());
    }

    #[test]
    fn test_multiple_client_side_filters() {
        let query = CelestrakQuery::gp()
            .group("active")
            .filter("INCLINATION", ">50")
            .filter("ECCENTRICITY", "<0.01")
            .filter("OBJECT_TYPE", "<>DEBRIS");
        assert_eq!(query.client_side_filters().len(), 3);
    }

    // -- Accessor tests --

    #[test]
    fn test_output_format_accessor_none() {
        let query = CelestrakQuery::gp();
        assert_eq!(query.output_format(), None);
    }

    #[test]
    fn test_output_format_accessor_set() {
        let query = CelestrakQuery::gp().format(CelestrakOutputFormat::Json);
        assert_eq!(query.output_format(), Some(CelestrakOutputFormat::Json));
    }

    // -- Builder immutability tests --

    #[test]
    fn test_builder_immutability() {
        let base = CelestrakQuery::gp().group("stations");
        let extended = base.clone().filter("INCLINATION", ">50");

        // Base query should not be affected by extending
        assert!(!base.has_client_side_processing());
        assert!(extended.has_client_side_processing());
        assert_eq!(base.build_url(), "GROUP=stations");
        assert_eq!(extended.build_url(), "GROUP=stations");
    }

    // -- Clone tests --

    #[test]
    fn test_query_clone() {
        let query = CelestrakQuery::gp()
            .group("stations")
            .filter("INCLINATION", ">50")
            .order_by("EPOCH", false)
            .limit(10);

        let cloned = query.clone();
        assert_eq!(query.build_url(), cloned.build_url());
        assert_eq!(query.query_type(), cloned.query_type());
        assert_eq!(
            query.client_side_filters().len(),
            cloned.client_side_filters().len()
        );
        assert_eq!(
            query.client_side_order_by().len(),
            cloned.client_side_order_by().len()
        );
        assert_eq!(query.client_side_limit(), cloned.client_side_limit());
    }

    // -- All format tests --

    #[test]
    fn test_all_output_formats() {
        let formats = vec![
            (CelestrakOutputFormat::Tle, "FORMAT=TLE"),
            (CelestrakOutputFormat::TwoLe, "FORMAT=2LE"),
            (CelestrakOutputFormat::ThreeLe, "FORMAT=3LE"),
            (CelestrakOutputFormat::Xml, "FORMAT=XML"),
            (CelestrakOutputFormat::Kvn, "FORMAT=KVN"),
            (CelestrakOutputFormat::Json, "FORMAT=JSON"),
            (CelestrakOutputFormat::JsonPretty, "FORMAT=JSON-PRETTY"),
            (CelestrakOutputFormat::Csv, "FORMAT=CSV"),
        ];

        for (format, expected) in formats {
            let query = CelestrakQuery::gp().format(format);
            assert_eq!(query.build_url(), expected);
        }
    }

    // -- Empty query test --

    #[test]
    fn test_empty_query() {
        let query = CelestrakQuery::gp();
        assert_eq!(query.build_url(), "");
    }

    // -- Debug test --

    #[test]
    fn test_query_debug() {
        let query = CelestrakQuery::gp().group("stations");
        let debug = format!("{:?}", query);
        assert!(debug.contains("stations"));
        assert!(debug.contains("GP"));
    }
}
