/*!
 * SpaceTrack blocking (synchronous) client
 *
 * Provides a synchronous wrapper around the async SpaceTrackClient
 * using an owned Tokio runtime.
 */

use std::marker::PhantomData;
use std::sync::Arc;

use serde::de::DeserializeOwned;
use tokio::runtime::{Builder, Runtime};

use super::client::SpaceTrackClient;
use super::operators::QueryValue;
use super::query::{
    HasTLEData, SpaceTrackOrder, SpaceTrackPredicate, SpaceTrackPredicateBuilder, SpaceTrackQuery,
};
use super::request_classes::{
    AnnouncementRecord, BoxscoreRecord, CDMPublicRecord, DecayRecord, GPHistoryRecord, GPRecord,
    LaunchSiteRecord, OMMRecord, RequestClass, SATCATChangeRecord, SATCATDebutRecord, SATCATRecord,
    TIPRecord, TLERecord,
};
use crate::propagators::SGPPropagator;
use crate::utils::BraheError;

/// Blocking SpaceTrack API client.
///
/// This is a synchronous wrapper around `SpaceTrackClient` that owns
/// a Tokio runtime for executing async operations. It can be used in
/// non-async contexts.
///
/// # Example
///
/// ```ignore
/// use brahe::spacetrack::BlockingSpaceTrackClient;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let client = BlockingSpaceTrackClient::new("username", "password")?;
///
///     // Query GP data for ISS
///     let response = client.generic_request(
///         "basicspacedata",
///         "gp",
///         &[("NORAD_CAT_ID", 25544.into())],
///         Some("json"),
///     )?;
///
///     Ok(())
/// }
/// ```
pub struct BlockingSpaceTrackClient {
    /// The async client
    inner: SpaceTrackClient,
    /// Owned Tokio runtime for executing async operations
    runtime: Arc<Runtime>,
}

impl BlockingSpaceTrackClient {
    /// Create a new blocking SpaceTrack client and authenticate.
    ///
    /// # Arguments
    ///
    /// * `identity` - SpaceTrack username
    /// * `password` - SpaceTrack password
    ///
    /// # Returns
    ///
    /// A new authenticated `BlockingSpaceTrackClient` or an error.
    pub fn new(identity: &str, password: &str) -> Result<Self, BraheError> {
        Self::with_base_url(identity, password, super::client::DEFAULT_BASE_URL)
    }

    /// Create a new blocking SpaceTrack client with a custom base URL.
    ///
    /// # Arguments
    ///
    /// * `identity` - SpaceTrack username
    /// * `password` - SpaceTrack password
    /// * `base_url` - Base URL for API requests
    ///
    /// # Returns
    ///
    /// A new authenticated `BlockingSpaceTrackClient` or an error.
    pub fn with_base_url(
        identity: &str,
        password: &str,
        base_url: &str,
    ) -> Result<Self, BraheError> {
        // Create a current-thread runtime for blocking operations
        let runtime = Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| BraheError::Error(format!("Failed to create runtime: {}", e)))?;

        // Create and authenticate the async client
        let inner = runtime.block_on(SpaceTrackClient::with_base_url(
            identity, password, base_url,
        ))?;

        Ok(Self {
            inner,
            runtime: Arc::new(runtime),
        })
    }

    /// Authenticate with SpaceTrack.
    ///
    /// This is called automatically when creating a new client, but can be
    /// called again if the session expires.
    pub fn authenticate(&mut self) -> Result<(), BraheError> {
        self.runtime
            .block_on(self.inner.authenticate())
            .map_err(Into::into)
    }

    /// Log out of SpaceTrack.
    pub fn logout(&mut self) -> Result<(), BraheError> {
        self.runtime
            .block_on(self.inner.logout())
            .map_err(Into::into)
    }

    /// Check if the client is authenticated.
    pub fn is_authenticated(&self) -> bool {
        self.inner.is_authenticated()
    }

    /// Get the base URL.
    pub fn base_url(&self) -> &str {
        self.inner.base_url()
    }

    /// Make a generic API request.
    ///
    /// # Arguments
    ///
    /// * `controller` - The controller name (e.g., "basicspacedata")
    /// * `class` - The request class name (e.g., "gp")
    /// * `predicates` - Query predicates as (name, value) pairs
    /// * `format` - Optional output format
    ///
    /// # Returns
    ///
    /// The response body as a string.
    pub fn generic_request(
        &self,
        controller: &str,
        class: &str,
        predicates: &[(&str, QueryValue)],
        format: Option<&str>,
    ) -> Result<String, BraheError> {
        self.runtime
            .block_on(
                self.inner
                    .generic_request(controller, class, predicates, format),
            )
            .map_err(Into::into)
    }

    /// Execute a typed request.
    ///
    /// # Type Parameters
    ///
    /// * `R` - The request class type implementing `RequestClass`
    ///
    /// # Arguments
    ///
    /// * `request` - The request builder
    ///
    /// # Returns
    ///
    /// A vector of the record type defined by the request class.
    #[allow(dead_code)] // Used by pymodule
    pub(crate) fn query<R: RequestClass>(&self, request: &R) -> Result<Vec<R::Record>, BraheError> {
        self.runtime
            .block_on(self.inner.query(request))
            .map_err(Into::into)
    }

    /// Execute a request and return TLEs as SGP propagators.
    ///
    /// # Arguments
    ///
    /// * `request` - The request builder (should be a GP or TLE request)
    /// * `step_size` - The propagator step size in seconds
    ///
    /// # Returns
    ///
    /// A vector of `SGPPropagator` instances.
    #[allow(dead_code)] // Used by pymodule
    pub(crate) fn fetch_propagators<R: RequestClass>(
        &self,
        request: &R,
        step_size: f64,
    ) -> Result<Vec<SGPPropagator>, BraheError> {
        self.runtime
            .block_on(self.inner.fetch_propagators(request, step_size))
    }

    // ========================================================================
    // Typed query methods
    // ========================================================================

    /// Query GP (General Perturbations) data.
    ///
    /// Returns the latest GP element sets for cataloged objects.
    ///
    /// # Arguments
    ///
    /// * `norad_cat_id` - Filter by NORAD catalog ID
    /// * `object_name` - Filter by object name (supports operators like "~~STARLINK%")
    /// * `object_id` - Filter by object ID (international designator)
    /// * `epoch` - Filter by epoch (supports operators like ">2024-01-01")
    /// * `object_type` - Filter by type (PAYLOAD, ROCKET BODY, DEBRIS)
    /// * `country_code` - Filter by country code
    /// * `limit` - Maximum number of results
    /// * `orderby` - Field to order by with direction (e.g., "epoch", "epoch desc")
    ///
    /// # Returns
    ///
    /// A vector of GP records.
    #[allow(clippy::too_many_arguments)]
    pub fn gp(
        &self,
        norad_cat_id: Option<u32>,
        object_name: Option<&str>,
        object_id: Option<&str>,
        epoch: Option<&str>,
        object_type: Option<&str>,
        country_code: Option<&str>,
        limit: Option<u32>,
        orderby: Option<&str>,
    ) -> Result<Vec<GPRecord>, BraheError> {
        self.runtime
            .block_on(self.inner.gp(
                norad_cat_id,
                object_name,
                object_id,
                epoch,
                object_type,
                country_code,
                limit,
                orderby,
            ))
            .map_err(Into::into)
    }

    /// Query GP data and return as SGP propagators.
    ///
    /// # Arguments
    ///
    /// * `step_size` - Propagator step size in seconds
    /// * `norad_cat_id` - Filter by NORAD catalog ID
    /// * `object_name` - Filter by object name
    /// * `limit` - Maximum number of results
    ///
    /// # Returns
    ///
    /// A vector of SGP propagators.
    pub fn gp_as_propagators(
        &self,
        step_size: f64,
        norad_cat_id: Option<u32>,
        object_name: Option<&str>,
        limit: Option<u32>,
    ) -> Result<Vec<SGPPropagator>, BraheError> {
        self.runtime.block_on(self.inner.gp_as_propagators(
            step_size,
            norad_cat_id,
            object_name,
            limit,
        ))
    }

    /// Query SATCAT (Satellite Catalog) data.
    ///
    /// Returns satellite catalog entries.
    ///
    /// # Arguments
    ///
    /// * `norad_cat_id` - Filter by NORAD catalog ID
    /// * `satname` - Filter by satellite name
    /// * `intldes` - Filter by international designator
    /// * `object_type` - Filter by type
    /// * `country` - Filter by country
    /// * `launch` - Filter by launch date
    /// * `current` - Filter by current status (Y/N)
    /// * `limit` - Maximum results
    /// * `orderby` - Field to order by
    ///
    /// # Returns
    ///
    /// A vector of SATCAT records.
    #[allow(clippy::too_many_arguments)]
    pub fn satcat(
        &self,
        norad_cat_id: Option<u32>,
        satname: Option<&str>,
        intldes: Option<&str>,
        object_type: Option<&str>,
        country: Option<&str>,
        launch: Option<&str>,
        current: Option<&str>,
        limit: Option<u32>,
        orderby: Option<&str>,
    ) -> Result<Vec<SATCATRecord>, BraheError> {
        self.runtime
            .block_on(self.inner.satcat(
                norad_cat_id,
                satname,
                intldes,
                object_type,
                country,
                launch,
                current,
                limit,
                orderby,
            ))
            .map_err(Into::into)
    }

    /// Query TLE (Two-Line Element) data.
    ///
    /// Note: This class is deprecated. Use gp() instead.
    ///
    /// # Arguments
    ///
    /// * `norad_cat_id` - Filter by NORAD catalog ID
    /// * `object_name` - Filter by object name
    /// * `epoch` - Filter by epoch
    /// * `limit` - Maximum results
    /// * `orderby` - Field to order by
    ///
    /// # Returns
    ///
    /// A vector of TLE records.
    pub fn tle(
        &self,
        norad_cat_id: Option<u32>,
        object_name: Option<&str>,
        epoch: Option<&str>,
        limit: Option<u32>,
        orderby: Option<&str>,
    ) -> Result<Vec<TLERecord>, BraheError> {
        self.runtime
            .block_on(
                self.inner
                    .tle(norad_cat_id, object_name, epoch, limit, orderby),
            )
            .map_err(Into::into)
    }

    /// Query decay data.
    ///
    /// Returns predicted and actual re-entry information.
    ///
    /// # Arguments
    ///
    /// * `norad_cat_id` - Filter by NORAD catalog ID
    /// * `decay_epoch` - Filter by decay epoch
    /// * `country` - Filter by country
    /// * `limit` - Maximum results
    ///
    /// # Returns
    ///
    /// A vector of decay records.
    pub fn decay(
        &self,
        norad_cat_id: Option<u32>,
        decay_epoch: Option<&str>,
        country: Option<&str>,
        limit: Option<u32>,
    ) -> Result<Vec<DecayRecord>, BraheError> {
        self.runtime
            .block_on(self.inner.decay(norad_cat_id, decay_epoch, country, limit))
            .map_err(Into::into)
    }

    /// Query TIP (Tracking and Impact Prediction) data.
    ///
    /// Returns tracking and impact prediction messages.
    ///
    /// # Arguments
    ///
    /// * `norad_cat_id` - Filter by NORAD catalog ID
    /// * `decay_epoch` - Filter by decay epoch
    /// * `high_interest` - Filter by high interest flag
    /// * `limit` - Maximum results
    ///
    /// # Returns
    ///
    /// A vector of TIP records.
    pub fn tip(
        &self,
        norad_cat_id: Option<u32>,
        decay_epoch: Option<&str>,
        high_interest: Option<&str>,
        limit: Option<u32>,
    ) -> Result<Vec<TIPRecord>, BraheError> {
        self.runtime
            .block_on(
                self.inner
                    .tip(norad_cat_id, decay_epoch, high_interest, limit),
            )
            .map_err(Into::into)
    }

    /// Query public CDM (Conjunction Data Message) data.
    ///
    /// Returns public conjunction data.
    ///
    /// # Arguments
    ///
    /// * `sat1_norad_cat_id` - Filter by SAT1 NORAD catalog ID
    /// * `sat2_norad_cat_id` - Filter by SAT2 NORAD catalog ID
    /// * `tca` - Filter by time of closest approach
    /// * `limit` - Maximum results
    ///
    /// # Returns
    ///
    /// A vector of CDM records.
    pub fn cdm_public(
        &self,
        sat1_norad_cat_id: Option<u32>,
        sat2_norad_cat_id: Option<u32>,
        tca: Option<&str>,
        limit: Option<u32>,
    ) -> Result<Vec<CDMPublicRecord>, BraheError> {
        self.runtime
            .block_on(
                self.inner
                    .cdm_public(sat1_norad_cat_id, sat2_norad_cat_id, tca, limit),
            )
            .map_err(Into::into)
    }

    /// Query boxscore (catalog statistics) data.
    ///
    /// Returns catalog summary statistics by country.
    ///
    /// # Arguments
    ///
    /// * `country` - Filter by country code
    ///
    /// # Returns
    ///
    /// A vector of boxscore records.
    pub fn boxscore(&self, country: Option<&str>) -> Result<Vec<BoxscoreRecord>, BraheError> {
        self.runtime
            .block_on(self.inner.boxscore(country))
            .map_err(Into::into)
    }

    /// Query launch site data.
    ///
    /// Returns launch facility information.
    ///
    /// # Arguments
    ///
    /// * `site_code` - Filter by site code
    ///
    /// # Returns
    ///
    /// A vector of launch site records.
    pub fn launch_site(
        &self,
        site_code: Option<&str>,
    ) -> Result<Vec<LaunchSiteRecord>, BraheError> {
        self.runtime
            .block_on(self.inner.launch_site(site_code))
            .map_err(Into::into)
    }

    /// Query SATCAT change data.
    ///
    /// Returns information about catalog changes.
    ///
    /// # Arguments
    ///
    /// * `norad_cat_id` - Filter by NORAD catalog ID
    /// * `change_made` - Filter by change timestamp
    /// * `limit` - Maximum results
    ///
    /// # Returns
    ///
    /// A vector of SATCAT change records.
    pub fn satcat_change(
        &self,
        norad_cat_id: Option<u32>,
        change_made: Option<&str>,
        limit: Option<u32>,
    ) -> Result<Vec<SATCATChangeRecord>, BraheError> {
        self.runtime
            .block_on(self.inner.satcat_change(norad_cat_id, change_made, limit))
            .map_err(Into::into)
    }

    /// Query SATCAT debut (new objects) data.
    ///
    /// Returns information about new catalog entries.
    ///
    /// # Arguments
    ///
    /// * `debut` - Filter by debut date
    /// * `object_type` - Filter by type
    /// * `country` - Filter by country
    /// * `limit` - Maximum results
    ///
    /// # Returns
    ///
    /// A vector of SATCAT debut records.
    pub fn satcat_debut(
        &self,
        debut: Option<&str>,
        object_type: Option<&str>,
        country: Option<&str>,
        limit: Option<u32>,
    ) -> Result<Vec<SATCATDebutRecord>, BraheError> {
        self.runtime
            .block_on(self.inner.satcat_debut(debut, object_type, country, limit))
            .map_err(Into::into)
    }

    /// Query announcement data.
    ///
    /// Returns Space-Track announcements.
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum results
    ///
    /// # Returns
    ///
    /// A vector of announcement records.
    pub fn announcement(&self, limit: Option<u32>) -> Result<Vec<AnnouncementRecord>, BraheError> {
        self.runtime
            .block_on(self.inner.announcement(limit))
            .map_err(Into::into)
    }

    // ========================================================================
    // New Query Builder API
    // ========================================================================

    /// Execute a query using the new SpaceTrackQuery builder.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The record type to deserialize the response into
    ///
    /// # Arguments
    ///
    /// * `query` - The SpaceTrackQuery to execute
    ///
    /// # Returns
    ///
    /// A vector of records of type T.
    pub fn execute_query<T: DeserializeOwned>(
        &self,
        query: &SpaceTrackQuery,
    ) -> Result<Vec<T>, BraheError> {
        self.runtime
            .block_on(self.inner.execute_query(query))
            .map_err(Into::into)
    }

    /// Create a query builder for GP (General Perturbations) records.
    pub fn gp_query(&self) -> BlockingSpaceTrackQueryBuilder<'_, GPRecord> {
        BlockingSpaceTrackQueryBuilder::new(self, SpaceTrackQuery::gp())
    }

    /// Create a query builder for TLE records.
    pub fn tle_query(&self) -> BlockingSpaceTrackQueryBuilder<'_, TLERecord> {
        BlockingSpaceTrackQueryBuilder::new(self, SpaceTrackQuery::tle())
    }

    /// Create a query builder for SATCAT (Satellite Catalog) records.
    pub fn satcat_query(&self) -> BlockingSpaceTrackQueryBuilder<'_, SATCATRecord> {
        BlockingSpaceTrackQueryBuilder::new(self, SpaceTrackQuery::satcat())
    }

    /// Create a query builder for OMM records.
    pub fn omm_query(&self) -> BlockingSpaceTrackQueryBuilder<'_, OMMRecord> {
        BlockingSpaceTrackQueryBuilder::new(self, SpaceTrackQuery::omm())
    }

    /// Create a query builder for GP History records.
    pub fn gp_history_query(&self) -> BlockingSpaceTrackQueryBuilder<'_, GPHistoryRecord> {
        BlockingSpaceTrackQueryBuilder::new(self, SpaceTrackQuery::gp_history())
    }

    /// Create a query builder for Decay records.
    pub fn decay_query(&self) -> BlockingSpaceTrackQueryBuilder<'_, DecayRecord> {
        BlockingSpaceTrackQueryBuilder::new(self, SpaceTrackQuery::decay())
    }

    /// Create a query builder for TIP records.
    pub fn tip_query(&self) -> BlockingSpaceTrackQueryBuilder<'_, TIPRecord> {
        BlockingSpaceTrackQueryBuilder::new(self, SpaceTrackQuery::tip())
    }

    /// Create a query builder for CDM Public records.
    pub fn cdm_public_query(&self) -> BlockingSpaceTrackQueryBuilder<'_, CDMPublicRecord> {
        BlockingSpaceTrackQueryBuilder::new(self, SpaceTrackQuery::cdm_public())
    }

    /// Create a query builder for Boxscore records.
    pub fn boxscore_query(&self) -> BlockingSpaceTrackQueryBuilder<'_, BoxscoreRecord> {
        BlockingSpaceTrackQueryBuilder::new(self, SpaceTrackQuery::boxscore())
    }

    /// Create a query builder for Launch Site records.
    pub fn launch_site_query(&self) -> BlockingSpaceTrackQueryBuilder<'_, LaunchSiteRecord> {
        BlockingSpaceTrackQueryBuilder::new(self, SpaceTrackQuery::launch_site())
    }

    /// Create a query builder for SATCAT Change records.
    pub fn satcat_change_query(&self) -> BlockingSpaceTrackQueryBuilder<'_, SATCATChangeRecord> {
        BlockingSpaceTrackQueryBuilder::new(self, SpaceTrackQuery::satcat_change())
    }

    /// Create a query builder for SATCAT Debut records.
    pub fn satcat_debut_query(&self) -> BlockingSpaceTrackQueryBuilder<'_, SATCATDebutRecord> {
        BlockingSpaceTrackQueryBuilder::new(self, SpaceTrackQuery::satcat_debut())
    }

    /// Create a query builder for Announcement records.
    pub fn announcement_query(&self) -> BlockingSpaceTrackQueryBuilder<'_, AnnouncementRecord> {
        BlockingSpaceTrackQueryBuilder::new(self, SpaceTrackQuery::announcement())
    }

    /// Create a generic query builder for any endpoint.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The record type to deserialize the response into
    ///
    /// # Arguments
    ///
    /// * `controller` - The controller name (e.g., "basicspacedata")
    /// * `class` - The class name (e.g., "gp")
    pub fn query_builder<T: DeserializeOwned>(
        &self,
        controller: &str,
        class: &str,
    ) -> BlockingSpaceTrackQueryBuilder<'_, T> {
        BlockingSpaceTrackQueryBuilder::new(self, SpaceTrackQuery::new(controller, class))
    }
}

// ============================================================================
// BlockingSpaceTrackQueryBuilder - Fluent query builder for blocking client
// ============================================================================

/// Fluent query builder for blocking SpaceTrack API requests.
///
/// This is the synchronous equivalent of `SpaceTrackQueryBuilder`.
pub struct BlockingSpaceTrackQueryBuilder<'a, T> {
    client: &'a BlockingSpaceTrackClient,
    query: SpaceTrackQuery,
    _phantom: PhantomData<T>,
}

impl<'a, T: DeserializeOwned> BlockingSpaceTrackQueryBuilder<'a, T> {
    /// Create a new query builder.
    pub(crate) fn new(client: &'a BlockingSpaceTrackClient, query: SpaceTrackQuery) -> Self {
        Self {
            client,
            query,
            _phantom: PhantomData,
        }
    }

    /// Add a predicate filter to the query.
    pub fn filter(mut self, predicate: SpaceTrackPredicate) -> Self {
        self.query = self.query.filter(predicate);
        self
    }

    /// Set the maximum number of results to return.
    pub fn limit(mut self, limit: u32) -> Self {
        self.query = self.query.limit(limit);
        self
    }

    /// Set ordering by field with specified direction.
    pub fn order_by(mut self, field: SpaceTrackPredicateBuilder, order: SpaceTrackOrder) -> Self {
        self.query = self.query.order_by(field, order);
        self
    }

    /// Set ordering by field in ascending order.
    pub fn order_by_asc(mut self, field: SpaceTrackPredicateBuilder) -> Self {
        self.query = self.query.order_by_asc(field);
        self
    }

    /// Set ordering by field in descending order.
    pub fn order_by_desc(mut self, field: SpaceTrackPredicateBuilder) -> Self {
        self.query = self.query.order_by_desc(field);
        self
    }

    /// Enable distinct results only.
    pub fn distinct(mut self) -> Self {
        self.query = self.query.distinct();
        self
    }

    /// Execute the query and return records.
    pub fn execute(self) -> Result<Vec<T>, BraheError> {
        self.client.execute_query(&self.query)
    }
}

/// Additional methods for query builders that return TLE-containing records.
impl<'a, T: DeserializeOwned + HasTLEData> BlockingSpaceTrackQueryBuilder<'a, T> {
    /// Execute the query and convert results to SGP propagators.
    pub fn as_sgp_propagators(self, step_size: f64) -> Result<Vec<SGPPropagator>, BraheError> {
        let records: Vec<T> = self.execute()?;
        records
            .iter()
            .map(|r| r.to_sgp_propagator(step_size))
            .collect()
    }

    /// Execute the query and convert results to SGP propagators, skipping errors.
    pub fn as_sgp_propagators_skip_errors(self, step_size: f64) -> Vec<SGPPropagator> {
        match self.execute() {
            Ok(records) => records
                .iter()
                .filter_map(|r| r.to_sgp_propagator(step_size).ok())
                .collect(),
            Err(_) => Vec::new(),
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    // Tests requiring network would go here with #[ignore] or feature flag
    // For now, we just test that the types compile correctly
    use super::*;

    #[test]
    fn test_blocking_client_type_exists() {
        // This test just verifies the types are properly defined
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        // BlockingSpaceTrackClient should be Send + Sync
        assert_send::<BlockingSpaceTrackClient>();
        assert_sync::<BlockingSpaceTrackClient>();
    }
}
