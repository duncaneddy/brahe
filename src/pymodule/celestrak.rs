// Python bindings for the CelestrakClient API client module.

// -- Enum wrappers --

/// Python wrapper for CelestrakQueryType enum
///
/// Type of CelestrakClient query endpoint.
///
/// Attributes:
///     GP: General Perturbations data (gp.php)
///     SUP_GP: Supplemental GP data (sup-gp.php)
///     SATCAT: Satellite Catalog data (satcat/records.php)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     qt = bh.celestrak.CelestrakQueryType.GP
///     print(qt)  # "gp"
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "CelestrakQueryType")]
#[derive(Clone)]
pub struct PyCelestrakQueryType {
    pub(crate) value: celestrak::CelestrakQueryType,
}

#[pymethods]
impl PyCelestrakQueryType {
    #[classattr]
    #[allow(non_snake_case)]
    fn GP() -> Self {
        PyCelestrakQueryType {
            value: celestrak::CelestrakQueryType::GP,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn SUP_GP() -> Self {
        PyCelestrakQueryType {
            value: celestrak::CelestrakQueryType::SupGP,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn SATCAT() -> Self {
        PyCelestrakQueryType {
            value: celestrak::CelestrakQueryType::SATCAT,
        }
    }

    fn __str__(&self) -> String {
        self.value.as_str().to_string()
    }

    fn __repr__(&self) -> String {
        format!("CelestrakQueryType.{:?}", self.value)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.value == other.value),
            CompareOp::Ne => Ok(self.value != other.value),
            _ => Err(exceptions::PyNotImplementedError::new_err(
                "Comparison not supported",
            )),
        }
    }
}

/// Python wrapper for CelestrakOutputFormat enum
///
/// Output format for CelestrakClient query results.
///
/// Attributes:
///     TLE: Two-Line Element format
///     TWO_LE: 2LE format (no name line)
///     THREE_LE: Three-Line Element format (includes name)
///     XML: XML format
///     KVN: CCSDS Keyword-Value Notation
///     JSON: JSON format
///     JSON_PRETTY: Pretty-printed JSON
///     CSV: CSV format
///
/// Example:
///     ```python
///     import brahe as bh
///
///     fmt = bh.celestrak.CelestrakOutputFormat.JSON
///     print(fmt)  # "JSON"
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "CelestrakOutputFormat")]
#[derive(Clone)]
pub struct PyCelestrakOutputFormat {
    pub(crate) value: celestrak::CelestrakOutputFormat,
}

#[pymethods]
impl PyCelestrakOutputFormat {
    #[classattr]
    #[allow(non_snake_case)]
    fn TLE() -> Self {
        PyCelestrakOutputFormat {
            value: celestrak::CelestrakOutputFormat::Tle,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn TWO_LE() -> Self {
        PyCelestrakOutputFormat {
            value: celestrak::CelestrakOutputFormat::TwoLe,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn THREE_LE() -> Self {
        PyCelestrakOutputFormat {
            value: celestrak::CelestrakOutputFormat::ThreeLe,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn XML() -> Self {
        PyCelestrakOutputFormat {
            value: celestrak::CelestrakOutputFormat::Xml,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn KVN() -> Self {
        PyCelestrakOutputFormat {
            value: celestrak::CelestrakOutputFormat::Kvn,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn JSON() -> Self {
        PyCelestrakOutputFormat {
            value: celestrak::CelestrakOutputFormat::Json,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn JSON_PRETTY() -> Self {
        PyCelestrakOutputFormat {
            value: celestrak::CelestrakOutputFormat::JsonPretty,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn CSV() -> Self {
        PyCelestrakOutputFormat {
            value: celestrak::CelestrakOutputFormat::Csv,
        }
    }

    fn __str__(&self) -> String {
        self.value.as_str().to_string()
    }

    fn __repr__(&self) -> String {
        format!("CelestrakOutputFormat.{:?}", self.value)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.value == other.value),
            CompareOp::Ne => Ok(self.value != other.value),
            _ => Err(exceptions::PyNotImplementedError::new_err(
                "Comparison not supported",
            )),
        }
    }
}

/// Python wrapper for SupGPSource enum
///
/// Supplemental GP data source identifier.
///
/// Attributes:
///     SPACEX: SpaceX operator-provided ephemerides
///     SPACEX_SUP: SpaceX extended ephemerides
///     PLANET: Planet Labs operator-provided ephemerides
///     ONEWEB: OneWeb operator-provided ephemerides
///     STARLINK: Starlink ephemerides
///     STARLINK_SUP: Starlink extended ephemerides
///     GEO: GEO protected zone supplemental data
///     GPS: GPS operational constellation
///     GLONASS: GLONASS operational constellation
///     METEOSAT: Meteosat supplemental data
///     INTELSAT: Intelsat supplemental data
///     SES: SES supplemental data
///     IRIDIUM: Iridium operator-provided ephemerides
///     IRIDIUM_NEXT: Iridium NEXT extended ephemerides
///     ORBCOMM: Orbcomm supplemental data
///     GLOBALSTAR: Globalstar supplemental data
///     SWARM_TECHNOLOGIES: Swarm supplemental data
///     AMATEUR: Amateur radio satellites
///     CELESTRAK: CelestrakClient special supplemental data
///     KUIPER: Kuiper operator-provided ephemerides
///
/// Example:
///     ```python
///     import brahe as bh
///
///     source = bh.celestrak.SupGPSource.SPACEX
///     print(source)  # "spacex"
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "SupGPSource")]
#[derive(Clone)]
pub struct PySupGPSource {
    pub(crate) value: celestrak::SupGPSource,
}

#[pymethods]
impl PySupGPSource {
    #[classattr]
    #[allow(non_snake_case)]
    fn SPACEX() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::SpaceX,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn SPACEX_SUP() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::SpaceXSup,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn PLANET() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::Planet,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn ONEWEB() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::OneWeb,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn STARLINK() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::Starlink,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn STARLINK_SUP() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::StarlinkSup,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn GEO() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::Geo,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn GPS() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::Gps,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn GLONASS() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::Glonass,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn METEOSAT() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::Meteosat,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn INTELSAT() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::Intelsat,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn SES() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::Ses,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn IRIDIUM() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::Iridium,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn IRIDIUM_NEXT() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::IridiumNext,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn ORBCOMM() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::Orbcomm,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn GLOBALSTAR() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::Globalstar,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn SWARM_TECHNOLOGIES() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::SwarmTechnologies,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn AMATEUR() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::Amateur,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn CELESTRAK() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::CelesTrak,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn KUIPER() -> Self {
        PySupGPSource {
            value: celestrak::SupGPSource::Kuiper,
        }
    }

    fn __str__(&self) -> String {
        self.value.as_str().to_string()
    }

    fn __repr__(&self) -> String {
        format!("SupGPSource.{:?}", self.value)
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.value == other.value),
            CompareOp::Ne => Ok(self.value != other.value),
            _ => Err(exceptions::PyNotImplementedError::new_err(
                "Comparison not supported",
            )),
        }
    }
}

// -- Query builder --

/// Fluent query builder for CelestrakClient API queries.
///
/// Constructs URL parameters for the Celestrak REST API endpoints.
/// All builder methods return a new instance for method chaining.
///
/// Use the class attributes ``gp``, ``sup_gp``, and ``satcat`` to create
/// queries targeting the respective endpoints, then chain builder methods.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # GP query for ISS
///     query = bh.celestrak.CelestrakQuery.gp.catnr(25544)
///
///     # GP query for stations group with filtering
///     query = (
///         bh.celestrak.CelestrakQuery.gp
///         .group("stations")
///         .filter("INCLINATION", ">50")
///     )
///
///     # Supplemental GP query
///     query = bh.celestrak.CelestrakQuery.sup_gp.source(bh.celestrak.SupGPSource.SPACEX)
///
///     # SATCAT query
///     query = bh.celestrak.CelestrakQuery.satcat.active(True).payloads(True)
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "CelestrakQuery")]
#[derive(Clone)]
pub struct PyCelestrakQuery {
    pub(crate) inner: celestrak::CelestrakQuery,
}

#[pymethods]
impl PyCelestrakQuery {
    /// GP (General Perturbations) query builder.
    ///
    /// Returns:
    ///     CelestrakQuery: New query targeting the GP endpoint.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     query = bh.celestrak.CelestrakQuery.gp.group("stations")
    ///     ```
    #[classattr]
    fn gp() -> Self {
        PyCelestrakQuery {
            inner: celestrak::CelestrakQuery::gp(),
        }
    }

    /// Supplemental GP query builder.
    ///
    /// Returns:
    ///     CelestrakQuery: New query targeting the Supplemental GP endpoint.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     query = bh.celestrak.CelestrakQuery.sup_gp.source(bh.celestrak.SupGPSource.STARLINK)
    ///     ```
    #[classattr]
    fn sup_gp() -> Self {
        PyCelestrakQuery {
            inner: celestrak::CelestrakQuery::sup_gp(),
        }
    }

    /// SATCAT (Satellite Catalog) query builder.
    ///
    /// Returns:
    ///     CelestrakQuery: New query targeting the SATCAT endpoint.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     query = bh.celestrak.CelestrakQuery.satcat.active(True)
    ///     ```
    #[classattr]
    fn satcat() -> Self {
        PyCelestrakQuery {
            inner: celestrak::CelestrakQuery::satcat(),
        }
    }

    /// Filter by satellite group name.
    ///
    /// Args:
    ///     name (str): Group name (e.g., "stations", "active", "gnss").
    ///
    /// Returns:
    ///     CelestrakQuery: New query with group filter applied.
    fn group(&self, name: &str) -> Self {
        PyCelestrakQuery {
            inner: self.inner.clone().group(name),
        }
    }

    /// Filter by NORAD catalog number.
    ///
    /// Args:
    ///     id (int): NORAD catalog number (e.g., 25544 for ISS).
    ///
    /// Returns:
    ///     CelestrakQuery: New query with CATNR filter applied.
    fn catnr(&self, id: u32) -> Self {
        PyCelestrakQuery {
            inner: self.inner.clone().catnr(id),
        }
    }

    /// Filter by international designator.
    ///
    /// Args:
    ///     intdes (str): International designator (e.g., "1998-067A").
    ///
    /// Returns:
    ///     CelestrakQuery: New query with INTDES filter applied.
    fn intdes(&self, intdes: &str) -> Self {
        PyCelestrakQuery {
            inner: self.inner.clone().intdes(intdes),
        }
    }

    /// Filter by satellite name (substring match).
    ///
    /// Args:
    ///     name (str): Satellite name to search for.
    ///
    /// Returns:
    ///     CelestrakQuery: New query with NAME filter applied.
    fn name_search(&self, name: &str) -> Self {
        PyCelestrakQuery {
            inner: self.inner.clone().name_search(name),
        }
    }

    /// Set a special query parameter.
    ///
    /// Args:
    ///     special (str): Special query value.
    ///
    /// Returns:
    ///     CelestrakQuery: New query with SPECIAL parameter applied.
    fn special(&self, special: &str) -> Self {
        PyCelestrakQuery {
            inner: self.inner.clone().special(special),
        }
    }

    /// Set the supplemental GP data source (SupGP queries only).
    ///
    /// Args:
    ///     source (SupGPSource): The supplemental data source.
    ///
    /// Returns:
    ///     CelestrakQuery: New query with SOURCE parameter applied.
    fn source(&self, source: &PySupGPSource) -> Self {
        PyCelestrakQuery {
            inner: self.inner.clone().source(source.value),
        }
    }

    /// Set the supplemental GP file parameter (SupGP queries only).
    ///
    /// Args:
    ///     file (str): File name parameter.
    ///
    /// Returns:
    ///     CelestrakQuery: New query with FILE parameter applied.
    fn file(&self, file: &str) -> Self {
        PyCelestrakQuery {
            inner: self.inner.clone().file(file),
        }
    }

    /// Filter to payloads only (SATCAT queries only).
    ///
    /// Args:
    ///     enabled (bool): True to include only payloads.
    ///
    /// Returns:
    ///     CelestrakQuery: New query with PAYLOADS flag.
    fn payloads(&self, enabled: bool) -> Self {
        PyCelestrakQuery {
            inner: self.inner.clone().payloads(enabled),
        }
    }

    /// Filter to on-orbit objects only (SATCAT queries only).
    ///
    /// Args:
    ///     enabled (bool): True to include only on-orbit objects.
    ///
    /// Returns:
    ///     CelestrakQuery: New query with ONORBIT flag.
    fn on_orbit(&self, enabled: bool) -> Self {
        PyCelestrakQuery {
            inner: self.inner.clone().on_orbit(enabled),
        }
    }

    /// Filter to active objects only (SATCAT queries only).
    ///
    /// Args:
    ///     enabled (bool): True to include only active objects.
    ///
    /// Returns:
    ///     CelestrakQuery: New query with ACTIVE flag.
    fn active(&self, enabled: bool) -> Self {
        PyCelestrakQuery {
            inner: self.inner.clone().active(enabled),
        }
    }

    /// Limit the maximum number of results (SATCAT queries only).
    ///
    /// Args:
    ///     count (int): Maximum number of records to return.
    ///
    /// Returns:
    ///     CelestrakQuery: New query with MAX parameter.
    fn max(&self, count: u32) -> Self {
        PyCelestrakQuery {
            inner: self.inner.clone().max(count),
        }
    }

    /// Set the output format.
    ///
    /// Args:
    ///     fmt (CelestrakOutputFormat): The desired output format.
    ///
    /// Returns:
    ///     CelestrakQuery: New query with FORMAT parameter.
    #[pyo3(name = "format")]
    fn set_format(&self, fmt: &PyCelestrakOutputFormat) -> Self {
        PyCelestrakQuery {
            inner: self.inner.clone().format(fmt.value),
        }
    }

    /// Add a client-side filter (applied after download).
    ///
    /// Uses SpaceTrack-compatible operator syntax:
    /// - ">50" = greater than 50
    /// - "<0.01" = less than 0.01
    /// - "<>DEBRIS" = not equal to DEBRIS
    /// - "25544--25600" = range 25544 to 25600
    /// - "~~STARLINK" = contains STARLINK (case-insensitive)
    /// - "^NOAA" = starts with NOAA (case-insensitive)
    /// - "25544" = exact match
    ///
    /// Args:
    ///     field (str): Field name (e.g., "INCLINATION", "OBJECT_TYPE").
    ///     value (str): Filter value with optional operator prefix.
    ///
    /// Returns:
    ///     CelestrakQuery: New query with filter added.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     from brahe.spacetrack import operators as op
    ///
    ///     query = (
    ///         bh.celestrak.CelestrakQuery.gp()
    ///         .group("active")
    ///         .filter("OBJECT_TYPE", op.not_equal("DEBRIS"))
    ///         .filter("INCLINATION", op.greater_than("50"))
    ///     )
    ///     ```
    fn filter(&self, field: &str, value: &str) -> Self {
        PyCelestrakQuery {
            inner: self.inner.clone().filter(field, value),
        }
    }

    /// Add a client-side ordering clause (applied after download).
    ///
    /// Args:
    ///     field (str): Field name to sort by.
    ///     ascending (bool): True for ascending order, False for descending.
    ///
    /// Returns:
    ///     CelestrakQuery: New query with ordering added.
    fn order_by(&self, field: &str, ascending: bool) -> Self {
        PyCelestrakQuery {
            inner: self.inner.clone().order_by(field, ascending),
        }
    }

    /// Set a client-side result limit (applied after download and filtering).
    ///
    /// Args:
    ///     count (int): Maximum number of records to return.
    ///
    /// Returns:
    ///     CelestrakQuery: New query with limit set.
    fn limit(&self, count: u32) -> Self {
        PyCelestrakQuery {
            inner: self.inner.clone().limit(count),
        }
    }

    /// Build the URL query string for this query.
    ///
    /// Returns:
    ///     str: URL-encoded query parameters (e.g., "GROUP=stations&FORMAT=JSON").
    fn build_url(&self) -> String {
        self.inner.build_url()
    }

    fn __str__(&self) -> String {
        self.inner.build_url()
    }

    fn __repr__(&self) -> String {
        format!(
            "CelestrakQuery({:?}, \"{}\")",
            self.inner.query_type(),
            self.inner.build_url()
        )
    }
}

// -- Response types --

/// CelestrakClient SATCAT record.
///
/// Contains metadata about a cataloged space object from CelestrakClient's
/// satellite catalog. All fields are optional strings.
///
/// Note: For GP records, use GPRecord (same type as SpaceTrack module).
///
/// Attributes:
///     object_name (str | None): Object name
///     object_id (str | None): International designator
///     norad_cat_id (int | None): NORAD catalog number
///     object_type (str | None): Object type code
///     ops_status_code (str | None): Operational status code
///     owner (str | None): Owner/operator
///     launch_date (str | None): Launch date
///     launch_site (str | None): Launch site
///     decay_date (str | None): Decay date
///     period (str | None): Orbital period (minutes)
///     inclination (str | None): Inclination (degrees)
///     apogee (str | None): Apogee altitude (km)
///     perigee (str | None): Perigee altitude (km)
///     rcs (str | None): Radar cross-section
///     data_status_code (str | None): Data status code
///     orbit_center (str | None): Orbit center
///     orbit_type (str | None): Orbit type
///
/// Example:
///     ```python
///     import brahe as bh
///
///     client = bh.celestrak.CelestrakClient()
///     records = client.get_satcat(catnr=25544)
///     print(records[0].object_name)  # "ISS (ZARYA)"
///     ```
#[pyclass(module = "brahe._brahe", from_py_object)]
#[pyo3(name = "CelestrakSATCATRecord")]
#[derive(Clone)]
pub struct PyCelestrakSATCATRecord {
    inner: celestrak::CelestrakSATCATRecord,
}

#[pymethods]
impl PyCelestrakSATCATRecord {
    #[getter] fn object_name(&self) -> Option<String> { self.inner.object_name.clone() }
    #[getter] fn object_id(&self) -> Option<String> { self.inner.object_id.clone() }
    #[getter] fn norad_cat_id(&self) -> Option<u32> { self.inner.norad_cat_id }
    #[getter] fn object_type(&self) -> Option<String> { self.inner.object_type.clone() }
    #[getter] fn ops_status_code(&self) -> Option<String> { self.inner.ops_status_code.clone() }
    #[getter] fn owner(&self) -> Option<String> { self.inner.owner.clone() }
    #[getter] fn launch_date(&self) -> Option<String> { self.inner.launch_date.clone() }
    #[getter] fn launch_site(&self) -> Option<String> { self.inner.launch_site.clone() }
    #[getter] fn decay_date(&self) -> Option<String> { self.inner.decay_date.clone() }
    #[getter] fn period(&self) -> Option<String> { self.inner.period.clone() }
    #[getter] fn inclination(&self) -> Option<String> { self.inner.inclination.clone() }
    #[getter] fn apogee(&self) -> Option<String> { self.inner.apogee.clone() }
    #[getter] fn perigee(&self) -> Option<String> { self.inner.perigee.clone() }
    #[getter] fn rcs(&self) -> Option<String> { self.inner.rcs.clone() }
    #[getter] fn data_status_code(&self) -> Option<String> { self.inner.data_status_code.clone() }
    #[getter] fn orbit_center(&self) -> Option<String> { self.inner.orbit_center.clone() }
    #[getter] fn orbit_type(&self) -> Option<String> { self.inner.orbit_type.clone() }

    fn __str__(&self) -> String {
        format!(
            "CelestrakSATCATRecord(name={:?}, norad_id={:?})",
            self.inner.object_name, self.inner.norad_cat_id
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

// -- Client --

/// CelestrakClient API client with caching.
///
/// Provides typed query execution for GP, supplemental GP, and SATCAT
/// data from CelestrakClient. No authentication is required. Responses
/// are cached locally to reduce server load.
///
/// Two tiers of API are available:
///
/// **Tier 1 — Compact convenience methods** (most common operations):
///     - ``get_gp()``: Look up GP records by catnr, group, name, or intdes
///     - ``get_sup_gp()``: Look up supplemental GP records
///     - ``get_satcat()``: Look up SATCAT records
///     - ``get_sgp_propagator()``: Get an SGP4 propagator directly
///
/// **Tier 2 — Query builder** (complex queries with filtering/sorting):
///     - ``query()``: Execute any ``CelestrakQuery`` and return typed results
///
/// Args:
///     base_url (str, optional): Custom base URL for testing.
///     cache_max_age (float, optional): Cache TTL in seconds. Default: 21600.0 (6 hours).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     client = bh.celestrak.CelestrakClient()
///
///     # Compact: look up ISS GP data
///     records = client.get_gp(catnr=25544)
///
///     # Compact: get an SGP4 propagator directly
///     prop = client.get_sgp_propagator(catnr=25544, step_size=60.0)
///
///     # Query builder: complex queries with filtering
///     query = (
///         bh.celestrak.CelestrakQuery.gp
///         .group("active")
///         .filter("INCLINATION", ">50")
///         .limit(10)
///     )
///     records = client.query(query)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "CelestrakClient")]
pub struct PyCelestrakClient {
    inner: celestrak::CelestrakClient,
}

#[pymethods]
impl PyCelestrakClient {
    #[new]
    #[pyo3(signature = (base_url=None, cache_max_age=None))]
    fn new(base_url: Option<&str>, cache_max_age: Option<f64>) -> Self {
        let client = match (base_url, cache_max_age) {
            (Some(url), Some(age)) => {
                celestrak::CelestrakClient::with_base_url_and_cache_age(url, age)
            }
            (Some(url), None) => celestrak::CelestrakClient::with_base_url(url),
            (None, Some(age)) => celestrak::CelestrakClient::with_cache_age(age),
            (None, None) => celestrak::CelestrakClient::new(),
        };
        PyCelestrakClient { inner: client }
    }

    // -- Tier 1: Compact convenience methods --

    /// Look up GP records by exactly one identifier.
    ///
    /// Provide exactly one of ``catnr``, ``group``, ``name``, or ``intdes``.
    ///
    /// Args:
    ///     catnr (int, optional): NORAD catalog number (e.g., 25544 for ISS).
    ///     group (str, optional): Satellite group name (e.g., "stations", "active").
    ///     name (str, optional): Satellite name to search for (partial match).
    ///     intdes (str, optional): International designator (e.g., "1998-067A").
    ///
    /// Returns:
    ///     list[GPRecord]: List of matching GP records.
    ///
    /// Raises:
    ///     ValueError: If zero or more than one identifier is provided.
    ///     BraheError: On network, cache, or parse errors.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.celestrak.CelestrakClient()
    ///     records = client.get_gp(catnr=25544)
    ///     records = client.get_gp(group="stations")
    ///     records = client.get_gp(name="ISS")
    ///     records = client.get_gp(intdes="1998-067A")
    ///     ```
    #[pyo3(signature = (*, catnr=None, group=None, name=None, intdes=None))]
    fn get_gp(
        &self,
        catnr: Option<u32>,
        group: Option<&str>,
        name: Option<&str>,
        intdes: Option<&str>,
    ) -> PyResult<Vec<PyGPRecord>> {
        let count = catnr.is_some() as u8
            + group.is_some() as u8
            + name.is_some() as u8
            + intdes.is_some() as u8;
        if count != 1 {
            return Err(exceptions::PyValueError::new_err(
                "Provide exactly one of: catnr, group, name, intdes",
            ));
        }

        let records = if let Some(id) = catnr {
            self.inner.get_gp_by_catnr(id)
        } else if let Some(g) = group {
            self.inner.get_gp_by_group(g)
        } else if let Some(n) = name {
            self.inner.get_gp_by_name(n)
        } else {
            self.inner.get_gp_by_intdes(intdes.unwrap())
        }
        .map_err(|e| BraheError::new_err(e.to_string()))?;

        Ok(records
            .into_iter()
            .map(|r| PyGPRecord { inner: r })
            .collect())
    }

    /// Look up supplemental GP records by source.
    ///
    /// Args:
    ///     source (SupGPSource): The supplemental data source.
    ///
    /// Returns:
    ///     list[GPRecord]: List of GP records from the supplemental source.
    ///
    /// Raises:
    ///     BraheError: On network, cache, or parse errors.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.celestrak.CelestrakClient()
    ///     records = client.get_sup_gp(bh.celestrak.SupGPSource.STARLINK)
    ///     ```
    fn get_sup_gp(&self, source: &PySupGPSource) -> PyResult<Vec<PyGPRecord>> {
        let records = self
            .inner
            .get_sup_gp(source.value)
            .map_err(|e| BraheError::new_err(e.to_string()))?;

        Ok(records
            .into_iter()
            .map(|r| PyGPRecord { inner: r })
            .collect())
    }

    /// Look up SATCAT records.
    ///
    /// At least one parameter must be provided.
    ///
    /// Args:
    ///     catnr (int, optional): NORAD catalog number.
    ///     active (bool, optional): Filter to active objects only.
    ///     payloads (bool, optional): Filter to payloads only.
    ///     on_orbit (bool, optional): Filter to on-orbit objects only.
    ///
    /// Returns:
    ///     list[CelestrakSATCATRecord]: List of matching SATCAT records.
    ///
    /// Raises:
    ///     ValueError: If no parameters are provided.
    ///     BraheError: On network, cache, or parse errors.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.celestrak.CelestrakClient()
    ///     records = client.get_satcat(catnr=25544)
    ///     records = client.get_satcat(active=True, payloads=True)
    ///     ```
    #[pyo3(signature = (*, catnr=None, active=None, payloads=None, on_orbit=None))]
    fn get_satcat(
        &self,
        catnr: Option<u32>,
        active: Option<bool>,
        payloads: Option<bool>,
        on_orbit: Option<bool>,
    ) -> PyResult<Vec<PyCelestrakSATCATRecord>> {
        if catnr.is_none() && active.is_none() && payloads.is_none() && on_orbit.is_none() {
            return Err(exceptions::PyValueError::new_err(
                "Provide at least one of: catnr, active, payloads, on_orbit",
            ));
        }

        let mut query = celestrak::CelestrakQuery::satcat();
        if let Some(id) = catnr {
            query = query.catnr(id);
        }
        if let Some(a) = active {
            query = query.active(a);
        }
        if let Some(p) = payloads {
            query = query.payloads(p);
        }
        if let Some(o) = on_orbit {
            query = query.on_orbit(o);
        }

        let records = self
            .inner
            .query_satcat(&query)
            .map_err(|e| BraheError::new_err(e.to_string()))?;

        Ok(records
            .into_iter()
            .map(|r| PyCelestrakSATCATRecord { inner: r })
            .collect())
    }

    /// Look up a satellite and return an SGP4 propagator.
    ///
    /// Queries GP data for the given catalog number and creates an
    /// SGPPropagator from the first result.
    ///
    /// Args:
    ///     catnr (int): NORAD catalog number.
    ///     step_size (float): Propagator step size in seconds. Default: 60.0.
    ///
    /// Returns:
    ///     SGPPropagator: Ready-to-use SGP4 propagator.
    ///
    /// Raises:
    ///     BraheError: If no records found or propagator creation fails.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.celestrak.CelestrakClient()
    ///     propagator = client.get_sgp_propagator(catnr=25544, step_size=60.0)
    ///     ```
    #[pyo3(signature = (*, catnr, step_size=60.0))]
    fn get_sgp_propagator(&self, catnr: u32, step_size: f64) -> PyResult<PySGPPropagator> {
        let propagator = self
            .inner
            .get_sgp_propagator_by_catnr(catnr, step_size)
            .map_err(|e| BraheError::new_err(e.to_string()))?;
        Ok(PySGPPropagator { propagator })
    }

    // -- Tier 2: Query builder methods --

    /// Execute a query and return typed results.
    ///
    /// Dispatches to the appropriate handler based on query type:
    /// GP and SupGP queries return ``list[GPRecord]``, SATCAT queries
    /// return ``list[CelestrakSATCATRecord]``.
    ///
    /// Args:
    ///     query (CelestrakQuery): The query to execute.
    ///
    /// Returns:
    ///     list[GPRecord] | list[CelestrakSATCATRecord]: Typed records based on query type.
    ///
    /// Raises:
    ///     BraheError: On network, cache, or parse errors.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.celestrak.CelestrakClient()
    ///
    ///     # GP query with filtering
    ///     query = (
    ///         bh.celestrak.CelestrakQuery.gp
    ///         .group("active")
    ///         .filter("INCLINATION", ">50")
    ///         .limit(10)
    ///     )
    ///     records = client.query(query)
    ///
    ///     # SATCAT query
    ///     query = bh.celestrak.CelestrakQuery.satcat.active(True)
    ///     records = client.query(query)
    ///     ```
    fn query<'py>(
        &self,
        py: Python<'py>,
        query: &PyCelestrakQuery,
    ) -> PyResult<Py<PyAny>> {
        match query.inner.query_type() {
            celestrak::CelestrakQueryType::GP | celestrak::CelestrakQueryType::SupGP => {
                let records = self
                    .inner
                    .query_gp(&query.inner)
                    .map_err(|e| BraheError::new_err(e.to_string()))?;

                let py_records: Vec<PyGPRecord> = records
                    .into_iter()
                    .map(|r| PyGPRecord { inner: r })
                    .collect();

                py_records.into_py_any(py)
            }
            celestrak::CelestrakQueryType::SATCAT => {
                let records = self
                    .inner
                    .query_satcat(&query.inner)
                    .map_err(|e| BraheError::new_err(e.to_string()))?;

                let py_records: Vec<PyCelestrakSATCATRecord> = records
                    .into_iter()
                    .map(|r| PyCelestrakSATCATRecord { inner: r })
                    .collect();

                py_records.into_py_any(py)
            }
        }
    }

    // -- Raw/download methods --

    /// Execute a query and return the raw response body.
    ///
    /// Args:
    ///     query (CelestrakQuery): The query to execute.
    ///
    /// Returns:
    ///     str: Raw response body in the requested format.
    ///
    /// Raises:
    ///     BraheError: On network or cache errors.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.celestrak.CelestrakClient()
    ///     query = (
    ///         bh.celestrak.CelestrakQuery.gp
    ///         .group("stations")
    ///         .format(bh.celestrak.CelestrakOutputFormat.THREE_LE)
    ///     )
    ///     tle_data = client.query_raw(query)
    ///     ```
    fn query_raw(&self, query: &PyCelestrakQuery) -> PyResult<String> {
        self.inner
            .query_raw(&query.inner)
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Execute a query and save the response to a file.
    ///
    /// Args:
    ///     query (CelestrakQuery): The query to execute.
    ///     filepath (str): Path to save the response to.
    ///
    /// Raises:
    ///     BraheError: On network, cache, or I/O errors.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.celestrak.CelestrakClient()
    ///     query = (
    ///         bh.celestrak.CelestrakQuery.gp
    ///         .group("stations")
    ///         .format(bh.celestrak.CelestrakOutputFormat.THREE_LE)
    ///     )
    ///     client.download(query, "stations.3le")
    ///     ```
    fn download(&self, query: &PyCelestrakQuery, filepath: &str) -> PyResult<()> {
        self.inner
            .download(&query.inner, Path::new(filepath))
            .map_err(|e| BraheError::new_err(e.to_string()))
    }
}
