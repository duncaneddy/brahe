// Python bindings for the SpaceTrack API client module.

// -- Enum wrappers --

/// Python wrapper for RequestController enum
///
/// SpaceTrack API request controller. Most queries use BasicSpaceData.
///
/// Attributes:
///     BASIC_SPACE_DATA: Basic space data controller (most common)
///     EXPANDED_SPACE_DATA: Expanded space data controller
///     FILE_SHARE: File share controller
///     SP_EPHEMERIS: SP ephemeris controller
///     PUBLIC_FILES: Public files controller
///
/// Example:
///     ```python
///     import brahe as bh
///
///     controller = bh.RequestController.BASIC_SPACE_DATA
///     print(controller)  # "BasicSpaceData"
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "RequestController")]
#[derive(Clone)]
pub struct PyRequestController {
    pub(crate) value: spacetrack::RequestController,
}

#[pymethods]
impl PyRequestController {
    #[classattr]
    #[allow(non_snake_case)]
    fn BASIC_SPACE_DATA() -> Self {
        PyRequestController {
            value: spacetrack::RequestController::BasicSpaceData,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn EXPANDED_SPACE_DATA() -> Self {
        PyRequestController {
            value: spacetrack::RequestController::ExpandedSpaceData,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn FILE_SHARE() -> Self {
        PyRequestController {
            value: spacetrack::RequestController::FileShare,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn SP_EPHEMERIS() -> Self {
        PyRequestController {
            value: spacetrack::RequestController::SPEphemeris,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn PUBLIC_FILES() -> Self {
        PyRequestController {
            value: spacetrack::RequestController::PublicFiles,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.value)
    }

    fn __repr__(&self) -> String {
        format!("RequestController.{:?}", self.value)
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

/// Python wrapper for RequestClass enum
///
/// SpaceTrack API request class. Determines which type of data to query.
///
/// Attributes:
///     GP: General Perturbations (OMM) data - current orbital elements
///     GP_HISTORY: GP history - historical orbital element sets
///     SATCAT: Satellite Catalog - object metadata
///     SATCAT_CHANGE: SATCAT changes
///     SATCAT_DEBUT: Newly cataloged objects
///     DECAY: Decay predictions and actual decay data
///     TIP: Tracking and Impact Prediction messages
///     CDM_PUBLIC: Public Conjunction Data Messages
///     BOXSCORE: Boxscore summary statistics
///     ANNOUNCEMENT: Space-Track announcements
///     LAUNCH_SITE: Launch site information
///
/// Example:
///     ```python
///     import brahe as bh
///
///     query = bh.SpaceTrackQuery(bh.RequestClass.GP)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "RequestClass")]
#[derive(Clone)]
pub struct PyRequestClass {
    pub(crate) value: spacetrack::RequestClass,
}

#[pymethods]
impl PyRequestClass {
    #[classattr]
    #[allow(non_snake_case)]
    fn GP() -> Self {
        PyRequestClass {
            value: spacetrack::RequestClass::GP,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn GP_HISTORY() -> Self {
        PyRequestClass {
            value: spacetrack::RequestClass::GPHistory,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn SATCAT() -> Self {
        PyRequestClass {
            value: spacetrack::RequestClass::SATCAT,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn SATCAT_CHANGE() -> Self {
        PyRequestClass {
            value: spacetrack::RequestClass::SATCATChange,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn SATCAT_DEBUT() -> Self {
        PyRequestClass {
            value: spacetrack::RequestClass::SATCATDebut,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn DECAY() -> Self {
        PyRequestClass {
            value: spacetrack::RequestClass::Decay,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn TIP() -> Self {
        PyRequestClass {
            value: spacetrack::RequestClass::TIP,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn CDM_PUBLIC() -> Self {
        PyRequestClass {
            value: spacetrack::RequestClass::CDMPublic,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn BOXSCORE() -> Self {
        PyRequestClass {
            value: spacetrack::RequestClass::Boxscore,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn ANNOUNCEMENT() -> Self {
        PyRequestClass {
            value: spacetrack::RequestClass::Announcement,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn LAUNCH_SITE() -> Self {
        PyRequestClass {
            value: spacetrack::RequestClass::LaunchSite,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.value)
    }

    fn __repr__(&self) -> String {
        format!("RequestClass.{:?}", self.value)
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

/// Python wrapper for SortOrder enum
///
/// Sort order for SpaceTrack query results.
///
/// Attributes:
///     ASC: Ascending order (smallest/earliest first)
///     DESC: Descending order (largest/latest first)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     query = bh.SpaceTrackQuery(bh.RequestClass.GP).order_by("EPOCH", bh.SortOrder.DESC)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "SortOrder")]
#[derive(Clone)]
pub struct PySortOrder {
    pub(crate) value: spacetrack::SortOrder,
}

#[pymethods]
impl PySortOrder {
    #[classattr]
    #[allow(non_snake_case)]
    fn ASC() -> Self {
        PySortOrder {
            value: spacetrack::SortOrder::Asc,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn DESC() -> Self {
        PySortOrder {
            value: spacetrack::SortOrder::Desc,
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.value)
    }

    fn __repr__(&self) -> String {
        format!("SortOrder.{:?}", self.value)
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

/// Python wrapper for OutputFormat enum
///
/// Output format for SpaceTrack query results.
///
/// Attributes:
///     JSON: JSON format (default)
///     XML: XML format
///     HTML: HTML format
///     CSV: CSV format
///     TLE: Two-Line Element format
///     THREE_LE: Three-Line Element format (includes object name)
///     KVN: CCSDS Keyword-Value Notation format
///
/// Example:
///     ```python
///     import brahe as bh
///
///     query = bh.SpaceTrackQuery(bh.RequestClass.GP).format(bh.OutputFormat.TLE)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OutputFormat")]
#[derive(Clone)]
pub struct PyOutputFormat {
    pub(crate) value: spacetrack::OutputFormat,
}

#[pymethods]
impl PyOutputFormat {
    #[classattr]
    #[allow(non_snake_case)]
    fn JSON() -> Self {
        PyOutputFormat {
            value: spacetrack::OutputFormat::JSON,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn XML() -> Self {
        PyOutputFormat {
            value: spacetrack::OutputFormat::XML,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn HTML() -> Self {
        PyOutputFormat {
            value: spacetrack::OutputFormat::HTML,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn CSV() -> Self {
        PyOutputFormat {
            value: spacetrack::OutputFormat::CSV,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn TLE() -> Self {
        PyOutputFormat {
            value: spacetrack::OutputFormat::TLE,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn THREE_LE() -> Self {
        PyOutputFormat {
            value: spacetrack::OutputFormat::ThreeLe,
        }
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn KVN() -> Self {
        PyOutputFormat {
            value: spacetrack::OutputFormat::KVN,
        }
    }

    fn __str__(&self) -> String {
        match self.value {
            spacetrack::OutputFormat::ThreeLe => "3LE".to_string(),
            _ => format!("{:?}", self.value),
        }
    }

    fn __repr__(&self) -> String {
        match self.value {
            spacetrack::OutputFormat::ThreeLe => "OutputFormat.3LE".to_string(),
            _ => format!("OutputFormat.{:?}", self.value),
        }
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

/// Fluent query builder for SpaceTrack API queries.
///
/// Constructs URL path strings for the Space-Track.org REST API.
/// All builder methods return self for method chaining.
///
/// Args:
///     request_class (RequestClass): The type of data to query.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     query = (
///         bh.SpaceTrackQuery(bh.RequestClass.GP)
///         .filter("NORAD_CAT_ID", "25544")
///         .order_by("EPOCH", bh.SortOrder.DESC)
///         .limit(1)
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "SpaceTrackQuery")]
#[derive(Clone)]
pub struct PySpaceTrackQuery {
    inner: spacetrack::SpaceTrackQuery,
}

#[pymethods]
impl PySpaceTrackQuery {
    #[new]
    fn new(request_class: &PyRequestClass) -> Self {
        PySpaceTrackQuery {
            inner: spacetrack::SpaceTrackQuery::new(request_class.value),
        }
    }

    /// Override the default controller for this query.
    ///
    /// Args:
    ///     controller (RequestController): The controller to use.
    ///
    /// Returns:
    ///     SpaceTrackQuery: Self for method chaining.
    fn controller(&self, controller: &PyRequestController) -> Self {
        PySpaceTrackQuery {
            inner: self.inner.clone().controller(controller.value),
        }
    }

    /// Add a filter predicate to the query.
    ///
    /// Args:
    ///     field (str): The field name (e.g., "NORAD_CAT_ID", "EPOCH").
    ///     value (str): The filter value, optionally with operator prefix.
    ///
    /// Returns:
    ///     SpaceTrackQuery: Self for method chaining.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     from brahe.spacetrack import operators as op
    ///
    ///     query = (
    ///         bh.SpaceTrackQuery(bh.RequestClass.GP)
    ///         .filter("NORAD_CAT_ID", "25544")
    ///         .filter("EPOCH", op.greater_than("2024-01-01"))
    ///     )
    ///     ```
    fn filter(&self, field: &str, value: &str) -> Self {
        PySpaceTrackQuery {
            inner: self.inner.clone().filter(field, value),
        }
    }

    /// Add an ordering clause to the query.
    ///
    /// Args:
    ///     field (str): The field to sort by.
    ///     order (SortOrder): The sort direction.
    ///
    /// Returns:
    ///     SpaceTrackQuery: Self for method chaining.
    fn order_by(&self, field: &str, order: &PySortOrder) -> Self {
        PySpaceTrackQuery {
            inner: self.inner.clone().order_by(field, order.value),
        }
    }

    /// Set the maximum number of results to return.
    ///
    /// Args:
    ///     count (int): Maximum number of records.
    ///
    /// Returns:
    ///     SpaceTrackQuery: Self for method chaining.
    fn limit(&self, count: u32) -> Self {
        PySpaceTrackQuery {
            inner: self.inner.clone().limit(count),
        }
    }

    /// Set the maximum number of results and an offset.
    ///
    /// Args:
    ///     count (int): Maximum number of records.
    ///     offset (int): Number of records to skip.
    ///
    /// Returns:
    ///     SpaceTrackQuery: Self for method chaining.
    fn limit_offset(&self, count: u32, offset: u32) -> Self {
        PySpaceTrackQuery {
            inner: self.inner.clone().limit_offset(count, offset),
        }
    }

    /// Set the output format for query results.
    ///
    /// Args:
    ///     fmt (OutputFormat): The desired output format.
    ///
    /// Returns:
    ///     SpaceTrackQuery: Self for method chaining.
    #[pyo3(name = "format")]
    fn set_format(&self, fmt: &PyOutputFormat) -> Self {
        PySpaceTrackQuery {
            inner: self.inner.clone().format(fmt.value),
        }
    }

    /// Specify which fields to include in the response.
    ///
    /// Args:
    ///     fields (list[str]): List of field names to include.
    ///
    /// Returns:
    ///     SpaceTrackQuery: Self for method chaining.
    fn predicates_filter(&self, fields: Vec<String>) -> Self {
        let field_refs: Vec<&str> = fields.iter().map(|s| s.as_str()).collect();
        PySpaceTrackQuery {
            inner: self.inner.clone().predicates_filter(&field_refs),
        }
    }

    /// Enable or disable metadata in the response.
    ///
    /// Args:
    ///     enabled (bool): Whether to include metadata.
    ///
    /// Returns:
    ///     SpaceTrackQuery: Self for method chaining.
    fn metadata(&self, enabled: bool) -> Self {
        PySpaceTrackQuery {
            inner: self.inner.clone().metadata(enabled),
        }
    }

    /// Enable or disable distinct results.
    ///
    /// Args:
    ///     enabled (bool): Whether to return distinct results.
    ///
    /// Returns:
    ///     SpaceTrackQuery: Self for method chaining.
    fn distinct(&self, enabled: bool) -> Self {
        PySpaceTrackQuery {
            inner: self.inner.clone().distinct(enabled),
        }
    }

    /// Enable or disable empty result return.
    ///
    /// Args:
    ///     enabled (bool): Whether to allow empty results.
    ///
    /// Returns:
    ///     SpaceTrackQuery: Self for method chaining.
    fn empty_result(&self, enabled: bool) -> Self {
        PySpaceTrackQuery {
            inner: self.inner.clone().empty_result(enabled),
        }
    }

    /// Set a favorites filter for the query.
    ///
    /// Args:
    ///     favorites (str): The favorites identifier.
    ///
    /// Returns:
    ///     SpaceTrackQuery: Self for method chaining.
    fn favorites(&self, favorites: &str) -> Self {
        PySpaceTrackQuery {
            inner: self.inner.clone().favorites(favorites),
        }
    }

    /// Build the URL path string for this query.
    ///
    /// Returns:
    ///     str: The URL path string.
    fn build(&self) -> String {
        self.inner.build()
    }

    fn __str__(&self) -> String {
        self.inner.build()
    }

    fn __repr__(&self) -> String {
        format!("SpaceTrackQuery(\"{}\")", self.inner.build())
    }
}

// -- Response types --

/// General Perturbations (OMM) record from the GP request class.
///
/// Contains orbital elements and metadata for a single satellite.
/// All fields are optional strings since Space-Track may omit fields.
///
/// Attributes:
///     ccsds_omm_vers (str | None): CCSDS OMM version
///     comment (str | None): Comment field
///     creation_date (str | None): Record creation date
///     originator (str | None): Data originator
///     object_name (str | None): Satellite common name
///     object_id (str | None): International designator
///     center_name (str | None): Center name
///     ref_frame (str | None): Reference frame
///     time_system (str | None): Time system
///     mean_element_theory (str | None): Mean element theory
///     epoch (str | None): Epoch of orbital elements
///     mean_motion (float | None): Mean motion (rev/day)
///     eccentricity (float | None): Eccentricity
///     inclination (float | None): Inclination (degrees)
///     ra_of_asc_node (float | None): RAAN (degrees)
///     arg_of_pericenter (float | None): Argument of pericenter (degrees)
///     mean_anomaly (float | None): Mean anomaly (degrees)
///     ephemeris_type (int | None): Ephemeris type
///     classification_type (str | None): Classification type
///     norad_cat_id (int | None): NORAD catalog ID
///     element_set_no (int | None): Element set number
///     rev_at_epoch (int | None): Revolution number at epoch
///     bstar (float | None): BSTAR drag coefficient
///     mean_motion_dot (float | None): First derivative of mean motion
///     mean_motion_ddot (float | None): Second derivative of mean motion
///     semimajor_axis (float | None): Semi-major axis (km)
///     period (float | None): Orbital period (minutes)
///     apoapsis (float | None): Apoapsis altitude (km)
///     periapsis (float | None): Periapsis altitude (km)
///     object_type (str | None): Object type
///     rcs_size (str | None): RCS size category
///     country_code (str | None): Country code
///     launch_date (str | None): Launch date
///     site (str | None): Launch site
///     decay_date (str | None): Decay date
///     file (int | None): File number
///     gp_id (int | None): GP record ID
///     tle_line0 (str | None): TLE line 0
///     tle_line1 (str | None): TLE line 1
///     tle_line2 (str | None): TLE line 2
///
/// Example:
///     ```python
///     import brahe as bh
///
///     client = bh.SpaceTrackClient("user@example.com", "password")
///     query = bh.SpaceTrackQuery(bh.RequestClass.GP).filter("NORAD_CAT_ID", "25544").limit(1)
///     records = client.query_gp(query)
///     print(records[0].object_name)  # "ISS (ZARYA)"
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "GPRecord")]
#[derive(Clone)]
#[allow(clippy::upper_case_acronyms)]
pub struct PyGPRecord {
    inner: spacetrack::GPRecord,
}

#[pymethods]
impl PyGPRecord {
    #[getter] fn ccsds_omm_vers(&self) -> Option<String> { self.inner.ccsds_omm_vers.clone() }
    #[getter] fn comment(&self) -> Option<String> { self.inner.comment.clone() }
    #[getter] fn creation_date(&self) -> Option<String> { self.inner.creation_date.clone() }
    #[getter] fn originator(&self) -> Option<String> { self.inner.originator.clone() }
    #[getter] fn object_name(&self) -> Option<String> { self.inner.object_name.clone() }
    #[getter] fn object_id(&self) -> Option<String> { self.inner.object_id.clone() }
    #[getter] fn center_name(&self) -> Option<String> { self.inner.center_name.clone() }
    #[getter] fn ref_frame(&self) -> Option<String> { self.inner.ref_frame.clone() }
    #[getter] fn time_system(&self) -> Option<String> { self.inner.time_system.clone() }
    #[getter] fn mean_element_theory(&self) -> Option<String> { self.inner.mean_element_theory.clone() }
    #[getter] fn epoch(&self) -> Option<String> { self.inner.epoch.clone() }
    #[getter] fn mean_motion(&self) -> Option<f64> { self.inner.mean_motion }
    #[getter] fn eccentricity(&self) -> Option<f64> { self.inner.eccentricity }
    #[getter] fn inclination(&self) -> Option<f64> { self.inner.inclination }
    #[getter] fn ra_of_asc_node(&self) -> Option<f64> { self.inner.ra_of_asc_node }
    #[getter] fn arg_of_pericenter(&self) -> Option<f64> { self.inner.arg_of_pericenter }
    #[getter] fn mean_anomaly(&self) -> Option<f64> { self.inner.mean_anomaly }
    #[getter] fn ephemeris_type(&self) -> Option<u8> { self.inner.ephemeris_type }
    #[getter] fn classification_type(&self) -> Option<String> { self.inner.classification_type.clone() }
    #[getter] fn norad_cat_id(&self) -> Option<u32> { self.inner.norad_cat_id }
    #[getter] fn element_set_no(&self) -> Option<u16> { self.inner.element_set_no }
    #[getter] fn rev_at_epoch(&self) -> Option<u32> { self.inner.rev_at_epoch }
    #[getter] fn bstar(&self) -> Option<f64> { self.inner.bstar }
    #[getter] fn mean_motion_dot(&self) -> Option<f64> { self.inner.mean_motion_dot }
    #[getter] fn mean_motion_ddot(&self) -> Option<f64> { self.inner.mean_motion_ddot }
    #[getter] fn semimajor_axis(&self) -> Option<f64> { self.inner.semimajor_axis }
    #[getter] fn period(&self) -> Option<f64> { self.inner.period }
    #[getter] fn apoapsis(&self) -> Option<f64> { self.inner.apoapsis }
    #[getter] fn periapsis(&self) -> Option<f64> { self.inner.periapsis }
    #[getter] fn object_type(&self) -> Option<String> { self.inner.object_type.clone() }
    #[getter] fn rcs_size(&self) -> Option<String> { self.inner.rcs_size.clone() }
    #[getter] fn country_code(&self) -> Option<String> { self.inner.country_code.clone() }
    #[getter] fn launch_date(&self) -> Option<String> { self.inner.launch_date.clone() }
    #[getter] fn site(&self) -> Option<String> { self.inner.site.clone() }
    #[getter] fn decay_date(&self) -> Option<String> { self.inner.decay_date.clone() }
    #[getter] fn file(&self) -> Option<u64> { self.inner.file }
    #[getter] fn gp_id(&self) -> Option<u32> { self.inner.gp_id }
    #[getter] fn tle_line0(&self) -> Option<String> { self.inner.tle_line0.clone() }
    #[getter] fn tle_line1(&self) -> Option<String> { self.inner.tle_line1.clone() }
    #[getter] fn tle_line2(&self) -> Option<String> { self.inner.tle_line2.clone() }

    fn __str__(&self) -> String {
        format!(
            "GPRecord(name={:?}, norad_id={:?}, epoch={:?})",
            self.inner.object_name, self.inner.norad_cat_id, self.inner.epoch
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

/// Satellite Catalog (SATCAT) record.
///
/// Contains metadata about a cataloged space object.
/// All fields are optional strings since Space-Track may omit fields.
///
/// Attributes:
///     intldes (str | None): International designator
///     norad_cat_id (int | None): NORAD catalog ID
///     object_type (str | None): Object type code
///     satname (str | None): Satellite name
///     country (str | None): Country/organization code
///     launch (str | None): Launch date
///     site (str | None): Launch site
///     decay (str | None): Decay date
///     period (str | None): Orbital period (minutes)
///     inclination (str | None): Inclination (degrees)
///     apogee (str | None): Apogee altitude (km)
///     perigee (str | None): Perigee altitude (km)
///     comment (str | None): Comment
///     commentcode (str | None): Comment code
///     rcsvalue (str | None): RCS value
///     rcs_size (str | None): RCS size category
///     file (str | None): File number
///     launch_year (str | None): Launch year
///     launch_num (str | None): Launch number
///     launch_piece (str | None): Launch piece
///     current (str | None): Current status
///     object_name (str | None): Object name
///     object_id (str | None): Object ID
///     object_number (str | None): Object number
///
/// Example:
///     ```python
///     import brahe as bh
///
///     client = bh.SpaceTrackClient("user@example.com", "password")
///     query = bh.SpaceTrackQuery(bh.RequestClass.SATCAT).filter("NORAD_CAT_ID", "25544").limit(1)
///     records = client.query_satcat(query)
///     print(records[0].satname)  # "ISS (ZARYA)"
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "SATCATRecord")]
#[derive(Clone)]
#[allow(clippy::upper_case_acronyms)]
pub struct PySATCATRecord {
    inner: spacetrack::SATCATRecord,
}

#[pymethods]
impl PySATCATRecord {
    #[getter] fn intldes(&self) -> Option<String> { self.inner.intldes.clone() }
    #[getter] fn norad_cat_id(&self) -> Option<u32> { self.inner.norad_cat_id }
    #[getter] fn object_type(&self) -> Option<String> { self.inner.object_type.clone() }
    #[getter] fn satname(&self) -> Option<String> { self.inner.satname.clone() }
    #[getter] fn country(&self) -> Option<String> { self.inner.country.clone() }
    #[getter] fn launch(&self) -> Option<String> { self.inner.launch.clone() }
    #[getter] fn site(&self) -> Option<String> { self.inner.site.clone() }
    #[getter] fn decay(&self) -> Option<String> { self.inner.decay.clone() }
    #[getter] fn period(&self) -> Option<String> { self.inner.period.clone() }
    #[getter] fn inclination(&self) -> Option<String> { self.inner.inclination.clone() }
    #[getter] fn apogee(&self) -> Option<String> { self.inner.apogee.clone() }
    #[getter] fn perigee(&self) -> Option<String> { self.inner.perigee.clone() }
    #[getter] fn comment(&self) -> Option<String> { self.inner.comment.clone() }
    #[getter] fn commentcode(&self) -> Option<String> { self.inner.commentcode.clone() }
    #[getter] fn rcsvalue(&self) -> Option<String> { self.inner.rcsvalue.clone() }
    #[getter] fn rcs_size(&self) -> Option<String> { self.inner.rcs_size.clone() }
    #[getter] fn file(&self) -> Option<String> { self.inner.file.clone() }
    #[getter] fn launch_year(&self) -> Option<String> { self.inner.launch_year.clone() }
    #[getter] fn launch_num(&self) -> Option<String> { self.inner.launch_num.clone() }
    #[getter] fn launch_piece(&self) -> Option<String> { self.inner.launch_piece.clone() }
    #[getter] fn current(&self) -> Option<String> { self.inner.current.clone() }
    #[getter] fn object_name(&self) -> Option<String> { self.inner.object_name.clone() }
    #[getter] fn object_id(&self) -> Option<String> { self.inner.object_id.clone() }
    #[getter] fn object_number(&self) -> Option<String> { self.inner.object_number.clone() }

    fn __str__(&self) -> String {
        format!(
            "SATCATRecord(name={:?}, norad_id={:?})",
            self.inner.satname, self.inner.norad_cat_id
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

/// FileShare file record from the fileshare/file request class.
///
/// Contains metadata about a file in a user's Space-Track file share.
/// All fields are optional strings since Space-Track may omit fields.
///
/// Attributes:
///     file_id (str | None): File identifier
///     file_name (str | None): File name
///     file_link (str | None): File download link
///     file_size (str | None): File size in bytes
///     file_conttype (str | None): File content type
///     folder_id (str | None): Folder identifier
///     created (str | None): Creation date
///
/// Example:
///     ```python
///     import brahe as bh
///
///     client = bh.SpaceTrackClient("user@example.com", "password")
///     files = client.fileshare_list_files()
///     for f in files:
///         print(f.file_name, f.file_size)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "FileShareFileRecord")]
#[derive(Clone)]
pub struct PyFileShareFileRecord {
    inner: spacetrack::FileShareFileRecord,
}

#[pymethods]
impl PyFileShareFileRecord {
    #[getter] fn file_id(&self) -> Option<String> { self.inner.file_id.clone() }
    #[getter] fn file_name(&self) -> Option<String> { self.inner.file_name.clone() }
    #[getter] fn file_link(&self) -> Option<String> { self.inner.file_link.clone() }
    #[getter] fn file_size(&self) -> Option<String> { self.inner.file_size.clone() }
    #[getter] fn file_conttype(&self) -> Option<String> { self.inner.file_conttype.clone() }
    #[getter] fn folder_id(&self) -> Option<String> { self.inner.folder_id.clone() }
    #[getter] fn created(&self) -> Option<String> { self.inner.created.clone() }

    fn __str__(&self) -> String {
        format!(
            "FileShareFileRecord(id={:?}, name={:?})",
            self.inner.file_id, self.inner.file_name
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

/// FileShare folder record from the fileshare/folder request class.
///
/// Contains metadata about a folder in a user's Space-Track file share.
/// All fields are optional strings since Space-Track may omit fields.
///
/// Attributes:
///     folder_id (str | None): Folder identifier
///     folder_name (str | None): Folder name
///     parent_folder_id (str | None): Parent folder identifier
///     created (str | None): Creation date
///
/// Example:
///     ```python
///     import brahe as bh
///
///     client = bh.SpaceTrackClient("user@example.com", "password")
///     folders = client.fileshare_list_folders()
///     for f in folders:
///         print(f.folder_name, f.folder_id)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "FolderRecord")]
#[derive(Clone)]
pub struct PyFolderRecord {
    inner: spacetrack::FolderRecord,
}

#[pymethods]
impl PyFolderRecord {
    #[getter] fn folder_id(&self) -> Option<String> { self.inner.folder_id.clone() }
    #[getter] fn folder_name(&self) -> Option<String> { self.inner.folder_name.clone() }
    #[getter] fn parent_folder_id(&self) -> Option<String> { self.inner.parent_folder_id.clone() }
    #[getter] fn created(&self) -> Option<String> { self.inner.created.clone() }

    fn __str__(&self) -> String {
        format!(
            "FolderRecord(id={:?}, name={:?})",
            self.inner.folder_id, self.inner.folder_name
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

/// SP Ephemeris file record from the spephemeris/file request class.
///
/// Contains metadata about an SP ephemeris file on Space-Track.
/// All fields are optional strings since Space-Track may omit fields.
///
/// Attributes:
///     file_id (str | None): File identifier
///     norad_cat_id (int | None): NORAD catalog ID
///     file_name (str | None): File name
///     file_link (str | None): File download link
///     file_size (str | None): File size in bytes
///     created (str | None): Creation date
///     epoch_start (str | None): Epoch start
///     epoch_stop (str | None): Epoch stop
///
/// Example:
///     ```python
///     import brahe as bh
///
///     client = bh.SpaceTrackClient("user@example.com", "password")
///     files = client.spephemeris_list_files()
///     for f in files:
///         print(f.file_name, f.norad_cat_id)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "SPEphemerisFileRecord")]
#[derive(Clone)]
pub struct PySPEphemerisFileRecord {
    inner: spacetrack::SPEphemerisFileRecord,
}

#[pymethods]
impl PySPEphemerisFileRecord {
    #[getter] fn file_id(&self) -> Option<String> { self.inner.file_id.clone() }
    #[getter] fn norad_cat_id(&self) -> Option<u32> { self.inner.norad_cat_id }
    #[getter] fn file_name(&self) -> Option<String> { self.inner.file_name.clone() }
    #[getter] fn file_link(&self) -> Option<String> { self.inner.file_link.clone() }
    #[getter] fn file_size(&self) -> Option<String> { self.inner.file_size.clone() }
    #[getter] fn created(&self) -> Option<String> { self.inner.created.clone() }
    #[getter] fn epoch_start(&self) -> Option<String> { self.inner.epoch_start.clone() }
    #[getter] fn epoch_stop(&self) -> Option<String> { self.inner.epoch_stop.clone() }

    fn __str__(&self) -> String {
        format!(
            "SPEphemerisFileRecord(id={:?}, norad_id={:?}, name={:?})",
            self.inner.file_id, self.inner.norad_cat_id, self.inner.file_name
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }
}

// -- Rate Limit Config --

/// Configuration for SpaceTrack API rate limiting.
///
/// Defines the maximum number of requests allowed per minute and per hour.
/// Defaults to 25 requests/minute and 250 requests/hour (~83% of
/// Space-Track.org's actual limits of 30/min and 300/hour).
///
/// Args:
///     max_per_minute (int): Maximum requests per rolling 60-second window. Default: 25.
///     max_per_hour (int): Maximum requests per rolling 3600-second window. Default: 250.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Default conservative limits
///     config = bh.RateLimitConfig()
///     print(config.max_per_minute)  # 25
///     print(config.max_per_hour)    # 250
///
///     # Custom limits
///     config = bh.RateLimitConfig(max_per_minute=10, max_per_hour=100)
///
///     # Disable rate limiting
///     config = bh.RateLimitConfig.disabled()
///
///     # Use with client
///     client = bh.SpaceTrackClient("user@example.com", "password", rate_limit=config)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "RateLimitConfig")]
#[derive(Clone)]
pub struct PyRateLimitConfig {
    pub(crate) inner: spacetrack::RateLimitConfig,
}

#[pymethods]
impl PyRateLimitConfig {
    #[new]
    #[pyo3(signature = (max_per_minute=25, max_per_hour=250))]
    fn new(max_per_minute: u32, max_per_hour: u32) -> Self {
        PyRateLimitConfig {
            inner: spacetrack::RateLimitConfig {
                max_per_minute,
                max_per_hour,
            },
        }
    }

    /// Create a configuration that disables rate limiting.
    ///
    /// Returns:
    ///     RateLimitConfig: Configuration with no rate limits.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     config = bh.RateLimitConfig.disabled()
    ///     ```
    #[staticmethod]
    fn disabled() -> Self {
        PyRateLimitConfig {
            inner: spacetrack::RateLimitConfig::disabled(),
        }
    }

    /// Maximum requests per rolling 60-second window.
    #[getter]
    fn max_per_minute(&self) -> u32 {
        self.inner.max_per_minute
    }

    /// Maximum requests per rolling 3600-second window.
    #[getter]
    fn max_per_hour(&self) -> u32 {
        self.inner.max_per_hour
    }

    fn __str__(&self) -> String {
        format!(
            "RateLimitConfig(max_per_minute={}, max_per_hour={})",
            self.inner.max_per_minute, self.inner.max_per_hour
        )
    }

    fn __repr__(&self) -> String {
        self.__str__()
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.inner == other.inner),
            CompareOp::Ne => Ok(self.inner != other.inner),
            _ => Err(exceptions::PyNotImplementedError::new_err(
                "Comparison not supported",
            )),
        }
    }
}

// -- Client --

/// SpaceTrack API client with session-based authentication.
///
/// Handles authentication and query execution against Space-Track.org.
/// Lazily authenticates on first query and re-authenticates on session expiry.
///
/// Args:
///     identity (str): Space-Track.org login email.
///     password (str): Space-Track.org password.
///     base_url (str, optional): Custom base URL for testing.
///     rate_limit (RateLimitConfig, optional): Rate limit configuration.
///         Defaults to 25 requests/minute, 250 requests/hour.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     client = bh.SpaceTrackClient("user@example.com", "password")
///
///     # With custom rate limits
///     config = bh.RateLimitConfig(max_per_minute=10, max_per_hour=100)
///     client = bh.SpaceTrackClient("user@example.com", "password", rate_limit=config)
///
///     query = (
///         bh.SpaceTrackQuery(bh.RequestClass.GP)
///         .filter("NORAD_CAT_ID", "25544")
///         .order_by("EPOCH", bh.SortOrder.DESC)
///         .limit(1)
///     )
///     data = client.query_json(query)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "SpaceTrackClient")]
pub struct PySpaceTrackClient {
    inner: spacetrack::SpaceTrackClient,
}

#[pymethods]
impl PySpaceTrackClient {
    #[new]
    #[pyo3(signature = (identity, password, base_url=None, rate_limit=None))]
    fn new(
        identity: &str,
        password: &str,
        base_url: Option<&str>,
        rate_limit: Option<&PyRateLimitConfig>,
    ) -> Self {
        let config = rate_limit
            .map(|rl| rl.inner.clone())
            .unwrap_or_default();

        let client = match base_url {
            Some(url) => spacetrack::SpaceTrackClient::with_base_url_and_rate_limit(
                identity, password, url, config,
            ),
            None => spacetrack::SpaceTrackClient::with_rate_limit(identity, password, config),
        };
        PySpaceTrackClient { inner: client }
    }

    /// Explicitly authenticate with Space-Track.org.
    ///
    /// Called automatically on first query. Call explicitly to verify credentials early.
    ///
    /// Raises:
    ///     BraheError: If authentication fails.
    fn authenticate(&self) -> PyResult<()> {
        self.inner.authenticate().map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Execute a query and return the raw response body as a string.
    ///
    /// Args:
    ///     query (SpaceTrackQuery): The query to execute.
    ///
    /// Returns:
    ///     str: Raw response body.
    ///
    /// Raises:
    ///     BraheError: On network, auth, or HTTP errors.
    fn query_raw(&self, query: &PySpaceTrackQuery) -> PyResult<String> {
        self.inner
            .query_raw(&query.inner)
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Execute a query and return parsed JSON values.
    ///
    /// Args:
    ///     query (SpaceTrackQuery): The query to execute (must use JSON format).
    ///
    /// Returns:
    ///     list[dict]: List of JSON objects.
    ///
    /// Raises:
    ///     BraheError: On network, auth, parse, or format errors.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.SpaceTrackClient("user@example.com", "password")
    ///     query = bh.SpaceTrackQuery(bh.RequestClass.GP).filter("NORAD_CAT_ID", "25544").limit(1)
    ///     data = client.query_json(query)
    ///     print(data[0]["OBJECT_NAME"])
    ///     ```
    fn query_json(&self, py: Python<'_>, query: &PySpaceTrackQuery) -> PyResult<Py<PyAny>> {
        let values = self
            .inner
            .query_json(&query.inner)
            .map_err(|e| BraheError::new_err(e.to_string()))?;

        // Convert serde_json::Value to Python objects
        let json_str = serde_json::to_string(&values)
            .map_err(|e| BraheError::new_err(format!("JSON serialization failed: {}", e)))?;

        let json_module = py.import("json")?;
        json_module
            .call_method1("loads", (json_str,))
            .map(|obj| obj.into())
    }

    /// Execute a GP query and return typed GP records.
    ///
    /// Args:
    ///     query (SpaceTrackQuery): The query to execute (must use JSON format).
    ///
    /// Returns:
    ///     list[GPRecord]: List of typed GP records.
    ///
    /// Raises:
    ///     BraheError: On network, auth, parse, or format errors.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.SpaceTrackClient("user@example.com", "password")
    ///     query = bh.SpaceTrackQuery(bh.RequestClass.GP).filter("NORAD_CAT_ID", "25544").limit(1)
    ///     records = client.query_gp(query)
    ///     print(records[0].object_name)
    ///     ```
    fn query_gp(&self, query: &PySpaceTrackQuery) -> PyResult<Vec<PyGPRecord>> {
        let records = self
            .inner
            .query_gp(&query.inner)
            .map_err(|e| BraheError::new_err(e.to_string()))?;

        Ok(records.into_iter().map(|r| PyGPRecord { inner: r }).collect())
    }

    /// Execute a SATCAT query and return typed SATCAT records.
    ///
    /// Args:
    ///     query (SpaceTrackQuery): The query to execute (must use JSON format).
    ///
    /// Returns:
    ///     list[SATCATRecord]: List of typed SATCAT records.
    ///
    /// Raises:
    ///     BraheError: On network, auth, parse, or format errors.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.SpaceTrackClient("user@example.com", "password")
    ///     query = bh.SpaceTrackQuery(bh.RequestClass.SATCAT).filter("NORAD_CAT_ID", "25544").limit(1)
    ///     records = client.query_satcat(query)
    ///     print(records[0].satname)
    ///     ```
    fn query_satcat(&self, query: &PySpaceTrackQuery) -> PyResult<Vec<PySATCATRecord>> {
        let records = self
            .inner
            .query_satcat(&query.inner)
            .map_err(|e| BraheError::new_err(e.to_string()))?;

        Ok(records
            .into_iter()
            .map(|r| PySATCATRecord { inner: r })
            .collect())
    }

    // ========================================
    // FileShare operations
    // ========================================

    /// Upload a file to the Space-Track file share.
    ///
    /// Args:
    ///     folder_id (str): Target folder identifier.
    ///     file_name (str): Name for the uploaded file.
    ///     file_data (bytes): File content as bytes.
    ///
    /// Returns:
    ///     str: Server response (typically JSON confirmation).
    ///
    /// Raises:
    ///     BraheError: On network, auth, or upload errors.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.SpaceTrackClient("user@example.com", "password")
    ///     response = client.fileshare_upload("100", "data.txt", b"file contents")
    ///     ```
    fn fileshare_upload(
        &self,
        folder_id: &str,
        file_name: &str,
        file_data: &[u8],
    ) -> PyResult<String> {
        self.inner
            .fileshare_upload(folder_id, file_name, file_data)
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    /// Download a file from the Space-Track file share.
    ///
    /// Args:
    ///     file_id (str): File identifier to download.
    ///
    /// Returns:
    ///     bytes: File content as bytes.
    ///
    /// Raises:
    ///     BraheError: On network, auth, or download errors.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.SpaceTrackClient("user@example.com", "password")
    ///     data = client.fileshare_download("12345")
    ///     ```
    fn fileshare_download<'py>(
        &self,
        py: Python<'py>,
        file_id: &str,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let data = self
            .inner
            .fileshare_download(file_id)
            .map_err(|e| BraheError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new(py, &data))
    }

    /// Download all files in a folder from the Space-Track file share.
    ///
    /// Args:
    ///     folder_id (str): Folder identifier to download.
    ///
    /// Returns:
    ///     bytes: Folder content as bytes (typically a zip archive).
    ///
    /// Raises:
    ///     BraheError: On network, auth, or download errors.
    fn fileshare_download_folder<'py>(
        &self,
        py: Python<'py>,
        folder_id: &str,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let data = self
            .inner
            .fileshare_download_folder(folder_id)
            .map_err(|e| BraheError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new(py, &data))
    }

    /// List files in the Space-Track file share.
    ///
    /// Returns:
    ///     list[FileShareFileRecord]: File metadata records.
    ///
    /// Raises:
    ///     BraheError: On network, auth, or parse errors.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.SpaceTrackClient("user@example.com", "password")
    ///     files = client.fileshare_list_files()
    ///     for f in files:
    ///         print(f.file_name, f.file_size)
    ///     ```
    fn fileshare_list_files(&self) -> PyResult<Vec<PyFileShareFileRecord>> {
        let records = self
            .inner
            .fileshare_list_files()
            .map_err(|e| BraheError::new_err(e.to_string()))?;
        Ok(records
            .into_iter()
            .map(|r| PyFileShareFileRecord { inner: r })
            .collect())
    }

    /// List folders in the Space-Track file share.
    ///
    /// Returns:
    ///     list[FolderRecord]: Folder metadata records.
    ///
    /// Raises:
    ///     BraheError: On network, auth, or parse errors.
    fn fileshare_list_folders(&self) -> PyResult<Vec<PyFolderRecord>> {
        let records = self
            .inner
            .fileshare_list_folders()
            .map_err(|e| BraheError::new_err(e.to_string()))?;
        Ok(records
            .into_iter()
            .map(|r| PyFolderRecord { inner: r })
            .collect())
    }

    /// Delete a file from the Space-Track file share.
    ///
    /// Args:
    ///     file_id (str): File identifier to delete.
    ///
    /// Returns:
    ///     str: Server response.
    ///
    /// Raises:
    ///     BraheError: On network, auth, or deletion errors.
    fn fileshare_delete(&self, file_id: &str) -> PyResult<String> {
        self.inner
            .fileshare_delete(file_id)
            .map_err(|e| BraheError::new_err(e.to_string()))
    }

    // ========================================
    // SP Ephemeris operations
    // ========================================

    /// Download an SP ephemeris file from Space-Track.
    ///
    /// Args:
    ///     file_id (str): SP ephemeris file identifier.
    ///
    /// Returns:
    ///     bytes: Ephemeris file content as bytes.
    ///
    /// Raises:
    ///     BraheError: On network, auth, or download errors.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.SpaceTrackClient("user@example.com", "password")
    ///     data = client.spephemeris_download("99999")
    ///     ```
    fn spephemeris_download<'py>(
        &self,
        py: Python<'py>,
        file_id: &str,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let data = self
            .inner
            .spephemeris_download(file_id)
            .map_err(|e| BraheError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new(py, &data))
    }

    /// List available SP ephemeris files.
    ///
    /// Returns:
    ///     list[SPEphemerisFileRecord]: Ephemeris file metadata records.
    ///
    /// Raises:
    ///     BraheError: On network, auth, or parse errors.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.SpaceTrackClient("user@example.com", "password")
    ///     files = client.spephemeris_list_files()
    ///     for f in files:
    ///         print(f.file_name, f.norad_cat_id)
    ///     ```
    fn spephemeris_list_files(&self) -> PyResult<Vec<PySPEphemerisFileRecord>> {
        let records = self
            .inner
            .spephemeris_list_files()
            .map_err(|e| BraheError::new_err(e.to_string()))?;
        Ok(records
            .into_iter()
            .map(|r| PySPEphemerisFileRecord { inner: r })
            .collect())
    }

    /// List SP ephemeris file history.
    ///
    /// Returns:
    ///     list[dict]: File history records as JSON objects.
    ///
    /// Raises:
    ///     BraheError: On network, auth, or parse errors.
    fn spephemeris_file_history(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let values = self
            .inner
            .spephemeris_file_history()
            .map_err(|e| BraheError::new_err(e.to_string()))?;

        let json_str = serde_json::to_string(&values)
            .map_err(|e| BraheError::new_err(format!("JSON serialization failed: {}", e)))?;

        let json_module = py.import("json")?;
        json_module
            .call_method1("loads", (json_str,))
            .map(|obj| obj.into())
    }

    // ========================================
    // Public Files operations
    // ========================================

    /// Download a public file from Space-Track (no auth required).
    ///
    /// Args:
    ///     file_name (str): Name of the public file to download.
    ///
    /// Returns:
    ///     bytes: File content as bytes.
    ///
    /// Raises:
    ///     BraheError: On network or download errors.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     client = bh.SpaceTrackClient("user@example.com", "password")
    ///     data = client.publicfiles_download("catalog.txt")
    ///     ```
    fn publicfiles_download<'py>(
        &self,
        py: Python<'py>,
        file_name: &str,
    ) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let data = self
            .inner
            .publicfiles_download(file_name)
            .map_err(|e| BraheError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new(py, &data))
    }

    /// List public file directories on Space-Track (no auth required).
    ///
    /// Returns:
    ///     list[dict]: Directory listing as JSON objects.
    ///
    /// Raises:
    ///     BraheError: On network or parse errors.
    fn publicfiles_list_dirs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let values = self
            .inner
            .publicfiles_list_dirs()
            .map_err(|e| BraheError::new_err(e.to_string()))?;

        let json_str = serde_json::to_string(&values)
            .map_err(|e| BraheError::new_err(format!("JSON serialization failed: {}", e)))?;

        let json_module = py.import("json")?;
        json_module
            .call_method1("loads", (json_str,))
            .map(|obj| obj.into())
    }
}

// -- Operator functions --

/// Greater-than operator for SpaceTrack queries.
///
/// Args:
///     value (str): The comparison value.
///
/// Returns:
///     str: Formatted operator string ">value".
///
/// Example:
///     ```python
///     from brahe.spacetrack import operators as op
///     op.greater_than("25544")  # ">25544"
///     ```
#[pyfunction]
#[pyo3(name = "spacetrack_greater_than")]
fn py_spacetrack_greater_than(value: &str) -> String {
    spacetrack::operators::greater_than(value)
}

/// Less-than operator for SpaceTrack queries.
///
/// Args:
///     value (str): The comparison value.
///
/// Returns:
///     str: Formatted operator string "<value".
#[pyfunction]
#[pyo3(name = "spacetrack_less_than")]
fn py_spacetrack_less_than(value: &str) -> String {
    spacetrack::operators::less_than(value)
}

/// Not-equal operator for SpaceTrack queries.
///
/// Args:
///     value (str): The comparison value.
///
/// Returns:
///     str: Formatted operator string "<>value".
#[pyfunction]
#[pyo3(name = "spacetrack_not_equal")]
fn py_spacetrack_not_equal(value: &str) -> String {
    spacetrack::operators::not_equal(value)
}

/// Inclusive range operator for SpaceTrack queries.
///
/// Args:
///     left (str): Lower bound (inclusive).
///     right (str): Upper bound (inclusive).
///
/// Returns:
///     str: Formatted operator string "left--right".
///
/// Example:
///     ```python
///     from brahe.spacetrack import operators as op
///     op.inclusive_range("25544", "25600")  # "25544--25600"
///     ```
#[pyfunction]
#[pyo3(name = "spacetrack_inclusive_range")]
fn py_spacetrack_inclusive_range(left: &str, right: &str) -> String {
    spacetrack::operators::inclusive_range(left, right)
}

/// Like/contains operator for SpaceTrack queries.
///
/// Args:
///     value (str): The pattern to match.
///
/// Returns:
///     str: Formatted operator string "~~value".
#[pyfunction]
#[pyo3(name = "spacetrack_like")]
fn py_spacetrack_like(value: &str) -> String {
    spacetrack::operators::like(value)
}

/// Starts-with operator for SpaceTrack queries.
///
/// Args:
///     value (str): The prefix to match.
///
/// Returns:
///     str: Formatted operator string "^value".
#[pyfunction]
#[pyo3(name = "spacetrack_startswith")]
fn py_spacetrack_startswith(value: &str) -> String {
    spacetrack::operators::startswith(value)
}

/// Current time reference for SpaceTrack queries.
///
/// Returns:
///     str: The string "now".
#[pyfunction]
#[pyo3(name = "spacetrack_now")]
fn py_spacetrack_now() -> String {
    spacetrack::operators::now()
}

/// Time offset from now for SpaceTrack queries.
///
/// Args:
///     days (int): Number of days offset (negative for past, positive for future).
///
/// Returns:
///     str: Formatted time reference "now-N" or "now+N".
///
/// Example:
///     ```python
///     from brahe.spacetrack import operators as op
///     op.now_offset(-7)   # "now-7"
///     op.now_offset(14)   # "now+14"
///     ```
#[pyfunction]
#[pyo3(name = "spacetrack_now_offset")]
fn py_spacetrack_now_offset(days: i32) -> String {
    spacetrack::operators::now_offset(days)
}

/// Null value reference for SpaceTrack queries.
///
/// Returns:
///     str: The string "null-val".
#[pyfunction]
#[pyo3(name = "spacetrack_null_val")]
fn py_spacetrack_null_val() -> String {
    spacetrack::operators::null_val()
}

/// OR list operator for SpaceTrack queries.
///
/// Args:
///     values (list[str]): List of values to match against.
///
/// Returns:
///     str: Comma-separated value string "val1,val2,val3".
///
/// Example:
///     ```python
///     from brahe.spacetrack import operators as op
///     op.or_list(["25544", "25545"])  # "25544,25545"
///     ```
#[pyfunction]
#[pyo3(name = "spacetrack_or_list")]
fn py_spacetrack_or_list(values: Vec<String>) -> String {
    let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();
    spacetrack::operators::or_list(&refs)
}
