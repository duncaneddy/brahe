// Python bindings for the SBDB Lookup API client module.

use brahe::datasets::sbdb::{SBDBClient, SBDBObject};

/// A resolved JPL Small-Body Database object.
///
/// Attributes:
///     spkid (int): Primary SPK-ID (NAIF ID).
///     full_name (str): Full designation and name.
///     des (str): Primary designation.
///     shortname (str | None): Name without alternate designation.
///     kind (str): Object kind code.
///     neo (bool): Whether the object is a near-Earth object.
///     gm (float | None): Gravitational parameter [m^3/s^2], if catalogued.
///     radius (float | None): Mean radius [m], if catalogued.
#[pyclass(name = "SBDBObject", module = "brahe._brahe", skip_from_py_object)]
#[derive(Clone)]
pub struct PySBDBObject {
    inner: SBDBObject,
}

#[pymethods]
impl PySBDBObject {
    #[getter]
    fn spkid(&self) -> i32 {
        self.inner.spkid
    }
    #[getter]
    fn full_name(&self) -> String {
        self.inner.full_name.clone()
    }
    #[getter]
    fn des(&self) -> String {
        self.inner.des.clone()
    }
    #[getter]
    fn shortname(&self) -> Option<String> {
        self.inner.shortname.clone()
    }
    #[getter]
    fn kind(&self) -> String {
        self.inner.kind.clone()
    }
    #[getter]
    fn neo(&self) -> bool {
        self.inner.neo
    }
    #[getter]
    fn gm(&self) -> Option<f64> {
        self.inner.gm
    }
    #[getter]
    fn radius(&self) -> Option<f64> {
        self.inner.radius
    }

    /// Return the object's NAIF ID (identical to its SPK-ID).
    ///
    /// Returns:
    ///     int: The NAIF ID.
    fn naif_id(&self) -> i32 {
        self.inner.naif_id()
    }

    fn __repr__(&self) -> String {
        format!(
            "SBDBObject(spkid={}, full_name='{}')",
            self.inner.spkid, self.inner.full_name
        )
    }
}

/// Client for the JPL Small-Body Database (SBDB) Lookup API.
///
/// Args:
///     base_url (str, optional): Override the SBDB base URL (for testing).
///     cache_max_age (int, optional): Cache max age in seconds (0 = always refetch).
///
/// Example:
///     ```python
///     import brahe as bh
///     client = bh.datasets.sbdb.SBDBClient()
///     ceres = client.lookup("Ceres")
///     print(ceres.naif_id())  # 2000001
///     ```
#[pyclass(name = "SBDBClient", module = "brahe._brahe")]
pub struct PySBDBClient {
    inner: SBDBClient,
}

#[pymethods]
impl PySBDBClient {
    #[new]
    #[pyo3(signature = (base_url=None, cache_max_age=None))]
    fn new(base_url: Option<String>, cache_max_age: Option<u64>) -> Self {
        let inner = match (base_url, cache_max_age) {
            (Some(url), Some(age)) => SBDBClient::with_base_url_and_cache_age(&url, age),
            (Some(url), None) => SBDBClient::with_base_url(&url),
            (None, Some(age)) => SBDBClient::with_cache_age(age),
            (None, None) => SBDBClient::new(),
        };
        PySBDBClient { inner }
    }

    /// Resolve a search string to an SBDBObject.
    ///
    /// Args:
    ///     sstr (str): Object search string, e.g. "Ceres" or "2000001".
    ///
    /// Returns:
    ///     SBDBObject: The resolved object.
    fn lookup(&self, sstr: &str) -> PyResult<PySBDBObject> {
        self.inner
            .lookup(sstr)
            .map(|inner| PySBDBObject { inner })
            .map_err(|e| BraheError::new_err(e.to_string()))
    }
}
