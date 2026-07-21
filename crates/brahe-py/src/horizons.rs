// Python bindings for the Horizons SPK generation API client module.

use brahe::datasets::horizons::{HorizonsClient, HorizonsSPKRequest, HorizonsSPKResponse};

/// A request to generate a small-body SPK over a time span.
///
/// Args:
///     command (str): Horizons COMMAND target, e.g. "DES=20000001;".
///     start (Epoch): SPK span start.
///     stop (Epoch): SPK span stop.
#[pyclass(
    name = "HorizonsSPKRequest",
    module = "brahe._brahe",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyHorizonsSPKRequest {
    inner: HorizonsSPKRequest,
}

#[pymethods]
impl PyHorizonsSPKRequest {
    #[new]
    fn new(command: &str, start: PyEpoch, stop: PyEpoch) -> Self {
        PyHorizonsSPKRequest {
            inner: HorizonsSPKRequest::new(command, start.obj, stop.obj),
        }
    }

    /// Create a request targeting a small body by SPK-ID.
    ///
    /// Args:
    ///     spkid (int): Small-body SPK-ID (NAIF ID).
    ///     start (Epoch): SPK span start.
    ///     stop (Epoch): SPK span stop.
    ///
    /// Returns:
    ///     HorizonsSPKRequest: The configured request.
    #[staticmethod]
    fn for_spkid(spkid: i32, start: PyEpoch, stop: PyEpoch) -> Self {
        PyHorizonsSPKRequest {
            inner: HorizonsSPKRequest::for_spkid(spkid, start.obj, stop.obj),
        }
    }

    /// Override the CENTER body (default "500@0", the SSB).
    ///
    /// Args:
    ///     center (str): Horizons center specification.
    ///
    /// Returns:
    ///     HorizonsSPKRequest: The request with the new center.
    fn with_center(&self, center: &str) -> Self {
        PyHorizonsSPKRequest {
            inner: self.inner.clone().with_center(center),
        }
    }

    #[getter]
    fn command(&self) -> String {
        self.inner.command.clone()
    }
    #[getter]
    fn center(&self) -> String {
        self.inner.center.clone()
    }
}

/// A handle to a generated (or cached) Horizons SPK kernel.
///
/// Attributes:
///     path (str): Path to the cached .bsp kernel.
///     spk_file_id (str | None): Horizons spk_file_id, if returned.
#[pyclass(name = "HorizonsSPKResponse", module = "brahe._brahe")]
pub struct PyHorizonsSPKResponse {
    inner: HorizonsSPKResponse,
}

#[pymethods]
impl PyHorizonsSPKResponse {
    #[getter]
    fn path(&self) -> String {
        self.inner.path().to_string_lossy().to_string()
    }
    #[getter]
    fn spk_file_id(&self) -> Option<String> {
        self.inner.spk_file_id().map(|s| s.to_string())
    }

    /// Read the raw SPK bytes from the cached file.
    ///
    /// Returns:
    ///     bytes: The kernel file contents.
    fn bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let data = self
            .inner
            .bytes()
            .map_err(|e| BraheError::new_err(e.to_string()))?;
        Ok(pyo3::types::PyBytes::new(py, &data))
    }

    /// Load the cached SPK into the global SPICE registry.
    ///
    /// Returns:
    ///     None: On success.
    fn load(&self) -> PyResult<()> {
        self.inner
            .load()
            .map_err(|e| BraheError::new_err(e.to_string()))
    }
}

/// Client for the JPL Horizons SPK generation API.
///
/// Args:
///     base_url (str, optional): Override the Horizons base URL (for testing).
///
/// Example:
///     ```python
///     import brahe as bh
///     client = bh.datasets.horizons.HorizonsClient()
///     req = bh.datasets.horizons.HorizonsSPKRequest.for_spkid(20000001, t0, t1)
///     resp = client.get_spk(req)
///     resp.load()
///     ```
#[pyclass(name = "HorizonsClient", module = "brahe._brahe")]
pub struct PyHorizonsClient {
    inner: HorizonsClient,
}

#[pymethods]
impl PyHorizonsClient {
    #[new]
    #[pyo3(signature = (base_url=None))]
    fn new(base_url: Option<String>) -> Self {
        let inner = match base_url {
            Some(url) => HorizonsClient::with_base_url(&url),
            None => HorizonsClient::new(),
        };
        PyHorizonsClient { inner }
    }

    /// Generate (or reuse a cached) SPK for the request.
    ///
    /// Args:
    ///     request (HorizonsSPKRequest): The SPK generation request.
    ///
    /// Returns:
    ///     HorizonsSPKResponse: Handle to the cached .bsp.
    fn get_spk(&self, request: &PyHorizonsSPKRequest) -> PyResult<PyHorizonsSPKResponse> {
        self.inner
            .get_spk(&request.inner)
            .map(|inner| PyHorizonsSPKResponse { inner })
            .map_err(|e| BraheError::new_err(e.to_string()))
    }
}
