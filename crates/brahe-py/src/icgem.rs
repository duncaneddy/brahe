// Python bindings for the ICGEM datasets module.

use brahe::datasets::icgem::{
    self, ICGEMBody as RustICGEMBody, IndexEntry as RustIndexEntry,
};

fn body_from_str(s: &str) -> RustICGEMBody {
    RustICGEMBody::from_name(s)
}

/// One ICGEM model index row.
#[pyclass(name = "ICGEMIndexEntry")]
#[derive(Clone)]
pub struct PyIndexEntry {
    inner: RustIndexEntry,
}

#[pymethods]
impl PyIndexEntry {
    /// Body string (e.g. "Earth", "Moon", "pluto").
    #[getter]
    fn body(&self) -> String { self.inner.body.as_name().to_string() }

    /// Model name.
    #[getter]
    fn name(&self) -> String { self.inner.name.clone() }

    /// Publication year, if known.
    #[getter]
    fn year(&self) -> Option<u16> { self.inner.year }

    /// Maximum spherical harmonic degree.
    #[getter]
    fn degree(&self) -> u32 { self.inner.degree }

    /// ICGEM relative download path (includes the opaque hash).
    #[getter]
    fn download_path(&self) -> String { self.inner.download_path.clone() }

    fn __repr__(&self) -> String {
        format!(
            "ICGEMIndexEntry(body='{}', name='{}', degree={}, year={:?})",
            self.inner.body.as_name(),
            self.inner.name,
            self.inner.degree,
            self.inner.year
        )
    }
}

/// List ICGEM models for a body.
///
/// Args:
///     body (str): Body name. Known: "earth", "moon", "mars", "venus", "ceres".
///         Any other name is treated as a custom celestial body and matched
///         against the ICGEM celestial catalog.
///
/// Returns:
///     list[ICGEMIndexEntry]
#[pyfunction]
#[pyo3(name = "icgem_list_models")]
fn py_icgem_list_models(body: &str) -> PyResult<Vec<PyIndexEntry>> {
    let body = body_from_str(body);
    let entries = icgem::list_icgem_models(body)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(entries.into_iter().map(|e| PyIndexEntry { inner: e }).collect())
}

/// Force-refresh the ICGEM index file for a single body.
#[pyfunction]
#[pyo3(name = "icgem_refresh_index")]
fn py_icgem_refresh_index(body: &str) -> PyResult<()> {
    icgem::refresh_icgem_index(body_from_str(body))
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Force-refresh both Earth and celestial index files.
#[pyfunction]
#[pyo3(name = "icgem_refresh_all_indexes")]
fn py_icgem_refresh_all_indexes() -> PyResult<()> {
    icgem::refresh_all_icgem_indexes()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Download a `.gfc` file from ICGEM (with caching).
///
/// Args:
///     body (str): Body name.
///     name (str): Model name, optionally suffixed with -DEGREE.
///     output_path (str, optional): Optional copy destination.
///
/// Returns:
///     str: Path to the resulting `.gfc` file.
#[pyfunction]
#[pyo3(name = "icgem_download_model", signature = (body, name, output_path=None))]
fn py_icgem_download_model(
    body: &str,
    name: &str,
    output_path: Option<&str>,
) -> PyResult<String> {
    let body = body_from_str(body);
    let out = output_path.map(std::path::PathBuf::from);
    let path = icgem::download_icgem_model(body, name, out)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    path.to_str()
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Path contains invalid UTF-8 characters",
            )
        })
        .map(|s| s.to_string())
}

// Functions are registered in lib.rs via add_function() calls.
