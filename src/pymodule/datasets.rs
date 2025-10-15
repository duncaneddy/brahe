// Python bindings for the datasets module.

use crate::datasets::celestrak;

/// Get satellite ephemeris data from CelesTrak
///
/// Downloads and parses 3LE (three-line element) data for the specified satellite group
/// from CelesTrak (https://celestrak.org).
///
/// Args:
///     group (str): Satellite group name (e.g., "active", "stations", "gnss", "last-30-days").
///         See https://celestrak.org/NORAD/elements/ for available groups.
///
/// Returns:
///     list[tuple[str, str, str]]: List of (name, line1, line2) tuples containing satellite
///         names and TLE lines.
///
/// Raises:
///     RuntimeError: If download fails or data cannot be parsed.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Download ephemeris for ground stations
///     ephemeris = bh.datasets.celestrak.get_ephemeris("stations")
///
///     # Print first 5 satellites
///     for name, line1, line2 in ephemeris[:5]:
///         print(f"Satellite: {name}")
///         print(f"  Line 1: {line1[:20]}...")
///     ```
#[pyfunction]
#[pyo3(name = "celestrak_get_ephemeris")]
fn py_celestrak_get_ephemeris(group: &str) -> PyResult<Vec<(String, String, String)>> {
    celestrak::get_ephemeris(group).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Get satellite ephemeris as SGP propagators from CelesTrak
///
/// Downloads and parses 3LE data from CelesTrak, then creates SGP4/SDP4 propagators
/// for each satellite. This is a convenient way to get ready-to-use propagators.
///
/// Args:
///     group (str): Satellite group name (e.g., "active", "stations", "gnss", "last-30-days").
///     step_size (float): Default step size for propagators in seconds.
///
/// Returns:
///     list[SGPPropagator]: List of configured SGP propagators (PySGPPropagator), one per satellite.
///
/// Raises:
///     RuntimeError: If download fails or no valid propagators can be created.
///
/// Note:
///     Satellites with invalid TLE data will be skipped with a warning printed to stderr.
///     The function will only raise an error if NO valid propagators can be created.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Get propagators for GNSS satellites with 60-second step size
///     propagators = bh.datasets.celestrak.get_ephemeris_as_propagators("gnss", 60.0)
///     print(f"Loaded {len(propagators)} GNSS satellites")
///
///     # Propagate first satellite
///     epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0, tsys="UTC")
///     state = propagators[0].propagate(epoch)
///     ```
#[pyfunction]
#[pyo3(name = "celestrak_get_ephemeris_as_propagators")]
fn py_celestrak_get_ephemeris_as_propagators(
    group: &str,
    step_size: f64,
) -> PyResult<Vec<PySGPPropagator>> {
    let propagators = celestrak::get_ephemeris_as_propagators(group, step_size)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Wrap each Rust SGPPropagator in PySGPPropagator
    Ok(propagators.into_iter().map(|propagator| PySGPPropagator { propagator }).collect())
}

/// Download satellite ephemeris from CelesTrak and save to file
///
/// Downloads 3LE data from CelesTrak and serializes to the specified file format.
/// The file can contain either 2-line elements (TLE, without names) or 3-line elements
/// (3LE, with satellite names), and can be saved as plain text, CSV, or JSON.
///
/// Args:
///     group (str): Satellite group name (e.g., "active", "stations", "gnss", "last-30-days").
///     filepath (str): Output file path. Parent directories will be created if needed.
///     content_format (str): Content format - "tle" (2-line without names) or "3le" (3-line with names).
///     file_format (str): File format - "txt" (plain text), "csv" (comma-separated), or "json" (JSON array).
///
/// Raises:
///     RuntimeError: If download fails, format is invalid, or file cannot be written.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Download GNSS satellites as 3LE in JSON format
///     bh.datasets.celestrak.download_ephemeris("gnss", "gnss_sats.json", "3le", "json")
///
///     # Download active satellites as 2LE in plain text
///     bh.datasets.celestrak.download_ephemeris("active", "active.txt", "tle", "txt")
///
///     # Download stations as 3LE in CSV format
///     bh.datasets.celestrak.download_ephemeris("stations", "stations.csv", "3le", "csv")
///     ```
#[pyfunction]
#[pyo3(name = "celestrak_download_ephemeris")]
fn py_celestrak_download_ephemeris(
    group: &str,
    filepath: &str,
    content_format: &str,
    file_format: &str,
) -> PyResult<()> {
    celestrak::download_ephemeris(group, filepath, content_format, file_format)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

// Functions are registered in mod.rs via add_function() calls
