// Python bindings for the datasets module.

use crate::datasets::celestrak;
use crate::datasets::groundstations;
use crate::datasets::naif;
use std::path::PathBuf;

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
///     ephemeris = bh.datasets.celestrak.get_tles("stations")
///
///     # Print first 5 satellites
///     for name, line1, line2 in ephemeris[:5]:
///         print(f"Satellite: {name}")
///         print(f"  Line 1: {line1[:20]}...")
///     ```
#[pyfunction]
#[pyo3(name = "celestrak_get_tles")]
fn py_celestrak_get_tles(group: &str) -> PyResult<Vec<(String, String, String)>> {
    celestrak::get_tles(group).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
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
///     propagators = bh.datasets.celestrak.get_tles_as_propagators("gnss", 60.0)
///     print(f"Loaded {len(propagators)} GNSS satellites")
///
///     # Propagate first satellite
///     epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0, tsys="UTC")
///     state = propagators[0].propagate(epoch)
///     ```
#[pyfunction]
#[pyo3(name = "celestrak_get_tles_as_propagators")]
fn py_celestrak_get_tles_as_propagators(
    group: &str,
    step_size: f64,
) -> PyResult<Vec<PySGPPropagator>> {
    let propagators = celestrak::get_tles_as_propagators(group, step_size)
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
///     bh.datasets.celestrak.download_tles("gnss", "gnss_sats.json", "3le", "json")
///
///     # Download active satellites as 2LE in plain text
///     bh.datasets.celestrak.download_tles("active", "active.txt", "tle", "txt")
///
///     # Download stations as 3LE in CSV format
///     bh.datasets.celestrak.download_tles("stations", "stations.csv", "3le", "csv")
///     ```
#[pyfunction]
#[pyo3(name = "celestrak_download_tles")]
fn py_celestrak_download_tles(
    group: &str,
    filepath: &str,
    content_format: &str,
    file_format: &str,
) -> PyResult<()> {
    celestrak::download_tles(group, filepath, content_format, file_format)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Load groundstation locations for a specific provider
///
/// Loads groundstation locations from embedded data. The data is compiled into
/// the binary and does not require external files or internet connection.
///
/// Args:
///     provider (str): Provider name (case-insensitive). Available providers:
///         - "atlas": Atlas Space Operations
///         - "aws": Amazon Web Services Ground Station
///         - "ksat": Kongsberg Satellite Services
///         - "leaf": Leaf Space
///         - "ssc": Swedish Space Corporation
///         - "viasat": Viasat
///
/// Returns:
///     list[PointLocation]: List of PointLocation objects with properties:
///         - name: Groundstation name
///         - provider: Provider name
///         - frequency_bands: List of supported frequency bands
///
/// Raises:
///     RuntimeError: If provider is unknown or data cannot be loaded.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Load KSAT groundstations
///     ksat_stations = bh.datasets.groundstations.load("ksat")
///
///     for station in ksat_stations:
///         print(f"{station.name}: ({station.lon():.2f}, {station.lat():.2f})")
///
///     # Check properties
///     props = ksat_stations[0].properties()
///     print(f"Frequency bands: {props['frequency_bands']}")
///     ```
#[pyfunction]
#[pyo3(name = "groundstations_load")]
fn py_groundstations_load(provider: &str) -> PyResult<Vec<PyPointLocation>> {
    let locations = groundstations::load_groundstations(provider)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    // Wrap each PointLocation in PyPointLocation
    Ok(locations
        .into_iter()
        .map(|loc| PyPointLocation { location: loc })
        .collect())
}

/// Load groundstations from a custom GeoJSON file
///
/// Loads groundstation locations from a user-provided GeoJSON file.
/// The file must be a FeatureCollection with Point geometries.
///
/// Args:
///     filepath (str): Path to GeoJSON file.
///
/// Returns:
///     list[PointLocation]: List of PointLocation objects.
///
/// Raises:
///     RuntimeError: If file cannot be read or parsed.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Load custom groundstations
///     stations = bh.datasets.groundstations.load_from_file("my_stations.geojson")
///     ```
#[pyfunction]
#[pyo3(name = "groundstations_load_from_file")]
fn py_groundstations_load_from_file(filepath: &str) -> PyResult<Vec<PyPointLocation>> {
    let locations = groundstations::load_groundstations_from_file(filepath)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(locations
        .into_iter()
        .map(|loc| PyPointLocation { location: loc })
        .collect())
}

/// Load all groundstations from all providers
///
/// Convenience function to load groundstations from all available providers.
///
/// Returns:
///     list[PointLocation]: Combined list of all groundstations.
///
/// Raises:
///     RuntimeError: If no groundstations can be loaded.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     all_stations = bh.datasets.groundstations.load_all()
///     print(f"Loaded {len(all_stations)} total groundstations")
///     ```
#[pyfunction]
#[pyo3(name = "groundstations_load_all")]
fn py_groundstations_load_all() -> PyResult<Vec<PyPointLocation>> {
    let locations = groundstations::load_all_groundstations()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(locations
        .into_iter()
        .map(|loc| PyPointLocation { location: loc })
        .collect())
}

/// Get list of available groundstation providers
///
/// Returns:
///     list[str]: List of provider names that can be used with load().
///
/// Example:
///     ```python
///     import brahe as bh
///
///     providers = bh.datasets.groundstations.list_providers()
///     print(f"Available: {', '.join(providers)}")
///     ```
#[pyfunction]
#[pyo3(name = "groundstations_list_providers")]
fn py_groundstations_list_providers() -> Vec<String> {
    groundstations::list_providers()
}

/// Get TLE data for a specific satellite by NORAD catalog number
///
/// Downloads 3LE data from CelesTrak for a single satellite identified by its
/// NORAD catalog number. Uses cached data if available and less than 6 hours old.
///
/// Args:
///     norad_id (int): NORAD catalog number (1-9 digits).
///     group (str, optional): Satellite group for fallback search if direct ID lookup fails.
///         Available groups can be found at https://celestrak.org/NORAD/elements/
///
/// Returns:
///     tuple[str, str, str]: Tuple of (name, line1, line2) containing satellite
///         name and TLE lines.
///
/// Raises:
///     RuntimeError: If download fails or satellite not found.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Get ISS TLE by NORAD ID (25544)
///     name, line1, line2 = bh.datasets.celestrak.get_tle_by_id(25544)
///     print(f"Satellite: {name}")
///     print(f"Line 1: {line1}")
///     print(f"Line 2: {line2}")
///
///     # With group fallback
///     tle = bh.datasets.celestrak.get_tle_by_id(25544, group="stations")
///     ```
///
/// Note:
///     You can find which group contains a specific NORAD ID at:
///     https://celestrak.org/NORAD/elements/master-gp-index.php
///
///     Data is cached for 6 hours to reduce server load and improve performance.
#[pyfunction]
#[pyo3(name = "celestrak_get_tle_by_id", signature = (norad_id, group=None))]
fn py_celestrak_get_tle_by_id(
    norad_id: u32,
    group: Option<&str>,
) -> PyResult<(String, String, String)> {
    celestrak::get_tle_by_id(norad_id, group)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Get TLE data for a specific satellite as an SGP propagator
///
/// Downloads TLE data from CelesTrak for a single satellite and creates an
/// SGP4/SDP4 propagator. Uses cached data if available and less than 6 hours old.
///
/// Args:
///     norad_id (int): NORAD catalog number (1-9 digits).
///     step_size (float): Default step size for propagator in seconds.
///     group (str, optional): Satellite group for fallback search if direct ID lookup fails.
///
/// Returns:
///     SGPPropagator: Configured SGP propagator (PySGPPropagator) ready to use.
///
/// Raises:
///     RuntimeError: If download fails, satellite not found, or TLE is invalid.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Get ISS as propagator with 60-second step size
///     propagator = bh.datasets.celestrak.get_tle_by_id_as_propagator(25544, 60.0)
///
///     # Propagate to current epoch
///     epoch = bh.Epoch.now()
///     state = propagator.propagate(epoch)
///     print(f"ISS position: {state[:3]}")
///
///     # With group fallback
///     prop = bh.datasets.celestrak.get_tle_by_id_as_propagator(
///         25544, 60.0, group="stations"
///     )
///     ```
///
/// Note:
///     You can find which group contains a specific NORAD ID at:
///     https://celestrak.org/NORAD/elements/master-gp-index.php
///
///     Data is cached for 6 hours to reduce server load and improve performance.
#[pyfunction]
#[pyo3(name = "celestrak_get_tle_by_id_as_propagator", signature = (norad_id, step_size, group=None))]
fn py_celestrak_get_tle_by_id_as_propagator(
    norad_id: u32,
    step_size: f64,
    group: Option<&str>,
) -> PyResult<PySGPPropagator> {
    let propagator = celestrak::get_tle_by_id_as_propagator(norad_id, group, step_size)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PySGPPropagator { propagator })
}

/// Get TLE data for a specific satellite by name
///
/// Searches for a satellite by name using a cascading search strategy:
/// 1. If a group is provided, search within that group first
/// 2. Fall back to searching the "active" group
/// 3. Fall back to using CelesTrak's NAME API
///
/// Uses cached data if available and less than 6 hours old.
///
/// Args:
///     name (str): Satellite name (case-insensitive, partial matches supported).
///     group (str, optional): Satellite group to search first.
///         Available groups can be found at https://celestrak.org/NORAD/elements/
///
/// Returns:
///     tuple[str, str, str]: Tuple of (name, line1, line2) containing satellite
///         name and TLE lines.
///
/// Raises:
///     RuntimeError: If download fails or satellite not found.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Search for ISS with group hint
///     name, line1, line2 = bh.datasets.celestrak.get_tle_by_name("ISS", group="stations")
///     print(f"Found: {name}")
///
///     # Search without group (uses cascading search)
///     tle = bh.datasets.celestrak.get_tle_by_name("STARLINK-1234")
///     ```
///
/// Note:
///     - Name matching is case-insensitive
///     - Partial names are supported (e.g., "ISS" will match "ISS (ZARYA)")
///     - If multiple satellites match, returns the first match
///     - Search order: specified group → "active" → NAME API
///     - Data is cached for 6 hours to reduce server load
#[pyfunction]
#[pyo3(name = "celestrak_get_tle_by_name", signature = (name, group=None))]
fn py_celestrak_get_tle_by_name(
    name: &str,
    group: Option<&str>,
) -> PyResult<(String, String, String)> {
    celestrak::get_tle_by_name(name, group)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}

/// Get TLE data for a specific satellite by name as an SGP propagator
///
/// Searches for a satellite by name and creates an SGP4/SDP4 propagator.
/// Uses cascading search strategy (specified group → active → NAME API).
/// Uses cached data if available and less than 6 hours old.
///
/// Args:
///     name (str): Satellite name (case-insensitive, partial matches supported).
///     step_size (float): Default step size for propagator in seconds.
///     group (str, optional): Satellite group to search first.
///
/// Returns:
///     SGPPropagator: Configured SGP propagator (PySGPPropagator) ready to use.
///
/// Raises:
///     RuntimeError: If download fails, satellite not found, or TLE is invalid.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Get ISS as propagator with 60-second step size
///     propagator = bh.datasets.celestrak.get_tle_by_name_as_propagator("ISS", 60.0, group="stations")
///
///     # Propagate to current epoch
///     epoch = bh.Epoch.now()
///     state = propagator.propagate(epoch)
///     print(f"Position: {state[:3]}")
///     ```
///
/// Note:
///     Data is cached for 6 hours to reduce server load and improve performance.
#[pyfunction]
#[pyo3(name = "celestrak_get_tle_by_name_as_propagator", signature = (name, step_size, group=None))]
fn py_celestrak_get_tle_by_name_as_propagator(
    name: &str,
    step_size: f64,
    group: Option<&str>,
) -> PyResult<PySGPPropagator> {
    let propagator = celestrak::get_tle_by_name_as_propagator(name, group, step_size)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    Ok(PySGPPropagator { propagator })
}

/// Download a DE kernel from NAIF with caching support
///
/// Downloads the specified DE (Development Ephemeris) kernel file from NASA JPL's NAIF
/// archive and caches it locally. If the kernel is already cached, returns the cached
/// path without re-downloading. Optionally copies the kernel to a user-specified location.
///
/// Args:
///     name (str): Kernel name. Supported kernels: "de430", "de432s", "de435", "de438",
///         "de440", "de440s", "de442", "de442s".
///     output_path (str, optional): Optional path to copy the kernel to after download/cache retrieval.
///         If not specified, returns the cache location.
///
/// Returns:
///     str: Path to the kernel file (cache location or output_path if specified).
///
/// Raises:
///     RuntimeError: If kernel name is unsupported, download fails, or file operations fail.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Download and cache de440s kernel
///     kernel_path = bh.datasets.naif.download_de_kernel("de440s")
///     print(f"Kernel cached at: {kernel_path}")
///
///     # Download and copy to specific location
///     kernel_path = bh.datasets.naif.download_de_kernel("de440s", "/path/to/my_kernel.bsp")
///     print(f"Kernel saved to: {kernel_path}")
///     ```
///
/// Note:
///     - DE kernels are long-term stable products and are not refreshed once cached
///     - Files are cached to ~/.cache/brahe/naif/ (or $BRAHE_CACHE/naif/ if set)
///     - Kernel files are large (de440s: ~17MB, de440: ~114MB)
///     - Available at: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/
#[pyfunction]
#[pyo3(name = "naif_download_de_kernel", signature = (name, output_path=None))]
fn py_naif_download_de_kernel(name: &str, output_path: Option<&str>) -> PyResult<String> {
    let output_pathbuf = output_path.map(PathBuf::from);
    let result_path = naif::download_de_kernel(name, output_pathbuf)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

    result_path
        .to_str()
        .ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "Path contains invalid UTF-8 characters",
            )
        })
        .map(|s| s.to_string())
}

// Functions are registered in mod.rs via add_function() calls
