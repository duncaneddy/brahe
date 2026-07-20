// Python bindings for the datasets module.

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

/// Download a NAIF kernel with caching support
///
/// Downloads the named NAIF kernel file (planetary DE, satellite ephemeris, or
/// binary PCK) from NASA JPL's NAIF archive and caches it locally. If the
/// kernel is already cached, returns the cached path without re-downloading.
/// Optionally copies the kernel to a user-specified location.
///
/// Args:
///     name (str): Kernel name. Supported: the planetary (DE) ephemerides
///         "de430", "de432s", "de435", "de438", "de440", "de440s", "de442",
///         "de442s"; the satellite ephemeris kernels "mar099", "mar099s",
///         "jup365", "sat441", "ura184", "nep097", "plu060"; and the binary
///         PCK "moon_pa_de440".
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
///     kernel_path = bh.datasets.naif.download_spice_kernel("de440s")
///     print(f"Kernel cached at: {kernel_path}")
///
///     # Download and copy to specific location
///     kernel_path = bh.datasets.naif.download_spice_kernel("de440s", "/path/to/my_kernel.bsp")
///     print(f"Kernel saved to: {kernel_path}")
///     ```
///
/// Note:
///     - DE kernels are long-term stable products and are not refreshed once cached
///     - Files are cached to ~/.cache/brahe/naif/ (or $BRAHE_CACHE/naif/ if set)
///     - Kernel files vary in size (de440s: ~33 MB, de440: ~120 MB)
///     - Available at: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/
#[pyfunction]
#[pyo3(name = "download_spice_kernel", signature = (name, output_path=None))]
fn py_download_spice_kernel(name: &str, output_path: Option<&str>) -> PyResult<String> {
    let output_pathbuf = output_path.map(PathBuf::from);
    let kernel = spice::SPICEKernel::from_name(name).ok_or_else(|| {
        let supported = spice::SPICEKernel::all()
            .iter()
            .map(|k| k.name())
            .collect::<Vec<_>>()
            .join(", ");
        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
            "Unsupported kernel name '{}'. Supported kernels: {}",
            name, supported
        ))
    })?;
    let result_path = naif::download_spice_kernel(kernel, output_pathbuf)
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

/// Load the Vallado SSN sensor sites.
///
/// Returns representative US Space Surveillance Network sensor sites with
/// locations from Vallado 4th Ed. Table 4-2, az/el/range limits from Table
/// 4-3, and bias/noise calibration values from Table 4-4. Data is embedded;
/// no network access is required.
///
/// Returns:
///     list[PointLocation]: 21 sensor sites with properties:
///         - sensor_type: "azel_range" or "radec"
///         - system, category, sensor_numbers: descriptive metadata
///         - az_min_deg/az_max_deg/el_min_deg/el_max_deg/range_max_m: optional limits
///         - range_bias_m/az_bias_deg/el_bias_deg and
///           range_noise_m/az_noise_deg/el_noise_deg: optional calibration
///
/// Raises:
///     RuntimeError: If the embedded data cannot be parsed.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     sensors = bh.datasets.ssn_sensors.load()
///     radars = [s for s in sensors if s.properties["sensor_type"] == "azel_range"]
///     ```
#[pyfunction]
#[pyo3(name = "ssn_sensors_load")]
fn py_ssn_sensors_load() -> PyResult<Vec<PyPointLocation>> {
    let locations = ssn_sensors::load_ssn_sensors()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
    Ok(locations
        .into_iter()
        .map(|loc| PyPointLocation { location: loc })
        .collect())
}

// Functions are registered in mod.rs via add_function() calls
