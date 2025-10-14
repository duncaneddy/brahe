
// Help functions for type conversions

/// Helper function to parse strings into appropriate EOPExtrapolation enumerations
fn string_to_eop_extrapolation(s: &str) -> Result<eop::EOPExtrapolation, BraheError> {
    match s {
        "Hold" => Ok(eop::EOPExtrapolation::Hold),
        "Zero" => Ok(eop::EOPExtrapolation::Zero),
        "Error" => Ok(eop::EOPExtrapolation::Error),
        _ => Err(BraheError::Error(format!(
            "Unknown EOP Extrapolation string \"{}\"",
            s
        ))),
    }
}

/// Helper function to convert EOPExtrapolation enumerations into representative string
fn eop_extrapolation_to_string(extrapolation: eop::EOPExtrapolation) -> String {
    match extrapolation {
        eop::EOPExtrapolation::Hold => String::from("Hold"),
        eop::EOPExtrapolation::Zero => String::from("Zero"),
        eop::EOPExtrapolation::Error => String::from("Error"),
    }
}

/// Helper function to parse strings into appropriate EOPType enumerations
fn string_to_eop_type(s: &str) -> Result<eop::EOPType, BraheError> {
    match s {
        "C04" => Ok(eop::EOPType::C04),
        "StandardBulletinA" => Ok(eop::EOPType::StandardBulletinA),
        "Unknown" => Ok(eop::EOPType::Unknown),
        "Static" => Ok(eop::EOPType::Static),
        _ => Err(BraheError::Error(format!(
            "Unknown EOP Type string \"{}\"",
            s
        ))),
    }
}

/// Helper function to convert EOPType enumerations into representative string
fn eop_type_to_string(eop_type: eop::EOPType) -> String {
    match eop_type {
        eop::EOPType::C04 => String::from("C04"),
        eop::EOPType::StandardBulletinA => String::from("StandardBulletinA"),
        eop::EOPType::Unknown => String::from("Unknown"),
        eop::EOPType::Static => String::from("Static"),
    }
}

// Module Method Wrappers

/// Download latest C04 Earth orientation parameter file. Will attempt to download the latest
/// parameter file to the specified location. Creating any missing directories as required.
///
/// The download source is the [IERS Earth Orientation Data Products](https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html)
///
/// Args:
///     filepath (str): Path of desired output file
#[pyfunction]
#[pyo3(text_signature = "(filepath)")]
#[pyo3(name = "download_c04_eop_file")]
fn py_download_c04_eop_file(filepath: &str) -> PyResult<()> {
    eop::download_c04_eop_file(filepath).unwrap();
    Ok(())
}

/// Download latest standard Earth orientation parameter file. Will attempt to download the latest
/// parameter file to the specified location. Creating any missing directories as required.
///
/// The download source is the [IERS Earth Orientation Data Products](https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html)
///
/// Args:
///     filepath (str): Path of desired output file
#[pyfunction]
#[pyo3(text_signature = "(filepath)")]
#[pyo3(name = "download_standard_eop_file")]
fn py_download_standard_eop_file(filepath: &str) -> PyResult<()> {
    eop::download_standard_eop_file(filepath).unwrap();
    Ok(())
}

/// Static Earth Orientation Parameter provider with constant values.
///
/// Provides EOP data using fixed values that don't change with time.
/// Useful for testing or scenarios where time-varying EOP data is not needed.
#[pyclass]
#[pyo3(name = "StaticEOPProvider")]
pub(crate) struct PyStaticEOPProvider {
    obj: eop::StaticEOPProvider,
}


#[pymethods]
impl PyStaticEOPProvider {
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.obj)
    }

    pub fn __str__(&self) -> String {
        self.obj.to_string()
    }

    /// Create a new static EOP provider with default values.
    ///
    /// Returns:
    ///     StaticEOPProvider: New provider with default EOP values
    #[new]
    pub fn new() -> Self {
        PyStaticEOPProvider {
            obj: eop::StaticEOPProvider::new(),
        }
    }

    /// Create a static EOP provider with all values set to zero.
    ///
    /// Returns:
    ///     StaticEOPProvider: Provider with all EOP values set to zero
    #[classmethod]
    pub fn from_zero(_cls: &Bound<'_, PyType>) -> Self {
        PyStaticEOPProvider {
            obj: eop::StaticEOPProvider::from_zero(),
        }
    }

    /// Create a static EOP provider with specified values.
    ///
    /// Args:
    ///     ut1_utc (float): UT1-UTC time difference in seconds
    ///     pm_x (float): Polar motion x-component in radians
    ///     pm_y (float): Polar motion y-component in radians
    ///     dx (float): Celestial pole offset dx in radians
    ///     dy (float): Celestial pole offset dy in radians
    ///     lod (float): Length of day offset in seconds
    ///
    /// Returns:
    ///     StaticEOPProvider: Provider with specified EOP values
    #[classmethod]
    pub fn from_values(_cls: &Bound<'_, PyType>, ut1_utc: f64, pm_x: f64, pm_y: f64, dx: f64, dy: f64, lod: f64) -> Self {
        PyStaticEOPProvider {
            obj: eop::StaticEOPProvider::from_values((ut1_utc, pm_x, pm_y, dx, dy, lod))
        }
    }

    /// Check if the provider is initialized.
    ///
    /// Returns:
    ///     bool: True if initialized
    pub fn is_initialized(&self) -> bool {
        self.obj.is_initialized()
    }

    /// Get the number of EOP data points.
    ///
    /// Returns:
    ///     int: Number of EOP data points
    pub fn len(&self) -> usize {
        self.obj.len()
    }

    /// Get the EOP data type.
    ///
    /// Returns:
    ///     str: EOP type string
    pub fn eop_type(&self) -> String {
        eop_type_to_string(self.obj.eop_type())
    }

    /// Get the extrapolation method.
    ///
    /// Returns:
    ///     str: Extrapolation method string
    pub fn extrapolation(&self) -> String {
        eop_extrapolation_to_string(self.obj.extrapolation())
    }

    /// Check if interpolation is enabled.
    ///
    /// Returns:
    ///     bool: True if interpolation is enabled
    pub fn interpolation(&self) -> bool {
        self.obj.interpolation()
    }

    /// Get the minimum Modified Julian Date in the dataset.
    ///
    /// Returns:
    ///     float: Minimum MJD
    pub fn mjd_min(&self) -> f64 {
        self.obj.mjd_min()
    }

    /// Get the maximum Modified Julian Date in the dataset.
    ///
    /// Returns:
    ///     float: Maximum MJD
    pub fn mjd_max(&self) -> f64 {
        self.obj.mjd_max()
    }

    /// Get the last Modified Julian Date with LOD data.
    ///
    /// Returns:
    ///     float: Last MJD with LOD data
    pub fn mjd_last_lod(&self) -> f64 {
        self.obj.mjd_last_lod()
    }

    /// Get the last Modified Julian Date with dx/dy data.
    ///
    /// Returns:
    ///     float: Last MJD with dx/dy data
    pub fn mjd_last_dxdy(&self) -> f64 {
        self.obj.mjd_last_dxdy()
    }

    /// Get UT1-UTC time difference for a given MJD.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: UT1-UTC time difference in seconds
    pub fn get_ut1_utc(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_ut1_utc(mjd)
    }

    /// Get polar motion components for a given MJD.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     tuple[float, float]: Polar motion x and y components in radians
    pub fn get_pm(&self, mjd: f64) -> Result<(f64, f64), BraheError> {
        self.obj.get_pm(mjd)
    }

    /// Get celestial pole offsets for a given MJD.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     tuple[float, float]: Celestial pole offsets dx and dy in radians
    pub fn get_dxdy(&self, mjd: f64) -> Result<(f64, f64), BraheError> {
        self.obj.get_dxdy(mjd)
    }

    /// Get length of day offset for a given MJD.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: Length of day offset in seconds
    pub fn get_lod(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_lod(mjd)
    }

    /// Get all EOP parameters for a given MJD.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     tuple[float, float, float, float, float, float]: UT1-UTC, pm_x, pm_y, dx, dy, lod
    pub fn get_eop(&self, mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), BraheError> {
        self.obj.get_eop(mjd)
    }

}

/// File-based Earth Orientation Parameter provider.
///
/// Loads EOP data from files in standard IERS formats and provides
/// interpolation and extrapolation capabilities.
#[pyclass]
#[pyo3(name = "FileEOPProvider")]
pub(crate) struct PyFileEOPProvider {
    obj: eop::FileEOPProvider,
}


#[pymethods]
impl PyFileEOPProvider {
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.obj)
    }

    pub fn __str__(&self) -> String {
        self.obj.to_string()
    }

    /// Create a new uninitialized file EOP provider.
    ///
    /// Returns:
    ///     FileEOPProvider: New uninitialized provider
    #[new]
    pub fn new() -> Self {
        PyFileEOPProvider {
            obj: eop::FileEOPProvider::new(),
        }
    }

    /// Create provider from a C04 format EOP file.
    ///
    /// Args:
    ///     filepath (str): Path to C04 EOP file
    ///     interpolate (bool): Enable interpolation between data points
    ///     extrapolate (str): Extrapolation method ("Hold", "Zero", or "Error")
    ///
    /// Returns:
    ///     FileEOPProvider: Provider initialized with C04 file data
    #[classmethod]
    pub fn from_c04_file(_cls: &Bound<'_, PyType>, filepath: &str, interpolate: bool, extrapolate: &str) -> Result<Self, BraheError> {
        Ok(PyFileEOPProvider {
            obj: eop::FileEOPProvider::from_c04_file(Path::new(filepath), interpolate, string_to_eop_extrapolation(extrapolate)?)?,
        })
    }

    /// Create provider from a standard IERS format EOP file.
    ///
    /// Args:
    ///     filepath (str): Path to standard IERS EOP file
    ///     interpolate (bool): Enable interpolation between data points
    ///     extrapolate (str): Extrapolation method ("Hold", "Zero", or "Error")
    ///
    /// Returns:
    ///     FileEOPProvider: Provider initialized with standard file data
    #[classmethod]
    pub fn from_standard_file(_cls: &Bound<'_, PyType>, filepath: &str, interpolate: bool, extrapolate: &str) -> Result<Self, BraheError> {
        Ok(PyFileEOPProvider {
            obj: eop::FileEOPProvider::from_standard_file(Path::new(filepath), interpolate, string_to_eop_extrapolation(extrapolate)?)?,
        })
    }

    /// Create provider from an EOP file with automatic format detection.
    ///
    /// Args:
    ///     filepath (str): Path to EOP file
    ///     interpolate (bool): Enable interpolation between data points
    ///     extrapolate (str): Extrapolation method ("Hold", "Zero", or "Error")
    ///
    /// Returns:
    ///     FileEOPProvider: Provider initialized with file data
    #[classmethod]
    pub fn from_file(_cls: &Bound<'_, PyType>, filepath: &str, interpolate: bool, extrapolate: &str) -> Result<Self, BraheError> {
        Ok(PyFileEOPProvider {
            obj: eop::FileEOPProvider::from_file(Path::new(filepath), interpolate, string_to_eop_extrapolation(extrapolate)?)?,
        })
    }

    /// Create provider from the default C04 EOP file location.
    ///
    /// Args:
    ///     interpolate (bool): Enable interpolation between data points
    ///     extrapolate (str): Extrapolation method ("Hold", "Zero", or "Error")
    ///
    /// Returns:
    ///     FileEOPProvider: Provider initialized with default C04 file
    #[classmethod]
    pub fn from_default_c04(_cls: &Bound<'_, PyType>, interpolate: bool, extrapolate: &str) -> Result<Self, BraheError> {
        Ok(PyFileEOPProvider {
            obj: eop::FileEOPProvider::from_default_c04(interpolate, string_to_eop_extrapolation(extrapolate)?)?,
        })
    }

    /// Create provider from the default standard IERS EOP file location.
    ///
    /// Args:
    ///     interpolate (bool): Enable interpolation between data points
    ///     extrapolate (str): Extrapolation method ("Hold", "Zero", or "Error")
    ///
    /// Returns:
    ///     FileEOPProvider: Provider initialized with default standard file
    #[classmethod]
    pub fn from_default_standard(_cls: &Bound<'_, PyType>, interpolate: bool, extrapolate: &str) -> Result<Self, BraheError> {
        Ok(PyFileEOPProvider {
            obj: eop::FileEOPProvider::from_default_standard(interpolate, string_to_eop_extrapolation(extrapolate)?)?,
        })
    }

    /// Create provider from default EOP file location with specified type.
    ///
    /// Args:
    ///     eop_type (str): EOP file type ("C04" or "StandardBulletinA")
    ///     interpolate (bool): Enable interpolation between data points
    ///     extrapolate (str): Extrapolation method ("Hold", "Zero", or "Error")
    ///
    /// Returns:
    ///     FileEOPProvider: Provider initialized with default file of specified type
    #[classmethod]
    pub fn from_default_file(_cls: &Bound<'_, PyType>, eop_type: &str, interpolate: bool, extrapolate: &str) -> Result<Self, BraheError> {
        Ok(PyFileEOPProvider {
            obj: eop::FileEOPProvider::from_default_file(string_to_eop_type(eop_type)?, interpolate, string_to_eop_extrapolation(extrapolate)?)?,
        })
    }

    /// Check if the provider is initialized.
    ///
    /// Returns:
    ///     bool: True if initialized
    pub fn is_initialized(&self) -> bool {
        self.obj.is_initialized()
    }

    /// Get the number of EOP data points.
    ///
    /// Returns:
    ///     int: Number of EOP data points
    pub fn len(&self) -> usize {
        self.obj.len()
    }

    /// Get the EOP data type.
    ///
    /// Returns:
    ///     str: EOP type string
    pub fn eop_type(&self) -> String {
        eop_type_to_string(self.obj.eop_type())
    }

    /// Get the extrapolation method.
    ///
    /// Returns:
    ///     str: Extrapolation method string
    pub fn extrapolation(&self) -> String {
        eop_extrapolation_to_string(self.obj.extrapolation())
    }

    /// Check if interpolation is enabled.
    ///
    /// Returns:
    ///     bool: True if interpolation is enabled
    pub fn interpolation(&self) -> bool {
        self.obj.interpolation()
    }

    /// Get the minimum Modified Julian Date in the dataset.
    ///
    /// Returns:
    ///     float: Minimum MJD
    pub fn mjd_min(&self) -> f64 {
        self.obj.mjd_min()
    }

    /// Get the maximum Modified Julian Date in the dataset.
    ///
    /// Returns:
    ///     float: Maximum MJD
    pub fn mjd_max(&self) -> f64 {
        self.obj.mjd_max()
    }

    /// Get the last Modified Julian Date with LOD data.
    ///
    /// Returns:
    ///     float: Last MJD with LOD data
    pub fn mjd_last_lod(&self) -> f64 {
        self.obj.mjd_last_lod()
    }

    /// Get the last Modified Julian Date with dx/dy data.
    ///
    /// Returns:
    ///     float: Last MJD with dx/dy data
    pub fn mjd_last_dxdy(&self) -> f64 {
        self.obj.mjd_last_dxdy()
    }

    /// Get UT1-UTC time difference for a given MJD.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: UT1-UTC time difference in seconds
    pub fn get_ut1_utc(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_ut1_utc(mjd)
    }

    /// Get polar motion components for a given MJD.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     tuple[float, float]: Polar motion x and y components in radians
    pub fn get_pm(&self, mjd: f64) -> Result<(f64, f64), BraheError> {
        self.obj.get_pm(mjd)
    }

    /// Get celestial pole offsets for a given MJD.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     tuple[float, float]: Celestial pole offsets dx and dy in radians
    pub fn get_dxdy(&self, mjd: f64) -> Result<(f64, f64), BraheError> {
        self.obj.get_dxdy(mjd)
    }

    /// Get length of day offset for a given MJD.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     float: Length of day offset in seconds
    pub fn get_lod(&self, mjd: f64) -> Result<f64, BraheError> {
        self.obj.get_lod(mjd)
    }

    /// Get all EOP parameters for a given MJD.
    ///
    /// Args:
    ///     mjd (float): Modified Julian Date
    ///
    /// Returns:
    ///     tuple[float, float, float, float, float, float]: UT1-UTC, pm_x, pm_y, dx, dy, lod
    pub fn get_eop(&self, mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), BraheError> {
        self.obj.get_eop(mjd)
    }

}

/// Set the global EOP provider using a static provider.
///
/// Args:
///     provider (StaticEOPProvider): Static EOP provider to set globally
#[pyfunction]
#[pyo3(text_signature = "(provider)")]
#[pyo3(name = "set_global_eop_provider_from_static_provider")]
pub fn py_set_global_eop_provider_from_static_provider(provider: &PyStaticEOPProvider) {
    eop::set_global_eop_provider(provider.obj);
}

/// Set the global EOP provider using a file-based provider.
///
/// Args:
///     provider (FileEOPProvider): File-based EOP provider to set globally
#[pyfunction]
#[pyo3(text_signature = "(provider)")]
#[pyo3(name = "set_global_eop_provider_from_file_provider")]
pub fn py_set_global_eop_provider_from_file_provider(provider: &PyFileEOPProvider) {
    // We have to clone the object because FileEOPProvider is not Copy and
    // cannot implement a trivial one. Passing the reference to set the
    // global provider would result in a transfer of ownership, so instead we
    // clone the object and pass the clone.
    //
    // It would be preferable not to duplicate the memory and have to do the
    // clone, but it is necessary to maintain the current API.
    eop::set_global_eop_provider(provider.obj.clone());
}

/// Get UT1-UTC time difference from the global EOP provider.
///
/// Args:
///     mjd (float): Modified Julian Date
///
/// Returns:
///     float: UT1-UTC time difference in seconds
#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_ut1_utc")]
pub fn py_get_global_ut1_utc(mjd: f64) -> Result<f64, BraheError> {
    eop::get_global_ut1_utc(mjd)
}

/// Get polar motion components from the global EOP provider.
///
/// Args:
///     mjd (float): Modified Julian Date
///
/// Returns:
///     tuple[float, float]: Polar motion x and y components in radians
#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_pm")]
pub fn py_get_global_pm(mjd: f64) -> Result<(f64, f64), BraheError> {
    eop::get_global_pm(mjd)
}

/// Get celestial pole offsets from the global EOP provider.
///
/// Args:
///     mjd (float): Modified Julian Date
///
/// Returns:
///     tuple[float, float]: Celestial pole offsets dx and dy in radians
#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_dxdy")]
pub fn py_get_global_dxdy(mjd: f64) -> Result<(f64, f64), BraheError> {
    eop::get_global_dxdy(mjd)
}

/// Get length of day offset from the global EOP provider.
///
/// Args:
///     mjd (float): Modified Julian Date
///
/// Returns:
///     float: Length of day offset in seconds
#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_lod")]
pub fn py_get_global_lod(mjd: f64) -> Result<f64, BraheError> {
    eop::get_global_lod(mjd)
}

/// Get all EOP parameters from the global EOP provider.
///
/// Args:
///     mjd (float): Modified Julian Date
///
/// Returns:
///     tuple[float, float, float, float, float, float]: UT1-UTC, pm_x, pm_y, dx, dy, lod
#[pyfunction]
#[pyo3(text_signature = "(mjd)")]
#[pyo3(name = "get_global_eop")]
pub fn py_get_global_eop(mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), BraheError> {
    eop::get_global_eop(mjd)
}

/// Check if the global EOP provider is initialized.
///
/// Returns:
///     bool: True if global EOP provider is initialized
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_initialization")]
pub fn py_get_global_eop_initialization() -> bool {
    eop::get_global_eop_initialization()
}

/// Get the number of EOP data points in the global provider.
///
/// Returns:
///     int: Number of EOP data points
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_len")]
pub fn py_get_global_eop_len() -> usize {
    eop::get_global_eop_len()
}

/// Get the EOP data type of the global provider.
///
/// Returns:
///     str: EOP type string
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_type")]
pub fn py_get_global_eop_type() -> String {
    eop_type_to_string(eop::get_global_eop_type())
}

/// Get the extrapolation method of the global EOP provider.
///
/// Returns:
///     str: Extrapolation method string
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_extrapolation")]
pub fn py_get_global_eop_extrapolation() -> String {
    eop_extrapolation_to_string(eop::get_global_eop_extrapolation())
}

/// Check if interpolation is enabled in the global EOP provider.
///
/// Returns:
///     bool: True if interpolation is enabled
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_interpolation")]
pub fn py_get_global_eop_interpolation() -> bool {
    eop::get_global_eop_interpolation()
}

/// Get the minimum Modified Julian Date in the global EOP dataset.
///
/// Returns:
///     float: Minimum MJD
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_mjd_min")]
pub fn py_get_global_eop_mjd_min() -> f64 {
    eop::get_global_eop_mjd_min()
}

/// Get the maximum Modified Julian Date in the global EOP dataset.
///
/// Returns:
///     float: Maximum MJD
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_mjd_max")]
pub fn py_get_global_eop_mjd_max() -> f64 {
    eop::get_global_eop_mjd_max()
}

/// Get the last Modified Julian Date with LOD data in the global provider.
///
/// Returns:
///     float: Last MJD with LOD data
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_mjd_last_lod")]
pub fn py_get_global_eop_mjd_last_lod() -> f64 {
    eop::get_global_eop_mjd_last_lod()
}

/// Get the last Modified Julian Date with dx/dy data in the global provider.
///
/// Returns:
///     float: Last MJD with dx/dy data
#[pyfunction]
#[pyo3(text_signature = "()")]
#[pyo3(name = "get_global_eop_mjd_last_dxdy")]
pub fn py_get_global_eop_mjd_last_dxdy() -> f64 {
    eop::get_global_eop_mjd_last_dxdy()
}