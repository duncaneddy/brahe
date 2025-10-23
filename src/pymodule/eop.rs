
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
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Download latest C04 EOP data
///     bh.download_c04_eop_file("./eop_data/finals2000A.all.csv")
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Download latest standard EOP data
///     bh.download_standard_eop_file("./eop_data/standard_eop.txt")
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create static EOP provider with default values
///     eop = bh.StaticEOPProvider()
///
///     # Create static EOP provider with zero values
///     eop_zero = bh.StaticEOPProvider.from_zero()
///
///     # Create with custom values
///     eop_custom = bh.StaticEOPProvider.from_values(0.1, 0.0, 0.0, 0.0, 0.0, 0.0)
///
///     # Set as global provider
///     bh.set_global_eop_provider_from_static_provider(eop_custom)
///     ```
#[pyclass(module = "brahe._brahe")]
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Create static EOP provider with defaults
    ///     eop = bh.StaticEOPProvider()
    ///     bh.set_global_eop_provider_from_static_provider(eop)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Create EOP provider with all zeros (no corrections)
    ///     eop = bh.StaticEOPProvider.from_zero()
    ///     bh.set_global_eop_provider_from_static_provider(eop)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Create EOP provider with custom values
    ///     eop = bh.StaticEOPProvider.from_values(
    ///         ut1_utc=0.1,
    ///         pm_x=1e-6,
    ///         pm_y=2e-6,
    ///         dx=1e-7,
    ///         dy=1e-7,
    ///         lod=0.001
    ///     )
    ///     bh.set_global_eop_provider_from_static_provider(eop)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.StaticEOPProvider.from_zero()
    ///     print(f"Is initialized: {eop.is_initialized()}")
    ///     ```
    pub fn is_initialized(&self) -> bool {
        self.obj.is_initialized()
    }

    /// Get the number of EOP data points.
    ///
    /// Returns:
    ///     int: Number of EOP data points
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.StaticEOPProvider.from_zero()
    ///     print(f"EOP data points: {eop.len()}")
    ///     ```
    pub fn len(&self) -> usize {
        self.obj.len()
    }

    /// Get the EOP data type.
    ///
    /// Returns:
    ///     str: EOP type string
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.StaticEOPProvider.from_zero()
    ///     print(f"EOP type: {eop.eop_type()}")
    ///     ```
    pub fn eop_type(&self) -> String {
        eop_type_to_string(self.obj.eop_type())
    }

    /// Get the extrapolation method.
    ///
    /// Returns:
    ///     str: Extrapolation method string
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.StaticEOPProvider.from_zero()
    ///     print(f"Extrapolation method: {eop.extrapolation()}")
    ///     ```
    pub fn extrapolation(&self) -> String {
        eop_extrapolation_to_string(self.obj.extrapolation())
    }

    /// Check if interpolation is enabled.
    ///
    /// Returns:
    ///     bool: True if interpolation is enabled
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.StaticEOPProvider.from_zero()
    ///     print(f"Interpolation enabled: {eop.interpolation()}")
    ///     ```
    pub fn interpolation(&self) -> bool {
        self.obj.interpolation()
    }

    /// Get the minimum Modified Julian Date in the dataset.
    ///
    /// Returns:
    ///     float: Minimum MJD
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.StaticEOPProvider.from_zero()
    ///     print(f"Minimum MJD: {eop.mjd_min()}")
    ///     ```
    pub fn mjd_min(&self) -> f64 {
        self.obj.mjd_min()
    }

    /// Get the maximum Modified Julian Date in the dataset.
    ///
    /// Returns:
    ///     float: Maximum MJD
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.StaticEOPProvider.from_zero()
    ///     print(f"Maximum MJD: {eop.mjd_max()}")
    ///     ```
    pub fn mjd_max(&self) -> f64 {
        self.obj.mjd_max()
    }

    /// Get the last Modified Julian Date with LOD data.
    ///
    /// Returns:
    ///     float: Last MJD with LOD data
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.StaticEOPProvider.from_zero()
    ///     print(f"Last MJD with LOD: {eop.mjd_last_lod()}")
    ///     ```
    pub fn mjd_last_lod(&self) -> f64 {
        self.obj.mjd_last_lod()
    }

    /// Get the last Modified Julian Date with dx/dy data.
    ///
    /// Returns:
    ///     float: Last MJD with dx/dy data
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.StaticEOPProvider.from_zero()
    ///     print(f"Last MJD with dx/dy: {eop.mjd_last_dxdy()}")
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.StaticEOPProvider.from_zero()
    ///     ut1_utc = eop.get_ut1_utc(58849.0)
    ///     print(f"UT1-UTC: {ut1_utc} seconds")
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.StaticEOPProvider.from_zero()
    ///     pm_x, pm_y = eop.get_pm(58849.0)
    ///     print(f"Polar motion: x={pm_x} rad, y={pm_y} rad")
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.StaticEOPProvider.from_zero()
    ///     dx, dy = eop.get_dxdy(58849.0)
    ///     print(f"Celestial pole offsets: dx={dx} rad, dy={dy} rad")
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.StaticEOPProvider.from_zero()
    ///     lod = eop.get_lod(58849.0)
    ///     print(f"Length of day offset: {lod} seconds")
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.StaticEOPProvider()
    ///     ut1_utc, pm_x, pm_y, dx, dy, lod = eop.get_eop(58849.0)
    ///     print(f"EOP: UT1-UTC={ut1_utc}s, PM=({pm_x},{pm_y})rad")
    ///     ```
    pub fn get_eop(&self, mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), BraheError> {
        self.obj.get_eop(mjd)
    }

}

/// File-based Earth Orientation Parameter provider.
///
/// Loads EOP data from files in standard IERS formats and provides
/// interpolation and extrapolation capabilities.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create from C04 file with interpolation
///     eop = bh.FileEOPProvider.from_c04_file(
///         "./eop_data/finals2000A.all.csv",
///         interpolate=True,
///         extrapolate="Hold"
///     )
///
///     # Create from standard file
///     eop = bh.FileEOPProvider.from_standard_file(
///         "./eop_data/finals.all",
///         interpolate=True,
///         extrapolate="Zero"
///     )
///
///     # Use default file location
///     eop = bh.FileEOPProvider.from_default_c04(True, "Hold")
///
///     # Set as global provider
///     bh.set_global_eop_provider_from_file_provider(eop)
///
///     # Get EOP data for a specific MJD
///     mjd = 60310.0
///     ut1_utc, pm_x, pm_y, dx, dy, lod = eop.get_eop(mjd)
///     ```
#[pyclass(module = "brahe._brahe")]
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_c04_file("./eop_data/finals2000A.all.csv", True, "Hold")
    ///     bh.set_global_eop_provider_from_file_provider(eop)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_standard_file("./eop_data/standard_eop.txt", True, "Hold")
    ///     bh.set_global_eop_provider_from_file_provider(eop)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_file("./eop_data/eop.txt", True, "Hold")
    ///     bh.set_global_eop_provider_from_file_provider(eop)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_default_c04(True, "Hold")
    ///     bh.set_global_eop_provider_from_file_provider(eop)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
    ///     bh.set_global_eop_provider_from_file_provider(eop)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_default_file("C04", True, "Hold")
    ///     bh.set_global_eop_provider_from_file_provider(eop)
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
    ///     print(f"is_initialized: {eop.is_initialized()}")
    ///     ```
    pub fn is_initialized(&self) -> bool {
        self.obj.is_initialized()
    }

    /// Get the number of EOP data points.
    ///
    /// Returns:
    ///     int: Number of EOP data points
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
    ///     print(f"EOP data points: {eop.len()}")
    ///     ```
    pub fn len(&self) -> usize {
        self.obj.len()
    }

    /// Get the EOP data type.
    ///
    /// Returns:
    ///     str: EOP type string
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
    ///     print(f"EOP type: {eop.eop_type()}")
    ///     ```
    pub fn eop_type(&self) -> String {
        eop_type_to_string(self.obj.eop_type())
    }

    /// Get the extrapolation method.
    ///
    /// Returns:
    ///     str: Extrapolation method string
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
    ///     print(f"Extrapolation: {eop.extrapolation()}")
    ///     ```
    pub fn extrapolation(&self) -> String {
        eop_extrapolation_to_string(self.obj.extrapolation())
    }

    /// Check if interpolation is enabled.
    ///
    /// Returns:
    ///     bool: True if interpolation is enabled
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
    ///     print(f"interpolation: {eop.interpolation()}")
    ///     ```
    pub fn interpolation(&self) -> bool {
        self.obj.interpolation()
    }

    /// Get the minimum Modified Julian Date in the dataset.
    ///
    /// Returns:
    ///     float: Minimum MJD
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
    ///     print(f"Minimum MJD: {eop.mjd_min()}")
    ///     ```
    pub fn mjd_min(&self) -> f64 {
        self.obj.mjd_min()
    }

    /// Get the maximum Modified Julian Date in the dataset.
    ///
    /// Returns:
    ///     float: Maximum MJD
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
    ///     print(f"mjd_max: {eop.mjd_max()}")
    ///     ```
    pub fn mjd_max(&self) -> f64 {
        self.obj.mjd_max()
    }

    /// Get the last Modified Julian Date with LOD data.
    ///
    /// Returns:
    ///     float: Last MJD with LOD data
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
    ///     print(f"Last MJD with LOD: {eop.mjd_last_lod()}")
    ///     ```
    pub fn mjd_last_lod(&self) -> f64 {
        self.obj.mjd_last_lod()
    }

    /// Get the last Modified Julian Date with dx/dy data.
    ///
    /// Returns:
    ///     float: Last MJD with dx/dy data
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
    ///     print(f"mjd_last_dxdy: {eop.mjd_last_dxdy()}")
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
    ///     ut1_utc = eop.get_ut1_utc(58849.0)
    ///     print(f"UT1-UTC: {ut1_utc} seconds")
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
    ///     pm_x, pm_y = eop.get_pm(58849.0)
    ///     print(f"Polar motion: x={pm_x} rad, y={pm_y} rad")
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
    ///     dx, dy = eop.get_dxdy(58849.0)
    ///     print(f"Celestial pole offsets: dx={dx} rad, dy={dy} rad")
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
    ///     lod = eop.get_lod(58849.0)
    ///     print(f"Length of day offset: {lod} seconds")
    ///     ```
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
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
    ///     ut1_utc, pm_x, pm_y, dx, dy, lod = eop.get_eop(58849.0)
    ///     print(f"EOP: UT1-UTC={ut1_utc}s, PM=({pm_x},{pm_y})rad")
    ///     ```
    pub fn get_eop(&self, mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), BraheError> {
        self.obj.get_eop(mjd)
    }

}

/// Caching EOP provider that automatically downloads updated files when stale.
///
/// This provider wraps a FileEOPProvider and adds automatic cache management.
/// It checks the age of the EOP file and downloads updated versions when the file
/// exceeds the maximum age threshold. If the file doesn't exist, it will be
/// downloaded on initialization.
///
/// Args:
///     eop_type (str): Type of EOP file - "C04" for IERS C04 format or
///         "StandardBulletinA" for IERS finals2000A.all format
///     max_age_seconds (int): Maximum age of file in seconds before triggering
///         a refresh. Common values: 86400 (1 day), 604800 (7 days)
///     auto_refresh (bool): If True, automatically checks file age and refreshes
///         on every data access. If False, only checks on initialization and
///         manual refresh() calls
///     interpolate (bool): Enable linear interpolation between tabulated EOP
///         values. Recommended: True for smoother data
///     extrapolate (str): Behavior for dates outside EOP data range:
///         "Hold" (use last known value), "Zero" (return 0.0), or "Error" (raise exception)
///     filepath (str, optional): Path to the EOP file (will be created if it doesn't exist).
///         If None, uses default cache location:
///         - StandardBulletinA: ~/.cache/brahe/finals.all.iau2000.txt
///         - C04: ~/.cache/brahe/EOP_20_C04_one_file_1962-now.txt
///
/// Raises:
///     RuntimeError: If file download fails or file is invalid
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Using default cache location (recommended)
///     provider = bh.CachingEOPProvider(
///         eop_type="StandardBulletinA",
///         max_age_seconds=7 * 86400,  # 7 days
///         auto_refresh=False,
///         interpolate=True,
///         extrapolate="Hold"
///     )
///     bh.set_global_eop_provider_from_caching_provider(provider)
///
///     # Check and refresh as needed
///     provider.refresh()
///
///     # With explicit filepath
///     provider = bh.CachingEOPProvider(
///         eop_type="StandardBulletinA",
///         max_age_seconds=7 * 86400,
///         auto_refresh=False,
///         interpolate=True,
///         extrapolate="Hold",
///         filepath="./eop_data/finals.all.iau2000.txt"
///     )
///
///     # Auto-refresh mode (convenience)
///     auto_provider = bh.CachingEOPProvider(
///         eop_type="StandardBulletinA",
///         max_age_seconds=24 * 3600,  # 24 hours
///         auto_refresh=True,  # Checks on every access
///         interpolate=True,
///         extrapolate="Hold"
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "CachingEOPProvider")]
pub(crate) struct PyCachingEOPProvider {
    obj: eop::CachingEOPProvider,
}

#[pymethods]
impl PyCachingEOPProvider {
    pub fn __repr__(&self) -> String {
        format!("CachingEOPProvider(type={}, len={}, auto_refresh={})",
                eop_type_to_string(self.obj.eop_type()),
                self.obj.len(),
                self.obj.auto_refresh)
    }

    pub fn __str__(&self) -> String {
        format!("CachingEOPProvider with {} data points (auto_refresh: {})",
                self.obj.len(),
                self.obj.auto_refresh)
    }

    /// Initialize a new caching EOP provider with automatic file management.
    ///
    /// Creates an EOP provider that automatically monitors file age and downloads
    /// updated data from IERS when the file exceeds the specified maximum age.
    /// If the file doesn't exist, it will be downloaded on initialization.
    ///
    /// Args:
    ///     eop_type (str): Type of EOP file - "C04" for IERS C04 format or
    ///         "StandardBulletinA" for IERS finals2000A.all format
    ///     max_age_seconds (int): Maximum age of file in seconds before triggering
    ///         a refresh. Common values: 86400 (1 day), 604800 (7 days)
    ///     auto_refresh (bool): If True, automatically checks file age and refreshes
    ///         on every data access. If False, only checks on initialization and
    ///         manual refresh() calls
    ///     interpolate (bool): Enable linear interpolation between tabulated EOP
    ///         values. Recommended: True for smoother data
    ///     extrapolate (str): Behavior for dates outside EOP data range:
    ///         - "Hold": Use last known value (recommended)
    ///         - "Zero": Return 0.0 for all parameters
    ///         - "Error": Raise an exception
    ///     filepath (str, optional): Path to the EOP file (will be created if it doesn't exist).
    ///         If None, uses default cache location:
    ///         - StandardBulletinA: ~/.cache/brahe/finals.all.iau2000.txt
    ///         - C04: ~/.cache/brahe/EOP_20_C04_one_file_1962-now.txt
    ///
    /// Returns:
    ///     CachingEOPProvider: Provider with automatic cache management
    ///
    /// Raises:
    ///     Exception: If file download fails or file format is invalid
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # Using default cache location (recommended)
    ///     provider = bh.CachingEOPProvider(
    ///         eop_type="StandardBulletinA",
    ///         max_age_seconds=7 * 86400,  # 7 days in seconds
    ///         auto_refresh=False,          # Manual refresh only
    ///         interpolate=True,            # Smooth interpolation
    ///         extrapolate="Hold"           # Hold last value
    ///     )
    ///
    ///     # Set as global provider for frame transformations
    ///     bh.set_global_eop_provider_from_caching_provider(provider)
    ///
    ///     # Check file status
    ///     print(f"File loaded at: {provider.file_epoch()}")
    ///     print(f"File age: {provider.file_age() / 86400:.1f} days")
    ///     ```
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     # With explicit filepath
    ///     provider = bh.CachingEOPProvider(
    ///         eop_type="StandardBulletinA",
    ///         max_age_seconds=7 * 86400,
    ///         auto_refresh=False,
    ///         interpolate=True,
    ///         extrapolate="Hold",
    ///         filepath="./eop_data/finals.all.iau2000.txt"
    ///     )
    ///
    ///     # Long-running service with auto-refresh
    ///     provider = bh.CachingEOPProvider(
    ///         eop_type="StandardBulletinA",
    ///         max_age_seconds=24 * 3600,  # 24 hours
    ///         auto_refresh=True,           # Check on every access
    ///         interpolate=True,
    ///         extrapolate="Hold"
    ///     )
    ///     bh.set_global_eop_provider_from_caching_provider(provider)
    ///
    ///     # Service runs continuously with always-current EOP data
    ///     while True:
    ///         # EOP data automatically refreshed if stale
    ///         perform_calculations()
    ///     ```
    #[new]
    #[pyo3(signature = (eop_type, max_age_seconds, auto_refresh, interpolate, extrapolate, filepath=None))]
    pub fn new(
        eop_type: &str,
        max_age_seconds: u64,
        auto_refresh: bool,
        interpolate: bool,
        extrapolate: &str,
        filepath: Option<&str>,
    ) -> Result<Self, BraheError> {
        Ok(PyCachingEOPProvider {
            obj: eop::CachingEOPProvider::new(
                filepath.map(Path::new),
                string_to_eop_type(eop_type)?,
                max_age_seconds,
                auto_refresh,
                interpolate,
                string_to_eop_extrapolation(extrapolate)?,
            )?,
        })
    }

    /// Manually refresh the cached EOP data.
    ///
    /// Checks if the file needs updating and downloads a new version if necessary.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingEOPProvider(
    ///         eop_type="StandardBulletinA",
    ///         max_age_seconds=7 * 86400,
    ///         auto_refresh=False,
    ///         interpolate=True,
    ///         extrapolate="Hold"
    ///     )
    ///
    ///     # Later, manually force a refresh check
    ///     provider.refresh()
    ///     ```
    pub fn refresh(&self) -> Result<(), BraheError> {
        self.obj.refresh()
    }

    /// Get the epoch when the EOP file was last loaded.
    ///
    /// Returns:
    ///     Epoch: Epoch in UTC when file was loaded
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingEOPProvider(
    ///         eop_type="StandardBulletinA",
    ///         max_age_seconds=7 * 86400,
    ///         auto_refresh=False,
    ///         interpolate=True,
    ///         extrapolate="Hold"
    ///     )
    ///
    ///     file_epoch = provider.file_epoch()
    ///     print(f"EOP file loaded at: {file_epoch}")
    ///     ```
    pub fn file_epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.obj.file_epoch(),
        }
    }

    /// Get the age of the currently loaded EOP file in seconds.
    ///
    /// Returns:
    ///     float: Age of the loaded file in seconds
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     provider = bh.CachingEOPProvider(
    ///         eop_type="StandardBulletinA",
    ///         max_age_seconds=7 * 86400,
    ///         auto_refresh=False,
    ///         interpolate=True,
    ///         extrapolate="Hold"
    ///     )
    ///
    ///     age = provider.file_age()
    ///     print(f"EOP file age: {age:.2f} seconds")
    ///     ```
    pub fn file_age(&self) -> f64 {
        self.obj.file_age()
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

    /// Get the EOP file type.
    ///
    /// Returns:
    ///     str: EOP type ("C04", "StandardBulletinA", etc.)
    pub fn eop_type(&self) -> String {
        eop_type_to_string(self.obj.eop_type())
    }

    /// Get the extrapolation method.
    ///
    /// Returns:
    ///     str: Extrapolation method ("Hold", "Zero", or "Error")
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

    /// Get the minimum MJD in the dataset.
    ///
    /// Returns:
    ///     float: Minimum Modified Julian Date
    pub fn mjd_min(&self) -> f64 {
        self.obj.mjd_min()
    }

    /// Get the maximum MJD in the dataset.
    ///
    /// Returns:
    ///     float: Maximum Modified Julian Date
    pub fn mjd_max(&self) -> f64 {
        self.obj.mjd_max()
    }

    /// Get the last MJD with valid LOD data.
    ///
    /// Returns:
    ///     float: Last MJD with length of day data
    pub fn mjd_last_lod(&self) -> f64 {
        self.obj.mjd_last_lod()
    }

    /// Get the last MJD with valid celestial pole offset data.
    ///
    /// Returns:
    ///     float: Last MJD with dX/dY data
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
    ///     tuple: (pm_x, pm_y, ut1_utc, dx, dy, lod)
    pub fn get_eop(&self, mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), BraheError> {
        self.obj.get_eop(mjd)
    }
}

/// Set the global EOP provider using a caching provider.
///
/// Args:
///     provider (CachingEOPProvider): Caching EOP provider to set globally
///
/// Example:
///     ```python
///     import brahe as bh
///
///     provider = bh.CachingEOPProvider(
///         eop_type="StandardBulletinA",
///         max_age_seconds=7 * 86400,
///         auto_refresh=False,
///         interpolate=True,
///         extrapolate="Hold"
///     )
///     bh.set_global_eop_provider_from_caching_provider(provider)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(provider)")]
#[pyo3(name = "set_global_eop_provider_from_caching_provider")]
pub fn py_set_global_eop_provider_from_caching_provider(provider: &PyCachingEOPProvider) {
    eop::set_global_eop_provider(provider.obj.clone());
}

/// Set the global EOP provider using a static provider.
///
/// Args:
///     provider (StaticEOPProvider): Static EOP provider to set globally
///
/// Example:
///     ```python
///     import brahe as bh
///
///     provider = bh.StaticEOPProvider.from_zero()
///     bh.set_global_eop_provider_from_static_provider(provider)
///     ```
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
///
/// Example:
///     ```python
///     import brahe as bh
///
///     provider = bh.FileEOPProvider.from_default_standard(True, "Hold")
///     bh.set_global_eop_provider_from_file_provider(provider)
///     ```
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

/// Set the global EOP provider using any supported provider type.
///
/// This function accepts any of the three EOP provider types: StaticEOPProvider,
/// FileEOPProvider, or CachingEOPProvider. This is the recommended way to set
/// the global EOP provider.
///
/// Args:
///     provider (StaticEOPProvider | FileEOPProvider | CachingEOPProvider): EOP provider to set globally
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Use with StaticEOPProvider
///     provider = bh.StaticEOPProvider.from_zero()
///     bh.set_global_eop_provider(provider)
///
///     # Use with FileEOPProvider
///     provider = bh.FileEOPProvider.from_default_standard(True, "Hold")
///     bh.set_global_eop_provider(provider)
///
///     # Use with CachingEOPProvider
///     provider = bh.CachingEOPProvider(
///         eop_type="StandardBulletinA",
///         max_age_seconds=7 * 86400,
///         auto_refresh=False,
///         interpolate=True,
///         extrapolate="Hold"
///     )
///     bh.set_global_eop_provider(provider)
///     ```
#[pyfunction]
#[pyo3(text_signature = "(provider)")]
#[pyo3(name = "set_global_eop_provider")]
pub fn py_set_global_eop_provider(provider: &Bound<'_, PyAny>) -> PyResult<()> {
    // Try to extract each type of provider
    if let Ok(static_provider) = provider.extract::<PyRef<PyStaticEOPProvider>>() {
        eop::set_global_eop_provider(static_provider.obj);
        Ok(())
    } else if let Ok(file_provider) = provider.extract::<PyRef<PyFileEOPProvider>>() {
        eop::set_global_eop_provider(file_provider.obj.clone());
        Ok(())
    } else if let Ok(caching_provider) = provider.extract::<PyRef<PyCachingEOPProvider>>() {
        eop::set_global_eop_provider(caching_provider.obj.clone());
        Ok(())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Provider must be StaticEOPProvider, FileEOPProvider, or CachingEOPProvider"
        ))
    }
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